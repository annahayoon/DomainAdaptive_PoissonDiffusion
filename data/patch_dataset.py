"""
Patch-based dataset integration for large image processing.

This module integrates the patch processing system with the existing
data loading and calibration infrastructure to enable efficient
processing of large scientific images.

Key features:
- Integration with format loaders (RAW, TIFF, FITS)
- Calibration-aware patch processing
- Memory-efficient dataset iteration
- Support for training and inference modes
- Seamless integration with existing transforms

Requirements addressed: 5.6, 5.7, 7.3, 7.6 from requirements.md
Task: 4.3 integration from tasks.md
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from core.calibration import CalibrationParams, SensorCalibration
from core.error_handlers import ErrorHandler, safe_operation
from core.logging_config import get_logger
from core.patch_processing import (
    MemoryEfficientPatchProcessor,
    PatchExtractor,
    PatchReconstructor,
    calculate_optimal_patch_size,
    create_patch_processor,
)
from core.transforms import ImageMetadata, ReversibleTransform
from data.loaders import FormatDetector
from data.loaders import ImageMetadata as LoaderMetadata

logger = get_logger(__name__)


@dataclass
class PatchDatasetConfig:
    """Configuration for patch-based dataset."""

    # Patch parameters
    patch_size: Union[int, Tuple[int, int]] = 512
    overlap: Union[int, Tuple[int, int]] = 64
    min_patch_size: Union[int, Tuple[int, int]] = 128

    # Memory management
    max_patches_in_memory: int = 16
    available_memory_gb: float = 4.0
    adaptive_patch_size: bool = True

    # Processing parameters
    device: str = "auto"
    num_workers: int = 4
    prefetch_factor: int = 2

    # Training parameters
    random_patches: bool = True  # For training: random patches vs systematic
    patches_per_image: int = 4  # Number of random patches per image (training)
    augment_patches: bool = True

    # Calibration parameters
    apply_calibration: bool = True
    normalize_to_electrons: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "patch_size": self.patch_size,
            "overlap": self.overlap,
            "min_patch_size": self.min_patch_size,
            "max_patches_in_memory": self.max_patches_in_memory,
            "available_memory_gb": self.available_memory_gb,
            "adaptive_patch_size": self.adaptive_patch_size,
            "device": self.device,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "random_patches": self.random_patches,
            "patches_per_image": self.patches_per_image,
            "augment_patches": self.augment_patches,
            "apply_calibration": self.apply_calibration,
            "normalize_to_electrons": self.normalize_to_electrons,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatchDatasetConfig":
        """Create from dictionary."""
        return cls(**data)


class PatchDataset(Dataset):
    """
    Dataset for patch-based processing of large images.

    This dataset integrates patch processing with the existing data loading
    and calibration infrastructure to enable efficient training and inference
    on large scientific images.
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        config: PatchDatasetConfig,
        calibration: Optional[SensorCalibration] = None,
        transform: Optional[ReversibleTransform] = None,
        mode: str = "train",
        domain: str = "microscopy",
    ):
        """
        Initialize patch dataset.

        Args:
            image_paths: List of paths to image files
            config: Dataset configuration
            calibration: Sensor calibration (optional)
            transform: Reversible transform (optional)
            mode: Dataset mode ('train', 'val', 'test', 'inference')
            domain: Domain type ('photography', 'microscopy', 'astronomy')
        """
        self.image_paths = [Path(p) for p in image_paths]
        self.config = config
        self.calibration = calibration
        self.transform = transform
        self.mode = mode
        self.domain = domain

        # Initialize components
        self.format_detector = FormatDetector()
        self.error_handler = ErrorHandler()

        # Setup device
        self.device = self._setup_device(config.device)

        # Initialize patch processing components
        self._initialize_patch_processors()

        # Pre-calculate patch information for systematic sampling
        if not config.random_patches:
            self._precalculate_patch_info()

        logger.info(
            f"Initialized PatchDataset: {len(self.image_paths)} images, "
            f"mode={mode}, domain={domain}, device={self.device}"
        )

    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = "cpu"

        return device

    def _initialize_patch_processors(self):
        """Initialize patch processing components."""
        # Create patch extractor
        self.patch_extractor = PatchExtractor(
            patch_size=self.config.patch_size,
            overlap=self.config.overlap,
            min_patch_size=self.config.min_patch_size,
            device=self.device,
        )

        # Create patch reconstructor (for inference)
        self.patch_reconstructor = PatchReconstructor(device=self.device)

        # Create memory-efficient processor (for inference)
        self.patch_processor = MemoryEfficientPatchProcessor(
            patch_size=self.config.patch_size,
            overlap=self.config.overlap,
            max_patches_in_memory=self.config.max_patches_in_memory,
            device=self.device,
        )

    def _precalculate_patch_info(self):
        """Pre-calculate patch information for systematic sampling."""
        logger.info("Pre-calculating patch information for systematic sampling")

        self.patch_info_cache = {}

        for i, image_path in enumerate(self.image_paths):
            try:
                # Load image metadata only
                _, metadata = self.format_detector.load_auto(image_path)

                # Calculate patch grid
                (
                    patch_infos,
                    num_rows,
                    num_cols,
                ) = self.patch_extractor.calculate_patch_grid(
                    metadata.height, metadata.width
                )

                self.patch_info_cache[i] = {
                    "patch_infos": patch_infos,
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "image_height": metadata.height,
                    "image_width": metadata.width,
                }

            except Exception as e:
                logger.warning(f"Failed to pre-calculate patches for {image_path}: {e}")
                self.patch_info_cache[i] = None

        # Calculate total number of patches
        self.total_patches = sum(
            len(info["patch_infos"])
            for info in self.patch_info_cache.values()
            if info is not None
        )

        logger.info(
            f"Pre-calculated {self.total_patches} patches from {len(self.image_paths)} images"
        )

    def __len__(self) -> int:
        """Get dataset length."""
        if self.config.random_patches:
            # For random patches, length is number of images * patches per image
            return len(self.image_paths) * self.config.patches_per_image
        else:
            # For systematic patches, length is total number of patches
            return getattr(self, "total_patches", len(self.image_paths))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        if self.config.random_patches:
            return self._get_random_patch(idx)
        else:
            return self._get_systematic_patch(idx)

    def _get_random_patch(self, idx: int) -> Dict[str, Any]:
        """Get random patch from dataset."""
        # Determine which image to use
        image_idx = idx // self.config.patches_per_image
        patch_idx = idx % self.config.patches_per_image

        image_path = self.image_paths[image_idx]

        try:
            # Load full image
            raw_data, loader_metadata = self.format_detector.load_auto(image_path)

            # Apply calibration if available
            if self.config.apply_calibration and self.calibration is not None:
                processed_data, mask = self.calibration.process_raw(
                    raw_data, return_mask=True
                )
                if self.config.normalize_to_electrons:
                    # Convert to electrons
                    electrons = processed_data  # Already in electrons from calibration
                else:
                    electrons = processed_data
            else:
                electrons = raw_data.astype(np.float32)
                mask = np.ones_like(electrons, dtype=bool)

            # Convert to tensor
            electrons_tensor = torch.from_numpy(electrons).float()
            mask_tensor = torch.from_numpy(mask.astype(np.float32))

            # Add dimensions if needed
            if electrons_tensor.ndim == 2:
                electrons_tensor = electrons_tensor.unsqueeze(0).unsqueeze(0)
            elif electrons_tensor.ndim == 3:
                electrons_tensor = electrons_tensor.unsqueeze(0)

            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            elif mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.unsqueeze(0)

            # Extract random patch
            patch_electrons = self._extract_random_patch(electrons_tensor)
            patch_mask = self._extract_random_patch(mask_tensor)

            # Apply transforms if available
            if self.transform is not None:
                # Create metadata for transform
                transform_metadata = ImageMetadata(
                    original_height=loader_metadata.height,
                    original_width=loader_metadata.width,
                    scale_factor=1.0,
                    crop_info=None,
                    pad_info=None,
                    pixel_size=getattr(loader_metadata, "pixel_size", 1.0),
                    pixel_unit=getattr(loader_metadata, "pixel_unit", "um"),
                    domain=self.domain,
                )

                # Normalize for transform
                if self.config.normalize_to_electrons and hasattr(
                    self.calibration, "params"
                ):
                    scale = self.calibration.params.get("scale", 1000.0)
                    normalized = patch_electrons / scale
                else:
                    normalized = patch_electrons / patch_electrons.max()

                normalized = torch.clamp(normalized, 0, 1)

                # Apply transform
                transformed, _ = self.transform.forward(
                    normalized.squeeze(0),  # Remove batch dim for transform
                    **transform_metadata.__dict__,
                )

                patch_electrons = transformed.unsqueeze(0)  # Add batch dim back

            # Create result
            result = {
                "electrons": patch_electrons.squeeze(0),  # Remove batch dim
                "mask": patch_mask.squeeze(0),
                "image_path": str(image_path),
                "image_idx": image_idx,
                "patch_idx": patch_idx,
                "domain": self.domain,
                "mode": self.mode,
            }

            # Add calibration info if available
            if self.calibration is not None:
                result["calibration_params"] = self.calibration.get_parameters()

            # Add loader metadata
            result["loader_metadata"] = loader_metadata.__dict__

            return result

        except Exception as e:
            logger.error(f"Failed to load patch from {image_path}: {e}")
            # Return dummy data to avoid breaking training
            return self._create_dummy_patch(image_idx, patch_idx)

    def _get_systematic_patch(self, idx: int) -> Dict[str, Any]:
        """Get systematic patch from pre-calculated grid."""
        # Find which image and patch this index corresponds to
        current_idx = 0

        for image_idx, patch_info in self.patch_info_cache.items():
            if patch_info is None:
                continue

            num_patches = len(patch_info["patch_infos"])

            if current_idx <= idx < current_idx + num_patches:
                # This is the right image
                patch_idx_in_image = idx - current_idx
                patch_info_obj = patch_info["patch_infos"][patch_idx_in_image]

                return self._load_specific_patch(image_idx, patch_info_obj)

            current_idx += num_patches

        # Fallback if index not found
        logger.warning(f"Patch index {idx} not found, using fallback")
        return self._create_dummy_patch(0, 0)

    def _extract_random_patch(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract random patch from image tensor."""
        B, C, H, W = image_tensor.shape

        # Determine patch size
        if isinstance(self.config.patch_size, int):
            patch_h = patch_w = self.config.patch_size
        else:
            patch_h, patch_w = self.config.patch_size

        # Ensure patch fits in image
        patch_h = min(patch_h, H)
        patch_w = min(patch_w, W)

        # Random position
        max_y = max(0, H - patch_h)
        max_x = max(0, W - patch_w)

        start_y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        start_x = np.random.randint(0, max_x + 1) if max_x > 0 else 0

        # Extract patch
        patch = image_tensor[
            :, :, start_y : start_y + patch_h, start_x : start_x + patch_w
        ]

        # Pad if necessary
        if patch.shape[2] < patch_h or patch.shape[3] < patch_w:
            pad_h = patch_h - patch.shape[2]
            pad_w = patch_w - patch.shape[3]
            patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")

        return patch

    def _load_specific_patch(self, image_idx: int, patch_info) -> Dict[str, Any]:
        """Load specific patch from image."""
        image_path = self.image_paths[image_idx]

        try:
            # Load full image
            raw_data, loader_metadata = self.format_detector.load_auto(image_path)

            # Apply calibration if available
            if self.config.apply_calibration and self.calibration is not None:
                processed_data, mask = self.calibration.process_raw(
                    raw_data, return_mask=True
                )
                electrons = processed_data
            else:
                electrons = raw_data.astype(np.float32)
                mask = np.ones_like(electrons, dtype=bool)

            # Convert to tensor
            electrons_tensor = torch.from_numpy(electrons).float()
            mask_tensor = torch.from_numpy(mask.astype(np.float32))

            # Add dimensions if needed
            if electrons_tensor.ndim == 2:
                electrons_tensor = electrons_tensor.unsqueeze(0).unsqueeze(0)
            elif electrons_tensor.ndim == 3:
                electrons_tensor = electrons_tensor.unsqueeze(0)

            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            elif mask_tensor.ndim == 3:
                mask_tensor = mask_tensor.unsqueeze(0)

            # Extract specific patch
            patch_electrons = self.patch_extractor.extract_patch_at_position(
                electrons_tensor,
                patch_info.start_y,
                patch_info.start_x,
                patch_info.height,
                patch_info.width,
            )

            patch_mask = self.patch_extractor.extract_patch_at_position(
                mask_tensor,
                patch_info.start_y,
                patch_info.start_x,
                patch_info.height,
                patch_info.width,
            )

            # Apply transforms if available
            if self.transform is not None:
                # Similar transform logic as in random patches
                transform_metadata = ImageMetadata(
                    original_height=loader_metadata.height,
                    original_width=loader_metadata.width,
                    scale_factor=1.0,
                    crop_info=None,
                    pad_info=None,
                    pixel_size=getattr(loader_metadata, "pixel_size", 1.0),
                    pixel_unit=getattr(loader_metadata, "pixel_unit", "um"),
                    domain=self.domain,
                )

                if self.config.normalize_to_electrons and hasattr(
                    self.calibration, "params"
                ):
                    scale = self.calibration.params.get("scale", 1000.0)
                    normalized = patch_electrons / scale
                else:
                    normalized = patch_electrons / patch_electrons.max()

                normalized = torch.clamp(normalized, 0, 1)

                transformed, _ = self.transform.forward(
                    normalized.squeeze(0), **transform_metadata.__dict__
                )

                patch_electrons = transformed.unsqueeze(0)

            # Create result
            result = {
                "electrons": patch_electrons.squeeze(0),
                "mask": patch_mask.squeeze(0),
                "image_path": str(image_path),
                "image_idx": image_idx,
                "patch_info": patch_info.to_dict(),
                "domain": self.domain,
                "mode": self.mode,
            }

            # Add calibration info if available
            if self.calibration is not None:
                result["calibration_params"] = self.calibration.get_parameters()

            # Add loader metadata
            result["loader_metadata"] = loader_metadata.__dict__

            return result

        except Exception as e:
            logger.error(f"Failed to load specific patch from {image_path}: {e}")
            return self._create_dummy_patch(image_idx, 0)

    def _create_dummy_patch(self, image_idx: int, patch_idx: int) -> Dict[str, Any]:
        """Create dummy patch data for error cases."""
        if isinstance(self.config.patch_size, int):
            patch_h = patch_w = self.config.patch_size
        else:
            patch_h, patch_w = self.config.patch_size

        dummy_electrons = torch.zeros(1, patch_h, patch_w)
        dummy_mask = torch.ones(1, patch_h, patch_w)

        return {
            "electrons": dummy_electrons,
            "mask": dummy_mask,
            "image_path": "dummy",
            "image_idx": image_idx,
            "patch_idx": patch_idx,
            "domain": self.domain,
            "mode": self.mode,
            "is_dummy": True,
        }

    def process_full_image(
        self,
        image_path: Union[str, Path],
        processing_func: Callable[[torch.Tensor], torch.Tensor],
        output_path: Optional[Union[str, Path]] = None,
    ) -> torch.Tensor:
        """
        Process full image using patch-based approach.

        Args:
            image_path: Path to image file
            processing_func: Function to apply to each patch
            output_path: Optional path to save result

        Returns:
            Processed full image tensor
        """
        image_path = Path(image_path)

        # Load full image
        raw_data, loader_metadata = self.format_detector.load_auto(image_path)

        # Apply calibration if available
        if self.config.apply_calibration and self.calibration is not None:
            processed_data, mask = self.calibration.process_raw(
                raw_data, return_mask=True
            )
            electrons = processed_data
        else:
            electrons = raw_data.astype(np.float32)
            mask = np.ones_like(electrons, dtype=bool)

        # Convert to tensor
        electrons_tensor = torch.from_numpy(electrons).float()

        # Add dimensions if needed
        if electrons_tensor.ndim == 2:
            electrons_tensor = electrons_tensor.unsqueeze(0).unsqueeze(0)
        elif electrons_tensor.ndim == 3:
            electrons_tensor = electrons_tensor.unsqueeze(0)

        # Adaptive patch size if enabled
        if self.config.adaptive_patch_size:
            H, W = electrons_tensor.shape[-2:]
            optimal_processor = create_patch_processor(
                H,
                W,
                available_memory_gb=self.config.available_memory_gb,
                device=self.device,
            )

            result = optimal_processor.process_image(
                electrons_tensor, processing_func, show_progress=True
            )
        else:
            result = self.patch_processor.process_image(
                electrons_tensor, processing_func, show_progress=True
            )

        # Save if requested
        if output_path:
            torch.save(result, output_path)
            logger.info(f"Saved processed image to {output_path}")

        return result

    def get_image_info(self, image_idx: int) -> Dict[str, Any]:
        """Get information about specific image."""
        image_path = self.image_paths[image_idx]

        try:
            _, metadata = self.format_detector.load_auto(image_path)

            info = {
                "path": str(image_path),
                "height": metadata.height,
                "width": metadata.width,
                "channels": metadata.channels,
                "format": metadata.format,
                "domain": metadata.domain,
                "bit_depth": metadata.bit_depth,
                "dtype": metadata.dtype,
            }

            # Add patch information if pre-calculated
            if hasattr(self, "patch_info_cache") and image_idx in self.patch_info_cache:
                patch_info = self.patch_info_cache[image_idx]
                if patch_info is not None:
                    info["num_patches"] = len(patch_info["patch_infos"])
                    info["patch_grid"] = (
                        patch_info["num_rows"],
                        patch_info["num_cols"],
                    )

            return info

        except Exception as e:
            logger.error(f"Failed to get info for {image_path}: {e}")
            return {"path": str(image_path), "error": str(e)}

    def save_config(self, config_path: Union[str, Path]):
        """Save dataset configuration."""
        config_path = Path(config_path)

        config_data = {
            "dataset_config": self.config.to_dict(),
            "image_paths": [str(p) for p in self.image_paths],
            "mode": self.mode,
            "domain": self.domain,
            "calibration_params": self.calibration.get_parameters()
            if self.calibration
            else None,
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved dataset configuration to {config_path}")

    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        calibration: Optional[SensorCalibration] = None,
        transform: Optional[ReversibleTransform] = None,
    ) -> "PatchDataset":
        """Load dataset from configuration file."""
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            config_data = json.load(f)

        dataset_config = PatchDatasetConfig.from_dict(config_data["dataset_config"])

        return cls(
            image_paths=config_data["image_paths"],
            config=dataset_config,
            calibration=calibration,
            transform=transform,
            mode=config_data["mode"],
            domain=config_data["domain"],
        )


def create_patch_dataloader(
    dataset: PatchDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create DataLoader for patch dataset.

    Args:
        dataset: Patch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory

    Returns:
        Configured DataLoader
    """
    if num_workers is None:
        num_workers = dataset.config.num_workers

    # Custom collate function to handle variable-sized data
    def collate_fn(batch):
        """Custom collate function for patch data."""
        # Stack tensors
        electrons = torch.stack([item["electrons"] for item in batch])
        masks = torch.stack([item["mask"] for item in batch])

        # Collect metadata
        metadata = {
            "image_paths": [item["image_path"] for item in batch],
            "image_indices": [item["image_idx"] for item in batch],
            "domains": [item["domain"] for item in batch],
            "modes": [item["mode"] for item in batch],
        }

        # Add patch-specific info if available
        if "patch_idx" in batch[0]:
            metadata["patch_indices"] = [item["patch_idx"] for item in batch]

        if "patch_info" in batch[0]:
            metadata["patch_infos"] = [item["patch_info"] for item in batch]

        # Add calibration params if available
        if "calibration_params" in batch[0]:
            metadata["calibration_params"] = [
                item["calibration_params"] for item in batch
            ]

        return {"electrons": electrons, "mask": masks, "metadata": metadata}

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        prefetch_factor=dataset.config.prefetch_factor if num_workers > 0 else 2,
    )

    return dataloader


# Utility functions for patch dataset creation
def create_patch_dataset_from_directory(
    data_dir: Union[str, Path],
    config: PatchDatasetConfig,
    domain: str,
    calibration: Optional[SensorCalibration] = None,
    transform: Optional[ReversibleTransform] = None,
    mode: str = "train",
    file_pattern: str = "*",
) -> PatchDataset:
    """
    Create patch dataset from directory of images.

    Args:
        data_dir: Directory containing images
        config: Dataset configuration
        domain: Domain type
        calibration: Sensor calibration
        transform: Reversible transform
        mode: Dataset mode
        file_pattern: File pattern to match

    Returns:
        Configured patch dataset
    """
    data_dir = Path(data_dir)

    # Find image files
    image_paths = []

    # Common image extensions by domain
    if domain == "photography":
        extensions = ["*.arw", "*.dng", "*.nef", "*.cr2", "*.raw"]
    elif domain == "microscopy":
        extensions = ["*.tif", "*.tiff"]
    elif domain == "astronomy":
        extensions = ["*.fits", "*.fit"]
    else:
        extensions = ["*.tif", "*.tiff", "*.fits", "*.fit", "*.arw", "*.dng"]

    for ext in extensions:
        image_paths.extend(data_dir.glob(ext))
        image_paths.extend(data_dir.glob(ext.upper()))

    # Apply file pattern if specified
    if file_pattern != "*":
        filtered_paths = []
        for path in image_paths:
            if path.match(file_pattern):
                filtered_paths.append(path)
        image_paths = filtered_paths

    logger.info(f"Found {len(image_paths)} images in {data_dir}")

    if not image_paths:
        raise ValueError(f"No images found in {data_dir} with pattern {file_pattern}")

    return PatchDataset(
        image_paths=image_paths,
        config=config,
        calibration=calibration,
        transform=transform,
        mode=mode,
        domain=domain,
    )
