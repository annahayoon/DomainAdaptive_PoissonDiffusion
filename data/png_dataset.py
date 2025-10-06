"""
PNG dataset loader for 8-bit PNG images.

This module provides dataset classes for loading 8-bit PNG images
into the diffusion model training pipeline.
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from core.logging_config import get_logger

# Import EDM components for compatibility
try:
    import sys

    edm_path = Path(__file__).parent.parent / "external" / "edm"
    sys.path.insert(0, str(edm_path))
    from training.dataset import Dataset as EDMDataset
except ImportError:
    EDMDataset = None

logger = get_logger(__name__)

if TYPE_CHECKING:
    # Forward reference for type hints
    pass


class EDMPNGDataset:
    """
    EDM-compatible wrapper for PNGDataset that provides the interface expected
    by EDM's native training loop.

    This wrapper:
    - Provides required properties (resolution, num_channels, label_dim)
    - Returns data in EDM-compatible format (uint8 images + labels)
    - Wraps our existing PNGDataset functionality
    - Compatible with EDM's InfiniteSampler and DataLoader
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        metadata_json: Union[str, Path],
        split: str = "train",
        max_files: Optional[int] = None,
        seed: int = 42,
        image_size: int = 256,
        channels: int = 3,
        domain: Optional[str] = None,  # Auto-detect if None
        use_labels: bool = True,
        label_dim: int = 3,  # One-hot domain encoding for 3 domains
        **kwargs,
    ):
        """
        Initialize EDM-compatible PNG dataset wrapper.

        Args:
            data_root: Root directory containing PNG files
            metadata_json: Metadata JSON file with predefined splits
            split: Data split (train, validation, test)
            max_files: Maximum number of files to load
            seed: Random seed for reproducibility
            image_size: Target image size (will resize if needed)
            channels: Number of channels (3 for RGB)
            domain: Domain name ('photography', 'microscopy', 'astronomy'). Auto-detected if None.
            use_labels: Enable conditioning labels (for domain encoding)
            label_dim: Dimension of label space (3 for one-hot domain encoding)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        # Auto-detect domain from metadata if not provided
        if domain is None:
            domain = self._detect_domain_from_metadata(metadata_json)

        self.domain = domain

        # Domain encoding mapping
        self.domain_to_label = {
            "photography": [1, 0, 0],
            "microscopy": [0, 1, 0],
            "astronomy": [0, 0, 1],
        }

        if domain not in self.domain_to_label:
            raise ValueError(
                f"Unsupported domain: {domain}. Supported: {list(self.domain_to_label.keys())}"
            )

        # Initialize dataset parameters (same as PNGDataset but integrated)
        self.data_root = Path(data_root)
        self.metadata_json = Path(metadata_json)
        self.split = split
        self.max_files = max_files
        self.seed = seed
        self.image_size = image_size
        self.channels = channels

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")

        if not self.metadata_json.exists():
            raise FileNotFoundError(
                f"Metadata JSON file not found: {self.metadata_json}"
            )

        # Find paired PNG files and create noisy mapping (integrated logic)
        self.image_files, self.noisy_mapping = self._find_paired_files()
        self.calibration_params = getattr(self, "calibration_params", {})

        # Create train/val/test split
        self.split_files = self._create_split()

        # Setup transforms
        self.transform = self._setup_transforms()

        # Set up raw shape for EDM compatibility (N, C, H, W)
        self._raw_shape = [len(self.split_files), channels, image_size, image_size]

        # Set up labels for domain encoding
        self._use_labels = use_labels
        self._label_dim = label_dim
        self._raw_labels = None
        self._raw_idx = np.arange(len(self.split_files), dtype=np.int64)
        self._xflip = np.zeros(len(self.split_files), dtype=np.uint8)

        # Set up dataset properties for EDM compatibility
        self._name = f"png_{split}"

        # Cache for loaded images
        self._cached_images = {}
        self._cache = False  # We'll implement caching if needed

        logger.info(
            f"EDMPNGDataset ready: {len(self.split_files)} CLEAN tiles for '{split}' split (prior training) - Domain: {self.domain}"
        )

    def _detect_domain_from_metadata(self, metadata_json: Union[str, Path]) -> str:
        """Auto-detect domain from metadata JSON file."""
        try:
            with open(metadata_json, "r") as f:
                metadata = json.load(f)
            domain = metadata.get("domain")
            if domain not in ["photography", "microscopy", "astronomy"]:
                raise ValueError(f"Unsupported domain in metadata: {domain}")
            return domain
        except Exception as e:
            logger.warning(f"Could not auto-detect domain from metadata: {e}")
            # Default to photography if detection fails
            return "photography"

    @staticmethod
    def _detect_domain_from_metadata_static(metadata_json: Union[str, Path]) -> str:
        """Static version of domain detection for factory function."""
        try:
            with open(metadata_json, "r") as f:
                metadata = json.load(f)
            domain = metadata.get("domain")
            if domain not in ["photography", "microscopy", "astronomy"]:
                raise ValueError(f"Unsupported domain in metadata: {domain}")
            return domain
        except Exception as e:
            logger.warning(f"Could not auto-detect domain from metadata: {e}")
            # Default to photography if detection fails
            return "photography"

    def _find_paired_files(self) -> Tuple[List[Path], Dict[Path, Path]]:
        """Find paired clean/noisy PNG files using metadata JSON file."""
        logger.info(f"Loading tile metadata from {self.metadata_json}")

        with open(self.metadata_json, "r") as f:
            metadata = json.load(f)

        # Extract tiles for current domain
        all_tiles = metadata.get("tiles", [])
        domain_tiles = [tile for tile in all_tiles if tile.get("domain") == self.domain]

        if not domain_tiles:
            raise FileNotFoundError(
                f"No {self.domain} tiles found in metadata {self.metadata_json}"
            )

        logger.info(f"Found {len(domain_tiles)} {self.domain} tiles in metadata")

        # Group tiles by split
        tiles_by_split = {"train": [], "validation": [], "test": []}
        for tile in domain_tiles:
            split = tile.get("split", "train")
            if split in tiles_by_split:
                tiles_by_split[split].append(tile)

        logger.info(
            f"Split distribution: { {k: len(v) for k, v in tiles_by_split.items()} }"
        )

        # Get tiles for current split
        if self.split not in tiles_by_split or not tiles_by_split[self.split]:
            raise FileNotFoundError(
                f"No tiles found for split '{self.split}' in metadata. "
                f"Available splits: {list(tiles_by_split.keys())}"
            )

        split_tiles = tiles_by_split[self.split]

        # FILTER OUT NOISY TILES FOR PRIOR TRAINING
        # Only use clean tiles (long exposures), not noisy tiles (short exposures)
        # Noisy tiles typically have short exposure times (e.g., "0.04s", "0.1s")
        # Clean tiles have long exposure times (e.g., "10s", "30s")
        clean_tiles_only = []
        noisy_tiles_filtered = []

        for tile in split_tiles:
            tile_id = tile.get("tile_id", "")
            # Check if this is a noisy tile (short exposure) by looking at the tile_id pattern
            # Noisy tiles contain exposure times like "0.04s", "0.1s", "0.033s" etc.
            # Clean tiles contain longer exposures like "10s", "30s" etc.
            is_noisy = False
            if "_0." in tile_id:  # Short exposure (e.g., 0.04s, 0.1s)
                is_noisy = True
            elif tile_id.endswith("_00_0s"):  # Some variations
                is_noisy = True

            if is_noisy:
                noisy_tiles_filtered.append(tile_id)
            else:
                clean_tiles_only.append(tile)

        logger.info(
            f"Split '{self.split}' - Total tiles in metadata: {len(split_tiles)}"
        )
        logger.info(f"  └─ Clean tiles (prior training): {len(clean_tiles_only)}")
        logger.info(f"  └─ Noisy tiles (filtered out): {len(noisy_tiles_filtered)}")

        split_tiles = clean_tiles_only  # Use only clean tiles

        # Extract clean file paths and build calibration parameter mapping
        clean_files = []
        noisy_mapping = {}
        self.calibration_params = {}  # Map tile_id to calibration parameters

        for tile in split_tiles:
            tile_id = tile.get("tile_id")
            if not tile_id:
                continue

            # Store calibration parameters for this tile
            # Use electron statistics from metadata for physics-aware training
            electron_min = tile.get("electron_min", 0.0)
            electron_max = tile.get("electron_max", 1000.0)
            electron_mean = tile.get("electron_mean", 500.0)
            electron_std = tile.get("electron_std", 100.0)

            self.calibration_params[tile_id] = {
                "gain": tile.get("gain", 1.0),
                "read_noise": tile.get("read_noise", 0.0),
                "background": electron_min,  # Use electron_min as background
                "scale": electron_max,  # Use electron_max as scale for physics-aware training
                "electron_min": electron_min,
                "electron_max": electron_max,
                "electron_mean": electron_mean,
                "electron_std": electron_std,
            }

            # Use the png_path from metadata (absolute path through symlink)
            clean_path_str = tile.get("png_path")
            if not clean_path_str:
                continue

            clean_path = Path(clean_path_str)

            # Check if the absolute path from metadata exists (through symlink)
            if clean_path.exists():
                clean_files.append(clean_path)
            else:
                # Fallback: try constructing path relative to data_root
                clean_path = self.data_root / f"{tile_id}.png"
                if clean_path.exists():
                    clean_files.append(clean_path)
                else:
                    logger.debug(f"Clean file not found: {tile_id}")
                    continue

            # Find corresponding noisy file
            # Get the directory of the clean file and construct noisy path
            clean_dir = clean_path.parent.parent  # Go up to png_tiles/photography/
            noisy_path = clean_dir / "noisy" / f"{tile_id}.png"

            if noisy_path.exists():
                noisy_mapping[clean_path] = noisy_path
            else:
                # Try pattern matching with scene_id
                scene_id = tile.get("scene_id", "")
                if scene_id:
                    # Extract scene number from scene_id (e.g., "photo_00145" -> "00145")
                    scene_num = scene_id.split("_")[-1] if "_" in scene_id else scene_id
                    tile_num = tile_id.split("_")[-1] if "_" in tile_id else ""

                    # Look for noisy files with this scene and tile
                    noisy_pattern = f"{self.domain}_*_{scene_num}_*_tile_{tile_num}.png"
                    noisy_dir = clean_dir / "noisy"
                    noisy_matches = list(noisy_dir.glob(noisy_pattern))

                    if noisy_matches:
                        noisy_mapping[clean_path] = noisy_matches[0]
                    else:
                        logger.debug(f"No noisy file found for {tile_id}")

        if not clean_files:
            raise FileNotFoundError(
                f"No valid clean files found for split '{self.split}'"
            )

        logger.info(
            f"✓ Loaded {len(clean_files)} CLEAN tiles (prior training) for '{self.split}' split"
        )
        logger.info(
            f"  └─ Calibration parameters: {len(self.calibration_params)} tiles"
        )
        logger.info(f"  └─ Noisy pairs available: {len(noisy_mapping)} tiles")
        return clean_files, noisy_mapping

    def _create_split(self) -> List[Path]:
        """Apply max_files limit to image files (splits already defined in metadata)."""
        # The split is already handled by _find_paired_files_from_metadata
        split_files = self.image_files.copy()

        # Apply max_files limit if specified
        if self.max_files and len(split_files) > self.max_files:
            split_files = split_files[: self.max_files]
            logger.info(f"Limited to {self.max_files} files for {self.split} split")

        return split_files

    def _setup_transforms(self) -> transforms.Compose:
        """Setup image transforms - keeps data in [0, 255] range."""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            # NOTE: Not using ToTensor() to avoid normalization to [0, 1]
            # Data stays in [0, 255] range for explicit normalization in training script
        ]

        return transforms.Compose(transform_list)

    def _get_raw_labels(self):
        """Get raw labels for EDM compatibility with domain-aware encoding."""
        if self._raw_labels is None:
            if self._use_labels:
                # Create one-hot labels based on domain
                self._raw_labels = np.zeros(
                    (len(self.split_files), self._label_dim), dtype=np.float32
                )
                domain_label = self.domain_to_label[self.domain]

                # Set the appropriate indices for this domain
                if self.domain == "photography":
                    self._raw_labels[:, 0] = domain_label[0]  # [1, 0, 0]
                elif self.domain == "microscopy":
                    self._raw_labels[:, 1] = domain_label[1]  # [0, 1, 0]
                elif self.domain == "astronomy":
                    self._raw_labels[:, 2] = domain_label[2]  # [0, 0, 1]
            else:
                self._raw_labels = np.zeros(
                    (len(self.split_files), 0), dtype=np.float32
                )
        return self._raw_labels

    def _load_raw_image(self, raw_idx):
        """Load raw image in EDM format (uint8, CHW)."""
        # Get image using integrated logic
        png_item = self._load_png_item(raw_idx)

        # Extract clean image (already in [0, 255] range, CHW format)
        clean_image = png_item["clean"]

        # Convert to numpy and ensure uint8 format
        image_np = clean_image.numpy()

        # If we have float values in [0, 255], convert to uint8
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        # Ensure shape is (C, H, W) - EDM expects CHW format
        if image_np.ndim == 2:
            # Grayscale to (1, H, W)
            image_np = np.expand_dims(image_np, axis=0)
        elif image_np.ndim == 3 and image_np.shape[0] != 3:  # HWC format
            # Convert from HWC to CHW
            image_np = np.transpose(image_np, (2, 0, 1))

        return image_np

    def __len__(self):
        """Return dataset length."""
        return len(self.split_files)

    def _load_png_item(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a clean PNG image (integrated version of PNGDataset.__getitem__).

        Returns dict with:
            - 'clean': Clean image data (C, H, W) in [0,1] range - used for training
            - 'electrons': Clean image data in electron units - used for physics-aware loss
            - 'domain': Domain tensor
            - 'metadata': Image metadata
            - 'domain_params': Domain-specific parameters from metadata calibration
        """
        try:
            clean_path = self.split_files[idx]

            # Load only the clean image
            clean = self._load_png_image_direct(clean_path)

            # Extract metadata
            metadata = {
                "clean_path": str(clean_path),
                "filename": clean_path.name,
                "image_size": self.image_size,
                "channels": self.channels,
            }

            # Get calibration parameters for this tile from metadata
            tile_id = clean_path.name.replace(
                ".png", ""
            )  # Extract tile_id from filename
            calib_params = self.calibration_params.get(tile_id, {})

            # Use electron statistics from metadata for physics-aware training
            scale = calib_params.get("electron_max", 1000.0)
            background = calib_params.get("electron_min", 0.0)

            calib_params = {
                "scale": scale,
                "gain": calib_params.get("gain", 1.0),
                "read_noise": calib_params.get("read_noise", 0.0),
                "background": background,
            }

            # Domain parameters from metadata
            domain_params = {
                "scale": calib_params["scale"],
                "gain": calib_params["gain"],
                "read_noise": calib_params["read_noise"],
                "background": calib_params["background"],
            }

            # Convert clean image from [0,1] to electron units for physics-aware training
            # Scale from normalized [0,1] to electron scale
            electrons = clean * calib_params["scale"] + calib_params["background"]

            # Domain encoding for physics-aware training
            domain_encoding = {"photography": 0, "microscopy": 1, "astronomy": 2}
            domain_id = domain_encoding.get(self.domain, 0)

            # CRITICAL FIX: Flatten domain_params into batch for validation compatibility
            # The validation step expects scale, read_noise, background at the top level
            return {
                "clean": clean,
                "electrons": electrons,
                "domain": torch.tensor(domain_id),
                "metadata": metadata,
                "domain_params": domain_params,
                # Flatten calibration parameters for trainer compatibility
                "scale": torch.tensor([calib_params["scale"]], dtype=torch.float32),
                "read_noise": torch.tensor(
                    [calib_params["read_noise"]], dtype=torch.float32
                ),
                "background": torch.tensor(
                    [calib_params["background"]], dtype=torch.float32
                ),
                "gain": torch.tensor([calib_params["gain"]], dtype=torch.float32),
            }

        except Exception as e:
            logger.warning(f"Error loading PNG {idx}: {e}")
            # Return dummy data to prevent training crashes (in [0, 1] range)
            return {
                "clean": torch.zeros(
                    self.channels, self.image_size, self.image_size
                ),  # [0, 1] range
                "electrons": torch.zeros(
                    self.channels, self.image_size, self.image_size
                ),  # [0, inf] range for electron units
                "domain": torch.tensor(0),
                "metadata": {"corrupted": True, "original_idx": idx},
                "domain_params": {
                    "scale": 1000.0,
                    "gain": 1.0,
                    "read_noise": 0.0,
                    "background": 0.0,
                },
                # Flatten calibration parameters for trainer compatibility
                "scale": torch.tensor([1000.0], dtype=torch.float32),
                "read_noise": torch.tensor([0.0], dtype=torch.float32),
                "background": torch.tensor([0.0], dtype=torch.float32),
                "gain": torch.tensor([1.0], dtype=torch.float32),
            }

    def _load_png_image_direct(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess a PNG image - keeps data in [0, 255] range."""
        try:
            # Load image
            image = Image.open(image_path)

            # Convert RGBA to RGB if needed for consistency with model configuration
            if image.mode == "RGBA" and self.channels == 3:
                # Convert RGBA to RGB by removing alpha channel
                # Create RGB image from RGBA
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])  # Use alpha as mask
                image = rgb_image
            elif image.mode != "RGB" and self.channels == 3:
                # Convert other modes to RGB
                image = image.convert("RGB")

            # Apply transforms (resize only, no normalization)
            image = self.transform(image)

            # Convert PIL Image to tensor in [0, 255] range (not normalized)
            # PIL Image is uint8 [0, 255], we keep it as float32 [0, 255]
            tensor = torch.from_numpy(np.array(image)).float()

            # Convert from (H, W, C) to (C, H, W) format
            tensor = (
                tensor.permute(2, 0, 1) if tensor.ndim == 3 else tensor.unsqueeze(0)
            )

            return tensor

        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return dummy image with correct number of channels in [0, 255] range
            return torch.zeros(self.channels, self.image_size, self.image_size)

    def __getitem__(self, idx):
        """Get item as (image, label) tuple for EDM compatibility."""
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image

        label = self._get_raw_labels()[raw_idx]

        if self._xflip[idx]:
            assert isinstance(image, np.ndarray)
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]  # Flip horizontally

        return image, label

    @property
    def name(self):
        """Dataset name."""
        return self._name

    @property
    def image_shape(self):
        """Image shape in CHW format."""
        return list(self._raw_shape[1:])

    @property
    def resolution(self):
        """Image resolution (assumes square images)."""
        return self._raw_shape[2]

    @property
    def num_channels(self):
        """Number of image channels."""
        return self._raw_shape[1]

    @property
    def label_shape(self):
        """Label shape."""
        if self._use_labels:
            return [self._label_dim]
        else:
            return [0]

    @property
    def label_dim(self):
        """Label dimension for conditioning."""
        return self._label_dim if self._use_labels else 0

    @property
    def has_labels(self):
        """Whether dataset has labels."""
        return self._use_labels

    @property
    def has_onehot_labels(self):
        """Whether labels are one-hot encoded."""
        return (
            self._use_labels
            and self._raw_labels is not None
            and self._raw_labels.dtype == np.float32
        )

    def get_calibration_params(self, idx):
        """Get calibration parameters for a specific index."""
        return self._load_png_item(idx).get("domain_params", {})

    def close(self):
        """Clean up resources."""
        # No specific cleanup needed for the integrated functionality
        pass


def create_edm_png_datasets(
    data_root: Union[str, Path],
    metadata_json: Union[str, Path],
    domain: Optional[str] = None,  # Auto-detect if None
    train_split: str = "train",
    val_split: str = "validation",
    max_files: Optional[int] = None,
    seed: int = 42,
    image_size: int = 256,
    channels: int = 3,
    label_dim: int = 3,
) -> Tuple[EDMPNGDataset, EDMPNGDataset]:
    """
    Create EDM-compatible training and validation datasets from PNG images.

    Args:
        data_root: Root directory containing PNG files
        metadata_json: Metadata JSON file with predefined splits
        domain: Domain name ('photography', 'microscopy', 'astronomy'). Auto-detected if None.
        train_split: Training split name
        val_split: Validation split name
        max_files: Maximum files per split (None for all)
        seed: Random seed
        image_size: Target image size
        channels: Number of channels (3 for RGB)
        label_dim: Label dimension for one-hot domain encoding

    Returns:
        Tuple of (train_dataset, val_dataset) compatible with EDM's native interface
    """
    # Auto-detect domain if not provided
    if domain is None:
        domain = EDMPNGDataset._detect_domain_from_metadata_static(metadata_json)

    logger.info(f"Creating EDM-compatible PNG datasets for {domain}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Channels: {channels}")
    logger.info(f"Label dim: {label_dim}")

    # Create training dataset
    train_dataset = EDMPNGDataset(
        data_root=data_root,
        metadata_json=metadata_json,
        split=train_split,
        max_files=max_files,
        seed=seed,
        image_size=image_size,
        channels=channels,
        domain=domain,
        use_labels=True,
        label_dim=label_dim,
    )

    # Create validation dataset
    val_max_files = max_files // 5 if max_files else None  # Use 20% for validation
    val_dataset = EDMPNGDataset(
        data_root=data_root,
        metadata_json=metadata_json,
        split=val_split,
        max_files=val_max_files,
        seed=seed + 1,
        image_size=image_size,
        channels=channels,
        domain=domain,
        use_labels=True,
        label_dim=label_dim,
    )

    logger.info(f"✅ Created EDM-compatible PNG datasets:")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Resolution: {train_dataset.resolution}x{train_dataset.resolution}")
    logger.info(f"  Channels: {train_dataset.num_channels}")
    logger.info(f"  Label dim: {train_dataset.label_dim}")

    return train_dataset, val_dataset
