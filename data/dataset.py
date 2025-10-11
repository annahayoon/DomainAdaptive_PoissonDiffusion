"""
PT dataset loader for 32-bit float PyTorch tensors.

This module provides dataset classes for loading 32-bit float .pt files
into the diffusion model training pipeline WITHOUT quantization.

Key features:
- Preserves full float32 precision
- Compatible with EDM's training interface
- Handles scientific imaging data (microscopy, astronomy, photography)
- Supports metadata and calibration parameters
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
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


class EDMPTDataset:
    """
    EDM-compatible dataset for 32-bit float .pt files.

    This dataset:
    - Loads .pt files directly without quantization
    - Preserves full float32 precision
    - Provides the interface expected by EDM's native training loop
    - Returns data in EDM-compatible format (float32 images + labels)
    - Compatible with EDM's InfiniteSampler and DataLoader

    Key difference from EDMPNGDataset:
    - No uint8 conversion - maintains float32 throughout
    - No 8-bit quantization loss
    - Assumes .pt files are already normalized or in electron units
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
        data_range: str = "normalized",  # 'normalized' [0,1] or 'electrons' or 'custom'
        **kwargs,
    ):
        """
        Initialize EDM-compatible PT dataset.

        Args:
            data_root: Root directory containing .pt files
            metadata_json: Metadata JSON file with predefined splits
            split: Data split (train, validation, test)
            max_files: Maximum number of files to load
            seed: Random seed for reproducibility
            image_size: Target image size (will validate shape matches)
            channels: Number of channels (1 for grayscale, 3 for RGB)
            domain: Domain name ('photography', 'microscopy', 'astronomy'). Auto-detected if None.
            use_labels: Enable conditioning labels (for domain encoding)
            label_dim: Dimension of label space (3 for one-hot domain encoding)
            data_range: Expected data range - 'normalized' [0,1], 'electrons', or 'custom'
            **kwargs: Additional arguments (ignored for compatibility)
        """
        # Auto-detect domain from metadata if not provided
        if domain is None:
            domain = self._detect_domain_from_metadata(metadata_json)

        self.domain = domain
        self.data_range = data_range

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

        # Initialize dataset parameters
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

        # Find .pt files and create mapping
        self.image_files, self.noisy_mapping = self._find_paired_files()
        self.calibration_params = getattr(self, "calibration_params", {})

        # Create train/val/test split
        self.split_files = self._create_split()

        # Set up raw shape for EDM compatibility (N, C, H, W)
        self._raw_shape = [len(self.split_files), channels, image_size, image_size]

        # Set up labels for domain encoding
        self._use_labels = use_labels
        self._label_dim = label_dim
        self._raw_labels = None
        self._raw_idx = np.arange(len(self.split_files), dtype=np.int64)
        self._xflip = np.zeros(len(self.split_files), dtype=np.uint8)

        # Set up dataset properties for EDM compatibility
        self._name = f"pt_{split}"

        # Cache for loaded images
        self._cached_images = {}
        self._cache = False  # Enable caching if needed for speed

        logger.info(
            f"EDMPTDataset ready: {len(self.split_files)} float32 .pt tiles for '{split}' split - Domain: {self.domain}"
        )
        logger.info(f"  Data range: {self.data_range}")
        logger.info(f"  Shape: [{channels}, {image_size}, {image_size}]")

    def _detect_domain_from_metadata(self, metadata_json: Union[str, Path]) -> str:
        """Auto-detect domain from metadata JSON file."""
        try:
            with open(metadata_json, "r") as f:
                metadata = json.load(f)

            # Try to get domain from top-level field first
            domain = metadata.get("domain")
            if domain and domain in ["photography", "microscopy", "astronomy"]:
                return domain

            # If no top-level domain, check if it's a comprehensive metadata file
            if "domains_processed" in metadata:
                # For comprehensive files, we need to determine domain from context
                # Since this method is called during __init__ and domain is already set,
                # return the current domain or default to photography
                if hasattr(self, "domain") and self.domain:
                    return self.domain
                return "photography"  # Default fallback

            # If domain is None or empty, default to photography
            if not domain:
                logger.warning(
                    f"Domain field is None or empty in metadata, defaulting to photography"
                )
                return "photography"

            # Validate domain value
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

            # Try to get domain from top-level field first
            domain = metadata.get("domain")
            if domain and domain in ["photography", "microscopy", "astronomy"]:
                return domain

            # If no top-level domain, check if it's a comprehensive metadata file
            if "domains_processed" in metadata:
                # For comprehensive files, we can't determine a single domain
                # Return None to indicate that domain should be explicitly specified
                return None

            # If domain is None or empty, return None to indicate explicit specification needed
            if not domain:
                logger.warning(f"Domain field is None or empty in metadata")
                return None

            # Validate domain value
            if domain not in ["photography", "microscopy", "astronomy"]:
                raise ValueError(f"Unsupported domain in metadata: {domain}")

            return domain

        except Exception as e:
            logger.warning(f"Could not auto-detect domain from metadata: {e}")
            # Return None if detection fails - caller should specify domain explicitly
            return None

    def _find_paired_files(self) -> Tuple[List[Path], Dict[Path, Path]]:
        """Find paired clean/noisy .pt files using metadata JSON file."""
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

        # FILTER OUT NOISY TILES FOR PRIOR TRAINING (same logic as PNG dataset)
        clean_tiles_only = []
        noisy_tiles_filtered = []

        for tile in split_tiles:
            tile_id = tile.get("tile_id", "")
            # Check if this is a noisy tile (short exposure)
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
            electron_min = tile.get("electron_min", 0.0)
            electron_max = tile.get("electron_max", 1000.0)
            electron_mean = tile.get("electron_mean", 500.0)
            electron_std = tile.get("electron_std", 100.0)

            self.calibration_params[tile_id] = {
                "gain": tile.get("gain", 1.0),
                "read_noise": tile.get("read_noise", 0.0),
                "background": electron_min,
                "scale": electron_max,
                "electron_min": electron_min,
                "electron_max": electron_max,
                "electron_mean": electron_mean,
                "electron_std": electron_std,
            }

            # Look for .pt file
            # Try multiple possible paths
            pt_path_candidates = [
                # From metadata (if it stores pt_path)
                Path(tile.get("pt_path", "")),
                # Replace .png with .pt in png_path
                Path(str(tile.get("png_path", "")).replace(".png", ".pt")),
                # Construct from data_root
                self.data_root / f"{tile_id}.pt",
                # Try clean subdirectory
                self.data_root / "clean" / f"{tile_id}.pt",
            ]

            clean_path = None
            for candidate in pt_path_candidates:
                if candidate.exists():
                    clean_path = candidate
                    break

            if clean_path is None:
                logger.debug(f"Clean .pt file not found for: {tile_id}")
                continue

            clean_files.append(clean_path)

            # Find corresponding noisy file
            clean_dir = (
                clean_path.parent.parent
                if clean_path.parent.name == "clean"
                else clean_path.parent
            )
            noisy_path = clean_dir / "noisy" / f"{tile_id}.pt"

            if noisy_path.exists():
                noisy_mapping[clean_path] = noisy_path
            else:
                logger.debug(f"No noisy .pt file found for {tile_id}")

        if not clean_files:
            raise FileNotFoundError(
                f"No valid .pt files found for split '{self.split}'"
            )

        logger.info(
            f"✓ Loaded {len(clean_files)} CLEAN .pt tiles (float32) for '{self.split}' split"
        )
        logger.info(
            f"  └─ Calibration parameters: {len(self.calibration_params)} tiles"
        )
        logger.info(f"  └─ Noisy pairs available: {len(noisy_mapping)} tiles")
        return clean_files, noisy_mapping

    def _create_split(self) -> List[Path]:
        """Use only the files designated for the current split in metadata JSON.

        The files in self.image_files are already filtered by _load_tile_metadata()
        to include only files from the current split (train/validation/test) as
        specified in the metadata JSON file.
        """
        # The split files are already filtered by _load_tile_metadata() based on JSON splits
        split_files = self.image_files.copy()

        # Apply max_files limit if specified
        if self.max_files and len(split_files) > self.max_files:
            split_files = split_files[: self.max_files]
            logger.info(f"Limited to {self.max_files} files for {self.split} split")

        logger.info(
            f"Using {len(split_files)} files for '{self.split}' split as designated in metadata JSON"
        )
        return split_files

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
        """
        Load raw image in EDM format (float32, CHW).

        CRITICAL: Returns float32 tensors already in [-1,1] range from pipeline!
        The pipeline applies domain-specific scaling followed by [-1,1] normalization.
        """
        # Get image using integrated logic
        pt_item = self._load_pt_item(raw_idx)

        # Extract clean image (already in float32, [-1,1] normalized by pipeline)
        clean_image = pt_item["clean"]

        # Convert to numpy and ensure float32 format
        image_np = (
            clean_image.numpy()
            if isinstance(clean_image, torch.Tensor)
            else clean_image
        )

        # Ensure float32 dtype
        if image_np.dtype != np.float32:
            image_np = image_np.astype(np.float32)

        # Ensure shape is (C, H, W) - EDM expects CHW format
        if image_np.ndim == 2:
            # Grayscale to (1, H, W)
            image_np = np.expand_dims(image_np, axis=0)
        elif image_np.ndim == 3 and image_np.shape[-1] in [1, 3]:  # HWC format
            # Convert from HWC to CHW
            image_np = np.transpose(image_np, (2, 0, 1))

        # Data is already in [-1,1] range from pipeline normalization
        # No additional normalization needed
        return image_np

    def __len__(self):
        """Return dataset length."""
        return len(self.split_files)

    def _load_pt_item(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a clean .pt file.

        Returns dict with:
            - 'clean': Clean image data (C, H, W) in float32 - used for training
            - 'electrons': Clean image data in electron units - used for physics-aware loss
            - 'domain': Domain tensor
            - 'metadata': Image metadata
            - 'domain_params': Domain-specific parameters from metadata calibration
        """
        try:
            clean_path = self.split_files[idx]

            # Load .pt file directly as float32
            clean = self._load_pt_image_direct(clean_path)

            # Extract metadata
            metadata = {
                "clean_path": str(clean_path),
                "filename": clean_path.name,
                "image_size": self.image_size,
                "channels": self.channels,
            }

            # Get calibration parameters for this tile from metadata
            tile_id = clean_path.name.replace(".pt", "")
            calib_params = self.calibration_params.get(tile_id, {})

            # Use electron statistics from metadata for physics-aware training
            scale = calib_params.get("electron_max", 1000.0)
            background = calib_params.get("electron_min", 0.0)

            calib_params_dict = {
                "scale": scale,
                "gain": calib_params.get("gain", 1.0),
                "read_noise": calib_params.get("read_noise", 0.0),
                "background": background,
            }

            # Domain parameters from metadata
            domain_params = {
                "scale": calib_params_dict["scale"],
                "gain": calib_params_dict["gain"],
                "read_noise": calib_params_dict["read_noise"],
                "background": calib_params_dict["background"],
            }

            # Convert clean image to electron units if needed
            if self.data_range == "normalized":
                # Assume data is in [0, 1], scale to electron units
                electrons = (
                    clean * calib_params_dict["scale"] + calib_params_dict["background"]
                )
            elif self.data_range == "electrons":
                # Data is already in electron units
                electrons = (
                    clean.clone()
                    if isinstance(clean, torch.Tensor)
                    else torch.from_numpy(clean.copy())
                )
            else:
                # Custom range - use as-is
                electrons = (
                    clean.clone()
                    if isinstance(clean, torch.Tensor)
                    else torch.from_numpy(clean.copy())
                )

            # Domain encoding for physics-aware training
            domain_encoding = {"photography": 0, "microscopy": 1, "astronomy": 2}
            domain_id = domain_encoding.get(self.domain, 0)

            return {
                "clean": clean,
                "electrons": electrons,
                "domain": torch.tensor(domain_id),
                "metadata": metadata,
                "domain_params": domain_params,
                # Flatten calibration parameters for trainer compatibility
                "scale": torch.tensor(
                    [calib_params_dict["scale"]], dtype=torch.float32
                ),
                "read_noise": torch.tensor(
                    [calib_params_dict["read_noise"]], dtype=torch.float32
                ),
                "background": torch.tensor(
                    [calib_params_dict["background"]], dtype=torch.float32
                ),
                "gain": torch.tensor([calib_params_dict["gain"]], dtype=torch.float32),
            }

        except Exception as e:
            logger.warning(f"Error loading .pt file {idx}: {e}")
            # Return dummy data to prevent training crashes
            return {
                "clean": torch.zeros(
                    self.channels, self.image_size, self.image_size, dtype=torch.float32
                ),
                "electrons": torch.zeros(
                    self.channels, self.image_size, self.image_size, dtype=torch.float32
                ),
                "domain": torch.tensor(0),
                "metadata": {"corrupted": True, "original_idx": idx},
                "domain_params": {
                    "scale": 1000.0,
                    "gain": 1.0,
                    "read_noise": 0.0,
                    "background": 0.0,
                },
                "scale": torch.tensor([1000.0], dtype=torch.float32),
                "read_noise": torch.tensor([0.0], dtype=torch.float32),
                "background": torch.tensor([0.0], dtype=torch.float32),
                "gain": torch.tensor([1.0], dtype=torch.float32),
            }

    def _load_pt_image_direct(self, image_path: Path) -> torch.Tensor:
        """Load a .pt file directly as float32 tensor.

        The .pt files are already in [-1,1] range from the pipeline normalization.
        """
        try:
            # Load .pt file (already in [-1,1] range from pipeline)
            tensor = torch.load(str(image_path), map_location="cpu")

            # Ensure float32
            if tensor.dtype != torch.float32:
                tensor = tensor.to(torch.float32)

            # Validate shape
            if tensor.ndim == 2:
                # Grayscale image (H, W)
                if tensor.shape != (self.image_size, self.image_size):
                    logger.warning(
                        f"Image shape {tensor.shape} doesn't match expected {(self.image_size, self.image_size)}"
                    )
                # Will be converted to (1, H, W) later
            elif tensor.ndim == 3:
                # Multi-channel image - could be (C, H, W) or (H, W, C)
                if tensor.shape[0] in [1, 3]:  # CHW format
                    expected_shape = (self.channels, self.image_size, self.image_size)
                else:  # HWC format
                    expected_shape = (self.image_size, self.image_size, self.channels)

                # For multi-domain datasets, allow both grayscale (1 channel) and RGB (3 channels)
                if tensor.shape != expected_shape and tensor.shape[:2] != (
                    self.image_size,
                    self.image_size,
                ):
                    # Check if this is a grayscale image in a multi-domain context
                    if (
                        hasattr(self, "domains")
                        and tensor.shape[0] == 1
                        and self.channels == 3
                    ):
                        logger.debug(
                            f"Grayscale image {tensor.shape} will be converted to RGB"
                        )
                    else:
                        logger.warning(
                            f"Image shape {tensor.shape} doesn't match expected size {self.image_size}"
                        )
            else:
                raise ValueError(f"Unexpected number of dimensions: {tensor.ndim}")

            # Ensure CHW format
            if tensor.ndim == 2:
                # Add channel dimension
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3]:
                # Convert HWC to CHW
                tensor = tensor.permute(2, 0, 1)

            # Data is already in [-1,1] range from pipeline normalization
            # Verify the range is approximately correct
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            if min_val < -1.1 or max_val > 1.1:
                logger.warning(
                    f"Image {image_path.name} has unexpected range [{min_val:.3f}, {max_val:.3f}], "
                    f"expected approximately [-1, 1]"
                )

            return tensor

        except Exception as e:
            logger.warning(f"Error loading .pt file {image_path}: {e}")
            # Return dummy image in [-1,1] range
            return torch.zeros(
                self.channels, self.image_size, self.image_size, dtype=torch.float32
            )

    def __getitem__(self, idx):
        """
        Get item as (image, label) tuple for EDM compatibility.

        CRITICAL: Returns float32 tensors already in [-1,1] range from pipeline!
        No additional normalization needed - data is ready for EDM training.
        """
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

        # Data is already in [-1,1] range from pipeline normalization
        # Return as-is for EDM training
        return image.copy(), label

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
        return self._load_pt_item(idx).get("domain_params", {})

    def close(self):
        """Clean up resources."""
        pass


def create_edm_pt_datasets(
    data_root: Union[str, Path],
    metadata_json: Union[str, Path],
    domain: Optional[str] = None,  # Must specify for comprehensive metadata files
    train_split: str = "train",
    val_split: str = "validation",
    max_files: Optional[int] = None,
    seed: int = 42,
    image_size: int = 256,
    channels: int = 3,
    label_dim: int = 3,
    data_range: str = "normalized",
) -> Tuple[EDMPTDataset, EDMPTDataset]:
    """
    Create EDM-compatible training and validation datasets from .pt files.

    Args:
        data_root: Root directory containing .pt files
        metadata_json: Metadata JSON file with predefined splits
        domain: Domain name ('photography', 'microscopy', 'astronomy'). Required for comprehensive metadata files.
        train_split: Training split name
        val_split: Validation split name
        max_files: Maximum files per split (None for all)
        seed: Random seed
        image_size: Target image size
        channels: Number of channels (1 for grayscale, 3 for RGB)
        label_dim: Label dimension for one-hot domain encoding
        data_range: Expected data range - 'normalized' [0,1], 'electrons', or 'custom'

    Returns:
        Tuple of (train_dataset, val_dataset) compatible with EDM's native interface
    """
    # Auto-detect domain if not provided and metadata supports it
    if domain is None:
        detected_domain = EDMPTDataset._detect_domain_from_metadata_static(
            metadata_json
        )
        if detected_domain is None:
            raise ValueError(
                f"Domain could not be auto-detected from metadata file {metadata_json}. "
                "Please specify domain explicitly. Supported domains: photography, microscopy, astronomy"
            )
        domain = detected_domain

    logger.info(f"Creating EDM-compatible PT datasets for {domain}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Channels: {channels}")
    logger.info(f"Label dim: {label_dim}")
    logger.info(f"Data range: {data_range}")

    # Create training dataset
    train_dataset = EDMPTDataset(
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
        data_range=data_range,
    )

    # Create validation dataset
    val_max_files = max_files // 5 if max_files else None  # Use 20% for validation
    val_dataset = EDMPTDataset(
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
        data_range=data_range,
    )

    logger.info(f"✅ Created EDM-compatible PT datasets (float32, no quantization):")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Resolution: {train_dataset.resolution}x{train_dataset.resolution}")
    logger.info(f"  Channels: {train_dataset.num_channels}")
    logger.info(f"  Label dim: {train_dataset.label_dim}")

    return train_dataset, val_dataset


class MultiDomainEDMPTDataset(EDMPTDataset):
    """
    Multi-domain EDM-compatible dataset for cross-domain generalization training.

    This dataset loads data from multiple domains (photography, microscopy, astronomy)
    by combining individual domain datasets and provides domain labels for conditional training.
    It loads each domain's data using their respective metadata files.

    Key features:
    - Loads data from all domains simultaneously using separate metadata files
    - Provides domain labels for conditional training
    - Combines individual domain datasets into a unified dataset
    - Maintains float32 precision throughout
    - Compatible with EDM's training interface
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        metadata_json: Union[
            str, Path
        ],  # Can be a list of metadata files or single comprehensive file
        split: str = "train",
        max_files: Optional[int] = None,
        seed: int = 42,
        image_size: int = 256,
        channels: int = 3,
        domains: Optional[List[str]] = None,  # List of domains to include
        use_labels: bool = True,
        label_dim: int = 3,  # One-hot domain encoding for 3 domains
        data_range: str = "normalized",
        **kwargs,
    ):
        """
        Initialize multi-domain EDM-compatible PT dataset.

        Args:
            data_root: Root directory containing .pt files
            metadata_json: Metadata JSON file(s) - can be single comprehensive file or list of domain-specific files
            split: Data split (train, validation, test)
            max_files: Maximum number of files to load
            seed: Random seed for reproducibility
            image_size: Target image size (will validate shape matches)
            channels: Number of channels (1 for grayscale, 3 for RGB)
            domains: List of domains to include ['photography', 'microscopy', 'astronomy']
            use_labels: Enable conditioning labels (for domain encoding)
            label_dim: Dimension of label space (3 for one-hot domain encoding)
            data_range: Expected data range - 'normalized' [0,1], 'electrons', or 'custom'
            **kwargs: Additional arguments (ignored for compatibility)
        """
        # Set up multi-domain configuration
        if domains is None:
            domains = ["photography", "microscopy", "astronomy"]

        self.domains = domains
        self.data_range = data_range

        # Domain encoding mapping (same as single-domain)
        self.domain_to_label = {
            "photography": [1, 0, 0],
            "microscopy": [0, 1, 0],
            "astronomy": [0, 0, 1],
        }

        # Validate domains
        for domain in domains:
            if domain not in self.domain_to_label:
                raise ValueError(
                    f"Unsupported domain: {domain}. Supported: {list(self.domain_to_label.keys())}"
                )

        # Initialize dataset parameters
        self.data_root = Path(data_root)
        self.split = split
        self.max_files = max_files
        self.seed = seed
        self.image_size = image_size
        self.channels = channels

        # Set up labels for domain encoding (needed before loading domain datasets)
        self._use_labels = use_labels
        self._label_dim = label_dim

        # Set domain to None for multi-domain dataset (will be determined per sample)
        self.domain = None

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")

        # Handle metadata - can be single file or list of domain-specific files
        self.metadata_json = metadata_json
        self.domain_datasets = self._load_domain_datasets()

        # Combine all domain datasets
        self.image_files, self.noisy_mapping = self._combine_domain_files()
        self.calibration_params = getattr(self, "calibration_params", {})

        # Create train/val/test split across all domains
        self.split_files = self._create_combined_split()

        # Set up raw shape for EDM compatibility (N, C, H, W)
        self._raw_shape = [len(self.split_files), channels, image_size, image_size]

        # Set up remaining label properties
        self._raw_labels = None
        self._raw_idx = np.arange(len(self.split_files), dtype=np.int64)
        self._xflip = np.zeros(len(self.split_files), dtype=np.uint8)

        # Set up dataset properties for EDM compatibility
        self._name = f"multi_domain_pt_{split}"

        # Cache for loaded images
        self._cached_images = {}

        logger.info(f"Initialized MultiDomainEDMPTDataset:")
        logger.info(f"  Domains: {self.domains}")
        logger.info(f"  Split: {self.split}")
        logger.info(f"  Files: {len(self.split_files)}")
        logger.info(f"  Image size: {image_size}x{image_size}")
        logger.info(f"  Channels: {channels}")
        logger.info(f"  Label dim: {label_dim}")

    def _load_domain_datasets(self) -> Dict[str, EDMPTDataset]:
        """Load individual domain datasets."""
        domain_datasets = {}

        # Default metadata file paths for each domain
        default_metadata_files = {
            "photography": "dataset/processed/metadata_photography_incremental.json",
            "microscopy": "dataset/processed/metadata_microscopy_incremental.json",
            "astronomy": "dataset/processed/metadata_astronomy_incremental.json",
        }

        # Domain-specific channel configurations
        domain_channels = {
            "photography": 3,  # RGB
            "microscopy": 1,  # Grayscale
            "astronomy": 1,  # Grayscale
        }

        # Determine metadata files to use
        if isinstance(self.metadata_json, (list, tuple)):
            # List of metadata files provided
            metadata_files = self.metadata_json
            if len(metadata_files) != len(self.domains):
                raise ValueError(
                    f"Number of metadata files ({len(metadata_files)}) must match "
                    f"number of domains ({len(self.domains)})"
                )
            metadata_dict = dict(zip(self.domains, metadata_files))
        else:
            # Single metadata file - try to determine if it's comprehensive or use defaults
            metadata_path = Path(self.metadata_json)
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    # Check if it's a comprehensive metadata file
                    if "domains_processed" in metadata:
                        logger.info("Using comprehensive metadata file")
                        # For comprehensive file, we'll handle it differently
                        return self._load_comprehensive_dataset(metadata_path)
                    else:
                        # Single domain file, use defaults for other domains
                        logger.info(
                            "Using single domain metadata file, will use defaults for other domains"
                        )
                        detected_domain = metadata.get("domain", "photography")
                        metadata_dict = default_metadata_files.copy()
                        metadata_dict[detected_domain] = str(metadata_path)
                except Exception as e:
                    logger.warning(
                        f"Could not parse metadata file {metadata_path}: {e}"
                    )
                    metadata_dict = default_metadata_files.copy()
            else:
                # File doesn't exist, use defaults
                logger.info(
                    f"Metadata file {metadata_path} not found, using default domain metadata files"
                )
                metadata_dict = default_metadata_files.copy()

        # Load each domain dataset
        for domain in self.domains:
            metadata_file = metadata_dict.get(domain)
            if not metadata_file:
                logger.warning(
                    f"No metadata file specified for domain {domain}, skipping"
                )
                continue

            metadata_path = Path(metadata_file)
            if not metadata_path.exists():
                logger.warning(
                    f"Metadata file for {domain} not found: {metadata_path}, skipping"
                )
                continue

            try:
                # Get domain-specific channel count
                domain_channel_count = domain_channels.get(domain, self.channels)

                # Create individual domain dataset
                domain_dataset = EDMPTDataset(
                    data_root=self.data_root,
                    metadata_json=metadata_path,
                    split=self.split,
                    max_files=self.max_files // len(self.domains)
                    if self.max_files
                    else None,
                    seed=self.seed,
                    image_size=self.image_size,
                    channels=domain_channel_count,  # Use domain-specific channel count
                    domain=domain,
                    use_labels=True,
                    label_dim=self._label_dim,
                    data_range=self.data_range,
                )
                domain_datasets[domain] = domain_dataset
                logger.info(
                    f"Loaded {domain} dataset: {len(domain_dataset)} samples ({domain_channel_count} channels)"
                )

            except Exception as e:
                logger.error(f"Failed to load {domain} dataset: {e}")
                continue

        if not domain_datasets:
            raise ValueError("No domain datasets could be loaded")

        logger.info(f"Successfully loaded {len(domain_datasets)} domain datasets")
        return domain_datasets

    def _load_comprehensive_dataset(
        self, metadata_path: Path
    ) -> Dict[str, EDMPTDataset]:
        """Load dataset from comprehensive metadata file."""
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Check if this is a comprehensive metadata file
            if "domains_processed" not in metadata:
                raise ValueError(
                    f"Metadata file {metadata_path} is not a comprehensive multi-domain file. "
                    "Expected 'domains_processed' field."
                )

            # Validate that all requested domains are present
            processed_domains = metadata.get("domains_processed", [])
            missing_domains = set(self.domains) - set(processed_domains)
            if missing_domains:
                raise ValueError(
                    f"Missing domains in metadata: {missing_domains}. "
                    f"Available domains: {processed_domains}"
                )

            # Create a single dataset from comprehensive metadata
            comprehensive_dataset = EDMPTDataset(
                data_root=self.data_root,
                metadata_json=metadata_path,
                split=self.split,
                max_files=self.max_files,
                seed=self.seed,
                image_size=self.image_size,
                channels=self.channels,
                domain=None,  # Will be auto-detected
                use_labels=True,
                label_dim=self._label_dim,
                data_range=self.data_range,
            )

            # Return as a single "comprehensive" domain
            return {"comprehensive": comprehensive_dataset}

        except Exception as e:
            logger.error(f"Failed to load comprehensive metadata: {e}")
            raise

    def _combine_domain_files(self) -> Tuple[List[Path], Dict[Path, Path]]:
        """Combine files from all domain datasets."""
        all_image_files = []
        all_noisy_mapping = {}

        for domain, dataset in self.domain_datasets.items():
            # Get files from this domain dataset
            domain_files = dataset.image_files
            domain_noisy_mapping = dataset.noisy_mapping

            # Add domain information to the file paths for later identification
            for img_file in domain_files:
                all_image_files.append(img_file)
                all_noisy_mapping[img_file] = domain_noisy_mapping.get(
                    img_file, img_file
                )

            logger.info(f"Added {len(domain_files)} files from {domain} domain")

        # Remove duplicates
        all_image_files = list(set(all_image_files))

        logger.info(f"Total files combined from all domains: {len(all_image_files)}")
        return all_image_files, all_noisy_mapping

    def _create_combined_split(self) -> List[Path]:
        """Create combined train/val/test split from all domain datasets."""
        all_split_files = []

        for domain, dataset in self.domain_datasets.items():
            # Get split files from this domain dataset
            domain_split_files = dataset.split_files
            all_split_files.extend(domain_split_files)
            logger.info(
                f"Added {len(domain_split_files)} {self.split} files from {domain} domain"
            )

        # Apply max_files limit if specified
        if self.max_files and len(all_split_files) > self.max_files:
            np.random.seed(self.seed)
            all_split_files = np.random.choice(
                all_split_files, self.max_files, replace=False
            ).tolist()

        logger.info(
            f"Created combined {self.split} split with {len(all_split_files)} files"
        )
        return all_split_files

    def _get_domain_from_file(self, file_path: Path) -> str:
        """Extract domain from file path."""
        # Try to extract domain from path
        path_parts = file_path.parts

        # Check if domain is in the path
        for domain in self.domains:
            if domain in path_parts:
                return domain

        # Check if domain is in filename
        filename = file_path.stem.lower()
        for domain in self.domains:
            if domain in filename:
                return domain

        # Default fallback
        logger.warning(
            f"Could not determine domain for {file_path}, defaulting to photography"
        )
        return "photography"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item with domain label."""
        # Get the image using the base class method
        image = self._load_pt_item(idx)["clean"]

        # Get domain label
        file_path = self.split_files[idx]
        domain = self._get_domain_from_file(file_path)
        domain_label = torch.tensor(self.domain_to_label[domain], dtype=torch.float32)

        # Convert grayscale to RGB if needed for multi-domain consistency
        if image.shape[0] == 1 and self.channels == 3:
            # Convert grayscale (1 channel) to RGB (3 channels) by repeating
            image = image.repeat(3, 1, 1)
            logger.debug(f"Converted grayscale to RGB for domain {domain}")
        elif image.shape[0] == 3 and self.channels == 1:
            # Convert RGB (3 channels) to grayscale (1 channel) by averaging
            image = image.mean(dim=0, keepdim=True)
            logger.debug(f"Converted RGB to grayscale for domain {domain}")

        return image, domain_label

    @property
    def resolution(self) -> int:
        """Get image resolution."""
        return self.image_size

    @property
    def num_channels(self) -> int:
        """Get number of channels."""
        return self.channels

    @property
    def label_dim(self) -> int:
        """Get label dimension."""
        return self._label_dim


def create_multi_domain_edm_pt_datasets(
    data_root: Union[str, Path],
    metadata_json: Union[
        str, Path, List[str]
    ],  # Can be single file or list of domain-specific files
    domains: Optional[List[str]] = None,
    train_split: str = "train",
    val_split: str = "validation",
    max_files: Optional[int] = None,
    seed: int = 42,
    image_size: int = 256,
    channels: int = 3,
    label_dim: int = 3,
    data_range: str = "normalized",
) -> Tuple[MultiDomainEDMPTDataset, MultiDomainEDMPTDataset]:
    """
    Create multi-domain EDM-compatible training and validation datasets.

    Args:
        data_root: Root directory containing .pt files
        metadata_json: Metadata JSON file(s) - can be single comprehensive file or list of domain-specific files
        domains: List of domains to include ['photography', 'microscopy', 'astronomy']
        train_split: Training split name
        val_split: Validation split name
        max_files: Maximum files per split (None for all)
        seed: Random seed
        image_size: Target image size
        channels: Number of channels (1 for grayscale, 3 for RGB)
        label_dim: Label dimension for one-hot domain encoding
        data_range: Expected data range - 'normalized' [0,1], 'electrons', or 'custom'

    Returns:
        Tuple of (train_dataset, val_dataset) compatible with EDM's native interface
    """
    if domains is None:
        domains = ["photography", "microscopy", "astronomy"]

    logger.info(f"Creating multi-domain EDM-compatible PT datasets")
    logger.info(f"Domains: {domains}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Channels: {channels}")
    logger.info(f"Label dim: {label_dim}")
    logger.info(f"Data range: {data_range}")

    # Create training dataset
    train_dataset = MultiDomainEDMPTDataset(
        data_root=data_root,
        metadata_json=metadata_json,
        split=train_split,
        max_files=max_files,
        seed=seed,
        image_size=image_size,
        channels=channels,
        domains=domains,
        use_labels=True,
        label_dim=label_dim,
        data_range=data_range,
    )

    # Create validation dataset
    val_max_files = max_files // 5 if max_files else None  # Use 20% for validation
    val_dataset = MultiDomainEDMPTDataset(
        data_root=data_root,
        metadata_json=metadata_json,
        split=val_split,
        max_files=val_max_files,
        seed=seed + 1,
        image_size=image_size,
        channels=channels,
        domains=domains,
        use_labels=True,
        label_dim=label_dim,
        data_range=data_range,
    )

    logger.info(
        f"✅ Created multi-domain EDM-compatible PT datasets (float32, no quantization):"
    )
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Resolution: {train_dataset.resolution}x{train_dataset.resolution}")
    logger.info(f"  Channels: {train_dataset.num_channels}")
    logger.info(f"  Label dim: {train_dataset.label_dim}")
    logger.info(f"  Domains: {domains}")

    return train_dataset, val_dataset
