"""
NPY dataset loader for 32-bit float numpy arrays.

This module provides dataset classes for loading 32-bit float .npy files
into the diffusion model training pipeline WITHOUT quantization.

Key features:
- Preserves full float32 precision (no 8-bit quantization)
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


class EDMNPYDataset:
    """
    EDM-compatible dataset for 32-bit float .npy files.

    This dataset:
    - Loads .npy files directly without quantization
    - Preserves full float32 precision
    - Provides the interface expected by EDM's native training loop
    - Returns data in EDM-compatible format (float32 images + labels)
    - Compatible with EDM's InfiniteSampler and DataLoader

    Key difference from EDMPNGDataset:
    - No uint8 conversion - maintains float32 throughout
    - No 8-bit quantization loss
    - Assumes .npy files are already normalized or in electron units
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
        Initialize EDM-compatible NPY dataset.

        Args:
            data_root: Root directory containing .npy files
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

        # Find .npy files and create mapping
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
        self._name = f"npy_{split}"

        # Cache for loaded images
        self._cached_images = {}
        self._cache = False  # Enable caching if needed for speed

        logger.info(
            f"EDMNPYDataset ready: {len(self.split_files)} float32 .npy tiles for '{split}' split - Domain: {self.domain}"
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
        """Find paired clean/noisy .npy files using metadata JSON file."""
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

            # Look for .npy file instead of .png
            # Try multiple possible paths
            npy_path_candidates = [
                # From metadata (if it stores npy_path)
                Path(tile.get("npy_path", "")),
                # Replace .png with .npy in png_path
                Path(str(tile.get("png_path", "")).replace(".png", ".npy")),
                # Construct from data_root
                self.data_root / f"{tile_id}.npy",
                # Try clean subdirectory
                self.data_root / "clean" / f"{tile_id}.npy",
            ]

            clean_path = None
            for candidate in npy_path_candidates:
                if candidate.exists():
                    clean_path = candidate
                    break

            if clean_path is None:
                logger.debug(f"Clean .npy file not found for: {tile_id}")
                continue

            clean_files.append(clean_path)

            # Find corresponding noisy file
            clean_dir = (
                clean_path.parent.parent
                if clean_path.parent.name == "clean"
                else clean_path.parent
            )
            noisy_path = clean_dir / "noisy" / f"{tile_id}.npy"

            if noisy_path.exists():
                noisy_mapping[clean_path] = noisy_path
            else:
                logger.debug(f"No noisy .npy file found for {tile_id}")

        if not clean_files:
            raise FileNotFoundError(
                f"No valid .npy files found for split '{self.split}'"
            )

        logger.info(
            f"✓ Loaded {len(clean_files)} CLEAN .npy tiles (float32) for '{self.split}' split"
        )
        logger.info(
            f"  └─ Calibration parameters: {len(self.calibration_params)} tiles"
        )
        logger.info(f"  └─ Noisy pairs available: {len(noisy_mapping)} tiles")
        return clean_files, noisy_mapping

    def _create_split(self) -> List[Path]:
        """Apply max_files limit to image files (splits already defined in metadata)."""
        split_files = self.image_files.copy()

        # Apply max_files limit if specified
        if self.max_files and len(split_files) > self.max_files:
            split_files = split_files[: self.max_files]
            logger.info(f"Limited to {self.max_files} files for {self.split} split")

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

        CRITICAL: Returns float32 arrays, NOT uint8!
        This overrides EDM's default uint8 expectation.
        """
        # Get image using integrated logic
        npy_item = self._load_npy_item(raw_idx)

        # Extract clean image (already in float32, normalized or electron units)
        clean_image = npy_item["clean"]

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

        return image_np

    def __len__(self):
        """Return dataset length."""
        return len(self.split_files)

    def _load_npy_item(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a clean .npy file.

        Returns dict with:
            - 'clean': Clean image data (C, H, W) in float32 - used for training
            - 'electrons': Clean image data in electron units - used for physics-aware loss
            - 'domain': Domain tensor
            - 'metadata': Image metadata
            - 'domain_params': Domain-specific parameters from metadata calibration
        """
        try:
            clean_path = self.split_files[idx]

            # Load .npy file directly as float32
            clean = self._load_npy_image_direct(clean_path)

            # Extract metadata
            metadata = {
                "clean_path": str(clean_path),
                "filename": clean_path.name,
                "image_size": self.image_size,
                "channels": self.channels,
            }

            # Get calibration parameters for this tile from metadata
            tile_id = clean_path.name.replace(".npy", "")
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
            logger.warning(f"Error loading .npy file {idx}: {e}")
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

    def _load_npy_image_direct(self, image_path: Path) -> torch.Tensor:
        """Load a .npy file directly as float32 tensor."""
        try:
            # Load .npy file
            image_np = np.load(str(image_path)).astype(np.float32)

            # Validate shape
            if image_np.ndim == 2:
                # Grayscale image (H, W)
                if image_np.shape != (self.image_size, self.image_size):
                    logger.warning(
                        f"Image shape {image_np.shape} doesn't match expected {(self.image_size, self.image_size)}"
                    )
                # Will be converted to (1, H, W) later
            elif image_np.ndim == 3:
                # Multi-channel image - could be (C, H, W) or (H, W, C)
                if image_np.shape[0] in [1, 3]:  # CHW format
                    expected_shape = (self.channels, self.image_size, self.image_size)
                else:  # HWC format
                    expected_shape = (self.image_size, self.image_size, self.channels)

                if image_np.shape != expected_shape and image_np.shape[:2] != (
                    self.image_size,
                    self.image_size,
                ):
                    logger.warning(
                        f"Image shape {image_np.shape} doesn't match expected size {self.image_size}"
                    )
            else:
                raise ValueError(f"Unexpected number of dimensions: {image_np.ndim}")

            # Convert to PyTorch tensor
            tensor = torch.from_numpy(image_np).float()

            # Ensure CHW format
            if tensor.ndim == 2:
                # Add channel dimension
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3]:
                # Convert HWC to CHW
                tensor = tensor.permute(2, 0, 1)

            return tensor

        except Exception as e:
            logger.warning(f"Error loading .npy file {image_path}: {e}")
            # Return dummy image
            return torch.zeros(
                self.channels, self.image_size, self.image_size, dtype=torch.float32
            )

    def __getitem__(self, idx):
        """
        Get item as (image, label) tuple for EDM compatibility.

        CRITICAL: Returns float32 arrays, NOT uint8!
        The training script will need to handle float32 data appropriately.
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
        return self._load_npy_item(idx).get("domain_params", {})

    def close(self):
        """Clean up resources."""
        pass


def create_edm_npy_datasets(
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
) -> Tuple[EDMNPYDataset, EDMNPYDataset]:
    """
    Create EDM-compatible training and validation datasets from .npy files.

    Args:
        data_root: Root directory containing .npy files
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
        detected_domain = EDMNPYDataset._detect_domain_from_metadata_static(
            metadata_json
        )
        if detected_domain is None:
            raise ValueError(
                f"Domain could not be auto-detected from metadata file {metadata_json}. "
                "Please specify domain explicitly. Supported domains: photography, microscopy, astronomy"
            )
        domain = detected_domain

    logger.info(f"Creating EDM-compatible NPY datasets for {domain}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Channels: {channels}")
    logger.info(f"Label dim: {label_dim}")
    logger.info(f"Data range: {data_range}")

    # Create training dataset
    train_dataset = EDMNPYDataset(
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
    val_dataset = EDMNPYDataset(
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

    logger.info(f"✅ Created EDM-compatible NPY datasets (float32, no quantization):")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Resolution: {train_dataset.resolution}x{train_dataset.resolution}")
    logger.info(f"  Channels: {train_dataset.num_channels}")
    logger.info(f"  Label dim: {train_dataset.label_dim}")

    return train_dataset, val_dataset
