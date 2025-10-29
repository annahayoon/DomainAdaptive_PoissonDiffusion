"""
PT dataset loader for 32-bit float PyTorch tensors.

This module provides dataset classes for loading 32-bit float .pt files
into the diffusion model training pipeline WITHOUT quantization.

Key features:
- Preserves full float32 precision
- Compatible with EDM's training interface
- Loads pre-normalized [-1, 1] data from preprocessing pipeline
- No domain or label support - simple image-only dataset
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
    - Returns data in EDM-compatible format (float32 images only)
    - Compatible with EDM's InfiniteSampler and DataLoader

    Data normalization:
    - Files are pre-normalized to [-1, 1] range during preprocessing
    - Processing pipeline: raw → [0,1] during demosaic → [-1,1] when saved as .pt
    - No additional normalization needed - data is ready for EDM training
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
            **kwargs: Additional arguments (ignored for compatibility)
        """

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

        # Create train/val/test split
        self.split_files = self._create_split()

        # Set up raw shape for EDM compatibility (N, C, H, W)
        self._raw_shape = [len(self.split_files), channels, image_size, image_size]

        # Set up basic properties
        self._raw_idx = np.arange(len(self.split_files), dtype=np.int64)
        self._xflip = np.zeros(len(self.split_files), dtype=np.uint8)

        # Set up dataset properties for EDM compatibility
        self._name = f"pt_{split}"

        # Cache for loaded images
        self._cached_images = {}
        self._cache = False  # Enable caching if needed for speed

        logger.info(
            f"EDMPTDataset ready: {len(self.split_files)} float32 .pt tiles for '{split}' split"
        )
        logger.info(f"  Shape: [{channels}, {image_size}, {image_size}]")
        logger.info(f"  Data range: [-1, 1] (pre-normalized during preprocessing)")

    def _find_paired_files(self) -> Tuple[List[Path], Dict[Path, Path]]:
        """Find paired clean/noisy .pt files using metadata JSON file."""
        logger.info(f"Loading tile metadata from {self.metadata_json}")

        with open(self.metadata_json, "r") as f:
            metadata = json.load(f)

        # Extract all tiles
        all_tiles = metadata.get("tiles", [])

        if not all_tiles:
            raise FileNotFoundError(f"No tiles found in metadata {self.metadata_json}")

        logger.info(f"Found {len(all_tiles)} tiles in metadata")

        # Group tiles by split
        tiles_by_split = {"train": [], "validation": [], "test": []}
        for tile in all_tiles:
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

            # Check if this is a noisy tile - prioritize explicit metadata fields
            is_noisy = False

            # First check explicit metadata fields (for astronomy_v2 and other updated datasets)
            if "is_clean" in tile:
                is_noisy = not tile.get("is_clean", True)
            elif "data_type" in tile:
                is_noisy = tile.get("data_type", "clean") == "noisy"
            else:
                # Fallback to pattern-based detection for older metadata files
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

        # Extract clean file paths
        clean_files = []
        noisy_mapping = {}

        for tile in split_tiles:
            tile_id = tile.get("tile_id")
            if not tile_id:
                continue

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

    def _load_raw_image(self, raw_idx):
        """
        Load raw image in EDM format (float32, CHW).

        Returns:
            numpy.ndarray: Image in (C, H, W) format, float32, [-1, 1] range
        """
        pt_item = self._load_pt_item(raw_idx)
        clean_image = pt_item["clean"]

        # Convert to numpy
        image_np = (
            clean_image.numpy()
            if isinstance(clean_image, torch.Tensor)
            else clean_image
        )

        # Ensure float32 dtype
        if image_np.dtype != np.float32:
            image_np = image_np.astype(np.float32)

        # Ensure CHW format
        if image_np.ndim == 2:
            image_np = np.expand_dims(image_np, axis=0)
        elif image_np.ndim == 3 and image_np.shape[-1] in [1, 3]:
            image_np = np.transpose(image_np, (2, 0, 1))

        return image_np

    def __len__(self):
        """Return dataset length."""
        return len(self.split_files)

    def _load_pt_item(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a .pt file.

        Args:
            idx: Index of the item to load

        Returns:
            dict with keys:
                - 'clean': torch.Tensor of shape (C, H, W), float32, [-1, 1] range
                - 'metadata': dict with file information
        """
        try:
            clean_path = self.split_files[idx]

            # Load .pt file directly as float32 (already in [-1,1] range)
            clean = self._load_pt_image_direct(clean_path)

            # Extract metadata
            metadata = {
                "clean_path": str(clean_path),
                "filename": clean_path.name,
                "image_size": self.image_size,
                "channels": self.channels,
            }

            return {
                "clean": clean,
                "metadata": metadata,
            }

        except Exception as e:
            logger.warning(f"Error loading .pt file {idx}: {e}")
            # Return dummy data to prevent training crashes
            return {
                "clean": torch.zeros(
                    self.channels, self.image_size, self.image_size, dtype=torch.float32
                ),
                "metadata": {"corrupted": True, "original_idx": idx},
            }

    def _load_pt_image_direct(self, image_path: Path) -> torch.Tensor:
        """
        Load a .pt file directly as float32 tensor.

        Args:
            image_path: Path to the .pt file

        Returns:
            torch.Tensor: Image tensor in (C, H, W) format, float32, [-1, 1] range
        """
        try:
            tensor = torch.load(str(image_path), map_location="cpu")

            # Ensure float32 dtype
            if tensor.dtype != torch.float32:
                tensor = tensor.to(torch.float32)

            # Validate and normalize shape
            if tensor.ndim == 2:
                if tensor.shape != (self.image_size, self.image_size):
                    logger.warning(
                        f"Image shape {tensor.shape} doesn't match expected {(self.image_size, self.image_size)}"
                    )
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 3:
                if tensor.shape[0] in [1, 3]:
                    expected_shape = (self.channels, self.image_size, self.image_size)
                else:
                    expected_shape = (self.image_size, self.image_size, self.channels)

                if tensor.shape != expected_shape and tensor.shape[:2] != (
                    self.image_size,
                    self.image_size,
                ):
                    logger.warning(
                        f"Image shape {tensor.shape} doesn't match expected size {self.image_size}"
                    )

                if tensor.shape[-1] in [1, 3]:
                    tensor = tensor.permute(2, 0, 1)
            else:
                raise ValueError(f"Unexpected number of dimensions: {tensor.ndim}")

            # Verify data range
            min_val, max_val = tensor.min().item(), tensor.max().item()
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
        Get item as image for EDM compatibility.

        Returns:
            numpy.ndarray: Image data in (C, H, W) format, float32, [-1, 1] range
        """
        raw_idx = self._raw_idx[idx]

        # Check cache first
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image

        # Apply horizontal flip if needed
        if self._xflip[idx]:
            assert isinstance(image, np.ndarray)
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]

        return image.copy()

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

    def get_calibration_params(self, idx):
        """Get calibration parameters for a specific index (legacy method)."""
        return {}  # No longer using calibration parameters

    def close(self):
        """Clean up resources."""
        pass


def create_edm_pt_datasets(
    data_root: Union[str, Path],
    metadata_json: Union[str, Path],
    train_split: str = "train",
    val_split: str = "validation",
    max_files: Optional[int] = None,
    seed: int = 42,
    image_size: int = 256,
    channels: int = 3,
) -> Tuple[EDMPTDataset, EDMPTDataset]:
    """
    Create EDM-compatible training and validation datasets from .pt files.

    Args:
        data_root: Root directory containing .pt files
        metadata_json: Metadata JSON file with predefined splits
        train_split: Training split name
        val_split: Validation split name
        max_files: Maximum files per split (None for all)
        seed: Random seed
        image_size: Target image size
        channels: Number of channels (1 for grayscale, 3 for RGB)

    Returns:
        Tuple of (train_dataset, val_dataset) compatible with EDM's native interface
    """
    logger.info(f"Creating EDM-compatible PT datasets")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Channels: {channels}")
    logger.info(f"Data range: [-1, 1] (pre-normalized)")

    # Create training dataset
    train_dataset = EDMPTDataset(
        data_root=data_root,
        metadata_json=metadata_json,
        split=train_split,
        max_files=max_files,
        seed=seed,
        image_size=image_size,
        channels=channels,
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
    )

    logger.info(f"✅ Created EDM-compatible PT datasets (float32, no quantization):")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Resolution: {train_dataset.resolution}x{train_dataset.resolution}")
    logger.info(f"  Channels: {train_dataset.num_channels}")

    return train_dataset, val_dataset
