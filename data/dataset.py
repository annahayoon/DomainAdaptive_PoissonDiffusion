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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from core.logging_config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    # Forward reference for type hints
    pass


class SimplePTDataset:
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
        image_size: int = 256,
        channels: int = 3,
        use_labels: bool = False,
        label_dim: int = 0,
        **kwargs,
    ):
        """
        Initialize EDM-compatible PT dataset.

        Args:
            data_root: Root directory containing .pt files
            metadata_json: Metadata JSON file with predefined splits
            split: Data split (train, validation, test)
            max_files: Maximum number of files to load
            image_size: Target image size (will validate shape matches)
            channels: Number of channels (1 for grayscale, 3 for RGB)
            use_labels: Enable conditioning labels? False = label dimension is zero.
            label_dim: Number of class labels, 0 = unconditional.
            **kwargs: Additional arguments (ignored for compatibility)
        """

        # Initialize dataset parameters
        self.data_root = Path(data_root)
        self.metadata_json = Path(metadata_json)
        self.split = split
        self.max_files = max_files
        self.image_size = image_size
        self.channels = channels
        self._use_labels = use_labels
        self._label_dim = label_dim if label_dim > 0 else 0
        self._raw_labels = None

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")

        if not self.metadata_json.exists():
            raise FileNotFoundError(
                f"Metadata JSON file not found: {self.metadata_json}"
            )

        # Find .pt files
        self.image_files = self._find_paired_files()

        # Create train/val/test split
        self.split_files = self._create_split()

        # Validate training data purity (CRITICAL for physics-informed approach)
        self._validate_training_data_purity()

        # Set up raw shape for EDM compatibility (N, C, H, W)
        self._raw_shape = [len(self.split_files), channels, image_size, image_size]

        # Set up basic properties
        self._raw_idx = np.arange(len(self.split_files), dtype=np.int64)

        logger.info(
            f"SimplePTDataset ready: {len(self.split_files)} float32 .pt tiles for '{split}' split"
        )
        logger.info(f"  Shape: [{channels}, {image_size}, {image_size}]")
        logger.info(f"  Data range: [-1, 1] (pre-normalized during preprocessing)")
        logger.info(
            f"  Label dimension: {self._label_dim} ({'conditional' if self._use_labels else 'unconditional'})"
        )

    def _find_paired_files(self) -> List[Path]:
        """Find long-exposure .pt files using metadata JSON file."""
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

        # FILTER OUT SHORT-EXPOSURE TILES FOR PRIOR TRAINING (only use long-exposure)
        long_tiles_only = []
        short_tiles_filtered = []

        for tile in split_tiles:
            tile_id = tile.get("tile_id", "")

            # Check if this is a short-exposure tile using metadata
            is_short = False

            # Check data_type field (required)
            if "data_type" in tile:
                data_type = tile.get("data_type", "long")
                is_short = data_type == "short"
            else:
                # Require explicit data_type field
                logger.warning(
                    f"Tile {tile_id} missing 'data_type' field. Assuming long-exposure."
                )

            if is_short:
                short_tiles_filtered.append(tile_id)
            else:
                long_tiles_only.append(tile)

        logger.info(
            f"Split '{self.split}' - Total tiles in metadata: {len(split_tiles)}"
        )
        logger.info(
            f"  └─ Long-exposure tiles (prior training): {len(long_tiles_only)}"
        )
        logger.info(
            f"  └─ Short-exposure tiles (filtered out): {len(short_tiles_filtered)}"
        )

        split_tiles = long_tiles_only  # Use only long-exposure tiles

        # Extract long-exposure file paths
        long_files = []

        for tile in split_tiles:
            tile_id = tile.get("tile_id")
            if not tile_id:
                continue

            # Look for .pt file
            # Try multiple possible paths
            pt_path_candidates = [
                # From metadata (if it stores pt_path)
                Path(tile.get("pt_path", "")),
                # Construct from data_root
                self.data_root / f"{tile_id}.pt",
                # Try long subdirectory
                self.data_root / "long" / f"{tile_id}.pt",
            ]

            long_path = None
            for candidate in pt_path_candidates:
                if candidate.exists():
                    long_path = candidate
                    break

            if long_path is None:
                logger.debug(f"Long-exposure .pt file not found for: {tile_id}")
                continue

            long_files.append(long_path)

        if not long_files:
            raise FileNotFoundError(
                f"No valid .pt files found for split '{self.split}'"
            )

        logger.info(
            f"✓ Loaded {len(long_files)} LONG-EXPOSURE .pt tiles (float32) for '{self.split}' split"
        )
        return long_files

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

    def _validate_training_data_purity(self):
        """
        ASSERT that training set contains ONLY long-exposure (clean) images.

        This is critical for the physics-informed approach:
        - Training learns P(x_clean) unconditionally
        - Short-exposure (noisy) images are for inference/validation only

        Raises:
            AssertionError: If contaminated training data is detected
        """
        if self.split != "train":
            # Only validate training split
            return

        # Load metadata to cross-check data_type field
        try:
            with open(self.metadata_json, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata for validation: {e}")
            return

        tiles = metadata.get("tiles", [])
        tiles_by_id = {tile.get("tile_id"): tile for tile in tiles}

        # Sample check first min(20, len(split_files)) files
        num_to_check = min(20, len(self.split_files))
        contaminated_files = []

        for file_path in self.split_files[:num_to_check]:
            tile_id = Path(file_path).stem
            matching_tile = tiles_by_id.get(tile_id)

            if matching_tile:
                data_type = matching_tile.get("data_type", "long")
                is_clean = matching_tile.get("is_clean", True)

                # Check via explicit fields or fallback
                is_short = (data_type == "short") or (not is_clean)

                if is_short:
                    contaminated_files.append((tile_id, data_type, is_clean))

        if contaminated_files:
            error_msg = (
                f"\n❌ CRITICAL: CONTAMINATED TRAINING DATA DETECTED!\n"
                f"   Training MUST use only LONG-exposure (clean) images.\n"
                f"   Found SHORT-exposure tiles in training set:\n"
            )
            for tile_id, data_type, is_clean in contaminated_files:
                error_msg += (
                    f"      - {tile_id}: data_type={data_type}, is_clean={is_clean}\n"
                )
            error_msg += (
                f"\n   This violates the core assumption:\n"
                f"   P(x_clean | y_noisy) = P(x_clean) * P(y_noisy | x_clean)\n"
                f"   Training must learn P(x_clean) on clean images only!\n"
            )
            raise AssertionError(error_msg)

        logger.info(
            f"✅ Training data purity validated: {len(self.split_files)} LONG-exposure (clean) images"
        )
        logger.info(
            f"   Sample validation: checked first {num_to_check} files - all clean ✓"
        )

    def _load_raw_image(self, raw_idx):
        """
        Load raw image in EDM format (float32, CHW).

        Returns:
            numpy.ndarray: Image in (C, H, W) format, float32, [-1, 1] range
        """
        pt_item = self._load_pt_item(raw_idx)
        long_image = pt_item["long"]

        # Convert to numpy
        image_np = (
            long_image.numpy() if isinstance(long_image, torch.Tensor) else long_image
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
                - 'long': torch.Tensor of shape (C, H, W), float32, [-1, 1] range
                - 'metadata': dict with file information
        """
        try:
            long_path = self.split_files[idx]

            # Load .pt file directly as float32 (already in [-1,1] range)
            long_image = self._load_pt_image_direct(long_path)

            # Extract metadata
            metadata = {
                "long_path": str(long_path),
                "filename": long_path.name,
                "image_size": self.image_size,
                "channels": self.channels,
            }

            return {
                "long": long_image,
                "metadata": metadata,
            }

        except Exception as e:
            logger.warning(f"Error loading .pt file {idx}: {e}")
            # Return dummy data to prevent training crashes
            return {
                "long": torch.zeros(
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
        Get item as (image, label) tuple for EDM compatibility.

        Returns:
            tuple: (image, label) where:
                - image: numpy.ndarray in (C, H, W) format, float32, [-1, 1] range
                - label: numpy.ndarray label (empty array for unconditional training)
        """
        raw_idx = self._raw_idx[idx]

        # Load image
        image = self._load_raw_image(raw_idx)

        return image.copy(), self.get_label(idx)

    @property
    def resolution(self):
        """Image resolution (assumes square images)."""
        return self._raw_shape[2]

    @property
    def num_channels(self):
        """Number of image channels."""
        return self._raw_shape[1]

    def _get_raw_labels(self):
        """Return raw labels array. For unconditional training, returns empty array."""
        if self._raw_labels is None:
            if self._use_labels:
                # Future: Load actual labels if needed for conditional training
                raise NotImplementedError(
                    "Label loading not yet implemented. Use use_labels=False for unconditional training."
                )
            else:
                # Unconditional: return empty labels array
                self._raw_labels = np.zeros(
                    [len(self.split_files), 0], dtype=np.float32
                )
        return self._raw_labels

    def get_label(self, idx):
        """Return label for index idx. For unconditional training, returns empty array."""
        raw_idx = self._raw_idx[idx]
        label = self._get_raw_labels()[raw_idx]
        return label.copy()

    @property
    def label_dim(self):
        """Label dimension. 0 for unconditional training."""
        return self._label_dim
