"""
EDM-compatible dataset loaders for 32-bit float PyTorch tensors and SIDD .MAT files.

Data is pre-normalized to [-1, 1] range during preprocessing.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from config.logging_config import get_logger
from core.demosaic import (
    demosaic_bayer_to_rgb,
    extract_camera_id_from_scene_name,
    get_cfa_pattern_from_scene_name,
)
from core.normalization import convert_range
from core.utils.data_utils import extract_scene_id_from_tile_id, extract_scene_id_padded
from core.utils.tensor_utils import _ensure_chw_format, ensure_tensor

logger = get_logger(__name__)


def _normalize_to_diffusion_range(image: np.ndarray) -> np.ndarray:
    """Normalize image from [0, 1] to [-1, 1] for diffusion model.

    Uses core.normalization.convert_range internally.
    """
    tensor = torch.from_numpy(image)
    normalized_tensor = convert_range(tensor, from_range="[0,1]", to_range="[-1,1]")
    return normalized_tensor.numpy()


def _normalize_split_name(split_name: str) -> str:
    """Normalize split name (val -> validation)."""
    return "validation" if split_name == "val" else split_name


def _extract_and_pad_scene_id(identifier: str, prefix: Optional[str] = None) -> str:
    """Extract scene ID from tile_id or filename and pad to 5 digits.

    Uses core.utils.data_utils.extract_scene_id_padded when possible, otherwise
    falls back to manual extraction.
    """
    # Try using core function first if it's a standard tile_id format
    scene_id = extract_scene_id_padded(identifier)
    if scene_id is not None:
        return scene_id

    # Fallback to manual extraction for non-standard formats
    if prefix and identifier.startswith(prefix):
        identifier = identifier.removeprefix(prefix)
    return identifier.split("_")[0].zfill(5)


def _ensure_float32(
    tensor_or_array: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Ensure tensor or array is float32 dtype.

    Note: For tensors, consider using ensure_tensor from core.utils.tensor_utils
    which provides more comprehensive functionality.
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return ensure_tensor(tensor_or_array, dtype=torch.float32)
    return (
        tensor_or_array
        if tensor_or_array.dtype == np.float32
        else tensor_or_array.astype(np.float32)
    )


def _ensure_chw_format_dataset(
    tensor_or_array: Union[torch.Tensor, np.ndarray],
    channels: int,
    image_size: int,
    is_tensor: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Ensure image is in (C, H, W) format.

    This is a dataset-specific wrapper around _ensure_chw_format from tensor_utils
    that handles both tensor and numpy array cases.
    """
    if isinstance(tensor_or_array, np.ndarray):
        # Convert numpy array to tensor for processing
        tensor = torch.from_numpy(tensor_or_array)
        tensor = _ensure_chw_format(tensor)
        if not is_tensor:
            return tensor.numpy()
        return tensor

    # For tensors, use the utility function directly
    return _ensure_chw_format(tensor_or_array)


class BaseEDMDataset:
    """Base class for EDM-compatible datasets with common label and property handling."""

    def __init__(
        self,
        split: str,
        image_size: int,
        channels: int,
        use_labels: bool = False,
        label_dim: int = 0,
        dataset_length: int = 0,
        dataset_name: str = "",
    ):
        self.split = split
        self.image_size = image_size
        self.channels = channels
        self._use_labels = use_labels
        self._label_dim = label_dim if label_dim > 0 else 0
        self._raw_labels = None
        self._raw_shape = [dataset_length, channels, image_size, image_size]
        self._raw_idx = np.arange(dataset_length, dtype=np.int64)
        self._name = dataset_name

    def _get_raw_labels(self):
        """Return raw labels array. For unconditional training, returns empty array."""
        if self._raw_labels is None:
            if self._use_labels:
                raise NotImplementedError(
                    "Label loading not yet implemented. Use use_labels=False for unconditional training."
                )
            self._raw_labels = np.zeros([len(self._raw_idx), 0], dtype=np.float32)
        return self._raw_labels

    def get_label(self, idx):
        """Return label for index. For unconditional training, returns empty array."""
        return self._get_raw_labels()[self._raw_idx[idx]].copy()

    @property
    def resolution(self):
        """Image resolution (assumes square images)."""
        return self._raw_shape[2]

    @property
    def num_channels(self):
        """Number of image channels."""
        return self._raw_shape[1]

    @property
    def label_dim(self):
        """Label dimension. 0 for unconditional training."""
        return self._label_dim

    @property
    def name(self):
        """Dataset name for EDM compatibility."""
        return self._name

    @property
    def image_shape(self):
        """Image shape as list [C, H, W] for EDM compatibility."""
        return list(self._raw_shape[1:])

    @property
    def label_shape(self):
        """Label shape for EDM compatibility."""
        return [self._label_dim]

    @property
    def has_labels(self):
        """Whether dataset has labels."""
        return self._label_dim > 0

    @property
    def has_onehot_labels(self):
        """Whether labels are one-hot encoded."""
        return False

    def get_details(self, idx):
        """Get detailed information about a dataset item (EDM compatibility)."""
        try:
            raw_idx = int(self._raw_idx[idx])
            return {
                "raw_idx": raw_idx,
                "xflip": False,
                "raw_label": self._get_raw_labels()[raw_idx].copy(),
            }
        except Exception:
            return {
                "raw_idx": idx,
                "xflip": False,
                "raw_label": np.array([], dtype=np.float32),
            }


class SimplePTDataset(BaseEDMDataset):
    """EDM-compatible dataset for 32-bit float .pt files."""

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
        self.data_root = Path(data_root)
        self.metadata_json = Path(metadata_json)
        self.max_files = max_files
        self._metadata_cache = None

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")
        if not self.metadata_json.exists():
            raise FileNotFoundError(
                f"Metadata JSON file not found: {self.metadata_json}"
            )

        self.image_files = self._find_paired_files()
        self.split_files = self._create_split()
        self._validate_training_data_purity()

        super().__init__(
            split=split,
            image_size=image_size,
            channels=channels,
            use_labels=use_labels,
            label_dim=label_dim,
            dataset_length=len(self.split_files),
            dataset_name=f"SimplePTDataset_{split}",
        )

    @property
    def _metadata(self) -> Dict[str, Any]:
        """Load and cache metadata JSON."""
        if self._metadata_cache is None:
            with open(self.metadata_json, "r") as f:
                self._metadata_cache = json.load(f)
        return self._metadata_cache

    def _extract_all_tiles_from_metadata(self) -> List[Dict[str, Any]]:
        """Extract all tiles from metadata, handling both flat and nested structures."""
        metadata = self._metadata
        if all_tiles := metadata.get("tiles", []):
            return all_tiles

        if "files" in metadata:
            return [
                tile
                for file_data in metadata["files"]
                if "tiles" in file_data
                for tile in file_data["tiles"]
            ]

        return []

    def _is_short_exposure_tile(self, tile: Dict[str, Any]) -> bool:
        """Check if a tile is short-exposure based on metadata."""
        if "data_type" in tile:
            return tile["data_type"] == "short"
        return not tile.get("is_clean", True)

    def _get_prefix_from_tile_id(self, tile_id: str) -> Optional[str]:
        """Extract camera prefix from tile_id."""
        if tile_id.startswith("fuji_"):
            return "fuji_"
        if tile_id.startswith("sony_"):
            return "sony_"
        return None

    def _find_pt_file_path(self, tile: Dict[str, Any]) -> Optional[Path]:
        """Find the .pt file path for a tile by trying multiple candidate locations."""
        if not (tile_id := tile.get("tile_id")):
            return None

        candidates = [
            self.data_root / f"{tile_id}.pt",
            self.data_root / "long" / f"{tile_id}.pt",
        ]

        if pt_path := tile.get("pt_path"):
            candidates.insert(0, Path(pt_path))

        return next((c for c in candidates if c.exists()), None)

    def _create_dummy_tensor(self) -> torch.Tensor:
        """Create a dummy tensor in the expected shape and dtype."""
        return torch.zeros(
            self.channels, self.image_size, self.image_size, dtype=torch.float32
        )

    def _detect_camera_type(self) -> str:
        """Detect camera type from metadata JSON file."""
        if "sensor" in (metadata := self._metadata):
            return metadata["sensor"].lower()

        if all_tiles := self._extract_all_tiles_from_metadata():
            if prefix := self._get_prefix_from_tile_id(all_tiles[0].get("tile_id", "")):
                return prefix.rstrip("_")

        return "fuji"

    def _load_split_files(self) -> dict:
        """Load the split files to determine which scenes belong to which split."""
        split_files = {"train": set(), "validation": set(), "test": set()}
        camera_type = self._detect_camera_type()
        split_dir = Path(
            "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/SID/splits"
        )

        for split_name in ["train", "val", "test"]:
            split_file = split_dir / f"{camera_type.capitalize()}_{split_name}_list.txt"
            if split_file.exists():
                with open(split_file, "r") as f:
                    for line in f:
                        if (stripped := line.strip()) and len(
                            parts := stripped.split("/")
                        ) >= 3:
                            split_files[_normalize_split_name(split_name)].add(
                                _extract_and_pad_scene_id(parts[-1])
                            )

        return split_files

    def _get_split_for_scene(self, scene_id: str, split_files: dict) -> str:
        """Determine which split a scene belongs to based on split files."""
        return next(
            (split for split, scenes in split_files.items() if scene_id in scenes),
            "train",
        )

    def _find_paired_files(self) -> List[Path]:
        """Find long-exposure .pt files using metadata JSON file and split files."""
        all_tiles = self._extract_all_tiles_from_metadata()
        if not all_tiles:
            raise FileNotFoundError(f"No tiles found in metadata {self.metadata_json}")

        split_files = self._load_split_files()
        tiles_by_split = {"train": [], "validation": [], "test": []}

        for tile in all_tiles:
            tile_id = tile.get("tile_id", "")
            if prefix := self._get_prefix_from_tile_id(tile_id):
                scene_id = _extract_and_pad_scene_id(tile_id, prefix)
                split = self._get_split_for_scene(scene_id, split_files)
                if split in tiles_by_split:
                    tiles_by_split[split].append(tile)

        if not tiles_by_split.get(self.split):
            raise FileNotFoundError(
                f"No tiles found for split '{self.split}' in metadata. "
                f"Available splits: {list(tiles_by_split.keys())}"
            )

        long_tiles = [
            tile
            for tile in tiles_by_split[self.split]
            if not self._is_short_exposure_tile(tile)
        ]
        long_files = [
            path
            for tile in long_tiles
            if (path := self._find_pt_file_path(tile)) is not None
        ]

        if not long_files:
            raise FileNotFoundError(
                f"No valid .pt files found for split '{self.split}'"
            )

        return long_files

    def _create_split(self) -> List[Path]:
        """Create split from image files with optional max_files limit."""
        return (
            self.image_files[: self.max_files] if self.max_files else self.image_files
        )

    def _validate_training_data_purity(self):
        """Assert that training set contains ONLY long-exposure (clean) images."""
        if self.split != "train":
            return

        try:
            all_tiles = self._extract_all_tiles_from_metadata()
            tiles_by_id = {tile.get("tile_id"): tile for tile in all_tiles}
        except Exception as e:
            logger.warning(f"Could not load metadata for validation: {e}")
            return

        contaminated = [
            (Path(f).stem, tiles_by_id[tile_id])
            for f in self.split_files[: min(20, len(self.split_files))]
            if (tile_id := Path(f).stem) in tiles_by_id
            and self._is_short_exposure_tile(tiles_by_id[tile_id])
        ]

        if contaminated:
            error_lines = [
                f"      - {tid}: data_type={t.get('data_type', 'long')}, is_clean={t.get('is_clean', True)}"
                for tid, t in contaminated
            ]
            error_msg = (
                f"\n❌ CRITICAL: CONTAMINATED TRAINING DATA DETECTED!\n"
                f"   Training MUST use only LONG-exposure (clean) images.\n"
                f"   Found SHORT-exposure tiles in training set:\n"
                + "\n".join(error_lines)
                + f"\n\n   This violates the core assumption:\n"
                f"   P(x_clean | y_noisy) = P(x_clean) * P(y_noisy | x_clean)\n"
                f"   Training must learn P(x_clean) on clean images only!\n"
            )
            raise AssertionError(error_msg)

    def _load_raw_image(self, raw_idx):
        """Load raw image in EDM format (float32, CHW)."""
        image = self._load_pt_item(raw_idx)["long"]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        return _ensure_chw_format_dataset(
            _ensure_float32(image), self.channels, self.image_size, is_tensor=False
        )

    def __len__(self):
        """Return dataset length."""
        return len(self.split_files)

    def _load_pt_item(self, idx: int) -> Dict[str, Any]:
        """Load and process a .pt file."""
        try:
            long_path = self.split_files[idx]
            long_image = self._load_pt_image_direct(long_path)
            metadata = {
                "long_path": str(long_path),
                "filename": long_path.name,
                "image_size": self.image_size,
                "channels": self.channels,
            }
            return {"long": long_image, "metadata": metadata}
        except Exception as e:
            logger.warning(f"Error loading .pt file {idx}: {e}")
            return {
                "long": self._create_dummy_tensor(),
                "metadata": {"corrupted": True, "original_idx": idx},
            }

    def _load_pt_image_direct(self, image_path: Path) -> torch.Tensor:
        """Load a .pt file directly as float32 tensor."""
        try:
            tensor = torch.load(str(image_path), map_location="cpu")
            tensor = _ensure_float32(tensor)
            tensor = _ensure_chw_format_dataset(
                tensor, self.channels, self.image_size, is_tensor=True
            )

            if tensor.ndim != 3 or tensor.shape[0] != self.channels:
                raise ValueError(
                    f"Invalid tensor shape {tensor.shape}, expected "
                    f"(C={self.channels}, H={self.image_size}, W={self.image_size})"
                )
            return tensor
        except Exception as e:
            logger.warning(f"Error loading .pt file {image_path}: {e}")
            return self._create_dummy_tensor()

    def __getitem__(self, idx):
        """Get item as (image, label) tuple for EDM compatibility."""
        raw_idx = self._raw_idx[idx]
        image = self._load_raw_image(raw_idx)
        return image.copy(), self.get_label(idx)


class SIDDRandomCropDataset(BaseEDMDataset):
    """Dataset for SIDD .MAT files with random non-overlapping 256x256 tiling."""

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        image_size: int = 256,
        channels: int = 3,
        use_labels: bool = False,
        label_dim: int = 0,
        tiles_per_image: Optional[int] = None,
        random_seed: Optional[int] = None,
        **kwargs,
    ):
        try:
            import config.config as config_module
        except ImportError:
            raise ImportError(
                "core.config not available. Install required dependencies or "
                "ensure core module is in Python path."
            )

        self.data_root = Path(data_root)
        self.tiles_per_image = tiles_per_image
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")

        self.image_files = self._find_gt_files()
        self.tiles = self._extract_random_tiles()

        super().__init__(
            split=split,
            image_size=image_size,
            channels=channels,
            use_labels=use_labels,
            label_dim=label_dim,
            dataset_length=len(self.tiles),
            dataset_name=f"SIDDTiled_{split}",
        )

    def _find_gt_files(self) -> List[Path]:
        """Find all GT_RAW .MAT files in SIDD structure, filtered by split."""
        split_name = self.split if self.split != "validation" else "val"
        possible_split_files = [
            self.data_root.parent / f"SIDD_{split_name}_list.txt",
            self.data_root.parent.parent / f"SIDD_{split_name}_list.txt",
            self.data_root / f"{split_name}_list.txt",
            Path(
                f"/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/SID/splits/SIDD_{split_name}_list.txt"
            ),
        ]

        split_file = next((c for c in possible_split_files if c.exists()), None)

        if split_file is None:
            valid_scenes = None
        else:
            with open(split_file, "r") as f:
                valid_scenes = set(line.strip() for line in f if line.strip())

        gt_files = []
        for scene_dir in sorted(self.data_root.iterdir()):
            if not scene_dir.is_dir():
                continue

            if valid_scenes is not None and scene_dir.name not in valid_scenes:
                continue

            gt_candidates = list(scene_dir.glob("GT_RAW_*.MAT"))
            if not gt_candidates:
                continue

            gt_files.append(gt_candidates[0])

        if not gt_files:
            raise FileNotFoundError(
                f"No GT_RAW_*.MAT files found in {self.data_root} for split '{self.split}'"
            )

        return gt_files

    def _load_mat_image(self, mat_path: Path) -> np.ndarray:
        """Load image from .MAT file and normalize to [0, 1] range."""
        from core.utils.file_utils import load_mat_file

        img = load_mat_file(mat_path, variable_name="x")
        if img is None:
            raise ValueError(f"Failed to load image from {mat_path}")

        img = np.squeeze(img).astype(np.float32)
        img_max = float(np.max(img))
        if img_max > 1.5:
            img = np.clip(img / img_max, 0.0, 1.0)
        elif img_max > 1.0:
            img = np.clip(img, 0.0, 1.0)
        return img

    def _generate_non_overlapping_positions(
        self, image_height: int, image_width: int
    ) -> List[Tuple[int, int]]:
        """Generate all possible non-overlapping tile positions for an image."""
        positions = []
        max_tiles_h = image_height // self.image_size
        max_tiles_w = image_width // self.image_size

        for row in range(max_tiles_h):
            y_start = row * self.image_size
            for col in range(max_tiles_w):
                x_start = col * self.image_size
                positions.append((y_start, x_start))

        return positions

    def _extract_random_tiles(self) -> List[Dict[str, Any]]:
        """Extract random non-overlapping tiles from all images."""
        all_tiles = []

        for mat_path in self.image_files:
            scene_name = mat_path.parent.name
            camera_id = extract_camera_id_from_scene_name(scene_name)

            image = self._load_mat_image(mat_path)

            cfa_pattern_str = (
                get_cfa_pattern_from_scene_name(scene_name, return_string=True)
                or "rggb"
            )

            if image.ndim == 2:
                image = demosaic_bayer_to_rgb(
                    image,
                    cfa_pattern=cfa_pattern_str,
                    output_channel_order="RGB",
                    alg_type="VNG",
                )

            if image.ndim == 3 and image.shape[2] == 3:
                image = np.transpose(image, (2, 0, 1))
            elif image.ndim == 2:
                image = np.expand_dims(image, axis=0)

            if image.ndim != 3 or image.shape[0] != self.channels:
                continue

            H, W = image.shape[1], image.shape[2]

            if H < self.image_size or W < self.image_size:
                continue

            all_positions = self._generate_non_overlapping_positions(H, W)
            if not all_positions:
                continue

            num_tiles = len(all_positions)
            if self.tiles_per_image is not None:
                num_tiles = min(num_tiles, self.tiles_per_image)

            selected_positions = random.sample(all_positions, num_tiles)

            for tile_idx, (y_start, x_start) in enumerate(selected_positions):
                y_end = y_start + self.image_size
                x_end = x_start + self.image_size

                tile_data = image[:, y_start:y_end, x_start:x_end].copy()

                if tile_data.shape != (self.channels, self.image_size, self.image_size):
                    continue

                tile_data = _normalize_to_diffusion_range(tile_data)

                all_tiles.append(
                    {
                        "tile_data": tile_data.astype(np.float32),
                        "file_path": mat_path,
                        "scene_name": scene_name,
                        "camera_id": camera_id,
                        "tile_idx": tile_idx,
                        "position": (y_start, x_start),
                    }
                )

        return all_tiles

    def __len__(self):
        """Return total number of tiles."""
        return len(self.tiles)

    def __getitem__(self, idx):
        """Get tile at index."""
        raw_idx = self._raw_idx[idx]
        tile = self.tiles[raw_idx]
        image = tile["tile_data"].copy()
        label = self.get_label(idx)
        return image, label


class TileDataset(Dataset):
    """Simple PyTorch Dataset that wraps a list of tile dictionaries.

    This dataset is used for loading tile metadata dictionaries in DataLoader.
    Each item is a dictionary containing tile information (e.g., tile_id, metadata).
    """

    def __init__(self, tiles: List[Dict[str, Any]]):
        """Initialize TileDataset with a list of tile dictionaries.

        Args:
            tiles: List of dictionaries, each containing tile information.
        """
        self.tiles = tiles

    def __len__(self) -> int:
        """Return the number of tiles in the dataset."""
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a tile dictionary by index.

        Args:
            idx: Index of the tile to retrieve.

        Returns:
            Dictionary containing tile information.
        """
        return self.tiles[idx]
