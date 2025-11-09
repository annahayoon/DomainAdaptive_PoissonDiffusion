"""
Data loading utility functions for the Poisson-Gaussian Diffusion project.

This module provides functions for loading images, metadata, tiles, and
managing data structures.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from core.normalization import convert_range
from core.utils.file_utils import _load_json_file
from core.utils.sensor_utils import (
    compute_sensor_range,
    denormalize_to_physical,
    get_black_level_white_level_from_metadata,
    get_sensor_calibration_params,
)
from core.utils.tensor_utils import _clamp_to_01, _ensure_chw_format, _validate_channels

logger = logging.getLogger(__name__)


def _get_sample_config():
    """Lazy import for config constants."""
    try:
        from config.config import SENSOR_NAME_MAPPING, SUPPORTED_SENSORS
        from config.sample_config import RESERVED_TILE_KEYS

        return RESERVED_TILE_KEYS, SENSOR_NAME_MAPPING, SUPPORTED_SENSORS
    except ImportError:
        # Fallback if imports not available
        try:
            from config.sample_config import (
                RESERVED_TILE_KEYS,
                SENSOR_NAME_MAPPING,
                SUPPORTED_SENSORS,
            )

            return RESERVED_TILE_KEYS, SENSOR_NAME_MAPPING, SUPPORTED_SENSORS
        except ImportError:
            return set(), {}, []


def load_tensor(
    file_path: Path,
    device: Optional[torch.device] = None,
    map_location: Optional[str] = None,
    weights_only: bool = False,
) -> Any:
    """Load tensor from file (supports both pickle and torch formats)."""
    import pickle  # nosec B403
    import sys

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if map_location is None:
        map_location = str(device) if device is not None else "cpu"

    # .pt and .pth files are PyTorch formats - use torch.load directly
    file_ext = file_path.suffix.lower()
    if file_ext in (".pt", ".pth"):
        try:
            checkpoint = torch.load(
                str(file_path), map_location=map_location, weights_only=weights_only
            )
            if device is not None and hasattr(checkpoint, "to"):
                checkpoint = checkpoint.to(device)
            return checkpoint
        except Exception as torch_e:
            raise RuntimeError(
                f"Failed to load PyTorch file {file_path}: {torch_e}"
            ) from torch_e

    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)  # nosec B301
    except Exception as pickle_e:
        if isinstance(pickle_e, (ModuleNotFoundError, AttributeError)) and (
            "numpy._core" in str(pickle_e) or "scalar" in str(pickle_e)
        ):
            raise RuntimeError(
                f"Model requires numpy 2.0+ (Python 3.10+) but environment has "
                f"Python {sys.version.split()[0]} with numpy {np.__version__}. "
                f"Try running with: /usr/bin/python3.10 or /usr/bin/python3.11"
            ) from pickle_e
        if file_ext not in (".pt", ".pth"):
            logger.warning(f"pickle.load failed, trying torch.load: {pickle_e}")
        try:
            checkpoint = torch.load(
                str(file_path), map_location=map_location, weights_only=weights_only
            )
            if device is not None and hasattr(checkpoint, "to"):
                checkpoint = checkpoint.to(device)
            return checkpoint
        except Exception as torch_e:
            raise RuntimeError(
                f"Failed to load {file_path}: pickle={pickle_e}, torch={torch_e}"
            ) from torch_e


def _extract_tensor_from_dict(
    tensor: Any, file_path: Path, preferred_keys: Optional[List[str]] = None
) -> torch.Tensor:
    """Extract tensor from dictionary structure."""
    if not isinstance(tensor, dict):
        return tensor

    keys_to_try = preferred_keys or ["short", "long", "image"]
    for key in keys_to_try:
        if key in tensor:
            return tensor[key]

    raise ValueError(f"Unrecognized dict structure in {file_path}")


def _construct_tile_path(tile_id: str, base_dir: Path) -> Path:
    """Construct path to tile file."""
    return base_dir / f"{tile_id}.pt"


def _create_tile_lookup(tiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create lookup dictionary from tile list."""
    return {tile.get("tile_id"): tile for tile in tiles if tile.get("tile_id")}


def _get_tile_metadata_dict(
    tile_id: str, tile_lookup: Dict[str, Dict[str, Any]], required: bool = True
) -> Optional[Dict[str, Any]]:
    """Get tile metadata dictionary."""
    tile_meta = tile_lookup.get(tile_id)
    if not tile_meta:
        if required:
            raise ValueError(f"Tile {tile_id} not found in metadata")
        return None
    return tile_meta


def _get_tile_metadata(
    tile_id: str,
    tile_lookup: Dict[str, Dict[str, Any]],
    field_name: str,
    default: Optional[Any] = None,
) -> Any:
    """Get specific field from tile metadata."""
    tile_meta = _get_tile_metadata_dict(tile_id, tile_lookup, required=True)
    value = tile_meta.get(field_name, default)
    if value is None and default is None:
        raise ValueError(f"No {field_name} in metadata for tile {tile_id}")
    return value


def _extract_tiles_from_metadata(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tiles list from metadata structure."""
    tiles = metadata.get("tiles", [])
    if not tiles:
        tiles = [
            tile
            for file_info in metadata.get("files", [])
            for tile in file_info.get("tiles", [])
        ]
    return tiles


def load_metadata_json(metadata_json: Path) -> Dict[str, Dict[str, Any]]:
    """Load metadata JSON and return tile lookup dictionary."""
    metadata = _load_json_file(metadata_json)
    tiles = _extract_tiles_from_metadata(metadata)
    return _create_tile_lookup(tiles)


def load_test_tiles(
    metadata_json: Path,
    split: str = "test",
    sensor_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load test tiles from metadata JSON."""
    metadata = _load_json_file(metadata_json)
    tiles = _extract_tiles_from_metadata(metadata)
    filtered_tiles = [tile for tile in tiles if tile.get("split") == split]

    if sensor_filter:
        sensor_lower = sensor_filter.lower()
        filtered_tiles = [
            tile
            for tile in filtered_tiles
            if sensor_lower in tile.get("sensor_type", "").lower()
        ]

    return filtered_tiles


def get_sensor_from_metadata(
    tile_id: str, tile_lookup: Dict[str, Dict[str, Any]]
) -> str:
    """Get sensor type from tile metadata."""
    _, _, SUPPORTED_SENSORS = _get_sample_config()
    sensor_type = _get_tile_metadata(tile_id, tile_lookup, "sensor_type")
    if sensor_type in SUPPORTED_SENSORS:
        return sensor_type
    raise ValueError(
        f"Unknown sensor_type '{sensor_type}' in metadata for tile {tile_id}"
    )


def get_exposure_time_from_metadata(
    tile_id: str, tile_lookup: Dict[str, Dict[str, Any]]
) -> float:
    """Get exposure time from tile metadata."""
    exposure_time = _get_tile_metadata(tile_id, tile_lookup, "exposure_time")
    if isinstance(exposure_time, str):
        exposure_time = exposure_time.rstrip("s").strip()
    try:
        return float(exposure_time)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid exposure_time '{exposure_time}' in metadata for tile {tile_id}: {e}"
        )


def find_long_tile_pair(
    short_tile_id: str, tile_lookup: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Find matching long exposure tile for a short exposure tile."""
    short_tile_meta = _get_tile_metadata_dict(
        short_tile_id, tile_lookup, required=False
    )
    if not short_tile_meta:
        logger.error(f"Short tile not found in metadata: {short_tile_id}")
        return None

    required_fields = ["scene_id", "sensor_type", "grid_x", "grid_y"]
    match_criteria = {field: short_tile_meta.get(field) for field in required_fields}

    missing_fields = [
        k
        for k, v in match_criteria.items()
        if v is None or (k in ["scene_id", "sensor_type"] and not v)
    ]
    if missing_fields:
        logger.error(
            f"Incomplete metadata for short tile {short_tile_id}: missing {missing_fields}"
        )
        return None

    long_candidates = [
        tile_meta
        for tile_id, tile_meta in tile_lookup.items()
        if (
            tile_meta.get("data_type") == "long"
            and all(
                tile_meta.get(field) == match_criteria[field]
                for field in required_fields
            )
        )
    ]

    if not long_candidates:
        logger.warning(f"No long exposure tile candidates found for {short_tile_id}")
        return None

    if len(long_candidates) > 1:
        logger.warning(
            f"Unexpected: Found {len(long_candidates)} long exposure candidates for "
            f"short tile {short_tile_id} (scene_id={match_criteria['scene_id']}, "
            f"grid=({match_criteria['grid_x']}, {match_criteria['grid_y']})). "
            f"Expected exactly one. Using first candidate."
        )

    return long_candidates[0]


def select_tiles(
    tile_ids: Optional[List[str]],
    available_tiles: List[Dict[str, Any]],
    num_examples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Select tiles either by ID or randomly."""
    if tile_ids is not None:
        available_lookup = _create_tile_lookup(available_tiles)
        return [
            available_lookup[tile_id]
            for tile_id in tile_ids
            if tile_id in available_lookup
        ]
    else:
        rng = np.random.RandomState(seed)
        selected_indices = rng.choice(
            len(available_tiles),
            size=min(num_examples, len(available_tiles)),
            replace=False,
        )
        return [available_tiles[i] for i in selected_indices]


def load_image(
    image_path: Path,
    device: torch.device,
    image_type: str = "image",
    target_channels: Optional[int] = None,
    add_batch_dim: bool = False,
    preferred_keys: Optional[List[str]] = None,
    return_metadata: bool = True,
    return_raw: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Load image tensor from file."""
    raw_tensor = load_tensor(
        image_path, device=device, map_location=str(device), weights_only=False
    )
    tensor = _extract_tensor_from_dict(raw_tensor, image_path, preferred_keys)
    tensor = tensor.float()
    tensor = _ensure_chw_format(tensor)

    tile_id = image_path.stem
    if target_channels is not None:
        tensor = _validate_channels(tensor, target_channels, image_type, tile_id)

    if add_batch_dim and tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    processed_tensor = tensor.to(torch.float32)
    result = [processed_tensor]

    if return_metadata:
        tensor_for_meta = (
            processed_tensor.squeeze(0) if add_batch_dim else processed_tensor
        )
        tensor_min = tensor_for_meta.min().item()
        tensor_max = tensor_for_meta.max().item()
        result.append(
            {
                "offset": 0.0,
                "original_range": [tensor_min, tensor_max],
                "processed_range": [tensor_min, tensor_max],
            }
        )

    if return_raw:
        result.append(raw_tensor)

    return tuple(result)


def load_image_from_tile_id(
    tile_id: str,
    image_dir: Path,
    device: torch.device,
    image_type: str = "image",
    target_channels: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load image from tile ID."""
    image_path = _construct_tile_path(tile_id, image_dir)
    return load_image(
        image_path,
        device,
        image_type,
        target_channels,
        add_batch_dim=False,
        return_metadata=True,
        return_raw=False,
    )


def load_short_image(
    tile_id: str,
    short_dir: Path,
    device: torch.device,
    target_channels: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load short exposure image."""
    return load_image_from_tile_id(tile_id, short_dir, device, "short", target_channels)


def load_long_image(
    tile_id: str,
    long_dir: Path,
    tile_lookup: Dict[str, Dict[str, Any]],
    sampler_device: torch.device,
    target_channels: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[Any]]:
    """Load long exposure image."""
    long_pair_info = find_long_tile_pair(tile_id, tile_lookup)
    if not long_pair_info:
        logger.warning(f"No long exposure tile pair found for {tile_id}")
        return None, None

    long_tile_id = long_pair_info.get("tile_id")
    long_path = _construct_tile_path(long_tile_id, long_dir)
    if not long_path.exists():
        logger.warning(f"Long exposure tile file not found: {long_path}")
        return None, None

    try:
        long_image, long_image_raw = load_image(
            long_path,
            sampler_device,
            "long",
            target_channels,
            add_batch_dim=True,
            preferred_keys=["long", "image"],
            return_metadata=False,
            return_raw=True,
        )
        return long_image, long_image_raw
    except Exception as e:
        logger.warning(f"Failed to load long exposure image: {e}")
        return None, None


def analyze_image_brightness(image: torch.Tensor) -> Dict[str, float]:
    """Analyze image brightness statistics."""
    img_01 = convert_range(image, "[-1,1]", "[0,1]")
    mean_brightness = img_01.mean().item()
    img_flat = img_01.flatten()

    brightness_thresholds = [
        (0.2, "Very Dark"),
        (0.4, "Dark"),
        (0.6, "Medium"),
        (0.8, "Bright"),
    ]
    brightness_category = next(
        (cat for thresh, cat in brightness_thresholds if mean_brightness < thresh),
        "Very Bright",
    )

    return {
        "mean": mean_brightness,
        "median": float(img_flat.median().item()),
        "std": float(img_flat.std().item()),
        "min": float(img_flat.min().item()),
        "max": float(img_flat.max().item()),
        "category": brightness_category,
    }


def get_exposure_ratio(tile_id: str, tile_lookup: Dict[str, Dict[str, Any]]) -> float:
    """Get exposure ratio from tile metadata."""
    short_exposure = get_exposure_time_from_metadata(tile_id, tile_lookup)
    long_pair_info = find_long_tile_pair(tile_id, tile_lookup)
    if not long_pair_info:
        raise ValueError(f"No long exposure pair found for {tile_id}")
    long_exposure = get_exposure_time_from_metadata(
        long_pair_info["tile_id"], tile_lookup
    )
    return long_exposure / short_exposure


def extract_scene_id_from_tile_id(tile_id: str) -> Optional[str]:
    """Extract scene ID from tile ID."""
    parts = tile_id.split("_")
    return parts[0] if len(parts) > 0 else None


def extract_scene_id_padded(tile_id: str) -> Optional[str]:
    """Extract scene ID from tile_id (padded to 5 digits).

    Args:
        tile_id: Tile ID string (e.g., "10003_00_0.04s")

    Returns:
        Scene ID padded to 5 digits (e.g., "10003"), or None if extraction fails
    """
    scene_id = extract_scene_id_from_tile_id(tile_id)
    return scene_id.zfill(5) if scene_id else None


def extract_tile_coordinates(
    tile: Dict[str, Any]
) -> Optional[Tuple[int, int, int, int]]:
    """Extract tile coordinates from tile dict.

    Args:
        tile: Dictionary containing tile metadata

    Returns:
        Tuple of (grid_x, grid_y, image_x, image_y) or None if any coordinate is missing
    """
    coords = (
        tile.get("grid_x"),
        tile.get("grid_y"),
        tile.get("image_x"),
        tile.get("image_y"),
    )
    return coords if all(c is not None for c in coords) else None


def detect_sensor_type_from_tile_id(tile_id: str) -> Optional[str]:
    """Detect sensor type from tile_id prefix.

    Args:
        tile_id: Tile ID string

    Returns:
        Sensor type ("sony" or "fuji") or None if not detected
    """
    if tile_id.startswith("sony_"):
        return "sony"
    elif tile_id.startswith("fuji_"):
        return "fuji"
    return None


def load_test_tiles_from_metadata_with_scenes(
    metadata_json: Path, test_scenes: Set[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load test tiles from metadata JSON file, including RAW file paths.

    Args:
        metadata_json: Path to metadata JSON file
        test_scenes: Set of test scene IDs (padded to 5 digits)

    Returns:
        Tuple of (long_tiles, short_tiles) where each is a list of tile info dictionaries
    """
    logger.info(f"Loading metadata from {metadata_json}")

    from core.utils.file_utils import _load_json_file

    metadata = _load_json_file(metadata_json)

    long_tiles = []
    short_tiles = []

    if "files" not in metadata:
        return long_tiles, short_tiles

    for file_data in metadata["files"]:
        file_metadata = file_data.get("file_metadata", {})
        raw_file_path = file_metadata.get("file_path", "")

        if not raw_file_path or not Path(raw_file_path).exists():
            continue

        for tile in file_data.get("tiles", []):
            tile_id = tile.get("tile_id", "")
            scene_id = extract_scene_id_padded(tile_id)

            if not scene_id or scene_id not in test_scenes:
                continue

            sensor_type = tile.get("sensor_type") or detect_sensor_type_from_tile_id(
                tile_id
            )
            if not sensor_type:
                continue

            coords = extract_tile_coordinates(tile)
            if not coords:
                continue

            tile_info = {
                "tile_id": tile_id,
                "raw_file_path": raw_file_path,
                "data_type": tile.get("data_type", "long"),
                "scene_id": scene_id,
                "sensor_type": sensor_type,
                "grid_x": coords[0],
                "grid_y": coords[1],
                "image_x": coords[2],
                "image_y": coords[3],
            }

            if tile_info["data_type"] == "long":
                long_tiles.append(tile_info)
            elif tile_info["data_type"] == "short":
                short_tiles.append(tile_info)

    logger.info(
        f"Found {len(long_tiles)} long tiles and {len(short_tiles)} short tiles for test set"
    )
    return long_tiles, short_tiles


def extract_frame_id_from_tile_id(tile_id: str) -> Optional[str]:
    """Extract frame ID from tile ID."""
    parts = tile_id.split("_")
    return parts[1] if len(parts) > 1 else None


def get_scene_exposure_key(scene_id: str, exposure_time: float) -> str:
    """Create scene-exposure key."""
    return f"{scene_id}_{exposure_time:.4f}s"


def get_scene_exposure_frame_key(
    scene_id: str, exposure_time: float, frame_id: str
) -> str:
    """Create scene-exposure-frame key."""
    return f"{scene_id}_{frame_id}_{exposure_time:.4f}s"


def apply_exposure_scaling(
    short_image: torch.Tensor, exposure_ratio: float
) -> torch.Tensor:
    """Apply exposure scaling to short image."""
    return _clamp_to_01(short_image * exposure_ratio)


def load_tile_image_data(
    tile_id: str,
    short_dir: Path,
    device: torch.device,
    img_channels: int,
    tile_lookup: Dict[str, Dict[str, Any]],
    long_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load all image data for a tile."""
    short_image, short_metadata = load_short_image(
        tile_id, short_dir, device, target_channels=img_channels
    )
    if short_image.ndim == 3:
        short_image = short_image.unsqueeze(0)
    short_image = short_image.to(torch.float32)

    extracted_sensor = get_sensor_from_metadata(tile_id, tile_lookup)
    black_level, white_level = get_black_level_white_level_from_metadata(
        tile_id, tile_lookup
    )
    s_sensor = compute_sensor_range(black_level, white_level)
    short_phys = denormalize_to_physical(short_image, black_level, white_level)

    long_image = None
    if long_dir is not None:
        try:
            long_image, _ = load_long_image(
                tile_id=tile_id,
                long_dir=long_dir,
                tile_lookup=tile_lookup,
                sampler_device=device,
                target_channels=img_channels,
            )
        except Exception as e:
            logger.warning(f"Could not load long image for {tile_id}: {e}")

    return {
        "short_image": short_image,
        "short_metadata": short_metadata,
        "long_image": long_image,
        "short_phys": short_phys,
        "black_level": black_level,
        "white_level": white_level,
        "s_sensor": s_sensor,
        "extracted_sensor": extracted_sensor,
        "exposure_ratio": get_exposure_ratio(tile_id, tile_lookup),
    }


def load_tile_and_metadata(
    tile_id: str,
    tile_info: Dict[str, Any],
    tile_lookup: Dict[str, Dict[str, Any]],
    short_dir: str,
    long_dir: Optional[str],
    device: torch.device,
    img_channels: int,
    sensor_ranges: Dict[str, Dict[str, float]],
    use_sensor_calibration: bool,
    sensor_name: Optional[str],
    conservative_factor: float,
    sigma_r: float,
) -> Optional[Dict[str, Any]]:
    """Load tile with all metadata and calibration."""
    grid_x = tile_info.get("grid_x")
    grid_y = tile_info.get("grid_y")
    tile_meta = tile_lookup.get(tile_id, {})

    if grid_x is None or grid_y is None:
        grid_x = grid_x or tile_meta.get("grid_x")
        grid_y = grid_y or tile_meta.get("grid_y")

    if grid_x is None or grid_y is None:
        logger.warning(f"Missing grid coordinates for {tile_id}, skipping")
        return None

    image_x = tile_info.get("image_x") or tile_meta.get("image_x")
    image_y = tile_info.get("image_y") or tile_meta.get("image_y")

    try:
        image_data = load_tile_image_data(
            tile_id=tile_id,
            short_dir=Path(short_dir),
            device=device,
            img_channels=img_channels,
            tile_lookup=tile_lookup,
            long_dir=Path(long_dir) if long_dir else None,
        )

        if use_sensor_calibration:
            estimated_sigma, noise_estimates, _, _ = get_sensor_calibration_params(
                sensor_name,
                image_data["extracted_sensor"],
                image_data["short_phys"],
                sensor_ranges,
                conservative_factor,
            )
        else:
            estimated_sigma = sigma_r
            noise_estimates = None

        return {
            "tile_id": tile_id,
            "tile_info": tile_info,
            **image_data,
            "estimated_sigma": estimated_sigma,
            "noise_estimates": noise_estimates,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "image_x": image_x,
            "image_y": image_y,
        }
    except Exception as e:
        logger.error(f"Failed to load {tile_id}: {e}")
        return None


def select_tiles_for_processing(
    tile_ids: Optional[List[str]],
    metadata_json: Path,
    tile_lookup: Dict[str, Dict[str, Any]],
    num_examples: int,
    seed: int,
    sensor_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Select tiles for processing."""
    if tile_ids:
        selected_tiles = [
            tile_info
            for tile_id in tile_ids
            if (tile_info := tile_lookup.get(tile_id)) is not None
        ]
        missing = len(tile_ids) - len(selected_tiles)
        if missing > 0:
            logger.warning(f"{missing} tile(s) not found in metadata")

        if not selected_tiles:
            raise ValueError("None of the provided tile_ids were found in metadata")

        return selected_tiles

    test_tiles = load_test_tiles(
        metadata_json, split="test", sensor_filter=sensor_filter
    )
    if not test_tiles:
        raise ValueError("No test tiles found")

    short_tiles = [tile for tile in test_tiles if tile.get("data_type") == "short"]
    if not short_tiles:
        raise ValueError("No short exposure tiles found")

    return select_tiles(None, short_tiles, num_examples, seed)


def load_noise_calibration(
    sensor_name: str,
    calibration_dir: Optional[Path],
    data_root: Optional[Path] = None,
) -> Optional[Dict[str, float]]:
    """Load noise calibration data for sensor."""
    if calibration_dir is None:
        if data_root is None:
            data_root = Path("dataset/processed")
        calibration_dir = data_root

    _, SENSOR_NAME_MAPPING, _ = _get_sample_config()
    for name in [sensor_name, SENSOR_NAME_MAPPING.get(sensor_name, sensor_name)]:
        calibration_file = calibration_dir / f"{name}_noise_calibration.json"
        if calibration_file.exists():
            try:
                with open(calibration_file, "r") as f:
                    calib_data = json.load(f)
                return {"a": float(calib_data["a"]), "b": float(calib_data["b"])}
            except Exception as e:
                logger.warning(
                    f"Failed to load calibration from {calibration_file}: {e}"
                )

    return None
