#!/usr/bin/env python3
"""
Sample utility functions for loading and finding images.

This module provides utilities for:
- Loading metadata JSON files
- Finding tile pairs
- Loading image files (.pt format)
- Extracting metadata information
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sample.metrics import compute_metrics_by_method, convert_range

if TYPE_CHECKING:
    from sample.gradients import PoissonGaussianGuidance

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions (Internal Utilities)
# ============================================================================


def _extract_tensor_from_dict(
    tensor: Any, file_path: Path, preferred_keys: Optional[List[str]] = None
) -> torch.Tensor:
    """
    Extract tensor from dictionary structure, trying preferred keys first.

    Args:
        tensor: Tensor or dict containing tensor
        file_path: Path for error messages
        preferred_keys: Optional list of keys to try first (default order used if None)

    Returns:
        Extracted tensor
    """
    if not isinstance(tensor, dict):
        return tensor

    default_keys = ["short", "long", "image"]
    keys_to_try = preferred_keys if preferred_keys else default_keys

    for key in keys_to_try:
        if key in tensor:
            return tensor[key]

    raise ValueError(f"Unrecognized dict structure in {file_path}")


def _ensure_chw_format(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is in CHW (channels, height, width) format.

    All images are RGB with 3 channels.

    Args:
        tensor: Input tensor in various formats

    Returns:
        Tensor in CHW format [3, H, W]
    """
    if tensor.ndim == 2:
        logger.warning(
            f"Received 2D tensor (H, W) - expected RGB image with 3 channels"
        )
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    elif tensor.ndim == 3:
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] == 3:
            pass
        else:
            logger.warning(
                f"Unexpected 3D tensor shape {tensor.shape} - expected [C, H, W] or [H, W, C] with C=3"
            )

    return tensor


def _validate_channels(
    tensor: torch.Tensor, target_channels: int, image_type: str = "", tile_id: str = ""
) -> torch.Tensor:
    """
    Validate tensor has expected number of channels (all images are RGB/3 channels).

    Args:
        tensor: Input tensor [C, H, W]
        target_channels: Expected number of channels (should be 3 for RGB)
        image_type: Type of image for logging (optional)
        tile_id: Tile ID for logging (optional)

    Returns:
        Tensor unchanged (validation only)
    """
    if tensor.shape[0] != target_channels:
        if image_type and tile_id:
            logger.warning(
                f"Channel mismatch for {image_type} image {tile_id}: "
                f"got {tensor.shape[0]} channels, expected {target_channels} channels"
            )

    return tensor


def _convert_minus1_to_01(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from [-1, 1] range to [0, 1] range.

    Args:
        tensor: Tensor in [-1, 1] range

    Returns:
        Tensor in [0, 1] range
    """
    return (tensor + 1.0) / 2.0


def _clamp_to_01(tensor: torch.Tensor) -> torch.Tensor:
    """
    Clamp tensor to [0, 1] range.

    Args:
        tensor: Input tensor

    Returns:
        Clamped tensor in [0, 1] range
    """
    return torch.clamp(tensor, 0.0, 1.0)


def _normalize_minus1_to_01_clamped(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from [-1, 1] range to [0, 1] range and clamp.

    Common pattern used in denormalization functions.

    Args:
        tensor: Tensor in [-1, 1] range

    Returns:
        Tensor in [0, 1] range, clamped
    """
    return _clamp_to_01(_convert_minus1_to_01(tensor))


def _construct_tile_path(tile_id: str, base_dir: Path) -> Path:
    """
    Construct path to tile .pt file.

    Args:
        tile_id: Tile ID
        base_dir: Base directory containing tiles

    Returns:
        Path to tile file
    """
    return base_dir / f"{tile_id}.pt"


def _create_tile_lookup(tiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Create lookup dictionary mapping tile_id to tile metadata.

    Args:
        tiles: List of tile metadata dictionaries

    Returns:
        Dictionary mapping tile_id to tile metadata
    """
    return {tile.get("tile_id"): tile for tile in tiles if tile.get("tile_id")}


# ============================================================================
# Metadata Operations
# ============================================================================


def _get_tile_metadata_dict(
    tile_id: str, tile_lookup: Dict[str, Dict[str, Any]], required: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Get tile metadata dictionary with consistent error handling.

    Args:
        tile_id: Tile ID
        tile_lookup: Dictionary mapping tile_id to tile metadata
        required: If True, raises ValueError on missing tile; if False, returns None

    Returns:
        Tile metadata dictionary, or None if not required and not found

    Raises:
        ValueError: If tile not found and required=True
    """
    tile_meta = tile_lookup.get(tile_id)
    if not tile_meta:
        if required:
            raise ValueError(f"Tile {tile_id} not found in metadata")
        return None
    return tile_meta


def _get_tile_metadata(
    tile_id: str, tile_lookup: Dict[str, Dict[str, Any]], field_name: str
) -> Any:
    """
    Get a field from tile metadata with consistent error handling.

    Args:
        tile_id: Tile ID
        tile_lookup: Dictionary mapping tile_id to tile metadata
        field_name: Name of the field to retrieve

    Returns:
        Field value

    Raises:
        ValueError: If tile not found or field missing
    """
    tile_meta = _get_tile_metadata_dict(tile_id, tile_lookup, required=True)

    value = tile_meta.get(field_name)
    if value is None:
        raise ValueError(f"No {field_name} in metadata for tile {tile_id}")

    return value


def _load_json_file(json_path: Path) -> Dict[str, Any]:
    """
    Load JSON file (internal helper).

    Args:
        json_path: Path to JSON file

    Returns:
        Parsed JSON dictionary
    """
    with open(json_path, "r") as f:
        return json.load(f)


def load_metadata_json(metadata_json: Path) -> Dict[str, Any]:
    """
    Load metadata JSON and create a lookup dictionary by tile_id.

    Args:
        metadata_json: Path to metadata JSON file

    Returns:
        Dictionary mapping tile_id to tile metadata
    """
    metadata = _load_json_file(metadata_json)
    tiles = metadata.get("tiles", [])
    tile_lookup = _create_tile_lookup(tiles)
    return tile_lookup


def load_test_tiles(
    metadata_json: Path,
    split: str = "test",
    sensor_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load test tile metadata from JSON file.

    Args:
        metadata_json: Path to metadata JSON file
        split: Data split to load (default: 'test')
        sensor_filter: Optional sensor filter ('sony', 'fuji', etc.)

    Returns:
        List of tile metadata dictionaries
    """
    metadata = _load_json_file(metadata_json)
    tiles = metadata.get("tiles", [])
    filtered_tiles = [tile for tile in tiles if tile.get("split") == split]

    if sensor_filter:
        filtered_tiles = [
            tile
            for tile in filtered_tiles
            if sensor_filter.lower() in tile.get("sensor_type", "").lower()
        ]

    return filtered_tiles


def get_sensor_from_metadata(
    tile_id: str, tile_lookup: Dict[str, Dict[str, Any]]
) -> str:
    """
    Get sensor information from metadata JSON.

    Args:
        tile_id: Tile ID
        tile_lookup: Dictionary mapping tile_id to tile metadata

    Returns:
        Sensor name (e.g., "sony", "fuji")

    Raises:
        ValueError: If sensor cannot be found in metadata
    """
    sensor_type = _get_tile_metadata(tile_id, tile_lookup, "sensor_type")

    if sensor_type in ["sony", "fuji"]:
        return sensor_type

    raise ValueError(
        f"Unknown sensor_type '{sensor_type}' in metadata for tile {tile_id}"
    )


def get_exposure_time_from_metadata(
    tile_id: str, tile_lookup: Dict[str, Dict[str, Any]]
) -> float:
    """
    Get exposure time from metadata JSON.

    Args:
        tile_id: Tile ID
        tile_lookup: Dictionary mapping tile_id to tile metadata

    Returns:
        Exposure time in seconds

    Raises:
        ValueError: If exposure time cannot be found in metadata
    """
    exposure_time = _get_tile_metadata(tile_id, tile_lookup, "exposure_time")

    try:
        return float(exposure_time)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid exposure_time '{exposure_time}' in metadata for tile {tile_id}: {e}"
        )


# ============================================================================
# Tile Finding & Selection
# ============================================================================


def find_long_tile_pair(
    short_tile_id: str, tile_lookup: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Find the long exposure tile pair for a given short exposure tile using metadata.

    Args:
        short_tile_id: The short exposure tile ID
        tile_lookup: Dictionary mapping tile_id to tile metadata

    Returns:
        Dictionary containing long exposure tile metadata if found, None otherwise
    """
    short_tile_meta = _get_tile_metadata_dict(
        short_tile_id, tile_lookup, required=False
    )
    if not short_tile_meta:
        logger.error(f"Short tile not found in metadata: {short_tile_id}")
        return None

    scene_id = short_tile_meta.get("scene_id")
    sensor_type = short_tile_meta.get("sensor_type")
    grid_x = short_tile_meta.get("grid_x")
    grid_y = short_tile_meta.get("grid_y")

    if not all([scene_id, sensor_type, grid_x is not None, grid_y is not None]):
        logger.error(f"Incomplete metadata for short tile {short_tile_id}")
        return None

    long_candidates = []
    for tile_id, tile_meta in tile_lookup.items():
        if (
            tile_meta.get("data_type") == "long"
            and tile_meta.get("sensor_type") == sensor_type
            and tile_meta.get("scene_id") == scene_id
            and tile_meta.get("grid_x") == grid_x
            and tile_meta.get("grid_y") == grid_y
        ):
            long_candidates.append(tile_meta)

    if not long_candidates:
        logger.warning(f"No long exposure tile candidates found for {short_tile_id}")
        return None

    if len(long_candidates) > 1:
        logger.warning(
            f"Unexpected: Found {len(long_candidates)} long exposure candidates for "
            f"short tile {short_tile_id} (scene_id={scene_id}, grid=({grid_x}, {grid_y})). "
            f"Expected exactly one. Using first candidate."
        )

    return long_candidates[0]


def select_tiles(
    tile_ids: Optional[List[str]],
    available_tiles: List[Dict],
    num_examples: int,
    seed: int,
) -> List[Dict]:
    """
    Select tiles to process based on tile_ids or random selection.
    """
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


# ============================================================================
# Image Loading
# ============================================================================


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
    """
    Unified function to load normalized .pt file with flexible options.

    Args:
        image_path: Path to .pt file (or can be constructed from tile_id + dir)
        device: Device to load tensor on
        image_type: Type of image for logging ("short", "long", etc.)
        target_channels: Target number of channels (for channel conversion)
        add_batch_dim: If True, add batch dimension [B,C,H,W], else [C,H,W]
        preferred_keys: Optional list of keys to try first for dict extraction
        return_metadata: If True, return metadata dict as second element
        return_raw: If True, return raw tensor as last element

    Returns:
        Tuple depending on flags:
        - If return_metadata=True, return_raw=False: (tensor, metadata)
        - If return_metadata=False, return_raw=True: (tensor, raw_tensor)
        - If both True: (tensor, metadata, raw_tensor)
        - If both False: (tensor,)

        Tensor shape: [B,C,H,W] if add_batch_dim=True, else [C,H,W]
        Tensor range: [-1,1] as saved by preprocessing pipeline
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    raw_tensor = torch.load(image_path, map_location=device)
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
        metadata = {
            "offset": 0.0,
            "original_range": [tensor_min, tensor_max],
            "processed_range": [tensor_min, tensor_max],
        }
        result.append(metadata)

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
    """
    Load image from tile_id and directory (convenience wrapper).

    Returns (tensor, metadata) with [C,H,W] shape (no batch dim).
    """
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
    """Load a short exposure .pt file and return both tensor and metadata."""
    return load_image_from_tile_id(tile_id, short_dir, device, "short", target_channels)


def load_long_image(
    tile_id: str,
    long_dir: Path,
    tile_lookup: Dict[str, Dict[str, Any]],
    sampler_device: torch.device,
    target_channels: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[Any]]:
    """
    Load long exposure image pair for a given short exposure tile.

    Args:
        tile_id: Short exposure tile ID
        long_dir: Directory containing long exposure .pt files
        tile_lookup: Dictionary mapping tile_id to tile metadata
        sampler_device: Device to load on
        target_channels: Target number of channels

    Returns:
        Tuple of (long_image, long_image_raw) or (None, None) if not found
    """
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


# ============================================================================
# Image Analysis
# ============================================================================


def analyze_image_brightness(image: torch.Tensor) -> Dict[str, float]:
    """Analyze image brightness characteristics."""
    img_01 = convert_range(image, "[-1,1]", "[0,1]")

    mean_brightness = img_01.mean().item()
    img_flat = img_01.flatten()

    brightness_category = (
        "Very Dark"
        if mean_brightness < 0.2
        else "Dark"
        if mean_brightness < 0.4
        else "Medium"
        if mean_brightness < 0.6
        else "Bright"
        if mean_brightness < 0.8
        else "Very Bright"
    )

    return {
        "mean": mean_brightness,
        "std": img_flat.std().item(),
        "min": img_flat.min().item(),
        "max": img_flat.max().item(),
        "p10": torch.quantile(img_flat, 0.1).item(),
        "p50": torch.quantile(img_flat, 0.5).item(),
        "p90": torch.quantile(img_flat, 0.9).item(),
        "category": brightness_category,
    }


# ============================================================================
# Exposure Ratio Operations
# ============================================================================


def get_exposure_ratio(tile_id: str, tile_lookup: Dict) -> float:
    """Extract exposure ratio from metadata."""
    short_exposure = get_exposure_time_from_metadata(tile_id, tile_lookup)
    long_pair_info = find_long_tile_pair(tile_id, tile_lookup)

    if long_pair_info:
        long_tile_id = long_pair_info.get("tile_id")
        long_exposure = get_exposure_time_from_metadata(long_tile_id, tile_lookup)
        if long_exposure > 0:
            return short_exposure / long_exposure

    return 1.0


def apply_exposure_scaling(
    short_image: torch.Tensor, exposure_ratio: float
) -> torch.Tensor:
    """
    Apply simple exposure scaling to short exposure input.

    Args:
        short_image: Short exposure image in [-1, 1] range
        exposure_ratio: Exposure ratio (t_short / t_long)

    Returns:
        Scaled image in [-1, 1] range
    """
    image_01 = _convert_minus1_to_01(short_image)
    scale_factor = 1.0 / exposure_ratio if exposure_ratio > 0 else 1.0
    scaled_01 = _clamp_to_01(image_01 * scale_factor)
    scaled_image = scaled_01 * 2.0 - 1.0

    return scaled_image


# ============================================================================
# Sensor Calibration & Normalization Utilities
# ============================================================================


def load_sensor_calibration_from_metadata(
    sensor_type: str, metadata_path: Path
) -> Tuple[float, float]:
    """Load black level and white level from metadata JSON file.

    Looks for values in the metadata file structure:
    - Checks files in metadata JSON for black_level and white_level
    - Raises ValueError if calibration values cannot be found

    Args:
        sensor_type: Sensor type ('sony' or 'fuji')
        metadata_path: Path to metadata JSON file (required)

    Returns:
        Tuple of (black_level, white_level) as floats

    Raises:
        FileNotFoundError: If metadata file cannot be found
        ValueError: If black_level or white_level cannot be found in metadata
    """
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path} for sensor '{sensor_type}'"
        )

    black_level = None
    white_level = None

    try:
        metadata = _load_json_file(metadata_path)

        pipeline_info = metadata.get("pipeline_info", {})
        sensor_ranges = pipeline_info.get("sensor_ranges", {})
        if sensor_type in sensor_ranges:
            sensor_range = sensor_ranges[sensor_type]
            if "max" in sensor_range:
                white_level = float(sensor_range["max"])

        files = metadata.get("files", [])
        for file_info in files:
            file_sensor = file_info.get("camera_type") or file_info.get("sensor_type")
            if file_sensor and file_sensor != sensor_type:
                continue

            if "black_level" in file_info:
                black_level = float(file_info["black_level"])
            if "white_level" in file_info:
                white_level = float(file_info["white_level"])
            if black_level is not None and white_level is not None:
                break
    except Exception as e:
        raise ValueError(f"Failed to load calibration from {metadata_path}: {e}") from e

    if black_level is None:
        raise ValueError(
            f"black_level not found in metadata file {metadata_path} for sensor '{sensor_type}'. "
            f"Ensure metadata file contains black_level in file entries."
        )

    if white_level is None:
        raise ValueError(
            f"white_level not found in metadata file {metadata_path} for sensor '{sensor_type}'. "
            f"Ensure metadata file contains white_level in file entries or pipeline_info.sensor_ranges."
        )

    return float(black_level), float(white_level)


def compute_sensor_range(black_level: float, white_level: float) -> float:
    """Compute sensor range from calibration values."""
    return white_level - black_level


def normalize_physical_to_normalized(
    y_physical: torch.Tensor,
    black_level: float,
    white_level: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Normalize physical ADU values to [0,1] using sensor calibration.

    Matches preprocessing pipeline: (y - black_level) / (white_level - black_level)

    Args:
        y_physical: Physical values in [black_level, white_level] ADU
        black_level: Sensor black level
        white_level: Sensor white level
        epsilon: Small constant for numerical stability

    Returns:
        Normalized values in [0, 1] range
    """
    sensor_range = compute_sensor_range(black_level, white_level)
    y_norm = (y_physical - black_level) / (sensor_range + epsilon)
    return _clamp_to_01(y_norm)


def denormalize_to_physical(
    tensor: torch.Tensor, black_level: float, white_level: float
) -> np.ndarray:
    """Convert tensor from [-1,1] model space to physical ADU units using calibration.

    Pipeline inversion:
    1. [-1,1] → [0,1]: (tensor + 1) / 2
    2. [0,1] → ADU: tensor_norm * (white_level - black_level) + black_level

    This matches the normalization in pack_raw_sony/pack_raw_fuji:
    - Normalization: (raw - black_level) / (white_level - black_level) → [0,1]
    - Denormalization: [0,1] → tensor_norm * (white_level - black_level) + black_level

    Args:
        tensor: Image tensor in [-1, 1] range
        black_level: Black level from sensor calibration
        white_level: White level from sensor calibration

    Returns:
        Image array in physical ADU units
    """
    tensor_norm = _normalize_minus1_to_01_clamped(tensor)
    sensor_range = compute_sensor_range(black_level, white_level)
    tensor_phys = tensor_norm * sensor_range + black_level

    return tensor_phys.cpu().numpy()


def get_sensor_calibration_params(
    sensor_name: Optional[str],
    extracted_sensor: str,
    short_phys: np.ndarray,
    sensor_ranges: Dict[str, Dict[str, float]],
    conservative_factor: float,
) -> Tuple[float, Dict[str, Any], Dict[str, float], float]:
    """Get calibrated sigma_max and noise estimates."""
    from sample.sensor_calibration import SensorCalibration

    if sensor_name is not None:
        calib_sensor_name = sensor_name
    else:
        sensor_mapping = {"sony": "sony_a7s_ii", "fuji": "fuji_xt2"}
        calib_sensor_name = sensor_mapping.get(extracted_sensor, extracted_sensor)

    mean_signal_physical = float(short_phys.mean())
    sensor_range = sensor_ranges.get(extracted_sensor, {"min": 0.0, "max": 16383.0})
    s_sensor = compute_sensor_range(sensor_range["min"], sensor_range["max"])

    calib_params = SensorCalibration.get_posterior_sampling_params(
        sensor_name=calib_sensor_name,
        mean_signal_physical=mean_signal_physical,
        s=s_sensor,
        conservative_factor=conservative_factor,
    )

    estimated_sigma = calib_params["sigma_max"]
    sensor_info = calib_params["sensor_info"]

    noise_estimates = {
        "method": "sensor_calibration",
        "sensor_name": calib_sensor_name,
        "extracted_sensor": extracted_sensor,
        "sigma_max_calibrated": estimated_sigma,
        "mean_signal_physical": mean_signal_physical,
        "sensor_specs": sensor_info,
    }

    return estimated_sigma, noise_estimates, sensor_range, s_sensor


def compute_residual_components(
    x0_hat: torch.Tensor,
    y_e_physical: torch.Tensor,
    black_level: float,
    white_level: float,
    s: float,
    alpha: float,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute common components for gradient computation: normalized observation,
    scaled observation, expected value, and residual.

    Args:
        x0_hat: Normalized prediction [B,C,H,W] in [0,1]
        y_e_physical: Physical observation [B,C,H,W] in ADU
        black_level: Sensor black level
        white_level: Sensor white level
        s: Scale factor (sensor_range)
        alpha: Exposure ratio
        epsilon: Numerical stability constant

    Returns:
        Tuple of (y_e_scaled, expected_at_short_exp, residual) all in sensor range units
    """
    y_e_norm = normalize_physical_to_normalized(
        y_e_physical, black_level, white_level, epsilon
    )
    y_e_scaled = y_e_norm * s
    expected_at_short_exp = alpha * s * x0_hat
    residual = y_e_scaled - expected_at_short_exp

    return y_e_scaled, expected_at_short_exp, residual


# ============================================================================
# Validation
# ============================================================================


def validate_exposure_ratio(
    short_tensor: torch.Tensor,
    long_tensor: torch.Tensor,
    assumed_alpha: float,
    sensor_type: str,
) -> Tuple[float, float]:
    """
    Validate exposure ratio by comparing configured value with measured brightness ratio.

    Args:
        short_tensor: Short exposure image tensor [C, H, W], range [-1,1]
        long_tensor: Long exposure image tensor [C, H, W], range [-1,1]
        assumed_alpha: Expected exposure ratio (t_short / t_long)
        sensor_type: Sensor type for logging

    Returns:
        (measured_alpha, error_percent)
    """
    short_mean = np.mean(_convert_minus1_to_01(short_tensor).detach().cpu().numpy())
    long_mean = np.mean(_convert_minus1_to_01(long_tensor).detach().cpu().numpy())

    if long_mean < 1e-6:
        logger.warning(
            f"{sensor_type}: Long exposure too dark (mean={long_mean:.6f}), skipping validation"
        )
        return assumed_alpha, 0.0

    measured_alpha = short_mean / long_mean
    error_percent = abs(measured_alpha - assumed_alpha) / assumed_alpha * 100

    if error_percent > 20.0:
        logger.error(
            f"{sensor_type}: Exposure ratio mismatch {error_percent:.1f}% (expected {assumed_alpha:.4f}, got {measured_alpha:.4f})"
        )
    elif error_percent > 10.0:
        logger.warning(
            f"{sensor_type}: Exposure ratio error {error_percent:.1f}% (expected {assumed_alpha:.4f}, got {measured_alpha:.4f})"
        )
    else:
        logger.debug(
            f"{sensor_type}: Exposure ratio validated ({measured_alpha:.4f}, error {error_percent:.1f}%)"
        )

    return measured_alpha, error_percent


def validate_sensor_range_consistency(
    s: float, black_level: float, white_level: float
) -> None:
    """
    Validate that s equals sensor_range for unit consistency.

    Raises:
        ValueError: If s does not equal sensor_range within tolerance
    """
    sensor_range = compute_sensor_range(black_level, white_level)
    if abs(s - sensor_range) > 1e-3:
        raise ValueError(
            f"s={s} must equal sensor_range={sensor_range} for unit consistency!\n"
            f"s = white_level - black_level ensures proper normalization.\n"
            f"This ensures proper comparison between observed and expected values."
        )


def validate_tensor_inputs(
    x0_hat: torch.Tensor,
    y_e: torch.Tensor,
    black_level: float,
    white_level: float,
    offset: float = 0.0,
) -> None:
    """
    Validate guidance input tensors for shape and range consistency.

    Args:
        x0_hat: Denoised estimate tensor [B,C,H,W], range [0,1]
        y_e: Observed measurement tensor [B,C,H,W], in physical units
        black_level: Sensor black level for range validation
        white_level: Sensor white level for range validation
        offset: Offset applied to data (for range validation)

    Raises:
        ValueError: If tensor shapes don't match or types are invalid
    """
    if not isinstance(x0_hat, torch.Tensor):
        raise ValueError(f"x0_hat must be a torch.Tensor, got {type(x0_hat)}")

    if not isinstance(y_e, torch.Tensor):
        raise ValueError(f"y_e must be a torch.Tensor, got {type(y_e)}")

    if x0_hat.shape != y_e.shape:
        raise ValueError(f"Shape mismatch: x0_hat {x0_hat.shape} vs y_e {y_e.shape}")

    if x0_hat.min() < 0.0 or x0_hat.max() > 1.0:
        logger.warning(
            f"x0_hat values outside [0,1] range: [{x0_hat.min():.4f}, {x0_hat.max():.4f}]"
        )

    if y_e.min() < black_level - offset or y_e.max() > white_level + offset:
        logger.warning(
            f"y_e values outside expected physical range: [{y_e.min():.4f}, {y_e.max():.4f}], "
            f"expected [{black_level}, {white_level}]"
        )


def validate_physical_consistency(
    x_enhanced: torch.Tensor,
    y_e_physical: torch.Tensor,
    s: float,
    sigma_r: float,
    exposure_ratio: float,
    sensor_min: float,
    sensor_max: float,
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    Validate physical consistency using reduced chi-squared statistic.

    Tests: y_short ≈ Poisson(α·s·x_long) + N(0, σ_r²)

    A physically consistent reconstruction should have χ²_red ≈ 1.0, indicating
    that residuals match the expected Poisson-Gaussian noise distribution.

    Args:
        x_enhanced: Enhanced image in [0,1] normalized space [B,C,H,W] (LONG exposure)
        y_e_physical: Observed short exposure measurement in physical units [B,C,H,W] (SHORT exposure)
        s: Scale factor used in PG guidance
        sigma_r: Read noise standard deviation (in physical units)
        exposure_ratio: Exposure ratio t_low / t_long
        sensor_min: Minimum physical value (sensor black level)
        sensor_max: Maximum physical value (sensor white level)
        epsilon: Small constant for numerical stability

    Returns:
        Dictionary with consistency metrics:
        - chi_squared: Reduced χ² statistic (should be ≈ 1.0)
        - chi_squared_std: Standard deviation of χ² per pixel
        - physically_consistent: Boolean flag (0.8 < χ² < 1.2)
        - mean_residual: Mean residual (should be ≈ 0)
        - max_residual: Maximum absolute residual
    """
    sensor_range = compute_sensor_range(sensor_min, sensor_max)
    y_e_norm = _clamp_to_01((y_e_physical - sensor_min) / (sensor_range + epsilon))

    y_e_scaled = y_e_norm * s

    expected_y_at_short_exp = exposure_ratio * s * x_enhanced

    variance_at_short_exp = exposure_ratio * s * x_enhanced + sigma_r**2 + epsilon

    residual = y_e_scaled - expected_y_at_short_exp

    chi_squared_map = (residual**2) / variance_at_short_exp

    chi_squared_red = chi_squared_map.mean().item()

    chi_squared_std = chi_squared_map.std().item()
    mean_residual = residual.mean().item()
    max_residual = residual.abs().max().item()

    is_consistent = 0.8 < chi_squared_red < 1.2

    return {
        "chi_squared": chi_squared_red,
        "chi_squared_std": chi_squared_std,
        "physically_consistent": is_consistent,
        "mean_residual": mean_residual,
        "max_residual": max_residual,
    }


class PhotonCountValidator:
    """
    Validates Poisson-Gaussian approximation quality across different photon count regimes.

    The Gaussian approximation to Poisson(λ) + N(0, σ_r²) breaks down when:
    - λ < 10 photons (Poisson is highly skewed)
    - λ < 3 photons (discrete nature dominates)

    This validator checks these conditions and provides guidance.
    """

    @staticmethod
    def estimate_photon_counts(
        y_e_physical: torch.Tensor,
        alpha: float,
        s: float,
        black_level: float,
        white_level: float,
    ) -> Dict[str, float]:
        """
        Estimate photon count statistics from observed measurement.

        Args:
            y_e_physical: Observed measurement in physical units (ADU)
            alpha: Exposure ratio
            s: Scale factor (= sensor_range = white_level - black_level)
            black_level: Sensor black level (from preprocessing calibration)
            white_level: Sensor white level (from preprocessing calibration)

        Returns:
            Dictionary with photon count statistics
        """
        y_norm = normalize_physical_to_normalized(
            y_e_physical, black_level, white_level
        )
        lambda_est = alpha * s * y_norm
        lambda_flat = lambda_est.flatten().cpu().numpy()

        return {
            "mean_photons": float(np.mean(lambda_flat)),
            "min_photons": float(np.min(lambda_flat)),
            "p10_photons": float(np.percentile(lambda_flat, 10)),
            "p50_photons": float(np.percentile(lambda_flat, 50)),
            "p90_photons": float(np.percentile(lambda_flat, 90)),
            "max_photons": float(np.max(lambda_flat)),
            "fraction_below_10": float(np.mean(lambda_flat < 10)),
            "fraction_below_3": float(np.mean(lambda_flat < 3)),
            "fraction_below_1": float(np.mean(lambda_flat < 1)),
        }

    @staticmethod
    def validate_approximation_quality(
        photon_stats: Dict[str, float], strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate that Gaussian approximation is appropriate.

        Returns:
            Dictionary with validation results and recommendations
        """
        warnings = []

        critical_threshold = 3.0 if strict else 1.0
        warning_threshold = 10.0 if strict else 5.0
        good_threshold = 20.0

        mean_photons = photon_stats["mean_photons"]
        frac_below_10 = photon_stats["fraction_below_10"]
        frac_below_3 = photon_stats["fraction_below_3"]

        if mean_photons >= good_threshold and frac_below_10 < 0.1:
            quality = "excellent"
            is_valid = True
        elif mean_photons >= warning_threshold and frac_below_3 < 0.2:
            quality = "good"
            is_valid = True
            warnings.append(
                f"Some pixels have low photon counts (mean={mean_photons:.1f}). "
                f"Gaussian approximation may be slightly inaccurate."
            )
        elif mean_photons >= critical_threshold:
            quality = "marginal"
            is_valid = not strict
            warnings.append(
                f"WARNING: Low photon counts detected (mean={mean_photons:.1f}). "
                f"{frac_below_10*100:.1f}% of pixels have λ < 10. "
                f"Gaussian approximation may introduce noticeable errors."
            )
        else:
            quality = "poor"
            is_valid = False
            warnings.append(
                f"CRITICAL: Very low photon counts (mean={mean_photons:.1f}). "
                f"{frac_below_3*100:.1f}% of pixels have λ < 3. "
                f"Gaussian approximation is inappropriate - consider discrete Poisson model."
            )

        if quality == "excellent":
            action = (
                "Gaussian approximation is highly accurate. Proceed with confidence."
            )
        elif quality == "good":
            action = "Gaussian approximation is adequate. Results should be reliable."
        elif quality == "marginal":
            action = "Consider: (1) Increase exposure time, (2) Use discrete Poisson model, or (3) Accept reduced accuracy."
        else:
            action = "CRITICAL: Switch to discrete Poisson diffusion model for accurate results."

        return {
            "is_valid": is_valid,
            "warnings": warnings,
            "approximation_quality": quality,
            "recommended_action": action,
            "photon_statistics": photon_stats,
        }

    @staticmethod
    def compute_approximation_error(
        lambda_mean: float,
        sigma_r: float,
    ) -> Dict[str, float]:
        """Compute theoretical approximation error metrics."""
        # Skewness of Poisson(λ) is 1/√λ
        skewness = (
            1.0 / np.sqrt(lambda_mean + sigma_r**2)
            if (lambda_mean + sigma_r**2) > 0
            else 0.0
        )

        # Approximate KL divergence using moment matching error
        kl_approx = skewness**2 / 12

        return {
            "kl_divergence_approx": kl_approx,
            "relative_variance_error": 0.0,  # By construction
            "skewness": skewness,
            "lambda_mean": lambda_mean,
            "sigma_r": sigma_r,
        }


# ============================================================================
# Optimization
# ============================================================================


def optimize_sigma(
    sampler: Any,
    short_image: torch.Tensor,
    long_image: torch.Tensor,
    class_labels: Optional[torch.Tensor],
    sigma_range: Tuple[float, float],
    num_trials: int = 10,
    num_steps: int = 18,
    metric: str = "ssim",
    pg_guidance: Optional["PoissonGaussianGuidance"] = None,
    y_e: Optional[torch.Tensor] = None,
    exposure_ratio: float = 1.0,
    no_heun: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Find optimal sigma_max by trying multiple values and maximizing SSIM (or minimizing MSE).

    Args:
        sampler: EDM posterior sampler instance with posterior_sample method
        short_image: Short exposure observation (for initialization if needed)
        long_image: Long exposure reference for metric computation
        class_labels: Class labels (not used for unconditional model)
        sigma_range: (min_sigma_max, max_sigma_max) to search
        num_trials: Number of sigma values to try
        num_steps: Sampling steps
        metric: 'ssim' (maximize) or 'mse' (minimize) or 'psnr' (maximize)
        pg_guidance: Poisson-Gaussian guidance module
        y_e: Observed noisy measurement for PG guidance (in physical units)
        exposure_ratio: Exposure ratio t_low / t_long
        no_heun: Disable Heun's 2nd order correction

    Returns:
        Tuple of (best_sigma, results_dict)
    """
    sigma_values = np.logspace(
        np.log10(sigma_range[0]), np.log10(sigma_range[1]), num=num_trials
    )

    best_sigma = sigma_values[0]
    best_metric_value = float("-inf") if metric in ["ssim", "psnr"] else float("inf")
    all_results = []

    for sigma in sigma_values:
        restored, _ = sampler.posterior_sample(
            short_image,
            sigma_max=sigma,
            class_labels=class_labels,
            num_steps=num_steps,
            pg_guidance=pg_guidance,
            no_heun=no_heun,
            y_e=y_e,
            exposure_ratio=exposure_ratio,
        )

        metrics = compute_metrics_by_method(
            long_image, restored, "pg", device=sampler.device
        )

        all_results.append(
            {
                "sigma": float(sigma),
                "ssim": metrics["ssim"],
                "psnr": metrics["psnr"],
                "mse": metrics["mse"],
            }
        )

        metric_value = metrics[metric]
        is_better = (
            metric_value > best_metric_value
            if metric in ["ssim", "psnr"]
            else metric_value < best_metric_value
        )

        if is_better:
            best_sigma = sigma
            best_metric_value = metric_value
            logger.debug(f"  New best: σ_max={sigma:.6f}, {metric}={metric_value:.4f}")

    logger.info(f"Best sigma_max: {best_sigma:.6f} ({metric}={best_metric_value:.4f})")

    return best_sigma, {
        "best_sigma": float(best_sigma),
        "best_metric": metric,
        "best_metric_value": float(best_metric_value),
        "all_trials": all_results,
    }
