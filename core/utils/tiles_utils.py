"""
Tile utility functions for the Poisson-Gaussian Diffusion project.

This module provides tile extraction, processing, conversion, and saving utilities.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Import shared utilities to avoid duplication
from core.utils.data_utils import _get_sample_config
from core.utils.file_utils import (
    _ensure_dir,
    convert_uint8_png_to_float32_png,
    save_json_file,
)
from core.utils.sensor_utils import SensorProcessingError, convert_raw_tile_to_png
from core.utils.tensor_utils import save_tensor

logger = logging.getLogger(__name__)


class InvalidTileConfiguration(SensorProcessingError):
    """Raised when tile extraction parameters are invalid"""

    pass


class TileInfo:
    """Simple data class for tile information."""

    def __init__(
        self, tile_data, grid_position, image_position, valid_ratio, is_edge_tile
    ):
        self.tile_data = tile_data
        self.grid_position = grid_position
        self.image_position = image_position
        self.valid_ratio = valid_ratio
        self.is_edge_tile = is_edge_tile


def extract_tiles(
    image: np.ndarray, rows: int, cols: int, tile_size: int = 256
) -> List[TileInfo]:
    """
    Extract tiles from image with even stride to cover entire FOV.

    Supports both RGB (3 channels) and raw sensor data (4 for Sony Bayer, 9 for Fuji X-Trans).
    Uses even strides to evenly distribute tiles across the entire image,
    providing fair representation of all regions.

    Args:
        image: Image as numpy array in CHW format (C, H, W)
               Accepted channel counts:
               - 3: RGB image
               - 4: Sony Bayer (RGGB)
               - 9: Fuji X-Trans
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        tile_size: Size of each tile (default: 256)

    Returns:
        List of TileInfo objects

    Raises:
        ValueError: If inputs fail validation
    """
    try:
        if image is None:
            raise InvalidTileConfiguration("image cannot be None")

        if image.size == 0:
            raise InvalidTileConfiguration("image cannot be empty")

        if len(image.shape) != 3:
            raise InvalidTileConfiguration(
                f"Expected 3D image shape (C, H, W), got {len(image.shape)}D: {image.shape}"
            )

        valid_channels = [3, 4, 9]
        if image.shape[0] not in valid_channels:
            raise InvalidTileConfiguration(
                f"Expected {valid_channels} channels, got {image.shape[0]}. "
                f"Valid formats: RGB (3), Sony Bayer (4), Fuji X-Trans (9)"
            )

        if rows <= 0 or cols <= 0:
            raise InvalidTileConfiguration(
                f"Grid dimensions must be positive: rows={rows}, cols={cols}"
            )

        if tile_size <= 0:
            raise InvalidTileConfiguration(
                f"tile_size must be positive, got {tile_size}"
            )

        C, H, W = image.shape

        if H < tile_size or W < tile_size:
            raise InvalidTileConfiguration(
                f"Image ({H}×{W}) is smaller than tile_size ({tile_size}×{tile_size}). "
                f"Cannot extract tiles from image smaller than a single tile."
            )

        stride_h = int(np.floor((H - tile_size) / (rows - 1))) if rows > 1 else 0
        stride_w = int(np.floor((W - tile_size) / (cols - 1))) if cols > 1 else 0

        tiles = []
        for row in range(rows):
            for col in range(cols):
                y_start = H - tile_size if row == rows - 1 else row * stride_h
                x_start = W - tile_size if col == cols - 1 else col * stride_w

                y_end = y_start + tile_size
                x_end = x_start + tile_size

                tile_data = image[:, y_start:y_end, x_start:x_end]

                assert tile_data.shape[-2:] == (
                    tile_size,
                    tile_size,
                ), f"Tile size mismatch: expected {tile_size}×{tile_size}, got {tile_data.shape[-2:]}"

                tile_info = TileInfo(
                    tile_data=tile_data,
                    grid_position=(col, row),
                    image_position=(x_start, y_start),
                    valid_ratio=1.0,
                    is_edge_tile=(
                        row == 0 or row == rows - 1 or col == 0 or col == cols - 1
                    ),
                )

                tiles.append(tile_info)

        return tiles

    except (InvalidTileConfiguration, AssertionError) as e:
        logger.error(f"Tile extraction failed: {e}")
        raise  # Re-raise after logging for caller to handle
    except Exception as e:
        logger.error(f"Unexpected error in tile extraction: {e}")
        raise InvalidTileConfiguration(f"Tile extraction failed: {e}") from e


def normalize_tile_to_range(
    tile_data: np.ndarray, domain_min: float, domain_max: float
) -> np.ndarray:
    """
    Normalize tile data from [domain_min, domain_max] to [0, 1].

    Args:
        tile_data: Tile data array
        domain_min: Minimum value in domain
        domain_max: Maximum value in domain

    Returns:
        Normalized tile data in [0, 1]
    """
    normalized = (tile_data - domain_min) / (domain_max - domain_min)
    return np.clip(normalized, 0, 1)


def _save_short_long_images(
    output_dir: Path,
    short_image: torch.Tensor,
    long_image: Optional[torch.Tensor],
    prefix: str = "",
) -> None:
    """Save short and long exposure images."""
    save_tensor(short_image, output_dir / f"{prefix}short.pt")
    if long_image is not None:
        save_tensor(long_image, output_dir / f"{prefix}long.pt")


def _save_restoration_results(
    output_dir: Path,
    restoration_results: Dict[str, torch.Tensor],
    prefix: str = "",
) -> None:
    """Save restoration results to files."""
    RESERVED_TILE_KEYS, _, _ = _get_sample_config()
    for method_name, restored_tensor in restoration_results.items():
        if restored_tensor is not None and method_name not in RESERVED_TILE_KEYS:
            filename = (
                f"{prefix}restored_{method_name}.pt"
                if prefix
                else f"restored_{method_name}.pt"
            )
            save_tensor(restored_tensor, output_dir / filename)


def save_tile_files(
    output_dir: Path,
    tile_id: str,
    short_image: torch.Tensor,
    long_image: Optional[torch.Tensor],
    restoration_results: Dict[str, torch.Tensor],
) -> None:
    """Save tile files (short, long, and restoration results)."""
    _ensure_dir(output_dir)
    _save_short_long_images(output_dir, short_image, long_image)
    _save_restoration_results(output_dir, restoration_results)


def save_scene_metrics_json(
    scene_dir: Path,
    scene_id: str,
    sensor: str,
    scene_metrics: List[Dict[str, Any]],
    aggregate_metrics: Dict[str, Any],
) -> None:
    """Save scene metrics to JSON file."""
    key_metric_names = [
        "tile_median_brightness",
        "pg_psnr",
        "pg_ssim",
        "gaussian_psnr",
        "gaussian_ssim",
    ]
    results_summary = {
        key: aggregate_metrics[key]
        for key in key_metric_names
        if aggregate_metrics and key in aggregate_metrics
    }

    save_json_file(
        scene_dir / "scene_metrics.json",
        {
            "scene_id": scene_id,
            "sensor": sensor,
            "num_tiles": len(scene_metrics),
            "results": results_summary,
            "aggregate_metrics": aggregate_metrics,
            "tiles": scene_metrics,
        },
    )


def save_scene_tiles(
    scene_dir: Path,
    tile_metadata_list: List[Dict[str, Any]],
    short_images: List[torch.Tensor],
    long_images: List[Optional[torch.Tensor]],
    restoration_results_batch: Dict[str, torch.Tensor],
) -> None:
    """Save all tiles for a scene."""
    tiles_subdir = scene_dir / "tiles"
    _ensure_dir(tiles_subdir)

    for i, meta in enumerate(tile_metadata_list):
        tile_id = meta.get("tile_id", f"tile_{i}")

        individual_results = {}
        for method, batch_tensor in restoration_results_batch.items():
            if batch_tensor is not None and batch_tensor.shape[0] > i:
                individual_results[method] = batch_tensor[i : i + 1]

        short_image = short_images[i]
        long_image = long_images[i] if i < len(long_images) else None

        _save_short_long_images(
            tiles_subdir, short_image, long_image, prefix=f"{tile_id}_"
        )
        _save_restoration_results(
            tiles_subdir, individual_results, prefix=f"{tile_id}_"
        )


def convert_tiles_to_png(
    tiles: List[Dict[str, Any]],
    output_base: Path,
    format_type: str,
    data_type: str,
) -> Dict[str, int]:
    """Convert list of tiles to PNG images.

    Args:
        tiles: List of tile info dictionaries containing:
            - tile_id: Tile identifier
            - raw_file_path: Path to RAW file
            - sensor_type: Sensor type ("sony" or "fuji")
            - image_x: X coordinate of tile in processed image
            - image_y: Y coordinate of tile in processed image
        output_base: Base output directory
        format_type: Format type ("uint8_png" or "float32_png")
        data_type: Data type ("long" or "short")

    Returns:
        Dictionary with conversion statistics: {"success": int, "failed": int, "skipped": int}
    """
    output_dir = output_base / format_type / data_type
    _ensure_dir(output_dir)

    stats = {"success": 0, "failed": 0, "skipped": 0}

    for tile_info in tqdm(tiles, desc=f"Converting {format_type}/{data_type}"):
        raw_file_path = Path(tile_info["raw_file_path"])
        tile_id = tile_info["tile_id"]
        sensor_type = tile_info.get("sensor_type")
        image_x = tile_info.get("image_x")
        image_y = tile_info.get("image_y")

        # Validation
        if not sensor_type:
            logger.warning(f"Missing sensor type for {tile_id}, skipping")
            stats["skipped"] += 1
            continue

        if not raw_file_path.exists():
            logger.warning(f"RAW file not found: {raw_file_path}")
            stats["skipped"] += 1
            continue

        if image_x is None or image_y is None:
            logger.warning(f"Missing tile coordinates for {tile_id}, skipping")
            stats["skipped"] += 1
            continue

        output_path = output_dir / f"{tile_id}.png"

        if output_path.exists():
            stats["skipped"] += 1
            continue

        # Convert
        success = convert_raw_tile_to_png(
            raw_file_path, output_path, image_x, image_y, sensor_type, format_type
        )

        stats["success" if success else "failed"] += 1

    return stats


def convert_uint8_tiles_to_float32_png(
    output_base: Path,
    data_type: str,
    sensor: Optional[str] = None,
) -> Dict[str, int]:
    """Convert all uint8_png tiles to float32_png format.

    Args:
        output_base: Base output directory containing uint8_png and float32_png subdirectories
        data_type: Data type ("long" or "short")
        sensor: Optional sensor filter (e.g., "sony" or "fuji")

    Returns:
        Dictionary with conversion statistics: {"success": int, "failed": int, "skipped": int}
    """
    uint8_dir = output_base / "uint8_png" / data_type
    float32_dir = output_base / "float32_png" / data_type
    _ensure_dir(float32_dir)

    stats = {"success": 0, "failed": 0, "skipped": 0}

    if not uint8_dir.exists():
        logger.warning(f"uint8_png directory not found: {uint8_dir}")
        return stats

    uint8_png_files = list(uint8_dir.glob("*.png"))

    if sensor:
        sensor_prefix = f"{sensor}_"
        uint8_png_files = [
            f for f in uint8_png_files if f.name.startswith(sensor_prefix)
        ]

    if not uint8_png_files:
        logger.warning(
            f"No PNG files found in {uint8_dir}"
            + (f" for sensor {sensor}" if sensor else "")
        )
        return stats

    logger.info(
        f"Found {len(uint8_png_files)} PNG files to convert"
        + (f" for {sensor}" if sensor else "")
    )

    for uint8_png_path in tqdm(
        uint8_png_files, desc=f"Converting uint8->float32/{data_type}"
    ):
        float32_png_path = float32_dir / uint8_png_path.name

        if float32_png_path.exists() and float32_png_path.with_suffix(".npy").exists():
            stats["skipped"] += 1
            continue

        success = convert_uint8_png_to_float32_png(uint8_png_path, float32_png_path)
        stats["success" if success else "failed"] += 1

    return stats
