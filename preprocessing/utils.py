"""
Utility functions for the preprocessing pipeline.

This module provides helper functions for file parsing, scene ID extraction,
data type determination, normalization, and other common operations.
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rawpy

from preprocessing.config import BLACK_LEVELS, CAMERA_CONFIGS, SENSOR_RANGES

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exception Types
# ============================================================================


class SensorProcessingError(Exception):
    """Base exception for sensor processing errors"""

    pass


class InvalidRawDataError(SensorProcessingError):
    """Raised when raw image data is invalid or corrupted"""

    pass


class InvalidTileConfiguration(SensorProcessingError):
    """Raised when tile extraction parameters are invalid"""

    pass


class SensorDetectionError(SensorProcessingError):
    """Raised when sensor type cannot be detected from file"""

    pass


# ============================================================================
# Data Classes and Utility Types
# ============================================================================


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


def get_pixel_stats(data: np.ndarray) -> Tuple[float, float, float, float]:
    """Get pixel statistics (min, max, mean, median) from data.

    Returns 0.0 for all metrics if data is None or contains no valid values.
    Uses numpy NaN-aware functions to handle invalid values gracefully.

    Args:
        data: Input array (any shape)

    Returns:
        Tuple of (min_val, max_val, mean_val, median_val) as float
        All values are 0.0 if data is invalid/empty (for type consistency)

    Example:
        >>> stats = get_pixel_stats(np.array([1, 2, 3, 4, 5]))
        >>> stats
        (1.0, 5.0, 3.0, 3.0)

        >>> stats = get_pixel_stats(None)
        >>> stats
        (0.0, 0.0, 0.0, 0.0)
    """
    if data is None or data.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Handle different data shapes
    flat_data = data.flatten()

    # Use numpy NaN-aware functions for robustness
    try:
        return (
            float(np.nanmin(flat_data)),
            float(np.nanmax(flat_data)),
            float(np.nanmean(flat_data)),
            float(np.nanmedian(flat_data)),
        )
    except (ValueError, RuntimeError):
        # All values are NaN
        return 0.0, 0.0, 0.0, 0.0


def determine_data_type(file_path: str, domain: str) -> str:
    """Determine if file contains long or short exposure data."""
    file_path_lower = file_path.lower()

    if domain == "photography":
        return "long" if "/long/" in file_path_lower else "short"

    return "unknown"


def load_sid_split_files(sid_data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load SID dataset split files and parse ISO/aperture information.
    Loads ALL exposure files listed in the split files (*.txt).

    Args:
        sid_data_path: Path to SID dataset root directory

    Returns:
        Dictionary mapping file paths to their split and metadata information
    """
    sid_path = Path(sid_data_path)
    splits_path = sid_path / "splits"
    file_info = {}

    # Load ALL files from split files
    split_names = [
        "train",
        "val",
        "test",
    ]  # Changed "validation" to "val" to match actual filenames
    split_files_data = {}  # Store all file entries from split files

    for camera_type, config in CAMERA_CONFIGS.items():
        camera_name = config["base_dir"]
        for split_name in split_names:
            filename = f"{camera_name}_{split_name}_list.txt"
            split_file_path = splits_path / filename

            if not split_file_path.exists():
                logger.warning(f"SID split file not found: {split_file_path}")
                continue

            logger.info(f"Loading {filename} ({split_name} split)")

            with open(split_file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue

                    # Parse format: ./Camera/short/short_file.ext ./Camera/long/long_file.ext ISO_value F_value
                    short_path, long_path, iso_str, aperture_str = parts

                    # Convert to absolute paths
                    short_abs_path = str(sid_path / short_path[2:])  # Remove "./"
                    long_abs_path = str(sid_path / long_path[2:])

                    # Store both files with their metadata
                    for file_path, file_type in [
                        (short_abs_path, "short"),
                        (long_abs_path, "long"),
                    ]:
                        # Extract scene_id and exposure time
                        filename_parts = Path(file_path).stem.split("_")
                        if len(filename_parts) >= 3:
                            scene_id = filename_parts[0]
                            exposure_time = filename_parts[2]

                            # Store in split_files_data
                            # Use full file path as part of key to ensure uniqueness
                            key = (
                                camera_type,
                                scene_id,
                                file_type,
                                exposure_time,
                                file_path,
                            )
                            split_files_data[key] = {
                                "file_path": file_path,
                                "split": split_name,
                                "iso": int(iso_str.replace("ISO", ""))
                                if iso_str.startswith("ISO")
                                else None,
                                "aperture": float(aperture_str.replace("F", ""))
                                if aperture_str.startswith("F")
                                else None,
                            }

    # Group files by scene_id to create scene-level metadata
    logger.info("Grouping files by scene_id...")

    scene_groups = {}  # {camera_type: {scene_id: {"short": [], "long": []}}}

    # Group all files from split files by scene_id
    for (
        camera_type,
        scene_id,
        file_type,
        exposure_time,
        file_path,
    ), file_data in split_files_data.items():
        if camera_type not in scene_groups:
            scene_groups[camera_type] = {}
        if scene_id not in scene_groups[camera_type]:
            scene_groups[camera_type][scene_id] = {"short": [], "long": []}

        scene_groups[camera_type][scene_id][file_type].append(
            {
                "file_path": file_data["file_path"],
                "exposure_time": exposure_time,
                "iso": file_data["iso"],
                "aperture": file_data["aperture"],
                "split": file_data["split"],
            }
        )

    # Create metadata for all files with scene-level information
    for camera_type, scenes in scene_groups.items():
        for scene_id, files_by_type in scenes.items():
            # Extract all exposure times for this scene
            short_exposures = [f["exposure_time"] for f in files_by_type["short"]]
            long_exposures = [f["exposure_time"] for f in files_by_type["long"]]

            # Find long partner (all short exposures pair with the one long exposure)
            long_partner = (
                files_by_type["long"][0]["file_path"] if files_by_type["long"] else None
            )

            # Create metadata for each file
            for exposure_type in ["short", "long"]:
                for file_data in files_by_type[exposure_type]:
                    file_path = file_data["file_path"]
                    exposure_time_str = file_data["exposure_time"]

                    metadata = {
                        "file_path": file_path,
                        "split": file_data["split"],
                        "scene_id": scene_id,
                        "file_type": exposure_type,
                        "camera_type": camera_type,
                        "exposure_time": exposure_time_str,
                        "iso": file_data["iso"],
                        "aperture": file_data["aperture"],
                        # Scene-level information
                        "all_short_exposures": short_exposures,
                        "all_long_exposures": long_exposures,
                        "long_partner": long_partner,
                    }

                    file_info[file_path] = metadata

    logger.info(f"Loaded information for {len(file_info)} files from split files")
    return file_info


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
        # Input validation
        if image is None:
            raise InvalidTileConfiguration("image cannot be None")

        if image.size == 0:
            raise InvalidTileConfiguration("image cannot be empty")

        # Verify image format (3D array)
        if len(image.shape) != 3:
            raise InvalidTileConfiguration(
                f"Expected 3D image shape (C, H, W), got {len(image.shape)}D: {image.shape}"
            )

        # Accept valid channel counts (RGB=3, Sony Bayer=4, Fuji X-Trans=9)
        valid_channels = [3, 4, 9]
        if image.shape[0] not in valid_channels:
            raise InvalidTileConfiguration(
                f"Expected {valid_channels} channels, got {image.shape[0]}. "
                f"Valid formats: RGB (3), Sony Bayer (4), Fuji X-Trans (9)"
            )

        # Validate grid configuration
        if rows <= 0 or cols <= 0:
            raise InvalidTileConfiguration(
                f"Grid dimensions must be positive: rows={rows}, cols={cols}"
            )

        if tile_size <= 0:
            raise InvalidTileConfiguration(
                f"tile_size must be positive, got {tile_size}"
            )

        C, H, W = image.shape

        # Verify image is large enough for requested tiles
        if H < tile_size or W < tile_size:
            raise InvalidTileConfiguration(
                f"Image ({H}×{W}) is smaller than tile_size ({tile_size}×{tile_size}). "
                f"Cannot extract tiles from image smaller than a single tile."
            )

        # Calculate stride
        stride_h = int(np.floor((H - tile_size) / (rows - 1))) if rows > 1 else 0
        stride_w = int(np.floor((W - tile_size) / (cols - 1))) if cols > 1 else 0

        tiles = []
        for row in range(rows):
            for col in range(cols):
                # Calculate tile position with even stride
                # Ensure the last tile ends at the image boundary
                y_start = H - tile_size if row == rows - 1 else row * stride_h
                x_start = W - tile_size if col == cols - 1 else col * stride_w

                y_end = y_start + tile_size
                x_end = x_start + tile_size

                # Extract tile
                tile_data = image[:, y_start:y_end, x_start:x_end]

                # Verify no padding needed
                assert tile_data.shape[-2:] == (
                    tile_size,
                    tile_size,
                ), f"Tile size mismatch: expected {tile_size}×{tile_size}, got {tile_data.shape[-2:]}"

                # Create tile info object
                tile_info = TileInfo(
                    tile_data=tile_data,
                    grid_position=(col, row),
                    image_position=(x_start, y_start),
                    valid_ratio=1.0,  # All pixels are valid (no padding!)
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


# ============================================================================
# Raw Image Demosaicing Functions (from SID dataset authors)
# ============================================================================


def _validate_raw_input(
    raw_image_data: np.ndarray, max_value: float, sensor_name: str, divisor: int
) -> Tuple[int, int]:
    """Validate raw image input for sensor processing."""
    if raw_image_data is None:
        raise InvalidRawDataError("raw_image_data cannot be None")

    if raw_image_data.size == 0:
        raise InvalidRawDataError("raw_image_data cannot be empty")

    if len(raw_image_data.shape) != 2:
        raise InvalidRawDataError(
            f"Expected 2D array (H, W), got shape {raw_image_data.shape}"
        )

    H, W = raw_image_data.shape

    if H % divisor != 0 or W % divisor != 0:
        raise InvalidRawDataError(
            f"{sensor_name} dimensions must be divisible by {divisor}. "
            f"Got {H}×{W}. This is a physics requirement for CFA patterns."
        )

    if max_value <= 0:
        raise InvalidRawDataError(
            f"max_value must be positive (got {max_value}). "
            f"Typically 16383 for 14-bit {sensor_name} sensors."
        )

    return H, W


def pack_raw_sony(raw_image_data: np.ndarray, max_value: float) -> np.ndarray:
    """
    Pack Sony Bayer pattern raw image to 4 channels (RGGB).

    SENSOR PHYSICS & DESIGN RATIONALE:

    Bayer CFA Pattern (2×2 repeating unit):
        R  G1
        G2 B



    Implementation based on Learning-to-See-in-the-Dark (Chen et al., CVPR 2018).
    Reference: https://github.com/cchen156/Learning-to-See-in-the-Dark

    Args:
        raw_image_data: Raw Bayer image as (H, W) uint16 array
        max_value: Maximum pixel value for normalization (e.g., 16383 for 14-bit ADC)
                  MUST be computed from entire dataset to ensure consistent statistics

    Returns:
        Packed image as (H/2, W/2, 4) float32 array
        Channels in order: [R, G1, G2, B]
        All values normalized to [0, 1]

    Raises:
        ValueError: If input validation fails
    """
    H, W = _validate_raw_input(raw_image_data, max_value, "Sony", 2)

    # Convert to float32 and subtract black level
    im = raw_image_data.astype(np.float32)
    black_level = BLACK_LEVELS["sony"]
    white_level = SENSOR_RANGES["sony"]["max"]
    im = np.maximum(im - black_level, 0) / (
        white_level - black_level
    )  # subtract the black level

    # Expand to 3D for channel concatenation
    im = np.expand_dims(im, axis=2)

    # Pack Bayer pattern into 4 channels (RGGB)
    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],  # R channel
            im[0:H:2, 1:W:2, :],  # G1 channel
            im[1:H:2, 1:W:2, :],  # B channel
            im[1:H:2, 0:W:2, :],  # G2 channel
        ),
        axis=2,
    )

    return out


def pack_raw_fuji(raw_image_data: np.ndarray, max_value: float) -> np.ndarray:
    """
    Pack Fuji X-Trans pattern raw image to 9 channels.

    SENSOR PHYSICS & DESIGN RATIONALE:

    X-Trans CFA Pattern (6×6 repeating unit):
    ──────────────────────────────────────────
    R  G  B  R  G  B
    G  B  R  G  B  R
    B  R  G  B  R  G
    R  G  B  R  G  B
    G  B  R  G  B  R
    B  R  G  B  R  G


    Implementation based on Learning-to-See-in-the-Dark (Chen et al., CVPR 2018).
    Reference: https://github.com/cchen156/Learning-to-See-in-the-Dark

    Args:
        raw_image_data: Raw X-Trans image as (H, W) uint16 array
        max_value: Maximum pixel value for normalization (e.g., 16383 for 14-bit ADC)
                  MUST be computed from entire dataset to ensure consistent statistics

    Returns:
        Packed image as (H/3, W/3, 9) float32 array
        All values normalized to [0, 1]
        Preserves spatial arrangement of X-Trans 6×6 unit

    Raises:
        ValueError: If input validation fails
    """
    # Validate input but allow cropping for Fuji
    if raw_image_data is None:
        raise InvalidRawDataError("raw_image_data cannot be None")
    if raw_image_data.size == 0:
        raise InvalidRawDataError("raw_image_data cannot be empty")
    if len(raw_image_data.shape) != 2:
        raise InvalidRawDataError(
            f"Expected 2D array (H, W), got shape {raw_image_data.shape}"
        )
    if max_value <= 0:
        raise InvalidRawDataError(f"max_value must be positive (got {max_value})")

    H, W = raw_image_data.shape

    # Convert to float32 and subtract black level
    im = raw_image_data.astype(np.float32)
    black_level = BLACK_LEVELS["fuji"]
    white_level = SENSOR_RANGES["fuji"]["max"]
    im = np.maximum(im - black_level, 0) / (
        white_level - black_level
    )  # subtract the black level

    # Crop to ensure dimensions are divisible by 6 (required for X-Trans pattern)
    H = (H // 6) * 6
    W = (W // 6) * 6
    im = im[:H, :W]

    out = np.zeros((H // 3, W // 3, 9))

    # 0 R
    out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

    # 1 G
    out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

    # 2 B
    out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

    # 3 R
    out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

    # 4 B
    out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
    out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
    out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
    out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

    out[:, :, 5] = im[1:H:3, 0:W:3]
    out[:, :, 6] = im[1:H:3, 1:W:3]
    out[:, :, 7] = im[2:H:3, 0:W:3]
    out[:, :, 8] = im[2:H:3, 1:W:3]

    return out


def pack_raw_to_rgb(
    raw_image_data: np.ndarray, sensor_type: str, max_value: float
) -> np.ndarray:
    """
    Pack raw image data and convert to RGB format.

    Pipeline:
    1. Pack raw Bayer/X-Trans data to sensor-specific channels
    2. Demosaic to RGB (3 channels)

    Args:
        raw_image_data: Raw Bayer/X-Trans image data (H, W)
        sensor_type: 'sony' or 'fuji'
        max_value: Maximum pixel value for normalization

    Returns:
        RGB image (3, H', W') normalized to [0, 1]
        - Sony: (3, H*2, W*2) - upsampled by 2x from packed
        - Fuji: (3, H*3, W*3) - upsampled by 3x from packed
    """
    from scipy.ndimage import zoom

    # Pack raw data to sensor-specific format
    if sensor_type == "sony":
        packed = pack_raw_sony(raw_image_data, max_value)  # (H/2, W/2, 4)

        # Convert from HWC to work with channels
        img_hwc = packed  # Already in HWC format

        # Separate channels from packed format
        R_sampled = img_hwc[:, :, 0]  # Red samples
        G1_sampled = img_hwc[:, :, 1]  # Green samples (row 0)
        G2_sampled = img_hwc[:, :, 2]  # Green samples (row 1)
        B_sampled = img_hwc[:, :, 3]  # Blue samples

        # Average G1 and G2 to get single G channel
        G_sampled = (G1_sampled + G2_sampled) / 2.0

        # Upsample each channel by 2x to get full resolution
        R_upsampled = zoom(R_sampled, (2, 2), order=1, mode="nearest")
        G_upsampled = zoom(G_sampled, (2, 2), order=1, mode="nearest")
        B_upsampled = zoom(B_sampled, (2, 2), order=1, mode="nearest")

        # Stack to form RGB (3, H*2, W*2)
        rgb = np.stack([R_upsampled, G_upsampled, B_upsampled], axis=0)

    elif sensor_type == "fuji":
        packed = pack_raw_fuji(raw_image_data, max_value)  # (H/3, W/3, 9)

        # Convert from HWC to work with channels
        img_hwc = packed  # Already in HWC format

        # Extract and average RGB components from channels 0-4
        # Channels 0-4 contain primary R, G, B samples from 6x6 X-Trans pattern
        # Channel 0: R, Channel 1: G, Channel 2: B, Channel 3: R, Channel 4: B
        # Channels 5-8 are additional subsampled data; used conservatively here
        R_components = img_hwc[:, :, 0] + img_hwc[:, :, 3]
        G_components = img_hwc[:, :, 1]
        B_components = img_hwc[:, :, 2] + img_hwc[:, :, 4]

        R_avg = R_components / 2.0
        G_avg = G_components
        B_avg = B_components / 2.0

        # Upsample to full resolution (X-Trans is packed at 1/3 resolution)
        R_upsampled = zoom(R_avg, (3, 3), order=1, mode="nearest")
        G_upsampled = zoom(G_avg, (3, 3), order=1, mode="nearest")
        B_upsampled = zoom(B_avg, (3, 3), order=1, mode="nearest")

        # Stack to form RGB (3, H*3, W*3)
        rgb = np.stack([R_upsampled, G_upsampled, B_upsampled], axis=0)

    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    return rgb


def _get_metadata_root_path(file_path: str) -> Path:
    """
    Get the root metadata path (parent.parent.parent) for a given file path.

    This is used to locate sensor metadata files stored at the dataset root level.

    Args:
        file_path: Path to any file in the dataset

    Returns:
        Path to the dataset root directory
    """
    return Path(file_path).parent.parent.parent


def demosaic_raw_to_rgb(raw_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Demosaic raw image to RGB format.

    Rationale:
    - Preserves original sensor data structure (Bayer/X-Trans patterns)
    - Maintains Poisson-Gaussian noise statistics critical for denoising research
    - Allows reconstruction if needed

    Args:
        raw_path: Path to .ARW (Sony) or .RAF (Fuji) file

    Returns:
        Tuple of (image, metadata)
        - image: (C, H, W) numpy array in float32, normalized to [0, 1]
        - metadata: Dictionary with camera info, white balance, etc.
    """
    import rawpy

    try:
        # Detect sensor type using SensorDetector
        from preprocessing.sensor_detector import SensorDetector, SensorType

        sensor_type = SensorDetector.detect(raw_path)

        # Get max value from config for this sensor type
        max_value = SENSOR_RANGES[sensor_type.value]["max"]

        with rawpy.imread(raw_path) as raw:
            # Extract raw Bayer/X-Trans data
            raw_image_data = raw.raw_image_visible

            # Pack raw data using sensor-specific functions
            # This calls pack_raw_sony or pack_raw_fuji internally based on sensor_type
            rgb_image = pack_raw_to_rgb(raw_image_data, sensor_type.value, max_value)

            # Extract metadata
            metadata = extract_raw_metadata(raw, raw_path, rgb_image, sensor_type.value)

            return rgb_image, metadata

    except Exception as e:
        logger.error(f"Error demosaicing raw file {raw_path}: {e}")
        return None, None


# ============================================================================
# Metadata Extraction Functions
# ============================================================================


def _parse_filename_metadata(
    file_path: str,
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Parse scene_id, camera_model, and exposure_time from filename."""
    from preprocessing.sensor_detector import SensorDetector

    file_path_obj = Path(file_path)
    filename_parts = file_path_obj.stem.split("_")

    # Extract scene_id (first part)
    scene_id = filename_parts[0] if len(filename_parts) >= 3 else None

    # Extract exposure_time (third part, remove 's' suffix if present)
    exposure_time = None
    if len(filename_parts) >= 3:
        try:
            exposure_time = float(filename_parts[2].rstrip("s"))
        except ValueError:
            pass

    # Detect camera model
    camera_model = None
    try:
        sensor_type_detected = SensorDetector.detect(file_path)
        camera_model = sensor_type_detected.value
    except Exception:
        pass

    return scene_id, camera_model, exposure_time


def _create_base_metadata(
    file_path: str, sensor_type: str, pair_info: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Create base metadata dictionary with common fields."""
    scene_id, camera_model, exposure_time = _parse_filename_metadata(file_path)

    metadata = {
        "file_path": file_path,
        "sensor_type": sensor_type,
        "camera_model": camera_model,
        "exposure_time": exposure_time,
        "scene_id": scene_id,
    }

    if pair_info:
        metadata.update(pair_info)

    return metadata


def _safe_tolist(value) -> Any:
    """Convert numpy array to list if possible, otherwise return original value."""
    return value.tolist() if hasattr(value, "tolist") else value


def extract_raw_metadata(
    raw, file_path: str, packed_image: np.ndarray, sensor_type: str
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from raw file using rawpy.

    Args:
        raw: rawpy object with opened raw file
        file_path: Path to the raw file
        packed_image: Packed image array (for channel count)
        sensor_type: Sensor type ('sony' or 'fuji')

    Returns:
        Dictionary containing all metadata from the raw file
    """
    # Detect Bayer pattern
    pattern = raw.raw_pattern
    pattern_shape = pattern.shape

    # Extract black level and white level
    black_level = np.array(raw.black_level_per_channel)
    white_level = raw.white_level

    # Create base metadata
    metadata = _create_base_metadata(file_path, sensor_type)

    # Add raw file-specific fields
    metadata.update(
        {
            # Basic camera settings (available in rawpy)
            "black_level": black_level.tolist(),
            "white_level": int(white_level),
            # Image dimensions and properties (available in rawpy)
            "raw_height": raw.sizes.raw_height,
            "raw_width": raw.sizes.raw_width,
            "height": raw.sizes.height,
            "width": raw.sizes.width,
            "pixel_aspect": raw.sizes.pixel_aspect,
            # Color information (available in rawpy)
            "color_desc": raw.color_desc.decode() if raw.color_desc else None,
            "num_colors": raw.num_colors,
            "raw_pattern": pattern.tolist(),
            "pattern_shape": pattern_shape,
            # Sensor-specific information
            "processing": "packed_raw_no_demosaic",  # Indicates packed raw format
            "channels": packed_image.shape[
                0
            ],  # Number of channels (4 for Sony, 9 for Fuji)
            # White balance (available in rawpy)
            "camera_whitebalance": _safe_tolist(raw.camera_whitebalance),
            "daylight_whitebalance": _safe_tolist(raw.daylight_whitebalance),
            # Color matrices (available in rawpy)
            "color_matrix": _safe_tolist(raw.color_matrix),
            "rgb_xyz_matrix": _safe_tolist(raw.rgb_xyz_matrix),
            # Additional properties (available in rawpy)
            "raw_type": str(raw.raw_type),
            # Note: ISO and aperture will be extracted from SID split files
        }
    )

    return metadata


# ============================================================================
# Long-Short Exposure Pair Finding Functions
# ============================================================================


def _create_scene_dict() -> Dict[str, Any]:
    """Create a defaultdict for storing scene files."""
    return defaultdict(lambda: {"long": None, "short": []})


def _select_short_exposure(
    short_files: List[Tuple[str, str]], camera_type: str
) -> Optional[str]:
    """Select the appropriate short exposure file based on camera type."""
    if camera_type == "sony":
        # Try to find 0.04s exposure first
        return next((p for e, p in short_files if e == "0.04s"), None)
    # Fallback: use first sorted file
    return sorted(short_files)[0][1] if short_files else None


def find_photography_pairs(
    all_photo_files: List[Path],
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Select long/short exposure pairs for photography (SID dataset).

    Pairing strategy:
    - Long: Long exposure files (in /long/ directory) - ground truth
    - Short: First short exposure file per scene (0.04s for Sony, first short for Fuji) - input

    Args:
        all_photo_files: List of all .ARW and .RAF file paths

    Returns:
        Tuple of (selected_file_paths, pair_metadata) where:
        - selected_file_paths: List of selected file paths (long + short pairs)
        - pair_metadata: Dict mapping file_path -> metadata with pair information
    """
    # Group files by camera type and scene
    camera_scenes = {cam: _create_scene_dict() for cam in CAMERA_CONFIGS.keys()}

    # Parse and group all files
    for file_path in all_photo_files:
        # Use SensorDetector to determine camera type
        from preprocessing.sensor_detector import SensorDetector

        try:
            sensor_type = SensorDetector.detect(str(file_path))
            camera_type = sensor_type.value
        except Exception:
            continue

        if not camera_type:
            continue

        parts = Path(file_path).stem.split("_")
        if len(parts) >= 3:
            scene_id, exposure = parts[0], parts[2]
            data_type = determine_data_type(str(file_path), "photography")

            if data_type == "long":
                camera_scenes[camera_type][scene_id]["long"] = str(file_path)
            elif data_type == "short":
                camera_scenes[camera_type][scene_id]["short"].append(
                    (exposure, str(file_path))
                )

    # Select pairs and create metadata
    selected_files, pair_metadata = [], {}

    for camera_type, scenes in camera_scenes.items():
        for scene_id, files in scenes.items():
            if not (files["long"] and files["short"]):
                continue

            short_file = _select_short_exposure(files["short"], camera_type)
            if not short_file:
                continue

            # Create pair metadata
            long_file, pair_id = files["long"], f"{camera_type}_{scene_id}"
            long_pair_info = {
                "pair_id": pair_id,
                "pair_type": "long",
                "pair_partner": short_file,
            }
            short_pair_info = {
                "pair_id": pair_id,
                "pair_type": "short",
                "pair_partner": long_file,
            }

            # Add files and metadata
            selected_files.extend([long_file, short_file])
            pair_metadata[long_file] = _create_base_metadata(
                long_file, camera_type, long_pair_info
            )
            pair_metadata[short_file] = _create_base_metadata(
                short_file, camera_type, short_pair_info
            )

    return selected_files, pair_metadata


def get_sensor_specific_range(
    file_path: str, sensor_ranges: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Get sensor-specific range for a given file path.

    Args:
        file_path: Path to the file
        sensor_ranges: Dictionary of sensor ranges

    Returns:
        Dictionary with min and max values for the specific sensor
    """
    file_ext = Path(file_path).suffix.upper()

    # Map file extension to sensor type
    sensor_type = None
    for st, config in CAMERA_CONFIGS.items():
        if file_ext == config["extension"].upper():
            sensor_type = st
            break

    # Return sensor-specific range if available, otherwise use fallback
    if sensor_type and sensor_type in sensor_ranges:
        return sensor_ranges[sensor_type]
    else:
        # Ultimate fallback
        return {"min": 0.0, "max": 16383.0}


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


def _add_optional_fields(
    target: Dict[str, Any], source: Dict[str, Any], *keys: str
) -> None:
    """Add optional fields from source to target if they exist."""
    for key in keys:
        if key in source and source[key] is not None:
            target[key] = source[key]


def create_file_metadata(
    file_path: str,
    metadata: Dict[str, Any],
    sid_info: Dict[str, Any],
    pair_info: Optional[Dict[str, str]] = None,
    domain_range: Optional[Dict[str, float]] = None,
    image_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Create consolidated file-level metadata without duplication."""
    file_metadata = {
        "file_path": file_path,
        "split": sid_info.get("split"),
        "scene_id": sid_info.get("scene_id"),
        "data_type": sid_info.get("file_type"),
        "camera_model": metadata.get("camera_model"),
        "exposure_time": metadata.get("exposure_time"),
        "camera_type": sid_info.get("camera_type"),
    }

    # Add optional fields conditionally
    _add_optional_fields(
        file_metadata,
        sid_info,
        "iso",
        "aperture",
        "all_short_exposures",
        "all_long_exposures",
        "long_partner",
    )

    if domain_range:
        file_metadata["domain_range"] = domain_range
    if image_stats:
        file_metadata["image_stats"] = image_stats
    if pair_info:
        file_metadata.update(pair_info)

    return file_metadata
