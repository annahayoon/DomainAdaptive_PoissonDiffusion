#!/usr/bin/env python3
"""
Configuration constants for the preprocessing pipeline.

This module contains all configuration values including tile sizes, target grids,
domain-specific normalization ranges, and file paths.

Environment Variables:
    DATA_PATH: Root path to data directory (default: ./data)
"""

import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# Base data paths - configurable via environment variable
BASE_DATA_PATH = Path(os.environ.get("DATA_PATH", "./data"))

# Validate paths exist on import
if not BASE_DATA_PATH.exists():
    logger.warning(
        f"DATA_PATH not found: {BASE_DATA_PATH}\n"
        f"To fix, set environment variable: export DATA_PATH=/path/to/data\n"
        f"Or pass --data_path argument to the pipeline script"
    )

RAW_DATA_PATH = BASE_DATA_PATH / "raw"
SONY_PATH = RAW_DATA_PATH / "SID" / "Sony"
FUJI_PATH = RAW_DATA_PATH / "SID" / "Fuji"

# Tile size for all domains
TILE_SIZE = 256

# TILE CONFIGURATION RATIONALE:
#
# Tile grids are domain-specific to maximize coverage while avoiding excessive padding.
# Each tile is 256×256 pixels (standard for diffusion models).
#
# Sony cameras (SID dataset):
# - Typical image size: ~6144×4096 pixels
# - Target grid: 12×18 tiles = 216 tiles per image
# - Grid size determines: height_per_tile = 4096 / 12 ≈ 341, width_per_tile = 6144 / 18 ≈ 341
# - We extract 256×256 tiles, leaving small margins
#
# Fuji cameras (SID dataset):
# - Typical image size: ~8192×5376 pixels
# - Target grid: 16×24 tiles = 384 tiles per image
# - Grid size determines: height_per_tile = 5376 / 16 ≈ 336, width_per_tile = 8192 / 24 ≈ 341
TILE_CONFIGS = {
    "sony": {
        "tile_size": 256,
        "target_tiles": 216,
        "target_grid": (12, 18),  # 12 rows, 18 columns - Sony grid
    },
    "fuji": {
        "tile_size": 256,
        "target_tiles": 384,
        "target_grid": (16, 24),  # 16 rows, 24 columns - Fuji grid
    },
}

# SENSOR RANGES RATIONALE:
#
# Sensor-specific normalization ranges for raw image processing.
# These represent the [min, max] pixel values for 14-bit ADC sensors.
#
# The normalization pipeline is:
#   1. Raw ADC: [sensor_min, sensor_max]
#   2. Subtract black level: x - black_level
#   3. Normalize to [0, 1]: (x - black_level) / (max - black_level)
#   4. Scale to [-1, 1]: 2 * x - 1  (standard for diffusion models)
#
# Sensor-specific values:
# - Sony: black level 512, white level 16383 (14-bit ADC)
# - Fuji: black level 1024, white level 16383 (14-bit ADC)
# - Using correct black level preserves sensor noise statistics
SENSOR_RANGES = {
    "sony": {"min": 0.0, "max": 16383.0},  # White level for 14-bit ADC
    "fuji": {"min": 0.0, "max": 16383.0},  # White level for 14-bit ADC
}

# Black levels for each sensor (minimum ADC value)
BLACK_LEVELS = {
    "sony": 512,  # Sony black level
    "fuji": 1024,  # Fuji black level
}

# CAMERA CONFIGURATION RATIONALE:
#
# Camera-specific settings for raw image processing.
# Defines file extensions, directory names, and channel counts for packed raw data.
#
# Sony cameras (SID dataset):
# - Extension: .ARW (Sony raw format)
# - Base directory: Sony/
# - Packed channels: 4 (RGGB Bayer pattern)
#
# Fuji cameras (SID dataset):
# - Extension: .RAF (Fuji raw format)
# - Base directory: Fuji/
# - Packed channels: 9 (X-Trans pattern)
CAMERA_CONFIGS = {
    "sony": {
        "extension": ".ARW",
        "base_dir": "Sony",
        "channels": 4,
    },
    "fuji": {
        "extension": ".RAF",
        "base_dir": "Fuji",
        "channels": 9,
    },
}
