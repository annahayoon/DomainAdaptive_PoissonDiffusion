#!/usr/bin/env python3
"""
Configuration constants for sampling and restoration.

This module imports common configs from config.config and defines
sample-specific constants for visualization and batch processing.
"""

# Import common configs from config.config to avoid duplication
from config.config import (
    DEFAULT_CONSERVATIVE_FACTOR,
    DEFAULT_KAPPA,
    DEFAULT_NUM_STEPS,
    DEFAULT_RHO,
    DEFAULT_SIGMA_MAX,
    DEFAULT_SIGMA_MIN,
    DEFAULT_SIGMA_R,
    DEFAULT_TAU,
    SENSOR_NAME_MAPPING,
    SUPPORTED_SENSORS,
    TILE_SIZE,
)

# SENSOR_RANGES was moved to core.sensor_config - try to import it, but make it optional
try:
    from config.config import SENSOR_RANGES
except ImportError:
    # SENSOR_RANGES is not available in config.config (moved to core.sensor_config)
    # Provide a default empty dict for backward compatibility
    SENSOR_RANGES = {}

# Re-export for backward compatibility
DEFAULT_SENSOR_RANGES = SENSOR_RANGES
DEFAULT_TILE_SIZE = TILE_SIZE

# Sample-specific constants
RESERVED_TILE_KEYS = {"short", "long"}
DEFAULT_STRATIFIED_ALPHA = 0.05
DEFAULT_CHUNK_SIZE = 32
DEFAULT_BATCH_CHUNK_SIZE = 8
# Increased from 16 to 64 to better utilize A40 GPU (46GB total, ~42GB free)
MAX_CHUNK_SIZE = 64
DEFAULT_EXPOSURE_RATIO_ERROR_THRESHOLD = 20.0

DEFAULT_DPI = 150
HIGH_DPI = 200
DEFAULT_FIGSIZE_WIDTH = 3.0
DEFAULT_FIGSIZE_HEIGHT = 9
SCENE_FIGSIZE_WIDTH = 20
SCENE_FIGSIZE_HEIGHT = 6

MIN_SAMPLES_PER_BIN = 20
MIN_VALID_BINS = 10
MIN_VALID_PIXELS = 10
DEFAULT_NUM_SAMPLES = 50000
DEFAULT_NUM_BINS = 10
DATA_RANGE_MIN = -1.0
DATA_RANGE_MAX = 1.0
