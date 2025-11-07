"""Sensor (camera) configuration constants and utilities.

This module provides sensor-specific constants like sensor ranges,
black levels, and other camera configurations.
"""

from typing import Dict

# Sensor-specific physical ranges (ADU units)
SENSOR_RANGES: Dict[str, Dict[str, float]] = {
    "sony": {"min": 0.0, "max": 16383.0},
    "fuji": {"min": 0.0, "max": 16383.0},
}

# Sensor-specific black levels (ADU)
BLACK_LEVELS: Dict[str, float] = {
    "sony": 512.0,
    "fuji": 1024.0,
}

# Sensor-specific white levels (ADU)
WHITE_LEVELS: Dict[str, float] = {
    "sony": 16383.0,
    "fuji": 16383.0,
}


def get_sensor_range(sensor: str) -> Dict[str, float]:
    """Get physical range for a sensor.

    Args:
        sensor: Sensor name (sony, fuji)

    Returns:
        Dictionary with 'min' and 'max' keys

    Raises:
        ValueError: If sensor is not recognized
    """
    sensor_lower = sensor.lower().strip()
    if sensor_lower not in SENSOR_RANGES:
        raise ValueError(
            f"Unknown sensor '{sensor}'. Supported sensors: {list(SENSOR_RANGES.keys())}"
        )
    return SENSOR_RANGES[sensor_lower].copy()


def get_sensor_black_level(sensor: str) -> float:
    """Get black level for a sensor.

    Args:
        sensor: Sensor name (sony, fuji)

    Returns:
        Black level in ADU

    Raises:
        ValueError: If sensor is not recognized
    """
    sensor_lower = sensor.lower().strip()
    if sensor_lower not in BLACK_LEVELS:
        raise ValueError(
            f"Unknown sensor '{sensor}'. Supported sensors: {list(BLACK_LEVELS.keys())}"
        )
    return BLACK_LEVELS[sensor_lower]


def get_sensor_white_level(sensor: str) -> float:
    """Get white level for a sensor.

    Args:
        sensor: Sensor name (sony, fuji)

    Returns:
        White level in ADU

    Raises:
        ValueError: If sensor is not recognized
    """
    sensor_lower = sensor.lower().strip()
    if sensor_lower not in WHITE_LEVELS:
        raise ValueError(
            f"Unknown sensor '{sensor}'. Supported sensors: {list(WHITE_LEVELS.keys())}"
        )
    return WHITE_LEVELS[sensor_lower]


def get_sensor_config(sensor: str) -> Dict[str, float]:
    """Get complete sensor configuration (range, black level, white level).

    Args:
        sensor: Sensor name (sony, fuji)

    Returns:
        Dictionary with 'min', 'max', 'black_level', and 'white_level' keys

    Raises:
        ValueError: If sensor is not recognized
    """
    sensor_range = get_sensor_range(sensor)
    return {
        "min": sensor_range["min"],
        "max": sensor_range["max"],
        "black_level": get_sensor_black_level(sensor),
        "white_level": get_sensor_white_level(sensor),
    }
