"""Normalization and denormalization utilities."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

# Try to import sensor_config, but make it optional
try:
    from .sensor_config import (
        get_sensor_black_level,
        get_sensor_config,
        get_sensor_white_level,
    )

    _SENSOR_CONFIG_AVAILABLE = True
except ImportError:
    _SENSOR_CONFIG_AVAILABLE = False

    # Provide stub functions if sensor_config is not available
    def get_sensor_config(sensor: str) -> Dict:
        raise ImportError(
            "sensor_config module not available. "
            "Please provide explicit black_level and white_level parameters."
        )

    def get_sensor_black_level(sensor: str) -> float:
        raise ImportError("sensor_config module not available")

    def get_sensor_white_level(sensor: str) -> float:
        raise ImportError("sensor_config module not available")


def _extract_range_parameters(
    sensor: Optional[str] = None,
    black_level: Optional[float] = None,
    white_level: Optional[float] = None,
    max_value: Optional[float] = None,
    range_dict: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """Extract range_min and range_max from various input formats.

    This helper function consolidates the duplicate range extraction logic
    used in both normalize_physical_to_normalized and denormalize_to_physical.

    Args:
        sensor: Sensor name (sony, fuji) - automatically gets black/white levels
        black_level: Explicit black level value
        white_level: Explicit white level value
        max_value: Maximum value for [0, max_value] range
        range_dict: Dictionary with 'min' and 'max' keys

    Returns:
        Tuple of (range_min, range_max)

    Raises:
        ValueError: If no valid range parameters are provided
    """
    if sensor is not None:
        # Use sensor configuration
        if not _SENSOR_CONFIG_AVAILABLE:
            raise ImportError(
                "sensor_config module not available. "
                "Please provide explicit black_level and white_level parameters instead of sensor name."
            )
        sensor_cfg = get_sensor_config(sensor)
        range_min = sensor_cfg["black_level"]
        range_max = sensor_cfg["white_level"]
    elif range_dict is not None:
        range_min = range_dict["min"]
        range_max = range_dict["max"]
    elif black_level is not None and white_level is not None:
        range_min = black_level
        range_max = white_level
    elif max_value is not None:
        range_min = 0.0
        range_max = max_value
    else:
        raise ValueError(
            "Must provide one of: sensor, (black_level, white_level), "
            "max_value, or range_dict"
        )

    return range_min, range_max


def denormalize_to_physical(
    tensor: torch.Tensor,
    sensor: Optional[str] = None,
    black_level: Optional[float] = None,
    white_level: Optional[float] = None,
    max_value: Optional[float] = None,
    range_dict: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Convert tensor from normalized space to physical units.

    Supports multiple input formats:
    1. Sensor-based: Use sensor name (sony, fuji) - automatically gets black/white levels
    2. Explicit: Use black_level and white_level directly
    3. Simple: Use max_value (for [0, max_value] range)
    4. Custom: Use range_dict with 'min' and 'max' keys
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    # Use convert_range to avoid duplicating the conversion logic
    tensor_norm = convert_range(tensor, from_range="[-1,1]", to_range="[0,1]")
    tensor_norm = torch.clamp(tensor_norm, 0, 1)

    range_min, range_max = _extract_range_parameters(
        sensor=sensor,
        black_level=black_level,
        white_level=white_level,
        max_value=max_value,
        range_dict=range_dict,
    )

    sensor_range = range_max - range_min
    tensor_phys = tensor_norm * sensor_range + range_min

    return tensor_phys.cpu().numpy()


def normalize_physical_to_normalized(
    physical: Union[torch.Tensor, np.ndarray],
    sensor: Optional[str] = None,
    black_level: Optional[float] = None,
    white_level: Optional[float] = None,
    max_value: Optional[float] = None,
    range_dict: Optional[Dict[str, float]] = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Convert from physical units to normalized [-1, 1] space.

    Supports multiple input formats (same as denormalize_to_physical).
    """
    if isinstance(physical, np.ndarray):
        physical = torch.from_numpy(physical)

    range_min, range_max = _extract_range_parameters(
        sensor=sensor,
        black_level=black_level,
        white_level=white_level,
        max_value=max_value,
        range_dict=range_dict,
    )

    sensor_range = range_max - range_min
    tensor_01 = (physical - range_min) / (sensor_range + epsilon)
    tensor_01 = torch.clamp(tensor_01, 0.0, 1.0)
    # Use convert_range to avoid duplicating the conversion logic
    tensor_norm = convert_range(tensor_01, from_range="[0,1]", to_range="[-1,1]")

    return tensor_norm


def convert_range(
    tensor: torch.Tensor, from_range: str = "[-1,1]", to_range: str = "[0,1]"
) -> torch.Tensor:
    """Convert tensor between normalized ranges."""
    if from_range == "[-1,1]" and to_range == "[0,1]":
        return (tensor + 1.0) / 2.0
    elif from_range == "[0,1]" and to_range == "[-1,1]":
        return tensor * 2.0 - 1.0
    else:
        raise ValueError(f"Unsupported range conversion: {from_range} -> {to_range}")


def reverse_normalize_from_neg_one_to_raw(
    value_norm: np.ndarray, sensor_type: str
) -> np.ndarray:
    """Reverse normalize from [-1,1] back to raw pixel values.

    This is different from reverse_normalize_to_raw which works with [0,1] range.

    Args:
        value_norm: Normalized values in [-1, 1] range
        sensor_type: Sensor type ('sony' or 'fuji')

    Returns:
        Raw pixel values
    """
    # Use convert_range to avoid duplicating the conversion logic
    value_tensor = torch.from_numpy(value_norm)
    value_01_tensor = convert_range(value_tensor, from_range="[-1,1]", to_range="[0,1]")
    value_01 = np.clip(value_01_tensor.numpy(), 0.0, 1.0)

    if not _SENSOR_CONFIG_AVAILABLE:
        raise ImportError(
            "sensor_config module not available. "
            "Cannot use reverse_normalize_from_neg_one_to_raw without sensor_config."
        )
    black_level = get_sensor_black_level(sensor_type)
    white_level = get_sensor_white_level(sensor_type)
    return value_01 * (white_level - black_level) + black_level
