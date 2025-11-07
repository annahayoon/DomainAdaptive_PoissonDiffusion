#!/usr/bin/env python3
"""
Centralized sensor detection and configuration.

This module provides unified sensor detection and configuration management,
replacing scattered detection logic throughout the codebase.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Supported sensor types"""

    SONY = "sony"
    FUJI = "fuji"


class SensorDetector:
    """Centralized sensor detection and configuration management."""

    # Sensor configurations - central source of truth
    CONFIGS = {
        SensorType.SONY: {
            "extension": ".ARW",
            "base_dir": "Sony",
            "channels": 4,
            "tile_grid": (12, 18),
            "bayer_pattern": "RGGB",
            "description": "Sony Alpha 7",
        },
        SensorType.FUJI: {
            "extension": ".RAF",
            "base_dir": "Fuji",
            "channels": 9,
            "tile_grid": (16, 24),
            "bayer_pattern": "X-Trans",
            "description": "Fuji X-Series",
        },
    }

    @classmethod
    def detect(cls, file_path: str) -> SensorType:
        """Detect sensor type from file path.

        Args:
            file_path: Path to raw image file

        Returns:
            SensorType.SONY or SensorType.FUJI

        Raises:
            SensorDetectionError: If sensor cannot be detected
        """
        from core.utils.sensor_utils import SensorDetectionError

        file_path_str = str(file_path).lower()
        file_path_obj = Path(file_path)

        for sensor_type, config in cls.CONFIGS.items():
            folder_name = config["base_dir"].lower()
            if f"/{folder_name}/" in file_path_str:
                return sensor_type

        suffix = file_path_obj.suffix.upper()
        for sensor_type, config in cls.CONFIGS.items():
            if suffix == config["extension"]:
                return sensor_type

        # Could not detect
        raise SensorDetectionError(
            f"Cannot detect sensor type for: {file_path}\n"
            f"Supported sensors:\n"
            f"  • Sony: /Sony/ folder or .ARW extension\n"
            f"  • Fuji: /Fuji/ folder or .RAF extension"
        )

    @classmethod
    def get_config(cls, sensor_type: SensorType) -> Dict[str, Any]:
        """Get configuration for sensor type.

        Args:
            sensor_type: SensorType enum value

        Returns:
            Dictionary containing sensor configuration

        Raises:
            KeyError: If sensor_type is not supported
        """
        if sensor_type not in cls.CONFIGS:
            raise KeyError(f"Unknown sensor type: {sensor_type}")

        return cls.CONFIGS[sensor_type].copy()

    @classmethod
    def get_config_by_path(cls, file_path: str) -> Dict[str, Any]:
        """Convenience method: detect sensor and get config in one call.

        Args:
            file_path: Path to raw image file

        Returns:
            Configuration dictionary for detected sensor
        """
        sensor_type = cls.detect(file_path)
        return cls.get_config(sensor_type)

    @classmethod
    def is_valid_extension(cls, file_path: str) -> bool:
        """Check if file has valid sensor extension.

        Args:
            file_path: Path to check

        Returns:
            True if file has .ARW or .RAF extension
        """
        suffix = Path(file_path).suffix.upper()
        return any(suffix == config["extension"] for config in cls.CONFIGS.values())

    @classmethod
    def list_supported_sensors(cls) -> Dict[str, str]:
        """List all supported sensors with descriptions.

        Returns:
            Dictionary mapping sensor names to descriptions
        """
        return {
            sensor.value: config["description"]
            for sensor, config in cls.CONFIGS.items()
        }

    @classmethod
    def get_tile_grid(cls, file_path: str) -> tuple:
        """Get tile grid dimensions for a file's sensor.

        Args:
            file_path: Path to raw image file

        Returns:
            Tuple of (rows, cols) for tile extraction
        """
        config = cls.get_config_by_path(file_path)
        return config["tile_grid"]

    @classmethod
    def get_channels(cls, file_path: str) -> int:
        """Get number of channels for a file's sensor.

        Args:
            file_path: Path to raw image file

        Returns:
            Number of channels after packing (4 for Sony, 9 for Fuji)
        """
        config = cls.get_config_by_path(file_path)
        return config["channels"]
