"""
Sensor calibration system for Domain-Adaptive Poisson-Gaussian Diffusion.

This module implements the calibration system for converting between ADU (Analog-to-Digital Units)
and electrons, handling sensor-specific parameters, and providing domain-specific defaults.

Key functionality:
- ADU ↔ electron conversions: electrons = (ADU - black_level) × gain
- Sensor parameter validation and loading from JSON
- Domain-specific default fallbacks
- Comprehensive error handling and diagnostics

Requirements addressed: 1.6, 5.5 from requirements.md
Task: 2.1 from tasks.md
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .error_handlers import safe_operation
from .exceptions import CalibrationError, ValidationError
from .interfaces import CalibrationManager
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationParams:
    """
    Complete sensor calibration parameters.

    All parameters should be in physically meaningful units.
    """

    # Core conversion parameters
    gain: float  # electrons/ADU conversion factor
    black_level: float  # ADU offset (sensor bias)
    white_level: float  # ADU saturation point

    # Noise characteristics
    read_noise: float  # electrons RMS
    dark_current: float = 0.0  # electrons/sec

    # Physical sensor properties
    quantum_efficiency: float = 1.0  # [0,1] photon detection efficiency
    pixel_size: float = 1.0  # physical size (μm or arcsec)
    pixel_unit: str = "pixel"  # 'um', 'arcsec', or 'pixel'

    # Sensor metadata
    sensor_name: str = "unknown"
    bit_depth: int = 16
    temperature: Optional[float] = None  # Celsius

    # Domain-specific parameters
    domain: str = "unknown"  # 'photography', 'microscopy', 'astronomy'

    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate that all parameters are physically reasonable."""
        errors = []

        # Core parameters must be positive
        if self.gain <= 0:
            errors.append(f"Gain must be positive, got {self.gain}")
        if self.white_level <= self.black_level:
            errors.append(
                f"White level ({self.white_level}) must be > black level ({self.black_level})"
            )
        if self.read_noise < 0:
            errors.append(f"Read noise must be non-negative, got {self.read_noise}")
        if self.dark_current < 0:
            errors.append(f"Dark current must be non-negative, got {self.dark_current}")

        # Physical constraints
        if not (0 <= self.quantum_efficiency <= 1):
            errors.append(
                f"Quantum efficiency must be in [0,1], got {self.quantum_efficiency}"
            )
        if self.pixel_size <= 0:
            errors.append(f"Pixel size must be positive, got {self.pixel_size}")
        if self.bit_depth not in [8, 10, 12, 14, 16, 32]:
            errors.append(f"Bit depth must be standard value, got {self.bit_depth}")

        # Check consistency
        max_adu = 2**self.bit_depth - 1
        if self.white_level > max_adu:
            errors.append(
                f"White level ({self.white_level}) exceeds bit depth capacity ({max_adu})"
            )

        if errors:
            raise CalibrationError(
                f"Invalid calibration parameters: {'; '.join(errors)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationParams":
        """Create from dictionary (JSON deserialization)."""
        return cls(**data)

    def get_dynamic_range_db(self) -> float:
        """Compute dynamic range in dB."""
        signal_range = (self.white_level - self.black_level) * self.gain
        return (
            20 * np.log10(signal_range / self.read_noise)
            if self.read_noise > 0
            else float("inf")
        )

    def get_full_well_capacity(self) -> float:
        """Estimate full well capacity in electrons."""
        return (self.white_level - self.black_level) * self.gain


class SensorCalibration(CalibrationManager):
    """
    Comprehensive sensor calibration management.

    Handles ADU ↔ electron conversions, parameter validation,
    and provides domain-specific defaults.
    """

    # Domain-specific default calibrations
    DOMAIN_DEFAULTS = {
        "photography": {
            "gain": 1.0,  # electrons/ADU (typical for modern CMOS)
            "black_level": 512,  # ADU (common for 14-bit)
            "white_level": 16383,  # ADU (14-bit max)
            "read_noise": 3.0,  # electrons RMS
            "dark_current": 0.1,  # electrons/sec
            "quantum_efficiency": 0.8,
            "pixel_size": 4.29,  # μm (Sony A7S)
            "pixel_unit": "um",
            "sensor_name": "Generic_Photography_Sensor",
            "bit_depth": 14,
            "domain": "photography",
        },
        "microscopy": {
            "gain": 0.5,  # electrons/ADU (scientific CMOS)
            "black_level": 100,  # ADU
            "white_level": 65535,  # ADU (16-bit)
            "read_noise": 1.5,  # electrons RMS (excellent sensor)
            "dark_current": 0.01,  # electrons/sec (cooled)
            "quantum_efficiency": 0.95,
            "pixel_size": 0.65,  # μm (at 20x magnification)
            "pixel_unit": "um",
            "sensor_name": "Generic_Scientific_CMOS",
            "bit_depth": 16,
            "domain": "microscopy",
        },
        "astronomy": {
            "gain": 2.0,  # electrons/ADU (typical CCD)
            "black_level": 0,  # ADU (CCD bias subtracted)
            "white_level": 65535,  # ADU (16-bit)
            "read_noise": 5.0,  # electrons RMS
            "dark_current": 0.05,  # electrons/sec
            "quantum_efficiency": 0.9,
            "pixel_size": 0.04,  # arcsec (Hubble WFC3)
            "pixel_unit": "arcsec",
            "sensor_name": "Generic_Astronomy_CCD",
            "bit_depth": 16,
            "domain": "astronomy",
        },
    }

    def __init__(
        self,
        calibration_file: Optional[Union[str, Path]] = None,
        params: Optional[CalibrationParams] = None,
        domain: Optional[str] = None,
    ):
        """
        Initialize sensor calibration.

        Args:
            calibration_file: Path to JSON calibration file
            params: Direct calibration parameters
            domain: Domain for default parameters if no file/params provided
        """
        self.params: Optional[CalibrationParams] = None
        self.calibration_file = Path(calibration_file) if calibration_file else None

        # Load calibration in priority order
        if params is not None:
            self.params = params
            logger.info("Using provided calibration parameters")
        elif calibration_file and Path(calibration_file).exists():
            self.params = self._load_from_file(calibration_file)
            logger.info(f"Loaded calibration from {calibration_file}")
        elif domain and domain in self.DOMAIN_DEFAULTS:
            self.params = CalibrationParams(**self.DOMAIN_DEFAULTS[domain])
            logger.warning(f"Using default calibration for domain '{domain}'")
        else:
            raise CalibrationError(
                f"No valid calibration source provided. "
                f"Available domains: {list(self.DOMAIN_DEFAULTS.keys())}"
            )

        # Validate loaded parameters
        self.validate_calibration()

        logger.info(
            f"Calibration initialized: {self.params.sensor_name} "
            f"(gain={self.params.gain:.3f} e⁻/ADU, "
            f"read_noise={self.params.read_noise:.1f} e⁻)"
        )

    def _load_from_file(self, filepath: Union[str, Path]) -> CalibrationParams:
        """Load calibration parameters from JSON file."""
        filepath = Path(filepath)

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Handle nested structure if present
            if "calibration" in data:
                data = data["calibration"]

            return CalibrationParams.from_dict(data)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise CalibrationError(f"Failed to load calibration from {filepath}: {e}")
        except Exception as e:
            raise CalibrationError(f"Unexpected error loading {filepath}: {e}")

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save current calibration parameters to JSON file."""
        if self.params is None:
            raise CalibrationError("No calibration parameters to save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, "w") as f:
                json.dump(
                    {
                        "calibration": self.params.to_dict(),
                        "metadata": {
                            "created_by": "SensorCalibration",
                            "version": "1.0",
                        },
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Calibration saved to {filepath}")

        except Exception as e:
            raise CalibrationError(f"Failed to save calibration to {filepath}: {e}")

    def validate_parameters(self) -> bool:
        """
        Validate that calibration parameters are physically reasonable.

        This implements the abstract method from CalibrationManager.

        Returns:
            True if calibration is valid

        Raises:
            CalibrationError: If validation fails
        """
        return self.validate_calibration()

    def validate_calibration(self) -> bool:
        """
        Validate current calibration parameters.

        Returns:
            True if calibration is valid

        Raises:
            CalibrationError: If validation fails
        """
        if self.params is None:
            raise CalibrationError("No calibration parameters loaded")

        # Parameters are validated in CalibrationParams.__post_init__
        # Additional validation can be added here

        # Check for reasonable dynamic range
        dr_db = self.params.get_dynamic_range_db()
        if dr_db < 40:  # Less than 40 dB is quite poor
            logger.warning(f"Low dynamic range: {dr_db:.1f} dB")

        # Check for reasonable full well capacity
        fwc = self.params.get_full_well_capacity()
        if fwc < 1000:  # Less than 1000 electrons is very small
            logger.warning(f"Small full well capacity: {fwc:.0f} electrons")

        return True

    @safe_operation("ADU to electrons conversion")
    def adu_to_electrons(
        self, adu: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert ADU values to electrons.

        Formula: electrons = (ADU - black_level) × gain

        Args:
            adu: ADU values (any shape)

        Returns:
            Electron values (same type and shape as input)
        """
        if self.params is None:
            raise CalibrationError("No calibration parameters loaded")

        is_torch = isinstance(adu, torch.Tensor)

        # Convert to electrons
        electrons = (adu - self.params.black_level) * self.params.gain

        # Ensure non-negative (physical constraint)
        if is_torch:
            electrons = torch.clamp(electrons, min=0.0)
        else:
            electrons = np.maximum(electrons, 0.0)

        return electrons

    @safe_operation("Electrons to ADU conversion")
    def electrons_to_adu(
        self, electrons: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert electron values to ADU.

        Formula: ADU = electrons / gain + black_level

        Args:
            electrons: Electron values (any shape)

        Returns:
            ADU values (same type and shape as input)
        """
        if self.params is None:
            raise CalibrationError("No calibration parameters loaded")

        is_torch = isinstance(electrons, torch.Tensor)

        # Convert to ADU
        adu = electrons / self.params.gain + self.params.black_level

        # Clamp to valid ADU range
        if is_torch:
            adu = torch.clamp(adu, min=0.0, max=self.params.white_level)
        else:
            adu = np.clip(adu, 0.0, self.params.white_level)

        return adu

    @safe_operation("Raw data processing")
    def process_raw(
        self,
        raw_adu: Union[np.ndarray, torch.Tensor],
        return_mask: bool = True,
        apply_dark_correction: bool = False,
        exposure_time: Optional[float] = None,
    ) -> Union[
        Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]],
        Union[np.ndarray, torch.Tensor],
    ]:
        """
        Process raw ADU data to calibrated electrons.

        Args:
            raw_adu: Raw ADU values from sensor
            return_mask: Whether to return valid pixel mask
            apply_dark_correction: Whether to subtract dark current
            exposure_time: Exposure time in seconds (needed for dark correction)

        Returns:
            If return_mask=True: (electrons, mask)
            If return_mask=False: electrons
        """
        if self.params is None:
            raise CalibrationError("No calibration parameters loaded")

        is_torch = isinstance(raw_adu, torch.Tensor)

        # Create validity mask
        if is_torch:
            # Valid pixels: not saturated, above black level
            mask = (raw_adu < self.params.white_level) & (
                raw_adu >= self.params.black_level
            )
        else:
            mask = (raw_adu < self.params.white_level) & (
                raw_adu >= self.params.black_level
            )

        # Convert to electrons
        electrons = self.adu_to_electrons(raw_adu)

        # Apply dark current correction if requested
        if apply_dark_correction and exposure_time is not None:
            dark_electrons = self.params.dark_current * exposure_time
            electrons = electrons - dark_electrons

            # Ensure non-negative after dark subtraction
            if is_torch:
                electrons = torch.clamp(electrons, min=0.0)
            else:
                electrons = np.maximum(electrons, 0.0)

        if return_mask:
            return electrons, mask
        else:
            return electrons

    def estimate_noise_variance(
        self, signal_electrons: Union[np.ndarray, torch.Tensor, float]
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """
        Estimate total noise variance for given signal level.

        Formula: Var = signal + read_noise²
        (Poisson shot noise + Gaussian read noise)

        Args:
            signal_electrons: Signal level in electrons

        Returns:
            Noise variance in electrons²
        """
        if self.params is None:
            raise CalibrationError("No calibration parameters loaded")

        # Total variance = Poisson variance + read noise variance
        variance = signal_electrons + self.params.read_noise**2

        return variance

    def estimate_snr_db(
        self, signal_electrons: Union[np.ndarray, torch.Tensor, float]
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """
        Estimate signal-to-noise ratio in dB.

        Args:
            signal_electrons: Signal level in electrons

        Returns:
            SNR in dB
        """
        if self.params is None:
            raise CalibrationError("No calibration parameters loaded")

        variance = self.estimate_noise_variance(signal_electrons)

        is_torch = isinstance(signal_electrons, torch.Tensor)

        if is_torch:
            noise_std = torch.sqrt(variance)
            snr_linear = signal_electrons / noise_std
            snr_db = 20 * torch.log10(torch.clamp(snr_linear, min=1e-10))
        else:
            noise_std = np.sqrt(variance)
            snr_linear = signal_electrons / noise_std
            snr_db = 20 * np.log10(np.maximum(snr_linear, 1e-10))

        return snr_db

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get comprehensive calibration summary for diagnostics."""
        if self.params is None:
            return {"status": "No calibration loaded"}

        return {
            "sensor_name": self.params.sensor_name,
            "domain": self.params.domain,
            "gain_e_per_adu": self.params.gain,
            "black_level_adu": self.params.black_level,
            "white_level_adu": self.params.white_level,
            "read_noise_e": self.params.read_noise,
            "dark_current_e_per_sec": self.params.dark_current,
            "quantum_efficiency": self.params.quantum_efficiency,
            "pixel_size": f"{self.params.pixel_size} {self.params.pixel_unit}",
            "bit_depth": self.params.bit_depth,
            "dynamic_range_db": self.params.get_dynamic_range_db(),
            "full_well_capacity_e": self.params.get_full_well_capacity(),
            "calibration_file": str(self.calibration_file)
            if self.calibration_file
            else "Default/Direct",
        }

    @classmethod
    def create_default_calibration_files(
        cls, output_dir: Union[str, Path]
    ) -> List[Path]:
        """
        Create default calibration files for all domains.

        Args:
            output_dir: Directory to save calibration files

        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_files = []

        for domain, defaults in cls.DOMAIN_DEFAULTS.items():
            params = CalibrationParams(**defaults)
            filepath = output_dir / f"{domain}_default.json"

            # Create temporary calibration to save file
            temp_cal = cls(params=params)
            temp_cal.save_to_file(filepath)
            created_files.append(filepath)

        logger.info(
            f"Created {len(created_files)} default calibration files in {output_dir}"
        )
        return created_files


# Convenience functions for common operations
def load_calibration(
    calibration_file: Optional[Union[str, Path]] = None, domain: Optional[str] = None
) -> SensorCalibration:
    """
    Convenience function to load sensor calibration.

    Args:
        calibration_file: Path to calibration JSON file
        domain: Domain for default calibration

    Returns:
        Initialized SensorCalibration instance
    """
    return SensorCalibration(calibration_file=calibration_file, domain=domain)


def create_calibration_from_params(
    gain: float, black_level: float, white_level: float, read_noise: float, **kwargs
) -> SensorCalibration:
    """
    Convenience function to create calibration from parameters.

    Args:
        gain: electrons/ADU conversion factor
        black_level: ADU offset
        white_level: ADU saturation
        read_noise: Read noise in electrons RMS
        **kwargs: Additional CalibrationParams arguments

    Returns:
        Initialized SensorCalibration instance
    """
    params = CalibrationParams(
        gain=gain,
        black_level=black_level,
        white_level=white_level,
        read_noise=read_noise,
        **kwargs,
    )
    return SensorCalibration(params=params)
