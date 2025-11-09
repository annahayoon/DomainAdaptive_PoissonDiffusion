"""
Sensor-specific utility functions for the Poisson-Gaussian Diffusion project.

This module provides sensor calibration, normalization, validation, calibration
conversion utilities, and demosaicing functions for raw image processing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rawpy
import torch
from PIL import Image

from core.normalization import convert_range

logger = logging.getLogger(__name__)


class SensorProcessingError(Exception):
    """Base exception for sensor processing errors"""

    pass


class InvalidRawDataError(SensorProcessingError):
    """Raised when raw image data is invalid or corrupted"""

    pass


def compute_sensor_range(black_level: float, white_level: float) -> float:
    """Compute sensor range (white_level - black_level)."""
    return white_level - black_level


def _get_sensor_config_module():
    """Get sensor_config module with proper error handling.

    This helper consolidates the repeated try/except import pattern
    used throughout sensor_utils.py.

    Returns:
        The sensor_config module

    Raises:
        ImportError: If sensor_config is not available
    """
    try:
        from config.sensor_config import get_sensor_config

        return get_sensor_config
    except ImportError:
        raise ImportError("config.sensor_config not available")


def load_sensor_calibration_from_metadata(sensor_type: str) -> Tuple[float, float]:
    """Load sensor calibration (black level, white level) from sensor config."""
    get_sensor_config = _get_sensor_config_module()

    try:
        sensor_cfg = get_sensor_config(sensor_type)
        return float(sensor_cfg["black_level"]), float(sensor_cfg["white_level"])
    except ValueError as e:
        raise ValueError(f"Unsupported sensor type '{sensor_type}': {e}")


def create_sensor_range_dict(
    black_level: float, white_level: float
) -> Dict[str, float]:
    """Create sensor range dictionary."""
    return {"min": black_level, "max": white_level}


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
    try:
        from config.config import CAMERA_CONFIGS

        get_sensor_config = _get_sensor_config_module()
    except ImportError:
        return {"min": 0.0, "max": 16383.0}

    file_ext = Path(file_path).suffix.upper()

    sensor_type = None
    for st, config in CAMERA_CONFIGS.items():
        if file_ext == config["extension"].upper():
            sensor_type = st
            break

    if sensor_type and sensor_type in sensor_ranges:
        return sensor_ranges[sensor_type]
    else:
        return {"min": 0.0, "max": 16383.0}


def resolve_calibration_directories(
    calibration_dir: Optional[str],
    metadata_json_path: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve calibration and data root directories."""
    from pathlib import Path

    calibration_dir_path = Path(calibration_dir) if calibration_dir else None
    data_root_path = (
        Path(metadata_json_path).parent.parent if metadata_json_path else None
    )
    return calibration_dir_path, data_root_path


def get_black_level_white_level_from_metadata(
    tile_id: str, tile_lookup: Dict[str, Dict[str, Any]]
) -> Tuple[float, float]:
    """Get black level and white level from tile metadata."""
    if not isinstance(tile_id, str):
        raise TypeError(f"tile_id must be str, got {type(tile_id)}: {tile_id}")
    if not isinstance(tile_lookup, dict):
        raise TypeError(
            f"tile_lookup must be dict, got {type(tile_lookup)}: {tile_lookup}"
        )

    # Import here to avoid circular dependency
    from core.utils.data_utils import get_sensor_from_metadata

    sensor_type = get_sensor_from_metadata(tile_id, tile_lookup)
    return load_sensor_calibration_from_metadata(sensor_type)


def normalize_physical_to_normalized(
    y_physical: torch.Tensor,
    black_level: float,
    white_level: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize physical values to normalized space using core.normalization."""
    from core.normalization import normalize_physical_to_normalized as core_normalize

    return core_normalize(
        y_physical, black_level=black_level, white_level=white_level, epsilon=epsilon
    )


def denormalize_to_physical(
    tensor: torch.Tensor, black_level: float, white_level: float
) -> np.ndarray:
    """Denormalize tensor to physical space using core.normalization."""
    from core.normalization import denormalize_to_physical as core_denormalize

    range_dict = {"min": black_level, "max": white_level}
    return core_denormalize(tensor, range_dict=range_dict)


def reverse_normalize_to_raw(
    normalized_value: np.ndarray, black_level: float, white_level: float
) -> np.ndarray:
    """Reverse normalize from [0,1] back to raw pixel values.

    Args:
        normalized_value: Normalized values in [0, 1] range
        black_level: Sensor black level
        white_level: Sensor white level

    Returns:
        Raw pixel values in physical sensor range
    """
    if black_level is None or white_level is None:
        return None

    normalized_value = np.clip(normalized_value, 0.0, 1.0)
    raw_value = normalized_value * (white_level - black_level) + black_level
    return raw_value


def convert_calibration_to_sigma_r(
    b_norm: float,
    sensor_range: float,
) -> float:
    """Convert calibration parameter b (in [-1, 1] domain) to sigma_r (read noise in physical units).

    Variance transformation: variance_phys = variance_norm * (sensor_range/2)^2
    Read noise: sigma_r^2 = b_norm * (sensor_range/2)^2
    """
    if b_norm < 0:
        raise ValueError(
            f"Invalid calibration: b_norm={b_norm} must be non-negative "
            f"(read noise variance cannot be negative)"
        )

    variance_phys = b_norm * ((sensor_range / 2.0) ** 2)

    if variance_phys < 0:
        raise ValueError(
            f"Invalid variance conversion: variance_phys={variance_phys} < 0 "
            f"(b_norm={b_norm}, sensor_range={sensor_range})"
        )

    sigma_r = np.sqrt(variance_phys)
    return float(sigma_r)


def convert_calibration_to_poisson_coeff(
    a_norm: float,
    sensor_range: float,
) -> float:
    """Convert calibration parameter a (in [-1, 1] domain) to Poisson coefficient in physical units.

    Conversion: a_phys = a_norm * (sensor_range/2)
    """
    if a_norm < 0:
        raise ValueError(
            f"Invalid calibration: a_norm={a_norm} must be non-negative "
            f"(Poisson coefficient cannot be negative)"
        )

    poisson_coeff = a_norm * (sensor_range / 2.0)

    return float(poisson_coeff)


def get_sensor_calibration_params(
    sensor_name: Optional[str],
    extracted_sensor: str,
    short_phys: np.ndarray,
    sensor_ranges: Dict[str, Dict[str, float]],
    conservative_factor: float,
    calibration_dir: Optional[Path] = None,
) -> Tuple[float, Dict[str, Any], Dict[str, Dict[str, float]], float]:
    """Get sensor calibration parameters for posterior sampling."""
    try:
        from core.utils.data_utils import _get_sample_config
        from sample.sensor_noise_calibrations import SensorCalibration

        _, SENSOR_NAME_MAPPING, _ = _get_sample_config()
    except ImportError:
        raise ImportError("sample.sensor_noise_calibrations not available")

    from pathlib import Path

    calib_sensor_name = sensor_name or SENSOR_NAME_MAPPING.get(
        extracted_sensor, extracted_sensor
    )
    mean_signal_physical = float(short_phys.mean())
    sensor_range = sensor_ranges.get(extracted_sensor, {"min": 0.0, "max": 16383.0})
    s_sensor = compute_sensor_range(sensor_range["min"], sensor_range["max"])

    if calibration_dir is not None:
        calibration_dir = Path(calibration_dir)

    calib_params = SensorCalibration.get_posterior_sampling_params(
        sensor_name=calib_sensor_name,
        mean_signal_physical=mean_signal_physical,
        s=s_sensor,
        conservative_factor=conservative_factor,
        calibration_dir=calibration_dir,
    )
    sigma_max = calib_params["sigma_max"]

    return (
        sigma_max,
        {
            "method": "sensor_calibration",
            "sensor_name": calib_sensor_name,
            "extracted_sensor": extracted_sensor,
            "sigma_max_calibrated": sigma_max,
            "mean_signal_physical": mean_signal_physical,
            "sensor_specs": calib_params["sensor_info"],
        },
        sensor_range,
        s_sensor,
    )


def compute_residual_components(
    x0_hat: torch.Tensor,
    y_e_physical: torch.Tensor,
    black_level: float,
    white_level: float,
    s: float,
    alpha: float,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute residual components for validation."""
    y_e_norm = normalize_physical_to_normalized(
        y_e_physical, black_level, white_level, epsilon
    )
    y_e_scaled = y_e_norm * s
    expected_at_short_exp = alpha * s * x0_hat
    return y_e_scaled, expected_at_short_exp, y_e_scaled - expected_at_short_exp


def validate_exposure_ratio(
    short_tensor: torch.Tensor,
    long_tensor: torch.Tensor,
    assumed_alpha: float,
    sensor_type: str,
) -> Tuple[float, float]:
    """Validate exposure ratio between short and long images."""
    short_01 = convert_range(short_tensor, "[-1,1]", "[0,1]").detach().cpu().numpy()
    long_01 = convert_range(long_tensor, "[-1,1]", "[0,1]").detach().cpu().numpy()
    short_mean = np.mean(short_01)
    long_mean = np.mean(long_01)

    if long_mean < 1e-6:
        return assumed_alpha, 0.0

    measured_alpha = short_mean / long_mean
    error_percent = abs(measured_alpha - assumed_alpha) / assumed_alpha * 100

    if error_percent > 20.0:
        logger.error(
            f"{sensor_type}: Exposure ratio mismatch {error_percent:.1f}% "
            f"(expected {assumed_alpha:.4f}, got {measured_alpha:.4f})"
        )
    elif error_percent > 10.0:
        logger.warning(
            f"{sensor_type}: Exposure ratio error {error_percent:.1f}% "
            f"(expected {assumed_alpha:.4f}, got {measured_alpha:.4f})"
        )

    return measured_alpha, error_percent


def validate_sensor_range_consistency(
    s: float, black_level: float, white_level: float
) -> None:
    """Validate that s equals sensor_range for unit consistency."""
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
    """Validate tensor inputs for guidance computation."""
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
    """Validate physical consistency using chi-squared test."""
    y_e_norm = normalize_physical_to_normalized(
        y_e_physical, sensor_min, sensor_max, epsilon
    )
    y_e_scaled = y_e_norm * s
    expected_y_at_short_exp = exposure_ratio * s * x_enhanced
    variance_at_short_exp = exposure_ratio * s * x_enhanced + sigma_r**2 + epsilon
    residual = y_e_scaled - expected_y_at_short_exp
    chi_squared_map = (residual**2) / variance_at_short_exp
    chi_squared_mean = chi_squared_map.mean().item()

    return {
        "chi_squared": chi_squared_mean,
        "chi_squared_std": chi_squared_map.std().item(),
        "physically_consistent": 0.8 < chi_squared_mean < 1.2,
        "mean_residual": residual.mean().item(),
        "max_residual": residual.abs().max().item(),
    }


class PhotonCountValidator:
    """Validator for photon count statistics and Gaussian approximation quality."""

    @staticmethod
    def estimate_photon_counts(
        y_e_physical: torch.Tensor,
        alpha: float,
        s: float,
        black_level: float,
        white_level: float,
    ) -> Dict[str, float]:
        """Estimate photon counts from physical image."""
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
        """Validate Gaussian approximation quality based on photon counts."""
        warnings_list = []
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
            warnings_list.append(
                f"Some pixels have low photon counts (mean={mean_photons:.1f}). "
                f"Gaussian approximation may be slightly inaccurate."
            )
        elif mean_photons >= critical_threshold:
            quality = "marginal"
            is_valid = not strict
            warnings_list.append(
                f"WARNING: Low photon counts detected (mean={mean_photons:.1f}). "
                f"{frac_below_10*100:.1f}% of pixels have λ < 10. "
                f"Gaussian approximation may introduce noticeable errors."
            )
        else:
            quality = "poor"
            is_valid = False
            warnings_list.append(
                f"CRITICAL: Very low photon counts (mean={mean_photons:.1f}). "
                f"{frac_below_3*100:.1f}% of pixels have λ < 3. "
                f"Gaussian approximation is inappropriate - consider discrete Poisson model."
            )

        action_map = {
            "excellent": "Gaussian approximation is highly accurate. Proceed with confidence.",
            "good": "Gaussian approximation is adequate. Results should be reliable.",
            "marginal": "Consider: (1) Increase exposure time, (2) Use discrete Poisson model, or (3) Accept reduced accuracy.",
            "poor": "CRITICAL: Switch to discrete Poisson diffusion model for accurate results.",
        }

        return {
            "is_valid": is_valid,
            "warnings": warnings_list,
            "approximation_quality": quality,
            "recommended_action": action_map[quality],
            "photon_statistics": photon_stats,
        }

    @staticmethod
    def compute_approximation_error(
        lambda_mean: float,
        sigma_r: float,
    ) -> Dict[str, float]:
        """Compute approximation error metrics."""
        variance = lambda_mean + sigma_r**2
        skewness = 1.0 / np.sqrt(variance) if variance > 0 else 0.0
        return {
            "kl_divergence_approx": skewness**2 / 12,
            "relative_variance_error": 0.0,
            "skewness": skewness,
            "lambda_mean": lambda_mean,
            "sigma_r": sigma_r,
        }


def optimize_sigma(
    sampler: Any,
    short_image: torch.Tensor,
    long_image: torch.Tensor,
    class_labels: Optional[torch.Tensor],
    sigma_range: Tuple[float, float],
    num_trials: int = 10,
    num_steps: int = 18,
    metric: str = "ssim",
    pg_guidance: Optional[Any] = None,
    y_e: Optional[torch.Tensor] = None,
    exposure_ratio: float = 1.0,
    no_heun: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Optimize sigma_max by searching over a range of values."""
    try:
        from core.metrics import compute_metrics_by_method
    except ImportError:
        raise ImportError("core.metrics not available for optimize_sigma")

    sigma_values = np.logspace(
        np.log10(sigma_range[0]), np.log10(sigma_range[1]), num=num_trials
    )

    best_sigma = sigma_values[0]
    best_metric_value = float("-inf") if metric in ["ssim", "psnr"] else float("inf")
    all_results = []

    for sigma in sigma_values:
        # Update sampler config
        sampler.config.sigma_max = sigma
        sampler.config.num_steps = num_steps
        sampler.config.no_heun = no_heun
        if pg_guidance is not None:
            sampler.guidance = pg_guidance

        result = sampler.sample(
            y_observed=y_e if y_e is not None else short_image,
            condition=class_labels,
            exposure_ratio=exposure_ratio,
        )
        restored = result["x_final"]

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

    return best_sigma, {
        "best_sigma": float(best_sigma),
        "best_metric": metric,
        "best_metric_value": float(best_metric_value),
        "all_trials": all_results,
    }


# ============================================================================
# DEMOSAICING UTILITIES
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


def _normalize_raw_image(
    raw_image: np.ndarray, black_level: float, white_level: float
) -> np.ndarray:
    """Normalize raw sensor image by subtracting black level and scaling to [0, 1].

    This is a common normalization pattern used across different sensor types.

    Args:
        raw_image: Raw image data as float32 array
        black_level: Sensor black level
        white_level: Sensor white level

    Returns:
        Normalized image in [0, 1] range
    """
    sensor_range = compute_sensor_range(black_level, white_level)
    return np.maximum(raw_image - black_level, 0) / sensor_range


def pack_raw_sony(raw_image_data: np.ndarray, max_value: float) -> np.ndarray:
    """Pack Sony Bayer pattern raw image to 4 channels (RGGB).

    Implementation based on Learning-to-See-in-the-Dark (Chen et al., CVPR 2018).

    Args:
        raw_image_data: Raw Bayer image as (H, W) uint16 array
        max_value: Maximum pixel value for normalization (e.g., 16383 for 14-bit ADC)

    Returns:
        Packed image as (H/2, W/2, 4) float32 array
        Channels in order: [R, G1, G2, B]
        All values normalized to [0, 1]

    Raises:
        ValueError: If input validation fails
    """
    get_sensor_config = _get_sensor_config_module()

    H, W = _validate_raw_input(raw_image_data, max_value, "Sony", 2)

    im = raw_image_data.astype(np.float32)
    sensor_cfg = get_sensor_config("sony")
    black_level = sensor_cfg["black_level"]
    white_level = sensor_cfg["white_level"]
    im = _normalize_raw_image(im, black_level, white_level)

    im = np.expand_dims(im, axis=2)
    out = np.concatenate(
        (
            im[0:H:2, 0:W:2, :],
            im[0:H:2, 1:W:2, :],
            im[1:H:2, 1:W:2, :],
            im[1:H:2, 0:W:2, :],
        ),
        axis=2,
    )

    return out


def pack_raw_fuji(raw_image_data: np.ndarray, max_value: float) -> np.ndarray:
    """Pack Fuji X-Trans pattern raw image to 9 channels.

    Implementation based on Learning-to-See-in-the-Dark (Chen et al., CVPR 2018).

    Args:
        raw_image_data: Raw X-Trans image as (H, W) uint16 array
        max_value: Maximum pixel value for normalization (e.g., 16383 for 14-bit ADC)

    Returns:
        Packed image as (H/3, W/3, 9) float32 array
        All values normalized to [0, 1]

    Raises:
        ValueError: If input validation fails
    """
    get_sensor_config = _get_sensor_config_module()

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

    im = raw_image_data.astype(np.float32)
    sensor_cfg = get_sensor_config("fuji")
    black_level = sensor_cfg["black_level"]
    white_level = sensor_cfg["white_level"]
    im = _normalize_raw_image(im, black_level, white_level)

    H = (H // 6) * 6
    W = (W // 6) * 6
    im = im[:H, :W]

    out = np.zeros((H // 3, W // 3, 9))

    out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

    out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

    out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

    out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

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
    """Pack raw image data and convert to RGB format.

    Args:
        raw_image_data: Raw Bayer/X-Trans image data (H, W)
        sensor_type: 'sony' or 'fuji'
        max_value: Maximum pixel value for normalization

    Returns:
        RGB image (3, H', W') normalized to [0, 1]
    """
    try:
        from scipy.ndimage import zoom
    except ImportError:
        raise ImportError("scipy.ndimage.zoom is required for pack_raw_to_rgb")

    if sensor_type == "sony":
        packed = pack_raw_sony(raw_image_data, max_value)
        img_hwc = packed

        R_sampled = img_hwc[:, :, 0]
        G1_sampled = img_hwc[:, :, 1]
        G2_sampled = img_hwc[:, :, 2]
        B_sampled = img_hwc[:, :, 3]

        G_sampled = (G1_sampled + G2_sampled) / 2.0

        R_upsampled = zoom(R_sampled, (2, 2), order=1, mode="nearest")
        G_upsampled = zoom(G_sampled, (2, 2), order=1, mode="nearest")
        B_upsampled = zoom(B_sampled, (2, 2), order=1, mode="nearest")

        rgb = np.stack([R_upsampled, G_upsampled, B_upsampled], axis=0)

    elif sensor_type == "fuji":
        packed = pack_raw_fuji(raw_image_data, max_value)
        img_hwc = packed

        R_components = img_hwc[:, :, 0] + img_hwc[:, :, 3]
        G_components = img_hwc[:, :, 1]
        B_components = img_hwc[:, :, 2] + img_hwc[:, :, 4]

        R_avg = R_components / 2.0
        G_avg = G_components
        B_avg = B_components / 2.0

        R_upsampled = zoom(R_avg, (3, 3), order=1, mode="nearest")
        G_upsampled = zoom(G_avg, (3, 3), order=1, mode="nearest")
        B_upsampled = zoom(B_avg, (3, 3), order=1, mode="nearest")

        rgb = np.stack([R_upsampled, G_upsampled, B_upsampled], axis=0)
        rgb = zoom(rgb, (1.0, 0.5, 0.5), order=1, mode="reflect")

    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    return rgb


def demosaic_raw_to_rgb(
    raw_path: str,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Demosaic raw image to RGB format.

    Args:
        raw_path: Path to .ARW (Sony) or .RAF (Fuji) file

    Returns:
        Tuple of (image, metadata)
        - image: (C, H, W) numpy array in float32, normalized to [0, 1]
        - metadata: Dictionary with camera info, white balance, etc.
    """
    try:
        import rawpy

        from core.sensor_detector import SensorDetector

        get_sensor_config = _get_sensor_config_module()
        from core.utils.metadata_utils import extract_raw_metadata
    except ImportError:
        logger.error(
            "rawpy or config.sensor_config not available for demosaic_raw_to_rgb"
        )
        return None, None

    try:
        sensor_type = SensorDetector.detect(raw_path)
        sensor_cfg = get_sensor_config(sensor_type.value)
        max_value = sensor_cfg["max"]

        with rawpy.imread(raw_path) as raw:
            raw_image_data = raw.raw_image_visible
            rgb_image = pack_raw_to_rgb(raw_image_data, sensor_type.value, max_value)
            metadata = extract_raw_metadata(raw, raw_path, rgb_image, sensor_type.value)
            return rgb_image, metadata

    except Exception as e:
        logger.error(f"Error demosaicing raw file {raw_path}: {e}")
        return None, None


class SensorDetectionError(SensorProcessingError):
    """Raised when sensor type cannot be detected from file"""

    pass


class NoiseRegimeClassifier:
    """Classify noise regime based on Poisson-to-Gaussian variance ratio."""

    def __init__(self, poisson_coeff: float, sigma_r: float):
        self.poisson_coeff = poisson_coeff
        self.sigma_r = sigma_r
        self.sigma_r_squared = sigma_r**2

    def classify(self, signal_mean: float) -> Tuple[str, float]:
        """Classify noise regime based on signal level.

        Args:
            signal_mean: Mean signal level in sensor range units [0, s]

        Returns:
            Tuple of (regime_name, variance_ratio)
        """
        poisson_var = self.poisson_coeff * signal_mean
        ratio = poisson_var / self.sigma_r_squared if self.sigma_r_squared > 0 else 0.0

        if ratio < 0.01:
            return "read_noise_dominated", ratio
        elif ratio < 0.2:
            return "transitional", ratio
        else:
            return "shot_noise_dominated", ratio

    def select_guidance(self, regime: str) -> str:
        """Select appropriate guidance method for given regime."""
        from core.guidance import select_guidance as select_guidance_func

        return select_guidance_func(regime)


# Noise regime constants
NOISE_REGIME_NAMES = ["read_noise_dominated", "transitional", "shot_noise_dominated"]
NOISE_REGIME_LABELS = ["Read-Noise\nDominated", "Transitional", "Shot-Noise\nDominated"]
NOISE_REGIME_COLORS = {
    "read_noise_dominated": "red",
    "transitional": "orange",
    "shot_noise_dominated": "green",
}
NOISE_REGIME_BAR_COLORS = ["red", "orange", "green"]

SHOT_NOISE_THRESHOLD = 0.2
READ_NOISE_THRESHOLD = 0.01


def extract_tile_from_raw_postprocess(
    raw_file_path: Path,
    image_x: int,
    image_y: int,
    sensor_type: str,
) -> np.ndarray:
    """Extract a tile from RAW file after processing with rawpy.postprocess().

    Args:
        raw_file_path: Path to RAW file (.ARW or .RAF)
        image_x: X coordinate of tile in processed image
        image_y: Y coordinate of tile in processed image
        sensor_type: Sensor type ("sony" or "fuji")

    Returns:
        RGB tile array as uint8 numpy array
    """
    try:
        from config.config import TILE_CONFIGS
    except ImportError:
        raise ImportError("core.config not available")

    tile_size = TILE_CONFIGS[sensor_type]["tile_size"]

    with rawpy.imread(str(raw_file_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,
            output_bps=8,
        )

    return rgb[image_y : image_y + tile_size, image_x : image_x + tile_size, :]


def convert_raw_tile_to_png(
    raw_file_path: Path,
    output_path: Path,
    image_x: int,
    image_y: int,
    sensor_type: str,
    format_type: str,
) -> bool:
    """Convert RAW file tile to PNG image.

    Args:
        raw_file_path: Path to RAW file
        output_path: Path to save PNG file
        image_x: X coordinate of tile in processed image
        image_y: Y coordinate of tile in processed image
        sensor_type: Sensor type ("sony" or "fuji")
        format_type: Format type ("uint8_png" or "float32_png")

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        from core.utils.file_utils import (
            save_tile_as_float32_png,
            save_tile_as_uint8_png,
        )

        tile = extract_tile_from_raw_postprocess(
            raw_file_path, image_x, image_y, sensor_type
        )

        if format_type == "uint8_png":
            save_tile_as_uint8_png(tile, output_path)
        elif format_type == "float32_png":
            save_tile_as_float32_png(tile, output_path)
        else:
            logger.error(f"Unknown format type: {format_type}")
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to convert {raw_file_path} to {format_type}: {e}")
        return False


def process_sensor_baseline_conversion(
    sensor: str,
    data_root: Path,
    splits_dir: Path,
    output_base: Path,
    format_configs: Dict[str, Dict[str, str]],
    formats: Optional[List[str]] = None,
) -> Optional[Dict[str, Dict[str, int]]]:
    """Process tiles for a single sensor for baseline conversion.

    Args:
        sensor: Sensor type ("sony" or "fuji")
        data_root: Root directory containing processed data
        splits_dir: Directory containing split files
        output_base: Base output directory
        format_configs: Dictionary mapping format types to their configuration
            (e.g., {"uint8_png": {"name": "...", "baselines": "..."}})
        formats: Optional list of formats to process (default: all formats in format_configs)

    Returns:
        Dictionary mapping format_data_type keys to statistics, or None if processing fails
    """
    logger.info("")
    logger.info(f"Processing {sensor.upper()} sensor...")
    logger.info("-" * 70)

    from core.utils.data_utils import load_test_tiles_from_metadata_with_scenes
    from core.utils.file_utils import find_split_file, load_test_scenes_from_split_file
    from core.utils.tiles_utils import (
        convert_tiles_to_png,
        convert_uint8_tiles_to_float32_png,
    )

    formats_to_process = formats if formats else list(format_configs.keys())
    needs_metadata = any(f != "float32_png" for f in formats_to_process)

    if needs_metadata:
        # Try to find split file using core utility
        split_file = find_split_file(sensor, split_type="test")
        if not split_file:
            # Fallback to explicit path
            split_file = splits_dir / f"{sensor.capitalize()}_test_list.txt"

        test_scenes = load_test_scenes_from_split_file(split_file)

        if not test_scenes:
            logger.warning(f"No test scenes found for {sensor}, skipping...")
            return None

        metadata_json = data_root / f"metadata_{sensor}_incremental.json"
        if not metadata_json.exists():
            logger.warning(f"Metadata file not found: {metadata_json}, skipping...")
            return None

        long_tiles, short_tiles = load_test_tiles_from_metadata_with_scenes(
            metadata_json, test_scenes
        )

        if not long_tiles and not short_tiles:
            logger.warning(f"No test tiles found for {sensor}, skipping...")
            return None
    else:
        long_tiles, short_tiles = [], []

    sensor_stats = {}

    for format_type in formats_to_process:
        format_name = format_configs.get(format_type, {}).get("name", format_type)
        logger.info(f"Converting to {format_name}...")

        if format_type == "float32_png":
            for data_type in ["long", "short"]:
                stats = convert_uint8_tiles_to_float32_png(
                    output_base, data_type, sensor=sensor
                )
                sensor_stats[f"{format_type}_{data_type}"] = stats
                logger.info(
                    f"  {data_type}: {stats['success']} success, "
                    f"{stats['failed']} failed, {stats['skipped']} skipped"
                )
        else:
            for data_type, tiles in [("long", long_tiles), ("short", short_tiles)]:
                stats = convert_tiles_to_png(tiles, output_base, format_type, data_type)
                sensor_stats[f"{format_type}_{data_type}"] = stats
                logger.info(
                    f"  {data_type}: {stats['success']} success, "
                    f"{stats['failed']} failed, {stats['skipped']} skipped"
                )

    return sensor_stats
