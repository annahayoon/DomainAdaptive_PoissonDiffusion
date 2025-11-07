"""
Error handling utilities for Poisson-Gaussian Diffusion.

Provides error handling, recovery mechanisms, diagnostic tools, and custom exceptions.
"""

import logging
import traceback
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

# ============================================================================
# Custom Exceptions
# ============================================================================


class PoissonDiffusionError(Exception):
    """Base exception for all project-specific errors."""

    pass


class CalibrationError(PoissonDiffusionError):
    """Raised when sensor calibration is invalid or missing."""

    pass


class TransformError(PoissonDiffusionError):
    """Raised when image transformation fails."""

    pass


class MetadataError(PoissonDiffusionError):
    """Raised when metadata is invalid or corrupted."""

    pass


class GuidanceError(PoissonDiffusionError):
    """Raised when likelihood guidance computation fails."""

    pass


class ModelError(PoissonDiffusionError):
    """Raised when model operations fail."""

    pass


class DataError(PoissonDiffusionError):
    """Raised when data loading or processing fails."""

    pass


class NumericalStabilityError(PoissonDiffusionError):
    """Raised when numerical instability is detected."""

    pass


class SensorError(PoissonDiffusionError):
    """Raised when sensor-specific operations fail."""

    pass


class ConfigurationError(PoissonDiffusionError):
    """Raised when configuration is invalid."""

    pass


class InvalidPhotonCountError(GuidanceError):
    """Raised when photon counts are invalid (negative, NaN, etc.)."""

    pass


class UnsupportedSensorError(SensorError):
    """Raised when an unsupported sensor is specified."""

    pass


class IncompatibleMetadataError(MetadataError):
    """Raised when metadata is incompatible with current operation."""

    pass


class ReconstructionError(TransformError):
    """Raised when image reconstruction fails."""

    pass


class CalibrationParameterError(CalibrationError):
    """Raised when calibration parameters are physically unreasonable."""

    pass


class ValidationError(PoissonDiffusionError):
    """Raised when validation checks fail."""

    pass


class SamplingError(PoissonDiffusionError):
    """Raised when sampling operations fail."""

    pass


class TrainingError(PoissonDiffusionError):
    """Raised when training operations fail."""

    pass


class PerformanceError(PoissonDiffusionError):
    """Raised when performance operations fail."""

    pass


class AnalysisError(PoissonDiffusionError):
    """Raised when analysis operations fail."""

    pass


# ============================================================================
# Error Handling Classes
# ============================================================================


class ErrorHandler:
    """Centralized error handling with recovery mechanisms."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        enable_recovery: bool = True,
        strict_mode: bool = False,
    ):
        self._logger_name = logger.name if logger else __name__
        self.enable_recovery = enable_recovery and not strict_mode
        self.strict_mode = strict_mode
        self.error_counts: Dict[str, int] = {}
        self.recovery_counts: Dict[str, int] = {}

    @property
    def logger(self):
        """Lazy logger property to avoid pickling issues."""
        return logging.getLogger(self._logger_name)

    def handle_error(
        self,
        error: Exception,
        context: str = "",
        recovery_func: Optional[Callable] = None,
        **recovery_kwargs,
    ) -> Any:
        """Handle an error with optional recovery."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        self.logger.error(f"Error in {context}: {error_type}: {str(error)}")
        self.logger.debug(f"Traceback: {traceback.format_exc()}")

        if self.enable_recovery and recovery_func is not None:
            try:
                self.logger.info(f"Attempting recovery for {error_type}")
                result = recovery_func(**recovery_kwargs)
                self.recovery_counts[error_type] = (
                    self.recovery_counts.get(error_type, 0) + 1
                )
                self.logger.info(f"Recovery successful for {error_type}")
                return result
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {error_type}: {recovery_error}")

        raise error

    def get_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            "error_counts": dict(self.error_counts),
            "recovery_counts": dict(self.recovery_counts),
            "recovery_rate": {
                error_type: self.recovery_counts.get(error_type, 0) / count
                for error_type, count in self.error_counts.items()
            },
        }


class NumericalStabilityManager:
    """Advanced numerical stability management with adaptive thresholds."""

    def __init__(
        self,
        eps_variance: float = 0.1,
        grad_clip: float = 10.0,
        range_min: float = 0.0,
        range_max: Optional[float] = None,
        adaptive: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.eps_variance = eps_variance
        self.grad_clip = grad_clip
        self.range_min = range_min
        self.range_max = range_max
        self.adaptive = adaptive
        self.logger = logger or logging.getLogger(__name__)
        self.variance_history: List[float] = []
        self.gradient_history: List[float] = []
        self.adaptation_count = 0

    def check_and_fix_tensor(
        self, tensor: torch.Tensor, name: str = "tensor", fix_issues: bool = True
    ) -> torch.Tensor:
        """Check tensor for issues and optionally fix them."""
        issues = []
        fixed_tensor = tensor.clone()

        nan_mask = torch.isnan(tensor)
        if nan_mask.any():
            issues.append(f"NaN values: {nan_mask.sum().item()}")
            if fix_issues:
                fixed_tensor = torch.where(
                    nan_mask, torch.zeros_like(tensor), fixed_tensor
                )
            else:
                raise NumericalStabilityError(f"{name} contains NaN values")

        inf_mask = torch.isinf(tensor)
        if inf_mask.any():
            issues.append(f"Inf values: {inf_mask.sum().item()}")
            if fix_issues:
                max_finite = torch.finfo(tensor.dtype).max / 10
                fixed_tensor = torch.where(
                    inf_mask & (tensor > 0),
                    torch.full_like(tensor, max_finite),
                    fixed_tensor,
                )
                fixed_tensor = torch.where(
                    inf_mask & (tensor < 0),
                    torch.full_like(tensor, -max_finite),
                    fixed_tensor,
                )
            else:
                raise NumericalStabilityError(f"{name} contains Inf values")

        if self.range_min is not None:
            below_min = fixed_tensor < self.range_min
            if below_min.any():
                issues.append(
                    f"Below minimum {self.range_min}: {below_min.sum().item()}"
                )
                if fix_issues:
                    fixed_tensor = torch.clamp(fixed_tensor, min=self.range_min)
                else:
                    raise NumericalStabilityError(
                        f"{name} contains values below minimum {self.range_min}"
                    )

        if self.range_max is not None:
            above_max = fixed_tensor > self.range_max
            if above_max.any():
                issues.append(
                    f"Above maximum {self.range_max}: {above_max.sum().item()}"
                )
                if fix_issues:
                    fixed_tensor = torch.clamp(fixed_tensor, max=self.range_max)
                else:
                    raise NumericalStabilityError(
                        f"{name} contains values above maximum {self.range_max}"
                    )

        if issues and fix_issues:
            self.logger.warning(f"Fixed {name}: {', '.join(issues)}")

        return fixed_tensor

    def stabilize_variance(
        self, variance: torch.Tensor, name: str = "variance"
    ) -> torch.Tensor:
        """Stabilize variance with adaptive thresholds."""
        if self.adaptive:
            current_min = variance.min().item()
            self.variance_history.append(current_min)

            if len(self.variance_history) > 100:
                adaptive_eps = np.percentile(self.variance_history[-100:], 5)
                if adaptive_eps > 0 and adaptive_eps < self.eps_variance:
                    old_eps = self.eps_variance
                    self.eps_variance = max(adaptive_eps, old_eps * 0.1)
                    self.adaptation_count += 1
                    self.logger.info(
                        f"Adapted variance threshold: {old_eps:.6f} -> {self.eps_variance:.6f}"
                    )

        original_min = variance.min().item()
        stabilized = torch.clamp(variance, min=self.eps_variance)

        if original_min < self.eps_variance:
            self.logger.debug(
                f"Stabilized {name}: min {original_min:.6f} -> {self.eps_variance:.6f}"
            )

        return stabilized

    def clip_gradients(
        self, gradients: torch.Tensor, name: str = "gradients"
    ) -> torch.Tensor:
        """Clip gradients with adaptive thresholds."""
        if self.adaptive:
            current_norm = gradients.norm().item()
            self.gradient_history.append(current_norm)

            if len(self.gradient_history) > 100:
                adaptive_clip = np.percentile(self.gradient_history[-100:], 95)
                if adaptive_clip > self.grad_clip:
                    old_clip = self.grad_clip
                    self.grad_clip = min(adaptive_clip, old_clip * 2.0)
                    self.adaptation_count += 1
                    self.logger.info(
                        f"Adapted gradient clip: {old_clip:.3f} -> {self.grad_clip:.3f}"
                    )

        original_norm = gradients.norm().item()
        clipped = torch.clamp(gradients, -self.grad_clip, self.grad_clip)

        if original_norm > self.grad_clip:
            self.logger.debug(
                f"Clipped {name}: norm {original_norm:.3f} -> {self.grad_clip:.3f}"
            )

        return clipped


def safe_operation(
    operation_name: str = "",
    recovery_func: Optional[Callable] = None,
    error_types: tuple = (Exception,),
    logger: Optional[logging.Logger] = None,
):
    """Decorator for safe operations with automatic error handling."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            op_logger = logger or logging.getLogger(func.__module__)

            try:
                return func(*args, **kwargs)
            except error_types as e:
                op_logger.error(f"Error in {op_name}: {type(e).__name__}: {e}")

                if recovery_func is not None:
                    try:
                        op_logger.info(f"Attempting recovery for {op_name}")
                        return recovery_func(*args, **kwargs)
                    except Exception as recovery_error:
                        op_logger.error(f"Recovery failed: {recovery_error}")

                raise e

        return wrapper

    return decorator


@contextmanager
def error_context(
    context_name: str,
    logger: Optional[logging.Logger] = None,
    suppress_errors: bool = False,
):
    """Context manager for error handling with detailed logging."""
    ctx_logger = logger or logging.getLogger(__name__)

    try:
        ctx_logger.debug(f"Entering context: {context_name}")
        yield
        ctx_logger.debug(f"Exiting context: {context_name}")
    except Exception as e:
        ctx_logger.error(f"Error in context {context_name}: {type(e).__name__}: {e}")
        ctx_logger.debug(f"Traceback: {traceback.format_exc()}")

        if not suppress_errors:
            raise


class DiagnosticCollector:
    """Collect and analyze diagnostic information for debugging."""

    def __init__(self):
        self.diagnostics: Dict[str, List[Any]] = {}
        self.timestamps: Dict[str, List[float]] = {}

    def record(self, category: str, data: Any, timestamp: Optional[float] = None):
        """Record diagnostic data."""
        import time

        if category not in self.diagnostics:
            self.diagnostics[category] = []
            self.timestamps[category] = []

        self.diagnostics[category].append(data)
        self.timestamps[category].append(timestamp or time.time())

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected diagnostics."""
        summary = {}

        for category, data_list in self.diagnostics.items():
            if not data_list:
                continue

            summary[category] = {
                "count": len(data_list),
                "latest": data_list[-1] if data_list else None,
            }

            if all(isinstance(x, (int, float)) for x in data_list):
                summary[category].update(
                    {
                        "mean": np.mean(data_list),
                        "std": np.std(data_list),
                        "min": np.min(data_list),
                        "max": np.max(data_list),
                    }
                )

        return summary

    def clear(self, category: Optional[str] = None):
        """Clear diagnostic data."""
        if category is None:
            self.diagnostics.clear()
            self.timestamps.clear()
        else:
            self.diagnostics.pop(category, None)
            self.timestamps.pop(category, None)
