"""
Advanced error handling utilities for Poisson-Gaussian Diffusion.

This module provides sophisticated error handling, recovery mechanisms,
and diagnostic tools for robust operation in production environments.
"""

import logging
import traceback
import warnings
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from .exceptions import (
    CalibrationError,
    ConfigurationError,
    DataError,
    DomainError,
    GuidanceError,
    MetadataError,
    ModelError,
    NumericalStabilityError,
    PoissonDiffusionError,
    TransformError,
)


class ErrorHandler:
    """
    Centralized error handling with recovery mechanisms.

    Note: This class is designed to be pickle-safe for distributed training.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        enable_recovery: bool = True,
        strict_mode: bool = False,
    ):
        """
        Initialize error handler.

        Args:
            logger: Logger instance for error reporting
            enable_recovery: Whether to attempt automatic recovery
            strict_mode: Whether to raise all errors (no recovery)
        """
        # Store logger name instead of logger instance to avoid pickle issues
        self._logger_name = logger.name if logger else __name__
        self.enable_recovery = enable_recovery and not strict_mode
        self.strict_mode = strict_mode

        # Error statistics
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
        """
        Handle an error with optional recovery.

        Args:
            error: The exception that occurred
            context: Context description for logging
            recovery_func: Function to attempt recovery
            **recovery_kwargs: Arguments for recovery function

        Returns:
            Recovery result if successful, otherwise re-raises
        """
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log the error
        self.logger.error(f"Error in {context}: {error_type}: {str(error)}")
        self.logger.debug(f"Traceback: {traceback.format_exc()}")

        # Attempt recovery if enabled and function provided
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

        # Re-raise the original error
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
    """
    Advanced numerical stability management with adaptive thresholds.
    """

    def __init__(
        self,
        eps_variance: float = 0.1,
        grad_clip: float = 10.0,
        range_min: float = 0.0,
        range_max: Optional[float] = None,
        adaptive: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize numerical stability manager.

        Args:
            eps_variance: Minimum variance value
            grad_clip: Gradient clipping threshold
            range_min: Minimum allowed value
            range_max: Maximum allowed value
            adaptive: Whether to adapt thresholds based on data
            logger: Logger for warnings and diagnostics
        """
        self.eps_variance = eps_variance
        self.grad_clip = grad_clip
        self.range_min = range_min
        self.range_max = range_max
        self.adaptive = adaptive
        self.logger = logger or logging.getLogger(__name__)

        # Adaptive statistics
        self.variance_history: List[float] = []
        self.gradient_history: List[float] = []
        self.adaptation_count = 0

    def check_and_fix_tensor(
        self, tensor: torch.Tensor, name: str = "tensor", fix_issues: bool = True
    ) -> torch.Tensor:
        """
        Check tensor for issues and optionally fix them.

        Args:
            tensor: Input tensor
            name: Name for logging
            fix_issues: Whether to fix issues automatically

        Returns:
            Fixed tensor (if fix_issues=True) or original tensor

        Raises:
            NumericalStabilityError: If issues found and fix_issues=False
        """
        issues = []
        fixed_tensor = tensor.clone()

        # Check for NaN values
        nan_mask = torch.isnan(tensor)
        if nan_mask.any():
            issues.append(f"NaN values: {nan_mask.sum().item()}")
            if fix_issues:
                fixed_tensor = torch.where(
                    nan_mask, torch.zeros_like(tensor), fixed_tensor
                )
            else:
                raise NumericalStabilityError(f"{name} contains NaN values")

        # Check for Inf values
        inf_mask = torch.isinf(tensor)
        if inf_mask.any():
            issues.append(f"Inf values: {inf_mask.sum().item()}")
            if fix_issues:
                # Replace with large but finite values
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

        # Check range violations
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

        # Log issues if any were found
        if issues and fix_issues:
            self.logger.warning(f"Fixed {name}: {', '.join(issues)}")

        return fixed_tensor

    def stabilize_variance(
        self, variance: torch.Tensor, name: str = "variance"
    ) -> torch.Tensor:
        """
        Stabilize variance with adaptive thresholds.

        Args:
            variance: Variance tensor
            name: Name for logging

        Returns:
            Stabilized variance
        """
        # Update statistics for adaptation
        if self.adaptive:
            current_min = variance.min().item()
            self.variance_history.append(current_min)

            # Adapt threshold if we have enough history
            if len(self.variance_history) > 100:
                # Use 5th percentile as adaptive threshold
                adaptive_eps = np.percentile(self.variance_history[-100:], 5)
                if adaptive_eps > 0 and adaptive_eps < self.eps_variance:
                    old_eps = self.eps_variance
                    self.eps_variance = max(adaptive_eps, old_eps * 0.1)
                    self.adaptation_count += 1
                    self.logger.info(
                        f"Adapted variance threshold: {old_eps:.6f} -> {self.eps_variance:.6f}"
                    )

        # Apply stabilization
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
        """
        Clip gradients with adaptive thresholds.

        Args:
            gradients: Gradient tensor
            name: Name for logging

        Returns:
            Clipped gradients
        """
        # Update statistics for adaptation
        if self.adaptive:
            current_norm = gradients.norm().item()
            self.gradient_history.append(current_norm)

            # Adapt threshold if we have enough history
            if len(self.gradient_history) > 100:
                # Use 95th percentile as adaptive threshold
                adaptive_clip = np.percentile(self.gradient_history[-100:], 95)
                if adaptive_clip > self.grad_clip:
                    old_clip = self.grad_clip
                    self.grad_clip = min(adaptive_clip, old_clip * 2.0)
                    self.adaptation_count += 1
                    self.logger.info(
                        f"Adapted gradient clip: {old_clip:.3f} -> {self.grad_clip:.3f}"
                    )

        # Apply clipping
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
    """
    Decorator for safe operations with automatic error handling.

    Args:
        operation_name: Name of the operation for logging
        recovery_func: Function to call for recovery
        error_types: Types of errors to catch
        logger: Logger instance
    """

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
    """
    Context manager for error handling with detailed logging.

    Args:
        context_name: Name of the context for logging
        logger: Logger instance
        suppress_errors: Whether to suppress errors (return None on error)
    """
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
    """
    Collect and analyze diagnostic information for debugging.
    """

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

            # Add statistics for numeric data
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
