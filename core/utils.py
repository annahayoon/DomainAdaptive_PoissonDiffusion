"""
Utility functions for the Poisson-Gaussian Diffusion project.

This module contains common utilities for numerical stability,
validation, logging, and other shared functionality.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

from .exceptions import ConfigurationError, NumericalStabilityError


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
        format_string: Custom format string

    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else []),
        ],
    )

    return logging.getLogger("poisson_diffusion")


def check_tensor_validity(
    tensor: torch.Tensor,
    name: str = "tensor",
    allow_nan: bool = False,
    allow_inf: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> None:
    """
    Check tensor for numerical validity.

    Args:
        tensor: Tensor to check
        name: Name for error messages
        allow_nan: Whether NaN values are allowed
        allow_inf: Whether Inf values are allowed
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Raises:
        NumericalStabilityError: If tensor is invalid
    """
    if not allow_nan and torch.isnan(tensor).any():
        raise NumericalStabilityError(f"{name} contains NaN values")

    if not allow_inf and torch.isinf(tensor).any():
        raise NumericalStabilityError(f"{name} contains Inf values")

    if min_val is not None and tensor.min() < min_val:
        raise NumericalStabilityError(
            f"{name} contains values below minimum {min_val}: {tensor.min()}"
        )

    if max_val is not None and tensor.max() > max_val:
        raise NumericalStabilityError(
            f"{name} contains values above maximum {max_val}: {tensor.max()}"
        )


def stabilize_variance(
    variance: torch.Tensor, eps: float = 0.1, name: str = "variance"
) -> torch.Tensor:
    """
    Stabilize variance for numerical stability.

    Args:
        variance: Variance tensor
        eps: Minimum variance value
        name: Name for logging

    Returns:
        Stabilized variance tensor
    """
    original_min = variance.min().item()
    stabilized = torch.clamp(variance, min=eps)

    if original_min < eps:
        warnings.warn(
            f"{name} had values below {eps} (min: {original_min:.6f}), "
            f"clamped for stability"
        )

    return stabilized


def clip_gradients(
    gradients: torch.Tensor, max_norm: float = 10.0, name: str = "gradients"
) -> torch.Tensor:
    """
    Clip gradients for numerical stability.

    Args:
        gradients: Gradient tensor
        max_norm: Maximum allowed norm
        name: Name for logging

    Returns:
        Clipped gradients
    """
    original_norm = gradients.norm().item()
    clipped = torch.clamp(gradients, -max_norm, max_norm)

    if original_norm > max_norm:
        warnings.warn(f"{name} norm {original_norm:.3f} exceeded {max_norm}, clipped")

    return clipped


def ensure_tensor(
    data: Union[torch.Tensor, np.ndarray, float, int],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert input to tensor with specified dtype and device.

    Args:
        data: Input data
        dtype: Target dtype
        device: Target device

    Returns:
        Tensor with specified properties
    """
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        tensor = torch.tensor(data)

    tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)

    return tensor


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        Device to use
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def estimate_memory_usage(
    tensor_shapes: list[Tuple[int, ...]],
    dtype: torch.dtype = torch.float32,
    overhead_factor: float = 1.5,
) -> float:
    """
    Estimate memory usage for tensors.

    Args:
        tensor_shapes: List of tensor shapes
        dtype: Data type
        overhead_factor: Factor for overhead (gradients, etc.)

    Returns:
        Estimated memory usage in GB
    """
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()

    total_elements = sum(np.prod(shape) for shape in tensor_shapes)
    total_bytes = total_elements * bytes_per_element * overhead_factor

    return total_bytes / (1024**3)  # Convert to GB


def validate_domain(domain: str) -> str:
    """
    Validate and normalize domain name.

    Args:
        domain: Domain name

    Returns:
        Normalized domain name

    Raises:
        ConfigurationError: If domain is invalid
    """
    valid_domains = {"photography", "microscopy", "astronomy"}
    domain_lower = domain.lower().strip()

    if domain_lower not in valid_domains:
        raise ConfigurationError(
            f"Invalid domain '{domain}'. Must be one of: {valid_domains}"
        )

    return domain_lower


def create_directory_structure(base_path: Union[str, Path]) -> None:
    """
    Create standard directory structure for the project.

    Args:
        base_path: Base directory path
    """
    base_path = Path(base_path)

    directories = [
        "core",
        "models",
        "data",
        "configs",
        "scripts",
        "tests",
        "docs",
        "logs",
        "outputs",
        "checkpoints",
    ]

    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)

        # Create __init__.py for Python packages
        if directory in ["core", "models", "data"]:
            init_file = base_path / directory / "__init__.py"
            if not init_file.exists():
                init_file.touch()


def format_bytes(bytes_value: float) -> str:
    """
    Format bytes in human-readable format.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class NumericalStabilityHandler:
    """Handler for numerical stability operations."""

    def __init__(
        self,
        eps_variance: float = 0.1,
        grad_clip: float = 10.0,
        range_min: float = 0.0,
        range_max: Optional[float] = None,
    ):
        self.eps_variance = eps_variance
        self.grad_clip = grad_clip
        self.range_min = range_min
        self.range_max = range_max

    def stabilize_variance(self, var: torch.Tensor) -> torch.Tensor:
        """Stabilize variance tensor."""
        return stabilize_variance(var, self.eps_variance)

    def clip_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """Clip gradient tensor."""
        return clip_gradients(grad, self.grad_clip)

    def enforce_range(self, x: torch.Tensor) -> torch.Tensor:
        """Enforce valid range on tensor."""
        return torch.clamp(x, min=self.range_min, max=self.range_max)

    def check_validity(self, tensor: torch.Tensor, name: str) -> None:
        """Check tensor validity."""
        check_tensor_validity(tensor, name)
