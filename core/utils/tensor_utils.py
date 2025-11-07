"""
Tensor utility functions for the Poisson-Gaussian Diffusion project.

This module provides tensor operations, validation, and conversion utilities.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from core.error_handlers import NumericalStabilityError

logger = logging.getLogger(__name__)


def check_tensor_validity(
    tensor: torch.Tensor,
    name: str = "tensor",
    allow_nan: bool = False,
    allow_inf: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> None:
    """Check tensor for numerical validity.

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


def ensure_tensor(
    data: Union[torch.Tensor, np.ndarray, float, int],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert input to tensor with specified dtype and device.

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
    """Get the best available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        Device to use
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def estimate_memory_usage(
    tensor_shapes: List[Tuple[int, ...]],
    dtype: torch.dtype = torch.float32,
    overhead_factor: float = 1.5,
) -> float:
    """Estimate memory usage for tensors.

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
    return total_bytes / (1024**3)


def format_bytes(bytes_value: float) -> str:
    """Format bytes in human-readable format.

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
    """Format time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def tensor_to_numpy(tensor: torch.Tensor, select_first: bool = True) -> np.ndarray:
    """
    Convert tensor to numpy array for plotting.

    Args:
        tensor: Input tensor (can be multi-dimensional)
        select_first: If True, select first batch and channel for multi-dimensional tensors

    Returns:
        Numpy array ready for plotting
    """
    if select_first and tensor.dim() > 3:
        # Take first batch and first channel if multi-dimensional
        tensor = tensor[0, 0] if tensor.dim() > 3 else tensor.squeeze()
    elif tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    return tensor.detach().cpu().numpy()


def get_image_range(tensor: torch.Tensor) -> dict:
    """
    Get min and max range of an image tensor.

    Args:
        tensor: Input tensor (can be multi-dimensional)

    Returns:
        Dictionary with 'min' and 'max' keys
    """
    if isinstance(tensor, torch.Tensor):
        return {
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
        }
    elif isinstance(tensor, np.ndarray):
        return {
            "min": float(np.min(tensor)),
            "max": float(np.max(tensor)),
        }
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")


def _ensure_chw_format(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is in CHW format (channels, height, width)."""
    if tensor.ndim == 2:
        logger.warning(
            f"Received 2D tensor (H, W) - expected RGB image with 3 channels"
        )
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    elif tensor.ndim == 3:
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] != 3:
            logger.warning(
                f"Unexpected 3D tensor shape {tensor.shape} - expected [C, H, W] or [H, W, C] with C=3"
            )
    return tensor


def _validate_channels(
    tensor: torch.Tensor, target_channels: int, image_type: str = "", tile_id: str = ""
) -> torch.Tensor:
    """Validate tensor has expected number of channels."""
    if tensor.shape[0] != target_channels and image_type and tile_id:
        logger.warning(
            f"Channel mismatch for {image_type} image {tile_id}: "
            f"got {tensor.shape[0]} channels, expected {target_channels} channels"
        )
    return tensor


def _clamp_to_01(tensor: torch.Tensor) -> torch.Tensor:
    """Clamp tensor values to [0, 1] range."""
    return torch.clamp(tensor, 0.0, 1.0)


def save_tensor(tensor: torch.Tensor, file_path: Union[str, Path]) -> None:
    """Save tensor to file.

    Args:
        tensor: Tensor to save
        file_path: Path to save the tensor
    """
    file_path = Path(file_path)
    torch.save(tensor.cpu(), file_path)


def get_pixel_stats(data: np.ndarray) -> Tuple[float, float, float, float]:
    """Get pixel statistics (min, max, mean, median) from data.

    Returns 0.0 for all metrics if data is None or contains no valid values.
    Uses numpy NaN-aware functions to handle invalid values gracefully.

    Args:
        data: Input array (any shape)

    Returns:
        Tuple of (min_val, max_val, mean_val, median_val) as float
        All values are 0.0 if data is invalid/empty (for type consistency)
    """
    if data is None or data.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    flat_data = data.flatten()
    try:
        return (
            float(np.nanmin(flat_data)),
            float(np.nanmax(flat_data)),
            float(np.nanmean(flat_data)),
            float(np.nanmedian(flat_data)),
        )
    except (ValueError, RuntimeError):
        return 0.0, 0.0, 0.0, 0.0


def extract_pixel_values_from_data(data: np.ndarray) -> np.ndarray:
    """Extract all pixel values from data array, handling different shapes."""
    if data is None:
        return np.array([])
    return data.flatten()
