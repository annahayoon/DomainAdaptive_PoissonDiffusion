"""
Utilities for loading and processing image tensors.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def load_image_tensor(image_path: Path) -> Tuple[Optional[np.ndarray], bool]:
    """
    Load a .pt file and return as numpy array.

    Args:
        image_path: Path to .pt file

    Returns:
        Tuple of (image_array, is_rgb)
    """
    if not image_path.exists():
        return None, False

    try:
        tensor = torch.load(image_path)
        # Convert to numpy and squeeze singleton dimensions
        array = tensor.numpy()
        is_rgb = False

        # Handle different tensor shapes
        if len(array.shape) == 4:  # (1, 1, 256, 256) or (1, 3, 256, 256)
            if array.shape[1] == 3:  # RGB
                array = array[0]  # Remove batch dimension, keep (3, 256, 256)
                is_rgb = True
            else:
                array = array[0, 0]  # Remove batch and channel dimensions
        elif len(array.shape) == 3:
            if array.shape[0] == 1:  # (1, 256, 256)
                array = array[0]  # Remove channel dimension
            elif array.shape[0] == 3:  # (3, 256, 256) - RGB image
                is_rgb = True
                # array stays as (3, 256, 256)
            else:  # (256, 256, 1) or similar
                array = array.squeeze()
        elif len(array.shape) == 2:  # (256, 256)
            pass  # Already correct shape
        else:
            print(f"Unexpected array shape: {array.shape} for {image_path}")
            return None, False

        return array, is_rgb
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, False


def load_tensor_from_pt(pt_path: Path) -> torch.Tensor:
    """Load tensor from .pt file."""
    return torch.load(pt_path, map_location="cpu")


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
