"""
Utilities for visualization: normalization, formatting, colors.
"""
from typing import Dict, Optional, Tuple

import numpy as np


def normalize_for_display(
    data: np.ndarray, percentile_range: Tuple[float, float] = (1, 99)
) -> Optional[np.ndarray]:
    """
    Normalize data for display using percentile clipping.

    Args:
        data: Input data array (can be CHW or HWC format for RGB)
        percentile_range: Percentile range for clipping

    Returns:
        Normalized array or None if invalid
    """
    if data is None:
        return None

    # Handle different data shapes
    if len(data.shape) == 3:
        if data.shape[0] == 3:  # RGB format (C, H, W) - convert to grayscale
            # Convert RGB to grayscale for display (consistent with preprocessing)
            display_data = np.mean(data, axis=0)
        elif data.shape[2] == 3:  # RGB format (H, W, C) - already in display format
            # For HWC format, convert to grayscale for consistency
            display_data = np.mean(data, axis=2)
        elif data.shape[0] == 4:  # RGGB format (C, H, W)
            display_data = data[0]  # Use red channel
        else:
            display_data = data.mean(axis=0)
    else:
        display_data = data

    # Ensure we have 2D data for display
    if len(display_data.shape) != 2:
        print(f"Warning: Expected 2D data for display, got shape {display_data.shape}")
        return None

    # Remove NaN and infinite values
    valid_mask = np.isfinite(display_data)
    if not np.any(valid_mask):
        return None

    # Clip to percentiles and normalize to [0, 1]
    p_low, p_high = np.percentile(display_data[valid_mask], percentile_range)
    if p_high <= p_low:
        return np.zeros_like(display_data)

    clipped = np.clip(display_data, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low)

    return normalized


def format_method_name(method: str) -> str:
    """Format method name for display."""
    method_names = {
        "noisy": "Noisy Input",
        "clean": "Clean Reference",
        "exposure_scaled": "Exposure Scaled",
        "gaussian_x0": "Gaussian x0",
        "gaussian_x0_cross": "Gaussian x0-cross",
        "pg_x0_single": "PG x0-single",
        "pg_x0": "PG x0",
        "pg_x0_cross": "PG x0-cross",
    }
    return method_names.get(method, method.replace("_", " ").title())


def get_method_colors() -> Dict[str, str]:
    """Get color mapping for methods."""
    return {
        "noisy": "gray",
        "clean": "white",
        "exposure_scaled": "#3498db",  # Blue
        "gaussian_x0": "#e67e22",  # Orange
        "gaussian_x0_cross": "#d35400",  # Dark orange
        "pg_x0": "#27ae60",  # Green
        "pg_x0_single": "#27ae60",  # Green
        "pg_x0_cross": "#229954",  # Dark green
    }


def get_method_filename_map() -> Dict[str, str]:
    """Map method names to filenames."""
    return {
        "noisy": "noisy.pt",
        "clean": "clean.pt",
        "exposure_scaled": "restored_exposure_scaled.pt",
        "gaussian_x0": "restored_gaussian_x0.pt",
        "pg_x0_single": "restored_pg_x0.pt",
        "gaussian_x0_cross": "restored_gaussian_x0_cross.pt",
        "pg_x0_cross": "restored_pg_x0_cross.pt",
    }
