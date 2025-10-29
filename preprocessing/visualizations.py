#!/usr/bin/env python3
"""
Visualization utilities for the preprocessing pipeline.

This module provides visualization functions for tiles and processing steps.
"""

import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from preprocessing.utils import get_pixel_stats

logger = logging.getLogger(__name__)


def prepare_for_tile_visualization(data: np.ndarray) -> Optional[np.ndarray]:
    """
    Prepare RGB data for tile visualization.

    Args:
        data: RGB image as numpy array in CHW format (3, H, W)

    Returns:
        RGB array in HWC format (H, W, 3) ready for matplotlib.imshow
    """
    if data is None:
        return None

    # Only handle RGB format (3 channels)
    if len(data.shape) != 3 or data.shape[0] != 3:
        logger.warning(
            f"prepare_for_tile_visualization: Expected RGB format (3, H, W), got {data.shape}"
        )
        return None

    # Transpose from CHW to HWC for RGB display
    display_data = np.transpose(data, (1, 2, 0))

    # Ensure we have 3D data (H, W, 3)
    if len(display_data.shape) != 3 or display_data.shape[2] != 3:
        return None

    return display_data


def simple_normalize(data: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Simple min-max normalization for display (no percentile clipping)."""
    if data is None:
        return None

    # Convert multi-channel to grayscale by averaging
    display_data = np.mean(data, axis=0) if len(data.shape) == 3 else data

    # Min-max normalize
    data_min, data_max = np.min(display_data), np.max(display_data)
    return (
        (display_data - data_min) / (data_max - data_min)
        if data_max > data_min
        else display_data
    )


@dataclass
class VizStepConfig:
    """Configuration for a visualization step."""

    key: str  # Key to extract from data dict
    label: str  # Row label
    precision: int = 3  # Decimal precision for stats
    extract_fn: Optional[Callable] = None  # Custom extraction function


def _plot_image_subplot(
    ax: plt.Axes,
    data: Optional[np.ndarray],
    title: str,
    stats: Optional[Tuple[float, float, float, float]] = None,
    cmap: str = "gray",
    vmin: Optional[float] = 0,
    vmax: Optional[float] = 1,
) -> None:
    """Plot a single image subplot with title and stats."""
    if data is not None:
        kwargs = {"cmap": cmap}
        if vmin is not None:
            kwargs["vmin"] = vmin
        if vmax is not None:
            kwargs["vmax"] = vmax
        ax.imshow(data, **kwargs)
        ax.set_title(title)
    ax.axis("off")


def _get_shape_str(shape_info: Any) -> str:
    """Get shape string from various formats."""
    if hasattr(shape_info, "shape"):
        return str(shape_info.shape)
    elif isinstance(shape_info, tuple):
        return str(shape_info)
    else:
        return str(shape_info)


def _format_stats_str(
    min_val: float,
    max_val: float,
    mean_val: float,
    median_val: float,
    precision: int = 3,
) -> str:
    """Format statistics string."""
    return f"[{min_val:.{precision}f}, {max_val:.{precision}f}] μ={mean_val:.{precision}f} m={median_val:.{precision}f}"


def extract_single_tile_for_viz(tile, tile_size=256):
    """Process an already-extracted tile for visualization purposes.

    Args:
        tile: Already-extracted tile, either as:
            - TileInfo object (from utils.py extract_tiles)
            - numpy array (C, H, W) format
        tile_size: Expected tile size (default: 256)

    Returns:
        Tile as numpy array in (C, H, W) format, padded if necessary
    """
    # Extract tile data if TileInfo object
    tile_data = tile.tile_data if hasattr(tile, "tile_data") else tile

    # Ensure proper shape
    if len(tile_data.shape) == 2:
        tile_data = tile_data[np.newaxis, :, :]

    C, H, W = tile_data.shape

    # Pad if necessary to ensure exact tile_size
    if H != tile_size or W != tile_size:
        padded = np.zeros((C, tile_size, tile_size), dtype=tile_data.dtype)
        padded[:, : min(H, tile_size), : min(W, tile_size)] = tile_data[
            :, : min(H, tile_size), : min(W, tile_size)
        ]
        return padded

    return tile_data


def _create_title(
    domain: str,
    exposure_type: str,
    label: str,
    raw_data: Any,
    stats: Tuple[float, float, float, float],
    precision: int = 3,
    suffix: str = "",
    tile_id: str = None,
) -> str:
    """Create standardized title for visualization subplot."""
    shape_str = _get_shape_str(raw_data)
    stats_str = _format_stats_str(*stats, precision=precision)
    tile_info = f"\nTile ID: {tile_id}" if tile_id and tile_id != "unknown" else ""
    return f"{domain.capitalize()} - {exposure_type.capitalize()} Exposure {label}{suffix}{tile_info}\nShape: {shape_str}\n{stats_str}"


def _create_tensor_title(
    domain: str,
    exposure_type: str,
    tensor: Any,
    stats: Tuple[float, float, float, float],
    tile_id: str = None,
) -> str:
    """Create title for tensor visualization with dtype info."""
    if tensor is None:
        tile_info = f"\nTile ID: {tile_id}" if tile_id and tile_id != "unknown" else ""
        return f"{exposure_type.capitalize()} Exposure Tensor [-1,1]{tile_info}"

    tile_info = f"\nTile ID: {tile_id}" if tile_id and tile_id != "unknown" else ""
    # Show dynamic range prominently
    dynamic_range = f"Dynamic Range: [{stats[0]:.4f}, {stats[1]:.4f}]"
    return (
        f"{domain.capitalize()} - {exposure_type.capitalize()} Exposure{tile_info}\n"
        f"Shape: {tensor.shape}, dtype: {tensor.dtype}\n"
        f"{dynamic_range} | Mean: {stats[2]:.4f} | STD: {stats[3]:.4f}\n"
        f"Target range: [-1, 1] (Display auto-scaled)"
    )


def _process_visualization_step(
    short_data: Dict[str, Any],
    long_data: Dict[str, Any],
    config: VizStepConfig,
    domain: str,
    row_idx: int,
    axes: np.ndarray,
    vmin: Optional[float] = 0,
    vmax: Optional[float] = 1,
) -> None:
    """Process a single visualization step for both short and long exposure."""
    # Extract data
    short_raw = (
        config.extract_fn(short_data.get(config.key))
        if config.extract_fn
        else short_data.get(config.key)
    )
    long_raw = (
        config.extract_fn(long_data.get(config.key))
        if config.extract_fn
        else long_data.get(config.key)
    )

    # Get statistics and prepare display
    short_stats = get_pixel_stats(short_raw)
    long_stats = get_pixel_stats(long_raw)
    short_display = simple_normalize(short_raw)
    long_display = simple_normalize(long_raw)

    # Extract tile_id information
    short_tile_id = short_data.get("corresponding_tile_id", "unknown")
    long_tile_id = long_data.get("corresponding_tile_id", "unknown")

    # Create titles and plot
    short_title = _create_title(
        domain,
        "short",
        config.label,
        short_raw,
        short_stats,
        config.precision,
        tile_id=short_tile_id,
    )
    long_title = _create_title(
        domain,
        "long",
        config.label,
        long_raw,
        long_stats,
        config.precision,
        tile_id=long_tile_id,
    )

    _plot_image_subplot(
        axes[row_idx, 0], short_display, short_title, vmin=vmin, vmax=vmax
    )
    _plot_image_subplot(
        axes[row_idx, 1], long_display, long_title, vmin=vmin, vmax=vmax
    )


def _process_tensor_step(
    short_data: Dict[str, Any],
    long_data: Dict[str, Any],
    domain: str,
    row_idx: int,
    axes: np.ndarray,
) -> None:
    """Process tensor visualization step (handles torch tensors)."""
    short_tensor = short_data.get("tensor")
    long_tensor = long_data.get("tensor")

    # Convert to numpy and get stats
    short_np = short_tensor.numpy() if short_tensor is not None else None
    long_np = long_tensor.numpy() if long_tensor is not None else None
    short_stats = get_pixel_stats(short_np)
    long_stats = get_pixel_stats(long_np)

    # Normalize tensor data from [-1,1] to [0,1] for consistent display
    short_np_normalized = (short_np + 1) / 2 if short_np is not None else None
    long_np_normalized = (long_np + 1) / 2 if long_np is not None else None

    # Prepare display data (same normalization as after tiling step)
    short_display = simple_normalize(short_np_normalized)
    long_display = simple_normalize(long_np_normalized)

    # Extract tile_id information
    short_tile_id = short_data.get("corresponding_tile_id", "unknown")
    long_tile_id = long_data.get("corresponding_tile_id", "unknown")

    # Create titles and plot (with same vmin/vmax as other steps)
    short_title = _create_tensor_title(
        domain, "short", short_tensor, short_stats, short_tile_id
    )
    long_title = _create_tensor_title(
        domain, "long", long_tensor, long_stats, long_tile_id
    )

    _plot_image_subplot(axes[row_idx, 0], short_display, short_title, vmin=0, vmax=1)
    _plot_image_subplot(axes[row_idx, 1], long_display, long_title, vmin=0, vmax=1)


def create_scene_visualization(viz_data: Dict[str, Any], output_dir: Path):
    """Create 4-step visualization for a single scene (one short + one long exposure pair)"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        domain = viz_data["domain"]
        sensor = viz_data.get(
            "sensor", domain
        )  # Fallback to domain if sensor not present
        scene_id = viz_data["scene_id"]
        short_data = viz_data["short"]
        long_data = viz_data["long"]

        # Extract tile_id information for display
        short_tile_id = short_data.get("corresponding_tile_id", "unknown")
        long_tile_id = long_data.get("corresponding_tile_id", "unknown")

        # Create figure with 3x2 subplots (3 steps x 2 images)
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))

        # Define visualization steps
        viz_steps = [
            VizStepConfig(key="raw_image", label="Raw", precision=3),
            VizStepConfig(key="tiled_image", label="Tiled", precision=3),
        ]

        # Process first 2 steps using common logic
        for idx, step_config in enumerate(viz_steps):
            _process_visualization_step(
                short_data, long_data, step_config, domain, idx, axes
            )

        # Process tensor step separately (has special handling)
        _process_tensor_step(short_data, long_data, domain, 2, axes)

        # Add column labels (aligned with subplot centers)
        fig.text(
            0.25,
            0.98,
            "Short Exposure Samples",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.75,
            0.98,
            "Long Exposure Samples",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

        # Add row labels (aligned with 3-row grid)
        row_labels = [
            "Raw Loading [0,1] normalized",
            "After Tiling",
            "Final Tensor [-1,1] (see dynamic range in title)",
        ]
        # Adjust vertical positions for 3 rows instead of 4
        row_positions = [0.83, 0.50, 0.17]  # Top, middle, bottom
        for j, (label, pos) in enumerate(zip(row_labels, row_positions)):
            fig.text(
                0.01,
                pos,
                label,
                ha="left",
                va="center",
                fontsize=12,
                fontweight="bold",
                rotation=90,
            )

        # Add overall title
        fig.suptitle(f"{sensor}_{scene_id}", fontsize=16, fontweight="bold")

        # Save the plot
        safe_scene_id = scene_id.replace("/", "_").replace(":", "_")
        output_path = output_dir / f"{sensor}_{safe_scene_id}_steps.png"
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, left=0.1, right=0.95)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return str(output_path)

    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
