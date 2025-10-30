#!/usr/bin/env python3
"""
Visualization utilities for image comparison and analysis.

This module provides functions for creating comprehensive comparison visualizations
showing input images, enhanced results, and metrics.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from sample.sample_utils import (
    denormalize_to_physical,
    load_sensor_calibration_from_metadata,
)

logger = logging.getLogger(__name__)


def _to_display_image(phys_array: np.ndarray) -> np.ndarray:
    """Convert physical array to RGB display format (H, W, 3).

    Args:
        phys_array: Array in (C, H, W) or (B, C, H, W) format with C=3 (RGB)

    Returns:
        RGB image array in (H, W, 3) format
    """
    if phys_array.ndim == 4:
        phys_array = phys_array[0]

    if phys_array.ndim == 3:
        if phys_array.shape[0] != 3:
            raise ValueError(
                f"Expected 3 RGB channels, got {phys_array.shape[0]} channels"
            )
        img = np.transpose(phys_array, (1, 2, 0))
    else:
        raise ValueError(
            f"Unexpected array shape: {phys_array.shape} (expected 3D (C, H, W) or 4D (B, C, H, W))"
        )
    return img


def _get_column_color(key: str) -> Optional[str]:
    """Get color for a column based on its key."""
    if "pg" in key:
        return "green"
    elif "gaussian" in key:
        return "orange"
    elif key == "short":
        return "blue"
    elif key == "exposure_scaled":
        return "purple"
    return None


def _format_metrics_text(metrics: Dict[str, float], key: str, sensor_type: str) -> str:
    """Format metrics text for display."""
    if key == "short":
        return f"PSNR: {metrics['psnr']:.1f}dB"

    metrics_lines = []
    metrics_lines.append(f"SSIM: {metrics['ssim']:.3f}")
    metrics_lines.append(f"PSNR: {metrics['psnr']:.1f}dB")

    if "lpips" in metrics:
        if np.isnan(metrics["lpips"]):
            metrics_lines.append("LPIPS: N/A")
        else:
            metrics_lines.append(f"LPIPS: {metrics['lpips']:.3f}")
    if "niqe" in metrics:
        if np.isnan(metrics["niqe"]):
            metrics_lines.append("NIQE: N/A")
        else:
            metrics_lines.append(f"NIQE: {metrics['niqe']:.1f}")

    return "\n".join(metrics_lines)


def _add_row_label(
    ax: Any,
    label: str,
    facecolor: str,
) -> None:
    """Add a row label to the left side of an axis.

    Args:
        ax: Matplotlib axis object
        label: Label text to display
        facecolor: Background color for the label box
    """
    ax.text(
        -0.08,
        0.5,
        label,
        transform=ax.transAxes,
        fontsize=9,
        va="center",
        ha="right",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor=facecolor, alpha=0.7),
    )


def create_comprehensive_comparison(
    short_image: torch.Tensor,
    enhancement_results: Dict[str, torch.Tensor],
    sensor_type: str,
    tile_id: str,
    save_path: Path,
    metadata_json_path: Path,
    long_image: Optional[torch.Tensor] = None,
    exposure_ratio: float = 1.0,
    metrics_results: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Create comprehensive comparison visualization with all guidance variants.

    Layout:
    Row 0: Input names and [min, max] ADU range
    Row 1: PG (x0) scale
    Row 2: Individual dynamic range (min/max for each method)
    Row 3: Metrics: SSIM, PSNR, LPIPS, NIQE

    Columns: Short Exposure, optionally Long Exposure Reference, Exposure Scaled, and enhancement methods (Gaussian, PG, etc.)

    Args:
        short_image: Short exposure input image tensor
        enhancement_results: Dictionary with enhancement results
        sensor_type: Sensor type for range lookup
        tile_id: Tile ID for title
        save_path: Path to save the visualization
        long_image: Optional long exposure reference image
        exposure_ratio: Exposure ratio for scaling
        metrics_results: Dictionary with metrics for each method
        metadata_json_path: Path to metadata JSON file for loading calibration values
    """
    method_priority = [
        ("gaussian_x0", "Gaussian (x0)"),
        ("gaussian", "Gaussian (x0)"),
        ("pg_x0", "PG (x0)"),
        ("pg", "PG (x0)"),
    ]

    available_methods = []
    method_labels = {}
    seen_labels = set()

    for method_key, label in method_priority:
        if method_key in enhancement_results and label not in seen_labels:
            available_methods.append(method_key)
            method_labels[method_key] = label
            seen_labels.add(label)

    has_long = long_image is not None

    n_cols = (
        1
        + (1 if has_long else 0)
        + (1 if "exposure_scaled" in enhancement_results else 0)
        + len(available_methods)
    )
    n_rows = 4

    fig = plt.figure(figsize=(3.0 * n_cols, 9))
    width_ratios = [1.0] * n_cols

    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.05,
        hspace=0.12,
        height_ratios=[0.2, 1.0, 1.0, 0.3],
    )

    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col_idx in range(n_cols):
            axes[row, col_idx] = fig.add_subplot(gs[row, col_idx])

    # Load sensor calibration values from metadata JSON
    black_level, white_level = load_sensor_calibration_from_metadata(
        sensor_type, metadata_json_path
    )
    unit_label = "ADU"

    def process_image(tensor: torch.Tensor, key: str) -> tuple:
        """Convert tensor to display format and compute range."""
        phys = denormalize_to_physical(tensor, black_level, white_level)
        img = _to_display_image(phys)
        img_range = (float(img.min()), float(img.max()))
        return img, img_range

    BASE_COLUMN_LABELS = {
        "short": "Short Exposure",
        "long": "Long Exposure Reference",
        "exposure_scaled": "Exposure Scaled",
    }

    def get_column_label(key: str) -> str:
        """Get display label for a column."""
        return BASE_COLUMN_LABELS.get(key, method_labels.get(key, key))

    # Build image processing list and columns in one pass
    images = {}
    ranges = {}
    columns = []

    image_sources = [
        ("short", short_image),
    ]
    if has_long:
        image_sources.append(("long", long_image))
    if "exposure_scaled" in enhancement_results:
        image_sources.append(
            ("exposure_scaled", enhancement_results["exposure_scaled"])
        )
    for method in available_methods:
        if method in enhancement_results:
            image_sources.append((method, enhancement_results[method]))

    for key, tensor in image_sources:
        images[key], ranges[key] = process_image(tensor, key)
        columns.append({"key": key, "label": get_column_label(key)})

    ref_method = None
    for candidate in ["pg_x0", "pg"]:
        if candidate in images:
            ref_method = candidate
            break
    if ref_method is None:
        ref_method = available_methods[0] if available_methods else "short"

    ref_min, ref_max = ranges[ref_method]
    short_min, short_max = ranges["short"]

    def render_column(row: int, col: int, key: str) -> None:
        """Helper to render a single column cell and turn off axis."""
        ax = axes[row, col]
        if row == 0:
            if key in images and key in ranges:
                min_val, max_val = ranges[key]
                color = _get_column_color(key)
                label = get_column_label(key)

                text_kwargs = {
                    "transform": ax.transAxes,
                    "ha": "center",
                    "va": "center",
                    "fontsize": 8,
                    "fontweight": "bold",
                }
                if color:
                    text_kwargs["color"] = color

                ax.text(
                    0.5,
                    0.5,
                    f"{label}\n[{min_val:.0f}, {max_val:.0f}] {unit_label}",
                    **text_kwargs,
                )
        elif row == 1:
            if key in images:
                ax.imshow(images[key], vmin=ref_min, vmax=ref_max)
        elif row == 2:
            if key in images and key in ranges:
                min_val, max_val = (
                    ranges[key] if key != "short" else (short_min, short_max)
                )
                ax.imshow(images[key], vmin=min_val, vmax=max_val)
        elif row == 3:
            metrics_text = ""
            text_color = None
            fontsize = 7

            if key == "long":
                metrics_text = ""
            elif key == "short" and metrics_results and key in metrics_results:
                metrics_text = _format_metrics_text(
                    metrics_results[key], key, sensor_type
                )
                text_color = "blue"
            elif metrics_results and key in metrics_results:
                metrics_text = _format_metrics_text(
                    metrics_results[key], key, sensor_type
                )
                text_color = _get_column_color(key)
            elif key not in ("short", "long"):
                metrics_text = "(No long\nreference)"
                text_color = "gray"
                fontsize = 6

            if metrics_text:
                text_kwargs = {
                    "transform": ax.transAxes,
                    "ha": "center",
                    "va": "center",
                    "fontsize": fontsize,
                }
                if text_color:
                    text_kwargs["color"] = text_color
                if "(No long" in metrics_text:
                    text_kwargs["style"] = "italic"
                else:
                    text_kwargs["fontweight"] = "bold"
                ax.text(0.5, 0.5, metrics_text, **text_kwargs)
        ax.axis("off")

    for row in range(n_rows):
        for col, col_def in enumerate(columns):
            render_column(row, col, col_def["key"])

    row_labels = [
        ("Input Names\n& Ranges", "lightcoral"),
        ("PG (x0)\nScale", "lightgreen"),
        ("Individual\nDynamic Range", "lightblue"),
        ("Metrics\nSSIM, PSNR, LPIPS, NIQE", "lightyellow"),
    ]
    for row_idx, (label, facecolor) in enumerate(row_labels):
        _add_row_label(axes[row_idx, 0], label, facecolor)

    plt.suptitle(
        f"Comprehensive Enhancement Comparison - {tile_id}\n"
        f"Row 0: Input Names & Ranges | Row 1: PG (x0) Scale | Row 2: Individual Dynamic Range | Row 3: Metrics | α={exposure_ratio:.4f}",
        fontsize=9,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comprehensive comparison visualization: {save_path}")
