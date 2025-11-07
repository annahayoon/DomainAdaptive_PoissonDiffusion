#!/usr/bin/env python3
"""
Consolidated visualization utilities for the entire project.

This module contains all visualization functions organized by category:
- General utilities: normalization, formatting, colors, tensor conversion
- Preprocessing visualizations: tile processing and scene visualization
- Sample visualizations: image comparison and analysis
- Comparison plots: 4-panel comparisons and sensor comparisons
- Residual plots: residual analysis and validation plots
"""

import json
import logging
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from scipy import stats

try:
    import seaborn as sns

    sns.set_palette("husl")
except ImportError:
    sns = None

# Import dependencies
from config.config import VizStepConfig

try:
    from core.utils.sensor_utils import demosaic_raw_to_rgb
    from core.utils.tensor_utils import get_pixel_stats
except ImportError:
    # Fallback if preprocessing is not available
    get_pixel_stats = None
    demosaic_raw_to_rgb = None

try:
    from core.error_handlers import AnalysisError
    from core.metrics import MetricResult, PhysicsMetrics
    from core.residual_analysis import ResidualValidationReport
except ImportError:
    PhysicsMetrics = None
    MetricResult = None
    AnalysisError = Exception
    ResidualValidationReport = None

try:
    from config.sample_config import (
        DEFAULT_DPI,
        DEFAULT_FIGSIZE_HEIGHT,
        DEFAULT_FIGSIZE_WIDTH,
        HIGH_DPI,
        SCENE_FIGSIZE_HEIGHT,
        SCENE_FIGSIZE_WIDTH,
    )
    from core.utils.sensor_utils import (
        denormalize_to_physical,
        load_sensor_calibration_from_metadata,
    )
except ImportError:
    # Fallback defaults
    DEFAULT_DPI = 150
    DEFAULT_FIGSIZE_HEIGHT = 6
    DEFAULT_FIGSIZE_WIDTH = 8
    HIGH_DPI = 300
    SCENE_FIGSIZE_HEIGHT = 8
    SCENE_FIGSIZE_WIDTH = 12
    denormalize_to_physical = None
    load_sensor_calibration_from_metadata = None

logger = logging.getLogger(__name__)

from core.utils.file_utils import load_tensor_from_pt, save_json_file

# Import additional utilities needed for scaling
from core.utils.tensor_utils import get_image_range, save_tensor

# Import all utilities from visualization_utils
from core.utils.visualization_utils import (  # General utilities; Preprocessing visualization utilities; Sample visualization utilities; Sensor comparison utilities; Common utilities; Logging and argument parsing utilities
    BASE_COLUMN_LABELS,
    METHOD_PRIORITY,
    REF_METHOD_CANDIDATES,
    ROW_LABELS,
    SCENE_METHOD_ORDER,
    _add_row_label,
    _add_text_with_bbox,
    _aggregate_method_metrics,
    _aggregate_metric_values,
    _build_image_sources,
    _create_tensor_title,
    _create_title,
    _extract_brightness_value,
    _extract_metrics_from_tile,
    _finalize_figure,
    _find_reference_method,
    _format_metrics_text,
    _format_optional_metric,
    _format_stats_str,
    _get_column_color,
    _get_column_label,
    _get_metrics_text_and_style,
    _get_pixel_range,
    _get_shape_str,
    _normalize_array_shape,
    _plot_image_subplot,
    _process_image_for_display,
    _select_available_methods,
    _tensor_to_display_image,
    _tensor_to_numpy,
    _to_display_image,
    _transpose_to_hwc,
    add_common_visualization_arguments,
    compute_scene_aggregate_metrics,
    ensure_output_dir,
    extract_brightness_range,
    extract_single_tile_for_viz,
    format_method_name,
    get_cross_sensor_results_path,
    get_method_colors,
    get_method_filename_map,
    get_optimized_results_path,
    get_preprocessing_base_path,
    get_project_root,
    get_results_base_path,
    get_sensor_display_name,
    get_sensor_pixel_range,
    get_sensor_scaling_factor,
    get_sensor_to_optimized_dir,
    load_photography_raw,
    load_results_json,
    normalize_for_display,
    normalize_image_to_sensor_range,
    prepare_for_tile_visualization,
    resolve_output_path,
    setup_visualization_logging,
    setup_visualization_paths,
    simple_normalize,
)

# ============================================================================
# PREPROCESSING VISUALIZATIONS
# ============================================================================


def _process_visualization_step(
    short_data: Dict[str, Any],
    long_data: Dict[str, Any],
    config: VizStepConfig,
    sensor: str,
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
    if get_pixel_stats:
        short_stats = get_pixel_stats(short_raw)
        long_stats = get_pixel_stats(long_raw)
    else:
        # Fallback if get_pixel_stats is not available
        short_stats = (0.0, 1.0, 0.5, 0.5)
        long_stats = (0.0, 1.0, 0.5, 0.5)

    short_display = simple_normalize(short_raw)
    long_display = simple_normalize(long_raw)

    # Extract tile_id information
    short_tile_id = short_data.get("corresponding_tile_id", "unknown")
    long_tile_id = long_data.get("corresponding_tile_id", "unknown")

    # Create titles and plot
    short_title = _create_title(
        sensor,
        "short",
        config.label,
        short_raw,
        short_stats,
        config.precision,
        tile_id=short_tile_id,
    )
    long_title = _create_title(
        sensor,
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
    sensor: str,
    row_idx: int,
    axes: np.ndarray,
) -> None:
    """Process tensor visualization step (handles torch tensors)."""
    short_tensor = short_data.get("tensor")
    long_tensor = long_data.get("tensor")

    # Convert to numpy and get stats
    short_np = short_tensor.numpy() if short_tensor is not None else None
    long_np = long_tensor.numpy() if long_tensor is not None else None

    if get_pixel_stats:
        short_stats = get_pixel_stats(short_np)
        long_stats = get_pixel_stats(long_np)
    else:
        short_stats = (0.0, 1.0, 0.5, 0.5)
        long_stats = (0.0, 1.0, 0.5, 0.5)

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
        sensor, "short", short_tensor, short_stats, short_tile_id
    )
    long_title = _create_tensor_title(
        sensor, "long", long_tensor, long_stats, long_tile_id
    )

    _plot_image_subplot(axes[row_idx, 0], short_display, short_title, vmin=0, vmax=1)
    _plot_image_subplot(axes[row_idx, 1], long_display, long_title, vmin=0, vmax=1)


def create_scene_visualization(viz_data: Dict[str, Any], output_dir: Path):
    """Create visualization for a single scene (one short + one long exposure pair)"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        sensor = viz_data.get("sensor", "unknown")
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
                short_data, long_data, step_config, sensor, idx, axes
            )

        # Process tensor step separately (has special handling)
        _process_tensor_step(short_data, long_data, sensor, 2, axes)

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


# ============================================================================
# SAMPLE VISUALIZATIONS
# ============================================================================


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
) -> None:
    """
    Create comprehensive comparison visualization with all guidance variants.

    Layout:
    Row 0: Input names and [min, max] ADU range
    Row 1: Individual dynamic range (min/max for each method)
    Row 2: All images scaled to PG x0 range
    Row 3: Metrics: SSIM, PSNR, LPIPS, NIQE

    Columns: Short Exposure, optionally Long Exposure Reference, Exposure Scaled,
    and enhancement methods (Gaussian, PG, etc.)
    """
    if load_sensor_calibration_from_metadata is None:
        raise ImportError("load_sensor_calibration_from_metadata is not available")

    available_methods, method_labels = _select_available_methods(enhancement_results)
    has_long = long_image is not None

    n_cols = (
        1
        + (1 if has_long else 0)
        + (1 if "exposure_scaled" in enhancement_results else 0)
        + len(available_methods)
    )
    n_rows = 4

    fig = plt.figure(figsize=(DEFAULT_FIGSIZE_WIDTH * n_cols, DEFAULT_FIGSIZE_HEIGHT))

    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        width_ratios=[1.0] * n_cols,
        wspace=0.05,
        hspace=0.12,
        height_ratios=[0.2, 1.0, 1.0, 0.3],
    )

    axes = np.array(
        [
            [fig.add_subplot(gs[row, col]) for col in range(n_cols)]
            for row in range(n_rows)
        ],
        dtype=object,
    )

    black_level, white_level = load_sensor_calibration_from_metadata(sensor_type)
    unit_label = "ADU"

    images = {}
    ranges = {}

    image_sources = _build_image_sources(
        short_image, long_image, enhancement_results, available_methods
    )

    for key, tensor in image_sources:
        images[key], ranges[key] = _process_image_for_display(
            tensor, black_level, white_level
        )

    pg_x0_key = _find_reference_method(ranges, available_methods)
    pg_range = ranges.get(pg_x0_key) if pg_x0_key else None

    def render_column(row: int, col: int, key: str) -> None:
        ax = axes[row, col]
        if row == 0 and key in ranges:
            min_val, max_val = ranges[key]
            _add_text_with_bbox(
                ax,
                0.5,
                0.5,
                f"{_get_column_label(key, method_labels)}\n[{min_val:.0f}, {max_val:.0f}] {unit_label}",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                color=_get_column_color(key),
            )
        elif row == 1 and key in images:
            ax.imshow(images[key])
        elif row == 2 and key in images and pg_range:
            pg_min, pg_max = pg_range
            ax.imshow(images[key], vmin=pg_min, vmax=pg_max)
        elif row == 3:
            (
                metrics_text,
                text_color,
                fontsize,
                style,
                fontweight,
            ) = _get_metrics_text_and_style(key, metrics_results)
            if metrics_text:
                _add_text_with_bbox(
                    ax,
                    0.5,
                    0.5,
                    metrics_text,
                    fontsize=fontsize,
                    fontweight=fontweight,
                    ha="center",
                    va="center",
                    color=text_color,
                    style=style,
                )
        ax.axis("off")

    for row in range(n_rows):
        for col, (key, _) in enumerate(image_sources):
            render_column(row, col, key)

    for row_idx, (label, facecolor) in enumerate(ROW_LABELS):
        _add_row_label(axes[row_idx, 0], label, facecolor)

    plt.suptitle(
        f"Comprehensive Enhancement Comparison - {tile_id}\n"
        f"Row 0: Input Names & Ranges | Row 1: Individual Dynamic Range | Row 2: PG x0 Scaled | Row 3: Metrics | α={exposure_ratio:.4f}",
        fontsize=9,
        fontweight="bold",
    )

    _finalize_figure(
        fig, save_path=save_path, show_plot=False, dpi=HIGH_DPI, bbox_inches="tight"
    )


def plot_noise_calibration(
    bin_means: np.ndarray,
    bin_vars: np.ndarray,
    a_norm: float,
    b_norm: float,
    domain: str = "[-1,1]",
    save_path: Optional[Path] = None,
    show_plot: bool = True,
) -> None:
    """
    Visualize the fitted noise model with measured data.

    Creates a scatter plot of measured variance vs signal mean with the fitted
    linear model overlaid. Useful for validating the quality of noise parameter estimation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(bin_means, bin_vars, alpha=0.6, s=50, label="Measured variance")

    mean_range = np.linspace(bin_means.min(), bin_means.max(), 100)
    fitted_vars = a_norm * mean_range + b_norm
    ax.plot(
        mean_range,
        fitted_vars,
        "r-",
        linewidth=2,
        label=f"Fit: var = {a_norm:.4e} · mean + {b_norm:.4e}",
    )

    ax.set_xlabel(f"Signal Mean (normalized {domain})", fontsize=12)
    ax.set_ylabel("Noise Variance", fontsize=12)
    ax.set_title("Poisson-Gaussian Noise Calibration", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _add_text_with_bbox(
        ax,
        0.05,
        0.95,
        f"a (Poisson) = {a_norm:.6e}\nb (Gaussian) = {b_norm:.6e}",
        fontsize=10,
        fontweight="normal",
        ha="left",
        va="top",
        facecolor="wheat",
        alpha=0.5,
    )

    _finalize_figure(
        fig,
        save_path=save_path,
        show_plot=show_plot,
        dpi=DEFAULT_DPI,
        bbox_inches="tight",
    )


# Note: Utility functions _extract_brightness_value, _extract_metrics_from_tile,
# _aggregate_metric_values, compute_scene_aggregate_metrics, _get_pixel_range,
# and _aggregate_method_metrics are imported from visualization_utils.py above.


def create_scene_visualization_sample(
    stitched_images: Dict[str, torch.Tensor],
    scene_id: str,
    sensor: str,
    scene_dir: Path,
    scene_metrics: List[Dict[str, Any]],
) -> None:
    """
    Create visualization showing stitched tiles from all methods in a single row.
    Each column shows: pixel value range [min, max], image, and metrics.
    """
    if load_sensor_calibration_from_metadata is None:
        raise ImportError("load_sensor_calibration_from_metadata is not available")

    available_methods = [m for m in SCENE_METHOD_ORDER if m in stitched_images]
    if not available_methods:
        return

    black_level, white_level = load_sensor_calibration_from_metadata(sensor)
    unit_label = "ADU"

    pixel_ranges = {}
    display_images = {}
    for method in available_methods:
        # Get pixel range for display
        pixel_ranges[method] = _get_pixel_range(
            stitched_images[method], black_level, white_level
        )
        # Convert to display image with proper denormalization and color handling
        display_img, _ = _process_image_for_display(
            stitched_images[method], black_level, white_level
        )
        # Convert to uint8 for matplotlib
        display_images[method] = (np.clip(display_img, 0, 1) * 255).astype(np.uint8)

    method_metrics = {}
    if scene_metrics:
        metric_keys = ["ssim", "psnr", "lpips", "niqe"]
        for method in available_methods:
            aggregated = _aggregate_method_metrics(scene_metrics, method, metric_keys)
            if aggregated:
                method_metrics[method] = aggregated

    num_methods = len(available_methods)
    fig = plt.figure(figsize=(SCENE_FIGSIZE_WIDTH, SCENE_FIGSIZE_HEIGHT))
    gs = GridSpec(3, num_methods, figure=fig, hspace=0.15, wspace=0.05)

    for col_idx, method in enumerate(available_methods):
        min_val, max_val = pixel_ranges[method]
        method_label = BASE_COLUMN_LABELS.get(method, method.upper())
        color = _get_column_color(method)

        ax_range = fig.add_subplot(gs[0, col_idx])
        _add_text_with_bbox(
            ax_range,
            0.5,
            0.5,
            f"{method_label}\n[{min_val:.0f}, {max_val:.0f}] {unit_label}",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            color=color,
        )
        ax_range.axis("off")

        ax_img = fig.add_subplot(gs[1, col_idx])
        ax_img.imshow(display_images[method])
        ax_img.axis("off")

        ax_metrics = fig.add_subplot(gs[2, col_idx])
        if method in method_metrics:
            metrics_text = _format_metrics_text(method_metrics[method], method)
            text_color = "blue" if method == "short" else color
            _add_text_with_bbox(
                ax_metrics,
                0.5,
                0.5,
                metrics_text,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                color=text_color,
            )
        elif method not in ("short", "long"):
            _add_text_with_bbox(
                ax_metrics,
                0.5,
                0.5,
                "(No metrics)",
                fontsize=8,
                fontweight="normal",
                ha="center",
                va="center",
                color="gray",
                style="italic",
            )
        ax_metrics.axis("off")

    fig.suptitle(
        f"Scene {scene_id} ({sensor}) - Full Image Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    _finalize_figure(
        fig,
        save_path=scene_dir / "scene_comparison.png",
        show_plot=False,
        dpi=DEFAULT_DPI,
        bbox_inches="tight",
    )


# ============================================================================
# COMPARISON PLOTS (4-Panel Comparisons)
# ============================================================================


class ComparisonPlotter:
    """
    Creates 4-panel comparison visualizations for paper figures.

    This class handles the creation of publication-quality comparison figures
    that demonstrate the superiority of physics-aware approaches over baselines.
    """

    def __init__(self, dpi: int = 300, figsize: Tuple[float, float] = (12, 10)):
        """
        Initialize comparison plotter.

        Args:
            dpi: Resolution for saved figures
            figsize: Figure size in inches
        """
        self.dpi = dpi
        self.figsize = figsize

        # Set matplotlib parameters for publication quality
        plt.rcParams.update(
            {
                "font.size": 10,
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
                "axes.labelsize": 10,
                "axes.titlesize": 11,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.dpi": dpi,
                "savefig.dpi": dpi,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.1,
            }
        )

        # Define color maps for different sensors
        self.colormaps = {
            "sony": "viridis",
            "fuji": "viridis",
        }

        # Define intensity ranges for different sensors (in electrons)
        self.intensity_ranges = {
            "sony": (0, 4000),
            "fuji": (0, 4000),
        }

    def create_4panel_comparison(
        self,
        our_result: torch.Tensor,
        l2_result: torch.Tensor,
        our_residuals: torch.Tensor,
        l2_residuals: torch.Tensor,
        sensor: str,
        our_metrics: Optional[Dict[str, float]] = None,
        l2_metrics: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        show_annotations: bool = True,
    ) -> plt.Figure:
        """
        Create 4-panel comparison figure.

        Args:
            our_result: Our method's restoration result (normalized [0,1])
            l2_result: L2 baseline restoration result (normalized [0,1])
            our_residuals: Our method's residuals (electrons)
            l2_residuals: L2 baseline's residuals (electrons)
            sensor: Sensor name ('sony' or 'fuji')
            our_metrics: Dictionary of metrics for our method
            l2_metrics: Dictionary of metrics for L2 method
            save_path: Path to save figure (optional)
            title: Figure title (optional)
            show_annotations: Whether to show statistical annotations

        Returns:
            matplotlib Figure object
        """
        # Convert to numpy for plotting
        from core.utils.tensor_utils import tensor_to_numpy

        our_result_np = tensor_to_numpy(our_result)
        l2_result_np = tensor_to_numpy(l2_result)
        our_residuals_np = tensor_to_numpy(our_residuals)
        l2_residuals_np = tensor_to_numpy(l2_residuals)

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # Get intensity range for this sensor
        vmin, vmax = self.intensity_ranges.get(sensor, (0, 1))

        # Get colormap for this sensor
        cmap = self.colormaps.get(sensor, "viridis")

        # Plot 1: Our Result
        self._plot_image_panel(
            axes[0, 0],
            our_result_np,
            "Our Result (DAPGD)",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        # Plot 2: L2 Result
        self._plot_image_panel(
            axes[0, 1], l2_result_np, "L2 Baseline", vmin=vmin, vmax=vmax, cmap=cmap
        )

        # Plot 3: Our Residuals
        self._plot_residuals_panel(
            axes[1, 0],
            our_residuals_np,
            "Our Residuals",
            sensor=sensor,
            metrics=our_metrics,
        )

        # Plot 4: L2 Residuals
        self._plot_residuals_panel(
            axes[1, 1],
            l2_residuals_np,
            "L2 Residuals",
            sensor=sensor,
            metrics=l2_metrics,
        )

        # Add main title if provided
        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        # Add statistical annotations if requested
        if show_annotations and our_metrics and l2_metrics:
            self._add_statistical_annotations(axes, our_metrics, l2_metrics, sensor)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            logger.info(f"Figure saved to: {save_path}")

        return fig

    def _plot_image_panel(
        self,
        ax: plt.Axes,
        image: np.ndarray,
        title: str,
        vmin: float = 0,
        vmax: float = 1,
        cmap: str = "viridis",
    ):
        """Plot a single image panel."""
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.axis("off")
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        return im

    def _plot_residuals_panel(
        self,
        ax: plt.Axes,
        residuals: np.ndarray,
        title: str,
        sensor: str,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Plot residuals panel with histogram."""
        # Main residual image
        im = ax.imshow(
            residuals,
            cmap="RdBu_r",
            vmin=-np.std(residuals) * 3,
            vmax=np.std(residuals) * 3,
            aspect="equal",
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.axis("off")
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Residuals (electrons)", fontsize=9)
        # Add histogram as inset
        self._add_residual_histogram(ax, residuals, metrics)

    def _add_residual_histogram(
        self,
        ax: plt.Axes,
        residuals: np.ndarray,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Add histogram of residuals as inset plot."""
        # Create inset axes for histogram
        ax_inset = ax.inset_axes([0.65, 0.05, 0.3, 0.25])
        # Flatten residuals and remove outliers for histogram
        flat_residuals = residuals.flatten()
        mean_res = np.mean(flat_residuals)
        std_res = np.std(flat_residuals)
        # Remove outliers beyond 3 sigma
        mask = np.abs(flat_residuals - mean_res) < 3 * std_res
        clean_residuals = flat_residuals[mask]
        if len(clean_residuals) > 0:
            # Create histogram
            ax_inset.hist(
                clean_residuals,
                bins=30,
                density=True,
                alpha=0.7,
                color="blue",
                edgecolor="black",
                linewidth=0.5,
            )
            # Add Gaussian fit
            x_gauss = np.linspace(mean_res - 3 * std_res, mean_res + 3 * std_res, 100)
            y_gauss = (1 / (std_res * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x_gauss - mean_res) / std_res) ** 2
            )
            ax_inset.plot(
                x_gauss, y_gauss, "r-", linewidth=1.5, alpha=0.8, label="Gaussian"
            )
            # Add statistical annotations
            if metrics:
                chi2 = metrics.get("chi2_consistency", 0)
                residual_std = metrics.get("residual_std", std_res)
                ax_inset.text(
                    0.05,
                    0.95,
                    f"χ² = {chi2:.2f}\nσ = {residual_std:.1f}",
                    transform=ax_inset.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            ax_inset.legend(fontsize=7)
            ax_inset.grid(True, alpha=0.3)
        # Remove ticks
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

    def _add_statistical_annotations(
        self,
        axes: np.ndarray,
        our_metrics: Dict[str, float],
        l2_metrics: Dict[str, float],
        sensor: str,
    ):
        """Add statistical comparison annotations."""
        # Extract key metrics
        our_chi2 = our_metrics.get("chi2_consistency", 0)
        l2_chi2 = l2_metrics.get("chi2_consistency", 0)
        our_psnr = our_metrics.get("psnr", 0)
        l2_psnr = l2_metrics.get("psnr", 0)
        # Add annotations to result panels
        self._add_metric_annotation(axes[0, 0], our_psnr, "PSNR")
        self._add_metric_annotation(axes[0, 1], l2_psnr, "PSNR")
        # Add method comparison
        self._add_method_comparison(axes, our_chi2, l2_chi2, sensor)

    def _add_metric_annotation(self, ax: plt.Axes, value: float, label: str):
        """Add metric annotation to panel."""
        ax.text(
            0.05,
            0.95,
            f"{label}: {value:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            verticalalignment="top",
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7, pad=0.3),
        )

    def _add_method_comparison(
        self, axes: np.ndarray, our_chi2: float, l2_chi2: float, sensor: str
    ):
        """Add comparison between methods."""
        # Create annotation text
        if our_chi2 < l2_chi2:
            comparison_text = (
                f"✓ DAPGD χ² = {our_chi2:.2f} better than L2 χ² = {l2_chi2:.2f}"
            )
        else:
            comparison_text = (
                f"L2 χ² = {l2_chi2:.2f} better than DAPGD χ² = {our_chi2:.2f}"
            )
        # Add to figure
        axes[0, 0].figure.text(
            0.02,
            0.02,
            comparison_text,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
            verticalalignment="bottom",
        )

    def create_sensor_comparison_grid(
        self,
        results_dict: Dict[str, Dict[str, torch.Tensor]],
        save_dir: str,
        sensor_order: List[str] = None,
    ):
        """
        Create comparison grid for sensor-specific results.

        Args:
            results_dict: Nested dict with sensor -> method -> tensors (e.g., 'sony' or 'fuji')
            save_dir: Directory to save figures
            sensor_order: Order of sensors for plotting (defaults to ['sony', 'fuji'])
        """
        if sensor_order is None:
            sensor_order = ["sony", "fuji"]
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for sensor in sensor_order:
            if sensor not in results_dict:
                logger.warning(f"Sensor {sensor} not found in results")
                continue
            sensor_data = results_dict[sensor]
            # Extract tensors for this sensor
            our_result = sensor_data.get("dapgd_result")
            l2_result = sensor_data.get("l2_result")
            our_residuals = sensor_data.get("dapgd_residuals")
            l2_residuals = sensor_data.get("l2_residuals")
            if None in [our_result, l2_result, our_residuals, l2_residuals]:
                logger.warning(f"Missing data for sensor {sensor}")
                continue
            # Get metrics
            our_metrics = sensor_data.get("dapgd_metrics", {})
            l2_metrics = sensor_data.get("l2_metrics", {})
            # Create figure
            fig = self.create_4panel_comparison(
                our_result=our_result,
                l2_result=l2_result,
                our_residuals=our_residuals,
                l2_residuals=l2_residuals,
                sensor=sensor,
                our_metrics=our_metrics,
                l2_metrics=l2_metrics,
                title=f"Comparison: {sensor.title()}",
                save_path=str(save_path / f"comparison_{sensor}.png"),
            )
            plt.close(fig)  # Close to free memory
        logger.info(
            f"Created comparison figures for {len(sensor_order)} sensors in {save_dir}"
        )


def create_paper_figure(
    our_result: torch.Tensor,
    l2_result: torch.Tensor,
    our_residuals: torch.Tensor,
    l2_residuals: torch.Tensor,
    sensor: str,
    our_metrics: Optional[Dict[str, float]] = None,
    l2_metrics: Optional[Dict[str, float]] = None,
    save_path: str = "comparison_figure.png",
) -> str:
    """
    Convenience function to create a single comparison figure.

    Args:
        our_result: Our method's restoration result
        l2_result: L2 baseline restoration result
        our_residuals: Our method's residuals
        l2_residuals: L2 baseline's residuals
        sensor: Sensor name
        our_metrics: Metrics for our method
        l2_metrics: Metrics for L2 method
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plotter = ComparisonPlotter()
    fig = plotter.create_4panel_comparison(
        our_result=our_result,
        l2_result=l2_result,
        our_residuals=our_residuals,
        l2_residuals=l2_residuals,
        sensor=sensor,
        our_metrics=our_metrics,
        l2_metrics=l2_metrics,
        save_path=save_path,
    )
    plt.close(fig)  # Close to free memory
    return save_path


# ============================================================================
# RESIDUAL PLOTS (Residual Analysis and Validation)
# ============================================================================


class ResidualPlotter:
    """Create publication-quality residual analysis plots."""

    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 150):
        """
        Initialize residual plotter.

        Args:
            figsize: Figure size (width, height) in inches
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style_params = {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }

    def create_4panel_comparison(
        self,
        pred_our: torch.Tensor,
        pred_baseline: torch.Tensor,
        noisy: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        background: float = 0.0,
        read_noise: float = 0.0,
        mask: Optional[torch.Tensor] = None,
        method_name: str = "DAPGD",
        baseline_name: str = "L2-Baseline",
        sensor: str = "unknown",
        output_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Create 4-panel comparison plot for publication.

        Args:
            pred_our: Our method prediction [B, C, H, W] (normalized)
            pred_baseline: Baseline prediction [B, C, H, W] (normalized)
            noisy: Noisy observation [B, C, H, W] (electrons)
            target: Ground truth [B, C, H, W] (normalized, optional)
            scale: Normalization scale
            background: Background offset
            read_noise: Read noise
            mask: Valid pixel mask
            method_name: Name of our method
            baseline_name: Name of baseline method
            sensor: Sensor name
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Set style
        with plt.style.context(["default"] + list(self.style_params.items())):
            fig, axes = plt.subplots(2, 4, figsize=self.figsize)
            # Select first image from batch
            our_pred = pred_our[0, 0].detach().cpu().numpy()
            baseline_pred = pred_baseline[0, 0].detach().cpu().numpy()
            noisy_img = noisy[0, 0].detach().cpu().numpy()
            if target is not None:
                target_img = target[0, 0].detach().cpu().numpy()
            else:
                target_img = None
            # Compute residuals
            our_pred_electrons = our_pred * scale + background
            baseline_pred_electrons = baseline_pred * scale + background
            our_residuals = noisy_img - our_pred_electrons
            baseline_residuals = noisy_img - baseline_pred_electrons
            expected_var_our = our_pred_electrons + read_noise**2
            expected_var_baseline = baseline_pred_electrons + read_noise**2
            our_norm_residuals = our_residuals / np.sqrt(expected_var_our)
            baseline_norm_residuals = baseline_residuals / np.sqrt(
                expected_var_baseline
            )
            if mask is not None:
                mask_img = mask[0, 0].detach().cpu().numpy()
                our_norm_residuals = our_norm_residuals * mask_img
                baseline_norm_residuals = baseline_norm_residuals * mask_img
            # Panel 1: Our method result
            im1 = axes[0, 0].imshow(our_pred, cmap="gray", vmin=0, vmax=1)
            axes[0, 0].set_title(f"{method_name} Result", fontweight="bold")
            axes[0, 0].axis("off")
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
            # Panel 2: Baseline result
            im2 = axes[0, 1].imshow(baseline_pred, cmap="gray", vmin=0, vmax=1)
            axes[0, 1].set_title(f"{baseline_name} Result", fontweight="bold")
            axes[0, 1].axis("off")
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
            # Panel 3: Our method residuals
            im3 = axes[0, 2].imshow(our_norm_residuals, cmap="RdBu_r", vmin=-3, vmax=3)
            axes[0, 2].set_title("Our Residuals (Normalized)", fontweight="bold")
            axes[0, 2].axis("off")
            plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
            # Panel 4: Baseline residuals
            im4 = axes[0, 3].imshow(
                baseline_norm_residuals, cmap="RdBu_r", vmin=-3, vmax=3
            )
            axes[0, 3].set_title("Baseline Residuals (Normalized)", fontweight="bold")
            axes[0, 3].axis("off")
            plt.colorbar(im4, ax=axes[0, 3], fraction=0.046, pad=0.04)
            # Bottom row: Statistics and analysis
            # Panel 5: Our residual histogram
            axes[1, 0].hist(
                our_norm_residuals.flatten(),
                bins=50,
                density=True,
                alpha=0.7,
                color="blue",
                label="Our residuals",
            )
            # Overlay normal distribution
            x = np.linspace(-4, 4, 100)
            y = stats.norm.pdf(x, 0, 1)
            axes[1, 0].plot(x, y, "r-", linewidth=2, label="N(0,1)")
            axes[1, 0].set_xlabel("Normalized Residuals")
            axes[1, 0].set_ylabel("Density")
            axes[1, 0].set_title("Our Residual Distribution")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            # Panel 6: Baseline residual histogram
            axes[1, 1].hist(
                baseline_norm_residuals.flatten(),
                bins=50,
                density=True,
                alpha=0.7,
                color="orange",
                label="Baseline residuals",
            )
            axes[1, 1].plot(x, y, "r-", linewidth=2, label="N(0,1)")
            axes[1, 1].set_xlabel("Normalized Residuals")
            axes[1, 1].set_ylabel("Density")
            axes[1, 1].set_title("Baseline Residual Distribution")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            # Panel 7: Q-Q plot for our method
            valid_residuals_our = our_norm_residuals.flatten()
            valid_residuals_our = valid_residuals_our[np.isfinite(valid_residuals_our)]
            if len(valid_residuals_our) > 100:
                # Simple Q-Q plot
                quantiles = np.linspace(0.01, 0.99, 100)
                theoretical_quantiles = stats.norm.ppf(quantiles)
                sample_quantiles = np.quantile(valid_residuals_our, quantiles)
                axes[1, 2].scatter(
                    theoretical_quantiles, sample_quantiles, alpha=0.6, s=20
                )
                # Reference line
                min_val = min(np.min(theoretical_quantiles), np.min(sample_quantiles))
                max_val = max(np.max(theoretical_quantiles), np.max(sample_quantiles))
                axes[1, 2].plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "r--",
                    alpha=0.7,
                    label="Reference line",
                )
                axes[1, 2].set_xlabel("Theoretical Quantiles")
                axes[1, 2].set_ylabel("Sample Quantiles")
                axes[1, 2].set_title("Q-Q Plot (Our Method)")
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            # Panel 8: Statistics summary
            axes[1, 3].axis("off")
            # Compute statistics
            our_mean = np.mean(our_norm_residuals)
            our_std = np.std(our_norm_residuals)
            baseline_mean = np.mean(baseline_norm_residuals)
            baseline_std = np.std(baseline_norm_residuals)
            # KS test
            our_finite = our_norm_residuals[np.isfinite(our_norm_residuals)]
            baseline_finite = baseline_norm_residuals[
                np.isfinite(baseline_norm_residuals)
            ]
            if len(our_finite) > 10 and len(baseline_finite) > 10:
                ks_stat_our, ks_p_our = stats.kstest(our_finite, "norm")
                ks_stat_baseline, ks_p_baseline = stats.kstest(baseline_finite, "norm")
            else:
                ks_stat_our, ks_p_our = float("nan"), float("nan")
                ks_stat_baseline, ks_p_baseline = float("nan"), float("nan")
            # Create statistics text
            stats_text = f"""
            {method_name} vs {baseline_name} Comparison

            Our Method Statistics:
            • Mean: {our_mean:.4f}
            • Std: {our_std:.4f}
            • KS vs N(0,1): {ks_stat_our:.4f} (p={ks_p_our:.4f})

            Baseline Statistics:
            • Mean: {baseline_mean:.4f}
            • Std: {baseline_std:.4f}
            • KS vs N(0,1): {ks_stat_baseline:.4f} (p={ks_p_baseline:.4f})

            Sensor: {sensor}
            Scale: {scale}
            Read Noise: {read_noise}
            """
            axes[1, 3].text(
                0.05,
                0.95,
                stats_text,
                transform=axes[1, 3].transAxes,
                fontsize=9,
                verticalalignment="top",
                fontfamily="monospace",
            )
            axes[1, 3].set_title("Statistical Summary")
            plt.tight_layout()
            # Save if path provided
            if output_path:
                output_path = Path(output_path)
                ensure_output_dir(output_path, create_parent=True)
                plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
                logger.info(f"4-panel comparison plot saved to {output_path}")
            return fig

    def create_residual_validation_plots(
        self,
        report: ResidualValidationReport,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Create comprehensive residual validation plots from a validation report.

        Args:
            report: Residual validation report
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        if ResidualValidationReport is None:
            raise ImportError("ResidualValidationReport is not available")
        with plt.style.context(["default"] + list(self.style_params.items())):
            fig = plt.figure(figsize=(16, 12))
            # Create subplot grid
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            # Panel 1: Statistical test results
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_statistical_tests(ax1, report)
            # Panel 2: Spatial correlation analysis
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_spatial_correlation(ax2, report)
            # Panel 3: Spectral analysis
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_spectral_analysis(ax3, report)
            # Panel 4: Distribution statistics
            ax4 = fig.add_subplot(gs[0, 3])
            self._plot_distribution_stats(ax4, report)
            # Panel 5: Overall assessment (large panel)
            ax5 = fig.add_subplot(gs[1:, :2])
            self._plot_overall_assessment(ax5, report)
            # Panel 6: Recommendations
            ax6 = fig.add_subplot(gs[1:, 2:])
            self._plot_recommendations(ax6, report)
            plt.suptitle(
                f"Residual Validation Report: {report.method_name}",
                fontsize=16,
                fontweight="bold",
            )
            # Save if path provided
            if output_path:
                output_path = Path(output_path)
                ensure_output_dir(output_path, create_parent=True)
                plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
                logger.info(f"Residual validation plots saved to {output_path}")
            return fig

    def _plot_statistical_tests(self, ax, report: ResidualValidationReport) -> None:
        """Plot statistical test results."""
        tests = ["KS Test", "Shapiro Test", "Anderson Test"]
        pvalues = [report.ks_pvalue, report.shapiro_pvalue, report.ks_pvalue]
        passed = [
            report.gaussian_fit,
            report.normal_by_shapiro,
            report.normal_by_anderson,
        ]
        colors = ["green" if p else "red" for p in passed]
        bars = ax.bar(tests, pvalues, color=colors, alpha=0.7)
        ax.axhline(y=0.05, color="black", linestyle="--", alpha=0.7, label="α = 0.05")
        ax.set_ylabel("p-value")
        ax.set_title("Normality Tests")
        ax.set_ylim(0, 1)
        ax.legend()
        for bar, pval in zip(bars, pvalues):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{pval:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _plot_spatial_correlation(self, ax, report: ResidualValidationReport) -> None:
        """Plot spatial correlation analysis."""
        if (
            hasattr(report, "autocorrelation_matrix")
            and report.autocorrelation_matrix is not None
        ):
            im = ax.imshow(
                report.autocorrelation_matrix, cmap="coolwarm", vmin=-1, vmax=1
            )
            ax.set_title("Spatial Autocorrelation")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Lag")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            # Fallback: text summary
            ax.text(
                0.5,
                0.5,
                f"X-autocorr: {report.autocorrelation_lag1_x:.3f}\n"
                f"Y-autocorr: {report.autocorrelation_lag1_y:.3f}\n"
                f"Uncorrelated: {report.spatial_uncorrelated}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
            )
            ax.set_title("Spatial Correlation")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

    def _plot_spectral_analysis(self, ax, report: ResidualValidationReport) -> None:
        """Plot spectral analysis results."""
        spectral_props = ["Spectral\nFlatness", "Spectral\nSlope", "High-freq\nPower"]
        spectral_values = [
            report.spectral_flatness,
            abs(report.spectral_slope),
            report.high_freq_power,
        ]
        colors = ["blue", "orange", "green"]
        bars = ax.bar(spectral_props, spectral_values, color=colors, alpha=0.7)
        ax.set_ylabel("Value")
        ax.set_title("Spectral Analysis")
        ax.axhline(
            y=0.8, color="red", linestyle="--", alpha=0.7, label="White noise threshold"
        )
        for bar, val in zip(bars, spectral_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.legend()

    def _plot_distribution_stats(self, ax, report: ResidualValidationReport) -> None:
        """Plot distribution statistics."""
        stats_names = ["Mean", "Std Dev", "Skewness", "Kurtosis"]
        stats_values = [report.mean, report.std_dev, report.skewness, report.kurtosis]
        bars = ax.bar(stats_names, stats_values, color="lightcoral", alpha=0.7)
        ax.set_ylabel("Value")
        ax.set_title("Distribution Statistics")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.7)
        for bar, val in zip(bars, stats_values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _plot_overall_assessment(self, ax, report: ResidualValidationReport) -> None:
        """Plot overall assessment results."""
        ax.text(
            0.5,
            0.8,
            f"Overall Physics Correctness",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
        # Color-coded result
        color = "green" if report.physics_correct else "red"
        status = "PASS ✓" if report.physics_correct else "FAIL ✗"
        ax.text(
            0.5,
            0.6,
            status,
            ha="center",
            va="center",
            fontsize=24,
            color=color,
            fontweight="bold",
        )
        # Summary text
        ax.text(
            0.5,
            0.3,
            report.validation_summary,
            ha="center",
            va="center",
            fontsize=12,
            wrap=True,
        )
        # Sample count
        ax.text(
            0.5,
            0.1,
            f"Based on {report.n_samples:,} samples",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def _plot_recommendations(self, ax, report: ResidualValidationReport) -> None:
        """Plot recommendations."""
        ax.text(
            0.5,
            0.95,
            "Recommendations",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        if report.recommendations:
            y_pos = 0.85
            for i, rec in enumerate(report.recommendations, 1):
                ax.text(
                    0.05,
                    y_pos,
                    f"{i}. {rec}",
                    ha="left",
                    va="top",
                    fontsize=10,
                    wrap=True,
                )
                y_pos -= 0.15
        else:
            ax.text(
                0.5,
                0.5,
                "No specific recommendations",
                ha="center",
                va="center",
                fontsize=12,
            )
        # Add metadata
        metadata_text = f"""
        Method: {report.method_name}
        Sensor: {report.sensor}
        Dataset: {report.dataset_name}
        Scale: {report.scale}
        Read Noise: {report.read_noise}
        Processing: {report.processing_time:.2f}s
        """
        ax.text(
            0.05,
            0.05,
            metadata_text,
            ha="left",
            va="bottom",
            fontsize=8,
            fontfamily="monospace",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")


def create_publication_plots(
    validation_reports: List[ResidualValidationReport],
    output_dir: Union[str, Path],
    method_names: Optional[List[str]] = None,
) -> None:
    """
    Create publication-ready plots comparing multiple methods.

    Args:
        validation_reports: List of validation reports
        output_dir: Directory to save plots
        method_names: Optional list of method names for legend
    """
    if ResidualValidationReport is None:
        raise ImportError("ResidualValidationReport is not available")

    output_dir = Path(output_dir)
    ensure_output_dir(output_dir, create_parent=False)

    plotter = ResidualPlotter()

    # Group reports by sensor
    sensor_reports = {}
    for report in validation_reports:
        sensor = report.sensor
        if sensor not in sensor_reports:
            sensor_reports[sensor] = []
        sensor_reports[sensor].append(report)

    # Create comparison plots for each sensor
    for sensor, reports in sensor_reports.items():
        if len(reports) < 2:
            continue

        logger.info(
            f"Creating publication plots for {sensor} sensor with {len(reports)} methods"
        )

        # Create statistical comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Extract method names
        methods = [r.method_name for r in reports]
        if method_names:
            methods = [
                method_names[i] if i < len(method_names) else m
                for i, m in enumerate(methods)
            ]

        # Plot 1: KS statistics
        ks_stats = [r.ks_statistic for r in reports]
        colors = plt.cm.tab10(np.linspace(0, 1, len(reports)))

        bars = axes[0, 0].bar(methods, ks_stats, color=colors)
        axes[0, 0].set_ylabel("KS Statistic")
        axes[0, 0].set_title(f"Normality Test (KS) - {sensor}")
        axes[0, 0].tick_params(axis="x", rotation=45)

        for bar, ks in zip(bars, ks_stats):
            height = bar.get_height()
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{ks:.3f}",
                ha="center",
                va="bottom",
            )

        # Plot 2: Physics correctness rate
        physics_correct = [r.physics_correct for r in reports]

        bars = axes[0, 1].bar(methods, physics_correct, color=colors)
        axes[0, 1].set_ylabel("Physics Correct Rate")
        axes[0, 1].set_title(f"Overall Physics Correctness - {sensor}")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis="x", rotation=45)

        for bar, correct in zip(bars, physics_correct):
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{correct:.2f}",
                ha="center",
                va="bottom",
            )

        # Plot 3: Spatial correlation
        spatial_uncorr = [r.spatial_uncorrelated for r in reports]

        bars = axes[1, 0].bar(methods, spatial_uncorr, color=colors)
        axes[1, 0].set_ylabel("Spatially Uncorrelated Rate")
        axes[1, 0].set_title(f"Spatial Correlation - {sensor}")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis="x", rotation=45)

        for bar, uncorr in zip(bars, spatial_uncorr):
            height = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{uncorr:.2f}",
                ha="center",
                va="bottom",
            )

        # Plot 4: Spectral flatness
        spectral_flatness = [r.spectral_flatness for r in reports]

        bars = axes[1, 1].bar(methods, spectral_flatness, color=colors)
        axes[1, 1].set_ylabel("Spectral Flatness")
        axes[1, 1].set_title(f"Spectral Analysis - {sensor}")
        axes[1, 1].axhline(y=0.8, color="red", linestyle="--", alpha=0.7)
        axes[1, 1].tick_params(axis="x", rotation=45)

        for bar, flatness in zip(bars, spectral_flatness):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{flatness:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            output_dir / f"residual_comparison_{sensor}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        logger.info(f"Publication plots saved for {sensor} sensor")


# ============================================================================
# SENSOR COMPARISON VISUALIZATIONS
# ============================================================================


def create_raw_data_comparison(
    samples: Dict[str, Dict[str, str]],
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False,
) -> None:
    """
    Create comparison visualization of raw data from sensors (Sony/Fuji).
    Shows noisy vs clean samples with pixel brightness ranges.

    Args:
        samples: Dictionary mapping sensor names to dicts with "noisy" and "clean" file paths
        output_path: Path to save figure (optional)
        show_plot: Whether to display the plot
    """
    load_functions = {
        "Sony": load_photography_raw,
        "Fuji": load_photography_raw,
    }

    # Create figure
    fig, axes = plt.subplots(len(samples), 2, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(
        "Raw Data Comparison: Noisy vs Clean Samples\nPixel Brightness Ranges",
        fontsize=16,
        fontweight="bold",
    )

    # Load and display each sample
    for i, (sensor, files) in enumerate(samples.items()):
        load_func = load_functions.get(sensor)
        if load_func is None:
            logger.warning(f"No load function for sensor: {sensor}")
            continue

        # Ensure consistent order: noisy first (j=0), clean second (j=1)
        ordered_items = [("noisy", files.get("noisy")), ("clean", files.get("clean"))]

        for j, (data_type, file_path) in enumerate(ordered_items):
            ax = axes[i, j]

            if file_path is None:
                ax.text(
                    0.5,
                    0.5,
                    "No file path",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{sensor} - {data_type.capitalize()}\nNo file")
                continue

            # Load data
            data, metadata = load_func(file_path)

            if data is not None:
                # Extract brightness range
                brightness_range = extract_brightness_range(data)

                # Handle RGB data (Sony/Fuji)
                if (
                    sensor in ["Sony", "Fuji"]
                    and len(data.shape) == 3
                    and data.shape[0] == 3
                ):
                    display_data = prepare_for_tile_visualization(data)
                    if display_data is not None:
                        # Simple normalization for RGB display (percentile clipping)
                        valid_mask = np.isfinite(display_data)
                        if np.any(valid_mask):
                            p_low, p_high = np.percentile(
                                display_data[valid_mask], (1, 99)
                            )
                            display_data = np.clip(display_data, p_low, p_high)
                            display_data = (display_data - p_low) / (
                                p_high - p_low + 1e-8
                            )
                        else:
                            display_data = None
                    else:
                        display_data = normalize_for_display(data)
                else:
                    display_data = normalize_for_display(data)

                if display_data is not None:
                    # Reshape for display if needed
                    if len(display_data.shape) == 1:
                        size = int(np.sqrt(len(display_data)))
                        if size * size == len(display_data):
                            display_data = display_data.reshape(size, size)
                        else:
                            next_size = size + 1
                            pad_size = next_size * next_size
                            padded = np.zeros(pad_size)
                            padded[: len(display_data)] = display_data
                            display_data = padded.reshape(next_size, next_size)

                    # Display image
                    if len(display_data.shape) == 3 and display_data.shape[2] == 3:
                        ax.imshow(display_data, aspect="equal")
                    else:
                        ax.imshow(display_data, cmap="gray", aspect="equal")
                    ax.set_title(
                        f"{sensor} - {data_type.capitalize()}\nBrightness: {brightness_range}",
                        fontsize=12,
                        fontweight="bold",
                    )
                    # Add brightness range as text overlay
                    _add_text_with_bbox(
                        ax,
                        0.02,
                        0.98,
                        brightness_range,
                        fontsize=10,
                        va="top",
                        ha="left",
                        facecolor="white",
                        alpha=0.8,
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Failed to process data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(
                        f"{sensor} - {data_type.capitalize()}\nFailed to process"
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Failed to load data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{sensor} - {data_type.capitalize()}\nFailed to load")

            ax.set_xticks([])
            ax.set_yticks([])

    # Add column labels
    fig.text(0.25, 0.95, "Noisy Samples", ha="center", fontsize=14, fontweight="bold")
    fig.text(0.75, 0.95, "Clean Samples", ha="center", fontsize=14, fontweight="bold")

    # Add row labels
    row_labels = ["Sony\n(ARW)", "Fuji\n(RAF)"]
    for i, label in enumerate(row_labels[: len(samples)]):
        fig.text(
            0.02,
            0.8 - i * 0.25,
            label,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            rotation=90,
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.1, right=0.95)

    if output_path:
        output_path = Path(output_path)
        ensure_output_dir(output_path, create_parent=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Comparison visualization saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ============================================================================
# SENSOR COMPARISON VISUALIZATIONS
# ============================================================================

# Constants for scaling
SCALING_METHODS = [
    "noisy",
    "clean",
    "restored_exposure_scaled",
    "restored_gaussian_x0",
    "restored_pg_x0",
]


def scale_and_save_image(
    image_path: Path, scale_factor: float, output_path: Path
) -> Optional[torch.Tensor]:
    """Scale image by multiplying with scale factor and save."""
    try:
        scaled_tensor = load_tensor_from_pt(image_path) * scale_factor
        save_tensor(scaled_tensor, output_path)
        logger.info(f"Scaled {image_path.name} by {scale_factor} -> {output_path.name}")
        return scaled_tensor
    except Exception as e:
        logger.error(f"Error scaling {image_path}: {e}")
        return None


def process_sensor_examples_for_scaling(
    sensor: str, examples: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Process examples for a sensor and return scaled data."""
    logger.info(f"\n=== Processing {sensor.upper()} ===")

    optimized_path = get_optimized_results_path(sensor)
    if not optimized_path:
        logger.warning(f"Unknown sensor: {sensor}")
        return {}

    scaled_data = {}
    for example_name in examples:
        example_path = optimized_path / example_name
        if not example_path.exists():
            logger.warning(f"Example not found: {example_path}")
            continue

        scaling_data = get_sensor_scaling_factor(sensor, example_path)
        if not scaling_data:
            logger.warning(f"Could not get scaling factor for {example_name}")
            continue

        scale_factor = scaling_data.get("scale_factor", 1.0)
        logger.info(f"Processing {example_name} with scale factor: {scale_factor}")

        scaled_dir = ensure_output_dir(example_path / "scaled", create_parent=True)
        method_ranges = {}

        for method in SCALING_METHODS:
            img_path = example_path / f"{method}.pt"
            if not img_path.exists():
                continue

            scaled_tensor = scale_and_save_image(
                img_path, scale_factor, scaled_dir / f"{method}_scaled.pt"
            )
            if scaled_tensor is not None:
                img_range = get_image_range(scaled_tensor)
                method_ranges[method] = img_range
                logger.info(
                    f"  {method}: [{img_range['min']:.3f}, {img_range['max']:.3f}]"
                )

        scaled_data[example_name] = {
            "scale_factor": scale_factor,
            "method_ranges": method_ranges,
        }

    return scaled_data


def scale_pt_files(examples: Optional[Dict[str, List[str]]] = None) -> None:
    """
    Scale .pt image files using sensor-specific preprocessing scale factors.

    Args:
        examples: Dictionary mapping sensor names to lists of example names.
                  If None, uses default examples.
    """
    if examples is None:
        examples = {
            "sony": [
                "example_00_sony_00135_00_0.1s_tile_0005",
                "example_02_sony_00135_00_0.1s_tile_0034",
            ],
            "fuji": [
                "example_00_fuji_00017_00_0.1s_tile_0009",
                "example_01_fuji_20184_00_0.033s_tile_0011",
                "example_02_fuji_00077_00_0.04s_tile_0022",
            ],
        }

    all_scaled_data = {
        sensor: process_sensor_examples_for_scaling(sensor, sensor_examples)
        for sensor, sensor_examples in examples.items()
    }

    output_path = (
        ensure_output_dir(Path("scaled_images_summary"), create_parent=False)
        / "scaled_ranges.json"
    )
    save_json_file(output_path, all_scaled_data)
    logger.info(
        f"\nScaling complete! Scaled images saved to individual 'scaled/' directories"
    )
    logger.info(f"Summary saved to {output_path}")
