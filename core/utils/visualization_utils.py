"""
Shared plotting utilities for visualization scripts.

This module contains common plotting functions used across multiple visualization scripts
to avoid code duplication.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Try importing optional dependencies
try:
    from core.utils.tensor_utils import get_pixel_stats
except ImportError:
    get_pixel_stats = None

try:
    from core.utils.sensor_utils import demosaic_raw_to_rgb
except ImportError:
    demosaic_raw_to_rgb = None

try:
    from core.utils.sensor_utils import (
        denormalize_to_physical,
        load_sensor_calibration_from_metadata,
    )
except ImportError:
    denormalize_to_physical = None
    load_sensor_calibration_from_metadata = None

try:
    from config.sample_config import (
        DEFAULT_DPI,
        DEFAULT_FIGSIZE_HEIGHT,
        DEFAULT_FIGSIZE_WIDTH,
        HIGH_DPI,
        SCENE_FIGSIZE_HEIGHT,
        SCENE_FIGSIZE_WIDTH,
    )
except ImportError:
    DEFAULT_DPI = 150
    DEFAULT_FIGSIZE_HEIGHT = 6
    DEFAULT_FIGSIZE_WIDTH = 8
    HIGH_DPI = 300
    SCENE_FIGSIZE_HEIGHT = 8
    SCENE_FIGSIZE_WIDTH = 12

logger = logging.getLogger(__name__)

# ============================================================================
# GENERAL UTILITIES
# ============================================================================


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


# ============================================================================
# PREPROCESSING VISUALIZATION UTILITIES
# ============================================================================


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
    sensor: str,
    exposure_type: str,
    label: str,
    raw_data: Any,
    stats: Tuple[float, float, float, float],
    precision: int = 3,
    tile_id: str = None,
) -> str:
    """Create standardized title for visualization subplot."""
    shape_str = _get_shape_str(raw_data)
    stats_str = _format_stats_str(*stats, precision=precision)
    tile_info = f"\nTile ID: {tile_id}" if tile_id and tile_id != "unknown" else ""
    return f"{sensor.capitalize()} - {exposure_type.capitalize()} Exposure {label}{tile_info}\nShape: {shape_str}\n{stats_str}"


def _create_tensor_title(
    sensor: str,
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
        f"{sensor.capitalize()} - {exposure_type.capitalize()} Exposure{tile_info}\n"
        f"Shape: {tensor.shape}, dtype: {tensor.dtype}\n"
        f"{dynamic_range} | Mean: {stats[2]:.4f} | STD: {stats[3]:.4f}\n"
        f"Target range: [-1, 1] (Display auto-scaled)"
    )


# ============================================================================
# SAMPLE VISUALIZATION UTILITIES
# ============================================================================

METHOD_PRIORITY = [
    ("gaussian_x0", "Gaussian (x0)"),
    ("gaussian", "Gaussian (x0)"),
    ("pg_x0", "PG (x0)"),
    ("pg", "PG (x0)"),
]

BASE_COLUMN_LABELS = {
    "short": "Short Exposure",
    "long": "Long Exposure Reference",
    "exposure_scaled": "Exposure Scaled",
}

ROW_LABELS = [
    ("Input Names\n& Ranges", "lightcoral"),
    ("Individual\nDynamic Range", "lightgreen"),
    ("PG x0\nScaled", "lightblue"),
    ("Metrics\nSSIM, PSNR, LPIPS, NIQE", "lightyellow"),
]

SCENE_METHOD_ORDER = [
    "short",
    "long",
    "exposure_scaled",
    "gaussian_x0",
    "pg_wls_x0",
    "pg_full_x0",
    "pg_simple_x0",
    "pg_x0",
]
REF_METHOD_CANDIDATES = ["pg_x0", "pg_wls_x0", "pg_full_x0", "pg_simple_x0", "pg"]


def _normalize_array_shape(array: np.ndarray, expected_channels: int = 3) -> np.ndarray:
    array = array[0] if array.ndim == 4 else array
    if array.ndim != 3:
        raise ValueError(
            f"Unexpected array shape: {array.shape} "
            f"(expected 3D (C, H, W) or 4D (B, C, H, W))"
        )
    if array.shape[0] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels, got {array.shape[0]} channels"
        )
    return array


def _transpose_to_hwc(array: np.ndarray) -> np.ndarray:
    return np.transpose(array, (1, 2, 0)) if array.shape[0] == 3 else array


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for plotting (uses core utility)."""
    from core.utils.tensor_utils import tensor_to_numpy as core_tensor_to_numpy

    return core_tensor_to_numpy(tensor, select_first=True)


def _tensor_to_display_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to display image (normalized to [0, 1] and uint8)."""
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.squeeze(0)
    img_np = img_tensor.cpu().numpy()
    if img_np.min() < 0:
        img_np = (img_np + 1) / 2
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    return _transpose_to_hwc(img_np)


def _to_display_image(phys_array: np.ndarray) -> np.ndarray:
    return _transpose_to_hwc(_normalize_array_shape(phys_array, expected_channels=3))


def _get_column_color(key: str) -> Optional[str]:
    color_map = {"short": "blue", "exposure_scaled": "purple"}
    if key in color_map:
        return color_map[key]
    if "pg" in key:
        return "green"
    if "gaussian" in key:
        return "orange"
    return None


def _format_optional_metric(value: float, metric_key: str, metric_label: str) -> str:
    if np.isnan(value):
        return f"{metric_label}: N/A"
    precision = 3 if metric_key == "lpips" else 1
    return f"{metric_label}: {value:.{precision}f}"


def _format_metrics_text(metrics: Dict[str, float], key: str) -> str:
    if key == "short":
        return f"PSNR: {metrics['psnr']:.1f}dB"

    metrics_lines = [
        f"SSIM: {metrics['ssim']:.3f}",
        f"PSNR: {metrics['psnr']:.1f}dB",
    ]

    for metric_key, metric_label in [("lpips", "LPIPS"), ("niqe", "NIQE")]:
        if metric_key in metrics:
            metrics_lines.append(
                _format_optional_metric(metrics[metric_key], metric_key, metric_label)
            )

    return "\n".join(metrics_lines)


def _add_text_with_bbox(
    ax: Any,
    x: float,
    y: float,
    text: str,
    fontsize: int = 9,
    fontweight: Optional[str] = "bold",
    ha: str = "center",
    va: str = "center",
    facecolor: Optional[str] = None,
    alpha: float = 0.7,
    color: Optional[str] = None,
    style: Optional[str] = None,
) -> None:
    text_kwargs = {
        "transform": ax.transAxes,
        "fontsize": fontsize,
        "ha": ha,
        "va": va,
    }
    if fontweight:
        text_kwargs["fontweight"] = fontweight
    if color:
        text_kwargs["color"] = color
    if style:
        text_kwargs["style"] = style
    if facecolor:
        text_kwargs["bbox"] = dict(boxstyle="round", facecolor=facecolor, alpha=alpha)
    ax.text(x, y, text, **text_kwargs)


def _add_row_label(ax: Any, label: str, facecolor: str) -> None:
    _add_text_with_bbox(
        ax,
        -0.08,
        0.5,
        label,
        fontsize=9,
        fontweight="bold",
        va="center",
        ha="right",
        facecolor=facecolor,
        alpha=0.7,
    )


def _finalize_figure(
    fig: Any,
    save_path: Optional[Path] = None,
    show_plot: bool = False,
    dpi: int = 150,
    bbox_inches: str = "tight",
) -> None:
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _select_available_methods(
    enhancement_results: Dict[str, torch.Tensor]
) -> Tuple[List[str], Dict[str, str]]:
    available_methods = []
    method_labels = {}
    seen_labels = set()
    for method_key, label in METHOD_PRIORITY:
        if method_key in enhancement_results and label not in seen_labels:
            available_methods.append(method_key)
            method_labels[method_key] = label
            seen_labels.add(label)
    return available_methods, method_labels


def _find_reference_method(
    images: Dict[str, np.ndarray], available_methods: List[str]
) -> str:
    for candidate in REF_METHOD_CANDIDATES:
        if candidate in images:
            return candidate
    return available_methods[0] if available_methods else "short"


def _process_image_for_display(
    tensor: torch.Tensor,
    black_level: float,
    white_level: float,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    if denormalize_to_physical is None:
        raise ImportError("denormalize_to_physical is not available")
    phys = denormalize_to_physical(tensor, black_level, white_level)
    img = _to_display_image(phys)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img, (float(img_min), float(img_max))


def _get_column_label(key: str, method_labels: Dict[str, str]) -> str:
    return BASE_COLUMN_LABELS.get(key, method_labels.get(key, key))


def _build_image_sources(
    short_image: torch.Tensor,
    long_image: Optional[torch.Tensor],
    enhancement_results: Dict[str, torch.Tensor],
    available_methods: List[str],
) -> List[Tuple[str, torch.Tensor]]:
    image_sources = [("short", short_image)]
    if long_image is not None:
        image_sources.append(("long", long_image))
    if "exposure_scaled" in enhancement_results:
        image_sources.append(
            ("exposure_scaled", enhancement_results["exposure_scaled"])
        )
    image_sources.extend(
        (method, enhancement_results[method])
        for method in available_methods
        if method in enhancement_results
    )
    return image_sources


def _get_metrics_text_and_style(
    key: str,
    metrics_results: Optional[Dict[str, Dict[str, float]]],
) -> Tuple[str, Optional[str], int, Optional[str], Optional[str]]:
    if key == "long":
        return "", None, 7, None, None

    if not metrics_results or key not in metrics_results:
        if key not in ("short", "long"):
            return "(No long\nreference)", "gray", 6, "italic", None
        return "", None, 7, None, None

    metrics_text = _format_metrics_text(metrics_results[key], key)
    color = "blue" if key == "short" else _get_column_color(key)
    return metrics_text, color, 7, None, "bold"


def _extract_brightness_value(brightness_data: Any) -> Optional[float]:
    if isinstance(brightness_data, dict):
        for key in ["p50", "median", "mean"]:
            if key in brightness_data:
                return float(brightness_data[key])
    if isinstance(brightness_data, (int, float)):
        return float(brightness_data)
    return None


def _extract_metrics_from_tile(
    tile: Dict[str, Any], method_key: str, metric_keys: List[str]
) -> Dict[str, List[float]]:
    result = {key: [] for key in metric_keys}
    method_metrics = tile.get("comprehensive_metrics", {}).get(method_key, {})
    for metric_key in metric_keys:
        if metric_key in method_metrics:
            result[metric_key].append(method_metrics[metric_key])
    return result


def _aggregate_metric_values(
    values: List[float], aggregation: str = "mean"
) -> Optional[float]:
    if not values:
        return None
    return float(np.median(values) if aggregation == "median" else np.mean(values))


def compute_scene_aggregate_metrics(
    scene_metrics: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if not scene_metrics:
        return {}

    metric_collections = {
        "pg": {"psnr": [], "ssim": []},
        "gaussian": {"psnr": [], "ssim": []},
    }
    all_brightness = []

    for tile in scene_metrics:
        brightness = _extract_brightness_value(tile.get("brightness_analysis"))
        if brightness is not None:
            all_brightness.append(brightness)

        for method_key, prefix in [("pg_x0", "pg"), ("gaussian_x0", "gaussian")]:
            metrics = _extract_metrics_from_tile(tile, method_key, ["psnr", "ssim"])
            metric_collections[prefix]["psnr"].extend(metrics["psnr"])
            metric_collections[prefix]["ssim"].extend(metrics["ssim"])

    aggregate: Dict[str, Any] = {}

    brightness_agg = _aggregate_metric_values(all_brightness, "median")
    if brightness_agg is not None:
        aggregate["tile_median_brightness"] = brightness_agg

    for prefix in ["pg", "gaussian"]:
        for metric in ["psnr", "ssim"]:
            agg_value = _aggregate_metric_values(
                metric_collections[prefix][metric], "mean"
            )
            if agg_value is not None:
                aggregate[f"{prefix}_{metric}"] = agg_value

    return aggregate


def _get_pixel_range(
    tensor: torch.Tensor,
    black_level: float,
    white_level: float,
) -> Tuple[float, float]:
    """Get min/max pixel values from tensor in physical space."""
    if denormalize_to_physical is None:
        raise ImportError("denormalize_to_physical is not available")
    phys = denormalize_to_physical(tensor, black_level, white_level)
    # denormalize_to_physical already returns a numpy array
    return float(phys.min()), float(phys.max())


def _aggregate_method_metrics(
    scene_metrics: List[Dict[str, Any]], method: str, metric_keys: List[str]
) -> Dict[str, float]:
    """Aggregate metrics for a specific method across all tiles."""
    metrics_list = []
    for tile_metric in scene_metrics:
        comp_metrics = tile_metric.get("comprehensive_metrics", {})
        if method in comp_metrics:
            metrics_list.append(comp_metrics[method])

    if not metrics_list:
        return {}

    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
        if values:
            aggregated[key] = float(np.mean(values))
    return aggregated


# ============================================================================
# SENSOR COMPARISON UTILITIES
# ============================================================================


def get_sensor_scaling_factor(
    sensor: str, tile_path: Union[str, Path]
) -> Optional[Dict[str, Any]]:
    """
    Get the sensor scaling factor from preprocessing (s parameter).

    Args:
        sensor: Sensor name
        tile_path: Path to tile directory containing results.json

    Returns:
        Dictionary with scale_factor, pixel_min, pixel_max, or None if not found
    """
    tile_path = Path(tile_path)
    results_file = tile_path / "results.json"
    if not results_file.exists():
        logger.warning(f"No results.json found for {tile_path}")
        return None

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        pg_params = data.get("pg_guidance_params", {})
        s = pg_params.get("s", 1.0)

        # Get brightness analysis for the actual pixel range
        brightness = data.get("brightness_analysis", {})
        pixel_min = brightness.get("min", 0)
        pixel_max = brightness.get("max", 1)

        return {"scale_factor": s, "pixel_min": pixel_min, "pixel_max": pixel_max}
    except Exception as e:
        logger.error(f"Error extracting scaling factor for {tile_path}: {e}")
        return None


def get_sensor_pixel_range(sensor: str) -> Dict[str, float]:
    """Get the typical pixel range for sensors (fallback)."""
    # sensor parameter is sensor name (sony, fuji)
    sensor_ranges = {
        "sony": {"min": 0.0, "max": 0.03},
        "fuji": {"min": 0.0, "max": 0.01},
    }
    return sensor_ranges.get(sensor, {"min": 0, "max": 1})


def normalize_image_to_sensor_range(
    image: np.ndarray, sensor_range: Dict[str, float]
) -> np.ndarray:
    """Normalize image to sensor-specific range."""
    if image is None or sensor_range is None:
        return image

    min_val, max_val = sensor_range["min"], sensor_range["max"]
    img_min, img_max = image.min(), image.max()

    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
        return normalized * (max_val - min_val) + min_val

    return image


def extract_brightness_range(data: np.ndarray) -> str:
    """Extract brightness range from image data using preprocessing utilities."""
    if data is None:
        return "N/A"
    if get_pixel_stats:
        stats = get_pixel_stats(data)
        min_val, max_val, mean_val, median_val = stats
        return f"[{min_val:.1f}, {max_val:.1f}] (mean: {mean_val:.1f})"
    return "N/A"


def load_photography_raw(
    file_path: Union[str, Path]
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """Load raw file (Sony ARW or Fuji RAF) and demosaic to RGB format using preprocessing utilities."""
    if demosaic_raw_to_rgb is None:
        logger.warning("demosaic_raw_to_rgb not available")
        return None, None
    try:
        rgb_image, metadata = demosaic_raw_to_rgb(file_path)
        if rgb_image is None:
            return None, None
        # rgb_image is already in (3, H, W) format, normalized to [0, 1]
        return rgb_image, metadata
    except Exception as e:
        logger.error(f"Error loading raw file {file_path}: {e}")
        return None, None


# ============================================================================
# COMMON UTILITIES (Path Handling, Output Directories, Sensor Mapping)
# ============================================================================


def setup_visualization_paths():
    """
    Setup common paths for visualization scripts.

    Adds visualization directory and preprocessing to sys.path.
    Should be called at the start of visualization scripts.
    """
    import sys
    from pathlib import Path

    viz_dir = Path(__file__).parent.parent.parent / "visualization"
    if str(viz_dir) not in sys.path:
        sys.path.insert(0, str(viz_dir))

    preprocessing_dir = viz_dir.parent / "preprocessing"
    if preprocessing_dir.exists() and str(preprocessing_dir) not in sys.path:
        sys.path.append(str(preprocessing_dir))


def ensure_output_dir(
    output_path: Union[str, Path], create_parent: bool = True
) -> Path:
    """
    Ensure output directory exists, creating parent directories if needed.

    Args:
        output_path: Path to output file or directory
        create_parent: If True, create parent directory; if False, create the path itself as directory

    Returns:
        Path object for the output directory
    """
    output_path = Path(output_path)
    if create_parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path.parent
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_results_base_path() -> Path:
    """Get the base results directory path."""
    return get_project_root() / "results"


def get_preprocessing_base_path() -> Path:
    """Get the base preprocessing directory path."""
    return get_project_root() / "preprocessing"


def get_sensor_to_optimized_dir() -> Dict[str, str]:
    """
    Get mapping from sensor name to optimized directory name.

    Returns:
        Dictionary mapping sensor names (e.g., 'sony', 'fuji') to directory names
    """
    return {
        "sony": "sony_optimized",
        "fuji": "fuji_optimized",
    }


def get_optimized_results_path(sensor: str) -> Optional[Path]:
    """
    Get the optimized results path for a sensor.

    Args:
        sensor: Sensor name ('sony' or 'fuji')

    Returns:
        Path to optimized results directory, or None if sensor not found
    """
    sensor_map = get_sensor_to_optimized_dir()
    if sensor not in sensor_map:
        return None

    return (
        get_results_base_path() / "optimized_inference_all_tiles" / sensor_map[sensor]
    )


def get_cross_sensor_results_path() -> Path:
    """Get the cross-sensor results base path."""
    return (
        get_results_base_path()
        / "cross_sensor_inference_all_tiles"
        / "sensors_cross_sensor"
    )


def get_sensor_display_name(sensor: str) -> str:
    """
    Get display name for a sensor.

    Args:
        sensor: Sensor name (e.g., 'sony', 'fuji')

    Returns:
        Display name (e.g., 'Sony', 'Fuji')
    """
    sensor_labels = {
        "sony": "Sony",
        "fuji": "Fuji",
    }
    return sensor_labels.get(sensor, sensor.replace("_", " ").title())


def resolve_output_path(
    user_path: Optional[Union[str, Path]],
    default_path: Union[str, Path],
    create_dir: bool = True,
) -> Path:
    """
    Resolve output path from user input or use default.

    Args:
        user_path: User-provided path (can be None)
        default_path: Default path to use if user_path is None
        create_dir: Whether to create the directory

    Returns:
        Resolved Path object
    """
    output_path = Path(user_path) if user_path else Path(default_path)
    if create_dir:
        ensure_output_dir(output_path, create_parent=True)
    return output_path


def setup_visualization_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup standardized logging for visualization scripts.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def add_common_visualization_arguments(parser) -> None:
    """
    Add common arguments to argument parser for visualization scripts.

    Args:
        parser: ArgumentParser instance (modified in place)
    """
    import argparse

    parser.add_argument(
        "--output", type=str, default=None, help="Output path for visualization/results"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively")


# ============================================================================
# EXISTING FUNCTIONS (keep these)
# ============================================================================


def load_results_json(results_file: Path) -> dict:
    """
    Load results from JSON file.

    This is a unified function that wraps core.utils._load_json_file
    for consistency across visualization scripts.

    Args:
        results_file: Path to JSON file

    Returns:
        Dictionary containing results data
    """
    from core.utils.metadata_utils import _load_json_file

    return _load_json_file(results_file)


def load_results_from_directory(results_dir: Path) -> Dict[str, Any]:
    """
    Load results from a results directory.

    Looks for:
    - results.json (main summary file)
    - stratified_results.json (if stratified evaluation was run)
    - Individual example_*/results.json files

    Args:
        results_dir: Directory containing results

    Returns:
        Dictionary with:
            - 'all_results': List of per-tile results
            - 'summary': Summary dict
            - 'stratified_results': Stratified evaluation results (if available)
    """
    import logging

    logger = logging.getLogger(__name__)

    results_data = {"all_results": [], "summary": {}, "stratified_results": None}

    # Try to load main results.json
    main_results_file = results_dir / "results.json"
    if main_results_file.exists():
        logger.info(f"Loading main results from {main_results_file}")
        data = load_results_json(main_results_file)

        if "results" in data:
            results_data["all_results"] = data["results"]
            results_data["summary"] = {k: v for k, v in data.items() if k != "results"}
        else:
            # Single result file
            results_data["all_results"] = [data]
            results_data["summary"] = data

    # Try to load stratified results
    stratified_file = results_dir / "stratified_results.json"
    if stratified_file.exists():
        logger.info(f"Loading stratified results from {stratified_file}")
        results_data["stratified_results"] = load_results_json(stratified_file)

    # Also look for individual example directories
    example_dirs = list(results_dir.glob("example_*"))
    if example_dirs and len(results_data["all_results"]) == 0:
        logger.info(
            f"Found {len(example_dirs)} example directories, loading individual results"
        )
        for example_dir in example_dirs:
            example_results = example_dir / "results.json"
            if example_results.exists():
                result = load_results_json(example_results)
                if result:
                    results_data["all_results"].append(result)

    logger.info(f"Loaded {len(results_data['all_results'])} result entries")
    return results_data


def normalize_metrics_for_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize metrics to [0,1] scale for composite score calculation.
    Higher is better for PSNR/SSIM, lower is better for LPIPS/NIQE.

    Args:
        df: DataFrame with metric columns

    Returns:
        DataFrame with normalized metric columns added (suffix '_norm')
    """
    # Normalize higher-is-better metrics
    for metric in ["psnr", "ssim"]:
        if metric in df.columns:
            col_min = df[metric].min()
            col_max = df[metric].max()
            if col_max > col_min:
                df[f"{metric}_norm"] = (df[metric] - col_min) / (col_max - col_min)
            else:
                df[f"{metric}_norm"] = 0.5

    # Normalize lower-is-better metrics (invert so higher is better)
    for metric in ["lpips", "niqe"]:
        if metric in df.columns:
            col_min = df[metric].min()
            col_max = df[metric].max()
            if col_max > col_min:
                df[f"{metric}_norm"] = 1 - (df[metric] - col_min) / (col_max - col_min)
            else:
                df[f"{metric}_norm"] = 0.5

    return df


def extract_metrics_from_results(
    results: List[Dict[str, Any]], method: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Extract metrics from results list.

    Args:
        results: List of result dictionaries
        method: Optional method name to filter by

    Returns:
        Dictionary mapping metric names to lists of values
    """
    metrics_dict: Dict[str, List[float]] = {}

    for result in results:
        # Check comprehensive_metrics first
        if "comprehensive_metrics" in result:
            methods_to_check = (
                [method] if method else result["comprehensive_metrics"].keys()
            )
            for m in methods_to_check:
                if m in result["comprehensive_metrics"]:
                    for metric_name, metric_value in result["comprehensive_metrics"][
                        m
                    ].items():
                        if metric_name not in metrics_dict:
                            metrics_dict[metric_name] = []
                        metrics_dict[metric_name].append(metric_value)
        # Fallback to top-level metrics
        elif "metrics" in result:
            if method is None or result.get("method") == method:
                for metric_name, metric_value in result["metrics"].items():
                    if metric_name not in metrics_dict:
                        metrics_dict[metric_name] = []
                    metrics_dict[metric_name].append(metric_value)

    return metrics_dict


def get_results_path_for_sensor(
    sensor: str, example_name: str, is_cross_sensor: bool = False
) -> Optional[Path]:
    """
    Get the results path for a given sensor and example.

    Args:
        sensor: Sensor name (e.g., "sony", "fuji")
        example_name: Name of the example directory
        is_cross_sensor: Whether to use cross-sensor results directory

    Returns:
        Path to results directory or None if invalid sensor
    """
    if is_cross_sensor:
        if sensor in ["sony", "fuji"]:
            return (
                Path(
                    "/home/jilab/Jae/results/cross_sensor_inference_all_tiles/sensors_cross_sensor"
                )
                / example_name
            )
        return None
    else:
        sensor_to_dir = {
            "sony": "sony_optimized",
            "fuji": "fuji_optimized",
        }
        if sensor in sensor_to_dir:
            return (
                Path("/home/jilab/Jae/results/optimized_inference_all_tiles")
                / sensor_to_dir[sensor]
                / example_name
            )
        return None


def load_and_prepare_image_for_display(
    image_path: Path, black_level: float, white_level: float
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Load and prepare image for display using core utilities.

    Args:
        image_path: Path to .pt image file
        black_level: Black level for denormalization
        white_level: White level for denormalization

    Returns:
        Tuple of (display_image, is_rgb_flag)
    """
    try:
        from core.normalization import denormalize_to_physical
        from core.utils.file_utils import load_tensor_from_pt

        img = load_tensor_from_pt(image_path)
        phys = denormalize_to_physical(
            img, black_level=black_level, white_level=white_level
        )
        display = normalize_for_display(phys)
        is_rgb = (
            display.ndim == 3 and display.shape[0] == 3
            if display is not None
            else False
        )
        return display, is_rgb
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None, False


def create_heatmap(
    results: dict,
    metric: str = "psnr",
    x_key: str = "kappa",
    y_key: str = "num_steps",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    results_key: str = "results",
) -> plt.Figure:
    """
    Create heatmap visualization of results.

    Generic function that can create heatmaps for any 2D parameter sweep.

    Args:
        results: Dictionary containing results data
        metric: Metric name to visualize (e.g., "psnr", "ssim")
        x_key: Key in result dict for x-axis values
        y_key: Key in result dict for y-axis values
        x_label: Label for x-axis (defaults to x_key)
        y_label: Label for y-axis (defaults to y_key)
        title: Title for the plot (defaults to metric name)
        results_key: Key in results dict containing the results list

    Returns:
        matplotlib Figure object
    """
    data = results.get(results_key, results.get("results", []))

    if not data:
        raise ValueError(f"No data found in results dict under key '{results_key}'")

    # Extract unique values
    x_values = sorted(set(r[x_key] for r in data if x_key in r))
    y_values = sorted(set(r[y_key] for r in data if y_key in r))

    if not x_values or not y_values:
        raise ValueError(f"Could not extract {x_key} or {y_key} values from data")

    # Create matrix
    matrix = np.zeros((len(y_values), len(x_values)))

    for result in data:
        if x_key not in result or y_key not in result:
            continue
        if metric not in result.get("metrics", {}):
            continue

        x_idx = x_values.index(result[x_key])
        y_idx = y_values.index(result[y_key])
        matrix[y_idx, x_idx] = result["metrics"][metric].get("mean", 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))

    # Format labels
    if isinstance(x_values[0], float):
        ax.set_xticklabels([f"{x:.2f}" for x in x_values])
    else:
        ax.set_xticklabels([str(x) for x in x_values])

    if isinstance(y_values[0], float):
        ax.set_yticklabels([f"{y:.2f}" for y in y_values])
    else:
        ax.set_yticklabels([str(y) for y in y_values])

    # Labels
    ax.set_xlabel(x_label or x_key.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_label or y_key.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        title or f"Hyperparameter Tuning: {metric.upper()} (higher is better)",
        fontsize=14,
    )

    # Add text annotations
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            text = ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] < matrix.mean() else "black",
            )

    # Colorbar
    plt.colorbar(im, ax=ax, label=metric.upper())
    plt.tight_layout()

    return fig


def create_line_plots(
    results: dict,
    metrics: List[str] = None,
    x_key: str = "kappa",
    y_key: str = "num_steps",
    results_key: str = "results",
) -> plt.Figure:
    """
    Create line plots showing metrics vs hyperparameters.

    Generic function that creates line plots for multiple metrics across
    different parameter values.

    Args:
        results: Dictionary containing results data
        metrics: List of metric names to plot (defaults to ["psnr", "ssim"])
        x_key: Key in result dict for x-axis values
        y_key: Key in result dict for y-axis values (used for grouping)
        results_key: Key in results dict containing the results list

    Returns:
        matplotlib Figure object
    """
    if metrics is None:
        metrics = ["psnr", "ssim"]

    data = results.get(results_key, results.get("results", []))

    if not data:
        raise ValueError(f"No data found in results dict under key '{results_key}'")

    # Group by x and y values
    x_values = sorted(set(r[x_key] for r in data if x_key in r))
    y_values = sorted(set(r[y_key] for r in data if y_key in r))

    if not x_values or not y_values:
        raise ValueError(f"Could not extract {x_key} or {y_key} values from data")

    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, n_metrics, figsize=(7 * n_metrics, 10))
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)

    for metric_idx, metric in enumerate(metrics):
        # Plot metric vs x_key (for each y_key value)
        ax = axes[0, metric_idx]
        for y_val in y_values:
            metric_values = []
            for x_val in x_values:
                result = next(
                    (
                        r
                        for r in data
                        if r.get(x_key) == x_val and r.get(y_key) == y_val
                    ),
                    None,
                )
                if result and metric in result.get("metrics", {}):
                    metric_values.append(result["metrics"][metric].get("mean", np.nan))
                else:
                    metric_values.append(np.nan)
            ax.plot(x_values, metric_values, marker="o", label=f"{y_key}={y_val}")
        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel(f"{metric.upper()} ({'dB' if metric == 'psnr' else ''})")
        ax.set_title(f"{metric.upper()} vs {x_key.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot metric vs y_key (for each x_key value)
        ax = axes[1, metric_idx]
        for x_val in x_values:
            metric_values = []
            for y_val in y_values:
                result = next(
                    (
                        r
                        for r in data
                        if r.get(x_key) == x_val and r.get(y_key) == y_val
                    ),
                    None,
                )
                if result and metric in result.get("metrics", {}):
                    metric_values.append(result["metrics"][metric].get("mean", np.nan))
                else:
                    metric_values.append(np.nan)
            ax.plot(y_values, metric_values, marker="o", label=f"{x_key}={x_val}")
        ax.set_xlabel(y_key.replace("_", " ").title())
        ax.set_ylabel(f"{metric.upper()} ({'dB' if metric == 'psnr' else ''})")
        ax.set_title(f"{metric.upper()} vs {y_key.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_bar_chart(
    methods: List[str],
    metric_values: List[float],
    metric_stds: Optional[List[float]] = None,
    metric_name: str = "Metric",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    highlight_best: bool = True,
    higher_is_better: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Create a bar chart for comparing methods.

    Args:
        methods: List of method names
        metric_values: List of metric values (one per method)
        metric_stds: Optional list of standard deviations for error bars
        metric_name: Name of the metric being plotted
        title: Plot title
        ylabel: Y-axis label
        highlight_best: Whether to highlight the best method
        higher_is_better: Whether higher values are better (for highlighting)
        ylim: Optional tuple for y-axis limits

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get colors for methods (use local function)
    colors_map = get_method_colors()
    colors = [colors_map.get(m, "#95a5a6") for m in methods]

    # Format method names (use local function)
    display_names = [format_method_name(m) for m in methods]

    # Create bars
    if metric_stds:
        bars = ax.bar(
            display_names,
            metric_values,
            yerr=metric_stds,
            capsize=5,
            color=colors,
            alpha=0.8,
        )
    else:
        bars = ax.bar(display_names, metric_values, color=colors, alpha=0.8)

    # Highlight best
    if highlight_best:
        if higher_is_better:
            best_idx = np.argmax(metric_values)
        else:
            best_idx = np.argmin(metric_values)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

    # Labels
    ax.set_ylabel(ylabel or metric_name, fontsize=12, fontweight="bold")
    ax.set_title(
        title
        or f"{metric_name} Comparison ({'Higher' if higher_is_better else 'Lower'} is Better)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    if ylim:
        ax.set_ylim(ylim)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    return fig
