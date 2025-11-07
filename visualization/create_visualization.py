#!/usr/bin/env python3
"""
Unified visualization creation script for low-light imaging restoration comparisons.

This script combines functionality from:
- create_comparison_visualization_fixed.py: Sensor-specific comparisons with scaling
- create_comprehensive_comparison.py: Single-sensor vs cross-sensor model comparisons
- create_raw_comparison.py: Raw data comparisons
- create_raw_distributions.py: Raw pixel value distribution plots
- scale_pt_files.py: Scale .pt image files using sensor-specific preprocessing scale factors

Usage:
    python create_visualization.py --mode sensor_comparison
    python create_visualization.py --mode comprehensive
    python create_visualization.py --mode raw_data
    python create_visualization.py --mode distributions
    python create_visualization.py --mode scale_files
"""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from core.normalization import (
    denormalize_to_physical,
    reverse_normalize_from_neg_one_to_raw,
)
from core.utils.file_utils import (
    extract_metrics_from_json,
    format_pixel_range,
    load_image_tensor,
    load_tensor_from_pt,
)
from core.utils.metadata_utils import load_comprehensive_metadata
from core.utils.sensor_utils import load_sensor_calibration_from_metadata
from core.utils.visualization_utils import get_results_path_for_sensor
from visualization.visualizations import (
    add_common_visualization_arguments,
    create_raw_data_comparison,
    ensure_output_dir,
    get_cross_sensor_results_path,
    get_method_filename_map,
    get_optimized_results_path,
    get_preprocessing_base_path,
    get_project_root,
    get_results_base_path,
    get_sensor_display_name,
    load_results_json,
    normalize_for_display,
    prepare_for_tile_visualization,
    resolve_output_path,
    scale_pt_files,
    setup_visualization_logging,
)

warnings.filterwarnings("ignore")
logger = setup_visualization_logging()


def find_cross_sensor_example(sensor: str, single_sensor_example: str) -> Optional[str]:
    """Find the corresponding cross-sensor example for a single-sensor example."""
    parts = single_sensor_example.split("_")
    if len(parts) >= 4:
        sensor_idx = None
        for i, part in enumerate(parts):
            if part in ["sony", "fuji"]:
                sensor_idx = i
                break

        if sensor_idx is not None:
            base_identifier = "_".join(parts[sensor_idx + 1 :])
            cross_sensor_dir = get_cross_sensor_results_path()

            if cross_sensor_dir and cross_sensor_dir.exists():
                for item in cross_sensor_dir.iterdir():
                    if item.is_dir() and base_identifier in item.name:
                        return item.name
    return None


def get_image_for_method(
    sensor: str, example_dir: str, method: str
) -> Tuple[Optional[np.ndarray], bool]:
    """Get image tensor for a specific method from the appropriate source."""
    tile_path = Path(example_dir)
    method_files = get_method_filename_map()

    is_cross_sensor = "cross" in method
    example_name = tile_path.name

    if is_cross_sensor:
        cross_example = find_cross_sensor_example(sensor, example_name)
        if not cross_example:
            return None, False
        example_name = cross_example

    source_path = get_results_path_for_sensor(
        sensor, example_name, is_cross_sensor=is_cross_sensor
    )
    if source_path is None:
        return None, False

    if method in method_files:
        original_image_path = source_path / method_files[method]
        image, is_rgb = load_image_tensor(original_image_path)
        return image, is_rgb

    return None, False


def extract_metrics_for_method(
    sensor: str, example_dir: str, method: str
) -> Optional[Dict]:
    """Extract metrics for a specific method from the appropriate results file."""
    is_cross_sensor = "cross" in method
    example_name = Path(example_dir).name

    if is_cross_sensor:
        cross_example = find_cross_sensor_example(sensor, example_name)
        if not cross_example:
            return None
        example_name = cross_example

    source_path = get_results_path_for_sensor(
        sensor, example_name, is_cross_sensor=is_cross_sensor
    )
    if source_path is None:
        return None

    return extract_metrics_from_json(source_path / "results.json", method)


def _create_sensor_subplot(
    ax,
    sensor: str,
    example_dir: str,
    methods: List[str],
    metrics_dict: Optional[Dict[str, Dict]] = None,
    scale_mode: str = "pg_x0_cross",
) -> Tuple[List[Dict], List[np.ndarray], float, float]:
    """Create subplot for a single sensor."""
    tile_path = Path(example_dir)
    images = []
    pixel_ranges = []
    is_rgb_flags = []

    for method in methods:
        img, is_rgb = get_image_for_method(sensor, tile_path, method)
        if img is not None:
            images.append(img)
            is_rgb_flags.append(is_rgb)
            pixel_ranges.append({"min": float(img.min()), "max": float(img.max())})
        else:
            images.append(np.zeros((256, 256)))
            is_rgb_flags.append(False)
            pixel_ranges.append({"min": 0, "max": 0})

    if scale_mode == "clean":
        clean_idx = 1
        if len(images) > clean_idx and images[clean_idx] is not None:
            vmin = float(images[clean_idx].min())
            vmax = float(images[clean_idx].max())
        else:
            all_mins = [
                r["min"] for r in pixel_ranges if r["min"] != 0 or r["max"] != 0
            ]
            all_maxs = [
                r["max"] for r in pixel_ranges if r["min"] != 0 or r["max"] != 0
            ]
            vmin = min(all_mins) if all_mins else 0
            vmax = max(all_maxs) if all_maxs else 1
    else:
        pg_x0_cross_idx = len(methods) - 1
        if len(images) > pg_x0_cross_idx and images[pg_x0_cross_idx] is not None:
            vmin = float(images[pg_x0_cross_idx].min())
            vmax = float(images[pg_x0_cross_idx].max())
        else:
            all_mins = [
                r["min"] for r in pixel_ranges if r["min"] != 0 or r["max"] != 0
            ]
            all_maxs = [
                r["max"] for r in pixel_ranges if r["min"] != 0 or r["max"] != 0
            ]
            vmin = min(all_mins) if all_mins else 0
            vmax = max(all_maxs) if all_maxs else 1

    has_rgb = any(is_rgb_flags)
    n_methods = len(methods)

    if has_rgb:
        composite_img = np.zeros((256, 256 * n_methods, 3))
    else:
        composite_img = np.zeros((256, 256 * n_methods))

    for i, img in enumerate(images):
        if len(img.shape) == 3 and img.shape[0] == 3:
            img_display = prepare_for_tile_visualization(img)
            if img_display is None:
                img_display = np.transpose(img, (1, 2, 0))
            img_display = (img_display - vmin) / (vmax - vmin + 1e-8)
            img_display = np.clip(img_display, 0, 1)
            composite_img[:, i * 256 : (i + 1) * 256, :] = img_display
        elif len(img.shape) == 2:
            composite_img[:, i * 256 : (i + 1) * 256] = img
        else:
            composite_img[:, i * 256 : (i + 1) * 256] = (
                img if len(img.shape) == 2 else img.mean(axis=0)
            )

    if has_rgb:
        ax.imshow(composite_img)
    else:
        ax.imshow(composite_img, cmap="gray", vmin=vmin, vmax=vmax)

    ax.set_xlim(0, composite_img.shape[1])
    ax.set_ylim(-90, composite_img.shape[0])
    ax.axis("off")

    for i, (method, pixel_range) in enumerate(zip(methods, pixel_ranges)):
        range_text = format_pixel_range(pixel_range)

        if method == "noisy" and metrics_dict and method in metrics_dict:
            metrics = metrics_dict[method]
            if metrics:
                psnr_val = metrics.get("psnr", 0)
                noisy_text = f"{range_text}\nPSNR={psnr_val:.3f}"
                ax.text(
                    i * 256 + 128,
                    -30,
                    noisy_text,
                    ha="center",
                    va="top",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
                )
        elif method == "clean":
            ax.text(
                i * 256 + 128,
                -30,
                range_text,
                ha="center",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
            )
        elif metrics_dict and method in metrics_dict:
            metrics = metrics_dict[method]
            if metrics:
                psnr_val = metrics.get("psnr", 0)
                ssim_val = metrics.get("ssim", 0)
                lpips_val = metrics.get("lpips", 0)
                niqe_val = metrics.get("niqe", "N/A")

                niqe_str = (
                    f"{niqe_val:.3f}"
                    if isinstance(niqe_val, (int, float))
                    else str(niqe_val)
                )

                metric_text = (
                    f"{range_text}\nPSNR={psnr_val:.3f}\nSSIM={ssim_val:.3f}\n"
                    f"LPIPS={lpips_val:.3f}\nNIQE={niqe_str}"
                )
                ax.text(
                    i * 256 + 128,
                    -30,
                    metric_text,
                    ha="center",
                    va="top",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                )
        else:
            ax.text(
                i * 256 + 128,
                -30,
                range_text,
                ha="center",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
            )

    return pixel_ranges, images, vmin, vmax


def create_sensor_comparison(
    examples: Dict[str, List[str]],
    methods: List[str],
    method_names: List[str],
    scale_mode: str = "pg_x0_cross",
    output_prefix: str = "comparison_visualization",
) -> None:
    """Create sensor comparison visualization."""
    sensors = list(examples.keys())
    fig_height = len(sensors) * 3.5
    fig_width = len(methods) * 2

    fig, axes = plt.subplots(len(sensors), 1, figsize=(fig_width, fig_height))
    if len(sensors) == 1:
        axes = [axes]

    for i, sensor in enumerate(sensors):
        example_name = examples[sensor][0] if sensor in examples else None
        if not example_name:
            continue

        optimized_path = get_optimized_results_path(sensor)
        if optimized_path is None:
            continue
        example_path = str(optimized_path / example_name)

        metrics_dict = {}
        for method in methods:
            metrics = extract_metrics_for_method(sensor, example_path, method)
            if metrics:
                metrics_dict[method] = metrics

        pixel_ranges, images, vmin, vmax = _create_sensor_subplot(
            axes[i],
            sensor,
            example_path,
            methods,
            metrics_dict=metrics_dict,
            scale_mode=scale_mode,
        )

        sensor_display_name = get_sensor_display_name(sensor)
        axes[i].set_title(
            f"{sensor_display_name}", fontsize=12, fontweight="bold", y=1.02
        )

    title_suffix = "\n(Clean Reference Scaled)" if scale_mode == "clean" else ""
    fig.suptitle(
        f"Low-light imaging restoration comparison{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.925 if scale_mode != "clean" else 0.85, bottom=0.15, hspace=0.3
    )

    total_width = len(methods) * 256
    for idx, sensor in enumerate(sensors):
        ax = axes[idx]
        ax_pos = ax.get_position()
        ax_left = ax_pos.x0
        ax_width = ax_pos.width

        for i, method_name in enumerate(method_names):
            panel_center = i * 256 + 128
            x_position = ax_left + (panel_center / total_width) * ax_width
            if idx == 0:
                fig.text(
                    x_position,
                    ax_pos.y1 + 0.03,
                    method_name,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    suffix = "_clean_range" if scale_mode == "clean" else ""
    output_path = f"{output_prefix}_sensor_scaled{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(
        f"{output_prefix}_sensor_scaled{suffix}_high_res.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_cross_sensor_comparison(
    single_sensor_paths: Dict[str, Path],
    cross_sensor_paths: Dict[str, Path],
    output_path: Optional[Path] = None,
) -> None:
    """
    Create comprehensive comparison visualization across single-sensor and cross-sensor models.

    Args:
        single_sensor_paths: Dictionary mapping sensor names to single-sensor result paths
        cross_sensor_paths: Dictionary mapping sensor names to cross-sensor result paths
        output_path: Optional output path for the figure
    """
    single_sensor_results = {}
    cross_sensor_results = {}

    for sensor, path in single_sensor_paths.items():
        if path.exists():
            single_sensor_results[sensor] = load_results_json(path / "results.json")

    for sensor, path in cross_sensor_paths.items():
        if path.exists():
            cross_sensor_results[sensor] = load_results_json(path / "results.json")

    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 9, figure=fig, hspace=0.3, wspace=0.1)

    methods = ["noisy", "exposure_scaled", "gaussian_x0", "pg_x0"]
    method_labels = ["Noisy", "Exposure Scaled", "Gaussian x0", "PG x0"]

    method_colors = {
        "noisy": "gray",
        "exposure_scaled": "orange",
        "gaussian_x0": "blue",
        "pg_x0": "red",
    }

    sensors = list(single_sensor_paths.keys())
    sensor_labels = {sensor: get_sensor_display_name(sensor) for sensor in sensors}

    for i, sensor in enumerate(sensors):
        if sensor not in single_sensor_results or sensor not in cross_sensor_results:
            continue

        single_path = single_sensor_paths[sensor]
        cross_path = cross_sensor_paths[sensor]

        clean_img = load_tensor_from_pt(single_path / "clean.pt")
        black_level, white_level = load_sensor_calibration_from_metadata(sensor)
        clean_phys = denormalize_to_physical(
            clean_img, black_level=black_level, white_level=white_level
        )
        clean_display = normalize_for_display(clean_phys)

        ax_clean = fig.add_subplot(gs[i, 0])
        if clean_display.ndim == 3 and clean_display.shape[0] == 3:
            clean_rgb = prepare_for_tile_visualization(clean_display)
            if clean_rgb is not None:
                ax_clean.imshow(clean_rgb)
            else:
                ax_clean.imshow(clean_display.transpose(1, 2, 0))
        else:
            ax_clean.imshow(clean_display, cmap="gray")
        ax_clean.set_title(
            f"{sensor_labels.get(sensor, sensor)}\nClean Reference",
            fontsize=10,
            fontweight="bold",
        )
        ax_clean.axis("off")

        for j, (method, label) in enumerate(zip(methods, method_labels)):
            ax = fig.add_subplot(gs[i, j + 1])
            single_img = load_tensor_from_pt(single_path / f"restored_{method}.pt")
            single_phys = denormalize_to_physical(
                single_img, black_level=black_level, white_level=white_level
            )
            single_display = normalize_for_display(single_phys)

            if single_display.ndim == 3 and single_display.shape[0] == 3:
                single_rgb = prepare_for_tile_visualization(single_display)
                if single_rgb is not None:
                    ax.imshow(single_rgb)
                else:
                    ax.imshow(single_display.transpose(1, 2, 0))
            else:
                ax.imshow(single_display, cmap="gray")

            metrics = single_sensor_results[sensor]["comprehensive_metrics"][method]
            ssim = metrics.get("ssim", 0)
            psnr = metrics.get("psnr", 0)

            ax.set_title(
                f"{label}\nSingle-Sensor\nSSIM: {ssim:.3f}, PSNR: {psnr:.1f}dB",
                fontsize=9,
                color=method_colors[method],
            )
            ax.axis("off")

        for j, (method, label) in enumerate(zip(methods, method_labels)):
            ax = fig.add_subplot(gs[i, j + 3])
            cross_img = load_tensor_from_pt(cross_path / f"restored_{method}.pt")
            cross_phys = denormalize_to_physical(
                cross_img, black_level=black_level, white_level=white_level
            )
            cross_display = normalize_for_display(cross_phys)

            if cross_display.ndim == 3 and cross_display.shape[0] == 3:
                cross_rgb = prepare_for_tile_visualization(cross_display)
                if cross_rgb is not None:
                    ax.imshow(cross_rgb)
                else:
                    ax.imshow(cross_display.transpose(1, 2, 0))
            else:
                ax.imshow(cross_display, cmap="gray")

            metrics = cross_sensor_results[sensor]["comprehensive_metrics"][method]
            ssim = metrics.get("ssim", 0)
            psnr = metrics.get("psnr", 0)

            ax.set_title(
                f"{label}\nCross-Sensor\nSSIM: {ssim:.3f}, PSNR: {psnr:.1f}dB",
                fontsize=9,
                color=method_colors[method],
            )
            ax.axis("off")

    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis("off")

    summary_text = "SUMMARY: Cross-Sensor vs Single-Sensor Performance\n\n"
    for sensor in sensors:
        if sensor not in single_sensor_results or sensor not in cross_sensor_results:
            continue

        summary_text += f"{sensor.upper()}:\n"
        single_pg = single_sensor_results[sensor]["comprehensive_metrics"]["pg_x0"]
        cross_pg = cross_sensor_results[sensor]["comprehensive_metrics"]["pg_x0"]

        single_ssim = single_pg.get("ssim", 0)
        single_psnr = single_pg.get("psnr", 0)
        cross_ssim = cross_pg.get("ssim", 0)
        cross_psnr = cross_pg.get("psnr", 0)

        ssim_diff = cross_ssim - single_ssim
        psnr_diff = cross_psnr - single_psnr

        summary_text += (
            f"  PG x0 - Single: SSIM={single_ssim:.3f}, PSNR={single_psnr:.1f}dB\n"
        )
        summary_text += (
            f"  PG x0 - Cross:  SSIM={cross_ssim:.3f}, PSNR={cross_psnr:.1f}dB\n"
        )
        summary_text += (
            f"  Difference:     SSIM={ssim_diff:+.3f}, PSNR={psnr_diff:+.1f}dB\n\n"
        )

    ax_summary.text(
        0.5,
        0.5,
        summary_text,
        transform=ax_summary.transAxes,
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    fig.text(
        0.5,
        0.95,
        "Cross-Sensor Low-Light Enhancement: Single-Sensor vs Cross-Sensor Models",
        fontsize=16,
        fontweight="bold",
        ha="center",
    )

    fig.text(
        0.05, 0.88, "Clean\nReference", fontsize=12, fontweight="bold", ha="center"
    )
    fig.text(
        0.25, 0.88, "Single-Sensor Model", fontsize=12, fontweight="bold", ha="center"
    )
    fig.text(
        0.75, 0.88, "Cross-Sensor Model", fontsize=12, fontweight="bold", ha="center"
    )

    if output_path is None:
        output_path = get_results_base_path() / "comprehensive_sensor_comparison.png"
    else:
        output_path = Path(output_path)
    ensure_output_dir(output_path, create_parent=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def process_data_values_with_metadata(
    data_values_path: Path, tile_info_map: Dict
) -> Dict:
    """Process data_values JSON using tile_info_map for data_type."""
    with open(data_values_path, "r") as f:
        data = json.load(f)

    organized_data = defaultdict(
        lambda: defaultdict(lambda: {"min": [], "median": [], "mean": [], "max": []})
    )

    tiles = data.get("tiles", [])

    for tile in tiles:
        tile_id = tile.get("tile_id", "")
        if tile_id not in tile_info_map:
            continue

        info = tile_info_map[tile_id]
        sensor_type = info["sensor_type"]
        data_type = info["data_type"]

        min_raw = reverse_normalize_from_neg_one_to_raw(tile["min"], sensor_type)
        median_raw = reverse_normalize_from_neg_one_to_raw(tile["median"], sensor_type)
        mean_raw = reverse_normalize_from_neg_one_to_raw(tile["mean"], sensor_type)
        max_raw = reverse_normalize_from_neg_one_to_raw(tile["max"], sensor_type)

        organized_data[sensor_type][data_type]["min"].append(min_raw)
        organized_data[sensor_type][data_type]["median"].append(median_raw)
        organized_data[sensor_type][data_type]["mean"].append(mean_raw)
        organized_data[sensor_type][data_type]["max"].append(max_raw)

    return organized_data


def create_distribution_plots(data_dict: Dict, output_path: Path) -> None:
    """Create distribution plots with min/median/max combined and mean separately."""
    try:
        import seaborn as sns

        sns.set_style("whitegrid")
    except ImportError:
        pass

    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300

    sensors = sorted(data_dict.keys())
    n_sensors = len(sensors)
    fig, axes = plt.subplots(n_sensors, 2, figsize=(16, 6 * n_sensors))

    if n_sensors == 1:
        axes = axes.reshape(1, -1)

    for sensor_idx, sensor_type in enumerate(sensors):
        sensor_data = data_dict[sensor_type]

        ax_range = axes[sensor_idx, 0]
        short_min = np.array(sensor_data.get("short", {}).get("min", []))
        short_median = np.array(sensor_data.get("short", {}).get("median", []))
        short_max = np.array(sensor_data.get("short", {}).get("max", []))
        long_min = np.array(sensor_data.get("long", {}).get("min", []))
        long_median = np.array(sensor_data.get("long", {}).get("median", []))
        long_max = np.array(sensor_data.get("long", {}).get("max", []))

        if len(short_min) > 0 and len(long_min) > 0:
            short_combined = np.concatenate([short_min, short_median, short_max])
            long_combined = np.concatenate([long_min, long_median, long_max])
            all_data = np.concatenate([short_combined, long_combined])
            min_val = np.min(all_data)
            max_val = np.max(all_data)
            use_log = max_val / (min_val + 1) > 100

            if use_log:
                bins = np.logspace(np.log10(min_val + 1), np.log10(max_val + 1), 100)
                ax_range.set_xscale("log")
            else:
                bins = np.linspace(min_val, max_val, 100)

            ax_range.hist(
                long_combined,
                bins=bins,
                alpha=0.6,
                label="Long Exposure (Min+Median+Max)",
                color="orange",
                density=True,
                edgecolor="black",
                linewidth=0.5,
            )
            ax_range.hist(
                short_combined,
                bins=bins,
                alpha=0.6,
                label="Short Exposure (Min+Median+Max)",
                color="blue",
                density=True,
                edgecolor="black",
                linewidth=0.5,
            )

            stats_text = (
                f"Short: μ={np.mean(short_combined):.1f}, σ={np.std(short_combined):.1f}\n"
                f"Long: μ={np.mean(long_combined):.1f}, σ={np.std(long_combined):.1f}"
            )

            ax_range.text(
                0.05,
                0.95,
                stats_text,
                transform=ax_range.transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax_range.set_xlabel("Raw Pixel Value", fontsize=11)
        ax_range.set_ylabel("Density", fontsize=11)
        ax_range.set_title(
            f"{sensor_type.upper()} - Pixel Value Range (Min, Median, Max)",
            fontsize=12,
            fontweight="bold",
        )
        ax_range.legend(fontsize=9, ncol=2)
        ax_range.grid(True, alpha=0.3)

        ax_mean = axes[sensor_idx, 1]
        short_mean = np.array(sensor_data.get("short", {}).get("mean", []))
        long_mean = np.array(sensor_data.get("long", {}).get("mean", []))

        if len(short_mean) > 0 and len(long_mean) > 0:
            min_val = min(np.min(short_mean), np.min(long_mean))
            max_val = max(np.max(short_mean), np.max(long_mean))
            use_log = max_val / (min_val + 1) > 100

            if use_log:
                bins = np.logspace(np.log10(min_val + 1), np.log10(max_val + 1), 100)
                ax_mean.set_xscale("log")
            else:
                bins = np.linspace(min_val, max_val, 100)

            ax_mean.hist(
                long_mean,
                bins=bins,
                alpha=0.6,
                label="Long Exposure",
                color="orange",
                density=True,
                edgecolor="black",
                linewidth=0.5,
            )
            ax_mean.hist(
                short_mean,
                bins=bins,
                alpha=0.6,
                label="Short Exposure",
                color="blue",
                density=True,
                edgecolor="black",
                linewidth=0.5,
            )

            stats_text = (
                f"Short: μ={np.mean(short_mean):.1f}, σ={np.std(short_mean):.1f}\n"
                f"Long: μ={np.mean(long_mean):.1f}, σ={np.std(long_mean):.1f}"
            )

            ax_mean.text(
                0.05,
                0.95,
                stats_text,
                transform=ax_mean.transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax_mean.set_xlabel("Mean Raw Pixel Value", fontsize=11)
        ax_mean.set_ylabel("Density", fontsize=11)
        ax_mean.set_title(
            f"{sensor_type.upper()} - Mean Pixel Value", fontsize=12, fontweight="bold"
        )
        ax_mean.legend(fontsize=10)
        ax_mean.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def create_raw_distributions(
    base_dir: Optional[Path] = None, output_path: Optional[Path] = None
) -> None:
    """Create raw pixel value distribution plots from metadata files."""
    if base_dir is None:
        base_dir = get_preprocessing_base_path() / "processed"
    else:
        base_dir = Path(base_dir)

    all_data = {}

    fuji_comprehensive = base_dir / "comprehensive_fuji_tiles_metadata.json"
    fuji_data_values = base_dir / "fuji_data_values.json"
    if fuji_comprehensive.exists() and fuji_data_values.exists():
        fuji_tile_map = load_comprehensive_metadata(fuji_comprehensive)
        fuji_data = process_data_values_with_metadata(fuji_data_values, fuji_tile_map)
        all_data.update(fuji_data)

    sony_comprehensive = base_dir / "comprehensive_sony_tiles_metadata.json"
    sony_data_values = base_dir / "sony_data_values.json"
    if sony_comprehensive.exists() and sony_data_values.exists():
        sony_tile_map = load_comprehensive_metadata(sony_comprehensive)
        sony_data = process_data_values_with_metadata(sony_data_values, sony_tile_map)
        all_data.update(sony_data)

    if not all_data:
        return

    if output_path is None:
        output_path = base_dir / "raw_pixel_distributions.png"
    else:
        output_path = Path(output_path)

    ensure_output_dir(output_path, create_parent=True)
    create_distribution_plots(all_data, output_path)


def main():
    """Main entry point for visualization creation."""
    parser = argparse.ArgumentParser(description="Create visualization comparisons")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "sensor_comparison",
            "comprehensive",
            "raw_data",
            "distributions",
            "scale_files",
        ],
        default="sensor_comparison",
        help="Visualization mode",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory for metadata files (distributions mode, default: preprocessing/processed)",
    )

    # Add common arguments (--output, --output_dir, --show)
    add_common_visualization_arguments(parser)

    args = parser.parse_args()

    if args.mode == "sensor_comparison":
        examples = {
            "sony": ["example_00_sony_00135_00_0.1s_tile_0005"],
            "fuji": [
                "example_00_fuji_00017_00_0.1s_tile_0009",
                "example_01_fuji_20184_00_0.033s_tile_0011",
            ],
        }
        methods = [
            "noisy",
            "clean",
            "exposure_scaled",
            "gaussian_x0_cross",
            "pg_x0_single",
            "pg_x0_cross",
        ]
        method_names = [
            "Noisy Input",
            "Clean Reference",
            "Exposure Scaled",
            "Gaussian x0-cross",
            "PG x0-single",
            "PG x0-cross",
        ]

        create_sensor_comparison(
            examples, methods, method_names, scale_mode="pg_x0_cross"
        )
        create_sensor_comparison(examples, methods, method_names, scale_mode="clean")

    elif args.mode == "comprehensive":
        base_path = get_results_base_path()
        single_sensor_paths = {
            "sony": base_path
            / "test_sony_single_sensor"
            / "example_00_sony_00135_00_0.1s_tile_0005",
        }
        cross_sensor_paths = {
            "sony": base_path
            / "test_sony_cross_sensor"
            / "example_00_sony_00135_00_0.1s_tile_0005",
        }

        output_path = resolve_output_path(args.output, None, create_dir=False)
        create_cross_sensor_comparison(
            single_sensor_paths, cross_sensor_paths, output_path
        )

    elif args.mode == "raw_data":
        samples = {
            "Sony": {
                "noisy": "/home/jilab/anna_OS_ML/DomainAdaptive_PoissonDiffusion/data/raw/SID/Sony/short/20201_00_0.04s.ARW",
                "clean": "/home/jilab/anna_OS_ML/DomainAdaptive_PoissonDiffusion/data/raw/SID/Sony/long/20201_00_10s.ARW",
            },
        }
        output_path = resolve_output_path(args.output, None, create_dir=False)
        if output_path is None:
            output_path = (
                get_project_root()
                / "dataset"
                / "processed"
                / "test_visualizations"
                / "comparison.png"
            )
        create_raw_data_comparison(
            samples, output_path=output_path, show_plot=args.show
        )

    elif args.mode == "distributions":
        base_dir = Path(args.base_dir) if args.base_dir else None
        output_path = resolve_output_path(args.output, None, create_dir=False)
        create_raw_distributions(base_dir=base_dir, output_path=output_path)

    elif args.mode == "scale_files":
        scale_pt_files()


if __name__ == "__main__":
    main()
