#!/usr/bin/env python3
"""
Unified performance visualization script.

This script combines functionality from:
- generate_performance_charts.py: Comprehensive comparison charts
- visualize_tuning_results.py: Hyperparameter tuning visualizations

Usage:
    # Comprehensive comparison mode
    python generate_performance_visualizations.py --mode comprehensive --summary_json results/summary.json

    # Hyperparameter tuning mode
    python generate_performance_visualizations.py --mode tuning results/hyperparameter_tuning/sony/tuning_results.json
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from visualization.visualizations import (
    add_common_visualization_arguments,
    ensure_output_dir,
    format_method_name,
    get_method_colors,
    load_results_json,
    resolve_output_path,
    setup_visualization_logging,
)

# Import tuning visualization utilities
try:
    from core.utils.visualization_utils import create_heatmap, create_line_plots
except ImportError:
    create_heatmap = None
    create_line_plots = None

logger = setup_visualization_logging()


# ============================================================================
# Comprehensive Comparison Mode Functions
# ============================================================================


def extract_metrics(summary_data):
    """Extract metrics for all methods from summary."""
    if "comprehensive_aggregate_metrics" not in summary_data:
        print("No comprehensive metrics found in summary")
        return None

    metrics_data = summary_data["comprehensive_aggregate_metrics"]

    # Extract method names and metrics
    methods = []
    ssim_vals = []
    psnr_vals = []
    lpips_vals = []
    niqe_vals = []

    ssim_stds = []
    psnr_stds = []
    lpips_stds = []
    niqe_stds = []

    for method, metrics in metrics_data.items():
        if method in ["noisy", "clean"]:
            continue

        methods.append(method)
        ssim_vals.append(metrics.get("mean_ssim", 0))
        psnr_vals.append(metrics.get("mean_psnr", 0))
        lpips_vals.append(metrics.get("mean_lpips", np.nan))
        niqe_vals.append(metrics.get("mean_niqe", np.nan))

        ssim_stds.append(metrics.get("std_ssim", 0))
        psnr_stds.append(metrics.get("std_psnr", 0))
        lpips_stds.append(metrics.get("std_lpips", 0))
        niqe_stds.append(metrics.get("std_niqe", 0))

    return {
        "methods": methods,
        "ssim": {"mean": ssim_vals, "std": ssim_stds},
        "psnr": {"mean": psnr_vals, "std": psnr_stds},
        "lpips": {"mean": lpips_vals, "std": lpips_stds},
        "niqe": {"mean": niqe_vals, "std": niqe_stds},
    }


def _create_bar_subplot(
    ax,
    methods: List[str],
    metric_values: List[float],
    metric_stds: Optional[List[float]],
    metric_name: str,
    title: str,
    ylabel: str,
    higher_is_better: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
):
    """Create a bar chart subplot using shared utilities."""
    from core.utils.visualization_utils import format_method_name, get_method_colors

    method_colors_map = get_method_colors()
    display_names = [format_method_name(m) for m in methods]
    colors = [method_colors_map.get(m, "#95a5a6") for m in methods]

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
    if higher_is_better:
        best_idx = np.argmax(metric_values)
    else:
        # For lower-is-better, find minimum among valid values
        valid_values = [v if v > 0 else np.inf for v in metric_values]
        best_idx = (
            np.argmin(valid_values) if any(v < np.inf for v in valid_values) else 0
        )

    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    # Labels
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    if ylim:
        ax.set_ylim(ylim)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def create_bar_charts(metrics_data, output_dir: Path):
    """Create bar charts for each metric."""
    methods = metrics_data["methods"]

    # Create 2x2 subplot for all metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SSIM (higher is better)
    _create_bar_subplot(
        axes[0, 0],
        methods=methods,
        metric_values=metrics_data["ssim"]["mean"],
        metric_stds=metrics_data["ssim"]["std"],
        metric_name="SSIM",
        title="Structural Similarity (Higher is Better)",
        ylabel="SSIM",
        higher_is_better=True,
        ylim=(0, 1.0),
    )

    # PSNR (higher is better)
    _create_bar_subplot(
        axes[0, 1],
        methods=methods,
        metric_values=metrics_data["psnr"]["mean"],
        metric_stds=metrics_data["psnr"]["std"],
        metric_name="PSNR",
        title="Peak Signal-to-Noise Ratio (Higher is Better)",
        ylabel="PSNR (dB)",
        higher_is_better=True,
    )

    # LPIPS (lower is better)
    lpips_means = [v if not np.isnan(v) else 0 for v in metrics_data["lpips"]["mean"]]
    lpips_stds = [v if not np.isnan(v) else 0 for v in metrics_data["lpips"]["std"]]
    _create_bar_subplot(
        axes[1, 0],
        methods=methods,
        metric_values=lpips_means,
        metric_stds=lpips_stds,
        metric_name="LPIPS",
        title="Learned Perceptual Similarity (Lower is Better)",
        ylabel="LPIPS",
        higher_is_better=False,
    )

    # NIQE (lower is better)
    niqe_means = [v if not np.isnan(v) else 0 for v in metrics_data["niqe"]["mean"]]
    niqe_stds = [v if not np.isnan(v) else 0 for v in metrics_data["niqe"]["std"]]
    _create_bar_subplot(
        axes[1, 1],
        methods=methods,
        metric_values=niqe_means,
        metric_stds=niqe_stds,
        metric_name="NIQE",
        title="Natural Image Quality (Lower is Better)",
        ylabel="NIQE",
        higher_is_better=False,
    )

    plt.suptitle(
        "Comprehensive Comparison: All Methods", fontsize=15, fontweight="bold"
    )
    plt.tight_layout()

    output_path = output_dir / "performance_bar_charts.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Bar charts saved: {output_path}")
    return output_path


def create_radar_chart(metrics_data, output_dir: Path):
    """Create radar chart for multi-metric comparison."""
    methods = metrics_data["methods"]

    # Normalize metrics to [0, 1] for radar chart
    ssim_norm = np.array(metrics_data["ssim"]["mean"])
    psnr_norm = (np.array(metrics_data["psnr"]["mean"]) - 20) / 15  # [20, 35] -> [0, 1]
    psnr_norm = np.clip(psnr_norm, 0, 1)

    lpips_vals = np.array(
        [v if not np.isnan(v) else 0.5 for v in metrics_data["lpips"]["mean"]]
    )
    lpips_norm = 1.0 - np.clip(lpips_vals / 0.6, 0, 1)  # Invert (lower is better)

    niqe_vals = np.array(
        [v if not np.isnan(v) else 15 for v in metrics_data["niqe"]["mean"]]
    )
    niqe_norm = 1.0 - np.clip(niqe_vals / 30, 0, 1)  # Invert (lower is better)

    # Set up radar chart
    categories = [
        "SSIM\n(Structure)",
        "PSNR\n(Fidelity)",
        "LPIPS\n(Perceptual)",
        "NIQE\n(Natural)",
    ]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Plot each method
    colors_map = get_method_colors()

    for i, method in enumerate(methods):
        if method in ["noisy", "clean"]:
            continue

        values = [ssim_norm[i], psnr_norm[i], lpips_norm[i], niqe_norm[i]]
        values += values[:1]  # Complete the circle

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=format_method_name(method),
            color=colors_map.get(method, "#95a5a6"),
        )
        ax.fill(angles, values, alpha=0.15, color=colors_map.get(method, "#95a5a6"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.title(
        "Multi-Metric Radar Comparison\n(All metrics normalized to [0,1], higher is better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    output_path = output_dir / "performance_radar_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Radar chart saved: {output_path}")
    return output_path


def create_tradeoff_plots(metrics_data, output_dir: Path):
    """Create trade-off scatter plots."""
    methods = metrics_data["methods"]

    colors_map = get_method_colors()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # SSIM vs LPIPS trade-off
    ax = axes[0]
    for i, method in enumerate(methods):
        if method in ["noisy", "clean"]:
            continue

        ssim = metrics_data["ssim"]["mean"][i]
        lpips = metrics_data["lpips"]["mean"][i]

        if not np.isnan(lpips):
            ax.scatter(
                ssim,
                lpips,
                s=200,
                alpha=0.7,
                color=colors_map.get(method, "#95a5a6"),
                edgecolors="black",
                linewidth=2,
                label=format_method_name(method),
            )
            ax.text(
                ssim,
                lpips,
                format_method_name(method),
                fontsize=9,
                ha="center",
                va="bottom",
            )

    ax.set_xlabel("SSIM (Higher is Better)", fontsize=12, fontweight="bold")
    ax.set_ylabel("LPIPS (Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_title("Structure vs Perceptual Trade-off", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower LPIPS is better, so invert

    # PSNR vs NIQE trade-off
    ax = axes[1]
    for i, method in enumerate(methods):
        if method in ["noisy", "clean"]:
            continue

        psnr = metrics_data["psnr"]["mean"][i]
        niqe = metrics_data["niqe"]["mean"][i]

        if not np.isnan(niqe):
            ax.scatter(
                psnr,
                niqe,
                s=200,
                alpha=0.7,
                color=colors_map.get(method, "#95a5a6"),
                edgecolors="black",
                linewidth=2,
                label=format_method_name(method),
            )
            ax.text(
                psnr,
                niqe,
                format_method_name(method),
                fontsize=9,
                ha="center",
                va="bottom",
            )

    ax.set_xlabel("PSNR (Higher is Better)", fontsize=12, fontweight="bold")
    ax.set_ylabel("NIQE (Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_title("Fidelity vs Naturalness Trade-off", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower NIQE is better

    plt.suptitle("Performance Trade-off Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "performance_tradeoff_plots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Trade-off plots saved: {output_path}")
    return output_path


def create_summary_table(metrics_data, output_dir: Path):
    """Create summary table image."""
    methods = metrics_data["methods"]

    # Prepare table data
    table_data = []
    for i, method in enumerate(methods):
        if method in ["noisy", "clean"]:
            continue

        row = [
            method.replace("_", " ").title(),
            f"{metrics_data['ssim']['mean'][i]:.3f} ± {metrics_data['ssim']['std'][i]:.3f}",
            f"{metrics_data['psnr']['mean'][i]:.1f} ± {metrics_data['psnr']['std'][i]:.1f}",
            f"{metrics_data['lpips']['mean'][i]:.3f} ± {metrics_data['lpips']['std'][i]:.3f}"
            if not np.isnan(metrics_data["lpips"]["mean"][i])
            else "N/A",
            f"{metrics_data['niqe']['mean'][i]:.1f} ± {metrics_data['niqe']['std'][i]:.1f}"
            if not np.isnan(metrics_data["niqe"]["mean"][i])
            else "N/A",
        ]
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("tight")
    ax.axis("off")

    col_labels = ["Method", "SSIM ↑", "PSNR (dB) ↑", "LPIPS ↓", "NIQE ↓"]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.18, 0.18, 0.18, 0.18],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style rows
    colors_map = get_method_colors()

    for i, row in enumerate(table_data):
        method_lower = row[0].lower()
        color = colors_map.get(method_lower.replace(" ", "_"), "#ecf0f1")
        for j in range(len(col_labels)):
            table[(i + 1, j)].set_facecolor(color)

    plt.title("Performance Summary Table", fontsize=14, fontweight="bold", pad=20)

    output_path = output_dir / "performance_summary_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Summary table saved: {output_path}")
    return output_path


# ============================================================================
# Hyperparameter Tuning Mode Functions
# ============================================================================


def print_tuning_summary(results: dict) -> None:
    """Print summary statistics for tuning results."""
    data = results["results"]

    print("=" * 60)
    print("Hyperparameter Tuning Summary")
    print("=" * 60)
    print(f"Sensor: {results.get('sensor', 'unknown')}")
    print(f"Num examples: {results.get('num_examples', 'unknown')}")
    print()

    # Find best configurations
    best_psnr = max(data, key=lambda x: x["metrics"]["psnr"]["mean"])
    best_ssim = max(data, key=lambda x: x["metrics"]["ssim"]["mean"])

    # Best combined score
    best_combined = max(
        data,
        key=lambda x: x["metrics"]["psnr"]["mean"] * 0.5
        + x["metrics"]["ssim"]["mean"] * 50.0,
    )

    print("Best PSNR:")
    print(f"  kappa={best_psnr['kappa']}, num_steps={best_psnr['num_steps']}")
    print(
        f"  PSNR: {best_psnr['metrics']['psnr']['mean']:.2f}±{best_psnr['metrics']['psnr']['std']:.2f} dB"
    )
    print(
        f"  SSIM: {best_psnr['metrics']['ssim']['mean']:.4f}±{best_psnr['metrics']['ssim']['std']:.4f}"
    )
    print(f"  Time: {best_psnr['avg_time_per_tile']:.2f}s/tile")
    print()

    print("Best SSIM:")
    print(f"  kappa={best_ssim['kappa']}, num_steps={best_ssim['num_steps']}")
    print(
        f"  PSNR: {best_ssim['metrics']['psnr']['mean']:.2f}±{best_ssim['metrics']['psnr']['std']:.2f} dB"
    )
    print(
        f"  SSIM: {best_ssim['metrics']['ssim']['mean']:.4f}±{best_ssim['metrics']['ssim']['std']:.4f}"
    )
    print(f"  Time: {best_ssim['avg_time_per_tile']:.2f}s/tile")
    print()

    print("Best Combined Score (PSNR*0.5 + SSIM*50):")
    print(f"  kappa={best_combined['kappa']}, num_steps={best_combined['num_steps']}")
    print(
        f"  PSNR: {best_combined['metrics']['psnr']['mean']:.2f}±{best_combined['metrics']['psnr']['std']:.2f} dB"
    )
    print(
        f"  SSIM: {best_combined['metrics']['ssim']['mean']:.4f}±{best_combined['metrics']['ssim']['std']:.4f}"
    )
    print(f"  Time: {best_combined['avg_time_per_tile']:.2f}s/tile")
    print("=" * 60)


def create_tuning_visualizations(results: dict, output_dir: Path) -> None:
    """Create visualizations for hyperparameter tuning results."""
    if create_heatmap is None or create_line_plots is None:
        print("\nWARNING: Tuning visualization utilities not available.")
        print("Install required dependencies or check core.utils.visualization_utils")
        return

    try:
        # Create heatmaps
        print("\nGenerating heatmaps...")
        fig_psnr = create_heatmap(
            results,
            metric="psnr",
            x_key="kappa",
            y_key="num_steps",
            x_label="Kappa (guidance strength)",
            y_label="Num Steps",
        )
        fig_psnr.savefig(output_dir / "heatmap_psnr.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_dir / 'heatmap_psnr.png'}")

        fig_ssim = create_heatmap(
            results,
            metric="ssim",
            x_key="kappa",
            y_key="num_steps",
            x_label="Kappa (guidance strength)",
            y_label="Num Steps",
        )
        fig_ssim.savefig(output_dir / "heatmap_ssim.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_dir / 'heatmap_ssim.png'}")

        # Create line plots
        print("\nGenerating line plots...")
        fig_lines = create_line_plots(
            results,
            metrics=["psnr", "ssim"],
            x_key="kappa",
            y_key="num_steps",
        )
        fig_lines.savefig(output_dir / "line_plots.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_dir / 'line_plots.png'}")

        print("\nVisualization complete!")

    except Exception as e:
        print(f"\nWARNING: Failed to generate plots: {e}")
        print("Summary statistics printed above.")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate performance visualizations (comprehensive comparison or hyperparameter tuning)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["comprehensive", "tuning"],
        default="comprehensive",
        help="Visualization mode: 'comprehensive' for method comparison, 'tuning' for hyperparameter tuning",
    )

    # Comprehensive mode arguments
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Path to summary.json from comprehensive comparison (required for comprehensive mode)",
    )

    # Tuning mode arguments
    parser.add_argument(
        "tuning_results_file",
        type=Path,
        nargs="?",
        default=None,
        help="Path to tuning_results.json file (required for tuning mode)",
    )

    # Common arguments
    add_common_visualization_arguments(parser)

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots (only print summary, tuning mode only)",
    )

    args = parser.parse_args()

    if args.mode == "comprehensive":
        # Comprehensive comparison mode
        if args.summary_json is None:
            parser.error("--summary_json is required for comprehensive mode")

        summary_path = Path(args.summary_json)
        if not summary_path.exists():
            print(f"Error: Summary file not found: {summary_path}")
            return

        print(f"Loading results from: {summary_path}")
        summary_data = load_results_json(summary_path)

        # Extract metrics
        metrics_data = extract_metrics(summary_data)
        if metrics_data is None:
            return

        # Create output directory
        default_output = Path("results/performance_charts")
        output_dir = resolve_output_path(
            args.output_dir, default_output, create_dir=True
        )

        print(f"\nGenerating performance charts...")
        print(f"Methods: {', '.join(metrics_data['methods'])}")
        print(f"Output directory: {output_dir}")
        print("")

        # Generate charts
        create_bar_charts(metrics_data, output_dir)
        create_radar_chart(metrics_data, output_dir)
        create_tradeoff_plots(metrics_data, output_dir)
        create_summary_table(metrics_data, output_dir)

        print("\n" + "=" * 60)
        print("✅ All performance charts generated successfully!")
        print("=" * 60)
        print(f"Charts saved to: {output_dir}")
        print("")

    elif args.mode == "tuning":
        # Hyperparameter tuning mode
        if args.tuning_results_file is None:
            parser.error("tuning_results_file is required for tuning mode")

        if not args.tuning_results_file.exists():
            print(f"ERROR: Results file not found: {args.tuning_results_file}")
            sys.exit(1)

        if load_results_json is None:
            logger.error("load_results_json utility not available")
            sys.exit(1)

        # Load results
        results = load_results_json(args.tuning_results_file)

        # Print summary
        print_tuning_summary(results)

        # Generate plots
        if not args.no_plots:
            default_output = args.tuning_results_file.parent
            output_dir = resolve_output_path(
                args.output_dir, default_output, create_dir=True
            )
            create_tuning_visualizations(results, output_dir)


if __name__ == "__main__":
    main()
