#!/usr/bin/env python3
"""
Generate performance comparison charts from comprehensive comparison results.

This script analyzes the results from sample_noisy_pt_lle_guidance.py and creates:
1. Bar charts comparing methods across all metrics
2. Box plots showing performance distribution
3. Radar charts for multi-metric comparison
4. Trade-off analysis plots
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import format_method_name, get_method_colors


def load_results(summary_json: Path):
    """Load summary results from JSON file."""
    with open(summary_json, "r") as f:
        data = json.load(f)
    return data


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


def create_bar_charts(metrics_data, output_dir: Path):
    """Create bar charts for each metric."""
    methods = metrics_data["methods"]

    # Use shared utilities for method names and colors
    method_colors_map = get_method_colors()
    display_names = []
    colors = []
    for m in methods:
        display_names.append(
            format_method_name(m)
            if m
            in ["exposure_scaled", "gaussian_x0", "gaussian_score", "pg_x0", "pg_score"]
            else m
        )
        colors.append(method_colors_map.get(m, "#95a5a6"))

    # Create 2x2 subplot for all metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # SSIM (higher is better)
    ax = axes[0, 0]
    bars = ax.bar(
        display_names,
        metrics_data["ssim"]["mean"],
        yerr=metrics_data["ssim"]["std"],
        capsize=5,
        color=colors,
        alpha=0.8,
    )
    ax.set_ylabel("SSIM", fontsize=12, fontweight="bold")
    ax.set_title(
        "Structural Similarity (Higher is Better)", fontsize=13, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Highlight best
    best_idx = np.argmax(metrics_data["ssim"]["mean"])
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    # PSNR (higher is better)
    ax = axes[0, 1]
    bars = ax.bar(
        display_names,
        metrics_data["psnr"]["mean"],
        yerr=metrics_data["psnr"]["std"],
        capsize=5,
        color=colors,
        alpha=0.8,
    )
    ax.set_ylabel("PSNR (dB)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Peak Signal-to-Noise Ratio (Higher is Better)", fontsize=13, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Highlight best
    best_idx = np.argmax(metrics_data["psnr"]["mean"])
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    # LPIPS (lower is better)
    ax = axes[1, 0]
    lpips_means = [v if not np.isnan(v) else 0 for v in metrics_data["lpips"]["mean"]]
    lpips_stds = [v if not np.isnan(v) else 0 for v in metrics_data["lpips"]["std"]]
    bars = ax.bar(
        display_names, lpips_means, yerr=lpips_stds, capsize=5, color=colors, alpha=0.8
    )
    ax.set_ylabel("LPIPS", fontsize=12, fontweight="bold")
    ax.set_title(
        "Learned Perceptual Similarity (Lower is Better)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Highlight best (minimum)
    valid_lpips = [v for v in lpips_means if v > 0]
    if valid_lpips:
        best_idx = np.argmin([v if v > 0 else np.inf for v in lpips_means])
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

    # NIQE (lower is better)
    ax = axes[1, 1]
    niqe_means = [v if not np.isnan(v) else 0 for v in metrics_data["niqe"]["mean"]]
    niqe_stds = [v if not np.isnan(v) else 0 for v in metrics_data["niqe"]["std"]]
    bars = ax.bar(
        display_names, niqe_means, yerr=niqe_stds, capsize=5, color=colors, alpha=0.8
    )
    ax.set_ylabel("NIQE", fontsize=12, fontweight="bold")
    ax.set_title(
        "Natural Image Quality (Lower is Better)", fontsize=13, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Highlight best (minimum)
    valid_niqe = [v for v in niqe_means if v > 0]
    if valid_niqe:
        best_idx = np.argmin([v if v > 0 else np.inf for v in niqe_means])
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

    plt.suptitle(
        "Comprehensive Comparison: All Methods", fontsize=15, fontweight="bold"
    )
    plt.tight_layout()

    output_path = output_dir / "performance_bar_charts.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Bar charts saved: {output_path}")
    return output_path


def create_radar_chart(metrics_data, output_dir: Path):
    """Create radar chart for multi-metric comparison."""
    methods = metrics_data["methods"]

    # Normalize metrics to [0, 1] for radar chart
    # SSIM: already in [0, 1]
    # PSNR: normalize to [0, 1] using typical range [20, 35]
    # LPIPS: normalize to [0, 1] using range [0, 0.6], inverted (lower is better)
    # NIQE: normalize to [0, 1] using range [0, 30], inverted (lower is better)

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
        # Use color from method colors map if available
        color = colors_map.get(method_lower.replace(" ", "_"), "#ecf0f1")
        for j in range(len(col_labels)):
            table[(i + 1, j)].set_facecolor(color)

    plt.title("Performance Summary Table", fontsize=14, fontweight="bold", pad=20)

    output_path = output_dir / "performance_summary_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Summary table saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate performance charts from comprehensive comparison"
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        required=True,
        help="Path to summary.json from comprehensive comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/performance_charts",
        help="Output directory for charts",
    )

    args = parser.parse_args()

    # Load results
    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}")
        return

    print(f"Loading results from: {summary_path}")
    summary_data = load_results(summary_path)

    # Extract metrics
    metrics_data = extract_metrics(summary_data)
    if metrics_data is None:
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()
