#!/usr/bin/env python3
"""
Create Combined Histogram Visualization

Combines all metric histograms into a single figure with:
- Rows: Metrics (brightness_increase, enhanced_mean, ssim, psnr)
- Columns: Sigma values

Usage:
    python sample/create_combined_histogram.py \
        --csv_path results/comprehensive_enhancement_analysis/enhancement_metrics_all_tiles.csv \
        --output_path results/comprehensive_enhancement_analysis/combined_histograms.png
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_combined_histogram(csv_path: Path, output_path: Path):
    """Create publication-quality combined histogram visualization."""
    logger.info("Loading data from CSV...")

    df = pd.read_csv(csv_path)

    # Get unique sigma values - filter to show every 0.0002 interval
    all_sigma_values = sorted(df["sigma"].unique())
    # Select specific sigma values at 0.0002 intervals
    target_sigmas = [
        0.0002,
        0.0004,
        0.0006,
        0.0008,
        0.001,
        0.0012,
        0.0014,
        0.0016,
        0.0018,
        0.002,
    ]
    sigma_values = [
        s for s in all_sigma_values if any(abs(s - t) < 1e-9 for t in target_sigmas)
    ]
    n_sigmas = len(sigma_values)

    logger.info(f"Found {n_sigmas} sigma values: {sigma_values}")
    logger.info(f"Total data points: {len(df)} ({len(df) // n_sigmas} tiles)")

    # Define metrics to plot (in publication order)
    # Common metrics for low-light enhancement evaluation:
    # 1. PSNR - Peak Signal-to-Noise Ratio (quality)
    # 2. SSIM - Structural Similarity (perceptual quality)
    # 3. Enhanced Mean - Overall brightness level

    metrics = ["psnr", "ssim", "enhanced_mean"]
    metric_labels = {
        "psnr": "PSNR (dB) - Higher is Better",
        "ssim": "SSIM - Higher is Better",
        "enhanced_mean": "Enhanced Mean Value - Higher is Brighter",
    }

    n_metrics = len(metrics)

    # Set publication-quality style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2,
        }
    )

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_metrics, n_sigmas, figsize=(2.8 * n_sigmas, 2.8 * n_metrics)
    )

    logger.info(f"Creating {n_metrics} x {n_sigmas} grid of histograms...")

    # First pass: collect all data for each metric to determine y-axis ranges
    metric_ranges = {}
    for metric in metrics:
        all_counts = []
        for sigma in sigma_values:
            data = df[df["sigma"] == sigma][metric]
            counts, _ = np.histogram(data, bins=25)
            all_counts.extend(counts)
        metric_ranges[metric] = (0, max(all_counts) * 1.05)  # Add 5% headroom

    # Determine best sigmas for highlighting based on available metrics
    agg_dict = {"psnr": "mean", "ssim": "mean"}
    for metric in metrics:
        if metric not in agg_dict:
            agg_dict[metric] = "mean"

    sigma_summary = df.groupby("sigma").agg(agg_dict)

    best_psnr_sigma = sigma_summary["psnr"].idxmax()
    best_ssim_sigma = sigma_summary["ssim"].idxmax()

    # Additional best sigmas based on available metrics
    best_metric_sigmas = {}
    for metric in metrics:
        if metric == "psnr":
            best_metric_sigmas[metric] = best_psnr_sigma
        elif metric == "ssim":
            best_metric_sigmas[metric] = best_ssim_sigma
        elif metric == "mse" and "mse" in sigma_summary.columns:
            best_metric_sigmas[metric] = sigma_summary[
                "mse"
            ].idxmin()  # Lower is better
        elif metric == "enhanced_range" and "enhanced_range" in sigma_summary.columns:
            best_metric_sigmas[metric] = sigma_summary[
                "enhanced_range"
            ].idxmax()  # Higher is better
        elif metric == "enhanced_std" and "enhanced_std" in sigma_summary.columns:
            best_metric_sigmas[metric] = sigma_summary[
                "enhanced_std"
            ].idxmax()  # Higher detail
        elif metric == "enhanced_mean" and "enhanced_mean" in sigma_summary.columns:
            best_metric_sigmas[metric] = sigma_summary[
                "enhanced_mean"
            ].idxmax()  # Brighter

    # Balanced score: 50% PSNR, 50% SSIM
    normalized_psnr = (sigma_summary["psnr"] - sigma_summary["psnr"].min()) / (
        sigma_summary["psnr"].max() - sigma_summary["psnr"].min() + 1e-8
    )
    normalized_ssim = (sigma_summary["ssim"] - sigma_summary["ssim"].min()) / (
        sigma_summary["ssim"].max() - sigma_summary["ssim"].min() + 1e-8
    )

    balance_score = 0.5 * normalized_psnr + 0.5 * normalized_ssim
    best_balanced_sigma = balance_score.idxmax()

    logger.info(f"Best PSNR sigma: {best_psnr_sigma}")
    logger.info(f"Best SSIM sigma: {best_ssim_sigma}")
    for metric, sigma_val in best_metric_sigmas.items():
        logger.info(f"Best {metric} sigma: {sigma_val}")
    logger.info(f"Best balanced sigma: {best_balanced_sigma}")

    # Plot histograms
    for i, metric in enumerate(metrics):
        for j, sigma in enumerate(sigma_values):
            ax = axes[i, j]

            # Get data for this sigma
            data = df[df["sigma"] == sigma][metric]

            # Determine if this is a best sigma for this metric
            is_best_for_metric = False
            border_color = "#333333"
            bg_color = "#2E86AB"

            # Check if this is the best sigma for this metric
            if metric in best_metric_sigmas and sigma == best_metric_sigmas[metric]:
                is_best_for_metric = True
                border_color = "#D62828"
                bg_color = "#E63946"
            elif sigma == best_balanced_sigma:
                border_color = "#F77F00"
                bg_color = "#FFA500"

            # Create histogram with better styling
            n, bins, patches = ax.hist(
                data,
                bins=25,
                alpha=0.75,
                edgecolor="black",
                color=bg_color,
                linewidth=0.8,
            )

            # Add statistics
            mean_val = data.mean()
            std_val = data.std()

            # Add mean line
            ax.axvline(mean_val, color="black", linestyle="--", linewidth=2, alpha=0.8)

            # Set title (only for top row)
            if i == 0:
                if sigma < 0.001:
                    sigma_str = f"{sigma:.1e}"
                else:
                    sigma_str = f"{sigma:.4f}"

                # Add star for best overall
                if sigma == best_balanced_sigma:
                    title_str = f"σ = {sigma_str} ★"
                    title_color = "#F77F00"
                else:
                    title_str = f"σ = {sigma_str}"
                    title_color = "black"

                ax.set_title(
                    title_str, fontsize=11, fontweight="bold", pad=8, color=title_color
                )

            # Set ylabel (only for leftmost column)
            if j == 0:
                ax.set_ylabel("Count", fontsize=10, fontweight="bold")

            # Add statistics box with better formatting
            stats_text = f"μ = {mean_val:.5f}\nσ = {std_val:.5f}"

            # Add marker if best for this metric
            if is_best_for_metric:
                stats_text = f"★ BEST ★\n{stats_text}"

            ax.text(
                0.97,
                0.97,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    edgecolor=border_color,
                    alpha=0.95,
                    linewidth=2,
                ),
            )

            # Set x-axis label (only for bottom row)
            if i == n_metrics - 1:
                ax.set_xlabel("Value", fontsize=9)
                ax.tick_params(axis="x", labelsize=8, rotation=0)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis="y", labelsize=8)

            # Set consistent y-axis range for this row
            ax.set_ylim(metric_ranges[metric])

            # Grid styling
            ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
            ax.set_axisbelow(True)

            # Improve spine visibility with highlighting for best sigmas
            for spine in ax.spines.values():
                spine.set_linewidth(3 if is_best_for_metric else 1.2)
                spine.set_edgecolor(border_color)

    # Add row labels on the left side
    for i, metric in enumerate(metrics):
        axes[i, 0].text(
            -0.35,
            0.5,
            metric_labels[metric],
            transform=axes[i, 0].transAxes,
            fontsize=13,
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            fontweight="bold",
        )

    # Main title with better formatting and recommendations
    n_tiles = len(df) // n_sigmas
    title_text = (
        f"Low-Light Enhancement: Metric Distributions Across Sigma Values\n"
        f"Analysis of {n_tiles} Photography Test Tiles | "
        f"★ Best Overall: σ={best_balanced_sigma:.4f} | "
        f"Red = Best for Metric, Orange = Best Balanced"
    )

    plt.suptitle(title_text, fontsize=14, fontweight="bold", y=0.995)

    # Adjust layout
    plt.tight_layout(rect=[0.02, 0, 1, 0.99])

    # Save figure at high quality
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()

    # Reset style
    plt.rcParams.update(plt.rcParamsDefault)

    logger.info(f"✓ Publication-quality combined histogram saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create combined histogram visualization"
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to enhancement metrics CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for combined histogram",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return

    create_combined_histogram(csv_path, output_path)

    logger.info("=" * 80)
    logger.info("🎉 COMBINED HISTOGRAM CREATED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
