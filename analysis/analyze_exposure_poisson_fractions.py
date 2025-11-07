#!/usr/bin/env python3
"""
Analyze Real Exposure Data: Compute Poisson Fractions and Create Scatter Plots

This script loads real test data (short/long exposure pairs) and computes
the actual Poisson fractions based on exposure ratios and signal levels.
Creates scatter plots showing the relationship between exposure time and
noise regime (Poisson fraction).

Usage:
    # Simplest - all defaults (uses test set, stitched scenes)
    python analysis/analyze_exposure_poisson_fractions.py --sensor fuji

    # With custom directory
    python analysis/analyze_exposure_poisson_fractions.py \
        --sensor fuji \
        --test_images_dir results/guidance_comparison_fuji

    # Using tiles instead of scenes
    python analysis/analyze_exposure_poisson_fractions.py \
        --sensor fuji \
        --use_tiles
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils.analysis_utils import save_json_safe
from core.utils.data_utils import load_noise_calibration
from core.utils.file_utils import (
    count_regimes,
    find_scene_directories,
    find_split_file,
    load_stitched_image,
    parse_exposure_from_scene_dir,
    parse_exposure_from_split_line,
)
from core.utils.sensor_utils import (
    NOISE_REGIME_BAR_COLORS,
    NOISE_REGIME_COLORS,
    NOISE_REGIME_LABELS,
    NOISE_REGIME_NAMES,
    READ_NOISE_THRESHOLD,
    SHOT_NOISE_THRESHOLD,
    NoiseRegimeClassifier,
    convert_calibration_to_poisson_coeff,
    convert_calibration_to_sigma_r,
    load_sensor_calibration_from_metadata,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_poisson_fraction_from_image(
    long_image: torch.Tensor,
    exposure_ratio: float,
    poisson_coeff: float,
    sigma_r: float,
    sensor_range: float,
) -> Tuple[float, float, float, str]:
    """
    Compute Poisson fraction from real image data.

    Args:
        long_image: Long exposure image in normalized [-1, 1]
        exposure_ratio: Short exposure / Long exposure time ratio
        poisson_coeff: Poisson coefficient (a in variance = a·μ + b)
        sigma_r: Read noise standard deviation
        sensor_range: Sensor range (white_level - black_level)

    Returns:
        Tuple of (poisson_fraction, variance_ratio, signal_mean, regime)
    """
    # Convert from normalized [-1,1] to [0,1] to sensor range [0, s]
    long_01 = (long_image + 1) / 2
    long_sensor_range = long_01 * sensor_range  # [0, s] units

    # Expected signal at short exposure (in sensor range units)
    signal_mean = (long_sensor_range * exposure_ratio).mean().item()

    # Compute Poisson fraction
    poisson_term = poisson_coeff * signal_mean
    total_variance = poisson_term + sigma_r**2
    poisson_fraction = poisson_term / total_variance if total_variance > 0 else 0.0

    # Variance ratio for regime classification
    variance_ratio = poisson_term / (sigma_r**2) if sigma_r ** 2 > 0 else 0.0

    classifier = NoiseRegimeClassifier(poisson_coeff, sigma_r)
    regime, _ = classifier.classify(signal_mean)

    return poisson_fraction, variance_ratio, signal_mean, regime


def analyze_scenes_from_directory(
    test_images_dir: Path,
    sensor: str,
    poisson_coeff: float,
    sigma_r: float,
    sensor_range: float,
    split_file: Optional[Path] = None,
) -> List[Dict]:
    """Analyze scenes from guidance comparison results directory."""
    results = []

    # Load split file to get exposure times
    exposure_map = {}  # scene_id -> (short_exposure, long_exposure, exposure_ratio)

    if split_file and split_file.exists():
        with open(split_file, "r") as f:
            for line in f:
                parsed = parse_exposure_from_split_line(line)
                if parsed:
                    scene_id, short_exp, long_exp = parsed
                    exposure_ratio = short_exp / long_exp if long_exp > 0 else 0.0
                    exposure_map[scene_id] = (short_exp, long_exp, exposure_ratio)

    # Scan scene directories
    scene_dirs = find_scene_directories(test_images_dir, prefix="")

    for scene_dir in tqdm(scene_dirs, desc="Analyzing scenes"):
        # Try to get exposure from directory name
        short_exposure = parse_exposure_from_scene_dir(scene_dir)

        # Load long exposure image
        stitched_file = scene_dir / "stitched_long.pt"
        long_image = load_stitched_image(stitched_file, device="cpu")
        if long_image is None:
            continue

        # Get exposure ratio
        if short_exposure:
            # Assume long exposure is 10s (common in SID)
            long_exposure = 10.0
            exposure_ratio = short_exposure / long_exposure
        else:
            # Try to get from split file
            scene_id = scene_dir.name.split("_")[1] if "_" in scene_dir.name else None
            if scene_id and scene_id in exposure_map:
                short_exp, long_exp, exposure_ratio = exposure_map[scene_id]
            else:
                # Skip if we can't determine exposure
                continue

        # Compute Poisson fraction
        (
            poisson_fraction,
            variance_ratio,
            signal_mean,
            regime,
        ) = compute_poisson_fraction_from_image(
            long_image, exposure_ratio, poisson_coeff, sigma_r, sensor_range
        )

        results.append(
            {
                "scene_dir": str(scene_dir.name),
                "exposure_ratio": exposure_ratio,
                "short_exposure": short_exposure
                if short_exposure
                else (short_exp if "short_exp" in locals() else None),
                "long_exposure": long_exposure
                if short_exposure
                else (long_exp if "long_exp" in locals() else None),
                "poisson_fraction": poisson_fraction,
                "variance_ratio": variance_ratio,
                "signal_mean": signal_mean,
                "regime": regime,
            }
        )

    return results


def create_scatter_plots(results: List[Dict], output_dir: Path):
    """Create scatter plots showing Poisson fractions vs exposure ratios."""

    if not results:
        logger.warning("No results to plot")
        return

    # Extract data
    exposure_ratios = [r["exposure_ratio"] for r in results]
    poisson_fractions = [r["poisson_fraction"] for r in results]
    variance_ratios = [r["variance_ratio"] for r in results]
    signal_means = [r["signal_mean"] for r in results]
    regimes = [r["regime"] for r in results]

    # Color map for regimes
    colors = [NOISE_REGIME_COLORS.get(r, "gray") for r in regimes]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Exposure ratio vs Poisson fraction (main plot)
    ax = axes[0, 0]
    scatter = ax.scatter(
        exposure_ratios,
        poisson_fractions,
        c=colors,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(
        y=SHOT_NOISE_THRESHOLD,
        color="green",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Shot-noise threshold (20%)",
    )
    ax.set_xlabel("Exposure Ratio (short/long)", fontsize=11)
    ax.set_ylabel("Poisson Fraction", fontsize=11)
    ax.set_title("Exposure Ratio vs Poisson Fraction", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.6, label="Read-noise dominated"),
        Patch(facecolor="orange", alpha=0.6, label="Transitional"),
        Patch(facecolor="green", alpha=0.6, label="Shot-noise dominated"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Plot 2: Exposure ratio vs Variance ratio (log scale)
    ax = axes[0, 1]
    ax.scatter(
        exposure_ratios,
        variance_ratios,
        c=colors,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(
        y=READ_NOISE_THRESHOLD,
        color="orange",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Read-noise threshold",
    )
    ax.axhline(
        y=SHOT_NOISE_THRESHOLD,
        color="green",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Shot-noise threshold",
    )
    ax.set_xlabel("Exposure Ratio (short/long)", fontsize=11)
    ax.set_ylabel("Variance Ratio (a·μ / σ_r²)", fontsize=11)
    ax.set_title("Regime Classification", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Signal level vs Poisson fraction
    ax = axes[1, 0]
    ax.scatter(
        signal_means,
        poisson_fractions,
        c=colors,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(
        y=SHOT_NOISE_THRESHOLD,
        color="green",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Shot-noise threshold (20%)",
    )
    ax.set_xlabel("Signal Mean (sensor range units)", fontsize=11)
    ax.set_ylabel("Poisson Fraction", fontsize=11)
    ax.set_title("Signal Level vs Poisson Fraction", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Regime distribution histogram
    ax = axes[1, 1]
    regime_counts = count_regimes(regimes)
    counts = [regime_counts.get(r, 0) for r in NOISE_REGIME_NAMES]

    bars = ax.bar(
        NOISE_REGIME_LABELS,
        counts,
        color=NOISE_REGIME_BAR_COLORS,
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_ylabel("Number of Scenes", fontsize=11)
    ax.set_title("Regime Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.suptitle(
        "Real Exposure Data: Poisson Fraction Analysis",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    fig.savefig(
        output_dir / "exposure_poisson_analysis.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze real exposure data to compute Poisson fractions"
    )
    parser.add_argument(
        "--test_images_dir",
        type=str,
        default=None,
        help="Directory with test scenes (auto-detected if not provided). "
        "For scenes: results/guidance_comparison_{sensor}. "
        "For tiles: dataset/processed/pt_tiles/{sensor}/long",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="fuji",
        choices=["sony", "fuji"],
        help="Sensor type",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: results/exposure_poisson_analysis_{sensor})",
    )
    parser.add_argument(
        "--use_scenes",
        action="store_true",
        help="Use stitched scenes from guidance_comparison results (default: True for scenes)",
    )
    parser.add_argument(
        "--use_tiles",
        action="store_true",
        help="Use individual tiles instead of stitched scenes",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Split file (auto-detected if not provided, uses test set by default)",
    )

    args = parser.parse_args()

    # Auto-detect test_images_dir if not provided
    if args.test_images_dir is None:
        if args.use_tiles:
            # Use tiles from dataset
            test_images_dir = Path(f"dataset/processed/pt_tiles/{args.sensor}/long")
        else:
            # Use stitched scenes from guidance_comparison results (default)
            test_images_dir = Path(f"results/guidance_comparison_{args.sensor}")
            args.use_scenes = True  # Auto-enable if using guidance_comparison

        if not test_images_dir.exists():
            raise ValueError(
                f"Test images directory not found: {test_images_dir}\n"
                f"For scenes, run test_guidance_comparison_unified.sh first.\n"
                f"For tiles, ensure dataset is processed."
            )
        logger.info(f"Auto-detected test images directory: {test_images_dir}")
    else:
        test_images_dir = Path(args.test_images_dir)
        if not test_images_dir.exists():
            raise ValueError(f"Test images directory not found: {test_images_dir}")

    # Auto-detect split file (default: test set)
    if args.split_file is None:
        split_file = find_split_file(args.sensor, split_type="test")
        if split_file:
            logger.info(f"Auto-detected test split file: {split_file}")
        else:
            logger.warning(
                f"Test split file not auto-detected. Expected: "
                f"dataset/splits/{args.sensor.capitalize()}_test_list.txt"
            )
            split_file = None
    else:
        split_file = Path(args.split_file)
        if not split_file.exists():
            raise ValueError(f"Split file not found: {split_file}")

    # Auto-detect output_dir if not provided
    if args.output_dir is None:
        output_dir = Path(f"results/exposure_poisson_analysis_{args.sensor}")
        logger.info(f"Auto-detected output directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sensor calibration
    black_level, white_level = load_sensor_calibration_from_metadata(args.sensor)
    sensor_range = white_level - black_level

    # Load noise calibration
    calibration_data = load_noise_calibration(
        args.sensor, Path("dataset/processed"), data_root=Path("dataset/processed")
    )

    if calibration_data is None:
        raise ValueError(f"Noise calibration not found for {args.sensor}")

    sigma_r = convert_calibration_to_sigma_r(calibration_data["b"], sensor_range)
    poisson_coeff = convert_calibration_to_poisson_coeff(
        calibration_data["a"], sensor_range
    )

    logger.info(f"Sensor: {args.sensor}, Output: {output_dir}")

    # Analyze scenes
    results = analyze_scenes_from_directory(
        test_images_dir, args.sensor, poisson_coeff, sigma_r, sensor_range, split_file
    )

    if not results:
        logger.error("No scenes found to analyze")
        return

    logger.info(f"Analyzed {len(results)} scenes")

    # Save results
    results_file = output_dir / "exposure_poisson_results.json"
    save_json_safe(results, results_file)

    # Create scatter plots
    create_scatter_plots(results, output_dir)

    # Print summary
    poisson_fracs = [r["poisson_fraction"] for r in results]
    exposure_ratios_list = [r["exposure_ratio"] for r in results]

    print("\n" + "=" * 80)
    print("EXPOSURE-POISSON FRACTION ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total scenes analyzed: {len(results)}")
    print(f"\nExposure ratios:")
    print(f"  Min: {min(exposure_ratios_list):.4f}")
    print(f"  Max: {max(exposure_ratios_list):.4f}")
    print(f"  Mean: {np.mean(exposure_ratios_list):.4f}")
    print(f"\nPoisson fractions:")
    print(f"  Min: {min(poisson_fracs)*100:.2f}%")
    print(f"  Max: {max(poisson_fracs)*100:.2f}%")
    print(f"  Mean: {np.mean(poisson_fracs)*100:.2f}%")

    # Regime distribution
    regimes_list = [r["regime"] for r in results]
    regime_counts = count_regimes(regimes_list)

    print(f"\nRegime distribution:")
    for regime, count in sorted(regime_counts.items()):
        print(f"  {regime}: {count} ({count/len(results)*100:.1f}%)")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
