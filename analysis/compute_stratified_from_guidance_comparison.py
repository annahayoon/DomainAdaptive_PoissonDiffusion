#!/usr/bin/env python3
"""
Compute stratified PSNR improvements from guidance comparison results.

This script reads stitched scene images from test_guidance_comparison_unified.sh output
and computes stratified PSNR improvements (PG vs Gaussian) by signal level,
matching the format shown in QUICK_START.md.

Usage:
    python analysis/compute_stratified_from_guidance_comparison.py \
        --results_dir results/guidance_comparison_sony \
        --sensor sony \
        --output results/stratified_results.json
"""

import argparse
import logging

# Add project root to path
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.stratified_evaluation import (
    StratifiedEvaluator,
    format_significance_marker,
)
from config.sensor_config import get_sensor_config
from core.utils.analysis_utils import ensure_tensor_format, save_json_safe
from core.utils.file_utils import find_scene_directories, load_stitched_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_stratified_for_scene(
    scene_dir: Path,
    evaluator: StratifiedEvaluator,
    baseline_method: str = "gaussian_x0",
    proposed_method: str = "pg_x0",
    clean_method: str = "long",
) -> Optional[Dict]:
    """
    Compute stratified metrics for a single scene.

    Returns:
        Dict with stratified comparison results, or None if files are missing
    """
    # Load required images
    clean_path = scene_dir / f"stitched_{clean_method}.pt"
    baseline_path = scene_dir / f"stitched_{baseline_method}.pt"
    proposed_path = scene_dir / f"stitched_{proposed_method}.pt"

    clean = load_stitched_image(clean_path)
    baseline = load_stitched_image(baseline_path)
    proposed = load_stitched_image(proposed_path)

    if clean is None or baseline is None or proposed is None:
        missing = []
        if clean is None:
            missing.append(clean_method)
        if baseline is None:
            missing.append(baseline_method)
        if proposed is None:
            missing.append(proposed_method)
        logger.warning(f"Scene {scene_dir.name}: Missing files for {missing}, skipping")
        return None

    # Ensure tensors are in correct format: (C, H, W)
    clean = ensure_tensor_format(clean, expected_dims=3)
    baseline = ensure_tensor_format(baseline, expected_dims=3)
    proposed = ensure_tensor_format(proposed, expected_dims=3)

    # Compute stratified metrics
    method_results = {
        baseline_method: baseline,
        proposed_method: proposed,
    }

    stratified_comparison = evaluator.compare_methods_stratified(clean, method_results)

    # Compute improvements
    if (
        baseline_method in stratified_comparison
        and proposed_method in stratified_comparison
    ):
        improvements = evaluator.compute_improvement_matrix(
            stratified_comparison[baseline_method],
            stratified_comparison[proposed_method],
        )
    else:
        improvements = {}

    return {
        "scene_id": scene_dir.name,
        "stratified_metrics": stratified_comparison,
        "improvements": improvements,
    }


def print_stratified_results(
    aggregated_results: Dict,
    all_scene_results: List[Dict],
    baseline_method: str = "gaussian_x0",
    proposed_method: str = "pg_x0",
):
    """Print stratified results in the format shown in QUICK_START.md."""
    print("=" * 70)
    print("🔬 STRATIFIED EVALUATION: Computing metrics by signal level")
    print("=" * 70)
    print()

    # Print per-scene improvements
    significance = aggregated_results.get("statistical_significance", {})
    bin_order = ["very_low", "low", "medium", "high"]

    for scene_idx, scene_result in enumerate(all_scene_results):
        scene_id = scene_result.get("scene_id", f"scene_{scene_idx+1}")
        improvements = scene_result.get("improvements", {})

        if not improvements:
            continue

        print(f"  Scene {scene_idx + 1}/{len(all_scene_results)}: {scene_id}")
        print(
            f"    Stratified PSNR improvements ({proposed_method} vs {baseline_method}):"
        )

        for bin_name in bin_order:
            if bin_name in improvements:
                delta = improvements[bin_name]
                print(f"      {bin_name:12s} : {delta:+.1f} dB")

        print()

    # Print aggregated results
    print("-" * 70)
    print("Aggregated Stratified Results (All Scenes):")
    print("-" * 70)

    for bin_name in bin_order:
        if bin_name not in significance:
            continue

        sig_data = significance[bin_name]
        mean_imp = sig_data.get("mean_improvement", float("nan"))
        std_imp = sig_data.get("std_improvement", float("nan"))
        p_corrected = sig_data.get("p_value_corrected", float("nan"))
        n_samples = sig_data.get("n_samples", 0)

        marker = format_significance_marker(p_corrected)

        if not np.isnan(mean_imp) and not np.isnan(std_imp):
            print(
                f"  {bin_name:12s} : Δ PSNR = {mean_imp:+.1f}±{std_imp:.1f} dB, "
                f"p_corrected = {p_corrected:.4f} {marker:3s} (n={n_samples})"
            )
        else:
            print(f"  {bin_name:12s} : N/A (n={n_samples})")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute stratified PSNR improvements from guidance comparison results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing guidance comparison results (e.g., results/guidance_comparison_sony)",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        required=True,
        choices=["sony", "fuji"],
        help="Sensor type (sony or fuji)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: results_dir/stratified_results.json)",
    )
    parser.add_argument(
        "--baseline_method",
        type=str,
        default="gaussian_x0",
        help="Baseline method name (default: gaussian_x0)",
    )
    parser.add_argument(
        "--proposed_method",
        type=str,
        default="pg_x0",
        help="Proposed method name (default: pg_x0)",
    )
    parser.add_argument(
        "--clean_method",
        type=str,
        default="long",
        help="Clean reference method name (default: long)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return 1

    # Determine sensor ranges
    sensor_cfg = get_sensor_config(args.sensor)
    black_level = sensor_cfg["black_level"]
    white_level = 16383.0  # 14-bit ADC max

    # Initialize stratified evaluator
    evaluator = StratifiedEvaluator(
        sensor_ranges={"min": black_level, "max": white_level}
    )

    scene_dirs = find_scene_directories(results_dir)

    if not scene_dirs:
        logger.error(f"No scene directories found in {results_dir}")
        return 1

    all_scene_results = []
    for scene_dir in scene_dirs:
        scene_result = compute_stratified_for_scene(
            scene_dir,
            evaluator,
            baseline_method=args.baseline_method,
            proposed_method=args.proposed_method,
            clean_method=args.clean_method,
        )

        if scene_result:
            all_scene_results.append(scene_result)

    if not all_scene_results:
        logger.error("No valid scene results found!")
        return 1

    # Aggregate results and compute statistical significance
    aggregated_results = evaluator.test_statistical_significance(
        all_scene_results,
        baseline_method=args.baseline_method,
        proposed_method=args.proposed_method,
        alpha=args.alpha,
        return_per_scene_improvements=True,
    )

    # Prepare full output
    output_data = {
        "results_dir": str(results_dir),
        "sensor": args.sensor,
        "black_level": black_level,
        "white_level": white_level,
        "baseline_method": args.baseline_method,
        "proposed_method": args.proposed_method,
        "clean_method": args.clean_method,
        "num_scenes": len(all_scene_results),
        "comparison_by_scene": aggregated_results["comparison_by_scene"],
        "statistical_significance": aggregated_results["statistical_significance"],
    }

    # Print results
    print_stratified_results(
        aggregated_results,
        all_scene_results,
        baseline_method=args.baseline_method,
        proposed_method=args.proposed_method,
    )

    # Save to file
    if args.output is None:
        output_path = results_dir / "stratified_results.json"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_safe(output_data, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
