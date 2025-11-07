#!/usr/bin/env python3
"""
Full integration test for stratified evaluation with realistic data structures.
This mimics how the sampling code will use stratified evaluation.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analysis.stratified_evaluation import (
    StratifiedEvaluator,
    format_stratified_results_table,
)


def test_full_workflow():
    """Test complete stratified evaluation workflow."""
    print("\n" + "=" * 70)
    print("FULL INTEGRATION TEST: Realistic Workflow")
    print("=" * 70)

    # Initialize evaluator (same as in sampling code)
    evaluator = StratifiedEvaluator(
        sensor_ranges={
            "min": 512,  # Sony a7S II black level
            "max": 16383,  # Sony a7S II white level
        }
    )
    print("✅ Created StratifiedEvaluator with Sony sensor ranges")

    # Simulate processing multiple tiles (like the main loop)
    num_tiles = 5
    all_results = []
    all_stratified_comparison = {}
    all_improvements = {}

    print(f"\nSimulating processing {num_tiles} tiles...")

    for tile_idx in range(num_tiles):
        tile_id = f"tile_{tile_idx:04d}"

        # Create synthetic restoration results (mimic posterior sampling output)
        clean = torch.rand(3, 256, 256) * 2 - 1  # [-1, 1]

        # Gaussian baseline (worse)
        gaussian_enhanced = clean + torch.randn_like(clean) * 0.15
        gaussian_enhanced = torch.clamp(gaussian_enhanced, -1, 1)

        # PG-guided (better, especially in dark areas)
        # Add signal-dependent improvement
        dark_mask = clean < -0.5  # Very dark pixels
        pg_enhanced = clean.clone()
        pg_enhanced[dark_mask] += (
            torch.randn_like(clean[dark_mask]) * 0.08
        )  # Less noise in dark
        pg_enhanced[~dark_mask] += (
            torch.randn_like(clean[~dark_mask]) * 0.15
        )  # Similar in bright
        pg_enhanced = torch.clamp(pg_enhanced, -1, 1)

        restoration_results = {"gaussian_x0": gaussian_enhanced, "pg_x0": pg_enhanced}

        # Compute stratified metrics
        stratified_comparison = evaluator.compare_methods_stratified(
            clean, restoration_results
        )

        # Compute improvements
        improvements = evaluator.compute_improvement_matrix(
            stratified_comparison["gaussian_x0"], stratified_comparison["pg_x0"]
        )

        # Store results
        all_stratified_comparison[tile_id] = stratified_comparison
        all_improvements[tile_id] = improvements

        result_info = {
            "tile_id": tile_id,
            "restoration_results": restoration_results,
            "stratified_metrics": stratified_comparison,
            "stratified_improvements": improvements,
        }
        all_results.append(result_info)

        # Log per-tile results
        print(f"\n  {tile_id}:")
        for bin_name, gain in improvements.items():
            if not np.isnan(gain):
                print(f"    {bin_name:12s}: {gain:+.2f} dB")

    print("\n" + "-" * 70)
    print("Aggregating results across all tiles...")
    print("-" * 70)

    # Statistical testing (same as main code)
    stratified_significance = evaluator.test_statistical_significance(
        all_results, baseline_method="gaussian_x0", proposed_method="pg_x0", alpha=0.05
    )

    # Print aggregated results
    print("\nAggregated Statistical Results:")
    for bin_name, stats in stratified_significance.items():
        if not np.isnan(stats["mean_improvement"]):
            sig_marker = "***" if stats.get("significant", False) else "   "
            print(
                f"  {bin_name:12s}: "
                f"Δ PSNR = {stats['mean_improvement']:+.2f}±{stats['std_improvement']:.2f} dB, "
                f"p_corrected = {stats['p_value_corrected']:.4f} {sig_marker} "
                f"(n={stats['n_samples']})"
            )

    # Test formatted table output
    print("\n" + "-" * 70)
    print("Formatted Results Table (for paper):")
    print("-" * 70)

    comparison = {}
    for tile_id, metrics in all_stratified_comparison.items():
        comparison[tile_id] = metrics

    # For paper, we'd aggregate across tiles
    # Here we just show the format
    table = format_stratified_results_table(
        {
            "gaussian_x0": all_stratified_comparison["tile_0000"]["gaussian_x0"],
            "pg_x0": all_stratified_comparison["tile_0000"]["pg_x0"],
        },
        method_names=["gaussian_x0", "pg_x0"],
    )
    print(table)

    # Test saving results (like main code does)
    print("\n" + "-" * 70)
    print("Testing JSON serialization...")
    print("-" * 70)

    stratified_results = {
        "comparison_by_tile": all_stratified_comparison,
        "improvements_by_tile": all_improvements,
        "statistical_significance": stratified_significance,
    }

    # Try to serialize to JSON
    try:
        json_str = json.dumps(stratified_results, indent=2, default=str)
        print(f"✅ Successfully serialized {len(json_str)} characters to JSON")

        # Verify we can deserialize
        deserialized = json.loads(json_str)
        print(f"✅ Successfully deserialized results")
        print(f"   - Tiles analyzed: {len(deserialized['comparison_by_tile'])}")
        print(f"   - ADC bins tested: {len(deserialized['statistical_significance'])}")

    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

    print("\n" + "=" * 70)
    print("✅ FULL INTEGRATION TEST PASSED")
    print("=" * 70)
    print("\nKey Verification Points:")
    print("  ✅ StratifiedEvaluator initialization")
    print("  ✅ Per-tile metric computation")
    print("  ✅ Statistical significance testing")
    print("  ✅ Formatted table output")
    print("  ✅ JSON serialization")
    print("  ✅ Deserialization and verification")
    print("\nReady for integration with sampling code!")

    return True


if __name__ == "__main__":
    try:
        success = test_full_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
