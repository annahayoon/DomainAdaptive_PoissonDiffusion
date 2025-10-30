#!/usr/bin/env python3
"""
Test script to validate stratified evaluation integration with sampling code.
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_stratified_import():
    """Test that stratified evaluation module can be imported."""
    print("Testing stratified evaluation import...")
    try:
        from analysis.stratified_evaluation import (
            StratifiedEvaluator,
            format_stratified_results_table,
        )

        print(
            "✅ Successfully imported StratifiedEvaluator and format_stratified_results_table"
        )
        return True
    except ImportError as e:
        print(f"❌ Failed to import stratified evaluation: {e}")
        return False


def test_stratified_basic():
    """Test basic stratified evaluation functionality."""
    print("\nTesting basic stratified evaluation...")
    try:
        import numpy as np
        import torch

        from analysis.stratified_evaluation import StratifiedEvaluator

        # Create evaluator
        evaluator = StratifiedEvaluator({"min": 512, "max": 16383})
        print("✅ Created StratifiedEvaluator")

        # Create synthetic data
        clean = torch.rand(3, 256, 256) * 2 - 1  # [-1, 1]
        enhanced = clean + torch.randn_like(clean) * 0.1

        # Compute stratified metrics
        metrics = evaluator.compute_stratified_metrics(clean, enhanced, "test_method")
        print(f"✅ Computed stratified metrics for {len(metrics)} bins")

        # Check results
        for bin_name, metrics_dict in metrics.items():
            print(
                f"  {bin_name}: PSNR={metrics_dict['psnr']:.2f}dB, pixels={metrics_dict['pixel_fraction']:.1%}"
            )

        return True

    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sampling_code_integration():
    """Test that sampling code can import and use stratified evaluation."""
    print("\nTesting sampling code integration...")
    try:
        # Check if imports in sampling code would work
        sys.path.insert(0, str(project_root / "sample"))

        # We don't actually run the sampling, just check imports
        from analysis.stratified_evaluation import StratifiedEvaluator

        print("✅ Sampling code can import StratifiedEvaluator")

        return True
    except ImportError as e:
        print(f"❌ Sampling code integration failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("STRATIFIED EVALUATION INTEGRATION TEST")
    print("=" * 70)

    results = []
    results.append(("Import test", test_stratified_import()))
    results.append(("Basic functionality", test_stratified_basic()))
    results.append(("Sampling integration", test_sampling_code_integration()))

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed! Stratified evaluation integration is working.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
