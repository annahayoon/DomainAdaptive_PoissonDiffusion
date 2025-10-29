#!/usr/bin/env python3
"""
Adaptive Kappa Comparison Test

This script runs both constant kappa and adaptive kappa versions
on the same test tiles to compare performance.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_inference_test(test_name, use_adaptive_kappa=False, max_tiles=10):
    """
    Run inference test with specified parameters.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/adaptive_kappa_comparison_{timestamp}_{test_name}"

    # Base command
    cmd = [
        "python",
        "sample/sample_noisy_pt_lle_PGguidance_adaptivekappa.py",
        "--model_path",
        "results/edm_pt_training_photography_20251008_032055/best_model.pkl",
        "--metadata_json",
        "dataset/processed/comprehensive_tiles_metadata.json",
        "--noisy_dir",
        "dataset/processed/pt_tiles/photography/noisy",
        "--clean_dir",
        "dataset/processed/pt_tiles/photography/clean",
        "--output_dir",
        output_dir,
        "--domain",
        "photography",
        "--kappa",
        "0.8",
        "--sigma_r",
        "4.0",
        "--num_steps",
        "15",
        "--run_methods",
        "gaussian_x0",
        "pg_x0",
        "--use_sensor_calibration",
        "--preserve_details",
        "--edge_aware",
        "--max_tiles",
        str(max_tiles),
        "--log_level",
        "INFO",
    ]

    # Add adaptive kappa parameters if enabled
    if use_adaptive_kappa:
        cmd.extend(
            [
                "--use_adaptive_kappa",
                "--min_kappa",
                "0.05",
                "--max_kappa",
                "2.0",
                "--signal_threshold",
                "50.0",
            ]
        )

    print(f"Running {test_name} test...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800
        )  # 30 min timeout

        if result.returncode == 0:
            print(f"✅ {test_name} test completed successfully")
            return output_dir, True
        else:
            print(f"❌ {test_name} test failed")
            print(f"Error: {result.stderr}")
            return output_dir, False

    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} test timed out")
        return output_dir, False
    except Exception as e:
        print(f"💥 {test_name} test failed with exception: {e}")
        return output_dir, False


def analyze_results(constant_dir, adaptive_dir):
    """
    Analyze and compare results from both tests.
    """
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)

    # Look for results files
    constant_results = None
    adaptive_results = None

    # Find results files
    for results_dir in [constant_dir, adaptive_dir]:
        if os.path.exists(results_dir):
            # Look for CSV files with results
            for file in Path(results_dir).rglob("*.csv"):
                if "results" in file.name.lower():
                    if "constant" in str(results_dir):
                        constant_results = file
                    else:
                        adaptive_results = file
                    break

    if not constant_results or not adaptive_results:
        print("❌ Could not find results files")
        return

    print(f"Constant kappa results: {constant_results}")
    print(f"Adaptive kappa results: {adaptive_results}")

    # Load and compare results
    try:
        import pandas as pd

        # Load results
        df_constant = pd.read_csv(constant_results)
        df_adaptive = pd.read_csv(adaptive_results)

        # Filter for PG guidance results
        pg_constant = df_constant[df_constant["method"] == "pg_x0"]
        pg_adaptive = df_adaptive[df_adaptive["method"] == "pg_x0"]

        if len(pg_constant) == 0 or len(pg_adaptive) == 0:
            print("❌ No PG guidance results found")
            return

        print(f"\nConstant kappa PG results: {len(pg_constant)} tiles")
        print(f"Adaptive kappa PG results: {len(pg_adaptive)} tiles")

        # Compare metrics
        metrics = ["ssim", "psnr", "lpips", "niqe"]

        print("\n" + "=" * 80)
        print("METRIC COMPARISON")
        print("=" * 80)

        for metric in metrics:
            if metric in pg_constant.columns and metric in pg_adaptive.columns:
                const_mean = pg_constant[metric].mean()
                adapt_mean = pg_adaptive[metric].mean()
                diff = adapt_mean - const_mean

                if metric in ["ssim", "psnr"]:
                    # Higher is better
                    improvement = "✅" if diff > 0 else "❌"
                else:
                    # Lower is better
                    improvement = "✅" if diff < 0 else "❌"

                print(
                    f"{metric.upper():6s}: Constant {const_mean:.4f} vs Adaptive {adapt_mean:.4f} (Δ={diff:+.4f}) {improvement}"
                )

        # Save comparison results
        comparison_results = {
            "test_timestamp": datetime.now().isoformat(),
            "constant_kappa_results": str(constant_results),
            "adaptive_kappa_results": str(adaptive_results),
            "metric_comparison": {},
        }

        for metric in metrics:
            if metric in pg_constant.columns and metric in pg_adaptive.columns:
                comparison_results["metric_comparison"][metric] = {
                    "constant_mean": float(pg_constant[metric].mean()),
                    "adaptive_mean": float(pg_adaptive[metric].mean()),
                    "difference": float(
                        pg_adaptive[metric].mean() - pg_constant[metric].mean()
                    ),
                }

        with open("adaptive_kappa_comparison_results.json", "w") as f:
            json.dump(comparison_results, f, indent=2)

        print(f"\nComparison results saved to adaptive_kappa_comparison_results.json")

    except ImportError:
        print("❌ pandas not available for analysis")
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")


def main():
    """
    Main comparison test.
    """
    print("=" * 80)
    print("ADAPTIVE KAPPA COMPARISON TEST")
    print("=" * 80)

    # Test parameters
    max_tiles = 10  # Small test for quick results

    print(f"Testing on {max_tiles} photography tiles...")
    print("This will run both constant and adaptive kappa versions")

    # Run constant kappa test
    print("\n" + "=" * 50)
    print("RUNNING CONSTANT KAPPA TEST")
    print("=" * 50)

    constant_dir, constant_success = run_inference_test(
        "constant_kappa", use_adaptive_kappa=False, max_tiles=max_tiles
    )

    # Run adaptive kappa test
    print("\n" + "=" * 50)
    print("RUNNING ADAPTIVE KAPPA TEST")
    print("=" * 50)

    adaptive_dir, adaptive_success = run_inference_test(
        "adaptive_kappa", use_adaptive_kappa=True, max_tiles=max_tiles
    )

    # Analyze results
    if constant_success and adaptive_success:
        print("\n" + "=" * 50)
        print("ANALYZING RESULTS")
        print("=" * 50)

        analyze_results(constant_dir, adaptive_dir)

        print("\n" + "=" * 80)
        print("TEST COMPLETED")
        print("=" * 80)
        print("✅ Both tests completed successfully")
        print("📊 Check the comparison results above")
        print("📁 Results saved in:")
        print(f"   - Constant kappa: {constant_dir}")
        print(f"   - Adaptive kappa: {adaptive_dir}")

    else:
        print("\n" + "=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        print("❌ One or both tests failed")
        print(f"Constant kappa: {'✅' if constant_success else '❌'}")
        print(f"Adaptive kappa: {'✅' if adaptive_success else '❌'}")


if __name__ == "__main__":
    main()
