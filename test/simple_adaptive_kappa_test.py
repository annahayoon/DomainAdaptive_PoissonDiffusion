#!/usr/bin/env python3
"""
Simple Adaptive Kappa Test

This script tests the adaptive kappa functionality by running
a minimal inference test on a few photography tiles.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_simple_test():
    """
    Run a simple test with minimal arguments.
    """
    print("=" * 80)
    print("SIMPLE ADAPTIVE KAPPA TEST")
    print("=" * 80)

    # Create output directory
    output_dir = "results/simple_adaptive_kappa_test"
    os.makedirs(output_dir, exist_ok=True)

    # Test 1: Constant kappa
    print("\n1. Testing constant kappa (baseline)...")
    cmd_constant = [
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
        f"{output_dir}/constant_kappa",
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
        "--num_examples",
        "5",
    ]

    print(f"Command: {' '.join(cmd_constant)}")

    try:
        result = subprocess.run(
            cmd_constant, capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("✅ Constant kappa test completed")
        else:
            print(f"❌ Constant kappa test failed: {result.stderr[:200]}...")
            return False
    except Exception as e:
        print(f"❌ Constant kappa test failed: {e}")
        return False

    # Test 2: Adaptive kappa
    print("\n2. Testing adaptive kappa...")
    cmd_adaptive = [
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
        f"{output_dir}/adaptive_kappa",
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
        "--num_examples",
        "5",
        "--use_adaptive_kappa",
        "--min_kappa",
        "0.05",
        "--max_kappa",
        "2.0",
        "--signal_threshold",
        "50.0",
    ]

    print(f"Command: {' '.join(cmd_adaptive)}")

    try:
        result = subprocess.run(
            cmd_adaptive, capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("✅ Adaptive kappa test completed")
        else:
            print(f"❌ Adaptive kappa test failed: {result.stderr[:200]}...")
            return False
    except Exception as e:
        print(f"❌ Adaptive kappa test failed: {e}")
        return False

    # Analyze results
    print("\n3. Analyzing results...")
    analyze_simple_results(output_dir)

    return True


def analyze_simple_results(output_dir):
    """
    Analyze the simple test results.
    """
    print("\n" + "=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)

    # Look for results files
    constant_dir = Path(output_dir) / "constant_kappa"
    adaptive_dir = Path(output_dir) / "adaptive_kappa"

    print(f"Constant kappa results: {constant_dir}")
    print(f"Adaptive kappa results: {adaptive_dir}")

    # Check if directories exist
    if not constant_dir.exists():
        print("❌ Constant kappa results directory not found")
        return

    if not adaptive_dir.exists():
        print("❌ Adaptive kappa results directory not found")
        return

    # List files in each directory
    print(f"\nConstant kappa files:")
    for file in constant_dir.rglob("*"):
        if file.is_file():
            print(f"  {file.relative_to(constant_dir)}")

    print(f"\nAdaptive kappa files:")
    for file in adaptive_dir.rglob("*"):
        if file.is_file():
            print(f"  {file.relative_to(adaptive_dir)}")

    # Look for CSV results
    constant_csv = None
    adaptive_csv = None

    for file in constant_dir.rglob("*.csv"):
        if "results" in file.name.lower():
            constant_csv = file
            break

    for file in adaptive_dir.rglob("*.csv"):
        if "results" in file.name.lower():
            adaptive_csv = file
            break

    if constant_csv and adaptive_csv:
        print(f"\nFound results files:")
        print(f"  Constant: {constant_csv}")
        print(f"  Adaptive: {adaptive_csv}")

        # Try to load and compare
        try:
            import pandas as pd

            df_const = pd.read_csv(constant_csv)
            df_adapt = pd.read_csv(adaptive_csv)

            print(f"\nConstant kappa results: {len(df_const)} rows")
            print(f"Adaptive kappa results: {len(df_adapt)} rows")

            # Show sample results
            if len(df_const) > 0:
                print(f"\nConstant kappa sample:")
                print(df_const.head())

            if len(df_adapt) > 0:
                print(f"\nAdaptive kappa sample:")
                print(df_adapt.head())

        except ImportError:
            print("❌ pandas not available for detailed analysis")
        except Exception as e:
            print(f"❌ Error analyzing results: {e}")
    else:
        print("❌ No CSV results files found")

    print(f"\n✅ Test completed successfully!")
    print(f"📁 Results saved in: {output_dir}")


def main():
    """
    Main test function.
    """
    print("Testing adaptive kappa functionality...")

    success = run_simple_test()

    if success:
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print("✅ Both constant and adaptive kappa tests completed")
        print("📊 Check the results directories for detailed output")
        print("🔍 Look for CSV files with metrics comparison")
        print("\nNext steps:")
        print("1. Compare the metrics in the CSV files")
        print("2. Look for improvements in low-signal regions")
        print("3. Check if adaptive kappa shows better stability")
    else:
        print("\n" + "=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        print("❌ One or both tests failed")
        print("🔧 Check the error messages above for debugging")


if __name__ == "__main__":
    main()
