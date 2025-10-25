#!/usr/bin/env python3
"""
Inference Examples for Cross-Domain Low-Light Enhancement

This script provides example commands for running posterior sampling with
sensor-specific optimized parameters for Sony and Fuji cameras.

Usage:
    python sample/inference_examples.py

Or run individual commands directly:
    python sample/sample_noisy_pt_lle_PGguidance.py [OPTIONS]
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print the description."""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False

def main():
    """Run inference examples with optimized parameters."""
    
    # Base paths
    model_path = "results/edm_pt_training_photography_20251008_032055/best_model.pkl"
    metadata_json = "dataset/processed/comprehensive_tiles_metadata.json"
    noisy_dir = "dataset/processed/pt_tiles/photography/noisy"
    clean_dir = "dataset/processed/pt_tiles/photography/clean"
    
    # Check if required files exist
    required_files = [model_path, metadata_json, noisy_dir, clean_dir]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure all required files exist before running inference.")
        return False
    
    print("🎯 Cross-Domain Low-Light Enhancement Inference Examples")
    print("=" * 80)
    print("This script demonstrates how to run posterior sampling with")
    print("sensor-specific optimized parameters for Sony and Fuji cameras.")
    print("\nOptimized Parameters:")
    print("📷 Sony: κ=0.7, σ_r=3.0, steps=20")
    print("   - Outperforms single-domain on PSNR (+2.69dB Gaussian, +2.87dB PG)")
    print("   - Outperforms single-domain on NIQE (-11.03 Gaussian, -9.34 PG)")
    print("📷 Fuji: κ=0.6, σ_r=3.5, steps=22")
    print("   - Outperforms single-domain on LPIPS (-0.0933 Gaussian, -0.0957 PG)")
    print("   - Outperforms single-domain on NIQE (-6.66 Gaussian, -5.60 PG)")
    
    # Common arguments
    base_args = [
        "python", "sample/sample_noisy_pt_lle_PGguidance.py",
        "--model_path", model_path,
        "--metadata_json", metadata_json,
        "--noisy_dir", noisy_dir,
        "--clean_dir", clean_dir,
        "--domain", "photography",
        "--num_examples", "5",
        "--use_sensor_calibration",
        "--sensor_name", "sony_a7s",
        "--s", "15871",
        "--sigma_r", "5.0",
        "--kappa", "0.5",
        "--preserve_details",
        "--adaptive_strength",
        "--edge_aware"
    ]
    
    # Example 1: Sony tiles with optimized parameters
    sony_cmd = base_args + [
        "--output_dir", "results/inference_sony_optimized",
        "--sensor_filter", "sony",
        "--cross_domain_kappa", "0.7",
        "--cross_domain_sigma_r", "3.0",
        "--cross_domain_num_steps", "20"
    ]
    
    # Example 2: Fuji tiles with optimized parameters
    fuji_cmd = base_args + [
        "--output_dir", "results/inference_fuji_optimized",
        "--sensor_filter", "fuji",
        "--cross_domain_kappa", "0.6",
        "--cross_domain_sigma_r", "3.5",
        "--cross_domain_num_steps", "22",
        "--sensor_name", "fuji"
    ]
    
    # Example 3: Mixed sensors with default (Sony) parameters
    mixed_cmd = base_args + [
        "--output_dir", "results/inference_mixed_default",
        "--cross_domain_kappa", "0.7",
        "--cross_domain_sigma_r", "3.0",
        "--cross_domain_num_steps", "20"
    ]
    
    # Example 4: Sony tiles with single-domain only (baseline)
    sony_single_cmd = base_args + [
        "--output_dir", "results/inference_sony_single_domain",
        "--sensor_filter", "sony",
        "--num_steps", "18"
    ]
    
    # Example 5: Fuji tiles with single-domain only (baseline)
    fuji_single_cmd = base_args + [
        "--output_dir", "results/inference_fuji_single_domain",
        "--sensor_filter", "fuji",
        "--num_steps", "18"
    ]
    
    # Run examples
    examples = [
        (sony_cmd, "Sony Tiles with Optimized Cross-Domain Parameters"),
        (fuji_cmd, "Fuji Tiles with Optimized Cross-Domain Parameters"),
        (mixed_cmd, "Mixed Sensors with Default (Sony) Parameters"),
        (sony_single_cmd, "Sony Tiles - Single Domain Baseline"),
        (fuji_single_cmd, "Fuji Tiles - Single Domain Baseline")
    ]
    
    print(f"\n🎯 Running {len(examples)} inference examples...")
    
    success_count = 0
    for cmd, description in examples:
        if run_command(cmd, description):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"🎉 Inference Examples Completed!")
    print(f"✅ Successful: {success_count}/{len(examples)}")
    print(f"❌ Failed: {len(examples) - success_count}/{len(examples)}")
    print(f"{'='*80}")
    
    if success_count == len(examples):
        print("\n📊 All examples completed successfully!")
        print("Check the results/ directory for output visualizations and metrics.")
    else:
        print(f"\n⚠️  {len(examples) - success_count} examples failed.")
        print("Check the error messages above for details.")
    
    return success_count == len(examples)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
