#!/usr/bin/env python3
"""
Optimized cross-domain parameter optimization for all domains.
Runs each domain separately with 20 tiles per parameter combination for faster completion.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_domain_optimization(domain_sensor, num_tiles=20):
    """Run cross-domain optimization for a specific domain/sensor combination."""

    # Domain configurations
    configs = {
        "photography_sony": {
            "model_path": "results/edm_pt_training_multi_domain_20251010_182807/best_model.pkl",
            "metadata_json": "dataset/processed/comprehensive_tiles_metadata.json",
            "noisy_dir": "dataset/processed/pt_tiles/photography/noisy",
            "clean_dir": "dataset/processed/pt_tiles/photography/clean",
            "sensor_filter": "sony",
            "num_examples": num_tiles
        },
        "photography_fuji": {
            "model_path": "results/edm_pt_training_multi_domain_20251010_182807/best_model.pkl",
            "metadata_json": "dataset/processed/comprehensive_tiles_metadata.json",
            "noisy_dir": "dataset/processed/pt_tiles/photography/noisy",
            "clean_dir": "dataset/processed/pt_tiles/photography/clean",
            "sensor_filter": "fuji",
            "num_examples": num_tiles
        },
        "astronomy": {
            "model_path": "results/edm_pt_training_multi_domain_20251010_182807/best_model.pkl",
            "metadata_json": "dataset/processed/comprehensive_tiles_metadata.json",
            "noisy_dir": "dataset/processed/pt_tiles/astronomy/noisy",
            "clean_dir": "dataset/processed/pt_tiles/astronomy/clean",
            "num_examples": num_tiles
        },
        "microscopy": {
            "model_path": "results/edm_pt_training_microscopy_20251008_044631/best_model.pkl",
            "metadata_json": "dataset/processed/comprehensive_tiles_metadata.json",
            "noisy_dir": "dataset/processed/pt_tiles/microscopy/noisy",
            "clean_dir": "dataset/processed/pt_tiles/microscopy/clean",
            "num_examples": num_tiles
        }
    }

    if domain_sensor not in configs:
        print(f"❌ Unknown domain/sensor combination: {domain_sensor}")
        return False

    config = configs[domain_sensor]

    # Build command
    cmd = [
        "python", "sample/cross_domain_optimization.py",
        "--model_path", config["model_path"],
        "--metadata_json", config["metadata_json"],
        "--noisy_dir", config["noisy_dir"],
        "--clean_dir", config["clean_dir"],
        "--num_examples", str(config["num_examples"])
    ]

    # Add sensor filter for photography domains
    if "photography" in domain_sensor and "sensor_filter" in config:
        cmd.extend(["--sensor_filter", config["sensor_filter"]])

    print(f"\n🚀 Running cross-domain optimization for: {domain_sensor.upper()}")
    print(f"   Tiles per combination: {num_tiles}")
    print(f"   Command: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        end_time = time.time()

        print(f"   ✅ {domain_sensor} optimization completed in {end_time - start_time:.1f} seconds")

        # Check if results were saved
        results_dir = Path("results/cross_domain_optimization")
        if results_dir.exists():
            print(f"   📊 Results saved to: {results_dir}")
        else:
            print(f"   ⚠️  Results directory not found: {results_dir}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"   ❌ {domain_sensor} optimization failed with exit code {e.returncode}")
        print(f"   Error: {e.stderr}")
        return False
    except subprocess.TimeoutExpired as e:
        print(f"   ❌ {domain_sensor} optimization timed out after {e.timeout} seconds")
        return False

def main():
    """Run cross-domain optimization for all domains."""

    print("🎯 Optimized Cross-Domain Parameter Optimization")
    print("=" * 80)
    print("Running each domain separately with 20 tiles per parameter combination")
    print("This provides statistically significant results while completing faster")
    print("=" * 80)

    domains = [
        "photography_sony",
        "photography_fuji",
        "astronomy",
        "microscopy"
    ]

    success_count = 0
    total_count = len(domains)

    for domain in domains:
        success = run_domain_optimization(domain, num_tiles=20)
        if success:
            success_count += 1

        # Small delay between domains to avoid overwhelming the system
        time.sleep(2)

    print("\n🎉 Cross-Domain Optimization Complete!")
    print(f"✅ Successful domains: {success_count}/{total_count}")
    print(f"❌ Failed domains: {total_count - success_count}/{total_count}")

    if success_count == total_count:
        print("\n📋 Next steps:")
        print("1. Run: python sample/analyze_optimization_results.py --results_dir results/cross_domain_optimization")
        print("2. Update INFERENCE_GUIDE.md with optimal parameters")
        print("3. Test optimal parameters on additional validation tiles")
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
