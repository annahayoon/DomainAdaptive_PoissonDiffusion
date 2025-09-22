#!/usr/bin/env python3
"""
Run photography preprocessing for SID dataset.
This script processes the Sony SID dataset for low-light photography.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from preprocessing.photography_processor import PhotographyProcessor


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("PHOTOGRAPHY PREPROCESSING - SID DATASET")
    print("=" * 60)

    # Configuration
    raw_data_root = "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/SID"
    output_root = "/home/jilab/Jae/data/preprocessed_photography"

    print(f"Raw data root: {raw_data_root}")
    print(f"Output root: {output_root}")

    # Check if raw data exists
    if not Path(raw_data_root).exists():
        print(f"❌ ERROR: Raw data directory not found: {raw_data_root}")
        return 1

    # Create output directory
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # Initialize processor with optimized settings
    print("\n📋 Initializing Photography Processor...")
    processor = PhotographyProcessor(
        raw_data_root=raw_data_root,
        output_root=output_root,
        num_tiles_per_scene=50,  # Good balance of data and speed
        tile_size=128,  # Standard tile size
        sample_size_for_estimation=20,  # Reasonable sample for noise estimation
        split_seed=42,  # Reproducible splits
        use_parallel=True,  # Enable parallel processing
        n_workers=4,  # Use 4 workers for good performance
    )

    print("✅ Processor initialized successfully")

    # Discover scenes
    print("\n🔍 Discovering SID scenes...")
    start_time = time.time()
    scene_pairs = processor.discover_sid_scenes()
    discovery_time = time.time() - start_time

    print(f"✅ Found {len(scene_pairs)} scene pairs in {discovery_time:.2f}s")

    if len(scene_pairs) == 0:
        print("❌ ERROR: No scene pairs found. Check data structure.")
        return 1

    # Show some examples
    print("\n📋 Example scene pairs:")
    for i, (noisy_path, clean_path) in enumerate(scene_pairs[:3]):
        print(f"  {i+1}. Noisy: {noisy_path.name}")
        print(f"     Clean: {clean_path.name}")

    # Estimate noise parameters
    print(f"\n🔬 Estimating noise parameters from {processor.sample_size} scenes...")
    start_time = time.time()
    gain, read_noise = processor.estimate_noise_parameters(scene_pairs)
    estimation_time = time.time() - start_time

    print(f"✅ Noise estimation completed in {estimation_time:.2f}s")
    print(f"   📊 Estimated gain: {gain:.3f}")
    print(f"   📊 Estimated read noise: {read_noise:.3f}")

    # Compute normalization scales
    print("\n📏 Computing normalization scales...")
    start_time = time.time()

    print("   Computing noisy scale...")
    noisy_scale = processor.compute_normalization_scale(
        scene_pairs, gain, use_noisy=True
    )

    print("   Computing clean scale...")
    clean_scale = processor.compute_normalization_scale(
        scene_pairs, gain, use_noisy=False
    )

    scale_time = time.time() - start_time

    print(f"✅ Scale computation completed in {scale_time:.2f}s")
    print(f"   📊 Noisy scale (99.9th percentile): {noisy_scale:.1f}")
    print(f"   📊 Clean scale (99.9th percentile): {clean_scale:.1f}")

    # Process all scenes
    print(f"\n🚀 Processing all {len(scene_pairs)} scenes...")
    print("   This may take a while depending on dataset size...")

    start_time = time.time()
    processor.process_all_scenes(
        scene_pairs=scene_pairs,
        gain=gain,
        read_noise=read_noise,
        noisy_scale=noisy_scale,
        clean_scale=clean_scale,
    )
    processing_time = time.time() - start_time

    print(
        f"✅ All scenes processed in {processing_time:.2f}s ({processing_time/60:.1f} minutes)"
    )

    # Final summary
    total_time = discovery_time + estimation_time + scale_time + processing_time
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE! 🎉")
    print("=" * 60)
    print(f"📊 Total scenes processed: {len(scene_pairs)}")
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"📁 Output directory: {output_root}")
    print(f"🔧 Gain: {gain:.3f}, Read noise: {read_noise:.3f}")
    print(f"📏 Scales - Noisy: {noisy_scale:.1f}, Clean: {clean_scale:.1f}")
    print("\n✅ Photography preprocessing completed successfully!")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: Preprocessing failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
