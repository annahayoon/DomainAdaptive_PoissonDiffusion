#!/usr/bin/env python
"""
Demonstration of synthetic data generation.

This script shows that the Poisson-Gaussian noise generation is working correctly
by creating a simple example and validating key properties.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add scripts to path
sys.path.append(str(Path(__file__).parent))
from generate_synthetic_data import SyntheticConfig, SyntheticDataGenerator


def demo_noise_statistics():
    """Demonstrate that noise statistics are correct."""
    print("=== SYNTHETIC DATA GENERATION DEMO ===\n")

    # Create simple configuration
    config = SyntheticConfig(image_size=64, save_plots=False)

    generator = SyntheticDataGenerator(config)

    # Test case: medium photon count with moderate read noise
    clean = np.ones((64, 64), dtype=np.float32) * 0.5
    photon_level = 100.0
    read_noise = 3.0
    background = 10.0

    print(f"Test parameters:")
    print(f"  Clean signal level: {clean[0,0]:.1f} (normalized)")
    print(f"  Photon level: {photon_level} electrons")
    print(f"  Read noise: {read_noise} electrons RMS")
    print(f"  Background: {background} electrons")
    print()

    # Generate multiple samples for statistics
    samples = []
    noise_params_list = []

    for i in range(10):
        noisy, noise_params = generator.add_poisson_gaussian_noise(
            clean, photon_level, read_noise, background
        )
        samples.append(noisy)
        noise_params_list.append(noise_params)

    samples = np.array(samples)

    # Compute statistics
    mean_signal = np.mean(samples)
    var_signal = np.var(samples)
    std_signal = np.sqrt(var_signal)

    # Expected statistics
    expected_mean = photon_level * 0.5 + background
    expected_var = expected_mean + read_noise**2
    expected_std = np.sqrt(expected_var)

    print("Statistical validation:")
    print(f"  Measured mean: {mean_signal:.1f} electrons")
    print(f"  Expected mean: {expected_mean:.1f} electrons")
    print(f"  Error: {abs(mean_signal - expected_mean):.1f} electrons")
    print()

    print(f"  Measured variance: {var_signal:.1f}")
    print(f"  Expected variance: {expected_var:.1f}")
    print(f"  Error: {abs(var_signal - expected_var):.1f}")
    print()

    print(f"  Measured std: {std_signal:.1f} electrons")
    print(f"  Expected std: {expected_std:.1f} electrons")
    print(f"  Error: {abs(std_signal - expected_std):.1f} electrons")
    print()

    # Check SNR
    avg_snr = np.mean([p["snr_db"] for p in noise_params_list])
    expected_snr = 10 * np.log10(expected_mean / expected_std)

    print(f"  Measured SNR: {avg_snr:.1f} dB")
    print(f"  Expected SNR: {expected_snr:.1f} dB")
    print(f"  Error: {abs(avg_snr - expected_snr):.1f} dB")
    print()

    # Validation
    mean_ok = abs(mean_signal - expected_mean) < 2.0
    var_ok = abs(var_signal - expected_var) / expected_var < 0.2  # 20% tolerance
    snr_ok = abs(avg_snr - expected_snr) < 1.0

    print("Validation results:")
    print(f"  Mean correct: {'✓' if mean_ok else '✗'}")
    print(f"  Variance correct: {'✓' if var_ok else '✗'}")
    print(f"  SNR correct: {'✓' if snr_ok else '✗'}")
    print()

    overall_ok = mean_ok and var_ok and snr_ok
    print(f"Overall validation: {'PASS ✓' if overall_ok else 'FAIL ✗'}")

    return overall_ok


def demo_different_regimes():
    """Demonstrate behavior in different photon regimes."""
    print("\n=== DIFFERENT PHOTON REGIMES ===\n")

    config = SyntheticConfig(save_plots=False)
    generator = SyntheticDataGenerator(config)

    # Test different photon levels
    test_cases = [
        (1.0, "Ultra-low"),
        (10.0, "Low"),
        (100.0, "Medium"),
        (1000.0, "High"),
    ]

    clean = np.ones((32, 32), dtype=np.float32) * 0.6
    read_noise = 2.0

    print("Regime analysis:")
    for photon_level, regime_name in test_cases:
        noisy, params = generator.add_poisson_gaussian_noise(
            clean, photon_level, read_noise
        )

        print(
            f"  {regime_name:10} ({photon_level:4.0f} photons): "
            f"SNR = {params['snr_db']:5.1f} dB, "
            f"χ² = {params['theoretical_variance']/params['mean_photons']:4.2f}"
        )

    print()


def demo_pattern_generation():
    """Demonstrate different pattern types."""
    print("=== PATTERN GENERATION ===\n")

    config = SyntheticConfig(save_plots=False)
    generator = SyntheticDataGenerator(config)

    patterns = ["constant", "gradient", "checkerboard", "gaussian_spots"]

    print("Pattern validation:")
    for pattern_type in patterns:
        pattern = generator.generate_pattern(pattern_type, 32)

        print(
            f"  {pattern_type:12}: "
            f"shape={pattern.shape}, "
            f"range=[{pattern.min():.2f}, {pattern.max():.2f}], "
            f"mean={pattern.mean():.2f}"
        )

    print()


def main():
    """Run all demonstrations."""
    print("POISSON-GAUSSIAN SYNTHETIC DATA GENERATION")
    print("=" * 50)

    # Demo 1: Basic noise statistics
    stats_ok = demo_noise_statistics()

    # Demo 2: Different regimes
    demo_different_regimes()

    # Demo 3: Pattern generation
    demo_pattern_generation()

    # Summary
    print("=== SUMMARY ===\n")
    if stats_ok:
        print("✓ Synthetic data generation is working correctly!")
        print("✓ Poisson-Gaussian noise model is implemented properly.")
        print("✓ Ready for use in physics validation (Phase 2.2.1).")
    else:
        print("✗ Issues detected in synthetic data generation.")
        print("✗ Review implementation before proceeding.")

    print("\nFiles created:")
    print("  - scripts/generate_synthetic_data.py")
    print("  - scripts/verify_physics.py")
    print("  - tests/test_synthetic_data.py")
    print("  - configs/synthetic_validation.yaml")

    return 0 if stats_ok else 1


if __name__ == "__main__":
    exit(main())
