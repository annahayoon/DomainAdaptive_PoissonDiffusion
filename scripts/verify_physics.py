#!/usr/bin/env python
"""
Physics validation script using synthetic data.

This script validates that our Poisson-Gaussian noise model is statistically
correct by generating synthetic data and checking key properties:
1. Chi-squared statistic ≈ 1
2. Variance matches theoretical prediction
3. SNR calculations are accurate
4. Low-photon regime behaves correctly

This is critical for Phase 2.2.1 validation checkpoint.

Usage:
    python scripts/verify_physics.py
    python scripts/verify_physics.py --config configs/synthetic_validation.yaml
    python scripts/verify_physics.py --quick  # Fast validation with reduced data
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Import synthetic data generator
from generate_synthetic_data import SyntheticConfig, SyntheticDataGenerator


@dataclass
class ValidationResults:
    """Results from physics validation."""

    chi2_mean: float
    chi2_std: float
    chi2_in_range: float  # Fraction in acceptable range

    variance_error_mean: float
    variance_error_std: float
    variance_error_max: float

    snr_error_mean: float
    snr_error_std: float

    low_photon_valid: bool
    high_photon_valid: bool

    overall_pass: bool


class PhysicsValidator:
    """
    Validate physics correctness of Poisson-Gaussian noise model.
    """

    def __init__(self, config: SyntheticConfig):
        """
        Initialize validator.

        Args:
            config: Configuration for synthetic data generation
        """
        self.config = config
        self.generator = SyntheticDataGenerator(config)

        # Validation thresholds
        self.chi2_range = (0.8, 1.2)
        self.variance_tolerance = 0.15  # 15%
        self.snr_tolerance = 1.0  # 1 dB

    def validate_noise_statistics(self, num_trials: int = 50) -> ValidationResults:
        """
        Validate noise statistics across multiple trials.

        Args:
            num_trials: Number of independent trials for statistical validation

        Returns:
            Validation results
        """
        print(f"Running physics validation with {num_trials} trials...")

        chi2_values = []
        variance_errors = []
        snr_errors = []

        # Test different regimes
        test_conditions = [
            (1.0, 0.5),  # Ultra-low photons
            (10.0, 1.0),  # Low photons
            (100.0, 2.0),  # Medium photons
            (1000.0, 5.0),  # High photons
        ]

        for photon_level, read_noise in test_conditions:
            print(f"Testing: {photon_level} photons, {read_noise} e⁻ read noise")

            for trial in range(num_trials // len(test_conditions)):
                # Generate clean pattern
                clean = self.generator.generate_pattern("constant", 64)

                # Add noise
                noisy, params = self.generator.add_poisson_gaussian_noise(
                    clean, photon_level, read_noise, self.config.background_level
                )

                # Validate chi-squared
                chi2 = self._compute_chi2(clean, noisy, photon_level, read_noise)
                chi2_values.append(chi2)

                # Validate variance
                variance_error = self._compute_variance_error(
                    clean, noisy, photon_level, read_noise
                )
                variance_errors.append(variance_error)

                # Validate SNR
                snr_error = self._compute_snr_error(params, photon_level, read_noise)
                snr_errors.append(snr_error)

        # Compute summary statistics
        chi2_values = np.array(chi2_values)
        variance_errors = np.array(variance_errors)
        snr_errors = np.array(snr_errors)

        # Check fraction of chi2 values in acceptable range
        chi2_in_range = np.mean(
            (chi2_values >= self.chi2_range[0]) & (chi2_values <= self.chi2_range[1])
        )

        # Check regime-specific behavior
        low_photon_valid = self._validate_low_photon_regime()
        high_photon_valid = self._validate_high_photon_regime()

        # Overall pass/fail
        overall_pass = (
            chi2_in_range > 0.8
            and np.abs(np.mean(variance_errors))  # 80% of samples should have good chi2
            < self.variance_tolerance
            and np.abs(np.mean(snr_errors)) < self.snr_tolerance
            and low_photon_valid
            and high_photon_valid
        )

        results = ValidationResults(
            chi2_mean=float(np.mean(chi2_values)),
            chi2_std=float(np.std(chi2_values)),
            chi2_in_range=float(chi2_in_range),
            variance_error_mean=float(np.mean(variance_errors)),
            variance_error_std=float(np.std(variance_errors)),
            variance_error_max=float(np.max(np.abs(variance_errors))),
            snr_error_mean=float(np.mean(snr_errors)),
            snr_error_std=float(np.std(snr_errors)),
            low_photon_valid=low_photon_valid,
            high_photon_valid=high_photon_valid,
            overall_pass=overall_pass,
        )

        return results

    def _compute_chi2(
        self,
        clean: np.ndarray,
        noisy: np.ndarray,
        photon_level: float,
        read_noise: float,
    ) -> float:
        """Compute chi-squared statistic per pixel."""
        lambda_e = photon_level * clean + self.config.background_level
        variance = lambda_e + read_noise**2

        chi2_per_pixel = np.mean((noisy - lambda_e) ** 2 / variance)
        return chi2_per_pixel

    def _compute_variance_error(
        self,
        clean: np.ndarray,
        noisy: np.ndarray,
        photon_level: float,
        read_noise: float,
    ) -> float:
        """Compute relative error in variance prediction."""
        lambda_e = photon_level * clean + self.config.background_level
        theoretical_var = np.mean(lambda_e + read_noise**2)

        residuals = noisy - lambda_e
        empirical_var = np.var(residuals)

        relative_error = (empirical_var - theoretical_var) / theoretical_var
        return relative_error

    def _compute_snr_error(
        self, params: Dict, photon_level: float, read_noise: float
    ) -> float:
        """Compute error in SNR calculation."""
        # Theoretical SNR
        signal = params["mean_photons"]
        noise_var = signal + read_noise**2
        theoretical_snr = 10 * np.log10(signal / np.sqrt(noise_var))

        # Reported SNR
        reported_snr = params["snr_db"]

        return reported_snr - theoretical_snr

    def _validate_low_photon_regime(self) -> bool:
        """Validate behavior in low-photon regime (<10 photons)."""
        print("Validating low-photon regime...")

        # Test with very low photon counts
        clean = np.ones((64, 64), dtype=np.float32) * 0.5
        photon_level = 2.0  # Very low
        read_noise = 1.0

        # Generate multiple samples
        chi2_values = []
        for _ in range(20):
            noisy, _ = self.generator.add_poisson_gaussian_noise(
                clean, photon_level, read_noise
            )
            chi2 = self._compute_chi2(clean, noisy, photon_level, read_noise)
            chi2_values.append(chi2)

        # In low-photon regime, discrete nature is important
        # Chi2 should still be reasonable (within factor of 2)
        mean_chi2 = np.mean(chi2_values)
        return 0.5 < mean_chi2 < 2.0

    def _validate_high_photon_regime(self) -> bool:
        """Validate behavior in high-photon regime (>1000 photons)."""
        print("Validating high-photon regime...")

        clean = np.ones((64, 64), dtype=np.float32) * 0.7
        photon_level = 5000.0  # High
        read_noise = 3.0

        # Generate samples
        chi2_values = []
        for _ in range(10):
            noisy, _ = self.generator.add_poisson_gaussian_noise(
                clean, photon_level, read_noise
            )
            chi2 = self._compute_chi2(clean, noisy, photon_level, read_noise)
            chi2_values.append(chi2)

        # In high-photon regime, should be very close to Gaussian
        # Chi2 should be very close to 1
        mean_chi2 = np.mean(chi2_values)
        return 0.9 < mean_chi2 < 1.1

    def validate_theoretical_predictions(self) -> Dict[str, bool]:
        """
        Validate key theoretical predictions.

        Returns:
            Dictionary of test results
        """
        print("Validating theoretical predictions...")

        results = {}

        # Test 1: Poisson variance property (var = mean for Poisson component)
        results["poisson_variance"] = self._test_poisson_variance()

        # Test 2: Additive noise property
        results["additive_noise"] = self._test_additive_noise()

        # Test 3: SNR scaling with photon count
        results["snr_scaling"] = self._test_snr_scaling()

        # Test 4: Read noise independence
        results["read_noise_independence"] = self._test_read_noise_independence()

        return results

    def _test_poisson_variance(self) -> bool:
        """Test that Poisson component has variance equal to mean."""
        clean = np.ones((100, 100), dtype=np.float32) * 0.6
        photon_level = 100.0

        # Generate many samples with no read noise
        samples = []
        for _ in range(50):
            noisy, _ = self.generator.add_poisson_gaussian_noise(
                clean, photon_level, read_noise=0.0
            )
            samples.append(noisy)

        samples = np.array(samples)
        mean_signal = np.mean(samples)
        var_signal = np.var(samples)

        # For pure Poisson, variance should equal mean
        relative_error = abs(var_signal - mean_signal) / mean_signal
        return relative_error < 0.1  # 10% tolerance

    def _test_additive_noise(self) -> bool:
        """Test that read noise adds in quadrature."""
        clean = np.ones((64, 64), dtype=np.float32) * 0.5
        photon_level = 200.0

        # Test different read noise levels
        read_noises = [0.0, 2.0, 5.0, 10.0]
        variances = []

        for read_noise in read_noises:
            samples = []
            for _ in range(20):
                noisy, _ = self.generator.add_poisson_gaussian_noise(
                    clean, photon_level, read_noise
                )
                samples.append(noisy)

            var_empirical = np.var(samples)
            variances.append(var_empirical)

        # Check that variance increases quadratically with read noise
        # var_total = var_poisson + read_noise^2
        var_poisson = variances[0]  # read_noise = 0

        for i, read_noise in enumerate(read_noises[1:], 1):
            expected_var = var_poisson + read_noise**2
            actual_var = variances[i]
            relative_error = abs(actual_var - expected_var) / expected_var

            if relative_error > 0.2:  # 20% tolerance
                return False

        return True

    def _test_snr_scaling(self) -> bool:
        """Test SNR scaling with photon count."""
        clean = np.ones((64, 64), dtype=np.float32) * 0.4
        read_noise = 2.0

        photon_levels = [10.0, 40.0, 160.0, 640.0]  # 4x increases
        snr_values = []

        for photon_level in photon_levels:
            _, params = self.generator.add_poisson_gaussian_noise(
                clean, photon_level, read_noise
            )
            snr_values.append(params["snr_db"])

        # For shot-noise limited case, SNR should increase as sqrt(signal)
        # In dB: SNR_dB ∝ 10*log10(sqrt(signal)) = 5*log10(signal)

        # Check that 4x photon increase gives ~3dB SNR increase (in shot-noise limit)
        for i in range(len(photon_levels) - 1):
            snr_increase = snr_values[i + 1] - snr_values[i]
            # Should be between 2-4 dB for 4x photon increase
            if not (1.5 < snr_increase < 4.5):
                return False

        return True

    def _test_read_noise_independence(self) -> bool:
        """Test that read noise is independent of signal level."""
        read_noise = 5.0
        photon_levels = [10.0, 100.0, 1000.0]

        for photon_level in photon_levels:
            # Test with different signal levels
            clean_low = np.ones((64, 64), dtype=np.float32) * 0.2
            clean_high = np.ones((64, 64), dtype=np.float32) * 0.8

            # Generate samples
            samples_low = []
            samples_high = []

            for _ in range(20):
                noisy_low, _ = self.generator.add_poisson_gaussian_noise(
                    clean_low, photon_level, read_noise
                )
                noisy_high, _ = self.generator.add_poisson_gaussian_noise(
                    clean_high, photon_level, read_noise
                )

                samples_low.append(noisy_low)
                samples_high.append(noisy_high)

            # Compute noise variance (subtract mean)
            var_low = np.var(np.array(samples_low) - np.mean(samples_low))
            var_high = np.var(np.array(samples_high) - np.mean(samples_high))

            # Read noise contribution should be the same
            # Total variance = Poisson variance + read noise^2
            poisson_var_low = photon_level * 0.2
            poisson_var_high = photon_level * 0.8

            read_var_low = var_low - poisson_var_low
            read_var_high = var_high - poisson_var_high

            # Read noise variances should be similar
            if abs(read_var_low - read_var_high) / read_noise**2 > 0.3:
                return False

        return True

    def create_validation_report(
        self, stats_results: ValidationResults, theory_results: Dict[str, bool]
    ) -> None:
        """
        Create comprehensive validation report.

        Args:
            stats_results: Statistical validation results
            theory_results: Theoretical prediction test results
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots
        self._create_validation_plots(stats_results)

        # Create text report
        report_path = output_dir / "physics_validation_report.txt"

        with open(report_path, "w") as f:
            f.write("PHYSICS VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("OVERALL RESULT: ")
            if stats_results.overall_pass and all(theory_results.values()):
                f.write("PASS ✓\n\n")
            else:
                f.write("FAIL ✗\n\n")

            f.write("STATISTICAL VALIDATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Chi-squared mean: {stats_results.chi2_mean:.3f} (target: ~1.0)\n")
            f.write(f"Chi-squared std:  {stats_results.chi2_std:.3f}\n")
            f.write(
                f"Chi-squared in range [{self.chi2_range[0]}, {self.chi2_range[1]}]: "
            )
            f.write(f"{stats_results.chi2_in_range:.1%}\n\n")

            f.write(f"Variance error mean: {stats_results.variance_error_mean:.3f} ")
            f.write(f"(target: <{self.variance_tolerance})\n")
            f.write(f"Variance error max:  {stats_results.variance_error_max:.3f}\n\n")

            f.write(f"SNR error mean: {stats_results.snr_error_mean:.3f} dB ")
            f.write(f"(target: <{self.snr_tolerance} dB)\n")
            f.write(f"SNR error std:  {stats_results.snr_error_std:.3f} dB\n\n")

            f.write("REGIME VALIDATION:\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"Low-photon regime:  {'PASS' if stats_results.low_photon_valid else 'FAIL'}\n"
            )
            f.write(
                f"High-photon regime: {'PASS' if stats_results.high_photon_valid else 'FAIL'}\n\n"
            )

            f.write("THEORETICAL PREDICTIONS:\n")
            f.write("-" * 30 + "\n")
            for test_name, passed in theory_results.items():
                f.write(f"{test_name.replace('_', ' ').title()}: ")
                f.write(f"{'PASS' if passed else 'FAIL'}\n")

            f.write("\nRECOMMendations:\n")
            f.write("-" * 30 + "\n")

            if stats_results.overall_pass and all(theory_results.values()):
                f.write("✓ Physics implementation is correct and ready for use.\n")
                f.write("✓ Proceed with Phase 2.2 guidance implementation.\n")
            else:
                f.write("✗ Issues detected in physics implementation.\n")
                if stats_results.chi2_in_range < 0.8:
                    f.write("- Chi-squared values indicate noise model problems\n")
                if abs(stats_results.variance_error_mean) > self.variance_tolerance:
                    f.write("- Variance prediction is inaccurate\n")
                if not stats_results.low_photon_valid:
                    f.write("- Low-photon regime behavior is incorrect\n")
                if not stats_results.high_photon_valid:
                    f.write("- High-photon regime behavior is incorrect\n")

                for test_name, passed in theory_results.items():
                    if not passed:
                        f.write(
                            f"- {test_name.replace('_', ' ').title()} test failed\n"
                        )

        print(f"Validation report saved to: {report_path}")

    def _create_validation_plots(self, results: ValidationResults) -> None:
        """Create validation plots."""
        output_dir = Path(self.config.output_dir) / "validation_plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Chi-squared histogram
        ax1.axvline(
            results.chi2_mean,
            color="blue",
            linestyle="-",
            label=f"Mean: {results.chi2_mean:.3f}",
        )
        ax1.axvline(1.0, color="red", linestyle="--", label="Target: 1.0")
        ax1.axvspan(
            self.chi2_range[0],
            self.chi2_range[1],
            alpha=0.2,
            color="green",
            label="Acceptable range",
        )
        ax1.set_xlabel("Chi-squared per pixel")
        ax1.set_ylabel("Density")
        ax1.set_title("Chi-squared Distribution")
        ax1.legend()

        # Validation status
        tests = ["Chi2 Range", "Variance", "SNR", "Low Photon", "High Photon"]
        status = [
            results.chi2_in_range > 0.8,
            abs(results.variance_error_mean) < self.variance_tolerance,
            abs(results.snr_error_mean) < self.snr_tolerance,
            results.low_photon_valid,
            results.high_photon_valid,
        ]

        colors = ["green" if s else "red" for s in status]
        bars = ax2.bar(tests, [1 if s else 0 for s in status], color=colors, alpha=0.7)
        ax2.set_ylabel("Pass/Fail")
        ax2.set_title("Validation Test Results")
        ax2.set_ylim(0, 1.2)

        # Add text annotations
        for bar, test, passed in zip(bars, tests, status):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                "PASS" if passed else "FAIL",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Error distributions
        ax3.text(
            0.1,
            0.8,
            f"Variance Error Mean: {results.variance_error_mean:.4f}",
            transform=ax3.transAxes,
        )
        ax3.text(
            0.1,
            0.6,
            f"Variance Error Max: {results.variance_error_max:.4f}",
            transform=ax3.transAxes,
        )
        ax3.text(
            0.1,
            0.4,
            f"SNR Error Mean: {results.snr_error_mean:.3f} dB",
            transform=ax3.transAxes,
        )
        ax3.text(
            0.1,
            0.2,
            f"SNR Error Std: {results.snr_error_std:.3f} dB",
            transform=ax3.transAxes,
        )
        ax3.set_title("Error Statistics")
        ax3.axis("off")

        # Overall status
        overall_status = "PASS" if results.overall_pass else "FAIL"
        color = "green" if results.overall_pass else "red"

        ax4.text(
            0.5,
            0.5,
            f"OVERALL\n{overall_status}",
            transform=ax4.transAxes,
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.2),
        )
        ax4.set_title("Physics Validation Result")
        ax4.axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "validation_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Validation plots saved to: {output_dir}")


def main():
    """Main function for physics validation."""
    parser = argparse.ArgumentParser(description="Validate physics implementation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/synthetic_validation.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/physics_validation",
        help="Output directory",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick validation with reduced data"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of trials for statistical validation",
    )

    args = parser.parse_args()

    # Load configuration
    if Path(args.config).exists():
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)

        # Filter out extra keys not in SyntheticConfig
        valid_keys = [
            "image_size",
            "num_images",
            "photon_levels",
            "read_noise_levels",
            "background_level",
            "pattern_types",
            "output_dir",
            "save_plots",
            "save_metadata",
        ]
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = SyntheticConfig(**filtered_config)
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = SyntheticConfig()

    # Override output directory
    config.output_dir = args.output_dir
    config.save_plots = True

    # Quick mode
    if args.quick:
        config.photon_levels = [1.0, 10.0, 100.0, 1000.0]
        config.read_noise_levels = [1.0, 5.0]
        config.pattern_types = ["constant", "gradient"]
        args.trials = 20
        print("Running in quick mode with reduced parameter space")

    print("Starting physics validation...")
    print(f"Output directory: {config.output_dir}")
    print(f"Number of trials: {args.trials}")

    # Run validation
    validator = PhysicsValidator(config)

    # Statistical validation
    print("\n" + "=" * 50)
    print("STATISTICAL VALIDATION")
    print("=" * 50)
    stats_results = validator.validate_noise_statistics(num_trials=args.trials)

    # Theoretical predictions
    print("\n" + "=" * 50)
    print("THEORETICAL PREDICTIONS")
    print("=" * 50)
    theory_results = validator.validate_theoretical_predictions()

    # Create comprehensive report
    validator.create_validation_report(stats_results, theory_results)

    # Print summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    overall_pass = stats_results.overall_pass and all(theory_results.values())

    print(f"Overall result: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print(f"Chi-squared mean: {stats_results.chi2_mean:.3f} (target: ~1.0)")
    print(f"Chi-squared in range: {stats_results.chi2_in_range:.1%}")
    print(f"Variance error: {stats_results.variance_error_mean:.3f}")
    print(f"SNR error: {stats_results.snr_error_mean:.3f} dB")

    print("\nTheoretical tests:")
    for test_name, passed in theory_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")

    if overall_pass:
        print("\n✓ Physics implementation is correct!")
        print("✓ Ready to proceed with Phase 2.2 guidance implementation.")
    else:
        print("\n✗ Issues detected in physics implementation.")
        print("✗ Review validation report before proceeding.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
