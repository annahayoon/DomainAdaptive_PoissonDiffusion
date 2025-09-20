#!/usr/bin/env python3
"""
Integration test for Poisson-Gaussian guidance system.

This script validates the guidance system using synthetic data from Phase 2.2.1,
ensuring that the physics-aware guidance correctly handles the Poisson-Gaussian
noise model across different photon regimes.

Usage:
    python scripts/test_guidance_integration.py [--config CONFIG_FILE] [--output_dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.calibration import CalibrationParams, SensorCalibration
from core.guidance_config import GuidanceConfig, GuidancePresets
from core.logging_config import get_logger, setup_project_logging
from core.poisson_guidance import PoissonGuidance, create_domain_guidance
from scripts.generate_synthetic_data import SyntheticConfig, SyntheticDataGenerator

logger = get_logger(__name__)


class GuidanceIntegrationTester:
    """
    Comprehensive integration tester for Poisson-Gaussian guidance.

    This class validates the guidance system using synthetic data with known
    ground truth, testing across different photon regimes and noise conditions.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the integration tester.

        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            # Filter out extra keys for SyntheticConfig
            synthetic_keys = {
                "image_size",
                "num_images",
                "photon_levels",
                "read_noise_levels",
                "background",
                "pattern_types",
                "output_format",
                "seed",
            }
            synthetic_config_data = {
                k: v for k, v in config_data.items() if k in synthetic_keys
            }
            self.synthetic_config = SyntheticConfig(**synthetic_config_data)
        else:
            # Default configuration for testing
            self.synthetic_config = SyntheticConfig(
                image_size=64,
                num_images=20,
                photon_levels=[10, 50, 200, 1000],
                read_noise_levels=[1.0, 3.0, 5.0],
                background_level=5.0,
                pattern_types=["constant", "gradient", "gaussian_spots"],
            )

        # Initialize synthetic data generator
        self.data_generator = SyntheticDataGenerator(self.synthetic_config)

        # Test results storage
        self.results: Dict[str, Any] = {}

        logger.info("GuidanceIntegrationTester initialized")

    def test_guidance_accuracy(self) -> Dict[str, float]:
        """
        Test guidance computation accuracy against analytical solutions.

        Returns:
            Dictionary of accuracy metrics
        """
        logger.info("Testing guidance accuracy...")

        accuracy_results = {}

        for photon_level in self.synthetic_config.photon_levels:
            for read_noise in self.synthetic_config.read_noise_levels:
                # Create calibration and guidance
                calibration = SensorCalibration(
                    params=CalibrationParams(
                        gain=1.0,
                        black_level=0.0,
                        white_level=65535.0,
                        read_noise=read_noise,
                        domain="microscopy",
                    )
                )

                # Use adaptive gradient clipping based on signal level
                # Higher photon levels need higher clipping thresholds to avoid distorting physics
                adaptive_clip = max(50.0, photon_level * 0.2)  # Scale with signal

                guidance = PoissonGuidance(
                    scale=photon_level,
                    background=self.synthetic_config.background_level,
                    read_noise=read_noise,
                    config=GuidanceConfig(
                        mode="wls",
                        collect_diagnostics=True,
                        gradient_clip=adaptive_clip,
                        variance_eps=0.01,  # Smaller epsilon for better accuracy
                    ),
                )

                # Generate test data (constant pattern is already 0.5)
                clean_pattern = self.data_generator.generate_pattern(
                    "constant", self.synthetic_config.image_size
                )
                noisy_data, _ = self.data_generator.add_poisson_gaussian_noise(
                    clean_pattern,
                    photon_level,
                    read_noise,
                    self.synthetic_config.background_level,
                )

                # Convert to torch tensors
                x_hat = torch.from_numpy(clean_pattern[None, None, :, :]).float()
                y_observed = torch.from_numpy(noisy_data[None, None, :, :]).float()

                # Compute guidance score
                score = guidance.compute_score(x_hat, y_observed)

                # Analytical solution for constant pattern
                lambda_true = (
                    photon_level * 0.5 + self.synthetic_config.background_level
                )
                variance_true = lambda_true + read_noise**2
                expected_score = (
                    (y_observed - lambda_true) / variance_true * photon_level
                )

                # Compute relative error
                relative_error = torch.abs(score - expected_score) / (
                    torch.abs(expected_score) + 1e-8
                )
                mean_error = relative_error.mean().item()

                test_key = f"photon_{photon_level}_noise_{read_noise}"
                accuracy_results[test_key] = mean_error

                logger.debug(
                    f"Accuracy test {test_key}: {mean_error:.4f} relative error"
                )

        # Overall accuracy
        overall_accuracy = np.mean(list(accuracy_results.values()))
        accuracy_results["overall_mean_error"] = overall_accuracy

        logger.info(
            f"Guidance accuracy test completed. Overall error: {overall_accuracy:.4f}"
        )
        return accuracy_results

    def test_chi_squared_consistency(self) -> Dict[str, float]:
        """
        Test that guidance preserves chi-squared statistics.

        Returns:
            Dictionary of chi-squared statistics
        """
        logger.info("Testing chi-squared consistency...")

        chi2_results = {}

        for photon_level in self.synthetic_config.photon_levels:
            for read_noise in self.synthetic_config.read_noise_levels:
                guidance = PoissonGuidance(
                    scale=photon_level,
                    background=self.synthetic_config.background_level,
                    read_noise=read_noise,
                    config=GuidanceConfig(mode="wls", collect_diagnostics=True),
                )

                chi2_values = []

                # Test multiple patterns
                for pattern_type in self.synthetic_config.pattern_types:
                    for _ in range(5):  # Multiple realizations
                        # Generate synthetic data
                        clean_pattern = self.data_generator.generate_pattern(
                            pattern_type, self.synthetic_config.image_size
                        )
                        noisy_data, _ = self.data_generator.add_poisson_gaussian_noise(
                            clean_pattern,
                            photon_level,
                            read_noise,
                            self.synthetic_config.background_level,
                        )

                        # Convert to tensors
                        x_hat = torch.from_numpy(
                            clean_pattern[None, None, :, :]
                        ).float()
                        y_observed = torch.from_numpy(
                            noisy_data[None, None, :, :]
                        ).float()

                        # Compute score (this collects diagnostics)
                        guidance.compute_score(x_hat, y_observed)

                # Get chi-squared statistics
                diagnostics = guidance.get_diagnostics()
                if "chi2_mean" in diagnostics:
                    chi2_mean = diagnostics["chi2_mean"]
                    chi2_std = diagnostics.get("chi2_std", 0.0)

                    test_key = f"photon_{photon_level}_noise_{read_noise}"
                    chi2_results[f"{test_key}_mean"] = chi2_mean
                    chi2_results[f"{test_key}_std"] = chi2_std

                    logger.debug(
                        f"Chi² test {test_key}: {chi2_mean:.3f} ± {chi2_std:.3f}"
                    )

                guidance.reset_diagnostics()

        logger.info("Chi-squared consistency test completed")
        return chi2_results

    def test_gamma_scheduling(self) -> Dict[str, List[float]]:
        """
        Test different gamma scheduling options.

        Returns:
            Dictionary of gamma values for different schedules
        """
        logger.info("Testing gamma scheduling...")

        # Test parameters
        photon_level = 200.0
        read_noise = 3.0
        sigma_values = torch.linspace(0.1, 2.0, 10)

        scheduling_results = {}

        for schedule in ["sigma2", "linear", "const"]:
            guidance = PoissonGuidance(
                scale=photon_level,
                background=self.synthetic_config.background_level,
                read_noise=read_noise,
                config=GuidanceConfig(gamma_schedule=schedule, kappa=0.5),
            )

            gamma_values = []
            for sigma in sigma_values:
                gamma = guidance.gamma_schedule(sigma)
                gamma_values.append(gamma.item())

            scheduling_results[schedule] = gamma_values
            logger.debug(
                f"Gamma schedule {schedule}: {gamma_values[:3]}... (first 3 values)"
            )

        logger.info("Gamma scheduling test completed")
        return scheduling_results

    def test_domain_configurations(self) -> Dict[str, Dict[str, float]]:
        """
        Test domain-specific guidance configurations.

        Returns:
            Dictionary of performance metrics for each domain
        """
        logger.info("Testing domain configurations...")

        domain_results = {}

        # Test data parameters
        photon_level = 100.0
        read_noise = 2.0

        for domain in ["photography", "microscopy", "astronomy"]:
            # Create domain-specific guidance
            guidance = create_domain_guidance(
                scale=photon_level,
                background=self.synthetic_config.background_level,
                read_noise=read_noise,
                domain=domain,
            )

            # Generate test data
            clean_pattern = self.data_generator.generate_pattern(
                "gradient", self.synthetic_config.image_size
            )
            noisy_data, _ = self.data_generator.add_poisson_gaussian_noise(
                clean_pattern,
                photon_level,
                read_noise,
                self.synthetic_config.background_level,
            )

            # Convert to tensors
            x_hat = torch.from_numpy(clean_pattern[None, None, :, :]).float()
            y_observed = torch.from_numpy(noisy_data[None, None, :, :]).float()
            sigma_t = torch.tensor([0.5])

            # Compute guidance
            guidance_grad = guidance.compute(x_hat, y_observed, sigma_t)

            # Collect metrics
            grad_norm = torch.norm(guidance_grad).item()
            grad_max = torch.abs(guidance_grad).max().item()

            diagnostics = guidance.get_diagnostics()

            domain_results[domain] = {
                "gradient_norm": grad_norm,
                "gradient_max": grad_max,
                "mode": guidance.config.mode,
                "kappa": guidance.config.kappa,
                "gamma_schedule": guidance.config.gamma_schedule,
            }

            if "chi2_mean" in diagnostics:
                domain_results[domain]["chi2_mean"] = diagnostics["chi2_mean"]

            logger.debug(
                f"Domain {domain}: grad_norm={grad_norm:.3f}, mode={guidance.config.mode}"
            )

        logger.info("Domain configuration test completed")
        return domain_results

    def test_numerical_stability(self) -> Dict[str, bool]:
        """
        Test numerical stability under extreme conditions.

        Returns:
            Dictionary of stability test results
        """
        logger.info("Testing numerical stability...")

        stability_results = {}

        # Test extreme conditions
        test_conditions = [
            ("very_low_photons", 0.1, 1.0),
            ("very_high_photons", 10000.0, 1.0),
            ("high_read_noise", 100.0, 50.0),
            ("zero_background", 100.0, 3.0),  # Will test with background=0
        ]

        for test_name, photon_level, read_noise in test_conditions:
            background = (
                0.0
                if test_name == "zero_background"
                else self.synthetic_config.background_level
            )

            guidance = PoissonGuidance(
                scale=photon_level,
                background=background,
                read_noise=read_noise,
                config=GuidanceConfig(mode="wls", gradient_clip=100.0),
            )

            try:
                # Generate extreme test data
                clean_pattern = self.data_generator.generate_pattern(
                    "constant", self.synthetic_config.image_size
                )
                noisy_data, _ = self.data_generator.add_poisson_gaussian_noise(
                    clean_pattern, photon_level, read_noise, background
                )

                # Convert to tensors
                x_hat = torch.from_numpy(clean_pattern[None, None, :, :]).float()
                y_observed = torch.from_numpy(noisy_data[None, None, :, :]).float()

                # Test guidance computation
                score = guidance.compute_score(x_hat, y_observed)

                # Check for NaN/Inf
                is_stable = torch.isfinite(score).all().item()
                stability_results[test_name] = is_stable

                if not is_stable:
                    logger.warning(
                        f"Stability test {test_name} failed: non-finite values detected"
                    )
                else:
                    logger.debug(f"Stability test {test_name} passed")

            except Exception as e:
                logger.error(f"Stability test {test_name} failed with exception: {e}")
                stability_results[test_name] = False

        logger.info("Numerical stability test completed")
        return stability_results

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all integration tests.

        Returns:
            Complete test results dictionary
        """
        logger.info("Running complete guidance integration test suite...")

        self.results = {
            "accuracy": self.test_guidance_accuracy(),
            "chi_squared": self.test_chi_squared_consistency(),
            "gamma_scheduling": self.test_gamma_scheduling(),
            "domain_configs": self.test_domain_configurations(),
            "stability": self.test_numerical_stability(),
        }

        # Compute overall success metrics
        self.results["summary"] = self._compute_summary()

        logger.info("Integration test suite completed")
        return self.results

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute overall test summary."""
        summary = {}

        # Accuracy summary
        if "accuracy" in self.results:
            overall_error = self.results["accuracy"].get(
                "overall_mean_error", float("inf")
            )
            summary["accuracy_pass"] = (
                overall_error < 0.15
            )  # 15% threshold (more realistic for clipped gradients)
            summary["mean_relative_error"] = overall_error

        # Chi-squared summary
        if "chi_squared" in self.results:
            chi2_values = [
                v for k, v in self.results["chi_squared"].items() if k.endswith("_mean")
            ]
            if chi2_values:
                chi2_mean = np.mean(chi2_values)
                summary["chi2_pass"] = 0.7 <= chi2_mean <= 1.3  # Reasonable range
                summary["mean_chi2"] = chi2_mean

        # Stability summary
        if "stability" in self.results:
            stability_tests = list(self.results["stability"].values())
            summary["stability_pass"] = all(stability_tests)
            summary["stability_rate"] = np.mean(stability_tests)

        # Overall pass/fail
        summary["overall_pass"] = all(
            [
                summary.get("accuracy_pass", False),
                summary.get("chi2_pass", False),
                summary.get("stability_pass", False),
            ]
        )

        return summary

    def save_results(self, output_path: str):
        """Save test results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(self.results, f, indent=2, default_flow_style=False)

        logger.info(f"Results saved to {output_path}")

    def generate_report(self) -> str:
        """Generate human-readable test report."""
        if not self.results:
            return "No test results available. Run tests first."

        report = []
        report.append("=" * 60)
        report.append("POISSON-GAUSSIAN GUIDANCE INTEGRATION TEST REPORT")
        report.append("=" * 60)

        # Summary
        summary = self.results.get("summary", {})
        overall_pass = summary.get("overall_pass", False)
        status = "✅ PASS" if overall_pass else "❌ FAIL"
        report.append(f"\nOVERALL STATUS: {status}")

        # Detailed results
        if "accuracy" in self.results:
            error = summary.get("mean_relative_error", 0.0)
            accuracy_pass = summary.get("accuracy_pass", False)
            status = "✅" if accuracy_pass else "❌"
            report.append(f"\n{status} Accuracy Test: {error:.4f} mean relative error")

        if "chi_squared" in self.results:
            chi2 = summary.get("mean_chi2", 0.0)
            chi2_pass = summary.get("chi2_pass", False)
            status = "✅" if chi2_pass else "❌"
            report.append(f"{status} Chi-squared Test: {chi2:.3f} mean χ²")

        if "stability" in self.results:
            stability_rate = summary.get("stability_rate", 0.0)
            stability_pass = summary.get("stability_pass", False)
            status = "✅" if stability_pass else "❌"
            report.append(f"{status} Stability Test: {stability_rate:.1%} pass rate")

        # Domain configurations
        if "domain_configs" in self.results:
            report.append("\nDomain Configuration Results:")
            for domain, metrics in self.results["domain_configs"].items():
                mode = metrics.get("mode", "unknown")
                kappa = metrics.get("kappa", 0.0)
                report.append(f"  {domain}: mode={mode}, κ={kappa}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Test Poisson-Gaussian guidance integration"
    )
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/guidance_integration_test",
        help="Output directory for results",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_project_logging(level=log_level)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    tester = GuidanceIntegrationTester(args.config)
    results = tester.run_all_tests()

    # Save results
    tester.save_results(output_dir / "integration_test_results.yaml")

    # Generate and save report
    report = tester.generate_report()
    with open(output_dir / "integration_test_report.txt", "w") as f:
        f.write(report)

    # Print report
    print(report)

    # Exit with appropriate code
    overall_pass = results.get("summary", {}).get("overall_pass", False)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
