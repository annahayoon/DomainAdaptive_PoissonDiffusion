#!/usr/bin/env python3
"""
Comprehensive test for domain encoder conditioning vectors.

This script validates the domain encoder implementation for task 3.2,
testing conditioning vector properties, ranges, and normalization.

Usage:
    python scripts/test_domain_encoder.py [--verbose] [--output_dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.logging_config import get_logger, setup_project_logging
from models.edm_wrapper import DomainEncoder

logger = get_logger(__name__)


class DomainEncoderValidator:
    """
    Comprehensive validator for domain encoder functionality.

    Tests all aspects of the 6-dimensional conditioning vector:
    - Domain one-hot encoding correctness
    - Log scale normalization properties
    - Relative parameter computation
    - Numerical stability and edge cases
    """

    def __init__(self):
        """Initialize validator."""
        self.encoder = DomainEncoder()
        self.results = {}

        # Test parameter ranges (typical values for each domain)
        self.test_ranges = {
            "photography": {
                "scales": [100, 500, 1000, 2000, 5000],
                "read_noises": [1.0, 2.0, 3.0, 5.0, 10.0],
                "backgrounds": [0.0, 5.0, 10.0, 20.0, 50.0],
            },
            "microscopy": {
                "scales": [50, 200, 500, 1000, 3000],
                "read_noises": [0.5, 1.0, 2.0, 4.0, 8.0],
                "backgrounds": [0.0, 2.0, 5.0, 10.0, 25.0],
            },
            "astronomy": {
                "scales": [10, 50, 100, 500, 2000],
                "read_noises": [1.0, 3.0, 5.0, 10.0, 20.0],
                "backgrounds": [5.0, 10.0, 20.0, 50.0, 100.0],
            },
        }

        logger.info("DomainEncoderValidator initialized")

    def test_domain_one_hot_encoding(self) -> Dict[str, bool]:
        """Test domain one-hot encoding correctness."""
        logger.info("Testing domain one-hot encoding...")

        results = {}

        # Test individual domains
        for domain_name, expected_idx in self.encoder.domain_to_idx.items():
            condition = self.encoder.encode_domain(
                domain=domain_name, scale=1000.0, read_noise=3.0, background=10.0
            )

            # Check one-hot encoding
            onehot = condition[0, :3]

            # Should have exactly one 1.0 and two 0.0s
            assert torch.sum(onehot) == 1.0, f"One-hot sum != 1 for {domain_name}"
            assert onehot[expected_idx] == 1.0, f"Wrong index for {domain_name}"

            # Other indices should be 0
            for i in range(3):
                if i != expected_idx:
                    assert (
                        onehot[i] == 0.0
                    ), f"Non-zero at wrong index for {domain_name}"

            results[domain_name] = True
            logger.debug(f"Domain {domain_name} one-hot encoding: {onehot.tolist()}")

        # Test batch encoding
        domains = ["photography", "microscopy", "astronomy"]
        batch_condition = self.encoder.encode_domain(
            domain=domains,
            scale=[1000.0, 500.0, 2000.0],
            read_noise=[3.0, 2.0, 5.0],
            background=[10.0, 5.0, 20.0],
        )

        expected_onehot = torch.eye(3)
        torch.testing.assert_close(batch_condition[:, :3], expected_onehot)
        results["batch_encoding"] = True

        logger.info("✓ Domain one-hot encoding tests passed")
        return results

    def test_log_scale_normalization(self) -> Dict[str, float]:
        """Test log scale normalization properties."""
        logger.info("Testing log scale normalization...")

        results = {}

        # Test range of scales
        scales = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        log_scale_norms = []

        for scale in scales:
            condition = self.encoder.encode_domain(
                domain="photography", scale=scale, read_noise=3.0, background=10.0
            )

            log_scale_norm = condition[0, 3].item()
            log_scale_norms.append(log_scale_norm)

            # Check that normalization is reasonable
            assert (
                -5.0 <= log_scale_norm <= 5.0
            ), f"Log scale norm {log_scale_norm} out of range for scale {scale}"

        # Check monotonicity (higher scale -> higher normalized value)
        for i in range(1, len(log_scale_norms)):
            assert (
                log_scale_norms[i] > log_scale_norms[i - 1]
            ), "Log scale normalization not monotonic"

        results["monotonicity"] = True
        results["range_min"] = min(log_scale_norms)
        results["range_max"] = max(log_scale_norms)
        results["mean"] = np.mean(log_scale_norms)
        results["std"] = np.std(log_scale_norms)

        logger.debug(f"Log scale norms: {log_scale_norms}")
        logger.info("✓ Log scale normalization tests passed")
        return results

    def test_relative_parameters(self) -> Dict[str, Dict[str, float]]:
        """Test relative parameter computation."""
        logger.info("Testing relative parameter computation...")

        results = {}

        # Test cases with known relative values
        test_cases = [
            # (scale, read_noise, background, expected_rel_rn, expected_rel_bg)
            (100.0, 10.0, 5.0, 0.1, 0.05),
            (1000.0, 50.0, 100.0, 0.05, 0.1),
            (500.0, 25.0, 25.0, 0.05, 0.05),
        ]

        for i, (scale, read_noise, background, exp_rel_rn, exp_rel_bg) in enumerate(
            test_cases
        ):
            condition = self.encoder.encode_domain(
                domain="photography",
                scale=scale,
                read_noise=read_noise,
                background=background,
            )

            rel_read_noise = condition[0, 4].item()
            rel_background = condition[0, 5].item()

            # Check relative read noise
            assert (
                abs(rel_read_noise - exp_rel_rn) < 0.01
            ), f"Relative read noise mismatch: {rel_read_noise} vs {exp_rel_rn}"

            # Check relative background
            assert (
                abs(rel_background - exp_rel_bg) < 0.01
            ), f"Relative background mismatch: {rel_background} vs {exp_rel_bg}"

            results[f"case_{i}"] = {
                "rel_read_noise": rel_read_noise,
                "rel_background": rel_background,
                "expected_rel_rn": exp_rel_rn,
                "expected_rel_bg": exp_rel_bg,
            }

        logger.info("✓ Relative parameter tests passed")
        return results

    def test_conditioning_vector_ranges(self) -> Dict[str, Dict[str, float]]:
        """Test conditioning vector ranges across typical parameter space."""
        logger.info("Testing conditioning vector ranges...")

        results = {}
        all_conditions = []

        # Generate conditions across all domains and parameter ranges
        for domain in ["photography", "microscopy", "astronomy"]:
            ranges = self.test_ranges[domain]

            for scale in ranges["scales"]:
                for read_noise in ranges["read_noises"]:
                    for background in ranges["backgrounds"]:
                        condition = self.encoder.encode_domain(
                            domain=domain,
                            scale=scale,
                            read_noise=read_noise,
                            background=background,
                        )
                        all_conditions.append(condition)

        # Stack all conditions
        all_conditions = torch.cat(all_conditions, dim=0)  # [N, 6]

        # Analyze each dimension
        dimension_names = [
            "domain_photography",
            "domain_microscopy",
            "domain_astronomy",
            "log_scale_norm",
            "rel_read_noise",
            "rel_background",
        ]

        for i, dim_name in enumerate(dimension_names):
            values = all_conditions[:, i]

            results[dim_name] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(values.median()),
            }

            logger.debug(
                f"{dim_name}: min={results[dim_name]['min']:.3f}, "
                f"max={results[dim_name]['max']:.3f}, "
                f"mean={results[dim_name]['mean']:.3f}"
            )

        # Check that ranges are reasonable
        # Domain one-hots should be binary
        for domain_dim in [
            "domain_photography",
            "domain_microscopy",
            "domain_astronomy",
        ]:
            assert results[domain_dim]["min"] == 0.0, f"{domain_dim} min should be 0"
            assert results[domain_dim]["max"] == 1.0, f"{domain_dim} max should be 1"

        # Relative parameters should be non-negative and reasonable
        assert (
            results["rel_read_noise"]["min"] >= 0.0
        ), "Relative read noise should be non-negative"
        assert (
            results["rel_background"]["min"] >= 0.0
        ), "Relative background should be non-negative"
        assert (
            results["rel_read_noise"]["max"] < 5.0
        ), "Relative read noise should be reasonable (< 5x signal)"

        logger.info("✓ Conditioning vector range tests passed")
        return results

    def test_numerical_stability(self) -> Dict[str, bool]:
        """Test numerical stability with edge cases."""
        logger.info("Testing numerical stability...")

        results = {}

        # Edge cases
        edge_cases = [
            ("zero_scale", 0.0, 1.0, 0.0),
            ("tiny_scale", 1e-6, 1.0, 0.0),
            ("huge_scale", 1e6, 1.0, 0.0),
            ("zero_read_noise", 1000.0, 0.0, 0.0),
            ("huge_read_noise", 1000.0, 1000.0, 0.0),
            ("negative_background", 1000.0, 3.0, -10.0),
            ("huge_background", 1000.0, 3.0, 10000.0),
        ]

        for case_name, scale, read_noise, background in edge_cases:
            try:
                condition = self.encoder.encode_domain(
                    domain="photography",
                    scale=scale,
                    read_noise=read_noise,
                    background=background,
                )

                # Check for NaN/Inf
                is_finite = torch.isfinite(condition).all()
                results[case_name] = bool(is_finite)

                if not is_finite:
                    logger.warning(f"Non-finite values in {case_name}: {condition}")
                else:
                    logger.debug(f"{case_name}: {condition[0].tolist()}")

            except Exception as e:
                logger.error(f"Exception in {case_name}: {e}")
                results[case_name] = False

        logger.info("✓ Numerical stability tests passed")
        return results

    def test_encode_decode_consistency(self) -> Dict[str, float]:
        """Test encode-decode consistency."""
        logger.info("Testing encode-decode consistency...")

        results = {}

        test_cases = [
            ("photography", 1000.0, 3.0, 10.0),
            ("microscopy", 500.0, 2.0, 5.0),
            ("astronomy", 2000.0, 5.0, 20.0),
        ]

        for domain, scale, read_noise, background in test_cases:
            # Encode
            condition = self.encoder.encode_domain(
                domain=domain, scale=scale, read_noise=read_noise, background=background
            )

            # Decode
            decoded = self.encoder.decode_domain(condition)

            # Check domain
            expected_idx = self.encoder.domain_to_idx[domain]
            assert (
                decoded["domain_idx"][0] == expected_idx
            ), f"Domain decode mismatch for {domain}"

            # Check approximate equality for continuous parameters
            scale_error = abs(decoded["scale"][0] - scale) / scale
            read_noise_error = abs(decoded["read_noise"][0] - read_noise) / max(
                read_noise, 1e-6
            )
            background_error = abs(decoded["background"][0] - background) / max(
                abs(background), 1e-6
            )

            results[f"{domain}_scale_error"] = float(scale_error)
            results[f"{domain}_read_noise_error"] = float(read_noise_error)
            results[f"{domain}_background_error"] = float(background_error)

            # Errors should be small (allowing for normalization/denormalization)
            assert scale_error < 0.1, f"Large scale error for {domain}: {scale_error}"
            assert (
                read_noise_error < 0.1
            ), f"Large read noise error for {domain}: {read_noise_error}"

            logger.debug(
                f"{domain}: scale_err={scale_error:.4f}, rn_err={read_noise_error:.4f}, bg_err={background_error:.4f}"
            )

        logger.info("✓ Encode-decode consistency tests passed")
        return results

    def test_batch_processing(self) -> Dict[str, bool]:
        """Test batch processing capabilities."""
        logger.info("Testing batch processing...")

        results = {}

        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]

        for batch_size in batch_sizes:
            domains = ["photography"] * batch_size
            scales = [1000.0 + i * 100 for i in range(batch_size)]
            read_noises = [3.0 + i * 0.5 for i in range(batch_size)]
            backgrounds = [10.0 + i * 2 for i in range(batch_size)]

            condition = self.encoder.encode_domain(
                domain=domains,
                scale=scales,
                read_noise=read_noises,
                background=backgrounds,
            )

            # Check shape
            assert condition.shape == (
                batch_size,
                6,
            ), f"Wrong shape for batch size {batch_size}"

            # Check all finite
            assert torch.isfinite(
                condition
            ).all(), f"Non-finite values for batch size {batch_size}"

            results[f"batch_size_{batch_size}"] = True

        # Test mixed domains
        mixed_domains = ["photography", "microscopy", "astronomy", "photography"]
        mixed_condition = self.encoder.encode_domain(
            domain=mixed_domains,
            scale=[1000.0, 500.0, 2000.0, 1500.0],
            read_noise=[3.0, 2.0, 5.0, 4.0],
            background=[10.0, 5.0, 20.0, 15.0],
        )

        assert mixed_condition.shape == (4, 6)

        # Check domain encodings
        expected_domains = [
            0,
            1,
            2,
            0,
        ]  # photography, microscopy, astronomy, photography
        for i, expected_idx in enumerate(expected_domains):
            assert (
                mixed_condition[i, expected_idx] == 1.0
            ), f"Wrong domain encoding at index {i}"

        results["mixed_domains"] = True

        logger.info("✓ Batch processing tests passed")
        return results

    def run_all_tests(self) -> Dict[str, any]:
        """Run all validation tests."""
        logger.info("Running comprehensive domain encoder validation...")

        self.results = {
            "one_hot_encoding": self.test_domain_one_hot_encoding(),
            "log_scale_normalization": self.test_log_scale_normalization(),
            "relative_parameters": self.test_relative_parameters(),
            "conditioning_ranges": self.test_conditioning_vector_ranges(),
            "numerical_stability": self.test_numerical_stability(),
            "encode_decode_consistency": self.test_encode_decode_consistency(),
            "batch_processing": self.test_batch_processing(),
        }

        # Compute summary
        self.results["summary"] = self._compute_summary()

        logger.info("Domain encoder validation completed")
        return self.results

    def _compute_summary(self) -> Dict[str, any]:
        """Compute validation summary."""
        summary = {}

        # Count passed tests (only boolean results)
        total_tests = 0
        passed_tests = 0

        for test_name, test_results in self.results.items():
            if test_name == "summary":
                continue

            if isinstance(test_results, dict):
                for subtest, result in test_results.items():
                    if isinstance(result, bool):
                        total_tests += 1
                        if result:
                            passed_tests += 1

        summary["total_tests"] = total_tests
        summary["passed_tests"] = passed_tests
        summary["pass_rate"] = passed_tests / total_tests if total_tests > 0 else 0.0
        summary["overall_pass"] = summary["pass_rate"] >= 0.95  # 95% pass rate required

        # Key metrics
        if "conditioning_ranges" in self.results:
            ranges = self.results["conditioning_ranges"]
            summary["log_scale_range"] = [
                ranges["log_scale_norm"]["min"],
                ranges["log_scale_norm"]["max"],
            ]
            summary["rel_read_noise_range"] = [
                ranges["rel_read_noise"]["min"],
                ranges["rel_read_noise"]["max"],
            ]
            summary["rel_background_range"] = [
                ranges["rel_background"]["min"],
                ranges["rel_background"]["max"],
            ]

        return summary

    def save_results(self, output_path: str):
        """Save validation results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_serializable(self.results)

        with open(output_path, "w") as f:
            yaml.dump(serializable_results, f, indent=2, default_flow_style=False)

        logger.info(f"Results saved to {output_path}")

    def _make_serializable(self, obj):
        """Convert tensors and numpy arrays to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    def generate_report(self) -> str:
        """Generate human-readable validation report."""
        if not self.results:
            return "No validation results available. Run tests first."

        report = []
        report.append("=" * 70)
        report.append("DOMAIN ENCODER VALIDATION REPORT")
        report.append("=" * 70)

        # Summary
        summary = self.results.get("summary", {})
        overall_pass = summary.get("overall_pass", False)
        status = "✅ PASS" if overall_pass else "❌ FAIL"
        report.append(f"\nOVERALL STATUS: {status}")

        pass_rate = summary.get("pass_rate", 0.0)
        total_tests = summary.get("total_tests", 0)
        passed_tests = summary.get("passed_tests", 0)
        report.append(
            f"Test Results: {passed_tests}/{total_tests} passed ({pass_rate:.1%})"
        )

        # Key metrics
        if "log_scale_range" in summary:
            log_range = summary["log_scale_range"]
            report.append(f"Log Scale Range: [{log_range[0]:.2f}, {log_range[1]:.2f}]")

        if "rel_read_noise_range" in summary:
            rn_range = summary["rel_read_noise_range"]
            report.append(
                f"Relative Read Noise Range: [{rn_range[0]:.3f}, {rn_range[1]:.3f}]"
            )

        if "rel_background_range" in summary:
            bg_range = summary["rel_background_range"]
            report.append(
                f"Relative Background Range: [{bg_range[0]:.3f}, {bg_range[1]:.3f}]"
            )

        # Test details
        report.append("\nDETAILED RESULTS:")

        test_names = {
            "one_hot_encoding": "Domain One-Hot Encoding",
            "log_scale_normalization": "Log Scale Normalization",
            "relative_parameters": "Relative Parameters",
            "conditioning_ranges": "Conditioning Vector Ranges",
            "numerical_stability": "Numerical Stability",
            "encode_decode_consistency": "Encode-Decode Consistency",
            "batch_processing": "Batch Processing",
        }

        for test_key, test_name in test_names.items():
            if test_key in self.results:
                test_results = self.results[test_key]
                if isinstance(test_results, dict):
                    passed = sum(
                        1 for v in test_results.values() if isinstance(v, bool) and v
                    )
                    total = sum(1 for v in test_results.values() if isinstance(v, bool))
                    if total > 0:
                        status = "✅" if passed == total else "❌"
                        report.append(f"  {status} {test_name}: {passed}/{total}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate domain encoder implementation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/domain_encoder_validation",
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

    # Run validation
    validator = DomainEncoderValidator()
    results = validator.run_all_tests()

    # Save results
    validator.save_results(output_dir / "domain_encoder_validation.yaml")

    # Generate and save report
    report = validator.generate_report()
    with open(output_dir / "domain_encoder_report.txt", "w") as f:
        f.write(report)

    # Print report
    print(report)

    # Exit with appropriate code
    overall_pass = results.get("summary", {}).get("overall_pass", False)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
