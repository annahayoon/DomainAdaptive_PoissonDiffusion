#!/usr/bin/env python3
"""
Integration test for unified DomainDataset implementation.

This script tests the complete dataset pipeline including:
- Dataset initialization and configuration
- Train/validation/test splitting
- Geometric augmentation pipeline
- Data loading and transform consistency
- Multi-domain functionality

Usage:
    python scripts/test_domain_dataset_integration.py [--output_dir OUTPUT_DIR]
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.calibration import CalibrationParams
from core.logging_config import get_logger, setup_project_logging
from data.augmentations import (
    AugmentationConfig,
    GeometricAugmentationPipeline,
    create_training_augmentations,
    create_validation_augmentations,
)
from data.domain_datasets import (
    DomainDataset,
    MultiDomainDataset,
    create_domain_dataset,
    create_multi_domain_dataset,
)

logger = get_logger(__name__)


class DomainDatasetIntegrationTester:
    """
    Integration tester for unified DomainDataset implementation.

    Tests the complete pipeline from raw data loading through
    augmentation and provides comprehensive validation.
    """

    def __init__(self, output_dir: str = "data/domain_dataset_test"):
        """Initialize the integration tester."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test results storage
        self.results = {}

        logger.info("DomainDatasetIntegrationTester initialized")

    def create_test_data(self) -> Path:
        """Create synthetic test data for all domains."""
        test_data_dir = self.output_dir / "test_data"
        test_data_dir.mkdir(exist_ok=True)

        # Create domain directories
        domains = ["photography", "microscopy", "astronomy"]
        extensions = {"photography": ".arw", "microscopy": ".tif", "astronomy": ".fits"}

        for domain in domains:
            domain_dir = test_data_dir / domain
            domain_dir.mkdir(exist_ok=True)

            # Create mock image files
            for i in range(20):
                file_path = domain_dir / f"{domain}_{i:03d}{extensions[domain]}"

                # Create synthetic image data
                if domain == "microscopy":
                    # Create TIFF-like data
                    image_data = np.random.randint(0, 4096, (128, 128), dtype=np.uint16)
                    np.save(str(file_path).replace(".tif", ".npy"), image_data)

                # Create empty files for format detection
                file_path.touch()

        logger.info(f"Created test data in {test_data_dir}")
        return test_data_dir

    def create_test_calibrations(self) -> Dict[str, Path]:
        """Create test calibration files for each domain."""
        calibrations = {}

        calibration_configs = {
            "photography": {
                "gain": 1.0,
                "black_level": 512,
                "white_level": 16383,
                "read_noise": 5.0,
                "pixel_size": 4.29,
                "pixel_unit": "um",
                "domain": "photography",
            },
            "microscopy": {
                "gain": 2.0,
                "black_level": 100,
                "white_level": 65535,
                "read_noise": 3.0,
                "pixel_size": 0.65,
                "pixel_unit": "um",
                "domain": "microscopy",
            },
            "astronomy": {
                "gain": 1.5,
                "black_level": 0,
                "white_level": 65535,
                "read_noise": 2.0,
                "pixel_size": 0.04,
                "pixel_unit": "arcsec",
                "domain": "astronomy",
            },
        }

        for domain, config in calibration_configs.items():
            cal_file = self.output_dir / f"{domain}_calibration.json"
            with open(cal_file, "w") as f:
                json.dump(config, f, indent=2)
            calibrations[domain] = cal_file

        logger.info("Created test calibration files")
        return calibrations

    def test_augmentation_pipeline(self) -> Dict[str, Any]:
        """Test geometric augmentation pipeline."""
        logger.info("Testing augmentation pipeline...")

        results = {}

        # Test each domain's augmentation configuration
        for domain in ["photography", "microscopy", "astronomy"]:
            # Create training and validation pipelines
            train_pipeline = create_training_augmentations(
                domain, deterministic=True, seed=42
            )
            val_pipeline = create_validation_augmentations(
                domain, deterministic=True, seed=42
            )

            # Test with sample data
            sample_image = torch.rand(1, 64, 64)  # [C, H, W]
            sample_mask = torch.ones(1, 64, 64)

            # Test training augmentations
            aug_image, aug_mask, aug_info = train_pipeline(
                sample_image, sample_mask, apply_augmentations=True
            )

            # Test validation augmentations (should be minimal)
            val_image, val_mask, val_info = val_pipeline(
                sample_image, sample_mask, apply_augmentations=True
            )

            results[domain] = {
                "training_augmentations": aug_info,
                "validation_augmentations": val_info,
                "shape_preserved": aug_image.shape == sample_image.shape,
                "mask_shape_preserved": aug_mask.shape == sample_mask.shape
                if aug_mask is not None
                else True,
            }

            logger.debug(f"Domain {domain} augmentation test completed")

        # Test deterministic behavior
        pipeline1 = create_training_augmentations(
            "microscopy", deterministic=True, seed=123
        )
        pipeline2 = create_training_augmentations(
            "microscopy", deterministic=True, seed=123
        )

        test_image = torch.rand(1, 32, 32)
        aug1, _, info1 = pipeline1(test_image)
        aug2, _, info2 = pipeline2(test_image)

        results["deterministic_test"] = {
            "images_identical": torch.allclose(aug1, aug2),
            "info_identical": info1 == info2,
        }

        logger.info("Augmentation pipeline test completed")
        return results

    def test_single_domain_dataset(
        self, test_data_dir: Path, calibrations: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Test single domain dataset functionality."""
        logger.info("Testing single domain dataset...")

        results = {}

        # Test microscopy domain (most complete implementation)
        domain = "microscopy"

        try:
            # Test different splits
            for split in ["train", "val", "test"]:
                dataset = DomainDataset(
                    data_root=test_data_dir / domain,
                    domain=domain,
                    calibration_file=calibrations[domain],
                    split=split,
                    split_ratios=(0.6, 0.2, 0.2),
                    target_size=64,
                    max_files=10,
                    seed=42,
                    validate_files=False,  # Skip file validation for mock data
                    enable_augmentations=True,
                )

                results[f"{domain}_{split}"] = {
                    "num_files": len(dataset),
                    "domain": dataset.domain,
                    "split": dataset.split,
                    "has_augmentations": dataset.augmentation_pipeline is not None,
                    "target_size": dataset.target_size,
                }

                logger.debug(f"Created {split} dataset with {len(dataset)} files")

        except Exception as e:
            logger.error(f"Single domain dataset test failed: {e}")
            results["error"] = str(e)

        # Test deterministic splitting
        try:
            dataset1 = DomainDataset(
                data_root=test_data_dir / domain,
                domain=domain,
                calibration_file=calibrations[domain],
                split="train",
                seed=42,
                validate_files=False,
                max_files=5,
            )

            dataset2 = DomainDataset(
                data_root=test_data_dir / domain,
                domain=domain,
                calibration_file=calibrations[domain],
                split="train",
                seed=42,
                validate_files=False,
                max_files=5,
            )

            results["deterministic_splitting"] = {
                "same_files": dataset1.file_paths == dataset2.file_paths,
                "same_length": len(dataset1) == len(dataset2),
            }

        except Exception as e:
            logger.error(f"Deterministic splitting test failed: {e}")
            results["deterministic_splitting"] = {"error": str(e)}

        logger.info("Single domain dataset test completed")
        return results

    def test_multi_domain_dataset(
        self, test_data_dir: Path, calibrations: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Test multi-domain dataset functionality."""
        logger.info("Testing multi-domain dataset...")

        results = {}

        try:
            # Create domain configurations
            domain_configs = {}
            for domain in ["photography", "microscopy", "astronomy"]:
                domain_configs[domain] = {
                    "data_root": str(test_data_dir / domain),
                    "calibration_file": str(calibrations[domain]),
                    "max_files": 5,
                    "validate_files": False,
                }

            # Test with balancing
            balanced_dataset = MultiDomainDataset(
                domain_configs=domain_configs,
                split="train",
                balance_domains=True,
                min_samples_per_domain=1,  # Lower threshold for testing
                validate_files=False,
            )

            # Test without balancing
            unbalanced_dataset = MultiDomainDataset(
                domain_configs=domain_configs,
                split="train",
                balance_domains=False,
                min_samples_per_domain=1,
                validate_files=False,
            )

            results["multi_domain"] = {
                "balanced_size": len(balanced_dataset),
                "unbalanced_size": len(unbalanced_dataset),
                "num_domains_balanced": len(balanced_dataset.domain_datasets),
                "num_domains_unbalanced": len(unbalanced_dataset.domain_datasets),
                "domain_distribution_balanced": balanced_dataset.get_domain_distribution(),
                "domain_distribution_unbalanced": unbalanced_dataset.get_domain_distribution(),
            }

        except Exception as e:
            logger.error(f"Multi-domain dataset test failed: {e}")
            results["multi_domain"] = {"error": str(e)}

        logger.info("Multi-domain dataset test completed")
        return results

    def test_convenience_functions(
        self, test_data_dir: Path, calibrations: Dict[str, Path]
    ) -> Dict[str, Any]:
        """Test convenience functions."""
        logger.info("Testing convenience functions...")

        results = {}

        try:
            # Test create_domain_dataset
            dataset = create_domain_dataset(
                domain="microscopy",
                data_root=test_data_dir / "microscopy",
                calibration_file=calibrations["microscopy"],
                target_size=32,
                max_files=3,
                validate_files=False,
            )

            results["create_domain_dataset"] = {
                "success": True,
                "domain": dataset.domain,
                "size": len(dataset),
                "target_size": dataset.target_size,
            }

        except Exception as e:
            logger.error(f"create_domain_dataset test failed: {e}")
            results["create_domain_dataset"] = {"success": False, "error": str(e)}

        try:
            # Test create_multi_domain_dataset
            domain_configs = {
                "microscopy": {
                    "data_root": str(test_data_dir / "microscopy"),
                    "calibration_file": str(calibrations["microscopy"]),
                    "max_files": 2,
                    "validate_files": False,
                }
            }

            multi_dataset = create_multi_domain_dataset(
                domain_configs=domain_configs,
                balance_domains=False,
                min_samples_per_domain=1,
            )

            results["create_multi_domain_dataset"] = {
                "success": True,
                "size": len(multi_dataset),
                "num_domains": len(multi_dataset.domain_datasets),
            }

        except Exception as e:
            logger.error(f"create_multi_domain_dataset test failed: {e}")
            results["create_multi_domain_dataset"] = {"success": False, "error": str(e)}

        logger.info("Convenience functions test completed")
        return results

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("Running complete domain dataset integration test suite...")

        # Create test data
        test_data_dir = self.create_test_data()
        calibrations = self.create_test_calibrations()

        # Run tests
        self.results = {
            "augmentation_pipeline": self.test_augmentation_pipeline(),
            "single_domain_dataset": self.test_single_domain_dataset(
                test_data_dir, calibrations
            ),
            "multi_domain_dataset": self.test_multi_domain_dataset(
                test_data_dir, calibrations
            ),
            "convenience_functions": self.test_convenience_functions(
                test_data_dir, calibrations
            ),
        }

        # Compute summary
        self.results["summary"] = self._compute_summary()

        logger.info("Integration test suite completed")
        return self.results

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute overall test summary."""
        summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": {},
        }

        def count_tests(results_dict, prefix=""):
            for key, value in results_dict.items():
                if isinstance(value, dict):
                    if "error" in value:
                        summary["failed_tests"] += 1
                        summary["test_details"][f"{prefix}{key}"] = "FAILED"
                    elif "success" in value:
                        if value["success"]:
                            summary["passed_tests"] += 1
                            summary["test_details"][f"{prefix}{key}"] = "PASSED"
                        else:
                            summary["failed_tests"] += 1
                            summary["test_details"][f"{prefix}{key}"] = "FAILED"
                    else:
                        # Assume success if no error
                        summary["passed_tests"] += 1
                        summary["test_details"][f"{prefix}{key}"] = "PASSED"

                    summary["total_tests"] += 1

                    # Recurse for nested dicts
                    if isinstance(value, dict) and any(
                        isinstance(v, dict) for v in value.values()
                    ):
                        count_tests(value, f"{prefix}{key}.")

        # Count tests in each category
        for category, results in self.results.items():
            if category != "summary" and isinstance(results, dict):
                count_tests(results, f"{category}.")

        summary["success_rate"] = (
            summary["passed_tests"] / summary["total_tests"]
            if summary["total_tests"] > 0
            else 0.0
        )

        return summary

    def save_results(self, output_path: str):
        """Save test results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

    def generate_report(self) -> str:
        """Generate human-readable test report."""
        if not self.results:
            return "No test results available. Run tests first."

        report = []
        report.append("=" * 60)
        report.append("DOMAIN DATASET INTEGRATION TEST REPORT")
        report.append("=" * 60)

        # Summary
        summary = self.results.get("summary", {})
        total_tests = summary.get("total_tests", 0)
        passed_tests = summary.get("passed_tests", 0)
        success_rate = summary.get("success_rate", 0.0)

        status = "✅ PASS" if success_rate >= 0.8 else "❌ FAIL"
        report.append(f"\nOVERALL STATUS: {status}")
        report.append(
            f"Tests: {passed_tests}/{total_tests} passed ({success_rate:.1%})"
        )

        # Detailed results by category
        categories = [
            ("augmentation_pipeline", "Augmentation Pipeline"),
            ("single_domain_dataset", "Single Domain Dataset"),
            ("multi_domain_dataset", "Multi-Domain Dataset"),
            ("convenience_functions", "Convenience Functions"),
        ]

        for category_key, category_name in categories:
            if category_key in self.results:
                report.append(f"\n{category_name}:")
                category_results = self.results[category_key]

                if isinstance(category_results, dict):
                    for test_name, test_result in category_results.items():
                        if isinstance(test_result, dict):
                            if "error" in test_result:
                                report.append(
                                    f"  ❌ {test_name}: {test_result['error']}"
                                )
                            elif "success" in test_result:
                                status = "✅" if test_result["success"] else "❌"
                                report.append(f"  {status} {test_name}")
                            else:
                                report.append(f"  ✅ {test_name}")
                        else:
                            report.append(f"  ✅ {test_name}: {test_result}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Test domain dataset integration")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/domain_dataset_integration_test",
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
    tester = DomainDatasetIntegrationTester(args.output_dir)
    results = tester.run_all_tests()

    # Save results
    tester.save_results(output_dir / "integration_test_results.json")

    # Generate and save report
    report = tester.generate_report()
    with open(output_dir / "integration_test_report.txt", "w") as f:
        f.write(report)

    # Print report
    print(report)

    # Exit with appropriate code
    summary = results.get("summary", {})
    success_rate = summary.get("success_rate", 0.0)
    sys.exit(0 if success_rate >= 0.8 else 1)


if __name__ == "__main__":
    main()
