#!/usr/bin/env python3
"""
Validation script for Task 4.2: Create unified DomainDataset

This script validates that all requirements for task 4.2 are met:
- Implement base dataset class with reversible transforms integration
- Add train/validation/test splitting with deterministic seeding
- Include geometric augmentation pipeline (flips, rotations)
- Verify data loading correctness and transform consistency

Usage:
    python scripts/validate_task_4_2.py [--output_dir OUTPUT_DIR]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.calibration import CalibrationParams
from core.logging_config import get_logger, setup_project_logging
from data.augmentations import AugmentationConfig, GeometricAugmentationPipeline
from data.domain_datasets import DomainDataset, MultiDomainDataset

logger = get_logger(__name__)


class Task42Validator:
    """
    Validator for Task 4.2 requirements.

    Validates all specified requirements:
    1. Base dataset class with reversible transforms integration
    2. Train/validation/test splitting with deterministic seeding
    3. Geometric augmentation pipeline (flips, rotations)
    4. Data loading correctness and transform consistency
    """

    def __init__(self, output_dir: str = "data/task_4_2_validation"):
        """Initialize the validator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.validation_results = {}

        logger.info("Task 4.2 Validator initialized")

    def create_minimal_test_setup(self) -> Dict[str, Any]:
        """Create minimal test setup for validation."""
        test_dir = self.output_dir / "test_setup"
        test_dir.mkdir(exist_ok=True)

        # Create microscopy test directory
        micro_dir = test_dir / "microscopy"
        micro_dir.mkdir(exist_ok=True)

        # Create test files
        for i in range(12):  # Enough for proper splitting
            (micro_dir / f"test_{i:03d}.tif").touch()

        # Create calibration file
        cal_file = test_dir / "microscopy_cal.json"
        calibration = {
            "gain": 2.0,
            "black_level": 100,
            "white_level": 65535,
            "read_noise": 3.0,
            "pixel_size": 0.65,
            "pixel_unit": "um",
            "domain": "microscopy",
        }

        with open(cal_file, "w") as f:
            json.dump(calibration, f)

        return {"data_root": micro_dir, "calibration_file": cal_file, "num_files": 12}

    def validate_requirement_1_base_dataset_class(
        self, test_setup: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Requirement 1: Implement base dataset class with reversible transforms integration
        """
        logger.info(
            "Validating Requirement 1: Base dataset class with reversible transforms integration"
        )

        results = {
            "requirement": "Base dataset class with reversible transforms integration",
            "tests": {},
        }

        try:
            # Test 1.1: Dataset initialization
            dataset = DomainDataset(
                data_root=test_setup["data_root"],
                domain="microscopy",
                calibration_file=test_setup["calibration_file"],
                target_size=64,
                validate_files=False,
            )

            results["tests"]["dataset_initialization"] = {
                "passed": True,
                "details": f"Successfully initialized dataset with {len(dataset)} files",
            }

            # Test 1.2: Reversible transform integration
            has_transform = (
                hasattr(dataset, "transform") and dataset.transform is not None
            )
            results["tests"]["reversible_transform_integration"] = {
                "passed": has_transform,
                "details": f"Transform object present: {has_transform}",
            }

            # Test 1.3: Domain configuration
            has_domain_config = (
                hasattr(dataset, "config") and dataset.config is not None
            )
            results["tests"]["domain_configuration"] = {
                "passed": has_domain_config,
                "details": f"Domain config present: {has_domain_config}, domain: {dataset.domain}",
            }

            # Test 1.4: Calibration integration
            has_calibration = (
                hasattr(dataset, "calibration") and dataset.calibration is not None
            )
            results["tests"]["calibration_integration"] = {
                "passed": has_calibration,
                "details": f"Calibration present: {has_calibration}",
            }

        except Exception as e:
            results["tests"]["dataset_initialization"] = {
                "passed": False,
                "details": f"Failed to initialize dataset: {str(e)}",
            }

        # Overall requirement status
        all_tests_passed = all(test["passed"] for test in results["tests"].values())
        results["passed"] = all_tests_passed

        logger.info(
            f"Requirement 1 validation: {'PASSED' if all_tests_passed else 'FAILED'}"
        )
        return results

    def validate_requirement_2_deterministic_splitting(
        self, test_setup: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Requirement 2: Add train/validation/test splitting with deterministic seeding
        """
        logger.info(
            "Validating Requirement 2: Train/validation/test splitting with deterministic seeding"
        )

        results = {
            "requirement": "Train/validation/test splitting with deterministic seeding",
            "tests": {},
        }

        try:
            # Test 2.1: Create datasets for each split
            splits = ["train", "val", "test"]
            split_ratios = (0.6, 0.2, 0.2)
            seed = 42

            datasets = {}
            for split in splits:
                datasets[split] = DomainDataset(
                    data_root=test_setup["data_root"],
                    domain="microscopy",
                    calibration_file=test_setup["calibration_file"],
                    split=split,
                    split_ratios=split_ratios,
                    seed=seed,
                    validate_files=False,
                )

            # Check that all splits are created
            splits_created = all(len(datasets[split]) > 0 for split in splits)
            results["tests"]["all_splits_created"] = {
                "passed": splits_created,
                "details": f"Split sizes: {[(split, len(datasets[split])) for split in splits]}",
            }

            # Test 2.2: Check split ratios are approximately correct
            total_files = sum(len(datasets[split]) for split in splits)
            actual_ratios = [len(datasets[split]) / total_files for split in splits]
            ratio_tolerance = 0.15  # Allow some tolerance due to integer division

            ratios_correct = all(
                abs(actual - expected) <= ratio_tolerance
                for actual, expected in zip(actual_ratios, split_ratios)
            )

            results["tests"]["split_ratios_correct"] = {
                "passed": ratios_correct,
                "details": f"Expected: {split_ratios}, Actual: {[round(r, 2) for r in actual_ratios]}",
            }

            # Test 2.3: Deterministic seeding
            # Create two datasets with same seed
            dataset1 = DomainDataset(
                data_root=test_setup["data_root"],
                domain="microscopy",
                calibration_file=test_setup["calibration_file"],
                split="train",
                seed=123,
                validate_files=False,
            )

            dataset2 = DomainDataset(
                data_root=test_setup["data_root"],
                domain="microscopy",
                calibration_file=test_setup["calibration_file"],
                split="train",
                seed=123,
                validate_files=False,
            )

            deterministic = dataset1.file_paths == dataset2.file_paths
            results["tests"]["deterministic_seeding"] = {
                "passed": deterministic,
                "details": f"Same seed produces same file order: {deterministic}",
            }

            # Test 2.4: Different seeds produce different results
            dataset3 = DomainDataset(
                data_root=test_setup["data_root"],
                domain="microscopy",
                calibration_file=test_setup["calibration_file"],
                split="train",
                seed=456,
                validate_files=False,
            )

            different_seeds_different = dataset1.file_paths != dataset3.file_paths
            results["tests"]["different_seeds_different_results"] = {
                "passed": different_seeds_different,
                "details": f"Different seeds produce different file order: {different_seeds_different}",
            }

        except Exception as e:
            results["tests"]["splitting_error"] = {
                "passed": False,
                "details": f"Error in splitting validation: {str(e)}",
            }

        # Overall requirement status
        all_tests_passed = all(test["passed"] for test in results["tests"].values())
        results["passed"] = all_tests_passed

        logger.info(
            f"Requirement 2 validation: {'PASSED' if all_tests_passed else 'FAILED'}"
        )
        return results

    def validate_requirement_3_geometric_augmentations(
        self, test_setup: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Requirement 3: Include geometric augmentation pipeline (flips, rotations)
        """
        logger.info(
            "Validating Requirement 3: Geometric augmentation pipeline (flips, rotations)"
        )

        results = {
            "requirement": "Geometric augmentation pipeline (flips, rotations)",
            "tests": {},
        }

        try:
            # Test 3.1: Dataset has augmentation pipeline
            dataset = DomainDataset(
                data_root=test_setup["data_root"],
                domain="microscopy",
                calibration_file=test_setup["calibration_file"],
                split="train",
                enable_augmentations=True,
                validate_files=False,
            )

            has_augmentation_pipeline = (
                hasattr(dataset, "augmentation_pipeline")
                and dataset.augmentation_pipeline is not None
            )

            results["tests"]["augmentation_pipeline_present"] = {
                "passed": has_augmentation_pipeline,
                "details": f"Augmentation pipeline present: {has_augmentation_pipeline}",
            }

            # Test 3.2: Augmentation pipeline is correct type
            if has_augmentation_pipeline:
                is_correct_type = isinstance(
                    dataset.augmentation_pipeline, GeometricAugmentationPipeline
                )
                results["tests"]["correct_augmentation_type"] = {
                    "passed": is_correct_type,
                    "details": f"Correct augmentation type: {type(dataset.augmentation_pipeline)}",
                }

            # Test 3.3: Test individual augmentations
            config = AugmentationConfig(
                enable_rotation=True,
                enable_horizontal_flip=True,
                enable_vertical_flip=True,
                deterministic=True,
                seed=42,
            )

            pipeline = GeometricAugmentationPipeline(config)

            # Test with sample data
            sample_image = torch.rand(1, 32, 32)
            sample_mask = torch.ones(1, 32, 32)

            # Test rotation
            aug_image, aug_mask, aug_info = pipeline(sample_image, sample_mask)

            rotation_works = (
                "rotation_angle" in aug_info and aug_image.shape == sample_image.shape
            )

            results["tests"]["rotation_augmentation"] = {
                "passed": rotation_works,
                "details": f"Rotation augmentation works: {rotation_works}, info: {aug_info.get('rotation_angle', 'N/A')}",
            }

            # Test flipping
            flip_works = "horizontal_flip" in aug_info and "vertical_flip" in aug_info

            results["tests"]["flip_augmentation"] = {
                "passed": flip_works,
                "details": f"Flip augmentation works: {flip_works}",
            }

            # Test 3.4: Validation dataset has minimal augmentations
            val_dataset = DomainDataset(
                data_root=test_setup["data_root"],
                domain="microscopy",
                calibration_file=test_setup["calibration_file"],
                split="val",
                enable_augmentations=True,
                validate_files=False,
            )

            val_has_pipeline = (
                hasattr(val_dataset, "augmentation_pipeline")
                and val_dataset.augmentation_pipeline is not None
            )

            results["tests"]["validation_minimal_augmentations"] = {
                "passed": val_has_pipeline,
                "details": f"Validation dataset has augmentation pipeline: {val_has_pipeline}",
            }

        except Exception as e:
            results["tests"]["augmentation_error"] = {
                "passed": False,
                "details": f"Error in augmentation validation: {str(e)}",
            }

        # Overall requirement status
        all_tests_passed = all(test["passed"] for test in results["tests"].values())
        results["passed"] = all_tests_passed

        logger.info(
            f"Requirement 3 validation: {'PASSED' if all_tests_passed else 'FAILED'}"
        )
        return results

    def validate_requirement_4_data_loading_consistency(
        self, test_setup: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Requirement 4: Verify data loading correctness and transform consistency
        """
        logger.info(
            "Validating Requirement 4: Data loading correctness and transform consistency"
        )

        results = {
            "requirement": "Data loading correctness and transform consistency",
            "tests": {},
        }

        try:
            # Test 4.1: Dataset item structure
            dataset = DomainDataset(
                data_root=test_setup["data_root"],
                domain="microscopy",
                calibration_file=test_setup["calibration_file"],
                target_size=64,
                validate_files=False,
                max_files=1,  # Just test one item
            )

            # Mock the loader to avoid actual file loading
            from unittest.mock import MagicMock

            import numpy as np

            from data.loaders import ImageMetadata

            mock_loader = MagicMock()
            mock_loader.load_with_validation.return_value = (
                np.random.rand(32, 32).astype(np.float32),
                ImageMetadata(
                    filepath="test.tif",
                    format=".tif",
                    domain="microscopy",
                    height=32,
                    width=32,
                    channels=1,
                    bit_depth=16,
                    dtype="uint16",
                ),
            )
            dataset.loader = mock_loader

            # Mock calibration
            mock_calibration = MagicMock()
            mock_calibration.process_raw.return_value = (
                np.random.rand(32, 32).astype(np.float32),
                np.ones((32, 32), dtype=np.float32),
            )
            mock_calibration.params = CalibrationParams(
                gain=2.0,
                black_level=100,
                white_level=65535,
                read_noise=3.0,
                pixel_size=0.65,
                pixel_unit="um",
                domain="microscopy",
            )
            dataset.calibration = mock_calibration

            # Mock transform
            mock_transform = MagicMock()
            mock_transform.forward.return_value = (
                torch.rand(1, 64, 64),
                {"target_size": 64},
            )
            dataset.transform = mock_transform

            # Test item loading
            try:
                item = dataset[0]

                # Check required fields
                required_fields = [
                    "raw_adu",
                    "electrons",
                    "normalized",
                    "transformed",
                    "mask",
                    "original_mask",
                    "image_metadata",
                    "transform_metadata",
                    "augmentation_info",
                    "calibration_params",
                    "scale",
                    "file_path",
                    "split",
                    "domain",
                ]

                missing_fields = [
                    field for field in required_fields if field not in item
                ]

                results["tests"]["item_structure_complete"] = {
                    "passed": len(missing_fields) == 0,
                    "details": f"Missing fields: {missing_fields}"
                    if missing_fields
                    else "All required fields present",
                }

                # Check data types
                type_checks = {
                    "raw_adu": np.ndarray,
                    "electrons": np.ndarray,
                    "normalized": torch.Tensor,
                    "transformed": torch.Tensor,
                    "mask": torch.Tensor,
                    "augmentation_info": dict,
                    "domain": str,
                    "split": str,
                }

                type_errors = []
                for field, expected_type in type_checks.items():
                    if field in item and not isinstance(item[field], expected_type):
                        type_errors.append(
                            f"{field}: expected {expected_type}, got {type(item[field])}"
                        )

                results["tests"]["data_types_correct"] = {
                    "passed": len(type_errors) == 0,
                    "details": f"Type errors: {type_errors}"
                    if type_errors
                    else "All data types correct",
                }

                # Check tensor shapes
                if "transformed" in item and isinstance(
                    item["transformed"], torch.Tensor
                ):
                    shape_correct = item["transformed"].shape[-2:] == (
                        64,
                        64,
                    )  # Target size
                    results["tests"]["transform_shape_correct"] = {
                        "passed": shape_correct,
                        "details": f"Transformed shape: {item['transformed'].shape}, expected: (..., 64, 64)",
                    }

            except Exception as e:
                results["tests"]["item_loading_error"] = {
                    "passed": False,
                    "details": f"Error loading dataset item: {str(e)}",
                }

            # Test 4.2: Multi-domain dataset
            try:
                domain_configs = {
                    "microscopy": {
                        "data_root": str(test_setup["data_root"]),
                        "calibration_file": str(test_setup["calibration_file"]),
                        "max_files": 2,
                    }
                }

                multi_dataset = MultiDomainDataset(
                    domain_configs=domain_configs,
                    validate_files=False,
                    min_samples_per_domain=1,
                )

                multi_dataset_works = len(multi_dataset) > 0
                results["tests"]["multi_domain_dataset"] = {
                    "passed": multi_dataset_works,
                    "details": f"Multi-domain dataset created with {len(multi_dataset)} samples",
                }

            except Exception as e:
                results["tests"]["multi_domain_dataset"] = {
                    "passed": False,
                    "details": f"Multi-domain dataset error: {str(e)}",
                }

        except Exception as e:
            results["tests"]["data_loading_error"] = {
                "passed": False,
                "details": f"Error in data loading validation: {str(e)}",
            }

        # Overall requirement status
        all_tests_passed = all(test["passed"] for test in results["tests"].values())
        results["passed"] = all_tests_passed

        logger.info(
            f"Requirement 4 validation: {'PASSED' if all_tests_passed else 'FAILED'}"
        )
        return results

    def run_validation(self) -> Dict[str, Any]:
        """Run complete Task 4.2 validation."""
        logger.info("Running Task 4.2 validation...")

        # Create test setup
        test_setup = self.create_minimal_test_setup()

        # Run all requirement validations
        self.validation_results = {
            "task": "Task 4.2: Create unified DomainDataset",
            "requirements": {
                "requirement_1": self.validate_requirement_1_base_dataset_class(
                    test_setup
                ),
                "requirement_2": self.validate_requirement_2_deterministic_splitting(
                    test_setup
                ),
                "requirement_3": self.validate_requirement_3_geometric_augmentations(
                    test_setup
                ),
                "requirement_4": self.validate_requirement_4_data_loading_consistency(
                    test_setup
                ),
            },
        }

        # Compute overall status
        all_requirements_passed = all(
            req["passed"] for req in self.validation_results["requirements"].values()
        )

        self.validation_results["overall_status"] = {
            "passed": all_requirements_passed,
            "summary": f"{'PASSED' if all_requirements_passed else 'FAILED'} - Task 4.2 validation",
        }

        logger.info(
            f"Task 4.2 validation completed: {'PASSED' if all_requirements_passed else 'FAILED'}"
        )
        return self.validation_results

    def save_results(self, output_path: str):
        """Save validation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        logger.info(f"Validation results saved to {output_path}")

    def generate_report(self) -> str:
        """Generate human-readable validation report."""
        if not self.validation_results:
            return "No validation results available. Run validation first."

        report = []
        report.append("=" * 70)
        report.append("TASK 4.2 VALIDATION REPORT")
        report.append("Create unified DomainDataset")
        report.append("=" * 70)

        # Overall status
        overall = self.validation_results.get("overall_status", {})
        status = "✅ PASSED" if overall.get("passed", False) else "❌ FAILED"
        report.append(f"\nOVERALL STATUS: {status}")

        # Requirements breakdown
        requirements = self.validation_results.get("requirements", {})

        for req_key, req_data in requirements.items():
            req_num = req_key.replace("requirement_", "")
            req_name = req_data.get("requirement", f"Requirement {req_num}")
            req_status = "✅ PASSED" if req_data.get("passed", False) else "❌ FAILED"

            report.append(f"\n{req_status} Requirement {req_num}: {req_name}")

            # Test details
            tests = req_data.get("tests", {})
            for test_name, test_data in tests.items():
                test_status = "✅" if test_data.get("passed", False) else "❌"
                test_details = test_data.get("details", "")
                report.append(f"  {test_status} {test_name}: {test_details}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Validate Task 4.2 implementation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/task_4_2_validation",
        help="Output directory for validation results",
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
    validator = Task42Validator(args.output_dir)
    results = validator.run_validation()

    # Save results
    validator.save_results(output_dir / "task_4_2_validation_results.json")

    # Generate and save report
    report = validator.generate_report()
    with open(output_dir / "task_4_2_validation_report.txt", "w") as f:
        f.write(report)

    # Print report
    print(report)

    # Exit with appropriate code
    overall_passed = results.get("overall_status", {}).get("passed", False)
    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
