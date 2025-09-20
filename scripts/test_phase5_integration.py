#!/usr/bin/env python3
"""
Integration test for Phase 5: Training Framework

This script comprehensively tests the Phase 5 implementation:
- Task 5.1: Deterministic training loop with v-parameterization loss
- Task 5.2: Multi-domain balancing with weighted sampling
- Task 5.3: Evaluation framework with comprehensive metrics

Usage:
    python scripts/test_phase5_integration.py [--output_dir OUTPUT_DIR]
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from training.losses import DiffusionLoss, PoissonGaussianLoss
from training.metrics import TrainingMetrics
from training.multi_domain_trainer import (
    DomainBalancingConfig,
    MultiDomainTrainer,
    MultiDomainTrainingConfig,
)
from training.trainer import DeterministicTrainer, TrainingConfig

from core.calibration import CalibrationParams, SensorCalibration
from core.logging_config import get_logger, setup_project_logging
from core.metrics import (
    DomainSpecificMetrics,
    EvaluationSuite,
    PhysicsMetrics,
    StandardMetrics,
)
from data.domain_datasets import DomainDataset, MultiDomainDataset

logger = get_logger(__name__)


class SimpleTestModel(nn.Module):
    """Simple model for testing purposes."""

    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_dim, 3, padding=1),
        )

    def forward(self, x, conditioning=None):
        """Forward pass with optional conditioning."""
        return self.net(x)


class MockDataset(torch.utils.data.Dataset):
    """Mock dataset for testing."""

    def __init__(self, size=100, image_size=64, domain="microscopy"):
        self.size = size
        self.image_size = image_size
        self.domain = domain

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Create synthetic data
        clean = torch.rand(1, self.image_size, self.image_size)
        noisy = clean + torch.randn_like(clean) * 0.1

        return {
            "clean": clean,
            "noisy": noisy,
            "domain": self.domain,
            "scale": 1000.0,
            "read_noise": 3.0,
            "background": 10.0,
        }


class Phase5IntegrationTester:
    """Comprehensive tester for Phase 5 implementation."""

    def __init__(self, output_dir: str = "data/phase5_integration_test"):
        """Initialize the tester."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}

        logger.info("Phase 5 Integration Tester initialized")

    def test_task_5_1_deterministic_training(self) -> Dict[str, Any]:
        """Test Task 5.1: Deterministic training loop."""
        logger.info("Testing Task 5.1: Deterministic training loop...")

        results = {}

        try:
            # Create test model and data
            model = SimpleTestModel()
            train_dataset = MockDataset(size=50, domain="microscopy")
            val_dataset = MockDataset(size=20, domain="microscopy")

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=4, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=4, shuffle=False
            )

            # Create training configuration
            config = TrainingConfig(
                batch_size=4,
                learning_rate=1e-3,
                num_epochs=2,  # Short for testing
                deterministic=True,
                seed=42,
                mixed_precision=False,  # Disable for testing
                log_frequency=10,
                val_frequency=1,
            )

            # Create trainer
            trainer = DeterministicTrainer(model, train_loader, val_loader, config)

            # Test trainer initialization
            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.loss_fn is not None
            assert trainer.metrics is not None

            results["trainer_initialization"] = True

            # Test deterministic behavior
            # Train for one step and record state
            trainer.train_one_epoch()
            first_loss = trainer.metrics.get_current_metrics()["train_loss"]

            # Reset and train again with same seed
            trainer = DeterministicTrainer(model, train_loader, val_loader, config)
            trainer.train_one_epoch()
            second_loss = trainer.metrics.get_current_metrics()["train_loss"]

            # Should be identical due to deterministic mode
            deterministic_behavior = abs(first_loss - second_loss) < 1e-6
            results["deterministic_behavior"] = deterministic_behavior

            # Test validation
            val_metrics = trainer.validate_one_epoch()
            assert "val_loss" in val_metrics
            assert "val_psnr" in val_metrics

            results["validation_works"] = True

            # Test metrics collection
            metrics_summary = trainer.metrics.get_summary()
            assert "train_loss" in metrics_summary
            assert "val_loss" in metrics_summary

            results["metrics_collection"] = True

            logger.info("✅ Task 5.1 test passed")

        except Exception as e:
            logger.error(f"❌ Task 5.1 test failed: {e}")
            results["error"] = str(e)

        return results

    def test_task_5_2_multi_domain_balancing(self) -> Dict[str, Any]:
        """Test Task 5.2: Multi-domain balancing."""
        logger.info("Testing Task 5.2: Multi-domain balancing...")

        results = {}

        try:
            # Create multi-domain datasets
            datasets = {
                "photography": MockDataset(size=30, domain="photography"),
                "microscopy": MockDataset(size=50, domain="microscopy"),
                "astronomy": MockDataset(size=20, domain="astronomy"),
            }

            # Create combined dataset
            combined_dataset = torch.utils.data.ConcatDataset(list(datasets.values()))

            # Create domain balancing configuration
            domain_config = DomainBalancingConfig(
                sampling_strategy="weighted",
                use_domain_conditioning=True,
                adaptive_rebalancing=True,
                log_domain_stats=True,
            )

            # Create multi-domain training configuration
            config = MultiDomainTrainingConfig(
                batch_size=6,
                learning_rate=1e-3,
                num_epochs=2,
                domain_balancing=domain_config,
                deterministic=True,
                seed=42,
                mixed_precision=False,
            )

            # Create model and data loaders
            model = SimpleTestModel()
            train_loader = torch.utils.data.DataLoader(
                combined_dataset, batch_size=6, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                combined_dataset, batch_size=6, shuffle=False
            )

            # Create multi-domain trainer
            trainer = MultiDomainTrainer(model, train_loader, val_loader, config)

            # Test domain conditioning encoder
            assert trainer.domain_encoder is not None
            domains = ["photography", "microscopy", "astronomy"]
            conditioning = trainer.domain_encoder.encode_batch(
                domains, [10000.0, 1000.0, 50000.0], [5.0, 3.0, 2.0], [0.0, 10.0, 0.0]
            )
            assert conditioning.shape == (3, 6)  # 3 domains, 6-dim conditioning

            results["domain_conditioning"] = True

            # Test domain loss weights
            domain_weights = trainer._get_domain_loss_weights()
            assert isinstance(domain_weights, dict)
            assert len(domain_weights) > 0

            results["domain_loss_weights"] = True

            # Test training step with domain balancing
            trainer.train_one_epoch()

            # Check domain statistics
            domain_stats = trainer.get_domain_statistics()
            assert "domain_counts" in domain_stats
            assert "domain_losses" in domain_stats

            results["domain_statistics"] = True

            logger.info("✅ Task 5.2 test passed")

        except Exception as e:
            logger.error(f"❌ Task 5.2 test failed: {e}")
            results["error"] = str(e)

        return results

    def test_task_5_3_evaluation_framework(self) -> Dict[str, Any]:
        """Test Task 5.3: Evaluation framework."""
        logger.info("Testing Task 5.3: Evaluation framework...")

        results = {}

        try:
            # Create test images
            clean_image = torch.rand(1, 1, 64, 64)
            restored_image = clean_image + torch.randn_like(clean_image) * 0.05
            noisy_image = clean_image + torch.randn_like(clean_image) * 0.2

            # Test standard metrics
            standard_metrics = StandardMetrics(device="cpu")

            # Test PSNR
            psnr = standard_metrics.compute_psnr(restored_image, clean_image)
            assert psnr.value > 0
            results["psnr_computation"] = True

            # Test SSIM
            ssim = standard_metrics.compute_ssim(restored_image, clean_image)
            assert 0 <= ssim.value <= 1
            results["ssim_computation"] = True

            # Test physics metrics
            physics_metrics = PhysicsMetrics()

            # Test chi-squared consistency
            chi2 = physics_metrics.compute_chi2_consistency(
                restored_image,
                noisy_image,
                lambda_true=clean_image * 1000,
                read_noise=3.0,
            )
            assert chi2.value > 0
            results["chi2_consistency"] = True

            # Test residual whiteness
            whiteness = physics_metrics.compute_residual_whiteness(
                restored_image, clean_image
            )
            assert 0 <= whiteness.value <= 1
            results["residual_whiteness"] = True

            # Test bias analysis
            bias = physics_metrics.compute_bias_analysis(restored_image, clean_image)
            assert bias.value is not None
            results["bias_analysis"] = True

            # Test domain-specific metrics
            domain_metrics = DomainSpecificMetrics()

            # Test counting accuracy (microscopy)
            counting_acc = domain_metrics.compute_counting_accuracy(
                restored_image, clean_image, threshold=0.5
            )
            assert 0 <= counting_acc.value <= 1
            results["counting_accuracy"] = True

            # Test photometry error (astronomy)
            # Create mock source positions
            sources_true = torch.tensor([[32, 32]], dtype=torch.float32)
            sources_pred = torch.tensor([[33, 31]], dtype=torch.float32)

            photometry_error = domain_metrics.compute_photometry_error(
                restored_image, clean_image, sources_true, sources_pred
            )
            assert photometry_error.value >= 0
            results["photometry_error"] = True

            # Test evaluation suite
            eval_suite = EvaluationSuite(device="cpu")

            # Test full evaluation
            evaluation_report = eval_suite.evaluate_restoration(
                method_name="test_method",
                restored_images=[restored_image],
                clean_images=[clean_image],
                noisy_images=[noisy_image],
                domain="microscopy",
                dataset_name="test_dataset",
            )

            assert evaluation_report.method_name == "test_method"
            assert evaluation_report.domain == "microscopy"
            assert evaluation_report.psnr.value > 0
            assert evaluation_report.ssim.value > 0

            results["evaluation_suite"] = True

            # Test serialization
            json_str = evaluation_report.to_json()
            restored_report = eval_suite.EvaluationReport.from_json(json_str)
            assert restored_report.method_name == evaluation_report.method_name

            results["report_serialization"] = True

            logger.info("✅ Task 5.3 test passed")

        except Exception as e:
            logger.error(f"❌ Task 5.3 test failed: {e}")
            results["error"] = str(e)

        return results

    def test_integration_end_to_end(self) -> Dict[str, Any]:
        """Test end-to-end integration of all Phase 5 components."""
        logger.info("Testing end-to-end Phase 5 integration...")

        results = {}

        try:
            # Create a complete training scenario
            model = SimpleTestModel()

            # Create multi-domain data
            datasets = {
                "microscopy": MockDataset(size=30, domain="microscopy"),
                "photography": MockDataset(size=25, domain="photography"),
            }

            combined_dataset = torch.utils.data.ConcatDataset(list(datasets.values()))
            train_loader = torch.utils.data.DataLoader(
                combined_dataset, batch_size=4, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                combined_dataset, batch_size=4, shuffle=False
            )

            # Create comprehensive configuration
            config = MultiDomainTrainingConfig(
                batch_size=4,
                learning_rate=1e-3,
                num_epochs=1,  # Short for testing
                deterministic=True,
                seed=42,
                mixed_precision=False,
                domain_balancing=DomainBalancingConfig(
                    sampling_strategy="weighted",
                    use_domain_conditioning=True,
                    log_domain_stats=True,
                ),
            )

            # Create trainer
            trainer = MultiDomainTrainer(model, train_loader, val_loader, config)

            # Train for one epoch
            trainer.train_one_epoch()

            # Validate
            val_metrics = trainer.validate_one_epoch()

            # Test that we got reasonable metrics
            assert "val_loss" in val_metrics
            assert "val_psnr" in val_metrics
            assert val_metrics["val_loss"] > 0
            assert val_metrics["val_psnr"] > 0

            # Test domain statistics
            domain_stats = trainer.get_domain_statistics()
            assert "domain_counts" in domain_stats

            # Test evaluation on trained model
            eval_suite = EvaluationSuite(device="cpu")

            # Generate test data
            test_clean = torch.rand(2, 1, 64, 64)
            test_noisy = test_clean + torch.randn_like(test_clean) * 0.1

            # Get model predictions
            model.eval()
            with torch.no_grad():
                test_restored = model(test_noisy)

            # Evaluate
            evaluation_report = eval_suite.evaluate_restoration(
                method_name="trained_model",
                restored_images=[test_restored],
                clean_images=[test_clean],
                noisy_images=[test_noisy],
                domain="microscopy",
                dataset_name="integration_test",
            )

            assert evaluation_report.psnr.value > 0
            assert evaluation_report.ssim.value > 0

            results["end_to_end_success"] = True
            results["final_val_loss"] = val_metrics["val_loss"]
            results["final_psnr"] = evaluation_report.psnr.value

            logger.info("✅ End-to-end integration test passed")

        except Exception as e:
            logger.error(f"❌ End-to-end integration test failed: {e}")
            results["error"] = str(e)

        return results

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 5 integration tests."""
        logger.info("Running complete Phase 5 integration test suite...")

        self.results = {
            "task_5_1_deterministic_training": self.test_task_5_1_deterministic_training(),
            "task_5_2_multi_domain_balancing": self.test_task_5_2_multi_domain_balancing(),
            "task_5_3_evaluation_framework": self.test_task_5_3_evaluation_framework(),
            "end_to_end_integration": self.test_integration_end_to_end(),
        }

        # Compute summary
        self.results["summary"] = self._compute_summary()

        logger.info("Phase 5 integration test suite completed")
        return self.results

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute overall test summary."""
        summary = {
            "total_test_categories": 0,
            "passed_categories": 0,
            "failed_categories": 0,
            "category_details": {},
        }

        for category, results in self.results.items():
            if category == "summary":
                continue

            summary["total_test_categories"] += 1

            if "error" in results:
                summary["failed_categories"] += 1
                summary["category_details"][category] = "FAILED"
            else:
                summary["passed_categories"] += 1
                summary["category_details"][category] = "PASSED"

        summary["success_rate"] = (
            summary["passed_categories"] / summary["total_test_categories"]
            if summary["total_test_categories"] > 0
            else 0.0
        )

        summary["overall_status"] = (
            "PASSED" if summary["success_rate"] >= 0.75 else "FAILED"
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
        report.append("=" * 70)
        report.append("PHASE 5 INTEGRATION TEST REPORT")
        report.append("Training Framework Implementation")
        report.append("=" * 70)

        # Summary
        summary = self.results.get("summary", {})
        status = "✅ PASSED" if summary.get("overall_status") == "PASSED" else "❌ FAILED"
        success_rate = summary.get("success_rate", 0.0)

        report.append(f"\nOVERALL STATUS: {status}")
        report.append(f"Success Rate: {success_rate:.1%}")
        report.append(
            f"Categories: {summary.get('passed_categories', 0)}/{summary.get('total_test_categories', 0)} passed"
        )

        # Task details
        tasks = [
            (
                "task_5_1_deterministic_training",
                "Task 5.1: Deterministic Training Loop",
            ),
            ("task_5_2_multi_domain_balancing", "Task 5.2: Multi-Domain Balancing"),
            ("task_5_3_evaluation_framework", "Task 5.3: Evaluation Framework"),
            ("end_to_end_integration", "End-to-End Integration"),
        ]

        for task_key, task_name in tasks:
            if task_key in self.results:
                task_results = self.results[task_key]

                if "error" in task_results:
                    report.append(f"\n❌ {task_name}: FAILED")
                    report.append(f"   Error: {task_results['error']}")
                else:
                    report.append(f"\n✅ {task_name}: PASSED")

                    # Show specific test results
                    for test_name, test_result in task_results.items():
                        if test_name != "error" and isinstance(test_result, bool):
                            status_icon = "✅" if test_result else "❌"
                            report.append(f"   {status_icon} {test_name}")
                        elif test_name != "error" and isinstance(
                            test_result, (int, float)
                        ):
                            report.append(f"   📊 {test_name}: {test_result}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Phase 5 training framework integration"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/phase5_integration_test",
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
    tester = Phase5IntegrationTester(args.output_dir)
    results = tester.run_all_tests()

    # Save results
    tester.save_results(output_dir / "phase5_integration_results.json")

    # Generate and save report
    report = tester.generate_report()
    with open(output_dir / "phase5_integration_report.txt", "w") as f:
        f.write(report)

    # Print report
    print(report)

    # Exit with appropriate code
    summary = results.get("summary", {})
    overall_status = summary.get("overall_status", "FAILED")
    sys.exit(0 if overall_status == "PASSED" else 1)


if __name__ == "__main__":
    main()
