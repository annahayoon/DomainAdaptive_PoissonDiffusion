"""
Comprehensive tests for the evaluation metrics framework.

Tests all standard, physics, and domain-specific metrics to ensure
correctness and robustness.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from core.metrics import (
    DomainSpecificMetrics,
    EvaluationReport,
    EvaluationSuite,
    MetricResult,
    PhysicsMetrics,
    StandardMetrics,
    load_baseline_results,
    save_evaluation_results,
)


class TestStandardMetrics:
    """Test standard image quality metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"  # Use CPU for testing
        self.metrics = StandardMetrics(device=self.device)

        # Create test images
        self.pred = torch.rand(2, 1, 64, 64)
        self.target = torch.rand(2, 1, 64, 64)
        self.mask = torch.ones_like(self.pred)

    def test_compute_psnr_basic(self):
        """Test basic PSNR computation."""
        result = self.metrics.compute_psnr(self.pred, self.target)

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert result.value > 0  # PSNR should be positive
        assert "mse" in result.metadata
        assert "data_range" in result.metadata

    def test_compute_psnr_perfect_match(self):
        """Test PSNR with perfect reconstruction."""
        result = self.metrics.compute_psnr(self.target, self.target)

        assert result.value >= 50.0  # Should be very high for perfect match
        assert result.metadata["mse"] < 1e-10

    def test_compute_psnr_with_mask(self):
        """Test PSNR computation with mask."""
        # Create mask that excludes half the image
        mask = torch.zeros_like(self.pred)
        mask[:, :, :32, :] = 1.0

        result = self.metrics.compute_psnr(self.pred, self.target, mask=mask)

        assert isinstance(result.value, float)
        assert result.value > 0

    def test_compute_psnr_empty_mask(self):
        """Test PSNR with empty mask."""
        empty_mask = torch.zeros_like(self.pred)

        result = self.metrics.compute_psnr(self.pred, self.target, mask=empty_mask)

        assert result.value == 0.0
        assert "error" in result.metadata

    def test_compute_ssim_basic(self):
        """Test basic SSIM computation."""
        result = self.metrics.compute_ssim(self.pred, self.target)

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert -1 <= result.value <= 1  # SSIM can be negative for very different images
        assert "std" in result.metadata
        assert "num_channels" in result.metadata

    def test_compute_ssim_perfect_match(self):
        """Test SSIM with perfect reconstruction."""
        result = self.metrics.compute_ssim(self.target, self.target)

        assert result.value >= 0.99  # Should be very close to 1 for perfect match

    def test_compute_lpips_mock(self):
        """Test LPIPS computation with mocked network."""
        # Mock LPIPS network since it requires external dependency
        with patch("core.metrics.StandardMetrics._get_lpips_net") as mock_lpips:
            mock_net = MagicMock()
            mock_net.return_value = torch.tensor([[[[0.1]]]])
            mock_lpips.return_value = mock_net

            result = self.metrics.compute_lpips(self.pred, self.target)

            assert isinstance(result, MetricResult)
            assert isinstance(result.value, float)

    def test_compute_lpips_unavailable(self):
        """Test LPIPS when network is unavailable."""
        with patch("core.metrics.StandardMetrics._get_lpips_net", return_value=None):
            result = self.metrics.compute_lpips(self.pred, self.target)

            assert np.isnan(result.value)
            assert "error" in result.metadata

    def test_compute_ms_ssim_mock(self):
        """Test MS-SSIM computation with mocked library."""
        # Skip this test if pytorch_msssim is not available
        pytest.importorskip("pytorch_msssim")

        with patch("pytorch_msssim.ms_ssim") as mock_ms_ssim:
            mock_ms_ssim.return_value = torch.tensor(0.85)

            # Need larger images for MS-SSIM
            large_pred = torch.rand(1, 1, 160, 160)
            large_target = torch.rand(1, 1, 160, 160)

            result = self.metrics.compute_ms_ssim(large_pred, large_target)

            assert isinstance(result, MetricResult)
            assert isinstance(result.value, float)

    def test_shape_mismatch_error(self):
        """Test error handling for shape mismatches."""
        wrong_shape = torch.rand(2, 1, 32, 32)  # Different size

        with pytest.raises(ValueError, match="Shape mismatch"):
            self.metrics.compute_psnr(self.pred, wrong_shape)

        with pytest.raises(ValueError, match="Shape mismatch"):
            self.metrics.compute_ssim(self.pred, wrong_shape)


class TestPhysicsMetrics:
    """Test physics-specific metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = PhysicsMetrics()

        # Create test data with known physics
        self.scale = 1000.0
        self.background = 10.0
        self.read_noise = 5.0

        # Ground truth normalized image
        self.pred = torch.ones(1, 1, 32, 32) * 0.5  # 500 electrons

        # Generate Poisson-Gaussian noisy observation
        torch.manual_seed(42)
        lambda_e = self.pred * self.scale + self.background  # 510 electrons
        self.noisy = (
            torch.poisson(lambda_e) + torch.randn_like(lambda_e) * self.read_noise
        )

        self.mask = torch.ones_like(self.pred)

    def test_compute_chi2_consistency_perfect(self):
        """Test χ² consistency with perfect prediction."""
        result = self.metrics.compute_chi2_consistency(
            self.pred, self.noisy, self.scale, self.background, self.read_noise
        )

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert result.value > 0  # χ² should be positive
        assert "std" in result.metadata
        assert "median" in result.metadata
        assert "ks_statistic" in result.metadata
        assert "ks_pvalue" in result.metadata

    def test_compute_chi2_consistency_with_mask(self):
        """Test χ² consistency with mask."""
        # Create mask that excludes half the image
        mask = torch.zeros_like(self.pred)
        mask[:, :, :16, :] = 1.0

        result = self.metrics.compute_chi2_consistency(
            self.pred,
            self.noisy,
            self.scale,
            self.background,
            self.read_noise,
            mask=mask,
        )

        assert isinstance(result.value, float)
        assert result.value > 0

    def test_compute_chi2_consistency_empty_mask(self):
        """Test χ² consistency with empty mask."""
        empty_mask = torch.zeros_like(self.pred)

        result = self.metrics.compute_chi2_consistency(
            self.pred,
            self.noisy,
            self.scale,
            self.background,
            self.read_noise,
            mask=empty_mask,
        )

        assert np.isnan(result.value)
        assert "error" in result.metadata

    def test_compute_residual_whiteness(self):
        """Test residual whiteness computation."""
        result = self.metrics.compute_residual_whiteness(
            self.pred, self.noisy, self.scale, self.background
        )

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        # Whiteness might be NaN for small images, which is acceptable
        assert isinstance(result.value, float)
        # Check that either we have valid results or an error message
        if np.isnan(result.value):
            assert "error" in result.metadata
        else:
            assert "std" in result.metadata
            assert "num_samples" in result.metadata

    def test_compute_residual_whiteness_small_image(self):
        """Test residual whiteness with small image."""
        small_pred = torch.ones(1, 1, 8, 8) * 0.5
        small_noisy = torch.ones(1, 1, 8, 8) * 500

        result = self.metrics.compute_residual_whiteness(
            small_pred, small_noisy, self.scale, self.background
        )

        # Should handle small images gracefully
        assert isinstance(result, MetricResult)

    def test_compute_bias_analysis(self):
        """Test bias analysis computation."""
        result = self.metrics.compute_bias_analysis(
            self.pred, self.noisy, self.scale, self.background
        )

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert result.value >= 0  # Absolute relative bias should be non-negative
        assert "bias_electrons" in result.metadata
        assert "relative_bias_percent" in result.metadata
        assert "mean_signal" in result.metadata
        assert "t_statistic" in result.metadata
        assert "p_value" in result.metadata

    def test_compute_bias_analysis_with_confidence_interval(self):
        """Test bias analysis with confidence interval."""
        # Use larger image for better statistics
        large_pred = torch.ones(1, 1, 64, 64) * 0.5
        large_noisy = torch.ones(1, 1, 64, 64) * 500

        result = self.metrics.compute_bias_analysis(
            large_pred, large_noisy, self.scale, self.background
        )

        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.confidence_interval[1]


class TestDomainSpecificMetrics:
    """Test domain-specific metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = DomainSpecificMetrics()

        # Create test images with known features
        self.pred = torch.zeros(1, 1, 64, 64)
        self.target = torch.zeros(1, 1, 64, 64)

        # Add some "sources" to both images
        self.pred[0, 0, 20:25, 20:25] = 0.8
        self.pred[0, 0, 40:45, 40:45] = 0.6

        self.target[0, 0, 20:25, 20:25] = 0.9
        self.target[0, 0, 40:45, 40:45] = 0.7

        self.mask = torch.ones_like(self.pred)

    def test_compute_counting_accuracy(self):
        """Test counting accuracy for microscopy."""
        result = self.metrics.compute_counting_accuracy(
            self.pred, self.target, threshold=0.1
        )

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert 0 <= result.value <= 1  # Accuracy should be in [0, 1]
        assert "pred_count" in result.metadata
        assert "target_count" in result.metadata
        assert "count_error" in result.metadata
        assert "iou" in result.metadata

    def test_compute_counting_accuracy_perfect_match(self):
        """Test counting accuracy with perfect match."""
        result = self.metrics.compute_counting_accuracy(
            self.target, self.target, threshold=0.1
        )

        assert result.value == 1.0  # Perfect accuracy
        assert result.metadata["count_error"] == 0.0
        assert result.metadata["iou"] == 1.0

    def test_compute_counting_accuracy_with_mask(self):
        """Test counting accuracy with mask."""
        # Mask out one of the sources
        mask = torch.ones_like(self.pred)
        mask[:, :, 40:45, 40:45] = 0.0

        result = self.metrics.compute_counting_accuracy(
            self.pred, self.target, threshold=0.1, mask=mask
        )

        assert isinstance(result.value, float)
        assert 0 <= result.value <= 1

    def test_compute_photometry_error(self):
        """Test photometry error for astronomy."""
        source_positions = [(22, 22), (42, 42)]  # Centers of our test sources

        result = self.metrics.compute_photometry_error(
            self.pred, self.target, source_positions, aperture_radius=3
        )

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert result.value >= 0  # Error should be non-negative
        assert "std" in result.metadata
        assert "num_sources" in result.metadata
        assert "aperture_radius" in result.metadata
        assert "individual_errors" in result.metadata

    def test_compute_photometry_error_no_sources(self):
        """Test photometry error with no valid sources."""
        source_positions = [(100, 100)]  # Outside image bounds

        result = self.metrics.compute_photometry_error(
            self.pred, self.target, source_positions, aperture_radius=3
        )

        assert np.isnan(result.value)
        assert "error" in result.metadata

    def test_compute_photometry_error_batch_size_error(self):
        """Test photometry error with wrong batch size."""
        multi_batch = torch.rand(2, 1, 64, 64)
        source_positions = [(22, 22)]

        with pytest.raises(ValueError, match="batch size 1"):
            self.metrics.compute_photometry_error(
                multi_batch, self.target, source_positions
            )

    def test_compute_resolution_metric(self):
        """Test resolution preservation metric."""
        result = self.metrics.compute_resolution_metric(self.pred, self.target)

        assert isinstance(result, MetricResult)
        assert isinstance(result.value, float)
        assert -1 <= result.value <= 1  # Correlation-based metric
        assert "correlation" in result.metadata
        assert "power_ratio" in result.metadata
        assert "pred_power" in result.metadata
        assert "target_power" in result.metadata

    def test_compute_resolution_metric_perfect_match(self):
        """Test resolution metric with perfect match."""
        result = self.metrics.compute_resolution_metric(self.target, self.target)

        assert result.value >= 0.99  # Should be very close to 1
        assert abs(result.metadata["correlation"] - 1.0) < 0.01

    def test_compute_resolution_metric_with_mask(self):
        """Test resolution metric with mask."""
        mask = torch.ones_like(self.pred)
        mask[:, :, :32, :] = 0.0  # Mask out half the image

        result = self.metrics.compute_resolution_metric(
            self.pred, self.target, mask=mask
        )

        assert isinstance(result.value, float)

    def test_compute_resolution_metric_insufficient_pixels(self):
        """Test resolution metric with insufficient valid pixels."""
        tiny_pred = torch.rand(1, 1, 3, 3)
        tiny_target = torch.rand(1, 1, 3, 3)

        result = self.metrics.compute_resolution_metric(tiny_pred, tiny_target)

        assert np.isnan(result.value)
        assert "error" in result.metadata


class TestEvaluationSuite:
    """Test the comprehensive evaluation suite."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.suite = EvaluationSuite(device=self.device)

        # Create test data
        self.scale = 1000.0
        self.background = 10.0
        self.read_noise = 5.0

        self.pred = torch.rand(1, 1, 64, 64)
        self.target = torch.rand(1, 1, 64, 64)

        # Generate noisy observation
        torch.manual_seed(42)
        lambda_e = self.pred * self.scale + self.background
        self.noisy = (
            torch.poisson(lambda_e) + torch.randn_like(lambda_e) * self.read_noise
        )

        self.mask = torch.ones_like(self.pred)

    def test_evaluate_restoration_photography(self):
        """Test complete evaluation for photography domain."""
        report = self.suite.evaluate_restoration(
            pred=self.pred,
            target=self.target,
            noisy=self.noisy,
            scale=self.scale,
            domain="photography",
            background=self.background,
            read_noise=self.read_noise,
            mask=self.mask,
            method_name="test_method",
            dataset_name="test_dataset",
        )

        assert isinstance(report, EvaluationReport)
        assert report.method_name == "test_method"
        assert report.dataset_name == "test_dataset"
        assert report.domain == "photography"

        # Check that all standard metrics are computed
        assert isinstance(report.psnr.value, float)
        assert isinstance(report.ssim.value, float)
        assert isinstance(report.lpips.value, float)
        assert isinstance(report.ms_ssim.value, float)

        # Check that all physics metrics are computed
        assert isinstance(report.chi2_consistency.value, float)
        assert isinstance(report.residual_whiteness.value, float)
        assert isinstance(report.bias_analysis.value, float)

        # Check domain-specific metrics
        assert "resolution_preservation" in report.domain_metrics

        assert report.num_images == 1
        assert report.processing_time > 0

    def test_evaluate_restoration_microscopy(self):
        """Test complete evaluation for microscopy domain."""
        report = self.suite.evaluate_restoration(
            pred=self.pred,
            target=self.target,
            noisy=self.noisy,
            scale=self.scale,
            domain="microscopy",
            background=self.background,
            read_noise=self.read_noise,
            method_name="test_method",
            dataset_name="test_dataset",
        )

        assert report.domain == "microscopy"

        # Check microscopy-specific metrics
        assert "counting_accuracy" in report.domain_metrics
        assert "resolution_preservation" in report.domain_metrics

    def test_evaluate_restoration_astronomy(self):
        """Test complete evaluation for astronomy domain."""
        # Add source positions for photometry
        domain_params = {"source_positions": [(32, 32)], "aperture_radius": 5}

        report = self.suite.evaluate_restoration(
            pred=self.pred,
            target=self.target,
            noisy=self.noisy,
            scale=self.scale,
            domain="astronomy",
            background=self.background,
            read_noise=self.read_noise,
            method_name="test_method",
            dataset_name="test_dataset",
            domain_specific_params=domain_params,
        )

        assert report.domain == "astronomy"

        # Check astronomy-specific metrics
        assert "photometry_error" in report.domain_metrics
        assert "resolution_preservation" in report.domain_metrics

    def test_compare_methods(self):
        """Test method comparison functionality."""
        # Create multiple reports
        reports = []
        for i, method in enumerate(["method_a", "method_b"]):
            pred = torch.rand(1, 1, 64, 64) * (0.8 + i * 0.1)  # Different quality

            report = self.suite.evaluate_restoration(
                pred=pred,
                target=self.target,
                noisy=self.noisy,
                scale=self.scale,
                domain="photography",
                method_name=method,
                dataset_name="test_dataset",
            )
            reports.append(report)

        # Compare methods
        comparison = self.suite.compare_methods(reports)

        assert isinstance(comparison, dict)
        assert "test_dataset_photography" in comparison

        result = comparison["test_dataset_photography"]
        assert result["dataset"] == "test_dataset"
        assert result["domain"] == "photography"
        assert "metrics" in result
        assert "best_methods" in result

        # Check that methods are listed
        assert result["metrics"]["methods"] == ["method_a", "method_b"]
        assert len(result["metrics"]["psnr"]) == 2

    def test_compare_methods_empty_list(self):
        """Test method comparison with empty list."""
        with pytest.raises(ValueError, match="No reports provided"):
            self.suite.compare_methods([])

    def test_generate_summary_statistics(self):
        """Test summary statistics generation."""
        # Create multiple reports for the same method
        reports = []
        for i in range(3):
            pred = torch.rand(1, 1, 64, 64)

            report = self.suite.evaluate_restoration(
                pred=pred,
                target=self.target,
                noisy=self.noisy,
                scale=self.scale,
                domain="photography",
                method_name="test_method",
                dataset_name=f"dataset_{i}",
            )
            reports.append(report)

        summary = self.suite.generate_summary_statistics(reports)

        assert isinstance(summary, dict)
        assert "test_method_photography" in summary

        stats = summary["test_method_photography"]
        assert stats["method"] == "test_method"
        assert stats["domain"] == "photography"
        assert stats["num_evaluations"] == 3

        # Check that statistics are computed
        assert "psnr_stats" in stats
        assert "ssim_stats" in stats
        assert "chi2_stats" in stats

        psnr_stats = stats["psnr_stats"]
        assert "mean" in psnr_stats
        assert "std" in psnr_stats
        assert "min" in psnr_stats
        assert "max" in psnr_stats

    def test_generate_summary_statistics_empty_list(self):
        """Test summary statistics with empty list."""
        summary = self.suite.generate_summary_statistics([])
        assert summary == {}


class TestMetricResultAndReport:
    """Test MetricResult and EvaluationReport classes."""

    def test_metric_result_to_dict(self):
        """Test MetricResult serialization."""
        result = MetricResult(
            value=25.5, confidence_interval=(24.0, 27.0), metadata={"test": "value"}
        )

        data = result.to_dict()

        assert data["value"] == 25.5
        assert data["confidence_interval"] == (24.0, 27.0)
        assert data["metadata"] == {"test": "value"}

    def test_evaluation_report_serialization(self):
        """Test EvaluationReport JSON serialization."""
        # Create a simple report
        psnr = MetricResult(value=25.0, metadata={"mse": 0.01})
        ssim = MetricResult(value=0.85, metadata={"std": 0.02})
        lpips = MetricResult(value=0.15)
        ms_ssim = MetricResult(value=0.90)
        chi2 = MetricResult(value=1.05, metadata={"ks_pvalue": 0.3})
        whiteness = MetricResult(value=0.75)
        bias = MetricResult(value=0.5, metadata={"p_value": 0.8})

        domain_metrics = {"resolution_preservation": MetricResult(value=0.88)}

        report = EvaluationReport(
            method_name="test_method",
            dataset_name="test_dataset",
            domain="photography",
            psnr=psnr,
            ssim=ssim,
            lpips=lpips,
            ms_ssim=ms_ssim,
            chi2_consistency=chi2,
            residual_whiteness=whiteness,
            bias_analysis=bias,
            domain_metrics=domain_metrics,
            num_images=5,
            processing_time=12.5,
        )

        # Test JSON serialization
        json_str = report.to_json()
        assert isinstance(json_str, str)

        # Test deserialization
        restored_report = EvaluationReport.from_json(json_str)

        assert restored_report.method_name == report.method_name
        assert restored_report.dataset_name == report.dataset_name
        assert restored_report.domain == report.domain
        assert restored_report.psnr.value == report.psnr.value
        assert restored_report.ssim.value == report.ssim.value
        assert restored_report.num_images == report.num_images
        assert restored_report.processing_time == report.processing_time

        # Check domain metrics
        assert "resolution_preservation" in restored_report.domain_metrics
        assert restored_report.domain_metrics["resolution_preservation"].value == 0.88


class TestUtilityFunctions:
    """Test utility functions for baseline comparison and file I/O."""

    def test_save_and_load_evaluation_results(self):
        """Test saving and loading evaluation results."""
        # Create test reports
        reports = []
        for i in range(2):
            psnr = MetricResult(value=25.0 + i, metadata={"mse": 0.01})
            ssim = MetricResult(value=0.85 + i * 0.05)
            lpips = MetricResult(value=0.15)
            ms_ssim = MetricResult(value=0.90)
            chi2 = MetricResult(value=1.05)
            whiteness = MetricResult(value=0.75)
            bias = MetricResult(value=0.5)

            report = EvaluationReport(
                method_name=f"method_{i}",
                dataset_name="test_dataset",
                domain="photography",
                psnr=psnr,
                ssim=ssim,
                lpips=lpips,
                ms_ssim=ms_ssim,
                chi2_consistency=chi2,
                residual_whiteness=whiteness,
                bias_analysis=bias,
                domain_metrics={},
                num_images=1,
                processing_time=10.0,
            )
            reports.append(report)

        # Test saving
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            save_evaluation_results(reports, temp_file)

            # Verify file was created and contains data
            assert Path(temp_file).exists()

            with open(temp_file, "r") as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["method_name"] == "method_0"
            assert data[1]["method_name"] == "method_1"

        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)

    def test_load_baseline_results(self):
        """Test loading baseline results from directory."""
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test JSON files
            test_data = {
                "method_name": "baseline_method",
                "dataset_name": "test_dataset",
                "domain": "photography",
                "psnr": {"value": 25.0, "confidence_interval": None, "metadata": {}},
                "ssim": {"value": 0.85, "confidence_interval": None, "metadata": {}},
                "lpips": {"value": 0.15, "confidence_interval": None, "metadata": {}},
                "ms_ssim": {"value": 0.90, "confidence_interval": None, "metadata": {}},
                "chi2_consistency": {
                    "value": 1.05,
                    "confidence_interval": None,
                    "metadata": {},
                },
                "residual_whiteness": {
                    "value": 0.75,
                    "confidence_interval": None,
                    "metadata": {},
                },
                "bias_analysis": {
                    "value": 0.5,
                    "confidence_interval": None,
                    "metadata": {},
                },
                "domain_metrics": {},
                "num_images": 1,
                "processing_time": 10.0,
            }

            # Save test file
            test_file = Path(temp_dir) / "baseline_method.json"
            with open(test_file, "w") as f:
                json.dump(test_data, f)

            # Test loading
            baseline_results = load_baseline_results(temp_dir)

            assert isinstance(baseline_results, dict)
            assert "baseline_method" in baseline_results
            assert len(baseline_results["baseline_method"]) == 1

            report = baseline_results["baseline_method"][0]
            assert isinstance(report, EvaluationReport)
            assert report.method_name == "baseline_method"
            assert report.psnr.value == 25.0

    def test_load_baseline_results_empty_directory(self):
        """Test loading baseline results from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_results = load_baseline_results(temp_dir)
            assert baseline_results == {}

    def test_load_baseline_results_invalid_json(self):
        """Test loading baseline results with invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid JSON file
            invalid_file = Path(temp_dir) / "invalid.json"
            with open(invalid_file, "w") as f:
                f.write("invalid json content")

            # Should handle gracefully and return empty results
            baseline_results = load_baseline_results(temp_dir)
            assert baseline_results == {}


if __name__ == "__main__":
    pytest.main([__file__])
