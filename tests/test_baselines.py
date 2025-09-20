"""
Comprehensive tests for baseline comparison methods.

Tests all baseline methods including classical, deep learning,
and self-supervised approaches for image denoising.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from core.baselines import (
    AnscombeBaseline,
    BaselineComparator,
    BaselineMethod,
    BM3DBaseline,
    DnCNNBaseline,
    GaussianBaseline,
    L2GuidedDiffusionBaseline,
    NAFNetBaseline,
    Noise2VoidBaseline,
    RichardsonLucyBaseline,
    WienerFilterBaseline,
    create_baseline_suite,
    run_baseline_comparison,
)
from core.metrics import EvaluationSuite


class TestBaselineMethod:
    """Test abstract baseline method functionality."""

    def test_abstract_methods(self):
        """Test that BaselineMethod is properly abstract."""
        with pytest.raises(TypeError):
            BaselineMethod("test", "cuda")


class TestClassicalBaselines:
    """Test classical baseline methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"  # Use CPU for testing
        self.batch_size = 2
        self.channels = 1
        self.height = 32
        self.width = 32

        # Create synthetic test data
        self.scale = 1000.0
        self.background = 100.0
        self.read_noise = 5.0

        # Generate clean image
        clean = torch.rand(self.batch_size, self.channels, self.height, self.width)
        clean = clean * 0.8 + 0.1  # Scale to [0.1, 0.9]

        # Generate noisy observation (electrons)
        clean_electrons = clean * self.scale + self.background
        noisy_electrons = (
            torch.poisson(clean_electrons)
            + torch.randn_like(clean_electrons) * self.read_noise
        )

        self.clean = clean
        self.noisy = noisy_electrons

    def test_bm3d_baseline_availability(self):
        """Test BM3D baseline availability checking."""
        baseline = BM3DBaseline(device=self.device)

        # Should work regardless of bm3d availability
        assert isinstance(baseline.is_available, bool)
        assert baseline.name == "BM3D"

    def test_bm3d_baseline_denoise(self):
        """Test BM3D denoising functionality."""
        baseline = BM3DBaseline(device=self.device)

        if baseline.is_available:
            result = baseline.denoise(
                self.noisy, self.scale, self.background, self.read_noise
            )

            assert result.shape == self.clean.shape
            assert result.dtype == torch.float32
            assert torch.all(result >= 0) and torch.all(result <= 1)
        else:
            # Should raise error if not available
            with pytest.raises(RuntimeError):
                baseline.denoise(
                    self.noisy, self.scale, self.background, self.read_noise
                )

    def test_anscombe_baseline(self):
        """Test Anscombe+BM3D baseline."""
        baseline = AnscombeBaseline(device=self.device)

        if baseline.is_available:
            result = baseline.denoise(
                self.noisy, self.scale, self.background, self.read_noise
            )

            assert result.shape == self.clean.shape
            assert result.dtype == torch.float32
            assert torch.all(result >= 0) and torch.all(result <= 1)

            # Should be different from input (some denoising occurred)
            input_norm = (self.noisy - self.background) / self.scale
            input_norm = torch.clamp(input_norm, 0, 1)
            assert not torch.allclose(result, input_norm, atol=0.01)

    def test_richardson_lucy_baseline(self):
        """Test Richardson-Lucy baseline."""
        baseline = RichardsonLucyBaseline(device=self.device, num_iterations=5)

        assert baseline.is_available  # Always available

        result = baseline.denoise(
            self.noisy, self.scale, self.background, self.read_noise
        )

        assert result.shape == self.clean.shape
        assert result.dtype == torch.float32
        assert torch.all(result >= 0) and torch.all(result <= 1)

        # Check parameters
        params = baseline.get_parameters()
        assert params["num_iterations"] == 5

    def test_gaussian_baseline(self):
        """Test Gaussian filtering baseline."""
        baseline = GaussianBaseline(device=self.device, sigma=1.5)

        assert baseline.is_available  # Always available

        result = baseline.denoise(
            self.noisy, self.scale, self.background, self.read_noise
        )

        assert result.shape == self.clean.shape
        assert result.dtype == torch.float32
        assert torch.all(result >= 0) and torch.all(result <= 1)

        # Should be smoother than input
        input_norm = (self.noisy - self.background) / self.scale
        input_norm = torch.clamp(input_norm, 0, 1)

        # Check that result is smoother (lower high-frequency content)
        result_grad = torch.gradient(result.squeeze())[0]
        input_grad = torch.gradient(input_norm.squeeze())[0]
        assert result_grad.abs().mean() < input_grad.abs().mean()

        # Check parameters
        params = baseline.get_parameters()
        assert params["sigma"] == 1.5

    def test_wiener_filter_baseline(self):
        """Test Wiener filter baseline."""
        baseline = WienerFilterBaseline(device=self.device, noise_variance=0.02)

        assert baseline.is_available  # Always available

        result = baseline.denoise(
            self.noisy, self.scale, self.background, self.read_noise
        )

        assert result.shape == self.clean.shape
        assert result.dtype == torch.float32
        assert torch.all(result >= 0) and torch.all(result <= 1)

        # Check parameters
        params = baseline.get_parameters()
        assert params["noise_variance"] == 0.02


class TestDeepLearningBaselines:
    """Test deep learning baseline methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.batch_size = 1
        self.channels = 1
        self.height = 64
        self.width = 64

        # Create test data
        self.scale = 1000.0
        self.background = 100.0
        self.read_noise = 5.0

        clean = (
            torch.rand(self.batch_size, self.channels, self.height, self.width) * 0.8
            + 0.1
        )
        clean_electrons = clean * self.scale + self.background
        noisy_electrons = (
            torch.poisson(clean_electrons)
            + torch.randn_like(clean_electrons) * self.read_noise
        )

        self.clean = clean
        self.noisy = noisy_electrons

    def test_dncnn_baseline_creation(self):
        """Test DnCNN baseline creation."""
        baseline = DnCNNBaseline(device=self.device, num_layers=10)

        assert baseline.is_available  # Should create built-in model
        assert baseline.name == "DnCNN"
        assert baseline.num_layers == 10

        # Check model creation
        assert baseline.model is not None
        assert isinstance(baseline.model, torch.nn.Module)

    def test_dncnn_baseline_denoise(self):
        """Test DnCNN denoising."""
        baseline = DnCNNBaseline(
            device=self.device, num_layers=5
        )  # Smaller for testing

        result = baseline.denoise(
            self.noisy, self.scale, self.background, self.read_noise
        )

        assert result.shape == self.clean.shape
        assert result.dtype == torch.float32
        assert torch.all(result >= 0) and torch.all(result <= 1)

        # Check parameters
        params = baseline.get_parameters()
        assert params["num_layers"] == 5
        assert params["architecture"] == "DnCNN-like"

    def test_nafnet_baseline_creation(self):
        """Test NAFNet baseline creation."""
        baseline = NAFNetBaseline(device=self.device)

        assert baseline.is_available  # Should create built-in model
        assert baseline.name == "NAFNet"

        # Check model creation
        assert baseline.model is not None
        assert isinstance(baseline.model, torch.nn.Module)

    def test_nafnet_baseline_denoise(self):
        """Test NAFNet denoising."""
        baseline = NAFNetBaseline(device=self.device)

        result = baseline.denoise(
            self.noisy, self.scale, self.background, self.read_noise
        )

        assert result.shape == self.clean.shape
        assert result.dtype == torch.float32
        assert torch.all(result >= 0) and torch.all(result <= 1)

        # Check parameters
        params = baseline.get_parameters()
        assert params["architecture"] == "NAFNet-like"

    def test_model_loading_from_file(self):
        """Test loading models from file."""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            # Create a simple model and save it
            simple_model = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 1, 3, padding=1),
            )
            torch.save(simple_model.state_dict(), tmp.name)
            tmp_path = tmp.name

        try:
            # Test DnCNN with model file
            baseline = DnCNNBaseline(model_path=tmp_path, device=self.device)

            # Should try to load from file (may fail due to architecture mismatch)
            # but should fall back to built-in model
            assert baseline.is_available

        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestSelfSupervisedBaselines:
    """Test self-supervised baseline methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.batch_size = 1
        self.channels = 1
        self.height = 32
        self.width = 32

        # Create test data
        self.scale = 1000.0
        self.background = 100.0
        self.read_noise = 5.0

        clean = (
            torch.rand(self.batch_size, self.channels, self.height, self.width) * 0.8
            + 0.1
        )
        clean_electrons = clean * self.scale + self.background
        noisy_electrons = (
            torch.poisson(clean_electrons)
            + torch.randn_like(clean_electrons) * self.read_noise
        )

        self.clean = clean
        self.noisy = noisy_electrons

    def test_noise2void_baseline(self):
        """Test Noise2Void baseline."""
        baseline = Noise2VoidBaseline(device=self.device, mask_ratio=0.2)

        assert baseline.is_available  # Always available
        assert baseline.name == "Noise2Void"

        result = baseline.denoise(
            self.noisy, self.scale, self.background, self.read_noise
        )

        assert result.shape == self.clean.shape
        assert result.dtype == torch.float32
        assert torch.all(result >= 0) and torch.all(result <= 1)

        # Check parameters
        params = baseline.get_parameters()
        assert params["mask_ratio"] == 0.2


class TestDiffusionBaselines:
    """Test diffusion-based baseline methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.batch_size = 1
        self.channels = 1
        self.height = 32
        self.width = 32

        # Create test data
        self.scale = 1000.0
        self.background = 100.0
        self.read_noise = 5.0

        clean = (
            torch.rand(self.batch_size, self.channels, self.height, self.width) * 0.8
            + 0.1
        )
        clean_electrons = clean * self.scale + self.background
        noisy_electrons = (
            torch.poisson(clean_electrons)
            + torch.randn_like(clean_electrons) * self.read_noise
        )

        self.clean = clean
        self.noisy = noisy_electrons

    def test_l2_guided_diffusion_baseline(self):
        """Test L2-guided diffusion baseline."""
        # Create mock diffusion model
        mock_model = Mock()
        mock_model.return_value = (
            torch.randn(self.batch_size, self.channels, self.height, self.width) * 0.1
        )

        baseline = L2GuidedDiffusionBaseline(mock_model, device=self.device)

        assert baseline.is_available
        assert baseline.name == "L2-Guided-Diffusion"

        result = baseline.denoise(
            self.noisy,
            self.scale,
            self.background,
            self.read_noise,
            steps=5,
            guidance_weight=0.5,  # Fewer steps for testing
        )

        assert result.shape == self.clean.shape
        assert result.dtype == torch.float32
        assert torch.all(result >= 0) and torch.all(result <= 1)

        # Check that model was called
        assert mock_model.called

        # Check parameters
        params = baseline.get_parameters()
        assert params["guidance_type"] == "L2"
        assert params["model_type"] == "EDM"

    def test_l2_guided_diffusion_without_model(self):
        """Test L2-guided diffusion without model."""
        baseline = L2GuidedDiffusionBaseline(None, device=self.device)

        assert not baseline.is_available

        with pytest.raises(RuntimeError):
            baseline.denoise(self.noisy, self.scale, self.background, self.read_noise)


class TestBaselineComparator:
    """Test baseline comparison framework."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.comparator = BaselineComparator(device=self.device)

        # Create test data
        self.batch_size = 1
        self.channels = 1
        self.height = 32
        self.width = 32

        self.scale = 1000.0
        self.background = 100.0
        self.read_noise = 5.0

        clean = (
            torch.rand(self.batch_size, self.channels, self.height, self.width) * 0.8
            + 0.1
        )
        clean_electrons = clean * self.scale + self.background
        noisy_electrons = (
            torch.poisson(clean_electrons)
            + torch.randn_like(clean_electrons) * self.read_noise
        )

        self.clean = clean
        self.noisy = noisy_electrons

    def test_comparator_initialization(self):
        """Test baseline comparator initialization."""
        assert isinstance(self.comparator.baselines, dict)
        assert isinstance(self.comparator.available_baselines, dict)
        assert len(self.comparator.available_baselines) > 0

        # Check that some standard baselines are available
        expected_baselines = [
            "Gaussian",
            "Richardson-Lucy",
            "DnCNN",
            "NAFNet",
            "Noise2Void",
            "Wiener",
        ]
        for baseline_name in expected_baselines:
            assert baseline_name in self.comparator.available_baselines

    def test_add_custom_baseline(self):
        """Test adding custom baseline method."""
        # Create custom baseline
        custom_baseline = GaussianBaseline(device=self.device, sigma=2.0)
        custom_baseline.name = "Custom-Gaussian"

        initial_count = len(self.comparator.available_baselines)
        self.comparator.add_baseline("Custom-Gaussian", custom_baseline)

        assert len(self.comparator.available_baselines) == initial_count + 1
        assert "Custom-Gaussian" in self.comparator.available_baselines

    def test_add_diffusion_baseline(self):
        """Test adding diffusion baseline."""
        # Create mock model
        mock_model = Mock()
        mock_model.return_value = torch.randn(1, 1, 32, 32) * 0.1

        initial_count = len(self.comparator.available_baselines)
        self.comparator.add_diffusion_baseline(mock_model)

        assert len(self.comparator.available_baselines) == initial_count + 1
        assert "L2-Guided-Diffusion" in self.comparator.available_baselines

    def test_evaluate_single_baseline(self):
        """Test evaluating a single baseline method."""
        # Use simple Gaussian baseline for testing
        baseline = self.comparator.available_baselines["Gaussian"]

        result = baseline.denoise(
            self.noisy, self.scale, self.background, self.read_noise
        )

        assert result.shape == self.clean.shape
        assert torch.all(result >= 0) and torch.all(result <= 1)

    @patch("core.baselines.logger")
    def test_evaluate_all_baselines(self, mock_logger):
        """Test evaluating all baseline methods."""
        # Use smaller subset for testing
        test_baselines = ["Gaussian", "Richardson-Lucy"]

        # Create temporary comparator with limited baselines
        temp_comparator = BaselineComparator(device=self.device)
        temp_comparator.available_baselines = {
            name: temp_comparator.available_baselines[name] for name in test_baselines
        }

        results = temp_comparator.evaluate_all_baselines(
            noisy=self.noisy,
            target=self.clean,
            scale=self.scale,
            domain="test",
            background=self.background,
            read_noise=self.read_noise,
            dataset_name="test_dataset",
            save_results=False,
        )

        assert len(results) == len(test_baselines)

        for method_name in test_baselines:
            assert method_name in results
            report = results[method_name]

            # Check that report has required attributes
            assert hasattr(report, "psnr")
            assert hasattr(report, "ssim")
            assert hasattr(report, "method_name")
            assert report.method_name == method_name


class TestUtilityFunctions:
    """Test utility functions for baseline integration."""

    def test_create_baseline_suite(self):
        """Test creating baseline suite."""
        suite = create_baseline_suite(device="cpu")

        assert isinstance(suite, BaselineComparator)
        assert len(suite.available_baselines) > 0

    def test_create_baseline_suite_with_diffusion(self):
        """Test creating baseline suite with diffusion model."""
        mock_model = Mock()
        mock_model.return_value = torch.randn(1, 1, 32, 32) * 0.1

        suite = create_baseline_suite(device="cpu", diffusion_model=mock_model)

        assert isinstance(suite, BaselineComparator)
        assert "L2-Guided-Diffusion" in suite.available_baselines


class TestErrorHandling:
    """Test error handling in baseline methods."""

    def test_baseline_with_invalid_input(self):
        """Test baseline behavior with invalid input."""
        baseline = GaussianBaseline(device="cpu")

        # Test with NaN input
        invalid_input = torch.full((1, 1, 32, 32), float("nan"))

        # Should handle gracefully (clamp will convert NaN to 0)
        result = baseline.denoise(invalid_input, 1000.0, 0.0, 5.0)
        assert not torch.isnan(result).any()

    def test_baseline_with_extreme_parameters(self):
        """Test baseline behavior with extreme parameters."""
        baseline = GaussianBaseline(device="cpu", sigma=100.0)  # Very large sigma

        test_input = torch.rand(1, 1, 32, 32) * 1000 + 100

        result = baseline.denoise(test_input, 1000.0, 100.0, 5.0)

        assert result.shape == (1, 1, 32, 32)
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_unavailable_baseline_fallback(self):
        """Test fallback behavior for unavailable baselines."""
        # Create baseline that claims to be unavailable
        baseline = BM3DBaseline(device="cpu")
        baseline.is_available = False  # Force unavailable

        with pytest.raises(RuntimeError):
            baseline.denoise(torch.rand(1, 1, 32, 32), 1000.0, 0.0, 5.0)


class TestIntegration:
    """Integration tests for baseline comparison."""

    def test_end_to_end_comparison(self):
        """Test complete baseline comparison workflow."""
        # Create small test dataset
        device = "cpu"
        comparator = BaselineComparator(device=device)

        # Limit to fast baselines for testing
        test_baselines = ["Gaussian", "Richardson-Lucy"]
        comparator.available_baselines = {
            name: comparator.available_baselines[name] for name in test_baselines
        }

        # Create test data
        clean = torch.rand(1, 1, 32, 32) * 0.8 + 0.1
        noisy = clean + torch.randn_like(clean) * 0.1

        # Run comparison
        results = comparator.evaluate_all_baselines(
            noisy=noisy * 1000,  # Convert to electrons
            target=clean,
            scale=1000.0,
            domain="test",
            save_results=False,
        )

        assert len(results) == len(test_baselines)

        # Check that all methods produced valid results
        for method_name, report in results.items():
            assert not np.isnan(report.psnr.value)
            assert not np.isnan(report.ssim.value)
            assert report.method_name == method_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
