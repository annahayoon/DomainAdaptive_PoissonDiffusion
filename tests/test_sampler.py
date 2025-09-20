"""
Comprehensive tests for EDM posterior sampler with guidance.

Tests cover:
- Sampling configuration validation
- Noise schedule generation
- Guided sampling pipeline
- Different solver implementations
- Integration with EDM wrapper and guidance
- Batch processing and diagnostics
"""

# Import sampler components
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))

from core.exceptions import ConfigurationError, SamplingError
from core.guidance_config import GuidanceConfig
from core.poisson_guidance import PoissonGuidance
from models.edm_wrapper import DomainEncoder, EDMConfig, EDMModelWrapper
from models.sampler import (
    EDMPosteriorSampler,
    SamplingConfig,
    create_edm_sampler,
    create_fast_sampler,
    create_high_quality_sampler,
    sample_batch,
)


class TestSamplingConfig:
    """Test sampling configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SamplingConfig()

        assert config.num_steps == 18
        assert config.sigma_min == 0.002
        assert config.sigma_max == 80.0
        assert config.rho == 7.0
        assert config.guidance_scale == 1.0
        assert config.solver == "heun"
        assert config.clip_denoised is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SamplingConfig(
            num_steps=50, guidance_scale=1.5, solver="euler", clip_range=(0.1, 0.9)
        )

        assert config.num_steps == 50
        assert config.guidance_scale == 1.5
        assert config.solver == "euler"
        assert config.clip_range == (0.1, 0.9)

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid num_steps
        with pytest.raises(ConfigurationError, match="num_steps must be positive"):
            SamplingConfig(num_steps=0)

        # Invalid sigma range
        with pytest.raises(ConfigurationError, match="sigma_min.*must be.*sigma_max"):
            SamplingConfig(sigma_min=1.0, sigma_max=0.5)

        # Invalid guidance steps
        with pytest.raises(ConfigurationError, match="guidance_end_step must be"):
            SamplingConfig(guidance_start_step=10, guidance_end_step=5)

        # Invalid solver
        with pytest.raises(ConfigurationError, match="Unknown solver"):
            SamplingConfig(solver="invalid_solver")


class TestEDMPosteriorSampler:
    """Test EDM posterior sampler functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create mock EDM model."""
        model = MagicMock(spec=EDMModelWrapper)
        model.encode_conditioning.return_value = torch.randn(1, 6)

        def mock_forward(*args, **kwargs):
            # Return tensor with same shape as input
            x = args[0] if args else kwargs.get("x", torch.randn(1, 1, 32, 32))
            return torch.randn_like(x)

        model.forward = mock_forward
        model.__call__ = mock_forward
        return model

    @pytest.fixture
    def mock_guidance(self):
        """Create mock guidance."""
        guidance = MagicMock(spec=PoissonGuidance)
        guidance.compute.return_value = torch.randn(1, 1, 32, 32) * 0.1
        return guidance

    @pytest.fixture
    def sampler_config(self):
        """Create test sampling configuration."""
        return SamplingConfig(
            num_steps=5,  # Small for fast testing
            sigma_min=0.01,
            sigma_max=10.0,
            collect_diagnostics=True,
        )

    @pytest.fixture
    def sampler(self, mock_model, mock_guidance, sampler_config):
        """Create test sampler."""
        return EDMPosteriorSampler(
            model=mock_model, guidance=mock_guidance, config=sampler_config
        )

    def test_sampler_initialization(self, mock_model, mock_guidance, sampler_config):
        """Test sampler initialization."""
        sampler = EDMPosteriorSampler(
            model=mock_model, guidance=mock_guidance, config=sampler_config
        )

        assert sampler.model is mock_model
        assert sampler.guidance is mock_guidance
        assert sampler.config is sampler_config

    def test_sampler_validation(self, mock_guidance, sampler_config):
        """Test sampler component validation."""
        # Invalid model type
        with pytest.raises(ConfigurationError, match="Model must be EDMModelWrapper"):
            EDMPosteriorSampler(
                model="invalid_model", guidance=mock_guidance, config=sampler_config
            )

        # Invalid guidance type
        with pytest.raises(
            ConfigurationError, match="Guidance must be PoissonGuidance"
        ):
            EDMPosteriorSampler(
                model=MagicMock(spec=EDMModelWrapper),
                guidance="invalid_guidance",
                config=sampler_config,
            )

    def test_noise_schedule_creation(self, sampler):
        """Test noise schedule generation."""
        # Default schedule
        schedule = sampler.create_noise_schedule()
        assert schedule.shape == (sampler.config.num_steps + 1,)
        assert abs(schedule[0] - sampler.config.sigma_max) < 1e-6  # Start at max
        assert abs(schedule[-1] - sampler.config.sigma_min) < 1e-6  # End at min
        assert torch.all(schedule[:-1] >= schedule[1:])  # Monotonically decreasing

        # Custom number of steps
        custom_schedule = sampler.create_noise_schedule(num_steps=10)
        assert custom_schedule.shape == (11,)

    def test_guidance_step_logic(self, sampler):
        """Test guidance application logic."""
        # Default: apply guidance at all steps
        assert sampler._should_apply_guidance(0) is True
        assert sampler._should_apply_guidance(2) is True
        assert sampler._should_apply_guidance(4) is True

        # With start step
        sampler.config.guidance_start_step = 2
        assert sampler._should_apply_guidance(0) is False
        assert sampler._should_apply_guidance(1) is False
        assert sampler._should_apply_guidance(2) is True
        assert sampler._should_apply_guidance(3) is True

        # With end step
        sampler.config.guidance_end_step = 3
        assert sampler._should_apply_guidance(2) is True
        assert sampler._should_apply_guidance(3) is False
        assert sampler._should_apply_guidance(4) is False

    def test_denoise_step(self, sampler):
        """Test denoising step."""
        x = torch.randn(1, 1, 32, 32)
        sigma = torch.tensor(1.0)
        condition = torch.randn(1, 6)
        y_observed = torch.randn(1, 1, 32, 32)

        x_denoised = sampler._denoise_step(x, sigma, condition, y_observed, None, 0)

        assert x_denoised.shape == x.shape
        assert torch.isfinite(x_denoised).all()

        # Check that model was called correctly
        sampler.model.assert_called_once()

    def test_step_solvers(self, sampler):
        """Test different ODE solvers."""
        x = torch.randn(1, 1, 32, 32)
        d = torch.randn(1, 1, 32, 32) * 0.1
        sigma_curr = torch.tensor(1.0)
        sigma_next = torch.tensor(0.8)

        # Test Euler solver
        sampler.config.solver = "euler"
        x_euler = sampler._take_step(x, d, sigma_curr, sigma_next, 0)
        assert x_euler.shape == x.shape
        assert torch.isfinite(x_euler).all()

        # Test Heun solver
        sampler.config.solver = "heun"
        x_heun = sampler._take_step(x, d, sigma_curr, sigma_next, 0)
        assert x_heun.shape == x.shape
        assert torch.isfinite(x_heun).all()

        # Test DPM solver
        sampler.config.solver = "dpm"
        x_dpm = sampler._take_step(x, d, sigma_curr, sigma_next, 0)
        assert x_dpm.shape == x.shape
        assert torch.isfinite(x_dpm).all()

        # Test invalid solver
        sampler.config.solver = "invalid"
        with pytest.raises(SamplingError, match="Unknown solver"):
            sampler._take_step(x, d, sigma_curr, sigma_next, 0)

    def test_stochastic_sampling(self, sampler):
        """Test stochastic sampling parameters."""
        x = torch.randn(1, 1, 32, 32)
        d = torch.randn(1, 1, 32, 32) * 0.1
        sigma_curr = torch.tensor(1.0)
        sigma_next = torch.tensor(0.8)

        # Enable stochasticity
        sampler.config.S_churn = 0.1
        sampler.config.S_min = 0.5
        sampler.config.S_max = 2.0

        # Should add stochasticity when sigma is in range
        x_stochastic = sampler._take_step(x, d, sigma_curr, sigma_next, 0)
        assert x_stochastic.shape == x.shape
        assert torch.isfinite(x_stochastic).all()

    def test_sample_basic(self, sampler):
        """Test basic sampling functionality."""
        y_observed = torch.randn(1, 1, 32, 32)
        condition = torch.randn(1, 6)

        # Disable diagnostics for this test to avoid norm computation issues
        sampler.config.collect_diagnostics = False

        results = sampler.sample(y_observed=y_observed, condition=condition)

        # Check results structure
        assert "x_final" in results
        assert "x_init" in results
        assert "sigma_schedule" in results
        assert "condition" in results

        # Check shapes
        assert results["x_final"].shape == y_observed.shape
        assert results["x_init"].shape == y_observed.shape
        assert results["condition"].shape == condition.shape

        # Check that final result is finite
        assert torch.isfinite(results["x_final"]).all()

    def test_sample_with_domain_params(self, sampler):
        """Test sampling with domain parameters."""
        y_observed = torch.randn(1, 1, 32, 32)

        results = sampler.sample(
            y_observed=y_observed,
            domain="photography",
            scale=1000.0,
            read_noise=3.0,
            background=10.0,
        )

        assert "x_final" in results
        assert results["x_final"].shape == y_observed.shape

        # Check that model.encode_conditioning was called
        sampler.model.encode_conditioning.assert_called_once()

    def test_sample_missing_parameters(self, sampler):
        """Test error handling for missing parameters."""
        y_observed = torch.randn(1, 1, 32, 32)

        with pytest.raises(
            ValueError, match="Must provide either 'condition' or all domain parameters"
        ):
            sampler.sample(
                y_observed=y_observed, domain="photography"  # Missing other parameters
            )

    def test_sample_with_mask(self, sampler):
        """Test sampling with pixel mask."""
        y_observed = torch.randn(1, 1, 32, 32)
        condition = torch.randn(1, 6)
        mask = torch.ones(1, 1, 32, 32)
        mask[:, :, :10, :10] = 0  # Mask out corner

        results = sampler.sample(y_observed=y_observed, condition=condition, mask=mask)

        assert results["x_final"].shape == y_observed.shape

        # Check that guidance was called with mask
        # (This would need more sophisticated mocking to verify)

    def test_sample_with_init(self, sampler):
        """Test sampling with custom initialization."""
        y_observed = torch.randn(1, 1, 32, 32)
        condition = torch.randn(1, 6)
        x_init = torch.randn(1, 1, 32, 32) * 0.5

        results = sampler.sample(
            y_observed=y_observed, condition=condition, x_init=x_init
        )

        # Initial sample should be scaled version of x_init
        # (exact relationship depends on noise schedule)
        assert results["x_init"].shape == x_init.shape

    def test_diagnostics_collection(self, sampler):
        """Test diagnostics collection."""
        y_observed = torch.randn(1, 1, 32, 32)
        condition = torch.randn(1, 6)

        # Enable diagnostics
        sampler.config.collect_diagnostics = True

        results = sampler.sample(y_observed=y_observed, condition=condition)

        assert "diagnostics" in results
        diagnostics = results["diagnostics"]

        # Check diagnostic keys
        expected_keys = [
            "sigma_values",
            "guidance_norms",
            "denoised_norms",
            "step_sizes",
        ]
        for key in expected_keys:
            assert key in diagnostics
            assert len(diagnostics[key]) == sampler.config.num_steps

        # Get summary diagnostics
        summary = sampler.get_diagnostics()
        assert "config" in summary
        assert summary["config"]["num_steps"] == sampler.config.num_steps

    def test_intermediates_saving(self, sampler):
        """Test intermediate state saving."""
        y_observed = torch.randn(1, 1, 32, 32)
        condition = torch.randn(1, 6)

        # Enable intermediate saving
        sampler.config.save_intermediates = True

        results = sampler.sample(y_observed=y_observed, condition=condition)

        assert "intermediates" in results
        intermediates = results["intermediates"]

        assert len(intermediates) == sampler.config.num_steps

        # Check intermediate structure
        for i, intermediate in enumerate(intermediates):
            assert "step" in intermediate
            assert "sigma" in intermediate
            assert "x" in intermediate
            assert "x_denoised" in intermediate
            assert intermediate["step"] == i
            assert intermediate["x"].shape == y_observed.shape

    def test_clipping(self, sampler):
        """Test output clipping."""
        y_observed = torch.randn(1, 1, 32, 32)
        condition = torch.randn(1, 6)

        # Mock model to return values outside clip range
        sampler.model.return_value = torch.ones(1, 1, 32, 32) * 2.0  # > 1.0

        # Enable clipping
        sampler.config.clip_denoised = True
        sampler.config.clip_range = (0.0, 1.0)

        results = sampler.sample(y_observed=y_observed, condition=condition)

        # Final result should be clipped
        x_final = results["x_final"]
        assert x_final.min() >= 0.0
        assert x_final.max() <= 1.0


class TestFactoryFunctions:
    """Test factory functions for creating samplers."""

    @pytest.fixture
    def mock_model(self):
        return MagicMock(spec=EDMModelWrapper)

    @pytest.fixture
    def mock_guidance(self):
        return MagicMock(spec=PoissonGuidance)

    def test_create_edm_sampler(self, mock_model, mock_guidance):
        """Test basic EDM sampler creation."""
        sampler = create_edm_sampler(
            model=mock_model, guidance=mock_guidance, num_steps=20, guidance_scale=1.5
        )

        assert isinstance(sampler, EDMPosteriorSampler)
        assert sampler.config.num_steps == 20
        assert sampler.config.guidance_scale == 1.5
        assert sampler.config.solver == "heun"  # default

    def test_create_fast_sampler(self, mock_model, mock_guidance):
        """Test fast sampler creation."""
        sampler = create_fast_sampler(model=mock_model, guidance=mock_guidance)

        assert isinstance(sampler, EDMPosteriorSampler)
        assert sampler.config.num_steps == 10  # Fewer steps
        assert sampler.config.solver == "euler"  # Faster solver
        assert sampler.config.collect_diagnostics is False  # Skip diagnostics

    def test_create_high_quality_sampler(self, mock_model, mock_guidance):
        """Test high-quality sampler creation."""
        sampler = create_high_quality_sampler(model=mock_model, guidance=mock_guidance)

        assert isinstance(sampler, EDMPosteriorSampler)
        assert sampler.config.num_steps == 50  # More steps
        assert sampler.config.solver == "heun"  # Higher-order solver
        assert sampler.config.guidance_scale == 1.2  # Stronger guidance
        assert sampler.config.collect_diagnostics is True
        assert sampler.config.save_intermediates is True


class TestBatchProcessing:
    """Test batch processing functionality."""

    @pytest.fixture
    def mock_sampler(self):
        """Create mock sampler for batch testing."""
        sampler = MagicMock(spec=EDMPosteriorSampler)

        def mock_sample(**kwargs):
            y_observed = kwargs["y_observed"]
            return {
                "x_final": torch.randn_like(y_observed),
                "x_init": torch.randn_like(y_observed),
            }

        sampler.sample = mock_sample
        return sampler

    def test_sample_batch(self, mock_sampler):
        """Test batch sampling function."""
        batch_size = 3
        y_observed_batch = torch.randn(batch_size, 1, 32, 32)
        domain_batch = ["photography", "microscopy", "astronomy"]
        scale_batch = [1000.0, 500.0, 2000.0]
        read_noise_batch = [3.0, 2.0, 5.0]
        background_batch = [10.0, 5.0, 20.0]

        results = sample_batch(
            sampler=mock_sampler,
            y_observed_batch=y_observed_batch,
            domain_batch=domain_batch,
            scale_batch=scale_batch,
            read_noise_batch=read_noise_batch,
            background_batch=background_batch,
        )

        assert len(results) == batch_size

        for i, result in enumerate(results):
            assert "x_final" in result
            assert "x_init" in result
            assert result["x_final"].shape == (1, 1, 32, 32)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_sampling_time_estimation(self):
        """Test sampling time estimation."""
        # Create minimal sampler for testing
        config = SamplingConfig(num_steps=10)
        sampler = EDMPosteriorSampler(None, None, config)  # Dummy initialization

        time_estimate = sampler.estimate_sampling_time(batch_size=2, image_size=64)

        assert "total_time_sec" in time_estimate
        assert "model_time_sec" in time_estimate
        assert "guidance_time_sec" in time_estimate
        assert "time_per_step_sec" in time_estimate
        assert time_estimate["batch_size"] == 2
        assert time_estimate["image_size"] == 64

        # Time should be positive
        assert time_estimate["total_time_sec"] > 0
        assert time_estimate["model_time_sec"] > 0


class TestIntegration:
    """Test integration with real components (if available)."""

    @pytest.mark.skipif(True, reason="Requires full model setup")
    def test_real_sampling_pipeline(self):
        """Test with real EDM model and guidance (integration test)."""
        # This would test the full pipeline with real components
        # Skipped for unit tests but useful for integration testing
        pass


if __name__ == "__main__":
    pytest.main([__file__])
