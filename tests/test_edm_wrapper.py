"""
Comprehensive tests for EDM wrapper with domain conditioning.

Tests cover:
- EDM model integration and wrapper functionality
- Domain encoding and conditioning vector generation
- FiLM layer implementation
- Model initialization and configuration
- Memory usage estimation
- Integration with external EDM codebase
"""

# Import EDM wrapper components
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))

from core.exceptions import ConfigurationError, ModelError
from models.edm_wrapper import (
    EDM_AVAILABLE,
    DomainEncoder,
    EDMConfig,
    EDMModelWrapper,
    FiLMLayer,
    create_domain_edm_wrapper,
    create_edm_wrapper,
    test_edm_wrapper_basic,
)


class TestEDMConfig:
    """Test EDM configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EDMConfig()

        assert config.img_resolution == 128
        assert config.img_channels == 1
        assert config.model_channels == 128
        assert config.label_dim == 6
        assert config.channel_mult == [1, 2, 2, 2]
        assert config.attn_resolutions == [16]
        assert config.sigma_min == 0.002
        assert config.sigma_max == 80.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = EDMConfig(
            img_resolution=256,
            model_channels=192,
            channel_mult=[1, 2, 4, 8],
            attn_resolutions=[32, 16],
        )

        assert config.img_resolution == 256
        assert config.model_channels == 192
        assert config.channel_mult == [1, 2, 4, 8]
        assert config.attn_resolutions == [32, 16]

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = EDMConfig(img_resolution=64, model_channels=96)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["img_resolution"] == 64
        assert config_dict["model_channels"] == 96
        assert config_dict["label_dim"] == 6


class TestDomainEncoder:
    """Test domain encoding functionality."""

    @pytest.fixture
    def encoder(self):
        """Create domain encoder for testing."""
        return DomainEncoder()

    def test_single_domain_encoding(self, encoder):
        """Test encoding single domain parameters."""
        condition = encoder.encode_domain(
            domain="photography", scale=1000.0, read_noise=3.0, background=10.0
        )

        assert condition.shape == (1, 6)

        # Check domain one-hot (photography = index 0)
        assert condition[0, 0] == 1.0  # photography
        assert condition[0, 1] == 0.0  # microscopy
        assert condition[0, 2] == 0.0  # astronomy

        # Check that other parameters are finite
        assert torch.isfinite(condition[0, 3:]).all()

    def test_batch_domain_encoding(self, encoder):
        """Test encoding batch of domain parameters."""
        domains = ["photography", "microscopy", "astronomy"]
        scales = [1000.0, 500.0, 2000.0]
        read_noises = [3.0, 2.0, 5.0]
        backgrounds = [10.0, 5.0, 20.0]

        condition = encoder.encode_domain(
            domain=domains,
            scale=torch.tensor(scales),
            read_noise=torch.tensor(read_noises),
            background=torch.tensor(backgrounds),
        )

        assert condition.shape == (3, 6)

        # Check domain one-hot encodings
        assert condition[0, 0] == 1.0  # photography
        assert condition[1, 1] == 1.0  # microscopy
        assert condition[2, 2] == 1.0  # astronomy

        # Check all values are finite
        assert torch.isfinite(condition).all()

    def test_tensor_domain_encoding(self, encoder):
        """Test encoding with tensor domain indices."""
        domain_indices = torch.tensor([0, 1, 2])  # photography, microscopy, astronomy

        condition = encoder.encode_domain(
            domain=domain_indices, scale=1000.0, read_noise=3.0, background=10.0
        )

        assert condition.shape == (3, 6)

        # Check one-hot encodings
        expected_onehot = torch.eye(3)
        torch.testing.assert_close(condition[:, :3], expected_onehot)

    def test_device_handling(self, encoder):
        """Test proper device handling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")

        condition = encoder.encode_domain(
            domain="photography",
            scale=1000.0,
            read_noise=3.0,
            background=10.0,
            device=device,
        )

        assert condition.device.type == device.type

    def test_decode_domain(self, encoder):
        """Test decoding conditioning vector."""
        # Encode
        original_scale = 1000.0
        original_read_noise = 3.0
        original_background = 10.0

        condition = encoder.encode_domain(
            domain="microscopy",
            scale=original_scale,
            read_noise=original_read_noise,
            background=original_background,
        )

        # Decode
        decoded = encoder.decode_domain(condition)

        assert decoded["domain_idx"][0] == 1  # microscopy

        # Check approximate equality (allowing for normalization/denormalization)
        assert abs(decoded["scale"][0] - original_scale) < 50.0
        assert abs(decoded["read_noise"][0] - original_read_noise) < 0.5
        assert abs(decoded["background"][0] - original_background) < 2.0

    def test_invalid_domain(self, encoder):
        """Test error handling for invalid domain."""
        with pytest.raises(KeyError):
            encoder.encode_domain(
                domain="invalid_domain", scale=1000.0, read_noise=3.0, background=10.0
            )

    def test_relative_parameters(self, encoder):
        """Test relative parameter computation."""
        # Test case where read_noise and background are significant fractions of scale
        condition = encoder.encode_domain(
            domain="photography",
            scale=100.0,  # Low scale
            read_noise=10.0,  # 10% of scale
            background=5.0,  # 5% of scale
        )

        # Relative read noise should be ~0.1
        rel_read_noise = condition[0, 4].item()
        assert abs(rel_read_noise - 0.1) < 0.01

        # Relative background should be ~0.05
        rel_background = condition[0, 5].item()
        assert abs(rel_background - 0.05) < 0.01


class TestFiLMLayer:
    """Test FiLM (Feature-wise Linear Modulation) layer."""

    def test_film_layer_creation(self):
        """Test FiLM layer initialization."""
        film = FiLMLayer(feature_dim=64, condition_dim=6)

        assert film.feature_dim == 64
        assert film.condition_dim == 6
        assert isinstance(film.gamma_layer, torch.nn.Linear)
        assert isinstance(film.beta_layer, torch.nn.Linear)

    def test_film_forward(self):
        """Test FiLM layer forward pass."""
        film = FiLMLayer(feature_dim=32, condition_dim=6)

        # Test data
        batch_size = 2
        features = torch.randn(batch_size, 32, 16, 16)
        condition = torch.randn(batch_size, 6)

        # Forward pass
        output = film(features, condition)

        assert output.shape == features.shape
        assert torch.isfinite(output).all()

    def test_film_identity_initialization(self):
        """Test that FiLM layer initializes to identity transformation."""
        film = FiLMLayer(feature_dim=16, condition_dim=4)

        # Zero conditioning should give identity transformation
        features = torch.randn(1, 16, 8, 8)
        zero_condition = torch.zeros(1, 4)

        output = film(features, zero_condition)

        # Should be approximately equal to input (gamma=1, beta=0)
        torch.testing.assert_close(output, features, atol=1e-6, rtol=1e-6)

    def test_film_modulation_effect(self):
        """Test that FiLM layer actually modulates features."""
        film = FiLMLayer(feature_dim=8, condition_dim=2)

        # Manually set some weights to make the layer responsive
        with torch.no_grad():
            film.gamma_layer.weight[
                0, 0
            ] = 1.0  # First feature responds to first condition
            film.beta_layer.weight[
                1, 1
            ] = 1.0  # Second feature responds to second condition

        features = torch.ones(1, 8, 4, 4)  # All ones

        # Different conditions should give different outputs
        condition1 = torch.tensor([[1.0, 0.0]])
        condition2 = torch.tensor([[0.0, 1.0]])

        output1 = film(features, condition1)
        output2 = film(features, condition2)

        # Outputs should be different
        assert not torch.allclose(output1, output2)


@pytest.mark.skipif(not EDM_AVAILABLE, reason="EDM not available")
class TestEDMModelWrapper:
    """Test EDM model wrapper functionality."""

    @pytest.fixture
    def small_config(self):
        """Create small configuration for fast testing."""
        return EDMConfig(
            img_resolution=32,
            img_channels=1,
            model_channels=32,
            num_blocks=2,
            channel_mult=[1, 2],
        )

    @pytest.fixture
    def wrapper(self, small_config):
        """Create EDM wrapper for testing."""
        return EDMModelWrapper(config=small_config)

    def test_wrapper_initialization(self, small_config):
        """Test EDM wrapper initialization."""
        wrapper = EDMModelWrapper(config=small_config)

        assert wrapper.config == small_config
        assert wrapper.conditioning_mode == "class_labels"
        assert wrapper.condition_dim == 6
        assert hasattr(wrapper, "edm_model")
        assert hasattr(wrapper, "domain_encoder")

    def test_wrapper_invalid_config(self):
        """Test wrapper with invalid configuration."""
        invalid_config = EDMConfig(label_dim=4)  # Should be 6

        with pytest.raises(ConfigurationError):
            EDMModelWrapper(config=invalid_config)

    def test_wrapper_forward_with_condition(self, wrapper):
        """Test forward pass with pre-computed conditioning."""
        batch_size = 2
        x = torch.randn(batch_size, 1, 32, 32)
        sigma = torch.tensor([1.0, 0.5])
        condition = torch.randn(batch_size, 6)

        with torch.no_grad():
            output = wrapper(x, sigma, condition=condition)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_wrapper_forward_with_domain_params(self, wrapper):
        """Test forward pass with domain parameters."""
        batch_size = 2
        x = torch.randn(batch_size, 1, 32, 32)
        sigma = torch.tensor([1.0, 0.5])

        with torch.no_grad():
            output = wrapper(
                x,
                sigma,
                domain=["photography", "microscopy"],
                scale=[1000.0, 500.0],
                read_noise=[3.0, 2.0],
                background=[10.0, 5.0],
            )

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_wrapper_missing_parameters(self, wrapper):
        """Test error handling for missing parameters."""
        x = torch.randn(1, 1, 32, 32)
        sigma = torch.tensor([1.0])

        with pytest.raises(
            ValueError, match="Must provide either 'condition' or all domain parameters"
        ):
            wrapper(x, sigma, domain="photography")  # Missing other parameters

    def test_wrapper_batch_size_mismatch(self, wrapper):
        """Test handling of batch size mismatch."""
        x = torch.randn(2, 1, 32, 32)
        sigma = torch.tensor([1.0, 0.5])
        condition = torch.randn(1, 6)  # Single condition for batch of 2

        with torch.no_grad():
            output = wrapper(x, sigma, condition=condition)

        # Should expand condition to match batch size
        assert output.shape == x.shape

    def test_encode_conditioning(self, wrapper):
        """Test conditioning encoding method."""
        condition = wrapper.encode_conditioning(
            domain=["photography", "microscopy"],
            scale=[1000.0, 500.0],
            read_noise=[3.0, 2.0],
            background=[10.0, 5.0],
        )

        assert condition.shape == (2, 6)
        assert torch.isfinite(condition).all()

    def test_get_model_info(self, wrapper):
        """Test model information retrieval."""
        info = wrapper.get_model_info()

        assert "model_type" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "condition_dim" in info
        assert info["condition_dim"] == 6
        assert info["model_type"] == "edm_wrapper"

    def test_memory_estimation(self, wrapper):
        """Test memory usage estimation."""
        memory_info = wrapper.estimate_memory_usage(batch_size=4)

        assert "parameters_mb" in memory_info
        assert "total_estimated_mb" in memory_info
        assert memory_info["batch_size"] == 4
        assert memory_info["total_estimated_mb"] > 0


class TestFactoryFunctions:
    """Test factory functions for creating EDM wrappers."""

    @pytest.mark.skipif(not EDM_AVAILABLE, reason="EDM not available")
    def test_create_edm_wrapper(self):
        """Test basic EDM wrapper creation."""
        wrapper = create_edm_wrapper(img_resolution=64, model_channels=64)

        assert isinstance(wrapper, EDMModelWrapper)
        assert wrapper.config.img_resolution == 64
        assert wrapper.config.model_channels == 64

    @pytest.mark.skipif(not EDM_AVAILABLE, reason="EDM not available")
    def test_create_domain_edm_wrapper(self):
        """Test domain-specific EDM wrapper creation."""
        # Test each domain
        for domain in ["photography", "microscopy", "astronomy"]:
            wrapper = create_domain_edm_wrapper(domain=domain, img_resolution=64)

            assert isinstance(wrapper, EDMModelWrapper)
            assert wrapper.config.img_resolution == 64

    def test_create_domain_edm_wrapper_invalid(self):
        """Test error handling for invalid domain."""
        with pytest.raises(ValueError, match="Unknown domain"):
            create_domain_edm_wrapper(domain="invalid_domain")


class TestEDMIntegration:
    """Test integration with external EDM codebase."""

    @pytest.mark.skipif(not EDM_AVAILABLE, reason="EDM not available")
    def test_edm_import(self):
        """Test that EDM can be imported successfully."""
        # This should not raise an exception if EDM is properly set up
        from models.edm_wrapper import EDMPrecond

        assert EDMPrecond is not None

    @pytest.mark.skipif(not EDM_AVAILABLE, reason="EDM not available")
    def test_basic_edm_functionality(self):
        """Test basic EDM model functionality."""
        success = test_edm_wrapper_basic()
        assert success, "Basic EDM wrapper test failed"

    @pytest.mark.skipif(not EDM_AVAILABLE, reason="EDM not available")
    def test_edm_conditioning_pathway(self):
        """Test that EDM accepts class_labels conditioning."""
        from models.edm_wrapper import EDMPrecond

        # Create minimal EDM model
        model = EDMPrecond(
            img_resolution=32, img_channels=1, model_channels=32, label_dim=6
        )

        # Test forward pass with conditioning
        x = torch.randn(1, 1, 32, 32)
        sigma = torch.tensor([1.0])
        class_labels = torch.randn(1, 6)

        with torch.no_grad():
            output = model(x, sigma, class_labels=class_labels)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_edm_not_available(self):
        """Test behavior when EDM is not available."""
        with patch("models.edm_wrapper.EDM_AVAILABLE", False):
            with pytest.raises(ModelError, match="EDM not available"):
                EDMModelWrapper()

    @pytest.mark.skipif(not EDM_AVAILABLE, reason="EDM not available")
    def test_invalid_conditioning_mode(self):
        """Test invalid conditioning mode."""
        config = EDMConfig(img_resolution=32, model_channels=32)

        with pytest.raises(ConfigurationError, match="Unknown conditioning_mode"):
            EDMModelWrapper(config=config, conditioning_mode="invalid_mode")

    @pytest.mark.skipif(not EDM_AVAILABLE, reason="EDM not available")
    def test_film_not_implemented(self):
        """Test that FiLM conditioning raises NotImplementedError."""
        config = EDMConfig(img_resolution=32, model_channels=32)
        wrapper = EDMModelWrapper(config=config, conditioning_mode="film")

        x = torch.randn(1, 1, 32, 32)
        sigma = torch.tensor([1.0])
        condition = torch.randn(1, 6)

        with pytest.raises(NotImplementedError):
            wrapper(x, sigma, condition=condition)


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    @pytest.fixture
    def encoder(self):
        return DomainEncoder()

    def test_extreme_scale_values(self, encoder):
        """Test encoding with extreme scale values."""
        # Very small scale
        condition_small = encoder.encode_domain(
            domain="photography", scale=1e-3, read_noise=1.0, background=0.0
        )
        assert torch.isfinite(condition_small).all()

        # Very large scale
        condition_large = encoder.encode_domain(
            domain="photography", scale=1e6, read_noise=1.0, background=0.0
        )
        assert torch.isfinite(condition_large).all()

    def test_zero_scale_handling(self, encoder):
        """Test handling of zero or near-zero scale."""
        # Should clamp to minimum value to avoid division by zero
        condition = encoder.encode_domain(
            domain="photography", scale=0.0, read_noise=1.0, background=0.0
        )
        assert torch.isfinite(condition).all()

    def test_negative_parameters(self, encoder):
        """Test handling of negative parameters."""
        # Negative read noise (should still work)
        condition = encoder.encode_domain(
            domain="photography",
            scale=1000.0,
            read_noise=-1.0,  # Negative (unphysical but should not crash)
            background=0.0,
        )
        assert torch.isfinite(condition).all()


if __name__ == "__main__":
    pytest.main([__file__])
