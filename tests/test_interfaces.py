"""
Test the core interfaces and base classes.
"""

from abc import ABC

import numpy as np
import pytest
import torch

from core.exceptions import PoissonDiffusionError
from core.interfaces import (
    BaseConfig,
    CalibrationManager,
    Dataset,
    DomainEncoder,
    Evaluator,
    GuidanceComputer,
    MetadataContainer,
    ModelWrapper,
    Sampler,
    Transform,
)


class TestInterfaces:
    """Test that all interfaces are properly defined."""

    def test_interfaces_are_abstract(self):
        """Test that all interfaces are abstract base classes."""
        interfaces = [
            Transform,
            GuidanceComputer,
            CalibrationManager,
            MetadataContainer,
            DomainEncoder,
            ModelWrapper,
            Sampler,
            Dataset,
            Evaluator,
        ]

        for interface in interfaces:
            assert issubclass(
                interface, ABC
            ), f"{interface.__name__} should be abstract"

            # Should not be able to instantiate directly
            with pytest.raises(TypeError):
                interface()

    def test_base_config_functionality(self):
        """Test BaseConfig functionality."""

        # Create a concrete config class
        from dataclasses import dataclass

        @dataclass
        class TestConfig(BaseConfig):
            param1: int = 10
            param2: str = "test"
            param3: float = 1.5

        # Test creation and conversion
        config = TestConfig(param1=20, param2="hello", param3=2.5)

        # Test to_dict
        config_dict = config.to_dict()
        expected = {"param1": 20, "param2": "hello", "param3": 2.5}
        assert config_dict == expected

        # Test from_dict
        restored = TestConfig.from_dict(config_dict)
        assert restored.param1 == 20
        assert restored.param2 == "hello"
        assert restored.param3 == 2.5

        # Test YAML conversion
        yaml_str = config.to_yaml()
        assert "param1: 20" in yaml_str
        assert "param2: hello" in yaml_str

        # Test from_yaml
        restored_yaml = TestConfig.from_yaml(yaml_str)
        assert restored_yaml.param1 == 20
        assert restored_yaml.param2 == "hello"
        assert restored_yaml.param3 == 2.5


class TestExceptions:
    """Test custom exceptions."""

    def test_exception_hierarchy(self):
        """Test that all exceptions inherit from base exception."""
        from core.exceptions import (
            CalibrationError,
            ConfigurationError,
            DataError,
            DomainError,
            GuidanceError,
            MetadataError,
            ModelError,
            NumericalStabilityError,
            PoissonDiffusionError,
            TransformError,
        )

        exceptions = [
            CalibrationError,
            TransformError,
            MetadataError,
            GuidanceError,
            ModelError,
            DataError,
            NumericalStabilityError,
            DomainError,
            ConfigurationError,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, PoissonDiffusionError)
            assert issubclass(exc_class, Exception)

            # Test that they can be raised
            with pytest.raises(exc_class):
                raise exc_class("Test message")


class TestUtils:
    """Test utility functions."""

    def test_tensor_validity_checking(self):
        """Test tensor validity checking."""
        from core.utils import NumericalStabilityError, check_tensor_validity

        # Valid tensor should pass
        valid_tensor = torch.randn(10, 10)
        check_tensor_validity(valid_tensor)  # Should not raise

        # NaN tensor should fail
        nan_tensor = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(NumericalStabilityError, match="NaN"):
            check_tensor_validity(nan_tensor)

        # Inf tensor should fail
        inf_tensor = torch.tensor([1.0, float("inf"), 3.0])
        with pytest.raises(NumericalStabilityError, match="Inf"):
            check_tensor_validity(inf_tensor)

        # Range violations should fail
        range_tensor = torch.tensor([-1.0, 0.5, 2.0])
        with pytest.raises(NumericalStabilityError, match="below minimum"):
            check_tensor_validity(range_tensor, min_val=0.0)

        with pytest.raises(NumericalStabilityError, match="above maximum"):
            check_tensor_validity(range_tensor, max_val=1.0)

    def test_variance_stabilization(self):
        """Test variance stabilization."""
        from core.utils import stabilize_variance

        # Test with some negative/zero values
        variance = torch.tensor([0.0, -0.1, 0.05, 1.0, 2.0])
        stabilized = stabilize_variance(variance, eps=0.1)

        # All values should be >= eps
        assert (stabilized >= 0.1).all()

        # Values above eps should be unchanged
        assert stabilized[3] == 1.0
        assert stabilized[4] == 2.0

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        from core.utils import clip_gradients

        # Test with large gradients
        gradients = torch.tensor([-15.0, -5.0, 0.0, 5.0, 15.0])
        clipped = clip_gradients(gradients, max_norm=10.0)

        # Values should be clipped to [-10, 10]
        expected = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
        assert torch.allclose(clipped, expected)

    def test_device_selection(self):
        """Test device selection."""
        from core.utils import get_device

        device = get_device(prefer_cuda=True)
        assert isinstance(device, torch.device)

        # Should be cuda if available, cpu otherwise
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"

        # Should be cpu if cuda not preferred
        cpu_device = get_device(prefer_cuda=False)
        assert cpu_device.type == "cpu"

    def test_domain_validation(self):
        """Test domain validation."""
        from core.exceptions import ConfigurationError
        from core.utils import validate_domain

        # Valid domains should pass
        assert validate_domain("photography") == "photography"
        assert validate_domain("MICROSCOPY") == "microscopy"
        assert validate_domain(" Astronomy ") == "astronomy"

        # Invalid domain should fail
        with pytest.raises(ConfigurationError, match="Invalid domain"):
            validate_domain("invalid_domain")

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        from core.utils import estimate_memory_usage

        # Test with known shapes
        shapes = [(1, 1, 128, 128), (1, 1, 128, 128)]  # Two 128x128 images
        memory_gb = estimate_memory_usage(shapes, dtype=torch.float32)

        # Should be reasonable (small for test shapes)
        assert 0 < memory_gb < 1.0  # Less than 1GB for small test

    def test_formatting_functions(self):
        """Test formatting utility functions."""
        from core.utils import format_bytes, format_time

        # Test byte formatting
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024**2) == "1.0 MB"
        assert format_bytes(1024**3) == "1.0 GB"

        # Test time formatting
        assert format_time(30) == "30.0s"
        assert format_time(90) == "1.5m"
        assert format_time(3660) == "1.0h"


if __name__ == "__main__":
    pytest.main([__file__])
