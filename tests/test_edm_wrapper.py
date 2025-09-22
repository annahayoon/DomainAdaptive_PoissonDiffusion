#!/usr/bin/env python3
"""
Test script for EDM wrapper functionality.

This script tests the EDM wrapper implementation including:
- Domain encoding
- Model creation and forward pass
- Memory usage estimation
- Integration with external EDM codebase

Usage:
    python scripts/test_edm_wrapper.py [--skip-edm] [--verbose]
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.logging_config import get_logger, setup_project_logging
from models.edm_wrapper import (
    EDM_AVAILABLE,
    DomainEncoder,
    EDMConfig,
    EDMModelWrapper,
    FiLMLayer,
    create_domain_edm_wrapper,
    create_edm_wrapper,
)

logger = get_logger(__name__)


def test_domain_encoder():
    """Test domain encoder functionality."""
    print("Testing Domain Encoder...")

    encoder = DomainEncoder()

    # Test single domain
    condition = encoder.encode_domain(
        domain="photography", scale=1000.0, read_noise=3.0, background=10.0
    )

    assert condition.shape == (1, 6), f"Expected shape (1, 6), got {condition.shape}"
    assert torch.isfinite(condition).all(), "Condition contains non-finite values"

    # Test batch encoding
    batch_condition = encoder.encode_domain(
        domain=["photography", "microscopy", "astronomy"],
        scale=[1000.0, 500.0, 2000.0],
        read_noise=[3.0, 2.0, 5.0],
        background=[10.0, 5.0, 20.0],
    )

    assert batch_condition.shape == (
        3,
        6,
    ), f"Expected shape (3, 6), got {batch_condition.shape}"
    assert torch.isfinite(
        batch_condition
    ).all(), "Batch condition contains non-finite values"

    # Test decode
    decoded = encoder.decode_domain(condition)
    assert "domain_idx" in decoded
    assert "scale" in decoded

    print("✓ Domain Encoder tests passed")
    return True


def test_film_layer():
    """Test FiLM layer functionality."""
    print("Testing FiLM Layer...")

    film = FiLMLayer(feature_dim=32, condition_dim=6)

    # Test forward pass
    features = torch.randn(2, 32, 16, 16)
    condition = torch.randn(2, 6)

    output = film(features, condition)

    assert (
        output.shape == features.shape
    ), f"Output shape {output.shape} != input shape {features.shape}"
    assert torch.isfinite(output).all(), "Output contains non-finite values"

    # Test identity with zero condition
    zero_condition = torch.zeros(1, 6)
    identity_features = torch.randn(1, 32, 8, 8)
    identity_output = film(identity_features, zero_condition)

    # Should be close to identity (gamma=1, beta=0)
    torch.testing.assert_close(identity_output, identity_features, atol=1e-5, rtol=1e-5)

    print("✓ FiLM Layer tests passed")
    return True


def test_edm_config():
    """Test EDM configuration."""
    print("Testing EDM Config...")

    # Default config
    config = EDMConfig()
    assert config.img_resolution == 128
    assert config.label_dim == 6

    # Custom config
    custom_config = EDMConfig(img_resolution=256, model_channels=192)
    assert custom_config.img_resolution == 256
    assert custom_config.model_channels == 192

    # Serialization
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["img_resolution"] == 128

    print("✓ EDM Config tests passed")
    return True


def test_edm_wrapper_creation():
    """Test EDM wrapper creation (without EDM)."""
    print("Testing EDM Wrapper Creation...")

    if not EDM_AVAILABLE:
        print("⚠ EDM not available, testing error handling...")

        try:
            wrapper = EDMModelWrapper()
            assert False, "Should have raised ModelError"
        except Exception as e:
            assert "EDM not available" in str(e)
            print("✓ Proper error handling when EDM unavailable")
            return True

    # Test with small config for speed
    config = EDMConfig(img_resolution=32, model_channels=32, num_blocks=2)

    wrapper = EDMModelWrapper(config=config)

    # Test model info
    info = wrapper.get_model_info()
    assert info["condition_dim"] == 6
    assert info["model_type"] == "edm_wrapper"

    # Test memory estimation
    memory_info = wrapper.estimate_memory_usage(batch_size=2)
    assert "total_estimated_mb" in memory_info
    assert memory_info["total_estimated_mb"] > 0

    # Test conditioning encoding
    condition = wrapper.encode_conditioning(
        domain=["photography", "microscopy"],
        scale=[1000.0, 500.0],
        read_noise=[3.0, 2.0],
        background=[10.0, 5.0],
    )
    assert condition.shape == (2, 6)

    print("✓ EDM Wrapper creation tests passed")
    return True


def test_edm_wrapper_forward():
    """Test EDM wrapper forward pass."""
    if not EDM_AVAILABLE:
        print("⚠ Skipping forward pass test (EDM not available)")
        return True

    print("Testing EDM Wrapper Forward Pass...")

    # Small model for testing
    config = EDMConfig(img_resolution=32, model_channels=32, num_blocks=2)

    wrapper = EDMModelWrapper(config=config)
    wrapper.eval()

    # Test data
    batch_size = 2
    x = torch.randn(batch_size, 1, 32, 32)
    sigma = torch.tensor([1.0, 0.5])

    # Test with pre-computed condition
    condition = wrapper.encode_conditioning(
        domain=["photography", "microscopy"],
        scale=[1000.0, 500.0],
        read_noise=[3.0, 2.0],
        background=[10.0, 5.0],
    )

    with torch.no_grad():
        output = wrapper(x, sigma, condition=condition)

    assert (
        output.shape == x.shape
    ), f"Output shape {output.shape} != input shape {x.shape}"
    assert torch.isfinite(output).all(), "Output contains non-finite values"

    # Test with domain parameters
    with torch.no_grad():
        output2 = wrapper(
            x,
            sigma,
            domain=["photography", "microscopy"],
            scale=[1000.0, 500.0],
            read_noise=[3.0, 2.0],
            background=[10.0, 5.0],
        )

    assert output2.shape == x.shape
    torch.testing.assert_close(output, output2, atol=1e-6, rtol=1e-6)

    print("✓ EDM Wrapper forward pass tests passed")
    return True


def test_factory_functions():
    """Test factory functions."""
    if not EDM_AVAILABLE:
        print("⚠ Skipping factory function tests (EDM not available)")
        return True

    print("Testing Factory Functions...")

    # Test basic wrapper creation
    wrapper = create_edm_wrapper(img_resolution=64, model_channels=64)
    assert wrapper.config.img_resolution == 64
    assert wrapper.config.model_channels == 64

    # Test domain-specific wrappers
    for domain in ["photography", "microscopy", "astronomy"]:
        domain_wrapper = create_domain_edm_wrapper(domain=domain, img_resolution=64)
        assert domain_wrapper.config.img_resolution == 64

    print("✓ Factory function tests passed")
    return True


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("Testing Numerical Stability...")

    encoder = DomainEncoder()

    # Test extreme values
    test_cases = [
        # (scale, read_noise, background)
        (1e-3, 1.0, 0.0),  # Very small scale
        (1e6, 1.0, 0.0),  # Very large scale
        (100.0, 0.0, 0.0),  # Zero read noise
        (100.0, 1.0, -5.0),  # Negative background
    ]

    for scale, read_noise, background in test_cases:
        condition = encoder.encode_domain(
            domain="photography",
            scale=scale,
            read_noise=read_noise,
            background=background,
        )

        assert torch.isfinite(
            condition
        ).all(), f"Non-finite values for scale={scale}, read_noise={read_noise}, background={background}"

    print("✓ Numerical stability tests passed")
    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test EDM wrapper functionality")
    parser.add_argument(
        "--skip-edm", action="store_true", help="Skip tests requiring EDM"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        setup_project_logging(level="DEBUG")
    else:
        setup_project_logging(level="INFO")

    print("=" * 60)
    print("EDM WRAPPER TEST SUITE")
    print("=" * 60)
    print(f"EDM Available: {EDM_AVAILABLE}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print()

    # Run tests
    tests = [
        test_domain_encoder,
        test_film_layer,
        test_edm_config,
        test_numerical_stability,
    ]

    if not args.skip_edm:
        tests.extend(
            [
                test_edm_wrapper_creation,
                test_edm_wrapper_forward,
                test_factory_functions,
            ]
        )

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} failed with exception: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()

    print()
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
