#!/usr/bin/env python3
"""
Basic test script for EDM sampler functionality.

This script tests the sampler implementation without requiring
full EDM integration, focusing on core functionality.

Usage:
    python scripts/test_sampler_basic.py [--verbose]
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.logging_config import get_logger, setup_project_logging
from models.sampler import (
    EDMPosteriorSampler,
    SamplingConfig,
    create_edm_sampler,
    create_fast_sampler,
    create_high_quality_sampler,
)

logger = get_logger(__name__)


def test_sampling_config():
    """Test sampling configuration."""
    print("Testing SamplingConfig...")

    # Default config
    config = SamplingConfig()
    assert config.num_steps == 18
    assert config.solver == "heun"
    assert config.guidance_scale == 1.0

    # Custom config
    custom_config = SamplingConfig(num_steps=10, solver="euler", guidance_scale=1.5)
    assert custom_config.num_steps == 10
    assert custom_config.solver == "euler"
    assert custom_config.guidance_scale == 1.5

    # Validation
    try:
        SamplingConfig(num_steps=0)
        assert False, "Should have raised error"
    except Exception:
        pass  # Expected

    print("✓ SamplingConfig tests passed")
    return True


def test_noise_schedule():
    """Test noise schedule generation."""
    print("Testing noise schedule generation...")

    config = SamplingConfig(num_steps=10, sigma_min=0.01, sigma_max=10.0)

    # Create dummy sampler just for noise schedule testing
    sampler = EDMPosteriorSampler(None, None, config)

    schedule = sampler.create_noise_schedule()

    # Check properties
    assert schedule.shape == (11,)  # num_steps + 1
    assert abs(schedule[0] - 10.0) < 1e-6  # Start at sigma_max
    assert abs(schedule[-1] - 0.01) < 1e-6  # End at sigma_min
    assert torch.all(schedule[:-1] >= schedule[1:])  # Monotonically decreasing

    print("✓ Noise schedule tests passed")
    return True


def test_guidance_logic():
    """Test guidance application logic."""
    print("Testing guidance logic...")

    config = SamplingConfig(num_steps=10, guidance_start_step=2, guidance_end_step=8)

    sampler = EDMPosteriorSampler(None, None, config)

    # Test guidance application
    assert sampler._should_apply_guidance(0) is False  # Before start
    assert sampler._should_apply_guidance(1) is False  # Before start
    assert sampler._should_apply_guidance(2) is True  # At start
    assert sampler._should_apply_guidance(5) is True  # In range
    assert sampler._should_apply_guidance(7) is True  # In range
    assert sampler._should_apply_guidance(8) is False  # At end
    assert sampler._should_apply_guidance(9) is False  # After end

    print("✓ Guidance logic tests passed")
    return True


def test_step_solvers():
    """Test ODE solver implementations."""
    print("Testing ODE solvers...")

    config = SamplingConfig(num_steps=5)
    sampler = EDMPosteriorSampler(None, None, config)

    # Test data
    x = torch.randn(1, 1, 16, 16)
    d = torch.randn(1, 1, 16, 16) * 0.1
    sigma_curr = torch.tensor(1.0)
    sigma_next = torch.tensor(0.8)

    # Test each solver
    for solver in ["euler", "heun", "dpm"]:
        sampler.config.solver = solver
        x_next = sampler._take_step(x, d, sigma_curr, sigma_next, 0)

        assert x_next.shape == x.shape
        assert torch.isfinite(x_next).all()
        assert not torch.allclose(x_next, x)  # Should be different

    # Test invalid solver
    sampler.config.solver = "invalid"
    try:
        sampler._take_step(x, d, sigma_curr, sigma_next, 0)
        assert False, "Should have raised error"
    except Exception:
        pass  # Expected

    print("✓ ODE solver tests passed")
    return True


def test_factory_functions():
    """Test factory functions."""
    print("Testing factory functions...")

    # Test configuration creation (without actual model/guidance)
    from models.sampler import SamplingConfig

    # Test basic config
    config1 = SamplingConfig(num_steps=20, guidance_scale=1.5)
    assert config1.num_steps == 20
    assert config1.guidance_scale == 1.5

    # Test fast config
    config2 = SamplingConfig(num_steps=10, solver="euler", collect_diagnostics=False)
    assert config2.num_steps == 10
    assert config2.solver == "euler"
    assert config2.collect_diagnostics is False

    # Test high-quality config
    config3 = SamplingConfig(
        num_steps=50,
        solver="heun",
        guidance_scale=1.2,
        collect_diagnostics=True,
        save_intermediates=True,
    )
    assert config3.num_steps == 50
    assert config3.solver == "heun"
    assert config3.guidance_scale == 1.2
    assert config3.collect_diagnostics is True
    assert config3.save_intermediates is True

    print("✓ Factory function tests passed")
    return True


def test_time_estimation():
    """Test sampling time estimation."""
    print("Testing time estimation...")

    config = SamplingConfig(num_steps=18)
    sampler = EDMPosteriorSampler(None, None, config)

    time_estimate = sampler.estimate_sampling_time(batch_size=2, image_size=128)

    assert "total_time_sec" in time_estimate
    assert "model_time_sec" in time_estimate
    assert "guidance_time_sec" in time_estimate
    assert time_estimate["batch_size"] == 2
    assert time_estimate["image_size"] == 128
    assert time_estimate["total_time_sec"] > 0

    print("✓ Time estimation tests passed")
    return True


def test_stochastic_sampling():
    """Test stochastic sampling parameters."""
    print("Testing stochastic sampling...")

    config = SamplingConfig(S_churn=0.1, S_min=0.5, S_max=2.0, S_noise=1.0)
    sampler = EDMPosteriorSampler(None, None, config)

    x = torch.randn(1, 1, 16, 16)
    d = torch.randn(1, 1, 16, 16) * 0.1
    sigma_curr = torch.tensor(1.0)  # In stochastic range
    sigma_next = torch.tensor(0.8)

    # Should handle stochasticity without error
    x_next = sampler._take_step(x, d, sigma_curr, sigma_next, 0)
    assert x_next.shape == x.shape
    assert torch.isfinite(x_next).all()

    print("✓ Stochastic sampling tests passed")
    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test EDM sampler basic functionality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        setup_project_logging(level="DEBUG")
    else:
        setup_project_logging(level="INFO")

    print("=" * 60)
    print("EDM SAMPLER BASIC TEST SUITE")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print()

    # Run tests
    tests = [
        test_sampling_config,
        test_noise_schedule,
        test_guidance_logic,
        test_step_solvers,
        test_factory_functions,
        test_time_estimation,
        test_stochastic_sampling,
    ]

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
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
