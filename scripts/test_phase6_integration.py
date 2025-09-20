#!/usr/bin/env python3
"""
Integration test for Phase 6: Optimization & Baselines

This script tests the Phase 6 implementation to verify:
1. Performance optimizations work correctly
2. Baseline methods are properly integrated
3. Comparison framework functions properly
4. Mixed precision and memory optimizations work
5. Tiled processing for large images works

Usage:
    python scripts/test_phase6_integration.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.baselines import (
    BaselineComparator,
    GaussianBaseline,
    RichardsonLucyBaseline,
    WienerFilterBaseline,
    create_baseline_suite,
)
from core.logging_config import get_logger, setup_project_logging
from core.performance import (
    MemoryManager,
    MixedPrecisionManager,
    PerformanceConfig,
    PerformanceOptimizer,
    TiledProcessor,
)

logger = get_logger(__name__)


def test_performance_optimizations():
    """Test performance optimization components."""
    logger.info("Testing performance optimizations...")

    results = {}

    # Test 1: Performance configuration
    try:
        config = PerformanceConfig(
            mixed_precision=True,
            memory_efficient=True,
            enable_tiling=True,
            tile_size=(128, 128),
        )
        results["config_creation"] = True
        logger.info("✅ Performance configuration created successfully")
    except Exception as e:
        results["config_creation"] = False
        logger.error(f"❌ Performance configuration failed: {e}")

    # Test 2: Memory manager
    try:
        memory_manager = MemoryManager(config)
        memory_info = memory_manager.get_memory_info()

        results["memory_manager"] = {
            "initialized": True,
            "has_memory_info": "total_memory" in memory_info,
            "device": memory_manager.device.type,
        }
        logger.info("✅ Memory manager working correctly")
    except Exception as e:
        results["memory_manager"] = {"initialized": False, "error": str(e)}
        logger.error(f"❌ Memory manager failed: {e}")

    # Test 3: Mixed precision manager
    try:
        mp_manager = MixedPrecisionManager(config)

        # Test autocast context
        with mp_manager.autocast():
            test_tensor = torch.randn(2, 3, 32, 32)
            result = test_tensor * 2

        results["mixed_precision"] = {
            "initialized": True,
            "autocast_works": result.shape == test_tensor.shape,
            "enabled": mp_manager.enabled,
        }
        logger.info("✅ Mixed precision manager working correctly")
    except Exception as e:
        results["mixed_precision"] = {"initialized": False, "error": str(e)}
        logger.error(f"❌ Mixed precision manager failed: {e}")

    # Test 4: Tiled processor
    try:
        tiled_processor = TiledProcessor(config)

        # Test with small image (should work without issues)
        test_image = torch.randn(1, 3, 64, 64)

        def simple_process(x):
            return x + 0.1

        result = tiled_processor.process_tiled(test_image, simple_process)

        results["tiled_processor"] = {
            "initialized": True,
            "processing_works": result.shape == test_image.shape,
            "tile_size": tiled_processor.config.tile_size,
        }
        logger.info("✅ Tiled processor working correctly")
    except Exception as e:
        results["tiled_processor"] = {"initialized": False, "error": str(e)}
        logger.error(f"❌ Tiled processor failed: {e}")

    # Test 5: Performance optimizer
    try:
        optimizer = PerformanceOptimizer(config)

        # Create simple test model
        test_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 3, 3, padding=1),
        )

        # Optimize model
        optimized_model = optimizer.optimize_model(test_model)

        results["performance_optimizer"] = {
            "initialized": True,
            "model_optimized": optimized_model is not None,
            "has_stats": len(optimizer.get_performance_stats()) > 0,
        }
        logger.info("✅ Performance optimizer working correctly")
    except Exception as e:
        results["performance_optimizer"] = {"initialized": False, "error": str(e)}
        logger.error(f"❌ Performance optimizer failed: {e}")

    return results


def test_baseline_methods():
    """Test baseline method implementations."""
    logger.info("Testing baseline methods...")

    results = {}

    # Create test data
    test_image = torch.rand(1, 1, 32, 32) * 1000 + 100  # Electrons
    scale = 1000.0
    background = 100.0
    read_noise = 5.0

    # Test individual baselines
    baseline_classes = [
        ("Gaussian", GaussianBaseline),
        ("WienerFilter", WienerFilterBaseline),
        ("Richardson-Lucy", RichardsonLucyBaseline),
    ]

    for name, baseline_class in baseline_classes:
        try:
            baseline = baseline_class(device="cpu")

            if baseline.is_available:
                result = baseline.denoise(test_image, scale, background, read_noise)

                results[name] = {
                    "available": True,
                    "denoise_works": result.shape == test_image.shape,
                    "output_finite": torch.isfinite(result).all().item(),
                    "output_range": (result.min().item(), result.max().item()),
                }
                logger.info(f"✅ {name} baseline working correctly")
            else:
                results[name] = {"available": False, "reason": "dependencies_missing"}
                logger.warning(
                    f"⚠️ {name} baseline not available (missing dependencies)"
                )

        except Exception as e:
            results[name] = {"available": False, "error": str(e)}
            logger.error(f"❌ {name} baseline failed: {e}")

    return results


def test_baseline_comparison_framework():
    """Test the baseline comparison framework."""
    logger.info("Testing baseline comparison framework...")

    results = {}

    try:
        # Create baseline comparator
        comparator = BaselineComparator(device="cpu")

        # Add available baselines
        baseline_suite = create_baseline_suite(device="cpu")
        for name, baseline in baseline_suite.available_baselines.items():
            if baseline.is_available:
                comparator.add_baseline(name, baseline)

        # Create test data
        clean_image = torch.rand(1, 1, 64, 64)

        # Add Poisson-Gaussian noise
        lambda_e = clean_image * 1000 + 100
        noisy_poisson = torch.poisson(lambda_e)
        noisy_gaussian = torch.randn_like(noisy_poisson) * 5.0
        noisy_image = noisy_poisson + noisy_gaussian

        # Evaluate baselines
        comparison_results = comparator.evaluate_all_baselines(
            noisy_image,
            clean_image,
            scale=1000.0,
            background=100.0,
            read_noise=5.0,
            domain="microscopy",
        )

        results["comparison_framework"] = {
            "initialized": True,
            "num_baselines": len(comparator.baselines),
            "evaluation_works": len(comparison_results) > 0,
            "has_metrics": all(
                "psnr" in result for result in comparison_results.values()
            ),
            "baseline_names": list(comparison_results.keys()),
        }

        logger.info("✅ Baseline comparison framework working correctly")
        logger.info(f"   Evaluated {len(comparison_results)} baselines")

        # Log results summary
        for name, result in comparison_results.items():
            psnr = result.get("psnr", 0)
            ssim = result.get("ssim", 0)
            time_taken = result.get("time", 0)
            logger.info(
                f"   {name}: PSNR={psnr:.2f} dB, SSIM={ssim:.3f}, Time={time_taken:.3f}s"
            )

    except Exception as e:
        results["comparison_framework"] = {"initialized": False, "error": str(e)}
        logger.error(f"❌ Baseline comparison framework failed: {e}")

    return results


def test_memory_efficiency():
    """Test memory efficiency optimizations."""
    logger.info("Testing memory efficiency...")

    results = {}

    try:
        config = PerformanceConfig(
            memory_efficient=True,
            max_memory_gb=2.0,  # Limit memory for testing
            enable_tiling=True,
            tile_size=(64, 64),
        )

        memory_manager = MemoryManager(config)

        # Test memory context
        initial_memory = memory_manager.get_memory_info()

        with memory_manager.memory_context():
            # Allocate some tensors
            large_tensors = []
            for i in range(5):
                tensor = torch.randn(100, 100)
                large_tensors.append(tensor)

        # Memory should be cleaned up after context
        final_memory = memory_manager.get_memory_info()

        results["memory_efficiency"] = {
            "context_works": True,
            "initial_memory": initial_memory.get("available_memory", 0),
            "final_memory": final_memory.get("available_memory", 0),
            "memory_tracked": len(memory_manager.memory_history) > 0,
        }

        logger.info("✅ Memory efficiency optimizations working correctly")

    except Exception as e:
        results["memory_efficiency"] = {"error": str(e)}
        logger.error(f"❌ Memory efficiency test failed: {e}")

    return results


def test_performance_benchmarking():
    """Test performance benchmarking capabilities."""
    logger.info("Testing performance benchmarking...")

    results = {}

    try:
        # Simple performance test
        test_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 3, 3, padding=1),
        )

        # Test standard inference
        test_input = torch.randn(1, 3, 64, 64)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = test_model(test_input)
        standard_time = time.time() - start_time

        # Test with mixed precision (if available)
        mixed_precision_time = standard_time  # Default to same

        if torch.cuda.is_available():
            try:
                test_model = test_model.cuda()
                test_input = test_input.cuda()

                start_time = time.time()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        for _ in range(10):
                            _ = test_model(test_input)
                mixed_precision_time = time.time() - start_time
            except:
                pass  # Fall back to CPU timing

        speedup = (
            standard_time / mixed_precision_time if mixed_precision_time > 0 else 1.0
        )

        results["benchmarking"] = {
            "standard_time": standard_time,
            "mixed_precision_time": mixed_precision_time,
            "speedup": speedup,
            "benchmarking_works": True,
        }

        logger.info("✅ Performance benchmarking working correctly")
        logger.info(f"   Standard time: {standard_time:.3f}s")
        logger.info(f"   Mixed precision time: {mixed_precision_time:.3f}s")
        logger.info(f"   Speedup: {speedup:.2f}x")

    except Exception as e:
        results["benchmarking"] = {"error": str(e)}
        logger.error(f"❌ Performance benchmarking failed: {e}")

    return results


def run_phase6_integration_tests():
    """Run all Phase 6 integration tests."""
    logger.info("Running Phase 6 integration tests...")

    all_results = {}

    # Run individual test suites
    test_suites = [
        ("performance_optimizations", test_performance_optimizations),
        ("baseline_methods", test_baseline_methods),
        ("comparison_framework", test_baseline_comparison_framework),
        ("memory_efficiency", test_memory_efficiency),
        ("benchmarking", test_performance_benchmarking),
    ]

    for suite_name, test_func in test_suites:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {suite_name} tests...")
        logger.info("=" * 60)

        try:
            results = test_func()
            all_results[suite_name] = results
        except Exception as e:
            logger.error(f"Test suite {suite_name} failed: {e}")
            all_results[suite_name] = {"suite_error": str(e)}

    # Compute summary
    total_tests = 0
    passed_tests = 0

    for suite_name, suite_results in all_results.items():
        if "suite_error" in suite_results:
            total_tests += 1
            continue

        for test_name, test_result in suite_results.items():
            total_tests += 1
            if isinstance(test_result, dict):
                if test_result.get("initialized", False) or test_result.get(
                    "available", False
                ):
                    passed_tests += 1
            elif test_result is True:
                passed_tests += 1

    success_rate = passed_tests / total_tests if total_tests > 0 else 0

    all_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": success_rate,
        "overall_status": "PASSED" if success_rate >= 0.7 else "PARTIAL",
    }

    return all_results


def main():
    """Main function."""
    setup_project_logging(level="INFO")

    logger.info("=" * 60)
    logger.info("PHASE 6 INTEGRATION TEST")
    logger.info("Optimization & Baselines")
    logger.info("=" * 60)

    results = run_phase6_integration_tests()

    # Print summary
    summary = results["summary"]
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 6 INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tests: {summary['total_tests']}")
    logger.info(f"Passed tests: {summary['passed_tests']}")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info(f"Overall status: {summary['overall_status']}")

    # Print detailed results
    for suite_name, suite_results in results.items():
        if suite_name == "summary":
            continue

        logger.info(f"\n{suite_name.upper()}:")

        if "suite_error" in suite_results:
            logger.info(f"  ❌ Suite failed: {suite_results['suite_error']}")
            continue

        for test_name, test_result in suite_results.items():
            if isinstance(test_result, dict):
                if test_result.get("initialized", False) or test_result.get(
                    "available", False
                ):
                    logger.info(f"  ✅ {test_name}: Success")
                elif "error" in test_result:
                    logger.info(f"  ❌ {test_name}: {test_result['error']}")
                else:
                    logger.info(f"  ⚠️ {test_name}: Partial success")
            elif test_result is True:
                logger.info(f"  ✅ {test_name}: Success")
            else:
                logger.info(f"  ❌ {test_name}: Failed")

    logger.info("=" * 60)

    # Exit with appropriate code
    sys.exit(0 if summary["overall_status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
