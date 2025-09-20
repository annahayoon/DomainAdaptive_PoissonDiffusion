#!/usr/bin/env python3
"""
Integration test for Task 4.3: Patch extraction for large images

This script tests the patch processing implementation to verify:
1. Core patch extraction and reconstruction works
2. Integration with existing calibration system
3. Integration with existing data loaders
4. Memory-efficient processing capabilities
5. Quality of reconstruction

Usage:
    python scripts/test_task_4_3_integration.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.calibration import CalibrationParams, SensorCalibration
from core.logging_config import get_logger, setup_project_logging
from core.patch_processing import (
    MemoryEfficientPatchProcessor,
    PatchExtractor,
    PatchReconstructor,
    calculate_optimal_patch_size,
)
from data.patch_dataset import PatchDataset, PatchDatasetConfig

logger = get_logger(__name__)


def test_basic_patch_processing():
    """Test basic patch extraction and reconstruction."""
    logger.info("Testing basic patch processing...")

    # Create test image
    image = torch.rand(256, 256, device="cpu")

    # Initialize extractor and reconstructor
    extractor = PatchExtractor(patch_size=128, overlap=32, device="cpu")
    reconstructor = PatchReconstructor(blending_mode="linear", device="cpu")

    # Extract patches
    patches, patch_infos = extractor.extract_patches(image)
    logger.info(f"Extracted {len(patches)} patches")

    # Simple processing (identity)
    processed_patches = patches

    # Reconstruct
    reconstructed = reconstructor.reconstruct_image(
        processed_patches, patch_infos, 256, 256
    )

    # Check quality
    mse = torch.mean((reconstructed.squeeze() - image) ** 2).item()
    psnr = -10 * np.log10(mse) if mse > 0 else float("inf")

    logger.info(f"Reconstruction PSNR: {psnr:.2f} dB")

    # Should have high quality reconstruction
    assert psnr > 30, f"Poor reconstruction quality: {psnr:.2f} dB"

    return {"num_patches": len(patches), "reconstruction_psnr": psnr, "mse": mse}


def test_memory_efficient_processing():
    """Test memory-efficient patch processing."""
    logger.info("Testing memory-efficient processing...")

    # Create larger test image
    image = torch.rand(512, 512, device="cpu")

    # Initialize processor
    processor = MemoryEfficientPatchProcessor(
        patch_size=128, overlap=32, max_patches_in_memory=4, device="cpu"
    )

    # Simple processing function (add small noise)
    def add_noise(patch):
        return patch + torch.randn_like(patch) * 0.01

    # Process image
    result = processor.process_image(image, add_noise)

    # Check result
    assert (
        result.shape == image.shape
    ), f"Shape mismatch: {result.shape} vs {image.shape}"

    # Should be similar but not identical (due to added noise)
    mse = torch.mean((result - image) ** 2).item()
    assert 0.0001 < mse < 0.01, f"Unexpected MSE: {mse}"

    logger.info(f"Memory-efficient processing completed, MSE: {mse:.6f}")

    return {"input_shape": image.shape, "output_shape": result.shape, "mse": mse}


def test_calibration_integration():
    """Test integration with calibration system."""
    logger.info("Testing calibration integration...")

    # Create calibration parameters
    cal_params = CalibrationParams(
        gain=2.0,
        black_level=100,
        white_level=16383,
        read_noise=3.0,
        pixel_size=0.65,
        pixel_unit="um",
        domain="microscopy",
    )

    calibration = SensorCalibration(params=cal_params)

    # Create synthetic ADU data
    adu_data = np.random.randint(100, 4000, (256, 256), dtype=np.uint16)

    # Apply calibration
    electrons, mask = calibration.process_raw(adu_data, return_mask=True)

    # Convert to tensor
    electrons_tensor = torch.from_numpy(electrons).float()

    # Process with patches
    extractor = PatchExtractor(patch_size=128, overlap=32, device="cpu")
    reconstructor = PatchReconstructor(device="cpu")

    patches, patch_infos = extractor.extract_patches(electrons_tensor)

    # Simple processing (identity)
    processed_patches = patches

    # Reconstruct
    reconstructed = reconstructor.reconstruct_image(
        processed_patches, patch_infos, 256, 256
    )

    # Check quality
    mse = torch.mean((reconstructed.squeeze() - electrons_tensor) ** 2).item()
    psnr = -10 * np.log10(mse) if mse > 0 else float("inf")

    logger.info(f"Calibration integration PSNR: {psnr:.2f} dB")

    assert psnr > 30, f"Poor reconstruction with calibration: {psnr:.2f} dB"

    return {
        "calibration_applied": True,
        "reconstruction_psnr": psnr,
        "electrons_range": (electrons.min(), electrons.max()),
    }


def test_optimal_patch_sizing():
    """Test optimal patch size calculation."""
    logger.info("Testing optimal patch sizing...")

    # Test different scenarios
    scenarios = [
        (1024, 1024, 2.0),  # Small memory
        (2048, 2048, 8.0),  # Medium memory
        (4096, 4096, 16.0),  # Large memory
    ]

    results = []

    for height, width, memory_gb in scenarios:
        patch_h, patch_w = calculate_optimal_patch_size(
            height, width, memory_gb, safety_factor=0.8
        )

        logger.info(
            f"Image {height}x{width}, {memory_gb}GB -> patch {patch_h}x{patch_w}"
        )

        # Patches should be reasonable size
        assert 64 <= patch_h <= height, f"Invalid patch height: {patch_h}"
        assert 64 <= patch_w <= width, f"Invalid patch width: {patch_w}"

        results.append(
            {
                "image_size": (height, width),
                "memory_gb": memory_gb,
                "patch_size": (patch_h, patch_w),
            }
        )

    return results


def test_patch_dataset_config():
    """Test patch dataset configuration."""
    logger.info("Testing patch dataset configuration...")

    # Create configuration
    config = PatchDatasetConfig(
        patch_size=256,
        overlap=64,
        random_patches=True,
        patches_per_image=4,
        apply_calibration=True,
    )

    # Test serialization
    config_dict = config.to_dict()
    restored_config = PatchDatasetConfig.from_dict(config_dict)

    assert restored_config.patch_size == config.patch_size
    assert restored_config.overlap == config.overlap
    assert restored_config.random_patches == config.random_patches

    logger.info("Patch dataset configuration test passed")

    return {
        "config_created": True,
        "serialization_works": True,
        "patch_size": config.patch_size,
    }


def run_integration_tests():
    """Run all integration tests."""
    logger.info("Running Task 4.3 integration tests...")

    results = {}

    try:
        results["basic_processing"] = test_basic_patch_processing()
        logger.info("✅ Basic patch processing test passed")
    except Exception as e:
        logger.error(f"❌ Basic patch processing test failed: {e}")
        results["basic_processing"] = {"error": str(e)}

    try:
        results["memory_efficient"] = test_memory_efficient_processing()
        logger.info("✅ Memory-efficient processing test passed")
    except Exception as e:
        logger.error(f"❌ Memory-efficient processing test failed: {e}")
        results["memory_efficient"] = {"error": str(e)}

    try:
        results["calibration_integration"] = test_calibration_integration()
        logger.info("✅ Calibration integration test passed")
    except Exception as e:
        logger.error(f"❌ Calibration integration test failed: {e}")
        results["calibration_integration"] = {"error": str(e)}

    try:
        results["optimal_sizing"] = test_optimal_patch_sizing()
        logger.info("✅ Optimal patch sizing test passed")
    except Exception as e:
        logger.error(f"❌ Optimal patch sizing test failed: {e}")
        results["optimal_sizing"] = {"error": str(e)}

    try:
        results["dataset_config"] = test_patch_dataset_config()
        logger.info("✅ Dataset configuration test passed")
    except Exception as e:
        logger.error(f"❌ Dataset configuration test failed: {e}")
        results["dataset_config"] = {"error": str(e)}

    # Compute summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if "error" not in r)

    results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": passed_tests / total_tests,
        "overall_status": "PASSED" if passed_tests == total_tests else "PARTIAL",
    }

    return results


def main():
    """Main function."""
    setup_project_logging(level="INFO")

    logger.info("=" * 60)
    logger.info("TASK 4.3 INTEGRATION TEST")
    logger.info("Patch extraction for large images")
    logger.info("=" * 60)

    results = run_integration_tests()

    # Print summary
    summary = results["summary"]
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tests: {summary['total_tests']}")
    logger.info(f"Passed tests: {summary['passed_tests']}")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info(f"Overall status: {summary['overall_status']}")

    # Print detailed results
    for test_name, test_result in results.items():
        if test_name == "summary":
            continue

        if "error" in test_result:
            logger.info(f"❌ {test_name}: {test_result['error']}")
        else:
            logger.info(f"✅ {test_name}: Success")

    logger.info("=" * 60)

    # Exit with appropriate code
    sys.exit(0 if summary["overall_status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
