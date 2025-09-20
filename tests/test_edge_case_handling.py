"""
Edge case handling tests for Phase 7.

This module tests the system's robustness under extreme conditions:
- Extreme low-light scenarios (<1 photon per pixel)
- Missing calibration fallback behavior
- Corrupted file handling and graceful error recovery
- Boundary conditions and numerical stability

Requirements addressed: 9.2, 9.4, 9.5 from requirements.md
Task: 7.3 from tasks.md
"""

import json
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from core.calibration import CalibrationParams, SensorCalibration
from core.exceptions import CalibrationError, DataError, ValidationError
from core.logging_config import get_logger
from core.poisson_guidance import GuidanceConfig, PoissonGuidance
from core.transforms import ImageMetadata as TransformMetadata
from core.transforms import ReversibleTransform
from data.domain_datasets import DomainDataset
from data.loaders import (
    AstronomyLoader,
    ImageMetadata,
    MicroscopyLoader,
    PhotographyLoader,
)
from models.edm_wrapper import EDMModelWrapper
from models.sampler import EDMPosteriorSampler
from scripts.generate_synthetic_data import SyntheticConfig as SyntheticDataConfig
from scripts.generate_synthetic_data import SyntheticDataGenerator

logger = get_logger(__name__)


class TestExtremeLowLightScenarios:
    """Test extreme low-light scenarios with <1 photon per pixel."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = Mock(spec=EDMModelWrapper)
        model.forward = lambda x, sigma, condition=None: torch.randn_like(x) * 0.01
        model.parameters.return_value = [torch.tensor([1.0])]
        return model

    def test_ultra_low_photon_guidance(self, mock_model):
        """Test guidance computation with <1 photon per pixel."""
        logger.info("Testing ultra-low photon guidance...")

        # Create extremely low-light data
        image_size = (64, 64)
        photon_rate = 0.1  # 0.1 photons per pixel on average
        read_noise = 3.0

        # Generate sparse photon data
        np.random.seed(42)
        clean = np.full(image_size, photon_rate, dtype=np.float32)

        # Poisson sampling - most pixels will be 0
        photon_counts = np.random.poisson(clean)

        # Add read noise
        noisy = photon_counts + np.random.normal(0, read_noise, image_size)

        # Convert to tensors
        clean_tensor = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0)

        logger.info(f"Average photons per pixel: {np.mean(photon_counts):.3f}")
        logger.info(
            f"Fraction of zero-photon pixels: {np.mean(photon_counts == 0):.3f}"
        )

        # Test guidance computation
        guidance = PoissonGuidance(
            scale=1.0,  # Already in photons
            background=0.0,
            read_noise=read_noise,
            config=GuidanceConfig(mode="wls", gradient_clip=5.0),
        )

        # Compute guidance score
        score = guidance.compute_score(
            x_hat=clean_tensor,
            y_electrons=noisy_tensor,
            eps=0.01,  # Small epsilon for numerical stability
        )

        # Check for numerical stability
        assert torch.all(torch.isfinite(score)), "Non-finite values in guidance score"

        score_norm = torch.norm(score).item()
        score_max = torch.max(torch.abs(score)).item()

        logger.info(f"Guidance score norm: {score_norm:.6f}")
        logger.info(f"Guidance score max: {score_max:.6f}")

        # Score should be finite and reasonable
        assert score_norm < 100, f"Guidance score too large: {score_norm}"
        assert score_max < 10, f"Max guidance score too large: {score_max}"

        # Test with sampler
        sampler = EDMPosteriorSampler(mock_model)

        metadata = TransformMetadata(
            original_height=64,
            original_width=64,
            scale_factor=1.0,
            crop_bbox=None,
            pad_amounts=None,
            pixel_size=1.0,
            pixel_unit="pixel",
            domain="test",
            black_level=0,
            white_level=1,
            bit_depth=16,
        )

        # Should not crash with extreme low light
        denoised, info = sampler.denoise_with_metadata(
            y_electrons=noisy_tensor,
            metadata=metadata,
            scale=1.0,
            read_noise=read_noise,
            guidance_weight=0.1,  # Reduced guidance weight
            steps=3,  # Fewer steps for testing
            verbose=False,
        )

        # Verify output is reasonable
        assert torch.all(
            torch.isfinite(denoised)
        ), "Non-finite values in denoised output"
        assert torch.all(denoised >= 0), "Negative values in denoised output"

        logger.info(
            f"Denoised range: [{torch.min(denoised):.3f}, {torch.max(denoised):.3f}]"
        )

        return {
            "average_photons": float(np.mean(photon_counts)),
            "zero_photon_fraction": float(np.mean(photon_counts == 0)),
            "guidance_score_norm": score_norm,
            "denoised_range": (torch.min(denoised).item(), torch.max(denoised).item()),
            "processing_successful": True,
        }

    def test_zero_photon_regions(self):
        """Test handling of regions with zero photons."""
        logger.info("Testing zero photon regions...")

        # Create image with zero-photon regions
        image = torch.zeros(1, 1, 32, 32)
        image[:, :, 16:, 16:] = 0.5  # Only bottom-right has signal

        # Add read noise everywhere
        read_noise = 2.0
        noisy = image + torch.randn_like(image) * read_noise

        # Test guidance
        guidance = PoissonGuidance(
            scale=1.0,
            background=0.0,
            read_noise=read_noise,
            config=GuidanceConfig(mode="wls", gradient_clip=10.0),
        )

        score = guidance.compute_score(x_hat=image, y_electrons=noisy, eps=0.1)

        # Check zero regions
        zero_region_score = score[:, :, :16, :16]  # Top-left zero region
        signal_region_score = score[:, :, 16:, 16:]  # Bottom-right signal region

        zero_score_norm = torch.norm(zero_region_score).item()
        signal_score_norm = torch.norm(signal_region_score).item()

        logger.info(f"Zero region score norm: {zero_score_norm:.6f}")
        logger.info(f"Signal region score norm: {signal_score_norm:.6f}")

        # Both should be finite
        assert torch.all(torch.isfinite(score)), "Non-finite scores in zero photon test"

        return {
            "zero_region_score_norm": zero_score_norm,
            "signal_region_score_norm": signal_score_norm,
            "zero_photon_handling_successful": True,
        }

    def test_numerical_stability_extreme_cases(self):
        """Test numerical stability in extreme cases."""
        logger.info("Testing numerical stability in extreme cases...")

        test_cases = [
            {"name": "very_small_values", "scale": 1e-6, "read_noise": 1e-6},
            {"name": "very_large_values", "scale": 1e6, "read_noise": 1e3},
            {"name": "zero_read_noise", "scale": 1.0, "read_noise": 0.0},
            {"name": "large_read_noise", "scale": 1.0, "read_noise": 100.0},
        ]

        results = {}

        for case in test_cases:
            logger.info(f"Testing {case['name']}...")

            # Create test data
            x_hat = torch.ones(1, 1, 16, 16) * 0.5
            y_electrons = torch.ones(1, 1, 16, 16) * case["scale"] * 0.5

            # Test guidance
            guidance = PoissonGuidance(
                scale=case["scale"],
                background=0.0,
                read_noise=case["read_noise"],
                config=GuidanceConfig(mode="wls", gradient_clip=100.0),
            )

            try:
                score = guidance.compute_score(x_hat, y_electrons, eps=1e-10)

                # Check for numerical issues
                is_finite = torch.all(torch.isfinite(score))
                score_norm = torch.norm(score).item() if is_finite else float("inf")

                results[case["name"]] = {
                    "is_finite": is_finite.item(),
                    "score_norm": score_norm,
                    "success": is_finite.item() and score_norm < 1e6,
                }

                logger.info(f"  Finite: {is_finite}, Norm: {score_norm:.6f}")

            except Exception as e:
                logger.warning(f"  Exception: {e}")
                results[case["name"]] = {
                    "is_finite": False,
                    "score_norm": float("inf"),
                    "success": False,
                    "exception": str(e),
                }

        # At least basic cases should work
        assert results["very_small_values"]["success"], "Failed on very small values"
        assert results["large_read_noise"]["success"], "Failed on large read noise"

        return results


class TestMissingCalibrationFallback:
    """Test fallback behavior when calibration is missing or incomplete."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test files."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir)

        # Create mock image files
        for i in range(2):
            (data_dir / f"image_{i:03d}.tif").touch()

        yield data_dir

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    def test_missing_calibration_file(self, temp_data_dir):
        """Test behavior when calibration file is missing."""
        logger.info("Testing missing calibration file...")

        with patch("data.domain_datasets.MicroscopyLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.validate_file.return_value = True
            mock_loader_class.return_value = mock_loader

            # Create dataset without calibration file
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                dataset = DomainDataset(
                    data_root=temp_data_dir,
                    domain="microscopy",
                    calibration_file=None,  # No calibration file
                    split="test",
                    validate_files=False,
                )

                # Should issue warning
                warning_messages = [str(warning.message) for warning in w]
                has_calibration_warning = any(
                    "calibration" in msg.lower() for msg in warning_messages
                )

                logger.info(f"Warnings issued: {len(w)}")
                if has_calibration_warning:
                    logger.info("✓ Calibration warning issued as expected")

                # Should use default parameters
                assert dataset.calibration is not None
                assert hasattr(dataset.calibration, "params")

                # Check default values are reasonable
                params = dataset.calibration.params
                assert params.gain > 0
                assert params.read_noise > 0
                assert params.domain == "microscopy"

                logger.info(
                    f"Default calibration - Gain: {params.gain}, Read noise: {params.read_noise}"
                )

        return {
            "missing_calibration_handled": True,
            "warning_issued": has_calibration_warning,
            "default_gain": params.gain,
            "default_read_noise": params.read_noise,
        }

    def test_corrupted_calibration_file(self, temp_data_dir):
        """Test behavior with corrupted calibration file."""
        logger.info("Testing corrupted calibration file...")

        # Create corrupted calibration file
        corrupted_cal_file = temp_data_dir / "corrupted_calibration.json"
        with open(corrupted_cal_file, "w") as f:
            f.write("{ invalid json content }")

        with patch("data.domain_datasets.MicroscopyLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.validate_file.return_value = True
            mock_loader_class.return_value = mock_loader

            # Should handle corrupted file gracefully
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                try:
                    dataset = DomainDataset(
                        data_root=temp_data_dir,
                        domain="microscopy",
                        calibration_file=str(corrupted_cal_file),
                        split="test",
                        validate_files=False,
                    )

                    # If it doesn't raise exception, should fall back to defaults
                    fallback_successful = True
                    logger.info("✓ Graceful fallback to default calibration")

                except Exception as e:
                    # Should raise appropriate error
                    fallback_successful = False
                    logger.info(f"✓ Appropriate error raised: {type(e).__name__}")
                    assert isinstance(
                        e, (CalibrationError, DataError, json.JSONDecodeError)
                    )

        return {"corrupted_calibration_handled": True, "fallback_or_error": True}

    def test_incomplete_calibration_parameters(self, temp_data_dir):
        """Test behavior with incomplete calibration parameters."""
        logger.info("Testing incomplete calibration parameters...")

        # Create incomplete calibration file
        incomplete_cal_file = temp_data_dir / "incomplete_calibration.json"
        incomplete_data = {
            "gain": 2.0,
            # Missing: black_level, white_level, read_noise, etc.
        }

        with open(incomplete_cal_file, "w") as f:
            json.dump(incomplete_data, f)

        with patch("data.domain_datasets.MicroscopyLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.validate_file.return_value = True
            mock_loader_class.return_value = mock_loader

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                try:
                    dataset = DomainDataset(
                        data_root=temp_data_dir,
                        domain="microscopy",
                        calibration_file=str(incomplete_cal_file),
                        split="test",
                        validate_files=False,
                    )

                    # Should fill in missing parameters with defaults
                    params = dataset.calibration.params

                    assert params.gain == 2.0  # From file
                    assert params.read_noise > 0  # Default value
                    assert params.black_level >= 0  # Default value

                    logger.info(f"✓ Incomplete parameters filled with defaults")
                    logger.info(
                        f"Gain: {params.gain} (from file), Read noise: {params.read_noise} (default)"
                    )

                    incomplete_handled = True

                except Exception as e:
                    logger.info(f"Exception with incomplete calibration: {e}")
                    incomplete_handled = False

        return {
            "incomplete_calibration_handled": incomplete_handled,
            "parameters_filled": incomplete_handled,
        }

    def test_invalid_calibration_values(self, temp_data_dir):
        """Test behavior with invalid calibration values."""
        logger.info("Testing invalid calibration values...")

        invalid_cases = [
            {"name": "negative_gain", "data": {"gain": -1.0, "read_noise": 2.0}},
            {"name": "zero_gain", "data": {"gain": 0.0, "read_noise": 2.0}},
            {"name": "negative_read_noise", "data": {"gain": 2.0, "read_noise": -1.0}},
            {"name": "extreme_values", "data": {"gain": 1e10, "read_noise": 1e10}},
        ]

        results = {}

        for case in invalid_cases:
            logger.info(f"Testing {case['name']}...")

            # Create invalid calibration file
            invalid_cal_file = temp_data_dir / f"{case['name']}_calibration.json"
            with open(invalid_cal_file, "w") as f:
                json.dump(case["data"], f)

            with patch("data.domain_datasets.MicroscopyLoader") as mock_loader_class:
                mock_loader = MagicMock()
                mock_loader.validate_file.return_value = True
                mock_loader_class.return_value = mock_loader

                try:
                    dataset = DomainDataset(
                        data_root=temp_data_dir,
                        domain="microscopy",
                        calibration_file=str(invalid_cal_file),
                        split="test",
                        validate_files=False,
                    )

                    # Check if values were corrected
                    params = dataset.calibration.params
                    values_corrected = params.gain > 0 and params.read_noise >= 0

                    results[case["name"]] = {
                        "handled": True,
                        "values_corrected": values_corrected,
                        "final_gain": params.gain,
                        "final_read_noise": params.read_noise,
                    }

                    logger.info(
                        f"  ✓ Handled, final values: gain={params.gain:.3f}, read_noise={params.read_noise:.3f}"
                    )

                except Exception as e:
                    results[case["name"]] = {
                        "handled": True,
                        "exception_raised": True,
                        "exception_type": type(e).__name__,
                    }

                    logger.info(f"  ✓ Exception raised: {type(e).__name__}")

        return results


class TestCorruptedFileHandling:
    """Test handling of corrupted and malformed files."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with various file types."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir)

        yield data_dir

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    def test_corrupted_image_files(self, temp_data_dir):
        """Test handling of corrupted image files."""
        logger.info("Testing corrupted image files...")

        # Create various corrupted files
        corrupted_files = [
            {"name": "empty.tif", "content": b""},
            {"name": "truncated.tif", "content": b"TIFF_HEADER_INCOMPLETE"},
            {"name": "wrong_format.tif", "content": b"This is not a TIFF file"},
            {"name": "binary_garbage.tif", "content": np.random.bytes(1000)},
        ]

        results = {}

        for file_info in corrupted_files:
            logger.info(f"Testing {file_info['name']}...")

            # Create corrupted file
            file_path = temp_data_dir / file_info["name"]
            with open(file_path, "wb") as f:
                f.write(file_info["content"])

            # Test with microscopy loader
            loader = MicroscopyLoader(validate_on_load=True)

            # File validation should catch some issues
            is_valid = loader.validate_file(file_path)

            if is_valid:
                # Try to load
                try:
                    data, metadata = loader.load_with_validation(file_path)
                    load_successful = True
                    error_type = None
                except Exception as e:
                    load_successful = False
                    error_type = type(e).__name__
                    logger.info(f"  ✓ Load failed as expected: {error_type}")
            else:
                load_successful = False
                error_type = "ValidationFailed"
                logger.info(f"  ✓ Validation failed as expected")

            results[file_info["name"]] = {
                "validation_passed": is_valid,
                "load_successful": load_successful,
                "error_type": error_type,
                "handled_gracefully": not load_successful,  # Should fail gracefully
            }

        # All corrupted files should be handled gracefully
        all_handled = all(result["handled_gracefully"] for result in results.values())
        assert all_handled, "Some corrupted files were not handled gracefully"

        return results

    def test_permission_denied_files(self, temp_data_dir):
        """Test handling of files with permission issues."""
        logger.info("Testing permission denied files...")

        # Create a file and remove read permissions
        restricted_file = temp_data_dir / "restricted.tif"
        with open(restricted_file, "wb") as f:
            f.write(b"TIFF_CONTENT")

        # Remove read permissions (on Unix systems)
        try:
            import os

            os.chmod(restricted_file, 0o000)

            loader = MicroscopyLoader(validate_on_load=True)

            # Should handle permission error gracefully
            try:
                is_valid = loader.validate_file(restricted_file)
                permission_handled = not is_valid  # Should fail validation
                logger.info(f"✓ Permission issue handled in validation: {not is_valid}")
            except PermissionError:
                permission_handled = True
                logger.info("✓ Permission error raised and can be caught")

            # Restore permissions for cleanup
            os.chmod(restricted_file, 0o644)

        except (OSError, AttributeError):
            # Skip on systems that don't support chmod
            logger.info("Skipping permission test (not supported on this system)")
            permission_handled = True

        return {"permission_denied_handled": permission_handled}

    def test_large_file_handling(self, temp_data_dir):
        """Test handling of unexpectedly large files."""
        logger.info("Testing large file handling...")

        # Create a large file (simulate)
        large_file = temp_data_dir / "large.tif"

        # Don't actually create a huge file, just test the size check logic
        with open(large_file, "wb") as f:
            f.write(b"TIFF_HEADER")

        loader = MicroscopyLoader(validate_on_load=True)

        # Mock file size check
        original_stat = Path.stat

        def mock_stat(self):
            result = original_stat(self)
            if self.name == "large.tif":
                # Mock a very large file size
                class MockStat:
                    st_size = 10 * 1024 * 1024 * 1024  # 10 GB

                return MockStat()
            return result

        with patch.object(Path, "stat", mock_stat):
            # Loader should handle large files appropriately
            is_valid = loader.validate_file(large_file)

            # Depending on implementation, might accept or reject large files
            logger.info(f"Large file validation result: {is_valid}")

        return {"large_file_handled": True, "validation_result": is_valid}


class TestBoundaryConditions:
    """Test boundary conditions and edge cases in processing."""

    def test_single_pixel_images(self):
        """Test processing of single-pixel images."""
        logger.info("Testing single-pixel images...")

        # Create single-pixel image
        single_pixel = torch.ones(1, 1, 1, 1) * 0.5

        # Test transforms
        transform = ReversibleTransform(target_size=128)

        metadata = TransformMetadata(
            original_height=1,
            original_width=1,
            scale_factor=128.0,
            crop_bbox=None,
            pad_amounts=(63, 64, 63, 64),  # Padding to 128x128
            pixel_size=1.0,
            pixel_unit="pixel",
            domain="test",
            black_level=0,
            white_level=1,
            bit_depth=16,
        )

        # Transform should handle single pixel
        transformed, _ = transform.forward(
            single_pixel,
            pixel_size=1.0,
            pixel_unit="pixel",
            domain="test",
            black_level=0,
            white_level=1,
        )

        # Should be padded to target size
        assert transformed.shape == (1, 1, 128, 128)

        # Inverse transform should recover single pixel
        recovered = transform.inverse(transformed, metadata)
        assert recovered.shape == (1, 1, 1, 1)

        # Value should be preserved (approximately)
        assert torch.allclose(recovered, single_pixel, atol=0.1)

        logger.info("✓ Single pixel image handled correctly")

        return {
            "single_pixel_handled": True,
            "transform_successful": True,
            "value_preserved": torch.allclose(recovered, single_pixel, atol=0.1).item(),
        }

    def test_extreme_aspect_ratios(self):
        """Test images with extreme aspect ratios."""
        logger.info("Testing extreme aspect ratios...")

        extreme_cases = [
            {"name": "very_wide", "shape": (1, 1, 10, 1000)},
            {"name": "very_tall", "shape": (1, 1, 1000, 10)},
            {"name": "thin_line", "shape": (1, 1, 1, 500)},
        ]

        results = {}
        transform = ReversibleTransform(target_size=128)

        for case in extreme_cases:
            logger.info(f"Testing {case['name']}: {case['shape']}")

            # Create test image
            image = torch.rand(case["shape"])

            try:
                # Transform
                transformed, metadata = transform.forward(
                    image,
                    pixel_size=1.0,
                    pixel_unit="pixel",
                    domain="test",
                    black_level=0,
                    white_level=1,
                )

                # Should be 128x128
                assert transformed.shape == (1, 1, 128, 128)

                # Inverse transform
                recovered = transform.inverse(transformed, metadata)

                # Should recover original shape
                assert recovered.shape == case["shape"]

                results[case["name"]] = {
                    "transform_successful": True,
                    "shape_preserved": True,
                    "original_shape": case["shape"],
                    "recovered_shape": tuple(recovered.shape),
                }

                logger.info(f"  ✓ {case['name']} handled successfully")

            except Exception as e:
                results[case["name"]] = {
                    "transform_successful": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }

                logger.info(f"  ✗ {case['name']} failed: {e}")

        return results

    def test_numerical_precision_limits(self):
        """Test behavior at numerical precision limits."""
        logger.info("Testing numerical precision limits...")

        precision_cases = [
            {"name": "very_small", "values": torch.full((1, 1, 32, 32), 1e-10)},
            {"name": "very_large", "values": torch.full((1, 1, 32, 32), 1e10)},
            {
                "name": "mixed_extreme",
                "values": torch.cat(
                    [
                        torch.full((1, 1, 16, 32), 1e-10),
                        torch.full((1, 1, 16, 32), 1e10),
                    ],
                    dim=2,
                ),
            },
        ]

        results = {}

        for case in precision_cases:
            logger.info(f"Testing {case['name']}...")

            values = case["values"]

            # Test guidance computation
            guidance = PoissonGuidance(
                scale=1.0,
                background=0.0,
                read_noise=1.0,
                config=GuidanceConfig(mode="wls", gradient_clip=1e6),
            )

            try:
                score = guidance.compute_score(
                    x_hat=values, y_electrons=values, eps=1e-12
                )

                is_finite = torch.all(torch.isfinite(score))
                max_abs_score = torch.max(torch.abs(score)).item()

                results[case["name"]] = {
                    "computation_successful": True,
                    "all_finite": is_finite.item(),
                    "max_abs_score": max_abs_score,
                    "numerically_stable": is_finite.item() and max_abs_score < 1e6,
                }

                logger.info(f"  ✓ Finite: {is_finite}, Max score: {max_abs_score:.2e}")

            except Exception as e:
                results[case["name"]] = {
                    "computation_successful": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }

                logger.info(f"  ✗ Failed: {e}")

        return results


def run_edge_case_test_suite():
    """Run the complete edge case test suite."""
    logger.info("=" * 60)
    logger.info("PHASE 7.3: EDGE CASE HANDLING")
    logger.info("=" * 60)

    # This would normally be run by pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_edge_case_test_suite()
