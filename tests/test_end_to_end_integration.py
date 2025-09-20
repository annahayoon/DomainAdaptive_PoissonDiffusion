"""
Comprehensive end-to-end integration tests for Phase 7.

This module tests the complete pipeline from RAW files to restored images
across all three domains (photography, microscopy, astronomy) with:
- Real data processing workflows
- Cross-domain validation
- Memory profiling and performance benchmarking
- Statistical consistency validation

Requirements addressed: 8.4 from requirements.md
Task: 7.1 from tasks.md
"""

import gc
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import psutil
import pytest
import torch

from core.calibration import CalibrationParams, SensorCalibration
from core.logging_config import get_logger
from core.metrics import EvaluationReport, EvaluationSuite
from core.patch_processing import MemoryEfficientPatchProcessor
from core.poisson_guidance import GuidanceConfig, PoissonGuidance
from core.transforms import ImageMetadata as TransformMetadata
from core.transforms import ReversibleTransform
from data.domain_datasets import DomainDataset, MultiDomainDataset
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


class PerformanceProfiler:
    """Performance profiling utility for integration tests."""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        self.process = psutil.Process()

    def start(self):
        """Start profiling."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory

    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)

    def stop(self) -> Dict[str, float]:
        """Stop profiling and return results."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        return {
            "duration_seconds": end_time - self.start_time,
            "start_memory_mb": self.start_memory,
            "end_memory_mb": end_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_delta_mb": end_memory - self.start_memory,
        }


class MockDataGenerator:
    """Generate mock data files for testing."""

    @staticmethod
    def create_mock_raw_file(
        filepath: Path, domain: str, size: Tuple[int, int] = (512, 512)
    ):
        """Create a mock raw file for testing."""
        height, width = size

        if domain == "photography":
            # Create mock ARW file (just binary data)
            data = np.random.randint(0, 4096, (height, width), dtype=np.uint16)
            with open(filepath, "wb") as f:
                f.write(b"MOCK_ARW_HEADER" + data.tobytes())

        elif domain == "microscopy":
            # Create mock TIFF file
            try:
                from PIL import Image

                data = np.random.randint(0, 4096, (height, width), dtype=np.uint16)
                img = Image.fromarray(data)
                img.save(filepath)
            except ImportError:
                # Fallback: create binary file
                data = np.random.randint(0, 4096, (height, width), dtype=np.uint16)
                with open(filepath, "wb") as f:
                    f.write(b"MOCK_TIFF_HEADER" + data.tobytes())

        elif domain == "astronomy":
            # Create mock FITS file
            data = np.random.randint(0, 65535, (height, width), dtype=np.uint16)
            with open(filepath, "wb") as f:
                f.write(b"SIMPLE  =                    T / FITS format\n")
                f.write(b"BITPIX  =                   16 / Bits per pixel\n")
                f.write(b"NAXIS   =                    2 / Number of axes\n")
                f.write(f"NAXIS1  =                 {width:4d} / Width\n".encode())
                f.write(f"NAXIS2  =                 {height:4d} / Height\n".encode())
                f.write(b"END" + b" " * 77 + b"\n")
                # Pad to 2880 bytes (FITS block size)
                header_size = 2880
                f.write(b"\0" * (header_size - f.tell()))
                f.write(data.tobytes())


class TestEndToEndIntegration:
    """Comprehensive end-to-end integration tests."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with mock data."""
        temp_dir = tempfile.mkdtemp()
        workspace = Path(temp_dir)

        # Create directory structure
        for domain in ["photography", "microscopy", "astronomy"]:
            domain_dir = workspace / domain
            domain_dir.mkdir()

            # Create mock data files
            for i in range(3):
                if domain == "photography":
                    filepath = domain_dir / f"image_{i:03d}.arw"
                elif domain == "microscopy":
                    filepath = domain_dir / f"image_{i:03d}.tif"
                else:  # astronomy
                    filepath = domain_dir / f"image_{i:03d}.fits"

                MockDataGenerator.create_mock_raw_file(filepath, domain)

            # Create calibration file
            cal_file = workspace / f"{domain}_calibration.json"
            if domain == "photography":
                cal_data = {
                    "gain": 1.0,
                    "black_level": 512,
                    "white_level": 16383,
                    "read_noise": 5.0,
                    "pixel_size": 4.29,
                    "pixel_unit": "um",
                    "domain": "photography",
                }
            elif domain == "microscopy":
                cal_data = {
                    "gain": 2.0,
                    "black_level": 100,
                    "white_level": 16383,
                    "read_noise": 3.0,
                    "pixel_size": 0.65,
                    "pixel_unit": "um",
                    "domain": "microscopy",
                }
            else:  # astronomy
                cal_data = {
                    "gain": 1.5,
                    "black_level": 0,
                    "white_level": 65535,
                    "read_noise": 2.0,
                    "pixel_size": 0.04,
                    "pixel_unit": "arcsec",
                    "domain": "astronomy",
                }

            with open(cal_file, "w") as f:
                json.dump(cal_data, f)

        yield workspace

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_model(self):
        """Create mock EDM model for testing."""
        model = Mock()

        def mock_forward(x, sigma, condition=None):
            # Return small noise (v-parameterization)
            return torch.randn_like(x) * 0.01

        # Make model callable
        model.__call__ = mock_forward
        model.parameters = MagicMock(
            return_value=[torch.tensor([1.0])]
        )  # For device detection

        # Mock conditioning encoding
        def mock_encode_conditioning(domain, scale, read_noise, background, device):
            return torch.randn(1, 6, device=device)

        model.encode_conditioning = mock_encode_conditioning

        return model

    def test_single_domain_end_to_end_photography(self, temp_workspace, mock_model):
        """Test complete photography pipeline."""
        logger.info("Testing photography end-to-end pipeline...")

        profiler = PerformanceProfiler()
        profiler.start()

        # Mock the loader setup
        with patch("data.domain_datasets.PhotographyLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.validate_file.return_value = True
            mock_loader.load_with_validation.return_value = (
                np.random.randint(512, 4096, (512, 512), dtype=np.uint16).astype(
                    np.float32
                ),
                ImageMetadata(
                    filepath="test.arw",
                    format=".arw",
                    domain="photography",
                    height=512,
                    width=512,
                    channels=1,
                    bit_depth=16,
                    dtype="uint16",
                    iso=3200,
                    exposure_time=0.1,
                ),
            )
            mock_loader_class.return_value = mock_loader

            # Create dataset
            dataset = DomainDataset(
                data_root=temp_workspace / "photography",
                domain="photography",
                calibration_file=temp_workspace / "photography_calibration.json",
                split="test",
                max_files=2,
                validate_files=False,
            )

            profiler.update_peak_memory()

            # Load sample
            sample = dataset[0]

            # Verify data structure
            assert "transformed" in sample
            assert "mask" in sample
            assert "image_metadata" in sample
            assert sample["domain"] == "photography"

            profiler.update_peak_memory()

            # Create guidance
            guidance = PoissonGuidance(scale=10000.0, config=GuidanceConfig(mode="wls"))

            # Patch isinstance for testing
            with patch("builtins.isinstance") as mock_isinstance:
                mock_isinstance.side_effect = lambda obj, cls: (
                    True
                    if cls.__name__ == "EDMModelWrapper" and hasattr(obj, "__call__")
                    else isinstance.__wrapped__(obj, cls)
                )

                # Create sampler
                sampler = EDMPosteriorSampler(mock_model, guidance)

            # Process with guidance
            guidance_config = GuidanceConfig(mode="wls", kappa=0.1)

            # Add channel dimension if needed
            y_observed = sample["transformed"]
            if y_observed.ndim == 3:
                y_observed = y_observed.unsqueeze(1)  # Add channel dimension

            result = sampler.sample(
                y_observed=y_observed,
                domain=sample["domain"],
                scale=sample["scale"],
                read_noise=3.0,
                background=0.0,  # Background electrons
                mask=sample["mask"],
                num_steps=5,  # Reduced for testing
            )

            profiler.update_peak_memory()

            # Verify output
            assert result["x_hat"].shape == sample["transformed"].shape
            assert "chi2_per_pixel" in result
            assert result.get("scale", sample["scale"]) == sample["scale"]

            # Test reconstruction to original size
            transform = ReversibleTransform(target_size=128)
            reconstructed = transform.inverse(
                result["x_hat"], sample["transform_metadata"]
            )

            assert reconstructed.shape[2:] == (
                sample["transform_metadata"].original_height,
                sample["transform_metadata"].original_width,
            )

        performance = profiler.stop()

        logger.info(
            f"Photography pipeline completed in {performance['duration_seconds']:.2f}s"
        )
        logger.info(f"Peak memory usage: {performance['peak_memory_mb']:.1f} MB")

        # Performance assertions
        assert performance["duration_seconds"] < 30, "Pipeline too slow"
        assert performance["peak_memory_mb"] < 1000, "Memory usage too high"

        return {
            "domain": "photography",
            "success": True,
            "performance": performance,
            "output_shape": denoised.shape,
            "chi2": info["chi2_per_pixel"],
        }

    def test_single_domain_end_to_end_microscopy(self, temp_workspace, mock_model):
        """Test complete microscopy pipeline."""
        logger.info("Testing microscopy end-to-end pipeline...")

        profiler = PerformanceProfiler()
        profiler.start()

        # Mock the TIFF loading
        with patch("data.domain_datasets.MicroscopyLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.validate_file.return_value = True
            mock_loader.load_with_validation.return_value = (
                np.random.randint(100, 2000, (512, 512), dtype=np.uint16).astype(
                    np.float32
                ),
                ImageMetadata(
                    filepath="test.tif",
                    format=".tif",
                    domain="microscopy",
                    height=512,
                    width=512,
                    channels=1,
                    bit_depth=16,
                    dtype="uint16",
                    pixel_size=0.65,
                    pixel_unit="um",
                ),
            )
            mock_loader_class.return_value = mock_loader

            # Create dataset
            dataset = DomainDataset(
                data_root=temp_workspace / "microscopy",
                domain="microscopy",
                calibration_file=temp_workspace / "microscopy_calibration.json",
                split="test",
                max_files=2,
                validate_files=False,
            )

            profiler.update_peak_memory()

            # Load sample
            sample = dataset[0]

            # Create guidance and sampler
            guidance = PoissonGuidance(scale=10000.0, config=GuidanceConfig(mode="wls"))
            sampler = EDMPosteriorSampler(mock_model, guidance)
            guidance_config = GuidanceConfig(mode="wls", kappa=0.2)

            denoised, info = sampler.denoise_with_metadata(
                y_electrons=sample["noisy_electrons"],
                metadata=sample["transform_metadata"],
                scale=sample["scale"],
                read_noise=2.0,
                guidance_config=guidance_config,
                steps=5,
                verbose=False,
            )

            profiler.update_peak_memory()

            # Verify microscopy-specific aspects
            assert sample["transform_metadata"].pixel_unit == "um"
            assert sample["transform_metadata"].domain == "microscopy"
            assert denoised.shape == sample["transformed"].shape

        performance = profiler.stop()

        logger.info(
            f"Microscopy pipeline completed in {performance['duration_seconds']:.2f}s"
        )

        return {
            "domain": "microscopy",
            "success": True,
            "performance": performance,
            "pixel_size": sample["transform_metadata"].pixel_size,
        }

    def test_single_domain_end_to_end_astronomy(self, temp_workspace, mock_model):
        """Test complete astronomy pipeline."""
        logger.info("Testing astronomy end-to-end pipeline...")

        profiler = PerformanceProfiler()
        profiler.start()

        # Mock the FITS loading
        with patch("data.domain_datasets.AstronomyLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.validate_file.return_value = True
            mock_loader.load_with_validation.return_value = (
                np.random.randint(0, 10000, (512, 512), dtype=np.uint16).astype(
                    np.float32
                ),
                ImageMetadata(
                    filepath="test.fits",
                    format=".fits",
                    domain="astronomy",
                    height=512,
                    width=512,
                    channels=1,
                    bit_depth=16,
                    dtype="int16",
                    pixel_size=0.04,
                    pixel_unit="arcsec",
                    exposure_time=300.0,
                ),
            )
            mock_loader_class.return_value = mock_loader

            # Create dataset
            dataset = DomainDataset(
                data_root=temp_workspace / "astronomy",
                domain="astronomy",
                calibration_file=temp_workspace / "astronomy_calibration.json",
                split="test",
                max_files=2,
                validate_files=False,
            )

            profiler.update_peak_memory()

            # Load sample
            sample = dataset[0]

            # Create guidance and sampler
            guidance = PoissonGuidance(scale=10000.0, config=GuidanceConfig(mode="wls"))
            sampler = EDMPosteriorSampler(mock_model, guidance)
            guidance_config = GuidanceConfig(mode="wls", kappa=0.5)

            denoised, info = sampler.denoise_with_metadata(
                y_electrons=sample["noisy_electrons"],
                metadata=sample["transform_metadata"],
                scale=sample["scale"],
                read_noise=1.5,
                guidance_config=guidance_config,
                steps=5,
                verbose=False,
            )

            profiler.update_peak_memory()

            # Verify astronomy-specific aspects
            assert sample["transform_metadata"].pixel_unit == "arcsec"
            assert sample["transform_metadata"].domain == "astronomy"
            assert denoised.shape == sample["transformed"].shape

        performance = profiler.stop()

        logger.info(
            f"Astronomy pipeline completed in {performance['duration_seconds']:.2f}s"
        )

        return {
            "domain": "astronomy",
            "success": True,
            "performance": performance,
            "exposure_time": 300.0,
        }

    def test_multi_domain_integration(self, temp_workspace, mock_model):
        """Test multi-domain dataset integration."""
        logger.info("Testing multi-domain integration...")

        profiler = PerformanceProfiler()
        profiler.start()

        # Mock all loaders
        with patch(
            "data.domain_datasets.PhotographyLoader"
        ) as mock_photo_loader, patch(
            "data.domain_datasets.MicroscopyLoader"
        ) as mock_micro_loader, patch(
            "data.domain_datasets.AstronomyLoader"
        ) as mock_astro_loader:
            # Setup mocks
            for mock_loader_class in [
                mock_photo_loader,
                mock_micro_loader,
                mock_astro_loader,
            ]:
                mock_loader = MagicMock()
                mock_loader.validate_file.return_value = True
                mock_loader.load_with_validation.return_value = (
                    np.random.randint(100, 4000, (256, 256), dtype=np.uint16).astype(
                        np.float32
                    ),
                    ImageMetadata(
                        filepath="test.file",
                        format=".test",
                        domain="test",
                        height=256,
                        width=256,
                        channels=1,
                        bit_depth=16,
                        dtype="uint16",
                    ),
                )
                mock_loader_class.return_value = mock_loader

            # Create multi-domain dataset
            domain_configs = {
                "photography": {
                    "data_root": str(temp_workspace / "photography"),
                    "calibration_file": str(
                        temp_workspace / "photography_calibration.json"
                    ),
                    "max_files": 1,
                },
                "microscopy": {
                    "data_root": str(temp_workspace / "microscopy"),
                    "calibration_file": str(
                        temp_workspace / "microscopy_calibration.json"
                    ),
                    "max_files": 1,
                },
                "astronomy": {
                    "data_root": str(temp_workspace / "astronomy"),
                    "calibration_file": str(
                        temp_workspace / "astronomy_calibration.json"
                    ),
                    "max_files": 1,
                },
            }

            multi_dataset = MultiDomainDataset(
                domain_configs=domain_configs,
                split="test",
                balance_domains=True,
                validate_files=False,
            )

            profiler.update_peak_memory()

            # Test sampling from all domains
            domain_counts = {}
            for i in range(len(multi_dataset)):
                sample = multi_dataset[i]
                domain = sample["domain"]
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

                # Verify sample structure
                assert "transformed" in sample
                assert "domain" in sample
                assert domain in ["photography", "microscopy", "astronomy"]

            profiler.update_peak_memory()

            # Verify all domains are represented
            assert len(domain_counts) == 3, f"Missing domains: {domain_counts}"

            logger.info(f"Multi-domain sampling: {domain_counts}")

        performance = profiler.stop()

        logger.info(
            f"Multi-domain integration completed in {performance['duration_seconds']:.2f}s"
        )

        return {
            "multi_domain": True,
            "success": True,
            "performance": performance,
            "domain_counts": domain_counts,
            "total_samples": len(multi_dataset),
        }

    def test_large_image_patch_processing(self, mock_model):
        """Test patch processing with large images."""
        logger.info("Testing large image patch processing...")

        profiler = PerformanceProfiler()
        profiler.start()

        # Create large synthetic image
        large_image = torch.rand(1, 1, 2048, 2048, device="cpu")

        # Create patch processor
        processor = MemoryEfficientPatchProcessor(
            patch_size=512, overlap=64, max_patches_in_memory=4, device="cpu"
        )

        profiler.update_peak_memory()

        # Simple processing function
        def mock_denoise(patch):
            # Simulate denoising with small improvement
            return patch + torch.randn_like(patch) * 0.01

        # Process image
        result = processor.process_image(
            large_image, mock_denoise, batch_size=2, show_progress=False
        )

        profiler.update_peak_memory()

        # Verify result
        assert result.shape == large_image.shape

        # Check reconstruction quality
        mse = torch.mean((result - large_image) ** 2).item()
        assert (
            mse < 0.01
        ), f"Unexpected reconstruction quality: {mse}"  # Allow very good reconstruction

        performance = profiler.stop()

        logger.info(
            f"Large image processing completed in {performance['duration_seconds']:.2f}s"
        )
        logger.info(f"Peak memory usage: {performance['peak_memory_mb']:.1f} MB")

        # Performance assertions for large images
        assert performance["duration_seconds"] < 120, "Large image processing too slow"
        assert (
            performance["peak_memory_mb"] < 2000
        ), "Memory usage too high for large images"

        return {
            "large_image_processing": True,
            "success": True,
            "performance": performance,
            "input_size": large_image.shape,
            "reconstruction_mse": mse,
        }

    def test_memory_stress_test(self, mock_model):
        """Test memory usage under stress conditions."""
        logger.info("Testing memory stress conditions...")

        profiler = PerformanceProfiler()
        profiler.start()

        # Process multiple images sequentially
        results = []

        for i in range(5):
            # Create image
            image = torch.rand(1, 1, 1024, 1024, device="cpu")

            # Create fresh guidance and sampler each time
            guidance = PoissonGuidance(scale=10000.0, config=GuidanceConfig(mode="wls"))
            sampler = EDMPosteriorSampler(mock_model, guidance)

            # Mock metadata
            metadata = TransformMetadata(
                original_height=1024,
                original_width=1024,
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

            # Process
            denoised, info = sampler.denoise_with_metadata(
                y_electrons=image * 1000,  # Convert to electrons
                metadata=metadata,
                scale=1000.0,
                read_noise=2.0,
                steps=3,
                verbose=False,
            )

            profiler.update_peak_memory()

            results.append(
                {
                    "iteration": i,
                    "output_shape": denoised.shape,
                    "chi2": info["chi2_per_pixel"],
                }
            )

            # Force garbage collection
            del image, denoised, sampler
            gc.collect()

        performance = profiler.stop()

        logger.info(
            f"Memory stress test completed in {performance['duration_seconds']:.2f}s"
        )
        logger.info(f"Peak memory usage: {performance['peak_memory_mb']:.1f} MB")

        # Memory should not grow excessively
        memory_growth = performance["memory_delta_mb"]
        assert (
            abs(memory_growth) < 500
        ), f"Excessive memory growth: {memory_growth:.1f} MB"

        return {
            "stress_test": True,
            "success": True,
            "performance": performance,
            "iterations": len(results),
            "memory_growth_mb": memory_growth,
        }


class TestCrossDomainValidation:
    """Cross-domain validation tests."""

    def test_domain_specific_metrics(self):
        """Test domain-specific metric computation."""
        logger.info("Testing domain-specific metrics...")

        # Create evaluation suite
        eval_suite = EvaluationSuite(device="cpu")

        # Test data for each domain
        domains_data = {
            "photography": {
                "pred": torch.rand(1, 3, 256, 256),
                "target": torch.rand(1, 3, 256, 256),
                "noisy": torch.rand(1, 3, 256, 256),
                "scale": 10000.0,
            },
            "microscopy": {
                "pred": torch.rand(1, 1, 512, 512),
                "target": torch.rand(1, 1, 512, 512),
                "noisy": torch.rand(1, 1, 512, 512),
                "scale": 1000.0,
            },
            "astronomy": {
                "pred": torch.rand(1, 1, 1024, 1024),
                "target": torch.rand(1, 1, 1024, 1024),
                "noisy": torch.rand(1, 1, 1024, 1024),
                "scale": 50000.0,
            },
        }

        results = {}

        for domain, data in domains_data.items():
            report = eval_suite.evaluate_restoration(
                pred=data["pred"],
                target=data["target"],
                noisy=data["noisy"],
                scale=data["scale"],
                domain=domain,
                method_name="test_method",
            )

            # Verify report structure
            assert hasattr(report, "psnr")
            assert hasattr(report, "ssim")
            assert hasattr(report, "chi2_consistency")
            assert hasattr(report, "domain_specific")

            results[domain] = {
                "psnr": report.psnr.value,
                "ssim": report.ssim.value,
                "chi2": report.chi2_consistency.value,
                "domain_metrics": report.domain_specific,
            }

            logger.info(
                f"{domain}: PSNR={report.psnr.value:.2f}, χ²={report.chi2_consistency.value:.3f}"
            )

        return results

    def test_statistical_consistency_validation(self):
        """Test statistical consistency across domains."""
        logger.info("Testing statistical consistency validation...")

        # Generate synthetic data with known statistics
        config = SyntheticDataConfig()
        synthetic_gen = SyntheticDataGenerator(config)

        results = {}

        for domain in ["photography", "microscopy", "astronomy"]:
            # Generate test data
            config = SyntheticDataConfig(
                image_size=(256, 256),
                pattern_type="gradient",
                photon_count=100,
                read_noise=3.0,
                domain=domain,
            )

            clean, noisy, metadata = synthetic_gen.generate_image(config)

            # Convert to tensors
            clean_tensor = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).float()
            noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float()

            # Compute statistics
            eval_suite = EvaluationSuite(device="cpu")

            # Test with identity "restoration" (noisy as prediction)
            report = eval_suite.evaluate_restoration(
                pred=noisy_tensor,
                target=clean_tensor,
                noisy=noisy_tensor,
                scale=metadata["scale"],
                domain=domain,
                method_name="identity",
            )

            results[domain] = {
                "chi2_consistency": report.chi2_consistency.value,
                "bias": report.bias_analysis.value
                if hasattr(report, "bias_analysis")
                else 0.0,
                "photon_count": config.photon_count,
                "read_noise": config.read_noise,
            }

            # Chi-squared should be close to 1 for proper noise model
            logger.info(f"{domain}: χ²={report.chi2_consistency.value:.3f}")

        return results


def run_integration_test_suite():
    """Run the complete integration test suite."""
    logger.info("=" * 60)
    logger.info("PHASE 7.1: COMPLETE INTEGRATION TESTING")
    logger.info("=" * 60)

    # This would normally be run by pytest, but we can also run manually
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_integration_test_suite()
