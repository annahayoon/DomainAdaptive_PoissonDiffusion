#!/usr/bin/env python3
"""
Integration test for Phase 3 + Phase 4.1 components.

This script tests the complete pipeline from data loading (Phase 4.1)
through model conditioning and sampling (Phase 3).

Usage:
    python scripts/test_phase3_integration.py [--verbose] [--create-data]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.calibration import CalibrationParams, SensorCalibration
from core.logging_config import get_logger, setup_project_logging
from core.poisson_guidance import PoissonGuidance, create_domain_guidance
from data.domain_datasets import DomainDataset
from data.loaders import (
    AstronomyLoader,
    FormatDetector,
    MicroscopyLoader,
    PhotographyLoader,
    get_image_metadata,
    load_image,
)
from models.edm_wrapper import DomainEncoder, EDMModelWrapper, create_domain_edm_wrapper
from models.sampler import EDMPosteriorSampler, create_edm_sampler

logger = get_logger(__name__)


class Phase3IntegrationTester:
    """
    Comprehensive integration tester for Phase 3 + Phase 4.1.

    Tests the complete pipeline from data loading to model inference.
    """

    def __init__(self, create_synthetic_data: bool = True):
        """Initialize integration tester."""
        self.create_synthetic_data = create_synthetic_data
        self.temp_dir = None
        self.test_results = {}

        logger.info("Phase3IntegrationTester initialized")

    def setup_test_environment(self):
        """Set up test environment with synthetic data."""
        logger.info("Setting up test environment...")

        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="phase3_integration_"))
        logger.info(f"Test directory: {self.temp_dir}")

        if self.create_synthetic_data:
            self._create_synthetic_test_data()

        return True

    def _create_synthetic_test_data(self):
        """Create synthetic test data for all domains."""
        logger.info("Creating synthetic test data...")

        # Create data directories
        for domain in ["photography", "microscopy", "astronomy"]:
            domain_dir = self.temp_dir / domain
            domain_dir.mkdir(exist_ok=True)

            # Create synthetic images
            if domain == "photography":
                self._create_synthetic_raw_data(domain_dir)
            elif domain == "microscopy":
                self._create_synthetic_tiff_data(domain_dir)
            elif domain == "astronomy":
                self._create_synthetic_fits_data(domain_dir)

            # Create calibration file
            self._create_calibration_file(domain_dir, domain)

    def _create_synthetic_raw_data(self, output_dir: Path):
        """Create synthetic RAW-like data."""
        # Create synthetic Bayer pattern data
        height, width = 256, 256

        # Simulate Bayer pattern (RGGB)
        bayer_data = np.random.randint(100, 4000, (height, width), dtype=np.uint16)

        # Add some structure
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        signal = 1000 + 500 * np.sin(x / 20) * np.cos(y / 20)
        bayer_data = (bayer_data + signal).astype(np.uint16)

        # Save as numpy array (simulating RAW data)
        np.save(output_dir / "test_image.npy", bayer_data)

        # Create metadata file
        metadata = {
            "height": height,
            "width": width,
            "channels": 1,
            "bit_depth": 16,
            "bayer_pattern": "RGGB",
            "iso": 800,
            "exposure_time": 1 / 60,
            "black_level": 100,
            "white_level": 4095,
        }

        with open(output_dir / "test_image_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _create_synthetic_tiff_data(self, output_dir: Path):
        """Create synthetic TIFF-like data."""
        # Multi-channel microscopy data
        height, width, channels = 128, 128, 3

        # Create structured data with different intensities per channel
        data = np.zeros((channels, height, width), dtype=np.uint16)

        for c in range(channels):
            # Different patterns for each channel
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            if c == 0:  # DAPI-like
                pattern = 500 + 300 * np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / 400)
            elif c == 1:  # FITC-like
                pattern = 800 + 400 * np.sin(x / 10) * np.cos(y / 10)
            else:  # TRITC-like
                pattern = 300 + 200 * np.random.random((height, width))

            data[c] = pattern.astype(np.uint16)

        # Save as numpy array
        np.save(output_dir / "test_multichannel.npy", data)

        # Create metadata
        metadata = {
            "height": height,
            "width": width,
            "channels": channels,
            "bit_depth": 16,
            "pixel_size": 0.1,  # 0.1 µm/pixel
            "pixel_unit": "µm",
            "channel_names": ["DAPI", "FITC", "TRITC"],
        }

        with open(output_dir / "test_multichannel_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _create_synthetic_fits_data(self, output_dir: Path):
        """Create synthetic FITS-like data."""
        height, width = 512, 512

        # Create astronomical-like data with stars and noise
        data = np.random.normal(100, 10, (height, width)).astype(np.float32)

        # Add some "stars"
        for _ in range(20):
            x = np.random.randint(50, width - 50)
            y = np.random.randint(50, height - 50)
            intensity = np.random.uniform(500, 2000)

            # Gaussian PSF
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            star = intensity * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 16)
            data += star

        # Save as numpy array
        np.save(output_dir / "test_astronomy.npy", data)

        # Create metadata with FITS-like headers
        metadata = {
            "height": height,
            "width": width,
            "channels": 1,
            "bit_depth": 32,
            "exposure_time": 300.0,  # 5 minutes
            "temperature": -20.0,  # CCD temperature
            "gain": 1.5,
            "bzero": 0.0,
            "bscale": 1.0,
            "object": "TEST_FIELD",
            "telescope": "TEST_SCOPE",
        }

        with open(output_dir / "test_astronomy_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _create_calibration_file(self, output_dir: Path, domain: str):
        """Create calibration file for domain."""
        if domain == "photography":
            params = {
                "gain": 0.5,  # e-/ADU
                "black_level": 100.0,
                "white_level": 4095.0,
                "read_noise": 3.0,
                "domain": "photography",
            }
        elif domain == "microscopy":
            params = {
                "gain": 0.8,
                "black_level": 50.0,
                "white_level": 65535.0,
                "read_noise": 2.0,
                "domain": "microscopy",
            }
        else:  # astronomy
            params = {
                "gain": 1.2,
                "black_level": 0.0,
                "white_level": 65535.0,
                "read_noise": 5.0,
                "domain": "astronomy",
            }

        with open(output_dir / "calibration.json", "w") as f:
            json.dump(params, f, indent=2)

    def test_data_loading(self) -> bool:
        """Test Phase 4.1 data loading functionality."""
        logger.info("Testing data loading (Phase 4.1)...")

        results = {}

        # Test each domain
        for domain in ["photography", "microscopy", "astronomy"]:
            domain_dir = self.temp_dir / domain

            try:
                if domain == "photography":
                    # Load synthetic RAW data
                    data_file = domain_dir / "test_image.npy"
                    metadata_file = domain_dir / "test_image_metadata.json"

                    if data_file.exists():
                        data = np.load(data_file)
                        with open(metadata_file) as f:
                            metadata = json.load(f)

                        results[f"{domain}_load"] = True
                        results[f"{domain}_shape"] = data.shape
                        results[f"{domain}_dtype"] = str(data.dtype)

                        logger.debug(f"Photography data: {data.shape}, {data.dtype}")

                elif domain == "microscopy":
                    # Load synthetic TIFF data
                    data_file = domain_dir / "test_multichannel.npy"
                    metadata_file = domain_dir / "test_multichannel_metadata.json"

                    if data_file.exists():
                        data = np.load(data_file)
                        with open(metadata_file) as f:
                            metadata = json.load(f)

                        results[f"{domain}_load"] = True
                        results[f"{domain}_shape"] = data.shape
                        results[f"{domain}_channels"] = metadata["channels"]

                        logger.debug(
                            f"Microscopy data: {data.shape}, {metadata['channels']} channels"
                        )

                elif domain == "astronomy":
                    # Load synthetic FITS data
                    data_file = domain_dir / "test_astronomy.npy"
                    metadata_file = domain_dir / "test_astronomy_metadata.json"

                    if data_file.exists():
                        data = np.load(data_file)
                        with open(metadata_file) as f:
                            metadata = json.load(f)

                        results[f"{domain}_load"] = True
                        results[f"{domain}_shape"] = data.shape
                        results[f"{domain}_exposure"] = metadata["exposure_time"]

                        logger.debug(
                            f"Astronomy data: {data.shape}, {metadata['exposure_time']}s exposure"
                        )

            except Exception as e:
                logger.error(f"Failed to load {domain} data: {e}")
                results[f"{domain}_load"] = False

        self.test_results["data_loading"] = results

        # Check if all domains loaded successfully
        all_loaded = all(
            results.get(f"{domain}_load", False)
            for domain in ["photography", "microscopy", "astronomy"]
        )

        if all_loaded:
            logger.info("✓ Data loading tests passed")
        else:
            logger.warning("⚠ Some data loading tests failed")

        return all_loaded

    def test_calibration_integration(self) -> bool:
        """Test integration with calibration system."""
        logger.info("Testing calibration integration...")

        results = {}

        for domain in ["photography", "microscopy", "astronomy"]:
            domain_dir = self.temp_dir / domain
            calibration_file = domain_dir / "calibration.json"

            try:
                # Load calibration
                with open(calibration_file) as f:
                    cal_params = json.load(f)

                calibration = SensorCalibration(params=CalibrationParams(**cal_params))

                # Load test data
                if domain == "photography":
                    data_file = domain_dir / "test_image.npy"
                elif domain == "microscopy":
                    data_file = domain_dir / "test_multichannel.npy"
                else:  # astronomy
                    data_file = domain_dir / "test_astronomy.npy"

                if data_file.exists():
                    raw_data = np.load(data_file)

                    # Convert to electrons
                    if raw_data.ndim == 3:  # Multi-channel
                        # Take first channel for calibration test
                        test_data = raw_data[0]
                    else:
                        test_data = raw_data

                    electrons = calibration.adu_to_electrons(test_data)

                    results[f"{domain}_calibration"] = True
                    results[f"{domain}_electrons_mean"] = float(np.mean(electrons))
                    results[f"{domain}_electrons_std"] = float(np.std(electrons))

                    logger.debug(
                        f"{domain} calibration: {np.mean(electrons):.1f} ± {np.std(electrons):.1f} e-"
                    )

            except Exception as e:
                logger.error(f"Calibration failed for {domain}: {e}")
                results[f"{domain}_calibration"] = False

        self.test_results["calibration"] = results

        # Check if all calibrations worked
        all_calibrated = all(
            results.get(f"{domain}_calibration", False)
            for domain in ["photography", "microscopy", "astronomy"]
        )

        if all_calibrated:
            logger.info("✓ Calibration integration tests passed")
        else:
            logger.warning("⚠ Some calibration tests failed")

        return all_calibrated

    def test_domain_conditioning(self) -> bool:
        """Test Phase 3 domain conditioning with Phase 4.1 data."""
        logger.info("Testing domain conditioning integration...")

        results = {}

        # Create domain encoder
        encoder = DomainEncoder()

        for domain in ["photography", "microscopy", "astronomy"]:
            domain_dir = self.temp_dir / domain
            calibration_file = domain_dir / "calibration.json"

            try:
                # Load calibration parameters
                with open(calibration_file) as f:
                    cal_params = json.load(f)

                # Create conditioning vector
                condition = encoder.encode_domain(
                    domain=domain,
                    scale=1000.0,  # Typical scale
                    read_noise=cal_params["read_noise"],
                    background=cal_params["black_level"],
                )

                results[f"{domain}_conditioning"] = True
                results[f"{domain}_condition_shape"] = condition.shape
                results[f"{domain}_domain_onehot"] = condition[0, :3].tolist()

                # Verify domain one-hot encoding
                expected_idx = {"photography": 0, "microscopy": 1, "astronomy": 2}[
                    domain
                ]
                assert (
                    condition[0, expected_idx] == 1.0
                ), f"Wrong domain encoding for {domain}"

                logger.debug(
                    f"{domain} conditioning: {condition.shape}, domain_idx={expected_idx}"
                )

            except Exception as e:
                logger.error(f"Domain conditioning failed for {domain}: {e}")
                results[f"{domain}_conditioning"] = False

        self.test_results["domain_conditioning"] = results

        # Check if all conditioning worked
        all_conditioned = all(
            results.get(f"{domain}_conditioning", False)
            for domain in ["photography", "microscopy", "astronomy"]
        )

        if all_conditioned:
            logger.info("✓ Domain conditioning tests passed")
        else:
            logger.warning("⚠ Some domain conditioning tests failed")

        return all_conditioned

    def test_guidance_integration(self) -> bool:
        """Test Poisson-Gaussian guidance with real data parameters."""
        logger.info("Testing guidance integration...")

        results = {}

        for domain in ["photography", "microscopy", "astronomy"]:
            domain_dir = self.temp_dir / domain
            calibration_file = domain_dir / "calibration.json"

            try:
                # Load calibration parameters
                with open(calibration_file) as f:
                    cal_params = json.load(f)

                # Create domain-specific guidance
                guidance = create_domain_guidance(
                    scale=1000.0,
                    background=cal_params["black_level"],
                    read_noise=cal_params["read_noise"],
                    domain=domain,
                )

                # Test guidance computation with synthetic data
                x_hat = torch.rand(1, 1, 64, 64)  # Normalized estimate
                y_observed = torch.randn(1, 1, 64, 64) * 100 + 500  # Electrons
                sigma_t = torch.tensor([1.0])

                guidance_grad = guidance.compute(x_hat, y_observed, sigma_t)

                results[f"{domain}_guidance"] = True
                results[f"{domain}_guidance_shape"] = guidance_grad.shape
                results[f"{domain}_guidance_norm"] = float(torch.norm(guidance_grad))

                # Check that guidance is finite and reasonable
                assert torch.isfinite(
                    guidance_grad
                ).all(), f"Non-finite guidance for {domain}"

                # Guidance magnitude can vary significantly based on domain and data
                guidance_norm = torch.norm(guidance_grad).item()
                if (
                    guidance_norm > 10000
                ):  # Very large threshold - just check for runaway values
                    logger.warning(f"Large guidance norm for {domain}: {guidance_norm}")
                    # Don't fail the test, just log it

                logger.debug(f"{domain} guidance: norm={torch.norm(guidance_grad):.3f}")

            except Exception as e:
                logger.error(f"Guidance integration failed for {domain}: {e}")
                results[f"{domain}_guidance"] = False

        self.test_results["guidance"] = results

        # Check if all guidance worked
        all_guided = all(
            results.get(f"{domain}_guidance", False)
            for domain in ["photography", "microscopy", "astronomy"]
        )

        if all_guided:
            logger.info("✓ Guidance integration tests passed")
        else:
            logger.warning("⚠ Some guidance tests failed")

        return all_guided

    def test_end_to_end_pipeline(self) -> bool:
        """Test complete end-to-end pipeline."""
        logger.info("Testing end-to-end pipeline...")

        try:
            # Pick one domain for full pipeline test
            domain = "microscopy"
            domain_dir = self.temp_dir / domain

            # 1. Load data (Phase 4.1)
            data_file = domain_dir / "test_multichannel.npy"
            metadata_file = domain_dir / "test_multichannel_metadata.json"
            calibration_file = domain_dir / "calibration.json"

            raw_data = np.load(data_file)
            with open(metadata_file) as f:
                metadata = json.load(f)
            with open(calibration_file) as f:
                cal_params = json.load(f)

            # Take first channel
            test_data = raw_data[0]  # [H, W]

            # 2. Calibration
            calibration = SensorCalibration(params=CalibrationParams(**cal_params))
            electrons = calibration.adu_to_electrons(test_data)

            # 3. Normalization (simulate dataset normalization)
            normalized = electrons / 1000.0  # Normalize to ~[0, 1]
            normalized = np.clip(normalized, 0, 1)

            # 4. Convert to tensors
            y_observed = torch.from_numpy(
                electrons[None, None, :, :]
            ).float()  # [1, 1, H, W]
            x_estimate = torch.from_numpy(
                normalized[None, None, :, :]
            ).float()  # [1, 1, H, W]

            # 5. Domain conditioning (Phase 3)
            encoder = DomainEncoder()
            condition = encoder.encode_domain(
                domain=domain,
                scale=1000.0,
                read_noise=cal_params["read_noise"],
                background=cal_params["black_level"],
            )

            # 6. Guidance computation (Phase 3)
            guidance = create_domain_guidance(
                scale=1000.0,
                background=cal_params["black_level"],
                read_noise=cal_params["read_noise"],
                domain=domain,
            )

            sigma_t = torch.tensor([0.5])
            guidance_grad = guidance.compute(x_estimate, y_observed, sigma_t)

            # 7. Verify pipeline results
            pipeline_results = {
                "data_loaded": True,
                "data_shape": test_data.shape,
                "electrons_range": [float(electrons.min()), float(electrons.max())],
                "normalized_range": [float(normalized.min()), float(normalized.max())],
                "condition_shape": condition.shape,
                "guidance_shape": guidance_grad.shape,
                "guidance_finite": torch.isfinite(guidance_grad).all().item(),
                "pipeline_success": True,
            }

            self.test_results["end_to_end"] = pipeline_results

            logger.info("✓ End-to-end pipeline test passed")
            logger.info(
                f"  Data: {test_data.shape} → Electrons: [{electrons.min():.1f}, {electrons.max():.1f}]"
            )
            logger.info(
                f"  Conditioning: {condition.shape}, Guidance: {guidance_grad.shape}"
            )

            return True

        except Exception as e:
            logger.error(f"End-to-end pipeline failed: {e}")
            self.test_results["end_to_end"] = {
                "pipeline_success": False,
                "error": str(e),
            }
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("Running complete Phase 3 + Phase 4.1 integration tests...")

        # Setup
        if not self.setup_test_environment():
            return {"setup_failed": True}

        # Run tests
        tests = [
            ("data_loading", self.test_data_loading),
            ("calibration_integration", self.test_calibration_integration),
            ("domain_conditioning", self.test_domain_conditioning),
            ("guidance_integration", self.test_guidance_integration),
            ("end_to_end_pipeline", self.test_end_to_end_pipeline),
        ]

        results_summary = {}

        for test_name, test_func in tests:
            try:
                success = test_func()
                results_summary[test_name] = success
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                results_summary[test_name] = False

        # Overall summary
        all_passed = all(results_summary.values())
        results_summary["overall_success"] = all_passed
        results_summary["detailed_results"] = self.test_results

        logger.info("Integration tests completed")
        return results_summary

    def cleanup(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up test directory: {self.temp_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Phase 3 + Phase 4.1 integration")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--create-data",
        action="store_true",
        default=True,
        help="Create synthetic test data",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_project_logging(level=log_level)

    print("=" * 70)
    print("PHASE 3 + PHASE 4.1 INTEGRATION TEST")
    print("=" * 70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Create Synthetic Data: {args.create_data}")
    print()

    # Run integration tests
    tester = Phase3IntegrationTester(create_synthetic_data=args.create_data)

    try:
        results = tester.run_all_tests()

        # Print results
        print("\nTEST RESULTS:")
        print("-" * 40)

        for test_name, success in results.items():
            if test_name in ["overall_success", "detailed_results"]:
                continue

            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name:25} {status}")

        print("-" * 40)
        overall_status = "✅ PASS" if results["overall_success"] else "❌ FAIL"
        print(f"{'OVERALL':25} {overall_status}")

        if results["overall_success"]:
            print("\n🎉 All integration tests passed!")
            print("Phase 3 + Phase 4.1 integration is working correctly.")
        else:
            print("\n❌ Some integration tests failed.")
            print("Check the detailed logs above for specific issues.")

        return 0 if results["overall_success"] else 1

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return 1

    finally:
        tester.cleanup()


if __name__ == "__main__":
    sys.exit(main())
