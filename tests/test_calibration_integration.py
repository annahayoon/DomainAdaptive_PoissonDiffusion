"""
Integration tests for calibration system with other components.

Tests the interaction between calibration and transforms, ensuring
the complete pipeline works correctly for different domains.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from core.calibration import CalibrationParams, SensorCalibration
from core.exceptions import CalibrationError
from core.transforms import ImageMetadata, ReversibleTransform


class TestCalibrationTransformIntegration:
    """Test integration between calibration and transform systems."""

    @pytest.fixture
    def photography_calibration(self):
        """Photography domain calibration."""
        return SensorCalibration(domain="photography")

    @pytest.fixture
    def microscopy_calibration(self):
        """Microscopy domain calibration."""
        return SensorCalibration(domain="microscopy")

    @pytest.fixture
    def astronomy_calibration(self):
        """Astronomy domain calibration."""
        return SensorCalibration(domain="astronomy")

    @pytest.fixture
    def transform(self):
        """Reversible transform for testing."""
        return ReversibleTransform(target_size=128)

    def test_photography_pipeline(self, photography_calibration, transform):
        """Test complete photography processing pipeline."""
        # Simulate RAW camera data
        raw_adu = np.random.randint(
            photography_calibration.params.black_level + 100,
            photography_calibration.params.white_level - 100,
            size=(200, 300),
        ).astype(np.float32)

        # Convert to tensor and add batch/channel dimensions
        raw_tensor = torch.from_numpy(raw_adu).unsqueeze(0).unsqueeze(0)

        # Process through calibration
        electrons, mask = photography_calibration.process_raw(
            raw_tensor, return_mask=True
        )

        # Normalize for model (simple normalization for testing)
        scale = 10000.0  # electrons
        normalized = electrons / scale
        normalized = torch.clamp(normalized, 0, 1)

        # Transform to model size
        transformed, metadata = transform.forward(
            normalized,
            pixel_size=photography_calibration.params.pixel_size,
            pixel_unit=photography_calibration.params.pixel_unit,
            domain=photography_calibration.params.domain,
            black_level=photography_calibration.params.black_level,
            white_level=photography_calibration.params.white_level,
        )

        # Verify transform
        assert transformed.shape == (1, 1, 128, 128)
        assert metadata.domain == "photography"
        assert metadata.pixel_unit == "um"

        # Inverse transform
        reconstructed = transform.inverse(transformed, metadata)

        # Verify reconstruction
        assert reconstructed.shape == normalized.shape

        # Convert back to ADU for validation
        reconstructed_electrons = reconstructed * scale
        reconstructed_adu = photography_calibration.electrons_to_adu(
            reconstructed_electrons
        )

        # Should be reasonable (allowing for interpolation error)
        assert torch.all(reconstructed_adu >= 0)
        assert torch.all(
            reconstructed_adu <= photography_calibration.params.white_level
        )

    def test_microscopy_pipeline(self, microscopy_calibration, transform):
        """Test complete microscopy processing pipeline."""
        # Simulate fluorescence microscopy data (typically lower signal)
        raw_adu = np.random.randint(
            microscopy_calibration.params.black_level + 50,
            microscopy_calibration.params.black_level + 5000,  # Lower signal
            size=(512, 512),
        ).astype(np.float32)

        raw_tensor = torch.from_numpy(raw_adu).unsqueeze(0).unsqueeze(0)

        # Process with dark correction (common in microscopy)
        electrons = microscopy_calibration.process_raw(
            raw_tensor, return_mask=False, apply_dark_correction=True, exposure_time=1.0
        )

        # Verify calibration characteristics
        assert torch.all(electrons >= 0)

        # Check SNR estimation
        mean_signal = electrons.mean().item()
        snr_db = microscopy_calibration.estimate_snr_db(mean_signal)
        assert isinstance(snr_db, float)
        assert snr_db > 0  # Should have positive SNR

        # Transform pipeline
        scale = 5000.0  # Lower scale for microscopy
        normalized = torch.clamp(electrons / scale, 0, 1)

        transformed, metadata = transform.forward(
            normalized,
            pixel_size=microscopy_calibration.params.pixel_size,
            pixel_unit=microscopy_calibration.params.pixel_unit,
            domain=microscopy_calibration.params.domain,
        )

        assert metadata.domain == "microscopy"
        assert metadata.pixel_unit == "um"
        assert metadata.pixel_size < 1.0  # Should be sub-micron

    def test_astronomy_pipeline(self, astronomy_calibration, transform):
        """Test complete astronomy processing pipeline."""
        # Simulate astronomy CCD data (very low signal)
        raw_adu = np.random.poisson(
            lam=50, size=(1024, 1024)  # Very low photon count
        ).astype(np.float32)

        # Add read noise
        read_noise_adu = np.random.normal(
            0,
            astronomy_calibration.params.read_noise / astronomy_calibration.params.gain,
            size=raw_adu.shape,
        )
        raw_adu = raw_adu + read_noise_adu

        raw_tensor = torch.from_numpy(raw_adu).unsqueeze(0).unsqueeze(0)

        # Process
        electrons, mask = astronomy_calibration.process_raw(
            raw_tensor, return_mask=True
        )

        # Verify low-light characteristics
        mean_electrons = electrons.mean().item()
        assert mean_electrons < 200  # Should be low signal

        # Estimate noise characteristics
        variance = astronomy_calibration.estimate_noise_variance(electrons)
        assert torch.all(variance > 0)

        # Transform pipeline
        scale = 1000.0  # Very low scale for astronomy
        normalized = torch.clamp(electrons / scale, 0, 1)

        transformed, metadata = transform.forward(
            normalized,
            pixel_size=astronomy_calibration.params.pixel_size,
            pixel_unit=astronomy_calibration.params.pixel_unit,
            domain=astronomy_calibration.params.domain,
        )

        assert metadata.domain == "astronomy"
        assert metadata.pixel_unit == "arcsec"

    def test_cross_domain_metadata_consistency(self, transform):
        """Test that metadata is consistent across domains."""
        domains = ["photography", "microscopy", "astronomy"]

        for domain in domains:
            cal = SensorCalibration(domain=domain)

            # Create test data
            test_data = torch.randn(1, 1, 100, 150)

            # Transform with domain-specific metadata
            transformed, metadata = transform.forward(
                test_data,
                pixel_size=cal.params.pixel_size,
                pixel_unit=cal.params.pixel_unit,
                domain=cal.params.domain,
                black_level=cal.params.black_level,
                white_level=cal.params.white_level,
            )

            # Verify metadata consistency
            assert metadata.domain == domain
            assert metadata.pixel_unit == cal.params.pixel_unit
            assert metadata.black_level == cal.params.black_level
            assert metadata.white_level == cal.params.white_level

            # Verify reconstruction works
            reconstructed = transform.inverse(transformed, metadata)
            assert reconstructed.shape == test_data.shape


class TestCalibrationFileIntegration:
    """Test calibration file operations in realistic scenarios."""

    def test_create_and_use_calibration_files(self):
        """Test creating calibration files and using them."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create default calibration files
            created_files = SensorCalibration.create_default_calibration_files(temp_dir)

            assert len(created_files) == 3

            # Test each created file
            for filepath in created_files:
                # Load calibration from file
                cal = SensorCalibration(calibration_file=filepath)

                # Verify it works for processing
                test_adu = np.array([1000, 2000, 5000])
                electrons = cal.adu_to_electrons(test_adu)

                assert np.all(electrons >= 0)
                assert len(electrons) == len(test_adu)

                # Verify summary works
                summary = cal.get_calibration_summary()
                assert "sensor_name" in summary
                assert "domain" in summary

    def test_custom_calibration_workflow(self):
        """Test creating, saving, and loading custom calibration."""
        # Create custom calibration
        custom_params = CalibrationParams(
            gain=1.2,
            black_level=256,
            white_level=32767,
            read_noise=2.5,
            dark_current=0.05,
            quantum_efficiency=0.85,
            pixel_size=3.76,
            pixel_unit="um",
            sensor_name="Custom_Test_Sensor",
            bit_depth=16,
            domain="photography",
        )

        cal = SensorCalibration(params=custom_params)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save custom calibration
            cal.save_to_file(temp_path)

            # Load it back
            cal_loaded = SensorCalibration(calibration_file=temp_path)

            # Verify parameters match
            assert cal_loaded.params.gain == 1.2
            assert cal_loaded.params.sensor_name == "Custom_Test_Sensor"
            assert cal_loaded.params.pixel_size == 3.76

            # Test processing with custom calibration
            test_adu = np.array([500, 1000, 10000])
            electrons = cal_loaded.adu_to_electrons(test_adu)

            # Verify conversion uses custom parameters
            expected = (test_adu - 256) * 1.2
            expected = np.maximum(expected, 0)
            np.testing.assert_array_almost_equal(electrons, expected)

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCalibrationValidation:
    """Test calibration validation in realistic scenarios."""

    def test_realistic_parameter_ranges(self):
        """Test that realistic parameter ranges are accepted."""
        # Test various realistic sensor configurations
        realistic_configs = [
            # High-end DSLR
            {
                "gain": 0.8,
                "black_level": 2048,
                "white_level": 16383,
                "read_noise": 2.0,
                "bit_depth": 14,
                "domain": "photography",
            },
            # Scientific CMOS
            {
                "gain": 0.46,
                "black_level": 100,
                "white_level": 65535,
                "read_noise": 1.2,
                "bit_depth": 16,
                "domain": "microscopy",
            },
            # Astronomy CCD
            {
                "gain": 1.5,
                "black_level": 0,
                "white_level": 65535,
                "read_noise": 4.0,
                "bit_depth": 16,
                "domain": "astronomy",
            },
        ]

        for config in realistic_configs:
            params = CalibrationParams(**config)
            cal = SensorCalibration(params=params)

            # Should validate successfully
            assert cal.validate_calibration()

            # Should produce reasonable dynamic range
            dr_db = params.get_dynamic_range_db()
            assert dr_db > 40  # Reasonable minimum
            assert dr_db < 100  # Reasonable maximum

    def test_edge_case_validation(self):
        """Test validation of edge cases."""
        # Very low gain (high conversion factor)
        params = CalibrationParams(
            gain=0.1, black_level=100, white_level=16383, read_noise=5.0
        )
        cal = SensorCalibration(params=params)
        assert cal.validate_calibration()

        # Very high gain (low conversion factor)
        params = CalibrationParams(
            gain=10.0, black_level=100, white_level=16383, read_noise=1.0
        )
        cal = SensorCalibration(params=params)
        assert cal.validate_calibration()

        # Minimal dynamic range (should warn but not fail)
        params = CalibrationParams(
            gain=1.0, black_level=100, white_level=200, read_noise=50.0
        )
        cal = SensorCalibration(params=params)
        # Should validate but with warnings
        assert cal.validate_calibration()


class TestNumericalAccuracy:
    """Test numerical accuracy of calibration operations."""

    def test_conversion_accuracy(self):
        """Test accuracy of ADU ↔ electron conversions."""
        cal = SensorCalibration(domain="photography")

        # Test with various precisions
        for dtype in [np.float32, np.float64]:
            adu = np.array([1000.5, 2000.7, 5000.3], dtype=dtype)

            # Round trip conversion
            electrons = cal.adu_to_electrons(adu)
            adu_recovered = cal.electrons_to_adu(electrons)

            # Should be accurate to floating point precision
            np.testing.assert_array_almost_equal(adu, adu_recovered, decimal=5)

    def test_noise_estimation_accuracy(self):
        """Test accuracy of noise estimation."""
        cal = SensorCalibration(domain="microscopy")

        # Test Poisson + Gaussian noise model
        signal_levels = np.array([1, 10, 100, 1000, 10000])

        for signal in signal_levels:
            variance = cal.estimate_noise_variance(signal)
            expected_variance = signal + cal.params.read_noise**2

            assert abs(variance - expected_variance) < 1e-10

    def test_snr_calculation_accuracy(self):
        """Test SNR calculation accuracy."""
        cal = SensorCalibration(domain="astronomy")

        # Known test case
        signal = 1000.0
        read_noise = cal.params.read_noise

        snr_db = cal.estimate_snr_db(signal)

        # Manual calculation
        noise_variance = signal + read_noise**2
        noise_std = np.sqrt(noise_variance)
        expected_snr_db = 20 * np.log10(signal / noise_std)

        assert abs(snr_db - expected_snr_db) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
