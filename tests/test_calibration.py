"""
Comprehensive tests for sensor calibration system.

Tests cover:
- Parameter validation and edge cases
- ADU ↔ electron conversions accuracy
- Domain-specific defaults
- File I/O operations
- Error handling and recovery
- Integration with different data types (numpy/torch)
"""

import json

# Import calibration system
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))
from core.calibration import (
    CalibrationParams,
    SensorCalibration,
    create_calibration_from_params,
    load_calibration,
)
from core.exceptions import CalibrationError, ValidationError


class TestCalibrationParams:
    """Test CalibrationParams dataclass and validation."""

    def test_valid_parameters(self):
        """Test creation with valid parameters."""
        params = CalibrationParams(
            gain=1.5,
            black_level=100,
            white_level=16383,
            read_noise=2.0,
            dark_current=0.1,
            quantum_efficiency=0.8,
            pixel_size=4.0,
            pixel_unit="um",
            sensor_name="Test_Sensor",
            bit_depth=14,
            domain="photography",
        )

        assert params.gain == 1.5
        assert params.black_level == 100
        assert params.white_level == 16383
        assert params.read_noise == 2.0
        assert params.domain == "photography"

    def test_parameter_validation_gain(self):
        """Test gain parameter validation."""
        with pytest.raises(CalibrationError, match="Gain must be positive"):
            CalibrationParams(
                gain=0.0, black_level=100, white_level=16383, read_noise=2.0  # Invalid
            )

        with pytest.raises(CalibrationError, match="Gain must be positive"):
            CalibrationParams(
                gain=-1.0, black_level=100, white_level=16383, read_noise=2.0  # Invalid
            )

    def test_parameter_validation_levels(self):
        """Test black/white level validation."""
        with pytest.raises(
            CalibrationError, match="White level .* must be > black level"
        ):
            CalibrationParams(
                gain=1.0,
                black_level=1000,
                white_level=500,  # Invalid: white < black
                read_noise=2.0,
            )

        with pytest.raises(
            CalibrationError, match="White level .* must be > black level"
        ):
            CalibrationParams(
                gain=1.0,
                black_level=1000,
                white_level=1000,  # Invalid: white = black
                read_noise=2.0,
            )

    def test_parameter_validation_noise(self):
        """Test noise parameter validation."""
        with pytest.raises(CalibrationError, match="Read noise must be non-negative"):
            CalibrationParams(
                gain=1.0, black_level=100, white_level=16383, read_noise=-1.0  # Invalid
            )

        with pytest.raises(CalibrationError, match="Dark current must be non-negative"):
            CalibrationParams(
                gain=1.0,
                black_level=100,
                white_level=16383,
                read_noise=2.0,
                dark_current=-0.1,  # Invalid
            )

    def test_parameter_validation_quantum_efficiency(self):
        """Test quantum efficiency validation."""
        with pytest.raises(CalibrationError, match="Quantum efficiency must be in"):
            CalibrationParams(
                gain=1.0,
                black_level=100,
                white_level=16383,
                read_noise=2.0,
                quantum_efficiency=1.5,  # Invalid: > 1
            )

        with pytest.raises(CalibrationError, match="Quantum efficiency must be in"):
            CalibrationParams(
                gain=1.0,
                black_level=100,
                white_level=16383,
                read_noise=2.0,
                quantum_efficiency=-0.1,  # Invalid: < 0
            )

    def test_parameter_validation_bit_depth(self):
        """Test bit depth validation."""
        with pytest.raises(CalibrationError, match="Bit depth must be standard value"):
            CalibrationParams(
                gain=1.0,
                black_level=100,
                white_level=16383,
                read_noise=2.0,
                bit_depth=13,  # Invalid: not standard
            )

    def test_parameter_validation_consistency(self):
        """Test parameter consistency checks."""
        with pytest.raises(
            CalibrationError, match="White level .* exceeds bit depth capacity"
        ):
            CalibrationParams(
                gain=1.0,
                black_level=100,
                white_level=70000,  # Invalid: exceeds 16-bit
                read_noise=2.0,
                bit_depth=16,
            )

    def test_dynamic_range_calculation(self):
        """Test dynamic range calculation."""
        params = CalibrationParams(
            gain=1.0, black_level=100, white_level=16383, read_noise=3.0
        )

        dr_db = params.get_dynamic_range_db()

        # Expected: 20 * log10((16383-100) * 1.0 / 3.0) ≈ 64.7 dB
        expected_dr = 20 * np.log10((16383 - 100) / 3.0)
        assert abs(dr_db - expected_dr) < 0.1

    def test_full_well_capacity(self):
        """Test full well capacity calculation."""
        params = CalibrationParams(
            gain=2.0, black_level=100, white_level=16383, read_noise=3.0
        )

        fwc = params.get_full_well_capacity()
        expected_fwc = (16383 - 100) * 2.0  # 32566 electrons

        assert abs(fwc - expected_fwc) < 1.0

    def test_serialization(self):
        """Test dictionary serialization/deserialization."""
        params = CalibrationParams(
            gain=1.5,
            black_level=100,
            white_level=16383,
            read_noise=2.0,
            sensor_name="Test_Sensor",
        )

        # Serialize
        data = params.to_dict()
        assert isinstance(data, dict)
        assert data["gain"] == 1.5
        assert data["sensor_name"] == "Test_Sensor"

        # Deserialize
        restored = CalibrationParams.from_dict(data)
        assert restored.gain == params.gain
        assert restored.sensor_name == params.sensor_name
        assert restored.read_noise == params.read_noise


class TestSensorCalibration:
    """Test SensorCalibration class functionality."""

    @pytest.fixture
    def valid_params(self):
        """Create valid calibration parameters for testing."""
        return CalibrationParams(
            gain=1.0,
            black_level=512,
            white_level=16383,
            read_noise=3.0,
            dark_current=0.1,
            quantum_efficiency=0.8,
            pixel_size=4.29,
            pixel_unit="um",
            sensor_name="Test_Sensor",
            bit_depth=14,
            domain="photography",
        )

    @pytest.fixture
    def calibration(self, valid_params):
        """Create SensorCalibration instance for testing."""
        return SensorCalibration(params=valid_params)

    def test_initialization_with_params(self, valid_params):
        """Test initialization with direct parameters."""
        cal = SensorCalibration(params=valid_params)

        assert cal.params is not None
        assert cal.params.gain == 1.0
        assert cal.params.sensor_name == "Test_Sensor"

    def test_initialization_with_domain_default(self):
        """Test initialization with domain defaults."""
        cal = SensorCalibration(domain="photography")

        assert cal.params is not None
        assert cal.params.domain == "photography"
        assert cal.params.gain > 0
        assert cal.params.sensor_name == "Generic_Photography_Sensor"

    def test_initialization_failure(self):
        """Test initialization failure cases."""
        with pytest.raises(CalibrationError, match="No valid calibration source"):
            SensorCalibration()  # No parameters provided

        with pytest.raises(CalibrationError, match="No valid calibration source"):
            SensorCalibration(domain="invalid_domain")

    def test_adu_to_electrons_numpy(self, calibration):
        """Test ADU to electrons conversion with numpy arrays."""
        # Test data
        adu = np.array([512, 1024, 2048, 16383])  # black_level=512, gain=1.0

        electrons = calibration.adu_to_electrons(adu)

        expected = np.array([0, 512, 1536, 15871])  # (adu - 512) * 1.0
        np.testing.assert_array_almost_equal(electrons, expected)

        # Check non-negative constraint
        adu_negative = np.array([0, 100, 512])  # Below black level
        electrons_negative = calibration.adu_to_electrons(adu_negative)
        assert np.all(electrons_negative >= 0)

    def test_adu_to_electrons_torch(self, calibration):
        """Test ADU to electrons conversion with torch tensors."""
        adu = torch.tensor([512.0, 1024.0, 2048.0, 16383.0])

        electrons = calibration.adu_to_electrons(adu)

        expected = torch.tensor([0.0, 512.0, 1536.0, 15871.0])
        torch.testing.assert_close(electrons, expected)

        # Check type preservation
        assert isinstance(electrons, torch.Tensor)

    def test_electrons_to_adu_numpy(self, calibration):
        """Test electrons to ADU conversion with numpy arrays."""
        electrons = np.array([0, 512, 1536, 15871])

        adu = calibration.electrons_to_adu(electrons)

        expected = np.array([512, 1024, 2048, 16383])  # electrons / 1.0 + 512
        np.testing.assert_array_almost_equal(adu, expected)

        # Check clamping to valid range
        electrons_high = np.array([50000])  # Very high
        adu_high = calibration.electrons_to_adu(electrons_high)
        assert adu_high[0] <= calibration.params.white_level

    def test_electrons_to_adu_torch(self, calibration):
        """Test electrons to ADU conversion with torch tensors."""
        electrons = torch.tensor([0.0, 512.0, 1536.0, 15871.0])

        adu = calibration.electrons_to_adu(electrons)

        expected = torch.tensor([512.0, 1024.0, 2048.0, 16383.0])
        torch.testing.assert_close(adu, expected)

    def test_round_trip_conversion(self, calibration):
        """Test ADU → electrons → ADU round trip."""
        original_adu = np.array([600, 1000, 5000, 15000])

        # Round trip
        electrons = calibration.adu_to_electrons(original_adu)
        recovered_adu = calibration.electrons_to_adu(electrons)

        np.testing.assert_array_almost_equal(original_adu, recovered_adu, decimal=3)

    def test_process_raw_basic(self, calibration):
        """Test basic raw processing."""
        raw_adu = np.array([[1000, 2000], [3000, 16383]])  # Include saturated pixel

        electrons, mask = calibration.process_raw(raw_adu, return_mask=True)

        # Check conversion
        expected_electrons = (raw_adu - 512) * 1.0
        expected_electrons = np.maximum(expected_electrons, 0)
        np.testing.assert_array_almost_equal(electrons, expected_electrons)

        # Check mask (saturated pixel should be masked)
        expected_mask = raw_adu < 16383  # white_level
        np.testing.assert_array_equal(mask, expected_mask)

    def test_process_raw_dark_correction(self, calibration):
        """Test raw processing with dark current correction."""
        raw_adu = np.array([1000, 2000])
        exposure_time = 10.0  # seconds

        electrons = calibration.process_raw(
            raw_adu,
            return_mask=False,
            apply_dark_correction=True,
            exposure_time=exposure_time,
        )

        # Expected: (adu - black_level) * gain - dark_current * exposure_time
        expected = (raw_adu - 512) * 1.0 - 0.1 * 10.0  # dark_current=0.1
        expected = np.maximum(expected, 0)

        np.testing.assert_array_almost_equal(electrons, expected)

    def test_noise_variance_estimation(self, calibration):
        """Test noise variance estimation."""
        signal_electrons = np.array([100, 1000, 10000])

        variance = calibration.estimate_noise_variance(signal_electrons)

        # Expected: signal + read_noise²
        expected = signal_electrons + 3.0**2  # read_noise=3.0
        np.testing.assert_array_almost_equal(variance, expected)

    def test_snr_estimation(self, calibration):
        """Test SNR estimation."""
        signal_electrons = 1000.0

        snr_db = calibration.estimate_snr_db(signal_electrons)

        # Expected: 20 * log10(signal / sqrt(signal + read_noise²))
        noise_std = np.sqrt(1000 + 9)  # read_noise²=9
        expected_snr = 20 * np.log10(1000 / noise_std)

        assert abs(snr_db - expected_snr) < 0.1

    def test_calibration_summary(self, calibration):
        """Test calibration summary generation."""
        summary = calibration.get_calibration_summary()

        assert isinstance(summary, dict)
        assert "sensor_name" in summary
        assert "gain_e_per_adu" in summary
        assert "dynamic_range_db" in summary
        assert summary["sensor_name"] == "Test_Sensor"
        assert summary["gain_e_per_adu"] == 1.0


class TestFileOperations:
    """Test file I/O operations."""

    @pytest.fixture
    def temp_calibration_file(self):
        """Create temporary calibration file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            calibration_data = {
                "calibration": {
                    "gain": 1.5,
                    "black_level": 100,
                    "white_level": 16383,
                    "read_noise": 2.0,
                    "dark_current": 0.05,
                    "quantum_efficiency": 0.9,
                    "pixel_size": 4.0,
                    "pixel_unit": "um",
                    "sensor_name": "Test_File_Sensor",
                    "bit_depth": 14,
                    "domain": "photography",
                }
            }
            json.dump(calibration_data, f, indent=2)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_load_from_file(self, temp_calibration_file):
        """Test loading calibration from file."""
        cal = SensorCalibration(calibration_file=temp_calibration_file)

        assert cal.params is not None
        assert cal.params.gain == 1.5
        assert cal.params.sensor_name == "Test_File_Sensor"
        assert cal.params.black_level == 100

    def test_save_to_file(self):
        """Test saving calibration to file."""
        params = CalibrationParams(
            gain=2.0,
            black_level=200,
            white_level=32767,
            read_noise=1.5,
            sensor_name="Save_Test_Sensor",
        )

        cal = SensorCalibration(params=params)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save
            cal.save_to_file(temp_path)

            # Verify file exists and load back
            assert Path(temp_path).exists()

            cal_loaded = SensorCalibration(calibration_file=temp_path)
            assert cal_loaded.params.gain == 2.0
            assert cal_loaded.params.sensor_name == "Save_Test_Sensor"

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_invalid_file(self):
        """Test loading from invalid file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            with pytest.raises(CalibrationError, match="Failed to load calibration"):
                SensorCalibration(calibration_file=temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file with domain fallback."""
        cal = SensorCalibration(
            calibration_file="nonexistent_file.json", domain="microscopy"
        )

        # Should fall back to domain default
        assert cal.params is not None
        assert cal.params.domain == "microscopy"


class TestDomainDefaults:
    """Test domain-specific default calibrations."""

    def test_photography_defaults(self):
        """Test photography domain defaults."""
        cal = SensorCalibration(domain="photography")

        assert cal.params.domain == "photography"
        assert cal.params.pixel_unit == "um"
        assert cal.params.bit_depth == 14
        assert cal.params.gain > 0
        assert cal.params.read_noise > 0

    def test_microscopy_defaults(self):
        """Test microscopy domain defaults."""
        cal = SensorCalibration(domain="microscopy")

        assert cal.params.domain == "microscopy"
        assert cal.params.pixel_unit == "um"
        assert cal.params.bit_depth == 16
        assert cal.params.read_noise < 2.0  # Should be low for scientific sensor
        assert cal.params.quantum_efficiency > 0.9  # Should be high

    def test_astronomy_defaults(self):
        """Test astronomy domain defaults."""
        cal = SensorCalibration(domain="astronomy")

        assert cal.params.domain == "astronomy"
        assert cal.params.pixel_unit == "arcsec"
        assert cal.params.black_level == 0  # CCD bias subtracted
        assert cal.params.bit_depth == 16

    def test_create_default_files(self):
        """Test creation of default calibration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            created_files = SensorCalibration.create_default_calibration_files(temp_dir)

            assert len(created_files) == 3  # photography, microscopy, astronomy

            for filepath in created_files:
                assert filepath.exists()
                assert filepath.suffix == ".json"

                # Verify file can be loaded
                cal = SensorCalibration(calibration_file=filepath)
                assert cal.params is not None


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_load_calibration_function(self):
        """Test load_calibration convenience function."""
        cal = load_calibration(domain="photography")

        assert isinstance(cal, SensorCalibration)
        assert cal.params.domain == "photography"

    def test_create_calibration_from_params_function(self):
        """Test create_calibration_from_params convenience function."""
        cal = create_calibration_from_params(
            gain=1.5,
            black_level=100,
            white_level=16383,
            read_noise=2.0,
            sensor_name="Convenience_Test",
        )

        assert isinstance(cal, SensorCalibration)
        assert cal.params.gain == 1.5
        assert cal.params.sensor_name == "Convenience_Test"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_no_calibration_loaded_error(self):
        """Test operations without loaded calibration."""
        # Create calibration without parameters (should fail)
        with pytest.raises(CalibrationError):
            cal = SensorCalibration.__new__(SensorCalibration)
            cal.params = None
            cal.adu_to_electrons(np.array([1000]))

    def test_extreme_values(self):
        """Test handling of extreme input values."""
        cal = SensorCalibration(domain="photography")

        # Very large ADU values
        large_adu = np.array([1e10])
        electrons = cal.adu_to_electrons(large_adu)
        assert np.all(np.isfinite(electrons))

        # Very small/negative ADU values
        small_adu = np.array([-1000, 0])
        electrons = cal.adu_to_electrons(small_adu)
        assert np.all(electrons >= 0)  # Should be clamped

    def test_mixed_data_types(self):
        """Test handling of different input data types."""
        cal = SensorCalibration(domain="photography")

        # Test with different numpy dtypes
        for dtype in [np.int16, np.int32, np.float32, np.float64]:
            adu = np.array([1000, 2000], dtype=dtype)
            electrons = cal.adu_to_electrons(adu)
            assert electrons.dtype in [np.float32, np.float64]

        # Test with torch tensors of different dtypes
        for dtype in [torch.float32, torch.float64]:
            adu = torch.tensor([1000.0, 2000.0], dtype=dtype)
            electrons = cal.adu_to_electrons(adu)
            assert isinstance(electrons, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__])
