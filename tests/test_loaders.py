"""
Comprehensive tests for format-specific data loaders.

Tests all three domain loaders (photography, microscopy, astronomy)
with synthetic data and validation of metadata extraction.
"""

import json
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from core.exceptions import DataError, ValidationError
from data.loaders import (
    AstronomyLoader,
    FormatDetector,
    ImageMetadata,
    MicroscopyLoader,
    PhotographyLoader,
    get_image_metadata,
    get_supported_formats,
    load_image,
)


class TestImageMetadata:
    """Test ImageMetadata dataclass."""

    def test_metadata_creation(self):
        """Test basic metadata creation."""
        metadata = ImageMetadata(
            filepath="/test/image.tif",
            format=".tif",
            domain="microscopy",
            height=512,
            width=512,
            channels=1,
            bit_depth=16,
            dtype="uint16",
        )

        assert metadata.filepath == "/test/image.tif"
        assert metadata.domain == "microscopy"
        assert metadata.height == 512
        assert metadata.width == 512

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        metadata = ImageMetadata(
            filepath="/test/image.tif",
            format=".tif",
            domain="microscopy",
            height=512,
            width=512,
            channels=1,
            bit_depth=16,
            dtype="uint16",
            pixel_size=0.65,
            pixel_unit="um",
        )

        data_dict = metadata.to_dict()

        assert "filepath" in data_dict
        assert "pixel_size" in data_dict
        assert data_dict["pixel_size"] == 0.65
        assert data_dict["pixel_unit"] == "um"

        # None values should be excluded
        metadata.exposure_time = None
        data_dict = metadata.to_dict()
        assert "exposure_time" not in data_dict


class TestPhotographyLoader:
    """Test photography RAW loader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = PhotographyLoader(validate_on_load=False)

    def test_supported_extensions(self):
        """Test supported file extensions."""
        extensions = self.loader.get_supported_extensions()

        assert ".arw" in extensions
        assert ".dng" in extensions
        assert ".nef" in extensions
        assert ".cr2" in extensions

    def test_file_validation(self):
        """Test file validation logic."""
        # Test non-existent file
        assert not self.loader.validate_file("/nonexistent/file.arw")

        # Test unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            assert not self.loader.validate_file(tmp.name)

        # Test empty file
        with tempfile.NamedTemporaryFile(suffix=".arw") as tmp:
            assert not self.loader.validate_file(tmp.name)

    def test_load_raw_data_mock(self):
        """Test RAW data loading with mocked rawpy."""
        # Mock the rawpy module at the loader level
        with patch.object(self.loader, "rawpy") as mock_rawpy:
            # Create mock RAW file
            mock_raw = Mock()
            mock_raw.raw_image = np.random.randint(
                0, 4096, (3000, 4000), dtype=np.uint16
            )
            mock_raw.camera_make = "Sony"
            mock_raw.camera_model = "A7S"
            mock_raw.other = {"ISOSpeedRatings": 3200, "ExposureTime": 0.1}
            mock_raw.black_level_per_channel = [512, 512, 512, 512]
            mock_raw.white_level = 16383
            mock_raw.color_desc = b"RGGB"
            mock_raw.raw_colors = np.array([0, 1, 1, 2])

            mock_rawpy.imread.return_value.__enter__.return_value = mock_raw

            # Test loading
            with tempfile.NamedTemporaryFile(suffix=".arw") as tmp:
                tmp.write(b"fake raw data")
                tmp.flush()

                raw_data, metadata = self.loader.load_raw_data(tmp.name)

                assert raw_data.shape == (3000, 4000)
                assert raw_data.dtype == np.float32
                assert metadata.domain == "photography"
                assert metadata.iso == 3200
                assert metadata.exposure_time == 0.1
                assert metadata.black_level == 512
                assert metadata.white_level == 16383

    def test_demosaic_option(self):
        """Test demosaicing option."""
        loader = PhotographyLoader(demosaic=True, validate_on_load=False)

        # Mock the rawpy module at the loader level
        with patch.object(loader, "rawpy") as mock_rawpy:
            # Mock demosaiced output
            mock_raw = Mock()
            mock_raw.raw_image = np.random.randint(
                0, 4096, (3000, 4000), dtype=np.uint16
            )
            mock_raw.postprocess.return_value = np.random.randint(
                0, 65535, (3000, 4000, 3), dtype=np.uint16
            )
            mock_raw.camera_make = "Sony"
            mock_raw.camera_model = "A7S"
            mock_raw.other = {}

            mock_rawpy.imread.return_value.__enter__.return_value = mock_raw

            with tempfile.NamedTemporaryFile(suffix=".arw") as tmp:
                tmp.write(b"fake raw data")
                tmp.flush()

                raw_data, metadata = loader.load_raw_data(tmp.name)

                assert raw_data.shape == (3000, 4000, 3)
                assert metadata.channels == 3

    def test_statistics(self):
        """Test loader statistics."""
        stats = self.loader.get_statistics()

        assert "files_loaded" in stats
        assert "load_errors" in stats
        assert "success_rate" in stats
        assert stats["success_rate"] == 0.0  # No files loaded yet


class TestMicroscopyLoader:
    """Test microscopy TIFF loader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = MicroscopyLoader(validate_on_load=False)

    def test_supported_extensions(self):
        """Test supported file extensions."""
        extensions = self.loader.get_supported_extensions()

        assert ".tif" in extensions
        assert ".tiff" in extensions

    def create_test_tiff(self, shape=(512, 512), dtype=np.uint16):
        """Create a test TIFF file."""
        data = np.random.randint(0, 2**12, shape, dtype=dtype)

        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)

        try:
            from PIL import Image

            if len(shape) == 2:
                img = Image.fromarray(data)
            else:
                # Multi-channel
                img = Image.fromarray(data[0])  # Take first channel
            img.save(tmp.name)
        except ImportError:
            # Create fake file for testing
            tmp.write(b"fake tiff data")

        tmp.flush()
        return tmp.name

    def test_load_tiff_data_mock(self):
        """Test TIFF data loading with mocked libraries."""
        # Mock the tifffile module at the loader level
        with patch.object(self.loader, "tifffile") as mock_tifffile:
            # Mock tifffile
            mock_tif = Mock()
            mock_tif.asarray.return_value = np.random.randint(
                0, 4096, (512, 512), dtype=np.uint16
            )

            mock_page = Mock()
            mock_page.imagelength = 512
            mock_page.imagewidth = 512
            mock_page.bitspersample = 16
            mock_page.samplesperpixel = 1
            mock_page.tags = {}

            mock_tif.pages = [mock_page]
            mock_tif.imagej_metadata = None

            mock_tifffile.TiffFile.return_value.__enter__.return_value = mock_tif

            with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
                tmp.write(b"fake tiff data")
                tmp.flush()

                raw_data, metadata = self.loader.load_raw_data(tmp.name)

                assert raw_data.shape == (512, 512)
                assert raw_data.dtype == np.float32
                assert metadata.domain == "microscopy"
                assert metadata.height == 512
                assert metadata.width == 512
                assert metadata.bit_depth == 16

    def test_multi_channel_tiff(self):
        """Test multi-channel TIFF loading."""
        # Mock the tifffile module at the loader level
        with patch.object(self.loader, "tifffile") as mock_tifffile:
            # Mock multi-channel data
            mock_tif = Mock()
            mock_tif.asarray.return_value = np.random.randint(
                0, 4096, (3, 512, 512), dtype=np.uint16
            )

            mock_page = Mock()
            mock_page.imagelength = 512
            mock_page.imagewidth = 512
            mock_page.bitspersample = 16
            mock_page.samplesperpixel = 3
            mock_page.tags = {}

            mock_tif.pages = [mock_page]
            mock_tif.imagej_metadata = None

            mock_tifffile.TiffFile.return_value.__enter__.return_value = mock_tif

            with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
                tmp.write(b"fake tiff data")
                tmp.flush()

                raw_data, metadata = self.loader.load_raw_data(tmp.name)

                assert raw_data.shape == (3, 512, 512)
                assert metadata.channels == 3

    def test_imagej_metadata_extraction(self):
        """Test ImageJ metadata extraction."""
        with patch.object(self.loader, "tifffile") as mock_tifffile:
            mock_tif = Mock()
            mock_tif.asarray.return_value = np.random.randint(
                0, 4096, (512, 512), dtype=np.uint16
            )

            mock_page = Mock()
            mock_page.imagelength = 512
            mock_page.imagewidth = 512
            mock_page.bitspersample = 16
            mock_page.samplesperpixel = 1
            mock_page.tags = {}

            mock_tif.pages = [mock_page]
            mock_tif.imagej_metadata = {"spacing": 0.65, "unit": "micron"}

            mock_tifffile.TiffFile.return_value.__enter__.return_value = mock_tif

            with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
                tmp.write(b"fake tiff data")
                tmp.flush()

                raw_data, metadata = self.loader.load_raw_data(tmp.name)

                assert metadata.pixel_size == 0.65
                assert metadata.pixel_unit == "um"
                assert "imagej" in metadata.extra_metadata


class TestAstronomyLoader:
    """Test astronomy FITS loader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = AstronomyLoader(validate_on_load=False)

    def test_supported_extensions(self):
        """Test supported file extensions."""
        extensions = self.loader.get_supported_extensions()

        assert ".fits" in extensions
        assert ".fit" in extensions
        assert ".fts" in extensions

    def test_load_fits_data_mock(self):
        """Test FITS data loading with mocked astropy."""
        with patch.object(self.loader, "fits") as mock_fits:
            # Mock FITS data
            mock_data = np.random.randint(0, 65535, (1024, 1024), dtype=np.uint16)

            mock_header = {
                "NAXIS1": 1024,
                "NAXIS2": 1024,
                "BITPIX": 16,
                "EXPTIME": 300.0,
                "GAIN": 1.5,
                "CCD-TEMP": -20.0,
                "OBJECT": "M31",
                "TELESCOP": "Test Telescope",
                "CDELT1": -0.000111,  # -0.4 arcsec/pixel
                "CDELT2": 0.000111,
            }

            mock_hdu = Mock()
            mock_hdu.data = mock_data
            mock_hdu.header = mock_header

            mock_hdul = [mock_hdu]  # Make it a list for indexing

            mock_fits.open.return_value.__enter__.return_value = mock_hdul

            with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
                tmp.write(b"fake fits data")
                tmp.flush()

                raw_data, metadata = self.loader.load_raw_data(tmp.name)

                assert raw_data.shape == (1024, 1024)
                assert raw_data.dtype == np.float32
                assert metadata.domain == "astronomy"
                assert metadata.height == 1024
                assert metadata.width == 1024
                assert metadata.bit_depth == 16
                assert metadata.exposure_time == 300.0
                assert metadata.gain == 1.5
                assert metadata.temperature == -20.0
                assert abs(metadata.pixel_size - 0.4) < 0.01  # ~0.4 arcsec
                assert metadata.pixel_unit == "arcsec"

    def test_fits_scaling(self):
        """Test FITS BZERO/BSCALE scaling."""
        loader = AstronomyLoader(apply_scaling=True, validate_on_load=False)

        with patch.object(loader, "fits") as mock_fits:
            mock_data = np.array([[100, 200], [300, 400]], dtype=np.int16)

            mock_header = {
                "NAXIS1": 2,
                "NAXIS2": 2,
                "BITPIX": 16,
                "BZERO": 32768.0,
                "BSCALE": 1.0,
            }

            mock_hdu = Mock()
            mock_hdu.data = mock_data
            mock_hdu.header = mock_header

            mock_hdul = [mock_hdu]  # Make it a list for indexing

            mock_fits.open.return_value.__enter__.return_value = mock_hdul

            with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
                tmp.write(b"fake fits data")
                tmp.flush()

                raw_data, metadata = loader.load_raw_data(tmp.name)

                # Data should be scaled: data * BSCALE + BZERO
                expected = mock_data.astype(np.float32) * 1.0 + 32768.0
                np.testing.assert_array_equal(raw_data, expected)

    def test_wcs_extraction(self):
        """Test WCS coordinate system extraction."""
        with patch.object(self.loader, "fits") as mock_fits, patch.object(
            self.loader, "wcs"
        ) as mock_wcs:
            mock_data = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)

            mock_header = {
                "NAXIS1": 512,
                "NAXIS2": 512,
                "BITPIX": 16,
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 10.68458,
                "CRVAL2": 41.26917,
                "CRPIX1": 256.0,
                "CRPIX2": 256.0,
            }

            mock_hdu = Mock()
            mock_hdu.data = mock_data
            mock_hdu.header = mock_header

            mock_hdul = [mock_hdu]  # Make it a list for indexing

            mock_fits.open.return_value.__enter__.return_value = mock_hdul

            # Mock WCS
            mock_wcs_obj = Mock()
            mock_wcs_obj.has_celestial = True
            mock_wcs_obj.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            mock_wcs_obj.wcs.crval = [10.68458, 41.26917]
            mock_wcs_obj.wcs.crpix = [256.0, 256.0]

            mock_wcs.WCS.return_value = mock_wcs_obj

            with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
                tmp.write(b"fake fits data")
                tmp.flush()

                raw_data, metadata = self.loader.load_raw_data(tmp.name)

                assert "wcs" in metadata.extra_metadata
                assert metadata.extra_metadata["wcs"]["has_celestial"] is True


class TestFormatDetector:
    """Test automatic format detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = FormatDetector()

    def test_format_detection(self):
        """Test format detection from extensions."""
        assert self.detector.detect_format("image.arw") == "photography"
        assert self.detector.detect_format("image.tif") == "microscopy"
        assert self.detector.detect_format("image.fits") == "astronomy"

        # Test case insensitive
        assert self.detector.detect_format("IMAGE.ARW") == "photography"
        assert self.detector.detect_format("IMAGE.TIFF") == "microscopy"
        assert self.detector.detect_format("IMAGE.FIT") == "astronomy"

    def test_unsupported_format(self):
        """Test unsupported format handling."""
        with pytest.raises(DataError, match="Unsupported file format"):
            self.detector.detect_format("image.jpg")

    def test_supported_formats(self):
        """Test getting supported formats."""
        formats = self.detector.get_supported_formats()

        assert "photography" in formats
        assert "microscopy" in formats
        assert "astronomy" in formats

        assert ".arw" in formats["photography"]
        assert ".tif" in formats["microscopy"]
        assert ".fits" in formats["astronomy"]

    def test_auto_load(self):
        """Test automatic loading."""
        # Create a simple detector with mocked loaders
        detector = FormatDetector()

        # Mock the photography loader
        mock_loader = Mock()
        mock_loader.load_with_validation.return_value = (
            np.random.rand(100, 100),
            ImageMetadata(
                filepath="test.arw",
                format=".arw",
                domain="photography",
                height=100,
                width=100,
                channels=1,
                bit_depth=16,
                dtype="uint16",
            ),
        )
        detector.loaders["photography"] = mock_loader

        with tempfile.NamedTemporaryFile(suffix=".arw") as tmp:
            tmp.write(b"fake data")
            tmp.flush()

            data, metadata = detector.load_auto(tmp.name)

            assert data.shape == (100, 100)
            assert metadata.domain == "photography"


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("data.loaders.FormatDetector")
    def test_load_image(self, mock_detector_class):
        """Test load_image convenience function."""
        mock_detector = Mock()
        mock_detector.load_auto.return_value = (
            np.random.rand(100, 100),
            ImageMetadata(
                filepath="test.tif",
                format=".tif",
                domain="microscopy",
                height=100,
                width=100,
                channels=1,
                bit_depth=16,
                dtype="uint16",
            ),
        )
        mock_detector_class.return_value = mock_detector

        data, metadata = load_image("test.tif")

        assert data.shape == (100, 100)
        assert metadata.domain == "microscopy"

    @patch("data.loaders.FormatDetector")
    def test_get_image_metadata(self, mock_detector_class):
        """Test get_image_metadata convenience function."""
        mock_detector = Mock()
        mock_detector.get_metadata_auto.return_value = ImageMetadata(
            filepath="test.fits",
            format=".fits",
            domain="astronomy",
            height=1024,
            width=1024,
            channels=1,
            bit_depth=16,
            dtype="int16",
        )
        mock_detector_class.return_value = mock_detector

        metadata = get_image_metadata("test.fits")

        assert metadata.domain == "astronomy"
        assert metadata.height == 1024

    @patch("data.loaders.FormatDetector")
    def test_get_supported_formats_function(self, mock_detector_class):
        """Test get_supported_formats convenience function."""
        mock_detector = Mock()
        mock_detector.get_supported_formats.return_value = {
            "photography": [".arw", ".dng"],
            "microscopy": [".tif", ".tiff"],
            "astronomy": [".fits", ".fit"],
        }
        mock_detector_class.return_value = mock_detector

        formats = get_supported_formats()

        assert "photography" in formats
        assert ".arw" in formats["photography"]


class TestErrorHandling:
    """Test error handling in loaders."""

    def test_validation_errors(self):
        """Test data validation errors."""
        loader = PhotographyLoader(validate_on_load=True)

        # Test with invalid data
        with patch.object(loader, "load_raw_data") as mock_load:
            # Return data with NaN values
            mock_load.return_value = (
                np.array([[np.nan, 1], [2, 3]]),
                ImageMetadata(
                    filepath="test.arw",
                    format=".arw",
                    domain="photography",
                    height=2,
                    width=2,
                    channels=1,
                    bit_depth=16,
                    dtype="uint16",
                ),
            )

            with pytest.raises(DataError):
                loader.load_with_validation("test.arw")

    def test_shape_mismatch_validation(self):
        """Test shape mismatch validation."""
        loader = MicroscopyLoader(validate_on_load=True)

        with patch.object(loader, "load_raw_data") as mock_load:
            # Return data with mismatched shape
            mock_load.return_value = (
                np.random.rand(100, 200),  # 100x200 data
                ImageMetadata(
                    filepath="test.tif",
                    format=".tif",
                    domain="microscopy",
                    height=512,  # Claims 512x512
                    width=512,
                    channels=1,
                    bit_depth=16,
                    dtype="uint16",
                ),
            )

            with pytest.raises(DataError):
                loader.load_with_validation("test.tif")

    def test_file_not_found(self):
        """Test file not found handling."""
        loader = AstronomyLoader(validate_on_load=True)

        with pytest.raises(DataError):
            loader.load_with_validation("/nonexistent/file.fits")

    def test_error_statistics(self):
        """Test error statistics tracking."""
        loader = PhotographyLoader(validate_on_load=False)

        # Mock a failing load
        with patch.object(loader, "load_raw_data", side_effect=Exception("Test error")):
            with pytest.raises(DataError):
                loader.load_with_validation("test.arw")

        stats = loader.get_statistics()
        assert stats["load_errors"] == 1
        assert stats["success_rate"] == 0.0


class TestIntegration:
    """Integration tests with real file operations."""

    def test_create_synthetic_files(self):
        """Test creating and loading synthetic files."""
        # Create synthetic TIFF
        data = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            try:
                from PIL import Image

                img = Image.fromarray(data)
                img.save(tmp.name)

                # Test loading
                loader = MicroscopyLoader(validate_on_load=False)

                # Mock the tifffile loading since we created with PIL
                with patch.object(loader, "tifffile") as mock_tifffile:
                    mock_tif = Mock()
                    mock_tif.asarray.return_value = data

                    mock_page = Mock()
                    mock_page.imagelength = 256
                    mock_page.imagewidth = 256
                    mock_page.bitspersample = 16
                    mock_page.samplesperpixel = 1
                    mock_page.tags = {}

                    mock_tif.pages = [mock_page]
                    mock_tif.imagej_metadata = None

                    mock_tifffile.TiffFile.return_value.__enter__.return_value = (
                        mock_tif
                    )

                    loaded_data, metadata = loader.load_raw_data(tmp.name)

                    assert loaded_data.shape == (256, 256)
                    assert metadata.domain == "microscopy"

            except ImportError:
                pytest.skip("PIL not available for integration test")
            finally:
                Path(tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
