"""
Format-specific data loaders for Domain-Adaptive Poisson-Gaussian Diffusion.

This module implements robust data loading for different imaging domains:
- Photography: RAW formats (.arw, .dng, .nef, .cr2) with Bayer handling
- Microscopy: TIFF formats (.tif, .tiff) with multi-channel support
- Astronomy: FITS format (.fits, .fit) with header metadata extraction

Key features:
- Automatic format detection and validation
- Comprehensive error handling with graceful fallbacks
- Integration with calibration system
- Metadata extraction and preservation
- Memory-efficient loading for large files

Requirements addressed: 5.1-5.4 from requirements.md
Task: 4.1 from tasks.md
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from core.calibration import CalibrationParams, SensorCalibration
from core.error_handlers import ErrorHandler, safe_operation
from core.exceptions import CalibrationError, DataError, ValidationError
from core.interfaces import Dataset
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ImageMetadata:
    """
    Comprehensive metadata for loaded images.

    This contains all information needed for proper processing
    and integration with the calibration system.
    """

    # File information
    filepath: str
    format: str
    domain: str

    # Image properties
    height: int
    width: int
    channels: int
    bit_depth: int
    dtype: str

    # Physical properties
    pixel_size: Optional[float] = None
    pixel_unit: Optional[str] = None

    # Acquisition parameters
    exposure_time: Optional[float] = None
    iso: Optional[int] = None
    gain: Optional[float] = None
    temperature: Optional[float] = None

    # Sensor properties
    black_level: Optional[float] = None
    white_level: Optional[float] = None

    # Domain-specific metadata
    extra_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


class BaseLoader(ABC):
    """
    Abstract base class for format-specific loaders.

    Provides common functionality and interface for all loaders.
    """

    def __init__(
        self,
        error_handler: Optional[ErrorHandler] = None,
        validate_on_load: bool = True,
        memory_map: bool = False,
    ):
        """
        Initialize base loader.

        Args:
            error_handler: Custom error handler
            validate_on_load: Whether to validate data on loading
            memory_map: Whether to use memory mapping for large files
        """
        self.error_handler = error_handler or ErrorHandler()
        self.validate_on_load = validate_on_load
        self.memory_map = memory_map

        # Statistics
        self.files_loaded = 0
        self.load_errors = 0

    @abstractmethod
    def load_raw_data(
        self, filepath: Union[str, Path]
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Load raw data from file.

        Args:
            filepath: Path to image file

        Returns:
            (raw_data, metadata) tuple
        """
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        pass

    @abstractmethod
    def extract_metadata(self, filepath: Union[str, Path]) -> ImageMetadata:
        """Extract metadata from file without loading full data."""
        pass

    def validate_file(self, filepath: Union[str, Path]) -> bool:
        """
        Validate that file can be loaded.

        Args:
            filepath: Path to validate

        Returns:
            True if file is valid
        """
        filepath = Path(filepath)

        # Check existence
        if not filepath.exists():
            logger.warning(f"File does not exist: {filepath}")
            return False

        # Check extension
        if filepath.suffix.lower() not in self.get_supported_extensions():
            logger.warning(f"Unsupported extension: {filepath.suffix}")
            return False

        # Check file size
        if filepath.stat().st_size == 0:
            logger.warning(f"Empty file: {filepath}")
            return False

        return True

    def load_with_validation(
        self, filepath: Union[str, Path]
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Load data with comprehensive validation.

        Args:
            filepath: Path to image file

        Returns:
            (raw_data, metadata) tuple

        Raises:
            DataError: If loading fails
        """
        filepath = Path(filepath)

        # Validate file
        if self.validate_on_load and not self.validate_file(filepath):
            raise DataError(f"File validation failed: {filepath}")

        try:
            # Load data
            raw_data, metadata = self.load_raw_data(filepath)

            # Validate loaded data
            if self.validate_on_load:
                self._validate_loaded_data(raw_data, metadata)

            self.files_loaded += 1
            logger.debug(f"Successfully loaded: {filepath}")

            return raw_data, metadata

        except Exception as e:
            self.load_errors += 1
            error_msg = f"Failed to load {filepath}: {str(e)}"
            logger.error(error_msg)
            raise DataError(error_msg) from e

    def _validate_loaded_data(self, data: np.ndarray, metadata: ImageMetadata) -> None:
        """Validate loaded data consistency."""
        # Check data shape consistency
        if len(data.shape) < 2:
            raise ValidationError("Image data must be at least 2D")

        # Check metadata consistency
        if data.shape[0] != metadata.height or data.shape[1] != metadata.width:
            raise ValidationError(
                f"Data shape {data.shape} doesn't match metadata "
                f"({metadata.height}, {metadata.width})"
            )

        # Check for reasonable data range
        if data.min() < 0:
            logger.warning("Negative values in image data")

        if np.isnan(data).any():
            raise ValidationError("NaN values found in image data")

        if np.isinf(data).any():
            raise ValidationError("Infinite values found in image data")

    def get_statistics(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            "files_loaded": self.files_loaded,
            "load_errors": self.load_errors,
            "success_rate": (
                self.files_loaded / (self.files_loaded + self.load_errors)
                if (self.files_loaded + self.load_errors) > 0
                else 0.0
            ),
        }


class PhotographyLoader(BaseLoader):
    """
    Loader for photography RAW formats.

    Supports:
    - Sony ARW files
    - Adobe DNG files
    - Nikon NEF files
    - Canon CR2 files

    Features:
    - Bayer pattern handling (keep raw or demosaic)
    - EXIF metadata extraction
    - Automatic exposure parameter detection
    """

    def __init__(self, demosaic: bool = False, use_camera_wb: bool = False, **kwargs):
        """
        Initialize photography loader.

        Args:
            demosaic: Whether to demosaic Bayer data
            use_camera_wb: Whether to apply camera white balance
            **kwargs: Arguments for BaseLoader
        """
        super().__init__(**kwargs)
        self.demosaic = demosaic
        self.use_camera_wb = use_camera_wb

        # Check rawpy availability
        try:
            import rawpy

            self.rawpy = rawpy
        except ImportError:
            raise ImportError(
                "rawpy is required for photography loading. "
                "Install with: pip install rawpy"
            )

    def get_supported_extensions(self) -> List[str]:
        """Get supported RAW extensions."""
        return [".arw", ".dng", ".nef", ".cr2", ".raw"]

    @safe_operation("RAW data loading")
    def load_raw_data(
        self, filepath: Union[str, Path]
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Load RAW photography data.

        Args:
            filepath: Path to RAW file

        Returns:
            (raw_data, metadata) tuple
        """
        filepath = Path(filepath)

        with self.rawpy.imread(str(filepath)) as raw:
            # Extract basic metadata
            metadata = self._extract_raw_metadata(raw, filepath)

            if self.demosaic:
                # Demosaic to RGB
                rgb = raw.postprocess(
                    use_camera_wb=self.use_camera_wb,
                    half_size=False,
                    no_auto_bright=True,
                    output_bps=16,
                )
                # Convert to HWC format
                raw_data = rgb.astype(np.float32)
                metadata.channels = 3

            else:
                # Keep raw Bayer data
                raw_data = raw.raw_image.copy().astype(np.float32)
                metadata.channels = 1

                # Add Bayer pattern info to metadata
                if metadata.extra_metadata is None:
                    metadata.extra_metadata = {}
                metadata.extra_metadata["bayer_pattern"] = self._get_bayer_pattern(raw)
                metadata.extra_metadata["raw_colors"] = raw.raw_colors.copy()

        return raw_data, metadata

    def extract_metadata(self, filepath: Union[str, Path]) -> ImageMetadata:
        """Extract metadata without loading full image."""
        filepath = Path(filepath)

        with self.rawpy.imread(str(filepath)) as raw:
            return self._extract_raw_metadata(raw, filepath)

    def _extract_raw_metadata(self, raw, filepath: Path) -> ImageMetadata:
        """Extract comprehensive metadata from RAW file."""
        # Basic image properties
        height, width = raw.raw_image.shape

        # Determine bit depth
        bit_depth = 16  # Most RAW files are 16-bit
        if hasattr(raw, "raw_image_dtype"):
            if raw.raw_image_dtype == np.uint8:
                bit_depth = 8
            elif raw.raw_image_dtype == np.uint16:
                bit_depth = 16

        # Extract EXIF data
        iso = None
        exposure_time = None

        try:
            # Try to get ISO
            if hasattr(raw, "other") and "ISOSpeedRatings" in raw.other:
                iso = raw.other["ISOSpeedRatings"]
            elif hasattr(raw, "other") and "ISO" in raw.other:
                iso = raw.other["ISO"]
        except:
            logger.debug("Could not extract ISO from RAW file")

        try:
            # Try to get exposure time
            if hasattr(raw, "other") and "ExposureTime" in raw.other:
                exposure_time = float(raw.other["ExposureTime"])
        except:
            logger.debug("Could not extract exposure time from RAW file")

        # Camera-specific calibration estimates
        black_level = None
        white_level = None

        try:
            if hasattr(raw, "black_level_per_channel"):
                black_level = float(np.mean(raw.black_level_per_channel))
            if hasattr(raw, "white_level"):
                white_level = float(raw.white_level)
        except:
            logger.debug("Could not extract calibration levels from RAW file")

        return ImageMetadata(
            filepath=str(filepath),
            format=filepath.suffix.lower(),
            domain="photography",
            height=height,
            width=width,
            channels=1,  # Will be updated if demosaiced
            bit_depth=bit_depth,
            dtype=str(raw.raw_image.dtype),
            pixel_size=None,  # Would need camera database
            pixel_unit="pixel",
            exposure_time=exposure_time,
            iso=iso,
            black_level=black_level,
            white_level=white_level,
            extra_metadata={
                "camera_make": getattr(raw, "camera_make", "unknown"),
                "camera_model": getattr(raw, "camera_model", "unknown"),
            },
        )

    def _get_bayer_pattern(self, raw) -> str:
        """Extract Bayer pattern string."""
        try:
            # rawpy provides color description
            if hasattr(raw, "color_desc"):
                return raw.color_desc.decode("utf-8")
            else:
                return "RGGB"  # Default assumption
        except:
            return "RGGB"


class MicroscopyLoader(BaseLoader):
    """
    Loader for microscopy TIFF formats.

    Supports:
    - Single and multi-channel TIFF files
    - 8, 12, 16-bit data with proper scaling
    - ImageJ/Fiji metadata extraction
    - Multi-page TIFF stacks

    Features:
    - Automatic bit depth detection and scaling
    - Channel separation and naming
    - Physical pixel size extraction
    - Acquisition parameter extraction
    """

    def __init__(
        self, channel_axis: int = 0, normalize_channels: bool = True, **kwargs
    ):
        """
        Initialize microscopy loader.

        Args:
            channel_axis: Axis for channel dimension (0 or -1)
            normalize_channels: Whether to normalize channels independently
            **kwargs: Arguments for BaseLoader
        """
        super().__init__(**kwargs)
        self.channel_axis = channel_axis
        self.normalize_channels = normalize_channels

        # Check dependencies
        try:
            import tifffile
            from PIL import Image

            self.Image = Image
            self.tifffile = tifffile
        except ImportError:
            raise ImportError(
                "PIL and tifffile are required for microscopy loading. "
                "Install with: pip install Pillow tifffile"
            )

    def get_supported_extensions(self) -> List[str]:
        """Get supported TIFF extensions."""
        return [".tif", ".tiff"]

    @safe_operation("TIFF data loading")
    def load_raw_data(
        self, filepath: Union[str, Path]
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Load TIFF microscopy data.

        Args:
            filepath: Path to TIFF file

        Returns:
            (raw_data, metadata) tuple
        """
        filepath = Path(filepath)

        # Try tifffile first (better for scientific TIFF)
        try:
            with self.tifffile.TiffFile(str(filepath)) as tif:
                # Load image data
                raw_data = tif.asarray()

                # Extract metadata
                metadata = self._extract_tiff_metadata(tif, filepath)

        except Exception as e:
            logger.warning(f"tifffile failed, trying PIL: {e}")

            # Fallback to PIL
            with self.Image.open(filepath) as img:
                raw_data = np.array(img)
                metadata = self._extract_pil_metadata(img, filepath)

        # Ensure float32 for processing
        raw_data = raw_data.astype(np.float32)

        # Handle multi-channel data
        if raw_data.ndim > 2:
            if self.channel_axis == -1:
                # Move channels to first axis
                raw_data = np.moveaxis(raw_data, -1, 0)
            metadata.channels = raw_data.shape[0] if raw_data.ndim == 3 else 1
        else:
            metadata.channels = 1

        return raw_data, metadata

    def extract_metadata(self, filepath: Union[str, Path]) -> ImageMetadata:
        """Extract metadata without loading full image."""
        filepath = Path(filepath)

        try:
            with self.tifffile.TiffFile(str(filepath)) as tif:
                return self._extract_tiff_metadata(tif, filepath, load_data=False)
        except:
            with self.Image.open(filepath) as img:
                return self._extract_pil_metadata(img, filepath)

    def _extract_tiff_metadata(
        self, tif, filepath: Path, load_data: bool = True
    ) -> ImageMetadata:
        """Extract metadata from tifffile object."""
        page = tif.pages[0]

        # Basic properties
        height = page.imagelength
        width = page.imagewidth
        bit_depth = page.bitspersample

        # Determine channels
        channels = 1
        if hasattr(page, "samplesperpixel"):
            channels = page.samplesperpixel
        elif load_data:
            data_shape = tif.asarray().shape
            if len(data_shape) > 2:
                channels = data_shape[0] if self.channel_axis == 0 else data_shape[-1]

        # Physical pixel size
        pixel_size = None
        pixel_unit = "pixel"

        try:
            # Try to get resolution info
            if hasattr(page, "tags") and "XResolution" in page.tags:
                x_res = page.tags["XResolution"].value
                if isinstance(x_res, (list, tuple)):
                    x_res = x_res[0] / x_res[1]

                # Convert to micrometers (assuming resolution in DPI)
                if hasattr(page, "tags") and "ResolutionUnit" in page.tags:
                    unit = page.tags["ResolutionUnit"].value
                    if unit == 2:  # Inches
                        pixel_size = 25400.0 / x_res  # μm per pixel
                        pixel_unit = "um"
                    elif unit == 3:  # Centimeters
                        pixel_size = 10000.0 / x_res  # μm per pixel
                        pixel_unit = "um"
        except:
            logger.debug("Could not extract pixel size from TIFF")

        # ImageJ metadata
        extra_metadata = {}
        try:
            if hasattr(tif, "imagej_metadata") and tif.imagej_metadata:
                extra_metadata["imagej"] = tif.imagej_metadata

                # Try to extract pixel size from ImageJ metadata
                if "spacing" in tif.imagej_metadata:
                    pixel_size = tif.imagej_metadata["spacing"]
                    pixel_unit = "um"
        except:
            logger.debug("Could not extract ImageJ metadata")

        return ImageMetadata(
            filepath=str(filepath),
            format=filepath.suffix.lower(),
            domain="microscopy",
            height=height,
            width=width,
            channels=channels,
            bit_depth=bit_depth,
            dtype=f"uint{bit_depth}",
            pixel_size=pixel_size,
            pixel_unit=pixel_unit,
            extra_metadata=extra_metadata if extra_metadata else None,
        )

    def _extract_pil_metadata(self, img, filepath: Path) -> ImageMetadata:
        """Extract metadata from PIL Image object."""
        width, height = img.size

        # Determine bit depth
        bit_depth = 8
        if img.mode == "I;16":
            bit_depth = 16
        elif img.mode == "L":
            bit_depth = 8
        elif img.mode == "RGB":
            bit_depth = 8

        # Channels
        channels = len(img.getbands()) if hasattr(img, "getbands") else 1

        return ImageMetadata(
            filepath=str(filepath),
            format=filepath.suffix.lower(),
            domain="microscopy",
            height=height,
            width=width,
            channels=channels,
            bit_depth=bit_depth,
            dtype=f"uint{bit_depth}",
            pixel_size=None,
            pixel_unit="pixel",
        )


class AstronomyLoader(BaseLoader):
    """
    Loader for astronomy FITS formats.

    Supports:
    - Single and multi-extension FITS files
    - 16, 32-bit integer and floating-point data
    - Comprehensive header metadata extraction
    - World Coordinate System (WCS) information

    Features:
    - Automatic data scaling and offset handling
    - Header keyword extraction
    - Physical coordinate system support
    - Multi-extension file handling
    """

    def __init__(self, extension: int = 0, apply_scaling: bool = True, **kwargs):
        """
        Initialize astronomy loader.

        Args:
            extension: FITS extension to load (0 = primary)
            apply_scaling: Whether to apply BZERO/BSCALE scaling
            **kwargs: Arguments for BaseLoader
        """
        super().__init__(**kwargs)
        self.extension = extension
        self.apply_scaling = apply_scaling

        # Check astropy availability
        try:
            from astropy import wcs
            from astropy.io import fits

            self.fits = fits
            self.wcs = wcs
        except ImportError:
            raise ImportError(
                "astropy is required for astronomy loading. "
                "Install with: pip install astropy"
            )

    def get_supported_extensions(self) -> List[str]:
        """Get supported FITS extensions."""
        return [".fits", ".fit", ".fts"]

    @safe_operation("FITS data loading")
    def load_raw_data(
        self, filepath: Union[str, Path]
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Load FITS astronomy data.

        Args:
            filepath: Path to FITS file

        Returns:
            (raw_data, metadata) tuple
        """
        filepath = Path(filepath)

        with self.fits.open(str(filepath)) as hdul:
            # Get the specified extension
            hdu = hdul[self.extension]

            # Load data
            raw_data = hdu.data.copy()

            # Apply scaling if requested
            if self.apply_scaling:
                if "BZERO" in hdu.header or "BSCALE" in hdu.header:
                    bzero = hdu.header.get("BZERO", 0.0)
                    bscale = hdu.header.get("BSCALE", 1.0)
                    raw_data = raw_data * bscale + bzero

            # Ensure 2D (some FITS files have extra dimensions)
            if raw_data.ndim > 2:
                # Take first slice of higher dimensions
                while raw_data.ndim > 2:
                    raw_data = raw_data[0]

            # Convert to float32
            raw_data = raw_data.astype(np.float32)

            # Extract metadata
            metadata = self._extract_fits_metadata(hdu, filepath)

        return raw_data, metadata

    def extract_metadata(self, filepath: Union[str, Path]) -> ImageMetadata:
        """Extract metadata without loading full image."""
        filepath = Path(filepath)

        with self.fits.open(str(filepath)) as hdul:
            hdu = hdul[self.extension]
            return self._extract_fits_metadata(hdu, filepath)

    def _extract_fits_metadata(self, hdu, filepath: Path) -> ImageMetadata:
        """Extract comprehensive metadata from FITS HDU."""
        header = hdu.header

        # Basic image properties
        if hdu.data is not None:
            if hdu.data.ndim >= 2:
                height, width = hdu.data.shape[-2:]
            else:
                height = width = 1
        else:
            # Get from header
            width = header.get("NAXIS1", 0)
            height = header.get("NAXIS2", 0)

        # Bit depth and data type
        bitpix = header.get("BITPIX", -32)
        if bitpix == 8:
            bit_depth = 8
            dtype = "uint8"
        elif bitpix == 16:
            bit_depth = 16
            dtype = "int16"
        elif bitpix == 32:
            bit_depth = 32
            dtype = "int32"
        elif bitpix == -32:
            bit_depth = 32
            dtype = "float32"
        elif bitpix == -64:
            bit_depth = 64
            dtype = "float64"
        else:
            bit_depth = 32
            dtype = "float32"

        # Physical pixel scale
        pixel_size = None
        pixel_unit = "pixel"

        # Try to get pixel scale from header
        try:
            # Common keywords for pixel scale
            if "CDELT1" in header:
                pixel_size = abs(float(header["CDELT1"]) * 3600)  # Convert to arcsec
                pixel_unit = "arcsec"
            elif "CD1_1" in header:
                pixel_size = abs(float(header["CD1_1"]) * 3600)  # Convert to arcsec
                pixel_unit = "arcsec"
            elif "PIXSCALE" in header:
                pixel_size = float(header["PIXSCALE"])
                pixel_unit = "arcsec"
        except:
            logger.debug("Could not extract pixel scale from FITS header")

        # Exposure time
        exposure_time = None
        for keyword in ["EXPTIME", "EXPOSURE", "TEXP"]:
            if keyword in header:
                try:
                    exposure_time = float(header[keyword])
                    break
                except:
                    continue

        # Temperature
        temperature = None
        for keyword in ["CCD-TEMP", "TEMP", "TEMPERATURE"]:
            if keyword in header:
                try:
                    temperature = float(header[keyword])
                    break
                except:
                    continue

        # Gain
        gain = None
        for keyword in ["GAIN", "EGAIN"]:
            if keyword in header:
                try:
                    gain = float(header[keyword])
                    break
                except:
                    continue

        # Extract important header keywords
        extra_metadata = {}
        important_keywords = [
            "OBJECT",
            "OBSERVER",
            "TELESCOP",
            "INSTRUME",
            "FILTER",
            "DATE-OBS",
            "RA",
            "DEC",
            "AIRMASS",
            "SEEING",
        ]

        for keyword in important_keywords:
            if keyword in header:
                extra_metadata[keyword.lower()] = header[keyword]

        # WCS information
        try:
            w = self.wcs.WCS(header)
            if w.has_celestial:
                extra_metadata["wcs"] = {
                    "has_celestial": True,
                    "ctype": [w.wcs.ctype[0], w.wcs.ctype[1]],
                    "crval": [w.wcs.crval[0], w.wcs.crval[1]],
                    "crpix": [w.wcs.crpix[0], w.wcs.crpix[1]],
                }
        except:
            logger.debug("Could not extract WCS information")

        return ImageMetadata(
            filepath=str(filepath),
            format=filepath.suffix.lower(),
            domain="astronomy",
            height=height,
            width=width,
            channels=1,
            bit_depth=bit_depth,
            dtype=dtype,
            pixel_size=pixel_size,
            pixel_unit=pixel_unit,
            exposure_time=exposure_time,
            gain=gain,
            temperature=temperature,
            extra_metadata=extra_metadata if extra_metadata else None,
        )


class FormatDetector:
    """
    Automatic format detection and loader selection.

    Provides unified interface for loading any supported format
    with automatic format detection and appropriate loader selection.
    """

    def __init__(self):
        """Initialize format detector with all loaders."""
        self.loaders = {
            "photography": PhotographyLoader(),
            "microscopy": MicroscopyLoader(),
            "astronomy": AstronomyLoader(),
        }

        # Extension to domain mapping
        self.extension_map = {}
        for domain, loader in self.loaders.items():
            for ext in loader.get_supported_extensions():
                self.extension_map[ext] = domain

    def detect_format(self, filepath: Union[str, Path]) -> str:
        """
        Detect format from file extension.

        Args:
            filepath: Path to file

        Returns:
            Domain name ('photography', 'microscopy', 'astronomy')

        Raises:
            DataError: If format not supported
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        if ext not in self.extension_map:
            raise DataError(f"Unsupported file format: {ext}")

        return self.extension_map[ext]

    def load_auto(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Automatically detect format and load data.

        Args:
            filepath: Path to image file

        Returns:
            (raw_data, metadata) tuple
        """
        domain = self.detect_format(filepath)
        loader = self.loaders[domain]
        return loader.load_with_validation(filepath)

    def get_metadata_auto(self, filepath: Union[str, Path]) -> ImageMetadata:
        """
        Automatically detect format and extract metadata.

        Args:
            filepath: Path to image file

        Returns:
            Image metadata
        """
        domain = self.detect_format(filepath)
        loader = self.loaders[domain]
        return loader.extract_metadata(filepath)

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get all supported formats by domain."""
        return {
            domain: loader.get_supported_extensions()
            for domain, loader in self.loaders.items()
        }


# Convenience functions for direct use
def load_image(filepath: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
    """
    Load image with automatic format detection.

    Args:
        filepath: Path to image file

    Returns:
        (raw_data, metadata) tuple
    """
    detector = FormatDetector()
    return detector.load_auto(filepath)


def get_image_metadata(filepath: Union[str, Path]) -> ImageMetadata:
    """
    Get image metadata with automatic format detection.

    Args:
        filepath: Path to image file

    Returns:
        Image metadata
    """
    detector = FormatDetector()
    return detector.get_metadata_auto(filepath)


def get_supported_formats() -> Dict[str, List[str]]:
    """Get all supported file formats."""
    detector = FormatDetector()
    return detector.get_supported_formats()
