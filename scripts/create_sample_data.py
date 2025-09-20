#!/usr/bin/env python
"""
Create sample data files for testing format loaders.

This script generates synthetic data files in all supported formats
to test the data loading pipeline without requiring real data.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_tiff(
    output_path: Path,
    shape: tuple = (512, 512),
    bit_depth: int = 16,
    channels: int = 1,
    pixel_size: float = 0.65,
    add_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Create sample TIFF file for microscopy testing.

    Args:
        output_path: Output file path
        shape: Image shape (height, width)
        bit_depth: Bit depth (8, 16)
        channels: Number of channels
        pixel_size: Pixel size in micrometers
        add_metadata: Whether to add ImageJ metadata

    Returns:
        Metadata dictionary
    """
    try:
        import tifffile
        from PIL import Image
    except ImportError:
        logger.error("PIL and tifffile required for TIFF creation")
        return {}

    # Generate synthetic data
    if channels == 1:
        if bit_depth == 8:
            data = np.random.randint(0, 256, shape, dtype=np.uint8)
        else:
            data = np.random.randint(0, 4096, shape, dtype=np.uint16)
    else:
        if bit_depth == 8:
            data = np.random.randint(0, 256, (channels, *shape), dtype=np.uint8)
        else:
            data = np.random.randint(0, 4096, (channels, *shape), dtype=np.uint16)

    # Add some structure to make it more realistic
    y, x = np.ogrid[: shape[0], : shape[1]]
    center_y, center_x = shape[0] // 2, shape[1] // 2

    # Add circular structures
    for i in range(5):
        cy = np.random.randint(shape[0] // 4, 3 * shape[0] // 4)
        cx = np.random.randint(shape[1] // 4, 3 * shape[1] // 4)
        radius = np.random.randint(20, 50)
        intensity = np.random.randint(1000, 3000)

        mask = (y - cy) ** 2 + (x - cx) ** 2 < radius**2
        if channels == 1:
            data[mask] = np.minimum(data[mask] + intensity, 2 ** (bit_depth) - 1)
        else:
            for c in range(channels):
                data[c][mask] = np.minimum(
                    data[c][mask] + intensity, 2 ** (bit_depth) - 1
                )

    # Create metadata
    metadata = {
        "spacing": pixel_size,
        "unit": "micron",
        "channels": channels,
        "slices": 1,
        "frames": 1,
    }

    # Save with tifffile for better metadata support
    if add_metadata:
        tifffile.imwrite(
            str(output_path),
            data,
            imagej=True,
            resolution=(1.0 / pixel_size, 1.0 / pixel_size),
            metadata=metadata,
        )
    else:
        # Use PIL for simpler TIFF
        if channels == 1:
            img = Image.fromarray(data)
        else:
            img = Image.fromarray(data[0])  # Take first channel
        img.save(str(output_path))

    logger.info(f"Created TIFF: {output_path}")

    return {
        "filepath": str(output_path),
        "format": ".tif",
        "domain": "microscopy",
        "height": shape[0],
        "width": shape[1],
        "channels": channels,
        "bit_depth": bit_depth,
        "pixel_size": pixel_size,
        "pixel_unit": "um",
    }


def create_sample_fits(
    output_path: Path,
    shape: tuple = (1024, 1024),
    bit_depth: int = 16,
    pixel_scale: float = 0.4,
    add_wcs: bool = True,
    exposure_time: float = 300.0,
) -> Dict[str, Any]:
    """
    Create sample FITS file for astronomy testing.

    Args:
        output_path: Output file path
        shape: Image shape (height, width)
        bit_depth: Bit depth (16, 32, -32)
        pixel_scale: Pixel scale in arcseconds
        add_wcs: Whether to add WCS information
        exposure_time: Exposure time in seconds

    Returns:
        Metadata dictionary
    """
    try:
        from astropy import wcs
        from astropy.io import fits
    except ImportError:
        logger.error("astropy required for FITS creation")
        return {}

    # Generate synthetic astronomical data
    if bit_depth == 16:
        data = np.random.randint(0, 65535, shape, dtype=np.uint16)
        bitpix = 16
    elif bit_depth == 32:
        data = np.random.randint(0, 2**31 - 1, shape, dtype=np.int32)
        bitpix = 32
    else:  # -32 (float32)
        data = np.random.rand(*shape).astype(np.float32) * 65535
        bitpix = -32

    # Add some astronomical-like features
    y, x = np.ogrid[: shape[0], : shape[1]]

    # Add point sources (stars)
    for i in range(20):
        sy = np.random.randint(50, shape[0] - 50)
        sx = np.random.randint(50, shape[1] - 50)
        intensity = np.random.randint(5000, 20000)
        sigma = np.random.uniform(1.5, 3.0)

        # Gaussian PSF
        psf = intensity * np.exp(-((y - sy) ** 2 + (x - sx) ** 2) / (2 * sigma**2))
        data = data + psf.astype(data.dtype)

    # Add background gradient
    bg_gradient = 1000 * (x / shape[1] + y / shape[0])
    data = data + bg_gradient.astype(data.dtype)

    # Clip to valid range
    if bit_depth > 0:
        data = np.clip(data, 0, 2**bit_depth - 1)

    # Create FITS header
    header = fits.Header()
    header["BITPIX"] = bitpix
    header["NAXIS"] = 2
    header["NAXIS1"] = shape[1]
    header["NAXIS2"] = shape[0]
    header["EXPTIME"] = exposure_time
    header["GAIN"] = 1.5
    header["CCD-TEMP"] = -20.0
    header["OBJECT"] = "Test Field"
    header["TELESCOP"] = "Test Telescope"
    header["INSTRUME"] = "Test Camera"
    header["FILTER"] = "V"
    header["DATE-OBS"] = "2024-01-01T00:00:00"
    header["RA"] = 10.68458
    header["DEC"] = 41.26917
    header["AIRMASS"] = 1.2

    # Add WCS information
    if add_wcs:
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRVAL1"] = 10.68458  # RA in degrees
        header["CRVAL2"] = 41.26917  # Dec in degrees
        header["CRPIX1"] = shape[1] / 2  # Reference pixel
        header["CRPIX2"] = shape[0] / 2
        header["CDELT1"] = -pixel_scale / 3600  # Pixel scale in degrees
        header["CDELT2"] = pixel_scale / 3600
        header["PIXSCALE"] = pixel_scale

    # Create HDU and save
    hdu = fits.PrimaryHDU(data, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(str(output_path), overwrite=True)

    logger.info(f"Created FITS: {output_path}")

    return {
        "filepath": str(output_path),
        "format": ".fits",
        "domain": "astronomy",
        "height": shape[0],
        "width": shape[1],
        "channels": 1,
        "bit_depth": abs(bit_depth),
        "pixel_size": pixel_scale,
        "pixel_unit": "arcsec",
        "exposure_time": exposure_time,
    }


def create_sample_raw_metadata(
    output_path: Path,
    shape: tuple = (3000, 4000),
    iso: int = 3200,
    exposure_time: float = 0.1,
    camera_make: str = "Test",
    camera_model: str = "Camera",
) -> Dict[str, Any]:
    """
    Create metadata for a simulated RAW file.

    Since we can't easily create real RAW files, this creates
    the metadata that would be extracted from one.
    """
    metadata = {
        "filepath": str(output_path),
        "format": ".arw",
        "domain": "photography",
        "height": shape[0],
        "width": shape[1],
        "channels": 1,
        "bit_depth": 16,
        "pixel_size": 4.29,  # Sony A7S pixel size
        "pixel_unit": "um",
        "iso": iso,
        "exposure_time": exposure_time,
        "black_level": 512,
        "white_level": 16383,
        "camera_make": camera_make,
        "camera_model": camera_model,
        "bayer_pattern": "RGGB",
    }

    # Save metadata as JSON (since we can't create real RAW)
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created RAW metadata: {metadata_path}")
    return metadata


def create_calibration_files(output_dir: Path) -> Dict[str, str]:
    """
    Create sample calibration files for each domain.

    Args:
        output_dir: Directory to save calibration files

    Returns:
        Dictionary mapping domain to calibration file path
    """
    calibration_files = {}

    # Photography calibration
    photo_cal = {
        "gain": 1.0,
        "black_level": 512.0,
        "white_level": 16383.0,
        "read_noise": 5.0,
        "dark_current": 0.1,
        "quantum_efficiency": 0.8,
        "pixel_size": 4.29,
        "pixel_unit": "um",
        "sensor_name": "Sony IMX410",
        "bit_depth": 16,
        "domain": "photography",
    }

    photo_path = output_dir / "photography_calibration.json"
    with open(photo_path, "w") as f:
        json.dump(photo_cal, f, indent=2)
    calibration_files["photography"] = str(photo_path)

    # Microscopy calibration
    micro_cal = {
        "gain": 2.0,
        "black_level": 100.0,
        "white_level": 65535.0,
        "read_noise": 3.0,
        "dark_current": 0.05,
        "quantum_efficiency": 0.9,
        "pixel_size": 0.65,
        "pixel_unit": "um",
        "sensor_name": "Scientific CMOS",
        "bit_depth": 16,
        "domain": "microscopy",
    }

    micro_path = output_dir / "microscopy_calibration.json"
    with open(micro_path, "w") as f:
        json.dump(micro_cal, f, indent=2)
    calibration_files["microscopy"] = str(micro_path)

    # Astronomy calibration
    astro_cal = {
        "gain": 1.5,
        "black_level": 0.0,
        "white_level": 65535.0,
        "read_noise": 2.0,
        "dark_current": 0.02,
        "quantum_efficiency": 0.95,
        "pixel_size": 0.04,
        "pixel_unit": "arcsec",
        "sensor_name": "CCD Camera",
        "bit_depth": 16,
        "domain": "astronomy",
    }

    astro_path = output_dir / "astronomy_calibration.json"
    with open(astro_path, "w") as f:
        json.dump(astro_cal, f, indent=2)
    calibration_files["astronomy"] = str(astro_path)

    logger.info(f"Created calibration files in {output_dir}")
    return calibration_files


def main():
    """Main function to create sample data."""
    parser = argparse.ArgumentParser(
        description="Create sample data files for testing format loaders"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/sample_data"),
        help="Output directory for sample files",
    )
    parser.add_argument(
        "--num-files",
        "-n",
        type=int,
        default=5,
        help="Number of files to create per format",
    )
    parser.add_argument(
        "--create-calibration", action="store_true", help="Create calibration files"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each domain
    photo_dir = args.output_dir / "photography"
    micro_dir = args.output_dir / "microscopy"
    astro_dir = args.output_dir / "astronomy"

    photo_dir.mkdir(exist_ok=True)
    micro_dir.mkdir(exist_ok=True)
    astro_dir.mkdir(exist_ok=True)

    all_metadata = []

    # Create TIFF files for microscopy
    logger.info("Creating microscopy TIFF files...")
    for i in range(args.num_files):
        # Vary parameters
        shape = (np.random.randint(256, 1024), np.random.randint(256, 1024))
        channels = np.random.choice([1, 3])
        bit_depth = np.random.choice([8, 16])
        pixel_size = np.random.uniform(0.1, 2.0)

        output_path = micro_dir / f"sample_{i:03d}.tif"
        metadata = create_sample_tiff(
            output_path,
            shape=shape,
            channels=channels,
            bit_depth=bit_depth,
            pixel_size=pixel_size,
        )
        all_metadata.append(metadata)

    # Create FITS files for astronomy
    logger.info("Creating astronomy FITS files...")
    for i in range(args.num_files):
        # Vary parameters
        shape = (np.random.randint(512, 2048), np.random.randint(512, 2048))
        bit_depth = np.random.choice([16, 32, -32])
        pixel_scale = np.random.uniform(0.1, 1.0)
        exposure_time = np.random.uniform(60, 600)

        output_path = astro_dir / f"sample_{i:03d}.fits"
        metadata = create_sample_fits(
            output_path,
            shape=shape,
            bit_depth=bit_depth,
            pixel_scale=pixel_scale,
            exposure_time=exposure_time,
        )
        all_metadata.append(metadata)

    # Create RAW metadata for photography
    logger.info("Creating photography RAW metadata...")
    for i in range(args.num_files):
        # Vary parameters
        shape = (np.random.randint(2000, 4000), np.random.randint(3000, 6000))
        iso = np.random.choice([800, 1600, 3200, 6400])
        exposure_time = np.random.uniform(0.01, 1.0)

        output_path = photo_dir / f"sample_{i:03d}.arw"
        metadata = create_sample_raw_metadata(
            output_path, shape=shape, iso=iso, exposure_time=exposure_time
        )
        all_metadata.append(metadata)

    # Create calibration files
    if args.create_calibration:
        logger.info("Creating calibration files...")
        calibration_files = create_calibration_files(args.output_dir)

        # Save calibration file paths
        cal_info = {
            "calibration_files": calibration_files,
            "created_date": "2024-01-01",
            "description": "Sample calibration files for testing",
        }

        with open(args.output_dir / "calibration_info.json", "w") as f:
            json.dump(cal_info, f, indent=2)

    # Save summary metadata
    summary = {
        "total_files": len(all_metadata),
        "files_per_domain": args.num_files,
        "domains": ["photography", "microscopy", "astronomy"],
        "created_date": "2024-01-01",
        "files": all_metadata,
    }

    with open(args.output_dir / "sample_data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Created {len(all_metadata)} sample files in {args.output_dir}")
    logger.info("Summary saved to sample_data_summary.json")


if __name__ == "__main__":
    main()
