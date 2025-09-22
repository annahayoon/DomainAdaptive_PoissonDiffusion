#!/usr/bin/env python3
"""
Quick preprocessing script to convert sample FITS files to .pt format for evaluation.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_fits_to_pt(fits_path: Path, output_path: Path) -> Dict[str, Any]:
    """Convert a FITS file to the .pt format expected by the evaluation script."""
    try:
        from astropy.io import fits
    except ImportError:
        logger.error("astropy required for FITS processing")
        return {}

    # Load FITS file
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header

    # Extract calibration parameters from header
    gain = header.get("GAIN", 1.5)
    read_noise = header.get("RDNOISE", 2.0)

    # Convert ADU to electrons
    electrons = data * gain

    # Normalize to [0, 1] range for model input
    # Use 99.9th percentile to avoid outliers
    scale = np.percentile(electrons, 99.9)
    noisy_norm = electrons / scale
    noisy_norm = np.clip(noisy_norm, 0, 1)

    # Add channel dimension [1, H, W]
    noisy_norm = noisy_norm[np.newaxis, :, :]

    # For evaluation, we'll use the same image as both noisy and clean
    # (since we don't have ground truth for synthetic data)
    clean_norm = noisy_norm.copy()

    # Create metadata
    metadata = {
        "filepath": str(fits_path),
        "domain": "astronomy",
        "gain": float(gain),
        "read_noise": float(read_noise),
        "scale": float(scale),
        "shape": noisy_norm.shape,
        "header_keys": dict(header),
    }

    # Create the data structure expected by evaluation script
    scene_data = {
        "noisy_norm": torch.from_numpy(noisy_norm),
        "clean_norm": torch.from_numpy(clean_norm),
        "metadata": metadata,
    }

    # Save as .pt file
    torch.save(scene_data, output_path)

    logger.info(f"Processed {fits_path.name} -> {output_path.name}")
    logger.info(f"  Shape: {noisy_norm.shape}, Scale: {scale:.2f}")

    return metadata


def main():
    """Process all sample FITS files."""
    input_dir = Path("data/sample_astronomy/astronomy")
    output_dir = Path("data/preprocessed_photography/posterior/astronomy/test")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all FITS files
    fits_files = sorted(list(input_dir.glob("*.fits")))

    if not fits_files:
        logger.error(f"No FITS files found in {input_dir}")
        return

    logger.info(f"Processing {len(fits_files)} FITS files...")

    for i, fits_path in enumerate(fits_files):
        output_path = output_dir / f"astronomy_sample_{i:03d}.pt"
        preprocess_fits_to_pt(fits_path, output_path)

    logger.info(f"Processed {len(fits_files)} astronomy samples")
    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
