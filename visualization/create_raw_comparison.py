#!/usr/bin/env python3
"""
Create comparison visualization of raw data from all three domains
Shows noisy vs clean samples with pixel brightness ranges
"""

import sys
import warnings
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# Add paths for domain processors
sys.path.append("/home/jilab/Jae/preprocessing")
sys.path.append(
    "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/microscopy/Supplementary Files for BioSR/IO_MRC_Python"
)

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import normalize_for_display

# Import preprocessing utilities for consistent demosaicing
from preprocessing.utils import demosaic_raw_to_rgb, get_pixel_stats
from preprocessing.visualizations import prepare_for_tile_visualization


def load_photography_raw(file_path):
    """Load photography raw file and demosaic to RGB format using preprocessing utilities"""
    try:
        # Use preprocessing demosaicing function which handles both Sony and Fuji sensors
        rgb_image, metadata = demosaic_raw_to_rgb(file_path)

        if rgb_image is None:
            return None, None

        # rgb_image is already in (3, H, W) format, normalized to [0, 1]
        return rgb_image, metadata
    except Exception as e:
        print(f"Error loading photography file {file_path}: {e}")
        return None, None


def load_microscopy_raw(file_path):
    """Load microscopy MRC file"""
    try:
        from read_mrc import read_mrc

        header, data = read_mrc(file_path)
        data = data.astype(np.float32)

        # Handle 3D data (take first slice)
        if len(data.shape) == 3:
            data = data[:, :, 0]

        metadata = {
            "data_type": str(data.dtype),
            "shape": data.shape,
            "header": header if header is not None else {},
        }
        return data, metadata
    except Exception as e:
        print(f"Error loading microscopy file {file_path}: {e}")
        return None, None


def load_astronomy_raw(file_path):
    """Load astronomy FITS file"""
    try:
        from astropy.io import fits

        with fits.open(file_path) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = dict(hdul[0].header)

        metadata = {
            "telescope": header.get("TELESCOP", "HST"),
            "instrument": header.get("INSTRUME", "ACS"),
            "detector": header.get("DETECTOR", "WFC"),
            "filter": header.get("FILTER", "CLEAR"),
            "exposure_time": header.get("EXPTIME", 0.0),
            "full_header": header,
        }
        return data, metadata
    except Exception as e:
        print(f"Error loading astronomy file {file_path}: {e}")
        return None, None


def extract_brightness_range(data):
    """Extract brightness range from image data using preprocessing utilities"""
    if data is None:
        return "N/A"

    # Use preprocessing utility for consistent statistics
    stats = get_pixel_stats(data)
    min_val, max_val, mean_val, median_val = stats

    return f"[{min_val:.1f}, {max_val:.1f}] (mean: {mean_val:.1f})"


# Use shared normalize_for_display utility


def create_comparison_visualization():
    """Create the main comparison visualization"""

    # Define sample files
    samples = {
        "Photography": {
            "noisy": "/home/jilab/anna_OS_ML/DomainAdaptive_PoissonDiffusion/data/raw/SID/Sony/short/20201_00_0.04s.ARW",
            "clean": "/home/jilab/anna_OS_ML/DomainAdaptive_PoissonDiffusion/data/raw/SID/Sony/long/20201_00_10s.ARW",
        },
        "Microscopy": {
            "noisy": "/home/jilab/anna_OS_ML/DomainAdaptive_PoissonDiffusion/data/raw/microscopy/structures/Microtubules/Cell_001/RawSIMData_level_01.mrc",
            "clean": "/home/jilab/anna_OS_ML/DomainAdaptive_PoissonDiffusion/data/raw/microscopy/structures/Microtubules/Cell_001/RawSIMData_gt.mrc",
        },
        "Astronomy": {
            "clean": "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/astronomy/hla_associations/direct_images/j8hp9fq1q_detection_sci.fits",
            "noisy": "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/astronomy/hla_associations/g800l_images/j8hp9fq1q_g800l_sci.fits",
        },
    }

    # Load functions for each domain
    load_functions = {
        "Photography": load_photography_raw,
        "Microscopy": load_microscopy_raw,
        "Astronomy": load_astronomy_raw,
    }

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(
        "Raw Data Comparison: Noisy vs Clean Samples\nPixel Brightness Ranges",
        fontsize=16,
        fontweight="bold",
    )

    # Load and display each sample
    for i, (domain, files) in enumerate(samples.items()):
        load_func = load_functions[domain]

        # Ensure consistent order: noisy first (j=0), clean second (j=1)
        ordered_items = [("noisy", files["noisy"]), ("clean", files["clean"])]

        for j, (data_type, file_path) in enumerate(ordered_items):
            ax = axes[i, j]

            # Load data
            data, metadata = load_func(file_path)

            if data is not None:
                # Extract brightness range
                brightness_range = extract_brightness_range(data)

                # Handle photography RGB data - prepare for display using preprocessing utilities
                if (
                    domain == "Photography"
                    and len(data.shape) == 3
                    and data.shape[0] == 3
                ):
                    # Photography data is already RGB (3, H, W) from demosaic_raw_to_rgb
                    # Use preprocessing visualization utility to prepare for display (CHW -> HWC)
                    display_data = prepare_for_tile_visualization(data)
                    if display_data is not None:
                        # Simple normalization for RGB display (percentile clipping)
                        valid_mask = np.isfinite(display_data)
                        if np.any(valid_mask):
                            p_low, p_high = np.percentile(
                                display_data[valid_mask], (1, 99)
                            )
                            display_data = np.clip(display_data, p_low, p_high)
                            display_data = (display_data - p_low) / (
                                p_high - p_low + 1e-8
                            )
                        else:
                            display_data = None
                    else:
                        # Fallback to grayscale conversion
                        display_data = normalize_for_display(data)
                else:
                    # Normalize for display (grayscale or other formats)
                    display_data = normalize_for_display(data)

                if display_data is not None:
                    # Reshape for display if needed
                    if len(display_data.shape) == 1:
                        # Try to reshape to square
                        size = int(np.sqrt(len(display_data)))
                        if size * size == len(display_data):
                            display_data = display_data.reshape(size, size)
                        else:
                            # Find the next perfect square
                            next_size = size + 1
                            pad_size = next_size * next_size
                            padded = np.zeros(pad_size)
                            padded[: len(display_data)] = display_data
                            display_data = padded.reshape(next_size, next_size)

                    # Display image - handle RGB vs grayscale
                    if len(display_data.shape) == 3 and display_data.shape[2] == 3:
                        # RGB image (H, W, 3) - use RGB display
                        im = ax.imshow(display_data, aspect="equal")
                    else:
                        # Grayscale image - use gray colormap
                        im = ax.imshow(display_data, cmap="gray", aspect="equal")
                    ax.set_title(
                        f"{domain} - {data_type.capitalize()}\nBrightness: {brightness_range}",
                        fontsize=12,
                        fontweight="bold",
                    )

                    # Add brightness range as text overlay
                    ax.text(
                        0.02,
                        0.98,
                        brightness_range,
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Failed to load data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{domain} - {data_type.capitalize()}\nFailed to load")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Failed to load data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{domain} - {data_type.capitalize()}\nFailed to load")

            ax.set_xticks([])
            ax.set_yticks([])

    # Add column labels
    fig.text(0.25, 0.95, "Noisy Samples", ha="center", fontsize=14, fontweight="bold")
    fig.text(0.75, 0.95, "Clean Samples", ha="center", fontsize=14, fontweight="bold")

    # Add row labels
    row_labels = ["Photography\n(Sony ARW)", "Microscopy\n(MRC)", "Astronomy\n(FITS)"]
    for i, label in enumerate(row_labels):
        fig.text(
            0.02,
            0.8 - i * 0.25,
            label,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            rotation=90,
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.1, right=0.95)

    # Save the figure
    output_path = "/home/jilab/Jae/dataset/processed/test_visualizations/comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Comparison visualization saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    create_comparison_visualization()
