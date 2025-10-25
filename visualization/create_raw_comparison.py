#!/usr/bin/env python3
"""
Create comparison visualization of raw data from all three domains
Shows noisy vs clean samples with pixel brightness ranges
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add paths for domain processors
sys.path.append('/home/jilab/Jae/preprocessing')
sys.path.append('/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/microscopy/Supplementary Files for BioSR/IO_MRC_Python')

def load_photography_raw(file_path):
    """Load photography raw file using rawpy and convert to RGGB format"""
    try:
        import rawpy
        with rawpy.imread(file_path) as raw:
            # Get raw Bayer data
            bayer = raw.raw_image_visible.astype(np.float32)
            
            # Extract black level and white level
            black_level = np.array(raw.black_level_per_channel)
            white_level = raw.white_level
            
            # Pack Bayer to 4-channel RGGB format (same as domain_processors.py)
            packed = _pack_bayer_to_4channel(bayer, black_level, "sony")
            
            metadata = {
                "black_level": black_level,
                "white_level": white_level,
                "camera_model": getattr(raw, "camera", "Sony"),
                "iso": getattr(raw, "iso", None),
                "exposure_time": getattr(raw, "exposure_time", None),
            }
        return packed, metadata
    except Exception as e:
        print(f"Error loading photography file {file_path}: {e}")
        return None, None

def _pack_bayer_to_4channel(bayer, black_level, camera_type="sony"):
    """Convert Bayer pattern to 4-channel packed format (from domain_processors.py)"""
    H, W = bayer.shape
    
    # Create black level map
    black_map = np.zeros((H, W), dtype=np.float32)
    black_map[0::2, 0::2] = black_level[0]  # R
    black_map[0::2, 1::2] = black_level[1]  # G1
    black_map[1::2, 0::2] = black_level[2]  # G2
    black_map[1::2, 1::2] = black_level[3]  # B
    
    # Subtract black level
    bayer_corrected = np.maximum(bayer - black_map, 0)
    
    # Pack to 4-channel
    packed = np.zeros((4, H // 2, W // 2), dtype=np.float32)
    packed[0] = bayer_corrected[0::2, 0::2]  # R
    packed[1] = bayer_corrected[0::2, 1::2]  # G1
    packed[2] = bayer_corrected[1::2, 0::2]  # G2
    packed[3] = bayer_corrected[1::2, 1::2]  # B
    
    return packed

def demosaic_rggb_to_rgb(rggb_image):
    """Demosaic RGGB Bayer pattern to RGB (from process_tiles_pipeline.py)"""
    if rggb_image.shape[0] != 4:
        raise ValueError(f"Expected RGGB with 4 channels, got {rggb_image.shape[0]} channels")
    
    R = rggb_image[0]  # Red channel
    G1 = rggb_image[1]  # Green channel 1
    G2 = rggb_image[2]  # Green channel 2
    B = rggb_image[3]  # Blue channel
    
    G = (G1 + G2) / 2.0
    rgb_image = np.stack([R, G, B], axis=0)
    
    return rgb_image

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
    """Extract brightness range from image data"""
    if data is None:
        return "N/A"
    
    # Handle different data shapes
    if len(data.shape) == 3:
        if data.shape[0] == 3:  # RGB format
            # Use all channels for brightness range
            flat_data = data.flatten()
        elif data.shape[0] == 4:  # RGGB format
            flat_data = data[0].flatten()  # Use red channel
        else:
            flat_data = data.mean(axis=0).flatten()
    else:
        flat_data = data.flatten()
    
    # Remove any NaN or infinite values
    flat_data = flat_data[np.isfinite(flat_data)]
    
    if len(flat_data) == 0:
        return "N/A"
    
    min_val = np.min(flat_data)
    max_val = np.max(flat_data)
    mean_val = np.mean(flat_data)
    
    return f"[{min_val:.1f}, {max_val:.1f}] (mean: {mean_val:.1f})"

def normalize_for_display(data, percentile_range=(1, 99)):
    """Normalize data for display using percentile clipping"""
    if data is None:
        return None
    
    # Handle different data shapes
    if len(data.shape) == 3:
        if data.shape[0] == 3:  # RGB format
            # Convert RGB to grayscale for display
            display_data = np.mean(data, axis=0)
        elif data.shape[0] == 4:  # RGGB format
            display_data = data[0]  # Use red channel
        else:
            display_data = data.mean(axis=0)
    else:
        display_data = data
    
    # Ensure we have 2D data for display
    if len(display_data.shape) != 2:
        print(f"Warning: Expected 2D data for display, got shape {display_data.shape}")
        return None
    
    # Remove NaN and infinite values
    valid_mask = np.isfinite(display_data)
    if not np.any(valid_mask):
        return None
    
    # Clip to percentiles and normalize to [0, 1]
    p_low, p_high = np.percentile(display_data[valid_mask], percentile_range)
    clipped = np.clip(display_data, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low)
    
    return normalized


def create_comparison_visualization():
    """Create the main comparison visualization"""
    
    # Define sample files
    samples = {
        'Photography': {
            'noisy': '/home/jilab/anna_OS_ML/DomainAdaptive_PoissonDiffusion/data/raw/SID/Sony/short/20201_00_0.04s.ARW',
            'clean': '/home/jilab/anna_OS_ML/DomainAdaptive_PoissonDiffusion/data/raw/SID/Sony/long/20201_00_10s.ARW'
        },
        'Microscopy': {
            'noisy': '/home/jilab/anna_OS_ML/DomainAdaptive_PoissonDiffusion/data/raw/microscopy/structures/Microtubules/Cell_001/RawSIMData_level_01.mrc',
            'clean': '/home/jilab/anna_OS_ML/DomainAdaptive_PoissonDiffusion/data/raw/microscopy/structures/Microtubules/Cell_001/RawSIMData_gt.mrc'
        },
        'Astronomy': {
            'clean': '/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/astronomy/hla_associations/direct_images/j8hp9fq1q_detection_sci.fits',
            'noisy': '/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/astronomy/hla_associations/g800l_images/j8hp9fq1q_g800l_sci.fits'
        }
    }
    
    # Load functions for each domain
    load_functions = {
        'Photography': load_photography_raw,
        'Microscopy': load_microscopy_raw,
        'Astronomy': load_astronomy_raw
    }
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle('Raw Data Comparison: Noisy vs Clean Samples\nPixel Brightness Ranges', fontsize=16, fontweight='bold')
    
    # Load and display each sample
    for i, (domain, files) in enumerate(samples.items()):
        load_func = load_functions[domain]
        
        # Ensure consistent order: noisy first (j=0), clean second (j=1)
        ordered_items = [('noisy', files['noisy']), ('clean', files['clean'])]
        
        for j, (data_type, file_path) in enumerate(ordered_items):
            ax = axes[i, j]
            
            # Load data
            data, metadata = load_func(file_path)
            
            if data is not None:
                # Extract brightness range
                brightness_range = extract_brightness_range(data)
                
                # Handle photography RGGB data specially - demosaic to RGB
                if domain == 'Photography' and data.shape[0] == 4:
                    # Demosaic RGGB to RGB using the same method as process_tiles_pipeline.py
                    rgb_data = demosaic_rggb_to_rgb(data)
                    display_data = normalize_for_display(rgb_data)
                else:
                    # Normalize for display
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
                            padded[:len(display_data)] = display_data
                            display_data = padded.reshape(next_size, next_size)
                    
                    # Display image
                    im = ax.imshow(display_data, cmap='gray', aspect='equal')
                    ax.set_title(f'{domain} - {data_type.capitalize()}\nBrightness: {brightness_range}', 
                               fontsize=12, fontweight='bold')
                    
                    # Add brightness range as text overlay
                    ax.text(0.02, 0.98, brightness_range, transform=ax.transAxes, 
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                else:
                    ax.text(0.5, 0.5, 'Failed to load data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{domain} - {data_type.capitalize()}\nFailed to load')
            else:
                ax.text(0.5, 0.5, 'Failed to load data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{domain} - {data_type.capitalize()}\nFailed to load')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add column labels
    fig.text(0.25, 0.95, 'Noisy Samples', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.75, 0.95, 'Clean Samples', ha='center', fontsize=14, fontweight='bold')
    
    # Add row labels
    row_labels = ['Photography\n(Sony ARW)', 'Microscopy\n(MRC)', 'Astronomy\n(FITS)']
    for i, label in enumerate(row_labels):
        fig.text(0.02, 0.8 - i*0.25, label, ha='left', va='center', fontsize=12, fontweight='bold', rotation=90)
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.1, right=0.95)
    
    # Save the figure
    output_path = '/home/jilab/Jae/dataset/processed/test_visualizations/comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison visualization saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    create_comparison_visualization()

