#!/usr/bin/env python3
"""
Standalone script to visualize physics-based calibration effects
without Spark dependencies
"""

import os
import sys
import numpy as np
from pathlib import Path
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import domain processors
sys.path.insert(0, '/home/jilab/anna_OS_ML')
from domain_processors import create_processor

# Import MRC reader
sys.path.append('/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/microscopy/Supplementary Files for BioSR/IO_MRC_Python')
from read_mrc import read_mrc


def extract_iso_from_filename(file_path: str) -> int:
    """Extract ISO from SID dataset filename"""
    filename = Path(file_path).name
    
    # Check if it's a Sony file (ARW) - most are ISO200
    if filename.endswith('.ARW'):
        # Sony files are typically ISO200 based on the dataset lists
        return 200
    # Check if it's a Fuji file (RAF) - most are ISO2000  
    elif filename.endswith('.RAF'):
        # Fuji files are typically ISO2000 based on the dataset lists
        return 2000
    else:
        return 2000  # Default fallback


def get_physics_calibration(domain: str, metadata: dict = None):
    """Get domain-specific physics-based calibration parameters"""
    
    if domain == "photography":
        # Sony A7S II camera sensor calibration
        iso = metadata.get("iso", 2000) if metadata else 2000
        if iso is None:
            iso = 2000  # Default fallback
        
        # Sony A7S II calibration parameters based on ISO
        if iso >= 4000:
            gain = 0.79  # e-/ADU at ISO 4000 (unity gain)
            read_noise = 2.5  # electrons
        elif iso >= 2000:
            gain = 2.1  # e-/ADU at ISO 2000
            read_noise = 6.0  # electrons
        elif iso >= 1000:
            gain = 4.2  # e-/ADU at ISO 1000 (estimated)
            read_noise = 8.0  # electrons
        elif iso >= 500:
            gain = 8.4  # e-/ADU at ISO 500 (estimated)
            read_noise = 10.0  # electrons
        elif iso >= 200:
            gain = 5.0  # e-/ADU at ISO 200 (1/0.198 DN/e⁻ = ~5 e-/DN)
            read_noise = 3.56  # electrons RMS at ISO 200
        else:
            gain = 2.1  # Default fallback
            read_noise = 6.0
        method = "photon_transfer_curve"
        
    elif domain == "microscopy":
        # BioSR sCMOS detector calibration
        noise_level = metadata.get("noise_level", 5) if metadata else 5
        gain = 1.0 + (noise_level - 5) * 0.1  # Range: 0.6-1.4
        read_noise = 1.5 + (noise_level - 5) * 0.2  # Range: 0.5-2.5
        method = "noise_level_analysis"
        
    elif domain == "astronomy":
        # HST ACS_WFC detector calibration
        gain = 1.0  # e-/ADU
        read_noise = 3.5  # electrons
        method = "fits_header_analysis"
        
    else:
        gain = 1.0
        read_noise = 1.0
        method = "default"
    
    return gain, read_noise, method


def apply_physics_calibration(image: np.ndarray, gain: float, read_noise: float):
    """Apply physics-based calibration: ADU → electrons"""
    # Convert ADU to electrons
    image_electrons = image.astype(np.float32) * gain
    
    # Apply read noise correction (subtract noise floor)
    image_calibrated = image_electrons - read_noise
    
    # Don't clip negative values - allow them to preserve the full dynamic range
    # image_calibrated = np.maximum(image_calibrated, 0.0)
    
    return image_calibrated


def demosaic_rggb_to_rgb(rggb_image: np.ndarray, gains: np.ndarray = None):
    """
    Demosaic RGGB Bayer pattern to RGB
    
    Args:
        rggb_image: RGGB image with shape (4, H, W) - [R, G1, G2, B]
        gains: Optional per-channel gains for white balance [R_gain, G_gain, B_gain]
    
    Returns:
        RGB image with shape (3, H, W) - [R, G, B]
    """
    if rggb_image.shape[0] != 4:
        raise ValueError(f"Expected RGGB with 4 channels, got {rggb_image.shape[0]} channels")
    
    R = rggb_image[0]  # Red channel
    G1 = rggb_image[1]  # Green channel 1
    G2 = rggb_image[2]  # Green channel 2
    B = rggb_image[3]   # Blue channel
    
    # Average the two green channels
    G = (G1 + G2) / 2.0
    
    # Apply white balance gains if provided
    if gains is not None and len(gains) >= 3:
        R = R * gains[0]
        G = G * gains[1]
        B = B * gains[2]
    
    # Stack to RGB
    rgb_image = np.stack([R, G, B], axis=0)
    
    return rgb_image


def apply_photography_calibration(rggb_image: np.ndarray, gain: float, read_noise: float, metadata: dict):
    """
    Apply photography-specific calibration to RGGB data
    
    Args:
        rggb_image: RGGB image with shape (4, H, W)
        gain: Overall gain (e-/ADU)
        read_noise: Read noise (electrons)
        metadata: Image metadata with white balance info
    
    Returns:
        Calibrated RGB image with shape (3, H, W)
    """
    # Step 1: Apply physics calibration to each RGGB channel
    # Each channel may have slightly different characteristics
    calibrated_rggb = np.zeros_like(rggb_image)
    
    for i in range(4):
        channel = rggb_image[i]
        # Apply physics calibration: ADU → electrons
        channel_electrons = channel * gain
        channel_calibrated = channel_electrons - read_noise
        # Don't clip negative values - allow them to preserve the full dynamic range
        calibrated_rggb[i] = channel_calibrated
    
    # Step 2: Demosaic RGGB to RGB
    # Get white balance gains from metadata if available
    wb_gains = None
    if 'daylight_whitebalance' in metadata:
        wb = metadata['daylight_whitebalance']
        if len(wb) >= 3:
            # Normalize white balance gains (typically G=1.0)
            wb_gains = np.array([wb[0], (wb[1] + wb[2])/2, wb[3]])  # R, G, B
            wb_gains = wb_gains / wb_gains[1]  # Normalize by green
    
    rgb_calibrated = demosaic_rggb_to_rgb(calibrated_rggb, wb_gains)
    
    return rgb_calibrated


def apply_astronomical_scaling(image: np.ndarray, domain: str):
    """Apply domain-specific scaling optimized for astronomical data"""
    
    if domain == "astronomy":
        # Use professional astronomy asinh preprocessing
        try:
            from astronomy_asinh_preprocessing import AstronomyAsinhPreprocessor, AsinhScalingConfig
            
            # Create astronomy preprocessor
            config = AsinhScalingConfig(
                softening_factor=1e-3,  # β parameter for asinh
                scale_percentile=99.5,  # Capture bright features
                background_percentile=10.0,  # Robust background estimation
                cosmic_ray_threshold=5.0  # Detect cosmic rays
            )
            astro_preprocessor = AstronomyAsinhPreprocessor(config)
            
            # Prepare calibration parameters
            calibration_dict = {
                'gain': 1.0,  # Already in electrons
                'read_noise': 3.5,  # HST typical
                'background': 0.0,  # Will be estimated
                'scale_factor': None  # Will be computed
            }
            
            # Apply professional astronomy preprocessing
            result = astro_preprocessor.preprocess_astronomy_image(
                image, 
                calibration_dict, 
                apply_cosmic_ray_removal=True
            )
            
            # Get the properly scaled image
            scaled = result['preprocessed_image']
            preprocessing_metadata = result['preprocessing_metadata']
            
            logger.info(f"   📊 Professional astronomy asinh scaling:")
            logger.info(f"      Background: {preprocessing_metadata['background_level']:.2f} e-")
            logger.info(f"      Scale factor: {preprocessing_metadata['scale_factor']:.2f} e-")
            logger.info(f"      Cosmic rays detected: {preprocessing_metadata['cosmic_rays_detected']}")
            logger.info(f"      Softening factor: {preprocessing_metadata['softening_factor']:.3f}")
            
        except ImportError:
            # Fallback to simple astronomical scaling
            logger.warning("   ⚠️ astronomy_asinh_preprocessing module not available, using fallback")
            image_flat = image.flatten()
            image_flat = image_flat[image_flat > 0]
            
            if len(image_flat) > 0:
                p1 = np.percentile(image_flat, 1)
                softening = max(p1, 1e-6)
                scaled = np.arcsinh(image / (softening + 1e-10))
                scaled = scaled * 2.0
                logger.info(f"   📊 Fallback astronomical scaling: softening={softening:.6f}")
            else:
                scaled = image
                logger.warning("   ⚠️ No positive values found for astronomical scaling")
            
    else:
        # For photography and microscopy, use standard linear scaling
        # No asinh scaling - just normalize to [0,1] range
        img_min = float(np.min(image))
        img_max = float(np.max(image))
        
        if img_max > img_min and np.isfinite(img_min) and np.isfinite(img_max):
            scaled = (image - img_min) / (img_max - img_min)
        else:
            scaled = np.clip(image, 0, 1)
        
        logger.info(f"   📊 Linear scaling for {domain}: range=[{img_min:.3f}, {img_max:.3f}]")
    
    return scaled


def visualize_calibration_effects():
    """Create 3-panel visualizations for all domains"""
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    logger.info("🎨 Creating calibration effect visualizations...")
    logger.info("="*60)
    
    base_path = Path("/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data")
    output_path = base_path / "calibration_visualizations"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample files for each domain
    test_files = {
        "photography": {
            "noisy": str(base_path / "raw/SID/Sony/short/00001_00_0.04s.ARW"),
            "clean": str(base_path / "raw/SID/Sony/long/00001_00_10s.ARW")
        },
        "microscopy": {
            "noisy": str(base_path / "raw/microscopy/structures/Microtubules/Cell_005/RawSIMData_level_02.mrc"),
            "clean": str(base_path / "raw/microscopy/structures/Microtubules/Cell_005/RawSIMData_gt.mrc")
        },
        "astronomy": {
            "clean": str(base_path / "raw/astronomy/hla_associations/direct_images/j8hp9fq1q_detection_sci.fits"),  # Direct image = clean reference
            "noisy": str(base_path / "raw/astronomy/hla_associations/g800l_images/j8hp9fq1q_g800l_sci.fits")  # G800L = noisy (spectroscopic with artifacts)
        }
    }
    
    for domain_name, file_pair in test_files.items():
        logger.info(f"\n🔬 Processing {domain_name.upper()} domain...")
        
        try:
            noisy_path = file_pair["noisy"]
            clean_path = file_pair["clean"]
            
            if not Path(noisy_path).exists() or not Path(clean_path).exists():
                logger.warning(f"⚠️ Files not found for {domain_name}, skipping...")
                continue
            
            # Create processor
            processor = create_processor(domain_name)
            
            # Load noisy image
            logger.info(f"📖 Loading noisy: {Path(noisy_path).name}")
            if domain_name == "microscopy" and noisy_path.lower().endswith('.mrc'):
                header, data = read_mrc(noisy_path)
                if len(data.shape) == 3:
                    noisy_raw = data[:, :, 0].astype(np.float32)[np.newaxis, :, :]
                else:
                    noisy_raw = data.astype(np.float32)
                    if len(noisy_raw.shape) == 2:
                        noisy_raw = noisy_raw[np.newaxis, :, :]
                metadata_noisy = {"domain": domain_name}
            else:
                noisy_raw, metadata_noisy = processor.load_image(noisy_path)
            
            # Load clean image
            logger.info(f"📖 Loading clean: {Path(clean_path).name}")
            if domain_name == "microscopy" and clean_path.lower().endswith('.mrc'):
                header, data = read_mrc(clean_path)
                if len(data.shape) == 3:
                    clean_raw = data[:, :, 0].astype(np.float32)[np.newaxis, :, :]
                else:
                    clean_raw = data.astype(np.float32)
                    if len(clean_raw.shape) == 2:
                        clean_raw = clean_raw[np.newaxis, :, :]
                metadata_clean = {"domain": domain_name}
            else:
                clean_raw, metadata_clean = processor.load_image(clean_path)
            
            # Get calibration parameters
            if domain_name == "photography":
                # Extract ISO from filename for SID dataset
                extracted_iso = extract_iso_from_filename(noisy_path)
                metadata_noisy["iso"] = extracted_iso
                logger.info(f"📷 Extracted ISO from filename: {extracted_iso}")
            
            gain, read_noise, calibration_method = get_physics_calibration(domain_name, metadata_noisy)
            logger.info(f"📊 Calibration: gain={gain:.3f}, read_noise={read_noise:.3f}, method={calibration_method}")
            
            # Debug: Print actual ISO from metadata
            if domain_name == "photography":
                actual_iso = metadata_noisy.get("iso")
                logger.info(f"📷 Using ISO: {actual_iso}")
            
            # Extract center tile for visualization (256×256) BEFORE processing
            tile_size = 256
            h, w = noisy_raw.shape[1], noisy_raw.shape[2]
            center_y, center_x = h // 2, w // 2
            y1, y2 = center_y - tile_size//2, center_y + tile_size//2
            x1, x2 = center_x - tile_size//2, center_x + tile_size//2
            
            # Ensure we don't go out of bounds
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            
            # Extract tiles from raw data
            noisy_tile_raw = noisy_raw[:, y1:y2, x1:x2]
            clean_tile_raw = clean_raw[:, y1:y2, x1:x2]
            
            # Apply calibration to the extracted tile
            if domain_name == "photography":
                # Apply physics-based calibration: ADU → electrons (no white balance)
                noisy_tile_calibrated = apply_physics_calibration(noisy_tile_raw, gain, read_noise)
                logger.info(f"   📷 Applied physics calibration (gain={gain:.1f}, read_noise={read_noise:.1f})")
            elif domain_name == "astronomy":
                # Just convert ADU to electrons, don't subtract read noise as it makes values too low
                noisy_tile_calibrated = noisy_tile_raw * gain
                noisy_tile_calibrated = np.maximum(noisy_tile_calibrated, 0.0)
            else:
                # Apply full physics calibration for microscopy
                noisy_tile_calibrated = apply_physics_calibration(noisy_tile_raw, gain, read_noise)
            
            # Apply domain-specific scaling (optimized for astronomical data)
            noisy_tile_scaled = apply_astronomical_scaling(noisy_tile_calibrated, domain_name)
            
            # Normalize and convert to 8-bit
            noisy_min, noisy_max = float(np.min(noisy_tile_scaled)), float(np.max(noisy_tile_scaled))
            if noisy_max > noisy_min:
                noisy_norm = (noisy_tile_scaled - noisy_min) / (noisy_max - noisy_min)
            else:
                noisy_norm = np.clip(noisy_tile_scaled, 0, 1)
            noisy_tile_8bit = np.clip(noisy_norm * 255.0, 0, 255).astype(np.uint8)
            
            # Normalize raw tiles for display [0,1]
            noisy_tile_raw_norm = (noisy_tile_raw - np.min(noisy_tile_raw)) / (np.max(noisy_tile_raw) - np.min(noisy_tile_raw) + 1e-8)
            clean_tile_raw_norm = (clean_tile_raw - np.min(clean_tile_raw)) / (np.max(clean_tile_raw) - np.min(clean_tile_raw) + 1e-8)
            noisy_tile_calibrated_norm = noisy_tile_scaled
            
            # Create 3-panel visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Panel 1: Original noisy tile
            if noisy_tile_raw_norm.shape[0] == 1:
                axes[0].imshow(noisy_tile_raw_norm[0], cmap='gray', vmin=0, vmax=1)
            elif noisy_tile_raw_norm.shape[0] == 3:
                axes[0].imshow(noisy_tile_raw_norm.transpose(1, 2, 0))
            elif noisy_tile_raw_norm.shape[0] == 4:
                # For 4-channel RGGB Bayer pattern, create RGB visualization
                if domain_name == "photography":
                    # Simple RGB reconstruction from RGGB: R, G1, G2, B -> R, G, B
                    rgb_tile = np.stack([
                        noisy_tile_raw_norm[0],  # R channel
                        (noisy_tile_raw_norm[1] + noisy_tile_raw_norm[2]) / 2,  # Average G channels
                        noisy_tile_raw_norm[3]   # B channel
                    ], axis=0)
                    axes[0].imshow(rgb_tile.transpose(1, 2, 0))
                else:
                    # For other domains, use first channel
                    axes[0].imshow(noisy_tile_raw_norm[0], cmap='gray', vmin=0, vmax=1)
            axes[0].set_title(f'Original Noisy Tile\n(RAW RGGB values)', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Panel 2: Original clean tile
            if clean_tile_raw_norm.shape[0] == 1:
                axes[1].imshow(clean_tile_raw_norm[0], cmap='gray', vmin=0, vmax=1)
            elif clean_tile_raw_norm.shape[0] == 3:
                axes[1].imshow(clean_tile_raw_norm.transpose(1, 2, 0))
            elif clean_tile_raw_norm.shape[0] == 4:
                # For 4-channel RGGB Bayer pattern, create RGB visualization
                if domain_name == "photography":
                    # Simple RGB reconstruction from RGGB: R, G1, G2, B -> R, G, B
                    rgb_tile = np.stack([
                        clean_tile_raw_norm[0],  # R channel
                        (clean_tile_raw_norm[1] + clean_tile_raw_norm[2]) / 2,  # Average G channels
                        clean_tile_raw_norm[3]   # B channel
                    ], axis=0)
                    axes[1].imshow(rgb_tile.transpose(1, 2, 0))
                else:
                    # For other domains, use first channel
                    axes[1].imshow(clean_tile_raw_norm[0], cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Original Clean Tile\n(RAW RGGB values)', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            # Panel 3: Calibrated tile (8-bit PNG)
            if noisy_tile_calibrated_norm.shape[0] == 1:
                axes[2].imshow(noisy_tile_calibrated_norm[0], cmap='gray', vmin=0, vmax=1)
            elif noisy_tile_calibrated_norm.shape[0] == 3:
                # RGB image (photography after demosaicing)
                axes[2].imshow(noisy_tile_calibrated_norm.transpose(1, 2, 0))
            elif noisy_tile_calibrated_norm.shape[0] == 4:
                # For 4-channel RGGB Bayer pattern, create RGB visualization
                if domain_name == "photography":
                    # Simple RGB reconstruction from RGGB: R, G1, G2, B -> R, G, B
                    rgb_tile = np.stack([
                        noisy_tile_calibrated_norm[0],  # R channel
                        (noisy_tile_calibrated_norm[1] + noisy_tile_calibrated_norm[2]) / 2,  # Average G channels
                        noisy_tile_calibrated_norm[3]   # B channel
                    ], axis=0)
                    axes[2].imshow(rgb_tile.transpose(1, 2, 0))
                else:
                    # For other domains, use first channel
                    axes[2].imshow(noisy_tile_calibrated_norm[0], cmap='gray', vmin=0, vmax=1)
            if domain_name == "astronomy":
                axes[2].set_title(f'Physics-Calibrated + Astronomical Scaling\n(8-bit PNG, asinh optimized for low signal)', fontsize=12, fontweight='bold')
            elif domain_name == "photography":
                axes[2].set_title(f'Physics-Calibrated RGGB\n(8-bit PNG, gain={gain:.1f}, read_noise={read_noise:.1f})', fontsize=12, fontweight='bold')
            else:
                axes[2].set_title(f'Physics-Calibrated + Linear Scaled\n(8-bit PNG, {calibration_method})', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            # Add overall title
            fig.suptitle(f'{domain_name.upper()} Domain: Physics-Based Calibration Effect\n' +
                        f'Gain: {gain:.3f} e-/ADU, Read Noise: {read_noise:.3f} e-',
                        fontsize=14, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            
            # Save visualization
            output_file = output_path / f"{domain_name}_calibration_comparison.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved visualization: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {domain_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue
    
    logger.info(f"\n🎉 Visualizations saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    visualize_calibration_effects()

