#!/usr/bin/env python3
"""
Unified Tiles Pipeline with Domain-Specific Calibration
Domain-specific physics-based calibration methods
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from complete_systematic_tiling import SystematicTiler, SystematicTilingConfig

# Setup logging first
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the BioSR MRC reader for microscopy support
sys.path.append(
    "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/microscopy/Supplementary Files for BioSR/IO_MRC_Python"
)
try:
    from read_mrc import read_mrc

    MRC_READER_AVAILABLE = True
    logger.info("✅ BioSR MRC reader available for microscopy support")
except ImportError:
    MRC_READER_AVAILABLE = False
    logger.warning(
        "⚠️ BioSR MRC reader not available - microscopy MRC files may not load properly"
    )


class SimpleTilesPipeline:
    """Simple file-based pipeline for unified 256×256 tile extraction to PNG"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)

        # Unified tile configuration
        self.tile_size = 256
        self.overlap_ratios = {
            "photography_sony": 0.09,  # ~9% overlap for Sony (1424×2128 → 6×9 = 54 tiles, overlap: 23×22px)
            "photography_fuji": 0.023,  # ~2.3% overlap for Fuji (1008×1508 → 4×6 = 24 tiles, overlap: 6×6px)
            "microscopy": 0.027,  # ~2.7% overlap for microscopy (1004×1004 → 4×4 = 16 tiles, overlap: 7×7px)
            "astronomy": 0.095,  # ~9.5% overlap for astronomy (2116×2110 → 9×9 = 81 tiles, overlap: 24×25px)
        }

        self.sony_tile_config = {
            "tile_size": 256,
            "target_tiles": 54,  # 6×9 = 54 tiles
            "target_grid": (6, 9),  # 6 rows, 9 columns
        }

        self.fuji_tile_config = {
            "tile_size": 256,
            "target_tiles": 24,  # 4×6 = 24 tiles
            "target_grid": (4, 6),  # 4 rows, 6 columns
        }

        self.microscopy_tile_config = {
            "tile_size": 256,
            "target_tiles": 16,  # 4×4 = 16 tiles
            "target_grid": (4, 4),  # 4 rows, 4 columns
        }

        self.astronomy_tile_config = {
            "tile_size": 256,
            "target_tiles": 81,  # 9×9 = 81 tiles
            "target_grid": (9, 9),  # 9 rows, 9 columns
        }

        self.downsample_factors = {
            "sony": 2.0,  # Sony: 2848×4256 → 1424×2128 → 6×9 = 54 tiles
            "fuji": 4.0,  # Fuji: 4032×6032 → 1008×1508 → 4×6 = 24 tiles
            "microscopy": 1.0,  # Microscopy: 1004×1004 → 1004×1004 → 4×4 = 16 tiles (no downsampling)
            "astronomy": 2.0,  # Astronomy: 4232×4220 → 2116×2110 → 9×9 = 81 tiles (maintains aspect ratio)
        }

        # Domain-specific normalization ranges based on pixel distribution analysis
        # These ranges come from domain_pixel_distribution.md comprehensive analysis
        # Used for consistent [-1,1] normalization across all domains for EDM training
        self.domain_ranges = {
            "astronomy": {"min": -65.00, "max": 385.00},  # Updated astronomy range
            "microscopy": {"min": 0.00, "max": 65535.00},  # Microscopy range (16-bit)
            "photography": {"min": 0.00, "max": 15871.00},  # Photography range
        }

    def load_photography_raw(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load photography raw file using rawpy and convert to RGGB format"""
        try:
            import rawpy

            with rawpy.imread(file_path) as raw:
                # Get raw Bayer data
                bayer = raw.raw_image_visible.astype(np.float32)

                # Extract black level and white level
                black_level = np.array(raw.black_level_per_channel)
                white_level = raw.white_level

                # Pack Bayer to 4-channel RGGB format
                packed = self._pack_bayer_to_4channel(bayer, black_level)

                metadata = {
                    "black_level": black_level,
                    "white_level": white_level,
                    "camera_model": getattr(raw, "camera", "Unknown"),
                    "iso": getattr(raw, "iso", None),
                    "exposure_time": getattr(raw, "exposure_time", None),
                    "file_path": file_path,
                }
            return packed, metadata
        except Exception as e:
            logger.error(f"Error loading photography file {file_path}: {e}")
            return None, None

    def _pack_bayer_to_4channel(
        self, bayer: np.ndarray, black_level: np.ndarray
    ) -> np.ndarray:
        """Convert Bayer pattern to 4-channel packed format"""
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

    def load_astronomy_raw(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load astronomy FITS file"""
        try:
            from astropy.io import fits

            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float32)
                header = dict(hdul[0].header)

            # Add channel dimension if needed
            if len(data.shape) == 2:
                data = data[np.newaxis, :, :]

            metadata = {
                "telescope": header.get("TELESCOP", "HST"),
                "instrument": header.get("INSTRUME", "ACS"),
                "detector": header.get("DETECTOR", "WFC"),
                "filter": header.get("FILTER", "CLEAR"),
                "exposure_time": header.get("EXPTIME", 0.0),
                "full_header": header,
                "file_path": file_path,
            }
            return data, metadata
        except Exception as e:
            logger.error(f"Error loading astronomy file {file_path}: {e}")
            return None, None

    def demosaic_rggb_to_rgb(
        self,
        rggb_image: np.ndarray,
    ):
        """
        Demosaic RGGB Bayer pattern to RGB (no white balance correction)
        Works for both Sony and Fuji cameras since they both use RGGB Bayer patterns
        Based on create_raw_comparison.py implementation

        Args:
            rggb_image: RGGB image with shape (4, H, W) - [R, G1, G2, B]
        Returns:
            RGB image with shape (3, H, W) - [R, G, B]
        """
        if rggb_image.shape[0] != 4:
            raise ValueError(
                f"Expected RGGB with 4 channels, got {rggb_image.shape[0]} channels"
            )

        R = rggb_image[0]  # Red channel
        G1 = rggb_image[1]  # Green channel 1
        G2 = rggb_image[2]  # Green channel 2
        B = rggb_image[3]  # Blue channel

        # Average the two green channels
        G = (G1 + G2) / 2.0
        rgb_image = np.stack([R, G, B], axis=0)

        return rgb_image

    def get_pixel_stats(self, data):
        """Get pixel statistics (min, max, mean, median) from data"""
        if data is None:
            return "N/A", "N/A", "N/A", "N/A"

        # Handle different data shapes
        if len(data.shape) == 3:
            flat_data = data.flatten()
        else:
            flat_data = data.flatten()

        # Remove NaN and infinite values
        valid_mask = np.isfinite(flat_data)
        if not np.any(valid_mask):
            return "N/A", "N/A", "N/A", "N/A"

        valid_data = flat_data[valid_mask]
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        mean_val = np.mean(valid_data)
        median_val = np.median(valid_data)

        return min_val, max_val, mean_val, median_val

    def normalize_for_display(self, data, percentile_range=(1, 99)):
        """Normalize data for display using percentile clipping"""
        if data is None:
            return None

        # Handle different data shapes
        if len(data.shape) == 3:
            if data.shape[0] == 3:  # RGB format
                # Convert RGB to grayscale for display
                display_data = np.mean(data, axis=0)
            elif data.shape[0] == 4:  # RGGB format
                # Demosaic RGGB to RGB first, then convert to grayscale
                rgb_data = self.demosaic_rggb_to_rgb(data)
                display_data = np.mean(rgb_data, axis=0)
            else:
                display_data = data.mean(axis=0)
        else:
            display_data = data

        # Ensure we have 2D data for display
        if len(display_data.shape) != 2:
            logger.warning(
                f"Expected 2D data for display, got shape {display_data.shape}"
            )
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

    def extract_single_tile_for_viz(
        self, image, tile_size=256, preserve_intensity=True
    ):
        """Extract a single tile from the image for visualization purposes"""
        if len(image.shape) == 3:
            C, H, W = image.shape
        else:
            H, W = image.shape
            C = 1
            image = image[np.newaxis, :, :]

        if preserve_intensity:
            # Find the region with maximum intensity
            if C == 4:  # RGGB
                intensity = np.mean(image, axis=0)
            elif C == 3:  # RGB
                intensity = np.mean(image, axis=0)
            else:  # Grayscale
                intensity = image[0]

            # Find maximum intensity location
            max_y, max_x = np.unravel_index(np.argmax(intensity), intensity.shape)

            # Calculate tile bounds around maximum intensity point
            y_start = max(0, max_y - tile_size // 2)
            y_end = min(H, y_start + tile_size)
            x_start = max(0, max_x - tile_size // 2)
            x_end = min(W, x_start + tile_size)

            # Adjust if we hit boundaries
            if y_end - y_start < tile_size:
                y_start = max(0, y_end - tile_size)
            if x_end - x_start < tile_size:
                x_start = max(0, x_end - tile_size)
        else:
            # Center-based extraction
            center_y = H // 2
            center_x = W // 2

            y_start = max(0, center_y - tile_size // 2)
            y_end = min(H, y_start + tile_size)
            x_start = max(0, center_x - tile_size // 2)
            x_end = min(W, x_start + tile_size)

        # Extract tile
        tile = image[:, y_start:y_end, x_start:x_end]

        # Pad if necessary
        if tile.shape[1] != tile_size or tile.shape[2] != tile_size:
            padded_tile = np.zeros((C, tile_size, tile_size), dtype=tile.dtype)
            padded_tile[:, : tile.shape[1], : tile.shape[2]] = tile
            tile = padded_tile

        return tile

    def create_scene_visualization(self, viz_data: Dict[str, Any], output_dir: Path):
        """Create 4-step visualization for a single scene (one noisy + one clean pair)"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            domain = viz_data["domain"]
            scene_id = viz_data["scene_id"]
            noisy_data = viz_data["noisy"]
            clean_data = viz_data["clean"]

            # Create figure with 4x2 subplots (4 steps x 2 images)
            fig, axes = plt.subplots(4, 2, figsize=(12, 24))

            # Step 1: Raw loading - NO percentile normalization, just min-max scaling
            noisy_raw_data = noisy_data.get("raw_image")
            clean_raw_data = clean_data.get("raw_image")

            noisy_min, noisy_max, noisy_mean, noisy_median = self.get_pixel_stats(
                noisy_raw_data
            )
            clean_min, clean_max, clean_mean, clean_median = self.get_pixel_stats(
                clean_raw_data
            )

            # Simple min-max normalization for display (no percentile clipping)
            def simple_normalize(data):
                if data is None:
                    return None
                # Handle multi-channel data
                if len(data.shape) == 3:
                    if data.shape[0] == 3:  # RGB
                        display_data = np.mean(data, axis=0)
                    elif data.shape[0] == 4:  # RGGB
                        rgb_data = self.demosaic_rggb_to_rgb(data)
                        display_data = np.mean(rgb_data, axis=0)
                    else:
                        display_data = data.mean(axis=0)
                else:
                    display_data = data
                # Min-max normalize
                data_min, data_max = np.min(display_data), np.max(display_data)
                if data_max - data_min > 0:
                    return (display_data - data_min) / (data_max - data_min)
                return display_data

            noisy_raw = simple_normalize(noisy_raw_data)
            clean_raw = simple_normalize(clean_raw_data)

            if noisy_raw is not None:
                axes[0, 0].imshow(noisy_raw, cmap="gray", vmin=0, vmax=1)
                axes[0, 0].set_title(
                    f'{domain.capitalize()} - Noisy Raw\nShape: {noisy_data.get("original_shape")}\n[{noisy_min:.3f}, {noisy_max:.3f}] μ={noisy_mean:.3f} m={noisy_median:.3f}'
                )
            axes[0, 0].axis("off")

            if clean_raw is not None:
                axes[0, 1].imshow(clean_raw, cmap="gray", vmin=0, vmax=1)
                axes[0, 1].set_title(
                    f'{domain.capitalize()} - Clean Raw\nShape: {clean_data.get("original_shape")}\n[{clean_min:.3f}, {clean_max:.3f}] μ={clean_mean:.3f} m={clean_median:.3f}'
                )
            axes[0, 1].axis("off")

            # Step 2: After tiling - NO percentile normalization
            noisy_tiled_data = noisy_data.get("tiled_image")
            clean_tiled_data = clean_data.get("tiled_image")

            (
                noisy_tiled_min,
                noisy_tiled_max,
                noisy_tiled_mean,
                noisy_tiled_median,
            ) = self.get_pixel_stats(noisy_tiled_data)
            (
                clean_tiled_min,
                clean_tiled_max,
                clean_tiled_mean,
                clean_tiled_median,
            ) = self.get_pixel_stats(clean_tiled_data)

            noisy_tiled = simple_normalize(noisy_tiled_data)
            clean_tiled = simple_normalize(clean_tiled_data)

            if noisy_tiled is not None:
                axes[1, 0].imshow(noisy_tiled, cmap="gray", vmin=0, vmax=1)
                domain_range = self.domain_ranges[domain]
                axes[1, 0].set_title(
                    f'{domain.capitalize()} - Noisy Tiled\nDomain Range: [{domain_range["min"]:.3f}, {domain_range["max"]:.3f}]\nShape: {noisy_tiled_data.shape}\n[{noisy_tiled_min:.3f}, {noisy_tiled_max:.3f}] μ={noisy_tiled_mean:.3f} m={noisy_tiled_median:.3f}'
                )
            axes[1, 0].axis("off")

            if clean_tiled is not None:
                axes[1, 1].imshow(clean_tiled, cmap="gray", vmin=0, vmax=1)
                domain_range = self.domain_ranges[domain]
                axes[1, 1].set_title(
                    f'{domain.capitalize()} - Clean Tiled\nDomain Range: [{domain_range["min"]:.3f}, {domain_range["max"]:.3f}]\nShape: {clean_tiled_data.shape}\n[{clean_tiled_min:.3f}, {clean_tiled_max:.3f}] μ={clean_tiled_mean:.3f} m={clean_tiled_median:.3f}'
                )
            axes[1, 1].axis("off")

            # Step 3: After domain normalization [min,max] -> [0,1] - NO percentile normalization
            noisy_domain_norm_data = noisy_data.get("domain_normalized")
            clean_domain_norm_data = clean_data.get("domain_normalized")

            (
                noisy_norm_min,
                noisy_norm_max,
                noisy_norm_mean,
                noisy_norm_median,
            ) = self.get_pixel_stats(noisy_domain_norm_data)
            (
                clean_norm_min,
                clean_norm_max,
                clean_norm_mean,
                clean_norm_median,
            ) = self.get_pixel_stats(clean_domain_norm_data)

            noisy_domain_norm = simple_normalize(noisy_domain_norm_data)
            clean_domain_norm = simple_normalize(clean_domain_norm_data)

            if noisy_domain_norm is not None:
                axes[2, 0].imshow(noisy_domain_norm, cmap="gray", vmin=0, vmax=1)
                domain_range = self.domain_ranges[domain]
                axes[2, 0].set_title(
                    f'{domain.capitalize()} - Noisy Domain Normalized\n[{domain_range["min"]:.3f}, {domain_range["max"]:.3f}] -> [0,1]\nShape: {noisy_domain_norm_data.shape}\n[{noisy_norm_min:.6f}, {noisy_norm_max:.6f}] μ={noisy_norm_mean:.6f} m={noisy_norm_median:.6f}'
                )
            axes[2, 0].axis("off")

            if clean_domain_norm is not None:
                axes[2, 1].imshow(clean_domain_norm, cmap="gray", vmin=0, vmax=1)
                domain_range = self.domain_ranges[domain]
                axes[2, 1].set_title(
                    f'{domain.capitalize()} - Clean Domain Normalized\n[{domain_range["min"]:.3f}, {domain_range["max"]:.3f}] -> [0,1]\nShape: {clean_domain_norm_data.shape}\n[{clean_norm_min:.6f}, {clean_norm_max:.6f}] μ={clean_norm_mean:.6f} m={clean_norm_median:.6f}'
                )
            axes[2, 1].axis("off")

            # Step 4: After tensor conversion [-1,1]
            noisy_tensor = noisy_data.get("tensor")
            clean_tensor = clean_data.get("tensor")

            (
                noisy_tensor_min,
                noisy_tensor_max,
                noisy_tensor_mean,
                noisy_tensor_median,
            ) = self.get_pixel_stats(
                noisy_tensor.numpy() if noisy_tensor is not None else None
            )
            (
                clean_tensor_min,
                clean_tensor_max,
                clean_tensor_mean,
                clean_tensor_median,
            ) = self.get_pixel_stats(
                clean_tensor.numpy() if clean_tensor is not None else None
            )

            if noisy_tensor is not None:
                # For VISUALIZATION ONLY: Apply percentile normalization
                noisy_tensor_display = self.normalize_for_display(
                    noisy_tensor.numpy(), percentile_range=(1, 99)
                )
                axes[3, 0].imshow(noisy_tensor_display, cmap="gray")
                axes[3, 0].set_title(
                    f"{domain.capitalize()} - Noisy Tensor [-1,1]\nShape: {noisy_tensor.shape}, dtype: {noisy_tensor.dtype}\nActual range: [{noisy_tensor_min:.6f}, {noisy_tensor_max:.6f}] μ={noisy_tensor_mean:.6f}\n(Display: percentile normalized for visibility)"
                )
            axes[3, 0].axis("off")

            if clean_tensor is not None:
                # For VISUALIZATION ONLY: Apply percentile normalization
                clean_tensor_display = self.normalize_for_display(
                    clean_tensor.numpy(), percentile_range=(1, 99)
                )
                axes[3, 1].imshow(clean_tensor_display, cmap="gray")
                axes[3, 1].set_title(
                    f"{domain.capitalize()} - Clean Tensor [-1,1]\nShape: {clean_tensor.shape}, dtype: {clean_tensor.dtype}\nActual range: [{clean_tensor_min:.6f}, {clean_tensor_max:.6f}] μ={clean_tensor_mean:.6f}\n(Display: percentile normalized for visibility)"
                )
            axes[3, 1].axis("off")

            # Add column labels
            fig.text(
                0.25, 0.95, "Noisy Samples", ha="center", fontsize=14, fontweight="bold"
            )
            fig.text(
                0.75, 0.95, "Clean Samples", ha="center", fontsize=14, fontweight="bold"
            )

            # Add row labels
            row_labels = [
                "Raw Loading",
                "After Tiling",
                "Domain Normalization [min,max]->[0,1]",
                "After Tensor Conversion [-1,1]",
            ]
            for j, label in enumerate(row_labels):
                fig.text(
                    0.02,
                    0.8 - j * 0.2,
                    label,
                    ha="left",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                )

            # Add overall title
            fig.suptitle(
                f"Domain: {domain.capitalize()} - Scene: {scene_id}",
                fontsize=16,
                fontweight="bold",
            )

            # Save the plot
            safe_scene_id = scene_id.replace("/", "_").replace(":", "_")
            output_path = output_dir / f"{domain}_{safe_scene_id}_steps.png"
            plt.tight_layout()
            plt.subplots_adjust(top=0.9, left=0.1, right=0.95)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"📊 Saved visualization: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def save_tile_as_pt(
        self,
        tile_data: np.ndarray,
        tile_id: str,
        domain: str,
        data_type: str,
        domain_range: Dict[str, float],
    ) -> Dict[str, Any]:
        """Save tile as .pt file and return metadata

        Photography: Saves as RGB (3, 256, 256) float32
        Microscopy: Saves as grayscale (1, 256, 256) float32
        Astronomy: Saves as grayscale (1, 256, 256) float32

        Normalization: [domain_min, domain_max] → [0,1] → [-1,1]
        """
        try:
            # Ensure proper shape
            if len(tile_data.shape) == 2:
                tile_data = tile_data[np.newaxis, :, :]

            # Create output directory
            output_dir = self.base_path / "processed" / "pt_tiles" / domain / data_type
            output_dir.mkdir(parents=True, exist_ok=True)

            # Handle domain-specific channel requirements
            if domain == "photography":
                # Photography MUST be RGB (3 channels)
                if tile_data.shape[0] == 3:
                    # Already RGB
                    actual_channels = 3
                elif tile_data.shape[0] == 4:
                    # Convert RGGB to RGB
                    rgb_tile = self.demosaic_rggb_to_rgb(tile_data)
                    tile_data = rgb_tile
                    actual_channels = 3
                else:
                    raise ValueError(
                        f"Photography tile has unexpected channels: {tile_data.shape[0]} "
                        f"(expected 3 for RGB or 4 for RGGB)"
                    )
            elif domain == "microscopy" or domain == "astronomy":
                # Microscopy and Astronomy MUST be grayscale (1 channel)
                if tile_data.shape[0] == 1:
                    actual_channels = 1
                elif tile_data.shape[0] == 3:
                    # Convert RGB to grayscale (shouldn't happen, but handle it)
                    logger.warning(
                        f"{domain.capitalize()} tile has 3 channels, converting to grayscale"
                    )
                    tile_data = np.mean(tile_data, axis=0, keepdims=True)
                    actual_channels = 1
                else:
                    # Force to 1 channel
                    tile_data = tile_data[0:1, :, :]
                    actual_channels = 1
            else:
                # Unknown domain - default to whatever it is
                actual_channels = tile_data.shape[0]

            # Final validation
            expected_shape = (actual_channels, self.tile_size, self.tile_size)
            if tile_data.shape != expected_shape:
                raise ValueError(
                    f"Tile shape mismatch for {domain}: {tile_data.shape} != {expected_shape}"
                )

            pt_path = output_dir / f"{tile_id}.pt"

            # Apply domain-specific normalization: [domain_min, domain_max] → [0,1] → [-1,1]
            # This matches the logic in process_tiles_test.py
            domain_min = domain_range["min"]
            domain_max = domain_range["max"]

            # Step 1: Normalize to [0,1] using domain range
            normalized_tile = (tile_data - domain_min) / (domain_max - domain_min)
            normalized_tile = np.clip(normalized_tile, 0, 1)

            # Step 2: Convert to tensor and normalize to [-1,1]
            tensor_data = torch.from_numpy(normalized_tile.astype(np.float32))
            tensor_data = 2 * tensor_data - 1  # [0,1] → [-1,1]

            # Save as PyTorch tensor
            torch.save(tensor_data, pt_path)

            # Return metadata
            return {
                "tile_id": tile_id,
                "pt_path": str(pt_path),
                "domain": domain,
                "data_type": data_type,
                "domain_range": domain_range,
                "tile_size": self.tile_size,
                "channels": actual_channels,
                "processing_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to save tile as .pt: {e}")
            return None

    def process_file_to_pt_tiles(
        self, file_path: str, domain: str, create_viz: bool = False
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Process a single file to .pt tiles with domain-specific normalization - CORRECT FLOW: Load→Tile→Normalize

        Args:
            file_path: Path to the file to process
            domain: Domain name (photography/microscopy/astronomy)
            create_viz: If True, collect intermediate data for visualization

        Returns:
            Tuple of (processed_tiles, viz_data) where viz_data is None unless create_viz=True
        """
        try:
            logger.info(f"Processing {domain} file: {Path(file_path).name}")

            # === STEP 1: Load full image using simplified loaders ===
            if domain == "photography":
                image, metadata = self.load_photography_raw(file_path)
            elif domain == "microscopy":
                image, metadata = self.load_microscopy_mrc(file_path)
            elif domain == "astronomy":
                image, metadata = self.load_astronomy_raw(file_path)
            else:
                logger.warning(f"Unknown domain: {domain}")
                return []

            if image is None or image.size == 0:
                logger.warning(f"Empty or invalid image: {file_path}")
                return []

            # Add file path to metadata
            metadata["file_path"] = file_path

            # === STEP 1.5: Apply domain-specific downsampling ===
            if domain == "photography":
                original_shape = image.shape

                # Determine camera type from file path
                file_path = metadata.get("file_path", "")
                if file_path.endswith(".ARW"):
                    camera_type = "sony"
                    downsample_factor = self.downsample_factors["sony"]
                elif file_path.endswith(".RAF"):
                    camera_type = "fuji"
                    downsample_factor = self.downsample_factors["fuji"]
                else:
                    camera_type = "sony"  # Default to Sony
                    downsample_factor = self.downsample_factors["sony"]

                # Use proper downsampling with anti-aliasing (better for diffusion training)
                image = self._downsample_with_antialiasing(image, downsample_factor)

            elif domain == "microscopy":
                # Apply microscopy downsampling (factor 1.0 = no downsampling)
                downsample_factor = self.downsample_factors["microscopy"]
                if downsample_factor != 1.0:
                    original_shape = image.shape
                    image = self._downsample_with_antialiasing(image, downsample_factor)
                else:
                    pass

            elif domain == "astronomy":
                # Apply astronomy downsampling while maintaining aspect ratio
                original_shape = image.shape

                if len(image.shape) == 3:
                    H, W = image.shape[1], image.shape[2]
                else:
                    H, W = image.shape[0], image.shape[1]

                # Calculate downsampling factor to reach target size while maintaining aspect ratio
                # Target: approximately 2110×2116 (maintains 4232×4220 aspect ratio)
                target_max_dim = 2116  # Use the larger dimension as target
                current_max_dim = max(H, W)

                if current_max_dim > target_max_dim:
                    downsample_factor = current_max_dim / target_max_dim
                    image = self._downsample_with_antialiasing(image, downsample_factor)
                    if len(image.shape) == 3:
                        new_H, new_W = image.shape[1], image.shape[2]
                    else:
                        new_H, new_W = image.shape[0], image.shape[1]
                else:
                    pass

            # === STEP 2: Apply domain-specific normalization ===
            # Store original data range for reference
            orig_min = float(np.min(image))
            orig_max = float(np.max(image))
            orig_mean = float(np.mean(image))
            orig_std = float(np.std(image))

            # Get domain-specific normalization range
            domain_range = self.domain_ranges.get(domain, {"min": 0.0, "max": 1000.0})

            # Initialize visualization data if requested
            viz_data = None
            viz_tile = None

            if create_viz:
                # Store raw image for visualization
                viz_data = {
                    "raw_image": image.copy(),
                    "original_shape": image.shape,
                }

                # Extract one representative tile for visualization
                viz_tile = self.extract_single_tile_for_viz(
                    image, tile_size=256, preserve_intensity=True
                )
                viz_data["tiled_image"] = viz_tile.copy()

            # NO NORMALIZATION YET - tiles will be normalized individually when saved
            # This preserves the raw pixel values for tiling
            # Normalization happens in save_tile_as_pt: [domain_min, domain_max] → [0,1] → [-1,1]

            # === STEP 3: NOW tile the raw image ===
            # Convert to CHW format if needed (tiler expects this)
            if len(image.shape) == 2:
                image = image[np.newaxis, :, :]  # Add channel dimension
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # Convert HWC → CHW for RGB
                image = image.transpose(2, 0, 1)

            # Extract tiles from raw float32 image
            # Use custom tiling for ALL domains to ensure proper overlap and NO padding
            if domain == "photography":
                file_path_str = metadata.get("file_path", "")
                if file_path_str.endswith(".ARW"):
                    # Sony files: custom tiling to get exactly 54 tiles (6×9)
                    tile_infos = self._extract_sony_tiles(image)
                elif file_path_str.endswith(".RAF"):
                    # Fuji files: custom tiling to get exactly 24 tiles (4×6)
                    tile_infos = self._extract_fuji_tiles(image)
                else:
                    # Fallback to Sony custom tiling
                    tile_infos = self._extract_sony_tiles(image)
            elif domain == "astronomy":
                # Custom tiling for astronomy to get exactly 81 tiles (9×9)
                tile_infos = self._extract_astronomy_tiles(image)
            elif domain == "microscopy":
                # Custom tiling for microscopy to get exactly 16 tiles (4×4)
                tile_infos = self._extract_microscopy_tiles(image)
            else:
                # Fallback for unknown domains
                logger.warning(f"Unknown domain: {domain}, using SystematicTiler")
                overlap_ratio = self.overlap_ratios.get(domain, 0.1)
                config = SystematicTilingConfig(
                    tile_size=self.tile_size,
                    overlap_ratio=overlap_ratio,
                    coverage_mode="complete",
                    edge_handling="pad_reflect",
                    min_valid_ratio=0.5,
                )
                tiler = SystematicTiler(config)
                tile_infos = tiler.extract_tiles(normalized_image)

            if not tile_infos:
                logger.warning(f"No tiles extracted from {file_path}")
                return []

            # === STEP 4: Process tiles and separate prior/posterior, assign splits ===
            processed_tiles = []
            # Note: In the new workflow, domain-specific normalization happens to the FULL image before tiling,
            # so all tiles are already normalized. No separate raw_tiles/normalized_tiles stages.
            max_tiles = len(tile_infos)  # Process ALL tiles - no limit

            # Determine data type (prior=clean vs posterior=noisy)
            data_type = self._determine_data_type(file_path, domain)

            # Get scene ID and assign train/test/val split
            scene_id = self._get_scene_id(file_path, domain)
            split = self._assign_split(scene_id, data_type)

            for i in range(max_tiles):
                tile_info = tile_infos[i]
                try:
                    # Get float32 tile data (already calibrated and scaled to [0,1])
                    tile_float = tile_info.tile_data

                    # Validate tile data
                    if tile_float is None or tile_float.size == 0:
                        logger.warning(f"Empty tile {i} from {file_path}")
                        continue

                    # Ensure proper shape
                    if len(tile_float.shape) == 2:
                        tile_float = tile_float[np.newaxis, :, :]

                    # For photography, apply demosaicing from RGGB to RGB
                    if domain == "photography" and tile_float.shape[0] == 4:
                        # Apply demosaicing to convert RGGB to RGB
                        rgb_tile = self.demosaic_rggb_to_rgb(tile_float)
                        tile_to_save = rgb_tile
                    else:
                        tile_to_save = tile_float

                    # Save as .pt with unique tile_id
                    # For photography: include camera type to avoid Sony/Fuji collisions
                    # For microscopy: include parent directory structure to avoid file collisions
                    base_stem = Path(file_path).stem

                    if domain == "photography":
                        # Add camera type to differentiate Sony (.ARW) vs Fuji (.RAF)
                        file_ext = Path(file_path).suffix.upper()
                        camera_prefix = (
                            "sony"
                            if file_ext == ".ARW"
                            else "fuji"
                            if file_ext == ".RAF"
                            else "unknown"
                        )
                        tile_id = f"{domain}_{camera_prefix}_{base_stem}_tile_{i:04d}"
                    elif domain == "microscopy":
                        # Add parent directory info (structure/cell) to avoid collisions
                        path_parts = Path(file_path).parts
                        # Extract structure and cell from path like: structures/F-actin/Cell_005/file.mrc
                        structure = "unknown"
                        cell = "unknown"
                        if "structures" in path_parts:
                            struct_idx = path_parts.index("structures")
                            if len(path_parts) > struct_idx + 1:
                                structure = path_parts[struct_idx + 1]
                            if len(path_parts) > struct_idx + 2:
                                cell = path_parts[struct_idx + 2]
                        # Create unique ID: microscopy_structure_cell_filename
                        tile_id = (
                            f"{domain}_{structure}_{cell}_{base_stem}_tile_{i:04d}"
                        )
                    else:
                        # Default for other domains
                        tile_id = f"{domain}_{base_stem}_tile_{i:04d}"

                    tile_metadata = self.save_tile_as_pt(
                        tile_to_save,
                        tile_id,
                        domain,
                        data_type,
                        domain_range,
                    )

                    if tile_metadata:
                        # Add comprehensive metadata for .pt tile storage
                        tile_metadata.update(
                            {
                                "tile_size": self.tile_size,
                                "grid_x": int(tile_info.grid_position[0]),
                                "grid_y": int(tile_info.grid_position[1]),
                                "image_x": int(tile_info.image_position[0]),
                                "image_y": int(tile_info.image_position[1]),
                                "channels": int(tile_to_save.shape[0]),
                                "quality_score": float(np.mean(tile_to_save))
                                if np.isfinite(np.mean(tile_to_save))
                                else 0.0,
                                "valid_ratio": float(tile_info.valid_ratio),
                                "is_edge_tile": bool(tile_info.is_edge_tile),
                                "overlap_ratio": float(
                                    self.overlap_ratios.get(domain, 0.1)
                                ),
                                "systematic_coverage": True,
                                "original_min": float(orig_min),
                                "original_max": float(orig_max),
                                "original_mean": float(orig_mean),
                                "original_std": float(orig_std),
                                "normalized_min": float(np.min(tile_to_save))
                                if np.isfinite(np.min(tile_to_save))
                                else -1.0,
                                "normalized_max": float(np.max(tile_to_save))
                                if np.isfinite(np.max(tile_to_save))
                                else 1.0,
                                "normalized_mean": float(np.mean(tile_to_save))
                                if np.isfinite(np.mean(tile_to_save))
                                else 0.0,
                                "normalized_std": float(np.std(tile_to_save))
                                if np.isfinite(np.std(tile_to_save))
                                else 0.5,
                                "split": split,
                                "scene_id": scene_id,
                                "domain_range": domain_range,
                            }
                        )
                        processed_tiles.append(tile_metadata)

                except Exception as e:
                    logger.error(f"Failed to process tile {i} from {file_path}: {e}")
                    continue

            # === STEP 5: Process visualization tile if requested ===
            if create_viz and viz_tile is not None and viz_data is not None:
                try:
                    # Apply demosaicing if needed (photography RGGB -> RGB)
                    if domain == "photography" and viz_tile.shape[0] == 4:
                        viz_tile_processed = self.demosaic_rggb_to_rgb(viz_tile)
                    else:
                        viz_tile_processed = viz_tile

                    # Apply domain normalization [domain_min, domain_max] → [0,1]
                    domain_min = domain_range["min"]
                    domain_max = domain_range["max"]
                    normalized_viz = (viz_tile_processed - domain_min) / (
                        domain_max - domain_min
                    )
                    normalized_viz = np.clip(normalized_viz, 0, 1)

                    # Convert to tensor [-1,1]
                    tensor_viz = torch.from_numpy(normalized_viz.astype(np.float32))
                    tensor_viz = 2 * tensor_viz - 1  # [0,1] → [-1,1]

                    # Store all processing steps for visualization
                    viz_data["domain_normalized"] = normalized_viz.copy()
                    viz_data["tensor"] = tensor_viz

                    logger.info(
                        f"✅ Collected visualization data for {Path(file_path).name}"
                    )

                except Exception as e:
                    logger.error(f"Failed to process visualization tile: {e}")
                    viz_data = None

            return processed_tiles, viz_data

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return [], None

    def _select_microscopy_file_pairs(self, all_mrc_files: List[Path]) -> List[str]:
        """
        Select clean/noisy pairs per microscopy cell.
        Pairing strategy:
        - Standard cells (CCPs, Microtubules, F-actin, F-actin_nonlinear): 1 pair per cell
          - Noisy: RawSIMData_gt, Clean: SIM_gt or SIM_gt_a
        - ER cells: 6 pairs per cell (one for each level_01 through level_06)
          - Noisy: RawGTSIMData_level_XX, Clean: GTSIM_level_XX

        Args:
            all_mrc_files: List of all .mrc file paths

        Returns:
            List of selected file paths (clean + noisy pairs)
        """
        import re
        from collections import defaultdict

        # Group files by cell
        # Structure: cell_data[cell_key] = {"RawSIMData_gt": path, "SIM_gt": path, "GTSIM": {level: path}, ...}
        cell_files = defaultdict(lambda: {"GTSIM": {}, "RawGTSIMData": {}})

        for file_path in all_mrc_files:
            file_path_obj = Path(file_path)
            parts = file_path_obj.parts
            filename = file_path_obj.stem

            # Extract structure and cell
            structure = "unknown"
            cell = "unknown"

            if "structures" in parts:
                struct_idx = parts.index("structures")
                if struct_idx + 1 < len(parts):
                    structure = parts[struct_idx + 1]

            if structure != "unknown":
                try:
                    struct_idx = parts.index(structure)
                    if struct_idx + 1 < len(parts):
                        potential_cell = parts[struct_idx + 1]
                        if potential_cell.startswith("Cell_"):
                            cell = potential_cell
                except ValueError:
                    pass

            cell_key = f"{structure}/{cell}"

            # Check if it's a GT file by filename or subdirectory
            # For CCPs/Microtubules/F-actin: filename has "_gt"
            # For ER: subdirectory name indicates type (GTSIM, RawGTSIMData, RawSIMData)

            if "_gt" in filename.lower() or filename.lower().endswith("gt"):
                # Standard naming: files with _gt in filename
                if "rawsimdata" in filename.lower():
                    cell_files[cell_key]["RawSIMData_gt"] = str(file_path)
                elif filename.lower() == "sim_gt_a":
                    # F-actin_nonlinear: use SIM_gt_a as clean target
                    cell_files[cell_key]["SIM_gt_a"] = str(file_path)
                elif filename.lower() in ["sim_gt", "sim"]:
                    cell_files[cell_key]["SIM_gt"] = str(file_path)
            else:
                # ER-style: subdirectory-based naming (all files have _level_XX)
                if len(parts) >= 2:
                    subdirectory = parts[-2]  # Parent directory

                    # Extract level number from filename (e.g., "level_01" from "GTSIM_level_01.mrc")
                    level_match = re.search(r"level_(\d+)", filename.lower())
                    if level_match:
                        level_num = level_match.group(1)  # e.g., "01", "02", etc.

                        if "rawgtsimdata" in subdirectory.lower():
                            # RawGTSIMData = noisy (raw GT SIM data)
                            cell_files[cell_key]["RawGTSIMData"][level_num] = str(
                                file_path
                            )
                        elif subdirectory.lower() == "gtsim":
                            # GTSIM = clean (processed SIM data)
                            cell_files[cell_key]["GTSIM"][level_num] = str(file_path)

        # For each cell, select clean/noisy pair(s)
        selected_files = []
        cells_with_pairs = 0
        cells_without_pairs = 0
        total_cells = len(cell_files)
        total_pairs = 0

        for cell_key, files in cell_files.items():
            cell_has_pairs = False

            # Check for F-actin_nonlinear pairing: RawSIMData_gt (noisy) + SIM_gt_a (clean)
            if "RawSIMData_gt" in files and "SIM_gt_a" in files:
                selected_files.append(files["SIM_gt_a"])  # Clean (F-actin_nonlinear)
                selected_files.append(files["RawSIMData_gt"])  # Noisy
                cell_has_pairs = True
                total_pairs += 1

            # Check for standard pairing: RawSIMData_gt (noisy) + SIM_gt (clean)
            elif "RawSIMData_gt" in files and "SIM_gt" in files:
                selected_files.append(files["SIM_gt"])  # Clean
                selected_files.append(files["RawSIMData_gt"])  # Noisy
                cell_has_pairs = True
                total_pairs += 1

            # Check for ER-style pairing: GTSIM (clean) + RawGTSIMData (noisy)
            # One-to-one mapping by level number
            elif isinstance(files.get("GTSIM"), dict) and isinstance(
                files.get("RawGTSIMData"), dict
            ):
                gtsim_levels = files["GTSIM"]
                rawgtsim_levels = files["RawGTSIMData"]

                # Find matching levels
                matching_levels = set(gtsim_levels.keys()) & set(rawgtsim_levels.keys())

                if matching_levels:
                    for level_num in sorted(matching_levels):
                        selected_files.append(gtsim_levels[level_num])  # Clean
                        selected_files.append(rawgtsim_levels[level_num])  # Noisy
                        total_pairs += 1
                    cell_has_pairs = True

            if cell_has_pairs:
                cells_with_pairs += 1
            else:
                cells_without_pairs += 1

        logger.info(
            f"🔬 Microscopy file selection (RawSIMData_gt=noisy, SIM_gt/SIM_gt_a=clean):"
        )
        logger.info(f"   • Total unique cells found: {total_cells}")
        logger.info(f"   • Cells with complete clean/noisy pairs: {cells_with_pairs}")
        logger.info(f"   • Cells without complete pairs: {cells_without_pairs}")
        logger.info(f"   • Total pairs: {total_pairs} (ER cells have 6 pairs each)")
        logger.info(
            f"   • Total files selected: {len(selected_files)} (from {len(all_mrc_files)} total)"
        )
        logger.info(f"   • Clean files: {len(selected_files) // 2}")
        logger.info(f"   • Noisy files: {len(selected_files) // 2}")
        logger.info(
            f"   • Expected tiles: {len(selected_files)} files × 16 tiles = {len(selected_files) * 16:,} total"
        )

        return selected_files

    def _select_photography_file_pairs(self, all_photo_files: List[Path]) -> List[str]:
        """
        Select clean/noisy pairs for photography (SID dataset).
        Pairing strategy:
        - Clean: Long exposure files (in /long/ directory)
        - Noisy: First short exposure file per scene (0.04s for Sony, first short for Fuji)

        Args:
            all_photo_files: List of all .ARW and .RAF file paths

        Returns:
            List of selected file paths (clean + noisy pairs)
        """
        import re
        from collections import defaultdict

        # Group files by camera type and scene
        sony_scenes = defaultdict(lambda: {"long": None, "short": []})
        fuji_scenes = defaultdict(lambda: {"long": None, "short": []})

        for file_path in all_photo_files:
            file_path_obj = Path(file_path)
            filename = file_path_obj.stem

            # Determine camera type
            is_sony = file_path_obj.suffix.lower() == ".arw"
            is_fuji = file_path_obj.suffix.lower() == ".raf"

            # Parse filename: e.g., "00001_00_10s" or "00001_00_0.04s"
            # Format: {scene_id}_{camera_id}_{exposure}
            parts = filename.split("_")
            if len(parts) >= 3:
                scene_id = parts[0]  # e.g., "00001"
                exposure = parts[2]  # e.g., "10s" or "0.04s"

                # Determine if long or short exposure
                is_long = "/long/" in str(file_path).lower()
                is_short = "/short/" in str(file_path).lower()

                if is_sony:
                    if is_long:
                        sony_scenes[scene_id]["long"] = str(file_path)
                    elif is_short:
                        sony_scenes[scene_id]["short"].append(
                            (exposure, str(file_path))
                        )
                elif is_fuji:
                    if is_long:
                        fuji_scenes[scene_id]["long"] = str(file_path)
                    elif is_short:
                        fuji_scenes[scene_id]["short"].append(
                            (exposure, str(file_path))
                        )

        # Select pairs
        selected_files = []
        sony_pairs = 0
        fuji_pairs = 0

        # Sony: Select long + first 0.04s short per scene
        for scene_id, files in sony_scenes.items():
            if files["long"] and files["short"]:
                # Find 0.04s exposure (or first if 0.04s not available)
                short_files = files["short"]
                target_exposure = "0.04s"

                # Look for 0.04s first
                short_file = None
                for exp, path in short_files:
                    if exp == target_exposure:
                        short_file = path
                        break

                # If no 0.04s, take first short
                if not short_file and short_files:
                    short_file = sorted(short_files)[0][
                        1
                    ]  # Sort by exposure, take first

                if short_file:
                    selected_files.append(files["long"])  # Clean
                    selected_files.append(short_file)  # Noisy
                    sony_pairs += 1

        # Fuji: Select long + first short per scene
        for scene_id, files in fuji_scenes.items():
            if files["long"] and files["short"]:
                # Take first short exposure (sorted alphabetically by exposure)
                short_file = sorted(files["short"])[0][1]

                selected_files.append(files["long"])  # Clean
                selected_files.append(short_file)  # Noisy
                fuji_pairs += 1

        logger.info(f"📸 Photography file selection (long=clean, first short=noisy):")
        logger.info(f"   • Sony scenes with pairs: {sony_pairs}")
        logger.info(f"   • Fuji scenes with pairs: {fuji_pairs}")
        logger.info(f"   • Total pairs: {sony_pairs + fuji_pairs}")
        logger.info(
            f"   • Total files selected: {len(selected_files)} (from {len(all_photo_files)} total)"
        )
        logger.info(
            f"   • Expected tiles: Sony: {sony_pairs * 54:,}, Fuji: {fuji_pairs * 24:,}, Total: {sony_pairs * 54 + fuji_pairs * 24:,}"
        )

        return selected_files

    def reconstruct_metadata_from_incremental(self) -> Dict[str, Any]:
        """
        Reconstruct comprehensive metadata from incremental saves
        Useful if the main metadata file was not saved due to interruption
        """
        logger.info("🔧 Attempting to reconstruct metadata from incremental saves...")

        processed_dir = self.base_path / "processed"
        incremental_files = list(processed_dir.glob("metadata_*_incremental.json"))

        if not incremental_files:
            logger.error("❌ No incremental metadata files found")
            return None

        all_tiles = []
        domains = {}

        for inc_file in incremental_files:
            try:
                with open(inc_file, "r") as f:
                    inc_data = json.load(f)

                domain_name = inc_data.get("domain", "unknown")
                domains[domain_name] = {
                    "files_processed": inc_data.get("files_processed", 0),
                    "tiles_generated": inc_data.get("tiles_generated", 0),
                }
                all_tiles.extend(inc_data.get("tiles", []))

                logger.info(
                    f"✅ Loaded {len(inc_data.get('tiles', []))} tiles from {inc_file.name}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to load {inc_file.name}: {e}")

        reconstructed_metadata = {
            "pipeline_info": {
                "total_tiles": len(all_tiles),
                "domains_processed": list(domains.keys()),
                "domain_stats": domains,
                "processing_timestamp": datetime.now().isoformat(),
                "tile_size": self.tile_size,
                "overlap_ratios": self.overlap_ratios,
                "reconstructed": True,
            },
            "tiles": all_tiles,
        }

        # Save the reconstructed metadata
        metadata_path = (
            self.base_path / "processed" / "comprehensive_tiles_metadata.json"
        )
        try:
            with open(metadata_path, "w") as f:
                json.dump(reconstructed_metadata, f, indent=2, default=str)
            logger.info(f"✅ Reconstructed metadata saved to: {metadata_path}")
            logger.info(f"📊 Total tiles reconstructed: {len(all_tiles):,}")
        except Exception as e:
            logger.error(f"❌ Failed to save reconstructed metadata: {e}")

        return reconstructed_metadata

    def run_pt_tiles_pipeline(
        self, max_files_per_domain: int = None, create_visualizations: bool = False
    ):
        """Run the complete .pt tiles pipeline with domain-specific normalization and optional visualizations

        Args:
            max_files_per_domain: Maximum files to process per domain (None = all files)
            create_visualizations: If True, create 4-step visualization for first scene per domain
        """

        # Define ALL files for each domain
        sample_files = {"photography": [], "microscopy": [], "astronomy": []}

        # Find ALL photography files and select clean/noisy pairs
        sony_files = list(
            Path(
                "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/SID/Sony"
            ).rglob("*.ARW")
        )
        fuji_files = list(
            Path(
                "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/SID/Fuji"
            ).rglob("*.RAF")
        )
        all_photo_files = sony_files + fuji_files
        sample_files["photography"] = self._select_photography_file_pairs(
            all_photo_files
        )

        # Find ALL microscopy files and select clean/noisy pairs
        all_microscopy_files = list(
            Path(
                "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/microscopy"
            ).rglob("*.mrc")
        )
        sample_files["microscopy"] = self._select_microscopy_file_pairs(
            all_microscopy_files
        )

        # Find ALL astronomy files and interleave clean/noisy for better pairing
        astronomy_path = Path(
            "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/astronomy"
        )

        # Separate clean (detection) and noisy (g800l) files
        clean_astronomy = sorted(astronomy_path.rglob("*detection*.fits"))
        noisy_astronomy = sorted(astronomy_path.rglob("*g800l*.fits"))

        # Interleave clean and noisy files so pairs are processed together
        astronomy_files = []
        for clean_file, noisy_file in zip(clean_astronomy, noisy_astronomy):
            astronomy_files.append(str(clean_file))
            astronomy_files.append(str(noisy_file))

        # Add any remaining files if lists are unequal length
        if len(clean_astronomy) > len(noisy_astronomy):
            astronomy_files.extend(
                [str(f) for f in clean_astronomy[len(noisy_astronomy) :]]
            )
        elif len(noisy_astronomy) > len(clean_astronomy):
            astronomy_files.extend(
                [str(f) for f in noisy_astronomy[len(clean_astronomy) :]]
            )

        sample_files["astronomy"] = astronomy_files

        all_tiles_metadata = []
        results = {"domains": {}, "total_tiles": 0}

        # Track visualization data per scene
        viz_data_by_scene = (
            {} if create_visualizations else None
        )  # {domain: {scene_id: {"noisy": data, "clean": data}}}
        viz_output_dir = (
            self.base_path / "processed" / "visualizations"
            if create_visualizations
            else None
        )

        for domain_name, file_list in sample_files.items():
            domain_tiles = []
            processed_files = 0

            # Initialize viz tracking for this domain
            if create_visualizations:
                viz_data_by_scene[domain_name] = {}

            for file_path in file_list[: max_files_per_domain or len(file_list)]:
                if not Path(file_path).exists():
                    continue

                try:
                    # Determine data type and scene
                    data_type = self._determine_data_type(file_path, domain_name)
                    scene_id = self._get_scene_id(file_path, domain_name)

                    # Check if we should collect visualization data
                    # Collect for first scene per domain, and for both noisy and clean
                    should_create_viz = False
                    if create_visualizations:
                        if scene_id not in viz_data_by_scene[domain_name]:
                            # First time seeing this scene - initialize and collect
                            viz_data_by_scene[domain_name][scene_id] = {}
                            should_create_viz = True
                        elif data_type not in viz_data_by_scene[domain_name][scene_id]:
                            # We have this scene but missing this data type (clean or noisy)
                            should_create_viz = True

                    # Process file with or without visualization
                    tiles, viz_data = self.process_file_to_pt_tiles(
                        file_path, domain_name, create_viz=should_create_viz
                    )

                    # Store visualization data if collected
                    if viz_data is not None and should_create_viz:
                        viz_data_by_scene[domain_name][scene_id][data_type] = viz_data

                        # If we have both noisy and clean for this scene, create visualization immediately
                        scene_data = viz_data_by_scene[domain_name][scene_id]
                        if "noisy" in scene_data and "clean" in scene_data:
                            viz_output = {
                                "domain": domain_name,
                                "scene_id": scene_id,
                                "noisy": scene_data["noisy"],
                                "clean": scene_data["clean"],
                            }
                            self.create_scene_visualization(viz_output, viz_output_dir)
                            logger.info(
                                f"📊 Created visualization for {domain_name} scene: {scene_id}"
                            )

                    domain_tiles.extend(tiles)
                    processed_files += 1

                    # Save incremental metadata after EACH file (for better progress tracking)
                    try:
                        incremental_path = (
                            self.base_path
                            / "processed"
                            / f"metadata_{domain_name}_incremental.json"
                        )
                        incremental_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(incremental_path, "w") as f:
                            json.dump(
                                {
                                    "domain": domain_name,
                                    "files_processed": processed_files,
                                    "tiles_generated": len(domain_tiles),
                                    "tiles": domain_tiles,
                                    "timestamp": datetime.now().isoformat(),
                                },
                                f,
                                indent=2,
                                default=str,
                            )
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to save incremental metadata: {e}")

                except Exception as e:
                    logger.error(f"❌ Error processing {file_path}: {e}")
                    continue

            all_tiles_metadata.extend(domain_tiles)
            results["domains"][domain_name] = {
                "files_processed": processed_files,
                "tiles_generated": len(domain_tiles),
            }
            results["total_tiles"] += len(domain_tiles)

        # Save comprehensive metadata (includes all calibration, spatial, and processing info)
        metadata_path = (
            self.base_path / "processed" / "comprehensive_tiles_metadata.json"
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Add processing summary to metadata
        comprehensive_metadata = {
            "pipeline_info": {
                "total_tiles": results["total_tiles"],
                "domains_processed": list(results["domains"].keys()),
                "domain_stats": results["domains"],
                "processing_timestamp": datetime.now().isoformat(),
                "tile_size": self.tile_size,
                "overlap_ratios": self.overlap_ratios,
                "domain_ranges": self.domain_ranges,
                "storage_format": "pt_float32",
                "normalization": "[domain_min, domain_max] → [0,1] → [-1,1]",
            },
            "tiles": all_tiles_metadata,
        }

        # Save with error handling
        try:
            with open(metadata_path, "w") as f:
                json.dump(comprehensive_metadata, f, indent=2, default=str)
            logger.info(f"✅ Comprehensive metadata saved to: {metadata_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save comprehensive metadata: {e}")
            # Try to save to backup location
            backup_path = (
                self.base_path
                / "processed"
                / "comprehensive_tiles_metadata_backup.json"
            )
            try:
                with open(backup_path, "w") as f:
                    json.dump(comprehensive_metadata, f, indent=2, default=str)
                logger.info(f"✅ Metadata saved to backup location: {backup_path}")
            except Exception as e2:
                logger.error(f"❌ Failed to save backup metadata: {e2}")

        return results

    def load_microscopy_mrc(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load microscopy MRC file using the BioSR reader"""
        if not MRC_READER_AVAILABLE:
            raise ImportError("BioSR MRC reader not available")

        try:
            header, data = read_mrc(file_path)

            # Convert to standard format
            if len(data.shape) == 3:
                image = data[:, :, 0].astype(np.float32)  # Take first slice
            else:
                image = data.astype(np.float32)

            # Add channel dimension if needed
            if len(image.shape) == 2:
                image = image[np.newaxis, :, :]

            metadata = {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "nx": int(header["nx"][0]),
                "ny": int(header["ny"][0]),
                "nz": int(header["nz"][0]),
                "mode": int(header["mode"][0]),
                "file_type": "mrc",
                "microscopy_format": True,
            }

            return image, metadata

        except Exception as e:
            logger.error(f"❌ Error reading MRC file {file_path}: {e}")
            raise

    def _determine_data_type(self, file_path: str, domain: str) -> str:
        """Determine if file contains clean or noisy data"""

        file_path_lower = file_path.lower()
        filename_lower = Path(file_path).name.lower()

        if domain == "photography":
            return "clean" if "/long/" in file_path_lower else "noisy"
        elif domain == "microscopy":
            # Check for noisy patterns first (more specific)
            if "rawsimdata" in filename_lower or "rawgtsimdata" in file_path_lower:
                return "noisy"
            # Then check for clean patterns
            elif "sim_gt" in filename_lower or "gtsim" in file_path_lower:
                return "clean"
            # Fallback: if has "gt" in name, assume clean
            elif "gt" in filename_lower:
                return "clean"
            else:
                return "noisy"
        elif domain == "astronomy":
            # Hubble Legacy Archive:
            # - Direct image (detection_sci) = CLEAN reference (high SNR photometry)
            # - G800L grism (g800l_sci) = NOISY (spectroscopic with artifacts)
            return "clean" if "detection" in filename_lower else "noisy"

        return "unknown"

    def _get_scene_id(self, file_path: str, domain: str) -> str:
        """Extract scene/frame identifier to ensure all tiles from same scene stay together"""

        filename = Path(file_path).stem

        if domain == "photography":
            # Extract the scene number (e.g., "00001" from "00001_00_0.04s")
            # This ensures both short and long exposure from same scene stay together
            parts = filename.split("_")
            if len(parts) >= 1:
                return f"photo_{parts[0]}"  # e.g., "photo_00001"
            return f"photo_{filename}"

        elif domain == "microscopy":
            # Extract structure type, cell, and filename for unique scene identification
            # Parse the file path to get structure and cell information
            import re

            file_path = Path(file_path)
            parts = file_path.parts

            # Look for structure and cell in the path
            structure = "unknown"
            cell = "unknown"

            # Find structure name (should be after 'structures/')
            if "structures" in parts:
                struct_idx = parts.index("structures")
                if struct_idx + 1 < len(parts):
                    structure = parts[struct_idx + 1]

            # Find cell name (should be after structure)
            if structure != "unknown":
                try:
                    struct_idx = parts.index(structure)
                    if struct_idx + 1 < len(parts):
                        potential_cell = parts[struct_idx + 1]
                        # Check if it looks like a cell directory (Cell_XXX)
                        if potential_cell.startswith("Cell_"):
                            cell = potential_cell
                except ValueError:
                    pass

            # Extract base name by removing suffixes (same logic as _select_microscopy_file_pairs)
            base_name = filename
            base_name = re.sub(r"^GT_", "", base_name)  # Remove GT_ prefix
            base_name = re.sub(
                r"_gt$", "", base_name, flags=re.IGNORECASE
            )  # Remove _gt suffix
            base_name = re.sub(r"_level_\d+$", "", base_name)  # Remove _level_XX suffix
            base_name = re.sub(r"_noisy$", "", base_name)  # Remove _noisy suffix

            # Create unique scene ID: micro_{structure}_{cell}_{base_name}
            return f"micro_{structure}_{cell}_{base_name}"

        elif domain == "astronomy":
            # Extract the observation ID (first part before underscore)
            # e.g., "j8hqbifjq" from "j8hqbifjq_detection_sci" or "j8hqbifjq_g800l_sci"
            parts = filename.split("_")
            if len(parts) >= 1:
                return f"astro_{parts[0]}"  # e.g., "astro_j8hqbifjq"
            return f"astro_{filename}"

        return f"unknown_{filename}"

    def _assign_split(self, scene_id: str, data_type: str) -> str:
        """
        Assign train/test/validation split based on scene_id using random assignment

        FIXED STRATEGY - NO DATA LEAKAGE:
        - Each scene gets assigned to ONE split only (train/val/test)
        - ALL data types (clean/noisy) from same scene go to same split
        - 70% train, 15% validation, 15% test
        - This prevents data leakage between training and testing
        """

        # Use random seed based on scene_id for consistent assignment within scene
        import hashlib
        import random

        # Create deterministic seed from scene_id for consistent assignment within scene
        seed = int(
            hashlib.md5(scene_id.encode(), usedforsecurity=False).hexdigest(), 16
        ) % (2**32)
        random.seed(seed)

        # Random assignment with proper distribution - SAME FOR ALL DATA TYPES IN SCENE
        split_val = random.random() * 100  # 0-100

        if split_val < 70:
            return "train"
        elif split_val < 85:
            return "validation"
        else:
            return "test"

    def _extract_sony_tiles(self, image: np.ndarray) -> List[Any]:
        """Extract exactly 54 tiles (6×9 grid) from Sony images with proper overlap"""
        try:
            if len(image.shape) == 3:
                C, H, W = image.shape
            else:
                H, W = image.shape
                C = 1
                image = image[np.newaxis, :, :]

            # We want exactly 54 tiles: 6 rows × 9 columns, all 256×256 (NO PADDING!)
            rows, cols = 6, 9
            tile_size = 256

            # Calculate stride to achieve proper coverage with overlap
            # Formula: (stride × (n-1)) + tile_size = image_size
            stride_h = int(np.floor((H - tile_size) / (rows - 1))) if rows > 1 else 0
            stride_w = int(np.floor((W - tile_size) / (cols - 1))) if cols > 1 else 0

            tiles = []
            for row in range(rows):
                for col in range(cols):
                    # Calculate tile position with proper overlap
                    if row == rows - 1:
                        # Last row: align to bottom edge
                        y_start = H - tile_size
                    else:
                        y_start = row * stride_h

                    if col == cols - 1:
                        # Last column: align to right edge
                        x_start = W - tile_size
                    else:
                        x_start = col * stride_w

                    y_end = y_start + tile_size
                    x_end = x_start + tile_size

                    # Extract tile (guaranteed to be exactly 256×256)
                    if len(image.shape) == 3:
                        tile_data = image[:, y_start:y_end, x_start:x_end]
                    else:
                        tile_data = image[y_start:y_end, x_start:x_end]

                    # Verify no padding needed
                    assert tile_data.shape[-2:] == (
                        tile_size,
                        tile_size,
                    ), f"Sony tile size mismatch: expected {tile_size}×{tile_size}, got {tile_data.shape[-2:]}"

                    # Create tile info object
                    tile_info = type(
                        "TileInfo",
                        (),
                        {
                            "tile_data": tile_data,
                            "grid_position": (col, row),
                            "image_position": (x_start, y_start),
                            "valid_ratio": 1.0,  # All pixels are valid (no padding!)
                            "is_edge_tile": (
                                row == 0
                                or row == rows - 1
                                or col == 0
                                or col == cols - 1
                            ),
                        },
                    )()

                    tiles.append(tile_info)

            overlap_h = tile_size - stride_h
            overlap_w = tile_size - stride_w
            return tiles

        except Exception as e:
            logger.error(f"Sony tile extraction failed: {e}")
            return []

    def _extract_fuji_tiles(self, image: np.ndarray) -> List[Any]:
        """Extract exactly 24 tiles (4×6 grid) from Fuji images with proper overlap"""
        try:
            if len(image.shape) == 3:
                C, H, W = image.shape
            else:
                H, W = image.shape
                C = 1
                image = image[np.newaxis, :, :]

            # We want exactly 24 tiles: 4 rows × 6 columns, all 256×256 (NO PADDING!)
            # FIXED: Changed from 6×4 to 4×6 to match image aspect ratio (H=1008, W=1508)
            rows, cols = 4, 6
            tile_size = 256

            # Calculate stride to achieve proper coverage with overlap
            # Formula: (stride × (n-1)) + tile_size = image_size
            stride_h = int(np.floor((H - tile_size) / (rows - 1))) if rows > 1 else 0
            stride_w = int(np.floor((W - tile_size) / (cols - 1))) if cols > 1 else 0

            tiles = []
            for row in range(rows):
                for col in range(cols):
                    # Calculate tile position with proper overlap
                    if row == rows - 1:
                        # Last row: align to bottom edge
                        y_start = H - tile_size
                    else:
                        y_start = row * stride_h

                    if col == cols - 1:
                        # Last column: align to right edge
                        x_start = W - tile_size
                    else:
                        x_start = col * stride_w

                    y_end = y_start + tile_size
                    x_end = x_start + tile_size

                    # Extract tile (guaranteed to be exactly 256×256)
                    if len(image.shape) == 3:
                        tile_data = image[:, y_start:y_end, x_start:x_end]
                    else:
                        tile_data = image[y_start:y_end, x_start:x_end]

                    # Verify no padding needed
                    assert tile_data.shape[-2:] == (
                        tile_size,
                        tile_size,
                    ), f"Fuji tile size mismatch: expected {tile_size}×{tile_size}, got {tile_data.shape[-2:]}"

                    # Create tile info object
                    tile_info = type(
                        "TileInfo",
                        (),
                        {
                            "tile_data": tile_data,
                            "grid_position": (col, row),
                            "image_position": (x_start, y_start),
                            "valid_ratio": 1.0,  # All pixels are valid (no padding!)
                            "is_edge_tile": (
                                row == 0
                                or row == rows - 1
                                or col == 0
                                or col == cols - 1
                            ),
                        },
                    )()

                    tiles.append(tile_info)

            overlap_h = tile_size - stride_h
            overlap_w = tile_size - stride_w
            return tiles

        except Exception as e:
            logger.error(f"Fuji tile extraction failed: {e}")
            return []

    def _extract_astronomy_tiles(self, image: np.ndarray) -> List[Any]:
        """Extract exactly 81 tiles (9×9 grid) from astronomy images with proper overlap"""
        try:
            if len(image.shape) == 3:
                C, H, W = image.shape
            else:
                H, W = image.shape
                C = 1
                image = image[np.newaxis, :, :]

            # We want exactly 81 tiles: 9 rows × 9 columns, all 256×256 (NO PADDING!)
            rows, cols = 9, 9
            tile_size = 256

            # Calculate stride to achieve proper coverage with overlap
            # Formula: (stride × (n-1)) + tile_size = image_size
            stride_h = int(np.floor((H - tile_size) / (rows - 1))) if rows > 1 else 0
            stride_w = int(np.floor((W - tile_size) / (cols - 1))) if cols > 1 else 0

            tiles = []
            for row in range(rows):
                for col in range(cols):
                    # Calculate tile position with proper overlap
                    if row == rows - 1:
                        # Last row: align to bottom edge
                        y_start = H - tile_size
                    else:
                        y_start = row * stride_h

                    if col == cols - 1:
                        # Last column: align to right edge
                        x_start = W - tile_size
                    else:
                        x_start = col * stride_w

                    y_end = y_start + tile_size
                    x_end = x_start + tile_size

                    # Extract tile (guaranteed to be exactly 256×256)
                    if len(image.shape) == 3:
                        tile_data = image[:, y_start:y_end, x_start:x_end]
                    else:
                        tile_data = image[y_start:y_end, x_start:x_end]

                    # Verify no padding needed
                    assert tile_data.shape[-2:] == (
                        tile_size,
                        tile_size,
                    ), f"Astronomy tile size mismatch: expected {tile_size}×{tile_size}, got {tile_data.shape[-2:]}"

                    # Create tile info object
                    tile_info = type(
                        "TileInfo",
                        (),
                        {
                            "tile_data": tile_data,
                            "grid_position": (col, row),
                            "image_position": (x_start, y_start),
                            "valid_ratio": 1.0,  # All pixels are valid (no padding!)
                            "is_edge_tile": (
                                row == 0
                                or row == rows - 1
                                or col == 0
                                or col == cols - 1
                            ),
                        },
                    )()

                    tiles.append(tile_info)

            overlap_h = tile_size - stride_h
            overlap_w = tile_size - stride_w
            return tiles

        except Exception as e:
            logger.error(f"Astronomy tile extraction failed: {e}")
            return []

    def _extract_microscopy_tiles(self, image: np.ndarray) -> List[Any]:
        """Extract exactly 16 tiles (4×4 grid) from microscopy images with proper overlap"""
        try:
            if len(image.shape) == 3:
                C, H, W = image.shape
            else:
                H, W = image.shape
                C = 1
                image = image[np.newaxis, :, :]

            # We want exactly 16 tiles: 4 rows × 4 columns, all 256×256 (NO PADDING!)
            rows, cols = 4, 4
            tile_size = 256

            # Calculate stride to achieve proper coverage with overlap
            # Formula: (stride × (n-1)) + tile_size = image_size
            stride_h = int(np.floor((H - tile_size) / (rows - 1))) if rows > 1 else 0
            stride_w = int(np.floor((W - tile_size) / (cols - 1))) if cols > 1 else 0

            tiles = []
            for row in range(rows):
                for col in range(cols):
                    # Calculate tile position with proper overlap
                    if row == rows - 1:
                        # Last row: align to bottom edge
                        y_start = H - tile_size
                    else:
                        y_start = row * stride_h

                    if col == cols - 1:
                        # Last column: align to right edge
                        x_start = W - tile_size
                    else:
                        x_start = col * stride_w

                    y_end = y_start + tile_size
                    x_end = x_start + tile_size

                    # Extract tile (guaranteed to be exactly 256×256)
                    if len(image.shape) == 3:
                        tile_data = image[:, y_start:y_end, x_start:x_end]
                    else:
                        tile_data = image[y_start:y_end, x_start:x_end]

                    # Verify no padding needed
                    assert tile_data.shape[-2:] == (
                        tile_size,
                        tile_size,
                    ), f"Microscopy tile size mismatch: expected {tile_size}×{tile_size}, got {tile_data.shape[-2:]}"

                    # Create tile info object
                    tile_info = type(
                        "TileInfo",
                        (),
                        {
                            "tile_data": tile_data,
                            "grid_position": (col, row),
                            "image_position": (x_start, y_start),
                            "valid_ratio": 1.0,  # All pixels are valid (no padding!)
                            "is_edge_tile": (
                                row == 0
                                or row == rows - 1
                                or col == 0
                                or col == cols - 1
                            ),
                        },
                    )()

                    tiles.append(tile_info)

            overlap_h = tile_size - stride_h
            overlap_w = tile_size - stride_w
            return tiles

        except Exception as e:
            logger.error(f"Microscopy tile extraction failed: {e}")
            return []

    def _downsample_with_antialiasing(
        self, image: np.ndarray, factor: float
    ) -> np.ndarray:
        """
        Downsample image with anti-aliasing while maintaining aspect ratio
        Best practice for diffusion training to avoid aliasing artifacts and distortion
        """
        try:
            from scipy.ndimage import gaussian_filter, zoom

            # Apply Gaussian filter before downsampling (anti-aliasing)
            sigma = factor / 3.0  # Standard deviation for Gaussian filter

            if len(image.shape) == 3:  # CHW format
                C, H, W = image.shape

                # Apply Gaussian filter to each channel
                filtered_image = np.zeros_like(image)
                for c in range(C):
                    filtered_image[c] = gaussian_filter(image[c], sigma=sigma)

                # Calculate new dimensions maintaining aspect ratio (use exact division)
                new_H = int(H / factor)
                new_W = int(W / factor)

                # Downsample with zoom maintaining aspect ratio
                zoom_factors = (1, new_H / H, new_W / W)
                downsampled = zoom(
                    filtered_image, zoom_factors, order=1
                )  # Linear interpolation

                downsampling_factor = W / new_W

            else:  # HW format
                H, W = image.shape

                # Apply Gaussian filter
                filtered_image = gaussian_filter(image, sigma=sigma)

                # Calculate new dimensions maintaining aspect ratio (use exact division)
                new_H = int(H / factor)
                new_W = int(W / factor)

                # Downsample with zoom maintaining aspect ratio
                zoom_factors = (new_H / H, new_W / W)
                downsampled = zoom(
                    filtered_image, zoom_factors, order=1
                )  # Linear interpolation

                downsampling_factor = W / new_W

            return downsampled.astype(image.dtype)

        except ImportError:
            # Fallback to simple downsampling if scipy not available
            # scipy not available, using simple downsampling
            step = int(factor)
            if len(image.shape) == 3:  # CHW format
                return image[:, ::step, ::step]
            else:  # HW format
                return image[::step, ::step]


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple .pt Tiles Pipeline with Domain-Specific Range Normalization for [-1,1]"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data",
        help="Base path for data",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum files per domain (default: None = all files)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate 4-step visualizations for first scene per domain",
    )

    args = parser.parse_args()

    # Run .pt tiles pipeline
    pipeline = SimpleTilesPipeline(args.base_path)
    results = pipeline.run_pt_tiles_pipeline(
        args.max_files, create_visualizations=args.visualize
    )

    if results.get("total_tiles", 0) > 0:
        print(f"\n🎊 SUCCESS: .pt Tiles Pipeline Completed!")
        print(f"📊 Total .pt tiles generated: {results['total_tiles']:,}")
        print(f"📐 Tile size: 256×256")
        print(f"💾 .pt files (float32) saved to: {args.base_path}/processed/pt_tiles/")
        print(
            f"🎯 Domain-specific range normalization applied: [domain_min, domain_max] → [0,1] → [-1,1]"
        )
        print(f"   • Photography: [0, 15871] → [-1,1]")
        print(f"   • Microscopy: [0, 65535] → [-1,1]")
        print(f"   • Astronomy: [-65, 385] → [-1,1]")

        if args.visualize:
            print(
                f"\n📊 Visualizations saved to: {args.base_path}/processed/visualizations/"
            )

        print(f"\n📋 Domain Results:")
        for domain, stats in results.get("domains", {}).items():
            print(
                f"   • {domain.upper()}: {stats['tiles_generated']} tiles from {stats['files_processed']} files"
            )

        print(f"\n🎯 Ready for diffusion model training!")
    else:
        print(f"\n❌ FAILED: No tiles were generated")
        print(f"Total tiles processed: {results.get('total_tiles', 0)}")


if __name__ == "__main__":
    # Start the .pt tiles pipeline with domain-specific range normalization for [-1,1]
    logger.info(
        "🚀 Starting .pt tiles pipeline with domain-specific range normalization: [domain_min, domain_max] → [0,1] → [-1,1]..."
    )
    main()
