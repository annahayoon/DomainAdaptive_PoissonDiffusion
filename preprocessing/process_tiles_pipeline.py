#!/usr/bin/env python3
"""
Unified Tiles Pipeline with Domain-Specific Calibration
Domain-specific physics-based calibration methods
"""

import io
import json
import logging
import os
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from complete_systematic_tiling import SystematicTiler, SystematicTilingConfig

# Import domain processors and tiling
from domain_processors import create_processor
from PIL import Image

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

    def prepare_tile_data(self, tile_data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Prepare tile data for storage as PNG (lossless compression, 8-bit for cross-domain consistency)"""

        try:
            if len(tile_data.shape) == 2:
                tile_data = tile_data[np.newaxis, :, :]

            tile_data = np.ascontiguousarray(tile_data.astype(np.float32))

            original_shape = tile_data.shape
            channels = original_shape[0]

            # Convert from float32 [0,1] to uint8 [0,255] for consistent 8-bit representation
            # This ensures all domains (photography, microscopy, astronomy) have the same bit depth
            # which is important for cross-domain training
            tile_uint8 = np.clip(tile_data * 255.0, 0, 255).astype(np.uint8)

            # Handle different channel configurations
            if channels == 1:
                # Grayscale image (microscopy, astronomy)
                img_array = tile_uint8[0]
                pil_image = Image.fromarray(img_array, mode="L")  # 8-bit grayscale
            elif channels == 3:
                # RGB image (photography)
                img_array = tile_uint8.transpose(1, 2, 0)
                pil_image = Image.fromarray(img_array, mode="RGB")  # 8-bit RGB

            # Save to PNG in memory buffer
            buffer = io.BytesIO()
            pil_image.save(
                buffer, format="PNG", compress_level=6
            )  # 6 is balanced compression
            png_bytes = buffer.getvalue()

            storage_info = {
                "method": "png_lossless_8bit",
                "original_size": tile_data.nbytes,
                "stored_size": len(png_bytes),
                "compression_ratio": tile_data.nbytes / len(png_bytes)
                if len(png_bytes) > 0
                else 1.0,
                "shape": original_shape,
                "dtype": str(tile_data.dtype),
                "png_mode": pil_image.mode,
                "png_bit_depth": 8,
                "channels": channels,
            }

            return png_bytes, storage_info

        except Exception as e:
            logger.error(f"PNG preparation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            error_info = {
                "method": "failed",
                "original_size": tile_data.nbytes if tile_data is not None else 0,
                "stored_size": 0,
                "error": str(e),
            }
            return b"", error_info

    def demosaic_rggb_to_rgb(
        self,
        rggb_image: np.ndarray,
        gains: np.ndarray = None,
        camera_type: str = "generic",
    ):
        """
        Demosaic RGGB Bayer pattern to RGB (no white balance correction)
        Works for both Sony and Fuji cameras since they both use RGGB Bayer patterns

        Args:
            rggb_image: RGGB image with shape (4, H, W) - [R, G1, G2, B]
            gains: Ignored (no white balance correction)
            camera_type: Camera type ("sony", "fuji", or "generic") - ignored for demosaicing
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

        G = (G1 + G2) / 2.0
        rgb_image = np.stack([R, G, B], axis=0)

        return rgb_image

    def save_tile_as_png(
        self,
        tile_data: np.ndarray,
        tile_id: str,
        domain: str,
        data_type: str,
        gain: float,
        read_noise: float,
        calibration_method: str,
    ) -> Dict[str, Any]:
        """Save tile as PNG file and return metadata"""
        try:
            from PIL import Image

            # Ensure proper shape and convert to uint8
            if len(tile_data.shape) == 2:
                tile_data = tile_data[np.newaxis, :, :]

            # Convert to uint8 [0,255]
            tile_uint8 = np.clip(tile_data * 255.0, 0, 255).astype(np.uint8)

            # Create output directory
            output_dir = self.base_path / "processed" / "png_tiles" / domain / data_type
            output_dir.mkdir(parents=True, exist_ok=True)

            # Handle photography RGB data (already converted from RGGB)
            if domain == "photography" and tile_uint8.shape[0] == 3:
                # Photography data is already converted to RGB (3 channels)
                img_array = tile_uint8.transpose(1, 2, 0)
                pil_image = Image.fromarray(img_array, mode="RGB")
                actual_channels = 3
            elif domain == "photography" and tile_uint8.shape[0] == 4:
                tile_float = tile_uint8.astype(np.float32) / 255.0
                rgb_tile_float = self.demosaic_rggb_to_rgb(tile_float)
                rgb_tile_uint8 = np.clip(rgb_tile_float * 255.0, 0, 255).astype(
                    np.uint8
                )
                img_array = rgb_tile_uint8.transpose(1, 2, 0)
                pil_image = Image.fromarray(img_array, mode="RGB")
                actual_channels = 3
            elif tile_uint8.shape[0] == 1:
                img_array = tile_uint8[0]
                pil_image = Image.fromarray(img_array, mode="L")
                actual_channels = 1
            elif tile_uint8.shape[0] == 3:
                img_array = tile_uint8.transpose(1, 2, 0)
                pil_image = Image.fromarray(img_array, mode="RGB")
                actual_channels = 3
            else:
                img_array = tile_uint8[0]
                pil_image = Image.fromarray(img_array, mode="L")
                actual_channels = 1

            png_path = output_dir / f"{tile_id}.png"
            pil_image.save(png_path, format="PNG", compress_level=6)

            # Return metadata
            return {
                "tile_id": tile_id,
                "png_path": str(png_path),
                "domain": domain,
                "data_type": data_type,
                "gain": gain,
                "read_noise": read_noise,
                "calibration_method": calibration_method,
                "tile_size": self.tile_size,
                "channels": actual_channels,
                "processing_timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to save tile as PNG: {e}")
            return None

    def process_file_to_png_tiles(
        self, file_path: str, domain: str
    ) -> List[Dict[str, Any]]:
        """Process a single file to PNG tiles with physics calibration - CORRECT FLOW: Calibrate→Convert→Tile"""
        try:
            logger.info(f"Processing {domain} file: {Path(file_path).name}")

            # Initialize processor
            processor = create_processor(domain)

            # === STEP 1: Load full image ===
            if domain == "microscopy" and file_path.lower().endswith(".mrc"):
                image, metadata = self.load_microscopy_mrc(file_path)
            else:
                image, metadata = processor.load_image(file_path)

            if image is None or image.size == 0:
                logger.warning(f"Empty or invalid image: {file_path}")
                return []

            # Add file path to metadata for calibration
            metadata["file_path"] = file_path

            # Image loaded successfully

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

                # Image downsampled with anti-aliasing

            elif domain == "microscopy":
                # Apply microscopy downsampling (factor 1.0 = no downsampling)
                downsample_factor = self.downsample_factors["microscopy"]
                if downsample_factor != 1.0:
                    original_shape = image.shape
                    image = self._downsample_with_antialiasing(image, downsample_factor)
                    logger.info(
                        f"🔬 Microscopy downsampled: {original_shape} → {image.shape} (factor: {downsample_factor:.2f}x)"
                    )
                else:
                    logger.info(
                        f"🔬 Microscopy: no downsampling needed (factor: {downsample_factor:.2f}x)"
                    )

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
                    logger.info(
                        f"🌟 Astronomy downsampled: {original_shape} → {image.shape} (factor: {downsample_factor:.2f}x)"
                    )
                    logger.info(
                        f"   Aspect ratio preserved: {W/H:.3f} → {new_W/new_H:.3f}"
                    )
                else:
                    logger.info(
                        f"🌟 Astronomy image already at target size or smaller: {image.shape}"
                    )

            # === STEP 2: Apply domain-specific calibration to FULL image ===
            gain, read_noise, calibration_method = self._get_physics_based_calibration(
                domain, metadata
            )
            # Applying domain-specific calibration

            # Store original data range for reference
            orig_min = float(np.min(image))
            orig_max = float(np.max(image))

            # Apply domain-specific calibration
            if domain == "photography":
                # Apply physics-based calibration: ADU → electrons (exact match)
                calibrated_image = self._apply_physics_calibration(
                    image, gain, read_noise, domain
                )
                # Physics calibration applied
            else:
                # Apply physics-based calibration: ADU → electrons → normalized
                calibrated_image = self._apply_physics_calibration(
                    image, gain, read_noise, domain
                )
                # Physics calibration applied

            # Store electron range for physics analysis
            electron_min = float(np.min(calibrated_image))
            electron_max = float(np.max(calibrated_image))
            electron_mean = float(np.mean(calibrated_image))
            electron_std = float(np.std(calibrated_image))

            # === STEP 3: Apply domain-specific scaling for dynamic range compression ===
            if domain == "photography":
                # For photography, use linear scaling (exact match to working script)
                img_min, img_max = float(np.min(calibrated_image)), float(
                    np.max(calibrated_image)
                )
                if img_max > img_min:
                    scaled_image = (calibrated_image - img_min) / (img_max - img_min)
                else:
                    scaled_image = np.clip(calibrated_image, 0, 1)
                # Linear scaling applied
            elif domain == "astronomy":
                # Use professional astronomy asinh preprocessing
                try:
                    from astronomy_asinh_preprocessing import (
                        AsinhScalingConfig,
                        AstronomyAsinhPreprocessor,
                    )

                    # Create astronomy preprocessor with optimized config
                    config = AsinhScalingConfig(
                        softening_factor=1e-3,  # β parameter for asinh
                        scale_percentile=99.5,  # Capture bright features
                        background_percentile=10.0,  # Robust background estimation
                        cosmic_ray_threshold=5.0,  # Detect cosmic rays
                    )
                    astro_preprocessor = AstronomyAsinhPreprocessor(config)

                    # Prepare calibration parameters
                    calibration_dict = {
                        "gain": gain,
                        "read_noise": read_noise,
                        "background": 0.0,  # Will be estimated
                        "scale_factor": None,  # Will be computed
                    }

                    # Apply professional astronomy preprocessing
                    result = astro_preprocessor.preprocess_astronomy_image(
                        calibrated_image,
                        calibration_dict,
                        apply_cosmic_ray_removal=True,
                    )

                    # Get the properly scaled image
                    scaled_image = result["preprocessed_image"]
                    preprocessing_metadata = result["preprocessing_metadata"]

                    # Update calibration method to reflect professional preprocessing
                    calibration_method = "astronomy_asinh"

                    logger.info(f"✅ Applied professional astronomy asinh scaling")

                except ImportError:
                    # Fallback to simple astronomical scaling if module not available
                    logger.warning(
                        "⚠️ astronomy_asinh_preprocessing module not available, using fallback"
                    )
                    image_flat = calibrated_image.flatten()
                    image_flat = image_flat[image_flat > 0]

                    if len(image_flat) > 0:
                        p1 = np.percentile(image_flat, 1)
                        softening = max(p1, 1e-6)
                        scaled_image = np.arcsinh(
                            calibrated_image / (softening + 1e-10)
                        )
                        scaled_image = scaled_image * 2.0
                        # Fallback astronomical scaling applied
                    else:
                        scaled_image = calibrated_image
                        logger.warning(
                            "⚠️ No positive values found for astronomical scaling"
                        )
            else:
                # For microscopy, use percentile-based for very dark images
                img_min, img_max = float(np.min(calibrated_image)), float(
                    np.max(calibrated_image)
                )
                img_mean = float(np.mean(calibrated_image))

                # If image is very dark (mean < 1% of max), use percentile-based normalization
                if img_mean < img_max * 0.01 and img_max > 0:
                    logger.warning(
                        f"Very dark image detected, using percentile normalization"
                    )
                    # Use 1st and 99th percentiles for better contrast
                    p1 = np.percentile(calibrated_image, 1)
                    p99 = np.percentile(calibrated_image, 99)
                    if p99 > p1:
                        scaled_image = np.clip(
                            (calibrated_image - p1) / (p99 - p1), 0, 1
                        )
                        # Percentile normalization applied
                    else:
                        scaled_image = np.clip(
                            calibrated_image / (img_max + 1e-8), 0, 1
                        )
                        # Simple max normalization applied
                else:
                    # Standard normalization
                    if img_max > img_min:
                        scaled_image = (calibrated_image - img_min) / (
                            img_max - img_min
                        )
                    else:
                        scaled_image = np.clip(calibrated_image, 0, 1)
                    # Simple normalization applied

            # === STEP 4: Normalize and convert to 8-bit [0,255] ===
            img_min = float(np.min(scaled_image))
            img_max = float(np.max(scaled_image))

            if img_max > img_min and np.isfinite(img_min) and np.isfinite(img_max):
                normalized_image = (scaled_image - img_min) / (img_max - img_min)
            else:
                logger.warning(
                    f"Invalid range after scaling: [{img_min}, {img_max}], using fallback"
                )
                normalized_image = np.clip(scaled_image, 0, 1)

            # Convert to 8-bit [0, 255]
            image_8bit = np.clip(normalized_image * 255.0, 0, 255).astype(np.uint8)
            # Converted to 8-bit

            # === STEP 5: NOW tile the calibrated 8-bit image ===
            # Convert back to CHW format if needed (tiler expects this)
            if len(image_8bit.shape) == 2:
                image_8bit = image_8bit[np.newaxis, :, :]  # Add channel dimension
            elif len(image_8bit.shape) == 3 and image_8bit.shape[2] == 3:
                # Convert HWC → CHW for RGB
                image_8bit = image_8bit.transpose(2, 0, 1)

            # Get systematic tiling configuration
            overlap_ratio = self.overlap_ratios.get(domain, 0.1)
            config = SystematicTilingConfig(
                tile_size=self.tile_size,
                overlap_ratio=overlap_ratio,
                coverage_mode="complete",
                edge_handling="pad_reflect",
                min_valid_ratio=0.5,
            )

            # Extract tiles from calibrated 8-bit image
            # Use custom tiling for ALL domains to ensure proper overlap and NO padding
            if domain == "photography":
                file_path_str = metadata.get("file_path", "")
                if file_path_str.endswith(".ARW"):
                    # Sony files: custom tiling to get exactly 54 tiles (6×9)
                    tile_infos = self._extract_sony_tiles(image_8bit)
                elif file_path_str.endswith(".RAF"):
                    # Fuji files: custom tiling to get exactly 24 tiles (4×6)
                    tile_infos = self._extract_fuji_tiles(image_8bit)
                else:
                    # Fallback to Sony custom tiling
                    tile_infos = self._extract_sony_tiles(image_8bit)
            elif domain == "astronomy":
                # Custom tiling for astronomy to get exactly 81 tiles (9×9)
                tile_infos = self._extract_astronomy_tiles(image_8bit)
            elif domain == "microscopy":
                # Custom tiling for microscopy to get exactly 16 tiles (4×4)
                tile_infos = self._extract_microscopy_tiles(image_8bit)
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
                tile_infos = tiler.extract_tiles(image_8bit)

            if not tile_infos:
                logger.warning(f"No tiles extracted from {file_path}")
                return []

            # Tiles extracted

            # === STEP 6-8: Process tiles and separate prior/posterior, assign splits ===
            processed_tiles = []
            # Note: In the new workflow, calibration happens to the FULL image before tiling,
            # so all tiles are already calibrated. No separate raw_tiles/calibrated_tiles stages.
            max_tiles = len(tile_infos)  # Process ALL tiles - no limit

            # Determine data type (prior=clean vs posterior=noisy)
            data_type = self._determine_data_type(file_path, domain)

            # Get scene ID and assign train/test/val split
            scene_id = self._get_scene_id(file_path, domain)
            split = self._assign_split(scene_id, data_type)

            # File classified and split assigned

            for i in range(max_tiles):
                tile_info = tile_infos[i]
                try:
                    # Get 8-bit tile data (already calibrated and converted)
                    tile_8bit = tile_info.tile_data

                    # Validate tile data
                    if tile_8bit is None or tile_8bit.size == 0:
                        logger.warning(f"Empty tile {i} from {file_path}")
                        continue

                    # Tile is already 8-bit [0, 255], convert to float32 [0,1] for storage metadata
                    tile_float = tile_8bit.astype(np.float32) / 255.0

                    # Ensure proper shape
                    if len(tile_float.shape) == 2:
                        tile_float = tile_float[np.newaxis, :, :]

                    # For photography, apply basic demosaicing (no white balance correction)
                    if domain == "photography" and tile_float.shape[0] == 4:
                        # Apply basic demosaicing without white balance correction
                        rgb_tile = self.demosaic_rggb_to_rgb(tile_float)
                        tile_to_save = rgb_tile
                        # Demosaicing applied
                    else:
                        tile_to_save = tile_float

                    # Save as PNG
                    tile_id = f"{domain}_{Path(file_path).stem}_tile_{i:04d}"
                    tile_metadata = self.save_tile_as_png(
                        tile_to_save,
                        tile_id,
                        domain,
                        data_type,
                        gain,
                        read_noise,
                        calibration_method,
                    )

                    if tile_metadata:
                        # Add comprehensive metadata for PNG tile storage
                        tile_metadata.update(
                            {
                                "tile_size": self.tile_size,
                                "grid_x": int(tile_info.grid_position[0]),
                                "grid_y": int(tile_info.grid_position[1]),
                                "image_x": int(tile_info.image_position[0]),
                                "image_y": int(tile_info.image_position[1]),
                                "channels": int(tile_float.shape[0]),
                                "quality_score": float(np.mean(tile_float))
                                if np.isfinite(np.mean(tile_float))
                                else 0.0,
                                "valid_ratio": float(tile_info.valid_ratio),
                                "is_edge_tile": bool(tile_info.is_edge_tile),
                                "overlap_ratio": float(
                                    self.overlap_ratios.get(domain, 0.1)
                                ),
                                "systematic_coverage": True,
                                "original_min": float(orig_min),
                                "original_max": float(orig_max),
                                "electron_min": float(electron_min),
                                "electron_max": float(electron_max),
                                "electron_mean": float(electron_mean),
                                "electron_std": float(electron_std),
                                "norm_min": float(np.min(tile_float))
                                if np.isfinite(np.min(tile_float))
                                else 0.0,
                                "norm_max": float(np.max(tile_float))
                                if np.isfinite(np.max(tile_float))
                                else 1.0,
                                "norm_mean": float(np.mean(tile_float))
                                if np.isfinite(np.mean(tile_float))
                                else 0.5,
                                "norm_std": float(np.std(tile_float))
                                if np.isfinite(np.std(tile_float))
                                else 0.1,
                                "split": split,
                                "scene_id": scene_id,
                            }
                        )
                        processed_tiles.append(tile_metadata)

                except Exception as e:
                    logger.error(f"Failed to process tile {i} from {file_path}: {e}")
                    continue

            logger.info(
                f"Generated {len(processed_tiles)} tiles from {Path(file_path).name}"
            )

            return processed_tiles

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return []

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

    def run_png_tiles_pipeline(self, max_files_per_domain: int = None):
        """Run the complete PNG tiles pipeline"""

        # Starting PNG Tiles Pipeline

        # Define ALL files for each domain
        sample_files = {"photography": [], "microscopy": [], "astronomy": []}

        # Find ALL photography files and select clean/noisy pairs
        # Discovering photography files
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
        # Found photography file pairs

        # Find ALL microscopy files and select clean/noisy pairs
        # Discovering microscopy files
        all_microscopy_files = list(
            Path(
                "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/microscopy"
            ).rglob("*.mrc")
        )
        sample_files["microscopy"] = self._select_microscopy_file_pairs(
            all_microscopy_files
        )
        # Found microscopy file pairs

        # Find ALL astronomy files
        # Discovering astronomy files
        astronomy_files = list(
            Path(
                "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/astronomy"
            ).rglob("*.fits")
        )
        sample_files["astronomy"] = [str(f) for f in astronomy_files]
        # Found astronomy files

        all_tiles_metadata = []
        results = {"domains": {}, "total_tiles": 0}

        for domain_name, file_list in sample_files.items():
            # Processing domain

            domain_tiles = []
            processed_files = 0

            for file_path in file_list[: max_files_per_domain or len(file_list)]:
                if not Path(file_path).exists():
                    # File not found
                    continue

                try:
                    tiles = self.process_file_to_png_tiles(file_path, domain_name)
                    domain_tiles.extend(tiles)
                    processed_files += 1
                except Exception as e:
                    logger.error(f"❌ Error processing {file_path}: {e}")
                    continue

            all_tiles_metadata.extend(domain_tiles)
            results["domains"][domain_name] = {
                "files_processed": processed_files,
                "tiles_generated": len(domain_tiles),
            }
            results["total_tiles"] += len(domain_tiles)

            # Generated tiles from files

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
            },
            "tiles": all_tiles_metadata,
        }

        with open(metadata_path, "w") as f:
            json.dump(comprehensive_metadata, f, indent=2, default=str)

        logger.info(
            f"PNG Tiles Pipeline Completed: {results['total_tiles']:,} tiles generated"
        )
        logger.info(f"📊 Comprehensive metadata saved to: {metadata_path}")

        return results

    def load_microscopy_mrc(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load microscopy MRC file using the BioSR reader"""
        if not MRC_READER_AVAILABLE:
            raise ImportError("BioSR MRC reader not available")

        try:
            # Loading MRC file
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

    def _get_physics_based_calibration(
        self, domain: str, metadata: Dict[str, Any] = None
    ) -> Tuple[float, float, str]:
        """Get domain-specific physics-based calibration parameters"""
        # Extract ISO and camera type from filename for SID dataset
        if domain == "photography" and metadata and "file_path" in metadata:
            file_path = metadata["file_path"]
            if file_path.endswith(".ARW"):
                # Sony files are typically ISO 200
                metadata["iso"] = 200
                metadata["camera_type"] = "sony"
            elif file_path.endswith(".RAF"):
                # Fuji files are typically ISO 1000 (from SID dataset)
                metadata["iso"] = 1000
                metadata["camera_type"] = "fuji"
        return get_physics_based_calibration(domain, metadata)

    def _determine_data_type(self, file_path: str, domain: str) -> str:
        """Determine if file contains clean or noisy data"""

        file_path_lower = file_path.lower()

        if domain == "photography":
            return "clean" if "/long/" in file_path_lower else "noisy"
        elif domain == "microscopy":
            return "clean" if "gt" in Path(file_path).name.lower() else "noisy"
        elif domain == "astronomy":
            # Hubble Legacy Archive:
            # - Direct image (detection_sci) = CLEAN reference (high SNR photometry)
            # - G800L grism (g800l_sci) = NOISY (spectroscopic with artifacts)
            return "clean" if "detection" in Path(file_path).name.lower() else "noisy"

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
        seed = int(hashlib.md5(scene_id.encode()).hexdigest(), 16) % (2**32)
        random.seed(seed)

        # Random assignment with proper distribution - SAME FOR ALL DATA TYPES IN SCENE
        split_val = random.random() * 100  # 0-100

        if split_val < 70:
            return "train"
        elif split_val < 85:
            return "validation"
        else:
            return "test"

    def _apply_physics_calibration(
        self, image: np.ndarray, gain: float, read_noise: float, domain: str
    ) -> np.ndarray:
        """
        Apply physics-based calibration to convert sensor ADU values to calibrated electrons

        Process:
        1. Convert ADU (Analog-to-Digital Units) to electrons using gain
        2. Apply read noise correction
        3. Clip negative values (physical constraint)

        Args:
            image: Raw image in ADU units (from sensor)
            gain: Conversion factor (electrons/ADU)
            read_noise: Read noise in electrons
            domain: Domain name for domain-specific processing

        Returns:
            Calibrated image in electron units
        """
        try:
            # Convert ADU to electrons
            # Formula: electrons = ADU × gain
            image_electrons = image.astype(np.float32) * gain

            # Apply read noise correction (subtract noise floor)
            # For very low signals, we need to account for read noise
            image_calibrated = image_electrons - read_noise

            # Don't clip negative values - allow them to preserve the full dynamic range
            # This matches the working approach from visualize_calibration_standalone.py
            # image_calibrated = np.maximum(image_calibrated, 0.0)

            # Calibration applied

            return image_calibrated

        except Exception as e:
            logger.error(f"Physics calibration failed: {e}, using original image")
            return image.astype(np.float32)

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
            logger.info(
                f"   ✅ Sony: {len(tiles)} tiles (256×256, overlap: {overlap_h}×{overlap_w}px)"
            )
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
            logger.info(
                f"   ✅ Fuji: {len(tiles)} tiles (256×256, overlap: {overlap_h}×{overlap_w}px)"
            )
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
            logger.info(
                f"   ✅ Astronomy: {len(tiles)} tiles (256×256, overlap: {overlap_h}×{overlap_w}px)"
            )
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
            logger.info(
                f"   ✅ Microscopy: {len(tiles)} tiles (256×256, overlap: {overlap_h}×{overlap_w}px)"
            )
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
        description="Simple PNG Tiles Pipeline with Domain-Specific Calibration"
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

    args = parser.parse_args()

    # Run PNG tiles pipeline
    pipeline = SimpleTilesPipeline(args.base_path)
    results = pipeline.run_png_tiles_pipeline(args.max_files)

    if results.get("total_tiles", 0) > 0:
        print(f"\n🎊 SUCCESS: PNG Tiles Pipeline Completed!")
        print(f"📊 Total PNG tiles generated: {results['total_tiles']:,}")
        print(f"📐 Tile size: 256×256")
        print(f"💾 PNG files saved to: {args.base_path}/processed/png_tiles/")

        print(f"\n📋 Domain Results:")
        for domain, stats in results.get("domains", {}).items():
            print(
                f"   • {domain.upper()}: {stats['tiles_generated']} tiles from {stats['files_processed']} files"
            )

        print(f"\n🎯 Ready for diffusion model training!")
    else:
        print(f"\n❌ FAILED: No tiles were generated")
        print(f"Total tiles processed: {results.get('total_tiles', 0)}")


def get_physics_based_calibration(
    domain: str, metadata: Dict[str, Any] = None
) -> Tuple[float, float, str]:
    """Get domain-specific physics-based calibration parameters"""

    if domain == "photography":
        # Check camera type from metadata if available
        camera_type = metadata.get("camera_type", "sony") if metadata else "sony"

        if camera_type == "sony":
            # Sony A7S II parameters (corrected values)
            iso = metadata.get("iso", 200) if metadata else 200
            if iso is not None and iso >= 4000:
                gain = 0.79  # e-/ADU at ISO 4000 (unity gain)
                read_noise = 2.5  # electrons above ISO 4000
            elif iso >= 200:
                gain = 5.0  # e-/ADU at ISO 200 (1/0.198 DN/e⁻ = ~5 e-/DN)
                read_noise = 3.56  # electrons RMS at ISO 200
            else:
                gain = 2.1  # e-/ADU fallback
                read_noise = 6.0  # electrons fallback
            method = "photon_transfer_curve"

        elif camera_type == "fuji":
            # Fuji X-Trans parameters (corrected values for X-T3/X-T4)
            iso = metadata.get("iso", 1000) if metadata else 1000
            if iso >= 1000:
                gain = 1.8  # e-/ADU at ISO 1000 (1/0.55 DN/e⁻ = ~1.8 e-/DN)
                read_noise = (
                    3.75  # electrons RMS at ISO 1000 (average of 3.5-4.0 range)
                )
            elif iso >= 200:
                gain = 1.8  # e-/ADU at ISO 200 (same as 1000 for Fuji)
                read_noise = 3.75  # electrons RMS at ISO 200
            else:
                gain = 0.75  # e-/ADU fallback
                read_noise = 2.5  # electrons fallback
            method = "photon_transfer_curve"

        else:
            # Default fallback
            gain = 1.0
            read_noise = 3.0
            method = "default"

    elif domain == "microscopy":
        # BioSR sCMOS parameters (measured values)
        noise_level = metadata.get("noise_level", 5) if metadata else 5

        # Scale gain and read noise based on noise level (1-9 scale)
        # Higher noise level = lower gain, higher read noise
        gain = 1.0 + (noise_level - 5) * 0.1  # Range: 0.6-1.4
        read_noise = 1.5 + (noise_level - 5) * 0.2  # Range: 0.5-2.5

        method = "noise_level_analysis"

    elif domain == "astronomy":
        # HST instrument parameters (measured values)
        if metadata:
            instrument_name = metadata.get("instrument", "ACS")
            detector_name = metadata.get("detector", "WFC")
            # Construct proper instrument key (e.g., "ACS_WFC", "WFC3_UVIS")
            instrument = f"{instrument_name}_{detector_name}"
        else:
            instrument = "ACS_WFC"

        # Instrument-specific calibration parameters
        instrument_params = {
            "ACS_WFC": {
                "gain": 1.0,
                "read_noise": 3.5,
                "method": "fits_header_analysis",
            },
            "WFC3_UVIS": {
                "gain": 1.5,
                "read_noise": 3.09,
                "method": "fits_header_analysis",
            },
            "WFC3_IR": {
                "gain": 2.5,
                "read_noise": 15.0,
                "method": "fits_header_analysis",
            },
            "WFPC2_WF": {
                "gain": 7.0,
                "read_noise": 6.5,
                "method": "fits_header_analysis",
            },
            "WFPC2_PC": {
                "gain": 7.0,
                "read_noise": 6.5,
                "method": "fits_header_analysis",
            },
        }

        if instrument in instrument_params:
            params = instrument_params[instrument]
            gain = params["gain"]
            read_noise = params["read_noise"]
            method = params["method"]
        else:
            # Default for unknown instruments
            gain = 1.0
            read_noise = 3.0
            method = "default"

    else:
        # Fallback for unknown domains
        gain = 1.0
        read_noise = 1.0
        method = "default"

    return gain, read_noise, method


def get_calibration_description(
    domain: str, gain: float, read_noise: float, method: str
) -> str:
    """Get human-readable description of calibration parameters"""
    if domain == "photography":
        return f"Camera sensor calibration (Gain: {gain:.2f} e-/ADU, Read noise: {read_noise:.1f} e-)"
    elif domain == "microscopy":
        return f"sCMOS detector calibration (Gain: {gain:.2f} e-/ADU, Read noise: {read_noise:.1f} e-)"
    elif domain == "astronomy":
        return f"HST instrument calibration (Gain: {gain:.1f} e-/ADU, Read noise: {read_noise:.1f} e-)"
    else:
        return f"Generic calibration (Gain: {gain:.2f} e-/ADU, Read noise: {read_noise:.1f} e-)"


if __name__ == "__main__":
    # Start the PNG tiles pipeline
    logger.info("🚀 Starting PNG tiles pipeline...")
    main()
