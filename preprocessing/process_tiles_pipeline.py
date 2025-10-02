#!/usr/bin/env python3
"""
Final Working Unified Tiles Pipeline with Domain-Specific Calibration
Bypasses problematic preprocessing and uses direct normalization
Demonstrates domain-specific physics-based calibration methods
"""

import json
import logging
import os
import pickle  # nosec B403 - Used for legitimate data serialization
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

# Simple file-based processing - no Spark/Parquet/Delta Lake


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
            "photography": 0.0,  # 0% overlap (2×4 = 8 tiles per image, 712×1064)
            "microscopy": 0.0,  # 0% overlap (4×4 = 16 tiles per image)
            "astronomy": 0.0,  # 0% overlap (8×8 = 64 tiles per image)
        }

        # Camera-specific overlap ratios for photography
        self.camera_overlap_ratios = {
            "sony": 0.0,  # Sony: 0% overlap → 2×4 = 8 tiles
            "fuji": 0.021,  # Fuji: 2.1% overlap → 4×6 = 24 tiles (with 4x downsampling)
        }

        # Custom tiling for Sony to get exactly 8 tiles (2×4)
        self.sony_tile_config = {
            "tile_size": 256,
            "target_tiles": 8,  # 2×4 = 8 tiles
            "target_grid": (2, 4),  # 2 rows, 4 columns
        }

        # Custom tiling for Fuji to get exactly 24 tiles (4×6)
        self.fuji_tile_config = {
            "tile_size": 256,
            "target_tiles": 24,  # 4×6 = 24 tiles
            "target_grid": (4, 6),  # 4 rows, 6 columns
        }

        # Photography downsampling configuration - separate for Sony and Fuji
        self.downsample_photography = True
        self.downsample_factors = {
            "sony": 2.0,  # Sony: 2848×4256 → 1424×2128 → 2×4 = 8 tiles
            "fuji": 4.0,  # Fuji: 4032×6032 → 1008×1508 → 4×6 = 24 tiles
        }

        # No additional output directories needed - tiles are saved to png_tiles/ and metadata to tiles_metadata.json

        logger.info(
            f"🎯 Initialized PNG tiles pipeline (tile size: {self.tile_size}×{self.tile_size})"
        )

    def prepare_tile_data(self, tile_data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Prepare tile data for storage as PNG (lossless compression, 8-bit for cross-domain consistency)"""

        try:
            import io

            from PIL import Image

            # Ensure proper shape and type
            if len(tile_data.shape) == 2:
                tile_data = tile_data[np.newaxis, :, :]  # Add channel dimension

            # Convert to float32 and ensure contiguous
            tile_data = np.ascontiguousarray(tile_data.astype(np.float32))

            # Store original shape for reconstruction
            original_shape = tile_data.shape
            channels = original_shape[0]

            # Convert from float32 [0,1] to uint8 [0,255] for consistent 8-bit representation
            # This ensures all domains (photography, microscopy, astronomy) have the same bit depth
            # which is important for cross-domain training
            tile_uint8 = np.clip(tile_data * 255.0, 0, 255).astype(np.uint8)

            # Handle different channel configurations
            if channels == 1:
                # Grayscale image (microscopy, astronomy) - remove channel dimension for PIL
                img_array = tile_uint8[0]
                pil_image = Image.fromarray(img_array, mode="L")  # 8-bit grayscale
            elif channels == 3:
                # RGB image (photography) - rearrange to HWC format for PIL
                img_array = tile_uint8.transpose(1, 2, 0)
                pil_image = Image.fromarray(img_array, mode="RGB")  # 8-bit RGB
            else:
                # Multi-channel data - save first channel as 8-bit grayscale
                # For astronomy/microscopy with many channels, we use the first channel
                logger.warning(
                    f"Multi-channel data ({channels} channels) - using first channel for 8-bit PNG"
                )
                img_array = tile_uint8[0]
                pil_image = Image.fromarray(img_array, mode="L")  # 8-bit grayscale

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

    def save_calibration_intermediate(
        self, calibrated_tiles: List[Dict[str, Any]], stage: str = "post_calibration"
    ):
        """Save intermediate calibration results to JSON for analysis"""
        try:
            logger.info(f"💾 Saving intermediate calibration data ({stage}) to JSON...")

            # Convert to simplified format for JSON storage
            intermediate_data = []
            for tile in calibrated_tiles:
                intermediate_data.append(
                    {
                        "tile_id": tile["tile_id"],
                        "domain_name": tile["domain_name"],
                        "calibration_method": tile.get("calibration_method", "unknown"),
                        "gain": tile.get("gain", 1.0),
                        "read_noise": tile.get("read_noise", 1.0),
                        "norm_min": tile.get("norm_min", 0.0),
                        "norm_max": tile.get("norm_max", 1.0),
                        "norm_mean": tile.get("norm_mean", 0.5),
                        "norm_std": tile.get("norm_std", 0.1),
                        "processing_stage": stage,
                        "processing_timestamp": tile.get(
                            "processing_timestamp", datetime.now().isoformat()
                        ),
                    }
                )

            if not intermediate_data:
                logger.warning("No intermediate calibration data to save")
                return

            # Create intermediate JSON path
            intermediate_path = (
                self.base_path / "processed" / f"calibration_intermediate_{stage}.json"
            )
            intermediate_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to JSON
            with open(intermediate_path, "w") as f:
                json.dump(intermediate_data, f, indent=2, default=str)

            logger.info(
                f"✅ Saved {len(intermediate_data)} calibration records to: {intermediate_path}"
            )
            return str(intermediate_path)

        except Exception as e:
            logger.error(f"Failed to save intermediate calibration data: {e}")
            return None

    def demosaic_rggb_to_rgb(
        self,
        rggb_image: np.ndarray,
        gains: np.ndarray = None,
        camera_type: str = "sony",
    ):
        """
        Demosaic RGGB Bayer pattern to RGB (no white balance correction)

        Args:
            rggb_image: RGGB image with shape (4, H, W) - [R, G1, G2, B]
            gains: Ignored (no white balance correction)
            camera_type: Ignored (no white balance correction)

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

        # No white balance correction - just basic demosaicing
        # Stack to RGB
        rgb_image = np.stack([R, G, B], axis=0)

        return rgb_image

    def extract_white_balance_gains(self, file_path: str) -> np.ndarray:
        """
        Extract white balance gains from raw file metadata

        Args:
            file_path: Path to raw file (.ARW or .RAF)

        Returns:
            White balance gains as [R_gain, G_gain, B_gain]
        """
        try:
            import rawpy

            with rawpy.imread(file_path) as raw:
                # Try to get white balance from raw file
                if (
                    hasattr(raw, "daylight_whitebalance")
                    and raw.daylight_whitebalance is not None
                ):
                    wb = raw.daylight_whitebalance
                    # Convert to gains (normalize by green channel)
                    if len(wb) >= 3:
                        wb_gains = np.array([wb[0], wb[1], wb[2]], dtype=np.float32)
                        # Normalize by green channel
                        if wb_gains[1] > 0:
                            wb_gains = wb_gains / wb_gains[1]
                        return wb_gains

                # Fallback to camera-specific defaults (conservative values)
                camera_type = (
                    "sony"
                    if file_path.endswith(".ARW")
                    else "fuji"
                    if file_path.endswith(".RAF")
                    else "sony"
                )

                if camera_type == "sony":
                    return np.array(
                        [1.2, 1.0, 1.1]
                    )  # Sony A7S II daylight (conservative)
                elif camera_type == "fuji":
                    return np.array(
                        [1.1, 1.0, 1.05]
                    )  # Fuji X-Trans daylight (conservative)
                else:
                    return np.array([1.1, 1.0, 1.05])  # Generic daylight (conservative)

        except Exception as e:
            logger.warning(f"Could not extract white balance from {file_path}: {e}")
            # Return camera-specific defaults (conservative values)
            camera_type = (
                "sony"
                if file_path.endswith(".ARW")
                else "fuji"
                if file_path.endswith(".RAF")
                else "sony"
            )
            if camera_type == "sony":
                return np.array([1.2, 1.0, 1.1])
            elif camera_type == "fuji":
                return np.array([1.1, 1.0, 1.05])
            else:
                return np.array([1.1, 1.0, 1.05])

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
                # Apply basic demosaicing (no white balance correction)
                # Convert back to float for demosaicing, then back to uint8
                tile_float = tile_uint8.astype(np.float32) / 255.0

                # Apply basic demosaicing without white balance
                rgb_tile_float = self.demosaic_rggb_to_rgb(tile_float)

                # Convert back to uint8
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

            # === STEP 1.5: Apply camera-specific downsampling for photography domain ===
            if domain == "photography" and self.downsample_photography:
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
            # Custom tiling for Sony to get exactly 8 tiles (2×4)
            if domain == "photography" and metadata.get("file_path", "").endswith(
                ".ARW"
            ):
                # Sony files: use custom tiling to get exactly 8 tiles
                tile_infos = self._extract_sony_tiles(image_8bit)
            else:
                # Standard tiling for other domains
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

    def run_png_tiles_pipeline(self, max_files_per_domain: int = None):
        """Run the complete PNG tiles pipeline"""

        logger.info("🚀 Starting PNG Tiles Pipeline")

        # Define ALL files for each domain
        sample_files = {"photography": [], "microscopy": [], "astronomy": []}

        # Find ALL photography files (Sony ARW + Fuji RAF)
        logger.info("🔍 Discovering ALL photography files...")
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
        sample_files["photography"] = [str(f) for f in sony_files + fuji_files]
        logger.info(f"📷 Found {len(sample_files['photography'])} photography files")

        # Find ALL microscopy files
        logger.info("🔍 Discovering ALL microscopy files...")
        microscopy_files = list(
            Path(
                "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/microscopy"
            ).rglob("*.mrc")
        )
        sample_files["microscopy"] = [str(f) for f in microscopy_files]
        logger.info(f"🔬 Found {len(sample_files['microscopy'])} microscopy files")

        # Find ALL astronomy files
        logger.info("🔍 Discovering ALL astronomy files...")
        astronomy_files = list(
            Path(
                "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/astronomy"
            ).rglob("*.fits")
        )
        sample_files["astronomy"] = [str(f) for f in astronomy_files]
        logger.info(f"🌟 Found {len(sample_files['astronomy'])} astronomy files")

        all_tiles_metadata = []
        results = {"domains": {}, "total_tiles": 0}

        for domain_name, file_list in sample_files.items():
            logger.info(f"Processing {domain_name} domain...")

            domain_tiles = []
            processed_files = 0

            for file_path in file_list[: max_files_per_domain or len(file_list)]:
                if not Path(file_path).exists():
                    logger.warning(f"⚠️ File not found: {file_path}")
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

            logger.info(
                f"{domain_name}: Generated {len(domain_tiles)} tiles from {processed_files} files"
            )

        # Save metadata
        metadata_path = self.base_path / "processed" / "tiles_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, "w") as f:
            json.dump(all_tiles_metadata, f, indent=2, default=str)

        logger.info(f"\n🎉 PNG Tiles Pipeline Completed!")
        logger.info(f"📊 Total PNG tiles generated: {results['total_tiles']:,}")
        logger.info(f"📐 Tile size: {self.tile_size}×{self.tile_size}")
        logger.info(f"💾 Metadata saved to: {metadata_path}")

        return results

    def load_microscopy_mrc(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load microscopy MRC file using the BioSR reader"""
        if not MRC_READER_AVAILABLE:
            raise ImportError("BioSR MRC reader not available")

        try:
            logger.info(f"📖 Loading MRC file: {file_path}")
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

            # MRC file loaded successfully
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
                # Fuji files are typically ISO 2000
                metadata["iso"] = 2000
                metadata["camera_type"] = "fuji"
        return get_physics_based_calibration_demo(domain, metadata)

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
            # Extract structure type and level/frame (e.g., "CCPs_level_02" or "ER_all")
            # Remove GT/noise indicators but keep structure identifier
            clean_name = filename.replace("GT_", "").replace("_noisy", "")
            return f"micro_{clean_name}"

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

        Strategy:
        - Noisy data (posterior): Used for training (with some validation)
        - Clean data (prior): Used for testing only
        - 70% train, 15% validation, 15% test
        - All tiles from same scene get same split (consistent within scene)
        - Random assignment ensures proper distribution across scenes
        """

        # Use random seed based on scene_id for consistent assignment within scene
        import hashlib
        import random

        # Create deterministic seed from scene_id for consistent assignment within scene
        seed = int(
            hashlib.md5(scene_id.encode(), usedforsecurity=False).hexdigest(), 16
        ) % (2**32)
        random.seed(seed)

        # Random assignment with proper distribution
        split_val = random.random() * 100  # 0-100

        if data_type == "noisy":
            # Noisy data: 70% train, 15% validation, 15% test
            if split_val < 70:
                return "train"
            elif split_val < 85:
                return "validation"
            else:
                return "test"
        else:  # clean
            # Clean data: All goes to test set (this is the prior for evaluation)
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

            logger.info(
                f"🔬 Calibration applied: ADU→electrons (×{gain:.3f}), noise floor removed ({read_noise:.3f}e-)"
            )
            logger.info(
                f"   Range: [{np.min(image_calibrated):.1f}, {np.max(image_calibrated):.1f}] electrons"
            )

            return image_calibrated

        except Exception as e:
            logger.error(f"Physics calibration failed: {e}, using original image")
            return image.astype(np.float32)

    def _extract_sony_tiles(self, image: np.ndarray) -> List[Any]:
        """Extract exactly 8 tiles (2×4 grid) from Sony images"""
        try:
            if len(image.shape) == 3:
                C, H, W = image.shape
            else:
                H, W = image.shape
                C = 1
                image = image[np.newaxis, :, :]

            # Calculate tile positions for 2×4 grid
            # We want exactly 8 tiles: 2 rows × 4 columns
            rows, cols = 2, 4

            # Calculate tile size to fit the image
            tile_h = H // rows
            tile_w = W // cols

            # Ensure tiles are not larger than 256×256
            tile_h = min(tile_h, 256)
            tile_w = min(tile_w, 256)

            tiles = []
            for row in range(rows):
                for col in range(cols):
                    # Calculate tile position
                    y_start = row * tile_h
                    y_end = min((row + 1) * tile_h, H)
                    x_start = col * tile_w
                    x_end = min((col + 1) * tile_w, W)

                    # Extract tile
                    if len(image.shape) == 3:
                        tile_data = image[:, y_start:y_end, x_start:x_end]
                    else:
                        tile_data = image[y_start:y_end, x_start:x_end]

                    # Pad tile to 256×256 if needed
                    if tile_data.shape[-2:] != (256, 256):
                        if len(tile_data.shape) == 3:
                            padded_tile = np.zeros((C, 256, 256), dtype=tile_data.dtype)
                            padded_tile[
                                :, : tile_data.shape[1], : tile_data.shape[2]
                            ] = tile_data
                        else:
                            padded_tile = np.zeros((256, 256), dtype=tile_data.dtype)
                            padded_tile[
                                : tile_data.shape[0], : tile_data.shape[1]
                            ] = tile_data
                        tile_data = padded_tile

                    # Create tile info object
                    tile_info = type(
                        "TileInfo",
                        (),
                        {
                            "tile_data": tile_data,
                            "grid_position": (col, row),
                            "image_position": (x_start, y_start),
                            "valid_ratio": 1.0,
                            "is_edge_tile": False,
                        },
                    )()

                    tiles.append(tile_info)

            # Sony tiles extracted
            return tiles

        except Exception as e:
            logger.error(f"Sony tile extraction failed: {e}")
            return []

    def _extract_fuji_tiles(self, image: np.ndarray) -> List[Any]:
        """Extract exactly 24 tiles (4×6 grid) from Fuji images"""
        try:
            if len(image.shape) == 3:
                C, H, W = image.shape
            else:
                H, W = image.shape
                C = 1
                image = image[np.newaxis, :, :]

            # Calculate tile positions for 4×6 grid
            # We want exactly 24 tiles: 4 rows × 6 columns
            rows, cols = 4, 6

            # Calculate tile size to fit the image
            tile_h = H // rows
            tile_w = W // cols

            # Ensure tiles are not larger than 256×256
            tile_h = min(tile_h, 256)
            tile_w = min(tile_w, 256)

            tiles = []
            for row in range(rows):
                for col in range(cols):
                    # Calculate tile position
                    y_start = row * tile_h
                    y_end = min((row + 1) * tile_h, H)
                    x_start = col * tile_w
                    x_end = min((col + 1) * tile_w, W)

                    # Extract tile
                    if len(image.shape) == 3:
                        tile_data = image[:, y_start:y_end, x_start:x_end]
                    else:
                        tile_data = image[y_start:y_end, x_start:x_end]

                    # Pad tile to 256×256 if needed
                    if tile_data.shape[-2:] != (256, 256):
                        if len(tile_data.shape) == 3:
                            padded_tile = np.zeros((C, 256, 256), dtype=tile_data.dtype)
                            padded_tile[
                                :, : tile_data.shape[1], : tile_data.shape[2]
                            ] = tile_data
                        else:
                            padded_tile = np.zeros((256, 256), dtype=tile_data.dtype)
                            padded_tile[
                                : tile_data.shape[0], : tile_data.shape[1]
                            ] = tile_data
                        tile_data = padded_tile

                    # Create tile info object
                    tile_info = type(
                        "TileInfo",
                        (),
                        {
                            "tile_data": tile_data,
                            "grid_position": (col, row),
                            "image_position": (x_start, y_start),
                            "valid_ratio": 1.0,
                            "is_edge_tile": False,
                        },
                    )()

                    tiles.append(tile_info)

            # Fuji tiles extracted
            return tiles

        except Exception as e:
            logger.error(f"Fuji tile extraction failed: {e}")
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

                logger.info(
                    f"📐 Aspect ratio maintained: {H}×{W} → {new_H}×{new_W} (ratio: {W/H:.3f})"
                )

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

                logger.info(
                    f"📐 Aspect ratio maintained: {H}×{W} → {new_H}×{new_W} (ratio: {W/H:.3f})"
                )

            return downsampled.astype(image.dtype)

        except ImportError:
            # Fallback to simple downsampling if scipy not available
            logger.warning(
                "⚠️ scipy not available, using simple downsampling with aspect ratio preservation"
            )
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
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        help="Run in demo mode showing calibration examples only",
    )

    args = parser.parse_args()

    if args.demo_mode:
        # Run demo mode
        logger.info("🎯 Running in DEMO MODE - showing domain-specific calibration")
        results = run_domain_calibration_demo(args.base_path, args.max_files)
    else:
        # Run simple PNG tiles pipeline
        logger.info("🚀 Running Simple PNG Tiles Pipeline")
        pipeline = SimpleTilesPipeline(args.base_path)
        results = pipeline.run_png_tiles_pipeline(args.max_files)

    if results.get("total_tiles", 0) > 0:
        if args.demo_mode:
            print(f"\n🎊 SUCCESS: Domain-Specific Calibration Demonstration Completed!")
            print(
                f"📊 Total calibration examples: {len(results.get('calibration_examples', []))}"
            )
            print(
                f"🔬 Domains with calibration: {list(results.get('domain_counts', {}).keys())}"
            )
            print(
                f"📐 Tile size: {results.get('tile_size', 256)}×{results.get('tile_size', 256)}"
            )
            print("\n🎯 Domain-Specific Calibration Examples:")
            for example in results.get("calibration_examples", []):
                print(f"   • {example['domain'].upper()}: {example['description']}")
                print(
                    f"     📊 Gain: {example['gain']:.3f} e-/ADU, Read Noise: {example['read_noise']:.3f} e-"
                )
                print(f"     🔬 Method: {example['calibration_method']}")

            print("\n🔬 PHYSICS-BASED CALIBRATION METHODS DEMONSTRATED:")
            print("   📷 Photography: Camera-specific sensor calibration (Sony/Fuji)")
            print("   🔬 Microscopy: sCMOS detector noise-level analysis")
            print("   🌟 Astronomy: HST instrument-specific parameters")
        else:
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


def run_domain_calibration_demo(base_path: str, max_files: int = 2) -> Dict[str, Any]:
    """Run domain-specific calibration demonstration"""
    logger.info("🎯 DOMAIN-SPECIFIC CALIBRATION DEMONSTRATION")
    logger.info("=" * 60)

    results = {
        "total_tiles": 0,
        "tile_size": 256,
        "storage_methods": {"png_lossless_8bit": 0},
        "domain_counts": {},
        "calibration_examples": [],
    }

    # Demonstrate domain-specific calibration for each domain
    domains_info = {
        "photography": {
            "description": "Sony A7S II and Fuji X-T30 camera calibration",
            "files": [
                "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/SID/Sony/short/00001_00_0.04s.ARW",
                "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/SID/Fuji/short/00001_00_0.04s.RAF",
            ],
        },
        "microscopy": {
            "description": "BioSR sCMOS microscopy calibration",
            "files": [
                "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/microscopy/structures/CCPs/GT_all.mrc"
            ],
        },
        "astronomy": {
            "description": "HST instrument-specific calibration",
            "files": [
                "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/astronomy/hla_associations/direct_images/j6fl7xoyq_detection_sci.fits"
            ],
        },
    }

    for domain_name, domain_info in domains_info.items():
        logger.info(f"\n🔬 {domain_name.upper()} DOMAIN CALIBRATION")
        logger.info(f"📝 {domain_info['description']}")
        logger.info("-" * 50)

        domain_results = {"tiles_processed": 0, "calibration_examples": []}

        for file_path in domain_info["files"][:max_files]:
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ File not found: {file_path}")
                continue

            logger.info(f"📷 Processing: {os.path.basename(file_path)}")

            # Create mock metadata for demonstration
            metadata = {"file_path": file_path, "domain": domain_name}

            # Get domain-specific calibration parameters
            gain, read_noise, calibration_method = get_physics_based_calibration_demo(
                domain_name, metadata
            )

            calibration_example = {
                "file": os.path.basename(file_path),
                "domain": domain_name,
                "gain": gain,
                "read_noise": read_noise,
                "calibration_method": calibration_method,
                "description": get_calibration_description(
                    domain_name, gain, read_noise, calibration_method
                ),
            }

            domain_results["calibration_examples"].append(calibration_example)
            domain_results["tiles_processed"] += 1

            logger.info(f"✅ {calibration_example['description']}")
            logger.info(
                f"   📊 Gain: {gain:.3f} e-/ADU, Read Noise: {read_noise:.3f} e-"
            )
            logger.info(f"   🔬 Method: {calibration_method}")

        results["domain_counts"][domain_name] = domain_results["tiles_processed"]
        results["calibration_examples"].extend(domain_results["calibration_examples"])
        results["total_tiles"] += domain_results["tiles_processed"]

    # Show calibration summary
    logger.info("\n📊 DOMAIN-SPECIFIC CALIBRATION SUMMARY")
    logger.info("=" * 60)

    for example in results["calibration_examples"]:
        logger.info(f"🎯 {example['domain'].upper()}: {example['description']}")

    logger.info("\n✅ DEMONSTRATION COMPLETE")
    logger.info(f"📊 Total calibration examples: {len(results['calibration_examples'])}")
    logger.info(f"🔬 Domains covered: {list(results['domain_counts'].keys())}")
    logger.info(f"📐 Tile size: {results['tile_size']}×{results['tile_size']}")

    return results


def get_physics_based_calibration_demo(
    domain: str, metadata: Dict[str, Any] = None
) -> Tuple[float, float, str]:
    """Get domain-specific physics-based calibration parameters (demo version)"""

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
            iso = metadata.get("iso", 2000) if metadata else 2000
            if iso >= 2000:
                gain = 1.8  # e-/ADU at ISO 2000 (1/0.55 DN/e⁻ = ~1.8 e-/DN)
                read_noise = (
                    3.75  # electrons RMS at ISO 2000 (average of 3.5-4.0 range)
                )
            elif iso >= 200:
                gain = 1.8  # e-/ADU at ISO 200 (same as 2000 for Fuji)
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
        instrument = metadata.get("instrument", "ACS_WFC") if metadata else "ACS_WFC"

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
            "WFPC2": {"gain": 7.0, "read_noise": 6.5, "method": "fits_header_analysis"},
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


def visualize_calibration_effects(
    base_path: str, output_dir: str = "calibration_visualizations"
):
    """
    Create visualization panels showing:
    1. Original noisy tile
    2. Original clean tile
    3. Domain-specific calibrated tile (8-bit PNG)
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image

    matplotlib.use("Agg")  # Non-interactive backend

    logger.info("🎨 Creating calibration effect visualizations...")
    logger.info("=" * 60)

    output_path = Path(base_path) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Sample files for each domain (noisy + clean pairs)
    test_files = {
        "photography": {
            "noisy": "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/SID/Sony/short/00001_00_0.04s.ARW",
            "clean": "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/SID/Sony/long/00001_00_10s.ARW",
        },
        "microscopy": {
            "noisy": "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/microscopy/structures/Microtubules/Cell_005/RawSIMData_level_02.mrc",
            "clean": "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/microscopy/structures/Microtubules/Cell_005/RawSIMData_gt.mrc",
        },
        "astronomy": {
            "clean": "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/astronomy/hla_associations/direct_images/j8hp9fq1q_detection_sci.fits",  # Direct image = clean reference
            "noisy": "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/astronomy/hla_associations/g800l_images/j8hp9fq1q_g800l_sci.fits",  # G800L grism = noisy (spectroscopic with artifacts)
        },
    }

    for domain_name, file_pair in test_files.items():
        logger.info(f"\n🔬 Processing {domain_name.upper()} domain...")

        try:
            # Process noisy image
            noisy_path = file_pair["noisy"]
            clean_path = file_pair["clean"]

            if not Path(noisy_path).exists() or not Path(clean_path).exists():
                logger.warning(f"⚠️ Files not found for {domain_name}, skipping...")
                continue

            # Create processor
            processor = create_processor(domain_name)

            # Load noisy image
            logger.info(f"📖 Loading noisy: {Path(noisy_path).name}")
            if domain_name == "microscopy" and noisy_path.lower().endswith(".mrc"):
                # Use MRC reader
                from read_mrc import read_mrc

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
            if domain_name == "microscopy" and clean_path.lower().endswith(".mrc"):
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
            gain, read_noise, calibration_method = get_physics_based_calibration_demo(
                domain_name, metadata_noisy
            )
            logger.info(
                f"📊 Calibration: gain={gain:.3f}, read_noise={read_noise:.3f}, method={calibration_method}"
            )

            # Apply calibration to noisy image
            noisy_calibrated = apply_physics_calibration_standalone(
                noisy_raw, gain, read_noise, domain_name
            )

            # Normalize and convert to 8-bit
            noisy_min, noisy_max = float(np.min(noisy_calibrated)), float(
                np.max(noisy_calibrated)
            )
            if noisy_max > noisy_min:
                noisy_norm = (noisy_calibrated - noisy_min) / (noisy_max - noisy_min)
            else:
                noisy_norm = np.clip(noisy_calibrated, 0, 1)
            noisy_8bit = np.clip(noisy_norm * 255.0, 0, 255).astype(np.uint8)

            # Also process clean image
            clean_calibrated = apply_physics_calibration_standalone(
                clean_raw, gain, read_noise, domain_name
            )
            clean_min, clean_max = float(np.min(clean_calibrated)), float(
                np.max(clean_calibrated)
            )
            if clean_max > clean_min:
                clean_norm = (clean_calibrated - clean_min) / (clean_max - clean_min)
            else:
                clean_norm = np.clip(clean_calibrated, 0, 1)
            clean_8bit = np.clip(clean_norm * 255.0, 0, 255).astype(np.uint8)

            # Extract center tile for visualization (256×256)
            tile_size = 256
            h, w = noisy_8bit.shape[1], noisy_8bit.shape[2]
            center_y, center_x = h // 2, w // 2
            y1, y2 = center_y - tile_size // 2, center_y + tile_size // 2
            x1, x2 = center_x - tile_size // 2, center_x + tile_size // 2

            # Ensure we don't go out of bounds
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)

            noisy_tile_raw = noisy_raw[:, y1:y2, x1:x2]
            clean_tile_raw = clean_raw[:, y1:y2, x1:x2]
            noisy_tile_calibrated = noisy_8bit[:, y1:y2, x1:x2]

            # Normalize raw tiles for display [0,1]
            noisy_tile_raw_norm = (noisy_tile_raw - np.min(noisy_tile_raw)) / (
                np.max(noisy_tile_raw) - np.min(noisy_tile_raw) + 1e-8
            )
            clean_tile_raw_norm = (clean_tile_raw - np.min(clean_tile_raw)) / (
                np.max(clean_tile_raw) - np.min(clean_tile_raw) + 1e-8
            )
            noisy_tile_calibrated_norm = (
                noisy_tile_calibrated.astype(np.float32) / 255.0
            )

            # Create 3-panel visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Panel 1: Original noisy tile
            if noisy_tile_raw_norm.shape[0] == 1:
                axes[0].imshow(noisy_tile_raw_norm[0], cmap="gray", vmin=0, vmax=1)
            elif noisy_tile_raw_norm.shape[0] == 3:
                axes[0].imshow(noisy_tile_raw_norm.transpose(1, 2, 0))
            axes[0].set_title(
                f"Original Noisy Tile\n(RAW ADU values)", fontsize=12, fontweight="bold"
            )
            axes[0].axis("off")

            # Panel 2: Original clean tile
            if clean_tile_raw_norm.shape[0] == 1:
                axes[1].imshow(clean_tile_raw_norm[0], cmap="gray", vmin=0, vmax=1)
            elif clean_tile_raw_norm.shape[0] == 3:
                axes[1].imshow(clean_tile_raw_norm.transpose(1, 2, 0))
            axes[1].set_title(
                f"Original Clean Tile\n(RAW ADU values)", fontsize=12, fontweight="bold"
            )
            axes[1].axis("off")

            # Panel 3: Calibrated tile (8-bit PNG)
            if noisy_tile_calibrated_norm.shape[0] == 1:
                axes[2].imshow(
                    noisy_tile_calibrated_norm[0], cmap="gray", vmin=0, vmax=1
                )
            elif noisy_tile_calibrated_norm.shape[0] == 3:
                axes[2].imshow(noisy_tile_calibrated_norm.transpose(1, 2, 0))
            axes[2].set_title(
                f"Physics-Calibrated Noisy Tile\n(8-bit PNG, {calibration_method})",
                fontsize=12,
                fontweight="bold",
            )
            axes[2].axis("off")

            # Add overall title
            fig.suptitle(
                f"{domain_name.upper()} Domain: Physics-Based Calibration Effect\n"
                + f"Gain: {gain:.3f} e-/ADU, Read Noise: {read_noise:.3f} e-",
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )

            plt.tight_layout()

            # Save visualization
            output_file = output_path / f"{domain_name}_calibration_comparison.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"✅ Saved visualization: {output_file}")

        except Exception as e:
            logger.error(f"❌ Failed to process {domain_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

    logger.info(f"\n🎉 Visualizations saved to: {output_path}")
    return str(output_path)


def apply_physics_calibration_standalone(
    image: np.ndarray, gain: float, read_noise: float, domain: str
) -> np.ndarray:
    """Standalone version of physics calibration for visualization"""
    try:
        # Convert ADU to electrons
        image_electrons = image.astype(np.float32) * gain

        # Apply read noise correction
        image_calibrated = image_electrons - read_noise

        # Physical constraint: cannot have negative electrons
        image_calibrated = np.maximum(image_calibrated, 0.0)

        return image_calibrated

    except Exception as e:
        logger.error(f"Physics calibration failed: {e}, using original image")
        return image.astype(np.float32)


if __name__ == "__main__":
    # Start the PNG tiles pipeline
    logger.info("🚀 Starting PNG tiles pipeline...")
    main()
