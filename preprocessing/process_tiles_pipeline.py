#!/usr/bin/env python3
"""
Photography Tiles Pipeline

This module provides:
1. SimpleTilesPipeline: Main pipeline for extracting 256×256 tiles from photography data

Domain-specific calibration methods:
- Photography: Sony ARW / Fuji RAF raw files → RGB tiles

Usage:
    python process_tiles_pipeline.py --base_path /path/to/data
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rawpy
import torch
from tqdm import tqdm

# Import configuration and utilities
from preprocessing.config import (
    FUJI_PATH,
    SENSOR_RANGES,
    SONY_PATH,
    TILE_CONFIGS,
    TILE_SIZE,
)
from preprocessing.sensor_detector import SensorDetector, SensorType
from preprocessing.utils import (
    TileInfo,
    create_file_metadata,
    demosaic_raw_to_rgb,
    determine_data_type,
    extract_raw_metadata,
    extract_tiles,
    find_photography_pairs,
    get_pixel_stats,
)
from preprocessing.utils import get_sensor_specific_range as utils_get_sensor_range
from preprocessing.utils import load_sid_split_files, normalize_tile_to_range
from preprocessing.visualizations import (
    create_scene_visualization,
    extract_single_tile_for_viz,
)

# Setup logging first
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleTilesPipeline:
    """Simple file-based pipeline for photography 256×256 tile extraction to PNG"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)

        # Use configuration from config module
        self.tile_size = TILE_SIZE
        self.sony_tile_config = TILE_CONFIGS["sony"]
        self.fuji_tile_config = TILE_CONFIGS["fuji"]
        self.sensor_ranges = SENSOR_RANGES

        # Use predefined sensor ranges from configuration
        # Sensor-specific normalization preserves Poisson-Gaussian noise statistics
        # for diffusion model training

    def load_photography_raw(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load photography raw file and demosaic to RGB.

        Uses rawpy to demosaic raw images to RGB format.

        Args:
            file_path: Path to .ARW (Sony) or .RAF (Fuji) file

        Returns:
            Tuple of (rgb_image, metadata)
            - rgb_image: (3, H, W) numpy array in float32, normalized to [0, 1]
            - metadata: Dictionary with raw file metadata
        """
        return demosaic_raw_to_rgb(file_path)

    def get_sensor_specific_range(self, file_path: str) -> Dict[str, float]:
        """Get sensor-specific range for a given file path (delegates to utils)."""
        return utils_get_sensor_range(file_path, self.sensor_ranges)

    def create_scene_visualization(self, viz_data: Dict[str, Any], output_dir: Path):
        """Create 4-step visualization for a single scene (delegates to visualizations module)."""
        return create_scene_visualization(viz_data, output_dir)

    def _safe_write_json(self, path: Path, data: dict) -> None:
        """
        Atomically write JSON data to file.

        Writes to a temporary file first, then atomically moves it to the target path.
        This prevents partial writes if the process crashes mid-write.

        Args:
            path: Target file path
            data: Dictionary to serialize
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file in same directory (for atomic rename on Unix)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as tmp:
            json.dump(data, tmp, indent=2, default=str)
            tmp_path = tmp.name

        # Atomic move on Unix, force overwrite on Windows
        try:
            shutil.move(tmp_path, str(path))
        except Exception as e:
            # Cleanup temp file on error
            try:
                os.unlink(tmp_path)
            except:
                pass
            raise e

    def extract_tiles_for_camera(
        self, image: np.ndarray, camera_type: str
    ) -> List[Any]:
        """Extract tiles using camera-specific config (DRY consolidation)."""
        config = TILE_CONFIGS[camera_type]
        return extract_tiles(
            image, config["target_grid"][0], config["target_grid"][1], self.tile_size
        )

    # ========================================================================
    # Private helper methods for process_file_to_pt_tiles (extracted methods)
    # ========================================================================

    def _load_and_validate_image(
        self, file_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[Dict]]:
        """Load image and collect image statistics.

        Validates inputs early to fail fast on critical errors.

        Args:
            file_path: Path to raw image file

        Returns:
            Tuple of (image, metadata, image_stats) or (None, None, None) on failure

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file is not readable
        """
        # === INPUT VALIDATION (Fail Fast) ===

        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Raw image file not found: {file_path}")

        # Validate file is readable
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Cannot read raw image file: {file_path}")

        # === LOADING ===
        try:
            image, metadata = self.load_photography_raw(file_path)

            if image is None or image.size == 0:
                logger.warning(f"Empty image loaded: {file_path}")
                return None, None, None

            metadata["file_path"] = file_path

            # Collect statistics
            image_stats = {
                "min": float(np.min(image)),
                "max": float(np.max(image)),
                "mean": float(np.mean(image)),
                "std": float(np.std(image)),
            }

            return image, metadata, image_stats

        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            return None, None, None

    def _process_single_tile(
        self,
        tile_info: Any,
        tile_index: int,
        file_path: str,
        sensor_range: Dict,
        sid_info: Dict,
        camera_prefix: str,
        data_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Process a single tile and return its metadata.

        Args:
            tile_info: TileInfo object containing tile data and position
            tile_index: Index of the tile in the tile list
            file_path: Path to the source file
            sensor_range: Dictionary with min/max values for normalization
            sid_info: SID file information dictionary
            camera_prefix: Camera type prefix (e.g., 'sony', 'fuji')
            data_type: Data type ('short' or 'long')

        Returns:
            Tile metadata dictionary or None if processing failed
        """
        try:
            tile_float = tile_info.tile_data

            # Validate tile data exists
            if tile_float is None or tile_float.size == 0:
                return None

            # Generate tile ID (sensor-based instead of domain-based)
            base_stem = Path(file_path).stem
            tile_id = f"{camera_prefix}_{base_stem}_tile_{tile_index:04d}"

            # Save tile (validation happens inside save_tile_as_pt)
            tile_metadata = self.save_tile_as_pt(
                tile_float,
                tile_id,
                data_type,
                sensor_range,
                file_path,
            )

            if not tile_metadata:
                return None

            # Extract spatial and scene information
            scene_id = sid_info.get("scene_id")
            exposure_time = sid_info.get("exposure_time")
            tile_key = f"{camera_prefix}_{scene_id}_tile_{tile_info.grid_position[0]}_{tile_info.grid_position[1]}"

            # Find corresponding long tile for short exposures
            corresponding_long_tile_path = None
            if data_type == "short":
                long_partner = sid_info.get("long_partner")
                if long_partner:
                    long_stem = Path(long_partner).stem
                    corresponding_long_tile_id = (
                        f"{camera_prefix}_{long_stem}_tile_{tile_index:04d}"
                    )
                    corresponding_long_tile_path = str(
                        self.base_path
                        / "processed"
                        / "pt_tiles"
                        / camera_prefix
                        / "long"
                        / f"{corresponding_long_tile_id}.pt"
                    )

            # Add tile-specific metadata
            tile_metadata.update(
                {
                    "scene_id": scene_id,
                    "exposure_time": exposure_time,
                    "tile_key": tile_key,
                    "grid_x": int(tile_info.grid_position[0]),
                    "grid_y": int(tile_info.grid_position[1]),
                    "image_x": int(tile_info.image_position[0]),
                    "image_y": int(tile_info.image_position[1]),
                    "quality_score": float(np.mean(tile_float))
                    if np.isfinite(np.mean(tile_float))
                    else 0.0,
                    "valid_ratio": float(tile_info.valid_ratio),
                    "is_edge_tile": bool(tile_info.is_edge_tile),
                    "tile_stats": {
                        "min": float(np.min(tile_float))
                        if np.isfinite(np.min(tile_float))
                        else -1.0,
                        "max": float(np.max(tile_float))
                        if np.isfinite(np.max(tile_float))
                        else 1.0,
                        "mean": float(np.mean(tile_float))
                        if np.isfinite(np.mean(tile_float))
                        else 0.0,
                        "std": float(np.std(tile_float))
                        if np.isfinite(np.std(tile_float))
                        else 0.5,
                    },
                }
            )

            if corresponding_long_tile_path:
                tile_metadata["corresponding_long_tile"] = corresponding_long_tile_path

            return tile_metadata

        except Exception as e:
            logger.error(f"Failed to process tile {tile_index} from {file_path}: {e}")
            return None

    def save_tile_as_pt(
        self,
        tile_data: np.ndarray,
        tile_id: str,
        data_type: str,
        sensor_range: Dict[str, float],
        file_path: str = None,
    ) -> Dict[str, Any]:
        """Save tile as .pt file and return metadata

        Saves as RGB (3, 256, 256) float32
        Data already normalized to [0,1] in demosaic_raw_to_rgb
        Final normalization: [0,1] → [-1,1] for diffusion models
        """
        try:
            # Ensure proper shape
            if len(tile_data.shape) == 2:
                tile_data = tile_data[np.newaxis, :, :]

            # Create output directory (sensor-based structure)
            # Extract sensor type from file_path to determine output location
            if file_path:
                try:
                    sensor_type = SensorDetector.detect(file_path)
                    sensor_name = sensor_type.value
                except Exception:
                    sensor_name = "unknown"
            else:
                sensor_name = "unknown"

            output_dir = (
                self.base_path / "processed" / "pt_tiles" / sensor_name / data_type
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # Accept valid channel counts: RGB (3), Sony Bayer (4), Fuji X-Trans (9)
            valid_channels = [3, 4, 9]
            if tile_data.shape[0] not in valid_channels:
                raise ValueError(
                    f"Tile has unexpected channels: {tile_data.shape[0]} "
                    f"(expected {valid_channels}: RGB=3, Sony Bayer=4, Fuji X-Trans=9)"
                )
            actual_channels = tile_data.shape[0]

            # Final validation
            expected_shape = (actual_channels, self.tile_size, self.tile_size)
            if tile_data.shape != expected_shape:
                raise ValueError(
                    f"Tile shape mismatch: {tile_data.shape} != {expected_shape}"
                )

            pt_path = output_dir / f"{tile_id}.pt"

            # Data already normalized to [0,1] in demosaic_raw_to_rgb
            # Convert to tensor and normalize to [-1,1] for diffusion models
            tensor_data = torch.from_numpy(tile_data.astype(np.float32))
            tensor_data = 2 * tensor_data - 1  # [0,1] → [-1,1]

            # Save as PyTorch tensor
            torch.save(tensor_data, pt_path)

            # Return metadata
            metadata = {
                "tile_id": tile_id,
                "pt_path": str(pt_path),
                "sensor_type": sensor_name,
                "data_type": data_type,
                "sensor_range": sensor_range,
                "tile_size": self.tile_size,
                "channels": actual_channels,
                "processing_timestamp": datetime.now().isoformat(),
            }

            return metadata

        except Exception as e:
            logger.error(f"Failed to save tile as .pt: {e}")
            return None

    def process_file_to_pt_tiles(
        self,
        file_path: str,
        create_viz: bool = False,
        pair_metadata: Dict[str, Dict[str, str]] = None,
        sid_file_info: Dict[str, Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Process a single file to .pt tiles with sensor-specific normalization - CORRECT FLOW: Load→Tile→Normalize

        Args:
            file_path: Path to the file to process
            create_viz: If True, collect intermediate data for visualization
            pair_metadata: Dictionary containing pair relationship information
            sid_file_info: Dictionary containing SID split file information

        Returns:
            Tuple of (processed_tiles, viz_data) where viz_data is None unless create_viz=True

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file is not readable
        """
        # === INPUT VALIDATION (Fail Fast on Critical Errors) ===

        # Validate file exists and is readable
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Raw image file not found: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(
                f"Cannot read raw image file (permission denied): {file_path}"
            )

        # Validate file has valid extension
        valid_extensions = {".ARW", ".RAF"}
        if file_path_obj.suffix.upper() not in valid_extensions:
            raise ValueError(
                f"Invalid file extension: {file_path_obj.suffix}. Expected .ARW or .RAF"
            )

        try:
            # === STEP 1: Load full image ===
            image, metadata = self.load_photography_raw(file_path)

            if image is None or image.size == 0:
                return None, None

            # Add file path to metadata
            metadata["file_path"] = file_path

            # === STEP 2: Store original data range for reference ===
            orig_min = float(np.min(image))
            orig_max = float(np.max(image))
            orig_mean = float(np.mean(image))
            orig_std = float(np.std(image))

            # Get sensor-specific normalization range
            sensor_range = self.get_sensor_specific_range(file_path)

            # Initialize visualization data if requested
            viz_data = None
            viz_tile = None

            # NORMALIZATION ALREADY DONE in demosaic_raw_to_rgb → pack_raw_to_rgb
            # Raw images are normalized to [0,1] during demosaicing (black level subtraction + white level normalization)
            # Tiles will be further normalized to [-1,1] when saved as .pt files

            # === STEP 3: NOW tile the raw image ===
            # Convert to CHW format if needed (tiler expects this)
            if len(image.shape) == 2:
                image = image[np.newaxis, :, :]  # Add channel dimension
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # Convert HWC → CHW for RGB
                image = image.transpose(2, 0, 1)

            # Extract tiles from raw float32 image
            # Use sensor-specific tiling to ensure proper overlap and NO padding
            if create_viz:
                # Store raw image for visualization
                viz_data = {
                    "raw_image": image.copy(),
                    "original_shape": image.shape,
                }

            # Use SensorDetector to determine camera type and extract tiles
            file_path_str = metadata.get("file_path", "")
            try:
                sensor_type = SensorDetector.detect(file_path_str)
                if sensor_type == SensorType.SONY:
                    # Sony files: custom tiling to get tiles based on TILE_CONFIGS
                    tile_infos = self.extract_tiles_for_camera(image, "sony")
                elif sensor_type == SensorType.FUJI:
                    # Fuji files: custom tiling to get tiles based on TILE_CONFIGS
                    tile_infos = self.extract_tiles_for_camera(image, "fuji")
                else:
                    logger.error(f"Unknown sensor type: {sensor_type}")
                    return None, None, None
            except Exception as e:
                logger.error(f"Failed to detect sensor type for {file_path_str}: {e}")
                return None, None, None

            if not tile_infos:
                return None, None

            # === STEP 4: Create file-level metadata (no duplication) ===
            if sid_file_info and file_path in sid_file_info:
                sid_info = sid_file_info[file_path]
                data_type = sid_info["file_type"]
            else:
                raise ValueError(
                    f"SID file info not available for {file_path}. "
                    f"Must load SID split files first using load_sid_split_files()."
                )

            # Extract one representative tile for visualization (after tiles are extracted)
            if create_viz:
                # Use a consistent tile selection strategy based on scene_id to ensure
                # short and long exposures show the same spatial region
                # Use scene_id hash to get a consistent but varied tile index
                import hashlib

                scene_id = sid_info.get("scene_id", "unknown")
                scene_hash = int(
                    hashlib.md5(scene_id.encode(), usedforsecurity=False).hexdigest()[
                        :8
                    ],
                    16,
                )  # noqa: B324
                viz_tile_idx = scene_hash % len(
                    tile_infos
                )  # Ensure index is within bounds

                viz_tile = extract_single_tile_for_viz(
                    tile_infos[viz_tile_idx], tile_size=256
                )
                viz_data["tiled_image"] = viz_tile.copy()
                # Store the tile index used for visualization so we can track which actual tile it corresponds to
                viz_data["viz_tile_index"] = viz_tile_idx

            # Create consolidated file metadata once
            file_metadata = create_file_metadata(
                file_path=file_path,
                metadata=metadata,
                sid_info=sid_info,
                pair_info=pair_metadata.get(file_path) if pair_metadata else None,
                domain_range=sensor_range,  # Use sensor_range instead of domain_range
                image_stats={
                    "min": orig_min,
                    "max": orig_max,
                    "mean": orig_mean,
                    "std": orig_std,
                },
            )

            # === STEP 5: Process tiles with only tile-specific metadata ===
            processed_tiles = []
            max_tiles = len(tile_infos)

            # Determine camera type once for all tiles
            try:
                sensor_type = SensorDetector.detect(file_path)
                camera_type = sensor_type.value
            except Exception as e:
                logger.error(f"Failed to detect camera type for {file_path}: {e}")
                camera_type = "unknown"
            camera_prefix = camera_type or "unknown"

            for i in range(max_tiles):
                tile_info = tile_infos[i]
                tile_metadata = self._process_single_tile(
                    tile_info=tile_info,
                    tile_index=i,
                    file_path=file_path,
                    sensor_range=sensor_range,
                    sid_info=sid_info,
                    camera_prefix=camera_prefix,
                    data_type=data_type,
                )

                if tile_metadata:
                    processed_tiles.append(tile_metadata)

            # === STEP 5: Process visualization tile if requested ===
            if create_viz and viz_tile is not None and viz_data is not None:
                try:
                    # Accept valid channel counts: RGB (3), Sony Bayer (4), Fuji X-Trans (9)
                    valid_channels = [3, 4, 9]
                    if viz_tile.shape[0] not in valid_channels:
                        raise ValueError(
                            f"Unexpected channel count for viz tile: {viz_tile.shape[0]} (expected {valid_channels})"
                        )
                    viz_tile_processed = viz_tile

                    # Data already normalized to [0,1] in demosaic_raw_to_rgb
                    # Convert to tensor [-1,1] for diffusion models
                    tensor_viz = torch.from_numpy(viz_tile_processed.astype(np.float32))
                    tensor_viz = 2 * tensor_viz - 1  # [0,1] → [-1,1]

                    # Store all processing steps for visualization
                    viz_data["tensor"] = tensor_viz

                except Exception as e:
                    logger.error(f"Failed to process visualization tile: {e}")
                    viz_data = None

            # Return structured metadata: file-level metadata + tiles
            return {"file_metadata": file_metadata, "tiles": processed_tiles}, viz_data

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return None, None

    def run_pt_tiles_pipeline(
        self,
        max_files_per_domain: int = None,
        create_visualizations: bool = False,
        dry_run: bool = False,
        sensors: list = None,
    ):
        """Run the complete .pt tiles pipeline with sensor-specific normalization and optional visualizations

        Args:
            max_files_per_domain: Maximum files to process per domain (None = all files)
            create_visualizations: If True, create 4-step visualization for first scene per domain
            dry_run: If True, validate metadata and paths without processing files.
                    Useful for debugging configuration issues before long processing runs.

        DRY_RUN MODE:
            When dry_run=True, the pipeline will:
            1. Load and validate all configuration files
            2. Check file existence and readability
            3. Validate sensor detection for all files
            4. Verify output directory structure
            5. Test first 3 files end-to-end WITHOUT saving
            6. Report any validation errors clearly

            This is useful for:
            - Testing new data paths
            - Validating sensor configuration
            - Checking for permission/path issues
            - Verifying dataset structure before long processing
        """

        # Define ALL files for photography domain
        sample_files = {"photography": []}

        # Load SID split files to get official train/val/test assignments and ISO/aperture info
        sid_data_path = str(self.base_path / "raw" / "SID")
        sid_file_info = load_sid_split_files(sid_data_path)

        # Get all files from SID split files (these are the official files to process)
        all_files = list(sid_file_info.keys())

        # Filter by sensor type if specified
        if sensors:
            filtered_files = [
                f for f in all_files if sid_file_info[f].get("camera_type") in sensors
            ]
            sample_files["photography"] = filtered_files

            sensor_list = ", ".join(sensors)
            logger.info(
                f"Processing {sensor_list} files only: {len(filtered_files)} files"
            )
            logger.info(f"Total files in dataset: {len(all_files)}")
            logger.info(
                f"Skipping other sensors: {len(all_files) - len(filtered_files)}"
            )
        else:
            # Process all files (default behavior)
            sample_files["photography"] = all_files
            logger.info(f"Processing all sensor types: {len(all_files)} files")

        # === DRY RUN MODE: Validate without processing ===
        if dry_run:
            logger.info("=" * 70)
            logger.info("DRY RUN MODE - Validation Only (No Files Processed)")
            logger.info("=" * 70)

            # Validate configuration
            logger.info(f"\n✓ Config validation:")
            logger.info(f"  - Base path: {self.base_path}")
            logger.info(f"  - SID data path: {sid_data_path}")
            logger.info(
                f"  - Total files in dataset: {len(sample_files['photography'])}"
            )

            # Validate sensor detection
            logger.info(f"\n✓ Sensor detection validation (first 3 files):")
            validation_errors = []
            for i, file_path in enumerate(sample_files["photography"][:3]):
                try:
                    if not Path(file_path).exists():
                        validation_errors.append(f"  ✗ File not found: {file_path}")
                        continue
                    sensor_type = SensorDetector.detect(file_path)
                    logger.info(f"  ✓ {Path(file_path).name}: {sensor_type.value}")
                except Exception as e:
                    validation_errors.append(f"  ✗ {Path(file_path).name}: {e}")

            if validation_errors:
                logger.error(f"\nValidation errors found:")
                for error in validation_errors:
                    logger.error(error)
                return {"status": "dry_run_failed", "errors": validation_errors}

            # Validate output directory
            output_dir = self.base_path / "processed" / "pt_tiles"
            logger.info(f"\n✓ Output directory ready:")
            logger.info(f"  - Output dir: {output_dir}")
            logger.info(f"  - Writable: {os.access(self.base_path, os.W_OK)}")

            logger.info("\n" + "=" * 70)
            logger.info("Dry run validation PASSED - Configuration is OK!")
            logger.info("To run full processing, set dry_run=False")
            logger.info("=" * 70 + "\n")

            return {"status": "dry_run_passed", "files_validated": 3}

        # === STEP 1: Use default sensor ranges (no auto-discovery) ===
        # Use predefined sensor ranges from config instead of auto-discovery
        logger.info("Using predefined sensor ranges from configuration")

        # Create pair metadata from SID file info
        # Use long_partner from metadata (all short exposures pair with the one long exposure)
        pair_metadata = {}
        for file_path, info in sid_file_info.items():
            scene_id = info.get("scene_id")
            camera_type = info.get("camera_type")
            file_type = info.get("file_type")

            # Use the long_partner from metadata (computed in load_sid_split_files)
            long_partner = info.get("long_partner")

            pair_metadata[file_path] = {
                "pair_id": f"{camera_type}_{scene_id}",
                "pair_type": file_type,
                "pair_partner": long_partner,  # All short exposures reference the same long exposure
                "all_short_exposures": info.get("all_short_exposures", []),
                "all_long_exposures": info.get("all_long_exposures", []),
            }

        all_tiles_metadata = []
        results = {"domains": {}, "total_tiles": 0}

        # Track visualization data per scene
        viz_data_by_scene = (
            {} if create_visualizations else None
        )  # {domain: {scene_id: {"short": data, "long": data}}}
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

            for file_path in tqdm(
                file_list[: max_files_per_domain or len(file_list)],
                desc=f"Processing {domain_name}",
            ):
                if not Path(file_path).exists():
                    continue

                try:
                    # Determine data type and scene using SID file info (required for SID dataset)
                    if sid_file_info and file_path in sid_file_info:
                        sid_info = sid_file_info[file_path]
                        data_type = sid_info["file_type"]  # "long" or "short"
                        scene_id = sid_info["scene_id"]
                    else:
                        # SID file info is required for SID dataset
                        raise ValueError(
                            f"SID file info not available for {file_path}. "
                            f"Must load SID split files first using load_sid_split_files()."
                        )

                    # Check if we should collect visualization data
                    # Collect for first scene per domain, and for both short and long exposure
                    should_create_viz = False
                    if create_visualizations:
                        if scene_id not in viz_data_by_scene[domain_name]:
                            # First time seeing this scene - initialize and collect
                            viz_data_by_scene[domain_name][scene_id] = {}
                            should_create_viz = True
                        elif data_type not in viz_data_by_scene[domain_name][scene_id]:
                            # We have this scene but missing this data type (long or short)
                            should_create_viz = True

                    # Process file with or without visualization
                    result = self.process_file_to_pt_tiles(
                        file_path,
                        create_viz=should_create_viz,
                        pair_metadata=pair_metadata,
                        sid_file_info=sid_file_info,
                    )

                    if result is None:
                        continue

                    file_data, viz_data = result

                    # Skip if no file data
                    if file_data is None:
                        continue

                    # Store visualization data if collected
                    if viz_data is not None and should_create_viz:
                        # Add tile_id information to visualization data
                        if "viz_tile_index" in viz_data:
                            viz_tile_idx = viz_data["viz_tile_index"]
                            # Find the corresponding saved tile metadata
                            if (
                                file_data
                                and "tiles" in file_data
                                and viz_tile_idx < len(file_data["tiles"])
                            ):
                                corresponding_tile = file_data["tiles"][viz_tile_idx]
                                viz_data[
                                    "corresponding_tile_id"
                                ] = corresponding_tile.get("tile_id")
                                viz_data[
                                    "corresponding_tile_path"
                                ] = corresponding_tile.get("pt_path")

                        viz_data_by_scene[domain_name][scene_id][data_type] = viz_data

                        # If we have both short and long for this scene, create visualization immediately
                        scene_data = viz_data_by_scene[domain_name][scene_id]
                        if "short" in scene_data and "long" in scene_data:
                            viz_output = {
                                "domain": domain_name,
                                "sensor": sid_info["camera_type"],
                                "scene_id": scene_id,
                                "short": scene_data["short"],
                                "long": scene_data["long"],
                            }
                            self.create_scene_visualization(viz_output, viz_output_dir)

                    # Add file data to domain_tiles (file_metadata + tiles)
                    domain_tiles.append(file_data)
                    processed_files += 1

                    # Save incremental metadata after EACH file (for better progress tracking)
                    try:
                        incremental_path = (
                            self.base_path
                            / "processed"
                            / f"metadata_{domain_name}_incremental.json"
                        )
                        # Calculate total tiles from all files
                        total_tiles = sum(len(f["tiles"]) for f in domain_tiles)
                        metadata = {
                            "domain": domain_name,
                            "files_processed": processed_files,
                            "tiles_generated": total_tiles,
                            "files": domain_tiles,
                            "timestamp": datetime.now().isoformat(),
                        }
                        self._safe_write_json(incremental_path, metadata)
                    except Exception as e:
                        logger.warning(f"Failed to save incremental metadata: {e}")
                        pass

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue

            all_tiles_metadata.extend(domain_tiles)
            total_tiles = sum(len(f["tiles"]) for f in domain_tiles)
            results["domains"][domain_name] = {
                "files_processed": processed_files,
                "tiles_generated": total_tiles,
            }
            results["total_tiles"] += total_tiles

        # Save comprehensive metadata (includes all calibration, spatial, and processing info)
        metadata_path = (
            self.base_path / "processed" / "comprehensive_tiles_metadata.json"
        )

        # Add processing summary to metadata
        comprehensive_metadata = {
            "pipeline_info": {
                "total_tiles": results["total_tiles"],
                "domains_processed": list(results["domains"].keys()),
                "domain_stats": results["domains"],
                "processing_timestamp": datetime.now().isoformat(),
                "tile_size": self.tile_size,
                "sensor_ranges": self.sensor_ranges,
                "storage_format": "pt_float32",
                "normalization": "[raw] → [0,1] in demosaic → [-1,1] in save",
                "sensor_specific_normalization": True,
            },
            "files": all_tiles_metadata,
        }

        # Save with atomic writes and error handling
        try:
            self._safe_write_json(metadata_path, comprehensive_metadata)
        except Exception as e:
            logger.error(f"Failed to save comprehensive metadata: {e}")
            # Try to save to backup location
            backup_path = (
                self.base_path
                / "processed"
                / "comprehensive_tiles_metadata_backup.json"
            )
            try:
                self._safe_write_json(backup_path, comprehensive_metadata)
                logger.info(f"Saved backup metadata to {backup_path}")
            except Exception as e2:
                logger.error(f"Failed to save backup metadata: {e2}")

        return results


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple .pt Tiles Pipeline with Domain-Specific Range Normalization for [-1,1]"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to data directory (overrides DATA_PATH environment variable)",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Base path for data (deprecated: use --data_path instead)",
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
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration without processing (no files written)",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        choices=["sony", "fuji"],
        default=None,
        help="Sensor types to process: 'sony', 'fuji', or both (default: all)",
    )

    args = parser.parse_args()

    # Set DATA_PATH environment variable if provided
    if args.data_path:
        os.environ["DATA_PATH"] = args.data_path
        data_path = args.data_path
    elif args.base_path:
        # Support deprecated --base_path for backward compatibility
        os.environ["DATA_PATH"] = args.base_path
        data_path = args.base_path
    else:
        # Use environment variable or default
        data_path = os.environ.get(
            "DATA_PATH", "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data"
        )

    # Run .pt tiles pipeline for photography
    pipeline = SimpleTilesPipeline(data_path)
    results = pipeline.run_pt_tiles_pipeline(
        args.max_files,
        create_visualizations=args.visualize,
        dry_run=args.dry_run,
        sensors=args.sensors,
    )

    if results.get("total_tiles", 0) > 0:
        print(f"\n🎊 SUCCESS: .pt Tiles Pipeline Completed!")
        print(f"📊 Total .pt tiles generated: {results['total_tiles']:,}")
        print(f"📐 Tile size: 256×256")
        print(f"💾 .pt files (float32) saved to: {data_path}/processed/pt_tiles/")
        print(f"🎯 Sensor-specific normalization applied: raw → [0,1] → [-1,1]")
        print(f"   • Normalization done during demosaicing + final [-1,1] scaling")

        if args.visualize:
            print(f"\n📊 Visualizations saved to: {data_path}/processed/visualizations/")

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
    main()
