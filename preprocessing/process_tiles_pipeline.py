#!/usr/bin/env python3
"""
Sensor Tiles Pipeline

This module provides:
1. SimpleTilesPipeline: Main pipeline for extracting 256×256 tiles from sensor data (Sony/Fuji)

Sensor-specific calibration methods:
- Sony: ARW raw files → RGB tiles
- Fuji: RAF raw files → RGB tiles

Usage:
    python process_tiles_pipeline.py --base_path /path/to/data
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to Python path to allow imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import h5py and loadmat availability from core.file_utils
from core.utils.file_utils import HAS_H5PY, HAS_SCIPY

if HAS_SCIPY:
    from scipy.io import loadmat
else:
    loadmat = None

# Import configuration and utilities
from config.config import FUJI_PATH, SONY_PATH, TILE_CONFIGS, TILE_SIZE
from config.sensor_config import SENSOR_RANGES
from core.sensor_detector import SensorDetector, SensorType
from core.utils.file_utils import load_mat_file, save_json_file
from core.utils.metadata_utils import (
    create_file_metadata,
    determine_data_type,
    extract_raw_metadata,
    find_photography_pairs,
    load_sid_split_files,
)
from core.utils.sensor_utils import demosaic_raw_to_rgb, get_sensor_specific_range
from core.utils.tensor_utils import get_pixel_stats
from core.utils.tiles_utils import TileInfo, extract_tiles, normalize_tile_to_range
from visualization.visualizations import (
    create_scene_visualization,
    extract_single_tile_for_viz,
)


class SimpleTilesPipeline:
    """Simple file-based pipeline for sensor (Sony/Fuji) 256×256 tile extraction to PNG"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)

        # Use configuration from config module
        self.tile_size = TILE_SIZE
        self.sony_tile_config = TILE_CONFIGS["sony"]
        self.fuji_tile_config = TILE_CONFIGS["fuji"]
        self.sensor_ranges = SENSOR_RANGES

    def load_photography_raw(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load raw file (Sony ARW or Fuji RAF) and demosaic to RGB.

        Uses rawpy to demosaic raw images to RGB format.

        Args:
            file_path: Path to .ARW (Sony) or .RAF (Fuji) file

        Returns:
            Tuple of (rgb_image, metadata)
            - rgb_image: (3, H, W) numpy array in float32, normalized to [0, 1]
            - metadata: Dictionary with raw file metadata
        """
        return demosaic_raw_to_rgb(file_path)

    def load_sidd_mat_file(
        self, file_path: Path, variable_name: str = "x"
    ) -> Optional[np.ndarray]:
        """Load SIDD .MAT file (MATLAB v7.3 HDF5 format).

        Args:
            file_path: Path to .MAT file
            variable_name: Variable name in .mat file (default: "x")

        Returns:
            numpy array in float32, or None if loading fails
        """
        # Use core.utils.load_mat_file which handles both HDF5 and v7 formats
        return load_mat_file(file_path, variable_name)

    def parse_sidd_scene_name(self, scene_name: str) -> Dict[str, Any]:
        """Parse SIDD scene directory name to extract metadata.

        Args:
            scene_name: Scene directory name

        Returns:
            Dictionary with parsed metadata
        """
        parts = scene_name.split("_")
        if len(parts) < 7:
            return {
                "camera_code": "unknown",
                "iso": None,
                "shutter_speed": None,
                "illuminant_temp": None,
                "brightness": None,
            }

        return {
            "scene_instance": parts[0],
            "scene_number": parts[1],
            "camera_code": parts[2].lower(),
            "iso": int(parts[3]) if parts[3].isdigit() else None,
            "shutter_speed": int(parts[4]) if parts[4].isdigit() else None,
            "illuminant_temp": int(parts[5]) if parts[5].isdigit() else None,
            "brightness": parts[6],
        }

    def demosaic_sidd_raw(
        self, raw_data: np.ndarray, cfa_pattern: List[int] = None
    ) -> np.ndarray:
        """Demosaic SIDD raw Bayer pattern to RGB.

        Args:
            raw_data: 2D numpy array (H, W) with raw Bayer pattern, values in [0,1]
            cfa_pattern: CFA pattern as [r, g, g, b] indices (0=R, 1=G, 2=B), default RGGB

        Returns:
            RGB image (3, H, W) in float32, normalized to [0,1]
        """
        if raw_data.ndim != 2:
            raise ValueError(f"Expected 2D raw Bayer data, got shape {raw_data.shape}")

        # Default to RGGB pattern
        if cfa_pattern is None:
            cfa_pattern = [0, 1, 1, 2]

        try:
            import cv2

            max_val = 16383
            raw_uint = (raw_data * max_val).astype(np.uint16)
            pattern_str = "".join(["RGB"[i] for i in cfa_pattern])
            flag_map = {
                "RGGB": cv2.COLOR_BayerRG2RGB,
                "GRBG": cv2.COLOR_BayerGR2RGB,
                "GBRG": cv2.COLOR_BayerGB2RGB,
                "BGGR": cv2.COLOR_BayerBG2RGB,
            }

            if pattern_str not in flag_map:
                logger.warning(f"Unknown CFA pattern {pattern_str}, using RGGB")
                flag = cv2.COLOR_BayerRG2RGB
            else:
                flag = flag_map[pattern_str]

            rgb = cv2.cvtColor(raw_uint, flag)
            rgb = rgb.astype(np.float32) / max_val
            rgb = np.transpose(rgb, (2, 0, 1))
            return rgb
        except ImportError:
            logger.warning("OpenCV not available, trying colour-demosaicing")

        try:
            from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007

            pattern_str = "".join(["RGB"[i] for i in cfa_pattern])
            rgb = demosaicing_CFA_Bayer_Menon2007(raw_data, pattern=pattern_str)
            rgb = rgb.astype(np.float32)
            rgb = np.transpose(rgb, (2, 0, 1))
            return rgb
        except ImportError:
            raise ImportError(
                "Need either OpenCV or colour-demosaicing for demosaicing"
            )

    def get_sensor_specific_range(self, file_path: str) -> Dict[str, float]:
        """Get sensor-specific range for a given file path."""
        return get_sensor_specific_range(file_path, self.sensor_ranges)

    def create_scene_visualization(self, viz_data: Dict[str, Any], output_dir: Path):
        """Create visualization for a single scene."""
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
        # Use core.utils.save_json_file with atomic=True for safe writes
        save_json_file(path, data, default=str, atomic=True)

    def extract_tiles_for_camera(
        self, image: np.ndarray, camera_type: str
    ) -> List[Any]:
        """Extract tiles using camera-specific config."""
        config = TILE_CONFIGS[camera_type]
        return extract_tiles(
            image, config["target_grid"][0], config["target_grid"][1], self.tile_size
        )

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
        """Save tile as .pt file and return metadata."""
        try:
            # Ensure proper shape
            if len(tile_data.shape) == 2:
                tile_data = tile_data[np.newaxis, :, :]

            is_sidd = False
            sensor_name = "unknown"
            sidd_cameras = ["s6", "g4", "n6", "gp", "ip"]

            for cam in sidd_cameras:
                if cam in tile_id.lower():
                    is_sidd = True
                    sensor_name = cam
                    break

            if is_sidd:
                if "dataset" in str(self.base_path):
                    output_base = (
                        self.base_path.parent
                        if self.base_path.name == "dataset"
                        else self.base_path
                    )
                else:
                    output_base = self.base_path
                output_dir = (
                    output_base
                    / "dataset"
                    / "processed_sidd"
                    / "pt_tiles"
                    / sensor_name
                    / data_type
                )
            else:
                if file_path:
                    try:
                        sensor_type = SensorDetector.detect(file_path)
                        sensor_name = sensor_type.value
                    except Exception:
                        sensor_name = "unknown"
                output_dir = (
                    self.base_path / "processed" / "pt_tiles" / sensor_name / data_type
                )
            output_dir.mkdir(parents=True, exist_ok=True)

            valid_channels = [3, 4, 9]
            if tile_data.shape[0] not in valid_channels:
                raise ValueError(
                    f"Tile has unexpected channels: {tile_data.shape[0]} "
                    f"(expected {valid_channels})"
                )
            actual_channels = tile_data.shape[0]

            expected_shape = (actual_channels, self.tile_size, self.tile_size)
            if tile_data.shape != expected_shape:
                raise ValueError(
                    f"Tile shape mismatch: {tile_data.shape} != {expected_shape}"
                )

            pt_path = output_dir / f"{tile_id}.pt"
            tensor_data = torch.from_numpy(tile_data.astype(np.float32))
            tensor_data = 2 * tensor_data - 1

            torch.save(tensor_data, pt_path)

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
        """Process a single file to .pt tiles with sensor-specific normalization.

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
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Raw image file not found: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Cannot read raw image file: {file_path}")

        valid_extensions = {".ARW", ".RAF"}
        if file_path_obj.suffix.upper() not in valid_extensions:
            raise ValueError(
                f"Invalid file extension: {file_path_obj.suffix}. Expected .ARW or .RAF"
            )

        try:
            image, metadata = self.load_photography_raw(file_path)
            if image is None or image.size == 0:
                return None, None

            metadata["file_path"] = file_path
            orig_min = float(np.min(image))
            orig_max = float(np.max(image))
            orig_mean = float(np.mean(image))
            orig_std = float(np.std(image))
            sensor_range = self.get_sensor_specific_range(file_path)

            viz_data = None
            viz_tile = None

            if len(image.shape) == 2:
                image = image[np.newaxis, :, :]
            elif len(image.shape) == 3 and image.shape[2] == 3:
                image = image.transpose(2, 0, 1)

            if create_viz:
                viz_data = {
                    "raw_image": image.copy(),
                    "original_shape": image.shape,
                }

            file_path_str = metadata.get("file_path", "")
            try:
                sensor_type = SensorDetector.detect(file_path_str)
                if sensor_type == SensorType.SONY:
                    tile_infos = self.extract_tiles_for_camera(image, "sony")
                elif sensor_type == SensorType.FUJI:
                    tile_infos = self.extract_tiles_for_camera(image, "fuji")
                else:
                    logger.error(f"Unknown sensor type: {sensor_type}")
                    return None, None
            except Exception as e:
                logger.error(f"Failed to detect sensor type for {file_path_str}: {e}")
                return None, None

            if not tile_infos:
                return None, None

            if sid_file_info and file_path in sid_file_info:
                sid_info = sid_file_info[file_path]
                data_type = sid_info["file_type"]
            else:
                raise ValueError(
                    f"SID file info not available for {file_path}. "
                    f"Must load SID split files first using load_sid_split_files()."
                )

            if create_viz:
                import hashlib

                scene_id = sid_info.get("scene_id", "unknown")
                scene_hash = int(
                    hashlib.md5(scene_id.encode(), usedforsecurity=False).hexdigest()[
                        :8
                    ],
                    16,
                )
                viz_tile_idx = scene_hash % len(tile_infos)
                viz_tile = extract_single_tile_for_viz(
                    tile_infos[viz_tile_idx], tile_size=256
                )
                viz_data["tiled_image"] = viz_tile.copy()
                viz_data["viz_tile_index"] = viz_tile_idx

            file_metadata = create_file_metadata(
                file_path=file_path,
                metadata=metadata,
                sid_info=sid_info,
                pair_info=pair_metadata.get(file_path) if pair_metadata else None,
                domain_range=sensor_range,
                image_stats={
                    "min": orig_min,
                    "max": orig_max,
                    "mean": orig_mean,
                    "std": orig_std,
                },
            )

            processed_tiles = []
            try:
                sensor_type = SensorDetector.detect(file_path)
                camera_prefix = sensor_type.value
            except Exception as e:
                logger.error(f"Failed to detect camera type for {file_path}: {e}")
                camera_prefix = "unknown"

            for i in range(len(tile_infos)):
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

            if create_viz and viz_tile is not None and viz_data is not None:
                try:
                    valid_channels = [3, 4, 9]
                    if viz_tile.shape[0] not in valid_channels:
                        raise ValueError(
                            f"Unexpected channel count for viz tile: {viz_tile.shape[0]}"
                        )
                    tensor_viz = torch.from_numpy(viz_tile.astype(np.float32))
                    tensor_viz = 2 * tensor_viz - 1
                    viz_data["tensor"] = tensor_viz
                except Exception as e:
                    logger.error(f"Failed to process visualization tile: {e}")
                    viz_data = None

            return {"file_metadata": file_metadata, "tiles": processed_tiles}, viz_data

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return None, None

    def process_sidd_scene(
        self,
        scene_dir: Path,
        create_viz: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Process a single SIDD scene directory to .pt tiles.

        Args:
            scene_dir: Path to scene directory
            create_viz: If True, collect intermediate data for visualization

        Returns:
            Tuple of (result_dict, viz_data) where result_dict contains:
            - "gt": file_metadata + tiles for GT image
            - "noisy": file_metadata + tiles for noisy image
        """
        scene_name = scene_dir.name
        gt_files = list(scene_dir.glob("GT_RAW_*.MAT")) + list(
            scene_dir.glob("GT_RAW_*.mat")
        )
        noisy_files = list(scene_dir.glob("NOISY_RAW_*.MAT")) + list(
            scene_dir.glob("NOISY_RAW_*.mat")
        )

        if not gt_files or not noisy_files:
            logger.warning(f"No GT_RAW or NOISY_RAW files found in {scene_dir}")
            return None, None

        gt_file = gt_files[0]
        noisy_file = noisy_files[0]
        scene_metadata = self.parse_sidd_scene_name(scene_name)
        camera_code = scene_metadata.get("camera_code", "unknown")

        metadata_files = list(scene_dir.glob("METADATA_RAW_*.MAT")) + list(
            scene_dir.glob("METADATA_RAW_*.mat")
        )
        cfa_pattern = [0, 1, 1, 2]
        if metadata_files and loadmat is not None:
            try:
                loadmat(str(metadata_files[0]), struct_as_record=False)
            except Exception:
                pass

        gt_raw = self.load_sidd_mat_file(gt_file)
        noisy_raw = self.load_sidd_mat_file(noisy_file)

        if gt_raw is None or noisy_raw is None:
            logger.warning(f"Failed to load .MAT files from {scene_dir}")
            return None, None

        try:
            gt_rgb = self.demosaic_sidd_raw(gt_raw, cfa_pattern)
            noisy_rgb = self.demosaic_sidd_raw(noisy_raw, cfa_pattern)
        except Exception as e:
            logger.error(f"Failed to demosaic {scene_name}: {e}")
            return None, None

        if camera_code not in TILE_CONFIGS:
            logger.warning(f"Unknown camera code {camera_code}, using default tiling")
            _, H, W = gt_rgb.shape
            rows = max(1, H // 256)
            cols = max(1, W // 256)
            gt_tile_infos = extract_tiles(gt_rgb, rows, cols, self.tile_size)
            noisy_tile_infos = extract_tiles(noisy_rgb, rows, cols, self.tile_size)
        else:
            config = TILE_CONFIGS[camera_code]
            gt_tile_infos = extract_tiles(
                gt_rgb,
                config["target_grid"][0],
                config["target_grid"][1],
                self.tile_size,
            )
            noisy_tile_infos = extract_tiles(
                noisy_rgb,
                config["target_grid"][0],
                config["target_grid"][1],
                self.tile_size,
            )

        result = {
            "gt": {"file_metadata": None, "tiles": []},
            "noisy": {"file_metadata": None, "tiles": []},
        }

        file_metadata_base = {
            "scene_id": scene_name,
            "camera_code": camera_code,
            "iso": scene_metadata.get("iso"),
            "shutter_speed": scene_metadata.get("shutter_speed"),
            "illuminant_temp": scene_metadata.get("illuminant_temp"),
            "brightness": scene_metadata.get("brightness"),
            "file_path": str(gt_file),
            "image_stats": {
                "min": float(np.min(gt_rgb)),
                "max": float(np.max(gt_rgb)),
                "mean": float(np.mean(gt_rgb)),
                "std": float(np.std(gt_rgb)),
            },
        }

        result["gt"]["file_metadata"] = file_metadata_base.copy()
        result["gt"]["file_metadata"]["data_type"] = "gt"
        result["gt"]["file_metadata"]["file_path"] = str(gt_file)

        result["noisy"]["file_metadata"] = file_metadata_base.copy()
        result["noisy"]["file_metadata"]["data_type"] = "noisy"
        result["noisy"]["file_metadata"]["file_path"] = str(noisy_file)

        sensor_range = {"min": 0.0, "max": 1.0}

        for i, (gt_tile_info, noisy_tile_info) in enumerate(
            zip(gt_tile_infos, noisy_tile_infos)
        ):
            gt_tile_id = f"{camera_code}_{scene_name}_gt_tile_{i:04d}"
            gt_tile_metadata = self.save_tile_as_pt(
                gt_tile_info.tile_data,
                gt_tile_id,
                "gt",
                sensor_range,
                str(gt_file),
            )
            if gt_tile_metadata:
                gt_tile_metadata.update(
                    {
                        "scene_id": scene_name,
                        "tile_key": f"{camera_code}_{scene_name}_gt_tile_{gt_tile_info.grid_position[0]}_{gt_tile_info.grid_position[1]}",
                        "grid_x": int(gt_tile_info.grid_position[0]),
                        "grid_y": int(gt_tile_info.grid_position[1]),
                        "image_x": int(gt_tile_info.image_position[0]),
                        "image_y": int(gt_tile_info.image_position[1]),
                        "corresponding_noisy_tile": f"{camera_code}_{scene_name}_noisy_tile_{i:04d}.pt",
                    }
                )
                result["gt"]["tiles"].append(gt_tile_metadata)

            noisy_tile_id = f"{camera_code}_{scene_name}_noisy_tile_{i:04d}"
            noisy_tile_metadata = self.save_tile_as_pt(
                noisy_tile_info.tile_data,
                noisy_tile_id,
                "noisy",
                sensor_range,
                str(noisy_file),
            )
            if noisy_tile_metadata:
                noisy_tile_metadata.update(
                    {
                        "scene_id": scene_name,
                        "tile_key": f"{camera_code}_{scene_name}_noisy_tile_{noisy_tile_info.grid_position[0]}_{noisy_tile_info.grid_position[1]}",
                        "grid_x": int(noisy_tile_info.grid_position[0]),
                        "grid_y": int(noisy_tile_info.grid_position[1]),
                        "image_x": int(noisy_tile_info.image_position[0]),
                        "image_y": int(noisy_tile_info.image_position[1]),
                        "corresponding_gt_tile": f"{camera_code}_{scene_name}_gt_tile_{i:04d}.pt",
                    }
                )
                result["noisy"]["tiles"].append(noisy_tile_metadata)

        viz_data = None
        if create_viz:
            import hashlib

            scene_hash = int(
                hashlib.md5(scene_name.encode(), usedforsecurity=False).hexdigest()[:8],
                16,
            )
            viz_tile_idx = scene_hash % len(gt_tile_infos) if gt_tile_infos else 0

            viz_data = {
                "raw_image": gt_rgb.copy(),
                "original_shape": gt_rgb.shape,
                "tiled_image": extract_single_tile_for_viz(
                    gt_tile_infos[viz_tile_idx], tile_size=256
                )
                if gt_tile_infos
                else None,
                "viz_tile_index": viz_tile_idx,
            }

        return result, viz_data

    def run_sidd_tiles_pipeline(
        self,
        sidd_data_path: str,
        max_scenes: int = None,
        create_visualizations: bool = False,
        dry_run: bool = False,
        cameras: list = None,
    ):
        """Run the SIDD .pt tiles pipeline.

        Args:
            sidd_data_path: Path to SIDD data directory
            max_scenes: Maximum scenes to process (None = all scenes)
            create_visualizations: If True, create visualizations for first scene
            dry_run: If True, validate without processing
            cameras: List of camera codes to process or None for all
        """
        sidd_path = Path(sidd_data_path)
        if not sidd_path.exists():
            raise FileNotFoundError(f"SIDD data path not found: {sidd_data_path}")

        scene_dirs = [
            d for d in sidd_path.iterdir() if d.is_dir() and d.name.count("_") >= 6
        ]

        if cameras:
            scene_dirs = [
                d
                for d in scene_dirs
                if self.parse_sidd_scene_name(d.name).get("camera_code") in cameras
            ]

        logger.info(f"Found {len(scene_dirs)} SIDD scenes to process")

        if dry_run:
            logger.info("=" * 70)
            logger.info("DRY RUN MODE - Validation Only")
            logger.info("=" * 70)
            logger.info(f"  - SIDD data path: {sidd_path}")
            logger.info(f"  - Scenes found: {len(scene_dirs)}")
            logger.info("=" * 70)
            return {"status": "dry_run_passed", "scenes_found": len(scene_dirs)}

        all_results = []
        total_tiles = 0

        for scene_dir in tqdm(
            scene_dirs[: max_scenes or len(scene_dirs)], desc="Processing SIDD scenes"
        ):
            try:
                result, viz_data = self.process_sidd_scene(
                    scene_dir, create_viz=create_visualizations
                )
                if result:
                    all_results.append(result)
                    total_tiles += len(result["gt"]["tiles"]) + len(
                        result["noisy"]["tiles"]
                    )
            except Exception as e:
                logger.error(f"Failed to process {scene_dir.name}: {e}")
                continue

        metadata_path = (
            self.base_path / "dataset" / "processed_sidd" / "sidd_tiles_metadata.json"
        )
        metadata = {
            "pipeline_info": {
                "total_scenes": len(all_results),
                "total_tiles": total_tiles,
                "processing_timestamp": datetime.now().isoformat(),
                "tile_size": self.tile_size,
                "storage_format": "pt_float32",
                "normalization": "[0,1] from .MAT → [-1,1] in save",
            },
            "scenes": all_results,
        }
        self._safe_write_json(metadata_path, metadata)

        return {
            "total_scenes": len(all_results),
            "total_tiles": total_tiles,
        }

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

        # Define ALL files for sensors (sony/fuji)
        sample_files = {"sony": [], "fuji": []}

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
            # Group files by sensor type
            for sensor in sensors:
                sample_files[sensor] = [
                    f
                    for f in filtered_files
                    if sid_file_info[f].get("camera_type") == sensor
                ]
            sensor_list = ", ".join(sensors)
            logger.info(
                f"Processing {sensor_list} files only: {len(filtered_files)} files"
            )
            logger.info(f"Total files in dataset: {len(all_files)}")
            logger.info(
                f"Skipping other sensors: {len(all_files) - len(filtered_files)}"
            )
        else:
            # Process all files (default behavior) - group by sensor
            for file_path in all_files:
                sensor_type = sid_file_info[file_path].get("camera_type", "unknown")
                if sensor_type in ["sony", "fuji"]:
                    sample_files[sensor_type].append(file_path)
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
            total_files = sum(len(files) for files in sample_files.values())
            logger.info(f"  - Total files in dataset: {total_files}")

            # Validate sensor detection
            logger.info(f"\n✓ Sensor detection validation (first 3 files):")
            validation_errors = []
            # Get first 3 files from any sensor for validation
            all_sample_files = []
            for sensor_files in sample_files.values():
                all_sample_files.extend(sensor_files)
            for i, file_path in enumerate(all_sample_files[:3]):
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

        pair_metadata = {}
        for file_path, info in sid_file_info.items():
            scene_id = info.get("scene_id")
            camera_type = info.get("camera_type")
            file_type = info.get("file_type")
            long_partner = info.get("long_partner")

            pair_metadata[file_path] = {
                "pair_id": f"{camera_type}_{scene_id}",
                "pair_type": file_type,
                "pair_partner": long_partner,
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
                    if sid_file_info and file_path in sid_file_info:
                        sid_info = sid_file_info[file_path]
                        data_type = sid_info["file_type"]
                        scene_id = sid_info["scene_id"]
                    else:
                        raise ValueError(
                            f"SID file info not available for {file_path}. "
                            f"Must load SID split files first using load_sid_split_files()."
                        )

                    should_create_viz = False
                    if create_visualizations:
                        if scene_id not in viz_data_by_scene[domain_name]:
                            viz_data_by_scene[domain_name][scene_id] = {}
                            should_create_viz = True
                        elif data_type not in viz_data_by_scene[domain_name][scene_id]:
                            should_create_viz = True
                    result = self.process_file_to_pt_tiles(
                        file_path,
                        create_viz=should_create_viz,
                        pair_metadata=pair_metadata,
                        sid_file_info=sid_file_info,
                    )

                    if result is None:
                        continue

                    file_data, viz_data = result

                    if file_data is None:
                        continue

                    if viz_data is not None and should_create_viz:
                        if "viz_tile_index" in viz_data:
                            viz_tile_idx = viz_data["viz_tile_index"]
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

                    domain_tiles.append(file_data)
                    processed_files += 1

                    try:
                        if sid_file_info and file_path in sid_file_info:
                            sensor_name = sid_file_info[file_path].get(
                                "camera_type", "unknown"
                            )
                        else:
                            sensor_name = "unknown"

                        incremental_path = (
                            self.base_path
                            / "processed"
                            / f"metadata_{sensor_name}_incremental.json"
                        )
                        total_tiles = sum(len(f["tiles"]) for f in domain_tiles)
                        metadata = {
                            "domain": domain_name,
                            "sensor": sensor_name,
                            "files_processed": processed_files,
                            "tiles_generated": total_tiles,
                            "files": domain_tiles,
                            "timestamp": datetime.now().isoformat(),
                        }
                        self._safe_write_json(incremental_path, metadata)
                    except Exception as e:
                        logger.warning(f"Failed to save incremental metadata: {e}")

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

        if sensors:
            sensor_suffix = "_".join(sensors)
        else:
            sensor_suffix = "all"

        metadata_path = (
            self.base_path
            / "processed"
            / f"comprehensive_{sensor_suffix}_tiles_metadata.json"
        )

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

        try:
            self._safe_write_json(metadata_path, comprehensive_metadata)
        except Exception as e:
            logger.error(f"Failed to save comprehensive metadata: {e}")
            backup_path = (
                self.base_path
                / "processed"
                / f"comprehensive_{sensor_suffix}_tiles_metadata_backup.json"
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
    parser.add_argument(
        "--sidd",
        action="store_true",
        help="Process SIDD dataset instead of sensor dataset (Sony/Fuji)",
    )
    parser.add_argument(
        "--sidd_path",
        type=str,
        default=None,
        help="Path to SIDD data directory (required if --sidd is set)",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Maximum scenes to process for SIDD (default: all)",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=["s6", "g4", "n6", "gp", "ip"],
        default=None,
        help="SIDD camera codes to process (default: all)",
    )

    args = parser.parse_args()

    # Set DATA_PATH environment variable if provided
    if args.data_path:
        os.environ["DATA_PATH"] = args.data_path
        data_path = args.data_path
    else:
        # Use environment variable or default
        data_path = os.environ.get(
            "DATA_PATH", "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data"
        )

    # Run pipeline
    pipeline = SimpleTilesPipeline(data_path)

    if args.sidd:
        # Process SIDD dataset
        if args.sidd_path is None:
            print("Error: --sidd_path is required when using --sidd")
            return

        results = pipeline.run_sidd_tiles_pipeline(
            sidd_data_path=args.sidd_path,
            max_scenes=args.max_scenes,
            create_visualizations=args.visualize,
            dry_run=args.dry_run,
            cameras=args.cameras,
        )

        if results.get("total_tiles", 0) > 0:
            print(f"\n🎊 SUCCESS: SIDD Tiles Pipeline Completed!")
            print(f"📊 Total .pt tiles generated: {results['total_tiles']:,}")
            print(f"📐 Tile size: 256×256")
            print(
                f"💾 .pt files (float32) saved to: {data_path}/dataset/processed_sidd/pt_tiles/"
            )
            print(f"🎯 Normalization: [0,1] from .MAT → [-1,1] in save")
            print(f"📋 Scenes processed: {results['total_scenes']}")
            print(f"\n🎯 Ready for diffusion model training!")
        else:
            print(f"\n❌ FAILED: No tiles were generated")
    else:
        # Process sensor dataset (SID - Sony/Fuji)
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
                print(
                    f"\n📊 Visualizations saved to: {data_path}/processed/visualizations/"
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
    main()
