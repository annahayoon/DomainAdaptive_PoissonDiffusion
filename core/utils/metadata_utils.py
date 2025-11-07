"""
Metadata utility functions for the Poisson-Gaussian Diffusion project.

This module provides functions for parsing, creating, and managing metadata
from raw image files, filenames, and dataset split files.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import shared utility from file_utils to avoid duplication
from core.utils.file_utils import _load_json_file

logger = logging.getLogger(__name__)


def determine_data_type(file_path: str) -> str:
    """Determine if file contains long or short exposure data."""
    file_path_lower = file_path.lower()
    return "long" if "/long/" in file_path_lower else "short"


def load_sid_split_files(sid_data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load SID dataset split files and parse ISO/aperture information.
    Loads ALL exposure files listed in the split files (*.txt).

    Args:
        sid_data_path: Path to SID dataset root directory

    Returns:
        Dictionary mapping file paths to their split and metadata information
    """
    try:
        from config.config import CAMERA_CONFIGS
    except ImportError:
        logger.warning("core.config not available, CAMERA_CONFIGS not loaded")
        return {}

    sid_path = Path(sid_data_path)
    splits_path = sid_path / "splits"
    file_info = {}

    split_names = ["train", "val", "test"]
    split_files_data = {}

    for camera_type, config in CAMERA_CONFIGS.items():
        camera_name = config["base_dir"]
        for split_name in split_names:
            filename = f"{camera_name}_{split_name}_list.txt"
            split_file_path = splits_path / filename

            if not split_file_path.exists():
                logger.warning(f"SID split file not found: {split_file_path}")
                continue

            with open(split_file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue

                    short_path, long_path, iso_str, aperture_str = parts
                    short_abs_path = str(sid_path / short_path[2:])
                    long_abs_path = str(sid_path / long_path[2:])

                    for file_path, file_type in [
                        (short_abs_path, "short"),
                        (long_abs_path, "long"),
                    ]:
                        filename_parts = Path(file_path).stem.split("_")
                        if len(filename_parts) >= 3:
                            scene_id = filename_parts[0]
                            exposure_time = filename_parts[2]

                            key = (
                                camera_type,
                                scene_id,
                                file_type,
                                exposure_time,
                                file_path,
                            )
                            split_files_data[key] = {
                                "file_path": file_path,
                                "split": split_name,
                                "iso": int(iso_str.replace("ISO", ""))
                                if iso_str.startswith("ISO")
                                else None,
                                "aperture": float(aperture_str.replace("F", ""))
                                if aperture_str.startswith("F")
                                else None,
                            }

    scene_groups = {}
    for (
        camera_type,
        scene_id,
        file_type,
        exposure_time,
        file_path,
    ), file_data in split_files_data.items():
        if camera_type not in scene_groups:
            scene_groups[camera_type] = {}
        if scene_id not in scene_groups[camera_type]:
            scene_groups[camera_type][scene_id] = {"short": [], "long": []}

        scene_groups[camera_type][scene_id][file_type].append(
            {
                "file_path": file_data["file_path"],
                "exposure_time": exposure_time,
                "iso": file_data["iso"],
                "aperture": file_data["aperture"],
                "split": file_data["split"],
            }
        )

    for camera_type, scenes in scene_groups.items():
        for scene_id, files_by_type in scenes.items():
            short_exposures = [f["exposure_time"] for f in files_by_type["short"]]
            long_exposures = [f["exposure_time"] for f in files_by_type["long"]]
            long_partner = (
                files_by_type["long"][0]["file_path"] if files_by_type["long"] else None
            )

            for exposure_type in ["short", "long"]:
                for file_data in files_by_type[exposure_type]:
                    file_path = file_data["file_path"]
                    exposure_time_str = file_data["exposure_time"]

                    metadata = {
                        "file_path": file_path,
                        "split": file_data["split"],
                        "scene_id": scene_id,
                        "file_type": exposure_type,
                        "camera_type": camera_type,
                        "exposure_time": exposure_time_str,
                        "iso": file_data["iso"],
                        "aperture": file_data["aperture"],
                        "all_short_exposures": short_exposures,
                        "all_long_exposures": long_exposures,
                        "long_partner": long_partner,
                    }

                    file_info[file_path] = metadata

    return file_info


def _parse_filename_metadata(
    file_path: str,
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Parse scene_id, camera_model, and exposure_time from filename."""
    try:
        from core.sensor_detector import SensorDetector
    except ImportError:
        return None, None, None

    file_path_obj = Path(file_path)
    filename_parts = file_path_obj.stem.split("_")

    scene_id = filename_parts[0] if len(filename_parts) >= 3 else None

    exposure_time = None
    if len(filename_parts) >= 3:
        try:
            exposure_time = float(filename_parts[2].rstrip("s"))
        except ValueError:
            pass

    camera_model = None
    try:
        sensor_type_detected = SensorDetector.detect(file_path)
        camera_model = sensor_type_detected.value
    except Exception:
        pass

    return scene_id, camera_model, exposure_time


def _create_base_metadata(
    file_path: str, sensor_type: str, pair_info: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Create base metadata dictionary with common fields."""
    scene_id, camera_model, exposure_time = _parse_filename_metadata(file_path)

    metadata = {
        "file_path": file_path,
        "sensor_type": sensor_type,
        "camera_model": camera_model,
        "exposure_time": exposure_time,
        "scene_id": scene_id,
    }

    if pair_info:
        metadata.update(pair_info)

    return metadata


def _safe_tolist(value) -> Any:
    """Convert numpy array to list if possible, otherwise return original value."""
    return value.tolist() if hasattr(value, "tolist") else value


def extract_raw_metadata(
    raw, file_path: str, packed_image: np.ndarray, sensor_type: str
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from raw file using rawpy.

    Args:
        raw: rawpy object with opened raw file
        file_path: Path to the raw file
        packed_image: Packed image array (for channel count)
        sensor_type: Sensor type ('sony' or 'fuji')

    Returns:
        Dictionary containing all metadata from the raw file
    """
    pattern = raw.raw_pattern
    pattern_shape = pattern.shape
    black_level = np.array(raw.black_level_per_channel)
    white_level = raw.white_level
    metadata = _create_base_metadata(file_path, sensor_type)

    metadata.update(
        {
            "black_level": black_level.tolist(),
            "white_level": int(white_level),
            "raw_height": raw.sizes.raw_height,
            "raw_width": raw.sizes.raw_width,
            "height": raw.sizes.height,
            "width": raw.sizes.width,
            "pixel_aspect": raw.sizes.pixel_aspect,
            "color_desc": raw.color_desc.decode() if raw.color_desc else None,
            "num_colors": raw.num_colors,
            "raw_pattern": pattern.tolist(),
            "pattern_shape": pattern_shape,
            "processing": "packed_raw_no_demosaic",
            "channels": packed_image.shape[0],
            "camera_whitebalance": _safe_tolist(raw.camera_whitebalance),
            "daylight_whitebalance": _safe_tolist(raw.daylight_whitebalance),
            "color_matrix": _safe_tolist(raw.color_matrix),
            "rgb_xyz_matrix": _safe_tolist(raw.rgb_xyz_matrix),
            "raw_type": str(raw.raw_type),
        }
    )

    return metadata


def _create_scene_dict() -> Dict[str, Any]:
    """Create a defaultdict for storing scene files."""
    return defaultdict(lambda: {"long": None, "short": []})


def _select_short_exposure(
    short_files: List[Tuple[str, str]], camera_type: str
) -> Optional[str]:
    """Select the appropriate short exposure file based on camera type."""
    if camera_type == "sony":
        return next((p for e, p in short_files if e == "0.04s"), None)
    return sorted(short_files)[0][1] if short_files else None


def find_photography_pairs(
    all_photo_files: List[Path],
) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Select long/short exposure pairs for sensors (SID dataset - Sony/Fuji).

    Pairing strategy:
    - Long: Long exposure files (in /long/ directory) - ground truth
    - Short: First short exposure file per scene (0.04s for Sony, first short for Fuji) - input

    Args:
        all_photo_files: List of all .ARW and .RAF file paths

    Returns:
        Tuple of (selected_file_paths, pair_metadata) where:
        - selected_file_paths: List of selected file paths (long + short pairs)
        - pair_metadata: Dict mapping file_path -> metadata with pair information
    """
    try:
        from config.config import CAMERA_CONFIGS
        from core.sensor_detector import SensorDetector
    except ImportError:
        logger.warning("preprocessing modules not available")
        return [], {}

    camera_scenes = {cam: _create_scene_dict() for cam in CAMERA_CONFIGS.keys()}

    for file_path in all_photo_files:
        try:
            sensor_type = SensorDetector.detect(str(file_path))
            camera_type = sensor_type.value
        except Exception:
            continue

        if not camera_type:
            continue

        parts = Path(file_path).stem.split("_")
        if len(parts) >= 3:
            scene_id, exposure = parts[0], parts[2]
            data_type = determine_data_type(str(file_path))

            if data_type == "long":
                camera_scenes[camera_type][scene_id]["long"] = str(file_path)
            elif data_type == "short":
                camera_scenes[camera_type][scene_id]["short"].append(
                    (exposure, str(file_path))
                )

    selected_files, pair_metadata = [], {}

    for camera_type, scenes in camera_scenes.items():
        for scene_id, files in scenes.items():
            if not (files["long"] and files["short"]):
                continue

            short_file = _select_short_exposure(files["short"], camera_type)
            if not short_file:
                continue

            long_file, pair_id = files["long"], f"{camera_type}_{scene_id}"
            long_pair_info = {
                "pair_id": pair_id,
                "pair_type": "long",
                "pair_partner": short_file,
            }
            short_pair_info = {
                "pair_id": pair_id,
                "pair_type": "short",
                "pair_partner": long_file,
            }

            selected_files.extend([long_file, short_file])
            pair_metadata[long_file] = _create_base_metadata(
                long_file, camera_type, long_pair_info
            )
            pair_metadata[short_file] = _create_base_metadata(
                short_file, camera_type, short_pair_info
            )

    return selected_files, pair_metadata


def _add_optional_fields(
    target: Dict[str, Any], source: Dict[str, Any], *keys: str
) -> None:
    """Add optional fields from source to target if they exist."""
    for key in keys:
        if key in source and source[key] is not None:
            target[key] = source[key]


def create_file_metadata(
    file_path: str,
    metadata: Dict[str, Any],
    sid_info: Dict[str, Any],
    pair_info: Optional[Dict[str, str]] = None,
    domain_range: Optional[Dict[str, float]] = None,
    image_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Create consolidated file-level metadata without duplication."""
    file_metadata = {
        "file_path": file_path,
        "split": sid_info.get("split"),
        "scene_id": sid_info.get("scene_id"),
        "data_type": sid_info.get("file_type"),
        "camera_model": metadata.get("camera_model"),
        "exposure_time": metadata.get("exposure_time"),
        "camera_type": sid_info.get("camera_type"),
    }

    _add_optional_fields(
        file_metadata,
        sid_info,
        "iso",
        "aperture",
        "all_short_exposures",
        "all_long_exposures",
        "long_partner",
    )

    if "black_level" in metadata:
        black_level = metadata["black_level"]
        if isinstance(black_level, (list, np.ndarray)):
            file_metadata["black_level"] = float(np.mean(black_level))
        else:
            file_metadata["black_level"] = float(black_level)

    if "white_level" in metadata:
        file_metadata["white_level"] = int(metadata["white_level"])

    if domain_range:
        file_metadata["domain_range"] = domain_range
    if image_stats:
        file_metadata["image_stats"] = image_stats
    if pair_info:
        file_metadata.update(pair_info)

    return file_metadata


def load_comprehensive_metadata(comprehensive_json_path: Path) -> Dict[str, Dict]:
    """Load comprehensive metadata and create tile_id -> data_type mapping.

    Args:
        comprehensive_json_path: Path to comprehensive metadata JSON file

    Returns:
        Dictionary mapping tile_id to sensor_type, data_type, and tile_stats
    """
    logger.info(f"Loading comprehensive metadata: {comprehensive_json_path.name}...")
    data = _load_json_file(comprehensive_json_path)

    tile_info_map = {}
    files = data.get("files", [])

    for file_entry in files:
        tiles = file_entry.get("tiles", [])
        for tile in tiles:
            tile_id = tile.get("tile_id", "")
            sensor_type = tile.get("sensor_type", "unknown")
            data_type = tile.get("data_type", "unknown")

            if tile_id and sensor_type != "unknown" and data_type != "unknown":
                tile_info_map[tile_id] = {
                    "sensor_type": sensor_type,
                    "data_type": data_type,
                    "tile_stats": tile.get("tile_stats", {}),
                }

    logger.info(f"Loaded {len(tile_info_map)} tile entries")
    return tile_info_map
