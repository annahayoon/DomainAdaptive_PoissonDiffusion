"""
Demosaicing utilities for SIDD dataset.

Provides functions for loading Bayer pattern mappings, extracting camera IDs,
converting patterns, and demosaicing 2D Bayer patterns to RGB.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from config.logging_config import get_logger

_IDX_TO_CHAR_LOWER = {0: "r", 1: "g", 2: "b"}
_IDX_TO_CHAR_UPPER = {0: "R", 1: "G", 2: "B"}
_CHAR_TO_IDX = {"r": 0, "g": 1, "b": 2}

_BAYER_PATTERN_CACHE: Optional[Dict[str, str]] = None


def load_bayer_patterns_csv(csv_path: Optional[Path] = None) -> Dict[str, str]:
    """Load Bayer pattern mapping from CSV file.

    CSV format: camera_id,bayer_pattern
    Example: S6,grbg

    Args:
        csv_path: Path to bayer_patterns.csv. If None, searches in common locations.

    Returns:
        Dictionary mapping camera_id -> bayer_pattern (e.g., {"S6": "grbg", "IP": "rggb"})
    """
    global _BAYER_PATTERN_CACHE

    if _BAYER_PATTERN_CACHE is not None:
        return _BAYER_PATTERN_CACHE

    if csv_path is None:
        project_root = Path(__file__).parent.parent
        possible_paths = [
            Path(
                "/home/jilab/Jae/external/dataset/sidd/SIDD_Medium_Raw/bayer_patterns.csv"
            ),
            project_root
            / "external"
            / "dataset"
            / "sidd"
            / "SIDD_Medium_Raw"
            / "bayer_patterns.csv",
            project_root / "external" / "dataset" / "sidd" / "bayer_patterns.csv",
        ]
        csv_path = next((p for p in possible_paths if p.exists()), None)

    if csv_path is None or not csv_path.exists():
        logger = get_logger(__name__)
        logger.warning(
            "Bayer patterns CSV not found, using default RGGB for all cameras"
        )
        _BAYER_PATTERN_CACHE = {}
        return _BAYER_PATTERN_CACHE

    pattern_map = {}
    logger = get_logger(__name__)
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                camera_id = row.get("camera_id", "").strip().upper()
                bayer_pattern = row.get("bayer_pattern", "").strip().lower()
                if camera_id and bayer_pattern:
                    pattern_map[camera_id] = bayer_pattern
        logger.info(f"Loaded {len(pattern_map)} Bayer patterns from {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to load Bayer patterns CSV: {e}, using defaults")
        pattern_map = {}

    _BAYER_PATTERN_CACHE = pattern_map
    return pattern_map


def extract_camera_id_from_scene_name(scene_name: str) -> Optional[str]:
    """Extract camera ID from SIDD scene name.

    SIDD scene names typically have format: {number}_{number}_{CAMERA_ID}_{...}
    Examples:
        "0080_004_S6_00200_00050_3200_N" -> "S6"
        "0073_003_IP_00200_01000_5500_L" -> "IP"

    Args:
        scene_name: Scene directory name

    Returns:
        Camera ID (e.g., "S6", "IP", "GP") or None if not found
    """
    parts = scene_name.split("_")
    if len(parts) >= 3:
        camera_id = parts[2].strip().upper()
        if len(camera_id) in [2, 3] and camera_id.isalnum():
            return camera_id
    return None


def bayer_pattern_string_to_indices(pattern_str: str) -> List[int]:
    """Convert Bayer pattern string to numeric indices.

    Pattern strings: "rggb", "bggr", "grbg", "gbrg"
    Numeric format: [0, 1, 1, 2] where 0=R, 1=G, 2=B

    Args:
        pattern_str: Bayer pattern string (lowercase, e.g., "rggb")

    Returns:
        List of 4 indices [r, g, g, b] where 0=R, 1=G, 2=B
    """
    pattern_str = pattern_str.lower().strip()
    logger = get_logger(__name__)

    if len(pattern_str) != 4:
        logger.warning(
            f"Invalid Bayer pattern string '{pattern_str}', using default RGGB"
        )
        return [0, 1, 1, 2]

    indices = []
    for char in pattern_str:
        if char in _CHAR_TO_IDX:
            indices.append(_CHAR_TO_IDX[char])
        else:
            logger.warning(
                f"Invalid character '{char}' in Bayer pattern '{pattern_str}', using default RGGB"
            )
            return [0, 1, 1, 2]

    return indices


def get_cfa_pattern_from_scene_name(
    scene_name: str,
    bayer_patterns_map: Optional[Dict[str, str]] = None,
    return_string: bool = False,
) -> Optional[Union[List[int], str]]:
    """Get CFA pattern from scene name using camera ID lookup.

    Args:
        scene_name: Scene directory name
        bayer_patterns_map: Optional pre-loaded Bayer pattern mapping. If None, loads from CSV.
        return_string: If True, return pattern string (e.g., "rggb"). If False, return numeric indices.

    Returns:
        If return_string=False: CFA pattern as list [r, g, g, b] indices (0=R, 1=G, 2=B), or None if not found
        If return_string=True: Bayer pattern string (e.g., "rggb"), or None if not found
    """
    if bayer_patterns_map is None:
        bayer_patterns_map = load_bayer_patterns_csv()

    camera_id = extract_camera_id_from_scene_name(scene_name)
    if camera_id is None:
        return None

    bayer_pattern_str = bayer_patterns_map.get(camera_id)
    if bayer_pattern_str is None:
        return None

    return (
        bayer_pattern_str
        if return_string
        else bayer_pattern_string_to_indices(bayer_pattern_str)
    )


def get_opencv_demosaic_flag(
    cfa_pattern: List[int], output_channel_order: str = "RGB", alg_type: str = "VNG"
) -> int:
    """Get OpenCV demosaicing flag based on CFA pattern.

    Args:
        cfa_pattern: 4-element list [r, g, g, b] where 0=R, 1=G, 2=B
        output_channel_order: 'RGB' or 'BGR'
        alg_type: '' (simple), 'EA' (edge-aware), or 'VNG' (variable number of gradients)

    Returns:
        OpenCV COLOR_BAYER_* flag
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for demosaicing")

    flag_map = {
        ((0, 1, 1, 2), "BGR", ""): "COLOR_BAYER_BG2BGR",
        ((2, 1, 1, 0), "BGR", ""): "COLOR_BAYER_RG2BGR",
        ((1, 0, 2, 1), "BGR", ""): "COLOR_BAYER_GB2BGR",
        ((1, 2, 0, 1), "BGR", ""): "COLOR_BAYER_GR2BGR",
        ((0, 1, 1, 2), "RGB", ""): "COLOR_BAYER_BG2RGB",
        ((2, 1, 1, 0), "RGB", ""): "COLOR_BAYER_RG2RGB",
        ((1, 0, 2, 1), "RGB", ""): "COLOR_BAYER_GB2RGB",
        ((1, 2, 0, 1), "RGB", ""): "COLOR_BAYER_GR2RGB",
        ((0, 1, 1, 2), "BGR", "EA"): "COLOR_BAYER_BG2BGR_EA",
        ((2, 1, 1, 0), "BGR", "EA"): "COLOR_BAYER_RG2BGR_EA",
        ((1, 0, 2, 1), "BGR", "EA"): "COLOR_BAYER_GB2BGR_EA",
        ((1, 2, 0, 1), "BGR", "EA"): "COLOR_BAYER_GR2BGR_EA",
        ((0, 1, 1, 2), "RGB", "EA"): "COLOR_BAYER_BG2RGB_EA",
        ((2, 1, 1, 0), "RGB", "EA"): "COLOR_BAYER_RG2RGB_EA",
        ((1, 0, 2, 1), "RGB", "EA"): "COLOR_BAYER_GB2RGB_EA",
        ((1, 2, 0, 1), "RGB", "EA"): "COLOR_BAYER_GR2RGB_EA",
        ((0, 1, 1, 2), "BGR", "VNG"): "COLOR_BAYER_BG2BGR_VNG",
        ((2, 1, 1, 0), "BGR", "VNG"): "COLOR_BAYER_RG2BGR_VNG",
        ((1, 0, 2, 1), "BGR", "VNG"): "COLOR_BAYER_GB2BGR_VNG",
        ((1, 2, 0, 1), "BGR", "VNG"): "COLOR_BAYER_GR2BGR_VNG",
        ((0, 1, 1, 2), "RGB", "VNG"): "COLOR_BAYER_BG2RGB_VNG",
        ((2, 1, 1, 0), "RGB", "VNG"): "COLOR_BAYER_RG2RGB_VNG",
        ((1, 0, 2, 1), "RGB", "VNG"): "COLOR_BAYER_GB2RGB_VNG",
        ((1, 2, 0, 1), "RGB", "VNG"): "COLOR_BAYER_GR2RGB_VNG",
    }

    key = (tuple(cfa_pattern), output_channel_order, alg_type)
    flag_name = flag_map.get(
        key,
        "COLOR_BAYER_BG2RGB_VNG"
        if output_channel_order == "RGB"
        else "COLOR_BAYER_BG2BGR_VNG",
    )

    return getattr(cv2, flag_name)


def demosaic_bayer_to_rgb(
    bayer_2d: np.ndarray,
    cfa_pattern: Union[str, List[int]],
    output_channel_order: str = "RGB",
    alg_type: str = "VNG",
) -> np.ndarray:
    """Demosaic 2D Bayer pattern to RGB.

    Supports both pattern string (e.g., "rggb") and numeric indices (e.g., [0,1,1,2]).
    Uses OpenCV or colour-demosaicing library.

    Args:
        bayer_2d: 2D Bayer array with shape (H, W) in [0, 1] range
        cfa_pattern: CFA pattern as string (e.g., "rggb") or list [0,1,1,2]. Can be lowercase.
        output_channel_order: 'RGB' or 'BGR' (only used for OpenCV demosaicing)
        alg_type: Algorithm type: '' (simple), 'EA' (edge-aware), 'VNG' (variable number of gradients), or 'menon2007'

    Returns:
        RGB image with shape (3, H, W) in [0, 1] range
    """
    if isinstance(cfa_pattern, str):
        pattern_indices = bayer_pattern_string_to_indices(cfa_pattern)
        pattern_str_lower = cfa_pattern.lower()
    else:
        pattern_indices = cfa_pattern
        pattern_str_lower = "".join(
            [_IDX_TO_CHAR_LOWER.get(idx, "r") for idx in cfa_pattern]
        )

    pattern_str_upper = pattern_str_lower.upper()

    if alg_type in ["", "EA", "VNG"]:
        try:
            import cv2

            max_val = 255 if alg_type == "VNG" else 16383
            raw_uint = (bayer_2d * max_val).astype(
                np.uint8 if alg_type == "VNG" else np.uint16
            )

            flag = get_opencv_demosaic_flag(
                pattern_indices, output_channel_order, alg_type
            )
            rgb = cv2.cvtColor(raw_uint, flag)
            rgb = rgb.astype(np.float32) / max_val
            return np.transpose(rgb, (2, 0, 1))
        except ImportError:
            logger = get_logger(__name__)
            logger.warning("OpenCV not available, trying colour-demosaicing")

    pattern_str_for_demosaic = "".join(
        [_IDX_TO_CHAR_UPPER.get(idx, "R") for idx in pattern_indices]
    )

    try:
        from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007

        rgb = demosaicing_CFA_Bayer_Menon2007(
            bayer_2d, pattern=pattern_str_for_demosaic
        )
        rgb = rgb.astype(np.float32)
        return np.transpose(rgb, (2, 0, 1))
    except ImportError:
        raise ImportError(
            "Need either OpenCV or colour-demosaicing for demosaicing. "
            "Install with: pip install opencv-python OR pip install colour-demosaicing"
        )
