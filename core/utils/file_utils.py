"""
File I/O utility functions for the Poisson-Gaussian Diffusion project.

This module provides file loading, saving, and path resolution utilities.
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

try:
    from scipy.io import loadmat

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    loadmat = None

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

logger = logging.getLogger(__name__)


def load_mat_file(file_path: Path, variable_name: str = "x") -> Optional[np.ndarray]:
    """Load .mat file and extract variable. Supports MATLAB v7 and v7.3 (HDF5) formats."""
    if not file_path.exists():
        return None

    if HAS_H5PY:
        try:
            with h5py.File(str(file_path), "r") as f:
                if variable_name in f:
                    ref = f[variable_name]
                    data = (
                        np.array(ref)
                        if isinstance(ref, h5py.Dataset)
                        else np.array(f[ref[0, 0]])
                    )
                elif len(f.keys()) == 1:
                    key = list(f.keys())[0]
                    ref = f[key]
                    data = (
                        np.array(ref)
                        if isinstance(ref, h5py.Dataset)
                        else np.array(f[ref[0, 0]])
                    )
                else:
                    return None

                if data.ndim == 2:
                    data = data.T
                elif data.ndim == 3:
                    data = np.transpose(data, (2, 1, 0))

                return data.astype(np.float32)
        except (OSError, ValueError, KeyError):
            pass

    if not HAS_SCIPY or loadmat is None:
        return None

    try:
        mat_data = loadmat(str(file_path))
        keys = [k for k in mat_data.keys() if not k.startswith("__")]

        if variable_name in mat_data:
            data = mat_data[variable_name]
        elif len(keys) == 1:
            data = mat_data[keys[0]]
        else:
            return None

        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        return None

    except NotImplementedError:
        return None
    except Exception:
        return None


def load_metadata_from_mat(file_path: Path) -> dict:
    """Load metadata from .mat file and extract black level and white level."""
    if not HAS_SCIPY or loadmat is None:
        raise ImportError("scipy.io.loadmat is required for loading .mat files")

    try:
        mat_data = loadmat(str(file_path))

        # Try to extract black level and white level
        black_level = None
        white_level = None

        # Common keys in MATLAB metadata files
        possible_keys = [
            "BlackLevel",
            "black_level",
            "WhiteLevel",
            "white_level",
            "metadata",
            "Metadata",
            "info",
            "Info",
        ]

        for key in possible_keys:
            if key in mat_data:
                value = mat_data[key]
                if isinstance(value, np.ndarray):
                    value = value.flatten()
                    if len(value) > 0:
                        if "black" in key.lower():
                            black_level = float(value[0])
                        elif "white" in key.lower():
                            white_level = float(value[0])

        return {
            "black_level": black_level,
            "white_level": white_level,
            "raw_data": mat_data,
        }
    except Exception as e:
        logger.warning(f"Error loading metadata from {file_path}: {e}")
        return {"black_level": None, "white_level": None, "raw_data": {}}


def save_json_file(
    file_path: Path,
    data: Dict[str, Any],
    default: Optional[Any] = None,
    atomic: bool = False,
) -> None:
    """Save JSON data to file, optionally using atomic writes.

    Args:
        file_path: Target file path
        data: Dictionary to serialize
        default: Default function for JSON serialization
        atomic: If True, write to temp file first then atomically move (prevents partial writes)
    """
    _ensure_dir(file_path.parent)

    if atomic:
        # Write to temporary file in same directory (for atomic rename on Unix)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=file_path.parent, delete=False
        ) as tmp:
            json.dump(data, tmp, indent=2, default=default)
            tmp_path = tmp.name

        # Atomic move on Unix, force overwrite on Windows
        try:
            shutil.move(tmp_path, str(file_path))
        except Exception as e:
            # Cleanup temp file on error
            try:
                os.unlink(tmp_path)
            except:
                pass
            raise e
    else:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=default)


def _load_json_file(json_path: Path) -> Dict[str, Any]:
    """Internal function to load JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def load_tensor_from_pt(pt_path: Path) -> torch.Tensor:
    """Load tensor from .pt file."""
    return torch.load(pt_path, map_location="cpu")


def load_stitched_image(file_path: Path, device: str = "cpu") -> Optional[torch.Tensor]:
    """
    Load a stitched .pt tensor file.

    This is a unified function for loading stitched images from analysis scripts.
    Handles both individual .pt files and stitched scene files.

    Args:
        file_path: Path to the .pt file
        device: Device to load the tensor on (default: 'cpu')

    Returns:
        Loaded tensor, or None if loading fails
    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return None

    try:
        tensor = torch.load(file_path, map_location=device)
        # Ensure it's a tensor
        if not isinstance(tensor, torch.Tensor):
            logger.warning(f"File {file_path} does not contain a tensor")
            return None

        # Remove batch dimension if present (common in stitched scenes)
        if tensor.ndim == 4:
            tensor = tensor[0]

        return tensor
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def load_image_tensor(image_path: Path) -> Tuple[Optional[np.ndarray], bool]:
    """
    Load a .pt file and return as numpy array.

    Args:
        image_path: Path to .pt file

    Returns:
        Tuple of (image_array, is_rgb)
    """
    if not image_path.exists():
        return None, False

    try:
        tensor = torch.load(image_path, map_location="cpu")
        # Convert to numpy and squeeze singleton dimensions
        array = tensor.numpy() if hasattr(tensor, "numpy") else tensor
        is_rgb = False

        # Handle different tensor shapes
        if len(array.shape) == 4:  # (1, 1, 256, 256) or (1, 3, 256, 256)
            if array.shape[1] == 3:  # RGB
                array = array[0]  # Remove batch dimension, keep (3, 256, 256)
                is_rgb = True
            else:
                array = array[0, 0]  # Remove batch and channel dimensions
        elif len(array.shape) == 3:
            if array.shape[0] == 1:  # (1, 256, 256)
                array = array[0]  # Remove channel dimension
            elif array.shape[0] == 3:  # (3, 256, 256) - RGB image
                is_rgb = True
                # array stays as (3, 256, 256)
            else:  # (256, 256, 1) or similar
                array = array.squeeze()
        elif len(array.shape) == 2:  # (256, 256)
            pass  # Already correct shape
        else:
            logger.warning(f"Unexpected array shape: {array.shape} for {image_path}")
            return None, False

        return array, is_rgb
    except Exception as e:
        logger.warning(f"Error loading {image_path}: {e}")
        return None, False


def find_split_file(sensor: str, split_type: str = "test") -> Optional[Path]:
    """Find split file using standard pattern.

    Tries in order:
    1. dataset/splits/{Sensor}_{split_type}_list.txt
    2. dataset/processed/../splits/{Sensor}_{split_type}_list.txt

    Sensor name is capitalized: first letter uppercase, rest lowercase.
    """
    sensor_capitalized = sensor.capitalize()

    splits_dir = Path("dataset/splits")
    split_file = splits_dir / f"{sensor_capitalized}_{split_type}_list.txt"
    if split_file.exists():
        return split_file

    data_root = Path("dataset/processed")
    splits_dir_fallback = data_root.parent / "splits"
    split_file_fallback = (
        splits_dir_fallback / f"{sensor_capitalized}_{split_type}_list.txt"
    )
    if split_file_fallback.exists():
        return split_file_fallback

    return None


def find_metadata_json(
    sensor: str, data_root: Path = Path("dataset/processed")
) -> Optional[Path]:
    """Find metadata JSON using standard pattern.

    Tries in order:
    1. comprehensive_{sensor}_tiles_metadata.json
    2. comprehensive_all_tiles_metadata.json
    """
    sensor_metadata = data_root / f"comprehensive_{sensor}_tiles_metadata.json"
    if sensor_metadata.exists():
        return sensor_metadata

    all_metadata = data_root / "comprehensive_all_tiles_metadata.json"
    if all_metadata.exists():
        return all_metadata

    return None


def find_latest_model(sensor: str) -> Optional[Path]:
    """Find latest model for sensor.

    Pattern: results/edm_{sensor}_training_*/best_model.pkl
    Returns most recently modified training directory.
    """
    training_dirs = [
        d for d in Path("results").glob(f"edm_{sensor}_training_*") if d.is_dir()
    ]

    if not training_dirs:
        return None

    training_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for training_dir in training_dirs:
        model_path = training_dir / "best_model.pkl"
        if model_path.exists():
            return model_path

    return None


def parse_filename_from_split_line(line: str) -> Optional[Tuple[str, str, str]]:
    """Parse (scene_id, frame_id, exposure_str) from a split file line.

    Returns None if parsing fails.
    """
    parts = line.strip().split()
    if len(parts) < 2:
        return None

    short_filename = Path(parts[0]).name
    filename_no_ext = (
        short_filename.replace(".ARW", "")
        .replace(".RAF", "")
        .replace(".arw", "")
        .replace(".raf", "")
    )
    name_parts = filename_no_ext.split("_")

    if len(name_parts) >= 3:
        return (name_parts[0], name_parts[1], name_parts[2])
    return None


def parse_exposure_from_split_line(line: str) -> Optional[Tuple[str, float, float]]:
    """Parse (scene_id, short_exposure, long_exposure) from split file line.

    Format: ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9
    Returns: (scene_id, short_exposure, long_exposure) or None
    """
    parts = line.strip().split()
    if len(parts) < 2:
        return None

    short_path = parts[0]
    long_path = parts[1]

    short_filename = Path(short_path).name
    short_parts = (
        short_filename.replace(".ARW", "")
        .replace(".RAF", "")
        .replace(".arw", "")
        .replace(".raf", "")
        .split("_")
    )
    if len(short_parts) >= 3:
        scene_id = short_parts[0]
        try:
            short_exposure_str = short_parts[2]
            short_exposure = float(short_exposure_str.replace("s", ""))

            long_filename = Path(long_path).name
            long_parts = (
                long_filename.replace(".ARW", "")
                .replace(".RAF", "")
                .replace(".arw", "")
                .replace(".raf", "")
                .split("_")
            )
            if len(long_parts) >= 3:
                long_exposure_str = long_parts[2]
                long_exposure = float(long_exposure_str.replace("s", ""))

                return (scene_id, short_exposure, long_exposure)
        except (ValueError, IndexError):
            pass

    return None


def parse_exposure_from_scene_dir(scene_dir: Path) -> Optional[float]:
    """
    Parse exposure time from scene directory name.

    Format: scene_10019_fuji_00_0.1s -> 0.1

    Args:
        scene_dir: Path to scene directory

    Returns:
        Exposure time in seconds, or None if parsing fails
    """
    dir_name = scene_dir.name
    if dir_name.startswith("scene_"):
        parts = dir_name.split("_")
        for part in parts:
            if part.endswith("s"):
                try:
                    return float(part[:-1])
                except ValueError:
                    pass
    return None


def find_scene_directories(results_dir: Path, prefix: str = "scene_") -> List[Path]:
    """
    Find all scene directories in results directory.

    Args:
        results_dir: Root results directory
        prefix: Prefix for scene directories (default: "scene_")

    Returns:
        List of scene directory paths, sorted by name
    """
    if not results_dir.exists():
        return []

    scene_dirs = [
        d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
    ]

    return sorted(scene_dirs, key=lambda p: p.name)


def count_regimes(regimes: List[str]) -> Dict[str, int]:
    """
    Count occurrences of each regime type.

    Args:
        regimes: List of regime strings

    Returns:
        Dictionary mapping regime names to counts
    """
    regime_counts = {}
    for regime in regimes:
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    return regime_counts


def extract_metrics_from_json(
    results_file: Path, method: str, method_key_map: Optional[Dict[str, str]] = None
) -> Optional[Dict]:
    """
    Extract metrics for a specific method from results.json.

    Args:
        results_file: Path to results.json
        method: Method name (e.g., 'pg_x0', 'gaussian_x0')
        method_key_map: Optional mapping from method names to JSON keys

    Returns:
        Dictionary of metrics or None if not found
    """
    if not results_file.exists():
        return None

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        comprehensive_metrics = data.get("comprehensive_metrics", {})

        # Default method key mapping
        if method_key_map is None:
            method_key_map = {
                "noisy": "noisy",
                "clean": "clean",
                "exposure_scaled": "exposure_scaled",
                "gaussian_x0": "gaussian_x0",
                "pg_x0_single": "pg_x0",
                "pg_x0_cross": "pg_x0_cross",
                "gaussian_x0_cross": "gaussian_x0_cross",
            }

        if method in method_key_map:
            key = method_key_map[method]
            return comprehensive_metrics.get(key)

    except Exception as e:
        logger.warning(f"Error extracting metrics for {method}: {e}")

    return None


def extract_pixel_range(tile_path: Path) -> Optional[Dict]:
    """Extract pixel range from results.json."""
    results_file = tile_path / "results.json"
    if not results_file.exists():
        return None

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        brightness = data.get("brightness_analysis", {})
        return {
            "min": brightness.get("min", 0),
            "max": brightness.get("max", 1),
            "mean": brightness.get("mean", 0),
            "std": brightness.get("std", 0),
        }
    except Exception:
        return None


def format_pixel_range(pixel_range: Dict) -> str:
    """Format pixel range for display."""
    return f"[{pixel_range['min']:.0f}, {pixel_range['max']:.0f}]"


def format_metrics(metrics_dict: Dict, methods_to_show: List[str]) -> str:
    """Format metrics string for display."""
    lines = []
    for method in methods_to_show:
        if method in metrics_dict:
            m = metrics_dict[method]
            line = f"{method.replace('_', '-').replace('x0', 'x0')}: "
            line += f"PSNR={m.get('psnr', 0):.1f}, SSIM={m.get('ssim', 0):.3f}, "
            line += f"LPIPS={m.get('lpips', 0):.3f}, NIQE={m.get('niqe', 'N/A')}"
            lines.append(line)
    return "\n".join(lines)


def format_metric_value(value, metric_name: str = "") -> str:
    """
    Format a single metric value for display.

    Args:
        value: Metric value
        metric_name: Name of metric (for special formatting)

    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"

    if isinstance(value, str):
        return value

    if metric_name == "psnr":
        return f"{value:.1f}"
    elif metric_name in ["ssim", "lpips"]:
        return f"{value:.3f}"
    elif metric_name == "niqe":
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        return str(value)
    else:
        return f"{value:.3f}"


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import sys

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    edm_path = project_root / "external" / "edm"
    if str(edm_path) not in sys.path:
        sys.path.insert(0, str(edm_path))

    from external.edm.torch_utils import distributed as dist

    if config_path is None:
        raise ValueError("Config path is required")

    config_path = Path(config_path)
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    if not HAS_YAML:
        raise ImportError("yaml module is required for load_yaml_config")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dist.print0(f"Loaded configuration from: {config_path}")
    return config


def find_mat_file_pairs(data_root: Path, include_metadata: bool = False):
    """
    Find pairs of noisy and ground truth .mat files in SIDD dataset.

    Args:
        data_root: Root directory of SIDD dataset
        include_metadata: Whether to include metadata files

    Returns:
        List of dictionaries with 'gt_path', 'noisy_path', 'metadata_path', 'suffix'
    """
    data_root = Path(data_root)
    gt_dict = {}
    noisy_dict = {}
    metadata_file = None

    # Find all GT and NOISY files
    for file_path in data_root.rglob("*.MAT"):
        filename = file_path.name.upper()
        if filename.startswith("GT_RAW_"):
            suffix = filename.replace("GT_RAW_", "").replace(".MAT", "")
            gt_dict[suffix] = file_path
        elif filename.startswith("NOISY_RAW_"):
            suffix = filename.replace("NOISY_RAW_", "").replace(".MAT", "")
            noisy_dict[suffix] = file_path

    # Find metadata file if requested
    if include_metadata:
        metadata_files = list(data_root.rglob("*metadata*.MAT"))
        if metadata_files:
            metadata_file = metadata_files[0]

    # Match pairs
    pairs = []
    for suffix in gt_dict.keys():
        if suffix in noisy_dict:
            pairs.append(
                {
                    "gt_path": gt_dict[suffix],
                    "noisy_path": noisy_dict[suffix],
                    "metadata_path": metadata_file,
                    "suffix": suffix,
                }
            )

    return pairs


def _ensure_dir(path: Path) -> None:
    """Internal function to ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def load_test_scenes_from_split_file(split_file: Path) -> Set[str]:
    """Load test scene IDs from split file.

    Args:
        split_file: Path to split file (e.g., Sony_test_list.txt)

    Returns:
        Set of scene IDs (padded to 5 digits)
    """
    test_scenes = set()

    if not split_file.exists():
        logger.warning(f"Split file not found: {split_file}")
        return test_scenes

    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("/")
            if len(parts) >= 3:
                filename = parts[-1]
                scene_id = filename.split("_")[0]
                test_scenes.add(scene_id.zfill(5))

    logger.info(f"Loaded {len(test_scenes)} test scenes from {split_file.name}")
    return test_scenes


def save_tile_as_uint8_png(tile: np.ndarray, output_path: Path) -> None:
    """Save tile as uint8 PNG.

    Args:
        tile: Image array as uint8 numpy array
        output_path: Path to save the PNG file
    """
    _ensure_dir(output_path.parent)
    Image.fromarray(tile).save(output_path)


def save_tile_as_float32_png(tile: np.ndarray, output_path: Path) -> None:
    """Save tile as float32 PNG (8-bit PNG + .npy file).

    Args:
        tile: Image array as uint8 numpy array
        output_path: Path to save the PNG file (will also create .npy file)
    """
    tile_float32 = tile.astype(np.float32) / 255.0
    _ensure_dir(output_path.parent)
    image_uint8 = (tile_float32 * 255.0).astype(np.uint8)
    Image.fromarray(image_uint8, mode="RGB").save(output_path)
    np.save(output_path.with_suffix(".npy"), tile_float32)


def convert_uint8_png_to_float32_png(
    uint8_png_path: Path, float32_png_path: Path
) -> bool:
    """Convert a uint8 PNG file to float32 PNG format.

    Args:
        uint8_png_path: Path to input uint8 PNG file
        float32_png_path: Path to output float32 PNG file

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        img = Image.open(uint8_png_path)
        tile_uint8 = np.array(img).astype(np.uint8)
        save_tile_as_float32_png(tile_uint8, float32_png_path)
        return True
    except Exception as e:
        logger.error(f"Failed to convert {uint8_png_path} to float32_png: {e}")
        return False
