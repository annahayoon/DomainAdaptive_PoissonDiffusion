#!/usr/bin/env python3
"""
Sensor Noise Calibration Module

Implements Poisson-Gaussian noise parameter estimation from processed sensor data
normalized to [-1, 1] range using short/long exposure pairs.

See README.md in this directory for detailed mathematical documentation.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.io import loadmat

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Try to import h5py for MATLAB v7.3 (HDF5) support
try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logger.warning("h5py not available. Cannot load MATLAB v7.3 (HDF5) .mat files.")

from config.sample_config import (
    DATA_RANGE_MAX,
    DATA_RANGE_MIN,
    DEFAULT_NUM_BINS,
    DEFAULT_NUM_SAMPLES,
    MIN_SAMPLES_PER_BIN,
    MIN_VALID_BINS,
)
from core.utils.data_utils import (
    _create_tile_lookup,
    load_long_image,
    load_metadata_json,
    load_short_image,
    load_test_tiles,
)
from core.utils.file_utils import load_mat_file, save_json_file
from core.utils.metadata_utils import _load_json_file as _load_json_file_utils


def _validate_image_pairs(
    short_images: List[Union[np.ndarray, torch.Tensor]],
    long_images: List[Union[np.ndarray, torch.Tensor]],
) -> None:
    """Validate that image pairs have matching shapes and counts."""
    if len(short_images) != len(long_images):
        raise ValueError(
            f"Mismatch in number of images: {len(short_images)} short vs {len(long_images)} long"
        )

    if len(short_images) == 0:
        raise ValueError("No image pairs provided")

    for short_img, long_img in zip(short_images, long_images):
        if short_img.shape != long_img.shape:
            raise ValueError(
                f"Shape mismatch: short {short_img.shape} vs long {long_img.shape}"
            )


def _load_image_pair(
    tile_id: str,
    short_dir: Path,
    long_dir: Path,
    tile_lookup: Optional[Dict],
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load a single short/long image pair."""
    try:
        short_tensor, _ = load_short_image(tile_id, short_dir, device)
        if tile_lookup is None:
            return None

        long_tensor, _ = load_long_image(tile_id, long_dir, tile_lookup, device)
        if long_tensor is None:
            return None

        if long_tensor.ndim == 4:
            long_tensor = long_tensor.squeeze(0)
        return short_tensor, long_tensor
    except Exception as e:
        logger.warning(f"Failed to load pair for tile {tile_id}: {e}")
        return None


def _normalize_tensors_to_device(
    short_tensors: List[torch.Tensor],
    long_tensors: List[torch.Tensor],
    device: Optional[torch.device] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Normalize tensor lists to a common device."""
    device = _get_device(device)

    short_images = []
    long_images = []
    for short_t, long_t in zip(short_tensors, long_tensors):
        if isinstance(short_t, torch.Tensor):
            short_images.append(short_t.detach().to(device))
            long_images.append(long_t.detach().to(device))
        else:
            short_images.append(torch.as_tensor(short_t, device=device))
            long_images.append(torch.as_tensor(long_t, device=device))
    return short_images, long_images


def _collect_residuals_and_means(
    short_images: List[torch.Tensor], long_images: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect residuals (short - long) and corresponding signal means (long)."""
    residuals = []
    means = []

    for short_img, long_img in zip(short_images, long_images):
        residual = short_img - long_img
        residuals.append(residual.flatten())
        means.append(long_img.flatten())

    return torch.cat(residuals), torch.cat(means)


def _compute_residuals_from_pairs(
    image_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    pixels_per_pair: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute residuals and means from a list of image pairs.

    Args:
        image_pairs: List of (noisy/short, gt/long) tensor pairs
        pixels_per_pair: Optional limit on pixels to sample per pair

    Returns:
        (residuals, means) flattened tensors
    """
    all_residuals = []
    all_means = []

    for noisy_img, gt_img in image_pairs:
        residual = noisy_img - gt_img
        means_flat = gt_img.flatten()
        residuals_flat = residual.flatten()

        if pixels_per_pair is not None and len(means_flat) > pixels_per_pair:
            n_pixels = len(means_flat)
            idx = torch.randperm(n_pixels, device=means_flat.device)[:pixels_per_pair]
            all_residuals.append(residuals_flat[idx])
            all_means.append(means_flat[idx])
        else:
            all_residuals.append(residuals_flat)
            all_means.append(means_flat)

    return torch.cat(all_residuals), torch.cat(all_means)


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    """Get device, defaulting to CPU if None."""
    return device if device is not None else torch.device("cpu")


def _get_output_dir(output_dir: Optional[Path], data_root: Path) -> Path:
    """Get output directory, defaulting to data_root if None."""
    return output_dir if output_dir is not None else data_root


def _save_calibration_file(
    output_dir: Path,
    sensor_name: str,
    a: float,
    b: float,
    num_samples: int,
    num_bins: int,
    **extra_metadata: Any,
) -> Path:
    """
    Save calibration parameters to JSON file.

    Args:
        output_dir: Directory to save the file
        sensor_name: Name of the sensor
        a: Slope parameter
        b: Intercept parameter
        num_samples: Number of samples used
        num_bins: Number of bins used
        **extra_metadata: Additional metadata to include

    Returns:
        Path to the saved calibration file
    """
    calibration_file = output_dir / f"{sensor_name}_noise_calibration.json"
    calibration_data = {
        "sensor": sensor_name,
        "a": float(a),
        "b": float(b),
        "num_samples": num_samples,
        "num_bins": num_bins,
        **extra_metadata,
    }

    save_json_file(calibration_file, calibration_data)

    logger.info(
        f"Saved calibration for {sensor_name}: a={a:.6e}, b={b:.6e} "
        f"to {calibration_file}"
    )

    return calibration_file


def _compute_binned_variance(
    means: torch.Tensor,
    residuals: torch.Tensor,
    num_bins: int,
    data_min: float = DATA_RANGE_MIN,
    data_max: float = DATA_RANGE_MAX,
    min_samples_per_bin: int = MIN_SAMPLES_PER_BIN,
    min_valid_bins: int = MIN_VALID_BINS,
    use_adaptive_range: bool = True,
    use_quantile_binning: bool = False,
    _retry_attempt: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute binned variance for mean-variance relationship estimation."""
    if use_adaptive_range:
        actual_min = float(means.min())
        actual_max = float(means.max())
        data_range = actual_max - actual_min
        padding = data_range * 0.01
        data_min = actual_min - padding
        data_max = actual_max + padding
        if data_range < 0.1:
            logger.warning(
                f"Narrow data range ({data_range:.4f}). Consider quantile binning."
            )

    if use_quantile_binning:
        quantiles = torch.linspace(0, 1, num_bins + 1, device=means.device)
        bin_edges = torch.quantile(means, quantiles)
    else:
        bin_edges = torch.linspace(
            data_min, data_max, num_bins + 1, device=means.device
        )

    bin_indices = torch.bucketize(means, bin_edges[1:], right=True) - 1
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    bin_means = []
    bin_vars = []
    bin_counts = []
    for i in range(num_bins):
        mask = bin_indices == i
        bin_residuals = residuals[mask]
        bin_counts.append(len(bin_residuals))
        if len(bin_residuals) > min_samples_per_bin:
            bin_means.append(torch.mean(means[mask]))
            bin_vars.append(torch.var(bin_residuals, unbiased=True))

    valid_bins = sum(1 for c in bin_counts if c > min_samples_per_bin)
    max_bin_count = max(bin_counts) if bin_counts else 0
    concentration_ratio = max_bin_count / len(means) if len(means) > 0 else 0

    if (
        not use_quantile_binning
        and _retry_attempt == 0
        and concentration_ratio > 0.5
        and valid_bins < num_bins // 2
    ):
        logger.warning(
            f"Data heavily concentrated ({concentration_ratio:.1%}). Switching to quantile binning."
        )
        return _compute_binned_variance(
            means,
            residuals,
            num_bins,
            data_min,
            data_max,
            min_samples_per_bin,
            min_valid_bins,
            use_adaptive_range,
            use_quantile_binning=True,
            _retry_attempt=1,
        )

    required_bins = min(min_valid_bins, max(3, num_bins // 2))
    if len(bin_means) < required_bins:
        raise ValueError(
            f"Insufficient bins with data ({len(bin_means)}/{required_bins} required, "
            f"out of {num_bins} total bins). "
            f"Data concentration: {concentration_ratio:.1%} in largest bin. "
            f"Try using --use-quantile-binning, reducing num_bins, reducing min_samples_per_bin, "
            f"or reducing min_valid_bins."
        )
    return torch.stack(bin_means), torch.stack(bin_vars)


def _fit_linear_model(
    means: torch.Tensor, variances: torch.Tensor
) -> Tuple[float, float]:
    """Fit linear model variance = a * mean + b using least squares."""
    X = torch.stack([means, torch.ones_like(means)], dim=1)
    coeffs = torch.linalg.lstsq(X, variances.unsqueeze(1)).solution.squeeze()
    return float(coeffs[0]), float(coeffs[1])


def _transform_to_01_domain(
    bin_means: torch.Tensor, bin_vars: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform bin means and variances from [-1, 1] to [0, 1] domain."""
    bin_means_01 = (bin_means + 1.0) / 2.0
    bin_vars_01 = bin_vars / 4.0
    return bin_means_01, bin_vars_01


def _transform_params_to_neg1pos1(a_01: float, b_01: float) -> Tuple[float, float]:
    """Transform noise parameters from [0, 1] to [-1, 1] domain.

    Args:
        a_01: Slope parameter in [0, 1] domain
        b_01: Intercept parameter in [0, 1] domain

    Returns:
        (a_norm, b_norm) parameters in [-1, 1] domain
    """
    a_norm = 2.0 * a_01
    b_norm = 2.0 * a_01 + 4.0 * b_01
    return a_norm, b_norm


def _compute_noise_params_from_bins(
    bin_means: torch.Tensor,
    bin_vars: torch.Tensor,
    min_valid_bins: int = MIN_VALID_BINS,
) -> Tuple[float, float]:
    """
    Compute noise parameters (a, b) from binned data.

    Transforms bins to [0,1] domain, fits linear model, then transforms back to [-1,1].

    Args:
        bin_means: Mean values in bins (in [-1, 1] domain)
        bin_vars: Variance values in bins (in [-1, 1] domain)
        min_valid_bins: Minimum number of valid bins required

    Returns:
        (a_norm, b_norm) parameters in [-1, 1] domain
    """
    # Transform to [0,1] for physically valid Poisson modeling (mean >= 0)
    bin_means_01, bin_vars_01 = _transform_to_01_domain(bin_means, bin_vars)

    valid = bin_means_01 >= 0.0
    if valid.sum() < min_valid_bins:
        logger.warning("Not enough valid signal bins. Using all bins.")
        valid = torch.ones_like(bin_means_01, dtype=torch.bool)

    a_01, b_01 = _fit_linear_model(bin_means_01[valid], bin_vars_01[valid])

    # Transform parameters back to [-1,1]
    return _transform_params_to_neg1pos1(a_01, b_01)


def _detect_mat_variable_name(file_path: Path) -> Optional[str]:
    """Auto-detect variable name in .mat file."""
    if not file_path.exists():
        return None

    try:
        from scipy.io import loadmat

        mat_data = loadmat(str(file_path))
        keys = [k for k in mat_data.keys() if not k.startswith("__")]
        if len(keys) == 1:
            return keys[0]
        elif "x" in keys:
            return "x"
    except Exception:
        pass

    return None


def _transform_01_to_neg1pos1(data: np.ndarray) -> np.ndarray:
    """Transform data from [0, 1] to [-1, 1] domain."""
    return 2.0 * data - 1.0


def _detect_directory_structure(data_dir: Path) -> bool:
    """Auto-detect SIDD structure (scene subdirectories with NOISY_RAW/GT_RAW)."""
    if not data_dir.exists():
        return False

    subdirs_with_mat = 0
    for item in data_dir.iterdir():
        if item.is_dir():
            mat_files = list(item.glob("*.MAT")) + list(item.glob("*.mat"))
            if mat_files:
                subdirs_with_mat += 1
                has_noisy = any(
                    "NOISY_RAW" in f.name.upper() or "noisy_raw" in f.name.lower()
                    for f in mat_files
                )
                has_gt = any(
                    "GT_RAW" in f.name.upper() or "gt_raw" in f.name.lower()
                    for f in mat_files
                )
                if has_noisy and has_gt:
                    return True

    return subdirs_with_mat >= 2


def _detect_data_mode(data_root: Path) -> str:
    """Auto-detect data mode by checking for .mat vs .pt files."""
    if not data_root.exists():
        return "pt"

    mat_files = list(data_root.rglob("*.MAT")) + list(data_root.rglob("*.mat"))
    if mat_files:
        for mat_file in mat_files[:10]:
            if (
                "NOISY_RAW" in mat_file.name.upper()
                or "GT_RAW" in mat_file.name.upper()
            ):
                return "mat"
        return "mat"

    return "pt" if list(data_root.rglob("*.pt")) else "pt"


def _find_mat_file_pairs(
    noisy_dir: Path,
    gt_dir: Path,
    pattern: Optional[str] = None,
    search_subdirs: Optional[bool] = None,
) -> List[Tuple[Path, Path]]:
    """
    Find matching noisy/GT .mat file pairs from SIDD-style directories.

    Supports two directory structures:
    1. Flat structure: noisy_dir/NOISY_RAW_*.MAT and gt_dir/GT_RAW_*.MAT
    2. SIDD structure: Each scene in its own subdirectory with NOISY_RAW_*.MAT and GT_RAW_*.MAT

    SIDD format: {scene_id}_NOISY_RAW_{id}.MAT and {scene_id}_GT_RAW_{id}.MAT
    Or: scene_dir/NOISY_RAW_010.MAT and scene_dir/GT_RAW_010.MAT

    Args:
        noisy_dir: Directory containing noisy .mat files (or root directory with scene subdirs)
        gt_dir: Directory containing ground truth .mat files (or root directory with scene subdirs)
        pattern: Optional regex pattern to match file names (default: matches NOISY_RAW/GT_RAW pattern)
        search_subdirs: If True, search in subdirectories (for SIDD structure).
                       If None, auto-detect based on directory structure.

    Returns:
        List of (noisy_path, gt_path) tuples
    """
    if search_subdirs is None:
        search_subdirs = _detect_directory_structure(noisy_dir)

    pairs = []

    if search_subdirs:
        scene_dirs = {d.name for d in noisy_dir.iterdir() if d.is_dir()}
        for scene_name in scene_dirs:
            scene_noisy_dir = noisy_dir / scene_name
            scene_gt_dir = (
                gt_dir / scene_name if gt_dir != noisy_dir else scene_noisy_dir
            )

            noisy_files = (
                list(scene_noisy_dir.glob("NOISY_RAW_*.MAT"))
                + list(scene_noisy_dir.glob("NOISY_RAW_*.mat"))
                + list(scene_noisy_dir.glob("noisy_raw_*.MAT"))
                + list(scene_noisy_dir.glob("noisy_raw_*.mat"))
            )

            gt_files = (
                list(scene_gt_dir.glob("GT_RAW_*.MAT"))
                + list(scene_gt_dir.glob("GT_RAW_*.mat"))
                + list(scene_gt_dir.glob("gt_raw_*.MAT"))
                + list(scene_gt_dir.glob("gt_raw_*.mat"))
            )

            for noisy_file in noisy_files:
                noisy_suffix = re.search(r"_RAW_(.+)$", noisy_file.name, re.IGNORECASE)
                if noisy_suffix:
                    suffix = noisy_suffix.group(1)
                    for gt_file in gt_files:
                        if gt_file.name.endswith(suffix):
                            pairs.append((noisy_file, gt_file))
                            break
    else:
        if pattern is None:
            pattern = r"(.+)_NOISY_RAW_(.+)\.MAT$"

        noisy_files = {}
        for mat_file in list(noisy_dir.glob("*.MAT")) + list(noisy_dir.glob("*.mat")):
            match = re.match(pattern, mat_file.name, re.IGNORECASE)
            if match:
                ext = mat_file.suffix
                gt_filename = f"{match.group(1)}_GT_RAW_{match.group(2)}{ext}"
                noisy_files[gt_filename] = mat_file
                noisy_files[gt_filename.lower()] = mat_file

        for gt_file in list(gt_dir.glob("*.MAT")) + list(gt_dir.glob("*.mat")):
            if gt_file.name in noisy_files or gt_file.name.lower() in noisy_files:
                noisy_path = noisy_files.get(gt_file.name) or noisy_files.get(
                    gt_file.name.lower()
                )
                if noisy_path:
                    pairs.append((noisy_path, gt_file))
    return pairs


def _load_mat_image_pair(
    noisy_path: Path,
    gt_path: Path,
    device: torch.device,
    variable_name: Optional[str] = None,
    transform_to_neg1pos1: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load noisy/GT image pair from .mat files."""
    if variable_name is None:
        variable_name = _detect_mat_variable_name(noisy_path) or "x"
    noisy_data = load_mat_file(noisy_path, variable_name)
    gt_data = load_mat_file(gt_path, variable_name)

    if noisy_data is None or gt_data is None or noisy_data.shape != gt_data.shape:
        return None

    if transform_to_neg1pos1:
        noisy_data = _transform_01_to_neg1pos1(noisy_data)
        gt_data = _transform_01_to_neg1pos1(gt_data)

    noisy_tensor = torch.as_tensor(noisy_data, dtype=torch.float32, device=device)
    gt_tensor = torch.as_tensor(gt_data, dtype=torch.float32, device=device)

    if noisy_tensor.ndim == 2:
        noisy_tensor = noisy_tensor.unsqueeze(0)
        gt_tensor = gt_tensor.unsqueeze(0)
    elif noisy_tensor.ndim == 3 and noisy_tensor.shape[2] <= 3:
        noisy_tensor = noisy_tensor.permute(2, 0, 1)
        gt_tensor = gt_tensor.permute(2, 0, 1)

    return noisy_tensor, gt_tensor


def _load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, returning None if file doesn't exist or parse fails.

    This is a wrapper around core.utils._load_json_file that handles errors gracefully.
    """
    if not file_path.exists():
        return None
    try:
        return _load_json_file_utils(file_path)
    except Exception:
        return None


def _build_tile_lookup_from_metadata(
    metadata: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Build tile lookup from metadata 'files' structure."""
    all_tiles = []
    for file_data in metadata.get("files", []):
        all_tiles.extend(file_data.get("tiles", []))
    return _create_tile_lookup(all_tiles)


def estimate_noise_params(
    short_images: List[torch.Tensor],
    long_images: List[torch.Tensor],
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_bins: int = DEFAULT_NUM_BINS,
    return_bins: bool = False,
    device: Optional[torch.device] = None,
) -> Union[Tuple[float, float], Tuple[float, float, torch.Tensor, torch.Tensor]]:
    """
    Estimate noise parameters a, b from short/long exposure pairs.

    Fits Poisson-Gaussian model in [0,1] domain (for physical validity) then transforms
    parameters to [-1,1] domain for guidance.

    Args:
        short_images: List of torch tensors in [-1, 1] (short exposures)
        long_images: List of torch tensors in [-1, 1] (long exposures, clean reference)
        num_samples: Number of pixel samples for fitting (default: 50000)
        num_bins: Number of bins for mean-variance estimation (default: 50)
        return_bins: If True, return bin_means and bin_vars for diagnostics
        device: Optional device (default: CPU)

    Returns:
        If return_bins=False: (a_norm, b_norm) parameters in [-1, 1] domain
        If return_bins=True: (a_norm, b_norm, bin_means, bin_vars)
    """
    device = _get_device(device)
    short_images, long_images = _normalize_tensors_to_device(
        short_images, long_images, device
    )
    _validate_image_pairs(short_images, long_images)

    residuals, means = _collect_residuals_and_means(short_images, long_images)

    if len(residuals) > num_samples:
        idx = torch.randperm(len(residuals), device=residuals.device)[:num_samples]
        residuals = residuals[idx]
        means = means[idx]

    bin_means, bin_vars = _compute_binned_variance(
        means,
        residuals,
        num_bins,
        DATA_RANGE_MIN,
        DATA_RANGE_MAX,
        use_adaptive_range=True,  # Use adaptive range by default
    )

    # Compute noise parameters from binned data
    a_norm, b_norm = _compute_noise_params_from_bins(bin_means, bin_vars)

    if return_bins:
        return a_norm, b_norm, bin_means, bin_vars
    return a_norm, b_norm


def estimate_noise_from_processed_data(
    short_dir: Optional[Path] = None,
    long_dir: Optional[Path] = None,
    metadata_json: Optional[Path] = None,
    tile_ids: Optional[List[str]] = None,
    split: str = "val",
    short_tensors: Optional[List[torch.Tensor]] = None,
    long_tensors: Optional[List[torch.Tensor]] = None,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_bins: int = DEFAULT_NUM_BINS,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Estimate noise parameters from processed [-1, 1] data.

    Can work with either .pt files from directories or preloaded tensor lists.

    Args:
        short_dir: Directory with short exposure .pt files (if not using tensors)
        long_dir: Directory with long exposure .pt files (if not using tensors)
        metadata_json: Optional metadata JSON for finding pairs
        tile_ids: Optional list of tile IDs. If None and metadata_json provided,
                 uses all tiles from specified split.
        split: Data split when loading from metadata (default: "val")
        short_tensors: Optional preloaded short exposure tensors
        long_tensors: Optional preloaded long exposure tensors
        num_samples: Number of samples for fitting (default: 50000)
        num_bins: Number of bins for estimation (default: 50)
        device: Device to load tensors on (default: CPU)

    Returns:
        (a, b) noise parameters in [-1, 1] domain
    """
    if short_tensors is not None and long_tensors is not None:
        return estimate_noise_params(
            short_tensors, long_tensors, num_samples, num_bins, device=device
        )

    if short_dir is None or long_dir is None:
        raise ValueError(
            "Must provide either (short_dir, long_dir) or (short_tensors, long_tensors)"
        )

    device = _get_device(device)

    tile_lookup = None
    if metadata_json is not None and metadata_json.exists():
        tile_lookup = load_metadata_json(metadata_json)

    if tile_ids is None:
        if tile_lookup is not None:
            split_tiles = load_test_tiles(metadata_json, split=split)
            tile_ids = [
                tile.get("tile_id") for tile in split_tiles if tile.get("tile_id")
            ]
        else:
            raise ValueError(
                "Either tile_ids or metadata_json must be provided to find image pairs"
            )

    short_tensors = []
    long_tensors = []
    for tile_id in tile_ids:
        pair = _load_image_pair(tile_id, short_dir, long_dir, tile_lookup, device)
        if pair is not None:
            short_tensor, long_tensor = pair
            short_tensors.append(short_tensor)
            long_tensors.append(long_tensor)

    if len(short_tensors) == 0:
        raise ValueError("No valid image pairs could be loaded")

    return estimate_noise_params(
        short_tensors, long_tensors, num_samples, num_bins, device=device
    )


def load_tile_ids_from_split_file(
    split_file: Path,
    sensor: str,
    metadata_json: Path,
    data_type: str = "short",
) -> List[str]:
    """
    Load tile IDs from a split file (e.g., Fuji_val_list.txt).

    Parses split files that contain paths like:
    ./Fuji/short/20015_00_0.1s.RAF ./Fuji/long/20015_00_10s.RAF ISO1000 F2.8

    Maps these to tile IDs in the metadata by matching file paths.

    Args:
        split_file: Path to split file (e.g., dataset/splits/Fuji_val_list.txt)
        sensor: Sensor name (e.g., "fuji", "sony")
        metadata_json: Path to metadata JSON file
        data_type: Type of tiles to load ("short" or "long", default: "short")

    Returns:
        List of tile IDs matching the split file entries
    """
    if not split_file.exists():
        logger.warning(f"Split file not found: {split_file}")
        return []

    # Load full metadata (not just tile lookup) to access file_metadata
    metadata = _load_json_file(metadata_json)
    if metadata is None:
        logger.warning(f"Could not load metadata from {metadata_json}")
        return []

    files = metadata.get("files", [])

    # Build tile lookup from files structure (tiles are nested under files, not top-level)
    tile_lookup = _build_tile_lookup_from_metadata(metadata)

    # Parse split file to extract filenames
    split_filenames = set()
    with open(split_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            # Parse format: ./Sensor/short/filename.ext ./Sensor/long/filename.ext ISO F
            short_path = parts[0]

            # Extract filename (e.g., "20015_00_0.1s.RAF")
            short_filename = Path(short_path).name
            split_filenames.add(short_filename)

    # Match files by filename in file_metadata
    matching_file_paths = set()
    for file_data in files:
        file_meta = file_data.get("file_metadata", {})
        file_path = file_meta.get("file_path", "")
        if file_path:
            file_name = Path(file_path).name
            if file_name in split_filenames:
                matching_file_paths.add(file_path)

    # Collect all tile IDs from matching files
    matching_tile_ids = []
    for file_data in files:
        file_meta = file_data.get("file_metadata", {})
        file_path = file_meta.get("file_path", "")

        if file_path not in matching_file_paths:
            continue

        tiles = file_data.get("tiles", [])
        for tile in tiles:
            tile_id = tile.get("tile_id")
            tile_data_type = tile.get("data_type", "")

            if tile_id and tile_data_type == data_type:
                matching_tile_ids.append(tile_id)

    return matching_tile_ids


def calibrate_all_sensors(
    data_root: Path = Path("dataset/processed"),
    splits_dir: Optional[Path] = None,
    split: str = "val",
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_bins: int = DEFAULT_NUM_BINS,
    device: Optional[torch.device] = None,
    output_dir: Optional[Path] = None,
    min_samples_per_bin: int = MIN_SAMPLES_PER_BIN,
    min_valid_bins: int = MIN_VALID_BINS,
    use_adaptive_range: bool = True,
    use_quantile_binning: bool = False,
    max_tiles: Optional[int] = None,
    batch_size: int = 500,
) -> Dict[str, Dict[str, float]]:
    """
    Calibrate noise parameters for all sensors found in metadata files.

    Loads all validation pairs for each sensor using split files (e.g., Fuji_val_list.txt),
    estimates noise parameters, and saves results to {sensor}_noise_calibration.json files.

    Args:
        data_root: Root directory containing processed data (default: dataset/processed)
        splits_dir: Directory containing split files (default: dataset/splits)
        split: Data split to use for calibration (default: "val")
        num_samples: Number of samples for fitting (default: 50000)
        num_bins: Number of bins for estimation (default: 50)
        device: Device to load tensors on (default: CPU)
        output_dir: Directory to save calibration files (default: data_root)
        min_samples_per_bin: Minimum samples required per bin (default: 20)
        min_valid_bins: Minimum number of valid bins required (default: 10)
        use_adaptive_range: Use actual data min/max for binning instead of [-1,1] (default: True)
        use_quantile_binning: Use quantile-based binning for even distribution (default: False)
        max_tiles: Maximum number of tiles to process (None = process all tiles) (default: None)
        batch_size: Number of tiles to process in each batch (default: 500)

    Returns:
        Dictionary mapping sensor names to calibration parameters {"a": float, "b": float}
    """
    device = _get_device(device)
    output_dir = _get_output_dir(output_dir, data_root)

    if splits_dir is None:
        splits_dir = data_root.parent / "splits"

    # Find all metadata_{sensor}_incremental.json files
    metadata_pattern = re.compile(r"metadata_(\w+)_incremental\.json$")
    metadata_files = list(data_root.glob("metadata_*_incremental.json"))

    if not metadata_files:
        raise ValueError(f"No metadata files found in {data_root}")

    all_results = {}

    for metadata_file in sorted(metadata_files):
        match = metadata_pattern.match(metadata_file.name)
        if not match:
            logger.warning(
                f"Skipping {metadata_file.name} - doesn't match expected pattern"
            )
            continue

        sensor = match.group(1)
        logger.info(f"Processing sensor: {sensor}")

        try:
            # Determine tile directories
            short_dir = data_root / "pt_tiles" / sensor / "short"
            long_dir = data_root / "pt_tiles" / sensor / "long"

            if not short_dir.exists() or not long_dir.exists():
                logger.warning(
                    f"Skipping {sensor}: directories not found "
                    f"({short_dir} or {long_dir})"
                )
                continue

            # Load tiles from split file (e.g., Fuji_val_list.txt)
            sensor_capitalized = sensor.capitalize()
            split_file = splits_dir / f"{sensor_capitalized}_{split}_list.txt"

            if not split_file.exists():
                logger.warning(
                    f"Split file not found: {split_file}. "
                    f"Looking for {sensor_capitalized}_{split}_list.txt in {splits_dir}"
                )
                continue

            tile_ids = load_tile_ids_from_split_file(
                split_file, sensor, metadata_file, data_type="short"
            )

            if not tile_ids:
                logger.warning(
                    f"No {split} tiles found for sensor {sensor} from {split_file.name}"
                )
                continue

            metadata = _load_json_file(metadata_file)
            if metadata is None:
                continue

            tile_lookup = _build_tile_lookup_from_metadata(metadata)

            long_tile_by_location = {}
            for tile_id, tile_meta in tile_lookup.items():
                if tile_meta.get("data_type") == "long":
                    key = (
                        tile_meta.get("scene_id"),
                        tile_meta.get("sensor_type"),
                        tile_meta.get("grid_x"),
                        tile_meta.get("grid_y"),
                    )
                    if all(k is not None for k in key):
                        long_tile_by_location[key] = tile_id

            valid_tile_ids = []
            for tile_id in tile_ids:
                tile_meta = tile_lookup.get(tile_id)
                if not tile_meta:
                    continue

                key = (
                    tile_meta.get("scene_id"),
                    tile_meta.get("sensor_type"),
                    tile_meta.get("grid_x"),
                    tile_meta.get("grid_y"),
                )

                if all(k is not None for k in key) and key in long_tile_by_location:
                    valid_tile_ids.append(tile_id)

            if not valid_tile_ids:
                continue

            max_tiles_to_process = (
                min(len(valid_tile_ids), max_tiles)
                if max_tiles
                else len(valid_tile_ids)
            )
            pixels_per_tile_total = 3 * 256 * 256

            if max_tiles_to_process > 1000:
                pixels_per_tile_sample = min(2000, int(pixels_per_tile_total * 0.1))
            elif max_tiles_to_process > 500:
                pixels_per_tile_sample = min(5000, int(pixels_per_tile_total * 0.2))
            else:
                samples_needed_per_bin = min_samples_per_bin * num_bins
                target_total_samples = max(
                    samples_needed_per_bin * 2, num_samples * 10, 50000
                )
                pixels_per_tile_sample = min(
                    target_total_samples // max_tiles_to_process,
                    int(pixels_per_tile_total * 0.5),
                )

            pixels_per_tile_sample = max(1000, pixels_per_tile_sample)

            all_residuals = []
            all_means = []
            tiles_processed = 0

            for batch_start in range(0, max_tiles_to_process, batch_size):
                batch_tile_ids = valid_tile_ids[batch_start : batch_start + batch_size]

                batch_short = []
                batch_long = []
                for tile_id in batch_tile_ids:
                    pair = _load_image_pair(
                        tile_id, short_dir, long_dir, tile_lookup, device
                    )
                    if pair is not None:
                        batch_short.append(pair[0])
                        batch_long.append(pair[1])

                if batch_short:
                    batch_pairs = list(zip(batch_short, batch_long))
                    batch_residuals, batch_means = _compute_residuals_from_pairs(
                        batch_pairs, pixels_per_pair=pixels_per_tile_sample
                    )
                    all_residuals.append(batch_residuals)
                    all_means.append(batch_means)
                    tiles_processed += len(batch_pairs)

            if not all_residuals:
                continue

            residuals = torch.cat(all_residuals)
            means = torch.cat(all_means)

            samples_needed = min_samples_per_bin * num_bins * 2
            max_samples_for_binning = max(samples_needed, num_samples * 5)

            if len(residuals) > max_samples_for_binning:
                idx = torch.randperm(len(residuals), device=residuals.device)[
                    :max_samples_for_binning
                ]
                residuals = residuals[idx]
                means = means[idx]

            bin_means, bin_vars = _compute_binned_variance(
                means,
                residuals,
                num_bins,
                DATA_RANGE_MIN,
                DATA_RANGE_MAX,
                min_samples_per_bin=min_samples_per_bin,
                min_valid_bins=min_valid_bins,
                use_adaptive_range=use_adaptive_range,
                use_quantile_binning=use_quantile_binning,
            )

            a, b = _compute_noise_params_from_bins(bin_means, bin_vars, min_valid_bins)
            all_results[sensor] = {"a": a, "b": b}

            _save_calibration_file(
                output_dir=output_dir,
                sensor_name=sensor,
                a=a,
                b=b,
                num_samples=num_samples,
                num_bins=num_bins,
                split=split,
                n_tiles=tiles_processed,
            )

        except Exception as e:
            logger.error(f"Failed to process sensor {sensor}: {e}", exc_info=True)
            continue

    return all_results


def estimate_noise_from_mat_files(
    noisy_dir: Path,
    gt_dir: Path,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_bins: int = DEFAULT_NUM_BINS,
    device: Optional[torch.device] = None,
    variable_name: Optional[str] = None,
    max_pairs: Optional[int] = None,
    batch_size: int = 100,
    search_subdirs: Optional[bool] = None,
) -> Tuple[float, float]:
    """
    Estimate noise parameters from .mat files with [0,1] normalized data.

    Loads matching noisy/GT pairs from .mat files, transforms to [-1,1] domain,
    and estimates Poisson-Gaussian noise parameters.

    Args:
        noisy_dir: Directory containing noisy .mat files (SIDD format: *_NOISY_RAW_*.MAT)
                   or root directory with scene subdirectories
        gt_dir: Directory containing ground truth .mat files (SIDD format: *_GT_RAW_*.MAT)
                or root directory with scene subdirectories (can be same as noisy_dir)
        num_samples: Number of pixel samples for fitting (default: 50000)
        num_bins: Number of bins for mean-variance estimation (default: 50)
        device: Device to load tensors on (default: CPU)
        variable_name: Variable name in .mat files (None = auto-detect, default: "x")
        max_pairs: Maximum number of image pairs to process (None = process all)
        batch_size: Number of pairs to process in each batch (default: 100)
        search_subdirs: If True, search in subdirectories (for SIDD structure where each scene has its own dir).
                       If None, auto-detect based on directory structure.

    Returns:
        (a, b) noise parameters in [-1, 1] domain
    """
    device = _get_device(device)

    # Find matching pairs
    pairs = _find_mat_file_pairs(noisy_dir, gt_dir, search_subdirs=search_subdirs)

    if len(pairs) == 0:
        raise ValueError(
            f"No matching .mat file pairs found in {noisy_dir} and {gt_dir}. "
            f"Expected format: *_NOISY_RAW_*.MAT and *_GT_RAW_*.MAT"
        )

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    logger.info(
        f"Processing {len(pairs)} .mat file pairs in batches of {batch_size} to avoid OOM"
    )

    # Use smaller batch size for loading to avoid OOM (50 pairs = ~2-3GB for large images)
    load_batch_size = min(batch_size, 50)

    all_residuals = []
    all_means = []
    pairs_processed = 0

    for batch_start in range(0, len(pairs), load_batch_size):
        batch_end = min(batch_start + load_batch_size, len(pairs))
        batch_num = (batch_start // load_batch_size) + 1
        total_batches = (len(pairs) + load_batch_size - 1) // load_batch_size
        logger.info(
            f"Processing batch {batch_num}/{total_batches}: pairs {batch_start+1}-{batch_end}/{len(pairs)}"
        )

        batch_pairs = pairs[batch_start:batch_end]
        batch_image_pairs = []

        # Load this batch
        for noisy_path, gt_path in batch_pairs:
            pair = _load_mat_image_pair(
                noisy_path, gt_path, device, variable_name, transform_to_neg1pos1=True
            )
            if pair is not None:
                batch_image_pairs.append(pair)
                pairs_processed += 1

        # Compute residuals and means for this batch
        if batch_image_pairs:
            batch_residuals, batch_means = _compute_residuals_from_pairs(
                batch_image_pairs
            )
            all_residuals.append(batch_residuals)
            all_means.append(batch_means)

            # Clear batch from memory immediately
            del batch_image_pairs, batch_residuals, batch_means

        # Optional: periodically concatenate and sample if memory is getting tight
        # This helps reduce memory usage by keeping only aggregated data
        if len(all_residuals) > 10:  # Every 10 batches (~500 pairs)
            logger.info(
                f"  Aggregating {len(all_residuals)} batches to reduce memory usage"
            )
            aggregated_residuals = torch.cat(all_residuals)
            aggregated_means = torch.cat(all_means)

            # Downsample if we have too many pixels already
            if len(aggregated_residuals) > num_samples * 2:
                idx = torch.randperm(
                    len(aggregated_residuals), device=aggregated_residuals.device
                )[: num_samples * 2]
                aggregated_residuals = aggregated_residuals[idx]
                aggregated_means = aggregated_means[idx]

            all_residuals = [aggregated_residuals]
            all_means = [aggregated_means]

    logger.info(f"Loaded {pairs_processed}/{len(pairs)} image pairs successfully")

    if not all_residuals:
        raise ValueError("No valid image pairs could be loaded from .mat files")

    # Final concatenation of all accumulated data
    residuals = torch.cat(all_residuals)
    means = torch.cat(all_means)

    logger.info(f"Total pixels collected: {len(residuals):,}")

    # Final sampling if needed
    if len(residuals) > num_samples:
        logger.info(
            f"Downsampling from {len(residuals):,} to {num_samples:,} pixels for fitting"
        )
        idx = torch.randperm(len(residuals), device=residuals.device)[:num_samples]
        residuals = residuals[idx]
        means = means[idx]

    bin_means, bin_vars = _compute_binned_variance(
        means,
        residuals,
        num_bins,
        DATA_RANGE_MIN,
        DATA_RANGE_MAX,
        use_adaptive_range=True,
    )

    return _compute_noise_params_from_bins(bin_means, bin_vars)


def calibrate_sensors_from_mat_files(
    data_root: Path,
    noisy_subdir: str = "noisy",
    gt_subdir: str = "gt",
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_bins: int = DEFAULT_NUM_BINS,
    device: Optional[torch.device] = None,
    output_dir: Optional[Path] = None,
    variable_name: Optional[str] = None,
    max_pairs: Optional[int] = None,
    batch_size: int = 100,
    search_subdirs: Optional[bool] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calibrate noise parameters for all sensors from .mat files.

    Searches for sensor subdirectories in data_root, each containing noisy/ and gt/ subdirectories
    with matching .mat files.

    Args:
        data_root: Root directory containing sensor subdirectories
                   OR directory with scene subdirectories (if search_subdirs=True)
        noisy_subdir: Subdirectory name for noisy images (default: "noisy")
                      Use "." for SIDD structure where files are in scene directories
        gt_subdir: Subdirectory name for GT images (default: "gt")
                   Use "." for SIDD structure where files are in scene directories
        num_samples: Number of samples for fitting (default: 50000)
        num_bins: Number of bins for estimation (default: 50)
        device: Device to load tensors on (default: CPU)
        output_dir: Directory to save calibration files (default: data_root)
        variable_name: Variable name in .mat files (None = auto-detect, default: "x")
        max_pairs: Maximum number of pairs per sensor (None = process all)
        batch_size: Number of pairs to process per batch (default: 100)
        search_subdirs: If True, search in subdirectories (for SIDD structure).
                       If None, auto-detect based on directory structure.

    Returns:
        Dictionary mapping sensor names to calibration parameters {"a": float, "b": float}
    """
    device = _get_device(device)
    output_dir = _get_output_dir(output_dir, data_root)

    # Auto-detect structure if not specified
    if search_subdirs is None:
        search_subdirs = _detect_directory_structure(data_root)

    all_results = {}

    if search_subdirs:
        noisy_dir = data_root
        gt_dir = data_root

        try:
            a, b = estimate_noise_from_mat_files(
                noisy_dir=noisy_dir,
                gt_dir=gt_dir,
                num_samples=num_samples,
                num_bins=num_bins,
                device=device,
                variable_name=variable_name,
                max_pairs=max_pairs,
                batch_size=batch_size,
                search_subdirs=True,
            )

            sensor_name = data_root.name if data_root.name else "sidd"
            all_results[sensor_name] = {"a": a, "b": b}

            _save_calibration_file(
                output_dir=output_dir,
                sensor_name=sensor_name,
                a=a,
                b=b,
                num_samples=num_samples,
                num_bins=num_bins,
                data_source="mat_files",
                n_pairs=len(
                    _find_mat_file_pairs(noisy_dir, gt_dir, search_subdirs=True)
                ),
            )

        except Exception as e:
            logger.error(f"Failed to process SIDD data: {e}", exc_info=True)
    else:
        sensor_dirs = [d for d in data_root.iterdir() if d.is_dir()]

        if not sensor_dirs:
            raise ValueError(f"No sensor directories found in {data_root}")

        for sensor_dir in sorted(sensor_dirs):
            sensor_name = sensor_dir.name
            logger.info(f"Processing sensor: {sensor_name}")

            noisy_dir = sensor_dir / noisy_subdir
            gt_dir = sensor_dir / gt_subdir

            if not noisy_dir.exists() or not gt_dir.exists():
                logger.warning(
                    f"Skipping {sensor_name}: directories not found "
                    f"({noisy_dir} or {gt_dir})"
                )
            continue

            try:
                a, b = estimate_noise_from_mat_files(
                    noisy_dir=noisy_dir,
                    gt_dir=gt_dir,
                    num_samples=num_samples,
                    num_bins=num_bins,
                    device=device,
                    variable_name=variable_name,
                    max_pairs=max_pairs,
                    batch_size=batch_size,
                    search_subdirs=False,
                )

                all_results[sensor_name] = {"a": a, "b": b}

                _save_calibration_file(
                    output_dir=output_dir,
                    sensor_name=sensor_name,
                    a=a,
                    b=b,
                    num_samples=num_samples,
                    num_bins=num_bins,
                    data_source="mat_files",
                    n_pairs=len(_find_mat_file_pairs(noisy_dir, gt_dir)),
                )

            except Exception as e:
                logger.error(
                    f"Failed to process sensor {sensor_name}: {e}", exc_info=True
                )
                continue

    return all_results


class SensorCalibration:
    """Sensor calibration utility for computing posterior sampling parameters."""

    # Cache for loaded calibration files
    _calibration_cache: Dict[str, Dict[str, float]] = {}

    @classmethod
    def _load_calibration_file(
        cls,
        sensor_name: str,
        calibration_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Load noise calibration parameters from JSON file.

        Args:
            sensor_name: Sensor name (e.g., "sony", "fuji", "sony_a7s_ii")
            calibration_dir: Directory containing calibration JSON files.
                            If None, searches in default locations.

        Returns:
            Dictionary with 'a' and 'b' parameters in [-1, 1] domain, or None if not found
        """
        # Check cache first
        cache_key = f"{sensor_name}:{calibration_dir}"
        if cache_key in cls._calibration_cache:
            return cls._calibration_cache[cache_key]

        # Search in provided directory
        if calibration_dir is not None and calibration_dir.exists():
            calibration_file = calibration_dir / f"{sensor_name}_noise_calibration.json"
            calib_data = _load_json_file(calibration_file)
            if calib_data is not None:
                result = {"a": float(calib_data["a"]), "b": float(calib_data["b"])}
                cls._calibration_cache[cache_key] = result
                return result

        for default_dir in [Path("dataset/processed"), Path(".")]:
            if default_dir.exists():
                calibration_file = default_dir / f"{sensor_name}_noise_calibration.json"
                calib_data = _load_json_file(calibration_file)
                if calib_data is not None:
                    result = {"a": float(calib_data["a"]), "b": float(calib_data["b"])}
                    cls._calibration_cache[cache_key] = result
                    return result

        sensor_mapping = {
            "sony_a7s_ii": "sony",
            "fuji_xt2": "fuji",
        }
        mapped_name = sensor_mapping.get(sensor_name.lower(), sensor_name.lower())

        if mapped_name != sensor_name.lower():
            return cls._load_calibration_file(mapped_name, calibration_dir)

        return None

    @classmethod
    def get_posterior_sampling_params(
        cls,
        sensor_name: str,
        mean_signal_physical: float,
        s: float,
        conservative_factor: float = 1.0,
        calibration_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Get calibrated sigma_max and sensor info for posterior sampling.

        Uses calibrated noise parameters (a, b) from JSON files generated by
        sensor_noise_calibrations.py to compute sigma_max based on the actual
        measured noise characteristics.

        Args:
            sensor_name: Name of the sensor (e.g., "sony", "fuji", "sony_a7s_ii")
            mean_signal_physical: Mean signal level in physical units (ADU)
            s: Sensor range parameter (white_level - black_level)
            conservative_factor: Multiplier for sigma_max (default: 1.0)
            calibration_dir: Directory containing calibration JSON files (optional)

        Returns:
            Dictionary with:
                - "sigma_max": Calibrated maximum noise level
                - "sensor_info": Sensor specifications dictionary
        """
        # Load calibration parameters from JSON file
        calib_params = cls._load_calibration_file(sensor_name, calibration_dir)

        if calib_params is None:
            raise FileNotFoundError(
                f"Calibration file not found for sensor '{sensor_name}'. "
                f"Searched in: {calibration_dir or 'default locations'}. "
                f"Please run sensor_noise_calibrations.py to generate calibration files."
            )

        a = calib_params["a"]
        b = calib_params["b"]

        mean_signal_normalized = max(
            -1.0, min(1.0, mean_signal_physical / s if s > 0 else 0.0)
        )
        variance_normalized = max(0.0, a * mean_signal_normalized + b)
        sigma_normalized = (
            (variance_normalized**0.5) if variance_normalized > 0 else 0.01
        )
        sigma_max = max(0.01, min(0.5, sigma_normalized * conservative_factor))

        sensor_info_full = {
            "sensor_name": sensor_name,
            "mean_signal_physical": mean_signal_physical,
            "mean_signal_normalized": mean_signal_normalized,
            "sensor_range": s,
            "calibration_params": {
                "a": a,
                "b": b,
            },
            "variance_normalized": variance_normalized,
            "sigma_normalized": sigma_normalized,
            "conservative_factor": conservative_factor,
            "calibration_source": "measured",
        }

        result = {
            "sigma_max": sigma_max,
            "sensor_info": sensor_info_full,
        }

        return result


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Calibrate noise parameters for all sensors using validation tiles or .mat files"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["pt", "mat", "auto"],
        help="Data mode: 'pt' for .pt tensor files, 'mat' for .mat files with [0,1] normalized data, "
        "'auto' to auto-detect (default: auto)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/processed"),
        help="Root directory containing processed data (default: dataset/processed)",
    )
    parser.add_argument(
        "--noisy-subdir",
        type=str,
        default="noisy",
        help="Subdirectory name for noisy images when using --mode mat (default: noisy). "
        "Ignored when --search-subdirs is used (SIDD structure: files are in same scene directory)",
    )
    parser.add_argument(
        "--gt-subdir",
        type=str,
        default="gt",
        help="Subdirectory name for GT images when using --mode mat (default: gt). "
        "Ignored when --search-subdirs is used (SIDD structure: files are in same scene directory)",
    )
    parser.add_argument(
        "--variable-name",
        type=str,
        default=None,
        help="Variable name in .mat files when using --mode mat (None = auto-detect, default: x)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of image pairs to process when using --mode mat (default: None = process all)",
    )
    parser.add_argument(
        "--search-subdirs",
        type=str,
        default="auto",
        choices=["true", "false", "auto"],
        help="Search in subdirectories when using --mode mat (true/false/auto). "
        "Auto-detect if 'auto' (default: auto)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        help="Directory containing split files (default: dataset/splits)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Data split to use for calibration (default: val)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of samples for fitting (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help=f"Number of bins for estimation (default: {DEFAULT_NUM_BINS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save calibration files (default: same as data-root)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for computation (default: cpu)",
    )
    parser.add_argument(
        "--min-samples-per-bin",
        type=int,
        default=MIN_SAMPLES_PER_BIN,
        help=f"Minimum samples required per bin (default: {MIN_SAMPLES_PER_BIN})",
    )
    parser.add_argument(
        "--min-valid-bins",
        type=int,
        default=MIN_VALID_BINS,
        help=f"Minimum number of valid bins required (default: {MIN_VALID_BINS})",
    )
    parser.add_argument(
        "--no-adaptive-range",
        action="store_true",
        help="Disable adaptive range binning (use theoretical [-1,1] range)",
    )
    parser.add_argument(
        "--use-quantile-binning",
        action="store_true",
        help="Use quantile-based binning for even distribution (default: False)",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Maximum number of tiles to process (default: None = process all tiles)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of tiles to process in each batch (default: 500)",
    )

    args = parser.parse_args()

    # Validate data root exists
    if not args.data_root.exists():
        logger.error(f"Data root directory does not exist: {args.data_root}")
        sys.exit(1)

    if args.mode is None or args.mode == "auto":
        args.mode = _detect_data_mode(args.data_root)

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU.")
        device = torch.device("cpu")

    try:
        if args.mode == "mat":
            search_subdirs_value = (
                None
                if args.search_subdirs == "auto"
                else (args.search_subdirs == "true")
            )

            results = calibrate_sensors_from_mat_files(
                data_root=args.data_root,
                noisy_subdir=args.noisy_subdir,
                gt_subdir=args.gt_subdir,
                num_samples=args.num_samples,
                num_bins=args.num_bins,
                device=device,
                output_dir=args.output_dir,
                variable_name=args.variable_name,
                max_pairs=args.max_pairs,
                batch_size=args.batch_size,
                search_subdirs=search_subdirs_value,
            )
        else:
            results = calibrate_all_sensors(
                data_root=args.data_root,
                splits_dir=args.splits_dir,
                split=args.split,
                num_samples=args.num_samples,
                num_bins=args.num_bins,
                device=device,
                output_dir=args.output_dir,
                min_samples_per_bin=args.min_samples_per_bin,
                min_valid_bins=args.min_valid_bins,
                use_adaptive_range=not args.no_adaptive_range,
                use_quantile_binning=args.use_quantile_binning,
                max_tiles=args.max_tiles,
                batch_size=args.batch_size,
            )

        logger.info("Calibration Summary")
        for sensor, params in results.items():
            logger.info(f"{sensor}: a={params['a']:.6e}, b={params['b']:.6e}")

    except Exception as e:
        logger.error(f"Calibration failed: {e}", exc_info=True)
        sys.exit(1)
