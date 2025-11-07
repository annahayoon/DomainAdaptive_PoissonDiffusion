"""
Sampling utilities for EDM-based image restoration.

This module provides utilities for running posterior sampling workflows,
including guidance module creation, restoration method orchestration,
sampling configuration, and batch processing utilities.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from config.sample_config import DEFAULT_BATCH_CHUNK_SIZE as DEFAULT_CHUNK_SIZE
from config.sample_config import DEFAULT_TILE_SIZE, MAX_CHUNK_SIZE
from core.guidance import GaussianGuidance, PoissonGaussianGuidance
from core.metrics import (
    compute_metrics_by_method,
    filter_metrics_for_json,
    get_nan_metrics_for_method,
)
from core.normalization import convert_range
from core.sampler import EDMSampler
from core.utils.data_utils import (
    analyze_image_brightness,
    apply_exposure_scaling,
    get_scene_exposure_frame_key,
)
from core.utils.sensor_utils import (
    convert_calibration_to_poisson_coeff,
    convert_calibration_to_sigma_r,
)
from core.utils.tiles_utils import save_scene_metrics_json

try:
    from visualization.visualizations import compute_scene_aggregate_metrics
    from visualization.visualizations import (
        create_scene_visualization_sample as create_scene_visualization,
    )
except ImportError:
    # Fallback if visualization module not available
    create_scene_visualization = None
    compute_scene_aggregate_metrics = None

logger = logging.getLogger(__name__)


def create_guidance_modules(
    args: Any,
    sensor_range: Dict[str, float],
    s_sensor: float,
    exposure_ratio: float,
    calibration_data: Optional[Dict[str, float]] = None,
    black_level: Optional[float] = None,
    white_level: Optional[float] = None,
    kappa: Optional[float] = None,
    tau: Optional[float] = None,
    pg_mode: str = "wls",
    guidance_level: str = "x0",
    sigma_r: Optional[float] = None,
) -> Tuple[PoissonGaussianGuidance, GaussianGuidance]:
    """
    Create Poisson-Gaussian and Gaussian guidance modules.

    Args:
        args: Argument namespace (for backward compatibility)
        sensor_range: Dictionary with 'min' and 'max' keys for sensor range
        s_sensor: Sensor scaling factor
        exposure_ratio: Ratio of long to short exposure
        calibration_data: Optional calibration data with 'a' and 'b' keys
        black_level: Black level (defaults to sensor_range['min'])
        white_level: White level (defaults to sensor_range['max'])
        kappa: Guidance strength multiplier (from args.kappa if args provided)
        tau: Guidance threshold (from args.tau if args provided)
        pg_mode: PG guidance mode ('wls' or 'full')
        guidance_level: Guidance level ('x0' or 'score')
        sigma_r: Read noise standard deviation (used if calibration_data is None)

    Returns:
        Tuple of (pg_guidance, gaussian_guidance)
    """
    black_level = black_level or sensor_range["min"]
    white_level = white_level or sensor_range["max"]

    # Extract parameters from args if provided, otherwise use direct parameters
    if hasattr(args, "kappa"):
        kappa = kappa or args.kappa
    if hasattr(args, "tau"):
        tau = tau or getattr(args, "tau", 0.0)
    if hasattr(args, "pg_mode"):
        pg_mode = pg_mode or args.pg_mode
    if hasattr(args, "guidance_level"):
        guidance_level = guidance_level or args.guidance_level
    if hasattr(args, "sigma_r"):
        sigma_r = sigma_r or args.sigma_r

    if calibration_data is not None:
        missing = [k for k in ["a", "b"] if k not in calibration_data]
        if missing:
            raise ValueError(
                f"Invalid calibration data: missing {missing}. "
                f"Got keys: {list(calibration_data.keys())}"
            )
        sigma_r_calib = convert_calibration_to_sigma_r(calibration_data["b"], s_sensor)
        poisson_coeff = convert_calibration_to_poisson_coeff(
            calibration_data["a"], s_sensor
        )
        sigma_r = sigma_r_calib
    else:
        if sigma_r is None:
            raise ValueError("Either calibration_data or sigma_r must be provided")
        poisson_coeff = None

    common_params = {
        "s": s_sensor,
        "sigma_r": sigma_r,
        "black_level": black_level,
        "white_level": white_level,
        "offset": 0.0,
        "exposure_ratio": exposure_ratio,
        "kappa": kappa or 0.1,
        "tau": tau or 0.0,
    }

    pg_guidance = PoissonGaussianGuidance(
        **common_params,
        mode=pg_mode,
        guidance_level=guidance_level,
        poisson_coeff=poisson_coeff,
    )

    gaussian_guidance = GaussianGuidance(**common_params)

    return pg_guidance, gaussian_guidance


def run_restoration_methods(
    sampler: EDMSampler,
    short_image: torch.Tensor,
    long_image: Optional[torch.Tensor],
    sigma_max: float,
    num_steps: int,
    pg_guidance: Optional[PoissonGaussianGuidance],
    gaussian_guidance: Optional[GaussianGuidance],
    y_e: torch.Tensor,
    exposure_ratio: float,
    run_methods: List[str],
    apply_exposure_scaling_func: Optional[callable] = None,
    class_labels: Optional[torch.Tensor] = None,
    no_heun: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Run restoration methods for a single image.

    Args:
        sampler: EDM sampler instance
        short_image: Short exposure image tensor
        long_image: Optional long exposure image tensor (for reference)
        sigma_max: Maximum noise level for sampling
        num_steps: Number of sampling steps
        pg_guidance: Optional Poisson-Gaussian guidance module
        gaussian_guidance: Optional Gaussian guidance module
        y_e: Observed image in [0,1] range
        exposure_ratio: Ratio of long to short exposure
        run_methods: List of method names to run
        apply_exposure_scaling_func: Optional function to apply exposure scaling
        class_labels: Optional class labels for conditional sampling
        no_heun: Whether to disable Heun's 2nd order correction

    Returns:
        Dictionary mapping method names to restored image tensors
    """
    restoration_results = {}
    run_methods_set = set(run_methods)

    if "exposure_scaled" in run_methods_set and apply_exposure_scaling_func:
        restoration_results["exposure_scaled"] = apply_exposure_scaling_func(
            short_image, exposure_ratio
        )

    # Update sampler config
    sampler.config.num_steps = num_steps
    sampler.config.sigma_max = sigma_max
    sampler.config.no_heun = no_heun

    for method, guidance, is_gaussian in [
        ("gaussian_x0", gaussian_guidance, True),
        ("pg_x0", pg_guidance, False),
    ]:
        if method in run_methods_set and guidance is not None:
            sampler.guidance = guidance
            result = sampler.sample(
                y_observed=y_e,
                condition=class_labels,
                exposure_ratio=exposure_ratio,
                apply_exposure_scaling=True,
            )
            restoration_results[method] = result["x_final"]

    if "short" in run_methods_set:
        restoration_results["short"] = short_image
    if "long" in run_methods_set and long_image is not None:
        restoration_results["long"] = long_image
    if "noisy" in run_methods_set:
        restoration_results["noisy"] = short_image
    if "gt" in run_methods_set and long_image is not None:
        restoration_results["gt"] = long_image

    return restoration_results


def compute_metrics_for_restorations(
    reference_image: torch.Tensor,
    restoration_results: Dict[str, torch.Tensor],
    device: torch.device,
    skip_methods: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for all restoration methods against a reference image.

    Args:
        reference_image: Reference image (e.g., ground truth or long exposure)
        restoration_results: Dictionary mapping method names to restored images
        device: Device for metric computation
        skip_methods: Optional list of method names to skip

    Returns:
        Dictionary mapping method names to metric dictionaries
    """
    if skip_methods is None:
        skip_methods = ["gt", "exposure_scaled"]

    metrics_results = {}

    for method, restored_tensor in restoration_results.items():
        if method in skip_methods or restored_tensor is None:
            continue

        try:
            metrics_results[method] = compute_metrics_by_method(
                reference_image, restored_tensor, method, device=device
            )
        except Exception as e:
            logger.warning(f"Metrics computation failed for {method}: {e}")
            metrics_results[method] = get_nan_metrics_for_method(method)

    return metrics_results


def prepare_observed_image(
    noisy_image: torch.Tensor,
    target_range: str = "[0,1]",
) -> torch.Tensor:
    """
    Prepare observed image for guidance (convert to [0,1] range).

    Args:
        noisy_image: Noisy image tensor (typically in [-1,1] range)
        target_range: Target range (default: "[0,1]")

    Returns:
        Observed image in target range, clamped to valid values
    """
    if target_range == "[0,1]":
        y_e = torch.clamp(convert_range(noisy_image, "[-1,1]", "[0,1]"), 0.0, 1.0)
    else:
        y_e = noisy_image

    return y_e


def configure_guidance_exposure(
    pg_guidance: Optional[PoissonGaussianGuidance],
    gaussian_guidance: Optional[GaussianGuidance],
    exposure_ratio: float,
    offset: float = 0.0,
) -> None:
    """
    Configure guidance modules with exposure ratio and offset.

    Args:
        pg_guidance: Optional Poisson-Gaussian guidance module
        gaussian_guidance: Optional Gaussian guidance module
        exposure_ratio: Ratio of long to short exposure
        offset: Optional offset value
    """
    for guidance_module in [pg_guidance, gaussian_guidance]:
        if guidance_module is not None:
            guidance_module.alpha = exposure_ratio
            guidance_module.offset = offset


def should_apply_guidance(
    step: int,
    guidance_start_step: int = 0,
    guidance_end_step: Optional[int] = None,
) -> bool:
    """Check if guidance should be applied at current step.

    Args:
        step: Current sampling step
        guidance_start_step: Step at which to start applying guidance
        guidance_end_step: Step at which to stop applying guidance (None = apply until end)

    Returns:
        True if guidance should be applied, False otherwise
    """
    if step < guidance_start_step:
        return False

    if guidance_end_step is not None and step >= guidance_end_step:
        return False

    return True


def create_edm_noise_schedule(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create EDM noise schedule.

    Args:
        num_steps: Number of sampling steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        rho: Schedule parameter (typically 7.0 for EDM)
        device: Device for the schedule tensor

    Returns:
        Noise schedule tensor [num_steps + 1]
    """
    t = torch.linspace(0, 1, num_steps + 1)
    sigma_min_rho = sigma_min ** (1 / rho)
    sigma_max_rho = sigma_max ** (1 / rho)
    sigma_t = (sigma_max_rho + t * (sigma_min_rho - sigma_max_rho)) ** rho

    if device is not None:
        sigma_t = sigma_t.to(device)

    return sigma_t


def prepare_conditioning(
    model: Any,
    condition: Optional[torch.Tensor],
    scale: Optional[Union[float, torch.Tensor]] = None,
    read_noise: Optional[Union[float, torch.Tensor]] = None,
    background: Optional[Union[float, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    conditioning_type: str = "dapgd",
) -> torch.Tensor:
    """Prepare conditioning vector from model or parameters.

    Args:
        model: Model with encode_conditioning method
        condition: Pre-computed conditioning vector (if provided, returned as-is)
        scale: Scale parameter
        read_noise: Read noise parameter
        background: Background parameter
        device: Target device
        conditioning_type: Type of conditioning ('dapgd' or 'l2')

    Returns:
        Conditioning vector [B, 3]

    Raises:
        ValueError: If condition is None and not all required parameters are provided
    """
    if condition is not None:
        return condition

    # Check required parameters
    if any(p is None for p in [scale, read_noise, background]):
        raise ValueError(
            "Must provide either 'condition' or all required parameters "
            "(scale, read_noise, background)"
        )

    return model.encode_conditioning(
        scale=scale,
        read_noise=read_noise,
        background=background,
        device=device,
        conditioning_type=conditioning_type,
    )


# ============================================================================
# BATCH PROCESSING UTILITIES
# ============================================================================


def _log_progress(message: str) -> None:
    """Log progress message."""
    logger.info(message)


def _free_gpu_memory() -> None:
    """Free GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _calculate_chunk_size(
    args: Any,
    num_images: int,
    default_size: int = DEFAULT_CHUNK_SIZE,
    max_size: int = MAX_CHUNK_SIZE,
) -> int:
    """Calculate chunk size for batch processing."""
    chunk_size = getattr(args, "batch_size", None) or default_size
    if chunk_size <= 0:
        chunk_size = default_size
    return max(1, min(chunk_size, max_size, num_images))


def _detach_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Detach and clone tensor to avoid keeping references to batch tensors."""
    return tensor.detach().clone()


def _ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has batch dimension (adds if ndim == 3)."""
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def _batch_cat_tensors(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate list of tensors, ensuring each has batch dimension."""
    return torch.cat([_ensure_batch_dim(img) for img in tensor_list], dim=0)


def _move_to_cpu_and_save(tensor: torch.Tensor, save_path: Path) -> torch.Tensor:
    """Move tensor to CPU and save to file."""
    tensor_cpu = tensor.cpu()
    torch.save(tensor_cpu, save_path)
    return tensor_cpu


def _build_tile_result_dict(
    tile_id: str,
    grid_x: int,
    grid_y: int,
    sigma_max: float,
    exposure_ratio: float,
    brightness_analysis: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    scene_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build tile result dictionary."""
    result = {
        "tile_id": tile_id,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "sigma_max_used": float(sigma_max),
        "exposure_ratio": float(exposure_ratio),
    }
    if brightness_analysis is not None:
        result["brightness_analysis"] = brightness_analysis
    if metrics is not None:
        result["metrics"] = metrics
        result["comprehensive_metrics"] = metrics
    if scene_id is not None:
        result["scene_id"] = scene_id
    return result


def _get_scene_directory(
    output_dir: Path,
    scene_id: str,
    sensor: str,
    frame_id: str,
    exposure_time: float,
) -> Path:
    """Generate scene directory name: scene_{scene_id}_{sensor}_{frame_id}_{exposure_time}s"""
    exposure_str = f"{exposure_time:.4f}".rstrip("0").rstrip(".")
    return output_dir / f"scene_{scene_id}_{sensor}_{frame_id}_{exposure_str}s"


def _run_guidance_sample(
    sampler: Any,
    short_batch: torch.Tensor,
    y_e_batch: torch.Tensor,
    sigma_max: float,
    exposure_ratio: float,
    guidance: Any,
    guidance_level: str,
    num_steps: int,
    no_heun: bool,
    is_gaussian: bool,
    class_labels: Any = None,
) -> torch.Tensor:
    """Run guidance sampling for a batch of images.

    This is a shared function used by both single-image and batch processing.

    Args:
        sampler: The EDM sampler instance
        short_batch: Short exposure images [B, C, H, W] or [C, H, W]
        y_e_batch: Physical measurements [B, C, H, W] or [C, H, W]
        sigma_max: Maximum noise level
        exposure_ratio: Exposure ratio
        guidance: Guidance module (PoissonGaussianGuidance or GaussianGuidance)
        guidance_level: Guidance level ('x0' or 'score')
        num_steps: Number of sampling steps
        no_heun: Whether to disable Heun's correction
        is_gaussian: Whether this is Gaussian guidance (vs PG guidance)
        class_labels: Optional class labels for conditional sampling

    Returns:
        Restored images [B, C, H, W] or [C, H, W]
    """
    guidance.guidance_level = guidance_level
    sampler.config.num_steps = num_steps
    sampler.config.sigma_max = sigma_max
    sampler.config.no_heun = no_heun
    sampler.guidance = guidance

    result = sampler.sample(
        y_observed=y_e_batch,
        condition=class_labels,
        exposure_ratio=exposure_ratio,
    )
    return result["x_final"]


def run_restoration_methods_batch(
    args: Any,
    sampler: Any,
    short_batch: torch.Tensor,
    y_e_batch: torch.Tensor,
    sigma_max: float,
    exposure_ratio: float,
    pg_guidance: Any,
    gaussian_guidance: Any,
    long_images: List[Optional[torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Run restoration methods for a batch of images."""
    restoration_results_batch = {}
    if hasattr(args, "run_methods") and args.run_methods:
        run_methods = set(args.run_methods)
    else:
        # Default methods for visualization and comparison (WLS PG and Gaussian)
        run_methods = {"short", "long", "pg_x0", "gaussian_x0"}

    with torch.no_grad():
        if "exposure_scaled" in run_methods:
            restoration_results_batch["exposure_scaled"] = apply_exposure_scaling(
                short_batch, exposure_ratio
            )
        # Process WLS PG mode
        if "pg_x0" in run_methods:
            restoration_results_batch["pg_x0"] = _run_guidance_sample(
                sampler,
                short_batch,
                y_e_batch,
                sigma_max,
                exposure_ratio,
                pg_guidance,
                "x0",
                args.num_steps,
                args.no_heun,
                is_gaussian=False,
                class_labels=None,
            )

        # Process Gaussian mode if available
        if "gaussian_x0" in run_methods and gaussian_guidance is not None:
            restoration_results_batch["gaussian_x0"] = _run_guidance_sample(
                sampler,
                short_batch,
                y_e_batch,
                sigma_max,
                exposure_ratio,
                gaussian_guidance,
                "x0",
                args.num_steps,
                args.no_heun,
                is_gaussian=True,
                class_labels=None,
            )

    if "short" in run_methods:
        restoration_results_batch["short"] = short_batch
    if "long" in run_methods and long_images and long_images[0] is not None:
        restoration_results_batch["long"] = torch.cat(
            [img for img in long_images if img is not None], dim=0
        )

    return restoration_results_batch


def process_batch_in_chunks(
    short_images: List[torch.Tensor],
    y_e_list: List[torch.Tensor],
    long_images: List[Optional[torch.Tensor]],
    args: Any,
    sampler: Any,
    sigma_max: float,
    exposure_ratio: float,
    pg_guidance_dict: Dict[str, Any],
    gaussian_guidance: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict[str, torch.Tensor]:
    """
    Process tiles in batch mode with single-exposure denoising.
    """
    chunk_size = min(chunk_size, len(short_images))
    num_chunks = (len(short_images) + chunk_size - 1) // chunk_size

    _log_progress(
        f"Processing {len(short_images)} tiles in {num_chunks} chunks of size {chunk_size}"
    )

    all_restoration_results = defaultdict(list)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(short_images))

        _log_progress(
            f"Processing chunk {chunk_idx + 1}/{num_chunks} (tiles {start_idx}-{end_idx-1})"
        )

        short_chunk = torch.cat(short_images[start_idx:end_idx], dim=0)
        y_e_chunk = torch.cat(y_e_list[start_idx:end_idx], dim=0)
        long_chunk = long_images[start_idx:end_idx]

        # Process WLS PG mode and Gaussian
        pg_guidance = pg_guidance_dict["wls"]
        chunk_results = run_restoration_methods_batch(
            args,
            sampler,
            short_chunk,
            y_e_chunk,
            sigma_max,
            exposure_ratio,
            pg_guidance,
            gaussian_guidance,
            long_chunk,
        )

        for method, batch_tensor in chunk_results.items():
            if batch_tensor is not None:
                all_restoration_results[method].append(batch_tensor)

        del short_chunk, y_e_chunk, long_chunk
        _free_gpu_memory()

    # Concatenate all results
    result_dict = {
        method: torch.cat(tensor_list, dim=0)
        for method, tensor_list in all_restoration_results.items()
        if tensor_list
    }

    return result_dict


def group_tiles_by_scene(
    selected_tiles: List[Dict[str, Any]],
    tile_lookup: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[str, str, float], List[Dict[str, Any]]]:
    """Group tiles by scene using (scene_id, frame_id, exposure_time) key."""
    scene_groups = defaultdict(list)
    for tile_info in selected_tiles:
        tile_id = tile_info["tile_id"]
        scene_key = get_scene_exposure_frame_key(tile_id, tile_lookup)
        if scene_key:
            scene_groups[scene_key].append(tile_info)
        else:
            logger.warning(
                f"Could not extract scene/exposure/frame key from {tile_id}, processing individually"
            )
            scene_groups[("unknown", "00", 0.0)].append(tile_info)
    return scene_groups


def extract_tile_results_from_batch(
    restoration_results_batch: Dict[str, torch.Tensor],
    tile_index: int,
) -> Dict[str, torch.Tensor]:
    """Extract a single tile's results from batch tensors.

    Returns tensors with shape [1, C, H, W] to preserve batch dimension for metrics computation.
    """
    results = {}
    for method, batch_tensor in restoration_results_batch.items():
        if batch_tensor is not None:
            # Extract single tile: [B, C, H, W] -> [1, C, H, W]
            tile_result = batch_tensor[tile_index : tile_index + 1]

            # Validate channel count - should be 3 for RGB images
            if tile_result.ndim == 4 and tile_result.shape[1] == 1:
                # If we somehow got 1 channel, try to expand it to 3 channels
                # This shouldn't happen, but handle it gracefully
                logger.warning(
                    f"Tile result for {method} has 1 channel, expanding to 3 channels. "
                    f"Shape: {tile_result.shape}, Expected: [1, 3, H, W]"
                )
                # Repeat the channel 3 times: [1, 1, H, W] -> [1, 3, H, W]
                tile_result = tile_result.repeat(1, 3, 1, 1)

            results[method] = tile_result
    return results


def _build_grid_to_metadata_map(
    tile_metadata_list: List[Dict[str, Any]],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Build mapping from grid coordinates to tile metadata."""
    return {
        (meta["grid_x"], meta["grid_y"]): meta
        for meta in tile_metadata_list
        if meta.get("grid_x") is not None and meta.get("grid_y") is not None
    }


def _validate_tile_positions(
    tiles_dict: Dict[Tuple[int, int], torch.Tensor],
    grid_to_meta: Dict[Tuple[int, int], Dict[str, Any]],
    sensor_type: Optional[str] = None,
) -> None:
    """Validate that all tiles have required position metadata."""
    missing_positions = []
    for gx, gy in tiles_dict.keys():
        if (gx, gy) not in grid_to_meta:
            missing_positions.append(f"grid({gx}, {gy}) - no metadata")
        elif (
            "image_x" not in grid_to_meta[(gx, gy)]
            or "image_y" not in grid_to_meta[(gx, gy)]
        ):
            missing_positions.append(f"grid({gx}, {gy}) - missing image_x/image_y")

    if missing_positions:
        sensor_info = f" for sensor '{sensor_type}'" if sensor_type else ""
        raise ValueError(
            f"Missing image positions for {len(missing_positions)} tiles{sensor_info}. "
            f"Comprehensive metadata with image_x/image_y is required for accurate stitching.\n"
            f"First few missing: {missing_positions[:5]}\n"
            f"Please use comprehensive_{sensor_type}_tiles_metadata.json from preprocessing pipeline."
        )


def _calculate_image_dimensions(
    grid_to_meta: Dict[Tuple[int, int], Dict[str, Any]],
    tile_size: int,
) -> Tuple[int, int]:
    """Calculate full image dimensions from tile metadata."""
    max_y = max(
        meta["image_y"] + tile_size
        for meta in grid_to_meta.values()
        if "image_y" in meta
    )
    max_x = max(
        meta["image_x"] + tile_size
        for meta in grid_to_meta.values()
        if "image_x" in meta
    )
    return max_y, max_x


def stitch_tiles_to_grid(
    tiles_dict: Dict[Tuple[int, int], torch.Tensor],
    tile_size: int = DEFAULT_TILE_SIZE,
    sensor_type: Optional[str] = None,
    tile_metadata_list: Optional[List[Dict[str, Any]]] = None,
) -> Optional[torch.Tensor]:
    """Stitch tiles into a full image grid."""
    if not tiles_dict:
        return None

    if not tile_metadata_list:
        raise ValueError(
            "tile_metadata_list is required for accurate stitching. "
            "Comprehensive metadata with image_x/image_y positions is required. "
            "Use comprehensive_{sensor}_tiles_metadata.json from preprocessing pipeline."
        )

    first_tile = next(iter(tiles_dict.values()))
    channels = first_tile.shape[1]

    grid_to_meta = _build_grid_to_metadata_map(tile_metadata_list)
    _validate_tile_positions(tiles_dict, grid_to_meta, sensor_type)

    max_y, max_x = _calculate_image_dimensions(grid_to_meta, tile_size)

    full_image = torch.zeros(
        (1, channels, max_y, max_x),
        dtype=first_tile.dtype,
        device=first_tile.device,
    )

    for (grid_x, grid_y), tile in tiles_dict.items():
        meta = grid_to_meta.get((grid_x, grid_y))
        if not meta or "image_x" not in meta or "image_y" not in meta:
            logger.error(
                f"Missing image position for tile at grid({grid_x}, {grid_y}) - skipping"
            )
            continue

        tile_tensor = tile.unsqueeze(0) if tile.ndim == 3 else tile
        full_image[
            :,
            :,
            meta["image_y"] : meta["image_y"] + tile_size,
            meta["image_x"] : meta["image_x"] + tile_size,
        ] = tile_tensor

    return full_image


def process_scene_batch(
    scene_tiles: List[Dict],
    scene_id: str,
    args: Any,
    sampler: Any,
    tile_lookup: Dict,
    output_dir: Path,
    scene_idx: int,
    load_tile_and_metadata_func: Any,
    setup_guidance_modules_func: Any,
    compute_all_metrics_func: Any,
) -> List[Dict[str, Any]]:
    """Process a scene's tiles in batch mode."""
    logger.info(f"Starting to process scene {scene_id} with {len(scene_tiles)} tiles")
    _log_progress(f"Loading {len(scene_tiles)} tiles for scene {scene_id}...")

    short_images = []
    long_images = []
    y_e_list = []
    tile_metadata_list = []

    for tile_idx, tile_info in enumerate(scene_tiles):
        if not tile_idx % 50:
            _log_progress(f"Loaded {tile_idx + 1}/{len(scene_tiles)} tiles...")
        tile_data = load_tile_and_metadata_func(
            tile_info["tile_id"], tile_info, tile_lookup, args, sampler
        )
        if tile_data is None:
            continue

        short_images.append(tile_data["short_image"])
        long_images.append(tile_data["long_image"])
        y_e_list.append(torch.from_numpy(tile_data["short_phys"]).to(sampler.device))
        tile_metadata_list.append(tile_data)

    if not short_images:
        return []

    first_meta = tile_metadata_list[0]
    sigma_max = first_meta["estimated_sigma"]
    exposure_ratio = first_meta["exposure_ratio"]

    # Store metadata in args
    args._first_meta = first_meta

    pg_guidance_dict, gaussian_guidance, _ = setup_guidance_modules_func(
        args, first_meta, exposure_ratio
    )

    chunk_size = _calculate_chunk_size(args, len(short_images))
    _log_progress(f"Processing {len(short_images)} tiles in chunks of {chunk_size}...")

    restoration_results_batch = process_batch_in_chunks(
        short_images,
        y_e_list,
        long_images,
        args,
        sampler,
        sigma_max,
        exposure_ratio,
        pg_guidance_dict,
        gaussian_guidance,
        chunk_size=chunk_size,
    )

    _free_gpu_memory()

    sensor = first_meta["extracted_sensor"]
    # Extract frame_id and exposure_time from first tile
    first_tile_id = scene_tiles[0]["tile_id"]
    scene_key = get_scene_exposure_frame_key(first_tile_id, tile_lookup)
    if scene_key:
        _, frame_id, exposure_time = scene_key
    else:
        frame_id = "00"
        exposure_time = first_meta.get("exposure_ratio", 0.0)

    scene_dir = _get_scene_directory(
        output_dir, scene_id, sensor, frame_id, exposure_time
    )
    scene_dir.mkdir(exist_ok=True)

    scene_metrics = []
    tile_results_dict = defaultdict(dict)

    _log_progress("Extracting results and computing metrics...")

    for i, meta in enumerate(tile_metadata_list):
        individual_results = extract_tile_results_from_batch(
            restoration_results_batch, i
        )
        for method, tile_tensor in individual_results.items():
            # Detach and clone to avoid keeping references to large batch tensors
            detached_tensor = _detach_tensor(tile_tensor)
            tile_results_dict[method][
                (meta["grid_x"], meta["grid_y"])
            ] = detached_tensor

        long_image = long_images[i]
        if long_image is None:
            continue

        metrics_results = compute_all_metrics_func(
            long_image,
            short_images[i],
            individual_results,
            y_e_list[i].cpu().numpy(),
            y_e_list[i],
            meta["s_sensor"],
            meta["extracted_sensor"],
            sampler,
        )

        filtered_metrics = filter_metrics_for_json(
            metrics_results, sensor_type=meta["extracted_sensor"]
        )
        scene_metrics.append(
            _build_tile_result_dict(
                tile_id=meta["tile_id"],
                grid_x=meta["grid_x"],
                grid_y=meta["grid_y"],
                sigma_max=sigma_max,
                exposure_ratio=exposure_ratio,
                brightness_analysis=analyze_image_brightness(short_images[i]),
                metrics=filtered_metrics,
            )
        )

    _log_progress("Stitching tiles into full scene images...")

    stitched_images = {}
    for method, tiles_dict in tile_results_dict.items():
        if tiles_dict:
            stitched = stitch_tiles_to_grid(
                tiles_dict,
                tile_size=DEFAULT_TILE_SIZE,
                sensor_type=sensor,
                tile_metadata_list=tile_metadata_list,
            )
            if stitched is not None:
                stitched_images[method] = _move_to_cpu_and_save(
                    stitched, scene_dir / f"stitched_{method}.pt"
                )

    # Debug: Check if PG and Gaussian produce identical results
    if "pg_x0" in stitched_images and "gaussian_x0" in stitched_images:
        pg_img = stitched_images["pg_x0"]
        gaussian_img = stitched_images["gaussian_x0"]
        if torch.allclose(pg_img, gaussian_img, atol=1e-6):
            logger.warning(
                f"WARNING: PG and Gaussian results are identical (within 1e-6 tolerance) "
                f"for scene {scene_id}. This may indicate a bug or poisson_coeff ≈ 0."
            )
        else:
            max_diff = (pg_img - gaussian_img).abs().max().item()
            mean_diff = (pg_img - gaussian_img).abs().mean().item()
            logger.info(
                f"PG vs Gaussian differences for scene {scene_id}: "
                f"max={max_diff:.6f}, mean={mean_diff:.6f}"
            )

    del tile_results_dict, restoration_results_batch
    _free_gpu_memory()

    if not args.skip_visualization and create_scene_visualization is not None:
        create_scene_visualization(
            stitched_images, scene_id, sensor, scene_dir, scene_metrics
        )

    if compute_scene_aggregate_metrics is not None:
        aggregate_metrics = compute_scene_aggregate_metrics(scene_metrics)
        save_scene_metrics_json(
            scene_dir, scene_id, sensor, scene_metrics, aggregate_metrics
        )

    return [
        _build_tile_result_dict(
            tile_id=meta["tile_id"],
            grid_x=meta["grid_x"],
            grid_y=meta["grid_y"],
            sigma_max=meta["estimated_sigma"],
            exposure_ratio=meta["exposure_ratio"],
            scene_id=scene_id,
        )
        for meta in tile_metadata_list
    ]


def process_tiles_in_batches(
    selected_tiles: List[Dict],
    args: Any,
    sampler: Any,
    tile_lookup: Dict,
    output_dir: Path,
    process_single_tile_func: Any,
    load_tile_and_metadata_func: Any,
    setup_guidance_modules_func: Any,
    compute_all_metrics_func: Any,
) -> List[Dict[str, Any]]:
    """Process tiles in batches, grouping by scene."""
    scene_groups = group_tiles_by_scene(selected_tiles, tile_lookup)
    all_results = []
    scene_idx = 0

    for scene_key, tiles in scene_groups.items():
        if scene_key[0] == "unknown":
            for tile_info in tiles:
                result = process_single_tile_func(
                    scene_idx, tile_info, args, sampler, tile_lookup, output_dir
                )
                if result is not None:
                    all_results.append(result)
                scene_idx += 1
        else:
            scene_id, frame_id, exposure_time = scene_key
            scene_results = process_scene_batch(
                tiles,
                scene_id,
                args,
                sampler,
                tile_lookup,
                output_dir,
                scene_idx,
                load_tile_and_metadata_func,
                setup_guidance_modules_func,
                compute_all_metrics_func,
            )
            all_results.extend(scene_results)
            scene_idx += 1

    return all_results
