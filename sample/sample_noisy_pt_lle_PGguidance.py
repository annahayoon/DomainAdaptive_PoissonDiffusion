#!/usr/bin/env python3
"""
Posterior Sampling for Image Restoration using EDM Model with Poisson-Gaussian Guidance

This script performs posterior sampling on short exposure test images using EDM with
exposure-aware Poisson-Gaussian measurement guidance for physics-informed restoration.

Theory: We sample from p(x|y) ∝ p(y|x) p(x) where p(x) is the EDM-learned prior
and p(y|x) is the Poisson-Gaussian likelihood.

Usage:
    python sample/sample_noisy_pt_lle_PGguidance.py \
        --model_path results/edm_pt_training_sony_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --short_dir dataset/processed/pt_tiles/sony/short \
        --output_dir results/low_light_enhancement \
        --num_examples 3 \
        --num_steps 18 \
        --sigma_r 3.0 \
        --kappa 0.1 \
        --use_sensor_calibration \
        --use_noise_calibration \
        --calibration_dir dataset/processed
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

import logging

import external.edm.dnnlib
from config.sample_config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONSERVATIVE_FACTOR,
    DEFAULT_EXPOSURE_RATIO_ERROR_THRESHOLD,
    DEFAULT_KAPPA,
    DEFAULT_NUM_STEPS,
    DEFAULT_RHO,
    DEFAULT_SENSOR_RANGES,
    DEFAULT_SIGMA_MIN,
    DEFAULT_SIGMA_R,
    DEFAULT_STRATIFIED_ALPHA,
    DEFAULT_TAU,
    SENSOR_NAME_MAPPING,
)
from core.dataset import TileDataset
from core.gradients import run_gradient_verification
from core.guidance import GaussianGuidance, PoissonGaussianGuidance
from core.metrics import (
    compute_metrics_by_method,
    filter_metrics_for_json,
    get_nan_metrics_for_method,
)
from core.normalization import convert_range
from core.sampler import EDMPosteriorSampler
from core.utils.data_utils import (
    analyze_image_brightness,
    apply_exposure_scaling,
    extract_scene_id_from_tile_id,
    find_long_tile_pair,
    get_exposure_ratio,
    get_exposure_time_from_metadata,
    get_scene_exposure_key,
    load_long_image,
    load_metadata_json,
    load_noise_calibration,
    load_short_image,
    load_tensor,
    load_test_tiles,
    load_tile_and_metadata,
    load_tile_image_data,
    select_tiles,
    select_tiles_for_processing,
)
from core.utils.file_utils import save_json_file
from core.utils.sampling_utils import (
    _calculate_chunk_size,
    _detach_tensor,
    _free_gpu_memory,
    _log_progress,
    _move_to_cpu_and_save,
    extract_tile_results_from_batch,
    group_tiles_by_scene,
    process_batch_in_chunks,
    process_scene_batch,
    process_tiles_in_batches,
    run_restoration_methods_batch,
    stitch_tiles_to_grid,
)
from core.utils.sensor_utils import (
    compute_sensor_range,
    convert_calibration_to_poisson_coeff,
    convert_calibration_to_sigma_r,
    create_sensor_range_dict,
    denormalize_to_physical,
    get_black_level_white_level_from_metadata,
    get_sensor_calibration_params,
    get_sensor_from_metadata,
    optimize_sigma,
    resolve_calibration_directories,
    validate_exposure_ratio,
    validate_physical_consistency,
)
from core.utils.tiles_utils import (
    save_scene_metrics_json,
    save_scene_tiles,
    save_tile_files,
)
from sample.sensor_noise_calibrations import SensorCalibration
from visualization.visualizations import (
    compute_scene_aggregate_metrics,
    create_comprehensive_comparison,
)
from visualization.visualizations import (
    create_scene_visualization_sample as create_scene_visualization,
)

try:
    from analysis.stratified_evaluation import StratifiedEvaluator

    STRATIFIED_EVAL_AVAILABLE = True
except ImportError:
    STRATIFIED_EVAL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Posterior sampling for image restoration using EDM model with Poisson-Gaussian measurement guidance"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to trained model checkpoint (.pkl)",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        required=False,
        help="Path to metadata JSON file with test split",
    )
    parser.add_argument(
        "--short_dir",
        type=str,
        required=False,
        help="Directory containing short exposure .pt files",
    )
    parser.add_argument(
        "--long_dir",
        type=str,
        default=None,
        help="Directory containing long exposure reference .pt files (optional, for optimization)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/low_light_enhancement"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=3,
        help="Number of example images to enhance",
    )
    parser.add_argument(
        "--tile_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific tile IDs to process",
    )
    parser.add_argument(
        "--skip_visualization",
        action="store_true",
        help="Skip creating visualization images",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="Number of posterior sampling steps",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_heun", action="store_true", help="Disable Heun's 2nd order correction"
    )

    parser.add_argument(
        "--use_sensor_calibration",
        action="store_true",
        help="Use calibrated sensor parameters",
    )
    parser.add_argument(
        "--sensor_name",
        type=str,
        default=None,
        help="Sensor model name for calibration",
    )
    parser.add_argument(
        "--calibration_dir",
        type=str,
        default=None,
        help="Directory containing noise calibration JSON files (from sensor_noise_calibrations.py). "
        "If not specified, searches in dataset/processed",
    )
    parser.add_argument(
        "--no-noise-calibration",
        dest="use_noise_calibration",
        action="store_false",
        default=True,
        help="Disable noise calibration and use --sigma_r instead (not recommended - calibration is enabled by default)",
    )
    parser.add_argument(
        "--sensor_filter", type=str, default=None, help="Filter tiles by sensor type"
    )
    parser.add_argument(
        "--conservative_factor",
        type=float,
        default=DEFAULT_CONSERVATIVE_FACTOR,
        help="Conservative multiplier for sigma_max",
    )

    parser.add_argument(
        "--optimize_sigma", action="store_true", help="Search for optimal sigma_max"
    )
    parser.add_argument(
        "--sigma_range",
        type=float,
        nargs=2,
        default=[0.0001, 0.01],
        help="Min and max sigma_max for optimization",
    )
    parser.add_argument(
        "--num_sigma_trials",
        type=int,
        default=10,
        help="Number of sigma_max values to try",
    )
    parser.add_argument(
        "--optimization_metric",
        type=str,
        default="ssim",
        choices=["ssim", "psnr", "mse"],
        help="Metric to optimize",
    )

    parser.add_argument("--s", type=float, default=None, help="Scale factor")
    parser.add_argument(
        "--sigma_r",
        type=float,
        default=DEFAULT_SIGMA_R,
        help="Read noise standard deviation (ADU) - ONLY used if --no-noise-calibration is set (not recommended, use calibration instead)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=DEFAULT_KAPPA,
        help="Guidance strength multiplier",
    )
    parser.add_argument(
        "--tau", type=float, default=DEFAULT_TAU, help="Guidance threshold"
    )
    parser.add_argument(
        "--pg_mode",
        type=str,
        default="wls",
        choices=["wls", "full"],
        help="PG guidance mode",
    )
    parser.add_argument(
        "--guidance_level",
        type=str,
        default="x0",
        choices=["x0", "score"],
        help="Guidance level",
    )
    parser.add_argument(
        "--compare_gaussian", action="store_true", help="Also run Gaussian guidance"
    )
    parser.add_argument(
        "--include_score_level",
        action="store_true",
        help="Include score-level guidance",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=None,
        help="Observation noise for Gaussian guidance",
    )

    parser.add_argument(
        "--run_methods",
        type=str,
        nargs="+",
        choices=["short", "long", "exposure_scaled", "gaussian_x0", "pg_x0"],
        default=["short", "long", "exposure_scaled", "gaussian_x0", "pg_x0"],
        help="Methods to run",
    )
    parser.add_argument(
        "--evaluate_stratified",
        action="store_true",
        help="Enable stratified evaluation",
    )
    parser.add_argument(
        "--stratified_domain_min", type=float, default=512, help="Black level ADU"
    )
    parser.add_argument(
        "--stratified_domain_max", type=float, default=16383, help="White level ADU"
    )
    parser.add_argument(
        "--test_gradients", action="store_true", help="Run gradient verification tests"
    )
    parser.add_argument(
        "--validate_exposure_ratios",
        action="store_true",
        help="Validate exposure ratios",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (process multiple tiles simultaneously)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading (0=sequential, >0=parallel)",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches to prefetch per worker",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Enable multi-GPU processing using torch.distributed",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3')",
    )

    return parser


def _validate_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.test_gradients:
        return

    if args.model_path is None:
        parser.error("--model_path is required for inference")
    if args.metadata_json is None:
        parser.error("--metadata_json is required for inference")
    if args.short_dir is None:
        parser.error("--short_dir is required for inference")
    if args.optimize_sigma and args.long_dir is None:
        parser.error("--optimize_sigma requires --long_dir to be specified")


def _create_guidance_modules(
    args: argparse.Namespace,
    sensor_range: Dict[str, float],
    s_sensor: float,
    exposure_ratio: float,
    calibration_data: Optional[Dict[str, float]] = None,
    black_level: Optional[float] = None,
    white_level: Optional[float] = None,
) -> Tuple[PoissonGaussianGuidance, GaussianGuidance]:
    black_level = black_level or sensor_range["min"]
    white_level = white_level or sensor_range["max"]

    if calibration_data is not None:
        missing = [k for k in ["a", "b"] if k not in calibration_data]
        if missing:
            raise ValueError(
                f"Invalid calibration data: missing {missing}. "
                f"Got keys: {list(calibration_data.keys())}"
            )
        sigma_r = convert_calibration_to_sigma_r(calibration_data["b"], s_sensor)
        poisson_coeff = convert_calibration_to_poisson_coeff(
            calibration_data["a"], s_sensor
        )
    else:
        sigma_r = args.sigma_r
        poisson_coeff = None
        logger.warning(
            f"Using command-line sigma_r={sigma_r:.3f} ADU (calibration disabled)"
        )

    common_params = {
        "s": s_sensor,
        "sigma_r": sigma_r,
        "black_level": black_level,
        "white_level": white_level,
        "offset": 0.0,
        "exposure_ratio": exposure_ratio,
        "kappa": args.kappa,
        "tau": args.tau,
    }

    pg_guidance = PoissonGaussianGuidance(
        **common_params,
        mode=args.pg_mode,
        guidance_level=args.guidance_level,
        poisson_coeff=poisson_coeff,
    )

    gaussian_guidance = GaussianGuidance(**common_params)

    return pg_guidance, gaussian_guidance


def _select_best_restoration_method(
    restoration_results: Dict[str, torch.Tensor],
    short_image: torch.Tensor,
) -> torch.Tensor:
    """Select the best available restoration method (priority: pg_x0 > gaussian_x0 > exposure_scaled > short)."""
    for method in ["pg_x0", "gaussian_x0", "exposure_scaled"]:
        if method in restoration_results:
            return restoration_results[method]
    return short_image


def _run_restoration_methods(
    args: argparse.Namespace,
    sampler: EDMPosteriorSampler,
    short_image: torch.Tensor,
    long_image: torch.Tensor,
    sigma_used: float,
    pg_guidance: PoissonGaussianGuidance,
    gaussian_guidance: GaussianGuidance,
    y_e: torch.Tensor,
    exposure_ratio: float,
    class_labels: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Run restoration methods for a single image."""
    restoration_results = {}
    run_methods = set(args.run_methods)

    if "exposure_scaled" in run_methods:
        restoration_results["exposure_scaled"] = apply_exposure_scaling(
            short_image, exposure_ratio
        )

    for method, guidance, is_gaussian in [
        ("gaussian_x0", gaussian_guidance, True),
        ("pg_x0", pg_guidance, False),
    ]:
        if method in run_methods:
            restored, _ = sampler.posterior_sample(
                y_observed=short_image,
                sigma_max=sigma_used,
                num_steps=args.num_steps,
                exposure_ratio=exposure_ratio,
                pg_guidance=guidance if not is_gaussian else None,
                gaussian_guidance=guidance if is_gaussian else None,
                y_e=y_e,
                no_heun=args.no_heun,
                class_labels=class_labels,
                apply_exposure_scaling=True,
            )
            restoration_results[method] = restored

    if "short" in run_methods:
        restoration_results["short"] = short_image
    if "long" in run_methods and long_image is not None:
        restoration_results["long"] = long_image

    return restoration_results


def _compute_all_metrics(
    long_image: torch.Tensor,
    short_image: torch.Tensor,
    restoration_results: Dict[str, torch.Tensor],
    short_phys: np.ndarray,
    y_e: torch.Tensor,
    s_sensor: float,
    extracted_sensor: str,
    sampler: EDMPosteriorSampler,
) -> Dict[str, Dict[str, float]]:
    metrics_results = {}

    try:
        metrics_results["short"] = compute_metrics_by_method(
            long_image, short_image, "short", device=sampler.device
        )
    except Exception as e:
        logger.warning(f"Metrics computation failed for short: {e}")
        metrics_results["short"] = get_nan_metrics_for_method("short")

    for method, restored_tensor in restoration_results.items():
        if restored_tensor is None:
            continue

        try:
            metrics_results[method] = compute_metrics_by_method(
                long_image, restored_tensor, method, device=sampler.device
            )
        except Exception as e:
            logger.warning(f"Metrics computation failed for {method}: {e}")
            metrics_results[method] = get_nan_metrics_for_method(method)

    return metrics_results


def _configure_guidance_modules(
    pg_guidance: PoissonGaussianGuidance,
    gaussian_guidance: GaussianGuidance,
    exposure_ratio: float,
    offset: float = 0.0,
) -> None:
    for guidance_module in [pg_guidance, gaussian_guidance]:
        guidance_module.alpha = exposure_ratio
        guidance_module.offset = offset


def _setup_guidance_modules(
    args: argparse.Namespace,
    tile_metadata: Dict[str, Any],
    exposure_ratio: float,
) -> Tuple[Any, Any, Optional[Dict[str, float]]]:
    calibration_data = None
    if args.use_noise_calibration:
        calibration_dir, data_root = resolve_calibration_directories(
            args.calibration_dir, args.metadata_json
        )
        calibration_data = load_noise_calibration(
            tile_metadata["extracted_sensor"], calibration_dir, data_root=data_root
        )

    sensor_range = create_sensor_range_dict(
        tile_metadata["black_level"], tile_metadata["white_level"]
    )

    pg_guidance, gaussian_guidance = _create_guidance_modules(
        args,
        sensor_range,
        tile_metadata["s_sensor"],
        exposure_ratio,
        calibration_data=calibration_data,
        black_level=tile_metadata["black_level"],
        white_level=tile_metadata["white_level"],
    )

    offset = tile_metadata["short_metadata"].get("offset", 0.0)
    _configure_guidance_modules(pg_guidance, gaussian_guidance, exposure_ratio, offset)

    return pg_guidance, gaussian_guidance, calibration_data


def _process_single_tile(
    idx: int,
    tile_info: Dict,
    args: argparse.Namespace,
    sampler: EDMPosteriorSampler,
    tile_lookup: Dict,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    tile_id = tile_info["tile_id"]

    try:
        tile_metadata = load_tile_and_metadata(
            tile_id=tile_id,
            tile_info=tile_info,
            tile_lookup=tile_lookup,
            short_dir=args.short_dir,
            long_dir=args.long_dir,
            device=sampler.device,
            img_channels=sampler.net.img_channels,
            sensor_ranges=sampler.sensor_ranges,
            use_sensor_calibration=args.use_sensor_calibration,
            sensor_name=args.sensor_name,
            conservative_factor=args.conservative_factor,
            sigma_r=args.sigma_r,
        )
        if tile_metadata is None:
            return None

        short_image = tile_metadata["short_image"]
        short_metadata = tile_metadata["short_metadata"]
        long_image = tile_metadata["long_image"]
        short_phys = tile_metadata["short_phys"]
        black_level = tile_metadata["black_level"]
        white_level = tile_metadata["white_level"]
        s_sensor = tile_metadata["s_sensor"]
        extracted_sensor = tile_metadata["extracted_sensor"]
        exposure_ratio = tile_metadata["exposure_ratio"]
        estimated_sigma = tile_metadata["estimated_sigma"]
        noise_estimates = tile_metadata.get("noise_estimates")

        if not args.use_sensor_calibration:
            raise ValueError(
                "Sensor calibration is required. Please use --use_sensor_calibration"
            )

        short_brightness = analyze_image_brightness(short_image)
        sensor_range = create_sensor_range_dict(black_level, white_level)

        pg_guidance, gaussian_guidance, calibration_data = _setup_guidance_modules(
            args, tile_metadata, exposure_ratio
        )

        if calibration_data is None and args.use_noise_calibration:
            raise FileNotFoundError(
                f"Calibration data required but not found for sensor '{extracted_sensor}'. "
                f"Please run sensor_noise_calibrations.py to generate calibration files, "
                f"or use --no-noise-calibration to disable calibration and use --sigma_r instead."
            )

        y_e = torch.from_numpy(short_phys).to(sampler.device)

        if long_image is None:
            raise FileNotFoundError(f"Long exposure image not found for tile {tile_id}")

        if args.validate_exposure_ratios:
            try:
                _, error_percent = validate_exposure_ratio(
                    short_image, long_image, exposure_ratio, extracted_sensor
                )
                if error_percent > DEFAULT_EXPOSURE_RATIO_ERROR_THRESHOLD:
                    logger.error(
                        "Exposure ratio mismatch detected! Consider updating hardcoded values."
                    )
            except Exception as e:
                logger.warning(f"Exposure ratio validation failed: {e}")

    except FileNotFoundError as e:
        logger.error(f"Long exposure image required but not found for tile {tile_id}")
        raise
    except Exception as e:
        logger.error(f"Failed to process tile {tile_id}: {e}")
        traceback.print_exc()
        return None

    opt_results = None
    if args.optimize_sigma:
        best_sigma, opt_results = optimize_sigma(
            sampler,
            short_image,
            long_image,
            None,
            sigma_range=tuple(args.sigma_range),
            num_trials=args.num_sigma_trials,
            num_steps=args.num_steps,
            metric=args.optimization_metric,
            pg_guidance=pg_guidance,
            y_e=y_e,
            exposure_ratio=exposure_ratio,
            no_heun=args.no_heun,
        )
        sigma_used = best_sigma
    else:
        sigma_used = estimated_sigma

    restoration_results = _run_restoration_methods(
        args,
        sampler,
        short_image,
        long_image,
        sigma_used,
        pg_guidance,
        gaussian_guidance,
        y_e,
        exposure_ratio,
        None,
    )

    restored = _select_best_restoration_method(restoration_results, short_image)

    metrics_results = _compute_all_metrics(
        long_image,
        short_image,
        restoration_results,
        short_phys,
        y_e,
        s_sensor,
        extracted_sensor,
        sampler,
    )

    restored_01 = convert_range(restored, "[-1,1]", "[0,1]")
    consistency = validate_physical_consistency(
        restored_01,
        y_e,
        s_sensor,
        args.sigma_r,
        exposure_ratio,
        sensor_range["min"],
        sensor_range["max"],
    )

    result_info = {
        "tile_id": tile_id,
        "sigma_max_used": float(sigma_used),
        "exposure_ratio": float(exposure_ratio),
        "brightness_analysis": short_brightness,
        "use_pg_guidance": True,
        "pg_guidance_params": {
            "s": args.s,
            "sigma_r": pg_guidance.sigma_r,
            "exposure_ratio": float(exposure_ratio),
            "kappa": args.kappa,
            "tau": args.tau,
            "mode": args.pg_mode,
        },
        "noise_calibration_used": calibration_data is not None,
        "physical_consistency": consistency,
        "comprehensive_metrics": filter_metrics_for_json(
            metrics_results, sensor_type=extracted_sensor
        ),
        "restoration_methods": list(restoration_results.keys()),
    }

    if args.use_sensor_calibration:
        result_info["sigma_determination"] = "sensor_calibration"
        result_info["sensor_calibration"] = noise_estimates
        result_info["extracted_sensor"] = extracted_sensor

    if calibration_data is not None:
        poisson_coeff_used = pg_guidance.poisson_coeff
        result_info["noise_calibration"] = {
            "source": "sensor_noise_calibrations.py",
            "a": calibration_data["a"],
            "b": calibration_data["b"],
            "sigma_r_from_calibration": convert_calibration_to_sigma_r(
                calibration_data["b"], s_sensor
            ),
            "poisson_coeff_from_calibration": poisson_coeff_used,
            "poisson_coeff_conversion_note": "a_norm * (sensor_range/2) converted from [-1,1] to physical space",
        }

    filtered_metrics = filter_metrics_for_json(
        metrics_results, sensor_type=extracted_sensor
    )
    if "pg_score" in filtered_metrics:
        result_info["metrics"] = filtered_metrics["pg_score"]
    elif "pg_x0" in filtered_metrics:
        result_info["metrics"] = filtered_metrics["pg_x0"]

    if opt_results is not None:
        result_info["optimization_results"] = opt_results

    return result_info


def _select_tiles_for_processing(
    args: argparse.Namespace, tile_lookup: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return select_tiles_for_processing(
        tile_ids=args.tile_ids,
        metadata_json=Path(args.metadata_json),
        tile_lookup=tile_lookup,
        num_examples=args.num_examples,
        seed=args.seed,
        sensor_filter=args.sensor_filter,
    )


def _run_stratified_evaluation(
    args: argparse.Namespace,
    all_results: List[Dict[str, Any]],
    sampler: EDMPosteriorSampler,
) -> Optional[Dict[str, Any]]:
    if not args.evaluate_stratified or not STRATIFIED_EVAL_AVAILABLE:
        return None

    try:
        stratified_eval = StratifiedEvaluator(
            sensor_ranges={
                "min": args.stratified_domain_min,
                "max": args.stratified_domain_max,
            }
        )
        all_stratified_comparison = {}
        all_improvements = {}

        for idx, result_info in enumerate(all_results):
            tile_id = result_info["tile_id"]
            if "restoration_results" not in result_info:
                logger.warning(f"Skipping {tile_id}: no restoration_results")
                continue

            long_path = Path(args.long_dir) / f"{tile_id}.pt" if args.long_dir else None
            if long_path and long_path.exists():
                try:
                    long_tile = load_tensor(
                        long_path,
                        device=sampler.device,
                        map_location=str(sampler.device),
                        weights_only=False,
                    )
                except Exception as e:
                    logger.error(f"Failed to load {long_path}: {e}")
                    continue
            else:
                continue

            restoration_results = result_info["restoration_results"]
            stratified_comparison = stratified_eval.compare_methods_stratified(
                long_tile, restoration_results
            )

            if (
                "gaussian_x0" in stratified_comparison
                and "pg_x0" in stratified_comparison
            ):
                improvements = stratified_eval.compute_improvement_matrix(
                    stratified_comparison["gaussian_x0"],
                    stratified_comparison["pg_x0"],
                )
                all_stratified_comparison[tile_id] = stratified_comparison
                all_improvements[tile_id] = improvements
                result_info["stratified_metrics"] = stratified_comparison
                result_info["stratified_improvements"] = improvements

        if all_stratified_comparison:
            stratified_significance = stratified_eval.test_statistical_significance(
                all_results,
                baseline_method="gaussian_x0",
                proposed_method="pg_x0",
                alpha=DEFAULT_STRATIFIED_ALPHA,
            )

            stratified_results = {
                "comparison_by_tile": all_stratified_comparison,
                "improvements_by_tile": all_improvements,
                "statistical_significance": stratified_significance,
            }

            stratified_results_path = Path(args.output_dir) / "stratified_results.json"
            save_json_file(stratified_results_path, stratified_results, default=str)

            return stratified_results

    except Exception as e:
        logger.error(f"Stratified evaluation failed: {e}", exc_info=True)
        logger.warning("Continuing with non-stratified results...")

    return None


def _build_summary(
    args: argparse.Namespace,
    all_results: List[Dict[str, Any]],
    stratified_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = {
        "num_samples": len(all_results),
        "optimize_sigma": args.optimize_sigma,
        "use_pg_guidance": True,
        "pg_guidance_params": {
            "s": args.s,
            "sigma_r": args.sigma_r,
            "kappa": args.kappa,
            "tau": args.tau,
            "mode": args.pg_mode,
        },
        "results": all_results,
    }

    if args.use_sensor_calibration:
        summary["sigma_determination"] = "sensor_calibration"
        summary["conservative_factor"] = args.conservative_factor
        summary["sensor_extraction"] = (
            "auto_detected" if args.sensor_name is None else "manual"
        )

    if len(all_results) > 0:
        chi_squared_values = [
            r["physical_consistency"]["chi_squared"]
            for r in all_results
            if "physical_consistency" in r
        ]
        if chi_squared_values:
            summary["aggregate_physical_consistency"] = {
                "mean_chi_squared": float(np.mean(chi_squared_values)),
                "std_chi_squared": float(np.std(chi_squared_values)),
                "median_chi_squared": float(np.median(chi_squared_values)),
                "num_physically_consistent": sum(
                    r["physical_consistency"]["physically_consistent"]
                    for r in all_results
                    if "physical_consistency" in r
                ),
                "total_samples": len(chi_squared_values),
            }

    if len(all_results) > 0 and "comprehensive_metrics" in all_results[0]:
        all_methods = set()
        for result in all_results:
            if "comprehensive_metrics" in result:
                all_methods.update(result["comprehensive_metrics"].keys())

        summary["comprehensive_aggregate_metrics"] = {}
        for method in all_methods:
            metrics_for_method = [
                result["comprehensive_metrics"][method]
                for result in all_results
                if "comprehensive_metrics" in result
                and method in result["comprehensive_metrics"]
            ]

            if metrics_for_method:
                metric_names = set()
                for m in metrics_for_method:
                    metric_names.update(m.keys())

                metric_stats = {}
                for metric_name in metric_names:
                    values = [
                        m[metric_name]
                        for m in metrics_for_method
                        if metric_name in m and not np.isnan(m[metric_name])
                    ]
                    if values:
                        metric_stats[f"mean_{metric_name}"] = np.mean(values)
                        metric_stats[f"std_{metric_name}"] = np.std(values)

                metric_stats["num_samples"] = len(metrics_for_method)
                summary["comprehensive_aggregate_metrics"][method] = metric_stats

    if stratified_results is not None:
        summary["stratified_evaluation"] = {
            "enabled": True,
            "sensor_ranges": {
                "min": args.stratified_domain_min,
                "max": args.stratified_domain_max,
            },
            "statistical_significance": stratified_results.get(
                "statistical_significance"
            ),
            "num_tiles_analyzed": len(stratified_results.get("comparison_by_tile", {})),
        }
    else:
        summary["stratified_evaluation"] = {
            "enabled": False,
            "reason": "Stratified evaluation not enabled or no clean images provided",
        }

    return summary


def main() -> None:
    parser = _setup_argument_parser()
    args = parser.parse_args()

    if args.test_gradients:
        run_gradient_verification()
        return

    _validate_arguments(args, parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = EDMPosteriorSampler(model_path=args.model_path, device=args.device)

    if args.s is None:
        default_range = DEFAULT_SENSOR_RANGES["sony"]
        args.s = default_range["max"] - default_range["min"]

    tile_lookup = load_metadata_json(Path(args.metadata_json))

    try:
        selected_tiles = _select_tiles_for_processing(args, tile_lookup)
    except ValueError as e:
        logger.error(str(e))
        return
    all_results = []

    if args.batch_size > 1:
        all_results = process_tiles_in_batches(
            selected_tiles,
            args,
            sampler,
            tile_lookup,
            output_dir,
            _process_single_tile,
            lambda tile_id, tile_info, tile_lookup, args, sampler: load_tile_and_metadata(
                tile_id=tile_id,
                tile_info=tile_info,
                tile_lookup=tile_lookup,
                short_dir=args.short_dir,
                long_dir=args.long_dir,
                device=sampler.device,
                img_channels=sampler.net.img_channels,
                sensor_ranges=sampler.sensor_ranges,
                use_sensor_calibration=args.use_sensor_calibration,
                sensor_name=args.sensor_name,
                conservative_factor=args.conservative_factor,
                sigma_r=args.sigma_r,
            ),
            _setup_guidance_modules,
            _compute_all_metrics,
        )
    else:
        if args.num_workers > 0:
            from torch.utils.data import DataLoader

            tile_dataset = TileDataset(selected_tiles)
            tile_loader = DataLoader(
                tile_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                prefetch_factor=args.prefetch_factor,
            )

            for idx, batch in enumerate(tile_loader):
                tile_info = batch[0]
                result_info = _process_single_tile(
                    idx, tile_info, args, sampler, tile_lookup, output_dir
                )
                if result_info is not None:
                    all_results.append(result_info)
        else:
            for idx, tile_info in enumerate(selected_tiles):
                result_info = _process_single_tile(
                    idx, tile_info, args, sampler, tile_lookup, output_dir
                )
                if result_info is not None:
                    all_results.append(result_info)

    stratified_results = _run_stratified_evaluation(args, all_results, sampler)

    summary = _build_summary(args, all_results, stratified_results)
    save_json_file(output_dir / "results.json", summary)


if __name__ == "__main__":
    main()
