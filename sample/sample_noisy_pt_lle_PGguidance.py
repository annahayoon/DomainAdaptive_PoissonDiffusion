#!/usr/bin/env python3
"""
Posterior Sampling for Image Restoration using EDM Model with Poisson-Gaussian Guidance

This script performs posterior sampling on short exposure test images using EDM with
exposure-aware Poisson-Gaussian measurement guidance for physics-informed restoration.

Theory: We sample from p(x|y) ∝ p(y|x) p(x) where p(x) is the EDM-learned prior
and p(y|x) is the Poisson-Gaussian likelihood.

Usage:
    python sample/sample_noisy_pt_lle_PGguidance.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --short_dir dataset/processed/pt_tiles/photography/short \
        --output_dir results/posterior_sampling \
        --num_examples 3 \
        --num_steps 18 \
        --sigma_r 5.0 \
        --kappa 0.5
"""

import argparse
import json
import pickle
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
from sample.gradients import (
    GaussianGuidance,
    PoissonGaussianGuidance,
    run_gradient_verification,
)
from sample.metrics import (
    compute_metrics_by_method,
    convert_range,
    filter_metrics_for_json,
)
from sample.sample_utils import (
    analyze_image_brightness,
    apply_exposure_scaling,
    denormalize_to_physical,
    find_long_tile_pair,
    get_exposure_ratio,
    get_exposure_time_from_metadata,
    get_sensor_calibration_params,
    get_sensor_from_metadata,
    load_long_image,
    load_metadata_json,
    load_short_image,
    load_test_tiles,
    optimize_sigma,
    select_tiles,
    validate_exposure_ratio,
    validate_physical_consistency,
)
from sample.sample_visualizations import create_comprehensive_comparison
from sample.sensor_calibration import SensorCalibration

try:
    from analysis.stratified_evaluation import StratifiedEvaluator

    STRATIFIED_EVAL_AVAILABLE = True
except ImportError:
    STRATIFIED_EVAL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EDMPosteriorSampler:
    """EDM-based posterior sampler with Poisson-Gaussian measurement guidance."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        domain_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize the enhancer with trained model and sensor configurations."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
        except Exception as e:
            logger.warning(f"torch.load failed: {e}, trying pickle.load...")
            try:
                with open(model_path, "rb") as f:
                    checkpoint = pickle.load(f)  # nosec B301
            except (ModuleNotFoundError, AttributeError) as pickle_e:
                if "numpy._core" in str(pickle_e) or "scalar" in str(pickle_e):
                    logger.error("=" * 80)
                    logger.error("NUMPY VERSION MISMATCH DETECTED")
                    logger.error("=" * 80)
                    logger.error(f"Model file: {model_path}")
                    logger.error(f"Current numpy version: {np.__version__}")
                    logger.error(f"Current Python version: {sys.version.split()[0]}")
                    logger.error("")
                    logger.error(
                        "The photography model was saved with numpy 2.0+ which is incompatible"
                    )
                    logger.error(
                        "with Python 3.8 and numpy 1.24.4 in this environment."
                    )
                    logger.error("")
                    logger.error("SOLUTIONS:")
                    logger.error(
                        "  1. Use Python 3.10+: /usr/bin/python3.10 or /usr/bin/python3.11"
                    )
                    logger.error(
                        "  2. Repackage the model on a numpy 1.x compatible system"
                    )
                    logger.error(
                        "  3. Contact the model provider for a compatible version"
                    )
                    logger.error("=" * 80)
                    raise RuntimeError(
                        f"Photography model requires numpy 2.0+ (Python 3.10+) but environment has "
                        f"Python {sys.version.split()[0]} with numpy {np.__version__}. "
                        f"Try running with: /usr/bin/python3.10 or /usr/bin/python3.11"
                    ) from pickle_e
                raise

        self.net = checkpoint["ema"].to(self.device)
        self.net.eval()
        self.exposure_ratio = None
        self.sensor_ranges = {
            "sony": {"min": 0.0, "max": 16383.0},
            "fuji": {"min": 0.0, "max": 16383.0},
        }

    def posterior_sample(
        self,
        y_observed: torch.Tensor,
        sigma_max: float,
        class_labels: Optional[torch.Tensor] = None,
        num_steps: int = 18,
        rho: float = 7.0,
        pg_guidance: Optional[PoissonGaussianGuidance] = None,
        gaussian_guidance: Optional[GaussianGuidance] = None,
        y_e: Optional[torch.Tensor] = None,
        x_init: Optional[torch.Tensor] = None,
        exposure_ratio: float = 1.0,
        no_heun: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Posterior sampling using EDM model with measurement guidance.

        Args:
            y_observed: Observed short exposure image (B, C, H, W) in [-1, 1] (model space)
            sigma_max: Maximum noise level to start from
            class_labels: Optional class labels (not used for unconditional model)
            num_steps: Number of sampling steps
            rho: Time step parameter (EDM default: 7.0)
            pg_guidance: Poisson-Gaussian guidance module (physics-informed)
            gaussian_guidance: Gaussian guidance module (standard, for comparison)
            y_e: Observed short exposure measurement in physical units (for PG guidance)
            x_init: Optional initialization (if None, uses observation)
            exposure_ratio: Exposure ratio t_low / t_long
            no_heun: Disable Heun's 2nd order correction for 2x speedup (minimal quality loss)

        Returns:
            Tuple of (restored_tensor, results_dict)
        """
        self.exposure_ratio = exposure_ratio
        sigma_min = max(0.002, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )

        if x_init is None:
            x_init = (
                apply_exposure_scaling(y_observed, exposure_ratio)
                .to(torch.float64)
                .to(self.device)
            )
        else:
            x_init = x_init.to(torch.float64).to(self.device)
        x = x_init

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_denoised = self.net(x, t_cur, class_labels).to(torch.float64)

            # Apply guidance to current prediction
            guidance = pg_guidance if pg_guidance is not None else gaussian_guidance
            guidance_contribution = None
            if guidance is not None and y_e is not None:
                x_denoised, guidance_contribution = self._apply_guidance(
                    x_denoised, guidance, y_e, t_cur.item()
                )

            d_cur = (x - x_denoised) / t_cur
            # Apply score-level guidance contribution to derivative
            if guidance_contribution is not None:
                d_cur = d_cur - t_cur * guidance_contribution
            x_next = x + (t_next - t_cur) * d_cur

            if not no_heun and i < num_steps - 1:
                x_denoised_next = self.net(x_next, t_next, class_labels).to(
                    torch.float64
                )
                # Apply guidance to next prediction
                guidance_contribution_next = None
                if guidance is not None and y_e is not None:
                    x_denoised_next, guidance_contribution_next = self._apply_guidance(
                        x_denoised_next, guidance, y_e, t_next.item()
                    )

                d_next = (x_next - x_denoised_next) / t_next
                # Apply score-level guidance contribution to next derivative
                if guidance_contribution_next is not None:
                    d_next = d_next - t_next * guidance_contribution_next
                x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_next)
            x = x_next
        restored_output = torch.clamp(x, -1.0, 1.0)

        results = {
            "restored": restored_output,
            "observed": y_observed,
            "sigma_max": sigma_max,
            "num_steps": num_steps,
            "pg_guidance_used": pg_guidance is not None,
            "initialization": "gaussian_noise" if x_init is None else "provided",
        }

        return restored_output, results

    def _apply_guidance(
        self,
        x_denoised: torch.Tensor,
        guidance: Any,
        y_e: torch.Tensor,
        t: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply guidance to denoised prediction."""
        x_denoised_01 = torch.clamp(
            convert_range(x_denoised, "[-1,1]", "[0,1]"), 0.0, 1.0
        )

        if guidance.guidance_level == "x0":
            x_guided = guidance(x_denoised_01, y_e, t)
            x_denoised_guided = torch.clamp(
                convert_range(x_guided, "[0,1]", "[-1,1]"), -1.0, 1.0
            )
            return x_denoised_guided, None
        elif guidance.guidance_level == "score":
            likelihood_gradient = guidance.compute_likelihood_gradient(
                x_denoised_01, y_e
            )
            guidance_contribution = guidance.kappa * likelihood_gradient
            return x_denoised, guidance_contribution
        else:
            raise ValueError(f"Unknown guidance_level: {guidance.guidance_level}")


def _setup_argument_parser() -> argparse.ArgumentParser:
    """Setup and return argument parser with all arguments grouped by category."""
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
        "--num_steps", type=int, default=18, help="Number of posterior sampling steps"
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
        "--sensor_filter", type=str, default=None, help="Filter tiles by sensor type"
    )
    parser.add_argument(
        "--conservative_factor",
        type=float,
        default=1.0,
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
        "--sigma_r", type=float, default=5.0, help="Read noise standard deviation (ADU)"
    )
    parser.add_argument(
        "--kappa", type=float, default=0.1, help="Guidance strength multiplier"
    )
    parser.add_argument("--tau", type=float, default=0.01, help="Guidance threshold")
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
    parser.add_argument("--fast_metrics", action="store_true", help="(Deprecated)")
    parser.add_argument(
        "--validate_exposure_ratios",
        action="store_true",
        help="Validate exposure ratios",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for processing"
    )

    return parser


def _validate_arguments(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Validate required arguments."""
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
) -> Tuple[PoissonGaussianGuidance, GaussianGuidance]:
    """Create and configure guidance modules with shared parameters."""
    # Common parameters for both guidance types
    common_params = {
        "s": s_sensor,
        "sigma_r": args.sigma_r,
        "domain_min": sensor_range["min"],
        "domain_max": sensor_range["max"],
        "offset": 0.0,
        "exposure_ratio": exposure_ratio,
        "kappa": args.kappa,
        "tau": args.tau,
    }

    pg_guidance = PoissonGaussianGuidance(
        **common_params,
        mode=args.pg_mode,
        guidance_level=args.guidance_level,
    )

    gaussian_guidance = GaussianGuidance(**common_params)

    return pg_guidance, gaussian_guidance


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
    """Run all requested restoration methods."""
    restoration_results = {}

    if "exposure_scaled" in args.run_methods:
        restoration_results["exposure_scaled"] = apply_exposure_scaling(
            short_image, exposure_ratio
        )

    # Common parameters for posterior sampling
    _common_sample_params = {
        "sigma_max": sigma_used,
        "class_labels": class_labels,
        "num_steps": args.num_steps,
        "no_heun": args.no_heun,
        "y_e": y_e,
        "exposure_ratio": exposure_ratio,
    }

    if "gaussian_x0" in args.run_methods:
        gaussian_guidance.guidance_level = "x0"
        restored_gaussian_x0, _ = sampler.posterior_sample(
            short_image,
            gaussian_guidance=gaussian_guidance,
            **_common_sample_params,
        )
        restoration_results["gaussian_x0"] = restored_gaussian_x0

    if "pg_x0" in args.run_methods:
        pg_guidance.guidance_level = "x0"
        restored_pg_x0, _ = sampler.posterior_sample(
            short_image,
            pg_guidance=pg_guidance,
            **_common_sample_params,
        )
        restoration_results["pg_x0"] = restored_pg_x0

    if "short" in args.run_methods:
        restoration_results["short"] = short_image
    if "long" in args.run_methods and long_image is not None:
        restoration_results["long"] = long_image

    return restoration_results


def _get_nan_metrics_for_method(method: str) -> Dict[str, float]:
    """Get appropriate NaN metrics dictionary based on method type."""
    if method == "short":
        return {"psnr": float("nan")}
    elif method == "exposure_scaled":
        return {
            "psnr": float("nan"),
            "ssim": float("nan"),
            "mse": float("nan"),
        }
    else:
        return {
            "psnr": float("nan"),
            "ssim": float("nan"),
            "mse": float("nan"),
            "lpips": float("nan"),
            "niqe": float("nan"),
        }


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
    """Compute metrics for all restoration methods."""
    metrics_results = {}

    try:
        metrics_results["short"] = compute_metrics_by_method(
            long_image, short_image, "short", device=sampler.device
        )
    except Exception as e:
        logger.warning(f"Metrics computation failed for short: {e}")
        metrics_results["short"] = _get_nan_metrics_for_method("short")

    for method, restored_tensor in restoration_results.items():
        if restored_tensor is None:
            continue

        try:
            metrics_results[method] = compute_metrics_by_method(
                long_image, restored_tensor, method, device=sampler.device
            )
        except Exception as e:
            logger.warning(f"Metrics computation failed for {method}: {e}")
            metrics_results[method] = _get_nan_metrics_for_method(method)

    return metrics_results


def _process_single_tile(
    idx: int,
    tile_info: Dict,
    args: argparse.Namespace,
    sampler: EDMPosteriorSampler,
    tile_lookup: Dict,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Process a single tile and return result info."""
    tile_id = tile_info["tile_id"]
    logger.info(f"Processing tile: {tile_id}")

    try:
        short_image, short_metadata = load_short_image(
            tile_id,
            Path(args.short_dir),
            sampler.device,
            target_channels=sampler.net.img_channels,
        )
        if short_image.ndim == 3:
            short_image = short_image.unsqueeze(0)
        short_image = short_image.to(torch.float32)

        extracted_sensor = get_sensor_from_metadata(tile_id, tile_lookup)
        from preprocessing.config import BLACK_LEVELS, SENSOR_RANGES

        black_level = BLACK_LEVELS.get(extracted_sensor, 512.0)
        white_level = SENSOR_RANGES.get(extracted_sensor, {"max": 16383.0})["max"]
        short_phys = denormalize_to_physical(short_image, black_level, white_level)
        short_brightness = analyze_image_brightness(short_image)

        if not args.use_sensor_calibration:
            raise ValueError(
                "Sensor calibration is required. Please use --use_sensor_calibration"
            )

        (
            estimated_sigma,
            noise_estimates,
            sensor_range,
            s_sensor,
        ) = get_sensor_calibration_params(
            args.sensor_name,
            extracted_sensor,
            short_phys,
            sampler.sensor_ranges,
            args.conservative_factor,
        )

        exposure_ratio = get_exposure_ratio(tile_id, tile_lookup)
        pg_guidance, gaussian_guidance = _create_guidance_modules(
            args, sensor_range, s_sensor, exposure_ratio
        )

        # Set guidance parameters
        offset = short_metadata.get("offset", 0.0)
        for guidance_module in [pg_guidance, gaussian_guidance]:
            guidance_module.alpha = exposure_ratio
            guidance_module.offset = offset

        y_e = torch.from_numpy(short_phys).to(sampler.device)

        # Load long exposure image
        long_image = None
        if args.long_dir is not None:
            try:
                long_image, _ = load_long_image(
                    tile_id=tile_id,
                    long_dir=Path(args.long_dir),
                    tile_lookup=tile_lookup,
                    sampler_device=sampler.device,
                    target_channels=sampler.net.img_channels,
                )
            except Exception as long_error:
                logger.error(f"Error loading long exposure image: {long_error}")
                traceback.print_exc()
                long_image = None

        if long_image is None:
            raise FileNotFoundError(f"Long exposure image not found for tile {tile_id}")

        # Validate exposure ratios if requested
        if args.validate_exposure_ratios:
            try:
                measured_alpha, error_percent = validate_exposure_ratio(
                    short_image, long_image, exposure_ratio, extracted_sensor
                )
                if error_percent > 20.0:
                    logger.error(
                        "Exposure ratio mismatch detected! Consider updating hardcoded values."
                    )
            except Exception as e:
                logger.warning(f"Exposure ratio validation failed: {e}")

    except FileNotFoundError as e:
        logger.error(f"Long exposure image required but not found for tile {tile_id}")
        raise e
    except Exception as e:
        logger.error(f"Failed to process tile {tile_id}: {e}")
        traceback.print_exc()
        return None

    opt_results = None
    if args.optimize_sigma:
        if long_image is None:
            logger.warning(
                f"No long exposure image available for optimization, skipping {tile_id}"
            )
            return None

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

    if "pg_x0" in restoration_results:
        restored = restoration_results["pg_x0"]
    elif "gaussian_x0" in restoration_results:
        restored = restoration_results["gaussian_x0"]
    elif "exposure_scaled" in restoration_results:
        restored = restoration_results["exposure_scaled"]
    else:
        restored = short_image

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

    sample_dir = (
        output_dir
        if "example_" in str(output_dir)
        else output_dir / f"example_{idx:02d}_{tile_id}"
    )
    sample_dir.mkdir(exist_ok=True)

    torch.save(short_image.cpu(), sample_dir / "short.pt")
    for method_name, restored_tensor in restoration_results.items():
        if restored_tensor is not None and method_name not in ["short", "long"]:
            torch.save(restored_tensor.cpu(), sample_dir / f"restored_{method_name}.pt")
    torch.save(long_image.cpu(), sample_dir / "long.pt")

    result_info = {
        "tile_id": tile_id,
        "sigma_max_used": float(sigma_used),
        "exposure_ratio": float(exposure_ratio),
        "brightness_analysis": short_brightness,
        "use_pg_guidance": True,
        "pg_guidance_params": {
            "s": args.s,
            "sigma_r": args.sigma_r,
            "exposure_ratio": float(exposure_ratio),
            "kappa": args.kappa,
            "tau": args.tau,
            "mode": args.pg_mode,
        },
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

    filtered_metrics = filter_metrics_for_json(
        metrics_results, sensor_type=extracted_sensor
    )
    if "pg_score" in filtered_metrics:
        result_info["metrics"] = filtered_metrics["pg_score"]
    elif "pg_x0" in filtered_metrics:
        result_info["metrics"] = filtered_metrics["pg_x0"]

    if opt_results is not None:
        result_info["optimization_results"] = opt_results

    with open(sample_dir / "results.json", "w") as f:
        json.dump(result_info, f, indent=2)

    if not args.skip_visualization:
        comparison_path = sample_dir / "restoration_comparison.png"
        create_comprehensive_comparison(
            short_image=short_image,
            enhancement_results=restoration_results,
            sensor_type=extracted_sensor,
            tile_id=tile_id,
            save_path=comparison_path,
            metadata_json_path=Path(args.metadata_json),
            long_image=long_image,
            exposure_ratio=exposure_ratio,
            metrics_results=metrics_results,
        )

    return result_info


def _build_summary(
    args: argparse.Namespace,
    all_results: List[Dict],
    stratified_results: Optional[Dict],
) -> Dict[str, Any]:
    """Build summary dictionary with aggregate metrics."""
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
            "domain_ranges": {
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


def main():
    """Main function for posterior sampling with Poisson-Gaussian measurement guidance."""
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

    default_sensor_ranges = {"min": 0.0, "max": 16383.0}
    if args.s is None:
        args.s = default_sensor_ranges["max"] - default_sensor_ranges["min"]

    tile_lookup = load_metadata_json(Path(args.metadata_json))
    test_tiles = load_test_tiles(
        Path(args.metadata_json), split="test", sensor_filter=args.sensor_filter
    )

    if len(test_tiles) == 0:
        logger.error("No test tiles found")
        return

    # Filter to only short exposure tiles
    short_tiles = [tile for tile in test_tiles if tile.get("data_type") == "short"]

    if len(short_tiles) == 0:
        logger.error("No short exposure tiles found")
        return

    selected_tiles = select_tiles(
        args.tile_ids, short_tiles, args.num_examples, args.seed
    )
    all_results = []

    for idx, tile_info in enumerate(selected_tiles):
        result_info = _process_single_tile(
            idx, tile_info, args, sampler, tile_lookup, output_dir
        )
        if result_info is not None:
            all_results.append(result_info)

    stratified_results = None
    if args.evaluate_stratified and STRATIFIED_EVAL_AVAILABLE:
        try:
            stratified_eval = StratifiedEvaluator(
                domain_ranges={
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

                long_path = (
                    Path(args.long_dir) / f"{tile_id}.pt" if args.long_dir else None
                )
                if long_path and long_path.exists():
                    long_tile = torch.load(long_path, map_location=sampler.device)
                else:
                    logger.debug(
                        f"Long exposure image not found for {tile_id}, skipping stratified analysis"
                    )
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
                    alpha=0.05,
                )

                stratified_results = {
                    "comparison_by_tile": all_stratified_comparison,
                    "improvements_by_tile": all_improvements,
                    "statistical_significance": stratified_significance,
                }

                stratified_results_path = (
                    Path(args.output_dir) / "stratified_results.json"
                )
                with open(stratified_results_path, "w") as f:
                    json.dump(stratified_results, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Stratified evaluation failed: {e}", exc_info=True)
            logger.warning("Continuing with non-stratified results...")

    summary = _build_summary(args, all_results, stratified_results)

    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Processed {len(all_results)} tiles")


if __name__ == "__main__":
    main()
