#!/usr/bin/env python3
"""Posterior Sampling for SIDD Dataset using EDM Model with Poisson-Gaussian Guidance."""

import argparse
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import h5py

import numpy as np
import torch
from scipy.io import loadmat

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

import logging

import external.edm.dnnlib
from config.sample_config import (
    DEFAULT_KAPPA,
    DEFAULT_NUM_STEPS,
    DEFAULT_RHO,
    DEFAULT_SIGMA_R,
)
from core.demosaic import (
    demosaic_bayer_to_rgb,
    extract_camera_id_from_scene_name,
    get_cfa_pattern_from_scene_name,
    load_bayer_patterns_csv,
)
from core.guidance import GaussianGuidance, PoissonGaussianGuidance
from core.metrics import compute_metrics_by_method, get_nan_metrics_for_method
from core.normalization import convert_range
from core.sampler import EDMPosteriorSampler
from core.utils.file_utils import load_mat_file, save_json_file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_cfa_pattern_from_metadata(metadata_dict: dict) -> Optional[List[int]]:
    """Extract CFA pattern from SIDD metadata MAT file."""
    metadata = metadata_dict.get("metadata", None)

    if metadata is None:
        possible_keys = ["CFAPattern", "CFAPattern2", "cfa_pattern", "CFA"]
        for key in possible_keys:
            if key in metadata_dict:
                val = metadata_dict[key]
                if isinstance(val, np.ndarray):
                    val = val.flatten()
                    if len(val) == 4:
                        cfa_idx = [int(x) for x in val]
                        return cfa_idx
        return None

    if isinstance(metadata, np.ndarray) and metadata.dtype == np.dtype("O"):
        try:
            if metadata.dtype.names:
                if "CFAPattern" in metadata.dtype.names:
                    cfa_val = metadata["CFAPattern"][0, 0]
                    if isinstance(cfa_val, np.ndarray):
                        cfa_val = cfa_val.flatten()
                        if len(cfa_val) == 4:
                            return [int(x) for x in cfa_val]
                elif "UnknownTags" in metadata.dtype.names:
                    ut = metadata["UnknownTags"][0, 0]
                    if isinstance(ut, np.ndarray) and len(ut) >= 2:
                        try:
                            cfa_val = ut[1]
                            if isinstance(cfa_val, np.ndarray):
                                cfa_val = cfa_val.flatten()
                                if len(cfa_val) == 4:
                                    return [int(x) for x in cfa_val]
                        except (IndexError, TypeError):
                            pass
        except (AttributeError, IndexError, TypeError):
            pass

    if isinstance(metadata, dict):
        if "CFAPattern" in metadata:
            cfa_val = metadata["CFAPattern"]
            if isinstance(cfa_val, (list, np.ndarray)):
                cfa_val = np.array(cfa_val).flatten()
                if len(cfa_val) == 4:
                    return [int(x) for x in cfa_val]

    if "SubIFDs" in metadata_dict:
        subifds = metadata_dict["SubIFDs"]
        if isinstance(subifds, (list, np.ndarray)) and len(subifds) > 0:
            subifd = subifds[0]
            if isinstance(subifd, dict) and "UnknownTags" in subifd:
                ut = subifd["UnknownTags"]
                if isinstance(ut, (list, np.ndarray)) and len(ut) >= 2:
                    try:
                        cfa_val = ut[1]
                        if isinstance(cfa_val, (list, np.ndarray)):
                            cfa_val = np.array(cfa_val).flatten()
                            if len(cfa_val) == 4:
                                return [int(x) for x in cfa_val]
                    except (IndexError, TypeError):
                        pass

    return None


def load_sidd_scene(
    scene_dir: Path, demosaic_type: str = "VNG"
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    """Load noisy and ground truth images from SIDD scene directory."""
    scene_name = scene_dir.name
    noisy_files = list(scene_dir.glob("NOISY_RAW_*.MAT")) + list(
        scene_dir.glob("NOISY_RAW_*.mat")
    )
    gt_files = list(scene_dir.glob("GT_RAW_*.MAT")) + list(
        scene_dir.glob("GT_RAW_*.mat")
    )

    if not noisy_files or not gt_files:
        logger.warning(f"No NOISY_RAW or GT_RAW files found in {scene_dir}")
        return None, None, scene_name

    noisy_file = noisy_files[0]
    gt_file = gt_files[0]
    noisy_data = load_mat_file(noisy_file, variable_name="x")
    gt_data = load_mat_file(gt_file, variable_name="x")

    if noisy_data is None or gt_data is None:
        logger.warning(f"Failed to load .mat files from {scene_dir}")
        return None, None, scene_name

    metadata_files = list(scene_dir.glob("METADATA_RAW_*.MAT")) + list(
        scene_dir.glob("METADATA_RAW_*.mat")
    )
    cfa_pattern = None

    if metadata_files:
        try:
            from scipy.io import loadmat as scipy_loadmat

            try:
                metadata_dict = scipy_loadmat(
                    str(metadata_files[0]), struct_as_record=False
                )
                cfa_pattern = extract_cfa_pattern_from_metadata(metadata_dict)
            except Exception:
                metadata_dict = scipy_loadmat(str(metadata_files[0]))
                cfa_pattern = extract_cfa_pattern_from_metadata(metadata_dict)
        except Exception:
            pass

    if cfa_pattern is None:
        cfa_pattern = get_cfa_pattern_from_scene_name(scene_name, return_string=False)

    if cfa_pattern is None:
        cfa_pattern = [0, 1, 1, 2]

    if noisy_data.ndim == 2:
        noisy_rgb = demosaic_bayer_to_rgb(
            noisy_data,
            cfa_pattern=cfa_pattern,
            output_channel_order="RGB",
            alg_type=demosaic_type,
        )
        gt_rgb = demosaic_bayer_to_rgb(
            gt_data,
            cfa_pattern=cfa_pattern,
            output_channel_order="RGB",
            alg_type=demosaic_type,
        )
    elif noisy_data.ndim == 3:
        if noisy_data.shape[2] <= 4:
            if noisy_data.shape[2] == 3:
                noisy_rgb = noisy_data
                gt_rgb = gt_data
            else:
                noisy_rgb = demosaic_bayer_to_rgb(
                    noisy_data[:, :, 0],
                    cfa_pattern=cfa_pattern,
                    output_channel_order="RGB",
                    alg_type=demosaic_type,
                )
                gt_rgb = demosaic_bayer_to_rgb(
                    gt_data[:, :, 0],
                    cfa_pattern=cfa_pattern,
                    output_channel_order="RGB",
                    alg_type=demosaic_type,
                )
        else:
            noisy_rgb = np.transpose(noisy_data, (1, 2, 0))
            gt_rgb = np.transpose(gt_data, (1, 2, 0))
    else:
        logger.warning(f"Unexpected data dimensions: {noisy_data.shape}")
        return None, None, scene_name

    noisy_rgb = ensure_rgb_format_3hw(noisy_rgb)
    gt_rgb = ensure_rgb_format_3hw(gt_rgb)
    noisy_tensor = torch.from_numpy(2.0 * noisy_rgb - 1.0).float()
    gt_tensor = torch.from_numpy(2.0 * gt_rgb - 1.0).float()

    return noisy_tensor, gt_tensor, scene_name


def ensure_rgb_format_3hw(image: np.ndarray) -> np.ndarray:
    """Ensure RGB image is in (3, H, W) format."""
    if image.ndim == 2:
        return np.stack([image, image, image], axis=0)
    elif image.ndim == 3:
        if image.shape[0] == 3:
            return image
        elif image.shape[2] == 3:
            return np.transpose(image, (2, 0, 1))
        else:
            if image.shape[0] == 3:
                return image
            return np.transpose(image, (2, 0, 1)) if image.shape[2] == 3 else image
    return image


def resize_to_divisible_by_8(
    image: torch.Tensor,
    max_size: int = 0,
    target_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Resize image to ensure dimensions are divisible by 8."""
    _, H, W = image.shape

    if target_size is not None:
        new_H, new_W = target_size
    elif max_size > 0 and (H > max_size or W > max_size):
        scale = min(max_size / H, max_size / W)
        new_H, new_W = int(H * scale), int(W * scale)
    else:
        new_H, new_W = H, W

    new_H = (new_H // 8) * 8
    new_W = (new_W // 8) * 8
    new_H = max(8, new_H)
    new_W = max(8, new_W)

    if new_H != H or new_W != W:
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(new_H, new_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return image


def format_metrics_for_display(metrics: Dict[str, Any]) -> List[str]:
    """Format metrics dictionary into list of display strings."""
    lines = []
    if "ssim" in metrics:
        lines.append(f"SSIM: {metrics['ssim']:.3f}")
    if "psnr" in metrics:
        lines.append(f"PSNR: {metrics['psnr']:.1f}dB")
    if "lpips" in metrics:
        lines.append(f"LPIPS: {metrics['lpips']:.3f}")
    if "niqe" in metrics:
        lines.append(f"NIQE: {metrics['niqe']:.2f}")
    if "mse" in metrics:
        lines.append(f"MSE: {metrics['mse']:.6f}")
    return lines


def process_sidd_scene(
    scene_dir: Path,
    sampler: EDMPosteriorSampler,
    calibration_data: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Process a single SIDD scene."""
    scene_name = scene_dir.name
    logger.info(f"Processing scene: {scene_name}")

    # Load images
    demosaic_type = getattr(args, "demosaic_type", "VNG")
    noisy_image, gt_image, _ = load_sidd_scene(scene_dir, demosaic_type=demosaic_type)
    if noisy_image is None or gt_image is None:
        logger.error(f"Failed to load images from {scene_dir}")
        return None

    max_size = getattr(args, "max_size", 1024)
    noisy_image = resize_to_divisible_by_8(noisy_image, max_size=max_size)
    gt_image = resize_to_divisible_by_8(gt_image, max_size=max_size)

    if noisy_image.ndim == 3:
        noisy_image = noisy_image.unsqueeze(0)
    if gt_image.ndim == 3:
        gt_image = gt_image.unsqueeze(0)

    noisy_image = noisy_image.to(sampler.device)
    gt_image = gt_image.to(sampler.device)

    a_norm = calibration_data.get("a", 0.01)
    b_norm = calibration_data.get("b", 0.01)
    poisson_coeff = a_norm
    sigma_r = np.sqrt(max(b_norm, 1e-6))

    common_guidance_params = {
        "s": 1.0,
        "sigma_r": sigma_r,
        "black_level": 0.0,
        "white_level": 1.0,
        "exposure_ratio": 1.0,
        "kappa": args.kappa,
        "guidance_level": args.guidance_level,
    }

    pg_guidance = PoissonGaussianGuidance(
        **common_guidance_params,
        mode=args.pg_mode,
        poisson_coeff=poisson_coeff,
    )

    gaussian_guidance = GaussianGuidance(**common_guidance_params)

    y_e_01 = torch.clamp(convert_range(noisy_image, "[-1,1]", "[0,1]"), 0.0, 1.0)
    sigma_max = DEFAULT_SIGMA_R

    filtered_methods = [m for m in args.run_methods if m != "exposure_scaled"]

    restoration_results = {}

    if "noisy" in filtered_methods:
        restoration_results["noisy"] = noisy_image

    if "gt" in filtered_methods:
        restoration_results["gt"] = gt_image

    restoration_configs = [
        ("gaussian_x0", gaussian_guidance, None, "Gaussian guidance"),
        ("pg_x0", None, pg_guidance, "Poisson-Gaussian guidance"),
    ]

    for method_name, gaussian_guid, pg_guid, description in restoration_configs:
        if method_name in filtered_methods:
            logger.info(f"Running {description}...")
            restored, _ = sampler.posterior_sample(
                y_observed=noisy_image,
                sigma_max=sigma_max,
                num_steps=args.num_steps,
                gaussian_guidance=gaussian_guid,
                pg_guidance=pg_guid,
                y_e=y_e_01,
                no_heun=args.no_heun,
            )
            restoration_results[method_name] = restored

    metrics_results = {}
    for method, restored_tensor in restoration_results.items():
        if method == "gt" or method == "exposure_scaled":
            continue
        try:
            if "gt" in restoration_results:
                metrics_results[method] = compute_metrics_by_method(
                    gt_image, restored_tensor, method, device=sampler.device
                )
        except Exception as e:
            logger.warning(f"Metrics computation failed for {method}: {e}")
            metrics_results[method] = get_nan_metrics_for_method(method)

    # Save results
    scene_output_dir = output_dir / f"scene_{scene_name}"
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    for method, restored_tensor in restoration_results.items():
        if method == "exposure_scaled":
            continue
        if restored_tensor.ndim == 4:
            restored_tensor = restored_tensor.squeeze(0)
        output_path = scene_output_dir / f"restored_{method}.pt"
        torch.save(restored_tensor.cpu(), output_path)

    metrics_path = scene_output_dir / "scene_metrics.json"
    save_json_file(metrics_path, metrics_results)

    try:
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        methods_order = ["noisy", "gt", "gaussian_x0", "pg_x0"]
        methods_to_show = [
            m
            for m in methods_order
            if m in restoration_results and m != "exposure_scaled"
        ]
        num_methods = len(methods_to_show)
        fig = plt.figure(figsize=(4 * num_methods, 12))
        gs = gridspec.GridSpec(3, num_methods, figure=fig, hspace=0.15, wspace=0.1)

        pixel_ranges = {}
        display_images = {}

        for method in methods_to_show:
            img = restoration_results[method]
            if img.ndim == 4:
                img = img.squeeze(0)

            img_01 = torch.clamp(
                convert_range(img.cpu(), "[-1,1]", "[0,1]"), 0.0, 1.0
            ).numpy()
            if img_01.ndim == 3:
                min_val = float(img_01.min())
                max_val = float(img_01.max())
                if img_01.shape[0] == 3:
                    display_img = np.transpose(img_01, (1, 2, 0))
                else:
                    display_img = img_01
            else:
                min_val = float(img_01.min())
                max_val = float(img_01.max())
                display_img = img_01

            pixel_ranges[method] = (min_val, max_val)
            display_images[method] = display_img

        def format_metrics_text(metrics_dict, method):
            if method == "noisy":
                return "Noisy Input"
            if method == "gt":
                return "Ground Truth"

            if method not in metrics_dict:
                return "(No metrics)"

            metrics = metrics_dict[method]
            lines = format_metrics_for_display(metrics)
            return "\n".join(lines) if lines else "(No metrics)"

        method_labels = {
            "noisy": "Noisy",
            "gt": "Ground Truth",
            "gaussian_x0": "Gaussian (x0)",
            "pg_x0": "PG (x0)",
        }

        # Get method colors
        method_colors = {
            "noisy": "blue",
            "gt": "green",
            "gaussian_x0": "orange",
            "pg_x0": "red",
        }

        for idx, method in enumerate(methods_to_show):
            min_val, max_val = pixel_ranges[method]
            display_img = display_images[method]
            method_label = method_labels.get(method, method)
            color = method_colors.get(method, "black")

            ax_range = fig.add_subplot(gs[0, idx])
            rect = Rectangle(
                (0, 0), 1, 1, facecolor="white", alpha=0.7, edgecolor=color, linewidth=2
            )
            ax_range.add_patch(rect)
            ax_range.text(
                0.5,
                0.5,
                f"{method_label}\n[{min_val:.3f}, {max_val:.3f}]",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )
            ax_range.set_xlim(0, 1)
            ax_range.set_ylim(0, 1)
            ax_range.axis("off")

            ax_img = fig.add_subplot(gs[1, idx])
            if display_img.ndim == 2:
                ax_img.imshow(display_img, cmap="gray")
            else:
                ax_img.imshow(display_img)
            ax_img.axis("off")

            ax_metrics = fig.add_subplot(gs[2, idx])
            if method in metrics_results:
                metrics_text = format_metrics_text(metrics_results, method)
                ax_metrics.text(
                    0.5,
                    0.5,
                    metrics_text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                    bbox=dict(
                        boxstyle="round", facecolor="white", alpha=0.7, edgecolor=color
                    ),
                )
            else:
                ax_metrics.text(
                    0.5,
                    0.5,
                    "(No metrics)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontstyle="italic",
                    color="gray",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
                )
            ax_metrics.set_xlim(0, 1)
            ax_metrics.set_ylim(0, 1)
            ax_metrics.axis("off")

        fig.suptitle(f"Scene: {scene_name}", fontsize=14, fontweight="bold", y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        viz_path = scene_output_dir / "scene_comparison.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved visualization to {viz_path}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Scene: {scene_name}")
        logger.info(f"{'='*60}")
        for method in methods_to_show:
            min_val, max_val = pixel_ranges[method]
            logger.info(f"\n{method_labels.get(method, method)}:")
            logger.info(f"  Pixel range: [{min_val:.3f}, {max_val:.3f}]")
            if method in metrics_results:
                metrics = metrics_results[method]
                metric_lines = format_metrics_for_display(metrics)
                if metric_lines:
                    logger.info(f"  Metrics: {' | '.join(metric_lines)}")
                else:
                    logger.info(f"  Metrics: (None)")
            else:
                logger.info(f"  Metrics: (Not computed)")
        logger.info(f"{'='*60}\n")
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")

    return {
        "scene_name": scene_name,
        "metrics": metrics_results,
        "methods": list(restoration_results.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Posterior sampling for SIDD dataset using EDM model with Poisson-Gaussian guidance"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pkl)",
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Path to SIDD scene directory (contains NOISY_RAW_*.MAT and GT_RAW_*.MAT)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sidd_test",
        help="Output directory for results",
    )
    parser.add_argument(
        "--calibration_file",
        type=str,
        required=True,
        help="Path to noise calibration JSON file (Data_noise_calibration.json)",
    )
    parser.add_argument(
        "--run_methods",
        type=str,
        nargs="+",
        default=["noisy", "gt", "gaussian_x0", "pg_x0"],
        help="Methods to run",
    )
    parser.add_argument(
        "--guidance_level",
        type=str,
        default="x0",
        choices=["x0", "score"],
        help="Guidance level",
    )
    parser.add_argument(
        "--pg_mode",
        type=str,
        default="wls",
        choices=["wls", "naive"],
        help="Poisson-Gaussian guidance mode",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=DEFAULT_KAPPA,
        help="Guidance strength",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no_heun",
        action="store_true",
        help="Disable Heun's 2nd order correction",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1024,
        help="Maximum dimension for resizing large SIDD images (0 = no resize, default: 1024). "
        "SIDD images are ~5328×3000 which can cause OOM. Use 512 or 1024 for processing.",
    )
    parser.add_argument(
        "--demosaic_type",
        type=str,
        default="VNG",
        choices=["", "EA", "VNG", "menon2007"],
        help="Demosaicing algorithm type. Options: '' (simple OpenCV), 'EA' (edge-aware), "
        "'VNG' (variable number of gradients, default), 'menon2007' (Menon 2007 algorithm). "
        "Following simple-camera-pipeline options.",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration
    from core.utils.metadata_utils import _load_json_file

    calibration_data = _load_json_file(Path(args.calibration_file))

    # Initialize sampler
    sampler = EDMPosteriorSampler(model_path=args.model_path, device=args.device)

    # Process scene
    scene_dir = Path(args.scene_dir)
    result = process_sidd_scene(
        scene_dir=scene_dir,
        sampler=sampler,
        calibration_data=calibration_data,
        args=args,
        output_dir=output_dir,
    )

    if result is None:
        logger.error("Failed to process scene")
        sys.exit(1)

    logger.info(f"Processing complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
