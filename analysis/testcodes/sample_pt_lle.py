#!/usr/bin/env python3
"""
Low-Light Image Enhancement using EDM Model with Per-Tile Sigma Determination

This script uses the trained EDM model to enhance REAL noisy/low-light test images by:
1. Loading actual noisy test images from the dataset
2. Estimating noise level per tile using various methods (MAD, std, local variance)
3. Optionally optimizing sigma per tile using clean references (SSIM/PSNR)
4. Applying enhancement with determined sigma values
5. Creating comprehensive comparison visualizations and metrics

Key Features:
- Automatic noise level estimation for each tile
- Optional sigma optimization to find best enhancement parameters
- SSIM/PSNR metrics computation against clean references
- Comprehensive metrics reporting and visualization

Usage (with noise estimation only):
    python sample/sample_pt_lle.py \
        --model_path results/edm_pt_training_sony_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/sony/noisy \
        --output_dir results/low_light_enhancement \
        --domain sony \
        --num_examples 3 \
        --num_steps 18 \
        --noise_method local_var

Usage (with brightness-adaptive sigma and CFG - RECOMMENDED for low-light):
    python sample/sample_pt_lle.py \
        --model_path results/edm_pt_training_sony_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/sony/noisy \
        --output_dir results/low_light_enhancement_adaptive \
        --domain sony \
        --num_examples 3 \
        --num_steps 18 \
        --use_adaptive_sigma \
        --base_sigma 0.02 \
        --brightness_scale 10.0 \
        --cfg_scale 2.0

Usage (with sigma optimization):
    python sample/sample_pt_lle.py \
        --model_path results/edm_pt_training_sony_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/sony/noisy \
        --clean_dir dataset/processed/pt_tiles/sony/clean \
        --output_dir results/low_light_enhancement_optimized \
        --domain sony \
        --num_examples 3 \
        --num_steps 18 \
        --optimize_sigma \
        --sigma_range 0.01 0.1 \
        --num_sigma_trials 10 \
        --optimization_metric ssim
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Add project root and EDM to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

# Setup logging
import logging

# Import EDM components
import external.edm.dnnlib
from core.utils.analysis_utils import load_json_safe, save_json_safe
from external.edm.torch_utils import distributed as dist

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NoiseEstimator:
    """Estimate noise level in images using various methods."""

    @staticmethod
    def estimate_std(image: torch.Tensor) -> float:
        """Simple standard deviation estimate."""
        return image.std().item()

    @staticmethod
    def estimate_mad(image: torch.Tensor) -> float:
        """
        Median Absolute Deviation (MAD) based noise estimation.
        More robust to outliers and signal content.
        """
        # Convert to numpy for median operations
        img = image.cpu().numpy()
        median = np.median(img)
        mad = np.median(np.abs(img - median))
        # Scale MAD to estimate standard deviation (assuming Gaussian noise)
        sigma = 1.4826 * mad
        return float(sigma)

    @staticmethod
    def estimate_local_variance(image: torch.Tensor, patch_size: int = 8) -> float:
        """
        Estimate noise from local variance in homogeneous regions.
        Uses the minimum local variance as noise estimate.
        """
        img = image.cpu().numpy()

        if img.ndim == 4:  # (B, C, H, W)
            img = img[0, 0]
        elif img.ndim == 3:  # (C, H, W)
            img = img[0]

        h, w = img.shape
        min_variance = float("inf")

        # Slide window and compute local variances
        for i in range(0, h - patch_size + 1, patch_size // 2):
            for j in range(0, w - patch_size + 1, patch_size // 2):
                patch = img[i : i + patch_size, j : j + patch_size]
                variance = np.var(patch)
                min_variance = min(min_variance, variance)

        return float(np.sqrt(min_variance))

    @staticmethod
    def estimate_noise_comprehensive(image: torch.Tensor) -> Dict[str, float]:
        """
        Compute multiple noise estimates and return all.
        """
        estimates = {
            "std": NoiseEstimator.estimate_std(image),
            "mad": NoiseEstimator.estimate_mad(image),
            "local_var": NoiseEstimator.estimate_local_variance(image),
        }
        # Use local_var as default (best for images with structure)
        estimates["recommended"] = estimates["local_var"]
        return estimates

    @staticmethod
    def brightness_adaptive_sigma(
        image: torch.Tensor,
        base_noise: float,
        base_sigma: float = 0.02,
        brightness_scale: float = 10.0,
        min_sigma: float = 0.002,
        max_sigma: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute brightness-adaptive sigma for enhancement.
        Darker images get larger sigma for stronger enhancement.

        Args:
            image: Input image tensor
            base_noise: Estimated noise level
            base_sigma: Base enhancement strength (default: 0.02)
            brightness_scale: How much darkness affects sigma (default: 10.0)
            min_sigma: Minimum sigma value
            max_sigma: Maximum sigma value

        Returns:
            Dict with adaptive sigma and analysis
        """
        # Convert to [0, 1] range
        img_01 = (image + 1.0) / 2.0
        img_01 = torch.clamp(img_01, 0, 1)

        # Compute brightness statistics
        mean_brightness = img_01.mean().item()
        p50_brightness = torch.quantile(img_01.flatten(), 0.5).item()

        # Darkness factor: 0 for bright, 1 for very dark
        darkness = 1.0 - mean_brightness

        # Adaptive sigma: larger for darker images
        # Formula: sigma = base_sigma * (1 + brightness_scale * darkness)
        adaptive_sigma = base_sigma * (1.0 + brightness_scale * darkness)

        # Clamp to reasonable range
        adaptive_sigma = max(min_sigma, min(max_sigma, adaptive_sigma))

        # Also provide a noise-relative version
        noise_relative_sigma = base_noise * (1.0 + brightness_scale * darkness)
        noise_relative_sigma = max(min_sigma, min(max_sigma, noise_relative_sigma))

        return {
            "mean_brightness": mean_brightness,
            "median_brightness": p50_brightness,
            "darkness_factor": darkness,
            "base_noise": base_noise,
            "adaptive_sigma": adaptive_sigma,
            "noise_relative_sigma": noise_relative_sigma,
            "brightness_scale": brightness_scale,
        }


def compute_metrics(
    clean: np.ndarray, enhanced: np.ndarray, data_range: float = None
) -> Dict[str, float]:
    """
    Compute SSIM and PSNR between clean and enhanced images.

    Args:
        clean: Clean reference image
        enhanced: Enhanced image
        data_range: Data range for metrics (if None, computed from clean)

    Returns:
        Dictionary with metrics
    """
    # Ensure 2D arrays for grayscale
    if clean.ndim == 4:  # (B, C, H, W)
        clean = clean[0, 0]
        enhanced = enhanced[0, 0]
    elif clean.ndim == 3:  # (C, H, W)
        clean = clean[0]
        enhanced = enhanced[0]

    if data_range is None:
        data_range = clean.max() - clean.min()

    # Compute metrics
    ssim_val = ssim(clean, enhanced, data_range=data_range)
    psnr_val = psnr(clean, enhanced, data_range=data_range)

    # Also compute MSE
    mse = np.mean((clean - enhanced) ** 2)

    return {
        "ssim": float(ssim_val),
        "psnr": float(psnr_val),
        "mse": float(mse),
    }


def load_test_tiles(
    metadata_json: Path, domain: str, split: str = "test"
) -> List[Dict[str, Any]]:
    """Load test tile metadata from JSON file."""
    logger.info(f"Loading {split} tiles for {domain} from {metadata_json}")

    metadata = load_json_safe(metadata_json)

    # Filter tiles by domain and split
    tiles = metadata.get("tiles", [])
    filtered_tiles = [
        tile
        for tile in tiles
        if tile.get("domain") == domain and tile.get("split") == split
    ]

    logger.info(f"Found {len(filtered_tiles)} {split} tiles for {domain}")
    return filtered_tiles


def load_image(
    tile_id: str, image_dir: Path, device: torch.device, image_type: str = "image"
) -> torch.Tensor:
    """Load a .pt file (noisy or clean) using core utilities."""
    from core.utils.data_utils import load_tensor

    image_path = image_dir / f"{tile_id}.pt"

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load the image - handle both direct tensor loading and dict formats
    tensor = torch.load(image_path, map_location=device)

    # Handle different tensor formats
    if isinstance(tensor, dict):
        if "noisy" in tensor:
            tensor = tensor["noisy"]
        elif "noisy_norm" in tensor:
            tensor = tensor["noisy_norm"]
        elif "clean" in tensor:
            tensor = tensor["clean"]
        elif "clean_norm" in tensor:
            tensor = tensor["clean_norm"]
        elif "image" in tensor:
            tensor = tensor["image"]
        else:
            raise ValueError(f"Unrecognized dict structure in {image_path}")

    # Ensure float32
    tensor = tensor.float()

    # Ensure CHW format
    if tensor.ndim == 2:  # (H, W)
        tensor = tensor.unsqueeze(0)  # (1, H, W)
    elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3]:  # (H, W, C)
        tensor = tensor.permute(2, 0, 1)  # (C, H, W)

    logger.debug(
        f"Loaded {image_type} image: {tile_id}, shape={tensor.shape}, range=[{tensor.min():.3f}, {tensor.max():.3f}]"
    )

    return tensor


def load_noisy_image(
    tile_id: str, noisy_dir: Path, device: torch.device
) -> torch.Tensor:
    """Load a noisy .pt file."""
    return load_image(tile_id, noisy_dir, device, "noisy")


def load_clean_image(
    tile_id: str, clean_dir: Path, device: torch.device
) -> torch.Tensor:
    """Load a clean .pt file."""
    return load_image(tile_id, clean_dir, device, "clean")


def analyze_image_brightness(image: torch.Tensor) -> Dict[str, float]:
    """Analyze image brightness characteristics."""
    # Convert to [0, 1] range for analysis
    img_01 = (image + 1.0) / 2.0

    # Calculate statistics
    mean_brightness = img_01.mean().item()
    std_brightness = img_01.std().item()
    min_brightness = img_01.min().item()
    max_brightness = img_01.max().item()

    # Calculate percentile brightness
    img_flat = img_01.flatten()
    p10 = torch.quantile(img_flat, 0.1).item()
    p50 = torch.quantile(img_flat, 0.5).item()
    p90 = torch.quantile(img_flat, 0.9).item()

    # Categorize brightness
    if mean_brightness < 0.2:
        brightness_category = "Very Dark"
    elif mean_brightness < 0.4:
        brightness_category = "Dark"
    elif mean_brightness < 0.6:
        brightness_category = "Medium"
    elif mean_brightness < 0.8:
        brightness_category = "Bright"
    else:
        brightness_category = "Very Bright"

    return {
        "mean": mean_brightness,
        "std": std_brightness,
        "min": min_brightness,
        "max": max_brightness,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "category": brightness_category,
    }


class EDMLowLightEnhancer:
    """EDM-based low-light image enhancer."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        sensor_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize the enhancer with trained model and sensor configurations."""
        from core.utils.training_utils import load_edm_model

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if sensor_ranges is None:
            from config.sensor_config import SENSOR_RANGES

            self.sensor_ranges = SENSOR_RANGES.copy()
        else:
            self.sensor_ranges = sensor_ranges

        # Load model using centralized utility
        self.net, _ = load_edm_model(model_path, device=device)
        self.net = self.net.to(self.device)

    def enhance_low_light(
        self,
        low_light_image: torch.Tensor,
        enhancement_sigma: float,
        class_labels: Optional[torch.Tensor] = None,
        num_steps: int = 18,
        rho: float = 7.0,
        cfg_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Enhance low-light image using EDM model with classifier-free guidance.

        Args:
            low_light_image: Low-light input (B, C, H, W) in [-1, 1]
            enhancement_sigma: The enhancement strength (higher = more enhancement)
            class_labels: Domain labels
            num_steps: Number of enhancement steps
            rho: Time step parameter
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)

        Returns:
            Tuple of (enhanced_tensor, results_dict)
        """
        logger.info(
            f"Enhancing low-light image with sigma={enhancement_sigma:.3f}, CFG scale={cfg_scale:.1f}"
        )

        # For enhancement, we want to go from the current image towards a brighter version
        # Instead of denoising (sigma -> 0), we enhance (current -> brighter)

        # Set up enhancement schedule: start from current image, enhance towards brighter
        sigma_min = max(0.002, self.net.sigma_min)
        sigma_max = min(enhancement_sigma, self.net.sigma_max)

        # Time step discretization - go from current image towards enhanced version
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

        logger.info(
            f"Enhancement steps: {t_steps[0]:.3f} -> {t_steps[-1]:.3f} ({len(t_steps)-1} steps)"
        )

        # Start from the low-light image
        x = low_light_image.to(torch.float64).to(self.device)

        # Enhancement loop with CFG
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # Apply classifier-free guidance if enabled
            if cfg_scale > 1.0 and class_labels is not None:
                # Get conditional prediction
                enhanced_cond = self.net(x, t_cur, class_labels).to(torch.float64)

                # Get unconditional prediction (null class labels)
                null_labels = torch.zeros_like(class_labels)
                enhanced_uncond = self.net(x, t_cur, null_labels).to(torch.float64)

                # Apply CFG: enhanced = enhanced_uncond + cfg_scale * (enhanced_cond - enhanced_uncond)
                enhanced = enhanced_uncond + cfg_scale * (
                    enhanced_cond - enhanced_uncond
                )
            else:
                # Standard prediction without CFG
                enhanced = self.net(x, t_cur, class_labels).to(torch.float64)

            # EDM denoising/enhancement: move x towards the model's prediction
            # The negative (t_next - t_cur) naturally handles the direction
            d_cur = (x - enhanced) / t_cur

            # Apply enhancement step
            x_next = x + (t_next - t_cur) * d_cur

            # Heun's 2nd order correction
            if i < num_steps - 1:
                # Apply CFG for correction step too
                if cfg_scale > 1.0 and class_labels is not None:
                    enhanced_next_cond = self.net(x_next, t_next, class_labels).to(
                        torch.float64
                    )
                    enhanced_next_uncond = self.net(x_next, t_next, null_labels).to(
                        torch.float64
                    )
                    enhanced_next = enhanced_next_uncond + cfg_scale * (
                        enhanced_next_cond - enhanced_next_uncond
                    )
                else:
                    enhanced_next = self.net(x_next, t_next, class_labels).to(
                        torch.float64
                    )

                d_prime = (x_next - enhanced_next) / t_next
                x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

            x = x_next

        enhanced_output = x

        # Clamp to valid range and ensure enhancement moves towards positive values
        enhanced_output = torch.clamp(enhanced_output, -1, 1)

        logger.info(
            f"✓ Enhancement completed: range [{enhanced_output.min():.3f}, {enhanced_output.max():.3f}]"
        )

        results = {
            "enhanced": enhanced_output,
            "original": low_light_image,
            "enhancement_sigma": enhancement_sigma,
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
        }

        return enhanced_output, results

    def optimize_sigma(
        self,
        low_light_image: torch.Tensor,
        clean_image: torch.Tensor,
        class_labels: Optional[torch.Tensor],
        sigma_range: Tuple[float, float],
        num_trials: int = 10,
        num_steps: int = 18,
        metric: str = "ssim",
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Find optimal sigma by trying multiple values and maximizing SSIM (or minimizing MSE).

        Args:
            low_light_image: Low-light input
            clean_image: Clean reference
            class_labels: Domain labels
            sigma_range: (min_sigma, max_sigma) to search
            num_trials: Number of sigma values to try
            num_steps: Enhancement steps
            metric: 'ssim' (maximize) or 'mse' (minimize) or 'psnr' (maximize)

        Returns:
            Tuple of (best_sigma, results_dict)
        """
        logger.info(
            f"Optimizing sigma in range [{sigma_range[0]:.6f}, {sigma_range[1]:.6f}] with {num_trials} trials"
        )

        # Generate sigma values to try (log-spaced)
        sigma_values = np.logspace(
            np.log10(sigma_range[0]), np.log10(sigma_range[1]), num=num_trials
        )

        clean_np = clean_image.cpu().numpy()
        best_sigma = sigma_values[0]
        best_metric_value = (
            float("-inf") if metric in ["ssim", "psnr"] else float("inf")
        )
        all_results = []

        for sigma in sigma_values:
            # Enhance with this sigma
            enhanced, _ = self.enhance_low_light(
                low_light_image,
                enhancement_sigma=sigma,
                class_labels=class_labels,
                num_steps=num_steps,
            )

            # Compute metrics
            enhanced_np = enhanced.cpu().numpy()
            metrics = compute_metrics(clean_np, enhanced_np)

            # Track results
            all_results.append(
                {
                    "sigma": float(sigma),
                    "ssim": metrics["ssim"],
                    "psnr": metrics["psnr"],
                    "mse": metrics["mse"],
                }
            )

            # Update best
            metric_value = metrics[metric]
            is_better = (
                metric_value > best_metric_value
                if metric in ["ssim", "psnr"]
                else metric_value < best_metric_value
            )

            if is_better:
                best_sigma = sigma
                best_metric_value = metric_value
                logger.debug(f"  New best: σ={sigma:.6f}, {metric}={metric_value:.4f}")

        logger.info(
            f"✓ Best sigma: {best_sigma:.6f} ({metric}={best_metric_value:.4f})"
        )

        return best_sigma, {
            "best_sigma": float(best_sigma),
            "best_metric": metric,
            "best_metric_value": float(best_metric_value),
            "all_trials": all_results,
        }

    def denormalize_to_physical(self, tensor: torch.Tensor, sensor: str) -> np.ndarray:
        """Convert tensor from [-1,1] model space to physical units."""
        from core.normalization import denormalize_to_physical

        # Use centralized utility with sensor range from instance
        sensor_range = (
            self.sensor_ranges.get(sensor) if hasattr(self, "sensor_ranges") else None
        )
        if sensor_range:
            return denormalize_to_physical(tensor, range_dict=sensor_range)
        elif sensor:
            return denormalize_to_physical(tensor, sensor=sensor)
        else:
            # Fallback to default sensor range
            return denormalize_to_physical(tensor, sensor="sony")


def create_enhancement_comparison(
    noisy_image: torch.Tensor,
    enhancement_results: Dict[float, torch.Tensor],
    domain: str,
    tile_id: str,
    save_path: Path,
):
    """Create comprehensive enhancement comparison visualization with physical units."""
    logger.info("Creating low-light enhancement comparison visualization...")

    # Sort sigma values
    sigmas = sorted(enhancement_results.keys())
    n_sigmas = len(sigmas)

    # Create figure with one row: noisy input and enhanced outputs
    fig, axes = plt.subplots(1, n_sigmas + 1, figsize=(3 * (n_sigmas + 1), 3))

    # Denormalize to physical units using centralized utility
    from core.normalization import denormalize_to_physical as denorm_util

    def prepare_for_display(phys_array, scale_range=None):
        """
        Prepare physical array for display using percentile normalization.

        Args:
            phys_array: Physical array to display
            scale_range: Optional tuple (min_scale, max_scale) to use for normalization.
                        If None, compute from this array's percentiles.

        Returns: (display_array, min_val, max_val, p1_val, p99_val, scale_min, scale_max)
        """
        # Handle different array shapes
        if phys_array.ndim == 4:  # (B, C, H, W)
            if phys_array.shape[1] == 3:  # RGB
                img = np.transpose(phys_array[0], (1, 2, 0))
            else:  # Grayscale
                img = phys_array[0, 0]
        elif phys_array.ndim == 3:  # (C, H, W)
            if phys_array.shape[0] == 3:  # RGB
                img = np.transpose(phys_array, (1, 2, 0))
            else:
                img = phys_array[0]
        elif phys_array.ndim == 2:  # (H, W)
            img = phys_array
        else:
            img = phys_array

        # Get statistics
        min_val = float(np.min(img))
        max_val = float(np.max(img))

        # Compute or use provided scale range
        valid_mask = np.isfinite(img)
        if scale_range is None:
            # Compute scale from this image's percentiles
            if np.any(valid_mask):
                p1, p99 = np.percentile(img[valid_mask], [1, 99])
                scale_min, scale_max = p1, p99
            else:
                p1, p99 = min_val, max_val
                scale_min, scale_max = min_val, max_val
        else:
            # Use provided scale
            scale_min, scale_max = scale_range
            if np.any(valid_mask):
                p1, p99 = np.percentile(img[valid_mask], [1, 99])
            else:
                p1, p99 = min_val, max_val

        # Apply normalization using the scale
        img_clipped = np.clip(img, scale_min, scale_max)
        img_norm = (img_clipped - scale_min) / (scale_max - scale_min + 1e-8)

        return img_norm, min_val, max_val, p1, p99, scale_min, scale_max

    # Get sensor-specific unit label
    domain_units = {
        "sony": "ADU",
        "fuji": "ADU",
    }
    unit_label = domain_units.get(domain, "units")

    # Process noisy input first to get the reference scale
    noisy_phys = denorm_util(noisy_image, domain=domain)
    (
        noisy_display,
        noisy_min,
        noisy_max,
        noisy_p1,
        noisy_p99,
        scale_min,
        scale_max,
    ) = prepare_for_display(noisy_phys)

    # Store the reference scale from noisy input
    reference_scale = (scale_min, scale_max)

    axes[0].imshow(noisy_display, cmap="gray" if noisy_image.shape[1] == 1 else None)
    axes[0].set_title(
        f"Noisy Input (Reference)\n"
        f"Range: [{noisy_min:.1f}, {noisy_max:.1f}] {unit_label}\n"
        f"Display Scale: [{scale_min:.1f}, {scale_max:.1f}]",
        fontsize=8,
    )
    axes[0].axis("off")

    # Process and display enhanced results using the SAME scale as noisy input
    for i, sigma in enumerate(sigmas):
        enhanced_tensor = enhancement_results[sigma]
        enhanced_phys = denorm_util(enhanced_tensor, domain=domain)
        enhanced_display, enh_min, enh_max, enh_p1, enh_p99, _, _ = prepare_for_display(
            enhanced_phys, scale_range=reference_scale  # Use same scale as noisy input
        )

        axes[i + 1].imshow(
            enhanced_display, cmap="gray" if enhanced_display.ndim == 2 else None
        )
        axes[i + 1].set_title(
            f"Enhanced σ={sigma:.6f}\n"
            f"Range: [{enh_min:.1f}, {enh_max:.1f}] {unit_label}\n"
            f"Display Scale: [{scale_min:.1f}, {scale_max:.1f}]",
            fontsize=8,
        )
        axes[i + 1].axis("off")

    # Main title
    plt.suptitle(
        f"Low-Light Enhancement (Physical Units: {unit_label}) - {tile_id}\n"
        f"Domain: {domain} | Enhancement σ ∈ [{min(sigmas):.6f}, {max(sigmas):.3f}]\n"
        f"All images shown with same display scale [{scale_min:.1f}, {scale_max:.1f}] {unit_label} for direct comparison",
        fontsize=11,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(
        f"✓ Enhancement comparison saved: {save_path} (physical units with common scale [{scale_min:.1f}, {scale_max:.1f}] {unit_label})"
    )


def main():
    """Main function for low-light image enhancement."""
    parser = argparse.ArgumentParser(
        description="Low-light image enhancement using EDM model with per-tile sigma determination"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pkl)",
    )

    # Data arguments
    parser.add_argument(
        "--metadata_json",
        type=str,
        required=True,
        help="Path to metadata JSON file with test split",
    )
    parser.add_argument(
        "--noisy_dir",
        type=str,
        required=True,
        help="Directory containing noisy .pt files",
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        default=None,
        help="Directory containing clean reference .pt files (optional, for optimization)",
    )

    # Enhancement arguments
    parser.add_argument(
        "--num_steps", type=int, default=18, help="Number of enhancement steps"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="sony",
        choices=["sony", "fuji"],
        help="Sensor name (sony or fuji)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger guidance)",
    )

    # Noise estimation arguments
    parser.add_argument(
        "--noise_method",
        type=str,
        default="local_var",
        choices=["std", "mad", "local_var"],
        help="Noise estimation method (local_var is best for images with structure)",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=1.0,
        help="Scale factor for estimated noise (default: 1.0)",
    )

    # Brightness-adaptive sigma arguments
    parser.add_argument(
        "--use_adaptive_sigma",
        action="store_true",
        help="Use brightness-adaptive sigma scaling (recommended for low-light)",
    )
    parser.add_argument(
        "--base_sigma",
        type=float,
        default=0.02,
        help="Base sigma for adaptive scaling (default: 0.02)",
    )
    parser.add_argument(
        "--brightness_scale",
        type=float,
        default=10.0,
        help="How much darkness affects sigma (default: 10.0)",
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help="Minimum sigma value (default: 0.002)",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=0.5,
        help="Maximum sigma value (default: 0.5)",
    )

    # Sigma optimization arguments
    parser.add_argument(
        "--optimize_sigma",
        action="store_true",
        help="Search for optimal sigma for each tile (requires --clean_dir)",
    )
    parser.add_argument(
        "--sigma_range",
        type=float,
        nargs=2,
        default=[0.0001, 0.01],
        help="Min and max sigma for optimization search",
    )
    parser.add_argument(
        "--num_sigma_trials",
        type=int,
        default=10,
        help="Number of sigma values to try during optimization",
    )
    parser.add_argument(
        "--optimization_metric",
        type=str,
        default="ssim",
        choices=["ssim", "psnr", "mse"],
        help="Metric to optimize (default: ssim)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="results/low_light_enhancement"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=3,
        help="Number of example images to enhance",
    )

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validate arguments
    if args.optimize_sigma and args.clean_dir is None:
        parser.error("--optimize_sigma requires --clean_dir to be specified")

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("LOW-LIGHT IMAGE ENHANCEMENT WITH PER-TILE SIGMA DETERMINATION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Metadata: {args.metadata_json}")
    logger.info(f"Noisy dir: {args.noisy_dir}")
    logger.info(f"Clean dir: {args.clean_dir if args.clean_dir else 'Not provided'}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Enhancement steps: {args.num_steps}")
    logger.info(f"Number of examples: {args.num_examples}")
    logger.info(f"Noise method: {args.noise_method}")
    logger.info(f"CFG scale: {args.cfg_scale}")
    logger.info(f"Brightness-adaptive sigma: {args.use_adaptive_sigma}")
    if args.use_adaptive_sigma:
        logger.info(f"  Base sigma: {args.base_sigma}")
        logger.info(f"  Brightness scale: {args.brightness_scale}")
        logger.info(f"  Sigma range: [{args.sigma_min}, {args.sigma_max}]")
    logger.info(f"Optimize sigma: {args.optimize_sigma}")
    if args.optimize_sigma:
        logger.info(f"  Sigma range: [{args.sigma_range[0]}, {args.sigma_range[1]}]")
        logger.info(f"  Trials: {args.num_sigma_trials}")
        logger.info(f"  Metric: {args.optimization_metric}")
    logger.info("=" * 80)

    # Initialize enhancer
    enhancer = EDMLowLightEnhancer(
        model_path=args.model_path,
        device=args.device,
    )

    # Load test tiles
    test_tiles = load_test_tiles(Path(args.metadata_json), args.domain, split="test")

    if len(test_tiles) == 0:
        logger.error(f"No test tiles found for domain {args.domain}")
        return

    # Filter tiles to only those that exist in the required directories
    available_tiles = []
    for tile_info in test_tiles:
        tile_id = tile_info["tile_id"]
        noisy_path = Path(args.noisy_dir) / f"{tile_id}.pt"

        # Check if noisy file exists
        if not noisy_path.exists():
            continue

        # If optimization is enabled, also check for clean file
        if args.optimize_sigma:
            clean_path = Path(args.clean_dir) / f"{tile_id}.pt"
            if not clean_path.exists():
                continue

        available_tiles.append(tile_info)

    if args.optimize_sigma:
        logger.info(
            f"Found {len(available_tiles)} tiles with both noisy and clean files"
        )
    else:
        logger.info(f"Found {len(available_tiles)} tiles with noisy files")

    if len(available_tiles) == 0:
        logger.error(f"No suitable tiles found in {args.noisy_dir}")
        return

    # Randomly select from available tiles
    rng = np.random.RandomState(args.seed)
    selected_indices = rng.choice(
        len(available_tiles),
        size=min(args.num_examples, len(available_tiles)),
        replace=False,
    )
    selected_tiles = [available_tiles[i] for i in selected_indices]

    logger.info(f"Selected {len(selected_tiles)} test tiles for enhancement")

    # Create domain labels (sensor-specific: sony or fuji) using centralized utility
    from core.utils.training_utils import create_domain_labels

    class_labels = create_domain_labels(
        domain=args.domain,
        batch_size=1,
        label_dim=enhancer.net.label_dim,
        device=enhancer.device,
    )

    # Process each selected tile with per-tile sigma determination
    all_results = []

    for idx, tile_info in enumerate(selected_tiles):
        tile_id = tile_info["tile_id"]
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Processing example {idx+1}/{len(selected_tiles)}: {tile_id}")
        logger.info(f"{'='*60}")

        try:
            # Load noisy image
            noisy_image = load_noisy_image(
                tile_id, Path(args.noisy_dir), enhancer.device
            )
            if noisy_image.ndim == 3:
                noisy_image = noisy_image.unsqueeze(0)
            noisy_image = noisy_image.to(torch.float32)

            # Denormalize to physical units for logging
            noisy_phys = enhancer.denormalize_to_physical(noisy_image, args.domain)
            domain_units = {
                "sony": "ADU",
                "fuji": "ADU",
            }
            unit_label = domain_units.get(args.domain, "units")

            # Analyze noisy image brightness
            noisy_brightness = analyze_image_brightness(noisy_image)
            logger.info(
                f"  Brightness: {noisy_brightness['category']} (mean={noisy_brightness['mean']:.3f} normalized)"
            )
            logger.info(
                f"  Normalized range: [{noisy_image.min():.4f}, {noisy_image.max():.4f}], std={noisy_image.std():.4f}"
            )
            logger.info(
                f"  Physical range: [{noisy_phys.min():.1f}, {noisy_phys.max():.1f}] {unit_label}"
            )

            # Estimate noise level
            noise_estimates = NoiseEstimator.estimate_noise_comprehensive(noisy_image)
            estimated_noise = noise_estimates[args.noise_method] * args.noise_scale

            logger.info(f"  Noise estimates: {noise_estimates}")
            logger.info(f"  Using {args.noise_method}: noise_σ={estimated_noise:.6f}")

            # Apply brightness-adaptive scaling if enabled
            if args.use_adaptive_sigma:
                adaptive_info = NoiseEstimator.brightness_adaptive_sigma(
                    noisy_image,
                    base_noise=estimated_noise,
                    base_sigma=args.base_sigma,
                    brightness_scale=args.brightness_scale,
                    min_sigma=args.sigma_min,
                    max_sigma=args.sigma_max,
                )
                estimated_sigma = adaptive_info["adaptive_sigma"]

                logger.info(f"  Brightness-adaptive scaling:")
                logger.info(
                    f"    Mean brightness: {adaptive_info['mean_brightness']:.6f}"
                )
                logger.info(
                    f"    Darkness factor: {adaptive_info['darkness_factor']:.6f}"
                )
                logger.info(f"    Adaptive σ: {adaptive_info['adaptive_sigma']:.6f}")
                logger.info(
                    f"    Noise-relative σ: {adaptive_info['noise_relative_sigma']:.6f}"
                )
            else:
                estimated_sigma = estimated_noise
                adaptive_info = None

            # Determine sigma to use
            enhancement_results = {}
            opt_results = None

            if args.optimize_sigma:
                # Load clean reference
                clean_image = load_clean_image(
                    tile_id, Path(args.clean_dir), enhancer.device
                )
                if clean_image.ndim == 3:
                    clean_image = clean_image.unsqueeze(0)
                clean_image = clean_image.to(torch.float32)

                logger.info("  Optimizing sigma...")
                best_sigma, opt_results = enhancer.optimize_sigma(
                    noisy_image,
                    clean_image,
                    class_labels,
                    sigma_range=tuple(args.sigma_range),
                    num_trials=args.num_sigma_trials,
                    num_steps=args.num_steps,
                    metric=args.optimization_metric,
                )
                sigma_used = best_sigma

                # Enhance with best sigma
                logger.info(f"  Enhancing with optimal σ={sigma_used:.6f}")
                enhanced, _ = enhancer.enhance_low_light(
                    noisy_image,
                    enhancement_sigma=sigma_used,
                    class_labels=class_labels,
                    num_steps=args.num_steps,
                    cfg_scale=args.cfg_scale,
                )
                enhancement_results[sigma_used] = enhanced

                # Compute final metrics
                enhanced_np = enhanced.cpu().numpy()
                clean_np = clean_image.cpu().numpy()
                metrics = compute_metrics(clean_np, enhanced_np)
                logger.info(
                    f"  Final metrics: SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}dB"
                )
            else:
                # Use estimated sigma
                sigma_used = estimated_sigma
                logger.info(f"  Enhancing with estimated σ={sigma_used:.6f}")
                enhanced, _ = enhancer.enhance_low_light(
                    noisy_image,
                    enhancement_sigma=sigma_used,
                    class_labels=class_labels,
                    num_steps=args.num_steps,
                    cfg_scale=args.cfg_scale,
                )
                enhancement_results[sigma_used] = enhanced
                metrics = None

            # Save results
            sample_dir = output_dir / f"example_{idx:02d}_{tile_id}"
            sample_dir.mkdir(exist_ok=True)

            # Save tensors
            torch.save(noisy_image.cpu(), sample_dir / "noisy.pt")

            # Save enhanced results
            for sigma, enhanced in enhancement_results.items():
                torch.save(
                    enhanced.cpu(), sample_dir / f"enhanced_sigma_{sigma:.7f}.pt"
                )

            # Save clean if available
            if args.optimize_sigma:
                torch.save(clean_image.cpu(), sample_dir / "clean.pt")

            # Save metrics and parameters
            result_info = {
                "tile_id": tile_id,
                "noise_estimates": noise_estimates,
                "estimated_noise": float(estimated_noise),
                "estimated_sigma": float(estimated_sigma),
                "sigma_used": float(sigma_used),
                "noise_method": args.noise_method,
                "brightness_analysis": noisy_brightness,
                "use_adaptive_sigma": args.use_adaptive_sigma,
            }

            if adaptive_info is not None:
                result_info["adaptive_sigma_info"] = adaptive_info

            if metrics is not None:
                result_info["metrics"] = metrics

            if opt_results is not None:
                result_info["optimization_results"] = opt_results

            save_json_safe(result_info, sample_dir / "results.json")

            all_results.append(result_info)

            # Create visualization
            comparison_path = sample_dir / "enhancement_comparison.png"
            create_enhancement_comparison(
                noisy_image=noisy_image,
                enhancement_results=enhancement_results,
                domain=args.domain,
                tile_id=tile_id,
                save_path=comparison_path,
            )

            logger.info(f"✓ Saved to {sample_dir}")

        except Exception as e:
            logger.error(f"Failed to process tile {tile_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save summary
    summary = {
        "domain": args.domain,
        "num_samples": len(all_results),
        "noise_method": args.noise_method,
        "use_adaptive_sigma": args.use_adaptive_sigma,
        "optimize_sigma": args.optimize_sigma,
        "results": all_results,
    }

    if args.use_adaptive_sigma:
        summary["adaptive_sigma_params"] = {
            "base_sigma": args.base_sigma,
            "brightness_scale": args.brightness_scale,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
        }

    if args.optimize_sigma and len(all_results) > 0:
        summary["aggregate_metrics"] = {
            "mean_ssim": np.mean(
                [r["metrics"]["ssim"] for r in all_results if "metrics" in r]
            ),
            "mean_psnr": np.mean(
                [r["metrics"]["psnr"] for r in all_results if "metrics" in r]
            ),
            "mean_mse": np.mean(
                [r["metrics"]["mse"] for r in all_results if "metrics" in r]
            ),
            "std_ssim": np.std(
                [r["metrics"]["ssim"] for r in all_results if "metrics" in r]
            ),
            "std_psnr": np.std(
                [r["metrics"]["psnr"] for r in all_results if "metrics" in r]
            ),
        }

    logger.info("\n" + "=" * 80)
    logger.info("🎉 LOW-LIGHT ENHANCEMENT COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"📁 All results saved to: {output_dir}")
    logger.info(
        f"📊 Processed {len(all_results)} tiles with per-tile sigma determination"
    )
    if args.optimize_sigma and len(all_results) > 0:
        logger.info(
            f"📈 Mean SSIM: {summary['aggregate_metrics']['mean_ssim']:.4f} ± {summary['aggregate_metrics']['std_ssim']:.4f}"
        )
        logger.info(
            f"📈 Mean PSNR: {summary['aggregate_metrics']['mean_psnr']:.2f} ± {summary['aggregate_metrics']['std_psnr']:.2f} dB"
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
