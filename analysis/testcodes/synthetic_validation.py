#!/usr/bin/env python3
"""
Synthetic Validation: Demonstrate PG > Gaussian in Shot-Noise-Dominated Regime

This script creates synthetic test cases at moderate exposure ratios (0.1, 0.2, 0.3)
or direct Poisson contribution percentages (5%, 10%, 15%, 20%, 30%)
where shot noise is appreciable and heteroscedastic guidance should outperform
homoscedastic guidance.

Key features:
- Regime Classifier: Automatically classifies noise regime (read-noise vs shot-noise dominated)
- Multi-Frame Averaging (Option D): Average N frames to reduce read noise by sqrt(N) while
  preserving shot noise characteristics, naturally shifting toward shot-noise dominated regime
- Regime Sweep Visualization: Key figure showing "when physics matters" - PSNR gain vs
  Poisson-to-Gaussian variance ratio

Outputs are saved in the same format as test_guidance_comparison_unified.sh for easy comparison:
- scene_metrics.json: Same format as real scene evaluations
- scene_comparison.png: Visual comparison (when using scenes)
- regime_sweep_analysis.png: Regime-aware analysis visualization
- stitched_*.pt files: Restored images in same format

Runtime: ~30-60 minutes depending on GPU and number of test images
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.utils.analysis_utils import load_json_safe, save_json_safe
from core.utils.data_utils import (
    load_metadata_json,
    load_noise_calibration,
    load_tile_and_metadata,
    select_tiles_for_processing,
)
from core.utils.file_utils import (
    count_regimes,
    find_latest_model,
    find_metadata_json,
    find_scene_directories,
    find_split_file,
    parse_filename_from_split_line,
)
from core.utils.sensor_utils import (
    READ_NOISE_THRESHOLD,
    SHOT_NOISE_THRESHOLD,
    NoiseRegimeClassifier,
    convert_calibration_to_poisson_coeff,
    convert_calibration_to_sigma_r,
    denormalize_to_physical,
    load_sensor_calibration_from_metadata,
)
from core.utils.tiles_utils import save_scene_metrics_json
from sample.sample_noisy_pt_lle_PGguidance import EDMPosteriorSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_regime_sweep_visualization(
    all_results: List[Dict],
    poisson_coeff: float,
    sigma_r: float,
    output_dir: Path,
    mode: str,
) -> None:
    """
    Create regime sweep figure showing PSNR gain vs signal/σ_r² ratio.

    This is the key figure showing "when physics matters" - demonstrating
    that PG guidance outperforms Gaussian in shot-noise dominated regimes.
    """
    if not all_results:
        return

    # Extract variance ratios and regime info from results
    # (Results already include variance_ratio computed with effective_sigma_r for multi-frame averaging)
    variance_ratios = []
    psnr_gains = []
    regimes = []
    poisson_fractions = []

    for result in all_results:
        # Use variance_ratio from result (already accounts for effective_sigma_r)
        variance_ratio = result.get("variance_ratio", 0.0)
        if variance_ratio <= 0:
            continue

        variance_ratios.append(variance_ratio)
        psnr_gains.append(result["psnr_gain"])
        regimes.append(result.get("regime", "unknown"))
        poisson_fractions.append(result.get("poisson_fraction", 0.0))

    if len(variance_ratios) == 0:
        logger.warning("No valid results for regime sweep visualization")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: PSNR gain vs variance ratio (main result)
    ax = axes[0, 0]
    scatter = ax.scatter(
        variance_ratios,
        psnr_gains,
        c=[poisson_fractions],
        cmap="viridis",
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="No difference")
    ax.axvline(
        x=READ_NOISE_THRESHOLD,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label="Read-noise threshold",
    )
    ax.axvline(
        x=SHOT_NOISE_THRESHOLD,
        color="green",
        linestyle=":",
        linewidth=1.5,
        label="Shot-noise threshold",
    )
    ax.set_xlabel("Poisson-to-Gaussian Variance Ratio (a·μ / σ_r²)", fontsize=11)
    ax.set_ylabel("PSNR Gain (PG - Gaussian) [dB]", fontsize=11)
    ax.set_title(
        "Regime Sweep: When Does Physics Matter?", fontsize=12, fontweight="bold"
    )
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label="Poisson Fraction")

    # Plot 2: PSNR gain by regime
    ax = axes[0, 1]
    regime_gains = {
        "read_noise_dominated": [],
        "transitional": [],
        "shot_noise_dominated": [],
    }
    for i, regime in enumerate(regimes):
        regime_gains[regime].append(psnr_gains[i])

    positions = [0, 1, 2]
    labels = ["Read-Noise\nDominated", "Transitional", "Shot-Noise\nDominated"]
    data_to_plot = [
        regime_gains["read_noise_dominated"],
        regime_gains["transitional"],
        regime_gains["shot_noise_dominated"],
    ]

    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        labels=labels,
        patch_artist=True,
        showmeans=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax.set_ylabel("PSNR Gain (PG - Gaussian) [dB]", fontsize=11)
    ax.set_title("PSNR Gain by Noise Regime", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 3: Variance ratio distribution histogram
    ax = axes[1, 0]
    ax.hist(variance_ratios, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Poisson-to-Gaussian Variance Ratio (a·μ / σ_r²)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Variance Ratios", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Poisson fraction vs PSNR gain
    ax = axes[1, 1]
    scatter = ax.scatter(
        poisson_fractions,
        psnr_gains,
        c=variance_ratios,
        cmap="plasma",
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Poisson Contribution Fraction", fontsize=11)
    ax.set_ylabel("PSNR Gain (PG - Gaussian) [dB]", fontsize=11)
    ax.set_title("PSNR Gain vs Poisson Fraction", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Variance Ratio")

    plt.suptitle(
        "Regime-Aware Analysis: Understanding When Physics Matters",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    fig.savefig(
        output_dir / "regime_sweep_analysis.png", dpi=DEFAULT_DPI, bbox_inches="tight"
    )
    plt.close(fig)


def create_synthetic_summary_visualization(
    results: List[Dict],
    test_value: float,
    mode: str,
    scene_dir: Path,
    sensor: str,
) -> None:
    """Create a summary visualization showing PG vs Gaussian comparison."""
    if not results:
        return

    # Create a simple comparison showing aggregate metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract metrics
    pg_psnrs = [r["pg_psnr"] for r in results]
    gauss_psnrs = [r["gaussian_psnr"] for r in results]
    psnr_gains = [r["psnr_gain"] for r in results]

    # Plot PSNR comparison
    ax = axes[0]
    x_pos = np.arange(len(results))
    width = 0.35
    ax.bar(x_pos - width / 2, pg_psnrs, width, label="PG", color="green", alpha=0.7)
    ax.bar(
        x_pos + width / 2,
        gauss_psnrs,
        width,
        label="Gaussian",
        color="orange",
        alpha=0.7,
    )
    ax.set_xlabel("Image Index")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"PSNR Comparison (n={len(results)})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot PSNR gain distribution
    ax = axes[1]
    ax.hist(psnr_gains, bins=min(20, len(results)), edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No difference")
    ax.set_xlabel("PSNR Gain (dB)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"PSNR Gain Distribution\nMean: {np.mean(psnr_gains):.3f} dB")
    ax.legend()
    ax.grid(True, alpha=0.3)

    test_label = (
        f"Poisson {test_value*100:.1f}%"
        if mode == "poisson_percentage"
        else f"Ratio {test_value}"
    )
    plt.suptitle(
        f"Synthetic Validation - {test_label} (n={len(results)} images)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(
        scene_dir / "synthetic_comparison.png", dpi=DEFAULT_DPI, bbox_inches="tight"
    )
    plt.close(fig)


def _load_test_split_tile_ids_simple(split_file: Path, sensor: str) -> Set[str]:
    """Fallback: Generate tile IDs from split file patterns when metadata unavailable."""
    tile_ids = set()
    num_tiles = 216 if sensor == "sony" else 96 if sensor == "fuji" else 100

    with open(split_file, "r") as f:
        for line in f:
            parsed = parse_filename_from_split_line(line)
            if parsed:
                scene_id, frame_id, exposure_str = parsed
                for tile_idx in range(num_tiles):
                    tile_ids.add(
                        f"{sensor}_{scene_id}_{frame_id}_{exposure_str}_tile_{tile_idx:04d}"
                    )

    return tile_ids


def _load_split_scene_keys(split_file: Path) -> Set[Tuple[str, str, str]]:
    """Extract (scene_id, frame_id, exposure_str) tuples from split file for scene matching."""
    scene_keys = set()

    with open(split_file, "r") as f:
        for line in f:
            parsed = parse_filename_from_split_line(line)
            if parsed:
                scene_keys.add(parsed)

    return scene_keys


def filter_images_by_split(
    image_files: List[Path],
    split_file: Optional[Path],
    sensor: str,
    metadata_json: Optional[Path] = None,
    use_scenes: bool = False,
) -> List[Path]:
    """Filter image files to only include those in validation split."""
    if split_file is None or not split_file.exists():
        return image_files

    filtered = []
    seen_scene_ids = set()

    if use_scenes:
        # For scenes, match based on (scene_id, frame_id, exposure_str) tuples
        split_scene_keys = _load_split_scene_keys(split_file)

        for img_file in image_files:
            filename = img_file.stem
            if filename == "stitched_long":
                scene_dir = img_file.parent.name
                if scene_dir.startswith("scene_"):
                    # Parse: scene_10019_fuji_00_0.033s -> (10019, 00, 0.033s)
                    parts = scene_dir.split("_")
                    if len(parts) >= 5:
                        scene_id = parts[1]
                        if scene_id in seen_scene_ids:
                            continue
                        frame_id, exposure_str = parts[3], parts[4]
                        scene_key = (scene_id, frame_id, exposure_str)
                        if scene_key in split_scene_keys:
                            filtered.append(img_file)
                            seen_scene_ids.add(scene_id)
    else:
        # For tiles, use tile ID matching
        test_tile_ids = set()
        if metadata_json and metadata_json.exists():
            try:
                tile_id_list = load_tile_ids_from_split_file(
                    split_file, sensor, metadata_json, data_type="long"
                )
                test_tile_ids = set(tile_id_list)
            except Exception:
                test_tile_ids = _load_test_split_tile_ids_simple(split_file, sensor)
        else:
            test_tile_ids = _load_test_split_tile_ids_simple(split_file, sensor)

        for img_file in image_files:
            filename = img_file.stem
            if filename in test_tile_ids:
                # Extract scene_id from tile_id format: {sensor}_{scene_id}_{frame_id}_{exposure_str}_tile_{XXXX}
                parts = filename.split("_")
                if len(parts) >= 2:
                    scene_id = parts[1]
                    if scene_id not in seen_scene_ids:
                        filtered.append(img_file)
                        seen_scene_ids.add(scene_id)

    return filtered


def compute_target_signal_for_poisson_percentage(
    target_poisson_fraction: float,
    poisson_coeff: float,
    sigma_r_squared: float,
) -> float:
    """Compute signal level needed to achieve target Poisson contribution percentage."""
    return (target_poisson_fraction / (1 - target_poisson_fraction)) * (
        sigma_r_squared / poisson_coeff
    )


def average_multiple_frames(
    noisy_frames: List[torch.Tensor],
) -> torch.Tensor:
    """
    Average multiple short exposures to reduce read noise while preserving shot noise.

    Key physics:
    - Read noise variance reduces by factor N (std by sqrt(N)) when averaging N frames
    - Shot noise variance remains signal-dependent (doesn't reduce by averaging)
    - This naturally shifts the regime toward shot-noise dominance

    Args:
        noisy_frames: List of noisy frames, each in normalized [-1, 1] format

    Returns:
        Averaged frame in normalized [-1, 1] format
    """
    stacked = torch.stack(noisy_frames, dim=0)
    averaged = torch.mean(stacked, dim=0)
    return averaged


def add_synthetic_poisson_gaussian_noise(
    clean_image: torch.Tensor,
    exposure_ratio: float,
    poisson_coeff: float,
    sigma_r: float,
    sensor_range: float,
    black_level: float,
    target_signal: Optional[float] = None,
    num_frames: int = 1,
) -> Tuple[torch.Tensor, float, float, float]:
    """
    Add synthetic Poisson-Gaussian noise to a clean image.

    Physics: variance = poisson_coeff * signal + σ_r²
    where signal is in sensor range units [0, s] (black level subtracted).

    When num_frames > 1, generates multiple noisy frames and averages them.
    This reduces read noise by sqrt(N) while preserving shot noise characteristics,
    naturally shifting toward shot-noise dominated regime.

    Args:
        clean_image: Clean image in normalized [-1, 1]
        exposure_ratio: Exposure ratio for short exposure
        poisson_coeff: Poisson coefficient (a in variance = a*μ + b)
        sigma_r: Read noise standard deviation
        sensor_range: Sensor range (white_level - black_level)
        black_level: Black level offset
        target_signal: Optional target signal level (overrides exposure_ratio)
        num_frames: Number of frames to average (for multi-frame mode)

    Returns:
        Tuple of (noisy_image, snr, exposure_ratio_used, effective_sigma_r)
        where effective_sigma_r accounts for multi-frame averaging
    """
    # Convert from normalized [-1,1] to [0,1] to sensor range [0, s]
    clean_01 = (clean_image + 1) / 2
    clean_sensor_range = clean_01 * sensor_range  # [0, s] units

    if target_signal is not None:
        # Scale image to achieve target mean signal level (in sensor range units)
        current_mean = clean_sensor_range.mean().item()
        if current_mean > 0:
            scale_factor = target_signal / current_mean
            short_sensor_range = clean_sensor_range * scale_factor
            short_sensor_range = torch.clamp(short_sensor_range, 0, sensor_range)
        else:
            short_sensor_range = clean_sensor_range
        exposure_ratio_used = 1.0
    else:
        # Scale by exposure ratio to simulate short exposure
        short_sensor_range = clean_sensor_range * exposure_ratio
        exposure_ratio_used = exposure_ratio

    # Generate noisy frames
    noisy_frames = []
    for _ in range(num_frames):
        # Add Poisson-Gaussian noise: variance = poisson_coeff * signal + σ_r²
        # where signal is in sensor range units [0, s]
        variance = poisson_coeff * short_sensor_range + sigma_r**2
        noise = torch.randn_like(short_sensor_range) * torch.sqrt(variance)
        noisy_sensor_range = torch.clamp(short_sensor_range + noise, 0, sensor_range)

        # Convert back to absolute ADU for clamping and output
        short_phys = short_sensor_range + black_level
        noisy_phys = noisy_sensor_range + black_level
        noisy_phys = torch.clamp(noisy_phys, black_level, black_level + sensor_range)

        # Convert to normalized [-1, 1]
        noisy_01 = (noisy_phys - black_level) / sensor_range
        noisy_norm = noisy_01 * 2 - 1
        noisy_frames.append(noisy_norm)

    # Average frames if multi-frame mode
    if num_frames > 1:
        noisy_norm = average_multiple_frames(noisy_frames)
        # Effective read noise after averaging N frames: σ_r / sqrt(N)
        effective_sigma_r = sigma_r / np.sqrt(num_frames)
    else:
        noisy_norm = noisy_frames[0]
        effective_sigma_r = sigma_r

    # Compute SNR (signal and noise both in sensor range units)
    # After averaging, variance = poisson_coeff * signal + (sigma_r/sqrt(N))²
    signal_mean = short_sensor_range.mean().item()
    effective_variance = poisson_coeff * signal_mean + effective_sigma_r**2
    noise_std = np.sqrt(effective_variance)
    snr = signal_mean / noise_std if noise_std > 0 else 0

    return noisy_norm, snr, exposure_ratio_used, effective_sigma_r


def _prepare_image_for_guidance(
    noisy_image: torch.Tensor,
    exposure_ratio: float,
    black_level: float,
    white_level: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare image for guidance: convert to physical space and initialize x_T."""
    y_e_np = denormalize_to_physical(noisy_image, black_level, white_level)
    y_e = torch.from_numpy(y_e_np).to(noisy_image.device)
    x_T = torch.clamp(noisy_image * (1.0 / exposure_ratio), -1, 1)
    return x_T, y_e


def evaluate_single_image(
    clean_image: torch.Tensor,
    exposure_ratio: float,
    sampler: EDMPosteriorSampler,
    pg_guidance: PoissonGaussianGuidance,
    gaussian_guidance: GaussianGuidance,
    poisson_coeff: float,
    sigma_r: float,
    sensor_range: float,
    black_level: float,
    num_steps: int = 18,
    kappa: float = 0.15,
    target_signal: Optional[float] = None,
    target_poisson_fraction: Optional[float] = None,
    num_frames: int = 1,
    save_restored_images: bool = False,
    output_dir: Optional[Path] = None,
    image_name: Optional[str] = None,
) -> Dict:
    """
    Evaluate both PG and Gaussian guidance on a single image.

    Supports multi-frame averaging (Option D) which reduces read noise by sqrt(N)
    while preserving shot noise characteristics, naturally shifting toward
    shot-noise dominated regime.
    """

    # Add synthetic noise (with optional multi-frame averaging)
    (
        noisy_image,
        snr,
        exposure_ratio_used,
        effective_sigma_r,
    ) = add_synthetic_poisson_gaussian_noise(
        clean_image,
        exposure_ratio,
        poisson_coeff,
        sigma_r,
        sensor_range,
        black_level,
        target_signal=target_signal,
        num_frames=num_frames,
    )

    # Compute Poisson contribution
    # Signal is measured in sensor range units [0, s], not absolute ADU
    clean_01 = (clean_image + 1) / 2
    if target_signal is not None:
        # Use target signal directly (already in sensor range units)
        expected_signal = target_signal
    else:
        # Compute expected signal level at short exposure (in sensor range units [0, s])
        clean_mean_01 = clean_01.mean().item()
        clean_mean_sensor_range = clean_mean_01 * sensor_range  # [0, s]
        expected_signal = clean_mean_sensor_range * exposure_ratio

    # Use effective sigma_r for variance calculations (accounts for multi-frame averaging)
    poisson_term = poisson_coeff * expected_signal
    total_variance = poisson_term + effective_sigma_r**2
    poisson_fraction = poisson_term / total_variance if total_variance > 0 else 0.0

    # If we have a target Poisson fraction, use it (may differ slightly due to quantization)
    if target_poisson_fraction is not None:
        poisson_fraction = target_poisson_fraction

    # Classify noise regime using effective_sigma_r (after multi-frame averaging)
    classifier = NoiseRegimeClassifier(poisson_coeff, effective_sigma_r)
    regime, variance_ratio = classifier.classify(expected_signal)
    recommended_guidance = classifier.select_guidance(regime)

    # Prepare image for guidance (convert to physical space)
    white_level = black_level + sensor_range
    x_T, y_e = _prepare_image_for_guidance(
        noisy_image, exposure_ratio_used, black_level, white_level
    )

    # Set guidance parameters (use effective_sigma_r for guidance)
    # Note: We update sigma_r in guidance to reflect multi-frame averaging
    pg_guidance.alpha = exposure_ratio_used
    pg_guidance.kappa = kappa
    pg_guidance.sigma_r = effective_sigma_r  # Update for multi-frame averaging

    gaussian_guidance.alpha = exposure_ratio_used
    gaussian_guidance.kappa = kappa
    gaussian_guidance.sigma_r = effective_sigma_r  # Update for multi-frame averaging

    # Estimate sigma_max
    sigma_max = 0.5  # Reasonable default

    # Run PG guidance using existing function
    pg_restored = _run_guidance_sample(
        sampler,
        x_T,
        y_e,
        sigma_max,
        exposure_ratio_used,
        pg_guidance,
        "x0",
        num_steps,
        no_heun=True,
        is_gaussian=False,
        class_labels=None,
    )

    # Run Gaussian guidance using existing function
    gaussian_restored = _run_guidance_sample(
        sampler,
        x_T,
        y_e,
        sigma_max,
        exposure_ratio_used,
        gaussian_guidance,
        "x0",
        num_steps,
        no_heun=True,
        is_gaussian=True,
        class_labels=None,
    )

    if save_restored_images and output_dir and image_name:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(clean_image.cpu(), output_dir / f"{image_name}_clean.pt")
        torch.save(noisy_image.cpu(), output_dir / f"{image_name}_noisy.pt")
        torch.save(pg_restored.cpu(), output_dir / f"{image_name}_pg_x0.pt")
        torch.save(gaussian_restored.cpu(), output_dir / f"{image_name}_gaussian_x0.pt")

    # Compute metrics (while tensors are still on GPU)
    device = clean_image.device
    pg_metrics = compute_metrics_by_method(
        clean_image, pg_restored, "pg_x0", device=device
    )
    gaussian_metrics = compute_metrics_by_method(
        clean_image, gaussian_restored, "gaussian_x0", device=device
    )

    # Move tensors to CPU and delete GPU versions (only if not saving images)
    if not save_restored_images:
        pg_restored_cpu = pg_restored.cpu()
        gaussian_restored_cpu = gaussian_restored.cpu()
        clean_image_cpu = clean_image.cpu()
        noisy_image_cpu = noisy_image.cpu()
        # Delete GPU tensors to free memory
        del pg_restored, gaussian_restored, clean_image, noisy_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        pg_restored_cpu = None
        gaussian_restored_cpu = None
        clean_image_cpu = None
        noisy_image_cpu = None

    return {
        "exposure_ratio": exposure_ratio,
        "snr": snr,
        "expected_signal": expected_signal,
        "poisson_fraction": poisson_fraction,
        "regime": regime,
        "variance_ratio": variance_ratio,
        "recommended_guidance": recommended_guidance,
        "num_frames": num_frames,
        "effective_sigma_r": effective_sigma_r,
        "sigma_r_reduction": effective_sigma_r / sigma_r if sigma_r > 0 else 1.0,
        "pg_psnr": pg_metrics["psnr"],
        "gaussian_psnr": gaussian_metrics["psnr"],
        "pg_ssim": pg_metrics["ssim"],
        "gaussian_ssim": gaussian_metrics["ssim"],
        "psnr_gain": pg_metrics["psnr"] - gaussian_metrics["psnr"],
        "ssim_gain": pg_metrics["ssim"] - gaussian_metrics["ssim"],
        "pg_restored": pg_restored_cpu,
        "gaussian_restored": gaussian_restored_cpu,
        "clean_image": clean_image_cpu,
        "noisy_image": noisy_image_cpu,
    }


def get_sensor_defaults(sensor: str) -> Dict[str, Any]:
    """
    Get sensor-specific default hyperparameters matching test_guidance_comparison_unified.sh.

    Returns:
        Dict with 'kappa' and 'num_steps' keys
    """
    if sensor == "sony":
        return {"kappa": 0.15, "num_steps": 10}
    elif sensor == "fuji":
        return {"kappa": 0.05, "num_steps": 18}
    else:
        return {"kappa": 0.15, "num_steps": 18}  # Default fallback


def save_sweep_results(
    all_results: List[Dict],
    test_values: List[float],
    mode: str,
    sensor: str,
    output_dir: Path,
    poisson_coeff: float,
    sigma_r: float,
    kappa: float = None,
    num_steps: int = None,
) -> None:
    """
    Save sweep results in structured format (JSON and CSV) like tune_hyperparameters.sh.

    Args:
        all_results: All evaluation results
        test_values: List of test values (exposure ratios or Poisson percentages)
        mode: "exposure_ratio" or "poisson_percentage"
        sensor: Sensor name
        output_dir: Output directory
        poisson_coeff: Poisson coefficient
        sigma_r: Read noise standard deviation
    """
    # Group results by test value
    results_by_value = {}
    for test_value in test_values:
        if mode == "poisson_percentage":
            ratio_results = [
                r
                for r in all_results
                if r.get("target_poisson_percentage") == test_value * 100
            ]
        else:
            ratio_results = [
                r for r in all_results if abs(r["exposure_ratio"] - test_value) < 1e-6
            ]
        results_by_value[test_value] = ratio_results

    # Prepare JSON structure
    sweep_results = {
        "sensor": sensor,
        "mode": mode,
        "test_values": test_values,
        "num_images": len(all_results) // len(test_values) if test_values else 0,
        "num_frames": all_results[0].get("num_frames", 1) if all_results else 1,
        "kappa": kappa
        if kappa is not None
        else (all_results[0].get("kappa", 0.15) if all_results else 0.15),
        "num_steps": num_steps
        if num_steps is not None
        else (all_results[0].get("num_steps", 18) if all_results else 18),
        "results": [],
    }

    # Prepare CSV data
    csv_rows = []
    csv_header = "test_value,pg_psnr_mean,pg_psnr_std,pg_ssim_mean,pg_ssim_std,gaussian_psnr_mean,gaussian_psnr_std,gaussian_ssim_mean,gaussian_ssim_std,psnr_gain_mean,psnr_gain_std,poisson_fraction_mean,variance_ratio_mean,regime_read_noise,regime_transitional,regime_shot_noise"

    for test_value in test_values:
        ratio_results = results_by_value.get(test_value, [])
        if len(ratio_results) == 0:
            continue

        # Compute aggregate metrics
        pg_psnrs = [r["pg_psnr"] for r in ratio_results]
        gaussian_psnrs = [r["gaussian_psnr"] for r in ratio_results]
        pg_ssims = [r["pg_ssim"] for r in ratio_results]
        gaussian_ssims = [r["gaussian_ssim"] for r in ratio_results]
        psnr_gains = [r["psnr_gain"] for r in ratio_results]
        poisson_fracs = [r["poisson_fraction"] for r in ratio_results]
        variance_ratios = [r.get("variance_ratio", 0.0) for r in ratio_results]
        regimes = [r.get("regime", "unknown") for r in ratio_results]

        # Count regimes
        regime_counts = count_regimes(regimes)

        # Compute statistics
        pg_psnr_mean = float(np.mean(pg_psnrs))
        pg_psnr_std = float(np.std(pg_psnrs))
        gaussian_psnr_mean = float(np.mean(gaussian_psnrs))
        gaussian_psnr_std = float(np.std(gaussian_psnrs))
        pg_ssim_mean = float(np.mean(pg_ssims))
        pg_ssim_std = float(np.std(pg_ssims))
        gaussian_ssim_mean = float(np.mean(gaussian_ssims))
        gaussian_ssim_std = float(np.std(gaussian_ssims))
        psnr_gain_mean = float(np.mean(psnr_gains))
        psnr_gain_std = float(np.std(psnr_gains))
        poisson_fraction_mean = float(np.mean(poisson_fracs))
        variance_ratio_mean = (
            float(np.mean(variance_ratios)) if variance_ratios else 0.0
        )

        # Statistical test
        t_stat, p_value = stats.ttest_1samp(psnr_gains, 0)

        # Add to JSON
        sweep_results["results"].append(
            {
                "test_value": float(test_value),
                "metrics": {
                    "pg": {
                        "psnr": {"mean": pg_psnr_mean, "std": pg_psnr_std},
                        "ssim": {"mean": pg_ssim_mean, "std": pg_ssim_std},
                    },
                    "gaussian": {
                        "psnr": {"mean": gaussian_psnr_mean, "std": gaussian_psnr_std},
                        "ssim": {"mean": gaussian_ssim_mean, "std": gaussian_ssim_std},
                    },
                    "psnr_gain": {"mean": psnr_gain_mean, "std": psnr_gain_std},
                },
                "poisson_fraction": {"mean": poisson_fraction_mean},
                "variance_ratio": {"mean": variance_ratio_mean},
                "regime_distribution": regime_counts,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "n_images": len(ratio_results),
            }
        )

        # Add to CSV
        csv_rows.append(
            f"{test_value},{pg_psnr_mean:.4f},{pg_psnr_std:.4f},"
            f"{pg_ssim_mean:.6f},{pg_ssim_std:.6f},"
            f"{gaussian_psnr_mean:.4f},{gaussian_psnr_std:.4f},"
            f"{gaussian_ssim_mean:.6f},{gaussian_ssim_std:.6f},"
            f"{psnr_gain_mean:.4f},{psnr_gain_std:.4f},"
            f"{poisson_fraction_mean:.4f},{variance_ratio_mean:.4f},"
            f"{regime_counts.get('read_noise_dominated', 0)},"
            f"{regime_counts.get('transitional', 0)},"
            f"{regime_counts.get('shot_noise_dominated', 0)}"
        )

    json_file = output_dir / "sweep_results.json"
    save_json_safe(sweep_results, json_file)

    csv_file = output_dir / "sweep_results.csv"
    with open(csv_file, "w") as f:
        f.write(csv_header + "\n")
        for row in csv_rows:
            f.write(row + "\n")


def create_sweep_visualization(
    sweep_results: Dict,
    output_dir: Path,
) -> None:
    """
    Create visualization showing performance vs exposure ratio (like tune_hyperparameters.sh).

    Shows how PG guidance gains vary across different noise regimes.
    """
    if not sweep_results.get("results"):
        return

    results = sweep_results["results"]
    test_values = [r["test_value"] for r in results]

    # Extract data
    psnr_gains = [r["metrics"]["psnr_gain"]["mean"] for r in results]
    psnr_gain_stds = [r["metrics"]["psnr_gain"]["std"] for r in results]
    poisson_fractions = [r["poisson_fraction"]["mean"] for r in results]
    variance_ratios = [r["variance_ratio"]["mean"] for r in results]
    pg_psnrs = [r["metrics"]["pg"]["psnr"]["mean"] for r in results]
    gaussian_psnrs = [r["metrics"]["gaussian"]["psnr"]["mean"] for r in results]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: PSNR gain vs exposure ratio (main result)
    ax = axes[0, 0]
    ax.errorbar(
        test_values,
        psnr_gains,
        yerr=psnr_gain_stds,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        capsize=5,
        label="PSNR Gain (PG - Gaussian)",
        color="green",
    )
    ax.axhline(
        y=0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="No difference"
    )
    ax.set_xlabel("Exposure Ratio", fontsize=11)
    ax.set_ylabel("PSNR Gain [dB]", fontsize=11)
    ax.set_title("PSNR Gain vs Exposure Ratio", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: PSNR comparison (PG vs Gaussian)
    ax = axes[0, 1]
    ax.plot(
        test_values,
        pg_psnrs,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        label="PG Guidance",
        color="green",
    )
    ax.plot(
        test_values,
        gaussian_psnrs,
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=8,
        label="Gaussian Guidance",
        color="orange",
    )
    ax.set_xlabel("Exposure Ratio", fontsize=11)
    ax.set_ylabel("PSNR [dB]", fontsize=11)
    ax.set_title("PSNR Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Poisson fraction vs exposure ratio
    ax = axes[1, 0]
    ax.plot(
        test_values,
        poisson_fractions,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        label="Poisson Fraction",
        color="blue",
    )
    ax.axhline(
        y=SHOT_NOISE_THRESHOLD,
        color="green",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Shot-noise threshold (20%)",
    )
    ax.set_xlabel("Exposure Ratio", fontsize=11)
    ax.set_ylabel("Poisson Contribution Fraction", fontsize=11)
    ax.set_title("Noise Regime Indicator", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Variance ratio vs exposure ratio
    ax = axes[1, 1]
    ax.plot(
        test_values,
        variance_ratios,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        label="Variance Ratio",
        color="purple",
    )
    ax.axhline(
        y=READ_NOISE_THRESHOLD,
        color="orange",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Read-noise threshold",
    )
    ax.axhline(
        y=SHOT_NOISE_THRESHOLD,
        color="green",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Shot-noise threshold",
    )
    ax.set_xlabel("Exposure Ratio", fontsize=11)
    ax.set_ylabel("Variance Ratio (a·μ / σ_r²)", fontsize=11)
    ax.set_title("Regime Classification", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle(
        "Exposure Ratio Sweep: Demonstrating Regime-Dependent Performance",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    fig.savefig(output_dir / "sweep_analysis.png", dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def find_best_exposure_ratios(sweep_results: Dict) -> Dict[str, Any]:
    """
    Find best exposure ratios like tune_hyperparameters.sh finds best hyperparameters.

    Returns summary of best ratios by different metrics.
    """
    if not sweep_results.get("results"):
        return {}

    results = sweep_results["results"]

    # Find best by PSNR gain
    best_psnr_gain = max(results, key=lambda x: x["metrics"]["psnr_gain"]["mean"])

    # Find best by PG PSNR
    best_pg_psnr = max(results, key=lambda x: x["metrics"]["pg"]["psnr"]["mean"])

    # Find first significant gain (p < 0.05)
    significant_results = [r for r in results if r["significant"]]
    first_significant = (
        min(significant_results, key=lambda x: x["test_value"])
        if significant_results
        else None
    )

    summary = {
        "best_by_psnr_gain": {
            "test_value": best_psnr_gain["test_value"],
            "psnr_gain": best_psnr_gain["metrics"]["psnr_gain"]["mean"],
            "p_value": best_psnr_gain["p_value"],
            "poisson_fraction": best_psnr_gain["poisson_fraction"]["mean"],
            "variance_ratio": best_psnr_gain["variance_ratio"]["mean"],
        },
        "best_by_pg_psnr": {
            "test_value": best_pg_psnr["test_value"],
            "pg_psnr": best_pg_psnr["metrics"]["pg"]["psnr"]["mean"],
            "psnr_gain": best_pg_psnr["metrics"]["psnr_gain"]["mean"],
        },
        "first_significant": {
            "test_value": first_significant["test_value"],
            "psnr_gain": first_significant["metrics"]["psnr_gain"]["mean"],
            "p_value": first_significant["p_value"],
        }
        if first_significant
        else None,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic validation for heteroscedastic guidance"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to EDM model (auto-detected if not provided, uses latest training)",
    )
    parser.add_argument(
        "--test_images_dir",
        type=str,
        required=True,
        help="Directory with test images (long exposure .pt files or stitched scenes)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/synthetic_validation",
        help="Output directory",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="fuji",
        choices=["sony", "fuji"],
        help="Sensor type",
    )
    parser.add_argument(
        "--num_images", type=int, default=20, help="Number of test images"
    )
    parser.add_argument(
        "--sweep_poisson_range",
        type=float,
        nargs="+",
        default=None,
        help="Custom Poisson percentage range for sweep (default: 5 10 15 20 30). "
        "Tests multiple Poisson fractions systematically for CVPR paper Fig 1.",
    )
    parser.add_argument(
        "--val_split_file",
        type=str,
        default=None,
        help="Path to validation split file (auto-detected if not provided). "
        "Looks for dataset/splits/{Sensor}_val_list.txt",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        default=None,
        help="Path to metadata JSON (auto-detected if not provided). "
        "Looks for comprehensive_{sensor}_tiles_metadata.json or comprehensive_all_tiles_metadata.json",
    )
    parser.add_argument(
        "--use_scenes",
        action="store_true",
        help="If set, look for stitched_long.pt files in scene directories instead of individual tiles. "
        "WARNING: This can cause OOM with large stitched scenes. Use --use_tiles instead.",
    )
    parser.add_argument(
        "--use_tiles",
        action="store_true",
        help="Process individual tiles in batches, then stitch together (recommended to avoid OOM). "
        "Requires --metadata_json and --short_dir/--long_dir pointing to tile directories.",
    )
    parser.add_argument(
        "--tile_batch_size",
        type=int,
        default=50,
        help="Number of tiles to process in each batch when using --use_tiles. Default: 50. "
        "Smaller values reduce memory usage but may be slower.",
    )
    parser.add_argument(
        "--short_dir",
        type=str,
        default=None,
        help="Directory containing short exposure .pt tile files (required for --use_tiles)",
    )
    parser.add_argument(
        "--long_dir",
        type=str,
        default=None,
        help="Directory containing long exposure .pt tile files (required for --use_tiles)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of sampling steps (uses sensor-specific default if not provided)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=None,
        help="Guidance strength (uses sensor-specific default if not provided)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=1,
        help="Number of frames to average (Option D: multi-frame averaging). "
        "Averaging N frames reduces read noise by sqrt(N) while preserving "
        "shot noise characteristics, naturally shifting toward shot-noise dominated regime.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Auto-detect model if not provided
    if args.model_path is None:
        model_path = find_latest_model(args.sensor)
        if model_path is None:
            raise ValueError(
                f"Model not found and --model_path not provided.\n"
                f"Expected pattern: results/edm_{args.sensor}_training_*/best_model.pkl"
            )
        args.model_path = str(model_path)
    else:
        if not Path(args.model_path).exists():
            raise ValueError(f"Model file not found: {args.model_path}")

    if args.metadata_json is None:
        metadata_path = find_metadata_json(args.sensor)
        if metadata_path:
            args.metadata_json = str(metadata_path)
    else:
        if not Path(args.metadata_json).exists():
            raise ValueError(f"Metadata file not found: {args.metadata_json}")

    if args.val_split_file is None:
        split_file = find_split_file(args.sensor, split_type="val")
        if split_file:
            args.val_split_file = str(split_file)
    else:
        if not Path(args.val_split_file).exists():
            raise ValueError(f"Split file not found: {args.val_split_file}")

    sensor_defaults = get_sensor_defaults(args.sensor)
    if args.kappa is None:
        args.kappa = sensor_defaults["kappa"]
    if args.num_steps is None:
        args.num_steps = sensor_defaults["num_steps"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "poisson_percentage"
    sweep_mode = True

    if args.sweep_poisson_range:
        test_values = [p / 100.0 for p in sorted(args.sweep_poisson_range)]
    else:
        test_values = [0.05, 0.10, 0.15, 0.20, 0.30]

    sampler = EDMPosteriorSampler(model_path=args.model_path, device=args.device)

    # Load sensor calibration
    black_level, white_level = load_sensor_calibration_from_metadata(args.sensor)
    sensor_range = white_level - black_level

    # Load noise calibration
    calibration_data = load_noise_calibration(
        args.sensor, Path("dataset/processed"), data_root=Path("dataset/processed")
    )

    if calibration_data is None:
        raise ValueError(f"Noise calibration not found for {args.sensor}")

    sigma_r = convert_calibration_to_sigma_r(calibration_data["b"], sensor_range)
    poisson_coeff = convert_calibration_to_poisson_coeff(
        calibration_data["a"], sensor_range
    )

    # Create guidance modules
    common_params = {
        "s": sensor_range,
        "sigma_r": sigma_r,
        "black_level": black_level,
        "white_level": white_level,
        "offset": 0.0,
        "exposure_ratio": 1.0,  # Will be updated per image
        "kappa": args.kappa,
        "tau": 0.01,
    }

    pg_guidance = PoissonGaussianGuidance(
        **common_params,
        mode="wls",
        guidance_level="x0",
        poisson_coeff=poisson_coeff,
    )

    gaussian_guidance = GaussianGuidance(**common_params)

    # Validate tile-based mode requirements
    if args.use_tiles:
        if not args.metadata_json:
            raise ValueError("--metadata_json is required when using --use_tiles")
        if not args.short_dir or not args.long_dir:
            raise ValueError(
                "--short_dir and --long_dir are required when using --use_tiles"
            )
        if args.use_scenes:
            raise ValueError("Cannot use both --use_tiles and --use_scenes together")

    # Load test images
    test_images_dir = Path(args.test_images_dir)

    # Tile-based processing mode (recommended to avoid OOM)
    if args.use_tiles:
        # Load metadata
        tile_lookup = load_metadata_json(Path(args.metadata_json))

        # Load tile IDs from split file (like test_guidance_comparison_unified.sh does)
        split_file = Path(args.val_split_file) if args.val_split_file else None
        if split_file and split_file.exists():
            test_tile_ids = load_tile_ids_from_split_file(
                split_file, args.sensor, Path(args.metadata_json), data_type="short"
            )
        else:
            if split_file is None or not split_file.exists():
                split_file = find_split_file(args.sensor, split_type="test")
                if split_file is None or not split_file.exists():
                    split_file = find_split_file(args.sensor, split_type="val")
            if split_file and split_file.exists():
                test_tile_ids = load_tile_ids_from_split_file(
                    split_file, args.sensor, Path(args.metadata_json), data_type="short"
                )
            else:
                raise ValueError(
                    f"No split file found. Please provide --val_split_file or ensure "
                    f"dataset/splits/{args.sensor.capitalize()}_test_list.txt exists"
                )

        if not test_tile_ids:
            raise ValueError(f"No tile IDs found in split file: {split_file}")

        selected_tiles = select_tiles_for_processing(
            tile_ids=test_tile_ids,
            metadata_json=Path(args.metadata_json),
            tile_lookup=tile_lookup,
            num_examples=args.num_images,
            seed=args.seed,
            sensor_filter=args.sensor,
        )

        scene_groups = group_tiles_by_scene(selected_tiles, tile_lookup)

        if len(scene_groups) > args.num_images:
            random.seed(args.seed)
            scene_keys = list(scene_groups.keys())
            selected_scene_keys = random.sample(scene_keys, args.num_images)
            scene_groups = {key: scene_groups[key] for key in selected_scene_keys}

        # Store scene groups for later processing
        tile_scene_groups = scene_groups
        image_files = None  # Not used in tile mode
    elif args.use_scenes:
        scene_dirs = find_scene_directories(test_images_dir, prefix="")
        scene_id_to_files = {}
        for scene_dir in scene_dirs:
            dir_name = scene_dir.name
            if dir_name.startswith("scene_"):
                parts = dir_name.split("_")
                if len(parts) >= 2:
                    scene_id = parts[1]
                    stitched_file = scene_dir / "stitched_long.pt"
                    if stitched_file.exists() and scene_id not in scene_id_to_files:
                        scene_id_to_files[scene_id] = stitched_file
        image_files = sorted(scene_id_to_files.values())
    else:
        image_files = sorted(list(test_images_dir.glob("*.pt")))

    # Only validate image_files if not using tile mode
    if not args.use_tiles:
        if len(image_files) == 0:
            if args.use_scenes:
                raise ValueError(
                    f"No stitched_long.pt files found in subdirectories of {test_images_dir}"
                )
            else:
                raise ValueError(f"No .pt files found in {test_images_dir}")

    def _validate_filtered_images(image_files: List[Path], use_scenes: bool) -> None:
        """Validate that filtered image list is not empty."""
        if len(image_files) == 0:
            if use_scenes:
                raise ValueError(
                    f"No stitched scenes match validation split after filtering.\n"
                    f"This usually means validation scenes haven't been stitched yet.\n"
                    f"Try running without --use_scenes to use individual tiles instead."
                )
            else:
                raise ValueError(f"No images match validation split after filtering")

    # Filter by validation split if provided (only for non-tile mode)
    if not args.use_tiles:
        metadata_json = Path(args.metadata_json) if args.metadata_json else None
        if args.val_split_file:
            split_file = Path(args.val_split_file)
            if split_file.exists():
                image_files = filter_images_by_split(
                    image_files,
                    split_file,
                    args.sensor,
                    metadata_json=metadata_json,
                    use_scenes=args.use_scenes,
                )
                image_files = sorted(image_files)
                _validate_filtered_images(image_files, args.use_scenes)
            else:
                logger.warning(
                    f"Split file was set but doesn't exist: {args.val_split_file}"
                )

        if len(image_files) > args.num_images:
            random.seed(args.seed)
            image_files = sorted(random.sample(image_files, args.num_images))
        else:
            image_files = image_files[: args.num_images]

    def _compute_test_parameters(
        test_value: float,
        mode: str,
        poisson_coeff: float,
        sigma_r_squared: float,
    ) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Compute test parameters (exposure_ratio, target_signal, target_poisson_fraction).

        Returns:
            Tuple of (exposure_ratio, target_signal, target_poisson_fraction)
        """
        if mode == "poisson_percentage":
            target_poisson_fraction = test_value
            target_signal = compute_target_signal_for_poisson_percentage(
                target_poisson_fraction, poisson_coeff, sigma_r_squared
            )
            exposure_ratio = 1.0
        else:
            exposure_ratio = test_value
            target_signal = None
            target_poisson_fraction = None
        return exposure_ratio, target_signal, target_poisson_fraction

    def _group_results_by_test_value(
        all_results: List[Dict],
        test_values: List[float],
        mode: str,
    ) -> Dict[float, List[Dict]]:
        """Group results by test value and return mapping."""
        results_by_test_value = {}
        for test_value in test_values:
            if mode == "poisson_percentage":
                ratio_results = [
                    r
                    for r in all_results
                    if r.get("target_poisson_percentage") == test_value * 100
                ]
                key = f"poisson_{test_value*100:.0f}pct"
                display_key = f"Poisson {test_value*100:.0f}%"
            else:
                ratio_results = [
                    r
                    for r in all_results
                    if abs(r["exposure_ratio"] - test_value) < 1e-6
                ]
                key = f"ratio_{test_value}"
                display_key = f"Ratio {test_value}"
            results_by_test_value[test_value] = ratio_results
        return results_by_test_value

    def _load_clean_image(img_file: Path, device: str) -> torch.Tensor:
        """Load and prepare clean image from file."""
        clean_image = torch.load(img_file, map_location=device)
        if clean_image.ndim == 4:
            clean_image = clean_image[0]
        return clean_image

    if sweep_mode:
        logger.info(
            f"Poisson sweep: {len(test_values)} fractions, {args.num_images} images"
        )

    all_results = []
    sigma_r_squared = sigma_r**2

    # Tile-based processing function
    def process_scene_tiles(
        scene_key: Tuple[str, str, float],
        scene_tiles: List[Dict],
        test_value: float,
        exposure_ratio: float,
        target_signal: Optional[float],
        target_poisson_fraction: Optional[float],
    ) -> Dict:
        """Process a scene's tiles in batches, then stitch together."""
        scene_id, frame_id, exposure_time = scene_key

        # Load tiles for this scene in batches to avoid OOM
        long_tiles = {}
        short_tiles = {}
        tile_metadata_list = []

        tile_batch_size = args.tile_batch_size
        num_tile_batches = (len(scene_tiles) + tile_batch_size - 1) // tile_batch_size

        for batch_idx in range(num_tile_batches):
            start_idx = batch_idx * tile_batch_size
            end_idx = min(start_idx + tile_batch_size, len(scene_tiles))
            batch_tiles = scene_tiles[start_idx:end_idx]

            for tile_info in batch_tiles:
                try:
                    tile_data = load_tile_and_metadata(
                        tile_id=tile_info["tile_id"],
                        tile_info=tile_info,
                        tile_lookup=tile_lookup,
                        short_dir=args.short_dir,
                        long_dir=args.long_dir,
                        device=args.device,
                        img_channels=sampler.net.img_channels,
                        sensor_ranges=sampler.sensor_ranges,
                        use_sensor_calibration=True,
                        sensor_name=None,
                        conservative_factor=1.0,
                        sigma_r=sigma_r,
                    )
                    if tile_data is None:
                        continue

                    long_image = tile_data["long_image"]  # Already in [-1,1]
                    short_image = tile_data["short_image"]
                    short_phys = tile_data["short_phys"]

                    # Get grid position from metadata
                    grid_x = tile_data.get("grid_x", 0)
                    grid_y = tile_data.get("grid_y", 0)

                    # Add synthetic noise to long exposure tile
                    long_noisy, _, _, _ = add_synthetic_poisson_gaussian_noise(
                        long_image,
                        exposure_ratio,
                        poisson_coeff,
                        sigma_r,
                        sensor_range,
                        black_level,
                        target_signal=target_signal,
                        num_frames=args.num_frames,
                    )

                    # Prepare for guidance
                    white_level = black_level + sensor_range
                    x_T, y_e = _prepare_image_for_guidance(
                        long_noisy, exposure_ratio, black_level, white_level
                    )

                    # Update guidance parameters
                    pg_guidance.alpha = exposure_ratio
                    pg_guidance.kappa = args.kappa
                    pg_guidance.sigma_r = sigma_r
                    gaussian_guidance.alpha = exposure_ratio
                    gaussian_guidance.kappa = args.kappa
                    gaussian_guidance.sigma_r = sigma_r

                    sigma_max = tile_data.get("estimated_sigma", 0.5)

                    # Run restoration on tile
                    pg_restored_tile = _run_guidance_sample(
                        sampler,
                        x_T,
                        y_e,
                        sigma_max,
                        exposure_ratio,
                        pg_guidance,
                        "x0",
                        args.num_steps,
                        no_heun=True,
                        is_gaussian=False,
                        class_labels=None,
                    )

                    gaussian_restored_tile = _run_guidance_sample(
                        sampler,
                        x_T,
                        y_e,
                        sigma_max,
                        exposure_ratio,
                        gaussian_guidance,
                        "x0",
                        args.num_steps,
                        no_heun=True,
                        is_gaussian=True,
                        class_labels=None,
                    )

                    # Store tiles with grid positions
                    long_tiles[(grid_x, grid_y)] = long_image.cpu()
                    short_tiles[(grid_x, grid_y)] = long_noisy.cpu()

                    # Store for stitching
                    tile_metadata_list.append(
                        {
                            "tile_id": tile_info["tile_id"],
                            "grid_x": grid_x,
                            "grid_y": grid_y,
                            "image_x": tile_data.get("image_x", grid_x * 256),
                            "image_y": tile_data.get("image_y", grid_y * 256),
                            "pg_restored": pg_restored_tile.cpu(),
                            "gaussian_restored": gaussian_restored_tile.cpu(),
                        }
                    )

                    # Clear GPU memory after each tile
                    del (
                        long_image,
                        short_image,
                        long_noisy,
                        x_T,
                        y_e,
                        pg_restored_tile,
                        gaussian_restored_tile,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(
                        f"Failed to process tile {tile_info.get('tile_id', 'unknown')}: {e}"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            # Clear GPU cache after each batch of tiles
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not tile_metadata_list:
            return None

        # Stitch tiles together
        stitched_long = stitch_tiles_to_grid(
            long_tiles,
            tile_size=256,
            sensor_type=args.sensor,
            tile_metadata_list=tile_metadata_list,
        )
        stitched_short = stitch_tiles_to_grid(
            short_tiles,
            tile_size=256,
            sensor_type=args.sensor,
            tile_metadata_list=tile_metadata_list,
        )

        pg_tiles_dict = {
            (t["grid_x"], t["grid_y"]): t["pg_restored"] for t in tile_metadata_list
        }
        gaussian_tiles_dict = {
            (t["grid_x"], t["grid_y"]): t["gaussian_restored"]
            for t in tile_metadata_list
        }

        stitched_pg = stitch_tiles_to_grid(
            pg_tiles_dict,
            tile_size=256,
            sensor_type=args.sensor,
            tile_metadata_list=tile_metadata_list,
        )
        stitched_gaussian = stitch_tiles_to_grid(
            gaussian_tiles_dict,
            tile_size=256,
            sensor_type=args.sensor,
            tile_metadata_list=tile_metadata_list,
        )

        if stitched_long is None or stitched_pg is None or stitched_gaussian is None:
            return None

        # Compute metrics on stitched scene
        stitched_long_gpu = stitched_long.to(args.device)
        stitched_pg_gpu = stitched_pg.to(args.device)
        stitched_gaussian_gpu = stitched_gaussian.to(args.device)

        pg_metrics = compute_metrics_by_method(
            stitched_long_gpu, stitched_pg_gpu, "pg_x0", device=args.device
        )
        gaussian_metrics = compute_metrics_by_method(
            stitched_long_gpu, stitched_gaussian_gpu, "gaussian_x0", device=args.device
        )

        # Compute Poisson fraction
        clean_01 = (stitched_long_gpu + 1) / 2
        if target_signal is not None:
            expected_signal = target_signal
        else:
            clean_mean_01 = clean_01.mean().item()
            expected_signal = clean_mean_01 * sensor_range * exposure_ratio

        poisson_term = poisson_coeff * expected_signal
        total_variance = poisson_term + sigma_r**2
        poisson_fraction = poisson_term / total_variance if total_variance > 0 else 0.0

        if target_poisson_fraction is not None:
            poisson_fraction = target_poisson_fraction

        classifier = NoiseRegimeClassifier(poisson_coeff, sigma_r)
        regime, variance_ratio = classifier.classify(expected_signal)

        # Clean up
        del stitched_long_gpu, stitched_pg_gpu, stitched_gaussian_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "exposure_ratio": exposure_ratio,
            "snr": expected_signal / np.sqrt(total_variance)
            if total_variance > 0
            else 0.0,
            "expected_signal": expected_signal,
            "poisson_fraction": poisson_fraction,
            "regime": regime,
            "variance_ratio": variance_ratio,
            "recommended_guidance": classifier.select_guidance(regime),
            "num_frames": args.num_frames,
            "pg_psnr": pg_metrics["psnr"],
            "gaussian_psnr": gaussian_metrics["psnr"],
            "pg_ssim": pg_metrics["ssim"],
            "gaussian_ssim": gaussian_metrics["ssim"],
            "psnr_gain": pg_metrics["psnr"] - gaussian_metrics["psnr"],
            "ssim_gain": pg_metrics["ssim"] - gaussian_metrics["ssim"],
            "scene_id": scene_id,
            "stitched_long": stitched_long.cpu(),
            "stitched_short": stitched_short.cpu()
            if stitched_short is not None
            else None,
            "stitched_pg": stitched_pg.cpu(),
            "stitched_gaussian": stitched_gaussian.cpu(),
        }

    for test_value in test_values:
        (
            exposure_ratio,
            target_signal,
            target_poisson_fraction,
        ) = _compute_test_parameters(test_value, mode, poisson_coeff, sigma_r_squared)

        ratio_results = []
        desc_label = (
            f"Poisson {target_poisson_fraction*100:.0f}%"
            if mode == "poisson_percentage"
            else f"Ratio {exposure_ratio}"
        )

        # Create scene directory for this test value (for incremental saving)
        if mode == "poisson_percentage":
            scene_key = f"synthetic_poisson_{test_value*100:.0f}pct"
        else:
            scene_key = f"synthetic_ratio_{test_value}"
        scene_dir = output_dir / scene_key
        scene_dir.mkdir(exist_ok=True)

        if args.use_tiles:
            # Process scenes using tiles
            for scene_key_tuple, scene_tiles in tqdm(
                tile_scene_groups.items(), desc=desc_label
            ):
                try:
                    result = process_scene_tiles(
                        scene_key_tuple,
                        scene_tiles,
                        test_value,
                        exposure_ratio,
                        target_signal,
                        target_poisson_fraction,
                    )
                    if result is not None:
                        result["target_poisson_percentage"] = (
                            target_poisson_fraction * 100
                            if mode == "poisson_percentage"
                            else None
                        )
                        ratio_results.append(result)

                        # Save this scene immediately
                        scene_id = result.get("scene_id", "unknown")
                        scene_subdir = scene_dir / f"scene_{scene_id}"
                        scene_subdir.mkdir(exist_ok=True)

                        if (
                            "stitched_long" in result
                            and result["stitched_long"] is not None
                        ):
                            torch.save(
                                result["stitched_long"],
                                scene_subdir / "stitched_long.pt",
                            )
                            if result.get("stitched_short") is not None:
                                torch.save(
                                    result["stitched_short"],
                                    scene_subdir / "stitched_short.pt",
                                )
                            if result.get("stitched_pg") is not None:
                                torch.save(
                                    result["stitched_pg"],
                                    scene_subdir / "stitched_pg_x0.pt",
                                )
                            if result.get("stitched_gaussian") is not None:
                                torch.save(
                                    result["stitched_gaussian"],
                                    scene_subdir / "stitched_gaussian_x0.pt",
                                )

                            # Create PNG visualization for this scene immediately
                            # Images are already on CPU from process_scene_tiles
                            stitched_short = (
                                result.get("stitched_short")
                                if result.get("stitched_short") is not None
                                else result["stitched_long"]
                            )
                            stitched_pg = (
                                result.get("stitched_pg")
                                if result.get("stitched_pg") is not None
                                else result["stitched_long"]
                            )
                            stitched_gaussian = (
                                result.get("stitched_gaussian")
                                if result.get("stitched_gaussian") is not None
                                else result["stitched_long"]
                            )

                            scene_stitched_images = {
                                "long": result["stitched_long"],
                                "short": stitched_short,
                                "pg_x0": stitched_pg,
                                "gaussian_x0": stitched_gaussian,
                            }

                            # Format metrics for this single scene
                            scene_metrics_single = [
                                {
                                    "comprehensive_metrics": {
                                        "pg_x0": {
                                            "psnr": result["pg_psnr"],
                                            "ssim": result["pg_ssim"],
                                        },
                                        "gaussian_x0": {
                                            "psnr": result["gaussian_psnr"],
                                            "ssim": result["gaussian_ssim"],
                                        },
                                    }
                                }
                            ]

                            try:
                                create_scene_visualization(
                                    scene_stitched_images,
                                    f"{scene_key}_scene_{scene_id}",
                                    args.sensor,
                                    scene_subdir,
                                    scene_metrics_single,
                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to create PNG visualization for scene {scene_id}: {e}"
                                )

                        # Update JSON results file incrementally
                        all_results.append(result)
                        results_file = output_dir / "synthetic_validation_results.json"
                        results_for_json = [
                            {
                                k: v
                                for k, v in r.items()
                                if not isinstance(v, torch.Tensor)
                            }
                            for r in all_results
                        ]
                        save_json_safe(results_for_json, results_file)

                except Exception as e:
                    logger.error(f"Failed to process scene {scene_key_tuple}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
        else:
            # Process stitched scenes (original mode)
            for img_file in tqdm(image_files, desc=desc_label):
                try:
                    # Load clean image
                    clean_image = _load_clean_image(img_file, args.device)

                    # Evaluate
                    image_basename = img_file.stem
                    result = evaluate_single_image(
                        clean_image,
                        exposure_ratio,
                        sampler,
                        pg_guidance,
                        gaussian_guidance,
                        poisson_coeff,
                        sigma_r,
                        sensor_range,
                        black_level,
                        args.num_steps,
                        args.kappa,
                        target_signal=target_signal,
                        target_poisson_fraction=target_poisson_fraction,
                        num_frames=args.num_frames,
                        save_restored_images=False,  # We'll save them later grouped by test value
                        output_dir=None,
                        image_name=None,
                    )

                    result["image_file"] = str(img_file.name)
                    result["image_basename"] = image_basename
                    if mode == "poisson_percentage":
                        result["target_poisson_percentage"] = (
                            target_poisson_fraction * 100
                        )
                    ratio_results.append(result)

                    # Clear GPU memory after each image
                    del clean_image
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Failed on {img_file}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

        if len(ratio_results) == 0:
            continue

        # Note: For tile-based mode, results are already saved incrementally in the loop above
        # For non-tile mode, we still need to save here
        if not args.use_tiles:
            all_results.extend(ratio_results)

        # Create visualization after each test value completes (tile-based mode)
        if args.use_tiles and len(ratio_results) > 0:
            stitched_images = {}
            for result in ratio_results:
                if "stitched_long" in result and result["stitched_long"] is not None:
                    # Use first scene for visualization
                    if not stitched_images:
                        stitched_images["long"] = result["stitched_long"]
                        stitched_images["short"] = (
                            result.get("stitched_short")
                            if result.get("stitched_short") is not None
                            else result["stitched_long"]
                        )
                        stitched_images["pg_x0"] = (
                            result.get("stitched_pg")
                            if result.get("stitched_pg") is not None
                            else result["stitched_long"]
                        )
                        stitched_images["gaussian_x0"] = (
                            result.get("stitched_gaussian")
                            if result.get("stitched_gaussian") is not None
                            else result["stitched_long"]
                        )

            if stitched_images:

                def _format_scene_metrics(
                    results: List[Dict], include_tile_id: bool = False
                ) -> List[Dict]:
                    """Format results into scene metrics format."""
                    formatted = []
                    for result in results:
                        metric_entry = {
                            "comprehensive_metrics": {
                                "pg_x0": {
                                    "psnr": result["pg_psnr"],
                                    "ssim": result["pg_ssim"],
                                },
                                "gaussian_x0": {
                                    "psnr": result["gaussian_psnr"],
                                    "ssim": result["gaussian_ssim"],
                                },
                            }
                        }
                        if include_tile_id:
                            metric_entry["tile_id"] = result.get(
                                "image_file", "unknown"
                            )
                        formatted.append(metric_entry)
                    return formatted

                scene_metrics_for_vis = _format_scene_metrics(ratio_results)
                create_scene_visualization(
                    stitched_images,
                    scene_key,
                    args.sensor,
                    scene_dir,
                    scene_metrics_for_vis,
                )

                scene_metrics_for_format = _format_scene_metrics(
                    ratio_results, include_tile_id=True
                )
                aggregate_metrics = compute_scene_aggregate_metrics(
                    scene_metrics_for_format
                )
                save_scene_metrics_json(
                    scene_dir,
                    scene_key,
                    args.sensor,
                    scene_metrics_for_format,
                    aggregate_metrics,
                )

    results_file = output_dir / "synthetic_validation_results.json"
    results_for_json = [
        {k: v for k, v in r.items() if not isinstance(v, torch.Tensor)}
        for r in all_results
    ]
    save_json_safe(results_for_json, results_file)

    create_regime_sweep_visualization(
        all_results, poisson_coeff, sigma_r, output_dir, mode
    )

    # Group results by test value
    results_by_test_value = _group_results_by_test_value(all_results, test_values, mode)

    for test_value, ratio_results in results_by_test_value.items():
        if len(ratio_results) == 0:
            continue

        # Create scene-like directory for this test value
        if mode == "poisson_percentage":
            scene_key = f"synthetic_poisson_{test_value*100:.0f}pct"
        else:
            scene_key = f"synthetic_ratio_{test_value}"

        scene_dir = output_dir / scene_key
        scene_dir.mkdir(exist_ok=True)

        stitched_images = {}
        for result in tqdm(ratio_results, desc=f"Saving {scene_key}"):
            if args.use_tiles:
                # Tile-based mode: results already contain stitched images
                if "stitched_long" in result and result["stitched_long"] is not None:
                    stitched_long = result["stitched_long"]
                    stitched_short = result.get("stitched_short")
                    stitched_pg = result.get("stitched_pg")
                    stitched_gaussian = result.get("stitched_gaussian")

                    # Create unique subdirectory for this scene
                    scene_id = result.get("scene_id", "unknown")
                    scene_subdir = scene_dir / f"scene_{scene_id}"
                    scene_subdir.mkdir(exist_ok=True)

                    # Save stitched images to scene subdirectory
                    torch.save(stitched_long, scene_subdir / "stitched_long.pt")
                    if stitched_short is not None:
                        torch.save(stitched_short, scene_subdir / "stitched_short.pt")
                    if stitched_pg is not None:
                        torch.save(stitched_pg, scene_subdir / "stitched_pg_x0.pt")
                    if stitched_gaussian is not None:
                        torch.save(
                            stitched_gaussian, scene_subdir / "stitched_gaussian_x0.pt"
                        )

                    # Use first scene for visualization (or could aggregate all)
                    if not stitched_images:
                        stitched_images["long"] = stitched_long
                        stitched_images["short"] = (
                            stitched_short
                            if stitched_short is not None
                            else stitched_long
                        )
                        stitched_images["pg_x0"] = (
                            stitched_pg if stitched_pg is not None else stitched_long
                        )
                        stitched_images["gaussian_x0"] = (
                            stitched_gaussian
                            if stitched_gaussian is not None
                            else stitched_long
                        )
                continue

            # Scene-based mode: load and re-process
            img_file = Path(args.test_images_dir)
            if args.use_scenes:
                # For scenes, find the scene directory
                scene_name = Path(result.get("image_file", "")).parent.name
                img_file = img_file / scene_name / "stitched_long.pt"
            else:
                img_file = img_file / result.get("image_file", "")

            if not img_file.exists():
                continue

            try:
                clean_image = _load_clean_image(img_file, args.device)

                # Re-run evaluation with save enabled
                (
                    exposure_ratio,
                    target_signal,
                    target_poisson_fraction,
                ) = _compute_test_parameters(
                    test_value, mode, poisson_coeff, sigma_r_squared
                )

                eval_result = evaluate_single_image(
                    clean_image,
                    exposure_ratio,
                    sampler,
                    pg_guidance,
                    gaussian_guidance,
                    poisson_coeff,
                    sigma_r,
                    sensor_range,
                    black_level,
                    args.num_steps,
                    args.kappa,
                    target_signal=target_signal,
                    target_poisson_fraction=target_poisson_fraction,
                    num_frames=args.num_frames,
                    save_restored_images=False,  # We'll save manually in the correct format
                    output_dir=None,
                    image_name=None,
                )

                # Move to CPU and save immediately
                clean_cpu = clean_image.cpu()
                noisy_cpu = eval_result["noisy_image"].cpu()
                pg_restored_cpu = eval_result["pg_restored"].cpu()
                gaussian_restored_cpu = eval_result["gaussian_restored"].cpu()

                torch.save(clean_cpu, scene_dir / "stitched_long.pt")
                torch.save(noisy_cpu, scene_dir / "stitched_short.pt")
                torch.save(pg_restored_cpu, scene_dir / "stitched_pg_x0.pt")
                torch.save(gaussian_restored_cpu, scene_dir / "stitched_gaussian_x0.pt")

                if not stitched_images:
                    stitched_images["long"] = clean_cpu
                    stitched_images["short"] = noisy_cpu
                    stitched_images["pg_x0"] = pg_restored_cpu
                    stitched_images["gaussian_x0"] = gaussian_restored_cpu

                # Clear GPU memory
                del (
                    clean_image,
                    eval_result,
                    clean_cpu,
                    noisy_cpu,
                    pg_restored_cpu,
                    gaussian_restored_cpu,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed to save restored images for {img_file}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        def _format_scene_metrics(
            results: List[Dict], include_tile_id: bool = False
        ) -> List[Dict]:
            """Format results into scene metrics format."""
            formatted = []
            for result in results:
                metric_entry = {
                    "comprehensive_metrics": {
                        "pg_x0": {
                            "psnr": result["pg_psnr"],
                            "ssim": result["pg_ssim"],
                        },
                        "gaussian_x0": {
                            "psnr": result["gaussian_psnr"],
                            "ssim": result["gaussian_ssim"],
                        },
                    }
                }
                if include_tile_id:
                    metric_entry["tile_id"] = result.get("image_file", "unknown")
                formatted.append(metric_entry)
            return formatted

        if stitched_images:
            scene_metrics_for_vis = _format_scene_metrics(ratio_results)
            create_scene_visualization(
                stitched_images,
                scene_key,
                args.sensor,
                scene_dir,
                scene_metrics_for_vis,
            )
        else:
            # Create a simpler summary visualization for tiles
            create_synthetic_summary_visualization(
                ratio_results, test_value, mode, scene_dir, args.sensor
            )

        scene_metrics_for_format = _format_scene_metrics(
            ratio_results, include_tile_id=True
        )

        aggregate_metrics = compute_scene_aggregate_metrics(scene_metrics_for_format)
        save_scene_metrics_json(
            scene_dir,
            scene_key,
            args.sensor,
            scene_metrics_for_format,
            aggregate_metrics,
        )

    # Create summary
    summary = {}
    for test_value in test_values:
        # Reuse grouping function to get results for this test value
        ratio_results = _group_results_by_test_value(all_results, [test_value], mode)[
            test_value
        ]

        if len(ratio_results) == 0:
            continue

        # Compute key and display_key
        if mode == "poisson_percentage":
            key = f"poisson_{test_value*100:.0f}pct"
            display_key = f"Poisson {test_value*100:.0f}%"
        else:
            key = f"ratio_{test_value}"
            display_key = f"Ratio {test_value}"

        psnr_gains = [r["psnr_gain"] for r in ratio_results]
        ssim_gains = [r["ssim_gain"] for r in ratio_results]
        poisson_fracs = [r["poisson_fraction"] for r in ratio_results]
        regimes = [r.get("regime", "unknown") for r in ratio_results]
        variance_ratios = [r.get("variance_ratio", 0.0) for r in ratio_results]
        num_frames_list = [r.get("num_frames", 1) for r in ratio_results]

        # Count regime distribution
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        t_stat, p_value = stats.ttest_1samp(psnr_gains, 0)

        summary[key] = {
            "test_value": float(test_value),
            "display_name": display_key,
            "n_images": len(ratio_results),
            "num_frames": int(np.mean(num_frames_list)) if num_frames_list else 1,
            "mean_psnr_gain": float(np.mean(psnr_gains)),
            "std_psnr_gain": float(np.std(psnr_gains)),
            "mean_ssim_gain": float(np.mean(ssim_gains)),
            "mean_poisson_fraction": float(np.mean(poisson_fracs)),
            "mean_variance_ratio": float(np.mean(variance_ratios))
            if variance_ratios
            else 0.0,
            "regime_distribution": regime_counts,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    summary_file = output_dir / "summary.json"
    save_json_safe(summary, summary_file)

    if sweep_mode:
        save_sweep_results(
            all_results,
            test_values,
            mode,
            args.sensor,
            output_dir,
            poisson_coeff,
            sigma_r,
            kappa=args.kappa,
            num_steps=args.num_steps,
        )

        sweep_results_file = output_dir / "sweep_results.json"
        if sweep_results_file.exists():
            sweep_results = load_json_safe(sweep_results_file)

            create_sweep_visualization(sweep_results, output_dir)
            best_ratios = find_best_exposure_ratios(sweep_results)

            print("\n" + "=" * 80)
            print("EXPOSURE RATIO SWEEP: BEST RESULTS")
            print("=" * 80)
            if best_ratios:
                if best_ratios.get("best_by_psnr_gain"):
                    best = best_ratios["best_by_psnr_gain"]
                    print("Best by PSNR Gain:")
                    print(f"  Exposure ratio: {best['test_value']:.3f}")
                    print(
                        f"  PSNR gain: {best['psnr_gain']:.3f} dB (p={best['p_value']:.4f})"
                    )
                    print(f"  Poisson fraction: {best['poisson_fraction']*100:.1f}%")
                    print(f"  Variance ratio: {best['variance_ratio']:.4f}")
                    print()

                if best_ratios.get("best_by_pg_psnr"):
                    best = best_ratios["best_by_pg_psnr"]
                    print("Best by PG PSNR:")
                    print(f"  Exposure ratio: {best['test_value']:.3f}")
                    print(f"  PG PSNR: {best['pg_psnr']:.2f} dB")
                    print(f"  PSNR gain: {best['psnr_gain']:.3f} dB")
                    print()

                if best_ratios.get("first_significant"):
                    first = best_ratios["first_significant"]
                    print("First Significant Gain (p < 0.05):")
                    print(f"  Exposure ratio: {first['test_value']:.3f}")
                    print(
                        f"  PSNR gain: {first['psnr_gain']:.3f} dB (p={first['p_value']:.4f})"
                    )
                    print()

            print("Sweep results saved to:")
            print(f"  - JSON: {output_dir / 'sweep_results.json'}")
            print(f"  - CSV: {output_dir / 'sweep_results.csv'}")
            print(f"  - Visualization: {output_dir / 'sweep_analysis.png'}")
            print()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for key, data in sorted(summary.items()):
        display_name = data["display_name"]
        psnr_gain = data["mean_psnr_gain"]
        p_val = data["p_value"]
        poisson_frac = data["mean_poisson_fraction"]
        variance_ratio = data.get("mean_variance_ratio", 0.0)
        regime_dist = data.get("regime_distribution", {})
        num_frames = data.get("num_frames", 1)
        sig_marker = (
            "✓✓✓"
            if p_val < 0.001
            else "✓✓"
            if p_val < 0.01
            else "✓"
            if p_val < 0.05
            else "✗"
        )

        print(f"{display_name}:")
        if num_frames > 1:
            print(
                f"  Multi-frame averaging: {num_frames} frames (read noise reduced by {1/np.sqrt(num_frames):.2f}x)"
            )
        print(f"  Poisson contribution: {poisson_frac*100:.1f}%")
        print(f"  Variance ratio (a·μ/σ_r²): {variance_ratio:.4f}")
        if regime_dist:
            regime_str = ", ".join(
                [f"{k}: {v}" for k, v in sorted(regime_dist.items())]
            )
            print(f"  Noise regime: {regime_str}")
        print(f"  PSNR gain: {psnr_gain:.3f} dB (p={p_val:.4f}) {sig_marker}")
        print()


if __name__ == "__main__":
    main()
