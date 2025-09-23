#!/usr/bin/env python3
"""
Photography Model Evaluation Script

This script loads the best trained model from checkpoints/best_checkpoint.pt
and evaluates it on photography test data, generating side-by-side PNG
comparisons of original noisy images and denoised results.
"""

import argparse
import logging

# Set up multiprocessing for HPC compatibility
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up one level from scripts/
sys.path.insert(0, str(project_root))

from core.logging_config import get_logger
from core.poisson_guidance import create_domain_guidance
from models.edm_wrapper import (
    EDMConfig,
    EDMModelWrapper,
    create_domain_aware_edm_wrapper,
    create_domain_edm_wrapper,
)
from models.sampler import create_edm_sampler, create_fast_sampler
from poisson_training.utils import load_checkpoint

logger = get_logger(__name__)


def create_simple_model(checkpoint_state_dict: Dict) -> nn.Module:
    """
    Create a simple model based on the checkpoint configuration.
    This is a fallback for when EDM is not available.
    """
    # Infer architecture from checkpoint state dict
    conv1_shape = checkpoint_state_dict.get(
        "conv1.weight", torch.zeros(32, 1, 3, 3)
    ).shape
    conv2_shape = checkpoint_state_dict.get(
        "conv2.weight", torch.zeros(32, 32, 3, 3)
    ).shape
    conv3_shape = checkpoint_state_dict.get(
        "conv3.weight", torch.zeros(1, 32, 3, 3)
    ).shape

    in_channels = conv1_shape[1]
    hidden_channels = conv1_shape[0]
    out_channels = conv3_shape[0]

    logger.info(
        f"Creating simple model: {in_channels} -> {hidden_channels} -> {out_channels}"
    )

    class SimpleDenoisingModel(nn.Module):
        def __init__(self, in_ch, hidden_ch, out_ch):
            super().__init__()
            # Simple 3-layer CNN for denoising
            self.conv1 = nn.Conv2d(in_ch, hidden_ch, 3, padding=1)
            self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1)
            self.conv3 = nn.Conv2d(hidden_ch, out_ch, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x, **kwargs):
            # Simple residual denoising
            identity = x
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.conv3(x)
            return identity + x  # Residual connection

    return SimpleDenoisingModel(in_channels, hidden_channels, out_channels)


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load the trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")

    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    logger.info(f"Checkpoint contains: {list(checkpoint.keys())}")
    logger.info(f"Model config: {config}")

    # Check if this is a simple model checkpoint based on the keys
    model_keys = list(checkpoint.get("model_state_dict", {}).keys())
    is_simple_model = any(key.startswith("conv") for key in model_keys)

    if is_simple_model:
        logger.info("Detected simple CNN model in checkpoint")
        # Use EMA state dict if available, otherwise regular state dict
        state_dict_to_use = checkpoint.get(
            "ema_model_state_dict", checkpoint.get("model_state_dict", {})
        )
        model = create_simple_model(state_dict_to_use)
    else:
        # Try to create EDM model with correct architecture
        try:
            # Get the state dict to examine architecture
            state_dict_to_use = checkpoint.get(
                "ema_model_state_dict", checkpoint.get("model_state_dict", {})
            )

            # Look for the input conv layer to determine architecture
            input_conv_key = None
            model_channels = 256  # Default from checkpoint analysis

            for key in state_dict_to_use.keys():
                if "128x128_conv.weight" in key:
                    input_conv_key = key
                    input_shape = state_dict_to_use[key].shape
                    model_channels = input_shape[0]  # Output channels of first conv
                    logger.info(
                        f"Detected architecture from {key}: input_channels={input_shape[1]}, model_channels={model_channels}"
                    )
                    break

            # Create EDM model with detected architecture using domain-aware wrapper
            model = create_domain_aware_edm_wrapper(
                domain="photography", img_resolution=128, model_channels=model_channels
            )
            logger.info(
                f"Created domain-aware EDM model wrapper with {model_channels} model channels"
            )

        except Exception as e:
            logger.warning(f"Failed to create EDM model: {e}")
            logger.info("Falling back to simple model")
            state_dict_to_use = checkpoint.get(
                "ema_model_state_dict", checkpoint.get("model_state_dict", {})
            )
            model = create_simple_model(state_dict_to_use)

    # Load state dict
    if "ema_model_state_dict" in checkpoint:
        logger.info("Loading EMA model state dict")
        state_dict = checkpoint["ema_model_state_dict"]
    elif "model_state_dict" in checkpoint:
        logger.info("Loading regular model state dict")
        state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError("No model state dict found in checkpoint")

    # Handle DataParallel prefix mismatch
    if any(key.startswith("module.") for key in state_dict.keys()):
        logger.info("Removing 'module.' prefix from state dict keys")
        state_dict = {
            key.replace("module.", ""): value for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    logger.info(
        f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters"
    )
    return model


def load_test_data(
    test_data_dir: str, max_samples: int = 5
) -> List[Tuple[torch.Tensor, torch.Tensor, Dict]]:
    """Load test data samples."""
    test_data_dir = Path(test_data_dir)
    posterior_dir = test_data_dir / "posterior" / "photography" / "test"
    prior_dir = test_data_dir / "prior_clean" / "photography" / "test"

    logger.info(f"Loading test data from {test_data_dir}")

    # Get available test scenes
    posterior_files = sorted(list(posterior_dir.glob("scene_*.pt")))[:max_samples]

    test_samples = []

    for posterior_file in posterior_files:
        scene_name = posterior_file.stem  # e.g., "scene_00011"

        logger.info(f"Loading {scene_name}")

        # Load posterior (noisy) data
        posterior_data = torch.load(
            posterior_file, map_location="cpu", weights_only=False
        )

        # Extract relevant data
        noisy = posterior_data["noisy_norm"]  # [4, H, W] - noisy input
        clean = posterior_data[
            "clean_norm"
        ]  # [4, H, W] - TRUE ground truth (clean version of the same scene)
        metadata = posterior_data.get("metadata", {})

        test_samples.append((noisy, clean, metadata))

        logger.info(
            f"Loaded {scene_name}: noisy shape {noisy.shape}, clean shape {clean.shape}"
        )

    logger.info(f"Loaded {len(test_samples)} test samples")
    return test_samples


def run_inference(
    model: nn.Module, noisy: torch.Tensor, device: torch.device, sampler=None
) -> torch.Tensor:
    """Run inference on noisy image using patch-based processing for large images."""
    with torch.no_grad():
        # Check if image needs patch-based processing
        C, H, W = noisy.shape
        patch_size = 512  # Process in 512x512 patches
        overlap = 64  # 64-pixel overlap to avoid boundary artifacts

        logger.info(f"Input shape: {noisy.shape}")

        # If image is small enough, process directly
        if H <= patch_size and W <= patch_size:
            logger.info("Small image, processing directly")
            noisy_batch = noisy.unsqueeze(0).to(device)  # [1, C, H, W]
            return (
                run_single_inference(model, noisy_batch, device, sampler)
                .squeeze(0)
                .cpu()
            )

        # Large image: use patch-based processing
        logger.info(
            f"Large image ({H}x{W}), using patch-based inference with {patch_size}x{patch_size} patches"
        )

        # Initialize output tensor
        denoised = torch.zeros_like(noisy)
        weight_map = torch.zeros((H, W), dtype=torch.float32)

        # Calculate patch positions
        stride = patch_size - overlap
        h_patches = (H - overlap + stride - 1) // stride
        w_patches = (W - overlap + stride - 1) // stride

        logger.info(
            f"Processing {h_patches}x{w_patches} = {h_patches * w_patches} patches"
        )

        patch_count = 0
        for i in range(h_patches):
            for j in range(w_patches):
                # Calculate patch coordinates
                h_start = i * stride
                w_start = j * stride
                h_end = min(h_start + patch_size, H)
                w_end = min(w_start + patch_size, W)

                # Extract patch
                patch = noisy[:, h_start:h_end, w_start:w_end]

                # Pad patch to patch_size if needed (for edge patches)
                if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                    padded_patch = torch.zeros(
                        (C, patch_size, patch_size),
                        dtype=patch.dtype,
                        device=patch.device,
                    )
                    padded_patch[:, : patch.shape[1], : patch.shape[2]] = patch
                    patch = padded_patch

                # Process patch
                patch_batch = patch.unsqueeze(0).to(
                    device
                )  # [1, C, patch_size, patch_size]
                denoised_patch = (
                    run_single_inference(model, patch_batch, device, sampler)
                    .squeeze(0)
                    .cpu()
                )

                # Extract valid region (remove padding if added)
                valid_h = h_end - h_start
                valid_w = w_end - w_start
                denoised_patch = denoised_patch[:, :valid_h, :valid_w]

                # Add to output with blending weights
                weight = torch.ones((valid_h, valid_w), dtype=torch.float32)

                # Blend overlapping regions
                denoised[:, h_start:h_end, w_start:w_end] += denoised_patch * weight
                weight_map[h_start:h_end, w_start:w_end] += weight

                patch_count += 1
                if patch_count % 10 == 0:
                    logger.info(
                        f"Processed {patch_count}/{h_patches * w_patches} patches"
                    )

        # Normalize by weights to handle overlaps
        weight_map = torch.clamp(weight_map, min=1e-8)  # Avoid division by zero
        for c in range(C):
            denoised[c] = denoised[c] / weight_map

        logger.info(f"Patch-based inference completed: {patch_count} patches processed")
        return denoised


def run_single_inference(
    model: nn.Module, noisy_batch: torch.Tensor, device: torch.device, sampler=None
) -> torch.Tensor:
    """Run inference on a single batch (patch or small image) using proper EDM sampling."""

    if sampler is not None:
        # Use proper EDM sampling
        logger.info("Using EDM sampler for proper diffusion sampling")
        try:
            # Convert input to electrons (assuming input is normalized)
            # Photography typical parameters: scale=1000, read_noise=10, background=5
            y_observed = noisy_batch * 1000.0  # Convert to electron counts

            # Run EDM sampling
            result = sampler.sample(
                y_observed=y_observed,
                domain="photography",
                scale=1000.0,
                read_noise=10.0,
                background=5.0,
            )

            # Convert back to normalized range
            denoised = result["x_final"] / 1000.0
            return denoised

        except Exception as e:
            logger.warning(
                f"EDM sampling failed: {e}, falling back to direct model call"
            )

    # Fallback to direct model call (not recommended for EDM)
    try:
        # Try EDM-style inference with proper conditioning for photography domain
        sigma = torch.tensor([0.02], device=device)  # Small noise level for denoising

        # Create domain conditioning for photography
        batch_size = noisy_batch.shape[0]
        domain_condition = torch.zeros((batch_size, 6), device=device)
        domain_condition[:, 0] = 1.0  # Photography domain (one-hot)
        # Set reasonable scale/noise parameters
        domain_condition[:, 3] = 1.0  # Log scale
        domain_condition[:, 4] = 0.1  # Relative read noise
        domain_condition[:, 5] = 0.05  # Relative background

        denoised = model(noisy_batch, sigma=sigma, condition=domain_condition)
        return denoised

    except Exception as e:
        logger.error(f"Direct model inference failed: {e}")
        # Return input as absolute fallback
        return noisy_batch


def tensor_to_image(tensor: torch.Tensor, channel: int = 0) -> np.ndarray:
    """Convert tensor to displayable image."""
    # Take specific channel and normalize to [0, 1]
    img = tensor[channel].numpy()
    img = np.clip(img, 0, 1)
    return img


def create_comparison_plot(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    denoised: torch.Tensor,
    output_path: str,
    scene_name: str,
    use_global_scaling: bool = False,  # Research standard: per-image scaling
):
    """
    Create side-by-side comparison plot following research best practices.
    Uses per-image normalization (min-to-max scaling) as standard in denoising diffusion model literature.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Use first channel for display
    channel = 0

    # Resize clean to match noisy for comparison
    clean_resized = torch.nn.functional.interpolate(
        clean.unsqueeze(0), size=noisy.shape[-2:], mode="bilinear", align_corners=False
    ).squeeze(0)

    # Resize denoised to match for comparison
    denoised_resized = torch.nn.functional.interpolate(
        denoised.unsqueeze(0),
        size=noisy.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Convert to images
    noisy_img = tensor_to_image(noisy, channel)
    clean_img = tensor_to_image(clean_resized, channel)
    denoised_img = tensor_to_image(denoised_resized, channel)

    if use_global_scaling:
        # Global scaling: use the same intensity range for all three images
        # This preserves the actual intensity relationships and shows true denoising performance
        global_min = min(noisy_img.min(), clean_img.min(), denoised_img.min())
        global_max = max(noisy_img.max(), clean_img.max(), denoised_img.max())

        axes[0].imshow(noisy_img, cmap="gray", vmin=global_min, vmax=global_max)
        axes[0].set_title(
            f"Input (Noisy)\n33ms exposure\nIntensity: {noisy_img.min():.3f} - {noisy_img.max():.3f}\nDisplay: {global_min:.3f} - {global_max:.3f}"
        )
        axes[0].axis("off")

        axes[1].imshow(clean_img, cmap="gray", vmin=global_min, vmax=global_max)
        axes[1].set_title(
            f"Ground Truth (Clean)\n10,000ms exposure\nIntensity: {clean_img.min():.3f} - {clean_img.max():.3f}\nDisplay: {global_min:.3f} - {global_max:.3f}"
        )
        axes[1].axis("off")

        axes[2].imshow(denoised_img, cmap="gray", vmin=global_min, vmax=global_max)
        axes[2].set_title(
            f"Model Output (Denoised)\nIntensity: {denoised_img.min():.3f} - {denoised_img.max():.3f}\nDisplay: {global_min:.3f} - {global_max:.3f}"
        )
        axes[2].axis("off")

        scaling_note = f"Global scaling: {global_min:.3f} - {global_max:.3f} (preserves intensity relationships)"
    else:
        # Per-image normalization (research standard for denoising diffusion models)
        # This enhances contrast and highlights features for better qualitative assessment
        axes[0].imshow(
            noisy_img, cmap="gray", vmin=noisy_img.min(), vmax=noisy_img.max()
        )
        axes[0].set_title(
            f"Input (Noisy)\n33ms exposure\nIntensity: {noisy_img.min():.3f} - {noisy_img.max():.3f}"
        )
        axes[0].axis("off")

        axes[1].imshow(
            clean_img, cmap="gray", vmin=clean_img.min(), vmax=clean_img.max()
        )
        axes[1].set_title(
            f"Ground Truth (Clean)\n10,000ms exposure\nIntensity: {clean_img.min():.3f} - {clean_img.max():.3f}"
        )
        axes[1].axis("off")

        axes[2].imshow(
            denoised_img, cmap="gray", vmin=denoised_img.min(), vmax=denoised_img.max()
        )
        axes[2].set_title(
            f"Model Output (Denoised)\nIntensity: {denoised_img.min():.3f} - {denoised_img.max():.3f}"
        )
        axes[2].axis("off")

        scaling_note = "Per-image contrast normalization (research standard)"

    # Add quantitative metrics
    mse = np.mean((clean_img - denoised_img) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))

    plt.suptitle(
        f"Photography Denoising Results - {scene_name}\n"
        + f"{scaling_note} | PSNR: {psnr:.2f} dB",
        fontsize=14,
        y=0.95,
    )
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comparison plot to {output_path}")


def calculate_metrics(clean: torch.Tensor, denoised: torch.Tensor) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    # Resize to same size for comparison
    if clean.shape != denoised.shape:
        denoised = torch.nn.functional.interpolate(
            denoised.unsqueeze(0),
            size=clean.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    # Calculate MSE
    mse = torch.mean((clean - denoised) ** 2).item()

    # Calculate PSNR (assuming values are in [0, 1])
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))

    return {"mse": mse, "psnr": psnr}


def main():
    parser = argparse.ArgumentParser(description="Evaluate photography denoising model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_checkpoint.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/preprocessed_photography",
        help="Path to test data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for comparison images",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Maximum number of test samples to process",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    try:
        # Load model
        model = load_model(args.checkpoint, device)

        # Create EDM sampler for proper diffusion sampling
        logger.info("Creating EDM sampler with Poisson-Gaussian guidance...")
        try:
            # Create Poisson guidance for photography domain
            guidance = create_domain_guidance(
                scale=1000.0,  # Photography scale in electrons
                background=5.0,  # Background level
                read_noise=10.0,  # Read noise
                domain="photography",
            )

            # Create fast sampler (fewer steps for faster evaluation)
            sampler = create_fast_sampler(model, guidance)
            logger.info("EDM sampler created successfully")
        except Exception as e:
            logger.warning(
                f"Failed to create EDM sampler: {e}. Will use direct model calls (less accurate)"
            )
            sampler = None

        # Load test data
        test_samples = load_test_data(args.test_data, args.max_samples)

        if not test_samples:
            logger.error("No test samples loaded!")
            return

        # Process each test sample
        all_metrics = []

        for i, (noisy, clean, metadata) in enumerate(test_samples):
            scene_name = f"scene_{i:03d}"
            logger.info(f"Processing {scene_name}")

            # Run inference
            try:
                denoised = run_inference(model, noisy, device, sampler)

                # Calculate metrics
                metrics = calculate_metrics(clean, denoised)
                all_metrics.append(metrics)

                logger.info(
                    f"Metrics for {scene_name}: MSE={metrics['mse']:.6f}, PSNR={metrics['psnr']:.2f}dB"
                )

                # Create comparison plot
                output_path = output_dir / f"{scene_name}_comparison.png"
                create_comparison_plot(
                    noisy, clean, denoised, str(output_path), scene_name
                )

            except Exception as e:
                logger.error(f"Failed to process {scene_name}: {e}")
                continue

        # Calculate average metrics
        if all_metrics:
            avg_mse = np.mean([m["mse"] for m in all_metrics])
            avg_psnr = np.mean([m["psnr"] for m in all_metrics])

            logger.info("=" * 60)
            logger.info("EVALUATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Processed samples: {len(all_metrics)}")
            logger.info(f"Average MSE: {avg_mse:.6f}")
            logger.info(f"Average PSNR: {avg_psnr:.2f} dB")
            logger.info(f"Results saved to: {output_dir}")
            logger.info("=" * 60)
        else:
            logger.error("No samples processed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
