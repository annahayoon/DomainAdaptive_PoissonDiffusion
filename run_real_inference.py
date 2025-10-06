#!/usr/bin/env python3
"""
Real Inference Script for Photography Model

Runs actual diffusion sampling on PNG images from the dataset
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Setup
sys.path.insert(0, str(Path(__file__).parent))
from core.logging_config import get_logger
from core.poisson_guidance import create_domain_guidance
from models.edm_wrapper import EDMConfig, create_domain_aware_edm_wrapper
from models.sampler import create_edm_sampler
from utils import load_checkpoint

logger = get_logger(__name__)


def load_image(path):
    """Load image as tensor [1, C, H, W] in range [0, 1]"""
    img = Image.open(path)
    img_array = np.array(img).astype(np.float32) / 255.0

    if img_array.ndim == 2:
        img_array = img_array[..., np.newaxis]

    # Convert to CHW
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def save_image(tensor, path):
    """Save tensor as image"""
    # Ensure tensor is on CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert from [1, C, H, W] to [H, W, C]
    img_array = tensor.squeeze(0).permute(1, 2, 0).numpy()

    # Clip and convert to uint8
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)

    # Save
    img = Image.fromarray(img_array)
    img.save(path)


def save_comparison(noisy, restored, clean, output_path, metrics=None):
    """Save side-by-side comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convert to numpy for display
    noisy_np = noisy.squeeze(0).permute(1, 2, 0).cpu().numpy()
    restored_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
    clean_np = clean.squeeze(0).permute(1, 2, 0).cpu().numpy()

    axes[0].imshow(noisy_np)
    axes[0].set_title("Noisy (0.1s)")
    axes[0].axis("off")

    axes[1].imshow(np.clip(restored_np, 0, 1))
    title = "Restored (EDM+PG)"
    if metrics and "psnr" in metrics:
        title += f'\nPSNR: {metrics["psnr"]:.2f} dB'
    axes[1].set_title(title)
    axes[1].axis("off")

    axes[2].imshow(clean_np)
    axes[2].set_title("Clean (10s)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_psnr(pred, target):
    """Compute PSNR"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def compute_ssim(pred, target):
    """Simplified SSIM computation"""
    # This is a simplified version - for production use pytorch_ssim or skimage
    c1 = 0.01**2
    c2 = 0.03**2

    mu1 = pred.mean()
    mu2 = target.mean()

    sigma1_sq = pred.var()
    sigma2_sq = target.var()
    sigma12 = ((pred - mu1) * (target - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return ssim.item()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load the trained model"""
    logger.info(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = load_checkpoint(checkpoint_path, device=device)

    config = checkpoint.get("config", {})
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Global step: {checkpoint.get('global_step', 'unknown')}")

    # Detect architecture from checkpoint
    model_state = checkpoint.get("ema_model_state_dict") or checkpoint.get(
        "model_state_dict"
    )
    if model_state is None:
        raise ValueError("No model state found in checkpoint")

    logger.info(
        "  Using EMA model"
        if "ema_model_state_dict" in checkpoint
        else "  Using regular model"
    )

    # Infer input channels and model channels
    first_conv_key = None
    for key in model_state.keys():
        if "conv.weight" in key and len(model_state[key].shape) == 4:
            first_conv_key = key
            break

    if first_conv_key:
        conv_weight = model_state[first_conv_key]
        input_channels = conv_weight.shape[1]
        model_channels = conv_weight.shape[0]
        logger.info(
            f"  Detected: input_channels={input_channels}, model_channels={model_channels}"
        )
    else:
        input_channels = 3
        model_channels = 256
        logger.warning(
            f"  Could not detect architecture, using defaults: {input_channels}, {model_channels}"
        )

    # Create model wrapper
    logger.info("Creating EDM model wrapper...")
    model = create_domain_aware_edm_wrapper(
        domain="photography",
        img_resolution=256,
        model_channels=model_channels,
        conditioning_mode="class_labels",
        channel_mult=[1, 2, 2, 2],
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=[16],
        dropout=0.1,
    )

    # Move to device
    model = model.to(device)

    # Load state dict
    logger.info("Loading model weights...")
    model.load_state_dict(model_state)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters")

    return model, config


@torch.no_grad()
def run_inference(model, noisy_image, sampler, device):
    """Run diffusion sampling to restore image"""
    noisy_image = noisy_image.to(device)

    # Create domain conditioning for photography (0.1s exposure)
    # [exposure_time, ISO, scale_s, read_noise_sigma, has_clean, domain_id]
    domain_condition = torch.tensor(
        [[0.1, 800.0, 1000.0, 10.0, 0.0, 0.0]],  # Photography domain
        device=device,
        dtype=torch.float32,
    )

    logger.info("Running diffusion sampling...")

    # Run sampling
    restored = sampler.sample(
        noisy=noisy_image, model=model, domain_condition=domain_condition
    )

    return restored


def main():
    print("=" * 80)
    print("REAL INFERENCE TEST - Photography Model")
    print("=" * 80)

    # Configuration - Using fully trained model (225k steps)
    checkpoint_path = Path(
        "results/research_steps_20250924_082658/checkpoints/best_model.pt"
    )
    noisy_dir = Path("dataset/processed/png_tiles/photography/noisy")
    clean_dir = Path("dataset/processed/png_tiles/photography/clean")
    output_dir = Path("results/real_inference_step225k")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use GPU now that it's free
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, device)

    # Create sampler with PG guidance
    logger.info("Creating EDM sampler with Poisson-Gaussian guidance...")

    guidance = create_domain_guidance(
        guidance_type="pg", scale_s=1000.0, read_noise_sigma=10.0, device=device
    )

    sampler = create_edm_sampler(
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        num_steps=18,  # Standard sampling steps
        solver="euler",
        guidance=guidance,
        guidance_scale=0.8,
        device=device,
    )

    logger.info(f"Sampler created: {sampler.num_steps} steps")

    # Select test scenes
    test_scenes = [
        "photography_fuji_00001_00",
        "photography_fuji_00010_00",
        "photography_fuji_00020_00",
    ]

    logger.info(f"\nProcessing {len(test_scenes)} test scenes...")

    results = []

    for i, scene in enumerate(test_scenes, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Scene {i}/{len(test_scenes)}: {scene}")
        logger.info("=" * 60)

        noisy_file = f"{scene}_0.1s_tile_0000.png"
        clean_file = f"{scene}_10s_tile_0000.png"

        noisy_path = noisy_dir / noisy_file
        clean_path = clean_dir / clean_file

        if not (noisy_path.exists() and clean_path.exists()):
            logger.warning(f"Skipping (files missing)")
            continue

        # Load images
        noisy = load_image(noisy_path).to(device)
        clean = load_image(clean_path).to(device)

        logger.info(f"Loaded: {noisy.shape}")

        # Input metrics
        input_psnr = compute_psnr(noisy, clean)
        logger.info(f"Input PSNR: {input_psnr:.2f} dB")

        # Run inference
        restored = run_inference(model, noisy, sampler, device)

        # Compute metrics
        output_psnr = compute_psnr(restored, clean)
        output_ssim = compute_ssim(restored, clean)

        logger.info(
            f"✓ Output PSNR: {output_psnr:.2f} dB (Δ = {output_psnr - input_psnr:+.2f} dB)"
        )
        logger.info(f"✓ Output SSIM: {output_ssim:.4f}")

        metrics = {
            "psnr": output_psnr,
            "ssim": output_ssim,
            "input_psnr": input_psnr,
            "improvement": output_psnr - input_psnr,
        }

        # Save outputs
        comparison_path = output_dir / f"comparison_{i:02d}_{scene}.png"
        restored_path = output_dir / f"restored_{i:02d}_{scene}.png"

        save_comparison(noisy, restored, clean, comparison_path, metrics)
        save_image(restored, restored_path)

        logger.info(f"Saved: {comparison_path.name}")
        logger.info(f"Saved: {restored_path.name}")

        results.append(
            {
                "scene": scene,
                "input_psnr": input_psnr,
                "output_psnr": output_psnr,
                "ssim": output_ssim,
                "improvement": output_psnr - input_psnr,
            }
        )

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info("=" * 80)

    if results:
        avg_input_psnr = np.mean([r["input_psnr"] for r in results])
        avg_output_psnr = np.mean([r["output_psnr"] for r in results])
        avg_ssim = np.mean([r["ssim"] for r in results])
        avg_improvement = np.mean([r["improvement"] for r in results])

        logger.info(f"Scenes processed: {len(results)}")
        logger.info(f"Average Input PSNR:  {avg_input_psnr:.2f} dB")
        logger.info(f"Average Output PSNR: {avg_output_psnr:.2f} dB")
        logger.info(f"Average Improvement: {avg_improvement:+.2f} dB")
        logger.info(f"Average SSIM:        {avg_ssim:.4f}")
        logger.info(f"\nOutput directory: {output_dir}")
    else:
        logger.error("No scenes processed!")

    logger.info("=" * 80)
    logger.info("✓ INFERENCE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
