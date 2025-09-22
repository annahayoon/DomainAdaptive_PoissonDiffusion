#!/usr/bin/env python3
"""
Multi-Domain Model Evaluation Script

This script loads the best trained model from checkpoints/best_checkpoint.pt
and evaluates it on multiple domains (photography, microscopy, astronomy) test data,
generating side-by-side PNG comparisons of original noisy images and denoised results.
"""

import argparse
import json
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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logging_config import get_logger
from models.edm_wrapper import EDMConfig, EDMModelWrapper, create_domain_edm_wrapper
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


def load_model(
    checkpoint_path: str, device: torch.device, domain: str = "photography"
) -> nn.Module:
    """Load the trained model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path} for domain {domain}")

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
        # Try to create EDM model
        try:
            model = create_domain_edm_wrapper(
                domain=domain, img_resolution=128, model_channels=128
            )
            logger.info(f"Created EDM model wrapper for {domain}")
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
        model.load_state_dict(checkpoint["ema_model_state_dict"])
    elif "model_state_dict" in checkpoint:
        logger.info("Loading regular model state dict")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("No model state dict found in checkpoint")

    model.to(device)
    model.eval()

    logger.info(
        f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters"
    )
    return model


def load_test_data(
    test_data_dir: str, domain: str, max_samples: int = 5
) -> List[Tuple[torch.Tensor, torch.Tensor, Dict]]:
    """Load test data samples for a specific domain."""
    test_data_dir = Path(test_data_dir)
    posterior_dir = test_data_dir / "posterior" / domain / "test"
    prior_dir = test_data_dir / "prior_clean" / domain / "test"

    logger.info(f"Loading {domain} test data from {test_data_dir}")

    # Check if domain directory exists
    if not posterior_dir.exists():
        logger.warning(f"No posterior test data found for {domain} at {posterior_dir}")
        return []

    # Get available test scenes
    posterior_files = sorted(list(posterior_dir.glob("*.pt")))[:max_samples]

    # If no test files, try training files as fallback for microscopy/astronomy
    if not posterior_files and domain in ["microscopy", "astronomy"]:
        train_dir = test_data_dir / "posterior" / domain / "train"
        if train_dir.exists():
            logger.info(
                f"No test files found for {domain}, using training samples instead"
            )
            posterior_files = sorted(list(train_dir.glob("*.pt")))[:max_samples]

    if not posterior_files:
        logger.warning(f"No .pt files found in {posterior_dir}")
        return []

    test_samples = []

    for posterior_file in posterior_files:
        scene_name = posterior_file.stem  # e.g., "scene_00011"

        logger.info(f"Loading {scene_name} from {domain}")

        try:
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
        except Exception as e:
            logger.error(f"Failed to load {posterior_file}: {e}")
            continue

    logger.info(f"Loaded {len(test_samples)} test samples for {domain}")
    return test_samples


def run_inference(
    model: nn.Module, noisy: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Run inference on noisy image."""
    with torch.no_grad():
        # Move to device and add batch dimension
        noisy_batch = noisy.unsqueeze(0).to(device)  # [1, C, H, W]

        # Check if model expects single channel input
        if hasattr(model, "conv1") and model.conv1.in_channels == 1:
            # Convert 4-channel to 1-channel by averaging (simple approach)
            if noisy_batch.shape[1] == 4:
                noisy_batch = torch.mean(
                    noisy_batch, dim=1, keepdim=True
                )  # [1, 1, H, W]
                logger.info(f"Converted input from 4 channels to 1 channel for model")

        # For EDM models, we might need additional parameters
        try:
            # Try EDM-style inference first
            denoised = model(noisy_batch, sigma=torch.tensor([0.1], device=device))
        except Exception as e:
            logger.info(f"EDM-style inference failed: {e}, trying simple forward")
            # Fallback to simple forward pass
            denoised = model(noisy_batch)

        # If model outputs single channel but we need 4, replicate
        if denoised.shape[1] == 1 and noisy.shape[0] == 4:
            denoised = denoised.repeat(1, 4, 1, 1)  # [1, 4, H, W]
            logger.info("Replicated single channel output to 4 channels")

        return denoised.squeeze(0).cpu()  # [C, H, W]


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
    domain: str,
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

    # Per-image normalization (research standard for denoising diffusion models)
    # This enhances contrast and highlights features for better qualitative assessment
    axes[0].imshow(noisy_img, cmap="gray", vmin=noisy_img.min(), vmax=noisy_img.max())
    axes[0].set_title(
        f"Input (Noisy)\nIntensity: {noisy_img.min():.3f} - {noisy_img.max():.3f}"
    )
    axes[0].axis("off")

    axes[1].imshow(clean_img, cmap="gray", vmin=clean_img.min(), vmax=clean_img.max())
    axes[1].set_title(
        f"Ground Truth (Clean)\nIntensity: {clean_img.min():.3f} - {clean_img.max():.3f}"
    )
    axes[1].axis("off")

    axes[2].imshow(
        denoised_img, cmap="gray", vmin=denoised_img.min(), vmax=denoised_img.max()
    )
    axes[2].set_title(
        f"Model Output (Denoised)\nIntensity: {denoised_img.min():.3f} - {denoised_img.max():.3f}"
    )
    axes[2].axis("off")

    # Add quantitative metrics
    mse = np.mean((clean_img - denoised_img) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-8))

    plt.suptitle(
        f"{domain.title()} Denoising Results - {scene_name}\n"
        + f"Per-image contrast normalization (research standard) | PSNR: {psnr:.2f} dB",
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


def evaluate_domain(
    model: nn.Module,
    domain: str,
    test_data_dir: str,
    output_dir: Path,
    max_samples: int,
    device: torch.device,
) -> Dict[str, any]:
    """Evaluate model on a specific domain."""
    logger.info(f"Evaluating {domain} domain")

    # Load test data for this domain
    test_samples = load_test_data(test_data_dir, domain, max_samples)

    if not test_samples:
        logger.warning(f"No test samples available for {domain}")
        return {
            "domain": domain,
            "samples_processed": 0,
            "avg_mse": None,
            "avg_psnr": None,
            "results": [],
        }

    # Create domain output directory
    domain_output_dir = output_dir / domain
    domain_output_dir.mkdir(exist_ok=True)

    # Process each test sample
    all_metrics = []
    results = []

    for i, (noisy, clean, metadata) in enumerate(test_samples):
        scene_name = f"{domain}_scene_{i:03d}"
        logger.info(f"Processing {scene_name}")

        # Run inference
        try:
            denoised = run_inference(model, noisy, device)

            # Calculate metrics
            metrics = calculate_metrics(clean, denoised)
            all_metrics.append(metrics)

            logger.info(
                f"Metrics for {scene_name}: MSE={metrics['mse']:.6f}, PSNR={metrics['psnr']:.2f}dB"
            )

            # Create comparison plot
            output_path = domain_output_dir / f"{scene_name}_comparison.png"
            create_comparison_plot(
                noisy, clean, denoised, str(output_path), scene_name, domain
            )

            results.append(
                {
                    "scene_name": scene_name,
                    "mse": metrics["mse"],
                    "psnr": metrics["psnr"],
                }
            )

        except Exception as e:
            logger.error(f"Failed to process {scene_name}: {e}")
            continue

    # Calculate average metrics
    if all_metrics:
        avg_mse = np.mean([m["mse"] for m in all_metrics])
        avg_psnr = np.mean([m["psnr"] for m in all_metrics])

        logger.info(f"{domain.upper()} DOMAIN RESULTS:")
        logger.info(f"  Samples processed: {len(all_metrics)}")
        logger.info(f"  Average MSE: {avg_mse:.6f}")
        logger.info(f"  Average PSNR: {avg_psnr:.2f} dB")

        return {
            "domain": domain,
            "samples_processed": len(all_metrics),
            "avg_mse": avg_mse,
            "avg_psnr": avg_psnr,
            "results": results,
        }
    else:
        logger.error(f"No samples processed successfully for {domain}!")
        return {
            "domain": domain,
            "samples_processed": 0,
            "avg_mse": None,
            "avg_psnr": None,
            "results": [],
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-domain denoising model"
    )
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
        help="Maximum number of test samples to process per domain",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["photography", "microscopy", "astronomy"],
        help="Domains to evaluate",
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
        # Load model (using photography as default domain for model loading)
        model = load_model(args.checkpoint, device, domain="photography")

        # Evaluate each domain
        all_results = []

        for domain in args.domains:
            domain_results = evaluate_domain(
                model, domain, args.test_data, output_dir, args.max_samples, device
            )
            all_results.append(domain_results)

        # Save comprehensive results
        results_summary = {
            "checkpoint": args.checkpoint,
            "test_data": args.test_data,
            "device": str(device),
            "evaluation_date": str(Path().cwd()),
            "domains_evaluated": all_results,
        }

        results_file = output_dir / "evaluation_summary.json"
        with open(results_file, "w") as f:
            json.dump(results_summary, f, indent=2)

        # Print final summary
        logger.info("=" * 80)
        logger.info("MULTI-DOMAIN EVALUATION SUMMARY")
        logger.info("=" * 80)

        for result in all_results:
            if result["samples_processed"] > 0:
                logger.info(
                    f"{result['domain'].upper():>12}: {result['samples_processed']:>3} samples | "
                    f"MSE: {result['avg_mse']:.6f} | PSNR: {result['avg_psnr']:.2f} dB"
                )
            else:
                logger.info(f"{result['domain'].upper():>12}: No samples available")

        logger.info("=" * 80)
        logger.info(f"Detailed results saved to: {results_file}")
        logger.info(f"Comparison images saved to: {output_dir}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
