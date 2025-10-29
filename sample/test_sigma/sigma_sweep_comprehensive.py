#!/usr/bin/env python3
"""
Comprehensive Sigma Sweep with Single Input Comparison

This script tests multiple sigma values on the same noisy input image and creates
a comprehensive comparison visualization showing how denoising quality varies with sigma.

Key Features:
- Uses the same noisy input for all sigma values
- Tests wide range of sigma values (0.000001 to 0.01)
- Creates comprehensive comparison visualization
- Saves all results in organized structure

Usage:
    python sample/sigma_sweep_comprehensive.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/photography/noisy \
        --output_dir results/comprehensive_sigma_sweep \
        --domain photography \
        --num_examples 3 \
        --num_steps 18
"""

import argparse
import json
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
from tqdm import tqdm

# Add project root and EDM to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

# Setup logging
import logging

# Import EDM components
import external.edm.dnnlib
from external.edm.torch_utils import distributed as dist

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_tiles(
    metadata_json: Path, domain: str, split: str = "test"
) -> List[Dict[str, Any]]:
    """Load test tile metadata from JSON file."""
    logger.info(f"Loading {split} tiles for {domain} from {metadata_json}")

    with open(metadata_json, "r") as f:
        metadata = json.load(f)

    # Filter tiles by domain and split
    tiles = metadata.get("tiles", [])
    filtered_tiles = [
        tile
        for tile in tiles
        if tile.get("domain") == domain and tile.get("split") == split
    ]

    logger.info(f"Found {len(filtered_tiles)} {split} tiles for {domain}")
    return filtered_tiles


def load_noisy_image(
    tile_id: str, noisy_dir: Path, device: torch.device
) -> torch.Tensor:
    """Load a noisy .pt file."""
    noisy_path = noisy_dir / f"{tile_id}.pt"

    if not noisy_path.exists():
        raise FileNotFoundError(f"Noisy file not found: {noisy_path}")

    # Load the noisy image
    noisy_tensor = torch.load(noisy_path, map_location=device)

    # Handle different tensor formats
    if isinstance(noisy_tensor, dict):
        if "noisy" in noisy_tensor:
            noisy_tensor = noisy_tensor["noisy"]
        elif "noisy_norm" in noisy_tensor:
            noisy_tensor = noisy_tensor["noisy_norm"]
        elif "image" in noisy_tensor:
            noisy_tensor = noisy_tensor["image"]
        else:
            raise ValueError(f"Unrecognized dict structure in {noisy_path}")

    # Ensure float32
    noisy_tensor = noisy_tensor.float()

    # Ensure CHW format
    if noisy_tensor.ndim == 2:  # (H, W)
        noisy_tensor = noisy_tensor.unsqueeze(0)  # (1, H, W)
    elif noisy_tensor.ndim == 3 and noisy_tensor.shape[-1] in [1, 3]:  # (H, W, C)
        noisy_tensor = noisy_tensor.permute(2, 0, 1)  # (C, H, W)

    logger.debug(
        f"Loaded noisy image: {tile_id}, shape={noisy_tensor.shape}, range=[{noisy_tensor.min():.3f}, {noisy_tensor.max():.3f}]"
    )

    return noisy_tensor


class EDMDenoiser:
    """EDM denoiser for comprehensive sigma testing."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        domain_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize the denoiser with trained model and domain configurations."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.domain_ranges = domain_ranges or {
            "photography": {"min": 0.0, "max": 15871.0},
            "microscopy": {"min": 0.0, "max": 65535.0},
            "astronomy": {"min": -65.0, "max": 385.0},
        }

        # Load model
        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.net = checkpoint["ema"].to(self.device)
        self.net.eval()

        logger.info("✓ Model loaded successfully")
        logger.info(f"  Resolution: {self.net.img_resolution}")
        logger.info(f"  Channels: {self.net.img_channels}")
        logger.info(f"  Label dim: {self.net.label_dim}")
        logger.info(f"  Sigma range: [{self.net.sigma_min}, {self.net.sigma_max}]")

    def denoise(
        self,
        noisy_image: torch.Tensor,
        noise_sigma: float,
        class_labels: Optional[torch.Tensor] = None,
        num_steps: int = 18,
        rho: float = 7.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Denoise an image starting from known/estimated noise level."""
        logger.info(f"Denoising from sigma={noise_sigma:.6f} to sigma=0.0")

        # Set up noise schedule from noise_sigma to sigma_min
        sigma_min = max(0.002, self.net.sigma_min)
        sigma_max = min(noise_sigma, self.net.sigma_max)

        # Time step discretization
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
            f"Time steps: {t_steps[0]:.6f} -> {t_steps[-1]:.6f} ({len(t_steps)-1} steps)"
        )

        # Start from the noisy image
        x = noisy_image.to(torch.float64).to(self.device)

        # Denoising loop
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # Get model prediction
            denoised = self.net(x, t_cur, class_labels).to(torch.float64)

            # Euler step
            d_cur = (x - denoised) / t_cur
            x_next = x + (t_next - t_cur) * d_cur

            # Heun's 2nd order correction
            if i < num_steps - 1:
                denoised_next = self.net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised_next) / t_next
                x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

            x = x_next

        denoised_output = x

        logger.info(
            f"✓ Denoising completed: range [{denoised_output.min():.3f}, {denoised_output.max():.3f}]"
        )

        results = {
            "final": denoised_output,
            "initial": noisy_image,
            "noise_sigma": noise_sigma,
            "num_steps": num_steps,
        }

        return denoised_output, results

    def denormalize_to_physical(self, tensor: torch.Tensor, domain: str) -> np.ndarray:
        """Convert tensor from [-1,1] model space to physical units."""
        domain_range = self.domain_ranges.get(domain, {"min": 0.0, "max": 1.0})

        # Step 1: [-1,1] → [0,1]
        tensor_norm = (tensor + 1.0) / 2.0
        tensor_norm = torch.clamp(tensor_norm, 0, 1)

        # Step 2: [0,1] → [domain_min, domain_max]
        domain_min = domain_range["min"]
        domain_max = domain_range["max"]
        tensor_phys = tensor_norm * (domain_max - domain_min) + domain_min

        return tensor_phys.cpu().numpy()


def create_comprehensive_comparison(
    noisy_image: torch.Tensor,
    sigma_results: Dict[float, torch.Tensor],
    domain: str,
    tile_id: str,
    save_path: Path,
):
    """Create comprehensive sigma comparison visualization."""
    logger.info("Creating comprehensive sigma comparison visualization...")

    # Sort sigma values
    sigmas = sorted(sigma_results.keys())
    n_sigmas = len(sigmas)

    # Create figure with two rows: model space and physical units
    fig, axes = plt.subplots(2, n_sigmas + 1, figsize=(3 * (n_sigmas + 1), 6))

    # Denormalize to physical units
    domain_ranges = {
        "photography": {"min": 0.0, "max": 15871.0},
        "microscopy": {"min": 0.0, "max": 65535.0},
        "astronomy": {"min": -65.0, "max": 385.0},
    }

    def denormalize_to_physical(tensor, domain):
        domain_range = domain_ranges.get(domain, {"min": 0.0, "max": 1.0})
        tensor_norm = (tensor + 1.0) / 2.0
        tensor_norm = torch.clamp(tensor_norm, 0, 1)
        domain_min = domain_range["min"]
        domain_max = domain_range["max"]
        tensor_phys = tensor_norm * (domain_max - domain_min) + domain_min
        return tensor_phys.cpu().numpy()

    noisy_physical = denormalize_to_physical(noisy_image, domain)

    # Display normalization function (percentile-based)
    def norm_display_percentile(x):
        """Apply percentile normalization for display."""
        if x.ndim == 4:  # (B, C, H, W)
            if x.shape[1] == 3:  # RGB
                img = np.transpose(x[0], (1, 2, 0))
            else:  # Grayscale or multi-channel
                img = x[0, 0]
        elif x.ndim == 3:  # (C, H, W)
            if x.shape[0] == 3:  # RGB
                img = np.transpose(x, (1, 2, 0))
            else:
                img = x[0]
        else:
            img = x

        valid_mask = np.isfinite(img)
        if np.any(valid_mask):
            p_low, p_high = np.percentile(img[valid_mask], [1, 99])
            img_clipped = np.clip(img, p_low, p_high)
            img_norm = (img_clipped - p_low) / (p_high - p_low + 1e-8)
        else:
            img_norm = np.zeros_like(img)
        return img_norm, img.min(), img.max()

    # Row 1: Model space [-1, 1]
    # Noisy reference
    def norm_display_model(x):
        x = x[0].cpu().numpy()
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))
        else:
            x = x[0]
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x

    axes[0, 0].imshow(
        norm_display_model(noisy_image),
        cmap="gray" if noisy_image.shape[1] == 1 else None,
    )
    axes[0, 0].set_title(
        f"Noisy Input\nRange: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]",
        fontsize=9,
    )
    axes[0, 0].axis("off")

    # Denoised results for each sigma
    for i, sigma in enumerate(sigmas):
        denoised_tensor = sigma_results[sigma]
        denoised_display = norm_display_model(denoised_tensor)

        axes[0, i + 1].imshow(
            denoised_display, cmap="gray" if denoised_display.ndim == 2 else None
        )
        axes[0, i + 1].set_title(
            f"σ = {sigma:.6f}\nRange: [{denoised_tensor.min():.3f}, {denoised_tensor.max():.3f}]",
            fontsize=9,
        )
        axes[0, i + 1].axis("off")

    # Row 2: Physical units with percentile normalization
    # Noisy reference
    noisy_display, noisy_min, noisy_max = norm_display_percentile(noisy_physical)
    axes[1, 0].imshow(noisy_display, cmap="gray" if noisy_display.ndim == 2 else None)
    axes[1, 0].set_title(
        f"Noisy Input\nRange: [{noisy_min:.1f}, {noisy_max:.1f}]\nDomain: {domain}",
        fontsize=9,
    )
    axes[1, 0].axis("off")

    # Denoised results for each sigma
    for i, sigma in enumerate(sigmas):
        denoised_tensor = sigma_results[sigma]
        denoised_physical = denormalize_to_physical(denoised_tensor, domain)
        denoised_display, denoised_min, denoised_max = norm_display_percentile(
            denoised_physical
        )

        axes[1, i + 1].imshow(
            denoised_display, cmap="gray" if denoised_display.ndim == 2 else None
        )
        axes[1, i + 1].set_title(
            f"σ = {sigma:.6f}\nRange: [{denoised_min:.1f}, {denoised_max:.1f}]",
            fontsize=9,
        )
        axes[1, i + 1].axis("off")

    # Add row labels
    axes[0, 0].set_ylabel("Model Space\n[-1, 1]", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel(
        "Physical Units\n+ Display Norm", fontsize=10, fontweight="bold"
    )

    # Main title
    plt.suptitle(
        f"EDM Denoising Sigma Sweep - {tile_id}\n"
        f"Domain: {domain} | Testing σ ∈ [{min(sigmas):.6f}, {max(sigmas):.6f}]",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Comprehensive comparison saved: {save_path}")


def main():
    """Main function for comprehensive sigma sweep."""
    parser = argparse.ArgumentParser(
        description="Comprehensive EDM sigma sweep with single input comparison"
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

    # Denoising arguments
    parser.add_argument(
        "--num_steps", type=int, default=18, help="Number of denoising steps"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="photography",
        choices=["photography", "microscopy", "astronomy"],
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="results/comprehensive_sigma_sweep"
    )
    parser.add_argument(
        "--num_examples", type=int, default=3, help="Number of example images to test"
    )

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EDM SIGMA SWEEP")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Metadata: {args.metadata_json}")
    logger.info(f"Noisy dir: {args.noisy_dir}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Denoising steps: {args.num_steps}")
    logger.info(f"Number of examples: {args.num_examples}")
    logger.info("=" * 80)

    # Initialize denoiser
    denoiser = EDMDenoiser(
        model_path=args.model_path,
        device=args.device,
    )

    # Load test tiles
    test_tiles = load_test_tiles(Path(args.metadata_json), args.domain, split="test")

    if len(test_tiles) == 0:
        logger.error(f"No test tiles found for domain {args.domain}")
        return

    # Filter tiles to only those that exist in the noisy directory
    available_tiles = []
    for tile_info in test_tiles:
        tile_id = tile_info["tile_id"]
        noisy_path = Path(args.noisy_dir) / f"{tile_id}.pt"
        if noisy_path.exists():
            available_tiles.append(tile_info)

    logger.info(f"Found {len(available_tiles)} tiles with corresponding noisy files")

    if len(available_tiles) == 0:
        logger.error(
            f"No tiles found with corresponding noisy files in {args.noisy_dir}"
        )
        return

    # Randomly select from available tiles
    rng = np.random.RandomState(args.seed)
    selected_indices = rng.choice(
        len(available_tiles),
        size=min(args.num_examples, len(available_tiles)),
        replace=False,
    )
    selected_tiles = [available_tiles[i] for i in selected_indices]

    logger.info(f"Selected {len(selected_tiles)} test tiles for sigma sweep")

    # Create domain labels
    if denoiser.net.label_dim > 0:
        class_labels = torch.zeros(1, denoiser.net.label_dim, device=denoiser.device)
        if args.domain == "photography":
            class_labels[:, 0] = 1.0
        elif args.domain == "microscopy":
            class_labels[:, 1] = 1.0
        elif args.domain == "astronomy":
            class_labels[:, 2] = 1.0
    else:
        class_labels = None

    # Define sigma values to test (ultra-low range for very clean images)
    sigma_values = [
        0.0000001,
        0.0000002,
        0.0000005,
        0.000001,
        0.000002,
        0.000005,
        0.00001,
        0.00002,
        0.00005,
        0.0001,
    ]

    logger.info(f"Testing sigma values: {sigma_values}")

    # Process each selected tile
    for idx, tile_info in enumerate(selected_tiles):
        tile_id = tile_info["tile_id"]
        logger.info(f"\n🎯 Processing example {idx+1}/{len(selected_tiles)}: {tile_id}")

        try:
            # Load noisy image
            noisy_image = load_noisy_image(
                tile_id, Path(args.noisy_dir), denoiser.device
            )
            if noisy_image.ndim == 3:
                noisy_image = noisy_image.unsqueeze(0)
            noisy_image = noisy_image.to(torch.float32)

            logger.info(
                f"  Noisy range: [{noisy_image.min():.4f}, {noisy_image.max():.4f}], std={noisy_image.std():.4f}"
            )

            # Test all sigma values on the same noisy input
            sigma_results = {}

            for sigma in tqdm(sigma_values, desc=f"Testing sigmas for {tile_id}"):
                try:
                    # Denoise
                    denoised, results = denoiser.denoise(
                        noisy_image,
                        noise_sigma=sigma,
                        class_labels=class_labels,
                        num_steps=args.num_steps,
                    )

                    sigma_results[sigma] = denoised

                except Exception as e:
                    logger.warning(f"Failed to denoise with sigma={sigma}: {e}")
                    continue

            # Create comprehensive comparison
            comparison_path = (
                output_dir / f"example_{idx:02d}_{tile_id}_sigma_sweep.png"
            )
            try:
                create_comprehensive_comparison(
                    noisy_image=noisy_image,
                    sigma_results=sigma_results,
                    domain=args.domain,
                    tile_id=tile_id,
                    save_path=comparison_path,
                )
            except Exception as e:
                logger.error(f"Failed to create comparison visualization: {e}")
                # Continue without visualization

            # Save tensors for reference
            sample_dir = output_dir / f"example_{idx:02d}_{tile_id}"
            sample_dir.mkdir(exist_ok=True)

            torch.save(noisy_image.cpu(), sample_dir / "noisy.pt")

            # Save denoised results for each sigma
            for sigma, denoised in sigma_results.items():
                torch.save(
                    denoised.cpu(), sample_dir / f"denoised_sigma_{sigma:.6f}.pt"
                )

            logger.info(f"✓ Example {idx+1} completed and saved to {sample_dir}")

        except Exception as e:
            logger.error(f"Failed to process tile {tile_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    logger.info("=" * 80)
    logger.info("🎉 COMPREHENSIVE SIGMA SWEEP COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"📁 All results saved to: {output_dir}")
    logger.info(
        f"📊 Tested {len(sigma_values)} sigma values on {len(selected_tiles)} examples"
    )
    logger.info(
        f"🎯 Each example shows: noisy input → denoised outputs across sigma range"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
