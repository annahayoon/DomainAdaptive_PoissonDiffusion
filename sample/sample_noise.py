#!/usr/bin/env python3
"""
Comprehensive EDM Sampling with Step-by-Step Visualizations

This script demonstrates the complete EDM sampling process with detailed visualizations:

1. Native EDM Sampling (starting from pure noise)
2. Step-by-step denoising visualization
3. Domain-specific denormalization
4. Physical units conversion
5. Display-ready visualization

Key Features:
- Starts from pure noise (native EDM generation)
- Shows every denoising step with visualizations
- Handles domain-specific normalization ranges
- Creates comprehensive comparison panels
- Saves intermediate results for analysis

Usage:
    python sample_noise.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --output_dir results/edm_noise_sampling_photography\
        --num_steps 18 \
        --domain photography \
        --num_samples 4
        --batch_size 4
        --sigma_max 80.0
        --rho 7.0
        --seed 42
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
project_root = Path(__file__).parent.parent  # Go up from sample/ to project root
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


class ComprehensiveEDMSampler:
    """
    Comprehensive EDM sampler with step-by-step visualization capabilities.

    This sampler starts from pure noise and generates samples using EDM's native
    sampling process, with detailed visualization of each step.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        domain_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize the sampler with trained model and domain configurations."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.domain_ranges = domain_ranges or {
            "photography": {"min": 0.0, "max": 15871.0},
            "microscopy": {"min": 0.0, "max": 65535.0},
            "astronomy": {"min": -65.0, "max": 385.0},
        }

        # Load model
        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)  # nosec B301

        self.net = checkpoint["ema"].to(self.device)
        self.net.eval()

        logger.info("✓ Model loaded successfully")
        logger.info(f"  Resolution: {self.net.img_resolution}")
        logger.info(f"  Channels: {self.net.img_channels}")
        logger.info(f"  Label dim: {self.net.label_dim}")
        logger.info(f"  Sigma range: [{self.net.sigma_min}, {self.net.sigma_max}]")

    def sample_native_edm(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        randn_like: callable = torch.randn_like,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
        save_steps: bool = True,
        output_dir: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform native EDM sampling starting from pure noise.

        Args:
            batch_size: Number of samples to generate
            class_labels: Class conditioning (domain labels)
            latents: Initial latent noise tensor (if None, will be generated)
            randn_like: Random noise function (default: torch.randn_like)
            num_steps: Number of sampling steps
            sigma_min, sigma_max: Noise level range
            rho: Time step discretization parameter
            S_churn, S_min, S_max, S_noise: Stochasticity parameters
            save_steps: Whether to save intermediate steps
            output_dir: Directory to save step visualizations

        Returns:
            Tuple of (final_tensor, results_dict):
                - final_tensor: Final sampled image tensor
                - results_dict: Dictionary containing all sampling results and intermediate steps
        """
        logger.info(
            f"Starting native EDM sampling: {num_steps} steps, batch_size={batch_size}"
        )

        # Adjust noise levels based on network capabilities
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Initialize from pure noise (native EDM approach) or use provided latents
        if latents is None:
            shape = (
                batch_size,
                self.net.img_channels,
                self.net.img_resolution,
                self.net.img_resolution,
            )
            latents = torch.randn(shape, device=self.device)
            logger.info(
                f"✓ Initialized from pure noise: {latents.shape}, range [{latents.min():.3f}, {latents.max():.3f}]"
            )
        else:
            latents = latents.to(self.device)
            logger.info(
                f"✓ Using provided latents: {latents.shape}, range [{latents.min():.3f}, {latents.max():.3f}]"
            )

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
        )  # t_N = 0

        # Storage for intermediate results
        step_results = []

        # Main sampling loop
        x_next = latents.to(torch.float64) * t_steps[0]  # Start with noise at sigma_max

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            # Increase noise temporarily (stochasticity)
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * randn_like(x_cur)

            # Euler step (denoising)
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)

            # Compute update direction
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction (Heun's method)
            if i < num_steps - 1:
                denoised_next = self.net(x_next, t_next, class_labels).to(
                    torch.float64
                )
                d_prime = (x_next - denoised_next) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            # Store intermediate results for visualization
            if save_steps:
                step_data = {
                    "step": i,
                    "t_cur": t_cur.item(),
                    "t_next": t_next.item(),
                    "x_hat": x_hat.clone(),
                    "denoised": denoised.clone(),
                    "x_next": x_next.clone(),
                    "noise_level": t_cur.item(),
                }
                step_results.append(step_data)

                # Save visualization for this step
                if output_dir is not None:
                    self._save_step_visualization(step_data, i, output_dir)

        logger.info(
            f"✓ Sampling completed. Final range: [{x_next.min():.3f}, {x_next.max():.3f}]"
        )

        results = {
            "final_sample": x_next,
            "initial_noise": latents,
            "step_results": step_results,
            "class_labels": class_labels,
            "sampling_params": {
                "num_steps": num_steps,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            },
        }
        
        return x_next, results

    def _save_step_visualization(
        self, step_data: Dict[str, Any], step_idx: int, output_dir: Path
    ):
        """Save visualization for a single sampling step."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Convert tensors to numpy for visualization
            x_hat = step_data["x_hat"].cpu().numpy()
            denoised = step_data["denoised"].cpu().numpy()
            x_next = step_data["x_next"].cpu().numpy()

            # Show first channel/first sample for each
            if x_hat.ndim == 4:
                x_hat_vis = (
                    x_hat[0, 0] if x_hat.shape[1] > 0 else np.mean(x_hat[0], axis=0)
                )
                denoised_vis = (
                    denoised[0, 0]
                    if denoised.shape[1] > 0
                    else np.mean(denoised[0], axis=0)
                )
                x_next_vis = (
                    x_next[0, 0] if x_next.shape[1] > 0 else np.mean(x_next[0], axis=0)
                )
            else:
                x_hat_vis = x_hat[0] if x_hat.ndim == 3 else x_hat
                denoised_vis = denoised[0] if denoised.ndim == 3 else denoised
                x_next_vis = x_next[0] if x_next.ndim == 3 else x_next

            # Normalize for display
            def normalize_for_display(img):
                img = np.clip(img, -3, 3)  # Clip extremes
                return (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Plot each stage
            axes[0].imshow(normalize_for_display(x_hat_vis), cmap="gray")
            axes[0].set_title(
                f'Step {step_idx}: Noisy Input\nσ = {step_data["noise_level"]:.3f}'
            )
            axes[0].axis("off")

            axes[1].imshow(normalize_for_display(denoised_vis), cmap="gray")
            axes[1].set_title(f"Step {step_idx}: Denoised Prediction")
            axes[1].axis("off")

            axes[2].imshow(normalize_for_display(x_next_vis), cmap="gray")
            axes[2].set_title(f"Step {step_idx}: Updated Sample")
            axes[2].axis("off")

            plt.tight_layout()

            # Save step visualization
            step_dir = output_dir / "steps"
            step_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                step_dir / f"step_{step_idx:03d}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to save step visualization {step_idx}: {e}")

    def denormalize_to_physical(self, tensor: torch.Tensor, domain: str) -> np.ndarray:
        """
        Convert tensor from [-1,1] model space to physical units.

        Args:
            tensor: Image tensor in [-1, 1] range, shape (B, C, H, W)
            domain: Domain name for range lookup

        Returns:
            Image array in physical units
        """
        domain_range = self.domain_ranges.get(domain, {"min": 0.0, "max": 1.0})

        # Step 1: [-1,1] → [0,1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)

        # Step 2: [0,1] → [domain_min, domain_max]
        domain_min = domain_range["min"]
        domain_max = domain_range["max"]
        tensor = tensor * (domain_max - domain_min) + domain_min

        return tensor.cpu().numpy()

    def create_display_visualization(
        self,
        tensor: torch.Tensor,
        domain: str,
        title: str = "Image",
        save_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Create display-ready visualization using percentile normalization.

        Args:
            tensor: Image tensor in [-1, 1] range
            domain: Domain name for physical units conversion
            title: Plot title
            save_path: Optional path to save the visualization

        Returns:
            Display-ready uint8 image array
        """
        # Convert to physical units first
        physical = self.denormalize_to_physical(tensor, domain)

        # Handle multi-channel data
        if physical.ndim == 4:  # (B, C, H, W)
            if physical.shape[1] == 3:  # RGB
                display_data = np.transpose(physical[0], (1, 2, 0))  # HWC
            elif physical.shape[1] == 1:  # Grayscale
                display_data = physical[0, 0]  # HW
            else:
                display_data = np.mean(physical[0], axis=0)  # Average channels
        elif physical.ndim == 3:  # (C, H, W)
            if physical.shape[0] == 3:  # RGB
                display_data = np.transpose(physical, (1, 2, 0))  # HWC
            elif physical.shape[0] == 1:  # Grayscale
                display_data = physical[0]  # HW
            else:
                display_data = np.mean(physical, axis=0)  # Average channels
        else:  # (H, W)
            display_data = physical

        # Apply percentile normalization (1-99%)
        valid_mask = np.isfinite(display_data)
        if np.any(valid_mask):
            p_low, p_high = np.percentile(display_data[valid_mask], [1, 99])
            clipped = np.clip(display_data, p_low, p_high)
            normalized = (clipped - p_low) / (p_high - p_low + 1e-8)
            display = (normalized * 255).clip(0, 255).astype(np.uint8)
        else:
            display = np.zeros_like(display_data, dtype=np.uint8)

        # Save if requested
        if save_path is not None:
            plt.figure(figsize=(8, 8))
            plt.imshow(display, cmap="gray" if display.ndim == 2 else None)
            plt.title(
                f"{title}\nDomain: {domain}\nPhysical range: [{display_data.min():.1f}, {display_data.max():.1f}]"
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

        return display

    def create_comprehensive_comparison(
        self,
        initial_noise: torch.Tensor,
        final_sample: torch.Tensor,
        step_results: List[Dict[str, Any]],
        domain: str,
        save_path: Path,
    ):
        """
        Create comprehensive comparison visualization showing the entire sampling process.

        Args:
            initial_noise: Initial noise tensor
            final_sample: Final sampled image
            step_results: List of intermediate step results
            domain: Domain name
            save_path: Path to save the comparison
        """
        logger.info("Creating comprehensive comparison visualization...")

        # Create figure with multiple panels
        n_steps_to_show = min(5, len(step_results))  # Show first few + last few steps
        step_indices = (
            [0, 1, 2] + [len(step_results) - 2, len(step_results) - 1]
            if len(step_results) > 3
            else list(range(len(step_results)))
        )

        fig, axes = plt.subplots(
            2, len(step_indices) + 2, figsize=(4 * (len(step_indices) + 2), 8)
        )

        # Row 1: Model space [-1, 1]
        # Initial noise
        self._add_panel(
            axes[0, 0], initial_noise, "Initial Noise", "[-1, 1] Model Space"
        )

        # Intermediate steps
        for i, step_idx in enumerate(step_indices):
            if step_idx < len(step_results):
                step_data = step_results[step_idx]
                self._add_panel(
                    axes[0, i + 1],
                    step_data["x_next"],
                    f"Step {step_idx}",
                    f"σ = {step_data['noise_level']:.3f}",
                )

        # Final sample
        self._add_panel(
            axes[0, -1], final_sample, "Final Sample", "[-1, 1] Model Space"
        )

        # Row 2: Physical units + Display normalization
        # Initial noise in physical units
        noise_physical = self.denormalize_to_physical(initial_noise, domain)
        self._add_panel_physical(axes[1, 0], noise_physical, "Initial Noise", domain)

        # Intermediate steps in physical units
        for i, step_idx in enumerate(step_indices):
            if step_idx < len(step_results):
                step_data = step_results[step_idx]
                step_physical = self.denormalize_to_physical(
                    step_data["x_next"], domain
                )
                self._add_panel_physical(
                    axes[1, i + 1], step_physical, f"Step {step_idx}", domain
                )

        # Final sample in physical units
        final_physical = self.denormalize_to_physical(final_sample, domain)
        self._add_panel_physical(axes[1, -1], final_physical, "Final Sample", domain)

        # Add row labels
        axes[0, 0].set_ylabel("Model Space\n[-1, 1]", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel(
            "Physical Units\n+ Display Norm", fontsize=12, fontweight="bold"
        )

        plt.suptitle(
            f"Native EDM Sampling Process - Domain: {domain}\n"
            + f"Starting from pure noise → Progressive denoising → Final sample",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"✓ Comprehensive comparison saved: {save_path}")

    def _add_panel(self, ax, tensor, title, subtitle):
        """Add a panel showing tensor in model space [-1, 1]."""
        if tensor.ndim == 4:
            img = (
                tensor[0, 0].cpu().numpy()
                if tensor.shape[1] > 0
                else np.mean(tensor[0].cpu().numpy(), axis=0)
            )
        elif tensor.ndim == 3:
            img = (
                tensor[0].cpu().numpy()
                if tensor.shape[0] > 0
                else np.mean(tensor.cpu().numpy(), axis=0)
            )
        else:
            img = tensor.cpu().numpy()

        # Normalize for display
        img = np.clip(img, -3, 3)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        ax.imshow(img_norm, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.axis("off")

    def _add_panel_physical(self, ax, physical_array, title, domain):
        """Add a panel showing data in physical units with display normalization."""
        if physical_array.ndim == 4:
            img = (
                physical_array[0, 0]
                if physical_array.shape[1] > 0
                else np.mean(physical_array[0], axis=0)
            )
        elif physical_array.ndim == 3:
            img = (
                physical_array[0]
                if physical_array.shape[0] > 0
                else np.mean(physical_array, axis=0)
            )
        else:
            img = physical_array

        # Apply display normalization (percentile 1-99%)
        valid_mask = np.isfinite(img)
        if np.any(valid_mask):
            p_low, p_high = np.percentile(img[valid_mask], [1, 99])
            img_norm = np.clip(img, p_low, p_high)
            img_norm = (img_norm - p_low) / (p_high - p_low + 1e-8)
        else:
            img_norm = np.zeros_like(img)

        ax.imshow(img_norm, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\nRange: [{img.min():.1f}, {img.max():.1f}]", fontsize=10)
        ax.axis("off")


def main():
    """Main function for comprehensive EDM sampling with visualizations."""
    parser = argparse.ArgumentParser(
        description="Comprehensive EDM sampling with step-by-step visualizations (native EDM from noise)"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pkl)",
    )

    # Sampling arguments
    parser.add_argument(
        "--num_steps", type=int, default=18, help="Number of sampling steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of samples to generate"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="photography",
        choices=["photography", "microscopy", "astronomy"],
        help="Domain for conditioning and normalization",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comprehensive_sampling_viz",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of different samples to generate",
    )

    # Sampling parameters
    parser.add_argument(
        "--sigma_max", type=float, default=80.0, help="Maximum noise level"
    )
    parser.add_argument(
        "--rho", type=float, default=7.0, help="Time step discretization parameter"
    )

    # Device arguments
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for sampling"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EDM SAMPLING WITH STEP-BY-STEP VISUALIZATIONS")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Sampling steps: {args.num_steps}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info("=" * 80)

    # Initialize sampler
    sampler = ComprehensiveEDMSampler(
        model_path=args.model_path,
        device=args.device,
    )

    # Create domain labels
    if sampler.net.label_dim > 0:
        class_labels = torch.zeros(
            args.batch_size, sampler.net.label_dim, device=sampler.device
        )
        if args.domain == "photography":
            class_labels[:, 0] = 1.0  # Photography domain
        elif args.domain == "microscopy":
            class_labels[:, 1] = 1.0  # Microscopy domain
        elif args.domain == "astronomy":
            class_labels[:, 2] = 1.0  # Astronomy domain
        logger.info(f"✓ Using domain conditioning: {args.domain}")
    else:
        class_labels = None
        logger.info("✓ No domain conditioning (unconditional model)")

    # Generate multiple samples
    all_results = []

    for sample_idx in range(args.num_samples):
        logger.info(f"\n🎯 Generating sample {sample_idx + 1}/{args.num_samples}")

        # Create output directory for this sample
        sample_dir = output_dir / f"sample_{sample_idx:02d}"
        sample_dir.mkdir(exist_ok=True)

        # Sample with step-by-step visualization
        final_tensor, results = sampler.sample_native_edm(
            batch_size=args.batch_size,
            class_labels=class_labels,
            num_steps=args.num_steps,
            sigma_max=args.sigma_max,
            rho=args.rho,
            save_steps=True,
            output_dir=sample_dir,
        )

        all_results.append(results)

        # Create comprehensive comparison
        comparison_path = sample_dir / "comprehensive_comparison.png"
        sampler.create_comprehensive_comparison(
            initial_noise=results["initial_noise"],
            final_sample=results["final_sample"],
            step_results=results["step_results"],
            domain=args.domain,
            save_path=comparison_path,
        )

        # Save individual final sample visualization
        final_vis_path = sample_dir / "final_sample_display.png"
        display_img = sampler.create_display_visualization(
            results["final_sample"],
            args.domain,
            title=f"Final Sample {sample_idx + 1} - {args.domain.capitalize()}",
            save_path=final_vis_path,
        )

        # Save physical units data
        physical_data = sampler.denormalize_to_physical(
            results["final_sample"], args.domain
        )
        np.save(sample_dir / "final_sample_physical.npy", physical_data)

        logger.info(f"✓ Sample {sample_idx + 1} completed and saved to {sample_dir}")

    # Create summary visualization
    logger.info("\n📊 Creating summary visualization...")

    # Create a grid showing all final samples
    fig, axes = plt.subplots(2, args.num_samples, figsize=(4 * args.num_samples, 8))

    for sample_idx in range(args.num_samples):
        results = all_results[sample_idx]

        # Model space [-1, 1]
        final_tensor = results["final_sample"]
        if final_tensor.ndim == 4:
            img_model = final_tensor[0, 0].cpu().numpy()
        else:
            img_model = (
                final_tensor[0].cpu().numpy()
                if final_tensor.ndim == 3
                else final_tensor.cpu().numpy()
            )

        img_model_norm = (img_model - img_model.min()) / (
            img_model.max() - img_model.min() + 1e-8
        )
        axes[0, sample_idx].imshow(img_model_norm, cmap="gray", vmin=0, vmax=1)
        axes[0, sample_idx].set_title(f"Sample {sample_idx + 1}\nModel Space [-1, 1]")
        axes[0, sample_idx].axis("off")

        # Display space (physical units with percentile normalization)
        display_img = sampler.create_display_visualization(
            results["final_sample"],
            args.domain,
            save_path=None,  # Don't save individual
        )
        axes[1, sample_idx].imshow(
            display_img, cmap="gray" if display_img.ndim == 2 else None
        )
        axes[1, sample_idx].set_title(f"Sample {sample_idx + 1}\nDisplay Normalized")
        axes[1, sample_idx].axis("off")

    plt.suptitle(
        f"Native EDM Sampling Results - {args.domain.capitalize()}\n"
        + f"All samples generated from pure noise using {args.num_steps} steps",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    summary_path = output_dir / "all_samples_summary.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Summary visualization saved: {summary_path}")

    # Save sampling metadata
    metadata = {
        "sampling_info": {
            "model_path": args.model_path,
            "domain": args.domain,
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "sigma_max": args.sigma_max,
            "rho": args.rho,
            "device": args.device,
            "seed": args.seed,
        },
        "domain_ranges": sampler.domain_ranges,
        "model_info": {
            "resolution": sampler.net.img_resolution,
            "channels": sampler.net.img_channels,
            "label_dim": sampler.net.label_dim,
            "sigma_min": sampler.net.sigma_min,
            "sigma_max": sampler.net.sigma_max,
        },
        "output_files": {
            "summary_visualization": str(summary_path),
            "individual_samples": [
                str(output_dir / f"sample_{i:02d}") for i in range(args.num_samples)
            ],
        },
    }

    metadata_path = output_dir / "sampling_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Metadata saved: {metadata_path}")
    logger.info("=" * 80)
    logger.info("🎉 COMPREHENSIVE SAMPLING COMPLETED!")
    logger.info(f"📁 All results saved to: {output_dir}")
    logger.info(f"📊 Summary: {args.num_samples} samples, {args.num_steps} steps each")
    logger.info(f"🎯 Native EDM sampling from pure noise → denoised samples")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

