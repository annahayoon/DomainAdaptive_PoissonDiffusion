#!/usr/bin/env python3
"""
EDM Denoising Test with Synthetic Gaussian Noise

This script tests the denoising capability by:
1. Loading CLEAN test images
2. Adding known Gaussian noise (σ)
3. Denoising using EDM sampler starting at the known noise level
4. Comparing input clean vs noisy vs denoised

This verifies the denoising algorithm works correctly with proper data.

Usage:
    python sample_clean_plus_synthetic_noise.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --clean_dir dataset/processed/pt_tiles/photography/clean \
        --output_dir results/synthetic_noise_denoising_test \
        --noise_sigma 2.0 \
        --num_steps 18 \
        --domain photography \
        --num_samples 8
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import sys
import logging

# Add project root and EDM to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

import external.edm.dnnlib
from external.edm.torch_utils import distributed as dist

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EDMDenoiser:
    """EDM-based denoiser for images with synthetic Gaussian noise."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        self.net = checkpoint["ema"].to(self.device)
        self.net.eval()
        
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Resolution: {self.net.img_resolution}")
        logger.info(f"  Channels: {self.net.img_channels}")
    
    def denoise(
        self,
        noisy_image: torch.Tensor,
        noise_sigma: float,
        class_labels: Optional[torch.Tensor] = None,
        num_steps: int = 18,
        rho: float = 7.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Denoise an image starting from known noise level.
        
        Args:
            noisy_image: Noisy input (B, C, H, W) in [-1, 1]
            noise_sigma: The noise level in the input
            class_labels: Domain labels
            num_steps: Number of denoising steps
            rho: Time step parameter
        """
        logger.info(f"Denoising from sigma={noise_sigma:.3f} to sigma=0.0")
        
        # Set up noise schedule from noise_sigma to sigma_min
        sigma_min = max(0.002, self.net.sigma_min)
        sigma_max = min(noise_sigma, self.net.sigma_max)
        
        # Time step discretization
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        
        logger.info(f"Time steps: {t_steps[0]:.3f} -> {t_steps[-1]:.3f} ({len(t_steps)-1} steps)")
        
        # Start from the noisy image (no additional scaling!)
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
        
        logger.info(f"✓ Denoising completed: range [{denoised_output.min():.3f}, {denoised_output.max():.3f}]")
        
        return denoised_output, {"final": denoised_output, "initial": noisy_image}


def main():
    parser = argparse.ArgumentParser(description="Test EDM denoising with synthetic Gaussian noise")
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--noise_sigma", type=float, default=0.1, help="Gaussian noise level to add (keep small to preserve structure)")
    parser.add_argument("--num_steps", type=int, default=18, help="Denoising steps")
    parser.add_argument("--domain", type=str, default="photography")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("EDM DENOISING TEST WITH SYNTHETIC GAUSSIAN NOISE")
    logger.info("=" * 80)
    logger.info(f"Clean images: {args.clean_dir}")
    logger.info(f"Noise level: σ = {args.noise_sigma}")
    logger.info(f"Denoising steps: {args.num_steps}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)
    
    # Load denoiser
    denoiser = EDMDenoiser(args.model_path, args.device)
    
    # Get clean files
    clean_files = sorted(Path(args.clean_dir).glob("*.pt"))[:args.num_samples]
    logger.info(f"Found {len(clean_files)} clean images")
    
    # Domain labels
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
    
    # Process each image
    for idx, clean_file in enumerate(clean_files):
        logger.info(f"\n🎯 Processing {idx+1}/{len(clean_files)}: {clean_file.name}")
        
        # Load clean image
        clean = torch.load(clean_file, map_location=denoiser.device)
        if clean.ndim == 3:
            clean = clean.unsqueeze(0)
        clean = clean.to(torch.float32)
        
        logger.info(f"  Clean range: [{clean.min():.4f}, {clean.max():.4f}], std={clean.std():.4f}")
        
        # Add Gaussian noise
        noise = torch.randn_like(clean) * args.noise_sigma
        noisy = clean + noise
        
        logger.info(f"  Noisy range: [{noisy.min():.4f}, {noisy.max():.4f}], std={noisy.std():.4f}")
        
        # Denoise
        denoised, results = denoiser.denoise(
            noisy,
            noise_sigma=args.noise_sigma,
            class_labels=class_labels,
            num_steps=args.num_steps,
        )
        
        # Calculate metrics
        mse_noisy = ((clean - noisy) ** 2).mean().item()
        mse_denoised = ((clean - denoised) ** 2).mean().item()
        psnr_noisy = -10 * np.log10(mse_noisy) if mse_noisy > 0 else float('inf')
        psnr_denoised = -10 * np.log10(mse_denoised) if mse_denoised > 0 else float('inf')
        
        logger.info(f"  MSE: Noisy={mse_noisy:.4f}, Denoised={mse_denoised:.4f}")
        logger.info(f"  PSNR: Noisy={psnr_noisy:.2f}dB, Denoised={psnr_denoised:.2f}dB")
        logger.info(f"  Improvement: {psnr_denoised - psnr_noisy:.2f}dB")
        
        # Save visualization
        sample_dir = output_dir / f"sample_{idx:02d}_{clean_file.stem}"
        sample_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Normalize for display
        def norm_display(x):
            x = x[0].cpu().numpy()
            if x.shape[0] == 3:
                x = np.transpose(x, (1, 2, 0))
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            else:
                x = x[0]
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            return x
        
        # Clean
        axes[0].imshow(norm_display(clean), cmap='gray' if clean.shape[1] == 1 else None)
        axes[0].set_title(f"Clean (Ground Truth)\nRange: [{clean.min():.2f}, {clean.max():.2f}]")
        axes[0].axis('off')
        
        # Noisy
        axes[1].imshow(norm_display(noisy), cmap='gray' if noisy.shape[1] == 1 else None)
        axes[1].set_title(f"Noisy (σ={args.noise_sigma})\nPSNR: {psnr_noisy:.2f}dB")
        axes[1].axis('off')
        
        # Denoised
        axes[2].imshow(norm_display(denoised), cmap='gray' if denoised.shape[1] == 1 else None)
        axes[2].set_title(f"Denoised\nPSNR: {psnr_denoised:.2f}dB\nΔ: +{psnr_denoised - psnr_noisy:.2f}dB")
        axes[2].axis('off')
        
        plt.suptitle(f"EDM Denoising Test - {clean_file.stem}\nNoise σ={args.noise_sigma}, Steps={args.num_steps}")
        plt.tight_layout()
        plt.savefig(sample_dir / "comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save tensors
        torch.save(clean.cpu(), sample_dir / "clean.pt")
        torch.save(noisy.cpu(), sample_dir / "noisy.pt")
        torch.save(denoised.cpu(), sample_dir / "denoised.pt")
        
        logger.info(f"✓ Saved to {sample_dir}")
    
    logger.info("=" * 80)
    logger.info("🎉 DENOISING TEST COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

