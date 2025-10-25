#!/usr/bin/env python3
"""
Visualize Enhancement Results for Best Sigma Values

Creates a comparison visualization showing:
- Noisy input
- Enhanced with σ=0.0008 (best PSNR/SSIM)
- Enhanced with σ=0.002 (best brightness/balanced)

Usage:
    python sample/visualize_best_sigmas.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/photography/noisy \
        --output_path results/comprehensive_enhancement_analysis/best_sigma_comparison.png \
        --domain photography \
        --num_samples 3
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_tiles(metadata_json: Path, domain: str, split: str = "test") -> List[Dict]:
    """Load test tile metadata."""
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)
    
    tiles = metadata.get('tiles', [])
    filtered_tiles = [
        tile for tile in tiles
        if tile.get('domain') == domain and tile.get('split') == split
    ]
    
    return filtered_tiles


def load_noisy_image(tile_id: str, noisy_dir: Path, device: torch.device) -> torch.Tensor:
    """Load a noisy .pt file."""
    noisy_path = noisy_dir / f"{tile_id}.pt"
    
    if not noisy_path.exists():
        raise FileNotFoundError(f"Noisy file not found: {noisy_path}")
    
    noisy_tensor = torch.load(noisy_path, map_location=device)
    
    if isinstance(noisy_tensor, dict):
        if 'noisy' in noisy_tensor:
            noisy_tensor = noisy_tensor['noisy']
        elif 'noisy_norm' in noisy_tensor:
            noisy_tensor = noisy_tensor['noisy_norm']
        elif 'image' in noisy_tensor:
            noisy_tensor = noisy_tensor['image']
    
    noisy_tensor = noisy_tensor.float()
    
    if noisy_tensor.ndim == 2:
        noisy_tensor = noisy_tensor.unsqueeze(0)
    elif noisy_tensor.ndim == 3 and noisy_tensor.shape[-1] in [1, 3]:
        noisy_tensor = noisy_tensor.permute(2, 0, 1)
    
    return noisy_tensor


class EDMLowLightEnhancer:
    """EDM-based low-light image enhancer."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.net = checkpoint["ema"].to(self.device)
        self.net.eval()

        logger.info("✓ Model loaded successfully")

    def enhance_low_light(
        self,
        low_light_image: torch.Tensor,
        enhancement_sigma: float,
        class_labels: torch.Tensor = None,
        num_steps: int = 18,
        rho: float = 7.0,
    ) -> torch.Tensor:
        """Enhance low-light image using EDM model."""
        sigma_min = max(0.002, self.net.sigma_min)
        sigma_max = min(enhancement_sigma, self.net.sigma_max)
        
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        
        x = low_light_image.to(torch.float64).to(self.device)
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            enhanced = self.net(x, t_cur, class_labels).to(torch.float64)
            d_cur = (x - enhanced) / t_cur
            x_next = x + (t_next - t_cur) * d_cur
            
            if i < num_steps - 1:
                enhanced_next = self.net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - enhanced_next) / t_next
                x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
            
            x = x_next
        
        return torch.clamp(x, -1, 1)


def create_comparison_visualization(
    samples: List[Dict],
    output_path: Path
):
    """Create comparison visualization for best sigma values."""
    logger.info("Creating best sigma comparison visualization...")
    
    n_samples = len(samples)
    
    # Create figure: 3 columns (noisy, σ=0.0008, σ=0.002) × n_samples rows
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    def norm_display(tensor):
        """Normalize tensor for display."""
        x = tensor[0].cpu().numpy() if tensor.ndim == 4 else tensor.cpu().numpy()
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))
        else:
            x = x[0] if x.ndim == 3 else x
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x
    
    for i, sample in enumerate(samples):
        tile_id = sample['tile_id']
        noisy = sample['noisy']
        enhanced_0008 = sample['enhanced_0.0008']
        enhanced_002 = sample['enhanced_0.002']
        
        # Column 1: Noisy input
        axes[i, 0].imshow(norm_display(noisy), cmap='gray' if noisy.shape[1] == 1 else None)
        axes[i, 0].set_title(f'Noisy Input\nRange: [{noisy.min():.4f}, {noisy.max():.4f}]', fontsize=10)
        axes[i, 0].axis('off')
        
        # Column 2: Enhanced σ=0.0008
        axes[i, 1].imshow(norm_display(enhanced_0008), cmap='gray' if enhanced_0008.shape[1] == 1 else None)
        axes[i, 1].set_title(f'Enhanced σ=0.0008\nRange: [{enhanced_0008.min():.4f}, {enhanced_0008.max():.4f}]', fontsize=10)
        axes[i, 1].axis('off')
        
        # Column 3: Enhanced σ=0.002
        axes[i, 2].imshow(norm_display(enhanced_002), cmap='gray' if enhanced_002.shape[1] == 1 else None)
        axes[i, 2].set_title(f'Enhanced σ=0.002\nRange: [{enhanced_002.min():.4f}, {enhanced_002.max():.4f}]', fontsize=10)
        axes[i, 2].axis('off')
        
        # Add row label
        axes[i, 0].text(-0.1, 0.5, f'{tile_id}', 
                       transform=axes[i, 0].transAxes,
                       fontsize=9, rotation=90,
                       verticalalignment='center',
                       fontweight='bold')
    
    # Add column labels
    fig.text(0.25, 0.98, 'Noisy Input', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.58, 0.98, 'Enhanced σ=0.0008\n(Best PSNR/SSIM)', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.83, 0.98, 'Enhanced σ=0.002\n(Best Balanced)', ha='center', fontsize=12, fontweight='bold')
    
    plt.suptitle(
        'Low-Light Enhancement: Best Sigma Comparison\n'
        'Comparing optimal sigma values on 3 test samples',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Comparison visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize enhancement results for best sigma values")
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--metadata_json", type=str, required=True)
    parser.add_argument("--noisy_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--domain", type=str, default="photography")
    parser.add_argument("--num_steps", type=int, default=18)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("="*80)
    logger.info("VISUALIZING BEST SIGMA VALUES")
    logger.info("="*80)
    
    # Initialize enhancer
    enhancer = EDMLowLightEnhancer(args.model_path, args.device)
    
    # Load test tiles
    test_tiles = load_test_tiles(Path(args.metadata_json), args.domain, split="test")
    
    # Filter to available tiles
    available_tiles = []
    for tile_info in test_tiles:
        tile_id = tile_info["tile_id"]
        noisy_path = Path(args.noisy_dir) / f"{tile_id}.pt"
        if noisy_path.exists():
            available_tiles.append(tile_info)
    
    logger.info(f"Found {len(available_tiles)} tiles with noisy files")
    
    # Randomly select samples
    import random
    random.seed(args.seed)
    selected_tiles = random.sample(available_tiles, min(args.num_samples, len(available_tiles)))
    
    logger.info(f"Selected {len(selected_tiles)} tiles for visualization")
    
    # Create domain labels
    if enhancer.net.label_dim > 0:
        class_labels = torch.zeros(1, enhancer.net.label_dim, device=enhancer.device)
        if args.domain == "photography":
            class_labels[:, 0] = 1.0
        elif args.domain == "microscopy":
            class_labels[:, 1] = 1.0
        elif args.domain == "astronomy":
            class_labels[:, 2] = 1.0
    else:
        class_labels = None
    
    # Process samples
    samples = []
    
    for tile_info in selected_tiles:
        tile_id = tile_info["tile_id"]
        logger.info(f"Processing {tile_id}...")
        
        # Load noisy image
        noisy_image = load_noisy_image(tile_id, Path(args.noisy_dir), enhancer.device)
        if noisy_image.ndim == 3:
            noisy_image = noisy_image.unsqueeze(0)
        noisy_image = noisy_image.to(torch.float32)
        
        # Enhance with σ=0.0008
        logger.info("  Enhancing with σ=0.0008...")
        enhanced_0008 = enhancer.enhance_low_light(
            noisy_image,
            enhancement_sigma=0.0008,
            class_labels=class_labels,
            num_steps=args.num_steps,
        )
        
        # Enhance with σ=0.002
        logger.info("  Enhancing with σ=0.002...")
        enhanced_002 = enhancer.enhance_low_light(
            noisy_image,
            enhancement_sigma=0.002,
            class_labels=class_labels,
            num_steps=args.num_steps,
        )
        
        samples.append({
            'tile_id': tile_id,
            'noisy': noisy_image,
            'enhanced_0.0008': enhanced_0008,
            'enhanced_0.002': enhanced_002,
        })
    
    # Create visualization
    create_comparison_visualization(samples, Path(args.output_path))
    
    logger.info("="*80)
    logger.info("🎉 VISUALIZATION COMPLETED!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
