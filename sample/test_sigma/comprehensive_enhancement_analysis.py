#!/usr/bin/env python3
"""
Comprehensive Enhancement Analysis on All Test Tiles

This script:
1. Tests enhancement on ALL test tiles (not just 3 examples)
2. Outputs metrics as a pandas DataFrame (CSV)
3. Generates histogram distributions for each metric per sigma value

Usage:
    python sample/comprehensive_enhancement_analysis.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/photography/noisy \
        --output_dir results/comprehensive_enhancement_analysis \
        --domain photography \
        --num_steps 18
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
    """Load test tile metadata from JSON file."""
    logger.info(f"Loading {split} tiles for {domain} from {metadata_json}")
    
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)
    
    # Filter tiles by domain and split
    tiles = metadata.get('tiles', [])
    filtered_tiles = [
        tile for tile in tiles
        if tile.get('domain') == domain and tile.get('split') == split
    ]
    
    logger.info(f"Found {len(filtered_tiles)} {split} tiles for {domain}")
    return filtered_tiles


def load_noisy_image(tile_id: str, noisy_dir: Path, device: torch.device) -> torch.Tensor:
    """Load a noisy .pt file."""
    noisy_path = noisy_dir / f"{tile_id}.pt"
    
    if not noisy_path.exists():
        raise FileNotFoundError(f"Noisy file not found: {noisy_path}")
    
    noisy_tensor = torch.load(noisy_path, map_location=device)
    
    # Handle different tensor formats
    if isinstance(noisy_tensor, dict):
        if 'noisy' in noisy_tensor:
            noisy_tensor = noisy_tensor['noisy']
        elif 'noisy_norm' in noisy_tensor:
            noisy_tensor = noisy_tensor['noisy_norm']
        elif 'image' in noisy_tensor:
            noisy_tensor = noisy_tensor['image']
    
    noisy_tensor = noisy_tensor.float()
    
    # Ensure CHW format
    if noisy_tensor.ndim == 2:
        noisy_tensor = noisy_tensor.unsqueeze(0)
    elif noisy_tensor.ndim == 3 and noisy_tensor.shape[-1] in [1, 3]:
        noisy_tensor = noisy_tensor.permute(2, 0, 1)
    
    return noisy_tensor


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate PSNR."""
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def calculate_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate Mean Squared Error."""
    return F.mse_loss(img1, img2).item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate SSIM."""
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    if img1.shape[1] == 3:
        img1 = 0.299 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
    if img2.shape[1] == 3:
        img2 = 0.299 * img2[:, 0:1] + 0.587 * img2[:, 1:2] + 0.114 * img2[:, 2:3]
    
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


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
        
        enhanced_output = torch.clamp(x, -1, 1)
        return enhanced_output


def main():
    parser = argparse.ArgumentParser(description="Comprehensive enhancement analysis on all test tiles")
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--metadata_json", type=str, required=True)
    parser.add_argument("--noisy_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--domain", type=str, default="photography")
    parser.add_argument("--num_steps", type=int, default=18)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of tiles to sample (default: 100)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE ENHANCEMENT ANALYSIS ON ALL TEST TILES")
    logger.info("="*80)
    
    # Initialize enhancer
    enhancer = EDMLowLightEnhancer(args.model_path, args.device)
    
    # Load test tiles
    test_tiles = load_test_tiles(Path(args.metadata_json), args.domain, split="test")
    
    # Filter to only available tiles
    available_tiles = []
    for tile_info in test_tiles:
        tile_id = tile_info["tile_id"]
        noisy_path = Path(args.noisy_dir) / f"{tile_id}.pt"
        if noisy_path.exists():
            available_tiles.append(tile_info)
    
    logger.info(f"Found {len(available_tiles)} tiles with noisy files")
    
    # Sample tiles if requested
    if args.num_samples < len(available_tiles):
        import random
        random.seed(42)
        available_tiles = random.sample(available_tiles, args.num_samples)
        logger.info(f"Sampled {len(available_tiles)} tiles for analysis")
    
    # Define sigma values
    sigma_values = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 
                   0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002]
    
    logger.info(f"Testing {len(sigma_values)} sigma values: {sigma_values}")
    
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
    
    # Results storage
    results = []
    
    # Process each tile
    for tile_info in tqdm(available_tiles, desc="Processing tiles"):
        tile_id = tile_info["tile_id"]
        
        try:
            # Load noisy image
            noisy_image = load_noisy_image(tile_id, Path(args.noisy_dir), enhancer.device)
            if noisy_image.ndim == 3:
                noisy_image = noisy_image.unsqueeze(0)
            noisy_image = noisy_image.to(torch.float32)
            
            noisy_mean = noisy_image.mean().item()
            noisy_std = noisy_image.std().item()
            
            # Test each sigma
            for sigma in sigma_values:
                # Enhance
                enhanced = enhancer.enhance_low_light(
                    noisy_image,
                    enhancement_sigma=sigma,
                    class_labels=class_labels,
                    num_steps=args.num_steps,
                )
                
                # Calculate metrics
                psnr = calculate_psnr(noisy_image, enhanced)
                ssim = calculate_ssim(noisy_image, enhanced)
                mse = calculate_mse(noisy_image, enhanced)
                brightness_increase = (enhanced.mean() - noisy_image.mean()).item()
                enhanced_mean = enhanced.mean().item()
                enhanced_std = enhanced.std().item()
                enhanced_range = (enhanced.max() - enhanced.min()).item()
                
                # Store results
                results.append({
                    'tile_id': tile_id,
                    'sigma': sigma,
                    'psnr': psnr,
                    'ssim': ssim,
                    'mse': mse,
                    'brightness_increase': brightness_increase,
                    'noisy_mean': noisy_mean,
                    'noisy_std': noisy_std,
                    'enhanced_mean': enhanced_mean,
                    'enhanced_std': enhanced_std,
                    'enhanced_range': enhanced_range,
                })
        
        except Exception as e:
            logger.warning(f"Failed to process {tile_id}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save DataFrame to CSV
    csv_path = output_dir / 'enhancement_metrics_all_tiles.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved metrics DataFrame to {csv_path}")
    
    # Print summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS BY SIGMA")
    logger.info("="*80)
    
    summary = df.groupby('sigma').agg({
        'psnr': ['mean', 'std', 'min', 'max'],
        'ssim': ['mean', 'std', 'min', 'max'],
        'mse': ['mean', 'std', 'min', 'max'],
        'brightness_increase': ['mean', 'std', 'min', 'max'],
        'enhanced_mean': ['mean', 'std', 'min', 'max'],
        'enhanced_range': ['mean', 'std', 'min', 'max']
    }).round(6)
    
    logger.info(f"\n{summary}")
    
    # Save summary to CSV
    summary_path = output_dir / 'enhancement_summary_statistics.csv'
    summary.to_csv(summary_path)
    logger.info(f"✓ Saved summary statistics to {summary_path}")
    
    # Create histogram visualizations
    create_histograms(df, sigma_values, output_dir)
    
    # Create best sigma recommendation
    recommend_best_sigma(df, output_dir)
    
    logger.info("="*80)
    logger.info("🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
    logger.info("="*80)


def create_histograms(df: pd.DataFrame, sigma_values: List[float], output_dir: Path):
    """Create histogram distributions for each metric per sigma value."""
    logger.info("Creating histogram visualizations...")
    
    metrics = ['psnr', 'ssim', 'brightness_increase', 'enhanced_mean']
    metric_labels = {
        'psnr': 'PSNR (dB)',
        'ssim': 'SSIM',
        'brightness_increase': 'Brightness Increase',
        'enhanced_mean': 'Enhanced Mean Value'
    }
    
    # Create figure with subplots for each metric
    for metric in metrics:
        n_sigmas = len(sigma_values)
        n_cols = 5
        n_rows = (n_sigmas + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        axes = axes.flatten() if n_sigmas > 1 else [axes]
        
        for i, sigma in enumerate(sigma_values):
            ax = axes[i]
            data = df[df['sigma'] == sigma][metric]
            
            ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'σ = {sigma:.4f}', fontsize=10)
            ax.set_xlabel(metric_labels[metric], fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = data.mean()
            std_val = data.std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
            ax.legend(fontsize=8)
        
        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.suptitle(f'{metric_labels[metric]} Distribution Across All Test Tiles', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        fig_path = output_dir / f'histogram_{metric}_by_sigma.png'
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved {metric} histograms to {fig_path}")


def recommend_best_sigma(df: pd.DataFrame, output_dir: Path):
    """Recommend best sigma based on comprehensive analysis."""
    logger.info("\n" + "="*80)
    logger.info("BEST SIGMA RECOMMENDATION")
    logger.info("="*80)
    
    # Group by sigma and calculate means
    sigma_summary = df.groupby('sigma').agg({
        'psnr': 'mean',
        'ssim': 'mean',
        'brightness_increase': 'mean',
        'enhanced_mean': 'mean'
    })
    
    # Find best sigma for each metric
    best_psnr_sigma = sigma_summary['psnr'].idxmax()
    best_ssim_sigma = sigma_summary['ssim'].idxmax()
    best_brightness_sigma = sigma_summary['brightness_increase'].idxmax()
    
    logger.info(f"Best PSNR: σ = {best_psnr_sigma:.4f} ({sigma_summary.loc[best_psnr_sigma, 'psnr']:.2f} dB)")
    logger.info(f"Best SSIM: σ = {best_ssim_sigma:.4f} ({sigma_summary.loc[best_ssim_sigma, 'ssim']:.4f})")
    logger.info(f"Best Brightness Increase: σ = {best_brightness_sigma:.4f} ({sigma_summary.loc[best_brightness_sigma, 'brightness_increase']:.6f})")
    
    # Balanced recommendation (60% brightness, 40% SSIM)
    normalized_brightness = (sigma_summary['brightness_increase'] - sigma_summary['brightness_increase'].min()) / \
                           (sigma_summary['brightness_increase'].max() - sigma_summary['brightness_increase'].min() + 1e-8)
    normalized_ssim = (sigma_summary['ssim'] - sigma_summary['ssim'].min()) / \
                     (sigma_summary['ssim'].max() - sigma_summary['ssim'].min() + 1e-8)
    
    balance_score = 0.6 * normalized_brightness + 0.4 * normalized_ssim
    best_balanced_sigma = balance_score.idxmax()
    
    logger.info(f"\n✓ BALANCED RECOMMENDATION: σ = {best_balanced_sigma:.4f}")
    logger.info(f"  - PSNR: {sigma_summary.loc[best_balanced_sigma, 'psnr']:.2f} dB")
    logger.info(f"  - SSIM: {sigma_summary.loc[best_balanced_sigma, 'ssim']:.4f}")
    logger.info(f"  - Brightness Increase: {sigma_summary.loc[best_balanced_sigma, 'brightness_increase']:.6f}")
    logger.info(f"  - Enhanced Mean: {sigma_summary.loc[best_balanced_sigma, 'enhanced_mean']:.6f}")
    
    # Save recommendation
    recommendation = {
        'best_psnr_sigma': float(best_psnr_sigma),
        'best_ssim_sigma': float(best_ssim_sigma),
        'best_brightness_sigma': float(best_brightness_sigma),
        'best_balanced_sigma': float(best_balanced_sigma),
        'metrics': {
            'psnr': float(sigma_summary.loc[best_balanced_sigma, 'psnr']),
            'ssim': float(sigma_summary.loc[best_balanced_sigma, 'ssim']),
            'brightness_increase': float(sigma_summary.loc[best_balanced_sigma, 'brightness_increase']),
            'enhanced_mean': float(sigma_summary.loc[best_balanced_sigma, 'enhanced_mean'])
        }
    }
    
    import json
    rec_path = output_dir / 'best_sigma_recommendation.json'
    with open(rec_path, 'w') as f:
        json.dump(recommendation, f, indent=2)
    
    logger.info(f"✓ Saved recommendation to {rec_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
