#!/usr/bin/env python3
"""
Diagnose Noise Levels in Test Images

This script analyzes the actual noise levels in the test images to understand
why denoising might not be working effectively.

Usage:
    python sample/diagnose_noise_levels.py \
        --results_dir results/comprehensive_sigma_sweep \
        --domain photography
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_noise_levels(results_dir: Path, domain: str, clean_dir: Path = None):
    """Analyze actual noise levels in test images."""
    logger.info("Analyzing noise levels in test images...")
    
    # Find example/sample directories
    example_dirs = [d for d in results_dir.iterdir() 
                   if d.is_dir() and (d.name.startswith('example_') or d.name.startswith('sample_'))]
    
    if not example_dirs:
        logger.error("No example/sample directories found")
        return
    
    all_analysis = {}
    all_clean_analysis = {}
    
    for example_dir in example_dirs:
        logger.info(f"Analyzing {example_dir.name}")
        
        # Load noisy image
        noisy_path = example_dir / "noisy.pt"
        if not noisy_path.exists():
            logger.warning(f"Noisy file not found: {noisy_path}")
            continue
            
        noisy_image = torch.load(noisy_path, map_location='cpu')
        
        # Analyze noise characteristics
        analysis = analyze_image_noise(noisy_image, example_dir.name)
        all_analysis[example_dir.name] = analysis
        
        # Print analysis
        logger.info(f"\nNoise Analysis for {example_dir.name}:")
        logger.info(f"  Image shape: {noisy_image.shape}")
        logger.info(f"  Value range: [{noisy_image.min():.6f}, {noisy_image.max():.6f}]")
        logger.info(f"  Mean: {noisy_image.mean():.6f}")
        logger.info(f"  Std: {noisy_image.std():.6f}")
        logger.info(f"  Estimated noise std: {analysis['estimated_noise_std']:.6f}")
        logger.info(f"  Signal-to-noise ratio: {analysis['snr']:.2f} dB")
        logger.info(f"  Noise level category: {analysis['noise_category']}")
        
        # Try to load clean tile if clean_dir is provided
        if clean_dir is not None:
            # Extract tile_id from directory name
            # Format: example_XX_tile_id or sample_XX_tile_id
            parts = example_dir.name.split('_', 2)
            if len(parts) >= 3:
                tile_id = parts[2]  # Get everything after "example_XX_" or "sample_XX_"
                clean_path = clean_dir / f"{tile_id}.pt"
                
                if clean_path.exists():
                    try:
                        clean_image = torch.load(clean_path, map_location='cpu')
                        
                        # Handle different tensor formats
                        if isinstance(clean_image, dict):
                            if 'clean' in clean_image:
                                clean_image = clean_image['clean']
                            elif 'image' in clean_image:
                                clean_image = clean_image['image']
                        
                        clean_analysis = analyze_image_noise(clean_image, f"{example_dir.name}_clean")
                        all_clean_analysis[example_dir.name] = clean_analysis
                        
                        logger.info(f"  Found clean tile: {clean_path.name}")
                        logger.info(f"  Clean mean: {clean_image.mean():.6f}, Clean std: {clean_image.std():.6f}")
                    except Exception as e:
                        logger.warning(f"  Failed to load clean tile: {e}")
                else:
                    logger.info(f"  Clean tile not found: {clean_path}")
    
    # Create visualization
    create_noise_analysis_visualization(all_analysis, results_dir)
    
    # Create histogram visualization
    create_histogram_visualization(all_analysis, all_clean_analysis, results_dir)
    
    # Provide recommendations
    provide_recommendations(all_analysis)


def analyze_image_noise(image: torch.Tensor, name: str) -> Dict:
    """Analyze noise characteristics of an image."""
    
    # Basic statistics
    min_val = image.min().item()
    max_val = image.max().item()
    mean_val = image.mean().item()
    std_val = image.std().item()
    
    # Estimate noise level using different methods
    
    # Method 1: Standard deviation
    estimated_noise_std = std_val
    
    # Method 2: High-frequency analysis (if image is large enough)
    if image.shape[-1] > 64 and image.shape[-2] > 64:
        # Extract center patch for analysis
        h, w = image.shape[-2], image.shape[-1]
        center_h, center_w = h // 2, w // 2
        patch_size = min(64, h, w)
        
        if image.ndim == 4:
            patch = image[0, :, center_h-patch_size//2:center_h+patch_size//2, 
                         center_w-patch_size//2:center_w+patch_size//2]
        elif image.ndim == 3:
            patch = image[:, center_h-patch_size//2:center_h+patch_size//2, 
                         center_w-patch_size//2:center_w+patch_size//2]
        else:
            patch = image[center_h-patch_size//2:center_h+patch_size//2, 
                         center_w-patch_size//2:center_w+patch_size//2]
        
        # Calculate high-frequency content
        if patch.ndim == 3:
            patch_gray = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]
        else:
            patch_gray = patch[0] if patch.ndim == 3 else patch
        
        # Apply Laplacian filter to detect edges/noise
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)
        
        patch_gray_2d = patch_gray.unsqueeze(0).unsqueeze(0)
        high_freq = F.conv2d(patch_gray_2d, laplacian_kernel, padding=1)
        high_freq_std = high_freq.std().item()
        
        # Alternative noise estimation
        estimated_noise_std_hf = high_freq_std / 4.0  # Rough scaling
    else:
        estimated_noise_std_hf = std_val
    
    # Method 3: Difference between consecutive pixels (if applicable)
    if image.shape[-1] > 1 and image.shape[-2] > 1:
        if image.ndim == 4:
            img_2d = image[0, 0] if image.shape[1] == 1 else image[0].mean(dim=0)
        elif image.ndim == 3:
            img_2d = image[0] if image.shape[0] == 1 else image.mean(dim=0)
        else:
            img_2d = image
        
        # Calculate horizontal and vertical differences
        h_diff = torch.diff(img_2d, dim=1)
        v_diff = torch.diff(img_2d, dim=0)
        
        # Estimate noise from differences
        h_noise_std = h_diff.std().item() / np.sqrt(2)  # Divide by sqrt(2) for difference
        v_noise_std = v_diff.std().item() / np.sqrt(2)
        estimated_noise_std_diff = (h_noise_std + v_noise_std) / 2
    else:
        estimated_noise_std_diff = std_val
    
    # Use the most conservative estimate
    estimated_noise_std = min(estimated_noise_std, estimated_noise_std_hf, estimated_noise_std_diff)
    
    # Calculate signal-to-noise ratio
    signal_power = mean_val ** 2
    noise_power = estimated_noise_std ** 2
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Categorize noise level
    if estimated_noise_std < 0.001:
        noise_category = "Very Low"
    elif estimated_noise_std < 0.01:
        noise_category = "Low"
    elif estimated_noise_std < 0.1:
        noise_category = "Medium"
    else:
        noise_category = "High"
    
    return {
        'name': name,
        'shape': image.shape,
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val,
        'estimated_noise_std': estimated_noise_std,
        'estimated_noise_std_hf': estimated_noise_std_hf,
        'estimated_noise_std_diff': estimated_noise_std_diff,
        'snr': snr,
        'noise_category': noise_category,
        'image': image
    }


def create_noise_analysis_visualization(all_analysis: Dict, results_dir: Path):
    """Create visualization of noise analysis."""
    logger.info("Creating noise analysis visualization...")
    
    n_examples = len(all_analysis)
    fig, axes = plt.subplots(2, n_examples, figsize=(5 * n_examples, 10))
    
    if n_examples == 1:
        axes = axes.reshape(2, 1)
    
    for i, (name, analysis) in enumerate(all_analysis.items()):
        image = analysis['image']
        
        # Top row: Original image
        if image.ndim == 4:
            img_display = image[0, 0].cpu().numpy() if image.shape[1] == 1 else image[0].mean(dim=0).cpu().numpy()
        elif image.ndim == 3:
            img_display = image[0].cpu().numpy() if image.shape[0] == 1 else image.mean(dim=0).cpu().numpy()
        else:
            img_display = image.cpu().numpy()
        
        # Normalize for display
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
        
        axes[0, i].imshow(img_display, cmap='gray')
        axes[0, i].set_title(f"{name}\nRange: [{analysis['min']:.4f}, {analysis['max']:.4f}]")
        axes[0, i].axis('off')
        
        # Bottom row: Noise analysis
        axes[1, i].text(0.1, 0.9, f"Mean: {analysis['mean']:.6f}", transform=axes[1, i].transAxes, fontsize=10)
        axes[1, i].text(0.1, 0.8, f"Std: {analysis['std']:.6f}", transform=axes[1, i].transAxes, fontsize=10)
        axes[1, i].text(0.1, 0.7, f"Est. Noise: {analysis['estimated_noise_std']:.6f}", transform=axes[1, i].transAxes, fontsize=10)
        axes[1, i].text(0.1, 0.6, f"SNR: {analysis['snr']:.2f} dB", transform=axes[1, i].transAxes, fontsize=10)
        axes[1, i].text(0.1, 0.5, f"Category: {analysis['noise_category']}", transform=axes[1, i].transAxes, fontsize=10)
        axes[1, i].text(0.1, 0.4, f"HF Noise: {analysis['estimated_noise_std_hf']:.6f}", transform=axes[1, i].transAxes, fontsize=10)
        axes[1, i].text(0.1, 0.3, f"Diff Noise: {analysis['estimated_noise_std_diff']:.6f}", transform=axes[1, i].transAxes, fontsize=10)
        axes[1, i].set_title("Noise Analysis")
        axes[1, i].axis('off')
    
    plt.suptitle('Noise Level Analysis in Test Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save visualization
    noise_analysis_path = results_dir / 'noise_level_analysis.png'
    plt.savefig(noise_analysis_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Noise analysis visualization saved: {noise_analysis_path}")


def create_histogram_visualization(all_analysis: Dict, all_clean_analysis: Dict, results_dir: Path):
    """Create histogram distributions for pixel values and noise per test image."""
    logger.info("Creating histogram visualization...")
    
    n_examples = len(all_analysis)
    has_clean = len(all_clean_analysis) > 0
    
    # Create figure with 5 rows if clean data available, otherwise 3 rows
    n_rows = 5 if has_clean else 3
    fig = plt.figure(figsize=(6 * n_examples, 4.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_examples, hspace=0.35, wspace=0.3)
    
    for i, (name, analysis) in enumerate(all_analysis.items()):
        image = analysis['image']
        
        # Extract pixel values
        if image.ndim == 4:
            img_flat = image[0].flatten().cpu().numpy()
        elif image.ndim == 3:
            img_flat = image.flatten().cpu().numpy()
        else:
            img_flat = image.flatten().cpu().numpy()
        
        # Row 1: Pixel Value Histogram
        ax1 = fig.add_subplot(gs[0, i])
        ax1.hist(img_flat, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
        ax1.axvline(analysis['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {analysis['mean']:.4f}")
        ax1.axvline(analysis['mean'] - analysis['std'], color='orange', linestyle=':', linewidth=1.5, label=f"±1σ")
        ax1.axvline(analysis['mean'] + analysis['std'], color='orange', linestyle=':', linewidth=1.5)
        ax1.set_xlabel('Pixel Value', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f"{name}\nPixel Value Distribution", fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Row 2: Noise Estimation Histogram (using high-frequency content)
        # Calculate local differences as proxy for noise
        h_diff = None  # Initialize for later comparison
        if image.shape[-1] > 1 and image.shape[-2] > 1:
            if image.ndim == 4:
                img_2d = image[0, 0].cpu().numpy() if image.shape[1] == 1 else image[0].mean(dim=0).cpu().numpy()
            elif image.ndim == 3:
                img_2d = image[0].cpu().numpy() if image.shape[0] == 1 else image.mean(dim=0).cpu().numpy()
            else:
                img_2d = image.cpu().numpy()
            
            # Calculate horizontal differences
            h_diff = np.diff(img_2d, axis=1).flatten()
            
            ax2 = fig.add_subplot(gs[1, i])
            ax2.hist(h_diff, bins=100, alpha=0.7, edgecolor='black', color='coral')
            ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax2.axvline(h_diff.std(), color='red', linestyle='--', linewidth=2, label=f"Noise σ: {h_diff.std():.6f}")
            ax2.axvline(-h_diff.std(), color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Pixel-to-Pixel Difference', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title(f"Noise Distribution (Horizontal Diff)", fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
        else:
            ax2 = fig.add_subplot(gs[1, i])
            ax2.text(0.5, 0.5, 'Image too small\nfor noise analysis', 
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.axis('off')
        
        # Row 3: Statistics Summary
        ax3 = fig.add_subplot(gs[2, i])
        ax3.axis('off')
        
        stats_text = (
            f"Noisy Tile Statistics:\n\n"
            f"Pixel Value Statistics:\n"
            f"  Mean: {analysis['mean']:.6f}\n"
            f"  Std: {analysis['std']:.6f}\n"
            f"  Min: {analysis['min']:.6f}\n"
            f"  Max: {analysis['max']:.6f}\n\n"
            f"Noise Estimates:\n"
            f"  Overall: {analysis['estimated_noise_std']:.6f}\n"
            f"  High-Freq: {analysis['estimated_noise_std_hf']:.6f}\n"
            f"  Diff-Based: {analysis['estimated_noise_std_diff']:.6f}\n\n"
            f"Signal Quality:\n"
            f"  SNR: {analysis['snr']:.2f} dB\n"
            f"  Category: {analysis['noise_category']}\n"
        )
        
        ax3.text(0.1, 0.95, stats_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Rows 4 & 5: Clean tile comparison (if available)
        if has_clean and name in all_clean_analysis:
            clean_analysis = all_clean_analysis[name]
            clean_image = clean_analysis['image']
            
            # Extract clean pixel values
            if clean_image.ndim == 4:
                clean_img_flat = clean_image[0].flatten().cpu().numpy()
            elif clean_image.ndim == 3:
                clean_img_flat = clean_image.flatten().cpu().numpy()
            else:
                clean_img_flat = clean_image.flatten().cpu().numpy()
            
            # Row 4: Clean Pixel Value Histogram
            ax4 = fig.add_subplot(gs[3, i])
            ax4.hist(clean_img_flat, bins=100, alpha=0.7, edgecolor='black', color='green')
            ax4.axvline(clean_analysis['mean'], color='darkgreen', linestyle='--', linewidth=2, 
                       label=f"Mean: {clean_analysis['mean']:.4f}")
            ax4.axvline(clean_analysis['mean'] - clean_analysis['std'], color='olive', linestyle=':', linewidth=1.5, label=f"±1σ")
            ax4.axvline(clean_analysis['mean'] + clean_analysis['std'], color='olive', linestyle=':', linewidth=1.5)
            ax4.set_xlabel('Pixel Value (Clean)', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title(f"Clean Tile Pixel Distribution", fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            
            # Row 5: Clean Noise Distribution
            if clean_image.shape[-1] > 1 and clean_image.shape[-2] > 1:
                if clean_image.ndim == 4:
                    clean_img_2d = clean_image[0, 0].cpu().numpy() if clean_image.shape[1] == 1 else clean_image[0].mean(dim=0).cpu().numpy()
                elif clean_image.ndim == 3:
                    clean_img_2d = clean_image[0].cpu().numpy() if clean_image.shape[0] == 1 else clean_image.mean(dim=0).cpu().numpy()
                else:
                    clean_img_2d = clean_image.cpu().numpy()
                
                # Calculate horizontal differences for clean
                clean_h_diff = np.diff(clean_img_2d, axis=1).flatten()
                
                ax5 = fig.add_subplot(gs[4, i])
                ax5.hist(clean_h_diff, bins=100, alpha=0.7, edgecolor='black', color='lightgreen')
                ax5.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
                ax5.axvline(clean_h_diff.std(), color='darkgreen', linestyle='--', linewidth=2, 
                           label=f"Clean σ: {clean_h_diff.std():.6f}")
                ax5.axvline(-clean_h_diff.std(), color='darkgreen', linestyle='--', linewidth=2)
                ax5.set_xlabel('Pixel-to-Pixel Difference (Clean)', fontsize=11)
                ax5.set_ylabel('Frequency', fontsize=11)
                ax5.set_title(f"Clean Tile Noise Distribution", fontsize=12, fontweight='bold')
                ax5.legend(fontsize=9)
                ax5.grid(True, alpha=0.3)
                
                # Add comparison text
                if h_diff is not None:
                    noise_reduction = ((h_diff.std() - clean_h_diff.std()) / h_diff.std() * 100)
                    ax5.text(0.98, 0.95, f"Noise Δ: {noise_reduction:.1f}% reduction\nNoisy: {h_diff.std():.6f}\nClean: {clean_h_diff.std():.6f}", 
                            transform=ax5.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            else:
                ax5 = fig.add_subplot(gs[4, i])
                ax5.text(0.5, 0.5, 'Image too small\nfor noise analysis', 
                        ha='center', va='center', fontsize=12, transform=ax5.transAxes)
                ax5.axis('off')
        elif has_clean:
            # No clean tile available for this sample
            ax4 = fig.add_subplot(gs[3, i])
            ax4.text(0.5, 0.5, 'Clean tile\nnot available', 
                    ha='center', va='center', fontsize=12, transform=ax4.transAxes)
            ax4.axis('off')
            
            ax5 = fig.add_subplot(gs[4, i])
            ax5.text(0.5, 0.5, 'Clean tile\nnot available', 
                    ha='center', va='center', fontsize=12, transform=ax5.transAxes)
            ax5.axis('off')
    
    title = 'Pixel Value and Noise Distribution Analysis'
    if has_clean:
        title += ' (Noisy vs Clean Comparison)'
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    
    # Save histogram visualization
    histogram_path = results_dir / 'pixel_noise_histograms.png'
    plt.savefig(histogram_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Histogram visualization saved: {histogram_path}")


def provide_recommendations(all_analysis: Dict):
    """Provide recommendations based on noise analysis."""
    logger.info("\n" + "="*80)
    logger.info("NOISE ANALYSIS RECOMMENDATIONS")
    logger.info("="*80)
    
    # Collect statistics
    noise_stds = [analysis['estimated_noise_std'] for analysis in all_analysis.values()]
    snrs = [analysis['snr'] for analysis in all_analysis.values()]
    
    min_noise = min(noise_stds)
    max_noise = max(noise_stds)
    avg_noise = np.mean(noise_stds)
    
    min_snr = min(snrs)
    max_snr = max(snrs)
    avg_snr = np.mean(snrs)
    
    logger.info(f"NOISE STATISTICS:")
    logger.info(f"  Estimated noise std: {min_noise:.6f} - {max_noise:.6f} (avg: {avg_noise:.6f})")
    logger.info(f"  Signal-to-noise ratio: {min_snr:.2f} - {max_snr:.2f} dB (avg: {avg_snr:.2f} dB)")
    
    # Analyze each example
    for name, analysis in all_analysis.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Noise level: {analysis['noise_category']}")
        logger.info(f"  Estimated noise std: {analysis['estimated_noise_std']:.6f}")
        logger.info(f"  SNR: {analysis['snr']:.2f} dB")
        
        # Provide specific recommendations
        if analysis['estimated_noise_std'] < 0.0001:
            logger.info(f"  → Very clean image - denoising may not be necessary")
            logger.info(f"  → If denoising, use very small sigma (< 0.0001)")
        elif analysis['estimated_noise_std'] < 0.001:
            logger.info(f"  → Low noise - use small sigma (0.0001 - 0.001)")
        elif analysis['estimated_noise_std'] < 0.01:
            logger.info(f"  → Medium noise - use moderate sigma (0.001 - 0.01)")
        else:
            logger.info(f"  → High noise - use larger sigma (0.01 - 0.1)")
    
    # Overall recommendations
    logger.info(f"\nOVERALL RECOMMENDATIONS:")
    
    if avg_noise < 0.0001:
        logger.info("✓ Images are very clean - denoising may not be beneficial")
        logger.info("✓ If denoising, use sigma < 0.0001")
        logger.info("✓ Consider testing even lower sigma values (0.0000001 - 0.000001)")
    elif avg_noise < 0.001:
        logger.info("✓ Images have low noise - use small sigma values")
        logger.info("✓ Recommended sigma range: 0.0001 - 0.001")
    elif avg_noise < 0.01:
        logger.info("✓ Images have medium noise - use moderate sigma values")
        logger.info("✓ Recommended sigma range: 0.001 - 0.01")
    else:
        logger.info("✓ Images have high noise - use larger sigma values")
        logger.info("✓ Recommended sigma range: 0.01 - 0.1")
    
    # Check if current sigma range is appropriate
    if avg_noise < 0.0001 and min_noise < 0.0001:
        logger.info("⚠ Current sigma range (0.000001 - 0.01) may be too high for these clean images")
        logger.info("✓ Consider testing much lower sigma values (0.0000001 - 0.000001)")
    elif avg_noise > 0.01:
        logger.info("⚠ Current sigma range (0.000001 - 0.01) may be too low for these noisy images")
        logger.info("✓ Consider testing higher sigma values (0.01 - 0.1)")
    else:
        logger.info("✓ Current sigma range (0.000001 - 0.01) is appropriate")
    
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Diagnose noise levels in test images")
    
    parser.add_argument("--results_dir", type=str, required=True, 
                       help="Directory containing sigma sweep results")
    parser.add_argument("--domain", type=str, default="photography",
                       choices=["photography", "microscopy", "astronomy"],
                       help="Domain for analysis")
    parser.add_argument("--clean_dir", type=str, default=None,
                       help="Directory containing clean tile .pt files for comparison")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    clean_dir = Path(args.clean_dir) if args.clean_dir else None
    if clean_dir and not clean_dir.exists():
        logger.warning(f"Clean directory not found: {clean_dir}, proceeding without clean comparison")
        clean_dir = None
    
    # Analyze noise levels
    analyze_noise_levels(results_dir, args.domain, clean_dir)
    
    logger.info("=" * 80)
    logger.info("🎉 NOISE LEVEL ANALYSIS COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
