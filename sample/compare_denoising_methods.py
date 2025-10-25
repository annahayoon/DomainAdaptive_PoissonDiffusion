#!/usr/bin/env python3
"""
Comprehensive Comparison of Denoising Methods

This script compares different denoising approaches by loading existing results:
1. Low-light enhancement with PG guidance (from low_light_enhancement_pg/)
2. Low-light enhancement without PG guidance (from low_light_local_variance/)
3. Synthetic noise denoising (from synthetic_noise_denoising_test/)

Usage:
    python sample/compare_denoising_methods.py \
        --pg_results_dir results/low_light_enhancement_pg \
        --local_var_results_dir results/low_light_local_variance \
        --synthetic_results_dir results/synthetic_noise_denoising_test \
        --output_dir results/denoising_comparison \
        --domain photography \
        --num_examples 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Setup logging
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResultsLoader:
    """Load existing denoising results from different directories."""
    
    @staticmethod
    def load_pg_results(pg_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Load PG guidance results."""
        results = {}
        
        # Load summary
        summary_path = pg_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            for result_info in summary.get('results', []):
                tile_id = result_info['tile_id']
                
                # Find corresponding example directory
                example_dirs = [d for d in pg_dir.iterdir() if d.is_dir() and tile_id in d.name]
                if example_dirs:
                    example_dir = example_dirs[0]
                    
                    # Load enhanced result
                    enhanced_files = list(example_dir.glob("enhanced_sigma_*.pt"))
                    if enhanced_files:
                        enhanced_tensor = torch.load(enhanced_files[0])
                        results[tile_id] = {
                            'enhanced': enhanced_tensor,
                            'metadata': result_info,
                            'method': 'PG Guidance'
                        }
        
        return results
    
    @staticmethod
    def load_local_var_results(local_var_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Load local variance results (without PG guidance)."""
        results = {}
        
        # Load summary
        summary_path = local_var_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            for result_info in summary.get('results', []):
                tile_id = result_info['tile_id']
                
                # Find corresponding example directory
                example_dirs = [d for d in local_var_dir.iterdir() if d.is_dir() and tile_id in d.name]
                if example_dirs:
                    example_dir = example_dirs[0]
                    
                    # Load enhanced result
                    enhanced_files = list(example_dir.glob("enhanced_sigma_*.pt"))
                    if enhanced_files:
                        enhanced_tensor = torch.load(enhanced_files[0])
                        results[tile_id] = {
                            'enhanced': enhanced_tensor,
                            'metadata': result_info,
                            'method': 'Local Variance'
                        }
        
        return results
    
    @staticmethod
    def load_normal_guidance_results(normal_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Load normal L2 guidance results."""
        results = {}
        
        # Load summary
        summary_path = normal_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            for result_info in summary.get('results', []):
                tile_id = result_info['tile_id']
                
                # Find corresponding example directory
                example_dirs = [d for d in normal_dir.iterdir() if d.is_dir() and tile_id in d.name]
                if example_dirs:
                    example_dir = example_dirs[0]
                    
                    # Load enhanced result (different format: enhanced.pt instead of enhanced_sigma_*.pt)
                    enhanced_path = example_dir / "enhanced.pt"
                    if enhanced_path.exists():
                        enhanced_tensor = torch.load(enhanced_path)
                        results[tile_id] = {
                            'enhanced': enhanced_tensor,
                            'metadata': result_info,
                            'method': 'Normal L2 Guidance'
                        }
        
        return results
    
    @staticmethod
    def load_synthetic_results(synthetic_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Load synthetic noise denoising results."""
        results = {}
        
        # Find all sample directories
        sample_dirs = [d for d in synthetic_dir.iterdir() if d.is_dir() and d.name.startswith('sample_')]
        
        for sample_dir in sample_dirs:
            # Load denoised, clean, and noisy results
            denoised_path = sample_dir / "denoised.pt"
            clean_path = sample_dir / "clean.pt"
            noisy_path = sample_dir / "noisy.pt"
            
            if denoised_path.exists() and clean_path.exists() and noisy_path.exists():
                denoised_tensor = torch.load(denoised_path)
                clean_tensor = torch.load(clean_path)
                noisy_tensor = torch.load(noisy_path)
                
                # Extract tile info from directory name or create synthetic ID
                sample_name = sample_dir.name
                results[sample_name] = {
                    'denoised': denoised_tensor,
                    'clean': clean_tensor,
                    'noisy': noisy_tensor,
                    'method': 'Synthetic Denoising'
                }
        
        return results


def compute_metrics(clean: np.ndarray, enhanced: np.ndarray, data_range: float = None) -> Dict[str, float]:
    """
    Compute SSIM and PSNR between clean and enhanced images.
    
    Args:
        clean: Clean reference image
        enhanced: Enhanced image
        data_range: Data range for metrics (if None, computed from clean)
    
    Returns:
        Dictionary with metrics
    """
    # Ensure 2D arrays for grayscale
    if clean.ndim == 4:  # (B, C, H, W)
        clean = clean[0, 0]
    elif clean.ndim == 3:  # (C, H, W)
        clean = clean[0]
    
    if enhanced.ndim == 4:  # (B, C, H, W)
        enhanced = enhanced[0, 0]
    elif enhanced.ndim == 3:  # (C, H, W)
        enhanced = enhanced[0]
    
    # Check if shapes match, if not return None
    if clean.shape != enhanced.shape:
        logger.warning(f"Shape mismatch: clean {clean.shape} vs enhanced {enhanced.shape}, skipping metrics")
        return {
            'ssim': float('nan'),
            'psnr': float('nan'),
            'mse': float('nan'),
        }
    
    if data_range is None:
        data_range = clean.max() - clean.min()
    
    # Compute metrics
    ssim_val = ssim(clean, enhanced, data_range=data_range)
    psnr_val = psnr(clean, enhanced, data_range=data_range)
    
    # Also compute MSE
    mse = np.mean((clean - enhanced) ** 2)
    
    return {
        'ssim': float(ssim_val),
        'psnr': float(psnr_val),
        'mse': float(mse),
    }


def load_noisy_image_from_results(tile_id: str, results_dir: Path) -> Optional[torch.Tensor]:
    """Load noisy image from results directory."""
    # Find example directory containing this tile_id
    example_dirs = [d for d in results_dir.iterdir() if d.is_dir() and tile_id in d.name]
    if example_dirs:
        example_dir = example_dirs[0]
        noisy_path = example_dir / "noisy.pt"
        if noisy_path.exists():
            return torch.load(noisy_path)
    return None


def find_corresponding_clean_reference(tile_id: str, domain: str) -> Optional[torch.Tensor]:
    """
    Find the clean reference image that corresponds to the noisy low-light image.
    
    For SID dataset, this means finding the long exposure version of the same scene.
    Example: photography_fuji_00030_00_0.033s_tile_0019 -> photography_fuji_00030_00_10s_tile_0019
    """
    if domain != "photography":
        logger.warning(f"Clean reference lookup not implemented for domain: {domain}")
        return None
    
    # Extract scene info from tile_id
    # Format: photography_fuji_00030_00_0.033s_tile_0019
    parts = tile_id.split('_')
    if len(parts) < 6:
        logger.warning(f"Unexpected tile_id format: {tile_id}")
        return None
    
    # Try different exposure times for clean reference
    clean_dir = Path("dataset/processed/pt_tiles/photography/clean")
    if not clean_dir.exists():
        logger.warning(f"Clean directory not found: {clean_dir}")
        return None
    
    # Try different exposure times: 10s, 30s, etc.
    exposure_times = ["10s", "30s", "1s", "5s"]
    
    for exposure_time in exposure_times:
        clean_parts = parts.copy()
        # Find the exposure time part and replace with clean exposure time
        for i, part in enumerate(clean_parts):
            if part.endswith('s') and '.' in part:
                clean_parts[i] = exposure_time
                break
        
        clean_tile_id = '_'.join(clean_parts)
        clean_path = clean_dir / f"{clean_tile_id}.pt"
        
        if clean_path.exists():
            logger.info(f"Found clean reference: {clean_path}")
            return torch.load(clean_path)
    
    # Fallback: search for any clean reference with similar scene ID
    # Extract scene ID (without tile number and exposure time)
    scene_parts = parts[:-3]  # Remove exposure time and tile number
    scene_id = '_'.join(scene_parts)
    
    # Find any clean reference from the same scene
    clean_files = list(clean_dir.glob(f"{scene_id}_*_tile_*.pt"))
    if clean_files:
        # Try to find the same tile number
        target_tile = parts[-1]  # e.g., "tile_0035"
        matching_files = [f for f in clean_files if target_tile in f.name]
        if matching_files:
            clean_path = matching_files[0]
            logger.info(f"Using clean reference with same tile: {clean_path}")
            return torch.load(clean_path)
        else:
            # Use any available clean reference from same scene
            clean_path = clean_files[0]
            logger.info(f"Using clean reference from same scene: {clean_path}")
            return torch.load(clean_path)
    
    logger.warning(f"Clean reference not found for: {clean_tile_id}")
    return None


def generate_synthetic_clean_reference(noisy_image: torch.Tensor, domain: str) -> torch.Tensor:
    """
    Generate a synthetic clean reference by scaling up the noisy image.
    This simulates what the clean reference would look like for fair comparison.
    """
    # Convert from [-1, 1] to [0, 1] range
    noisy_norm = (noisy_image + 1.0) / 2.0
    
    # Scale up to simulate longer exposure (clean reference should be brighter)
    # Use a reasonable scaling factor based on exposure time ratio
    # If noisy is 0.04s and clean is 10s, ratio is 250x
    # But we'll use a more moderate scaling for visualization
    scale_factor = 50.0  # Moderate scaling for visualization
    
    clean_norm = noisy_norm * scale_factor
    clean_norm = torch.clamp(clean_norm, 0, 1)
    
    # Convert back to [-1, 1] range
    clean_image = clean_norm * 2.0 - 1.0
    
    logger.info(f"Generated synthetic clean reference with scale factor {scale_factor}")
    logger.info(f"Noisy range: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")
    logger.info(f"Clean range: [{clean_image.min():.3f}, {clean_image.max():.3f}]")
    return clean_image


def analyze_image_brightness(image: torch.Tensor) -> Dict[str, float]:
    """Analyze image brightness characteristics."""
    # Convert to [0, 1] range for analysis
    img_01 = (image + 1.0) / 2.0
    
    # Calculate statistics
    mean_brightness = img_01.mean().item()
    std_brightness = img_01.std().item()
    min_brightness = img_01.min().item()
    max_brightness = img_01.max().item()
    
    # Calculate percentile brightness
    img_flat = img_01.flatten()
    p10 = torch.quantile(img_flat, 0.1).item()
    p50 = torch.quantile(img_flat, 0.5).item()
    p90 = torch.quantile(img_flat, 0.9).item()
    
    # Categorize brightness
    if mean_brightness < 0.2:
        brightness_category = "Very Dark"
    elif mean_brightness < 0.4:
        brightness_category = "Dark"
    elif mean_brightness < 0.6:
        brightness_category = "Medium"
    elif mean_brightness < 0.8:
        brightness_category = "Bright"
    else:
        brightness_category = "Very Bright"
    
    return {
        'mean': mean_brightness,
        'std': std_brightness,
        'min': min_brightness,
        'max': max_brightness,
        'p10': p10,
        'p50': p50,
        'p90': p90,
        'category': brightness_category
    }


def denormalize_to_physical(tensor: torch.Tensor, domain: str) -> np.ndarray:
    """
    Convert tensor from [-1,1] model space to physical units.

    Args:
        tensor: Image tensor in [-1, 1] range, shape (B, C, H, W)
        domain: Domain name for range lookup

    Returns:
        Image array in physical units
    """
    domain_ranges = {
        "photography": {"min": 0.0, "max": 15871.0},
        "microscopy": {"min": 0.0, "max": 65535.0},
        "astronomy": {"min": -65.0, "max": 385.0},
    }
    
    domain_range = domain_ranges.get(domain, {"min": 0.0, "max": 1.0})

    # Step 1: [-1,1] → [0,1]
    tensor_norm = (tensor + 1.0) / 2.0
    tensor_norm = torch.clamp(tensor_norm, 0, 1)

    # Step 2: [0,1] → [domain_min, domain_max]
    domain_min = domain_range["min"]
    domain_max = domain_range["max"]
    tensor_phys = tensor_norm * (domain_max - domain_min) + domain_min

    return tensor_phys.cpu().numpy()


def create_comparison_visualization(
    noisy_image: torch.Tensor,
    method_results: Dict[str, torch.Tensor],
    domain: str,
    tile_id: str,
    save_path: Path,
    clean_image: Optional[torch.Tensor] = None,
    synthetic_noisy_image: Optional[torch.Tensor] = None,
):
    """Create comprehensive comparison visualization."""
    logger.info("Creating comprehensive denoising comparison visualization...")
    
    # Get method names and results
    method_names = list(method_results.keys())
    n_methods = len(method_names)
    
    # Add clean image panel if available
    n_panels = n_methods + 1  # +1 for noisy input
    if clean_image is not None:
        n_panels += 1  # +1 for clean reference
    
    # Create figure with one row: noisy input, clean reference (if available), and all method results
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))
    
    def prepare_for_display(phys_array, scale_range=None):
        """
        Prepare physical array for display using percentile normalization.
        """
        # Handle different array shapes
        if phys_array.ndim == 4:  # (B, C, H, W)
            if phys_array.shape[1] == 3:  # RGB
                img = np.transpose(phys_array[0], (1, 2, 0))
            else:  # Grayscale
                img = phys_array[0, 0]
        elif phys_array.ndim == 3:  # (C, H, W)
            if phys_array.shape[0] == 3:  # RGB
                img = np.transpose(phys_array, (1, 2, 0))
            else:
                img = phys_array[0]
        elif phys_array.ndim == 2:  # (H, W)
            img = phys_array
        else:
            img = phys_array
        
        # Get statistics
        min_val = float(np.min(img))
        max_val = float(np.max(img))
        
        # Compute or use provided scale range
        valid_mask = np.isfinite(img)
        if scale_range is None:
            # Compute scale from this image's percentiles
            if np.any(valid_mask):
                p1, p99 = np.percentile(img[valid_mask], [1, 99])
                scale_min, scale_max = p1, p99
            else:
                p1, p99 = min_val, max_val
                scale_min, scale_max = min_val, max_val
        else:
            # Use provided scale
            scale_min, scale_max = scale_range
            if np.any(valid_mask):
                p1, p99 = np.percentile(img[valid_mask], [1, 99])
            else:
                p1, p99 = min_val, max_val
        
        # Apply normalization using the scale
        img_clipped = np.clip(img, scale_min, scale_max)
        img_norm = (img_clipped - scale_min) / (scale_max - scale_min + 1e-8)
        
        return img_norm, min_val, max_val, p1, p99, scale_min, scale_max
    
    # Get domain-specific unit label
    domain_units = {
        "photography": "ADU",
        "microscopy": "intensity",
        "astronomy": "counts",
    }
    unit_label = domain_units.get(domain, "units")
    
    # Process all images and find optimal scaling strategy
    all_phys_arrays = []
    all_ranges = []
    
    # Add noisy image
    noisy_phys = denormalize_to_physical(noisy_image, domain)
    all_phys_arrays.append(noisy_phys)
    all_ranges.append((noisy_phys.min(), noisy_phys.max()))
    
    # Add clean image if available
    if clean_image is not None:
        clean_phys = denormalize_to_physical(clean_image, domain)
        all_phys_arrays.append(clean_phys)
        all_ranges.append((clean_phys.min(), clean_phys.max()))
    
    # Removed synthetic noisy image processing
    
    # Add method results
    for method_name, result_tensor in method_results.items():
        result_phys = denormalize_to_physical(result_tensor, domain)
        all_phys_arrays.append(result_phys)
        all_ranges.append((result_phys.min(), result_phys.max()))
    
    # Always use adaptive scaling for individual panels to show details
    use_adaptive_scaling = True
    logger.info(f"Using adaptive scaling for individual panels to show details")
    
    # Process noisy input with its own scaling
    noisy_display, noisy_min, noisy_max, noisy_p1, noisy_p99, noisy_scale_min, noisy_scale_max = prepare_for_display(noisy_phys)
    
    # Display noisy input
    axes[0].imshow(noisy_display, cmap='gray' if noisy_image.shape[1] == 1 else None)
    axes[0].set_title(
        f"Noisy Input\n"
        f"Range: [{noisy_min:.1f}, {noisy_max:.1f}] {unit_label}\n"
        f"Display Scale: [{noisy_scale_min:.1f}, {noisy_scale_max:.1f}]", 
        fontsize=8
    )
    axes[0].axis('off')
    
    # Display clean reference if available
    panel_idx = 1
    if clean_image is not None:
        clean_phys = denormalize_to_physical(clean_image, domain)
        clean_display, clean_min, clean_max, clean_p1, clean_p99, clean_scale_min, clean_scale_max = prepare_for_display(clean_phys)
        
        axes[panel_idx].imshow(clean_display, cmap='gray' if clean_display.ndim == 2 else None)
        axes[panel_idx].set_title(
            f"Clean Reference (Paired)\n"
            f"Range: [{clean_min:.1f}, {clean_max:.1f}] {unit_label}\n"
            f"Display Scale: [{clean_scale_min:.1f}, {clean_scale_max:.1f}]", 
            fontsize=8
        )
        axes[panel_idx].axis('off')
        panel_idx += 1
    
    # Removed synthetic noisy input panel
    
    # Process and display method results with metrics
    for i, method_name in enumerate(method_names):
        result_tensor = method_results[method_name]  # Direct tensor access
        
        result_phys = denormalize_to_physical(result_tensor, domain)
        result_display, res_min, res_max, res_p1, res_p99, res_scale_min, res_scale_max = prepare_for_display(result_phys)
        
        axes[panel_idx + i].imshow(result_display, cmap='gray' if result_display.ndim == 2 else None)
        
        # Compute metrics if clean reference is available
        title_text = f"{method_name.replace('_', ' ').title()}\n"
        if clean_image is not None:
            # Compute PSNR and SSIM
            metrics = compute_metrics(
                clean_image.cpu().numpy(), 
                result_tensor.cpu().numpy()
            )
            if not np.isnan(metrics['psnr']):
                title_text += f"PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.3f}\n"
            else:
                title_text += f"Metrics: N/A (shape mismatch)\n"
        
        title_text += f"Range: [{res_min:.1f}, {res_max:.1f}] {unit_label}\n"
        title_text += f"Display Scale: [{res_scale_min:.1f}, {res_scale_max:.1f}]"
        
        axes[panel_idx + i].set_title(title_text, fontsize=8)
        axes[panel_idx + i].axis('off')
    
    # Main title
    comparison_type = "Low-Light Enhancement" if clean_image is None else "Synthetic Denoising"
    plt.suptitle(
        f"{comparison_type} Methods Comparison (Physical Units: {unit_label}) - {tile_id}\n"
        f"Domain: {domain} | Each panel scaled to its own min/max for optimal visibility",
        fontsize=11,
        fontweight="bold"
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Comparison visualization saved: {save_path}")


def main():
    """Main function for denoising methods comparison."""
    parser = argparse.ArgumentParser(
        description="Comprehensive comparison of existing denoising methods"
    )

    # Results directories
    parser.add_argument("--pg_results_dir", type=str, required=True, 
                       help="Directory containing PG guidance results")
    parser.add_argument("--local_var_results_dir", type=str, required=True,
                       help="Directory containing local variance results")
    parser.add_argument("--normal_results_dir", type=str, required=True,
                       help="Directory containing normal guidance results")
    parser.add_argument("--synthetic_results_dir", type=str, required=True,
                       help="Directory containing synthetic denoising results")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/denoising_comparison")
    parser.add_argument("--domain", type=str, default="photography",
                       choices=["photography", "microscopy", "astronomy"])
    parser.add_argument("--num_examples", type=int, default=3, help="Number of example images to compare")

    # Device arguments
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE DENOISING METHODS COMPARISON")
    logger.info("=" * 80)
    logger.info(f"PG results dir: {args.pg_results_dir}")
    logger.info(f"Local var results dir: {args.local_var_results_dir}")
    logger.info(f"Normal results dir: {args.normal_results_dir}")
    logger.info(f"Synthetic results dir: {args.synthetic_results_dir}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Number of examples: {args.num_examples}")
    logger.info("=" * 80)

    # Load existing results
    logger.info("Loading existing denoising results...")
    
    pg_results = ResultsLoader.load_pg_results(Path(args.pg_results_dir))
    local_var_results = ResultsLoader.load_local_var_results(Path(args.local_var_results_dir))
    normal_results = ResultsLoader.load_normal_guidance_results(Path(args.normal_results_dir))
    synthetic_results = ResultsLoader.load_synthetic_results(Path(args.synthetic_results_dir))
    
    logger.info(f"Loaded {len(pg_results)} PG guidance results")
    logger.info(f"Loaded {len(local_var_results)} local variance results")
    logger.info(f"Loaded {len(normal_results)} normal guidance results")
    logger.info(f"Loaded {len(synthetic_results)} synthetic denoising results")
    
    # Find common tile IDs between all results
    pg_tile_ids = set(pg_results.keys())
    local_var_tile_ids = set(local_var_results.keys())
    normal_tile_ids = set(normal_results.keys())
    common_tile_ids = pg_tile_ids.intersection(local_var_tile_ids).intersection(normal_tile_ids)
    
    logger.info(f"Found {len(common_tile_ids)} common tile IDs between all results")
    
    if len(common_tile_ids) == 0:
        logger.error("No common tile IDs found between PG and local var results")
        return
    
    # Select examples to compare
    selected_tile_ids = list(common_tile_ids)[:args.num_examples]
    logger.info(f"Selected {len(selected_tile_ids)} tiles for comparison: {selected_tile_ids}")
    
    # Process each selected tile
    all_results = []
    
    for idx, tile_id in enumerate(selected_tile_ids):
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Processing example {idx+1}/{len(selected_tile_ids)}: {tile_id}")
        logger.info(f"{'='*60}")
        
        try:
            # Load noisy image from PG results directory
            noisy_image = load_noisy_image_from_results(tile_id, Path(args.pg_results_dir))
            if noisy_image is None:
                logger.warning(f"Noisy image not found for {tile_id}, skipping...")
                continue
            
            if noisy_image.ndim == 3:
                noisy_image = noisy_image.unsqueeze(0)
            
            # Analyze noisy image brightness
            noisy_brightness = analyze_image_brightness(noisy_image)
            noisy_phys = denormalize_to_physical(noisy_image, args.domain)
            
            domain_units = {
                "photography": "ADU",
                "microscopy": "intensity", 
                "astronomy": "counts",
            }
            unit_label = domain_units.get(args.domain, "units")
            
            logger.info(f"  Brightness: {noisy_brightness['category']} (mean={noisy_brightness['mean']:.3f} normalized)")
            logger.info(f"  Normalized range: [{noisy_image.min():.4f}, {noisy_image.max():.4f}], std={noisy_image.std():.4f}")
            logger.info(f"  Physical range: [{noisy_phys.min():.1f}, {noisy_phys.max():.1f}] {unit_label}")
            
            # Collect method results
            method_results = {}
            
            # 1. PG Guidance result
            if tile_id in pg_results:
                pg_result = pg_results[tile_id]['enhanced']
                method_results['pg_guidance'] = pg_result
                logger.info(f"  ✓ Loaded PG guidance result: range [{pg_result.min():.3f}, {pg_result.max():.3f}]")
            
            # 2. Local Variance result (no PG guidance)
            if tile_id in local_var_results:
                local_var_result = local_var_results[tile_id]['enhanced']
                method_results['local_variance'] = local_var_result
                logger.info(f"  ✓ Loaded local variance result: range [{local_var_result.min():.3f}, {local_var_result.max():.3f}]")
            
            # 3. Normal Guidance result (without local variance)
            if tile_id in normal_results:
                normal_result = normal_results[tile_id]['enhanced']
                method_results['normal_guidance'] = normal_result
                logger.info(f"  ✓ Loaded normal guidance result: range [{normal_result.min():.3f}, {normal_result.max():.3f}]")
            
            # 4. Find corresponding clean reference for fair comparison
            clean_image = find_corresponding_clean_reference(tile_id, args.domain)
            
            if clean_image is not None:
                # Found real clean reference
                logger.info(f"  ✓ Found corresponding clean reference: range [{clean_image.min():.3f}, {clean_image.max():.3f}]")
            else:
                logger.warning(f"  ⚠ No clean reference found for {tile_id}, skipping clean reference panel")
            
            # Remove synthetic denoising from method results
            if 'synthetic_denoising' in method_results:
                del method_results['synthetic_denoising']
                logger.info(f"  ✓ Removed synthetic denoising from comparison")
            
            if not method_results:
                logger.warning(f"No method results found for {tile_id}, skipping...")
                continue
            
            # Save results
            sample_dir = output_dir / f"example_{idx:02d}_{tile_id}"
            sample_dir.mkdir(exist_ok=True)
            
            # Save tensors
            torch.save(noisy_image.cpu(), sample_dir / "noisy.pt")
            
            # Save method results
            for method_name, result_tensor in method_results.items():
                torch.save(result_tensor.cpu(), sample_dir / f"{method_name}.pt")
            
            # Save metadata
            result_info = {
                'tile_id': tile_id,
                'brightness_analysis': noisy_brightness,
                'methods_compared': list(method_results.keys()),
                'pg_metadata': pg_results[tile_id]['metadata'] if tile_id in pg_results else None,
                'local_var_metadata': local_var_results[tile_id]['metadata'] if tile_id in local_var_results else None,
            }
            
            with open(sample_dir / "results.json", 'w') as f:
                json.dump(result_info, f, indent=2)
            
            all_results.append(result_info)
            
            # Create fair comparison visualization using same-scene pairs
            comparison_path = sample_dir / "methods_comparison.png"
            create_comparison_visualization(
                noisy_image=noisy_image,
                method_results=method_results,
                domain=args.domain,
                tile_id=tile_id,
                save_path=comparison_path,
                clean_image=clean_image,  # Use corresponding clean reference from same scene
                synthetic_noisy_image=None,  # No synthetic panels
            )
            logger.info(f"✓ Created fair comparison with same-scene pairs: {comparison_path}")
            
            logger.info(f"✓ Saved to {sample_dir}")
            
        except Exception as e:
            logger.error(f"Failed to process tile {tile_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary = {
        'domain': args.domain,
        'num_samples': len(all_results),
        'pg_results_dir': args.pg_results_dir,
        'local_var_results_dir': args.local_var_results_dir,
        'synthetic_results_dir': args.synthetic_results_dir,
        'results': all_results,
    }
    
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 DENOISING METHODS COMPARISON COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"📁 All results saved to: {output_dir}")
    logger.info(f"📊 Processed {len(all_results)} tiles")
    logger.info(f"🔬 Methods compared: PG Guidance, Local Variance, Synthetic Denoising")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
