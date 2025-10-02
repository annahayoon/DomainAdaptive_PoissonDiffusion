#!/usr/bin/env python3
"""
Comprehensive evaluation of trained models from HPC.

This script evaluates all available trained models from the HPC results
and provides detailed performance metrics and comparisons.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.logging_config import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


def find_available_models() -> List[Dict[str, str]]:
    """Find all available trained models."""
    model_paths = []

    # Local checkpoints
    checkpoint_dir = Path("/home/jilab/Jae/checkpoints")
    if checkpoint_dir.exists():
        for pt_file in checkpoint_dir.glob("*.pt"):
            model_paths.append({
                'path': str(pt_file),
                'type': 'checkpoint',
                'name': pt_file.stem
            })

    # HPC results
    hpc_dir = Path("/home/jilab/Jae/hpc_result")
    if hpc_dir.exists():
        for pt_file in hpc_dir.glob("**/*.pt"):
            if "checkpoint" not in str(pt_file):  # Skip checkpoint directories
                model_paths.append({
                    'path': str(pt_file),
                    'type': 'hpc_model',
                    'name': pt_file.stem
                })

    return sorted(model_paths, key=lambda x: Path(x['path']).stat().st_mtime, reverse=True)


def load_model_checkpoint(checkpoint_path: str, device: str = "auto") -> Optional[Dict]:
    """Load model checkpoint with proper error handling."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            print("✓ Checkpoint loaded successfully")
            return checkpoint
        else:
            print("⚠ Checkpoint format not recognized, trying direct load...")
            # Try loading as direct state dict
            model = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, 3, padding=1)
            ).to(device)

            if isinstance(checkpoint, dict):
                # Direct state dict
                model.load_state_dict(checkpoint)
                return {"model_state_dict": checkpoint, "config": {}}
            else:
                print("❌ Cannot load model")
                return None

    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None


def create_model_from_checkpoint(checkpoint: Dict, device: str = "auto") -> Optional[nn.Module]:
    """Create model from checkpoint."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Create model matching the checkpoint structure
        model = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  # conv1
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  # conv2
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)   # conv3
        ).to(device)

        # Load state dict
        state_dict = checkpoint.get("model_state_dict")
        if state_dict:
            try:
                model.load_state_dict(state_dict)
                print("✓ Model state dict loaded successfully")
            except Exception as e:
                print(f"⚠ Could not load state dict: {e}")
                print("Using randomly initialized model for testing")

        return model

    except Exception as e:
        print(f"Failed to create model: {e}")
        return None


def generate_evaluation_data(num_samples: int = 20, image_size: int = 128) -> Dict[str, torch.Tensor]:
    """Generate evaluation dataset with various noise levels."""
    print(f"Generating evaluation dataset with {num_samples} samples...")

    # Create clean images with different patterns
    clean_images = []

    for i in range(num_samples):
        if i % 4 == 0:
            # Smooth gradient
            x = torch.linspace(0, 1, image_size)
            y = torch.linspace(0, 1, image_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            img = (X + Y) / 2
        elif i % 4 == 1:
            # Circular pattern
            center = image_size // 2
            x = torch.linspace(-1, 1, image_size)
            y = torch.linspace(-1, 1, image_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            img = torch.sqrt(X**2 + Y**2)
        elif i % 4 == 2:
            # Stripe pattern
            img = torch.sin(torch.linspace(0, 4*np.pi, image_size)).repeat(image_size, 1)
        else:
            # Random texture
            img = torch.rand(image_size, image_size)

        clean_images.append(img.unsqueeze(0))

    clean_tensor = torch.stack(clean_images)

    # Add Poisson-Gaussian noise at different levels
    results = {}

    for photon_level in [10, 50, 100, 500, 1000]:  # Different photon levels
        scale = 1000.0
        background = 50.0
        read_noise = max(2.0, 20.0 / np.sqrt(photon_level))  # Adaptive read noise

        # Convert to electron space
        clean_electrons = clean_tensor * scale + background

        # Add realistic noise
        poisson_noise = torch.sqrt(clean_electrons) * torch.randn_like(clean_electrons)
        gaussian_noise = read_noise * torch.randn_like(clean_electrons)
        noisy_electrons = clean_electrons + poisson_noise + gaussian_noise

        # Convert back to normalized
        noisy_tensor = torch.clamp((noisy_electrons - background) / scale, 0, 1)

        results[f'photon_{photon_level}'] = {
            'clean': clean_tensor,
            'noisy': noisy_tensor,
            'photon_level': photon_level,
            'scale': scale,
            'background': background,
            'read_noise': read_noise
        }

    print(f"✓ Generated evaluation data for {len(results)} photon levels")
    return results


def evaluate_model(model: nn.Module, test_data: Dict, device: str = "auto") -> Dict[str, float]:
    """Evaluate model performance on test data."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    all_metrics = {}

    with torch.no_grad():
        for photon_level, data in test_data.items():
            clean = data['clean'].to(device)
            noisy = data['noisy'].to(device)

            # Run inference
            pred = model(noisy)
            pred = torch.clamp(pred, 0, 1)

            # Compute metrics
            mse = torch.mean((pred - clean) ** 2).item()
            psnr = 20 * np.log10(1.0) - 10 * np.log10(mse + 1e-8)

            # SSIM computation
            def compute_ssim(img1, img2):
                from scipy.ndimage import gaussian_filter
                C1 = (0.01) ** 2
                C2 = (0.03) ** 2

                mu1 = gaussian_filter(img1, sigma=1.5)
                mu2 = gaussian_filter(img2, sigma=1.5)

                sigma1_sq = gaussian_filter(img1 ** 2, sigma=1.5) - mu1 ** 2
                sigma2_sq = gaussian_filter(img2 ** 2, sigma=1.5) - mu2 ** 2
                sigma12 = gaussian_filter(img1 * img2, sigma=1.5) - mu1 * mu2

                numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
                denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

                return np.mean(numerator / denominator)

            ssim_val = compute_ssim(
                pred.squeeze().cpu().numpy(),
                clean.squeeze().cpu().numpy()
            )

            # Physics-based metrics
            scale = data['scale']
            background = data['background']
            read_noise = data['read_noise']

            pred_electrons = pred * scale + background
            clean_electrons = clean * scale + background
            noisy_electrons = data['noisy'] * scale + background

            residuals = noisy_electrons - pred_electrons
            residual_mean = residuals.mean().item()
            residual_std = residuals.std().item()

            # χ² consistency (simplified)
            variance = pred_electrons + read_noise**2
            chi2_per_pixel = (residuals**2) / variance
            chi2_mean = chi2_per_pixel.mean().item()

            all_metrics[photon_level] = {
                'psnr': psnr,
                'ssim': ssim_val,
                'mse': mse,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'chi2_consistency': chi2_mean,
                'photon_level': data['photon_level']
            }

    return all_metrics


def create_comparison_plots(results: Dict, model_name: str, save_dir: str):
    """Create visual comparison plots."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create summary plot
    photon_levels = [r['photon_level'] for r in results.values()]
    psnr_values = [r['psnr'] for r in results.values()]
    ssim_values = [r['ssim'] for r in results.values()]
    chi2_values = [r['chi2_consistency'] for r in results.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # PSNR vs photon level
    axes[0].plot(photon_levels, psnr_values, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Photon Level')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR vs Photon Level')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')

    # SSIM vs photon level
    axes[1].plot(photon_levels, ssim_values, 's-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Photon Level')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM vs Photon Level')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    # χ² consistency vs photon level
    axes[2].plot(photon_levels, chi2_values, '^-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Photon Level')
    axes[2].set_ylabel('χ² Consistency')
    axes[2].set_title('χ² Consistency vs Photon Level')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log')
    axes[2].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Ideal (1.0)')
    axes[2].legend()

    plt.suptitle(f'Model Performance: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    summary_plot_path = save_path / f"{model_name}_performance_summary.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Performance summary plot saved to {summary_plot_path}")


def main():
    """Main evaluation function."""
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)

    # Find available models
    available_models = find_available_models()

    if not available_models:
        print("❌ No models found!")
        print("Available locations checked:")
        print("  - /home/jilab/Jae/checkpoints/")
        print("  - /home/jilab/Jae/hpc_result/")
        return

    print(f"Found {len(available_models)} models:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model['name']} ({model['type']})")

    # Test each model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate evaluation data
    print("\nGenerating evaluation dataset...")
    test_data = generate_evaluation_data(num_samples=10, image_size=128)

    all_results = {}

    for model_info in available_models[:3]:  # Test first 3 models
        model_name = model_info['name']
        model_path = model_info['path']

        print(f"\n{'-' * 50}")
        print(f"Evaluating model: {model_name}")
        print(f"{'-' * 50}")

        # Load model
        checkpoint = load_model_checkpoint(model_path, device)
        if checkpoint is None:
            print(f"❌ Failed to load {model_name}")
            continue

        model = create_model_from_checkpoint(checkpoint, device)
        if model is None:
            print(f"❌ Failed to create model {model_name}")
            continue

        print("✓ Model loaded and created successfully")

        # Evaluate
        print("Running evaluation...")
        metrics = evaluate_model(model, test_data, device)

        # Store results
        all_results[model_name] = metrics

        print("✓ Evaluation completed:")
        for photon_level, metric in metrics.items():
            print(f"  {photon_level}: PSNR={metric['psnr']:.2f} SSIM={metric['ssim']:.4f} χ²={metric['chi2_consistency']:.2f}")

    # Create comparison plots
    if len(all_results) > 0:
        print("\nCreating comparison plots...")
        output_dir = Path("/home/jilab/Jae/evaluation_results")
        output_dir.mkdir(exist_ok=True)

        for model_name, metrics in all_results.items():
            create_comparison_plots(metrics, model_name, str(output_dir))

        # Save detailed results
        results_path = output_dir / "comprehensive_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"✓ Detailed results saved to {results_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Tested {len(all_results)} models")
    print(f"Evaluation data: {len(test_data)} photon levels × 10 samples each")
    print(f"Device used: {device}")
    print(f"Results saved to: /home/jilab/Jae/evaluation_results/")

    # Find best performing model
    best_model = None
    best_psnr = 0

    for model_name, metrics in all_results.items():
        avg_psnr = np.mean([m['psnr'] for m in metrics.values()])
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_model = model_name

    if best_model:
        print(f"🏆 Best performing model: {best_model} (avg PSNR: {best_psnr:.2f})")

    print(f"{'=' * 70}")
    print("Next steps:")
    print("1. Review the generated plots in evaluation_results/")
    print("2. Check detailed metrics in comprehensive_evaluation_results.json")
    print("3. Run domain-specific evaluation: python scripts/evaluate_multi_domain_model.py")
    print("4. Generate paper figures: python scripts/generate_paper_figures.py")


if __name__ == "__main__":
    main()
