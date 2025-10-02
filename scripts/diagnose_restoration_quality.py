#!/usr/bin/env python3
"""
Diagnostic Script: Restoration Quality Investigation

Investigates why restored images appear noisier than input observations.
Possible causes:
1. Model not fully trained (90K steps insufficient)
2. Data preprocessing issues
3. Coordinate transformation problems
4. Model expecting different input format
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.logging_config import get_logger
from core.physics_aware_sampler import create_physics_aware_sampler
from models.edm_wrapper import DomainEncoder, load_pretrained_edm

logger = get_logger(__name__)


def analyze_data_ranges(clean, noisy, domain):
    """Analyze the data ranges to understand the problem."""
    print(f"\n=== DATA RANGE ANALYSIS: {domain.upper()} ===")
    
    # Clean data analysis
    print(f"Clean data:")
    print(f"  Shape: {clean.shape}")
    print(f"  Range: [{clean.min():.6f}, {clean.max():.6f}]")
    print(f"  Mean: {clean.mean():.6f}")
    print(f"  Std: {clean.std():.6f}")
    
    # Noisy data analysis  
    print(f"Noisy data:")
    print(f"  Shape: {noisy.shape}")
    print(f"  Range: [{noisy.min():.6f}, {noisy.max():.6f}]")
    print(f"  Mean: {noisy.mean():.6f}")
    print(f"  Std: {noisy.std():.6f}")
    
    # Check if data looks reasonable
    if clean.max() > 1.0:
        print(f"  ⚠️  WARNING: Clean data exceeds [0,1] range!")
    if noisy.min() < 0:
        print(f"  ⚠️  WARNING: Noisy data has negative values!")
    if noisy.max() > 10000:
        print(f"  ⚠️  WARNING: Noisy data has very large values (physical space?)!")


def test_model_without_guidance(model, clean, noisy, domain, device):
    """Test model with no guidance to see baseline performance."""
    print(f"\n=== TESTING MODEL WITHOUT GUIDANCE ===")
    
    config = {
        "photography": {"scale": 1000.0, "background": 100.0, "read_noise": 10.0},
        "microscopy": {"scale": 500.0, "background": 50.0, "read_noise": 5.0},
    }[domain]
    
    # Create sampler with NO guidance
    sampler = create_physics_aware_sampler(
        model=model,
        scale=config["scale"],
        background=config["background"],
        read_noise=config["read_noise"],
        guidance_weight=0.0,  # NO GUIDANCE - pure prior
        device=device,
    )
    
    # Create conditioning
    domain_encoder = DomainEncoder()
    conditioning = domain_encoder.encode_domain(
        domain=domain,
        scale=config["scale"],
        read_noise=config["read_noise"],
        background=config["background"],
        device=device,
        conditioning_type="dapgd",
    )
    
    # Sample with no guidance (should just be denoising)
    with torch.no_grad():
        result_dict = sampler.sample(
            y_observed=noisy,  # This might be the issue!
            metadata=None,
            condition=conditioning,
            steps=18,
            guidance_weight=0.0,
            return_intermediates=False,
        )
    
    result = result_dict['sample']
    
    # Analyze result
    print(f"Model output (no guidance):")
    print(f"  Shape: {result.shape}")
    print(f"  Range: [{result.min():.6f}, {result.max():.6f}]")
    print(f"  Mean: {result.mean():.6f}")
    print(f"  Std: {result.std():.6f}")
    
    # Compute metrics
    def compute_psnr(pred, target):
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    # Compare with clean (first channel only)
    clean_single = clean[:, :1, :, :] if clean.shape[1] == 4 else clean
    result_single = result[:, :1, :, :] if result.shape[1] == 4 else result
    noisy_single = noisy[:, :1, :, :] if noisy.shape[1] == 4 else noisy
    
    psnr_result_vs_clean = compute_psnr(result_single, clean_single)
    psnr_noisy_vs_clean = compute_psnr(noisy_single, clean_single)
    
    print(f"\nPSNR Analysis:")
    print(f"  Noisy vs Clean: {psnr_noisy_vs_clean:.2f} dB")
    print(f"  Result vs Clean: {psnr_result_vs_clean:.2f} dB")
    print(f"  Improvement: {psnr_result_vs_clean - psnr_noisy_vs_clean:+.2f} dB")
    
    if psnr_result_vs_clean < psnr_noisy_vs_clean:
        print(f"  ❌ MODEL IS MAKING THINGS WORSE!")
        print(f"     This suggests a fundamental problem.")
    else:
        print(f"  ✅ Model is improving the image")
    
    return result


def test_different_input_formats(model, clean, noisy, domain, device):
    """Test different ways of providing input to the model."""
    print(f"\n=== TESTING DIFFERENT INPUT FORMATS ===")
    
    config = {
        "photography": {"scale": 1000.0, "background": 100.0, "read_noise": 10.0},
        "microscopy": {"scale": 500.0, "background": 50.0, "read_noise": 5.0},
    }[domain]
    
    # Create conditioning
    domain_encoder = DomainEncoder()
    conditioning = domain_encoder.encode_domain(
        domain=domain,
        scale=config["scale"],
        read_noise=config["read_noise"],
        background=config["background"],
        device=device,
        conditioning_type="dapgd",
    )
    
    # Test 1: Use clean image as "noisy" input (should be perfect reconstruction)
    print("\nTest 1: Using clean image as input (should be perfect)")
    sampler = create_physics_aware_sampler(
        model=model, scale=config["scale"], background=config["background"],
        read_noise=config["read_noise"], guidance_weight=0.0, device=device,
    )
    
    with torch.no_grad():
        result_dict = sampler.sample(
            y_observed=clean * config["scale"],  # Convert to physical space
            metadata=None, condition=conditioning, steps=18,
            guidance_weight=0.0, return_intermediates=False,
        )
    
    result_clean_input = result_dict['sample']
    clean_single = clean[:, :1, :, :] if clean.shape[1] == 4 else clean
    result_single = result_clean_input[:, :1, :, :] if result_clean_input.shape[1] == 4 else result_clean_input
    
    psnr_clean_input = 20 * torch.log10(1.0 / torch.sqrt(torch.mean((result_single - clean_single) ** 2))).item()
    print(f"  PSNR with clean input: {psnr_clean_input:.2f} dB")
    
    if psnr_clean_input < 30:
        print(f"  ❌ Even with clean input, model performs poorly!")
        print(f"     This suggests the model is not properly trained or has wrong architecture.")
    else:
        print(f"  ✅ Model can reconstruct clean images well")
    
    # Test 2: Use normalized noisy input
    print("\nTest 2: Using normalized noisy input")
    noisy_normalized = torch.clamp(noisy / config["scale"], 0, 1)
    
    with torch.no_grad():
        result_dict = sampler.sample(
            y_observed=noisy_normalized * config["scale"],
            metadata=None, condition=conditioning, steps=18,
            guidance_weight=0.0, return_intermediates=False,
        )
    
    result_norm_input = result_dict['sample']
    result_single = result_norm_input[:, :1, :, :] if result_norm_input.shape[1] == 4 else result_norm_input
    
    psnr_norm_input = 20 * torch.log10(1.0 / torch.sqrt(torch.mean((result_single - clean_single) ** 2))).item()
    print(f"  PSNR with normalized noisy input: {psnr_norm_input:.2f} dB")
    
    return result_clean_input, result_norm_input


def create_detailed_comparison_plot(clean, noisy, result_no_guidance, result_with_guidance, domain, output_path):
    """Create detailed comparison plot with histograms and statistics."""
    
    # Convert to numpy (first channel only)
    def to_numpy(tensor):
        if tensor.shape[1] == 4:
            return tensor[0, 0].detach().cpu().numpy()
        else:
            return tensor[0, 0].detach().cpu().numpy()
    
    # Helper function to get colormap based on domain
    def get_colormap(domain):
        if domain in ['astronomy', 'microscopy']:
            return 'gray'
        else:
            return 'viridis'
    
    clean_np = to_numpy(clean)
    noisy_np = to_numpy(noisy)
    result_no_guidance_np = to_numpy(result_no_guidance)
    result_with_guidance_np = to_numpy(result_with_guidance)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Get appropriate colormap for domain
    cmap = get_colormap(domain)
    
    # Images with dynamic range for each panel
    ax1 = plt.subplot(3, 4, 1)
    plt.imshow(clean_np, cmap=cmap, vmin=clean_np.min(), vmax=clean_np.max())
    plt.title(f'Clean Ground Truth\nRange: [{clean_np.min():.3f}, {clean_np.max():.3f}]')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    ax2 = plt.subplot(3, 4, 2)
    plt.imshow(noisy_np, cmap=cmap, vmin=noisy_np.min(), vmax=noisy_np.max())
    plt.title(f'Noisy Observation\nRange: [{noisy_np.min():.1f}, {noisy_np.max():.1f}]')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    ax3 = plt.subplot(3, 4, 3)
    plt.imshow(result_no_guidance_np, cmap=cmap, vmin=result_no_guidance_np.min(), vmax=result_no_guidance_np.max())
    plt.title(f'Prior Only (κ=0)\nRange: [{result_no_guidance_np.min():.3f}, {result_no_guidance_np.max():.3f}]')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    ax4 = plt.subplot(3, 4, 4)
    plt.imshow(result_with_guidance_np, cmap=cmap, vmin=result_with_guidance_np.min(), vmax=result_with_guidance_np.max())
    plt.title(f'Physics-Aware (κ=1.0)\nRange: [{result_with_guidance_np.min():.3f}, {result_with_guidance_np.max():.3f}]')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Histograms
    ax5 = plt.subplot(3, 4, 5)
    plt.hist(clean_np.flatten(), bins=50, alpha=0.7, label='Clean', density=True)
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.title('Clean Distribution')
    plt.legend()
    
    ax6 = plt.subplot(3, 4, 6)
    plt.hist(noisy_np.flatten(), bins=50, alpha=0.7, label='Noisy', density=True, color='orange')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.title('Noisy Distribution')
    plt.legend()
    
    ax7 = plt.subplot(3, 4, 7)
    plt.hist(result_no_guidance_np.flatten(), bins=50, alpha=0.7, label='Prior Only', density=True, color='green')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.title('Prior Only Distribution')
    plt.legend()
    
    ax8 = plt.subplot(3, 4, 8)
    plt.hist(result_with_guidance_np.flatten(), bins=50, alpha=0.7, label='Physics-Aware', density=True, color='red')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.title('Physics-Aware Distribution')
    plt.legend()
    
    # Difference maps
    ax9 = plt.subplot(3, 4, 9)
    diff_noisy = noisy_np - clean_np
    plt.imshow(diff_noisy, cmap='RdBu', vmin=-np.abs(diff_noisy).max(), vmax=np.abs(diff_noisy).max())
    plt.title(f'Noisy - Clean\nStd: {diff_noisy.std():.3f}')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    ax10 = plt.subplot(3, 4, 10)
    diff_prior = result_no_guidance_np - clean_np
    plt.imshow(diff_prior, cmap='RdBu', vmin=-np.abs(diff_prior).max(), vmax=np.abs(diff_prior).max())
    plt.title(f'Prior - Clean\nStd: {diff_prior.std():.3f}')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    ax11 = plt.subplot(3, 4, 11)
    diff_physics = result_with_guidance_np - clean_np
    plt.imshow(diff_physics, cmap='RdBu', vmin=-np.abs(diff_physics).max(), vmax=np.abs(diff_physics).max())
    plt.title(f'Physics - Clean\nStd: {diff_physics.std():.3f}')
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # Statistics
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Compute PSNRs
    def compute_psnr(pred, target):
        mse = np.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    psnr_noisy = compute_psnr(noisy_np, clean_np)
    psnr_prior = compute_psnr(result_no_guidance_np, clean_np)
    psnr_physics = compute_psnr(result_with_guidance_np, clean_np)
    
    stats_text = f"""
STATISTICS:

PSNR Analysis:
• Noisy vs Clean: {psnr_noisy:.2f} dB
• Prior vs Clean: {psnr_prior:.2f} dB  
• Physics vs Clean: {psnr_physics:.2f} dB

Improvements:
• Prior vs Noisy: {psnr_prior - psnr_noisy:+.2f} dB
• Physics vs Prior: {psnr_physics - psnr_prior:+.2f} dB

Noise Levels (Std):
• Noisy residual: {diff_noisy.std():.4f}
• Prior residual: {diff_prior.std():.4f}
• Physics residual: {diff_physics.std():.4f}

Data Ranges:
• Clean: [{clean_np.min():.3f}, {clean_np.max():.3f}]
• Noisy: [{noisy_np.min():.1f}, {noisy_np.max():.1f}]
• Prior: [{result_no_guidance_np.min():.3f}, {result_no_guidance_np.max():.3f}]
• Physics: [{result_with_guidance_np.min():.3f}, {result_with_guidance_np.max():.3f}]
"""
    
    ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes, fontsize=10, 
              verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'{domain.capitalize()} Domain: Detailed Restoration Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Detailed analysis saved to: {output_path}")


def main():
    """Main diagnostic function."""
    print("=" * 80)
    print("RESTORATION QUALITY DIAGNOSTIC")
    print("=" * 80)
    print("Investigating why restored images appear noisier than input...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("~/checkpoint_step_0090000.pth").expanduser()
    data_root = Path("~/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed/posterior").expanduser()
    
    # Load model
    print(f"\n1. Loading model from {model_path}")
    model = load_pretrained_edm(str(model_path), device=device)
    model.eval()
    print(f"   Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    
    # Test both domains
    for domain in ["photography", "microscopy"]:
        print(f"\n" + "=" * 60)
        print(f"ANALYZING {domain.upper()} DOMAIN")
        print("=" * 60)
        
        # Load sample data
        test_dir = data_root / domain / "test"
        test_files = list(test_dir.glob("*.pt"))
        
        if not test_files:
            print(f"No test files found for {domain}")
            continue
        
        # Load first file
        data = torch.load(test_files[0], map_location="cpu", weights_only=False)
        clean = data.get("clean_norm", data.get("clean"))
        noisy = data.get("noisy_norm", data.get("noisy"))
        
        # Process data (same as in visual demo)
        if clean.dim() > 2:
            if clean.shape[0] == 4:
                clean = clean
            else:
                clean = clean.mean(dim=0 if clean.dim() == 3 else 1)
        
        if noisy.dim() > 2:
            if noisy.shape[0] == 4:
                noisy = noisy
            else:
                noisy = noisy.mean(dim=0 if noisy.dim() == 3 else 1)
        
        # Resize and format
        import torch.nn.functional as F
        target_size = (128, 128)
        
        if clean.shape[-2:] != target_size:
            if clean.dim() == 2:
                clean = clean.unsqueeze(0).unsqueeze(0)
            elif clean.dim() == 3:
                clean = clean.unsqueeze(0)
            clean = F.interpolate(clean, size=target_size, mode="bilinear", align_corners=False)
            clean = clean.squeeze(0) if clean.shape[0] == 1 else clean
        
        if noisy.shape[-2:] != target_size:
            if noisy.dim() == 2:
                noisy = noisy.unsqueeze(0).unsqueeze(0)
            elif noisy.dim() == 3:
                noisy = noisy.unsqueeze(0)
            noisy = F.interpolate(noisy, size=target_size, mode="bilinear", align_corners=False)
            noisy = noisy.squeeze(0) if noisy.shape[0] == 1 else noisy
        
        # Ensure 4 channels
        if clean.dim() == 2:
            clean = clean.unsqueeze(0)
        if clean.shape[0] == 1:
            clean = clean.repeat(4, 1, 1)
        
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(0)
        if noisy.shape[0] == 1:
            noisy = noisy.repeat(4, 1, 1)
        
        # Add batch dimension
        clean = clean.unsqueeze(0).to(device)
        noisy = noisy.unsqueeze(0).to(device)
        
        # Ensure clean is normalized
        clean = torch.clamp(clean, 0, 1)
        
        # Analyze data ranges
        analyze_data_ranges(clean, noisy, domain)
        
        # Test model without guidance
        result_no_guidance = test_model_without_guidance(model, clean, noisy, domain, device)
        
        # Test different input formats
        result_clean_input, result_norm_input = test_different_input_formats(model, clean, noisy, domain, device)
        
        # Test with guidance
        print(f"\n=== TESTING WITH GUIDANCE ===")
        config = {
            "photography": {"scale": 1000.0, "background": 100.0, "read_noise": 10.0},
            "microscopy": {"scale": 500.0, "background": 50.0, "read_noise": 5.0},
        }[domain]
        
        sampler = create_physics_aware_sampler(
            model=model, scale=config["scale"], background=config["background"],
            read_noise=config["read_noise"], guidance_weight=1.0, device=device,
        )
        
        domain_encoder = DomainEncoder()
        conditioning = domain_encoder.encode_domain(
            domain=domain, scale=config["scale"], read_noise=config["read_noise"],
            background=config["background"], device=device, conditioning_type="dapgd",
        )
        
        with torch.no_grad():
            result_dict = sampler.sample(
                y_observed=noisy, metadata=None, condition=conditioning,
                steps=18, guidance_weight=1.0, return_intermediates=False,
            )
        
        result_with_guidance = result_dict['sample']
        
        # Create detailed comparison
        output_path = f"diagnostic_results_{domain}_detailed.png"
        create_detailed_comparison_plot(
            clean, noisy, result_no_guidance, result_with_guidance, domain, output_path
        )
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("Check the generated diagnostic plots for detailed analysis.")


if __name__ == "__main__":
    main()
