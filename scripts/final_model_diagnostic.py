#!/usr/bin/env python3
"""
Final Model Diagnostic

Tests the most basic functionality to understand if the model has any
denoising capability at all, or if there are fundamental architectural issues.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.logging_config import get_logger
from core.physics_aware_sampler import create_physics_aware_sampler
from models.edm_wrapper import DomainEncoder, load_pretrained_edm

logger = get_logger(__name__)


def test_identity_reconstruction(model, device):
    """Test 1: Can model reconstruct an image it's denoising?"""
    print("=== TEST 1: IDENTITY RECONSTRUCTION ===")
    print("Goal: Test if model can at least preserve image structure")

    # Create a simple test pattern
    test_image = torch.zeros(1, 4, 128, 128, device=device)

    # Add a simple pattern (rectangle)
    test_image[:, :, 32:96, 32:96] = 0.5  # Gray rectangle in center

    # Add some noise
    noise_level = 0.1
    noisy_image = test_image + torch.randn_like(test_image) * noise_level
    noisy_image = torch.clamp(noisy_image, 0, 1)

    print(f"Input pattern range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    print(f"Noisy pattern range: [{noisy_image.min():.3f}, {noisy_image.max():.3f}]")

    # Test model
    config = {"scale": 1000.0, "background": 100.0, "read_noise": 10.0}
    sampler = create_physics_aware_sampler(
        model=model, scale=config["scale"], background=config["background"],
        read_noise=config["read_noise"], guidance_weight=0.0, device=device,
    )

    domain_encoder = DomainEncoder()
    conditioning = domain_encoder.encode_domain(
        domain="photography", scale=config["scale"], read_noise=config["read_noise"],
        background=config["background"], device=device, conditioning_type="dapgd",
    )

    with torch.no_grad():
        result_dict = sampler.sample(
            y_observed=noisy_image * config["scale"],
            metadata=None, condition=conditioning, steps=18,
            guidance_weight=0.0, return_intermediates=False,
        )

    result = result_dict['sample']

    # Compute PSNR
    mse = torch.mean((result - test_image) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"PSNR: {psnr:.2f} dB")

    if psnr > 20:
        print("✅ Model can reconstruct simple patterns well")
    elif psnr > 10:
        print("⚠️  Model can somewhat reconstruct patterns")
    else:
        print("❌ Model cannot reconstruct even simple patterns")

    return result, psnr


def test_noise_reduction(model, device):
    """Test 2: Can model reduce noise at all?"""
    print("\n=== TEST 2: NOISE REDUCTION ===")
    print("Goal: Test if model has any denoising capability")

    # Create clean image
    clean = torch.rand(1, 4, 128, 128, device=device) * 0.8 + 0.1  # [0.1, 0.9]

    # Add significant noise
    noise_std = 0.3
    noisy = clean + torch.randn_like(clean) * noise_std
    noisy = torch.clamp(noisy, 0, 1)

    print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")

    # Test model
    config = {"scale": 1000.0, "background": 100.0, "read_noise": 10.0}
    sampler = create_physics_aware_sampler(
        model=model, scale=config["scale"], background=config["background"],
        read_noise=config["read_noise"], guidance_weight=0.0, device=device,
    )

    domain_encoder = DomainEncoder()
    conditioning = domain_encoder.encode_domain(
        domain="photography", scale=config["scale"], read_noise=config["read_noise"],
        background=config["background"], device=device, conditioning_type="dapgd",
    )

    with torch.no_grad():
        result_dict = sampler.sample(
            y_observed=noisy * config["scale"],
            metadata=None, condition=conditioning, steps=18,
            guidance_weight=0.0, return_intermediates=False,
        )

    result = result_dict['sample']

    # Compute noise levels
    def compute_noise_level(img):
        return torch.std(img).item()

    clean_noise = compute_noise_level(clean)
    noisy_noise = compute_noise_level(noisy)
    result_noise = compute_noise_level(result)

    print(f"Clean noise level: {clean_noise:.4f}")
    print(f"Noisy noise level: {noisy_noise:.4f}")
    print(f"Result noise level: {result_noise:.4f}")

    # Compute PSNR
    mse = torch.mean((result - clean) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    print(f"PSNR: {psnr:.2f} dB")

    if result_noise < noisy_noise:
        reduction = (noisy_noise - result_noise) / noisy_noise * 100
        print(f"✅ Model reduced noise by {reduction:.1f}%")
    else:
        print("❌ Model did not reduce noise")

    return result, psnr, result_noise, noisy_noise


def test_model_parameters(model):
    """Test 3: Analyze model parameters and architecture."""
    print("\n=== TEST 3: MODEL ARCHITECTURE ANALYSIS ===")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Check for frozen layers
    frozen_layers = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_layers.append(name)

    if frozen_layers:
        print(f"Frozen layers: {len(frozen_layers)}")
        print(f"Sample frozen layers: {frozen_layers[:3]}")
    else:
        print("All layers are trainable")

    # Check parameter distributions
    all_params = []
    for param in model.parameters():
        if param.requires_grad:
            all_params.extend(param.flatten().detach().cpu().numpy())

    param_std = np.std(all_params)
    param_mean = np.mean(all_params)

    print(f"Parameter distribution: mean={param_mean:.6f}, std={param_std:.6f}")

    if param_std < 1e-6:
        print("⚠️  WARNING: Parameters have very small variance - possible initialization issue")
    else:
        print("✅ Parameter variance looks normal")


def main():
    """Main diagnostic function."""
    print("=" * 80)
    print("FINAL MODEL DIAGNOSTIC")
    print("=" * 80)
    print("Testing fundamental model capabilities:")
    print("1. Can it reconstruct simple patterns?")
    print("2. Can it reduce noise at all?")
    print("3. Are model parameters reasonable?")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("~/checkpoint_step_0090000.pth").expanduser()

    print(f"Device: {device}")
    print(f"Model: {model_path}")

    # Load model
    print("\n1. Loading unified model...")
    model = load_pretrained_edm(str(model_path), device=device)
    model.eval()
    print(f"   ✅ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Test 1: Identity reconstruction
    result1, psnr1 = test_identity_reconstruction(model, device)

    # Test 2: Noise reduction
    result2, psnr2, result_noise, noisy_noise = test_noise_reduction(model, device)

    # Test 3: Model parameters
    test_model_parameters(model)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC RESULTS")
    print("=" * 80)

    # Analyze results
    print(f"Test 1 - Identity Reconstruction: {psnr1:.2f} dB")
    print(f"Test 2 - Noise Reduction: {psnr2:.2f} dB")

    if psnr1 > 25 and psnr2 > 15:
        print("🎯 CONCLUSION: Model is fundamentally sound")
        print("   The issue is likely data preprocessing or training configuration")
    elif psnr1 > 15 or psnr2 > 10:
        print("⚠️  CONCLUSION: Model has some capability but needs more training")
        print("   Training progress (20%) is the primary issue")
    else:
        print("❌ CONCLUSION: Model has fundamental architectural issues")
        print("   Even basic functionality is not working")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
