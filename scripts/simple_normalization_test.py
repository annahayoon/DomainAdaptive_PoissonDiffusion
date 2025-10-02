#!/usr/bin/env python3
"""
Simple Normalization Test

A simpler test to understand if data normalization is the issue.
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


def test_microscopy_normalization():
    """Test microscopy data with proper normalization."""
    print("=== TESTING MICROSCOPY WITH NORMALIZATION ===")

    # Load data
    test_dir = Path("~/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed/posterior").expanduser() / "microscopy" / "test"
    test_files = list(test_dir.glob("*.pt"))

    if not test_files:
        print("No test files found")
        return

    data = torch.load(test_files[0], map_location="cpu", weights_only=False)
    clean = data.get("clean_norm", data.get("clean"))
    noisy = data.get("noisy_norm", data.get("noisy"))

    print(f"Original clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"Original noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")

    # Normalize clean data
    clean_normalized = torch.clamp(clean, 0, 1)
    print(f"Normalized clean range: [{clean_normalized.min():.3f}, {clean_normalized.max():.3f}]")

    # Process for model
    clean_tensor = process_for_model(clean_normalized, "microscopy")
    noisy_tensor = process_for_model(noisy, "microscopy")

    print(f"Processed clean shape: {clean_tensor.shape}")
    print(f"Processed noisy shape: {noisy_tensor.shape}")

    return clean_tensor, noisy_tensor


def process_for_model(tensor: torch.Tensor, domain: str) -> torch.Tensor:
    """Process tensor to match model input format."""
    # Handle single channel to 4 channels
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif tensor.dim() == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(4, 1, 1)  # [4, H, W]
            tensor = tensor.unsqueeze(0)  # [1, 4, H, W]
        else:
            tensor = tensor.unsqueeze(0)  # [1, C, H, W]

    # Resize if needed
    if tensor.shape[-2:] != (128, 128):
        tensor = torch.nn.functional.interpolate(
            tensor, size=(128, 128), mode="bilinear", align_corners=False
        )

    return tensor


def test_model_on_normalized_data(model, clean, noisy, device):
    """Test model on normalized data."""
    print("\n=== TESTING MODEL ON NORMALIZED DATA ===")

    # Move to device
    clean = clean.to(device)
    noisy = noisy.to(device)

    # Create sampler
    config = {"scale": 500.0, "background": 50.0, "read_noise": 5.0}
    sampler = create_physics_aware_sampler(
        model=model, scale=config["scale"], background=config["background"],
        read_noise=config["read_noise"], guidance_weight=0.0, device=device,
    )

    # Create conditioning
    domain_encoder = DomainEncoder()
    conditioning = domain_encoder.encode_domain(
        domain="microscopy", scale=config["scale"], read_noise=config["read_noise"],
        background=config["background"], device=device, conditioning_type="dapgd",
    )

    # Test with normalized noisy input
    noisy_physical = noisy * config["scale"]

    print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"Noisy physical range: [{noisy_physical.min():.1f}, {noisy_physical.max():.1f}]")

    with torch.no_grad():
        result_dict = sampler.sample(
            y_observed=noisy_physical,
            metadata=None, condition=conditioning, steps=18,
            guidance_weight=0.0, return_intermediates=False,
        )

    result = result_dict['sample']

    # Compute PSNR (first channel only)
    clean_single = clean[:, :1, :, :] if clean.shape[1] == 4 else clean
    result_single = result[:, :1, :, :] if result.shape[1] == 4 else result

    mse = torch.mean((result_single - clean_single) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    print(f"Result range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"PSNR: {psnr:.2f} dB")

    if psnr > 20:
        print("✅ Good restoration quality!")
    elif psnr > 10:
        print("⚠️  Moderate restoration quality")
    else:
        print("❌ Poor restoration quality")

    return result, psnr


def main():
    """Main test function."""
    print("=" * 60)
    print("SIMPLE NORMALIZATION TEST")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("~/checkpoint_step_0090000.pth").expanduser()

    print(f"Device: {device}")
    print(f"Model: {model_path}")

    # Load model
    print("\n1. Loading model...")
    model = load_pretrained_edm(str(model_path), device=device)
    model.eval()
    print(f"   ✅ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Test microscopy
    print("\n2. Testing microscopy...")
    clean, noisy = test_microscopy_normalization()

    # Test model
    result, psnr = test_model_on_normalized_data(model, clean, noisy, device)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    if psnr > 20:
        print("🎯 CONCLUSION: Model works well with proper normalization!")
        print("   The issue is likely data normalization mismatch.")
    elif psnr > 10:
        print("⚠️  CONCLUSION: Normalization helps but model still needs more training.")
    else:
        print("❌ CONCLUSION: Model has fundamental issues even with normalization.")


if __name__ == "__main__":
    main()
