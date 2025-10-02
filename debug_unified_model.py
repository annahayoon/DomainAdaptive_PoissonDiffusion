#!/usr/bin/env python3
"""
Debug Script for Unified Model Issues

This script provides diagnostic tools to help identify the root cause
of numerical instability in the unified model evaluation.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.edm_wrapper import load_pretrained_edm, DomainEncoder
from models.sampler import EDMPosteriorSampler
from core.guidance_config import GuidanceConfig
from core.poisson_guidance import PoissonGuidance

def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint contents for debugging."""
    print("=" * 60)
    print("CHECKPOINT INSPECTION")
    print("=" * 60)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"Training step: {checkpoint.get('step', 'unknown')}")
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\nTraining config keys: {list(config.keys())}")
        print(f"Model channels: {config.get('model_channels', 'unknown')}")
        print(f"Domains: {config.get('domains', 'unknown')}")
        print(f"Data root: {config.get('data_root', 'unknown')}")
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Find input layer to determine channels
        for key, tensor in state_dict.items():
            if "128x128_conv.weight" in key and "enc" in key:
                print(f"\nInput layer: {key}")
                print(f"Input shape: {tensor.shape}")
                print(f"Input channels: {tensor.shape[1]}")
                break
        
        # Find output layer
        for key, tensor in state_dict.items():
            if "128x128_aux_conv.weight" in key and "dec" in key:
                print(f"Output layer: {key}")
                print(f"Output shape: {tensor.shape}")
                print(f"Output channels: {tensor.shape[0]}")
                break
        
        # Model size
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Model size: {total_params * 4 / 1024**3:.2f} GB (fp32)")

def test_model_loading(checkpoint_path):
    """Test model loading without inference."""
    print("\n" + "=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)
    
    try:
        model = load_pretrained_edm(checkpoint_path, device="cpu")
        print("✅ Model loaded successfully")
        
        # Test model info
        if hasattr(model, 'config'):
            config = model.config
            print(f"Image channels: {config.img_channels}")
            print(f"Model channels: {config.model_channels}")
            print(f"Label dim: {config.label_dim}")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, config.img_channels, 128, 128)
        dummy_noise = torch.randn(1)
        dummy_condition = torch.randn(1, config.label_dim)
        
        print(f"Testing forward pass...")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Condition shape: {dummy_condition.shape}")
        
        with torch.no_grad():
            try:
                # Test model forward (without guidance)
                output = model(dummy_input, dummy_noise, dummy_condition)
                print(f"✅ Forward pass successful")
                print(f"Output shape: {output.shape}")
                print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
                
                return model, True
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                return model, False
                
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, False

def test_domain_conditioning():
    """Test domain conditioning vector generation."""
    print("\n" + "=" * 60)
    print("DOMAIN CONDITIONING TEST")
    print("=" * 60)
    
    encoder = DomainEncoder()
    
    domains = ["photography", "microscopy", "astronomy"]
    scales = [1000.0, 500.0, 200.0]
    
    for domain, scale in zip(domains, scales):
        try:
            # Test both conditioning types
            for cond_type in ["dapgd", "l2"]:
                conditioning = encoder.encode_domain(
                    domain=domain,
                    scale=scale,
                    read_noise=10.0,
                    background=50.0,
                    device="cpu",
                    conditioning_type=cond_type
                )
                
                print(f"{domain} ({cond_type}): {conditioning.shape} -> {conditioning.flatten()}")
                
        except Exception as e:
            print(f"❌ {domain} conditioning failed: {e}")

def test_guidance_computation(model):
    """Test guidance computation without full sampling."""
    print("\n" + "=" * 60)
    print("GUIDANCE COMPUTATION TEST")
    print("=" * 60)
    
    if model is None:
        print("❌ No model available for guidance test")
        return
    
    # Create test data
    clean = torch.rand(1, 4, 128, 128) * 0.8 + 0.1  # [0.1, 0.9] range
    noisy = clean + torch.randn_like(clean) * 0.1
    
    print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    
    # Test different guidance strengths
    guidance_strengths = [0.01, 0.1, 1.0]
    
    for strength in guidance_strengths:
        print(f"\nTesting guidance strength: {strength}")
        
        try:
            # Create guidance
            guidance_config = GuidanceConfig(
                kappa=strength,
                gamma_schedule="sigma2",
                gradient_clip=10.0,
            )
            
            guidance = PoissonGuidance(
                scale=1000.0,
                background=100.0,
                read_noise=10.0,
                config=guidance_config,
            )
            
            # Test single guidance computation
            sigma = torch.tensor([1.0])
            
            # Test model prediction first
            conditioning = torch.randn(1, 6)  # Dummy conditioning
            
            with torch.no_grad():
                model_pred = model(noisy, sigma, conditioning)
                print(f"  Model pred range: [{model_pred.min():.3f}, {model_pred.max():.3f}]")
                
                # Test guidance computation
                guidance_correction = guidance.compute(
                    x_hat=model_pred,
                    y_observed=noisy,
                    sigma=sigma
                )
                
                print(f"  Guidance range: [{guidance_correction.min():.3f}, {guidance_correction.max():.3f}]")
                print(f"  Guidance norm: {torch.norm(guidance_correction):.3f}")
                
        except Exception as e:
            print(f"  ❌ Guidance test failed: {e}")

def analyze_test_data(data_root):
    """Analyze test data format and statistics."""
    print("\n" + "=" * 60)
    print("TEST DATA ANALYSIS")
    print("=" * 60)
    
    data_path = Path(data_root).expanduser()
    
    for domain in ["photography", "microscopy", "astronomy"]:
        domain_path = data_path / domain / "test"
        
        if not domain_path.exists():
            print(f"❌ {domain}: No test directory found")
            continue
        
        test_files = list(domain_path.glob("*.pt"))
        print(f"\n{domain.upper()}: {len(test_files)} test files")
        
        if test_files:
            # Analyze first file
            try:
                data = torch.load(test_files[0], map_location="cpu", weights_only=False)
                print(f"  Data keys: {list(data.keys()) if isinstance(data, dict) else 'tensor'}")
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: {value.shape}, range [{value.min():.3f}, {value.max():.3f}]")
                
            except Exception as e:
                print(f"  ❌ Failed to load {test_files[0].name}: {e}")

def main():
    """Run all diagnostic tests."""
    checkpoint_path = Path("~/checkpoint_step_0090000.pth").expanduser()
    data_root = "~/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed/posterior"
    
    print("UNIFIED MODEL DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data root: {data_root}")
    
    # 1. Inspect checkpoint
    inspect_checkpoint(checkpoint_path)
    
    # 2. Test model loading
    model, load_success = test_model_loading(checkpoint_path)
    
    # 3. Test domain conditioning
    test_domain_conditioning()
    
    # 4. Test guidance computation
    if load_success:
        test_guidance_computation(model)
    
    # 5. Analyze test data
    analyze_test_data(data_root)
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("Share this report with your advisor to identify the root cause.")

if __name__ == "__main__":
    main()
