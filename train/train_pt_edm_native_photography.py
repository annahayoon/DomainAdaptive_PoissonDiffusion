#!/usr/bin/env python3
"""
Train unconditional diffusion model on preprocessed image tiles.

This script trains a domain-agnostic EDM diffusion model using the inverse problem approach:

TRAINING PHASE (this script):
- Learn P(x_clean) unconditionally on long-exposure/clean images only
- No labels, no conditioning on observations
- Pure unconditional diffusion on clean distribution
- Dataset is automatically validated to contain only LONG-exposure tiles

INFERENCE PHASE (separate script - sample/sample_noisy_pt_lle_PGguidance.py):
- Load trained unconditional model
- Apply heteroscedastic Poisson-Gaussian likelihood gradient as guidance
- Supports BOTH simplified and exact variance formulas:
  * Simplified: Var ≈ α·s·x + σ_r² (computationally efficient, default)
  * Exact: Var = α·x/(white-black) + σ_r² (theoretically precise, for ablations)
- Apply sensor-specific calibration at inference time

DATA FORMAT:
- Preprocessing converts RAW → demosaic to RGB → normalize to [-1, 1]
- RGB format (not packed RAW) for generalizability:
  * Works on any RGB image, not just specific camera RAW formats
  * Enables cross-domain and cross-sensor generalization
  * Signal-dependent variance structure remains dominant despite interpolation
- Training uses ONLY long-exposure (clean) tiles
- Short-exposure (noisy) tiles used only for validation/testing

This approach matches DPS (Diffusion Posterior Sampling) and DDRM paradigms,
scientifically validated in top venues (CVPR, NeurIPS, ICLR).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root and EDM to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

# Import training utilities
from config_utils import (
    apply_config_to_args,
    create_loss_config,
    create_network_config,
    create_optimizer_config,
    find_latest_checkpoint,
    load_checkpoint,
    load_yaml_config,
    setup_logging,
    setup_output_directory,
)
from training_utils import training_loop

# Import EDM native components
import external.edm.dnnlib

# Import dataset
from data.dataset import SimplePTDataset
from external.edm.torch_utils import distributed as dist

# Import validation metrics
try:
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logging.warning("torchmetrics not available - validation metrics disabled")


def validate_dataset_compatibility(
    dataset: SimplePTDataset, device: torch.device
) -> bool:
    """
    Validate that dataset is compatible with training expectations.

    Checks:
    - Tensors are float32
    - Shape is (C, H, W)
    - Values are in [-1, 1] range
    - Only clean/long exposures are loaded

    The _validate_training_data_purity() is called automatically in SimplePTDataset.__init__()
    """
    dist.print0("Validating dataset compatibility...")

    # Check a few samples
    for idx in [0, min(1, len(dataset) - 1), len(dataset) - 1]:
        try:
            # Get raw image
            image, label = dataset[idx]

            # Validate dtype
            if image.dtype != torch.float32 and image.dtype != np.float32:
                dist.print0(
                    f"❌ Wrong dtype at index {idx}: {image.dtype}, expected float32"
                )
                return False

            # Validate shape
            if image.ndim != 3:
                dist.print0(f"❌ Wrong shape at index {idx}: {image.shape}, expected 3D")
                return False

            if image.shape[0] != dataset.num_channels:
                dist.print0(
                    f"❌ Wrong channel count at index {idx}: {image.shape[0]}, expected {dataset.num_channels}"
                )
                return False

            # Validate range
            if isinstance(image, np.ndarray):
                image_tensor = torch.from_numpy(image)
            else:
                image_tensor = image

            min_val = image_tensor.min().item()
            max_val = image_tensor.max().item()

            if min_val < -1.1 or max_val > 1.1:
                dist.print0(
                    f"⚠️  Image {idx} outside expected range [-1,1]: [{min_val:.3f}, {max_val:.3f}]"
                )
                # Warning, but not a failure

        except Exception as e:
            dist.print0(f"❌ Error loading sample {idx}: {e}")
            return False

    dist.print0(f"✅ Dataset validation passed: {len(dataset)} samples")
    dist.print0(
        f"   Shape: ({dataset.num_channels}, {dataset.resolution}, {dataset.resolution})"
    )
    dist.print0(f"   Dtype: float32")
    dist.print0(f"   Range: [-1, 1]")
    return True


def create_validation_metrics(device: torch.device):
    """
    Create validation metrics for image restoration evaluation.

    For unconditional diffusion + inverse problem, we need:
    - PSNR: Peak signal-to-noise ratio (pixel-level fidelity)
    - SSIM: Structural similarity index (perceptual similarity)
    """
    if not METRICS_AVAILABLE:
        dist.print0("⚠️  torchmetrics not available - using manual metrics computation")
        return None

    metrics = {
        "psnr": PeakSignalNoiseRatio(data_range=2.0).to(device),  # [-1,1] → range=2
        "ssim": StructuralSimilarityIndexMeasure(data_range=2.0).to(
            device
        ),  # [-1,1] → range=2
    }

    dist.print0("✅ Validation metrics initialized: PSNR, SSIM")
    return metrics


def main():
    """
    Train unconditional EDM diffusion model on clean images.

    This implements the standard inverse problem paradigm:
    1. Train P(x_clean) unconditionally on clean images
    2. At inference, solve P(x_clean | y_noisy) using physics-informed guidance
    3. Guidance encodes domain-specific physics (sensor noise models)

    Critical: Only long-exposure (clean) tiles used for training.
    Short-exposure (noisy) tiles used only for validation/testing.

    The dataset's _validate_training_data_purity() method will ASSERT that training
    data contains only long-exposure images, preventing model corruption.
    """
    parser = argparse.ArgumentParser(
        description="Train unconditional diffusion model on preprocessed clean images"
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to data directory containing preprocessed .pt tiles",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        required=True,
        help="Path to metadata JSON file with train/validation splits",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Total batch size across GPUs"
    )
    parser.add_argument(
        "--batch_gpu", type=int, default=None, help="Batch size per GPU (memory limit)"
    )
    parser.add_argument(
        "--total_kimg", type=int, default=None, help="Training duration in kimg"
    )
    parser.add_argument(
        "--ema_halflife_kimg", type=int, default=None, help="EMA half-life in kimg"
    )
    parser.add_argument(
        "--lr_rampup_kimg", type=int, default=None, help="LR ramp-up in kimg"
    )
    parser.add_argument(
        "--kimg_per_tick",
        type=int,
        default=None,
        help="Progress print interval in kimg",
    )
    parser.add_argument(
        "--snapshot_ticks",
        type=int,
        default=None,
        help="Snapshot save interval in ticks",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Early stopping patience in ticks",
    )

    # Model arguments
    parser.add_argument(
        "--img_resolution", type=int, default=None, help="Image resolution"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Number of image channels (3 for RGB, 1 for grayscale)",
    )
    parser.add_argument(
        "--model_channels", type=int, default=None, help="Model base channels"
    )
    parser.add_argument(
        "--channel_mult",
        type=int,
        nargs="+",
        default=None,
        help="Channel multipliers per resolution",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/edm_diffusion_training",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint file",
    )

    # Device arguments
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    # Config file argument - REQUIRED
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., config/diffusion.yaml, config/sony.yaml)",
    )

    # Additional options
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for parallel data loading",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Pin memory for faster GPU transfer",
    )

    args = parser.parse_args()

    # Load config file (required - will raise ValueError if missing)
    try:
        config = load_yaml_config(args.config)
        apply_config_to_args(args, config)
        dist.print0(f"Configuration loaded from: {args.config}")
    except ValueError as e:
        dist.print0(f"ERROR: {e}")
        sys.exit(1)

    # Initialize distributed training
    dist.init()

    # Setup output directory and logging
    run_dir = args.output_dir
    setup_output_directory(run_dir)
    setup_logging(run_dir)

    # ============================================================================
    # CRITICAL: Create UNCONDITIONAL dataset (no labels, no conditioning)
    # ============================================================================

    dist.print0("=" * 70)
    dist.print0("LOADING UNCONDITIONAL DATASET (clean images only)")
    dist.print0("=" * 70)

    # Create training dataset - UNCONDITIONAL TRAINING
    # Key: use_labels=False means no class conditioning
    dataset_kwargs = dict(
        class_name="data.dataset.SimplePTDataset",
        data_root=args.data_root,
        metadata_json=args.metadata_json,
        split="train",
        image_size=args.img_resolution,
        channels=args.channels,
        use_labels=False,  # ✅ UNCONDITIONAL - no labels
        label_dim=0,  # ✅ No label dimension
    )

    train_dataset = external.edm.dnnlib.util.construct_class_by_name(**dataset_kwargs)

    dist.print0(f"✅ Training dataset: {len(train_dataset)} clean samples")
    dist.print0(f"   Type: Unconditional (P(x_clean) only)")

    # Validate dataset
    if not validate_dataset_compatibility(train_dataset, torch.device("cpu")):
        raise RuntimeError("Dataset validation failed")

    # Create validation dataset
    val_dataset_kwargs = dataset_kwargs.copy()
    val_dataset_kwargs["split"] = "validation"
    val_dataset = external.edm.dnnlib.util.construct_class_by_name(**val_dataset_kwargs)
    dist.print0(f"✅ Validation dataset: {len(val_dataset)} clean samples")

    # ============================================================================
    # CRITICAL: Configure DataLoader with optimal settings for A40 (40GB)
    # ============================================================================

    dist.print0("=" * 70)
    dist.print0("CONFIGURING DATALOADERS")
    dist.print0("=" * 70)

    # Create DataLoaders with optimal workers and pinning
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_gpu or 32,  # Per-GPU batch size
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,  # Keep workers alive
        drop_last=True,  # Drop incomplete batches for consistency
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_gpu or 32,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    dist.print0(f"✅ DataLoaders configured:")
    dist.print0(f"   Training batches: {len(train_loader)}")
    dist.print0(f"   Validation batches: {len(val_loader)}")
    dist.print0(f"   Workers: {args.num_workers}, Pin memory: {args.pin_memory}")

    # ============================================================================
    # Setup validation metrics
    # ============================================================================

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    validation_metrics = create_validation_metrics(device)

    # Configure network, loss, and optimizer
    dist.print0("=" * 70)
    dist.print0("CONFIGURING MODEL, LOSS, AND OPTIMIZER")
    dist.print0("=" * 70)

    # ✅ Network is UNCONDITIONAL (no class conditioning)
    network_kwargs = create_network_config(train_dataset, args)
    loss_kwargs = create_loss_config()
    optimizer_kwargs = create_optimizer_config(args)

    dist.print0(f"✅ Model: Unconditional UNet (no class conditioning)")
    dist.print0(f"✅ Loss: L2 denoising score matching")

    # Resume from checkpoint if specified
    checkpoint_path = args.resume_from
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(args.output_dir)
    resume_state, start_kimg = load_checkpoint(checkpoint_path)

    # ============================================================================
    # Run training loop
    # ============================================================================

    dist.print0("=" * 70)
    dist.print0("STARTING UNCONDITIONAL DIFFUSION TRAINING")
    dist.print0("=" * 70)
    dist.print0("Training paradigm: Inverse Problem (DPS/DDRM style)")
    dist.print0("- Phase 1 (this script): Train P(x_clean) unconditionally")
    dist.print0(
        "- Phase 2 (inference script): Solve P(x_clean | y_noisy) with guidance"
    )
    dist.print0("=" * 70)
    dist.print0("")
    dist.print0("INFERENCE GUIDANCE OPTIONS:")
    dist.print0("  --variance_formula='simplified' (default): Var ≈ α·s·x + σ_r²")
    dist.print0("  --variance_formula='exact': Var = α·x/(white-black) + σ_r²")
    dist.print0("")

    training_loop(
        run_dir=run_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        network_kwargs=network_kwargs,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        seed=args.seed,
        batch_size=args.batch_size,
        batch_gpu=args.batch_gpu,
        total_kimg=args.total_kimg,
        ema_halflife_kimg=args.ema_halflife_kimg,
        lr_rampup_kimg=args.lr_rampup_kimg,
        kimg_per_tick=args.kimg_per_tick,
        snapshot_ticks=args.snapshot_ticks,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
        resume_state=resume_state,
        start_kimg=start_kimg,
    )

    dist.print0("=" * 70)
    dist.print0("UNCONDITIONAL DIFFUSION TRAINING COMPLETED")
    dist.print0("=" * 70)
    dist.print0("Next step: Create inference script to apply physics-informed guidance")
    dist.print0("=" * 70)


if __name__ == "__main__":
    import numpy as np

    main()
