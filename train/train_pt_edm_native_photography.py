#!/usr/bin/env python3
"""
Train restoration model using EDM's native training code with FLOAT32 .pt files.

This script uses EDM's native utilities with 32-bit float .pt files (NO QUANTIZATION):
- Uses torch_utils.distributed for distributed training
- Uses torch_utils.training_stats for metrics tracking
- Uses torch_utils.misc for model utilities (EMA, checkpointing, etc.)
- Uses dnnlib.util for general utilities
- Follows EDM's training_loop.py structure

Key Differences from PNG training:
- Loads 32-bit float .pt files instead of 8-bit PNG
- NO quantization loss - preserves full precision
- Data normalized by pipeline: [domain_min, domain_max] → [0,1] → [-1,1]
  * Photography: [0, 15871] → [0,1] → [-1,1]
  * Microscopy: [0, 65535] → [0,1] → [-1,1]
  * Astronomy: [-65, 385] → [0,1] → [-1,1]
- Float32 throughout the pipeline
- Data already in [-1, 1] range, ready for EDM training
"""

import argparse
import copy
import json
import os
import pickle  # nosec B403
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

# Add project root and EDM to path
project_root = Path(__file__).parent.parent  # Go up from train/ to project root
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

# Import EDM native components
import external.edm.dnnlib

# Import our PT dataset
from data.dataset import create_edm_pt_datasets
from external.edm.torch_utils import distributed as dist
from external.edm.torch_utils import misc, training_stats
from external.edm.training.loss import EDMLoss
from external.edm.training.networks import EDMPrecond


def training_loop(
    run_dir=".",
    train_dataset=None,
    val_dataset=None,
    network_kwargs={},
    loss_kwargs={},
    optimizer_kwargs={},
    seed=0,
    batch_size=4,
    batch_gpu=None,
    total_kimg=10000,
    ema_halflife_kimg=500,
    lr_rampup_kimg=1000,
    kimg_per_tick=50,
    snapshot_ticks=10,
    early_stopping_patience=5,
    device=torch.device("cuda"),
    resume_state=None,
    start_kimg=0,
):
    """
    Main training loop for float32 .pt data following EDM's structure.

    CRITICAL: This handles float32 data already in [-1, 1] range from pipeline!
    Pipeline applies domain-specific scaling followed by [-1, 1] normalization:
    - Photography: ADU / 16000.0 -> normalize to [-1, 1]
    - Microscopy: ADU / 65535.0 -> normalize to [-1, 1]
    - Astronomy: (ADU + 5.0) / 110.0 -> normalize to [-1, 1]

    No additional normalization needed - data is ready for EDM training.
    """
    # Initialize
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = True

    # Fresh start - will be updated if resuming
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = 0

    # Select batch size per GPU
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Setup data loader using EDM's InfiniteSampler
    dist.print0("Loading dataset...")
    dataset_sampler = misc.InfiniteSampler(
        dataset=train_dataset,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            pin_memory=True,
            num_workers=2,
            prefetch_factor=2,
        )
    )

    dist.print0(f"Dataset: {len(train_dataset)} training samples (float32 .pt)")
    dist.print0(f"         {len(val_dataset)} validation samples (float32 .pt)")
    dist.print0(f"         Data already normalized to [-1, 1] by pipeline")

    # Construct network using EDM's pattern
    dist.print0("Constructing network...")
    net = external.edm.dnnlib.util.construct_class_by_name(**network_kwargs)
    net.train().requires_grad_(True).to(device)

    # Print model summary using EDM's utility
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros(
                [batch_gpu, net.img_channels, net.img_resolution, net.img_resolution],
                device=device,
            )
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer and loss using EDM's pattern
    dist.print0("Setting up optimizer...")

    # Create/restore loss function
    if resume_state is not None and "loss_fn" in resume_state:
        loss_fn = resume_state["loss_fn"]
        dist.print0("Restored loss function from checkpoint")
    else:
        loss_fn = external.edm.dnnlib.util.construct_class_by_name(**loss_kwargs)

    # Create optimizer
    optimizer = external.edm.dnnlib.util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )

    # Restore optimizer state if resuming
    if resume_state is not None and "optimizer_state" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        dist.print0("Restored optimizer state from checkpoint")

    # Wrap with DDP if using distributed training
    if dist.get_world_size() > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    else:
        ddp = net

    # Create EMA model using EDM's pattern
    if resume_state is not None and "ema" in resume_state:
        ema = resume_state["ema"].to(device)
        dist.print0("Restored EMA model from checkpoint")

        # Calculate current training state
        cur_nimg = start_kimg * 1000
        cur_tick = cur_nimg // (kimg_per_tick * 1000)
        tick_start_nimg = cur_tick * kimg_per_tick * 1000

        dist.print0(
            f"Checkpoint loaded - continuing from {start_kimg} kimg (tick {cur_tick})"
        )
        dist.print0(
            f"Remaining: {total_kimg - start_kimg} kimg to reach {total_kimg} kimg"
        )
    else:
        ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Training loop following EDM's structure
    remaining_kimg = total_kimg - start_kimg
    if resume_state is not None:
        dist.print0(
            f"Training for remaining {remaining_kimg} kimg (to reach {total_kimg} kimg)..."
        )
    else:
        dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()

    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    best_val_loss = float("inf")
    patience_counter = 0

    while True:
        # Accumulate gradients
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                # Get batch - returns float32 tensors in [0, ~1] range from pipeline
                images, labels = next(dataset_iterator)

                # Move to device
                images = images.to(device)
                labels = labels.to(device)

                # CRITICAL: Handle float32 data properly
                # Pipeline already normalized to [-1, 1] using domain-specific scaling + normalization:
                # 1. Domain-specific scaling: image / fixed_scale (photography: 16000, microscopy: 65535, astronomy: 110)
                # 2. Astronomy offset: (image + 5.0) / fixed_scale
                # 3. Final normalization: transforms.Normalize(mean=[0.5], std=[0.5]) -> [-1, 1]

                # Convert to float32 if not already
                images = images.to(torch.float32)

                # Data is already in [-1, 1] range from pipeline normalization
                # No additional normalization needed

                # Compute loss using EDM's native loss
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=None)
                training_stats.report("Loss/loss", loss)
                loss.sum().mul(1.0 / batch_gpu_total).backward()

        # Update weights with gradient clipping
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
        for param in net.parameters():
            if param.grad is not None:
                misc.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        optimizer.step()

        # Update EMA using EDM's formula
        ema_halflife_nimg = ema_halflife_kimg * 1000
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Update progress
        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000

        # Perform maintenance tasks once per tick
        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)
        ):
            continue

        # Print status line using EDM's format
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {external.edm.dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]

        # Add GPU memory stats if available
        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / 2**30
            current_mem_gb = torch.cuda.memory_allocated(device) / 2**30
            fields += [
                f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', peak_mem_gb):<6.2f}"
            ]
            fields += [f"gpu_cur {current_mem_gb:<6.2f}"]
            torch.cuda.reset_peak_memory_stats()

        dist.print0(" ".join(fields))

        # Save network snapshot using EDM's format
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            # Run validation
            val_loss = validate(ema, val_dataset, loss_fn, device)
            dist.print0(f"Validation loss: {val_loss:.4f}")

            # Check if best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                dist.print0(f"New best validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                dist.print0(
                    f"No improvement. Patience: {patience_counter}/{early_stopping_patience}"
                )

                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    dist.print0(
                        f"Early stopping triggered after {patience_counter} checks without improvement"
                    )
                    dist.print0(f"Best validation loss: {best_val_loss:.4f}")
                    done = True

            # Save checkpoint
            data = dict(
                ema=ema,
                loss_fn=loss_fn,
                optimizer_state=optimizer.state_dict(),
                dataset_kwargs=dict(),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if dist.get_world_size() > 1:
                        misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value

            if dist.get_rank() == 0:
                checkpoint_path = os.path.join(
                    run_dir, f"network-snapshot-{cur_nimg//1000:06d}.pkl"
                )
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(data, f)
                dist.print0(f"Saved checkpoint: {checkpoint_path}")

                # Save best model
                if is_best:
                    best_path = os.path.join(run_dir, "best_model.pkl")
                    with open(best_path, "wb") as f:
                        pickle.dump(data, f)
                    dist.print0(f"Saved best model: {best_path}")

            del data

        # Update logs
        try:
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
                stats_jsonl.write(
                    json.dumps(
                        dict(
                            training_stats.default_collector.as_dict(),
                            timestamp=time.time(),
                        )
                    )
                    + "\n"
                )
                stats_jsonl.flush()
        except Exception as e:
            dist.print0(f"ERROR in logging: {e}")
            if dist.get_rank() == 0 and stats_jsonl is not None:
                try:
                    stats_jsonl.close()
                except:
                    pass
                stats_jsonl = None
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time

        if done:
            break

    # Done
    dist.print0()
    dist.print0("Exiting...")


@torch.no_grad()
def validate(ema_model, val_dataset, loss_fn, device):
    """Run validation with float32 .pt data in [-1, 1] range from pipeline."""
    ema_model.eval()

    val_sampler = misc.InfiniteSampler(
        dataset=val_dataset,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=42,
    )
    val_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=val_dataset,
            sampler=val_sampler,
            batch_size=4,
            pin_memory=True,
            num_workers=2,
        )
    )

    total_loss = 0.0
    num_batches = 0

    for _ in range(min(50, len(val_dataset) // 4)):
        try:
            # Get batch - float32 tensors in [-1, 1] from pipeline
            images, labels = next(val_iterator)

            # Move to device
            images = images.to(device)
            labels = labels.to(device)

            # Convert to float32 - data is already in [-1, 1] range from pipeline
            # Pipeline applies domain-specific scaling + transforms.Normalize(mean=[0.5], std=[0.5])
            images = images.to(torch.float32)
            # No additional normalization needed - tensors are already in [-1, 1] range

            # Compute loss
            loss = loss_fn(
                net=ema_model, images=images, labels=labels, augment_pipe=None
            )
            total_loss += loss.mean().item()
            num_batches += 1

        except StopIteration:
            break

    return total_loss / max(num_batches, 1)


def main():
    """Main training function for float32 .pt data.

    The training script expects .pt files that have been processed by the pipeline with:
    - Domain-specific physics calibration (ADU → electrons)
    - Domain-specific scaling to [0, ~1] range:
      * Photography: electrons / 16000.0
      * Microscopy: electrons / 65535.0
      * Astronomy: (electrons + 5.0) / 110.0
    - Final normalization to [-1, 1] using transforms.Normalize(mean=[0.5], std=[0.5])
    - Metadata JSON with splits, calibration parameters, and scaling info

    The training loop will:
    1. Load float32 .pt data already in [-1, 1] range
    2. Use data directly for EDM training (no additional normalization)
    3. Train the diffusion model with full precision (no quantization)
    """
    parser = argparse.ArgumentParser(
        description="Train model with EDM native training using float32 .pt files (NO QUANTIZATION)"
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to data directory containing .pt tiles",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        required=True,
        help="Path to metadata JSON file with splits and calibration info",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Total batch size")
    parser.add_argument(
        "--batch_gpu", type=int, default=None, help="Limit batch size per GPU"
    )
    parser.add_argument(
        "--total_kimg", type=int, default=10000, help="Training duration in kimg"
    )
    parser.add_argument(
        "--ema_halflife_kimg", type=int, default=500, help="EMA half-life in kimg"
    )
    parser.add_argument(
        "--lr_rampup_kimg", type=int, default=1000, help="LR ramp-up in kimg"
    )
    parser.add_argument(
        "--kimg_per_tick", type=int, default=50, help="Progress print interval in kimg"
    )
    parser.add_argument(
        "--snapshot_ticks", type=int, default=10, help="Snapshot save interval in ticks"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )

    # Model arguments
    parser.add_argument(
        "--img_resolution", type=int, default=256, help="Image resolution"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Number of image channels (1 for grayscale, 3 for RGB)",
    )
    parser.add_argument(
        "--model_channels", type=int, default=192, help="Model channels"
    )
    parser.add_argument(
        "--channel_mult",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Channel multipliers",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/edm_npy_training",
        help="Output directory",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume training from checkpoint file",
    )

    # Device arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for training"
    )

    args = parser.parse_args()

    # Initialize distributed training
    dist.init()

    # Setup output directory
    run_dir = args.output_dir
    if dist.get_rank() == 0:
        os.makedirs(run_dir, exist_ok=True)
        dist.print0("=" * 60)
        dist.print0("EDM NATIVE TRAINING WITH FLOAT32 .PT FILES")
        dist.print0("NO QUANTIZATION - FULL PRECISION")
        dist.print0("=" * 60)

    # Setup logging
    if dist.get_rank() == 0:
        log_file = os.path.join(run_dir, "log.txt")
        logger = external.edm.dnnlib.util.Logger(
            file_name=log_file, file_mode="a", should_flush=True
        )

    # Create datasets
    dist.print0("Loading float32 .pt datasets...")
    dataset_kwargs = dict(
        class_name="data.dataset.EDMPTDataset",
        data_root=args.data_root,
        metadata_json=args.metadata_json,
        split="train",
        image_size=args.img_resolution,
        channels=args.channels,
        domain="photography",  # Specify domain explicitly for comprehensive metadata files
        use_labels=True,
        label_dim=3,
        data_range="normalized",  # Pipeline outputs [-1, 1] normalized data
        max_size=None,
    )

    # Create datasets
    train_dataset = external.edm.dnnlib.util.construct_class_by_name(**dataset_kwargs)
    val_dataset_kwargs = dataset_kwargs.copy()
    val_dataset_kwargs["split"] = "validation"
    val_dataset = external.edm.dnnlib.util.construct_class_by_name(**val_dataset_kwargs)

    # Configure network
    network_kwargs = external.edm.dnnlib.EasyDict(
        class_name="external.edm.training.networks.EDMPrecond",
        img_resolution=train_dataset.resolution,
        img_channels=train_dataset.num_channels,
        label_dim=train_dataset.label_dim,
        use_fp16=False,
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        model_type="DhariwalUNet",
        model_channels=args.model_channels,
        channel_mult=args.channel_mult,
    )

    # Configure loss
    loss_kwargs = external.edm.dnnlib.EasyDict(
        class_name="external.edm.training.loss.EDMLoss",
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
    )

    # Configure optimizer
    optimizer_kwargs = external.edm.dnnlib.EasyDict(
        class_name="torch.optim.Adam",
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Resume from checkpoint if specified
    resume_state = None
    start_kimg = 0
    checkpoint_path = args.resume_from

    if checkpoint_path is None:
        # Auto-find latest checkpoint
        checkpoint_dir = args.output_dir
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [
                f
                for f in os.listdir(checkpoint_dir)
                if f.startswith("network-snapshot-") and f.endswith(".pkl")
            ]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.split("-")[2].split(".")[0]))
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
                dist.print0(f"Auto-resuming from: {checkpoint_path}")

    if checkpoint_path is not None:
        dist.print0(f"Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path, "rb") as f:
            resume_state = pickle.load(f)  # nosec B301
        checkpoint_name = os.path.basename(checkpoint_path)
        start_kimg = int(checkpoint_name.split("-")[2].split(".")[0])
        dist.print0(f"Resuming from {start_kimg} kimg")

    # Run training loop
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

    dist.print0("=" * 60)
    dist.print0("TRAINING COMPLETED SUCCESSFULLY")
    dist.print0("=" * 60)


if __name__ == "__main__":
    main()
