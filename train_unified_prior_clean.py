#!/usr/bin/env python3
"""
Optimized unified training script for prior_clean dataset.

This script is specifically optimized for the prior_clean dataset which contains:
- 42,109 total clean image tiles (128x128)
- 3 domains: Photography (4ch), Microscopy (1ch), Astronomy (1ch)
- Perfect for diffusion prior training with domain conditioning

Key optimizations:
- H100 GPU optimization with large batch sizes
- Multi-domain balanced sampling
- Efficient data loading for tiled dataset
- Progressive training with proper scheduling

Usage:
    python train_unified_prior_clean.py --data_root /opt/dlami/nvme/preprocessed/prior_clean
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.error_handlers import ErrorHandler
from core.logging_config import LoggingManager
from models.edm_wrapper import create_edm_wrapper, create_progressive_edm
from poisson_training import MultiDomainTrainingConfig, set_deterministic_mode

# Setup logging
logging_manager = LoggingManager()
logger = logging_manager.setup_logging(
    level="INFO",
    log_dir="logs",
    console_output=True,
    file_output=True,
    json_format=False,
)


class PriorCleanDataset(Dataset):
    """Optimized dataset for prior_clean multi-domain training."""

    def __init__(
        self,
        data_root: str,
        domains: List[str] = ["photography", "microscopy", "astronomy"],
        split: str = "train",
        max_files_per_domain: Optional[int] = None,
        balance_domains: bool = True,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.domains = domains
        self.split = split
        self.max_files_per_domain = max_files_per_domain
        self.balance_domains = balance_domains
        self.seed = seed

        # Domain info
        self.domain_info = {
            "photography": {"id": 0, "channels": 4},
            "microscopy": {"id": 1, "channels": 1},
            "astronomy": {"id": 2, "channels": 1},
        }

        # Load files for each domain
        self.files = []
        self.domain_weights = []

        for domain in self.domains:
            domain_path = self.data_root / domain / split
            if not domain_path.exists():
                logger.warning(
                    f"Domain {domain} split {split} not found: {domain_path}"
                )
                continue

            # Get all files for this domain
            domain_files = list(domain_path.glob("*.pt"))
            domain_files.sort()  # For reproducibility

            # Apply max files limit
            if self.max_files_per_domain is not None:
                domain_files = domain_files[: self.max_files_per_domain]

            # Add domain info to each file
            for file_path in domain_files:
                self.files.append(
                    {
                        "path": file_path,
                        "domain": domain,
                        "domain_id": self.domain_info[domain]["id"],
                        "channels": self.domain_info[domain]["channels"],
                    }
                )

            logger.info(f"  {domain}: {len(domain_files):,} files")

        if len(self.files) == 0:
            raise ValueError(f"No files found in {self.data_root}")

        # Create sampling weights for balanced training
        if self.balance_domains:
            self._create_balanced_weights()

        logger.info(f"Total {split} files: {len(self.files):,}")

    def _create_balanced_weights(self):
        """Create sampling weights for balanced domain training."""
        domain_counts = {}
        for file_info in self.files:
            domain = file_info["domain"]
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Calculate weights (inverse frequency)
        total_files = len(self.files)
        domain_weights = {}
        for domain, count in domain_counts.items():
            domain_weights[domain] = total_files / (len(self.domains) * count)

        # Assign weights to each file
        self.sample_weights = []
        for file_info in self.files:
            domain = file_info["domain"]
            self.sample_weights.append(domain_weights[domain])

        logger.info("Domain balancing weights:")
        for domain, weight in domain_weights.items():
            count = domain_counts[domain]
            logger.info(f"  {domain}: {count:,} files, weight: {weight:.4f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a single sample with unified 4-channel format."""
        file_info = self.files[idx]
        file_path = file_info["path"]
        domain = file_info["domain"]
        domain_id = file_info["domain_id"]
        channels = file_info["channels"]

        # Load the .pt file
        data = torch.load(file_path, map_location="cpu")

        # Extract clean image
        clean_norm = data["clean_norm"]  # [C, 128, 128]
        metadata = data.get("metadata", {})

        # Ensure 4 channels for unified model
        if clean_norm.shape[0] < 4:
            # Pad with zeros to reach 4 channels
            pad_channels = 4 - clean_norm.shape[0]
            padding = torch.zeros(pad_channels, 128, 128)
            clean_norm = torch.cat([clean_norm, padding], dim=0)

        # Create domain conditioning vector (6D)
        # [domain_one_hot(3), log_scale_norm(1), rel_read_noise(1), rel_background(1)]

        # Domain one-hot
        domain_onehot = torch.zeros(3)
        domain_onehot[domain_id] = 1.0

        # Domain-specific parameters (estimated from research)
        domain_params = {
            "photography": {"scale": 10000.0, "read_noise": 3.0, "background": 0.0},
            "microscopy": {"scale": 5000.0, "read_noise": 2.0, "background": 0.0},
            "astronomy": {"scale": 50000.0, "read_noise": 1.0, "background": 0.0},
        }

        params = domain_params[domain]
        scale = params["scale"]
        read_noise = params["read_noise"]
        background = params["background"]

        # Normalized parameters
        log_scale_norm = torch.log10(torch.tensor(scale)).clamp(-1, 1)
        rel_read_noise = torch.tensor(read_noise / scale).clamp(0, 1)
        rel_background = torch.tensor(background / scale).clamp(0, 1)

        condition = torch.cat(
            [
                domain_onehot,
                log_scale_norm.unsqueeze(0),
                rel_read_noise.unsqueeze(0),
                rel_background.unsqueeze(0),
            ]
        )

        return {
            "clean_norm": clean_norm,
            "condition": condition,
            "domain": domain,
            "domain_id": torch.tensor(domain_id),
            "metadata": metadata,
        }

    def get_balanced_sampler(self) -> Optional[WeightedRandomSampler]:
        """Get balanced sampler for training."""
        if self.balance_domains and hasattr(self, "sample_weights"):
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.files),
                replacement=True,
            )
        return None


class PriorCleanTrainingManager:
    """Optimized training manager for prior_clean dataset."""

    def __init__(
        self,
        data_root: str,
        domains: List[str] = ["photography", "microscopy", "astronomy"],
        output_dir: str = "results/prior_clean_training",
        device: str = "auto",
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.domains = domains
        self.output_dir = Path(output_dir)
        self.device = self._setup_device(device)
        self.seed = seed

        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        # Initialize error handler
        self.error_handler = ErrorHandler(
            logger=logger, enable_recovery=True, strict_mode=False
        )

        logger.info("🌟 Prior Clean Training Manager initialized")
        logger.info(f"  Data root: {self.data_root}")
        logger.info(f"  Domains: {', '.join(self.domains)}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Seed: {self.seed}")

    def _setup_device(self, device: str) -> str:
        """Setup computation device with H100-specific optimizations."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"

                # H100-specific optimizations
                device_name = torch.cuda.get_device_name()
                is_h100 = "H100" in device_name

                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                torch.backends.cuda.enable_flash_sdp(True)

                # H100-specific settings
                if is_h100:
                    logger.info("🚀 H100 detected - enabling advanced optimizations")
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

                logger.info(f"🚀 Using CUDA device: {device_name}")
                logger.info(
                    f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )

                if is_h100:
                    logger.info(
                        "  🔥 H100 optimizations: TF32, Flash Attention, 80GB HBM3"
                    )

            else:
                device = "cpu"
                logger.info("💻 Using CPU device")

        return device

    def create_datasets(
        self, max_files_per_domain: Optional[int] = None, balance_domains: bool = True
    ) -> Dict[str, PriorCleanDataset]:
        """Create train/val datasets."""
        logger.info("📁 Creating prior_clean datasets...")

        datasets = {}
        for split in ["train", "val"]:
            try:
                dataset = PriorCleanDataset(
                    data_root=str(self.data_root),
                    domains=self.domains,
                    split=split,
                    max_files_per_domain=max_files_per_domain,
                    balance_domains=balance_domains,
                    seed=self.seed,
                )
                datasets[split] = dataset
                logger.info(f"  {split}: {len(dataset):,} samples")
            except Exception as e:
                logger.warning(f"  {split}: Failed to load - {e}")

        if not datasets:
            raise ValueError("No datasets could be created")

        return datasets

    def create_model(
        self,
        use_multi_resolution: bool = False,
        mixed_precision: bool = True,
        h100_optimizations: bool = True,
        **model_kwargs,
    ) -> nn.Module:
        """Create optimized unified model."""
        logger.info("🤖 Creating optimized unified model...")

        # H100-optimized model configuration
        if h100_optimizations and "H100" in torch.cuda.get_device_name():
            logger.info("🔥 Applying H100 optimizations to model")
            # Larger model for H100
            default_channels = model_kwargs.get("model_channels", 320)
            default_blocks = model_kwargs.get("num_blocks", 8)
        else:
            # Standard configuration
            default_channels = model_kwargs.get("model_channels", 256)
            default_blocks = model_kwargs.get("num_blocks", 6)

        if use_multi_resolution:
            logger.info("📈 Using Progressive Multi-Resolution EDM")

            model = create_progressive_edm(
                min_resolution=32,
                max_resolution=128,
                num_stages=4,
                model_channels=default_channels,
                img_channels=4,  # Unified 4-channel
                label_dim=6,  # Domain conditioning
                use_fp16=False,
                dropout=0.1,
                **{
                    k: v
                    for k, v in model_kwargs.items()
                    if k
                    not in [
                        "model_channels",
                        "img_channels",
                        "label_dim",
                        "use_fp16",
                        "dropout",
                    ]
                },
            )
        else:
            logger.info("📊 Using Standard EDM")

            model_config = {
                "img_resolution": 128,
                "img_channels": 4,  # Unified 4-channel
                "model_channels": default_channels,
                "channel_mult": model_kwargs.get("channel_mult", [1, 2, 3, 4]),
                "channel_mult_emb": model_kwargs.get("channel_mult_emb", 8),
                "num_blocks": default_blocks,
                "attn_resolutions": model_kwargs.get("attn_resolutions", [16, 32, 64]),
                "label_dim": 6,  # Domain conditioning
                "use_fp16": False,
                "dropout": 0.1,
            }

            model = create_edm_wrapper(**model_config)

        # Move to device
        model = model.to(self.device)
        param_count = sum(p.numel() for p in model.parameters())

        logger.info("✅ Unified model created successfully")
        logger.info(f"  Parameters: {param_count:,}")
        logger.info(f"  Domains: {', '.join(self.domains)}")
        logger.info(f"  Architecture: {default_channels}ch, {default_blocks} blocks")

        return model

    def validate_model(
        self, model: nn.Module, val_loader: DataLoader, device: torch.device, step: int
    ) -> float:
        """Run validation and return average loss."""
        model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                clean = batch["clean_norm"].to(device)
                condition = batch["condition"].to(device)

                # Use same noise schedule as training
                noise = torch.randn_like(clean)
                sigma = torch.exp(
                    torch.randn(clean.shape[0], device=device) * 1.2 - 1.2
                )
                sigma = sigma.view(-1, 1, 1, 1)
                noisy = clean + sigma * noise

                # Forward pass
                sigma_flat = sigma.squeeze()
                predicted = model(noisy, sigma_flat, condition=condition)

                # v-parameterization loss
                c_skip = 1 / (sigma**2 + 1)
                c_out = sigma / (sigma**2 + 1).sqrt()
                target = (clean - c_skip * noisy) / c_out
                loss = F.mse_loss(predicted, target)

                total_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        model.train()
        return avg_val_loss

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        config: Dict[str, Any],
        loss: float = None,
        val_loss: float = None,
        is_best: bool = False,
        ema_model: nn.Module = None,
        scheduler=None,
        warmup_scheduler=None,
    ):
        """Save a checkpoint with comprehensive metadata."""
        checkpoint_info = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
        }

        # Save scheduler states
        if scheduler is not None:
            checkpoint_info["scheduler_state_dict"] = scheduler.state_dict()
        if warmup_scheduler is not None:
            checkpoint_info[
                "warmup_scheduler_state_dict"
            ] = warmup_scheduler.state_dict()

        # Add EMA model to checkpoint
        if ema_model is not None:
            checkpoint_info["ema_model_state_dict"] = ema_model.state_dict()

        if loss is not None:
            checkpoint_info["train_loss"] = loss
        if val_loss is not None:
            checkpoint_info["val_loss"] = val_loss

        # Regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{step:07d}.pth"
        torch.save(checkpoint_info, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

        # Best model checkpoint
        if is_best and val_loss is not None:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint_info, best_path)
            logger.info(f"🏆 Best model updated: {best_path} (val_loss: {val_loss:.6f})")

    def save_phase_checkpoint(
        self, model: nn.Module, step: int, config: Dict[str, Any]
    ):
        """Save checkpoint at different training phases for unified model."""
        # Phase milestones (in training steps)
        phases = {
            100000: "photography_only",
            200000: "photography_microscopy",
            300000: "all_domains_phase1",
            400000: "all_domains_phase2",
            450000: "final_unified_model",
        }

        # Find the closest phase milestone
        phase_step = None
        phase_name = None
        for milestone in sorted(phases.keys()):
            if step >= milestone:
                phase_step = milestone
                phase_name = phases[milestone]
            else:
                break

        if phase_step is not None:
            phase_checkpoint_path = (
                self.output_dir / f"phase_{phase_name}_step_{phase_step:06d}.pth"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "step": phase_step,
                    "phase": phase_name,
                    "config": config,
                },
                phase_checkpoint_path,
            )
            logger.info(f"🎯 Phase checkpoint saved: {phase_checkpoint_path}")
            logger.info(f"   Phase: {phase_name} (step {phase_step:,})")

    def create_optimized_config(
        self,
        h100_optimizations: bool = True,
        mixed_precision: bool = True,
        **config_kwargs,
    ) -> Dict[str, Any]:
        """Create H100-optimized training configuration."""

        is_h100 = "H100" in torch.cuda.get_device_name()

        if h100_optimizations and is_h100:
            logger.info("🔥 Creating H100-optimized configuration")

            # H100-optimized settings with reduced training time
            config = {
                # Training scale - reduced from 1M to 450K steps
                "max_steps": config_kwargs.get(
                    "max_steps", 450000
                ),  # Reduced for faster completion
                "batch_size": config_kwargs.get(
                    "batch_size", 32
                ),  # Large batch for H100
                "gradient_accumulation_steps": config_kwargs.get(
                    "gradient_accumulation_steps", 2
                ),
                "learning_rate": config_kwargs.get(
                    "learning_rate", 1e-4
                ),  # Higher for large batches
                # Model architecture
                "model_channels": config_kwargs.get("model_channels", 320),
                "num_blocks": config_kwargs.get("num_blocks", 8),
                "channel_mult_emb": config_kwargs.get("channel_mult_emb", 8),
                # Optimization
                "mixed_precision": mixed_precision,
                "precision": "bf16" if mixed_precision else "fp32",
                "gradient_clip_norm": config_kwargs.get("gradient_clip_norm", 1.0),
                # Scheduling
                "warmup_steps": config_kwargs.get("warmup_steps", 20000),
                "lr_scheduler": config_kwargs.get("lr_scheduler", "cosine"),
                # Validation and saving - frequent for research safety
                "val_frequency": config_kwargs.get(
                    "val_frequency", 5000
                ),  # Every 5K steps
                "save_frequency_steps": config_kwargs.get(
                    "save_frequency_steps", 5000
                ),  # Every 5K steps
                "phase_save_frequency": config_kwargs.get(
                    "phase_save_frequency", 50000
                ),  # Every 50K steps
                # Data loading
                "num_workers": config_kwargs.get("num_workers", 8),
                "pin_memory": True,
                "persistent_workers": True,
                # EMA for better inference
                "ema_decay": config_kwargs.get("ema_decay", 0.999),
            }

            logger.info("🔥 H100 Configuration:")
            logger.info(
                f"  Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['gradient_accumulation_steps']})"
            )
            logger.info(f"  Learning rate: {config['learning_rate']}")
            logger.info(
                f"  Model: {config['model_channels']}ch, {config['num_blocks']} blocks"
            )
            logger.info(f"  Precision: {config['precision']}")

        else:
            logger.info("📊 Creating standard configuration")

            # Conservative settings - reduced steps for faster completion
            config = {
                "max_steps": config_kwargs.get(
                    "max_steps", 450000
                ),  # Reduced for faster completion
                "batch_size": config_kwargs.get("batch_size", 16),
                "gradient_accumulation_steps": config_kwargs.get(
                    "gradient_accumulation_steps", 4
                ),
                "learning_rate": config_kwargs.get("learning_rate", 5e-5),
                "model_channels": config_kwargs.get("model_channels", 256),
                "num_blocks": config_kwargs.get("num_blocks", 6),
                "mixed_precision": False,
                "precision": "fp32",
                "gradient_clip_norm": config_kwargs.get("gradient_clip_norm", 0.5),
                "warmup_steps": config_kwargs.get("warmup_steps", 10000),
                "lr_scheduler": config_kwargs.get("lr_scheduler", "cosine"),
                "val_frequency": config_kwargs.get(
                    "val_frequency", 5000
                ),  # Every 5K steps
                "save_frequency_steps": config_kwargs.get(
                    "save_frequency_steps", 5000
                ),  # Every 5K steps
                "phase_save_frequency": config_kwargs.get(
                    "phase_save_frequency", 50000
                ),  # Every 50K steps
                "num_workers": config_kwargs.get("num_workers", 4),
                "pin_memory": True,
                "persistent_workers": False,
                "ema_decay": config_kwargs.get("ema_decay", 0.999),
            }

        # Override with any provided kwargs
        config.update(config_kwargs)

        return config


def main():
    """Main training function for prior_clean dataset."""
    parser = argparse.ArgumentParser(
        description="Train unified model on prior_clean multi-domain dataset"
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to prior_clean dataset root",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["photography", "microscopy", "astronomy"],
        help="Domains to train on",
    )

    # Training arguments
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Maximum training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation",
    )

    # Model arguments
    parser.add_argument(
        "--model_channels", type=int, default=None, help="Model channels"
    )
    parser.add_argument("--num_blocks", type=int, default=None, help="Number of blocks")
    parser.add_argument(
        "--channel_mult_emb",
        type=int,
        default=None,
        help="Channel multiplier for embeddings",
    )
    parser.add_argument(
        "--multi_resolution", action="store_true", help="Use multi-resolution"
    )

    # Optimization arguments
    parser.add_argument(
        "--h100_optimizations",
        action="store_true",
        default=True,
        help="Enable H100 optimizations",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Use mixed precision",
    )
    parser.add_argument(
        "--conservative", action="store_true", help="Use conservative settings"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="results/prior_clean_training"
    )
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    # Performance arguments
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)

    # Testing arguments
    parser.add_argument("--max_files_per_domain", type=int, default=None)
    parser.add_argument("--quick_test", action="store_true", help="Quick test mode")
    parser.add_argument(
        "--val_frequency", type=int, default=5000, help="Run validation every N steps"
    )
    parser.add_argument(
        "--save_frequency_steps",
        type=int,
        default=5000,
        help="Save regular checkpoints every N steps",
    )
    parser.add_argument(
        "--phase_save_frequency",
        type=int,
        default=50000,
        help="Save phase checkpoints every N steps",
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.max_files_per_domain = 100
        args.max_steps = 1000
        args.batch_size = 8
        args.conservative = True
        logger.info("🧪 Quick test mode enabled")

    # Conservative mode
    if args.conservative:
        args.h100_optimizations = False
        args.mixed_precision = False
        logger.info("🛡️ Conservative mode enabled")

    # Don't clear cache at start - it can cause fragmentation issues
    # PyTorch's memory allocator works better when allowed to manage memory naturally
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info("📊 Reset CUDA memory statistics")

    # Initialize training manager
    logger.info("🌟 INITIALIZING PRIOR_CLEAN UNIFIED TRAINING")
    logger.info("=" * 70)

    training_manager = PriorCleanTrainingManager(
        data_root=args.data_root,
        domains=args.domains,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )

    # Create datasets
    datasets = training_manager.create_datasets(
        max_files_per_domain=args.max_files_per_domain,
        balance_domains=True,
    )

    # Create optimized configuration
    config_kwargs = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ["h100_optimizations", "mixed_precision"]
    }
    config = training_manager.create_optimized_config(
        h100_optimizations=args.h100_optimizations,
        mixed_precision=args.mixed_precision,
        **config_kwargs,
    )

    # Create model
    model = training_manager.create_model(
        use_multi_resolution=args.multi_resolution,
        mixed_precision=config["mixed_precision"],
        h100_optimizations=args.h100_optimizations,
        model_channels=config["model_channels"],
        num_blocks=config["num_blocks"],
    )

    # Create data loaders
    train_sampler = datasets["train"].get_balanced_sampler()

    train_loader = DataLoader(
        datasets["train"],
        batch_size=config["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config.get("persistent_workers", False),
        drop_last=True,
    )

    val_loader = None
    if "val" in datasets:
        val_loader = DataLoader(
            datasets["val"],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            drop_last=False,
        )

    logger.info("🚀 Starting optimized unified training...")
    logger.info(f"  Total training samples: {len(datasets['train']):,}")
    logger.info(
        f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}"
    )
    logger.info(
        f"  Steps per epoch: {len(train_loader) // config['gradient_accumulation_steps']:,}"
    )
    logger.info(f"  Max steps: {config['max_steps']:,}")

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-3
    )

    # Learning rate scheduler setup
    scheduler = None
    warmup_scheduler = None

    if config.get("lr_scheduler") == "cosine":
        logger.info(f"🔄 Setting up cosine annealing scheduler")
        logger.info(f"  Warmup steps: {config.get('warmup_steps', 0):,}")
        logger.info(f"  Max steps: {config['max_steps']:,}")
        logger.info(f"  Initial LR: {config['learning_rate']:.2e}")

        warmup_steps = config.get("warmup_steps", 0)
        main_steps = config["max_steps"] - warmup_steps
        min_lr = config["learning_rate"] * 0.01  # 1% of initial LR

        if warmup_steps > 0:
            # Warmup scheduler: gradually increase LR from 1% to 100%
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,  # Start at 1% of LR
                end_factor=1.0,  # End at 100% of LR
                total_iters=warmup_steps,
            )
            logger.info(
                f"  Warmup: {config['learning_rate'] * 0.01:.2e} → {config['learning_rate']:.2e}"
            )

        # Main cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_steps, eta_min=min_lr
        )
        logger.info(f"  Cosine decay: {config['learning_rate']:.2e} → {min_lr:.2e}")
    else:
        logger.info("📊 Using fixed learning rate (no scheduler)")

    # EMA model for better inference performance
    ema_model = None
    ema_decay = config.get("ema_decay", 0.999)
    if ema_decay > 0:
        import copy

        ema_model = copy.deepcopy(model).eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        logger.info(f"✅ EMA model created with decay: {ema_decay}")

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if config["mixed_precision"] else None

    # Resume from checkpoint if provided
    start_step = 0
    if args.resume_checkpoint:
        logger.info(f"📥 Loading checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(
            args.resume_checkpoint, map_location=training_manager.device
        )

        # Load model and optimizer
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"]

        # Load scheduler states if they exist and we're using the same scheduler type
        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("  ✅ Scheduler state loaded")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not load scheduler state: {e}")
                logger.info("  📊 Using fresh scheduler (likely due to config changes)")

        if "warmup_scheduler_state_dict" in checkpoint and warmup_scheduler is not None:
            try:
                warmup_scheduler.load_state_dict(
                    checkpoint["warmup_scheduler_state_dict"]
                )
                logger.info("  ✅ Warmup scheduler state loaded")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not load warmup scheduler state: {e}")

        # Load EMA model if it exists
        if "ema_model_state_dict" in checkpoint and ema_model is not None:
            ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            logger.info("  ✅ EMA model state loaded")

        logger.info(f"  ✅ Resumed from step {start_step:,}")
        logger.info(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        # If we're resuming and the current LR differs significantly from config, log it
        expected_lr = config["learning_rate"]
        current_lr = optimizer.param_groups[0]["lr"]
        if abs(current_lr - expected_lr) > expected_lr * 0.1:  # >10% difference
            logger.info(f"  📊 LR mismatch detected:")
            logger.info(f"    Config LR: {expected_lr:.2e}")
            logger.info(f"    Loaded LR: {current_lr:.2e}")
            logger.info(f"    Scheduler will continue from loaded state")

    # Best model tracking
    best_val_loss = float("inf")
    step = start_step
    model.train()

    try:
        while step < config["max_steps"]:
            for batch in train_loader:
                if step >= config["max_steps"]:
                    break

                # Move to device
                clean = batch["clean_norm"].to(training_manager.device)
                condition = batch["condition"].to(training_manager.device)

                # Training step
                optimizer.zero_grad()

                with torch.amp.autocast(
                    "cuda",
                    enabled=config["mixed_precision"],
                    dtype=torch.bfloat16
                    if config.get("precision") == "bf16"
                    else torch.float16,
                ):
                    # Proper diffusion training with noise
                    noise = torch.randn_like(clean)

                    # Sample random noise levels (EDM schedule)
                    sigma = torch.exp(
                        torch.randn(clean.shape[0], device=training_manager.device)
                        * 1.2
                        - 1.2
                    )
                    sigma = sigma.view(-1, 1, 1, 1)

                    # Add noise to clean images
                    noisy = clean + sigma * noise

                    # Predict the noise (v-parameterization for EDM)
                    sigma_flat = sigma.squeeze()
                    predicted = model(noisy, sigma_flat, condition=condition)

                    # Compute v-parameterization target
                    c_skip = 1 / (sigma**2 + 1)
                    c_out = sigma / (sigma**2 + 1).sqrt()
                    target = (clean - c_skip * noisy) / c_out

                    # MSE loss in v-space
                    loss = F.mse_loss(predicted, target)

                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # Learning rate scheduling
                current_lr = optimizer.param_groups[0]["lr"]
                if warmup_scheduler is not None and step < config.get(
                    "warmup_steps", 0
                ):
                    warmup_scheduler.step()
                    if step % 1000 == 0:  # Log LR changes during warmup
                        new_lr = optimizer.param_groups[0]["lr"]
                        logger.info(f"  Warmup LR: {current_lr:.2e} → {new_lr:.2e}")
                elif scheduler is not None:
                    scheduler.step()
                    if step % 5000 == 0:  # Log LR changes during main training
                        new_lr = optimizer.param_groups[0]["lr"]
                        logger.info(f"  Scheduled LR: {current_lr:.2e} → {new_lr:.2e}")

                # Update EMA model
                if ema_model is not None:
                    with torch.no_grad():
                        for ema_param, param in zip(
                            ema_model.parameters(), model.parameters()
                        ):
                            ema_param.data.mul_(ema_decay).add_(
                                param.data, alpha=1 - ema_decay
                            )

                # Logging
                if step % 100 == 0:
                    logger.info(f"Step {step:,}: Loss = {loss.item():.6f}")
                    
                    # Periodic memory cleanup (only if severe fragmentation, skip early steps)
                    if step > 1000 and step % 5000 == 0 and torch.cuda.is_available():  # Skip first 1000 steps
                        allocated_gb = torch.cuda.memory_allocated() / 1e9
                        reserved_gb = torch.cuda.memory_reserved() / 1e9
                        fragmentation_gb = reserved_gb - allocated_gb
                        
                        # Only clear if fragmentation is severe (>20GB) AND we're using too much memory
                        if fragmentation_gb > 20.0 and reserved_gb > 70.0:  # Much more conservative
                            torch.cuda.empty_cache()
                            logger.info(f"  🧹 Cleared cache: Allocated: {allocated_gb:.1f}GB, Reserved: {reserved_gb:.1f}GB, Fragmentation: {fragmentation_gb:.1f}GB")
                        elif step % 10000 == 0:  # Just log memory status occasionally
                            logger.info(f"  📊 Memory: Allocated: {allocated_gb:.1f}GB, Reserved: {reserved_gb:.1f}GB, Fragmentation: {fragmentation_gb:.1f}GB")

                # Validation every N steps
                if (
                    step % config["val_frequency"] == 0
                    and step > 0
                    and val_loader is not None
                ):
                    logger.info(f"🔍 Running validation at step {step:,}...")
                    val_loss = training_manager.validate_model(
                        model, val_loader, training_manager.device, step
                    )
                    logger.info(f"   Validation Loss: {val_loss:.6f}")

                    # Track best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        training_manager.save_checkpoint(
                            model,
                            optimizer,
                            step,
                            config,
                            val_loss=val_loss,
                            is_best=True,
                            ema_model=ema_model,
                            scheduler=scheduler,
                            warmup_scheduler=warmup_scheduler,
                        )
                        logger.info(
                            f"   🏆 New best model! Previous: {best_val_loss:.6f} → Current: {val_loss:.6f}"
                        )
                    else:
                        logger.info(f"   No improvement (best: {best_val_loss:.6f})")

                # Regular checkpointing every N steps
                if step % config["save_frequency_steps"] == 0 and step > 0:
                    training_manager.save_checkpoint(
                        model,
                        optimizer,
                        step,
                        config,
                        loss.item(),
                        ema_model=ema_model,
                        scheduler=scheduler,
                        warmup_scheduler=warmup_scheduler,
                    )

                # Phase-based checkpointing for unified training
                if step % config.get("phase_save_frequency", 50000) == 0 and step > 0:
                    training_manager.save_phase_checkpoint(model, step, config)

                step += 1

        logger.info("✅ Training completed!")

        # Save final model
        final_path = training_manager.output_dir / "final_model.pth"
        final_checkpoint = {
            "model_state_dict": model.state_dict(),
            "step": step,
            "config": config,
        }

        # Add EMA model to final checkpoint
        if ema_model is not None:
            final_checkpoint["ema_model_state_dict"] = ema_model.state_dict()

        torch.save(final_checkpoint, final_path)

        logger.info(f"Final model saved: {final_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

        # Save interrupted checkpoint
        interrupt_path = training_manager.output_dir / "interrupted_checkpoint.pth"
        interrupt_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": config,
        }

        # Add scheduler states to interrupted checkpoint
        if scheduler is not None:
            interrupt_checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if warmup_scheduler is not None:
            interrupt_checkpoint[
                "warmup_scheduler_state_dict"
            ] = warmup_scheduler.state_dict()

        # Add EMA model to interrupted checkpoint
        if ema_model is not None:
            interrupt_checkpoint["ema_model_state_dict"] = ema_model.state_dict()

        torch.save(interrupt_checkpoint, interrupt_path)

        logger.info(f"Interrupted checkpoint saved: {interrupt_path}")


if __name__ == "__main__":
    main()
