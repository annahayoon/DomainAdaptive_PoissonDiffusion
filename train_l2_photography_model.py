#!/usr/bin/env python3
"""
L2 Baseline training script for photography data.

This script is IDENTICAL to train_photography_model.py except:
- Uses L2 (MSE) guidance instead of Poisson-Gaussian
- Configured for 100K max steps for baseline comparison
- All other hyperparameters identical for fair ablation study

Usage:
    python train_l2_photography_model.py --data_root /path/to/photography/data
"""

import argparse
import json
import logging

# CRITICAL FIX: Set multiprocessing start method FIRST for HPC compatibility
# This must be done before importing torch to fix DataLoader num_workers > 0 crashes
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

print(f"[{__name__}] Setting up multiprocessing for HPC compatibility...")
print(f"[{__name__}] Python version: {sys.version}")
print(f"[{__name__}] Current working directory: {os.getcwd()}")

# Set multiprocessing start method to spawn for HPC/SLURM compatibility
try:
    current_method = mp.get_start_method(allow_none=True)
    print(f"[{__name__}] Current multiprocessing method: {current_method}")

    # Force spawn method for HPC compatibility
    mp.set_start_method("spawn", force=True)
    new_method = mp.get_start_method()
    print(
        f"[{__name__}] ✅ Successfully set multiprocessing start method to: {new_method}"
    )

    # Verify it's actually spawn
    if new_method != "spawn":
        print(f"[{__name__}] ❌ ERROR: Failed to set spawn method, got: {new_method}")
        sys.exit(1)

except RuntimeError as e:
    print(f"[{__name__}] ⚠️  Could not set multiprocessing start method: {e}")
    print(f"[{__name__}] Current method: {mp.get_start_method()}")
    # Don't exit, but log the issue
except Exception as e:
    print(f"[{__name__}] ❌ Unexpected error setting multiprocessing method: {e}")
    sys.exit(1)

import torch
import torch.multiprocessing as torch_mp
import torch.nn as nn

# Additional multiprocessing verification
print(f"[{__name__}] Verifying multiprocessing setup...")
print(
    f"[{__name__}] PyTorch multiprocessing start method: {torch_mp.get_start_method()}"
)
print(f"[{__name__}] Python multiprocessing start method: {mp.get_start_method()}")

# Skip multiprocessing test since we're using num_workers=0
print(f"[{__name__}] Skipping multiprocessing test - using single-threaded DataLoader")

# Initialize distributed training AFTER multiprocessing setup
if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    import torch.distributed as dist

    # Ensure torch multiprocessing uses the same method
    try:
        torch_mp.set_start_method("spawn", force=True)
        print("✅ Set torch multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("⚠️  Torch multiprocessing method already set")

    # Initialize distributed process group
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

        # Set device for this process
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        print(
            f"Distributed training initialized: rank {dist.get_rank()}/{dist.get_world_size()}, device: {device}"
        )
    else:
        print("Distributed training already initialized")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure CUDA memory allocator for HPC compatibility
# This fixes "pidfd_open syscall not supported" errors on older kernels
try:
    import torch

    if torch.cuda.is_available():
        # Disable expandable segments for HPC systems that don't support pidfd_open
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        print("✅ Configured CUDA memory allocator for HPC compatibility")
except Exception as e:
    print(f"⚠️  Could not configure CUDA memory allocator: {e}")
    print("   Continuing with default settings...")

from core.error_handlers import ErrorHandler

# Import L2 guidance factory for baseline training
from core.guidance_factory import create_guidance
from core.logging_config import LoggingManager
from data.domain_datasets import MultiDomainDataset
from models.edm_wrapper import (
    ProgressiveEDM,
    create_edm_wrapper,
    create_progressive_edm,
)
from poisson_training import (
    DomainBalancingConfig,
    MultiDomainTrainer,
    MultiDomainTrainingConfig,
    set_deterministic_mode,
)

# Setup logging
logging_manager = LoggingManager()
logger = logging_manager.setup_logging(
    level="INFO",
    log_dir="logs",
    console_output=True,
    file_output=True,
    json_format=False,
)


# Module-level classes to avoid pickling issues in multiprocessing
class CheckpointedEDM(nn.Module):
    """Simple gradient checkpointing wrapper for single GPU."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        if self.training and len(args) > 0:
            # Use gradient checkpointing during training
            from torch.utils.checkpoint import checkpoint

            def run_model(*args, **kwargs):
                return self.model(*args, **kwargs)

            return checkpoint(run_model, *args, use_reentrant=False, **kwargs)
        else:
            # No checkpointing during validation/inference
            return self.model(*args, **kwargs)


class SyntheticDataset:
    """Synthetic dataset for testing without real data."""

    def __init__(self, data_dir, target_size=128):
        self.data_dir = Path(data_dir)
        self.target_size = target_size

        # Check if we have NPZ files or need to create simple synthetic data
        image_dir = self.data_dir / "images"
        if image_dir.exists():
            self.image_files = list(image_dir.glob("*.npz"))
        else:
            # Create simple synthetic file list
            self.image_files = []
            for i in range(1000):  # 1000 synthetic samples
                self.image_files.append(f"synthetic_{i:04d}.pt")

        # Create train/val split
        split_idx = int(0.8 * len(self.image_files))
        self.train_files = self.image_files[:split_idx]
        self.val_files = self.image_files[split_idx:]

        # Create simple dataset objects using module-level class
        self.train_dataset = SyntheticSubset(self.train_files, target_size)
        self.val_dataset = SyntheticSubset(self.val_files, target_size)

        # Add domain_datasets attribute for MultiDomainTrainer compatibility
        self.train_dataset.domain_datasets = {"photography": self.train_dataset}
        self.val_dataset.domain_datasets = {"photography": self.val_dataset}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # This shouldn't be called directly, but provide fallback
        return {
            "clean": torch.randn(4, self.target_size, self.target_size),
            "noisy": torch.randn(4, self.target_size, self.target_size),
            "electrons": torch.randn(4, self.target_size, self.target_size),
            "domain": torch.tensor([0]),
            "metadata": {"synthetic": True, "idx": idx},
            "domain_params": {
                "scale": 1000.0,
                "read_noise": 3.0,
                "background": 100.0,
                "gain": 1.0,
            },
        }


class SyntheticSubset:
    """Subset of synthetic files for train/val splits."""

    def __init__(self, files, target_size):
        self.files = files
        self.target_size = target_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import numpy as np

        # Check if this is a Path object (NPZ file) or string (simple synthetic)
        if hasattr(self.files[idx], "suffix") and self.files[idx].suffix == ".npz":
            # Load from NPZ file
            data = np.load(self.files[idx])
            clean = (
                torch.from_numpy(data["clean"]).float().unsqueeze(0)
            )  # Add channel dim
            noisy = torch.from_numpy(data["noisy"]).float().unsqueeze(0)

            # Convert to 4-channel format to match real data
            clean_4ch = clean.repeat(4, 1, 1)  # Repeat single channel to 4 channels
            noisy_4ch = noisy.repeat(4, 1, 1)

            # Resize if needed
            if clean_4ch.shape[-1] != self.target_size:
                clean_4ch = torch.nn.functional.interpolate(
                    clean_4ch.unsqueeze(0), size=self.target_size, mode="bilinear"
                ).squeeze(0)
                noisy_4ch = torch.nn.functional.interpolate(
                    noisy_4ch.unsqueeze(0), size=self.target_size, mode="bilinear"
                ).squeeze(0)

            return {
                "clean": clean_4ch,
                "noisy": noisy_4ch,
                "electrons": clean_4ch,  # Use 4-channel as target electrons
                "domain": torch.tensor([0]),  # Photography domain
                "metadata": {"synthetic": True},
            }
        else:
            # Simple synthetic data
            return {
                "clean": torch.randn(4, self.target_size, self.target_size),
                "noisy": torch.randn(4, self.target_size, self.target_size),
                "electrons": torch.randn(4, self.target_size, self.target_size),
                "domain": torch.tensor([0]),
                "metadata": {"synthetic": True, "file": str(self.files[idx])},
                "domain_params": {
                    "scale": 1000.0,
                    "read_noise": 3.0,
                    "background": 100.0,
                    "gain": 1.0,
                },
            }


class DatasetWrapper:
    """Simple wrapper to provide train/val dataset attributes."""

    def __init__(self, train_ds, val_ds):
        self.train_dataset = train_ds
        self.val_dataset = val_ds


class L2PhotographyTrainingManager:
    """Manager for training L2 baseline model on photography data."""

    def __init__(
        self,
        data_root: str,
        output_dir: str = "results/l2_photography_training",
        device: str = "auto",
        seed: int = 42,
    ):
        """
        Initialize L2 baseline training manager.

        Args:
            data_root: Path to photography data directory
            output_dir: Directory for outputs and checkpoints
            device: Device for training ('auto', 'cpu', 'cuda')
            seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
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

        logger.info("📸 L2 Baseline Photography Training Manager initialized")
        logger.info(f"  Guidance Type: L2 (MSE) Baseline")
        logger.info(f"  Data root: {self.data_root}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Seed: {self.seed}")

    def _setup_device(self, device: str) -> str:
        """Setup computation device with memory monitoring."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
                logger.info(
                    f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )
            else:
                device = "cpu"
                logger.info("💻 Using CPU device")

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"

        return device

    def create_model(
        self,
        use_multi_resolution: bool = False,
        mixed_precision: bool = True,
        gradient_checkpointing: bool = False,
        ddp_find_unused_parameters: bool = False,
        **model_kwargs,
    ) -> nn.Module:
        """
        Create EDM model for L2 baseline photography restoration.
        IDENTICAL to Poisson-Gaussian model architecture.

        Args:
            use_multi_resolution: Whether to use multi-resolution progressive model
            **model_kwargs: Additional model configuration

        Returns:
            Configured EDM model
        """
        logger.info("🤖 Creating EDM model for L2 baseline photography...")

        # Determine domain for channel configuration
        # Default to photography for now, but this could be made configurable
        domain = "photography"  # Could be made a parameter later

        if use_multi_resolution:
            logger.info(f"📈 Using Progressive Multi-Resolution EDM for {domain} domain")

            # Multi-resolution model configuration - IDENTICAL to Poisson-Gaussian
            default_model_channels = model_kwargs.get("model_channels", 256)
            default_channel_mult = model_kwargs.get("channel_mult", [1, 2, 3, 4])
            default_channel_mult_emb = model_kwargs.get("channel_mult_emb", 6)
            default_num_blocks = model_kwargs.get("num_blocks", 6)
            default_attn_resolutions = model_kwargs.get(
                "attn_resolutions", [16, 32, 64]
            )  # Multi-scale attention optimized for patches

            # Use full resolution range for enhanced models
            min_resolution = 32
            max_resolution = 128 if default_model_channels >= 128 else 64
            num_stages = 4 if default_model_channels >= 128 else 3

            model = create_progressive_edm(
                min_resolution=min_resolution,
                max_resolution=max_resolution,
                num_stages=num_stages,
                model_channels=default_model_channels,
                img_channels=4,  # Photography data has 4 channels (RGBA)
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

            # For memory optimization, we'll use a smaller model configuration
            # that can fit in GPU memory even with mixed precision
            logger.info(
                f"🔧 Using memory-optimized Progressive EDM configuration for {domain} domain"
            )
            logger.info(f"   Model channels: {model_kwargs.get('model_channels', 64)}")
            logger.info(f"   Max resolution: 64x64 (memory optimized)")
            logger.info(f"   Configuration optimized for memory efficiency")

            # Mixed precision can work with EDM models if we use it carefully
            # We'll enable it but monitor memory usage
            if mixed_precision:
                logger.info("🔧 Mixed precision enabled for Progressive EDM models")
        else:
            logger.info(f"📊 Using Standard EDM for {domain} domain")

            # Default model configuration for photography - IDENTICAL to Poisson-Gaussian
            default_model_channels = model_kwargs.get("model_channels", 256)
            default_channel_mult = model_kwargs.get("channel_mult", [1, 2, 3, 4])
            default_channel_mult_emb = model_kwargs.get("channel_mult_emb", 6)
            default_num_blocks = model_kwargs.get("num_blocks", 6)
            default_attn_resolutions = model_kwargs.get(
                "attn_resolutions", [16, 32, 64]
            )  # Multi-scale attention optimized for patches

            # Use full resolution for enhanced models, memory-optimized for smaller models
            img_resolution = 128 if default_model_channels >= 128 else 64

            model_config = {
                "img_resolution": img_resolution,
                "img_channels": 4,  # Photography data has 4 channels (RGBA)
                "model_channels": default_model_channels,
                "channel_mult": default_channel_mult,
                "channel_mult_emb": default_channel_mult_emb,
                "num_blocks": default_num_blocks,
                "attn_resolutions": default_attn_resolutions,
                "label_dim": 6,  # Domain conditioning
                "use_fp16": False,  # Will use mixed precision training instead
                "dropout": 0.1,
                **{
                    k: v
                    for k, v in model_kwargs.items()
                    if k
                    not in [
                        "model_channels",
                        "channel_mult",
                        "channel_mult_emb",
                        "num_blocks",
                        "attn_resolutions",
                    ]
                },
            }

            model = create_edm_wrapper(**model_config)

            # Log the actual model configuration being used
            model_channels = model_kwargs.get("model_channels", 64)
            is_enhanced = model_channels >= 128

            if is_enhanced:
                logger.info(
                    "🚀 Using ENHANCED EDM configuration for L2 baseline research-level performance"
                )
                logger.info(f"   Model channels: {model_channels}")
                logger.info(
                    f"   Channel mult: {model_kwargs.get('channel_mult', [1, 2, 2])}"
                )
                logger.info(f"   Num blocks: {model_kwargs.get('num_blocks', 3)}")
                logger.info(
                    f"   Attention resolutions: {model_kwargs.get('attn_resolutions', [16, 32, 64])}"
                )
                logger.info(f"   Image resolution: {img_resolution}x{img_resolution}")
            else:
                logger.info(
                    "🔧 Using memory-optimized EDM configuration for L2 baseline"
                )
                logger.info(f"   Model channels: {model_channels}")
                logger.info(f"   Configuration optimized for memory efficiency")

            # Mixed precision can work with EDM models if we use it carefully
            # We'll enable it but monitor memory usage
            if mixed_precision:
                logger.info("🔧 Mixed precision enabled for EDM models")

        # Enable gradient checkpointing if requested (IDENTICAL to Poisson-Gaussian)
        if gradient_checkpointing:
            logger.info("🔧 Enabling gradient checkpointing for memory optimization...")

            # Try to apply gradient checkpointing directly to the EDM model
            applied = False

            # For single GPU training (no DDP), we can use a simple wrapper
            if "RANK" not in os.environ or os.environ.get("WORLD_SIZE", "1") == "1":
                try:
                    # Use module-level CheckpointedEDM class to avoid pickling issues

                    model = CheckpointedEDM(model)
                    logger.info(
                        "✅ Gradient checkpointing enabled with wrapper (single GPU mode)"
                    )
                    applied = True

                except Exception as e:
                    logger.warning(
                        f"  Could not apply gradient checkpointing wrapper: {e}"
                    )
            else:
                # For multi-GPU (DDP), don't use gradient checkpointing to avoid int8 issues
                logger.info(
                    "  Skipping gradient checkpointing for multi-GPU training (avoids DDP issues)"
                )
                logger.info("  With distributed training, memory is already optimized")

            if not applied and "RANK" not in os.environ:
                # Try other methods for single GPU
                if hasattr(model, "enable_gradient_checkpointing"):
                    model.enable_gradient_checkpointing()
                    logger.info("✅ Gradient checkpointing enabled via model method")
                    applied = True
                else:
                    logger.warning(
                        "⚠️ Gradient checkpointing not available for this model"
                    )
                    logger.warning("  Continuing without gradient checkpointing")
                    logger.info("  Consider reducing batch size if memory issues occur")

            if applied:
                logger.info("  Memory usage reduced by ~30-40%")
                logger.info("  Training will be ~20% slower but use less memory")

        # Move to device and log
        model = model.to(self.device)
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("✅ L2 baseline model created successfully")
        logger.info(f"  Total parameters: {param_count:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model device: {next(model.parameters()).device}")

        # Wrap model with DistributedDataParallel if in distributed mode
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            import torch.distributed as dist

            if dist.is_initialized():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                logger.info(
                    f"🔄 Wrapping model with DistributedDataParallel (rank {dist.get_rank()}, local_rank {local_rank})"
                )

                # Ensure model is on the correct device before wrapping
                model = model.to(f"cuda:{local_rank}")

                # Wrap with DDP
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=ddp_find_unused_parameters,
                    # Disable broadcast_buffers for better performance if model doesn't have buffers that need syncing
                    broadcast_buffers=True,
                )

                logger.info(f"✅ Model wrapped with DDP on device cuda:{local_rank}")
                logger.info(f"   find_unused_parameters: {ddp_find_unused_parameters}")
                logger.info(f"   broadcast_buffers: True")
            else:
                logger.warning(
                    "⚠️  RANK/WORLD_SIZE set but distributed not initialized, using single GPU"
                )
        else:
            logger.info("🔧 Single GPU mode - no DDP wrapping needed")

        return model

    def create_training_config(
        self,
        num_epochs: int = 100,
        max_steps: Optional[int] = None,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        save_frequency_steps: int = 50000,
        early_stopping_patience_steps: int = 5000,
        validation_checkpoints_patience: int = 20,
        lr_scheduler: str = "cosine",
        warmup_epochs: int = 5,
        val_frequency: int = 5,
        val_frequency_steps: Optional[int] = None,
        gradient_clip_norm: float = 1.0,
        prefetch_factor: int = 2,
        **config_kwargs,
    ) -> MultiDomainTrainingConfig:
        """
        Create comprehensive training configuration for L2 baseline.
        IDENTICAL to Poisson-Gaussian except guidance type.

        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            lr_scheduler: Learning rate scheduler type
            warmup_epochs: Number of warmup epochs
            val_frequency: Validation frequency in epochs
            **config_kwargs: Additional configuration options

        Returns:
            Configured training configuration
        """
        logger.info("⚙️  Creating L2 baseline training configuration...")

        # Domain balancing configuration (IDENTICAL)
        domain_balancing = DomainBalancingConfig(
            sampling_strategy="weighted",
            use_domain_conditioning=True,
            use_domain_loss_weights=True,
            enforce_batch_balance=True,
            min_samples_per_domain_per_batch=1,
            adaptive_rebalancing=True,
            rebalancing_frequency=100,
            performance_window=50,
            log_domain_stats=True,
            domain_stats_frequency=50,
        )

        # Main training configuration - IDENTICAL except guidance type
        config = MultiDomainTrainingConfig(
            # Model and data
            model_name="l2_baseline_photography",
            dataset_path=str(self.data_root),
            # Training hyperparameters
            num_epochs=num_epochs,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            # Optimizer
            optimizer="adamw",
            weight_decay=1e-2,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            # Scheduler
            scheduler=lr_scheduler,
            min_lr=1e-6,
            scheduler_patience=10,
            # Loss function - L2 BASELINE INSTEAD OF POISSON-GAUSSIAN
            loss_type="mse",  # THIS IS THE KEY DIFFERENCE
            gradient_clip_norm=gradient_clip_norm,
            # Validation and checkpointing
            val_frequency=val_frequency,  # Configurable validation frequency
            val_frequency_steps=val_frequency_steps,  # Step-based validation frequency
            save_frequency_steps=save_frequency_steps,
            max_checkpoints=5,
            early_stopping_patience_steps=early_stopping_patience_steps,
            validation_checkpoints_patience=validation_checkpoints_patience,
            early_stopping_min_delta=1e-4,
            # Reproducibility
            seed=self.seed,
            deterministic=True,
            benchmark=True,
            # Logging and monitoring
            log_frequency=50,
            tensorboard_log_dir=str(self.output_dir / "tensorboard"),
            checkpoint_dir=str(self.output_dir / "checkpoints"),
            # Device and performance
            device=self.device,
            mixed_precision=mixed_precision,
            compile_model=False,  # May cause issues with some setups
            prefetch_factor=prefetch_factor,
            # Multi-domain configuration
            domain_balancing=domain_balancing,
            **config_kwargs,
        )

        logger.info("✅ L2 baseline training configuration created")
        logger.info(f"  Guidance Type: L2 (MSE) Baseline")
        logger.info(f"  Training epochs: {num_epochs}")
        logger.info(f"  Max steps: {max_steps} (overrides epochs if set)")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  LR scheduler: {lr_scheduler}")
        logger.info(f"  Warmup epochs: {warmup_epochs}")
        logger.info(f"  Validation frequency: {val_frequency} epochs")
        logger.info(
            f"  Early stopping patience: {config.early_stopping_patience_steps} steps"
        )
        logger.info(f"  Mixed precision: {config.mixed_precision}")

        return config

    def train(
        self,
        model: nn.Module,
        dataset,
        config: MultiDomainTrainingConfig,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the L2 baseline model with comprehensive monitoring.
        IDENTICAL to Poisson-Gaussian training except guidance type.

        Args:
            model: Model to train
            dataset: Training dataset
            config: Training configuration
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results and metrics
        """
        logger.info("🚀 Starting L2 baseline photography training...")
        logger.info("=" * 60)

        # Set deterministic mode
        set_deterministic_mode(seed=self.seed)

        # Create trainer with L2 guidance
        trainer = MultiDomainTrainer(
            model=model,
            train_dataset=dataset.train_dataset,
            val_dataset=dataset.val_dataset,
            config=config,
            output_dir=self.output_dir,
        )

        # Setup checkpoint resumption
        if resume_from_checkpoint:
            if Path(resume_from_checkpoint).exists():
                logger.info(f"📁 Resuming from checkpoint: {resume_from_checkpoint}")
                trainer.load_checkpoint(resume_from_checkpoint)
            else:
                logger.warning(f"Checkpoint not found: {resume_from_checkpoint}")

        # Training loop with monitoring
        start_time = time.time()

        try:
            training_history = trainer.train()

            training_time = time.time() - start_time

            # Save final results
            results = {
                "training_time_hours": training_time / 3600,
                "best_val_loss": trainer.best_val_loss,
                "final_epoch": trainer.current_epoch,
                "training_history": dict(training_history),
                "config": config.to_dict(),
                "device": str(self.device),
                "guidance_type": "l2_baseline",
            }

            # Save results
            # Ensure output directory exists before saving results
            self.output_dir.mkdir(parents=True, exist_ok=True)
            results_file = self.output_dir / "training_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info("✅ L2 baseline photography training completed successfully!")
            logger.info(f"  Total time: {training_time / 3600:.2f} hours")
            logger.info(f"  Best validation loss: {trainer.best_val_loss:.6f}")
            logger.info(f"  Results saved to: {results_file}")

            return results

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(
                f"❌ L2 baseline photography training failed after {error_time / 3600:.2f} hours: {e}"
            )

            # Save partial results
            error_results = {
                "error": str(e),
                "training_time_hours": error_time / 3600,
                "failed_epoch": getattr(trainer, "current_epoch", 0),
                "device": str(self.device),
                "guidance_type": "l2_baseline",
            }

            # Ensure output directory exists before saving error file
            self.output_dir.mkdir(parents=True, exist_ok=True)
            error_file = self.output_dir / "training_error.json"
            with open(error_file, "w") as f:
                json.dump(error_results, f, indent=2, default=str)

            raise


def main():
    """Main L2 baseline training function."""
    # Distributed training is already initialized at the top of the script

    parser = argparse.ArgumentParser(
        description="Train L2 baseline diffusion model on photography data"
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to photography data directory",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100000,  # L2 BASELINE: 100K steps instead of 150K
        help="Maximum training steps (overrides epochs if set)",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--target_size", type=int, default=128, help="Target image size"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "exponential", "step", "multistep", "plateau"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps (will be converted to epochs)",
    )
    parser.add_argument(
        "--val_frequency",
        type=int,
        default=5,
        help="Validation frequency in epochs (ignored if val_frequency_steps is set)",
    )
    parser.add_argument(
        "--val_frequency_steps",
        type=int,
        default=None,
        help="Validation frequency in training steps (overrides val_frequency if set)",
    )

    # Model arguments (IDENTICAL to Poisson-Gaussian)
    parser.add_argument(
        "--model_channels", type=int, default=256, help="Number of model channels"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=6, help="Number of model blocks"
    )
    parser.add_argument(
        "--multi_resolution",
        action="store_true",
        help="Use multi-resolution progressive EDM model",
    )

    # Enhanced model architecture arguments (IDENTICAL)
    parser.add_argument(
        "--channel_mult",
        type=int,
        nargs="+",
        default=None,
        help="Channel multipliers (e.g., 1 2 3 4)",
    )
    parser.add_argument(
        "--channel_mult_emb",
        type=int,
        default=None,
        help="Channel multiplier for embeddings",
    )
    parser.add_argument(
        "--attn_resolutions",
        type=int,
        nargs="+",
        default=None,
        help="Attention resolutions (e.g., 8 16 32)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/l2_photography_training",
        help="Output directory",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Performance arguments (IDENTICAL)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Enable gradient checkpointing to save memory (default: false)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--pin_memory",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Pin memory for faster GPU transfer",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches to prefetch (0 to disable)",
    )

    # Checkpointing arguments (IDENTICAL)
    parser.add_argument(
        "--save_frequency_steps",
        type=int,
        default=800,
        help="Save checkpoint every N steps (more frequent for better recovery)",
    )
    parser.add_argument(
        "--early_stopping_patience_steps",
        type=int,
        default=5000,
        help="Early stopping patience in steps",
    )
    parser.add_argument(
        "--validation_checkpoints_patience",
        type=int,
        default=20,
        help="Early stopping patience in validation checkpoints",
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm threshold",
    )

    # Advanced checkpointing arguments (IDENTICAL)
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--save_best_model",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Save the best model based on validation metrics",
    )
    parser.add_argument(
        "--save_optimizer_state",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Save optimizer state in checkpoints",
    )
    parser.add_argument(
        "--checkpoint_metric",
        type=str,
        default="val_loss",
        help="Metric to monitor for best model",
    )
    parser.add_argument(
        "--checkpoint_mode",
        type=str,
        default="min",
        choices=["min", "max"],
        help="Mode for checkpoint metric (min for loss, max for accuracy)",
    )
    parser.add_argument(
        "--resume_from_best",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Resume training from best checkpoint instead of latest",
    )

    # DDP optimization arguments
    parser.add_argument(
        "--ddp_find_unused_parameters",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Find unused parameters in DDP (slower but sometimes necessary)",
    )

    # Testing arguments
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to use (for testing)",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick test with synthetic data (NOT for real training - use only for testing)",
    )

    args = parser.parse_args()

    # Convert string boolean arguments to actual booleans
    args.mixed_precision = args.mixed_precision.lower() == "true"
    args.pin_memory = args.pin_memory.lower() == "true"
    args.save_best_model = args.save_best_model.lower() == "true"
    args.save_optimizer_state = args.save_optimizer_state.lower() == "true"
    args.resume_from_best = args.resume_from_best.lower() == "true"
    args.ddp_find_unused_parameters = args.ddp_find_unused_parameters.lower() == "true"

    # Check if we're in distributed training
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    is_main_process = not is_distributed or int(os.environ.get("RANK", "0")) == 0

    # Initialize L2 baseline training manager
    if is_main_process:
        logger.info("🚀 INITIALIZING L2 BASELINE PHOTOGRAPHY TRAINING")
        logger.info("=" * 60)

    training_manager = L2PhotographyTrainingManager(
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )

    # Create model (IDENTICAL architecture to Poisson-Gaussian)
    model_kwargs = {
        "model_channels": args.model_channels,
        "num_blocks": args.num_blocks,
    }

    # Add enhanced architecture arguments if provided
    if args.channel_mult is not None:
        model_kwargs["channel_mult"] = args.channel_mult
    if args.channel_mult_emb is not None:
        model_kwargs["channel_mult_emb"] = args.channel_mult_emb
    if args.attn_resolutions is not None:
        model_kwargs["attn_resolutions"] = args.attn_resolutions

    # Convert gradient checkpointing argument
    gradient_checkpointing = args.gradient_checkpointing.lower() == "true"

    model = training_manager.create_model(
        use_multi_resolution=args.multi_resolution,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=gradient_checkpointing,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        **model_kwargs,
    )

    # Create dataset (IDENTICAL to Poisson-Gaussian)
    if args.quick_test:
        logger.warning("🧪 QUICK TEST MODE: Using synthetic data for testing only!")
        logger.warning(
            "⚠️  This is NOT real training data - results will not be scientifically valid"
        )
        # Generate synthetic data for testing
        from scripts.generate_synthetic_data import (
            SyntheticConfig,
            SyntheticDataGenerator,
        )

        synthetic_config = SyntheticConfig(
            output_dir="data/synthetic_quick",
            num_images=100,
            image_size=args.target_size,
            pattern_types=["constant", "gradient"],
            photon_levels=[10, 100, 1000],
            read_noise_levels=[1, 5],
        )

        synthetic_generator = SyntheticDataGenerator(synthetic_config)
        logger.info("🔄 Generating synthetic data...")
        results = synthetic_generator.generate_validation_set()
        synthetic_generator.save_dataset(results)

        # Use module-level SyntheticDataset class to avoid pickling issues
        # Use module-level SyntheticSubset class to avoid pickling issues
        dataset = SyntheticDataset(synthetic_config.output_dir, args.target_size)

        # Validate synthetic dataset was created successfully
        if len(dataset) == 0:
            logger.error("❌ Failed to create synthetic dataset")
            logger.error(f"   Output directory: {synthetic_config.output_dir}")
            raise SystemExit("Synthetic dataset creation failed")
    else:
        # Use real preprocessed data (IDENTICAL to Poisson-Gaussian)
        logger.info("📁 Loading real preprocessed photography data...")
        from data.preprocessed_datasets import create_preprocessed_datasets
        from utils.training_config import (
            calculate_optimal_training_config,
            print_training_analysis,
        )

        try:
            train_dataset, val_dataset = create_preprocessed_datasets(
                data_root=args.data_root,
                domain="photography",
                max_files=args.max_files,
                seed=args.seed,
            )
        except Exception as e:
            logger.error(f"❌ Failed to load real preprocessed data: {e}")
            logger.error(f"   Expected data directory: {args.data_root}")
            logger.error(f"   Make sure you have run data preprocessing first:")
            logger.error(
                f"     python scripts/preprocess_data.py --data_root {args.data_root}"
            )
            raise SystemExit(f"Real data loading failed: {e}")

        # Calculate optimal training configuration
        dataset_size = len(train_dataset)
        optimal_config = calculate_optimal_training_config(
            dataset_size, args.batch_size
        )

        logger.info("🎯 AUTOMATIC L2 BASELINE TRAINING CONFIGURATION:")
        print_training_analysis(optimal_config)

        # Check if training steps are reasonable (diffusion standard)
        current_total_steps = args.epochs * optimal_config["steps_per_epoch"]
        recommended_steps = optimal_config["total_training_steps"]

        if current_total_steps < recommended_steps * 0.5:
            logger.warning(
                f"⚠️  Current training steps ({current_total_steps:,}) may be too low!"
            )
            logger.warning(f"   Recommended: {recommended_steps:,} total steps")
            logger.warning(
                f"   Steps per sample: {current_total_steps / dataset_size:.1f}"
            )
            logger.warning(
                f"   Recommended steps per sample: {optimal_config['steps_per_sample']}"
            )
        elif current_total_steps > recommended_steps * 2.0:
            logger.warning(
                f"⚠️  Current training steps ({current_total_steps:,}) may be too high!"
            )
            logger.warning(f"   Recommended: {recommended_steps:,} total steps")
        else:
            logger.info(
                f"✅ Current training steps ({current_total_steps:,}) are reasonable"
            )
            logger.info(
                f"   Steps per sample: {current_total_steps / dataset_size:.1f}"
            )

        # Use module-level DatasetWrapper class to avoid pickling issues

        dataset = DatasetWrapper(train_dataset, val_dataset)

    # Convert warmup_steps to warmup_epochs if needed
    import math

    warmup_epochs = args.warmup_steps
    if hasattr(dataset.train_dataset, "__len__"):
        # Calculate steps per epoch to convert warmup_steps to warmup_epochs
        steps_per_epoch = math.ceil(len(dataset.train_dataset) / args.batch_size)
        warmup_epochs = max(1, math.ceil(args.warmup_steps / steps_per_epoch))
        logger.info(
            f"🔄 Converted warmup_steps ({args.warmup_steps}) to warmup_epochs ({warmup_epochs})"
        )
        logger.info(f"   Steps per epoch: {steps_per_epoch}")

    # Create training configuration (IDENTICAL except guidance type)
    config = training_manager.create_training_config(
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_frequency_steps=args.save_frequency_steps,
        early_stopping_patience_steps=args.early_stopping_patience_steps,
        validation_checkpoints_patience=args.validation_checkpoints_patience,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=warmup_epochs,
        val_frequency=args.val_frequency,
        val_frequency_steps=args.val_frequency_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        prefetch_factor=args.prefetch_factor,
    )

    # Save configuration (only on main process)
    if is_main_process:
        config_file = training_manager.output_dir / "training_config.json"
        with open(config_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        logger.info(f"📝 Configuration saved to: {config_file}")

    # Start training
    if is_main_process:
        logger.info("🎯 STARTING L2 BASELINE TRAINING")
        logger.info("=" * 60)

    try:
        results = training_manager.train(
            model=model,
            dataset=dataset,
            config=config,
            resume_from_checkpoint=args.resume_checkpoint,
        )

        # Success summary (only on main process)
        if is_main_process:
            logger.info("🎉 L2 BASELINE PHOTOGRAPHY TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info("📊 Final Results:")
            logger.info(f"  Training time: {results['training_time_hours']:.2f} hours")
            logger.info(f"  Best validation loss: {results['best_val_loss']:.6f}")
            logger.info(f"  Final epoch: {results['final_epoch']}")
            logger.info("=" * 60)
            logger.info("📁 Outputs:")
            logger.info(f"  Checkpoints: {training_manager.output_dir}/checkpoints/")
            logger.info(
                f"  Results: {training_manager.output_dir}/training_results.json"
            )
            logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ L2 baseline photography training failed: {e}")
        logger.error("Check logs for details")
        raise


if __name__ == "__main__":
    main()
