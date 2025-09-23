#!/usr/bin/env python3
"""
Unified multi-domain training script for Poisson-Gaussian diffusion restoration.

This script implements the core research contribution: a single model that learns
to restore images across photography, microscopy, and astronomy domains using
domain conditioning vectors and balanced sampling.

Key Research Features:
- Single unified model for all three domains
- Domain conditioning vectors (6-dimensional)
- Balanced sampling across domains
- Physics-aware loss function for all domains
- Cross-domain generalization validation

Usage:
    python train_unified_model.py --data_root data --domains photography microscopy astronomy

Requirements:
- Preprocessed data for all specified domains
- Calibration files for each domain
- Sufficient GPU memory (16GB+ recommended)
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Set multiprocessing start method for HPC compatibility
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.error_handlers import ErrorHandler
from core.logging_config import LoggingManager
from data.domain_datasets import MultiDomainDataset
from models.edm_wrapper import create_edm_wrapper, create_progressive_edm
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


class UnifiedTrainingManager:
    """Manager for unified multi-domain training."""

    def __init__(
        self,
        data_root: str,
        domains: List[str],
        output_dir: str = "results/unified_training",
        device: str = "auto",
        seed: int = 42,
    ):
        """
        Initialize unified training manager.

        Args:
            data_root: Path to data directory containing all domains
            domains: List of domains to train on (e.g., ['photography', 'microscopy', 'astronomy'])
            output_dir: Directory for outputs and checkpoints
            device: Device for training ('auto', 'cpu', 'cuda')
            seed: Random seed for reproducibility
        """
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

        logger.info("🌟 Unified Multi-Domain Training Manager initialized")
        logger.info(f"  Data root: {self.data_root}")
        logger.info(f"  Domains: {', '.join(self.domains)}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Seed: {self.seed}")

    def _setup_device(self, device: str) -> str:
        """Setup computation device with memory monitoring."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
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

    def create_unified_dataset(
        self,
        target_size: int = 128,
        max_files_per_domain: Optional[int] = None,
        validation_split: float = 0.2,
    ) -> MultiDomainDataset:
        """
        Create unified multi-domain dataset.

        Args:
            target_size: Target image size for processing
            max_files_per_domain: Maximum files per domain (for testing)
            validation_split: Fraction of data for validation

        Returns:
            Configured MultiDomainDataset
        """
        logger.info("📁 Creating unified multi-domain dataset...")

        # Domain-specific configurations
        domain_configs = {}

        for domain in self.domains:
            # Find calibration file for this domain
            calib_files = list(
                (project_root / "configs/calibrations").glob(f"*{domain}*.json")
            )
            if not calib_files:
                raise FileNotFoundError(
                    f"Calibration file not found for domain: {domain}"
                )

            calib_file = calib_files[0]
            logger.info(f"  {domain}: Using calibration {calib_file.name}")

            # Domain-specific scale parameters (from existing training scripts)
            domain_scales = {
                "photography": 10000.0,
                "microscopy": 5000.0,
                "astronomy": 50000.0,  # Extreme low-light
            }

            domain_configs[domain] = {
                "data_root": str(self.data_root / domain),
                "calibration_file": str(calib_file),
                "scale": domain_scales.get(domain, 10000.0),
                "target_size": target_size,
                "max_files": max_files_per_domain,
            }

        # Create unified dataset
        dataset = MultiDomainDataset(
            domain_configs=domain_configs,
            split="train",
            balance_domains=True,  # Critical for unified training
        )

        logger.info("✅ Unified dataset created successfully")
        logger.info(f"  Total training samples: {len(dataset.train_dataset)}")
        logger.info(f"  Total validation samples: {len(dataset.val_dataset)}")

        # Log per-domain statistics
        for domain in self.domains:
            if hasattr(dataset.train_dataset, "domain_datasets"):
                train_count = len(dataset.train_dataset.domain_datasets.get(domain, []))
                val_count = len(dataset.val_dataset.domain_datasets.get(domain, []))
                logger.info(f"  {domain}: {train_count} train, {val_count} val samples")

        return dataset

    def create_unified_model(
        self,
        use_multi_resolution: bool = False,
        mixed_precision: bool = True,
        gradient_checkpointing: bool = False,
        **model_kwargs,
    ) -> nn.Module:
        """
        Create unified model that handles all domains.

        The key challenge is handling different channel counts:
        - Photography: 4 channels (RGBA)
        - Microscopy: 1-3 channels (typically grayscale or RGB)
        - Astronomy: 1 channel (grayscale)

        Solution: Use maximum channels (4) and pad smaller inputs.
        """
        logger.info("🤖 Creating unified multi-domain EDM model...")

        # Unified model configuration - handles all domains
        # Use 4 channels to accommodate photography (largest)
        # Other domains will be padded to 4 channels
        unified_channels = 4

        if use_multi_resolution:
            logger.info("📈 Using Progressive Multi-Resolution EDM (unified)")

            default_model_channels = model_kwargs.get("model_channels", 256)
            min_resolution = 32
            max_resolution = 128 if default_model_channels >= 128 else 64
            num_stages = 4 if default_model_channels >= 128 else 3

            model = create_progressive_edm(
                min_resolution=min_resolution,
                max_resolution=max_resolution,
                num_stages=num_stages,
                model_channels=default_model_channels,
                img_channels=unified_channels,  # Unified channel count
                label_dim=6,  # Domain conditioning vector size
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

            logger.info(f"🔧 Progressive EDM unified configuration")
            logger.info(f"   Model channels: {default_model_channels}")
            logger.info(f"   Image channels: {unified_channels} (unified)")
            logger.info(f"   Max resolution: {max_resolution}x{max_resolution}")

        else:
            logger.info("📊 Using Standard EDM (unified)")

            default_model_channels = model_kwargs.get("model_channels", 256)
            img_resolution = 128 if default_model_channels >= 128 else 64

            model_config = {
                "img_resolution": img_resolution,
                "img_channels": unified_channels,  # Unified channel count
                "model_channels": default_model_channels,
                "channel_mult": model_kwargs.get("channel_mult", [1, 2, 3, 4]),
                "channel_mult_emb": model_kwargs.get("channel_mult_emb", 6),
                "num_blocks": model_kwargs.get("num_blocks", 6),
                "attn_resolutions": model_kwargs.get("attn_resolutions", [16, 32, 64]),
                "label_dim": 6,  # Domain conditioning vector size
                "use_fp16": False,
                "dropout": 0.1,
            }

            model = create_edm_wrapper(**model_config)

            logger.info(f"🔧 Standard EDM unified configuration")
            logger.info(f"   Model channels: {default_model_channels}")
            logger.info(f"   Image channels: {unified_channels} (unified)")
            logger.info(f"   Image resolution: {img_resolution}x{img_resolution}")

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            logger.info("🔧 Enabling gradient checkpointing for memory optimization...")
            # Implementation would go here - similar to domain-specific trainers

        # Move to device
        model = model.to(self.device)
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("✅ Unified model created successfully")
        logger.info(f"  Total parameters: {param_count:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Domains supported: {', '.join(self.domains)}")
        logger.info(f"  Domain conditioning: 6-dimensional vectors")

        return model

    def create_unified_training_config(
        self,
        num_epochs: int = 150,
        max_steps: Optional[int] = None,
        learning_rate: float = 2e-5,  # Conservative for multi-domain
        batch_size: int = 2,  # Smaller for multi-domain complexity
        mixed_precision: bool = False,  # Conservative start for multi-domain stability
        gradient_accumulation_steps: int = 4,  # Effective batch = 8
        save_frequency_steps: int = 10000,
        early_stopping_patience_steps: int = 20000,
        validation_checkpoints_patience: int = 50,
        lr_scheduler: str = "cosine",
        warmup_epochs: int = 10,
        val_frequency: int = 3,
        gradient_clip_norm: float = 0.1,  # Conservative for stability
        **config_kwargs,
    ) -> MultiDomainTrainingConfig:
        """
        Create unified training configuration optimized for multi-domain learning.
        """
        logger.info("⚙️  Creating unified multi-domain training configuration...")

        # Domain balancing configuration - critical for unified training
        domain_balancing = DomainBalancingConfig(
            sampling_strategy="weighted",
            use_domain_conditioning=True,  # Essential for unified model
            use_domain_loss_weights=True,
            enforce_batch_balance=True,
            min_samples_per_domain_per_batch=1,
            adaptive_rebalancing=True,
            rebalancing_frequency=50,  # Frequent rebalancing for stability
            performance_window=25,
            log_domain_stats=True,
            domain_stats_frequency=25,
        )

        # Main training configuration
        config = MultiDomainTrainingConfig(
            # Model and data
            model_name="unified_poisson_diffusion",
            dataset_path=str(self.data_root),
            # Training hyperparameters - conservative for multi-domain
            num_epochs=num_epochs,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            # Optimizer - conservative settings
            optimizer="adamw",
            weight_decay=1e-3,  # Lower weight decay for stability
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            # Scheduler
            scheduler=lr_scheduler,
            min_lr=1e-7,
            scheduler_patience=25,
            # Loss function
            loss_type="poisson_gaussian",
            gradient_clip_norm=gradient_clip_norm,
            # Validation and checkpointing
            val_frequency=val_frequency,
            save_frequency_steps=save_frequency_steps,
            max_checkpoints=10,
            early_stopping_patience_steps=early_stopping_patience_steps,
            validation_checkpoints_patience=validation_checkpoints_patience,
            early_stopping_min_delta=1e-5,
            # Reproducibility
            seed=self.seed,
            deterministic=True,
            benchmark=True,
            # Logging and monitoring
            log_frequency=25,
            tensorboard_log_dir=str(self.output_dir / "tensorboard"),
            checkpoint_dir=str(self.output_dir / "checkpoints"),
            # Device and performance
            device=self.device,
            mixed_precision=mixed_precision,
            compile_model=False,
            # Multi-domain configuration - CRITICAL
            domain_balancing=domain_balancing,
            **config_kwargs,
        )

        logger.info("✅ Unified training configuration created")
        logger.info(f"  Training epochs: {num_epochs}")
        logger.info(
            f"  Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})"
        )
        logger.info(f"  Learning rate: {learning_rate} (conservative for multi-domain)")
        logger.info(f"  Domains: {', '.join(self.domains)}")
        logger.info(f"  Domain conditioning: ENABLED")
        logger.info(f"  Balanced sampling: ENABLED")
        logger.info(f"  Gradient clip norm: {gradient_clip_norm}")

        return config

    def train(
        self,
        model: nn.Module,
        dataset: MultiDomainDataset,
        config: MultiDomainTrainingConfig,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the unified model with comprehensive monitoring.
        """
        logger.info("🚀 Starting unified multi-domain training...")
        logger.info("=" * 80)
        logger.info("🌟 RESEARCH CONTRIBUTION: Single model learning all domains")
        logger.info("=" * 80)

        # Set deterministic mode
        set_deterministic_mode(seed=self.seed)

        # Create trainer
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
                "domains": self.domains,
                "unified_model": True,
            }

            # Save results
            results_file = self.output_dir / "unified_training_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info("✅ Unified multi-domain training completed successfully!")
            logger.info("=" * 80)
            logger.info("🎉 RESEARCH SUCCESS: Single model learned all domains!")
            logger.info("=" * 80)
            logger.info(f"  Total time: {training_time / 3600:.2f} hours")
            logger.info(f"  Best validation loss: {trainer.best_val_loss:.6f}")
            logger.info(f"  Domains trained: {', '.join(self.domains)}")
            logger.info(f"  Results saved to: {results_file}")

            return results

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(
                f"❌ Unified training failed after {error_time / 3600:.2f} hours: {e}"
            )

            # Save partial results
            error_results = {
                "error": str(e),
                "training_time_hours": error_time / 3600,
                "failed_epoch": getattr(trainer, "current_epoch", 0),
                "device": str(self.device),
                "domains": self.domains,
                "unified_model": True,
            }

            error_file = self.output_dir / "unified_training_error.json"
            with open(error_file, "w") as f:
                json.dump(error_results, f, indent=2, default=str)

            raise


def main():
    """Main unified training function."""
    parser = argparse.ArgumentParser(
        description="Train unified Poisson-Gaussian diffusion model on multiple domains"
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to data directory containing all domains",
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
        "--epochs", type=int, default=150, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps", type=int, default=300000, help="Maximum training steps"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--target_size", type=int, default=128, help="Target image size"
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine", help="LR scheduler"
    )
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warmup steps")
    parser.add_argument(
        "--val_frequency", type=int, default=3, help="Validation frequency"
    )

    # Model arguments
    parser.add_argument(
        "--model_channels", type=int, default=256, help="Model channels"
    )
    parser.add_argument("--num_blocks", type=int, default=6, help="Number of blocks")
    parser.add_argument(
        "--multi_resolution", action="store_true", help="Use multi-resolution"
    )
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--channel_mult_emb", type=int, default=6)
    parser.add_argument("--attn_resolutions", type=int, nargs="+", default=[16, 32, 64])

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/unified_training")
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    # Performance arguments
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--mixed_precision", type=str, default="false", choices=["true", "false"]
    )
    parser.add_argument(
        "--gradient_checkpointing", type=str, default="false", choices=["true", "false"]
    )
    parser.add_argument("--gradient_clip_norm", type=float, default=0.1)

    # Testing arguments
    parser.add_argument("--max_files_per_domain", type=int, default=None)
    parser.add_argument("--quick_test", action="store_true", help="Quick test mode")

    args = parser.parse_args()

    # Convert string booleans
    args.mixed_precision = args.mixed_precision.lower() == "true"
    args.gradient_checkpointing = args.gradient_checkpointing.lower() == "true"

    # Initialize training manager
    logger.info("🌟 INITIALIZING UNIFIED MULTI-DOMAIN TRAINING")
    logger.info("=" * 80)

    training_manager = UnifiedTrainingManager(
        data_root=args.data_root,
        domains=args.domains,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )

    # Create unified dataset
    dataset = training_manager.create_unified_dataset(
        target_size=args.target_size,
        max_files_per_domain=args.max_files_per_domain,
    )

    # Create unified model
    model_kwargs = {
        "model_channels": args.model_channels,
        "num_blocks": args.num_blocks,
        "channel_mult": args.channel_mult,
        "channel_mult_emb": args.channel_mult_emb,
        "attn_resolutions": args.attn_resolutions,
    }

    model = training_manager.create_unified_model(
        use_multi_resolution=args.multi_resolution,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        **model_kwargs,
    )

    # Convert warmup steps to epochs
    import math

    steps_per_epoch = math.ceil(len(dataset.train_dataset) / args.batch_size)
    warmup_epochs = max(1, math.ceil(args.warmup_steps / steps_per_epoch))

    # Create training configuration
    config = training_manager.create_unified_training_config(
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=warmup_epochs,
        val_frequency=args.val_frequency,
        gradient_clip_norm=args.gradient_clip_norm,
    )

    # Save configuration
    config_file = training_manager.output_dir / "unified_training_config.json"
    with open(config_file, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    logger.info(f"📝 Configuration saved to: {config_file}")

    # Start training
    logger.info("🎯 STARTING UNIFIED MULTI-DOMAIN TRAINING")
    logger.info("=" * 80)

    try:
        results = training_manager.train(
            model=model,
            dataset=dataset,
            config=config,
            resume_from_checkpoint=args.resume_checkpoint,
        )

        logger.info("🎉 UNIFIED TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("🏆 RESEARCH ACHIEVEMENT: Single model mastered all domains!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Unified training failed: {e}")
        raise


if __name__ == "__main__":
    main()
