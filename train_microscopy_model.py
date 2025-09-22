#!/usr/bin/env python3
"""
Comprehensive training script for Poisson-Gaussian diffusion restoration with microscopy data.

This script provides:
- Real microscopy data loading and preprocessing (TIFF format)
- Multi-domain training setup optimized for low-photon microscopy
- Comprehensive monitoring and logging
- GPU optimization and memory management
- Early stopping and checkpointing
- Physics-aware loss functions and metrics for microscopy

Usage:
    python train_microscopy_model.py --data_root /path/to/microscopy/data

Requirements:
- Microscopy data in TIFF format (.tif, .tiff)
- Calibration files for each domain
- Sufficient GPU memory (16GB+ recommended)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.error_handlers import ErrorHandler
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


class MicroscopyTrainingManager:
    """Manager for training Poisson-Gaussian model on microscopy data."""

    def __init__(
        self,
        data_root: str,
        output_dir: str = "results/microscopy_training",
        device: str = "auto",
        seed: int = 42,
    ):
        """
        Initialize training manager.

        Args:
            data_root: Path to microscopy data directory
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

        logger.info("🔬 Microscopy Training Manager initialized")
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

    def create_microscopy_dataset(
        self,
        batch_size: int = 4,
        target_size: int = 128,
        max_files: Optional[int] = None,
        validation_split: float = 0.2,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> MultiDomainDataset:
        """
        Create microscopy dataset with proper preprocessing.

        Args:
            batch_size: Batch size for training
            target_size: Target image size for processing
            max_files: Maximum number of files to use (for testing)
            validation_split: Fraction of data for validation

        Returns:
            Configured MultiDomainDataset
        """
        logger.info("📁 Creating microscopy dataset...")

        # Check data availability
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")

        # Find calibration file
        calib_files = list(
            (project_root / "configs/calibrations").glob("*microscopy*.json")
        )
        if not calib_files:
            raise FileNotFoundError("Microscopy calibration file not found")

        calib_file = calib_files[0]
        logger.info(f"  Using calibration: {calib_file.name}")

        # Create dataset configuration
        domain_configs = {
            "microscopy": {
                "data_root": str(self.data_root),
                "calibration_file": str(calib_file),
                "scale": 1000.0,  # Microscopy scale for low-photon imaging
                "target_size": target_size,
                "max_files": max_files,
            }
        }

        dataset = MultiDomainDataset(
            domain_configs=domain_configs,
            split="train",
            balance_domains=False,  # Only one domain
        )

        logger.info("✅ Dataset created successfully")
        logger.info(f"  Training samples: {len(dataset.train_dataset)}")
        logger.info(f"  Validation samples: {len(dataset.val_dataset)}")
        logger.info(f"  Image size: {target_size}x{target_size}")
        logger.info("  Optimized for low-photon microscopy imaging")

        return dataset

    def create_model(
        self,
        use_multi_resolution: bool = False,
        mixed_precision: bool = True,
        **model_kwargs,
    ) -> nn.Module:
        """
        Create EDM model for microscopy restoration (standard or multi-resolution).

        Args:
            use_multi_resolution: Whether to use multi-resolution progressive model
            mixed_precision: Whether to enable mixed precision training
            **model_kwargs: Additional model configuration

        Returns:
            Configured EDM model
        """
        logger.info("🤖 Creating EDM model for microscopy...")

        # Determine domain for channel configuration
        domain = "microscopy"

        if use_multi_resolution:
            logger.info(f"📈 Using Progressive Multi-Resolution EDM for {domain} domain")

            # Multi-resolution model configuration - UNIFIED BASE ARCHITECTURE
            # Same architecture for all domains for fair comparison
            default_model_channels = model_kwargs.get(
                "model_channels", 256
            )  # Unified base architecture
            default_channel_mult = model_kwargs.get(
                "channel_mult", [1, 2, 3, 4]
            )  # Unified base architecture
            default_channel_mult_emb = model_kwargs.get("channel_mult_emb", 6)
            default_num_blocks = model_kwargs.get(
                "num_blocks", 6
            )  # Unified base architecture
            default_attn_resolutions = model_kwargs.get(
                "attn_resolutions", [16, 32, 64]
            )  # Unified base architecture

            # Use resolution range optimized for microscopy
            min_resolution = 32
            max_resolution = 128 if default_model_channels >= 96 else 64
            num_stages = 4 if default_model_channels >= 96 else 3

            model = create_progressive_edm(
                min_resolution=min_resolution,
                max_resolution=max_resolution,
                num_stages=num_stages,
                model_channels=default_model_channels,
                img_channels=1,  # Microscopy data typically has 1 channel (grayscale)
                label_dim=6,  # Domain conditioning
                use_fp16=False,
                dropout=0.05,  # Lower dropout for precision
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

            logger.info(f"🔧 Progressive EDM configuration for {domain} domain")
            logger.info(f"   Model channels: {default_model_channels}")
            logger.info(f"   Max resolution: {max_resolution}x{max_resolution}")
            logger.info(f"   Optimized for low-photon precision imaging")

            if mixed_precision:
                logger.info("🔧 Mixed precision enabled for Progressive EDM models")
        else:
            logger.info(f"📊 Using Standard EDM for {domain} domain")

            # Default model configuration for microscopy - UNIFIED BASE ARCHITECTURE
            default_model_channels = model_kwargs.get(
                "model_channels", 256
            )  # Unified base architecture
            default_channel_mult = model_kwargs.get(
                "channel_mult", [1, 2, 3, 4]
            )  # Unified base architecture
            default_channel_mult_emb = model_kwargs.get("channel_mult_emb", 6)
            default_num_blocks = model_kwargs.get(
                "num_blocks", 6
            )  # Unified base architecture
            default_attn_resolutions = model_kwargs.get(
                "attn_resolutions", [16, 32, 64]
            )  # Unified base architecture

            # Use resolution optimized for microscopy
            img_resolution = 128 if default_model_channels >= 96 else 64

            model_config = {
                "img_resolution": img_resolution,
                "img_channels": 1,  # Microscopy data typically has 1 channel
                "model_channels": default_model_channels,
                "channel_mult": default_channel_mult,
                "channel_mult_emb": default_channel_mult_emb,
                "num_blocks": default_num_blocks,
                "attn_resolutions": default_attn_resolutions,
                "label_dim": 6,  # Domain conditioning
                "use_fp16": False,  # Will use mixed precision training instead
                "dropout": 0.05,  # Lower dropout for precision
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
            model_channels = model_kwargs.get("model_channels", 96)
            is_enhanced = model_channels >= 128

            if is_enhanced:
                logger.info(
                    "🚀 Using ENHANCED EDM configuration for microscopy precision"
                )
                logger.info(f"   Model channels: {model_channels}")
                logger.info(
                    f"   Channel mult: {model_kwargs.get('channel_mult', [1, 2, 3])}"
                )
                logger.info(f"   Num blocks: {model_kwargs.get('num_blocks', 4)}")
                logger.info(
                    f"   Attention resolutions: {model_kwargs.get('attn_resolutions', [16, 32, 64])}"
                )
                logger.info(f"   Image resolution: {img_resolution}x{img_resolution}")
            else:
                logger.info(
                    "🔧 Using precision-optimized EDM configuration for microscopy"
                )
                logger.info(f"   Model channels: {model_channels}")
                logger.info(f"   Configuration optimized for low-photon precision")

            if mixed_precision:
                logger.info("🔧 Mixed precision enabled for EDM models")

        # Move to device and log
        model = model.to(self.device)
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("✅ Model created successfully")
        logger.info(f"  Total parameters: {param_count:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model device: {next(model.parameters()).device}")
        logger.info(f"  Optimized for microscopy low-photon imaging")

        return model

    def create_training_config(
        self,
        num_epochs: int = 150,  # More epochs for microscopy precision
        learning_rate: float = 5e-5,  # Lower learning rate for stability
        batch_size: int = 4,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        save_frequency_steps: int = 25000,  # More frequent saves
        early_stopping_patience_steps: int = 7500,  # More patience for convergence
        validation_checkpoints_patience: int = 30,
        lr_scheduler: str = "cosine",
        warmup_epochs: int = 10,  # More warmup for stability
        val_frequency: int = 3,  # More frequent validation
        gradient_clip_norm: float = 0.5,  # More conservative clipping
        **config_kwargs,
    ) -> MultiDomainTrainingConfig:
        """
        Create comprehensive training configuration optimized for microscopy.

        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate (lower for microscopy stability)
            batch_size: Training batch size
            lr_scheduler: Learning rate scheduler type
            warmup_epochs: Number of warmup epochs
            val_frequency: Validation frequency in epochs
            **config_kwargs: Additional configuration options

        Returns:
            Configured training configuration
        """
        logger.info("⚙️  Creating microscopy training configuration...")

        # Domain balancing configuration optimized for microscopy
        domain_balancing = DomainBalancingConfig(
            sampling_strategy="weighted",
            use_domain_conditioning=True,
            use_domain_loss_weights=True,
            enforce_batch_balance=True,
            min_samples_per_domain_per_batch=1,
            adaptive_rebalancing=True,
            rebalancing_frequency=50,  # More frequent rebalancing
            performance_window=25,  # Shorter window for responsiveness
            log_domain_stats=True,
            domain_stats_frequency=25,  # More frequent logging
        )

        # Main training configuration
        config = MultiDomainTrainingConfig(
            # Model and data
            model_name="poisson_diffusion_microscopy",
            dataset_path=str(self.data_root),
            # Training hyperparameters optimized for microscopy
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            # Optimizer with conservative settings
            optimizer="adamw",
            weight_decay=5e-3,  # Lower weight decay for stability
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            # Scheduler
            scheduler=lr_scheduler,
            min_lr=5e-7,  # Very low minimum learning rate
            scheduler_patience=15,  # More patience
            # Loss function
            loss_type="poisson_gaussian",
            gradient_clip_norm=gradient_clip_norm,
            # Validation and checkpointing
            val_frequency=val_frequency,
            save_frequency_steps=save_frequency_steps,
            max_checkpoints=7,  # Keep more checkpoints
            early_stopping_patience_steps=early_stopping_patience_steps,
            validation_checkpoints_patience=validation_checkpoints_patience,
            early_stopping_min_delta=5e-5,  # Smaller delta for precision
            # Reproducibility
            seed=self.seed,
            deterministic=True,
            benchmark=True,
            # Logging and monitoring
            log_frequency=25,  # More frequent logging
            tensorboard_log_dir=str(self.output_dir / "tensorboard"),
            checkpoint_dir=str(self.output_dir / "checkpoints"),
            # Device and performance
            device=self.device,
            mixed_precision=mixed_precision,
            compile_model=False,  # May cause issues with some setups
            # Multi-domain configuration
            domain_balancing=domain_balancing,
            **config_kwargs,
        )

        logger.info("✅ Microscopy training configuration created")
        logger.info(f"  Training epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate} (optimized for microscopy)")
        logger.info(f"  LR scheduler: {lr_scheduler}")
        logger.info(f"  Warmup epochs: {warmup_epochs}")
        logger.info(f"  Validation frequency: {val_frequency} epochs")
        logger.info(
            f"  Early stopping patience: {config.early_stopping_patience_steps} steps"
        )
        logger.info(f"  Mixed precision: {config.mixed_precision}")
        logger.info(f"  Gradient clip norm: {gradient_clip_norm} (conservative)")

        return config

    def setup_monitoring(self):
        """Setup comprehensive monitoring and logging for microscopy."""
        logger.info("📊 Setting up microscopy monitoring...")

        # Create monitoring directory
        monitoring_dir = self.output_dir / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)

        # Setup TensorBoard logging
        tensorboard_dir = monitoring_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)

        # Create monitoring script for real-time tracking
        monitoring_script = monitoring_dir / "monitor_microscopy_training.py"

        monitoring_code = '''
import time
import torch
import psutil
import json
from pathlib import Path

def monitor_microscopy_training(log_file="logs/training_microscopy.log"):
    """Monitor microscopy training progress in real-time."""
    print("🔬 Real-time Microscopy Training Monitor")
    print("=" * 60)

    last_epoch = -1
    last_loss = None

    while True:
        try:
            # Check if training is still running
            if not any("python" in p.info["name"] and "microscopy" in " ".join(p.cmdline())
                      for p in psutil.process_iter(["pid", "name", "cmdline"])):
                print("⏹️  Training process ended")
                break

            # Monitor GPU if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info()[0]
                gpu_util = torch.cuda.utilization()
                print(f"🖥️  GPU Memory: {gpu_memory / 1e9:.1f} GB used, Utilization: {gpu_util}%")

            # Monitor CPU and RAM
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            print(f"💻 CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}%")

            # Check for new log entries (simplified)
            if Path(log_file).exists():
                print("📝 Microscopy training log updated...")

            print("-" * 40)
            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("⏹️  Monitor stopped by user")
            break
        except Exception as e:
            print(f"⚠️  Monitor error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_microscopy_training()
'''

        with open(monitoring_script, "w") as f:
            f.write(monitoring_code)

        # Make monitoring script executable
        monitoring_script.chmod(0o755)

        logger.info("✅ Microscopy monitoring setup complete")
        logger.info(f"  TensorBoard: {tensorboard_dir}")
        logger.info(f"  Monitor script: {monitoring_script}")
        logger.info("  Use: python monitor_microscopy_training.py to track training")

        return monitoring_dir

    def train(
        self,
        model: nn.Module,
        dataset: MultiDomainDataset,
        config: MultiDomainTrainingConfig,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model with comprehensive monitoring.

        Args:
            model: Model to train
            dataset: Training dataset
            config: Training configuration
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results and metrics
        """
        logger.info("🚀 Starting microscopy training...")
        logger.info("=" * 60)

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
                "domain": "microscopy",
            }

            # Save results
            results_file = self.output_dir / "training_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info("✅ Microscopy training completed successfully!")
            logger.info(f"  Total time: {training_time / 3600:.2f} hours")
            logger.info(f"  Best validation loss: {trainer.best_val_loss:.6f}")
            logger.info(f"  Results saved to: {results_file}")

            return results

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(
                f"❌ Microscopy training failed after {error_time / 3600:.2f} hours: {e}"
            )

            # Save partial results
            error_results = {
                "error": str(e),
                "training_time_hours": error_time / 3600,
                "failed_epoch": getattr(trainer, "current_epoch", 0),
                "device": str(self.device),
                "domain": "microscopy",
            }

            error_file = self.output_dir / "training_error.json"
            with open(error_file, "w") as f:
                json.dump(error_results, f, indent=2, default=str)

            raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Poisson-Gaussian diffusion model on microscopy data"
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to microscopy data directory",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=150000,
        help="Maximum training steps (overrides epochs if set)",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (optimized for microscopy)",
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
        default=2000,
        help="Number of warmup steps (will be converted to epochs)",
    )
    parser.add_argument(
        "--val_frequency",
        type=int,
        default=3,
        help="Validation frequency in epochs (ignored if val_frequency_steps is set)",
    )
    parser.add_argument(
        "--val_frequency_steps",
        type=int,
        default=None,
        help="Validation frequency in training steps (overrides val_frequency if set)",
    )

    # Model arguments
    parser.add_argument(
        "--model_channels",
        type=int,
        default=256,
        help="Number of model channels (unified base architecture)",
    )
    parser.add_argument(
        "--num_blocks", type=int, default=6, help="Number of model blocks"
    )
    parser.add_argument(
        "--multi_resolution",
        action="store_true",
        help="Use multi-resolution progressive EDM model",
    )

    # Enhanced model architecture arguments
    parser.add_argument(
        "--channel_mult",
        type=int,
        nargs="+",
        default=None,
        help="Channel multipliers (e.g., 1 2 3)",
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
        help="Attention resolutions (e.g., 8 16)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/microscopy_training",
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

    # Performance arguments
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

    # Checkpointing arguments
    parser.add_argument(
        "--save_frequency_steps",
        type=int,
        default=25000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--early_stopping_patience_steps",
        type=int,
        default=7500,
        help="Early stopping patience in steps",
    )
    parser.add_argument(
        "--validation_checkpoints_patience",
        type=int,
        default=30,
        help="Early stopping patience in validation checkpoints",
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=0.5,
        help="Gradient clipping norm threshold (conservative)",
    )

    # Advanced checkpointing arguments
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=7,
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
        "--quick_test", action="store_true", help="Run quick test with synthetic data"
    )

    args = parser.parse_args()

    # Convert string boolean arguments to actual booleans
    args.mixed_precision = args.mixed_precision.lower() == "true"
    args.pin_memory = args.pin_memory.lower() == "true"
    args.save_best_model = args.save_best_model.lower() == "true"
    args.save_optimizer_state = args.save_optimizer_state.lower() == "true"
    args.resume_from_best = args.resume_from_best.lower() == "true"
    args.ddp_find_unused_parameters = args.ddp_find_unused_parameters.lower() == "true"

    # Initialize training manager
    logger.info("🚀 INITIALIZING MICROSCOPY TRAINING")
    logger.info("=" * 60)

    training_manager = MicroscopyTrainingManager(
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )

    # Setup monitoring
    monitoring_dir = training_manager.setup_monitoring()

    # Create model (enable multi-resolution if requested)
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

    model = training_manager.create_model(
        use_multi_resolution=args.multi_resolution,
        mixed_precision=args.mixed_precision,
        **model_kwargs,
    )

    # Create dataset
    if args.quick_test:
        logger.info("🧪 Running quick test with synthetic data...")
        # Generate synthetic data for testing
        from scripts.generate_synthetic_data import (
            SyntheticConfig,
            SyntheticDataGenerator,
        )

        synthetic_config = SyntheticConfig(
            output_dir="data/synthetic_microscopy_quick",
            num_images=100,
            image_size=args.target_size,
            pattern_types=[
                "constant",
                "gradient",
                "spots",
            ],  # Microscopy-relevant patterns
            photon_levels=[5, 50, 500],  # Lower photon counts for microscopy
            read_noise_levels=[1, 3],
        )

        synthetic_generator = SyntheticDataGenerator(synthetic_config)
        logger.info("🔄 Generating synthetic microscopy data...")
        results = synthetic_generator.generate_validation_set()
        synthetic_generator.save_dataset(results)

        # Create a simple synthetic dataset wrapper
        class SyntheticDataset:
            def __init__(self, data_dir, target_size=128):
                self.data_dir = Path(data_dir)
                self.target_size = target_size
                self.image_files = list((self.data_dir / "images").glob("*.npz"))

                # Create train/val split
                split_idx = int(0.8 * len(self.image_files))
                self.train_files = self.image_files[:split_idx]
                self.val_files = self.image_files[split_idx:]

                # Create simple dataset objects
                self.train_dataset = SyntheticSubset(self.train_files, target_size)
                self.val_dataset = SyntheticSubset(self.val_files, target_size)

                # Add domain_datasets attribute for MultiDomainTrainer compatibility
                self.train_dataset.domain_datasets = {"microscopy": self.train_dataset}
                self.val_dataset.domain_datasets = {"microscopy": self.val_dataset}

        class SyntheticSubset:
            def __init__(self, files, target_size):
                self.files = files
                self.target_size = target_size

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                data = np.load(self.files[idx])
                clean = (
                    torch.from_numpy(data["clean"]).float().unsqueeze(0)
                )  # Add channel dim
                noisy = torch.from_numpy(data["noisy"]).float().unsqueeze(0)

                # Microscopy typically uses single channel
                clean_1ch = clean
                noisy_1ch = noisy

                # Resize if needed
                if clean_1ch.shape[-1] != self.target_size:
                    clean_1ch = torch.nn.functional.interpolate(
                        clean_1ch.unsqueeze(0), size=self.target_size, mode="bilinear"
                    ).squeeze(0)
                    noisy_1ch = torch.nn.functional.interpolate(
                        noisy_1ch.unsqueeze(0), size=self.target_size, mode="bilinear"
                    ).squeeze(0)

                return {
                    "clean": clean_1ch,
                    "noisy": noisy_1ch,
                    "electrons": clean_1ch,  # Use as target electrons
                    "domain": torch.tensor([1]),  # Microscopy domain
                    "metadata": {"synthetic": True},
                }

        dataset = SyntheticDataset(synthetic_config.output_dir, args.target_size)
    else:
        # Use real preprocessed data
        logger.info("📁 Loading real preprocessed microscopy data...")
        from data.preprocessed_datasets import create_preprocessed_datasets
        from utils.training_config import (
            calculate_optimal_training_config,
            print_training_analysis,
        )

        train_dataset, val_dataset = create_preprocessed_datasets(
            data_root=args.data_root,
            domain="microscopy",
            max_files=args.max_files,
            seed=args.seed,
        )

        # Calculate optimal training configuration
        dataset_size = len(train_dataset)
        optimal_config = calculate_optimal_training_config(
            dataset_size, args.batch_size
        )

        logger.info("🎯 AUTOMATIC MICROSCOPY TRAINING CONFIGURATION:")
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

        # Create a simple dataset wrapper with train/val attributes
        class DatasetWrapper:
            def __init__(self, train_ds, val_ds):
                self.train_dataset = train_ds
                self.val_dataset = val_ds

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

    # Create training configuration
    config = training_manager.create_training_config(
        num_epochs=args.epochs,
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
        gradient_clip_norm=args.gradient_clip_norm,
    )

    # Save configuration
    config_file = training_manager.output_dir / "training_config.json"
    with open(config_file, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    logger.info(f"📝 Configuration saved to: {config_file}")

    # Start training
    logger.info("🎯 STARTING MICROSCOPY TRAINING")
    logger.info("=" * 60)

    try:
        results = training_manager.train(
            model=model,
            dataset=dataset,
            config=config,
            resume_from_checkpoint=args.resume_checkpoint,
        )

        # Success summary
        logger.info("🎉 MICROSCOPY TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("📊 Final Results:")
        logger.info(f"  Training time: {results['training_time_hours']:.2f} hours")
        logger.info(f"  Best validation loss: {results['best_val_loss']:.6f}")
        logger.info(f"  Final epoch: {results['final_epoch']}")
        logger.info("=" * 60)
        logger.info("📁 Outputs:")
        logger.info(f"  Checkpoints: {training_manager.output_dir}/checkpoints/")
        logger.info(f"  Results: {training_manager.output_dir}/training_results.json")
        logger.info(f"  Logs: {monitoring_dir}/")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Microscopy training failed: {e}")
        logger.error("Check logs for details")
        raise


if __name__ == "__main__":
    main()
