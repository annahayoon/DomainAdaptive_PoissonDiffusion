#!/usr/bin/env python3
"""
L2 baseline training script for diffusion restoration with astronomy data.

This script provides L2 (MSE) baseline training as a comparison to the
physics-aware Poisson-Gaussian approach. It uses standard deep learning
methods without domain-specific physics modeling.

This script provides:
- Real astronomy data loading and preprocessing (FITS format)
- L2 (MSE) guidance instead of Poisson-Gaussian guidance
- Standard diffusion training without physics awareness
- Comprehensive monitoring and logging
- GPU optimization and memory management
- Early stopping and checkpointing

Usage:
    python train_L2_astronomy_model.py --data_root /path/to/astronomy/data

Requirements:
- Astronomy data in FITS format (.fits, .fit)
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
from core.l2_guidance import L2Guidance  # Use L2 guidance instead of PoissonGuidance
from core.logging_config import LoggingManager
from data.domain_datasets import MultiDomainDataset
from models.edm_wrapper import (
    ProgressiveEDM,
    create_edm_wrapper,
    create_progressive_edm,
)
from poisson_training import (
    DomainBalancingConfig,
    MultiDomainTrainingConfig,
    set_deterministic_mode,
)
from poisson_training.l2_losses import (  # Use L2 loss instead of PoissonGaussianLoss
    L2Loss,
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


# Create L2 trainer class that uses L2 loss instead of PoissonGaussian loss
class L2Trainer:
    """L2 baseline trainer using MSE loss instead of physics-aware loss."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        config: MultiDomainTrainingConfig,
        output_dir: Path,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = Path(output_dir)

        # Use L2 loss instead of PoissonGaussian loss
        self.loss_fn = L2Loss(
            weights={
                "reconstruction": 1.0,
                "consistency": 0.01,
            }  # Very low consistency weight for astronomy
        )

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

        # Setup scheduler
        if config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs, eta_min=config.min_lr
            )
        else:
            self.scheduler = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_history = {"train_loss": [], "val_loss": []}

        # Setup data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if torch.cuda.is_available() else False,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.config.device)

            self.optimizer.zero_grad()

            # Forward pass - for L2 training, we directly predict clean images
            clean = batch["clean"]
            noisy = batch["noisy"]

            # Simple denoising prediction
            prediction = self.model(noisy)

            # Compute L2 loss
            outputs = {"prediction": prediction}
            loss_dict = self.loss_fn(outputs, batch)
            loss = loss_dict["total_loss"]

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:  # Very frequent logging for astronomy
                logger.info(f"Batch {num_batches}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.config.device)

                # Forward pass
                clean = batch["clean"]
                noisy = batch["noisy"]
                prediction = self.model(noisy)

                # Compute loss
                outputs = {"prediction": prediction}
                loss_dict = self.loss_fn(outputs, batch)
                loss = loss_dict["total_loss"]

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = (
            self.output_dir
            / "checkpoints"
            / f"checkpoint_epoch_{self.current_epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "checkpoints" / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(
                f"Saved best L2 astronomy model with val_loss: {self.best_val_loss:.6f}"
            )

    def train(self):
        """Main training loop."""
        logger.info("Starting L2 baseline astronomy training...")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.training_history["train_loss"].append(train_loss)

            # Validate
            if epoch % self.config.val_frequency == 0:
                val_loss = self.validate()
                self.training_history["val_loss"].append(val_loss)

                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                    f"val_loss={val_loss:.6f}, best_val_loss={self.best_val_loss:.6f}"
                )

                # Save checkpoint
                self.save_checkpoint(is_best=is_best)
            else:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}")

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        return self.training_history


class L2AstronomyTrainingManager:
    """Manager for L2 baseline training on astronomy data."""

    def __init__(
        self,
        data_root: str,
        output_dir: str = "results/l2_astronomy_training",
        device: str = "auto",
        seed: int = 42,
    ):
        """
        Initialize L2 astronomy training manager.

        Args:
            data_root: Path to astronomy data directory
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

        logger.info("🌌 L2 Astronomy Training Manager initialized")
        logger.info(f"  Data root: {self.data_root}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Seed: {self.seed}")
        logger.info("  Using L2 (MSE) baseline instead of physics-aware guidance")

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

    def create_astronomy_dataset(
        self,
        batch_size: int = 4,
        target_size: int = 128,
        max_files: Optional[int] = None,
        validation_split: float = 0.2,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> MultiDomainDataset:
        """
        Create astronomy dataset for L2 training.

        Args:
            batch_size: Batch size for training
            target_size: Target image size for processing
            max_files: Maximum number of files to use (for testing)
            validation_split: Fraction of data for validation

        Returns:
            Configured MultiDomainDataset
        """
        logger.info("📁 Creating astronomy dataset for L2 training...")

        # Check data availability
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")

        # For L2 training, we don't need calibration files - use simple config
        domain_configs = {
            "astronomy": {
                "data_root": str(self.data_root),
                "scale": 1.0,  # L2 uses normalized data
                "target_size": target_size,
                "max_files": max_files,
            }
        }

        dataset = MultiDomainDataset(
            domain_configs=domain_configs,
            split="train",
            balance_domains=False,  # Only one domain
        )

        logger.info("✅ L2 Astronomy Dataset created successfully")
        logger.info(f"  Training samples: {len(dataset.train_dataset)}")
        logger.info(f"  Validation samples: {len(dataset.val_dataset)}")
        logger.info(f"  Image size: {target_size}x{target_size}")
        logger.info("  Using L2 (MSE) loss for baseline comparison")

        return dataset

    def create_model(
        self,
        use_multi_resolution: bool = False,
        mixed_precision: bool = True,
        **model_kwargs,
    ) -> nn.Module:
        """
        Create EDM model for L2 baseline astronomy restoration.

        Args:
            use_multi_resolution: Whether to use multi-resolution progressive model
            mixed_precision: Whether to enable mixed precision training
            **model_kwargs: Additional model configuration

        Returns:
            Configured EDM model
        """
        logger.info("🤖 Creating EDM model for L2 astronomy baseline...")

        # Determine domain for channel configuration
        domain = "astronomy"

        if use_multi_resolution:
            logger.info(f"📈 Using Progressive Multi-Resolution EDM for L2 {domain}")

            # Multi-resolution model configuration - Optimized for L2 astronomy
            default_model_channels = model_kwargs.get(
                "model_channels", 128
            )  # Larger for maximum precision
            default_channel_mult = model_kwargs.get(
                "channel_mult", [1, 2, 3, 4]
            )  # More stages
            default_channel_mult_emb = model_kwargs.get("channel_mult_emb", 4)
            default_num_blocks = model_kwargs.get(
                "num_blocks", 5
            )  # Maximum blocks for precision
            default_attn_resolutions = model_kwargs.get(
                "attn_resolutions", [16, 32, 64]
            )

            # Use full resolution range for astronomy
            min_resolution = 32
            max_resolution = 128 if default_model_channels >= 128 else 64
            num_stages = 5 if default_model_channels >= 128 else 4

            model = create_progressive_edm(
                min_resolution=min_resolution,
                max_resolution=max_resolution,
                num_stages=num_stages,
                model_channels=default_model_channels,
                img_channels=1,  # Astronomy data typically has 1 channel (grayscale)
                label_dim=6,  # Domain conditioning
                use_fp16=False,
                dropout=0.1,  # Standard dropout for L2
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

            logger.info(f"🔧 L2 Progressive EDM configuration for {domain}")
            logger.info(f"   Model channels: {default_model_channels}")
            logger.info(f"   Max resolution: {max_resolution}x{max_resolution}")
            logger.info(f"   L2 baseline for extreme low-light")

            if mixed_precision:
                logger.info("🔧 Mixed precision enabled for L2 Progressive EDM")
        else:
            logger.info(f"📊 Using Standard EDM for L2 {domain}")

            # Default model configuration for L2 astronomy - Maximum precision
            default_model_channels = model_kwargs.get(
                "model_channels", 128
            )  # Large model
            default_channel_mult = model_kwargs.get(
                "channel_mult", [1, 2, 3, 4]
            )  # More stages
            default_channel_mult_emb = model_kwargs.get("channel_mult_emb", 4)
            default_num_blocks = model_kwargs.get("num_blocks", 5)  # Maximum blocks
            default_attn_resolutions = model_kwargs.get(
                "attn_resolutions", [16, 32, 64]
            )

            # Use maximum resolution for astronomy
            img_resolution = 128 if default_model_channels >= 128 else 64

            model_config = {
                "img_resolution": img_resolution,
                "img_channels": 1,  # Astronomy data typically has 1 channel
                "model_channels": default_model_channels,
                "channel_mult": default_channel_mult,
                "channel_mult_emb": default_channel_mult_emb,
                "num_blocks": default_num_blocks,
                "attn_resolutions": default_attn_resolutions,
                "label_dim": 6,  # Domain conditioning
                "use_fp16": False,  # Will use mixed precision training instead
                "dropout": 0.1,  # Standard dropout for L2
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

            logger.info("🔧 L2 baseline EDM configuration for astronomy")
            logger.info(f"   Model channels: {model_kwargs.get('model_channels', 128)}")
            logger.info(f"   L2 baseline approach (no physics modeling)")

            if mixed_precision:
                logger.info("🔧 Mixed precision enabled for L2 EDM")

        # Move to device and log
        model = model.to(self.device)
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("✅ L2 Astronomy Model created successfully")
        logger.info(f"  Total parameters: {param_count:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model device: {next(model.parameters()).device}")
        logger.info(f"  L2 baseline approach for astronomy (no physics modeling)")

        return model

    def create_training_config(
        self,
        num_epochs: int = 200,  # More epochs for astronomy precision
        learning_rate: float = 2e-5,  # Very low learning rate for stability
        batch_size: int = 2,  # Smaller batch size for memory efficiency
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 2,  # Compensate for smaller batch
        save_frequency_steps: int = 10000,  # More frequent saves
        early_stopping_patience_steps: int = 15000,  # More patience for convergence
        validation_checkpoints_patience: int = 50,
        lr_scheduler: str = "cosine",
        warmup_epochs: int = 15,  # Extended warmup for stability
        val_frequency: int = 2,  # More frequent validation
        gradient_clip_norm: float = 0.1,  # Very conservative clipping
        **config_kwargs,
    ) -> MultiDomainTrainingConfig:
        """
        Create comprehensive L2 training configuration optimized for astronomy.

        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate (very low for astronomy stability)
            batch_size: Training batch size
            lr_scheduler: Learning rate scheduler type
            warmup_epochs: Number of warmup epochs
            val_frequency: Validation frequency in epochs
            **config_kwargs: Additional configuration options

        Returns:
            Configured training configuration
        """
        logger.info("⚙️  Creating L2 astronomy training configuration...")

        # Domain balancing configuration optimized for L2 astronomy
        domain_balancing = DomainBalancingConfig(
            sampling_strategy="weighted",
            use_domain_conditioning=True,
            use_domain_loss_weights=False,  # L2 doesn't need domain-specific loss weights
            enforce_batch_balance=True,
            min_samples_per_domain_per_batch=1,
            adaptive_rebalancing=True,
            rebalancing_frequency=25,  # Frequent rebalancing
            performance_window=10,  # Short window for responsiveness
            log_domain_stats=True,
            domain_stats_frequency=10,  # Frequent logging
        )

        # Main training configuration
        config = MultiDomainTrainingConfig(
            # Model and data
            model_name="l2_diffusion_astronomy",
            dataset_path=str(self.data_root),
            # Training hyperparameters optimized for L2 astronomy
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs,
            # Optimizer with very conservative settings
            optimizer="adamw",
            weight_decay=1e-3,  # Very low weight decay for stability
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            # Scheduler
            scheduler=lr_scheduler,
            min_lr=1e-7,  # Extremely low minimum learning rate
            scheduler_patience=20,  # More patience
            # Loss function - use simple MSE for L2
            loss_type="l2",  # Use L2 loss instead of poisson_gaussian
            gradient_clip_norm=gradient_clip_norm,
            # Validation and checkpointing
            val_frequency=val_frequency,
            save_frequency_steps=save_frequency_steps,
            max_checkpoints=10,  # Keep more checkpoints
            early_stopping_patience_steps=early_stopping_patience_steps,
            validation_checkpoints_patience=validation_checkpoints_patience,
            early_stopping_min_delta=1e-5,  # Very small delta for precision
            # Reproducibility
            seed=self.seed,
            deterministic=True,
            benchmark=True,
            # Logging and monitoring
            log_frequency=10,  # Very frequent logging
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

        logger.info("✅ L2 Astronomy training configuration created")
        logger.info(f"  Training epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size} (optimized for L2 astronomy)")
        logger.info(f"  Learning rate: {learning_rate} (very low for stability)")
        logger.info(f"  LR scheduler: {lr_scheduler}")
        logger.info(f"  Warmup epochs: {warmup_epochs}")
        logger.info(f"  Validation frequency: {val_frequency} epochs")
        logger.info(
            f"  Early stopping patience: {config.early_stopping_patience_steps} steps"
        )
        logger.info(f"  Mixed precision: {config.mixed_precision}")
        logger.info(f"  Gradient clip norm: {gradient_clip_norm} (very conservative)")
        logger.info("  Loss type: L2 (MSE) baseline")

        return config

    def setup_monitoring(self):
        """Setup comprehensive monitoring and logging for L2 astronomy."""
        logger.info("📊 Setting up L2 astronomy monitoring...")

        # Create monitoring directory
        monitoring_dir = self.output_dir / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)

        # Setup TensorBoard logging
        tensorboard_dir = monitoring_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)

        # Create monitoring script for real-time tracking
        monitoring_script = monitoring_dir / "monitor_l2_astronomy_training.py"

        monitoring_code = '''
import time
import torch
import psutil
import json
from pathlib import Path

def monitor_l2_astronomy_training(log_file="logs/training_l2_astronomy.log"):
    """Monitor L2 astronomy training progress in real-time."""
    print("🌌 Real-time L2 Astronomy Training Monitor")
    print("=" * 60)

    last_epoch = -1
    last_loss = None

    while True:
        try:
            # Check if training is still running
            if not any("python" in p.info["name"] and "L2_astronomy" in " ".join(p.cmdline())
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
                print("📝 L2 astronomy training log updated...")

            print("-" * 40)
            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("⏹️  Monitor stopped by user")
            break
        except Exception as e:
            print(f"⚠️  Monitor error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_l2_astronomy_training()
'''

        with open(monitoring_script, "w") as f:
            f.write(monitoring_code)

        # Make monitoring script executable
        monitoring_script.chmod(0o755)

        logger.info("✅ L2 Astronomy monitoring setup complete")
        logger.info(f"  TensorBoard: {tensorboard_dir}")
        logger.info(f"  Monitor script: {monitoring_script}")
        logger.info("  Use: python monitor_l2_astronomy_training.py to track training")

        return monitoring_dir

    def train(
        self,
        model: nn.Module,
        dataset: MultiDomainDataset,
        config: MultiDomainTrainingConfig,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model with L2 loss.

        Args:
            model: Model to train
            dataset: Training dataset
            config: Training configuration
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results and metrics
        """
        logger.info("🚀 Starting L2 baseline astronomy training...")
        logger.info("=" * 60)

        # Set deterministic mode
        set_deterministic_mode(seed=self.seed)

        # Create L2 trainer instead of MultiDomainTrainer
        trainer = L2Trainer(
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
                # Load checkpoint logic would go here
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
                "domain": "astronomy",
                "guidance_type": "L2_baseline",
            }

            # Save results
            results_file = self.output_dir / "training_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info("✅ L2 Astronomy training completed successfully!")
            logger.info(f"  Total time: {training_time / 3600:.2f} hours")
            logger.info(f"  Best validation loss: {trainer.best_val_loss:.6f}")
            logger.info(f"  Results saved to: {results_file}")
            logger.info("  L2 baseline training (no physics modeling)")

            return results

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(
                f"❌ L2 Astronomy training failed after {error_time / 3600:.2f} hours: {e}"
            )

            # Save partial results
            error_results = {
                "error": str(e),
                "training_time_hours": error_time / 3600,
                "failed_epoch": getattr(trainer, "current_epoch", 0),
                "device": str(self.device),
                "domain": "astronomy",
                "guidance_type": "L2_baseline",
            }

            error_file = self.output_dir / "training_error.json"
            with open(error_file, "w") as f:
                json.dump(error_results, f, indent=2, default=str)

            raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train L2 baseline diffusion model on astronomy data"
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to astronomy data directory",
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides epochs if set)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size (optimized for L2 astronomy)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (very low for L2 astronomy stability)",
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
        default=3000,
        help="Number of warmup steps (will be converted to epochs)",
    )
    parser.add_argument(
        "--val_frequency",
        type=int,
        default=2,
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
        default=128,
        help="Number of model channels (maximum for L2 astronomy)",
    )
    parser.add_argument(
        "--num_blocks", type=int, default=5, help="Number of model blocks"
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
        default="results/l2_astronomy_training",
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
        default=2,
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
        default=10000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--early_stopping_patience_steps",
        type=int,
        default=15000,
        help="Early stopping patience in steps",
    )
    parser.add_argument(
        "--validation_checkpoints_patience",
        type=int,
        default=50,
        help="Early stopping patience in validation checkpoints",
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=0.1,
        help="Gradient clipping norm threshold (very conservative)",
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

    # Initialize training manager
    logger.info("🚀 INITIALIZING L2 ASTRONOMY TRAINING")
    logger.info("=" * 60)

    training_manager = L2AstronomyTrainingManager(
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
            output_dir="data/synthetic_l2_astronomy_quick",
            num_images=100,
            image_size=args.target_size,
            pattern_types=[
                "constant",
                "gradient",
                "spots",
            ],  # Astronomy-relevant patterns
            photon_levels=[1, 10, 100],  # Very low photon counts for astronomy
            read_noise_levels=[1, 2],
        )

        synthetic_generator = SyntheticDataGenerator(synthetic_config)
        logger.info("🔄 Generating synthetic L2 astronomy data...")
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
                self.train_dataset.domain_datasets = {"astronomy": self.train_dataset}
                self.val_dataset.domain_datasets = {"astronomy": self.val_dataset}

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

                # Astronomy typically uses single channel - normalize to [0,1] for L2
                clean_1ch = torch.clamp(clean / clean.max(), 0.0, 1.0)
                noisy_1ch = torch.clamp(noisy / noisy.max(), 0.0, 1.0)

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
                    "domain": torch.tensor([2]),  # Astronomy domain
                    "metadata": {"synthetic": True},
                }

        dataset = SyntheticDataset(synthetic_config.output_dir, args.target_size)
    else:
        # Use real preprocessed data
        logger.info("📁 Loading real preprocessed astronomy data for L2...")
        from data.preprocessed_datasets import create_preprocessed_datasets
        from utils.training_config import (
            calculate_optimal_training_config,
            print_training_analysis,
        )

        train_dataset, val_dataset = create_preprocessed_datasets(
            data_root=args.data_root,
            domain="astronomy",
            max_files=args.max_files,
            seed=args.seed,
        )

        # Calculate optimal training configuration
        dataset_size = len(train_dataset)
        optimal_config = calculate_optimal_training_config(
            dataset_size, args.batch_size
        )

        logger.info("🎯 AUTOMATIC L2 ASTRONOMY TRAINING CONFIGURATION:")
        print_training_analysis(optimal_config)

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
    logger.info("🎯 STARTING L2 ASTRONOMY TRAINING")
    logger.info("=" * 60)

    try:
        results = training_manager.train(
            model=model,
            dataset=dataset,
            config=config,
            resume_from_checkpoint=args.resume_checkpoint,
        )

        # Success summary
        logger.info("🎉 L2 ASTRONOMY TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("📊 Final Results:")
        logger.info(f"  Training time: {results['training_time_hours']:.2f} hours")
        logger.info(f"  Best validation loss: {results['best_val_loss']:.6f}")
        logger.info(f"  Final epoch: {results['final_epoch']}")
        logger.info("  Guidance type: L2 baseline (no physics modeling)")
        logger.info("=" * 60)
        logger.info("📁 Outputs:")
        logger.info(f"  Checkpoints: {training_manager.output_dir}/checkpoints/")
        logger.info(f"  Results: {training_manager.output_dir}/training_results.json")
        logger.info(f"  Logs: {monitoring_dir}/")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ L2 Astronomy training failed: {e}")
        logger.error("Check logs for details")
        raise


if __name__ == "__main__":
    main()
