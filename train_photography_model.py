#!/usr/bin/env python3
"""
Comprehensive training script for Poisson-Gaussian diffusion restoration with photography data.

This script provides:
- Real photography data loading and preprocessing
- Multi-domain training setup
- Comprehensive monitoring and logging
- GPU optimization and memory management
- Early stopping and checkpointing
- Physics-aware loss functions and metrics

Usage:
    python train_photography_model.py --data_root /path/to/photography/data

Requirements:
- Photography data in RAW format (.arw, .dng, .nef, .cr2)
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
from models.edm_wrapper import create_edm_wrapper
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


class PhotographyTrainingManager:
    """Manager for training Poisson-Gaussian model on photography data."""

    def __init__(
        self,
        data_root: str,
        output_dir: str = "results/photography_training",
        device: str = "auto",
        seed: int = 42,
    ):
        """
        Initialize training manager.

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

        logger.info("🔬 Photography Training Manager initialized")
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

    def create_photography_dataset(
        self,
        batch_size: int = 4,
        target_size: int = 128,
        max_files: Optional[int] = None,
        validation_split: float = 0.2,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> MultiDomainDataset:
        """
        Create photography dataset with proper preprocessing.

        Args:
            batch_size: Batch size for training
            target_size: Target image size for processing
            max_files: Maximum number of files to use (for testing)
            validation_split: Fraction of data for validation

        Returns:
            Configured MultiDomainDataset
        """
        logger.info("📁 Creating photography dataset...")

        # Check data availability
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")

        # Find calibration file
        calib_files = list(
            (project_root / "configs/calibrations").glob("*photography*.json")
        )
        if not calib_files:
            raise FileNotFoundError("Photography calibration file not found")

        calib_file = calib_files[0]
        logger.info(f"  Using calibration: {calib_file.name}")

        # Create dataset configuration
        domain_configs = {
            "photography": {
                "data_root": str(self.data_root),
                "calibration_file": str(calib_file),
                "scale": 10000.0,  # Default photography scale
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

        return dataset

    def create_model(self, **model_kwargs) -> nn.Module:
        """
        Create EDM model for photography restoration.

        Args:
            **model_kwargs: Additional model configuration

        Returns:
            Configured EDM model
        """
        logger.info("🤖 Creating EDM model...")

        # Default model configuration for photography
        model_config = {
            "img_resolution": 128,
            "img_channels": 1,
            "model_channels": 128,
            "channel_mult": [1, 2, 2, 2],
            "num_blocks": 4,
            "attn_resolutions": [16],
            "label_dim": 6,  # Domain conditioning
            "use_fp16": False,
            "dropout": 0.1,
            **model_kwargs,
        }

        model = create_edm_wrapper(**model_config)

        # Move to device and log
        model = model.to(self.device)
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("✅ Model created successfully")
        logger.info(f"  Total parameters: {param_count:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model device: {next(model.parameters()).device}")

        return model

    def create_training_config(
        self,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        save_frequency: int = 5,
        val_frequency: int = 1,
        **config_kwargs,
    ) -> MultiDomainTrainingConfig:
        """
        Create comprehensive training configuration.

        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            **config_kwargs: Additional configuration options

        Returns:
            Configured training configuration
        """
        logger.info("⚙️  Creating training configuration...")

        # Domain balancing configuration
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

        # Main training configuration
        config = MultiDomainTrainingConfig(
            # Model and data
            model_name="poisson_diffusion_photography",
            dataset_path=str(self.data_root),
            # Training hyperparameters
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_epochs=5,
            # Optimizer
            optimizer="adamw",
            weight_decay=1e-2,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            # Scheduler
            scheduler="cosine",
            min_lr=1e-6,
            scheduler_patience=10,
            # Loss function
            loss_type="poisson_gaussian",
            gradient_clip_norm=1.0,
            # Validation and checkpointing
            val_frequency=val_frequency,
            save_frequency=save_frequency,
            max_checkpoints=5,
            early_stopping_patience=20,
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
            # Multi-domain configuration
            domain_balancing=domain_balancing,
            **config_kwargs,
        )

        logger.info("✅ Training configuration created")
        logger.info(f"  Training epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Early stopping patience: {config.early_stopping_patience}")
        logger.info(f"  Mixed precision: {config.mixed_precision}")

        return config

    def setup_monitoring(self):
        """Setup comprehensive monitoring and logging."""
        logger.info("📊 Setting up monitoring...")

        # Create monitoring directory
        monitoring_dir = self.output_dir / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)

        # Setup TensorBoard logging
        tensorboard_dir = monitoring_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)

        # Create monitoring script for real-time tracking
        monitoring_script = monitoring_dir / "monitor_training.py"

        monitoring_code = '''
import time
import torch
import psutil
import json
from pathlib import Path

def monitor_training(log_file="logs/training_photography.log"):
    """Monitor training progress in real-time."""
    print("📊 Real-time Training Monitor")
    print("=" * 50)

    last_epoch = -1
    last_loss = None

    while True:
        try:
            # Check if training is still running
            if not any("python" in p.info["name"] and "train" in " ".join(p.cmdline())
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
                print("📝 Training log updated...")

            print("-" * 30)
            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("⏹️  Monitor stopped by user")
            break
        except Exception as e:
            print(f"⚠️  Monitor error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_training()
'''

        with open(monitoring_script, "w") as f:
            f.write(monitoring_code)

        # Make monitoring script executable
        monitoring_script.chmod(0o755)

        logger.info("✅ Monitoring setup complete")
        logger.info(f"  TensorBoard: {tensorboard_dir}")
        logger.info(f"  Monitor script: {monitoring_script}")
        logger.info("  Use: python monitor_training.py to track training")

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
        logger.info("🚀 Starting training...")
        logger.info("=" * 60)

        # Set deterministic mode
        set_deterministic_mode(seed=self.seed)

        # Create trainer
        trainer = MultiDomainTrainer(
            model=model,
            train_dataset=dataset.train_dataset,
            val_dataset=dataset.val_dataset,
            config=config,
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
            }

            # Save results
            results_file = self.output_dir / "training_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info("✅ Training completed successfully!")
            logger.info(f"  Total time: {training_time / 3600:.2f} hours")
            logger.info(f"  Best validation loss: {trainer.best_val_loss:.6f}")
            logger.info(f"  Results saved to: {results_file}")

            return results

        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"❌ Training failed after {error_time / 3600:.2f} hours: {e}")

            # Save partial results
            error_results = {
                "error": str(e),
                "training_time_hours": error_time / 3600,
                "failed_epoch": getattr(trainer, "current_epoch", 0),
                "device": str(self.device),
            }

            error_file = self.output_dir / "training_error.json"
            with open(error_file, "w") as f:
                json.dump(error_results, f, indent=2, default=str)

            raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Poisson-Gaussian diffusion model on photography data"
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
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--target_size", type=int, default=128, help="Target image size"
    )

    # Model arguments
    parser.add_argument(
        "--model_channels", type=int, default=128, help="Number of model channels"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=4, help="Number of model blocks"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/photography_training",
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
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--pin_memory",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Pin memory for faster GPU transfer",
    )

    # Checkpointing arguments
    parser.add_argument(
        "--save_every", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--validate_every", type=int, default=1, help="Validate every N epochs"
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
    logger.info("🚀 INITIALIZING PHOTOGRAPHY TRAINING")
    logger.info("=" * 60)

    training_manager = PhotographyTrainingManager(
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )

    # Setup monitoring
    monitoring_dir = training_manager.setup_monitoring()

    # Create model
    model = training_manager.create_model(
        model_channels=args.model_channels, num_blocks=args.num_blocks
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
                self.train_dataset.domain_datasets = {"photography": self.train_dataset}
                self.val_dataset.domain_datasets = {"photography": self.val_dataset}

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

                # Resize if needed
                if clean.shape[-1] != self.target_size:
                    clean = torch.nn.functional.interpolate(
                        clean.unsqueeze(0), size=self.target_size, mode="bilinear"
                    ).squeeze(0)
                    noisy = torch.nn.functional.interpolate(
                        noisy.unsqueeze(0), size=self.target_size, mode="bilinear"
                    ).squeeze(0)

                return {
                    "clean": clean,
                    "noisy": noisy,
                    "electrons": clean,  # Use clean as target electrons
                    "domain": torch.tensor([0]),  # Photography domain
                    "metadata": {"synthetic": True},
                }

        dataset = SyntheticDataset(synthetic_config.output_dir, args.target_size)
    else:
        # Use real preprocessed data
        logger.info("📁 Loading real preprocessed photography data...")
        from data.preprocessed_datasets import create_preprocessed_datasets
        from utils.training_config import (
            calculate_optimal_training_config,
            print_training_analysis,
        )

        train_dataset, val_dataset = create_preprocessed_datasets(
            data_root=args.data_root,
            domain="photography",
            max_files=args.max_files,
            seed=args.seed,
        )

        # Calculate optimal training configuration
        dataset_size = len(train_dataset)
        optimal_config = calculate_optimal_training_config(
            dataset_size, args.batch_size
        )

        logger.info("🎯 AUTOMATIC TRAINING CONFIGURATION:")
        print_training_analysis(optimal_config)

        # Warn if current epochs are suboptimal
        if args.epochs < optimal_config["recommended_epochs"] * 0.5:
            logger.warning(f"⚠️  Current epochs ({args.epochs}) may be too low!")
            logger.warning(
                f"   Recommended: {optimal_config['recommended_epochs']} epochs"
            )
            logger.warning(f"   For optimal results, consider increasing epochs")
        elif args.epochs > optimal_config["recommended_epochs"] * 1.5:
            logger.warning(f"⚠️  Current epochs ({args.epochs}) may be too high!")
            logger.warning(
                f"   Recommended: {optimal_config['recommended_epochs']} epochs"
            )
        else:
            logger.info(f"✅ Current epochs ({args.epochs}) are reasonable")

        # Create a simple dataset wrapper with train/val attributes
        class DatasetWrapper:
            def __init__(self, train_ds, val_ds):
                self.train_dataset = train_ds
                self.val_dataset = val_ds

        dataset = DatasetWrapper(train_dataset, val_dataset)

    # Create training configuration
    config = training_manager.create_training_config(
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_frequency=args.save_every,
        val_frequency=args.validate_every,
    )

    # Save configuration
    config_file = training_manager.output_dir / "training_config.json"
    with open(config_file, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    logger.info(f"📝 Configuration saved to: {config_file}")

    # Start training
    logger.info("🎯 STARTING TRAINING")
    logger.info("=" * 60)

    try:
        results = training_manager.train(
            model=model,
            dataset=dataset,
            config=config,
            resume_from_checkpoint=args.resume_checkpoint,
        )

        # Success summary
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
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
        logger.error(f"❌ Training failed: {e}")
        logger.error("Check logs for details")
        raise


if __name__ == "__main__":
    main()
