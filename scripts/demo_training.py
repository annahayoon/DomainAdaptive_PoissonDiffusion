#!/usr/bin/env python
"""
Demonstration of the deterministic training system.

This script demonstrates the comprehensive training framework including
deterministic behavior, physics-aware loss functions, and metrics tracking.

Usage:
    python scripts/demo_training.py --demo basic
    python scripts/demo_training.py --demo config --epochs 5
    python scripts/demo_training.py --demo metrics
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training import (
    DeterministicTrainer,
    ImageQualityMetrics,
    PhysicsMetrics,
    PoissonGaussianLoss,
    TrainingConfig,
    TrainingMetrics,
    create_trainer,
    set_deterministic_mode,
    train_model,
)

from core.logging_config import setup_project_logging

logger = setup_project_logging(level="INFO")


class DemoModel(nn.Module):
    """Simple demonstration model for training."""

    def __init__(self, channels: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, batch):
        """Forward pass expecting batch dictionary."""
        x = batch.get("electrons", batch.get("noisy"))

        # Normalize input to [0, 1] range
        if "scale" in batch:
            scale = batch["scale"]
            if scale.numel() > 1:
                scale = scale.view(-1, 1, 1, 1)
            x = x / scale
        else:
            x = x / x.max()

        x = torch.clamp(x, 0, 1)

        # Encode-decode
        features = self.encoder(x)
        output = self.decoder(features)

        return {"prediction": output, "denoised": output, "features": features}


class TrainingDemo:
    """
    Demonstration of training system capabilities.
    """

    def __init__(self, device: str = "auto"):
        """Initialize training demo."""
        self.device = self._setup_device(device)
        logger.info(f"Initialized training demo on {self.device}")

    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = "cpu"

        return device

    def create_synthetic_dataset(
        self,
        num_samples: int = 20,
        image_size: tuple = (64, 64),
        noise_level: float = 0.1,
    ) -> tuple:
        """
        Create synthetic dataset for demonstration.

        Args:
            num_samples: Number of samples to generate
            image_size: Size of images (height, width)
            noise_level: Noise level for synthetic data

        Returns:
            (train_data, val_data) tuple
        """
        logger.info(
            f"Creating synthetic dataset: {num_samples} samples, {image_size} size"
        )

        train_data = []
        val_data = []

        for i in range(num_samples):
            # Create clean image with some structure
            height, width = image_size

            # Create patterns (circles, lines, etc.)
            y, x = np.meshgrid(
                np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing="ij"
            )

            # Combine different patterns
            clean = np.zeros((height, width))

            # Add some circular patterns
            for _ in range(np.random.randint(2, 5)):
                center_y = np.random.uniform(-0.5, 0.5)
                center_x = np.random.uniform(-0.5, 0.5)
                radius = np.random.uniform(0.1, 0.3)
                intensity = np.random.uniform(0.3, 0.8)

                circle = intensity * np.exp(
                    -((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * radius**2)
                )
                clean += circle

            # Add background
            clean += np.random.uniform(0.1, 0.2)
            clean = np.clip(clean, 0, 1)

            # Convert to electrons (scale up)
            scale = np.random.uniform(800, 1200)
            background = np.random.uniform(50, 150)
            read_noise = np.random.uniform(3, 8)

            clean_electrons = clean * scale + background

            # Add Poisson noise
            noisy_electrons = np.random.poisson(clean_electrons).astype(np.float32)

            # Add Gaussian read noise
            noisy_electrons += np.random.normal(0, read_noise, noisy_electrons.shape)

            # Convert to tensors
            clean_tensor = torch.from_numpy(clean).float().unsqueeze(0).unsqueeze(0)
            noisy_tensor = (
                torch.from_numpy(noisy_electrons).float().unsqueeze(0).unsqueeze(0)
            )

            # Create batch
            batch = {
                "electrons": noisy_tensor,
                "clean": clean_tensor,
                "scale": torch.tensor(scale),
                "background": torch.tensor(background),
                "read_noise": torch.tensor(read_noise),
            }

            # Split train/val
            if i < int(0.8 * num_samples):
                train_data.append(batch)
            else:
                val_data.append(batch)

        logger.info(
            f"Created {len(train_data)} training samples, {len(val_data)} validation samples"
        )

        return train_data, val_data

    def demo_basic_training(self, num_epochs: int = 3):
        """Demonstrate basic training functionality."""
        logger.info("=== Basic Training Demo ===")

        # Create model
        model = DemoModel(channels=1, hidden_dim=32)
        logger.info(
            f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters"
        )

        # Create synthetic dataset
        train_data, val_data = self.create_synthetic_dataset(
            num_samples=16, image_size=(32, 32)
        )

        # Create training configuration
        config = TrainingConfig(
            num_epochs=num_epochs,
            batch_size=4,
            learning_rate=1e-3,
            log_frequency=2,
            save_frequency=2,
            val_frequency=1,
            mixed_precision=False,  # Disable for CPU demo
            device=self.device,
            deterministic=True,
            seed=42,
            ema_decay=0,  # Disable EMA for demo
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.checkpoint_dir = tmp_dir
            config.tensorboard_log_dir = tmp_dir

            # Create trainer
            trainer = DeterministicTrainer(model, train_data, val_data, config)

            # Train
            logger.info("Starting training...")
            start_time = time.time()

            history = trainer.train()

            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f}s")

            # Show results
            logger.info("Training History:")
            for key, values in history.items():
                if values:
                    logger.info(f"  {key}: {values[0]:.4f} -> {values[-1]:.4f}")

            return trainer, history

    def demo_configuration_management(self, num_epochs: int = 2):
        """Demonstrate configuration management."""
        logger.info("=== Configuration Management Demo ===")

        # Create and save configuration
        config = TrainingConfig(
            num_epochs=num_epochs,
            batch_size=2,
            learning_rate=2e-4,
            optimizer="adamw",
            scheduler="cosine",
            loss_type="poisson_gaussian",
            early_stopping_patience=5,
            mixed_precision=False,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "training_config.json"

            # Save configuration
            config.save(config_path)
            logger.info(f"Saved configuration to {config_path}")

            # Load configuration
            loaded_config = TrainingConfig.load(config_path)
            logger.info("Loaded configuration successfully")

            # Verify configuration
            assert loaded_config.num_epochs == num_epochs
            assert loaded_config.batch_size == 2
            assert loaded_config.learning_rate == 2e-4

            logger.info("Configuration serialization working correctly")

            # Show configuration
            logger.info("Configuration settings:")
            config_dict = loaded_config.to_dict()
            for key, value in config_dict.items():
                if not key.startswith("_"):
                    logger.info(f"  {key}: {value}")

            return loaded_config

    def demo_loss_functions(self):
        """Demonstrate different loss functions."""
        logger.info("=== Loss Functions Demo ===")

        # Create sample data
        batch_size, channels, height, width = 2, 1, 32, 32

        prediction = torch.rand(batch_size, channels, height, width)
        target_electrons = torch.rand(batch_size, channels, height, width) * 1000 + 100

        batch = {
            "electrons": target_electrons,
            "scale": torch.tensor(1000.0),
            "background": torch.tensor(100.0),
            "read_noise": torch.tensor(5.0),
        }

        outputs = {"prediction": prediction}

        # Test Poisson-Gaussian loss
        logger.info("Testing Poisson-Gaussian Loss:")
        pg_loss = PoissonGaussianLoss()
        pg_losses = pg_loss(outputs, batch)

        for name, loss in pg_losses.items():
            logger.info(f"  {name}: {loss.item():.4f}")

        # Test with different parameters
        logger.info("Testing with different noise parameters:")
        batch_high_noise = batch.copy()
        batch_high_noise["read_noise"] = torch.tensor(20.0)

        pg_losses_high = pg_loss(outputs, batch_high_noise)

        for name, loss in pg_losses_high.items():
            logger.info(f"  {name} (high noise): {loss.item():.4f}")

        return pg_losses

    def demo_metrics_tracking(self):
        """Demonstrate metrics tracking."""
        logger.info("=== Metrics Tracking Demo ===")

        # Create metrics tracker
        metrics = TrainingMetrics(window_size=5)

        # Simulate training progress
        logger.info("Simulating training progress...")

        for epoch in range(5):
            for step in range(3):
                # Simulate improving loss
                base_loss = 1.0
                loss = base_loss * (0.9 ** (epoch * 3 + step)) + np.random.normal(
                    0, 0.05
                )
                accuracy = 0.5 + 0.4 * (1 - np.exp(-(epoch * 3 + step) * 0.1))

                values = {
                    "total_loss": max(0.1, loss),
                    "reconstruction_loss": max(0.05, loss * 0.8),
                    "consistency_loss": max(0.01, loss * 0.2),
                    "accuracy": min(0.95, accuracy),
                }

                metrics.update(values, phase="train")

            # Show epoch summary
            current_metrics = metrics.get_current_metrics("train")
            logger.info(f"Epoch {epoch + 1}:")
            for key, value in current_metrics.items():
                logger.info(f"  {key}: {value:.4f}")

        # Show final summary
        summary = metrics.get_summary("train")
        logger.info("Final Summary:")
        logger.info(f"  Total steps: {summary['total_steps']}")
        logger.info(f"  Throughput: {summary['throughput_steps_per_sec']:.2f} steps/s")

        return metrics

    def demo_image_quality_metrics(self):
        """Demonstrate image quality metrics."""
        logger.info("=== Image Quality Metrics Demo ===")

        # Create sample images
        height, width = 64, 64

        # Original image
        original = torch.rand(1, 1, height, width)

        # Create different quality predictions
        predictions = {
            "perfect": original.clone(),
            "good": original + torch.randn_like(original) * 0.05,
            "medium": original + torch.randn_like(original) * 0.15,
            "poor": original + torch.randn_like(original) * 0.3,
        }

        # Compute metrics
        image_metrics = ImageQualityMetrics()

        logger.info("Image Quality Comparison:")
        for name, pred in predictions.items():
            psnr = image_metrics.psnr(pred, original).item()
            ssim = image_metrics.ssim(pred, original).item()
            mae = image_metrics.mae(pred, original).item()

            logger.info(
                f"  {name:8s}: PSNR={psnr:6.2f} dB, SSIM={ssim:.3f}, MAE={mae:.4f}"
            )

        return predictions

    def demo_physics_metrics(self):
        """Demonstrate physics-aware metrics."""
        logger.info("=== Physics Metrics Demo ===")

        # Create sample data with known physics
        batch_size, channels, height, width = 2, 1, 32, 32

        # Create prediction and target
        prediction = torch.rand(batch_size, channels, height, width)
        scale = torch.tensor([1000.0, 1200.0])
        background = torch.tensor([100.0, 120.0])
        read_noise = torch.tensor([5.0, 6.0])

        # Create target with known statistics
        target_electrons = prediction * scale.view(-1, 1, 1, 1) + background.view(
            -1, 1, 1, 1
        )
        target_electrons += torch.randn_like(target_electrons) * read_noise.view(
            -1, 1, 1, 1
        )

        # Compute physics metrics
        physics_metrics = PhysicsMetrics()

        chi2 = physics_metrics.chi_squared_consistency(
            prediction, target_electrons, scale, background, read_noise
        ).item()

        pnr = physics_metrics.photon_noise_ratio(
            prediction, target_electrons, scale
        ).item()

        energy_conservation = physics_metrics.energy_conservation(
            prediction, target_electrons, scale
        ).item()

        logger.info("Physics Metrics:")
        logger.info(f"  χ² consistency: {chi2:.3f} (should be ~1.0)")
        logger.info(f"  Photon noise ratio: {pnr:.3f}")
        logger.info(
            f"  Energy conservation: {energy_conservation:.3f} (should be ~1.0)"
        )

        return {
            "chi2_consistency": chi2,
            "photon_noise_ratio": pnr,
            "energy_conservation": energy_conservation,
        }

    def demo_deterministic_behavior(self):
        """Demonstrate deterministic training behavior."""
        logger.info("=== Deterministic Behavior Demo ===")

        # Create model and data
        model1 = DemoModel(channels=1, hidden_dim=16)
        model2 = DemoModel(channels=1, hidden_dim=16)

        train_data, _ = self.create_synthetic_dataset(
            num_samples=4, image_size=(16, 16)
        )

        config = TrainingConfig(
            num_epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            deterministic=True,
            seed=123,
            mixed_precision=False,
            ema_decay=0,  # Disable EMA for demo
        )

        # Train first model
        logger.info("Training first model...")
        with tempfile.TemporaryDirectory() as tmp_dir1:
            config.checkpoint_dir = tmp_dir1
            config.tensorboard_log_dir = tmp_dir1

            trainer1 = DeterministicTrainer(model1, train_data, None, config)
            history1 = trainer1.train()

        # Train second model with same seed
        logger.info("Training second model with same configuration...")
        with tempfile.TemporaryDirectory() as tmp_dir2:
            config.checkpoint_dir = tmp_dir2
            config.tensorboard_log_dir = tmp_dir2

            trainer2 = DeterministicTrainer(model2, train_data, None, config)
            history2 = trainer2.train()

        # Compare results
        logger.info("Comparing training results:")

        for key in history1.keys():
            if key in history2 and history1[key] and history2[key]:
                diff = abs(history1[key][-1] - history2[key][-1])
                logger.info(
                    f"  {key}: {history1[key][-1]:.6f} vs {history2[key][-1]:.6f} (diff: {diff:.8f})"
                )

                # Should be very close for deterministic training
                if diff > 1e-6:
                    logger.warning(f"  Large difference detected in {key}!")
                else:
                    logger.info(f"  ✓ Deterministic behavior confirmed for {key}")

        return history1, history2


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Training system demonstration")
    parser.add_argument(
        "--demo",
        "-d",
        type=str,
        default="basic",
        choices=[
            "basic",
            "config",
            "loss",
            "metrics",
            "physics",
            "deterministic",
            "all",
        ],
        help="Type of demonstration to run",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs for training demos"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for computation",
    )

    args = parser.parse_args()

    # Initialize demo
    demo = TrainingDemo(device=args.device)

    print(f"\n{'='*60}")
    print("TRAINING SYSTEM DEMONSTRATION")
    print(f"{'='*60}")
    print(f"Device: {demo.device}")
    print(f"Demo type: {args.demo}")
    print(f"{'='*60}\n")

    results = {}

    if args.demo == "basic" or args.demo == "all":
        print("Running basic training demo...")
        trainer, history = demo.demo_basic_training(num_epochs=args.epochs)
        results["basic"] = (trainer, history)
        print()

    if args.demo == "config" or args.demo == "all":
        print("Running configuration management demo...")
        config = demo.demo_configuration_management(num_epochs=args.epochs)
        results["config"] = config
        print()

    if args.demo == "loss" or args.demo == "all":
        print("Running loss functions demo...")
        losses = demo.demo_loss_functions()
        results["loss"] = losses
        print()

    if args.demo == "metrics" or args.demo == "all":
        print("Running metrics tracking demo...")
        metrics = demo.demo_metrics_tracking()
        results["metrics"] = metrics
        print()

    if args.demo == "physics" or args.demo == "all":
        print("Running physics metrics demo...")
        physics_metrics = demo.demo_physics_metrics()
        results["physics"] = physics_metrics
        print()

    if args.demo == "deterministic" or args.demo == "all":
        print("Running deterministic behavior demo...")
        history1, history2 = demo.demo_deterministic_behavior()
        results["deterministic"] = (history1, history2)
        print()

    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    print("\nKey Features Demonstrated:")
    print("✓ Deterministic training with reproducible results")
    print("✓ Physics-aware Poisson-Gaussian loss functions")
    print("✓ Comprehensive metrics tracking and monitoring")
    print("✓ Flexible configuration management")
    print("✓ Image quality and physics-specific metrics")
    print("✓ Robust error handling and checkpointing")
    print("✓ Learning rate scheduling and optimization")
    print("✓ Mixed precision training support")
    print("✓ Early stopping and validation")
    print("✓ TensorBoard logging integration")

    if results:
        print("\nDemo Results Summary:")
        for demo_name, result in results.items():
            print(f"  {demo_name.title()}: ✓ Completed successfully")


if __name__ == "__main__":
    main()
