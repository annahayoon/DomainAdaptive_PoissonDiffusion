"""
Deterministic training loop for Poisson-Gaussian diffusion restoration.

This module provides a comprehensive training framework with deterministic behavior,
reproducible results, and robust error handling for scientific reproducibility.

Key features:
- Deterministic training with fixed seeds
- Comprehensive logging and metrics
- Model checkpointing and resuming
- Validation loop with early stopping
- Memory-efficient training
- Integration with existing data pipeline

Requirements addressed: 8.1, 8.2, 8.3 from requirements.md
Task: 5.1 from tasks.md
"""

import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.error_handlers import ErrorHandler, safe_operation
from core.exceptions import TrainingError, ValidationError
from core.logging_config import get_logger
from poisson_training.losses import DiffusionLoss, PoissonGaussianLoss
from poisson_training.metrics import TrainingMetrics
from poisson_training.schedulers import get_scheduler
from poisson_training.utils import (
    load_checkpoint,
    save_checkpoint,
    set_deterministic_mode,
)

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    # Model and data
    model_name: str = "poisson_diffusion"
    dataset_path: str = "data/processed"

    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    max_steps: Optional[
        int
    ] = None  # Maximum training steps (overrides num_epochs if set)
    warmup_epochs: int = 5

    # Optimizer settings
    optimizer: str = "adamw"  # adamw, adam, sgd
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Scheduler settings
    scheduler: str = "cosine"  # cosine, linear, exponential, plateau
    min_lr: float = 1e-6
    scheduler_patience: int = 10

    # Loss function settings
    loss_type: str = "poisson_gaussian"  # poisson_gaussian, mse, l1
    loss_weights: Dict[str, float] = None

    # Regularization
    gradient_clip_norm: float = 1.0
    dropout_rate: float = 0.1

    # Validation and checkpointing
    val_frequency: int = 1  # Validate every N epochs
    val_frequency_steps: Optional[
        int
    ] = None  # Validate every N steps (overrides val_frequency if set)
    save_frequency_steps: int = (
        50000  # Save checkpoint every N steps (diffusion standard)
    )
    max_checkpoints: int = 5
    early_stopping_patience_steps: int = (
        100000  # Step-based patience (diffusion standard)
    )
    validation_checkpoints_patience: int = 20  # Validation checkpoint-based patience
    early_stopping_min_delta: float = 1e-4

    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False

    # Logging and monitoring
    log_frequency: int = 100  # Log every N steps
    tensorboard_log_dir: str = "logs/tensorboard"
    checkpoint_dir: str = "checkpoints"

    # Device and performance
    device: str = "auto"
    mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0 compilation
    num_workers: int = 8  # Increased for better I/O parallelism
    pin_memory: bool = True
    persistent_workers: bool = True  # Avoid worker respawning overhead
    prefetch_factor: int = 2  # Prefetch batches for better pipeline

    # Advanced settings
    accumulate_grad_batches: int = 1
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999  # Exponential moving average

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.loss_weights is None:
            self.loss_weights = {"reconstruction": 1.0, "consistency": 0.1}

        # Ensure directories exist
        Path(self.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(**data)

    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved training config to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load configuration from file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class DeterministicTrainer:
    """
    Deterministic training loop for Poisson-Gaussian diffusion restoration.

    This trainer ensures reproducible results through careful seed management
    and deterministic operations while providing comprehensive training features.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize deterministic trainer.

        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or TrainingConfig()

        # Setup device
        self.device = self._setup_device()
        self.model = self.model.to(self.device)

        # Setup deterministic mode
        if self.config.deterministic:
            set_deterministic_mode(self.config.seed, self.config.benchmark)

        # Initialize components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.loss_fn = self._setup_loss_function()
        self.metrics = TrainingMetrics()
        self.error_handler = ErrorHandler()

        # Setup logging (will be initialized lazily to avoid pickling issues)
        self.writer = None
        self._tensorboard_log_dir = self.config.tensorboard_log_dir

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        self.validation_checkpoint_counter = 0
        self.training_history = defaultdict(self._create_list)

        # Mixed precision
        if self.config.mixed_precision:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # Model compilation (PyTorch 2.0)
        if self.config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            logger.info("Model compiled with PyTorch 2.0")

        # EMA model
        if self.config.ema_decay > 0:
            self.ema_model = self._setup_ema_model()
        else:
            self.ema_model = None

        logger.info(f"Initialized DeterministicTrainer on {self.device}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        # Handle different dataloader types
        if hasattr(self.train_dataloader, "dataset"):
            logger.info(f"Training samples: {len(self.train_dataloader.dataset)}")
        else:
            logger.info(f"Training batches: {len(self.train_dataloader)}")

        if self.val_dataloader:
            if hasattr(self.val_dataloader, "dataset"):
                logger.info(f"Validation samples: {len(self.val_dataloader.dataset)}")
            else:
                logger.info(f"Validation batches: {len(self.val_dataloader)}")

    def _create_list(self):
        """Create a list for defaultdict factory (pickle-safe)."""
        return list()

    def _get_writer(self):
        """Lazily initialize SummaryWriter to avoid pickling issues."""
        if self.writer is None:
            self.writer = SummaryWriter(self._tensorboard_log_dir)
        return self.writer

    def __getstate__(self):
        """Custom pickle state to handle non-serializable objects."""
        state = self.__dict__.copy()
        # Remove unpickleable objects
        state["writer"] = None  # Will be recreated lazily
        return state

    def __setstate__(self, state):
        """Custom unpickle state to restore trainer."""
        self.__dict__.update(state)
        # writer will be created lazily when needed

    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)

        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = torch.device("cpu")

        return device

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        if self.config.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )
        elif self.config.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        return optimizer

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        return get_scheduler(
            self.optimizer,
            scheduler_type=self.config.scheduler,
            num_epochs=self.config.num_epochs,
            warmup_epochs=self.config.warmup_epochs,
            min_lr=self.config.min_lr,
            patience=self.config.scheduler_patience,
        )

    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function."""
        if self.config.loss_type == "poisson_gaussian":
            return PoissonGaussianLoss(weights=self.config.loss_weights)
        elif self.config.loss_type == "mse":
            return nn.MSELoss()
        elif self.config.loss_type == "l1":
            return nn.L1Loss()
        else:
            return DiffusionLoss(loss_type=self.config.loss_type)

    def _setup_ema_model(self) -> nn.Module:
        """Setup exponential moving average model."""
        try:
            # Try to create model with config if available
            if hasattr(self.model, "config"):
                ema_model = type(self.model)(self.model.config)
            else:
                # Try to create model with no arguments (for simple models)
                try:
                    ema_model = type(self.model)()
                except TypeError:
                    # If that fails, create a copy by loading state dict
                    import copy

                    ema_model = copy.deepcopy(self.model)
        except Exception as e:
            logger.warning(f"Failed to create EMA model: {e}, using deepcopy")
            import copy

            ema_model = copy.deepcopy(self.model)

        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        ema_model.eval()

        # Disable gradients for EMA model
        for param in ema_model.parameters():
            param.requires_grad = False

        return ema_model

    def _update_ema_model(self):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return

        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.data.mul_(self.config.ema_decay).add_(
                    model_param.data, alpha=1 - self.config.ema_decay
                )

    @safe_operation("Training step")
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform single training step.

        Args:
            batch: Training batch

        Returns:
            Dictionary of loss values
        """
        self.model.train()

        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Extract inputs for EDM model
        x = batch["noisy"].float()  # Noisy input image, ensure float32
        # Generate random noise levels for diffusion training
        sigma = (
            torch.rand(x.shape[0], device=x.device, dtype=torch.float32) * 10.0 + 0.1
        )  # Random sigma in [0.1, 10.1]

        # Forward pass with mixed precision
        if self.config.mixed_precision and self.scaler is not None:
            with torch.amp.autocast("cuda"):
                model_output = self.model(
                    x,
                    sigma,
                    domain="photography",
                    scale=1000.0,
                    read_noise=5.0,
                    background=10.0,
                )
                # Wrap output in expected format for loss function
                outputs = {"prediction": model_output, "denoised": model_output}
                loss_dict = self.loss_fn(outputs, batch)
                total_loss = sum(loss_dict.values())
        else:
            model_output = self.model(
                x,
                sigma,
                domain="photography",
                scale=1000.0,
                read_noise=5.0,
                background=10.0,
            )
            # Wrap output in expected format for loss function
            outputs = {"prediction": model_output, "denoised": model_output}
            loss_dict = self.loss_fn(outputs, batch)
            total_loss = sum(loss_dict.values())

        # Backward pass
        if self.config.accumulate_grad_batches > 1:
            total_loss = total_loss / self.config.accumulate_grad_batches

        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.scale(total_loss).backward()

            # Gradient accumulation
            if (self.global_step + 1) % self.config.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            total_loss.backward()

            # Gradient accumulation
            if (self.global_step + 1) % self.config.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

        # Update EMA model
        if (self.global_step + 1) % self.config.accumulate_grad_batches == 0:
            self._update_ema_model()

        # Convert losses to float for logging
        loss_dict = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }
        loss_dict["total_loss"] = total_loss.item()

        return loss_dict

    @safe_operation("Validation step")
    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform single validation step.

        Args:
            batch: Validation batch

        Returns:
            Dictionary of loss values
        """
        # Use EMA model for validation if available
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()

        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Extract inputs for EDM model (same as train_step)
        x = batch["noisy"].float()  # Noisy input image, ensure float32
        # Generate random noise levels for diffusion validation
        sigma = (
            torch.rand(x.shape[0], device=x.device, dtype=torch.float32) * 10.0 + 0.1
        )  # Random sigma in [0.1, 10.1]

        with torch.no_grad():
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    model_output = model(
                        x,
                        sigma,
                        domain="photography",
                        scale=1000.0,
                        read_noise=5.0,
                        background=10.0,
                    )
                    # Wrap output in expected format for loss function
                    outputs = {"prediction": model_output, "denoised": model_output}
                    loss_dict = self.loss_fn(outputs, batch)
            else:
                model_output = model(
                    x,
                    sigma,
                    domain="photography",
                    scale=1000.0,
                    read_noise=5.0,
                    background=10.0,
                )
                # Wrap output in expected format for loss function
                outputs = {"prediction": model_output, "denoised": model_output}
                loss_dict = self.loss_fn(outputs, batch)

        # Convert losses to float
        loss_dict = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }
        loss_dict["total_loss"] = sum(loss_dict.values())

        return loss_dict

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        epoch_losses = defaultdict(self._create_list)

        # Setup progress tracking
        num_batches = len(self.train_dataloader)
        log_interval = max(1, num_batches // 10)  # Log 10 times per epoch

        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                # Training step
                loss_dict = self.train_step(batch)

                # Accumulate losses
                for key, value in loss_dict.items():
                    epoch_losses[key].append(value)

                # Update metrics
                self.metrics.update(loss_dict, phase="train")

                # Logging
                if (
                    batch_idx % self.config.log_frequency == 0
                    or batch_idx % log_interval == 0
                ):
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch {self.current_epoch:3d} | "
                        f"Batch {batch_idx:4d}/{num_batches:4d} | "
                        f"Loss: {loss_dict['total_loss']:.4f} | "
                        f"LR: {lr:.2e}"
                    )

                    # TensorBoard logging
                    for key, value in loss_dict.items():
                        self._get_writer().add_scalar(
                            f"train/{key}", value, self.global_step
                        )
                    self._get_writer().add_scalar(
                        "train/learning_rate", lr, self.global_step
                    )

                self.global_step += 1

                # Step-based validation and checkpointing (if step-based validation is enabled)
                if self.config.val_frequency_steps is not None:
                    # Step-based validation
                    val_occurred = False
                    if (
                        self.val_dataloader
                        and self.global_step % self.config.val_frequency_steps == 0
                    ):
                        logger.info(
                            f"\n=== Step-based Validation at Step {self.global_step} ==="
                        )
                        val_losses = self.validate_epoch()
                        val_occurred = True

                        # Record validation history
                        for key, value in val_losses.items():
                            self.training_history[f"val_{key}"].append(value)

                        # Check if this is the best model
                        is_best = False
                        if val_losses:
                            val_loss = val_losses["total_loss"]
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                is_best = True
                                logger.info(
                                    f"🎯 New best validation loss: {val_loss:.6f}"
                                )

                        # Save checkpoint if it's the best model or regular checkpoint interval
                        if (
                            self.global_step % self.config.save_frequency_steps == 0
                            or is_best
                        ):
                            self.save_checkpoint(is_best=is_best)

                        # Log step-based validation results
                        log_data = {
                            "step": self.global_step,
                            "train_loss": loss_dict["total_loss"],
                            "learning_rate": lr,
                            "val_loss": val_losses.get("total_loss", 0.0),
                        }
                        self._save_step_log(log_data)

                        logger.info(
                            f"Step {self.global_step} | Train Loss: {loss_dict['total_loss']:.4f} | Val Loss: {val_losses.get('total_loss', 0.0):.4f}"
                        )

                    # Regular step-based checkpointing (without validation)
                    elif self.global_step % self.config.save_frequency_steps == 0:
                        logger.info(
                            f"\n=== Step-based Checkpoint at Step {self.global_step} ==="
                        )
                        self.save_checkpoint(is_best=False)

                        # Log step-based checkpoint (without validation)
                        log_data = {
                            "step": self.global_step,
                            "train_loss": loss_dict["total_loss"],
                            "learning_rate": lr,
                        }
                        self._save_step_log(log_data)

                        logger.info(
                            f"Step {self.global_step} | Train Loss: {loss_dict['total_loss']:.4f} | Checkpoint Saved"
                        )

            except Exception as e:
                logger.error(f"Error in training step {batch_idx}: {e}")
                if self.config.deterministic:
                    raise  # Re-raise in deterministic mode
                continue

        # Calculate epoch averages
        epoch_avg_losses = {
            key: np.mean(values) for key, values in epoch_losses.items()
        }

        return epoch_avg_losses

    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.

        Returns:
            Dictionary of average validation losses
        """
        if self.val_dataloader is None:
            return {}

        # Use EMA model for validation if available
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()

        epoch_losses = defaultdict(self._create_list)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                try:
                    # Validation step
                    loss_dict = self.val_step(batch)

                    # Accumulate losses
                    for key, value in loss_dict.items():
                        epoch_losses[key].append(value)

                    # Update metrics
                    self.metrics.update(loss_dict, phase="val")

                except Exception as e:
                    logger.error(f"Error in validation step {batch_idx}: {e}")
                    continue

        # Calculate epoch averages
        epoch_avg_losses = {
            key: np.mean(values) for key, values in epoch_losses.items()
        }

        # TensorBoard logging
        for key, value in epoch_avg_losses.items():
            self._get_writer().add_scalar(f"val/{key}", value, self.current_epoch)

        return epoch_avg_losses

    def should_early_stop(self, val_loss: float) -> bool:
        """
        Check if training should stop early.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return (
                self.early_stopping_counter >= self.config.early_stopping_patience_steps
            )

    def should_early_stop_checkpoints(self, val_loss: float) -> bool:
        """
        Check if training should stop early based on validation checkpoints.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.validation_checkpoint_counter = 0
            return False
        else:
            self.validation_checkpoint_counter += 1
            return (
                self.validation_checkpoint_counter
                >= self.config.validation_checkpoints_patience
            )

    def _save_step_log(self, log_data: Dict[str, Any]) -> None:
        """
        Save step-based training log to file.

        Args:
            log_data: Dictionary containing step log data
        """
        log_dir = Path(self.config.checkpoint_dir).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "training_steps.log"

        with open(log_file, "a") as f:
            f.write(
                f"{log_data['step']},{log_data['train_loss']:.6f},{log_data['learning_rate']:.8f}"
            )
            if "val_loss" in log_data:
                f.write(f",{log_data['val_loss']:.6f}")
            f.write("\n")

    def save_checkpoint(self, is_best: bool = False, extra_info: Optional[Dict] = None):
        """
        Save training checkpoint.

        Args:
            is_best: Whether this is the best checkpoint
            extra_info: Additional information to save
        """
        checkpoint_data = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_val_loss": self.best_val_loss,
            "early_stopping_counter": self.early_stopping_counter,
            "validation_checkpoint_counter": self.validation_checkpoint_counter,
            "config": self.config.to_dict(),
            "training_history": dict(self.training_history),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state()
            if torch.cuda.is_available()
            else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        # Add EMA model if available
        if self.ema_model is not None:
            checkpoint_data["ema_model_state_dict"] = self.ema_model.state_dict()

        # Add scaler state if using mixed precision
        if self.scaler is not None:
            checkpoint_data["scaler_state_dict"] = self.scaler.state_dict()

        # Add extra info
        if extra_info:
            checkpoint_data.update(extra_info)

        # Save checkpoint
        checkpoint_path = (
            Path(self.config.checkpoint_dir)
            / f"checkpoint_step_{self.global_step:08d}.pt"
        )
        save_checkpoint(checkpoint_data, checkpoint_path)

        # Save best model only
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            save_checkpoint(
                checkpoint_data, best_path, backup=False
            )  # Don't backup best model
            logger.info(f"Saved best model: {best_path}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Clean up old checkpoints to save disk space."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))

        if len(checkpoints) > self.config.max_checkpoints:
            for checkpoint in checkpoints[: -self.config.max_checkpoints]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")

    def load_checkpoint(
        self, checkpoint_path: Union[str, Path], resume_training: bool = True
    ):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            resume_training: Whether to resume training state
        """
        checkpoint_data = load_checkpoint(checkpoint_path, self.device)

        # Load model state
        self.model.load_state_dict(checkpoint_data["model_state_dict"])

        if resume_training:
            # Load training state
            self.current_epoch = checkpoint_data["epoch"]
            self.global_step = checkpoint_data["global_step"]
            self.best_val_loss = checkpoint_data["best_val_loss"]
            self.early_stopping_counter = checkpoint_data["early_stopping_counter"]
            self.validation_checkpoint_counter = checkpoint_data.get(
                "validation_checkpoint_counter", 0
            )
            # Restore training history with pickle-safe approach
            history_data = checkpoint_data.get("training_history", {})
            self.training_history = defaultdict(self._create_list)
            for key, values in history_data.items():
                self.training_history[key].extend(values)

            # Load optimizer and scheduler
            self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            if self.scheduler and checkpoint_data.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

            # Load EMA model
            if self.ema_model and "ema_model_state_dict" in checkpoint_data:
                self.ema_model.load_state_dict(checkpoint_data["ema_model_state_dict"])

            # Load scaler
            if self.scaler and "scaler_state_dict" in checkpoint_data:
                self.scaler.load_state_dict(checkpoint_data["scaler_state_dict"])

            # Restore RNG states for reproducibility
            if self.config.deterministic:
                torch.set_rng_state(checkpoint_data["rng_state"])
                if checkpoint_data.get("cuda_rng_state") and torch.cuda.is_available():
                    torch.cuda.set_rng_state(checkpoint_data["cuda_rng_state"])
                np.random.set_state(checkpoint_data["numpy_rng_state"])
                random.setstate(checkpoint_data["python_rng_state"])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def train_step_based(
        self, resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Step-based training loop.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training history dictionary
        """
        logger.info("Starting step-based training...")

        # Load checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

        # Calculate max steps
        if self.config.max_steps is not None:
            max_steps = self.config.max_steps
        else:
            # Calculate from epochs
            steps_per_epoch = len(self.train_dataloader)
            max_steps = self.config.num_epochs * steps_per_epoch

        logger.info(f"Training for {max_steps} steps")

        # Create infinite data iterator
        train_iterator = iter(self.train_dataloader)

        try:
            while self.global_step < max_steps:
                try:
                    # Get next batch (restart iterator if needed)
                    try:
                        batch = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(self.train_dataloader)
                        batch = next(train_iterator)
                        self.current_epoch += 1
                        logger.info(
                            f"Completed epoch {self.current_epoch}, continuing training..."
                        )

                    # Training step
                    self.model.train()
                    loss_dict = self.train_step(batch)

                    # Update metrics
                    self.metrics.update(loss_dict, phase="train")

                    # Logging
                    if self.global_step % self.config.log_frequency == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        logger.info(
                            f"Step {self.global_step:6d}/{max_steps:6d} | "
                            f"Epoch {self.current_epoch:3d} | "
                            f"Loss: {loss_dict['total_loss']:.4f} | "
                            f"LR: {lr:.2e}"
                        )

                        # TensorBoard logging
                        for key, value in loss_dict.items():
                            self._get_writer().add_scalar(
                                f"train/{key}", value, self.global_step
                            )
                        self._get_writer().add_scalar(
                            "train/learning_rate", lr, self.global_step
                        )

                    self.global_step += 1

                    # Step-based validation
                    if (
                        self.config.val_frequency_steps is not None
                        and self.val_dataloader
                        and self.global_step % self.config.val_frequency_steps == 0
                    ):
                        logger.info(f"\n=== Validation at Step {self.global_step} ===")
                        val_losses = self.validate_epoch()

                        # Record validation history
                        for key, value in val_losses.items():
                            self.training_history[f"val_{key}"].append(value)

                        # Check if this is the best model
                        is_best = False
                        if val_losses:
                            val_loss = val_losses["total_loss"]
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                is_best = True
                                logger.info(
                                    f"🎯 New best validation loss: {val_loss:.6f}"
                                )

                        # Learning rate scheduling (if using validation-based scheduler)
                        if self.scheduler and isinstance(
                            self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            self.scheduler.step(val_losses["total_loss"])

                        # Log validation results
                        log_data = {
                            "step": self.global_step,
                            "train_loss": loss_dict["total_loss"],
                            "learning_rate": lr,
                            "val_loss": val_losses.get("total_loss", 0.0),
                        }
                        self._save_step_log(log_data)

                        logger.info(
                            f"Step {self.global_step} | Train: {loss_dict['total_loss']:.4f} | Val: {val_losses.get('total_loss', 0.0):.4f}"
                        )

                        # Save checkpoint if it's the best or regular interval
                        if (
                            self.global_step % self.config.save_frequency_steps == 0
                            or is_best
                        ):
                            self.save_checkpoint(is_best=is_best)

                    # Regular step-based checkpointing (without validation)
                    elif self.global_step % self.config.save_frequency_steps == 0:
                        logger.info(f"\n=== Checkpoint at Step {self.global_step} ===")
                        self.save_checkpoint(is_best=False)

                        # Log checkpoint
                        log_data = {
                            "step": self.global_step,
                            "train_loss": loss_dict["total_loss"],
                            "learning_rate": lr,
                        }
                        self._save_step_log(log_data)

                    # Non-validation-based learning rate scheduling
                    if self.scheduler and not isinstance(
                        self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step()

                except Exception as e:
                    logger.error(f"Error in training step {self.global_step}: {e}")
                    if self.config.deterministic:
                        raise
                    continue

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(is_best=False, extra_info={"interrupted": True})

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.save_checkpoint(is_best=False, extra_info={"error": str(e)})
            raise

        logger.info("Step-based training completed successfully!")
        return self.training_history

    def train(
        self, resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Main training loop. Uses step-based training if val_frequency_steps is set.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training history
        """
        # Use step-based training if step-based validation is configured
        if (
            self.config.val_frequency_steps is not None
            or self.config.max_steps is not None
        ):
            return self.train_step_based(resume_from_checkpoint)

        logger.info("Starting epoch-based training...")
        logger.info(f"Configuration: {self.config}")

        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint, resume_training=True)
            logger.info(f"Resumed training from epoch {self.current_epoch}")

        # Save initial configuration
        config_path = Path(self.config.checkpoint_dir) / "training_config.json"
        self.config.save(config_path)

        try:
            for epoch in range(self.current_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
                logger.info("-" * 50)

                # Training phase
                train_losses = self.train_epoch()

                # Validation phase (only if step-based validation is not enabled)
                val_losses = {}
                if (
                    self.val_dataloader
                    and self.config.val_frequency_steps is None
                    and (epoch + 1) % self.config.val_frequency == 0
                ):
                    val_losses = self.validate_epoch()

                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        val_loss = val_losses.get(
                            "total_loss", train_losses["total_loss"]
                        )
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Record training history
                for key, value in train_losses.items():
                    self.training_history[f"train_{key}"].append(value)
                for key, value in val_losses.items():
                    self.training_history[f"val_{key}"].append(value)

                # Epoch summary
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
                logger.info(f"Train Loss: {train_losses['total_loss']:.4f}")
                if val_losses:
                    logger.info(f"Val Loss: {val_losses['total_loss']:.4f}")

                # Check if this is the best model (for epoch-based validation only)
                # Note: When step-based validation is enabled, best model tracking happens in train_epoch()
                is_best = False
                if val_losses and self.config.val_frequency_steps is None:
                    val_loss = val_losses["total_loss"]
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        is_best = True

                # Step-based checkpointing (diffusion standard)
                # Only save checkpoint here if step-based validation is NOT enabled
                # (when step-based validation is enabled, checkpointing happens in train_epoch())
                if self.config.val_frequency_steps is None and (
                    self.global_step % self.config.save_frequency_steps == 0 or is_best
                ):
                    self.save_checkpoint(is_best=is_best)

                # Step-based logging every 5000 steps (diffusion standard)
                if self.global_step % 5000 == 0:
                    log_data = {
                        "step": self.global_step,
                        "train_loss": train_losses["total_loss"],
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                    if val_losses:
                        log_data["val_loss"] = val_losses["total_loss"]

                    # Log to console
                    log_msg = f"Step {self.global_step:8d} | "
                    log_msg += f"Train Loss: {train_losses['total_loss']:.4f} | "
                    log_msg += f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                    if val_losses:
                        log_msg += f" | Val Loss: {val_losses['total_loss']:.4f}"
                    logger.info(log_msg)

                    # Save to log file
                    self._save_step_log(log_data)

                # Early stopping (validation checkpoint-based)
                if val_losses and self.should_early_stop_checkpoints(
                    val_losses["total_loss"]
                ):
                    logger.info(
                        f"Early stopping triggered after {self.validation_checkpoint_counter} validation checkpoints"
                    )
                    logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
                    break

                # TensorBoard logging
                self._get_writer().add_scalar(
                    "epoch/train_loss", train_losses["total_loss"], epoch
                )
                if val_losses:
                    self._get_writer().add_scalar(
                        "epoch/val_loss", val_losses["total_loss"], epoch
                    )
                self._get_writer().add_scalar(
                    "epoch/learning_rate", self.optimizer.param_groups[0]["lr"], epoch
                )

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(is_best=False, extra_info={"interrupted": True})

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            self.save_checkpoint(is_best=False, extra_info={"error": str(e)})
            raise

        finally:
            # Final cleanup
            if self.writer is not None:
                self.writer.close()
            logger.info("Training completed")

        return dict(self.training_history)

    def get_model_for_inference(self) -> nn.Module:
        """
        Get model for inference (EMA model if available, otherwise main model).

        Returns:
            Model ready for inference
        """
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()
        return model


# Utility functions for training
def create_trainer(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    config_path: Optional[str] = None,
    **config_kwargs,
) -> DeterministicTrainer:
    """
    Create trainer with configuration.

    Args:
        model: Model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        config_path: Path to configuration file
        **config_kwargs: Additional configuration parameters

    Returns:
        Configured trainer
    """
    if config_path:
        config = TrainingConfig.load(config_path)
        # Override with any provided kwargs
        for key, value in config_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        config = TrainingConfig(**config_kwargs)

    return DeterministicTrainer(model, train_dataloader, val_dataloader, config)


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None,
    resume_from: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    High-level training function.

    Args:
        model: Model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        config: Training configuration
        resume_from: Checkpoint to resume from

    Returns:
        (trained_model, training_history) tuple
    """
    trainer = DeterministicTrainer(model, train_dataloader, val_dataloader, config)
    history = trainer.train(resume_from_checkpoint=resume_from)
    trained_model = trainer.get_model_for_inference()

    return trained_model, history
