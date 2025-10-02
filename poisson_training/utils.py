"""
Training utilities for deterministic and reproducible training.

This module provides utility functions for setting up deterministic training,
checkpointing, and other training-related functionality.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from core.logging_config import get_logger

logger = get_logger(__name__)


def set_deterministic_mode(seed: int = 42, benchmark: bool = False):
    """
    Set deterministic mode for reproducible training.

    Args:
        seed: Random seed
        benchmark: Whether to use cudnn benchmark (faster but non-deterministic)
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)

    # Set CUDA random seed (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = benchmark

    # Set environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Use deterministic algorithms (PyTorch >= 1.8)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info(f"Set deterministic mode with seed {seed}")


def save_checkpoint(
    checkpoint_data: Dict[str, Any],
    checkpoint_path: Union[str, Path],
    backup: bool = True,
) -> None:
    """
    Save training checkpoint.

    Args:
        checkpoint_data: Dictionary containing checkpoint data
        checkpoint_path: Path to save checkpoint
        backup: Whether to create backup of existing checkpoint
    """
    checkpoint_path = Path(checkpoint_path)

    # Create backup if file exists
    if backup and checkpoint_path.exists():
        backup_path = checkpoint_path.with_suffix(".bak")
        checkpoint_path.rename(backup_path)

    # Ensure directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Union[str, Path], device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to

    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint with PyTorch 2.6+ compatibility
    try:
        # First try with weights_only=True (secure mode)
        if device is None:
            checkpoint_data = torch.load(checkpoint_path, weights_only=True)
        else:
            checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        logger.warning(f"Failed to load with weights_only=True: {e}")
        logger.warning("Falling back to weights_only=False (trusted checkpoint)")
        
        # Fallback to weights_only=False for compatibility with older checkpoints
        # This is safe since we trust our own checkpoints
        if device is None:
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        else:
            checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint_data


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Get model size in MB.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model sizes
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size = param_size + buffer_size

    return {
        "parameters_mb": param_size / 1024 / 1024,
        "buffers_mb": buffer_size / 1024 / 1024,
        "total_mb": total_size / 1024 / 1024,
    }


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_device_info() -> Dict[str, Any]:
    """
    Get device information.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name()
        info["memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
        info["memory_cached"] = torch.cuda.memory_reserved() / 1024**3  # GB
        info["max_memory"] = torch.cuda.max_memory_allocated() / 1024**3  # GB

    return info


def log_system_info():
    """Log system and environment information."""
    import platform
    import sys

    logger.info("System Information:")
    logger.info(f"  Python version: {sys.version}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  CPU count: {os.cpu_count()}")

    device_info = get_device_info()
    logger.info(f"  CUDA available: {device_info['cuda_available']}")
    if device_info["cuda_available"]:
        logger.info(f"  GPU count: {device_info['device_count']}")
        logger.info(f"  Current GPU: {device_info['device_name']}")
        logger.info(
            f"  GPU memory: {device_info['memory_allocated']:.1f}GB allocated, "
            f"{device_info['memory_cached']:.1f}GB cached"
        )


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Create experiment directory with timestamp.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of experiment

    Returns:
        Path to created experiment directory
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Saved configuration to {config_path}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


class EarlyStopping:
    """
    Early stopping utility.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for monitoring metric
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.counter = 0
        self.best_weights = None

        if mode == "min":
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current score
            model: Model to potentially save weights from

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best weights")
                return True
            return False


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]["lr"]


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    Set learning rate for optimizer.

    Args:
        optimizer: PyTorch optimizer
        lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def clip_gradients(
    model: torch.nn.Module, max_norm: float, norm_type: float = 2.0
) -> float:
    """
    Clip gradients by norm.

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use

    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)


def freeze_model(model: torch.nn.Module):
    """
    Freeze all parameters in model.

    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: torch.nn.Module):
    """
    Unfreeze all parameters in model.

    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.

    Returns:
        Dictionary with memory usage in GB
    """
    import psutil

    # System memory
    memory = psutil.virtual_memory()

    result = {
        "system_total_gb": memory.total / 1024**3,
        "system_used_gb": memory.used / 1024**3,
        "system_available_gb": memory.available / 1024**3,
        "system_percent": memory.percent,
    }

    # GPU memory
    if torch.cuda.is_available():
        result.update(
            {
                "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_cached_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )

    return result
