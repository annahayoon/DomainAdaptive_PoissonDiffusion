"""
Memory optimization utilities for automatic configuration based on available GPU memory.

This module provides utilities to:
1. Detect available GPU memory
2. Automatically configure batch sizes and model parameters
3. Enable gradient checkpointing when needed
4. Optimize data loading parameters
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psutil
import torch
import yaml

from core.logging_config import get_logger

logger = get_logger(__name__)


class MemoryOptimizer:
    """Automatically optimize configuration based on available memory."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize memory optimizer.

        Args:
            config_path: Path to memory optimization config file
        """
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent / "configs" / "memory_optimized.yaml"
            )

        with open(config_path, "r") as f:
            self.memory_configs = yaml.safe_load(f)

        self.gpu_memory_gb = self._detect_gpu_memory()
        self.system_memory_gb = self._detect_system_memory()

        logger.info(f"Detected GPU memory: {self.gpu_memory_gb:.1f} GB")
        logger.info(f"Detected system memory: {self.system_memory_gb:.1f} GB")

    def _detect_gpu_memory(self) -> float:
        """Detect available GPU memory in GB."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU configuration")
            return 0.0

        try:
            # Get memory of the current GPU
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            return total_memory / (1024**3)  # Convert to GB
        except Exception as e:
            logger.error(f"Failed to detect GPU memory: {e}")
            return 8.0  # Default to 8GB as conservative estimate

    def _detect_system_memory(self) -> float:
        """Detect available system memory in GB."""
        try:
            return psutil.virtual_memory().total / (1024**3)
        except Exception as e:
            logger.error(f"Failed to detect system memory: {e}")
            return 16.0  # Default to 16GB

    def get_optimal_config(self) -> Dict[str, Any]:
        """
        Get optimal configuration based on detected memory.

        Returns:
            Optimized configuration dictionary
        """
        # Find appropriate config based on GPU memory
        config_name = self._select_config_by_memory()
        base_config = self.memory_configs[config_name].copy()

        # Apply safety margin
        safety_margin = self.memory_configs["auto_detect"]["safety_margin"]
        self._apply_safety_margin(base_config, safety_margin)

        # Adjust for system memory constraints
        self._adjust_for_system_memory(base_config)

        logger.info(f"Selected configuration: {config_name} (with adjustments)")
        logger.info(f"Batch size: {base_config['training']['batch_size']}")
        logger.info(
            f"Gradient accumulation: {base_config['training']['accumulate_grad_batches']}"
        )
        logger.info(f"Mixed precision: {base_config['training']['mixed_precision']}")

        return base_config

    def _select_config_by_memory(self) -> str:
        """Select configuration based on GPU memory."""
        thresholds = self.memory_configs["auto_detect"]["thresholds"]

        # Sort thresholds by memory requirement
        thresholds = sorted(thresholds, key=lambda x: x["memory_gb"])

        # Find the largest config that fits in memory
        selected_config = "gpu_8gb"  # Default to smallest
        for threshold in thresholds:
            if self.gpu_memory_gb >= threshold["memory_gb"]:
                selected_config = threshold["config"]
            else:
                break

        return selected_config

    def _apply_safety_margin(self, config: Dict[str, Any], safety_margin: float):
        """Apply safety margin to batch size."""
        current_batch_size = config["training"]["batch_size"]
        safe_batch_size = max(1, int(current_batch_size * safety_margin))

        if safe_batch_size != current_batch_size:
            # Adjust gradient accumulation to maintain effective batch size
            accumulate_grad_batches = config["training"].get(
                "accumulate_grad_batches", 1
            )
            effective_batch_size = current_batch_size * accumulate_grad_batches

            new_accumulate = max(1, effective_batch_size // safe_batch_size)

            config["training"]["batch_size"] = safe_batch_size
            config["training"]["accumulate_grad_batches"] = new_accumulate

            logger.info(
                f"Applied safety margin: batch_size {current_batch_size} -> {safe_batch_size}"
            )

    def _adjust_for_system_memory(self, config: Dict[str, Any]):
        """Adjust data loading parameters based on system memory."""
        # Reduce num_workers if system memory is low
        if self.system_memory_gb < 16:
            config["data"]["num_workers"] = min(config["data"]["num_workers"], 4)
            config["data"]["prefetch_factor"] = 2
            logger.info("Reduced data loading workers due to low system memory")

        # Disable persistent workers if memory is very low
        if self.system_memory_gb < 8:
            config["data"]["persistent_workers"] = False
            config["data"]["num_workers"] = min(config["data"]["num_workers"], 2)
            logger.info("Disabled persistent workers due to very low system memory")

    def estimate_memory_usage(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate memory usage for a given configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary with memory usage estimates in GB
        """
        # Model parameters estimation
        model_channels = config["model"]["model_channels"]
        channel_mult = config["model"]["channel_mult"]

        # Rough estimation based on EDM architecture
        # This is a simplified calculation
        total_params = 0
        current_channels = model_channels

        for mult in channel_mult:
            next_channels = model_channels * mult
            # Conv layers, attention, etc.
            total_params += current_channels * next_channels * 9  # 3x3 conv
            total_params += next_channels * 4  # Normalization and bias
            current_channels = next_channels

        # Convert to memory (4 bytes per float32 parameter)
        model_memory_gb = (total_params * 4) / (1024**3)

        # Activation memory estimation
        batch_size = config["training"]["batch_size"]
        img_resolution = config.get("model", {}).get("img_resolution", 128)

        # Rough activation memory (depends on architecture depth)
        activation_memory_gb = (
            batch_size
            * img_resolution
            * img_resolution
            * sum(channel_mult)
            * model_channels
            * 4
        ) / (1024**3)

        # Gradient memory (same as model parameters)
        gradient_memory_gb = model_memory_gb

        # Optimizer state (AdamW has 2x parameter memory)
        optimizer_memory_gb = model_memory_gb * 2

        total_memory_gb = (
            model_memory_gb
            + activation_memory_gb
            + gradient_memory_gb
            + optimizer_memory_gb
        )

        return {
            "model_parameters": model_memory_gb,
            "activations": activation_memory_gb,
            "gradients": gradient_memory_gb,
            "optimizer_state": optimizer_memory_gb,
            "total_estimated": total_memory_gb,
        }

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate if a configuration will fit in available memory.

        Args:
            config: Configuration to validate

        Returns:
            (is_valid, message) tuple
        """
        memory_usage = self.estimate_memory_usage(config)
        total_estimated = memory_usage["total_estimated"]

        # Add some buffer for PyTorch overhead
        buffer_factor = 1.2
        total_with_buffer = total_estimated * buffer_factor

        if total_with_buffer > self.gpu_memory_gb:
            return False, (
                f"Estimated memory usage ({total_with_buffer:.1f} GB) exceeds "
                f"available GPU memory ({self.gpu_memory_gb:.1f} GB)"
            )

        return (
            True,
            f"Configuration should fit in {self.gpu_memory_gb:.1f} GB GPU memory",
        )

    def suggest_improvements(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest improvements to reduce memory usage.

        Args:
            config: Current configuration

        Returns:
            Improved configuration
        """
        improved_config = config.copy()

        # Check if current config fits
        is_valid, message = self.validate_config(config)
        if is_valid:
            return improved_config

        logger.info(f"Current config doesn't fit: {message}")
        logger.info("Suggesting improvements...")

        # Reduce batch size
        current_batch = improved_config["training"]["batch_size"]
        if current_batch > 1:
            new_batch = max(1, current_batch // 2)
            # Increase gradient accumulation to maintain effective batch size
            current_accum = improved_config["training"].get(
                "accumulate_grad_batches", 1
            )
            new_accum = current_accum * 2

            improved_config["training"]["batch_size"] = new_batch
            improved_config["training"]["accumulate_grad_batches"] = new_accum
            logger.info(f"Reduced batch size: {current_batch} -> {new_batch}")

        # Enable gradient checkpointing
        if not improved_config["training"].get("gradient_checkpointing", False):
            improved_config["training"]["gradient_checkpointing"] = True
            logger.info("Enabled gradient checkpointing")

        # Reduce model size if still doesn't fit
        is_valid, _ = self.validate_config(improved_config)
        if not is_valid:
            current_channels = improved_config["model"]["model_channels"]
            new_channels = max(64, int(current_channels * 0.8))
            improved_config["model"]["model_channels"] = new_channels
            logger.info(f"Reduced model channels: {current_channels} -> {new_channels}")

        return improved_config


def get_auto_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get automatically optimized configuration.

    Args:
        config_path: Path to memory optimization config file

    Returns:
        Optimized configuration dictionary
    """
    optimizer = MemoryOptimizer(config_path)
    return optimizer.get_optimal_config()


def validate_memory_config(config: Dict[str, Any]) -> bool:
    """
    Validate if configuration will fit in available memory.

    Args:
        config: Configuration to validate

    Returns:
        True if configuration should fit
    """
    optimizer = MemoryOptimizer()
    is_valid, message = optimizer.validate_config(config)

    if not is_valid:
        logger.warning(message)
    else:
        logger.info(message)

    return is_valid


if __name__ == "__main__":
    # Demo usage
    optimizer = MemoryOptimizer()
    config = optimizer.get_optimal_config()

    print("Optimal configuration:")
    print(yaml.dump(config, default_flow_style=False))

    memory_usage = optimizer.estimate_memory_usage(config)
    print(f"\nEstimated memory usage:")
    for key, value in memory_usage.items():
        print(f"  {key}: {value:.2f} GB")
