"""
Checkpoint utilities for model saving and loading.

This module provides functions for saving and loading model checkpoints
with proper error handling and device mapping.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from core.logging_config import get_logger

logger = get_logger(__name__)


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

    try:
        # weights_only=True (secure mode)
        if device is None:
            checkpoint_data = torch.load(checkpoint_path, weights_only=True)
        else:
            checkpoint_data = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
    except Exception as e:
        logger.warning(f"Failed to load with weights_only=True: {e}")
        logger.warning("Falling back to weights_only=False (trusted checkpoint)")

        # Fallback to weights_only=False for compatibility with older checkpoints
        if device is None:
            checkpoint_data = torch.load(checkpoint_path, weights_only=False)
        else:
            checkpoint_data = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint_data
