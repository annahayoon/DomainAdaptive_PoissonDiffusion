"""
Deterministic training utilities for reproducible results.

This module provides functions for setting up deterministic training
with proper seed management and reproducible behavior.
"""

import os
import random
from typing import Optional

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
