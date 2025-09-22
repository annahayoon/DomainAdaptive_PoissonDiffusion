"""
DataLoader cleanup utilities to prevent resource leaks.

This module provides utilities to properly manage DataLoader lifecycle
and prevent semaphore leaks when using multiprocessing workers.
"""

import gc
import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def cleanup_dataloader(dataloader: Optional[DataLoader]) -> None:
    """
    Properly cleanup a DataLoader to prevent resource leaks.

    This function ensures that:
    1. Worker processes are properly terminated
    2. Semaphores are released
    3. Memory is freed

    Args:
        dataloader: DataLoader to cleanup (can be None)
    """
    if dataloader is None:
        return

    try:
        # If the dataloader has an active iterator, clean it up
        if hasattr(dataloader, "_iterator"):
            if dataloader._iterator is not None:
                # For multiprocessing dataloaders, properly shutdown workers
                if hasattr(dataloader._iterator, "_shutdown_workers"):
                    dataloader._iterator._shutdown_workers()
                # Delete the iterator
                del dataloader._iterator
                dataloader._iterator = None

        # Force garbage collection to release resources
        gc.collect()

    except Exception as e:
        logger.warning(f"Error during dataloader cleanup: {e}")


def create_safe_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 0,
    persistent_workers: bool = False,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader with safe defaults to prevent resource leaks.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        num_workers: Number of worker processes
        persistent_workers: Whether to keep workers alive between epochs
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader configured for safe operation
    """
    # Safety checks for multiprocessing
    safe_num_workers = num_workers
    safe_persistent_workers = persistent_workers

    # Disable persistent workers if validation is frequent
    # (persistent workers can leak resources with frequent iterator recreation)
    if num_workers > 0:
        # Only use persistent workers for training, not validation
        if kwargs.get("shuffle", True) == False:  # Likely validation
            safe_persistent_workers = False
            logger.debug("Disabled persistent_workers for validation DataLoader")

    # Add timeout for worker processes to prevent hanging
    if "timeout" not in kwargs and num_workers > 0:
        kwargs["timeout"] = 60  # 60 second timeout

    # Create the DataLoader with safe settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=safe_num_workers,
        persistent_workers=safe_persistent_workers and (safe_num_workers > 0),
        **kwargs,
    )

    return dataloader


class DataLoaderContextManager:
    """
    Context manager for safe DataLoader usage with automatic cleanup.

    Usage:
        with DataLoaderContextManager(dataloader) as dl:
            for batch in dl:
                # process batch
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader

    def __enter__(self):
        return self.dataloader

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_dataloader(self.dataloader)
        return False  # Don't suppress exceptions
