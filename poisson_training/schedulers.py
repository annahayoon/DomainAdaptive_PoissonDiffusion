"""
Learning rate schedulers for training.

This module provides various learning rate scheduling strategies
optimized for diffusion model training.
"""

import math
from typing import Optional, Union

import torch
import torch.optim as optim

from core.logging_config import get_logger

logger = get_logger(__name__)


class WarmupScheduler:
    """
    Warmup scheduler that can be combined with other schedulers.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_factor = self.current_epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group["lr"] = base_lr * warmup_factor
        else:
            # Use base scheduler
            if self.base_scheduler:
                self.base_scheduler.step()

    def state_dict(self):
        """Get state dict."""
        state = {"current_epoch": self.current_epoch, "base_lrs": self.base_lrs}
        if self.base_scheduler:
            state["base_scheduler"] = self.base_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.current_epoch = state_dict["current_epoch"]
        self.base_lrs = state_dict["base_lrs"]
        if self.base_scheduler and "base_scheduler" in state_dict:
            self.base_scheduler.load_state_dict(state_dict["base_scheduler"])


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    patience: int = 10,
    **kwargs,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        patience: Patience for plateau scheduler
        **kwargs: Additional scheduler arguments

    Returns:
        Learning rate scheduler
    """
    base_scheduler = None

    if scheduler_type.lower() == "cosine":
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs, eta_min=min_lr
        )

    elif scheduler_type.lower() == "linear":
        base_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr / optimizer.param_groups[0]["lr"],
            total_iters=num_epochs - warmup_epochs,
        )

    elif scheduler_type.lower() == "exponential":
        gamma = (min_lr / optimizer.param_groups[0]["lr"]) ** (
            1 / (num_epochs - warmup_epochs)
        )
        base_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif scheduler_type.lower() == "plateau":
        base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=patience,
            min_lr=min_lr,
            verbose=True,
        )

    elif scheduler_type.lower() == "step":
        step_size = kwargs.get("step_size", num_epochs // 3)
        gamma = kwargs.get("gamma", 0.1)
        base_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    elif scheduler_type.lower() == "multistep":
        milestones = kwargs.get("milestones", [num_epochs // 3, 2 * num_epochs // 3])
        gamma = kwargs.get("gamma", 0.1)
        base_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )

    elif scheduler_type.lower() == "none":
        return None

    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using cosine")
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs, eta_min=min_lr
        )

    # Wrap with warmup if needed
    if warmup_epochs > 0 and base_scheduler is not None:
        return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
    else:
        return base_scheduler
