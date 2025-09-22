"""
Learning rate schedulers and resolution schedulers for training.

This module provides various learning rate scheduling strategies
optimized for diffusion model training, plus resolution scheduling
for progressive growing architectures.
"""

import math
from typing import List, Optional, Tuple, Union

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

    # Create the base scheduler
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
            verbose=False,  # Changed to False to avoid warnings
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


class ResolutionScheduler:
    """
    Scheduler for progressive growing of model resolution during training.

    This scheduler manages the transition from low to high resolution training
    in progressive growing architectures, allowing the model to learn coarse
    features first before fine details.
    """

    def __init__(
        self,
        min_resolution: int = 32,
        max_resolution: int = 128,
        num_stages: int = 4,
        epochs_per_stage: int = 25,
        resolution_growth_mode: str = "step",
    ):
        """
        Initialize resolution scheduler.

        Args:
            min_resolution: Starting resolution
            max_resolution: Final resolution
            num_stages: Number of progressive growing stages
            epochs_per_stage: Epochs to train at each resolution
            resolution_growth_mode: How to transition between stages
        """
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.num_stages = num_stages
        self.epochs_per_stage = epochs_per_stage
        self.resolution_growth_mode = resolution_growth_mode

        # Calculate stage resolutions (32, 64, 96, 128)
        self.stage_resolutions = [32, 64, 96, 128]

        self.current_stage = 0
        self.current_epoch = 0

        logger.info(f"ResolutionScheduler initialized: {self.stage_resolutions}")

    def step(self, epoch: Optional[int] = None) -> Tuple[bool, int, str]:
        """
        Step the resolution scheduler.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (should_grow, new_resolution, message)
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        # Calculate which stage we're in
        target_stage = min(
            self.current_epoch // self.epochs_per_stage, self.num_stages - 1
        )

        old_stage = self.current_stage
        new_resolution = self.stage_resolutions[target_stage]

        if target_stage != self.current_stage:
            old_resolution = self.stage_resolutions[self.current_stage]
            message = (
                f"Progressive growing: Stage {self.current_stage} → {target_stage} "
                f"({old_resolution} → {new_resolution})"
            )
            self.current_stage = target_stage
            return True, new_resolution, message
        else:
            return False, new_resolution, f"Continuing at stage {target_stage}"

    def get_current_resolution(self) -> int:
        """Get current training resolution."""
        return self.stage_resolutions[self.current_stage]

    def should_grow_resolution(self, epoch: Optional[int] = None) -> bool:
        """
        Check if resolution should grow at the given epoch.

        Args:
            epoch: Current epoch (uses internal if None)

        Returns:
            True if resolution should grow
        """
        if epoch is not None:
            effective_epoch = epoch
        else:
            effective_epoch = self.current_epoch

        # Check if we're at a stage transition point
        stage_transition_epochs = [
            stage * self.epochs_per_stage for stage in range(1, self.num_stages)
        ]

        return effective_epoch in stage_transition_epochs

    def get_resolution_for_epoch(self, epoch: int) -> int:
        """Get the resolution that should be used for a given epoch."""
        stage = min(epoch // self.epochs_per_stage, self.num_stages - 1)
        return self.stage_resolutions[stage]

    def get_stage_info(self) -> dict:
        """Get information about current stage and schedule."""
        return {
            "current_stage": self.current_stage,
            "current_resolution": self.get_current_resolution(),
            "stage_resolutions": self.stage_resolutions,
            "epochs_per_stage": self.epochs_per_stage,
            "current_epoch": self.current_epoch,
            "total_stages": self.num_stages,
        }


class AdaptiveResolutionScheduler:
    """
    Adaptive resolution scheduler that adjusts resolution based on training progress.

    This scheduler can dynamically adjust resolution based on validation performance,
    loss convergence, or other metrics, allowing for more flexible training schedules.
    """

    def __init__(
        self,
        min_resolution: int = 32,
        max_resolution: int = 128,
        adaptation_metric: str = "loss",
        patience: int = 5,
        improvement_threshold: float = 0.01,
    ):
        """
        Initialize adaptive resolution scheduler.

        Args:
            min_resolution: Minimum resolution
            max_resolution: Maximum resolution
            adaptation_metric: Metric to monitor ('loss', 'psnr', 'ssim')
            patience: Epochs to wait before considering resolution change
            improvement_threshold: Minimum improvement to trigger resolution change
        """
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.adaptation_metric = adaptation_metric
        self.patience = patience
        self.improvement_threshold = improvement_threshold

        self.current_resolution = min_resolution
        self.best_metric_value = (
            float("inf") if adaptation_metric == "loss" else float("-inf")
        )
        self.patience_counter = 0
        self.history = []

        logger.info(
            f"AdaptiveResolutionScheduler initialized: {min_resolution} → {max_resolution}"
        )

    def step(self, current_metric: float, current_epoch: int) -> Tuple[int, bool, str]:
        """
        Step the adaptive scheduler.

        Args:
            current_metric: Current value of monitored metric
            current_epoch: Current epoch

        Returns:
            (new_resolution, should_change, reason)
        """
        self.history.append(
            {
                "epoch": current_epoch,
                "metric": current_metric,
                "resolution": self.current_resolution,
            }
        )

        # Check if current metric is better than best
        is_better = (
            (current_metric < self.best_metric_value - self.improvement_threshold)
            if self.adaptation_metric == "loss"
            else (current_metric > self.best_metric_value + self.improvement_threshold)
        )

        if is_better:
            self.best_metric_value = current_metric
            self.patience_counter = 0
            return self.current_resolution, False, "Metric improved, keeping resolution"

        # Metric didn't improve
        self.patience_counter += 1

        if self.patience_counter >= self.patience:
            # Time to consider resolution change
            if self.current_resolution < self.max_resolution:
                new_resolution = min(self.current_resolution * 2, self.max_resolution)
                self.current_resolution = new_resolution
                self.patience_counter = 0
                return (
                    new_resolution,
                    True,
                    f"Metric stagnant, growing to {new_resolution}",
                )

            elif self.current_resolution > self.min_resolution:
                new_resolution = max(self.current_resolution // 2, self.min_resolution)
                self.current_resolution = new_resolution
                self.patience_counter = 0
                return (
                    new_resolution,
                    True,
                    f"Metric stagnant, reducing to {new_resolution}",
                )

        return self.current_resolution, False, "No change needed"

    def get_resolution_history(self) -> List[dict]:
        """Get history of resolution changes."""
        return self.history.copy()

    def reset(self):
        """Reset scheduler state."""
        self.best_metric_value = (
            float("inf") if self.adaptation_metric == "loss" else float("-inf")
        )
        self.patience_counter = 0
        self.history = []
        self.current_resolution = self.min_resolution


def create_resolution_scheduler(
    scheduler_type: str = "progressive", **kwargs
) -> Union[ResolutionScheduler, AdaptiveResolutionScheduler]:
    """
    Create resolution scheduler.

    Args:
        scheduler_type: Type of scheduler ('progressive', 'adaptive')
        **kwargs: Scheduler-specific parameters

    Returns:
        Configured resolution scheduler
    """
    if scheduler_type.lower() == "progressive":
        return ResolutionScheduler(**kwargs)
    elif scheduler_type.lower() == "adaptive":
        return AdaptiveResolutionScheduler(**kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
