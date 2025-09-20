"""
Training module for Poisson-Gaussian diffusion restoration.

This module provides comprehensive training infrastructure including:
- Deterministic training loop
- Physics-aware loss functions
- Training metrics and monitoring
- Learning rate scheduling
- Checkpointing and resuming

Key components:
- DeterministicTrainer: Main training loop with reproducible results
- PoissonGaussianLoss: Physics-aware loss function
- TrainingMetrics: Comprehensive metrics tracking
- Training utilities for deterministic behavior

Requirements addressed: 8.1, 8.2, 8.3, 2.2, 2.5 from requirements.md
Tasks: 5.1, 5.2 from tasks.md
"""

from .losses import (
    CombinedLoss,
    ConsistencyLoss,
    DiffusionLoss,
    PerceptualLoss,
    PoissonGaussianLoss,
    create_loss_function,
)
from .metrics import (
    ImageQualityMetrics,
    MetricsComputer,
    PhysicsMetrics,
    ProgressTracker,
    TrainingMetrics,
    compute_batch_metrics,
    create_metrics_summary,
    log_metrics,
)
from .multi_domain_trainer import (
    DomainBalancedSampler,
    DomainBalancingConfig,
    DomainConditioningEncoder,
    MultiDomainTrainer,
    MultiDomainTrainingConfig,
)
from .schedulers import WarmupScheduler, get_scheduler
from .trainer import DeterministicTrainer, TrainingConfig, create_trainer, train_model
from .utils import (
    EarlyStopping,
    clip_gradients,
    count_parameters,
    create_experiment_dir,
    format_time,
    freeze_model,
    get_device_info,
    get_lr,
    get_memory_usage,
    get_model_size,
    load_checkpoint,
    load_config,
    log_system_info,
    save_checkpoint,
    save_config,
    set_deterministic_mode,
    set_lr,
    unfreeze_model,
)

__all__ = [
    # Main trainer
    "DeterministicTrainer",
    "TrainingConfig",
    "create_trainer",
    "train_model",
    # Multi-domain trainer
    "MultiDomainTrainer",
    "MultiDomainTrainingConfig",
    "DomainBalancingConfig",
    "DomainConditioningEncoder",
    "DomainBalancedSampler",
    # Loss functions
    "PoissonGaussianLoss",
    "DiffusionLoss",
    "ConsistencyLoss",
    "PerceptualLoss",
    "CombinedLoss",
    "create_loss_function",
    # Metrics
    "TrainingMetrics",
    "ImageQualityMetrics",
    "PhysicsMetrics",
    "MetricsComputer",
    "ProgressTracker",
    "compute_batch_metrics",
    "log_metrics",
    "create_metrics_summary",
    # Schedulers
    "WarmupScheduler",
    "get_scheduler",
    # Utilities
    "set_deterministic_mode",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "get_model_size",
    "format_time",
    "get_device_info",
    "log_system_info",
    "create_experiment_dir",
    "save_config",
    "load_config",
    "EarlyStopping",
    "get_lr",
    "set_lr",
    "clip_gradients",
    "freeze_model",
    "unfreeze_model",
    "get_memory_usage",
]
