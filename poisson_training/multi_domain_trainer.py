"""
Multi-domain training framework for Poisson-Gaussian diffusion restoration.

This module extends the deterministic training framework to support balanced
training across multiple imaging domains (photography, microscopy, astronomy)
with domain-specific loss weighting and performance monitoring.

Key features:
- Weighted sampling across domains
- Domain-specific loss weighting
- Balanced batch composition strategies
- Domain conditioning vector generation
- Per-domain performance monitoring
- Adaptive domain balancing

Requirements addressed: 2.2, 2.5 from requirements.md
Task: 5.2 from tasks.md
"""

import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from core.error_handlers import ErrorHandler, safe_operation
from core.exceptions import TrainingError, ValidationError
from core.logging_config import get_logger
from data.domain_datasets import DomainDataset, MultiDomainDataset

from .losses import PoissonGaussianLoss
from .metrics import TrainingMetrics
from .trainer import DeterministicTrainer, TrainingConfig

logger = get_logger(__name__)


@dataclass
class DomainBalancingConfig:
    """Configuration for domain balancing strategies."""

    # Sampling strategy
    sampling_strategy: str = "weighted"  # "weighted", "uniform", "adaptive"

    # Domain weights (if None, computed automatically)
    domain_weights: Optional[Dict[str, float]] = None

    # Loss weighting
    use_domain_loss_weights: bool = True
    domain_loss_weights: Optional[Dict[str, float]] = None

    # Batch composition
    enforce_batch_balance: bool = True
    min_samples_per_domain_per_batch: int = 1

    # Adaptive balancing
    adaptive_rebalancing: bool = True
    rebalancing_frequency: int = 100  # batches
    performance_window: int = 50  # batches for performance tracking

    # Domain conditioning
    use_domain_conditioning: bool = True
    conditioning_dim: int = 6  # 3 (domain) + 1 (scale) + 2 (noise params)

    # Monitoring
    log_domain_stats: bool = True
    domain_stats_frequency: int = 50  # batches


@dataclass
class MultiDomainTrainingConfig(TrainingConfig):
    """Extended training configuration for multi-domain training."""

    # Multi-domain specific settings
    domain_balancing: DomainBalancingConfig = field(
        default_factory=DomainBalancingConfig
    )

    # Domain-specific learning rates (optional)
    domain_learning_rates: Optional[Dict[str, float]] = None

    # Domain-specific schedulers (optional)
    domain_schedulers: Optional[Dict[str, str]] = None


class DomainConditioningEncoder:
    """Encodes domain information into conditioning vectors."""

    def __init__(self, domains: List[str], conditioning_dim: int = 6):
        """
        Initialize domain conditioning encoder.

        Args:
            domains: List of domain names
            conditioning_dim: Total conditioning vector dimension
        """
        self.domains = sorted(domains)  # Ensure consistent ordering
        self.domain_to_idx = {domain: idx for idx, domain in enumerate(self.domains)}
        self.conditioning_dim = conditioning_dim

        logger.info(
            f"Initialized domain conditioning for {len(self.domains)} domains: {self.domains}"
        )

    def encode(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Encode batch metadata into domain conditioning vectors.

        Args:
            batch: Batch dictionary containing metadata

        Returns:
            Conditioning tensor of shape (batch_size, conditioning_dim)
        """
        batch_size = len(batch.get("domain", []))
        conditioning = torch.zeros(batch_size, self.conditioning_dim)

        # Extract domain information
        domains = batch.get("domain", ["photography"] * batch_size)
        scales = batch.get("scale", torch.ones(batch_size) * 1000.0)
        read_noises = batch.get("read_noise", torch.ones(batch_size) * 5.0)
        backgrounds = batch.get("background", torch.ones(batch_size) * 100.0)

        for i in range(batch_size):
            # Domain one-hot encoding (first 3 dimensions)
            domain = domains[i] if isinstance(domains, list) else domains[i].item()
            if domain in self.domain_to_idx:
                conditioning[i, self.domain_to_idx[domain]] = 1.0

            # Normalized scale parameter (dimension 3)
            scale = scales[i] if isinstance(scales, torch.Tensor) else scales
            scale_val = scale.item() if isinstance(scale, torch.Tensor) else scale
            # Log-normalize scale to [-1, 1] range (assuming scale range 10-100000)
            log_scale = math.log10(max(scale_val, 10))
            normalized_scale = 2 * (log_scale - 1) / (5 - 1) - 1  # Map [1,5] to [-1,1]
            conditioning[i, 3] = max(-1, min(1, normalized_scale))

            # Relative noise parameters (dimensions 4-5)
            read_noise = (
                read_noises[i] if isinstance(read_noises, torch.Tensor) else read_noises
            )
            background = (
                backgrounds[i] if isinstance(backgrounds, torch.Tensor) else backgrounds
            )

            read_noise_val = (
                read_noise.item()
                if isinstance(read_noise, torch.Tensor)
                else read_noise
            )
            background_val = (
                background.item()
                if isinstance(background, torch.Tensor)
                else background
            )

            # Relative read noise (read_noise / sqrt(scale))
            rel_read_noise = read_noise_val / math.sqrt(scale_val)
            conditioning[i, 4] = min(
                1.0, rel_read_noise / 0.1
            )  # Normalize by typical value

            # Relative background (background / scale)
            rel_background = background_val / scale_val
            conditioning[i, 5] = min(1.0, rel_background)

        return conditioning

    def get_domain_names(self) -> List[str]:
        """Get list of domain names."""
        return self.domains.copy()


class DomainBalancedSampler:
    """Handles balanced sampling across domains."""

    def __init__(
        self,
        dataset: MultiDomainDataset,
        config: DomainBalancingConfig,
        batch_size: int,
    ):
        """
        Initialize domain balanced sampler.

        Args:
            dataset: Multi-domain dataset
            config: Domain balancing configuration
            batch_size: Training batch size
        """
        self.dataset = dataset
        self.config = config
        self.batch_size = batch_size

        # Get domain information
        self.domains = list(dataset.domain_datasets.keys())
        self.domain_sizes = {
            domain: len(ds) for domain, ds in dataset.domain_datasets.items()
        }

        # Initialize sampling weights
        self._compute_sampling_weights()

        # Performance tracking for adaptive rebalancing
        self.domain_performance = defaultdict(list)
        self.batch_count = 0

        logger.info(
            f"Initialized domain balanced sampler for {len(self.domains)} domains"
        )
        logger.info(f"Domain sizes: {self.domain_sizes}")
        logger.info(f"Sampling strategy: {config.sampling_strategy}")

    def _compute_sampling_weights(self):
        """Compute sampling weights for each domain."""
        if self.config.domain_weights is not None:
            # Use provided weights
            self.domain_weights = self.config.domain_weights.copy()
        else:
            # Compute balanced weights
            if self.config.sampling_strategy == "uniform":
                # Equal weight for each domain
                self.domain_weights = {domain: 1.0 for domain in self.domains}
            elif self.config.sampling_strategy == "weighted":
                # Inverse frequency weighting
                total_samples = sum(self.domain_sizes.values())
                self.domain_weights = {
                    domain: total_samples / (len(self.domains) * size)
                    for domain, size in self.domain_sizes.items()
                }
            else:  # adaptive
                # Start with uniform, will be updated based on performance
                self.domain_weights = {domain: 1.0 for domain in self.domains}

        # Normalize weights
        total_weight = sum(self.domain_weights.values())
        self.domain_weights = {
            domain: weight / total_weight
            for domain, weight in self.domain_weights.items()
        }

        logger.info(f"Domain sampling weights: {self.domain_weights}")

    def create_batch_indices(self) -> List[int]:
        """Create balanced batch indices."""
        if not self.config.enforce_batch_balance:
            # Simple weighted random sampling
            return self._weighted_random_sampling()
        else:
            # Enforce minimum samples per domain per batch
            return self._balanced_batch_sampling()

    def _weighted_random_sampling(self) -> List[int]:
        """Weighted random sampling across all samples."""
        # Create sample weights based on domain weights
        sample_weights = []
        sample_indices = []

        for domain in self.domains:
            domain_dataset = self.dataset.domain_datasets[domain]
            domain_weight = self.domain_weights[domain]

            for i in range(len(domain_dataset)):
                # Get global index for this sample
                global_idx = self.dataset._get_global_index(domain, i)
                sample_indices.append(global_idx)
                sample_weights.append(domain_weight)

        # Sample with replacement
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=self.batch_size, replacement=True
        )

        return [sample_indices[i] for i in sampler]

    def _balanced_batch_sampling(self) -> List[int]:
        """Balanced batch sampling with minimum samples per domain."""
        batch_indices = []
        min_per_domain = self.config.min_samples_per_domain_per_batch

        # Ensure we can satisfy minimum requirements
        total_min_samples = len(self.domains) * min_per_domain
        if total_min_samples > self.batch_size:
            logger.warning(
                f"Cannot satisfy minimum {min_per_domain} samples per domain "
                f"with batch size {self.batch_size}. Using weighted sampling."
            )
            return self._weighted_random_sampling()

        # Sample minimum required samples from each domain
        for domain in self.domains:
            domain_dataset = self.dataset.domain_datasets[domain]
            domain_indices = random.sample(range(len(domain_dataset)), min_per_domain)

            for domain_idx in domain_indices:
                global_idx = self.dataset._get_global_index(domain, domain_idx)
                batch_indices.append(global_idx)

        # Fill remaining slots with weighted sampling
        remaining_slots = self.batch_size - len(batch_indices)
        if remaining_slots > 0:
            additional_indices = self._weighted_random_sampling()[:remaining_slots]
            batch_indices.extend(additional_indices)

        # Shuffle to avoid domain ordering bias
        random.shuffle(batch_indices)

        return batch_indices

    def update_performance(self, domain_losses: Dict[str, float]):
        """Update domain performance for adaptive rebalancing."""
        if not self.config.adaptive_rebalancing:
            return

        self.batch_count += 1

        # Record performance
        for domain, loss in domain_losses.items():
            self.domain_performance[domain].append(loss)

            # Keep only recent performance
            if len(self.domain_performance[domain]) > self.config.performance_window:
                self.domain_performance[domain].pop(0)

        # Rebalance if needed
        if self.batch_count % self.config.rebalancing_frequency == 0:
            self._adaptive_rebalancing()

    def _adaptive_rebalancing(self):
        """Adaptively rebalance domain weights based on performance."""
        if len(self.domain_performance) < len(self.domains):
            return  # Not enough data yet

        # Compute average recent performance for each domain
        domain_avg_losses = {}
        for domain in self.domains:
            if domain in self.domain_performance and self.domain_performance[domain]:
                domain_avg_losses[domain] = np.mean(self.domain_performance[domain])
            else:
                domain_avg_losses[domain] = float("inf")

        # Increase weight for domains with higher loss (worse performance)
        max_loss = max(domain_avg_losses.values())
        min_loss = min(
            loss for loss in domain_avg_losses.values() if loss != float("inf")
        )

        if max_loss > min_loss:
            for domain in self.domains:
                loss = domain_avg_losses[domain]
                if loss != float("inf"):
                    # Higher loss -> higher weight (more training focus)
                    normalized_loss = (loss - min_loss) / (max_loss - min_loss)
                    self.domain_weights[domain] = 0.5 + 0.5 * normalized_loss
                else:
                    self.domain_weights[domain] = 1.0

            # Normalize weights
            total_weight = sum(self.domain_weights.values())
            self.domain_weights = {
                domain: weight / total_weight
                for domain, weight in self.domain_weights.items()
            }

            logger.info(f"Adaptive rebalancing: new weights {self.domain_weights}")


class MultiDomainTrainer(DeterministicTrainer):
    """Extended trainer for multi-domain training with balanced sampling."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: MultiDomainDataset,
        val_dataset: Optional[MultiDomainDataset],
        config: MultiDomainTrainingConfig,
        loss_fn: Optional[nn.Module] = None,
        metrics: Optional[TrainingMetrics] = None,
    ):
        """
        Initialize multi-domain trainer.

        Args:
            model: Model to train
            train_dataset: Multi-domain training dataset
            val_dataset: Multi-domain validation dataset
            config: Multi-domain training configuration
            loss_fn: Loss function (will create if None)
            metrics: Metrics tracker (will create if None)
        """
        # Initialize domain conditioning
        self.domains = list(train_dataset.domain_datasets.keys())
        self.domain_encoder = DomainConditioningEncoder(
            domains=self.domains,
            conditioning_dim=config.domain_balancing.conditioning_dim,
        )

        # Initialize domain balanced sampler
        self.domain_sampler = DomainBalancedSampler(
            dataset=train_dataset,
            config=config.domain_balancing,
            batch_size=config.batch_size,
        )

        # Create custom data loaders with balanced sampling
        train_dataloader = self._create_balanced_dataloader(
            train_dataset, config, is_training=True
        )
        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = self._create_balanced_dataloader(
                val_dataset, config, is_training=False
            )

        # Initialize parent trainer
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
        )

        # Override loss_fn and metrics from parent initialization
        if loss_fn is not None:
            self.loss_fn = loss_fn
        if metrics is not None:
            self.metrics = metrics

        # Multi-domain specific attributes
        self.config = config
        self.domain_balancing_config = config.domain_balancing

        # Domain-specific loss weights
        self.domain_loss_weights = self._setup_domain_loss_weights()

        # Domain performance tracking
        self.domain_metrics = defaultdict(lambda: defaultdict(list))
        self.batch_domain_stats = defaultdict(int)

        logger.info(f"Initialized multi-domain trainer for {len(self.domains)} domains")
        logger.info(f"Domains: {self.domains}")

    def _create_balanced_dataloader(
        self,
        dataset: MultiDomainDataset,
        config: MultiDomainTrainingConfig,
        is_training: bool,
    ) -> DataLoader:
        """Create data loader with balanced sampling."""
        if is_training and config.domain_balancing.sampling_strategy != "uniform":
            # Use custom batch sampling for training
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,  # We handle sampling manually
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=getattr(config, "persistent_workers", True),
                prefetch_factor=getattr(config, "prefetch_factor", 2),
                collate_fn=self._collate_fn,
            )
        else:
            # Standard data loader for validation or uniform sampling
            return DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=is_training,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                persistent_workers=getattr(config, "persistent_workers", True),
                prefetch_factor=getattr(config, "prefetch_factor", 2),
                collate_fn=self._collate_fn,
            )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function that preserves domain information."""
        # Standard collation for tensor fields
        collated = {}

        # Handle tensor fields
        tensor_fields = [
            "electrons",
            "clean",
            "noisy",
            "scale",
            "background",
            "read_noise",
        ]
        for field in tensor_fields:
            if field in batch[0]:
                values = [item[field] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    collated[field] = torch.stack(values)
                else:
                    collated[field] = torch.tensor(values)

        # Handle metadata fields
        metadata_fields = ["domain", "filepath", "metadata"]
        for field in metadata_fields:
            if field in batch[0]:
                collated[field] = [item[field] for item in batch]

        # Add domain conditioning if enabled
        if self.domain_balancing_config.use_domain_conditioning:
            collated["domain_conditioning"] = self.domain_encoder.encode(collated)

        return collated

    def _setup_domain_loss_weights(self) -> Dict[str, float]:
        """Setup domain-specific loss weights."""
        if not self.domain_balancing_config.use_domain_loss_weights:
            return {domain: 1.0 for domain in self.domains}

        if self.domain_balancing_config.domain_loss_weights is not None:
            return self.domain_balancing_config.domain_loss_weights.copy()

        # Compute automatic loss weights based on domain difficulty
        # For now, use uniform weights (can be made adaptive later)
        return {domain: 1.0 for domain in self.domains}

    def _train_epoch(self) -> Dict[str, float]:
        """Train one epoch with domain balancing."""
        self.model.train()
        epoch_metrics = defaultdict(list)
        domain_batch_counts = defaultdict(int)

        # Custom batch iteration with balanced sampling
        num_batches = len(self.train_dataloader.dataset) // self.config.batch_size

        for batch_idx in range(num_batches):
            # Create balanced batch
            if self.domain_balancing_config.sampling_strategy != "uniform":
                batch_indices = self.domain_sampler.create_batch_indices()
                batch = self._create_batch_from_indices(batch_indices)
            else:
                # Use standard data loader
                batch = next(iter(self.train_dataloader))

            # Move to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()

            if self.config.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch)
                    losses = self._compute_domain_aware_loss(outputs, batch)
                    total_loss = losses["total_loss"]

                self.scaler.scale(total_loss).backward()

                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                losses = self._compute_domain_aware_loss(outputs, batch)
                total_loss = losses["total_loss"]

                total_loss.backward()

                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )

                self.optimizer.step()

            # Update EMA model
            if self.ema_model is not None:
                self._update_ema_model()

            # Record metrics
            for key, value in losses.items():
                epoch_metrics[key].append(
                    value.item() if isinstance(value, torch.Tensor) else value
                )

            # Track domain statistics
            domains_in_batch = batch.get("domain", [])
            for domain in domains_in_batch:
                domain_batch_counts[domain] += 1

            # Update domain performance for adaptive rebalancing
            if hasattr(losses, "domain_losses"):
                self.domain_sampler.update_performance(losses["domain_losses"])

            # Logging
            if batch_idx % self.config.log_frequency == 0:
                self._log_training_progress(batch_idx, num_batches, losses)

            # Domain statistics logging
            if (
                self.domain_balancing_config.log_domain_stats
                and batch_idx % self.domain_balancing_config.domain_stats_frequency == 0
            ):
                self._log_domain_statistics(domain_batch_counts, batch_idx)

        # Compute epoch averages
        return {key: np.mean(values) for key, values in epoch_metrics.items()}

    def _create_batch_from_indices(self, indices: List[int]) -> Dict[str, Any]:
        """Create batch from dataset indices."""
        batch_items = [self.train_dataloader.dataset[idx] for idx in indices]
        return self._collate_fn(batch_items)

    def _compute_domain_aware_loss(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Compute loss with domain-specific weighting."""
        # Get base losses
        base_losses = self.loss_fn(outputs, batch)

        # Apply domain-specific weighting if enabled
        if self.domain_balancing_config.use_domain_loss_weights:
            domains = batch.get("domain", [])
            batch_size = len(domains)

            # Compute per-sample domain weights
            domain_weights = torch.ones(batch_size, device=self.device)
            for i, domain in enumerate(domains):
                if domain in self.domain_loss_weights:
                    domain_weights[i] = self.domain_loss_weights[domain]

            # Apply weights to losses
            weighted_losses = {}
            for key, loss in base_losses.items():
                if key != "total_loss" and isinstance(loss, torch.Tensor):
                    if loss.dim() == 0:  # Scalar loss
                        weighted_losses[key] = loss
                    else:  # Per-sample loss
                        weighted_losses[key] = (loss * domain_weights).mean()
                else:
                    weighted_losses[key] = loss

            # Recompute total loss
            total_loss = sum(
                loss
                for key, loss in weighted_losses.items()
                if key != "total_loss" and isinstance(loss, torch.Tensor)
            )
            weighted_losses["total_loss"] = total_loss

            return weighted_losses

        return base_losses

    def _log_domain_statistics(self, domain_counts: Dict[str, int], batch_idx: int):
        """Log domain distribution statistics."""
        total_samples = sum(domain_counts.values())
        if total_samples == 0:
            return

        domain_percentages = {
            domain: (count / total_samples) * 100
            for domain, count in domain_counts.items()
        }

        stats_str = ", ".join(
            [
                f"{domain}: {percentage:.1f}%"
                for domain, percentage in domain_percentages.items()
            ]
        )

        logger.info(f"Batch {batch_idx} domain distribution: {stats_str}")

    def _log_training_progress(
        self, batch_idx: int, num_batches: int, losses: Dict[str, torch.Tensor]
    ):
        """Log training progress with domain information."""
        loss_str = ", ".join(
            [
                f"{key}: {value.item():.4f}"
                if isinstance(value, torch.Tensor)
                else f"{key}: {value:.4f}"
                for key, value in losses.items()
                if key != "domain_losses"
            ]
        )

        logger.info(
            f"Epoch {self.current_epoch:3d} | "
            f"Batch {batch_idx:4d}/{num_batches:4d} | "
            f"{loss_str} | "
            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        )

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get comprehensive domain training statistics."""
        return {
            "domains": self.domains,
            "domain_weights": self.domain_sampler.domain_weights,
            "domain_loss_weights": self.domain_loss_weights,
            "domain_sizes": self.domain_sampler.domain_sizes,
            "sampling_strategy": self.domain_balancing_config.sampling_strategy,
            "conditioning_enabled": self.domain_balancing_config.use_domain_conditioning,
            "adaptive_rebalancing": self.domain_balancing_config.adaptive_rebalancing,
        }
