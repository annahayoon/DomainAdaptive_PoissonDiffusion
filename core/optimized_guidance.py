"""
Optimized guidance computation for Poisson-Gaussian diffusion restoration.

This module provides highly optimized implementations of guidance computation
with focus on performance, memory efficiency, and numerical stability.

Key optimizations:
- Vectorized operations with minimal memory allocation
- Mixed precision support
- Fused operations to reduce memory bandwidth
- Optimized gradient computation
- Batch processing optimization

Requirements addressed: 7.1-7.6 from requirements.md
Task: 6.1 from tasks.md
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from core.error_handlers import ErrorHandler, safe_operation
from core.exceptions import PoissonDiffusionError
from core.logging_config import get_logger
from core.performance import MixedPrecisionManager, PerformanceConfig

logger = get_logger(__name__)


class GuidanceError(PoissonDiffusionError):
    """Raised when guidance computation fails."""

    pass


@dataclass
class GuidanceConfig:
    """Configuration for optimized guidance computation."""

    # Numerical stability
    eps_variance: float = 0.1
    eps_likelihood: float = 1e-8
    max_gradient_norm: float = 1000.0

    # Optimization settings
    use_fused_ops: bool = True
    use_vectorized_ops: bool = True
    cache_intermediate: bool = True

    # Memory optimization
    inplace_operations: bool = True
    gradient_checkpointing: bool = False

    # Precision settings
    guidance_dtype: torch.dtype = torch.float32
    computation_dtype: torch.dtype = torch.float32


class OptimizedPoissonGuidance(nn.Module):
    """Optimized Poisson-Gaussian guidance computation."""

    def __init__(
        self,
        config: GuidanceConfig = None,
        performance_config: PerformanceConfig = None,
    ):
        super().__init__()
        self.config = config or GuidanceConfig()
        self.performance_config = performance_config or PerformanceConfig()

        # Initialize mixed precision manager
        self.mixed_precision = MixedPrecisionManager(self.performance_config)

        # Precompute constants
        self.register_buffer("eps_variance", torch.tensor(self.config.eps_variance))
        self.register_buffer("eps_likelihood", torch.tensor(self.config.eps_likelihood))
        self.register_buffer(
            "max_grad_norm", torch.tensor(self.config.max_gradient_norm)
        )

        # Cache for intermediate computations
        self._cache = {}

        logger.info("Initialized OptimizedPoissonGuidance")

    def forward(
        self,
        x_pred: torch.Tensor,
        y_obs: torch.Tensor,
        scale: torch.Tensor,
        background: torch.Tensor,
        read_noise: torch.Tensor,
        guidance_weight: float = 1.0,
        mode: str = "wls",
    ) -> torch.Tensor:
        """
        Compute optimized Poisson-Gaussian guidance.

        Args:
            x_pred: Predicted clean image [B, C, H, W]
            y_obs: Observed noisy image [B, C, H, W]
            scale: Scale parameter (electrons per normalized unit) [B] or scalar
            background: Background level (electrons) [B] or scalar
            read_noise: Read noise standard deviation (electrons) [B] or scalar
            guidance_weight: Guidance strength
            mode: Guidance mode ("wls" or "exact")

        Returns:
            Guidance gradient [B, C, H, W]
        """
        with self.mixed_precision.autocast_context():
            if mode == "wls":
                return self._compute_wls_guidance(
                    x_pred, y_obs, scale, background, read_noise, guidance_weight
                )
            elif mode == "exact":
                return self._compute_exact_guidance(
                    x_pred, y_obs, scale, background, read_noise, guidance_weight
                )
            else:
                raise ValueError(f"Unknown guidance mode: {mode}")

    def _compute_wls_guidance(
        self,
        x_pred: torch.Tensor,
        y_obs: torch.Tensor,
        scale: torch.Tensor,
        background: torch.Tensor,
        read_noise: torch.Tensor,
        guidance_weight: float,
    ) -> torch.Tensor:
        """Compute WLS (Weighted Least Squares) guidance - optimized version."""
        # Ensure proper tensor shapes for broadcasting
        scale, background, read_noise = self._prepare_parameters(
            scale, background, read_noise, x_pred.shape
        )

        if self.config.use_fused_ops:
            return self._compute_wls_guidance_fused(
                x_pred, y_obs, scale, background, read_noise, guidance_weight
            )
        else:
            return self._compute_wls_guidance_standard(
                x_pred, y_obs, scale, background, read_noise, guidance_weight
            )

    def _compute_wls_guidance_fused(
        self,
        x_pred: torch.Tensor,
        y_obs: torch.Tensor,
        scale: torch.Tensor,
        background: torch.Tensor,
        read_noise: torch.Tensor,
        guidance_weight: float,
    ) -> torch.Tensor:
        """Fused WLS guidance computation for maximum efficiency."""
        # Convert prediction to electron space and compute variance in one operation
        # lambda = scale * x_pred + background
        # variance = lambda + read_noise^2
        # gradient = scale * (y_obs - lambda) / variance

        # Fused operation: compute lambda and variance together
        lambda_pred = torch.addcmul(
            background, scale, x_pred
        )  # scale * x_pred + background

        # Compute variance with numerical stability
        read_noise_sq = read_noise.square()
        variance = lambda_pred + read_noise_sq + self.eps_variance

        # Compute residual and gradient in fused operation
        residual = y_obs - lambda_pred
        gradient = scale * residual / variance

        # Apply guidance weight and clamp for stability
        if guidance_weight != 1.0:
            gradient = gradient * guidance_weight

        # Clamp gradient norm for numerical stability
        if self.config.max_gradient_norm > 0:
            gradient = torch.clamp(
                gradient, -self.config.max_gradient_norm, self.config.max_gradient_norm
            )

        return gradient

    def _compute_wls_guidance_standard(
        self,
        x_pred: torch.Tensor,
        y_obs: torch.Tensor,
        scale: torch.Tensor,
        background: torch.Tensor,
        read_noise: torch.Tensor,
        guidance_weight: float,
    ) -> torch.Tensor:
        """Standard WLS guidance computation."""
        # Convert to electron space
        lambda_pred = scale * x_pred + background

        # Compute variance
        variance = lambda_pred + read_noise.square() + self.eps_variance

        # Compute gradient: ∇ log p(y|x) = scale * (y - λ) / variance
        residual = y_obs - lambda_pred
        gradient = guidance_weight * scale * residual / variance

        # Clamp for numerical stability
        if self.config.max_gradient_norm > 0:
            gradient = torch.clamp(
                gradient, -self.config.max_gradient_norm, self.config.max_gradient_norm
            )

        return gradient

    def _compute_exact_guidance(
        self,
        x_pred: torch.Tensor,
        y_obs: torch.Tensor,
        scale: torch.Tensor,
        background: torch.Tensor,
        read_noise: torch.Tensor,
        guidance_weight: float,
    ) -> torch.Tensor:
        """Compute exact Poisson-Gaussian guidance with variance derivatives."""
        # Ensure proper tensor shapes
        scale, background, read_noise = self._prepare_parameters(
            scale, background, read_noise, x_pred.shape
        )

        # Convert to electron space
        lambda_pred = scale * x_pred + background
        lambda_pred = torch.clamp(lambda_pred, min=self.eps_likelihood)

        # Compute variance
        read_noise_sq = read_noise.square()
        variance = lambda_pred + read_noise_sq + self.eps_variance

        # Exact gradient includes variance derivative term
        # ∇ log p(y|x) = scale * [(y - λ)/variance - 0.5 * (y - λ)²/variance²]
        residual = y_obs - lambda_pred
        residual_sq = residual.square()

        # Main term
        main_term = residual / variance

        # Variance derivative term
        variance_term = 0.5 * residual_sq / variance.square()

        # Combined gradient
        gradient = guidance_weight * scale * (main_term - variance_term)

        # Clamp for numerical stability
        if self.config.max_gradient_norm > 0:
            gradient = torch.clamp(
                gradient, -self.config.max_gradient_norm, self.config.max_gradient_norm
            )

        return gradient

    def _prepare_parameters(
        self,
        scale: torch.Tensor,
        background: torch.Tensor,
        read_noise: torch.Tensor,
        target_shape: torch.Size,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare parameters for broadcasting."""
        batch_size = target_shape[0]

        # Ensure parameters are tensors
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(
                scale, device=target_shape.device, dtype=self.config.guidance_dtype
            )
        if not isinstance(background, torch.Tensor):
            background = torch.tensor(
                background, device=target_shape.device, dtype=self.config.guidance_dtype
            )
        if not isinstance(read_noise, torch.Tensor):
            read_noise = torch.tensor(
                read_noise, device=target_shape.device, dtype=self.config.guidance_dtype
            )

        # Reshape for broadcasting if needed
        if scale.numel() == 1:
            scale = scale.expand(batch_size)
        if background.numel() == 1:
            background = background.expand(batch_size)
        if read_noise.numel() == 1:
            read_noise = read_noise.expand(batch_size)

        # Reshape to [B, 1, 1, 1] for broadcasting
        if scale.dim() == 1:
            scale = scale.view(-1, 1, 1, 1)
        if background.dim() == 1:
            background = background.view(-1, 1, 1, 1)
        if read_noise.dim() == 1:
            read_noise = read_noise.view(-1, 1, 1, 1)

        return scale, background, read_noise

    def compute_likelihood(
        self,
        x_pred: torch.Tensor,
        y_obs: torch.Tensor,
        scale: torch.Tensor,
        background: torch.Tensor,
        read_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute optimized Poisson-Gaussian log-likelihood."""
        with self.mixed_precision.autocast_context():
            # Prepare parameters
            scale, background, read_noise = self._prepare_parameters(
                scale, background, read_noise, x_pred.shape
            )

            # Convert to electron space
            lambda_pred = scale * x_pred + background
            lambda_pred = torch.clamp(lambda_pred, min=self.eps_likelihood)

            # Poisson term: -λ + y * log(λ)
            poisson_term = -lambda_pred + y_obs * torch.log(lambda_pred)

            # Gaussian term: -0.5 * (y - λ)² / σ²
            residual = y_obs - lambda_pred
            gaussian_term = -0.5 * residual.square() / read_noise.square()

            # Combined log-likelihood
            log_likelihood = poisson_term + gaussian_term

            return log_likelihood.sum(dim=[1, 2, 3])  # Sum over spatial dimensions


class OptimizedSampler:
    """Optimized diffusion sampling with guidance."""

    def __init__(
        self,
        guidance: OptimizedPoissonGuidance,
        performance_config: PerformanceConfig = None,
    ):
        self.guidance = guidance
        self.performance_config = performance_config or PerformanceConfig()
        self.mixed_precision = MixedPrecisionManager(self.performance_config)

        logger.info("Initialized OptimizedSampler")

    def sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y_obs: torch.Tensor,
        scale: torch.Tensor,
        background: torch.Tensor,
        read_noise: torch.Tensor,
        guidance_weight: float = 1.0,
        guidance_mode: str = "wls",
        **model_kwargs,
    ) -> torch.Tensor:
        """Optimized sampling step with guidance."""
        with self.mixed_precision.autocast_context():
            # Model prediction
            with torch.no_grad():
                model_output = model(x_t, t, **model_kwargs)

            # Extract prediction (assuming v-parameterization or similar)
            if isinstance(model_output, dict):
                x_pred = model_output.get(
                    "prediction",
                    model_output.get("v_pred", model_output.get("eps_pred")),
                )
            else:
                x_pred = model_output

            # Compute guidance
            guidance_grad = self.guidance(
                x_pred,
                y_obs,
                scale,
                background,
                read_noise,
                guidance_weight,
                guidance_mode,
            )

            # Apply guidance to prediction
            guided_pred = x_pred + guidance_grad

            return guided_pred

    def sample_batch(
        self,
        model: nn.Module,
        x_t_batch: torch.Tensor,
        t_batch: torch.Tensor,
        y_obs_batch: torch.Tensor,
        scale_batch: torch.Tensor,
        background_batch: torch.Tensor,
        read_noise_batch: torch.Tensor,
        guidance_weight: float = 1.0,
        guidance_mode: str = "wls",
        **model_kwargs,
    ) -> torch.Tensor:
        """Optimized batch sampling with guidance."""
        batch_size = x_t_batch.shape[0]
        results = []

        # Process in chunks if batch is too large
        max_chunk_size = self._estimate_max_chunk_size(x_t_batch)

        for i in range(0, batch_size, max_chunk_size):
            end_idx = min(i + max_chunk_size, batch_size)

            # Extract chunk
            x_t_chunk = x_t_batch[i:end_idx]
            t_chunk = t_batch[i:end_idx]
            y_obs_chunk = y_obs_batch[i:end_idx]
            scale_chunk = (
                scale_batch[i:end_idx] if scale_batch.numel() > 1 else scale_batch
            )
            background_chunk = (
                background_batch[i:end_idx]
                if background_batch.numel() > 1
                else background_batch
            )
            read_noise_chunk = (
                read_noise_batch[i:end_idx]
                if read_noise_batch.numel() > 1
                else read_noise_batch
            )

            # Process chunk
            chunk_result = self.sample_step(
                model,
                x_t_chunk,
                t_chunk,
                y_obs_chunk,
                scale_chunk,
                background_chunk,
                read_noise_chunk,
                guidance_weight,
                guidance_mode,
                **model_kwargs,
            )

            results.append(chunk_result)

        return torch.cat(results, dim=0)

    def _estimate_max_chunk_size(self, sample_tensor: torch.Tensor) -> int:
        """Estimate maximum chunk size based on available memory."""
        if not torch.cuda.is_available():
            return sample_tensor.shape[0]  # No chunking for CPU

        # Estimate memory usage per sample
        sample_memory = sample_tensor[0:1].numel() * sample_tensor.element_size()

        # Get available GPU memory
        available_memory = (
            torch.cuda.get_device_properties(0).total_memory * 0.7
        )  # Use 70%
        current_memory = torch.cuda.memory_allocated()
        free_memory = available_memory - current_memory

        # Estimate chunk size (conservative factor of 4 for intermediate computations)
        estimated_chunk_size = int(free_memory / (sample_memory * 4))

        # Clamp to reasonable range
        chunk_size = max(1, min(estimated_chunk_size, sample_tensor.shape[0]))

        return chunk_size


class GuidanceOptimizer:
    """Optimizer for guidance computation parameters."""

    def __init__(self, guidance: OptimizedPoissonGuidance):
        self.guidance = guidance
        self.optimization_history = []

    def optimize_guidance_weight(
        self,
        model: nn.Module,
        validation_data: List[Dict[str, torch.Tensor]],
        weight_range: Tuple[float, float] = (0.1, 10.0),
        num_trials: int = 10,
    ) -> float:
        """Optimize guidance weight based on validation performance."""
        logger.info("Optimizing guidance weight...")

        best_weight = 1.0
        best_score = float("-inf")

        # Test different guidance weights
        weights = np.logspace(
            np.log10(weight_range[0]), np.log10(weight_range[1]), num_trials
        )

        for weight in weights:
            scores = []

            for batch in validation_data:
                with torch.no_grad():
                    # Compute guidance with this weight
                    guidance_grad = self.guidance(
                        batch["prediction"],
                        batch["observed"],
                        batch["scale"],
                        batch["background"],
                        batch["read_noise"],
                        guidance_weight=weight,
                    )

                    # Compute validation metric (e.g., likelihood)
                    likelihood = self.guidance.compute_likelihood(
                        batch["prediction"] + guidance_grad,
                        batch["observed"],
                        batch["scale"],
                        batch["background"],
                        batch["read_noise"],
                    )

                    scores.append(likelihood.mean().item())

            avg_score = np.mean(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_weight = weight

            logger.debug(f"Weight {weight:.3f}: score {avg_score:.3f}")

        logger.info(
            f"Optimal guidance weight: {best_weight:.3f} (score: {best_score:.3f})"
        )
        return best_weight
