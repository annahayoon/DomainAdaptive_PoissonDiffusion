#!/usr/bin/env python3
"""
L2 (MSE) likelihood guidance for diffusion sampling baseline.

This module implements simple L2 guidance as a baseline to compare against
our Poisson-Gaussian approach. It assumes uniform Gaussian noise and uses
simple MSE-based likelihood gradients.

Mathematical Foundation:
- Assumes: y ~ N(s·x + b, σ²) with uniform noise variance
- Gradient: ∇ log p(y|x) = s·(y - s·x - b) / σ²
- Scheduling: γ(σ) = κ·σ² (identical to Poisson-Gaussian for fair comparison)

This provides a perfect ablation study that isolates the contribution of our
physics-aware approach by sharing all infrastructure except guidance computation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .error_handlers import safe_operation
from .exceptions import GuidanceError, NumericalStabilityError
from .guidance_config import GuidanceConfig
from .interfaces import GuidanceComputer
from .logging_config import get_logger

logger = get_logger(__name__)


class L2Guidance(GuidanceComputer):
    """
    L2 (MSE) likelihood guidance for diffusion sampling.

    This baseline assumes y ~ N(s·x + b, σ²) with uniform noise variance
    and computes simple MSE gradients for comparison with Poisson-Gaussian.
    """

    def __init__(
        self,
        scale: float,
        background: float = 0.0,
        noise_variance: float = 1.0,
        config: Optional[GuidanceConfig] = None,
    ):
        """Initialize L2 guidance with same interface as PoissonGuidance."""
        self.scale = scale
        self.background = background
        self.noise_variance = noise_variance
        self.config = config or GuidanceConfig()

        # Validate parameters
        if scale <= 0:
            raise GuidanceError(f"Scale must be positive, got {scale}")
        if noise_variance <= 0:
            raise GuidanceError(
                f"Noise variance must be positive, got {noise_variance}"
            )

        logger.info(
            f"L2Guidance initialized: scale={scale:.1f} e⁻, "
            f"noise_var={noise_variance:.1f} e⁻²"
        )

    @safe_operation("L2 score computation")
    def compute_score(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute L2 likelihood score: ∇ log p(y|x) = s·(y - prediction) / σ².

        Args:
            x_hat: Current estimate [B, C, H, W] (normalized [0,1])
            y_observed: Observed data [B, C, H, W] (electrons)
            mask: Valid pixel mask [B, C, H, W] (optional)

        Returns:
            Likelihood score [B, C, H, W]
        """
        # Convert prediction to electron space
        prediction_electrons = self.scale * x_hat + self.background

        # Compute residual
        residual = y_observed - prediction_electrons

        # L2 gradient: scale * residual / noise_variance
        score = (self.scale / self.noise_variance) * residual

        # Apply mask if provided
        if mask is not None:
            score = score * mask

        # Clamp for numerical stability (same as Poisson guidance)
        if self.config.gradient_clip > 0:
            score = torch.clamp(
                score, -self.config.gradient_clip, self.config.gradient_clip
            )

        return score

    def gamma_schedule(self, sigma: torch.Tensor) -> torch.Tensor:
        """Use identical scheduling to Poisson-Gaussian for fair comparison."""
        if self.config.gamma_schedule == "sigma2":
            gamma = self.config.kappa * sigma.square()
        elif self.config.gamma_schedule == "linear":
            gamma = self.config.kappa * sigma
        elif self.config.gamma_schedule == "const":
            gamma = torch.full_like(sigma, self.config.kappa)
        else:
            raise GuidanceError(f"Unknown gamma schedule: {self.config.gamma_schedule}")

        return gamma

    @safe_operation("L2 guidance computation")
    def compute(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        sigma_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute scaled guidance gradient (shared implementation)."""
        # Compute base score
        score = self.compute_score(x_hat, y_observed, mask, **kwargs)

        # Compute guidance weight
        gamma = self.gamma_schedule(sigma_t)

        # Ensure gamma has correct shape for broadcasting
        if gamma.dim() == 1:  # [B] -> [B, 1, 1, 1]
            gamma = gamma.view(-1, 1, 1, 1)
        elif gamma.dim() == 0:  # scalar -> scalar
            pass
        else:
            raise GuidanceError(f"Unexpected gamma shape: {gamma.shape}")

        # Scale score by guidance weight
        guidance = score * gamma

        return guidance

    def compute_likelihood(
        self,
        x: torch.Tensor,
        y_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute L2 likelihood (negative MSE).

        Args:
            x: Predicted image (normalized [0,1])
            y_observed: Observed data
            mask: Optional pixel mask
            **kwargs: Additional arguments (ignored)

        Returns:
            Negative MSE loss
        """
        # Normalize y_observed if needed
        if y_observed.max() > 2.0:
            y_normalized = torch.clamp(y_observed / self.scale, 0.0, 1.0)
        else:
            y_normalized = torch.clamp(y_observed, 0.0, 1.0)

        x = torch.clamp(x, 0.0, 1.0)

        # Compute MSE
        mse = torch.mean((x - y_normalized) ** 2, dim=[1, 2, 3])

        # Apply mask if provided
        if mask is not None:
            mask_mean = torch.mean(mask.float(), dim=[1, 2, 3])
            mse = mse * mask_mean

        # Return negative MSE as "likelihood"
        return -mse

    def validate_inputs(
        self, x_hat: torch.Tensor, y_observed: torch.Tensor
    ) -> Tuple[bool, str]:
        """
        Validate input tensors for L2 guidance.

        Args:
            x_hat: Current estimate
            y_observed: Observed data

        Returns:
            Tuple of (is_valid, error_message)
        """
        if x_hat.shape != y_observed.shape:
            return False, f"Shape mismatch: {x_hat.shape} vs {y_observed.shape}"

        if torch.isnan(x_hat).any() or torch.isnan(y_observed).any():
            return False, "NaN values detected in inputs"

        if torch.isinf(x_hat).any() or torch.isinf(y_observed).any():
            return False, "Inf values detected in inputs"

        return True, ""

    def get_diagnostics(self) -> Dict[str, float]:
        """
        Get diagnostic information for L2 guidance.

        Returns:
            Dictionary of diagnostic information
        """
        return {
            "guidance_type": "L2",
            "scale": self.scale,
            "background": self.background,
            "noise_variance": self.noise_variance,
            "kappa": self.config.kappa,
            "gradient_clip": self.config.gradient_clip,
        }
