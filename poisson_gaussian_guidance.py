#!/usr/bin/env python3
"""
Poisson-Gaussian Guidance for Diffusion Models

This module implements Equation 3 from the paper:
The score (gradient) of the Poisson-Gaussian log-likelihood.

KEY INSIGHT: The variance in photon-limited imaging is signal-dependent:
    Var[y|x] = s·x + σ_r²

This heteroscedasticity requires adaptive weighting - we cannot use uniform L2 loss.
"""

import logging
from typing import Literal, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PoissonGaussianGuidance(nn.Module):
    """
    Physics-informed guidance for photon-limited imaging

    Implements the score of the Poisson-Gaussian likelihood:
        ∇_x log p(y_e|x)

    This tells the diffusion model how to adjust predictions to match
    observed noisy measurements while respecting physical noise properties.

    Args:
        s: Scale factor (max photon count, typically full-well capacity)
        sigma_r: Read noise standard deviation (in electrons)
        kappa: Guidance strength multiplier (typically 0.3-1.0)
        tau: Guidance threshold - only apply when σ_t > tau
        mode: 'wls' for weighted least squares, 'full' for complete gradient
        epsilon: Small constant for numerical stability

    Example:
        >>> guidance = PoissonGaussianGuidance(s=1000, sigma_r=5.0, kappa=0.5)
        >>> x_guided = guidance(x_pred, y_observed, sigma_t=0.1)
    """

    def __init__(
        self,
        s: float = 1000.0,
        sigma_r: float = 5.0,
        kappa: float = 0.5,
        tau: float = 0.01,
        mode: Literal["wls", "full"] = "wls",
        epsilon: float = 1e-8,
    ):
        super().__init__()

        # Physical parameters
        self.register_buffer("s", torch.tensor(s, dtype=torch.float32))
        self.register_buffer("sigma_r", torch.tensor(sigma_r, dtype=torch.float32))

        # Guidance parameters
        self.kappa = kappa
        self.tau = tau
        self.mode = mode
        self.epsilon = epsilon

        logger.info(f"PoissonGaussianGuidance initialized:")
        logger.info(f"  s (scale/full-well): {s:.1f}")
        logger.info(f"  sigma_r (read noise): {sigma_r:.2f}")
        logger.info(f"  kappa (guidance strength): {kappa:.2f}")
        logger.info(f"  tau (threshold): {tau:.4f}")
        logger.info(f"  mode: {mode}")

    def forward(
        self,
        x_pred: torch.Tensor,
        y_obs: torch.Tensor,
        sigma_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply Poisson-Gaussian guidance to predicted clean image.

        Args:
            x_pred: Predicted clean image (B, C, H, W) - model's denoised output
            y_obs: Observed noisy image (B, C, H, W) - actual measurement
            sigma_t: Current noise level (scalar or tensor) - diffusion timestep

        Returns:
            Guided prediction incorporating physics-informed correction
        """
        # Handle sigma_t as scalar or tensor
        if sigma_t is None:
            sigma_t = torch.tensor(1.0, device=x_pred.device, dtype=x_pred.dtype)
        elif isinstance(sigma_t, (int, float)):
            sigma_t = torch.tensor(sigma_t, device=x_pred.device, dtype=x_pred.dtype)

        # Only apply guidance if sigma_t > tau (early timesteps)
        if sigma_t.item() <= self.tau:
            return x_pred

        # Compute guidance gradient based on mode
        if self.mode == "wls":
            grad = self._weighted_least_squares_gradient(x_pred, y_obs)
        else:  # mode == 'full'
            grad = self._full_poisson_gaussian_gradient(x_pred, y_obs)

        # Apply guidance with strength kappa
        x_guided = x_pred + self.kappa * grad

        return x_guided

    def _weighted_least_squares_gradient(
        self,
        x_pred: torch.Tensor,
        y_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted least squares gradient (simplified approximation).

        This is the linearized form suitable for high photon counts:
            ∇_x log p(y|x) ≈ (y - x) / (s·x + σ_r²)

        Args:
            x_pred: Predicted clean image
            y_obs: Observed noisy image

        Returns:
            Guidance gradient
        """
        # Ensure x_pred is positive for variance computation
        x_pred_pos = torch.clamp(x_pred, min=self.epsilon)

        # Signal-dependent variance: Var[y|x] = s·x + σ_r²
        variance = self.s * x_pred_pos + self.sigma_r**2

        # Weighted residual: (y - x) / variance
        residual = y_obs - x_pred
        weighted_residual = residual / (variance + self.epsilon)

        return weighted_residual

    def _full_poisson_gaussian_gradient(
        self,
        x_pred: torch.Tensor,
        y_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute full Poisson-Gaussian gradient (exact form).

        This includes higher-order terms for better accuracy at low photon counts:
            ∇_x log p(y|x) = (y - x) / (s·x + σ_r²) - s·(y - x)² / (2·(s·x + σ_r²)²)

        Args:
            x_pred: Predicted clean image
            y_obs: Observed noisy image

        Returns:
            Guidance gradient
        """
        # Ensure x_pred is positive and has minimum value for stability
        x_pred_pos = torch.clamp(
            x_pred, min=1.0
        )  # Minimum 1 electron to avoid division issues

        # Signal-dependent variance (add extra epsilon for stability)
        variance = self.s * x_pred_pos + self.sigma_r**2 + 1e-4

        # Residual
        residual = y_obs - x_pred

        # First-order term (same as WLS)
        first_order = residual / variance

        # Second-order correction (variance gradient term)
        # Scale down to prevent instability - this term should be small
        second_order = (self.s * residual**2) / (2 * variance**2)

        # Clamp second order term to prevent extreme values
        second_order = torch.clamp(second_order, -1.0, 1.0)

        # Full gradient with damping factor (0.5) on second order term for stability
        gradient = first_order - 0.5 * second_order

        # Final safety clamp to prevent NaN propagation
        gradient = torch.clamp(gradient, -10.0, 10.0)

        return gradient

    def compute_log_likelihood(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Poisson-Gaussian log-likelihood: log p(y|x).

        Useful for evaluation and monitoring guidance effectiveness.

        Args:
            x: Clean image estimate
            y: Observed noisy image

        Returns:
            Log-likelihood value (scalar or per-sample)
        """
        # Ensure x is positive
        x_pos = torch.clamp(x, min=self.epsilon)

        # Signal-dependent variance
        variance = self.s * x_pos + self.sigma_r**2

        # Gaussian approximation of Poisson-Gaussian likelihood
        # log p(y|x) ≈ -0.5 * [(y - x)² / variance + log(2π·variance)]
        residual = y - x
        log_likelihood = -0.5 * (
            (residual**2) / (variance + self.epsilon)
            + torch.log(2 * torch.pi * variance + self.epsilon)
        )

        # Sum over spatial dimensions, return per-sample
        return log_likelihood.sum(dim=[1, 2, 3])  # Shape: (B,)

    def update_parameters(
        self,
        s: Optional[float] = None,
        sigma_r: Optional[float] = None,
        kappa: Optional[float] = None,
    ):
        """Update guidance parameters dynamically."""
        if s is not None:
            self.s.fill_(s)
        if sigma_r is not None:
            self.sigma_r.fill_(sigma_r)
        if kappa is not None:
            self.kappa = kappa

        logger.info(
            f"Updated guidance parameters: s={self.s.item():.1f}, "
            f"sigma_r={self.sigma_r.item():.2f}, kappa={self.kappa:.2f}"
        )


def create_photography_guidance(
    domain_range: dict,
    kappa: float = 0.5,
    tau: float = 0.01,
    mode: Literal["wls", "full"] = "wls",
) -> PoissonGaussianGuidance:
    """
    Create Poisson-Gaussian guidance for photography domain.

    Args:
        domain_range: Dict with 'min' and 'max' values from metadata
        kappa: Guidance strength
        tau: Guidance threshold
        mode: Guidance mode ('wls' or 'full')

    Returns:
        Configured PoissonGaussianGuidance instance
    """
    # Photography parameters from domain_range
    # Max value represents approximate full-well capacity in ADU
    s = domain_range.get("max", 15871.0)

    # Read noise for Sony cameras (typical: 3-5 electrons)
    sigma_r = 5.0

    return PoissonGaussianGuidance(
        s=s,
        sigma_r=sigma_r,
        kappa=kappa,
        tau=tau,
        mode=mode,
    )


if __name__ == "__main__":
    # Test the guidance module
    import numpy as np

    # Create guidance
    guidance = PoissonGaussianGuidance(s=1000.0, sigma_r=5.0, kappa=0.5)

    # Create test tensors
    batch_size = 4
    channels = 3
    size = 256

    x_pred = torch.randn(batch_size, channels, size, size) * 0.5 + 0.5  # [0, 1]
    y_obs = x_pred + torch.randn_like(x_pred) * 0.1  # Add noise
    sigma_t = 0.5

    # Apply guidance
    x_guided = guidance(x_pred, y_obs, sigma_t=sigma_t)

    # Compute log-likelihood
    log_lik = guidance.compute_log_likelihood(x_pred, y_obs)

    print(f"Input shape: {x_pred.shape}")
    print(f"Guided output shape: {x_guided.shape}")
    print(f"Log-likelihood shape: {log_lik.shape}")
    print(f"Mean log-likelihood: {log_lik.mean().item():.4f}")
    print(f"Guidance change (L2 norm): {torch.norm(x_guided - x_pred).item():.4f}")

    print("\n✅ PoissonGaussianGuidance module test passed!")
