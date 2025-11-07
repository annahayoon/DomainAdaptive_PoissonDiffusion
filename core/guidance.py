"""Guidance computation modules for Poisson-Gaussian guided diffusion models.

This module contains all guidance-related classes and functions used for physics-informed
guidance in diffusion-based image restoration. It provides:
- GuidanceComputer: Abstract base class for guidance computation
- PoissonGaussianGuidance: Signal-dependent noise modeling
- GaussianGuidance: Simplified constant variance guidance
- Validation utilities for parameter consistency and photon count regimes
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Import utilities from sensor_utils and tensor_utils
from core.utils.sensor_utils import (
    compute_residual_components,
    compute_sensor_range,
    validate_sensor_range_consistency,
    validate_tensor_inputs,
)


class GuidanceComputer(ABC):
    """Abstract base class for likelihood guidance computation."""

    @abstractmethod
    def compute_score(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute likelihood score ∇_x log p(y|x)."""
        pass

    @abstractmethod
    def compute(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        sigma_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute scaled guidance gradient."""
        pass

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic statistics from recent computations."""
        pass


def _apply_guidance_step(
    x0_hat: torch.Tensor,
    gradient: torch.Tensor,
    sigma_t: float,
    kappa: float,
) -> torch.Tensor:
    """Apply guidance step with standard formula."""
    step_size = kappa * (sigma_t**2)
    x0_guided = x0_hat + step_size * gradient
    return torch.clamp(x0_guided, 0.0, 1.0)


def validate_guidance_inputs(x0_hat: torch.Tensor, y_e: torch.Tensor) -> None:
    """Validate that input tensors have compatible shapes and appropriate value ranges."""
    if x0_hat.shape != y_e.shape:
        raise ValueError(f"Shape mismatch: x0_hat {x0_hat.shape} vs y_e {y_e.shape}")

    if torch.any(x0_hat < 0) or torch.any(x0_hat > 1):
        logger.warning("x0_hat values outside [0,1] range detected")

    if torch.any(y_e < 0):
        logger.warning("y_e contains negative values")


class BaseGuidance(nn.Module):
    """Base class for guidance modules with common functionality."""

    def __init__(
        self,
        s: float,
        sigma_r: float,
        black_level: float,
        white_level: float,
        offset: float = 0.0,
        exposure_ratio: float = 1.0,
        kappa: float = 0.5,
        tau: float = 0.01,
        epsilon: float = 1e-8,
        guidance_level: str = "x0",
    ):
        super().__init__()
        validate_sensor_range_consistency(s, black_level, white_level)

        self.s = s
        self.sigma_r = sigma_r
        self.black_level = black_level
        self.white_level = white_level
        self.offset = offset
        self.alpha = exposure_ratio
        self.kappa = kappa
        self.tau = tau
        self.epsilon = epsilon
        self.guidance_level = guidance_level
        self.sigma_r_squared = sigma_r**2
        self.sensor_range = compute_sensor_range(black_level, white_level)

    def _validate_inputs(self, x0_hat: torch.Tensor, y_e: torch.Tensor):
        """Validate inputs for guidance computation."""
        validate_tensor_inputs(
            x0_hat, y_e, self.black_level, self.white_level, self.offset
        )

    @abstractmethod
    def _compute_gradient(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient - must be implemented by subclasses."""
        pass

    def forward(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor, sigma_t: float
    ) -> torch.Tensor:
        """Apply guidance to prediction."""
        if sigma_t <= self.tau:
            return x0_hat

        self._validate_inputs(x0_hat, y_e)
        gradient = self._compute_gradient(x0_hat, y_e)
        return _apply_guidance_step(x0_hat, gradient, sigma_t, self.kappa)

    def compute_likelihood_gradient(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor
    ) -> torch.Tensor:
        """Compute likelihood gradient for score-level guidance."""
        validate_guidance_inputs(x0_hat, y_e)
        if hasattr(self, "_current_sigma_t") and self._current_sigma_t <= self.tau:
            return torch.zeros_like(x0_hat)
        return self._compute_gradient(x0_hat, y_e)


class GaussianGuidance(BaseGuidance):
    """
    Exposure-aware Gaussian likelihood guidance (for comparison)

    Implements the score of a Gaussian likelihood with exposure awareness:
    p(y|x) = N(y | α·s·x, σ_r²I)

    This is a simplified version of PoissonGaussianGuidance that:
    - Uses constant variance (σ_r²) instead of signal-dependent variance
    - BUT accounts for exposure ratio (α) in the forward model
    - Uses the same physical parameters (s, σ_r) as PG guidance

    Args:
        s: Scale factor for numerical stability (must equal sensor_range for unit consistency)
        sigma_r: Read noise standard deviation in physical units
        black_level: Sensor black level (from preprocessing calibration)
        white_level: Sensor white level (from preprocessing calibration)
        exposure_ratio: t_low / t_long (exposure ratio linking short/long exposures)
        kappa: Guidance strength multiplier
        tau: Guidance threshold - only apply when σ_t > tau
        epsilon: Small constant for numerical stability
        guidance_level: 'x0' or 'score' (like PG guidance)
    """

    def _compute_gradient(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute homoscedastic Gaussian gradient with fixed variance.

        Uses mean variance across all pixels for true homoscedastic baseline.
        ∇_x log p(y|x) = α·s·(y - α·s·x) / σ²_mean
        """
        _, _, residual = compute_residual_components(
            x0_hat,
            y_e,
            self.black_level,
            self.white_level,
            self.s,
            self.alpha,
            self.epsilon,
        )
        variance_per_pixel = (
            self.alpha * self.s * x0_hat + self.sigma_r_squared + self.epsilon
        )
        sigma_squared_mean = variance_per_pixel.mean()
        return self.alpha * self.s * residual / (sigma_squared_mean + self.epsilon)


class PoissonGaussianGuidance(BaseGuidance):
    """
    Physics-informed guidance for photon-limited imaging

    Implements the score of the Poisson-Gaussian likelihood:
    ∇_x log p(y_e|x)

    Args:
        s: Scale factor for normalized comparison (s = sensor_range)
        sigma_r: Read noise standard deviation (in physical units)
        black_level: Sensor black level (from preprocessing calibration)
        white_level: Sensor white level (from preprocessing calibration)
        offset: Offset applied to data (0.0 for sensors)
        exposure_ratio: Exposure ratio t_low / t_long (e.g., 0.01)
        kappa: Guidance strength multiplier (typically 0.3-1.0)
        tau: Guidance threshold - only apply when σ_t > tau
        mode: 'wls' for weighted least squares, 'simple' for simple PG, 'exact' for exact PG, 'full' for complete gradient
        epsilon: Small constant for numerical stability
        guidance_level: 'x0' or 'score' (DPS)
        poisson_coeff: Calibrated Poisson coefficient (default: 1.0 for theoretical Poisson)
    """

    def __init__(
        self,
        s: float,
        sigma_r: float,
        black_level: float,
        white_level: float,
        offset: float = 0.0,
        exposure_ratio: float = 1.0,
        kappa: float = 0.5,
        tau: float = 0.01,
        mode: str = "wls",
        epsilon: float = 1e-8,
        guidance_level: str = "x0",
        poisson_coeff: Optional[float] = None,
    ):
        super().__init__(
            s,
            sigma_r,
            black_level,
            white_level,
            offset,
            exposure_ratio,
            kappa,
            tau,
            epsilon,
            guidance_level,
        )
        self.mode = mode
        self.poisson_coeff = poisson_coeff if poisson_coeff is not None else 1.0

    def _compute_gradient(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor
    ) -> torch.Tensor:
        """Compute ∇_x log p(y_e|x) based on mode."""
        if self.mode == "wls":
            return self._wls_gradient(x0_hat, y_e)
        elif self.mode == "simple":
            return self._simple_pg_gradient(x0_hat, y_e)
        elif self.mode == "exact":
            return self._exact_pg_gradient(x0_hat, y_e)
        else:  # mode == 'full'
            return self._full_gradient(x0_hat, y_e)

    def _compute_variance(self, x0_hat: torch.Tensor) -> torch.Tensor:
        """Compute signal-dependent variance at short exposure."""
        mean_at_short_exp = self.alpha * self.s * x0_hat
        variance = (
            self.poisson_coeff * mean_at_short_exp + self.sigma_r_squared + self.epsilon
        )
        return variance

    def _convert_gradient_to_normalized_space(
        self, gradient_sensor_units: torch.Tensor
    ) -> torch.Tensor:
        """Convert gradient from sensor range units to normalized [0,1] space."""
        return self.alpha * self.s * gradient_sensor_units

    def _wls_gradient(
        self, x0_hat: torch.Tensor, y_e_physical: torch.Tensor
    ) -> torch.Tensor:
        """Heteroscedastic PG gradient using weighted least squares (WLS) approximation."""
        y_e_scaled, expected_at_short_exp, residual = compute_residual_components(
            x0_hat,
            y_e_physical,
            self.black_level,
            self.white_level,
            self.s,
            self.alpha,
            self.epsilon,
        )
        variance = self._compute_variance(x0_hat)
        gradient_mu_space = residual / variance
        return self._convert_gradient_to_normalized_space(gradient_mu_space)

    def _simple_pg_gradient(
        self, x0_hat: torch.Tensor, y_e_physical: torch.Tensor
    ) -> torch.Tensor:
        """Simple PG gradient that ignores read noise: ∇ = y/μ - 1"""
        y_e_scaled, expected_at_short_exp, _ = compute_residual_components(
            x0_hat,
            y_e_physical,
            self.black_level,
            self.white_level,
            self.s,
            self.alpha,
            self.epsilon,
        )
        mu = expected_at_short_exp
        gradient_sensor_units = y_e_scaled / (mu + self.epsilon) - 1.0
        return self._convert_gradient_to_normalized_space(gradient_sensor_units)

    def _exact_pg_gradient(
        self, x0_hat: torch.Tensor, y_e_physical: torch.Tensor
    ) -> torch.Tensor:
        """Exact Poisson-Gaussian gradient using exact posterior E[K|Y]."""
        y_e_scaled, expected_at_short_exp, _ = compute_residual_components(
            x0_hat,
            y_e_physical,
            self.black_level,
            self.white_level,
            self.s,
            self.alpha,
            self.epsilon,
        )
        mu = expected_at_short_exp

        max_k_global = int(
            torch.clamp(
                torch.max(
                    mu + 5.0 * torch.sqrt(mu + self.sigma_r_squared + self.epsilon)
                )
                + 1,
                min=10,
                max=200,
            ).item()
        )

        log_mu = torch.log(mu + self.epsilon)
        log_norm = -0.5 * torch.log(2 * np.pi * self.sigma_r_squared + self.epsilon)
        inv_2sigma_r2 = 0.5 / (self.sigma_r_squared + self.epsilon)

        E_K_given_Y = torch.zeros_like(mu)
        normalizer = torch.zeros_like(mu)

        for k in range(0, max_k_global + 1):
            k_float = float(k)
            log_poisson = (
                -mu
                + k_float * log_mu
                - torch.lgamma(torch.tensor(k_float + 1.0, device=mu.device))
            )
            log_gaussian = log_norm - inv_2sigma_r2 * (y_e_scaled - k_float) ** 2
            log_w = log_poisson + log_gaussian
            w = torch.exp(torch.clamp(log_w, min=-50.0, max=50.0))
            E_K_given_Y = E_K_given_Y + k_float * w
            normalizer = normalizer + w

        E_K_given_Y = E_K_given_Y / (normalizer + self.epsilon)
        gradient_sensor_units = E_K_given_Y / (mu + self.epsilon) - 1.0
        return self._convert_gradient_to_normalized_space(gradient_sensor_units)

    def _full_gradient(
        self, x0_hat: torch.Tensor, y_e_physical: torch.Tensor
    ) -> torch.Tensor:
        """Full gradient including variance correction term."""
        y_e_scaled, expected_at_short_exp, residual = compute_residual_components(
            x0_hat,
            y_e_physical,
            self.black_level,
            self.white_level,
            self.s,
            self.alpha,
            self.epsilon,
        )
        variance = self._compute_variance(x0_hat)
        mean_term_sensor_units = residual / variance
        variance_term_sensor_units = -1.0 / (2.0 * variance + self.epsilon)
        gradient_sensor_units = mean_term_sensor_units + variance_term_sensor_units
        return self._convert_gradient_to_normalized_space(gradient_sensor_units)


def select_guidance(regime: str) -> str:
    """Select appropriate guidance method for given regime."""
    if regime == "read_noise_dominated":
        return "gaussian"
    else:
        return "pg"
