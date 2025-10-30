#!/usr/bin/env python3
"""
Gradient computation modules for Poisson-Gaussian guided diffusion models.

This module contains all gradient-related classes and functions used for physics-informed
guidance in diffusion-based image restoration. It provides:
- PoissonGaussianGuidance: Signal-dependent noise modeling
- GaussianGuidance: Simplified constant variance guidance
- Validation utilities for parameter consistency and photon count regimes
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
from sample.sample_utils import (
    PhotonCountValidator,
    compute_residual_components,
    compute_sensor_range,
    normalize_physical_to_normalized,
    validate_sensor_range_consistency,
    validate_tensor_inputs,
)


def _apply_guidance_step(
    x0_hat: torch.Tensor,
    gradient: torch.Tensor,
    sigma_t: float,
    kappa: float,
) -> torch.Tensor:
    """
    Apply guidance step with standard formula.

    Args:
        x0_hat: Current estimate [B,C,H,W]
        gradient: Guidance gradient [B,C,H,W]
        sigma_t: Current noise level
        kappa: Guidance strength multiplier

    Returns:
        Guided estimate clamped to [0,1]
    """
    step_size = kappa * (sigma_t**2)
    x0_guided = x0_hat + step_size * gradient
    return torch.clamp(x0_guided, 0.0, 1.0)


def _initialize_guidance_base(
    self,
    s: float,
    sigma_r: float,
    black_level: float,
    white_level: float,
    offset: float,
    exposure_ratio: float,
    kappa: float,
    tau: float,
    epsilon: float,
    guidance_level: str,
) -> None:
    """
    Initialize common guidance attributes shared by both GaussianGuidance and PoissonGaussianGuidance.

    Args:
        self: Guidance module instance
        s: Scale factor
        sigma_r: Read noise standard deviation
        black_level: Sensor black level
        white_level: Sensor white level
        offset: Data offset
        exposure_ratio: Exposure ratio
        kappa: Guidance strength multiplier
        tau: Guidance threshold
        epsilon: Numerical stability constant
        guidance_level: Guidance level ('x0' or 'score')
    """
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


def validate_guidance_inputs(x0_hat: torch.Tensor, y_e: torch.Tensor) -> None:
    """Shared validation function for guidance inputs.

    Validates that input tensors have compatible shapes and appropriate value ranges.
    This is a simplified version used by compute_likelihood_gradient methods.

    Args:
        x0_hat: Denoised estimate tensor [B, C, H, W] in [0, 1] range
        y_e: Observed measurement tensor [B, C, H, W] in physical units

    Raises:
        ValueError: If tensor shapes don't match
    """
    if x0_hat.shape != y_e.shape:
        raise ValueError(f"Shape mismatch: x0_hat {x0_hat.shape} vs y_e {y_e.shape}")

    if torch.any(x0_hat < 0) or torch.any(x0_hat > 1):
        logger.warning("x0_hat values outside [0,1] range detected")

    if torch.any(y_e < 0):
        logger.warning("y_e contains negative values")


class PhysicsConsistencyValidator:
    """
    Validates physical consistency of guidance parameters.

    UNIT FLOW (Critical):

    1. INPUT UNITS:
       - x_pred: [0,1] normalized (model output)
       - y_observed: [black_level, white_level] physical units (ADU)
       - s: MUST EQUAL sensor_range = white_level - black_level
       - sigma_r: Physical units matching y_observed

    2. CONSTRAINT: s = sensor_range
       Without this, forward model gives mismatched units.
       If s ≠ sensor_range, then y_phys and expected_phys are in different scales.

    3. NORMALIZATION:
       Matches preprocessing pipeline: (y - black_level) / (white_level - black_level)
    """

    @staticmethod
    def validate_parameter_units(
        s: float, black_level: float, white_level: float
    ) -> Dict[str, Any]:
        """Validate unit consistency of guidance parameters."""
        sensor_range = compute_sensor_range(black_level, white_level)

        if abs(s - sensor_range) > 1e-3:
            raise ValueError(
                f"UNIT CONSISTENCY VIOLATION: s={s} must equal sensor_range={sensor_range}\n"
                f"REASON: s scales normalized predictions to physical units.\n"
                f"If s ≠ sensor_range, predicted and observed physical scales don't match.\n"
                f"\n"
                f"FIX: Set s = white_level - black_level = {sensor_range}"
            )

        return {
            "is_valid": True,
            "s": s,
            "sensor_range": sensor_range,
            "black_level": black_level,
            "white_level": white_level,
        }


class GaussianGuidance(nn.Module):
    """
    Exposure-aware Gaussian likelihood guidance (for comparison)

    Implements the score of a Gaussian likelihood with exposure awareness:
    p(y|x) = N(y | α·s·x, σ_r²I)

    This is a simplified version of PoissonGaussianGuidance that:
    - Uses constant variance (σ_r²) instead of signal-dependent variance
    - BUT accounts for exposure ratio (α) in the forward model
    - Uses the same physical parameters (s, σ_r) as PG guidance

    Physical Unit Handling:
    - Works in the same physical space as PoissonGaussianGuidance
    - x0_hat: [0,1] normalized prediction
    - y_e: Physical units (ADU) using sensor calibration (black_level, white_level)
    - Converts to/from physical space for guidance computation using preprocessing normalization

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
        guidance_level: str = "x0",  # 'x0' or 'score'
    ):
        super().__init__()

        # Initialize common guidance attributes
        _initialize_guidance_base(
            self,
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

        logger.info(
            f"GaussianGuidance: s={s}, σ_r={sigma_r}, α={exposure_ratio:.4f}, κ={kappa}"
        )

    def _validate_inputs(self, x0_hat: torch.Tensor, y_e: torch.Tensor):
        """
        Validate inputs for guidance computation.

        Args:
            x0_hat: Current denoised estimate [B,C,H,W], range [0,1]
            y_e: Observed short exposure measurement [B,C,H,W], in physical units

        Raises:
            ValueError: If inputs are invalid
        """
        validate_tensor_inputs(
            x0_hat, y_e, self.black_level, self.white_level, self.offset
        )

    def forward(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor, sigma_t: float
    ) -> torch.Tensor:
        """
        Apply exposure-aware Gaussian likelihood guidance - SAME normalization as PG guidance

        Forward model: y_short = N(α·s·x_long, σ_r²)

        where:
        - x_long (x0_hat): Prediction at LONG exposure (what we want)
        - y_short (y_e): Observation at SHORT exposure (what we have)
        - α (self.alpha): Exposure ratio linking them (t_low / t_long)
        - σ_r: Read noise (constant variance assumption)

        Args:
            x0_hat: Current denoised estimate [B,C,H,W], range [0,1]
            y_e: Observed short exposure measurement [B,C,H,W], in physical units (electrons)
            sigma_t: Current noise level (sigma)

        Returns:
            x0_guided: Guided estimate [B,C,H,W], range [0,1]
        """
        if sigma_t <= self.tau:
            return x0_hat

        self._validate_inputs(x0_hat, y_e)
        gradient = self._compute_gaussian_gradient(x0_hat, y_e)
        return _apply_guidance_step(x0_hat, gradient, sigma_t, self.kappa)

    def _compute_gaussian_gradient(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute homoscedastic Gaussian gradient with fixed variance.

        Paper formula: ∇ = (y - μ)/σ² with fixed σ² = mean(μ + σ_r²)

        Note: Uses mean variance across all pixels for true homoscedastic baseline.
        This differs from using just σ_r² which doesn't match the paper specification.

        ∇_x log p(y|x) = α·s·(y - α·s·x) / σ²_mean

        Args:
            x0_hat: Normalized prediction [B,C,H,W] in [0,1]
            y_e: Physical observation [B,C,H,W] in ADU

        Returns:
            Gradient tensor [B,C,H,W]
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

    def compute_likelihood_gradient(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute likelihood gradient for score-level guidance.

        This is the theoretically pure approach where we add the likelihood
        gradient directly to the score instead of modifying x₀.

        Args:
            x0_hat: Current denoised estimate [B,C,H,W], range [0,1]
            y_e: Observed short exposure measurement [B,C,H,W], in electrons

        Returns:
            likelihood_gradient: ∇_x log p(y|x) [B,C,H,W]
        """
        validate_guidance_inputs(x0_hat, y_e)
        if hasattr(self, "_current_sigma_t") and self._current_sigma_t <= self.tau:
            return torch.zeros_like(x0_hat)
        return self._compute_gaussian_gradient(x0_hat, y_e)


class PoissonGaussianGuidance(nn.Module):
    """
    Physics-informed guidance for photon-limited imaging

    Implements the score of the Poisson-Gaussian likelihood:
    ∇_x log p(y_e|x)

    This tells the diffusion model how to adjust predictions to match
    observed short exposure measurements while respecting physical noise properties.

    Physical Unit Requirements:
    - x_pred must be in [0,1] normalized space
    - y_observed must be in PHYSICAL UNITS (ADU) using sensor calibration
    - s: Scale factor for normalized comparison (s = sensor_range = white_level - black_level)
    - sigma_r: Read noise in same physical units as y_observed
    - black_level, white_level: Sensor calibration values from preprocessing

    The key insight: We normalize BOTH y_observed and expected values to [0, s]
    scale internally, ensuring proper comparison. Normalization matches preprocessing:
    (y - black_level) / (white_level - black_level) → [0,1]

    UNIT CONSISTENCY CLARIFICATION:
    Since s = sensor_range, the normalization chain is:
    1. x ∈ [-1,1] (model space)
    2. x_norm ∈ [0,1] (normalized model space)
    3. x_phys ∈ [black_level, white_level] (physical units)

    The scale factor s = sensor_range ensures that when we compute:
    - y_scaled = y_norm * s (where y_norm = (y_phys - black_level) / sensor_range)
    - expected = alpha * s * x_norm

    Both y_scaled and expected are in the same units as the original y_phys,
    ensuring proper comparison in the likelihood computation.

    Args:
        s: Scale factor for normalized comparison (s = sensor_range)
        sigma_r: Read noise standard deviation (in physical units)
        black_level: Sensor black level (from preprocessing calibration)
        white_level: Sensor white level (from preprocessing calibration)
        offset: Offset applied to data (0.0 for photography)
        exposure_ratio: Exposure ratio t_low / t_long (e.g., 0.01)
        kappa: Guidance strength multiplier (typically 0.3-1.0)
        tau: Guidance threshold - only apply when σ_t > tau
        mode: 'wls' for weighted least squares (heteroscedastic), 'simple' for simple PG, 'exact' for exact PG, 'full' for complete gradient
        epsilon: Small constant for numerical stability
        guidance_level: 'x0' or 'score' (DPS)

    Example:
        >>> # For Sony sensor: black_level=512, white_level=16383
        >>> # sensor_range = 16383 - 512 = 15871
        >>> guidance = PoissonGaussianGuidance(
        ...     s=15871.0, sigma_r=5.0,
        ...     black_level=512.0, white_level=16383.0,
        ...     offset=0.0, kappa=0.5
        ... )
    """

    def __init__(
        self,
        s: float,
        sigma_r: float,
        black_level: float,
        white_level: float,
        offset: float = 0.0,
        exposure_ratio: float = 1.0,  # t_low / t_long
        kappa: float = 0.5,
        tau: float = 0.01,
        mode: str = "wls",
        epsilon: float = 1e-8,
        guidance_level: str = "x0",  # 'x0' (empirically stable, default) or 'score' (DPS, theoretically pure)
    ):
        super().__init__()

        _initialize_guidance_base(
            self,
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

        logger.info(
            f"PoissonGaussianGuidance: s={s}, σ_r={sigma_r}, α={exposure_ratio:.4f}, κ={kappa}, mode={mode}"
        )

    def _validate_inputs(self, x0_hat: torch.Tensor, y_e: torch.Tensor):
        """
        Validate inputs for guidance computation.

        Args:
            x0_hat: Current denoised estimate [B,C,H,W], range [0,1]
            y_e: Observed short exposure measurement [B,C,H,W], in physical units

        Raises:
            ValueError: If inputs are invalid
        """
        validate_tensor_inputs(
            x0_hat, y_e, self.black_level, self.white_level, self.offset
        )

    def forward(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor, sigma_t: float
    ) -> torch.Tensor:
        """
        Apply guidance to prediction

        Args:
            x0_hat: Current denoised estimate [B,C,H,W], range [0,1]
            y_e: Observed short exposure measurement [B,C,H,W], in electrons
            sigma_t: Current noise level (sigma)

        Returns:
            x0_guided: Guided estimate [B,C,H,W], range [0,1]
        """
        if sigma_t <= self.tau:
            return x0_hat

        self._validate_inputs(x0_hat, y_e)
        gradient = self._compute_gradient(x0_hat, y_e)
        return _apply_guidance_step(x0_hat, gradient, sigma_t, self.kappa)

    def _compute_gradient(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ∇_x log p(y_e|x)

        Returns gradient with same shape as x0_hat

        Mode options:
        - 'wls': Heteroscedastic PG (weighted least squares) - ∇ = (y - μ)/(μ + σ_r²)
        - 'full': Full gradient with variance correction term
        - 'simple': Simple PG (ignores read noise) - ∇ = y/μ - 1
        - 'exact': Exact PG using exact Poisson-Gaussian posterior E[K|Y]
        """
        if self.mode == "wls":
            return self._wls_gradient(x0_hat, y_e)
        elif self.mode == "simple":
            return self._simple_pg_gradient(x0_hat, y_e)
        elif self.mode == "exact":
            return self._exact_pg_gradient(x0_hat, y_e)
        else:  # mode == 'full'
            return self._full_gradient(x0_hat, y_e)

    def _compute_variance(self, x0_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute signal-dependent variance at short exposure.

        Variance = α·s·x0_hat + σ_r²

        Args:
            x0_hat: Normalized prediction [B,C,H,W] in [0,1]

        Returns:
            Variance tensor [B,C,H,W]
        """
        return self.alpha * self.s * x0_hat + self.sigma_r_squared + self.epsilon

    def _convert_gradient_to_normalized_space(
        self, gradient_sensor_units: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert gradient from sensor range units to normalized [0,1] space.

        Common pattern: gradient = alpha * s * gradient_sensor_units

        Args:
            gradient_sensor_units: Gradient in sensor range units (∂log p / ∂μ)

        Returns:
            Gradient in normalized [0,1] space (∂log p / ∂x0_hat)
        """
        return self.alpha * self.s * gradient_sensor_units

    def _wls_gradient(
        self, x0_hat: torch.Tensor, y_e_physical: torch.Tensor
    ) -> torch.Tensor:
        """
        Heteroscedastic PG gradient using weighted least squares (WLS) approximation.

        Paper formula (raw gradient in μ-space): ∇_μ log p = (y - μ)/(μ + σ_r²)

        Mathematical derivation:
        - Approximate Poisson-Gaussian as Gaussian: p(y|μ) ≈ N(y; μ, μ + σ_r²)
        - Log-likelihood: log p(y|μ) = -0.5*log(2π(μ + σ_r²)) - 0.5*(y-μ)²/(μ + σ_r²)
        - WLS gradient in μ-space (dropping variance correction): ∇_μ log p ≈ (y-μ)/(μ + σ_r²)
        - Convert to x0_hat space via chain rule: ∇_{x0_hat} = ∇_μ * (∂μ/∂x0_hat) = (y-μ)/(μ + σ_r²) * α·s

        Implementation note:
        - The raw formula (y-μ)/(μ + σ_r²) is computed in sensor range units (μ-space)
        - The scaling factor α·s comes from the chain rule conversion: μ = α·s·x0_hat, so ∂μ/∂x0_hat = α·s
        - This is the correct implementation for applying guidance in normalized [0,1] space

        Forward model: y_short = Poisson(α·s·x_long) + N(0, σ_r²)

        where:
        - x_long (x0_hat): Prediction at LONG exposure (what we want)
        - y_short (y_e_physical): Observation at SHORT exposure (what we have)
        - α (self.alpha): Exposure ratio linking them (t_short / t_long)

        Input units:
          x0_hat: [0,1] normalized long-exposure prediction
          y_e_physical: [black_level, white_level] physical units (ADU)

        Internal computation (working in sensor range units, not absolute ADU):
          1. Normalize y to [0,1]: y_norm = (y_phys - black_level) / (white_level - black_level)
          2. Scale to sensor range [0,s]: y_scaled = y_norm * s  (where s = white_level - black_level)
          3. Expected in sensor range units: μ = α·s·x0_hat  (same scale as y_scaled)
          4. Variance: σ² = α·s·x0_hat + σ_r² (heteroscedastic)
          5. Gradient: ∇ = α·s·(y_scaled - μ) / σ²

        Note: Working in sensor range units [0, s] rather than absolute ADU [black_level, white_level]
        is valid because black_level cancels when computing brightness differences in the residual.

        Output units:
          gradient in [0,1] space (∂log p / ∂x0_hat)

        Physical interpretation:
        - x0_hat: Bright long-exposure prediction
        - α * x0_hat: What that image would look like at short exposure
        - residual: Difference between observed and expected (same units!)
        - Gradient tells model how to adjust bright prediction to match dark observation
        """
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
        """
        Simple PG gradient that ignores read noise.

        Paper formula: ∇ = y/μ - 1

        Mathematical derivation:
        - For pure Poisson: p(y|μ) = exp(-μ) * μ^y / y!
        - Log-likelihood: log p(y|μ) = -μ + y*log(μ) - log(y!)
        - Gradient: ∇_μ log p(y|μ) = -1 + y/μ = y/μ - 1

        This assumes pure Poisson noise (no Gaussian read noise component σ_r²).
        In practice, this is less accurate than heteroscedastic PG but simpler.

        Args:
            x0_hat: Normalized prediction [B,C,H,W] in [0,1]
            y_e_physical: Physical observation [B,C,H,W] in ADU

        Returns:
            Gradient tensor [B,C,H,W]
        """
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
        """
        Exact Poisson-Gaussian gradient using exact posterior E[K|Y].

        Uses series expansion to compute the exact posterior mean:
        E[K|Y] = Σ_{k=0}^∞ k * p(k|y) where p(k|y) ∝ Poisson(k; μ) * N(y; k, σ_r²)

        The gradient is then: ∇ = (E[K|Y]/μ - 1) * α·s

        This is computationally expensive but most accurate at very low photon counts.

        Args:
            x0_hat: Normalized prediction [B,C,H,W] in [0,1]
            y_e_physical: Physical observation [B,C,H,W] in ADU

        Returns:
            Gradient tensor [B,C,H,W]
        """
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
        """
        Full gradient including variance correction term with exposure ratio.

        Forward model: y_low = Poisson(α·s·x_long) + N(0, σ_r²)

        The full gradient is: WLS gradient + variance correction term
        From paper eq. 112: ∇_μ log p(y|μ) = (y-μ)/(μ+σ_r²) - 1/(2(μ+σ_r²))
        """
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
        variance_term_sensor_units = (residual**2) / (2 * variance**2) - 1 / (
            2 * variance
        )
        mean_term = self._convert_gradient_to_normalized_space(mean_term_sensor_units)
        variance_term = self._convert_gradient_to_normalized_space(
            variance_term_sensor_units
        )
        return mean_term + variance_term

    def compute_likelihood_gradient(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute likelihood gradient for score-level guidance.

        This is the theoretically pure approach where we add the likelihood
        gradient directly to the score instead of modifying x₀.

        Args:
            x0_hat: Current denoised estimate [B,C,H,W], range [0,1]
            y_e: Observed short exposure measurement [B,C,H,W], in electrons

        Returns:
            likelihood_gradient: ∇_x log p(y|x) [B,C,H,W]
        """
        validate_guidance_inputs(x0_hat, y_e)
        return self._compute_gradient(x0_hat, y_e)


def compute_log_likelihood_for_test(y_e, x0_hat, pg_guidance, b, c, h, w):
    """
    Compute log-likelihood for a single pixel for gradient verification testing.

    This is a simplified version that computes the likelihood for a single pixel
    to enable finite difference gradient checking.
    """
    x_pixel = x0_hat[b : b + 1, c : c + 1, h : h + 1, w : w + 1]
    y_pixel = y_e[b : b + 1, c : c + 1, h : h + 1, w : w + 1]
    grad = pg_guidance.compute_likelihood_gradient(x_pixel, y_pixel)
    return -0.5 * grad.sum().item()


def test_pg_gradient_correctness(
    metadata_path: Path,
    sensor_type: str,
    short_tile_id: str,
):
    """
    Verify PG guidance gradient against finite differences.

    This test ensures the analytical gradient computation is correct by comparing
    it to numerical gradients computed via finite differences.

    Args:
        metadata_path: Path to metadata JSON file to load sensor calibration (required).
        sensor_type: Sensor type ('sony' or 'fuji') for loading calibration (required).
        short_tile_id: Short exposure tile ID to use for extracting exposure ratio (required).

    Raises:
        ValueError: If metadata_path is not provided or sensor calibration cannot be loaded.
        FileNotFoundError: If metadata_path does not exist.
    """
    if metadata_path is None:
        raise ValueError("metadata_path is required for sensor calibration")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    from sample.sample_utils import get_exposure_ratio, load_metadata_json

    tile_lookup = load_metadata_json(metadata_path)
    exposure_ratio = get_exposure_ratio(short_tile_id, tile_lookup)
    if exposure_ratio <= 0 or exposure_ratio >= 1:
        raise ValueError(
            f"Invalid exposure ratio {exposure_ratio} for tile {short_tile_id}. "
            f"Expected 0 < ratio < 1 (short exposure / long exposure)."
        )

    from sample.sample_utils import load_sensor_calibration_from_metadata

    black_level, white_level = load_sensor_calibration_from_metadata(
        sensor_type, metadata_path
    )

    s = white_level - black_level
    sigma_r = 5.0
    kappa = 0.5
    epsilon = 1e-8

    pg_guidance = PoissonGaussianGuidance(
        s=s,
        sigma_r=sigma_r,
        black_level=black_level,
        white_level=white_level,
        exposure_ratio=exposure_ratio,
        kappa=kappa,
        epsilon=epsilon,
    )

    test_cases = [(1, 1, 32, 32), (1, 3, 16, 16)]

    for batch_size, channels, height, width in test_cases:
        torch.manual_seed(42)
        x0_hat = torch.rand(batch_size, channels, height, width, requires_grad=True)
        y_e = (
            torch.rand(batch_size, channels, height, width)
            * (white_level - black_level)
            + black_level
        )

        grad_analytical = pg_guidance.compute_likelihood_gradient(x0_hat, y_e)

        eps = 1e-5
        grad_numerical = torch.zeros_like(x0_hat)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        x_plus = x0_hat.clone()
                        x_plus[b, c, h, w] += eps
                        log_lik = compute_log_likelihood_for_test(
                            y_e, x0_hat, pg_guidance, b, c, h, w
                        )
                        log_lik_plus = compute_log_likelihood_for_test(
                            y_e, x_plus, pg_guidance, b, c, h, w
                        )
                        grad_numerical[b, c, h, w] = (log_lik_plus - log_lik) / eps

        diff = torch.abs(grad_analytical - grad_numerical).mean()
        max_diff = torch.abs(grad_analytical - grad_numerical).max()

        assert diff < 1e-3, f"Gradient verification failed: mean diff {diff:.6f} > 1e-3"
        assert (
            max_diff < 1e-2
        ), f"Gradient verification failed: max diff {max_diff:.6f} > 1e-2"


def run_gradient_verification(
    metadata_path: Optional[Path] = None, sensor_type: Optional[str] = None
):
    """
    Run gradient verification tests if requested.

    Args:
        metadata_path: Path to metadata JSON file to load sensor calibration (required).
        sensor_type: Sensor type ('sony' or 'fuji') for loading calibration (required).

    Raises:
        ValueError: If metadata_path, sensor_type, or short_tile_id are not provided when --test_gradients is used.
    """
    import argparse
    import sys

    if "--test_gradients" in sys.argv:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--test_gradients",
            action="store_true",
            help="Run gradient verification tests",
        )
        parser.add_argument(
            "--metadata_json",
            type=str,
            required=True,
            help="Path to metadata JSON file for sensor calibration (required)",
        )
        parser.add_argument(
            "--sensor_type",
            type=str,
            required=True,
            help="Sensor type (sony or fuji) (required)",
        )
        parser.add_argument(
            "--short_tile_id",
            type=str,
            required=True,
            help="Short exposure tile ID to use for extracting exposure ratio (required)",
        )
        args, _ = parser.parse_known_args()

        if not args.metadata_json:
            raise ValueError(
                "--metadata_json is required when running gradient verification tests"
            )
        if not args.sensor_type:
            raise ValueError(
                "--sensor_type is required when running gradient verification tests"
            )
        if not args.short_tile_id:
            raise ValueError(
                "--short_tile_id is required when running gradient verification tests"
            )

        test_pg_gradient_correctness(
            metadata_path=Path(args.metadata_json),
            sensor_type=args.sensor_type,
            short_tile_id=args.short_tile_id,
        )
        sys.exit(0)
