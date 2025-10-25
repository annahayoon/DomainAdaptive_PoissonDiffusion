"""
Poisson-Gaussian likelihood guidance for diffusion sampling.

This module implements the core physics-aware guidance system that incorporates
exact Poisson-Gaussian likelihood into diffusion sampling. It supports both
WLS (Weighted Least Squares) and exact likelihood modes with configurable
scheduling and numerical stability measures.

Mathematical Foundation:
- WLS Mode: ∇ log p(y|x) = s·(y - λ)/(λ + σ_r²)
- Exact Mode: Includes variance derivative terms
- Guidance Weighting: γ(σ) = κ·σ² (or linear/constant)

Requirements addressed: 1.1-1.4, 4.5 from requirements.md
Task: 2.2 from tasks.md
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .error_handlers import safe_operation
from .exceptions import GuidanceError, NumericalStabilityError
from .guidance_config import GuidanceConfig
from .interfaces import GuidanceComputer
from .logging_config import get_logger

logger = get_logger(__name__)


class PoissonGuidance(GuidanceComputer):
    """
    Poisson-Gaussian likelihood guidance for diffusion sampling.

    This class implements physics-aware guidance that incorporates the exact
    Poisson-Gaussian noise model into diffusion sampling. It supports multiple
    modes and scheduling options for different use cases.
    """

    def __init__(
        self,
        scale: float,
        background: float = 0.0,
        read_noise: float = 0.0,
        config: Optional[GuidanceConfig] = None,
    ):
        """
        Initialize Poisson-Gaussian guidance.

        Args:
            scale: Dataset normalization scale (electrons)
            background: Background offset (electrons)
            read_noise: Read noise standard deviation (electrons)
            config: Guidance configuration (uses default if None)
        """
        self.scale = scale
        self.background = background
        self.read_noise = read_noise
        self.config = config or GuidanceConfig()

        # Validate parameters
        self._validate_parameters()

        # Diagnostic storage
        if self.config.collect_diagnostics:
            self.grad_norms: deque = deque(maxlen=self.config.max_diagnostics)
            self.chi2_values: deque = deque(maxlen=self.config.max_diagnostics)
            self.snr_values: deque = deque(maxlen=self.config.max_diagnostics)
            self.gamma_values: deque = deque(maxlen=self.config.max_diagnostics)
        else:
            self.grad_norms = None
            self.chi2_values = None
            self.snr_values = None
            self.gamma_values = None

        logger.info(
            f"PoissonGuidance initialized: scale={scale:.1f} e⁻, "
            f"read_noise={read_noise:.1f} e⁻, mode={self.config.mode}"
        )

    def _validate_parameters(self):
        """Validate initialization parameters."""
        errors = []

        if self.scale <= 0:
            errors.append(f"Scale must be positive, got {self.scale}")
        if self.read_noise < 0:
            errors.append(f"Read noise must be non-negative, got {self.read_noise}")
        if self.background < 0:
            errors.append(f"Background should be non-negative, got {self.background}")

        if errors:
            raise GuidanceError(f"Invalid parameters: {'; '.join(errors)}")

    @safe_operation("Poisson-Gaussian score computation")
    def compute_score(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute likelihood score ∇_x log p(y|x).

        This is the core physics computation that implements the exact
        Poisson-Gaussian likelihood gradient.

        Args:
            x_hat: Current estimate [B, C, H, W] (normalized [0,1])
            y_observed: Observed data [B, C, H, W] (electrons)
            mask: Valid pixel mask [B, C, H, W] (optional)
            **kwargs: Additional parameters (unused)

        Returns:
            Score gradient [B, C, H, W]
        """
        # Validate inputs
        self._validate_tensors(x_hat, y_observed, mask)

        # Convert normalized estimate to electron space
        lambda_e = self.scale * x_hat + self.background

        # Compute variance under Poisson-Gaussian model
        # Var = λ + σ_r² (Poisson shot noise + Gaussian read noise)
        variance = lambda_e + self.read_noise**2

        # Add numerical stability epsilon - use read noise level as minimum
        min_variance = max(self.config.variance_eps, self.read_noise**2 * 0.1)
        variance = torch.clamp(variance, min=min_variance)

        # Compute score based on mode
        if self.config.mode == "wls":
            # Weighted Least Squares approximation
            # ∇ log p(y|x) ≈ (y - λ) / Var
            score = (y_observed - lambda_e) / variance

        elif self.config.mode == "exact":
            # Exact heteroscedastic likelihood with variance derivatives
            # ∇ log p(y|x) = (y - λ)/Var + 0.5 * [(y - λ)²/Var² - 1/Var]
            residual = y_observed - lambda_e
            score = residual / variance + 0.5 * (
                residual**2 / variance**2 - 1.0 / variance
            )
        else:
            raise GuidanceError(f"Unknown guidance mode: {self.config.mode}")

        # Apply mask if provided
        if mask is not None and self.config.enable_masking:
            score = score * mask

        # Scale by s (chain rule: ∂λ/∂x = s)
        gradient = score * self.scale

        # Apply simple gradient clipping for numerical stability
        if self.config.gradient_clip > 0:
            gradient = torch.clamp(gradient, -self.config.gradient_clip, self.config.gradient_clip)

        # Optional gradient normalization
        if self.config.normalize_gradients:
            # Compute norm over spatial dimensions for each batch element
            grad_norm = torch.norm(
                gradient.view(gradient.shape[0], -1), dim=1, keepdim=True
            )
            grad_norm = grad_norm.view(-1, 1, 1, 1)  # Reshape for broadcasting
            gradient = gradient / torch.clamp(grad_norm, min=1e-8)

        # Collect diagnostics
        if self.config.collect_diagnostics:
            self._collect_score_diagnostics(
                gradient, y_observed, lambda_e, variance, mask
            )

        return gradient

    def gamma_schedule(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute guidance weight γ(σ).

        The guidance weight balances the prior and likelihood terms.
        Different schedules are appropriate for different scenarios.

        Args:
            sigma: Current noise level [B] or scalar

        Returns:
            Guidance weight [B] or scalar
        """
        kappa = self.config.kappa

        # Adaptive kappa based on signal level (experimental)
        if self.config.adaptive_kappa:
            # Reduce guidance strength for very low signals
            signal_level = self.scale * 0.5  # Rough estimate
            if signal_level < 10:  # Very low photon regime
                kappa = kappa * 0.5

        if self.config.gamma_schedule == "sigma2":
            # γ(σ) = κ·σ² (default, balances prior and likelihood)
            gamma = kappa * (sigma**2)
        elif self.config.gamma_schedule == "linear":
            # γ(σ) = κ·σ (linear decay with noise level)
            gamma = kappa * sigma
        elif self.config.gamma_schedule == "const":
            # γ(σ) = κ (constant weight)
            gamma = torch.full_like(sigma, kappa)
        else:
            raise GuidanceError(f"Unknown gamma schedule: {self.config.gamma_schedule}")

        return gamma

    @safe_operation("Guidance computation")
    def compute(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        sigma_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute scaled guidance gradient.

        This is the main interface that combines score computation with
        scheduling and returns the final guidance gradient.

        Args:
            x_hat: Current estimate [B, C, H, W] (normalized)
            y_observed: Observed data [B, C, H, W] (electrons)
            sigma_t: Current noise level [B] or scalar
            mask: Valid pixel mask [B, C, H, W] (optional)
            **kwargs: Additional parameters

        Returns:
            Guidance gradient [B, C, H, W]
        """
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

        # Collect diagnostics
        if self.config.collect_diagnostics:
            self._collect_guidance_diagnostics(guidance, gamma)

        return guidance

    def get_diagnostics(self) -> Dict[str, float]:
        """
        Get diagnostic statistics from recent computations.

        Returns:
            Dictionary of diagnostic statistics
        """
        if not self.config.collect_diagnostics:
            return {"diagnostics_disabled": True}

        diagnostics = {}

        # Gradient statistics
        if self.grad_norms and len(self.grad_norms) > 0:
            grad_norms = list(self.grad_norms)
            diagnostics.update(
                {
                    "grad_norm_mean": float(np.mean(grad_norms)),
                    "grad_norm_std": float(np.std(grad_norms)),
                    "grad_norm_max": float(np.max(grad_norms)),
                    "grad_norm_min": float(np.min(grad_norms)),
                }
            )

        # Chi-squared statistics
        if self.chi2_values and len(self.chi2_values) > 0:
            chi2_vals = list(self.chi2_values)
            diagnostics.update(
                {
                    "chi2_mean": float(np.mean(chi2_vals)),
                    "chi2_std": float(np.std(chi2_vals)),
                    "chi2_median": float(np.median(chi2_vals)),
                }
            )

        # SNR statistics
        if self.snr_values and len(self.snr_values) > 0:
            snr_vals = list(self.snr_values)
            diagnostics.update(
                {
                    "snr_mean_db": float(np.mean(snr_vals)),
                    "snr_std_db": float(np.std(snr_vals)),
                    "snr_min_db": float(np.min(snr_vals)),
                }
            )

        # Gamma statistics
        if self.gamma_values and len(self.gamma_values) > 0:
            gamma_vals = list(self.gamma_values)
            diagnostics.update(
                {
                    "gamma_mean": float(np.mean(gamma_vals)),
                    "gamma_std": float(np.std(gamma_vals)),
                    "gamma_max": float(np.max(gamma_vals)),
                }
            )

        # Configuration info
        diagnostics.update(
            {
                "mode": self.config.mode,
                "gamma_schedule": self.config.gamma_schedule,
                "kappa": self.config.kappa,
                "scale": self.scale,
                "read_noise": self.read_noise,
                "num_samples": len(self.grad_norms) if self.grad_norms else 0,
            }
        )

        return diagnostics

    def reset_diagnostics(self):
        """Reset diagnostic collections."""
        if self.config.collect_diagnostics:
            if self.grad_norms:
                self.grad_norms.clear()
            if self.chi2_values:
                self.chi2_values.clear()
            if self.snr_values:
                self.snr_values.clear()
            if self.gamma_values:
                self.gamma_values.clear()

    def _validate_tensors(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """Validate input tensors."""
        # Check shapes
        if x_hat.shape != y_observed.shape:
            raise GuidanceError(
                f"Shape mismatch: x_hat {x_hat.shape} vs y_observed {y_observed.shape}"
            )

        if mask is not None and mask.shape != x_hat.shape:
            raise GuidanceError(
                f"Mask shape {mask.shape} doesn't match input {x_hat.shape}"
            )

        # Check for NaN/Inf
        if torch.isnan(x_hat).any() or torch.isinf(x_hat).any():
            raise NumericalStabilityError("x_hat contains NaN or Inf values")

        if torch.isnan(y_observed).any() or torch.isinf(y_observed).any():
            raise NumericalStabilityError("y_observed contains NaN or Inf values")

        # Check value ranges
        if x_hat.min() < -0.1 or x_hat.max() > 1.1:
            logger.warning(
                f"x_hat values outside [0,1]: [{x_hat.min():.3f}, {x_hat.max():.3f}]"
            )

        if y_observed.min() < -1.0:
            logger.warning(
                f"y_observed has negative values: min={y_observed.min():.3f}"
            )

    def _collect_score_diagnostics(
        self,
        gradient: torch.Tensor,
        y_observed: torch.Tensor,
        lambda_e: torch.Tensor,
        variance: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """Collect diagnostics from score computation."""
        with torch.no_grad():
            # Gradient norm
            grad_norm = torch.norm(gradient).item()
            self.grad_norms.append(grad_norm)

            # Chi-squared per pixel
            residual_sq = (y_observed - lambda_e) ** 2
            chi2_per_pixel = residual_sq / variance

            if mask is not None:
                chi2_mean = (chi2_per_pixel * mask).sum() / mask.sum()
            else:
                chi2_mean = chi2_per_pixel.mean()

            self.chi2_values.append(chi2_mean.item())

            # SNR estimation
            signal = lambda_e.mean()
            noise_std = torch.sqrt(variance.mean())
            snr_db = 20 * torch.log10(torch.clamp(signal / noise_std, min=1e-10))
            self.snr_values.append(snr_db.item())

    def _collect_guidance_diagnostics(
        self, guidance: torch.Tensor, gamma: torch.Tensor
    ):
        """Collect diagnostics from guidance computation."""
        with torch.no_grad():
            if isinstance(gamma, torch.Tensor):
                if gamma.numel() == 1:
                    gamma_val = gamma.item()
                else:
                    gamma_val = gamma.mean().item()
            else:
                gamma_val = float(gamma)

            self.gamma_values.append(gamma_val)

    def estimate_optimal_kappa(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        sigma_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_chi2: float = 1.0,
    ) -> float:
        """
        Estimate optimal kappa value based on chi-squared target.

        This is an experimental feature for adaptive guidance strength.

        Args:
            x_hat: Current estimate
            y_observed: Observed data
            sigma_t: Current noise level
            mask: Valid pixel mask
            target_chi2: Target chi-squared value

        Returns:
            Estimated optimal kappa
        """
        with torch.no_grad():
            # Compute current chi-squared
            lambda_e = self.scale * x_hat + self.background
            variance = lambda_e + self.read_noise**2
            variance = torch.clamp(variance, min=self.config.variance_eps)

            chi2_per_pixel = ((y_observed - lambda_e) ** 2) / variance

            if mask is not None:
                current_chi2 = (chi2_per_pixel * mask).sum() / mask.sum()
            else:
                current_chi2 = chi2_per_pixel.mean()

            # Simple heuristic: adjust kappa to reach target chi2
            if current_chi2 > target_chi2:
                # Too much mismatch, increase guidance
                optimal_kappa = self.config.kappa * (current_chi2 / target_chi2)
            else:
                # Good fit, maybe reduce guidance slightly
                optimal_kappa = self.config.kappa * 0.9

            # Clamp to reasonable range
            optimal_kappa = max(0.1, min(2.0, float(optimal_kappa)))

            return optimal_kappa

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get comprehensive configuration summary."""
        return {
            "guidance_config": self.config.to_dict(),
            "physics_parameters": {
                "scale": self.scale,
                "background": self.background,
                "read_noise": self.read_noise,
            },
            "diagnostics_enabled": self.config.collect_diagnostics,
            "num_diagnostic_samples": len(self.grad_norms) if self.grad_norms else 0,
        }


# Convenience functions
def create_poisson_guidance(
    scale: float,
    background: float = 0.0,
    read_noise: float = 0.0,
    preset: str = "default",
    **config_overrides,
) -> PoissonGuidance:
    """
    Convenience function to create PoissonGuidance with preset configuration.

    Args:
        scale: Dataset normalization scale (electrons)
        background: Background offset (electrons)
        read_noise: Read noise standard deviation (electrons)
        preset: Configuration preset name
        **config_overrides: Configuration parameter overrides

    Returns:
        Configured PoissonGuidance instance
    """
    from .guidance_config import create_guidance_config

    config = create_guidance_config(preset, **config_overrides)
    return PoissonGuidance(scale, background, read_noise, config)


def create_domain_guidance(
    scale: float,
    background: float = 0.0,
    read_noise: float = 0.0,
    domain: str = "photography",
    **config_overrides,
) -> PoissonGuidance:
    """
    Create domain-optimized PoissonGuidance.

    Args:
        scale: Dataset normalization scale (electrons)
        background: Background offset (electrons)
        read_noise: Read noise standard deviation (electrons)
        domain: Domain name ('photography', 'microscopy', 'astronomy')
        **config_overrides: Configuration parameter overrides

    Returns:
        Domain-optimized PoissonGuidance instance
    """
    from .guidance_config import GuidancePresets

    config = GuidancePresets.for_domain(domain)
    if config_overrides:
        config = config.copy(**config_overrides)

    return PoissonGuidance(scale, background, read_noise, config)
