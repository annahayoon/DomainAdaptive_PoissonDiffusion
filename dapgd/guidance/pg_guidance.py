"""
Poisson-Gaussian Guidance for Diffusion Models

This module implements Equation 3 from the DAPGD paper:
The score (gradient) of the Poisson-Gaussian log-likelihood.

KEY INSIGHT: The variance in photon-limited imaging is signal-dependent:
Var[y|x] = s·x + σ_r²

This heteroscedasticity requires adaptive weighting - we cannot use uniform L2 loss.

PHYSICAL INTERPRETATION:
- Residual (y_e - s·x): prediction error
- Variance (s·x + σ_r²): local noise level (signal-dependent!)
- s scaling: convert back to [0,1] space

This naturally:
- Makes small corrections in bright regions (high variance denominator)
- Makes large corrections in dark regions (low variance denominator)
"""

import logging
from typing import Literal, Optional, Tuple

import numpy as np
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
        s: float,
        sigma_r: float,
        kappa: float = 0.5,
        tau: float = 0.01,
        mode: Literal["wls", "full"] = "wls",
        epsilon: float = 1e-8,
    ):
        super().__init__()

        # Validate inputs
        if s <= 0:
            raise ValueError(f"Scale factor s must be positive, got {s}")
        if sigma_r < 0:
            raise ValueError(f"Read noise sigma_r must be non-negative, got {sigma_r}")
        if not 0 < kappa <= 2:
            logger.warning(f"Unusual kappa value: {kappa}. Typical range is 0.3-1.0")
        if mode not in ["wls", "full"]:
            raise ValueError(f"Mode must be 'wls' or 'full', got {mode}")

        # Store as buffers (moved to correct device automatically)
        self.register_buffer("s", torch.tensor(s, dtype=torch.float32))
        self.register_buffer("sigma_r", torch.tensor(sigma_r, dtype=torch.float32))
        self.register_buffer("kappa", torch.tensor(kappa, dtype=torch.float32))
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.float32))

        self.epsilon = epsilon
        self.mode = mode

        logger.info(
            f"Initialized PG Guidance: s={s}, σ_r={sigma_r}, "
            f"κ={kappa}, τ={tau}, mode={mode}"
        )

    def forward(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor, sigma_t: float
    ) -> torch.Tensor:
        """
        Apply guidance to prediction

        Args:
            x0_hat: Current denoised estimate [B,C,H,W], range [0,1]
            y_e: Observed noisy measurement [B,C,H,W], in electrons
            sigma_t: Current noise level (sigma)

        Returns:
            x0_guided: Guided estimate [B,C,H,W], range [0,1]
        """

        # Check if guidance should be applied
        if sigma_t <= self.tau:
            return x0_hat

        # Validate inputs
        self._validate_inputs(x0_hat, y_e)

        # Compute gradient
        gradient = self._compute_gradient(x0_hat, y_e)

        # Apply guidance with schedule
        # Schedule: κ · σ_t² · ∇
        # Larger steps when noise is high, smaller when low
        step_size = self.kappa * (sigma_t**2)

        x0_guided = x0_hat + step_size * gradient

        # Clamp to valid range [0, 1]
        x0_guided = torch.clamp(x0_guided, 0.0, 1.0)

        return x0_guided

    def _compute_gradient(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ∇_x log p(y_e|x)

        Returns gradient with same shape as x0_hat
        """

        if self.mode == "wls":
            return self._wls_gradient(x0_hat, y_e)
        else:  # mode == 'full'
            return self._full_gradient(x0_hat, y_e)

    def _wls_gradient(self, x0_hat: torch.Tensor, y_e: torch.Tensor) -> torch.Tensor:
        """
        Weighted Least Squares gradient (Equation 3, first term)

        Formula: s · (y_e - s·x) / (s·x + σ_r²)

        Physical interpretation:
        - Residual (y_e - s·x): prediction error
        - Variance (s·x + σ_r²): local noise level (signal-dependent!)
        - s scaling: convert back to [0,1] space

        This naturally:
        - Makes small corrections in bright regions (high variance denominator)
        - Makes large corrections in dark regions (low variance denominator)
        """

        # Expected measurement if x0_hat were true
        expected_y = self.s * x0_hat

        # Residual: how far are we from observation?
        residual = y_e - expected_y

        # Local variance (KEY: signal-dependent!)
        # Add epsilon to prevent division by zero
        variance = self.s * x0_hat + self.sigma_r**2 + self.epsilon

        # Gradient: weighted residual
        gradient = self.s * residual / variance

        return gradient

    def _full_gradient(self, x0_hat: torch.Tensor, y_e: torch.Tensor) -> torch.Tensor:
        """
        Full gradient including variance term (complete Equation 3)

        Adds second-order correction:
        s² · (y_e - s·x)² / (2·(s·x + σ_r²)²) - s² / (2·(s·x + σ_r²))

        In practice, this term is 10-100× smaller than mean term.
        Use for ablation studies to show WLS is sufficient.
        """

        expected_y = self.s * x0_hat
        residual = y_e - expected_y
        variance = self.s * x0_hat + self.sigma_r**2 + self.epsilon

        # Mean term (same as WLS)
        mean_term = self.s * residual / variance

        # Variance term (second-order correction)
        variance_term = (self.s**2) * (residual**2) / (2 * variance**2) - (
            self.s**2
        ) / (2 * variance)

        return mean_term + variance_term

    def _validate_inputs(self, x0_hat: torch.Tensor, y_e: torch.Tensor):
        """
        Validate input tensors

        PURPOSE: Catch common errors early with helpful messages
        """

        if x0_hat.shape != y_e.shape:
            raise ValueError(
                f"Shape mismatch: x0_hat {x0_hat.shape} vs y_e {y_e.shape}"
            )

        if torch.isnan(x0_hat).any():
            raise ValueError("x0_hat contains NaN values")

        if torch.isnan(y_e).any():
            raise ValueError("y_e contains NaN values")

        # Check range of x0_hat (should be [0,1])
        if x0_hat.min() < -0.1 or x0_hat.max() > 1.1:
            logger.warning(
                f"x0_hat range [{x0_hat.min():.3f}, {x0_hat.max():.3f}] "
                "is outside expected [0,1]. Consider clamping before guidance."
            )

        # Check y_e is non-negative (physical constraint)
        if y_e.min() < 0:
            logger.warning(
                f"y_e has negative values (min={y_e.min():.3f}). "
                "This is unphysical for photon counts."
            )

    def get_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get expected variance for intensity x

        PURPOSE: Utility for computing chi-squared test

        Args:
            x: Intensity in [0,1] range

        Returns:
            variance: Expected variance in electron space
        """

        return self.s * x + self.sigma_r**2

    def extra_repr(self) -> str:
        """String representation for print() and logging"""
        return (
            f"s={self.s.item():.1f}, σ_r={self.sigma_r.item():.2f}, "
            f"κ={self.kappa.item():.2f}, τ={self.tau.item():.3f}, mode={self.mode}"
        )


# Utility functions for testing and data simulation


def simulate_poisson_gaussian_noise(
    clean_image: torch.Tensor, s: float, sigma_r: float, seed: Optional[int] = None
) -> torch.Tensor:
    """
    Simulate Poisson-Gaussian noise for testing

    PURPOSE: Generate synthetic noisy data with known ground truth

    Args:
        clean_image: Clean image in [0,1] range
        s: Scale factor
        sigma_r: Read noise
        seed: Random seed for reproducibility

    Returns:
        noisy_image: Noisy observation in electron space

    Example:
        >>> clean = torch.rand(1, 3, 256, 256)
        >>> noisy = simulate_poisson_gaussian_noise(clean, s=1000, sigma_r=5)
    """

    if seed is not None:
        torch.manual_seed(seed)

    # Poisson noise (photon arrival)
    photon_count = s * clean_image
    noisy = torch.poisson(photon_count)

    # Gaussian read noise (sensor electronics)
    read_noise = sigma_r * torch.randn_like(clean_image)
    noisy = noisy + read_noise

    return noisy


def compute_chi_squared(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    s: float,
    sigma_r: float,
    return_per_pixel: bool = False,
) -> torch.Tensor:
    """
    Compute chi-squared statistic for physical validation

    PURPOSE: Validate that restoration is physically consistent
    A well-calibrated method achieves χ²_red ≈ 1.0

    Args:
        predicted: Restored image [B,C,H,W] in [0,1] range
        observed: Noisy observation [B,C,H,W] in electrons
        s: Scale factor
        sigma_r: Read noise
        return_per_pixel: If True, return per-pixel chi-squared

    Returns:
        chi2_reduced: Reduced chi-squared value
    """

    # Forward project prediction
    expected = s * predicted

    # Compute variance (signal-dependent)
    variance = s * predicted + sigma_r**2

    # Compute chi-squared
    residual = observed - expected
    chi2_per_pixel = (residual**2) / variance

    if return_per_pixel:
        return chi2_per_pixel

    # Sum and normalize by degrees of freedom
    chi2_total = chi2_per_pixel.sum().item()
    dof = observed.numel()

    chi2_reduced = chi2_total / dof

    return chi2_reduced


def analyze_residual_statistics(
    predicted: torch.Tensor, observed: torch.Tensor, s: float, sigma_r: float
) -> dict:
    """
    Analyze residual statistics for diagnostics

    PURPOSE: Detailed analysis for debugging and validation

    Returns dictionary with:
    - chi2_reduced: Overall chi-squared
    - residual_mean: Mean residual (should be ~0)
    - residual_std: Std of normalized residuals (should be ~1)
    - outlier_fraction: Fraction of pixels with |residual| > 3σ
    """

    # Forward project
    expected = s * predicted
    variance = s * predicted + sigma_r**2

    # Residuals
    residual = observed - expected
    normalized_residual = residual / torch.sqrt(variance)

    # Chi-squared
    chi2 = compute_chi_squared(predicted, observed, s, sigma_r)

    # Statistics
    stats = {
        "chi2_reduced": chi2,
        "residual_mean": residual.mean().item(),
        "residual_std": residual.std().item(),
        "normalized_residual_mean": normalized_residual.mean().item(),
        "normalized_residual_std": normalized_residual.std().item(),
        "outlier_fraction": (normalized_residual.abs() > 3).float().mean().item(),
    }

    return stats


# Factory functions for easy creation


def create_guidance_from_domain(
    domain: str,
    kappa: float = 0.5,
    tau: float = 0.01,
    mode: Literal["wls", "full"] = "wls",
) -> PoissonGaussianGuidance:
    """
    Create guidance with domain-appropriate parameters

    Args:
        domain: Domain type ('photo', 'micro', 'astro')
        kappa: Guidance strength
        tau: Guidance threshold
        mode: Gradient computation mode

    Returns:
        Configured PoissonGaussianGuidance instance

    Raises:
        ValueError: If domain not recognized
    """

    # Domain-specific parameters (from our YAML configs)
    domain_params = {
        "photo": {"s": 79351.0, "sigma_r": 3.6, "background": 0.0},
        "micro": {"s": 65534.0, "sigma_r": 1.5, "background": 0.0},
        "astro": {"s": 121.0, "sigma_r": 3.5, "background": 3.5},
    }

    if domain not in domain_params:
        raise ValueError(
            f"Unknown domain: {domain}. Expected one of {list(domain_params.keys())}"
        )

    params = domain_params[domain]

    logger.info(f"Creating {domain} guidance: s={params['s']}, σ_r={params['sigma_r']}")

    return PoissonGaussianGuidance(
        s=params["s"], sigma_r=params["sigma_r"], kappa=kappa, tau=tau, mode=mode
    )


# Test functions for validation


def test_gradient_numerically(
    guidance: PoissonGaussianGuidance,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 1e-4,
    tolerance: float = 0.05,  # 5% tolerance for numerical verification
) -> dict:
    """
    Verify gradient computation using finite differences

    PURPOSE: Validate that analytical gradient matches numerical gradient

    Returns dictionary with error metrics
    """

    # Analytical gradient
    grad_analytical = guidance._compute_gradient(x, y)

    # Numerical gradient (slow - only check a few pixels)
    grad_numerical = torch.zeros_like(x)

    # Only check center region (faster)
    h, w = x.shape[2:4]
    h_start, w_start = h // 4, w // 4
    h_end, w_end = 3 * h // 4, 3 * w // 4

    def log_likelihood(x_test):
        """Compute log p(y|x)"""
        expected = guidance.s * x_test
        variance = guidance.s * x_test + guidance.sigma_r**2 + guidance.epsilon
        residual = y - expected
        return -0.5 * (residual**2 / variance).sum()

    count = 0
    for i in range(h_start, h_end, 4):  # Subsample
        for j in range(w_start, w_end, 4):
            x_plus = x.clone()
            x_plus[0, 0, i, j] += epsilon

            x_minus = x.clone()
            x_minus[0, 0, i, j] -= epsilon

            grad_numerical[0, 0, i, j] = (
                log_likelihood(x_plus) - log_likelihood(x_minus)
            ) / (2 * epsilon)
            count += 1

    # Compare
    mask = grad_numerical != 0
    if mask.sum() == 0:
        return {"error": float("nan"), "count": 0}

    abs_error = (grad_analytical[mask] - grad_numerical[mask]).abs()
    rel_error = abs_error / (grad_numerical[mask].abs() + 1e-10)

    return {
        "mean_abs_error": abs_error.mean().item(),
        "max_abs_error": abs_error.max().item(),
        "mean_rel_error": rel_error.mean().item(),
        "max_rel_error": rel_error.max().item(),
        "num_checked": count,
        "within_tolerance": rel_error.mean().item() < tolerance,
    }


def test_guidance_properties():
    """Test that guidance behaves as expected"""

    logger.info("Testing PG guidance properties...")

    # Create test guidance
    guidance = PoissonGaussianGuidance(s=1000, sigma_r=5.0, kappa=0.5)

    # Test 1: Gradient direction
    x = torch.ones(1, 1, 4, 4) * 0.5  # x = 0.5
    y_high = torch.ones(1, 1, 4, 4) * 600  # y = 600 > 500 = s*x
    y_low = torch.ones(1, 1, 4, 4) * 400  # y = 400 < 500 = s*x

    grad_high = guidance._compute_gradient(x, y_high)
    grad_low = guidance._compute_gradient(x, y_low)

    assert (grad_high > 0).all(), "Gradient should be positive when y > s*x"
    assert (grad_low < 0).all(), "Gradient should be negative when y < s*x"
    logger.info("✓ Gradient direction test passed")

    # Test 2: Gradient at optimum
    y_opt = torch.ones(1, 1, 4, 4) * 500  # Exactly s*x
    grad_opt = guidance._compute_gradient(x, y_opt)
    assert grad_opt.abs().max() < 0.01, "Gradient should be small at optimum"
    logger.info("✓ Optimum test passed")

    # Test 3: Full application
    x_guided = guidance(x, y_low, sigma_t=0.1)
    assert x_guided.shape == x.shape, "Output shape should match input"
    assert not torch.isnan(x_guided).any(), "Output should not contain NaN"
    logger.info("✓ Full application test passed")

    # Test 4: Numerical gradient verification
    x_test = torch.rand(1, 1, 8, 8) * 0.5 + 0.25
    y_test = simulate_poisson_gaussian_noise(x_test, s=1000, sigma_r=5.0, seed=42)

    errors = test_gradient_numerically(
        guidance, x_test, y_test, tolerance=0.05
    )  # 5% tolerance

    # Should agree within 5% (numerical differentiation is approximate)
    assert errors[
        "within_tolerance"
    ], f"Gradient error {errors['mean_rel_error']:.2%} exceeds 5% tolerance"
    logger.info("✓ Numerical gradient verification passed")

    logger.info("✅ All PG guidance tests passed!")
    return True


if __name__ == "__main__":
    # Run comprehensive tests
    logger.info("Running PG guidance tests...")

    try:
        success = test_guidance_properties()
        if success:
            print("\n🎉 All Poisson-Gaussian guidance tests PASSED!")
            print("Core physics implementation is working correctly.")
        else:
            print("\n❌ Some tests failed!")
            exit(1)

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
