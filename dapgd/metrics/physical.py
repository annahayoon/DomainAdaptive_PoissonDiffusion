"""
Physical validation metrics for DAPGD

Metrics that verify physical consistency:
- Chi-squared test for noise model validation
- Photon conservation checks
- Signal-to-noise ratio analysis
- Residual distribution analysis

PURPOSE: Ensure restorations are physically meaningful
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_chi_squared(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    s: float,
    sigma_r: float,
    return_per_pixel: bool = False,
) -> Union[float, torch.Tensor]:
    """
    Compute reduced chi-squared statistic

    PURPOSE: Validate physical consistency of reconstruction
    A well-calibrated method achieves χ²_red ≈ 1.0

    PHYSICAL INTERPRETATION:
    - χ² < 1: Overfitting (too smooth, underestimating noise)
    - χ² > 1: Underfitting (residuals larger than expected)
    - χ² ≈ 1: Correctly calibrated noise model

    Args:
        predicted: Restored image [B,C,H,W] in [0,1] range
        observed: Noisy observation [B,C,H,W] in electrons
        s: Scale factor (electrons per unit intensity)
        sigma_r: Read noise standard deviation (electrons)
        return_per_pixel: If True, return per-pixel chi-squared

    Returns:
        chi2_reduced: Reduced chi-squared value
    """

    # Forward project prediction to electron space
    expected = s * predicted

    # Compute variance (signal-dependent Poisson-Gaussian noise)
    variance = s * predicted + sigma_r**2

    # Compute chi-squared per pixel
    residual = observed - expected
    chi2_per_pixel = (residual**2) / variance

    if return_per_pixel:
        return chi2_per_pixel

    # Sum and normalize by degrees of freedom
    chi2_total = chi2_per_pixel.sum().item()
    dof = observed.numel()

    chi2_reduced = chi2_total / dof

    return chi2_reduced


def check_photon_conservation(
    input_photons: torch.Tensor, output_photons: torch.Tensor, tolerance: float = 0.05
) -> Tuple[bool, float]:
    """
    Check if total photon count is approximately conserved

    PURPOSE: Sanity check for physically meaningful restoration

    Args:
        input_photons: Total photons in input
        output_photons: Total photons in output
        tolerance: Relative tolerance (e.g., 0.05 = 5%)

    Returns:
        is_conserved: True if within tolerance
        relative_error: Relative difference
    """

    input_total = input_photons.sum().item()
    output_total = output_photons.sum().item()

    relative_error = abs(output_total - input_total) / input_total
    is_conserved = relative_error < tolerance

    return is_conserved, relative_error


def analyze_noise_model(
    predicted: torch.Tensor, observed: torch.Tensor, s: float, sigma_r: float
) -> Dict[str, float]:
    """
    Comprehensive noise model analysis

    PURPOSE: Detailed validation of physical consistency

    Returns:
        Dictionary with noise model validation metrics
    """

    # Forward project
    expected = s * predicted
    variance = s * predicted + sigma_r**2

    # Residuals
    residual = observed - expected
    normalized_residual = residual / torch.sqrt(variance)

    # Chi-squared
    chi2 = compute_chi_squared(predicted, observed, s, sigma_r)

    # Statistical tests
    residual_mean = residual.mean().item()
    residual_std = residual.std().item()

    # Normalized residual statistics (should be N(0,1))
    norm_res_mean = normalized_residual.mean().item()
    norm_res_std = normalized_residual.std().item()

    # Outlier analysis (pixels with |residual| > 3σ)
    outlier_fraction = (normalized_residual.abs() > 3).float().mean().item()

    # Signal-to-noise ratio
    snr = (predicted.mean() * s) / torch.sqrt(variance.mean())

    return {
        "chi2_reduced": chi2,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "normalized_residual_mean": norm_res_mean,
        "normalized_residual_std": norm_res_std,
        "outlier_fraction": outlier_fraction,
        "mean_snr": snr.item(),
    }


def validate_physics_consistency(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    s: float,
    sigma_r: float,
    tolerance_chi2: float = 0.5,
    tolerance_conservation: float = 0.1,
) -> Dict[str, bool]:
    """
    Comprehensive physics consistency validation

    Args:
        predicted: Restored image
        observed: Noisy observation
        s: Scale factor
        sigma_r: Read noise
        tolerance_chi2: Chi-squared tolerance around 1.0
        tolerance_conservation: Photon conservation tolerance

    Returns:
        Dictionary with validation results
    """

    analysis = analyze_noise_model(predicted, observed, s, sigma_r)

    # Validate chi-squared (should be close to 1.0)
    chi2_valid = abs(analysis["chi2_reduced"] - 1.0) < tolerance_chi2

    # Validate residual distribution (should be normal)
    residual_valid = (
        abs(analysis["normalized_residual_mean"]) < 0.1
        and abs(analysis["normalized_residual_std"] - 1.0) < 0.2
    )

    # Validate outlier fraction (should be low)
    outlier_valid = analysis["outlier_fraction"] < 0.01  # < 1% outliers

    # Validate photon conservation (approximate)
    expected_photons = (s * predicted).sum().item()
    observed_photons = observed.sum().item()
    conservation_valid, _ = check_photon_conservation(
        torch.tensor(observed_photons),
        torch.tensor(expected_photons),
        tolerance_conservation,
    )

    return {
        "chi_squared_valid": chi2_valid,
        "residual_distribution_valid": residual_valid,
        "outlier_rate_valid": outlier_valid,
        "photon_conservation_valid": conservation_valid,
        "overall_valid": chi2_valid
        and residual_valid
        and outlier_valid
        and conservation_valid,
    }


class PhysicsValidator:
    """
    Comprehensive physics validation for image restoration

    PURPOSE: Automated validation pipeline for physical consistency
    """

    def __init__(
        self,
        s: float,
        sigma_r: float,
        chi2_tolerance: float = 0.5,
        conservation_tolerance: float = 0.1,
    ):
        self.s = s
        self.sigma_r = sigma_r
        self.chi2_tolerance = chi2_tolerance
        self.conservation_tolerance = conservation_tolerance

    def validate(
        self, predicted: torch.Tensor, observed: torch.Tensor
    ) -> Dict[str, any]:
        """
        Run complete physics validation

        Args:
            predicted: Restored image [B,C,H,W]
            observed: Noisy observation [B,C,H,W]

        Returns:
            Complete validation report
        """

        # Run all validations
        noise_analysis = analyze_noise_model(predicted, observed, self.s, self.sigma_r)
        consistency_checks = validate_physics_consistency(
            predicted,
            observed,
            self.s,
            self.sigma_r,
            self.chi2_tolerance,
            self.conservation_tolerance,
        )

        # Generate summary
        summary = {
            "noise_model_analysis": noise_analysis,
            "consistency_checks": consistency_checks,
            "parameters": {
                "s": self.s,
                "sigma_r": self.sigma_r,
                "chi2_tolerance": self.chi2_tolerance,
                "conservation_tolerance": self.conservation_tolerance,
            },
        }

        return summary

    def __call__(
        self, predicted: torch.Tensor, observed: torch.Tensor
    ) -> Dict[str, any]:
        """Convenient validation call"""
        return self.validate(predicted, observed)


# Factory functions for domain-specific validators


def create_physics_validator(domain: str) -> PhysicsValidator:
    """
    Create physics validator with domain-appropriate parameters

    Args:
        domain: Domain type ('photo', 'micro', 'astro')

    Returns:
        Configured PhysicsValidator instance
    """

    # Domain-specific parameters
    domain_params = {
        "photo": {"s": 79351.0, "sigma_r": 3.6, "chi2_tolerance": 0.5},
        "micro": {"s": 65534.0, "sigma_r": 1.5, "chi2_tolerance": 0.3},
        "astro": {
            "s": 121.0,
            "sigma_r": 3.5,
            "chi2_tolerance": 1.0,
        },  # More lenient for challenging domain
    }

    if domain not in domain_params:
        raise ValueError(f"Unknown domain: {domain}")

    params = domain_params[domain]

    logger.info(
        f"Creating {domain} physics validator: s={params['s']}, σ_r={params['sigma_r']}"
    )

    return PhysicsValidator(
        s=params["s"],
        sigma_r=params["sigma_r"],
        chi2_tolerance=params["chi2_tolerance"],
    )


# Test functions


def test_physical_validation():
    """Test physical validation metrics"""

    logger.info("Testing physical validation...")

    # Create test data with known noise model
    clean = torch.rand(1, 1, 32, 32) * 0.5 + 0.25  # Moderate intensity

    # Simulate Poisson-Gaussian noise
    s_true = 1000.0
    sigma_r_true = 5.0

    from ..guidance.pg_guidance import simulate_poisson_gaussian_noise

    noisy = simulate_poisson_gaussian_noise(clean, s_true, sigma_r_true, seed=42)

    # Create validator with correct parameters
    validator = PhysicsValidator(s=s_true, sigma_r=sigma_r_true)

    # Test validation
    validation = validator.validate(clean, noisy)

    logger.info("✓ Physical validation completed")
    logger.info(
        f"  Chi-squared: {validation['noise_model_analysis']['chi2_reduced']:.4f}"
    )
    logger.info(
        f"  Normalized residual mean: {validation['noise_model_analysis']['normalized_residual_mean']:.4f}"
    )
    logger.info(
        f"  Normalized residual std: {validation['noise_model_analysis']['normalized_residual_std']:.4f}"
    )

    # Chi-squared should be close to 1.0 for correct model
    chi2 = validation["noise_model_analysis"]["chi2_reduced"]
    assert 0.5 < chi2 < 2.0, f"Chi-squared {chi2:.4f} not in expected range [0.5, 2.0]"

    # Normalized residuals should be approximately N(0,1)
    norm_mean = validation["noise_model_analysis"]["normalized_residual_mean"]
    norm_std = validation["noise_model_analysis"]["normalized_residual_std"]
    assert (
        abs(norm_mean) < 0.2
    ), f"Normalized residual mean {norm_mean:.4f} too far from 0"
    assert (
        0.8 < norm_std < 1.2
    ), f"Normalized residual std {norm_std:.4f} not close to 1"

    logger.info("✅ Physical validation tests passed!")
    return True


def test_domain_validators():
    """Test domain-specific validator creation"""

    logger.info("Testing domain-specific validators...")

    # Test all domains
    domains = ["photo", "micro", "astro"]

    for domain in domains:
        validator = create_physics_validator(domain)

        # Check parameters are set correctly
        if domain == "photo":
            assert validator.s == 79351.0
            assert validator.sigma_r == 3.6
        elif domain == "micro":
            assert validator.s == 65534.0
            assert validator.sigma_r == 1.5
        elif domain == "astro":
            assert validator.s == 121.0
            assert validator.sigma_r == 3.5

        logger.info(f"✓ {domain} validator created with correct parameters")

    logger.info("✅ Domain validator tests passed!")
    return True


if __name__ == "__main__":
    # Run comprehensive tests
    logger.info("Running physical validation tests...")

    try:
        success1 = test_physical_validation()
        success2 = test_domain_validators()

        if success1 and success2:
            print("\n🎉 All physical validation tests PASSED!")
            print("Physics consistency checking is working correctly.")
        else:
            print("\n❌ Some physical validation tests FAILED!")
            exit(1)

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
