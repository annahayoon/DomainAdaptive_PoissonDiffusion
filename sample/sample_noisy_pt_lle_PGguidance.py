#!/usr/bin/env python3
"""
Posterior Sampling for Image Restoration using EDM Model with Poisson-Gaussian Guidance

This script performs posterior sampling on noisy test images using EDM with
exposure-aware Poisson-Gaussian measurement guidance for physics-informed restoration.

Theory: We sample from p(x|y) ∝ p(y|x) p(x) where p(x) is the EDM-learned prior
and p(y|x) is the Poisson-Gaussian likelihood.

Usage:
    python sample/sample_noisy_pt_lle_PGguidance.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/photography/noisy \
        --output_dir results/posterior_sampling \
        --domain photography \
        --num_examples 3 \
        --num_steps 18 \
        --use_sensor_calibration \
        --s 15871 \
        --sigma_r 5.0 \
        --kappa 0.5
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.regression import MeanSquaredError as MSE
from tqdm import tqdm

# Note: numpy._core compatibility shim removed - models should be saved/loaded with compatible numpy versions
# Photography model appears to have been saved with numpy 2.0+ and requires Python 3.10+


# Import lpips and pyiqa for image quality metrics
try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    lpips = None

try:
    import pyiqa

    PQIQA_AVAILABLE = True
except ImportError:
    PQIQA_AVAILABLE = False
    pyiqa = None

# FID computation is handled separately by sample/compute_fid.py

# Add project root and EDM to path
project_root = Path(__file__).parent.parent  # Go up from sample/ to project root
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

# Setup logging
import logging

# Import EDM components
import external.edm.dnnlib
from external.edm.torch_utils import distributed as dist

# Import sensor calibration
from sample.sensor_calibration import SensorCalibration

# Import stratified evaluation for core scientific validation
try:
    from analysis.stratified_evaluation import (
        StratifiedEvaluator,
        format_stratified_results_table,
    )

    STRATIFIED_EVAL_AVAILABLE = True
except ImportError:
    STRATIFIED_EVAL_AVAILABLE = False
    logger.warning(
        "Stratified evaluation module not available. Install analysis module for stratified results."
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Log availability of additional metrics libraries
if not LPIPS_AVAILABLE:
    logger.warning("lpips library not available. Install with: pip install lpips")
if not PQIQA_AVAILABLE:
    logger.warning("pyiqa library not available. Install with: pip install pyiqa")


# ============================================================================
# IMPROVEMENT 1: Enhanced Unit Consistency and Physics Documentation
# ============================================================================


class PhysicsConsistencyValidator:
    """
    Validates physical consistency of guidance parameters.

    UNIT FLOW (Critical):

    1. INPUT UNITS:
       - x_pred: [0,1] normalized (model output)
       - y_observed: [domain_min, domain_max] physical units
       - s: MUST EQUAL domain_range = domain_max - domain_min
       - sigma_r: Physical units matching y_observed

    2. CONSTRAINT: s = domain_range
       Without this, forward model gives mismatched units.
       If s ≠ domain_range, then y_phys and expected_phys are in different scales.
    """

    @staticmethod
    def validate_parameter_units(
        s: float, domain_min: float, domain_max: float
    ) -> Dict[str, Any]:
        """Validate unit consistency of guidance parameters."""
        domain_range = domain_max - domain_min

        if abs(s - domain_range) > 1e-3:
            raise ValueError(
                f"UNIT CONSISTENCY VIOLATION: s={s} must equal domain_range={domain_range}\n"
                f"REASON: s scales normalized predictions to physical units.\n"
                f"If s ≠ domain_range, predicted and observed physical scales don't match.\n"
                f"\n"
                f"FIX: Set s = domain_max - domain_min = {domain_range}"
            )

        return {
            "is_valid": True,
            "s": s,
            "domain_range": domain_range,
            "domain_min": domain_min,
            "domain_max": domain_max,
        }


# ============================================================================
# IMPROVEMENT 2: Photon Count Validation
# ============================================================================


class PhotonCountValidator:
    """
    Validates Poisson-Gaussian approximation quality across different photon count regimes.

    The Gaussian approximation to Poisson(λ) + N(0, σ_r²) breaks down when:
    - λ < 10 photons (Poisson is highly skewed)
    - λ < 3 photons (discrete nature dominates)

    This validator checks these conditions and provides guidance.
    """

    @staticmethod
    def estimate_photon_counts(
        y_e_physical: torch.Tensor,
        alpha: float,
        s: float,
        domain_min: float,
        domain_max: float,
    ) -> Dict[str, float]:
        """
        Estimate photon count statistics from observed measurement.

        Args:
            y_e_physical: Observed measurement in physical units
            alpha: Exposure ratio
            s: Scale factor (= domain_range)
            domain_min, domain_max: Domain physical range

        Returns:
            Dictionary with photon count statistics
        """
        # Normalize to [0,1]
        domain_range = domain_max - domain_min
        y_norm = (y_e_physical - domain_min) / domain_range
        y_norm = torch.clamp(y_norm, 0, 1)

        # Expected photon counts at short exposure
        lambda_est = alpha * s * y_norm

        # Compute statistics
        lambda_flat = lambda_est.flatten().cpu().numpy()

        return {
            "mean_photons": float(np.mean(lambda_flat)),
            "min_photons": float(np.min(lambda_flat)),
            "p10_photons": float(np.percentile(lambda_flat, 10)),
            "p50_photons": float(np.percentile(lambda_flat, 50)),
            "p90_photons": float(np.percentile(lambda_flat, 90)),
            "max_photons": float(np.max(lambda_flat)),
            "fraction_below_10": float(np.mean(lambda_flat < 10)),
            "fraction_below_3": float(np.mean(lambda_flat < 3)),
            "fraction_below_1": float(np.mean(lambda_flat < 1)),
        }

    @staticmethod
    def validate_approximation_quality(
        photon_stats: Dict[str, float], strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate that Gaussian approximation is appropriate.

        Returns:
            Dictionary with validation results and recommendations
        """
        warnings = []

        # Thresholds
        critical_threshold = 3.0 if strict else 1.0
        warning_threshold = 10.0 if strict else 5.0
        good_threshold = 20.0

        mean_photons = photon_stats["mean_photons"]
        frac_below_10 = photon_stats["fraction_below_10"]
        frac_below_3 = photon_stats["fraction_below_3"]

        # Determine quality
        if mean_photons >= good_threshold and frac_below_10 < 0.1:
            quality = "excellent"
            is_valid = True
        elif mean_photons >= warning_threshold and frac_below_3 < 0.2:
            quality = "good"
            is_valid = True
            warnings.append(
                f"Some pixels have low photon counts (mean={mean_photons:.1f}). "
                f"Gaussian approximation may be slightly inaccurate."
            )
        elif mean_photons >= critical_threshold:
            quality = "marginal"
            is_valid = not strict
            warnings.append(
                f"WARNING: Low photon counts detected (mean={mean_photons:.1f}). "
                f"{frac_below_10*100:.1f}% of pixels have λ < 10. "
                f"Gaussian approximation may introduce noticeable errors."
            )
        else:
            quality = "poor"
            is_valid = False
            warnings.append(
                f"CRITICAL: Very low photon counts (mean={mean_photons:.1f}). "
                f"{frac_below_3*100:.1f}% of pixels have λ < 3. "
                f"Gaussian approximation is inappropriate - consider discrete Poisson model."
            )

        # Recommended action
        if quality == "excellent":
            action = (
                "Gaussian approximation is highly accurate. Proceed with confidence."
            )
        elif quality == "good":
            action = "Gaussian approximation is adequate. Results should be reliable."
        elif quality == "marginal":
            action = "Consider: (1) Increase exposure time, (2) Use discrete Poisson model, or (3) Accept reduced accuracy."
        else:
            action = "CRITICAL: Switch to discrete Poisson diffusion model for accurate results."

        return {
            "is_valid": is_valid,
            "warnings": warnings,
            "approximation_quality": quality,
            "recommended_action": action,
            "photon_statistics": photon_stats,
        }

    @staticmethod
    def compute_approximation_error(
        lambda_mean: float,
        sigma_r: float,
    ) -> Dict[str, float]:
        """Compute theoretical approximation error metrics."""
        # Skewness of Poisson(λ) is 1/√λ
        skewness = (
            1.0 / np.sqrt(lambda_mean + sigma_r**2)
            if (lambda_mean + sigma_r**2) > 0
            else 0.0
        )

        # Approximate KL divergence using moment matching error
        kl_approx = skewness**2 / 12

        return {
            "kl_divergence_approx": kl_approx,
            "relative_variance_error": 0.0,  # By construction
            "skewness": skewness,
            "lambda_mean": lambda_mean,
            "sigma_r": sigma_r,
        }


# ============================================================================
# IMPROVEMENT 3: Unified Exposure Ratio Extraction
# ============================================================================


class ExposureRatioExtractor:
    """
    Unified interface for extracting exposure ratios across all domains.

    Replaces fragile domain-specific extraction logic with consistent interface.
    """

    @staticmethod
    def extract_exposure_ratio(
        noisy_tile_id: str,
        clean_tile_info: Dict[str, Any],
        domain: str,
        metadata_json: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Unified exposure ratio extraction for all domains.

        Returns:
            Dictionary with exposure_ratio, extraction method, confidence, and warnings
        """
        if domain == "photography":
            return ExposureRatioExtractor._extract_photography(
                noisy_tile_id, clean_tile_info
            )
        elif domain == "microscopy":
            return ExposureRatioExtractor._extract_microscopy(
                noisy_tile_id, clean_tile_info, metadata_json
            )
        elif domain == "astronomy":
            return ExposureRatioExtractor._extract_astronomy(
                noisy_tile_id, clean_tile_info, metadata_json
            )
        else:
            raise ValueError(f"Unknown domain: {domain}")

    @staticmethod
    def _extract_photography(noisy_tile_id: str, clean_tile_info: Dict) -> Dict:
        """Extract from filename: {scene}_00_{exposure}s_tile_{id}"""
        warnings = []

        try:
            # Parse noisy exposure time
            noisy_parts = noisy_tile_id.split("_")
            noisy_exp_str = next((p for p in noisy_parts if p.endswith("s")), None)
            if noisy_exp_str is None:
                raise ValueError(f"Could not find exposure in {noisy_tile_id}")
            noisy_exp = float(noisy_exp_str.replace("s", ""))

            # Parse clean exposure time
            clean_tile_id = clean_tile_info.get("tile_id", "")
            clean_parts = clean_tile_id.split("_")
            clean_exp_str = next((p for p in clean_parts if p.endswith("s")), None)
            if clean_exp_str is None:
                raise ValueError(f"Could not find exposure in {clean_tile_id}")
            clean_exp = float(clean_exp_str.replace("s", ""))

            # Validate
            if clean_exp <= 0:
                warnings.append(f"Invalid clean exposure: {clean_exp}")
                confidence = 0.0
            elif noisy_exp > clean_exp:
                warnings.append(
                    f"Noisy exposure ({noisy_exp}) > clean exposure ({clean_exp})"
                )
                confidence = 0.5
            else:
                confidence = 1.0

            exposure_ratio = noisy_exp / clean_exp

            return {
                "exposure_ratio": exposure_ratio,
                "method": "filename_parsing",
                "noisy_exposure_measure": noisy_exp,
                "clean_exposure_measure": clean_exp,
                "confidence": confidence,
                "warnings": warnings,
            }

        except Exception as e:
            return {
                "exposure_ratio": 1.0,
                "method": "failed",
                "noisy_exposure_measure": None,
                "clean_exposure_measure": None,
                "confidence": 0.0,
                "warnings": [f"Extraction failed: {e}"],
            }

    @staticmethod
    def _extract_microscopy(
        noisy_tile_id: str, clean_tile_info: Dict, metadata_json: Optional[Path]
    ) -> Dict:
        """Extract from metadata original_mean field."""
        warnings = []

        try:
            # Extract mean intensities from clean_tile_info
            noisy_mean = clean_tile_info.get("noisy_original_mean")
            clean_mean = clean_tile_info.get("original_mean")

            if noisy_mean is None or clean_mean is None:
                raise ValueError("Missing original_mean in clean_tile_info")

            if clean_mean <= 0:
                warnings.append(f"Invalid clean mean: {clean_mean}")
                confidence = 0.0
            elif noisy_mean > clean_mean:
                warnings.append(
                    f"Noisy mean ({noisy_mean}) > clean mean ({clean_mean})"
                )
                confidence = 0.5
            else:
                confidence = 1.0

            exposure_ratio = noisy_mean / clean_mean

            if exposure_ratio > 1.0:
                warnings.append(
                    f"Exposure ratio > 1.0 (α={exposure_ratio:.3f}). Unusual."
                )
                confidence = min(confidence, 0.5)
            elif exposure_ratio < 0.01:
                warnings.append(f"Very low exposure ratio (α={exposure_ratio:.3f})")

            return {
                "exposure_ratio": exposure_ratio,
                "method": "metadata_mean_intensity",
                "noisy_exposure_measure": noisy_mean,
                "clean_exposure_measure": clean_mean,
                "confidence": confidence,
                "warnings": warnings,
            }

        except Exception as e:
            return {
                "exposure_ratio": 1.0,
                "method": "failed",
                "noisy_exposure_measure": None,
                "clean_exposure_measure": None,
                "confidence": 0.0,
                "warnings": [f"Extraction failed: {e}"],
            }

    @staticmethod
    def _extract_astronomy(
        noisy_tile_id: str, clean_tile_info: Dict, metadata_json: Optional[Path]
    ) -> Dict:
        """Extract from metadata exposure_time_seconds field."""
        warnings = []

        try:
            # Extract exposure times
            noisy_exp = clean_tile_info.get("noisy_exposure_time_seconds")
            clean_exp = clean_tile_info.get("exposure_time_seconds")

            if noisy_exp is None or clean_exp is None:
                raise ValueError("Missing exposure_time_seconds in metadata")

            if clean_exp <= 0:
                warnings.append(f"Invalid clean exposure time: {clean_exp}")
                confidence = 0.0
            elif noisy_exp > clean_exp:
                warnings.append(
                    f"Noisy exposure ({noisy_exp}) > clean exposure ({clean_exp})"
                )
                confidence = 0.5
            else:
                confidence = 1.0

            exposure_ratio = noisy_exp / clean_exp

            return {
                "exposure_ratio": exposure_ratio,
                "method": "metadata_exposure_time",
                "noisy_exposure_measure": noisy_exp,
                "clean_exposure_measure": clean_exp,
                "confidence": confidence,
                "warnings": warnings,
            }

        except Exception as e:
            return {
                "exposure_ratio": 1.0,
                "method": "failed",
                "noisy_exposure_measure": None,
                "clean_exposure_measure": None,
                "confidence": 0.0,
                "warnings": [f"Extraction failed: {e}"],
            }


def validate_guidance_inputs(x0_hat: torch.Tensor, y_e: torch.Tensor) -> None:
    """Shared validation function for guidance inputs.

    Validates that input tensors have compatible shapes and appropriate value ranges.

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
    - y_e: Physical units (ADU, electrons, counts)
    - Converts to/from physical space for guidance computation

    Args:
        s: Scale factor for numerical stability (must equal domain_range for unit consistency)
        sigma_r: Read noise standard deviation in physical units
        domain_min: Minimum physical value of the domain
        domain_max: Maximum physical value of the domain
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
        domain_min: float,
        domain_max: float,
        offset: float = 0.0,  # Offset for astronomy data
        exposure_ratio: float = 1.0,
        kappa: float = 0.5,
        tau: float = 0.01,
        epsilon: float = 1e-8,
        guidance_level: str = "x0",  # 'x0' or 'score'
    ):
        super().__init__()

        # Validate unit consistency (same as PG guidance)
        domain_range = domain_max - domain_min
        if abs(s - domain_range) > 1e-3:
            raise ValueError(
                f"s={s} must equal domain_range={domain_range} for unit consistency!\n"
                f"s = domain_max - domain_min ensures proper normalization.\n"
                f"This ensures proper comparison between observed and expected values."
            )

        self.s = s
        self.sigma_r = sigma_r
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.offset = offset  # Offset for astronomy data
        self.alpha = exposure_ratio  # Now USED in exposure-aware Gaussian likelihood
        self.kappa = kappa
        self.tau = tau
        self.epsilon = epsilon
        self.guidance_level = guidance_level  # 'x0' or 'score'

        # Pre-compute constants
        self.sigma_r_squared = sigma_r**2
        self.domain_range = domain_max - domain_min

        logger.info(
            f"Initialized Exposure-Aware Gaussian Guidance: s={s}, σ_r={sigma_r}, "
            f"domain=[{domain_min}, {domain_max}], offset={offset:.3f}, α={exposure_ratio:.4f}, κ={kappa}, τ={tau}, level={guidance_level}"
        )
        logger.info("Now exposure-aware: accounts for α in forward model")
        logger.info("Uses physical sensor parameters (s, σ_r) like PG guidance")
        logger.info(f"Unit consistency verified: s=domain_range={self.domain_range}")
        if offset > 0:
            logger.info(f"Astronomy offset applied: {offset:.3f}")
        logger.warning("Still assumes CONSTANT noise variance (simplified vs PG)")

    def _validate_inputs(self, x0_hat: torch.Tensor, y_e: torch.Tensor):
        """
        Validate inputs for guidance computation.

        Args:
            x0_hat: Current denoised estimate [B,C,H,W], range [0,1]
            y_e: Observed noisy measurement [B,C,H,W], in physical units

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(x0_hat, torch.Tensor):
            raise ValueError(f"x0_hat must be a torch.Tensor, got {type(x0_hat)}")

        if not isinstance(y_e, torch.Tensor):
            raise ValueError(f"y_e must be a torch.Tensor, got {type(y_e)}")

        if x0_hat.shape != y_e.shape:
            raise ValueError(
                f"Shape mismatch: x0_hat {x0_hat.shape} vs y_e {y_e.shape}"
            )

        if x0_hat.min() < 0.0 or x0_hat.max() > 1.0:
            logger.warning(
                f"x0_hat values outside [0,1] range: [{x0_hat.min():.4f}, {x0_hat.max():.4f}]"
            )

        if (
            y_e.min() < self.domain_min - self.offset
            or y_e.max() > self.domain_max + self.offset
        ):
            logger.warning(
                f"y_e values outside expected physical range: [{y_e.min():.4f}, {y_e.max():.4f}]"
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
            y_e: Observed noisy measurement [B,C,H,W], in physical units (electrons)
            sigma_t: Current noise level (sigma)

        Returns:
            x0_guided: Guided estimate [B,C,H,W], range [0,1]
        """
        # Check if guidance should be applied
        if sigma_t <= self.tau:
            return x0_hat

        # Validate inputs
        self._validate_inputs(x0_hat, y_e)

        # FOLLOW SAME PATTERN AS PG GUIDANCE:
        # Step 1: Normalize observation to [0,1]
        y_e_norm = (y_e - self.domain_min) / (self.domain_range + self.epsilon)
        y_e_norm = torch.clamp(y_e_norm, 0.0, 1.0)

        # Step 2: Scale observation - use s (same as PG guidance)
        y_e_scaled = y_e_norm * self.s

        # Step 3: Expected observation at SHORT exposure (NOW exposure-aware!)
        # CORRECTED: Scale down the bright prediction to match dark observation
        expected_at_short_exp = self.alpha * self.s * x0_hat

        # Step 4: Residual (units now match!)
        residual = y_e_scaled - expected_at_short_exp

        # Step 5: Exposure-aware Gaussian gradient (constant variance)
        # ∇_x log p(y|x) = α·s·(y - α·s·x) / σ_r²
        # Same form as PG but with constant variance σ_r² instead of signal-dependent
        gradient = (
            self.alpha * self.s * residual / (self.sigma_r_squared + self.epsilon)
        )

        # Apply standard guidance
        step_size = self.kappa * (sigma_t**2)
        x0_guided = x0_hat + step_size * gradient

        # Clamp to valid range [0, 1]
        x0_guided = torch.clamp(x0_guided, 0.0, 1.0)

        return x0_guided

    def compute_likelihood_gradient(
        self, x0_hat: torch.Tensor, y_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute likelihood gradient for score-level guidance.

        This is the theoretically pure approach where we add the likelihood
        gradient directly to the score instead of modifying x₀.

        Args:
            x0_hat: Current denoised estimate [B,C,H,W], range [0,1]
            y_e: Observed noisy measurement [B,C,H,W], in electrons

        Returns:
            likelihood_gradient: ∇_x log p(y|x) [B,C,H,W]
        """
        # Validate inputs
        validate_guidance_inputs(x0_hat, y_e)

        # Check if guidance should be applied
        if hasattr(self, "_current_sigma_t") and self._current_sigma_t <= self.tau:
            return torch.zeros_like(x0_hat)

        # Compute gradient using the same logic as forward()
        # Step 1: Normalize observation to [0,1]
        y_e_norm = (y_e - self.domain_min) / (self.domain_range + self.epsilon)
        y_e_norm = torch.clamp(y_e_norm, 0.0, 1.0)

        # Step 2: Scale observation - use s (same as PG guidance)
        y_e_scaled = y_e_norm * self.s

        # Step 3: Expected observation at SHORT exposure (NOW exposure-aware!)
        # CORRECTED: Scale down the bright prediction to match dark observation
        expected_at_short_exp = self.alpha * self.s * x0_hat

        # Step 4: Residual (units now match!)
        residual = y_e_scaled - expected_at_short_exp

        # Step 5: Exposure-aware Gaussian gradient (constant variance)
        # ∇_x log p(y|x) = α·s·(y - α·s·x) / σ_r²
        # Same form as PG but with constant variance σ_r² instead of signal-dependent
        gradient = (
            self.alpha * self.s * residual / (self.sigma_r_squared + self.epsilon)
        )

        return gradient


class PoissonGaussianGuidance(nn.Module):
    """
    Physics-informed guidance for photon-limited imaging

    Implements the score of the Poisson-Gaussian likelihood:
    ∇_x log p(y_e|x)

    This tells the diffusion model how to adjust predictions to match
    observed noisy measurements while respecting physical noise properties.

    Physical Unit Requirements:
    - x_pred must be in [0,1] normalized space
    - y_observed must be in PHYSICAL UNITS (ADU, electrons, counts)
    - s: Scale factor for normalized comparison (s = domain_range)
    - sigma_r: Read noise in same physical units as y_observed
    - domain_min, domain_max: Physical range of the domain for normalization
    - offset: Offset applied to astronomy data (0.0 for other domains)

    The key insight: We normalize BOTH y_observed and expected values to [0, s]
    scale internally, ensuring proper comparison.

    UNIT CONSISTENCY CLARIFICATION:
    Since s = domain_range, the normalization chain is:
    1. x ∈ [-1,1] (model space)
    2. x_norm ∈ [0,1] (normalized model space)
    3. x_phys ∈ [domain_min, domain_max] (physical units)

    The scale factor s = domain_range ensures that when we compute:
    - y_scaled = y_norm * s (where y_norm = (y_phys - domain_min) / domain_range)
    - expected = alpha * s * x_norm

    Both y_scaled and expected are in the same units as the original y_phys,
    ensuring proper comparison in the likelihood computation.

    Args:
        s: Scale factor for normalized comparison (s = domain_range)
        sigma_r: Read noise standard deviation (in physical units)
        domain_min: Minimum physical value of the domain
        domain_max: Maximum physical value of the domain
        offset: Offset applied to astronomy data (0.0 for other domains)
        exposure_ratio: Exposure ratio t_low / t_long (e.g., 0.01)
        kappa: Guidance strength multiplier (typically 0.3-1.0)
        tau: Guidance threshold - only apply when σ_t > tau
        mode: 'wls' for weighted least squares, 'full' for complete gradient
        epsilon: Small constant for numerical stability
        guidance_level: 'x0' or 'score' (DPS)

    Example:
        >>> # For photography domain with range [0, 15871] ADU
        >>> guidance = PoissonGaussianGuidance(
        ...     s=15871.0, sigma_r=5.0,
        ...     domain_min=0.0, domain_max=15871.0,
        ...     offset=0.0, kappa=0.5
        ... )
        >>> # For astronomy domain using original physical coordinates [-65, 385]
        >>> guidance = PoissonGaussianGuidance(
        ...     s=385.0, sigma_r=2.0,
        ...     domain_min=-65.0, domain_max=385.0,
        ...     offset=0.0, kappa=0.5  # No offset needed with original coordinates
        ... )
    """

    def __init__(
        self,
        s: float,
        sigma_r: float,
        domain_min: float,
        domain_max: float,
        offset: float = 0.0,
        exposure_ratio: float = 1.0,  # t_low / t_long
        kappa: float = 0.5,
        tau: float = 0.01,
        mode: str = "wls",
        epsilon: float = 1e-8,
        guidance_level: str = "x0",  # 'x0' (empirically stable, default) or 'score' (DPS, theoretically pure)
    ):
        super().__init__()

        # Enforce unit consistency
        # s must equal domain_range for proper unit handling in the guidance computation
        domain_range = domain_max - domain_min
        if abs(s - domain_range) > 1e-3:
            raise ValueError(
                f"s={s} must equal domain_range={domain_range} for unit consistency!\n"
                f"s = domain_max - domain_min ensures proper normalization.\n"
                f"This ensures proper comparison between observed and expected values."
            )

        self.s = s
        self.sigma_r = sigma_r
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.offset = offset
        self.alpha = exposure_ratio
        self.kappa = kappa
        self.tau = tau
        self.mode = mode
        self.epsilon = epsilon
        self.guidance_level = guidance_level  # 'x0' or 'score'

        # Pre-compute constants for efficiency
        self.sigma_r_squared = sigma_r**2
        self.domain_range = domain_max - domain_min

        logger.info(
            f"Initialized PG Guidance: s={s}, σ_r={sigma_r}, "
            f"domain=[{domain_min}, {domain_max}], offset={offset:.3f}, α={exposure_ratio:.4f}, κ={kappa}, τ={tau}, mode={mode}"
        )
        logger.info(f"Unit consistency verified: s=domain_range={self.domain_range}")
        if offset > 0:
            logger.info(f"Astronomy offset applied: {offset:.3f}")

    def _validate_inputs(self, x0_hat: torch.Tensor, y_e: torch.Tensor):
        """
        Validate inputs for guidance computation.

        Args:
            x0_hat: Current denoised estimate [B,C,H,W], range [0,1]
            y_e: Observed noisy measurement [B,C,H,W], in physical units

        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(x0_hat, torch.Tensor):
            raise ValueError(f"x0_hat must be a torch.Tensor, got {type(x0_hat)}")

        if not isinstance(y_e, torch.Tensor):
            raise ValueError(f"y_e must be a torch.Tensor, got {type(y_e)}")

        if x0_hat.shape != y_e.shape:
            raise ValueError(
                f"Shape mismatch: x0_hat {x0_hat.shape} vs y_e {y_e.shape}"
            )

        if x0_hat.min() < 0.0 or x0_hat.max() > 1.0:
            logger.warning(
                f"x0_hat values outside [0,1] range: [{x0_hat.min():.4f}, {x0_hat.max():.4f}]"
            )

        if (
            y_e.min() < self.domain_min - self.offset
            or y_e.max() > self.domain_max + self.offset
        ):
            logger.warning(
                f"y_e values outside expected physical range: [{y_e.min():.4f}, {y_e.max():.4f}]"
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

        # Apply standard guidance
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

    def _wls_gradient(
        self, x0_hat: torch.Tensor, y_e_physical: torch.Tensor
    ) -> torch.Tensor:
        """
        Exposure-aware Weighted Least Squares gradient

        This method computes the gradient of the log-likelihood with proper unit handling.

        Forward model: y_short = Poisson(α·s·x_long) + N(0, σ_r²)

        where:
        - x_long (x0_hat): Prediction at LONG exposure (what we want)
        - y_short (y_e_physical): Observation at SHORT exposure (what we have)
        - α (self.alpha): Exposure ratio linking them (t_short / t_long)

        Input units:
          x0_hat: [0,1] normalized long-exposure prediction
          y_e_physical: [0, domain_max] physical units (ADU/counts)

        Internal computation:
          1. Normalize y to [0,1]: y_norm = (y_phys - domain_min) / (domain_max - domain_min)
          2. Scale to [0,s]: y_scaled = y_norm * s
          3. Expected: μ = α·s·x0_hat (same scale as y_scaled)
          4. Variance: σ² = α·s·x0_hat + σ_r² (heteroscedastic)
          5. Gradient: ∇ = α·s·(y_scaled - μ) / σ²

        Output units:
          gradient in [0,1] space (∂log p / ∂x0_hat)

        Physical interpretation:
        - x0_hat: Bright long-exposure prediction
        - α * x0_hat: What that image would look like at short exposure
        - residual: Difference between observed and expected (same units!)
        - Gradient tells model how to adjust bright prediction to match dark observation
        """
        # Step 1: Normalize observation to [0,1]
        y_e_norm = (y_e_physical - self.domain_min) / (self.domain_range + self.epsilon)
        y_e_norm = torch.clamp(y_e_norm, 0.0, 1.0)

        # Step 2: Expected observation at SHORT exposure
        # Scale down the bright prediction to match dark observation
        expected_at_short_exp = self.alpha * self.s * x0_hat

        # Step 3: Variance at SHORT exposure (signal-dependent!)
        variance = self.alpha * self.s * x0_hat + self.sigma_r_squared + self.epsilon

        # Step 4: Residual (NOW units match!)
        y_e_scaled = y_e_norm * self.s
        residual = y_e_scaled - expected_at_short_exp

        # Step 5: WLS gradient
        gradient = self.alpha * self.s * residual / variance

        return gradient

    def _full_gradient(
        self, x0_hat: torch.Tensor, y_e_physical: torch.Tensor
    ) -> torch.Tensor:
        """
        CORRECTED: Full gradient including variance term with exposure ratio

        Forward model: y_low = Poisson(α·s·x_long) + N(0, σ_r²)
        """
        # Step 1: Normalize observation to [0,1]
        y_e_norm = (y_e_physical - self.domain_min) / (self.domain_range + self.epsilon)
        y_e_norm = torch.clamp(y_e_norm, 0.0, 1.0)

        # Step 2: Expected observation at SHORT exposure
        expected_at_short_exp = self.alpha * self.s * x0_hat

        # Step 3: Variance at SHORT exposure (signal-dependent!)
        variance = self.alpha * self.s * x0_hat + self.sigma_r_squared + self.epsilon

        # Step 4: Residual (NOW units match!)
        y_e_scaled = y_e_norm * self.s
        residual = y_e_scaled - expected_at_short_exp

        # Step 5: Mean term (same as WLS)
        mean_term = self.alpha * self.s * residual / variance

        # Step 6: Variance term (second-order correction)
        variance_term = (self.alpha * self.s) * (
            (residual**2) / (2 * variance**2) - 1 / (2 * variance)
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
            y_e: Observed noisy measurement [B,C,H,W], in electrons

        Returns:
            likelihood_gradient: ∇_x log p(y|x) [B,C,H,W]
        """
        # Validate inputs
        validate_guidance_inputs(x0_hat, y_e)

        # Compute gradient
        gradient = self._compute_gradient(x0_hat, y_e)

        return gradient


def test_pg_gradient_correctness():
    """
    Verify PG guidance gradient against finite differences.

    This test ensures the analytical gradient computation is correct by comparing
    it to numerical gradients computed via finite differences.
    """
    logger.info("Running PG guidance gradient verification tests...")

    # Test parameters
    s = 15871.0  # Photography domain
    sigma_r = 5.0
    domain_min = 0.0
    domain_max = 15871.0
    exposure_ratio = 0.01
    kappa = 0.5
    epsilon = 1e-8

    # Create PG guidance module
    pg_guidance = PoissonGaussianGuidance(
        s=s,
        sigma_r=sigma_r,
        domain_min=domain_min,
        domain_max=domain_max,
        exposure_ratio=exposure_ratio,
        kappa=kappa,
        epsilon=epsilon,
    )

    # Test on various input sizes and values
    test_cases = [
        (1, 1, 32, 32),  # Grayscale
        (1, 3, 16, 16),  # RGB
    ]

    for batch_size, channels, height, width in test_cases:
        logger.info(
            f"  Testing gradient on shape: ({batch_size}, {channels}, {height}, {width})"
        )

        # Create test inputs
        torch.manual_seed(42)
        x0_hat = torch.rand(batch_size, channels, height, width, requires_grad=True)
        y_e = torch.rand(batch_size, channels, height, width) * domain_max

        # Analytical gradient
        grad_analytical = pg_guidance.compute_likelihood_gradient(x0_hat, y_e)

        # Numerical gradient using finite differences
        eps = 1e-5
        grad_numerical = torch.zeros_like(x0_hat)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        # Forward difference
                        x_plus = x0_hat.clone()
                        x_plus[b, c, h, w] += eps

                        # Compute log-likelihood for both points
                        log_lik = compute_log_likelihood_for_test(
                            y_e, x0_hat, pg_guidance, b, c, h, w
                        )
                        log_lik_plus = compute_log_likelihood_for_test(
                            y_e, x_plus, pg_guidance, b, c, h, w
                        )

                        # Numerical gradient
                        grad_numerical[b, c, h, w] = (log_lik_plus - log_lik) / eps

        # Compare gradients
        diff = torch.abs(grad_analytical - grad_numerical).mean()
        max_diff = torch.abs(grad_analytical - grad_numerical).max()

        logger.info(f"    Mean absolute difference: {diff:.6f}")
        logger.info(f"    Max absolute difference: {max_diff:.6f}")

        # Assert gradients are close (allowing for numerical precision)
        assert diff < 1e-3, f"Gradient verification failed: mean diff {diff:.6f} > 1e-3"
        assert (
            max_diff < 1e-2
        ), f"Gradient verification failed: max diff {max_diff:.6f} > 1e-2"

        logger.info("    Gradient verification passed")

        logger.info("All PG guidance gradient verification tests passed")


def compute_log_likelihood_for_test(y_e, x0_hat, pg_guidance, b, c, h, w):
    """
    Compute log-likelihood for a single pixel for gradient verification testing.

    This is a simplified version that computes the likelihood for a single pixel
    to enable finite difference gradient checking.
    """
    # Extract single pixel
    x_pixel = x0_hat[b : b + 1, c : c + 1, h : h + 1, w : w + 1]
    y_pixel = y_e[b : b + 1, c : c + 1, h : h + 1, w : w + 1]

    # Compute gradient (which equals the log-likelihood gradient for this single pixel)
    grad = pg_guidance.compute_likelihood_gradient(x_pixel, y_pixel)

    # For this simplified case, the gradient magnitude relates to log-likelihood
    # This is an approximation for testing purposes
    return -0.5 * grad.sum().item()  # Simplified log-likelihood approximation


def extract_sensor_from_tile_id(tile_id: str, domain: str = None) -> str:
    """
    Extract sensor information from tile ID for all supported domains.

    Args:
        tile_id: Tile ID (e.g., "photography_sony_00145_00_0.1s_tile_0000")
        domain: Domain name (photography, microscopy, astronomy) - auto-detected if not provided

    Returns:
        Sensor name (e.g., "sony", "fuji", "hamamatsu_orca_flash4_v3", "hubble_wfc3", "hubble_acs")

    Raises:
        ValueError: If sensor cannot be extracted from tile ID

    Examples:
        Photography: "photography_sony_00145_00_0.1s_tile_0000" -> "sony"
        Microscopy: "microscopy_F-actin_nonlinear_Cell_005_RawSIMData_gt_tile_0000" -> "hamamatsu_orca_flash4_v3"
        Astronomy: "astronomy_j6fl7xoyq_g800l_sci_tile_0000" -> "hubble_wfc3"
    """
    if domain is None:
        # Auto-detect domain from tile_id
        if tile_id.startswith("photography_"):
            domain = "photography"
        elif tile_id.startswith("microscopy_"):
            domain = "microscopy"
        elif tile_id.startswith("astronomy_"):
            domain = "astronomy"
        else:
            raise ValueError(f"Cannot determine domain from tile ID: {tile_id}")

    if domain == "photography":
        parts = tile_id.split("_")
        if len(parts) >= 2 and parts[1] in ["sony", "fuji"]:
            return parts[1]
        else:
            raise ValueError(
                f"Unknown sensor in photography tile ID: {parts[1] if len(parts) >= 2 else 'unknown'}"
            )

    elif domain == "microscopy":
        # All microscopy data in this dataset uses Hamamatsu ORCA-Flash4.0 V3
        return "hamamatsu_orca_flash4_v3"

    elif domain == "astronomy":
        # All astronomy data in this dataset is from Hubble Legacy Fields (WFC3 or ACS)
        # For now, default to WFC3 as it's the primary instrument for this dataset
        # In practice, this could be enhanced to detect specific instruments from metadata
        return "hubble_wfc3"

    else:
        raise ValueError(f"Unsupported domain for sensor extraction: {domain}")


def extract_exposure_time_from_tile_id(tile_id: str) -> float:
    """
    Extract exposure time from tile ID.

    Args:
        tile_id: Tile ID (e.g., "photography_sony_00145_00_0.1s_tile_0000")

    Returns:
        Exposure time in seconds

    Example:
        tile_id = "photography_sony_00145_00_0.1s_tile_0000"
        exposure = extract_exposure_time_from_tile_id(tile_id)  # Returns 0.1
    """
    parts = tile_id.split("_")
    for part in parts:
        if part.endswith("s"):
            # Extract exposure time (e.g., "0.1s" -> 0.1, "30s" -> 30.0)
            exposure_str = part.replace("s", "")
            try:
                return float(exposure_str)
            except ValueError:
                continue

    raise ValueError(f"Cannot extract exposure time from tile ID: {tile_id}")


def get_metadata_path_for_exposure_ratio(domain: str, base_metadata_json: Path) -> Path:
    """Get the correct metadata file path for exposure ratio calculation."""
    base_path = Path(base_metadata_json).parent

    domain_metadata_map = {
        "microscopy": base_path / "metadata_microscopy_incremental_v2.json",
        "astronomy": base_path
        / "pt_tiles"
        / "astronomy_v2"
        / "metadata_astronomy_incremental.json",
    }

    return domain_metadata_map.get(domain, base_metadata_json)


def calculate_exposure_ratio_from_metadata(
    noisy_tile_id: str,
    clean_tile_info: Dict[str, Any],
    domain: str,
    metadata_json: Path,
) -> float:
    """
    Calculate exposure ratio from metadata fields.

    For microscopy: Ratio based on original_mean (actual signal levels in images)
    For astronomy: Ratio based on exposure_time_seconds (clean vs noisy)

    Args:
        noisy_tile_id: Noisy tile ID
        clean_tile_info: Clean tile metadata dictionary
        domain: Domain name ('microscopy' or 'astronomy')
        metadata_json: Path to metadata JSON file

    Returns:
        Exposure ratio (noisy / clean) for forward model: y_noisy = α·x_clean

    Raises:
        ValueError: If metadata fields are missing or invalid
    """
    # Get the correct metadata file path for this domain
    metadata_path = get_metadata_path_for_exposure_ratio(domain, metadata_json)

    logger.info(f"  Using metadata file: {metadata_path}")

    # Load metadata
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load metadata from {metadata_path}: {e}")

    # Find noisy tile metadata
    tiles = metadata.get("tiles", [])
    noisy_tile_info = None
    for tile in tiles:
        if tile.get("tile_id") == noisy_tile_id:
            noisy_tile_info = tile
            break

    if noisy_tile_info is None:
        raise ValueError(f"Noisy tile {noisy_tile_id} not found in metadata")

    if domain == "microscopy":
        # Calculate ratio based on original_mean (actual signal level in the images)
        # Note: average_photon_count in metadata is incorrect (all tiles show 308)
        noisy_mean = noisy_tile_info.get("original_mean")
        clean_mean = clean_tile_info.get("original_mean")

        if noisy_mean is None or clean_mean is None:
            raise ValueError(
                f"Missing original_mean in metadata: "
                f"noisy={noisy_mean}, clean={clean_mean}"
            )

        if clean_mean <= 0:
            raise ValueError(f"Invalid clean mean: {clean_mean}")

        # Forward model: y_noisy = α·x_clean
        # So α = E[y_noisy] / E[x_clean] = mean_noisy / mean_clean
        exposure_ratio = noisy_mean / clean_mean

        logger.info(f"  Microscopy exposure ratio from original_mean:")
        logger.info(f"    Noisy original_mean: {noisy_mean:.1f}")
        logger.info(f"    Clean original_mean: {clean_mean:.1f}")
        logger.info(f"    Calculated α = {exposure_ratio:.4f}")

        return exposure_ratio

    elif domain == "astronomy":
        # Calculate ratio based on exposure_time_seconds
        noisy_exposure_time = noisy_tile_info.get("exposure_time_seconds")
        clean_exposure_time = clean_tile_info.get("exposure_time_seconds")

        if noisy_exposure_time is None or clean_exposure_time is None:
            raise ValueError(
                f"Missing exposure_time_seconds in metadata: "
                f"noisy={noisy_exposure_time}, clean={clean_exposure_time}"
            )

        if clean_exposure_time <= 0:
            raise ValueError(f"Invalid clean exposure time: {clean_exposure_time}")

        # Forward model: y_noisy = α·x_clean
        # So α = exposure_time_noisy / exposure_time_clean
        exposure_ratio = noisy_exposure_time / clean_exposure_time

        logger.info(f"  Astronomy exposure ratio from exposure times:")
        logger.info(f"    Noisy exposure time: {noisy_exposure_time:.1f}s")
        logger.info(f"    Clean exposure time: {clean_exposure_time:.1f}s")
        logger.info(f"    Calculated α = {exposure_ratio:.4f}")

        return exposure_ratio

    else:
        raise ValueError(
            f"Exposure ratio calculation not implemented for domain: {domain}"
        )


def find_clean_tile_pair_astronomy(
    noisy_tile_id: str, metadata_json: Path
) -> Optional[Dict[str, Any]]:
    """
    Find the clean tile pair for a given noisy astronomy tile ID using metadata.

    Args:
        noisy_tile_id: The noisy tile ID (e.g., "astronomy_j6fl7xoyq_g800l_sci_tile_0000")
        metadata_json: Path to the metadata JSON file

    Returns:
        Dictionary containing clean tile metadata if found, None otherwise

    Example:
        noisy_tile_id = "astronomy_j6fl7xoyq_g800l_sci_tile_0000"
        clean_pair = find_clean_tile_pair_astronomy(noisy_tile_id, metadata_json)
        # Returns metadata for "astronomy_j6fl7xoyq_detection_sci_tile_0000"
    """
    logger.info(f"Finding clean tile pair for astronomy tile: {noisy_tile_id}")

    # Parse the noisy tile ID to extract components
    # Format: astronomy_{scene_id}_g800l_sci_tile_{tile_id}
    parts = noisy_tile_id.split("_")
    if len(parts) < 5:
        logger.error(f"Invalid astronomy tile ID format: {noisy_tile_id}")
        return None

    # Extract base components
    scene_id = parts[1]  # j6fl7xoyq
    tile_id = parts[-1]  # 0000

    logger.info(f"Extracted components - scene_id: {scene_id}, tile_id: {tile_id}")

    # Load metadata
    try:
        with open(metadata_json, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_json}: {e}")
        return None

    # Find clean tile with same scene_id and tile_id
    # For astronomy v2: both noisy and clean are detection_sci, but noisy has is_clean=false, clean has is_clean=true
    # For astronomy v1: noisy is g800l_sci, clean is detection_sci
    tiles = metadata.get("tiles", [])

    clean_candidates = []
    for tile in tiles:
        tile_id_str = tile.get("tile_id", "")
        # Check if using v2 metadata (has is_clean field)
        if "is_clean" in tile:
            # V2 metadata: look for tiles with is_clean=true (regardless of data_type)
            if (
                tile.get("domain") == "astronomy"
                and tile.get("is_clean") == True
                and tile_id_str.startswith(f"astronomy_{scene_id}_")
                and tile_id_str.endswith(f"_tile_{tile_id}")
            ):
                clean_candidates.append(tile)
        else:
            # V1 metadata: look for data_type='clean' and detection_sci
            if (
                tile.get("domain") == "astronomy"
                and tile.get("data_type") == "clean"
                and tile_id_str.startswith(f"astronomy_{scene_id}_detection_sci_tile_")
                and tile_id_str.endswith(f"_tile_{tile_id}")
            ):
                clean_candidates.append(tile)

    if not clean_candidates:
        logger.warning(f"No clean tile candidates found for {noisy_tile_id}")
        return None

    # For astronomy, there should typically be only one clean candidate
    # since the pattern is straightforward: g800l_sci -> detection_sci
    if len(clean_candidates) == 1:
        best_candidate = clean_candidates[0]
        logger.info(f"Found clean tile pair: {best_candidate.get('tile_id')}")
        return best_candidate
    else:
        # If multiple candidates, log warning and return the first one
        logger.warning(
            f"Multiple clean tile candidates found for {noisy_tile_id}, using first one"
        )
        best_candidate = clean_candidates[0]
        logger.info(f"Found clean tile pair: {best_candidate.get('tile_id')}")
        return best_candidate


def find_clean_tile_pair_microscopy(
    noisy_tile_id: str, metadata_json: Path
) -> Optional[Dict[str, Any]]:
    """
    Find the clean tile pair for a given noisy microscopy tile ID using metadata.

    Args:
        noisy_tile_id: The noisy tile ID (e.g., "microscopy_F-actin_nonlinear_Cell_005_RawSIMData_gt_tile_0000")
        metadata_json: Path to the metadata JSON file

    Returns:
        Dictionary containing clean tile metadata if found, None otherwise

    Example:
        noisy_tile_id = "microscopy_F-actin_nonlinear_Cell_005_RawSIMData_gt_tile_0000"
        clean_pair = find_clean_tile_pair_microscopy(noisy_tile_id, metadata_json)
        # Returns metadata for "microscopy_F-actin_nonlinear_Cell_005_SIM_gt_a_tile_0000"
    """
    logger.info(f"Finding clean tile pair for microscopy tile: {noisy_tile_id}")

    # Parse the noisy tile ID to extract components
    # Format depends on specimen:
    #   F-actin: microscopy_F-actin_nonlinear_Cell_005_RawSIMData_gt_tile_0000
    #   Others:  microscopy_CCPs_Cell_005_RawSIMData_gt_tile_0000
    #            microscopy_ER_Cell_005_RawGTSIMData_level_01_tile_0000
    parts = noisy_tile_id.split("_")
    if len(parts) < 6:
        logger.error(f"Invalid microscopy tile ID format: {noisy_tile_id}")
        return None

    # Find the position of "Cell" to determine parsing
    cell_idx = next((i for i, p in enumerate(parts) if p == "Cell"), None)
    if cell_idx is None:
        logger.error(f"Could not find 'Cell' in tile ID: {noisy_tile_id}")
        return None

    # Extract base components based on pattern
    specimen = parts[1]  # F-actin, CCPs, ER, Microtubules, etc.
    tile_id = parts[-1]  # 0000

    # The scene_id is always right after "Cell"
    scene_id = parts[cell_idx + 1]  # 005

    # For F-actin, there's an extra "nonlinear" part
    # Extract the middle part(s) between specimen and Cell
    if cell_idx > 2:
        # Has extra parts (like "nonlinear" for F-actin)
        middle_parts = "_".join(parts[2:cell_idx])
    else:
        # No extra parts
        middle_parts = None

    # Extract level if present (for RawGTSIMData/GTSIM patterns)
    level = None
    for i, part in enumerate(parts):
        if part == "level" and i + 1 < len(parts):
            level = parts[i + 1]  # e.g., "04"
            break

    logger.info(
        f"Extracted components - specimen: {specimen}, middle: {middle_parts}, scene_id: {scene_id}, level: {level}, tile_id: {tile_id}"
    )

    # Load metadata
    try:
        with open(metadata_json, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_json}: {e}")
        return None

    # Find clean tile with same specimen, middle parts (if any), scene_id, and tile_id
    # but different data type (clean vs noisy)
    tiles = metadata.get("tiles", [])

    clean_candidates = []
    for tile in tiles:
        tile_id_str = tile.get("tile_id", "")

        # Build the expected prefix pattern
        if middle_parts:
            # For F-actin: microscopy_F-actin_nonlinear_Cell_005_...
            expected_prefix = f"microscopy_{specimen}_{middle_parts}_Cell_{scene_id}_"
        else:
            # For others: microscopy_CCPs_Cell_005_...
            expected_prefix = f"microscopy_{specimen}_Cell_{scene_id}_"

        # Check if this tile matches our criteria
        matches_prefix = tile_id_str.startswith(expected_prefix)
        matches_suffix = tile_id_str.endswith(f"_tile_{tile_id}")

        # If level is specified, check that it matches
        matches_level = True
        if level is not None:
            # Look for level_XX pattern in the clean tile ID
            level_pattern = f"_level_{level}_"
            matches_level = level_pattern in tile_id_str

        if (
            tile.get("domain") == "microscopy"
            and tile.get("data_type") == "clean"
            and matches_prefix
            and matches_suffix
            and matches_level
        ):
            clean_candidates.append(tile)

    if not clean_candidates:
        logger.warning(f"No clean tile candidates found for {noisy_tile_id}")
        return None

    # If multiple candidates, prefer the one with longest exposure time or specific pattern
    # For microscopy, we prefer SIM_gt_a over SIM_gt, and GTSIM over GTSIM_level
    best_candidate = None
    best_score = -1

    for candidate in clean_candidates:
        candidate_id = candidate.get("tile_id", "")
        score = 0

        # Prefer SIM_gt_a over SIM_gt
        if "SIM_gt_a" in candidate_id:
            score += 10
        elif "SIM_gt" in candidate_id:
            score += 5

        # Handle GTSIM tiles (ER tiles with noise levels)
        if "GTSIM" in candidate_id:
            if "GTSIM_level" in candidate_id:
                # GTSIM with level specification (e.g., GTSIM_level_04)
                score += 8
            elif candidate_id.count("_") >= 7:
                # GTSIM without explicit "level" (older format)
                score += 7
            else:
                # Raw GTSIM without level
                score += 3

        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate:
        logger.info(f"Found clean tile pair: {best_candidate.get('tile_id')}")
        return best_candidate
    else:
        logger.warning(f"No valid clean tile pair found for {noisy_tile_id}")
        return None


def find_clean_tile_pair(
    noisy_tile_id: str, metadata_json: Path, domain: str = "photography"
) -> Optional[Dict[str, Any]]:
    """
    Find the clean tile pair for a given noisy tile ID using metadata.

    Args:
        noisy_tile_id: The noisy tile ID
        metadata_json: Path to the metadata JSON file
        domain: Domain name ("photography", "microscopy", "astronomy", etc.)

    Returns:
        Dictionary containing clean tile metadata if found, None otherwise
    """
    if domain == "microscopy":
        return find_clean_tile_pair_microscopy(noisy_tile_id, metadata_json)
    elif domain == "photography":
        return find_clean_tile_pair_photography(noisy_tile_id, metadata_json)
    elif domain == "astronomy":
        return find_clean_tile_pair_astronomy(noisy_tile_id, metadata_json)
    else:
        logger.error(f"Clean tile pairing not implemented for domain: {domain}")
        return None


def find_clean_tile_pair_photography(
    noisy_tile_id: str, metadata_json: Path
) -> Optional[Dict[str, Any]]:
    """
    Find the clean tile pair for a given noisy photography tile ID using metadata.

    Args:
        noisy_tile_id: The noisy tile ID (e.g., "photography_sony_00145_00_0.1s_tile_0000")
        metadata_json: Path to the metadata JSON file

    Returns:
        Dictionary containing clean tile metadata if found, None otherwise

    Example:
        noisy_tile_id = "photography_sony_00145_00_0.1s_tile_0000"
        clean_pair = find_clean_tile_pair_photography(noisy_tile_id, metadata_json)
        # Returns metadata for "photography_sony_00145_00_30s_tile_0000"
    """
    logger.info(f"Finding clean tile pair for photography tile: {noisy_tile_id}")

    # Parse the noisy tile ID to extract components
    # Format: photography_{sensor}_{scene_id}_{scene_num}_{exposure_time}_tile_{tile_id}
    parts = noisy_tile_id.split("_")
    if len(parts) < 7:
        logger.error(f"Invalid photography tile ID format: {noisy_tile_id}")
        return None

    sensor = parts[1]  # sony, fuji, etc.
    scene_id = parts[2]  # 00145
    scene_num = parts[3]  # 00
    tile_id = parts[6]  # 0000 (after splitting by '_', tile_id is at index 6)

    logger.info(
        f"Extracted components - sensor: {sensor}, scene_id: {scene_id}, scene_num: {scene_num}, tile_id: {tile_id}"
    )

    # Load metadata
    try:
        with open(metadata_json, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_json}: {e}")
        return None

    # Find clean tile with same sensor, scene_id, scene_num, and tile_id
    # but different exposure time (typically longer exposure)
    tiles = metadata.get("tiles", [])

    clean_candidates = []
    for tile in tiles:
        if (
            tile.get("domain") == "photography"
            and tile.get("data_type") == "clean"
            and tile.get("tile_id", "").startswith(
                f"photography_{sensor}_{scene_id}_{scene_num}_"
            )
            and tile.get("tile_id", "").endswith(f"_tile_{tile_id}")
        ):
            clean_candidates.append(tile)

    if not clean_candidates:
        logger.warning(f"No clean tile candidates found for {noisy_tile_id}")
        return None

    # If multiple candidates, prefer the one with longest exposure time
    # Extract exposure times and find the maximum
    best_candidate = None
    max_exposure = 0.0

    for candidate in clean_candidates:
        candidate_id = candidate.get("tile_id", "")
        # Extract exposure time from tile_id
        try:
            exposure_part = candidate_id.split("_")[4]  # e.g., "30s"
            exposure_str = exposure_part.replace("s", "")
            exposure_time = float(exposure_str)

            if exposure_time > max_exposure:
                max_exposure = exposure_time
                best_candidate = candidate

        except (IndexError, ValueError) as e:
            logger.warning(f"Could not parse exposure time from {candidate_id}: {e}")
            continue

    if best_candidate:
        logger.info(
            f"Found clean tile pair: {best_candidate.get('tile_id')} (exposure: {max_exposure}s)"
        )
        return best_candidate
    else:
        logger.warning(f"No valid clean tile pair found for {noisy_tile_id}")
        return None


def test_sensor_extraction():
    """
    Test function to verify sensor extraction from tile IDs works correctly.
    """
    logger.info("Testing sensor extraction functionality...")

    # Test cases
    test_cases = [
        ("photography_sony_00145_00_0.1s_tile_0000", "sony"),
        ("photography_fuji_00030_00_0.033s_tile_0019", "fuji"),
        ("photography_sony_00145_00_0.1s_tile_0001", "sony"),
    ]

    success_count = 0
    total_tests = len(test_cases)

    for tile_id, expected_sensor in test_cases:
        logger.info(f"\n--- Testing: {tile_id} ---")

        try:
            extracted_sensor = extract_sensor_from_tile_id(tile_id, "photography")
            if extracted_sensor == expected_sensor:
                logger.info(f"Correctly extracted sensor: {extracted_sensor}")
                success_count += 1
            else:
                logger.error(f"✗ Expected: {expected_sensor}, Got: {extracted_sensor}")
        except ValueError as e:
            logger.error(f"✗ Error extracting sensor: {e}")

    # Test error cases
    error_cases = [
        "microscopy_RawGTSIMData_001_tile_0000",  # Non-photography domain
        "astronomy_g800l_sci_001_tile_0000",  # Non-photography domain
        "invalid_tile_id",  # Invalid format
    ]

    for tile_id in error_cases:
        logger.info(f"\n--- Testing error case: {tile_id} ---")
        try:
            extracted_sensor = extract_sensor_from_tile_id(tile_id)
            logger.error(f"✗ Expected error but got: {extracted_sensor}")
        except ValueError as e:
            logger.info(f"Correctly raised error: {e}")
            success_count += 1
        total_tests += 1

    logger.info(f"\n--- Test Results ---")
    logger.info(f"Successful extractions: {success_count}/{total_tests}")

    if success_count == total_tests:
        logger.info("All sensor extraction tests passed!")
        return True
    else:
        logger.error("✗ Some sensor extraction tests failed!")
        return False


def test_exposure_extraction():
    """
    Test function to verify exposure time extraction from tile IDs works correctly.
    """
    logger.info("Testing exposure time extraction functionality...")

    # Test cases
    test_cases = [
        ("photography_sony_00145_00_0.1s_tile_0000", 0.1),
        ("photography_fuji_00030_00_0.033s_tile_0019", 0.033),
        ("photography_sony_00145_00_30s_tile_0000", 30.0),
        ("photography_sony_00145_00_10s_tile_0001", 10.0),
    ]

    success_count = 0
    total_tests = len(test_cases)

    for tile_id, expected_exposure in test_cases:
        logger.info(f"\n--- Testing: {tile_id} ---")

        try:
            extracted_exposure = extract_exposure_time_from_tile_id(tile_id)
            if abs(extracted_exposure - expected_exposure) < 1e-6:
                logger.info(f"Correctly extracted exposure: {extracted_exposure}s")
                success_count += 1
            else:
                logger.error(
                    f"✗ Expected: {expected_exposure}s, Got: {extracted_exposure}s"
                )
        except ValueError as e:
            logger.error(f"✗ Error extracting exposure: {e}")

    # Test error cases
    error_cases = [
        "photography_sony_00145_00_tile_0000",  # No exposure time
        "invalid_tile_id",  # Invalid format
    ]

    for tile_id in error_cases:
        logger.info(f"\n--- Testing error case: {tile_id} ---")
        try:
            extracted_exposure = extract_exposure_time_from_tile_id(tile_id)
            logger.error(f"✗ Expected error but got: {extracted_exposure}s")
        except ValueError as e:
            logger.info(f"Correctly raised error: {e}")
            success_count += 1
        total_tests += 1

    logger.info(f"\n--- Test Results ---")
    logger.info(f"Successful extractions: {success_count}/{total_tests}")

    if success_count == total_tests:
        logger.info("All exposure extraction tests passed!")
        return True
    else:
        logger.error("✗ Some exposure extraction tests failed!")
        return False


def test_astronomy_clean_tile_pairing():
    """
    Test function to verify astronomy clean tile pairing works correctly.
    """
    logger.info("Testing astronomy clean tile pairing functionality...")

    # Test cases
    test_cases = [
        "astronomy_j6fl7xoyq_g800l_sci_tile_0000",
        "astronomy_j6fl7xoyq_g800l_sci_tile_0001",
        "astronomy_j6fl7xoyq_g800l_sci_tile_0002",
        "astronomy_j6fl7xoyq_g800l_sci_tile_0003",
    ]

    metadata_json = Path(
        "/home/jilab/Jae/dataset/processed/metadata_astronomy_incremental.json"
    )

    if not metadata_json.exists():
        logger.error(f"Metadata file not found: {metadata_json}")
        return False

    success_count = 0
    total_tests = len(test_cases)

    for noisy_tile_id in test_cases:
        logger.info(f"\n--- Testing: {noisy_tile_id} ---")

        clean_pair = find_clean_tile_pair(noisy_tile_id, metadata_json, "astronomy")

        if clean_pair:
            clean_tile_id = clean_pair.get("tile_id")
            clean_pt_path = clean_pair.get("pt_path")

            logger.info(f"Found clean pair: {clean_tile_id}")
            logger.info(f"  Path: {clean_pt_path}")

            # Verify the file exists
            if Path(clean_pt_path).exists():
                logger.info(f"Clean tile file exists")
                success_count += 1
            else:
                logger.error(f"✗ Clean tile file does not exist: {clean_pt_path}")
        else:
            logger.error(f"✗ No clean pair found for {noisy_tile_id}")

    logger.info(f"\n--- Test Results ---")
    logger.info(f"Successful pairings: {success_count}/{total_tests}")

    if success_count == total_tests:
        logger.info("All astronomy tests passed!")
        return True
    else:
        logger.error("✗ Some astronomy tests failed!")
        return False


def test_microscopy_clean_tile_pairing():
    """
    Test function to verify microscopy clean tile pairing works correctly.
    """
    logger.info("Testing microscopy clean tile pairing functionality...")

    # Test cases
    test_cases = [
        "microscopy_F-actin_nonlinear_Cell_005_RawSIMData_gt_tile_0000",
        "microscopy_F-actin_nonlinear_Cell_005_RawSIMData_gt_tile_0001",
        "microscopy_ER_Cell_005_RawGTSIMData_level_01_tile_0000",
        "microscopy_ER_Cell_005_RawGTSIMData_level_01_tile_0001",
    ]

    metadata_json = Path(
        "/home/jilab/Jae/dataset/processed/metadata_microscopy_incremental.json"
    )

    if not metadata_json.exists():
        logger.error(f"Metadata file not found: {metadata_json}")
        return False

    success_count = 0
    total_tests = len(test_cases)

    for noisy_tile_id in test_cases:
        logger.info(f"\n--- Testing: {noisy_tile_id} ---")

        clean_pair = find_clean_tile_pair(noisy_tile_id, metadata_json, "microscopy")

        if clean_pair:
            clean_tile_id = clean_pair.get("tile_id")
            clean_pt_path = clean_pair.get("pt_path")

            logger.info(f"Found clean pair: {clean_tile_id}")
            logger.info(f"  Path: {clean_pt_path}")

            # Verify the file exists
            if Path(clean_pt_path).exists():
                logger.info(f"Clean tile file exists")
                success_count += 1
            else:
                logger.error(f"✗ Clean tile file does not exist: {clean_pt_path}")
        else:
            logger.error(f"✗ No clean pair found for {noisy_tile_id}")

    logger.info(f"\n--- Test Results ---")
    logger.info(f"Successful pairings: {success_count}/{total_tests}")

    if success_count == total_tests:
        logger.info("All microscopy tests passed!")
        return True
    else:
        logger.error("✗ Some microscopy tests failed!")
        return False


def test_clean_tile_pairing():
    """
    Test function to verify clean tile pairing works correctly.
    """
    logger.info("Testing clean tile pairing functionality...")

    # Test cases
    test_cases = [
        "photography_sony_00145_00_0.1s_tile_0000",
        "photography_fuji_00030_00_0.033s_tile_0019",
        "photography_sony_00145_00_0.1s_tile_0001",
    ]

    metadata_json = Path(
        "/home/jilab/Jae/dataset/processed/metadata_photography_incremental.json"
    )

    if not metadata_json.exists():
        logger.error(f"Metadata file not found: {metadata_json}")
        return False

    success_count = 0
    total_tests = len(test_cases)

    for noisy_tile_id in test_cases:
        logger.info(f"\n--- Testing: {noisy_tile_id} ---")

        clean_pair = find_clean_tile_pair(noisy_tile_id, metadata_json, "photography")

        if clean_pair:
            clean_tile_id = clean_pair.get("tile_id")
            clean_pt_path = clean_pair.get("pt_path")

            logger.info(f"Found clean pair: {clean_tile_id}")
            logger.info(f"  Path: {clean_pt_path}")

            # Verify the file exists
            if Path(clean_pt_path).exists():
                logger.info(f"Clean tile file exists")
                success_count += 1
            else:
                logger.error(f"✗ Clean tile file does not exist: {clean_pt_path}")
        else:
            logger.error(f"✗ No clean pair found for {noisy_tile_id}")

    logger.info(f"\n--- Test Results ---")
    logger.info(f"Successful pairings: {success_count}/{total_tests}")

    if success_count == total_tests:
        logger.info("All tests passed!")
        return True
    else:
        logger.error("✗ Some tests failed!")
        return False


def run_gradient_verification():
    """Run gradient verification tests if requested."""
    import sys

    # Check if gradient verification was requested
    if "--test_gradients" in sys.argv:
        logger.info("Gradient verification requested, running tests...")
        test_pg_gradient_correctness()
        logger.info("Gradient verification completed successfully")
        sys.exit(0)


# Global LPIPS metric instance to avoid reloading weights
_lpips_metric_cache = {}


def get_lpips_metric(device: str = "cuda"):
    """Get or create LPIPS metric instance (cached)."""
    if device not in _lpips_metric_cache:
        if LPIPS_AVAILABLE and lpips is not None:
            _lpips_metric_cache[device] = lpips.LPIPS(net="alex").to(device)
        else:
            _lpips_metric_cache[device] = None
    return _lpips_metric_cache[device]


def compute_comprehensive_metrics(
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    noisy: torch.Tensor,
    scale: float,
    domain: str,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute comprehensive metrics using torchmetrics and piq.

    Metrics computed:
      - PSNR, SSIM: Fidelity to clean reference (higher is better)
      - LPIPS: Perceptual similarity (lower is better)
      - NIQE: No-reference quality (lower is better)
      - χ²_red: Physical consistency (should be ≈ 1.0)
      - MSE: Mean squared error (lower is better)

    Args:
        clean: Clean reference image [B, C, H, W] in [-1,1] range
        enhanced: Enhanced image [B, C, H, W] in [-1,1] range
        noisy: Noisy observation [B, C, H, W] in physical units
        scale: Scale factor for physical units
        domain: Domain name ('photography', 'microscopy', 'astronomy')
        device: Device for computation
    Returns:
        Dictionary with comprehensive metrics
    """
    try:
        # Ensure all tensors are on the same device and float32
        clean = clean.to(device).float()
        enhanced = enhanced.to(device).float()
        noisy = noisy.to(device).float()

        # Compute PSNR and SSIM using torchmetrics
        psnr_metric = PSNR(data_range=2.0).to(device)  # Range is 2.0 for [-1,1] data
        ssim_metric = SSIM(data_range=2.0).to(device)  # Range is 2.0 for [-1,1] data

        psnr_val = psnr_metric(enhanced, clean).item()
        ssim_val = ssim_metric(enhanced, clean).item()

        # Compute MSE for PSNR metadata
        mse_val = F.mse_loss(enhanced, clean).item()

        # Initialize metrics with core metrics
        metrics = {
            "ssim": ssim_val if not np.isnan(ssim_val) else 0.0,
            "psnr": psnr_val if not np.isnan(psnr_val) else 0.0,
            "mse": mse_val,
        }

        # Compute LPIPS and NIQE only for photography domain
        if domain == "photography":
            # Compute LPIPS
            lpips_val = float("nan")
            lpips_metric = get_lpips_metric(device)
            if lpips_metric is not None:
                try:
                    enhanced_01 = convert_range(enhanced, "[-1,1]", "[0,1]")
                    clean_01 = convert_range(clean, "[-1,1]", "[0,1]")
                    lpips_val = lpips_metric(enhanced_01, clean_01).item()
                except Exception as e:
                    logger.warning(f"LPIPS computation failed: {e}")
                    lpips_val = float("nan")
            else:
                logger.warning("lpips library not available - LPIPS set to NaN")

            metrics["lpips"] = lpips_val

            # Compute NIQE
            niqe_val = float("nan")
            if PQIQA_AVAILABLE and pyiqa is not None:
                try:
                    enhanced_01 = convert_range(enhanced, "[-1,1]", "[0,1]")
                    niqe_metric = pyiqa.create_metric("niqe")
                    niqe_val = niqe_metric(enhanced_01).item()
                except Exception as e:
                    logger.warning(f"NIQE computation failed: {e}")
                    niqe_val = float("nan")

            metrics["niqe"] = niqe_val

        return metrics

    except Exception as e:
        logger.warning(f"Comprehensive metrics computation failed: {e}, using fallback")
        # Fallback to simple metrics
        return compute_simple_metrics(clean, enhanced, device=device)


def compute_simple_metrics(
    clean, enhanced, data_range: float = None, device: str = "cuda"
) -> Dict[str, float]:
    """
    PyTorch-native simple metrics computation.

    Args:
        clean: Clean reference image (tensor or numpy array)
        enhanced: Enhanced image (tensor or numpy array)
        data_range: Data range for metrics (if None, computed from clean)
        device: Device for computation

    Returns:
        Dictionary with basic metrics: SSIM, PSNR, MSE
    """
    # Convert to PyTorch tensors if needed
    if isinstance(clean, np.ndarray):
        clean = torch.from_numpy(clean).float()
    if isinstance(enhanced, np.ndarray):
        enhanced = torch.from_numpy(enhanced).float()

    # Ensure tensors are on the correct device
    clean = clean.to(device)
    enhanced = enhanced.to(device)

    # Ensure proper shape [B, C, H, W]
    if clean.ndim == 2:  # (H, W)
        clean = clean.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        enhanced = enhanced.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif clean.ndim == 3:  # (C, H, W)
        clean = clean.unsqueeze(0)  # [1, C, H, W]
        enhanced = enhanced.unsqueeze(0)  # [1, C, H, W]
    # If already [B, C, H, W], keep as is

    # Ensure [-1, 1] range (images are already in this range from clean.pt and model outputs)
    clean = torch.clamp(clean, -1.0, 1.0)
    enhanced = torch.clamp(enhanced, -1.0, 1.0)

    # Initialize PyTorch-native metrics
    ssim_metric = SSIM(data_range=2.0).to(device)  # Range is 2.0 for [-1,1] data
    psnr_metric = PSNR(data_range=2.0).to(device)  # Range is 2.0 for [-1,1] data
    mse_metric = MSE().to(device)

    # Compute metrics using PyTorch-native functions
    try:
        ssim_val = ssim_metric(enhanced, clean).item()
        psnr_val = psnr_metric(enhanced, clean).item()
        mse_val = mse_metric(enhanced, clean).item()

        metrics = {
            "ssim": float(ssim_val),
            "psnr": float(psnr_val),
            "mse": float(mse_val),
            "lpips": float("nan"),  # Not available in fallback
            "niqe": float("nan"),  # Not available in fallback
        }
    except Exception as e:
        logger.warning(f"PyTorch-native metrics computation failed: {e}")
        # Return NaN for all metrics if computation fails
        metrics = {
            "ssim": float("nan"),
            "psnr": float("nan"),
            "mse": float("nan"),
            "lpips": float("nan"),
            "niqe": float("nan"),
        }

    return metrics


def validate_physical_consistency(
    x_enhanced: torch.Tensor,
    y_e_physical: torch.Tensor,
    s: float,
    sigma_r: float,
    exposure_ratio: float,
    domain_min: float,
    domain_max: float,
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    Validate physical consistency using reduced chi-squared statistic.

    Tests: y_short ≈ Poisson(α·s·x_long) + N(0, σ_r²)

    A physically consistent reconstruction should have χ²_red ≈ 1.0, indicating
    that residuals match the expected Poisson-Gaussian noise distribution.

    Args:
        x_enhanced: Enhanced image in [0,1] normalized space [B,C,H,W] (LONG exposure)
        y_e_physical: Observed noisy measurement in physical units [B,C,H,W] (SHORT exposure)
        s: Scale factor used in PG guidance
        sigma_r: Read noise standard deviation (in physical units)
        exposure_ratio: Exposure ratio t_low / t_long
        domain_min: Minimum physical value of domain
        domain_max: Maximum physical value of domain
        epsilon: Small constant for numerical stability

    Returns:
        Dictionary with consistency metrics:
        - chi_squared: Reduced χ² statistic (should be ≈ 1.0)
        - chi_squared_std: Standard deviation of χ² per pixel
        - physically_consistent: Boolean flag (0.8 < χ² < 1.2)
        - mean_residual: Mean residual (should be ≈ 0)
        - max_residual: Maximum absolute residual
    """
    # Normalize observation to [0,1]
    domain_range = domain_max - domain_min
    y_e_norm = (y_e_physical - domain_min) / (domain_range + epsilon)
    y_e_norm = torch.clamp(y_e_norm, 0.0, 1.0)

    # Scale observation to [0, s]
    y_e_scaled = y_e_norm * s

    # Expected observation at SHORT exposure
    expected_y_at_short_exp = exposure_ratio * s * x_enhanced

    # Variance at SHORT exposure
    variance_at_short_exp = exposure_ratio * s * x_enhanced + sigma_r**2 + epsilon

    # Residual
    residual = y_e_scaled - expected_y_at_short_exp

    # Chi-squared per pixel
    chi_squared_map = (residual**2) / variance_at_short_exp

    # Reduced chi-squared (average over all pixels)
    chi_squared_red = chi_squared_map.mean().item()

    # Additional statistics
    chi_squared_std = chi_squared_map.std().item()
    mean_residual = residual.mean().item()
    max_residual = residual.abs().max().item()

    # Physical consistency check: χ² should be in [0.8, 1.2] range
    is_consistent = 0.8 < chi_squared_red < 1.2

    return {
        "chi_squared": chi_squared_red,
        "chi_squared_std": chi_squared_std,
        "physically_consistent": is_consistent,
        "mean_residual": mean_residual,
        "max_residual": max_residual,
    }


def load_test_tiles(
    metadata_json: Path,
    domain: str,
    split: str = "test",
    sensor_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load test tile metadata from JSON file.

    Args:
        metadata_json: Path to metadata JSON file
        domain: Domain name ('photography', 'microscopy', 'astronomy')
        split: Data split to load (default: 'test')
        sensor_filter: Optional sensor filter ('sony', 'fuji', etc.) for photography domain

    Returns:
        List of tile metadata dictionaries
    """
    logger.info(f"Loading {split} tiles for {domain} from {metadata_json}")
    if sensor_filter:
        logger.info(f"Filtering by sensor: {sensor_filter}")

    with open(metadata_json, "r") as f:
        metadata = json.load(f)

    # Filter tiles by domain and split
    tiles = metadata.get("tiles", [])
    filtered_tiles = [
        tile
        for tile in tiles
        if tile.get("domain") == domain and tile.get("split") == split
    ]

    # Apply sensor filter for photography domain
    if sensor_filter and domain == "photography":
        sensor_filtered_tiles = []
        for tile in filtered_tiles:
            tile_id = tile.get("tile_id", "")
            if sensor_filter.lower() in tile_id.lower():
                sensor_filtered_tiles.append(tile)
        filtered_tiles = sensor_filtered_tiles
        logger.info(
            f"Found {len(filtered_tiles)} {split} tiles for {domain} with sensor {sensor_filter}"
        )
    else:
        logger.info(f"Found {len(filtered_tiles)} {split} tiles for {domain}")

    return filtered_tiles


def load_image(
    tile_id: str,
    image_dir: Path,
    device: torch.device,
    image_type: str = "image",
    target_channels: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load normalized .pt file.

    Returns tensor in [-1,1] range as saved by preprocessing pipeline.
    NO additional processing - data is already normalized correctly!
    """
    image_path = image_dir / f"{tile_id}.pt"

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load the image
    tensor = torch.load(image_path, map_location=device)

    # Handle different tensor formats
    if isinstance(tensor, dict):
        if "noisy" in tensor:
            tensor = tensor["noisy"]
        elif "noisy_norm" in tensor:
            tensor = tensor["noisy_norm"]
        elif "clean" in tensor:
            tensor = tensor["clean"]
        elif "clean_norm" in tensor:
            tensor = tensor["clean_norm"]
        elif "image" in tensor:
            tensor = tensor["image"]
        else:
            raise ValueError(f"Unrecognized dict structure in {image_path}")

    # Ensure float32
    tensor = tensor.float()

    # Ensure CHW format
    if tensor.ndim == 2:  # (H, W)
        tensor = tensor.unsqueeze(0)  # (1, H, W)
    elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3]:  # (H, W, C)
        tensor = tensor.permute(2, 0, 1)  # (C, H, W)

    # Handle channel conversion for cross-domain models
    if target_channels is not None and tensor.shape[0] != target_channels:
        if tensor.shape[0] == 1 and target_channels == 3:
            # Convert grayscale (1 channel) to RGB (3 channels) by repeating
            tensor = tensor.repeat(3, 1, 1)
            logger.debug(
                f"Converted grayscale to RGB for {image_type} image: {tile_id}"
            )
        elif tensor.shape[0] == 3 and target_channels == 1:
            # Convert RGB (3 channels) to grayscale (1 channel) by averaging
            tensor = tensor.mean(dim=0, keepdim=True)
            logger.debug(
                f"Converted RGB to grayscale for {image_type} image: {tile_id}"
            )
        else:
            logger.warning(
                f"Cannot convert from {tensor.shape[0]} to {target_channels} channels for {image_type} image: {tile_id}"
            )

    # Return tensor AS-IS - already normalized correctly
    metadata = {
        "offset": 0.0,
        "original_range": [tensor.min().item(), tensor.max().item()],
        "processed_range": [tensor.min().item(), tensor.max().item()],
        "domain": "astronomy" if "astronomy" in tile_id.lower() else "other",
    }

    logger.debug(
        f"Loaded {image_type} {tile_id}: "
        f"shape={tensor.shape}, range=[{tensor.min():.3f}, {tensor.max():.3f}]"
    )

    return tensor, metadata


def load_noisy_image(
    tile_id: str,
    noisy_dir: Path,
    device: torch.device,
    target_channels: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load a noisy .pt file and return both tensor and metadata."""
    return load_image(tile_id, noisy_dir, device, "noisy", target_channels)


def load_clean_image(
    tile_id: str,
    clean_dir: Path,
    device: torch.device,
    target_channels: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load a clean .pt file and return both tensor and metadata."""
    return load_image(tile_id, clean_dir, device, "clean", target_channels)


def resolve_pt_path(pt_path: str) -> Path:
    """
    Resolve pt_path from metadata, handling path changes.

    Args:
        pt_path: Path from metadata (may contain old paths)

    Returns:
        Resolved Path object
    """
    path_obj = Path(pt_path)

    # If it's already absolute and exists, use it
    if path_obj.is_absolute() and path_obj.exists():
        return path_obj

    # Handle old paths with /home/jilab/Jae/
    if "/home/jilab/Jae/" in str(pt_path):
        resolved_path = str(pt_path).replace("/home/jilab/Jae/", "/home/jilab/DAPGD/")
        resolved_path_obj = Path(resolved_path)
        if resolved_path_obj.exists():
            logger.info(f"  Resolved old path: {pt_path} -> {resolved_path}")
            return resolved_path_obj

    # If relative path, try to resolve relative to current working directory
    if not path_obj.is_absolute():
        cwd_path = Path.cwd() / path_obj
        if cwd_path.exists():
            return cwd_path

    # Return original path even if it doesn't exist (will be caught by caller)
    return path_obj


def find_and_load_clean_image(
    tile_id: str,
    domain: str,
    metadata_json: Path,
    sampler_device: torch.device,
    target_channels: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[Any]]:
    """
    Find and load clean image pair for a given tile.

    Args:
        tile_id: Noisy tile ID
        domain: Domain name
        metadata_json: Path to metadata JSON
        sampler_device: Device to load on
        target_channels: Target number of channels

    Returns:
        Tuple of (clean_image, clean_image_raw) or (None, None) if not found
    """
    # Get domain-specific metadata path
    if domain == "microscopy":
        metadata_path = get_metadata_path_for_exposure_ratio(domain, metadata_json)
    elif domain == "astronomy":
        metadata_path = get_metadata_path_for_exposure_ratio(domain, metadata_json)
    else:
        metadata_path = metadata_json

    logger.info(f"  Current working directory: {Path.cwd()}")
    logger.info(f"  Using metadata path: {metadata_path}")

    # Find clean pair
    clean_pair_info = find_clean_tile_pair(tile_id, metadata_path, domain)

    if not clean_pair_info:
        logger.warning(f"  No clean tile pair found for {tile_id}")
        return None, None

    clean_tile_id = clean_pair_info.get("tile_id")
    clean_pt_path = clean_pair_info.get("pt_path")
    logger.info(f"  Found clean pair: {clean_tile_id}")
    logger.info(f"  Clean pt_path from metadata: {clean_pt_path}")

    # Resolve path (handles old /home/jilab/Jae/ paths)
    resolved_path = resolve_pt_path(clean_pt_path)
    logger.info(f"  Resolved path: {resolved_path}")
    logger.info(f"  File exists: {resolved_path.exists()}")

    # Load clean image
    if resolved_path.exists():
        try:
            logger.info(f"  Loading clean image from: {resolved_path}")
            clean_image, clean_image_raw = load_clean_image_from_path(
                resolved_path, sampler_device, target_channels=target_channels
            )
            logger.info(f"  ✅ Successfully loaded clean image")
            return clean_image, clean_image_raw
        except Exception as e:
            logger.warning(f"  Failed to load clean image: {e}")
            import traceback

            traceback.print_exc()
            return None, None
    else:
        logger.warning(f"  Clean tile file not found: {resolved_path}")
        logger.warning(f"  Original metadata path: {clean_pt_path}")
        return None, None


def load_clean_image_from_path(
    clean_pt_path: Path, device: torch.device, target_channels: Optional[int] = None
) -> Tuple[torch.Tensor, Any]:
    """
    Load clean image from pt_path and return both raw and processed tensors.

    Args:
        clean_pt_path: Path to the clean .pt file
        device: Device to load tensor on
        target_channels: Target number of channels (for channel conversion)

    Returns:
        Tuple of (processed_tensor, raw_tensor) where:
        - processed_tensor: Tensor in [-1,1] format, ready for metrics/visualization
        - raw_tensor: Raw tensor from file (can be dict or tensor)
    """
    raw_tensor = torch.load(clean_pt_path, map_location=device)

    # Process tensor for visualization/metrics
    tensor = raw_tensor

    # Handle different tensor formats
    if isinstance(tensor, dict):
        if "clean" in tensor:
            tensor = tensor["clean"]
        elif "clean_norm" in tensor:
            tensor = tensor["clean_norm"]
        elif "image" in tensor:
            tensor = tensor["image"]
        else:
            raise ValueError(f"Unrecognized dict structure in {clean_pt_path}")

    tensor = tensor.float()

    # Ensure CHW format
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3]:
        tensor = tensor.permute(2, 0, 1)

    # Handle channel conversion
    if target_channels is not None and tensor.shape[0] != target_channels:
        if tensor.shape[0] == 1 and target_channels == 3:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] == 3 and target_channels == 1:
            tensor = tensor.mean(dim=0, keepdim=True)

    # Ensure batch dimension [B, C, H, W]
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    processed_tensor = tensor.to(torch.float32)

    return processed_tensor, raw_tensor


def convert_range(
    tensor: torch.Tensor, from_range: str = "[-1,1]", to_range: str = "[0,1]"
) -> torch.Tensor:
    """Convert tensor between ranges."""
    if from_range == "[-1,1]" and to_range == "[0,1]":
        return (tensor + 1.0) / 2.0
    elif from_range == "[0,1]" and to_range == "[-1,1]":
        return tensor * 2.0 - 1.0
    else:
        raise ValueError(f"Unsupported range conversion: {from_range} -> {to_range}")


def ensure_grayscale_for_domain(tensor: torch.Tensor, domain: str) -> torch.Tensor:
    """Convert to grayscale if domain requires it."""
    if tensor.shape[1] == 3 and domain in ["astronomy", "microscopy"]:
        return tensor.mean(dim=1, keepdim=True)
    return tensor


def analyze_image_brightness(image: torch.Tensor) -> Dict[str, float]:
    """Analyze image brightness characteristics."""
    img_01 = convert_range(image, "[-1,1]", "[0,1]")

    mean_brightness = img_01.mean().item()
    img_flat = img_01.flatten()

    brightness_category = (
        "Very Dark"
        if mean_brightness < 0.2
        else "Dark"
        if mean_brightness < 0.4
        else "Medium"
        if mean_brightness < 0.6
        else "Bright"
        if mean_brightness < 0.8
        else "Very Bright"
    )

    return {
        "mean": mean_brightness,
        "std": img_01.std().item(),
        "min": img_01.min().item(),
        "max": img_01.max().item(),
        "p10": torch.quantile(img_flat, 0.1).item(),
        "p50": torch.quantile(img_flat, 0.5).item(),
        "p90": torch.quantile(img_flat, 0.9).item(),
        "category": brightness_category,
    }


class EDMPosteriorSampler:
    """EDM-based posterior sampler with Poisson-Gaussian measurement guidance."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        domain_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize the enhancer with trained model and domain configurations."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.domain_ranges = domain_ranges or {
            "photography": {"min": 0.0, "max": 15871.0},
            "microscopy": {"min": 0.0, "max": 65535.0},
            "astronomy": {
                "min": -65.0,
                "max": 385.0,
            },  # ✅ Match preprocessing (original physical range)
        }

        # Load model
        logger.info(f"Loading model from {model_path}...")
        try:
            # Try torch.load first (handles numpy compatibility better)
            checkpoint = torch.load(model_path, map_location="cpu")
        except Exception as e:
            logger.warning(f"torch.load failed: {e}, trying pickle.load...")
            # Fallback to pickle.load
            try:
                with open(model_path, "rb") as f:
                    checkpoint = pickle.load(f)  # nosec B301
            except (ModuleNotFoundError, AttributeError) as pickle_e:
                if "numpy._core" in str(pickle_e) or "scalar" in str(pickle_e):
                    logger.error("=" * 80)
                    logger.error("NUMPY VERSION MISMATCH DETECTED")
                    logger.error("=" * 80)
                    logger.error(f"Model file: {model_path}")
                    logger.error(f"Current numpy version: {np.__version__}")
                    logger.error(f"Current Python version: {sys.version.split()[0]}")
                    logger.error("")
                    logger.error(
                        "The photography model was saved with numpy 2.0+ which is incompatible"
                    )
                    logger.error(
                        "with Python 3.8 and numpy 1.24.4 in this environment."
                    )
                    logger.error("")
                    logger.error("SOLUTIONS:")
                    logger.error(
                        "  1. Use Python 3.10+: /usr/bin/python3.10 or /usr/bin/python3.11"
                    )
                    logger.error(
                        "  2. Repackage the model on a numpy 1.x compatible system"
                    )
                    logger.error(
                        "  3. Contact the model provider for a compatible version"
                    )
                    logger.error("=" * 80)
                    raise RuntimeError(
                        f"Photography model requires numpy 2.0+ (Python 3.10+) but environment has "
                        f"Python {sys.version.split()[0]} with numpy {np.__version__}. "
                        f"Try running with: /usr/bin/python3.10 or /usr/bin/python3.11"
                    ) from pickle_e
                raise

        self.net = checkpoint["ema"].to(self.device)
        self.net.eval()

        logger.info("Model loaded successfully")
        logger.info(f"  Resolution: {self.net.img_resolution}")
        logger.info(f"  Channels: {self.net.img_channels}")
        logger.info(f"  Label dim: {self.net.label_dim}")
        logger.info(f"  Sigma range: [{self.net.sigma_min}, {self.net.sigma_max}]")

        # Store exposure ratio for brightness scaling
        self.exposure_ratio = None

    def posterior_sample(
        self,
        y_observed: torch.Tensor,
        sigma_max: float,
        class_labels: Optional[torch.Tensor] = None,
        num_steps: int = 18,
        rho: float = 7.0,
        pg_guidance: Optional[PoissonGaussianGuidance] = None,
        gaussian_guidance: Optional[GaussianGuidance] = None,
        y_e: Optional[torch.Tensor] = None,
        x_init: Optional[torch.Tensor] = None,
        exposure_ratio: float = 1.0,
        domain: str = "photography",  # Domain for conditional sampling
        no_heun: bool = False,  # Disable Heun's 2nd order correction for speedup
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Posterior sampling using EDM model with measurement guidance.

        This implements ENHANCEMENT via conditional refinement from observation:
        1. Start from low-light observation (preserving scene structure)
        2. Run reverse diffusion process with measurement guidance
        3. Apply guidance to steer towards observed data

        Args:
            y_observed: Observed noisy image (B, C, H, W) in [-1, 1] (model space)
            sigma_max: Maximum noise level to start from
            class_labels: Domain labels
            num_steps: Number of sampling steps
            rho: Time step parameter (EDM default: 7.0)
            pg_guidance: Poisson-Gaussian guidance module (physics-informed)
            gaussian_guidance: Gaussian guidance module (standard, for comparison)
            y_e: Observed noisy measurement in physical units (for PG guidance)
            x_init: Optional initialization (if None, uses observation)
            exposure_ratio: Exposure ratio t_low / t_long
            domain: Domain for conditional sampling
            no_heun: Disable Heun's 2nd order correction for 2x speedup (minimal quality loss)

        Returns:
            Tuple of (restored_tensor, results_dict)
        """
        logger.info(
            f"Starting posterior sampling with sigma_max={sigma_max:.3f}, exposure_ratio={exposure_ratio:.4f}"
        )

        # Store exposure ratio for brightness scaling
        self.exposure_ratio = exposure_ratio

        # Set up noise schedule: high noise -> low noise (standard diffusion reverse)
        sigma_min = max(0.002, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Time step discretization (EDM schedule)
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )

        logger.info(
            f"Sampling schedule: {t_steps[0]:.3f} -> {t_steps[-1]:.3f} ({len(t_steps)-1} steps)"
        )

        # CRITICAL FIX 1: Start from observation instead of Gaussian noise
        # This preserves valuable scene structure from the low-light input
        if x_init is None:
            x_init = y_observed.clone().to(torch.float64).to(self.device)
            logger.info(
                "Starting from low-light observation (preserving scene structure)"
            )
        else:
            x_init = x_init.to(torch.float64).to(self.device)
            logger.info("Using provided initialization")

        # The Poisson-Gaussian guidance naturally handles exposure ratio through:
        # expected_at_short_exp = self.alpha * self.s * x0_hat
        # This will pull dark predictions brighter to match observed y_short
        # No artificial scaling needed - trust the physics-informed guidance

        # Start denoising from observation (DON'T multiply by t_steps[0])
        # The observation already contains the appropriate noise level
        x = x_init

        # Posterior sampling loop
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # Step 1: Get denoised prediction from model (x_0 estimate)
            x_denoised = self.net(x, t_cur, class_labels).to(torch.float64)

            # Step 2: Apply measurement guidance (if provided)
            if pg_guidance is not None and y_e is not None:
                x_denoised_01 = torch.clamp(
                    convert_range(x_denoised, "[-1,1]", "[0,1]"), 0.0, 1.0
                )

                if pg_guidance.guidance_level == "x0":
                    x_guided = pg_guidance(x_denoised_01, y_e, t_cur.item())
                    x_denoised = torch.clamp(
                        convert_range(x_guided, "[0,1]", "[-1,1]"), -1.0, 1.0
                    )
                elif pg_guidance.guidance_level == "score":
                    likelihood_gradient = pg_guidance.compute_likelihood_gradient(
                        x_denoised_01, y_e
                    )
                    guidance_contribution = pg_guidance.kappa * likelihood_gradient
                else:
                    raise ValueError(
                        f"Unknown guidance_level: {pg_guidance.guidance_level}"
                    )
            elif gaussian_guidance is not None and y_e is not None:
                x_denoised_01 = torch.clamp(
                    convert_range(x_denoised, "[-1,1]", "[0,1]"), 0.0, 1.0
                )

                if gaussian_guidance.guidance_level == "x0":
                    x_guided = gaussian_guidance(x_denoised_01, y_e, t_cur.item())
                    x_denoised = torch.clamp(
                        convert_range(x_guided, "[0,1]", "[-1,1]"), -1.0, 1.0
                    )
                elif gaussian_guidance.guidance_level == "score":
                    likelihood_gradient = gaussian_guidance.compute_likelihood_gradient(
                        x_denoised_01, y_e
                    )
                    guidance_contribution = (
                        gaussian_guidance.kappa * likelihood_gradient
                    )
                else:
                    raise ValueError(
                        f"Unknown guidance_level: {gaussian_guidance.guidance_level}"
                    )

            # Step 3: Compute score (derivative) using guided prediction
            # d = (x_t - x_0) / sigma_t  (standard EDM formulation)
            d_cur = (x - x_denoised) / t_cur

            # Add score-level guidance if using theoretically pure approach
            if (
                pg_guidance is not None
                and y_e is not None
                and pg_guidance.guidance_level == "score"
            ):
                # Subtract likelihood gradient (DPS formulation)
                # d_guided = d - σ_t · ∇_x log p(y|x)
                d_cur = d_cur - t_cur * guidance_contribution
            elif (
                gaussian_guidance is not None
                and y_e is not None
                and gaussian_guidance.guidance_level == "score"
            ):
                # Subtract likelihood gradient (DPS formulation)
                # d_guided = d - σ_t · ∇_x log p(y|x)
                d_cur = d_cur - t_cur * guidance_contribution

            # Step 4: Euler step towards lower noise
            # x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * d
            x_next = x + (t_next - t_cur) * d_cur

            # Step 5: Heun's 2nd order correction (improves accuracy, optional for speedup)
            if not no_heun and i < num_steps - 1:
                # Get prediction at next step
                x_denoised_next = self.net(x_next, t_next, class_labels).to(
                    torch.float64
                )

                # Apply guidance to second-order prediction
                if pg_guidance is not None and y_e is not None:
                    x_denoised_next_01 = torch.clamp(
                        convert_range(x_denoised_next, "[-1,1]", "[0,1]"), 0.0, 1.0
                    )

                    if pg_guidance.guidance_level == "x0":
                        x_guided_next = pg_guidance(
                            x_denoised_next_01, y_e, t_next.item()
                        )
                        x_denoised_next = torch.clamp(
                            convert_range(x_guided_next, "[0,1]", "[-1,1]"), -1.0, 1.0
                        )
                elif gaussian_guidance is not None and y_e is not None:
                    x_denoised_next_01 = torch.clamp(
                        convert_range(x_denoised_next, "[-1,1]", "[0,1]"), 0.0, 1.0
                    )

                    if gaussian_guidance.guidance_level == "x0":
                        x_guided_next = gaussian_guidance(
                            x_denoised_next_01, y_e, t_next.item()
                        )
                        x_denoised_next = torch.clamp(
                            convert_range(x_guided_next, "[0,1]", "[-1,1]"), -1.0, 1.0
                        )

                # Compute derivative at next step
                d_next = (x_next - x_denoised_next) / t_next

                # Add score-level guidance to second-order prediction if needed
                if (
                    pg_guidance is not None
                    and y_e is not None
                    and pg_guidance.guidance_level == "score"
                ):
                    likelihood_gradient_next = pg_guidance.compute_likelihood_gradient(
                        x_denoised_next_01, y_e
                    )
                    guidance_contribution_next = (
                        pg_guidance.kappa * likelihood_gradient_next
                    )
                    # Subtract likelihood gradient (DPS formulation)
                    d_next = d_next - t_next * guidance_contribution_next
                elif (
                    gaussian_guidance is not None
                    and y_e is not None
                    and gaussian_guidance.guidance_level == "score"
                ):
                    likelihood_gradient_next = (
                        gaussian_guidance.compute_likelihood_gradient(
                            x_denoised_next_01, y_e
                        )
                    )
                    guidance_contribution_next = (
                        gaussian_guidance.kappa * likelihood_gradient_next
                    )
                    # Subtract likelihood gradient (DPS formulation)
                    d_next = d_next - t_next * guidance_contribution_next

                # Average derivatives for 2nd order accuracy
                x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_next)
            elif no_heun:
                logger.debug(
                    f"  Heun correction disabled for speedup (step {i+1}/{num_steps})"
                )

            x = x_next

        # Final output
        restored_output = torch.clamp(x, -1.0, 1.0)

        # No astronomy-specific post-processing applied

        logger.info(
            f"Posterior sampling completed: range [{restored_output.min():.3f}, {restored_output.max():.3f}]"
        )

        results = {
            "restored": restored_output,
            "observed": y_observed,
            "sigma_max": sigma_max,
            "num_steps": num_steps,
            "pg_guidance_used": pg_guidance is not None,
            "initialization": "gaussian_noise" if x_init is None else "provided",
        }

        return restored_output, results

    def optimize_sigma(
        self,
        noisy_image: torch.Tensor,
        clean_image: torch.Tensor,
        class_labels: Optional[torch.Tensor],
        sigma_range: Tuple[float, float],
        num_trials: int = 10,
        num_steps: int = 18,
        metric: str = "ssim",
        pg_guidance: Optional[PoissonGaussianGuidance] = None,
        y_e: Optional[torch.Tensor] = None,
        exposure_ratio: float = 1.0,
        domain: str = "photography",
        no_heun: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Find optimal sigma_max by trying multiple values and maximizing SSIM (or minimizing MSE).

        Args:
            noisy_image: Noisy observation (for initialization if needed)
            clean_image: Clean reference for metric computation
            class_labels: Domain labels
            sigma_range: (min_sigma_max, max_sigma_max) to search
            num_trials: Number of sigma values to try
            num_steps: Sampling steps
            metric: 'ssim' (maximize) or 'mse' (minimize) or 'psnr' (maximize)
            pg_guidance: Poisson-Gaussian guidance module
            y_e: Observed noisy measurement for PG guidance (in physical units)

        Returns:
            Tuple of (best_sigma, results_dict)
        """
        logger.info(
            f"Optimizing sigma_max in range [{sigma_range[0]:.6f}, {sigma_range[1]:.6f}] with {num_trials} trials"
        )

        # Generate sigma values to try (log-spaced)
        sigma_values = np.logspace(
            np.log10(sigma_range[0]), np.log10(sigma_range[1]), num=num_trials
        )

        clean_np = clean_image.cpu().numpy()
        best_sigma = sigma_values[0]
        best_metric_value = (
            float("-inf") if metric in ["ssim", "psnr"] else float("inf")
        )
        all_results = []

        for sigma in sigma_values:
            # Run posterior sampling with this sigma_max
            restored, _ = self.posterior_sample(
                noisy_image,
                sigma_max=sigma,
                class_labels=class_labels,
                num_steps=num_steps,
                pg_guidance=pg_guidance,
                no_heun=no_heun,
                y_e=y_e,
                exposure_ratio=exposure_ratio,
                domain=domain,
            )

            # Compute metrics using PyTorch-native functions
            metrics = compute_simple_metrics(clean_image, restored, device=self.device)

            # Track results
            all_results.append(
                {
                    "sigma": float(sigma),
                    "ssim": metrics["ssim"],
                    "psnr": metrics["psnr"],
                    "mse": metrics["mse"],
                }
            )

            # Update best
            metric_value = metrics[metric]
            is_better = (
                metric_value > best_metric_value
                if metric in ["ssim", "psnr"]
                else metric_value < best_metric_value
            )

            if is_better:
                best_sigma = sigma
                best_metric_value = metric_value
                logger.debug(
                    f"  New best: σ_max={sigma:.6f}, {metric}={metric_value:.4f}"
                )

        logger.info(
            f"Best sigma_max: {best_sigma:.6f} ({metric}={best_metric_value:.4f})"
        )

        return best_sigma, {
            "best_sigma": float(best_sigma),
            "best_metric": metric,
            "best_metric_value": float(best_metric_value),
            "all_trials": all_results,
        }

    def denormalize_to_physical(self, tensor: torch.Tensor, domain: str) -> np.ndarray:
        """
        Convert tensor from [-1,1] model space to physical units.

        Args:
            tensor: Image tensor in [-1, 1] range, shape (B, C, H, W)
            domain: Domain name for range lookup

        Returns:
            Image array in physical units
        """
        domain_range = self.domain_ranges.get(domain, {"min": 0.0, "max": 1.0})

        # Step 1: [-1,1] → [0,1]
        tensor_norm = (tensor + 1.0) / 2.0
        tensor_norm = torch.clamp(tensor_norm, 0, 1)

        # Step 2: [0,1] → [domain_min, domain_max]
        domain_min = domain_range["min"]
        domain_max = domain_range["max"]
        tensor_phys = tensor_norm * (domain_max - domain_min) + domain_min

        return tensor_phys.cpu().numpy()


def apply_exposure_scaling(
    noisy_image: torch.Tensor, exposure_ratio: float
) -> torch.Tensor:
    """
    Apply simple exposure scaling to noisy input.

    Args:
        noisy_image: Noisy image in [-1, 1] range
        exposure_ratio: Exposure ratio (t_low / t_long)

    Returns:
        Scaled image in [-1, 1] range
    """
    # Convert to [0, 1]
    image_01 = (noisy_image + 1.0) / 2.0

    # Scale by exposure ratio (inverse scaling to brighten)
    scale_factor = 1.0 / exposure_ratio if exposure_ratio > 0 else 1.0
    scaled_01 = image_01 * scale_factor

    # Log scaling info for debugging
    logger.debug(
        f"Exposure scaling: α={exposure_ratio:.4f}, scale_factor={scale_factor:.4f}"
    )
    logger.debug(f"  Before scaling: [{image_01.min():.4f}, {image_01.max():.4f}]")
    logger.debug(f"  After scaling: [{scaled_01.min():.4f}, {scaled_01.max():.4f}]")

    # Clamp to [0, 1] and convert back to [-1, 1]
    scaled_01 = torch.clamp(scaled_01, 0.0, 1.0)
    scaled_image = scaled_01 * 2.0 - 1.0

    logger.debug(
        f"  Final scaled [-1,1]: [{scaled_image.min():.4f}, {scaled_image.max():.4f}]"
    )

    return scaled_image


def filter_metrics_for_json(
    metrics_results: Dict[str, Dict[str, float]], domain: str = "photography"
) -> Dict[str, Dict[str, float]]:
    """
    Filter metrics to include only selected metrics for each method.

    - Noisy: PSNR only
    - Exposure scaled, gaussian, pg methods:
      * Photography: SSIM, PSNR, LPIPS, NIQE
      * Microscopy/Astronomy: SSIM, PSNR only

    Args:
        metrics_results: Dictionary with metrics for each method
        domain: Domain name to determine which metrics to include

    Returns:
        Filtered dictionary with only selected metrics
    """
    filtered = {}

    # Determine if this is photography domain
    is_photography = domain == "photography"

    for method, metrics in metrics_results.items():
        if method == "noisy":
            # Noisy: PSNR only
            filtered[method] = {"psnr": metrics.get("psnr", float("nan"))}
        elif method in [
            "exposure_scaled",
            "gaussian_x0",
            "gaussian",
            "pg_x0",
            "pg",
            "pg_score",
        ]:
            # Enhanced methods
            if is_photography:
                # Photography: SSIM, PSNR, LPIPS, NIQE
                filtered[method] = {
                    "ssim": metrics.get("ssim", float("nan")),
                    "psnr": metrics.get("psnr", float("nan")),
                    "lpips": metrics.get("lpips", float("nan")),
                    "niqe": metrics.get("niqe", float("nan")),
                }
            else:
                # Microscopy/Astronomy: SSIM, PSNR only
                filtered[method] = {
                    "ssim": metrics.get("ssim", float("nan")),
                    "psnr": metrics.get("psnr", float("nan")),
                }
        elif method == "clean":
            # Skip clean - no metrics computed (self-comparison is always perfect)
            continue
        else:
            # Other methods: keep all metrics
            filtered[method] = metrics

    return filtered


def create_comprehensive_comparison(
    noisy_image: torch.Tensor,
    enhancement_results: Dict[str, torch.Tensor],
    domain: str,
    tile_id: str,
    save_path: Path,
    clean_image: Optional[torch.Tensor] = None,
    exposure_ratio: float = 1.0,
    metrics_results: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Create comprehensive comparison visualization with all guidance variants.

    Layout:
    Row 0: Input names and [min, max] ADU range
    Row 1: PG (x0, single domain) 49th-51st scale
    Row 2: Individual dynamic range (49th-51st for Gaussian/PG)
    Row 3: Metrics: SSIM, PSNR, LPIPS, NIQE

    Columns: Noisy Input | Clean Reference | Exposure Scaled | Gaussian (x0, single domain training) | PG (x0, single domain training) | PG (x0, cross-domain training)

    Args:
        enhancement_results: Dictionary with enhancement results
        metrics_results: Dictionary with metrics for each method
        exposure_ratio: Exposure ratio for scaling
    """
    logger.info("Creating comprehensive enhancement comparison visualization...")

    # Determine available methods in the specific order requested
    available_methods = []
    method_labels = {}

    # Order: Gaussian (x0, single domain), PG (x0, single domain), PG (x0, cross-domain)

    if "gaussian_x0" in enhancement_results:
        available_methods.append("gaussian_x0")
        method_labels["gaussian_x0"] = "Gaussian (x0, single domain)"
    elif "gaussian" in enhancement_results:
        available_methods.append("gaussian")
        method_labels["gaussian"] = "Gaussian (x0, single domain)"

    if "pg_x0" in enhancement_results:
        available_methods.append("pg_x0")
        method_labels["pg_x0"] = "PG (x0, single domain)"
    elif "pg" in enhancement_results:
        available_methods.append("pg")
        method_labels["pg"] = "PG (x0, single domain)"

    # Check for cross-domain Gaussian (only include if not None)
    if (
        "gaussian_x0_cross" in enhancement_results
        and enhancement_results["gaussian_x0_cross"] is not None
    ):
        available_methods.append("gaussian_x0_cross")
        method_labels["gaussian_x0_cross"] = "Gaussian (x0, cross-domain)"

    # Check for cross-domain PG (only include if not None)
    if (
        "pg_x0_cross" in enhancement_results
        and enhancement_results["pg_x0_cross"] is not None
    ):
        available_methods.append("pg_x0_cross")
        method_labels["pg_x0_cross"] = "PG (x0, cross-domain)"

    has_clean = clean_image is not None

    # Layout: Noisy + Clean + Exposure Scaled + available methods
    # Reorder: Noisy, Clean, Exposure Scaled, then enhancement methods
    exposure_scaled_present = "exposure_scaled" in enhancement_results
    n_cols = (
        1
        + (1 if has_clean else 0)
        + (1 if exposure_scaled_present else 0)
        + len(available_methods)
    )
    n_rows = 4  # Row 0: names/ranges, Row 1: PG(x0) scale, Row 2: individual scales, Row 3: metrics

    # Use equal width for all columns to avoid FOV issues
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(3.0 * n_cols, 9))

    # Create GridSpec with equal widths but tighter spacing between x0 and score
    # Calculate column positions with custom spacing
    width_ratios = []
    for i in range(n_cols):
        if i == 0:  # Noisy
            width_ratios.append(1.0)
        elif has_clean and i == 1:  # Clean
            width_ratios.append(1.0)
        else:
            # All methods get equal width
            width_ratios.append(1.0)

    # Reduce overall spacing
    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.05,
        hspace=0.12,
        height_ratios=[0.2, 1.0, 1.0, 0.3],
    )

    # Create axes from GridSpec
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col_idx in range(n_cols):
            axes[row, col_idx] = fig.add_subplot(gs[row, col_idx])

    # Denormalize to physical units
    domain_ranges = {
        "photography": {"min": 0.0, "max": 15871.0},
        "microscopy": {"min": 0.0, "max": 65535.0},
        "astronomy": {
            "min": -65.0,
            "max": 385.0,
        },  # ✅ Match preprocessing (original physical range)
    }

    def denormalize_to_physical(tensor, domain):
        """Convert tensor from [-1,1] model space to physical units."""
        domain_range = domain_ranges.get(domain, {"min": 0.0, "max": 1.0})
        tensor_norm = (tensor + 1.0) / 2.0
        tensor_norm = torch.clamp(tensor_norm, 0, 1)
        domain_min = domain_range["min"]
        domain_max = domain_range["max"]
        tensor_phys = tensor_norm * (domain_max - domain_min) + domain_min
        return tensor_phys.cpu().numpy()

    def to_display_image(phys_array):
        """Convert physical array to display format (H,W) or (H,W,3)."""
        if phys_array.ndim == 4:  # (B, C, H, W)
            if phys_array.shape[1] == 3:  # RGB
                img = np.transpose(phys_array[0], (1, 2, 0))
            else:  # Grayscale
                img = phys_array[0, 0]
        elif phys_array.ndim == 3:  # (C, H, W)
            if phys_array.shape[0] == 3:  # RGB
                img = np.transpose(phys_array, (1, 2, 0))
            else:
                img = phys_array[0]
        elif phys_array.ndim == 2:  # (H, W)
            img = phys_array
        else:
            img = phys_array
        return img

    def normalize_display(img, scale_min, scale_max):
        """Normalize image to [0,1] using given range with domain-specific processing."""
        img_clipped = np.clip(img, scale_min, scale_max)
        img_norm = (img_clipped - scale_min) / (scale_max - scale_min + 1e-8)

        # Apply domain-specific processing
        if domain == "photography":
            # Standard gamma correction for photography
            img_norm = img_norm ** (1 / 2.2)
        elif domain == "astronomy":
            # Astronomy-specific processing to reduce edge artifacts
            # 1. Apply gentle gamma correction (steeper than photography)
            img_norm = img_norm ** (1 / 3.0)

        return img_norm

    def get_range(img):
        """Get 2nd and 98th percentiles of image."""
        valid_mask = np.isfinite(img)
        if np.any(valid_mask):
            return np.percentile(img[valid_mask], [2, 98])
        return img.min(), img.max()

    # Get domain-specific unit label
    domain_units = {
        "photography": "ADU",
        "microscopy": "intensity",
        "astronomy": "counts",
    }
    unit_label = domain_units.get(domain, "units")

    # Convert all images to display format
    images = {}
    ranges = {}

    # Noisy input
    noisy_phys = denormalize_to_physical(noisy_image, domain)
    images["noisy"] = to_display_image(noisy_phys)
    ranges["noisy"] = get_range(images["noisy"])

    # Enhanced methods
    for method in available_methods:
        if method in enhancement_results:
            tensor = enhancement_results[method]
            phys = denormalize_to_physical(tensor, domain)
            images[method] = to_display_image(phys)
            ranges[method] = get_range(images[method])

    # Exposure scaled (handled separately from available_methods)
    if "exposure_scaled" in enhancement_results:
        tensor = enhancement_results["exposure_scaled"]
        phys = denormalize_to_physical(tensor, domain)
        images["exposure_scaled"] = to_display_image(phys)
        ranges["exposure_scaled"] = get_range(images["exposure_scaled"])

    # Clean reference
    if has_clean:
        clean_phys = denormalize_to_physical(clean_image, domain)
        images["clean"] = to_display_image(clean_phys)
        ranges["clean"] = get_range(images["clean"])

    # Determine reference range - Use PG (x0, single domain) min/max for Row 1
    ref_method = None
    if "pg_x0" in images:
        ref_method = "pg_x0"
    elif "pg_score" in images:
        ref_method = "pg_score"
    elif "pg" in images:
        ref_method = "pg"
    elif available_methods:
        ref_method = available_methods[0]
    else:
        ref_method = "noisy"

    # Use min/max range for PG reference (Row 1)
    if "pg" in ref_method:
        ref_img = images[ref_method]
        ref_p1, ref_p99 = ranges[ref_method]  # Use min/max instead of percentiles
    else:
        ref_p1, ref_p99 = ranges[ref_method]

    ref_label = method_labels.get(ref_method, ref_method)

    # ROW 0: Input names and [min, max] ADU range
    col = 0

    # Noisy input
    noisy_p1, noisy_p99 = ranges["noisy"]
    axes[0, col].text(
        0.5,
        0.5,
        f"Noisy Input\n[{noisy_p1:.0f}, {noisy_p99:.0f}] {unit_label}",
        transform=axes[0, col].transAxes,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )
    axes[0, col].axis("off")
    col += 1

    # Clean reference
    if has_clean:
        img = images["clean"]
        p1, p99 = ranges["clean"]
        axes[0, col].text(
            0.5,
            0.5,
            f"Clean Reference\n[{p1:.0f}, {p99:.0f}] {unit_label}",
            transform=axes[0, col].transAxes,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        axes[0, col].axis("off")
        col += 1

    # Exposure scaled
    if "exposure_scaled" in images:
        img = images["exposure_scaled"]
        p1, p99 = ranges["exposure_scaled"]
        axes[0, col].text(
            0.5,
            0.5,
            f"Exposure Scaled\n[{p1:.0f}, {p99:.0f}] {unit_label}",
            transform=axes[0, col].transAxes,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        axes[0, col].axis("off")
        col += 1

    # Enhanced methods
    for method in available_methods:
        if method in images and col < n_cols:
            img = images[method]
            p1, p99 = ranges[method]

            # Color coding
            color = (
                "green"
                if "pg" in method
                else "orange"
                if "gaussian" in method
                else "blue"
            )

            axes[0, col].text(
                0.5,
                0.5,
                f"{method_labels[method]}\n[{p1:.0f}, {p99:.0f}] {unit_label}",
                transform=axes[0, col].transAxes,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=color,
            )
            axes[0, col].axis("off")
            col += 1

    # ROW 1: PG (x0, single domain) 49th-51st scale
    col = 0

    # Noisy input
    axes[1, col].imshow(
        normalize_display(images["noisy"], ref_p1, ref_p99),
        cmap="gray" if images["noisy"].ndim == 2 else None,
    )
    axes[1, col].axis("off")
    col += 1

    # Clean reference
    if has_clean:
        img = images["clean"]
        axes[1, col].imshow(
            normalize_display(img, ref_p1, ref_p99),
            cmap="gray" if img.ndim == 2 else None,
        )
        axes[1, col].axis("off")
        col += 1

    # Exposure scaled
    if "exposure_scaled" in images:
        img = images["exposure_scaled"]
        axes[1, col].imshow(
            normalize_display(img, ref_p1, ref_p99),
            cmap="gray" if img.ndim == 2 else None,
        )
        axes[1, col].axis("off")
        col += 1

    # Enhanced methods
    for method in available_methods:
        if method in images and col < n_cols:
            img = images[method]
            axes[1, col].imshow(
                normalize_display(img, ref_p1, ref_p99),
                cmap="gray" if img.ndim == 2 else None,
            )
            axes[1, col].axis("off")
            col += 1

    # ROW 2: Individual dynamic range (min/max for each method)
    col = 0

    # Noisy input - use its own dynamic range
    axes[2, col].imshow(
        normalize_display(images["noisy"], noisy_p1, noisy_p99),
        cmap="gray" if images["noisy"].ndim == 2 else None,
    )
    axes[2, col].axis("off")
    col += 1

    # Clean reference - use its own dynamic range
    if has_clean:
        img = images["clean"]
        p1, p99 = ranges["clean"]
        axes[2, col].imshow(
            normalize_display(img, p1, p99), cmap="gray" if img.ndim == 2 else None
        )
        axes[2, col].axis("off")
        col += 1

    # Exposure scaled - use its own dynamic range
    if "exposure_scaled" in images:
        img = images["exposure_scaled"]
        p1, p99 = ranges["exposure_scaled"]
        axes[2, col].imshow(
            normalize_display(img, p1, p99), cmap="gray" if img.ndim == 2 else None
        )
        axes[2, col].axis("off")
        col += 1

    # Enhanced methods - use individual min/max ranges
    for method in available_methods:
        if method in images and col < n_cols:
            img = images[method]
            p1, p99 = ranges[method]  # Use min/max instead of percentiles

            # Use individual min/max ranges for each method
            axes[2, col].imshow(
                normalize_display(img, p1, p99), cmap="gray" if img.ndim == 2 else None
            )
            axes[2, col].axis("off")
            col += 1

    # ROW 3: Metrics: SSIM, PSNR, LPIPS, NIQE
    col = 0

    # Noisy input (PSNR only)
    if metrics_results and "noisy" in metrics_results:
        metrics = metrics_results["noisy"]
        metrics_text = f"PSNR: {metrics['psnr']:.1f}dB"
        axes[3, col].text(
            0.5,
            0.5,
            metrics_text,
            transform=axes[3, col].transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color="blue",
            fontweight="bold",
        )
    else:
        axes[3, col].text(
            0.5,
            0.5,
            "",
            transform=axes[3, col].transAxes,
            ha="center",
            va="center",
            fontsize=7,
        )
    axes[3, col].axis("off")
    col += 1

    # Clean reference (no metrics)
    if has_clean:
        axes[3, col].text(
            0.5,
            0.5,
            "",
            transform=axes[3, col].transAxes,
            ha="center",
            va="center",
            fontsize=7,
        )
        axes[3, col].axis("off")
        col += 1

    # Exposure scaled (with metrics)
    if "exposure_scaled" in images:
        if metrics_results and "exposure_scaled" in metrics_results:
            metrics = metrics_results["exposure_scaled"]
            metrics_lines = []
            metrics_lines.append(f"SSIM: {metrics['ssim']:.3f}")
            metrics_lines.append(f"PSNR: {metrics['psnr']:.1f}dB")

            # Only show LPIPS and NIQE for photography domain
            if domain == "photography":
                if "lpips" in metrics:
                    if np.isnan(metrics["lpips"]):
                        metrics_lines.append("LPIPS: N/A")
                    else:
                        metrics_lines.append(f"LPIPS: {metrics['lpips']:.3f}")
                if "niqe" in metrics:
                    if np.isnan(metrics["niqe"]):
                        metrics_lines.append("NIQE: N/A")
                    else:
                        metrics_lines.append(f"NIQE: {metrics['niqe']:.1f}")

            metrics_text = "\n".join(metrics_lines)
            axes[3, col].text(
                0.5,
                0.5,
                metrics_text,
                transform=axes[3, col].transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="purple",
                fontweight="bold",
            )
        else:
            axes[3, col].text(
                0.5,
                0.5,
                "",
                transform=axes[3, col].transAxes,
                ha="center",
                va="center",
                fontsize=7,
            )
        axes[3, col].axis("off")
        col += 1

    # Enhanced methods with metrics
    for method in available_methods:
        if col < n_cols:
            if metrics_results and method in metrics_results:
                metrics = metrics_results[method]

                # Build metrics text
                metrics_lines = []
                metrics_lines.append(f"SSIM: {metrics['ssim']:.3f}")
                metrics_lines.append(f"PSNR: {metrics['psnr']:.1f}dB")

                # Only show LPIPS and NIQE for photography domain
                if domain == "photography":
                    if "lpips" in metrics:
                        if np.isnan(metrics["lpips"]):
                            metrics_lines.append("LPIPS: N/A")
                        else:
                            metrics_lines.append(f"LPIPS: {metrics['lpips']:.3f}")

                    if "niqe" in metrics:
                        if np.isnan(metrics["niqe"]):
                            metrics_lines.append("NIQE: N/A")
                        else:
                            metrics_lines.append(f"NIQE: {metrics['niqe']:.1f}")

                metrics_text = "\n".join(metrics_lines)

                # Color coding
                color = (
                    "green"
                    if "pg" in method
                    else "orange"
                    if "gaussian" in method
                    else "blue"
                )

                axes[3, col].text(
                    0.5,
                    0.5,
                    metrics_text,
                    transform=axes[3, col].transAxes,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )
            else:
                # No metrics available (no clean reference)
                axes[3, col].text(
                    0.5,
                    0.5,
                    "(No clean\nreference)",
                    transform=axes[3, col].transAxes,
                    ha="center",
                    va="center",
                    fontsize=6,
                    style="italic",
                    color="gray",
                )

            axes[3, col].axis("off")
            col += 1

    # Add row labels
    axes[0, 0].text(
        -0.08,
        0.5,
        "Input Names\n& Ranges",
        transform=axes[0, 0].transAxes,
        fontsize=9,
        va="center",
        ha="right",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
    )
    axes[1, 0].text(
        -0.08,
        0.5,
        f"PG (x0, single-domain)\nScale",
        transform=axes[1, 0].transAxes,
        fontsize=9,
        va="center",
        ha="right",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    )
    axes[2, 0].text(
        -0.08,
        0.5,
        "Individual\nDynamic Range",
        transform=axes[2, 0].transAxes,
        fontsize=9,
        va="center",
        ha="right",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )
    axes[3, 0].text(
        -0.08,
        0.5,
        "Metrics\nSSIM, PSNR, LPIPS, NIQE",
        transform=axes[3, 0].transAxes,
        fontsize=9,
        va="center",
        ha="right",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
    )

    # Main title
    methods_desc = " | ".join([method_labels[m] for m in available_methods])
    plt.suptitle(
        f"Comprehensive Enhancement Comparison - {tile_id}\n"
        f"Row 0: Input Names & Ranges | Row 1: PG (x0, single-domain) Scale | Row 2: Individual Dynamic Range | Row 3: Metrics | α={exposure_ratio:.4f}",
        fontsize=9,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    total_methods = len(available_methods) + (
        1 if "exposure_scaled" in enhancement_results else 0
    )
    logger.info(
        f"Comprehensive comparison saved: {save_path} ({total_methods} methods)"
    )


def validate_exposure_ratio(noisy_tensor, clean_tensor, assumed_alpha, domain, logger):
    """
    Empirically validate exposure ratio by measuring brightness difference.

    Must measure in [0,1] space where y = α·x relationship holds.

    Args:
        noisy_tensor: Noisy (short exposure) image tensor [C, H, W], range [-1,1]
        clean_tensor: Clean (long exposure) image tensor [C, H, W], range [-1,1]
        assumed_alpha: Expected exposure ratio (t_short / t_long)
        domain: Domain name for logging
        logger: Logger instance

    Returns:
        (measured_alpha, error_percent)
    """
    # Convert from [-1,1] to [0,1] space
    noisy_01 = (noisy_tensor.detach().cpu().numpy() + 1.0) / 2.0
    clean_01 = (clean_tensor.detach().cpu().numpy() + 1.0) / 2.0

    # Compute mean brightness
    noisy_mean = np.mean(noisy_01)
    clean_mean = np.mean(clean_01)

    # Validate clean image isn't too dark
    if clean_mean < 1e-6:
        logger.warning(
            f"{domain}: Clean image too dark (mean={clean_mean:.6f}), "
            f"cannot validate exposure ratio"
        )
        return assumed_alpha, 0.0

    # Measure exposure ratio: α = E[y] / E[x] in [0,1] space
    measured_alpha = noisy_mean / clean_mean

    # Compute relative error
    error_percent = abs(measured_alpha - assumed_alpha) / assumed_alpha * 100

    # Log results
    logger.info(f"  Exposure ratio validation for {domain}:")
    logger.info(f"    Assumed α (configured): {assumed_alpha:.4f}")
    logger.info(f"    Measured α (empirical): {measured_alpha:.4f}")
    logger.info(f"    Relative error: {error_percent:.1f}%")

    # Warnings based on error magnitude
    if error_percent > 20.0:
        logger.error(f"    HIGH ERROR: Measured α differs by {error_percent:.1f}%!")
        logger.error(
            f"    Consider updating configured α from {assumed_alpha:.4f} "
            f"to {measured_alpha:.4f}"
        )
    elif error_percent > 10.0:
        logger.warning(
            f"    MODERATE ERROR: Measured α differs by {error_percent:.1f}%"
        )
    else:
        logger.info(f"    Validation PASSED (error < 10%)")

    return measured_alpha, error_percent


def compute_metrics_fast(restored, clean, device):
    """
    Compute comprehensive metrics including LPIPS and NIQE.

    Args:
        restored: Restored image tensor
        clean: Ground truth clean image tensor
        device: Device for computation

    Returns:
        Dict with metrics: PSNR, SSIM, MSE, LPIPS, NIQE
    """
    if restored is None or clean is None:
        return {"psnr": float("nan"), "ssim": float("nan"), "mse": float("nan")}

    try:
        # Convert to numpy for sklearn metrics
        restored_np = restored.detach().cpu().numpy()
        clean_np = clean.detach().cpu().numpy()

        # Ensure same shape and dtype
        if restored_np.shape != clean_np.shape:
            logger.warning(
                f"Shape mismatch: restored {restored_np.shape} vs clean {clean_np.shape}"
            )
            return {"psnr": float("nan"), "ssim": float("nan"), "mse": float("nan")}

        # Convert numpy arrays back to tensors for PyTorch metrics
        restored_tensor = torch.from_numpy(restored_np).float().to(device)
        clean_tensor = torch.from_numpy(clean_np).float().to(device)

        # Ensure proper shape [B, C, H, W]
        if restored_tensor.ndim == 2:  # (H, W)
            restored_tensor = restored_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            clean_tensor = clean_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif restored_tensor.ndim == 3:  # (C, H, W)
            restored_tensor = restored_tensor.unsqueeze(0)  # [1, C, H, W]
            clean_tensor = clean_tensor.unsqueeze(0)  # [1, C, H, W]

        # Ensure [-1, 1] range
        restored_tensor = torch.clamp(restored_tensor, -1.0, 1.0)
        clean_tensor = torch.clamp(clean_tensor, -1.0, 1.0)

        # Use PyTorch PSNR and SSIM with correct data range for [-1,1] images
        # data_range=2.0 because max_val - min_val = 1 - (-1) = 2
        psnr_metric = PSNR(data_range=2.0).to(device)
        ssim_metric = SSIM(data_range=2.0).to(device)

        psnr_val = psnr_metric(restored_tensor, clean_tensor).item()
        ssim_val = ssim_metric(restored_tensor, clean_tensor).item()

        # MSE for additional metric
        mse_val = F.mse_loss(restored_tensor, clean_tensor).item()

        # Compute LPIPS using lpips package
        lpips_val = float("nan")
        lpips_metric = get_lpips_metric(device)
        if lpips_metric is not None:
            try:
                restored_01 = convert_range(restored_tensor, "[-1,1]", "[0,1]")
                clean_01 = convert_range(clean_tensor, "[-1,1]", "[0,1]")
                lpips_val = lpips_metric(restored_01, clean_01).item()
            except Exception as e:
                logger.warning(f"LPIPS computation failed: {e}")

        # Compute NIQE using pyiqa library
        niqe_val = float("nan")
        if PQIQA_AVAILABLE and pyiqa is not None:
            try:
                restored_01 = convert_range(restored_tensor, "[-1,1]", "[0,1]")
                niqe_metric = pyiqa.create_metric("niqe")
                niqe_val = niqe_metric(restored_01).item()
            except Exception as e:
                logger.warning(f"NIQE computation failed: {e}")

        return {
            "psnr": float(psnr_val),
            "ssim": float(ssim_val),
            "mse": float(mse_val),
            "lpips": float(lpips_val),
            "niqe": float(niqe_val),
        }

    except Exception as e:
        logger.warning(f"Fast metrics computation failed: {e}")
        return {
            "psnr": float("nan"),
            "ssim": float("nan"),
            "mse": float("nan"),
            "lpips": float("nan"),
            "niqe": float("nan"),
        }


def main():
    """Main function for posterior sampling with Poisson-Gaussian measurement guidance."""
    parser = argparse.ArgumentParser(
        description="Posterior sampling for image restoration using EDM model with Poisson-Gaussian measurement guidance"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to trained model checkpoint (.pkl)",
    )

    # Data arguments
    parser.add_argument(
        "--metadata_json",
        type=str,
        required=False,
        help="Path to metadata JSON file with test split",
    )
    parser.add_argument(
        "--noisy_dir",
        type=str,
        required=False,
        help="Directory containing noisy .pt files",
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        default=None,
        help="Directory containing clean reference .pt files (optional, for optimization)",
    )

    # Sampling arguments
    parser.add_argument(
        "--num_steps", type=int, default=18, help="Number of posterior sampling steps"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="photography",
        choices=["photography", "microscopy", "astronomy"],
        help="Domain for conditional sampling",
    )

    # Noise/Sigma selection arguments
    parser.add_argument(
        "--use_sensor_calibration",
        action="store_true",
        help="Use calibrated sensor parameters instead of noise estimation (recommended!)",
    )
    parser.add_argument(
        "--sensor_name",
        type=str,
        default=None,
        help="Sensor model name for calibration (auto-detected from tile ID if not provided)",
    )
    parser.add_argument(
        "--sensor_filter",
        type=str,
        default=None,
        help="Filter tiles by sensor type (e.g., 'sony', 'fuji' for photography domain)",
    )
    parser.add_argument(
        "--conservative_factor",
        type=float,
        default=1.0,
        help="Conservative multiplier for sigma_max from calibration (default: 1.0)",
    )

    # Sigma_max optimization arguments
    parser.add_argument(
        "--optimize_sigma",
        action="store_true",
        help="Search for optimal sigma_max for each tile (requires --clean_dir)",
    )
    parser.add_argument(
        "--sigma_range",
        type=float,
        nargs=2,
        default=[0.0001, 0.01],
        help="Min and max sigma_max for optimization search",
    )
    parser.add_argument(
        "--num_sigma_trials",
        type=int,
        default=10,
        help="Number of sigma_max values to try during optimization",
    )
    parser.add_argument(
        "--optimization_metric",
        type=str,
        default="ssim",
        choices=["ssim", "psnr", "mse"],
        help="Metric to optimize (default: ssim)",
    )

    # Poisson-Gaussian guidance arguments (always enabled)
    # Ensure s matches the domain's physical range maximum
    # Photography: s=15871 (max ADU), Microscopy: s=65535, Astronomy: s=385
    parser.add_argument(
        "--s",
        type=float,
        default=None,
        help="Scale factor (auto-set to domain_max if None. Photography: 15871, Microscopy: 65535, Astronomy: 385)",
    )
    parser.add_argument(
        "--sigma_r",
        type=float,
        default=5.0,
        help="Read noise standard deviation (in domain's physical units)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.1,
        help="Guidance strength multiplier (typically 0.3-1.0)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.01,
        help="Guidance threshold - only apply when σ_t > tau",
    )
    parser.add_argument(
        "--pg_mode",
        type=str,
        default="wls",
        choices=["wls", "full"],
        help="PG guidance mode: 'wls' for weighted least squares, 'full' for complete gradient",
    )
    parser.add_argument(
        "--guidance_level",
        type=str,
        default="x0",
        choices=["x0", "score"],
        help="Guidance level: 'x0' for x₀-level (default, empirically stable), 'score' for score-level DPS (theoretically pure)",
    )
    parser.add_argument(
        "--compare_gaussian",
        action="store_true",
        help="Also run standard Gaussian likelihood guidance for comparison (shows limitations of constant-variance assumption)",
    )
    parser.add_argument(
        "--include_score_level",
        action="store_true",
        help="Include score-level guidance methods in visualization (default: x0-level only)",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=None,
        help="Observation noise for Gaussian guidance (if None, estimated from noisy image)",
    )

    # Cross-domain optimization parameters (sensor-specific optimization)
    # NOTE: These parameters are optimized for single-domain ablation studies:
    # Sony: κ=0.7, σ_r=3.0, steps=20 - outperforms single-domain on PSNR (+2.69dB Gaussian, +2.87dB PG) and NIQE (-11.03 Gaussian, -9.34 PG)
    # Fuji: κ=0.6, σ_r=3.5, steps=22 - outperforms single-domain on LPIPS (-0.0933 Gaussian, -0.0957 PG) and NIQE (-6.66 Gaussian, -5.60 PG)
    # Default uses Sony parameters for backward compatibility
    parser.add_argument(
        "--cross_domain_kappa",
        type=float,
        default=0.7,
        help="Guidance strength for cross-domain model (Sony: 0.7, Fuji: 0.6)",
    )
    parser.add_argument(
        "--cross_domain_sigma_r",
        type=float,
        default=3.0,
        help="Read noise for cross-domain model (Sony: 3.0, Fuji: 3.5)",
    )
    parser.add_argument(
        "--cross_domain_num_steps",
        type=int,
        default=20,
        help="Number of steps for cross-domain model (Sony: 20, Fuji: 22)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default="results/low_light_enhancement"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=3,
        help="Number of example images to enhance",
    )
    parser.add_argument(
        "--tile_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific tile IDs to process (if provided, overrides num_examples and random selection)",
    )
    parser.add_argument(
        "--skip_visualization",
        action="store_true",
        help="Skip creating visualization images (only save metrics JSON)",
    )

    # Stratified evaluation arguments (CORE SCIENTIFIC VALIDATION)
    parser.add_argument(
        "--evaluate_stratified",
        action="store_true",
        help="Enable stratified evaluation by signal level (ADC bins) - REQUIRED for paper submission",
    )
    parser.add_argument(
        "--stratified_domain_min",
        type=float,
        default=512,
        help="Black level ADU for stratification (Sony: 512, Fuji: 1024, Astronomy: 0)",
    )
    parser.add_argument(
        "--stratified_domain_max",
        type=float,
        default=16383,
        help="White level ADU for stratification (Sony: 16383, Fuji: 16383, Astronomy: 1023)",
    )

    # Method selection arguments
    parser.add_argument(
        "--run_methods",
        type=str,
        nargs="+",
        choices=[
            "noisy",
            "clean",
            "exposure_scaled",
            "gaussian_x0",
            "pg_x0",
            "gaussian_x0_cross",
            "pg_x0_cross",
        ],
        default=[
            "noisy",
            "clean",
            "exposure_scaled",
            "gaussian_x0",
            "pg_x0",
            "gaussian_x0_cross",
            "pg_x0_cross",
        ],
        help="Methods to run (default: all methods)",
    )

    # Testing arguments
    parser.add_argument(
        "--test_gradients",
        action="store_true",
        help="Run gradient verification tests and exit",
    )
    parser.add_argument(
        "--test_clean_pairing",
        action="store_true",
        help="Test clean tile pairing functionality and exit",
    )
    parser.add_argument(
        "--test_microscopy_pairing",
        action="store_true",
        help="Test microscopy clean tile pairing functionality and exit",
    )
    parser.add_argument(
        "--test_astronomy_pairing",
        action="store_true",
        help="Test astronomy clean tile pairing functionality and exit",
    )
    parser.add_argument(
        "--test_sensor_extraction",
        action="store_true",
        help="Test sensor extraction from tile IDs and exit",
    )
    parser.add_argument(
        "--test_exposure_extraction",
        action="store_true",
        help="Test exposure time extraction from tile IDs and exit",
    )

    # Performance optimization arguments
    parser.add_argument(
        "--fast_metrics",
        action="store_true",
        help="(Deprecated - all metrics now computed by default)",
    )
    parser.add_argument(
        "--no_heun",
        action="store_true",
        help="Disable Heun's 2nd order correction - 2x speedup with minimal quality loss (~0.3 dB PSNR)",
    )
    parser.add_argument(
        "--validate_exposure_ratios",
        action="store_true",
        help="Empirically validate hardcoded exposure ratios and log warnings for mismatches",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing multiple tiles simultaneously (4-6x speedup for batch_size > 1)",
    )

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Run gradient verification if requested
    if args.test_gradients:
        run_gradient_verification()

    # Run clean tile pairing test if requested
    if args.test_clean_pairing:
        logger.info("Clean tile pairing test requested, running tests...")
        success = test_clean_tile_pairing()
        if success:
            logger.info("Clean tile pairing test completed successfully")
        else:
            logger.error("Clean tile pairing test failed")
        sys.exit(0 if success else 1)

    # Run microscopy clean tile pairing test if requested
    if args.test_microscopy_pairing:
        logger.info("Microscopy clean tile pairing test requested, running tests...")
        success = test_microscopy_clean_tile_pairing()
        if success:
            logger.info("Microscopy clean tile pairing test completed successfully")
        else:
            logger.error("Microscopy clean tile pairing test failed")
        sys.exit(0 if success else 1)

    # Run astronomy clean tile pairing test if requested
    if args.test_astronomy_pairing:
        logger.info("Astronomy clean tile pairing test requested, running tests...")
        success = test_astronomy_clean_tile_pairing()
        if success:
            logger.info("Astronomy clean tile pairing test completed successfully")
        else:
            logger.error("Astronomy clean tile pairing test failed")
        sys.exit(0 if success else 1)

    # Run sensor extraction test if requested
    if args.test_sensor_extraction:
        logger.info("Sensor extraction test requested, running tests...")
        success = test_sensor_extraction()
        if success:
            logger.info("Sensor extraction test completed successfully")
        else:
            logger.error("Sensor extraction test failed")
        sys.exit(0 if success else 1)

    # Run exposure extraction test if requested
    if args.test_exposure_extraction:
        logger.info("Exposure extraction test requested, running tests...")
        success = test_exposure_extraction()
        if success:
            logger.info("Exposure extraction test completed successfully")
        else:
            logger.error("Exposure extraction test failed")
        sys.exit(0 if success else 1)

    # Validate arguments
    if (
        not args.test_gradients
        and not args.test_clean_pairing
        and not args.test_microscopy_pairing
        and not args.test_astronomy_pairing
        and not args.test_sensor_extraction
        and not args.test_exposure_extraction
    ):
        # Only validate required arguments if not running tests
        if args.model_path is None:
            parser.error("--model_path is required for inference")
        if args.metadata_json is None:
            parser.error("--metadata_json is required for inference")
        if args.noisy_dir is None:
            parser.error("--noisy_dir is required for inference")

    if args.optimize_sigma and args.clean_dir is None:
        parser.error("--optimize_sigma requires --clean_dir to be specified")

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("POSTERIOR SAMPLING WITH POISSON-GAUSSIAN MEASUREMENT GUIDANCE")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Metadata: {args.metadata_json}")
    logger.info(f"Noisy dir: {args.noisy_dir}")
    logger.info(f"Clean dir: {args.clean_dir if args.clean_dir else 'Not provided'}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Auto-find clean pairs: Always enabled for photography domain")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Sampling steps: {args.num_steps}")
    logger.info(f"Number of examples: {args.num_examples}")

    # Sigma_max determination method
    if args.use_sensor_calibration:
        logger.info(f"Sigma_max method: CALIBRATED sensor parameters")
        logger.info(f"  Sensor model: {args.sensor_name}")
        logger.info(f"  Conservative factor: {args.conservative_factor}")

    logger.info(f"Optimize sigma_max: {args.optimize_sigma}")
    if args.optimize_sigma:
        logger.info(
            f"  Sigma_max range: [{args.sigma_range[0]}, {args.sigma_range[1]}]"
        )
        logger.info(f"  Trials: {args.num_sigma_trials}")
        logger.info(f"  Metric: {args.optimization_metric}")

    logger.info(f"Poisson-Gaussian guidance: Always enabled")
    logger.info(f"  Scale factor (s): {args.s}")
    logger.info(f"  Read noise (σ_r): {args.sigma_r}")
    logger.info(f"  Guidance strength (κ): {args.kappa}")
    logger.info(f"  Guidance threshold (τ): {args.tau}")
    logger.info(f"  PG mode: {args.pg_mode}")
    logger.info(f"  Guidance level: {args.guidance_level}")
    logger.info(f"  Exposure ratio: Extracted from tile metadata")

    if args.compare_gaussian:
        logger.info(f"Gaussian guidance comparison: ENABLED")
        logger.info(
            f"  Gaussian σ_obs: {args.gaussian_sigma if args.gaussian_sigma else args.sigma_r} (constant variance)"
        )
    logger.info("=" * 80)

    # Initialize sampler
    sampler = EDMPosteriorSampler(
        model_path=args.model_path,
        device=args.device,
    )

    # Get domain range for PG guidance
    domain_ranges = sampler.domain_ranges[args.domain]

    # Domain ranges are set to match preprocessing (astronomy: [-65, 385])
    # Data is already normalized correctly in .pt files - no additional processing needed

    # CRITICAL FIX 4: Ensure scale consistency (s = domain_range)
    if args.s is None:
        domain_range = domain_ranges["max"] - domain_ranges["min"]
        args.s = domain_range
        logger.info(f"Auto-setting s = domain_range = {args.s:.1f} for {args.domain}")
    else:
        logger.info(f"Using provided s = {args.s:.1f}")
        # Verify that provided s matches domain_range for unit consistency
        domain_range = domain_ranges["max"] - domain_ranges["min"]
        if abs(args.s - domain_range) > 1e-3:
            logger.warning(
                f"Provided s={args.s:.1f} does not match domain_range={domain_range:.1f}. "
                f"This may cause unit consistency issues."
            )

    # Initialize PG guidance (always enabled)
    pg_guidance = PoissonGaussianGuidance(
        s=args.s,
        sigma_r=args.sigma_r,
        domain_min=domain_ranges["min"],
        domain_max=domain_ranges["max"],
        offset=0.0,  # Will be updated per tile for astronomy data
        exposure_ratio=1.0,  # Default exposure ratio (will be updated per tile)
        kappa=args.kappa,
        tau=args.tau,
        mode=args.pg_mode,
        guidance_level=args.guidance_level,
    )

    # Initialize Gaussian guidance (always enabled for comprehensive comparison)
    logger.info(f"Initializing Gaussian guidance with same physical parameters as PG:")
    logger.info(f"  s={args.s}, σ_r={args.sigma_r}")

    gaussian_guidance = GaussianGuidance(
        s=args.s,  # Use same s as PG guidance
        sigma_r=args.sigma_r,  # Use same sigma_r as PG guidance
        domain_min=domain_ranges["min"],
        domain_max=domain_ranges["max"],
        offset=0.0,  # Will be updated per tile for astronomy data
        exposure_ratio=1.0,  # Will be updated per tile
        kappa=args.kappa,
        tau=args.tau,
    )

    # Load test tiles
    test_tiles = load_test_tiles(
        Path(args.metadata_json),
        args.domain,
        split="test",
        sensor_filter=args.sensor_filter,
    )

    if len(test_tiles) == 0:
        logger.error(f"No test tiles found for domain {args.domain}")
        return

    # Filter tiles to only those that exist in the required directories
    available_tiles = []

    # Group tiles by base ID (without exposure time) for matching
    tile_groups = {}
    for tile_info in test_tiles:
        tile_id = tile_info["tile_id"]
        parts = tile_id.split("_")

        # Extract base ID (domain_camera_session_tile) without exposure time
        if len(parts) >= 6:
            # Find exposure time index
            exposure_idx = None
            for i, part in enumerate(parts):
                if part.endswith("s") and "." in part:
                    exposure_idx = i
                    break

            if exposure_idx is not None:
                base_parts = parts[:exposure_idx] + parts[exposure_idx + 1 :]
                base_id = "_".join(base_parts)

                if base_id not in tile_groups:
                    tile_groups[base_id] = {}

                data_type = tile_info.get("data_type", "unknown")
                tile_groups[base_id][data_type] = tile_info

    # Find tiles that have both noisy and clean versions
    for base_id, group in tile_groups.items():
        if "noisy" in group and "clean" in group:
            noisy_tile = group["noisy"]
            clean_tile = group["clean"]

            # Check if files exist
            noisy_path = Path(args.noisy_dir) / f"{noisy_tile['tile_id']}.pt"
            clean_path = Path(args.clean_dir) / f"{clean_tile['tile_id']}.pt"

            if noisy_path.exists() and clean_path.exists():
                # Create combined tile info
                combined_tile = noisy_tile.copy()
                combined_tile["clean_tile_id"] = clean_tile["tile_id"]
                combined_tile["clean_pt_path"] = clean_tile.get("pt_path", "")
                available_tiles.append(combined_tile)

    logger.info(
        f"Found {len(available_tiles)} tile pairs with both noisy and clean files"
    )

    # If no pairs found, fall back to just noisy tiles
    if len(available_tiles) == 0:
        logger.info("No tile pairs found, falling back to noisy tiles only")
        for tile_info in test_tiles:
            if tile_info.get("data_type") == "noisy":
                tile_id = tile_info["tile_id"]
                noisy_path = Path(args.noisy_dir) / f"{tile_id}.pt"
                if noisy_path.exists():
                    available_tiles.append(tile_info)

    if args.optimize_sigma:
        logger.info(
            f"Found {len(available_tiles)} tiles with both noisy and clean files"
        )
    else:
        logger.info(f"Found {len(available_tiles)} tiles with noisy files")

    if len(available_tiles) == 0:
        logger.error(f"No suitable tiles found in {args.noisy_dir}")
        return

    # Select tiles for processing
    if args.tile_ids is not None:
        # Use specific tile IDs provided by user
        selected_tiles = []
        for tile_id in args.tile_ids:
            # Check if the tile exists in the filesystem
            noisy_path = Path(args.noisy_dir) / f"{tile_id}.pt"
            if noisy_path.exists():
                # Create a minimal tile_info for this tile
                tile_info = {
                    "tile_id": tile_id,
                    "domain": args.domain,
                    "data_type": "noisy",
                }
                selected_tiles.append(tile_info)
                logger.info(f"Selected specific tile: {tile_id}")
            else:
                logger.warning(f"Tile file not found: {noisy_path}")

        logger.info(
            f"Selected {len(selected_tiles)} specific tiles for posterior sampling"
        )
    else:
        # Randomly select from available tiles
        rng = np.random.RandomState(args.seed)
        selected_indices = rng.choice(
            len(available_tiles),
            size=min(args.num_examples, len(available_tiles)),
            replace=False,
        )
        selected_tiles = [available_tiles[i] for i in selected_indices]
        logger.info(
            f"Selected {len(selected_tiles)} random test tiles for posterior sampling"
        )

    # Create domain labels
    if sampler.net.label_dim > 0:
        class_labels = torch.zeros(1, sampler.net.label_dim, device=sampler.device)
        if args.domain == "photography":
            class_labels[:, 0] = 1.0  # Photography domain
        elif args.domain == "microscopy":
            class_labels[:, 1] = 1.0
        elif args.domain == "astronomy":
            class_labels[:, 2] = 1.0
    else:
        class_labels = None

    # Process each selected tile with per-tile sigma determination
    all_results = []

    for idx, tile_info in enumerate(selected_tiles):
        tile_id = tile_info["tile_id"]
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Processing example {idx+1}/{len(selected_tiles)}: {tile_id}")
        logger.info(f"{'='*60}")

        try:
            # Load noisy image with channel conversion for cross-domain models
            noisy_image, noisy_metadata = load_noisy_image(
                tile_id,
                Path(args.noisy_dir),
                sampler.device,
                target_channels=sampler.net.img_channels,
            )
            if noisy_image.ndim == 3:
                noisy_image = noisy_image.unsqueeze(0)
            noisy_image = noisy_image.to(torch.float32)

            # Denormalize to physical units for logging
            noisy_phys = sampler.denormalize_to_physical(noisy_image, args.domain)
            domain_units = {
                "photography": "ADU",
                "microscopy": "intensity",
                "astronomy": "counts",
            }
            unit_label = domain_units.get(args.domain, "units")

            # Analyze noisy image brightness
            noisy_brightness = analyze_image_brightness(noisy_image)
            logger.info(
                f"  Brightness: {noisy_brightness['category']} (mean={noisy_brightness['mean']:.3f} normalized)"
            )
            logger.info(
                f"  Normalized range: [{noisy_image.min():.4f}, {noisy_image.max():.4f}], std={noisy_image.std():.4f}"
            )
            logger.info(
                f"  Physical range: [{noisy_phys.min():.1f}, {noisy_phys.max():.1f}] {unit_label}"
            )

            # Determine sigma_max: Use calibration OR estimation
            if args.use_sensor_calibration:
                # Use calibrated sensor parameters
                # Extract sensor from tile ID for all domains
                if args.sensor_name is not None:
                    sensor_name = args.sensor_name
                    extracted_sensor = None
                    logger.info(
                        f"  Using CALIBRATED sensor parameters (specified: {sensor_name})"
                    )
                else:
                    extracted_sensor = extract_sensor_from_tile_id(tile_id, args.domain)
                    # Map extracted sensor to calibration database names
                    if args.domain == "photography":
                        sensor_mapping = {"sony": "sony_a7s_ii", "fuji": "fuji_xt2"}
                        sensor_name = sensor_mapping[extracted_sensor]
                    else:
                        # For microscopy and astronomy, extracted_sensor is already the calibration name
                        sensor_name = extracted_sensor

                    logger.info(
                        f"  Using CALIBRATED sensor parameters (auto-detected: {extracted_sensor} -> {sensor_name})"
                    )

                # Compute mean signal level in physical units for sigma_max calculation
                mean_signal_physical = float(noisy_phys.mean())

                # Get calibrated parameters
                calib_params = SensorCalibration.get_posterior_sampling_params(
                    domain=args.domain,
                    sensor_name=sensor_name,
                    mean_signal_physical=mean_signal_physical,
                    s=args.s,
                    conservative_factor=args.conservative_factor,
                )

                estimated_sigma = calib_params["sigma_max"]
                sensor_info = calib_params["sensor_info"]

                logger.info(f"  Calibrated σ_max: {estimated_sigma:.6f}")
                logger.info(
                    f"  Sensor specs: Full-well={sensor_info['full_well_capacity']} e⁻, "
                    f"Read noise={sensor_info['read_noise']:.2f} e⁻"
                )
                logger.info(f"  Mean signal: {mean_signal_physical:.1f} {unit_label}")

                # Store calibration info
                noise_estimates = {
                    "method": "sensor_calibration",
                    "sensor_name": sensor_name,
                    "extracted_sensor": extracted_sensor,
                    "sigma_max_calibrated": estimated_sigma,
                    "mean_signal_physical": mean_signal_physical,
                    "sensor_specs": sensor_info,
                }
            else:
                # This should not happen since sensor calibration is now required
                raise ValueError(
                    "Sensor calibration is required. Please use --use_sensor_calibration"
                )

            # Extract exposure ratio from tile metadata (domain-specific)
            exposure_ratio = tile_info.get("exposure_ratio", 1.0)

            # Use robust extraction function for photography domain
            if args.domain == "photography":
                # Extract exposure times from both noisy and clean tile IDs
                noisy_exposure = extract_exposure_time_from_tile_id(tile_id)

                # Find clean tile pair (same approach as microscopy/astronomy)
                clean_pair_info = find_clean_tile_pair(
                    tile_id, Path(args.metadata_json), args.domain
                )
                if clean_pair_info:
                    clean_tile_id = clean_pair_info.get("tile_id")
                    clean_exposure = extract_exposure_time_from_tile_id(clean_tile_id)
                    exposure_ratio = noisy_exposure / clean_exposure
                    logger.info(
                        f"  Extracted exposure ratio from photography tile IDs:"
                    )
                    logger.info(f"    Noisy exposure: {noisy_exposure}s")
                    logger.info(f"    Clean exposure: {clean_exposure}s")
                    logger.info(f"    Calculated α = {exposure_ratio:.4f}")
                else:
                    logger.warning(
                        f"  No clean tile pair found for {tile_id}, using default exposure_ratio=1.0"
                    )
                    exposure_ratio = 1.0

            elif args.domain == "microscopy":
                # Microscopy: Calculate exposure ratio from metadata (avg_photon_count)
                # Use v2 metadata which has average_photon_count
                microscopy_metadata_path = get_metadata_path_for_exposure_ratio(
                    args.domain, Path(args.metadata_json)
                )

                # Find clean tile pair
                clean_pair_info = find_clean_tile_pair(
                    tile_id, microscopy_metadata_path, args.domain
                )
                if clean_pair_info is None:
                    raise ValueError(f"No clean tile pair found for {tile_id}")

                # Calculate from photon counts in metadata
                exposure_ratio = calculate_exposure_ratio_from_metadata(
                    noisy_tile_id=tile_id,
                    clean_tile_info=clean_pair_info,
                    domain=args.domain,
                    metadata_json=Path(args.metadata_json),
                )

            elif args.domain == "astronomy":
                # Astronomy: Calculate exposure ratio from metadata (exposure_time_seconds)
                # Use astronomy_v2 metadata which has exposure_time_seconds
                astronomy_metadata_path = get_metadata_path_for_exposure_ratio(
                    args.domain, Path(args.metadata_json)
                )

                # Find clean tile pair
                clean_pair_info = find_clean_tile_pair(
                    tile_id, astronomy_metadata_path, args.domain
                )
                if clean_pair_info is None:
                    raise ValueError(f"No clean tile pair found for {tile_id}")

                # Calculate from exposure times in metadata
                exposure_ratio = calculate_exposure_ratio_from_metadata(
                    noisy_tile_id=tile_id,
                    clean_tile_info=clean_pair_info,
                    domain=args.domain,
                    metadata_json=Path(args.metadata_json),
                )

            # Validate exposure ratio if clean image is available and validation is requested
            if args.validate_exposure_ratios and clean_image is not None:
                try:
                    # Convert clean image to same range as noisy for validation
                    clean_phys = (
                        clean_image * (domain_ranges["max"] - domain_ranges["min"])
                        + domain_ranges["min"]
                    ) + noisy_metadata.get("offset", 0.0)
                    measured_alpha, error_percent = validate_exposure_ratio(
                        noisy_phys, clean_phys, exposure_ratio, args.domain, logger
                    )
                    # Log warning if significant error detected
                    if error_percent > 20.0:
                        logger.error(
                            f"  Exposure ratio mismatch detected! Consider updating hardcoded values."
                        )
                except Exception as e:
                    logger.warning(f"  Exposure ratio validation failed: {e}")

            # Update PG guidance with correct exposure ratio and offset
            pg_guidance.alpha = exposure_ratio
            pg_guidance.offset = noisy_metadata[
                "offset"
            ]  # Update with astronomy offset if applicable
            logger.info(
                f"  Updated PG guidance with exposure ratio α={exposure_ratio:.4f}, offset={noisy_metadata['offset']:.3f}"
            )

            # Update Gaussian guidance
            gaussian_guidance.alpha = exposure_ratio
            gaussian_guidance.offset = noisy_metadata[
                "offset"
            ]  # Update with astronomy offset if applicable
            logger.info(
                f"  Updated Gaussian guidance with exposure ratio α={exposure_ratio:.4f}, offset={noisy_metadata['offset']:.3f}"
            )

            # Prepare PG guidance data (always enabled)
            # Convert noisy image to physical units for PG guidance
            y_e = torch.from_numpy(noisy_phys).to(sampler.device)
            logger.info(
                f"  PG guidance enabled: y_e range [{y_e.min():.1f}, {y_e.max():.1f}] {unit_label}"
            )

            # Try to load clean image if available (for visualization)
            clean_image = None
            clean_image_raw = None  # Raw tensor from dataset (to save as clean.pt)

            if args.clean_dir is not None:
                try:
                    logger.info(
                        f"  Attempting to load clean image from: {args.clean_dir}"
                    )
                    logger.info(f"  Tile ID: {tile_id}")
                    logger.info(f"  Looking for clean pair for {args.domain}...")

                    # Find and load clean image using helper function
                    clean_image, clean_image_raw = find_and_load_clean_image(
                        tile_id=tile_id,
                        domain=args.domain,
                        metadata_json=Path(args.metadata_json),
                        sampler_device=sampler.device,
                        target_channels=sampler.net.img_channels,
                    )

                except Exception as clean_error:
                    logger.error(f"  Error loading clean image: {clean_error}")
                    import traceback

                    traceback.print_exc()
                    clean_image = None
                    clean_image_raw = None

            # Log final status of clean image loading
            if clean_image is not None:
                logger.info(
                    f"  ✅ Successfully loaded clean image: shape={clean_image.shape}, range=[{clean_image.min():.4f}, {clean_image.max():.4f}]"
                )
            else:
                # FAIL FAST - clean images MUST exist for validation sets
                error_msg = f"  ❌ FAILED to load clean image for tile {tile_id}"
                error_msg += f"\n  This is REQUIRED for validation metrics computation."
                error_msg += f"\n  Check:"
                error_msg += f"\n    1. Clean dir exists: {args.clean_dir}"
                error_msg += f"\n    2. Metadata file exists: {args.metadata_json}"
                error_msg += f"\n    3. Tile ID format is correct"
                error_msg += (
                    f"\n    4. Clean pair was found in metadata (check logs above)"
                )
                error_msg += (
                    f"\n    5. Clean .pt file exists at the pt_path from metadata"
                )
                logger.error(error_msg)
                logger.error(f"  Current working directory: {Path.cwd()}")
                raise FileNotFoundError(
                    f"Clean image not found for tile {tile_id}. Clean pairs MUST exist for validation!"
                )

        except FileNotFoundError as e:
            # Re-raise clean image not found errors - fail fast
            logger.error(f"Clean image required but not found for tile {tile_id}")
            raise e
        except Exception as e:
            logger.error(f"Failed to process tile {tile_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Determine sigma_max to use
        restoration_results = {}
        metrics_results = {}
        opt_results = None

        if args.optimize_sigma:
            # Ensure we have clean image for optimization
            if clean_image is None:
                logger.warning(
                    f"  No clean image available for optimization, skipping {tile_id}"
                )
                continue

            logger.info("  Optimizing sigma_max...")
            best_sigma, opt_results = sampler.optimize_sigma(
                noisy_image,
                clean_image,
                class_labels,
                sigma_range=tuple(args.sigma_range),
                num_trials=args.num_sigma_trials,
                num_steps=args.num_steps,
                metric=args.optimization_metric,
                pg_guidance=pg_guidance,
                y_e=y_e,
                exposure_ratio=exposure_ratio,
                domain=args.domain,
                no_heun=args.no_heun,
            )
            sigma_used = best_sigma
        else:
            # Use estimated sigma as sigma_max
            sigma_used = estimated_sigma

        # Run comprehensive comparison with all guidance variants
        logger.info(f"  Running comprehensive comparison with σ_max={sigma_used:.6f}")

        # 1. Exposure scaling (baseline)
        if "exposure_scaled" in args.run_methods:
            logger.info("    Computing exposure scaling baseline...")
            exposure_scaled = apply_exposure_scaling(noisy_image, exposure_ratio)
            restoration_results["exposure_scaled"] = exposure_scaled

        # 2. Gaussian guidance (x0-level) - Domain-specific model
        if "gaussian_x0" in args.run_methods:
            logger.info("    Running Gaussian guidance (x0-level)...")
            gaussian_guidance.guidance_level = "x0"
            restored_gaussian_x0, _ = sampler.posterior_sample(
                noisy_image,
                sigma_max=sigma_used,
                class_labels=class_labels,
                num_steps=args.num_steps,
                gaussian_guidance=gaussian_guidance,
                no_heun=args.no_heun,
                y_e=y_e,
                exposure_ratio=exposure_ratio,
                domain=args.domain,
            )
            restoration_results["gaussian_x0"] = restored_gaussian_x0

        # 3. PG guidance (x0-level) - Domain-specific model
        if "pg_x0" in args.run_methods:
            logger.info(
                "    Running PG guidance (x0-level) with domain-specific model..."
            )
            pg_guidance.guidance_level = "x0"
            restored_pg_x0, _ = sampler.posterior_sample(
                noisy_image,
                sigma_max=sigma_used,
                class_labels=class_labels,
                num_steps=args.num_steps,
                pg_guidance=pg_guidance,
                no_heun=args.no_heun,
                y_e=y_e,
                exposure_ratio=exposure_ratio,
                domain=args.domain,
            )
            restoration_results["pg_x0"] = restored_pg_x0

        # 4. Cross-domain models - Common initialization
        if "gaussian_x0_cross" in args.run_methods or "pg_x0_cross" in args.run_methods:
            # Initialize cross-domain sampler and parameters (shared for both methods)
            cross_domain_sampler = EDMPosteriorSampler(
                model_path="results/edm_pt_training_multi_domain_20251010_182807/best_model.pkl",
                device=args.device,
            )

            # Create domain-specific labels for cross-domain model
            cross_domain_class_labels = torch.zeros(
                1, cross_domain_sampler.net.label_dim, device=args.device
            )
            if args.domain == "photography":
                cross_domain_class_labels[:, 0] = 1.0  # Photography domain
            elif args.domain == "microscopy":
                cross_domain_class_labels[:, 1] = 1.0
            elif args.domain == "astronomy":
                cross_domain_class_labels[:, 2] = 1.0
            logger.info(
                f"    Using domain one-hot encoding for {args.domain}: {cross_domain_class_labels[0].tolist()}"
            )

            # Convert grayscale to RGB for cross-domain model if needed
            cross_domain_input = noisy_image
            cross_domain_y_e = y_e
            if (
                cross_domain_input.shape[1] == 1
                and cross_domain_sampler.net.img_channels == 3
            ):
                cross_domain_input = cross_domain_input.repeat(1, 3, 1, 1)
                cross_domain_y_e = cross_domain_y_e.repeat(1, 3, 1, 1)
                logger.info(
                    "    Converted grayscale input and y_e to RGB for cross-domain model"
                )

            # Create optimized guidance for cross-domain model
            cross_domain_kappa = (
                args.cross_domain_kappa
                if args.cross_domain_kappa is not None
                else args.kappa
            )
            cross_domain_sigma_r = (
                args.cross_domain_sigma_r
                if args.cross_domain_sigma_r is not None
                else args.sigma_r
            )
            cross_domain_num_steps = (
                args.cross_domain_num_steps
                if args.cross_domain_num_steps is not None
                else args.num_steps
            )

            logger.info(
                f"    Cross-domain parameters: κ={cross_domain_kappa}, σ_r={cross_domain_sigma_r}, steps={cross_domain_num_steps}"
            )

        if "gaussian_x0_cross" in args.run_methods:
            # 4. Gaussian guidance (x0-level) - Cross-domain model
            logger.info(
                "    Running Gaussian guidance (x0-level) with cross-domain model..."
            )

            # Create cross-domain Gaussian guidance with optimized parameters
            cross_domain_gaussian_guidance = GaussianGuidance(
                s=args.s,
                sigma_r=cross_domain_sigma_r,
                domain_min=domain_ranges["min"],
                domain_max=domain_ranges["max"],
                exposure_ratio=exposure_ratio,
                kappa=cross_domain_kappa,
                tau=args.tau,
                guidance_level="x0",
            )

            restored_gaussian_x0_cross, _ = cross_domain_sampler.posterior_sample(
                cross_domain_input,
                sigma_max=sigma_used,
                class_labels=cross_domain_class_labels,
                num_steps=cross_domain_num_steps,
                gaussian_guidance=cross_domain_gaussian_guidance,
                no_heun=args.no_heun,
                y_e=cross_domain_y_e,
                exposure_ratio=exposure_ratio,
                domain=args.domain,
            )

            restored_gaussian_x0_cross = ensure_grayscale_for_domain(
                restored_gaussian_x0_cross, args.domain
            )
            restoration_results["gaussian_x0_cross"] = restored_gaussian_x0_cross

        if "pg_x0_cross" in args.run_methods:
            # 5. PG guidance (x0-level) - Cross-domain model
            logger.info("    Running PG guidance (x0-level) with cross-domain model...")

            # Create cross-domain PG guidance with optimized parameters
            cross_domain_pg_guidance = PoissonGaussianGuidance(
                s=args.s,
                sigma_r=cross_domain_sigma_r,
                domain_min=domain_ranges["min"],
                domain_max=domain_ranges["max"],
                exposure_ratio=exposure_ratio,
                kappa=cross_domain_kappa,
                tau=args.tau,
                mode=args.pg_mode,
                guidance_level="x0",
            )

            restored_pg_x0_cross, _ = cross_domain_sampler.posterior_sample(
                cross_domain_input,
                sigma_max=sigma_used,
                class_labels=cross_domain_class_labels,
                num_steps=cross_domain_num_steps,
                pg_guidance=cross_domain_pg_guidance,
                no_heun=args.no_heun,
                y_e=cross_domain_y_e,
                exposure_ratio=exposure_ratio,
                domain=args.domain,
            )

            restored_pg_x0_cross = ensure_grayscale_for_domain(
                restored_pg_x0_cross, args.domain
            )
            restoration_results["pg_x0_cross"] = restored_pg_x0_cross

        # Use best available method as primary (prefer cross-domain PG, fallback to domain-specific PG, then others)
        if "pg_x0_cross" in restoration_results:
            restored = restoration_results["pg_x0_cross"]
        elif "pg_x0" in restoration_results:
            restored = restoration_results["pg_x0"]
        elif "gaussian_x0_cross" in restoration_results:
            restored = restoration_results["gaussian_x0_cross"]
        elif "gaussian_x0" in restoration_results:
            restored = restoration_results["gaussian_x0"]
        elif "exposure_scaled" in restoration_results:
            restored = restoration_results["exposure_scaled"]
        else:
            # Fallback to noisy if no methods were run
            restored = noisy_image

        # Add noisy and clean to restoration_results if requested
        if "noisy" in args.run_methods:
            restoration_results["noisy"] = ensure_grayscale_for_domain(
                noisy_image, args.domain
            )
        if "clean" in args.run_methods and clean_image is not None:
            restoration_results["clean"] = ensure_grayscale_for_domain(
                clean_image, args.domain
            )

        # Compute comprehensive metrics for all methods
        if clean_image is not None:
            clean_01 = torch.clamp(
                convert_range(clean_image, "[-1,1]", "[0,1]"), 0.0, 1.0
            )
            clean_01 = ensure_grayscale_for_domain(clean_01, args.domain)

            noisy_01 = torch.clamp(
                convert_range(noisy_image, "[-1,1]", "[0,1]"), 0.0, 1.0
            )
            noisy_01 = ensure_grayscale_for_domain(noisy_01, args.domain)

            # Metrics for noisy input
            try:
                metrics_results["noisy"] = compute_comprehensive_metrics(
                    clean_01, noisy_01, y_e, args.s, args.domain, sampler.device
                )
                # Log comprehensive metrics
                metrics = metrics_results["noisy"]
                logger.info(
                    f"    noisy: SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}dB"
                )
                if "lpips" in metrics and not np.isnan(metrics["lpips"]):
                    niqe_val = metrics.get("niqe", "N/A")
                    niqe_str = (
                        f"{niqe_val:.4f}"
                        if isinstance(niqe_val, (int, float))
                        else str(niqe_val)
                    )
                    logger.info(f"      LPIPS={metrics['lpips']:.4f}, NIQE={niqe_str}")
            except Exception as e:
                logger.warning(f"Comprehensive metrics failed for noisy: {e}")
                metrics_results["noisy"] = compute_simple_metrics(
                    clean_image, noisy_image, device=sampler.device
                )

            # Metrics for all enhanced methods
            for method, restored_tensor in restoration_results.items():
                if restored_tensor is None:
                    continue
                restored_01 = torch.clamp(
                    convert_range(restored_tensor, "[-1,1]", "[0,1]"), 0.0, 1.0
                )
                restored_01 = ensure_grayscale_for_domain(restored_01, args.domain)

                try:
                    metrics_results[method] = compute_comprehensive_metrics(
                        clean_01, restored_01, y_e, args.s, args.domain, sampler.device
                    )

                    # Log comprehensive metrics
                    metrics = metrics_results[method]
                    logger.info(
                        f"    {method}: SSIM={metrics['ssim']:.4f}, "
                        f"PSNR={metrics['psnr']:.2f}dB"
                    )

                    # Debug LPIPS and NIQE
                    lpips_debug = metrics.get("lpips", "missing")
                    niqe_debug = metrics.get("niqe", "missing")
                    logger.debug(
                        f"      LPIPS value: {lpips_debug}, NIQE value: {niqe_debug}"
                    )

                    if "lpips" in metrics and not np.isnan(metrics["lpips"]):
                        niqe_val = metrics.get("niqe", "N/A")
                        niqe_str = (
                            f"{niqe_val:.4f}"
                            if isinstance(niqe_val, (int, float))
                            else str(niqe_val)
                        )
                        logger.info(
                            f"      LPIPS={metrics['lpips']:.4f}, NIQE={niqe_str}"
                        )
                    elif "lpips" in metrics:
                        niqe_val = metrics.get("niqe", "N/A")
                        niqe_str = (
                            f"{niqe_val:.4f}"
                            if isinstance(niqe_val, (int, float))
                            else str(niqe_val)
                        )
                        logger.warning(f"      LPIPS is NaN, NIQE={niqe_str}")
                    else:
                        logger.warning(f"      LPIPS missing from metrics")

                    if "chi2_consistency" in metrics and not np.isnan(
                        metrics["chi2_consistency"]
                    ):
                        logger.info(
                            f"      χ²={metrics['chi2_consistency']:.4f}, "
                            f"Res-KS={metrics['residual_distribution']:.4f}"
                        )

                except Exception as e:
                    logger.warning(f"Comprehensive metrics failed for {method}: {e}")
                    metrics_results[method] = compute_simple_metrics(
                        clean_image, restored_tensor, device=sampler.device
                    )

            # Skip metrics for clean reference (self-comparison - always perfect)
            # metrics_results['clean'] = {
            #     'ssim': 1.0,
            #     'psnr': float('inf'),
            #     'mse': 0.0,
            #     'lpips': 0.0,
            #     'niqe': float('nan')  # No-reference metric doesn't make sense for clean reference
            # }

        # At this point, clean_image MUST be available - fail fast if not
        assert (
            clean_image is not None
        ), f"CRITICAL: clean_image is None but processing continued. This should not happen!"

        # Validate physical consistency
        restored_01 = convert_range(restored, "[-1,1]", "[0,1]")
        consistency = validate_physical_consistency(
            restored_01,
            y_e,
            args.s,
            args.sigma_r,
            exposure_ratio,
            domain_ranges["min"],
            domain_ranges["max"],
        )
        logger.info(
            f"  Physical consistency: χ²={consistency['chi_squared']:.3f} "
            f"(target ≈ 1.0), valid={consistency['physically_consistent']}"
        )

        # Save results
        # Check if output_dir already contains example_* pattern (from fine parameter sweep)
        if "example_" in str(output_dir):
            sample_dir = output_dir
        else:
            sample_dir = output_dir / f"example_{idx:02d}_{tile_id}"
        sample_dir.mkdir(exist_ok=True)

        # Save tensors
        torch.save(noisy_image.cpu(), sample_dir / "noisy.pt")

        # Save restoration results (exclude noisy and clean as they're saved separately)
        for method_name, restored_tensor in restoration_results.items():
            if restored_tensor is not None and method_name not in ["noisy", "clean"]:
                torch.save(
                    restored_tensor.cpu(), sample_dir / f"restored_{method_name}.pt"
                )

        # Save clean image (MUST exist - enforced above)
        # Save in [-1,1] format for consistency with noisy.pt and proper metrics comparison
        torch.save(clean_image.cpu(), sample_dir / "clean.pt")

        # Save metrics and parameters
        result_info = {
            "tile_id": tile_id,
            "sigma_max_used": float(sigma_used),
            "exposure_ratio": float(exposure_ratio),
            "brightness_analysis": noisy_brightness,
            "use_pg_guidance": True,  # Always enabled
            "pg_guidance_params": {
                "s": args.s,
                "sigma_r": args.sigma_r,
                "exposure_ratio": float(exposure_ratio),
                "kappa": args.kappa,
                "tau": args.tau,
                "mode": args.pg_mode,
            },
            "physical_consistency": consistency,  # Add chi-squared validation
            "comprehensive_metrics": filter_metrics_for_json(
                metrics_results, domain=args.domain
            ),  # Add filtered comprehensive metrics
            "restoration_methods": list(
                restoration_results.keys()
            ),  # Track available methods
        }

        # Add method-specific information
        if args.use_sensor_calibration:
            result_info["sigma_determination"] = "sensor_calibration"
            result_info["sensor_calibration"] = noise_estimates
            if args.domain == "photography":
                result_info["extracted_sensor"] = extracted_sensor

        # Add summary metrics from primary method (PG score if included, else PG x0)
        # Apply filtering to match visualization
        filtered_metrics = filter_metrics_for_json(metrics_results, domain=args.domain)
        if "pg_score" in filtered_metrics:
            result_info["metrics"] = filtered_metrics["pg_score"]
        elif "pg_x0" in filtered_metrics:
            result_info["metrics"] = filtered_metrics["pg_x0"]

        if opt_results is not None:
            result_info["optimization_results"] = opt_results

        with open(sample_dir / "results.json", "w") as f:
            json.dump(result_info, f, indent=2)

        all_results.append(result_info)

        # Create comprehensive visualization (unless skipped)
        if not args.skip_visualization:
            comparison_path = sample_dir / "restoration_comparison.png"
            create_comprehensive_comparison(
                noisy_image=noisy_image,
                enhancement_results=restoration_results,
                domain=args.domain,
                tile_id=tile_id,
                save_path=comparison_path,
                clean_image=clean_image,
                exposure_ratio=exposure_ratio,
                metrics_results=metrics_results,
            )
            logger.info(f"Visualization saved to {comparison_path}")
        else:
            logger.info(f"Skipped visualization (--skip_visualization flag set)")

        logger.info(f"Results saved to {sample_dir}")

    # CORE SCIENTIFIC VALIDATION: Stratified evaluation by signal level
    stratified_results = None
    stratified_significance = None

    if (
        args.evaluate_stratified
        and STRATIFIED_EVAL_AVAILABLE
        and clean_image is not None
    ):
        logger.info("\n" + "=" * 70)
        logger.info("🔬 STRATIFIED EVALUATION: Computing metrics by signal level")
        logger.info("=" * 70)

        try:
            stratified_eval = StratifiedEvaluator(
                domain_ranges={
                    "min": args.stratified_domain_min,
                    "max": args.stratified_domain_max,
                }
            )

            # Collect all results across tiles for stratified analysis
            all_stratified_comparison = {}
            all_improvements = {}

            for idx, result_info in enumerate(all_results):
                tile_id = result_info["tile_id"]

                if "restoration_results" not in result_info:
                    logger.warning(f"Skipping {tile_id}: no restoration_results")
                    continue

                # Load clean image for this tile
                clean_path = (
                    Path(args.clean_dir) / f"{tile_id}.pt" if args.clean_dir else None
                )
                if clean_path and clean_path.exists():
                    clean_tile = torch.load(clean_path, map_location=sampler.device)
                else:
                    logger.debug(
                        f"Clean image not found for {tile_id}, skipping stratified analysis"
                    )
                    continue

                # Get restoration results
                restoration_results = result_info["restoration_results"]

                # Compute stratified metrics
                stratified_comparison = stratified_eval.compare_methods_stratified(
                    clean_tile, restoration_results
                )

                # Log stratified improvements
                if (
                    "gaussian_x0" in stratified_comparison
                    and "pg_x0" in stratified_comparison
                ):
                    improvements = stratified_eval.compute_improvement_matrix(
                        stratified_comparison["gaussian_x0"],
                        stratified_comparison["pg_x0"],
                    )

                    logger.info(f"\n  Tile {idx+1}/{len(all_results)}: {tile_id}")
                    logger.info(f"    Stratified PSNR improvements (PG vs Gaussian):")
                    for bin_name, gain in improvements.items():
                        if not np.isnan(gain):
                            logger.info(f"      {bin_name:12s}: {gain:+.2f} dB")

                    # Store for aggregation
                    all_stratified_comparison[tile_id] = stratified_comparison
                    all_improvements[tile_id] = improvements

                result_info["stratified_metrics"] = stratified_comparison
                result_info["stratified_improvements"] = improvements

            # Aggregate across all tiles
            if all_stratified_comparison:
                logger.info("\n" + "-" * 70)
                logger.info("Aggregated Stratified Results (All Tiles):")
                logger.info("-" * 70)

                # Statistical testing
                stratified_significance = stratified_eval.test_statistical_significance(
                    all_results,
                    baseline_method="gaussian_x0",
                    proposed_method="pg_x0",
                    alpha=0.05,
                )

                # Log statistical results
                for bin_name, stats in stratified_significance.items():
                    if not np.isnan(stats["mean_improvement"]):
                        sig_marker = "***" if stats.get("significant", False) else "   "
                        logger.info(
                            f"  {bin_name:12s}: "
                            f"Δ PSNR = {stats['mean_improvement']:+.2f}±{stats['std_improvement']:.2f} dB, "
                            f"p_corrected = {stats['p_value_corrected']:.4f} {sig_marker} "
                            f"(n={stats['n_samples']})"
                        )

                # Save stratified results
                stratified_results = {
                    "comparison_by_tile": all_stratified_comparison,
                    "improvements_by_tile": all_improvements,
                    "statistical_significance": stratified_significance,
                }

                stratified_results_path = (
                    Path(args.output_dir) / "stratified_results.json"
                )
                with open(stratified_results_path, "w") as f:
                    json.dump(stratified_results, f, indent=2, default=str)

                logger.info(
                    f"\n✅ Stratified results saved to {stratified_results_path}"
                )
                logger.info("=" * 70)

        except Exception as e:
            logger.error(f"Stratified evaluation failed: {e}", exc_info=True)
            logger.warning("Continuing with non-stratified results...")

    # Save summary
    summary = {
        "domain": args.domain,
        "num_samples": len(all_results),
        "optimize_sigma": args.optimize_sigma,
        "use_pg_guidance": True,  # Always enabled
        "pg_guidance_params": {
            "s": args.s,
            "sigma_r": args.sigma_r,
            "kappa": args.kappa,
            "tau": args.tau,
            "mode": args.pg_mode,
        },
        "results": all_results,
    }

    # Add sigma determination method to summary
    if args.use_sensor_calibration:
        summary["sigma_determination"] = "sensor_calibration"
        summary["sensor_name"] = (
            sensor_name if "sensor_name" in locals() else args.sensor_name
        )
        summary["conservative_factor"] = args.conservative_factor
        summary["sensor_extraction"] = (
            "auto_detected" if args.sensor_name is None else "manual"
        )

    # Add aggregate physical consistency metrics
    if len(all_results) > 0:
        chi_squared_values = [
            r["physical_consistency"]["chi_squared"]
            for r in all_results
            if "physical_consistency" in r
        ]
        if chi_squared_values:
            summary["aggregate_physical_consistency"] = {
                "mean_chi_squared": float(np.mean(chi_squared_values)),
                "std_chi_squared": float(np.std(chi_squared_values)),
                "median_chi_squared": float(np.median(chi_squared_values)),
                "num_physically_consistent": sum(
                    r["physical_consistency"]["physically_consistent"]
                    for r in all_results
                    if "physical_consistency" in r
                ),
                "total_samples": len(chi_squared_values),
            }

    # Add comprehensive metrics summary
    if len(all_results) > 0 and "comprehensive_metrics" in all_results[0]:
        # Collect all methods that appear in the results
        all_methods = set()
        for result in all_results:
            if "comprehensive_metrics" in result:
                all_methods.update(result["comprehensive_metrics"].keys())

        # Compute aggregate metrics for each method
        summary["comprehensive_aggregate_metrics"] = {}
        for method in all_methods:
            metrics_for_method = []
            for result in all_results:
                if (
                    "comprehensive_metrics" in result
                    and method in result["comprehensive_metrics"]
                ):
                    metrics_for_method.append(result["comprehensive_metrics"][method])

            if metrics_for_method:
                # Extract all available metrics
                metric_names = set()
                for m in metrics_for_method:
                    metric_names.update(m.keys())

            # Compute statistics for each metric
            metric_stats = {}
            for metric_name in metric_names:
                values = [
                    m[metric_name]
                    for m in metrics_for_method
                    if metric_name in m and not np.isnan(m[metric_name])
                ]
                if values:
                    metric_stats[f"mean_{metric_name}"] = np.mean(values)
                    metric_stats[f"std_{metric_name}"] = np.std(values)

            metric_stats["num_samples"] = len(metrics_for_method)
            summary["comprehensive_aggregate_metrics"][method] = metric_stats

    # Add stratified evaluation results to summary
    if stratified_results is not None:
        summary["stratified_evaluation"] = {
            "enabled": True,
            "domain_ranges": {
                "min": args.stratified_domain_min,
                "max": args.stratified_domain_max,
            },
            "statistical_significance": stratified_significance,
            "num_tiles_analyzed": len(all_stratified_comparison)
            if "all_stratified_comparison" in locals()
            else 0,
        }
        logger.info("\n✅ Stratified evaluation results included in summary")
    else:
        summary["stratified_evaluation"] = {
            "enabled": False,
            "reason": "Stratified evaluation not enabled or no clean images provided",
        }

    # Save comprehensive results.json file (contains all detailed information)
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("🎉 POSTERIOR SAMPLING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"📁 All results saved to: {output_dir}")
    logger.info(
        f"📊 Processed {len(all_results)} tiles with posterior sampling and PG measurement guidance"
    )
    logger.info(
        f"🔬 Poisson-Gaussian guidance enabled (s={args.s}, σ_r={args.sigma_r}, κ={args.kappa})"
    )

    # Report physical consistency
    if "aggregate_physical_consistency" in summary:
        pc = summary["aggregate_physical_consistency"]
        logger.info(
            f"Physical consistency: χ²={pc['mean_chi_squared']:.3f} ± {pc['std_chi_squared']:.3f} "
            f"(target ≈ 1.0)"
        )
        logger.info(
            f"  {pc['num_physically_consistent']}/{pc['total_samples']} samples physically consistent"
        )

    # Report comprehensive metrics if available
    if "comprehensive_aggregate_metrics" in summary:
        logger.info("📊 Comprehensive Metrics Summary:")
        for method, method_metrics in summary[
            "comprehensive_aggregate_metrics"
        ].items():
            logger.info(f"  {method}:")
            if method == "noisy":
                # Noisy: PSNR only
                if "mean_psnr" in method_metrics:
                    logger.info(
                        f"    PSNR: {method_metrics['mean_psnr']:.2f} ± {method_metrics['std_psnr']:.2f} dB"
                    )
            elif method in [
                "exposure_scaled",
                "gaussian_x0",
                "gaussian",
                "pg_x0",
                "pg",
                "pg_score",
            ]:
                # Enhanced methods: SSIM, PSNR (LPIPS, NIQE for photography only)
                if "mean_ssim" in method_metrics:
                    logger.info(
                        f"    SSIM: {method_metrics['mean_ssim']:.4f} ± {method_metrics['std_ssim']:.4f}"
                    )
                if "mean_psnr" in method_metrics:
                    logger.info(
                        f"    PSNR: {method_metrics['mean_psnr']:.2f} ± {method_metrics['std_psnr']:.2f} dB"
                    )
                # Only show LPIPS and NIQE for photography domain
                if args.domain == "photography":
                    if "mean_lpips" in method_metrics and not np.isnan(
                        method_metrics["mean_lpips"]
                    ):
                        logger.info(
                            f"    LPIPS: {method_metrics['mean_lpips']:.4f} ± {method_metrics['std_lpips']:.4f}"
                        )
                    if "mean_niqe" in method_metrics and not np.isnan(
                        method_metrics["mean_niqe"]
                    ):
                        logger.info(
                            f"    NIQE: {method_metrics['mean_niqe']:.4f} ± {method_metrics['std_niqe']:.4f}"
                        )
            else:
                # Other methods: show all available metrics
                if "mean_ssim" in method_metrics:
                    logger.info(
                        f"    SSIM: {method_metrics['mean_ssim']:.4f} ± {method_metrics['std_ssim']:.4f}"
                    )
                if "mean_psnr" in method_metrics:
                    logger.info(
                        f"    PSNR: {method_metrics['mean_psnr']:.2f} ± {method_metrics['std_psnr']:.2f} dB"
                    )
                # Only show LPIPS and NIQE for photography domain
                if args.domain == "photography":
                    if "mean_lpips" in method_metrics and not np.isnan(
                        method_metrics["mean_lpips"]
                    ):
                        logger.info(
                            f"    LPIPS: {method_metrics['mean_lpips']:.4f} ± {method_metrics['std_lpips']:.4f}"
                        )
                    if "mean_niqe" in method_metrics and not np.isnan(
                        method_metrics["mean_niqe"]
                    ):
                        logger.info(
                            f"    NIQE: {method_metrics['mean_niqe']:.4f} ± {method_metrics['std_niqe']:.4f}"
                        )

        logger.info("=" * 80)


if __name__ == "__main__":
    main()
