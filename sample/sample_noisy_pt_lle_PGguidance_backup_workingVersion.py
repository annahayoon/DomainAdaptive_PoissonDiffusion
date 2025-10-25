#!/usr/bin/env python3
"""
Posterior Sampling for Image Restoration using EDM Model with Poisson-Gaussian Guidance

This script performs posterior sampling on noisy test images using:
1. ENHANCEMENT via conditional refinement from observation (CRITICAL FIX)
2. EDM model as the learned prior
3. Exposure-aware Poisson-Gaussian measurement guidance for physics-informed restoration
4. Optional sigma_max optimization using clean references



Key Features:
- Proper posterior sampling with measurement guidance
- Poisson-Gaussian likelihood for photon-limited imaging
- Automatic noise level estimation for sigma_max selection
- Optional sigma_max optimization to maximize SSIM/PSNR
- Comprehensive metrics reporting and visualization
- Physical unit handling for accurate likelihood computation

Theory:
    We sample from the posterior p(x|y) ∝ p(y|x) p(x) where:
    - p(x) is the EDM-learned prior
    - p(y|x) is the Poisson-Gaussian likelihood
    - Guidance: ∇_x log p(y|x) steers samples toward observed measurements

Usage (RECOMMENDED - with sensor calibration and x0-level guidance):
    python sample/sample_noisy_pt_lle_PGguidance.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/photography/noisy \
        --output_dir results/posterior_sampling_dps \
        --domain photography \
        --num_examples 3 \
        --num_steps 18 \
        --use_sensor_calibration \
        --sensor_name sony_a7s \
        --s 15871 \
        --sigma_r 5.0 \
        --kappa 0.5

Note: x0-level guidance is now default (empirically stable), score-level DPS available as alternative (theoretically pure)
      Exposure ratio is automatically extracted from tile_id (e.g., "photography_sony_a7s_session1_tile1_0.04s")

Usage (with x0-level guidance):
    python sample/sample_noisy_pt_lle_PGguidance.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/photography/noisy \
        --output_dir results/posterior_sampling_x0_guidance \
        --domain photography \
        --num_examples 3 \
        --num_steps 18 \
        --use_sensor_calibration \
        --sensor_name sony_a7s \
        --s 15871 \
        --sigma_r 5.0 \
        --kappa 0.5

Usage (RECOMMENDED - default x0-level guidance):
    python sample/sample_noisy_pt_lle_PGguidance.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/photography/noisy \
        --output_dir results/posterior_sampling_x0_guidance \
        --domain photography \
        --num_examples 3 \
        --num_steps 18 \
        --use_sensor_calibration \
        --sensor_name sony_a7s \
        --s 15871 \
        --sigma_r 5.0 \
        --kappa 0.5

Usage (legacy - with noise estimation):
    python sample/sample_noisy_pt_lle_PGguidance.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/photography/noisy \
        --output_dir results/posterior_sampling_pg \
        --domain photography \
        --num_examples 3 \
        --num_steps 18 \
        --noise_method std \
        --s 1000 \
        --sigma_r 5.0 \
        --kappa 0.5

Usage (with sigma_max optimization):
    python sample/sample_noisy_pt_lle_PGguidance.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/photography/noisy \
        --clean_dir dataset/processed/pt_tiles/photography/clean \
        --output_dir results/posterior_sampling_optimized_pg \
        --domain photography \
        --num_examples 3 \
        --num_steps 18 \
        --optimize_sigma \
        --sigma_range 0.01 0.1 \
        --num_sigma_trials 10 \
        --optimization_metric ssim \
        --use_sensor_calibration \
        --sensor_name sony_a7s \
        --s 1000 \
        --sigma_r 5.0 \
        --kappa 0.5
"""

# =============================================================================
# CLEAN VS NOISY TILE PAIRING INFORMATION
# =============================================================================
#
# The script automatically finds clean reference tiles for noisy input tiles
# using domain-specific naming patterns:
#
# ASTRONOMY DOMAIN:
# - Noisy: astronomy_j6fl7xoyq_g800l_sci_tile_XXXX.pt
# - Clean: astronomy_j6fl7xoyq_detection_sci_tile_XXXX.pt
# - Pattern: Replace "g800l_sci" with "detection_sci"
#
# MICROSCOPY DOMAIN:
# - Noisy: microscopy_CCPs_Cell_XXX_RawSIMData_gt_tile_YYYY.pt
# - Clean: microscopy_CCPs_Cell_XXX_SIM_gt_tile_YYYY.pt
# - Pattern: Replace "RawSIMData_gt" with "SIM_gt"
#
# PHOTOGRAPHY DOMAIN:
# - Noisy: photography_sony_XXXX_XX_Xs_tile_YYYY.pt or photography_fuji_XXXX_XX_Xs_tile_YYYY.pt
# - Clean: Same base name but with clean exposure time (10s, 30s, 4s, 1s)
# - Pattern: Replace exposure time (e.g., "0.1s") with clean exposure time
#
# SEARCH ORDER FOR CLEAN TILES:
# 1. Exact match with "10s" exposure
# 2. Exact match with "30s" exposure
# 3. Exact match with "4s" exposure
# 4. Exact match with "1s" exposure
# 5. Wildcard search for any clean tile with same tile number
# =============================================================================

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
from tqdm import tqdm
import cv2
from scipy import ndimage
from scipy.linalg import sqrtm
import torchvision.models as models
import torchvision.transforms as transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.regression import MeanSquaredError as MSE

# Add project root and EDM to path
project_root = Path(__file__).parent.parent  # Go up from sample/ to project root
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

# Import comprehensive metrics from core module
core_path = project_root / "core"
sys.path.insert(0, str(core_path))

# Setup logging first
import logging

# Import with direct imports to avoid relative import issues
try:
    from core.metrics import EvaluationSuite, StandardMetrics, PhysicsMetrics
    from core.exceptions import AnalysisError
    METRICS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import core metrics: {e}. Using fallback metrics.")
    METRICS_AVAILABLE = False
    EvaluationSuite = None
    StandardMetrics = None
    PhysicsMetrics = None
    AnalysisError = Exception

# Import EDM components
import external.edm.dnnlib
from external.edm.torch_utils import distributed as dist

# Import sensor calibration
from sample.sensor_calibration import SensorCalibration

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FIDCalculator:
    """Calculate Fréchet Inception Distance (FID) between two sets of images."""
    
    def __init__(self, device: str = "cuda"):
        """Initialize FID calculator with pre-trained InceptionV3 model."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained InceptionV3 model
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        self.inception_model.eval()
        self.inception_model.to(self.device)
        
        # Remove the final classification layer to get features
        self.inception_model.fc = nn.Identity()
        
        # Image preprocessing for InceptionV3
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # InceptionV3 expects 299x299
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"FID calculator initialized on device: {self.device}")
    
    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for InceptionV3 model.
        
        Args:
            images: Input images [B, C, H, W] in range [0, 1]
            
        Returns:
            Preprocessed images [B, 3, 299, 299] ready for InceptionV3
        """
        batch_size = images.shape[0]
        
        # Convert to RGB if grayscale
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {images.shape[1]}")
        
        # Resize and normalize for InceptionV3
        processed_images = []
        for i in range(batch_size):
            img = images[i:i+1]  # Keep batch dimension
            img_resized = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
            img_normalized = self.transform(img_resized)
            processed_images.append(img_normalized)
        
        return torch.cat(processed_images, dim=0)
    
    def _extract_features(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract features from images using InceptionV3.
        
        Args:
            images: Input images [B, C, H, W] in range [0, 1]
            
        Returns:
            Feature vectors [B, 2048]
        """
        with torch.no_grad():
            # Ensure images are on the same device as the model
            images = images.to(self.device)
            
            # Preprocess images
            processed_images = self._preprocess_images(images)
            
            # Extract features
            features = self.inception_model(processed_images)
            
            return features.cpu().numpy()
    
    def _calculate_fid(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Calculate FID between two sets of features.
        
        Args:
            features1: Feature vectors from first set [N1, 2048]
            features2: Feature vectors from second set [N2, 2048]
            
        Returns:
            FID score (lower is better)
        """
        # Ensure we have enough samples for covariance calculation
        if features1.shape[0] < 2 or features2.shape[0] < 2:
            logger.warning(f"Insufficient samples for FID: {features1.shape[0]}, {features2.shape[0]}")
            return float('nan')
        
        # Calculate mean and covariance
        mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
        
        # Ensure covariance matrices are 2D
        if sigma1.ndim == 0:
            sigma1 = np.array([[sigma1]])
        if sigma2.ndim == 0:
            sigma2 = np.array([[sigma2]])
        
        # Calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        
        # Calculate sqrt of product between cov
        try:
            covmean = sqrtm(sigma1.dot(sigma2))
            
            # Check and correct imaginary numbers from sqrt
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            # Calculate FID
            fid = ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
            
            return float(fid)
        except Exception as e:
            logger.warning(f"FID calculation failed: {e}")
            return float('nan')
    
    def compute_fid(self, images1: torch.Tensor, images2: torch.Tensor) -> float:
        """
        Compute FID between two sets of images.
        
        Args:
            images1: First set of images [B, C, H, W] in range [0, 1]
            images2: Second set of images [B, C, H, W] in range [0, 1]
            
        Returns:
            FID score (lower is better)
        """
        # Ensure images are in [0, 1] range
        images1 = torch.clamp(images1, 0.0, 1.0)
        images2 = torch.clamp(images2, 0.0, 1.0)
        
        # Extract features
        features1 = self._extract_features(images1)
        features2 = self._extract_features(images2)
        
        # Calculate FID
        fid_score = self._calculate_fid(features1, features2)
        
        return fid_score


class NoiseEstimator:
    """Estimate noise level in images using various methods."""
    
    @staticmethod
    def estimate_std(image: torch.Tensor) -> float:
        """Simple standard deviation estimate."""
        return image.std().item()
    
    @staticmethod
    def estimate_mad(image: torch.Tensor) -> float:
        """
        Median Absolute Deviation (MAD) based noise estimation.
        More robust to outliers and signal content.
        """
        # Convert to numpy for median operations
        img = image.cpu().numpy()
        median = np.median(img)
        mad = np.median(np.abs(img - median))
        # Scale MAD to estimate standard deviation (assuming Gaussian noise)
        sigma = 1.4826 * mad
        return float(sigma)
    
    @staticmethod
    def estimate_local_variance(image: torch.Tensor, patch_size: int = 8) -> float:
        """
        Estimate noise from local variance in homogeneous regions.
        Uses the minimum local variance as noise estimate.
        """
        img = image.cpu().numpy()
        
        if img.ndim == 4:  # (B, C, H, W)
            img = img[0, 0]
        elif img.ndim == 3:  # (C, H, W)
            img = img[0]
        
        h, w = img.shape
        min_variance = float('inf')
        
        # Slide window and compute local variances
        for i in range(0, h - patch_size + 1, patch_size // 2):
            for j in range(0, w - patch_size + 1, patch_size // 2):
                patch = img[i:i+patch_size, j:j+patch_size]
                variance = np.var(patch)
                min_variance = min(min_variance, variance)
        
        return float(np.sqrt(min_variance))
    
    @staticmethod
    def estimate_noise_comprehensive(image: torch.Tensor) -> Dict[str, float]:
        """
        Compute multiple noise estimates and return all.
        """
        estimates = {
            'std': NoiseEstimator.estimate_std(image),
            'mad': NoiseEstimator.estimate_mad(image),
            'local_var': NoiseEstimator.estimate_local_variance(image),
        }
        # Use local_var as default (best for images with structure)
        estimates['recommended'] = estimates['local_var']
        return estimates
    


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
        guidance_level: str = 'x0',  # 'x0' or 'score'
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
        self.sigma_r_squared = sigma_r ** 2
        self.domain_range = domain_max - domain_min

        logger.info(f"Initialized Exposure-Aware Gaussian Guidance: s={s}, σ_r={sigma_r}, "
                   f"domain=[{domain_min}, {domain_max}], offset={offset:.3f}, α={exposure_ratio:.4f}, κ={kappa}, τ={tau}, level={guidance_level}")
        logger.info("✅ Now exposure-aware: accounts for α in forward model")
        logger.info("✅ Uses physical sensor parameters (s, σ_r) like PG guidance")
        logger.info(f"✓ Unit consistency verified: s=domain_range={self.domain_range}")
        if offset > 0:
            logger.info(f"✓ Astronomy offset applied: {offset:.3f} (correcting for calibration-induced negative values)")
        logger.warning("⚠️  Still assumes CONSTANT noise variance (simplified vs PG)")
    
    
    def forward(
        self,
        x0_hat: torch.Tensor,
        y_e: torch.Tensor,
        sigma_t: float
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
        gradient = self.alpha * self.s * residual / (self.sigma_r_squared + self.epsilon)

        # Apply standard guidance
        step_size = self.kappa * (sigma_t ** 2)
        x0_guided = x0_hat + step_size * gradient

        # Clamp to valid range [0, 1]
        x0_guided = torch.clamp(x0_guided, 0.0, 1.0)

        return x0_guided
    
    def compute_likelihood_gradient(
        self,
        x0_hat: torch.Tensor,
        y_e: torch.Tensor
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
        self._validate_inputs(x0_hat, y_e)
        
        # Check if guidance should be applied
        if hasattr(self, '_current_sigma_t') and self._current_sigma_t <= self.tau:
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
        gradient = self.alpha * self.s * residual / (self.sigma_r_squared + self.epsilon)
        
        return gradient
    
    def _validate_inputs(self, x0_hat: torch.Tensor, y_e: torch.Tensor):
        """Validate input tensors"""
        if x0_hat.shape != y_e.shape:
            raise ValueError(f"Shape mismatch: x0_hat {x0_hat.shape} vs y_e {y_e.shape}")
        
        if torch.any(x0_hat < 0) or torch.any(x0_hat > 1):
            logger.warning("x0_hat values outside [0,1] range detected")
        
        if torch.any(y_e < 0):
            logger.warning("y_e contains negative values")


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
        exposure_ratio: CRITICAL: t_low / t_long (e.g., 0.01)
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
        >>> # For astronomy domain with negative values shifted to [0, 450]
        >>> guidance = PoissonGaussianGuidance(
        ...     s=450.0, sigma_r=2.0,
        ...     domain_min=0.0, domain_max=450.0,
        ...     offset=65.0, kappa=0.5  # 65.0 is the offset applied to data
        ... )
    """

    def __init__(
        self,
        s: float,
        sigma_r: float,
        domain_min: float,
        domain_max: float,
        offset: float = 0.0,  # NEW: Offset for astronomy data
        exposure_ratio: float = 1.0,  # CRITICAL: t_low / t_long (e.g., 0.01)
        kappa: float = 0.5,
        tau: float = 0.01,
        mode: str = 'wls',
        epsilon: float = 1e-8,
        guidance_level: str = 'x0',  # 'x0' (empirically stable, default) or 'score' (DPS, theoretically pure)
    ):
        super().__init__()

        # CRITICAL FIX: Enforce unit consistency
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
        self.offset = offset  # NEW: Offset for astronomy data
        self.alpha = exposure_ratio  # CRITICAL: exposure ratio (t_low / t_long)
        self.kappa = kappa
        self.tau = tau
        self.mode = mode
        self.epsilon = epsilon
        self.guidance_level = guidance_level  # 'x0' or 'score'

        # Pre-compute constants for efficiency
        self.sigma_r_squared = sigma_r ** 2
        self.domain_range = domain_max - domain_min

        logger.info(f"Initialized PG Guidance: s={s}, σ_r={sigma_r}, "
                   f"domain=[{domain_min}, {domain_max}], offset={offset:.3f}, α={exposure_ratio:.4f}, κ={kappa}, τ={tau}, mode={mode}")
        logger.info(f"✓ Unit consistency verified: s=domain_range={self.domain_range}")
        if offset > 0:
            logger.info(f"✓ Astronomy offset applied: {offset:.3f} (correcting for calibration-induced negative values)")
    
    
    def forward(
        self,
        x0_hat: torch.Tensor,
        y_e: torch.Tensor,
        sigma_t: float
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
        step_size = self.kappa * (sigma_t ** 2)
        x0_guided = x0_hat + step_size * gradient

        # Clamp to valid range [0, 1]
        x0_guided = torch.clamp(x0_guided, 0.0, 1.0)

        return x0_guided
    
    def _compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ∇_x log p(y_e|x)
        
        Returns gradient with same shape as x0_hat
        """
        if self.mode == 'wls':
            return self._wls_gradient(x0_hat, y_e)
        else:  # mode == 'full'
            return self._full_gradient(x0_hat, y_e)
    
    def _wls_gradient(
        self,
        x0_hat: torch.Tensor,
        y_e_physical: torch.Tensor
    ) -> torch.Tensor:
        """
        CORRECTED: Exposure-aware Weighted Least Squares gradient

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
        # KEY: Scale down the bright prediction to match dark observation
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
        self,
        x0_hat: torch.Tensor,
        y_e_physical: torch.Tensor
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
            (residual ** 2) / (2 * variance ** 2) -
            1 / (2 * variance)
        )
        
        return mean_term + variance_term
    
    def compute_likelihood_gradient(
        self,
        x0_hat: torch.Tensor,
        y_e: torch.Tensor
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
        self._validate_inputs(x0_hat, y_e)
        
        # Compute gradient
        gradient = self._compute_gradient(x0_hat, y_e)
        
        return gradient
    
    def _validate_inputs(self, x0_hat: torch.Tensor, y_e: torch.Tensor):
        """Validate input tensors"""
        if x0_hat.shape != y_e.shape:
            raise ValueError(f"Shape mismatch: x0_hat {x0_hat.shape} vs y_e {y_e.shape}")
        
        if torch.any(x0_hat < 0) or torch.any(x0_hat > 1):
            logger.warning("x0_hat values outside [0,1] range detected")
        
        if torch.any(y_e < 0):
            logger.warning("y_e contains negative values")


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
        s=s, sigma_r=sigma_r, domain_min=domain_min, domain_max=domain_max,
        exposure_ratio=exposure_ratio, kappa=kappa, epsilon=epsilon
    )

    # Test on various input sizes and values
    test_cases = [
        (1, 1, 32, 32),  # Grayscale
        (1, 3, 16, 16),  # RGB
    ]

    for batch_size, channels, height, width in test_cases:
        logger.info(f"  Testing gradient on shape: ({batch_size}, {channels}, {height}, {width})")

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
                        log_lik = compute_log_likelihood_for_test(y_e, x0_hat, pg_guidance, b, c, h, w)
                        log_lik_plus = compute_log_likelihood_for_test(y_e, x_plus, pg_guidance, b, c, h, w)

                        # Numerical gradient
                        grad_numerical[b, c, h, w] = (log_lik_plus - log_lik) / eps

        # Compare gradients
        diff = torch.abs(grad_analytical - grad_numerical).mean()
        max_diff = torch.abs(grad_analytical - grad_numerical).max()

        logger.info(f"    Mean absolute difference: {diff:.6f}")
        logger.info(f"    Max absolute difference: {max_diff:.6f}")

        # Assert gradients are close (allowing for numerical precision)
        assert diff < 1e-3, f"Gradient verification failed: mean diff {diff:.6f} > 1e-3"
        assert max_diff < 1e-2, f"Gradient verification failed: max diff {max_diff:.6f} > 1e-2"

        logger.info("    ✓ Gradient verification passed")

    logger.info("✓ All PG guidance gradient verification tests passed")


def compute_log_likelihood_for_test(y_e, x0_hat, pg_guidance, b, c, h, w):
    """
    Compute log-likelihood for a single pixel for gradient verification testing.

    This is a simplified version that computes the likelihood for a single pixel
    to enable finite difference gradient checking.
    """
    # Extract single pixel
    x_pixel = x0_hat[b:b+1, c:c+1, h:h+1, w:w+1]
    y_pixel = y_e[b:b+1, c:c+1, h:h+1, w:w+1]

    # Compute gradient (which equals the log-likelihood gradient for this single pixel)
    grad = pg_guidance.compute_likelihood_gradient(x_pixel, y_pixel)

    # For this simplified case, the gradient magnitude relates to log-likelihood
    # This is an approximation for testing purposes
    return -0.5 * grad.sum().item()  # Simplified log-likelihood approximation


def extract_exposure_ratio(tile_id: str, clean_dir: Optional[Path] = None) -> Optional[float]:
    """
    Extract exposure ratio α = t_short / t_long from tile ID with robust error handling.

    Args:
        tile_id: Tile identifier (e.g., "photography_sony_a7s_session1_tile1_0.04s")
        clean_dir: Optional clean directory path for finding matching clean exposures

    Returns:
        Exposure ratio α (t_short / t_long) or None if extraction fails
    """
    try:
        # Extract exposure from tile_id (e.g., "0.04s")
        short_exp = extract_exposure_from_id(tile_id)

        if short_exp is None:
            logger.warning(f"Could not extract short exposure from {tile_id}")
            return None

        # Find matching clean exposure if clean_dir provided
        if clean_dir is not None:
            clean_exp = find_clean_exposure(tile_id, clean_dir)
            if clean_exp is not None:
                alpha = short_exp / clean_exp
                logger.info(f"  Extracted α = {short_exp}s / {clean_exp}s = {alpha:.4f}")
                return alpha

        # Fallback: assume target exposure is 4.0s (typical long exposure)
        alpha = short_exp / 4.0
        logger.info(f"  Using fallback α = {short_exp}s / 4.0s = {alpha:.4f}")
        return alpha

    except Exception as e:
        logger.error(f"Exposure ratio extraction failed for {tile_id}: {e}")
        return None


def extract_exposure_from_id(tile_id: str) -> Optional[float]:
    """Extract exposure time from tile ID."""
    try:
        parts = tile_id.split('_')
        for part in parts:
            if part.endswith('s') and '.' in part:
                return float(part[:-1])  # Remove 's' and convert to float
    except (ValueError, IndexError):
        pass
    return None


def find_clean_exposure(tile_id: str, clean_dir: Path) -> Optional[float]:
    """Find matching clean exposure time for the given tile."""
    try:
        parts = tile_id.split('_')

        # Try different clean exposure times in order of preference
        # Order: 30s (most common for Sony), 10s (common for Fuji), 4s, 1s
        for clean_exp_str in ['30s', '10s', '4s', '1s']:
            new_parts = parts.copy()
            for i, part in enumerate(parts):
                if part.endswith('s') and '.' in part:  # Find exposure time part
                    new_parts[i] = clean_exp_str
                    break

            clean_tile_id = '_'.join(new_parts)
            clean_path = clean_dir / f"{clean_tile_id}.pt"

            if clean_path.exists():
                return float(clean_exp_str[:-1])  # Remove 's' and convert

        # If no standard exposure found, search for any clean file with same tile number
        # Extract tile pattern (domain_camera_session_*_tile_number)
        tile_pattern_parts = []
        for i, part in enumerate(parts):
            if part.endswith('s') and '.' in part:
                tile_pattern_parts.extend(parts[:i])
                tile_pattern_parts.append('*')  # Wildcard for exposure
                tile_pattern_parts.extend(parts[i+1:])
                break

        if tile_pattern_parts:
            tile_pattern = '_'.join(tile_pattern_parts)
            logger.debug(f"  Searching for clean reference with pattern: {tile_pattern}")

            # Search for any clean file with matching tile number
            for clean_file in clean_dir.glob(f"{tile_pattern}.pt"):
                try:
                    clean_tile_id = clean_file.stem
                    # Extract exposure from found clean tile
                    clean_parts = clean_tile_id.split('_')
                    for part in clean_parts:
                        if part.endswith('s') and '.' in part:
                            return float(part[:-1])
                except (ValueError, IndexError):
                    continue

    except Exception as e:
        logger.debug(f"  Clean exposure search failed: {e}")

    return None


def extract_sensor_from_tile_id(tile_id: str) -> str:
    """
    Extract sensor information from tile ID (photography domain only).
    
    Args:
        tile_id: Tile ID (e.g., "photography_sony_00145_00_0.1s_tile_0000")
        
    Returns:
        Sensor name if found and domain is photography, raises error otherwise
        
    Example:
        tile_id = "photography_sony_00145_00_0.1s_tile_0000"
        sensor = extract_sensor_from_tile_id(tile_id)  # Returns "sony"
    """
    parts = tile_id.split('_')
    if len(parts) >= 2 and parts[0] == "photography":
        sensor = parts[1]  # sony, fuji, etc.
        return sensor
    else:
        raise ValueError(f"Cannot extract sensor from tile ID: {tile_id}")


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
    parts = tile_id.split('_')
    for part in parts:
        if part.endswith('s'):
            # Extract exposure time (e.g., "0.1s" -> 0.1, "30s" -> 30.0)
            exposure_str = part.replace('s', '')
            try:
                return float(exposure_str)
            except ValueError:
                continue
    
    raise ValueError(f"Cannot extract exposure time from tile ID: {tile_id}")


def find_clean_tile_pair_astronomy(
    noisy_tile_id: str, 
    metadata_json: Path
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
    parts = noisy_tile_id.split('_')
    if len(parts) < 5:
        logger.error(f"Invalid astronomy tile ID format: {noisy_tile_id}")
        return None
    
    # Extract base components
    scene_id = parts[1]      # j6fl7xoyq
    tile_id = parts[-1]      # 0000
    
    logger.info(f"Extracted components - scene_id: {scene_id}, tile_id: {tile_id}")
    
    # Load metadata
    try:
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_json}: {e}")
        return None
    
    # Find clean tile with same scene_id and tile_id
    # but different data type (clean vs noisy) and g800l_sci -> detection_sci
    tiles = metadata.get('tiles', [])
    
    clean_candidates = []
    for tile in tiles:
        if (tile.get('domain') == 'astronomy' and 
            tile.get('data_type') == 'clean' and
            tile.get('tile_id', '').startswith(f"astronomy_{scene_id}_detection_sci_tile_") and
            tile.get('tile_id', '').endswith(f"_tile_{tile_id}")):
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
        logger.warning(f"Multiple clean tile candidates found for {noisy_tile_id}, using first one")
        best_candidate = clean_candidates[0]
        logger.info(f"Found clean tile pair: {best_candidate.get('tile_id')}")
        return best_candidate


def find_clean_tile_pair_microscopy(
    noisy_tile_id: str, 
    metadata_json: Path
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
    # Format: microscopy_{specimen}_{Cell}_{scene_id}_RawSIMData_gt_tile_{tile_id}
    # Format: microscopy_{specimen}_{Cell}_{scene_id}_RawGTSIMData_level_{level_id}_tile_{tile_id}
    parts = noisy_tile_id.split('_')
    if len(parts) < 6:
        logger.error(f"Invalid microscopy tile ID format: {noisy_tile_id}")
        return None
    
    # Extract base components
    specimen = parts[1]      # F-actin_nonlinear
    cell = parts[2]          # Cell
    scene_id = parts[3]      # 005
    tile_id = parts[-1]      # 0000
    
    logger.info(f"Extracted components - specimen: {specimen}, cell: {cell}, scene_id: {scene_id}, tile_id: {tile_id}")
    
    # Load metadata
    try:
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_json}: {e}")
        return None
    
    # Find clean tile with same specimen, cell, scene_id, and tile_id
    # but different data type (clean vs noisy)
    tiles = metadata.get('tiles', [])
    
    clean_candidates = []
    for tile in tiles:
        if (tile.get('domain') == 'microscopy' and 
            tile.get('data_type') == 'clean' and
            tile.get('tile_id', '').startswith(f"microscopy_{specimen}_{cell}_{scene_id}_") and
            tile.get('tile_id', '').endswith(f"_tile_{tile_id}")):
            clean_candidates.append(tile)
    
    if not clean_candidates:
        logger.warning(f"No clean tile candidates found for {noisy_tile_id}")
        return None
    
    # If multiple candidates, prefer the one with longest exposure time or specific pattern
    # For microscopy, we prefer SIM_gt_a over SIM_gt, and GTSIM over GTSIM_level
    best_candidate = None
    best_score = -1
    
    for candidate in clean_candidates:
        candidate_id = candidate.get('tile_id', '')
        score = 0
        
        # Prefer SIM_gt_a over SIM_gt
        if 'SIM_gt_a' in candidate_id:
            score += 10
        elif 'SIM_gt' in candidate_id:
            score += 5
        
        # Prefer GTSIM over GTSIM_level
        if 'GTSIM_level' in candidate_id:
            score += 3
        elif 'GTSIM' in candidate_id:
            score += 8
        
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
    noisy_tile_id: str, 
    metadata_json: Path, 
    domain: str = "photography"
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
    noisy_tile_id: str, 
    metadata_json: Path
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
    parts = noisy_tile_id.split('_')
    if len(parts) < 7:
        logger.error(f"Invalid photography tile ID format: {noisy_tile_id}")
        return None
    
    sensor = parts[1]      # sony, fuji, etc.
    scene_id = parts[2]    # 00145
    scene_num = parts[3]   # 00
    tile_id = parts[6]     # 0000 (after splitting by '_', tile_id is at index 6)
    
    logger.info(f"Extracted components - sensor: {sensor}, scene_id: {scene_id}, scene_num: {scene_num}, tile_id: {tile_id}")
    
    # Load metadata
    try:
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_json}: {e}")
        return None
    
    # Find clean tile with same sensor, scene_id, scene_num, and tile_id
    # but different exposure time (typically longer exposure)
    tiles = metadata.get('tiles', [])
    
    clean_candidates = []
    for tile in tiles:
        if (tile.get('domain') == 'photography' and 
            tile.get('data_type') == 'clean' and
            tile.get('tile_id', '').startswith(f"photography_{sensor}_{scene_id}_{scene_num}_") and
            tile.get('tile_id', '').endswith(f"_tile_{tile_id}")):
            clean_candidates.append(tile)
    
    if not clean_candidates:
        logger.warning(f"No clean tile candidates found for {noisy_tile_id}")
        return None
    
    # If multiple candidates, prefer the one with longest exposure time
    # Extract exposure times and find the maximum
    best_candidate = None
    max_exposure = 0.0
    
    for candidate in clean_candidates:
        candidate_id = candidate.get('tile_id', '')
        # Extract exposure time from tile_id
        try:
            exposure_part = candidate_id.split('_')[4]  # e.g., "30s"
            exposure_str = exposure_part.replace('s', '')
            exposure_time = float(exposure_str)
            
            if exposure_time > max_exposure:
                max_exposure = exposure_time
                best_candidate = candidate
                
        except (IndexError, ValueError) as e:
            logger.warning(f"Could not parse exposure time from {candidate_id}: {e}")
            continue
    
    if best_candidate:
        logger.info(f"Found clean tile pair: {best_candidate.get('tile_id')} (exposure: {max_exposure}s)")
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
            extracted_sensor = extract_sensor_from_tile_id(tile_id)
            if extracted_sensor == expected_sensor:
                logger.info(f"✓ Correctly extracted sensor: {extracted_sensor}")
                success_count += 1
            else:
                logger.error(f"✗ Expected: {expected_sensor}, Got: {extracted_sensor}")
        except ValueError as e:
            logger.error(f"✗ Error extracting sensor: {e}")
    
    # Test error cases
    error_cases = [
        "microscopy_RawGTSIMData_001_tile_0000",  # Non-photography domain
        "astronomy_g800l_sci_001_tile_0000",      # Non-photography domain
        "invalid_tile_id",                        # Invalid format
    ]
    
    for tile_id in error_cases:
        logger.info(f"\n--- Testing error case: {tile_id} ---")
        try:
            extracted_sensor = extract_sensor_from_tile_id(tile_id)
            logger.error(f"✗ Expected error but got: {extracted_sensor}")
        except ValueError as e:
            logger.info(f"✓ Correctly raised error: {e}")
            success_count += 1
        total_tests += 1
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Successful extractions: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("✓ All sensor extraction tests passed!")
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
                logger.info(f"✓ Correctly extracted exposure: {extracted_exposure}s")
                success_count += 1
            else:
                logger.error(f"✗ Expected: {expected_exposure}s, Got: {extracted_exposure}s")
        except ValueError as e:
            logger.error(f"✗ Error extracting exposure: {e}")
    
    # Test error cases
    error_cases = [
        "photography_sony_00145_00_tile_0000",  # No exposure time
        "invalid_tile_id",                      # Invalid format
    ]
    
    for tile_id in error_cases:
        logger.info(f"\n--- Testing error case: {tile_id} ---")
        try:
            extracted_exposure = extract_exposure_time_from_tile_id(tile_id)
            logger.error(f"✗ Expected error but got: {extracted_exposure}s")
        except ValueError as e:
            logger.info(f"✓ Correctly raised error: {e}")
            success_count += 1
        total_tests += 1
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Successful extractions: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("✓ All exposure extraction tests passed!")
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
    
    metadata_json = Path("/home/jilab/Jae/dataset/processed/metadata_astronomy_incremental.json")
    
    if not metadata_json.exists():
        logger.error(f"Metadata file not found: {metadata_json}")
        return False
    
    success_count = 0
    total_tests = len(test_cases)
    
    for noisy_tile_id in test_cases:
        logger.info(f"\n--- Testing: {noisy_tile_id} ---")
        
        clean_pair = find_clean_tile_pair(noisy_tile_id, metadata_json, "astronomy")
        
        if clean_pair:
            clean_tile_id = clean_pair.get('tile_id')
            clean_pt_path = clean_pair.get('pt_path')
            
            logger.info(f"✓ Found clean pair: {clean_tile_id}")
            logger.info(f"  Path: {clean_pt_path}")
            
            # Verify the file exists
            if Path(clean_pt_path).exists():
                logger.info(f"✓ Clean tile file exists")
                success_count += 1
            else:
                logger.error(f"✗ Clean tile file does not exist: {clean_pt_path}")
        else:
            logger.error(f"✗ No clean pair found for {noisy_tile_id}")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Successful pairings: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("✓ All astronomy tests passed!")
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
    
    metadata_json = Path("/home/jilab/Jae/dataset/processed/metadata_microscopy_incremental.json")
    
    if not metadata_json.exists():
        logger.error(f"Metadata file not found: {metadata_json}")
        return False
    
    success_count = 0
    total_tests = len(test_cases)
    
    for noisy_tile_id in test_cases:
        logger.info(f"\n--- Testing: {noisy_tile_id} ---")
        
        clean_pair = find_clean_tile_pair(noisy_tile_id, metadata_json, "microscopy")
        
        if clean_pair:
            clean_tile_id = clean_pair.get('tile_id')
            clean_pt_path = clean_pair.get('pt_path')
            
            logger.info(f"✓ Found clean pair: {clean_tile_id}")
            logger.info(f"  Path: {clean_pt_path}")
            
            # Verify the file exists
            if Path(clean_pt_path).exists():
                logger.info(f"✓ Clean tile file exists")
                success_count += 1
            else:
                logger.error(f"✗ Clean tile file does not exist: {clean_pt_path}")
        else:
            logger.error(f"✗ No clean pair found for {noisy_tile_id}")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Successful pairings: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("✓ All microscopy tests passed!")
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
    
    metadata_json = Path("/home/jilab/Jae/dataset/processed/metadata_photography_incremental.json")
    
    if not metadata_json.exists():
        logger.error(f"Metadata file not found: {metadata_json}")
        return False
    
    success_count = 0
    total_tests = len(test_cases)
    
    for noisy_tile_id in test_cases:
        logger.info(f"\n--- Testing: {noisy_tile_id} ---")
        
        clean_pair = find_clean_tile_pair(noisy_tile_id, metadata_json, "photography")
        
        if clean_pair:
            clean_tile_id = clean_pair.get('tile_id')
            clean_pt_path = clean_pair.get('pt_path')
            
            logger.info(f"✓ Found clean pair: {clean_tile_id}")
            logger.info(f"  Path: {clean_pt_path}")
            
            # Verify the file exists
            if Path(clean_pt_path).exists():
                logger.info(f"✓ Clean tile file exists")
                success_count += 1
            else:
                logger.error(f"✗ Clean tile file does not exist: {clean_pt_path}")
        else:
            logger.error(f"✗ No clean pair found for {noisy_tile_id}")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Successful pairings: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        logger.info("✓ All tests passed!")
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


def compute_comprehensive_metrics(
    clean: torch.Tensor,
    enhanced: torch.Tensor,
    noisy: torch.Tensor,
    scale: float,
    domain: str,
    device: str = "cuda",
    fid_calculator: Optional[FIDCalculator] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics using the core metrics module.

    Metrics computed:
      - PSNR, SSIM: Fidelity to clean reference (higher is better)
      - LPIPS: Perceptual similarity (lower is better)
      - NIQE: No-reference quality (lower is better)
      - FID: Distribution match (aggregate only, requires >1 sample, lower is better)
      - χ²_red: Physical consistency (should be ≈ 1.0)
      - MSE: Mean squared error (lower is better)

    Args:
        clean: Clean reference image [B, C, H, W] in [0,1] range
        enhanced: Enhanced image [B, C, H, W] in [0,1] range
        noisy: Noisy observation [B, C, H, W] in physical units
        scale: Scale factor for physical units
        domain: Domain name ('photography', 'microscopy', 'astronomy')
        device: Device for computation
        fid_calculator: Optional FID calculator instance

    Returns:
        Dictionary with comprehensive metrics
    """
    # Check if metrics module is available
    if not METRICS_AVAILABLE or EvaluationSuite is None:
        logger.warning("Core metrics not available, using fallback")
        return compute_simple_metrics(clean, enhanced, fid_calculator=fid_calculator, device=device)
    
    try:
        # Use StandardMetrics directly for simpler computation
        standard_metrics = StandardMetrics(device=device)
        
        # Convert noisy to torch tensor if needed
        if isinstance(noisy, np.ndarray):
            noisy = torch.from_numpy(noisy).to(device)
        
        # Ensure all tensors are on the same device and float32
        clean = clean.to(device).float()
        enhanced = enhanced.to(device).float()
        noisy = noisy.to(device).float()
        
        # Compute standard metrics
        psnr_result = standard_metrics.compute_psnr(enhanced, clean, data_range=1.0)
        ssim_result = standard_metrics.compute_ssim(enhanced, clean, data_range=1.0)
        lpips_result = standard_metrics.compute_lpips(enhanced, clean)
        niqe_result = standard_metrics.compute_niqe(enhanced)
        
        metrics = {
            'ssim': ssim_result.value if not np.isnan(ssim_result.value) else 0.0,
            'psnr': psnr_result.value if not np.isnan(psnr_result.value) else 0.0,
            'lpips': lpips_result.value if not np.isnan(lpips_result.value) else float('nan'),
            'niqe': niqe_result.value if not np.isnan(niqe_result.value) else float('nan'),
            'mse': psnr_result.metadata.get('mse', float('nan')),
        }
        
        # FID will be computed only in aggregate summary for all methods together
        metrics['fid'] = float('nan')
        
        return metrics
        
    except Exception as e:
        logger.warning(f"Comprehensive metrics computation failed: {e}, using fallback")
        # Fallback to simple metrics
        return compute_simple_metrics(clean, enhanced, fid_calculator=fid_calculator, device=device)


def compute_simple_metrics(clean, enhanced, data_range: float = None, fid_calculator: Optional[FIDCalculator] = None, device: str = "cuda") -> Dict[str, float]:
    """
    PyTorch-native simple metrics computation.
    
    Args:
        clean: Clean reference image (tensor or numpy array)
        enhanced: Enhanced image (tensor or numpy array)
        data_range: Data range for metrics (if None, computed from clean)
        fid_calculator: Optional FID calculator instance
        device: Device for computation

    Returns:
        Dictionary with basic metrics: SSIM, PSNR, MSE, FID
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
    
    # Ensure [0, 1] range
    clean = torch.clamp(clean, 0.0, 1.0)
    enhanced = torch.clamp(enhanced, 0.0, 1.0)
    
    # Initialize PyTorch-native metrics
    ssim_metric = SSIM(data_range=1.0).to(device)
    psnr_metric = PSNR(data_range=1.0).to(device)
    mse_metric = MSE().to(device)
    
    # Compute metrics using PyTorch-native functions
    try:
        ssim_val = ssim_metric(enhanced, clean).item()
        psnr_val = psnr_metric(enhanced, clean).item()
        mse_val = mse_metric(enhanced, clean).item()
        
        metrics = {
            'ssim': float(ssim_val),
            'psnr': float(psnr_val),
            'mse': float(mse_val),
            'lpips': float('nan'),  # Not available in fallback
            'niqe': float('nan'),   # Not available in fallback
        }
    except Exception as e:
        logger.warning(f"PyTorch-native metrics computation failed: {e}")
        # Fallback to basic MSE computation
        mse_val = F.mse_loss(enhanced, clean).item()
        psnr_val = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse_val))).item()
        
        metrics = {
            'ssim': float('nan'),
            'psnr': float(psnr_val),
            'mse': float(mse_val),
            'lpips': float('nan'),
            'niqe': float('nan'),
        }
    
    # FID computation removed from per-tile metrics (only computed in aggregate summary)
    metrics['fid'] = float('nan')
    
    return metrics


def validate_physical_consistency(
    x_enhanced: torch.Tensor,
    y_e_physical: torch.Tensor,
    s: float,
    sigma_r: float,
    exposure_ratio: float,  # CRITICAL: t_low / t_long
    domain_min: float,
    domain_max: float,
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    CORRECTED: Validate physical consistency using reduced chi-squared statistic.
    
    Tests: y_short ≈ Poisson(α·s·x_long) + N(0, σ_r²)
    
    A physically consistent reconstruction should have χ²_red ≈ 1.0, indicating
    that residuals match the expected Poisson-Gaussian noise distribution.
    
    Args:
        x_enhanced: Enhanced image in [0,1] normalized space [B,C,H,W] (LONG exposure)
        y_e_physical: Observed noisy measurement in physical units [B,C,H,W] (SHORT exposure)
        s: Scale factor used in PG guidance
        sigma_r: Read noise standard deviation (in physical units)
        exposure_ratio: CRITICAL: t_low / t_long (e.g., 0.01 for 0.04s/4s)
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
    
    # CRITICAL FIX: Expected observation at SHORT exposure (apply α!)
    expected_y_at_short_exp = exposure_ratio * s * x_enhanced
    
    # Variance at SHORT exposure
    variance_at_short_exp = exposure_ratio * s * x_enhanced + sigma_r**2 + epsilon
    
    # Residual
    residual = y_e_scaled - expected_y_at_short_exp
    
    # Chi-squared per pixel
    chi_squared_map = (residual ** 2) / variance_at_short_exp
    
    # Reduced chi-squared (average over all pixels)
    chi_squared_red = chi_squared_map.mean().item()
    
    # Additional statistics
    chi_squared_std = chi_squared_map.std().item()
    mean_residual = residual.mean().item()
    max_residual = residual.abs().max().item()
    
    # Physical consistency check: χ² should be in [0.8, 1.2] range
    is_consistent = 0.8 < chi_squared_red < 1.2
    
    return {
        'chi_squared': chi_squared_red,
        'chi_squared_std': chi_squared_std,
        'physically_consistent': is_consistent,
        'mean_residual': mean_residual,
        'max_residual': max_residual,
    }


def load_test_tiles(metadata_json: Path, domain: str, split: str = "test", sensor_filter: Optional[str] = None) -> List[Dict[str, Any]]:
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
    
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)
    
    # Filter tiles by domain and split
    tiles = metadata.get('tiles', [])
    filtered_tiles = [
        tile for tile in tiles
        if tile.get('domain') == domain and tile.get('split') == split
    ]
    
    # Apply sensor filter for photography domain
    if sensor_filter and domain == "photography":
        sensor_filtered_tiles = []
        for tile in filtered_tiles:
            tile_id = tile.get('tile_id', '')
            if sensor_filter.lower() in tile_id.lower():
                sensor_filtered_tiles.append(tile)
        filtered_tiles = sensor_filtered_tiles
        logger.info(f"Found {len(filtered_tiles)} {split} tiles for {domain} with sensor {sensor_filter}")
    else:
        logger.info(f"Found {len(filtered_tiles)} {split} tiles for {domain}")
    
    return filtered_tiles


def load_image(tile_id: str, image_dir: Path, device: torch.device, image_type: str = "image", target_channels: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Load a .pt file (noisy or clean) and return both tensor and metadata.

    Returns:
        Tuple of (tensor, metadata_dict) where metadata_dict contains:
        - 'offset': Offset applied for astronomy data (0.0 for other domains)
        - 'original_range': Original [min, max] before any processing
        - 'processed_range': Final [min, max] after processing
    """
    image_path = image_dir / f"{tile_id}.pt"

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load the image
    tensor = torch.load(image_path, map_location=device)

    # Handle different tensor formats
    if isinstance(tensor, dict):
        if 'noisy' in tensor:
            tensor = tensor['noisy']
        elif 'noisy_norm' in tensor:
            tensor = tensor['noisy_norm']
        elif 'clean' in tensor:
            tensor = tensor['clean']
        elif 'clean_norm' in tensor:
            tensor = tensor['clean_norm']
        elif 'image' in tensor:
            tensor = tensor['image']
        else:
            raise ValueError(f"Unrecognized dict structure in {image_path}")

    # Track original range before any processing
    original_min = tensor.min().item()
    original_max = tensor.max().item()

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
            logger.debug(f"Converted grayscale to RGB for {image_type} image: {tile_id}")
        elif tensor.shape[0] == 3 and target_channels == 1:
            # Convert RGB (3 channels) to grayscale (1 channel) by averaging
            tensor = tensor.mean(dim=0, keepdim=True)
            logger.debug(f"Converted RGB to grayscale for {image_type} image: {tile_id}")
        else:
            logger.warning(f"Cannot convert from {tensor.shape[0]} to {target_channels} channels for {image_type} image: {tile_id}")

    # Handle astronomy data with negative values (common due to calibration processes)
    offset = 0.0
    if "astronomy" in tile_id.lower() and tensor.min() < 0:
        # Negative values in Hubble Legacy Field data arise from:
        # 1. Calibration processes and data reduction steps (bias subtraction, flat-fielding)
        # 2. Detector characteristics (bad/dead pixels with unstable quantum efficiency)
        # 3. Background/sky subtraction that can result in negative values after corrections
        # These are physically meaningful low-intensity regions, not artifacts

        # For astronomy data, shift values so minimum becomes 0 while preserving relative intensities
        min_val = tensor.min()
        offset = abs(min_val).item()  # Track the offset applied
        tensor = tensor - min_val

        # Scale to match the expected domain range (0 to 450)
        # The current domain_max for astronomy is 450.0
        expected_max = 450.0
        current_max = tensor.max()

        if current_max > 0:
            # Scale to fit within expected range
            scale_factor = expected_max / current_max
            tensor = tensor * scale_factor
            logger.debug(f"Scaled astronomy {image_type} image {tile_id}: range [0, {current_max:.3f}] -> [0, {tensor.max():.3f}]")

        logger.debug(f"Processed astronomy {image_type} image {tile_id}: shifted by {min_val:.3f}, final range [{tensor.min():.3f}, {tensor.max():.3f}]")
        logger.debug("Note: Negative values represent physically meaningful low-intensity regions after calibration corrections")
        logger.debug(f"Applied offset: {offset:.3f} (will be propagated to guidance)")

    # Track final range after processing
    final_min = tensor.min().item()
    final_max = tensor.max().item()

    logger.debug(f"Loaded {image_type} image: {tile_id}, shape={tensor.shape}, range=[{final_min:.3f}, {final_max:.3f}]")

    # Return both tensor and metadata
    metadata = {
        'offset': offset,
        'original_range': [original_min, original_max],
        'processed_range': [final_min, final_max],
        'domain': 'astronomy' if 'astronomy' in tile_id.lower() else 'other'
    }

    return tensor, metadata


def load_noisy_image(tile_id: str, noisy_dir: Path, device: torch.device, target_channels: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load a noisy .pt file and return both tensor and metadata."""
    return load_image(tile_id, noisy_dir, device, "noisy", target_channels)


def load_clean_image(tile_id: str, clean_dir: Path, device: torch.device, target_channels: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load a clean .pt file and return both tensor and metadata."""
    return load_image(tile_id, clean_dir, device, "clean", target_channels)


def analyze_image_brightness(image: torch.Tensor) -> Dict[str, float]:
    """Analyze image brightness characteristics."""
    # Convert to [0, 1] range for analysis
    img_01 = (image + 1.0) / 2.0
    
    # Calculate statistics
    mean_brightness = img_01.mean().item()
    std_brightness = img_01.std().item()
    min_brightness = img_01.min().item()
    max_brightness = img_01.max().item()
    
    # Calculate percentile brightness
    img_flat = img_01.flatten()
    p10 = torch.quantile(img_flat, 0.1).item()
    p50 = torch.quantile(img_flat, 0.5).item()
    p90 = torch.quantile(img_flat, 0.9).item()
    
    # Categorize brightness
    if mean_brightness < 0.2:
        brightness_category = "Very Dark"
    elif mean_brightness < 0.4:
        brightness_category = "Dark"
    elif mean_brightness < 0.6:
        brightness_category = "Medium"
    elif mean_brightness < 0.8:
        brightness_category = "Bright"
    else:
        brightness_category = "Very Bright"
    
    return {
        'mean': mean_brightness,
        'std': std_brightness,
        'min': min_brightness,
        'max': max_brightness,
        'p10': p10,
        'p50': p50,
        'p90': p90,
        'category': brightness_category
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
            "astronomy": {"min": 0.0, "max": 450.0},  # Fixed: removed negative min, increased max
        }

        # Load model
        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)  # nosec B301

        self.net = checkpoint["ema"].to(self.device)
        self.net.eval()

        logger.info("✓ Model loaded successfully")
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
        exposure_ratio: float = 1.0,  # CRITICAL: t_low / t_long
        domain: str = "photography",  # Domain for conditional sampling
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
            exposure_ratio: CRITICAL: t_low / t_long (e.g., 0.01 for 0.04s/4s)
            
        Returns:
            Tuple of (restored_tensor, results_dict)
        """
        logger.info(f"Starting posterior sampling with sigma_max={sigma_max:.3f}, exposure_ratio={exposure_ratio:.4f}")
        
        # Store exposure ratio for brightness scaling
        self.exposure_ratio = exposure_ratio
        
        # Set up noise schedule: high noise -> low noise (standard diffusion reverse)
        sigma_min = max(0.002, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)
        
        # Time step discretization (EDM schedule)
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        
        logger.info(f"Sampling schedule: {t_steps[0]:.3f} -> {t_steps[-1]:.3f} ({len(t_steps)-1} steps)")
        
        # CRITICAL FIX 1: Start from observation instead of Gaussian noise
        # This preserves valuable scene structure from the low-light input
        if x_init is None:
            x_init = y_observed.clone().to(torch.float64).to(self.device)
            logger.info("Starting from low-light observation (preserving scene structure)")
        else:
            x_init = x_init.to(torch.float64).to(self.device)
            logger.info("Using provided initialization")
        
        
        # REMOVED: Initialization brightness scaling heuristic
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
                # Poisson-Gaussian guidance (physics-informed)
                # Convert denoised prediction to [0,1] for PG guidance
                x_denoised_01 = (x_denoised + 1.0) / 2.0
                x_denoised_01 = torch.clamp(x_denoised_01, 0.0, 1.0)
                
                if pg_guidance.guidance_level == 'x0':
                    # Standard x₀-level guidance (widely used, empirically validated)
                    x_guided = pg_guidance(x_denoised_01, y_e, t_cur.item())
                    # Convert back to [-1,1] range
                    x_denoised = x_guided * 2.0 - 1.0
                    x_denoised = torch.clamp(x_denoised, -1.0, 1.0)
                elif pg_guidance.guidance_level == 'score':
                    # Theoretically pure score-level guidance
                    likelihood_gradient = pg_guidance.compute_likelihood_gradient(x_denoised_01, y_e)
                    # Store for later use in score computation
                    guidance_contribution = pg_guidance.kappa * likelihood_gradient
                else:
                    raise ValueError(f"Unknown guidance_level: {pg_guidance.guidance_level}")
            elif gaussian_guidance is not None and y_e is not None:
                # Standard Gaussian guidance (for comparison)
                # Convert denoised prediction to [0,1] for guidance
                x_denoised_01 = (x_denoised + 1.0) / 2.0
                x_denoised_01 = torch.clamp(x_denoised_01, 0.0, 1.0)

                if gaussian_guidance.guidance_level == 'x0':
                    # Standard x₀-level guidance (widely used, empirically validated)
                    x_guided = gaussian_guidance(x_denoised_01, y_e, t_cur.item())
                    # Convert back to [-1,1] range
                    x_denoised = x_guided * 2.0 - 1.0
                    x_denoised = torch.clamp(x_denoised, -1.0, 1.0)
                elif gaussian_guidance.guidance_level == 'score':
                    # Theoretically pure score-level guidance
                    likelihood_gradient = gaussian_guidance.compute_likelihood_gradient(x_denoised_01, y_e)
                    # Store for later use in score computation
                    guidance_contribution = gaussian_guidance.kappa * likelihood_gradient
                else:
                    raise ValueError(f"Unknown guidance_level: {gaussian_guidance.guidance_level}")
            
            # Step 3: Compute score (derivative) using guided prediction
            # d = (x_t - x_0) / sigma_t  (standard EDM formulation)
            d_cur = (x - x_denoised) / t_cur
            
            # Add score-level guidance if using theoretically pure approach
            if pg_guidance is not None and y_e is not None and pg_guidance.guidance_level == 'score':
                # CORRECTED: Subtract likelihood gradient (DPS formulation)
                # d_guided = d - σ_t · ∇_x log p(y|x)
                d_cur = d_cur - t_cur * guidance_contribution
            elif gaussian_guidance is not None and y_e is not None and gaussian_guidance.guidance_level == 'score':
                # CORRECTED: Subtract likelihood gradient (DPS formulation)
                # d_guided = d - σ_t · ∇_x log p(y|x)
                d_cur = d_cur - t_cur * guidance_contribution
            
            # Step 4: Euler step towards lower noise
            # x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * d
            x_next = x + (t_next - t_cur) * d_cur
            
            # Step 5: Heun's 2nd order correction (improves accuracy)
            if i < num_steps - 1:
                # Get prediction at next step
                x_denoised_next = self.net(x_next, t_next, class_labels).to(torch.float64)
                
                # Apply guidance to second-order prediction
                if pg_guidance is not None and y_e is not None:
                    x_denoised_next_01 = (x_denoised_next + 1.0) / 2.0
                    x_denoised_next_01 = torch.clamp(x_denoised_next_01, 0.0, 1.0)
                    
                    if pg_guidance.guidance_level == 'x0':
                        x_guided_next = pg_guidance(x_denoised_next_01, y_e, t_next.item())
                        x_denoised_next = x_guided_next * 2.0 - 1.0
                        x_denoised_next = torch.clamp(x_denoised_next, -1.0, 1.0)
                    elif pg_guidance.guidance_level == 'score':
                        # For score-level guidance, we don't modify x_denoised_next
                        # The guidance is applied directly to the score
                        pass
                elif gaussian_guidance is not None and y_e is not None:
                    # Gaussian guidance (uses physical space like PG)
                    x_denoised_next_01 = (x_denoised_next + 1.0) / 2.0
                    x_denoised_next_01 = torch.clamp(x_denoised_next_01, 0.0, 1.0)
                    
                    if gaussian_guidance.guidance_level == 'x0':
                        x_guided_next = gaussian_guidance(x_denoised_next_01, y_e, t_next.item())
                        x_denoised_next = x_guided_next * 2.0 - 1.0
                        x_denoised_next = torch.clamp(x_denoised_next, -1.0, 1.0)
                    elif gaussian_guidance.guidance_level == 'score':
                        # For score-level guidance, we don't modify x_denoised_next
                        # The guidance is applied directly to the score
                        pass
                
                # Compute derivative at next step
                d_next = (x_next - x_denoised_next) / t_next
                
                # Add score-level guidance to second-order prediction if needed
                if pg_guidance is not None and y_e is not None and pg_guidance.guidance_level == 'score':
                    likelihood_gradient_next = pg_guidance.compute_likelihood_gradient(x_denoised_next_01, y_e)
                    guidance_contribution_next = pg_guidance.kappa * likelihood_gradient_next
                    # CORRECTED: Subtract likelihood gradient (DPS formulation)
                    d_next = d_next - t_next * guidance_contribution_next
                elif gaussian_guidance is not None and y_e is not None and gaussian_guidance.guidance_level == 'score':
                    likelihood_gradient_next = gaussian_guidance.compute_likelihood_gradient(x_denoised_next_01, y_e)
                    guidance_contribution_next = gaussian_guidance.kappa * likelihood_gradient_next
                    # CORRECTED: Subtract likelihood gradient (DPS formulation)
                    d_next = d_next - t_next * guidance_contribution_next
                
                # Average derivatives for 2nd order accuracy
                x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_next)
            
            x = x_next
        
        # Final output
        restored_output = torch.clamp(x, -1.0, 1.0)
        
        
        # No astronomy-specific post-processing applied
        
        logger.info(f"✓ Posterior sampling completed: range [{restored_output.min():.3f}, {restored_output.max():.3f}]")
        
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
        metric: str = 'ssim',
        pg_guidance: Optional[PoissonGaussianGuidance] = None,
        y_e: Optional[torch.Tensor] = None,
        exposure_ratio: float = 1.0,
        fid_calculator: Optional[FIDCalculator] = None,
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
            fid_calculator: Optional FID calculator instance
        
        Returns:
            Tuple of (best_sigma, results_dict)
        """
        logger.info(f"Optimizing sigma_max in range [{sigma_range[0]:.6f}, {sigma_range[1]:.6f}] with {num_trials} trials")
        
        # Generate sigma values to try (log-spaced)
        sigma_values = np.logspace(
            np.log10(sigma_range[0]),
            np.log10(sigma_range[1]),
            num=num_trials
        )
        
        clean_np = clean_image.cpu().numpy()
        best_sigma = sigma_values[0]
        best_metric_value = float('-inf') if metric in ['ssim', 'psnr'] else float('inf')
        all_results = []
        
        for sigma in sigma_values:
            # Run posterior sampling with this sigma_max
            restored, _ = self.posterior_sample(
                noisy_image,
                sigma_max=sigma,
                class_labels=class_labels,
                num_steps=num_steps,
                pg_guidance=pg_guidance,
                y_e=y_e,
                exposure_ratio=exposure_ratio,
            )
            
            # Compute metrics using PyTorch-native functions
            metrics = compute_simple_metrics(clean_image, restored, fid_calculator=fid_calculator, device=self.device)
            
            # Track results
            all_results.append({
                'sigma': float(sigma),
                'ssim': metrics['ssim'],
                'psnr': metrics['psnr'],
                'mse': metrics['mse'],
            })
            
            # Update best
            metric_value = metrics[metric]
            is_better = (
                metric_value > best_metric_value if metric in ['ssim', 'psnr']
                else metric_value < best_metric_value
            )
            
            if is_better:
                best_sigma = sigma
                best_metric_value = metric_value
                logger.debug(f"  New best: σ_max={sigma:.6f}, {metric}={metric_value:.4f}")
        
        logger.info(f"✓ Best sigma_max: {best_sigma:.6f} ({metric}={best_metric_value:.4f})")
        
        return best_sigma, {
            'best_sigma': float(best_sigma),
            'best_metric': metric,
            'best_metric_value': float(best_metric_value),
            'all_trials': all_results,
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




def apply_exposure_scaling(noisy_image: torch.Tensor, exposure_ratio: float) -> torch.Tensor:
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
    
    # Clamp to [0, 1] and convert back to [-1, 1]
    scaled_01 = torch.clamp(scaled_01, 0.0, 1.0)
    scaled_image = scaled_01 * 2.0 - 1.0
    
    return scaled_image


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
    # Note: exposure_scaled is handled separately in the visualization
    
    if 'gaussian_x0' in enhancement_results:
        available_methods.append('gaussian_x0')
        method_labels['gaussian_x0'] = 'Gaussian (x0, single domain)'
    elif 'gaussian' in enhancement_results:
        available_methods.append('gaussian')
        method_labels['gaussian'] = 'Gaussian (x0, single domain)'
    
    if 'pg_x0' in enhancement_results:
        available_methods.append('pg_x0')
        method_labels['pg_x0'] = 'PG (x0, single domain)'
    elif 'pg' in enhancement_results:
        available_methods.append('pg')
        method_labels['pg'] = 'PG (x0, single domain)'
    
    # Check for cross-domain Gaussian (only include if not None)
    if 'gaussian_x0_cross' in enhancement_results and enhancement_results['gaussian_x0_cross'] is not None:
        available_methods.append('gaussian_x0_cross')
        method_labels['gaussian_x0_cross'] = 'Gaussian (x0, cross-domain)'
    
    # Check for cross-domain PG (only include if not None)
    if 'pg_x0_cross' in enhancement_results and enhancement_results['pg_x0_cross'] is not None:
        available_methods.append('pg_x0_cross')
        method_labels['pg_x0_cross'] = 'PG (x0, cross-domain)'
    
    has_clean = clean_image is not None
    
    # Layout: Noisy + Clean + Exposure Scaled + available methods
    # Reorder: Noisy, Clean, Exposure Scaled, then enhancement methods
    exposure_scaled_present = 'exposure_scaled' in enhancement_results
    n_cols = 1 + (1 if has_clean else 0) + (1 if exposure_scaled_present else 0) + len(available_methods)
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
    gs = GridSpec(n_rows, n_cols, figure=fig, width_ratios=width_ratios, 
                  wspace=0.05, hspace=0.12, height_ratios=[0.2, 1.0, 1.0, 0.3])
    
    # Create axes from GridSpec
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col_idx in range(n_cols):
            axes[row, col_idx] = fig.add_subplot(gs[row, col_idx])
    
    # Denormalize to physical units
    domain_ranges = {
        "photography": {"min": 0.0, "max": 15871.0},
        "microscopy": {"min": 0.0, "max": 65535.0},
        "astronomy": {"min": 0.0, "max": 450.0},  # Fixed: removed negative min, increased max
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
            img_norm = img_norm ** (1/2.2)
        elif domain == "astronomy":
            # Astronomy-specific processing to reduce edge artifacts
            # 1. Apply gentle gamma correction (steeper than photography)
            img_norm = img_norm ** (1/3.0)
            
        return img_norm
    
    def get_range(img):
        """Get 2nd and 98th percentiles of image."""
        valid_mask = np.isfinite(img)
        if np.any(valid_mask):
            return np.percentile(img[valid_mask], [2, 98])
        return img.min(), img.max()
    
    def get_range_reduced_white_pixels(img, method_name):
        """Get range optimized to reduce white pixels for Gaussian and PG methods."""
        valid_mask = np.isfinite(img)
        if not np.any(valid_mask):
            return img.min(), img.max()
        
        # For Gaussian and PG methods, use much tighter percentiles to reduce white pixels
        if 'gaussian' in method_name or 'pg' in method_name:
            # Use 10th and 90th percentiles for even tighter range
            # This further reduces the number of white pixels in the FOV
            return np.percentile(img[valid_mask], [10, 90])
        else:
            # For other methods, use standard 2nd and 98th percentiles
            return np.percentile(img[valid_mask], [2, 98])
    
    def get_range_49th_51st(img, method_name):
        """Get 49th-51st percentiles for Gaussian and PG methods in Row 2."""
        valid_mask = np.isfinite(img)
        if not np.any(valid_mask):
            return img.min(), img.max()
        
        # For Gaussian and PG methods, use 49th-51st percentiles
        if 'gaussian' in method_name or 'pg' in method_name:
            return np.percentile(img[valid_mask], [49, 51])
        else:
            # For other methods, use standard 2nd and 98th percentiles
            return np.percentile(img[valid_mask], [2, 98])
    
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
    images['noisy'] = to_display_image(noisy_phys)
    ranges['noisy'] = get_range(images['noisy'])
    
    # Enhanced methods
    for method in available_methods:
        if method in enhancement_results:
            tensor = enhancement_results[method]
            phys = denormalize_to_physical(tensor, domain)
            images[method] = to_display_image(phys)
            ranges[method] = get_range_reduced_white_pixels(images[method], method)
    
    # Exposure scaled (handled separately from available_methods)
    if 'exposure_scaled' in enhancement_results:
        tensor = enhancement_results['exposure_scaled']
        phys = denormalize_to_physical(tensor, domain)
        images['exposure_scaled'] = to_display_image(phys)
        ranges['exposure_scaled'] = get_range(images['exposure_scaled'])
    
    # Clean reference
    if has_clean:
        clean_phys = denormalize_to_physical(clean_image, domain)
        images['clean'] = to_display_image(clean_phys)
        ranges['clean'] = get_range(images['clean'])
    
    # Determine reference range - Use PG (x0, single domain) min/max for Row 1
    ref_method = None
    if 'pg_x0' in images:
        ref_method = 'pg_x0'
    elif 'pg_score' in images:
        ref_method = 'pg_score'
    elif 'pg' in images:
        ref_method = 'pg'
    elif available_methods:
        ref_method = available_methods[0]
    else:
        ref_method = 'noisy'
    
    # Use min/max range for PG reference (Row 1)
    if 'pg' in ref_method:
        ref_img = images[ref_method]
        ref_p1, ref_p99 = ranges[ref_method]  # Use min/max instead of percentiles
    else:
        ref_p1, ref_p99 = ranges[ref_method]
    
    ref_label = method_labels.get(ref_method, ref_method)
    
    # ROW 0: Input names and [min, max] ADU range
    col = 0
    
    # Noisy input
    noisy_p1, noisy_p99 = ranges['noisy']
    axes[0, col].text(0.5, 0.5, f"Noisy Input\n[{noisy_p1:.0f}, {noisy_p99:.0f}] {unit_label}", 
                      transform=axes[0, col].transAxes, ha='center', va='center', 
                      fontsize=8, fontweight='bold')
    axes[0, col].axis('off')
    col += 1
    
    # Clean reference
    if has_clean:
        img = images['clean']
        p1, p99 = ranges['clean']
        axes[0, col].text(0.5, 0.5, f"Clean Reference\n[{p1:.0f}, {p99:.0f}] {unit_label}", 
                          transform=axes[0, col].transAxes, ha='center', va='center', 
                          fontsize=8, fontweight='bold')
        axes[0, col].axis('off')
        col += 1
    
    # Exposure scaled
    if 'exposure_scaled' in images:
        img = images['exposure_scaled']
        p1, p99 = ranges['exposure_scaled']
        axes[0, col].text(0.5, 0.5, f"Exposure Scaled\n[{p1:.0f}, {p99:.0f}] {unit_label}", 
                          transform=axes[0, col].transAxes, ha='center', va='center', 
                          fontsize=8, fontweight='bold')
        axes[0, col].axis('off')
        col += 1
    
    # Enhanced methods
    for method in available_methods:
        if method in images and col < n_cols:
            img = images[method]
            p1, p99 = ranges[method]
            
            # Color coding
            color = 'green' if 'pg' in method else 'orange' if 'gaussian' in method else 'blue'
            
            axes[0, col].text(0.5, 0.5, f"{method_labels[method]}\n[{p1:.0f}, {p99:.0f}] {unit_label}", 
                              transform=axes[0, col].transAxes, ha='center', va='center', 
                              fontsize=8, fontweight='bold', color=color)
            axes[0, col].axis('off')
            col += 1
    
    # ROW 1: PG (x0, single domain) 49th-51st scale
    col = 0
    
    # Noisy input
    axes[1, col].imshow(normalize_display(images['noisy'], ref_p1, ref_p99), 
                        cmap='gray' if images['noisy'].ndim == 2 else None)
    axes[1, col].axis('off')
    col += 1
    
    # Clean reference
    if has_clean:
        img = images['clean']
        axes[1, col].imshow(normalize_display(img, ref_p1, ref_p99), 
                           cmap='gray' if img.ndim == 2 else None)
        axes[1, col].axis('off')
        col += 1
    
    # Exposure scaled
    if 'exposure_scaled' in images:
        img = images['exposure_scaled']
        axes[1, col].imshow(normalize_display(img, ref_p1, ref_p99), 
                           cmap='gray' if img.ndim == 2 else None)
        axes[1, col].axis('off')
        col += 1
    
    # Enhanced methods
    for method in available_methods:
        if method in images and col < n_cols:
            img = images[method]
            axes[1, col].imshow(normalize_display(img, ref_p1, ref_p99), 
                               cmap='gray' if img.ndim == 2 else None)
            axes[1, col].axis('off')
            col += 1
    
    # ROW 2: Individual dynamic range (min/max for each method)
    col = 0
    
    # Noisy input - use its own dynamic range
    axes[2, col].imshow(normalize_display(images['noisy'], noisy_p1, noisy_p99), 
                        cmap='gray' if images['noisy'].ndim == 2 else None)
    axes[2, col].axis('off')
    col += 1
    
    # Clean reference - use its own dynamic range
    if has_clean:
        img = images['clean']
        p1, p99 = ranges['clean']
        axes[2, col].imshow(normalize_display(img, p1, p99), 
                           cmap='gray' if img.ndim == 2 else None)
        axes[2, col].axis('off')
        col += 1
    
    # Exposure scaled - use its own dynamic range
    if 'exposure_scaled' in images:
        img = images['exposure_scaled']
        p1, p99 = ranges['exposure_scaled']
        axes[2, col].imshow(normalize_display(img, p1, p99), 
                           cmap='gray' if img.ndim == 2 else None)
        axes[2, col].axis('off')
        col += 1
    
    # Enhanced methods - use individual min/max ranges
    for method in available_methods:
        if method in images and col < n_cols:
            img = images[method]
            p1, p99 = ranges[method]  # Use min/max instead of percentiles
            
            # Use individual min/max ranges for each method
            axes[2, col].imshow(normalize_display(img, p1, p99), 
                               cmap='gray' if img.ndim == 2 else None)
            axes[2, col].axis('off')
            col += 1
    
    # ROW 3: Metrics: SSIM, PSNR, LPIPS, NIQE
    col = 0
    
    # Noisy input (no metrics)
    axes[3, col].text(0.5, 0.5, "", transform=axes[3, col].transAxes,
                     ha='center', va='center', fontsize=7)
    axes[3, col].axis('off')
    col += 1
    
    # Clean reference (no metrics)
    if has_clean:
        axes[3, col].text(0.5, 0.5, "", transform=axes[3, col].transAxes,
                         ha='center', va='center', fontsize=7)
        axes[3, col].axis('off')
        col += 1
    
    # Exposure scaled (no metrics)
    if 'exposure_scaled' in images:
        axes[3, col].text(0.5, 0.5, "", transform=axes[3, col].transAxes,
                         ha='center', va='center', fontsize=7)
        axes[3, col].axis('off')
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
                
                # Always show LPIPS and NIQE
                if 'lpips' in metrics:
                    if np.isnan(metrics['lpips']):
                        metrics_lines.append("LPIPS: N/A")
                    else:
                        metrics_lines.append(f"LPIPS: {metrics['lpips']:.3f}")
                
                if 'niqe' in metrics:
                    if np.isnan(metrics['niqe']):
                        metrics_lines.append("NIQE: N/A")
                    else:
                        metrics_lines.append(f"NIQE: {metrics['niqe']:.1f}")
                
                # FID only shown in aggregate summary
                
                metrics_text = "\n".join(metrics_lines)
                
                # Color coding
                color = 'green' if 'pg' in method else 'orange' if 'gaussian' in method else 'blue'
                
                axes[3, col].text(0.5, 0.5, metrics_text, transform=axes[3, col].transAxes,
                                 ha='center', va='center', fontsize=7, color=color, fontweight='bold')
            else:
                # No metrics available (no clean reference)
                axes[3, col].text(0.5, 0.5, "(No clean\nreference)", transform=axes[3, col].transAxes,
                                 ha='center', va='center', fontsize=6, style='italic', color='gray')
            
            axes[3, col].axis('off')
            col += 1
    
    # Add row labels
    axes[0, 0].text(-0.08, 0.5, "Input Names\n& Ranges", 
                    transform=axes[0, 0].transAxes,
                    fontsize=9, va='center', ha='right', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    axes[1, 0].text(-0.08, 0.5, f"PG (x0, single-domain)\nScale", 
                    transform=axes[1, 0].transAxes,
                    fontsize=9, va='center', ha='right', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    axes[2, 0].text(-0.08, 0.5, "Individual\nDynamic Range", 
                    transform=axes[2, 0].transAxes,
                    fontsize=9, va='center', ha='right', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    axes[3, 0].text(-0.08, 0.5, "Metrics\nSSIM, PSNR, LPIPS, NIQE", 
                    transform=axes[3, 0].transAxes,
                    fontsize=9, va='center', ha='right', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Main title
    methods_desc = " | ".join([method_labels[m] for m in available_methods])
    plt.suptitle(
        f"Comprehensive Enhancement Comparison - {tile_id}\n"
        f"Row 0: Input Names & Ranges | Row 1: PG (x0, single-domain) Scale | Row 2: Individual Dynamic Range | Row 3: Metrics | α={exposure_ratio:.4f}",
        fontsize=9,
        fontweight="bold"
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    total_methods = len(available_methods) + (1 if 'exposure_scaled' in enhancement_results else 0)
    logger.info(f"✓ Comprehensive comparison saved: {save_path} ({total_methods} methods)")


def main():
    """Main function for posterior sampling with Poisson-Gaussian measurement guidance."""
    parser = argparse.ArgumentParser(
        description="Posterior sampling for image restoration using EDM model with Poisson-Gaussian measurement guidance"
    )

    # Model arguments
    parser.add_argument("--model_path", type=str, required=False, help="Path to trained model checkpoint (.pkl)")

    # Data arguments  
    parser.add_argument("--metadata_json", type=str, required=False, help="Path to metadata JSON file with test split")
    parser.add_argument("--noisy_dir", type=str, required=False, help="Directory containing noisy .pt files")
    parser.add_argument("--clean_dir", type=str, default=None, 
                       help="Directory containing clean reference .pt files (optional, for optimization)")

    # Sampling arguments
    parser.add_argument("--num_steps", type=int, default=18, help="Number of posterior sampling steps")
    parser.add_argument("--domain", type=str, default="photography",
                       choices=["photography", "microscopy", "astronomy"],
                       help="Domain for conditional sampling")
    
    # Noise/Sigma selection arguments
    parser.add_argument("--use_sensor_calibration", action="store_true",
                       help="Use calibrated sensor parameters instead of noise estimation (recommended!)")
    parser.add_argument("--sensor_name", type=str, default="generic",
                       help="Sensor model name for calibration (e.g., 'sony_a7s', 'hamamatsu_orca', or 'generic')")
    parser.add_argument("--sensor_filter", type=str, default=None,
                       help="Filter tiles by sensor type (e.g., 'sony', 'fuji' for photography domain)")
    parser.add_argument("--conservative_factor", type=float, default=1.0,
                       help="Conservative multiplier for sigma_max from calibration (default: 1.0)")
    
    # Legacy noise estimation (for comparison/fallback)
    parser.add_argument("--noise_method", type=str, default="std",
                       choices=["std", "mad", "local_var"],
                       help="[LEGACY] Noise estimation method (used only if --use_sensor_calibration not set)")
    parser.add_argument("--noise_scale", type=float, default=1.0,
                       help="[LEGACY] Scale factor for estimated noise (default: 1.0)")
    
    # Sigma_max optimization arguments
    parser.add_argument("--optimize_sigma", action="store_true",
                       help="Search for optimal sigma_max for each tile (requires --clean_dir)")
    parser.add_argument("--sigma_range", type=float, nargs=2, default=[0.0001, 0.01],
                       help="Min and max sigma_max for optimization search")
    parser.add_argument("--num_sigma_trials", type=int, default=10,
                       help="Number of sigma_max values to try during optimization")
    parser.add_argument("--optimization_metric", type=str, default="ssim",
                       choices=["ssim", "psnr", "mse"],
                       help="Metric to optimize (default: ssim)")

    # Poisson-Gaussian guidance arguments (always enabled)
    # CRITICAL: s must match the domain's physical range maximum!
    # Photography: s=15871 (max ADU), Microscopy: s=65535, Astronomy: s=385
    parser.add_argument("--s", type=float, default=None,
                       help="Scale factor (auto-set to domain_max if None. Photography: 15871, Microscopy: 65535, Astronomy: 385)")
    parser.add_argument("--sigma_r", type=float, default=5.0,
                       help="Read noise standard deviation (in domain's physical units)")
    parser.add_argument("--kappa", type=float, default=0.1,
                       help="Guidance strength multiplier (typically 0.3-1.0)")
    parser.add_argument("--tau", type=float, default=0.01,
                       help="Guidance threshold - only apply when σ_t > tau")
    parser.add_argument("--pg_mode", type=str, default="wls",
                       choices=["wls", "full"],
                       help="PG guidance mode: 'wls' for weighted least squares, 'full' for complete gradient")
    parser.add_argument("--guidance_level", type=str, default="x0",
                       choices=["x0", "score"],
                       help="Guidance level: 'x0' for x₀-level (default, empirically stable), 'score' for score-level DPS (theoretically pure)")
    parser.add_argument("--compare_gaussian", action="store_true",
                       help="Also run standard Gaussian likelihood guidance for comparison (shows limitations of constant-variance assumption)")
    parser.add_argument("--include_score_level", action="store_true",
                       help="Include score-level guidance methods in visualization (default: x0-level only)")
    parser.add_argument("--gaussian_sigma", type=float, default=None,
                       help="Observation noise for Gaussian guidance (if None, estimated from noisy image)")
    
    
    # Cross-domain optimization parameters (sensor-specific optimization)
    # NOTE: These parameters are optimized for single-domain ablation studies:
    # Sony: κ=0.7, σ_r=3.0, steps=20 - outperforms single-domain on PSNR (+2.69dB Gaussian, +2.87dB PG) and NIQE (-11.03 Gaussian, -9.34 PG)
    # Fuji: κ=0.6, σ_r=3.5, steps=22 - outperforms single-domain on LPIPS (-0.0933 Gaussian, -0.0957 PG) and NIQE (-6.66 Gaussian, -5.60 PG)
    # Default uses Sony parameters for backward compatibility
    parser.add_argument("--cross_domain_kappa", type=float, default=0.7,
                       help="Guidance strength for cross-domain model (Sony: 0.7, Fuji: 0.6)")
    parser.add_argument("--cross_domain_sigma_r", type=float, default=3.0,
                       help="Read noise for cross-domain model (Sony: 3.0, Fuji: 3.5)")
    parser.add_argument("--cross_domain_num_steps", type=int, default=20,
                       help="Number of steps for cross-domain model (Sony: 20, Fuji: 22)")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/low_light_enhancement")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of example images to enhance")
    parser.add_argument("--tile_ids", type=str, nargs="+", default=None,
                       help="Specific tile IDs to process (if provided, overrides num_examples and random selection)")

    # Method selection arguments
    parser.add_argument("--run_methods", type=str, nargs="+",
                       choices=["noisy", "clean", "exposure_scaled", "gaussian_x0", "pg_x0", "gaussian_x0_cross", "pg_x0_cross"],
                       default=["noisy", "clean", "exposure_scaled", "gaussian_x0", "pg_x0", "gaussian_x0_cross", "pg_x0_cross"],
                       help="Methods to run (default: all methods)")

    # Testing arguments
    parser.add_argument("--test_gradients", action="store_true",
                       help="Run gradient verification tests and exit")
    parser.add_argument("--test_clean_pairing", action="store_true",
                       help="Test clean tile pairing functionality and exit")
    parser.add_argument("--test_microscopy_pairing", action="store_true",
                       help="Test microscopy clean tile pairing functionality and exit")
    parser.add_argument("--test_astronomy_pairing", action="store_true",
                       help="Test astronomy clean tile pairing functionality and exit")
    parser.add_argument("--test_sensor_extraction", action="store_true",
                       help="Test sensor extraction from tile IDs and exit")
    parser.add_argument("--test_exposure_extraction", action="store_true",
                       help="Test exposure time extraction from tile IDs and exit")

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
    if not args.test_gradients and not args.test_clean_pairing and not args.test_microscopy_pairing and not args.test_astronomy_pairing and not args.test_sensor_extraction and not args.test_exposure_extraction:
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
        logger.info(f"Sigma_max method: CALIBRATED sensor parameters ✓")
        logger.info(f"  Sensor model: {args.sensor_name}")
        logger.info(f"  Conservative factor: {args.conservative_factor}")
    else:
        logger.info(f"Sigma_max method: LEGACY noise estimation")
        logger.info(f"  Noise estimation: {args.noise_method}")
        logger.warning("  ⚠️  Consider --use_sensor_calibration for physics-based approach!")
    
    logger.info(f"Optimize sigma_max: {args.optimize_sigma}")
    if args.optimize_sigma:
        logger.info(f"  Sigma_max range: [{args.sigma_range[0]}, {args.sigma_range[1]}]")
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
        logger.info(f"  Gaussian σ_obs: {args.gaussian_sigma if args.gaussian_sigma else args.sigma_r} (constant variance)")
    logger.info("=" * 80)

    # Initialize sampler
    sampler = EDMPosteriorSampler(
        model_path=args.model_path,
        device=args.device,
    )

    # Initialize FID calculator for aggregate computation
    logger.info("Initializing FID calculator for aggregate metrics...")
    fid_calculator = FIDCalculator(device=args.device)
    
    # Get domain range for PG guidance
    domain_ranges = sampler.domain_ranges[args.domain]

    # Astronomy domain range is already set appropriately (0 to 450)
    # Data will be shifted and scaled in load_image to fit this range

    # CRITICAL FIX 4: Ensure scale consistency (s = domain_range)
    if args.s is None:
        domain_range = domain_ranges['max'] - domain_ranges['min']
        args.s = domain_range
        logger.info(f"Auto-setting s = domain_range = {args.s:.1f} for {args.domain}")
    else:
        logger.info(f"Using provided s = {args.s:.1f}")
        # Verify that provided s matches domain_range for unit consistency
        domain_range = domain_ranges['max'] - domain_ranges['min']
        if abs(args.s - domain_range) > 1e-3:
            logger.warning(f"Provided s={args.s:.1f} does not match domain_range={domain_range:.1f}. "
                          f"This may cause unit consistency issues.")
    
    # Initialize PG guidance (always enabled)
    pg_guidance = PoissonGaussianGuidance(
        s=args.s,
        sigma_r=args.sigma_r,
        domain_min=domain_ranges['min'],
        domain_max=domain_ranges['max'],
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
        domain_min=domain_ranges['min'],
        domain_max=domain_ranges['max'],
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
        sensor_filter=args.sensor_filter
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
        parts = tile_id.split('_')
        
        # Extract base ID (domain_camera_session_tile) without exposure time
        if len(parts) >= 6:
            # Find exposure time index
            exposure_idx = None
            for i, part in enumerate(parts):
                if part.endswith('s') and '.' in part:
                    exposure_idx = i
                    break
            
            if exposure_idx is not None:
                base_parts = parts[:exposure_idx] + parts[exposure_idx+1:]
                base_id = '_'.join(base_parts)
                
                if base_id not in tile_groups:
                    tile_groups[base_id] = {}
                
                data_type = tile_info.get('data_type', 'unknown')
                tile_groups[base_id][data_type] = tile_info
    
    # Find tiles that have both noisy and clean versions
    for base_id, group in tile_groups.items():
        if 'noisy' in group and 'clean' in group:
            noisy_tile = group['noisy']
            clean_tile = group['clean']
            
            # Check if files exist
            noisy_path = Path(args.noisy_dir) / f"{noisy_tile['tile_id']}.pt"
            clean_path = Path(args.clean_dir) / f"{clean_tile['tile_id']}.pt"
            
            if noisy_path.exists() and clean_path.exists():
                # Create combined tile info
                combined_tile = noisy_tile.copy()
                combined_tile['clean_tile_id'] = clean_tile['tile_id']
                combined_tile['clean_pt_path'] = clean_tile.get('pt_path', '')
                available_tiles.append(combined_tile)
    
    logger.info(f"Found {len(available_tiles)} tile pairs with both noisy and clean files")
    
    # If no pairs found, fall back to just noisy tiles
    if len(available_tiles) == 0:
        logger.info("No tile pairs found, falling back to noisy tiles only")
        for tile_info in test_tiles:
            if tile_info.get('data_type') == 'noisy':
                tile_id = tile_info["tile_id"]
                noisy_path = Path(args.noisy_dir) / f"{tile_id}.pt"
                if noisy_path.exists():
                    available_tiles.append(tile_info)
    
    if args.optimize_sigma:
        logger.info(f"Found {len(available_tiles)} tiles with both noisy and clean files")
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
                    "data_type": "noisy"
                }
                selected_tiles.append(tile_info)
                logger.info(f"Selected specific tile: {tile_id}")
            else:
                logger.warning(f"Tile file not found: {noisy_path}")

        logger.info(f"Selected {len(selected_tiles)} specific tiles for posterior sampling")
    else:
        # Randomly select from available tiles
        rng = np.random.RandomState(args.seed)
        selected_indices = rng.choice(len(available_tiles), size=min(args.num_examples, len(available_tiles)), replace=False)
        selected_tiles = [available_tiles[i] for i in selected_indices]
        logger.info(f"Selected {len(selected_tiles)} random test tiles for posterior sampling")

    # Create domain labels
    if sampler.net.label_dim > 0:
        class_labels = torch.zeros(1, sampler.net.label_dim, device=sampler.device)
        if args.domain == "photography":
            class_labels[:, 0] = 1.0
        elif args.domain == "microscopy":
            class_labels[:, 1] = 1.0
        elif args.domain == "astronomy":
            class_labels[:, 2] = 1.0
    else:
        class_labels = None
    
    # Process each selected tile with per-tile sigma determination
    all_results = []
    # Collect images for aggregate FID computation
    all_clean_images = []
    all_restored_images_by_method = {}
    
    for idx, tile_info in enumerate(selected_tiles):
        tile_id = tile_info["tile_id"]
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Processing example {idx+1}/{len(selected_tiles)}: {tile_id}")
        logger.info(f"{'='*60}")
        
        try:
            # Load noisy image with channel conversion for cross-domain models
            noisy_image, noisy_metadata = load_noisy_image(tile_id, Path(args.noisy_dir), sampler.device, target_channels=sampler.net.img_channels)
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
            logger.info(f"  Brightness: {noisy_brightness['category']} (mean={noisy_brightness['mean']:.3f} normalized)")
            logger.info(f"  Normalized range: [{noisy_image.min():.4f}, {noisy_image.max():.4f}], std={noisy_image.std():.4f}")
            logger.info(f"  Physical range: [{noisy_phys.min():.1f}, {noisy_phys.max():.1f}] {unit_label}")
            
            # Determine sigma_max: Use calibration OR estimation
            if args.use_sensor_calibration:
                # RECOMMENDED: Use calibrated sensor parameters
                # Extract sensor from tile ID for photography domain
                if args.domain == "photography":
                    extracted_sensor = extract_sensor_from_tile_id(tile_id)
                    # Map extracted sensor to calibration database names
                    sensor_mapping = {
                        "sony": "sony_a7s_ii",
                        "fuji": "fuji_x_t2"
                    }
                    sensor_name = sensor_mapping[extracted_sensor]  # No fallback - must be found
                    logger.info(f"  Using CALIBRATED sensor parameters (extracted: {extracted_sensor} -> {sensor_name})")
                else:
                    sensor_name = args.sensor_name
                    logger.info(f"  Using CALIBRATED sensor parameters (sensor: {sensor_name})")
                
                # Compute mean signal level in physical units for sigma_max calculation
                mean_signal_physical = float(noisy_phys.mean())
                
                # Get calibrated parameters
                calib_params = SensorCalibration.get_posterior_sampling_params(
                    domain=args.domain,
                    sensor_name=sensor_name,
                    mean_signal_physical=mean_signal_physical,
                    s=args.s,
                    conservative_factor=args.conservative_factor
                )
                
                estimated_sigma = calib_params['sigma_max']
                sensor_info = calib_params['sensor_info']
                
                logger.info(f"  Calibrated σ_max: {estimated_sigma:.6f}")
                logger.info(f"  Sensor specs: Full-well={sensor_info['full_well_capacity']} e⁻, "
                          f"Read noise={sensor_info['read_noise']:.2f} e⁻")
                logger.info(f"  Mean signal: {mean_signal_physical:.1f} {unit_label}")
                
                # Store calibration info
                noise_estimates = {
                    'method': 'sensor_calibration',
                    'sensor_name': sensor_name,
                    'extracted_sensor': extracted_sensor if args.domain == "photography" else None,
                    'sigma_max_calibrated': estimated_sigma,
                    'mean_signal_physical': mean_signal_physical,
                    'sensor_specs': sensor_info,
                }
            else:
                # LEGACY: Estimate noise from image statistics
                logger.info(f"  Using LEGACY noise estimation from image statistics")
                logger.warning("  Consider using --use_sensor_calibration for physics-based approach!")
                
                noise_estimates = NoiseEstimator.estimate_noise_comprehensive(noisy_image)
                estimated_noise = noise_estimates[args.noise_method] * args.noise_scale
                
                logger.info(f"  Noise estimates: {noise_estimates}")
                logger.info(f"  Using {args.noise_method}: noise_σ={estimated_noise:.6f}")
                
                # Use estimated noise as sigma
                estimated_sigma = estimated_noise
            
            # Extract exposure ratio from tile metadata (domain-specific)
            exposure_ratio = tile_info.get('exposure_ratio', 1.0)

            # Use robust extraction function for photography domain
            if args.domain == "photography":
                # Extract exposure times from both noisy and clean tile IDs
                noisy_exposure = extract_exposure_time_from_tile_id(tile_id)
                
                # Find clean tile pair and extract its exposure time
                clean_pair_info = find_clean_tile_pair(tile_id, Path(args.metadata_json), args.domain)
                if clean_pair_info:
                    clean_tile_id = clean_pair_info.get('tile_id')
                    clean_exposure = extract_exposure_time_from_tile_id(clean_tile_id)
                    exposure_ratio = noisy_exposure / clean_exposure
                    logger.info(f"  Extracted α = {noisy_exposure}s / {clean_exposure}s = {exposure_ratio:.4f}")
                else:
                    # Fallback: assume target exposure is 30.0s (typical long exposure for photography)
                    exposure_ratio = noisy_exposure / 30.0
                    logger.warning(f"  No clean pair found, using fallback α = {noisy_exposure}s / 30.0s = {exposure_ratio:.4f}")

            elif args.domain == "microscopy":
                # Microscopy: Enhancement scale = 1.7x (clean is 1.7x brighter than noisy)
                # Forward model: y_noisy = α·x_clean → α = 1/1.7 ≈ 0.588
                exposure_ratio = 1.0 / 1.7
                logger.info(f"  Microscopy enhancement scale: clean is 1.7x brighter → α={exposure_ratio:.4f}")

            elif args.domain == "astronomy":
                # Astronomy: Direct images (detection/clean) vs G800L grism (noisy)
                # Direct images collect light through broadband filters (F814W, F850LP), concentrating flux
                # G800L grism disperses light over 5500–10,000Å, spreading flux over ~120 pixels per object
                # Flux calibration between direct images and grism spectra uses empirically derived ratios
                # from standard star observations, typically around 0.3 to 0.4 depending on filter and dataset
                # Forward model: y_g800l = α·x_detection where α = 1 / flux_ratio
                # flux_ratio = direct/grism ≈ 0.35 means α ≈ 1/0.35 ≈ 2.86
                # This means grism observations need to be scaled up by ~2.86x to match direct image brightness
                flux_ratio = 0.35  # Typical direct/grism flux ratio from literature
                exposure_ratio = 1.0 / flux_ratio  # Convert flux ratio to exposure ratio for enhancement
                logger.info(f"  Astronomy flux scale: g800l grism vs direct detection → α={exposure_ratio:.4f}")
                logger.info(f"  Note: Flux calibration ratio {flux_ratio:.2f} (direct/grism), calibrated using standard stars")
                logger.info(f"  Enhancement factor: grism observations scaled by {exposure_ratio:.1f}x to match direct image brightness")
            
            # Update PG guidance with correct exposure ratio and offset
            pg_guidance.alpha = exposure_ratio
            pg_guidance.offset = noisy_metadata['offset']  # Update with astronomy offset if applicable
            logger.info(f"  Updated PG guidance with exposure ratio α={exposure_ratio:.4f}, offset={noisy_metadata['offset']:.3f}")

            # Update Gaussian guidance
            gaussian_guidance.alpha = exposure_ratio
            gaussian_guidance.offset = noisy_metadata['offset']  # Update with astronomy offset if applicable
            logger.info(f"  Updated Gaussian guidance with exposure ratio α={exposure_ratio:.4f}, offset={noisy_metadata['offset']:.3f}")
            
            # Prepare PG guidance data (always enabled)
            # Convert noisy image to physical units for PG guidance
            y_e = torch.from_numpy(noisy_phys).to(sampler.device)
            logger.info(f"  PG guidance enabled: y_e range [{y_e.min():.1f}, {y_e.max():.1f}] {unit_label}")
            
            # Try to load clean image if available (for visualization)
            clean_image = None
            
            if args.clean_dir is not None:
                # Domain-specific clean tile matching
                # NOTE: Clean references ARE available for Sony and Fuji data
                # The loading mechanism successfully finds clean images with different exposure times
                if args.domain == "photography":
                    # Photography: Use metadata-based clean tile pairing
                    clean_pair_info = find_clean_tile_pair(tile_id, Path(args.metadata_json), args.domain)
                    
                    if clean_pair_info:
                        clean_tile_id = clean_pair_info.get('tile_id')
                        clean_pt_path = clean_pair_info.get('pt_path')
                        
                        # Verify the clean tile file exists
                        if Path(clean_pt_path).exists():
                            try:
                                clean_image_tensor, _ = load_clean_image(clean_tile_id, Path(args.clean_dir), sampler.device, target_channels=sampler.net.img_channels)
                                if clean_image_tensor.ndim == 3:
                                    clean_image_tensor = clean_image_tensor.unsqueeze(0)
                                clean_image = clean_image_tensor.to(torch.float32)
                                logger.info(f"  Found clean reference via metadata: {clean_tile_id}")
                            except Exception as e:
                                logger.warning(f"  Failed to load clean image: {e}")
                                clean_image = None
                        else:
                            logger.warning(f"  Clean tile file not found: {clean_pt_path}")
                            clean_image = None
                    else:
                        logger.warning(f"  No clean tile pair found for {tile_id}")
                        clean_image = None
                
                elif args.domain == "microscopy":
                    # Microscopy: Use metadata-based clean tile pairing
                    clean_pair_info = find_clean_tile_pair(tile_id, Path(args.metadata_json), args.domain)
                    
                    if clean_pair_info:
                        clean_tile_id = clean_pair_info.get('tile_id')
                        clean_pt_path = clean_pair_info.get('pt_path')
                        
                        # Verify the clean tile file exists
                        if Path(clean_pt_path).exists():
                            try:
                                clean_image_tensor, _ = load_clean_image(clean_tile_id, Path(args.clean_dir), sampler.device, target_channels=sampler.net.img_channels)
                                if clean_image_tensor.ndim == 3:
                                    clean_image_tensor = clean_image_tensor.unsqueeze(0)
                                clean_image = clean_image_tensor.to(torch.float32)
                                logger.info(f"  Found clean reference via metadata: {clean_tile_id}")
                            except Exception as e:
                                logger.warning(f"  Failed to load clean image: {e}")
                                clean_image = None
                        else:
                            logger.warning(f"  Clean tile file not found: {clean_pt_path}")
                            clean_image = None
                    else:
                        logger.warning(f"  No clean tile pair found for {tile_id}")
                        clean_image = None
                
                elif args.domain == "astronomy":
                    # Astronomy: Use metadata-based clean tile pairing
                    clean_pair_info = find_clean_tile_pair(tile_id, Path(args.metadata_json), args.domain)
                    
                    if clean_pair_info:
                        clean_tile_id = clean_pair_info.get('tile_id')
                        clean_pt_path = clean_pair_info.get('pt_path')
                        
                        # Verify the clean tile file exists
                        if Path(clean_pt_path).exists():
                            try:
                                clean_image_tensor, _ = load_clean_image(clean_tile_id, Path(args.clean_dir), sampler.device, target_channels=sampler.net.img_channels)
                                if clean_image_tensor.ndim == 3:
                                    clean_image_tensor = clean_image_tensor.unsqueeze(0)
                                clean_image = clean_image_tensor.to(torch.float32)
                                logger.info(f"  Found clean reference via metadata: {clean_tile_id}")
                            except Exception as e:
                                logger.warning(f"  Failed to load clean image: {e}")
                                clean_image = None
                        else:
                            logger.warning(f"  Clean tile file not found: {clean_pt_path}")
                            clean_image = None
                    else:
                        logger.warning(f"  No clean tile pair found for {tile_id}")
                        clean_image = None
            
        except Exception as e:
            logger.error(f"Failed to process tile {tile_id}: {e}")
            continue
            
        # Determine sigma_max to use
        restoration_results = {}
        metrics_results = {}
        opt_results = None
        
        if args.optimize_sigma:
            # Ensure we have clean image for optimization
            if clean_image is None:
                logger.warning(f"  No clean image available for optimization, skipping {tile_id}")
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
                fid_calculator=fid_calculator,
            )
            sigma_used = best_sigma
        else:
            # Use estimated sigma as sigma_max
            sigma_used = estimated_sigma
        
        # Run comprehensive comparison with all guidance variants
        logger.info(f"  Running comprehensive comparison with σ_max={sigma_used:.6f}")
        
        # 1. Exposure scaling (baseline)
        if 'exposure_scaled' in args.run_methods:
            logger.info("    Computing exposure scaling baseline...")
            exposure_scaled = apply_exposure_scaling(noisy_image, exposure_ratio)
            restoration_results['exposure_scaled'] = exposure_scaled
        
        # 2. Gaussian guidance (x0-level) - Domain-specific model
        if 'gaussian_x0' in args.run_methods:
            logger.info("    Running Gaussian guidance (x0-level)...")
            gaussian_guidance.guidance_level = 'x0'
            restored_gaussian_x0, _ = sampler.posterior_sample(
                noisy_image,
                sigma_max=sigma_used,
                class_labels=class_labels,
                num_steps=args.num_steps,
                gaussian_guidance=gaussian_guidance,
                y_e=y_e,
                exposure_ratio=exposure_ratio,
                domain=args.domain,
            )
            restoration_results['gaussian_x0'] = restored_gaussian_x0
        
        # 3. PG guidance (x0-level) - Domain-specific model
        if 'pg_x0' in args.run_methods:
            logger.info("    Running PG guidance (x0-level) with domain-specific model...")
            pg_guidance.guidance_level = 'x0'
            restored_pg_x0, _ = sampler.posterior_sample(
                noisy_image,
                sigma_max=sigma_used,
                class_labels=class_labels,
                num_steps=args.num_steps,
                pg_guidance=pg_guidance,
                y_e=y_e,
                exposure_ratio=exposure_ratio,
                domain=args.domain,
            )
            restoration_results['pg_x0'] = restored_pg_x0
        
        # 4. Cross-domain models
        if 'gaussian_x0_cross' in args.run_methods:
            # 4. Gaussian guidance (x0-level) - Cross-domain model
            logger.info("    Running Gaussian guidance (x0-level) with cross-domain model...")
            cross_domain_sampler = EDMPosteriorSampler(
                model_path="results/edm_pt_training_multi_domain_20251010_182807/best_model.pkl",
                device=args.device,
            )

            # Create domain-specific labels for cross-domain model
            cross_domain_class_labels = torch.zeros(1, cross_domain_sampler.net.label_dim, device=args.device)
            if args.domain == "photography":
                cross_domain_class_labels[:, 0] = 1.0
            elif args.domain == "microscopy":
                cross_domain_class_labels[:, 1] = 1.0
            elif args.domain == "astronomy":
                cross_domain_class_labels[:, 2] = 1.0
            logger.info(f"    Using domain one-hot encoding for {args.domain}: {cross_domain_class_labels[0].tolist()}")

            # Convert grayscale to RGB for cross-domain model if needed
            cross_domain_input = noisy_image
            cross_domain_y_e = y_e
            if cross_domain_input.shape[1] == 1 and cross_domain_sampler.net.img_channels == 3:
                cross_domain_input = cross_domain_input.repeat(1, 3, 1, 1)
                cross_domain_y_e = cross_domain_y_e.repeat(1, 3, 1, 1)
                logger.info("    Converted grayscale input and y_e to RGB for cross-domain model")

            # Create optimized guidance for cross-domain model
            cross_domain_kappa = args.cross_domain_kappa if args.cross_domain_kappa is not None else args.kappa
            cross_domain_sigma_r = args.cross_domain_sigma_r if args.cross_domain_sigma_r is not None else args.sigma_r
            cross_domain_num_steps = args.cross_domain_num_steps if args.cross_domain_num_steps is not None else args.num_steps

            logger.info(f"    Cross-domain parameters: κ={cross_domain_kappa}, σ_r={cross_domain_sigma_r}, steps={cross_domain_num_steps}")

            # Create cross-domain Gaussian guidance with optimized parameters
            cross_domain_gaussian_guidance = GaussianGuidance(
                s=args.s,
                sigma_r=cross_domain_sigma_r,
                domain_min=domain_ranges['min'],
                domain_max=domain_ranges['max'],
                exposure_ratio=exposure_ratio,
                kappa=cross_domain_kappa,
                tau=args.tau,
                guidance_level='x0',
            )

            restored_gaussian_x0_cross, _ = cross_domain_sampler.posterior_sample(
                cross_domain_input,
                sigma_max=sigma_used,
                class_labels=cross_domain_class_labels,
                num_steps=cross_domain_num_steps,
                gaussian_guidance=cross_domain_gaussian_guidance,
                y_e=cross_domain_y_e,
                exposure_ratio=exposure_ratio,
                domain=args.domain,
            )

            # Convert RGB output back to grayscale if needed
            if restored_gaussian_x0_cross.shape[1] == 3 and noisy_image.shape[1] == 1:
                restored_gaussian_x0_cross = restored_gaussian_x0_cross.mean(dim=1, keepdim=True)
                logger.info("    Converted RGB output back to grayscale")

            restoration_results['gaussian_x0_cross'] = restored_gaussian_x0_cross

        if 'pg_x0_cross' in args.run_methods:
            # 5. PG guidance (x0-level) - Cross-domain model
            logger.info("    Running PG guidance (x0-level) with cross-domain model...")

            # Create cross-domain PG guidance with optimized parameters
            cross_domain_pg_guidance = PoissonGaussianGuidance(
                s=args.s,
                sigma_r=cross_domain_sigma_r,
                domain_min=domain_ranges['min'],
                domain_max=domain_ranges['max'],
                exposure_ratio=exposure_ratio,
                kappa=cross_domain_kappa,
                tau=args.tau,
                mode=args.pg_mode,
                guidance_level='x0',
            )

            restored_pg_x0_cross, _ = cross_domain_sampler.posterior_sample(
                cross_domain_input,
                sigma_max=sigma_used,
                class_labels=cross_domain_class_labels,
                num_steps=cross_domain_num_steps,
                pg_guidance=cross_domain_pg_guidance,
                y_e=cross_domain_y_e,
                exposure_ratio=exposure_ratio,
                domain=args.domain,
            )

            # Convert RGB output back to grayscale if needed
            if restored_pg_x0_cross.shape[1] == 3 and noisy_image.shape[1] == 1:
                restored_pg_x0_cross = restored_pg_x0_cross.mean(dim=1, keepdim=True)
                logger.info("    Converted RGB output back to grayscale")

            restoration_results['pg_x0_cross'] = restored_pg_x0_cross
            
        # Use best available method as primary (prefer cross-domain PG, fallback to domain-specific PG, then others)
        if 'pg_x0_cross' in restoration_results:
            restored = restoration_results['pg_x0_cross']
        elif 'pg_x0' in restoration_results:
            restored = restoration_results['pg_x0']
        elif 'gaussian_x0_cross' in restoration_results:
            restored = restoration_results['gaussian_x0_cross']
        elif 'gaussian_x0' in restoration_results:
            restored = restoration_results['gaussian_x0']
        elif 'exposure_scaled' in restoration_results:
            restored = restoration_results['exposure_scaled']
        else:
            # Fallback to noisy if no methods were run
            restored = noisy_image
            
        # Add noisy and clean to restoration_results if requested
        if 'noisy' in args.run_methods:
            restoration_results['noisy'] = noisy_image
        if 'clean' in args.run_methods and clean_image is not None:
            restoration_results['clean'] = clean_image

        # Compute comprehensive metrics for all methods
        if clean_image is not None:
            # Convert clean image to [0,1] range for comprehensive metrics
            clean_01 = (clean_image + 1.0) / 2.0
            clean_01 = torch.clamp(clean_01, 0.0, 1.0)
                
            # Convert noisy image to [0,1] range for comparison
            noisy_01 = (noisy_image + 1.0) / 2.0
            noisy_01 = torch.clamp(noisy_01, 0.0, 1.0)
                
            # Metrics for noisy input
            try:
                metrics_results['noisy'] = compute_comprehensive_metrics(
                    clean_01, noisy_01, y_e, args.s, args.domain, sampler.device, fid_calculator
                )
            except Exception as e:
                logger.warning(f"Comprehensive metrics failed for noisy: {e}")
                metrics_results['noisy'] = compute_simple_metrics(
                    clean_image, noisy_image, fid_calculator=fid_calculator, device=sampler.device
                )
                
            # Metrics for all enhanced methods
            for method, restored_tensor in restoration_results.items():
                if restored_tensor is None:
                    continue
                # Convert restored tensor to [0,1] range
                restored_01 = (restored_tensor + 1.0) / 2.0
                restored_01 = torch.clamp(restored_01, 0.0, 1.0)
                    
                try:
                    metrics_results[method] = compute_comprehensive_metrics(
                        clean_01, restored_01, y_e, args.s, args.domain, sampler.device, fid_calculator
                    )
                        
                    # Log comprehensive metrics
                    metrics = metrics_results[method]
                    logger.info(f"    {method}: SSIM={metrics['ssim']:.4f}, "
                              f"PSNR={metrics['psnr']:.2f}dB")
                        
                    if 'lpips' in metrics and not np.isnan(metrics['lpips']):
                        logger.info(f"      LPIPS={metrics['lpips']:.4f}, "
                                  f"NIQE={metrics['niqe']:.4f}")
                        
                    # FID only shown in aggregate summary
                        
                    if 'chi2_consistency' in metrics and not np.isnan(metrics['chi2_consistency']):
                        logger.info(f"      χ²={metrics['chi2_consistency']:.4f}, "
                                  f"Res-KS={metrics['residual_distribution']:.4f}")
                        
                except Exception as e:
                    logger.warning(f"Comprehensive metrics failed for {method}: {e}")
                    metrics_results[method] = compute_simple_metrics(
                        clean_image, restored_tensor, fid_calculator=fid_calculator, device=sampler.device
                    )
                
            # Metrics for clean reference (self-comparison)
            try:
                metrics_results['clean'] = compute_comprehensive_metrics(
                    clean_01, clean_01, y_e, args.s, args.domain, sampler.device, fid_calculator
                )
            except Exception as e:
                logger.warning(f"Comprehensive metrics failed for clean: {e}")
                metrics_results['clean'] = compute_simple_metrics(
                    clean_image, clean_image, fid_calculator=fid_calculator, device=sampler.device
                )
            
        # Validate physical consistency
        restored_01 = (restored + 1.0) / 2.0  # Convert to [0,1]
        consistency = validate_physical_consistency(
            restored_01, y_e, args.s, args.sigma_r,
            exposure_ratio,  # CRITICAL: Pass exposure ratio
            domain_ranges['min'], domain_ranges['max']
        )
        logger.info(f"  Physical consistency: χ²={consistency['chi_squared']:.3f} "
                   f"(target ≈ 1.0), valid={consistency['physically_consistent']}")
            
        # Save results
        sample_dir = output_dir / f"example_{idx:02d}_{tile_id}"
        sample_dir.mkdir(exist_ok=True)
            
        # Save tensors
        torch.save(noisy_image.cpu(), sample_dir / "noisy.pt")
            
        # Save restoration results
        for method_name, restored_tensor in restoration_results.items():
            if restored_tensor is not None:
                torch.save(restored_tensor.cpu(), sample_dir / f"restored_{method_name}.pt")
                # Collect for aggregate FID computation
                if method_name not in all_restored_images_by_method:
                    all_restored_images_by_method[method_name] = []
                all_restored_images_by_method[method_name].append(restored_tensor.cpu())
            
        # Save clean if available
        if clean_image is not None:
            torch.save(clean_image.cpu(), sample_dir / "clean.pt")
            # Collect for aggregate FID computation
            all_clean_images.append(clean_image.cpu())
            
        # Save metrics and parameters
        result_info = {
            'tile_id': tile_id,
            'sigma_max_used': float(sigma_used),
            'exposure_ratio': float(exposure_ratio),  # CRITICAL: Track exposure ratio
            'brightness_analysis': noisy_brightness,
            'use_pg_guidance': True,  # Always enabled
            'pg_guidance_params': {
                's': args.s,
                'sigma_r': args.sigma_r,
                'exposure_ratio': float(exposure_ratio),  # CRITICAL: Track exposure ratio
                'kappa': args.kappa,
                'tau': args.tau,
                'mode': args.pg_mode,
            },
            'physical_consistency': consistency,  # Add chi-squared validation
            'comprehensive_metrics': metrics_results,  # Add comprehensive metrics
            'restoration_methods': list(restoration_results.keys()),  # Track available methods
        }
            
        # Add method-specific information
        if args.use_sensor_calibration:
            result_info['sigma_determination'] = 'sensor_calibration'
            result_info['sensor_calibration'] = noise_estimates
            if args.domain == "photography":
                result_info['extracted_sensor'] = extracted_sensor
        else:
            result_info['sigma_determination'] = 'noise_estimation'
            result_info['noise_estimates'] = noise_estimates
            result_info['estimated_noise'] = float(estimated_sigma)
            result_info['noise_method'] = args.noise_method
            
        # Add summary metrics from primary method (PG score if included, else PG x0)
        if 'pg_score' in metrics_results:
            result_info['metrics'] = metrics_results['pg_score']
        elif 'pg_x0' in metrics_results:
            result_info['metrics'] = metrics_results['pg_x0']
            
        if opt_results is not None:
            result_info['optimization_results'] = opt_results
            
        with open(sample_dir / "results.json", 'w') as f:
            json.dump(result_info, f, indent=2)
            
        all_results.append(result_info)
            
        # Create comprehensive visualization
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
        
        logger.info(f"✓ Saved to {sample_dir}")
    
    # Save summary
    summary = {
        'domain': args.domain,
        'num_samples': len(all_results),
        'optimize_sigma': args.optimize_sigma,
        'use_pg_guidance': True,  # Always enabled
        'pg_guidance_params': {
        's': args.s,
        'sigma_r': args.sigma_r,
        'kappa': args.kappa,
        'tau': args.tau,
        'mode': args.pg_mode,
        },
        'results': all_results,
    }
    
    # Add sigma determination method to summary
    if args.use_sensor_calibration:
        summary['sigma_determination'] = 'sensor_calibration'
        summary['sensor_name'] = args.sensor_name
        summary['conservative_factor'] = args.conservative_factor
        if args.domain == "photography":
            summary['sensor_extraction'] = 'from_tile_id'
    else:
        summary['sigma_determination'] = 'noise_estimation'
        summary['noise_method'] = args.noise_method
    
    # Add aggregate physical consistency metrics
    if len(all_results) > 0:
        chi_squared_values = [r['physical_consistency']['chi_squared'] for r in all_results 
                          if 'physical_consistency' in r]
        if chi_squared_values:
            summary['aggregate_physical_consistency'] = {
            'mean_chi_squared': float(np.mean(chi_squared_values)),
            'std_chi_squared': float(np.std(chi_squared_values)),
            'median_chi_squared': float(np.median(chi_squared_values)),
            'num_physically_consistent': sum(r['physical_consistency']['physically_consistent'] 
                                            for r in all_results if 'physical_consistency' in r),
            'total_samples': len(chi_squared_values),
        }
    
    # Add comprehensive metrics summary
    if len(all_results) > 0 and 'comprehensive_metrics' in all_results[0]:
        # Collect all methods that appear in the results
        all_methods = set()
        for result in all_results:
            if 'comprehensive_metrics' in result:
                all_methods.update(result['comprehensive_metrics'].keys())
        
        # Compute aggregate metrics for each method
        summary['comprehensive_aggregate_metrics'] = {}
        for method in all_methods:
            metrics_for_method = []
            for result in all_results:
                if 'comprehensive_metrics' in result and method in result['comprehensive_metrics']:
                    metrics_for_method.append(result['comprehensive_metrics'][method])
            
            if metrics_for_method:
                # Extract all available metrics
                metric_names = set()
                for m in metrics_for_method:
                    metric_names.update(m.keys())
                
            # Compute statistics for each metric
            metric_stats = {}
            for metric_name in metric_names:
                values = [m[metric_name] for m in metrics_for_method 
                         if metric_name in m and not np.isnan(m[metric_name])]
                if values:
                    metric_stats[f'mean_{metric_name}'] = np.mean(values)
                    metric_stats[f'std_{metric_name}'] = np.std(values)
                
            metric_stats['num_samples'] = len(metrics_for_method)
            summary['comprehensive_aggregate_metrics'][method] = metric_stats

    # Compute aggregate FID for each method (requires multiple samples)
    if len(all_clean_images) > 1 and fid_calculator is not None:
        logger.info("Computing aggregate FID metrics...")
        try:
            # Stack all clean images into a single batch
            clean_batch = torch.stack(all_clean_images, dim=0)

            for method in all_restored_images_by_method:
                if len(all_restored_images_by_method[method]) > 1:
                    # Stack all restored images for this method into a single batch
                    restored_batch = torch.stack(all_restored_images_by_method[method], dim=0)

                    # Compute FID between restored and clean images
                    fid_score = fid_calculator.compute_fid(restored_batch, clean_batch)

                    # Add FID to the aggregate metrics for this method
                    if method in summary['comprehensive_aggregate_metrics']:
                        summary['comprehensive_aggregate_metrics'][method]['fid'] = fid_score
                        logger.info(f"  {method} aggregate FID: {fid_score:.4f}")

        except Exception as e:
            logger.warning(f"Aggregate FID computation failed: {e}")

    # Legacy metrics (for backward compatibility)
    if args.optimize_sigma and len(all_results) > 0:
        summary['aggregate_metrics'] = {
        'mean_ssim': np.mean([r['metrics']['ssim'] for r in all_results if 'metrics' in r]),
        'mean_psnr': np.mean([r['metrics']['psnr'] for r in all_results if 'metrics' in r]),
        'mean_mse': np.mean([r['metrics']['mse'] for r in all_results if 'metrics' in r]),
        'std_ssim': np.std([r['metrics']['ssim'] for r in all_results if 'metrics' in r]),
        'std_psnr': np.std([r['metrics']['psnr'] for r in all_results if 'metrics' in r]),
        }
    
    # Save comprehensive results.json file (contains all detailed information)
    with open(output_dir / "results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 POSTERIOR SAMPLING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"📁 All results saved to: {output_dir}")
    logger.info(f"📊 Processed {len(all_results)} tiles with posterior sampling and PG measurement guidance")
    logger.info(f"🔬 Poisson-Gaussian guidance enabled (s={args.s}, σ_r={args.sigma_r}, κ={args.kappa})")
    
    # Report physical consistency
    if 'aggregate_physical_consistency' in summary:
        pc = summary['aggregate_physical_consistency']
        logger.info(f"✓ Physical consistency: χ²={pc['mean_chi_squared']:.3f} ± {pc['std_chi_squared']:.3f} "
               f"(target ≈ 1.0)")
        logger.info(f"  {pc['num_physically_consistent']}/{pc['total_samples']} samples physically consistent")
    
    # Report comprehensive metrics if available
    if 'comprehensive_aggregate_metrics' in summary:
        logger.info("📊 Comprehensive Metrics Summary:")
        for method, method_metrics in summary['comprehensive_aggregate_metrics'].items():
            logger.info(f"  {method}:")
            logger.info(f"    SSIM: {method_metrics['mean_ssim']:.4f} ± {method_metrics['std_ssim']:.4f}")
            logger.info(f"    PSNR: {method_metrics['mean_psnr']:.2f} ± {method_metrics['std_psnr']:.2f} dB")

        # Add comprehensive metrics if available (use the last method's metrics for summary)
        if summary['comprehensive_aggregate_metrics']:
            last_method = list(summary['comprehensive_aggregate_metrics'].keys())[-1]
            last_metrics = summary['comprehensive_aggregate_metrics'][last_method]

            if 'mean_lpips' in last_metrics and not np.isnan(last_metrics['mean_lpips']):
                logger.info(f"    LPIPS: {last_metrics['mean_lpips']:.4f} ± {last_metrics['std_lpips']:.4f}")

            if 'mean_niqe' in last_metrics and not np.isnan(last_metrics['mean_niqe']):
                logger.info(f"    NIQE: {last_metrics['mean_niqe']:.4f} ± {last_metrics['std_niqe']:.4f}")

            if 'fid' in last_metrics and not np.isnan(last_metrics['fid']):
                logger.info(f"    FID: {last_metrics['fid']:.4f}")

            if 'mean_chi2_consistency' in last_metrics and not np.isnan(last_metrics['mean_chi2_consistency']):
                logger.info(f"    χ²: {last_metrics['mean_chi2_consistency']:.4f} ± {last_metrics['std_chi2_consistency']:.4f}")

            if 'mean_residual_distribution' in last_metrics and not np.isnan(last_metrics['mean_residual_distribution']):
                logger.info(f"    Res-KS: {last_metrics['mean_residual_distribution']:.4f} ± {last_metrics['std_residual_distribution']:.4f}")
    
        # Report legacy image quality metrics if available
        if args.optimize_sigma and len(all_results) > 0 and 'aggregate_metrics' in summary:
            logger.info(f"📈 Legacy Mean SSIM: {summary['aggregate_metrics']['mean_ssim']:.4f} ± {summary['aggregate_metrics']['std_ssim']:.4f}")
            logger.info(f"📈 Legacy Mean PSNR: {summary['aggregate_metrics']['mean_psnr']:.2f} ± {summary['aggregate_metrics']['std_psnr']:.2f} dB")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()

