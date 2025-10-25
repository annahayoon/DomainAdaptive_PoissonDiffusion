#!/usr/bin/env python3
"""
Unified Denoising Methods Comparison

This script runs all four denoising methods on the same noisy test data:
1. Low-light enhancement with adaptive sigma (sample_pt_lle.py)
2. Poisson-Gaussian guidance (sample_noisy_pt_lle_PGguidance.py) 
3. DPS guidance (sample_noisy_pt_lle_DPSguidance.py)
4. Synthetic noise denoising (sample_clean_plus_synthetic_noise.py)

Then creates a comprehensive comparison visualization.

Usage:
    python sample/unified_denoising_comparison.py \
        --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
        --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
        --noisy_dir dataset/processed/pt_tiles/photography/noisy \
        --clean_dir dataset/processed/pt_tiles/photography/clean \
        --output_dir results/unified_denoising_comparison \
        --domain photography \
        --num_examples 3 \
        --num_steps 18
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
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add project root and EDM to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

# Setup logging
import logging

# Import EDM components
import external.edm.dnnlib
from external.edm.torch_utils import distributed as dist

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Import components from other scripts
class NoiseEstimator:
    """Estimate noise level in images using various methods."""
    
    @staticmethod
    def estimate_std(image: torch.Tensor) -> float:
        """Simple standard deviation estimate."""
        return image.std().item()
    
    @staticmethod
    def estimate_mad(image: torch.Tensor) -> float:
        """Median Absolute Deviation (MAD) based noise estimation."""
        img = image.cpu().numpy()
        median = np.median(img)
        mad = np.median(np.abs(img - median))
        sigma = 1.4826 * mad
        return float(sigma)
    
    @staticmethod
    def estimate_local_variance(image: torch.Tensor, patch_size: int = 8) -> float:
        """Estimate noise from local variance in homogeneous regions."""
        img = image.cpu().numpy()
        
        if img.ndim == 4:  # (B, C, H, W)
            img = img[0, 0]
        elif img.ndim == 3:  # (C, H, W)
            img = img[0]
        
        h, w = img.shape
        min_variance = float('inf')
        
        for i in range(0, h - patch_size + 1, patch_size // 2):
            for j in range(0, w - patch_size + 1, patch_size // 2):
                patch = img[i:i+patch_size, j:j+patch_size]
                variance = np.var(patch)
                min_variance = min(min_variance, variance)
        
        return float(np.sqrt(min_variance))
    
    @staticmethod
    def estimate_noise_comprehensive(image: torch.Tensor) -> Dict[str, float]:
        """Compute multiple noise estimates and return all."""
        estimates = {
            'std': NoiseEstimator.estimate_std(image),
            'mad': NoiseEstimator.estimate_mad(image),
            'local_var': NoiseEstimator.estimate_local_variance(image),
        }
        estimates['recommended'] = estimates['local_var']
        return estimates
    
    @staticmethod
    def brightness_adaptive_sigma(
        image: torch.Tensor,
        base_noise: float,
        base_sigma: float = 0.02,
        brightness_scale: float = 10.0,
        min_sigma: float = 0.002,
        max_sigma: float = 0.5,
    ) -> Dict[str, float]:
        """Compute brightness-adaptive sigma for enhancement."""
        img_01 = (image + 1.0) / 2.0
        img_01 = torch.clamp(img_01, 0, 1)
        
        mean_brightness = img_01.mean().item()
        darkness = 1.0 - mean_brightness
        
        adaptive_sigma = base_sigma * (1.0 + brightness_scale * darkness)
        adaptive_sigma = max(min_sigma, min(max_sigma, adaptive_sigma))
        
        noise_relative_sigma = base_noise * (1.0 + brightness_scale * darkness)
        noise_relative_sigma = max(min_sigma, min(max_sigma, noise_relative_sigma))
        
        return {
            'mean_brightness': mean_brightness,
            'darkness_factor': darkness,
            'base_noise': base_noise,
            'adaptive_sigma': adaptive_sigma,
            'noise_relative_sigma': noise_relative_sigma,
            'brightness_scale': brightness_scale,
        }


class PoissonGaussianGuidance(nn.Module):
    """
    Physics-informed guidance for photon-limited imaging
    
    Implements the score of the Poisson-Gaussian likelihood:
    ∇_x log p(y_e|x)
    
    This tells the diffusion model how to adjust predictions to match
    observed noisy measurements while respecting physical noise properties.
    
    CORRECTED: Physical Unit Requirements
    ======================================
    - x_pred must be in [0,1] normalized space
    - y_observed must be in PHYSICAL UNITS (ADU, electrons, counts)
    - s: Scale factor for numerical stability (typically 1000-10000)
    - sigma_r: Read noise in same physical units as y_observed
    - domain_min, domain_max: Physical range of the domain for normalization
    
    The key insight: We normalize BOTH y_observed and expected values to [0, s] 
    scale internally, ensuring proper comparison. This makes s independent of 
    the domain's physical range.
    
    Args:
        s: Scale factor for normalized comparison (default: 1000)
        sigma_r: Read noise standard deviation (in physical units)
        domain_min: Minimum physical value of the domain
        domain_max: Maximum physical value of the domain
        exposure_ratio: Ratio of low exposure to high exposure (alpha = t_low/t_long)
        kappa: Guidance strength multiplier (typically 0.3-1.0)
        tau: Guidance threshold - only apply when σ_t > tau
        mode: 'wls' for weighted least squares, 'full' for complete gradient
        epsilon: Small constant for numerical stability
    """
    
    def __init__(
        self,
        s: float,
        sigma_r: float,
        domain_min: float,
        domain_max: float,
        exposure_ratio: float = 1.0,  # FIX 2 (Phase 1): NEW parameter
        kappa: float = 0.5,
        tau: float = 0.01,
        mode: str = 'wls',
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.s = s
        self.sigma_r = sigma_r
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.alpha = exposure_ratio  # FIX 2 (Phase 1): Store exposure ratio
        self.kappa = kappa
        self.tau = tau
        self.mode = mode
        self.epsilon = epsilon
        
        # Pre-compute constants for efficiency
        self.sigma_r_squared = sigma_r ** 2
        self.domain_range = domain_max - domain_min
        
        logger.info(f"Initialized PG Guidance: s={s}, σ_r={sigma_r}, "
                   f"domain=[{domain_min}, {domain_max}], α={exposure_ratio}, κ={kappa}, τ={tau}, mode={mode}")
    
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
        
        # Apply guidance with schedule
        # Schedule: κ · σ_t² · ∇
        # Larger steps when noise is high, smaller when low
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
        Weighted Least Squares gradient (Equation 3, first term)
        
        FIX 2 (Phase 1): Include exposure ratio in forward model
        =========================================================
        Forward model: y_low = Poisson(α·s·x_clean) + N(0, σ_r²)
        
        where α = exposure_ratio = t_low/t_long
        
        Physical interpretation:
        - Residual (y_e_scaled - α·s·x): prediction error in normalized space
        - Variance (α·s·x + σ_r²): local noise level (signal-dependent!)
        - s scaling: convert back to [0,1] space
        """
        # Step 1: Normalize y_e from physical units to [0,1]
        y_e_norm = (y_e_physical - self.domain_min) / self.domain_range
        y_e_norm = torch.clamp(y_e_norm, 0.0, 1.0)
        
        # Step 2: Scale both to [0, s] range for stable computation
        y_e_scaled = y_e_norm * self.s
        x0_scaled = x0_hat * self.s
        
        # Step 3: Compute variance at LOW exposure (apply alpha!)
        # FIX 2: Include exposure ratio in expected photons and variance
        expected_photons = self.alpha * x0_scaled  # CRITICAL: multiply by alpha
        variance_scaled = self.alpha * x0_scaled + self.sigma_r_squared
        variance_scaled = torch.clamp(variance_scaled, min=self.epsilon)
        
        # Step 4: Weighted residual
        residual_scaled = y_e_scaled - expected_photons
        weighted_residual = residual_scaled / variance_scaled
        
        # Step 5: Scale back to [0,1] space (include alpha in numerator as per formula)
        gradient = self.alpha * weighted_residual / self.s
        
        return gradient
    
    def _full_gradient(
        self,
        x0_hat: torch.Tensor,
        y_e_physical: torch.Tensor
    ) -> torch.Tensor:
        """
        Full Poisson-Gaussian gradient (Equation 3, complete)
        
        Includes both data term and variance term for complete likelihood.
        """
        # Step 1: Normalize y_e from physical units to [0,1]
        y_e_norm = (y_e_physical - self.domain_min) / self.domain_range
        y_e_norm = torch.clamp(y_e_norm, 0.0, 1.0)
        
        # Step 2: Scale both to [0, s] range
        y_e_scaled = y_e_norm * self.s
        x0_scaled = x0_hat * self.s
        
        # Step 3: Compute variance
        variance_scaled = x0_scaled + self.sigma_r_squared
        variance_scaled = torch.clamp(variance_scaled, min=self.epsilon)
        
        # Step 4: Data term gradient
        residual_scaled = y_e_scaled - x0_scaled
        grad_data = residual_scaled / variance_scaled
        
        # Step 5: Variance term gradient
        grad_variance = -0.5 * (residual_scaled ** 2) / (variance_scaled ** 2)
        
        # Step 6: Total gradient
        gradient_scaled = grad_data + grad_variance
        
        # Step 7: Scale back to [0,1] space
        gradient = gradient_scaled / self.s
        
        return gradient
    
    def _validate_inputs(self, x0_hat: torch.Tensor, y_e: torch.Tensor):
        """Validate input tensors."""
        assert x0_hat.shape == y_e.shape, f"Shape mismatch: {x0_hat.shape} vs {y_e.shape}"
        assert x0_hat.device == y_e.device, "Device mismatch"
        assert x0_hat.min() >= 0.0 and x0_hat.max() <= 1.0, f"x0_hat must be in [0,1], got [{x0_hat.min():.3f}, {x0_hat.max():.3f}]"


def compute_metrics(clean: np.ndarray, enhanced: np.ndarray, data_range: float = None) -> Dict[str, float]:
    """Compute SSIM and PSNR between clean and enhanced images."""
    if clean.ndim == 4:  # (B, C, H, W)
        clean = clean[0, 0]
        enhanced = enhanced[0, 0]
    elif clean.ndim == 3:  # (C, H, W)
        clean = clean[0]
        enhanced = enhanced[0]
    
    # ISSUE 3 FIX: Compute proper data_range for [-1, 1] normalized space
    if data_range is None:
        # For normalized [-1, 1] space used by the model, data_range should be 2
        # Check if data is in [-1, 1] range
        if clean.min() >= -1.0 and clean.max() <= 1.0:
            data_range = 2.0  # Range from -1 to 1
        else:
            # For physical units, use the actual range
            data_range = clean.max() - clean.min()
            if data_range == 0:
                data_range = 1.0
    
    ssim_val = ssim(clean, enhanced, data_range=data_range)
    psnr_val = psnr(clean, enhanced, data_range=data_range)
    mse = np.mean((clean - enhanced) ** 2)
    
    return {
        'ssim': float(ssim_val),
        'psnr': float(psnr_val),
        'mse': float(mse),
    }


def load_test_tiles(metadata_json: Path, domain: str, split: str = "test") -> List[Dict[str, Any]]:
    """Load test tile metadata from JSON file."""
    logger.info(f"Loading {split} tiles for {domain} from {metadata_json}")
    
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)
    
    tiles = metadata.get('tiles', [])
    filtered_tiles = [
        tile for tile in tiles
        if tile.get('domain') == domain and tile.get('split') == split
    ]
    
    logger.info(f"Found {len(filtered_tiles)} {split} tiles for {domain}")
    return filtered_tiles


def load_image(tile_id: str, image_dir: Path, device: torch.device, image_type: str = "image") -> torch.Tensor:
    """Load a .pt file (noisy or clean)."""
    image_path = image_dir / f"{tile_id}.pt"
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    tensor = torch.load(image_path, map_location=device)
    
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
    
    tensor = tensor.float()
    
    if tensor.ndim == 2:  # (H, W)
        tensor = tensor.unsqueeze(0)  # (1, H, W)
    elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3]:  # (H, W, C)
        tensor = tensor.permute(2, 0, 1)  # (C, H, W)
    
    # Convert RGB to grayscale if needed (model expects 1 channel)
    if tensor.ndim == 4 and tensor.shape[1] == 3:  # RGB image (B, C, H, W)
        # Convert to grayscale using standard weights
        tensor_gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
        tensor = tensor_gray
    elif tensor.ndim == 3 and tensor.shape[0] == 3:  # RGB image (C, H, W)
        # Convert to grayscale using standard weights
        tensor_gray = 0.299 * tensor[0:1] + 0.587 * tensor[1:2] + 0.114 * tensor[2:3]
        tensor = tensor_gray
    
    return tensor


def analyze_image_brightness(image: torch.Tensor) -> Dict[str, float]:
    """Analyze image brightness characteristics."""
    img_01 = (image + 1.0) / 2.0
    
    mean_brightness = img_01.mean().item()
    std_brightness = img_01.std().item()
    min_brightness = img_01.min().item()
    max_brightness = img_01.max().item()
    
    img_flat = img_01.flatten()
    p10 = torch.quantile(img_flat, 0.1).item()
    p50 = torch.quantile(img_flat, 0.5).item()
    p90 = torch.quantile(img_flat, 0.9).item()
    
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


def get_domain_range(domain: str) -> Tuple[float, float]:
    """Get the physical range for a domain."""
    domain_ranges = {
        "photography": {"min": 0.0, "max": 15871.0},
        "microscopy": {"min": 0.0, "max": 65535.0},
        "astronomy": {"min": -65.0, "max": 385.0},
    }
    
    domain_range = domain_ranges.get(domain, {"min": 0.0, "max": 1.0})
    return domain_range["min"], domain_range["max"]

def denormalize_to_physical(tensor: torch.Tensor, domain: str) -> np.ndarray:
    """Convert tensor from [-1,1] model space to physical units.
    
    Reverses the 3-step conversion used in process_tiles_pipeline.py:
    Step 1 (saved in .pt): [domain_min, domain_max] → [0,1] (in preprocessing)
    Step 2 (saved in .pt): [0,1] → [-1,1] (in preprocessing)
    
    Here we reverse:
    Step 2 (reverse): [-1,1] → [0,1]
    Step 1 (reverse): [0,1] → [domain_min, domain_max]
    """
    domain_ranges = {
        "photography": {"min": 0.0, "max": 15871.0},
        "microscopy": {"min": 0.0, "max": 65535.0},
        "astronomy": {"min": -65.0, "max": 385.0},
    }
    
    domain_range = domain_ranges.get(domain, {"min": 0.0, "max": 1.0})
    domain_min = domain_range["min"]
    domain_max = domain_range["max"]
    
    # Step 2 reverse: [-1,1] → [0,1]
    tensor_norm = (tensor + 1.0) / 2.0
    tensor_norm = torch.clamp(tensor_norm, 0, 1)
    
    # Step 1 reverse: [0,1] → [domain_min, domain_max]
    tensor_phys = tensor_norm * (domain_max - domain_min) + domain_min
    
    return tensor_phys.cpu().numpy()


def extract_exposure_time(filename: str) -> Optional[float]:
    """Extract exposure time from photography filename.
    
    Format: {scene_id}_{camera_id}_{exposure}
    Examples: "00001_00_0.04s", "00001_00_10s"
    
    Returns: exposure time in seconds or None
    """
    import re
    
    # Pattern: exposure like "0.04s" or "10s" anywhere in filename
    match = re.search(r'(\d+\.?\d*)s', filename.lower())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def calculate_exposure_ratio(noisy_filename: str, clean_filename: str, domain: str) -> float:
    """Calculate exposure ratio α = t_short / t_long from filenames.
    
    For SID dataset:
    - Noisy (short exposure): 0.04s, 0.1s, etc.
    - Clean (long exposure): 10s, 30s
    - α = t_short / t_long
    
    Falls back to 0.01 if extraction fails.
    """
    if domain != "photography":
        return 0.01  # Default for other domains
    
    noisy_exp = extract_exposure_time(noisy_filename)
    clean_exp = extract_exposure_time(clean_filename)
    
    if noisy_exp is not None and clean_exp is not None and clean_exp > 0:
        alpha = noisy_exp / clean_exp
        logger.info(f"  Calculated exposure_ratio: α = {noisy_exp}s / {clean_exp}s = {alpha:.6f}")
        return alpha
    else:
        logger.warning(f"  Could not extract exposures from {noisy_filename} / {clean_filename}")
        logger.warning(f"  Extracted: noisy={noisy_exp}, clean={clean_exp}")
        logger.warning(f"  Falling back to default α=0.01")
        return 0.01


class UnifiedDenoiser:
    """Unified denoiser that implements all four methods."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        domain_ranges: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.domain_ranges = domain_ranges or {
            "photography": {"min": 0.0, "max": 15871.0},
            "microscopy": {"min": 0.0, "max": 65535.0},
            "astronomy": {"min": -65.0, "max": 385.0},
        }
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)
        
        self.net = checkpoint["ema"].to(self.device)
        self.net.eval()
        
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Resolution: {self.net.img_resolution}")
        logger.info(f"  Channels: {self.net.img_channels}")
        logger.info(f"  Label dim: {self.net.label_dim}")
        logger.info(f"  Sigma range: [{self.net.sigma_min}, {self.net.sigma_max}]")
    
    def method_1_cfg_lle(
        self,
        low_light_image: torch.Tensor,
        enhancement_sigma: float,
        class_labels: Optional[torch.Tensor] = None,
        num_steps: int = 18,
        rho: float = 7.0,
        cfg_scale: float = 2.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Method 1: Native EDM CFG Low-Light Enhancement."""
        logger.info(f"Method 1: Native EDM CFG LLE with sigma={enhancement_sigma:.3f}")
        
        sigma_min = max(0.002, self.net.sigma_min)
        sigma_max = min(enhancement_sigma, self.net.sigma_max)
        
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        
        x = low_light_image.to(torch.float64).to(self.device)
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            if cfg_scale > 1.0 and class_labels is not None:
                enhanced_cond = self.net(x, t_cur, class_labels).to(torch.float64)
                null_labels = torch.zeros_like(class_labels)
                enhanced_uncond = self.net(x, t_cur, null_labels).to(torch.float64)
                enhanced = enhanced_uncond + cfg_scale * (enhanced_cond - enhanced_uncond)
            else:
                enhanced = self.net(x, t_cur, class_labels).to(torch.float64)
            
            d_cur = (x - enhanced) / t_cur
            
            x_next = x + (t_next - t_cur) * d_cur
            
            if i < num_steps - 1:
                if cfg_scale > 1.0 and class_labels is not None:
                    enhanced_next_cond = self.net(x_next, t_next, class_labels).to(torch.float64)
                    enhanced_next_uncond = self.net(x_next, t_next, null_labels).to(torch.float64)
                    enhanced_next = enhanced_next_uncond + cfg_scale * (enhanced_next_cond - enhanced_next_uncond)
                else:
                    enhanced_next = self.net(x_next, t_next, class_labels).to(torch.float64)
                
                d_prime = (x_next - enhanced_next) / t_next
                x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
            
            x = x_next
        
        enhanced_output = torch.clamp(x, -1, 1)
        
        return enhanced_output, {
            "method": "cfg_lle",
            "enhancement_sigma": enhancement_sigma,
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
        }
    
    def method_2_pg_guidance(
        self,
        y_observed: torch.Tensor,
        sigma_max: float,
        class_labels: Optional[torch.Tensor] = None,
        num_steps: int = 18,
        rho: float = 7.0,
        pg_guidance: Optional[PoissonGaussianGuidance] = None,
        y_e: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Method 2: Poisson-Gaussian guidance."""
        logger.info(f"Method 2: PG guidance with sigma_max={sigma_max:.3f}")
        
        sigma_min = max(0.002, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)
        
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        
        # FIX 1 (Phase 1): Start from observation, not noise
        # CRITICAL FIX: Start from observation for enhancement mode
        x_init = y_observed.to(torch.float64).to(self.device)
        logger.info("Initialized from low-light observation (enhancement mode)")
        
        # For enhancement: start from observation directly (not scaled by t_steps[0])
        x = x_init.clone()
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # Model prediction
            denoised = self.net(x, t_cur, class_labels).to(torch.float64)
            
            # PG guidance
            if pg_guidance is not None and y_e is not None:
                # Convert denoised from [-1,1] to [0,1] for guidance
                denoised_norm = (denoised + 1.0) / 2.0
                denoised_norm = torch.clamp(denoised_norm, 0.0, 1.0)
                pg_grad_norm = pg_guidance(denoised_norm, y_e, t_cur.item())
                # Scale gradient from [0,1] space to [-1,1] space
                pg_grad = pg_grad_norm * 2.0
                denoised = denoised + pg_grad
            
            # Euler step
            d_cur = (x - denoised) / t_cur
            x_next = x + (t_next - t_cur) * d_cur
            
            # Heun's correction
            if i < num_steps - 1:
                denoised_next = self.net(x_next, t_next, class_labels).to(torch.float64)
                if pg_guidance is not None and y_e is not None:
                    denoised_next_norm = (denoised_next + 1.0) / 2.0
                    denoised_next_norm = torch.clamp(denoised_next_norm, 0.0, 1.0)
                    pg_grad_next_norm = pg_guidance(denoised_next_norm, y_e, t_next.item())
                    pg_grad_next = pg_grad_next_norm * 2.0
                    denoised_next = denoised_next + pg_grad_next
                
                d_prime = (x_next - denoised_next) / t_next
                x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
            
            x = x_next
        
        denoised_output = torch.clamp(x, -1, 1)
        
        return denoised_output, {
            "method": "pg_guidance",
            "sigma_max": sigma_max,
            "num_steps": num_steps,
            "pg_guidance": pg_guidance is not None,
        }
    
    def method_3_dps_guidance(
        self,
        noisy_image: torch.Tensor,
        scale: float,
        background: float = 0.0,
        read_noise: float = 0.0,
        steps: int = 18,
        guidance_weight: float = 1.0,
        condition: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Method 3: DPS guidance."""
        logger.info(f"Method 3: DPS guidance with scale={scale:.1f}, weight={guidance_weight:.1f}")
        
        # Convert noisy image to normalized space [0, 1]
        noisy_norm = (noisy_image + 1.0) / 2.0  # [-1,1] -> [0,1]
        noisy_norm = torch.clamp(noisy_norm, 0, 1)
        
        # Convert to model space [-1, 1] for EDM
        x = noisy_norm * 2.0 - 1.0
        
        # Create noise schedule (use EDM's schedule)
        device = x.device
        sigma_min = max(0.002, self.net.sigma_min)
        sigma_max = min(0.5, self.net.sigma_max)
        rho = 7.0
        
        step_indices = torch.arange(steps, dtype=torch.float64, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (steps - 1) * 
                  (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        
        # DPS sampling loop
        for i in range(len(t_steps) - 1):
            t_curr = t_steps[i]
            t_next = t_steps[i + 1]
            
            # Compute prior score
            with torch.no_grad():
                sigma_t = t_curr if isinstance(t_curr, torch.Tensor) else torch.tensor(t_curr, device=x.device)
                if sigma_t.ndim == 0:
                    sigma_t = sigma_t.unsqueeze(0)
                
                x_0_pred = self.net(x, sigma_t, condition)
                sigma_t = sigma_t.view(-1, 1, 1, 1)
                prior_score = (x - x_0_pred) / sigma_t
            
            # Compute likelihood gradient
            x_norm = (x + 1.0) / 2.0
            x_norm = torch.clamp(x_norm, 0, 1)
            x_electrons = x_norm * scale + background
            
            variance = x_electrons + read_noise**2
            variance = torch.clamp(variance, min=read_noise**2 * 0.1)
            
            # Use proper noisy image in electron space
            noisy_norm = (noisy_image + 1.0) / 2.0
            noisy_norm = torch.clamp(noisy_norm, 0, 1)
            noisy_electrons = noisy_norm * scale + background
            
            likelihood_grad_electrons = (noisy_electrons - x_electrons) / variance
            likelihood_grad_normalized = likelihood_grad_electrons / scale
            likelihood_grad_model = likelihood_grad_normalized * 2.0
            
            # Combine scores with proper step size based on noise schedule
            # Use EDM-style update
            dt = t_next - t_curr
            x = x + dt * (prior_score + guidance_weight * likelihood_grad_model)
            x = torch.clamp(x, -1, 1)
        
        # Return in [-1, 1] range like other methods
        denoised_output = torch.clamp(x, -1, 1)
        
        return denoised_output, {
            "method": "dps_guidance",
            "scale": scale,
            "background": background,
            "read_noise": read_noise,
            "guidance_weight": guidance_weight,
            "steps": steps,
        }
    
    def method_4_synthetic_denoising(
        self,
        noisy_image: torch.Tensor,
        noise_sigma: float,
        class_labels: Optional[torch.Tensor] = None,
        num_steps: int = 18,
        rho: float = 7.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Method 4: Synthetic noise denoising."""
        logger.info(f"Method 4: Synthetic denoising from sigma={noise_sigma:.3f}")
        
        sigma_min = max(0.002, self.net.sigma_min)
        sigma_max = min(noise_sigma, self.net.sigma_max)
        
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        
        x = noisy_image.to(torch.float64).to(self.device)
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            denoised = self.net(x, t_cur, class_labels).to(torch.float64)
            
            d_cur = (x - denoised) / t_cur
            x_next = x + (t_next - t_cur) * d_cur
            
            if i < num_steps - 1:
                denoised_next = self.net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised_next) / t_next
                x_next = x + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
            
            x = x_next
        
        denoised_output = torch.clamp(x, -1, 1)
        
        return denoised_output, {
            "method": "synthetic_denoising",
            "noise_sigma": noise_sigma,
            "num_steps": num_steps,
        }
    
    def denormalize_to_physical(self, tensor: torch.Tensor, domain: str) -> np.ndarray:
        """Convert tensor from [-1,1] model space to physical units."""
        return denormalize_to_physical(tensor, domain)


def find_corresponding_clean_reference(tile_id: str, domain: str, clean_dir: Path) -> Optional[torch.Tensor]:
    """Find the clean reference image that corresponds to the noisy low-light image."""
    if domain != "photography":
        logger.warning(f"Clean reference lookup not implemented for domain: {domain}")
        return None
    
    # Extract scene info from tile_id
    # Format: photography_fuji_00001_00_0.1s_tile_0000
    parts = tile_id.split('_')
    if len(parts) < 6:
        logger.warning(f"Unexpected tile_id format: {tile_id}")
        return None
    
    # Try different exposure times for clean reference
    # Common clean exposure times: 10s, 30s, 1s, 5s
    exposure_times = ["10s", "30s", "1s", "5s"]
    
    for exposure_time in exposure_times:
        clean_parts = parts.copy()
        # Find the exposure time part and replace it
        for i, part in enumerate(clean_parts):
            if part.endswith('s') and '.' in part:  # e.g., "0.1s"
                clean_parts[i] = exposure_time  # e.g., "10s"
                break
        
        clean_tile_id = '_'.join(clean_parts)
        clean_path = clean_dir / f"{clean_tile_id}.pt"
        
        if clean_path.exists():
            logger.info(f"Found clean reference: {clean_path}")
            clean_tensor = torch.load(clean_path)
            # Apply same RGB to grayscale conversion as load_image
            if clean_tensor.ndim == 4 and clean_tensor.shape[1] == 3:  # RGB image (B, C, H, W)
                clean_tensor = 0.299 * clean_tensor[:, 0:1] + 0.587 * clean_tensor[:, 1:2] + 0.114 * clean_tensor[:, 2:3]
            elif clean_tensor.ndim == 3 and clean_tensor.shape[0] == 3:  # RGB image (C, H, W)
                clean_tensor = 0.299 * clean_tensor[0:1] + 0.587 * clean_tensor[1:2] + 0.114 * clean_tensor[2:3]
            return clean_tensor
    
    # Fallback: search for any clean reference with similar scene ID
    # Extract scene ID without exposure time: photography_fuji_00001_00
    scene_parts = parts[:-2]  # Remove exposure time and tile number
    scene_id = '_'.join(scene_parts)
    
    # Find all clean files from the same scene
    clean_files = list(clean_dir.glob(f"{scene_id}_*_tile_*.pt"))
    if clean_files:
        target_tile = parts[-1]  # e.g., "tile_0000"
        # Try to find the same tile number
        matching_files = [f for f in clean_files if target_tile in f.name]
        if matching_files:
            clean_path = matching_files[0]
            logger.info(f"Using clean reference with same tile: {clean_path}")
            clean_tensor = torch.load(clean_path)
            # Apply same RGB to grayscale conversion as load_image
            if clean_tensor.ndim == 4 and clean_tensor.shape[1] == 3:  # RGB image (B, C, H, W)
                clean_tensor = 0.299 * clean_tensor[:, 0:1] + 0.587 * clean_tensor[:, 1:2] + 0.114 * clean_tensor[:, 2:3]
            elif clean_tensor.ndim == 3 and clean_tensor.shape[0] == 3:  # RGB image (C, H, W)
                clean_tensor = 0.299 * clean_tensor[0:1] + 0.587 * clean_tensor[1:2] + 0.114 * clean_tensor[2:3]
            return clean_tensor
        else:
            # Use any available clean reference from same scene
            clean_path = clean_files[0]
            logger.info(f"Using clean reference from same scene: {clean_path}")
            clean_tensor = torch.load(clean_path)
            # Apply same RGB to grayscale conversion as load_image
            if clean_tensor.ndim == 4 and clean_tensor.shape[1] == 3:  # RGB image (B, C, H, W)
                clean_tensor = 0.299 * clean_tensor[:, 0:1] + 0.587 * clean_tensor[:, 1:2] + 0.114 * clean_tensor[:, 2:3]
            elif clean_tensor.ndim == 3 and clean_tensor.shape[0] == 3:  # RGB image (C, H, W)
                clean_tensor = 0.299 * clean_tensor[0:1] + 0.587 * clean_tensor[1:2] + 0.114 * clean_tensor[2:3]
            return clean_tensor
    
    logger.warning(f"Clean reference not found for: {tile_id}")
    return None


def create_comprehensive_comparison(
    noisy_image: torch.Tensor,
    method_results: Dict[str, torch.Tensor],
    domain: str,
    tile_id: str,
    save_path: Path,
    clean_image: Optional[torch.Tensor] = None,
):
    """Create comprehensive comparison visualization."""
    logger.info("Creating comprehensive denoising comparison visualization...")
    
    method_names = list(method_results.keys())
    
    # ISSUE 1 FIX: Reorder methods so synthetic_denoising comes right after clean input
    # Order: cfg_lle, dps_guidance, pg_guidance, synthetic_denoising
    desired_order = ['synthetic_denoising', 'cfg_lle', 'dps_guidance', 'pg_guidance']
    method_names_ordered = [m for m in desired_order if m in method_names]
    # Add any methods not in desired order at the end
    for m in method_names:
        if m not in method_names_ordered:
            method_names_ordered.append(m)
    method_names = method_names_ordered
    
    n_methods = len(method_names)
    
    n_panels = n_methods + 1  # +1 for noisy input
    if clean_image is not None:
        n_panels += 1  # +1 for clean reference
    
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))
    
    def prepare_for_display(phys_array, scale_range=None):
        """Prepare physical array for display using percentile normalization."""
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
        
        min_val = float(np.min(img))
        max_val = float(np.max(img))
        
        valid_mask = np.isfinite(img)
        if scale_range is None:
            if np.any(valid_mask):
                p1, p99 = np.percentile(img[valid_mask], [1, 99])
                scale_min, scale_max = p1, p99
            else:
                p1, p99 = min_val, max_val
                scale_min, scale_max = min_val, max_val
        else:
            scale_min, scale_max = scale_range
            if np.any(valid_mask):
                p1, p99 = np.percentile(img[valid_mask], [1, 99])
            else:
                p1, p99 = min_val, max_val
        
        img_clipped = np.clip(img, scale_min, scale_max)
        img_norm = (img_clipped - scale_min) / (scale_max - scale_min + 1e-8)
        
        return img_norm, min_val, max_val, p1, p99, scale_min, scale_max
    
    domain_units = {
        "photography": "ADU",
        "microscopy": "intensity",
        "astronomy": "counts",
    }
    unit_label = domain_units.get(domain, "units")
    
    # Process noisy input
    noisy_phys = denormalize_to_physical(noisy_image, domain)
    noisy_display, noisy_min, noisy_max, noisy_p1, noisy_p99, noisy_scale_min, noisy_scale_max = prepare_for_display(noisy_phys)
    
    # Define two reference scales for different method types
    noisy_reference_scale = (noisy_scale_min, noisy_scale_max)  # For dark/enhancement methods
    
    # ISSUE 2 FIX: Check if display is grayscale (1 channel) or RGB (3 channels)
    is_grayscale_display = noisy_display.ndim == 2 or (noisy_display.ndim == 3 and noisy_display.shape[2] == 1)
    axes[0].imshow(noisy_display, cmap='gray' if is_grayscale_display else None)
    axes[0].set_title(
        f"Noisy Input\n"
        f"Range: [{noisy_min:.1f}, {noisy_max:.1f}] {unit_label}\n"
        f"Display Scale: [{noisy_scale_min:.1f}, {noisy_scale_max:.1f}]", 
        fontsize=8
    )
    axes[0].axis('off')
    
    # Display clean reference if available and compute its scale
    panel_idx = 1
    clean_reference_scale = None
    if clean_image is not None:
        clean_phys = denormalize_to_physical(clean_image, domain)
        # Compute clean reference scale (without forcing noisy scale)
        clean_display, clean_min, clean_max, clean_p1, clean_p99, clean_scale_min, clean_scale_max = prepare_for_display(clean_phys)
        
        # Store clean reference scale for bright methods
        clean_reference_scale = (clean_scale_min, clean_scale_max)
        
        is_grayscale_clean = clean_display.ndim == 2 or (clean_display.ndim == 3 and clean_display.shape[2] == 1)
        axes[panel_idx].imshow(clean_display, cmap='gray' if is_grayscale_clean else None)
        axes[panel_idx].set_title(
            f"Clean Reference\n"
            f"Range: [{clean_min:.1f}, {clean_max:.1f}] {unit_label}\n"
            f"Display Scale: [{clean_scale_min:.1f}, {clean_scale_max:.1f}]", 
            fontsize=8
        )
        axes[panel_idx].axis('off')
        panel_idx += 1
    
    # Define which methods use which scale
    # Methods using noisy scale (enhancement comparison)
    noisy_scale_methods = []
    # Methods using clean scale (denoising comparison)
    clean_scale_methods = ['synthetic_denoising']
    # Methods using their own scale (independent visualization)
    own_scale_methods = ['cfg_lle', 'pg_guidance', 'dps_guidance']
    
    # Process and display method results in reordered sequence
    for i, method_name in enumerate(method_names):
        result_tensor = method_results[method_name]
        
        result_phys = denormalize_to_physical(result_tensor, domain)
        
        # Choose appropriate scale based on method type
        if method_name in noisy_scale_methods:
            # Enhancement methods: compare with noisy input
            scale_to_use = noisy_reference_scale
            scale_note = "Noisy Scale"
        elif method_name in clean_scale_methods and clean_reference_scale is not None:
            # Denoising methods: compare with clean reference
            scale_to_use = clean_reference_scale
            scale_note = "Clean Scale"
        elif method_name in own_scale_methods:
            # Methods using their own independent scale
            scale_to_use = None
            scale_note = "Own Scale"
        else:
            # Fallback: use method's own scale
            scale_to_use = None
            scale_note = "Own Scale"
        
        result_display, res_min, res_max, res_p1, res_p99, res_scale_min, res_scale_max = prepare_for_display(
            result_phys, 
            scale_range=scale_to_use
        )
        
        # ISSUE 2 FIX: Check if result display is grayscale or RGB
        is_grayscale_result = result_display.ndim == 2 or (result_display.ndim == 3 and result_display.shape[2] == 1)
        axes[panel_idx + i].imshow(result_display, cmap='gray' if is_grayscale_result else None)
        
        # Compute metrics if clean reference is available
        title_text = f"{method_name.replace('_', ' ').title()} [{scale_note}]\n"
        if clean_image is not None:
            # ISSUE 3 FIX: Ensure metrics are computed on same sized data and in proper range
            clean_for_metrics = clean_image.cpu().numpy()
            result_for_metrics = result_tensor.cpu().numpy()
            
            # Ensure shapes match for metric computation
            if clean_for_metrics.shape == result_for_metrics.shape:
                metrics = compute_metrics(
                    clean_for_metrics, 
                    result_for_metrics
                )
                if not np.isnan(metrics['psnr']) and metrics['psnr'] > -1000:
                    title_text += f"PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.3f}\n"
                else:
                    title_text += f"Metrics: N/A (invalid range)\n"
            else:
                title_text += f"Metrics: N/A (shape mismatch)\n"
        
        title_text += f"Range: [{res_min:.1f}, {res_max:.1f}] {unit_label}\n"
        title_text += f"Display Scale: [{res_scale_min:.1f}, {res_scale_max:.1f}]"
        
        axes[panel_idx + i].set_title(title_text, fontsize=8)
        axes[panel_idx + i].axis('off')
    
    # Main title
    comparison_type = "Low-Light Enhancement" if clean_image is None else "Denoising Comparison"
    plt.suptitle(
        f"Unified {comparison_type} Methods (Physical Units: {unit_label}) - {tile_id}\n"
        f"Domain: {domain} | Mixed scaling: Each method uses optimal display scale (see panel titles)",
        fontsize=11,
        fontweight="bold"
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Comparison visualization saved: {save_path}")


def main():
    """Main function for unified denoising comparison."""
    parser = argparse.ArgumentParser(
        description="Unified denoising methods comparison"
    )

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint (.pkl)")

    # Data arguments  
    parser.add_argument("--metadata_json", type=str, required=True, help="Path to metadata JSON file with test split")
    parser.add_argument("--noisy_dir", type=str, required=True, help="Directory containing noisy .pt files")
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory containing clean reference .pt files")

    # Enhancement arguments
    parser.add_argument("--num_steps", type=int, default=18, help="Number of enhancement steps")
    parser.add_argument("--domain", type=str, default="photography",
                       choices=["photography", "microscopy", "astronomy"])
    parser.add_argument("--cfg_scale", type=float, default=2.0,
                       help="Classifier-free guidance scale for method 1")
    
    # Method-specific arguments
    parser.add_argument("--base_sigma", type=float, default=0.02,
                       help="Base sigma for adaptive scaling (method 1)")
    parser.add_argument("--brightness_scale", type=float, default=10.0,
                       help="Brightness scale for adaptive sigma (method 1)")
    parser.add_argument("--pg_scale", type=float, default=15871.0,
                       help="Scale parameter for PG guidance (method 2). If not provided, set to domain_max.")
    parser.add_argument("--pg_read_noise", type=float, default=5.0,
                       help="Read noise for PG guidance (method 2)")
    parser.add_argument("--dps_scale", type=float, default=15871.0,
                       help="Scale parameter for DPS guidance (method 3)")
    parser.add_argument("--dps_read_noise", type=float, default=5.0,
                       help="Read noise for DPS guidance (method 3)")
    parser.add_argument("--dps_guidance_weight", type=float, default=1.0,
                       help="DPS guidance weight (method 3)")
    parser.add_argument("--synthetic_noise_sigma", type=float, default=0.1,
                       help="Synthetic noise level for method 4")
    parser.add_argument("--s", type=float, default=None,
                       help="Scale parameter for Poisson-Gaussian guidance (method 2). If not provided, set to domain_max.")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/unified_denoising_comparison")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of example images to enhance")

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("UNIFIED DENOISING METHODS COMPARISON")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Metadata: {args.metadata_json}")
    logger.info(f"Noisy dir: {args.noisy_dir}")
    logger.info(f"Clean dir: {args.clean_dir}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Enhancement steps: {args.num_steps}")
    logger.info(f"Number of examples: {args.num_examples}")
    logger.info("=" * 80)

    # Initialize unified denoiser
    denoiser = UnifiedDenoiser(
        model_path=args.model_path,
        device=args.device,
    )

    # Load test tiles
    test_tiles = load_test_tiles(
        Path(args.metadata_json),
        args.domain,
        split="test"
    )

    if len(test_tiles) == 0:
        logger.error(f"No test tiles found for domain {args.domain}")
        return

    # Filter tiles to only those that exist in noisy directory and have corresponding clean files
    available_tiles = []
    for tile_info in test_tiles:
        tile_id = tile_info["tile_id"]
        noisy_path = Path(args.noisy_dir) / f"{tile_id}.pt"
        
        if noisy_path.exists():
            # Check if we can find a corresponding clean reference
            clean_ref = find_corresponding_clean_reference(tile_id, args.domain, Path(args.clean_dir))
            if clean_ref is not None:
                available_tiles.append(tile_info)
    
    logger.info(f"Found {len(available_tiles)} tiles with noisy files and corresponding clean references")
    
    if len(available_tiles) == 0:
        logger.error(f"No suitable tiles found")
        return
    
    # Randomly select from available tiles
    rng = np.random.RandomState(args.seed)
    selected_indices = rng.choice(len(available_tiles), size=min(args.num_examples, len(available_tiles)), replace=False)
    selected_tiles = [available_tiles[i] for i in selected_indices]

    logger.info(f"Selected {len(selected_tiles)} test tiles for comparison")

    # Create domain labels
    if denoiser.net.label_dim > 0:
        class_labels = torch.zeros(1, denoiser.net.label_dim, device=denoiser.device)
        if args.domain == "photography":
            class_labels[:, 0] = 1.0
        elif args.domain == "microscopy":
            class_labels[:, 1] = 1.0
        elif args.domain == "astronomy":
            class_labels[:, 2] = 1.0
    else:
        class_labels = None
    
    # Process each selected tile with all four methods
    all_results = []
    
    for idx, tile_info in enumerate(selected_tiles):
        tile_id = tile_info["tile_id"]
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Processing example {idx+1}/{len(selected_tiles)}: {tile_id}")
        logger.info(f"{'='*60}")
        
        try:
            # Load noisy image
            noisy_image = load_image(tile_id, Path(args.noisy_dir), denoiser.device, "noisy")
            
            # Find corresponding clean reference
            clean_image = find_corresponding_clean_reference(tile_id, args.domain, Path(args.clean_dir))
            if clean_image is None:
                logger.warning(f"No clean reference found for {tile_id}, skipping...")
                continue
            
            if noisy_image.ndim == 3:
                noisy_image = noisy_image.unsqueeze(0)
            if clean_image.ndim == 3:
                clean_image = clean_image.unsqueeze(0)
            
            noisy_image = noisy_image.to(torch.float32)
            clean_image = clean_image.to(torch.float32)
            
            # Analyze noisy image brightness
            noisy_brightness = analyze_image_brightness(noisy_image)
            logger.info(f"  Brightness: {noisy_brightness['category']} (mean={noisy_brightness['mean']:.3f})")
            
            # Estimate noise level for method 1
            noise_estimates = NoiseEstimator.estimate_noise_comprehensive(noisy_image)
            estimated_noise = noise_estimates['local_var']
            
            # Apply brightness-adaptive scaling for method 1
            adaptive_info = NoiseEstimator.brightness_adaptive_sigma(
                noisy_image,
                base_noise=estimated_noise,
                base_sigma=args.base_sigma,
                brightness_scale=args.brightness_scale,
            )
            enhancement_sigma = adaptive_info['adaptive_sigma']
            
            logger.info(f"  Using adaptive sigma: {enhancement_sigma:.6f}")

            # Run all four methods
            method_results = {}
            method_metadata = {}
            
            # Method 1: Native EDM CFG LLE
            logger.info("  Running Method 1: Native EDM CFG LLE...")
            start_time = time.time()
            enhanced_1, meta_1 = denoiser.method_1_cfg_lle(
                noisy_image,
                enhancement_sigma=enhancement_sigma,
                class_labels=class_labels,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
            )
            method_results['cfg_lle'] = enhanced_1
            method_metadata['cfg_lle'] = meta_1
            logger.info(f"    Completed in {time.time() - start_time:.2f}s")
            
            # Method 2: PG guidance
            logger.info("  Running Method 2: Poisson-Gaussian Guidance...")
            domain_min, domain_max = get_domain_range(args.domain)
            pg_scale = args.pg_scale if args.s is None else args.s
            
            # ISSUE 3 FIX (UPDATED): Calculate exposure_ratio dynamically from filenames
            # Extract tile_id to find corresponding clean reference
            noisy_tile_id = None
            clean_tile_id = None
            
            # Get noisy and clean tile IDs from metadata
            for tile_ref in test_tiles:
                if tile_ref["tile_id"] == tile_id:
                    noisy_tile_id = tile_id
                    break
            
            # Find corresponding clean reference tile_id
            if noisy_tile_id:
                clean_ref = find_corresponding_clean_reference(noisy_tile_id, args.domain, Path(args.clean_dir))
                if clean_ref is not None:
                    # Get the corresponding tile_id by checking what clean files exist
                    clean_files = list(Path(args.clean_dir).glob("*.pt"))
                    for cf in clean_files:
                        if cf.stem in str(clean_ref):
                            clean_tile_id = cf.stem
                            break
            
            # Calculate exposure_ratio from filenames
            exposure_ratio = 0.01  # Default fallback
            if noisy_tile_id and clean_tile_id:
                exposure_ratio = calculate_exposure_ratio(noisy_tile_id, clean_tile_id, args.domain)
            else:
                logger.warning(f"  Could not determine clean tile_id for {noisy_tile_id}, using default α=0.01")
            
            pg_guidance = PoissonGaussianGuidance(
                s=pg_scale,
                sigma_r=args.pg_read_noise,
                domain_min=domain_min,
                domain_max=domain_max,
                exposure_ratio=exposure_ratio,
                kappa=0.1,  # Reduced for stability
                tau=0.001,  # Reduced to allow guidance at sigma=0.008
                mode='wls',
            )
            if args.s is None:
                logger.info(f"  PG Guidance: s auto-set to domain_max ({domain_max:.1f}), alpha={exposure_ratio:.6f}")
            
            # Convert to electron space for PG guidance
            # ISSUE 3 FIX: Properly convert from [-1,1] model space to physical units first
            noisy_norm = (noisy_image + 1.0) / 2.0
            noisy_norm = torch.clamp(noisy_norm, 0, 1)
            noisy_electrons = noisy_norm * (domain_max - domain_min) + domain_min
            noisy_electrons = torch.clamp(noisy_electrons, domain_min, domain_max)

            start_time = time.time()
            enhanced_2, meta_2 = denoiser.method_2_pg_guidance(
                noisy_image,
                sigma_max=enhancement_sigma,
                class_labels=class_labels,
                num_steps=args.num_steps,
                pg_guidance=pg_guidance,
                y_e=noisy_electrons,
            )
            method_results['pg_guidance'] = enhanced_2
            method_metadata['pg_guidance'] = meta_2
            logger.info(f"    Completed in {time.time() - start_time:.2f}s")
            
            # Method 3: DPS guidance
            logger.info("  Running Method 3: DPS Guidance...")
            start_time = time.time()
            enhanced_3, meta_3 = denoiser.method_3_dps_guidance(
                noisy_image,
                scale=args.dps_scale,
                background=0.0,
                read_noise=args.dps_read_noise,
                steps=args.num_steps,
                guidance_weight=args.dps_guidance_weight,
                condition=class_labels,
            )
            method_results['dps_guidance'] = enhanced_3
            method_metadata['dps_guidance'] = meta_3
            logger.info(f"    Completed in {time.time() - start_time:.2f}s")
            
            # Method 4: Synthetic denoising (using clean image as reference)
            logger.info("  Running Method 4: Synthetic Denoising...")
            # Add synthetic noise to clean image
            synthetic_noise = torch.randn_like(clean_image) * args.synthetic_noise_sigma
            synthetic_noisy = clean_image + synthetic_noise
            
            start_time = time.time()
            enhanced_4, meta_4 = denoiser.method_4_synthetic_denoising(
                synthetic_noisy,
                noise_sigma=args.synthetic_noise_sigma,
                class_labels=class_labels,
                num_steps=args.num_steps,
            )
            method_results['synthetic_denoising'] = enhanced_4
            method_metadata['synthetic_denoising'] = meta_4
            logger.info(f"    Completed in {time.time() - start_time:.2f}s")
            
            # Save results
            sample_dir = output_dir / f"example_{idx:02d}_{tile_id}"
            sample_dir.mkdir(exist_ok=True)
            
            # Save tensors
            torch.save(noisy_image.cpu(), sample_dir / "noisy.pt")
            torch.save(clean_image.cpu(), sample_dir / "clean.pt")
            
            for method_name, result_tensor in method_results.items():
                torch.save(result_tensor.cpu(), sample_dir / f"{method_name}.pt")
            
            # Save metadata
            result_info = {
                'tile_id': tile_id,
                'noise_estimates': noise_estimates,
                'estimated_noise': float(estimated_noise),
                'enhancement_sigma': float(enhancement_sigma),
                'brightness_analysis': noisy_brightness,
                'adaptive_sigma_info': adaptive_info,
                'method_metadata': method_metadata,
                'synthetic_noise_sigma': args.synthetic_noise_sigma,
            }
            
            with open(sample_dir / "results.json", 'w') as f:
                json.dump(result_info, f, indent=2)
            
            all_results.append(result_info)
            
            # Create comprehensive comparison visualization
            comparison_path = sample_dir / "unified_comparison.png"
            create_comprehensive_comparison(
                noisy_image=noisy_image,
                method_results=method_results,
                domain=args.domain,
                tile_id=tile_id,
                save_path=comparison_path,
                clean_image=clean_image,
            )
            
            logger.info(f"✓ Saved to {sample_dir}")
            
        except Exception as e:
            logger.error(f"Failed to process tile {tile_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary = {
        'domain': args.domain,
        'num_samples': len(all_results),
        'methods': ['adaptive_sigma', 'pg_guidance', 'dps_guidance', 'synthetic_denoising'],
        'parameters': {
            'base_sigma': args.base_sigma,
            'brightness_scale': args.brightness_scale,
            'pg_scale': args.pg_scale,
            'pg_read_noise': args.pg_read_noise,
            'dps_scale': args.dps_scale,
            'dps_read_noise': args.dps_read_noise,
            'dps_guidance_weight': args.dps_guidance_weight,
            'synthetic_noise_sigma': args.synthetic_noise_sigma,
            'num_steps': args.num_steps,
            'cfg_scale': args.cfg_scale,
        },
        'results': all_results,
    }
    
    
    logger.info("\n" + "=" * 80)
    logger.info("🎉 UNIFIED DENOISING COMPARISON COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"📁 All results saved to: {output_dir}")
    logger.info(f"📊 Processed {len(all_results)} tiles with all four methods")
    logger.info(f"🔬 Methods: Adaptive Sigma, PG Guidance, DPS Guidance, Synthetic Denoising")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
