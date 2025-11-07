"""
Comprehensive evaluation metrics for Poisson-Gaussian diffusion restoration.

This module implements:
1. Standard image quality metrics (PSNR, SSIM, MSE, LPIPS, NIQE, FID)
2. Physics-specific metrics (χ² consistency, residual whiteness, bias analysis)
3. Sensor-specific metrics (resolution preservation)
4. Baseline comparison utilities

Requirements addressed: 6.3-6.6 from requirements.md
"""

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
import tqdm
from scipy import stats
from scipy.signal import periodogram

from .error_handlers import AnalysisError

# Try importing optional dependencies for standard image quality metrics
try:
    from torchmetrics.image import PeakSignalNoiseRatio as PSNR
    from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
    from torchmetrics.regression import MeanSquaredError as MSE

    TORCHMETRICS_AVAILABLE = True
except ImportError as e:
    TORCHMETRICS_AVAILABLE = False
    PSNR = None
    SSIM = None
    MSE = None

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

EDM_AVAILABLE = False
_lpips_metric_cache = {}

# Set up logging
logger = logging.getLogger(__name__)


# ============================================================================
# Standard Image Quality Metrics
# ============================================================================


def get_lpips_metric(device: str = "cuda"):
    """Get or create LPIPS metric instance (cached)."""
    if device not in _lpips_metric_cache:
        if LPIPS_AVAILABLE and lpips is not None:
            _lpips_metric_cache[device] = lpips.LPIPS(net="alex").to(device)
        else:
            _lpips_metric_cache[device] = None
    return _lpips_metric_cache[device]


def normalize_tensors_for_metrics(
    *tensors: torch.Tensor, device: str = "cuda"
) -> tuple:
    """
    Normalize tensors for metric computation: move to device, ensure float32, add batch dim.

    Args:
        *tensors: Variable number of tensors to normalize
        device: Target device

    Returns:
        Tuple of normalized tensors in [B, C, H, W] format
    """
    normalized = []
    for tensor in tensors:
        # Move to device and ensure float32
        tensor = tensor.to(device).float()
        # Ensure [B, C, H, W] format (torchmetrics requires batch dimension)
        if tensor.ndim == 3:  # [C, H, W]
            tensor = tensor.unsqueeze(0)  # [1, C, H, W]
        normalized.append(tensor)
    return tuple(normalized)


def get_standard_metrics(device: str = "cuda") -> tuple:
    """
    Get initialized standard metrics (PSNR, SSIM) for [-1,1] range data.

    Args:
        device: Target device

    Returns:
        Tuple of (psnr_metric, ssim_metric)
    """
    if not TORCHMETRICS_AVAILABLE:
        raise ImportError(
            "torchmetrics is required for PSNR and SSIM metrics. "
            "Please install torchmetrics or fix the lzma module issue."
        )
    psnr_metric = PSNR(data_range=2.0).to(device)  # Range is 2.0 for [-1,1] data
    ssim_metric = SSIM(data_range=2.0).to(device)  # Range is 2.0 for [-1,1] data
    return psnr_metric, ssim_metric


def compute_core_metrics(
    enhanced: torch.Tensor, long: torch.Tensor, device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute core metrics: PSNR, SSIM, MSE.

    Args:
        enhanced: Enhanced image tensor
        long: Reference image tensor
        device: Device for computation

    Returns:
        Dictionary with core metrics
    """
    psnr_metric, ssim_metric = get_standard_metrics(device)
    psnr_val = psnr_metric(enhanced, long).item()
    ssim_val = ssim_metric(enhanced, long).item()
    mse_val = F.mse_loss(enhanced, long).item()

    return {
        "ssim": ssim_val if not np.isnan(ssim_val) else 0.0,
        "psnr": psnr_val if not np.isnan(psnr_val) else 0.0,
        "mse": mse_val,
    }


def compute_lpips(
    enhanced: torch.Tensor, long: torch.Tensor, device: str = "cuda"
) -> float:
    """
    Compute LPIPS metric.

    Args:
        enhanced: Enhanced image tensor in [-1,1] range
        long: Reference image tensor in [-1,1] range
        device: Device for computation

    Returns:
        LPIPS value (NaN if computation fails)
    """
    lpips_val = float("nan")
    lpips_metric = get_lpips_metric(device)
    if lpips_metric is not None:
        try:
            from .normalization import convert_range

            enhanced_01 = convert_range(enhanced, "[-1,1]", "[0,1]")
            long_01 = convert_range(long, "[-1,1]", "[0,1]")
            lpips_val = lpips_metric(enhanced_01, long_01).item()
        except Exception as e:
            logger.warning(f"LPIPS computation failed: {e}")
            lpips_val = float("nan")
    else:
        logger.warning("lpips library not available - LPIPS set to NaN")

    return lpips_val


def compute_niqe(enhanced: torch.Tensor) -> float:
    """
    Compute NIQE metric.

    Args:
        enhanced: Enhanced image tensor in [-1,1] range

    Returns:
        NIQE value (NaN if computation fails)
    """
    niqe_val = float("nan")
    if PQIQA_AVAILABLE and pyiqa is not None:
        try:
            from .normalization import convert_range

            enhanced_01 = convert_range(enhanced, "[-1,1]", "[0,1]")
            niqe_metric = pyiqa.create_metric("niqe")
            niqe_val = niqe_metric(enhanced_01).item()
        except Exception as e:
            logger.warning(f"NIQE computation failed: {e}")
            niqe_val = float("nan")
    return niqe_val


def compute_all_metrics(
    enhanced: torch.Tensor, long: torch.Tensor, device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute all metrics: core metrics (PSNR, SSIM, MSE) + LPIPS + NIQE.

    Args:
        enhanced: Enhanced image tensor in [B, C, H, W] format
        long: Reference image tensor in [B, C, H, W] format
        device: Device for computation

    Returns:
        Dictionary with all metrics
    """
    # Compute core metrics (PSNR, SSIM, MSE)
    metrics = compute_core_metrics(enhanced, long, device)
    metrics["lpips"] = compute_lpips(enhanced, long, device)
    metrics["niqe"] = compute_niqe(enhanced)

    return metrics


def compute_metrics_by_method(
    long: torch.Tensor,
    enhanced: torch.Tensor,
    method: str,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute metrics based on method type.

    Metric sets:
      - "short": PSNR only
      - "exposure_scaled": PSNR, SSIM, MSE
      - All other restoration methods: PSNR, SSIM, MSE, LPIPS, NIQE

    Args:
        long: Reference image [C, H, W] or [B, C, H, W] in [-1,1] range
        enhanced: Enhanced image [C, H, W] or [B, C, H, W] in [-1,1] range
        method: Method name ('short', 'exposure_scaled', or restoration method)
        device: Device for computation

    Returns:
        Dictionary with method-specific metrics
    """
    # Normalize tensors for metric computation
    long, enhanced = normalize_tensors_for_metrics(long, enhanced, device=device)

    if method == "short":
        # Short exposure: PSNR only
        psnr_metric, _ = get_standard_metrics(device)
        psnr_val = psnr_metric(enhanced, long).item()
        return {
            "psnr": psnr_val if not np.isnan(psnr_val) else 0.0,
        }
    elif method == "exposure_scaled":
        # Exposure scaled: PSNR, SSIM, MSE only
        return compute_core_metrics(enhanced, long, device)
    else:
        # All other restoration methods: full metrics (PSNR, SSIM, MSE, LPIPS, NIQE)
        return compute_all_metrics(enhanced, long, device)


def filter_metrics_for_json(
    metrics_results: Dict[str, Dict[str, float]], sensor_type: str = "sony"
) -> Dict[str, Dict[str, float]]:
    """
    Filter metrics to include only selected metrics for each method.

    Metric sets:
      - Short: PSNR only
      - Exposure scaled: PSNR, SSIM, MSE
      - All other restoration methods: PSNR, SSIM, MSE, LPIPS, NIQE

    Args:
        metrics_results: Dictionary with metrics for each method
        sensor_type: Sensor type ('sony' or 'fuji') - kept for compatibility and logging

    Returns:
        Filtered dictionary with only selected metrics
    """
    filtered = {}

    for method, metrics in metrics_results.items():
        if method == "short":
            # Short exposure: PSNR only
            filtered[method] = {"psnr": metrics.get("psnr", float("nan"))}
        elif method == "exposure_scaled":
            # Exposure scaled: PSNR, SSIM, MSE only
            filtered[method] = {
                "psnr": metrics.get("psnr", float("nan")),
                "ssim": metrics.get("ssim", float("nan")),
                "mse": metrics.get("mse", float("nan")),
            }
        elif method in [
            "gaussian_x0",
            "gaussian",
            "pg_x0",
            "pg",
            "pg_score",
        ]:
            # All other restoration methods: PSNR, SSIM, MSE, LPIPS, NIQE
            filtered[method] = {
                "psnr": metrics.get("psnr", float("nan")),
                "ssim": metrics.get("ssim", float("nan")),
                "mse": metrics.get("mse", float("nan")),
                "lpips": metrics.get("lpips", float("nan")),
                "niqe": metrics.get("niqe", float("nan")),
            }
        elif method == "long":
            # Skip long - no metrics computed (self-comparison is always perfect)
            continue
        else:
            # Unknown method - include all available metrics
            filtered[method] = metrics

    return filtered


def get_nan_metrics_for_method(method: str) -> Dict[str, float]:
    """Get appropriate NaN metrics dictionary based on method type."""
    base_metrics = {"psnr": float("nan")}
    if method != "short":
        base_metrics.update({"ssim": float("nan"), "mse": float("nan")})
    if method not in ["short", "exposure_scaled"]:
        base_metrics.update({"lpips": float("nan"), "niqe": float("nan")})
    return base_metrics


def get_nan_metrics_dict(include_all: bool = True) -> Dict[str, float]:
    """
    Get a dictionary with NaN values for all metrics.

    Args:
        include_all: If True, include all metrics. If False, only core metrics.

    Returns:
        Dictionary with NaN values for metrics
    """
    metrics = {
        "ssim": float("nan"),
        "psnr": float("nan"),
        "mse": float("nan"),
    }
    if include_all:
        metrics["lpips"] = float("nan")
        metrics["niqe"] = float("nan")
    return metrics


# ============================================================================
# FID Calculator
# ============================================================================


def _setup_edm_for_fid():
    """
    Lazily set up EDM framework for FID computation.
    Only called when FIDCalculator is instantiated.

    Returns:
        True if EDM is available, False otherwise
    """
    global EDM_AVAILABLE

    # Only set up once
    if EDM_AVAILABLE:
        return True

    # Add project root and EDM to path only when needed
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    edm_path = project_root / "external" / "edm"
    if str(edm_path) not in sys.path:
        sys.path.insert(0, str(edm_path))

    # Try to import EDM utilities
    try:
        import external.edm.dnnlib

        EDM_AVAILABLE = True
        return True
    except ImportError as e:
        EDM_AVAILABLE = False
        logger.warning(
            f"EDM framework not available: {e}. FID computation requires EDM dependencies."
        )
        return False


class FIDCalculator:
    """Calculate Fréchet Inception Distance (FID) between two sets of images."""

    def __init__(self, device: torch.device = None):
        """
        Initialize FID calculator with pre-trained InceptionV3 model.

        Args:
            device: Device for computation. If None, auto-detects.
        """
        # Lazily set up EDM framework
        if not _setup_edm_for_fid():
            raise RuntimeError(
                "EDM framework required for NVIDIA Inception model. Please install EDM dependencies."
            )

        # Now import external since it's been set up
        import external.edm.dnnlib

        if device is None:
            from .utils.tensor_utils import get_device

            self.device = get_device(prefer_cuda=True)
        else:
            self.device = device

        # Load NVIDIA pre-trained Inception-v3 model (same as EDM)
        detector_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
        detector_kwargs = dict(return_features=True)

        try:
            with external.edm.dnnlib.util.open_url(detector_url, verbose=True) as f:
                detector_net = pickle.load(f).to(self.device)
            self.detector_net = detector_net
            self.feature_dim = 2048
            self.detector_kwargs = detector_kwargs
        except Exception as e:
            raise RuntimeError(f"Failed to load Inception model: {e}")

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using InceptionV3.

        Args:
            images: Input images [B, C, H, W] in range [0, 1]

        Returns:
            Feature vectors [B, 2048] as torch.Tensor
        """
        with torch.no_grad():
            # Handle grayscale conversion
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])

            # Extract features using NVIDIA model
            features = self.detector_net(
                images.to(self.device), **self.detector_kwargs
            ).to(torch.float64)

            return features

    def calculate_inception_stats(
        self, images: List[torch.Tensor], batch_size: int = 32
    ) -> tuple:
        """
        Calculate Inception statistics.

        Args:
            images: List of image tensors [C, H, W] in range [-1, 1]
            batch_size: Batch size for processing

        Returns:
            (mu, sigma) tuple of mean and covariance matrices
        """
        # Convert images from [-1, 1] to [0, 1]
        images_01 = [(img + 1.0) / 2.0 for img in images]

        # Batch processing
        mu = torch.zeros([self.feature_dim], dtype=torch.float64, device=self.device)
        sigma = torch.zeros(
            [self.feature_dim, self.feature_dim],
            dtype=torch.float64,
            device=self.device,
        )

        for i in tqdm.tqdm(range(0, len(images_01), batch_size), unit="batch"):
            batch = images_01[i : i + batch_size]
            batch_tensor = torch.stack(batch, dim=0)

            # Extract features
            features = self._extract_features(batch_tensor)

            # Accumulate statistics
            mu += features.sum(0)
            sigma += features.T @ features

        # Normalize
        mu /= len(images_01)
        sigma -= mu.ger(mu) * len(images_01)
        sigma /= len(images_01) - 1

        return mu.cpu().numpy(), sigma.cpu().numpy()

    def _calculate_fid_from_stats(
        self, mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
    ) -> float:
        """
        Calculate FID between two sets of statistics (mean and covariance).

        Args:
            mu1: Mean vector from first set [2048]
            sigma1: Covariance matrix from first set [2048, 2048]
            mu2: Mean vector from second set [2048]
            sigma2: Covariance matrix from second set [2048, 2048]

        Returns:
            FID score (lower is better)
        """
        m = np.square(mu1 - mu2).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
        fid = m + np.trace(sigma1 + sigma2 - s * 2)
        return float(np.real(fid))

    def compute_fid(
        self,
        images1: List[torch.Tensor],
        images2: List[torch.Tensor],
        batch_size: int = 32,
    ) -> float:
        """
        Compute FID between two sets of images.

        Args:
            images1: First set of images [C, H, W] in range [-1, 1]
            images2: Second set of images [C, H, W] in range [-1, 1]
            batch_size: Batch size for processing

        Returns:
            FID score (lower is better)
        """
        # Ensure we have enough samples
        if len(images1) < 2 or len(images2) < 2:
            logger.warning(
                f"Insufficient samples for FID: {len(images1)}, {len(images2)}"
            )
            return float("nan")

        # Calculate statistics
        mu1, sigma1 = self.calculate_inception_stats(images1, batch_size=batch_size)
        mu2, sigma2 = self.calculate_inception_stats(images2, batch_size=batch_size)

        # Calculate FID
        fid_score = self._calculate_fid_from_stats(mu1, sigma1, mu2, sigma2)

        return fid_score


def load_images_from_directory(
    directory: Path, prefix: str = "", suffix: str = ".pt", device: torch.device = None
) -> List[torch.Tensor]:
    """
    Load all .pt files from a directory.

    Args:
        directory: Directory containing .pt files
        prefix: Optional prefix to filter files
        suffix: File suffix to match (default: .pt)
        device: Device to load tensors to (optional, not used for loading)

    Returns:
        List of image tensors [C, H, W]
    """
    images = []

    # Find all .pt files
    pt_files = sorted(directory.glob(f"{prefix}*{suffix}"))

    for pt_file in pt_files:
        try:
            try:
                from .utils.data_utils import load_tensor

                tensor = load_tensor(
                    pt_file, device=None, map_location="cpu", weights_only=False
                )
            except Exception as e:
                logger.error(f"Failed to load {pt_file}: {e}")
                continue

            # Handle different tensor formats
            if isinstance(tensor, dict):
                if "restored" in tensor:
                    tensor = tensor["restored"]
                elif "noisy" in tensor:
                    tensor = tensor["noisy"]
                elif "clean" in tensor:
                    tensor = tensor["clean"]
                elif "image" in tensor:
                    tensor = tensor["image"]
                else:
                    raise ValueError(f"Unrecognized dict structure in {pt_file}")

            # Ensure float32
            tensor = tensor.float()

            # Ensure proper shape [C, H, W] or [H, W]
            if tensor.ndim == 2:  # (H, W)
                tensor = tensor.unsqueeze(0)  # (1, H, W)
            elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3]:  # (H, W, C)
                tensor = tensor.permute(2, 0, 1)  # (C, H, W)

            images.append(tensor)

        except Exception as e:
            logger.warning(f"Failed to load {pt_file}: {e}")
            continue

    return images


# ============================================================================
# Physics-Specific Metrics
# ============================================================================


class ResidualAnalyzer:
    """Comprehensive residual analyzer for physics validation.

    This class provides rigorous statistical validation of residuals to verify
    that they follow expected Poisson-Gaussian statistics (N(0,1) when normalized).
    """

    def __init__(self, device: str = "cpu"):
        """Initialize residual analyzer."""
        self.device = device

    def analyze_residuals(
        self,
        pred: torch.Tensor,
        noisy: torch.Tensor,
        scale: float,
        background: float = 0.0,
        read_noise: float = 0.0,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive residual analysis for physics validation.

        Args:
            pred: Predicted clean image [B, C, H, W] (normalized [0,1])
            noisy: Noisy observation [B, C, H, W] (electrons)
            scale: Normalization scale (electrons)
            background: Background offset (electrons)
            read_noise: Read noise standard deviation (electrons)
            mask: Valid pixel mask [B, C, H, W]

        Returns:
            Dictionary containing comprehensive residual statistics
        """
        # Convert prediction to electrons
        pred_electrons = pred * scale + background

        # Compute expected variance under Poisson-Gaussian model
        expected_variance = torch.clamp(pred_electrons + read_noise**2, min=1e-10)

        # Compute normalized residuals: (noisy - pred) / sqrt(pred + σ_r²)
        residuals = noisy - pred_electrons
        normalized_residuals = residuals / torch.sqrt(expected_variance)

        # Apply mask if provided
        if mask is not None:
            normalized_residuals = normalized_residuals * mask
            valid_pixels = mask.sum()
            if valid_pixels == 0:
                raise AnalysisError("No valid pixels for residual analysis")

        # Flatten residuals for analysis
        residuals_flat = normalized_residuals.flatten()
        if mask is not None:
            mask_flat = mask.flatten()
            residuals_flat = residuals_flat[mask_flat > 0.5]

        if len(residuals_flat) < 100:
            raise AnalysisError(
                f"Insufficient valid pixels for analysis: {len(residuals_flat)}"
            )

        # Convert to numpy for statistical tests
        residuals_np = residuals_flat.detach().cpu().numpy()

        # Perform comprehensive statistical analysis
        results = self._perform_statistical_tests(
            residuals_np, pred_electrons, expected_variance, mask
        )

        # Add basic statistics
        results.update(
            {
                "mean": residuals_np.mean(),
                "std_dev": residuals_np.std(),
                "skewness": stats.skew(residuals_np),
                "kurtosis": stats.kurtosis(residuals_np),
                "n_samples": len(residuals_np),
                "scale": scale,
                "background": background,
                "read_noise": read_noise,
            }
        )

        # Add spectral flatness from whiteness analysis
        spectral_analysis = self._compute_spectral_analysis(
            residuals_np, pred_electrons.shape
        )
        results["whiteness_spectral_flatness"] = spectral_analysis.get(
            "spectral_flatness", float("nan")
        )

        return results

    def _perform_statistical_tests(
        self,
        residuals_np: np.ndarray,
        pred_electrons: torch.Tensor,
        expected_variance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical tests on normalized residuals.

        Args:
            residuals_np: Normalized residuals as numpy array
            pred_electrons: Predicted electron counts
            expected_variance: Expected variance under Poisson-Gaussian model
            mask: Valid pixel mask

        Returns:
            Dictionary of statistical test results
        """
        results = {}

        # 1. Kolmogorov-Smirnov test against N(0,1)
        ks_stat, ks_pvalue = stats.kstest(residuals_np, "norm")
        results.update(
            {
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "gaussian_fit": ks_pvalue > 0.05,  # Accept normality at 5% level
            }
        )

        # 2. Spatial correlation analysis
        spatial_corr = self._compute_spatial_correlation(
            residuals_np, pred_electrons.shape
        )
        results.update(spatial_corr)

        # 3. Frequency domain analysis (power spectrum)
        spectral_analysis = self._compute_spectral_analysis(
            residuals_np, pred_electrons.shape
        )
        results.update(spectral_analysis)

        # 4. Additional statistical tests
        # Shapiro-Wilk test (more sensitive than KS for normality)
        try:
            shapiro_stat, shapiro_pvalue = stats.shapiro(
                residuals_np[:5000]
            )  # Limit for computation
            results.update(
                {
                    "shapiro_statistic": shapiro_stat,
                    "shapiro_pvalue": shapiro_pvalue,
                    "normal_by_shapiro": shapiro_pvalue > 0.05,
                }
            )
        except Exception:
            results.update(
                {
                    "shapiro_statistic": float("nan"),
                    "shapiro_pvalue": float("nan"),
                    "normal_by_shapiro": False,
                }
            )

        # 5. Anderson-Darling test (even more sensitive)
        try:
            ad_result = stats.anderson(residuals_np[:5000], dist="norm")
            results.update(
                {
                    "anderson_statistic": ad_result.statistic,
                    "anderson_critical_5": ad_result.critical_values[2],  # 5% level
                    "anderson_significance_5": ad_result.significance_level[2],
                    "normal_by_anderson": ad_result.statistic
                    < ad_result.critical_values[2],
                }
            )
        except Exception:
            results.update(
                {
                    "anderson_statistic": float("nan"),
                    "anderson_critical_5": float("nan"),
                    "anderson_significance_5": float("nan"),
                    "normal_by_anderson": False,
                }
            )

        # 6. Ljung-Box test for autocorrelation
        try:
            lb_stat, lb_pvalue = stats.acorr_ljungbox(residuals_np, lags=[1, 5, 10])
            results.update(
                {
                    "ljung_box_statistic": lb_stat.iloc[-1],  # Last lag
                    "ljung_box_pvalue": lb_pvalue.iloc[-1],
                    "white_noise_by_ljung_box": lb_pvalue.iloc[-1] > 0.05,
                }
            )
        except Exception:
            results.update(
                {
                    "ljung_box_statistic": float("nan"),
                    "ljung_box_pvalue": float("nan"),
                    "white_noise_by_ljung_box": False,
                }
            )

        return results

    def _compute_spatial_correlation(
        self, residuals_np: np.ndarray, image_shape: torch.Size
    ) -> Dict[str, Any]:
        """
        Compute spatial correlation of residuals.

        Args:
            residuals_np: Residuals as flattened numpy array
            image_shape: Shape of the original image tensor

        Returns:
            Dictionary of spatial correlation metrics
        """
        try:
            # Reshape to 2D for spatial analysis
            h, w = image_shape[-2], image_shape[-1]
            if len(residuals_np) < h * w:
                # Pad if necessary (though this shouldn't happen with proper masking)
                residuals_2d = np.zeros((h, w))
                residuals_2d.flat[: len(residuals_np)] = residuals_np
            else:
                residuals_2d = residuals_np[: h * w].reshape(h, w)

            # Compute autocorrelation at lag 1 in both directions
            autocorr_x = np.corrcoef(
                residuals_2d[:, :-1].flatten(), residuals_2d[:, 1:].flatten()
            )[0, 1]
            autocorr_y = np.corrcoef(
                residuals_2d[:-1, :].flatten(), residuals_2d[1:, :].flatten()
            )[0, 1]

            # Compute autocorrelation matrix for first few lags
            max_lag = min(5, min(h, w) // 2)
            autocorr_matrix = np.zeros((max_lag, max_lag))

            for i in range(max_lag):
                for j in range(max_lag):
                    if i == 0 and j == 0:
                        autocorr_matrix[i, j] = 1.0
                    else:
                        shifted = np.roll(residuals_2d, (i, j), axis=(0, 1))
                        mask = np.ones_like(residuals_2d)
                        mask = np.roll(mask, (i, j), axis=(0, 1))

                        # Only correlate where both pixels are valid
                        valid_mask = mask == 1
                        if valid_mask.sum() > 100:
                            autocorr_matrix[i, j] = np.corrcoef(
                                residuals_2d[valid_mask].flatten(),
                                shifted[valid_mask].flatten(),
                            )[0, 1]
                        else:
                            autocorr_matrix[i, j] = float("nan")

            return {
                "autocorrelation_lag1_x": autocorr_x,
                "autocorrelation_lag1_y": autocorr_y,
                "spatial_uncorrelated": abs(autocorr_x) < 0.1 and abs(autocorr_y) < 0.1,
                "autocorrelation_matrix": autocorr_matrix,
            }
        except Exception as e:
            logger.warning(f"Spatial correlation analysis failed: {e}")
            return {
                "autocorrelation_lag1_x": float("nan"),
                "autocorrelation_lag1_y": float("nan"),
                "spatial_uncorrelated": False,
                "autocorrelation_matrix": None,
            }

    def _compute_spectral_analysis(
        self, residuals_np: np.ndarray, image_shape: torch.Size
    ) -> Dict[str, Any]:
        """
        Compute spectral analysis of residuals.

        Args:
            residuals_np: Residuals as flattened numpy array
            image_shape: Shape of the original image tensor

        Returns:
            Dictionary of spectral analysis metrics
        """
        try:
            # Reshape to 2D for spectral analysis
            h, w = image_shape[-2], image_shape[-1]
            if len(residuals_np) < h * w:
                residuals_2d = np.zeros((h, w))
                residuals_2d.flat[: len(residuals_np)] = residuals_np
            else:
                residuals_2d = residuals_np[: h * w].reshape(h, w)

            # Compute 2D power spectral density
            # Use periodogram for each row and average
            row_psds = []
            for row in residuals_2d:
                if np.std(row) > 1e-10:  # Skip rows with no variation
                    freqs, psd = periodogram(
                        row, fs=1.0
                    )  # fs=1 since we're working in pixels
                    row_psds.append(psd)

            if not row_psds:
                return {
                    "spectral_flatness": float("nan"),
                    "spectral_slope": float("nan"),
                    "high_freq_power": float("nan"),
                    "white_spectrum": False,
                }

            # Average PSD across rows
            avg_psd = np.mean(row_psds, axis=0)

            # Skip DC component for analysis
            freqs = freqs[1:]
            avg_psd = avg_psd[1:]

            if len(avg_psd) < 10:
                return {
                    "spectral_flatness": float("nan"),
                    "spectral_slope": float("nan"),
                    "high_freq_power": float("nan"),
                    "white_spectrum": False,
                }

            # Compute spectral flatness (Wiener entropy)
            # White noise should have flat spectrum (value close to 1)
            spectral_flatness = np.exp(np.mean(np.log(avg_psd + 1e-10))) / np.mean(
                avg_psd
            )

            # Compute spectral slope (should be ~0 for white noise)
            # Use log-log regression on frequencies > 0.1 * Nyquist
            valid_idx = freqs > 0.1 * np.max(freqs)
            if np.sum(valid_idx) > 5:
                log_freqs = np.log(freqs[valid_idx])
                log_psd = np.log(avg_psd[valid_idx] + 1e-10)
                slope, _ = np.polyfit(log_freqs, log_psd, 1)
            else:
                slope = float("nan")

            # Compute high-frequency power (should be significant for white noise)
            high_freq_idx = freqs > 0.5 * np.max(freqs)
            high_freq_power = (
                np.mean(avg_psd[high_freq_idx]) if np.any(high_freq_idx) else 0.0
            )

            # White spectrum criteria
            white_spectrum = (
                spectral_flatness > 0.8
                and abs(slope) < 0.5  # High flatness
                and high_freq_power  # Small slope
                > 0.1  # Significant high-frequency content
            )

            return {
                "spectral_flatness": spectral_flatness,
                "spectral_slope": slope,
                "high_freq_power": high_freq_power,
                "white_spectrum": white_spectrum,
                "frequency_range": (freqs.min(), freqs.max()),
            }
        except Exception as e:
            logger.warning(f"Spectral analysis failed: {e}")
            return {
                "spectral_flatness": float("nan"),
                "spectral_slope": float("nan"),
                "high_freq_power": float("nan"),
                "white_spectrum": False,
                "frequency_range": (float("nan"), float("nan")),
            }


@dataclass
class MetricResult:
    """Container for metric computation results."""

    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report for a restoration method."""

    method_name: str
    dataset_name: str
    sensor: str

    # Physics metrics
    chi2_consistency: MetricResult
    residual_distribution: MetricResult  # N(0,1) test
    residual_whiteness: MetricResult  # Spectral flatness
    bias_analysis: MetricResult

    # Standard image quality metrics (optional)
    image_quality_metrics: Optional[Dict[str, float]] = None

    # Sensor-specific metrics
    sensor_metrics: Dict[str, MetricResult]

    # Summary statistics
    num_images: int
    processing_time: float

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationReport":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        # Convert nested dicts back to MetricResult objects
        for key in [
            "chi2_consistency",
            "residual_distribution",
            "residual_whiteness",
            "bias_analysis",
        ]:
            if key in data:
                data[key] = MetricResult(**data[key])

        # Handle sensor field
        if "sensor" in data:
            data["sensor"] = data["sensor"]

        if "sensor_metrics" in data:
            data["sensor_metrics"] = {
                k: MetricResult(**v) for k, v in data["sensor_metrics"].items()
            }

        return cls(**data)


class PhysicsMetrics:
    """Physics-specific metrics for Poisson-Gaussian restoration."""

    def __init__(self):
        """Initialize physics metrics."""
        pass

    def compute_chi2_consistency(
        self,
        pred: torch.Tensor,
        noisy: torch.Tensor,
        scale: float,
        background: float = 0.0,
        read_noise: float = 0.0,
        mask: Optional[torch.Tensor] = None,
    ) -> MetricResult:
        """
        Compute χ² consistency metric.

        Tests whether residuals follow expected Poisson-Gaussian statistics.
        Good restoration should have χ² ≈ 1.0.

        Args:
            pred: Predicted clean image [B, C, H, W] (normalized [0,1])
            noisy: Noisy observation [B, C, H, W] (electrons)
            scale: Normalization scale (electrons)
            background: Background offset (electrons)
            read_noise: Read noise standard deviation (electrons)
            mask: Valid pixel mask [B, C, H, W]

        Returns:
            χ² consistency metric result
        """
        # Convert prediction to electrons
        pred_electrons = pred * scale + background

        # Compute expected variance under Poisson-Gaussian model
        variance = torch.clamp(pred_electrons + read_noise**2, min=0.1)

        # Compute residuals
        residuals = noisy - pred_electrons

        # Compute χ² per pixel
        chi2_per_pixel = (residuals**2) / variance

        # Apply mask if provided
        if mask is not None:
            chi2_per_pixel = chi2_per_pixel * mask
            valid_pixels = mask.sum()
            if valid_pixels == 0:
                return MetricResult(
                    value=float("nan"), metadata={"error": "No valid pixels"}
                )
            chi2_mean = chi2_per_pixel.sum() / valid_pixels
        else:
            chi2_mean = chi2_per_pixel.mean()

        chi2_value = chi2_mean.item()

        # Compute statistics
        chi2_flat = chi2_per_pixel.flatten()
        if mask is not None:
            mask_flat = mask.flatten()
            chi2_flat = chi2_flat[mask_flat > 0.5]

        chi2_std = chi2_flat.std().item()
        chi2_median = chi2_flat.median().item()

        # Goodness of fit test (Kolmogorov-Smirnov against χ² distribution)
        chi2_np = chi2_flat.detach().cpu().numpy()
        ks_stat, ks_pvalue = stats.kstest(chi2_np, lambda x: stats.chi2.cdf(x, df=1))

        return MetricResult(
            value=chi2_value,
            metadata={
                "std": chi2_std,
                "median": chi2_median,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "scale": scale,
                "background": background,
                "read_noise": read_noise,
            },
        )

    def compute_residual_whiteness(
        self,
        pred: torch.Tensor,
        noisy: torch.Tensor,
        scale: float,
        background: float = 0.0,
        mask: Optional[torch.Tensor] = None,
    ) -> MetricResult:
        """
        Compute residual whiteness metric.

        Tests whether residuals are white noise (no spatial structure).
        Good restoration should have flat power spectrum.

        Args:
            pred: Predicted clean image [B, C, H, W] (normalized [0,1])
            noisy: Noisy observation [B, C, H, W] (electrons)
            scale: Normalization scale (electrons)
            background: Background offset (electrons)
            mask: Valid pixel mask [B, C, H, W]

        Returns:
            Residual whiteness metric result
        """
        # Convert prediction to electrons
        pred_electrons = pred * scale + background

        # Compute residuals
        residuals = noisy - pred_electrons

        # Apply mask if provided
        if mask is not None:
            residuals = residuals * mask

        whiteness_scores = []

        for b in range(residuals.shape[0]):
            for c in range(residuals.shape[1]):
                residual_img = residuals[b, c].detach().cpu().numpy()

                # Skip if too small for meaningful analysis
                if residual_img.size < 64 * 64:
                    continue

                # Compute power spectral density
                try:
                    freqs, psd = periodogram(residual_img.flatten())

                    # Skip DC component
                    freqs = freqs[1:]
                    psd = psd[1:]

                    if len(psd) < 10:
                        continue

                    # Compute flatness of power spectrum
                    # White noise should have flat spectrum
                    log_psd = np.log(psd + 1e-10)

                    # Measure deviation from flatness
                    # Use coefficient of variation of log PSD
                    whiteness = 1.0 / (
                        1.0 + np.std(log_psd) / (np.abs(np.mean(log_psd)) + 1e-10)
                    )
                    whiteness_scores.append(whiteness)

                except Exception as e:
                    logger.warning(f"Failed to compute whiteness: {e}")
                    continue

        if not whiteness_scores:
            return MetricResult(
                value=float("nan"), metadata={"error": "Could not compute whiteness"}
            )

        mean_whiteness = np.mean(whiteness_scores)
        std_whiteness = np.std(whiteness_scores)

        return MetricResult(
            value=mean_whiteness,
            metadata={
                "std": std_whiteness,
                "num_samples": len(whiteness_scores),
                "scale": scale,
                "background": background,
            },
        )

    def compute_bias_analysis(
        self,
        pred: torch.Tensor,
        noisy: torch.Tensor,
        scale: float,
        background: float = 0.0,
        mask: Optional[torch.Tensor] = None,
    ) -> MetricResult:
        """
        Compute bias analysis metric.

        Tests whether the restoration is unbiased (residuals have zero mean).
        Good restoration should have bias < 1% of signal level.

        Args:
            pred: Predicted clean image [B, C, H, W] (normalized [0,1])
            noisy: Noisy observation [B, C, H, W] (electrons)
            scale: Normalization scale (electrons)
            background: Background offset (electrons)
            mask: Valid pixel mask [B, C, H, W]

        Returns:
            Bias analysis metric result
        """
        # Convert prediction to electrons
        pred_electrons = pred * scale + background

        # Compute residuals
        residuals = noisy - pred_electrons

        # Apply mask if provided
        if mask is not None:
            residuals = residuals * mask
            valid_pixels = mask.sum()
            if valid_pixels == 0:
                return MetricResult(
                    value=float("nan"), metadata={"error": "No valid pixels"}
                )
            bias = residuals.sum() / valid_pixels
        else:
            bias = residuals.mean()

        bias_value = bias.item()

        # Compute relative bias (as percentage of mean signal)
        if mask is not None:
            mean_signal = (pred_electrons * mask).sum() / valid_pixels
        else:
            mean_signal = pred_electrons.mean()

        relative_bias = (bias_value / (mean_signal.item() + 1e-10)) * 100

        # Compute confidence interval for bias
        residuals_flat = residuals.flatten()
        if mask is not None:
            mask_flat = mask.flatten()
            residuals_flat = residuals_flat[mask_flat > 0.5]

        residuals_np = residuals_flat.detach().cpu().numpy()
        n_samples = len(residuals_np)

        if n_samples > 1:
            std_error = np.std(residuals_np) / np.sqrt(n_samples)
            ci_low = bias_value - 1.96 * std_error
            ci_high = bias_value + 1.96 * std_error
            ci = (ci_low, ci_high)
        else:
            ci = None

        # Statistical test for zero bias
        if n_samples > 30:
            t_stat = bias_value / (std_error + 1e-10)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_samples - 1))
        else:
            t_stat = float("nan")
            p_value = float("nan")

        return MetricResult(
            value=abs(relative_bias),  # Return absolute relative bias as percentage
            confidence_interval=ci,
            metadata={
                "bias_electrons": bias_value,
                "relative_bias_percent": relative_bias,
                "mean_signal": mean_signal.item(),
                "t_statistic": t_stat,
                "p_value": p_value,
                "n_samples": n_samples,
                "scale": scale,
                "background": background,
            },
        )


class SensorSpecificMetrics:
    """Sensor-specific metrics for imaging applications."""

    def __init__(self):
        """Initialize sensor-specific metrics."""
        pass

    def compute_resolution_metric(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> MetricResult:
        """
        Compute resolution preservation metric.

        Measures how well fine details are preserved in the restoration.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth image [B, C, H, W]
            mask: Valid pixel mask [B, C, H, W]

        Returns:
            Resolution metric result
        """
        # Compute high-frequency content preservation
        # Use Laplacian filter to extract high frequencies
        laplacian_kernel = torch.tensor(
            [[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]],
            dtype=pred.dtype,
            device=pred.device,
        )

        # Apply Laplacian filter
        pred_hf = F.conv2d(pred, laplacian_kernel, padding=1)
        target_hf = F.conv2d(target, laplacian_kernel, padding=1)

        # Apply mask if provided
        if mask is not None:
            pred_hf = pred_hf * mask
            target_hf = target_hf * mask

        # Compute correlation between high-frequency components
        pred_hf_flat = pred_hf.flatten()
        target_hf_flat = target_hf.flatten()

        # Remove masked pixels
        if mask is not None:
            mask_flat = mask.flatten()
            valid_idx = mask_flat > 0.5
            pred_hf_flat = pred_hf_flat[valid_idx]
            target_hf_flat = target_hf_flat[valid_idx]

        if len(pred_hf_flat) < 10:
            return MetricResult(
                value=float("nan"), metadata={"error": "Insufficient valid pixels"}
            )

        # Compute correlation coefficient
        pred_np = pred_hf_flat.detach().cpu().numpy()
        target_np = target_hf_flat.detach().cpu().numpy()

        correlation = np.corrcoef(pred_np, target_np)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # Compute power ratio
        pred_power = np.var(pred_np)
        target_power = np.var(target_np)

        if target_power > 1e-10:
            power_ratio = pred_power / target_power
        else:
            power_ratio = 1.0 if pred_power < 1e-10 else float("inf")

        # Combined resolution metric
        resolution_score = correlation * min(1.0, power_ratio)

        return MetricResult(
            value=resolution_score,
            metadata={
                "correlation": correlation,
                "power_ratio": power_ratio,
                "pred_power": pred_power,
                "target_power": target_power,
            },
        )


class EvaluationSuite:
    """
    Comprehensive evaluation suite for Poisson-Gaussian diffusion restoration.

    This is the main class that orchestrates all metric computations and
    provides a unified interface for evaluation.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize evaluation suite.

        Args:
            device: Device for computation ('cuda' or 'cpu')
        """
        self.device = device
        self.physics_metrics = PhysicsMetrics()
        self.sensor_metrics = SensorSpecificMetrics()
        self.residual_analyzer = ResidualAnalyzer(device)

        logger.info(f"EvaluationSuite initialized on device: {device}")

    def evaluate_restoration(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        noisy: torch.Tensor,
        scale: float,
        sensor: str,
        background: float = 0.0,
        read_noise: float = 0.0,
        mask: Optional[torch.Tensor] = None,
        method_name: str = "unknown",
        dataset_name: str = "unknown",
        sensor_specific_params: Optional[Dict[str, Any]] = None,
        compute_image_quality: bool = True,
    ) -> EvaluationReport:
        """
        Perform comprehensive evaluation of a restoration method.

        Args:
            pred: Predicted clean image [B, C, H, W] (normalized [0,1])
            target: Ground truth clean image [B, C, H, W] (normalized [0,1])
            noisy: Noisy observation [B, C, H, W] (electrons)
            scale: Normalization scale (electrons)
            sensor: Sensor name ('sony' or 'fuji')
            background: Background offset (electrons)
            read_noise: Read noise standard deviation (electrons)
            mask: Valid pixel mask [B, C, H, W]
            method_name: Name of the restoration method
            dataset_name: Name of the dataset
            sensor_specific_params: Additional parameters for sensor-specific metrics
            compute_image_quality: If True, compute standard image quality metrics

        Returns:
            Complete evaluation report
        """
        start_time = time.time()

        logger.info(f"Evaluating {method_name} on {dataset_name} ({sensor})")

        # Convert to electron space for consistent metrics
        pred_electrons = pred * scale + background
        target_electrons = target * scale + background

        # Physics metrics
        logger.debug("Computing physics metrics...")
        chi2 = self.physics_metrics.compute_chi2_consistency(
            pred, noisy, scale, background, read_noise, mask
        )

        # Residual analysis
        try:
            residual_stats = self.residual_analyzer.analyze_residuals(
                pred, noisy, scale, background, read_noise, mask
            )
            dist_meta = {
                "mean": residual_stats["mean"],
                "std_dev": residual_stats["std_dev"],
                "ks_statistic": residual_stats["ks_statistic"],
                "ks_pvalue": residual_stats["ks_pvalue"],
            }
            residual_dist = MetricResult(
                value=dist_meta["ks_statistic"], metadata=dist_meta
            )

            whiteness_meta = {
                "spectral_flatness": residual_stats.get(
                    "whiteness_spectral_flatness", float("nan")
                ),
                "autocorr_x": residual_stats.get(
                    "autocorrelation_lag1_x", float("nan")
                ),
                "autocorr_y": residual_stats.get(
                    "autocorrelation_lag1_y", float("nan")
                ),
            }
            residual_whiteness = MetricResult(
                value=whiteness_meta["spectral_flatness"], metadata=whiteness_meta
            )

            bias = MetricResult(
                value=abs(residual_stats["mean"]),
                metadata={"mean_residual": residual_stats["mean"]},
            )

        except AnalysisError as e:
            logger.warning(f"Residual analysis failed: {e}")
            nan_result = MetricResult(value=float("nan"), metadata={"error": str(e)})
            residual_dist = nan_result
            residual_whiteness = nan_result
            bias = nan_result

        # Standard image quality metrics (optional)
        image_quality_metrics = None
        if compute_image_quality:
            logger.debug("Computing standard image quality metrics...")
            try:
                # Convert to [-1, 1] range for standard metrics
                from .normalization import convert_range

                pred_norm = convert_range(pred, "[0,1]", "[-1,1]")
                target_norm = convert_range(target, "[0,1]", "[-1,1]")
                image_quality_metrics = compute_all_metrics(
                    pred_norm, target_norm, device=self.device
                )
            except Exception as e:
                logger.warning(f"Image quality metrics computation failed: {e}")
                image_quality_metrics = None

        # Sensor-specific metrics (sony/fuji)
        logger.debug("Computing sensor-specific metrics...")
        sensor_metrics_dict = {}

        # Resolution preservation
        resolution = self.sensor_metrics.compute_resolution_metric(
            pred, target, mask=mask
        )
        sensor_metrics_dict["resolution_preservation"] = resolution

        processing_time = time.time() - start_time

        # Create evaluation report
        report = EvaluationReport(
            method_name=method_name,
            dataset_name=dataset_name,
            sensor=sensor,
            chi2_consistency=chi2,
            residual_distribution=residual_dist,
            residual_whiteness=residual_whiteness,
            bias_analysis=bias,
            image_quality_metrics=image_quality_metrics,
            sensor_metrics=sensor_metrics_dict,
            num_images=pred.shape[0],
            processing_time=processing_time,
        )

        logger.info(f"Evaluation completed in {processing_time:.2f}s")
        logger.info(f"χ²: {chi2.value:.3f}, Res-KS: {residual_dist.value:.3f}")

        return report

    def compare_methods(
        self, reports: List[EvaluationReport], output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple restoration methods.

        Args:
            reports: List of evaluation reports to compare
            output_file: Optional file to save comparison results

        Returns:
            Comparison results dictionary
        """
        if not reports:
            raise ValueError("No reports provided for comparison")

        # Group reports by dataset and sensor
        grouped_reports = {}
        for report in reports:
            sensor = getattr(report, "sensor", "unknown")
            key = f"{report.dataset_name}_{sensor}"
            if key not in grouped_reports:
                grouped_reports[key] = []
            grouped_reports[key].append(report)

        comparison_results = {}

        for key, group_reports in grouped_reports.items():
            # Split on last underscore to handle dataset names with underscores
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                dataset_name, sensor = parts
            else:
                dataset_name, sensor = key, "unknown"

            # Extract metrics for comparison
            methods = [r.method_name for r in group_reports]

            metrics_comparison = {
                "methods": methods,
                "chi2_consistency": [r.chi2_consistency.value for r in group_reports],
                "residual_distribution": [
                    r.residual_distribution.value for r in group_reports
                ],
                "residual_whiteness": [
                    r.residual_whiteness.value for r in group_reports
                ],
                "bias_analysis": [r.bias_analysis.value for r in group_reports],
            }

            # Add image quality metrics if available
            if any(r.image_quality_metrics for r in group_reports):
                for metric_name in ["psnr", "ssim", "mse", "lpips", "niqe"]:
                    values = []
                    for r in group_reports:
                        if (
                            r.image_quality_metrics
                            and metric_name in r.image_quality_metrics
                        ):
                            values.append(r.image_quality_metrics[metric_name])
                        else:
                            values.append(float("nan"))
                    metrics_comparison[metric_name] = values

            # Add sensor-specific metrics
            for report in group_reports:
                sensor_metrics = getattr(
                    report, "sensor_metrics", getattr(report, "domain_metrics", {})
                )
                for metric_name, metric_result in sensor_metrics.items():
                    if metric_name not in metrics_comparison:
                        metrics_comparison[metric_name] = []
                    metrics_comparison[metric_name].append(metric_result.value)

            # Find best method for each metric
            best_methods = {}
            for metric_name, values in metrics_comparison.items():
                if metric_name == "methods":
                    continue

                valid_values = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
                if not valid_values:
                    best_methods[metric_name] = "N/A"
                    continue

                if metric_name in [
                    "chi2_consistency",
                    "bias_analysis",
                    "residual_distribution",
                    "mse",
                    "lpips",
                    "niqe",
                ]:
                    best_idx = min(valid_values, key=lambda x: x[1])[0]
                else:
                    best_idx = max(valid_values, key=lambda x: x[1])[0]

                best_methods[metric_name] = methods[best_idx]

            comparison_results[key] = {
                "dataset": dataset_name,
                "sensor": sensor,
                "metrics": metrics_comparison,
                "best_methods": best_methods,
            }

        # Save results if requested
        if output_file:
            with open(output_file, "w") as f:
                json.dump(comparison_results, f, indent=2, default=str)
            logger.info(f"Comparison results saved to {output_file}")

        return comparison_results

    def generate_summary_statistics(
        self, reports: List[EvaluationReport]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics across multiple evaluations.

        Args:
            reports: List of evaluation reports

        Returns:
            Summary statistics dictionary
        """
        if not reports:
            return {}

        grouped = {}
        for report in reports:
            sensor = getattr(report, "sensor", "unknown")
            key = f"{report.method_name}_{sensor}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(report)

        summary = {}

        for key, group_reports in grouped.items():
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                method_name, sensor = parts
            else:
                method_name, sensor = key, "unknown"

            chi2_values = [
                r.chi2_consistency.value
                for r in group_reports
                if not np.isnan(r.chi2_consistency.value)
            ]
            residual_ks_values = [
                r.residual_distribution.value
                for r in group_reports
                if not np.isnan(r.residual_distribution.value)
            ]

            summary[key] = {
                "method": method_name,
                "sensor": sensor,
                "num_evaluations": len(group_reports),
                "chi2_stats": {
                    "mean": np.mean(chi2_values) if chi2_values else float("nan"),
                    "std": np.std(chi2_values) if chi2_values else float("nan"),
                    "min": np.min(chi2_values) if chi2_values else float("nan"),
                    "max": np.max(chi2_values) if chi2_values else float("nan"),
                },
                "residual_ks_stats": {
                    "mean": np.mean(residual_ks_values)
                    if residual_ks_values
                    else float("nan"),
                    "std": np.std(residual_ks_values)
                    if residual_ks_values
                    else float("nan"),
                    "min": np.min(residual_ks_values)
                    if residual_ks_values
                    else float("nan"),
                    "max": np.max(residual_ks_values)
                    if residual_ks_values
                    else float("nan"),
                },
                "avg_processing_time": np.mean(
                    [r.processing_time for r in group_reports]
                ),
            }

        return summary


# ============================================================================
# Utility Functions
# ============================================================================


def load_baseline_results(baseline_dir: str) -> Dict[str, List[EvaluationReport]]:
    """
    Load baseline evaluation results from directory.

    Args:
        baseline_dir: Directory containing baseline result JSON files

    Returns:
        Dictionary mapping method names to lists of evaluation reports
    """
    import glob
    import os

    baseline_results = {}

    for json_file in glob.glob(os.path.join(baseline_dir, "*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Handle both single reports and lists of reports
            if isinstance(data, list):
                reports = [
                    EvaluationReport.from_json(json.dumps(item)) for item in data
                ]
            else:
                reports = [EvaluationReport.from_json(json.dumps(data))]

            method_name = reports[0].method_name
            if method_name not in baseline_results:
                baseline_results[method_name] = []
            baseline_results[method_name].extend(reports)

        except Exception as e:
            logger.warning(f"Failed to load baseline results from {json_file}: {e}")

    return baseline_results


def save_evaluation_results(reports: List[EvaluationReport], output_file: str) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        reports: List of evaluation reports
        output_file: Output JSON file path
    """
    data = [json.loads(report.to_json()) for report in reports]

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(reports)} evaluation results to {output_file}")


# ============================================================================
# Command-line Interface for FID
# ============================================================================


def main():
    """Main function for FID computation."""
    parser = argparse.ArgumentParser(
        description="Compute FID between two sets of images"
    )

    # Input arguments
    parser.add_argument(
        "--restored_dir",
        type=str,
        required=True,
        help="Directory containing restored/generated images (.pt files)",
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        required=True,
        help="Directory containing clean reference images (.pt files)",
    )
    parser.add_argument(
        "--restored_prefix",
        type=str,
        default="restored_",
        help="Prefix for restored image files (default: restored_)",
    )
    parser.add_argument(
        "--clean_prefix",
        type=str,
        default="",
        help="Prefix for clean image files (default: empty)",
    )

    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default="fid_results.json",
        help="Output JSON file for FID results",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (cuda or cpu)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for feature extraction"
    )

    args = parser.parse_args()

    # Setup
    restored_dir = Path(args.restored_dir)
    clean_dir = Path(args.clean_dir)
    output_file = Path(args.output_file)

    from .utils.file_utils import save_json_file
    from .utils.tensor_utils import get_device

    device = get_device(prefer_cuda=(args.device == "cuda"))

    # Load images
    restored_images = load_images_from_directory(
        restored_dir, prefix=args.restored_prefix
    )

    clean_images = load_images_from_directory(clean_dir, prefix=args.clean_prefix)

    if len(restored_images) == 0:
        logger.error("No restored images found!")
        return

    if len(clean_images) == 0:
        logger.error("No clean images found!")
        return

    logger.info(
        f"Loaded {len(restored_images)} restored images and {len(clean_images)} clean images"
    )

    # Initialize FID calculator
    fid_calculator = FIDCalculator(device=device)

    # Compute FID
    fid_score = fid_calculator.compute_fid(
        restored_images,
        clean_images,
        batch_size=args.batch_size,
    )

    # Save results
    results = {
        "fid_score": float(fid_score),
        "num_restored_images": len(restored_images),
        "num_clean_images": len(clean_images),
        "restored_dir": str(restored_dir),
        "clean_dir": str(clean_dir),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_json_file(output_file, results)

    print(f"FID Score: {fid_score:.4f}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
