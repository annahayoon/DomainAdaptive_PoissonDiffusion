"""
Comprehensive evaluation metrics for Poisson-Gaussian diffusion restoration.

This module implements:
1. Standard image quality metrics (PSNR, SSIM, LPIPS, MS-SSIM)
2. Physics-specific metrics (χ² consistency, residual whiteness, bias analysis)
3. Domain-specific metrics (counting accuracy, photometry error)
4. Baseline comparison utilities

Requirements addressed: 6.3-6.6 from requirements.md
"""

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.signal import periodogram
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import structural_similarity as ssim_skimage

from .exceptions import AnalysisError
from .residual_analysis import ResidualAnalyzer

# Set up logging
logger = logging.getLogger(__name__)


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
    domain: str

    # Standard metrics
    psnr: MetricResult
    ssim: MetricResult
    lpips: MetricResult
    ms_ssim: MetricResult

    # Physics metrics
    chi2_consistency: MetricResult
    residual_distribution: MetricResult  # N(0,1) test
    residual_whiteness: MetricResult  # Spectral flatness
    bias_analysis: MetricResult

    # Domain-specific metrics
    domain_metrics: Dict[str, MetricResult]

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
            "psnr",
            "ssim",
            "lpips",
            "ms_ssim",
            "chi2_consistency",
            "residual_distribution",
            "residual_whiteness",
            "bias_analysis",
        ]:
            if key in data:
                data[key] = MetricResult(**data[key])

        if "domain_metrics" in data:
            data["domain_metrics"] = {
                k: MetricResult(**v) for k, v in data["domain_metrics"].items()
            }

        return cls(**data)


class StandardMetrics:
    """Standard image quality metrics implementation."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize standard metrics.

        Args:
            device: Device for computation ('cuda' or 'cpu')
        """
        self.device = device
        self._lpips_net = None

    def _get_lpips_net(self):
        """Lazy loading of LPIPS network."""
        if self._lpips_net is None:
            try:
                import lpips

                self._lpips_net = lpips.LPIPS(net="alex").to(self.device)
                logger.info("LPIPS network loaded successfully")
            except ImportError:
                logger.warning("LPIPS not available. Install with: pip install lpips")
                self._lpips_net = None
        return self._lpips_net

    def compute_psnr(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: float = 1.0,
        mask: Optional[torch.Tensor] = None,
    ) -> MetricResult:
        """
        Compute Peak Signal-to-Noise Ratio.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth image [B, C, H, W]
            data_range: Maximum possible pixel value
            mask: Valid pixel mask [B, C, H, W]

        Returns:
            PSNR metric result
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        # Apply mask if provided
        if mask is not None:
            pred = pred * mask
            target = target * mask
            valid_pixels = mask.sum()
            if valid_pixels == 0:
                return MetricResult(value=0.0, metadata={"error": "No valid pixels"})

        # Compute MSE
        mse = F.mse_loss(pred, target, reduction="mean")

        # Handle perfect reconstruction
        if mse.item() < 1e-10:
            psnr_value = 100.0  # Very high PSNR for near-perfect reconstruction
        else:
            psnr_value = 20 * torch.log10(data_range / torch.sqrt(mse))
            psnr_value = psnr_value.item()

        # Compute confidence interval (bootstrap estimate)
        if pred.numel() > 1000:  # Only for reasonably sized images
            ci = self._bootstrap_psnr_ci(pred, target, data_range, mask)
        else:
            ci = None

        return MetricResult(
            value=psnr_value,
            confidence_interval=ci,
            metadata={"mse": mse.item(), "data_range": data_range},
        )

    def compute_ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: float = 1.0,
        mask: Optional[torch.Tensor] = None,
    ) -> MetricResult:
        """
        Compute Structural Similarity Index.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth image [B, C, H, W]
            data_range: Maximum possible pixel value
            mask: Valid pixel mask [B, C, H, W]

        Returns:
            SSIM metric result
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        # Convert to numpy for skimage SSIM
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        ssim_values = []

        for b in range(pred.shape[0]):
            for c in range(pred.shape[1]):
                pred_img = pred_np[b, c]
                target_img = target_np[b, c]

                # Apply mask if provided
                if mask is not None:
                    mask_img = mask[b, c].detach().cpu().numpy()
                    # For SSIM, we need to handle masked regions carefully
                    # Set masked regions to mean value to minimize impact
                    mean_val = (
                        target_img[mask_img > 0.5].mean()
                        if (mask_img > 0.5).any()
                        else 0.5
                    )
                    pred_img = pred_img * mask_img + mean_val * (1 - mask_img)
                    target_img = target_img * mask_img + mean_val * (1 - mask_img)

                # Compute SSIM
                ssim_val = ssim_skimage(
                    target_img,
                    pred_img,
                    data_range=data_range,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                )
                ssim_values.append(ssim_val)

        mean_ssim = np.mean(ssim_values)
        std_ssim = np.std(ssim_values) if len(ssim_values) > 1 else 0.0

        # Confidence interval
        if len(ssim_values) > 1:
            ci_low = mean_ssim - 1.96 * std_ssim / np.sqrt(len(ssim_values))
            ci_high = mean_ssim + 1.96 * std_ssim / np.sqrt(len(ssim_values))
            ci = (ci_low, ci_high)
        else:
            ci = None

        return MetricResult(
            value=mean_ssim,
            confidence_interval=ci,
            metadata={"std": std_ssim, "num_channels": len(ssim_values)},
        )

    def compute_lpips(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> MetricResult:
        """
        Compute Learned Perceptual Image Patch Similarity.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth image [B, C, H, W]
            mask: Valid pixel mask [B, C, H, W]

        Returns:
            LPIPS metric result
        """
        lpips_net = self._get_lpips_net()
        if lpips_net is None:
            return MetricResult(
                value=float("nan"), metadata={"error": "LPIPS not available"}
            )

        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        # Ensure images are in [-1, 1] range for LPIPS
        pred_norm = pred * 2.0 - 1.0
        target_norm = target * 2.0 - 1.0

        # Handle single channel images by replicating to 3 channels
        if pred.shape[1] == 1:
            pred_norm = pred_norm.repeat(1, 3, 1, 1)
            target_norm = target_norm.repeat(1, 3, 1, 1)

        pred_norm = pred_norm.to(self.device)
        target_norm = target_norm.to(self.device)

        with torch.no_grad():
            lpips_values = lpips_net(pred_norm, target_norm)

        # Apply mask weighting if provided
        if mask is not None:
            mask = mask.to(self.device)
            # Average mask over channels if multi-channel
            if mask.shape[1] > 1:
                mask = mask.mean(dim=1, keepdim=True)
            # Resize mask to match LPIPS output if needed
            if mask.shape[-2:] != lpips_values.shape[-2:]:
                mask = F.interpolate(
                    mask, size=lpips_values.shape[-2:], mode="bilinear"
                )

            # Weight LPIPS values by mask
            valid_pixels = mask.sum()
            if valid_pixels > 0:
                lpips_values = (lpips_values * mask).sum() / valid_pixels
            else:
                lpips_values = torch.tensor(float("nan"))
        else:
            lpips_values = lpips_values.mean()

        return MetricResult(value=lpips_values.item(), metadata={"network": "alex"})

    def compute_ms_ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: float = 1.0,
        mask: Optional[torch.Tensor] = None,
    ) -> MetricResult:
        """
        Compute Multi-Scale Structural Similarity Index.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Ground truth image [B, C, H, W]
            data_range: Maximum possible pixel value
            mask: Valid pixel mask [B, C, H, W]

        Returns:
            MS-SSIM metric result
        """
        try:
            from pytorch_msssim import ms_ssim
        except ImportError:
            logger.warning(
                "pytorch-msssim not available. Install with: pip install pytorch-msssim"
            )
            return MetricResult(
                value=float("nan"), metadata={"error": "MS-SSIM not available"}
            )

        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

        # MS-SSIM requires minimum image size
        min_size = 160  # Minimum size for 5 scales
        if min(pred.shape[-2:]) < min_size:
            # Pad images to minimum size
            pad_h = max(0, min_size - pred.shape[-2])
            pad_w = max(0, min_size - pred.shape[-1])
            pred = F.pad(pred, (0, pad_w, 0, pad_h), mode="reflect")
            target = F.pad(target, (0, pad_w, 0, pad_h), mode="reflect")
            if mask is not None:
                mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)

        pred = pred.to(self.device)
        target = target.to(self.device)

        # Apply mask by setting masked regions to target values
        if mask is not None:
            mask = mask.to(self.device)
            pred = pred * mask + target * (1 - mask)

        with torch.no_grad():
            ms_ssim_val = ms_ssim(
                pred, target, data_range=data_range, size_average=True
            )

        return MetricResult(
            value=ms_ssim_val.item(), metadata={"data_range": data_range, "scales": 5}
        )

    def _bootstrap_psnr_ci(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: float,
        mask: Optional[torch.Tensor],
        n_bootstrap: int = 100,
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for PSNR."""
        psnr_values = []

        # Flatten for bootstrap sampling
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        if mask is not None:
            mask_flat = mask.flatten()
            valid_idx = mask_flat > 0.5
            pred_flat = pred_flat[valid_idx]
            target_flat = target_flat[valid_idx]

        n_samples = len(pred_flat)
        if n_samples < 100:
            return None

        for _ in range(n_bootstrap):
            # Bootstrap sample
            idx = torch.randint(0, n_samples, (n_samples,))
            pred_boot = pred_flat[idx]
            target_boot = target_flat[idx]

            # Compute PSNR
            mse = F.mse_loss(pred_boot, target_boot)
            if mse.item() < 1e-10:
                psnr_boot = 100.0
            else:
                psnr_boot = 20 * torch.log10(data_range / torch.sqrt(mse))
                psnr_boot = psnr_boot.item()

            psnr_values.append(psnr_boot)

        # Compute 95% confidence interval
        psnr_values = np.array(psnr_values)
        ci_low = np.percentile(psnr_values, 2.5)
        ci_high = np.percentile(psnr_values, 97.5)

        return (ci_low, ci_high)


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


class DomainSpecificMetrics:
    """Domain-specific metrics for different imaging applications."""

    def __init__(self):
        """Initialize domain-specific metrics."""
        pass

    def compute_counting_accuracy(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.1,
        mask: Optional[torch.Tensor] = None,
    ) -> MetricResult:
        """
        Compute counting accuracy for microscopy applications.

        Measures how well the restoration preserves fluorophore counts.

        Args:
            pred: Predicted image [B, C, H, W] (normalized [0,1])
            target: Ground truth image [B, C, H, W] (normalized [0,1])
            threshold: Detection threshold for counting
            mask: Valid pixel mask [B, C, H, W]

        Returns:
            Counting accuracy metric result
        """
        # Apply mask if provided
        if mask is not None:
            pred = pred * mask
            target = target * mask

        # Threshold for detection
        pred_detections = (pred > threshold).float()
        target_detections = (target > threshold).float()

        # Count total detections
        pred_count = pred_detections.sum().item()
        target_count = target_detections.sum().item()

        # Compute relative error
        if target_count > 0:
            count_error = abs(pred_count - target_count) / target_count
        else:
            count_error = float("inf") if pred_count > 0 else 0.0

        # Compute spatial overlap (intersection over union)
        intersection = (pred_detections * target_detections).sum().item()
        union = (
            (pred_detections + target_detections - pred_detections * target_detections)
            .sum()
            .item()
        )

        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0

        # Combined accuracy metric
        accuracy = (1.0 - count_error) * iou
        accuracy = max(0.0, accuracy)  # Ensure non-negative

        return MetricResult(
            value=accuracy,
            metadata={
                "pred_count": pred_count,
                "target_count": target_count,
                "count_error": count_error,
                "iou": iou,
                "threshold": threshold,
            },
        )

    def compute_photometry_error(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source_positions: List[Tuple[int, int]],
        aperture_radius: int = 5,
        mask: Optional[torch.Tensor] = None,
    ) -> MetricResult:
        """
        Compute photometry error for astronomy applications.

        Measures accuracy of flux measurements for point sources.

        Args:
            pred: Predicted image [B, C, H, W] (normalized [0,1])
            target: Ground truth image [B, C, H, W] (normalized [0,1])
            source_positions: List of (y, x) positions for point sources
            aperture_radius: Radius for aperture photometry
            mask: Valid pixel mask [B, C, H, W]

        Returns:
            Photometry error metric result
        """
        if pred.shape[0] != 1:
            raise ValueError("Photometry error currently supports batch size 1")

        pred_img = pred[0, 0]  # Assume single channel
        target_img = target[0, 0]

        if mask is not None:
            mask_img = mask[0, 0]
        else:
            mask_img = torch.ones_like(pred_img)

        photometry_errors = []

        for y, x in source_positions:
            # Create circular aperture
            h, w = pred_img.shape
            yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            aperture = ((yy - y) ** 2 + (xx - x) ** 2) <= aperture_radius**2
            aperture = aperture.float()

            # Apply mask
            aperture = aperture * mask_img

            if aperture.sum() == 0:
                continue

            # Measure flux in aperture
            pred_flux = (pred_img * aperture).sum().item()
            target_flux = (target_img * aperture).sum().item()

            # Compute relative error
            if target_flux > 1e-6:
                rel_error = abs(pred_flux - target_flux) / target_flux
                photometry_errors.append(rel_error)

        if not photometry_errors:
            return MetricResult(
                value=float("nan"),
                metadata={"error": "No valid sources for photometry"},
            )

        mean_error = np.mean(photometry_errors)
        std_error = np.std(photometry_errors)

        return MetricResult(
            value=mean_error,
            metadata={
                "std": std_error,
                "num_sources": len(photometry_errors),
                "aperture_radius": aperture_radius,
                "individual_errors": photometry_errors,
            },
        )

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
        self.standard_metrics = StandardMetrics(device)
        self.physics_metrics = PhysicsMetrics()
        self.domain_metrics = DomainSpecificMetrics()
        self.residual_analyzer = ResidualAnalyzer(device)

        logger.info(f"EvaluationSuite initialized on device: {device}")

    def evaluate_restoration(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        noisy: torch.Tensor,
        scale: float,
        domain: str,
        background: float = 0.0,
        read_noise: float = 0.0,
        mask: Optional[torch.Tensor] = None,
        method_name: str = "unknown",
        dataset_name: str = "unknown",
        domain_specific_params: Optional[Dict[str, Any]] = None,
    ) -> EvaluationReport:
        """
        Perform comprehensive evaluation of a restoration method.

        Args:
            pred: Predicted clean image [B, C, H, W] (normalized [0,1])
            target: Ground truth clean image [B, C, H, W] (normalized [0,1])
            noisy: Noisy observation [B, C, H, W] (electrons)
            scale: Normalization scale (electrons)
            domain: Domain name ('photography', 'microscopy', 'astronomy')
            background: Background offset (electrons)
            read_noise: Read noise standard deviation (electrons)
            mask: Valid pixel mask [B, C, H, W]
            method_name: Name of the restoration method
            dataset_name: Name of the dataset
            domain_specific_params: Additional parameters for domain-specific metrics

        Returns:
            Complete evaluation report
        """
        import time

        start_time = time.time()

        logger.info(f"Evaluating {method_name} on {dataset_name} ({domain})")

        # --- Space Conversion for Metrics ---
        # De-normalize predictions and targets to electron space for consistent metrics.
        # This ensures PSNR/SSIM are calculated in the physical domain.
        pred_electrons = pred * scale + background
        target_electrons = target * scale + background
        data_range_electrons = scale  # The dynamic range is the scale factor

        # Standard metrics (calculated in electron space)
        logger.debug("Computing standard metrics in electron space...")
        psnr = self.standard_metrics.compute_psnr(
            pred_electrons, target_electrons, data_range=data_range_electrons, mask=mask
        )
        ssim = self.standard_metrics.compute_ssim(
            pred_electrons, target_electrons, data_range=data_range_electrons, mask=mask
        )

        # Perceptual metrics (calculated in normalized space as they expect [0,1] or [-1,1])
        lpips = self.standard_metrics.compute_lpips(pred, target, mask=mask)
        ms_ssim = self.standard_metrics.compute_ms_ssim(pred, target, data_range=1.0, mask=mask)

        # Physics metrics (operates on mixed spaces, handles conversion internally)
        logger.debug("Computing physics metrics...")
        chi2 = self.physics_metrics.compute_chi2_consistency(
            pred, noisy, scale, background, read_noise, mask
        )

        # --- Residual Analysis ---
        # The new ResidualAnalyzer provides a much more robust analysis
        # of the physical correctness of the restoration.
        try:
            residual_stats = self.residual_analyzer.analyze_residuals(
                pred_electrons, noisy, read_noise, mask
            )
            dist_meta = {
                "mean": residual_stats["mean"],
                "std_dev": residual_stats["std_dev"],
                "ks_statistic": residual_stats["ks_statistic"],
                "ks_pvalue": residual_stats["ks_pvalue"],
            }
            # The primary value for the distribution metric is the KS statistic,
            # which measures deviation from a perfect N(0,1). Lower is better.
            residual_dist = MetricResult(value=dist_meta["ks_statistic"], metadata=dist_meta)

            whiteness_meta = {
                "spectral_flatness": residual_stats["whiteness_spectral_flatness"],
                "autocorr_x": residual_stats["autocorrelation_lag1_x"],
                "autocorr_y": residual_stats["autocorrelation_lag1_y"],
            }
            # Primary value is spectral flatness. Closer to 1 is better.
            residual_whiteness = MetricResult(
                value=whiteness_meta["spectral_flatness"], metadata=whiteness_meta
            )

            # Bias is the mean of the normalized residuals. Closer to 0 is better.
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

        # Domain-specific metrics
        logger.debug("Computing domain-specific metrics...")
        domain_metrics_dict = {}

        if domain == "microscopy":
            # Counting accuracy
            counting = self.domain_metrics.compute_counting_accuracy(
                pred, target, mask=mask
            )
            domain_metrics_dict["counting_accuracy"] = counting

            # Resolution preservation
            resolution = self.domain_metrics.compute_resolution_metric(
                pred, target, mask=mask
            )
            domain_metrics_dict["resolution_preservation"] = resolution

        elif domain == "astronomy":
            # Photometry error (requires source positions)
            if domain_specific_params and "source_positions" in domain_specific_params:
                photometry = self.domain_metrics.compute_photometry_error(
                    pred,
                    target,
                    domain_specific_params["source_positions"],
                    aperture_radius=domain_specific_params.get("aperture_radius", 5),
                    mask=mask,
                )
                domain_metrics_dict["photometry_error"] = photometry

            # Resolution preservation
            resolution = self.domain_metrics.compute_resolution_metric(
                pred, target, mask=mask
            )
            domain_metrics_dict["resolution_preservation"] = resolution

        elif domain == "photography":
            # For photography, focus on perceptual quality
            # Resolution preservation is still relevant
            resolution = self.domain_metrics.compute_resolution_metric(
                pred, target, mask=mask
            )
            domain_metrics_dict["resolution_preservation"] = resolution

        processing_time = time.time() - start_time

        # Create evaluation report
        report = EvaluationReport(
            method_name=method_name,
            dataset_name=dataset_name,
            domain=domain,
            psnr=psnr,
            ssim=ssim,
            lpips=lpips,
            ms_ssim=ms_ssim,
            chi2_consistency=chi2,
            residual_distribution=residual_dist,
            residual_whiteness=residual_whiteness,
            bias_analysis=bias,
            domain_metrics=domain_metrics_dict,
            num_images=pred.shape[0],
            processing_time=processing_time,
        )

        logger.info(f"Evaluation completed in {processing_time:.2f}s")
        logger.info(
            f"PSNR: {psnr.value:.2f} dB, SSIM: {ssim.value:.3f}, χ²: {chi2.value:.3f}, "
            f"Res-KS: {residual_dist.value:.3f}"
        )

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

        # Group reports by dataset and domain
        grouped_reports = {}
        for report in reports:
            key = f"{report.dataset_name}_{report.domain}"
            if key not in grouped_reports:
                grouped_reports[key] = []
            grouped_reports[key].append(report)

        comparison_results = {}

        for key, group_reports in grouped_reports.items():
            # Split on last underscore to handle dataset names with underscores
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                dataset_name, domain = parts
            else:
                dataset_name, domain = key, "unknown"

            # Extract metrics for comparison
            methods = [r.method_name for r in group_reports]

            metrics_comparison = {
                "methods": methods,
                "psnr": [r.psnr.value for r in group_reports],
                "ssim": [r.ssim.value for r in group_reports],
                "lpips": [r.lpips.value for r in group_reports],
                "ms_ssim": [r.ms_ssim.value for r in group_reports],
                "chi2_consistency": [r.chi2_consistency.value for r in group_reports],
                "residual_distribution": [
                    r.residual_distribution.value for r in group_reports
                ],
                "residual_whiteness": [
                    r.residual_whiteness.value for r in group_reports
                ],
                "bias_analysis": [r.bias_analysis.value for r in group_reports],
            }

            # Add domain-specific metrics
            for report in group_reports:
                for metric_name, metric_result in report.domain_metrics.items():
                    if metric_name not in metrics_comparison:
                        metrics_comparison[metric_name] = []
                    metrics_comparison[metric_name].append(metric_result.value)

            # Find best method for each metric
            best_methods = {}
            for metric_name, values in metrics_comparison.items():
                if metric_name == "methods":
                    continue

                # Handle NaN values
                valid_values = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
                if not valid_values:
                    best_methods[metric_name] = "N/A"
                    continue

                # Determine if higher or lower is better
                if metric_name in [
                    "lpips",
                    "chi2_consistency",
                    "bias_analysis",
                    "photometry_error",
                    "residual_distribution",  # Lower KS-stat is better
                ]:
                    # Lower is better
                    best_idx = min(valid_values, key=lambda x: x[1])[0]
                else:
                    # Higher is better
                    best_idx = max(valid_values, key=lambda x: x[1])[0]

                best_methods[metric_name] = methods[best_idx]

            comparison_results[key] = {
                "dataset": dataset_name,
                "domain": domain,
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

        # Group by method and domain
        grouped = {}
        for report in reports:
            key = f"{report.method_name}_{report.domain}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(report)

        summary = {}

        for key, group_reports in grouped.items():
            # Split on last underscore to handle method names with underscores
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                method_name, domain = parts
            else:
                method_name, domain = key, "unknown"

            # Collect all metric values
            psnr_values = [
                r.psnr.value for r in group_reports if not np.isnan(r.psnr.value)
            ]
            ssim_values = [
                r.ssim.value for r in group_reports if not np.isnan(r.ssim.value)
            ]
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
                "domain": domain,
                "num_evaluations": len(group_reports),
                "psnr_stats": {
                    "mean": np.mean(psnr_values) if psnr_values else float("nan"),
                    "std": np.std(psnr_values) if psnr_values else float("nan"),
                    "min": np.min(psnr_values) if psnr_values else float("nan"),
                    "max": np.max(psnr_values) if psnr_values else float("nan"),
                },
                "ssim_stats": {
                    "mean": np.mean(ssim_values) if ssim_values else float("nan"),
                    "std": np.std(ssim_values) if ssim_values else float("nan"),
                    "min": np.min(ssim_values) if ssim_values else float("nan"),
                    "max": np.max(ssim_values) if ssim_values else float("nan"),
                },
                "chi2_stats": {
                    "mean": np.mean(chi2_values) if chi2_values else float("nan"),
                    "std": np.std(chi2_values) if chi2_values else float("nan"),
                    "min": np.min(chi2_values) if chi2_values else float("nan"),
                    "max": np.max(chi2_values) if chi2_values else float("nan"),
                },
                "residual_ks_stats": {
                    "mean": np.mean(residual_ks_values) if residual_ks_values else float("nan"),
                    "std": np.std(residual_ks_values) if residual_ks_values else float("nan"),
                    "min": np.min(residual_ks_values) if residual_ks_values else float("nan"),
                    "max": np.max(residual_ks_values) if residual_ks_values else float("nan"),
                },
                "avg_processing_time": np.mean(
                    [r.processing_time for r in group_reports]
                ),
            }

        return summary


# Utility functions for baseline comparison
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
