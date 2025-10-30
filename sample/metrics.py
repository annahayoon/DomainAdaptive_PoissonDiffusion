#!/usr/bin/env python3
"""
Image quality metrics computation module.

This module provides comprehensive image quality metrics for evaluating
restored/enhanced images including PSNR, SSIM, LPIPS, NIQE, and FID.
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
import tqdm
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.regression import MeanSquaredError as MSE

# Setup logging first
logger = logging.getLogger(__name__)

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

# EDM utilities for FID computation - imported lazily when needed
EDM_AVAILABLE = False

# Global LPIPS metric instance to avoid reloading weights
_lpips_metric_cache = {}


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
            enhanced_01 = convert_range(enhanced, "[-1,1]", "[0,1]")
            niqe_metric = pyiqa.create_metric("niqe")
            niqe_val = niqe_metric(enhanced_01).item()
        except Exception as e:
            logger.warning(f"NIQE computation failed: {e}")
            niqe_val = float("nan")
    return niqe_val


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


# ============================================================================
# FID (Fréchet Inception Distance) Computation
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
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load NVIDIA pre-trained Inception-v3 model (same as EDM)
        # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
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
        Matches EDM fid.py line 66-68 exactly.

        Args:
            images: Input images [B, C, H, W] in range [0, 1] (already converted from [-1,1])
                   Note: Images are loaded from .pt files as tensors, converted to [0,1] range,
                   and reshaped to (batch_size, 3, height, width) format required by Inception.

        Returns:
            Feature vectors [B, 2048] as torch.Tensor

        Note on normalization:
            Standard PyTorch Inception V3 typically requires ImageNet normalization (mean/std),
            but the NVIDIA Inception model (from StyleGAN3) handles normalization internally.
            Following EDM's approach, we pass [0,1] images directly without ImageNet mean/std.
        """
        with torch.no_grad():
            # Ensure images are in correct format: (batch_size, 3, height, width)
            # Handle grayscale conversion (same as EDM fid.py line 66-67)
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])

            # Extract features using NVIDIA model (same as EDM fid.py line 68)
            # The model expects: (batch_size, 3, H, W) tensors with pixel values in [0, 1]
            # The NVIDIA model handles normalization internally, unlike standard torchvision Inception
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
        # Convert images from [-1, 1] to [0, 1] (EDM's dataset provides images in this range)
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
            tensor = torch.load(pt_file, map_location="cpu")  # Load to CPU first

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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"FID Score: {fid_score:.4f}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
