"""
Image quality metrics for DAPGD

Implementations of standard metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)

PURPOSE: Quantitative evaluation of restoration quality
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> Union[float, np.ndarray]:
    """
    Compute Peak Signal-to-Noise Ratio

    PURPOSE: Standard metric for image reconstruction quality

    Args:
        prediction: Predicted image [B,C,H,W]
        target: Ground truth image [B,C,H,W]
        data_range: Maximum possible pixel value (1.0 for normalized images)
        reduction: 'mean' or 'none'

    Returns:
        PSNR in dB (higher is better)
    """

    # Compute MSE
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))

    # Avoid log(0)
    mse = torch.clamp(mse, min=1e-10)

    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))

    if reduction == "mean":
        return psnr.mean().item()
    else:
        return psnr.cpu().numpy()


def compute_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    reduction: str = "mean",
) -> Union[float, np.ndarray]:
    """
    Compute Structural Similarity Index

    PURPOSE: Perceptual metric that correlates with human judgment

    Args:
        prediction: Predicted image [B,C,H,W]
        target: Ground truth image [B,C,H,W]
        data_range: Maximum possible pixel value
        window_size: Size of Gaussian window
        reduction: 'mean' or 'none'

    Returns:
        SSIM value in [0, 1] (higher is better)
    """

    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        raise ImportError(
            "scikit-image required for SSIM. Install: pip install scikit-image"
        )

    # Convert to numpy
    pred_np = prediction.cpu().numpy()
    target_np = target.cpu().numpy()

    # Compute SSIM for each image in batch
    ssim_values = []
    for i in range(pred_np.shape[0]):
        # SSIM expects channel-last
        pred_img = np.transpose(pred_np[i], (1, 2, 0))
        target_img = np.transpose(target_np[i], (1, 2, 0))

        ssim_val = structural_similarity(
            pred_img,
            target_img,
            data_range=data_range,
            channel_axis=2,
            win_size=window_size,
        )
        ssim_values.append(ssim_val)

    ssim_values = np.array(ssim_values)

    if reduction == "mean":
        return ssim_values.mean()
    else:
        return ssim_values


def compute_lpips(
    prediction: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
    device: str = "cuda",
) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity

    PURPOSE: Deep learning-based perceptual metric

    Args:
        prediction: Predicted image [B,C,H,W]
        target: Ground truth image [B,C,H,W]
        net: Network to use ('alex', 'vgg', 'squeeze')
        device: Device for computation

    Returns:
        LPIPS distance (lower is better)
    """

    try:
        import lpips
    except ImportError:
        raise ImportError("lpips package required. Install: pip install lpips")

    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net=net).to(device)

    # LPIPS expects values in [-1, 1]
    pred_scaled = prediction * 2 - 1
    target_scaled = target * 2 - 1

    # Compute LPIPS
    with torch.no_grad():
        lpips_val = loss_fn(pred_scaled.to(device), target_scaled.to(device))

    return lpips_val.mean().item()


def compute_mse(
    prediction: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> Union[float, torch.Tensor]:
    """
    Compute Mean Squared Error

    Args:
        prediction: Predicted image [B,C,H,W]
        target: Ground truth image [B,C,H,W]
        reduction: 'mean' or 'none'

    Returns:
        MSE value (lower is better)
    """

    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))

    if reduction == "mean":
        return mse.mean().item()
    else:
        return mse


def compute_mae(
    prediction: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> Union[float, torch.Tensor]:
    """
    Compute Mean Absolute Error

    Args:
        prediction: Predicted image [B,C,H,W]
        target: Ground truth image [B,C,H,W]
        reduction: 'mean' or 'none'

    Returns:
        MAE value (lower is better)
    """

    mae = torch.mean(torch.abs(prediction - target), dim=(1, 2, 3))

    if reduction == "mean":
        return mae.mean().item()
    else:
        return mae


class ImageQualityMetrics:
    """
    Comprehensive image quality evaluation

    PURPOSE: Unified interface for computing multiple metrics

    Args:
        device: Device for computation
        data_range: Maximum possible pixel value
    """

    def __init__(self, device: str = "cuda", data_range: float = 1.0):
        self.device = device
        self.data_range = data_range

    def compute_all(
        self, prediction: torch.Tensor, target: torch.Tensor, compute_lpips: bool = True
    ) -> Dict[str, float]:
        """
        Compute all available metrics

        Args:
            prediction: Predicted images [B,C,H,W]
            target: Ground truth images [B,C,H,W]
            compute_lpips: Whether to compute LPIPS (slower)

        Returns:
            Dictionary with all computed metrics
        """

        metrics = {}

        # Basic metrics (fast)
        metrics["psnr"] = compute_psnr(prediction, target, self.data_range)
        metrics["mse"] = compute_mse(prediction, target)
        metrics["mae"] = compute_mae(prediction, target)

        # SSIM (medium speed)
        metrics["ssim"] = compute_ssim(prediction, target, self.data_range)

        # LPIPS (slowest, optional)
        if compute_lpips:
            metrics["lpips"] = compute_lpips(prediction, target, device=self.device)

        return metrics

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute specified metrics

        Args:
            prediction: Predicted images
            target: Ground truth images
            metrics: List of metrics to compute

        Returns:
            Dictionary with requested metrics
        """

        if metrics is None:
            return self.compute_all(prediction, target)

        result = {}

        for metric in metrics:
            if metric == "psnr":
                result["psnr"] = compute_psnr(prediction, target, self.data_range)
            elif metric == "ssim":
                result["ssim"] = compute_ssim(prediction, target, self.data_range)
            elif metric == "lpips":
                result["lpips"] = compute_lpips(prediction, target, device=self.device)
            elif metric == "mse":
                result["mse"] = compute_mse(prediction, target)
            elif metric == "mae":
                result["mae"] = compute_mae(prediction, target)
            else:
                logger.warning(f"Unknown metric: {metric}")

        return result


# Factory function for easy creation


def create_metrics_evaluator(
    device: str = "cuda", data_range: float = 1.0
) -> ImageQualityMetrics:
    """
    Create image quality metrics evaluator

    Args:
        device: Device for computation
        data_range: Maximum possible pixel value

    Returns:
        Configured ImageQualityMetrics instance
    """

    return ImageQualityMetrics(device=device, data_range=data_range)


# Test functions


def test_image_metrics():
    """Test image quality metrics"""

    logger.info("Testing image quality metrics...")

    # Create test data
    clean = torch.rand(2, 3, 64, 64)

    # Add different types of noise
    gaussian_noise = 0.1 * torch.randn_like(clean)
    salt_pepper = torch.rand_like(clean) > 0.95
    salt_pepper = salt_pepper.float() * (torch.rand_like(clean) > 0.5).float() * 2 - 1

    noisy_gaussian = torch.clamp(clean + gaussian_noise, 0, 1)
    noisy_sp = torch.clamp(clean + salt_pepper * 0.5, 0, 1)

    # Create evaluator
    evaluator = ImageQualityMetrics(device="cpu")

    # Test all metrics
    metrics_gauss = evaluator.compute_all(noisy_gaussian, clean, compute_lpips=False)
    metrics_sp = evaluator.compute_all(noisy_sp, clean, compute_lpips=False)

    logger.info("✓ Gaussian noise metrics computed")
    logger.info(
        f"  PSNR: {metrics_gauss['psnr']:.2f}, SSIM: {metrics_gauss['ssim']:.4f}"
    )
    logger.info(f"  MSE: {metrics_gauss['mse']:.6f}, MAE: {metrics_gauss['mae']:.6f}")

    logger.info("✓ Salt & pepper noise metrics computed")
    logger.info(f"  PSNR: {metrics_sp['psnr']:.2f}, SSIM: {metrics_sp['ssim']:.4f}")
    logger.info(f"  MSE: {metrics_sp['mse']:.6f}, MAE: {metrics_sp['mae']:.6f}")

    # Verify that salt & pepper has lower quality (higher error)
    assert (
        metrics_sp["psnr"] < metrics_gauss["psnr"]
    ), "Salt & pepper should have lower PSNR"
    assert (
        metrics_sp["mse"] > metrics_gauss["mse"]
    ), "Salt & pepper should have higher MSE"

    logger.info("✅ Image quality metrics tests passed!")
    return True


if __name__ == "__main__":
    # Run tests
    success = test_image_metrics()
    if success:
        print("\n🎉 Image quality metrics tests PASSED!")
        print("All metrics are working correctly.")
    else:
        print("\n❌ Image quality metrics tests FAILED!")
        exit(1)
