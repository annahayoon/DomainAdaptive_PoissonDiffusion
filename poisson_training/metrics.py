"""
Training metrics and monitoring for Poisson-Gaussian diffusion restoration.

This module provides comprehensive metrics tracking, logging, and monitoring
capabilities for training the diffusion model, including multi-resolution metrics.

Key features:
- Real-time metrics computation
- Physics-aware metrics (PSNR, SSIM, χ² consistency)
- Training progress monitoring
- Multi-resolution evaluation metrics
- Metric visualization and logging
- Resolution-aware quality assessment

Requirements addressed: 8.2, 8.3 from requirements.md
"""

import math
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from core.logging_config import get_logger

logger = get_logger(__name__)


# Module-level factory functions to avoid pickling issues with instance methods
def _create_list_dict():
    """Create a defaultdict with list values for metrics."""
    return defaultdict(list)


def _create_deque_dict(window_size):
    """Create a defaultdict with deque values for moving averages."""
    return defaultdict(lambda: deque(maxlen=window_size))


class TrainingMetrics:
    """
    Comprehensive metrics tracking for training.

    This class tracks various metrics during training including loss values,
    image quality metrics, and physics-specific metrics.

    Note: This class is designed to be pickle-safe for distributed training.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize training metrics.

        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.reset()

        logger.info(f"Initialized TrainingMetrics with window size {window_size}")

    def reset(self):
        """Reset all metrics."""
        # Use module-level functions instead of instance methods for pickle safety
        self.metrics = defaultdict(_create_list_dict)
        self.moving_averages = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.window_size))
        )
        self.step_times = deque(maxlen=self.window_size)
        self.last_update_time = time.time()

    def __getstate__(self):
        """Custom pickle state to handle non-serializable objects."""
        # Convert defaultdicts to regular dicts for pickling
        state = self.__dict__.copy()

        # Convert nested defaultdicts to regular dicts
        state["metrics"] = {
            phase: dict(phase_metrics) for phase, phase_metrics in self.metrics.items()
        }
        state["moving_averages"] = {
            phase: {metric: list(values) for metric, values in phase_metrics.items()}
            for phase, phase_metrics in self.moving_averages.items()
        }
        state["step_times"] = list(self.step_times)

        return state

    def __setstate__(self, state):
        """Custom unpickle state to restore defaultdicts."""
        # Restore regular attributes
        self.__dict__.update(state)

        # Restore defaultdicts with proper factory functions
        metrics_dict = defaultdict(_create_list_dict)
        for phase, phase_metrics in state["metrics"].items():
            for metric, values in phase_metrics.items():
                metrics_dict[phase][metric] = values
        self.metrics = metrics_dict

        # Restore moving averages defaultdict
        moving_avg_dict = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.window_size))
        )
        for phase, phase_metrics in state["moving_averages"].items():
            for metric, values in phase_metrics.items():
                # Restore as deque with proper maxlen
                deque_values = deque(values, maxlen=self.window_size)
                moving_avg_dict[phase][metric] = deque_values
        self.moving_averages = moving_avg_dict

        # Restore step_times as deque
        self.step_times = deque(state["step_times"], maxlen=self.window_size)

    def update(self, values: Dict[str, float], phase: str = "train"):
        """
        Update metrics with new values.

        Args:
            values: Dictionary of metric values
            phase: Training phase ('train', 'val', 'test')
        """
        current_time = time.time()

        # Track step time
        if hasattr(self, "last_update_time"):
            step_time = current_time - self.last_update_time
            self.step_times.append(step_time)

        self.last_update_time = current_time

        # Update metrics
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            self.metrics[phase][key].append(value)
            self.moving_averages[phase][key].append(value)

    def get_current_metrics(self, phase: str = "train") -> Dict[str, float]:
        """
        Get current metric values (moving averages).

        Args:
            phase: Training phase

        Returns:
            Dictionary of current metric values
        """
        current_metrics = {}

        for key, values in self.moving_averages[phase].items():
            if values:
                current_metrics[key] = np.mean(list(values))

        return current_metrics

    def get_latest_metrics(self, phase: str = "train") -> Dict[str, float]:
        """
        Get latest metric values.

        Args:
            phase: Training phase

        Returns:
            Dictionary of latest metric values
        """
        latest_metrics = {}

        for key, values in self.metrics[phase].items():
            if values:
                latest_metrics[key] = values[-1]

        return latest_metrics

    def get_metric_history(self, metric_name: str, phase: str = "train") -> List[float]:
        """
        Get full history of a specific metric.

        Args:
            metric_name: Name of the metric
            phase: Training phase

        Returns:
            List of metric values
        """
        return self.metrics[phase].get(metric_name, [])

    def get_throughput(self) -> float:
        """
        Get current throughput (steps per second).

        Returns:
            Steps per second
        """
        if not self.step_times:
            return 0.0

        avg_step_time = np.mean(list(self.step_times))
        return 1.0 / avg_step_time if avg_step_time > 0 else 0.0

    def get_summary(self, phase: str = "train") -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Args:
            phase: Training phase

        Returns:
            Dictionary with metrics summary
        """
        summary = {
            "current_metrics": self.get_current_metrics(phase),
            "latest_metrics": self.get_latest_metrics(phase),
            "throughput_steps_per_sec": self.get_throughput(),
            "total_steps": len(self.metrics[phase].get("total_loss", [])),
        }

        # Add statistics for key metrics
        for key in ["total_loss", "reconstruction_loss"]:
            if key in self.metrics[phase]:
                values = self.metrics[phase][key]
                if values:
                    summary[f"{key}_stats"] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "latest": values[-1],
                    }

        return summary


class ImageQualityMetrics:
    """
    Image quality metrics computation.

    Computes standard image quality metrics like PSNR, SSIM, etc.
    """

    @staticmethod
    def psnr(
        prediction: torch.Tensor, target: torch.Tensor, max_val: float = 1.0
    ) -> torch.Tensor:
        """
        Compute Peak Signal-to-Noise Ratio.

        Args:
            prediction: Predicted image
            target: Target image
            max_val: Maximum possible pixel value

        Returns:
            PSNR value
        """
        mse = torch.mean((prediction - target) ** 2)
        if mse == 0:
            return torch.tensor(float("inf"))

        psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr_val

    @staticmethod
    def ssim(
        prediction: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        max_val: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute Structural Similarity Index.

        Args:
            prediction: Predicted image
            target: Target image
            window_size: Size of sliding window
            sigma: Standard deviation of Gaussian window
            k1, k2: SSIM constants
            max_val: Maximum possible pixel value

        Returns:
            SSIM value
        """
        # Simplified SSIM computation
        mu1 = torch.mean(prediction)
        mu2 = torch.mean(target)

        sigma1_sq = torch.var(prediction)
        sigma2_sq = torch.var(target)
        sigma12 = torch.mean((prediction - mu1) * (target - mu2))

        c1 = (k1 * max_val) ** 2
        c2 = (k2 * max_val) ** 2

        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
        )

        return ssim_val

    @staticmethod
    def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Mean Absolute Error."""
        return torch.mean(torch.abs(prediction - target))

    @staticmethod
    def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Mean Squared Error."""
        return torch.mean((prediction - target) ** 2)


class PhysicsMetrics:
    """
    Physics-aware metrics for scientific image restoration.

    Computes metrics that are relevant for physics-based image restoration,
    particularly for Poisson-Gaussian noise models.
    """

    @staticmethod
    def chi_squared_consistency(
        prediction: torch.Tensor,
        target: torch.Tensor,
        scale: torch.Tensor,
        background: torch.Tensor,
        read_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute χ² consistency metric.

        This metric measures how well the residuals follow the expected
        Poisson-Gaussian statistics.

        Args:
            prediction: Predicted image (normalized [0,1])
            target: Target observation (electrons)
            scale: Normalization scale (electrons)
            background: Background level (electrons)
            read_noise: Read noise std (electrons)

        Returns:
            χ² consistency value (should be close to 1.0)
        """
        # Convert prediction to electrons
        pred_electrons = prediction * scale + background

        # Compute residuals
        residuals = target - pred_electrons

        # Expected variance (Poisson + Gaussian)
        expected_var = pred_electrons + read_noise**2

        # χ² statistic
        chi_squared = torch.sum(residuals**2 / expected_var) / residuals.numel()

        return chi_squared

    @staticmethod
    def photon_noise_ratio(
        prediction: torch.Tensor, target: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute photon noise ratio.

        This measures the ratio of signal to photon noise, which should
        be preserved in physics-aware restoration.

        Args:
            prediction: Predicted image (normalized [0,1])
            target: Target observation (electrons)
            scale: Normalization scale (electrons)

        Returns:
            Photon noise ratio
        """
        pred_electrons = prediction * scale
        target_norm = target / scale

        # Signal-to-noise ratio based on Poisson statistics
        pred_snr = torch.mean(pred_electrons / torch.sqrt(pred_electrons + 1e-8))
        target_snr = torch.mean(target_norm / torch.sqrt(target_norm + 1e-8))

        return pred_snr / (target_snr + 1e-8)

    @staticmethod
    def energy_conservation(
        prediction: torch.Tensor, target: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy conservation metric.

        This measures how well the total energy (photon count) is preserved.

        Args:
            prediction: Predicted image (normalized [0,1])
            target: Target observation (electrons)
            scale: Normalization scale (electrons)

        Returns:
            Energy conservation ratio (should be close to 1.0)
        """
        pred_electrons = prediction * scale
        target_norm = target / scale

        pred_energy = torch.sum(pred_electrons)
        target_energy = torch.sum(target_norm)

        return pred_energy / (target_energy + 1e-8)


class MetricsComputer:
    """
    Comprehensive metrics computer that combines all metric types.
    """

    def __init__(self, compute_physics_metrics: bool = True):
        """
        Initialize metrics computer.

        Args:
            compute_physics_metrics: Whether to compute physics-specific metrics
        """
        self.compute_physics_metrics = compute_physics_metrics
        self.image_metrics = ImageQualityMetrics()
        self.physics_metrics = PhysicsMetrics()

    def compute_all_metrics(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Compute all available metrics.

        Args:
            prediction: Model prediction
            target: Target image
            batch: Additional batch information for physics metrics

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Standard image quality metrics
        metrics["psnr"] = self.image_metrics.psnr(prediction, target).item()
        metrics["ssim"] = self.image_metrics.ssim(prediction, target).item()
        metrics["mae"] = self.image_metrics.mae(prediction, target).item()
        metrics["mse"] = self.image_metrics.mse(prediction, target).item()

        # Physics metrics (if enabled and batch info available)
        if self.compute_physics_metrics and batch is not None:
            try:
                scale = batch.get(
                    "scale", torch.tensor(1000.0, device=prediction.device)
                )
                background = batch.get(
                    "background", torch.tensor(0.0, device=prediction.device)
                )
                read_noise = batch.get(
                    "read_noise", torch.tensor(5.0, device=prediction.device)
                )

                # Ensure proper shapes
                if scale.numel() == 1:
                    scale = scale.expand(prediction.shape[0])
                if background.numel() == 1:
                    background = background.expand(prediction.shape[0])
                if read_noise.numel() == 1:
                    read_noise = read_noise.expand(prediction.shape[0])

                # Compute physics metrics
                target_electrons = batch.get(
                    "electrons", target * scale[0] + background[0]
                )

                metrics[
                    "chi2_consistency"
                ] = self.physics_metrics.chi_squared_consistency(
                    prediction, target_electrons, scale, background, read_noise
                ).item()

                metrics["photon_noise_ratio"] = self.physics_metrics.photon_noise_ratio(
                    prediction, target_electrons, scale
                ).item()

                metrics[
                    "energy_conservation"
                ] = self.physics_metrics.energy_conservation(
                    prediction, target_electrons, scale
                ).item()

            except Exception as e:
                logger.warning(f"Failed to compute physics metrics: {e}")

        return metrics


class ProgressTracker:
    """
    Training progress tracking and estimation.
    """

    def __init__(self):
        self.start_time = time.time()
        self.epoch_times = []
        self.step_times = deque(maxlen=100)

    def update_epoch(self, epoch: int, total_epochs: int):
        """Update epoch progress."""
        current_time = time.time()

        if hasattr(self, "last_epoch_time"):
            epoch_time = current_time - self.last_epoch_time
            self.epoch_times.append(epoch_time)

        self.last_epoch_time = current_time

    def update_step(self):
        """Update step progress."""
        current_time = time.time()

        if hasattr(self, "last_step_time"):
            step_time = current_time - self.last_step_time
            self.step_times.append(step_time)

        self.last_step_time = current_time

    def get_eta(self, current_epoch: int, total_epochs: int) -> float:
        """
        Get estimated time to completion.

        Args:
            current_epoch: Current epoch number
            total_epochs: Total number of epochs

        Returns:
            Estimated seconds to completion
        """
        if not self.epoch_times:
            return 0.0

        avg_epoch_time = np.mean(self.epoch_times[-10:])  # Use last 10 epochs
        remaining_epochs = total_epochs - current_epoch - 1

        return avg_epoch_time * remaining_epochs

    def get_throughput(self) -> float:
        """Get current throughput (steps per second)."""
        if not self.step_times:
            return 0.0

        avg_step_time = np.mean(list(self.step_times))
        return 1.0 / avg_step_time if avg_step_time > 0 else 0.0

    def format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def get_progress_summary(
        self, current_epoch: int, total_epochs: int
    ) -> Dict[str, str]:
        """Get formatted progress summary."""
        elapsed = time.time() - self.start_time
        eta = self.get_eta(current_epoch, total_epochs)
        throughput = self.get_throughput()

        return {
            "elapsed": self.format_time(elapsed),
            "eta": self.format_time(eta),
            "throughput": f"{throughput:.2f} steps/s",
            "progress": f"{current_epoch + 1}/{total_epochs} ({100*(current_epoch+1)/total_epochs:.1f}%)",
        }


# Utility functions
def compute_batch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Optional[Dict[str, torch.Tensor]] = None,
    compute_physics: bool = True,
) -> Dict[str, float]:
    """
    Compute metrics for a batch of predictions.

    Args:
        prediction: Model predictions
        target: Target images
        batch: Additional batch information
        compute_physics: Whether to compute physics metrics

    Returns:
        Dictionary of computed metrics
    """
    computer = MetricsComputer(compute_physics_metrics=compute_physics)
    return computer.compute_all_metrics(prediction, target, batch)


def log_metrics(
    metrics: Dict[str, float],
    epoch: int,
    phase: str = "train",
    logger_instance: Optional[Any] = None,
):
    """
    Log metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        epoch: Current epoch
        phase: Training phase
        logger_instance: Logger instance to use
    """
    if logger_instance is None:
        logger_instance = logger

    # Format metrics for logging
    metric_strs = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if "loss" in key.lower():
                metric_strs.append(f"{key}: {value:.4f}")
            elif key in ["psnr"]:
                metric_strs.append(f"{key}: {value:.2f} dB")
            elif key in ["ssim", "chi2_consistency", "energy_conservation"]:
                metric_strs.append(f"{key}: {value:.3f}")
            else:
                metric_strs.append(f"{key}: {value:.4f}")

    metrics_str = " | ".join(metric_strs)
    logger_instance.info(f"Epoch {epoch:3d} [{phase:5s}] | {metrics_str}")


def create_metrics_summary(
    train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None
) -> str:
    """
    Create formatted metrics summary.

    Args:
        train_metrics: Training metrics
        val_metrics: Validation metrics (optional)

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("METRICS SUMMARY")
    lines.append("=" * 60)

    # Training metrics
    lines.append("Training:")
    for key, value in train_metrics.items():
        if isinstance(value, float):
            if "loss" in key.lower():
                lines.append(f"  {key:20s}: {value:8.4f}")
            elif key == "psnr":
                lines.append(f"  {key:20s}: {value:8.2f} dB")
            elif key in ["ssim", "chi2_consistency"]:
                lines.append(f"  {key:20s}: {value:8.3f}")
            else:
                lines.append(f"  {key:20s}: {value:8.4f}")

    # Validation metrics
    if val_metrics:
        lines.append("\nValidation:")
        for key, value in val_metrics.items():
            if isinstance(value, float):
                if "loss" in key.lower():
                    lines.append(f"  {key:20s}: {value:8.4f}")
                elif key == "psnr":
                    lines.append(f"  {key:20s}: {value:8.2f} dB")
                elif key in ["ssim", "chi2_consistency"]:
                    lines.append(f"  {key:20s}: {value:8.3f}")
                else:
                    lines.append(f"  {key:20s}: {value:8.4f}")

    lines.append("=" * 60)

    return "\n".join(lines)


class MultiResolutionMetrics:
    """
    Multi-resolution evaluation metrics for progressive growing architectures.

    This class provides comprehensive evaluation of model performance across
    different resolution levels, tracking quality improvements and efficiency metrics.
    """

    def __init__(self):
        """Initialize multi-resolution metrics."""
        self.reset()
        logger.info("Initialized MultiResolutionMetrics")

    def _create_dict(self):
        """Create a dict for resolution metrics (pickle-safe)."""
        return dict()

    def reset(self):
        """Reset all multi-resolution metrics."""
        self.resolution_metrics = defaultdict(self._create_dict)
        self.progressive_improvements = []
        self.stage_transitions = []
        self.quality_efficiency_scores = []

    def evaluate_at_resolutions(
        self,
        model: torch.nn.Module,
        test_images: torch.Tensor,
        ground_truth: torch.Tensor,
        resolutions: List[int],
        device: str = "cuda",
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate model performance at multiple resolutions.

        Args:
            model: Model to evaluate
            test_images: Test input images
            ground_truth: Ground truth images
            resolutions: List of resolutions to evaluate
            device: Device for computation

        Returns:
            Dictionary mapping resolution to metrics
        """
        model.eval()
        metrics_by_resolution = {}

        with torch.no_grad():
            for resolution in resolutions:
                logger.info(f"Evaluating at resolution {resolution}×{resolution}")

                # Resize images to target resolution
                if test_images.shape[-1] != resolution:
                    test_resized = torch.nn.functional.interpolate(
                        test_images,
                        size=(resolution, resolution),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    test_resized = test_images

                if ground_truth.shape[-1] != resolution:
                    gt_resized = torch.nn.functional.interpolate(
                        ground_truth,
                        size=(resolution, resolution),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    gt_resized = ground_truth

                # Run inference
                start_time = time.time()
                predictions = model(test_resized.to(device))
                inference_time = time.time() - start_time

                # Move back to CPU for metrics
                predictions = predictions.cpu()
                test_resized = test_resized.cpu()
                gt_resized = gt_resized.cpu()

                # Compute metrics
                metrics = self._compute_resolution_metrics(
                    predictions, gt_resized, test_resized, inference_time
                )

                metrics_by_resolution[resolution] = metrics

                logger.info(
                    f"Resolution {resolution}: PSNR={metrics['psnr']:.2f}, "
                    f"Time={metrics['inference_time']:.3f}s"
                )

        return metrics_by_resolution

    def _compute_resolution_metrics(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        inputs: torch.Tensor,
        inference_time: float,
    ) -> Dict[str, float]:
        """Compute metrics for a specific resolution."""
        # PSNR
        mse = torch.mean((predictions - ground_truth) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

        # SSIM
        ssim = self._compute_ssim(predictions, ground_truth)

        # Input PSNR (noise level)
        input_mse = torch.mean((inputs - ground_truth) ** 2)
        input_psnr = 20 * torch.log10(1.0 / torch.sqrt(input_mse + 1e-8))

        # Improvement
        improvement = psnr - input_psnr

        # Efficiency metrics
        pixels_per_second = predictions.shape[-1] ** 2 / inference_time

        return {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "input_psnr": input_psnr.item(),
            "improvement": improvement.item(),
            "inference_time": inference_time,
            "pixels_per_second": pixels_per_second,
            "quality_efficiency": (psnr.item() ** 2) / inference_time,  # Quality²/time
        }

    def _compute_ssim(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> torch.Tensor:
        """Compute SSIM between predictions and ground truth."""
        # Simple SSIM implementation
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        # Convert to grayscale if needed
        if predictions.dim() == 4:
            pred_gray = torch.mean(predictions, dim=1, keepdim=True)
            gt_gray = torch.mean(ground_truth, dim=1, keepdim=True)
        else:
            pred_gray = predictions
            gt_gray = ground_truth

        mu1 = torch.nn.functional.avg_pool2d(
            pred_gray, window_size, stride=1, padding=window_size // 2
        )
        mu2 = torch.nn.functional.avg_pool2d(
            gt_gray, window_size, stride=1, padding=window_size // 2
        )

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            torch.nn.functional.avg_pool2d(
                pred_gray**2, window_size, stride=1, padding=window_size // 2
            )
            - mu1_sq
        )
        sigma2_sq = (
            torch.nn.functional.avg_pool2d(
                gt_gray**2, window_size, stride=1, padding=window_size // 2
            )
            - mu2_sq
        )
        sigma12 = (
            torch.nn.functional.avg_pool2d(
                pred_gray * gt_gray, window_size, stride=1, padding=window_size // 2
            )
            - mu1_mu2
        )

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / (denominator + 1e-8)
        return torch.mean(ssim_map)

    def compute_progressive_improvement(
        self, metrics_by_resolution: Dict[int, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute progressive improvement metrics across resolutions.

        Args:
            metrics_by_resolution: Metrics for each resolution level

        Returns:
            Dictionary with progressive improvement metrics
        """
        resolutions = sorted(metrics_by_resolution.keys())
        if len(resolutions) < 2:
            return {}

        improvements = []
        for i in range(1, len(resolutions)):
            prev_res = resolutions[i - 1]
            curr_res = resolutions[i]

            prev_metrics = metrics_by_resolution[prev_res]
            curr_metrics = metrics_by_resolution[curr_res]

            psnr_improvement = curr_metrics["psnr"] - prev_metrics["psnr"]
            time_increase = (
                curr_metrics["inference_time"] / prev_metrics["inference_time"]
            )

            improvements.append(
                {
                    "from_resolution": prev_res,
                    "to_resolution": curr_res,
                    "psnr_improvement": psnr_improvement,
                    "time_increase": time_increase,
                    "efficiency_ratio": psnr_improvement / (time_increase - 1)
                    if time_increase > 1
                    else 0,
                }
            )

        # Average improvements
        avg_psnr_improvement = np.mean(
            [imp["psnr_improvement"] for imp in improvements]
        )
        avg_time_increase = np.mean([imp["time_increase"] for imp in improvements])

        return {
            "avg_psnr_improvement_per_stage": avg_psnr_improvement,
            "avg_time_increase_per_stage": avg_time_increase,
            "total_stages": len(improvements),
            "stage_improvements": improvements,
            "best_resolution": max(
                resolutions, key=lambda r: metrics_by_resolution[r]["psnr"]
            ),
            "most_efficient_resolution": max(
                resolutions,
                key=lambda r: metrics_by_resolution[r]["quality_efficiency"],
            ),
        }

    def log_resolution_analysis(
        self,
        metrics_by_resolution: Dict[int, Dict[str, float]],
        progressive_improvements: Dict[str, float],
    ):
        """Log comprehensive multi-resolution analysis."""
        logger.info("=== Multi-Resolution Analysis ===")

        for resolution, metrics in metrics_by_resolution.items():
            logger.info(f"Resolution {resolution}×{resolution}:")
            logger.info(f"  PSNR: {metrics['psnr']:.2f} dB")
            logger.info(f"  SSIM: {metrics['ssim']:.4f}")
            logger.info(f"  Input PSNR: {metrics['input_psnr']:.2f} dB")
            logger.info(f"  Improvement: {metrics['improvement']:.2f} dB")
            logger.info(f"  Inference Time: {metrics['inference_time']:.3f}s")
            logger.info(f"  Quality Efficiency: {metrics['quality_efficiency']:.2f}")

        if progressive_improvements:
            logger.info("Progressive Improvements:")
            logger.info(
                f"  Avg PSNR improvement per stage: {progressive_improvements['avg_psnr_improvement_per_stage']:.2f} dB"
            )
            logger.info(
                f"  Avg time increase per stage: {progressive_improvements['avg_time_increase_per_stage']:.2f}x"
            )
            logger.info(
                f"  Best resolution: {progressive_improvements['best_resolution']}×{progressive_improvements['best_resolution']}"
            )
            logger.info(
                f"  Most efficient: {progressive_improvements['most_efficient_resolution']}×{progressive_improvements['most_efficient_resolution']}"
            )

    def get_recommendation(
        self,
        metrics_by_resolution: Dict[int, Dict[str, float]],
        constraints: Dict[str, Union[float, str]] = None,
    ) -> Dict[str, Any]:
        """
        Get resolution recommendation based on metrics and constraints.

        Args:
            metrics_by_resolution: Metrics for each resolution
            constraints: User constraints (max_time, min_quality, etc.)

        Returns:
            Recommendation dictionary
        """
        if constraints is None:
            constraints = {}

        max_time = constraints.get("max_time", 30.0)
        min_psnr = constraints.get("min_psnr", 25.0)
        preference = constraints.get(
            "preference", "balanced"
        )  # 'quality', 'speed', 'balanced'

        # Filter resolutions that meet constraints
        valid_resolutions = []
        for resolution, metrics in metrics_by_resolution.items():
            if metrics["inference_time"] <= max_time and metrics["psnr"] >= min_psnr:
                valid_resolutions.append((resolution, metrics))

        if not valid_resolutions:
            # No resolution meets all constraints, find best compromise
            best_resolution = max(
                metrics_by_resolution.keys(),
                key=lambda r: (
                    min(metrics_by_resolution[r]["psnr"], min_psnr)
                    / min_psnr
                    * min(max_time, metrics_by_resolution[r]["inference_time"])
                    / max_time
                ),
            )
        else:
            if preference == "quality":
                best_resolution = max(valid_resolutions, key=lambda x: x[1]["psnr"])[0]
            elif preference == "speed":
                best_resolution = min(
                    valid_resolutions, key=lambda x: x[1]["inference_time"]
                )[0]
            else:  # balanced
                best_resolution = max(
                    valid_resolutions, key=lambda x: x[1]["quality_efficiency"]
                )[0]

        return {
            "recommended_resolution": best_resolution,
            "reason": f"Selected based on {preference} preference",
            "valid_resolutions": [r for r, _ in valid_resolutions],
            "constraints_met": len(valid_resolutions) > 0,
        }
