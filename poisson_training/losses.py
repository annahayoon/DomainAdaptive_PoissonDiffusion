"""
Loss functions for Poisson-Gaussian diffusion training.

This module provides specialized loss functions for training diffusion models
with physics-aware Poisson-Gaussian noise modeling.

Key features:
- Poisson-Gaussian likelihood loss
- Diffusion model losses (v-parameterization, noise prediction)
- Consistency losses for physics constraints
- Weighted multi-objective losses

Requirements addressed: 4.2-4.6 from requirements.md
"""

import math
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.error_handlers import safe_operation
from core.logging_config import get_logger

logger = get_logger(__name__)


class PoissonGaussianLoss(nn.Module):
    """
    Physics-aware loss function for Poisson-Gaussian noise model.

    This loss function implements the negative log-likelihood of the
    Poisson-Gaussian noise model, providing physics-consistent training.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        eps: float = 1e-8,
        use_exact_likelihood: bool = True,
    ):
        """
        Initialize Poisson-Gaussian loss.

        Args:
            weights: Loss component weights
            eps: Small constant for numerical stability
            use_exact_likelihood: Whether to use exact likelihood (vs approximation)
        """
        super().__init__()
        self.weights = weights or {"reconstruction": 1.0, "consistency": 0.1}
        self.eps = eps
        self.use_exact_likelihood = use_exact_likelihood

        logger.info(f"Initialized PoissonGaussianLoss with weights: {self.weights}")

    def poisson_gaussian_nll(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        scale: torch.Tensor,
        background: torch.Tensor,
        read_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Poisson-Gaussian negative log-likelihood.

        Args:
            prediction: Model prediction (normalized [0,1])
            target: Target observation (electrons)
            scale: Normalization scale (electrons)
            background: Background level (electrons)
            read_noise: Read noise std (electrons)

        Returns:
            Negative log-likelihood loss
        """
        # Convert prediction back to electrons
        # Ensure proper broadcasting
        if scale.numel() > 1:
            scale = scale.view(-1, 1, 1, 1)
        if background.numel() > 1:
            background = background.view(-1, 1, 1, 1)
        if read_noise.numel() > 1:
            read_noise = read_noise.view(-1, 1, 1, 1)

        pred_electrons = prediction * scale + background
        pred_electrons = torch.clamp(pred_electrons, min=self.eps)

        if self.use_exact_likelihood:
            # Exact Poisson-Gaussian likelihood
            # For observation y ~ Poisson(λ) + N(0, σ²)
            # where λ = pred_electrons, σ = read_noise

            # Poisson component: -λ + y*log(λ) - log(y!)
            poisson_term = -pred_electrons + target * torch.log(
                pred_electrons + self.eps
            )

            # Gaussian component: -0.5 * ((y - λ) / σ)²
            gaussian_term = (
                -0.5 * ((target - pred_electrons) / (read_noise + self.eps)) ** 2
            )

            # Combined likelihood (ignoring constants)
            nll = -(poisson_term + gaussian_term)

        else:
            # Approximation: weighted MSE with Poisson variance
            variance = pred_electrons + read_noise**2
            mse = (target - pred_electrons) ** 2
            nll = mse / (variance + self.eps)

        return nll.mean()

    def consistency_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss to ensure physical constraints.

        Args:
            prediction: Model prediction (normalized [0,1])
            target: Target observation (electrons)
            scale: Normalization scale (electrons)

        Returns:
            Consistency loss
        """
        # Ensure predictions are in valid range
        range_loss = F.relu(-prediction).mean() + F.relu(prediction - 1).mean()

        # Ensure energy conservation (optional)
        # Ensure proper broadcasting
        if scale.numel() > 1:
            scale = scale.view(-1, 1, 1, 1)

        pred_electrons = prediction * scale
        target_norm = target / scale
        energy_loss = torch.abs(pred_electrons.mean() - target_norm.mean())

        return range_loss + 0.1 * energy_loss

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            outputs: Model outputs
            batch: Training batch

        Returns:
            Dictionary of loss components
        """
        prediction = outputs.get("prediction", outputs.get("denoised"))
        target = batch["electrons"]
        scale = batch.get("scale", torch.tensor(1000.0, device=prediction.device))
        background = batch.get(
            "background", torch.tensor(0.0, device=prediction.device)
        )
        read_noise = batch.get(
            "read_noise", torch.tensor(5.0, device=prediction.device)
        )

        # Ensure tensors have correct shapes
        if scale.numel() == 1:
            scale = scale.expand(prediction.shape[0])
        if background.numel() == 1:
            background = background.expand(prediction.shape[0])
        if read_noise.numel() == 1:
            read_noise = read_noise.expand(prediction.shape[0])

        losses = {}

        # Reconstruction loss
        losses["reconstruction"] = self.poisson_gaussian_nll(
            prediction, target, scale, background, read_noise
        )

        # Consistency loss
        losses["consistency"] = self.consistency_loss(prediction, target, scale)

        # Apply weights
        weighted_losses = {}
        for key, loss in losses.items():
            weight = self.weights.get(key, 1.0)
            weighted_losses[key] = weight * loss

        return weighted_losses


class DiffusionLoss(nn.Module):
    """
    Loss function for diffusion model training.

    Supports various parameterizations (noise prediction, v-parameterization, etc.)
    """

    def __init__(
        self,
        loss_type: str = "mse",
        parameterization: str = "noise",
        weighting: str = "uniform",
    ):
        """
        Initialize diffusion loss.

        Args:
            loss_type: Type of loss ("mse", "l1", "huber")
            parameterization: Model parameterization ("noise", "v", "x0")
            weighting: Loss weighting scheme ("uniform", "snr", "min_snr")
        """
        super().__init__()
        self.loss_type = loss_type
        self.parameterization = parameterization
        self.weighting = weighting

        # Base loss function
        if loss_type == "mse":
            self.base_loss = nn.MSELoss(reduction="none")
        elif loss_type == "l1":
            self.base_loss = nn.L1Loss(reduction="none")
        elif loss_type == "huber":
            self.base_loss = nn.SmoothL1Loss(reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def get_target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get target based on parameterization.

        Args:
            x0: Clean image
            noise: Added noise
            timesteps: Diffusion timesteps
            alphas_cumprod: Cumulative alpha values

        Returns:
            Target tensor
        """
        if self.parameterization == "noise":
            return noise
        elif self.parameterization == "x0":
            return x0
        elif self.parameterization == "v":
            # v-parameterization: v = α_t * noise - σ_t * x0
            alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sigma_t = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
            return alpha_t.sqrt() * noise - sigma_t * x0
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")

    def get_loss_weights(
        self, timesteps: torch.Tensor, alphas_cumprod: torch.Tensor
    ) -> torch.Tensor:
        """
        Get loss weights based on weighting scheme.

        Args:
            timesteps: Diffusion timesteps
            alphas_cumprod: Cumulative alpha values

        Returns:
            Loss weights
        """
        if self.weighting == "uniform":
            return torch.ones_like(timesteps, dtype=torch.float32)
        elif self.weighting == "snr":
            # Signal-to-noise ratio weighting
            alpha_t = alphas_cumprod[timesteps]
            snr = alpha_t / (1 - alpha_t)
            return snr
        elif self.weighting == "min_snr":
            # Min-SNR weighting (clipped SNR)
            alpha_t = alphas_cumprod[timesteps]
            snr = alpha_t / (1 - alpha_t)
            return torch.clamp(snr, max=5.0)
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion loss.

        Args:
            outputs: Model outputs
            batch: Training batch

        Returns:
            Dictionary of loss components
        """
        prediction = outputs["prediction"]
        target = self.get_target(
            batch["x0"], batch["noise"], batch["timesteps"], batch["alphas_cumprod"]
        )

        # Compute base loss
        loss = self.base_loss(prediction, target)

        # Apply weighting
        weights = self.get_loss_weights(batch["timesteps"], batch["alphas_cumprod"])
        weights = weights.view(-1, 1, 1, 1)

        weighted_loss = (loss * weights).mean()

        return {"diffusion_loss": weighted_loss}


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for ensuring physical constraints.
    """

    def __init__(self, lambda_consistency: float = 0.1):
        super().__init__()
        self.lambda_consistency = lambda_consistency

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute consistency loss.

        Args:
            outputs: Model outputs
            batch: Training batch

        Returns:
            Dictionary of loss components
        """
        prediction = outputs.get("prediction", outputs.get("denoised"))

        losses = {}

        # Range consistency: predictions should be in [0, 1]
        range_loss = F.relu(-prediction).mean() + F.relu(prediction - 1).mean()
        losses["range_consistency"] = self.lambda_consistency * range_loss

        # Smoothness consistency (optional)
        if prediction.dim() == 4:  # [B, C, H, W]
            grad_x = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])
            grad_y = torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :])
            smoothness_loss = grad_x.mean() + grad_y.mean()
            losses["smoothness_consistency"] = 0.01 * smoothness_loss

        return losses


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained features.
    """

    def __init__(
        self, feature_layers: Optional[list] = None, lambda_perceptual: float = 1.0
    ):
        super().__init__()
        self.lambda_perceptual = lambda_perceptual

        # Use VGG features (simplified version)
        try:
            import torchvision.models as models

            vgg = models.vgg16(pretrained=True).features
            self.feature_extractor = nn.Sequential(
                *list(vgg.children())[:16]
            )  # Up to conv3_3
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        except ImportError:
            logger.warning("torchvision not available, perceptual loss disabled")
            self.feature_extractor = None

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute perceptual loss.

        Args:
            outputs: Model outputs
            batch: Training batch

        Returns:
            Dictionary of loss components
        """
        if self.feature_extractor is None:
            return {}

        prediction = outputs.get("prediction", outputs.get("denoised"))
        target = batch.get("clean", batch.get("target"))

        if target is None:
            return {}

        # Convert to 3-channel if needed
        if prediction.shape[1] == 1:
            prediction = prediction.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        # Extract features
        pred_features = self.feature_extractor(prediction)
        target_features = self.feature_extractor(target)

        # Compute perceptual loss
        perceptual_loss = F.mse_loss(pred_features, target_features)

        return {"perceptual_loss": self.lambda_perceptual * perceptual_loss}


class CombinedLoss(nn.Module):
    """
    Combined loss function that can incorporate multiple loss types.
    """

    def __init__(
        self,
        loss_components: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.loss_components = nn.ModuleDict(loss_components)
        self.weights = weights or {name: 1.0 for name in loss_components.keys()}

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs
            batch: Training batch

        Returns:
            Dictionary of all loss components
        """
        all_losses = {}

        for name, loss_fn in self.loss_components.items():
            component_losses = loss_fn(outputs, batch)

            # Apply weight to each component
            weight = self.weights.get(name, 1.0)
            for loss_name, loss_value in component_losses.items():
                weighted_name = (
                    f"{name}_{loss_name}" if len(component_losses) > 1 else name
                )
                all_losses[weighted_name] = weight * loss_value

        return all_losses


# Utility functions
def create_loss_function(loss_type: str = "poisson_gaussian", **kwargs) -> nn.Module:
    """
    Create loss function based on type.

    Args:
        loss_type: Type of loss function
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function
    """
    if loss_type == "poisson_gaussian":
        return PoissonGaussianLoss(**kwargs)
    elif loss_type == "diffusion":
        return DiffusionLoss(**kwargs)
    elif loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "combined":
        # Create combined loss with common components
        components = {
            "reconstruction": PoissonGaussianLoss(),
            "consistency": ConsistencyLoss(),
        }
        if kwargs.get("use_perceptual", False):
            components["perceptual"] = PerceptualLoss()

        return CombinedLoss(components, kwargs.get("weights"))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
