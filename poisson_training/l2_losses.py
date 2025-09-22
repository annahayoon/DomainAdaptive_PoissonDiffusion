"""
L2 loss functions for baseline diffusion training.

This module provides simple L2 (MSE) loss functions as a baseline comparison
to the physics-aware Poisson-Gaussian losses. These implement standard
deep learning approaches without domain-specific physics modeling.

Key features:
- Simple MSE loss for reconstruction
- Standard diffusion model losses
- No physics-specific considerations

This serves as a baseline to demonstrate the benefits of physics-aware training.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.logging_config import get_logger

logger = get_logger(__name__)


class L2Loss(nn.Module):
    """
    Simple L2 (MSE) loss function for baseline comparison.

    This loss function implements standard MSE loss without any physics-specific
    considerations, providing a baseline comparison to Poisson-Gaussian loss.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        eps: float = 1e-8,
    ):
        """
        Initialize L2 loss.

        Args:
            weights: Loss component weights
            eps: Small constant for numerical stability (unused in L2)
        """
        super().__init__()
        self.weights = weights or {"reconstruction": 1.0, "consistency": 0.1}
        self.eps = eps

        logger.info(f"Initialized L2Loss with weights: {self.weights}")

    def mse_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE loss between prediction and target.

        Args:
            prediction: Model prediction (normalized [0,1])
            target: Target observation (normalized [0,1] or electrons)

        Returns:
            MSE loss
        """
        # Ensure both tensors are in the same range
        # If target has values > 2, assume it's in electrons and needs normalization
        if target.max() > 2.0:
            # Simple normalization - assume max value represents scale
            target_max = target.max()
            target_normalized = torch.clamp(target / target_max, 0.0, 1.0)
        else:
            target_normalized = torch.clamp(target, 0.0, 1.0)

        prediction = torch.clamp(prediction, 0.0, 1.0)

        # Compute MSE
        mse = F.mse_loss(prediction, target_normalized, reduction="mean")

        # Clamp for numerical stability
        mse = torch.clamp(mse, min=0.0, max=1e4)

        # Check for NaN and replace with large but finite loss
        if torch.isnan(mse).any() or torch.isinf(mse).any():
            logger.warning(
                "NaN or Inf detected in L2 loss computation, replacing with finite value"
            )
            mse = torch.where(
                torch.isnan(mse) | torch.isinf(mse),
                torch.tensor(10.0, device=mse.device, dtype=mse.dtype),
                mse,
            )

        return mse

    def consistency_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute simple consistency loss for L2 baseline.

        Args:
            prediction: Model prediction (normalized [0,1])
            target: Target observation

        Returns:
            Consistency loss (simple range constraint)
        """
        # Clamp inputs for stability
        prediction = torch.clamp(prediction, min=-10.0, max=10.0)

        # Ensure predictions are in valid range [0,1]
        range_loss = F.relu(-prediction).mean() + F.relu(prediction - 1).mean()
        range_loss = torch.clamp(range_loss, min=0.0, max=100.0)

        # Check for NaN
        if torch.isnan(range_loss).any():
            logger.warning("NaN detected in consistency loss, replacing with zero")
            range_loss = torch.tensor(0.0, device=range_loss.device)

        return range_loss

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        domain_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute L2 loss components.

        Args:
            outputs: Model outputs containing 'prediction'
            batch: Batch data containing target
            domain_params: Domain parameters (ignored for L2)

        Returns:
            Dictionary of loss components
        """
        prediction = outputs.get(
            "prediction", outputs.get("pred", outputs.get("output"))
        )
        if prediction is None:
            raise ValueError("No prediction found in outputs")

        # Try different target keys
        target = batch.get("clean", batch.get("target", batch.get("electrons")))
        if target is None:
            raise ValueError("No target found in batch")

        # Compute reconstruction loss
        reconstruction_loss = self.mse_loss(prediction, target)

        # Compute consistency loss
        consistency_loss = self.consistency_loss(prediction, target)

        # Combine losses
        total_loss = (
            self.weights["reconstruction"] * reconstruction_loss
            + self.weights["consistency"] * consistency_loss
        )

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "consistency_loss": consistency_loss,
            "l2_loss": reconstruction_loss,  # Alias for compatibility
        }


class L2DiffusionLoss(nn.Module):
    """
    L2 loss for diffusion training (standard approach).

    This implements standard diffusion training without physics-specific
    considerations, using simple MSE loss.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        parameterization: str = "noise",
        weighting: str = "uniform",
    ):
        """
        Initialize L2 diffusion loss.

        Args:
            loss_type: Type of base loss ("mse" or "huber")
            parameterization: Prediction target ("noise", "x0", or "v")
            weighting: Loss weighting scheme ("uniform" or "snr")
        """
        super().__init__()
        self.parameterization = parameterization
        self.weighting = weighting

        if loss_type == "mse":
            self.base_loss = nn.MSELoss(reduction="none")
        elif loss_type == "huber":
            self.base_loss = nn.SmoothL1Loss(reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        logger.info(
            f"Initialized L2DiffusionLoss: {loss_type}, {parameterization}, {weighting}"
        )

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
            # SNR weighting: weight = α_t / (1 - α_t)
            alpha_t = alphas_cumprod[timesteps]
            snr = alpha_t / (1 - alpha_t)
            return snr / (snr + 1)  # Normalized SNR weighting
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L2 diffusion loss.

        Args:
            prediction: Model prediction
            target: Target (based on parameterization)
            timesteps: Diffusion timesteps
            alphas_cumprod: Cumulative alpha values

        Returns:
            Diffusion loss
        """
        # Compute base loss
        loss = self.base_loss(prediction, target)

        # Get loss weights
        weights = self.get_loss_weights(timesteps, alphas_cumprod)
        weights = weights.view(-1, 1, 1, 1)  # Reshape for broadcasting

        # Apply weights
        weighted_loss = loss * weights

        # Reduce to scalar
        return weighted_loss.mean()


class L2CombinedLoss(nn.Module):
    """
    Combined L2 loss function for baseline training.
    """

    def __init__(
        self,
        loss_components: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.loss_components = nn.ModuleDict(loss_components)
        self.weights = weights or {name: 1.0 for name in loss_components.keys()}

        logger.info(
            f"Initialized L2CombinedLoss with components: {list(loss_components.keys())}"
        )

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined L2 loss.

        Args:
            outputs: Model outputs
            batch: Batch data

        Returns:
            Dictionary of loss components
        """
        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.loss_components.items():
            component_loss = loss_fn(outputs, batch)

            if isinstance(component_loss, dict):
                # Loss function returns multiple components
                for comp_name, comp_value in component_loss.items():
                    loss_dict[f"{name}_{comp_name}"] = comp_value
                    if "total" in comp_name.lower():
                        total_loss += self.weights.get(name, 1.0) * comp_value
            else:
                # Loss function returns single value
                loss_dict[name] = component_loss
                total_loss += self.weights.get(name, 1.0) * component_loss

        loss_dict["total_loss"] = total_loss
        return loss_dict
