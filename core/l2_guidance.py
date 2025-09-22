"""
L2 guidance for diffusion sampling - baseline comparison.

This module implements simple L2 (MSE) guidance as a baseline comparison
to the physics-aware Poisson-Gaussian guidance. This provides a standard
deep learning approach without domain-specific physics modeling.

Mathematical Foundation:
- L2 Guidance: ∇ log p(y|x) = 2 * (y - x)
- Simple MSE loss without noise model considerations
- Uniform weighting across all pixels

This serves as a baseline to demonstrate the benefits of physics-aware guidance.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .error_handlers import safe_operation
from .exceptions import GuidanceError, NumericalStabilityError
from .interfaces import GuidanceComputer
from .logging_config import get_logger

logger = get_logger(__name__)


class L2Guidance(GuidanceComputer):
    """
    Simple L2 (MSE) guidance for diffusion sampling.

    This class implements standard L2 guidance without physics-specific
    noise modeling, serving as a baseline comparison to Poisson-Gaussian guidance.
    """

    def __init__(
        self,
        scale: float = 1.0,
        gradient_clip: float = 100.0,
        normalize_gradients: bool = False,
    ):
        """
        Initialize L2 guidance.

        Args:
            scale: Scaling factor for gradients (default: 1.0)
            gradient_clip: Maximum absolute value for guidance gradients
            normalize_gradients: Whether to normalize gradients
        """
        self.scale = scale
        self.gradient_clip = gradient_clip
        self.normalize_gradients = normalize_gradients

        logger.info(
            f"L2Guidance initialized: scale={scale}, "
            f"gradient_clip={gradient_clip}, normalize_gradients={normalize_gradients}"
        )

    @safe_operation("L2 score computation")
    def compute_score(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute L2 guidance score.

        Args:
            x_hat: Current estimate (normalized [0,1])
            y_observed: Observed noisy data (electrons)
            mask: Optional pixel mask
            **kwargs: Additional arguments (ignored for L2)

        Returns:
            L2 guidance gradient
        """
        # For L2 guidance, we assume the observed data is also normalized [0,1]
        # or we need to normalize it to match x_hat

        # If y_observed is in electrons, normalize it by scale
        if y_observed.max() > 2.0:  # Heuristic to detect if data is in electrons
            y_normalized = torch.clamp(y_observed / self.scale, 0.0, 1.0)
        else:
            y_normalized = torch.clamp(y_observed, 0.0, 1.0)

        # Ensure x_hat is in valid range
        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        # Compute L2 gradient: ∇ MSE = 2 * (x_hat - y_normalized)
        # We want to minimize MSE, so gradient points toward target
        gradient = 2.0 * (y_normalized - x_hat)

        # Apply mask if provided
        if mask is not None:
            gradient = gradient * mask

        # Apply gradient clipping for numerical stability
        if self.gradient_clip > 0:
            gradient = torch.clamp(gradient, -self.gradient_clip, self.gradient_clip)

        # Optional gradient normalization
        if self.normalize_gradients:
            # Compute norm over spatial dimensions for each batch element
            grad_norm = torch.norm(
                gradient.view(gradient.shape[0], -1), dim=1, keepdim=True
            )
            grad_norm = grad_norm.view(-1, 1, 1, 1)  # Reshape for broadcasting
            gradient = gradient / torch.clamp(grad_norm, min=1e-8)

        return gradient

    def gamma_schedule(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Simple constant gamma schedule for L2 guidance.

        Args:
            sigma: Noise level (ignored for L2)

        Returns:
            Constant gamma value
        """
        return torch.ones_like(sigma)

    def compute_likelihood(
        self,
        x: torch.Tensor,
        y_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute L2 likelihood (negative MSE).

        Args:
            x: Predicted image (normalized [0,1])
            y_observed: Observed data
            mask: Optional pixel mask
            **kwargs: Additional arguments (ignored)

        Returns:
            Negative MSE loss
        """
        # Normalize y_observed if needed
        if y_observed.max() > 2.0:
            y_normalized = torch.clamp(y_observed / self.scale, 0.0, 1.0)
        else:
            y_normalized = torch.clamp(y_observed, 0.0, 1.0)

        x = torch.clamp(x, 0.0, 1.0)

        # Compute MSE
        mse = torch.mean((x - y_normalized) ** 2, dim=[1, 2, 3])

        # Apply mask if provided
        if mask is not None:
            mask_mean = torch.mean(mask.float(), dim=[1, 2, 3])
            mse = mse * mask_mean

        # Return negative MSE as "likelihood"
        return -mse

    def validate_inputs(
        self, x_hat: torch.Tensor, y_observed: torch.Tensor
    ) -> Tuple[bool, str]:
        """
        Validate input tensors for L2 guidance.

        Args:
            x_hat: Current estimate
            y_observed: Observed data

        Returns:
            Tuple of (is_valid, error_message)
        """
        if x_hat.shape != y_observed.shape:
            return False, f"Shape mismatch: {x_hat.shape} vs {y_observed.shape}"

        if torch.isnan(x_hat).any() or torch.isnan(y_observed).any():
            return False, "NaN values detected in inputs"

        if torch.isinf(x_hat).any() or torch.isinf(y_observed).any():
            return False, "Inf values detected in inputs"

        return True, ""

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information for L2 guidance.

        Returns:
            Dictionary of diagnostic information
        """
        return {
            "guidance_type": "L2",
            "scale": self.scale,
            "gradient_clip": self.gradient_clip,
            "normalize_gradients": self.normalize_gradients,
        }
