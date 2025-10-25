#!/usr/bin/env python3
"""
Physics-Aware EDM Sampler

Implements the correct "Un-normalize, Guide, Re-normalize" sampling loop
that properly handles the separation between the normalized prior space
and the physical likelihood space.

This addresses the critical insight that:
- Prior model p_θ(x) operates in normalized [0,1] space
- Likelihood guidance ∇log p(y|x) operates in physical space
- Proper sampling requires careful coordinate transformations between spaces
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.guidance_config import GuidanceConfig
from core.logging_config import get_logger
from core.poisson_guidance import PoissonGuidance
from core.transforms import ImageMetadata
from models.edm_wrapper import DomainEncoder, EDMModelWrapper

logger = get_logger(__name__)


class PhysicsAwareEDMSampler:
    """
    Physics-aware EDM sampler with proper coordinate transformations.

    This sampler implements the correct separation between:
    1. Prior denoising: p_θ(x) in normalized [0,1] space
    2. Likelihood guidance: ∇log p(y|x) in physical space

    Key insight: The catastrophic failure of naive implementation demonstrates
    why physics-informed guidance requires careful attention to physical units.
    """

    def __init__(
        self,
        model: EDMModelWrapper,
        guidance: PoissonGuidance,
        device: str = "cuda",
    ):
        """
        Initialize physics-aware sampler.

        Args:
            model: EDM model trained on normalized [0,1] data
            guidance: Poisson-Gaussian guidance in physical space
            device: Computation device
        """
        self.model = model
        self.guidance = guidance
        self.device = device

        # Extract physical parameters from guidance
        self.scale = guidance.scale  # Max photon count (s)
        self.background = guidance.background  # Background offset
        self.read_noise = guidance.read_noise  # Read noise std (σ_r)

        logger.info(f"PhysicsAwareEDMSampler initialized")
        logger.info(
            f"Physical parameters: scale={self.scale}, bg={self.background}, σ_r={self.read_noise}"
        )

    def normalize_to_model_space(self, x_physical: torch.Tensor) -> torch.Tensor:
        """Convert from physical space to normalized [0,1] model space."""
        return torch.clamp(x_physical / self.scale, 0, 1)

    def unnormalize_to_physical_space(self, x_normalized: torch.Tensor) -> torch.Tensor:
        """Convert from normalized [0,1] model space to physical space."""
        return x_normalized * self.scale

    def compute_physics_guidance(
        self,
        x_hat_0_normalized: torch.Tensor,
        y_observed_physical: torch.Tensor,
        sigma_t: torch.Tensor,
        guidance_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute guidance gradient in physical space with proper normalization.

        This implements the core "Un-normalize, Guide, Re-normalize" process:
        1. Convert model prediction to physical space
        2. Compute guidance gradient in physical space
        3. Apply guidance update in physical space
        4. Convert result back to normalized space

        Args:
            x_hat_0_normalized: Model's clean image prediction in [0,1]
            y_observed_physical: Noisy observation in physical units
            sigma_t: Current noise level
            guidance_weight: Guidance strength multiplier

        Returns:
            Guided prediction in normalized [0,1] space
        """
        # Step 1: Un-normalize model prediction to physical space
        x_hat_0_physical = self.unnormalize_to_physical_space(x_hat_0_normalized)

        # Step 2: Compute likelihood gradient in physical space
        # WLS score: ∇ ← s · (y_e - s*x̂_0) / (s*x̂_0 + σ_r²)
        numerator = y_observed_physical - x_hat_0_physical
        denominator = x_hat_0_physical + self.read_noise**2

        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-8)

        # Gradient in physical space
        gradient_physical = self.scale * numerator / denominator

        # Step 3: Apply guidance update in physical space
        # Update step size: κ * σ_t²
        update_step_size = guidance_weight * (sigma_t**2)

        # Apply update
        x_hat_0_guided_physical = (
            x_hat_0_physical + update_step_size * gradient_physical
        )

        # Step 4: Re-normalize back to [0,1] model space
        x_hat_0_guided_normalized = self.normalize_to_model_space(
            x_hat_0_guided_physical
        )

        # Log diagnostics
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug(f"Guidance step diagnostics:")
            logger.debug(
                f"  x̂_0 range (norm): [{x_hat_0_normalized.min():.3f}, {x_hat_0_normalized.max():.3f}]"
            )
            logger.debug(
                f"  x̂_0 range (phys): [{x_hat_0_physical.min():.1f}, {x_hat_0_physical.max():.1f}]"
            )
            logger.debug(
                f"  y_obs range: [{y_observed_physical.min():.1f}, {y_observed_physical.max():.1f}]"
            )
            logger.debug(f"  gradient norm: {torch.norm(gradient_physical):.3f}")
            logger.debug(
                f"  update norm: {torch.norm(update_step_size * gradient_physical):.3f}"
            )
            logger.debug(
                f"  guided range (norm): [{x_hat_0_guided_normalized.min():.3f}, {x_hat_0_guided_normalized.max():.3f}]"
            )

        return x_hat_0_guided_normalized

    def edm_step(
        self,
        x_t: torch.Tensor,
        x_hat_0: torch.Tensor,
        sigma_t: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one EDM denoising step.

        This implements the EDM update rule:
        x_{t-1} = x̂_0 + (σ_{t-1}/σ_t) * (x_t - x̂_0)
        """
        if sigma_next == 0:
            return x_hat_0

        # EDM step formula
        ratio = sigma_next / sigma_t
        x_next = x_hat_0 + ratio * (x_t - x_hat_0)

        return x_next

    def create_noise_schedule(self, num_steps: int = 18) -> torch.Tensor:
        """Create EDM noise schedule."""
        sigma_min, sigma_max = 0.002, 80.0
        rho = 7.0

        # EDM noise schedule
        step_indices = torch.arange(num_steps, device=self.device)
        sigmas = sigma_min ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_max ** (1 / rho) - sigma_min ** (1 / rho)
        )
        sigmas = sigmas**rho

        # Add sigma=0 at the end
        sigmas = torch.cat([sigmas, torch.zeros(1, device=self.device)])

        return sigmas

    def sample(
        self,
        y_observed: torch.Tensor,
        metadata: ImageMetadata,
        condition: torch.Tensor,
        steps: int = 18,
        guidance_weight: float = 1.0,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Physics-aware sampling with proper coordinate transformations.

        Args:
            y_observed: Noisy observation in physical units
            metadata: Image metadata
            condition: Domain conditioning vector
            steps: Number of denoising steps
            guidance_weight: Guidance strength
            return_intermediates: Whether to return intermediate states

        Returns:
            Dictionary with 'sample' and optionally 'intermediates'
        """
        batch_size = y_observed.shape[0]
        img_shape = y_observed.shape

        logger.info(f"Starting physics-aware sampling:")
        logger.info(f"  Steps: {steps}, guidance_weight: {guidance_weight}")
        logger.info(
            f"  y_observed range: [{y_observed.min():.1f}, {y_observed.max():.1f}]"
        )
        logger.info(f"  Physical params: s={self.scale}, σ_r={self.read_noise}")

        # Create noise schedule
        sigmas = self.create_noise_schedule(steps)

        # Initialize from pure noise (in normalized space)
        x_t = torch.randn(img_shape, device=self.device)

        # Storage for intermediates
        intermediates = [] if return_intermediates else None

        # Sampling loop
        for i in tqdm(range(steps), desc="Physics-aware sampling"):
            sigma_t = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Step 1: Get prior model's prediction (in normalized space)
            with torch.no_grad():
                # Model expects normalized input and produces normalized output
                # Pass conditioning via 'condition' parameter
                x_hat_0_prior = self.model(
                    x_t, sigma_t.unsqueeze(0), condition=condition
                )

                # Ensure output is in [0,1] (model should do this, but be safe)
                x_hat_0_prior = torch.clamp(x_hat_0_prior, 0, 1)

            # Step 2: Apply physics-aware guidance correction
            if guidance_weight > 0:
                x_hat_0_guided = self.compute_physics_guidance(
                    x_hat_0_normalized=x_hat_0_prior,
                    y_observed_physical=y_observed,
                    sigma_t=sigma_t,
                    guidance_weight=guidance_weight,
                )
            else:
                x_hat_0_guided = x_hat_0_prior

            # Step 3: Perform EDM denoising step (in normalized space)
            x_t = self.edm_step(x_t, x_hat_0_guided, sigma_t, sigma_next)

            # Store intermediate if requested
            if return_intermediates:
                intermediates.append(
                    {
                        "step": i,
                        "sigma_t": sigma_t.item(),
                        "x_t": x_t.clone(),
                        "x_hat_0_prior": x_hat_0_prior.clone(),
                        "x_hat_0_guided": x_hat_0_guided.clone(),
                    }
                )

            # Log progress
            if i % 5 == 0 or i == steps - 1:
                x_range = f"[{x_t.min():.3f}, {x_t.max():.3f}]"
                logger.debug(f"Step {i:2d}: σ={sigma_t:.3f}, x_t range: {x_range}")

        # Final result should be in [0,1] normalized space
        final_sample = torch.clamp(x_t, 0, 1)

        logger.info(
            f"Sampling complete. Final range: [{final_sample.min():.3f}, {final_sample.max():.3f}]"
        )

        result = {"sample": final_sample}
        if return_intermediates:
            result["intermediates"] = intermediates

        return result


def create_physics_aware_sampler(
    model: EDMModelWrapper,
    scale: float,
    background: float,
    read_noise: float,
    guidance_weight: float = 1.0,
    device: str = "cuda",
) -> PhysicsAwareEDMSampler:
    """
    Factory function to create physics-aware sampler.

    Args:
        model: EDM model wrapper
        scale: Physical scale parameter (max photon count)
        background: Background offset
        read_noise: Read noise standard deviation
        guidance_weight: Guidance strength (0.0 for no guidance)
        device: Computation device

    Returns:
        Configured PhysicsAwareEDMSampler
    """
    # Create guidance configuration (handle zero guidance case)
    if guidance_weight > 0:
        guidance_config = GuidanceConfig(
            kappa=guidance_weight,
            gamma_schedule="sigma2",
            gradient_clip=100.0,  # Can be higher now that we have proper scaling
        )

        # Create Poisson guidance
        guidance = PoissonGuidance(
            scale=scale,
            background=background,
            read_noise=read_noise,
            config=guidance_config,
        )
    else:
        # For no-guidance baseline, create minimal guidance that won't be used
        guidance_config = GuidanceConfig(
            kappa=1.0,  # Dummy value, won't be used
            gamma_schedule="sigma2",
            gradient_clip=100.0,
        )

        guidance = PoissonGuidance(
            scale=scale,
            background=background,
            read_noise=read_noise,
            config=guidance_config,
        )

    return PhysicsAwareEDMSampler(
        model=model,
        guidance=guidance,
        device=device,
    )
