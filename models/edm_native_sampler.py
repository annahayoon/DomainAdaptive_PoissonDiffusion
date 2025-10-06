"""
EDM Native Sampler with Physics-Aware Guidance.

This module provides a wrapper around EDM's native sampler that integrates
Poisson-Gaussian (PG) and L2 guidance for physics-aware image restoration.

Key features:
- Uses EDM's native edm_sampler for optimal performance
- Integrates physics-aware guidance (PG and L2) during inference
- Supports configurable guidance strength and scheduling
- Maintains EDM's native sampling efficiency with proper guidance integration
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from core.exceptions import ConfigurationError, SamplingError
from core.l2_guidance import L2Guidance
from core.logging_config import get_logger
from core.poisson_guidance import PoissonGuidance
from models.edm_wrapper import EDMModelWrapper

logger = get_logger(__name__)


@dataclass
class EDMNativeSamplingConfig:
    """Configuration for EDM native sampling with guidance."""

    # EDM sampling parameters (native EDM defaults)
    num_steps: int = 18
    """Number of sampling steps"""

    sigma_min: float = 0.002
    """Minimum noise level"""

    sigma_max: float = 80.0
    """Maximum noise level"""

    rho: float = 7.0
    """Time step exponent"""

    # Stochastic sampling parameters
    S_churn: float = 0.0
    """Stochasticity parameter"""

    S_min: float = 0.0
    """Minimum sigma for stochasticity"""

    S_max: float = float("inf")
    """Maximum sigma for stochasticity"""

    S_noise: float = 1.0
    """Noise scale for stochastic sampling"""

    # Guidance parameters
    guidance_scale: float = 1.0
    """Overall guidance strength multiplier"""

    guidance_start_step: int = 0
    """Step to start applying guidance (0 = from beginning)"""

    guidance_end_step: Optional[int] = None
    """Step to stop applying guidance (None = until end)"""

    guidance_type: str = "pg"
    """Type of guidance: 'pg' (Poisson-Gaussian) or 'l2' (L2)"""

    # Numerical stability
    clip_denoised: bool = True
    """Whether to clip denoised predictions"""

    clip_range: Tuple[float, float] = (0.0, 1.0)
    """Range for clipping denoised predictions"""

    def __post_init__(self):
        """Validate configuration."""
        if self.num_steps <= 0:
            raise ConfigurationError(
                f"num_steps must be positive, got {self.num_steps}"
            )

        if self.sigma_min >= self.sigma_max:
            raise ConfigurationError(
                f"sigma_min ({self.sigma_min}) must be < sigma_max ({self.sigma_max})"
            )

        if (
            self.guidance_end_step is not None
            and self.guidance_end_step <= self.guidance_start_step
        ):
            raise ConfigurationError("guidance_end_step must be > guidance_start_step")

        if self.guidance_type not in ["pg", "l2"]:
            raise ConfigurationError(f"Unknown guidance type: {self.guidance_type}")


class EDMNativeSampler:
    """
    EDM Native Sampler with Physics-Aware Guidance.

    This sampler uses EDM's native edm_sampler function and integrates
    physics-aware guidance (PG and L2) for image restoration.
    """

    def __init__(
        self,
        model: EDMModelWrapper,
        guidance: Optional[Union[PoissonGuidance, L2Guidance]] = None,
        config: Optional[EDMNativeSamplingConfig] = None,
    ):
        """
        Initialize EDM native sampler with guidance.

        Args:
            model: EDM model wrapper
            guidance: Physics guidance (PG or L2, optional for unconditional sampling)
            config: Sampling configuration
        """
        self.model = model
        self.guidance = guidance
        self.config = config or EDMNativeSamplingConfig()

        logger.info(
            f"EDMNativeSampler initialized: {self.config.num_steps} steps, "
            f"guidance={self.config.guidance_type if guidance else 'None'}, "
            f"guidance_scale={self.config.guidance_scale}"
        )

    def _should_apply_guidance(self, step: int) -> bool:
        """Check if guidance should be applied at current step."""
        if self.guidance is None:
            return False

        if step < self.config.guidance_start_step:
            return False

        if (
            self.config.guidance_end_step is not None
            and step >= self.config.guidance_end_step
        ):
            return False

        return True

    def _apply_guidance(
        self,
        denoised: torch.Tensor,
        y_observed: torch.Tensor,
        sigma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply physics-aware guidance to denoised prediction."""
        if self.guidance is None:
            return denoised

        # Compute guidance correction based on guidance type
        if self.config.guidance_type == "pg":
            # Poisson-Gaussian guidance
            guidance_correction = self.guidance.compute(
                denoised, y_observed, sigma, mask=mask
            )
        elif self.config.guidance_type == "l2":
            # L2 guidance - simple mean squared error guidance
            guidance_correction = y_observed - denoised
        else:
            raise ConfigurationError(
                f"Unknown guidance type: {self.config.guidance_type}"
            )

        # Scale guidance
        guidance_correction = guidance_correction * self.config.guidance_scale

        # Apply guidance to denoised prediction
        denoised_guided = denoised + guidance_correction

        return denoised_guided

    def sample(
        self,
        y_observed: torch.Tensor,
        domain: Optional[Union[str, List[str], torch.Tensor]] = None,
        scale: Optional[Union[float, torch.Tensor]] = None,
        read_noise: Optional[Union[float, torch.Tensor]] = None,
        background: Optional[Union[float, torch.Tensor]] = None,
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample using EDM's native sampler with physics-aware guidance.

        Args:
            y_observed: Observed noisy image [B, C, H, W] (electrons)
            domain: Domain specification
            scale: Scale parameter
            read_noise: Read noise parameter
            background: Background parameter
            condition: Pre-computed conditioning vector
            mask: Valid pixel mask
            **kwargs: Additional arguments for EDM sampler

        Returns:
            Dictionary with sampling results
        """
        # Prepare conditioning
        if condition is None:
            if any(p is None for p in [domain, scale, read_noise, background]):
                raise ValueError(
                    "Must provide either 'condition' or all domain parameters"
                )

            condition = self.model.encode_conditioning(
                domain=domain,
                scale=scale,
                read_noise=read_noise,
                background=background,
                device=y_observed.device,
            )

        # Initialize with noise
        batch_size, channels, height, width = y_observed.shape
        device = y_observed.device

        # Start from pure noise (EDM native approach)
        latents = torch.randn(batch_size, channels, height, width, device=device)

        logger.info(
            f"Starting EDM native sampling: {self.config.num_steps} steps with {self.config.guidance_type} guidance"
        )

        # Use our custom guided sampler that integrates with EDM's structure
        result = self._guided_edm_sampler(
            net=self.model,
            latents=latents,
            class_labels=condition,
            y_observed=y_observed,
            mask=mask,
            num_steps=self.config.num_steps,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
            rho=self.config.rho,
            S_churn=self.config.S_churn,
            S_min=self.config.S_min,
            S_max=self.config.S_max,
            S_noise=self.config.S_noise,
            **kwargs,
        )

        logger.info("EDM native sampling with guidance completed")

        return {
            "x_final": result,
            "condition": condition,
        }

    def _guided_edm_sampler(
        self,
        net,
        latents,
        class_labels=None,
        y_observed=None,
        mask=None,
        randn_like=torch.randn_like,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        """EDM sampler with integrated guidance (modified from EDM's edm_sampler)."""
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop with guidance integration.
        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(
                x_cur
            )

            # Euler step with guidance.
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)

            # Apply guidance if enabled and in the right step range
            if self._should_apply_guidance(i) and y_observed is not None:
                denoised = self._apply_guidance(denoised, y_observed, t_hat, mask)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)

                # Apply guidance to 2nd order correction as well
                if self._should_apply_guidance(i) and y_observed is not None:
                    denoised = self._apply_guidance(denoised, y_observed, t_next, mask)

                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


def create_edm_native_sampler(
    model: EDMModelWrapper,
    guidance: Optional[Union[PoissonGuidance, L2Guidance]] = None,
    num_steps: int = 18,
    guidance_scale: float = 1.0,
    guidance_type: str = "pg",
    **config_overrides,
) -> EDMNativeSampler:
    """
    Create EDM native sampler with guidance support.

    Args:
        model: EDM model wrapper
        guidance: Physics guidance (PG or L2, optional)
        num_steps: Number of sampling steps
        guidance_scale: Guidance strength
        guidance_type: Type of guidance ('pg' or 'l2')
        **config_overrides: Additional configuration

    Returns:
        Configured EDMNativeSampler
    """
    config = EDMNativeSamplingConfig(
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        guidance_type=guidance_type,
        **config_overrides,
    )

    return EDMNativeSampler(model=model, guidance=guidance, config=config)


def create_fast_native_sampler(
    model: EDMModelWrapper,
    guidance: Optional[Union[PoissonGuidance, L2Guidance]] = None,
    guidance_type: str = "pg",
    **config_overrides,
) -> EDMNativeSampler:
    """Create fast native sampler configuration."""
    config = EDMNativeSamplingConfig(
        num_steps=10,
        S_churn=0.0,
        guidance_scale=0.8,
        guidance_type=guidance_type,
        **config_overrides,
    )

    return EDMNativeSampler(model=model, guidance=guidance, config=config)


def create_high_quality_native_sampler(
    model: EDMModelWrapper,
    guidance: Optional[Union[PoissonGuidance, L2Guidance]] = None,
    guidance_type: str = "pg",
    **config_overrides,
) -> EDMNativeSampler:
    """Create high-quality native sampler configuration."""
    config = EDMNativeSamplingConfig(
        num_steps=50,
        S_churn=0.0,
        guidance_scale=1.2,
        guidance_type=guidance_type,
        **config_overrides,
    )

    return EDMNativeSampler(model=model, guidance=guidance, config=config)
