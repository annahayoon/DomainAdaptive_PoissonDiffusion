#!/usr/bin/env python3
"""
Factory system for creating different guidance computers.

This module provides a unified interface for creating Poisson-Gaussian
and L2 guidance computers, enabling easy switching between methods.
"""

from typing import Literal, Optional

from .guidance_config import GuidanceConfig
from .interfaces import GuidanceComputer
from .l2_guidance import L2Guidance
from .poisson_guidance import PoissonGuidance

GuidanceType = Literal["poisson", "l2"]


def create_guidance(
    guidance_type: GuidanceType,
    scale: float,
    background: float = 0.0,
    read_noise: float = 0.0,
    config: Optional[GuidanceConfig] = None,
) -> GuidanceComputer:
    """
    Factory function to create guidance computers.

    Args:
        guidance_type: Type of guidance ("poisson" or "l2")
        scale: Dataset normalization scale (electrons)
        background: Background offset (electrons)
        read_noise: Read noise standard deviation (electrons)
        config: Guidance configuration

    Returns:
        Configured guidance computer
    """
    if guidance_type == "poisson":
        return PoissonGuidance(
            scale=scale, background=background, read_noise=read_noise, config=config
        )
    elif guidance_type == "l2":
        # For L2, use read_noise as uniform noise standard deviation
        noise_variance = read_noise**2 if read_noise > 0 else 1.0
        return L2Guidance(
            scale=scale,
            background=background,
            noise_variance=noise_variance,
            config=config,
        )
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")


def create_guidance_from_config(config_dict: dict) -> GuidanceComputer:
    """Create guidance from configuration dictionary."""
    guidance_config = GuidanceConfig(**config_dict.get("guidance", {}))

    return create_guidance(
        guidance_type=config_dict["guidance"]["type"],
        scale=config_dict["data"]["scale"],
        background=config_dict["data"].get("background", 0.0),
        read_noise=config_dict["data"].get("read_noise", 0.0),
        config=guidance_config,
    )
