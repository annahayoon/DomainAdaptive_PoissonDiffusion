"""
Initialization strategies for guided diffusion sampling.

This module provides different ways to initialize the latent variable
for the reverse diffusion process.

"""

import numpy as np
import torch


def denormalize(x_norm, max_value):
    """Convert from model space [-1, 1] to physical units [0, max_value]."""
    return (x_norm + 1.0) * 0.5 * max_value


def normalize(x_phys, max_value):
    """Convert from physical units [0, max_value] to model space [-1, 1]."""
    return (x_phys / max_value) * 2.0 - 1.0


class InitializationStrategy:
    """Base class for initialization strategies."""

    def __init__(self, sigma_max=80.0):
        """
        Args:
            sigma_max: Maximum noise level in diffusion schedule
        """
        self.sigma_max = sigma_max

    def initialize(self, y_obs, guidance=None):
        """
        Initialize latent for diffusion sampling.

        Args:
            y_obs: Observed noisy image [B, C, H, W] in [-1, 1]
            guidance: Optional guidance module with noise parameters

        Returns:
            x_init: Initial latent [B, C, H, W] for sampling
        """
        raise NotImplementedError


class PurePriorInit(InitializationStrategy):
    """
    Initialize from pure Gaussian noise (standard posterior sampling).

    This is the theoretically correct approach for posterior sampling
    in guided diffusion methods (DPS, DiffPIR, etc.).

    x_T ~ N(0, σ_max² I)

    Pros:
    - Theoretically sound (true posterior sampling)
    - Standard in diffusion literature

    Cons:
    - Doesn't use observation structure
    - May converge slower
    """

    def initialize(self, y_obs, guidance=None):
        print(f" [Init] Pure prior: N(0, {self.sigma_max}²)")

        # Pure Gaussian noise at sigma_max level
        # Note: In EDM, images are normalized to have std ≈ 1
        # So sigma_max is in units of normalized intensity
        x_init = torch.randn_like(y_obs) * self.sigma_max

        return x_init


class ObservationInit(InitializationStrategy):
    """
    Initialize directly from noisy observation.

    This is NOT standard posterior sampling, but can work in practice
    if guidance is strong enough to correct the trajectory.

    x_T = y_obs

    Pros:
    - Provides strong structural initialization
    - Fast convergence (starts close to solution)

    Cons:
    - Not theoretically justified as posterior sampling
    - Noise level mismatch with σ_max
    """

    def initialize(self, y_obs, guidance=None):
        print(f" [Init] Observation-based (σ_obs ≈ variable)")
        return y_obs.clone()


class NoiseMatchingInit(InitializationStrategy):
    """
    Add calibrated noise to observation to match sigma_max.

    This is a hybrid approach: use observation structure but ensure
    correct noise level for the diffusion model.

    x_T = y_obs + N(0, (σ_max² - σ_obs²))

    Pros:
    - Combines observation structure with correct noise level
    - Theoretically more justified than pure observation

    Cons:
    - More complex
    - Requires estimating observation noise level
    """

    def initialize(self, y_obs, guidance):
        if guidance is None:
            raise ValueError("NoiseMatchingInit requires guidance module")

        # Convert observation to physical units
        y_phys = denormalize(y_obs, guidance.max_adu)

        # Estimate observation noise level (spatially varying)
        var_obs = guidance.gain * y_phys + guidance.sigma_r**2
        sigma_obs = torch.sqrt(var_obs.mean()).item()

        print(
            f" [Init] Noise-matching: σ_obs={sigma_obs:.1f} → σ_max={self.sigma_max:.1f}"
        )

        if sigma_obs >= self.sigma_max:
            print(f" [Init] Observation already noisy enough (σ_obs ≥ σ_max)")
            return y_obs

        # Compute additional noise needed (in physical units)
        sigma_add_phys = np.sqrt(self.sigma_max**2 - sigma_obs**2)

        # Convert to normalized space
        # In normalized space, noise scales with max_adu
        sigma_add_norm = sigma_add_phys / guidance.max_adu

        # Add isotropic Gaussian noise
        noise = torch.randn_like(y_obs) * sigma_add_norm
        x_init = y_obs + noise

        print(f" [Init] Added noise with σ={sigma_add_phys:.1f}")

        return x_init


def create_initializer(strategy="pure_prior", sigma_max=80.0):
    """
    Factory function to create initialization strategy.

    Args:
        strategy: One of ['pure_prior', 'observation', 'noise_matching']
        sigma_max: Maximum noise level in diffusion schedule

    Returns:
        InitializationStrategy instance

    Example:
    >>> initializer = create_initializer('pure_prior', sigma_max=80.0)
    >>> x_init = initializer.initialize(y_obs, guidance)
    """
    strategies = {
        "pure_prior": PurePriorInit,
        "observation": ObservationInit,
        "noise_matching": NoiseMatchingInit,
    }

    if strategy not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(
            f"Unknown strategy: '{strategy}'. " f"Choose from: {available}"
        )

    return strategies[strategy](sigma_max=sigma_max)
