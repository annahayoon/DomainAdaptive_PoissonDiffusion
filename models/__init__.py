"""
Models module for Domain-Adaptive Poisson-Gaussian Diffusion.

This module contains:
- EDM model wrapper with conditioning
- Domain encoding utilities
- Sampling algorithms
- Model factory functions
"""

from .edm_native_sampler import EDMNativeSampler, EDMNativeSamplingConfig
from .edm_wrapper import DomainEncoder, EDMModelWrapper, FiLMLayer
from .sampler import EDMPosteriorSampler, SamplingConfig

__all__ = [
    "EDMModelWrapper",
    "DomainEncoder",
    "FiLMLayer",
    "EDMPosteriorSampler",
    "SamplingConfig",
    "EDMNativeSampler",
    "EDMNativeSamplingConfig",
]
