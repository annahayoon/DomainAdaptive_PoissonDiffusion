"""
Data module for Domain-Adaptive Poisson-Gaussian Diffusion.

This module contains:
- Multi-format data loaders (RAW, TIFF, FITS)
- Domain-specific datasets
- Data preprocessing utilities
- Calibration management
"""

from .domain_datasets import DomainDataset, MultiDomainDataset
from .loaders import AstronomyLoader, MicroscopyLoader, PhotographyLoader

__all__ = [
    "DomainDataset",
    "MultiDomainDataset",
    "PhotographyLoader",
    "MicroscopyLoader",
    "AstronomyLoader",
]
