"""
Data module for Domain-Adaptive Poisson-Gaussian Diffusion.

This module contains:
- PNG dataset for training
- Data preprocessing utilities
- Calibration management
"""

from .png_dataset import create_edm_png_datasets

__all__ = [
    "create_edm_png_datasets",
]
