"""
Data preprocessing pipeline for cross-domain diffusion model.
Handles photography, microscopy, and astronomy domains.
"""

from .astronomy_processor import AstronomyProcessor
from .microscopy_processor import MicroscopyProcessor
from .photography_processor import PhotographyProcessor
from .preprocessing_utils import (
    compute_global_scale,
    create_saturation_mask,
    create_valid_mask,
    estimate_background_level,
    estimate_noise_params_photon_transfer,
    extract_tiles_with_augmentation,
    pack_bayer_to_channels,
    save_preprocessed_scene,
    save_preprocessed_tile,
    split_scenes_by_ratio,
)

__all__ = [
    # Core utilities
    "estimate_noise_params_photon_transfer",
    "compute_global_scale",
    "extract_tiles_with_augmentation",
    "pack_bayer_to_channels",
    "estimate_background_level",
    "split_scenes_by_ratio",
    "create_saturation_mask",
    "create_valid_mask",
    "save_preprocessed_tile",
    "save_preprocessed_scene",
    # Domain processors
    "PhotographyProcessor",
    "MicroscopyProcessor",
    "AstronomyProcessor",
]
