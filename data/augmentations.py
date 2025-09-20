"""
Geometric augmentation pipeline for domain datasets.

This module implements physics-aware augmentations that preserve the
statistical properties of Poisson-Gaussian noise while providing
geometric diversity for training.

Key features:
- Rotation, flipping, and cropping augmentations
- Noise-preserving transformations
- Deterministic seeding for reproducibility
- Integration with reversible transforms

Requirements addressed: 2.4, 5.6, 5.7 from requirements.md
Task: 4.2 augmentation component from tasks.md
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for geometric augmentations."""

    # Rotation
    enable_rotation: bool = True
    rotation_angles: List[int] = None  # Will be set in __post_init__

    # Flipping
    enable_horizontal_flip: bool = True
    enable_vertical_flip: bool = True
    flip_probability: float = 0.5

    # Cropping (for training diversity)
    enable_random_crop: bool = False
    crop_scale_range: Tuple[float, float] = (0.8, 1.0)

    # Brightness/contrast (careful with physics)
    enable_brightness: bool = False
    brightness_range: Tuple[float, float] = (0.9, 1.1)

    # Noise augmentation (add synthetic noise)
    enable_noise_augmentation: bool = False
    noise_scale_range: Tuple[float, float] = (0.5, 1.5)

    # Reproducibility
    deterministic: bool = True
    seed: int = 42

    def __post_init__(self):
        """Set default rotation angles."""
        if self.rotation_angles is None:
            self.rotation_angles = [0, 90, 180, 270]


class BaseAugmentation(ABC):
    """Base class for augmentation operations."""

    def __init__(self, config: AugmentationConfig):
        """Initialize augmentation with configuration."""
        self.config = config

        if config.deterministic:
            self.rng = np.random.RandomState(config.seed)
        else:
            self.rng = np.random.RandomState()

    @abstractmethod
    def apply(
        self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Apply augmentation to image and mask.

        Args:
            image: Input image tensor [C, H, W]
            mask: Optional mask tensor [C, H, W]

        Returns:
            (augmented_image, augmented_mask, transform_info)
        """
        pass

    def should_apply(self, probability: float = 1.0) -> bool:
        """Determine if augmentation should be applied."""
        return self.rng.random() < probability


class RotationAugmentation(BaseAugmentation):
    """Rotation augmentation with 90-degree increments."""

    def apply(
        self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Apply rotation augmentation."""
        if not self.config.enable_rotation:
            return image, mask, {"rotation_angle": 0}

        # Choose random rotation angle
        angle = self.rng.choice(self.config.rotation_angles)

        if angle == 0:
            return image, mask, {"rotation_angle": 0}

        # Apply rotation
        # Convert angle to number of 90-degree rotations
        k = angle // 90

        # Rotate image
        rotated_image = torch.rot90(image, k, dims=[-2, -1])

        # Rotate mask if provided
        rotated_mask = None
        if mask is not None:
            rotated_mask = torch.rot90(mask, k, dims=[-2, -1])

        return rotated_image, rotated_mask, {"rotation_angle": angle}


class FlipAugmentation(BaseAugmentation):
    """Horizontal and vertical flipping augmentation."""

    def apply(
        self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Apply flipping augmentation."""
        flipped_image = image
        flipped_mask = mask
        transform_info = {"horizontal_flip": False, "vertical_flip": False}

        # Horizontal flip
        if self.config.enable_horizontal_flip and self.should_apply(
            self.config.flip_probability
        ):
            flipped_image = torch.flip(flipped_image, dims=[-1])
            if flipped_mask is not None:
                flipped_mask = torch.flip(flipped_mask, dims=[-1])
            transform_info["horizontal_flip"] = True

        # Vertical flip
        if self.config.enable_vertical_flip and self.should_apply(
            self.config.flip_probability
        ):
            flipped_image = torch.flip(flipped_image, dims=[-2])
            if flipped_mask is not None:
                flipped_mask = torch.flip(flipped_mask, dims=[-2])
            transform_info["vertical_flip"] = True

        return flipped_image, flipped_mask, transform_info


class CropAugmentation(BaseAugmentation):
    """Random cropping augmentation for training diversity."""

    def apply(
        self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Apply random crop augmentation."""
        if not self.config.enable_random_crop:
            return image, mask, {"crop_applied": False}

        # Handle both 2D and 3D tensors
        if image.ndim == 2:
            H, W = image.shape
            C = 1
        else:
            C, H, W = image.shape

        # Choose random scale
        scale = self.rng.uniform(*self.config.crop_scale_range)

        # Calculate crop size
        crop_h = int(H * scale)
        crop_w = int(W * scale)

        # Choose random crop position
        top = self.rng.randint(0, H - crop_h + 1)
        left = self.rng.randint(0, W - crop_w + 1)

        # Apply crop
        if image.ndim == 2:
            cropped_image = image[top : top + crop_h, left : left + crop_w]
        else:
            cropped_image = image[:, top : top + crop_h, left : left + crop_w]

        cropped_mask = None
        if mask is not None:
            if mask.ndim == 2:
                cropped_mask = mask[top : top + crop_h, left : left + crop_w]
            else:
                cropped_mask = mask[:, top : top + crop_h, left : left + crop_w]

        # Resize back to original size
        if cropped_image.ndim == 2:
            cropped_image = (
                F.interpolate(
                    cropped_image.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
        else:
            cropped_image = F.interpolate(
                cropped_image.unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if cropped_mask is not None:
            if cropped_mask.ndim == 2:
                cropped_mask = (
                    F.interpolate(
                        cropped_mask.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
            else:
                cropped_mask = F.interpolate(
                    cropped_mask.unsqueeze(0), size=(H, W), mode="nearest"
                ).squeeze(0)

        transform_info = {
            "crop_applied": True,
            "crop_scale": scale,
            "crop_position": (top, left),
            "crop_size": (crop_h, crop_w),
        }

        return cropped_image, cropped_mask, transform_info


class BrightnessAugmentation(BaseAugmentation):
    """
    Careful brightness augmentation that preserves noise statistics.

    This applies multiplicative scaling which preserves the Poisson-Gaussian
    noise structure (scaling both signal and noise proportionally).
    """

    def apply(
        self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Apply brightness augmentation."""
        if not self.config.enable_brightness:
            return image, mask, {"brightness_factor": 1.0}

        # Choose random brightness factor
        factor = self.rng.uniform(*self.config.brightness_range)

        # Apply multiplicative scaling (preserves noise structure)
        augmented_image = image * factor

        # Clamp to valid range [0, 1] for normalized data
        augmented_image = torch.clamp(augmented_image, 0.0, 1.0)

        return augmented_image, mask, {"brightness_factor": factor}


class NoiseAugmentation(BaseAugmentation):
    """
    Add synthetic Poisson-Gaussian noise for data augmentation.

    This can help with robustness but should be used carefully
    as it changes the noise statistics.
    """

    def apply(
        self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Apply noise augmentation."""
        if not self.config.enable_noise_augmentation:
            return image, mask, {"noise_scale": 1.0}

        # Choose random noise scale
        noise_scale = self.rng.uniform(*self.config.noise_scale_range)

        # Generate Gaussian noise (simplified model)
        noise = torch.randn_like(image) * (noise_scale * 0.01)  # Small noise

        # Add noise
        noisy_image = image + noise

        # Clamp to valid range
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

        return noisy_image, mask, {"noise_scale": noise_scale}


class GeometricAugmentationPipeline:
    """
    Complete geometric augmentation pipeline.

    Applies multiple augmentations in sequence while maintaining
    proper noise statistics and providing comprehensive transform tracking.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmentation pipeline.

        Args:
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()

        # Initialize individual augmentations
        self.augmentations = [
            RotationAugmentation(self.config),
            FlipAugmentation(self.config),
            CropAugmentation(self.config),
            BrightnessAugmentation(self.config),
            NoiseAugmentation(self.config),
        ]

        logger.debug(
            f"Initialized augmentation pipeline with {len(self.augmentations)} augmentations"
        )

    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        apply_augmentations: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Apply augmentation pipeline.

        Args:
            image: Input image tensor [C, H, W]
            mask: Optional mask tensor [C, H, W]
            apply_augmentations: Whether to apply augmentations (False for validation)

        Returns:
            (augmented_image, augmented_mask, transform_info)
        """
        if not apply_augmentations:
            return image, mask, {"augmentations_applied": False}

        current_image = image
        current_mask = mask
        all_transform_info = {"augmentations_applied": True}

        # Apply each augmentation in sequence
        for aug in self.augmentations:
            current_image, current_mask, transform_info = aug.apply(
                current_image, current_mask
            )

            # Merge transform info
            all_transform_info.update(transform_info)

        return current_image, current_mask, all_transform_info

    def get_config(self) -> AugmentationConfig:
        """Get augmentation configuration."""
        return self.config

    def set_deterministic(self, deterministic: bool, seed: Optional[int] = None):
        """Set deterministic mode for all augmentations."""
        if seed is not None:
            self.config.seed = seed

        self.config.deterministic = deterministic

        # Update all augmentations
        for aug in self.augmentations:
            aug.config.deterministic = deterministic
            if deterministic:
                aug.rng = np.random.RandomState(self.config.seed)
            else:
                aug.rng = np.random.RandomState()


def create_augmentation_pipeline(
    domain: str, training: bool = True, **config_overrides
) -> GeometricAugmentationPipeline:
    """
    Create domain-specific augmentation pipeline.

    Args:
        domain: Domain name ('photography', 'microscopy', 'astronomy')
        training: Whether this is for training (enables augmentations)
        **config_overrides: Configuration overrides

    Returns:
        Configured augmentation pipeline
    """
    # Domain-specific defaults
    if domain == "photography":
        config = AugmentationConfig(
            enable_rotation=True,
            enable_horizontal_flip=True,
            enable_vertical_flip=True,
            enable_random_crop=training,
            enable_brightness=training,
            brightness_range=(0.95, 1.05),  # Conservative for photography
            enable_noise_augmentation=False,  # Photography has complex noise
        )

    elif domain == "microscopy":
        config = AugmentationConfig(
            enable_rotation=True,
            enable_horizontal_flip=True,
            enable_vertical_flip=True,
            enable_random_crop=training,
            enable_brightness=training,
            brightness_range=(0.9, 1.1),
            enable_noise_augmentation=training,
            noise_scale_range=(0.8, 1.2),
        )

    elif domain == "astronomy":
        config = AugmentationConfig(
            enable_rotation=True,
            enable_horizontal_flip=True,
            enable_vertical_flip=True,
            enable_random_crop=False,  # Preserve field of view
            enable_brightness=training,
            brightness_range=(0.95, 1.05),  # Conservative for photometry
            enable_noise_augmentation=False,  # Astronomy has well-characterized noise
        )

    else:
        # Default configuration
        config = AugmentationConfig()

    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return GeometricAugmentationPipeline(config)


# Convenience functions for common augmentation setups
def create_training_augmentations(
    domain: str, **kwargs
) -> GeometricAugmentationPipeline:
    """Create augmentation pipeline for training."""
    return create_augmentation_pipeline(domain, training=True, **kwargs)


def create_validation_augmentations(
    domain: str, **kwargs
) -> GeometricAugmentationPipeline:
    """Create augmentation pipeline for validation (minimal augmentations)."""
    return create_augmentation_pipeline(
        domain,
        training=False,
        enable_rotation=False,
        enable_horizontal_flip=False,
        enable_vertical_flip=False,
        enable_random_crop=False,
        enable_brightness=False,
        enable_noise_augmentation=False,
        **kwargs,
    )
