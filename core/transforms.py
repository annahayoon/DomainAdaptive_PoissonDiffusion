"""
Reversible image transforms with complete metadata tracking.
Critical for proper reconstruction across different scales.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .error_handlers import NumericalStabilityManager
from .exceptions import MetadataError, TransformError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ImageMetadata:
    """Complete metadata for perfect reconstruction."""

    # Original dimensions
    original_height: int
    original_width: int

    # Scaling information
    scale_factor: float
    crop_bbox: Optional[Tuple[int, int, int, int]] = None  # (top, left, height, width)
    pad_amounts: Optional[
        Tuple[int, int, int, int]
    ] = None  # (top, bottom, left, right)

    # Physical calibration
    pixel_size: float = 1.0  # Physical size per pixel
    pixel_unit: str = "pixel"  # 'um', 'arcsec', or 'pixel'

    # Sensor calibration (for reconstruction)
    black_level: float = 0.0
    white_level: float = 1.0

    # Domain and acquisition
    domain: str = "unknown"  # 'photography', 'microscopy', 'astronomy'
    bit_depth: int = 8

    # Optional acquisition parameters
    iso: Optional[int] = None
    exposure_time: Optional[float] = None
    wavelength: Optional[float] = None  # nm for microscopy
    telescope: Optional[str] = None  # for astronomy

    def to_json(self) -> str:
        """Serialize to JSON."""
        try:
            return json.dumps(asdict(self), indent=2)
        except Exception as e:
            logger.error(f"Failed to serialize metadata to JSON: {e}")
            raise MetadataError(f"Metadata serialization failed: {e}")

    @classmethod
    def from_json(cls, json_str: str) -> "ImageMetadata":
        """Deserialize from JSON."""
        try:
            data = json.loads(json_str)

            # Convert lists back to tuples for bbox and pad_amounts
            if "crop_bbox" in data and data["crop_bbox"] is not None:
                data["crop_bbox"] = tuple(data["crop_bbox"])
            if "pad_amounts" in data and data["pad_amounts"] is not None:
                data["pad_amounts"] = tuple(data["pad_amounts"])

            return cls(**data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise MetadataError(f"Invalid JSON format: {e}")
        except TypeError as e:
            logger.error(f"Invalid metadata structure: {e}")
            raise MetadataError(f"Invalid metadata structure: {e}")

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save metadata to JSON file."""
        filepath = Path(filepath)
        try:
            with open(filepath, "w") as f:
                f.write(self.to_json())
            logger.debug(f"Metadata saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {filepath}: {e}")
            raise MetadataError(f"Failed to save metadata: {e}")

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "ImageMetadata":
        """Load metadata from JSON file."""
        filepath = Path(filepath)
        try:
            with open(filepath, "r") as f:
                json_str = f.read()
            logger.debug(f"Metadata loaded from {filepath}")
            return cls.from_json(json_str)
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {filepath}")
            raise MetadataError(f"Metadata file not found: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load metadata from {filepath}: {e}")
            raise MetadataError(f"Failed to load metadata: {e}")

    def validate(self) -> None:
        """Validate metadata consistency."""
        errors = []

        # Check dimensions
        if self.original_height <= 0 or self.original_width <= 0:
            errors.append("Original dimensions must be positive")

        # Check scale factor
        if self.scale_factor <= 0:
            errors.append("Scale factor must be positive")

        # Check crop bbox if present
        if self.crop_bbox is not None:
            top, left, height, width = self.crop_bbox
            if any(x < 0 for x in [top, left, height, width]):
                errors.append("Crop bbox values must be non-negative")
            if height <= 0 or width <= 0:
                errors.append("Crop dimensions must be positive")

        # Check pad amounts if present
        if self.pad_amounts is not None:
            if any(x < 0 for x in self.pad_amounts):
                errors.append("Pad amounts must be non-negative")

        # Check physical parameters
        if self.pixel_size <= 0:
            errors.append("Pixel size must be positive")

        if self.pixel_unit not in ["um", "arcsec", "pixel"]:
            errors.append("Pixel unit must be 'um', 'arcsec', or 'pixel'")

        # Check sensor parameters
        if self.black_level < 0:
            errors.append("Black level must be non-negative")

        if self.white_level <= self.black_level:
            errors.append("White level must be greater than black level")

        # Check bit depth
        if self.bit_depth <= 0 or self.bit_depth > 32:
            errors.append("Bit depth must be between 1 and 32")

        # Check domain
        if self.domain not in [
            "photography",
            "microscopy",
            "astronomy",
            "unknown",
            "test",
        ]:
            errors.append(
                "Domain must be 'photography', 'microscopy', 'astronomy', 'unknown', or 'test'"
            )

        if errors:
            error_msg = "Metadata validation failed: " + "; ".join(errors)
            logger.error(error_msg)
            raise MetadataError(error_msg)


class ReversibleTransform:
    """
    Transform images to model size while preserving all information
    needed for perfect reconstruction.
    """

    def __init__(self, target_size: int = 128, mode: str = "bilinear"):
        """
        Args:
            target_size: Size for model input (square)
            mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        """
        if target_size <= 0:
            raise TransformError("Target size must be positive")

        if mode not in ["bilinear", "nearest", "bicubic"]:
            raise TransformError("Mode must be 'bilinear', 'nearest', or 'bicubic'")

        self.target_size = target_size
        self.mode = mode
        self.stability_manager = NumericalStabilityManager(
            range_min=None,  # Don't enforce range limits for transforms
            range_max=None,
            eps_variance=1e-8,
            grad_clip=1e6,  # Very high to avoid clipping in transforms
        )

        logger.info(
            f"ReversibleTransform initialized: target_size={target_size}, mode={mode}"
        )

    def forward(
        self,
        image: torch.Tensor,
        pixel_size: float = 1.0,
        pixel_unit: str = "pixel",
        domain: str = "unknown",
        black_level: float = 0.0,
        white_level: float = 1.0,
        **extra_metadata,
    ) -> Tuple[torch.Tensor, ImageMetadata]:
        """
        Transform image to model size, preserving all metadata.

        Args:
            image: Input image [B, C, H, W]
            pixel_size: Physical size per pixel
            pixel_unit: Unit of pixel_size
            domain: Image domain
            black_level: Sensor black level
            white_level: Sensor white level
            **extra_metadata: Additional metadata to preserve

        Returns:
            (transformed_image, metadata)
        """
        # Validate input
        if not isinstance(image, torch.Tensor):
            raise TransformError("Input must be a torch.Tensor")

        if image.dim() != 4:
            raise TransformError("Input must be 4D tensor [B, C, H, W]")

        # Check for numerical issues
        self.stability_manager.check_and_fix_tensor(
            image, "input image", fix_issues=False
        )

        B, C, H, W = image.shape
        device = image.device
        dtype = image.dtype

        logger.debug(f"Forward transform: input shape {image.shape}")

        # Calculate scale factor
        max_dim = max(H, W)
        scale_factor = self.target_size / max_dim

        # Initialize metadata
        metadata = ImageMetadata(
            original_height=H,
            original_width=W,
            scale_factor=scale_factor,
            crop_bbox=None,
            pad_amounts=None,
            pixel_size=pixel_size,
            pixel_unit=pixel_unit,
            domain=domain,
            black_level=black_level,
            white_level=white_level,
            bit_depth=int(np.log2(max(white_level - black_level, 1)) + 1),
            **extra_metadata,
        )

        # Validate metadata
        metadata.validate()

        # Step 1: Resize if needed
        if scale_factor != 1.0:
            new_H = int(H * scale_factor)
            new_W = int(W * scale_factor)

            # Ensure minimum size of 1
            new_H = max(1, new_H)
            new_W = max(1, new_W)

            logger.debug(f"Resizing from ({H}, {W}) to ({new_H}, {new_W})")

            image = F.interpolate(
                image,
                size=(new_H, new_W),
                mode=self.mode,
                align_corners=False if self.mode != "nearest" else None,
                antialias=True if self.mode == "bilinear" else False,
            )

            # Check for numerical issues after interpolation
            self.stability_manager.check_and_fix_tensor(
                image, "resized image", fix_issues=False
            )
        else:
            new_H, new_W = H, W

        # Step 2: Pad or crop to square
        if new_H != self.target_size or new_W != self.target_size:
            image, crop_bbox, pad_amounts = self._make_square(image, self.target_size)
            metadata.crop_bbox = crop_bbox
            metadata.pad_amounts = pad_amounts

        # Final validation
        self.stability_manager.check_and_fix_tensor(
            image, "transformed image", fix_issues=False
        )

        if image.shape[-2:] != (self.target_size, self.target_size):
            raise TransformError(
                f"Output shape {image.shape} does not match target size {self.target_size}"
            )

        logger.debug(f"Forward transform complete: output shape {image.shape}")

        return image, metadata

    def inverse(self, image: torch.Tensor, metadata: ImageMetadata) -> torch.Tensor:
        """
        Perfectly reverse the transformation.

        Args:
            image: Transformed image [B, C, target_size, target_size]
            metadata: Metadata from forward transform

        Returns:
            Original-size image [B, C, H, W]
        """
        # Validate inputs
        if not isinstance(image, torch.Tensor):
            raise TransformError("Input must be a torch.Tensor")

        if image.dim() != 4:
            raise TransformError("Input must be 4D tensor [B, C, H, W]")

        if not isinstance(metadata, ImageMetadata):
            raise TransformError("Metadata must be ImageMetadata instance")

        # Validate metadata
        metadata.validate()

        # Check for numerical issues
        self.stability_manager.check_and_fix_tensor(
            image, "input image for inverse", fix_issues=False
        )

        logger.debug(f"Inverse transform: input shape {image.shape}")

        # Step 1: Reverse padding/cropping
        if metadata.crop_bbox is not None:
            # Image was cropped, so pad it back
            top, left, crop_h, crop_w = metadata.crop_bbox

            # Calculate original size after scaling
            scaled_H = int(metadata.original_height * metadata.scale_factor)
            scaled_W = int(metadata.original_width * metadata.scale_factor)

            # Ensure minimum size
            scaled_H = max(1, scaled_H)
            scaled_W = max(1, scaled_W)

            logger.debug(
                f"Reversing crop: placing {image.shape} into ({scaled_H}, {scaled_W})"
            )

            # Create full-size tensor
            B, C = image.shape[:2]
            full = torch.zeros(
                B, C, scaled_H, scaled_W, device=image.device, dtype=image.dtype
            )

            # Validate crop region
            if (top + crop_h > scaled_H) or (left + crop_w > scaled_W):
                raise TransformError("Crop region exceeds scaled image dimensions")

            # Place image in correct position
            full[:, :, top : top + crop_h, left : left + crop_w] = image
            image = full

        elif metadata.pad_amounts is not None:
            # Image was padded, so crop it back
            top, bottom, left, right = metadata.pad_amounts
            H_total = image.shape[2]
            W_total = image.shape[3]

            logger.debug(f"Reversing padding: cropping from {image.shape}")

            # Validate pad amounts
            if top + bottom >= H_total or left + right >= W_total:
                raise TransformError("Pad amounts exceed image dimensions")

            image = image[:, :, top : H_total - bottom, left : W_total - right]

        # Step 2: Reverse scaling
        if metadata.scale_factor != 1.0:
            logger.debug(
                f"Reversing scale: from {image.shape} to ({metadata.original_height}, {metadata.original_width})"
            )

            image = F.interpolate(
                image,
                size=(metadata.original_height, metadata.original_width),
                mode=self.mode,
                align_corners=False if self.mode != "nearest" else None,
                antialias=True if self.mode == "bilinear" else False,
            )

        # Final validation
        self.stability_manager.check_and_fix_tensor(
            image, "reconstructed image", fix_issues=False
        )

        expected_shape = (
            image.shape[0],
            image.shape[1],
            metadata.original_height,
            metadata.original_width,
        )
        if image.shape != expected_shape:
            raise TransformError(
                f"Reconstructed shape {image.shape} does not match expected {expected_shape}"
            )

        logger.debug(f"Inverse transform complete: output shape {image.shape}")

        return image

    def _make_square(
        self, image: torch.Tensor, target_size: int
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[Tuple]]:
        """
        Make image square by center cropping or padding.

        Returns:
            (squared_image, crop_bbox, pad_amounts)
        """
        B, C, H, W = image.shape

        if H == target_size and W == target_size:
            return image, None, None

        logger.debug(f"Making square: {(H, W)} -> {target_size}")

        if H > target_size or W > target_size:
            # Need to crop
            crop_h = min(H, target_size)
            crop_w = min(W, target_size)
            top = (H - crop_h) // 2
            left = (W - crop_w) // 2

            logger.debug(f"Cropping: region ({top}, {left}, {crop_h}, {crop_w})")

            image = image[:, :, top : top + crop_h, left : left + crop_w]
            crop_bbox = (top, left, crop_h, crop_w)

            # After cropping, might still need padding
            if crop_h < target_size or crop_w < target_size:
                pad_h = target_size - crop_h
                pad_w = target_size - crop_w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left

                logger.debug(
                    f"Padding after crop: ({pad_top}, {pad_bottom}, {pad_left}, {pad_right})"
                )

                # Use replication padding if reflection would exceed dimensions
                H_curr, W_curr = image.shape[-2:]
                if (
                    pad_top >= H_curr
                    or pad_bottom >= H_curr
                    or pad_left >= W_curr
                    or pad_right >= W_curr
                ):
                    # Use replication padding for extreme cases
                    image = F.pad(
                        image,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="replicate",
                    )
                else:
                    # Use reflection padding for normal cases
                    image = F.pad(
                        image,
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="reflect",
                    )

                return image, crop_bbox, (pad_top, pad_bottom, pad_left, pad_right)

            return image, crop_bbox, None

        else:
            # Only padding needed
            pad_h = target_size - H
            pad_w = target_size - W
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            logger.debug(f"Padding: ({pad_top}, {pad_bottom}, {pad_left}, {pad_right})")

            # Use replication padding if reflection would exceed dimensions
            H_curr, W_curr = image.shape[-2:]
            if (
                pad_top >= H_curr
                or pad_bottom >= H_curr
                or pad_left >= W_curr
                or pad_right >= W_curr
            ):
                # Use replication padding for extreme cases
                image = F.pad(
                    image, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
                )
            else:
                # Use reflection padding for normal cases
                image = F.pad(
                    image, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
                )

            return image, None, (pad_top, pad_bottom, pad_left, pad_right)

    def test_reconstruction_error(
        self, image: torch.Tensor, **metadata_kwargs
    ) -> float:
        """
        Test reconstruction error for validation.

        Args:
            image: Test image
            **metadata_kwargs: Metadata for forward transform

        Returns:
            Maximum absolute reconstruction error
        """
        # Forward transform
        transformed, metadata = self.forward(image, **metadata_kwargs)

        # Inverse transform
        reconstructed = self.inverse(transformed, metadata)

        # Compute error
        error = (image - reconstructed).abs().max().item()

        logger.debug(f"Reconstruction error: {error}")

        return error
