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

from config.logging_config import get_logger

from .error_handlers import MetadataError, NumericalStabilityManager, TransformError

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

    bit_depth: int = 8

    # Optional acquisition parameters
    iso: Optional[int] = None
    exposure_time: Optional[float] = None

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


class AdaptiveResolutionManager:
    """
    Manages adaptive resolution selection for optimal quality-efficiency tradeoff.

    This class analyzes input images and selects the optimal resolution for processing
    based on image characteristics, computational constraints, and quality requirements.
    """

    def __init__(
        self,
        min_resolution: int = 32,
        max_resolution: int = 128,
        quality_estimator_model: str = "simple",
    ):
        """
        Initialize adaptive resolution manager.

        Args:
            min_resolution: Minimum processing resolution
            max_resolution: Maximum processing resolution
            quality_estimator_model: Model for quality estimation ('simple', 'cnn')
        """
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.quality_estimator_model = quality_estimator_model

        # Quality estimation parameters
        self.noise_sensitivity = 0.3  # How much noise affects resolution choice
        self.detail_sensitivity = 0.4  # How much detail affects resolution choice
        self.size_sensitivity = 0.3  # How much input size affects resolution choice

        logger.info(
            f"AdaptiveResolutionManager initialized: {min_resolution} → {max_resolution}"
        )

    def analyze_image_characteristics(self, image: torch.Tensor) -> Dict[str, float]:
        """
        Analyze image characteristics for resolution selection.

        Args:
            image: Input image tensor [B, C, H, W]

        Returns:
            Dictionary with characteristic scores
        """
        # Estimate noise level (simple variance-based method)
        noise_level = self._estimate_noise_level(image)

        # Estimate detail level (edge density)
        detail_level = self._estimate_detail_level(image)

        # Get input size factor
        input_size = max(image.shape[-2:])

        return {
            "noise_level": noise_level,
            "detail_level": detail_level,
            "input_size": input_size,
            "noise_score": min(1.0, noise_level * 2.0),  # Normalize to [0,1]
            "detail_score": detail_level,
            "size_score": min(1.0, input_size / self.max_resolution),
        }

    def _estimate_noise_level(self, image: torch.Tensor) -> float:
        """Estimate noise level in image."""
        # Simple noise estimation based on high-frequency variance
        if image.dim() == 4:
            # Use mean across batch and channels for simplicity
            img = image.mean(dim=(0, 1), keepdim=True)
        else:
            img = image.unsqueeze(0)

        # Compute local variance as noise proxy
        kernel_size = 3
        padding = kernel_size // 2

        # Mean filter
        mean_filt = torch.nn.functional.avg_pool2d(
            img, kernel_size, stride=1, padding=padding
        )

        # Variance filter
        var_filt = (
            torch.nn.functional.avg_pool2d(
                img**2, kernel_size, stride=1, padding=padding
            )
            - mean_filt**2
        )

        noise_estimate = var_filt.mean().item()
        return min(1.0, noise_estimate * 10.0)  # Scale and clamp

    def _estimate_detail_level(self, image: torch.Tensor) -> float:
        """Estimate detail level in image."""
        # Simple edge detection based approach
        if image.dim() == 4:
            img = image.mean(dim=1, keepdim=True)  # Convert to grayscale
        else:
            img = image.unsqueeze(1)

        # Sobel-like edge detection (simple gradient)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=image.device,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=image.device,
        )

        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        edges_x = torch.nn.functional.conv2d(img, sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(img, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)

        # Detail score is average edge magnitude
        detail_score = edges.mean().item()
        return min(1.0, detail_score * 2.0)  # Scale and clamp

    def select_optimal_resolution(
        self, image: torch.Tensor, constraints: Dict[str, Union[float, str]] = None
    ) -> Tuple[int, Dict[str, float]]:
        """
        Select optimal resolution for processing.

        Args:
            image: Input image tensor
            constraints: Processing constraints

        Returns:
            (optimal_resolution, analysis_results)
        """
        if constraints is None:
            constraints = {}

        # Default constraints
        max_memory_gb = constraints.get("max_memory_gb", 8.0)
        target_time_sec = constraints.get("target_time_sec", 30.0)
        quality_preference = constraints.get("quality_preference", "balanced")

        # Analyze image characteristics
        characteristics = self.analyze_image_characteristics(image)

        # Calculate resolution score based on characteristics
        resolution_score = (
            self.noise_sensitivity * characteristics["noise_score"]
            + self.detail_sensitivity * characteristics["detail_score"]
            + self.size_sensitivity * characteristics["size_score"]
        )

        # Adjust based on quality preference
        if quality_preference == "quality":
            resolution_score = min(1.0, resolution_score * 1.2)
        elif quality_preference == "speed":
            resolution_score = max(0.0, resolution_score * 0.8)

        # Map score to resolution
        # Higher score → higher resolution needed
        if resolution_score < 0.2:
            target_resolution = self.min_resolution
        elif resolution_score < 0.4:
            target_resolution = self.min_resolution * 2
        elif resolution_score < 0.6:
            target_resolution = self.min_resolution * 4
        elif resolution_score < 0.8:
            target_resolution = self.min_resolution * 8
        else:
            target_resolution = self.max_resolution

        # Apply memory constraint
        # Rough memory estimation: memory_gb ≈ (resolution/512)^2 * 8
        estimated_memory = (
            target_resolution / self.max_resolution
        ) ** 2 * max_memory_gb
        if estimated_memory > max_memory_gb:
            # Scale down resolution to fit memory
            scale_factor = (max_memory_gb / estimated_memory) ** 0.5
            target_resolution = max(
                self.min_resolution, int(target_resolution * scale_factor)
            )

        # Ensure resolution is power of 2
        target_resolution = self._closest_power_of_two(target_resolution)

        # Clamp to valid range
        target_resolution = max(
            self.min_resolution, min(self.max_resolution, target_resolution)
        )

        return target_resolution, characteristics

    def _closest_power_of_two(self, n: int) -> int:
        """Find closest power of 2 to n."""
        if n <= 0:
            return self.min_resolution

        # Find closest lower power of 2
        lower = 1 << (n.bit_length() - 1)

        # Check if n is closer to lower or upper
        upper = lower << 1

        if n - lower < upper - n:
            return max(self.min_resolution, lower)
        else:
            return min(self.max_resolution, upper)

    def get_batch_size_for_resolution(
        self, resolution: int, base_batch_size: int = 4
    ) -> int:
        """
        Calculate optimal batch size for given resolution.

        Args:
            resolution: Target resolution
            base_batch_size: Base batch size for max resolution

        Returns:
            Optimal batch size for the resolution
        """
        # Memory scales with resolution squared
        memory_scale = (resolution / self.max_resolution) ** 2

        # Adjust batch size inversely with memory scale
        if memory_scale <= 1.0:
            return int(base_batch_size / memory_scale)
        else:
            return max(1, int(base_batch_size / memory_scale))

    def estimate_processing_time(
        self, resolution: int, base_time: float = 10.0  # Base time for max resolution
    ) -> float:
        """
        Estimate processing time for given resolution.

        Args:
            resolution: Processing resolution
            base_time: Base processing time for max resolution

        Returns:
            Estimated processing time in seconds
        """
        # Time scales roughly with resolution squared
        time_scale = (resolution / self.max_resolution) ** 2
        return base_time * time_scale
