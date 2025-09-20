"""
Patch extraction and reconstruction for large image processing.

This module provides comprehensive patch-based processing capabilities for handling
large images that exceed memory limits or model input size constraints.

Key features:
- Memory-efficient patch extraction with overlap handling
- Seamless patch reconstruction with blending
- Adaptive patch sizing based on available memory
- Support for arbitrary image sizes and aspect ratios
- Integration with calibration and transform systems

Requirements addressed: 7.3, 7.6 from requirements.md
Task: 4.3 from tasks.md
"""

import gc
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from core.calibration import SensorCalibration
from core.error_handlers import ErrorHandler, safe_operation
from core.exceptions import DataError, ValidationError
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PatchInfo:
    """Information about an extracted patch."""

    # Patch coordinates in original image
    start_y: int
    start_x: int
    end_y: int
    end_x: int

    # Patch size
    height: int
    width: int

    # Overlap information
    overlap_top: int
    overlap_bottom: int
    overlap_left: int
    overlap_right: int

    # Patch index
    patch_id: int
    row: int
    col: int

    # Blending weights (for reconstruction)
    weight_map: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_y": self.start_y,
            "start_x": self.start_x,
            "end_y": self.end_y,
            "end_x": self.end_x,
            "height": self.height,
            "width": self.width,
            "overlap_top": self.overlap_top,
            "overlap_bottom": self.overlap_bottom,
            "overlap_left": self.overlap_left,
            "overlap_right": self.overlap_right,
            "patch_id": self.patch_id,
            "row": self.row,
            "col": self.col,
        }


class PatchExtractor:
    """
    Efficient patch extraction from large images.

    This class handles the extraction of overlapping patches from large images
    for processing with memory-constrained models.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 512,
        overlap: Union[int, Tuple[int, int]] = 64,
        min_patch_size: Union[int, Tuple[int, int]] = 128,
        padding_mode: str = "reflect",
        device: str = "cpu",
    ):
        """
        Initialize patch extractor.

        Args:
            patch_size: Size of patches to extract (height, width) or single int
            overlap: Overlap between adjacent patches (height, width) or single int
            min_patch_size: Minimum patch size for edge patches
            padding_mode: Padding mode for edge handling ('reflect', 'constant', 'replicate')
            device: Device for tensor operations
        """
        # Handle scalar inputs
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size

        if isinstance(overlap, int):
            self.overlap = (overlap, overlap)
        else:
            self.overlap = overlap

        if isinstance(min_patch_size, int):
            self.min_patch_size = (min_patch_size, min_patch_size)
        else:
            self.min_patch_size = min_patch_size

        self.padding_mode = padding_mode
        self.device = device

        # Validate parameters
        self._validate_parameters()

        logger.info(
            f"Initialized PatchExtractor: patch_size={self.patch_size}, "
            f"overlap={self.overlap}, device={self.device}"
        )

    def _validate_parameters(self):
        """Validate patch extraction parameters."""
        if self.patch_size[0] <= 0 or self.patch_size[1] <= 0:
            raise ValueError("Patch size must be positive")

        if self.overlap[0] < 0 or self.overlap[1] < 0:
            raise ValueError("Overlap must be non-negative")

        if (
            self.overlap[0] >= self.patch_size[0]
            or self.overlap[1] >= self.patch_size[1]
        ):
            raise ValueError("Overlap must be smaller than patch size")

        if self.min_patch_size[0] <= 0 or self.min_patch_size[1] <= 0:
            raise ValueError("Minimum patch size must be positive")

    def calculate_patch_grid(
        self, image_height: int, image_width: int
    ) -> Tuple[List[PatchInfo], int, int]:
        """
        Calculate patch grid for given image dimensions.

        Args:
            image_height: Height of input image
            image_width: Width of input image

        Returns:
            (patch_info_list, num_rows, num_cols) tuple
        """
        patch_h, patch_w = self.patch_size
        overlap_h, overlap_w = self.overlap

        # Calculate step sizes (patch size minus overlap)
        step_h = patch_h - overlap_h
        step_w = patch_w - overlap_w

        # Calculate number of patches needed
        num_rows = max(1, math.ceil((image_height - overlap_h) / step_h))
        num_cols = max(1, math.ceil((image_width - overlap_w) / step_w))

        patch_infos = []
        patch_id = 0

        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate patch coordinates
                start_y = row * step_h
                start_x = col * step_w

                # Ensure we don't exceed image boundaries
                end_y = min(start_y + patch_h, image_height)
                end_x = min(start_x + patch_w, image_width)

                # Adjust start coordinates if patch would be too small
                if end_y - start_y < self.min_patch_size[0]:
                    start_y = max(0, end_y - self.min_patch_size[0])
                if end_x - start_x < self.min_patch_size[1]:
                    start_x = max(0, end_x - self.min_patch_size[1])

                # Calculate actual patch size
                actual_h = end_y - start_y
                actual_w = end_x - start_x

                # Calculate overlaps for this patch
                overlap_top = overlap_h if row > 0 else 0
                overlap_bottom = overlap_h if row < num_rows - 1 else 0
                overlap_left = overlap_w if col > 0 else 0
                overlap_right = overlap_w if col < num_cols - 1 else 0

                # Create patch info
                patch_info = PatchInfo(
                    start_y=start_y,
                    start_x=start_x,
                    end_y=end_y,
                    end_x=end_x,
                    height=actual_h,
                    width=actual_w,
                    overlap_top=overlap_top,
                    overlap_bottom=overlap_bottom,
                    overlap_left=overlap_left,
                    overlap_right=overlap_right,
                    patch_id=patch_id,
                    row=row,
                    col=col,
                )

                patch_infos.append(patch_info)
                patch_id += 1

        logger.debug(
            f"Calculated patch grid: {num_rows}x{num_cols} = {len(patch_infos)} patches "
            f"for image {image_height}x{image_width}"
        )

        return patch_infos, num_rows, num_cols

    @safe_operation("Patch extraction")
    def extract_patches(
        self, image: torch.Tensor, return_info: bool = True
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[PatchInfo]]]:
        """
        Extract patches from input image.

        Args:
            image: Input image tensor [B, C, H, W] or [C, H, W]
            return_info: Whether to return patch information

        Returns:
            List of patch tensors, optionally with patch info list
        """
        # Handle different input dimensions
        if image.ndim == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        elif image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif image.ndim != 4:
            raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {image.ndim}D")

        B, C, H, W = image.shape

        if B > 1:
            logger.warning(f"Batch size {B} > 1, processing first image only")
            image = image[0:1]

        # Calculate patch grid
        patch_infos, num_rows, num_cols = self.calculate_patch_grid(H, W)

        # Extract patches
        patches = []

        for patch_info in patch_infos:
            # Extract patch
            patch = image[
                :,
                :,
                patch_info.start_y : patch_info.end_y,
                patch_info.start_x : patch_info.end_x,
            ]

            # Pad if necessary to reach target patch size
            if (
                patch.shape[2] < self.patch_size[0]
                or patch.shape[3] < self.patch_size[1]
            ):
                pad_h = max(0, self.patch_size[0] - patch.shape[2])
                pad_w = max(0, self.patch_size[1] - patch.shape[3])

                # Ensure padding doesn't exceed input dimensions
                pad_h = min(pad_h, patch.shape[2])
                pad_w = min(pad_w, patch.shape[3])

                # Pad: (left, right, top, bottom)
                padding = (0, pad_w, 0, pad_h)

                # Only pad if padding is reasonable
                if pad_w < patch.shape[3] and pad_h < patch.shape[2]:
                    patch = torch.nn.functional.pad(
                        patch, padding, mode=self.padding_mode
                    )

            patches.append(patch.to(self.device))

        logger.debug(f"Extracted {len(patches)} patches from image {H}x{W}")

        if return_info:
            return patches, patch_infos
        else:
            return patches

    def extract_patch_at_position(
        self,
        image: torch.Tensor,
        start_y: int,
        start_x: int,
        patch_height: Optional[int] = None,
        patch_width: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract single patch at specific position.

        Args:
            image: Input image tensor
            start_y: Starting Y coordinate
            start_x: Starting X coordinate
            patch_height: Patch height (default: self.patch_size[0])
            patch_width: Patch width (default: self.patch_size[1])

        Returns:
            Extracted patch tensor
        """
        if patch_height is None:
            patch_height = self.patch_size[0]
        if patch_width is None:
            patch_width = self.patch_size[1]

        # Handle input dimensions
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)

        B, C, H, W = image.shape

        # Clamp coordinates
        end_y = min(start_y + patch_height, H)
        end_x = min(start_x + patch_width, W)
        start_y = max(0, start_y)
        start_x = max(0, start_x)

        # Extract patch
        patch = image[:, :, start_y:end_y, start_x:end_x]

        # Pad if necessary
        actual_h, actual_w = patch.shape[2], patch.shape[3]
        if actual_h < patch_height or actual_w < patch_width:
            pad_h = max(0, patch_height - actual_h)
            pad_w = max(0, patch_width - actual_w)

            # Ensure padding doesn't exceed input dimensions
            pad_h = min(pad_h, actual_h)
            pad_w = min(pad_w, actual_w)

            padding = (0, pad_w, 0, pad_h)

            # Only pad if padding is reasonable
            if pad_w < actual_w and pad_h < actual_h:
                patch = torch.nn.functional.pad(patch, padding, mode=self.padding_mode)

        return patch.to(self.device)


class PatchReconstructor:
    """
    Seamless reconstruction of images from processed patches.

    This class handles the reconstruction of full images from overlapping patches
    with proper blending to avoid artifacts.
    """

    def __init__(
        self, blending_mode: str = "linear", edge_fade: float = 0.1, device: str = "cpu"
    ):
        """
        Initialize patch reconstructor.

        Args:
            blending_mode: Blending mode ('linear', 'cosine', 'gaussian')
            edge_fade: Fraction of overlap region to use for blending
            device: Device for tensor operations
        """
        self.blending_mode = blending_mode
        self.edge_fade = edge_fade
        self.device = device

        logger.info(
            f"Initialized PatchReconstructor: blending_mode={blending_mode}, "
            f"edge_fade={edge_fade}"
        )

    def create_blending_weights(
        self, patch_info: PatchInfo, patch_height: int, patch_width: int
    ) -> torch.Tensor:
        """
        Create blending weight map for a patch.

        Args:
            patch_info: Information about the patch
            patch_height: Height of the patch
            patch_width: Width of the patch

        Returns:
            Weight map tensor [H, W]
        """
        weights = torch.ones(patch_height, patch_width, device=self.device)

        # Calculate fade regions
        fade_top = int(patch_info.overlap_top * self.edge_fade)
        fade_bottom = int(patch_info.overlap_bottom * self.edge_fade)
        fade_left = int(patch_info.overlap_left * self.edge_fade)
        fade_right = int(patch_info.overlap_right * self.edge_fade)

        if self.blending_mode == "linear":
            # Linear fade at edges
            if fade_top > 0:
                fade_weights = torch.linspace(0, 1, fade_top, device=self.device)
                weights[:fade_top, :] *= fade_weights.unsqueeze(1)

            if fade_bottom > 0:
                fade_weights = torch.linspace(1, 0, fade_bottom, device=self.device)
                weights[-fade_bottom:, :] *= fade_weights.unsqueeze(1)

            if fade_left > 0:
                fade_weights = torch.linspace(0, 1, fade_left, device=self.device)
                weights[:, :fade_left] *= fade_weights.unsqueeze(0)

            if fade_right > 0:
                fade_weights = torch.linspace(1, 0, fade_right, device=self.device)
                weights[:, -fade_right:] *= fade_weights.unsqueeze(0)

        elif self.blending_mode == "cosine":
            # Cosine fade at edges
            if fade_top > 0:
                x = torch.linspace(0, np.pi / 2, fade_top, device=self.device)
                fade_weights = torch.sin(x)
                weights[:fade_top, :] *= fade_weights.unsqueeze(1)

            if fade_bottom > 0:
                x = torch.linspace(np.pi / 2, 0, fade_bottom, device=self.device)
                fade_weights = torch.sin(x)
                weights[-fade_bottom:, :] *= fade_weights.unsqueeze(1)

            if fade_left > 0:
                x = torch.linspace(0, np.pi / 2, fade_left, device=self.device)
                fade_weights = torch.sin(x)
                weights[:, :fade_left] *= fade_weights.unsqueeze(0)

            if fade_right > 0:
                x = torch.linspace(np.pi / 2, 0, fade_right, device=self.device)
                fade_weights = torch.sin(x)
                weights[:, -fade_right:] *= fade_weights.unsqueeze(0)

        elif self.blending_mode == "gaussian":
            # Gaussian-like fade (using distance from edge)
            y_coords, x_coords = torch.meshgrid(
                torch.arange(patch_height, device=self.device),
                torch.arange(patch_width, device=self.device),
                indexing="ij",
            )

            # Distance from each edge
            dist_top = y_coords.float()
            dist_bottom = (patch_height - 1 - y_coords).float()
            dist_left = x_coords.float()
            dist_right = (patch_width - 1 - x_coords).float()

            # Apply fade based on overlap regions
            if fade_top > 0:
                mask = dist_top < fade_top
                weights[mask] *= torch.exp(
                    -(((fade_top - dist_top[mask]) / (fade_top / 3)) ** 2)
                )

            if fade_bottom > 0:
                mask = dist_bottom < fade_bottom
                weights[mask] *= torch.exp(
                    -(((fade_bottom - dist_bottom[mask]) / (fade_bottom / 3)) ** 2)
                )

            if fade_left > 0:
                mask = dist_left < fade_left
                weights[mask] *= torch.exp(
                    -(((fade_left - dist_left[mask]) / (fade_left / 3)) ** 2)
                )

            if fade_right > 0:
                mask = dist_right < fade_right
                weights[mask] *= torch.exp(
                    -(((fade_right - dist_right[mask]) / (fade_right / 3)) ** 2)
                )

        return weights

    @safe_operation("Patch reconstruction")
    def reconstruct_image(
        self,
        patches: List[torch.Tensor],
        patch_infos: List[PatchInfo],
        output_height: int,
        output_width: int,
        output_channels: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Reconstruct full image from patches.

        Args:
            patches: List of processed patch tensors
            patch_infos: List of patch information
            output_height: Height of output image
            output_width: Width of output image
            output_channels: Number of output channels (inferred if None)

        Returns:
            Reconstructed image tensor [B, C, H, W]
        """
        if len(patches) != len(patch_infos):
            raise ValueError("Number of patches must match number of patch infos")

        if not patches:
            raise ValueError("No patches provided for reconstruction")

        # Infer output properties from first patch
        first_patch = patches[0]
        if first_patch.ndim == 3:
            first_patch = first_patch.unsqueeze(0)

        batch_size = first_patch.shape[0]
        if output_channels is None:
            output_channels = first_patch.shape[1]

        # Initialize output tensors
        reconstructed = torch.zeros(
            batch_size,
            output_channels,
            output_height,
            output_width,
            device=self.device,
            dtype=first_patch.dtype,
        )
        weight_sum = torch.zeros(
            output_height, output_width, device=self.device, dtype=first_patch.dtype
        )

        # Process each patch
        for patch, patch_info in zip(patches, patch_infos):
            # Ensure patch has correct dimensions
            if patch.ndim == 3:
                patch = patch.unsqueeze(0)

            # Get actual patch dimensions (may be smaller than expected due to padding)
            actual_h = min(patch.shape[2], patch_info.height)
            actual_w = min(patch.shape[3], patch_info.width)

            # Extract relevant part of patch (remove padding if any)
            patch_data = patch[:, :, :actual_h, :actual_w]

            # Create blending weights
            weights = self.create_blending_weights(patch_info, actual_h, actual_w)

            # Add to reconstruction
            y_start, y_end = patch_info.start_y, patch_info.start_y + actual_h
            x_start, x_end = patch_info.start_x, patch_info.start_x + actual_w

            # Weighted accumulation
            for c in range(output_channels):
                reconstructed[:, c, y_start:y_end, x_start:x_end] += (
                    patch_data[:, c] * weights
                )

            # Accumulate weights
            weight_sum[y_start:y_end, x_start:x_end] += weights

        # Normalize by accumulated weights
        weight_sum = torch.clamp(weight_sum, min=1e-8)  # Avoid division by zero
        for c in range(output_channels):
            reconstructed[:, c] /= weight_sum

        logger.debug(f"Reconstructed image: {reconstructed.shape}")

        return reconstructed


class MemoryEfficientPatchProcessor:
    """
    Memory-efficient patch processing pipeline.

    This class provides a complete pipeline for processing large images
    through patch extraction, processing, and reconstruction while
    managing memory usage efficiently.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 512,
        overlap: Union[int, Tuple[int, int]] = 64,
        max_patches_in_memory: int = 16,
        device: str = "auto",
        enable_checkpointing: bool = True,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize memory-efficient patch processor.

        Args:
            patch_size: Size of patches to extract
            overlap: Overlap between patches
            max_patches_in_memory: Maximum number of patches to keep in memory
            device: Device for processing ('auto', 'cuda', 'cpu')
            enable_checkpointing: Whether to save intermediate results
            checkpoint_dir: Directory for checkpoints
        """
        self.device = self._setup_device(device)
        self.max_patches_in_memory = max_patches_in_memory
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Initialize components
        self.extractor = PatchExtractor(
            patch_size=patch_size, overlap=overlap, device=self.device
        )
        self.reconstructor = PatchReconstructor(device=self.device)

        # Memory management
        self.patch_cache = {}
        self.processed_patches = {}

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized MemoryEfficientPatchProcessor: "
            f"patch_size={patch_size}, overlap={overlap}, "
            f"max_patches={max_patches_in_memory}, device={self.device}"
        )

    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = "cpu"

        return device

    def estimate_memory_usage(
        self,
        image_height: int,
        image_width: int,
        channels: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, float]:
        """
        Estimate memory usage for processing given image.

        Args:
            image_height: Height of input image
            image_width: Width of input image
            channels: Number of channels
            dtype: Data type

        Returns:
            Dictionary with memory estimates in MB
        """
        # Calculate patch grid
        patch_infos, _, _ = self.extractor.calculate_patch_grid(
            image_height, image_width
        )
        num_patches = len(patch_infos)

        # Bytes per element
        bytes_per_element = torch.tensor(0, dtype=dtype).element_size()

        # Memory for single patch
        patch_h, patch_w = self.extractor.patch_size
        patch_memory = channels * patch_h * patch_w * bytes_per_element

        # Memory for full image
        full_image_memory = channels * image_height * image_width * bytes_per_element

        # Memory for patches in cache
        cache_memory = self.max_patches_in_memory * patch_memory

        # Convert to MB
        mb = 1024 * 1024

        estimates = {
            "full_image_mb": full_image_memory / mb,
            "single_patch_mb": patch_memory / mb,
            "patch_cache_mb": cache_memory / mb,
            "total_patches": num_patches,
            "estimated_peak_mb": (full_image_memory + cache_memory) / mb,
        }

        return estimates

    def process_image(
        self,
        image: torch.Tensor,
        processing_func: Callable[[torch.Tensor], torch.Tensor],
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Process large image through patch-based pipeline.

        Args:
            image: Input image tensor
            processing_func: Function to apply to each patch
            batch_size: Number of patches to process simultaneously
            show_progress: Whether to show progress information

        Returns:
            Processed full image
        """
        # Handle input dimensions
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)

        B, C, H, W = image.shape

        if B > 1:
            logger.warning(f"Batch size {B} > 1, processing first image only")
            image = image[0:1]

        logger.info(f"Processing image {H}x{W} with {C} channels")

        # Estimate memory usage
        memory_est = self.estimate_memory_usage(H, W, C, image.dtype)
        logger.info(
            f"Estimated memory usage: {memory_est['estimated_peak_mb']:.1f} MB "
            f"for {memory_est['total_patches']} patches"
        )

        # Extract patches
        patches, patch_infos = self.extractor.extract_patches(image, return_info=True)

        # Process patches in batches
        processed_patches = []

        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i : i + batch_size]
            batch_infos = patch_infos[i : i + batch_size]

            if show_progress:
                logger.info(
                    f"Processing patch batch {i//batch_size + 1}/{math.ceil(len(patches)/batch_size)}"
                )

            # Process batch
            batch_processed = []
            for patch in batch_patches:
                try:
                    processed_patch = processing_func(patch)
                    batch_processed.append(processed_patch)
                except Exception as e:
                    logger.error(f"Failed to process patch: {e}")
                    # Use original patch as fallback
                    batch_processed.append(patch)

            processed_patches.extend(batch_processed)

            # Memory management
            if i % (self.max_patches_in_memory * batch_size) == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # Reconstruct image
        logger.info("Reconstructing full image from processed patches")
        reconstructed = self.reconstructor.reconstruct_image(
            processed_patches, patch_infos, H, W, C
        )

        return reconstructed

    def process_image_streaming(
        self,
        image: torch.Tensor,
        processing_func: Callable[[torch.Tensor], torch.Tensor],
        output_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Process image with streaming to minimize memory usage.

        Args:
            image: Input image tensor
            processing_func: Function to apply to each patch
            output_path: Optional path to save intermediate results

        Returns:
            Processed full image
        """
        # Handle input dimensions
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)

        B, C, H, W = image.shape

        # Initialize reconstruction tensors
        reconstructed = torch.zeros(B, C, H, W, device=self.device, dtype=image.dtype)
        weight_sum = torch.zeros(H, W, device=self.device, dtype=image.dtype)

        # Extract and process patches one by one
        patch_infos, _, _ = self.extractor.calculate_patch_grid(H, W)

        for i, patch_info in enumerate(patch_infos):
            logger.debug(f"Processing patch {i+1}/{len(patch_infos)}")

            # Extract single patch
            patch = self.extractor.extract_patch_at_position(
                image,
                patch_info.start_y,
                patch_info.start_x,
                patch_info.height,
                patch_info.width,
            )

            # Process patch
            try:
                processed_patch = processing_func(patch)
            except Exception as e:
                logger.warning(f"Failed to process patch {i}: {e}, using original")
                processed_patch = patch

            # Add to reconstruction immediately
            actual_h = min(processed_patch.shape[2], patch_info.height)
            actual_w = min(processed_patch.shape[3], patch_info.width)

            # Create weights
            weights = self.reconstructor.create_blending_weights(
                patch_info, actual_h, actual_w
            )

            # Add to reconstruction
            y_start, y_end = patch_info.start_y, patch_info.start_y + actual_h
            x_start, x_end = patch_info.start_x, patch_info.start_x + actual_w

            patch_data = processed_patch[:, :, :actual_h, :actual_w]

            for c in range(C):
                reconstructed[:, c, y_start:y_end, x_start:x_end] += (
                    patch_data[:, c] * weights
                )

            weight_sum[y_start:y_end, x_start:x_end] += weights

            # Clear memory
            del patch, processed_patch, patch_data

            # Periodic cleanup
            if i % 10 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # Normalize by weights
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        for c in range(C):
            reconstructed[:, c] /= weight_sum

        # Save if requested
        if output_path:
            torch.save(reconstructed, output_path)
            logger.info(f"Saved reconstructed image to {output_path}")

        return reconstructed


# Utility functions for patch processing
def calculate_optimal_patch_size(
    image_height: int,
    image_width: int,
    available_memory_gb: float = 4.0,
    channels: int = 1,
    dtype: torch.dtype = torch.float32,
    safety_factor: float = 0.8,
) -> Tuple[int, int]:
    """
    Calculate optimal patch size based on available memory.

    Args:
        image_height: Height of input image
        image_width: Width of input image
        available_memory_gb: Available memory in GB
        channels: Number of image channels
        dtype: Data type
        safety_factor: Safety factor for memory usage

    Returns:
        (patch_height, patch_width) tuple
    """
    # Bytes per element
    bytes_per_element = torch.tensor(0, dtype=dtype).element_size()

    # Available memory in bytes
    available_bytes = available_memory_gb * 1024**3 * safety_factor

    # Account for multiple copies (input, output, intermediate)
    effective_memory = available_bytes / 4  # Conservative estimate

    # Maximum pixels per patch
    max_pixels = effective_memory / (channels * bytes_per_element)

    # Calculate square patch size
    max_patch_size = int(math.sqrt(max_pixels))

    # Ensure reasonable bounds
    max_patch_size = max(64, min(max_patch_size, 2048))

    # Adjust to be divisible by 8 (common requirement for neural networks)
    max_patch_size = (max_patch_size // 8) * 8

    # Use same size for both dimensions
    patch_height = min(max_patch_size, image_height)
    patch_width = min(max_patch_size, image_width)

    logger.info(
        f"Calculated optimal patch size: {patch_height}x{patch_width} "
        f"for image {image_height}x{image_width} with {available_memory_gb:.1f}GB memory"
    )

    return patch_height, patch_width


def create_patch_processor(
    image_height: int,
    image_width: int,
    target_patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    overlap_ratio: float = 0.125,
    available_memory_gb: float = 4.0,
    device: str = "auto",
) -> MemoryEfficientPatchProcessor:
    """
    Create optimally configured patch processor for given image.

    Args:
        image_height: Height of input image
        image_width: Width of input image
        target_patch_size: Target patch size (calculated if None)
        overlap_ratio: Ratio of patch size to use for overlap
        available_memory_gb: Available memory in GB
        device: Device for processing

    Returns:
        Configured patch processor
    """
    if target_patch_size is None:
        patch_h, patch_w = calculate_optimal_patch_size(
            image_height, image_width, available_memory_gb
        )
        patch_size = (patch_h, patch_w)
    else:
        if isinstance(target_patch_size, int):
            patch_size = (target_patch_size, target_patch_size)
        else:
            patch_size = target_patch_size

    # Ensure patch size doesn't exceed image dimensions
    patch_h = min(patch_size[0], image_height)
    patch_w = min(patch_size[1], image_width)
    patch_size = (patch_h, patch_w)

    # Calculate overlap
    overlap_h = int(patch_size[0] * overlap_ratio)
    overlap_w = int(patch_size[1] * overlap_ratio)
    overlap = (overlap_h, overlap_w)

    # Calculate max patches in memory based on available memory
    bytes_per_element = 4  # float32
    patch_memory = patch_size[0] * patch_size[1] * bytes_per_element
    max_patches = max(1, int(available_memory_gb * 1024**3 * 0.5 / patch_memory))
    max_patches = min(max_patches, 64)  # Reasonable upper bound

    processor = MemoryEfficientPatchProcessor(
        patch_size=patch_size,
        overlap=overlap,
        max_patches_in_memory=max_patches,
        device=device,
    )

    return processor
