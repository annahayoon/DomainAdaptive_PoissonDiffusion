"""
Comprehensive tests for patch processing system.

Tests patch extraction, reconstruction, and memory-efficient processing
for large image handling.
"""

import gc
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from core.patch_processing import (
    MemoryEfficientPatchProcessor,
    PatchExtractor,
    PatchInfo,
    PatchReconstructor,
    calculate_optimal_patch_size,
    create_patch_processor,
)


class TestPatchInfo:
    """Test PatchInfo dataclass."""

    def test_patch_info_creation(self):
        """Test basic patch info creation."""
        patch_info = PatchInfo(
            start_y=10,
            start_x=20,
            end_y=138,
            end_x=148,
            height=128,
            width=128,
            overlap_top=16,
            overlap_bottom=16,
            overlap_left=16,
            overlap_right=16,
            patch_id=5,
            row=1,
            col=2,
        )

        assert patch_info.start_y == 10
        assert patch_info.start_x == 20
        assert patch_info.height == 128
        assert patch_info.width == 128
        assert patch_info.patch_id == 5

    def test_patch_info_serialization(self):
        """Test patch info serialization."""
        patch_info = PatchInfo(
            start_y=0,
            start_x=0,
            end_y=128,
            end_x=128,
            height=128,
            width=128,
            overlap_top=0,
            overlap_bottom=16,
            overlap_left=0,
            overlap_right=16,
            patch_id=0,
            row=0,
            col=0,
        )

        data_dict = patch_info.to_dict()

        assert isinstance(data_dict, dict)
        assert data_dict["start_y"] == 0
        assert data_dict["height"] == 128
        assert data_dict["patch_id"] == 0


class TestPatchExtractor:
    """Test patch extraction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.extractor = PatchExtractor(patch_size=128, overlap=32, device=self.device)

    def test_extractor_initialization(self):
        """Test patch extractor initialization."""
        assert self.extractor.patch_size == (128, 128)
        assert self.extractor.overlap == (32, 32)
        assert self.extractor.device == "cpu"

    def test_extractor_with_tuple_sizes(self):
        """Test extractor with tuple patch sizes."""
        extractor = PatchExtractor(
            patch_size=(256, 128), overlap=(64, 32), device="cpu"
        )

        assert extractor.patch_size == (256, 128)
        assert extractor.overlap == (64, 32)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid patch size
        with pytest.raises(ValueError):
            PatchExtractor(patch_size=0)

        # Invalid overlap
        with pytest.raises(ValueError):
            PatchExtractor(patch_size=128, overlap=-1)

        # Overlap too large
        with pytest.raises(ValueError):
            PatchExtractor(patch_size=128, overlap=128)

    def test_patch_grid_calculation(self):
        """Test patch grid calculation."""
        # Test with image that fits exactly
        patch_infos, num_rows, num_cols = self.extractor.calculate_patch_grid(256, 256)

        # With patch_size=128, overlap=32, step=96
        # 256 pixels needs ceil((256-32)/96) = ceil(224/96) = 3 patches per dimension
        assert num_rows == 3
        assert num_cols == 3
        assert len(patch_infos) == 9

        # Check first patch
        first_patch = patch_infos[0]
        assert first_patch.start_y == 0
        assert first_patch.start_x == 0
        assert first_patch.height == 128
        assert first_patch.width == 128
        assert first_patch.row == 0
        assert first_patch.col == 0

    def test_patch_grid_with_small_image(self):
        """Test patch grid with image smaller than patch size."""
        patch_infos, num_rows, num_cols = self.extractor.calculate_patch_grid(64, 64)

        # Should create single patch
        assert num_rows == 1
        assert num_cols == 1
        assert len(patch_infos) == 1

        patch = patch_infos[0]
        assert patch.height == 64
        assert patch.width == 64

    def test_extract_patches_2d_image(self):
        """Test patch extraction from 2D image."""
        # Create test image
        image = torch.rand(256, 256)

        patches, patch_infos = self.extractor.extract_patches(image, return_info=True)

        assert len(patches) == len(patch_infos)
        assert len(patches) == 9  # 3x3 grid

        # Check patch dimensions
        for patch in patches:
            assert patch.shape == (1, 1, 128, 128)  # [B, C, H, W]

    def test_extract_patches_3d_image(self):
        """Test patch extraction from 3D image."""
        # Create test image [C, H, W]
        image = torch.rand(3, 256, 256)

        patches, patch_infos = self.extractor.extract_patches(image, return_info=True)

        assert len(patches) == 9

        # Check patch dimensions
        for patch in patches:
            assert patch.shape == (1, 3, 128, 128)  # [B, C, H, W]

    def test_extract_patches_4d_image(self):
        """Test patch extraction from 4D image."""
        # Create test image [B, C, H, W]
        image = torch.rand(2, 3, 256, 256)

        patches, patch_infos = self.extractor.extract_patches(image, return_info=True)

        # Should process only first batch element
        assert len(patches) == 9

        for patch in patches:
            assert patch.shape == (1, 3, 128, 128)

    def test_extract_patches_without_info(self):
        """Test patch extraction without returning info."""
        image = torch.rand(256, 256)

        patches = self.extractor.extract_patches(image, return_info=False)

        assert isinstance(patches, list)
        assert len(patches) == 9

    def test_extract_patch_at_position(self):
        """Test extracting single patch at specific position."""
        image = torch.rand(256, 256)

        patch = self.extractor.extract_patch_at_position(image, 50, 60)

        assert patch.shape == (1, 1, 128, 128)

    def test_extract_patch_at_edge(self):
        """Test extracting patch at image edge with padding."""
        image = torch.rand(200, 200)

        # Extract patch near edge
        patch = self.extractor.extract_patch_at_position(image, 150, 150)

        # Should be padded to full patch size
        assert patch.shape == (1, 1, 128, 128)

    def test_different_padding_modes(self):
        """Test different padding modes."""
        extractor_reflect = PatchExtractor(patch_size=128, padding_mode="reflect")
        extractor_constant = PatchExtractor(patch_size=128, padding_mode="constant")

        image = torch.rand(100, 100)  # Smaller than patch size

        patch_reflect = extractor_reflect.extract_patches(image, return_info=False)[0]
        patch_constant = extractor_constant.extract_patches(image, return_info=False)[0]

        assert patch_reflect.shape == (1, 1, 128, 128)
        assert patch_constant.shape == (1, 1, 128, 128)

        # Patches should be different due to different padding
        assert not torch.allclose(patch_reflect, patch_constant)


class TestPatchReconstructor:
    """Test patch reconstruction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.reconstructor = PatchReconstructor(device=self.device)

    def test_reconstructor_initialization(self):
        """Test patch reconstructor initialization."""
        assert self.reconstructor.blending_mode == "linear"
        assert self.reconstructor.device == "cpu"

    def test_different_blending_modes(self):
        """Test different blending modes."""
        modes = ["linear", "cosine", "gaussian"]

        for mode in modes:
            reconstructor = PatchReconstructor(blending_mode=mode, device="cpu")
            assert reconstructor.blending_mode == mode

    def test_create_blending_weights(self):
        """Test blending weight creation."""
        patch_info = PatchInfo(
            start_y=0,
            start_x=0,
            end_y=128,
            end_x=128,
            height=128,
            width=128,
            overlap_top=0,
            overlap_bottom=32,
            overlap_left=0,
            overlap_right=32,
            patch_id=0,
            row=0,
            col=0,
        )

        weights = self.reconstructor.create_blending_weights(patch_info, 128, 128)

        assert weights.shape == (128, 128)
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)

        # Check that overlap regions have reduced weights
        assert weights[-1, -1] < weights[64, 64]  # Bottom-right corner should be faded

    def test_reconstruct_simple_image(self):
        """Test reconstruction of simple non-overlapping patches."""
        # Create simple test case: 2x2 patches from 256x256 image
        extractor = PatchExtractor(patch_size=128, overlap=0, device="cpu")

        # Create test image with known pattern
        original = torch.zeros(1, 1, 256, 256)
        original[:, :, :128, :128] = 1.0  # Top-left
        original[:, :, :128, 128:] = 2.0  # Top-right
        original[:, :, 128:, :128] = 3.0  # Bottom-left
        original[:, :, 128:, 128:] = 4.0  # Bottom-right

        # Extract patches
        patches, patch_infos = extractor.extract_patches(original, return_info=True)

        # Reconstruct
        reconstructed = self.reconstructor.reconstruct_image(
            patches, patch_infos, 256, 256, 1
        )

        # Should match original exactly (no overlap)
        assert torch.allclose(reconstructed, original, atol=1e-6)

    def test_reconstruct_overlapping_patches(self):
        """Test reconstruction with overlapping patches."""
        extractor = PatchExtractor(patch_size=128, overlap=32, device="cpu")

        # Create uniform test image
        original = torch.ones(1, 1, 256, 256) * 0.5

        # Extract patches
        patches, patch_infos = extractor.extract_patches(original, return_info=True)

        # Reconstruct
        reconstructed = self.reconstructor.reconstruct_image(
            patches, patch_infos, 256, 256, 1
        )

        # Should be close to original (some blending artifacts expected)
        assert torch.allclose(reconstructed, original, atol=0.1)

    def test_reconstruct_multichannel(self):
        """Test reconstruction of multi-channel image."""
        extractor = PatchExtractor(patch_size=64, overlap=16, device="cpu")

        # Create multi-channel test image
        original = torch.rand(1, 3, 128, 128)

        # Extract patches
        patches, patch_infos = extractor.extract_patches(original, return_info=True)

        # Reconstruct
        reconstructed = self.reconstructor.reconstruct_image(
            patches, patch_infos, 128, 128, 3
        )

        assert reconstructed.shape == (1, 3, 128, 128)
        # Should be reasonably close to original
        assert torch.allclose(reconstructed, original, atol=0.2)

    def test_reconstruct_empty_patches(self):
        """Test error handling with empty patch list."""
        with pytest.raises(ValueError):
            self.reconstructor.reconstruct_image([], [], 256, 256, 1)

    def test_reconstruct_mismatched_patches_info(self):
        """Test error handling with mismatched patches and info."""
        patches = [torch.rand(1, 1, 128, 128)]
        patch_infos = []  # Empty info list

        with pytest.raises(ValueError):
            self.reconstructor.reconstruct_image(patches, patch_infos, 256, 256, 1)


class TestMemoryEfficientPatchProcessor:
    """Test memory-efficient patch processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.processor = MemoryEfficientPatchProcessor(
            patch_size=64, overlap=16, max_patches_in_memory=4, device=self.device
        )

    def test_processor_initialization(self):
        """Test processor initialization."""
        assert self.processor.device == "cpu"
        assert self.processor.max_patches_in_memory == 4
        assert isinstance(self.processor.extractor, PatchExtractor)
        assert isinstance(self.processor.reconstructor, PatchReconstructor)

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        estimates = self.processor.estimate_memory_usage(512, 512, 3)

        assert "full_image_mb" in estimates
        assert "single_patch_mb" in estimates
        assert "patch_cache_mb" in estimates
        assert "total_patches" in estimates
        assert "estimated_peak_mb" in estimates

        assert estimates["total_patches"] > 0
        assert estimates["estimated_peak_mb"] > 0

    def test_process_image_identity(self):
        """Test processing image with identity function."""
        # Create test image
        image = torch.rand(1, 1, 128, 128)

        # Identity processing function
        def identity_func(patch):
            return patch

        # Process
        result = self.processor.process_image(image, identity_func, show_progress=False)

        assert result.shape == image.shape
        # Should be close to original (some blending artifacts)
        assert torch.allclose(result, image, atol=0.1)

    def test_process_image_simple_transform(self):
        """Test processing with simple transformation."""
        # Create test image
        image = torch.ones(1, 1, 128, 128) * 0.5

        # Double intensity function
        def double_func(patch):
            return patch * 2.0

        # Process
        result = self.processor.process_image(image, double_func, show_progress=False)

        assert result.shape == image.shape
        # Should be approximately doubled
        assert torch.allclose(result, image * 2.0, atol=0.1)

    def test_process_image_with_batching(self):
        """Test processing with different batch sizes."""
        image = torch.rand(1, 1, 128, 128)

        def identity_func(patch):
            return patch

        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            result = self.processor.process_image(
                image, identity_func, batch_size=batch_size, show_progress=False
            )
            assert result.shape == image.shape

    def test_process_image_streaming(self):
        """Test streaming processing."""
        image = torch.rand(1, 1, 128, 128)

        def identity_func(patch):
            return patch

        result = self.processor.process_image_streaming(image, identity_func)

        assert result.shape == image.shape
        assert torch.allclose(result, image, atol=0.1)

    def test_process_image_with_error_handling(self):
        """Test error handling during processing."""
        image = torch.rand(1, 1, 128, 128)

        # Function that fails sometimes
        call_count = 0

        def failing_func(patch):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second call
                raise RuntimeError("Simulated processing error")
            return patch

        # Should handle errors gracefully
        result = self.processor.process_image(image, failing_func, show_progress=False)

        assert result.shape == image.shape

    def test_device_setup(self):
        """Test device setup logic."""
        # Test auto device selection
        processor_auto = MemoryEfficientPatchProcessor(device="auto")
        assert processor_auto.device in ["cpu", "cuda"]

        # Test CPU fallback
        processor_cpu = MemoryEfficientPatchProcessor(device="cpu")
        assert processor_cpu.device == "cpu"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_calculate_optimal_patch_size(self):
        """Test optimal patch size calculation."""
        # Test with different memory constraints
        patch_h, patch_w = calculate_optimal_patch_size(
            2048, 2048, available_memory_gb=1.0
        )

        assert patch_h > 0
        assert patch_w > 0
        assert patch_h <= 2048
        assert patch_w <= 2048

        # Should be divisible by 8
        assert patch_h % 8 == 0
        assert patch_w % 8 == 0

    def test_calculate_optimal_patch_size_small_image(self):
        """Test optimal patch size for small image."""
        patch_h, patch_w = calculate_optimal_patch_size(
            128, 128, available_memory_gb=1.0
        )

        # Should not exceed image size
        assert patch_h <= 128
        assert patch_w <= 128

    def test_calculate_optimal_patch_size_large_memory(self):
        """Test optimal patch size with large memory."""
        patch_h, patch_w = calculate_optimal_patch_size(
            4096, 4096, available_memory_gb=16.0
        )

        # Should be reasonable size (not too large)
        assert patch_h <= 2048
        assert patch_w <= 2048

    def test_create_patch_processor(self):
        """Test patch processor creation."""
        processor = create_patch_processor(
            image_height=1024, image_width=1024, available_memory_gb=2.0, device="cpu"
        )

        assert isinstance(processor, MemoryEfficientPatchProcessor)
        assert processor.device == "cpu"

    def test_create_patch_processor_with_target_size(self):
        """Test patch processor creation with target size."""
        processor = create_patch_processor(
            image_height=1024, image_width=1024, target_patch_size=256, device="cpu"
        )

        assert processor.extractor.patch_size == (256, 256)

    def test_create_patch_processor_with_tuple_size(self):
        """Test patch processor creation with tuple target size."""
        processor = create_patch_processor(
            image_height=1024,
            image_width=1024,
            target_patch_size=(512, 256),
            device="cpu",
        )

        assert processor.extractor.patch_size == (512, 256)


class TestIntegration:
    """Integration tests for complete patch processing pipeline."""

    def test_end_to_end_processing(self):
        """Test complete end-to-end patch processing."""
        # Create large test image
        image = torch.rand(1, 3, 512, 512)

        # Create processor
        processor = MemoryEfficientPatchProcessor(
            patch_size=128, overlap=32, device="cpu"
        )

        # Simple processing function (add noise)
        def add_noise_func(patch):
            noise = torch.randn_like(patch) * 0.01
            return torch.clamp(patch + noise, 0, 1)

        # Process image
        result = processor.process_image(image, add_noise_func, show_progress=False)

        assert result.shape == image.shape
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

        # Should be different from original (noise added)
        assert not torch.allclose(result, image, atol=0.005)

    def test_large_image_processing(self):
        """Test processing of large image."""
        # Create large image (simulate high-resolution scientific image)
        image = torch.rand(1, 1, 1024, 1024)

        # Create processor with small patches to test many patches
        processor = MemoryEfficientPatchProcessor(
            patch_size=64,
            overlap=16,
            max_patches_in_memory=2,  # Force memory management
            device="cpu",
        )

        # Simple processing function
        def normalize_func(patch):
            return (patch - patch.mean()) / (patch.std() + 1e-8) * 0.1 + 0.5

        # Process with streaming to test memory efficiency
        result = processor.process_image_streaming(image, normalize_func)

        assert result.shape == image.shape
        assert torch.all(torch.isfinite(result))

    def test_reconstruction_quality(self):
        """Test reconstruction quality with known pattern."""
        # Create image with checkerboard pattern
        image = torch.zeros(1, 1, 256, 256)
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    image[:, :, i : i + 32, j : j + 32] = 1.0

        # Process with identity function
        processor = MemoryEfficientPatchProcessor(
            patch_size=64, overlap=16, device="cpu"
        )

        def identity_func(patch):
            return patch

        result = processor.process_image(image, identity_func, show_progress=False)

        # Check that pattern is preserved
        assert result.shape == image.shape

        # Calculate reconstruction error
        mse = torch.mean((result - image) ** 2)
        assert mse < 0.01  # Should have low reconstruction error

    def test_memory_cleanup(self):
        """Test that memory is properly cleaned up."""
        # This test is more about ensuring no memory leaks
        # Actual memory measurement would be platform-specific

        image = torch.rand(1, 1, 512, 512)
        processor = MemoryEfficientPatchProcessor(
            patch_size=128, overlap=32, max_patches_in_memory=2, device="cpu"
        )

        def identity_func(patch):
            return patch

        # Process multiple times to test cleanup
        for _ in range(3):
            result = processor.process_image(image, identity_func, show_progress=False)
            assert result.shape == image.shape

            # Force garbage collection
            del result
            gc.collect()


class TestErrorHandling:
    """Test error handling in patch processing."""

    def test_invalid_patch_size(self):
        """Test handling of invalid patch sizes."""
        with pytest.raises(ValueError):
            PatchExtractor(patch_size=0)

        with pytest.raises(ValueError):
            PatchExtractor(patch_size=-1)

    def test_invalid_overlap(self):
        """Test handling of invalid overlap."""
        with pytest.raises(ValueError):
            PatchExtractor(patch_size=128, overlap=-1)

        with pytest.raises(ValueError):
            PatchExtractor(patch_size=128, overlap=128)

    def test_empty_image(self):
        """Test handling of empty images."""
        extractor = PatchExtractor(patch_size=64, device="cpu")

        # Empty tensor
        empty_image = torch.empty(0, 0)

        with pytest.raises((ValueError, RuntimeError)):
            extractor.extract_patches(empty_image)

    def test_processing_function_error(self):
        """Test handling of processing function errors."""
        processor = MemoryEfficientPatchProcessor(
            patch_size=64, overlap=16, device="cpu"
        )

        image = torch.rand(1, 1, 128, 128)

        def failing_func(patch):
            raise RuntimeError("Processing failed")

        # Should handle errors gracefully and use original patches
        result = processor.process_image(image, failing_func, show_progress=False)

        assert result.shape == image.shape
        # Should be close to original (fallback behavior)
        assert torch.allclose(result, image, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
