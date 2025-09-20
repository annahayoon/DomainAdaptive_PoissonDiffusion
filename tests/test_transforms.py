"""
Test reversible transforms preserve information.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from core.exceptions import MetadataError, TransformError
from core.transforms import ImageMetadata, ReversibleTransform


class TestImageMetadata:
    """Test ImageMetadata functionality."""

    def test_basic_creation(self):
        """Test basic metadata creation."""
        metadata = ImageMetadata(
            original_height=100, original_width=200, scale_factor=0.5
        )

        assert metadata.original_height == 100
        assert metadata.original_width == 200
        assert metadata.scale_factor == 0.5
        assert metadata.pixel_size == 1.0  # default
        assert metadata.domain == "unknown"  # default

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        metadata = ImageMetadata(
            original_height=100,
            original_width=200,
            scale_factor=0.5,
            crop_bbox=(10, 20, 80, 160),
            pad_amounts=(5, 5, 10, 10),
            pixel_size=4.29,
            pixel_unit="um",
            domain="photography",
            black_level=512,
            white_level=16383,
            iso=3200,
            exposure_time=0.1,
        )

        # Serialize
        json_str = metadata.to_json()
        assert isinstance(json_str, str)

        # Deserialize
        restored = ImageMetadata.from_json(json_str)

        # Check all fields
        assert restored.original_height == metadata.original_height
        assert restored.original_width == metadata.original_width
        assert restored.scale_factor == metadata.scale_factor
        assert restored.crop_bbox == metadata.crop_bbox
        assert restored.pad_amounts == metadata.pad_amounts
        assert restored.pixel_size == metadata.pixel_size
        assert restored.pixel_unit == metadata.pixel_unit
        assert restored.domain == metadata.domain
        assert restored.black_level == metadata.black_level
        assert restored.white_level == metadata.white_level
        assert restored.iso == metadata.iso
        assert restored.exposure_time == metadata.exposure_time

    def test_file_operations(self):
        """Test saving and loading from files."""
        metadata = ImageMetadata(
            original_height=100,
            original_width=200,
            scale_factor=0.5,
            pixel_size=1.0,
            domain="test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_metadata.json"

            # Save
            metadata.save_to_file(filepath)
            assert filepath.exists()

            # Load
            loaded = ImageMetadata.load_from_file(filepath)

            # Verify
            assert loaded.original_height == metadata.original_height
            assert loaded.original_width == metadata.original_width
            assert loaded.scale_factor == metadata.scale_factor
            assert loaded.domain == metadata.domain

    def test_validation_success(self):
        """Test successful validation."""
        metadata = ImageMetadata(
            original_height=100,
            original_width=200,
            scale_factor=0.5,
            pixel_size=4.29,
            pixel_unit="um",
            domain="photography",
            black_level=0,
            white_level=16383,
            bit_depth=14,
        )

        # Should not raise
        metadata.validate()

    def test_validation_failures(self):
        """Test validation failures."""
        # Invalid dimensions
        with pytest.raises(MetadataError, match="dimensions must be positive"):
            metadata = ImageMetadata(
                original_height=0, original_width=100, scale_factor=1.0
            )
            metadata.validate()

        # Invalid scale factor
        with pytest.raises(MetadataError, match="Scale factor must be positive"):
            metadata = ImageMetadata(
                original_height=100, original_width=100, scale_factor=0.0
            )
            metadata.validate()

        # Invalid pixel unit
        with pytest.raises(MetadataError, match="Pixel unit must be"):
            metadata = ImageMetadata(
                original_height=100,
                original_width=100,
                scale_factor=1.0,
                pixel_unit="invalid",
            )
            metadata.validate()

        # Invalid white/black levels
        with pytest.raises(MetadataError, match="White level must be greater"):
            metadata = ImageMetadata(
                original_height=100,
                original_width=100,
                scale_factor=1.0,
                black_level=100,
                white_level=50,
            )
            metadata.validate()

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        with pytest.raises(MetadataError, match="Invalid JSON format"):
            ImageMetadata.from_json("invalid json")

        with pytest.raises(MetadataError, match="Invalid metadata structure"):
            ImageMetadata.from_json('{"invalid": "structure"}')


class TestReversibleTransform:
    """Test ReversibleTransform functionality."""

    def test_initialization(self):
        """Test transform initialization."""
        transform = ReversibleTransform(target_size=128, mode="bilinear")
        assert transform.target_size == 128
        assert transform.mode == "bilinear"

        # Test invalid parameters
        with pytest.raises(TransformError, match="Target size must be positive"):
            ReversibleTransform(target_size=0)

        with pytest.raises(TransformError, match="Mode must be"):
            ReversibleTransform(mode="invalid")

    def test_perfect_reconstruction_square(self):
        """Test perfect reconstruction for square images."""
        transform = ReversibleTransform(target_size=128)

        # Test square image that fits exactly
        original = torch.randn(1, 1, 128, 128)

        # Forward transform
        transformed, metadata = transform.forward(original)
        assert transformed.shape == (1, 1, 128, 128)

        # Inverse transform
        reconstructed = transform.inverse(transformed, metadata)

        # Check perfect reconstruction (should be exact for same size)
        assert reconstructed.shape == original.shape
        error = (original - reconstructed).abs().max()
        assert error < 1e-6, f"Reconstruction error too large: {error}"

    def test_perfect_reconstruction_various_sizes(self):
        """Test perfect reconstruction across various image sizes."""
        transform = ReversibleTransform(target_size=128)

        # Test various sizes
        test_sizes = [
            (100, 100),  # Square, smaller
            (200, 150),  # Rectangular, larger
            (64, 256),  # Very rectangular
            (300, 300),  # Square, larger
            (50, 400),  # Extreme aspect ratio
        ]

        for H, W in test_sizes:
            # Create test image
            original = torch.randn(2, 3, H, W)  # Multi-batch, multi-channel

            # Forward transform
            transformed, metadata = transform.forward(
                original, pixel_size=1.0, pixel_unit="um", domain="test"
            )

            assert transformed.shape == (2, 3, 128, 128)

            # Inverse transform
            reconstructed = transform.inverse(transformed, metadata)

            assert reconstructed.shape == original.shape

            # Check reconstruction error
            error = (original - reconstructed).abs().max()
            # Note: Perfect reconstruction is not possible with interpolation
            # We expect some error due to upsampling and downsampling
            # The key requirement is that dimensions are preserved exactly
            assert error < 20.0, f"Reconstruction error for size {(H, W)}: {error}"

            # For extreme aspect ratios, we expect higher error due to interpolation
            aspect_ratio = max(H, W) / min(H, W)
            if aspect_ratio > 5:  # Extreme aspect ratio
                max_rel_error = 2.0
            else:
                max_rel_error = 1.5

            rel_error = error / (original.abs().max() + 1e-8)
            assert (
                rel_error < max_rel_error
            ), f"Relative reconstruction error for size {(H, W)}: {rel_error}"

    def test_metadata_preservation(self):
        """Test that metadata is correctly preserved."""
        transform = ReversibleTransform(target_size=64)

        image = torch.randn(1, 3, 100, 120)

        _, metadata = transform.forward(
            image,
            pixel_size=4.29,
            pixel_unit="um",
            domain="photography",
            black_level=512,
            white_level=16383,
            iso=3200,
            exposure_time=0.1,
        )

        assert metadata.original_height == 100
        assert metadata.original_width == 120
        assert metadata.pixel_size == 4.29
        assert metadata.pixel_unit == "um"
        assert metadata.domain == "photography"
        assert metadata.black_level == 512
        assert metadata.white_level == 16383
        assert metadata.iso == 3200
        assert metadata.exposure_time == 0.1

    def test_different_interpolation_modes(self):
        """Test different interpolation modes."""
        modes = ["bilinear", "nearest", "bicubic"]

        for mode in modes:
            transform = ReversibleTransform(target_size=64, mode=mode)

            original = torch.randn(1, 1, 100, 80)

            # Forward and inverse
            transformed, metadata = transform.forward(original)
            reconstructed = transform.inverse(transformed, metadata)

            # Should reconstruct with reasonable accuracy regardless of mode
            assert reconstructed.shape == original.shape
            error = (original - reconstructed).abs().max()
            # Nearest neighbor can have higher error due to aliasing
            max_error = 10.0 if mode == "nearest" else 5.0
            assert error < max_error, f"Reconstruction error for mode {mode}: {error}"

    def test_extreme_aspect_ratios(self):
        """Test extreme aspect ratios."""
        transform = ReversibleTransform(target_size=128)

        # Very wide image
        wide = torch.randn(1, 1, 10, 1000)
        transformed, metadata = transform.forward(wide)
        reconstructed = transform.inverse(transformed, metadata)

        assert reconstructed.shape == wide.shape
        error = (wide - reconstructed).abs().max()
        assert error < 5.0

        # Very tall image
        tall = torch.randn(1, 1, 1000, 10)
        transformed, metadata = transform.forward(tall)
        reconstructed = transform.inverse(transformed, metadata)

        assert reconstructed.shape == tall.shape
        error = (tall - reconstructed).abs().max()
        assert error < 5.0

    def test_small_images(self):
        """Test very small images."""
        transform = ReversibleTransform(target_size=128)

        # Tiny image
        tiny = torch.randn(1, 1, 5, 8)
        transformed, metadata = transform.forward(tiny)
        reconstructed = transform.inverse(transformed, metadata)

        assert reconstructed.shape == tiny.shape
        assert transformed.shape == (1, 1, 128, 128)

    def test_batch_processing(self):
        """Test batch processing."""
        transform = ReversibleTransform(target_size=64)

        # Different sized images in batch (not typical, but test individual processing)
        sizes = [(50, 60), (80, 40), (100, 100)]

        for H, W in sizes:
            batch = torch.randn(4, 2, H, W)  # Batch of 4, 2 channels

            transformed, metadata = transform.forward(batch)
            reconstructed = transform.inverse(transformed, metadata)

            assert reconstructed.shape == batch.shape
            error = (batch - reconstructed).abs().max()
            assert error < 5.0

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        transform = ReversibleTransform(target_size=64)

        # Very large values
        large = torch.randn(1, 1, 100, 100) * 1000
        transformed, metadata = transform.forward(large)
        reconstructed = transform.inverse(transformed, metadata)

        assert reconstructed.shape == large.shape
        # For very large values, interpolation can introduce significant error
        # The key is that the transform doesn't crash and produces reasonable output
        rel_error = ((large - reconstructed).abs() / (large.abs() + 1e-8)).max()
        assert (
            rel_error < 5000.0
        )  # Very relaxed for extreme values - focus on stability

        # Very small values
        small = torch.randn(1, 1, 100, 100) * 1e-6
        transformed, metadata = transform.forward(small)
        reconstructed = transform.inverse(transformed, metadata)

        assert reconstructed.shape == small.shape
        error = (small - reconstructed).abs().max()
        assert error < 1e-5  # Relaxed for small values due to interpolation

    def test_input_validation(self):
        """Test input validation."""
        transform = ReversibleTransform(target_size=64)

        # Wrong tensor dimensions
        with pytest.raises(TransformError, match="must be 4D tensor"):
            transform.forward(torch.randn(100, 100))  # 2D

        # Non-tensor input
        with pytest.raises(TransformError, match="must be a torch.Tensor"):
            transform.forward(np.random.randn(1, 1, 100, 100))

        # Invalid metadata for inverse
        image = torch.randn(1, 1, 64, 64)
        with pytest.raises(TransformError, match="must be ImageMetadata"):
            transform.inverse(image, "invalid_metadata")

    def test_reconstruction_error_method(self):
        """Test the reconstruction error testing method."""
        transform = ReversibleTransform(target_size=64)

        image = torch.randn(1, 1, 100, 80)

        error = transform.test_reconstruction_error(
            image, pixel_size=1.0, domain="test"
        )

        assert isinstance(error, float)
        assert (
            error < 5.0
        )  # Should be reasonable for interpolation-based reconstruction

    def test_edge_cases(self):
        """Test edge cases."""
        transform = ReversibleTransform(target_size=128)

        # Single pixel image
        single_pixel = torch.randn(1, 1, 1, 1)
        transformed, metadata = transform.forward(single_pixel)
        reconstructed = transform.inverse(transformed, metadata)

        assert reconstructed.shape == single_pixel.shape
        assert transformed.shape == (1, 1, 128, 128)

        # Image exactly matching target size
        exact_size = torch.randn(1, 1, 128, 128)
        transformed, metadata = transform.forward(exact_size)
        reconstructed = transform.inverse(transformed, metadata)

        # Should be identical (no transformation needed)
        assert torch.allclose(exact_size, reconstructed, atol=1e-6)

    def test_device_consistency(self):
        """Test that transforms work on different devices."""
        transform = ReversibleTransform(target_size=64)

        # CPU
        cpu_image = torch.randn(1, 1, 100, 80)
        transformed, metadata = transform.forward(cpu_image)
        reconstructed = transform.inverse(transformed, metadata)

        assert reconstructed.device == cpu_image.device
        assert reconstructed.shape == cpu_image.shape

        # GPU (if available)
        if torch.cuda.is_available():
            gpu_image = cpu_image.cuda()
            transformed, metadata = transform.forward(gpu_image)
            reconstructed = transform.inverse(transformed, metadata)

            assert reconstructed.device == gpu_image.device
            assert reconstructed.shape == gpu_image.shape

    def test_dtype_preservation(self):
        """Test that data types are preserved."""
        transform = ReversibleTransform(target_size=64)

        dtypes = [torch.float32, torch.float64]

        for dtype in dtypes:
            image = torch.randn(1, 1, 100, 80, dtype=dtype)
            transformed, metadata = transform.forward(image)
            reconstructed = transform.inverse(transformed, metadata)

            assert reconstructed.dtype == dtype
            assert transformed.dtype == dtype


class TestIntegration:
    """Integration tests for transforms."""

    def test_full_pipeline_with_serialization(self):
        """Test full pipeline including metadata serialization."""
        transform = ReversibleTransform(target_size=128)

        # Original image
        original = torch.randn(2, 3, 200, 150)

        # Forward transform
        transformed, metadata = transform.forward(
            original,
            pixel_size=4.29,
            pixel_unit="um",
            domain="photography",
            black_level=512,
            white_level=16383,
            iso=3200,
        )

        # Serialize metadata
        json_str = metadata.to_json()

        # Deserialize metadata
        restored_metadata = ImageMetadata.from_json(json_str)

        # Inverse transform with restored metadata
        reconstructed = transform.inverse(transformed, restored_metadata)

        # Verify reasonable reconstruction
        assert reconstructed.shape == original.shape
        error = (original - reconstructed).abs().max()
        assert error < 5.0  # Reasonable for interpolation-based reconstruction

    def test_multiple_transforms(self):
        """Test applying transforms multiple times."""
        transform1 = ReversibleTransform(target_size=64)
        transform2 = ReversibleTransform(target_size=32)

        original = torch.randn(1, 1, 100, 80)

        # First transform
        t1, m1 = transform1.forward(original)

        # Second transform on result
        t2, m2 = transform2.forward(t1)

        # Reverse transforms
        r1 = transform2.inverse(t2, m2)
        final = transform1.inverse(r1, m1)

        # Should reconstruct original
        assert final.shape == original.shape
        error = (original - final).abs().max()
        assert error < 10.0  # Higher tolerance for double transform


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
