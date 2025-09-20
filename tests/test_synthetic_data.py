"""
Test synthetic data generation for physics validation.

These tests verify that the synthetic data generator produces
statistically correct Poisson-Gaussian noise patterns.
"""

import json

# Import the synthetic data generator
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))
from scripts.generate_synthetic_data import SyntheticConfig, SyntheticDataGenerator


class TestSyntheticConfig:
    """Test synthetic data configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SyntheticConfig()

        assert config.image_size == 128
        assert config.num_images == 100
        assert len(config.photon_levels) == 8
        assert len(config.read_noise_levels) == 5
        assert len(config.pattern_types) == 5
        assert config.background_level == 10.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = SyntheticConfig(
            image_size=64,
            photon_levels=[10.0, 100.0],
            read_noise_levels=[1.0, 5.0],
            pattern_types=["constant", "gradient"],
        )

        assert config.image_size == 64
        assert config.photon_levels == [10.0, 100.0]
        assert config.read_noise_levels == [1.0, 5.0]
        assert config.pattern_types == ["constant", "gradient"]


class TestSyntheticDataGenerator:
    """Test synthetic data generator functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SyntheticConfig(
            image_size=32,  # Small for fast testing
            photon_levels=[10.0, 100.0],
            read_noise_levels=[1.0, 5.0],
            pattern_types=["constant", "gradient"],
            save_plots=False,  # Skip plots for testing
        )

    @pytest.fixture
    def generator(self, config):
        """Create generator with temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            yield SyntheticDataGenerator(config)

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.config.image_size == 32
        assert generator.output_dir.exists()

    def test_generate_constant_pattern(self, generator):
        """Test constant pattern generation."""
        pattern = generator.generate_pattern("constant", 32)

        assert pattern.shape == (32, 32)
        assert pattern.dtype == np.float32
        assert np.allclose(pattern, 0.5)
        assert 0 <= pattern.min() <= pattern.max() <= 1

    def test_generate_gradient_pattern(self, generator):
        """Test gradient pattern generation."""
        pattern = generator.generate_pattern("gradient", 32)

        assert pattern.shape == (32, 32)
        assert pattern.dtype == np.float32
        assert 0 <= pattern.min() <= pattern.max() <= 1

        # Check gradient properties
        assert pattern[0, 0] < pattern[-1, -1]  # Should increase along diagonal

    def test_generate_checkerboard_pattern(self, generator):
        """Test checkerboard pattern generation."""
        pattern = generator.generate_pattern("checkerboard", 32)

        assert pattern.shape == (32, 32)
        assert pattern.dtype == np.float32
        assert 0 <= pattern.min() <= pattern.max() <= 1

        # Should have only two distinct values (approximately)
        unique_vals = np.unique(np.round(pattern, 1))
        assert len(unique_vals) <= 3  # Allow for some rounding

    def test_generate_gaussian_spots_pattern(self, generator):
        """Test Gaussian spots pattern generation."""
        # Fix random seed for reproducible test
        np.random.seed(42)
        pattern = generator.generate_pattern("gaussian_spots", 32)

        assert pattern.shape == (32, 32)
        assert pattern.dtype == np.float32
        assert 0 <= pattern.min() <= pattern.max() <= 1

        # Should have variation (not constant)
        assert pattern.std() > 0.1

    def test_generate_natural_image_pattern(self, generator):
        """Test natural image pattern generation."""
        np.random.seed(42)
        pattern = generator.generate_pattern("natural_image", 32)

        assert pattern.shape == (32, 32)
        assert pattern.dtype == np.float32
        assert 0 <= pattern.min() <= pattern.max() <= 1

        # Should have significant variation
        assert pattern.std() > 0.05

    def test_invalid_pattern_type(self, generator):
        """Test error handling for invalid pattern type."""
        with pytest.raises(ValueError, match="Unknown pattern type"):
            generator.generate_pattern("invalid_pattern", 32)


class TestPoissonGaussianNoise:
    """Test Poisson-Gaussian noise generation and statistics."""

    @pytest.fixture
    def generator(self):
        """Create minimal generator for noise testing."""
        config = SyntheticConfig(image_size=32, save_plots=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            yield SyntheticDataGenerator(config)

    def test_poisson_only_noise(self, generator):
        """Test pure Poisson noise (no read noise)."""
        np.random.seed(42)

        # Constant image
        clean = np.ones((32, 32), dtype=np.float32) * 0.5
        photon_level = 100.0
        read_noise = 0.0

        noisy, params = generator.add_poisson_gaussian_noise(
            clean, photon_level, read_noise
        )

        # Check output properties
        assert noisy.shape == clean.shape
        assert noisy.dtype == np.float32
        assert np.all(noisy >= 0)  # Non-negative

        # Check statistics
        expected_mean = photon_level * 0.5
        expected_var = expected_mean  # Poisson property

        assert abs(params["mean_photons"] - expected_mean) < 1.0
        assert abs(params["theoretical_variance"] - expected_var) < 1.0
        assert params["read_noise"] == 0.0
        assert np.isfinite(params["snr_db"])

    def test_gaussian_read_noise(self, generator):
        """Test Gaussian read noise addition."""
        np.random.seed(42)

        clean = np.ones((32, 32), dtype=np.float32) * 0.5
        photon_level = 100.0
        read_noise = 5.0

        noisy, params = generator.add_poisson_gaussian_noise(
            clean, photon_level, read_noise
        )

        # Check that read noise increases variance
        expected_var = photon_level * 0.5 + read_noise**2

        assert params["theoretical_variance"] > photon_level * 0.5
        assert abs(params["theoretical_variance"] - expected_var) < 1.0
        assert params["read_noise"] == read_noise

    def test_background_addition(self, generator):
        """Test background offset."""
        np.random.seed(42)

        clean = np.zeros((32, 32), dtype=np.float32)  # Zero signal
        photon_level = 0.0
        read_noise = 1.0
        background = 10.0

        noisy, params = generator.add_poisson_gaussian_noise(
            clean, photon_level, read_noise, background
        )

        # With zero signal, mean should be close to background
        assert abs(np.mean(noisy) - background) < 2.0  # Allow for noise
        assert params["background"] == background

    def test_low_photon_regime(self, generator):
        """Test very low photon counts (<1 per pixel)."""
        np.random.seed(42)

        clean = np.ones((32, 32), dtype=np.float32) * 0.1
        photon_level = 1.0  # Very low
        read_noise = 0.5

        noisy, params = generator.add_poisson_gaussian_noise(
            clean, photon_level, read_noise
        )

        # Should still produce valid results
        assert np.all(noisy >= 0)
        assert np.isfinite(params["snr_db"])
        assert params["mean_photons"] < 1.0

    def test_high_photon_regime(self, generator):
        """Test high photon counts (>1000 per pixel)."""
        np.random.seed(42)

        clean = np.ones((32, 32), dtype=np.float32) * 0.8
        photon_level = 5000.0  # High
        read_noise = 2.0

        noisy, params = generator.add_poisson_gaussian_noise(
            clean, photon_level, read_noise
        )

        # Should have high SNR (adjust threshold based on actual calculation)
        assert params["snr_db"] > 15.0  # High SNR
        assert params["mean_photons"] > 1000.0

        # In high photon regime, should approximate Gaussian
        # (Poisson approaches Gaussian for large λ)
        expected_mean = photon_level * 0.8
        assert abs(params["mean_photons"] - expected_mean) < 50.0


class TestStatisticalValidation:
    """Test statistical properties of generated data."""

    def test_poisson_variance_property(self):
        """Test that Poisson noise has variance equal to mean."""
        np.random.seed(42)

        # Generate large sample for statistical test
        clean = np.ones((100, 100), dtype=np.float32) * 0.5
        photon_level = 50.0

        config = SyntheticConfig(save_plots=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            generator = SyntheticDataGenerator(config)

            # Generate multiple samples
            samples = []
            for _ in range(10):
                noisy, _ = generator.add_poisson_gaussian_noise(
                    clean, photon_level, read_noise=0.0
                )
                samples.append(noisy)

            # Compute empirical statistics
            samples = np.array(samples)
            empirical_mean = np.mean(samples)
            empirical_var = np.var(samples)

            # For Poisson, variance should equal mean
            expected_mean = photon_level * 0.5

            # Allow some tolerance due to finite sampling
            assert abs(empirical_mean - expected_mean) < 2.0
            assert abs(empirical_var - empirical_mean) < 5.0

    def test_chi_squared_statistic(self):
        """Test chi-squared statistic for Poisson-Gaussian noise."""
        np.random.seed(42)

        config = SyntheticConfig(save_plots=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            generator = SyntheticDataGenerator(config)

            # Generate data with known parameters
            clean = np.ones((64, 64), dtype=np.float32) * 0.4
            photon_level = 100.0
            read_noise = 3.0

            noisy, params = generator.add_poisson_gaussian_noise(
                clean, photon_level, read_noise
            )

            # Compute chi-squared statistic manually
            lambda_e = photon_level * clean
            variance = lambda_e + read_noise**2
            chi2 = ((noisy - lambda_e) ** 2) / variance
            chi2_per_pixel = np.mean(chi2)

            # For correct noise model, chi2 per pixel should be ~1
            # Allow reasonable tolerance
            assert 0.5 < chi2_per_pixel < 2.0

    def test_snr_calculation(self):
        """Test SNR calculation accuracy."""
        config = SyntheticConfig(save_plots=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            generator = SyntheticDataGenerator(config)

            clean = np.ones((32, 32), dtype=np.float32) * 0.6
            photon_level = 200.0
            read_noise = 4.0

            _, params = generator.add_poisson_gaussian_noise(
                clean, photon_level, read_noise
            )

            # Calculate expected SNR
            signal = photon_level * 0.6
            noise_var = signal + read_noise**2
            expected_snr = 10 * np.log10(signal / np.sqrt(noise_var))

            assert abs(params["snr_db"] - expected_snr) < 1.0


class TestDatasetGeneration:
    """Test full dataset generation pipeline."""

    def test_generate_validation_set(self):
        """Test validation set generation."""
        config = SyntheticConfig(
            image_size=16,  # Very small for fast test
            photon_levels=[10.0, 100.0],
            read_noise_levels=[1.0, 2.0],
            pattern_types=["constant", "gradient"],
            save_plots=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            generator = SyntheticDataGenerator(config)

            results = generator.generate_validation_set()

            # Check structure
            assert "images" in results
            assert "metadata" in results
            assert "statistics" in results

            # Check counts (2 patterns × 2 photon levels × 2 read noise levels = 8)
            expected_count = 2 * 2 * 2
            assert len(results["images"]) == expected_count
            assert len(results["metadata"]) == expected_count
            assert len(results["statistics"]) == expected_count

            # Check first image
            first_image = results["images"][0]
            assert "clean" in first_image
            assert "noisy" in first_image
            assert "pattern_type" in first_image
            assert "noise_params" in first_image

            # Check image properties
            assert first_image["clean"].shape == (16, 16)
            assert first_image["noisy"].shape == (16, 16)
            assert first_image["clean"].dtype == np.float32
            assert first_image["noisy"].dtype == np.float32

    def test_save_dataset(self):
        """Test dataset saving functionality."""
        config = SyntheticConfig(
            image_size=8,
            photon_levels=[50.0],
            read_noise_levels=[2.0],
            pattern_types=["constant"],
            save_plots=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            generator = SyntheticDataGenerator(config)

            results = generator.generate_validation_set()
            generator.save_dataset(results)

            # Check directory structure
            output_dir = Path(temp_dir)
            assert (output_dir / "images").exists()
            assert (output_dir / "metadata").exists()
            assert (output_dir / "dataset_summary.json").exists()

            # Check files were created
            images_dir = output_dir / "images"
            metadata_dir = output_dir / "metadata"

            npz_files = list(images_dir.glob("*.npz"))
            json_files = list(metadata_dir.glob("*.json"))

            assert len(npz_files) == 1
            assert len(json_files) == 1

            # Check file contents
            data = np.load(npz_files[0])
            assert "clean" in data
            assert "noisy" in data
            assert data["clean"].shape == (8, 8)

            with open(json_files[0]) as f:
                metadata = json.load(f)
            assert "pattern_type" in metadata
            assert "photon_level" in metadata
            assert "snr_db" in metadata


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_zero_photon_level(self):
        """Test handling of zero photon level."""
        config = SyntheticConfig(save_plots=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            generator = SyntheticDataGenerator(config)

            clean = np.ones((16, 16), dtype=np.float32) * 0.5

            # Should handle zero photons gracefully
            noisy, params = generator.add_poisson_gaussian_noise(
                clean, photon_level=0.0, read_noise=1.0
            )

            assert np.all(noisy >= 0)
            assert params["photon_level"] == 0.0
            assert np.isfinite(params["snr_db"]) or params["snr_db"] == float("-inf")

    def test_negative_values_clamped(self):
        """Test that negative values are clamped to zero."""
        config = SyntheticConfig(save_plots=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            generator = SyntheticDataGenerator(config)

            # Very low signal with high read noise (can produce negative values)
            clean = np.ones((16, 16), dtype=np.float32) * 0.01

            noisy, _ = generator.add_poisson_gaussian_noise(
                clean, photon_level=1.0, read_noise=10.0
            )

            # Should be clamped to non-negative
            assert np.all(noisy >= 0)


if __name__ == "__main__":
    pytest.main([__file__])
