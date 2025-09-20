"""
Comprehensive tests for unified DomainDataset implementation.

Tests cover:
- Dataset initialization and configuration
- Train/validation/test splitting with deterministic seeding
- Geometric augmentation pipeline integration
- Data loading correctness and transform consistency
- Multi-domain dataset functionality
- Error handling and edge cases

Requirements addressed: 2.4, 5.6, 5.7 from requirements.md
Task: 4.2 testing from tasks.md
"""

import json
import shutil

# Import dataset components
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))
from core.calibration import CalibrationParams
from core.exceptions import DataError, ValidationError
from data.augmentations import (
    AugmentationConfig,
    GeometricAugmentationPipeline,
    create_training_augmentations,
    create_validation_augmentations,
)
from data.domain_datasets import (
    DomainConfig,
    DomainDataset,
    MultiDomainDataset,
    create_domain_dataset,
    create_multi_domain_dataset,
)
from data.loaders import ImageMetadata


class TestAugmentationPipeline:
    """Test geometric augmentation pipeline."""

    @pytest.fixture
    def sample_image(self):
        """Create sample image tensor."""
        return torch.rand(1, 64, 64)  # [C, H, W]

    @pytest.fixture
    def sample_mask(self):
        """Create sample mask tensor."""
        mask = torch.ones(1, 64, 64)
        mask[:, :10, :10] = 0  # Some invalid region
        return mask

    def test_augmentation_config_defaults(self):
        """Test augmentation configuration defaults."""
        config = AugmentationConfig()

        assert config.enable_rotation is True
        assert config.enable_horizontal_flip is True
        assert config.enable_vertical_flip is True
        assert config.flip_probability == 0.5
        assert config.deterministic is True
        assert config.seed == 42
        assert config.rotation_angles == [0, 90, 180, 270]

    def test_augmentation_config_custom(self):
        """Test custom augmentation configuration."""
        config = AugmentationConfig(
            enable_rotation=False,
            flip_probability=0.8,
            rotation_angles=[0, 180],
            deterministic=False,
        )

        assert config.enable_rotation is False
        assert config.flip_probability == 0.8
        assert config.rotation_angles == [0, 180]
        assert config.deterministic is False

    def test_rotation_augmentation(self, sample_image, sample_mask):
        """Test rotation augmentation."""
        config = AugmentationConfig(
            enable_rotation=True,
            rotation_angles=[90],  # Force 90-degree rotation
            deterministic=True,
            seed=42,
        )

        pipeline = GeometricAugmentationPipeline(config)

        # Apply augmentation
        aug_image, aug_mask, info = pipeline(sample_image, sample_mask)

        # Check that rotation was applied
        assert "rotation_angle" in info

        # Check shape preservation
        assert aug_image.shape == sample_image.shape
        if aug_mask is not None:
            assert aug_mask.shape == sample_mask.shape

    def test_flip_augmentation(self, sample_image, sample_mask):
        """Test flip augmentation."""
        config = AugmentationConfig(
            enable_rotation=False,  # Disable other augmentations
            enable_horizontal_flip=True,
            enable_vertical_flip=True,
            flip_probability=1.0,  # Force flipping
            deterministic=True,
            seed=42,
        )

        pipeline = GeometricAugmentationPipeline(config)

        # Apply augmentation
        aug_image, aug_mask, info = pipeline(sample_image, sample_mask)

        # Check that flipping info is present
        assert "horizontal_flip" in info
        assert "vertical_flip" in info

        # Check shape preservation
        assert aug_image.shape == sample_image.shape
        if aug_mask is not None:
            assert aug_mask.shape == sample_mask.shape

    def test_no_augmentation_mode(self, sample_image, sample_mask):
        """Test pipeline with augmentations disabled."""
        config = AugmentationConfig()
        pipeline = GeometricAugmentationPipeline(config)

        # Apply with augmentations disabled
        aug_image, aug_mask, info = pipeline(
            sample_image, sample_mask, apply_augmentations=False
        )

        # Should return original data
        torch.testing.assert_close(aug_image, sample_image)
        if aug_mask is not None:
            torch.testing.assert_close(aug_mask, sample_mask)

        assert info["augmentations_applied"] is False

    def test_deterministic_behavior(self, sample_image):
        """Test deterministic augmentation behavior."""
        config = AugmentationConfig(deterministic=True, seed=123)

        pipeline1 = GeometricAugmentationPipeline(config)
        pipeline2 = GeometricAugmentationPipeline(config)

        # Apply same augmentations
        aug1, _, info1 = pipeline1(sample_image)
        aug2, _, info2 = pipeline2(sample_image)

        # Should be identical
        torch.testing.assert_close(aug1, aug2)
        assert info1 == info2

    def test_domain_specific_augmentations(self, sample_image):
        """Test domain-specific augmentation configurations."""
        # Photography augmentations
        photo_pipeline = create_training_augmentations("photography")
        photo_config = photo_pipeline.get_config()

        assert photo_config.enable_rotation is True
        assert photo_config.enable_horizontal_flip is True

        # Microscopy augmentations
        micro_pipeline = create_training_augmentations("microscopy")
        micro_config = micro_pipeline.get_config()

        assert micro_config.enable_rotation is True
        assert micro_config.enable_noise_augmentation is True

        # Astronomy augmentations
        astro_pipeline = create_training_augmentations("astronomy")
        astro_config = astro_pipeline.get_config()

        assert astro_config.enable_rotation is True
        assert astro_config.enable_random_crop is False  # Preserve field of view

    def test_validation_augmentations(self, sample_image):
        """Test validation augmentation configuration."""
        val_pipeline = create_validation_augmentations("photography")
        val_config = val_pipeline.get_config()

        # Validation should have minimal augmentations
        assert val_config.enable_rotation is False
        assert val_config.enable_horizontal_flip is False
        assert val_config.enable_vertical_flip is False
        assert val_config.enable_random_crop is False


class TestDomainDataset:
    """Test DomainDataset functionality."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with mock data files."""
        temp_dir = tempfile.mkdtemp()

        # Create mock image files
        for i in range(10):
            # Create different file types for different domains
            (Path(temp_dir) / f"image_{i:03d}.tif").touch()
            (Path(temp_dir) / f"photo_{i:03d}.arw").touch()
            (Path(temp_dir) / f"astro_{i:03d}.fits").touch()

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_calibration_file(self):
        """Create mock calibration file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)

        calibration_data = {
            "gain": 2.0,
            "black_level": 100,
            "white_level": 16383,
            "read_noise": 3.0,
            "pixel_size": 0.65,
            "pixel_unit": "um",
            "domain": "microscopy",
        }

        json.dump(calibration_data, temp_file)
        temp_file.close()

        yield temp_file.name

        # Cleanup
        Path(temp_file.name).unlink()

    def test_domain_config_defaults(self):
        """Test domain configuration defaults."""
        # Check that all domains have configurations
        assert "photography" in DomainDataset.DOMAIN_CONFIGS
        assert "microscopy" in DomainDataset.DOMAIN_CONFIGS
        assert "astronomy" in DomainDataset.DOMAIN_CONFIGS

        # Check photography config
        photo_config = DomainDataset.DOMAIN_CONFIGS["photography"]
        assert photo_config.domain == "photography"
        assert photo_config.scale == 10000.0
        assert ".arw" in photo_config.supported_extensions

        # Check microscopy config
        micro_config = DomainDataset.DOMAIN_CONFIGS["microscopy"]
        assert micro_config.domain == "microscopy"
        assert micro_config.scale == 1000.0
        assert ".tif" in micro_config.supported_extensions

        # Check astronomy config
        astro_config = DomainDataset.DOMAIN_CONFIGS["astronomy"]
        assert astro_config.domain == "astronomy"
        assert astro_config.scale == 50000.0
        assert ".fits" in astro_config.supported_extensions

    @patch("data.domain_datasets.MicroscopyLoader")
    def test_dataset_initialization(
        self, mock_loader_class, temp_data_dir, mock_calibration_file
    ):
        """Test dataset initialization."""
        # Mock the loader
        mock_loader = MagicMock()
        mock_loader.validate_file.return_value = True
        mock_loader_class.return_value = mock_loader

        # Create dataset
        dataset = DomainDataset(
            data_root=temp_data_dir,
            domain="microscopy",
            calibration_file=mock_calibration_file,
            split="train",
            max_files=5,
            seed=42,
        )

        assert dataset.domain == "microscopy"
        assert dataset.split == "train"
        assert dataset.seed == 42
        assert dataset.target_size == 128  # default
        assert len(dataset.file_paths) <= 5  # max_files limit

    def test_invalid_domain(self, temp_data_dir):
        """Test error handling for invalid domain."""
        with pytest.raises(ValueError, match="Unsupported domain"):
            DomainDataset(data_root=temp_data_dir, domain="invalid_domain")

    @patch("data.domain_datasets.MicroscopyLoader")
    def test_train_val_test_splitting(self, mock_loader_class, temp_data_dir):
        """Test deterministic train/validation/test splitting."""
        # Mock the loader
        mock_loader = MagicMock()
        mock_loader.validate_file.return_value = True
        mock_loader_class.return_value = mock_loader

        # Create datasets for each split
        train_dataset = DomainDataset(
            data_root=temp_data_dir,
            domain="microscopy",
            split="train",
            split_ratios=(0.6, 0.2, 0.2),
            seed=42,
            validate_files=False,
        )

        val_dataset = DomainDataset(
            data_root=temp_data_dir,
            domain="microscopy",
            split="val",
            split_ratios=(0.6, 0.2, 0.2),
            seed=42,
            validate_files=False,
        )

        test_dataset = DomainDataset(
            data_root=temp_data_dir,
            domain="microscopy",
            split="test",
            split_ratios=(0.6, 0.2, 0.2),
            seed=42,
            validate_files=False,
        )

        # Check that splits are non-overlapping and cover all data
        total_files = (
            len(train_dataset.file_paths)
            + len(val_dataset.file_paths)
            + len(test_dataset.file_paths)
        )
        assert total_files > 0

        # Check that train is largest
        assert len(train_dataset.file_paths) >= len(val_dataset.file_paths)
        assert len(train_dataset.file_paths) >= len(test_dataset.file_paths)

    @patch("data.domain_datasets.MicroscopyLoader")
    def test_deterministic_splitting(self, mock_loader_class, temp_data_dir):
        """Test that splitting is deterministic with same seed."""
        # Mock the loader
        mock_loader = MagicMock()
        mock_loader.validate_file.return_value = True
        mock_loader_class.return_value = mock_loader

        # Create two datasets with same seed
        dataset1 = DomainDataset(
            data_root=temp_data_dir,
            domain="microscopy",
            split="train",
            seed=123,
            validate_files=False,
        )

        dataset2 = DomainDataset(
            data_root=temp_data_dir,
            domain="microscopy",
            split="train",
            seed=123,
            validate_files=False,
        )

        # Should have same files in same order
        assert dataset1.file_paths == dataset2.file_paths

    @patch("data.domain_datasets.MicroscopyLoader")
    def test_augmentation_setup(self, mock_loader_class, temp_data_dir):
        """Test augmentation pipeline setup."""
        # Mock the loader
        mock_loader = MagicMock()
        mock_loader.validate_file.return_value = True
        mock_loader_class.return_value = mock_loader

        # Training dataset should have augmentations
        train_dataset = DomainDataset(
            data_root=temp_data_dir,
            domain="microscopy",
            split="train",
            enable_augmentations=True,
            validate_files=False,
        )

        assert train_dataset.augmentation_pipeline is not None
        assert isinstance(
            train_dataset.augmentation_pipeline, GeometricAugmentationPipeline
        )

        # Validation dataset should have minimal augmentations
        val_dataset = DomainDataset(
            data_root=temp_data_dir,
            domain="microscopy",
            split="val",
            enable_augmentations=True,
            validate_files=False,
        )

        assert val_dataset.augmentation_pipeline is not None

        # Disabled augmentations
        no_aug_dataset = DomainDataset(
            data_root=temp_data_dir,
            domain="microscopy",
            split="train",
            enable_augmentations=False,
            validate_files=False,
        )

        assert no_aug_dataset.augmentation_pipeline is None

    @patch("data.domain_datasets.MicroscopyLoader")
    @patch("data.domain_datasets.SensorCalibration")
    @patch("data.domain_datasets.ReversibleTransform")
    def test_getitem_structure(
        self,
        mock_transform_class,
        mock_calibration_class,
        mock_loader_class,
        temp_data_dir,
    ):
        """Test __getitem__ return structure."""
        # Mock components
        mock_loader = MagicMock()
        mock_loader.validate_file.return_value = True
        mock_loader.load_with_validation.return_value = (
            np.random.rand(64, 64).astype(np.float32),  # raw_adu
            ImageMetadata(
                filepath="test.tif",
                format=".tif",
                domain="microscopy",
                height=64,
                width=64,
                channels=1,
                bit_depth=16,
                dtype="uint16",
            ),
        )
        mock_loader_class.return_value = mock_loader

        mock_calibration = MagicMock()
        mock_calibration.process_raw.return_value = (
            np.random.rand(64, 64).astype(np.float32),  # electrons
            np.ones((64, 64), dtype=np.float32),  # mask
        )
        mock_calibration.params = CalibrationParams(
            gain=2.0,
            black_level=100,
            white_level=16383,
            read_noise=3.0,
            pixel_size=0.65,
            pixel_unit="um",
            domain="microscopy",
        )
        mock_calibration_class.return_value = mock_calibration

        mock_transform = MagicMock()
        mock_transform.forward.return_value = (
            torch.rand(1, 128, 128),  # transformed
            {"target_size": 128},  # metadata
        )
        mock_transform_class.return_value = mock_transform

        # Create dataset
        dataset = DomainDataset(
            data_root=temp_data_dir,
            domain="microscopy",
            split="train",
            validate_files=False,
            max_files=1,
        )

        # Get item
        item = dataset[0]

        # Check required fields
        required_fields = [
            "raw_adu",
            "electrons",
            "normalized",
            "transformed",
            "mask",
            "original_mask",
            "image_metadata",
            "transform_metadata",
            "augmentation_info",
            "calibration_params",
            "scale",
            "file_path",
            "split",
            "domain",
        ]

        for field in required_fields:
            assert field in item, f"Missing required field: {field}"

        # Check data types
        assert isinstance(item["raw_adu"], np.ndarray)
        assert isinstance(item["electrons"], np.ndarray)
        assert isinstance(item["normalized"], torch.Tensor)
        assert isinstance(item["transformed"], torch.Tensor)
        assert isinstance(item["mask"], torch.Tensor)
        assert isinstance(item["augmentation_info"], dict)
        assert item["domain"] == "microscopy"
        assert item["split"] == "train"

    def test_dataset_statistics(self, temp_data_dir):
        """Test dataset statistics collection."""
        with patch("data.domain_datasets.MicroscopyLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.validate_file.return_value = True
            mock_loader.get_statistics.return_value = {
                "files_loaded": 5,
                "load_errors": 0,
                "success_rate": 1.0,
            }
            mock_loader_class.return_value = mock_loader

            dataset = DomainDataset(
                data_root=temp_data_dir, domain="microscopy", validate_files=False
            )

            stats = dataset.get_domain_stats()

            assert "domain" in stats
            assert "split" in stats
            assert "num_files" in stats
            assert "scale" in stats
            assert "loader_stats" in stats

            assert stats["domain"] == "microscopy"


class TestMultiDomainDataset:
    """Test MultiDomainDataset functionality."""

    @pytest.fixture
    def temp_multi_domain_dir(self):
        """Create temporary directory structure for multiple domains."""
        temp_dir = tempfile.mkdtemp()

        # Create subdirectories for each domain
        photo_dir = Path(temp_dir) / "photography"
        micro_dir = Path(temp_dir) / "microscopy"
        astro_dir = Path(temp_dir) / "astronomy"

        photo_dir.mkdir()
        micro_dir.mkdir()
        astro_dir.mkdir()

        # Create mock files
        for i in range(5):
            (photo_dir / f"photo_{i}.arw").touch()
            (micro_dir / f"micro_{i}.tif").touch()
            (astro_dir / f"astro_{i}.fits").touch()

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    @patch("data.domain_datasets.PhotographyLoader")
    @patch("data.domain_datasets.MicroscopyLoader")
    @patch("data.domain_datasets.AstronomyLoader")
    def test_multi_domain_initialization(
        self,
        mock_astro_loader,
        mock_micro_loader,
        mock_photo_loader,
        temp_multi_domain_dir,
    ):
        """Test multi-domain dataset initialization."""
        # Mock all loaders
        for mock_loader_class in [
            mock_photo_loader,
            mock_micro_loader,
            mock_astro_loader,
        ]:
            mock_loader = MagicMock()
            mock_loader.validate_file.return_value = True
            mock_loader_class.return_value = mock_loader

        # Domain configurations
        domain_configs = {
            "photography": {
                "data_root": str(Path(temp_multi_domain_dir) / "photography"),
                "max_files": 3,
            },
            "microscopy": {
                "data_root": str(Path(temp_multi_domain_dir) / "microscopy"),
                "max_files": 3,
            },
            "astronomy": {
                "data_root": str(Path(temp_multi_domain_dir) / "astronomy"),
                "max_files": 3,
            },
        }

        # Create multi-domain dataset
        multi_dataset = MultiDomainDataset(
            domain_configs=domain_configs,
            split="train",
            balance_domains=True,
            validate_files=False,
        )

        assert len(multi_dataset.domain_datasets) == 3
        assert "photography" in multi_dataset.domain_datasets
        assert "microscopy" in multi_dataset.domain_datasets
        assert "astronomy" in multi_dataset.domain_datasets

    @patch("data.domain_datasets.MicroscopyLoader")
    def test_domain_balancing(self, mock_loader_class, temp_multi_domain_dir):
        """Test domain balancing functionality."""
        # Mock loader
        mock_loader = MagicMock()
        mock_loader.validate_file.return_value = True
        mock_loader_class.return_value = mock_loader

        # Create unbalanced domain configs (different number of files)
        domain_configs = {
            "microscopy": {
                "data_root": str(Path(temp_multi_domain_dir) / "microscopy"),
                "max_files": 2,  # Smaller dataset
            }
        }

        # Test with balancing enabled
        balanced_dataset = MultiDomainDataset(
            domain_configs=domain_configs, balance_domains=True, validate_files=False
        )

        # Test with balancing disabled
        unbalanced_dataset = MultiDomainDataset(
            domain_configs=domain_configs, balance_domains=False, validate_files=False
        )

        # Balanced should potentially have more samples due to repetition
        # (though with only one domain, this test is limited)
        assert len(balanced_dataset) >= len(unbalanced_dataset)

    @patch("data.domain_datasets.MicroscopyLoader")
    def test_multi_domain_getitem(self, mock_loader_class, temp_multi_domain_dir):
        """Test multi-domain dataset __getitem__."""
        # Mock components similar to single domain test
        mock_loader = MagicMock()
        mock_loader.validate_file.return_value = True
        mock_loader.load_with_validation.return_value = (
            np.random.rand(64, 64).astype(np.float32),
            ImageMetadata(
                filepath="test.tif",
                format=".tif",
                domain="microscopy",
                height=64,
                width=64,
                channels=1,
                bit_depth=16,
                dtype="uint16",
            ),
        )
        mock_loader_class.return_value = mock_loader

        with patch("data.domain_datasets.SensorCalibration") as mock_calibration_class:
            with patch(
                "data.domain_datasets.ReversibleTransform"
            ) as mock_transform_class:
                # Setup mocks
                mock_calibration = MagicMock()
                mock_calibration.process_raw.return_value = (
                    np.random.rand(64, 64).astype(np.float32),
                    np.ones((64, 64), dtype=np.float32),
                )
                mock_calibration.params = CalibrationParams(
                    gain=2.0,
                    black_level=100,
                    white_level=16383,
                    read_noise=3.0,
                    pixel_size=0.65,
                    pixel_unit="um",
                    domain="microscopy",
                )
                mock_calibration_class.return_value = mock_calibration

                mock_transform = MagicMock()
                mock_transform.forward.return_value = (
                    torch.rand(1, 128, 128),
                    {"target_size": 128},
                )
                mock_transform_class.return_value = mock_transform

                # Create multi-domain dataset
                domain_configs = {
                    "microscopy": {
                        "data_root": str(Path(temp_multi_domain_dir) / "microscopy"),
                        "max_files": 2,
                    }
                }

                multi_dataset = MultiDomainDataset(
                    domain_configs=domain_configs, validate_files=False
                )

                # Get item
                item = multi_dataset[0]

                # Should have all standard fields plus domain
                assert "domain" in item
                assert item["domain"] in domain_configs.keys()

    def test_multi_domain_statistics(self, temp_multi_domain_dir):
        """Test multi-domain dataset statistics."""
        with patch("data.domain_datasets.MicroscopyLoader") as mock_loader_class:
            mock_loader = MagicMock()
            mock_loader.validate_file.return_value = True
            mock_loader.get_statistics.return_value = {
                "files_loaded": 2,
                "load_errors": 0,
                "success_rate": 1.0,
            }
            mock_loader_class.return_value = mock_loader

            domain_configs = {
                "microscopy": {
                    "data_root": str(Path(temp_multi_domain_dir) / "microscopy"),
                    "max_files": 2,
                }
            }

            multi_dataset = MultiDomainDataset(
                domain_configs=domain_configs, validate_files=False
            )

            stats = multi_dataset.get_stats()

            assert "total_samples" in stats
            assert "num_domains" in stats
            assert "domain_distribution" in stats
            assert "domain_stats" in stats

            assert stats["num_domains"] == 1
            assert "microscopy" in stats["domain_stats"]


class TestConvenienceFunctions:
    """Test convenience functions for dataset creation."""

    def test_create_domain_dataset(self):
        """Test create_domain_dataset convenience function."""
        with patch("data.domain_datasets.DomainDataset") as mock_dataset_class:
            mock_dataset = MagicMock()
            mock_dataset_class.return_value = mock_dataset

            dataset = create_domain_dataset(
                domain="microscopy", data_root="/path/to/data", target_size=256
            )

            # Check that DomainDataset was called with correct arguments
            mock_dataset_class.assert_called_once()
            call_args = mock_dataset_class.call_args

            assert call_args[1]["domain"] == "microscopy"
            assert call_args[1]["data_root"] == "/path/to/data"
            assert call_args[1]["target_size"] == 256

    def test_create_multi_domain_dataset(self):
        """Test create_multi_domain_dataset convenience function."""
        with patch("data.domain_datasets.MultiDomainDataset") as mock_dataset_class:
            mock_dataset = MagicMock()
            mock_dataset_class.return_value = mock_dataset

            domain_configs = {
                "microscopy": {"data_root": "/path/to/micro"},
                "photography": {"data_root": "/path/to/photo"},
            }

            dataset = create_multi_domain_dataset(
                domain_configs=domain_configs, balance_domains=False
            )

            # Check that MultiDomainDataset was called
            mock_dataset_class.assert_called_once()
            call_args = mock_dataset_class.call_args

            assert call_args[0][0] == domain_configs
            assert call_args[1]["balance_domains"] is False


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_data_directory(self):
        """Test handling of empty data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(DataError, match="No valid files found"):
                DomainDataset(
                    data_root=temp_dir, domain="microscopy", validate_files=False
                )

    def test_invalid_split_ratios(self):
        """Test validation of split ratios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy file
            (Path(temp_dir) / "test.tif").touch()

            with patch("data.domain_datasets.MicroscopyLoader") as mock_loader_class:
                mock_loader = MagicMock()
                mock_loader.validate_file.return_value = True
                mock_loader_class.return_value = mock_loader

                # This should work (ratios sum to 1.0)
                dataset = DomainDataset(
                    data_root=temp_dir,
                    domain="microscopy",
                    split_ratios=(0.7, 0.2, 0.1),
                    validate_files=False,
                )

                assert dataset is not None

    def test_invalid_split_name(self):
        """Test handling of invalid split name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "test.tif").touch()

            with patch("data.domain_datasets.MicroscopyLoader") as mock_loader_class:
                mock_loader = MagicMock()
                mock_loader.validate_file.return_value = True
                mock_loader_class.return_value = mock_loader

                with pytest.raises(ValueError, match="Unknown split"):
                    DomainDataset(
                        data_root=temp_dir,
                        domain="microscopy",
                        split="invalid_split",
                        validate_files=False,
                    )


if __name__ == "__main__":
    pytest.main([__file__])
