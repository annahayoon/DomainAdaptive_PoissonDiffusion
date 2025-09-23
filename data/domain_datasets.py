"""
Domain-specific datasets with calibration integration.

This module provides unified dataset classes that integrate the format loaders
with the calibration system, creating a complete data pipeline for training
and evaluation.

Key features:
- Integration with format-specific loaders
- Automatic calibration application
- Domain-specific parameter handling
- Train/validation/test splitting
- Memory-efficient data loading

Requirements addressed: 2.4, 5.6, 5.7 from requirements.md
Task: 4.1 integration from tasks.md
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from core.calibration import CalibrationParams, SensorCalibration
from core.error_handlers import ErrorHandler, safe_operation
from core.exceptions import CalibrationError, DataError, ValidationError
from core.logging_config import get_logger
from core.transforms import ImageMetadata as TransformMetadata
from core.transforms import ReversibleTransform

from .augmentations import (
    GeometricAugmentationPipeline,
    create_training_augmentations,
    create_validation_augmentations,
)
from .loaders import (
    AstronomyLoader,
    FormatDetector,
    ImageMetadata as LoaderImageMetadata,
    MicroscopyLoader,
    PhotographyLoader,
)

logger = get_logger(__name__)


@dataclass
class DomainConfig:
    """Configuration for a specific imaging domain."""

    # Data paths
    data_root: str
    calibration_file: str

    # Domain properties
    domain: str
    scale: float  # Normalization scale (electrons)

    # Default sensor parameters (fallbacks)
    default_pixel_size: float
    default_pixel_unit: str
    default_black_level: float
    default_white_level: float
    default_gain: float
    default_read_noise: float

    # File handling
    supported_extensions: List[str]
    recursive_search: bool = True

    # Processing options
    apply_dark_correction: bool = False
    normalize_channels: bool = True


class DomainDataset(Dataset):
    """
    Dataset for a single imaging domain with calibration integration.

    This class provides a complete data loading pipeline that:
    1. Loads raw data using appropriate format loader
    2. Applies sensor calibration to convert ADU → electrons
    3. Normalizes data for model input
    4. Handles metadata and transformations
    """

    # Domain-specific configurations
    DOMAIN_CONFIGS = {
        "photography": DomainConfig(
            data_root="",  # To be set by user
            calibration_file="",  # To be set by user
            domain="photography",
            scale=10000.0,  # Typical scale for photography
            default_pixel_size=4.29,  # μm (Sony A7S)
            default_pixel_unit="um",
            default_black_level=512,
            default_white_level=16383,
            default_gain=1.0,
            default_read_noise=5.0,
            supported_extensions=[".arw", ".dng", ".nef", ".cr2"],
            apply_dark_correction=False,
            normalize_channels=True,
        ),
        "microscopy": DomainConfig(
            data_root="",
            calibration_file="",
            domain="microscopy",
            scale=1000.0,  # Typical scale for microscopy
            default_pixel_size=0.65,  # μm at 20x
            default_pixel_unit="um",
            default_black_level=100,
            default_white_level=65535,
            default_gain=2.0,
            default_read_noise=3.0,
            supported_extensions=[".tif", ".tiff"],
            apply_dark_correction=False,
            normalize_channels=True,
        ),
        "astronomy": DomainConfig(
            data_root="",
            calibration_file="",
            domain="astronomy",
            scale=50000.0,  # Typical scale for astronomy
            default_pixel_size=0.04,  # arcsec (Hubble WFC3)
            default_pixel_unit="arcsec",
            default_black_level=0,
            default_white_level=65535,
            default_gain=1.5,
            default_read_noise=2.0,
            supported_extensions=[".fits", ".fit"],
            apply_dark_correction=True,
            normalize_channels=False,
        ),
    }

    def __init__(
        self,
        data_root: Union[str, Path],
        domain: str,
        calibration_file: Optional[Union[str, Path]] = None,
        scale: Optional[float] = None,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        target_size: int = 128,
        max_files: Optional[int] = None,
        seed: int = 42,
        error_handler: Optional[ErrorHandler] = None,
        validate_files: bool = True,
        enable_augmentations: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize domain dataset.

        Args:
            data_root: Root directory containing image files
            domain: Domain name ('photography', 'microscopy', 'astronomy')
            calibration_file: Path to calibration JSON file
            scale: Normalization scale (electrons), uses default if None
            split: Dataset split ('train', 'val', 'test')
            split_ratios: (train, val, test) ratios, must sum to 1.0
            target_size: Target size for model input
            max_files: Maximum number of files to load (for testing)
            seed: Random seed for reproducible splits
            error_handler: Custom error handler
            validate_files: Whether to validate files on initialization
            enable_augmentations: Whether to enable geometric augmentations
            augmentation_config: Configuration overrides for augmentations
        """
        self.data_root = Path(data_root)
        self.domain = domain
        self.split = split
        self.split_ratios = split_ratios
        self.target_size = target_size
        self.max_files = max_files
        self.seed = seed
        self.error_handler = error_handler or ErrorHandler()
        self.validate_files = validate_files
        self.enable_augmentations = enable_augmentations

        # Get domain configuration
        if domain not in self.DOMAIN_CONFIGS:
            raise ValueError(f"Unsupported domain: {domain}")
        self.config = self.DOMAIN_CONFIGS[domain]

        # Override scale if provided
        self.scale = scale if scale is not None else self.config.scale

        # Initialize calibration
        self._setup_calibration(calibration_file)

        # Initialize loader
        self._setup_loader()

        # Initialize transforms
        self.transform = ReversibleTransform(target_size=target_size)

        # Initialize augmentations
        self._setup_augmentations(augmentation_config)

        # Find and split files
        self._find_and_split_files()

        logger.info(
            f"Initialized {domain} dataset: {len(self.file_paths)} files "
            f"for {split} split"
        )

    def _setup_calibration(self, calibration_file: Optional[Union[str, Path]]):
        """Setup sensor calibration."""
        if calibration_file is not None:
            # Load from file
            self.calibration = SensorCalibration(str(calibration_file))
        else:
            # Create default calibration
            logger.warning(
                f"No calibration file provided for {self.domain}, "
                "using default parameters"
            )

            default_params = CalibrationParams(
                gain=self.config.default_gain,
                black_level=self.config.default_black_level,
                white_level=self.config.default_white_level,
                read_noise=self.config.default_read_noise,
                pixel_size=self.config.default_pixel_size,
                pixel_unit=self.config.default_pixel_unit,
                domain=self.domain,
            )

            self.calibration = SensorCalibration(params=default_params)

    def _setup_loader(self):
        """Setup appropriate format loader."""
        if self.domain == "photography":
            self.loader = PhotographyLoader(
                demosaic=False,  # Keep raw Bayer for now
                error_handler=self.error_handler,
                validate_on_load=self.validate_files,
            )
        elif self.domain == "microscopy":
            self.loader = MicroscopyLoader(
                normalize_channels=self.config.normalize_channels,
                error_handler=self.error_handler,
                validate_on_load=self.validate_files,
            )
        elif self.domain == "astronomy":
            self.loader = AstronomyLoader(
                apply_scaling=True,
                error_handler=self.error_handler,
                validate_on_load=self.validate_files,
            )
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

    def _setup_augmentations(self, augmentation_config: Optional[Dict[str, Any]]):
        """Setup geometric augmentation pipeline."""
        if not self.enable_augmentations:
            self.augmentation_pipeline = None
            return

        # Create appropriate augmentation pipeline based on split
        if self.split == "train":
            self.augmentation_pipeline = create_training_augmentations(
                self.domain,
                deterministic=True,
                seed=self.seed,
                **(augmentation_config or {}),
            )
        else:
            # Validation and test use minimal augmentations
            self.augmentation_pipeline = create_validation_augmentations(
                self.domain, deterministic=True, seed=self.seed
            )

        logger.debug(f"Setup {self.split} augmentations for {self.domain}")

    def _find_and_split_files(self):
        """Find all valid files and create train/val/test splits."""
        # Find all files
        all_files = []

        for ext in self.config.supported_extensions:
            if self.config.recursive_search:
                pattern = f"**/*{ext}"
            else:
                pattern = f"*{ext}"

            files = list(self.data_root.glob(pattern))
            all_files.extend(files)

        # Filter valid files
        if self.validate_files:
            valid_files = []
            for file_path in all_files:
                if self.loader.validate_file(file_path):
                    valid_files.append(file_path)
                else:
                    logger.debug(f"Skipping invalid file: {file_path}")
            all_files = valid_files

        # Limit number of files if requested
        if self.max_files is not None:
            all_files = all_files[: self.max_files]

        if len(all_files) == 0:
            raise DataError(f"No valid files found in {self.data_root}")

        # Create reproducible splits
        random.seed(self.seed)
        random.shuffle(all_files)

        n_files = len(all_files)
        train_end = int(self.split_ratios[0] * n_files)
        val_end = int((self.split_ratios[0] + self.split_ratios[1]) * n_files)

        if self.split == "train":
            self.file_paths = all_files[:train_end]
        elif self.split == "val":
            self.file_paths = all_files[train_end:val_end]
        elif self.split == "test":
            self.file_paths = all_files[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        logger.info(
            f"Found {n_files} total files, using {len(self.file_paths)} "
            f"for {self.split} split"
        )

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.file_paths)

    @safe_operation("Dataset item loading")
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a single item.

        Returns:
            Dictionary containing:
            - 'raw_adu': Raw ADU data (original format)
            - 'electrons': Calibrated electron data
            - 'normalized': Normalized data [0,1] for model
            - 'transformed': Data transformed to target size
            - 'mask': Valid pixel mask
            - 'metadata': Complete metadata for reconstruction
            - 'calibration_params': Calibration parameters used
        """
        file_path = self.file_paths[idx]

        try:
            # Load raw data
            raw_adu, image_metadata = self.loader.load_with_validation(file_path)

            # Apply calibration to get electrons
            electrons, mask = self.calibration.process_raw(
                raw_adu,
                return_mask=True,
                apply_dark_correction=self.config.apply_dark_correction,
                exposure_time=image_metadata.exposure_time,
            )

            # Normalize to [0, 1] range
            normalized = electrons / self.scale
            normalized = np.clip(normalized, 0.0, 1.0)

            # Convert to tensors
            if isinstance(normalized, np.ndarray):
                normalized = torch.from_numpy(normalized).float()
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()

            # Add batch and channel dimensions if needed
            if normalized.ndim == 2:
                normalized = normalized.unsqueeze(0)  # Add channel dim
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            # Add batch dimension for transform
            normalized = normalized.unsqueeze(0)
            mask = mask.unsqueeze(0)

            # Apply reversible transform
            transformed, transform_metadata = self.transform.forward(
                normalized,
                pixel_size=self.calibration.params.pixel_size,
                pixel_unit=self.calibration.params.pixel_unit,
                domain=self.domain,
                black_level=self.calibration.params.black_level,
                white_level=self.calibration.params.white_level,
                iso=image_metadata.iso,
                exposure_time=image_metadata.exposure_time,
            )

            # Transform mask the same way
            mask_transformed, _ = self.transform.forward(
                mask,
                pixel_size=self.calibration.params.pixel_size,
                pixel_unit=self.calibration.params.pixel_unit,
                domain=self.domain,
                black_level=0,
                white_level=1,
            )

            # Remove batch dimension
            transformed = transformed.squeeze(0)
            mask_transformed = mask_transformed.squeeze(0)
            normalized = normalized.squeeze(0)
            mask = mask.squeeze(0)

            # Apply geometric augmentations
            augmentation_info = {}
            if self.augmentation_pipeline is not None:
                # Apply augmentations to transformed data (training) or not (validation/test)
                apply_augs = self.split == "train" and self.enable_augmentations

                (
                    transformed,
                    mask_transformed,
                    augmentation_info,
                ) = self.augmentation_pipeline(
                    transformed, mask_transformed, apply_augmentations=apply_augs
                )

            return {
                "raw_adu": raw_adu,
                "electrons": electrons,
                "normalized": normalized,
                "transformed": transformed,
                "mask": mask_transformed,
                "original_mask": mask,
                "image_metadata": image_metadata,
                "transform_metadata": transform_metadata,
                "augmentation_info": augmentation_info,
                "calibration_params": self.calibration.params,
                "scale": self.scale,
                "file_path": str(file_path),
                "split": self.split,
                "domain": self.domain,
            }

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            raise DataError(f"Failed to load {file_path}: {str(e)}") from e

    def get_domain_stats(self) -> Dict[str, Any]:
        """Get statistics about the domain dataset."""
        return {
            "domain": self.domain,
            "split": self.split,
            "num_files": len(self.file_paths),
            "scale": self.scale,
            "target_size": self.target_size,
            "calibration_file": getattr(
                self.calibration, "calibration_file", "default"
            ),
            "loader_stats": self.loader.get_statistics(),
        }


class MultiDomainDataset(Dataset):
    """
    Combined dataset for multiple imaging domains.

    This class combines multiple domain datasets with proper balancing
    and unified interface for multi-domain training.
    """

    def __init__(
        self,
        domain_configs: Dict[str, Dict[str, Any]],
        split: str = "train",
        balance_domains: bool = True,
        min_samples_per_domain: int = 100,
        **kwargs,
    ):
        """
        Initialize multi-domain dataset.

        Args:
            domain_configs: Dictionary mapping domain names to config dicts
            split: Dataset split ('train', 'val', 'test')
            balance_domains: Whether to balance samples across domains
            min_samples_per_domain: Minimum samples per domain
            **kwargs: Additional arguments passed to DomainDataset
        """
        self.domain_configs = domain_configs
        self.split = split
        self.balance_domains = balance_domains
        self.min_samples_per_domain = min_samples_per_domain

        # Create individual domain datasets
        self.domain_datasets = {}
        self.domain_sizes = {}

        for domain, config in domain_configs.items():
            try:
                # Merge config and kwargs, with kwargs taking precedence
                merged_config = {**config, **kwargs}
                dataset = DomainDataset(domain=domain, split=split, **merged_config)

                if len(dataset) < min_samples_per_domain:
                    logger.warning(
                        f"Domain {domain} has only {len(dataset)} samples, "
                        f"minimum is {min_samples_per_domain}"
                    )

                self.domain_datasets[domain] = dataset
                self.domain_sizes[domain] = len(dataset)

            except Exception as e:
                logger.error(f"Failed to initialize {domain} dataset: {e}")
                continue

        if len(self.domain_datasets) == 0:
            raise DataError("No valid domain datasets could be created")

        # Create sampling indices
        self._create_sampling_indices()

        logger.info(
            f"Initialized multi-domain dataset with {len(self.domain_datasets)} "
            f"domains, total {len(self)} samples"
        )

    def _create_sampling_indices(self):
        """Create balanced sampling indices across domains."""
        self.indices = []

        if self.balance_domains:
            # Balance domains by repeating smaller datasets
            max_size = max(self.domain_sizes.values())

            for domain, dataset in self.domain_datasets.items():
                domain_size = len(dataset)

                # Calculate how many times to repeat this domain
                repeats = max_size // domain_size
                remainder = max_size % domain_size

                # Add full repeats
                for _ in range(repeats):
                    for i in range(domain_size):
                        self.indices.append((domain, i))

                # Add partial repeat
                for i in range(remainder):
                    self.indices.append((domain, i))

        else:
            # No balancing, just concatenate
            for domain, dataset in self.domain_datasets.items():
                for i in range(len(dataset)):
                    self.indices.append((domain, i))

    def _get_global_index(self, domain: str, domain_index: int) -> int:
        """Convert domain-specific index to global dataset index."""
        for global_idx, (idx_domain, idx_domain_idx) in enumerate(self.indices):
            if idx_domain == domain and idx_domain_idx == domain_index:
                return global_idx
        raise ValueError(f"Could not find global index for {domain}[{domain_index}]")

    def __len__(self) -> int:
        """Get total dataset size."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load item from appropriate domain dataset.

        Returns:
            Same as DomainDataset.__getitem__ with additional 'domain' field
        """
        domain, domain_idx = self.indices[idx]

        item = self.domain_datasets[domain][domain_idx]
        item["domain"] = domain

        return item

    def get_domain_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across domains."""
        distribution = {}
        for domain, _ in self.indices:
            distribution[domain] = distribution.get(domain, 0) + 1
        return distribution

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = {
            "total_samples": len(self),
            "num_domains": len(self.domain_datasets),
            "split": self.split,
            "balance_domains": self.balance_domains,
            "domain_distribution": self.get_domain_distribution(),
            "domain_stats": {},
        }

        for domain, dataset in self.domain_datasets.items():
            stats["domain_stats"][domain] = dataset.get_domain_stats()

        return stats


def create_domain_dataset(
    domain: str,
    data_root: Union[str, Path],
    calibration_file: Optional[Union[str, Path]] = None,
    **kwargs,
) -> DomainDataset:
    """
    Convenience function to create a domain dataset.

    Args:
        domain: Domain name
        data_root: Root directory with data
        calibration_file: Optional calibration file
        **kwargs: Additional arguments for DomainDataset

    Returns:
        Configured DomainDataset
    """
    return DomainDataset(
        data_root=data_root, domain=domain, calibration_file=calibration_file, **kwargs
    )


def create_multi_domain_dataset(
    domain_configs: Dict[str, Dict[str, Any]], **kwargs
) -> MultiDomainDataset:
    """
    Convenience function to create a multi-domain dataset.

    Args:
        domain_configs: Configuration for each domain
        **kwargs: Additional arguments for MultiDomainDataset

    Returns:
        Configured MultiDomainDataset
    """
    return MultiDomainDataset(domain_configs, **kwargs)
