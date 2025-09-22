"""
Preprocessed dataset loading utilities.

This module provides dataset classes for loading preprocessed .pt files
from the PKL-DiffusionDenoising data format. These classes are defined at
module level to be pickle-compatible with multiprocessing DataLoaders.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from core.logging_config import get_logger

logger = get_logger(__name__)


class PreprocessedDataset(Dataset):
    """
    Dataset for loading preprocessed .pt files from posterior directory.

    This dataset loads clean/noisy pairs from the posterior directory which
    contains real photography data with proper noise characteristics.

    Pickle-compatible for multiprocessing DataLoaders.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        domain: str = "photography",
        split: str = "train",
        max_files: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Initialize preprocessed dataset.

        Args:
            data_root: Root directory containing preprocessed data
            domain: Domain name (photography, microscopy, astronomy)
            split: Data split (train, val, test)
            max_files: Maximum number of files to load (None for all)
            seed: Random seed for file selection
        """
        self.data_root = Path(data_root)
        self.domain = domain
        self.split = split
        self.max_files = max_files
        self.seed = seed

        # Path to preprocessed data - use prior_clean for diffusion prior training
        self.data_dir = self.data_root / "prior_clean" / domain / split

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Load file list
        self.files = self._load_file_list()

        logger.info(f"Loaded {len(self.files)} files from {self.data_dir}")

    def _load_file_list(self) -> List[Path]:
        """Load and filter file list."""
        files = list(self.data_dir.glob("*.pt"))

        if len(files) == 0:
            raise ValueError(f"No .pt files found in {self.data_dir}")

        # Sort for reproducibility
        files.sort()

        # Apply max_files limit
        if self.max_files is not None and self.max_files > 0:
            import random

            rng = random.Random(self.seed)
            if len(files) > self.max_files:
                files = rng.sample(files, self.max_files)
                files.sort()  # Keep sorted for reproducibility

        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load a single sample.

        Returns:
            Dict containing:
                - clean: Clean image tensor [C, H, W]
                - noisy: Noisy image tensor [C, H, W]
                - electrons: Electron count tensor [C, H, W]
                - domain: Domain tensor [1]
                - metadata: Metadata dict
                - domain_params: Domain calibration parameters
        """
        file_path = self.files[idx]

        try:
            # Load preprocessed data
            data = torch.load(file_path, map_location="cpu")

            # Extract clean image from prior_clean data (only has clean_norm)
            clean = data["clean_norm"].float()

            # Handle negative values in clean image (common in astronomy due to background subtraction)
            # RESEARCH DECISION: Use adaptive offset approach for diffusion restoration
            #
            # Why offset (not clamp to zero):
            # 1. Information Preservation: Maintains all signal relationships
            # 2. Physical Consistency: Negative values are calibration artifacts, not physical
            # 3. Diffusion Learning: Model learns to restore true physical intensities
            # 4. Poisson Compatibility: Ensures non-negative inputs for noise modeling
            # 5. Inverse Transform: Offset can be reversed if needed for evaluation
            #
            # Alternative: Clamping to zero would preserve relative structure but lose
            # absolute information about true background levels and signal strengths.
            if clean.min() < 0:
                # Use only abs(min) without adding extra offset - this preserves the data distribution
                # The smallest value becomes 0, maintaining relative relationships
                offset = abs(clean.min().item())
                clean = clean + offset
                logger.debug(
                    f"Applied offset {offset:.3f} to handle negative values in clean image"
                )

                # Store offset for potential inverse transform during evaluation
                metadata = data.get("metadata", {})
                metadata["astronomy_offset"] = offset

            # Generate synthetic noisy version for training (diffusion prior learning)
            # Add Poisson + Gaussian noise to simulate real-world conditions
            electrons = clean * 100.0  # Scale to reasonable electron count

            # Add Poisson noise
            noisy = torch.poisson(electrons)

            # Add Gaussian read noise
            read_noise = 2.0  # Typical read noise level
            noisy = noisy + torch.randn_like(noisy) * read_noise

            # Convert back to normalized scale and ensure non-negative
            noisy = noisy / 100.0
            noisy = torch.clamp(noisy, 0.0, None)

            # Extract metadata
            metadata = data.get("metadata", {})
            if "scene_id" not in metadata:
                metadata["scene_id"] = file_path.stem

            # Extract calibration if available
            calibration = data.get("calibration", {})
            domain_params = {
                "scale": calibration.get("scale_clean", torch.tensor(100.0))
                .mean()
                .item()
                if torch.is_tensor(calibration.get("scale_clean"))
                else 100.0,
                "gain": calibration.get("gain", 2.5),
                "read_noise": calibration.get("read_noise", 2.0),
                "background": calibration.get("background", 0.0),
            }

            return {
                "clean": clean,
                "noisy": noisy,
                "electrons": electrons,
                "domain": torch.tensor([data.get("domain_id", 0)]),
                "metadata": metadata,
                "domain_params": domain_params,
            }

        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            # Return dummy data to prevent training crashes
            return {
                "clean": torch.zeros(4, 128, 128, dtype=torch.float32),
                "noisy": torch.zeros(4, 128, 128, dtype=torch.float32),
                "electrons": torch.zeros(4, 128, 128, dtype=torch.float32),
                "domain": torch.tensor([0]),
                "metadata": {"corrupted": True, "original_idx": idx},
                "domain_params": {
                    "scale": 100.0,
                    "gain": 2.5,
                    "read_noise": 2.0,
                    "background": 0.0,
                },
            }

    def __getstate__(self):
        """Support for pickling in multiprocessing."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Support for unpickling in multiprocessing."""
        self.__dict__.update(state)


class SimpleMultiDomainDataset:
    """
    Simple wrapper to make a single dataset look like a multi-domain dataset.

    This provides the interface expected by MultiDomainTrainer while being
    pickle-compatible for multiprocessing.
    """

    def __init__(self, dataset: Dataset, domain: str):
        """
        Initialize multi-domain wrapper.

        Args:
            dataset: Base dataset to wrap
            domain: Domain name
        """
        self.combined_dataset = dataset
        self.domain_datasets = {domain: dataset}

    def __len__(self) -> int:
        return len(self.combined_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.combined_dataset[idx]

    def __getstate__(self):
        """Support for pickling in multiprocessing."""
        return {
            "combined_dataset": self.combined_dataset,
            "domain_datasets": self.domain_datasets,
        }

    def __setstate__(self, state):
        """Support for unpickling in multiprocessing."""
        self.__dict__.update(state)


def create_preprocessed_datasets(
    data_root: Union[str, Path],
    domain: str = "photography",
    train_split: str = "train",
    val_split: str = "val",
    max_files: Optional[int] = None,
    seed: int = 42,
) -> Tuple[SimpleMultiDomainDataset, SimpleMultiDomainDataset]:
    """
    Create training and validation datasets from preprocessed data.

    Args:
        data_root: Root directory containing preprocessed data
        domain: Domain name (photography, microscopy, astronomy)
        train_split: Training split name
        val_split: Validation split name
        max_files: Maximum files per split (None for all)
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset) wrapped in multi-domain interface
    """
    logger.info(f"Creating preprocessed datasets for {domain}")
    logger.info(f"Data root: {data_root}")

    # Create training dataset
    train_base = PreprocessedDataset(
        data_root=data_root,
        domain=domain,
        split=train_split,
        max_files=max_files,
        seed=seed,
    )

    # Create validation dataset
    val_base = PreprocessedDataset(
        data_root=data_root,
        domain=domain,
        split=val_split,
        max_files=max_files // 5 if max_files else None,  # Use 20% for validation
        seed=seed + 1,  # Different seed for validation
    )

    # Wrap in multi-domain interface
    train_dataset = SimpleMultiDomainDataset(train_base, domain)
    val_dataset = SimpleMultiDomainDataset(val_base, domain)

    logger.info(f"✅ Created datasets:")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")

    return train_dataset, val_dataset
