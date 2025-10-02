"""
Parquet dataset loader for compressed tile data.

This module provides dataset classes for loading preprocessed parquet files
from the processed_spark/unified_tiles directory. Compatible with the existing
training pipeline.
"""

import logging
import pickle
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.logging_config import get_logger

logger = get_logger(__name__)


class ParquetTileDataset(Dataset):
    """
    Dataset for loading preprocessed parquet tile data.
    
    This dataset loads clean/noisy tile pairs from parquet files which
    contain compressed tile data with proper noise characteristics.
    
    Pickle-compatible for multiprocessing DataLoaders.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        max_files: Optional[int] = None,
        seed: int = 42,
        tile_size: int = 128,
    ):
        """
        Initialize parquet tile dataset.

        Args:
            data_root: Root directory containing parquet files
            split: Data split (train, val, test)
            max_files: Maximum number of files to load (None for all)
            seed: Random seed for file selection
            tile_size: Expected tile size (default 128x128)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.max_files = max_files
        self.seed = seed
        self.tile_size = tile_size

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")

        # Load parquet files and filter by split
        self.data = self._load_parquet_data()
        
        logger.info(f"Loaded {len(self.data)} tiles from parquet files (split: {split})")

    def _load_parquet_data(self) -> pd.DataFrame:
        """Load and filter parquet data."""
        # Find all parquet files
        parquet_files = []
        for file_path in self.data_root.rglob("*.parquet"):
            if file_path.is_file():
                parquet_files.append(file_path)
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_root}")
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        # Load and concatenate all parquet files
        dataframes = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                if len(df) > 0:  # Only add non-empty dataframes
                    dataframes.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No valid parquet data found")
        
        # Concatenate all data
        all_data = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Total tiles before filtering: {len(all_data)}")
        
        # Filter by split
        if 'split' in all_data.columns:
            split_data = all_data[all_data['split'] == self.split].copy()
        else:
            # If no split column, create train/val/test split based on tile_id hash
            np.random.seed(self.seed)
            n_total = len(all_data)
            indices = np.arange(n_total)
            np.random.shuffle(indices)
            
            if self.split == 'train':
                split_indices = indices[:int(0.8 * n_total)]
            elif self.split == 'val':
                split_indices = indices[int(0.8 * n_total):int(0.9 * n_total)]
            else:  # test
                split_indices = indices[int(0.9 * n_total):]
            
            split_data = all_data.iloc[split_indices].copy()
        
        logger.info(f"Tiles after split filtering ({self.split}): {len(split_data)}")
        
        # Apply max_files limit if specified
        if self.max_files and len(split_data) > self.max_files:
            split_data = split_data.sample(n=self.max_files, random_state=self.seed)
            logger.info(f"Limited to {self.max_files} tiles")
        
        return split_data.reset_index(drop=True)

    def _decompress_tile_data(self, compressed_data: str, compression_method: str) -> np.ndarray:
        """Decompress tile data from compressed format."""
        try:
            if compression_method == 'pickle+base64':
                # Decode base64 and unpickle
                decoded_data = base64.b64decode(compressed_data)
                tile_array = pickle.loads(decoded_data)
                return tile_array
            else:
                raise ValueError(f"Unsupported compression method: {compression_method}")
        except Exception as e:
            logger.error(f"Failed to decompress tile data: {e}")
            # Return dummy data to prevent crashes
            return np.zeros((4, self.tile_size, self.tile_size), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a single tile.
        
        Returns dict with:
            - 'clean': Clean tile data (4, H, W)
            - 'noisy': Noisy tile data (4, H, W) 
            - 'electrons': Electron count data (4, H, W)
            - 'domain': Domain tensor
            - 'metadata': Tile metadata
            - 'domain_params': Domain-specific parameters
        """
        try:
            row = self.data.iloc[idx]
            
            # Decompress tile data
            tile_data = self._decompress_tile_data(
                row['tile_data_compressed'], 
                row['compression_method']
            )
            
            # Convert to torch tensors and normalize to [-1, 1]
            if tile_data.ndim == 3 and tile_data.shape[0] == 4:
                # Data is already in (C, H, W) format
                clean = torch.from_numpy(tile_data).float()
                # Normalize to [-1, 1] range (assuming input is in [0, 1])
                clean = clean * 2.0 - 1.0
            else:
                # Handle other formats or create dummy data
                logger.warning(f"Unexpected tile data shape: {tile_data.shape}")
                clean = torch.zeros(4, self.tile_size, self.tile_size, dtype=torch.float32) - 1.0  # [-1, 1] range
            
            # For training, we need to generate noisy version
            # Use the calibration parameters to add realistic noise
            gain = row.get('gain', 1.0)
            read_noise = row.get('read_noise', 1.5)
            
            # Convert from [-1, 1] to [0, 1] for electron count calculation
            clean_01 = (clean + 1.0) / 2.0
            electrons = clean_01 * 1000.0  # Scale to reasonable electron count
            
            # Add Poisson noise (shot noise) and Gaussian read noise
            poisson_noise = torch.poisson(electrons) - electrons
            gaussian_noise = torch.randn_like(electrons) * read_noise
            
            # Create noisy version
            noisy_electrons = electrons + poisson_noise + gaussian_noise
            noisy_01 = torch.clamp(noisy_electrons / 1000.0, 0.0, 1.0)  # Normalize to [0, 1]
            noisy = noisy_01 * 2.0 - 1.0  # Convert to [-1, 1]
            
            # Extract metadata
            metadata = {
                'tile_id': row.get('tile_id', f'tile_{idx}'),
                'source_file': row.get('source_file', 'unknown'),
                'grid_x': row.get('grid_x', 0),
                'grid_y': row.get('grid_y', 0),
                'quality_score': row.get('quality_score', 0.5),
                'valid_ratio': row.get('valid_ratio', 1.0),
                'is_edge_tile': row.get('is_edge_tile', False),
            }
            
            # Domain parameters
            domain_params = {
                'scale': 1000.0,  # Electron scale
                'gain': gain,
                'read_noise': read_noise,
                'background': 0.0,
            }
            
            # Domain ID (0 for photography)
            domain_id = row.get('domain_id', 0)
            
            return {
                'clean': clean,
                'noisy': noisy,
                'electrons': electrons,
                'domain': torch.tensor([domain_id]),
                'metadata': metadata,
                'domain_params': domain_params,
            }
            
        except Exception as e:
            logger.warning(f"Error loading tile {idx}: {e}")
            # Return dummy data to prevent training crashes (in [-1, 1] range)
            return {
                'clean': torch.zeros(4, self.tile_size, self.tile_size, dtype=torch.float32) - 1.0,  # [-1, 1] range
                'noisy': torch.zeros(4, self.tile_size, self.tile_size, dtype=torch.float32) - 1.0,  # [-1, 1] range
                'electrons': torch.zeros(4, self.tile_size, self.tile_size, dtype=torch.float32),     # [0, inf] range
                'domain': torch.tensor([0]),
                'metadata': {'corrupted': True, 'original_idx': idx},
                'domain_params': {
                    'scale': 1000.0,
                    'gain': 1.0,
                    'read_noise': 1.5,
                    'background': 0.0,
                },
            }


class SimpleMultiDomainParquetDataset:
    """
    Simple wrapper to make parquet dataset look like a multi-domain dataset.
    
    This provides the interface expected by MultiDomainTrainer while being
    pickle-compatible for multiprocessing.
    """

    def __init__(self, dataset: Dataset, domain: str):
        """
        Initialize multi-domain wrapper.

        Args:
            dataset: Base parquet dataset to wrap
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
            'combined_dataset': self.combined_dataset,
            'domain_datasets': self.domain_datasets,
        }

    def __setstate__(self, state):
        """Support for unpickling in multiprocessing."""
        self.__dict__.update(state)


def create_parquet_datasets(
    data_root: Union[str, Path],
    domain: str = "photography",
    train_split: str = "train",
    val_split: str = "val",
    max_files: Optional[int] = None,
    seed: int = 42,
    tile_size: int = 128,
) -> Tuple[SimpleMultiDomainParquetDataset, SimpleMultiDomainParquetDataset]:
    """
    Create training and validation datasets from parquet data.

    Args:
        data_root: Root directory containing parquet files
        domain: Domain name (photography, microscopy, astronomy)
        train_split: Training split name
        val_split: Validation split name
        max_files: Maximum files per split (None for all)
        seed: Random seed
        tile_size: Expected tile size

    Returns:
        Tuple of (train_dataset, val_dataset) wrapped in multi-domain interface
    """
    logger.info(f"Creating parquet datasets for {domain}")
    logger.info(f"Data root: {data_root}")

    # Create training dataset
    train_base = ParquetTileDataset(
        data_root=data_root,
        split=train_split,
        max_files=max_files,
        seed=seed,
        tile_size=tile_size,
    )

    # Create validation dataset
    val_base = ParquetTileDataset(
        data_root=data_root,
        split=val_split,
        max_files=max_files // 5 if max_files else None,  # Use 20% for validation
        seed=seed + 1,  # Different seed for validation
        tile_size=tile_size,
    )

    # Wrap in multi-domain interface
    train_dataset = SimpleMultiDomainParquetDataset(train_base, domain)
    val_dataset = SimpleMultiDomainParquetDataset(val_base, domain)

    logger.info(f"✅ Created parquet datasets:")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")

    return train_dataset, val_dataset
