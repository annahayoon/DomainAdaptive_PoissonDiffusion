"""
PNG dataset loader for 8-bit PNG images.

This module provides dataset classes for loading 8-bit PNG images
and converting them to the format expected by the diffusion model training pipeline.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from core.logging_config import get_logger

logger = get_logger(__name__)


class PNGDataset(Dataset):
    """
    Dataset for loading 8-bit PNG images.

    Pickle-compatible for multiprocessing DataLoaders.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        max_files: Optional[int] = None,
        seed: int = 42,
        image_size: int = 256,
        channels: int = 4,  # RGBA for photography
    ):
        """
        Initialize PNG dataset.

        Args:
            data_root: Root directory containing PNG files
            split: Data split (train, val, test)
            max_files: Maximum number of files to load (None for all)
            seed: Random seed for file selection
            image_size: Target image size (will resize if needed)
            channels: Number of channels (3 for RGB, 4 for RGBA)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.max_files = max_files
        self.seed = seed
        self.image_size = image_size
        self.channels = channels

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")

        # Find paired PNG files and create noisy mapping
        self.image_files, self.noisy_mapping = self._find_paired_files()

        # Create train/val/test split
        self.split_files = self._create_split()

        # Setup transforms
        self.transform = self._setup_transforms()

        logger.info(
            f"Loaded {len(self.split_files)} paired PNG files for {split} split"
        )

    def _find_paired_files(self) -> Tuple[List[Path], Dict[Path, Path]]:
        """Find paired clean/noisy PNG files using scene_id and tile_id matching."""
        # Define clean and noisy directories
        self.clean_dir = self.data_root / "clean"
        self.noisy_dir = self.data_root / "noisy"

        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean directory not found: {self.clean_dir}")

        clean_files = []
        # Look for clean PNG files
        if self.clean_dir.exists():
            for pattern in ["*.png", "*.PNG"]:
                clean_files.extend(self.clean_dir.rglob(pattern))

        if not clean_files:
            raise FileNotFoundError(f"No clean PNG files found in {self.clean_dir}")

        # Sort for reproducibility
        clean_files.sort()

        # Find noisy files by scene_id + tile_id matching
        paired_files = []
        noisy_mapping = {}
        noisy_files_by_scene_tile = {}

        # First, organize noisy files by scene_id and tile_id
        if self.noisy_dir.exists() and self.noisy_dir.is_dir():
            for pattern in ["*.png", "*.PNG"]:
                for noisy_file in self.noisy_dir.rglob(pattern):
                    # Extract scene_id and tile_id from filename
                    # Format: photography_{scene_id}_{exposure}_tile_{tile_id}.png
                    parts = noisy_file.stem.split("_")
                    if len(parts) >= 6 and parts[0] == "photography":
                        scene_id = parts[1]
                        tile_id = parts[5]  # tile_XXXX, so parts[5] is XXXX
                        key = f"{scene_id}_{tile_id}"

                        if key not in noisy_files_by_scene_tile:
                            noisy_files_by_scene_tile[key] = []
                        noisy_files_by_scene_tile[key].append(noisy_file)

        logger.info(
            f"Found {len(noisy_files_by_scene_tile)} unique scene_id+tile_id combinations in noisy files"
        )

        # Now pair clean files with noisy files by scene_id and tile_id
        for clean_file in clean_files:
            # Extract scene_id and tile_id from clean filename
            clean_parts = clean_file.stem.split("_")
            if len(clean_parts) >= 6 and clean_parts[0] == "photography":
                clean_scene_id = clean_parts[1]
                clean_tile_id = clean_parts[5]  # tile_XXXX, so parts[5] is XXXX
                key = f"{clean_scene_id}_{clean_tile_id}"

                # Look for noisy files with matching scene_id and tile_id
                if key in noisy_files_by_scene_tile and noisy_files_by_scene_tile[key]:
                    # Use the first available noisy file for this scene_id + tile_id
                    noisy_file = noisy_files_by_scene_tile[key].pop(0)
                    paired_files.append(clean_file)
                    logger.debug(
                        f"Paired {clean_file.name} with {noisy_file.name} (scene_id: {clean_scene_id}, tile_id: {clean_tile_id})"
                    )
                else:
                    logger.warning(
                        f"No noisy file found for clean file {clean_file.name} (scene_id: {clean_scene_id}, tile_id: {clean_tile_id})"
                    )
            else:
                logger.warning(
                    f"Could not parse scene_id and tile_id from clean file: {clean_file.name}"
                )

        if not paired_files:
            available_noisy_count = len(
                [f for files in noisy_files_by_scene_tile.values() for f in files]
            )
            raise FileNotFoundError(
                f"No paired files found. Clean: {len(clean_files)}, Noisy available: {available_noisy_count}"
            )

        logger.info(f"Found {len(paired_files)} paired PNG files (clean+noisy)")
        logger.info(f"  Clean directory: {self.clean_dir}")
        logger.info(f"  Noisy directory: {self.noisy_dir}")
        logger.info(f"  Scene_id + tile_id based pairing successful")

        return paired_files, noisy_mapping

    def _create_split(self) -> List[Path]:
        """Create train/val/test split from image files."""
        np.random.seed(self.seed)

        # Shuffle files
        files = self.image_files.copy()
        np.random.shuffle(files)

        n_total = len(files)

        # Create splits: 80% train, 10% val, 10% test
        if self.split == "train":
            split_files = files[: int(0.8 * n_total)]
        elif self.split == "val":
            split_files = files[int(0.8 * n_total) : int(0.9 * n_total)]
        else:  # test
            split_files = files[int(0.9 * n_total) :]

        # Apply max_files limit if specified
        if self.max_files and len(split_files) > self.max_files:
            split_files = split_files[: self.max_files]

        return split_files

    def _setup_transforms(self) -> transforms.Compose:
        """Setup image transforms."""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),  # Converts to [0, 1] range
            transforms.Normalize(
                mean=[0.5] * self.channels, std=[0.5] * self.channels
            ),  # Convert to [-1, 1] range
        ]

        return transforms.Compose(transform_list)

    def _load_png_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess a PNG image."""
        try:
            # Load image
            image = Image.open(image_path)

            # Convert to RGBA if needed
            if self.channels == 4 and image.mode != "RGBA":
                if image.mode == "RGB":
                    # Add alpha channel
                    image = image.convert("RGBA")
                elif image.mode == "L":
                    # Convert grayscale to RGBA
                    image = image.convert("RGB").convert("RGBA")
                else:
                    image = image.convert("RGBA")
            elif self.channels == 3 and image.mode != "RGB":
                if image.mode == "RGBA":
                    # Remove alpha channel
                    image = image.convert("RGB")
                elif image.mode == "L":
                    # Convert grayscale to RGB
                    image = image.convert("RGB")
                else:
                    image = image.convert("RGB")
            elif self.channels == 1:
                image = image.convert("L")

            # Apply transforms
            tensor = self.transform(image)

            # Ensure correct number of channels
            if tensor.shape[0] != self.channels:
                if self.channels == 4 and tensor.shape[0] == 3:
                    # Add alpha channel (all ones)
                    alpha = torch.ones(1, tensor.shape[1], tensor.shape[2])
                    tensor = torch.cat([tensor, alpha], dim=0)
                elif self.channels == 3 and tensor.shape[0] == 4:
                    # Remove alpha channel
                    tensor = tensor[:3]
                elif self.channels == 1 and tensor.shape[0] == 3:
                    # Convert RGB to grayscale
                    tensor = transforms.Grayscale()(tensor)

            return tensor

        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return dummy image
            return torch.zeros(self.channels, self.image_size, self.image_size)

    def _simulate_noise(self, clean_image: torch.Tensor) -> torch.Tensor:
        """Simulate realistic camera noise on the clean image."""
        # Convert from [-1, 1] to [0, 1] for electron count calculation
        clean_01 = (clean_image + 1.0) / 2.0

        # Convert to electron count scale
        electrons = clean_01 * 1000.0  # Scale to reasonable electron count

        # Add Poisson noise (shot noise)
        poisson_noise = torch.poisson(electrons) - electrons

        # Add Gaussian read noise
        read_noise_std = 2.0  # Typical read noise
        gaussian_noise = torch.randn_like(electrons) * read_noise_std

        # Combine noises
        noisy_electrons = electrons + poisson_noise + gaussian_noise

        # Convert back to [0, 1] and then to [-1, 1] range
        noisy_01 = torch.clamp(noisy_electrons / 1000.0, 0.0, 1.0)
        noisy = noisy_01 * 2.0 - 1.0  # Convert to [-1, 1]

        return noisy

    def __len__(self) -> int:
        return len(self.split_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a paired clean/noisy PNG image.

        Returns dict with:
            - 'clean': Clean image data (C, H, W)
            - 'noisy': Noisy image data (C, H, W)
            - 'electrons': Electron count data (C, H, W)
            - 'domain': Domain tensor
            - 'metadata': Image metadata
            - 'domain_params': Domain-specific parameters
        """
        try:
            clean_path = self.split_files[idx]

            # Get corresponding noisy path from mapping
            if clean_path in self.noisy_mapping:
                noisy_path = self.noisy_mapping[clean_path]
                if noisy_path.exists():
                    # Load both clean and real noisy images
                    clean = self._load_png_image(clean_path)
                    noisy = self._load_png_image(noisy_path)
                    real_noise_used = True
                    logger.debug(f"Loaded real noisy image: {noisy_path.name}")
                else:
                    logger.warning(f"Noisy file missing in mapping: {noisy_path}")
                    # Fall back to synthetic noise if real noisy file missing
                    clean = self._load_png_image(clean_path)
                    noisy = self._simulate_noise(clean)
                    real_noise_used = False
            else:
                logger.warning(f"No noisy mapping found for: {clean_path.name}")
                # Fall back to synthetic noise if no mapping available
                clean = self._load_png_image(clean_path)
                noisy = self._simulate_noise(clean)
                real_noise_used = False

            # Ensure both images have the same dimensions
            if clean.shape != noisy.shape:
                logger.warning(
                    f"Shape mismatch for {clean_path.name}: clean {clean.shape} vs noisy {noisy.shape}"
                )
                # Resize noisy to match clean
                noisy = torch.nn.functional.interpolate(
                    noisy.unsqueeze(0), size=clean.shape[-2:], mode="bilinear"
                ).squeeze(0)

            # Create electron count version (convert from [-1,1] to electron scale)
            clean_01 = (clean + 1.0) / 2.0  # Convert to [0, 1]
            electrons = clean_01 * 1000.0  # Scale to electron count

            # Extract metadata
            metadata = {
                "clean_path": str(clean_path),
                "noisy_path": str(noisy_path)
                if real_noise_used and clean_path in self.noisy_mapping
                else None,
                "filename": clean_path.name,
                "image_size": self.image_size,
                "channels": self.channels,
                "real_noise_used": real_noise_used,
            }

            # Domain parameters
            domain_params = {
                "scale": 1000.0,  # Electron scale
                "gain": 1.0,
                "read_noise": 2.0,
                "background": 0.0,
            }

            # Domain ID (0 for photography)
            domain_id = 0

            return {
                "clean": clean,
                "noisy": noisy,
                "electrons": electrons,
                "domain": torch.tensor([domain_id]),
                "metadata": metadata,
                "domain_params": domain_params,
            }

        except Exception as e:
            logger.warning(f"Error loading PNG {idx}: {e}")
            # Return dummy data to prevent training crashes (in [-1, 1] range)
            return {
                "clean": torch.zeros(self.channels, self.image_size, self.image_size)
                - 1.0,  # [-1, 1] range
                "noisy": torch.zeros(self.channels, self.image_size, self.image_size)
                - 1.0,  # [-1, 1] range
                "electrons": torch.zeros(
                    self.channels, self.image_size, self.image_size
                ),  # [0, inf] range
                "domain": torch.tensor([0]),
                "metadata": {"corrupted": True, "original_idx": idx},
                "domain_params": {
                    "scale": 1000.0,
                    "gain": 1.0,
                    "read_noise": 2.0,
                    "background": 0.0,
                },
            }


class SimpleMultiDomainPNGDataset:
    """
    Simple wrapper to make PNG dataset look like a multi-domain dataset.

    This provides the interface expected by MultiDomainTrainer while being
    pickle-compatible for multiprocessing.
    """

    def __init__(self, dataset: Dataset, domain: str):
        """
        Initialize multi-domain wrapper.

        Args:
            dataset: Base PNG dataset to wrap
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


def create_png_datasets(
    data_root: Union[str, Path],
    domain: str = "photography",
    train_split: str = "train",
    val_split: str = "val",
    max_files: Optional[int] = None,
    seed: int = 42,
    image_size: int = 256,
    channels: int = 4,
) -> Tuple[SimpleMultiDomainPNGDataset, SimpleMultiDomainPNGDataset]:
    """
    Create training and validation datasets from PNG images.

    Args:
        data_root: Root directory containing PNG files
        domain: Domain name (photography, microscopy, astronomy)
        train_split: Training split name
        val_split: Validation split name
        max_files: Maximum files per split (None for all)
        seed: Random seed
        image_size: Target image size
        channels: Number of channels (3 for RGB, 4 for RGBA)

    Returns:
        Tuple of (train_dataset, val_dataset) wrapped in multi-domain interface
    """
    logger.info(f"Creating PNG datasets for {domain}")
    logger.info(f"Data root: {data_root}")

    # Create training dataset
    train_base = PNGDataset(
        data_root=data_root,
        split=train_split,
        max_files=max_files,
        seed=seed,
        image_size=image_size,
        channels=channels,
    )

    # Create validation dataset
    val_base = PNGDataset(
        data_root=data_root,
        split=val_split,
        max_files=max_files // 5 if max_files else None,  # Use 20% for validation
        seed=seed + 1,  # Different seed for validation
        image_size=image_size,
        channels=channels,
    )

    # Wrap in multi-domain interface
    train_dataset = SimpleMultiDomainPNGDataset(train_base, domain)
    val_dataset = SimpleMultiDomainPNGDataset(val_base, domain)

    logger.info(f"✅ Created PNG datasets:")
    logger.info(f"  Training: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Image size: {image_size}x{image_size}")
    logger.info(f"  Channels: {channels}")

    return train_dataset, val_dataset
