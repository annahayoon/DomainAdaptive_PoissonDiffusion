"""
Core preprocessing utilities for domain-adaptive Poisson-Gaussian diffusion.
Provides noise estimation, calibration, and normalization functions.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import HuberRegressor


def estimate_noise_params_photon_transfer(
    images: List[np.ndarray],
    black_levels: Optional[List[np.ndarray]] = None,
    min_patches: int = 50,
    patch_size: int = 32,
) -> Tuple[float, float]:
    """
    Estimate gain and read noise using photon transfer curve method.

    Args:
        images: List of raw images (ADU units)
        black_levels: List of black level arrays (same shape as images)
        min_patches: Minimum number of uniform patches to collect
        patch_size: Size of patches to extract

    Returns:
        gain: Gain in electrons per ADU
        read_noise: Read noise in electrons
    """
    patches_mean = []
    patches_var = []

    for i, img in enumerate(images):
        if black_levels is not None:
            img = np.maximum(img - black_levels[i], 0)

        # Find uniform regions (low gradient)
        grad_x = np.abs(np.gradient(img, axis=0))
        grad_y = np.abs(np.gradient(img, axis=1))
        gradient_mag = grad_x + grad_y

        # Skip images that are too small
        if img.shape[0] < patch_size + 10 or img.shape[1] < patch_size + 10:
            continue

        # Threshold for uniform regions (bottom 70% of gradients, extremely lenient for microscopy)
        uniform_threshold = np.percentile(gradient_mag, 70)
        uniform_mask = gradient_mag < uniform_threshold

        # Extract patches from uniform regions
        patch_count = 0
        max_attempts = min_patches * 100  # Even more attempts
        attempts = 0
        target_patches_per_image = max(1, min_patches // len(images))

        while patch_count < target_patches_per_image and attempts < max_attempts:
            attempts += 1

            # Random patch location with safety margin
            h = np.random.randint(0, max(1, img.shape[0] - patch_size))
            w = np.random.randint(0, max(1, img.shape[1] - patch_size))

            patch_mask = uniform_mask[h : h + patch_size, w : w + patch_size]

            # Accept patch if >20% is uniform (extremely lenient for microscopy) and has any signal
            if patch_mask.mean() > 0.2:
                patch = img[h : h + patch_size, w : w + patch_size]
                patch_mean = patch.mean()
                patch_var = patch.var(ddof=1)

                # Very lenient requirements for microscopy - any reasonable signal
                if patch_mean > 0.1 and patch_var > 0.001:
                    patches_mean.append(patch_mean)
                    patches_var.append(patch_var)
                    patch_count += 1

    if len(patches_mean) < min_patches:
        warnings.warn(f"Only found {len(patches_mean)} patches, using defaults")
        return 2.47, 1.82  # Default Sony A7S values

    # Robust linear fit: var = gain * mean + read_noise^2/gain
    X = np.array(patches_mean).reshape(-1, 1)
    y = np.array(patches_var)

    # Remove outliers
    q25, q75 = np.percentile(y, [25, 75])
    iqr = q75 - q25
    valid_mask = (y >= q25 - 1.5 * iqr) & (y <= q75 + 1.5 * iqr)

    if valid_mask.sum() < min_patches // 2:
        warnings.warn("Too many outliers in noise estimation, using defaults")
        return 2.47, 1.82

    X = X[valid_mask]
    y = y[valid_mask]

    # Fit linear model
    reg = HuberRegressor(epsilon=1.35)  # Robust to outliers
    reg.fit(X, y)

    gain = reg.coef_[0]
    read_var_adu = reg.intercept_
    read_noise = np.sqrt(max(read_var_adu * gain, 0))

    # Sanity checks
    if gain <= 0 or gain > 10:
        warnings.warn(f"Unrealistic gain {gain:.3f}, using default")
        gain = 2.47

    if read_noise < 0.5 or read_noise > 20:
        warnings.warn(f"Unrealistic read noise {read_noise:.3f}, using default")
        read_noise = 1.82

    return float(gain), float(read_noise)


def compute_global_scale(
    clean_images_electrons: List[np.ndarray],
    percentile: float = 99.9,
    max_pixels: int = int(1e7),
) -> float:
    """
    Compute global normalization scale from clean images.

    Args:
        clean_images_electrons: List of clean images in electron units
        percentile: Percentile to use for scale computation
        max_pixels: Maximum pixels to sample for efficiency

    Returns:
        scale: Normalization scale in electrons
    """
    all_pixels = []

    for img in clean_images_electrons:
        pixels = img.flatten()
        # Skip very dark pixels (likely background)
        pixels = pixels[pixels > np.percentile(pixels, 5)]
        all_pixels.extend(pixels)

    all_pixels = np.array(all_pixels)

    # Sample if too many pixels
    if len(all_pixels) > max_pixels:
        indices = np.random.choice(len(all_pixels), max_pixels, replace=False)
        all_pixels = all_pixels[indices]

    scale = np.percentile(all_pixels, percentile)

    # Sanity check
    if scale <= 0 or scale < 100:  # Too small for typical images
        warnings.warn(f"Computed scale {scale:.1f} seems too small")
        scale = 10000.0  # Reasonable default

    return float(scale)


def extract_tiles_with_augmentation(
    image: np.ndarray,
    num_tiles: int = 50,
    tile_size: int = 128,
    augment: bool = True,
    min_signal_threshold: float = 0.05,
) -> List[np.ndarray]:
    """
    Extract tiles from image with optional augmentation.

    Args:
        image: Input image [C, H, W]
        num_tiles: Number of tiles to extract
        tile_size: Size of square tiles
        augment: Whether to apply random augmentations
        min_signal_threshold: Minimum signal level for tile acceptance

    Returns:
        List of extracted tiles
    """
    C, H, W = image.shape
    tiles = []

    if H < tile_size or W < tile_size:
        warnings.warn(f"Image too small {H}x{W} for tiles {tile_size}x{tile_size}")
        return []

    max_attempts = num_tiles * 5
    attempts = 0

    while len(tiles) < num_tiles and attempts < max_attempts:
        attempts += 1

        # Random position
        h = np.random.randint(0, H - tile_size + 1)
        w = np.random.randint(0, W - tile_size + 1)

        tile = image[:, h : h + tile_size, w : w + tile_size].copy()

        # Check if tile has sufficient signal
        if tile.mean() < min_signal_threshold:
            continue

        if augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                tile = np.flip(tile, axis=2)

            # Random vertical flip
            if np.random.rand() > 0.5:
                tile = np.flip(tile, axis=1)

            # Random 90-degree rotations
            k = np.random.randint(0, 4)
            tile = np.rot90(tile, k, axes=(1, 2))

        tiles.append(tile.copy())

    return tiles


def create_saturation_mask(
    raw_image: np.ndarray,
    white_level: float,
    black_level: Optional[np.ndarray] = None,
    threshold_fraction: float = 0.95,
) -> np.ndarray:
    """
    Create saturation mask for raw image.

    Args:
        raw_image: Raw image data
        white_level: White level in ADU
        black_level: Black level array (same shape as raw_image)
        threshold_fraction: Fraction of white level for saturation threshold

    Returns:
        Boolean mask where True indicates saturated pixels
    """
    if black_level is not None:
        effective_white = white_level - black_level
    else:
        effective_white = white_level

    saturated = raw_image >= effective_white * threshold_fraction
    return saturated


def pack_bayer_to_channels(bayer: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """
    Pack Bayer pattern into 4-channel format.

    Args:
        bayer: Bayer pattern image [H, W]
        pattern: Color pattern (RGGB, GRBG, BGGR, GBRG)

    Returns:
        Packed image [4, H//2, W//2] in RGGB order
    """
    H, W = bayer.shape

    if H % 2 != 0 or W % 2 != 0:
        # Crop to even dimensions
        H = H // 2 * 2
        W = W // 2 * 2
        bayer = bayer[:H, :W]

    packed = np.zeros((4, H // 2, W // 2), dtype=bayer.dtype)

    if pattern == "RGGB":
        packed[0] = bayer[0::2, 0::2]  # R
        packed[1] = bayer[0::2, 1::2]  # G1
        packed[2] = bayer[1::2, 0::2]  # G2
        packed[3] = bayer[1::2, 1::2]  # B
    elif pattern == "GRBG":
        packed[1] = bayer[0::2, 0::2]  # G1
        packed[0] = bayer[0::2, 1::2]  # R
        packed[3] = bayer[1::2, 0::2]  # B
        packed[2] = bayer[1::2, 1::2]  # G2
    elif pattern == "BGGR":
        packed[3] = bayer[0::2, 0::2]  # B
        packed[2] = bayer[0::2, 1::2]  # G2
        packed[1] = bayer[1::2, 0::2]  # G1
        packed[0] = bayer[1::2, 1::2]  # R
    elif pattern == "GBRG":
        packed[2] = bayer[0::2, 0::2]  # G2
        packed[3] = bayer[0::2, 1::2]  # B
        packed[0] = bayer[1::2, 0::2]  # R
        packed[1] = bayer[1::2, 1::2]  # G1
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return packed


def estimate_background_level(
    image: np.ndarray,
    method: str = "percentile",
    percentile: float = 5.0,
    morphology_size: int = 30,
) -> float:
    """
    Estimate background level in image.

    Args:
        image: Input image
        method: 'percentile' or 'morphology'
        percentile: Percentile for percentile method
        morphology_size: Structuring element size for morphology method

    Returns:
        Background level
    """
    if method == "percentile":
        # Use percentile of darkest regions
        dark_pixels = image[image < np.percentile(image, 20)]
        if len(dark_pixels) > 0:
            background = np.percentile(dark_pixels, percentile)
        else:
            background = np.percentile(image, percentile)

    elif method == "morphology":
        try:
            from scipy.ndimage import gaussian_filter
            from skimage.morphology import disk, opening

            # Morphological opening to estimate background
            if image.ndim == 2:
                structure = disk(morphology_size)
                background_map = opening(image, structure)
                background_map = gaussian_filter(background_map, sigma=10)
                background = np.percentile(background_map, 10)
            else:
                # For multi-channel, process each channel
                backgrounds = []
                for c in range(image.shape[0]):
                    structure = disk(morphology_size)
                    bg_map = opening(image[c], structure)
                    bg_map = gaussian_filter(bg_map, sigma=10)
                    backgrounds.append(np.percentile(bg_map, 10))
                background = np.mean(backgrounds)
        except ImportError:
            warnings.warn("scikit-image not available, falling back to percentile")
            background = estimate_background_level(
                image, method="percentile", percentile=percentile
            )

    else:
        raise ValueError(f"Unknown background estimation method: {method}")

    return float(background)


def split_scenes_by_ratio(
    scene_list: List[Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split scenes into train/val/test sets.

    Args:
        scene_list: List of scenes to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility. If None, uses system entropy.

    Returns:
        (train_scenes, val_scenes, test_scenes)
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    if seed is not None:
        np.random.seed(seed)
    else:
        # Use system entropy for true randomness
        np.random.seed()

    indices = np.random.permutation(len(scene_list))

    n_train = int(len(scene_list) * train_ratio)
    n_val = int(len(scene_list) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    train_scenes = [scene_list[i] for i in train_indices]
    val_scenes = [scene_list[i] for i in val_indices]
    test_scenes = [scene_list[i] for i in test_indices]

    return train_scenes, val_scenes, test_scenes


def save_preprocessed_tile(
    tile: np.ndarray,
    domain_id: int,
    scene_id: str,
    tile_idx: int,
    save_path: Path,
    augmented: bool = False,
) -> None:
    """
    Save a preprocessed tile in the standard format.

    Args:
        tile: Normalized tile data [C, H, W]
        domain_id: Domain identifier (0=photo, 1=micro, 2=astro)
        scene_id: Original scene identifier
        tile_idx: Tile index within dataset
        save_path: Path to save the .pt file
        augmented: Whether augmentation was applied
    """
    tile_data = {
        "clean_norm": torch.from_numpy(tile).float(),
        "domain_id": domain_id,
        "metadata": {
            "scene_id": scene_id,
            "tile_idx": tile_idx,
            "augmented": augmented,
        },
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tile_data, save_path)


def save_preprocessed_scene(
    noisy_norm: np.ndarray,
    clean_norm: Optional[np.ndarray],
    calibration: Dict[str, float],
    masks: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    save_path: Path,
) -> None:
    """
    Save a preprocessed scene in the standard format.

    Args:
        noisy_norm: Normalized noisy image [C, H, W]
        clean_norm: Normalized clean image [C, H, W] or None
        calibration: Calibration parameters
        masks: Dictionary of masks (valid, saturated)
        metadata: Scene metadata
        save_path: Path to save the .pt file
    """
    scene_data = {
        "noisy_norm": torch.from_numpy(noisy_norm).float(),
        "clean_norm": torch.from_numpy(clean_norm).float()
        if clean_norm is not None
        else None,
        "calibration": calibration,
        "masks": {k: torch.from_numpy(v) for k, v in masks.items()},
        "metadata": metadata,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scene_data, save_path)


def create_valid_mask(image_shape: Tuple[int, int], border_crop: int = 0) -> np.ndarray:
    """
    Create a valid pixel mask.

    Args:
        image_shape: (H, W) shape of image
        border_crop: Number of border pixels to mark as invalid

    Returns:
        Boolean mask [1, H, W] where True indicates valid pixels
    """
    H, W = image_shape
    mask = np.ones((1, H, W), dtype=bool)

    if border_crop > 0:
        mask[:, :border_crop, :] = False
        mask[:, -border_crop:, :] = False
        mask[:, :, :border_crop] = False
        mask[:, :, -border_crop:] = False

    return mask
