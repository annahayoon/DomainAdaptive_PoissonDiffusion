"""
Enhanced preprocessing utilities with parallel processing and memory optimization.
"""

from __future__ import annotations

import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def parallel_extract_tiles_with_augmentation(
    image: np.ndarray,
    num_tiles: int = 50,
    tile_size: int = 128,
    augment: bool = True,
    min_signal_threshold: float = 0.05,
    n_workers: int = None,
    batch_size: int = 10,
) -> List[np.ndarray]:
    """
    Extract tiles from image with parallel processing.

    Args:
        image: Input image [C, H, W]
        num_tiles: Number of tiles to extract
        tile_size: Size of square tiles
        augment: Whether to apply augmentation
        min_signal_threshold: Minimum signal level for valid tile
        n_workers: Number of parallel workers (None = auto)
        batch_size: Number of tiles per worker batch

    Returns:
        List of tiles [C, tile_size, tile_size]
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count() - 1, 4)

    C, H, W = image.shape

    if H < tile_size or W < tile_size:
        # Image too small, return padded version
        padded = np.zeros((C, tile_size, tile_size), dtype=image.dtype)
        h_start = (tile_size - H) // 2
        w_start = (tile_size - W) // 2
        padded[:, h_start : h_start + H, w_start : w_start + W] = image
        return [padded]

    # Generate tile parameters
    tile_params = []
    for _ in range(num_tiles):
        h = np.random.randint(0, H - tile_size + 1)
        w = np.random.randint(0, W - tile_size + 1)

        # Random augmentation parameters
        flip_h = np.random.rand() > 0.5 if augment else False
        flip_v = np.random.rand() > 0.5 if augment else False
        rot_k = np.random.randint(0, 4) if augment else 0

        tile_params.append((h, w, flip_h, flip_v, rot_k))

    # Process tiles in parallel
    def extract_single_tile(params):
        h, w, flip_h, flip_v, rot_k = params
        tile = image[:, h : h + tile_size, w : w + tile_size].copy()

        # Check signal level
        if tile.max() < min_signal_threshold:
            return None

        # Apply augmentation
        if flip_h:
            tile = np.flip(tile, axis=2)
        if flip_v:
            tile = np.flip(tile, axis=1)
        if rot_k > 0:
            tile = np.rot90(tile, rot_k, axes=(1, 2))

        return tile

    tiles = []

    if n_workers > 1:
        # Use thread pool for I/O bound operations
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(extract_single_tile, params) for params in tile_params
            ]

            for future in futures:
                result = future.result()
                if result is not None:
                    tiles.append(result)
    else:
        # Sequential processing
        for params in tile_params:
            result = extract_single_tile(params)
            if result is not None:
                tiles.append(result)

    return tiles


def memory_efficient_compute_global_scale(
    image_loader: Callable,
    num_images: int,
    percentile: float = 99.9,
    batch_size: int = 10,
    sample_pixels_per_image: int = 100000,
) -> float:
    """
    Compute global scale with memory-efficient streaming.

    Args:
        image_loader: Function that yields images (as generator)
        num_images: Total number of images
        percentile: Percentile for scale computation
        batch_size: Number of images to process at once
        sample_pixels_per_image: Max pixels to sample per image

    Returns:
        scale: Global normalization scale
    """
    all_samples = []

    # Process images in batches
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)

        for img_idx in range(batch_start, batch_end):
            try:
                image = next(image_loader)

                # Sample pixels from this image
                flat_pixels = image.flatten()
                if len(flat_pixels) > sample_pixels_per_image:
                    # Random sampling for large images
                    indices = np.random.choice(
                        len(flat_pixels), sample_pixels_per_image, replace=False
                    )
                    sampled = flat_pixels[indices]
                else:
                    sampled = flat_pixels

                all_samples.extend(sampled)

                # Keep memory bounded
                if len(all_samples) > 1e7:
                    # Compute intermediate percentile and reduce
                    all_samples = [
                        np.percentile(all_samples, p) for p in np.linspace(0, 100, 1000)
                    ]

            except StopIteration:
                break

    # Final percentile computation
    if all_samples:
        scale = np.percentile(all_samples, percentile)
    else:
        scale = 1.0

    return float(scale)


def parallel_process_scenes(
    scenes: List[Any],
    process_fn: Callable,
    n_workers: int = None,
    desc: str = "Processing scenes",
) -> List[Any]:
    """
    Process scenes in parallel with progress bar.

    Args:
        scenes: List of scenes to process
        process_fn: Function to apply to each scene
        n_workers: Number of workers (None = auto)
        desc: Description for progress bar

    Returns:
        List of processed results
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count() - 1, 4)

    results = [None] * len(scenes)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_fn, scene): idx for idx, scene in enumerate(scenes)
        }

        # Collect results with progress bar
        with tqdm(total=len(scenes), desc=desc) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error processing scene {idx}: {e}")
                    results[idx] = None
                pbar.update(1)

    return results


def batch_save_tiles(
    tiles: List[np.ndarray],
    save_paths: List[Path],
    metadata_list: List[Dict],
    batch_size: int = 100,
) -> None:
    """
    Save tiles in batches to reduce I/O overhead.

    Args:
        tiles: List of tile arrays
        save_paths: List of save paths
        metadata_list: List of metadata dicts
        batch_size: Number of tiles to save at once
    """
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i : i + batch_size]
        batch_paths = save_paths[i : i + batch_size]
        batch_metadata = metadata_list[i : i + batch_size]

        for tile, path, metadata in zip(batch_tiles, batch_paths, batch_metadata):
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data
            data = {"clean_norm": torch.from_numpy(tile).float(), **metadata}

            # Save with compression
            torch.save(data, path, pickle_protocol=4)


def create_memory_mapped_dataset(
    scenes: List[Path], output_file: Path, dtype: np.dtype = np.float32
) -> np.memmap:
    """
    Create memory-mapped dataset for efficient training.

    Args:
        scenes: List of scene file paths
        output_file: Output memmap file path
        dtype: Data type for storage

    Returns:
        Memory-mapped array
    """
    # Determine total size needed
    first_scene = torch.load(scenes[0], map_location="cpu")
    shape = (
        first_scene["clean_norm"].shape
        if "clean_norm" in first_scene
        else first_scene["noisy_norm"].shape
    )

    total_shape = (len(scenes),) + shape

    # Create memory-mapped file
    mmap = np.memmap(output_file, dtype=dtype, mode="w+", shape=total_shape)

    # Fill memory-mapped array
    for idx, scene_path in enumerate(tqdm(scenes, desc="Creating memmap")):
        data = torch.load(scene_path, map_location="cpu")

        if "clean_norm" in data:
            mmap[idx] = data["clean_norm"].numpy()
        else:
            mmap[idx] = data["noisy_norm"].numpy()

    # Flush to disk
    del mmap

    # Return read-only version
    return np.memmap(output_file, dtype=dtype, mode="r", shape=total_shape)


def parallel_compute_noise_params(
    image_paths: List[Path],
    load_fn: Callable,
    n_workers: int = None,
    max_images: int = 20,
) -> Tuple[float, float]:
    """
    Compute noise parameters in parallel from multiple images.

    Args:
        image_paths: List of image file paths
        load_fn: Function to load image from path
        n_workers: Number of workers
        max_images: Maximum images to use

    Returns:
        (gain, read_noise) tuple
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count() - 1, 4)

    # Sample images
    sample_paths = image_paths[: min(max_images, len(image_paths))]

    def extract_patches(path):
        """Extract uniform patches from single image."""
        try:
            image = load_fn(path)

            # Find uniform regions
            gradient = np.abs(np.gradient(image)[0]) + np.abs(np.gradient(image)[1])
            uniform_mask = gradient < np.percentile(gradient, 10)

            patches = []
            patch_size = 32

            # Extract patches
            for _ in range(10):  # Max 10 patches per image
                h = np.random.randint(0, image.shape[0] - patch_size)
                w = np.random.randint(0, image.shape[1] - patch_size)

                if uniform_mask[h : h + patch_size, w : w + patch_size].mean() > 0.8:
                    patch = image[h : h + patch_size, w : w + patch_size]
                    patches.append(patch)

            return patches
        except Exception as e:
            print(f"Error extracting patches from {path}: {e}")
            return []

    # Extract patches in parallel
    all_patches = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(extract_patches, path) for path in sample_paths]

        for future in futures:
            patches = future.result()
            all_patches.extend(patches)

    if len(all_patches) < 5:
        warnings.warn("Too few patches for noise estimation, using defaults")
        return 2.47, 1.82  # Default values

    # Compute statistics
    means = [p.mean() for p in all_patches]
    variances = [p.var(ddof=1) for p in all_patches]

    # Robust linear fit
    try:
        from sklearn.linear_model import HuberRegressor

        X = np.array(means).reshape(-1, 1)
        y = np.array(variances)

        reg = HuberRegressor()
        reg.fit(X, y)

        gain = reg.coef_[0]
        read_var_adu = reg.intercept_
        read_noise = np.sqrt(max(read_var_adu * gain, 0))

    except ImportError:
        # Fallback to simple linear fit
        coeffs = np.polyfit(means, variances, 1)
        gain = coeffs[0]
        read_noise = np.sqrt(max(coeffs[1] * gain, 0))

    return float(gain), float(read_noise)


# Export functions
__all__ = [
    "parallel_extract_tiles_with_augmentation",
    "memory_efficient_compute_global_scale",
    "parallel_process_scenes",
    "batch_save_tiles",
    "create_memory_mapped_dataset",
    "parallel_compute_noise_params",
]
