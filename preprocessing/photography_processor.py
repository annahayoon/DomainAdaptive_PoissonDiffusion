"""
Photography domain preprocessing for SID dataset.
Handles RAW Sony ARW files with Bayer pattern processing.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .preprocessing_utils import (
    compute_global_scale,
    create_saturation_mask,
    create_valid_mask,
    estimate_background_level,
    estimate_noise_params_photon_transfer,
    extract_tiles_with_augmentation,
    pack_bayer_to_channels,
    save_preprocessed_scene,
    save_preprocessed_tile,
    split_scenes_by_ratio,
)

# Import parallel utilities if available
try:
    from .preprocessing_utils_parallel import (
        batch_save_tiles,
        memory_efficient_compute_global_scale,
        parallel_compute_noise_params,
        parallel_extract_tiles_with_augmentation,
    )

    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

try:
    import rawpy
except ImportError:
    rawpy = None


class PhotographyProcessor:
    """Processes Sony ARW files from SID dataset."""

    def __init__(
        self,
        raw_data_root: str,
        output_root: str,
        num_tiles_per_scene: int = 50,
        tile_size: int = 128,
        sample_size_for_estimation: int = 20,
        split_seed: Optional[int] = None,
        use_parallel: bool = True,
        n_workers: Optional[int] = None,
    ):
        self.raw_data_root = Path(raw_data_root)
        self.output_root = Path(output_root)
        self.num_tiles_per_scene = num_tiles_per_scene
        self.tile_size = tile_size
        self.sample_size = sample_size_for_estimation
        self.split_seed = split_seed
        self.use_parallel = use_parallel and PARALLEL_AVAILABLE
        self.n_workers = n_workers

        if self.use_parallel and not PARALLEL_AVAILABLE:
            print(
                "Warning: Parallel utilities not available, falling back to sequential processing"
            )

        if rawpy is None:
            raise ImportError(
                "rawpy is required for photography processing. Install with: pip install rawpy"
            )

        # Create output directories
        self.prior_dir = self.output_root / "prior_clean" / "photography"
        self.posterior_dir = self.output_root / "posterior" / "photography"
        self.manifest_dir = self.output_root / "manifests"

        for split in ["train", "val", "test"]:
            (self.prior_dir / split).mkdir(parents=True, exist_ok=True)
            (self.posterior_dir / split).mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def discover_sid_scenes(self) -> List[Tuple[Path, Path]]:
        """
        Discover SID scene pairs (noisy, clean).

        Returns:
            List of (noisy_path, clean_path) tuples
        """
        short_dir = self.raw_data_root / "Sony" / "short"
        long_dir = self.raw_data_root / "Sony" / "long"

        if not short_dir.exists() or not long_dir.exists():
            raise FileNotFoundError(
                f"SID directories not found: {short_dir}, {long_dir}"
            )

        # Find all short exposure files
        short_files = list(short_dir.glob("*.ARW"))
        scene_pairs = []

        for short_file in short_files:
            # Extract scene ID from filename (e.g., "00001_00_0.04s.ARW")
            match = re.match(r"(\d+_\d+)_.*\.ARW", short_file.name)
            if not match:
                continue

            scene_id = match.group(1)

            # Find corresponding long exposure file
            long_candidates = list(long_dir.glob(f"{scene_id}_*.ARW"))
            if len(long_candidates) == 1:
                scene_pairs.append((short_file, long_candidates[0]))
            elif len(long_candidates) > 1:
                # Take the longest exposure
                long_file = max(
                    long_candidates, key=lambda x: self._extract_exposure_time(x.name)
                )
                scene_pairs.append((short_file, long_file))

        # Shuffle scene pairs to ensure random order for proper splits
        np.random.seed()  # Use system entropy for random seed
        np.random.shuffle(scene_pairs)

        print(f"Found {len(scene_pairs)} SID scene pairs")
        return scene_pairs

    def _extract_exposure_time(self, filename: str) -> float:
        """Extract exposure time from filename."""
        match = re.search(r"(\d+(?:\.\d+)?)s", filename)
        if match:
            return float(match.group(1))
        return 0.0

    def _extract_iso_from_filename(self, filename: str) -> int:
        """Extract ISO from filename or return default."""
        # SID dataset doesn't encode ISO in filename, use metadata if available
        return 1600  # Typical for SID dataset

    def load_sony_raw(
        self, filepath: Path
    ) -> Tuple[np.ndarray, np.ndarray, float, str]:
        """
        Load Sony ARW file and extract raw data.

        Args:
            filepath: Path to ARW file

        Returns:
            bayer: Raw Bayer data [H, W]
            black_level: Black level per channel [4]
            white_level: White level scalar
            pattern: Color pattern string
        """
        with rawpy.imread(str(filepath)) as raw:
            # Get raw Bayer data (no demosaicing)
            bayer = raw.raw_image_visible.astype(np.float32)

            # Get calibration data
            black_level = np.array(raw.black_level_per_channel, dtype=np.float32)
            white_level = float(raw.white_level)

            # Get color pattern
            pattern_array = raw.raw_pattern

            # Convert pattern array to string
            if np.array_equal(pattern_array, [[0, 1], [1, 2]]):
                pattern = "RGGB"
            elif np.array_equal(pattern_array, [[1, 0], [2, 1]]):
                pattern = "GRBG"
            elif np.array_equal(pattern_array, [[2, 1], [1, 0]]):
                pattern = "BGGR"
            elif np.array_equal(pattern_array, [[1, 2], [0, 1]]):
                pattern = "GBRG"
            else:
                pattern = "RGGB"  # Default assumption

        return bayer, black_level, white_level, pattern

    def process_bayer_image(
        self,
        bayer: np.ndarray,
        black_level: np.ndarray,
        white_level: float,
        pattern: str = "RGGB",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process Bayer image: subtract black level and pack to 4-channel.

        Args:
            bayer: Raw Bayer data [H, W]
            black_level: Black level per channel [4]
            white_level: White level
            pattern: Color pattern

        Returns:
            packed: 4-channel image [4, H//2, W//2]
            sat_mask: Saturation mask [1, H//2, W//2]
        """
        H, W = bayer.shape

        # Create black level map
        black_map = np.zeros((H, W), dtype=np.float32)

        if pattern == "RGGB":
            black_map[0::2, 0::2] = black_level[0]  # R
            black_map[0::2, 1::2] = black_level[1]  # G1
            black_map[1::2, 0::2] = black_level[2]  # G2
            black_map[1::2, 1::2] = black_level[3]  # B
        else:
            # For other patterns, use average black level
            black_map.fill(black_level.mean())

        # Subtract black level
        bayer_corrected = np.maximum(bayer - black_map, 0)

        # Pack to 4-channel format
        packed = pack_bayer_to_channels(bayer_corrected, pattern)

        # Create saturation mask
        saturated = create_saturation_mask(bayer, white_level, black_map)

        # Downsample saturation mask to match packed image
        sat_mask = np.zeros((1, H // 2, W // 2), dtype=bool)
        sat_mask[0] = (
            saturated[0::2, 0::2]
            | saturated[0::2, 1::2]
            | saturated[1::2, 0::2]
            | saturated[1::2, 1::2]
        )

        return packed, sat_mask

    def estimate_noise_parameters(
        self, scene_pairs: List[Tuple[Path, Path]]
    ) -> Tuple[float, float]:
        """
        Estimate gain and read noise from sample scenes.

        Args:
            scene_pairs: List of (noisy_path, clean_path) pairs

        Returns:
            gain: Gain in electrons per ADU
            read_noise: Read noise in electrons
        """
        print("Estimating noise parameters from sample images...")

        sample_pairs = scene_pairs[: self.sample_size]
        sample_images = []
        sample_blacks = []

        for noisy_path, _ in tqdm(sample_pairs, desc="Loading samples"):
            try:
                bayer, black_level, white_level, pattern = self.load_sony_raw(
                    noisy_path
                )
                sample_images.append(bayer)

                # Create black level map
                H, W = bayer.shape
                black_map = np.zeros((H, W), dtype=np.float32)
                if pattern == "RGGB":
                    black_map[0::2, 0::2] = black_level[0]
                    black_map[0::2, 1::2] = black_level[1]
                    black_map[1::2, 0::2] = black_level[2]
                    black_map[1::2, 1::2] = black_level[3]
                else:
                    black_map.fill(black_level.mean())

                sample_blacks.append(black_map)
            except Exception as e:
                print(f"Error loading {noisy_path}: {e}")
                continue

        if len(sample_images) < 5:
            print("Warning: Too few samples for noise estimation, using defaults")
            return 2.47, 1.82  # Sony A7S defaults

        gain, read_noise = estimate_noise_params_photon_transfer(
            sample_images, sample_blacks, min_patches=20
        )

        print(f"Estimated gain: {gain:.3f} e-/ADU")
        print(f"Estimated read noise: {read_noise:.3f} e-")

        return gain, read_noise

    def compute_normalization_scale(
        self, scene_pairs: List[Tuple[Path, Path]], gain: float, use_noisy: bool = False
    ) -> float:
        """
        Compute global normalization scale from noisy or clean images.

        Args:
            scene_pairs: List of scene pairs
            gain: Gain in e-/ADU
            use_noisy: If True, compute scale from noisy images; otherwise from clean

        Returns:
            scale: Normalization scale in electrons
        """
        print(
            f"Computing global normalization scale ({'noisy' if use_noisy else 'clean'})..."
        )

        if self.use_parallel:
            # Memory-efficient streaming computation
            sample_pairs = scene_pairs[: min(100, len(scene_pairs))]

            def image_generator():
                for noisy_path, clean_path in sample_pairs:
                    try:
                        target_path = noisy_path if use_noisy else clean_path
                        bayer, black_level, white_level, pattern = self.load_sony_raw(
                            target_path
                        )
                        packed, _ = self.process_bayer_image(
                            bayer, black_level, white_level, pattern
                        )
                        electrons = packed * gain
                        yield electrons
                    except Exception as e:
                        print(f"Error processing {target_path}: {e}")
                        continue

            scale = memory_efficient_compute_global_scale(
                image_generator(),
                num_images=len(sample_pairs),
                percentile=99.9,
                batch_size=10,
                sample_pixels_per_image=100000,
            )
        else:
            # Original implementation
            sample_pairs = scene_pairs[: min(100, len(scene_pairs))]
            images_electrons = []

            for noisy_path, clean_path in tqdm(
                sample_pairs,
                desc=f"Processing {'noisy' if use_noisy else 'clean'} images",
            ):
                try:
                    target_path = noisy_path if use_noisy else clean_path
                    bayer, black_level, white_level, pattern = self.load_sony_raw(
                        target_path
                    )
                    packed, _ = self.process_bayer_image(
                        bayer, black_level, white_level, pattern
                    )

                    # Convert to electrons
                    electrons = packed * gain
                    images_electrons.append(electrons)

                except Exception as e:
                    print(f"Error processing {target_path}: {e}")
                    continue

            if not images_electrons:
                raise RuntimeError(
                    f"No {'noisy' if use_noisy else 'clean'} images could be processed for scale computation"
                )

            scale = compute_global_scale(images_electrons, percentile=99.9)

        print(
            f"Computed {'noisy' if use_noisy else 'clean'} scale: {scale:.1f} electrons"
        )
        return scale

    def process_all_scenes(
        self,
        scene_pairs: List[Tuple[Path, Path]],
        gain: float,
        read_noise: float,
        noisy_scale: float,
        clean_scale: float,
    ) -> int:
        """
        Process all scenes and save preprocessed data.

        Args:
            scene_pairs: List of all scene pairs
            gain: Gain in e-/ADU
            read_noise: Read noise in e-
            scale: Normalization scale in e-

        Returns:
            Total number of tiles generated
        """
        # Split scenes into train/val/test using consistent seed across domains
        train_scenes, val_scenes, test_scenes = split_scenes_by_ratio(
            scene_pairs,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=self.split_seed,
        )

        print(
            f"Split: {len(train_scenes)} train, {len(val_scenes)} val, {len(test_scenes)} test"
        )

        # Initialize tile index counters for each split
        tile_indices = {"train": 0, "val": 0, "test": 0}

        for split_name, scene_list in [
            ("train", train_scenes),
            ("val", val_scenes),
            ("test", test_scenes),
        ]:
            print(f"\nProcessing {split_name} split ({len(scene_list)} scenes)...")

            # Create random scene IDs for this split to avoid chronological ordering
            random_scene_ids = np.random.permutation(len(scene_list))
            local_scene_id_idx = 0

            for local_scene_id, (noisy_path, clean_path) in enumerate(tqdm(scene_list)):
                # Get random scene ID for this scene within the split
                current_scene_id = random_scene_ids[local_scene_id_idx]
                local_scene_id_idx += 1

                try:
                    # Load and process noisy image
                    noisy_bayer, black_n, white_n, pattern_n = self.load_sony_raw(
                        noisy_path
                    )
                    noisy_packed, sat_mask = self.process_bayer_image(
                        noisy_bayer, black_n, white_n, pattern_n
                    )
                    noisy_electrons = noisy_packed * gain
                    noisy_norm = noisy_electrons / noisy_scale

                    # Load and process clean image
                    clean_bayer, black_c, white_c, pattern_c = self.load_sony_raw(
                        clean_path
                    )
                    clean_packed, _ = self.process_bayer_image(
                        clean_bayer, black_c, white_c, pattern_c
                    )
                    clean_electrons = clean_packed * gain
                    clean_norm = clean_electrons / clean_scale

                    # Estimate background
                    background = estimate_background_level(
                        noisy_electrons, method="percentile"
                    )

                    # Create masks
                    valid_mask = create_valid_mask(noisy_norm.shape[1:])
                    masks = {
                        "valid": valid_mask,
                        "saturated": sat_mask,
                    }

                    # Prepare calibration with backward compatibility
                    # Primary scale field for backward compatibility
                    calibration = {
                        "scale": clean_scale,  # Use clean scale as primary for consistency
                        "gain": gain,
                        "read_noise": read_noise,
                        "background": background,
                        "black_level": float(black_n.mean()),
                        "white_level": float(white_n),
                    }

                    # Add dual scale information for advanced usage
                    # These are optional fields that new models can use
                    calibration["scale_noisy"] = noisy_scale
                    calibration["scale_clean"] = clean_scale

                    # Prepare metadata
                    metadata = {
                        "domain_id": 0,  # Photography
                        "scene_id": f"scene_{current_scene_id:05d}",
                        "original_shape": list(noisy_bayer.shape),
                        "bit_depth": 14,
                        "iso": self._extract_iso_from_filename(noisy_path.name),
                        "noisy_exposure_ms": self._extract_exposure_time(
                            noisy_path.name
                        )
                        * 1000,
                        "clean_exposure_ms": self._extract_exposure_time(
                            clean_path.name
                        )
                        * 1000,
                        "exposure_ms": self._extract_exposure_time(noisy_path.name)
                        * 1000,  # Keep for backward compatibility
                    }

                    # Save full scene for posterior evaluation
                    scene_save_path = (
                        self.posterior_dir
                        / split_name
                        / f"scene_{current_scene_id:05d}.pt"
                    )
                    save_preprocessed_scene(
                        noisy_norm,
                        clean_norm,
                        calibration,
                        masks,
                        metadata,
                        scene_save_path,
                    )

                    # Extract and save clean tiles for prior training (all splits)
                    if self.use_parallel:
                        tiles = parallel_extract_tiles_with_augmentation(
                            clean_norm,
                            num_tiles=self.num_tiles_per_scene,
                            tile_size=self.tile_size,
                            augment=True,
                            min_signal_threshold=0.05,
                            n_workers=self.n_workers,
                        )
                    else:
                        tiles = extract_tiles_with_augmentation(
                            clean_norm,
                            num_tiles=self.num_tiles_per_scene,
                            tile_size=self.tile_size,
                            augment=True,
                            min_signal_threshold=0.05,
                        )

                    for tile in tiles:
                        tile_save_path = (
                            self.prior_dir
                            / split_name
                            / f"tile_{tile_indices[split_name]:06d}.pt"
                        )
                        save_preprocessed_tile(
                            tile,
                            domain_id=0,
                            scene_id=f"scene_{current_scene_id:05d}",
                            tile_idx=tile_indices[split_name],
                            save_path=tile_save_path,
                            augmented=True,
                        )
                        tile_indices[split_name] += 1

                except Exception as e:
                    print(f"Error processing scene {current_scene_id}: {e}")
                    continue

        # Return total number of tiles generated across all splits
        return sum(tile_indices.values())

    def save_manifest(
        self,
        num_scenes: Dict[str, int],
        num_tiles: int,
        noisy_scale: float,
        clean_scale: float,
        gain: float,
        read_noise: float,
    ) -> None:
        """Save preprocessing manifest."""
        # Calculate expected tile counts per split based on actual scenes processed
        expected_tiles_per_split = {
            "train": num_scenes["train"] * self.num_tiles_per_scene,
            "val": num_scenes["val"] * self.num_tiles_per_scene,
            "test": num_scenes["test"] * self.num_tiles_per_scene,
        }

        manifest = {
            "domain": "photography",
            "date_processed": datetime.now().isoformat(),
            "num_scenes": num_scenes,
            "num_tiles": expected_tiles_per_split,
            "total_tiles": num_tiles,
            "scale_p999_noisy": noisy_scale,
            "scale_p999_clean": clean_scale,
            "gain_mean": gain,
            "read_noise_mean": read_noise,
            "tile_size": self.tile_size,
            "channels": 4,
            "preprocessing_version": "2.0",
        }

        manifest_path = self.manifest_dir / "photography.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest saved to {manifest_path}")

    def run_preprocessing(self) -> None:
        """Run complete preprocessing pipeline."""
        print("=== Photography Domain Preprocessing ===")

        # Step 1: Discover scenes
        scene_pairs = self.discover_sid_scenes()
        if not scene_pairs:
            raise RuntimeError("No scene pairs found")

        # Step 2: Estimate noise parameters
        gain, read_noise = self.estimate_noise_parameters(scene_pairs)

        # Step 3: Compute normalization scales for noisy and clean images
        noisy_scale = self.compute_normalization_scale(
            scene_pairs, gain, use_noisy=True
        )
        clean_scale = self.compute_normalization_scale(
            scene_pairs, gain, use_noisy=False
        )

        # Step 4: Process all scenes and get actual split counts
        total_tiles = self.process_all_scenes(
            scene_pairs, gain, read_noise, noisy_scale, clean_scale
        )

        # Get actual scene counts from the split
        train_scenes, val_scenes, test_scenes = split_scenes_by_ratio(
            scene_pairs,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=self.split_seed,
        )
        num_scenes = {
            "train": len(train_scenes),
            "val": len(val_scenes),
            "test": len(test_scenes),
        }

        # Step 5: Save manifest
        self.save_manifest(
            num_scenes, total_tiles, noisy_scale, clean_scale, gain, read_noise
        )

        print(f"\n=== Photography preprocessing complete! ===")
        print(f"Generated {total_tiles} training tiles")
        print(f"Processed {len(scene_pairs)} scenes total")


def main():
    """Main entry point for photography preprocessing."""
    import argparse
    import sys
    from pathlib import Path

    # Add parent directories to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    parser = argparse.ArgumentParser(
        description="Preprocess photography domain (SID dataset)"
    )
    parser.add_argument(
        "--raw_root", type=str, required=True, help="Path to raw SID data directory"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to preprocessed output directory",
    )
    parser.add_argument(
        "--num_tiles", type=int, default=50, help="Number of tiles per training scene"
    )
    parser.add_argument(
        "--tile_size", type=int, default=128, help="Size of square tiles"
    )

    args = parser.parse_args()

    processor = PhotographyProcessor(
        raw_data_root=args.raw_root,
        output_root=args.output_root,
        num_tiles_per_scene=args.num_tiles,
        tile_size=args.tile_size,
    )

    processor.run_preprocessing()


if __name__ == "__main__":
    main()
