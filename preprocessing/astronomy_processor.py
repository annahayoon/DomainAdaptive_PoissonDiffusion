"""
Astronomy domain preprocessing for SDSS data.
Handles FITS files with cosmic ray detection and background estimation.
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
    create_valid_mask,
    estimate_background_level,
    extract_tiles_with_augmentation,
    save_preprocessed_scene,
    save_preprocessed_tile,
    split_scenes_by_ratio,
)

# Import parallel utilities if available
try:
    from .preprocessing_utils_parallel import (
        memory_efficient_compute_global_scale,
        parallel_extract_tiles_with_augmentation,
    )

    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

try:
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
except ImportError:
    fits = None
    sigma_clipped_stats = None

try:
    from scipy.ndimage import median_filter
except ImportError:
    median_filter = None


class AstronomyProcessor:
    """Processes SDSS FITS files for astronomy domain."""

    def __init__(
        self,
        raw_data_root: str,
        output_root: str,
        num_tiles_per_frame: int = 30,
        tile_size: int = 128,
        clean_data_root: Optional[str] = None,
        use_coadds: bool = False,
        split_seed: Optional[int] = None,
        use_parallel: bool = True,
        n_workers: Optional[int] = None,
    ):
        self.raw_data_root = Path(raw_data_root)
        self.output_root = Path(output_root)
        self.num_tiles_per_frame = num_tiles_per_frame
        self.tile_size = tile_size
        self.clean_data_root = Path(clean_data_root) if clean_data_root else None
        self.use_coadds = use_coadds
        self.split_seed = split_seed
        self.use_parallel = use_parallel and PARALLEL_AVAILABLE
        self.n_workers = n_workers

        if self.use_parallel and not PARALLEL_AVAILABLE:
            print(
                "Warning: Parallel utilities not available, falling back to sequential processing"
            )

        if fits is None:
            raise ImportError(
                "astropy is required for astronomy processing. Install with: pip install astropy"
            )

        if median_filter is None:
            raise ImportError(
                "scipy is required for cosmic ray detection. Install with: pip install scipy"
            )

        # Create output directories
        self.prior_dir = self.output_root / "prior_clean" / "astronomy"
        self.posterior_dir = self.output_root / "posterior" / "astronomy"
        self.manifest_dir = self.output_root / "manifests"

        for split in ["train", "val", "test"]:
            (self.prior_dir / split).mkdir(parents=True, exist_ok=True)
            (self.posterior_dir / split).mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def discover_sdss_frames(self) -> List[Path]:
        """
        Discover FITS frames (SDSS, Hubble, or other astronomy data).

        Expected structure:
        raw_data_root/
        ├── complete_catalogs/     # Hubble Legacy Fields data
        │   ├── hlsp_hlf_hst_*.fits
        │   └── ...
        ├── frame-r-001234-1-0123.fits (SDSS)
        └── ...

        Returns:
            List of FITS file paths
        """
        # First, check if there's a complete_catalogs subdirectory
        complete_catalogs_dir = self.raw_data_root / "complete_catalogs"
        search_dirs = [self.raw_data_root]

        if complete_catalogs_dir.exists():
            search_dirs.insert(0, complete_catalogs_dir)
            print(f"Found complete_catalogs directory, searching there first")

        # Look for various astronomy FITS file patterns
        frame_patterns = [
            "frame-*.fits",  # SDSS frames
            "fpC-*.fits",  # SDSS fpC format
            "hlsp_hlf_hst_*_sci.fits",  # Hubble science files
            "hlsp_hlf_hst_*.fits",  # All Hubble files
            "**/*.fits",  # Any FITS files recursively
        ]

        frames = []
        for pattern in frame_patterns:
            for search_dir in search_dirs:
                found_frames = list(search_dir.glob(pattern))
                frames.extend(found_frames)
                if found_frames:
                    print(
                        f"Found {len(found_frames)} files with pattern '{pattern}' in {search_dir}"
                    )
                    break  # Use first pattern that finds files
            if frames:
                break  # Use first pattern that finds files across all directories

        # Remove duplicates and sort
        frames = sorted(list(set(frames)))

        print(f"Discovered {len(frames)} astronomy frames total")
        return frames

    def discover_clean_frames(self) -> Dict[str, Path]:
        """
        Discover clean frames (coadds or external clean data).

        Returns:
            Dictionary mapping frame IDs to clean frame paths
        """
        if not self.clean_data_root or not self.clean_data_root.exists():
            print("No clean data directory specified or found")
            return {}

        clean_frames = {}

        if self.use_coadds:
            # Look for coadd files
            coadd_files = list(self.clean_data_root.glob("*coadd*.fits"))
            for coadd_file in coadd_files:
                # Extract field/run info to match with frames
                match = re.search(r"(\d+)", coadd_file.stem)
                if match:
                    field_id = match.group(1)
                    clean_frames[field_id] = coadd_file
        else:
            # Look for clean frames with similar naming
            clean_files = list(self.clean_data_root.glob("*.fits"))
            for clean_file in clean_files:
                # Try to match based on filename
                clean_frames[clean_file.stem] = clean_file

        print(f"Found {len(clean_frames)} clean frames")
        return clean_frames

    def load_fits_image(self, filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load FITS image and extract header information.

        Args:
            filepath: Path to FITS file

        Returns:
            data: Image data array
            header_info: Relevant header information
        """
        with fits.open(str(filepath)) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header

            # Determine if this is Hubble or SDSS data based on filename
            filename = filepath.name.lower()
            is_hubble = "hlsp_hlf_hst" in filename

            if is_hubble:
                # Hubble-specific header extraction
                header_info = {
                    "gain": header.get("GAIN", 2.0),  # Default Hubble gain
                    "read_noise": header.get(
                        "RDNOISE", 3.0
                    ),  # Default Hubble read noise
                    "exptime": header.get(
                        "EXPTIME", header.get("EXPOSURE", 1000.0)
                    ),  # Exposure time
                    "filter": header.get(
                        "FILTER", header.get("FILTNAM1", "unknown")
                    ),  # Filter
                    "airmass": header.get("AIRMASS", 1.0),  # Airmass
                    "seeing": header.get("SEEING", 1.0),  # Seeing in arcsec
                    "telescope": "HST",
                    "instrument": header.get("INSTRUME", "unknown"),
                }
            else:
                # SDSS-specific header extraction
                header_info = {
                    "gain": header.get("GAIN", 4.62),  # Default SDSS gain
                    "read_noise": header.get(
                        "RDNOISE", 4.72
                    ),  # Default SDSS read noise
                    "exptime": header.get("EXPTIME", 53.9),  # Exposure time
                    "filter": header.get("FILTER", "r"),  # Filter
                    "airmass": header.get("AIRMASS", 1.0),  # Airmass
                    "seeing": header.get("SEEING", 1.5),  # Seeing in arcsec
                    "telescope": "SDSS",
                    "instrument": "SDSS",
                }

        return data, header_info

    def detect_cosmic_rays(
        self, image: np.ndarray, read_noise: float, threshold_sigma: float = 5.0
    ) -> np.ndarray:
        """
        Simple cosmic ray detection using median filtering.

        Args:
            image: Input image
            read_noise: Read noise level
            threshold_sigma: Detection threshold in sigma

        Returns:
            Boolean mask where True indicates cosmic rays
        """
        # Apply median filter
        median_filtered = median_filter(image, size=5)

        # Compute residual
        residual = image - median_filtered

        # Detect outliers
        cosmic_rays = residual > threshold_sigma * read_noise

        # Morphological cleanup (remove single pixels)
        try:
            from scipy.ndimage import binary_opening, generate_binary_structure

            struct = generate_binary_structure(2, 1)  # 4-connected
            cosmic_rays = binary_opening(cosmic_rays, structure=struct)
        except ImportError:
            pass  # Skip morphological cleanup if scipy not available

        return cosmic_rays

    def estimate_sky_background(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Estimate sky background using sigma-clipped statistics.

        Args:
            image: Input image

        Returns:
            background: Background level
            background_rms: Background RMS
        """
        if sigma_clipped_stats is not None:
            # Use astropy's robust statistics
            mean, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
            return float(median), float(std)
        else:
            # Fallback to simple percentile-based estimation
            background = np.percentile(image, 50)  # Median
            mad = np.median(np.abs(image - background))
            background_rms = 1.4826 * mad  # Convert MAD to std
            return float(background), float(background_rms)

    def compute_normalization_scale(self, frames: List[Path]) -> float:
        """
        Compute normalization scale from sample frames.

        Args:
            frames: List of frame paths

        Returns:
            scale: Normalization scale in electrons
        """
        print("Computing global normalization scale...")

        sample_frames = frames[: min(10, len(frames))]  # Reduced for faster processing
        all_electrons = []

        for frame_path in tqdm(sample_frames, desc="Processing frames for scale"):
            try:
                data, header_info = self.load_fits_image(frame_path)

                # Convert to electrons (SDSS fpC files are in ADU, Hubble may be in electrons)
                if header_info.get("telescope") == "HST":
                    # Hubble data is typically already in electrons
                    electrons = data
                else:
                    # SDSS fpC files are in ADU, convert to electrons
                    electrons = data * header_info["gain"]

                # Add channel dimension
                electrons = electrons[np.newaxis, :, :]
                all_electrons.append(electrons)

            except Exception as e:
                print(f"Error processing {frame_path}: {e}")
                continue

        if not all_electrons:
            raise RuntimeError("No frames could be processed for scale computation")

        scale = compute_global_scale(all_electrons, percentile=99.9)
        print(f"Computed scale: {scale:.1f} electrons")

        return scale

    def process_all_frames(
        self, frames: List[Path], clean_frames: Dict[str, Path], scale: float
    ) -> int:
        """
        Process all frames and save preprocessed data.

        Args:
            frames: List of all frame paths
            clean_frames: Dictionary of clean frame paths
            scale: Normalization scale

        Returns:
            Total number of tiles generated
        """
        # Split frames into train/val/test using consistent seed across domains
        train_frames, val_frames, test_frames = split_scenes_by_ratio(
            frames,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=self.split_seed,
        )

        print(
            f"Split: {len(train_frames)} train, {len(val_frames)} val, {len(test_frames)} test"
        )

        tile_idx = 0
        frame_count = {"train": 0, "val": 0, "test": 0}

        for split_name, frame_list in [
            ("train", train_frames),
            ("val", val_frames),
            ("test", test_frames),
        ]:
            print(f"\nProcessing {split_name} split ({len(frame_list)} frames)...")

            for frame_id, frame_path in enumerate(tqdm(frame_list)):
                try:
                    # Load noisy frame
                    data, header_info = self.load_fits_image(frame_path)

                    # Convert to electrons
                    if header_info.get("telescope") == "HST":
                        # Hubble data is typically already in electrons
                        electrons = data
                    else:
                        # SDSS fpC files are in ADU, convert to electrons
                        electrons = data * header_info["gain"]

                    # Estimate background
                    background, background_rms = self.estimate_sky_background(electrons)

                    # Normalize
                    noisy_norm = electrons / scale
                    noisy_norm = noisy_norm[np.newaxis, :, :]  # Add channel dimension

                    # Detect cosmic rays
                    cosmic_rays = self.detect_cosmic_rays(
                        data, header_info["read_noise"], threshold_sigma=5.0
                    )

                    # Look for corresponding clean frame
                    clean_norm = None
                    frame_stem = frame_path.stem

                    # Try different matching strategies
                    clean_path = None
                    for key, clean_candidate in clean_frames.items():
                        if key in frame_stem or frame_stem in key:
                            clean_path = clean_candidate
                            break

                    if clean_path and clean_path.exists():
                        try:
                            clean_data, clean_header_info = self.load_fits_image(
                                clean_path
                            )
                            if clean_header_info.get("telescope") == "HST":
                                clean_electrons = clean_data
                            else:
                                clean_electrons = clean_data * clean_header_info["gain"]
                            clean_norm = clean_electrons / scale
                            clean_norm = clean_norm[np.newaxis, :, :]
                        except Exception as e:
                            print(f"Error loading clean frame {clean_path}: {e}")
                            clean_norm = None

                    # Create masks
                    valid_mask = create_valid_mask(noisy_norm.shape[1:], border_crop=10)

                    # Combine cosmic ray mask with valid mask
                    cosmic_ray_mask = cosmic_rays[np.newaxis, :, :]
                    valid_mask = valid_mask & ~cosmic_ray_mask

                    # Saturation mask (SDSS rarely saturates, but check high values)
                    saturated = (data > 60000)[np.newaxis, :, :]

                    masks = {
                        "valid": valid_mask,
                        "saturated": saturated,
                    }

                    # Prepare calibration
                    calibration = {
                        "scale": scale,
                        "gain": header_info["gain"],
                        "read_noise": header_info["read_noise"],
                        "background": background,
                        "black_level": 0.0,  # SDSS fpC files are bias-subtracted
                        "white_level": 65535.0,
                    }

                    # Prepare metadata
                    metadata = {
                        "domain_id": 2,  # Astronomy
                        "scene_id": f"frame_{frame_id:05d}",
                        "original_shape": list(data.shape),
                        "bit_depth": 16,
                        "filter": header_info.get("filter", "r"),
                        "exptime": header_info.get("exptime", 53.9),
                        "seeing": header_info.get("seeing", 1.5),
                    }

                    # Save full frame for posterior evaluation
                    scene_save_path = (
                        self.posterior_dir / split_name / f"frame_{frame_id:05d}.pt"
                    )
                    save_preprocessed_scene(
                        noisy_norm,
                        clean_norm,
                        calibration,
                        masks,
                        metadata,
                        scene_save_path,
                    )

                    frame_count[split_name] += 1

                    # Extract tiles for prior training (all splits)
                    # For astronomy, we can use the noisy data for prior training
                    # since we don't always have clean ground truth
                    # Use clean data if available, otherwise use noisy
                    tile_source = clean_norm if clean_norm is not None else noisy_norm

                    # Only extract tiles from valid regions
                    if valid_mask.sum() > (
                        tile_source.shape[1] * tile_source.shape[2] * 0.5
                    ):
                        if self.use_parallel:
                            tiles = parallel_extract_tiles_with_augmentation(
                                tile_source,
                                num_tiles=self.num_tiles_per_frame,
                                tile_size=self.tile_size,
                                augment=True,
                                min_signal_threshold=0.01,  # Very low threshold for astronomy
                                n_workers=self.n_workers,
                            )
                        else:
                            tiles = extract_tiles_with_augmentation(
                                tile_source,
                                num_tiles=self.num_tiles_per_frame,
                                tile_size=self.tile_size,
                                augment=True,
                                min_signal_threshold=0.01,  # Very low threshold for astronomy
                            )

                        for tile in tiles:
                            # Check if tile is mostly valid
                            h_start = np.random.randint(
                                0, valid_mask.shape[1] - self.tile_size + 1
                            )
                            w_start = np.random.randint(
                                0, valid_mask.shape[2] - self.tile_size + 1
                            )
                            tile_mask = valid_mask[
                                0,
                                h_start : h_start + self.tile_size,
                                w_start : w_start + self.tile_size,
                            ]

                            if tile_mask.mean() > 0.8:  # 80% valid pixels
                                tile_save_path = (
                                    self.prior_dir
                                    / split_name
                                    / f"tile_{tile_idx:06d}.pt"
                                )
                                save_preprocessed_tile(
                                    tile,
                                    domain_id=2,
                                    scene_id=f"frame_{frame_id:05d}",
                                    tile_idx=tile_idx,
                                    save_path=tile_save_path,
                                    augmented=True,
                                )
                                tile_idx += 1

                except Exception as e:
                    print(f"Error processing frame {frame_id}: {e}")
                    continue

        return tile_idx, frame_count

    def save_manifest(
        self,
        num_frames: Dict[str, int],
        num_tiles: int,
        scale: float,
        mean_gain: float,
        mean_read_noise: float,
    ) -> None:
        """Save preprocessing manifest."""
        manifest = {
            "domain": "astronomy",
            "date_processed": datetime.now().isoformat(),
            "num_scenes": num_frames,
            "num_tiles": {
                "train": num_frames["train"] * self.num_tiles_per_frame,
                "val": num_frames["val"] * self.num_tiles_per_frame,
                "test": num_frames["test"] * self.num_tiles_per_frame,
            },
            "scale_p999": scale,
            "gain_mean": mean_gain,
            "read_noise_mean": mean_read_noise,
            "tile_size": self.tile_size,
            "channels": 1,
            "preprocessing_version": "2.0",
        }

        manifest_path = self.manifest_dir / "astronomy.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest saved to {manifest_path}")

    def run_preprocessing(self) -> None:
        """Run complete preprocessing pipeline."""
        print("=== Astronomy Domain Preprocessing ===")

        # Step 1: Discover frames
        frames = self.discover_sdss_frames()
        if not frames:
            raise RuntimeError("No frames found")

        # Step 2: Discover clean frames (optional)
        clean_frames = self.discover_clean_frames()

        # Step 3: Compute normalization scale
        scale = self.compute_normalization_scale(frames)

        # Step 4: Process all frames
        total_tiles, frame_counts = self.process_all_frames(frames, clean_frames, scale)

        # Step 5: Save manifest
        # Use default SDSS values for gain and read noise
        mean_gain = 4.62
        mean_read_noise = 4.72

        self.save_manifest(frame_counts, total_tiles, scale, mean_gain, mean_read_noise)

        print(f"\n=== Astronomy preprocessing complete! ===")
        print(f"Generated {total_tiles} training tiles")
        print(f"Processed {sum(frame_counts.values())} frames total")
        if clean_frames:
            print(f"Used {len(clean_frames)} clean reference frames")


def main():
    """Main entry point for astronomy preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess astronomy domain (SDSS)")
    parser.add_argument(
        "--raw_root", type=str, required=True, help="Path to raw SDSS data directory"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to preprocessed output directory",
    )
    parser.add_argument(
        "--clean_root", type=str, help="Path to clean frames (coadds or external)"
    )
    parser.add_argument(
        "--use_coadds", action="store_true", help="Use coadd files as clean references"
    )
    parser.add_argument(
        "--num_tiles", type=int, default=30, help="Number of tiles per frame"
    )
    parser.add_argument(
        "--tile_size", type=int, default=128, help="Size of square tiles"
    )

    args = parser.parse_args()

    processor = AstronomyProcessor(
        raw_data_root=args.raw_root,
        output_root=args.output_root,
        clean_data_root=args.clean_root,
        use_coadds=args.use_coadds,
        num_tiles_per_frame=args.num_tiles,
        tile_size=args.tile_size,
    )

    processor.run_preprocessing()


if __name__ == "__main__":
    main()
