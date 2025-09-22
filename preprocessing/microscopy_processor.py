"""
Microscopy domain preprocessing for fluorescence microscopy data.
Handles MRC files with flat-field correction and noise estimation.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress MRC file format warnings (they're expected for some files)
warnings.filterwarnings("ignore", message="Map ID string not found")
warnings.filterwarnings("ignore", message="Unrecognised machine stamp")

import numpy as np
import torch
from tqdm import tqdm

from .preprocessing_utils import (
    compute_global_scale,
    create_valid_mask,
    estimate_background_level,
    estimate_noise_params_photon_transfer,
    extract_tiles_with_augmentation,
    save_preprocessed_scene,
    save_preprocessed_tile,
    split_scenes_by_ratio,
)

# Import parallel utilities if available
try:
    from .preprocessing_utils_parallel import (
        memory_efficient_compute_global_scale,
        parallel_compute_noise_params,
        parallel_extract_tiles_with_augmentation,
    )

    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

try:
    import mrcfile
except ImportError:
    mrcfile = None


try:
    from scipy.ndimage import gaussian_filter
    from skimage.morphology import disk, opening
except ImportError:
    gaussian_filter = None
    disk = None
    opening = None


class MicroscopyProcessor:
    """Processes fluorescence microscopy MRC files."""

    def __init__(
        self,
        raw_data_root: str,
        output_root: str,
        num_tiles_per_frame: int = 20,
        tile_size: int = 128,
        sample_size_for_estimation: int = 20,
        data_in_electrons: bool = False,
        split_seed: Optional[int] = None,
        use_parallel: bool = True,
        n_workers: Optional[int] = None,
    ):
        self.raw_data_root = Path(raw_data_root)
        self.output_root = Path(output_root)
        self.num_tiles_per_frame = num_tiles_per_frame
        self.tile_size = tile_size
        self.sample_size = sample_size_for_estimation
        self.data_in_electrons = data_in_electrons
        self.split_seed = split_seed
        self.use_parallel = use_parallel and PARALLEL_AVAILABLE
        self.n_workers = n_workers

        if self.use_parallel and not PARALLEL_AVAILABLE:
            print(
                "Warning: Parallel utilities not available, falling back to sequential processing"
            )

        if mrcfile is None:
            raise ImportError(
                "mrcfile is required for microscopy processing. Install with: pip install mrcfile"
            )

        # Create output directories - use preprocessed structure
        self.prior_dir = self.output_root / "prior_clean" / "microscopy"
        self.posterior_dir = self.output_root / "posterior" / "microscopy"
        self.manifest_dir = self.output_root / "manifests"

        for split in ["train", "val", "test"]:
            (self.prior_dir / split).mkdir(parents=True, exist_ok=True)
            (self.posterior_dir / split).mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def _find_gt_and_level_files(self, cell_dir: Path) -> Tuple[List[Path], List[Path]]:
        """
        Find ground truth and level files handling different directory structures.

        Args:
            cell_dir: Path to cell directory

        Returns:
            (gt_files, level_files) tuple
        """
        gt_files = []
        level_files = []

        # Pattern 1: Direct files with _gt.mrc suffix (CCPs, F-actin, Microtubules, F-actin_Nonlinear)
        direct_gt_files = list(cell_dir.glob("*gt*.mrc"))
        direct_level_files = sorted(cell_dir.glob("*level_*.mrc"))

        if direct_gt_files:
            gt_files.extend(direct_gt_files)
        if direct_level_files:
            level_files.extend(direct_level_files)

        # Pattern 2: Subdirectory structure (ER)
        if not gt_files or not level_files:
            # Check for GTSIM subdirectory (use level_01 as ground truth)
            gtsim_dir = cell_dir / "GTSIM"
            if gtsim_dir.exists():
                gtsim_files = list(gtsim_dir.glob("*level_01.mrc"))
                if gtsim_files and not gt_files:
                    gt_files.extend(gtsim_files)

            # Check for RawGTSIMData subdirectory (use level_01 as ground truth)
            rawgtsim_dir = cell_dir / "RawGTSIMData"
            if rawgtsim_dir.exists():
                rawgtsim_files = list(rawgtsim_dir.glob("*level_01.mrc"))
                if rawgtsim_files and not gt_files:
                    gt_files.extend(rawgtsim_files)

            # Check for RawSIMData subdirectory (noise levels)
            rawsim_dir = cell_dir / "RawSIMData"
            if rawsim_dir.exists():
                rawsim_level_files = sorted(rawsim_dir.glob("*level_*.mrc"))
                if rawsim_level_files and not level_files:
                    level_files.extend(rawsim_level_files)

        return gt_files, level_files

    def _apply_flat_field_correction(
        self, image: np.ndarray, flat_field: np.ndarray
    ) -> np.ndarray:
        """
        Apply flat field correction with robust shape handling.

        Args:
            image: Input image
            flat_field: Flat field correction

        Returns:
            Corrected image
        """
        try:
            if image.shape == flat_field.shape:
                return image / flat_field
            else:
                # Try to resize flat field to match image
                try:
                    from scipy.ndimage import zoom

                    scale_y = image.shape[0] / flat_field.shape[0]
                    scale_x = image.shape[1] / flat_field.shape[1]
                    flat_field_resized = zoom(flat_field, (scale_y, scale_x), order=1)
                    # Ensure proper shape match after resize
                    if flat_field_resized.shape != image.shape:
                        # Crop or pad to exact match
                        min_h = min(image.shape[0], flat_field_resized.shape[0])
                        min_w = min(image.shape[1], flat_field_resized.shape[1])
                        flat_field_resized = flat_field_resized[:min_h, :min_w]
                        image = image[:min_h, :min_w]

                    # Ensure no division by zero
                    flat_field_resized = np.maximum(flat_field_resized, 0.1)
                    return image / flat_field_resized

                except ImportError:
                    print(
                        "Warning: scipy not available, skipping flat field correction"
                    )
                    return image
                except Exception as e:
                    print(f"Warning: Failed to apply flat field correction: {e}")
                    return image
        except Exception as e:
            print(f"Warning: Flat field correction failed: {e}")
            return image

    def discover_microscopy_sequences(self) -> List[Tuple[List[Path], List[Path]]]:
        """
        Discover microscopy sequences with paired noisy/clean images.

        Supports MRC data structure:
        raw_data_root/structures/
        ├── CCPs/Cell_001/
        │   ├── RawSIMData_gt.mrc (clean)
        │   ├── RawSIMData_level_01.mrc (noisy)
        │   └── RawSIMData_level_XX.mrc (various noise levels)

        Returns:
            List of (noisy_paths, clean_paths) tuples per sequence
        """
        sequences = []

        # Look for MRC structure
        structures_dir = self.raw_data_root / "structures"
        if structures_dir.exists():
            return self._discover_mrc_sequences(structures_dir)
        else:
            raise FileNotFoundError(
                f"No MRC structure found at {structures_dir}. "
                f"Expected directory structure: {self.raw_data_root}/structures/[CCPs|ER|F-actin|Microtubules]/Cell_XXX/"
            )

    def _discover_mrc_sequences(
        self, structures_dir: Path
    ) -> List[Tuple[List[Path], List[Path]]]:
        """Discover MRC sequences in super-resolution data structure."""
        sequences = []

        # Look through each structure type (CCPs, ER, F-actin, etc.)
        for structure_dir in sorted(structures_dir.glob("*")):
            if not structure_dir.is_dir():
                continue

            print(f"Processing structure: {structure_dir.name}")

            # Look through each cell
            for cell_dir in sorted(structure_dir.glob("Cell_*")):
                if not cell_dir.is_dir():
                    continue

                # Find ground truth and noise level files using comprehensive search
                gt_files, level_files = self._find_gt_and_level_files(cell_dir)

                if not gt_files:
                    print(f"Warning: No ground truth file found in {cell_dir}")
                    continue

                if not level_files:
                    print(f"Warning: No noise level files found in {cell_dir}")
                    continue

                # Use first ground truth file as clean reference
                clean_file = gt_files[0]

                # Create pairs with each noise level
                for level_file in level_files:
                    sequences.append(([level_file], [clean_file]))

                print(f"Found {len(level_files)} noise levels for {cell_dir.name}")

        print(f"Discovered {len(sequences)} MRC sequence pairs")
        return sequences

    def load_mrc_image(self, filepath: Path) -> np.ndarray:
        """Load MRC image and ensure proper format."""
        with mrcfile.open(str(filepath), mode="r", permissive=True) as mrc:
            image = mrc.data.astype(np.float32)

        # Ensure 2D (some MRCs might have extra dimensions)
        while image.ndim > 2:
            if image.shape[0] == 1:
                image = image[0]
            elif image.shape[-1] == 1:
                image = image[..., 0]
            else:
                # Take first channel if multi-channel
                image = image[0] if image.shape[0] < image.shape[-1] else image[..., 0]

        return image

    def load_image(self, filepath: Path) -> np.ndarray:
        """Load MRC image file."""
        if filepath.suffix.lower() == ".mrc":
            return self.load_mrc_image(filepath)
        else:
            raise ValueError(
                f"Unsupported file format: {filepath.suffix}. Only MRC files are supported."
            )

    def estimate_flat_field(
        self, sequences: List[Tuple[List[Path], List[Path]]]
    ) -> np.ndarray:
        """
        Estimate flat field correction from multiple clean images.

        Args:
            sequences: Sample sequences for estimation

        Returns:
            flat_field: Normalized flat field correction
        """
        print("Estimating flat field correction...")

        sample_sequences = sequences[: min(5, len(sequences))]
        clean_images = []

        for _, clean_paths in sample_sequences:
            for clean_path in clean_paths[:5]:  # Max 5 frames per sequence
                try:
                    image = self.load_image(clean_path)
                    clean_images.append(image)
                except Exception as e:
                    print(f"Error loading {clean_path}: {e}")
                    continue

        if not clean_images:
            print("Warning: No images for flat field estimation, using uniform field")
            # Return uniform flat field
            dummy_image = self.load_image(sequences[0][1][0])
            return np.ones_like(dummy_image)

        # Compute median of all clean images for flat field
        # First, ensure all images have the same shape
        min_h = min(img.shape[0] for img in clean_images)
        min_w = min(img.shape[1] for img in clean_images)

        cropped_images = []
        for img in clean_images:
            h, w = img.shape
            h_start = (h - min_h) // 2
            w_start = (w - min_w) // 2
            cropped = img[h_start : h_start + min_h, w_start : w_start + min_w]
            cropped_images.append(cropped)

        # Stack and compute median
        image_stack = np.stack(cropped_images, axis=0)
        flat_field = np.median(image_stack, axis=0)

        # Normalize to avoid division by zero
        flat_field = flat_field / np.percentile(flat_field, 90)
        flat_field = np.maximum(flat_field, 0.1)  # Avoid division by very small numbers

        print(f"Computed flat field from {len(clean_images)} images")
        return flat_field

    def estimate_noise_parameters_microscopy(
        self, sequences: List[Tuple[List[Path], List[Path]]], flat_field: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate gain and read noise for microscopy data.

        Args:
            sequences: Sample sequences
            flat_field: Flat field correction

        Returns:
            gain: Gain (1.0 if data already in electrons)
            read_noise: Read noise in electrons
        """
        if self.data_in_electrons:
            print("Data assumed to be in electrons, estimating read noise...")

            # Estimate read noise from dark regions
            dark_pixels = []
            sample_sequences = sequences[: self.sample_size]

            for noisy_paths, _ in sample_sequences:
                for noisy_path in noisy_paths[:3]:  # Max 3 frames per sequence
                    try:
                        image = self.load_image(noisy_path)

                        # Apply flat field correction with robust shape handling
                        image = self._apply_flat_field_correction(image, flat_field)

                        # Find dark regions (bottom 10% of intensities)
                        dark_threshold = np.percentile(image, 10)
                        dark_mask = image < dark_threshold

                        if dark_mask.sum() > 100:
                            dark_pixels.extend(image[dark_mask])

                    except Exception as e:
                        print(f"Error processing {noisy_path}: {e}")
                        continue

            if dark_pixels:
                read_noise = np.std(dark_pixels)
                print(f"Estimated read noise: {read_noise:.3f} e-")
            else:
                read_noise = 5.0  # Default
                print("Using default read noise: 5.0 e-")

            return 1.0, read_noise

        else:
            print("Estimating gain and read noise from ADU data...")

            # Collect sample images for photon transfer
            sample_images = []
            sample_sequences = sequences[: self.sample_size]

            for noisy_paths, _ in sample_sequences:
                for noisy_path in noisy_paths[:3]:
                    try:
                        image = self.load_image(noisy_path)

                        # Apply flat field correction with robust shape handling
                        image = self._apply_flat_field_correction(image, flat_field)

                        sample_images.append(image)

                    except Exception as e:
                        print(f"Error loading {noisy_path}: {e}")
                        continue

            if len(sample_images) < 5:
                print("Warning: Too few samples, using defaults")
                return 1.0, 5.0

            # Use photon transfer method
            gain, read_noise = estimate_noise_params_photon_transfer(
                sample_images, min_patches=3, patch_size=32
            )

            print(f"Estimated gain: {gain:.3f}")
            print(f"Estimated read noise: {read_noise:.3f} e-")

            return gain, read_noise

    def compute_normalization_scale(
        self,
        sequences: List[Tuple[List[Path], List[Path]]],
        flat_field: np.ndarray,
        gain: float,
    ) -> float:
        """Compute normalization scale from clean images."""
        print("Computing global normalization scale...")

        clean_images_electrons = []
        sample_sequences = sequences[: min(20, len(sequences))]

        for _, clean_paths in tqdm(sample_sequences, desc="Processing clean images"):
            for clean_path in clean_paths[:5]:  # Max 5 frames per sequence
                try:
                    image = self.load_image(clean_path)

                    # Apply flat field correction with shape adaptation
                    if image.shape == flat_field.shape:
                        image = image / flat_field
                    else:
                        try:
                            from scipy.ndimage import zoom

                            scale_y = image.shape[0] / flat_field.shape[0]
                            scale_x = image.shape[1] / flat_field.shape[1]
                            flat_field_resized = zoom(
                                flat_field, (scale_y, scale_x), order=1
                            )
                            image = image / flat_field_resized
                        except ImportError:
                            pass  # Skip flat field correction if scipy not available
                        except Exception:
                            pass  # Skip flat field correction on error

                    # Convert to electrons
                    electrons = image * gain

                    # Add channel dimension for compatibility
                    electrons = electrons[np.newaxis, :, :]
                    clean_images_electrons.append(electrons)

                except Exception as e:
                    print(f"Error processing {clean_path}: {e}")
                    continue

        if not clean_images_electrons:
            raise RuntimeError(
                "No clean images could be processed for scale computation"
            )

        scale = compute_global_scale(clean_images_electrons, percentile=99.9)
        print(f"Computed scale: {scale:.1f} electrons")

        return scale

    def process_all_sequences(
        self,
        sequences: List[Tuple[List[Path], List[Path]]],
        flat_field: np.ndarray,
        gain: float,
        read_noise: float,
        scale: float,
    ) -> int:
        """Process all sequences and save preprocessed data."""

        # Split sequences into train/val/test using consistent seed across domains
        train_seqs, val_seqs, test_seqs = split_scenes_by_ratio(
            sequences,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=self.split_seed,
        )

        print(
            f"Split: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test sequences"
        )
        print(f"Sample train sequences: {train_seqs[:3] if train_seqs else 'None'}")
        print(f"Sample val sequences: {val_seqs[:3] if val_seqs else 'None'}")
        print(f"Sample test sequences: {test_seqs[:3] if test_seqs else 'None'}")

        tile_idx = 0
        frame_count = {"train": 0, "val": 0, "test": 0}

        for split_name, seq_list in [
            ("train", train_seqs),
            ("val", val_seqs),
            ("test", test_seqs),
        ]:
            print(f"\nProcessing {split_name} split ({len(seq_list)} sequences)...")

            # Create random sequence IDs for this split to avoid chronological ordering
            random_seq_ids = np.random.permutation(len(seq_list))
            local_seq_id_idx = 0

            for local_seq_id, (noisy_paths, clean_paths) in enumerate(tqdm(seq_list)):
                # Get random sequence ID for this sequence within the split
                current_seq_id = random_seq_ids[local_seq_id_idx]
                local_seq_id_idx += 1

                # Process each frame pair in the sequence
                for local_frame_id, (noisy_path, clean_path) in enumerate(
                    zip(noisy_paths, clean_paths)
                ):
                    try:
                        # Load images
                        noisy = self.load_image(noisy_path)
                        clean = self.load_image(clean_path)

                        # Apply flat field correction to both images
                        noisy = self._apply_flat_field_correction(noisy, flat_field)
                        clean = self._apply_flat_field_correction(clean, flat_field)

                        # Convert to electrons
                        noisy_electrons = noisy * gain
                        clean_electrons = clean * gain

                        # Normalize
                        noisy_norm = noisy_electrons / scale
                        clean_norm = clean_electrons / scale

                        # Add channel dimension (microscopy is single channel)
                        noisy_norm = noisy_norm[np.newaxis, :, :]
                        clean_norm = clean_norm[np.newaxis, :, :]

                        # Estimate background
                        background = estimate_background_level(
                            noisy_electrons, method="morphology", morphology_size=30
                        )

                        # Create masks
                        valid_mask = create_valid_mask(
                            noisy_norm.shape[1:], border_crop=5
                        )

                        # Saturation mask (typical for 16-bit TIFF)
                        sat_threshold = (
                            65000 if noisy.max() > 1000 else 0.95
                        )  # Handle normalized data
                        saturated = (noisy > sat_threshold)[np.newaxis, :, :]

                        masks = {
                            "valid": valid_mask,
                            "saturated": saturated,
                        }

                        # Prepare calibration
                        calibration = {
                            "scale": scale,
                            "gain": gain,
                            "read_noise": read_noise,
                            "background": background,
                            "black_level": 100.0,  # Typical for sCMOS
                            "white_level": 65535.0,
                        }

                        # Prepare metadata
                        metadata = {
                            "domain_id": 1,  # Microscopy
                            "scene_id": f"seq{current_seq_id:03d}_frame{local_frame_id:03d}",
                            "original_shape": list(noisy.shape),
                            "bit_depth": 16,
                        }

                        # Save full frame for posterior evaluation
                        scene_save_path = (
                            self.posterior_dir
                            / split_name
                            / f"seq{current_seq_id:03d}_frame{local_frame_id:03d}.pt"
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

                        # Extract and save clean tiles for prior training (all splits)
                        if self.use_parallel:
                            tiles = parallel_extract_tiles_with_augmentation(
                                clean_norm,
                                num_tiles=self.num_tiles_per_frame,
                                tile_size=self.tile_size,
                                augment=True,
                                min_signal_threshold=0.02,  # Lower threshold for microscopy
                                n_workers=self.n_workers,
                            )
                        else:
                            tiles = extract_tiles_with_augmentation(
                                clean_norm,
                                num_tiles=self.num_tiles_per_frame,
                                tile_size=self.tile_size,
                                augment=True,
                                min_signal_threshold=0.02,  # Lower threshold for microscopy
                            )

                        for tile in tiles:
                            tile_save_path = (
                                self.prior_dir / split_name / f"tile_{tile_idx:06d}.pt"
                            )
                            save_preprocessed_tile(
                                tile,
                                domain_id=1,
                                scene_id=f"seq{current_seq_id:03d}_frame{local_frame_id:03d}",
                                tile_idx=tile_idx,
                                save_path=tile_save_path,
                                augmented=True,
                            )
                            tile_idx += 1

                    except Exception as e:
                        print(
                            f"Error processing seq{current_seq_id}_frame{local_frame_id}: {e}"
                        )
                        continue

        print(f"Processing complete. Final counts: {frame_count}")

        return tile_idx, frame_count

    def save_manifest(
        self,
        num_frames: Dict[str, int],
        num_tiles: int,
        scale: float,
        gain: float,
        read_noise: float,
    ) -> None:
        """Save preprocessing manifest."""
        manifest = {
            "domain": "microscopy",
            "date_processed": datetime.now().isoformat(),
            "num_scenes": num_frames,
            "num_tiles": {
                "train": num_frames["train"] * self.num_tiles_per_frame,
                "val": num_frames["val"] * self.num_tiles_per_frame,
                "test": num_frames["test"] * self.num_tiles_per_frame,
            },
            "scale_p999": scale,
            "gain_mean": gain,
            "read_noise_mean": read_noise,
            "tile_size": self.tile_size,
            "channels": 1,
            "preprocessing_version": "2.0",
        }

        manifest_path = self.manifest_dir / "microscopy.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest saved to {manifest_path}")

    def run_preprocessing(self) -> None:
        """Run complete preprocessing pipeline."""
        print("=== Microscopy Domain Preprocessing ===")

        # Step 1: Discover sequences
        sequences = self.discover_microscopy_sequences()
        if not sequences:
            raise RuntimeError("No sequences found")

        # Step 2: Estimate flat field
        flat_field = self.estimate_flat_field(sequences)

        # Step 3: Estimate noise parameters
        gain, read_noise = self.estimate_noise_parameters_microscopy(
            sequences, flat_field
        )

        # Step 4: Compute normalization scale
        scale = self.compute_normalization_scale(sequences, flat_field, gain)

        # Step 5: Process all sequences
        total_tiles, frame_counts = self.process_all_sequences(
            sequences, flat_field, gain, read_noise, scale
        )

        # Step 6: Save manifest
        self.save_manifest(frame_counts, total_tiles, scale, gain, read_noise)

        print(f"\n=== Microscopy preprocessing complete! ===")
        print(f"Generated {total_tiles} training tiles")
        print(f"Processed {sum(frame_counts.values())} frames total")


def main():
    """Main entry point for microscopy preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess microscopy domain")
    parser.add_argument(
        "--raw_root",
        type=str,
        required=True,
        help="Path to raw microscopy data directory",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to preprocessed output directory",
    )
    parser.add_argument(
        "--num_tiles", type=int, default=20, help="Number of tiles per frame"
    )
    parser.add_argument(
        "--tile_size", type=int, default=128, help="Size of square tiles"
    )
    parser.add_argument(
        "--data_in_electrons",
        action="store_true",
        help="Data is already in electron units",
    )

    args = parser.parse_args()

    processor = MicroscopyProcessor(
        raw_data_root=args.raw_root,
        output_root=args.output_root,
        num_tiles_per_frame=args.num_tiles,
        tile_size=args.tile_size,
        data_in_electrons=args.data_in_electrons,
    )

    processor.run_preprocessing()


if __name__ == "__main__":
    main()
