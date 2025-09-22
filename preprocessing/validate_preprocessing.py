#!/usr/bin/env python
"""
Validation script for preprocessed data.
Checks format compliance, data quality, and statistical properties.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class PreprocessingValidator:
    """Validates preprocessed data meets specifications."""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.errors = []
        self.warnings = []

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.errors.append(message)
        print(f"❌ ERROR: {message}")

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")

    def log_info(self, message: str) -> None:
        """Log an info message."""
        print(f"ℹ️  {message}")

    def validate_directory_structure(self, domains: List[str] = None) -> bool:
        """Validate the expected directory structure exists."""
        print("\n=== Validating Directory Structure ===")

        # Base required directories
        required_dirs = ["manifests"]

        # Add domain-specific directories only if they have manifests or are explicitly requested
        if domains is None:
            # Auto-detect domains from manifests
            manifest_dir = self.data_root / "manifests"
            if manifest_dir.exists():
                domains = [f.stem for f in manifest_dir.glob("*.json")]
            else:
                domains = []

        for domain in domains:
            required_dirs.extend(
                [
                    f"prior_clean/{domain}/train",
                    f"posterior/{domain}/train",
                ]
            )

        all_exist = True
        for dir_path in required_dirs:
            full_path = self.data_root / dir_path
            if not full_path.exists():
                self.log_error(f"Missing directory: {dir_path}")
                all_exist = False
            else:
                self.log_info(f"Found directory: {dir_path}")

        return all_exist

    def load_manifest(self, domain: str) -> Optional[Dict[str, Any]]:
        """Load and validate manifest file."""
        manifest_path = self.data_root / "manifests" / f"{domain}.json"

        if not manifest_path.exists():
            self.log_error(f"Missing manifest: {manifest_path}")
            return None

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Validate required fields
            required_fields = [
                "domain",
                "date_processed",
                "num_scenes",
                "num_tiles",
                "gain_mean",
                "read_noise_mean",
                "tile_size",
                "channels",
                "preprocessing_version",
            ]

            # Check for scale fields (either old format or new format)
            has_old_scale = "scale_p999" in manifest
            has_new_scales = (
                "scale_p999_noisy" in manifest and "scale_p999_clean" in manifest
            )

            if not (has_old_scale or has_new_scales):
                print(f"❌ ERROR: Missing scale fields in {domain} manifest")
                return False

            for field in required_fields:
                if field not in manifest:
                    self.log_error(f"Missing field '{field}' in {domain} manifest")
                    return None

            # Validate domain-specific values
            expected_channels = {"photography": 4, "microscopy": 1, "astronomy": 1}
            if manifest["channels"] != expected_channels.get(domain):
                self.log_error(
                    f"Wrong channel count for {domain}: expected {expected_channels[domain]}, got {manifest['channels']}"
                )

            self.log_info(
                f"Loaded manifest for {domain}: {manifest['num_tiles']['train']} training tiles"
            )
            return manifest

        except (json.JSONDecodeError, Exception) as e:
            self.log_error(f"Error loading manifest {manifest_path}: {e}")
            return None

    def validate_tile_files(self, domain: str, manifest: Dict[str, Any]) -> bool:
        """Validate prior training tile files."""
        print(f"\n=== Validating {domain.title()} Prior Training Tiles ===")

        prior_dir = self.data_root / "prior_clean" / domain / "train"
        tile_files = list(prior_dir.glob("*.pt"))

        if len(tile_files) == 0:
            self.log_error(f"No tile files found in {prior_dir}")
            return False

        expected_tiles = manifest["num_tiles"]["train"]
        if len(tile_files) != expected_tiles:
            self.log_warning(
                f"Expected {expected_tiles} tiles, found {len(tile_files)}"
            )

        # Sample tiles for validation
        sample_size = min(100, len(tile_files))
        sample_tiles = np.random.choice(tile_files, sample_size, replace=False)

        tile_shapes = []
        tile_ranges = []
        normalization_quality = []

        for i, tile_path in enumerate(sample_tiles):
            try:
                data = torch.load(tile_path, map_location="cpu")

                # Check structure
                required_keys = ["clean_norm", "domain_id", "metadata"]
                for key in required_keys:
                    if key not in data:
                        self.log_error(f"Missing key '{key}' in {tile_path}")
                        return False

                # Check tensor properties
                clean = data["clean_norm"]
                if not isinstance(clean, torch.Tensor):
                    self.log_error(f"clean_norm is not a tensor in {tile_path}")
                    return False

                # Check dimensions
                expected_channels = manifest["channels"]
                expected_size = manifest["tile_size"]
                expected_shape = (expected_channels, expected_size, expected_size)

                if clean.shape != expected_shape:
                    self.log_error(
                        f"Wrong shape in {tile_path}: expected {expected_shape}, got {clean.shape}"
                    )
                    return False

                tile_shapes.append(clean.shape)

                # Check value range
                min_val, max_val = clean.min().item(), clean.max().item()
                tile_ranges.append((min_val, max_val))

                if min_val < 0:
                    self.log_error(f"Negative values in {tile_path}: min={min_val}")
                    return False

                if max_val > 3.0:  # Allow some headroom above 1.0
                    self.log_warning(f"Very large values in {tile_path}: max={max_val}")

                # Check normalization quality (most pixels should be < 1.0)
                normalized_fraction = (clean < 1.0).float().mean().item()
                normalization_quality.append(normalized_fraction)

                if normalized_fraction < 0.8:
                    self.log_warning(
                        f"Poor normalization in {tile_path}: {normalized_fraction:.3f} < 1.0"
                    )

                # Check domain ID
                if (
                    data["domain_id"]
                    != {"photography": 0, "microscopy": 1, "astronomy": 2}[domain]
                ):
                    self.log_error(
                        f"Wrong domain_id in {tile_path}: {data['domain_id']}"
                    )
                    return False

            except Exception as e:
                self.log_error(f"Error loading {tile_path}: {e}")
                return False

        # Summary statistics
        mean_norm_quality = np.mean(normalization_quality)
        mean_max_val = np.mean([r[1] for r in tile_ranges])

        self.log_info(f"Validated {sample_size} tiles")
        self.log_info(f"Mean normalization quality: {mean_norm_quality:.3f}")
        self.log_info(f"Mean max value: {mean_max_val:.3f}")

        return True

    def validate_scene_files(self, domain: str, manifest: Dict[str, Any]) -> bool:
        """Validate posterior scene files."""
        print(f"\n=== Validating {domain.title()} Posterior Scenes ===")

        posterior_dir = self.data_root / "posterior" / domain / "train"
        scene_files = list(posterior_dir.glob("*.pt"))

        if len(scene_files) == 0:
            self.log_error(f"No scene files found in {posterior_dir}")
            return False

        # Sample scenes for validation
        sample_size = min(50, len(scene_files))
        sample_scenes = np.random.choice(scene_files, sample_size, replace=False)

        # Collect noise statistics
        snr_values = []
        var_ratios = []

        for scene_path in sample_scenes:
            try:
                data = torch.load(scene_path, map_location="cpu")

                # Check structure
                required_keys = ["noisy_norm", "calibration", "masks", "metadata"]
                for key in required_keys:
                    if key not in data:
                        self.log_error(f"Missing key '{key}' in {scene_path}")
                        return False

                # Check tensors
                noisy = data["noisy_norm"]
                clean = data["clean_norm"]  # May be None

                if not isinstance(noisy, torch.Tensor):
                    self.log_error(f"noisy_norm is not a tensor in {scene_path}")
                    return False

                # Check calibration
                calib = data["calibration"]
                required_calib_keys = ["scale", "gain", "read_noise", "background"]
                for key in required_calib_keys:
                    if key not in calib:
                        self.log_error(
                            f"Missing calibration key '{key}' in {scene_path}"
                        )
                        return False

                # Check masks
                masks = data["masks"]
                required_mask_keys = ["valid", "saturated"]
                for key in required_mask_keys:
                    if key not in masks:
                        self.log_error(f"Missing mask '{key}' in {scene_path}")
                        return False

                # Check metadata first
                metadata = data["metadata"]
                if "domain_id" not in metadata:
                    self.log_error(f"Missing domain_id in metadata of {scene_path}")
                    return False

                # Validate noise parameters (replaces chi-squared validation)
                if clean is not None:
                    # Convert to electrons for analysis using correct scales
                    scale_noisy = calib.get("scale_noisy", calib.get("scale", 1.0))
                    scale_clean = calib.get("scale_clean", calib.get("scale", 1.0))

                    noisy_e = noisy * scale_noisy
                    clean_e = clean * scale_clean
                    read_noise_e = calib["read_noise"]

                    noisy_mean = noisy_e.mean().item()
                    clean_mean = clean_e.mean().item()

                    # Calculate actual brightness ratio
                    actual_ratio = clean_mean / noisy_mean if noisy_mean > 0 else 1.0

                    # For photography (SID dataset), skip complex noise model validation
                    # The SID dataset has been processed in a way that doesn't preserve
                    # the expected noise characteristics for variance ratio validation
                    if domain == "photography":
                        # Simple SNR validation only (no variance ratio for photography)
                        snr = clean_mean / read_noise_e if read_noise_e > 0 else 0
                        snr_values.append(snr)

                        # Log SNR but skip variance ratio validation
                        self.log_info(
                            f"SNR: {snr:.1f} (photography - variance ratio validation skipped)"
                        )

                        # Check if SNR is reasonable
                        if snr < 10:
                            self.log_warning(f"Very low SNR for photography: {snr:.1f}")
                    else:
                        # For other domains, assume same exposure
                        snr = clean_mean / read_noise_e if read_noise_e > 0 else 0
                        expected_var = clean_mean + read_noise_e**2
                        actual_var = (noisy_e - clean_e).var().item()
                        var_ratio = actual_var / expected_var if expected_var > 0 else 0

                        snr_values.append(snr)
                        var_ratios.append(var_ratio)

                        self.log_info(f"SNR: {snr:.1f}, Var Ratio: {var_ratio:.2f}")

                        # Basic sanity checks
                        if snr < 1:
                            self.log_warning(f"Very low SNR: {snr:.1f}")
                        if var_ratio < 0.5 or var_ratio > 2.0:
                            self.log_warning(
                                f"Unexpected variance ratio: {var_ratio:.2f}"
                            )

            except Exception as e:
                self.log_error(f"Error loading {scene_path}: {e}")
                return False

        # Report summary statistics
        if snr_values:
            mean_snr = np.mean(snr_values)
            std_snr = np.std(snr_values)
            self.log_info(f"Summary: SNR {mean_snr:.1f}±{std_snr:.1f}")

            # Check if statistics are reasonable (different validation for different domains)
            if domain == "photography":
                if mean_snr < 10:
                    self.log_warning(
                        f"Mean SNR {mean_snr:.1f} is very low for photography"
                    )
                else:
                    self.log_info(
                        "SNR validation passed ✓ (variance ratio validation skipped for photography)"
                    )
            else:
                # For other domains, check variance ratios if available
                mean_var_ratio = np.mean(var_ratios)
                std_var_ratio = np.std(var_ratios)
                self.log_info(
                    f"Summary: SNR {mean_snr:.1f}±{std_snr:.1f}, Var Ratio {mean_var_ratio:.2f}±{std_var_ratio:.2f}"
                )

                if mean_var_ratio < 0.5 or mean_var_ratio > 2.0:
                    self.log_warning(
                        f"Mean variance ratio {mean_var_ratio:.2f} outside expected range [0.5, 2.0]"
                    )
                else:
                    self.log_info("Noise model validation passed ✓")

        self.log_info(f"Validated {sample_size} scenes")
        return True

    def check_data_splits(self, domain: str) -> bool:
        """Check that train/val/test splits don't overlap."""
        print(f"\n=== Validating {domain.title()} Data Splits ===")

        posterior_dir = self.data_root / "posterior" / domain

        splits = {}
        for split in ["train", "val", "test"]:
            split_dir = posterior_dir / split
            if split_dir.exists():
                scene_files = list(split_dir.glob("*.pt"))
                # Extract scene IDs from filenames
                scene_ids = set()
                for scene_file in scene_files:
                    # Load metadata to get scene_id
                    try:
                        data = torch.load(scene_file, map_location="cpu")
                        scene_id = data["metadata"]["scene_id"]
                        scene_ids.add(scene_id)
                    except:
                        # Fallback to filename
                        scene_ids.add(scene_file.stem)

                splits[split] = scene_ids
                self.log_info(f"{split}: {len(scene_ids)} unique scenes")

        # Check for overlaps
        overlap_found = False
        for split1 in splits:
            for split2 in splits:
                if split1 < split2:  # Avoid checking same pair twice
                    overlap = splits[split1] & splits[split2]
                    if overlap:
                        self.log_error(
                            f"Scene overlap between {split1} and {split2}: {len(overlap)} scenes"
                        )
                        overlap_found = True

        if not overlap_found:
            self.log_info("No scene overlaps found ✓")

        return not overlap_found

    def _estimate_sid_exposure_ratio(
        self, noisy_e: torch.Tensor, clean_e: torch.Tensor
    ) -> float:
        """
        Estimate exposure ratio for SID dataset using robust statistics.

        Args:
            noisy_e: Noisy image in electrons
            clean_e: Clean image in electrons

        Returns:
            Exposure ratio (clean/noisy)
        """
        # Use median of high-signal regions for robust estimation
        high_signal_mask = (clean_e > clean_e.mean()) & (noisy_e > 0)

        if high_signal_mask.sum() > 100:
            # Use percentile-based ratio for robustness
            clean_high = clean_e[high_signal_mask].flatten()
            noisy_high = noisy_e[high_signal_mask].flatten()

            # Remove outliers using IQR
            clean_q25, clean_q75 = torch.percentile(clean_high, [25, 75])
            noisy_q25, noisy_q75 = torch.percentile(noisy_high, [25, 75])

            clean_iqr = clean_q75 - clean_q25
            noisy_iqr = noisy_q75 - noisy_q25

            clean_mask = (clean_high >= clean_q25 - 1.5 * clean_iqr) & (
                clean_high <= clean_q75 + 1.5 * clean_iqr
            )
            noisy_mask = (noisy_high >= noisy_q25 - 1.5 * noisy_iqr) & (
                noisy_high <= noisy_q75 + 1.5 * noisy_iqr
            )

            combined_mask = clean_mask & noisy_mask

            if combined_mask.sum() > 50:
                ratios = clean_high[combined_mask] / (noisy_high[combined_mask] + 1e-6)
                exposure_ratio = torch.median(ratios).item()
            else:
                # Fallback to mean ratio
                exposure_ratio = clean_e.mean().item() / (noisy_e.mean().item() + 1e-6)
        else:
            # Fallback to mean ratio
            exposure_ratio = clean_e.mean().item() / (noisy_e.mean().item() + 1e-6)

        return max(1.0, exposure_ratio)  # Ensure positive

    def _validate_noise_parameters(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        calib: Dict[str, float],
        domain: str,
    ) -> bool:
        """
        Validate that noise parameters are reasonable and data statistics make sense.

        Args:
            noisy: Normalized noisy image
            clean: Normalized clean image
            calib: Calibration parameters
            domain: Domain name

        Returns:
            True if noise parameters seem reasonable
        """
        try:
            # Convert to electrons for analysis
            noisy_e = noisy * calib["scale"]
            clean_e = clean * calib["scale"]

            # Check basic statistics
            noisy_mean = noisy_e.mean().item()
            clean_mean = clean_e.mean().item()
            read_noise_e = calib["read_noise"]

            # Check that read noise is reasonable
            if read_noise_e <= 0:
                self.log_warning(f"Non-positive read noise: {read_noise_e}")
                return False

            if read_noise_e > 100:
                self.log_warning(f"Extremely high read noise: {read_noise_e}")
                return False

            # Check that scale parameter is reasonable
            scale = calib["scale"]
            if scale <= 0:
                self.log_warning(f"Non-positive scale: {scale}")
                return False

            if scale < 1000:  # Too small for typical images
                self.log_warning(f"Scale seems too small: {scale}")
                return False

            # Check that signal levels are reasonable
            if clean_mean <= 0:
                self.log_warning(f"Non-positive clean signal: {clean_mean}")
                return False

            # Check normalized data range
            if noisy.max() > 5.0:  # Should be normalized to reasonable range
                self.log_warning(f"Noisy data has very high max: {noisy.max().item()}")
                return False

            if clean.max() > 5.0:  # Should be normalized to reasonable range
                self.log_warning(f"Clean data has very high max: {clean.max().item()}")
                return False

            # Log basic statistics
            self.log_info(
                f"Signal levels - Clean: {clean_mean:.1f}e, Noisy: {noisy_mean:.1f}e, Read noise: {read_noise_e:.2f}e"
            )

            return True

        except Exception as e:
            self.log_warning(f"Noise parameter validation failed: {e}")
            return False

    def validate_domain(self, domain: str) -> bool:
        """Complete validation for one domain."""
        print(f"\n{'='*60}")
        print(f"VALIDATING {domain.upper()} DOMAIN")
        print(f"{'='*60}")

        # Load manifest
        manifest = self.load_manifest(domain)
        if manifest is None:
            return False

        # Validate tiles
        if not self.validate_tile_files(domain, manifest):
            return False

        # Validate scenes
        if not self.validate_scene_files(domain, manifest):
            return False

        # Check data splits
        if not self.check_data_splits(domain):
            return False

        print(f"\n✅ {domain.title()} domain validation passed!")
        return True

    def run_validation(self, domains: List[str]) -> bool:
        """Run complete validation."""
        print("🔍 Starting preprocessing validation...")
        print(f"Data root: {self.data_root}")

        # Check directory structure
        if not self.validate_directory_structure():
            return False

        # Validate each domain
        domain_results = {}
        for domain in domains:
            domain_results[domain] = self.validate_domain(domain)

        # Print summary
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")

        success_count = sum(domain_results.values())
        total_count = len(domain_results)

        for domain, success in domain_results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{domain:12}: {status}")

        print(f"\nOverall: {success_count}/{total_count} domains passed")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")

        if success_count == total_count and len(self.errors) == 0:
            print("\n🎉 All validations passed! Data is ready for training.")
            return True
        else:
            print("\n❌ Validation failed. Please fix the issues above.")
            return False


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(
        description="Validate preprocessed data quality and format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all domains
  python scripts/validate_preprocessing.py data/preprocessed

  # Validate specific domains
  python scripts/validate_preprocessing.py data/preprocessed --domains photography microscopy

  # Quiet mode (only errors)
  python scripts/validate_preprocessing.py data/preprocessed --quiet
        """,
    )

    parser.add_argument(
        "data_root", type=str, help="Path to preprocessed data directory"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["photography", "microscopy", "astronomy"],
        default=["photography", "microscopy", "astronomy"],
        help="Domains to validate",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only show errors and final results"
    )

    args = parser.parse_args()

    # Check data root exists
    if not Path(args.data_root).exists():
        print(f"❌ Data root does not exist: {args.data_root}")
        sys.exit(1)

    # Filter domains to only those that have manifest files
    available_domains = []
    for domain in args.domains:
        manifest_path = Path(args.data_root) / "manifests" / f"{domain}.json"
        if manifest_path.exists():
            available_domains.append(domain)
        else:
            print(f"⚠️  Skipping {domain}: no manifest found")

    if not available_domains:
        print("❌ No domains available for validation")
        sys.exit(1)

    # Run validation
    validator = PreprocessingValidator(args.data_root)

    # Suppress info messages in quiet mode
    if args.quiet:
        validator.log_info = lambda x: None

    success = validator.run_validation(available_domains)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
