#!/usr/bin/env python
"""
Main preprocessing script for all three domains.
Orchestrates photography, microscopy, and astronomy preprocessing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add the parent directory to the path to import cross_domain modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cross_domain.data.preprocessing.astronomy_processor import AstronomyProcessor
from cross_domain.data.preprocessing.microscopy_processor import MicroscopyProcessor
from cross_domain.data.preprocessing.photography_processor import PhotographyProcessor


def setup_directories(output_root: str) -> None:
    """Create the standard preprocessing directory structure."""
    output_path = Path(output_root)

    # Create main directories
    directories = [
        "manifests",
        "prior_clean/photography/train",
        "prior_clean/photography/val",
        "prior_clean/photography/test",
        "prior_clean/microscopy/train",
        "prior_clean/microscopy/val",
        "prior_clean/microscopy/test",
        "prior_clean/astronomy/train",
        "prior_clean/astronomy/val",
        "prior_clean/astronomy/test",
        "posterior/photography/train",
        "posterior/photography/val",
        "posterior/photography/test",
        "posterior/microscopy/train",
        "posterior/microscopy/val",
        "posterior/microscopy/test",
        "posterior/astronomy/train",
        "posterior/astronomy/val",
        "posterior/astronomy/test",
    ]

    for directory in directories:
        (output_path / directory).mkdir(parents=True, exist_ok=True)

    print(f"Created directory structure at {output_path}")


def preprocess_photography(
    raw_root: str, output_root: str, config: Dict[str, Any]
) -> bool:
    """Run photography domain preprocessing."""
    print("\n" + "=" * 60)
    print("PREPROCESSING PHOTOGRAPHY DOMAIN")
    print("=" * 60)

    try:
        processor = PhotographyProcessor(
            raw_data_root=raw_root,
            output_root=output_root,
            num_tiles_per_scene=config.get("photography_tiles_per_scene", 50),
            tile_size=config.get("tile_size", 128),
            sample_size_for_estimation=config.get("sample_size", 20),
            split_seed=config.get("split_seed", 42),
        )

        processor.run_preprocessing()
        print("✓ Photography preprocessing completed successfully")
        return True

    except Exception as e:
        print(f"✗ Photography preprocessing failed: {e}")
        if config.get("debug", False):
            import traceback

            traceback.print_exc()
        return False


def preprocess_microscopy(
    raw_root: str, output_root: str, config: Dict[str, Any]
) -> bool:
    """Run microscopy domain preprocessing."""
    print("\n" + "=" * 60)
    print("PREPROCESSING MICROSCOPY DOMAIN")
    print("=" * 60)

    try:
        processor = MicroscopyProcessor(
            raw_data_root=raw_root,
            output_root=output_root,
            num_tiles_per_frame=config.get("microscopy_tiles_per_frame", 20),
            tile_size=config.get("tile_size", 128),
            sample_size_for_estimation=config.get("sample_size", 20),
            data_in_electrons=config.get("microscopy_data_in_electrons", False),
            split_seed=config.get("split_seed", 42),
        )

        processor.run_preprocessing()
        print("✓ Microscopy preprocessing completed successfully")
        return True

    except Exception as e:
        print(f"✗ Microscopy preprocessing failed: {e}")
        if config.get("debug", False):
            import traceback

            traceback.print_exc()
        return False


def preprocess_astronomy(
    raw_root: str, output_root: str, config: Dict[str, Any]
) -> bool:
    """Run astronomy domain preprocessing."""
    print("\n" + "=" * 60)
    print("PREPROCESSING ASTRONOMY DOMAIN")
    print("=" * 60)

    try:
        processor = AstronomyProcessor(
            raw_data_root=raw_root,
            output_root=output_root,
            num_tiles_per_frame=config.get("astronomy_tiles_per_frame", 30),
            tile_size=config.get("tile_size", 128),
            clean_data_root=config.get("astronomy_clean_root"),
            use_coadds=config.get("astronomy_use_coadds", False),
            split_seed=config.get("split_seed", 42),
        )

        processor.run_preprocessing()
        print("✓ Astronomy preprocessing completed successfully")
        return True

    except Exception as e:
        print(f"✗ Astronomy preprocessing failed: {e}")
        if config.get("debug", False):
            import traceback

            traceback.print_exc()
        return False


def print_summary(results: Dict[str, bool], output_root: str) -> None:
    """Print preprocessing summary."""
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    success_count = sum(results.values())
    total_count = len(results)

    for domain, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{domain:12}: {status}")

    print(f"\nOverall: {success_count}/{total_count} domains completed successfully")

    if success_count == total_count:
        print(f"\n🎉 All preprocessing completed!")
        print(f"📁 Preprocessed data saved to: {output_root}")
        print(f"\nNext steps:")
        print(f"1. Run validation: python scripts/validate_preprocessing.py")
        print(f"2. Inspect samples: python scripts/inspect_data.py <path_to_pt_file>")
        print(f"3. Start training with preprocessed data")
    else:
        print(f"\n⚠️  Some domains failed. Check error messages above.")
        print(f"You can rerun individual domains or fix the issues and try again.")


def main():
    """Main preprocessing orchestrator."""
    parser = argparse.ArgumentParser(
        description="Preprocess all three domains for cross-domain diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess all domains with default settings
  python scripts/preprocess_all_domains.py \\
    --output_root data/preprocessed \\
    --photography_root data/raw/SID \\
    --microscopy_root data/raw/FMD \\
    --astronomy_root data/raw/astronomy

  # Skip domains that don't have data
  python scripts/preprocess_all_domains.py \\
    --output_root data/preprocessed \\
    --photography_root data/raw/SID \\
    --skip_microscopy \\
    --skip_astronomy

  # Custom tile settings and reproducible splits
  python scripts/preprocess_all_domains.py \\
    --output_root data/preprocessed \\
    --photography_root data/raw/SID \\
    --tile_size 256 \\
    --photography_tiles_per_scene 25 \\
    --split_seed 123
        """,
    )

    # Required arguments
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output directory for preprocessed data",
    )

    # Domain data roots
    parser.add_argument(
        "--photography_root",
        type=str,
        help="Path to raw photography data (SID dataset)",
    )
    parser.add_argument(
        "--microscopy_root", type=str, help="Path to raw microscopy data"
    )
    parser.add_argument(
        "--astronomy_root",
        type=str,
        help="Path to raw astronomy data (e.g., Hubble Legacy Fields)",
    )

    # Optional clean data for astronomy
    parser.add_argument(
        "--astronomy_clean_root", type=str, help="Path to clean astronomy data (coadds)"
    )
    parser.add_argument(
        "--astronomy_use_coadds",
        action="store_true",
        help="Use coadd files as clean references for astronomy",
    )

    # Skip options
    parser.add_argument(
        "--skip_photography",
        action="store_true",
        help="Skip photography domain preprocessing",
    )
    parser.add_argument(
        "--skip_microscopy",
        action="store_true",
        help="Skip microscopy domain preprocessing",
    )
    parser.add_argument(
        "--skip_astronomy",
        action="store_true",
        help="Skip astronomy domain preprocessing",
    )

    # Processing parameters
    parser.add_argument(
        "--tile_size", type=int, default=128, help="Size of square tiles for training"
    )
    parser.add_argument(
        "--photography_tiles_per_scene",
        type=int,
        default=50,
        help="Number of tiles to extract per photography scene",
    )
    parser.add_argument(
        "--microscopy_tiles_per_frame",
        type=int,
        default=20,
        help="Number of tiles to extract per microscopy frame",
    )
    parser.add_argument(
        "--astronomy_tiles_per_frame",
        type=int,
        default=30,
        help="Number of tiles to extract per astronomy frame",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20,
        help="Number of samples for noise parameter estimation",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for consistent train/val/test splits across domains",
    )

    # Microscopy-specific options
    parser.add_argument(
        "--microscopy_data_in_electrons",
        action="store_true",
        help="Microscopy data is already in electron units",
    )

    # General options
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with full error traces"
    )
    parser.add_argument(
        "--setup_only",
        action="store_true",
        help="Only create directory structure, don't process data",
    )

    args = parser.parse_args()

    # Validate arguments
    domains_to_process = []
    if not args.skip_photography and args.photography_root:
        domains_to_process.append("photography")
    if not args.skip_microscopy and args.microscopy_root:
        domains_to_process.append("microscopy")
    if not args.skip_astronomy and args.astronomy_root:
        domains_to_process.append("astronomy")

    if not domains_to_process:
        print("Error: No domains specified for processing!")
        print("Please provide data roots for at least one domain.")
        parser.print_help()
        sys.exit(1)

    # Setup output directories
    print("Setting up preprocessing directory structure...")
    setup_directories(args.output_root)

    if args.setup_only:
        print("Directory setup complete. Exiting (--setup_only specified).")
        return

    # Create configuration
    config = {
        "tile_size": args.tile_size,
        "photography_tiles_per_scene": args.photography_tiles_per_scene,
        "microscopy_tiles_per_frame": args.microscopy_tiles_per_frame,
        "astronomy_tiles_per_frame": args.astronomy_tiles_per_frame,
        "sample_size": args.sample_size,
        "microscopy_data_in_electrons": args.microscopy_data_in_electrons,
        "astronomy_clean_root": args.astronomy_clean_root,
        "astronomy_use_coadds": args.astronomy_use_coadds,
        "debug": args.debug,
        "split_seed": args.split_seed,
    }

    print(f"\nStarting preprocessing for domains: {', '.join(domains_to_process)}")
    print(f"Output directory: {args.output_root}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")

    # Process each domain
    results = {}

    if "photography" in domains_to_process:
        results["photography"] = preprocess_photography(
            args.photography_root, args.output_root, config
        )

    if "microscopy" in domains_to_process:
        results["microscopy"] = preprocess_microscopy(
            args.microscopy_root, args.output_root, config
        )

    if "astronomy" in domains_to_process:
        results["astronomy"] = preprocess_astronomy(
            args.astronomy_root, args.output_root, config
        )

    # Print summary
    print_summary(results, args.output_root)

    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Some failures


if __name__ == "__main__":
    main()
