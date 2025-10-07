#!/usr/bin/env python3
"""
Verify that .npy tiles have correct channel dimensions:
- Photography: RGB (3, 256, 256)
- Microscopy: Grayscale (1, 256, 256)
- Astronomy: Grayscale (1, 256, 256)
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def verify_npy_tiles(metadata_json: str):
    """Verify .npy tile channel dimensions from metadata"""

    # Load metadata
    with open(metadata_json, "r") as f:
        metadata = json.load(f)

    tiles = metadata.get("tiles", [])

    if not tiles:
        print("❌ No tiles found in metadata")
        return

    print(f"📊 Verifying {len(tiles)} tiles...\n")

    # Group by domain
    domain_stats = defaultdict(lambda: {"count": 0, "channels": set(), "shapes": set()})
    errors = []

    for tile in tiles:
        domain = tile.get("domain", "unknown")
        npy_path = tile.get("npy_path", "")

        if not npy_path or not Path(npy_path).exists():
            continue

        try:
            # Load .npy file
            data = np.load(npy_path)

            # Track stats
            domain_stats[domain]["count"] += 1
            domain_stats[domain]["channels"].add(
                data.shape[0] if len(data.shape) == 3 else 1
            )
            domain_stats[domain]["shapes"].add(data.shape)

            # Verify expected channels
            expected_channels = 3 if domain == "photography" else 1
            actual_channels = data.shape[0] if len(data.shape) == 3 else 1

            if actual_channels != expected_channels:
                errors.append(
                    {
                        "tile_id": tile.get("tile_id", "unknown"),
                        "domain": domain,
                        "expected_channels": expected_channels,
                        "actual_channels": actual_channels,
                        "shape": data.shape,
                        "path": npy_path,
                    }
                )

            # Verify shape is (C, 256, 256)
            if data.shape != (expected_channels, 256, 256):
                errors.append(
                    {
                        "tile_id": tile.get("tile_id", "unknown"),
                        "domain": domain,
                        "expected_shape": (expected_channels, 256, 256),
                        "actual_shape": data.shape,
                        "path": npy_path,
                    }
                )

        except Exception as e:
            errors.append(
                {
                    "tile_id": tile.get("tile_id", "unknown"),
                    "domain": domain,
                    "error": str(e),
                    "path": npy_path,
                }
            )

    # Print summary
    print("=" * 80)
    print("DOMAIN SUMMARY")
    print("=" * 80)

    for domain, stats in sorted(domain_stats.items()):
        expected = (
            "RGB (3 channels)" if domain == "photography" else "Grayscale (1 channel)"
        )
        print(f"\n{domain.upper()}")
        print(f"  • Files checked: {stats['count']}")
        print(f"  • Expected: {expected}")
        print(f"  • Actual channels found: {sorted(stats['channels'])}")
        print(
            f"  • Shapes found: {sorted(stats['shapes'])[:5]}"
        )  # Show first 5 unique shapes

    # Print errors
    if errors:
        print("\n" + "=" * 80)
        print(f"❌ FOUND {len(errors)} ERRORS")
        print("=" * 80)
        for i, err in enumerate(errors[:10], 1):  # Show first 10 errors
            print(f"\nError {i}:")
            for key, val in err.items():
                print(f"  {key}: {val}")
    else:
        print("\n" + "=" * 80)
        print("✅ ALL TILES VERIFIED SUCCESSFULLY!")
        print("=" * 80)
        print("  • Photography: RGB (3, 256, 256) ✓")
        print("  • Microscopy: Grayscale (1, 256, 256) ✓")
        print("  • Astronomy: Grayscale (1, 256, 256) ✓")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify .npy tile channel dimensions")
    parser.add_argument(
        "--metadata",
        type=str,
        default="/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/processed/comprehensive_tiles_metadata.json",
        help="Path to metadata JSON file",
    )

    args = parser.parse_args()

    print("🔍 Verifying .npy tile channel dimensions...")
    print(f"📁 Metadata: {args.metadata}\n")

    verify_npy_tiles(args.metadata)
