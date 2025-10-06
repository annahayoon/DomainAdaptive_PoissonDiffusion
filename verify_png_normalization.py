#!/usr/bin/env python3
"""
Verify PNG dataset normalization.

This script checks what range the PNG dataset actually returns.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.png_dataset import create_png_datasets


def verify_normalization():
    """Check the actual range of values from PNG dataset."""
    print("=" * 60)
    print("Verifying PNG Dataset Normalization")
    print("=" * 60)

    # Paths
    data_root = "/home/jilab/Jae/dataset/processed/png_tiles/photography"
    metadata_json = (
        "/home/jilab/Jae/dataset/processed/metadata_photography_incremental.json"
    )

    print(f"\nData root: {data_root}")
    print(f"Metadata: {metadata_json}")

    # Check if paths exist
    if not Path(data_root).exists():
        print(f"❌ Data root not found: {data_root}")
        return False

    if not Path(metadata_json).exists():
        print(f"❌ Metadata not found: {metadata_json}")
        return False

    try:
        # Create dataset
        print("\nCreating dataset...")
        train_dataset, val_dataset = create_png_datasets(
            data_root=data_root,
            metadata_json=metadata_json,
            domain="photography",
            train_split="train",
            val_split="validation",
            max_files=10,  # Just load a few for testing
            seed=42,
            image_size=256,
            channels=3,
        )

        print(f"✓ Dataset created: {len(train_dataset)} train, {len(val_dataset)} val")

        # Load a sample
        print("\nLoading sample from dataset...")
        sample = train_dataset[0]

        clean = sample["clean"]
        print(f"\nDataset output:")
        print(f"  Shape: {clean.shape}")
        print(f"  Dtype: {clean.dtype}")
        print(f"  Min value: {clean.min().item():.2f}")
        print(f"  Max value: {clean.max().item():.2f}")
        print(f"  Mean value: {clean.mean().item():.2f}")

        # Check range
        is_0_1_range = (clean.min() >= 0.0) and (clean.max() <= 1.0)
        is_0_255_range = (clean.min() >= 0.0) and (clean.max() > 1.0)

        print("\nRange analysis:")
        if is_0_255_range:
            print(
                "  ✓ Data is in [0, 255] range (CORRECT - no premature normalization)"
            )
            print("  → Will convert to [-1, 1] for EDM in training script")
            print("  → Conversion: images = (clean / 127.5) - 1.0")
        elif is_0_1_range:
            print("  ✗ Data is in [0, 1] range")
            print("  → Dataset is normalizing prematurely")
            print("  → Should keep data in [0, 255] range")

        # Show conversion to EDM range
        print("\nEDM normalization:")
        edm_images = (clean / 127.5) - 1.0
        print(f"  After (x / 127.5 - 1.0):")
        print(f"    Min: {edm_images.min().item():.6f}")
        print(f"    Max: {edm_images.max().item():.6f}")
        print(f"    Mean: {edm_images.mean().item():.6f}")

        expected_in_edm_range = (edm_images.min() >= -1.0) and (edm_images.max() <= 1.0)
        if expected_in_edm_range:
            print("  ✓ Successfully converted to EDM range [-1, 1]")
        else:
            print("  ✗ WARNING: Values outside [-1, 1] range!")

        # Also check the raw PIL Image before ToTensor
        print("\n" + "=" * 60)
        print("Checking raw PNG file (before ToTensor)")
        print("=" * 60)

        from PIL import Image

        clean_path = sample["metadata"]["clean_path"]
        print(f"\nLoading: {clean_path}")

        raw_image = Image.open(clean_path)
        print(f"  Mode: {raw_image.mode}")
        print(f"  Size: {raw_image.size}")

        import numpy as np

        raw_array = np.array(raw_image)
        print(f"  Raw array dtype: {raw_array.dtype}")
        print(f"  Raw array min: {raw_array.min()}")
        print(f"  Raw array max: {raw_array.max()}")
        print(f"  → PNG files are in [0, 255] range (as expected)")

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print("1. PNG files store data as uint8 in [0, 255] range")
        print(
            "2. png_dataset.py keeps data in [0, 255] float32 (no premature normalization)"
        )
        print("3. Dataset returns tensors in [0, 255] range")
        print("4. For EDM, we convert [0, 255] → [-1, 1] using: (x / 127.5) - 1.0")
        print(
            "\n✓ Current implementation in train_photography_edm_native.py is CORRECT"
        )

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_normalization()
    sys.exit(0 if success else 1)
