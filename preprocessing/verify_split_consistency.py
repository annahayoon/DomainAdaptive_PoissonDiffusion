#!/usr/bin/env python
"""
Verification script to check split consistency in preprocessed data.
Ensures that the same scenes appear in the same splits for both prior_clean and posterior.
"""

import json
from pathlib import Path
from typing import Dict, Set, Tuple

import torch


def extract_scene_id_from_tile(tile_path: Path) -> str:
    """Extract scene ID from a prior_clean tile file."""
    data = torch.load(tile_path, map_location="cpu")
    return data["metadata"]["scene_id"]


def extract_scene_id_from_posterior(posterior_path: Path) -> str:
    """Extract scene ID from a posterior scene file."""
    data = torch.load(posterior_path, map_location="cpu")
    return data["metadata"]["scene_id"]


def verify_domain_splits(domain: str, preprocessed_root: Path) -> Dict[str, Set[str]]:
    """
    Verify split consistency for a single domain.

    Returns:
        Dictionary mapping split name to set of scene IDs
    """
    prior_dir = preprocessed_root / "prior_clean" / domain
    posterior_dir = preprocessed_root / "posterior" / domain

    results = {}

    for split in ["train", "val", "test"]:
        print(f"\nChecking {domain}/{split}...")

        # Get scene IDs from prior_clean tiles
        prior_split_dir = prior_dir / split
        prior_scene_ids = set()

        if prior_split_dir.exists():
            tile_files = list(prior_split_dir.glob("tile_*.pt"))
            print(f"  Found {len(tile_files)} prior_clean tiles")

            for tile_path in tile_files[:100]:  # Sample first 100 tiles
                scene_id = extract_scene_id_from_tile(tile_path)
                prior_scene_ids.add(scene_id)

        # Get scene IDs from posterior scenes
        posterior_split_dir = posterior_dir / split
        posterior_scene_ids = set()

        if posterior_split_dir.exists():
            scene_files = list(posterior_split_dir.glob("*.pt"))
            print(f"  Found {len(scene_files)} posterior scenes")

            for scene_path in scene_files:
                scene_id = extract_scene_id_from_posterior(scene_path)
                posterior_scene_ids.add(scene_id)

        # Check consistency
        print(f"  Unique scenes in prior_clean: {len(prior_scene_ids)}")
        print(f"  Unique scenes in posterior: {len(posterior_scene_ids)}")

        # Verify that prior scenes are subset of posterior scenes
        if prior_scene_ids and posterior_scene_ids:
            if prior_scene_ids.issubset(posterior_scene_ids):
                print(f"  ✅ All prior_clean scenes found in posterior")
            else:
                missing = prior_scene_ids - posterior_scene_ids
                print(f"  ❌ Prior scenes not in posterior: {missing}")

        results[split] = posterior_scene_ids

    return results


def check_split_overlap(splits_dict: Dict[str, Set[str]]) -> bool:
    """
    Check if there's any overlap between train/val/test splits.

    Returns:
        True if no overlap, False if overlap exists
    """
    train_scenes = splits_dict.get("train", set())
    val_scenes = splits_dict.get("val", set())
    test_scenes = splits_dict.get("test", set())

    train_val_overlap = train_scenes & val_scenes
    train_test_overlap = train_scenes & test_scenes
    val_test_overlap = val_scenes & test_scenes

    no_overlap = True

    if train_val_overlap:
        print(f"  ❌ Train/Val overlap: {len(train_val_overlap)} scenes")
        no_overlap = False

    if train_test_overlap:
        print(f"  ❌ Train/Test overlap: {len(train_test_overlap)} scenes")
        no_overlap = False

    if val_test_overlap:
        print(f"  ❌ Val/Test overlap: {len(val_test_overlap)} scenes")
        no_overlap = False

    if no_overlap:
        print(f"  ✅ No overlap between splits!")

    return no_overlap


def main():
    """Run verification on all domains."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify split consistency in preprocessed data"
    )
    parser.add_argument(
        "--preprocessed_root",
        type=str,
        required=True,
        help="Path to preprocessed data root directory",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["photography", "microscopy", "astronomy", "all"],
        default="all",
        help="Domain to verify",
    )

    args = parser.parse_args()

    preprocessed_root = Path(args.preprocessed_root)
    if not preprocessed_root.exists():
        print(f"Error: Preprocessed root directory not found: {preprocessed_root}")
        return

    domains = (
        ["photography", "microscopy", "astronomy"]
        if args.domain == "all"
        else [args.domain]
    )

    print("=" * 60)
    print("SPLIT CONSISTENCY VERIFICATION")
    print("=" * 60)

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Domain: {domain.upper()}")
        print(f"{'='*60}")

        try:
            # Check split assignments
            splits = verify_domain_splits(domain, preprocessed_root)

            # Check for overlap
            print(f"\nChecking for split overlap in {domain}...")
            check_split_overlap(splits)

            # Load and display manifest stats
            manifest_path = preprocessed_root / "manifests" / f"{domain}.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)

                print(f"\nManifest statistics:")
                print(f"  Scenes: {manifest['num_scenes']}")
                print(f"  Scale: {manifest.get('scale_p999', 'N/A'):.1f}")

        except Exception as e:
            print(f"Error processing {domain}: {e}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
