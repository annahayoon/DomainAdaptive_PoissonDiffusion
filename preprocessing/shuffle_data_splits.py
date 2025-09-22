#!/usr/bin/env python
"""
Shuffle preprocessed data within train/val/test splits for better randomization.
This script reorganizes existing preprocessed files to create more random splits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cross_domain.data.preprocessing.preprocessing_utils import split_scenes_by_ratio


class DataShuffle:
    """Shuffle preprocessed data within train/val/test splits."""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def get_scene_files(self, domain: str, split: str) -> List[Path]:
        """Get all scene files for a domain and split."""
        posterior_dir = self.data_root / "posterior" / domain / split
        if posterior_dir.exists():
            return list(posterior_dir.glob("*.pt"))
        return []

    def get_tile_files(self, domain: str, split: str) -> List[Path]:
        """Get all tile files for a domain and split."""
        prior_dir = self.data_root / "prior_clean" / domain / split
        if prior_dir.exists():
            return list(prior_dir.glob("*.pt"))
        return []

    def get_all_scenes_by_domain(self, domain: str) -> Dict[str, List[Path]]:
        """Get all scene files organized by split for a domain."""
        scenes_by_split = {}
        for split in ["train", "val", "test"]:
            scenes = self.get_scene_files(domain, split)
            if scenes:
                scenes_by_split[split] = scenes
        return scenes_by_split

    def get_all_tiles_by_domain(self, domain: str) -> Dict[str, List[Path]]:
        """Get all tile files organized by split for a domain."""
        tiles_by_split = {}
        for split in ["train", "val", "test"]:
            tiles = self.get_tile_files(domain, split)
            if tiles:
                tiles_by_split[split] = tiles
        return tiles_by_split

    def shuffle_domain_splits(self, domain: str, seed: int = None) -> None:
        """Shuffle splits for a single domain."""
        print(f"Shuffling {domain} domain splits...")

        # Get all scene files for this domain
        scenes_by_split = self.get_all_scenes_by_domain(domain)
        tile_files_by_split = self.get_all_tiles_by_domain(domain)

        if not scenes_by_split:
            print(f"No scene files found for {domain}")
            return

        # Collect all scene files from all splits
        all_scenes = []
        for split, scenes in scenes_by_split.items():
            all_scenes.extend(scenes)

        print(f"Found {len(all_scenes)} total scene files")

        # Shuffle all scenes
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()  # Use system entropy
        np.random.shuffle(all_scenes)

        # Split scenes into new random splits
        train_scenes, val_scenes, test_scenes = split_scenes_by_ratio(
            all_scenes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=None
        )

        print(
            f"New split sizes: {len(train_scenes)} train, {len(val_scenes)} val, {len(test_scenes)} test"
        )

        # Create new directories if needed
        for split in ["train", "val", "test"]:
            for data_type in ["prior_clean", "posterior"]:
                split_dir = self.data_root / data_type / domain / split
                split_dir.mkdir(parents=True, exist_ok=True)

        # Move scene files to new splits
        print("Moving scene files to new splits...")
        self._move_files_to_splits(train_scenes, "train", "posterior", domain)
        self._move_files_to_splits(val_scenes, "val", "posterior", domain)
        self._move_files_to_splits(test_scenes, "test", "posterior", domain)

        # For tile files, we need to reorganize them based on which scenes they're from
        print("Reorganizing tile files based on new scene splits...")
        self._reorganize_tiles_by_scenes(train_scenes, val_scenes, test_scenes, domain)

        # Update manifest
        self._update_manifest(domain, train_scenes, val_scenes, test_scenes)

        print(f"✓ {domain} domain shuffling completed")

    def _move_files_to_splits(
        self, scene_files: List[Path], split: str, data_type: str, domain: str
    ) -> None:
        """Move scene files to the specified split."""
        target_dir = self.data_root / data_type / domain / split
        target_dir.mkdir(parents=True, exist_ok=True)

        for scene_file in scene_files:
            target_path = target_dir / scene_file.name
            if scene_file != target_path:
                shutil.move(str(scene_file), str(target_path))

    def _reorganize_tiles_by_scenes(
        self,
        train_scenes: List[Path],
        val_scenes: List[Path],
        test_scenes: List[Path],
        domain: str,
    ) -> None:
        """Reorganize tile files based on which scenes they belong to."""
        # Create mapping from scene_id to split
        scene_to_split = {}
        for split, scenes in [
            ("train", train_scenes),
            ("val", val_scenes),
            ("test", test_scenes),
        ]:
            for scene_file in scenes:
                # Extract scene ID from filename (e.g., scene_00001.pt -> scene_00001)
                scene_id = scene_file.stem.split("_")[1]  # Get the numeric part
                scene_to_split[scene_id] = split

        # Get all tile files for this domain
        all_tiles = []
        for split in ["train", "val", "test"]:
            tiles = self.get_tile_files(domain, split)
            all_tiles.extend(tiles)

        # Move tiles to their corresponding splits based on scene ID
        for tile_file in all_tiles:
            # Extract scene ID from tile filename (e.g., tile_000000.pt -> scene_00001)
            # The tile filename format is tile_XXXXXX.pt where XXXXXX is sequential
            # But we need to match it to the scene ID from the metadata
            try:
                tile_data = torch.load(tile_file, map_location="cpu")
                tile_scene_id = tile_data["metadata"]["scene_id"]
                scene_num = tile_scene_id.split("_")[1]  # Get numeric part

                if scene_num in scene_to_split:
                    target_split = scene_to_split[scene_num]
                    target_dir = self.data_root / "prior_clean" / domain / target_split
                    target_path = target_dir / tile_file.name

                    if tile_file != target_path:
                        shutil.move(str(tile_file), str(target_path))
                else:
                    print(
                        f"Warning: Could not find split for scene {scene_num} from tile {tile_file.name}"
                    )

            except Exception as e:
                print(f"Error processing tile {tile_file.name}: {e}")

    def _update_manifest(
        self,
        domain: str,
        train_scenes: List[Path],
        val_scenes: List[Path],
        test_scenes: List[Path],
    ) -> None:
        """Update the manifest file for the domain."""
        manifest_path = self.data_root / "manifests" / f"{domain}.json"

        if not manifest_path.exists():
            print(f"Warning: No manifest found for {domain}")
            return

        # Load existing manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Update scene counts
        manifest["num_scenes"] = {
            "train": len(train_scenes),
            "val": len(val_scenes),
            "test": len(test_scenes),
        }

        # Recalculate tile counts (assuming same tiles per scene as before)
        tiles_per_scene = manifest.get("num_tiles", {}).get("train", 0) // manifest.get(
            "num_scenes", {}
        ).get("train", 1)
        manifest["num_tiles"] = {
            "train": len(train_scenes) * tiles_per_scene,
            "val": len(val_scenes) * tiles_per_scene,
            "test": len(test_scenes) * tiles_per_scene,
        }

        # Update timestamp
        from datetime import datetime

        manifest["date_processed"] = datetime.now().isoformat()

        # Save updated manifest
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Updated manifest for {domain}")

    def shuffle_all_domains(self, domains: List[str] = None, seed: int = None) -> None:
        """Shuffle all domains."""
        if domains is None:
            # Auto-detect domains from manifests
            manifest_dir = self.data_root / "manifests"
            if manifest_dir.exists():
                domains = [f.stem for f in manifest_dir.glob("*.json")]
            else:
                domains = []

        print(f"Shuffling data splits for domains: {domains}")

        for domain in domains:
            try:
                self.shuffle_domain_splits(domain, seed)
            except Exception as e:
                print(f"Error shuffling {domain}: {e}")
                if seed is not None:  # Only show traceback in debug mode
                    import traceback

                    traceback.print_exc()

        print("✓ Data shuffling completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Shuffle preprocessed data within train/val/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Shuffle all domains
  python scripts/shuffle_data_splits.py data/preprocessed

  # Shuffle specific domains
  python scripts/shuffle_data_splits.py data/preprocessed --domains photography microscopy

  # Use specific random seed for reproducibility
  python scripts/shuffle_data_splits.py data/preprocessed --seed 42
        """,
    )

    parser.add_argument(
        "data_root", type=str, help="Path to preprocessed data directory"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["photography", "microscopy", "astronomy"],
        help="Domains to shuffle",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducible shuffling"
    )

    args = parser.parse_args()

    # Validate data root
    if not Path(args.data_root).exists():
        print(f"Error: Data root does not exist: {args.data_root}")
        sys.exit(1)

    # Create shuffler and run
    shuffler = DataShuffle(args.data_root)
    shuffler.shuffle_all_domains(args.domains, args.seed)

    # Exit with success
    sys.exit(0)


if __name__ == "__main__":
    main()
