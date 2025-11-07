#!/usr/bin/env python3
"""Extract metadata fields and pixel statistics from dataset files.

Supports:
1. Extracting key fields from SIDD metadata files
2. Extracting pixel statistics from comprehensive metadata JSON files
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.utils.file_utils import load_tensor_from_pt, save_json_file
from core.utils.tensor_utils import tensor_to_numpy


def extract_metadata_fields(file_path: Path) -> dict:
    """Extract key metadata fields from a SIDD metadata file.

    Args:
        file_path: Path to the metadata .MAT file

    Returns:
        Dictionary of extracted metadata fields, or None on error
    """
    try:
        data = loadmat(str(file_path), struct_as_record=False)
        metadata_struct = data["metadata"][0, 0]
        fields = {}

        for field in metadata_struct._fieldnames:
            val = getattr(metadata_struct, field)

            if isinstance(val, np.ndarray):
                if val.size == 0:
                    continue
                elif val.size == 1:
                    try:
                        if val.dtype.kind in ["i", "u", "f"]:
                            fields[field] = float(val.item())
                        else:
                            fields[field] = str(val.item())
                    except:
                        fields[field] = str(val.item())
                elif val.size > 1 and val.size < 20:
                    if val.dtype.kind in ["i", "u", "f"]:
                        fields[field] = val.flatten().tolist()
                    else:
                        fields[field] = [str(x) for x in val.flatten()]
                elif field in [
                    "BlackLevel",
                    "BlackLevels",
                    "WhiteLevel",
                    "ISO",
                    "ExposureTime",
                    "CFAPattern",
                    "BitsPerSample",
                    "Width",
                    "Height",
                ]:
                    if val.dtype.kind in ["i", "u", "f"]:
                        fields[field] = (
                            float(val.item())
                            if val.size == 1
                            else val.flatten().tolist()
                        )
                    else:
                        fields[field] = (
                            str(val.item())
                            if val.size == 1
                            else [str(x) for x in val.flatten()]
                        )

        return fields
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_pixel_stats_from_pt(pt_path: Path) -> dict:
    """Load .pt file and compute pixel statistics.

    Args:
        pt_path: Path to the .pt file

    Returns:
        Dictionary with 'min', 'median', 'mean', 'max' statistics, or None on error
    """
    try:
        data = load_tensor_from_pt(pt_path)
        if isinstance(data, torch.Tensor):
            pixel_values = tensor_to_numpy(data, select_first=False).flatten()
        elif isinstance(data, dict) and "tensor" in data:
            pixel_values = tensor_to_numpy(data["tensor"], select_first=False).flatten()
        else:
            print(f"Unexpected format in {pt_path}")
            return None

        return {
            "min": float(np.min(pixel_values)),
            "median": float(np.median(pixel_values)),
            "mean": float(np.mean(pixel_values)),
            "max": float(np.max(pixel_values)),
        }
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        return None


def extract_statistics_from_metadata(
    json_path: Path,
    output_path: Path,
    compute_median: bool = True,
) -> dict:
    """Extract pixel statistics from comprehensive metadata JSON.

    Args:
        json_path: Path to comprehensive metadata JSON file
        output_path: Path to save extracted statistics JSON
        compute_median: If True, compute median from .pt files; otherwise use tile_stats

    Returns:
        Dictionary with extracted statistics
    """
    print(f"\n=== Processing {json_path.name} ===")
    print("Loading JSON file...")

    with open(json_path, "r") as f:
        data = json.load(f)

    files = data.get("files", [])
    print(f"Found {len(files)} files")

    extracted_data = {
        "pipeline_info": data.get("pipeline_info", {}),
        "total_files": len(files),
        "tiles": [],
    }

    processed_tiles = 0
    skipped_tiles = 0

    for file_entry in tqdm(files, desc="Processing files"):
        tiles = file_entry.get("tiles", [])

        for tile in tiles:
            tile_id = tile.get("tile_id", "unknown")
            pt_path = tile.get("pt_path")
            tile_stats = tile.get("tile_stats", {})

            if compute_median and pt_path:
                pt_path_obj = Path(pt_path)
                if pt_path_obj.exists():
                    stats = extract_pixel_stats_from_pt(pt_path_obj)
                    if stats:
                        tile_data = {
                            "tile_id": tile_id,
                            "min": stats["min"],
                            "median": stats["median"],
                            "mean": stats["mean"],
                            "max": stats["max"],
                        }
                        extracted_data["tiles"].append(tile_data)
                        processed_tiles += 1
                    else:
                        skipped_tiles += 1
                else:
                    if tile_stats:
                        tile_data = {
                            "tile_id": tile_id,
                            "min": tile_stats.get("min", 0.0),
                            "median": None,
                            "mean": tile_stats.get("mean", 0.0),
                            "max": tile_stats.get("max", 0.0),
                        }
                        extracted_data["tiles"].append(tile_data)
                        processed_tiles += 1
                    else:
                        skipped_tiles += 1
            else:
                if tile_stats:
                    tile_data = {
                        "tile_id": tile_id,
                        "min": tile_stats.get("min", 0.0),
                        "median": None,
                        "mean": tile_stats.get("mean", 0.0),
                        "max": tile_stats.get("max", 0.0),
                    }
                    extracted_data["tiles"].append(tile_data)
                    processed_tiles += 1
                else:
                    skipped_tiles += 1

    print(f"\nProcessed: {processed_tiles} tiles")
    print(f"Skipped: {skipped_tiles} tiles")

    print(f"\nSaving to {output_path}...")
    save_json_file(output_path, extracted_data)

    print(f"Saved {len(extracted_data['tiles'])} tile statistics to {output_path}")
    return extracted_data


def extract_sidd_metadata_fields(data_root: Path, scene_names: list = None):
    """Extract metadata fields from SIDD scene directories.

    Args:
        data_root: Root directory containing SIDD scene subdirectories
        scene_names: List of scene names to process (None = process all)
    """
    if scene_names is None:
        # Default test scenes
        scene_names = [
            "0001_001_S6_00100_00060_3200_L",
            "0004_001_S6_00100_00060_4400_L",
            "0020_001_GP_00800_00350_5500_N",
        ]

    relevant_fields = [
        "BlackLevel",
        "BlackLevels",
        "WhiteLevel",
        "ISO",
        "ExposureTime",
        "CFAPattern",
        "BitsPerSample",
        "Width",
        "Height",
    ]

    for scene_name in scene_names:
        scene_dir = data_root / scene_name
        metadata_files = list(scene_dir.glob("METADATA_RAW_*.MAT"))
        if metadata_files:
            print(f"\n{'='*70}")
            print(f"Scene: {scene_name}")
            print(f"{'='*70}")
            fields = extract_metadata_fields(metadata_files[0])
            if fields:
                for field in relevant_fields:
                    if field in fields:
                        print(f"  {field}: {fields[field]}")

                if scene_name == scene_names[0]:
                    print(f"\n  All fields ({len(fields)}):")
                    for field in sorted(fields.keys()):
                        print(f"    {field}: {fields[field]}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata fields and pixel statistics from dataset files"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Subcommand for SIDD metadata fields
    parser_fields = subparsers.add_parser(
        "fields", help="Extract key fields from SIDD metadata files"
    )
    parser_fields.add_argument(
        "--data-root",
        type=str,
        default="/home/jilab/Jae/external/dataset/sidd/SIDD_Small_Raw_Only/Data",
        help="Root directory containing SIDD scene subdirectories",
    )
    parser_fields.add_argument(
        "--scenes",
        nargs="+",
        type=str,
        default=None,
        help="List of scene names to process (default: test scenes)",
    )

    # Subcommand for pixel statistics
    parser_stats = subparsers.add_parser(
        "stats", help="Extract pixel statistics from comprehensive metadata JSON"
    )
    parser_stats.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to comprehensive metadata JSON file",
    )
    parser_stats.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for statistics JSON (default: auto-generated)",
    )
    parser_stats.add_argument(
        "--no-median",
        action="store_true",
        help="Skip computing median from .pt files (use tile_stats only)",
    )

    # Batch processing for stats
    parser_batch = subparsers.add_parser(
        "batch-stats", help="Extract statistics for both Sony and Fuji metadata files"
    )
    parser_batch.add_argument(
        "--base-dir",
        type=str,
        default="preprocessing/processed",
        help="Base directory containing comprehensive metadata files",
    )
    parser_batch.add_argument(
        "--no-median",
        action="store_true",
        help="Skip computing median from .pt files (use tile_stats only)",
    )

    args = parser.parse_args()

    if args.command == "fields":
        extract_sidd_metadata_fields(Path(args.data_root), args.scenes)

    elif args.command == "stats":
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return

        if args.output is None:
            # Auto-generate output name
            output_path = input_path.parent / f"{input_path.stem}_data_values.json"
        else:
            output_path = Path(args.output)

        extract_statistics_from_metadata(
            input_path, output_path, compute_median=not args.no_median
        )

    elif args.command == "batch-stats":
        base_dir = Path(args.base_dir)

        # Process Fuji metadata
        fuji_input = base_dir / "comprehensive_fuji_tiles_metadata.json"
        fuji_output = base_dir / "fuji_data_values.json"

        if fuji_input.exists():
            extract_statistics_from_metadata(
                fuji_input, fuji_output, compute_median=not args.no_median
            )
        else:
            print(f"File not found: {fuji_input}")

        # Process Sony metadata
        sony_input = base_dir / "comprehensive_sony_tiles_metadata.json"
        sony_output = base_dir / "sony_data_values.json"

        if sony_input.exists():
            extract_statistics_from_metadata(
                sony_input, sony_output, compute_median=not args.no_median
            )
        else:
            print(f"File not found: {sony_input}")

        print("\n=== Extraction complete ===")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
