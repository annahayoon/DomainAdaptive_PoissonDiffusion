#!/usr/bin/env python3
"""Create pixel value distribution plots for SIDD dataset.

Supports both normalized [0,1] and raw pixel value distributions.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.utils.file_utils import (
    find_mat_file_pairs,
    load_mat_file,
    load_metadata_from_mat,
)
from core.utils.sensor_utils import reverse_normalize_to_raw
from core.utils.tensor_utils import extract_pixel_values_from_data


def process_dataset(
    data_root: Path,
    use_raw_values: bool = False,
    max_scenes: int = None,
    sample_pixels: int = None,
):
    """Process all scene directories and extract pixel values.

    Args:
        data_root: Root directory containing SIDD scene subdirectories
        use_raw_values: If True, convert normalized values to raw pixel values
        max_scenes: Maximum number of scenes to process (None = all)
        sample_pixels: Number of pixels to sample per image pair (None = all pixels)

    Returns:
        Dictionary with 'gt' and 'noisy' pixel arrays
    """
    print(f"Processing dataset from: {data_root}")

    pairs = find_mat_file_pairs(data_root, include_metadata=use_raw_values)
    print(f"Found {len(pairs)} image pairs")

    if max_scenes:
        pairs = pairs[:max_scenes]
        print(f"Processing first {len(pairs)} pairs")

    gt_pixels = []
    noisy_pixels = []
    metadata_stats = (
        {"with_metadata": 0, "without_metadata": 0} if use_raw_values else None
    )

    for pair in tqdm(pairs, desc="Loading image pairs"):
        gt_data = load_mat_file(pair["gt_path"])
        noisy_data = load_mat_file(pair["noisy_path"])

        if gt_data is None or noisy_data is None:
            continue

        # Handle raw value conversion if needed
        if use_raw_values:
            metadata = None
            if pair["metadata_path"] and pair["metadata_path"].exists():
                metadata = load_metadata_from_mat(pair["metadata_path"])
                if (
                    metadata["black_level"] is not None
                    and metadata["white_level"] is not None
                ):
                    metadata_stats["with_metadata"] += 1
                else:
                    metadata_stats["without_metadata"] += 1
            else:
                metadata_stats["without_metadata"] += 1

            if (
                metadata
                and metadata["black_level"] is not None
                and metadata["white_level"] is not None
            ):
                gt_vals = extract_pixel_values_from_data(gt_data)
                noisy_vals = extract_pixel_values_from_data(noisy_data)
                gt_vals = reverse_normalize_to_raw(
                    gt_vals, metadata["black_level"], metadata["white_level"]
                )
                noisy_vals = reverse_normalize_to_raw(
                    noisy_vals, metadata["black_level"], metadata["white_level"]
                )

                if gt_vals is None or noisy_vals is None:
                    continue
            else:
                print(f"Warning: No metadata for {pair['scene_dir']}, skipping")
                continue
        else:
            gt_vals = extract_pixel_values_from_data(gt_data)
            noisy_vals = extract_pixel_values_from_data(noisy_data)

        if sample_pixels and len(gt_vals) > sample_pixels:
            indices = np.random.choice(len(gt_vals), sample_pixels, replace=False)
            gt_vals = gt_vals[indices]
            noisy_vals = noisy_vals[indices]

        gt_pixels.append(gt_vals)
        noisy_pixels.append(noisy_vals)

    gt_all = np.concatenate(gt_pixels) if gt_pixels else np.array([])
    noisy_all = np.concatenate(noisy_pixels) if noisy_pixels else np.array([])

    print(f"\nTotal pixels extracted:")
    print(f"  Ground Truth: {len(gt_all):,}")
    print(f"  Noisy: {len(noisy_all):,}")
    if metadata_stats:
        print(f"\nMetadata statistics:")
        print(f"  Pairs with valid metadata: {metadata_stats['with_metadata']}")
        print(f"  Pairs without metadata: {metadata_stats['without_metadata']}")

    return {"gt": gt_all, "noisy": noisy_all}


def create_distribution_plots(
    data_dict: dict,
    output_path: Path,
    use_raw_values: bool = False,
):
    """Create distribution plots comparing noisy vs ground truth pixel values.

    Args:
        data_dict: Dictionary with 'gt' and 'noisy' pixel arrays
        output_path: Path to save the plot
        use_raw_values: If True, use raw pixel value labels
    """
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300

    gt_pixels = data_dict["gt"]
    noisy_pixels = data_dict["noisy"]

    if len(gt_pixels) == 0 or len(noisy_pixels) == 0:
        print("Error: No pixel data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    min_val = min(np.min(gt_pixels), np.min(noisy_pixels))
    max_val = max(np.max(gt_pixels), np.max(noisy_pixels))
    bins = np.linspace(min_val, max_val, 100)

    # Format values based on data type
    if use_raw_values:
        value_fmt = ".2f"
        xlabel = "Raw Pixel Value"
        title_prefix = "SIDD Dataset - Raw Pixel Value Distribution"
    else:
        value_fmt = ".4f"
        xlabel = "Pixel Value [0,1]"
        title_prefix = "SIDD Dataset - Pixel Value Distribution"

    # Full range plot
    ax1 = axes[0]
    ax1.hist(
        gt_pixels,
        bins=bins,
        alpha=0.6,
        label="Ground Truth",
        color="blue",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.hist(
        noisy_pixels,
        bins=bins,
        alpha=0.6,
        label="Noisy",
        color="orange",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )

    stats_text = (
        f"Ground Truth:\n"
        f"  μ={np.mean(gt_pixels):{value_fmt}}\n"
        f"  σ={np.std(gt_pixels):{value_fmt}}\n"
        f"  min={np.min(gt_pixels):{value_fmt}}\n"
        f"  max={np.max(gt_pixels):{value_fmt}}\n\n"
        f"Noisy:\n"
        f"  μ={np.mean(noisy_pixels):{value_fmt}}\n"
        f"  σ={np.std(noisy_pixels):{value_fmt}}\n"
        f"  min={np.min(noisy_pixels):{value_fmt}}\n"
        f"  max={np.max(noisy_pixels):{value_fmt}}"
    )

    ax1.text(
        0.05,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(f"{title_prefix} (Full Range)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Zoomed plot (1st-99th percentile)
    ax2 = axes[1]
    gt_clean = gt_pixels[
        (gt_pixels >= np.percentile(gt_pixels, 1))
        & (gt_pixels <= np.percentile(gt_pixels, 99))
    ]
    noisy_clean = noisy_pixels[
        (noisy_pixels >= np.percentile(noisy_pixels, 1))
        & (noisy_pixels <= np.percentile(noisy_pixels, 99))
    ]

    min_val_zoom = min(np.min(gt_clean), np.min(noisy_clean))
    max_val_zoom = max(np.max(gt_clean), np.max(noisy_clean))
    bins_zoom = np.linspace(min_val_zoom, max_val_zoom, 100)

    ax2.hist(
        gt_clean,
        bins=bins_zoom,
        alpha=0.6,
        label="Ground Truth",
        color="blue",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.hist(
        noisy_clean,
        bins=bins_zoom,
        alpha=0.6,
        label="Noisy",
        color="orange",
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )

    stats_text_zoom = (
        f"Zoomed View (1st-99th percentile):\n\n"
        f"Ground Truth:\n"
        f"  μ={np.mean(gt_clean):{value_fmt}}\n"
        f"  σ={np.std(gt_clean):{value_fmt}}\n\n"
        f"Noisy:\n"
        f"  μ={np.mean(noisy_clean):{value_fmt}}\n"
        f"  σ={np.std(noisy_clean):{value_fmt}}\n\n"
        f"Difference in mean:\n"
        f"  Δμ={np.mean(noisy_clean) - np.mean(gt_clean):{value_fmt}}"
    )

    ax2.text(
        0.05,
        0.95,
        stats_text_zoom,
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title(f"{title_prefix} (Zoomed)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")
    plt.close()


def print_summary_stats(data_dict: dict, use_raw_values: bool = False):
    """Print summary statistics for the pixel values."""
    gt_pixels = data_dict["gt"]
    noisy_pixels = data_dict["noisy"]

    value_fmt = ".2f" if use_raw_values else ".6f"
    title = "Raw Pixel Values" if use_raw_values else "Pixel Values in [0,1] range"

    print("\n" + "=" * 70)
    print(f"Summary Statistics ({title})")
    print("=" * 70)
    print(f"\nGround Truth:")
    print(f"  Mean:   {np.mean(gt_pixels):{value_fmt}}")
    print(f"  Std:    {np.std(gt_pixels):{value_fmt}}")
    print(f"  Min:    {np.min(gt_pixels):{value_fmt}}")
    print(f"  Max:    {np.max(gt_pixels):{value_fmt}}")
    print(f"  Median: {np.median(gt_pixels):{value_fmt}}")
    print(f"\nNoisy:")
    print(f"  Mean:   {np.mean(noisy_pixels):{value_fmt}}")
    print(f"  Std:    {np.std(noisy_pixels):{value_fmt}}")
    print(f"  Min:    {np.min(noisy_pixels):{value_fmt}}")
    print(f"  Max:    {np.max(noisy_pixels):{value_fmt}}")
    print(f"  Median: {np.median(noisy_pixels):{value_fmt}}")
    print(f"\nDifference (Noisy - Ground Truth):")
    print(
        f"  Mean difference:   {np.mean(noisy_pixels) - np.mean(gt_pixels):{value_fmt}}"
    )
    print(
        f"  Std difference:    {np.std(noisy_pixels) - np.std(gt_pixels):{value_fmt}}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create pixel value distribution plots for SIDD dataset"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/jilab/Jae/external/dataset/sidd/SIDD_Small_Raw_Only/Data",
        help="Root directory containing SIDD scene subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the distribution plot (default: auto-generated based on --raw flag)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw pixel values instead of normalized [0,1] values",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process (None = all)",
    )
    parser.add_argument(
        "--sample-pixels",
        type=int,
        default=None,
        help="Number of pixels to sample per image pair (None = all pixels, default: 100000 for raw)",
    )

    args = parser.parse_args()
    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"Error: Data root directory does not exist: {data_root}")
        return

    # Set default sample_pixels for raw mode if not specified
    if args.raw and args.sample_pixels is None:
        args.sample_pixels = 100000

    # Auto-generate output path if not provided
    if args.output is None:
        output_name = (
            "sidd_raw_pixel_distributions.png"
            if args.raw
            else "sidd_pixel_distributions.png"
        )
        args.output = str(Path(__file__).parent / "processed" / output_name)

    data_dict = process_dataset(
        data_root,
        use_raw_values=args.raw,
        max_scenes=args.max_scenes,
        sample_pixels=args.sample_pixels,
    )

    if len(data_dict["gt"]) == 0:
        print(
            "Error: No data was extracted. Check that .mat files exist and are readable."
        )
        return

    create_distribution_plots(data_dict, Path(args.output), use_raw_values=args.raw)
    print_summary_stats(data_dict, use_raw_values=args.raw)


if __name__ == "__main__":
    main()
