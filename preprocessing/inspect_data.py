#!/usr/bin/env python
"""
Visual inspection script for preprocessed data.
Displays images, histograms, and statistics for quality assessment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    plt = None


def load_preprocessed_file(filepath: Path) -> Dict[str, Any]:
    """Load a preprocessed .pt file."""
    try:
        data = torch.load(filepath, map_location="cpu")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def convert_to_displayable(image: torch.Tensor, domain_id: int) -> np.ndarray:
    """
    Convert tensor to displayable numpy array.

    Args:
        image: Input tensor [C, H, W]
        domain_id: Domain ID for proper color handling

    Returns:
        Displayable image array
    """
    img = image.numpy()

    if domain_id == 0 and img.shape[0] == 4:  # Photography (RGGB)
        # Simple demosaic for visualization
        rgb = np.zeros((3, img.shape[1], img.shape[2]))
        rgb[0] = img[0]  # R
        rgb[1] = (img[1] + img[2]) / 2  # G (average G1 and G2)
        rgb[2] = img[3]  # B

        # Transpose to HWC format
        rgb = np.transpose(rgb, (1, 2, 0))

        # Normalize for display
        rgb = np.clip(rgb, 0, None)
        if rgb.max() > 0:
            rgb = rgb / np.percentile(
                rgb, 99.5
            )  # Use 99.5th percentile for normalization
        rgb = np.clip(rgb, 0, 1)

        return rgb

    elif img.shape[0] == 1:  # Grayscale (microscopy/astronomy)
        gray = img[0]

        # Normalize for display
        gray = np.clip(gray, 0, None)
        if gray.max() > 0:
            gray = gray / np.percentile(gray, 99.5)
        gray = np.clip(gray, 0, 1)

        return gray

    else:
        # Multi-channel, take first 3 channels or duplicate single channel
        if img.shape[0] >= 3:
            rgb = img[:3]
        else:
            rgb = np.repeat(img[0:1], 3, axis=0)

        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = np.clip(rgb, 0, None)
        if rgb.max() > 0:
            rgb = rgb / np.percentile(rgb, 99.5)
        rgb = np.clip(rgb, 0, 1)

        return rgb


def plot_histogram(image: torch.Tensor, ax, title: str) -> None:
    """Plot histogram of image values."""
    values = image.flatten().numpy()

    # Remove extreme outliers for better visualization
    q1, q99 = np.percentile(values, [1, 99])
    values_clipped = values[(values >= q1) & (values <= q99)]

    ax.hist(values_clipped, bins=100, alpha=0.7, density=True)
    ax.set_xlabel("Normalized Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"Mean: {values.mean():.3f}\nStd: {values.std():.3f}\nMin: {values.min():.3f}\nMax: {values.max():.3f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )


def inspect_prior_tile(filepath: Path, show_histograms: bool = True) -> None:
    """Inspect a prior training tile."""
    if plt is None:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return

    data = load_preprocessed_file(filepath)

    print(f"\n=== Inspecting Prior Tile: {filepath.name} ===")
    print(f"Domain ID: {data['domain_id']}")
    print(f"Scene ID: {data['metadata']['scene_id']}")
    print(f"Tile Index: {data['metadata']['tile_idx']}")
    print(f"Augmented: {data['metadata']['augmented']}")

    # Get image data
    clean_img = data["clean_norm"]
    domain_id = data["domain_id"]

    print(f"Image shape: {clean_img.shape}")
    print(f"Value range: [{clean_img.min():.4f}, {clean_img.max():.4f}]")
    print(f"Mean: {clean_img.mean():.4f}")

    # Create display
    if show_histograms:
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig)

        # Image
        ax_img = fig.add_subplot(gs[:, 0])
        displayable = convert_to_displayable(clean_img, domain_id)

        if len(displayable.shape) == 3:
            ax_img.imshow(displayable)
        else:
            ax_img.imshow(displayable, cmap="gray")
        ax_img.set_title(f"Clean Tile (Domain {domain_id})")
        ax_img.axis("off")

        # Histogram
        ax_hist = fig.add_subplot(gs[0, 1])
        plot_histogram(clean_img, ax_hist, "Value Distribution")

        # Channel-wise histograms if multi-channel
        ax_channels = fig.add_subplot(gs[1, 1])
        for c in range(clean_img.shape[0]):
            channel_data = clean_img[c].flatten().numpy()
            ax_channels.hist(
                channel_data, bins=50, alpha=0.6, label=f"Channel {c}", density=True
            )
        ax_channels.set_xlabel("Normalized Value")
        ax_channels.set_ylabel("Density")
        ax_channels.set_title("Channel-wise Distributions")
        ax_channels.legend()
        ax_channels.grid(True, alpha=0.3)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        displayable = convert_to_displayable(clean_img, domain_id)

        if len(displayable.shape) == 3:
            ax.imshow(displayable)
        else:
            ax.imshow(displayable, cmap="gray")
        ax.set_title(f"Clean Tile (Domain {domain_id})")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def inspect_posterior_scene(
    filepath: Path, show_histograms: bool = True, show_masks: bool = True
) -> None:
    """Inspect a posterior scene file."""
    if plt is None:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return

    data = load_preprocessed_file(filepath)

    print(f"\n=== Inspecting Posterior Scene: {filepath.name} ===")
    print(f"Domain ID: {data['metadata']['domain_id']}")
    print(f"Scene ID: {data['metadata']['scene_id']}")
    print(f"Original shape: {data['metadata']['original_shape']}")

    # Get image data
    noisy_img = data["noisy_norm"]
    clean_img = data["clean_norm"]
    domain_id = data["metadata"]["domain_id"]
    calibration = data["calibration"]
    masks = data["masks"]

    print(f"Noisy image shape: {noisy_img.shape}")
    print(f"Noisy value range: [{noisy_img.min():.4f}, {noisy_img.max():.4f}]")

    has_clean = clean_img is not None
    if has_clean:
        print(f"Clean image shape: {clean_img.shape}")
        print(f"Clean value range: [{clean_img.min():.4f}, {clean_img.max():.4f}]")

    print("\nCalibration:")
    for key, value in calibration.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Create display
    n_cols = 3 if has_clean else 2
    n_rows = 2 if show_histograms else 1

    if show_masks:
        n_cols += 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    col = 0

    # Noisy image
    displayable_noisy = convert_to_displayable(noisy_img, domain_id)
    if len(displayable_noisy.shape) == 3:
        axes[0, col].imshow(displayable_noisy)
    else:
        axes[0, col].imshow(displayable_noisy, cmap="gray")
    axes[0, col].set_title("Noisy Image")
    axes[0, col].axis("off")

    if show_histograms:
        plot_histogram(noisy_img, axes[1, col], "Noisy Histogram")

    col += 1

    # Clean image (if available)
    if has_clean:
        displayable_clean = convert_to_displayable(clean_img, domain_id)
        if len(displayable_clean.shape) == 3:
            axes[0, col].imshow(displayable_clean)
        else:
            axes[0, col].imshow(displayable_clean, cmap="gray")
        axes[0, col].set_title("Clean Image")
        axes[0, col].axis("off")

        if show_histograms:
            plot_histogram(clean_img, axes[1, col], "Clean Histogram")

        col += 1

    # Difference image (if clean available)
    if has_clean:
        diff = noisy_img - clean_img
        diff_display = diff[0].numpy() if diff.shape[0] == 1 else diff.mean(0).numpy()

        im = axes[0, col].imshow(
            diff_display,
            cmap="RdBu_r",
            vmin=np.percentile(diff_display, 5),
            vmax=np.percentile(diff_display, 95),
        )
        axes[0, col].set_title("Difference (Noisy - Clean)")
        axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        if show_histograms:
            plot_histogram(diff, axes[1, col], "Difference Histogram")

        col += 1

    # Masks
    if show_masks:
        valid_mask = masks["valid"][0].numpy()
        saturated_mask = masks["saturated"][0].numpy()

        # Combine masks for visualization
        mask_display = np.zeros((*valid_mask.shape, 3))
        mask_display[valid_mask, 1] = 0.5  # Green for valid
        mask_display[saturated_mask, 0] = 1.0  # Red for saturated
        mask_display[~valid_mask & ~saturated_mask, 2] = 0.5  # Blue for invalid

        axes[0, col].imshow(mask_display)
        axes[0, col].set_title("Masks (Green=Valid, Red=Saturated, Blue=Invalid)")
        axes[0, col].axis("off")

        if show_histograms:
            # Show mask statistics
            valid_frac = valid_mask.mean()
            sat_frac = saturated_mask.mean()
            axes[1, col].bar(
                ["Valid", "Saturated", "Invalid"],
                [valid_frac, sat_frac, 1 - valid_frac - sat_frac],
            )
            axes[1, col].set_ylabel("Fraction of Pixels")
            axes[1, col].set_title("Mask Statistics")
            axes[1, col].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    # Print noise model validation if clean data available
    if has_clean:
        print("\n=== Noise Model Validation ===")

        # Convert to electrons
        noisy_e = noisy_img * calibration["scale"]
        clean_e = clean_img * calibration["scale"]

        # Compute residuals
        residual = noisy_e - clean_e

        # Expected variance (Poisson + read noise)
        var_expected = clean_e + calibration["read_noise"] ** 2

        # Chi-squared test
        chi2 = (residual**2 / var_expected).mean().item()
        print(f"Chi-squared statistic: {chi2:.3f} (should be ~1.0)")

        # SNR analysis
        snr = clean_e / torch.sqrt(var_expected)
        print(f"Mean SNR: {snr.mean():.2f}")
        print(f"SNR range: [{snr.min():.2f}, {snr.max():.2f}]")


def main():
    """Main inspection entry point."""
    parser = argparse.ArgumentParser(
        description="Visually inspect preprocessed data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a prior training tile
  python scripts/inspect_data.py data/preprocessed/prior_clean/photography/train/tile_000000.pt

  # Inspect a posterior scene
  python scripts/inspect_data.py data/preprocessed/posterior/microscopy/train/seq001_frame001.pt

  # Quick inspection without histograms
  python scripts/inspect_data.py data.pt --no-histograms

  # Inspect without masks
  python scripts/inspect_data.py data.pt --no-masks
        """,
    )

    parser.add_argument("filepath", type=str, help="Path to .pt file to inspect")
    parser.add_argument(
        "--no-histograms", action="store_true", help="Don't show histogram plots"
    )
    parser.add_argument(
        "--no-masks", action="store_true", help="Don't show mask visualization"
    )
    parser.add_argument(
        "--save", type=str, help="Save plot to file instead of displaying"
    )

    args = parser.parse_args()

    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    # Determine file type based on content
    data = load_preprocessed_file(filepath)

    if "clean_norm" in data and "noisy_norm" not in data:
        # Prior training tile
        inspect_prior_tile(filepath, show_histograms=not args.no_histograms)
    elif "noisy_norm" in data:
        # Posterior scene
        inspect_posterior_scene(
            filepath,
            show_histograms=not args.no_histograms,
            show_masks=not args.no_masks,
        )
    else:
        print(f"Unknown file format: {filepath}")
        print("Expected either prior tile or posterior scene format")
        sys.exit(1)

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {args.save}")


if __name__ == "__main__":
    main()
