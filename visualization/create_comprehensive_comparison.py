#!/usr/bin/env python3
"""
Create comprehensive comparison visualization across all three domains
(photography, microscopy, astronomy) using both single-domain and cross-domain models.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append("/home/jilab/Jae/preprocessing")

from utils import load_tensor_from_pt, normalize_for_display

# Import preprocessing utilities for consistent RGB handling
from preprocessing.visualizations import prepare_for_tile_visualization


def load_results_from_json(json_path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def denormalize_to_physical(tensor: torch.Tensor, domain: str) -> np.ndarray:
    """Denormalize tensor to physical units for display."""
    # Handle different tensor shapes
    if tensor.dim() == 4:  # (B, C, H, W)
        tensor = tensor.squeeze(0)  # Remove batch dimension
    if tensor.dim() == 3 and tensor.shape[0] == 1:  # (1, H, W) for grayscale
        tensor = tensor.squeeze(0)  # Remove channel dimension for grayscale

    if domain == "photography":
        # Photography: [-1, 1] -> [0, 15871] ADU
        return ((tensor + 1) / 2 * 15871).clamp(0, 15871).numpy()
    elif domain == "microscopy":
        # Microscopy: [-1, 1] -> [0, 65535] intensity
        return ((tensor + 1) / 2 * 65535).clamp(0, 65535).numpy()
    elif domain == "astronomy":
        # Astronomy: [-1, 1] -> [0, 450] counts
        return ((tensor + 1) / 2 * 450).clamp(0, 450).numpy()
    else:
        return tensor.numpy()


# Use shared normalize_for_display utility
from utils import normalize_for_display


def create_comprehensive_comparison():
    """Create comprehensive comparison visualization across all domains."""

    # Define paths
    base_path = Path("/home/jilab/Jae/results")

    # Single-domain results
    single_domain_paths = {
        "photography": base_path
        / "test_photography_single_domain"
        / "example_00_photography_sony_00135_00_0.1s_tile_0005",
        "microscopy": base_path
        / "test_microscopy_single_domain"
        / "example_00_microscopy_ER_Cell_002_RawGTSIMData_level_06_tile_0002",
        "astronomy": base_path
        / "test_astronomy_single_domain"
        / "example_00_astronomy_j8g6z3jdq_g800l_sci_tile_0071",
    }

    # Cross-domain results
    cross_domain_paths = {
        "photography": base_path
        / "test_photography_cross_domain"
        / "example_00_photography_sony_00135_00_0.1s_tile_0005",
        "microscopy": base_path
        / "test_microscopy_cross_domain"
        / "example_00_microscopy_ER_Cell_002_RawGTSIMData_level_06_tile_0002",
        "astronomy": base_path
        / "test_astronomy_cross_domain"
        / "example_00_astronomy_j8g6z3jdq_g800l_sci_tile_0071",
    }

    # Load all results
    single_domain_results = {}
    cross_domain_results = {}

    for domain, path in single_domain_paths.items():
        if path.exists():
            single_domain_results[domain] = load_results_from_json(
                path / "results.json"
            )

    for domain, path in cross_domain_paths.items():
        if path.exists():
            cross_domain_results[domain] = load_results_from_json(path / "results.json")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 9, figure=fig, hspace=0.3, wspace=0.1)

    # Define methods to compare
    methods = ["noisy", "exposure_scaled", "gaussian_x0", "pg_x0"]
    method_labels = ["Noisy", "Exposure Scaled", "Gaussian x0", "PG x0"]

    # Colors for different methods
    method_colors = {
        "noisy": "gray",
        "exposure_scaled": "orange",
        "gaussian_x0": "blue",
        "pg_x0": "red",
    }

    # Plot for each domain
    domains = ["photography", "microscopy", "astronomy"]
    domain_labels = ["Photography (Sony)", "Microscopy (SIM)", "Astronomy (Hubble)"]

    for i, domain in enumerate(domains):
        if domain not in single_domain_results or domain not in cross_domain_results:
            continue

        # Load images for this domain
        single_path = single_domain_paths[domain]
        cross_path = cross_domain_paths[domain]

        # Load clean reference
        clean_img = load_tensor_from_pt(single_path / "clean.pt")
        clean_phys = denormalize_to_physical(clean_img, domain)
        clean_display = normalize_for_display(clean_phys)

        # Plot clean reference
        ax_clean = fig.add_subplot(gs[i, 0])
        if clean_display.ndim == 3 and clean_display.shape[0] == 3:  # RGB (C, H, W)
            # Use preprocessing utility to convert CHW to HWC for display
            clean_rgb = prepare_for_tile_visualization(clean_display)
            if clean_rgb is not None:
                ax_clean.imshow(clean_rgb)
            else:
                # Fallback to manual transpose
                ax_clean.imshow(clean_display.transpose(1, 2, 0))
        else:  # Grayscale
            ax_clean.imshow(clean_display, cmap="gray")
        ax_clean.set_title(
            f"{domain_labels[i]}\nClean Reference", fontsize=10, fontweight="bold"
        )
        ax_clean.axis("off")

        # Plot methods for single-domain model
        for j, (method, label) in enumerate(zip(methods, method_labels)):
            ax = fig.add_subplot(gs[i, j + 1])

            # Load single-domain result
            single_img = load_tensor_from_pt(single_path / f"restored_{method}.pt")
            single_phys = denormalize_to_physical(single_img, domain)
            single_display = normalize_for_display(single_phys)

            if (
                single_display.ndim == 3 and single_display.shape[0] == 3
            ):  # RGB (C, H, W)
                # Use preprocessing utility to convert CHW to HWC for display
                single_rgb = prepare_for_tile_visualization(single_display)
                if single_rgb is not None:
                    ax.imshow(single_rgb)
                else:
                    # Fallback to manual transpose
                    ax.imshow(single_display.transpose(1, 2, 0))
            else:  # Grayscale
                ax.imshow(single_display, cmap="gray")

            # Get metrics
            metrics = single_domain_results[domain]["comprehensive_metrics"][method]
            ssim = metrics.get("ssim", 0)
            psnr = metrics.get("psnr", 0)

            ax.set_title(
                f"{label}\nSingle-Domain\nSSIM: {ssim:.3f}, PSNR: {psnr:.1f}dB",
                fontsize=9,
                color=method_colors[method],
            )
            ax.axis("off")

        # Plot methods for cross-domain model
        for j, (method, label) in enumerate(zip(methods, method_labels)):
            ax = fig.add_subplot(gs[i, j + 3])

            # Load cross-domain result
            cross_img = load_tensor_from_pt(cross_path / f"restored_{method}.pt")
            cross_phys = denormalize_to_physical(cross_img, domain)
            cross_display = normalize_for_display(cross_phys)

            if cross_display.ndim == 3 and cross_display.shape[0] == 3:  # RGB (C, H, W)
                # Use preprocessing utility to convert CHW to HWC for display
                cross_rgb = prepare_for_tile_visualization(cross_display)
                if cross_rgb is not None:
                    ax.imshow(cross_rgb)
                else:
                    # Fallback to manual transpose
                    ax.imshow(cross_display.transpose(1, 2, 0))
            else:  # Grayscale
                ax.imshow(cross_display, cmap="gray")

            # Get metrics
            metrics = cross_domain_results[domain]["comprehensive_metrics"][method]
            ssim = metrics.get("ssim", 0)
            psnr = metrics.get("psnr", 0)

            ax.set_title(
                f"{label}\nCross-Domain\nSSIM: {ssim:.3f}, PSNR: {psnr:.1f}dB",
                fontsize=9,
                color=method_colors[method],
            )
            ax.axis("off")

    # Add summary metrics row
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis("off")

    # Create summary table
    summary_text = "SUMMARY: Cross-Domain vs Single-Domain Performance\n\n"

    for domain in domains:
        if domain not in single_domain_results or domain not in cross_domain_results:
            continue

        summary_text += f"{domain.upper()}:\n"

        # Compare PG x0 performance
        single_pg = single_domain_results[domain]["comprehensive_metrics"]["pg_x0"]
        cross_pg = cross_domain_results[domain]["comprehensive_metrics"]["pg_x0"]

        single_ssim = single_pg.get("ssim", 0)
        single_psnr = single_pg.get("psnr", 0)
        cross_ssim = cross_pg.get("ssim", 0)
        cross_psnr = cross_pg.get("psnr", 0)

        ssim_diff = cross_ssim - single_ssim
        psnr_diff = cross_psnr - single_psnr

        summary_text += (
            f"  PG x0 - Single: SSIM={single_ssim:.3f}, PSNR={single_psnr:.1f}dB\n"
        )
        summary_text += (
            f"  PG x0 - Cross:  SSIM={cross_ssim:.3f}, PSNR={cross_psnr:.1f}dB\n"
        )
        summary_text += (
            f"  Difference:     SSIM={ssim_diff:+.3f}, PSNR={psnr_diff:+.1f}dB\n\n"
        )

    ax_summary.text(
        0.5,
        0.5,
        summary_text,
        transform=ax_summary.transAxes,
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    # Add column headers
    fig.text(
        0.5,
        0.95,
        "Cross-Domain Low-Light Enhancement: Single-Domain vs Cross-Domain Models",
        fontsize=16,
        fontweight="bold",
        ha="center",
    )

    # Column labels
    fig.text(
        0.05, 0.88, "Clean\nReference", fontsize=12, fontweight="bold", ha="center"
    )
    fig.text(
        0.25, 0.88, "Single-Domain Model", fontsize=12, fontweight="bold", ha="center"
    )
    fig.text(
        0.75, 0.88, "Cross-Domain Model", fontsize=12, fontweight="bold", ha="center"
    )

    # Save the visualization
    output_path = Path("/home/jilab/Jae/comprehensive_domain_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✓ Comprehensive comparison saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DOMAIN COMPARISON SUMMARY")
    print("=" * 80)

    for domain in domains:
        if domain not in single_domain_results or domain not in cross_domain_results:
            continue

        print(f"\n{domain.upper()} DOMAIN:")
        print("-" * 40)

        # Compare all methods
        for method in methods:
            single_metrics = single_domain_results[domain]["comprehensive_metrics"][
                method
            ]
            cross_metrics = cross_domain_results[domain]["comprehensive_metrics"][
                method
            ]

            single_ssim = single_metrics.get("ssim", 0)
            single_psnr = single_metrics.get("psnr", 0)
            cross_ssim = cross_metrics.get("ssim", 0)
            cross_psnr = cross_metrics.get("psnr", 0)

            ssim_diff = cross_ssim - single_ssim
            psnr_diff = cross_psnr - single_psnr

            print(f"  {method.upper()}:")
            print(
                f"    Single-Domain: SSIM={single_ssim:.3f}, PSNR={single_psnr:.1f}dB"
            )
            print(f"    Cross-Domain:  SSIM={cross_ssim:.3f}, PSNR={cross_psnr:.1f}dB")
            print(f"    Difference:    SSIM={ssim_diff:+.3f}, PSNR={psnr_diff:+.1f}dB")


if __name__ == "__main__":
    create_comprehensive_comparison()
