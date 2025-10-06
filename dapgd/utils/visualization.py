"""
Visualization utilities for DAPGD

Functions for creating figures, comparisons, and diagnostic plots.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


# Lazy imports for optional dependencies
def _get_matplotlib():
    """Lazy import of matplotlib"""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib required for visualization. Install: pip install matplotlib"
        )


def save_comparison_image(
    noisy: torch.Tensor,
    restored: torch.Tensor,
    ground_truth: torch.Tensor,
    output_path: Path,
    titles: Optional[List[str]] = None,
):
    """
    Create side-by-side comparison image

    PURPOSE: Visual validation of restoration quality

    Args:
        noisy: Noisy input [1,C,H,W]
        restored: Restored output [1,C,H,W]
        ground_truth: Ground truth [1,C,H,W]
        output_path: Where to save the figure
        titles: Custom titles for each panel
    """
    plt = _get_matplotlib()

    if titles is None:
        titles = ["Noisy Input", "Restored", "Ground Truth"]

    # Convert to numpy and handle dimensions
    def prepare_for_display(img):
        img = img.squeeze(0).cpu().numpy()
        if img.shape[0] == 1:  # Grayscale
            return img.squeeze(0), "gray"
        else:  # RGB
            return np.transpose(img, (1, 2, 0)), None

    noisy_np, cmap_noisy = prepare_for_display(noisy)
    restored_np, cmap_restored = prepare_for_display(restored)
    gt_np, cmap_gt = prepare_for_display(ground_truth)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(noisy_np, cmap=cmap_noisy)
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    axes[1].imshow(restored_np, cmap=cmap_restored)
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    axes[2].imshow(gt_np, cmap=cmap_gt)
    axes[2].set_title(titles[2])
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sampling_trajectory(
    trajectory: List[torch.Tensor],
    sigmas: torch.Tensor,
    output_path: Path,
    num_frames: int = 10,
):
    """
    Visualize the diffusion sampling trajectory

    PURPOSE: Diagnostic tool to understand sampling process

    Args:
        trajectory: List of x_t at each timestep
        sigmas: Noise levels at each step
        output_path: Where to save
        num_frames: Number of frames to show
    """
    plt = _get_matplotlib()

    # Select evenly spaced frames
    indices = np.linspace(0, len(trajectory) - 1, num_frames, dtype=int)

    fig, axes = plt.subplots(2, num_frames // 2, figsize=(20, 8))
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        img = trajectory[idx].squeeze(0).cpu().numpy()

        if img.shape[0] == 1:
            img = img.squeeze(0)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            img = np.transpose(img, (1, 2, 0))
            ax.imshow(np.clip(img, 0, 1))

        sigma = sigmas[idx].item() if idx < len(sigmas) else 0
        ax.set_title(f"t={idx}, σ={sigma:.3f}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residual_analysis(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    s: float,
    sigma_r: float,
    output_path: Path,
):
    """
    Create diagnostic plots for residual analysis

    PURPOSE: Validate physical consistency visually

    Creates:
    - Normalized residual histogram (should be N(0,1))
    - Residual vs. signal scatter plot
    - Chi-squared per-pixel map
    """
    plt = _get_matplotlib()
    from matplotlib.gridspec import GridSpec

    # Compute residuals
    expected = s * predicted
    variance = s * predicted + sigma_r**2
    residual = observed - expected
    normalized_residual = (residual / torch.sqrt(variance)).cpu().numpy().flatten()

    # Create figure
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)

    # 1. Histogram of normalized residuals
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(normalized_residual, bins=50, density=True, alpha=0.7, label="Empirical")

    # Overlay standard normal
    x = np.linspace(-4, 4, 100)
    ax1.plot(x, np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi), "r-", label="N(0,1)")
    ax1.set_xlabel("Normalized Residual")
    ax1.set_ylabel("Density")
    ax1.set_title("Residual Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residual vs. signal
    ax2 = fig.add_subplot(gs[0, 1])
    signal = expected.cpu().numpy().flatten()
    residual_np = residual.cpu().numpy().flatten()

    # Subsample for visualization
    subsample = np.random.choice(len(signal), min(10000, len(signal)), replace=False)
    ax2.scatter(signal[subsample], residual_np[subsample], alpha=0.1, s=1)
    ax2.axhline(0, color="r", linestyle="--")
    ax2.set_xlabel("Expected Signal")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual vs. Signal")
    ax2.grid(True, alpha=0.3)

    # 3. Chi-squared map
    ax3 = fig.add_subplot(gs[0, 2])
    chi2_map = ((residual**2) / variance).squeeze(0).cpu().numpy()
    if chi2_map.ndim == 3:
        chi2_map = chi2_map.mean(axis=0)  # Average over channels
    im = ax3.imshow(chi2_map, cmap="viridis", vmin=0, vmax=4)
    ax3.set_title("χ² Per Pixel")
    plt.colorbar(im, ax=ax3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_image_grid(
    images: List[torch.Tensor],
    titles: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    ncols: int = 4,
) -> Optional[np.ndarray]:
    """
    Create a grid of images

    Args:
        images: List of image tensors [C,H,W] or [1,C,H,W]
        titles: Optional titles for each image
        output_path: Where to save (if None, returns array)
        ncols: Number of columns in grid

    Returns:
        Grid image as numpy array if output_path is None
    """
    plt = _get_matplotlib()

    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (img, ax) in enumerate(zip(images, axes.flatten())):
        # Handle batch dimension
        if img.dim() == 4:
            img = img[0]

        img_np = img.cpu().numpy()

        # Convert to displayable format
        if img_np.shape[0] in [1, 3]:  # CHW format
            if img_np.shape[0] == 1:
                img_np = img_np.squeeze(0)
                ax.imshow(img_np, cmap="gray")
            else:
                img_np = np.transpose(img_np, (1, 2, 0))
                ax.imshow(np.clip(img_np, 0, 1))
        else:
            ax.imshow(img_np)

        if titles and idx < len(titles):
            ax.set_title(titles[idx])
        ax.axis("off")

    # Hide unused subplots
    for idx in range(n_images, nrows * ncols):
        axes.flatten()[idx].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    else:
        # Return as array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return img_array
