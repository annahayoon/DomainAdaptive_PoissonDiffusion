#!/usr/bin/env python3
"""
4-Panel Comparison Visualization for Paper Figures.

This module creates publication-quality 4-panel comparison figures showing:
- Our Result (DAPGD)
- L2 Result (baseline)
- Our Residuals
- L2 Residuals

With consistent colormap, intensity scaling, and statistical annotations.

Requirements addressed: 2.2 from evaluation_enhancement_todos.md
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase
import seaborn as sns

from core.metrics import PhysicsMetrics, MetricResult
from core.exceptions import AnalysisError

logger = logging.getLogger(__name__)

# Set style for publication quality
plt.style.use('default')
sns.set_palette("husl")


class ComparisonPlotter:
    """
    Creates 4-panel comparison visualizations for paper figures.

    This class handles the creation of publication-quality comparison figures
    that demonstrate the superiority of our physics-aware approach over L2 baseline.
    """

    def __init__(self, dpi: int = 300, figsize: Tuple[float, float] = (12, 10)):
        """
        Initialize comparison plotter.

        Args:
            dpi: Resolution for saved figures
            figsize: Figure size in inches
        """
        self.dpi = dpi
        self.figsize = figsize

        # Set matplotlib parameters for publication quality
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

        # Define color maps for different types of images
        self.colormaps = {
            'photography': 'viridis',
            'microscopy': 'plasma',
            'astronomy': 'inferno'
        }

        # Define intensity ranges for different domains (in electrons)
        self.intensity_ranges = {
            'photography': (0, 4000),
            'microscopy': (0, 1000),
            'astronomy': (0, 100)
        }

    def create_4panel_comparison(
        self,
        our_result: torch.Tensor,
        l2_result: torch.Tensor,
        our_residuals: torch.Tensor,
        l2_residuals: torch.Tensor,
        domain: str,
        our_metrics: Optional[Dict[str, float]] = None,
        l2_metrics: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        show_annotations: bool = True
    ) -> plt.Figure:
        """
        Create 4-panel comparison figure.

        Args:
            our_result: Our method's restoration result (normalized [0,1])
            l2_result: L2 baseline restoration result (normalized [0,1])
            our_residuals: Our method's residuals (electrons)
            l2_residuals: L2 baseline's residuals (electrons)
            domain: Domain name ('photography', 'microscopy', 'astronomy')
            our_metrics: Dictionary of metrics for our method
            l2_metrics: Dictionary of metrics for L2 method
            save_path: Path to save figure (optional)
            title: Figure title (optional)
            show_annotations: Whether to show statistical annotations

        Returns:
            matplotlib Figure object
        """
        # Convert to numpy for plotting
        our_result_np = self._tensor_to_numpy(our_result)
        l2_result_np = self._tensor_to_numpy(l2_result)
        our_residuals_np = self._tensor_to_numpy(our_residuals)
        l2_residuals_np = self._tensor_to_numpy(l2_residuals)

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # Get intensity range for this domain
        vmin, vmax = self.intensity_ranges.get(domain, (0, 1))

        # Get colormap for this domain
        cmap = self.colormaps.get(domain, 'viridis')

        # Plot 1: Our Result
        self._plot_image_panel(
            axes[0, 0], our_result_np, "Our Result (DAPGD)",
            vmin=vmin, vmax=vmax, cmap=cmap
        )

        # Plot 2: L2 Result
        self._plot_image_panel(
            axes[0, 1], l2_result_np, "L2 Baseline",
            vmin=vmin, vmax=vmax, cmap=cmap
        )

        # Plot 3: Our Residuals
        self._plot_residuals_panel(
            axes[1, 0], our_residuals_np, "Our Residuals",
            domain=domain, metrics=our_metrics
        )

        # Plot 4: L2 Residuals
        self._plot_residuals_panel(
            axes[1, 1], l2_residuals_np, "L2 Residuals",
            domain=domain, metrics=l2_metrics
        )

        # Add main title if provided
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # Add statistical annotations if requested
        if show_annotations and our_metrics and l2_metrics:
            self._add_statistical_annotations(
                axes, our_metrics, l2_metrics, domain
            )

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Figure saved to: {save_path}")

        return fig

    def _plot_image_panel(
        self,
        ax: plt.Axes,
        image: np.ndarray,
        title: str,
        vmin: float = 0,
        vmax: float = 1,
        cmap: str = 'viridis'
    ):
        """Plot a single image panel."""
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

        return im

    def _plot_residuals_panel(
        self,
        ax: plt.Axes,
        residuals: np.ndarray,
        title: str,
        domain: str,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Plot residuals panel with histogram."""
        # Main residual image
        im = ax.imshow(
            residuals,
            cmap='RdBu_r',
            vmin=-np.std(residuals)*3,
            vmax=np.std(residuals)*3,
            aspect='equal'
        )

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Residuals (electrons)', fontsize=9)

        # Add histogram as inset
        self._add_residual_histogram(ax, residuals, metrics)

    def _add_residual_histogram(
        self,
        ax: plt.Axes,
        residuals: np.ndarray,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Add histogram of residuals as inset plot."""
        # Create inset axes for histogram
        ax_inset = ax.inset_axes([0.65, 0.05, 0.3, 0.25])

        # Flatten residuals and remove outliers for histogram
        flat_residuals = residuals.flatten()
        mean_res = np.mean(flat_residuals)
        std_res = np.std(flat_residuals)

        # Remove outliers beyond 3 sigma
        mask = np.abs(flat_residuals - mean_res) < 3 * std_res
        clean_residuals = flat_residuals[mask]

        if len(clean_residuals) > 0:
            # Create histogram
            ax_inset.hist(
                clean_residuals, bins=30, density=True, alpha=0.7,
                color='blue', edgecolor='black', linewidth=0.5
            )

            # Add Gaussian fit
            x_gauss = np.linspace(mean_res - 3*std_res, mean_res + 3*std_res, 100)
            y_gauss = (1/(std_res * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x_gauss - mean_res)/std_res)**2)

            ax_inset.plot(x_gauss, y_gauss, 'r-', linewidth=1.5, alpha=0.8, label='Gaussian')

            # Add statistical annotations
            if metrics:
                chi2 = metrics.get('chi2_consistency', 0)
                residual_std = metrics.get('residual_std', std_res)

                ax_inset.text(
                    0.05, 0.95, f'χ² = {chi2:.2f}\nσ = {residual_std:.1f}',
                    transform=ax_inset.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

            ax_inset.legend(fontsize=7)
            ax_inset.grid(True, alpha=0.3)

        # Remove ticks
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

    def _add_statistical_annotations(
        self,
        axes: np.ndarray,
        our_metrics: Dict[str, float],
        l2_metrics: Dict[str, float],
        domain: str
    ):
        """Add statistical comparison annotations."""
        # Extract key metrics
        our_chi2 = our_metrics.get('chi2_consistency', 0)
        l2_chi2 = l2_metrics.get('chi2_consistency', 0)

        our_psnr = our_metrics.get('psnr', 0)
        l2_psnr = l2_metrics.get('psnr', 0)

        # Add annotations to result panels
        self._add_metric_annotation(axes[0, 0], our_psnr, 'PSNR')
        self._add_metric_annotation(axes[0, 1], l2_psnr, 'PSNR')

        # Add method comparison
        self._add_method_comparison(axes, our_chi2, l2_chi2, domain)

    def _add_metric_annotation(self, ax: plt.Axes, value: float, label: str):
        """Add metric annotation to panel."""
        ax.text(
            0.05, 0.95, f'{label}: {value:.2f}',
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            verticalalignment='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, pad=0.3)
        )

    def _add_method_comparison(self, axes: np.ndarray, our_chi2: float, l2_chi2: float, domain: str):
        """Add comparison between methods."""
        # Create annotation text
        if our_chi2 < l2_chi2:
            comparison_text = f"✓ DAPGD χ² = {our_chi2:.2f} better than L2 χ² = {l2_chi2:.2f}"
        else:
            comparison_text = f"L2 χ² = {l2_chi2:.2f} better than DAPGD χ² = {our_chi2:.2f}"
        # Add to figure
        axes[0, 0].figure.text(
            0.02, 0.02, comparison_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            verticalalignment='bottom'
        )

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array for plotting."""
        if tensor.dim() > 3:
            # Take first batch and first channel if multi-dimensional
            tensor = tensor[0, 0] if tensor.dim() > 3 else tensor.squeeze()

        return tensor.detach().cpu().numpy()

    def create_domain_comparison_grid(
        self,
        results_dict: Dict[str, Dict[str, torch.Tensor]],
        save_dir: str,
        domain_order: List[str] = None
    ):
        """
        Create comparison grid for multiple domains.

        Args:
            results_dict: Nested dict with domain -> method -> tensors
            save_dir: Directory to save figures
            domain_order: Order of domains for plotting
        """
        if domain_order is None:
            domain_order = ['photography', 'microscopy', 'astronomy']

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for domain in domain_order:
            if domain not in results_dict:
                logger.warning(f"Domain {domain} not found in results")
                continue

            domain_data = results_dict[domain]

            # Extract tensors for this domain
            our_result = domain_data.get('dapgd_result')
            l2_result = domain_data.get('l2_result')
            our_residuals = domain_data.get('dapgd_residuals')
            l2_residuals = domain_data.get('l2_residuals')

            if None in [our_result, l2_result, our_residuals, l2_residuals]:
                logger.warning(f"Missing data for domain {domain}")
                continue

            # Get metrics
            our_metrics = domain_data.get('dapgd_metrics', {})
            l2_metrics = domain_data.get('l2_metrics', {})

            # Create figure
            fig = self.create_4panel_comparison(
                our_result=our_result,
                l2_result=l2_result,
                our_residuals=our_residuals,
                l2_residuals=l2_residuals,
                domain=domain,
                our_metrics=our_metrics,
                l2_metrics=l2_metrics,
                title=f"Domain Comparison: {domain.title()}",
                save_path=str(save_path / f"comparison_{domain}.png")
            )

            plt.close(fig)  # Close to free memory

        logger.info(f"Created comparison figures for {len(domain_order)} domains in {save_dir}")

    def create_residual_analysis_plots(
        self,
        our_residuals: torch.Tensor,
        l2_residuals: torch.Tensor,
        domain: str,
        save_path: Optional[str] = None
    ):
        """
        Create detailed residual analysis plots.

        Args:
            our_residuals: Our method's residuals
            l2_residuals: L2 method's residuals
            domain: Domain name
            save_path: Path to save figure
        """
        our_res_np = self._tensor_to_numpy(our_residuals)
        l2_res_np = self._tensor_to_numpy(l2_residuals)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Residual images
        axes[0, 0].imshow(our_res_np, cmap='RdBu_r')
        axes[0, 0].set_title('Our Residuals')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(l2_res_np, cmap='RdBu_r')
        axes[0, 1].set_title('L2 Residuals')
        axes[0, 1].axis('off')

        # Histograms
        axes[0, 2].hist(our_res_np.flatten(), bins=50, alpha=0.7, label='Our', density=True)
        axes[0, 2].hist(l2_res_np.flatten(), bins=50, alpha=0.7, label='L2', density=True)
        axes[0, 2].set_xlabel('Residual Value')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].legend()
        axes[0, 2].set_title('Residual Distributions')

        # Power spectra
        our_freq, our_psd = self._compute_power_spectrum(our_res_np)
        l2_freq, l2_psd = self._compute_power_spectrum(l2_res_np)

        axes[1, 0].loglog(our_freq, our_psd, label='Our')
        axes[1, 0].loglog(l2_freq, l2_psd, label='L2')
        axes[1, 0].set_xlabel('Spatial Frequency')
        axes[1, 0].set_ylabel('Power Spectral Density')
        axes[1, 0].legend()
        axes[1, 0].set_title('Power Spectra')
        axes[1, 0].grid(True, alpha=0.3)

        # Autocorrelation
        our_acorr = self._compute_autocorrelation(our_res_np)
        l2_acorr = self._compute_autocorrelation(l2_res_np)

        axes[1, 1].plot(our_acorr, label='Our')
        axes[1, 1].plot(l2_acorr, label='L2')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].legend()
        axes[1, 1].set_title('Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)

        # QQ plot
        from scipy import stats
        our_flat = our_res_np.flatten()
        l2_flat = l2_res_np.flatten()

        # Normalize for comparison
        our_norm = (our_flat - np.mean(our_flat)) / np.std(our_flat)
        l2_norm = (l2_flat - np.mean(l2_flat)) / np.std(l2_flat)

        stats.probplot(our_norm, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q Plot (Our vs Normal)')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Residual analysis saved to: {save_path}")

        return fig

    def _compute_power_spectrum(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectrum of image."""
        from scipy.signal import periodogram

        # Compute 2D power spectrum
        freq_x = np.fft.fftfreq(image.shape[1])
        freq_y = np.fft.fftfreq(image.shape[0])

        # 2D FFT
        fft_2d = np.fft.fft2(image)
        power_spectrum = np.abs(fft_2d)**2

        # Radial average (simplified)
        center = (image.shape[0] // 2, image.shape[1] // 2)
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

        # Bin by radius
        r_flat = r.flatten()
        psd_flat = power_spectrum.flatten()

        # Simple averaging for demonstration
        max_r = int(np.max(r_flat)) + 1
        radial_psd = np.zeros(max_r)

        for i in range(max_r):
            mask = (r_flat >= i) & (r_flat < i + 1)
            if np.any(mask):
                radial_psd[i] = np.mean(psd_flat[mask])

        freqs = np.arange(max_r) / max(image.shape[0], image.shape[1])
        return freqs[1:], radial_psd[1:]  # Skip DC component

    def _compute_autocorrelation(self, image: np.ndarray) -> np.ndarray:
        """Compute autocorrelation of image."""
        # Simple 1D autocorrelation along center row
        center_row = image[image.shape[0] // 2]
        center_row = center_row - np.mean(center_row)

        # Compute autocorrelation
        result = np.correlate(center_row, center_row, mode='full')
        return result[result.size // 2:]  # Return positive lags only


def create_paper_figure(
    our_result: torch.Tensor,
    l2_result: torch.Tensor,
    our_residuals: torch.Tensor,
    l2_residuals: torch.Tensor,
    domain: str,
    our_metrics: Optional[Dict[str, float]] = None,
    l2_metrics: Optional[Dict[str, float]] = None,
    save_path: str = "comparison_figure.png"
) -> str:
    """
    Convenience function to create a single comparison figure.

    Args:
        our_result: Our method's restoration result
        l2_result: L2 baseline restoration result
        our_residuals: Our method's residuals
        l2_residuals: L2 baseline's residuals
        domain: Domain name
        our_metrics: Metrics for our method
        l2_metrics: Metrics for L2 method
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    plotter = ComparisonPlotter()

    fig = plotter.create_4panel_comparison(
        our_result=our_result,
        l2_result=l2_result,
        our_residuals=our_residuals,
        l2_residuals=l2_residuals,
        domain=domain,
        our_metrics=our_metrics,
        l2_metrics=l2_metrics,
        save_path=save_path
    )

    plt.close(fig)  # Close to free memory
    return save_path


if __name__ == "__main__":
    # Example usage
    print("Comparison Plotter module loaded successfully")

    # Create dummy data for testing
    dummy_image = torch.rand(1, 1, 128, 128)
    dummy_residuals = torch.randn(1, 1, 128, 128) * 10

    plotter = ComparisonPlotter()

    fig = plotter.create_4panel_comparison(
        our_result=dummy_image,
        l2_result=dummy_image + 0.1 * torch.randn_like(dummy_image),
        our_residuals=dummy_residuals,
        l2_residuals=dummy_residuals + 5 * torch.randn_like(dummy_residuals),
        domain="photography",
        our_metrics={"psnr": 28.5, "chi2_consistency": 1.2},
        l2_metrics={"psnr": 26.3, "chi2_consistency": 2.8}
    )

    print("Test figure created successfully")
    plt.close(fig)
