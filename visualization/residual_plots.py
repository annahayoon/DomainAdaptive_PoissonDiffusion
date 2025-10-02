"""
Residual visualization plots for Poisson-Gaussian diffusion validation.

This module creates publication-quality plots for residual analysis:
1. 4-panel comparison plots (method vs baseline)
2. Residual distribution plots
3. Spatial correlation heatmaps
4. Power spectral density plots
5. Statistical validation summaries

Requirements addressed: 2.3.1-2.3.5 from evaluation_enhancement_todos.md
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from scipy import stats

from core.exceptions import AnalysisError
from core.residual_analysis import ResidualValidationReport

# Set up logging
logger = logging.getLogger(__name__)


class ResidualPlotter:
    """Create publication-quality residual analysis plots."""

    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 150):
        """
        Initialize residual plotter.

        Args:
            figsize: Figure size (width, height) in inches
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style_params = {
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
        }

    def create_4panel_comparison(
        self,
        pred_our: torch.Tensor,
        pred_baseline: torch.Tensor,
        noisy: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        background: float = 0.0,
        read_noise: float = 0.0,
        mask: Optional[torch.Tensor] = None,
        method_name: str = "DAPGD",
        baseline_name: str = "L2-Baseline",
        domain: str = "unknown",
        output_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Create 4-panel comparison plot for publication.

        Args:
            pred_our: Our method prediction [B, C, H, W] (normalized)
            pred_baseline: Baseline prediction [B, C, H, W] (normalized)
            noisy: Noisy observation [B, C, H, W] (electrons)
            target: Ground truth [B, C, H, W] (normalized, optional)
            scale: Normalization scale
            background: Background offset
            read_noise: Read noise
            mask: Valid pixel mask
            method_name: Name of our method
            baseline_name: Name of baseline method
            domain: Domain name
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Set style
        with plt.style.context(['default'] + list(self.style_params.items())):
            fig, axes = plt.subplots(2, 4, figsize=self.figsize)

            # Select first image from batch
            our_pred = pred_our[0, 0].detach().cpu().numpy()
            baseline_pred = pred_baseline[0, 0].detach().cpu().numpy()
            noisy_img = noisy[0, 0].detach().cpu().numpy()

            if target is not None:
                target_img = target[0, 0].detach().cpu().numpy()
            else:
                target_img = None

            # Compute residuals
            our_pred_electrons = our_pred * scale + background
            baseline_pred_electrons = baseline_pred * scale + background

            our_residuals = noisy_img - our_pred_electrons
            baseline_residuals = noisy_img - baseline_pred_electrons

            expected_var_our = our_pred_electrons + read_noise**2
            expected_var_baseline = baseline_pred_electrons + read_noise**2

            our_norm_residuals = our_residuals / np.sqrt(expected_var_our)
            baseline_norm_residuals = baseline_residuals / np.sqrt(expected_var_baseline)

            if mask is not None:
                mask_img = mask[0, 0].detach().cpu().numpy()
                our_norm_residuals = our_norm_residuals * mask_img
                baseline_norm_residuals = baseline_norm_residuals * mask_img

            # Panel 1: Our method result
            im1 = axes[0, 0].imshow(our_pred, cmap='gray', vmin=0, vmax=1)
            axes[0, 0].set_title(f'{method_name} Result', fontweight='bold')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

            # Panel 2: Baseline result
            im2 = axes[0, 1].imshow(baseline_pred, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title(f'{baseline_name} Result', fontweight='bold')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

            # Panel 3: Our method residuals
            im3 = axes[0, 2].imshow(our_norm_residuals, cmap='RdBu_r', vmin=-3, vmax=3)
            axes[0, 2].set_title('Our Residuals (Normalized)', fontweight='bold')
            axes[0, 2].axis('off')
            plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

            # Panel 4: Baseline residuals
            im4 = axes[0, 3].imshow(baseline_norm_residuals, cmap='RdBu_r', vmin=-3, vmax=3)
            axes[0, 3].set_title('Baseline Residuals (Normalized)', fontweight='bold')
            axes[0, 3].axis('off')
            plt.colorbar(im4, ax=axes[0, 3], fraction=0.046, pad=0.04)

            # Bottom row: Statistics and analysis
            # Panel 5: Our residual histogram
            axes[1, 0].hist(our_norm_residuals.flatten(), bins=50, density=True,
                          alpha=0.7, color='blue', label='Our residuals')
            # Overlay normal distribution
            x = np.linspace(-4, 4, 100)
            y = stats.norm.pdf(x, 0, 1)
            axes[1, 0].plot(x, y, 'r-', linewidth=2, label='N(0,1)')
            axes[1, 0].set_xlabel('Normalized Residuals')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Our Residual Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Panel 6: Baseline residual histogram
            axes[1, 1].hist(baseline_norm_residuals.flatten(), bins=50, density=True,
                          alpha=0.7, color='orange', label='Baseline residuals')
            axes[1, 1].plot(x, y, 'r-', linewidth=2, label='N(0,1)')
            axes[1, 1].set_xlabel('Normalized Residuals')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Baseline Residual Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # Panel 7: Q-Q plot for our method
            valid_residuals_our = our_norm_residuals.flatten()
            valid_residuals_our = valid_residuals_our[np.isfinite(valid_residuals_our)]
            if len(valid_residuals_our) > 100:
                # Simple Q-Q plot
                quantiles = np.linspace(0.01, 0.99, 100)
                theoretical_quantiles = stats.norm.ppf(quantiles)
                sample_quantiles = np.quantile(valid_residuals_our, quantiles)

                axes[1, 2].scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, s=20)
                # Reference line
                min_val = min(np.min(theoretical_quantiles), np.min(sample_quantiles))
                max_val = max(np.max(theoretical_quantiles), np.max(sample_quantiles))
                axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7,
                              label='Reference line')
                axes[1, 2].set_xlabel('Theoretical Quantiles')
                axes[1, 2].set_ylabel('Sample Quantiles')
                axes[1, 2].set_title('Q-Q Plot (Our Method)')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)

            # Panel 8: Statistics summary
            axes[1, 3].axis('off')

            # Compute statistics
            our_mean = np.mean(our_norm_residuals)
            our_std = np.std(our_norm_residuals)
            baseline_mean = np.mean(baseline_norm_residuals)
            baseline_std = np.std(baseline_norm_residuals)

            # KS test
            our_finite = our_norm_residuals[np.isfinite(our_norm_residuals)]
            baseline_finite = baseline_norm_residuals[np.isfinite(baseline_norm_residuals)]

            if len(our_finite) > 10 and len(baseline_finite) > 10:
                ks_stat_our, ks_p_our = stats.kstest(our_finite, 'norm')
                ks_stat_baseline, ks_p_baseline = stats.kstest(baseline_finite, 'norm')
            else:
                ks_stat_our, ks_p_our = float('nan'), float('nan')
                ks_stat_baseline, ks_p_baseline = float('nan'), float('nan')

            # Create statistics text
            stats_text = f"""
            {method_name} vs {baseline_name} Comparison

            Our Method Statistics:
            • Mean: {our_mean:.4f}
            • Std: {our_std:.4f}
            • KS vs N(0,1): {ks_stat_our:.4f} (p={ks_p_our:.4f})

            Baseline Statistics:
            • Mean: {baseline_mean:.4f}
            • Std: {baseline_std:.4f}
            • KS vs N(0,1): {ks_stat_baseline:.4f} (p={ks_p_baseline:.4f})

            Domain: {domain}
            Scale: {scale}
            Read Noise: {read_noise}
            """

            axes[1, 3].text(0.05, 0.95, stats_text, transform=axes[1, 3].transAxes,
                          fontsize=9, verticalalignment='top', fontfamily='monospace')
            axes[1, 3].set_title('Statistical Summary')

            plt.tight_layout()

            # Save if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"4-panel comparison plot saved to {output_path}")

            return fig

    def create_residual_validation_plots(
        self,
        report: ResidualValidationReport,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Create comprehensive residual validation plots from a validation report.

        Args:
            report: Residual validation report
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        with plt.style.context(['default'] + list(self.style_params.items())):
            fig = plt.figure(figsize=(16, 12))

            # Create subplot grid
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

            # Panel 1: Statistical test results
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_statistical_tests(ax1, report)

            # Panel 2: Spatial correlation analysis
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_spatial_correlation(ax2, report)

            # Panel 3: Spectral analysis
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_spectral_analysis(ax3, report)

            # Panel 4: Distribution statistics
            ax4 = fig.add_subplot(gs[0, 3])
            self._plot_distribution_stats(ax4, report)

            # Panel 5: Overall assessment (large panel)
            ax5 = fig.add_subplot(gs[1:, :2])
            self._plot_overall_assessment(ax5, report)

            # Panel 6: Recommendations
            ax6 = fig.add_subplot(gs[1:, 2:])
            self._plot_recommendations(ax6, report)

            plt.suptitle(f'Residual Validation Report: {report.method_name}',
                        fontsize=16, fontweight='bold')

            # Save if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Residual validation plots saved to {output_path}")

            return fig

    def _plot_statistical_tests(self, ax, report: ResidualValidationReport) -> None:
        """Plot statistical test results."""
        tests = ['KS Test', 'Shapiro Test', 'Anderson Test']
        pvalues = [report.ks_pvalue, report.shapiro_pvalue, report.ks_pvalue]
        passed = [report.gaussian_fit, report.normal_by_shapiro, report.normal_by_anderson]

        colors = ['green' if p else 'red' for p in passed]
        bars = ax.bar(tests, pvalues, color=colors, alpha=0.7)

        ax.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='α = 0.05')
        ax.set_ylabel('p-value')
        ax.set_title('Normality Tests')
        ax.set_ylim(0, 1)
        ax.legend()

        for bar, pval in zip(bars, pvalues):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{pval:.3f}', ha='center', va='bottom', fontsize=8)

    def _plot_spatial_correlation(self, ax, report: ResidualValidationReport) -> None:
        """Plot spatial correlation analysis."""
        # Create a simple correlation matrix visualization
        if hasattr(report, 'autocorrelation_matrix') and report.autocorrelation_matrix is not None:
            im = ax.imshow(report.autocorrelation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title('Spatial Autocorrelation')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Lag')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            # Fallback: text summary
            ax.text(0.5, 0.5, f'X-autocorr: {report.autocorrelation_lag1_x:.3f}\n'
                              f'Y-autocorr: {report.autocorrelation_lag1_y:.3f}\n'
                              f'Uncorrelated: {report.spatial_uncorrelated}',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax.set_title('Spatial Correlation')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

    def _plot_spectral_analysis(self, ax, report: ResidualValidationReport) -> None:
        """Plot spectral analysis results."""
        spectral_props = ['Spectral\nFlatness', 'Spectral\nSlope', 'High-freq\nPower']
        spectral_values = [report.spectral_flatness, abs(report.spectral_slope), report.high_freq_power]
        colors = ['blue', 'orange', 'green']

        bars = ax.bar(spectral_props, spectral_values, color=colors, alpha=0.7)
        ax.set_ylabel('Value')
        ax.set_title('Spectral Analysis')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='White noise threshold')

        for bar, val in zip(bars, spectral_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax.legend()

    def _plot_distribution_stats(self, ax, report: ResidualValidationReport) -> None:
        """Plot distribution statistics."""
        stats_names = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis']
        stats_values = [report.mean, report.std_dev, report.skewness, report.kurtosis]

        bars = ax.bar(stats_names, stats_values, color='lightcoral', alpha=0.7)
        ax.set_ylabel('Value')
        ax.set_title('Distribution Statistics')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.7)

        for bar, val in zip(bars, stats_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    def _plot_overall_assessment(self, ax, report: ResidualValidationReport) -> None:
        """Plot overall assessment results."""
        ax.text(0.5, 0.8, f'Overall Physics Correctness',
               ha='center', va='center', fontsize=16, fontweight='bold')

        # Color-coded result
        color = 'green' if report.physics_correct else 'red'
        status = 'PASS ✓' if report.physics_correct else 'FAIL ✗'
        ax.text(0.5, 0.6, status, ha='center', va='center', fontsize=24,
               color=color, fontweight='bold')

        # Summary text
        ax.text(0.5, 0.3, report.validation_summary,
               ha='center', va='center', fontsize=12, wrap=True)

        # Sample count
        ax.text(0.5, 0.1, f'Based on {report.n_samples:,} samples',
               ha='center', va='center', fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def _plot_recommendations(self, ax, report: ResidualValidationReport) -> None:
        """Plot recommendations."""
        ax.text(0.5, 0.95, 'Recommendations', ha='center', va='center',
               fontsize=14, fontweight='bold')

        if report.recommendations:
            y_pos = 0.85
            for i, rec in enumerate(report.recommendations, 1):
                ax.text(0.05, y_pos, f"{i}. {rec}", ha='left', va='top',
                       fontsize=10, wrap=True)
                y_pos -= 0.15
        else:
            ax.text(0.5, 0.5, 'No specific recommendations',
                   ha='center', va='center', fontsize=12)

        # Add metadata
        metadata_text = f"""
        Method: {report.method_name}
        Domain: {report.domain}
        Dataset: {report.dataset_name}
        Scale: {report.scale}
        Read Noise: {report.read_noise}
        Processing: {report.processing_time:.2f}s
        """

        ax.text(0.05, 0.05, metadata_text, ha='left', va='bottom',
               fontsize=8, fontfamily='monospace')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def create_residual_diagnostic_plots(
        self,
        pred: torch.Tensor,
        noisy: torch.Tensor,
        scale: float,
        background: float = 0.0,
        read_noise: float = 0.0,
        mask: Optional[torch.Tensor] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Create detailed diagnostic plots for residual analysis.

        Args:
            pred: Predicted image [B, C, H, W]
            noisy: Noisy observation [B, C, H, W]
            scale: Normalization scale
            background: Background offset
            read_noise: Read noise
            mask: Valid pixel mask
            output_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        with plt.style.context(['default'] + list(self.style_params.items())):
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))

            # Select first image from batch
            pred_img = pred[0, 0].detach().cpu().numpy()
            noisy_img = noisy[0, 0].detach().cpu().numpy()

            # Compute residuals
            pred_electrons = pred_img * scale + background
            residuals = noisy_img - pred_electrons
            expected_var = pred_electrons + read_noise**2
            norm_residuals = residuals / np.sqrt(expected_var)

            if mask is not None:
                mask_img = mask[0, 0].detach().cpu().numpy()
                norm_residuals = norm_residuals * mask_img

            # Panel 1: Residual image
            im1 = axes[0, 0].imshow(norm_residuals, cmap='RdBu_r', vmin=-4, vmax=4)
            axes[0, 0].set_title('Normalized Residuals')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0])

            # Panel 2: Residual histogram with normal fit
            valid_residuals = norm_residuals[np.isfinite(norm_residuals)]
            axes[0, 1].hist(valid_residuals, bins=50, density=True, alpha=0.7)
            x = np.linspace(-4, 4, 100)
            y = stats.norm.pdf(x, 0, 1)
            axes[0, 1].plot(x, y, 'r-', linewidth=2, label='N(0,1)')
            axes[0, 1].set_xlabel('Normalized Residuals')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].set_title('Residual Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Panel 3: Q-Q plot
            if len(valid_residuals) > 100:
                quantiles = np.linspace(0.01, 0.99, 100)
                theoretical_quantiles = stats.norm.ppf(quantiles)
                sample_quantiles = np.quantile(valid_residuals, quantiles)

                axes[0, 2].scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, s=20)
                min_val = min(np.min(theoretical_quantiles), np.min(sample_quantiles))
                max_val = max(np.max(theoretical_quantiles), np.max(sample_quantiles))
                axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
                axes[0, 2].set_xlabel('Theoretical Quantiles')
                axes[0, 2].set_ylabel('Sample Quantiles')
                axes[0, 2].set_title('Q-Q Plot')
                axes[0, 2].grid(True, alpha=0.3)

            # Panel 4: Spatial autocorrelation (X direction)
            if norm_residuals.shape[0] > 10 and norm_residuals.shape[1] > 10:
                # Compute autocorrelation for rows
                row_autocorr = []
                for i in range(min(10, norm_residuals.shape[0])):
                    row = norm_residuals[i]
                    if np.std(row) > 1e-10:
                        autocorr = np.correlate(row, row, mode='full')
                        autocorr = autocorr[autocorr.size // 2:]
                        autocorr = autocorr / autocorr[0]
                        row_autocorr.append(autocorr[:20])  # First 20 lags

                if row_autocorr:
                    avg_autocorr = np.mean(row_autocorr, axis=0)
                    lags = np.arange(len(avg_autocorr))
                    axes[1, 0].plot(lags, avg_autocorr, 'b-')
                    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    axes[1, 0].set_xlabel('Lag')
                    axes[1, 0].set_ylabel('Autocorrelation')
                    axes[1, 0].set_title('Spatial Autocorrelation (X)')
                    axes[1, 0].grid(True, alpha=0.3)

            # Panel 5: Power spectral density
            try:
                from scipy.signal import periodogram
                # Compute PSD for each row and average
                row_psds = []
                for row in norm_residuals:
                    if np.std(row) > 1e-10:
                        freqs, psd = periodogram(row, fs=1.0)
                        row_psds.append(psd)

                if row_psds:
                    avg_psd = np.mean(row_psds, axis=0)
                    axes[1, 1].loglog(freqs, avg_psd, 'b-')
                    axes[1, 1].set_xlabel('Frequency')
                    axes[1, 1].set_ylabel('Power Spectral Density')
                    axes[1, 1].set_title('Power Spectrum')
                    axes[1, 1].grid(True, alpha=0.3)
            except ImportError:
                axes[1, 1].text(0.5, 0.5, 'PSD computation\nrequires scipy',
                               ha='center', va='center', transform=axes[1, 1].transAxes)

            # Panel 6: Residual vs predicted (scatter)
            valid_pred = pred_electrons[np.isfinite(norm_residuals)]
            valid_resid = norm_residuals[np.isfinite(norm_residuals)]

            if len(valid_pred) > 1000:
                # Sample for plotting
                idx = np.random.choice(len(valid_pred), 1000, replace=False)
                axes[1, 2].scatter(valid_pred[idx], valid_resid[idx],
                                alpha=0.1, s=1, color='blue')
                axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.7)
                axes[1, 2].set_xlabel('Predicted Signal (electrons)')
                axes[1, 2].set_ylabel('Normalized Residuals')
                axes[1, 2].set_title('Residuals vs Prediction')
                axes[1, 2].set_xlim(0, np.percentile(valid_pred, 99))
                axes[1, 2].grid(True, alpha=0.3)

            # Panel 7: Cumulative distribution
            sorted_residuals = np.sort(valid_residuals)
            y = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)
            axes[2, 0].plot(sorted_residuals, y, 'b-', label='Empirical CDF')
            # Normal CDF
            x = np.linspace(-4, 4, 100)
            y_norm = stats.norm.cdf(x, 0, 1)
            axes[2, 0].plot(x, y_norm, 'r--', label='Normal CDF')
            axes[2, 0].set_xlabel('Normalized Residuals')
            axes[2, 0].set_ylabel('Cumulative Probability')
            axes[2, 0].set_title('Cumulative Distribution')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

            # Panel 8: Statistics summary
            axes[2, 1].axis('off')

            # Compute comprehensive statistics
            mean_val = np.mean(valid_residuals)
            std_val = np.std(valid_residuals)
            skew_val = stats.skew(valid_residuals)
            kurt_val = stats.kurtosis(valid_residuals)

            ks_stat, ks_pval = stats.kstest(valid_residuals, 'norm')

            stats_text = f"""
            Residual Analysis Summary

            Basic Statistics:
            • Mean: {mean_val:.6f}
            • Std Dev: {std_val:.6f}
            • Skewness: {skew_val:.4f}
            • Kurtosis: {kurt_val:.4f}
            • N samples: {len(valid_residuals):,}

            Normality Tests:
            • KS statistic: {ks_stat:.4f}
            • KS p-value: {ks_pval:.4f}
            • Normal: {ks_pval > 0.05}

            Scale: {scale}
            Read Noise: {read_noise}
            """

            axes[2, 1].text(0.1, 0.9, stats_text, transform=axes[2, 1].transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace')

            # Panel 9: Variance analysis
            # Plot expected vs actual variance
            if len(valid_pred) > 100:
                # Bin by predicted value
                pred_bins = np.linspace(0, np.percentile(valid_pred, 95), 10)
                bin_centers = (pred_bins[:-1] + pred_bins[1:]) / 2

                actual_vars = []
                expected_vars = []

                for i in range(len(pred_bins) - 1):
                    mask = (valid_pred >= pred_bins[i]) & (valid_pred < pred_bins[i + 1])
                    if np.sum(mask) > 10:
                        actual_vars.append(np.var(valid_resid[mask]))
                        expected_vars.append(np.mean(expected_var[np.isfinite(norm_residuals)][mask]))

                if actual_vars:
                    axes[2, 2].plot(bin_centers[:len(actual_vars)], actual_vars, 'bo-', label='Actual')
                    axes[2, 2].plot(bin_centers[:len(expected_vars)], expected_vars, 'ro-', label='Expected')
                    axes[2, 2].set_xlabel('Predicted Signal')
                    axes[2, 2].set_ylabel('Variance')
                    axes[2, 2].set_title('Variance Analysis')
                    axes[2, 2].legend()
                    axes[2, 2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Residual diagnostic plots saved to {output_path}")

            return fig


def create_publication_plots(
    validation_reports: List[ResidualValidationReport],
    output_dir: Union[str, Path],
    method_names: Optional[List[str]] = None,
) -> None:
    """
    Create publication-ready plots comparing multiple methods.

    Args:
        validation_reports: List of validation reports
        output_dir: Directory to save plots
        method_names: Optional list of method names for legend
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter = ResidualPlotter()

    # Group reports by domain
    domain_reports = {}
    for report in validation_reports:
        domain = report.domain
        if domain not in domain_reports:
            domain_reports[domain] = []
        domain_reports[domain].append(report)

    # Create comparison plots for each domain
    for domain, reports in domain_reports.items():
        if len(reports) < 2:
            continue

        logger.info(f"Creating publication plots for {domain} domain with {len(reports)} methods")

        # Create statistical comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Extract method names
        methods = [r.method_name for r in reports]
        if method_names:
            methods = [method_names[i] if i < len(method_names) else m
                      for i, m in enumerate(methods)]

        # Plot 1: KS statistics
        ks_stats = [r.ks_statistic for r in reports]
        colors = plt.cm.tab10(np.linspace(0, 1, len(reports)))

        bars = axes[0, 0].bar(methods, ks_stats, color=colors)
        axes[0, 0].set_ylabel('KS Statistic')
        axes[0, 0].set_title(f'Normality Test (KS) - {domain}')
        axes[0, 0].tick_params(axis='x', rotation=45)

        for bar, ks in zip(bars, ks_stats):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{ks:.3f}', ha='center', va='bottom')

        # Plot 2: Physics correctness rate
        physics_correct = [r.physics_correct for r in reports]

        bars = axes[0, 1].bar(methods, physics_correct, color=colors)
        axes[0, 1].set_ylabel('Physics Correct Rate')
        axes[0, 1].set_title(f'Overall Physics Correctness - {domain}')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)

        for bar, correct in zip(bars, physics_correct):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{correct:.2f}', ha='center', va='bottom')

        # Plot 3: Spatial correlation
        spatial_uncorr = [r.spatial_uncorrelated for r in reports]

        bars = axes[1, 0].bar(methods, spatial_uncorr, color=colors)
        axes[1, 0].set_ylabel('Spatially Uncorrelated Rate')
        axes[1, 0].set_title(f'Spatial Correlation - {domain}')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)

        for bar, uncorr in zip(bars, spatial_uncorr):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{uncorr:.2f}', ha='center', va='bottom')

        # Plot 4: Spectral flatness
        spectral_flatness = [r.spectral_flatness for r in reports]

        bars = axes[1, 1].bar(methods, spectral_flatness, color=colors)
        axes[1, 1].set_ylabel('Spectral Flatness')
        axes[1, 1].set_title(f'Spectral Analysis - {domain}')
        axes[1, 1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].tick_params(axis='x', rotation=45)

        for bar, flatness in zip(bars, spectral_flatness):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{flatness:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / f"residual_comparison_{domain}.png",
                   dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Publication plots saved for {domain} domain")
