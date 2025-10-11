"""
Statistical validation for Poisson-Gaussian reconstruction.

This module checks if your restored images are statistically consistent
with the Poisson-Gaussian noise model. If the model is correct, the
normalized residuals should follow N(0,1), giving χ² ≈ 1.0.

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats


class StatisticalValidator:
    """
    Validates reconstruction consistency with Poisson-Gaussian noise model.

    Usage:
    validator = StatisticalValidator(guidance_module)
    results, is_consistent = validator.validate(noisy_image, restored_image)
    print(f"χ² = {results['chi_squared']:.3f}")
    """

    def __init__(self, guidance, save_dir=None):
        """
        Args:
            guidance: Your guidance module with noise parameters (gain, sigma_r, max_adu)
            save_dir: Optional directory to save diagnostic plots
        """
        self.guidance = guidance
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def compute_residuals(self, y_obs, x_pred):
        """
        Compute normalized residuals.

        For Poisson-Gaussian model, normalized residuals should be N(0,1):
        z = (y - x) / sqrt(gain * x + sigma_r²)

        Args:
            y_obs: Observed noisy image [B, C, H, W] in [-1, 1]
            x_pred: Predicted clean image [B, C, H, W] in [-1, 1]

        Returns:
            residuals: Normalized residuals [B, C, H, W]
        """
        # Convert from model space [-1, 1] to physical units [0, max_adu]
        y_phys = (y_obs + 1.0) * 0.5 * self.guidance.max_adu
        x_phys = (x_pred + 1.0) * 0.5 * self.guidance.max_adu

        # Ensure non-negative (clamp to small positive value)
        x_phys = torch.clamp(x_phys, min=1e-3)

        # Compute variance under Poisson-Gaussian model
        # Var(y|x) = gain * x + sigma_r²
        variance = self.guidance.gain * x_phys + self.guidance.sigma_r**2

        # Normalized residuals: should be N(0,1) if model is correct
        residuals = (y_phys - x_phys) / torch.sqrt(variance)

        return residuals

    def compute_chi_squared(self, residuals):
        """
        Compute reduced chi-squared statistic.

        χ² = (1/N) * Σ(residuals²)

        Should be ≈ 1.0 for consistent reconstruction.

        Args:
            residuals: Normalized residuals [B, C, H, W]

        Returns:
            chi_squared: Float value (target: ~1.0)
        """
        chi_squared = (residuals**2).mean().item()
        return chi_squared

    def test_normality(self, residuals, sample_size=5000):
        """
        Test if residuals follow standard normal distribution.

        Uses Shapiro-Wilk test. p-value > 0.05 indicates normality.

        Args:
            residuals: Normalized residuals [B, C, H, W]
            sample_size: Max samples for test (Shapiro-Wilk limited to 5000)

        Returns:
            dict with 'statistic', 'p_value', 'is_normal'
        """
        residuals_flat = residuals.flatten().cpu().numpy()

        # Subsample if too large
        if len(residuals_flat) > sample_size:
            indices = np.random.choice(len(residuals_flat), sample_size, replace=False)
            residuals_flat = residuals_flat[indices]

        # Shapiro-Wilk test for normality
        try:
            statistic, p_value = scipy_stats.shapiro(residuals_flat)
        except Exception as e:
            print(f"Warning: Normality test failed: {e}")
            statistic, p_value = 0.0, 0.0

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > 0.05,
        }

    def compute_moments(self, residuals):
        """
        Compute statistical moments of residuals.

        For N(0,1): mean=0, std=1, skewness=0, kurtosis=0

        Args:
            residuals: Normalized residuals [B, C, H, W]

        Returns:
            dict with 'mean', 'std', 'skewness', 'kurtosis'
        """
        residuals_flat = residuals.flatten().cpu().numpy()

        return {
            "mean": float(residuals_flat.mean()),
            "std": float(residuals_flat.std()),
            "skewness": float(scipy_stats.skew(residuals_flat)),
            "kurtosis": float(scipy_stats.kurtosis(residuals_flat)),
        }

    def validate(self, y_obs, x_pred, save_prefix=None):
        """
        Full statistical validation.

        Args:
            y_obs: Observed noisy image
            x_pred: Predicted clean image
            save_prefix: Optional prefix for saved plots

        Returns:
            results: Dict with all metrics
            is_consistent: Boolean indicating if reconstruction is consistent
        """
        # Compute residuals
        residuals = self.compute_residuals(y_obs, x_pred)

        # Chi-squared test
        chi_squared = self.compute_chi_squared(residuals)

        # Normality test
        normality = self.test_normality(residuals)

        # Moments
        moments = self.compute_moments(residuals)

        # Compile results
        results = {
            "chi_squared": chi_squared,
            "normality": normality,
            "moments": moments,
        }

        # Consistency criteria
        is_consistent = (
            0.8 < chi_squared < 1.2
            and abs(moments["mean"]) < 0.15  # χ² close to 1
            and 0.85 < moments["std"] < 1.15  # Mean close to 0
            and normality["is_normal"]  # Std close to 1  # Residuals are normal
        )

        results["is_consistent"] = is_consistent

        # Generate diagnostic plots if requested
        if self.save_dir and save_prefix:
            self.plot_diagnostics(residuals, results, save_prefix)

        return results, is_consistent

    def plot_diagnostics(self, residuals, results, prefix):
        """Generate diagnostic plots for paper."""
        residuals_np = residuals.flatten().cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Histogram with N(0,1) overlay
        ax = axes[0, 0]
        ax.hist(
            residuals_np,
            bins=50,
            density=True,
            alpha=0.7,
            edgecolor="black",
            label="Residuals",
        )
        x = np.linspace(-4, 4, 100)
        ax.plot(x, scipy_stats.norm.pdf(x, 0, 1), "r-", linewidth=2, label="N(0,1)")
        ax.set_xlabel("Normalized Residual")
        ax.set_ylabel("Density")
        ax.set_title(f'Residual Distribution (χ²={results["chi_squared"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Q-Q plot
        ax = axes[0, 1]
        scipy_stats.probplot(residuals_np, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot (Should be linear if normal)")
        ax.grid(True, alpha=0.3)

        # 3. Spatial residual map
        ax = axes[1, 0]
        residual_img = residuals[0].cpu().numpy()  # First image in batch
        if residual_img.ndim == 3:
            residual_img = residual_img.mean(axis=0)  # Average over channels
        im = ax.imshow(residual_img, cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_title("Spatial Residual Map")
        plt.colorbar(im, ax=ax, label="Normalized residual")

        # 4. Summary text
        ax = axes[1, 1]
        ax.axis("off")

        status_chi2 = "✓" if 0.8 < results["chi_squared"] < 1.2 else "✗"
        status_norm = "✓" if results["normality"]["is_normal"] else "✗"
        status_overall = (
            "✓ CONSISTENT" if results["is_consistent"] else "✗ INCONSISTENT"
        )

        summary_text = f"""
Statistical Validation Summary

Chi-squared Test:
χ² = {results['chi_squared']:.3f} {status_chi2}
Target: ~1.0 (range: 0.8-1.2)

Residual Moments:
Mean = {results['moments']['mean']:.4f} (target: 0)
Std = {results['moments']['std']:.4f} (target: 1)
Skew = {results['moments']['skewness']:.3f}
Kurt = {results['moments']['kurtosis']:.3f}

Normality Test (Shapiro-Wilk):
p-value = {results['normality']['p_value']:.4f} {status_norm}
Status: {'Normal' if results['normality']['is_normal'] else 'Non-normal'}

Overall: {status_overall}
"""

        ax.text(
            0.05,
            0.5,
            summary_text,
            fontsize=10,
            verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()
        save_path = self.save_dir / f"{prefix}_validation.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f" → Saved validation plot: {save_path.name}")


# Helper function for batch validation
def validate_batch(validator, noisy_batch, restored_batch):
    """
    Validate a batch of images and return aggregate statistics.

    Args:
        validator: StatisticalValidator instance
        noisy_batch: [B, C, H, W] noisy images
        restored_batch: [B, C, H, W] restored images

    Returns:
        batch_results: List of results for each image
        aggregate_stats: Overall statistics
    """
    batch_results = []

    for i in range(noisy_batch.shape[0]):
        noisy = noisy_batch[i : i + 1]
        restored = restored_batch[i : i + 1]

        results, is_consistent = validator.validate(noisy, restored)
        batch_results.append(results)

    # Aggregate statistics
    chi_squared_values = [r["chi_squared"] for r in batch_results]
    consistency_flags = [r["is_consistent"] for r in batch_results]

    aggregate_stats = {
        "mean_chi_squared": np.mean(chi_squared_values),
        "std_chi_squared": np.std(chi_squared_values),
        "consistency_rate": np.mean(consistency_flags),
        "num_samples": len(batch_results),
    }

    return batch_results, aggregate_stats
