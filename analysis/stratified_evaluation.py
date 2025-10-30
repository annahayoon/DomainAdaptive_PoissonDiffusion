"""
Stratified evaluation module for analyzing results by signal level.

This is THE CORE of the paper's contribution:
- Compute metrics stratified by ground truth ADC level
- Show that heteroscedastic guidance provides largest gains at low signal levels
- Validate statistical significance with Holm-corrected t-tests

Paper's central claim: "Expected: [TBD] dB PSNR gains in very low-light regions
(ADC < 100) where heteroscedastic weighting matters most."

This module makes that claim quantitative and testable.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


class StratifiedEvaluator:
    """
    Compute metrics stratified by ground truth signal level.

    This answers the key research question:
    "Are improvements largest where signal-dependent noise dominates?"
    """

    def __init__(self, domain_ranges: Dict[str, float]):
        """
        Initialize stratified evaluator.

        Args:
            domain_ranges: Dict with 'min' (black level ADU) and 'max' (white level ADU)
                          e.g., {'min': 512, 'max': 16383} for Sony a7S II
        """
        self.domain_ranges = domain_ranges

        # Define ADC bins from paper proposal
        # These correspond to different noise regimes
        self.adc_bins = {
            "very_low": (0, 100),  # Extreme low-light: dominated by shot noise
            "low": (100, 500),  # Low-light: shot noise significant
            "medium": (500, 2000),  # Moderate: transition region
            "high": (2000, float("inf")),  # Well-lit: shot noise negligible
        }

        logger.info(f"StratifiedEvaluator initialized with ADC bins:")
        for name, (min_adc, max_adc) in self.adc_bins.items():
            logger.info(f"  {name:12s}: [{min_adc:5.0f}, {max_adc:5.0f})")

    def denormalize_to_adc(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert model space [-1, 1] to physical ADC units.

        The model is trained on [-1, 1] normalized data.
        To stratify by signal level, we convert to actual ADC values
        from the preprocessing pipeline.

        Args:
            tensor: Tensor in [-1, 1] range

        Returns:
            Tensor in ADC units (physical sensor values)
        """
        # [-1, 1] → [0, 1]
        tensor_01 = (tensor + 1.0) / 2.0

        # [0, 1] → [black_level, white_level]
        adc_range = self.domain_ranges["max"] - self.domain_ranges["min"]
        adc = tensor_01 * adc_range + self.domain_ranges["min"]

        return adc

    def compute_stratified_metrics(
        self, clean: torch.Tensor, enhanced: torch.Tensor, method_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute PSNR stratified by clean image ADC level.

        This is THE KEY EXPERIMENT: Does heteroscedastic guidance help more
        in dark regions (low ADC) than bright regions (high ADC)?

        Args:
            clean: Ground truth image, shape (C, H, W), range [-1, 1]
            enhanced: Enhanced/reconstructed image, same shape and range
            method_name: Name of method (for logging)

        Returns:
            Dict mapping bin name → metrics dict with:
                - psnr: Peak signal-to-noise ratio in this bin
                - mean_abs_error: Mean absolute pixel difference
                - pixel_count: Number of pixels in this bin
                - pixel_fraction: Fraction of image in this bin
                - adc_range: String describing ADC range
                - mean_signal: Average ADC in this bin
        """
        # Convert to ADC for stratification
        clean_adc = self.denormalize_to_adc(clean)

        results = {}
        for bin_name, (min_adc, max_adc) in self.adc_bins.items():
            # Create spatial mask for pixels in this ADC range
            # Use mean across channels for multi-channel images
            if clean_adc.dim() == 3:
                clean_adc_mono = clean_adc.mean(dim=0)  # Average channels
            else:
                clean_adc_mono = clean_adc

            mask = (clean_adc_mono >= min_adc) & (clean_adc_mono < max_adc)

            if not mask.any():
                # No pixels in this range
                results[bin_name] = {
                    "psnr": float("nan"),
                    "mean_abs_error": float("nan"),
                    "pixel_count": 0,
                    "pixel_fraction": 0.0,
                    "adc_range": f"[{min_adc}, {max_adc})",
                    "mean_signal": float("nan"),
                }
                continue

            # Extract pixels in this range from ALL channels
            if clean.dim() == 3:
                # (C, H, W) → broadcast mask to channels
                mask_expanded = mask.unsqueeze(0).expand(clean.shape[0], -1, -1)
                clean_masked = clean[mask_expanded].reshape(clean.shape[0], -1)
                enhanced_masked = enhanced[mask_expanded].reshape(enhanced.shape[0], -1)
            else:
                clean_masked = clean[mask]
                enhanced_masked = enhanced[mask]

            # Compute MSE and PSNR for this subset
            mse = ((clean_masked - enhanced_masked) ** 2).mean()

            # Avoid log(0)
            if mse < 1e-10:
                psnr = float("inf")
            else:
                # PSNR for range [-1, 1]: max_val = 2
                psnr = 20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(mse)
                psnr = psnr.item()

            # Mean absolute error as secondary metric
            mae = (clean_masked - enhanced_masked).abs().mean()

            results[bin_name] = {
                "psnr": psnr,
                "mean_abs_error": mae.item(),
                "pixel_count": mask.sum().item(),
                "pixel_fraction": mask.float().mean().item(),
                "adc_range": f"[{min_adc}, {max_adc})",
                "mean_signal": clean_adc_mono[mask].mean().item(),
            }

        return results

    def compare_methods_stratified(
        self, clean: torch.Tensor, method_results: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare multiple methods with stratification.

        This generates data for your paper's comparison table.

        Args:
            clean: Ground truth image
            method_results: Dict mapping method_name → enhanced_image

        Returns:
            Nested dict: {method: {bin: {metric: value}}}
        """
        comparison = {}
        for method_name, enhanced in method_results.items():
            logger.debug(f"Computing stratified metrics for {method_name}")
            comparison[method_name] = self.compute_stratified_metrics(
                clean, enhanced, method_name
            )
        return comparison

    def compute_improvement_matrix(
        self,
        baseline_metrics: Dict[str, Dict[str, float]],
        proposed_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Compute PSNR gain for each bin.

        This creates the comparison matrix for your paper's Table 1!

        Args:
            baseline_metrics: Metrics from baseline method
            proposed_metrics: Metrics from your heteroscedastic method

        Returns:
            Dict mapping bin_name → PSNR improvement (dB)
        """
        improvements = {}
        for bin_name in self.adc_bins.keys():
            if bin_name not in baseline_metrics or bin_name not in proposed_metrics:
                improvements[bin_name] = float("nan")
                continue

            baseline_psnr = baseline_metrics[bin_name].get("psnr", float("nan"))
            proposed_psnr = proposed_metrics[bin_name].get("psnr", float("nan"))

            if np.isnan(baseline_psnr) or np.isnan(proposed_psnr):
                improvements[bin_name] = float("nan")
            else:
                improvements[bin_name] = proposed_psnr - baseline_psnr

        return improvements

    def test_statistical_significance(
        self,
        all_results: List[Dict],
        baseline_method: str,
        proposed_method: str,
        alpha: float = 0.05,
    ) -> Dict[str, Dict[str, float]]:
        """
        Test if improvements are statistically significant.

        Uses paired t-test with Holm-Bonferroni correction for multiple comparisons.
        This generates p-values for your paper's statistical claims.

        Args:
            all_results: List of per-tile results with stratified metrics
            baseline_method: Name of baseline method (e.g., 'gaussian_x0')
            proposed_method: Name of proposed method (e.g., 'pg_x0')
            alpha: Significance level (default 0.05)

        Returns:
            Dict mapping bin_name → {
                'mean_improvement': float,
                'std_improvement': float,
                't_statistic': float,
                'p_value': float,
                'p_value_corrected': float (Holm-corrected),
                'significant': bool,
                'n_samples': int
            }
        """
        # Collect per-bin metrics across all tiles
        bin_data = {
            bin_name: {"baseline": [], "proposed": []}
            for bin_name in self.adc_bins.keys()
        }

        for result in all_results:
            if "stratified_metrics" not in result:
                continue

            stratified = result["stratified_metrics"]

            if baseline_method not in stratified or proposed_method not in stratified:
                continue

            for bin_name in self.adc_bins.keys():
                baseline_entry = stratified[baseline_method].get(bin_name, {})
                proposed_entry = stratified[proposed_method].get(bin_name, {})

                baseline_psnr = baseline_entry.get("psnr", float("nan"))
                proposed_psnr = proposed_entry.get("psnr", float("nan"))

                if not (np.isnan(baseline_psnr) or np.isnan(proposed_psnr)):
                    bin_data[bin_name]["baseline"].append(baseline_psnr)
                    bin_data[bin_name]["proposed"].append(proposed_psnr)

        # Perform paired t-tests
        test_results = {}
        p_values = []
        bin_names_for_correction = []

        for bin_name, data in bin_data.items():
            n_samples = len(data["baseline"])

            if n_samples < 2:
                test_results[bin_name] = {
                    "mean_improvement": float("nan"),
                    "std_improvement": float("nan"),
                    "t_statistic": float("nan"),
                    "p_value": float("nan"),
                    "p_value_corrected": float("nan"),
                    "significant": False,
                    "n_samples": n_samples,
                }
                continue

            baseline = np.array(data["baseline"])
            proposed = np.array(data["proposed"])

            # Paired t-test: H0 = no difference
            # H1 = proposed > baseline (one-tailed)
            t_stat, p_value_two_tailed = ttest_rel(proposed, baseline)
            p_value = (
                p_value_two_tailed / 2 if t_stat > 0 else 1 - p_value_two_tailed / 2
            )

            improvements = proposed - baseline

            test_results[bin_name] = {
                "mean_improvement": float(np.mean(improvements)),
                "std_improvement": float(np.std(improvements)),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "p_value_corrected": float("nan"),  # Will update after Holm correction
                "significant": False,  # Will update after correction
                "n_samples": n_samples,
            }

            p_values.append(p_value)
            bin_names_for_correction.append(bin_name)

        # Holm-Bonferroni correction for multiple comparisons
        if p_values:
            reject, p_corrected, _, _ = multipletests(
                p_values, alpha=alpha, method="holm"
            )

            for i, bin_name in enumerate(bin_names_for_correction):
                test_results[bin_name]["p_value_corrected"] = float(p_corrected[i])
                test_results[bin_name]["significant"] = bool(reject[i])

        return test_results


def format_stratified_results_table(
    comparison: Dict[str, Dict[str, Dict[str, float]]], method_names: List[str] = None
) -> str:
    """
    Format stratified results as ASCII table for logging/reporting.

    This is how your Table 1 will look!
    """
    if method_names is None:
        method_names = list(comparison.keys())

    bin_order = ["very_low", "low", "medium", "high"]

    # Build table
    lines = []
    lines.append("=" * 100)
    lines.append(
        f"{'Method':<30} | {'Very Low':<20} | {'Low':<20} | {'Medium':<20} | {'High':<20}"
    )
    lines.append(
        f"{'':30} | {'(ADC<100)':<20} | {'(100-500)':<20} | {'(500-2000)':<20} | {'(>2000)':<20}"
    )
    lines.append("-" * 100)

    for method in method_names:
        if method not in comparison:
            continue

        psnr_values = []
        for bin_name in bin_order:
            if bin_name in comparison[method]:
                psnr = comparison[method][bin_name]["psnr"]
                pixel_frac = comparison[method][bin_name]["pixel_fraction"]

                if not np.isnan(psnr):
                    psnr_values.append(f"{psnr:.2f} dB ({pixel_frac:.1%})")
                else:
                    psnr_values.append("N/A")
            else:
                psnr_values.append("N/A")

        lines.append(
            f"{method:<30} | {psnr_values[0]:<20} | {psnr_values[1]:<20} | {psnr_values[2]:<20} | {psnr_values[3]:<20}"
        )

    lines.append("=" * 100)
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    evaluator = StratifiedEvaluator({"min": 512, "max": 16383})

    # Example: Generate some synthetic data
    clean = torch.rand(3, 256, 256) * 2 - 1  # [-1, 1]
    enhanced = clean + torch.randn_like(clean) * 0.1

    metrics = evaluator.compute_stratified_metrics(clean, enhanced, "test_method")

    print("Example stratified metrics:")
    for bin_name, metrics_dict in metrics.items():
        print(
            f"  {bin_name}: PSNR={metrics_dict['psnr']:.2f} dB, "
            f"pixels={metrics_dict['pixel_fraction']:.1%}"
        )
