"""Stratified evaluation module for analyzing results by signal level."""

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
    """Compute metrics stratified by ground truth signal level."""

    def __init__(self, sensor_ranges: Dict[str, float]):
        """Initialize stratified evaluator.

        Args:
            sensor_ranges: Dict with 'min' (black level ADU) and 'max' (white level ADU)
        """
        self.sensor_ranges = sensor_ranges

        self.adc_bins = {
            "very_low": (0, 100),
            "low": (100, 500),
            "medium": (500, 2000),
            "high": (2000, float("inf")),
        }

    def denormalize_to_adc(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert model space [-1, 1] to physical ADC units."""
        # [-1, 1] → [0, 1]
        tensor_01 = (tensor + 1.0) / 2.0

        # [0, 1] → [black_level, white_level]
        adc_range = self.sensor_ranges["max"] - self.sensor_ranges["min"]
        adc = tensor_01 * adc_range + self.sensor_ranges["min"]

        return adc

    def compute_stratified_metrics(
        self, clean: torch.Tensor, enhanced: torch.Tensor, method_name: str
    ) -> Dict[str, Dict[str, float]]:
        """Compute PSNR stratified by clean image ADC level."""
        # Convert to ADC for stratification
        clean_adc = self.denormalize_to_adc(clean)

        results = {}
        for bin_name, (min_adc, max_adc) in self.adc_bins.items():
            if clean_adc.dim() == 3:
                clean_adc_mono = clean_adc.mean(dim=0)  # Average channels
            else:
                clean_adc_mono = clean_adc

            mask = (clean_adc_mono >= min_adc) & (clean_adc_mono < max_adc)

            if not mask.any():
                results[bin_name] = {
                    "psnr": float("nan"),
                    "mean_abs_error": float("nan"),
                    "pixel_count": 0,
                    "pixel_fraction": 0.0,
                    "adc_range": f"[{min_adc}, {max_adc})",
                    "mean_signal": float("nan"),
                }
                continue

            if clean.dim() == 3:
                # (C, H, W) → broadcast mask to channels
                mask_expanded = mask.unsqueeze(0).expand(clean.shape[0], -1, -1)
                clean_masked = clean[mask_expanded].reshape(clean.shape[0], -1)
                enhanced_masked = enhanced[mask_expanded].reshape(enhanced.shape[0], -1)
            else:
                clean_masked = clean[mask]
                enhanced_masked = enhanced[mask]

            mse = ((clean_masked - enhanced_masked) ** 2).mean()

            if mse < 1e-10:
                psnr = float("inf")
            else:
                psnr = 20 * torch.log10(torch.tensor(2.0)) - 10 * torch.log10(mse)
                psnr = psnr.item()

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
        """Compare multiple methods with stratification."""
        comparison = {}
        for method_name, enhanced in method_results.items():
            comparison[method_name] = self.compute_stratified_metrics(
                clean, enhanced, method_name
            )
        return comparison

    def compute_improvement_matrix(
        self,
        baseline_metrics: Dict[str, Dict[str, float]],
        proposed_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Compute PSNR gain for each bin."""
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
        return_per_scene_improvements: bool = False,
    ) -> Dict:
        """Test if improvements are statistically significant.

        Uses paired t-test with Holm-Bonferroni correction for multiple comparisons.

        Args:
            all_results: List of result dictionaries with "stratified_metrics" key
            baseline_method: Name of baseline method
            proposed_method: Name of proposed method
            alpha: Significance level (default: 0.05)
            return_per_scene_improvements: If True, also return per-scene improvements

        Returns:
            Dict with "statistical_significance" key containing test results.
            If return_per_scene_improvements=True, also includes "comparison_by_scene" key.
        """
        bin_data = {
            bin_name: {"baseline": [], "proposed": []}
            for bin_name in self.adc_bins.keys()
        }

        # Also collect per-scene improvements if requested
        per_scene_improvements = [] if return_per_scene_improvements else None

        for result in all_results:
            if "stratified_metrics" not in result:
                continue

            stratified = result["stratified_metrics"]

            if baseline_method not in stratified or proposed_method not in stratified:
                continue

            scene_improvements = {} if return_per_scene_improvements else None
            for bin_name in self.adc_bins.keys():
                baseline_entry = stratified[baseline_method].get(bin_name, {})
                proposed_entry = stratified[proposed_method].get(bin_name, {})

                baseline_psnr = baseline_entry.get("psnr", float("nan"))
                proposed_psnr = proposed_entry.get("psnr", float("nan"))

                if not (np.isnan(baseline_psnr) or np.isnan(proposed_psnr)):
                    bin_data[bin_name]["baseline"].append(baseline_psnr)
                    bin_data[bin_name]["proposed"].append(proposed_psnr)
                    if return_per_scene_improvements:
                        scene_improvements[bin_name] = proposed_psnr - baseline_psnr

            if return_per_scene_improvements and scene_improvements:
                per_scene_improvements.append(
                    {
                        "scene_id": result.get("scene_id", "unknown"),
                        "improvements": scene_improvements,
                    }
                )

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
                "p_value_corrected": float("nan"),
                "significant": False,
                "n_samples": n_samples,
            }

            p_values.append(p_value)
            bin_names_for_correction.append(bin_name)

        if p_values:
            reject, p_corrected, _, _ = multipletests(
                p_values, alpha=alpha, method="holm"
            )

            for i, bin_name in enumerate(bin_names_for_correction):
                test_results[bin_name]["p_value_corrected"] = float(p_corrected[i])
                test_results[bin_name]["significant"] = bool(reject[i])

        result_dict = {"statistical_significance": test_results}
        if return_per_scene_improvements:
            result_dict["comparison_by_scene"] = per_scene_improvements
        return result_dict


def format_significance_marker(p_corrected: float) -> str:
    """Format significance marker based on p-value."""
    if np.isnan(p_corrected):
        return ""
    elif p_corrected < 0.001:
        return "***"
    elif p_corrected < 0.01:
        return "**"
    elif p_corrected < 0.05:
        return "*"
    else:
        return ""


def format_stratified_results_table(
    comparison: Dict[str, Dict[str, Dict[str, float]]], method_names: List[str] = None
) -> str:
    """Format stratified results as ASCII table for logging/reporting."""
    if method_names is None:
        method_names = list(comparison.keys())

    bin_order = ["very_low", "low", "medium", "high"]
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
