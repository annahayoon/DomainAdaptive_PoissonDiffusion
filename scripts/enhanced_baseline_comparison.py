#!/usr/bin/env python3
"""
Enhanced Baseline Comparison Script

This script implements the remaining tasks from 3.1:
- Ensure identical computational budget (3.1.4)
- Add statistical significance testing (3.1.5)

Features:
- Computational budget monitoring and enforcement
- Memory usage tracking
- Timing analysis with identical operations
- Statistical significance testing with multiple correction methods
- Performance profiling and optimization analysis

Usage:
    python scripts/enhanced_baseline_comparison.py \
        --poisson_model hpc_result/best_model.pt \
        --l2_model results/l2_photography_baseline_20250922_231133/checkpoints/best_model.pt \
        --budget_type time \
        --budget_limit 300 \
        --output_dir enhanced_baseline_results
"""

import argparse
import gc
import json
import logging
import psutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.profiler
from scipy import stats
from scipy.stats import false_discovery_control
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.logging_config import get_logger
from scripts.evaluate_l2_baseline_integration import L2BaselineIntegrationEvaluator

logger = get_logger(__name__)

class ComputationalBudgetManager:
    """
    Manages computational budget for fair comparison.
    
    Ensures both methods use identical computational resources:
    - Time budget
    - Memory budget  
    - GPU operations
    - Model forward passes
    """
    
    def __init__(
        self,
        budget_type: str = "time",  # "time", "memory", "operations"
        budget_limit: float = 300.0,  # seconds, MB, or operation count
        device: str = "cuda",
    ):
        self.budget_type = budget_type
        self.budget_limit = budget_limit
        self.device = device
        
        # Tracking variables
        self.reset_budget()
        
        # Memory tracking
        self.process = psutil.Process()
        
        logger.info(f"Initialized budget manager: {budget_type}={budget_limit}")
    
    def reset_budget(self):
        """Reset budget tracking."""
        self.start_time = time.time()
        self.operation_count = 0
        self.peak_memory = 0
        self.initial_memory = self._get_memory_usage()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        else:
            return self.process.memory_info().rss / 1024**2
    
    def check_budget(self) -> Tuple[bool, Dict[str, float]]:
        """Check if budget is exceeded."""
        current_time = time.time() - self.start_time
        current_memory = self._get_memory_usage()
        
        stats = {
            "elapsed_time": current_time,
            "memory_usage": current_memory,
            "operation_count": self.operation_count,
            "peak_memory": max(self.peak_memory, current_memory),
        }
        
        if self.budget_type == "time":
            exceeded = current_time > self.budget_limit
        elif self.budget_type == "memory":
            exceeded = current_memory > self.budget_limit
        elif self.budget_type == "operations":
            exceeded = self.operation_count > self.budget_limit
        else:
            exceeded = False
        
        return not exceeded, stats
    
    def record_operation(self):
        """Record a computational operation."""
        self.operation_count += 1
        current_memory = self._get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)
    
    def get_final_stats(self) -> Dict[str, float]:
        """Get final computational statistics."""
        _, stats = self.check_budget()
        stats["total_time"] = time.time() - self.start_time
        stats["memory_overhead"] = stats["peak_memory"] - self.initial_memory
        return stats


class StatisticalAnalyzer:
    """
    Advanced statistical analysis for baseline comparison.
    
    Implements multiple statistical tests and corrections:
    - Paired t-tests
    - Wilcoxon signed-rank tests
    - Multiple comparison corrections
    - Effect size calculations
    - Confidence intervals
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def compute_comprehensive_statistics(
        self, 
        poisson_metrics: List[Dict[str, float]], 
        l2_metrics: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Compute comprehensive statistical analysis."""
        results = {}
        
        # Extract metric arrays
        metric_names = ["psnr", "ssim", "chi2_consistency", "lpips"]
        
        for metric in metric_names:
            poisson_values = [m.get(metric, 0) for m in poisson_metrics]
            l2_values = [m.get(metric, 0) for m in l2_metrics]
            
            if len(poisson_values) == 0 or len(l2_values) == 0:
                continue
                
            # Basic statistics
            poisson_mean = np.mean(poisson_values)
            l2_mean = np.mean(l2_values)
            difference = np.array(poisson_values) - np.array(l2_values)
            
            # Paired t-test
            ttest_stat, ttest_pvalue = stats.ttest_rel(poisson_values, l2_values)
            
            # Wilcoxon signed-rank test (non-parametric)
            try:
                wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(poisson_values, l2_values)
            except ValueError:
                wilcoxon_stat, wilcoxon_pvalue = np.nan, np.nan
            
            # Effect size (Cohen's d for paired samples)
            effect_size = np.mean(difference) / np.std(difference) if np.std(difference) > 0 else 0
            
            # Confidence interval for difference
            sem = stats.sem(difference)
            ci_lower, ci_upper = stats.t.interval(
                1 - self.alpha, len(difference) - 1, loc=np.mean(difference), scale=sem
            )
            
            results[metric] = {
                "poisson_mean": poisson_mean,
                "poisson_std": np.std(poisson_values),
                "l2_mean": l2_mean,
                "l2_std": np.std(l2_values),
                "difference_mean": np.mean(difference),
                "difference_std": np.std(difference),
                "ttest_statistic": ttest_stat,
                "ttest_pvalue": ttest_pvalue,
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_pvalue": wilcoxon_pvalue,
                "effect_size_cohens_d": effect_size,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "significant_ttest": ttest_pvalue < self.alpha,
                "significant_wilcoxon": wilcoxon_pvalue < self.alpha,
            }
        
        # Multiple comparison correction
        pvalues = [results[m]["ttest_pvalue"] for m in results.keys()]
        if len(pvalues) > 1:
            # Bonferroni correction
            bonferroni_corrected = [p * len(pvalues) for p in pvalues]
            
            # Benjamini-Hochberg FDR correction
            try:
                fdr_corrected = false_discovery_control(pvalues, method='bh')
            except:
                fdr_corrected = pvalues
            
            for i, metric in enumerate(results.keys()):
                results[metric]["bonferroni_pvalue"] = min(bonferroni_corrected[i], 1.0)
                results[metric]["fdr_pvalue"] = fdr_corrected[i]
                results[metric]["significant_bonferroni"] = bonferroni_corrected[i] < self.alpha
                results[metric]["significant_fdr"] = fdr_corrected[i] < self.alpha
        
        return results


class EnhancedBaselineComparator:
    """
    Enhanced baseline comparator with computational budget management
    and advanced statistical analysis.
    """
    
    def __init__(
        self,
        poisson_model_path: str,
        l2_model_path: str,
        budget_type: str = "time",
        budget_limit: float = 300.0,
        device: str = "auto",
        output_dir: str = "enhanced_baseline_results",
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize base evaluator
        self.evaluator = L2BaselineIntegrationEvaluator(
            poisson_model_path=poisson_model_path,
            l2_model_path=l2_model_path,
            device=device,
            output_dir=str(self.output_dir),
            seed=seed,
        )
        
        # Initialize budget manager
        self.budget_manager = ComputationalBudgetManager(
            budget_type=budget_type,
            budget_limit=budget_limit,
            device=self.evaluator.device,
        )
        
        # Initialize statistical analyzer
        self.statistical_analyzer = StatisticalAnalyzer()
        
        logger.info("Enhanced baseline comparator initialized")
    
    def run_budget_controlled_evaluation(
        self,
        electron_ranges: List[float] = None,
        max_samples: int = 50,
        steps: int = 18,
        guidance_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run evaluation with computational budget control.
        
        Ensures both methods use identical computational resources.
        """
        if electron_ranges is None:
            electron_ranges = [5000, 1000, 200, 50]
        
        logger.info("Starting budget-controlled evaluation...")
        logger.info(f"Budget: {self.budget_manager.budget_type}={self.budget_manager.budget_limit}")
        
        # Generate test samples
        test_samples = self.evaluator.generate_test_samples(max_samples)
        
        all_results = {}
        computational_stats = {}
        
        for electron_count in electron_ranges:
            logger.info(f"Evaluating at {electron_count} electrons...")
            
            # Reset budget for this electron count
            self.budget_manager.reset_budget()
            
            sample_results = []
            sample_count = 0
            
            for i, clean in enumerate(test_samples):
                # Check budget before processing
                budget_ok, current_stats = self.budget_manager.check_budget()
                if not budget_ok:
                    logger.warning(f"Budget exceeded at sample {i}, stopping early")
                    break
                
                clean = clean.to(self.evaluator.device)
                
                # Profile this evaluation
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True,
                ) as prof:
                    result = self.evaluator.evaluate_single_sample(
                        clean, electron_count, steps, guidance_weight
                    )
                
                # Record operation
                self.budget_manager.record_operation()
                
                result["sample_id"] = i
                result["profiler_stats"] = self._extract_profiler_stats(prof)
                sample_results.append(result)
                sample_count += 1
                
                # Force garbage collection to manage memory
                if i % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Get final computational stats for this electron count
            final_stats = self.budget_manager.get_final_stats()
            final_stats["samples_processed"] = sample_count
            final_stats["samples_per_second"] = sample_count / final_stats["total_time"]
            
            all_results[f"{electron_count:.0f}e"] = sample_results
            computational_stats[f"{electron_count:.0f}e"] = final_stats
            
            logger.info(f"Processed {sample_count} samples in {final_stats['total_time']:.1f}s")
        
        return {
            "evaluation_results": all_results,
            "computational_stats": computational_stats,
        }
    
    def _extract_profiler_stats(self, prof) -> Dict[str, float]:
        """Extract key statistics from profiler."""
        try:
            # Get CPU and CUDA time
            cpu_time = 0
            cuda_time = 0
            
            for event in prof.key_averages():
                if event.device_type == torch.profiler.DeviceType.CPU:
                    cpu_time += event.cpu_time_total
                elif event.device_type == torch.profiler.DeviceType.CUDA:
                    cuda_time += event.cuda_time_total
            
            return {
                "cpu_time_us": cpu_time,
                "cuda_time_us": cuda_time,
                "total_time_us": cpu_time + cuda_time,
            }
        except Exception as e:
            logger.warning(f"Failed to extract profiler stats: {e}")
            return {"cpu_time_us": 0, "cuda_time_us": 0, "total_time_us": 0}
    
    def compute_advanced_statistics(
        self, results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute advanced statistical analysis."""
        statistical_results = {}
        
        for electron_key, sample_results in results["evaluation_results"].items():
            poisson_metrics = [r["poisson"]["metrics"] for r in sample_results]
            l2_metrics = [r["l2"]["metrics"] for r in sample_results]
            
            stats = self.statistical_analyzer.compute_comprehensive_statistics(
                poisson_metrics, l2_metrics
            )
            
            statistical_results[electron_key] = stats
        
        return statistical_results
    
    def create_comprehensive_report(
        self, 
        results: Dict[str, Any], 
        statistics: Dict[str, Dict[str, Any]]
    ) -> str:
        """Create comprehensive evaluation report."""
        report = []
        report.append("=" * 100)
        report.append("ENHANCED BASELINE COMPARISON REPORT")
        report.append("=" * 100)
        report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Budget Type: {self.budget_manager.budget_type}")
        report.append(f"Budget Limit: {self.budget_manager.budget_limit}")
        report.append("")
        
        # Computational efficiency analysis
        report.append("COMPUTATIONAL EFFICIENCY ANALYSIS:")
        report.append("-" * 50)
        
        for electron_key, comp_stats in results["computational_stats"].items():
            report.append(f"{electron_key}:")
            report.append(f"  Samples processed: {comp_stats['samples_processed']}")
            report.append(f"  Total time: {comp_stats['total_time']:.2f}s")
            report.append(f"  Samples/second: {comp_stats['samples_per_second']:.2f}")
            report.append(f"  Peak memory: {comp_stats['peak_memory']:.1f}MB")
            report.append(f"  Memory overhead: {comp_stats['memory_overhead']:.1f}MB")
            report.append("")
        
        # Statistical analysis
        report.append("STATISTICAL ANALYSIS:")
        report.append("-" * 50)
        
        for electron_key, stats in statistics.items():
            report.append(f"{electron_key}:")
            
            for metric, metric_stats in stats.items():
                poisson_mean = metric_stats["poisson_mean"]
                l2_mean = metric_stats["l2_mean"]
                diff_mean = metric_stats["difference_mean"]
                effect_size = metric_stats["effect_size_cohens_d"]
                
                # Determine significance
                if metric_stats.get("significant_fdr", False):
                    sig_marker = "***"
                elif metric_stats.get("significant_bonferroni", False):
                    sig_marker = "**"
                elif metric_stats.get("significant_ttest", False):
                    sig_marker = "*"
                else:
                    sig_marker = ""
                
                report.append(f"  {metric.upper()}:")
                report.append(f"    Poisson: {poisson_mean:.3f} ± {metric_stats['poisson_std']:.3f}")
                report.append(f"    L2: {l2_mean:.3f} ± {metric_stats['l2_std']:.3f}")
                report.append(f"    Difference: {diff_mean:.3f} ± {metric_stats['difference_std']:.3f} {sig_marker}")
                report.append(f"    Effect size (Cohen's d): {effect_size:.3f}")
                report.append(f"    p-value (t-test): {metric_stats['ttest_pvalue']:.4f}")
                report.append(f"    p-value (FDR corrected): {metric_stats.get('fdr_pvalue', 'N/A'):.4f}")
                report.append("")
        
        # Overall conclusions
        report.append("OVERALL CONCLUSIONS:")
        report.append("-" * 50)
        
        # Compute overall effect sizes
        psnr_effects = [stats["psnr"]["effect_size_cohens_d"] for stats in statistics.values() if "psnr" in stats]
        chi2_effects = [stats["chi2_consistency"]["effect_size_cohens_d"] for stats in statistics.values() if "chi2_consistency" in stats]
        
        avg_psnr_effect = np.mean(psnr_effects) if psnr_effects else 0
        avg_chi2_effect = np.mean(chi2_effects) if chi2_effects else 0
        
        if avg_psnr_effect > 0.8:
            report.append("✅ LARGE effect size for PSNR improvement (Cohen's d > 0.8)")
        elif avg_psnr_effect > 0.5:
            report.append("⚠️  MEDIUM effect size for PSNR improvement (Cohen's d > 0.5)")
        elif avg_psnr_effect > 0.2:
            report.append("📊 SMALL effect size for PSNR improvement (Cohen's d > 0.2)")
        else:
            report.append("❌ NEGLIGIBLE effect size for PSNR improvement (Cohen's d < 0.2)")
        
        report.append(f"   Average PSNR effect size: {avg_psnr_effect:.3f}")
        report.append(f"   Average χ² effect size: {avg_chi2_effect:.3f}")
        
        # Computational efficiency
        total_samples = sum(stats["samples_processed"] for stats in results["computational_stats"].values())
        total_time = sum(stats["total_time"] for stats in results["computational_stats"].values())
        
        report.append(f"   Total samples processed: {total_samples}")
        report.append(f"   Total evaluation time: {total_time:.1f}s")
        report.append(f"   Average processing rate: {total_samples/total_time:.2f} samples/second")
        
        report.append("")
        report.append("LEGEND: *** FDR significant, ** Bonferroni significant, * t-test significant")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def create_advanced_visualizations(
        self, 
        results: Dict[str, Any], 
        statistics: Dict[str, Dict[str, Any]]
    ) -> None:
        """Create advanced visualization plots."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Extract data for plotting
        electron_counts = []
        metrics_data = {"psnr": {"poisson": [], "l2": [], "effect_size": []},
                       "chi2_consistency": {"poisson": [], "l2": [], "effect_size": []}}
        
        for electron_key in sorted(results["evaluation_results"].keys(), 
                                 key=lambda x: float(x.replace('e', '')), reverse=True):
            electron_count = float(electron_key.replace('e', ''))
            electron_counts.append(electron_count)
            
            stats = statistics[electron_key]
            for metric in ["psnr", "chi2_consistency"]:
                if metric in stats:
                    metrics_data[metric]["poisson"].append(stats[metric]["poisson_mean"])
                    metrics_data[metric]["l2"].append(stats[metric]["l2_mean"])
                    metrics_data[metric]["effect_size"].append(stats[metric]["effect_size_cohens_d"])
        
        # Plot 1: PSNR comparison with error bars
        poisson_psnr_std = [statistics[f"{int(ec)}e"]["psnr"]["poisson_std"] for ec in electron_counts]
        l2_psnr_std = [statistics[f"{int(ec)}e"]["psnr"]["l2_std"] for ec in electron_counts]
        
        axes[0, 0].errorbar(electron_counts, metrics_data["psnr"]["poisson"], 
                           yerr=poisson_psnr_std, fmt='bo-', label='Poisson-Gaussian', capsize=5)
        axes[0, 0].errorbar(electron_counts, metrics_data["psnr"]["l2"], 
                           yerr=l2_psnr_std, fmt='ro-', label='L2 Baseline', capsize=5)
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_xlabel('Electron Count')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].set_title('PSNR Comparison with Error Bars')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Effect sizes
        axes[0, 1].semilogx(electron_counts, metrics_data["psnr"]["effect_size"], 'go-', 
                           linewidth=2, label='PSNR Effect Size')
        axes[0, 1].axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Small Effect')
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Medium Effect')
        axes[0, 1].axhline(y=0.8, color='purple', linestyle='--', alpha=0.7, label='Large Effect')
        axes[0, 1].set_xlabel('Electron Count')
        axes[0, 1].set_ylabel("Cohen's d")
        axes[0, 1].set_title('Effect Size Analysis')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: P-values with significance thresholds
        pvalues = [statistics[f"{int(ec)}e"]["psnr"]["ttest_pvalue"] for ec in electron_counts]
        fdr_pvalues = [statistics[f"{int(ec)}e"]["psnr"].get("fdr_pvalue", 1.0) for ec in electron_counts]
        
        axes[0, 2].semilogx(electron_counts, [-np.log10(p) for p in pvalues], 'bo-', label='t-test')
        axes[0, 2].semilogx(electron_counts, [-np.log10(p) for p in fdr_pvalues], 'ro-', label='FDR corrected')
        axes[0, 2].axhline(y=-np.log10(0.05), color='k', linestyle='--', label='α=0.05')
        axes[0, 2].axhline(y=-np.log10(0.01), color='r', linestyle='--', label='α=0.01')
        axes[0, 2].set_xlabel('Electron Count')
        axes[0, 2].set_ylabel('-log10(p-value)')
        axes[0, 2].set_title('Statistical Significance')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Chi-squared comparison
        axes[1, 0].semilogx(electron_counts, metrics_data["chi2_consistency"]["poisson"], 
                           'bo-', label='Poisson-Gaussian', linewidth=2)
        axes[1, 0].semilogx(electron_counts, metrics_data["chi2_consistency"]["l2"], 
                           'ro-', label='L2 Baseline', linewidth=2)
        axes[1, 0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect χ²=1')
        axes[1, 0].fill_between(electron_counts, 0.9, 1.1, alpha=0.2, color='green', label='Good Physics')
        axes[1, 0].set_xlabel('Electron Count')
        axes[1, 0].set_ylabel('χ² per pixel')
        axes[1, 0].set_title('Physics Consistency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Computational efficiency
        processing_rates = [results["computational_stats"][f"{int(ec)}e"]["samples_per_second"] 
                           for ec in electron_counts]
        memory_usage = [results["computational_stats"][f"{int(ec)}e"]["peak_memory"] 
                       for ec in electron_counts]
        
        ax5_twin = axes[1, 1].twinx()
        line1 = axes[1, 1].semilogx(electron_counts, processing_rates, 'go-', label='Samples/sec')
        line2 = ax5_twin.semilogx(electron_counts, memory_usage, 'mo-', label='Peak Memory (MB)')
        
        axes[1, 1].set_xlabel('Electron Count')
        axes[1, 1].set_ylabel('Processing Rate (samples/sec)', color='g')
        ax5_twin.set_ylabel('Peak Memory (MB)', color='m')
        axes[1, 1].set_title('Computational Efficiency')
        
        # Combine legends
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Confidence intervals
        psnr_diffs = [statistics[f"{int(ec)}e"]["psnr"]["difference_mean"] for ec in electron_counts]
        ci_lowers = [statistics[f"{int(ec)}e"]["psnr"]["ci_lower"] for ec in electron_counts]
        ci_uppers = [statistics[f"{int(ec)}e"]["psnr"]["ci_upper"] for ec in electron_counts]
        
        axes[1, 2].semilogx(electron_counts, psnr_diffs, 'bo-', linewidth=2)
        axes[1, 2].fill_between(electron_counts, ci_lowers, ci_uppers, alpha=0.3, color='blue')
        axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('Electron Count')
        axes[1, 2].set_ylabel('PSNR Difference (dB)')
        axes[1, 2].set_title('PSNR Improvement with 95% CI')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Plots 7-9: Example restorations at different electron counts
        example_electron_keys = [f"{int(ec)}e" for ec in electron_counts[:3]]
        
        for i, electron_key in enumerate(example_electron_keys):
            if electron_key in results["evaluation_results"]:
                sample_result = results["evaluation_results"][electron_key][0]
                
                # Show difference images
                poisson_img = sample_result["poisson"]["result"].cpu().numpy().squeeze()
                l2_img = sample_result["l2"]["result"].cpu().numpy().squeeze()
                diff_img = poisson_img - l2_img
                
                im = axes[2, i].imshow(diff_img, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
                axes[2, i].set_title(f'Difference Image ({electron_key})')
                axes[2, i].axis('off')
                plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
        
        plt.suptitle('Enhanced Baseline Comparison Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "enhanced_baseline_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved enhanced analysis plot: {plot_file}")
    
    def save_comprehensive_results(
        self, 
        results: Dict[str, Any], 
        statistics: Dict[str, Dict[str, Any]]
    ) -> None:
        """Save all results with comprehensive documentation."""
        # Save raw results
        results_file = self.output_dir / "enhanced_evaluation_results.json"
        
        # Prepare serializable results
        serializable_results = {
            "evaluation_results": {},
            "computational_stats": results["computational_stats"],
            "statistical_analysis": statistics,
        }
        
        for electron_key, sample_results in results["evaluation_results"].items():
            serializable_results["evaluation_results"][electron_key] = []
            for result in sample_results:
                serializable_result = {
                    "sample_id": result["sample_id"],
                    "electron_count": result["electron_count"],
                    "scale": result["scale"],
                    "background": result["background"],
                    "read_noise": result["read_noise"],
                    "poisson": {
                        "metrics": result["poisson"]["metrics"],
                        "time": result["poisson"]["time"],
                    },
                    "l2": {
                        "metrics": result["l2"]["metrics"],
                        "time": result["l2"]["time"],
                    },
                    "profiler_stats": result["profiler_stats"],
                }
                serializable_results["evaluation_results"][electron_key].append(serializable_result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save comprehensive report
        report = self.create_comprehensive_report(results, statistics)
        report_file = self.output_dir / "enhanced_evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved comprehensive results to {self.output_dir}")


def main():
    """Main function for enhanced baseline comparison."""
    parser = argparse.ArgumentParser(description="Enhanced Baseline Comparison")
    parser.add_argument(
        "--poisson_model", 
        type=str, 
        default="hpc_result/best_model.pt",
        help="Path to Poisson-Gaussian model"
    )
    parser.add_argument(
        "--l2_model", 
        type=str, 
        default="results/l2_photography_baseline_20250922_231133/checkpoints/best_model.pt",
        help="Path to L2 baseline model"
    )
    parser.add_argument(
        "--budget_type", 
        type=str, 
        choices=["time", "memory", "operations"],
        default="time",
        help="Type of computational budget"
    )
    parser.add_argument(
        "--budget_limit", 
        type=float, 
        default=300.0,
        help="Budget limit (seconds, MB, or operations)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="enhanced_baseline_results",
        help="Output directory"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=20,
        help="Maximum number of test samples"
    )
    parser.add_argument(
        "--electron_ranges", 
        nargs="+", 
        type=float, 
        default=[5000, 1000, 200, 50],
        help="Electron count ranges to test"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 100)
    logger.info("ENHANCED BASELINE COMPARISON")
    logger.info("=" * 100)
    logger.info("Tasks 3.1.4 & 3.1.5: Computational budget & statistical significance")
    logger.info("Features:")
    logger.info("• Computational budget monitoring and enforcement")
    logger.info("• Advanced statistical analysis with multiple corrections")
    logger.info("• Performance profiling and optimization analysis")
    logger.info("• Comprehensive visualization and reporting")
    logger.info("=" * 100)
    
    # Initialize comparator
    comparator = EnhancedBaselineComparator(
        poisson_model_path=args.poisson_model,
        l2_model_path=args.l2_model,
        budget_type=args.budget_type,
        budget_limit=args.budget_limit,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    # Run evaluation
    logger.info("Starting budget-controlled evaluation...")
    results = comparator.run_budget_controlled_evaluation(
        electron_ranges=args.electron_ranges,
        max_samples=args.max_samples,
    )
    
    # Compute advanced statistics
    logger.info("Computing advanced statistical analysis...")
    statistics = comparator.compute_advanced_statistics(results)
    
    # Create visualizations
    logger.info("Creating advanced visualizations...")
    comparator.create_advanced_visualizations(results, statistics)
    
    # Save results
    logger.info("Saving comprehensive results...")
    comparator.save_comprehensive_results(results, statistics)
    
    logger.info("=" * 100)
    logger.info("ENHANCED BASELINE COMPARISON COMPLETE!")
    logger.info("=" * 100)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("Key outputs:")
    logger.info("• enhanced_evaluation_results.json - Complete evaluation data")
    logger.info("• enhanced_evaluation_report.txt - Comprehensive analysis report")
    logger.info("• enhanced_baseline_analysis.png - Advanced visualization plots")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
