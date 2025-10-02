#!/usr/bin/env python3
"""
L2 Baseline Integration Evaluation Script

This script implements task 3.1 from evaluation_enhancement_todos.md:
Complete L2 baseline integration for fair, rigorous ablation study.

Key features:
- Identical sampling pipeline for both Poisson-Gaussian and L2 guidance
- Deterministic evaluation with fixed seeds
- Identical computational budget
- Statistical significance testing
- Fair conditioning strategy differences

Usage:
    python scripts/evaluate_l2_baseline_integration.py \
        --poisson_model hpc_result/best_model.pt \
        --l2_model results/l2_photography_baseline_20250922_231133/checkpoints/best_model.pt \
        --output_dir l2_baseline_integration_results \
        --num_samples 10 \
        --electron_ranges 5000 1000 200 50
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.baselines import UnifiedDiffusionBaseline
from core.calibration import SensorCalibration
from models.edm_wrapper import DomainEncoder
from core.guidance_config import GuidanceConfig
from core.guidance_factory import create_guidance
from core.l2_guidance import L2Guidance
from core.logging_config import get_logger
from core.metrics import EvaluationSuite
from core.poisson_guidance import PoissonGuidance
from core.transforms import ImageMetadata
from models.edm_wrapper import load_pretrained_edm
from models.sampler import EDMPosteriorSampler
from scripts.generate_synthetic_data import SyntheticConfig, SyntheticDataGenerator

logger = get_logger(__name__)

class L2BaselineIntegrationEvaluator:
    """
    Comprehensive L2 baseline integration evaluator.
    
    Implements fair comparison between Poisson-Gaussian and L2 guidance
    with identical infrastructure except guidance computation.
    """
    
    def __init__(
        self,
        poisson_model_path: str,
        l2_model_path: str,
        device: str = "auto",
        output_dir: str = "l2_baseline_integration_results",
        seed: int = 42,
    ):
        """Initialize evaluator with both model paths."""
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Set deterministic behavior
        self._set_deterministic_mode(seed)
        
        # Load models
        self.poisson_model = self._load_model(poisson_model_path, "Poisson-Gaussian")
        self.l2_model = self._load_model(l2_model_path, "L2")
        
        # Initialize evaluation suite
        self.evaluation_suite = EvaluationSuite(device=self.device)
        
        # Initialize synthetic data generator
        synthetic_config = SyntheticConfig(
            output_dir=str(self.output_dir / "temp_synthetic"),
            num_images=1,
            image_size=128,
            save_plots=False,
            save_metadata=False,
        )
        self.data_generator = SyntheticDataGenerator(synthetic_config)
        
        logger.info("L2BaselineIntegrationEvaluator initialized successfully")
        logger.info(f"Device: {self.device}")
        logger.info(f"Seed: {seed} (deterministic mode)")
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _set_deterministic_mode(self, seed: int) -> None:
        """Set deterministic behavior for reproducible evaluation."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set deterministic mode with seed {seed}")
    
    def _load_model(self, model_path: str, model_type: str):
        """Load model with error handling."""
        try:
            logger.info(f"Loading {model_type} model from {model_path}")
            model = load_pretrained_edm(model_path, device=self.device)
            model.eval()
            logger.info(f"Successfully loaded {model_type} model")
            return model
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            raise
    
    def create_identical_samplers(
        self,
        scale: float,
        background: float,
        read_noise: float,
        guidance_weight: float = 1.0,
    ) -> Tuple[EDMPosteriorSampler, EDMPosteriorSampler]:
        """
        Create identical samplers for both guidance types.
        
        This ensures fair comparison with identical sampling pipeline.
        """
        # Create guidance configurations (identical except type)
        guidance_config = GuidanceConfig(
            kappa=guidance_weight,
            gamma_schedule="sigma2",  # Identical scheduling
            gradient_clip=100.0,      # Identical stability
        )
        
        # Create Poisson guidance
        poisson_guidance = PoissonGuidance(
            scale=scale,
            background=background,
            read_noise=read_noise,
            config=guidance_config,
        )
        
        # Create L2 guidance with identical parameters
        # Estimate noise variance for L2 (simplified)
        noise_variance = scale + read_noise**2
        l2_guidance = L2Guidance(
            scale=scale,
            background=background,
            noise_variance=noise_variance,
            config=guidance_config,
        )
        
        # Create identical samplers
        poisson_sampler = EDMPosteriorSampler(
            model=self.poisson_model,
            guidance=poisson_guidance,
            device=self.device,
        )
        
        l2_sampler = EDMPosteriorSampler(
            model=self.l2_model,
            guidance=l2_guidance,
            device=self.device,
        )
        
        return poisson_sampler, l2_sampler
    
    def create_conditioning_vectors(
        self,
        domain: str,
        scale: float,
        read_noise: float,
        background: float,
        guidance_type: str,
    ) -> torch.Tensor:
        """
        Create conditioning vectors with fair comparison strategy.
        
        Both methods use the same DomainEncoder but with different conditioning_type:
        - DAPGD: Full physics-aware conditioning
        - L2: Simplified conditioning for fair comparison
        """
        domain_encoder = DomainEncoder()
        
        # Use the same encoder but different conditioning type
        conditioning = domain_encoder.encode_domain(
            domain=domain,
            scale=scale,
            read_noise=read_noise,
            background=background,
            device=self.device,
            conditioning_type=guidance_type,  # "poisson" or "l2"
        )
        
        return conditioning
    
    def apply_poisson_gaussian_noise(
        self,
        clean: torch.Tensor,
        electron_count: float,
        read_noise: float = 10.0,
        background: float = 5.0,
        quantum_efficiency: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        """Apply proper Poisson-Gaussian noise model."""
        clean_norm = torch.clamp(clean, 0, 1)
        photon_image = clean_norm * electron_count
        electron_image = photon_image * quantum_efficiency
        electron_image_with_bg = electron_image + background
        
        # Add Poisson noise
        poisson_noisy = torch.poisson(electron_image_with_bg)
        
        # Add Gaussian read noise
        read_noise_tensor = torch.normal(
            0, read_noise, size=electron_image_with_bg.shape,
            device=electron_image_with_bg.device
        )
        
        noisy_electrons = poisson_noisy + read_noise_tensor
        noisy_electrons = torch.clamp(noisy_electrons, min=0)
        
        return {
            "clean": clean_norm,
            "noisy": noisy_electrons,
            "scale": electron_count * quantum_efficiency,
            "background": background,
            "read_noise": read_noise,
            "quantum_efficiency": quantum_efficiency,
        }
    
    def evaluate_single_sample(
        self,
        clean: torch.Tensor,
        electron_count: float,
        steps: int = 18,
        guidance_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Evaluate both methods on a single sample with identical pipeline."""
        # Apply noise
        noise_data = self.apply_poisson_gaussian_noise(clean, electron_count)
        noisy = noise_data["noisy"]
        scale = noise_data["scale"]
        background = noise_data["background"]
        read_noise = noise_data["read_noise"]
        
        # Create identical samplers
        poisson_sampler, l2_sampler = self.create_identical_samplers(
            scale, background, read_noise, guidance_weight
        )
        
        # Create metadata (identical for both)
        metadata = ImageMetadata(
            original_height=clean.shape[-2],
            original_width=clean.shape[-1],
            scale_factor=1.0,
            pixel_size=1.0,
            pixel_unit="pixel",
            domain="photography",  # Default domain
        )
        
        results = {}
        
        # Evaluate Poisson-Gaussian method
        with torch.no_grad():
            torch.manual_seed(self.seed)  # Ensure identical sampling
            start_time = time.time()
            
            poisson_conditioning = self.create_conditioning_vectors(
                "photography", scale, read_noise, background, "poisson"
            )
            
            poisson_result, poisson_info = poisson_sampler.sample(
                y_observed=noisy,
                metadata=metadata,
                steps=steps,
                guidance_weight=guidance_weight,
                conditioning=poisson_conditioning,
            )
            poisson_time = time.time() - start_time
        
        # Evaluate L2 method with identical sampling
        with torch.no_grad():
            torch.manual_seed(self.seed)  # Ensure identical sampling
            start_time = time.time()
            
            l2_conditioning = self.create_conditioning_vectors(
                "photography", scale, read_noise, background, "l2"
            )
            
            l2_result, l2_info = l2_sampler.sample(
                y_observed=noisy,
                metadata=metadata,
                steps=steps,
                guidance_weight=guidance_weight,
                conditioning=l2_conditioning,
            )
            l2_time = time.time() - start_time
        
        # Compute metrics for both methods
        poisson_metrics = self.evaluation_suite.compute_comprehensive_metrics(
            prediction=poisson_result,
            target=clean,
            noisy=noisy,
            scale=scale,
            background=background,
            read_noise=read_noise,
        )
        
        l2_metrics = self.evaluation_suite.compute_comprehensive_metrics(
            prediction=l2_result,
            target=clean,
            noisy=noisy,
            scale=scale,
            background=background,
            read_noise=read_noise,
        )
        
        return {
            "electron_count": electron_count,
            "scale": scale,
            "background": background,
            "read_noise": read_noise,
            "clean": clean,
            "noisy": noisy,
            "poisson": {
                "result": poisson_result,
                "metrics": poisson_metrics,
                "time": poisson_time,
                "info": poisson_info,
            },
            "l2": {
                "result": l2_result,
                "metrics": l2_metrics,
                "time": l2_time,
                "info": l2_info,
            },
        }
    
    def generate_test_samples(self, num_samples: int = 10) -> List[torch.Tensor]:
        """Generate diverse test samples."""
        samples = []
        pattern_types = ["natural_image", "gaussian_spots", "gradient", "checkerboard", "constant"]
        
        for i in range(num_samples):
            pattern_type = pattern_types[i % len(pattern_types)]
            clean_pattern = self.data_generator.generate_pattern(pattern_type, 128)
            clean = torch.from_numpy(clean_pattern).float().unsqueeze(0).unsqueeze(0)
            samples.append(clean)
        
        return samples
    
    def run_comprehensive_evaluation(
        self,
        electron_ranges: List[float] = None,
        num_samples: int = 10,
        steps: int = 18,
        guidance_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation across electron ranges."""
        if electron_ranges is None:
            electron_ranges = [5000, 1000, 200, 50]
        
        logger.info(f"Running comprehensive evaluation:")
        logger.info(f"  Electron ranges: {electron_ranges}")
        logger.info(f"  Number of samples: {num_samples}")
        logger.info(f"  Sampling steps: {steps}")
        logger.info(f"  Guidance weight: {guidance_weight}")
        
        # Generate test samples
        test_samples = self.generate_test_samples(num_samples)
        
        all_results = {}
        
        for electron_count in electron_ranges:
            logger.info(f"Evaluating at {electron_count} electrons...")
            
            sample_results = []
            for i, clean in enumerate(tqdm(test_samples, desc=f"{electron_count}e")):
                clean = clean.to(self.device)
                result = self.evaluate_single_sample(
                    clean, electron_count, steps, guidance_weight
                )
                result["sample_id"] = i
                sample_results.append(result)
            
            all_results[f"{electron_count:.0f}e"] = sample_results
        
        return all_results
    
    def compute_statistical_significance(
        self, results: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistical significance tests."""
        significance_results = {}
        
        for electron_key, sample_results in results.items():
            poisson_psnrs = [r["poisson"]["metrics"]["psnr"] for r in sample_results]
            l2_psnrs = [r["l2"]["metrics"]["psnr"] for r in sample_results]
            
            poisson_chi2s = [r["poisson"]["metrics"]["chi2_consistency"] for r in sample_results]
            l2_chi2s = [r["l2"]["metrics"]["chi2_consistency"] for r in sample_results]
            
            # Paired t-test for PSNR
            psnr_stat, psnr_pvalue = stats.ttest_rel(poisson_psnrs, l2_psnrs)
            
            # Paired t-test for chi-squared
            chi2_stat, chi2_pvalue = stats.ttest_rel(poisson_chi2s, l2_chi2s)
            
            significance_results[electron_key] = {
                "psnr_improvement_mean": np.mean(poisson_psnrs) - np.mean(l2_psnrs),
                "psnr_improvement_std": np.std(np.array(poisson_psnrs) - np.array(l2_psnrs)),
                "psnr_ttest_stat": psnr_stat,
                "psnr_pvalue": psnr_pvalue,
                "psnr_significant": psnr_pvalue < 0.05,
                "chi2_poisson_mean": np.mean(poisson_chi2s),
                "chi2_l2_mean": np.mean(l2_chi2s),
                "chi2_ttest_stat": chi2_stat,
                "chi2_pvalue": chi2_pvalue,
                "chi2_significant": chi2_pvalue < 0.05,
            }
        
        return significance_results
    
    def create_comparison_visualization(
        self, results: Dict[str, Any], significance: Dict[str, Dict[str, float]]
    ) -> None:
        """Create comprehensive comparison visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        electron_counts = []
        poisson_psnrs = []
        l2_psnrs = []
        poisson_chi2s = []
        l2_chi2s = []
        psnr_improvements = []
        
        for electron_key, sample_results in results.items():
            electron_count = float(electron_key.replace('e', ''))
            electron_counts.append(electron_count)
            
            poisson_psnr = np.mean([r["poisson"]["metrics"]["psnr"] for r in sample_results])
            l2_psnr = np.mean([r["l2"]["metrics"]["psnr"] for r in sample_results])
            poisson_psnrs.append(poisson_psnr)
            l2_psnrs.append(l2_psnr)
            
            poisson_chi2 = np.mean([r["poisson"]["metrics"]["chi2_consistency"] for r in sample_results])
            l2_chi2 = np.mean([r["l2"]["metrics"]["chi2_consistency"] for r in sample_results])
            poisson_chi2s.append(poisson_chi2)
            l2_chi2s.append(l2_chi2)
            
            psnr_improvements.append(poisson_psnr - l2_psnr)
        
        # Plot 1: PSNR comparison
        axes[0, 0].semilogx(electron_counts, poisson_psnrs, 'bo-', label='Poisson-Gaussian', linewidth=2)
        axes[0, 0].semilogx(electron_counts, l2_psnrs, 'ro-', label='L2 Baseline', linewidth=2)
        axes[0, 0].set_xlabel('Electron Count')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].set_title('PSNR vs Electron Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: PSNR improvement
        axes[0, 1].semilogx(electron_counts, psnr_improvements, 'go-', linewidth=2)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Electron Count')
        axes[0, 1].set_ylabel('PSNR Improvement (dB)')
        axes[0, 1].set_title('Poisson-Gaussian Advantage')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Chi-squared comparison
        axes[0, 2].semilogx(electron_counts, poisson_chi2s, 'bo-', label='Poisson-Gaussian', linewidth=2)
        axes[0, 2].semilogx(electron_counts, l2_chi2s, 'ro-', label='L2 Baseline', linewidth=2)
        axes[0, 2].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect χ²=1')
        axes[0, 2].set_xlabel('Electron Count')
        axes[0, 2].set_ylabel('χ² per pixel')
        axes[0, 2].set_title('Physics Consistency (χ²)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Statistical significance
        pvalues = [significance[f"{int(ec)}e"]["psnr_pvalue"] for ec in electron_counts]
        significant = [p < 0.05 for p in pvalues]
        colors = ['green' if s else 'red' for s in significant]
        
        axes[1, 0].bar(range(len(electron_counts)), [-np.log10(p) for p in pvalues], color=colors)
        axes[1, 0].axhline(y=-np.log10(0.05), color='k', linestyle='--', label='p=0.05')
        axes[1, 0].set_xticks(range(len(electron_counts)))
        axes[1, 0].set_xticklabels([f"{int(ec)}e" for ec in electron_counts])
        axes[1, 0].set_ylabel('-log10(p-value)')
        axes[1, 0].set_title('Statistical Significance')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Example restoration comparison
        # Use first sample from lowest electron count
        lowest_electron_key = list(results.keys())[-1]  # Last key (lowest electron count)
        example_result = results[lowest_electron_key][0]
        
        clean_img = example_result["clean"].cpu().numpy().squeeze()
        poisson_img = example_result["poisson"]["result"].cpu().numpy().squeeze()
        l2_img = example_result["l2"]["result"].cpu().numpy().squeeze()
        
        axes[1, 1].imshow(poisson_img, cmap='viridis', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Poisson-Gaussian\n({lowest_electron_key})')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(l2_img, cmap='viridis', vmin=0, vmax=1)
        axes[1, 2].set_title(f'L2 Baseline\n({lowest_electron_key})')
        axes[1, 2].axis('off')
        
        plt.suptitle('L2 Baseline Integration Evaluation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "l2_baseline_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved comparison plot: {plot_file}")
    
    def save_results(
        self, 
        results: Dict[str, Any], 
        significance: Dict[str, Dict[str, float]]
    ) -> None:
        """Save comprehensive results."""
        # Save raw results
        results_file = self.output_dir / "evaluation_results.json"
        
        # Convert tensors to lists for JSON serialization
        serializable_results = {}
        for electron_key, sample_results in results.items():
            serializable_results[electron_key] = []
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
                }
                serializable_results[electron_key].append(serializable_result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save significance results
        significance_file = self.output_dir / "statistical_significance.json"
        with open(significance_file, 'w') as f:
            json.dump(significance, f, indent=2)
        
        # Create summary report
        summary = self.create_summary_report(results, significance)
        summary_file = self.output_dir / "evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Saved results to {self.output_dir}")
    
    def create_summary_report(
        self, 
        results: Dict[str, Any], 
        significance: Dict[str, Dict[str, float]]
    ) -> str:
        """Create human-readable summary report."""
        report = []
        report.append("=" * 80)
        report.append("L2 BASELINE INTEGRATION EVALUATION SUMMARY")
        report.append("=" * 80)
        report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Seed: {self.seed} (deterministic)")
        report.append("")
        
        report.append("PERFORMANCE COMPARISON:")
        report.append("-" * 40)
        
        for electron_key in sorted(results.keys(), key=lambda x: float(x.replace('e', '')), reverse=True):
            sample_results = results[electron_key]
            sig_data = significance[electron_key]
            
            poisson_psnr = np.mean([r["poisson"]["metrics"]["psnr"] for r in sample_results])
            l2_psnr = np.mean([r["l2"]["metrics"]["psnr"] for r in sample_results])
            improvement = sig_data["psnr_improvement_mean"]
            pvalue = sig_data["psnr_pvalue"]
            
            poisson_chi2 = np.mean([r["poisson"]["metrics"]["chi2_consistency"] for r in sample_results])
            l2_chi2 = np.mean([r["l2"]["metrics"]["chi2_consistency"] for r in sample_results])
            
            report.append(f"{electron_key}:")
            report.append(f"  PSNR: Poisson={poisson_psnr:.2f}dB, L2={l2_psnr:.2f}dB")
            report.append(f"  Improvement: {improvement:.2f}dB (p={pvalue:.4f})")
            report.append(f"  χ²: Poisson={poisson_chi2:.3f}, L2={l2_chi2:.3f}")
            report.append(f"  Significant: {'YES' if pvalue < 0.05 else 'NO'}")
            report.append("")
        
        report.append("PHYSICS VALIDATION:")
        report.append("-" * 40)
        for electron_key, sig_data in significance.items():
            poisson_chi2 = sig_data["chi2_poisson_mean"]
            l2_chi2 = sig_data["chi2_l2_mean"]
            
            poisson_physics = "GOOD" if 0.9 <= poisson_chi2 <= 1.1 else "POOR"
            l2_physics = "GOOD" if 0.9 <= l2_chi2 <= 1.1 else "POOR"
            
            report.append(f"{electron_key}: Poisson χ²={poisson_chi2:.3f} ({poisson_physics}), L2 χ²={l2_chi2:.3f} ({l2_physics})")
        
        report.append("")
        report.append("CONCLUSIONS:")
        report.append("-" * 40)
        
        # Overall performance
        all_improvements = [significance[k]["psnr_improvement_mean"] for k in significance.keys()]
        avg_improvement = np.mean(all_improvements)
        
        if avg_improvement > 1.0:
            report.append("✅ Poisson-Gaussian method shows significant advantage over L2 baseline")
        elif avg_improvement > 0.5:
            report.append("⚠️  Poisson-Gaussian method shows moderate advantage over L2 baseline")
        else:
            report.append("❌ Poisson-Gaussian method shows minimal advantage over L2 baseline")
        
        report.append(f"   Average PSNR improvement: {avg_improvement:.2f}dB")
        
        # Physics consistency
        poisson_chi2_values = [significance[k]["chi2_poisson_mean"] for k in significance.keys()]
        l2_chi2_values = [significance[k]["chi2_l2_mean"] for k in significance.keys()]
        
        poisson_good_physics = sum(1 for chi2 in poisson_chi2_values if 0.9 <= chi2 <= 1.1)
        l2_good_physics = sum(1 for chi2 in l2_chi2_values if 0.9 <= chi2 <= 1.1)
        
        report.append(f"   Physics consistency: Poisson {poisson_good_physics}/{len(poisson_chi2_values)}, L2 {l2_good_physics}/{len(l2_chi2_values)}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="L2 Baseline Integration Evaluation")
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
        "--output_dir", 
        type=str, 
        default="l2_baseline_integration_results",
        help="Output directory"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10,
        help="Number of test samples"
    )
    parser.add_argument(
        "--electron_ranges", 
        nargs="+", 
        type=float, 
        default=[5000, 1000, 200, 50],
        help="Electron count ranges to test"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=18,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--guidance_weight", 
        type=float, 
        default=1.0,
        help="Guidance weight"
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
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("L2 BASELINE INTEGRATION EVALUATION")
    logger.info("=" * 80)
    logger.info("Task 3.1: Complete L2 baseline integration")
    logger.info("Features:")
    logger.info("• Identical sampling pipeline for both methods")
    logger.info("• Deterministic evaluation with fixed seeds")
    logger.info("• Identical computational budget")
    logger.info("• Statistical significance testing")
    logger.info("• Fair conditioning strategy differences")
    logger.info("=" * 80)
    
    # Initialize evaluator
    evaluator = L2BaselineIntegrationEvaluator(
        poisson_model_path=args.poisson_model,
        l2_model_path=args.l2_model,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    # Run evaluation
    logger.info("Starting comprehensive evaluation...")
    results = evaluator.run_comprehensive_evaluation(
        electron_ranges=args.electron_ranges,
        num_samples=args.num_samples,
        steps=args.steps,
        guidance_weight=args.guidance_weight,
    )
    
    # Compute statistical significance
    logger.info("Computing statistical significance...")
    significance = evaluator.compute_statistical_significance(results)
    
    # Create visualizations
    logger.info("Creating comparison visualizations...")
    evaluator.create_comparison_visualization(results, significance)
    
    # Save results
    logger.info("Saving results...")
    evaluator.save_results(results, significance)
    
    logger.info("=" * 80)
    logger.info("L2 BASELINE INTEGRATION EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("Key outputs:")
    logger.info("• evaluation_results.json - Raw evaluation data")
    logger.info("• statistical_significance.json - Statistical analysis")
    logger.info("• evaluation_summary.txt - Human-readable summary")
    logger.info("• l2_baseline_comparison.png - Comparison plots")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
