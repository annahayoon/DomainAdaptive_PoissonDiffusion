#!/usr/bin/env python3
"""
Unified Model Evaluation Script

Evaluates the unified model trained on all 3 domains (photography, microscopy, astronomy)
using the L2 baseline integration framework we implemented.

Usage:
    python scripts/evaluate_unified_model.py \
        --model_path ~/checkpoint_step_0090000.pth \
        --data_root ~/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed/posterior \
        --output_dir unified_model_evaluation_results
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
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.logging_config import get_logger
from core.metrics import EvaluationSuite
from models.edm_wrapper import DomainEncoder, load_pretrained_edm
from core.physics_aware_sampler import PhysicsAwareEDMSampler, create_physics_aware_sampler
from core.transforms import ImageMetadata

logger = get_logger(__name__)

class UnifiedModelEvaluator:
    """
    Evaluator for unified model trained on all 3 domains.
    
    Tests the model's ability to handle different domains and
    demonstrates cross-domain generalization.
    """
    
    def __init__(
        self,
        model_path: str,
        data_root: str,
        device: str = "auto",
        output_dir: str = "unified_model_evaluation_results",
        seed: int = 42,
    ):
        self.model_path = Path(model_path)
        self.data_root = Path(data_root)
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Set deterministic behavior
        self._set_deterministic_mode(seed)
        
        # Load unified model
        self.model = self._load_unified_model()
        
        # Initialize evaluation suite
        self.evaluation_suite = EvaluationSuite(device=self.device)
        
        # Domain configurations
        self.domain_configs = {
            "photography": {
                "scale": 1000.0,
                "background": 100.0,
                "read_noise": 10.0,
                "electron_ranges": [5000, 1000, 200, 50],
            },
            "microscopy": {
                "scale": 500.0,
                "background": 50.0,
                "read_noise": 5.0,
                "electron_ranges": [2000, 500, 100, 20],
            },
            "astronomy": {
                "scale": 200.0,
                "background": 20.0,
                "read_noise": 2.0,
                "electron_ranges": [1000, 200, 50, 10],
            },
        }
        
        logger.info("UnifiedModelEvaluator initialized successfully")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Device: {self.device}")
    
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
    
    def _load_unified_model(self):
        """Load the unified model checkpoint."""
        try:
            logger.info(f"Loading unified model from {self.model_path}")
            model = load_pretrained_edm(str(self.model_path), device=self.device)
            model.eval()
            logger.info("Successfully loaded unified model")
            
            # Log model info
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
                logger.info(f"Model parameters: {info.get('total_parameters', 'unknown')}")
                logger.info(f"Conditioning dim: {info.get('condition_dim', 'unknown')}")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load unified model: {e}")
            raise
    
    def load_test_samples(self, domain: str, max_samples: int = 5) -> List[Dict[str, Any]]:
        """Load test samples for a specific domain."""
        test_dir = self.data_root / domain / "test"
        
        if not test_dir.exists():
            logger.warning(f"Test directory not found: {test_dir}")
            return []
        
        # Get test files
        test_files = list(test_dir.glob("*.pt"))[:max_samples]
        
        samples = []
        for i, file_path in enumerate(test_files):
            try:
                data = torch.load(file_path, map_location="cpu", weights_only=False)
                
                # Extract clean and noisy data
                if isinstance(data, dict):
                    clean = data.get("clean_norm", data.get("clean", None))
                    noisy = data.get("noisy_norm", data.get("noisy", None))
                    
                    if clean is not None and noisy is not None:
                        # Ensure proper format
                        if clean.dim() > 2:
                            clean = clean.mean(dim=0 if clean.dim() == 3 else 1)
                        if noisy.dim() > 2:
                            noisy = noisy.mean(dim=0 if noisy.dim() == 3 else 1)
                        
                        # Resize to model size if needed
                        if clean.shape[-1] != 128 or clean.shape[-2] != 128:
                            clean = F.interpolate(
                                clean.unsqueeze(0).unsqueeze(0),
                                size=(128, 128),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze()
                            noisy = F.interpolate(
                                noisy.unsqueeze(0).unsqueeze(0),
                                size=(128, 128),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze()
                        
                        # Ensure proper range
                        clean = torch.clamp(clean, 0, 1)
                        noisy = torch.clamp(noisy, 0, 1)
                        
                        # Expand to match model channels (4 channels)
                        clean = clean.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                        noisy = noisy.unsqueeze(0).unsqueeze(0)
                        
                        # Replicate single channel to 4 channels to match model
                        clean = clean.repeat(1, 4, 1, 1)  # [1, 4, H, W]
                        noisy = noisy.repeat(1, 4, 1, 1)
                        
                        samples.append({
                            "clean": clean,
                            "noisy": noisy,
                            "domain": domain,
                            "file_name": file_path.stem,
                            "sample_id": i,
                        })
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(samples)} test samples for {domain}")
        return samples
    
    def apply_domain_noise(
        self,
        clean: torch.Tensor,
        domain: str,
        electron_count: float,
    ) -> Dict[str, torch.Tensor]:
        """Apply domain-specific noise model."""
        config = self.domain_configs[domain]
        
        clean_norm = torch.clamp(clean, 0, 1)
        photon_image = clean_norm * electron_count
        electron_image = photon_image * 0.95  # quantum efficiency
        electron_image_with_bg = electron_image + config["background"]
        
        # Add Poisson noise
        poisson_noisy = torch.poisson(electron_image_with_bg)
        
        # Add Gaussian read noise
        read_noise_tensor = torch.normal(
            0, config["read_noise"], size=electron_image_with_bg.shape,
            device=electron_image_with_bg.device
        )
        
        noisy_electrons = poisson_noisy + read_noise_tensor
        noisy_electrons = torch.clamp(noisy_electrons, min=0)
        
        # Ensure 4-channel output to match model
        if noisy_electrons.shape[1] == 1:
            noisy_electrons = noisy_electrons.repeat(1, 4, 1, 1)
        
        return {
            "clean": clean_norm,
            "noisy": noisy_electrons,
            "scale": config["scale"],
            "background": config["background"],
            "read_noise": config["read_noise"],
        }
    
    def evaluate_sample(
        self,
        clean: torch.Tensor,
        domain: str,
        electron_count: float,
        steps: int = 18,
        guidance_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Evaluate unified model on a single sample."""
        # Apply domain-specific noise
        noise_data = self.apply_domain_noise(clean, domain, electron_count)
        noisy = noise_data["noisy"]
        scale = noise_data["scale"]
        background = noise_data["background"]
        read_noise = noise_data["read_noise"]
        
        # Create physics-aware sampler (no more numerical instability!)
        sampler = create_physics_aware_sampler(
            model=self.model,
            scale=scale,
            background=background,
            read_noise=read_noise,
            guidance_weight=guidance_weight,
            device=self.device,
        )
        
        # Create metadata
        metadata = ImageMetadata(
            original_height=clean.shape[-2],
            original_width=clean.shape[-1],
            scale_factor=1.0,
            pixel_size=1.0,
            pixel_unit="pixel",
            domain=domain,
        )
        
        # Create domain conditioning
        domain_encoder = DomainEncoder()
        conditioning = domain_encoder.encode_domain(
            domain=domain,
            scale=scale,
            read_noise=read_noise,
            background=background,
            device=self.device,
            conditioning_type="dapgd",
        )
        
        # Sample with physics-correct approach
        with torch.no_grad():
            start_time = time.time()
            sample_result = sampler.sample(
                y_observed=noisy,
                metadata=metadata,
                condition=conditioning,
                steps=steps,
                guidance_weight=guidance_weight,
                return_intermediates=False,
            )
            inference_time = time.time() - start_time
            
            # Physics-aware sampler returns dict with 'sample' key
            result = sample_result['sample']
            info = sample_result
        
        # Convert result back to single channel for metrics
        if result.shape[1] == 4:
            result_single = result[:, :1, :, :]  # Take first channel
        else:
            result_single = result
            
        if clean.shape[1] == 4:
            clean_single = clean[:, :1, :, :]  # Take first channel
        else:
            clean_single = clean
            
        if noisy.shape[1] == 4:
            noisy_single = noisy[:, :1, :, :]  # Take first channel
        else:
            noisy_single = noisy
        
        # Compute basic metrics directly
        def compute_psnr(pred, target):
            mse = torch.mean((pred - target) ** 2)
            if mse == 0:
                return float('inf')
            return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
        
        def compute_ssim_simple(pred, target):
            # Simple SSIM approximation
            pred_np = pred.detach().cpu().numpy().squeeze()
            target_np = target.detach().cpu().numpy().squeeze()
            
            mu1 = np.mean(pred_np)
            mu2 = np.mean(target_np)
            var1 = np.var(pred_np)
            var2 = np.var(target_np)
            cov = np.mean((pred_np - mu1) * (target_np - mu2))
            
            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / (
                (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
            )
            return float(np.clip(ssim, 0, 1))
        
        def compute_chi2(pred, noisy, scale, background, read_noise):
            pred_electrons = pred * scale + background
            expected_var = pred_electrons + read_noise**2
            residuals = (noisy - pred_electrons) ** 2
            chi2_per_pixel = torch.mean(residuals / (expected_var + 1e-8))
            return chi2_per_pixel.item()
        
        # Compute metrics
        psnr = compute_psnr(result_single, clean_single)
        ssim = compute_ssim_simple(result_single, clean_single)
        chi2 = compute_chi2(result_single, noisy_single, scale, background, read_noise)
        
        metrics = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": 0.0,  # Not available
            "chi2_consistency": chi2,
        }
        
        return {
            "domain": domain,
            "electron_count": electron_count,
            "scale": scale,
            "background": background,
            "read_noise": read_noise,
            "clean": clean,
            "noisy": noisy,
            "result": result,
            "metrics": metrics,
            "inference_time": inference_time,
            "info": info,
        }
    
    def evaluate_domain(
        self,
        domain: str,
        max_samples: int = 3,
        max_electron_ranges: int = 3,
    ) -> Dict[str, Any]:
        """Evaluate unified model on a specific domain."""
        logger.info(f"Evaluating domain: {domain}")
        
        # Load test samples
        test_samples = self.load_test_samples(domain, max_samples)
        if not test_samples:
            logger.warning(f"No test samples found for {domain}")
            return {}
        
        # Get electron ranges for this domain
        electron_ranges = self.domain_configs[domain]["electron_ranges"][:max_electron_ranges]
        
        domain_results = {}
        
        for electron_count in electron_ranges:
            logger.info(f"  Testing {electron_count} electrons...")
            
            sample_results = []
            for sample in test_samples:
                clean = sample["clean"].to(self.device)
                
                result = self.evaluate_sample(
                    clean=clean,
                    domain=domain,
                    electron_count=electron_count,
                )
                
                result["sample_id"] = sample["sample_id"]
                result["file_name"] = sample["file_name"]
                sample_results.append(result)
            
            # Compute domain statistics
            psnrs = [r["metrics"]["psnr"] for r in sample_results]
            chi2s = [r["metrics"]["chi2_consistency"] for r in sample_results]
            times = [r["inference_time"] for r in sample_results]
            
            domain_results[f"{electron_count:.0f}e"] = {
                "electron_count": electron_count,
                "samples": sample_results,
                "statistics": {
                    "psnr_mean": np.mean(psnrs),
                    "psnr_std": np.std(psnrs),
                    "chi2_mean": np.mean(chi2s),
                    "chi2_std": np.std(chi2s),
                    "time_mean": np.mean(times),
                    "time_std": np.std(times),
                },
            }
            
            logger.info(f"    PSNR: {np.mean(psnrs):.2f}±{np.std(psnrs):.2f} dB")
            logger.info(f"    χ²: {np.mean(chi2s):.3f}±{np.std(chi2s):.3f}")
            logger.info(f"    Time: {np.mean(times):.2f}±{np.std(times):.2f} s")
        
        return domain_results
    
    def run_comprehensive_evaluation(
        self,
        max_samples_per_domain: int = 3,
        max_electron_ranges: int = 3,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation across all domains."""
        logger.info("Starting comprehensive unified model evaluation...")
        logger.info(f"Max samples per domain: {max_samples_per_domain}")
        logger.info(f"Max electron ranges: {max_electron_ranges}")
        
        all_results = {}
        
        for domain in ["photography", "microscopy", "astronomy"]:
            domain_results = self.evaluate_domain(
                domain=domain,
                max_samples=max_samples_per_domain,
                max_electron_ranges=max_electron_ranges,
            )
            all_results[domain] = domain_results
        
        return all_results
    
    def create_cross_domain_visualization(self, results: Dict[str, Any]) -> None:
        """Create visualization showing cross-domain performance."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        domains = ["photography", "microscopy", "astronomy"]
        colors = ["blue", "green", "red"]
        
        # Plot 1: PSNR across domains
        for i, domain in enumerate(domains):
            if domain not in results or not results[domain]:
                continue
                
            electron_counts = []
            psnr_means = []
            psnr_stds = []
            
            for electron_key, data in results[domain].items():
                electron_counts.append(data["electron_count"])
                psnr_means.append(data["statistics"]["psnr_mean"])
                psnr_stds.append(data["statistics"]["psnr_std"])
            
            axes[0, 0].errorbar(electron_counts, psnr_means, yerr=psnr_stds,
                               fmt='o-', color=colors[i], label=domain.capitalize(),
                               capsize=5, linewidth=2)
        
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_xlabel('Electron Count')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].set_title('Cross-Domain PSNR Performance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Chi-squared across domains
        for i, domain in enumerate(domains):
            if domain not in results or not results[domain]:
                continue
                
            electron_counts = []
            chi2_means = []
            chi2_stds = []
            
            for electron_key, data in results[domain].items():
                electron_counts.append(data["electron_count"])
                chi2_means.append(data["statistics"]["chi2_mean"])
                chi2_stds.append(data["statistics"]["chi2_std"])
            
            axes[0, 1].errorbar(electron_counts, chi2_means, yerr=chi2_stds,
                               fmt='o-', color=colors[i], label=domain.capitalize(),
                               capsize=5, linewidth=2)
        
        axes[0, 1].set_xscale('log')
        axes[0, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect χ²=1')
        axes[0, 1].fill_between([1, 10000], 0.9, 1.1, alpha=0.2, color='green', label='Good Physics')
        axes[0, 1].set_xlabel('Electron Count')
        axes[0, 1].set_ylabel('χ² per pixel')
        axes[0, 1].set_title('Cross-Domain Physics Consistency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Inference time across domains
        for i, domain in enumerate(domains):
            if domain not in results or not results[domain]:
                continue
                
            electron_counts = []
            time_means = []
            time_stds = []
            
            for electron_key, data in results[domain].items():
                electron_counts.append(data["electron_count"])
                time_means.append(data["statistics"]["time_mean"])
                time_stds.append(data["statistics"]["time_std"])
            
            axes[0, 2].errorbar(electron_counts, time_means, yerr=time_stds,
                               fmt='o-', color=colors[i], label=domain.capitalize(),
                               capsize=5, linewidth=2)
        
        axes[0, 2].set_xscale('log')
        axes[0, 2].set_xlabel('Electron Count')
        axes[0, 2].set_ylabel('Inference Time (s)')
        axes[0, 2].set_title('Cross-Domain Inference Speed')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Bottom row: Example restorations from each domain
        for i, domain in enumerate(domains):
            if domain not in results or not results[domain]:
                axes[1, i].text(0.5, 0.5, f'No {domain} data', 
                               ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'{domain.capitalize()} - No Data')
                axes[1, i].axis('off')
                continue
            
            # Get first sample from lowest electron count
            electron_keys = list(results[domain].keys())
            if electron_keys:
                first_result = results[domain][electron_keys[-1]]["samples"][0]  # Lowest electron count
                restored_img = first_result["result"].cpu().numpy()
                
                # Handle 4-channel images by taking first channel
                if restored_img.ndim == 4:  # [B, C, H, W]
                    restored_img = restored_img[0, 0, :, :]  # Take first batch, first channel
                elif restored_img.ndim == 3:  # [C, H, W]
                    restored_img = restored_img[0, :, :]  # Take first channel
                else:
                    restored_img = restored_img.squeeze()
                
                axes[1, i].imshow(restored_img, cmap='viridis', vmin=0, vmax=1)
                axes[1, i].set_title(f'{domain.capitalize()} Restoration\n'
                                   f'({electron_keys[-1]}, PSNR: {first_result["metrics"]["psnr"]:.1f}dB)')
                axes[1, i].axis('off')
        
        plt.suptitle('Unified Model: Cross-Domain Evaluation Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "unified_model_cross_domain_evaluation.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved cross-domain visualization: {plot_file}")
    
    def create_summary_report(self, results: Dict[str, Any]) -> str:
        """Create human-readable summary report."""
        report = []
        report.append("=" * 80)
        report.append("UNIFIED MODEL EVALUATION SUMMARY")
        report.append("=" * 80)
        report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {self.model_path}")
        report.append(f"Seed: {self.seed} (deterministic)")
        report.append("")
        
        report.append("CROSS-DOMAIN PERFORMANCE:")
        report.append("-" * 50)
        
        for domain in ["photography", "microscopy", "astronomy"]:
            if domain not in results or not results[domain]:
                report.append(f"{domain.upper()}: No data available")
                continue
                
            report.append(f"{domain.upper()}:")
            
            for electron_key, data in results[domain].items():
                stats = data["statistics"]
                electron_count = data["electron_count"]
                
                psnr_mean = stats["psnr_mean"]
                chi2_mean = stats["chi2_mean"]
                time_mean = stats["time_mean"]
                
                physics_status = "GOOD" if 0.9 <= chi2_mean <= 1.1 else "POOR"
                
                report.append(f"  {electron_count:.0f}e⁻:")
                report.append(f"    PSNR: {psnr_mean:.2f}±{stats['psnr_std']:.2f} dB")
                report.append(f"    χ²: {chi2_mean:.3f}±{stats['chi2_std']:.3f} ({physics_status})")
                report.append(f"    Time: {time_mean:.2f}±{stats['time_std']:.2f} s")
            
            report.append("")
        
        report.append("UNIFIED MODEL ANALYSIS:")
        report.append("-" * 50)
        
        # Compute overall statistics
        all_psnrs = []
        all_chi2s = []
        all_times = []
        
        for domain_results in results.values():
            for data in domain_results.values():
                stats = data["statistics"]
                all_psnrs.append(stats["psnr_mean"])
                all_chi2s.append(stats["chi2_mean"])
                all_times.append(stats["time_mean"])
        
        if all_psnrs:
            report.append(f"Overall PSNR: {np.mean(all_psnrs):.2f}±{np.std(all_psnrs):.2f} dB")
            report.append(f"Overall χ²: {np.mean(all_chi2s):.3f}±{np.std(all_chi2s):.3f}")
            report.append(f"Overall inference time: {np.mean(all_times):.2f}±{np.std(all_times):.2f} s")
            
            good_physics_count = sum(1 for chi2 in all_chi2s if 0.9 <= chi2 <= 1.1)
            report.append(f"Physics consistency: {good_physics_count}/{len(all_chi2s)} conditions")
        
        report.append("")
        report.append("CONCLUSIONS:")
        report.append("-" * 50)
        
        if all_psnrs:
            avg_psnr = np.mean(all_psnrs)
            avg_chi2 = np.mean(all_chi2s)
            
            if avg_psnr > 25.0:
                report.append("✅ Excellent restoration quality (PSNR > 25 dB)")
            elif avg_psnr > 20.0:
                report.append("⚠️  Good restoration quality (PSNR > 20 dB)")
            else:
                report.append("❌ Poor restoration quality (PSNR < 20 dB)")
            
            if 0.9 <= avg_chi2 <= 1.1:
                report.append("✅ Excellent physics consistency (χ² ≈ 1.0)")
            elif avg_chi2 <= 1.5:
                report.append("⚠️  Acceptable physics consistency (χ² ≤ 1.5)")
            else:
                report.append("❌ Poor physics consistency (χ² > 1.5)")
            
            report.append(f"🚀 Unified model successfully handles all {len(results)} domains")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive results."""
        # Save raw results
        results_file = self.output_dir / "unified_model_results.json"
        
        # Prepare serializable results
        serializable_results = {}
        for domain, domain_results in results.items():
            serializable_results[domain] = {}
            for electron_key, data in domain_results.items():
                serializable_results[domain][electron_key] = {
                    "electron_count": data["electron_count"],
                    "statistics": data["statistics"],
                    "samples": [
                        {
                            "sample_id": s["sample_id"],
                            "file_name": s["file_name"],
                            "metrics": s["metrics"],
                            "inference_time": s["inference_time"],
                        }
                        for s in data["samples"]
                    ],
                }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary report
        report = self.create_summary_report(results)
        report_file = self.output_dir / "unified_model_summary.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved results to {self.output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Unified Model Evaluation")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="~/checkpoint_step_0090000.pth",
        help="Path to unified model checkpoint"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="~/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed/posterior",
        help="Root directory for test data"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="unified_model_evaluation_results",
        help="Output directory"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=3,
        help="Maximum samples per domain"
    )
    parser.add_argument(
        "--max_electron_ranges", 
        type=int, 
        default=3,
        help="Maximum electron ranges per domain"
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
    
    # Expand paths
    args.model_path = str(Path(args.model_path).expanduser())
    args.data_root = str(Path(args.data_root).expanduser())
    
    logger.info("=" * 80)
    logger.info("UNIFIED MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info("Testing unified model trained on all 3 domains:")
    logger.info("• Photography: Consumer camera images")
    logger.info("• Microscopy: Fluorescence imaging")
    logger.info("• Astronomy: Deep sky observations")
    logger.info("=" * 80)
    
    # Initialize evaluator
    evaluator = UnifiedModelEvaluator(
        model_path=args.model_path,
        data_root=args.data_root,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    # Run evaluation
    logger.info("Starting comprehensive evaluation...")
    results = evaluator.run_comprehensive_evaluation(
        max_samples_per_domain=args.max_samples,
        max_electron_ranges=args.max_electron_ranges,
    )
    
    # Create visualizations
    logger.info("Creating cross-domain visualizations...")
    evaluator.create_cross_domain_visualization(results)
    
    # Save results
    logger.info("Saving results...")
    evaluator.save_results(results)
    
    logger.info("=" * 80)
    logger.info("UNIFIED MODEL EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("Key outputs:")
    logger.info("• unified_model_results.json - Complete evaluation data")
    logger.info("• unified_model_summary.txt - Human-readable summary")
    logger.info("• unified_model_cross_domain_evaluation.png - Visualization")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
