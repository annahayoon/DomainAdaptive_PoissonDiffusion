#!/usr/bin/env python3
"""
Physics-Correct Unified Model Evaluation

This script implements the corrected evaluation using the "Un-normalize, Guide, Re-normalize"
sampling approach that properly separates the normalized prior space from the physical
likelihood space.

Key insight: The prior model p_θ(x) operates in [0,1] normalized space, while the
likelihood guidance ∇log p(y|x) operates in physical space. Proper sampling requires
careful coordinate transformations between these spaces.
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
from core.physics_aware_sampler import PhysicsAwareEDMSampler, create_physics_aware_sampler
from models.edm_wrapper import DomainEncoder, load_pretrained_edm
from core.transforms import ImageMetadata

logger = get_logger(__name__)


class PhysicsCorrectEvaluator:
    """
    Evaluator using physics-correct sampling approach.
    
    This evaluator demonstrates the proper way to combine:
    1. Normalized prior model (trained on [0,1] clean images)
    2. Physical likelihood guidance (operating on real noisy observations)
    """
    
    def __init__(
        self,
        model_path: str,
        data_root: str,
        device: str = "auto",
        output_dir: str = "physics_correct_evaluation_results",
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
        
        # Domain configurations (physical parameters)
        self.domain_configs = {
            "photography": {
                "scale": 1000.0,
                "background": 100.0,
                "read_noise": 10.0,
                "electron_ranges": [5000, 1000, 200],
            },
            "microscopy": {
                "scale": 500.0,
                "background": 50.0,
                "read_noise": 5.0,
                "electron_ranges": [2000, 500, 100],
            },
            "astronomy": {
                "scale": 200.0,
                "background": 20.0,
                "read_noise": 2.0,
                "electron_ranges": [1000, 200, 50],
            },
        }
        
        logger.info("PhysicsCorrectEvaluator initialized successfully")
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
            if hasattr(model, 'config'):
                config = model.config
                logger.info(f"Model channels: {config.img_channels}")
                logger.info(f"Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load unified model: {e}")
            raise
    
    def load_test_samples(self, domain: str, max_samples: int = 3) -> List[Dict[str, Any]]:
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
                        # Handle different channel formats
                        if clean.dim() > 2:
                            if clean.shape[0] == 4:  # 4-channel data (photography)
                                clean = clean  # Keep all 4 channels
                            else:
                                clean = clean.mean(dim=0 if clean.dim() == 3 else 1)  # Average other formats
                        
                        if noisy.dim() > 2:
                            if noisy.shape[0] == 4:  # 4-channel data
                                noisy = noisy  # Keep all 4 channels
                            else:
                                noisy = noisy.mean(dim=0 if noisy.dim() == 3 else 1)
                        
                        # Resize to model size if needed
                        target_size = (128, 128)
                        if clean.shape[-2:] != target_size:
                            if clean.dim() == 2:
                                clean = clean.unsqueeze(0).unsqueeze(0)
                            elif clean.dim() == 3:
                                clean = clean.unsqueeze(0)
                            
                            clean = F.interpolate(clean, size=target_size, mode="bilinear", align_corners=False)
                            clean = clean.squeeze(0) if clean.shape[0] == 1 else clean
                        
                        if noisy.shape[-2:] != target_size:
                            if noisy.dim() == 2:
                                noisy = noisy.unsqueeze(0).unsqueeze(0)
                            elif noisy.dim() == 3:
                                noisy = noisy.unsqueeze(0)
                            
                            noisy = F.interpolate(noisy, size=target_size, mode="bilinear", align_corners=False)
                            noisy = noisy.squeeze(0) if noisy.shape[0] == 1 else noisy
                        
                        # Ensure proper format for model (4 channels)
                        if clean.dim() == 2:
                            clean = clean.unsqueeze(0)  # Add channel dim
                        if clean.shape[0] == 1:
                            clean = clean.repeat(4, 1, 1)  # Replicate to 4 channels
                        
                        if noisy.dim() == 2:
                            noisy = noisy.unsqueeze(0)
                        if noisy.shape[0] == 1:
                            noisy = noisy.repeat(4, 1, 1)
                        
                        # Add batch dimension
                        clean = clean.unsqueeze(0)  # [1, 4, H, W]
                        noisy = noisy.unsqueeze(0)
                        
                        # CRITICAL: Ensure clean is in [0,1] (normalized space)
                        # The model was trained on normalized data
                        clean = torch.clamp(clean, 0, 1)
                        
                        # CRITICAL: Keep noisy in physical space for guidance
                        # Do NOT normalize noisy - it should be in physical units
                        
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
    
    def evaluate_sample_physics_correct(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        domain: str,
        scale: float,
        background: float,
        read_noise: float,
        steps: int = 18,
        guidance_weight: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Evaluate sample using physics-correct approach.
        
        Args:
            clean: Ground truth in normalized [0,1] space
            noisy: Noisy observation in physical space
            domain: Domain name
            scale: Physical scale parameter
            background: Background offset
            read_noise: Read noise std
            steps: Sampling steps
            guidance_weight: Guidance strength
        """
        logger.info(f"Evaluating {domain} sample with physics-correct approach")
        logger.info(f"  Clean range: [{clean.min():.3f}, {clean.max():.3f}] (normalized)")
        logger.info(f"  Noisy range: [{noisy.min():.1f}, {noisy.max():.1f}] (physical)")
        logger.info(f"  Physical params: s={scale}, bg={background}, σ_r={read_noise}")
        
        # Create physics-aware sampler
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
            conditioning_type="dapgd",  # Physics-aware conditioning
        )
        
        # Sample with physics-correct approach
        with torch.no_grad():
            start_time = time.time()
            
            result_dict = sampler.sample(
                y_observed=noisy,
                metadata=metadata,
                condition=conditioning,
                steps=steps,
                guidance_weight=guidance_weight,
                return_intermediates=False,
            )
            
            inference_time = time.time() - start_time
            result = result_dict['sample']
        
        logger.info(f"Sampling completed in {inference_time:.2f}s")
        logger.info(f"Result range: [{result.min():.3f}, {result.max():.3f}] (normalized)")
        
        # Compute metrics (all in normalized space for fair comparison)
        def compute_psnr(pred, target):
            mse = torch.mean((pred - target) ** 2)
            if mse == 0:
                return float('inf')
            return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
        
        def compute_ssim_simple(pred, target):
            # Simple SSIM on first channel
            pred_np = pred[0, 0].detach().cpu().numpy()
            target_np = target[0, 0].detach().cpu().numpy()
            
            mu1, mu2 = np.mean(pred_np), np.mean(target_np)
            var1, var2 = np.var(pred_np), np.var(target_np)
            cov = np.mean((pred_np - mu1) * (target_np - mu2))
            
            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / (
                (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
            )
            return float(np.clip(ssim, 0, 1))
        
        def compute_chi2_physics_correct(pred_norm, noisy_phys, scale, background, read_noise):
            # Convert prediction to physical space for chi2 calculation
            pred_phys = pred_norm * scale
            
            # Expected variance in physical space
            expected_var = pred_phys + read_noise**2
            
            # Chi-squared per pixel
            residuals = (noisy_phys - pred_phys) ** 2
            chi2_per_pixel = torch.mean(residuals / (expected_var + 1e-8))
            return chi2_per_pixel.item()
        
        # Compute metrics
        psnr = compute_psnr(result, clean)
        ssim = compute_ssim_simple(result, clean)
        chi2 = compute_chi2_physics_correct(result, noisy, scale, background, read_noise)
        
        metrics = {
            "psnr": psnr,
            "ssim": ssim,
            "chi2_consistency": chi2,
        }
        
        logger.info(f"Metrics: PSNR={psnr:.2f}dB, SSIM={ssim:.3f}, χ²={chi2:.3f}")
        
        return {
            "domain": domain,
            "scale": scale,
            "background": background,
            "read_noise": read_noise,
            "clean": clean,
            "noisy": noisy,
            "result": result,
            "metrics": metrics,
            "inference_time": inference_time,
        }
    
    def test_no_guidance_baseline(self, domain: str = "photography") -> Dict[str, Any]:
        """Test model without guidance to verify base model stability."""
        logger.info("=" * 60)
        logger.info("TESTING NO-GUIDANCE BASELINE (κ=0)")
        logger.info("=" * 60)
        
        # Load one sample
        samples = self.load_test_samples(domain, max_samples=1)
        if not samples:
            logger.error(f"No samples found for {domain}")
            return {}
        
        sample = samples[0]
        config = self.domain_configs[domain]
        
        # Test with no guidance
        result = self.evaluate_sample_physics_correct(
            clean=sample["clean"].to(self.device),
            noisy=sample["noisy"].to(self.device),
            domain=domain,
            scale=config["scale"],
            background=config["background"],
            read_noise=config["read_noise"],
            guidance_weight=0.0,  # NO GUIDANCE
        )
        
        logger.info("No-guidance baseline results:")
        logger.info(f"  PSNR: {result['metrics']['psnr']:.2f} dB")
        logger.info(f"  SSIM: {result['metrics']['ssim']:.3f}")
        logger.info(f"  χ²: {result['metrics']['chi2_consistency']:.3f}")
        logger.info(f"  Time: {result['inference_time']:.2f}s")
        
        return result
    
    def test_guidance_sweep(self, domain: str = "photography") -> Dict[str, Any]:
        """Test different guidance strengths to find stable range."""
        logger.info("=" * 60)
        logger.info("TESTING GUIDANCE STRENGTH SWEEP")
        logger.info("=" * 60)
        
        # Load one sample
        samples = self.load_test_samples(domain, max_samples=1)
        if not samples:
            logger.error(f"No samples found for {domain}")
            return {}
        
        sample = samples[0]
        config = self.domain_configs[domain]
        
        # Test different guidance strengths
        guidance_weights = [0.0, 0.1, 0.5, 1.0, 2.0]
        results = {}
        
        for weight in guidance_weights:
            logger.info(f"Testing guidance weight: {weight}")
            
            try:
                result = self.evaluate_sample_physics_correct(
                    clean=sample["clean"].to(self.device),
                    noisy=sample["noisy"].to(self.device),
                    domain=domain,
                    scale=config["scale"],
                    background=config["background"],
                    read_noise=config["read_noise"],
                    guidance_weight=weight,
                )
                
                results[f"weight_{weight}"] = result
                logger.info(f"  ✅ κ={weight}: PSNR={result['metrics']['psnr']:.2f}dB, χ²={result['metrics']['chi2_consistency']:.3f}")
                
            except Exception as e:
                logger.error(f"  ❌ κ={weight}: Failed - {e}")
                results[f"weight_{weight}"] = {"error": str(e)}
        
        return results
    
    def run_comprehensive_evaluation(self, max_samples: int = 2) -> Dict[str, Any]:
        """Run comprehensive evaluation with physics-correct approach."""
        logger.info("=" * 60)
        logger.info("PHYSICS-CORRECT COMPREHENSIVE EVALUATION")
        logger.info("=" * 60)
        
        all_results = {}
        
        for domain in ["photography", "microscopy", "astronomy"]:
            logger.info(f"Evaluating domain: {domain}")
            
            samples = self.load_test_samples(domain, max_samples)
            if not samples:
                logger.warning(f"No samples for {domain}")
                continue
            
            config = self.domain_configs[domain]
            domain_results = {}
            
            # Test one electron range per domain
            electron_count = config["electron_ranges"][0]
            logger.info(f"  Testing {electron_count} electrons...")
            
            sample_results = []
            for sample in samples:
                try:
                    result = self.evaluate_sample_physics_correct(
                        clean=sample["clean"].to(self.device),
                        noisy=sample["noisy"].to(self.device),
                        domain=domain,
                        scale=config["scale"],
                        background=config["background"],
                        read_noise=config["read_noise"],
                        guidance_weight=1.0,  # Standard guidance
                    )
                    
                    result["sample_id"] = sample["sample_id"]
                    result["file_name"] = sample["file_name"]
                    sample_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {sample['file_name']}: {e}")
            
            if sample_results:
                # Compute statistics
                psnrs = [r["metrics"]["psnr"] for r in sample_results]
                chi2s = [r["metrics"]["chi2_consistency"] for r in sample_results]
                times = [r["inference_time"] for r in sample_results]
                
                domain_results[f"{electron_count}e"] = {
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
            
            all_results[domain] = domain_results
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], test_name: str = "physics_correct") -> None:
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        results_file = self.output_dir / f"{test_name}_results_{timestamp}.json"
        
        # Prepare serializable results
        serializable_results = {}
        for domain, domain_results in results.items():
            if isinstance(domain_results, dict) and "statistics" in domain_results.get(list(domain_results.keys())[0], {}):
                serializable_results[domain] = {}
                for electron_key, data in domain_results.items():
                    serializable_results[domain][electron_key] = {
                        "electron_count": data.get("electron_count", 0),
                        "statistics": data.get("statistics", {}),
                        "num_samples": len(data.get("samples", [])),
                    }
            else:
                serializable_results[domain] = {"error": "No valid results"}
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Physics-Correct Unified Model Evaluation")
    parser.add_argument("--model_path", type=str, default="~/checkpoint_step_0090000.pth")
    parser.add_argument("--data_root", type=str, default="~/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed/posterior")
    parser.add_argument("--output_dir", type=str, default="physics_correct_evaluation_results")
    parser.add_argument("--test_mode", type=str, choices=["no_guidance", "guidance_sweep", "comprehensive"], default="no_guidance")
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Expand paths
    args.model_path = str(Path(args.model_path).expanduser())
    args.data_root = str(Path(args.data_root).expanduser())
    
    logger.info("=" * 80)
    logger.info("PHYSICS-CORRECT UNIFIED MODEL EVALUATION")
    logger.info("=" * 80)
    logger.info("Implementing proper 'Un-normalize, Guide, Re-normalize' approach")
    logger.info("Separating normalized prior space from physical likelihood space")
    logger.info("=" * 80)
    
    # Initialize evaluator
    evaluator = PhysicsCorrectEvaluator(
        model_path=args.model_path,
        data_root=args.data_root,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    # Run selected test
    if args.test_mode == "no_guidance":
        logger.info("Running no-guidance baseline test...")
        results = evaluator.test_no_guidance_baseline()
        evaluator.save_results({"no_guidance": results}, "no_guidance_baseline")
        
    elif args.test_mode == "guidance_sweep":
        logger.info("Running guidance strength sweep...")
        results = evaluator.test_guidance_sweep()
        evaluator.save_results({"guidance_sweep": results}, "guidance_sweep")
        
    elif args.test_mode == "comprehensive":
        logger.info("Running comprehensive evaluation...")
        results = evaluator.run_comprehensive_evaluation(args.max_samples)
        evaluator.save_results(results, "comprehensive")
    
    logger.info("=" * 80)
    logger.info("PHYSICS-CORRECT EVALUATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
