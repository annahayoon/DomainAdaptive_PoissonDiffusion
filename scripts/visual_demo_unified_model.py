#!/usr/bin/env python3
"""
Visual Demo: Unified Model Performance

Creates visual comparisons showing:
1. Clean ground truth images
2. Noisy observations  
3. Restored images without guidance (prior only)
4. Restored images with physics-aware guidance

This demonstrates the effectiveness of the physics-correct approach.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.logging_config import get_logger
from core.physics_aware_sampler import create_physics_aware_sampler
from models.edm_wrapper import DomainEncoder, load_pretrained_edm
from core.transforms import ImageMetadata

logger = get_logger(__name__)


class VisualDemoEvaluator:
    """Visual demonstration of unified model performance."""
    
    def __init__(
        self,
        model_path: str,
        data_root: str,
        device: str = "auto",
        seed: int = 42,
    ):
        self.model_path = Path(model_path)
        self.data_root = Path(data_root)
        self.device = self._setup_device(device)
        self.seed = seed
        
        # Set deterministic behavior
        self._set_deterministic_mode(seed)
        
        # Load unified model
        self.model = self._load_unified_model()
        
        # Domain configurations
        self.domain_configs = {
            "photography": {
                "scale": 1000.0,
                "background": 100.0,
                "read_noise": 10.0,
            },
            "microscopy": {
                "scale": 500.0,
                "background": 50.0,
                "read_noise": 5.0,
            },
            "astronomy": {
                "scale": 200.0,
                "background": 20.0,
                "read_noise": 2.0,
            },
        }
        
        logger.info("VisualDemoEvaluator initialized successfully")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def _set_deterministic_mode(self, seed: int) -> None:
        """Set deterministic behavior."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _load_unified_model(self):
        """Load the unified model checkpoint."""
        try:
            logger.info(f"Loading unified model from {self.model_path}")
            model = load_pretrained_edm(str(self.model_path), device=self.device)
            model.eval()
            logger.info("Successfully loaded unified model")
            return model
        except Exception as e:
            logger.error(f"Failed to load unified model: {e}")
            raise
    
    def load_sample_data(self, domain: str) -> Dict[str, torch.Tensor]:
        """Load one sample from the specified domain."""
        test_dir = self.data_root / domain / "test"
        
        if not test_dir.exists():
            raise ValueError(f"Test directory not found: {test_dir}")
        
        # Get first test file
        test_files = list(test_dir.glob("*.pt"))
        if not test_files:
            raise ValueError(f"No test files found in {test_dir}")
        
        # Load first file
        data = torch.load(test_files[0], map_location="cpu", weights_only=False)
        
        # Extract clean and noisy data
        if isinstance(data, dict):
            clean = data.get("clean_norm", data.get("clean", None))
            noisy = data.get("noisy_norm", data.get("noisy", None))
            
            if clean is None or noisy is None:
                raise ValueError(f"Could not find clean/noisy data in {test_files[0]}")
            
            # Handle different formats
            if clean.dim() > 2:
                if clean.shape[0] == 4:  # 4-channel data (photography)
                    clean = clean  # Keep all 4 channels
                else:
                    clean = clean.mean(dim=0 if clean.dim() == 3 else 1)
            
            if noisy.dim() > 2:
                if noisy.shape[0] == 4:  # 4-channel data
                    noisy = noisy  # Keep all 4 channels
                else:
                    noisy = noisy.mean(dim=0 if noisy.dim() == 3 else 1)
            
            # Resize to model size (128x128)
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
                clean = clean.unsqueeze(0)
            if clean.shape[0] == 1:
                clean = clean.repeat(4, 1, 1)
            
            if noisy.dim() == 2:
                noisy = noisy.unsqueeze(0)
            if noisy.shape[0] == 1:
                noisy = noisy.repeat(4, 1, 1)
            
            # Add batch dimension
            clean = clean.unsqueeze(0)  # [1, 4, H, W]
            noisy = noisy.unsqueeze(0)
            
            # CRITICAL: Ensure clean is in [0,1] (normalized space)
            clean = torch.clamp(clean, 0, 1)
            
            return {
                "clean": clean,
                "noisy": noisy,
                "file_name": test_files[0].stem,
            }
        
        else:
            raise ValueError(f"Unexpected data format in {test_files[0]}")
    
    def restore_image(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        domain: str,
        guidance_weight: float = 1.0,
        steps: int = 18,
    ) -> Dict[str, torch.Tensor]:
        """Restore image using physics-correct approach."""
        config = self.domain_configs[domain]
        
        # Create physics-aware sampler
        sampler = create_physics_aware_sampler(
            model=self.model,
            scale=config["scale"],
            background=config["background"],
            read_noise=config["read_noise"],
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
            scale=config["scale"],
            read_noise=config["read_noise"],
            background=config["background"],
            device=self.device,
            conditioning_type="dapgd",
        )
        
        # Sample
        with torch.no_grad():
            result_dict = sampler.sample(
                y_observed=noisy,
                metadata=metadata,
                condition=conditioning,
                steps=steps,
                guidance_weight=guidance_weight,
                return_intermediates=False,
            )
        
        return result_dict['sample']
    
    def create_comparison_figure(
        self,
        domain: str,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        restored_no_guidance: torch.Tensor,
        restored_with_guidance: torch.Tensor,
        output_path: str,
    ) -> None:
        """Create comparison figure showing all four images."""
        
        # Convert to numpy and take first channel for visualization
        def to_numpy_single_channel(tensor):
            return tensor[0, 0].detach().cpu().numpy()
        
        clean_np = to_numpy_single_channel(clean)
        noisy_np = to_numpy_single_channel(noisy)
        restored_no_guidance_np = to_numpy_single_channel(restored_no_guidance)
        restored_with_guidance_np = to_numpy_single_channel(restored_with_guidance)
        
        # Compute metrics
        def compute_psnr(pred, target):
            mse = np.mean((pred - target) ** 2)
            if mse == 0:
                return float('inf')
            return 20 * np.log10(1.0 / np.sqrt(mse))
        
        psnr_noisy = compute_psnr(noisy_np, clean_np)
        psnr_no_guidance = compute_psnr(restored_no_guidance_np, clean_np)
        psnr_with_guidance = compute_psnr(restored_with_guidance_np, clean_np)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Clean image
        im1 = axes[0, 0].imshow(clean_np, cmap='viridis', vmin=0, vmax=1)
        axes[0, 0].set_title('Clean Ground Truth', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Noisy image
        im2 = axes[0, 1].imshow(noisy_np, cmap='viridis')
        axes[0, 1].set_title(f'Noisy Observation\nPSNR: {psnr_noisy:.1f} dB', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Restored without guidance
        im3 = axes[1, 0].imshow(restored_no_guidance_np, cmap='viridis', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Prior Only (κ=0)\nPSNR: {psnr_no_guidance:.1f} dB', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Restored with guidance
        im4 = axes[1, 1].imshow(restored_with_guidance_np, cmap='viridis', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Physics-Aware (κ=1.0)\nPSNR: {psnr_with_guidance:.1f} dB', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Overall title
        improvement = psnr_with_guidance - psnr_no_guidance
        plt.suptitle(
            f'{domain.capitalize()} Domain: Physics-Aware Restoration\n'
            f'Improvement: {improvement:+.1f} dB (Physics-Aware vs Prior Only)',
            fontsize=16, fontweight='bold', y=0.98
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved comparison figure: {output_path}")
        logger.info(f"PSNR improvements: Noisy→Prior: {psnr_no_guidance-psnr_noisy:+.1f} dB, Prior→Physics: {improvement:+.1f} dB")
    
    def run_visual_demo(self, domains: List[str], output_dir: str = "visual_demo_results") -> None:
        """Run visual demonstration for specified domains."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("VISUAL DEMO: UNIFIED MODEL PERFORMANCE")
        logger.info("=" * 80)
        logger.info("Comparing: Clean → Noisy → Prior Only → Physics-Aware")
        logger.info("=" * 80)
        
        for domain in domains:
            logger.info(f"Processing {domain} domain...")
            
            try:
                # Load sample data
                sample_data = self.load_sample_data(domain)
                clean = sample_data["clean"].to(self.device)
                noisy = sample_data["noisy"].to(self.device)
                
                logger.info(f"  Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
                logger.info(f"  Noisy range: [{noisy.min():.1f}, {noisy.max():.1f}]")
                
                # Restore without guidance (prior only)
                logger.info("  Restoring with prior only (κ=0)...")
                restored_no_guidance = self.restore_image(
                    clean=clean,
                    noisy=noisy,
                    domain=domain,
                    guidance_weight=0.0,  # No guidance
                )
                
                # Restore with physics-aware guidance
                logger.info("  Restoring with physics-aware guidance (κ=1.0)...")
                restored_with_guidance = self.restore_image(
                    clean=clean,
                    noisy=noisy,
                    domain=domain,
                    guidance_weight=1.0,  # Full guidance
                )
                
                # Create comparison figure
                output_file = output_path / f"{domain}_comparison.png"
                self.create_comparison_figure(
                    domain=domain,
                    clean=clean,
                    noisy=noisy,
                    restored_no_guidance=restored_no_guidance,
                    restored_with_guidance=restored_with_guidance,
                    output_path=str(output_file),
                )
                
                logger.info(f"  ✅ {domain} demo complete")
                
            except Exception as e:
                logger.error(f"  ❌ {domain} demo failed: {e}")
        
        logger.info("=" * 80)
        logger.info("VISUAL DEMO COMPLETE!")
        logger.info(f"Results saved to: {output_path}")
        logger.info("=" * 80)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Visual Demo: Unified Model Performance")
    parser.add_argument("--model_path", type=str, default="~/checkpoint_step_0090000.pth")
    parser.add_argument("--data_root", type=str, default="~/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed/posterior")
    parser.add_argument("--domains", nargs="+", default=["photography", "microscopy"], 
                       choices=["photography", "microscopy", "astronomy"])
    parser.add_argument("--output_dir", type=str, default="visual_demo_results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Expand paths
    args.model_path = str(Path(args.model_path).expanduser())
    args.data_root = str(Path(args.data_root).expanduser())
    
    # Initialize evaluator
    evaluator = VisualDemoEvaluator(
        model_path=args.model_path,
        data_root=args.data_root,
        device=args.device,
        seed=args.seed,
    )
    
    # Run visual demo
    evaluator.run_visual_demo(
        domains=args.domains,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
