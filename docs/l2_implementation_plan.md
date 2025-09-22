# L2-Guided Diffusion Implementation Plan

## Executive Summary

This document provides a detailed implementation plan for creating the L2-Guided Diffusion baseline - the **most critical baseline** for our academic paper. This baseline will share all infrastructure with our Poisson-Gaussian method except the guidance computation, providing a perfect ablation study that isolates the contribution of our physics-aware approach.

## Implementation Strategy

### **Phase 1: Core L2 Guidance System (Day 1)**

#### **1.1 Create L2Guidance Class**
**File**: `core/l2_guidance.py`

```python
#!/usr/bin/env python3
"""
L2 (MSE) likelihood guidance for diffusion sampling baseline.

This module implements simple L2 guidance as a baseline to compare against
our Poisson-Gaussian approach. It assumes uniform Gaussian noise and uses
simple MSE-based likelihood gradients.
"""

import logging
from typing import Optional, Dict, Any
import torch
from .interfaces import GuidanceComputer
from .guidance_config import GuidanceConfig
from .exceptions import GuidanceError

logger = logging.getLogger(__name__)

class L2Guidance(GuidanceComputer):
    """
    L2 (MSE) likelihood guidance for diffusion sampling.

    This baseline assumes y ~ N(s·x + b, σ²) with uniform noise variance
    and computes simple MSE gradients for comparison with Poisson-Gaussian.
    """

    def __init__(
        self,
        scale: float,
        background: float = 0.0,
        noise_variance: float = 1.0,
        config: Optional[GuidanceConfig] = None,
    ):
        """Initialize L2 guidance with same interface as PoissonGuidance."""
        self.scale = scale
        self.background = background
        self.noise_variance = noise_variance
        self.config = config or GuidanceConfig()

        # Validate parameters
        if scale <= 0:
            raise GuidanceError(f"Scale must be positive, got {scale}")
        if noise_variance <= 0:
            raise GuidanceError(f"Noise variance must be positive, got {noise_variance}")

        logger.info(
            f"L2Guidance initialized: scale={scale:.1f} e⁻, "
            f"noise_var={noise_variance:.1f} e⁻²"
        )

    def compute_score(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute L2 likelihood score: ∇ log p(y|x) = s·(y - prediction) / σ².

        Args:
            x_hat: Current estimate [B, C, H, W] (normalized [0,1])
            y_observed: Observed data [B, C, H, W] (electrons)
            mask: Valid pixel mask [B, C, H, W] (optional)

        Returns:
            Likelihood score [B, C, H, W]
        """
        # Convert prediction to electron space
        prediction_electrons = self.scale * x_hat + self.background

        # Compute residual
        residual = y_observed - prediction_electrons

        # L2 gradient: scale * residual / noise_variance
        score = (self.scale / self.noise_variance) * residual

        # Apply mask if provided
        if mask is not None:
            score = score * mask

        # Clamp for numerical stability (same as Poisson guidance)
        if self.config.gradient_clip > 0:
            score = torch.clamp(
                score, -self.config.gradient_clip, self.config.gradient_clip
            )

        return score

    def gamma_schedule(self, sigma: torch.Tensor) -> torch.Tensor:
        """Use identical scheduling to Poisson-Gaussian for fair comparison."""
        if self.config.gamma_schedule == "sigma2":
            gamma = self.config.kappa * sigma.square()
        elif self.config.gamma_schedule == "linear":
            gamma = self.config.kappa * sigma
        elif self.config.gamma_schedule == "const":
            gamma = torch.full_like(sigma, self.config.kappa)
        else:
            raise GuidanceError(f"Unknown gamma schedule: {self.config.gamma_schedule}")

        return gamma

    def compute(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        sigma_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute scaled guidance gradient (shared implementation)."""
        # Compute base score
        score = self.compute_score(x_hat, y_observed, mask, **kwargs)

        # Compute guidance weight
        gamma = self.gamma_schedule(sigma_t)

        # Ensure gamma has correct shape for broadcasting
        if gamma.dim() == 1:  # [B] -> [B, 1, 1, 1]
            gamma = gamma.view(-1, 1, 1, 1)
        elif gamma.dim() == 0:  # scalar -> scalar
            pass
        else:
            raise GuidanceError(f"Unexpected gamma shape: {gamma.shape}")

        # Scale score by guidance weight
        guidance = score * gamma

        return guidance
```

#### **1.2 Create Guidance Factory System**
**File**: `core/guidance_factory.py`

```python
#!/usr/bin/env python3
"""
Factory system for creating different guidance computers.

This module provides a unified interface for creating Poisson-Gaussian
and L2 guidance computers, enabling easy switching between methods.
"""

from typing import Literal, Optional
from .interfaces import GuidanceComputer
from .poisson_guidance import PoissonGuidance
from .l2_guidance import L2Guidance
from .guidance_config import GuidanceConfig

GuidanceType = Literal["poisson", "l2"]

def create_guidance(
    guidance_type: GuidanceType,
    scale: float,
    background: float = 0.0,
    read_noise: float = 0.0,
    config: Optional[GuidanceConfig] = None,
) -> GuidanceComputer:
    """
    Factory function to create guidance computers.

    Args:
        guidance_type: Type of guidance ("poisson" or "l2")
        scale: Dataset normalization scale (electrons)
        background: Background offset (electrons)
        read_noise: Read noise standard deviation (electrons)
        config: Guidance configuration

    Returns:
        Configured guidance computer
    """
    if guidance_type == "poisson":
        return PoissonGuidance(
            scale=scale,
            background=background,
            read_noise=read_noise,
            config=config
        )
    elif guidance_type == "l2":
        # For L2, use read_noise as uniform noise standard deviation
        noise_variance = read_noise**2 if read_noise > 0 else 1.0
        return L2Guidance(
            scale=scale,
            background=background,
            noise_variance=noise_variance,
            config=config
        )
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")

def create_guidance_from_config(config_dict: dict) -> GuidanceComputer:
    """Create guidance from configuration dictionary."""
    guidance_config = GuidanceConfig(**config_dict.get("guidance", {}))

    return create_guidance(
        guidance_type=config_dict["guidance"]["type"],
        scale=config_dict["data"]["scale"],
        background=config_dict["data"].get("background", 0.0),
        read_noise=config_dict["data"].get("read_noise", 0.0),
        config=guidance_config
    )
```

### **Phase 2: Training Integration (Day 2)**

#### **2.1 Enhanced Training Configuration**
**File**: `configs/l2_baseline_photography.yaml`

```yaml
# L2 Baseline Training Configuration
# Identical to main config except guidance type

model:
  architecture: "edm"
  conditioning: true
  condition_dim: 6
  model_channels: 256
  # ... identical to main config

guidance:
  type: "l2"  # Only difference from main config
  mode: "wls"  # Keep same for consistency
  gamma_schedule: "sigma2"  # Identical scheduling
  kappa: 0.5  # Identical strength
  gradient_clip: 100.0  # Identical stability

training:
  # Identical training setup
  epochs: 100
  batch_size: 16
  learning_rate: 1e-4
  experiment_name: "l2_baseline_photography"
  deterministic: true
  seed: 42  # Same seed for fair comparison

data:
  # Identical data configuration
  domains: ["photography"]
  scale: 1000.0
  background: 100.0
  read_noise: 10.0
  # ... rest identical

evaluation:
  # Identical evaluation setup
  metrics: ["psnr", "ssim", "lpips", "chi2_consistency"]
  save_results: true
```

#### **2.2 Unified Training Script**
**File**: `scripts/train_with_guidance_type.py`

```python
#!/usr/bin/env python3
"""
Unified training script supporting both Poisson and L2 guidance.

This script allows training identical models with different guidance types
for perfect ablation studies.
"""

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.guidance_factory import create_guidance_from_config
from poisson_training.multi_domain_trainer import MultiDomainTrainer
from poisson_training.utils import load_config, set_deterministic_mode

def main():
    parser = argparse.ArgumentParser(description="Train model with specified guidance type")
    parser.add_argument("--config", required=True, help="Training configuration file")
    parser.add_argument("--guidance-type", choices=["poisson", "l2"],
                       help="Override guidance type from config")
    parser.add_argument("--experiment-suffix", default="",
                       help="Suffix to add to experiment name")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override guidance type if specified
    if args.guidance_type:
        config.guidance.type = args.guidance_type

    # Add suffix to experiment name for identification
    if args.experiment_suffix:
        config.training.experiment_name += f"_{args.experiment_suffix}"

    # Set deterministic mode for reproducibility
    if config.training.get("deterministic", False):
        set_deterministic_mode(config.training.get("seed", 42))

    # Create guidance computer
    guidance = create_guidance_from_config(config)

    # Create trainer with guidance
    trainer = MultiDomainTrainer(config, guidance=guidance)

    print(f"Starting training with {config.guidance.type} guidance")
    print(f"Experiment: {config.training.experiment_name}")
    print(f"Seed: {config.training.get('seed', 42)}")

    # Train model
    trainer.train()

    print(f"Training completed: {config.training.experiment_name}")

if __name__ == "__main__":
    main()
```

### **Phase 3: Evaluation Integration (Day 3)**

#### **3.1 Enhanced Baseline Framework**
**File**: `core/baselines.py` (additions)

```python
# Add to existing baselines.py

class UnifiedDiffusionBaseline(BaselineMethod):
    """
    Unified diffusion baseline supporting both Poisson and L2 guidance.

    This baseline allows direct comparison between guidance methods using
    identical model architectures and sampling procedures.
    """

    def __init__(
        self,
        model_path: str,
        guidance_type: Literal["poisson", "l2"],
        device: str = "cuda"
    ):
        self.model_path = model_path
        self.guidance_type = guidance_type
        name = f"{guidance_type.upper()}-Guided-Diffusion"
        super().__init__(name, device)

    def _check_availability(self) -> bool:
        """Check if model checkpoint exists."""
        return Path(self.model_path).exists()

    def _load_model(self):
        """Load the trained diffusion model."""
        from models.edm_wrapper import load_pretrained_edm
        model = load_pretrained_edm(self.model_path, device=self.device)
        return model

    def denoise(
        self,
        noisy: torch.Tensor,
        scale: float,
        background: float = 0.0,
        read_noise: float = 0.0,
        steps: int = 18,
        guidance_weight: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Denoise using unified diffusion with specified guidance type.

        Args:
            noisy: Noisy image [B, C, H, W] (electrons)
            scale: Normalization scale (electrons)
            background: Background offset (electrons)
            read_noise: Read noise standard deviation (electrons)
            steps: Number of diffusion steps
            guidance_weight: Guidance strength

        Returns:
            Denoised image [B, C, H, W] (normalized [0,1])
        """
        if not self.is_available:
            raise RuntimeError(f"{self.name} model not available")

        # Create appropriate guidance
        from core.guidance_factory import create_guidance
        from core.guidance_config import GuidanceConfig

        guidance_config = GuidanceConfig(kappa=guidance_weight)
        guidance = create_guidance(
            guidance_type=self.guidance_type,
            scale=scale,
            background=background,
            read_noise=read_noise,
            config=guidance_config
        )

        # Create sampler
        from core.edm_sampler import EDMPosteriorSampler
        sampler = EDMPosteriorSampler(self.model, guidance, device=self.device)

        # Convert to normalized space for model
        noisy_norm = (noisy - background) / scale
        noisy_norm = torch.clamp(noisy_norm, 0, 1)

        # Create metadata (simplified)
        from core.transforms import ImageMetadata
        metadata = ImageMetadata(
            original_height=noisy.shape[-2],
            original_width=noisy.shape[-1],
            scale_factor=1.0,
            pixel_size=1.0,
            pixel_unit="pixel",
            domain="unknown"
        )

        # Sample
        result, info = sampler.sample(
            y_observed=noisy,
            metadata=metadata,
            steps=steps,
            guidance_weight=guidance_weight
        )

        return torch.clamp(result, 0, 1)

    def get_parameters(self) -> Dict[str, Any]:
        """Get baseline parameters."""
        return {
            "guidance_type": self.guidance_type,
            "model_type": "EDM",
            "model_path": str(self.model_path)
        }

# Update BaselineComparator to include unified baselines
def _initialize_baselines(self) -> Dict[str, BaselineMethod]:
    """Initialize all baseline methods."""
    baselines = {}

    # ... existing baselines ...

    # Add unified diffusion baselines if models available
    poisson_model_path = "checkpoints/poisson_guided_model.pth"
    l2_model_path = "checkpoints/l2_guided_model.pth"

    if Path(poisson_model_path).exists():
        baselines["Poisson-Guided-Diffusion"] = UnifiedDiffusionBaseline(
            poisson_model_path, "poisson", device=self.device
        )

    if Path(l2_model_path).exists():
        baselines["L2-Guided-Diffusion"] = UnifiedDiffusionBaseline(
            l2_model_path, "l2", device=self.device
        )

    return baselines
```

#### **3.2 Comparative Evaluation Script**
**File**: `scripts/compare_guidance_methods.py`

```python
#!/usr/bin/env python3
"""
Compare Poisson vs L2 guidance methods for ablation study.

This script runs comparative evaluation between guidance methods
using identical evaluation protocols.
"""

import argparse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.baselines import BaselineComparator
from core.metrics import EvaluationSuite
from scripts.evaluate_baselines import BaselineEvaluationFramework

def main():
    parser = argparse.ArgumentParser(description="Compare guidance methods")
    parser.add_argument("--poisson-model", required=True,
                       help="Path to Poisson-guided model checkpoint")
    parser.add_argument("--l2-model", required=True,
                       help="Path to L2-guided model checkpoint")
    parser.add_argument("--test-data", required=True,
                       help="Path to test data directory")
    parser.add_argument("--output-dir", default="guidance_comparison",
                       help="Output directory for results")
    parser.add_argument("--domains", nargs="+",
                       default=["photography", "microscopy", "astronomy"],
                       help="Domains to evaluate")
    args = parser.parse_args()

    # Create evaluation framework
    evaluator = BaselineEvaluationFramework(
        output_dir=args.output_dir,
        device="cuda"
    )

    # Add unified diffusion baselines
    from core.baselines import UnifiedDiffusionBaseline

    poisson_baseline = UnifiedDiffusionBaseline(
        args.poisson_model, "poisson", device="cuda"
    )
    l2_baseline = UnifiedDiffusionBaseline(
        args.l2_model, "l2", device="cuda"
    )

    evaluator.baseline_comparator.add_baseline("Poisson-Guidance", poisson_baseline)
    evaluator.baseline_comparator.add_baseline("L2-Guidance", l2_baseline)

    # Load test data
    test_data = evaluator.load_real_data(args.test_data)

    # Run evaluation
    results = evaluator.evaluate_baselines(test_data)

    # Generate comparison report
    comparison_report = {
        "guidance_comparison": {},
        "statistical_analysis": {},
        "physics_validation": {}
    }

    for domain in args.domains:
        if domain in results:
            domain_results = results[domain]

            # Extract results for both methods
            poisson_results = domain_results.get("Poisson-Guidance", {})
            l2_results = domain_results.get("L2-Guidance", {})

            comparison_report["guidance_comparison"][domain] = {
                "poisson": {
                    "psnr": poisson_results.get("psnr", {}).get("value", 0),
                    "ssim": poisson_results.get("ssim", {}).get("value", 0),
                    "chi2": poisson_results.get("chi2_consistency", {}).get("value", 0),
                },
                "l2": {
                    "psnr": l2_results.get("psnr", {}).get("value", 0),
                    "ssim": l2_results.get("ssim", {}).get("value", 0),
                    "chi2": l2_results.get("chi2_consistency", {}).get("value", 0),
                },
                "improvement": {
                    "psnr_db": poisson_results.get("psnr", {}).get("value", 0) -
                              l2_results.get("psnr", {}).get("value", 0),
                    "ssim": poisson_results.get("ssim", {}).get("value", 0) -
                           l2_results.get("ssim", {}).get("value", 0),
                }
            }

    # Save comparison report
    output_file = Path(args.output_dir) / "guidance_comparison_report.json"
    with open(output_file, "w") as f:
        json.dump(comparison_report, f, indent=2)

    print(f"Guidance comparison completed!")
    print(f"Results saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("GUIDANCE COMPARISON SUMMARY")
    print("="*60)

    for domain, results in comparison_report["guidance_comparison"].items():
        print(f"\n{domain.upper()}:")
        print(f"  Poisson PSNR: {results['poisson']['psnr']:.2f} dB")
        print(f"  L2 PSNR:      {results['l2']['psnr']:.2f} dB")
        print(f"  Improvement:  {results['improvement']['psnr_db']:.2f} dB")
        print(f"  Poisson χ²:   {results['poisson']['chi2']:.3f}")
        print(f"  L2 χ²:        {results['l2']['chi2']:.3f}")

if __name__ == "__main__":
    main()
```

## Implementation Timeline

### **Day 1: Core Implementation**
- [ ] Create `core/l2_guidance.py` with L2Guidance class
- [ ] Create `core/guidance_factory.py` with factory system
- [ ] Update `core/interfaces.py` if needed
- [ ] Write unit tests for L2 guidance

### **Day 2: Training Integration**
- [ ] Create L2 baseline configuration files
- [ ] Create unified training script
- [ ] Test training pipeline with both guidance types
- [ ] Verify identical model architectures

### **Day 3: Evaluation Integration**
- [ ] Update `core/baselines.py` with UnifiedDiffusionBaseline
- [ ] Create comparative evaluation script
- [ ] Test evaluation pipeline
- [ ] Generate sample comparison reports

### **Day 4: Validation and Testing**
- [ ] Run synthetic data validation
- [ ] Verify numerical stability
- [ ] Test edge cases and error handling
- [ ] Performance profiling

## Expected Results

### **Physics Validation (Key Scientific Result)**
```
Method           | χ² Consistency | Bias (%) | Residual Structure
-----------------|----------------|----------|-------------------
Poisson-Guidance | 1.02 ± 0.03   | 0.8 ± 0.2| None (white noise)
L2-Guidance      | 1.67 ± 0.08   | 3.2 ± 0.5| Structured residuals
```

### **Performance Comparison**
```
Photon Level | Poisson PSNR | L2 PSNR | Improvement
-------------|--------------|---------|------------
< 10         | 32.4 ± 0.3   | 28.9 ± 0.4 | +3.5 dB
< 100        | 33.1 ± 0.3   | 30.2 ± 0.4 | +2.9 dB
> 1000       | 34.2 ± 0.3   | 33.8 ± 0.3 | +0.4 dB
```

## Success Criteria

### **Technical Success**
- [ ] L2 guidance produces reasonable denoising results
- [ ] Identical model architectures confirmed
- [ ] Fair comparison protocols validated
- [ ] Statistical significance achieved (p < 0.05)

### **Scientific Success**
- [ ] Poisson guidance shows χ² ≈ 1.0, L2 shows χ² > 1.3
- [ ] Clear performance advantage in low-photon regime
- [ ] Minimal difference in high-photon regime
- [ ] Structured residuals in L2, white noise in Poisson

### **Implementation Success**
- [ ] Clean, maintainable code architecture
- [ ] Comprehensive test coverage
- [ ] Clear documentation and examples
- [ ] Ready for academic paper submission

## Risk Mitigation

### **Potential Issues**
1. **L2 guidance too weak**: May need hyperparameter tuning
2. **Numerical instability**: Different stability profile than Poisson
3. **Convergence issues**: Different optimization landscape

### **Mitigation Strategies**
1. **Hyperparameter sweep**: Find optimal L2 parameters
2. **Stability monitoring**: Add diagnostics and safeguards
3. **Adaptive scheduling**: Method-specific parameter adjustment

## Conclusion

This L2-Guided Diffusion baseline is **critical** for our academic paper. It provides the perfect ablation study that isolates the contribution of our physics-aware approach. The implementation strategy ensures scientific rigor while maintaining code quality and maintainability.

**Next Steps**:
1. Begin implementation with `core/l2_guidance.py`
2. Create factory system for guidance selection
3. Integrate into training and evaluation pipelines
4. Run comparative experiments

This baseline will be the **key result** that demonstrates the scientific value of our Poisson-Gaussian physics modeling to conference reviewers.
