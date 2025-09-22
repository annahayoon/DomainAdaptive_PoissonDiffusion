# L2-Guided Diffusion Baseline: Design Document

## Overview

This document outlines the design for implementing L2-Guided Diffusion as a critical baseline for isolating the physics contribution of our Poisson-Gaussian approach. The L2 baseline will **share everything** with our method except the guidance mechanism, providing a perfect ablation study.

## Design Principles

### **1. Maximum Code Reuse**
- **Same model architecture**: Identical EDM wrapper with domain conditioning
- **Same training pipeline**: Identical datasets, preprocessing, and training loop
- **Same evaluation framework**: Identical metrics and statistical analysis
- **Only difference**: L2 guidance vs Poisson-Gaussian guidance

### **2. Clean Architecture Separation**
- **Guidance Interface**: Abstract base class allows swapping guidance methods
- **Configuration-Driven**: Single config parameter switches between guidance types
- **Unified Sampling**: Same sampler works with both guidance methods
- **Consistent API**: Same function signatures and return types

### **3. Scientific Rigor**
- **Identical Conditions**: Same random seeds, hyperparameters, and data splits
- **Fair Comparison**: Same computational budget and optimization settings
- **Statistical Validity**: Same evaluation protocols and significance testing

## Architecture Design

### **Current System (Poisson-Gaussian)**
```
Sampling Loop:
x_t → Model(x_t, σ_t, condition) → x_pred
     ↓
y_obs → PoissonGuidance(x_pred, y_obs, scale, σ_t) → guidance_grad
     ↓
x_pred + guidance_grad → x_{t-1}
```

### **Proposed System (L2 Guidance)**
```
Sampling Loop:
x_t → Model(x_t, σ_t, condition) → x_pred
     ↓
y_obs → L2Guidance(x_pred, y_obs, scale, σ_t) → guidance_grad
     ↓
x_pred + guidance_grad → x_{t-1}
```

**Key Insight**: Only the guidance computation changes!

## Implementation Strategy

### **1. Guidance Interface Abstraction**

#### **Enhanced GuidanceComputer Interface**
```python
# File: core/interfaces.py
class GuidanceComputer(ABC):
    """Abstract base class for likelihood guidance computation."""

    @abstractmethod
    def compute_score(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute likelihood score ∇ log p(y|x)."""
        pass

    @abstractmethod
    def gamma_schedule(self, sigma: torch.Tensor) -> torch.Tensor:
        """Compute guidance weight γ(σ)."""
        pass

    def compute(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        sigma_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute scaled guidance gradient (shared implementation)."""
        score = self.compute_score(x_hat, y_observed, mask, **kwargs)
        gamma = self.gamma_schedule(sigma_t)

        # Broadcast gamma to correct shape
        if gamma.dim() == 1:
            gamma = gamma.view(-1, 1, 1, 1)

        return score * gamma
```

### **2. L2 Guidance Implementation**

#### **Core L2 Guidance Class**
```python
# File: core/l2_guidance.py
class L2Guidance(GuidanceComputer):
    """
    L2 (MSE) likelihood guidance for diffusion sampling.

    This class implements simple L2 guidance that treats the observation
    as having Gaussian noise with uniform variance. It serves as a baseline
    to isolate the contribution of correct Poisson-Gaussian physics.

    Mathematical Foundation:
    - Assumes: y ~ N(s·x + b, σ²)
    - Gradient: ∇ log p(y|x) = s·(y - s·x - b) / σ²
    - Scheduling: γ(σ) = κ·σ² (same as Poisson-Gaussian)
    """

    def __init__(
        self,
        scale: float,
        background: float = 0.0,
        noise_variance: float = 1.0,
        config: Optional[GuidanceConfig] = None,
    ):
        """
        Initialize L2 guidance.

        Args:
            scale: Dataset normalization scale (electrons)
            background: Background offset (electrons)
            noise_variance: Assumed uniform noise variance (electrons²)
            config: Guidance configuration (same as Poisson-Gaussian)
        """
        self.scale = scale
        self.background = background
        self.noise_variance = noise_variance
        self.config = config or GuidanceConfig()

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
        Compute L2 likelihood score.

        L2 gradient: ∇ log p(y|x) = scale * (y - prediction) / noise_variance
        where prediction = scale * x + background
        """
        # Convert to electron space
        prediction_electrons = self.scale * x_hat + self.background

        # Compute residual
        residual = y_observed - prediction_electrons

        # L2 gradient (simple MSE derivative)
        score = (self.scale / self.noise_variance) * residual

        # Apply mask if provided
        if mask is not None:
            score = score * mask

        # Clamp for numerical stability
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
            raise ValueError(f"Unknown gamma schedule: {self.config.gamma_schedule}")

        return gamma
```

### **3. Unified Guidance Factory**

#### **Guidance Creation System**
```python
# File: core/guidance_factory.py
from typing import Literal, Union
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
```

### **4. Configuration Integration**

#### **Enhanced Training Configuration**
```yaml
# File: configs/l2_baseline_training.yaml
model:
  architecture: "edm"
  conditioning: true
  condition_dim: 6
  # ... same as main config

guidance:
  type: "l2"  # "poisson" or "l2"
  mode: "wls"  # Keep same for consistency
  gamma_schedule: "sigma2"  # Keep same for fair comparison
  kappa: 0.5  # Keep same for fair comparison
  gradient_clip: 100.0  # Keep same for fair comparison

training:
  # ... identical to main training config
  experiment_name: "l2_baseline_photography"

data:
  # ... identical to main data config

evaluation:
  # ... identical to main evaluation config
```

### **5. Sampling Integration**

#### **Enhanced Sampler**
```python
# File: core/edm_sampler.py (enhanced)
class EDMPosteriorSampler:
    """Enhanced sampler supporting multiple guidance types."""

    def __init__(
        self,
        model: EDMModelWrapper,
        guidance: GuidanceComputer,  # Now accepts any guidance type
        device: str = "cuda",
        **kwargs
    ):
        self.model = model
        self.guidance = guidance  # Polymorphic guidance
        self.device = device
        # ... rest unchanged

    def sample(
        self,
        y_observed: torch.Tensor,
        metadata: ImageMetadata,
        condition: Optional[torch.Tensor] = None,
        steps: int = 18,
        guidance_weight: float = 1.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Sample using any guidance type (same interface)."""
        # ... sampling loop unchanged
        # guidance.compute() works for both Poisson and L2
```

## Training Pipeline Integration

### **1. Unified Training Script**
```python
# File: scripts/train_with_guidance.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guidance-type", choices=["poisson", "l2"],
                       default="poisson")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config.guidance.type = args.guidance_type

    # Create guidance
    guidance = create_guidance(
        guidance_type=config.guidance.type,
        scale=config.data.scale,
        background=config.data.background,
        read_noise=config.data.read_noise,
        config=GuidanceConfig(**config.guidance)
    )

    # Create trainer (same for both)
    trainer = MultiDomainTrainer(config, guidance=guidance)

    # Train (same for both)
    trainer.train()

# Usage:
# python train_with_guidance.py --guidance-type poisson --config configs/main.yaml
# python train_with_guidance.py --guidance-type l2 --config configs/l2_baseline.yaml
```

### **2. Evaluation Integration**
```python
# File: core/baselines.py (enhanced)
class UnifiedDiffusionBaseline(BaselineMethod):
    """Unified baseline supporting both Poisson and L2 guidance."""

    def __init__(
        self,
        model_path: str,
        guidance_type: Literal["poisson", "l2"],
        device: str = "cuda"
    ):
        self.guidance_type = guidance_type
        name = f"{guidance_type.upper()}-Guided-Diffusion"
        super().__init__(name, device)

    def denoise(self, noisy, scale, background=0.0, read_noise=0.0, **kwargs):
        # Create appropriate guidance
        guidance = create_guidance(
            guidance_type=self.guidance_type,
            scale=scale,
            background=background,
            read_noise=read_noise
        )

        # Same sampling process for both
        sampler = EDMPosteriorSampler(self.model, guidance)
        return sampler.sample(noisy, **kwargs)
```

## Expected Results and Analysis

### **Scientific Hypothesis**
If our physics modeling is correct, we expect:
1. **Poisson guidance > L2 guidance** in low-photon regime (<100 photons)
2. **Similar performance** in high-photon regime (>1000 photons)
3. **χ² ≈ 1.0** for Poisson, **χ² > 1.3** for L2 in all regimes

### **Expected Performance Table**
```
Photon Level | Method           | PSNR↑  | χ²     | Bias(%)
-------------|------------------|--------|--------|--------
< 10         | Poisson-Guidance | 32.4   | 1.02   | 0.8
< 10         | L2-Guidance      | 28.9   | 1.67   | 3.2
< 100        | Poisson-Guidance | 33.1   | 1.01   | 0.5
< 100        | L2-Guidance      | 30.2   | 1.34   | 2.1
> 1000       | Poisson-Guidance | 34.2   | 1.00   | 0.2
> 1000       | L2-Guidance      | 33.8   | 1.15   | 0.4
```

## Implementation Benefits

### **1. Perfect Ablation Study**
- **Identical everything** except guidance computation
- **Same random seeds** for reproducible comparison
- **Same computational cost** for fair evaluation

### **2. Code Maintainability**
- **Single codebase** handles both methods
- **Shared infrastructure** reduces duplication
- **Configuration-driven** switching between methods

### **3. Scientific Rigor**
- **Controlled experiment** isolating physics contribution
- **Statistical validity** through identical evaluation protocols
- **Peer review acceptance** through transparent comparison

## Risk Mitigation

### **Potential Issues**
1. **L2 guidance too weak**: May need different hyperparameters
2. **Numerical instability**: L2 may have different stability profile
3. **Convergence differences**: Different guidance may need different schedules

### **Mitigation Strategies**
1. **Hyperparameter sweep**: Find optimal L2 parameters separately
2. **Stability monitoring**: Add numerical diagnostics to both methods
3. **Adaptive scheduling**: Allow method-specific scheduling if needed

## Conclusion

This design provides a **scientifically rigorous** and **implementation-efficient** approach to creating the L2-Guided Diffusion baseline. By sharing all infrastructure except the guidance computation, we ensure a fair comparison that isolates the contribution of our physics-aware approach.

**Next Steps**:
1. Implement L2Guidance class
2. Create guidance factory system
3. Update training and evaluation pipelines
4. Run comparative experiments

This baseline will be **critical** for demonstrating the scientific value of our Poisson-Gaussian physics modeling to conference reviewers.
