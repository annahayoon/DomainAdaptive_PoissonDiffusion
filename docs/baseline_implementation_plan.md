# Baseline Implementation Plan for Academic Paper Submission

## Executive Summary

This document outlines our comprehensive baseline implementation strategy for submitting our Domain-Adaptive Poisson-Gaussian Diffusion paper to top-tier ML conferences (ICLR/NeurIPS). We analyze our current implementation status, evaluation framework, and provide a detailed roadmap for completing the baseline comparisons needed for a strong academic submission.

## Current Implementation Status ✅

### **Core Framework (Complete)**
- ✅ **Physics-Aware Guidance**: Exact Poisson-Gaussian likelihood implementation
- ✅ **Multi-Domain Architecture**: Unified model with domain conditioning (6D vectors)
- ✅ **Multi-Resolution System**: Progressive growing (32→64→96→128) and hierarchical processing
- ✅ **Reversible Transforms**: Perfect reconstruction with metadata preservation
- ✅ **Training Infrastructure**: Complete deterministic training framework
- ✅ **EDM Integration**: External EDM codebase successfully integrated

### **Evaluation Framework (Excellent)**
- ✅ **Standard Metrics**: PSNR, SSIM, LPIPS, MS-SSIM with confidence intervals
- ✅ **Physics Metrics**: χ² consistency, residual whiteness, bias analysis
- ✅ **Domain-Specific Metrics**: Counting accuracy (microscopy), photometry (astronomy)
- ✅ **Statistical Analysis**: Comprehensive comparison framework with JSON serialization
- ✅ **Visualization Tools**: Automated plot generation and academic reporting

### **Current Baseline Implementation Status**

#### **✅ Implemented and Working**
1. **BM3D**: Standard Gaussian noise assumption
2. **Anscombe + BM3D**: Correct Poisson-Gaussian physics (our key physics competitor)
3. **Richardson-Lucy**: Iterative Poisson deconvolution
4. **Gaussian Filter**: Simple baseline
5. **Wiener Filter**: Optimal linear filtering
6. **L2-Guided Diffusion**: Same architecture as ours but with L2 guidance (critical ablation)
7. **Noise2Void**: Self-supervised approach with neighbor averaging

#### **⚠️ Partially Implemented (Built-in Versions)**
8. **DnCNN**: Simple built-in implementation, needs pretrained weights
9. **NAFNet**: Simplified architecture, needs pretrained weights

#### **❌ Missing Critical Baselines**
10. **Restormer**: Current SOTA transformer-based restoration
11. **SwinIR**: Strong transformer baseline
12. **Pretrained Diffusion Models**: HuggingFace integration needed

## What We Need to Implement Next

### **Priority 1: Essential for Paper Acceptance** (Must Have)

#### **1. Restormer Implementation** ⭐ **HIGHEST PRIORITY**
- **Why Critical**: Current SOTA that reviewers expect us to beat
- **Implementation Strategy**: Use official pretrained weights
- **Timeline**: 1-2 days
- **Action Items**:
  ```bash
  # Install Restormer
  git clone https://github.com/swz30/Restormer.git external/restormer
  pip install timm einops

  # Integrate into our baseline framework
  # File: core/baselines.py - add RestormerBaseline class
  ```

#### **2. Strengthen Deep Learning Baselines**
- **DnCNN with Pretrained Weights**: Download from official sources
- **NAFNet with Pretrained Weights**: Use official checkpoints
- **Timeline**: 1 day
- **Action Items**:
  ```bash
  # Download pretrained models
  wget https://github.com/cszn/DnCNN/releases/download/v1.0/DnCNN.pth
  wget https://github.com/megvii-research/NAFNet/releases/download/v1.0/NAFNet.pth
  ```

#### **3. Single-Domain Ablation Models** ⭐ **KEY FOR UNIFIED MODEL CLAIMS**
- **Purpose**: Show benefit of unified cross-domain training
- **Implementation**: Train 3 separate models (photography/microscopy/astronomy)
- **Timeline**: 1 week (parallel training)
- **Action Items**:
  ```bash
  # Train domain-specific models
  python train_photography_model.py --single_domain --epochs 100
  python train_microscopy_model.py --single_domain --epochs 100
  python train_astronomy_model.py --single_domain --epochs 100
  ```

### **Priority 2: Strong Supporting Evidence** (Recommended)

#### **4. Pretrained Diffusion Integration**
- **HuggingFace Diffusion Models**: For general comparison
- **Stable Diffusion Inpainting**: Repurposed for denoising
- **Timeline**: 2-3 days
- **Action Items**:
  ```python
  # Add to core/baselines.py
  from diffusers import StableDiffusionInpaintPipeline

  class StableDiffusionBaseline(BaselineMethod):
      def __init__(self):
          self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
              "runwayml/stable-diffusion-inpainting"
          )
  ```

#### **5. SwinIR Implementation**
- **Purpose**: Another strong transformer baseline
- **Implementation**: Use official pretrained weights
- **Timeline**: 1 day
- **Action Items**:
  ```bash
  git clone https://github.com/JingyunLiang/SwinIR.git external/swinir
  # Integrate similar to Restormer
  ```

### **Priority 3: Nice to Have** (If Time Permits)

#### **6. Advanced Self-Supervised Methods**
- **Noise2Clean**: If paired data available
- **Self2Self**: More sophisticated self-supervised approach
- **Timeline**: 2-3 days each

#### **7. Domain-Specific SOTA Methods**
- **Photography**: Low-light specific methods (SID, KinD++)
- **Microscopy**: Specialized denoising methods
- **Astronomy**: Astronomical image processing tools

## Implementation Timeline

### **Week 1: Critical Baselines**
- **Days 1-2**: Implement Restormer baseline
- **Days 3-4**: Strengthen DnCNN/NAFNet with pretrained weights
- **Days 5-7**: Begin single-domain model training

### **Week 2: Supporting Baselines**
- **Days 1-3**: Pretrained diffusion integration
- **Days 4-5**: SwinIR implementation
- **Days 6-7**: Complete single-domain training

### **Week 3: Evaluation and Analysis**
- **Days 1-3**: Run comprehensive baseline evaluation
- **Days 4-5**: Statistical analysis and significance testing
- **Days 6-7**: Create paper-ready results tables and figures

## Detailed Implementation Guide

### **1. Restormer Integration**

#### **File: `core/baselines.py`**
```python
class RestormerBaseline(DeepLearningBaseline):
    """Restormer transformer-based restoration baseline."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.model_path = model_path or "external/restormer/Denoising/pretrained_models/Restormer.pth"
        super().__init__("Restormer", model_path, device)

    def _check_availability(self) -> bool:
        try:
            import sys
            sys.path.append('external/restormer')
            from Restormer import Restormer
            return Path(self.model_path).exists()
        except ImportError:
            return False

    def _load_model(self) -> nn.Module:
        sys.path.append('external/restormer')
        from Restormer import Restormer

        model = Restormer(
            inp_channels=1,
            out_channels=1,
            dim=48,
            num_blocks=[4,6,6,8],
            num_refinement_blocks=4,
            heads=[1,2,4,8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias'
        )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['params'])
        model.to(self.device)
        model.eval()

        return model
```

### **2. Single-Domain Training Setup**

#### **File: `scripts/train_single_domain.py`**
```python
#!/usr/bin/env python3
"""Train single-domain models for ablation study."""

def train_single_domain_model(domain: str, config_path: str):
    """Train a model on single domain only."""

    # Load base config
    config = load_config(config_path)

    # Modify for single domain
    config.data.domains = [domain]  # Only train on one domain
    config.model.condition_dim = 3  # Reduced conditioning (no domain one-hot)
    config.training.experiment_name = f"single_domain_{domain}"

    # Initialize trainer
    trainer = MultiDomainTrainer(config)

    # Train model
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["photography", "microscopy", "astronomy"])
    parser.add_argument("--config", default="configs/single_domain.yaml")
    args = parser.parse_args()

    train_single_domain_model(args.domain, args.config)
```

### **3. Comprehensive Evaluation Script**

#### **File: `scripts/run_full_baseline_evaluation.py`**
```python
#!/usr/bin/env python3
"""Run comprehensive baseline evaluation for paper submission."""

def main():
    # Initialize evaluation framework
    evaluator = BaselineEvaluationFramework(
        config_path="configs/paper_evaluation.yaml",
        output_dir="paper_results"
    )

    # Load test datasets
    test_data = evaluator.load_test_data()

    # Evaluate all baselines
    results = evaluator.evaluate_all_baselines(test_data)

    # Generate statistical analysis
    stats = evaluator.compute_statistical_significance(results)

    # Create paper-ready tables
    evaluator.generate_paper_tables(results, stats)

    # Create visualizations
    evaluator.create_paper_figures(results)

    print("✅ Paper evaluation complete!")
    print(f"Results saved to: {evaluator.output_dir}")

if __name__ == "__main__":
    main()
```

## Expected Results for Paper

### **Main Results Table**
```
Method                | PSNR↑   | SSIM↑  | LPIPS↓ | χ²     | Time(s)
---------------------|---------|--------|--------|--------|--------
Ours (PG-Guidance)  | 32.4±0.3| 0.89±0.02| 0.12±0.01| 1.02±0.05| 2.3
L2-Guided Diffusion | 29.8±0.4| 0.84±0.03| 0.18±0.02| 1.45±0.08| 2.1
Restormer           | 30.1±0.4| 0.86±0.03| 0.15±0.02| 1.89±0.12| 0.5
Anscombe + BM3D     | 28.2±0.3| 0.81±0.02| 0.22±0.02| 1.08±0.04| 0.8
NAFNet              | 29.5±0.4| 0.85±0.03| 0.16±0.02| 1.76±0.10| 0.4
DnCNN               | 27.5±0.5| 0.79±0.04| 0.25±0.03| 2.34±0.15| 0.3
SwinIR              | 29.8±0.4| 0.85±0.03| 0.17±0.02| 1.82±0.11| 0.6
BM3D                | 25.8±0.4| 0.75±0.03| 0.31±0.03| 3.12±0.18| 1.2
Noise2Void          | 26.2±0.5| 0.76±0.04| 0.28±0.03| 2.89±0.17| 0.7
```

### **Physics Validation (Low Photon Regime)**
```
Photon Level | Method | χ² Consistency | Bias (%) | Residual Power
-------------|--------|----------------|----------|---------------
< 10         | Ours   | 1.01±0.03     | 0.8±0.2  | 0.02±0.01
< 10         | L2     | 1.67±0.08     | 3.2±0.5  | 0.15±0.03
< 10         | Restormer| 2.34±0.12   | 4.1±0.6  | 0.22±0.04
< 10         | Anscombe| 1.05±0.04    | 1.1±0.3  | 0.03±0.01
```

### **Cross-Domain Generalization**
```
Domain      | Single-Domain | Unified (Ours) | Drop (dB)
------------|---------------|----------------|----------
Photography | 32.8±0.3     | 32.4±0.3      | -0.4
Microscopy  | 31.2±0.4     | 30.9±0.4      | -0.3
Astronomy   | 29.6±0.5     | 29.1±0.5      | -0.5
Average     | 31.2         | 30.8          | -0.4
```

## Risk Mitigation

### **High-Risk Items**
1. **Restormer Integration Complexity**:
   - **Mitigation**: Start with simple wrapper, expand if needed
   - **Fallback**: Use simplified transformer architecture

2. **Pretrained Model Compatibility**:
   - **Mitigation**: Test on small samples first
   - **Fallback**: Use built-in implementations with warning

3. **Training Time for Single-Domain Models**:
   - **Mitigation**: Use smaller models or fewer epochs if needed
   - **Fallback**: Theoretical analysis of why unified should work

### **Medium-Risk Items**
1. **HuggingFace Integration**: May require significant adapter code
2. **Statistical Significance**: May need larger test sets for power

## Success Metrics

### **Paper Acceptance Requirements**
- ✅ Beat Restormer on PSNR by >1.5 dB in low-light regime
- ✅ Achieve χ² ≈ 1.0 (within [0.9, 1.1]) vs competitors χ² > 1.3
- ✅ Show <0.5 dB drop for unified vs single-domain models
- ✅ Statistical significance (p < 0.05) for key comparisons

### **Strong Paper Requirements**
- ✅ Beat all baselines across all metrics
- ✅ Demonstrate physics advantage in <100 photon regime
- ✅ Show cross-domain transfer benefits
- ✅ Comprehensive ablation studies

## Conclusion

Our baseline implementation framework is already **more comprehensive than most ML papers**. The key missing piece is **Restormer integration**, which is critical for reviewer acceptance. With the outlined 3-week implementation plan, we'll have a complete baseline suite that strongly supports our scientific claims.

**Next Steps**:
1. **Immediate**: Start Restormer integration (highest priority)
2. **This Week**: Complete critical baselines (Priority 1)
3. **Next Week**: Add supporting baselines (Priority 2)
4. **Week 3**: Comprehensive evaluation and paper preparation

The combination of our physics-correct approach + comprehensive baseline comparison + rigorous evaluation framework positions us well for top-tier conference acceptance.
