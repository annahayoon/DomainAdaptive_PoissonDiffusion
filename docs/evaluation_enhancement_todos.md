# Comprehensive Evaluation Framework Enhancement TODOs

## Overview

This document outlines critical enhancements needed for the evaluation framework to ensure scientific rigor, reproducibility, and compelling results for the ICLR submission. Based on the analysis of current implementation and requirements from proposal/design documents.

**Priority Levels:**
- 🚨 **CRITICAL**: Blocks paper submission, must fix immediately
- 🔥 **HIGH**: Major impact on results quality/credibility
- ⚠️ **MEDIUM**: Important for completeness
- 📝 **LOW**: Nice-to-have improvements

---

## Phase 1: Critical Fixes (Week 1) 🚨

### 1.1 Fix Guidance Space Conversion Issues 🚨
**Problem**: Inconsistent space handling between model ([0,1]) and physics (electrons)

**Files to modify:**
- `core/baselines.py` (L2GuidedDiffusionBaseline.denoise)
- `core/l2_guidance.py` (L2Guidance.compute_score)
- `core/poisson_guidance.py` (PoissonGuidance.compute)
- `core/sampling.py` (EDMPosteriorSampler)

**Tasks:**
- [x] **1.1.1** Create unified guidance interface that handles space conversion
- [x] **1.1.2** Ensure all guidance methods receive electron-space data
- [x] **1.1.3** Ensure guidance returns normalized-space corrections
- [x] **1.1.4** Add validation tests for space conversion correctness
- [x] **1.1.5** Document space conversion protocol clearly

**Implementation Template:**
```python
def unified_guidance_step(x0_norm, noisy_norm, calibration, guidance_type):
    """Unified guidance with proper space handling."""
    scale = calibration['scale']

    # Convert to electron space for physics
    x0_electrons = x0_norm * scale
    y_electrons = noisy_norm * scale

    if guidance_type == "poisson":
        score_electrons = poisson_guidance(x0_electrons, y_electrons, calibration)
    else:  # L2
        score_electrons = l2_guidance(x0_electrons, y_electrons, calibration)

    # Convert back to normalized space
    return score_electrons / scale
```

### 1.2 Implement Fair L2 Baseline Conditioning 🚨
**Problem**: L2 baseline uses physics-aware conditioning, creating unfair comparison

**Files to modify:**
- `core/domain_encoder.py`
- `core/baselines.py`
- `scripts/comprehensive_baseline_evaluation.py`

**Tasks:**
- [x] **1.2.1** Create separate conditioning for DAPGD vs L2
- [x] **1.2.2** DAPGD: `[domain_onehot(3), log_scale(1), rel_noise(1), rel_bg(1)]`
- [x] **1.2.3** L2: `[domain_onehot(3), noise_estimate(1), padding(2)]`
- [ ] **1.2.4** Add conditioning validation in evaluation pipeline
- [ ] **1.2.5** Document conditioning strategy differences

### 1.3 Verify Metrics Use Correct Physical Units 🚨
**Problem**: Some metrics may operate on normalized images instead of electron-space

**Files to check/modify:**
- `core/metrics.py` (PhysicsMetrics)
- `scripts/evaluate_multi_domain_model.py`
- `scripts/comprehensive_baseline_evaluation.py`

**Tasks:**
- [x] **1.3.1** Audit all metric calculations for space consistency
- [x] **1.3.2** Ensure chi-squared uses de-normalized predictions
- [x] **1.3.3** Ensure PSNR/SSIM use appropriate data range
- [x] **1.3.4** Add unit tests for metric space handling
- [x] **1.3.5** Add runtime assertions for metric input validation

---

## Phase 2: Residual Analysis Implementation (Week 1-2) 🔥

### 2.1 Implement Residual Map Computation 🔥
**Purpose**: Critical visual proof that our method produces white noise residuals

**New files to create:**
- `core/residual_analysis.py`
- `visualization/residual_plots.py`

**Tasks:**
- [x] **2.1.1** Implement normalized residual computation: `(y - pred) / sqrt(pred + σ_r²)`
- [x] **2.1.2** Add statistical tests for whiteness (autocorrelation, power spectrum)
- [x] **2.1.3** Implement residual histogram analysis
- [x] **2.1.4** Add residual structure detection (should be minimal for our method)

```python
class ResidualAnalyzer:
    def compute_normalized_residuals(self, pred_electrons, noisy_electrons, read_noise):
        """Compute statistically normalized residuals."""
        variance = pred_electrons + read_noise**2
        residuals = (noisy_electrons - pred_electrons) / torch.sqrt(variance)
        return residuals  # Should be ~ N(0,1) for correct physics

    def test_whiteness(self, residuals):
        """Test if residuals are white noise."""
        # Autocorrelation test
        # Power spectrum flatness test
        # Ljung-Box test
        pass
```

### 2.2 Create 4-Panel Comparison Visualization 🔥
**Purpose**: Paper-ready figures showing method comparison with residuals

**Files to create:**
- `visualization/comparison_plots.py`
- `scripts/generate_paper_figures.py`

**Tasks:**
- [x] **2.2.1** Implement 4-panel layout: [Our Result, L2 Result, Our Residuals, L2 Residuals]
- [x] **2.2.2** Add consistent colormap and intensity scaling
- [x] **2.2.3** Add statistical annotations (chi-squared values, whiteness scores)
- [x] **2.2.4** Generate figures for all three domains
- [x] **2.2.5** Create publication-quality output (300 DPI, proper fonts)

### 2.3 Statistical Residual Validation 🔥
**Purpose**: Quantitative proof our residuals follow expected statistics

**Tasks:**
- [ ] **2.3.1** Implement Kolmogorov-Smirnov test for Gaussian residuals
- [ ] **2.3.2** Add spatial correlation analysis
- [ ] **2.3.3** Implement frequency domain analysis (power spectrum)
- [ ] **2.3.4** Add statistical summary reporting
- [ ] **2.3.5** Create residual validation report generation

---

## Phase 3: Enhanced Baseline Comparison (Week 2) 🔥

### 3.1 Complete L2 Baseline Integration 🔥
**Purpose**: Ensure fair, rigorous ablation study

**Files to modify:**
- `core/baselines.py`
- `scripts/comprehensive_baseline_evaluation.py`
- `configs/baseline_evaluation.yaml`

**Tasks:**
- [ ] **3.1.1** Integrate properly trained L2 baseline model
- [ ] **3.1.2** Implement identical sampling pipeline for both methods
- [ ] **3.1.3** Add deterministic evaluation with fixed seeds
- [ ] **3.1.4** Ensure identical computational budget
- [ ] **3.1.5** Add statistical significance testing

### 3.2 Low-Light Performance Analysis 🔥
**Purpose**: Key selling point - show advantage increases at low photon counts

**New files to create:**
- `analysis/photon_scaling_analysis.py`
- `visualization/scaling_plots.py`

**Tasks:**
- [ ] **3.2.1** Extract photon counts from calibration data
- [ ] **3.2.2** Bin results by photon count ranges
- [ ] **3.2.3** Create performance vs photon count plots
- [ ] **3.2.4** Statistical analysis of scaling relationships
- [ ] **3.2.5** Generate scaling analysis report

```python
def analyze_photon_scaling(results_dict):
    """Analyze how performance scales with photon count."""
    photon_ranges = [(0, 10), (10, 50), (50, 100), (100, 500), (500, float('inf'))]

    for method in ['DAPGD', 'L2-Baseline']:
        for photon_range in photon_ranges:
            # Filter results by photon count
            # Compute average PSNR, chi-squared
            # Statistical significance vs other methods
            pass
```

### 3.3 Cross-Domain Generalization Analysis ⚠️
**Purpose**: Validate unified model claims

**Tasks:**
- [ ] **3.3.1** Implement domain-specific performance breakdown
- [ ] **3.3.2** Cross-domain transfer analysis (train on A, test on B)
- [ ] **3.3.3** Compare unified vs domain-specific models
- [ ] **3.3.4** Statistical analysis of generalization
- [ ] **3.3.5** Generate domain analysis report

---

## Phase 4: Comprehensive Evaluation Pipeline (Week 2-3) ⚠️

### 4.1 Standardized Evaluation Protocol ⚠️
**Purpose**: Reproducible, reviewer-friendly evaluation

**Files to create:**
- `scripts/standardized_evaluation.py`
- `configs/evaluation_protocol.yaml`
- `docs/evaluation_protocol.md`

**Tasks:**
- [ ] **4.1.1** Define standard evaluation splits and procedures
- [ ] **4.1.2** Implement deterministic evaluation with seed control
- [ ] **4.1.3** Add progress tracking and intermediate checkpointing
- [ ] **4.1.4** Create evaluation report templates
- [ ] **4.1.5** Add automatic result validation

### 4.2 Performance Benchmarking ⚠️
**Purpose**: Timing and efficiency analysis

**Files to create:**
- `benchmarks/timing_analysis.py`
- `benchmarks/memory_profiling.py`

**Tasks:**
- [ ] **4.2.1** Implement detailed timing breakdown
- [ ] **4.2.2** Memory usage profiling
- [ ] **4.2.3** GPU utilization analysis
- [ ] **4.2.4** Scalability testing (different image sizes)
- [ ] **4.2.5** Comparison with classical methods speed

### 4.3 Scientific Validation Suite ⚠️
**Purpose**: Physics-based validation beyond standard metrics

**Files to create:**
- `validation/physics_tests.py`
- `validation/synthetic_validation.py`

**Tasks:**
- [ ] **4.3.1** Synthetic data validation with known ground truth
- [ ] **4.3.2** Noise model validation tests
- [ ] **4.3.3** Calibration accuracy assessment
- [ ] **4.3.4** Edge case handling validation
- [ ] **4.3.5** Robustness testing (missing calibration, etc.)

---

## Phase 5: Evaluation Infrastructure (Week 3) 📝

### 5.1 Automated Report Generation 📝
**Purpose**: Streamlined results compilation

**Files to create:**
- `reporting/report_generator.py`
- `reporting/latex_table_generator.py`
- `templates/paper_results.tex`

**Tasks:**
- [ ] **5.1.1** Automated LaTeX table generation
- [ ] **5.1.2** Figure compilation and organization
- [ ] **5.1.3** Statistical summary generation
- [ ] **5.1.4** Result comparison tables
- [ ] **5.1.5** Appendix material generation

### 5.2 Evaluation Dashboard 📝
**Purpose**: Real-time monitoring and visualization

**Files to create:**
- `dashboard/evaluation_dashboard.py`
- `dashboard/live_plotting.py`

**Tasks:**
- [ ] **5.2.1** Web-based results dashboard
- [ ] **5.2.2** Live plotting during evaluation
- [ ] **5.2.3** Interactive result exploration
- [ ] **5.2.4** Comparison visualization tools
- [ ] **5.2.5** Export functionality for presentations

### 5.3 Reproducibility Package 📝
**Purpose**: Enable reviewer verification

**Files to create:**
- `reproducibility/environment_setup.py`
- `reproducibility/data_validation.py`
- `reproducibility/checksum_verification.py`

**Tasks:**
- [ ] **5.3.1** Complete environment specification
- [ ] **5.3.2** Data integrity verification
- [ ] **5.3.3** Model checkpoint validation
- [ ] **5.3.4** Result reproduction scripts
- [ ] **5.3.5** Troubleshooting guide

---

## Phase 6: Domain-Specific Enhancements (Week 3-4) 📝

### 6.1 Photography-Specific Evaluation 📝
**Tasks:**
- [ ] **6.1.1** Perceptual quality metrics (LPIPS, FID)
- [ ] **6.1.2** User study framework
- [ ] **6.1.3** RAW processing pipeline validation
- [ ] **6.1.4** ISO performance analysis
- [ ] **6.1.5** Color accuracy assessment

### 6.2 Microscopy-Specific Evaluation 📝
**Tasks:**
- [ ] **6.2.1** Counting accuracy validation
- [ ] **6.2.2** Resolution preservation analysis
- [ ] **6.2.3** Photobleaching impact assessment
- [ ] **6.2.4** Quantitative analysis validation
- [ ] **6.2.5** Multi-channel processing evaluation

### 6.3 Astronomy-Specific Evaluation 📝
**Tasks:**
- [ ] **6.3.1** Source detection accuracy
- [ ] **6.3.2** Photometry precision analysis
- [ ] **6.3.3** Cosmic ray handling
- [ ] **6.3.4** Calibration accuracy assessment
- [ ] **6.3.5** Astrometric precision validation

---

## Implementation Priority Schedule

### Week 1 (Critical Fixes)
- **Day 1-2**: Fix guidance space conversion (1.1)
- **Day 3-4**: Implement fair L2 conditioning (1.2)
- **Day 5**: Verify metrics units (1.3)

### Week 2 (Core Analysis)
- **Day 1-2**: Residual analysis implementation (2.1-2.2)
- **Day 3-4**: L2 baseline integration (3.1)
- **Day 5**: Low-light analysis (3.2)

### Week 3 (Polish & Validation)
- **Day 1-2**: Evaluation protocol standardization (4.1)
- **Day 3-4**: Scientific validation (4.3)
- **Day 5**: Report generation (5.1)

### Week 4 (Final Enhancement)
- **Day 1-2**: Domain-specific metrics (6.1-6.3)
- **Day 3-4**: Reproducibility package (5.3)
- **Day 5**: Final testing and validation

---

## Success Metrics

### Quantitative Targets
- [ ] **Chi-squared = 1.0 ± 0.1** for our method on synthetic data
- [ ] **>2 dB PSNR improvement** over L2 baseline at <100 photons
- [ ] **<0.5 dB drop** for unified vs domain-specific models
- [ ] **Statistical significance p < 0.01** for all key comparisons

### Qualitative Targets
- [ ] **White noise residuals** for our method vs structured artifacts for L2
- [ ] **Publication-ready figures** for all three domains
- [ ] **Compelling visual evidence** of physics correctness
- [ ] **Reviewer-convincing ablation study**

### Technical Targets
- [ ] **100% reproducible** evaluation pipeline
- [ ] **Comprehensive documentation** of all procedures
- [ ] **Automated report generation** for all results
- [ ] **Robust error handling** for edge cases

---

## Risk Mitigation

### High-Risk Items
1. **Space conversion bugs**: Could invalidate physics claims → Extensive unit testing
2. **Unfair L2 comparison**: Could weaken paper impact → Careful conditioning design
3. **Missing residual analysis**: Could lose key visual proof → Priority implementation

### Contingency Plans
- **If L2 baseline performs too well**: Investigate conditioning/training differences
- **If residuals aren't white**: Debug guidance computation and space conversion
- **If evaluation takes too long**: Implement parallel processing and caching
- **If results aren't reproducible**: Add deterministic seeding and environment control

---

## Deliverables Checklist

### For Paper Submission
- [ ] **Table 1**: Quantitative comparison across all domains and baselines
- [ ] **Figure 1**: 4-panel residual analysis showing physics correctness
- [ ] **Figure 2**: Performance vs photon count scaling analysis
- [ ] **Supplementary**: Complete evaluation protocol and additional results

### For Reproducibility
- [ ] **Complete evaluation code** with documentation
- [ ] **Preprocessed datasets** with checksums
- [ ] **Model checkpoints** for all methods
- [ ] **Result reproduction scripts** with expected outputs

### For Follow-up Research
- [ ] **Extensible evaluation framework** for new domains
- [ ] **Baseline integration templates** for new methods
- [ ] **Comprehensive documentation** for future development
- [ ] **Performance benchmarks** for optimization targets

---

## Notes

- **Priority Focus**: Phases 1-2 are critical for paper submission
- **Timeline**: Aggressive but achievable with focused effort
- **Dependencies**: Some tasks can run in parallel, others are sequential
- **Quality Control**: Each phase includes validation and testing
- **Documentation**: All implementations must be well-documented

This comprehensive enhancement plan will transform the evaluation framework into a robust, scientifically rigorous system that provides compelling evidence for the research claims while ensuring reproducibility and reviewer confidence.
