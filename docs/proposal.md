# Research Proposal: Domain-Adaptive Poisson-Gaussian Diffusion for Low-Light Image Restoration
---

## High-Level Research Vision

### The Problem We're Solving

Low-light imaging is a fundamental challenge across multiple scientific and consumer domains:

- **Photography**: Smartphone cameras in night mode, DSLR astrophotography
- **Microscopy**: Fluorescence imaging of live cells (photodamage limits)
- **Astronomy**: Observing distant galaxies with few photons

Current deep learning methods fail in extreme low-light because they assume Gaussian noise, but **photon arrival is fundamentally Poisson**. At <100 photons per pixel, this difference is critical.

### Why This is Scientifically Hard

#### Challenge 1: Physics-Aware Noise Modeling
- **Bright regions** (>1000 photons): Noise looks Gaussian (σ ≈ √signal)
- **Dark regions** (<100 photons): Discrete Poisson, highly non-Gaussian
- **Real sensors**: Mixed Poisson (photon) + Gaussian (read) noise
- **Current methods**: Wrongly assume uniform Gaussian → fail in dark regions

#### Challenge 2: Cross-Domain Generalization
Different domains have vastly different characteristics:
| Domain | Typical Photons | Pixel Size | Dynamic Range | Key Challenge |
|--------|----------------|------------|---------------|---------------|
| Photography | 10-10,000 | 4 μm | 14-bit | High ISO noise |
| Microscopy | 1-1,000 | 0.1-1 μm | 16-bit | Photobleaching |
| Astronomy | 0.1-100 | 0.04" | 32-bit | Cosmic rays |

Training separate models for each domain is impractical. We need **one model that adapts**.

#### Challenge 3: Scale and Metadata Preservation
- Images have different resolutions (phone: 4K, microscope: 512×512, telescope: varies)
- Physical units differ (μm vs arcseconds)
- Must preserve calibration for quantitative analysis
- Need perfect reconstruction for scientific validity

### Our Solution: Three Key Innovations

#### 1. Physically Correct Likelihood Guidance
Instead of generic L2 loss, we use the **exact Poisson-Gaussian likelihood**:
```
∇ log p(y|x) = s·(y - λ)/(λ + σ_r²)
```
This naturally weights dark regions (high noise) differently than bright regions.

#### 2. Unified Model with Domain Conditioning
One diffusion model handles all domains via conditioning:
- **Shared prior**: Natural image statistics are universal
- **Domain embedding**: Tells model about sensor characteristics
- **Scale awareness**: Normalized to photons, not arbitrary units

#### 3. Reversible Transforms with Complete Metadata
We solve the reconstruction problem completely:
- Transform any input size → model size (128×128)
- Track all scaling, cropping, padding operations
- Perfectly reconstruct original resolution
- Preserve physical units and calibration

### Why Our Approach is Novel

**Existing methods** fall into two camps:
1. **Classical** (BM3D, Anscombe VST): Correct physics but limited expressiveness
2. **Deep learning** (DnCNN, Noise2Void): Powerful but physics-ignorant

**We bridge both**:
- Diffusion prior captures complex image statistics (like deep learning)
- Poisson-Gaussian guidance ensures physical consistency (like classical)
- Domain conditioning enables generalization (novel)

### Expected Scientific Impact

#### Quantitative Improvements
- **2-3 dB PSNR gain** over state-of-the-art in <100 photon regime
- **χ² = 1** (statistically consistent) unlike existing methods (χ² >> 1)
- **Single model** matches domain-specific models (within 0.5 dB)

#### Practical Applications
1. **Photography**: Better night mode without computational overhead
2. **Microscopy**: Longer live-cell imaging with less photodamage
3. **Astronomy**: Detect fainter objects in same exposure time
4. **Medical**: Lower radiation dose X-ray/CT with same quality

#### Broader Impact
- **Open source**: Full code + pretrained models released
- **Educational**: Bridges physics and ML communities
- **Extensible**: Easy to add new domains (X-ray, radar, etc.)

---

## Research Methodology

### Phase 1: Physics Validation (Weeks 1-2)
**Goal**: Prove our noise model is correct

- Generate synthetic data with known Poisson-Gaussian statistics
- Verify likelihood gradients match theoretical predictions
- Compare with Anscombe VST and other approximations
- **Success metric**: Residuals are white noise (no structure)

### Phase 2: Single-Domain Mastery (Weeks 3-4)
**Goal**: Beat state-of-the-art on one domain first

- Start with photography (most data available)
- Train diffusion prior on clean images
- Add Poisson-Gaussian guidance for denoising
- **Success metric**: >2 dB improvement at ISO 12800+

### Phase 3: Cross-Domain Generalization (Weeks 5-6)
**Goal**: One model for all domains

- Add microscopy and astronomy data
- Implement domain conditioning
- Balance training across domains
- **Success metric**: <0.5 dB drop vs domain-specific

### Phase 4: Scientific Validation (Week 7)
**Goal**: Prove scientific utility

- Quantitative microscopy: Measure fluorophore counts
- Astronomy: Detect known faint sources
- Photography: Perceptual studies
- **Success metric**: Experts prefer our results

---

## Technical Approach (Summary)

### Core Algorithm
```python
# Training: Standard diffusion on clean images
x ~ p_data(x)  # Clean image
x_t = x + σ_t·ε  # Add noise
v_θ = model(x_t, σ_t, condition)  # Predict velocity
loss = ||v_θ - (ε - x)/σ_t||²

# Inference: Guided sampling with physics
for t in T..0:
    v = model(x_t, σ_t, condition)  # Prior
    x_0 = x_t - σ_t·v  # Denoised estimate

    # Physics-based correction
    grad = (y - s·x_0)/(s·x_0 + σ_r²)  # Poisson-Gaussian
    x_0 = x_0 + γ(σ_t)·grad  # Guided update

    x_{t-1} = step(x_0, v, σ_{t-1})  # Next timestep
```

### Key Design Decisions

1. **Why Diffusion?**
   - Best generative models currently
   - Natural way to incorporate likelihood guidance
   - Iterative refinement matches physics intuition

2. **Why Not End-to-End?**
   - Can't simulate all possible sensor configurations
   - Paired data is expensive/impossible in some domains
   - Unsupervised (prior + physics) is more flexible

3. **Why Unified Model?**
   - Shared low-level features (edges, textures)
   - More training data when combined
   - Single deployment for all applications

---

## Experimental Plan

### Datasets
| Domain | Dataset | Size | Purpose |
|--------|---------|------|---------|
| Photography | SID Sony/Fuji | 5,000 pairs | Main evaluation |
| Photography | ELD | 240 pairs | Cross-camera test |
| Microscopy | FMD | 12,000 images | Fluorescence |
| Microscopy | Private 2P | 500 volumes | Two-photon |
| Astronomy | Hubble Legacy | 10,000 fields | Deep sky |
| Astronomy | Ground-based | 1,000 images | Atmospheric |

### Baselines
1. **Classical**: BM3D, Anscombe+BM3D, Richardson-Lucy
2. **Supervised**: DnCNN, NAFNet, Restormer
3. **Unsupervised**: Noise2Void, Self2Self, Neighbor2Neighbor
4. **Diffusion**: DPS (L2), DiffPIR, our method with L2

### Metrics
- **Standard**: PSNR, SSIM, LPIPS
- **Physics**: χ² per pixel, residual whiteness, Poisson likelihood
- **Domain-specific**:
  - Photography: Perceptual quality (user study)
  - Microscopy: Counting accuracy, resolution (FWHM)
  - Astronomy: Source detection, photometry error

### Ablations
1. **Noise model**: Poisson-Gaussian vs Gaussian vs Poisson-only
2. **Guidance**: WLS vs exact vs Anscombe
3. **Conditioning**: With vs without domain embedding
4. **Scale**: Effect of training data scale diversity
5. **Architecture**: EDM vs DDPM vs Score-based

---

## Why This Research Matters

### Scientific Significance
- **First** physically-correct diffusion for photon-limited imaging
- **First** unified model across photography/microscopy/astronomy
- **First** to achieve χ² = 1 (statistical consistency) with deep learning

### Practical Impact
- **Immediate**: Drop-in replacement for existing denoisers
- **Near-term**: Enable new imaging modalities (ultra-low dose)
- **Long-term**: Foundation for physics-aware generative models

### Theoretical Contributions
1. **Proof** that correct likelihood beats generic L2 in low-photon regime
2. **Framework** for incorporating physics into diffusion models
3. **Analysis** of when domain transfer works vs fails

---

## Success Criteria and Deliverables

### Paper Deliverables
- [ ] **Main paper**: 8-page conference paper (ICLR/NeurIPS)
- [ ] **Supplement**: Full derivations, additional experiments
- [ ] **Code**: Complete implementation with documentation
- [ ] **Models**: Pre-trained checkpoints for all domains
- [ ] **Data**: Benchmark dataset with proper splits

### Quantitative Success Metrics
- [ ] >2 dB PSNR improvement at <100 photons
- [ ] χ² within [0.9, 1.1] on calibrated data
- [ ] Single model within 0.5 dB of domain-specific
- [ ] 5× faster than classical iterative methods

### Qualitative Success Metrics
- [ ] Astronomers can detect fainter sources
- [ ] Microscopists can image longer without photodamage
- [ ] Photographers prefer results in blind test

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Domain gap too large | Start with similar domains, gradually increase diversity |
| Calibration errors | Robust estimation, allow ±20% uncertainty |
| Training instability | Use EDM (most stable), careful lr scheduling |
| Compute requirements | Start small (128×128), progressive growing |
| Lack of paired data | Self-supervised with physics consistency |

---

## Timeline and Resources

### Timeline (8 weeks)
- **Weeks 1-2**: Physics validation, data preparation
- **Weeks 3-4**: Single domain implementation
- **Weeks 5-6**: Multi-domain training
- **Week 7**: Evaluation and ablations
- **Week 8**: Paper writing

### Compute Requirements
- **Training**: 4× A100 GPUs for 1 week
- **Inference**: Single GPU, ~5 seconds per megapixel
- **Storage**: ~500GB for all datasets

### Team Requirements
- **Lead researcher**: Algorithm development
- **Engineer**: Implementation and optimization
- **Domain experts**: One per domain for validation

---

## Conclusion

This research bridges the gap between **physics and learning** for low-light imaging. By respecting the fundamental Poisson nature of light while leveraging modern diffusion models, we can achieve both **mathematical correctness and practical performance**. The unified framework across domains demonstrates that **universal physical principles** can guide machine learning to generalize beyond traditional boundaries.

**The key insight**: When we encode the right physics (Poisson-Gaussian noise) into the right framework (diffusion models), we get models that not only perform better but are **scientifically trustworthy** — a critical requirement for scientific imaging and quantitative analysis.

This work opens the door to **physics-aware generative models** that could revolutionize computational imaging, from consumer cameras to scientific instruments, enabling us to **see what was previously hidden in the noise**.
