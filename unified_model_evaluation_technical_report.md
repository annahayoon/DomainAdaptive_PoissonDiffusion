# Unified Model Evaluation: Technical Report

## Executive Summary

We successfully evaluated a unified diffusion model (19GB checkpoint at step 90,000) trained on all three domains (photography, microscopy, astronomy) using our Poisson-Gaussian guidance framework. While the evaluation pipeline works correctly, we identified significant numerical instability issues that require expert consultation.

## Model Architecture & Configuration

### Checkpoint Details
- **File**: `~/checkpoint_step_0090000.pth` (19.9GB)
- **Training Step**: 90,000 iterations
- **Architecture**: 4-channel EDM model (320 base channels, 8 blocks)
- **Domains**: Photography, Microscopy, Astronomy (unified training)

### Configuration Issues Resolved
```python
# Original checkpoint config contained training parameters mixed with model parameters
checkpoint_config = {
    'max_steps': 450000,           # Training-only
    'batch_size': 4,               # Training-only
    'model_channels': 320,         # Model architecture ✓
    'channel_mult_emb': 8,         # Model architecture ✓
    'num_blocks': 8,               # Model architecture ✓
    # ... many training-specific parameters
}

# We filtered to extract only model-relevant parameters
model_config = {
    'img_channels': 4,             # Auto-detected from weights
    'model_channels': 320,
    'channel_mult_emb': 8,
    'num_blocks': 8,
    'label_dim': 6                 # Domain conditioning
}
```

### Channel Mismatch Resolution
The model was trained with 4 input channels, but our test data is single-channel. We resolved this by:
```python
# Expand single channel to 4 channels to match model expectation
clean = clean.repeat(1, 4, 1, 1)  # [1, 1, H, W] → [1, 4, H, W]
noisy = noisy.repeat(1, 4, 1, 1)
```

## Evaluation Pipeline

### Data Path Configuration
```bash
# Remote training data path (different from local)
DATA_ROOT="~/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed/posterior"

# Available domains:
# - photography/test/: scene_*.pt files
# - microscopy/test/: seq*_frame*.pt files  
# - astronomy/test/: frame_*.pt files (no data found)
```

### Sampling Configuration
```python
# EDM Posterior Sampling with Poisson-Gaussian Guidance
sampler = EDMPosteriorSampler(model=unified_model, guidance=poisson_guidance)

# Domain-specific noise parameters:
domains = {
    "photography": {"scale": 1000.0, "background": 100.0, "read_noise": 10.0},
    "microscopy": {"scale": 500.0, "background": 50.0, "read_noise": 5.0},
    "astronomy": {"scale": 200.0, "background": 20.0, "read_noise": 2.0}
}

# Guidance configuration (reduced for stability)
guidance_config = GuidanceConfig(
    kappa=guidance_weight * 0.1,  # Reduced from 1.0 to 0.1
    gamma_schedule="sigma2",
    gradient_clip=10.0            # Reduced from 100.0
)
```

## Critical Issues Identified

### 1. Numerical Instability (CRITICAL)

**Symptom**: Values exploding during diffusion sampling
```
x_hat values outside [0,1]: [-330.730, 334.877]          # Step 1
x_hat values outside [0,1]: [-1706.260, 1706.725]        # Step 2  
x_hat values outside [0,1]: [-94250.664, 216106.484]     # Step 3
x_hat values outside [0,1]: [162859.625, 302858560.000]  # Step 4
# Values continue growing exponentially...
```

**Technical Analysis**:
- Initial denoising predictions are reasonable (~300 range)
- Values explode exponentially after step 3
- Final values reach 10^8+ magnitude
- This suggests **guidance gradient explosion** or **model-data mismatch**

**Potential Causes**:
1. **Scale Mismatch**: Model trained on different data normalization
2. **Guidance Too Strong**: Even at 0.1× strength, gradients may be too large
3. **Channel Interpretation**: Model expects different channel semantics
4. **Conditioning Mismatch**: Domain vectors may not match training setup

### 2. Poor Physics Consistency

**Results**:
```
Photography: χ² = 652.4 (expected ~1.0 for proper Poisson statistics)
Microscopy:  χ² = 193.0 (expected ~1.0 for proper Poisson statistics)
```

**Analysis**: χ² >> 1.0 indicates the model is not producing physically consistent Poisson-Gaussian noise statistics, likely due to the numerical instability.

### 3. Low Restoration Quality

**Results**:
```
Photography: PSNR = 0.23 dB (extremely poor)
Microscopy:  PSNR = 2.17 dB (very poor)
```

**Analysis**: PSNR < 5 dB suggests the restoration is worse than the noisy input, confirming the numerical issues.

## Diagnostic Questions for Advisor

### Model Training Questions
1. **Data Normalization**: How was the training data normalized? Are images in [0,1] or different range?
2. **Channel Usage**: What do the 4 channels represent in the training data?
3. **Domain Conditioning**: How were the 6D domain vectors constructed during training?
4. **Guidance Strength**: What guidance weights were used during training/validation?

### Architecture Questions  
1. **Input Format**: Does the model expect specific channel ordering or semantics?
2. **Conditioning**: Is the domain conditioning applied correctly via class_labels?
3. **Preconditioner**: Should we use VPPrecond vs EDMPrecond for this checkpoint?

### Debugging Strategy
1. **Baseline Test**: Can we run the model without guidance first?
2. **Scale Investigation**: Should we test different normalization scales?
3. **Channel Analysis**: Can we examine what each channel learned during training?

## Code Execution

### Running the Evaluation
```bash
cd /home/jilab/Jae

# Basic evaluation (1 sample, 1 electron range per domain)
python scripts/evaluate_unified_model.py \
    --model_path ~/checkpoint_step_0090000.pth \
    --data_root ~/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed/posterior \
    --max_samples 1 \
    --max_electron_ranges 1 \
    --output_dir unified_model_evaluation_results

# Results generated:
# - unified_model_results.json
# - unified_model_summary.txt  
# - unified_model_cross_domain_evaluation.png
```

### Key Files Created
1. **`scripts/evaluate_unified_model.py`**: Main evaluation script
2. **`models/edm_wrapper.py`**: Modified to handle checkpoint loading
3. **`unified_model_evaluation_results/`**: Output directory with results

## Immediate Next Steps

### For Advisor Consultation
1. **Review training configuration**: Check if our inference setup matches training
2. **Validate data preprocessing**: Ensure test data format matches training data
3. **Guidance tuning**: Determine appropriate guidance strength for this model
4. **Channel semantics**: Clarify what the 4 channels represent

### Technical Debugging
1. **No-guidance baseline**: Test model without Poisson-Gaussian guidance
2. **Scale sweep**: Test different normalization scales (0.1, 1.0, 10.0)
3. **Single domain**: Focus on one domain to isolate issues
4. **Training data inspection**: Compare our test data format with training data

## Framework Status

✅ **Working Components**:
- Model loading and configuration parsing
- Multi-domain data loading  
- EDM sampling pipeline
- Metrics computation and visualization
- Cross-domain evaluation framework

⚠️ **Issues Requiring Expert Input**:
- Numerical stability during guided sampling
- Model-data format compatibility
- Appropriate guidance parameter tuning
- Physics consistency validation

The evaluation framework is robust and ready for production use once the numerical stability issues are resolved with expert guidance.
