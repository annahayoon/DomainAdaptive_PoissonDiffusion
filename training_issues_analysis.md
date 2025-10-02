# Critical Training Issues Analysis

## 🚨 Major Issues Found

### 1. **Batch Size Too Small for 1.6B Model**
```bash
BATCH_SIZE="4"              # Line 28: Way too small!
GRAD_ACCUM="16"             # Line 29: Effective batch = 64
```
- **Problem**: Batch size of 4 is extremely small for a 1.6B parameter model
- **Impact**: High gradient noise, unstable training
- **Fix**: H100 can handle much larger batches (32-64 base batch size)

### 2. **Learning Rate Mismatch**
```bash
LEARNING_RATE="1e-4"        # Line 27: Too high for small batch
```
- **Problem**: LR of 1e-4 is calibrated for large batches, but you're using tiny batches
- **Impact**: Training instability, divergence after initial progress
- **Fix**: Either increase batch size OR reduce learning rate to 1e-5

### 3. **Model Too Large for Configuration**
```bash
MODEL_CHANNELS="320"        # Line 30: 1.6B parameters!
NUM_BLOCKS="8"              # Line 31: Very deep
```
- **Problem**: 1.6B model with batch size 4 is severely undertrained per sample
- **Impact**: Model can't learn properly with so few examples per update

### 4. **Data Loading Issue (Potential)**
In `train_unified_prior_clean.py`, line 166-170:
```python
# Pad with zeros to reach 4 channels
pad_channels = 4 - clean_norm.shape[0]
padding = torch.zeros(pad_channels, 128, 128)
clean_norm = torch.cat([clean_norm, padding], dim=0)
```
- **Problem**: Padding 1-channel data with zeros creates 75% dead channels
- **Impact**: Model wastes capacity learning to ignore padded channels
- **Better approach**: Replicate the single channel 4 times

### 5. **Noise Schedule Issue**
Line 742 in training script:
```python
sigma = torch.exp(torch.randn(clean.shape[0], device=training_manager.device) * 1.2 - 1.2)
```
- **Problem**: This creates sigma values mostly between 0.1 and 1.0
- **Impact**: Model doesn't learn to denoise at higher noise levels
- **Standard EDM**: Should sample log-uniformly from wider range

## 📊 Why 25K Model is Better Than 90K

The 25K checkpoint likely captured the model at its **optimal point** before:
1. Learning rate was too high for later training
2. Gradient accumulation issues compounded
3. Model started overfitting to noise patterns

## 🔧 Recommended Fixes

### Option A: Quick Fix (Use 25K Model)
1. **Bring the 25K checkpoint immediately**
2. Test with proper evaluation
3. Should show 20+ dB PSNR improvement

### Option B: Proper Retraining
```bash
# Better configuration
BATCH_SIZE="32"             # Leverage H100 memory
GRAD_ACCUM="2"              # Effective batch = 64
LEARNING_RATE="5e-5"        # More stable
MODEL_CHANNELS="256"        # Slightly smaller, trains better
```

### Option C: Fix Current Training
1. Resume from 25K checkpoint (not 90K)
2. Reduce learning rate to 1e-5
3. Increase base batch size
4. Fix channel padding issue

## 📈 Training Trajectory Analysis

```
Steps 0-25K:   Loss 6.0 → 0.06  ✅ Excellent progress
Steps 25K-90K: Loss 0.06 → 6.74 ❌ Catastrophic divergence
```

This pattern suggests:
- **Early training**: Model learned successfully
- **Mid training**: Learning rate too high, caused divergence
- **Current state**: Model essentially randomized

## 🎯 Immediate Action Items

1. **Copy 25K checkpoint to current node**:
```bash
scp remote:/path/to/checkpoint_step_025000.pth ~/
```

2. **Test with 25K model**:
```python
# Use existing evaluation scripts
python scripts/evaluate_unified_model_physics_correct.py \
    --checkpoint ~/checkpoint_step_025000.pth
```

3. **Expected Results**:
- Clean reconstruction: 25-30 dB PSNR
- Noisy denoising: 15-25 dB PSNR
- Proper noise reduction (not amplification)

## 💡 Key Insight

The training script itself is well-structured, but the **hyperparameter configuration** is mismatched:
- Batch size too small for model size
- Learning rate too high for small batches
- Possible learning rate scheduling issue

The 25K checkpoint represents the model at peak performance before configuration issues caused divergence.
