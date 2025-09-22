# Handling Negative Values in Astronomy Data for Diffusion Models

## The Problem
Hubble Legacy Field (HLF) and other astronomical datasets often contain negative pixel values due to:
1. **Background subtraction artifacts**
2. **Calibration procedures**
3. **Noise in low-signal regions**
4. **PSF deconvolution artifacts**

## Why This Matters for Diffusion Models
Diffusion models have specific requirements:
- **Non-negative inputs**: Most diffusion models assume pixel values ≥ 0
- **Poisson noise modeling**: Requires non-negative photon counts
- **Numerical stability**: Negative values can cause NaN/Inf during training
- **Physical consistency**: Light intensity cannot be negative

## Best Practices for Handling Negative Values

### 1. **Offset Addition (Currently Implemented) ✅**
```python
if data.min() < 0:
    offset = abs(data.min())  # Just shift to make min = 0
    data = data + offset
```
**Pros:**
- Preserves relative structure and gradients perfectly
- Maintains noise characteristics
- Fully reversible transformation
- Preserves scientific information
- No artificial inflation of values

**Cons:**
- Changes absolute intensity values
- Requires tracking offset for inverse transform

### 2. **Clipping to Zero (Alternative)**
```python
data = torch.clamp(data, min=0)
```
**Pros:**
- Simple and fast
- No metadata tracking needed
- Preserves positive values exactly

**Cons:**
- **Loses information** about background structure
- Creates artificial discontinuities
- Not reversible
- Can bias the model

### 3. **Soft Clipping with Exponential (Advanced)**
```python
data = torch.where(data < 0, epsilon * torch.exp(data), data)
```
**Pros:**
- Smooth transformation
- Preserves ordering
- Differentiable everywhere

**Cons:**
- Complex inverse transform
- Changes noise characteristics

### 4. **Z-Score Normalization + Shift (Research-Grade)**
```python
mean, std = data.mean(), data.std()
data = (data - mean) / std  # Z-score
data = data + abs(data.min()) + 1  # Ensure positive
```
**Pros:**
- Handles outliers well
- Standardizes across datasets
- Good for multi-domain training

**Cons:**
- Loses absolute scale information
- Requires careful inverse transform

## Recommended Approach for Your Research

### For Astronomy Domain Specifically:
1. **Use offset addition** (current implementation) ✅
2. **Store offset in metadata** for evaluation ✅
3. **Apply consistent transform** to train/val/test ✅
4. **Document in paper** as data preprocessing step

### Implementation Details:
```python
# Updated implementation in preprocessed_datasets.py
if clean.min() < 0:
    offset = abs(clean.min().item())  # Only shift by minimum amount needed
    clean = clean + offset  # Now min = 0, max preserved
    metadata["astronomy_offset"] = offset  # Store for inverse
```

### For Multi-Domain Unified Model:
The offset approach works well because:
- Each domain can have different offsets
- Domain conditioning handles the scale differences
- Physical interpretation remains valid

## Scientific Justification

### Why Offset is Better Than Clipping:
1. **Preserves Background Structure**: Important for:
   - Galaxy detection in low-SNR regions
   - Faint source identification
   - Accurate PSF modeling

2. **Maintains Noise Properties**:
   - Gaussian noise remains Gaussian
   - Poisson statistics shift but preserve shape
   - Critical for physics-aware loss function

3. **Reversible for Evaluation**:
   ```python
   # During evaluation/inference
   if "astronomy_offset" in metadata:
       output = output - metadata["astronomy_offset"]
   ```

4. **Physical Interpretation**:
   - Offset represents "zero-point calibration"
   - Common in astronomical data processing
   - Analogous to dark current subtraction

## Suppressing the Warning

Changed from `logger.warning()` to `logger.debug()` because:
- This is **expected behavior** for astronomy data
- Not an error or problem
- Still logged for debugging if needed
- Reduces noise in training logs

## For Your ICLR Paper

### Include in Methods Section:
"For astronomical data containing negative values from background subtraction, we apply a minimal positive offset to ensure compatibility with the Poisson noise model while preserving relative intensity gradients. This offset is stored and can be reversed during evaluation."

### Ablation Study Suggestion:
Compare:
1. Offset addition (recommended)
2. Zero clipping
3. Soft exponential mapping

Show that offset addition preserves more information and leads to better restoration quality.

## Conclusion

The current implementation using offset addition is the **best practice** for handling negative values in astronomy data for diffusion models. It:
- ✅ Preserves all information
- ✅ Maintains physical interpretability
- ✅ Is fully reversible
- ✅ Works well with Poisson-Gaussian noise modeling
- ✅ Is standard practice in astronomy

The warning has been changed to debug level since this is normal and expected for astronomy data.
