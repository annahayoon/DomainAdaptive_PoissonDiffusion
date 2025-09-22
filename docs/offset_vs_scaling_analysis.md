# Offset vs Scaling Analysis for Negative Values

## Current Approach: Pure Offset (Better ✅)
```python
offset = abs(data.min())
data = data + offset
```

### Example:
- Original data range: [-2, 10]
- After offset: [0, 12]
- **Relative differences preserved**: (10 - (-2)) = 12 = (12 - 0)

## Alternative: Offset + Extra Constant (Previous)
```python
offset = abs(data.min()) + 1.0
data = data + offset
```

### Example:
- Original data range: [-2, 10]
- After offset: [1, 13]
- Still preserves relative differences but unnecessarily inflates values

## Why Pure Offset is Better

### 1. **Mathematical Properties**
- **Linear transformation**: `f(x) = x + c`
- **Preserves all derivatives**: `f'(x) = 1`
- **Preserves relative relationships**: `(a - b) = (a+c) - (b+c)`
- **Minimal modification**: Only shifts by exactly what's needed

### 2. **No Need for Additional Scaling**
When we add offset to handle negatives, we do NOT need to scale positive values because:

- **Relative distances preserved**: The difference between any two pixels remains the same
- **Gradient information intact**: All spatial gradients are unchanged
- **Noise characteristics maintained**: Additive offset doesn't change noise distribution shape
- **Dynamic range preserved**: Just shifted, not stretched or compressed

### 3. **Physical Interpretation**
```
Original: [-2, -1, 0, 1, 5, 10]  (with background subtraction)
Offset:   [ 0,  1, 2, 3, 7, 12]  (physical intensities)
```
- The offset represents the "true zero" of the detector
- Common in astronomy as "zero-point calibration"
- Analogous to dark current subtraction

## Why NOT to Scale After Offset

### Scaling Would Be Harmful:
```python
# DON'T DO THIS:
data = data + offset
data = data / data.max()  # Normalizing changes relative relationships!
```

### Problems with Scaling:
1. **Changes relative intensities**: A star 2x brighter should remain 2x brighter
2. **Alters noise properties**: Poisson noise scales with signal
3. **Loses absolute calibration**: Can't recover physical units
4. **Domain-specific issues**: Different images would have different scales

## Comparison Table

| Approach | Min Value | Max Value | Range | Relative Diff | Reversible |
|----------|-----------|-----------|--------|--------------|------------|
| Original | -2 | 10 | 12 | ✅ Original | - |
| Pure Offset | 0 | 12 | 12 | ✅ Preserved | ✅ Yes |
| Offset + 1 | 1 | 13 | 12 | ✅ Preserved | ✅ Yes |
| Offset + Scale | 0 | 1 | 1 | ❌ Changed | ⚠️ Partial |

## For Diffusion Models

### Why Pure Offset Works Best:
1. **Minimum = 0**: Perfect for Poisson noise (can't have negative photons)
2. **No artificial gaps**: No unnecessary gap between 0 and minimum value
3. **Natural range**: Keeps values in reasonable range for neural networks
4. **Consistent across batches**: Same offset for entire dataset

### During Training:
```python
# Forward pass
if data.min() < 0:
    offset = abs(data.min())
    data = data + offset
    store_offset(offset)

# Add noise (Poisson works now since data >= 0)
noisy = add_poisson_gaussian_noise(data)
```

### During Inference:
```python
# Model prediction
output = model(noisy)

# Reverse offset for evaluation
if offset_stored:
    output = output - offset
```

## Recommendation

**Use pure offset (abs(min)) without additional constants or scaling:**

```python
if clean.min() < 0:
    offset = abs(clean.min().item())  # ✅ BEST
    clean = clean + offset
    metadata["astronomy_offset"] = offset
```

**Benefits:**
- ✅ Minimal modification to data
- ✅ Preserves all relative relationships
- ✅ No artificial inflation
- ✅ Physically interpretable
- ✅ Fully reversible
- ✅ Works perfectly with Poisson-Gaussian noise model

## Summary

1. **Only add abs(min)** - no need for extra constants
2. **Don't scale** - preserve relative intensities
3. **Store offset** - for reversibility during evaluation
4. **Apply consistently** - same transform for train/val/test

This approach maintains the scientific integrity of the data while ensuring compatibility with diffusion models.
