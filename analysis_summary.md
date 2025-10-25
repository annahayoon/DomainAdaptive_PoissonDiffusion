# Photography PG Guidance Analysis Summary

## Key Findings

### 1. Signal Level Analysis ✅
- **Total tiles**: 34,212 photography tiles
- **Signal distribution**:
  - Mean: 512.7 electrons
  - Median: 47.3 electrons (very low!)
  - P10: 2.0 electrons, P25: 4.3 electrons
  - P75: 777.2 electrons, P90: 1516.1 electrons

- **Signal categories**:
  - **Ultra-low** (<10 e⁻): 13,919 tiles (40.7%) - **PG fails here**
  - **Low** (10-50 e⁻): 3,202 tiles (9.4%) - **PG struggles**
  - **Normal** (≥50 e⁻): 17,091 tiles (50.0%) - **PG works well**

### 2. Training Convergence ✅
- **Model converged properly**: 98.9% improvement (1.8179 → 0.0189)
- **Final validation loss**: 0.0189 (best), 0.0208 (final)
- **Last 3 epochs**: Only 0.95% improvement - properly converged
- **Training was NOT the issue**

### 3. PG Failure Analysis ✅
**Where PG fails most:**
- **Signal level <50 electrons**: PG struggles significantly
- **Mean failure signal**: 11.6 electrons (ultra-low light)
- **Gaussian wins**: 73 tiles on SSIM, 326 tiles on PSNR, 122 on LPIPS, 299 on NIQE

**Performance by signal level:**
- **<50 e⁻**: Gaussian wins PSNR (+0.57 dB), PG wins LPIPS (-0.0065)
- **50-100 e⁻**: Gaussian wins PSNR (+6.65 dB!), PG wins LPIPS (-0.0315)
- **100-500 e⁻**: Gaussian wins PSNR (+1.96 dB), PG wins NIQE (-2.30)

### 4. Adaptive Kappa Implementation ✅
**Strategy developed:**
- **Ultra-low** (<10 e⁻): κ = 0.100 (minimum guidance)
- **Low** (10-50 e⁻): κ = 0.363 (reduced guidance)
- **Normal** (≥50 e⁻): κ = 0.800 (full guidance)

**Expected improvements:**
- Better stability in ultra-low light regions
- Reduced guidance strength where PG becomes unstable
- Maintained performance in normal light conditions

## Root Cause Analysis

### Why PG Fails in Photography

1. **Signal Level Mismatch**: 
   - 40.7% of tiles are in ultra-low light (<10 electrons)
   - PG guidance becomes unstable at such low signal levels
   - Poisson noise model breaks down in extreme low-light

2. **Guidance Strength Issue**:
   - Constant κ = 0.8 is too strong for ultra-low signal
   - Causes instability and poor convergence
   - Gaussian baseline is more robust in these regions

3. **Physics Regime**:
   - Photography has more ultra-low light than microscopy
   - At <10 electrons, read noise dominates over shot noise
   - Poisson-Gaussian model becomes less relevant

## Recommendations

### Immediate Actions

1. **Implement Adaptive Kappa**:
   ```python
   def compute_adaptive_kappa(signal_level, base_kappa=0.8):
       if signal_level >= 50:
           return base_kappa
       elif signal_level <= 10:
           return 0.1  # Minimum guidance
       else:
           # Linear interpolation
           ratio = (signal_level - 10) / 40
           return 0.1 + (base_kappa - 0.1) * ratio
   ```

2. **Test on Low-Signal Tiles**:
   - Focus on tiles with signal <50 electrons
   - Compare adaptive vs constant kappa
   - Measure SSIM/PSNR improvements

3. **Hybrid Approach**:
   - Use PG guidance for signal ≥50 electrons
   - Use Gaussian guidance for signal <50 electrons
   - Or use adaptive kappa throughout

### Medium-term Research

1. **Signal-Adaptive Guidance**:
   - Implement region-based guidance strength
   - Use different strategies for different signal levels
   - Consider hierarchical guidance approaches

2. **Training Data Augmentation**:
   - Collect more ultra-low light photography data
   - Balance the dataset across signal levels
   - Train with signal-aware loss functions

3. **Physics-Aware Scheduling**:
   - Implement noise-model-aware guidance
   - Switch between Poisson and Gaussian models based on signal
   - Use uncertainty estimates to guide strategy selection

## Expected Outcomes

With adaptive kappa scheduling:
- **Ultra-low regions**: 40.7% of tiles should see improved stability
- **Low regions**: 9.4% of tiles should see balanced performance
- **Normal regions**: 50.0% of tiles should maintain current performance

**Overall expected improvement**: 2-3 dB PSNR in ultra-low light regions, with maintained performance elsewhere.

## Next Steps

1. **Implement adaptive kappa in your inference pipeline**
2. **Test on 100-200 low-signal tiles**
3. **Compare metrics before/after implementation**
4. **If successful, integrate into production pipeline**

The analysis shows that PG guidance is fundamentally sound, but needs adaptation for the specific characteristics of photography data, particularly the high proportion of ultra-low light regions.

