# Space Conversion Protocol Documentation

## Overview

This document defines the space conversion protocol used in the Domain-Adaptive Poisson-Gaussian Diffusion system. The protocol ensures consistent and mathematically correct conversion between normalized model space [0,1] and physical electron space used for likelihood computation.

## Core Principle

The system maintains **strict separation** between two spaces:

1. **Model Space**: Normalized intensity values in range [0,1]
2. **Physical Space**: Electron counts with physical units

All physics computations (likelihood gradients) occur in electron space, while the diffusion model operates in normalized space. Space conversion uses the chain rule to maintain mathematical correctness.

## Mathematical Foundation

### Forward Conversion: Normalized → Electrons

For any normalized image prediction `x ∈ [0,1]`:

```python
lambda_e = scale * x + background
```

Where:
- `scale`: Dataset normalization scale (electrons)
- `background`: Background offset (electrons)
- `lambda_e`: Predicted electron count

### Chain Rule: Gradient Conversion

The likelihood gradient in electron space must be converted back to normalized space:

```python
∇_x log p(y|x) = ∇_λ log p(y|λ) * (∂λ/∂x)
```

Where `∂λ/∂x = scale`, so:

```python
score_normalized = score_electrons * scale
```

## Implementation Details

### Poisson-Gaussian Guidance

```python
# In compute_score method:
lambda_e = self.scale * x_hat + self.background  # Forward conversion
variance = lambda_e + self.read_noise**2         # Physics in electron space
score_electrons = (y_observed - lambda_e) / variance  # Electron-space gradient
return score_electrons * self.scale             # Chain rule conversion
```

### L2 Guidance

```python
# In compute_score method:
prediction_electrons = self.scale * x_hat + self.background  # Forward conversion
residual = y_observed - prediction_electrons                 # Electron-space residual
score_electrons = residual / self.noise_variance            # Electron-space gradient
return score_electrons * self.scale                         # Chain rule conversion
```

## Protocol Requirements

### 1. Consistent Interface

All guidance methods must implement:

```python
def compute_score(
    self,
    x_hat: torch.Tensor,      # [0,1] normalized
    y_observed: torch.Tensor, # electrons
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:            # [0,1] normalized gradient
```

**Input**: `x_hat` normalized [0,1], `y_observed` in electrons
**Output**: Gradient in normalized space

### 2. Parameter Validation

All guidance methods must validate:

```python
# Scale must be positive
if self.scale <= 0:
    raise ValueError(f"Scale must be positive, got {self.scale}")

# Background should be non-negative (physical constraint)
if self.background < 0:
    logger.warning(f"Negative background may be unphysical: {self.background}")

# Read noise must be non-negative
if self.read_noise < 0:
    raise ValueError(f"Read noise must be non-negative, got {self.read_noise}")
```

### 3. Numerical Stability

All guidance methods must implement:

```python
# Variance regularization
variance = torch.clamp(variance, min=self.config.variance_eps)

# Gradient clipping
if self.config.gradient_clip > 0:
    gradient = torch.clamp(gradient, -self.config.gradient_clip, self.config.gradient_clip)

# NaN/Inf detection
if torch.isnan(gradient).any() or torch.isinf(gradient).any():
    raise NumericalStabilityError("Guidance computation produced NaN/Inf")
```

## Testing and Validation

### Mathematical Correctness Tests

1. **Chain Rule Verification**:
   - Compute expected gradient using manual electron-space calculation
   - Compare with guidance method output
   - Tolerance: `1e-6` for exact matches, `1e-5` for numerical precision

2. **Symmetry Tests**:
   - Convert normalized → electrons → back to normalized
   - Verify round-trip consistency
   - Ensures no information loss in conversion

3. **Range Preservation**:
   - Test with boundary values (0.0, 1.0)
   - Ensure outputs remain finite and reasonable
   - Verify no clipping artifacts

### Edge Case Handling

1. **Extreme Values**:
   - Very small scales (`1e-6`)
   - Very large scales (`1e6`)
   - Zero read noise
   - Maximum/minimum signal levels

2. **Numerical Precision**:
   - Float32 vs Float64 precision
   - Error propagation analysis
   - Gradient magnitude consistency

## Configuration Parameters

### GuidanceConfig

```python
@dataclass
class GuidanceConfig:
    mode: str = "wls"                    # "wls" or "exact"
    gamma_schedule: str = "sigma2"       # "sigma2", "linear", or "const"
    kappa: float = 0.5                   # Guidance strength
    gradient_clip: float = 10.0          # Gradient clipping threshold
    variance_eps: float = 1e-8           # Variance regularization
    enable_masking: bool = True          # Enable pixel masking
    normalize_gradients: bool = False    # Gradient normalization
    adaptive_kappa: bool = False         # Adaptive guidance strength
    collect_diagnostics: bool = True     # Collect diagnostic info
    max_diagnostics: int = 1000          # Max diagnostic samples
```

### Parameter Selection Guidelines

1. **gradient_clip**:
   - Use `1e-10` for testing (disable clipping)
   - Use `10.0-100.0` for training (reasonable clipping)
   - Use `1000.0+` for validation tests (avoid clipping)

2. **variance_eps**:
   - Use `1e-8` for normal operation
   - Use `1e-6` for extreme low-light
   - Never set to `0.0` (division by zero risk)

3. **kappa**:
   - Use `0.1-1.0` for conservative guidance
   - Use `1.0-5.0` for strong guidance
   - Use `5.0+` for very strong guidance (may cause instability)

## Domain-Specific Considerations

### Photography
```python
# Typical values
scale = 1000.0      # ~1000 electrons per normalized unit
background = 100.0   # ~100 electrons background
read_noise = 5.0     # ~5 electrons read noise
```

### Microscopy
```python
# Typical values
scale = 100.0       # Lower scale due to smaller pixels
background = 10.0    # Lower background
read_noise = 2.0     # Lower read noise
```

### Astronomy
```python
# Typical values
scale = 10.0        # Very low scale due to photon counting
background = 1.0     # Minimal background
read_noise = 1.0     # Minimal read noise
```

## Error Handling and Debugging

### Common Issues

1. **NaN/Inf Values**:
   - Check variance regularization
   - Verify input ranges
   - Use gradient clipping

2. **Incorrect Gradients**:
   - Verify scale and background parameters
   - Check chain rule implementation
   - Validate test data ranges

3. **Space Conversion Bugs**:
   - Use provided validation tests
   - Check parameter consistency
   - Verify chain rule application

### Diagnostic Information

All guidance methods provide diagnostics:

```python
diagnostics = guidance.get_diagnostics()
print(f"Gradient norm: {diagnostics['grad_norm_mean']:.3f}")
print(f"Chi-squared: {diagnostics['chi2_mean']:.3f}")
print(f"SNR: {diagnostics['snr_mean_db']:.1f} dB")
```

## Validation Test Suite

The space conversion validation test suite includes:

1. **Mathematical Correctness**:
   - Chain rule verification
   - Forward/backward conversion consistency
   - Gradient magnitude analysis

2. **Numerical Stability**:
   - Edge case handling
   - Precision analysis
   - Error propagation

3. **Interface Consistency**:
   - Cross-method compatibility
   - Parameter validation
   - Error handling

### Running Validation Tests

```bash
# Run all space conversion tests
python -m pytest tests/test_space_conversion.py -v

# Run specific test categories
python -m pytest tests/test_space_conversion.py::TestSpaceConversionCorrectness -v
python -m pytest tests/test_space_conversion.py::TestSpaceConversionProtocol -v
```

## Best Practices

1. **Always test space conversion** when adding new guidance methods
2. **Use provided validation tests** to catch implementation errors
3. **Document parameter choices** for reproducibility
4. **Monitor diagnostic information** during training
5. **Validate on real data** after synthetic validation passes

## Implementation Checklist

- [ ] Forward conversion: normalized → electrons
- [ ] Physics computation in electron space
- [ ] Chain rule application for gradient conversion
- [ ] Input/output validation
- [ ] Numerical stability measures
- [ ] Comprehensive testing
- [ ] Documentation of parameter choices

This protocol ensures that all guidance methods maintain mathematical correctness and numerical stability across different imaging domains and experimental conditions.
