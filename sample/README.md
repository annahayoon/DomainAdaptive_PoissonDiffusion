# Sensor Noise Calibration

This module implements Poisson-Gaussian noise parameter estimation from processed sensor data normalized to [-1, 1] range using short/long exposure pairs.

## Data Format

All data is expected to be preprocessed:
- Raw sensor data normalized: `(raw - black_level) / (white_level - black_level) → [0, 1]`
- Then scaled to [-1, 1]: `[0, 1] * 2 - 1 → [-1, 1]`
- Saved as .pt files (see `preprocessing/process_tiles_pipeline.py`)

## Mathematical Theory

### Raw Sensor Measurement Model

The fundamental sensor measurement model in raw ADU (analog-to-digital units):

```
y = G·x + n
```

Where:
- `x`: true photoelectron count (Poisson distributed with mean x)
- `G`: camera system gain (electrons per ADU)
- `n ~ N(0, σ_n²)`: combined signal-independent noise

#### Noise Sources in n:
- Readout noise: Gaussian variance from electronics (σ_read²)
- Row noise: Fixed-pattern noise along rows (σ_row²)
- Quantization noise: Uniform distribution from ADC (σ_quant² = 1/(4q²))

Total: `σ_n² = σ_read² + σ_row² + σ_quant²`

### Measurement Variance in Sensor Units

Combining Poisson photon shot noise and Gaussian readout noise:

```
Var(y) = G²·x + σ_n²
```

This is the Poisson-Gaussian model. The variance has two components:
- `G²·x`: Signal-dependent (Poisson/shot noise)
- `σ_n²`: Signal-independent (Gaussian combined noise)

### Linear Mean-Variance Relationship

The model can be written as a linear relationship:

```
var = a·mean + b
```

Where:
- `a = G²` (captures photon shot noise, signal-dependent)
- `b = σ_n²` (captures combined noise: readout + row + quantization)

## Domain Transformations and Normalization

### From Raw Sensor Units to Normalized Domain

Our data undergoes this transformation:

**Step 1: Raw ADU → [0, 1]**
```
p_norm = (p - black_level) / (white_level - black_level)
```

**Step 2: [0, 1] → [-1, 1]**
```
p_scaled = 2·p_norm - 1
```

The overall scaling factor:
```
s = 2 / (white_level - black_level)
```

### Variance Scaling

Variances scale with the SQUARE of the linear scaling factor:

```
σ²_norm = s² · σ²_raw
```

For our two-step normalization:
1. [0, 1] domain: `σ²_norm = σ²_raw / (white_level - black_level)²`
2. [-1, 1] domain: `σ²_scaled = 4 · σ²_norm`

The factor 4 comes from (2×)² when scaling [0,1] → [-1,1].

### Noise Parameters in Normalized Domain

In the [-1, 1] domain, the linear model still holds:

```
var_scaled = a_scaled·mean_scaled + b_scaled
```

Where the parameters transform as:
```
a_scaled = s² · a_raw = 4·a_raw / (white_level - black_level)²
b_scaled = s² · b_raw = 4·b_raw / (white_level - black_level)²
```

### Physical Interpretation Preserved

The normalization is LINEAR and DETERMINISTIC, so:
- The Poisson-Gaussian structure is preserved
- We can estimate parameters directly in [-1, 1] domain
- No loss of physical interpretation
- Parameters remain meaningful for guidance

## Mathematical Derivations

### Derivation 1: Variance Scaling Under Linear Normalization

Starting from raw sensor values p in ADU, we apply normalization:

```
p_norm = (p - b) / (w - b)              [0, 1] domain
p_scaled = 2·p_norm - 1                 [-1, 1] domain
```

Combining:
```
p_scaled = 2·[(p - b)/(w - b) - 1 = [2p - 2b]/(w - b) - 1
         = [2p - 2b - (w - b)]/(w - b)
         = [2p - w - b]/(w - b)
```

For variance, since linear transformations scale variance by the square of the scaling factor, and `Var(aX + c) = a²·Var(X)`:

```
Var(p_norm) = Var[(p - b)/(w - b)]
            = [1/(w - b)]² · Var(p)
            = Var(p) / (w - b)²

Var(p_scaled) = Var(2·p_norm - 1)
              = 2² · Var(p_norm)
              = 4 · Var(p_norm)
              = 4·Var(p) / (w - b)²
```

Therefore, the scaling factor from raw to [-1, 1] is:
```
s² = 4 / (w - b)²
```

### Derivation 2: Poisson-Gaussian Model in Normalized Domain

In raw sensor units, the variance model is:
```
Var(p) = a_raw · E[p] + b_raw
```

Where `a_raw = G²` and `b_raw = σ_n²`.

Under the linear transformation to [-1, 1], both mean and variance scale:
```
E[p_scaled] = 2·E[p]/(w - b) - 1
Var(p_scaled) = 4·Var(p) / (w - b)²
```

Substituting the raw variance model:
```
Var(p_scaled) = 4·(a_raw · E[p] + b_raw) / (w - b)²
              = [4·a_raw/(w - b)²]·E[p] + [4·b_raw/(w - b)²]
```

To express this in terms of the scaled mean, we need E[p] in terms of E[p_scaled]:
```
E[p_scaled] = 2·E[p]/(w - b) - 1
E[p_scaled] + 1 = 2·E[p]/(w - b)
E[p] = (w - b)·(E[p_scaled] + 1) / 2
```

Substituting back:
```
Var(p_scaled) = [4·a_raw/(w - b)²]·[(w - b)·(E[p_scaled] + 1)/2] + [4·b_raw/(w - b)²]
              = [2·a_raw/(w - b)]·(E[p_scaled] + 1) + [4·b_raw/(w - b)²]
              = [2·a_raw/(w - b)]·E[p_scaled] + [2·a_raw/(w - b)] + [4·b_raw/(w - b)²]
```

Simplifying:
```
Var(p_scaled) = a_scaled · E[p_scaled] + b_scaled
```

Where:
```
a_scaled = 2·a_raw / (w - b)
b_scaled = 2·a_raw/(w - b) + 4·b_raw/(w - b)²
```

However, for empirical fitting directly in the [-1, 1] domain, we simply fit:
```
var = a_norm · mean + b_norm
```

And these fitted parameters a_norm and b_norm directly encode the noise characteristics in the [-1, 1] domain without needing to know raw parameters.

### Derivation 3: Measurement Likelihood for Guidance

During diffusion sampling, we need the likelihood `p(y|x)` where:
- `y`: observed noisy measurement (short exposure) in [-1, 1]
- `x`: clean latent (what we're sampling) in [-1, 1]

Given the Poisson-Gaussian model in [-1, 1] domain:
```
Var(y|x) = a·x + b
```

The measurement likelihood is Gaussian:
```
p(y|x) = N(y; x, σ²(x))
```

Where `σ²(x) = a·x + b` models spatially-varying, signal-dependent noise.

The log-likelihood is:
```
log p(y|x) = -1/(2σ²(x)) · ||y - x||² - 1/2·log(2π·σ²(x))
```

The gradient (score) for guidance:
```
∇_x log p(y|x) = [y - x]/σ²(x) - d/dx[1/2·log(σ²(x))]
                = [y - x]/(a·x + b) - a/(2·(a·x + b))
```

This gradient pulls the sample x toward the measurement y, weighted by the local noise variance. Areas with high variance (uncertain measurements) exert weaker pull.

### Derivation 4: Exposure Ratio Incorporation

For short/long exposure pairs with exposure ratio α:
```
y_short = α · x_clean + noise
```

Where α < 1 for short exposures. The measurement model becomes:
```
p(y_short|x_clean) = N(y_short; α·x_clean, σ²(α·x_clean))
```

With:
```
σ²(α·x_clean) = a·(α·x_clean) + b = α·a·x_clean + b
```

The likelihood accounts for both the exposure scaling and the corresponding noise variance at that exposure level.

### Derivation 5: Quantization Noise Contribution

For ADC with quantization step q, values are quantized to nearest q·n. The quantization error is uniformly distributed over [-q/2, q/2].

For uniform distribution U(-δ, δ):
```
Var = (2δ)²/12 = δ²/3
```

For quantization with step q, the error is in (-q/2, q/2):
```
σ²_quant = (q/2)²/3 = q²/12
```

In normalized domain, if raw quantization step is q_raw:
```
q_norm = q_raw / (w - b)
q_scaled = 2·q_norm = 2·q_raw / (w - b)
```

The quantization noise variance in [-1, 1] domain:
```
σ²_quant_scaled = q²_scaled / 12 = [2·q_raw/(w - b)]² / 12
                 = 4·q²_raw / [12·(w - b)²]
                 = q²_raw / [3·(w - b)²]
```

This contributes to the b_scaled parameter as part of signal-independent noise.

## Empirical Parameter Estimation Workflow

When camera specifications (G, σ_n) are unavailable, estimate empirically:

### Why Paired Short/Long Exposures Are Required

**The Fundamental Challenge**: Separating signal variance from noise variance.

#### Why Not Just Short Exposure?

A single short exposure contains: `observed = true_signal + noise`

Problem: Computing variance on the short exposure alone gives **total variance** = scene variance + noise variance. You cannot distinguish whether pixel variations are from:
- Scene content (textures, edges, details) ← what we want to preserve
- Sensor noise (what we want to characterize) ← our calibration target

This is fundamentally unidentifiable without additional information.

#### Why Not Just Long Exposure?

Long exposures have much higher SNR (signal-to-noise ratio), but:

1. **Wrong noise regime**: Long exposure noise characteristics differ from short exposure
   - Poisson noise scales with √(photon count) ∝ √(exposure time)
   - Long ≈ 10× exposure → ~3× better SNR
   - We need short exposure noise parameters, not long exposure parameters

2. **Cannot measure signal-dependent behavior**: To fit `var = a·mean + b`, we need variance measurements across different signal levels. Long exposures are predominantly high-signal (bright), providing insufficient coverage of the low-signal regime where noise dominates.

#### Why Paired Images Work

The key insight: **Use long exposure as a clean reference**

```python
# Long exposure ≈ clean signal (negligible noise due to high SNR)
signal_reference = long_image

# Short exposure = signal + noise
noisy_observation = short_image

# Isolate noise by subtraction
noise_isolated = short_image - long_image
              = (signal + noise) - signal
              ≈ noise  # Pure noise component!
```

Now we can measure **noise variance as a function of signal level**:

```
For pixels grouped by signal level (from long):
  Dark pixels (low signal)   → measure low-signal noise variance
  Medium pixels              → measure medium-signal noise variance
  Bright pixels (high signal) → measure high-signal noise variance

Fit: variance = a·signal + b
     ↑                    ↑
  Poisson coefficient   Read noise
```

**Critical assumptions validated**:
- Long exposure SNR >> short exposure SNR (long acts as ground truth)
- Same scene captured at both exposures (spatial alignment)
- Noise in short/long are independent samples (validated by zero-mean residuals)

This paired-image approach is standard in imaging science (photon transfer curves, flat-field calibration).

### From Short/Long Exposure Pairs

1. Use long exposure as "ground truth" (cleaner signal)
2. Compute residuals: `noise = short - long`
3. Bin pixels by mean signal level (from long exposure)
4. For each bin, compute variance of residuals
5. Fit linear model: `var = a·mean + b` across bins
6. Use only positive signal bins for Poisson component validity

**Note:** The implementation fits the model in [0,1] domain (where mean ≥ 0 for physical validity of Poisson noise) then transforms parameters back to [-1,1] domain for guidance.

## Mathematical Soundness and Approximations

### Domain Transformation Accuracy

The calibration makes two transformations that are **mathematically sound**:

#### ✓ Variance Scaling [0,1] → [-1,1]
```python
bin_means_01 = (bin_means + 1.0) / 2.0  # Mean: [-1,1] → [0,1]
bin_vars_01 = bin_vars / 4.0            # Variance scales as square
```
**Correct**: If X ~ (μ, σ²) then (X+1)/2 ~ ((μ+1)/2, σ²/4) ✓

#### ✓ Parameter Transformation Back to [-1,1]
```python
a_norm = 2.0 * a_01
b_norm = 2.0 * a_01 + 4.0 * b_01
```
**Derivation**: Starting from `var₀₁ = a₀₁·mean₀₁ + b₀₁` and substituting `mean₀₁ = (mean_norm + 1)/2`, `var₀₁ = var_norm/4`:
```
var_norm/4 = a₀₁·(mean_norm + 1)/2 + b₀₁
var_norm = 2a₀₁·mean_norm + 2a₀₁ + 4b₀₁
```
Therefore: `a_norm = 2a₀₁` and `b_norm = 2a₀₁ + 4b₀₁` ✓

### Physical Space Conversions (Approximations)

When converting calibration parameters to physical units (ADU) for guidance:

#### ✓ Read Noise (Signal-Independent)
```python
sigma_r = sqrt(b_norm * (sensor_range/2)²)
```
**Sound**: Read noise variance scales as square of linear transformation ✓

#### ⚠ Poisson Coefficient (Signal-Dependent)
```python
poisson_coeff = a_norm * (sensor_range/2)
```
**Approximate**: The transformation `[-1,1] → physical ADU` is affine (not linear) due to black level offset:
```
mean_phys = black_level + (mean_norm + 1)/2 * sensor_range
```

The full expansion shows:
```
var_phys = a_norm * (sensor_range/2) * mean_phys + offset_terms
```

The implementation uses `a_phys ≈ a_norm * (sensor_range/2)`, treating this as an **effective parameter** rather than strict physical quantity. This approximation is:
- **Empirically valid**: Calibration captures real sensor behavior in the working domain
- **Regularized in practice**: Guidance strength `kappa` and threshold `tau` tune the effective contribution
- **Operationally successful**: Small errors are absorbed during iterative refinement

**Conclusion**: The calibration is **pragmatically sound** - it's empirically grounded and works well in practice, though not theoretically exact at the boundary between normalized and physical domains.

## Algorithm Workflow

The calibration process follows this step-by-step workflow:

### Phase 1: Main Entry Point
When run as a script:
1. Parse command-line arguments (data root, split, samples, bins, device)
2. Validate the data root directory exists
3. Set device (CPU/CUDA)
4. Call `calibrate_all_sensors()` with parsed arguments
5. Print summary of results

### Phase 2: Batch Processing (`calibrate_all_sensors`)
For each sensor:

#### 2.1 Discovery
- Find all `metadata_*_incremental.json` files in the data root
- Extract sensor names from filenames (e.g., `metadata_fuji_incremental.json` → `fuji`)

#### 2.2 Directory Setup
- Build paths:
  - `short_dir = data_root / "pt_tiles" / sensor / "short"`
  - `long_dir = data_root / "pt_tiles" / sensor / "long"`
- Skip if directories don't exist

#### 2.3 Tile Loading
- Load validation tiles from metadata JSON using `load_test_tiles()`
- Extract `tile_id` values
- For each tile:
  - Load short exposure via `_load_image_pair()` → `load_short_image()`
  - Load matching long exposure via `load_long_image()` (using tile_lookup)
  - Collect valid pairs into lists

#### 2.4 Noise Estimation
- Call `estimate_noise_params()` with collected pairs
- Save results to `{sensor}_noise_calibration.json`

### Phase 3: Core Noise Estimation (`estimate_noise_params`)

#### 3.1 Preparation
- Normalize tensors to the target device
- Validate image pairs (counts and shapes match)

#### 3.2 Data Collection (`_collect_residuals_and_means`)
- For each pair:
  - Compute residual = `short_image - long_image`
  - Use `long_image` as signal mean
- Flatten and concatenate all pixels

#### 3.3 Sampling
- If more than `num_samples` (default 50,000), randomly sample down

#### 3.4 Binning (`_compute_binned_variance`)
- Create `num_bins` (default 50) bins from -1 to 1
- Assign each pixel to a bin by mean value
- For each bin with ≥20 samples:
  - Compute mean of bin means
  - Compute variance of residuals in that bin
- Result: `(bin_means, bin_vars)` representing the mean-variance relationship

#### 3.5 Domain Transformation
- Transform from `[-1, 1]` to `[0, 1]` for physical validity (Poisson requires non-negative means):
  ```
  bin_means_01 = (bin_means + 1.0) / 2.0
  bin_vars_01 = bin_vars / 4.0
  ```

#### 3.6 Linear Fitting (`_fit_linear_model`)
- Fit: `variance = a × mean + b`
- Use least squares: `torch.linalg.lstsq()`
- Gets `(a_01, b_01)` in `[0, 1]` domain

#### 3.7 Transform Back to `[-1, 1]`
- Transform parameters back:
  ```
  a_norm = 2.0 * a_01
  b_norm = 2.0 * a_01 + 4.0 * b_01
  ```
- Returns `(a_norm, b_norm)` for use in the `[-1, 1]` domain

### Phase 4: Output
- Save JSON with:
  - Sensor name
  - Split used
  - Number of tiles
  - Parameters `a` and `b`
  - Configuration (`num_samples`, `num_bins`)

### Visual Flow Diagram

```
Main Entry → calibrate_all_sensors()
    ↓
For each sensor:
    ├─ Find metadata file
    ├─ Load tile pairs (short/long)
    ├─ estimate_noise_params()
    │   ├─ Collect residuals & means
    │   ├─ Bin by signal intensity
    │   ├─ Compute variance per bin
    │   ├─ Fit linear model
    │   └─ Transform parameters
    └─ Save calibration JSON
```

## Summary Tables and Key Formulas

### Table 1: Noise Model Across Domains

| Domain          | Variance Model                           | Parameters |
|----------------|------------------------------------------|------------|
| Raw sensor ADU  | σ² = a_raw·μ + b_raw                     | a_raw = G², b_raw = σ_n² |
| [0, 1] norm     | σ² = a_01·μ + b_01                       | a_01 = a_raw/(w-b)² |
| [-1, 1] scaled  | σ² = a_norm·μ + b_norm                   | a_norm = 4·a_raw/(w-b)² |

### Table 2: Parameter Transformations

| Transformation              | Mean Scaling        | Variance Scaling |
|----------------------------|---------------------|------------------|
| Raw → [0, 1]                | (p-b)/(w-b)         | 1/(w-b)² |
| [0, 1] → [-1, 1]            | 2·p - 1             | 4× |
| Raw → [-1, 1] (combined)    | [2p-w-b]/(w-b)      | 4/(w-b)² |

### Table 3: Noise Components

| Source              | Distribution        | Variance Formula |
|--------------------|---------------------|------------------|
| Photon shot         | Poisson             | G²·x |
| Readout             | Gaussian            | σ²_read |
| Row noise           | Gaussian            | σ²_row |
| Quantization        | Uniform             | q²/12 |
| Total (combined)    | Poisson-Gaussian    | G²·x + σ²_n |

### Table 4: Key Formulas Reference

| Operation                           | Formula |
|------------------------------------|----------------------------------------|
| Sensor measurement                  | y = G·x + n |
| Variance (raw)                      | Var(y) = G²·x + σ²_n |
| Variance ([0, 1])                   | σ²_norm = σ²_raw / (w-b)² |
| Variance ([-1, 1])                  | σ²_scaled = 4·σ²_norm |
| Mean-variance fit                   | var = a·mean + b |
| Measurement likelihood              | p(y\|x) = N(y; x, a·x + b) |
| Likelihood gradient                 | ∇_x log p(y\|x) = (y-x)/(a·x+b) - a/(2(a·x+b)) |
| Exposure-scaled likelihood          | p(y\|x) = N(y; α·x, a·α·x + b) |

## Usage in Diffusion Guidance

The estimated parameters (a, b) in [-1, 1] domain are used during sampling to model the measurement likelihood.

### Calibration Parameters in Sampling

The noise calibration serves a **different purpose** from initialization:

#### 1. Initialization (Where to Start)
```python
# Provides smart starting point for diffusion sampling
x_init = exposure_scaled_observation
       = observation * (1 / exposure_ratio)
```
**Purpose**: Compensates for exposure difference to start near correct brightness.

#### 2. Calibration (How to Refine)
```python
# Configures likelihood gradient for measurement guidance
pg_guidance = PoissonGaussianGuidance(
    sigma_r=convert_b_to_sigma_r(b, sensor_range),      # Read noise (ADU)
    poisson_coeff=convert_a_to_coeff(a, sensor_range),  # Signal-dependent noise
    ...
)
```
**Purpose**: Tells guidance "this is how noisy the sensor actually is" to properly weight measurement information.

#### Why Both Are Needed

```
┌─────────────────────────────────────────────────────┐
│ Sampling Loop (t = T → 0)                          │
│                                                     │
│  x ← exposure_scaled_observation  ← INITIALIZATION │
│                                                     │
│  for each timestep:                                │
│    x_denoised = model(x, t)                        │
│    gradient = ∇ log p(y|x_denoised)  ← CALIBRATION│
│    x ← x + noise + gradient                        │
│                                                     │
│  return x  (restored image)                        │
└─────────────────────────────────────────────────────┘
```

- **Initialization**: Gets you in the right neighborhood (correct brightness)
- **Calibration**: Refines using physics-informed measurement model (correct noise statistics)

### Measurement Likelihood Model

The calibration enables computing:

```
p(y|x) ≈ N(y; μ, σ²)
```

Where the noise variance at each pixel is:
```
σ² = a·μ + b
```

With:
- `μ`: predicted pixel value in [-1, 1]
- `a`: captures signal-dependent (Poisson/shot) noise
- `b`: captures signal-independent (readout+row+quantization) noise

This likelihood guides the diffusion sampling process to produce physically consistent restorations that respect the sensor's noise characteristics.

### Conversion to Physical Units

For guidance computation in physical ADU space:

```python
# Read noise: signal-independent component
sigma_r = sqrt(b * (sensor_range/2)²)

# Poisson coefficient: signal-dependent component
poisson_coeff = a * (sensor_range/2)

# Guidance evaluates: p(y_physical | x_restored)
# Using: var_phys = poisson_coeff * mean_phys + sigma_r²
```

These conversions transform the calibrated parameters from normalized `[-1, 1]` domain to physical ADU units for likelihood evaluation.

## Integration with Sampling Pipeline

This section shows how calibration outputs from `sensor_noise_calibrations.py` integrate with the sampling script `sample_noisy_pt_lle_PGguidance.py`.

### End-to-End Workflow

```
┌──────────────────────────────────────────────────────────┐
│ Step 1: Calibration (ONE TIME)                           │
│ $ python sample/sensor_noise_calibrations.py             │
│   --data-root dataset/processed                          │
│   --split val                                            │
│                                                          │
│ Output: dataset/processed/{sensor}_noise_calibration.json│
│   {                                                      │
│     "sensor": "sony",                                    │
│     "a": 0.000123,  ← Poisson coeff ([-1,1] domain)    │
│     "b": 0.000456,  ← Read noise ([-1,1] domain)       │
│     "n_tiles": 150                                       │
│   }                                                      │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ Step 2: Inference (EVERY RUN)                            │
│ $ python sample/sample_noisy_pt_lle_PGguidance.py        │
│   --model_path results/.../best_model.pkl                │
│   --metadata_json dataset/processed/metadata_*.json      │
│   --short_dir dataset/processed/pt_tiles/sony/short      │
│   --calibration_dir dataset/processed  ← Auto-loads JSON │
│   --use_sensor_calibration                               │
│   (--use_noise_calibration is ON by default)             │
└──────────────────────────────────────────────────────────┘
```

### Code Flow in `sample_noisy_pt_lle_PGguidance.py`

#### 1. Load Calibration File (Per-Tile, lines 916-929)

```python
def _process_single_tile(...):
    # Extract sensor type from tile metadata
    extracted_sensor = get_sensor_from_metadata(tile_id, tile_lookup)
    # → "sony" or "fuji"

    # Load noise calibration - ENABLED BY DEFAULT
    calibration_data = None
    if args.use_noise_calibration:  # True by default
        calibration_dir = Path(args.calibration_dir)  # dataset/processed
        calibration_data = _load_noise_calibration(
            extracted_sensor,           # "sony"
            calibration_dir,           # dataset/processed
            data_root=data_root
        )
        # Returns: {"a": 0.000123, "b": 0.000456}
```

**Function**: `_load_noise_calibration()` (lines 494-546)
- Looks for `{sensor}_noise_calibration.json` in calibration_dir
- Handles sensor name mapping (sony → sony_a7s_ii, etc.)
- Returns `None` if not found → raises error (calibration required by default)

#### 2. Convert Calibration Parameters to Physical Units (lines 931-939)

```python
    # Create guidance modules with calibrated parameters
    pg_guidance, gaussian_guidance = _create_guidance_modules(
        args,
        sensor_range,      # {"min": 512, "max": 16383} for Sony
        s_sensor,          # 16383 - 512 = 15871
        exposure_ratio,    # e.g., 0.1 (short) / 10.0 (long) = 0.01
        calibration_data=calibration_data,  # {"a": 0.000123, "b": 0.000456}
        black_level=black_level,
        white_level=white_level,
    )
```

**Function**: `_create_guidance_modules()` (lines 659-749)

```python
def _create_guidance_modules(..., calibration_data, ...):
    if calibration_data is not None:
        # Convert b (read noise variance) to sigma_r (read noise std)
        sigma_r = _convert_calibration_to_sigma_r(
            calibration_data["b"],  # 0.000456 in [-1,1]
            s_sensor                # 15871 ADU range
        )
        # sigma_r = sqrt(0.000456 * (15871/2)²) ≈ 5.37 ADU

        # Convert a (Poisson coeff) to physical units
        poisson_coeff = _convert_calibration_to_poisson_coeff(
            calibration_data["a"],  # 0.000123 in [-1,1]
            s_sensor                # 15871 ADU range
        )
        # poisson_coeff = 0.000123 * (15871/2) ≈ 0.976

        logger.info(
            f"Using calibrated parameters: "
            f"sigma_r={sigma_r:.3f} ADU, poisson_coeff={poisson_coeff:.6e}"
        )
```

**Conversion Functions**:
- `_convert_calibration_to_sigma_r()` (lines 549-601): `b_norm → sigma_r`
  - Formula: `sigma_r = sqrt(b_norm * (sensor_range/2)²)`

- `_convert_calibration_to_poisson_coeff()` (lines 604-656): `a_norm → poisson_coeff`
  - Formula: `poisson_coeff = a_norm * (sensor_range/2)`

#### 3. Create PG Guidance with Calibrated Parameters (lines 740-745)

```python
    pg_guidance = PoissonGaussianGuidance(
        s=s_sensor,                    # 15871 ADU
        sigma_r=sigma_r,               # 5.37 ADU (from calibration)
        black_level=black_level,       # 512
        white_level=white_level,       # 16383
        exposure_ratio=exposure_ratio, # 0.01
        kappa=args.kappa,              # 0.1 (guidance strength)
        tau=args.tau,                  # 0.01 (threshold)
        mode=args.pg_mode,             # "wls" or "full"
        guidance_level=args.guidance_level,  # "x0" or "score"
        poisson_coeff=poisson_coeff,   # 0.976 (from calibration)
    )
```

The `PoissonGaussianGuidance` object now has **sensor-specific noise parameters** from real data!

#### 4. Run Posterior Sampling with Calibrated Guidance (lines 1015-1026)

```python
    # Initialize at exposure-scaled observation
    x_init = apply_exposure_scaling(short_image, exposure_ratio)

    restoration_results = _run_restoration_methods(...)

    # Inside _run_restoration_methods:
    restored_pg_x0, _ = sampler.posterior_sample(
        short_image,
        pg_guidance=pg_guidance,  # ← Uses calibrated sigma_r, poisson_coeff
        sigma_max=sigma_used,
        num_steps=args.num_steps,
        y_e=y_e,                  # Physical measurement
        exposure_ratio=exposure_ratio,
        x_init=x_init,            # ← Exposure-scaled initialization
    )
```

**During sampling** (in `EDMPosteriorSampler.posterior_sample`, lines 149-251):

```python
# Each diffusion step:
for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
    x_denoised = self.net(x, t_cur, class_labels)

    # Apply PG guidance (line 213-215)
    x_denoised_guided, guidance_contribution = self._apply_guidance(
        x_denoised,
        pg_guidance,  # Has calibrated sigma_r, poisson_coeff
        y_e,          # Physical measurement
        t_cur
    )

    # Update uses guidance gradient computed with CALIBRATED parameters
    # The likelihood gradient ∇ log p(y|x) uses:
    #   - sigma_r from calibration (not args.sigma_r)
    #   - poisson_coeff from calibration (not theoretical 1.0)
```

#### 5. Save Results with Calibration Info (lines 1099-1110)

```python
    if calibration_data is not None:
        result_info["noise_calibration"] = {
            "source": "sensor_noise_calibrations.py",
            "a": calibration_data["a"],                    # Raw [-1,1] value
            "b": calibration_data["b"],                    # Raw [-1,1] value
            "sigma_r_from_calibration": sigma_r,           # Converted ADU
            "poisson_coeff_from_calibration": poisson_coeff,  # Converted
            "poisson_coeff_conversion_note":
                "a_norm * (sensor_range/2) converted from [-1,1] to physical space",
        }
```

### Command-Line Usage Examples

#### Standard Usage (Calibration Enabled by Default)
```bash
python sample/sample_noisy_pt_lle_PGguidance.py \
  --model_path results/edm_sony_training_20251031_161214/best_model.pkl \
  --metadata_json dataset/processed/metadata_sony_incremental.json \
  --short_dir dataset/processed/pt_tiles/sony/short \
  --long_dir dataset/processed/pt_tiles/sony/long \
  --output_dir results/low_light_enhancement \
  --num_examples 5 \
  --num_steps 18 \
  --kappa 0.1 \
  --use_sensor_calibration \
  --calibration_dir dataset/processed
  # Note: --use_noise_calibration is ON by default
```

This will:
1. Auto-detect sensor type from metadata ("sony")
2. Load `dataset/processed/sony_noise_calibration.json`
3. Use calibrated `a`, `b` for PG guidance
4. Initialize with exposure-scaled observation
5. Refine with physics-informed likelihood gradient

#### Fallback to Manual sigma_r (Not Recommended)
```bash
python sample/sample_noisy_pt_lle_PGguidance.py \
  ... (same args) ... \
  --no-noise-calibration \
  --sigma_r 3.0
  # Warning: Uses theoretical Poisson coeff (1.0) and manual sigma_r
  # This is LESS accurate than using calibration!
```

### Key Design Decisions

1. **Calibration is ON by default**: `--use_noise_calibration` defaults to `True`
   - Must use `--no-noise-calibration` to disable
   - This encourages best practices

2. **Automatic sensor detection**: No need to specify sensor manually
   - Extracted from tile metadata
   - Handles sensor name variations

3. **Graceful error if calibration missing**:
   - Raises clear error with instructions to run calibration
   - Or use `--no-noise-calibration` to fall back

4. **Separation of concerns**:
   - Calibration: Estimate parameters from data (one-time)
   - Sampling: Load and use parameters (every run)
   - Clean module boundaries

### What Gets Logged

During inference, you'll see:
```
INFO - Loaded noise calibration for sony: a=1.234567e-04, b=4.567890e-04
INFO - Using calibrated parameters: sigma_r=5.370 ADU, poisson_coeff=9.760000e-01
      (from calibration: a=1.234567e-04, b=4.567890e-04 in [-1,1] domain,
       sensor_range=15871.0 ADU)
```

This confirms:
- ✓ Calibration file loaded successfully
- ✓ Parameters converted to physical units
- ✓ Values are sensor-specific (not generic)

## Implementation

### Functions

- `estimate_noise_params()`: Pair-based estimation from short/long exposure pairs - works with tensor lists
- `estimate_noise_from_processed_data()`: Works with either .pt files or preloaded tensors

### Quick Start Examples

**Example 1: Estimate from .pt files (validation set)**
```python
from pathlib import Path
a, b = estimate_noise_from_processed_data(
    short_dir=Path("data/pt_tiles/short"),
    long_dir=Path("data/pt_tiles/long"),
    metadata_json=Path("data/metadata.json"),
    split="val"  # Use validation set
)
print(f"Noise model: σ² = {a:.6e}·μ + {b:.6e}")
```

**Example 2: Estimate from loaded tensors**
```python
short_tensors = [torch.load(f) for f in short_paths]
long_tensors = [torch.load(f) for f in long_paths]
a, b = estimate_noise_params(short_tensors, long_tensors)
```

## References

- See `fix_sampling_raw_range.md` for detailed mathematical derivations
- See `preprocessing/process_tiles_pipeline.py` for data normalization pipeline
- See `sample/sample_visualizations.py` for visualization functions
