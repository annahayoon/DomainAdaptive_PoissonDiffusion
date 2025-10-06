# DAPGD Usage Guide

## Quick Start

### 1. Basic Inference

```bash
# Run guided inference on a single image
python scripts/inference.py \
    --mode guided \
    --input data/test/image.png \
    --checkpoint checkpoints/model.pt \
    --s 1000 \
    --sigma_r 5 \
    --kappa 0.5

# Run on a directory of images
python scripts/inference.py \
    --mode guided \
    --input data/test/ \
    --output_dir results/ \
    --checkpoint checkpoints/model.pt \
    --s 1000 \
    --sigma_r 5
```

### 2. Domain-Specific Configuration

**Photography (10³ photons/pixel):**
```bash
python scripts/inference.py \
    --mode guided \
    --input data/photo/test/ \
    --checkpoint checkpoints/photo_model.pt \
    --s 1000 --sigma_r 5 --kappa 0.5
```

**Microscopy (10¹ photons/pixel):**
```bash
python scripts/inference.py \
    --mode guided \
    --input data/micro/test/ \
    --checkpoint checkpoints/micro_model.pt \
    --s 100 --sigma_r 2 --kappa 0.7
```

**Astronomy (10⁰ photons/pixel):**
```bash
python scripts/inference.py \
    --mode guided \
    --input data/astro/test/ \
    --checkpoint checkpoints/astro_model.pt \
    --s 10 --sigma_r 1 --kappa 1.0
```

## Python API

### Basic Usage

```python
import torch
from dapgd.sampling.dapgd_sampler import DAPGDSampler
from dapgd.sampling.edm_wrapper import EDMModelWrapper
from dapgd.guidance.pg_guidance import PoissonGaussianGuidance

# Load model
checkpoint = torch.load("checkpoints/model.pt")
network = checkpoint['ema']  # or checkpoint['model']

# Create wrapper
edm_wrapper = EDMModelWrapper(network, img_channels=3)

# Create guidance
guidance = PoissonGaussianGuidance(
    s=1000.0,
    sigma_r=5.0,
    kappa=0.5,
    tau=0.01
)

# Create sampler
sampler = DAPGDSampler(
    edm_wrapper=edm_wrapper,
    guidance=guidance,
    num_steps=50,
    device='cuda'
)

# Run inference
restored = sampler.sample(y_e=noisy_observation)
```

### Evaluating Results

```python
from dapgd.metrics.image_quality import compute_psnr, compute_ssim
from dapgd.metrics.physical import compute_chi_squared

# Image quality metrics
psnr = compute_psnr(restored, ground_truth)
ssim = compute_ssim(restored, ground_truth)

# Physical consistency
chi2 = compute_chi_squared(
    restored, noisy_observation,
    s=1000.0, sigma_r=5.0
)

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
print(f"χ²: {chi2:.4f} (should be ~1.0)")
```

### Noise Simulation

```python
from dapgd.guidance.pg_guidance import simulate_poisson_gaussian_noise

# Simulate Poisson-Gaussian noise
clean = torch.rand(1, 3, 256, 256)  # Range [0, 1]
noisy = simulate_poisson_gaussian_noise(
    clean,
    s=1000.0,
    sigma_r=5.0,
    seed=42
)
# noisy is in electron counts
```

## Configuration Files

Use YAML configuration files for different domains:

```yaml
# config/photography.yaml
physics:
  s: 1000.0
  sigma_r: 5.0
  background: 0.0

guidance:
  kappa: 0.5
  tau: 0.01
  mode: wls

sampling:
  num_steps: 50
  sigma_min: 0.002
  sigma_max: 80.0
```

Then use:
```bash
python scripts/inference.py --config config/photography.yaml --input data/test/
```

## Testing

Run the test suite:
```bash
# All tests
bash scripts/run_tests.sh

# Specific tests
pytest tests/test_pg_guidance.py -v
pytest tests/test_sampling.py -v
```

## Performance Profiling

Profile inference to identify bottlenecks:
```bash
python scripts/profile_inference.py \
    --checkpoint checkpoints/model.pt \
    --size 256 \
    --num_steps 50
```

## Debugging

Use the debugging utilities:

```python
from dapgd.utils.debugging import (
    SamplingDebugger,
    diagnose_sampling_failure,
    quick_sanity_check
)

# Quick sanity check
quick_sanity_check()

# Debug sampling
debugger = SamplingDebugger()
# In sampling loop:
debugger.check_step(x_t, sigma_t, step_idx)
debugger.print_summary()

# Diagnose failures
diagnostics = diagnose_sampling_failure(sampler, y_e)
print(diagnostics)
```

## Common Issues

### Issue 1: NaN in Sampling

**Solution**: Ensure inputs are properly clamped
```python
# Before guidance
denoised = torch.clamp(denoised, 0.0, 1.0)
```

### Issue 2: Poor Restoration Quality

**Solutions**:
- Increase guidance strength: `--kappa 0.7` or `--kappa 1.0`
- Decrease guidance threshold: `--tau 0.001`
- Check physical parameters are correct for your data

### Issue 3: Chi-Squared Far from 1.0

**If χ² < 1**: Reduce guidance strength (over-fitting)
**If χ² > 1**: Increase guidance strength or check parameters

## Advanced Features

### Calibration-Preserving Transforms

Handle arbitrary input sizes:
```python
from dapgd.data.transforms import CalibrationPreservingTransform

transform = CalibrationPreservingTransform(target_size=256)
y_transformed, metadata = transform.forward(y_noisy)

# Process
x_restored = sampler.sample(y_e=y_transformed)

# Restore original size
x_original = transform.inverse(x_restored, metadata)
```

### Batch Processing

Process multiple images efficiently:
```python
# Stack images
batch = torch.stack([img1, img2, img3, img4])

# Process in batch
results = sampler.sample(y_e=batch)
```

### Trajectory Recording

Record intermediate states:
```python
restored, trajectory = sampler.sample(
    y_e=noisy,
    return_trajectory=True
)

# trajectory is a list of states at each timestep
```

## Best Practices

1. **Always validate with χ²**: Aim for χ² ≈ 1.0 for physical consistency
2. **Start with default parameters**: Adjust based on results
3. **Use appropriate domain config**: Different domains need different parameters
4. **Profile before optimizing**: Identify actual bottlenecks
5. **Test with synthetic data**: Validate with known ground truth

## References

- Implementation Guide: `DAPGD_Implementation.md`
- Quick Reference: `QUICK_REFERENCE.md`
- Troubleshooting: `docs/troubleshooting.md`
