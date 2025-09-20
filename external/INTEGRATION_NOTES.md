# EDM Integration Notes

## Current Status

✅ **Setup scripts created**
✅ **Integration test framework ready**
⏳ **Awaiting EDM repository clone and testing**

## Integration Strategy

### 1. Minimal Modification Approach

We aim to integrate EDM with minimal modifications to preserve the original codebase's stability:

**Option A: Use existing label_dim parameter**
- EDM already supports class conditioning via `label_dim`
- We can repurpose this for our 6-dimensional domain conditioning
- Requires no code modifications to EDM

**Option B: Fork and modify EDM**
- Add explicit conditioning support
- More control but requires maintaining a fork
- Only if Option A proves insufficient

### 2. Conditioning Vector Design

Our 6-dimensional conditioning vector:
```python
condition = torch.cat([
    domain_one_hot,      # [3] - photography/microscopy/astronomy
    log_scale_norm,      # [1] - normalized log10(scale)
    rel_read_noise,      # [1] - read_noise / scale
    rel_background       # [1] - background / scale
])
```

### 3. Model Wrapper Architecture

```python
class EDMModelWrapper(nn.Module):
    """Wrapper around EDM that handles our conditioning."""
    
    def __init__(self, edm_config, condition_dim=6):
        super().__init__()
        
        # Load EDM model with label conditioning
        self.edm_model = EDMPrecond(
            **edm_config,
            label_dim=condition_dim
        )
        
        # Optional: additional conditioning layers
        self.condition_processor = nn.Identity()
    
    def forward(self, x, sigma, condition=None):
        # Process conditioning if needed
        if condition is not None:
            condition = self.condition_processor(condition)
        
        # Forward through EDM
        return self.edm_model(x, sigma, class_labels=condition)
```

### 4. Testing Strategy

1. **Basic Import Test**: Verify EDM can be imported
2. **Functionality Test**: Test forward pass with dummy data
3. **Conditioning Test**: Test with our 6D conditioning vectors
4. **Memory Test**: Verify memory usage is reasonable
5. **Gradient Test**: Ensure gradients flow properly

### 5. Fallback Plans

If EDM integration proves problematic:

**Plan B: Simplified U-Net**
- Implement basic U-Net with built-in conditioning
- Less optimal but more controllable

**Plan C: Alternative Diffusion Models**
- DDPM: Simpler parameterization
- Score-based models: Different but equivalent approach

## Expected Challenges

1. **Memory Usage**: EDM models are large (128M+ parameters)
2. **Conditioning Integration**: May need careful tuning
3. **Version Compatibility**: PyTorch/CUDA version alignment
4. **Sampling Speed**: EDM default 18 steps may be slow

## Success Criteria

- [ ] EDM imports successfully
- [ ] Basic forward pass works
- [ ] Conditioning integration functional
- [ ] Memory usage < 8GB for 128x128 images
- [ ] Training loop can be established
- [ ] Sampling produces reasonable outputs

## Next Steps After Integration

1. Create `models/edm_wrapper.py` with our wrapper class
2. Implement domain conditioning logic
3. Set up training pipeline
4. Test on synthetic data
5. Validate physics integration

## Resources

- EDM Paper: https://arxiv.org/abs/2206.00364
- EDM Code: https://github.com/NVlabs/edm
- Our Design Doc: `.kiro/specs/poisson-diffusion-restoration/design.md`