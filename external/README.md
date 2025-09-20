# External Dependencies Integration

This directory contains integration with external codebases required for the project.

## EDM (Elucidating the Design Space of Diffusion-Based Generative Models)

### Overview

We integrate the EDM codebase for the core diffusion model architecture. EDM provides:
- Stable v-parameterization for diffusion training
- Efficient sampling schedules
- Well-tested U-Net architecture

### Integration Steps

1. **Clone EDM Repository**
   ```bash
   cd external/
   git clone https://github.com/NVlabs/edm.git
   cd edm/
   ```

2. **Install EDM Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python -c "import torch; from training.networks import EDMPrecond; print('EDM import successful')"
   ```

### Required Modifications for Our Project

The EDM codebase needs the following modifications for domain conditioning:

#### 1. Add Conditioning Support to EDMPrecond

**File**: `training/networks.py`

**Modification**: Add conditioning input to the `EDMPrecond` class:

```python
class EDMPrecond(torch.nn.Module):
    def __init__(self, ..., condition_dim=0):
        # Add conditioning support
        self.condition_dim = condition_dim
        if condition_dim > 0:
            self.condition_encoder = torch.nn.Linear(condition_dim, model_channels)
    
    def forward(self, x, sigma, condition=None, **kwargs):
        # Integrate conditioning into forward pass
        if condition is not None and self.condition_dim > 0:
            cond_emb = self.condition_encoder(condition)
            # Apply conditioning via FiLM or cross-attention
```

#### 2. Modify U-Net for Conditioning

**File**: `training/networks.py` (UNetModel class)

**Modification**: Add conditioning pathways at multiple scales:

```python
class UNetModel(torch.nn.Module):
    def forward(self, x, timesteps, condition_emb=None, **kwargs):
        # Apply conditioning at each resolution level
        if condition_emb is not None:
            # FiLM conditioning: scale and shift features
            for block in self.input_blocks:
                if hasattr(block, 'apply_conditioning'):
                    x = block.apply_conditioning(x, condition_emb)
```

### Integration Testing

Create `external/test_edm_integration.py`:

```python
import torch
import sys
sys.path.append('edm')

def test_edm_import():
    """Test that EDM can be imported successfully."""
    try:
        from training.networks import EDMPrecond
        print("✓ EDM import successful")
        return True
    except ImportError as e:
        print(f"✗ EDM import failed: {e}")
        return False

def test_edm_basic_functionality():
    """Test basic EDM model creation and forward pass."""
    try:
        from training.networks import EDMPrecond
        
        # Create model
        model = EDMPrecond(
            img_resolution=128,
            img_channels=1,
            model_channels=128,
            condition_dim=6  # Our conditioning dimension
        )
        
        # Test forward pass
        x = torch.randn(1, 1, 128, 128)
        sigma = torch.tensor([1.0])
        condition = torch.randn(1, 6)
        
        with torch.no_grad():
            output = model(x, sigma, condition=condition)
        
        assert output.shape == x.shape
        print("✓ EDM basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ EDM basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing EDM integration...")
    test_edm_import()
    test_edm_basic_functionality()
```

### Version Compatibility

- **EDM Version**: Latest from main branch (as of project start)
- **PyTorch**: 2.0+ (required for EDM)
- **CUDA**: 11.8+ (for optimal performance)

### Known Issues and Workarounds

1. **Memory Usage**: EDM models can be memory-intensive. Use gradient checkpointing for training.
2. **Conditioning Integration**: May require careful tuning of conditioning strength.
3. **Sampling Speed**: EDM uses 18 steps by default; can be reduced to 5-10 for faster inference.

### Alternative Approaches

If EDM integration proves problematic, alternatives include:
1. **DDPM**: Simpler but less efficient sampling
2. **Score-based models**: Similar performance but different parameterization
3. **Custom U-Net**: Build from scratch with conditioning built-in

### Contact and Support

For EDM-specific issues, refer to:
- Original paper: https://arxiv.org/abs/2206.00364
- GitHub repository: https://github.com/NVlabs/edm
- Issues: Report integration problems in our project issues