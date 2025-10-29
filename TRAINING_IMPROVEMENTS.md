# Training Script Improvements - Peer Review

## Overview

The training script has been refactored to follow proper **separation of concerns**:
- **Preprocessing**: Handles all domain-specific complexity (sensor calibration, Bayer packing, normalization)
- **Training**: Simple unconditional diffusion on tensors
- **Inference**: Applies domain-specific physics-informed guidance

This aligns with the paper's architecture where domain-specific physics is used for **inference-time guidance**, not base model training.

---

## Critical Issues Fixed

### 1. ❌ **Removed Unnecessary Label Handling**

**Problem:** The code had `use_labels=True` and `label_dim=3`, making the model class-conditional when it should be unconditional.

```python
# BEFORE (WRONG)
dataset_kwargs = dict(
    use_labels=True,
    label_dim=3,
    ...
)
```

```python
# AFTER (CORRECT)
dataset_kwargs = dict(
    class_name="data.dataset.SimplePTDataset",
    data_root=args.data_root,
    metadata_json=args.metadata_json,
    split="train",
    image_size=args.img_resolution,
    channels=args.channels,
)
```

**Why:**
- EDM trains **unconditional** diffusion models: P(x), not P(x|y)
- Labels are only for class-conditional generation (ImageNet classes, etc.)
- For image restoration, conditioning comes from **inference-time guidance**, not training
- The base model learns the distribution of clean images without needing labels

---

### 2. ❌ **Removed Domain-Specific Code from Training**

**Problem:** The code had `domain="photography"` and domain-specific parameters, unnecessarily coupling domain concerns to training.

```python
# BEFORE (WRONG)
dataset_kwargs = dict(
    class_name="data.dataset.EDMPTDataset",
    domain="photography",  # Unnecessary coupling
    use_labels=True,       # Breaks unconditional training
    label_dim=3,
    data_range="normalized",
    max_size=None,
)
```

```python
# AFTER (CORRECT)
dataset_kwargs = dict(
    class_name="data.dataset.SimplePTDataset",
    data_root=args.data_root,
    metadata_json=args.metadata_json,
    split="train",
    image_size=args.img_resolution,
    channels=args.channels,
)
```

**Why:**
- All .pt files are **already preprocessed** with domain-specific handling
- Bayer packing, sensor calibration, normalization happen in the preprocessing pipeline
- Training just needs tensors in [-1, 1], regardless of source
- Removes tight coupling and improves code reusability
- Makes training truly domain-agnostic (can train on any preprocessed tensors)

---

### 3. ❌ **Removed Unused Import**

**Problem:** The code imported `create_edm_pt_datasets` but never used it.

```python
# BEFORE (WRONG)
from data.dataset import create_edm_pt_datasets

# Never used anywhere in the script
```

```python
# AFTER (CORRECT)
from data.dataset import SimplePTDataset

# Used for dataset construction
```

---

### 4. 🟡 **Simplified Dataset Class**

**Problem:** The `EDMPTDataset` was over-engineered for simple tensor loading.

**Solution:** Changed to `SimplePTDataset` which is simpler and clearer about responsibilities:

```python
class SimplePTDataset(torch.utils.data.Dataset):
    """Simple dataset for loading preprocessed .pt tiles."""

    def __init__(self, data_root, metadata_json, split, image_size, channels):
        # Simple initialization
        pass

    def __getitem__(self, idx):
        # Just load the tensor
        tensor = torch.load(...)
        # Return (tensor, empty_dict) for EDM compatibility
        return tensor, {}

    def __len__(self):
        return len(self.files)
```

**Why:**
- Training only needs to load tensors
- All complexity is in preprocessing (where it belongs)
- Easier to debug and maintain
- More reusable across domains
- Clearer separation of concerns

---

## Documentation Improvements

### Updated Docstring

**Before:** Focused on domain-specific details (demosaicing, black level subtraction, etc.)

**After:** Clear separation between training input and inference concerns

```python
"""
Train unconditional diffusion model on preprocessed image tiles.

This script trains a domain-agnostic diffusion model using EDM's framework.
All domain-specific preprocessing (sensor calibration, Bayer packing,
normalization) must be done beforehand via the preprocessing pipeline.

Physics-informed guidance (heteroscedastic Poisson-Gaussian, sensor-specific
noise models) is applied at inference time, not during training.

TRAINING INPUT:
- Preprocessed .pt files containing float32 tensors in [-1, 1] range
- Clean/long-exposure images for restoration tasks
- No labels needed (unconditional training)
- No domain-specific handling required

INFERENCE (separate script):
- Load trained unconditional model
- Apply physics-informed likelihood gradients for guidance
- Use domain-specific noise models and sensor calibration
"""
```

### Added Clarifying Comment

```python
# Configure network, loss, and optimizer
# Note: Network is unconditional (no class conditioning)
network_kwargs = create_network_config(train_dataset, args)
```

---

## Configuration Changes

Updated default config path and output directory names for clarity:

```python
# BEFORE
default="config/photo.yaml"
default="results/edm_photography_training"

# AFTER
default="config/diffusion.yaml"
default="results/edm_diffusion_training"
```

**Why:** Reflects that this is domain-agnostic diffusion training, not photography-specific.

---

## Architecture Alignment with Paper

The refactored code now properly implements the paper's architecture:

1. **Training Phase** (this script)
   - Train unconditional EDM on clean long-exposure images
   - Learn P(x) distribution without any domain assumptions
   - No labels, no conditioning, no sensor-specific code

2. **Inference Phase** (separate script - to be created)
   - Load trained unconditional model
   - Apply heteroscedastic Poisson-Gaussian guidance: ∇ = (y - x̂) / (x̂ + σ_r²)
   - Use sensor-specific read noise calibration
   - Domain-specific physics lives here

---

## Summary of Changes

| Aspect | Before | After | Benefit |
|--------|--------|-------|---------|
| Labels | `use_labels=True` | `use_labels=False` | Correct unconditional training |
| Domain | `domain="photography"` | Removed | Domain-agnostic training |
| Dataset | `EDMPTDataset` (complex) | `SimplePTDataset` (simple) | Better separation of concerns |
| Import | `create_edm_pt_datasets` | `SimplePTDataset` | No unused imports |
| Docs | Domain-specific details | Clear architecture | Reflects separation of concerns |
| Output dir | `edm_photography_training` | `edm_diffusion_training` | Clarity of purpose |

---

## Key Principles Applied

1. **Separation of Concerns**: Each component has a single responsibility
   - Preprocessing: Domain-specific complexity
   - Training: Generic tensor loading and diffusion training
   - Inference: Physics-informed guidance

2. **Simplicity**: Code is as simple as possible while achieving the scientific goal
   - Removed unnecessary parameters
   - Removed unused imports
   - Removed domain coupling

3. **Reusability**: Training script works with any preprocessed tensors
   - Not tied to photography domain
   - Can be used for other modalities
   - Different sensor types need only different preprocessing

4. **Clarity**: Documentation and code structure make the architecture clear
   - Explicit that training is unconditional
   - Explicit where domain-specific code belongs (inference)
   - Easy for others to understand and extend

---

## Next Steps

1. **Create `SimplePTDataset`** in `data/dataset.py` if not already present
   - Ensure it returns `(tensor, {})` tuples for EDM compatibility
   - Load preprocessed .pt files from disk
   - Support train/validation split from metadata JSON

2. **Create inference script** that applies physics-informed guidance
   - Load trained model
   - Implement heteroscedastic likelihood gradient
   - Apply at inference time with proper sensor calibration

3. **Update config files**
   - Rename/update `config/photo.yaml` → `config/diffusion.yaml`
   - Ensure domain-specific parameters are in inference config, not training config
