# Multi-Resolution Diffusion Model Implementation Plan

## Overview

This document outlines the comprehensive step-by-step plan to enhance our Poisson-Gaussian diffusion model with full multi-resolution capabilities. The plan integrates the newly implemented multi-resolution components with the existing preprocessing pipeline and training infrastructure.

## Current State Analysis

### ✅ **Completed Components**
1. **Multi-Resolution Model Architectures** (`models/edm_wrapper.py`)
   - `ProgressiveEDM`: Progressive growing from 32×32 → 64×64 → 96×96 → 128×128
   - `MultiScaleEDM`: Hierarchical multi-scale processing at [32, 64, 96, 128]
   - Factory functions: `create_progressive_edm()`, `create_multiscale_edm()`

2. **Resolution Schedulers** (`poisson_training/schedulers.py`)
   - `ResolutionScheduler`: Manages progressive resolution growth
   - `AdaptiveResolutionScheduler`: Dynamic resolution adjustment based on metrics
   - Factory function: `create_resolution_scheduler()`

3. **Adaptive Resolution Management** (`core/transforms.py`)
   - `AdaptiveResolutionManager`: Intelligent resolution selection based on image characteristics
   - Noise estimation, detail analysis, and constraint-based optimization

4. **Multi-Resolution Training Configuration** (`poisson_training/multi_domain_trainer.py`)
   - `MultiDomainTrainingConfig`: Extended with multi-resolution settings
   - Resolution scheduler integration points

5. **Multi-Resolution Metrics** (`poisson_training/metrics.py`)
   - `MultiResolutionMetrics`: Performance evaluation across resolution levels
   - Progressive improvement tracking and resolution recommendations

6. **Updated Project Documentation**
   - **Design Document** (`.kiro/specs/poisson-diffusion-restoration/design.md`): Updated with 32→64→96→128 progression
   - **Requirements Document** (`.kiro/specs/poisson-diffusion-restoration/requirements.md`): Updated performance targets and specifications
   - **Tasks Document** (`.kiro/specs/poisson-diffusion-restoration/tasks.md`): Multi-resolution implementation roadmap

### 🔄 **Preprocessing Pipeline Compatibility**
- **Status**: ✅ **Fully Compatible**
- **Tile Size**: 128×128 (matches final resolution stage)
- **Data Format**: `.pt` files with `clean_norm` tensor `[C, 128, 128]`
- **Location**: `/home/jilab/Jae/preprocessing/` (Reference: `preprocessing/README.md`)

## Implementation Roadmap

### **Phase 1: Integration & Testing** (Week 1)

#### **Step 1.1: Integrate Multi-Resolution Models**
**Reference**: `models/edm_wrapper.py` lines 754-1266

**Action Items**:
```python
# Update model creation in training scripts
from models.edm_wrapper import ProgressiveEDM, MultiScaleEDM

# Progressive growing model
model = ProgressiveEDM(
    min_resolution=32,
    max_resolution=128,
    num_stages=4,
    model_channels=128
)

# Multi-scale model  
model = MultiScaleEDM(
    scales=[32, 64, 96, 128],
    model_channels=128
)
```

**Files to Update**:
- `train_photography_model.py`: Replace standard EDM with ProgressiveEDM
- `configs/default.yaml`: Add multi-resolution model configurations
- `utils/training_config.py`: Add multi-resolution parameter calculations

#### **Step 1.2: Test Multi-Resolution Components**
**Reference**: `models/edm_wrapper.py` lines 1158-1266

**Action Items**:
```bash
# Run comprehensive tests
python -c "
from models.edm_wrapper import test_progressive_edm, test_multiscale_edm
print('Testing ProgressiveEDM...')
assert test_progressive_edm()
print('Testing MultiScaleEDM...')
assert test_multiscale_edm()
print('✅ All multi-resolution tests passed!')
"
```

**Expected Results**:
- ProgressiveEDM processes inputs at all 4 resolution stages
- MultiScaleEDM extracts features from all scales
- Resolution growing works correctly (32→64→96→128)

#### **Step 1.3: Validate Preprocessing Compatibility**
**Reference**: `preprocessing/README.md` lines 127-138

**Action Items**:
```python
# Verify data pipeline compatibility
from data.preprocessed_datasets import PreprocessedPriorDataset

dataset = PreprocessedPriorDataset(
    data_root="data/preprocessed",
    domain="photography",
    split="train"
)

# Verify tile dimensions match final resolution
sample = dataset[0]
assert sample['clean'].shape[-2:] == (128, 128), "Tile size mismatch!"
print("✅ Preprocessing compatibility verified")
```

### **Phase 2: Training Pipeline Integration** (Week 2)

#### **Step 2.1: Update Multi-Domain Trainer**
**Reference**: `poisson_training/multi_domain_trainer.py` lines 91-97, 465-475

**Action Items**:
1. **Enable Multi-Resolution Training**:
```python
# Update training configuration
config = MultiDomainTrainingConfig(
    multi_resolution=True,
    resolution_scheduler_type="progressive",
    min_resolution=32,
    max_resolution=128,
    num_resolution_stages=4,
    epochs_per_resolution_stage=25
)
```

2. **Integrate Resolution Scheduler**:
```python
# In MultiDomainTrainer.__init__()
if self.config.multi_resolution:
    self.resolution_scheduler = self._setup_resolution_scheduler()
```

3. **Update Training Loop**:
```python
# In training loop
if self.config.multi_resolution:
    current_resolution = self.resolution_scheduler.get_current_resolution()
    # Resize inputs to current resolution
    inputs = F.interpolate(inputs, size=(current_resolution, current_resolution))
```

#### **Step 2.2: Adapt Data Loading for Multi-Resolution**
**Reference**: `data/preprocessed_datasets.py` lines 22-100

**Action Items**:
1. **Create Multi-Resolution Dataset Wrapper**:
```python
class MultiResolutionDataset:
    def __init__(self, base_dataset, resolution_scheduler):
        self.base_dataset = base_dataset
        self.resolution_scheduler = resolution_scheduler
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        current_res = self.resolution_scheduler.get_current_resolution()
        
        # Resize to current training resolution
        if current_res != 128:
            sample['clean'] = F.interpolate(
                sample['clean'].unsqueeze(0), 
                size=(current_res, current_res)
            ).squeeze(0)
        
        return sample
```

2. **Update Data Loaders**:
```python
# Wrap existing datasets with multi-resolution capability
train_dataset = MultiResolutionDataset(base_dataset, resolution_scheduler)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
```

#### **Step 2.3: Implement Progressive Training Logic**
**Reference**: `poisson_training/schedulers.py` lines 154-256

**Action Items**:
1. **Resolution Stage Management**:
```python
# In training loop
def train_epoch(self):
    # Check if we should grow resolution
    if self.resolution_scheduler.should_grow_resolution(self.current_epoch):
        old_res = self.resolution_scheduler.get_current_resolution()
        self.resolution_scheduler.step_epoch()
        new_res = self.resolution_scheduler.get_current_resolution()
        
        logger.info(f"Growing resolution: {old_res} → {new_res}")
        
        # Update model if using ProgressiveEDM
        if hasattr(self.model, 'grow_resolution'):
            self.model.grow_resolution()
```

2. **Batch Size Adaptation**:
```python
# Adjust batch size based on resolution
def get_adaptive_batch_size(self, resolution: int, base_batch_size: int = 32) -> int:
    # Scale batch size inversely with resolution area
    scale_factor = (128 / resolution) ** 2
    return min(int(base_batch_size * scale_factor), 128)
```

### **Phase 3: Advanced Multi-Resolution Features** (Week 3)

#### **Step 3.1: Implement Adaptive Resolution Selection**
**Reference**: `core/transforms.py` lines 549-728

**Action Items**:
1. **Integrate Adaptive Manager**:
```python
# In inference pipeline
adaptive_manager = AdaptiveResolutionManager(
    min_resolution=32,
    max_resolution=128
)

# Select optimal resolution for each input
optimal_res, analysis = adaptive_manager.select_optimal_resolution(
    image=input_image,
    constraints={
        'max_memory_gb': 4.0,
        'max_time_seconds': 10.0,
        'quality_preference': 'balanced'
    }
)
```

2. **Create Resolution-Aware Inference**:
```python
def adaptive_inference(self, noisy_image: torch.Tensor) -> torch.Tensor:
    # Analyze input and select resolution
    optimal_res, _ = self.adaptive_manager.select_optimal_resolution(noisy_image)
    
    # Process at optimal resolution
    if optimal_res != noisy_image.shape[-1]:
        resized_input = F.interpolate(noisy_image, size=(optimal_res, optimal_res))
        output = self.model(resized_input)
        # Resize back to original size
        output = F.interpolate(output, size=noisy_image.shape[-2:])
    else:
        output = self.model(noisy_image)
    
    return output
```

#### **Step 3.2: Implement Multi-Resolution Metrics**
**Reference**: `poisson_training/metrics.py` lines 639-728

**Action Items**:
1. **Integrate Evaluation Metrics**:
```python
# In validation loop
multi_res_metrics = MultiResolutionMetrics()

# Evaluate at all resolutions
results = multi_res_metrics.evaluate_at_resolutions(
    model=self.model,
    test_images=test_batch,
    ground_truth=gt_batch,
    resolutions=[32, 64, 96, 128]
)

# Log progressive improvements
for res, metrics in results.items():
    logger.info(f"Resolution {res}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.3f}")
```

2. **Resolution Recommendation System**:
```python
# Get resolution recommendations
recommendations = multi_res_metrics.recommend_resolution(
    test_images=validation_set,
    constraints={'max_memory_gb': 4.0, 'min_quality_psnr': 28.0}
)

logger.info(f"Recommended resolution: {recommendations['optimal_resolution']}")
```

### **Phase 4: Production Integration** (Week 4)

#### **Step 4.1: Update Training Scripts**
**Reference**: `train_photography_model.py` lines 290-291, 279-281

**Action Items**:
1. **Main Training Script**:
```python
# Update train_photography_model.py
def create_multi_resolution_model(config):
    if config.model_type == "progressive":
        return ProgressiveEDM(
            min_resolution=config.min_resolution,
            max_resolution=config.max_resolution,
            num_stages=config.num_resolution_stages,
            model_channels=config.model_channels
        )
    elif config.model_type == "multiscale":
        return MultiScaleEDM(
            scales=config.scales,
            model_channels=config.model_channels
        )
    else:
        # Fallback to standard EDM
        return EDMWrapper(...)
```

2. **Configuration Updates**:
```yaml
# configs/multi_resolution.yaml
model:
  type: "progressive"  # or "multiscale"
  min_resolution: 32
  max_resolution: 128
  num_stages: 4
  model_channels: 128

training:
  multi_resolution: true
  epochs_per_resolution_stage: 25
  resolution_scheduler_type: "progressive"
  adaptive_batch_sizing: true
```

#### **Step 4.2: Create Evaluation Scripts**
**Action Items**:

2. **Performance Benchmarking**:
```python
# Benchmark inference times and memory usage
def benchmark_multi_resolution():
    resolutions = [32, 64, 96, 128]
    batch_sizes = [1, 4, 8, 16]
    
    for res in resolutions:
        for bs in batch_sizes:
            time, memory = benchmark_inference(
                resolution=res, 
                batch_size=bs
            )
            print(f"Resolution {res}, Batch {bs}: {time:.2f}s, {memory:.1f}GB")
```

#### **Step 4.3: Documentation and Testing**
**Action Items**:
1. **Update README Documentation**:
```markdown
# Multi-Resolution Training

## Quick Start
```bash
# Train progressive model
python train_photography_model.py --config configs/multi_resolution.yaml

```

## Performance Targets
| Resolution | GPU Memory | Inference Time | Quality (PSNR) |
|------------|------------|----------------|----------------|
| 32×32      | 1 GB       | 0.1s          | 24-26 dB      |
| 64×64      | 2 GB       | 0.3s          | 26-28 dB      |
| 96×96      | 3 GB       | 0.6s          | 28-30 dB      |
| 128×128    | 4 GB       | 1.0s          | 30-32 dB      |
```

2. **Comprehensive Testing Suite**:
```python
# tests/test_multi_resolution_integration.py
def test_end_to_end_multi_resolution():
    # Test complete pipeline
    config = MultiDomainTrainingConfig(multi_resolution=True)
    trainer = MultiDomainTrainer(config)
    
    # Train for a few steps
    trainer.train_steps(10)
    
    # Verify resolution progression
    assert trainer.resolution_scheduler.current_stage >= 0
    
    # Test inference at all resolutions
    for res in [32, 64, 96, 128]:
        output = trainer.model.inference_at_resolution(test_input, res)
        assert output.shape[-2:] == (res, res)
```

## Success Criteria

### **Phase 1 Success Metrics**:
- [ ] All multi-resolution models pass unit tests
- [ ] Preprocessing compatibility verified (128×128 tiles)
- [ ] Memory usage within expected bounds (4-8GB for 128×128)

### **Phase 2 Success Metrics**:
- [ ] Progressive training completes 4 resolution stages
- [ ] Resolution transitions are smooth (no training instability)
- [ ] Batch size adaptation works correctly

### **Phase 3 Success Metrics**:
- [ ] Adaptive resolution selection improves efficiency by 2-3x
- [ ] Multi-resolution metrics show progressive improvement
- [ ] Quality scales appropriately with resolution (1-2 dB per stage)

### **Phase 4 Success Metrics**:
- [ ] Production training scripts work end-to-end
- [ ] Performance meets targets (see table above)
- [ ] Documentation is complete and accurate

## Risk Mitigation

### **Potential Issues & Solutions**:

1. **Memory Issues at Higher Resolutions**:
   - **Risk**: OOM errors during training
   - **Solution**: Implement gradient checkpointing, reduce batch size automatically

2. **Training Instability During Resolution Transitions**:
   - **Risk**: Loss spikes when growing resolution
   - **Solution**: Gradual transition with learning rate warmup

3. **Preprocessing Pipeline Bottlenecks**:
   - **Risk**: Data loading becomes slow with resolution changes
   - **Solution**: Pre-compute multi-resolution tiles, use efficient data loaders

4. **Model Architecture Incompatibilities**:
   - **Risk**: EDM wrapper doesn't handle variable resolutions
   - **Solution**: Extensive testing, fallback to standard EDM if needed

## Timeline Summary

| Week | Phase | Key Deliverables |
|------|-------|-----------------|
| 1 | Integration & Testing | Multi-resolution models integrated, tested |
| 2 | Training Pipeline | Progressive training working end-to-end |
| 3 | Advanced Features | Adaptive resolution, comprehensive metrics |
| 4 | Production | Scripts updated, documentation complete |

## Document References

- **Architecture**: `.kiro/specs/poisson-diffusion-restoration/design.md`
- **Requirements**: `.kiro/specs/poisson-diffusion-restoration/requirements.md`
- **Task Tracking**: `.kiro/specs/poisson-diffusion-restoration/tasks.md`
- **Preprocessing**: `preprocessing/README.md`
- **Model Implementation**: `models/edm_wrapper.py`
- **Training Configuration**: `poisson_training/multi_domain_trainer.py`
- **Schedulers**: `poisson_training/schedulers.py`
- **Metrics**: `poisson_training/metrics.py`
- **Transforms**: `core/transforms.py`

---

**Next Action**: Begin Phase 1, Step 1.1 - Update `train_photography_model.py` to use `ProgressiveEDM` model.
