# Domain-Adaptive Poisson-Gaussian Diffusion for Low-Light Image Restoration

A unified framework for low-light image restoration across photography, microscopy, and astronomy domains using physically-correct Poisson-Gaussian noise modeling with diffusion models.

## Project Status

🚧 **Currently in development** - Phase 3: Model Training & Validation

## Recent Progress

- ✅ **Phase 0 Complete**: Prerequisites & Dependencies
  - ✅ **Task 0.1**: EDM external dependencies integrated successfully
  - ✅ **Task 0.2**: Development environment fully configured
- ✅ **Phase 1 Complete**: Core Infrastructure
  - ✅ **Task 1.1**: Project structure and base interfaces
  - ✅ **Task 1.2**: Error handling framework
  - ✅ **Task 1.3**: Reversible transforms with tests
- ✅ **Phase 2 Complete**: Physics Foundation
  - ✅ **Task 2.1**: Domain-specific physics calibration system
  - ✅ **Task 2.2**: Poisson-Gaussian noise modeling
  - ✅ **Task 2.3**: Complete preprocessing pipeline with fixed scaling
- 🚧 **Phase 3 In Progress**: Model Training & Validation
  - ✅ **Task 3.1**: EDM native training with float32 .npy (NO quantization)
  - ✅ **Task 3.2**: Multi-domain preprocessing pipeline
  - ✅ **Task 3.3**: Training data loader with physics metadata
  - ⏳ **Task 3.4**: Cross-domain training and validation

## Key Features

- **Physics-Aware**: Exact Poisson-Gaussian noise modeling with domain-specific calibration
- **Cross-Domain**: Unified model architecture for photography, microscopy, and astronomy
- **Full Precision**: Float32 .npy pipeline preserves all sensor information (NO quantization)
- **Fixed Scaling**: Domain-specific physics-based normalization for stable training
- **Metadata-Rich**: Complete calibration parameters and physics information preserved

## Architecture Overview

### Data Processing Pipeline

```
Preprocessing (process_tiles_pipeline.py):
Raw Sensor (ADU) → Physics Calibration (ADU→electrons) → Fixed Scaling ([0,~1]) → 256×256 tiles (.npy)
                         ↓                                        ↓                            ↓
                  Domain-specific gain/noise          Photography: ÷80000           Float32 precision
                                                     Microscopy: ÷66000            NO quantization
                                                     Astronomy: (+5.0)÷170         Metadata preserved

Training Pipeline (train_npy_edm_native.py):
Load .npy ([0,~1]) → Normalize to [-1,1] → EDM Diffusion → Physics Guidance → Train
         ↓                    ↓                    ↓                  ↓              ↓
  Float32 tensor      x * 2 - 1           Native EDM loss    Poisson-Gaussian   Full precision
  CHW format          For EDM             + conditioning       noise model       No data loss

Inference Pipeline:
Noisy Input → Load & Normalize → EDM Sampling → Inverse Scaling → Denoised Output
                                      ↓
                            Domain conditioning
                            + Physics guidance
```

## Recent Implementation Highlights

### Float32 .npy Training Pipeline (NVIDIA EDM)
- **No Quantization**: Full float32 precision from sensor to training (no uint8 conversion)
- **Direct Loading**: `np.load()` → `torch.from_numpy()` → EDM model (following NVIDIA EDM patterns)
- **Fixed Domain Scaling**: Physics-based normalization prevents per-image variation
  - Photography (Sony): gain 5.0 e⁻/ADU, scale by 80000
  - Photography (Fuji): gain 1.8 e⁻/ADU, scale by 80000
  - Microscopy: gain 1.0 e⁻/ADU, scale by 66000
  - Astronomy: gain 1.0 e⁻/ADU, offset +5.0, scale by 170
- **Metadata Preservation**: All calibration parameters (gain, read_noise, electron statistics) stored per tile

### Multi-Domain Preprocessing Pipeline
- **Systematic Tiling**: Domain-specific grid patterns with controlled overlap
  - Sony: 6×9 grid (54 tiles per image)
  - Fuji: 4×6 grid (24 tiles per image)
  - Microscopy: 4×4 grid (16 tiles per image)
  - Astronomy: 9×9 grid (81 tiles per image)
- **Clean/Noisy Pairing**: Automatic selection of paired samples
  - Photography: long exposure (clean) + short exposure (noisy)
  - Microscopy: SIM_gt (clean) + RawSIMData_gt (noisy)
  - Astronomy: detection_sci (clean) + g800l_sci (noisy)
- **Split Assignment**: Scene-based train/val/test splits (70/15/15) prevent data leakage

### EDM Native Training Integration
- **Native EDM Components**: Uses EDM's training loop, loss, optimizer, and EMA
- **Domain Conditioning**: One-hot domain encoding for cross-domain learning
- **Early Stopping**: Validation-based early stopping to prevent overfitting
- **Checkpoint Resume**: Automatic resume from latest checkpoint

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- PyTorch 2.0+

### External Dependencies

This project integrates with the EDM (Elucidating the Design Space of Diffusion-Based Generative Models) codebase. See `external/README.md` for integration details.

## Project Structure

```
Jae/
├── core/                              # Core algorithms and transforms
│   ├── calibration.py                # Domain-specific sensor calibration
│   ├── transforms.py                 # Reversible image transforms with metadata
│   └── logging_config.py            # Unified logging configuration
├── data/                              # Data loading and datasets
│   └── npy_dataset.py                # Float32 .npy dataset loader (EDM-compatible)
├── preprocessing/                     # Data preprocessing pipelines
│   ├── process_tiles_pipeline.py     # Multi-domain tile extraction with physics calibration
│   ├── domain_processors.py          # Domain-specific image loaders
│   └── complete_systematic_tiling.py # Systematic tiling with overlap control
├── train_npy_edm_native.py           # EDM native training with float32 .npy
├── external/                          # External dependencies
│   ├── edm/                          # NVIDIA EDM implementation
│   └── README.md                     # Integration documentation
├── dataset/                           # Processed datasets
│   └── processed/
│       ├── npy_tiles/                # Float32 256×256 tiles
│       └── metadata_*.json           # Tile metadata with calibration info
├── results/                           # Training results and checkpoints
└── docs/                             # Documentation
    └── DAPGD_Implementation.md       # Complete implementation guide
```

## Development Status

- [x] Project specification and design
- [x] External dependencies setup (EDM integration)
- [x] Core infrastructure implementation
- [x] Domain-specific physics calibration system
- [x] Multi-domain preprocessing pipeline with fixed scaling
- [x] Float32 .npy dataset loader (no quantization)
- [x] EDM native training implementation
- [x] Training data preparation for all domains
- [ ] Complete cross-domain training run
- [ ] Physics-guided sampling implementation
- [ ] Cross-domain validation and evaluation
- [ ] Comprehensive evaluation framework

## Quick Start

### 1. Preprocess Data

```bash
# Process all domains with physics calibration and fixed scaling
cd preprocessing
python process_tiles_pipeline.py \
    --base_path /path/to/data \
    --max_files None  # Process all files

# Outputs:
# - dataset/processed/npy_tiles/{domain}/{data_type}/*.npy
# - dataset/processed/comprehensive_tiles_metadata.json
```

### 2. Train Model

```bash
# Train on photography domain
python train_npy_edm_native.py \
    --data_root dataset/processed/npy_tiles/photography \
    --metadata_json dataset/processed/metadata_photography_incremental.json \
    --channels 3 \
    --batch_size 64 \
    --total_kimg 10000 \
    --output_dir results/photography_training

# Train on microscopy domain (grayscale)
python train_npy_edm_native.py \
    --data_root dataset/processed/npy_tiles/microscopy \
    --metadata_json dataset/processed/metadata_microscopy_incremental.json \
    --channels 1 \
    --batch_size 64 \
    --total_kimg 10000 \
    --output_dir results/microscopy_training

# Train on astronomy domain (grayscale)
python train_npy_edm_native.py \
    --data_root dataset/processed/npy_tiles/astronomy \
    --metadata_json dataset/processed/metadata_astronomy_incremental.json \
    --channels 1 \
    --batch_size 64 \
    --total_kimg 10000 \
    --output_dir results/astronomy_training
```

### 3. Monitor Training

```bash
# View training logs
tail -f results/{domain}_training/log.txt

# View training stats
python -c "import json; [print(json.dumps(json.loads(line), indent=2)) for line in open('results/{domain}_training/stats.jsonl')]"
```

## Technical Details

### Data Format
- **Storage**: Float32 .npy files (256×256 tiles)
- **Channels**:
  - Photography: 3 channels (RGB)
  - Microscopy: 1 channel (grayscale)
  - Astronomy: 1 channel (grayscale)
- **Value Range**: [0, ~1] after fixed domain-specific scaling
- **Metadata**: JSON with calibration parameters, splits, and physics info

### Physics Calibration
```python
# Step 1: Convert ADU to electrons
electrons = ADU * gain - read_noise

# Step 2: Apply domain-specific fixed scaling
# Photography
scaled = electrons / 80000.0

# Microscopy
scaled = electrons / 66000.0

# Astronomy (with offset for negative values)
scaled = (electrons + 5.0) / 170.0
```

### Training Normalization
```python
# Pipeline output: [0, ~1] range
# EDM expects: [-1, 1] range
normalized = images * 2.0 - 1.0
```

### Domain-Specific Parameters

| Domain | Sensor Type | Gain (e⁻/ADU) | Read Noise (e⁻) | Fixed Scale | Tile Grid |
|--------|-------------|---------------|-----------------|-------------|-----------|
| Photography (Sony) | CCD/CMOS | 5.0 | 3.56 | 80000 | 6×9 (54) |
| Photography (Fuji) | X-Trans | 1.8 | 3.75 | 80000 | 4×6 (24) |
| Microscopy | sCMOS | 1.0 | 1.5 | 66000 | 4×4 (16) |
| Astronomy (HST) | CCD | 1.0 | 3.5 | 170 | 9×9 (81) |

## Citation

```bibtex
@article{poisson-diffusion-2025,
  title={Domain-Adaptive Poisson-Gaussian Diffusion for Low-Light Image Restoration},
  author={[Yoon, Ha Yun Anna Yoon; Hong, Jaewan
  journal={[Journal]},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
