# Domain-Adaptive Poisson-Gaussian Diffusion for Low-Light Image Restoration

A framework for low-light image restoration in photography using physically-correct Poisson-Gaussian noise modeling with diffusion models.

## Project Status

🚧 **Currently in development** - Phase 4: Evaluation & Inference

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
- ✅ **Phase 3 Complete**: Data Processing & Training
  - ✅ **Task 3.1**: EDM native training with float32 .pt (NO quantization)
  - ✅ **Task 3.2**: Photography preprocessing pipeline (34,212 tiles generated)
  - ✅ **Task 3.3**: Training data loader with physics metadata
  - ✅ **Task 3.4**: Photography model training
- 🚧 **Phase 4 In Progress**: Evaluation & Inference
  - ⏳ **Task 4.1**: Inference pipeline implementation
  - ⏳ **Task 4.2**: Baseline comparisons (BM3D, NLM, Deep learning methods)

## Key Features

- **Physics-Aware**: Exact Poisson-Gaussian noise modeling with domain-specific normalization
- **Full Precision**: Float32 .pt pipeline preserves all sensor information (NO quantization)
- **Fixed Scaling**: Domain-specific range normalization for stable training
- **Metadata-Rich**: Complete tile statistics and normalization parameters preserved

## Data Processing Overview

Our preprocessing pipeline (detailed in `preprocessing/DATA_PROCESSING.md`) converts raw sensor data from photography into unified 256×256 float32 tiles with domain-specific normalization. The pipeline implements:

- **Domain-specific range normalization**: [domain_min, domain_max] → [0,1] → [-1,1] (directly on raw ADU values)
- **Scene-based splitting**: 70/15/15 train/val/test with no data leakage
- **Systematic tiling**: Grid-based extraction with controlled overlap
- **Perfect pairing**: 1:1 clean-noisy correspondence for supervised training

### Dataset Statistics

| Domain | Unique Scenes | Clean/Noisy Pairs | Tiles per Image | Total Tiles | Tile Grid | Normalization Range |
|--------|---------------|-------------------|-----------------|-------------|-----------|---------------------|
| **Photography** | 424 | 424 (231 Sony + 193 Fuji) | 54 (Sony), 24 (Fuji) | 34,212 | Sony: 6×9<br>Fuji: 4×6 | [0, 15871] |

**Note**: Photography maintains perfect 1:1 clean-noisy pairing (long/short exposure).

## Architecture Overview

### Data Processing Pipeline

```
Preprocessing (process_tiles_pipeline.py):
Raw Sensor (ADU) → Domain Normalization → 256×256 tiles (.pt)
                         ↓                            ↓
           [domain_min, domain_max]       Float32 precision
                    → [0,1] → [-1,1]      NO quantization
           Photography: [0, 15871]        Metadata preserved

Training Pipeline (train_pt_edm_native.py):
Load .pt ([-1,1]) → EDM Diffusion → Physics Guidance → Train
         ↓                  ↓                  ↓              ↓
  Float32 tensor   Native EDM loss    Poisson-Gaussian   Full precision
  CHW format       + conditioning       noise model       No data loss
  Already normalized                                     Directly ready for EDM

Inference Pipeline:
Noisy Input → Load & Normalize → EDM Sampling → Inverse Scaling → Denoised Output
                                      ↓
                            Domain conditioning
                            + Physics guidance
```

## Recent Implementation Highlights

### Float32 .pt Training Pipeline (NVIDIA EDM)
- **No Quantization**: Full float32 precision from sensor to training (no uint8 conversion)
- **Direct Loading**: `torch.load()` → EDM model (following NVIDIA EDM patterns)
- **Fixed Domain Normalization**: Domain-specific range normalization prevents per-image variation
  - Photography: [0, 15871] → [0,1] → [-1,1]
- **Metadata Preservation**: All tile statistics and normalization parameters stored per tile

### Photography Preprocessing Pipeline
- **Systematic Tiling**: Domain-specific grid patterns with controlled overlap
  - Sony: 6×9 grid (54 tiles per image)
  - Fuji: 4×6 grid (24 tiles per image)
- **Clean/Noisy Pairing**: Automatic selection of paired samples
  - Photography: long exposure (clean) + short exposure (noisy)
- **Split Assignment**: Scene-based train/val/test splits (70/15/15) prevent data leakage

### EDM Native Training Integration
- **Native EDM Components**: Uses EDM's training loop, loss, optimizer, and EMA
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
│   └── dataset.py                     # Float32 .pt dataset loader (EDM-compatible)
├── preprocessing/                     # Data preprocessing pipelines
│   ├── process_tiles_pipeline.py     # Multi-domain tile extraction with physics calibration
│   ├── domain_processors.py          # Domain-specific image loaders
│   └── complete_systematic_tiling.py # Systematic tiling with overlap control
├── train_pt_edm_native.py            # EDM native training with float32 .pt
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

### Completed ✅
- [x] Project specification and design
- [x] External dependencies setup (EDM integration)
- [x] Core infrastructure implementation
- [x] Domain-specific physics calibration system
- [x] Photography preprocessing pipeline with fixed scaling
- [x] Float32 .pt dataset loader (no quantization)
- [x] EDM native training implementation
- [x] Training data preparation for photography (34,212 tiles)
- [x] Photography model training

### In Progress 🚧
- [ ] **Inference pipeline implementation** (physics-guided sampling)
- [ ] **Baseline comparisons** (BM3D, Non-Local Means, DnCNN, Noise2Noise, etc.)

### Future Work 📋
- [ ] Validation and metrics (PSNR, SSIM, domain-specific metrics)
- [ ] Physics-guided sampling optimization
- [ ] Comprehensive evaluation framework
- [ ] Real-world deployment pipeline

## Quick Start

### 1. Preprocess Data

```bash
# Process photography data with physics calibration and fixed scaling
cd preprocessing
python process_tiles_pipeline.py \
    --base_path /path/to/data \
    --max_files None  # Process all files

# Outputs:
# - dataset/processed/pt_tiles/photography/{data_type}/*.pt
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
```

### 3. Monitor Training

```bash
# View training logs
tail -f results/photography_training/log.txt

# View training stats
python -c "import json; [print(json.dumps(json.loads(line), indent=2)) for line in open('results/photography_training/stats.jsonl')]"
```

## Technical Details

### Data Format
- **Storage**: Float32 .pt files (256×256 tiles)
- **Channels**: 3 channels (RGB)
- **Value Range**: [-1, 1] after domain-specific normalization
- **Metadata**: JSON with normalization parameters, splits, and tile statistics

### Domain Normalization
```python
# Apply domain-specific range normalization to raw ADU values
# [domain_min, domain_max] → [0,1] → [-1,1]

# Photography: [0, 15871] → [0,1] → [-1,1]
normalized = (ADU - 0.0) / (15871.0 - 0.0)
normalized = np.clip(normalized, 0, 1)
normalized = 2 * normalized - 1
```

### Domain-Specific Parameters

| Domain | Sensor Type | Normalization Range | Tile Grid |
|--------|-------------|---------------------|-----------|
| Photography (Sony) | CCD/CMOS | [0, 15871] | 6×9 (54) |
| Photography (Fuji) | X-Trans | [0, 15871] | 4×6 (24) |

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
