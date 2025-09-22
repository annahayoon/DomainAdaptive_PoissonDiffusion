# Cross-Domain Preprocessing Pipeline

This directory contains a comprehensive preprocessing pipeline for the domain-adaptive Poisson-Gaussian diffusion model. The pipeline handles three imaging domains: photography (SID dataset), microscopy (fluorescence), and astronomy (SDSS).

## Overview

The preprocessing pipeline converts raw imaging data into standardized `.pt` files that can be efficiently loaded during training and evaluation. This approach ensures:

- **Reproducible preprocessing**: Run once, use many times
- **Fast training**: No on-the-fly processing overhead
- **Consistent normalization**: Domain-specific scaling and calibration
- **Quality validation**: Built-in checks for data integrity

## Architecture

```
Raw Data → Preprocessing → .pt Files → Model Training
```

### Directory Structure

```
data/
├── raw/                          # Original downloads
│   ├── SID/Sony/                # Photography (ARW files)
│   ├── FMD/                     # Microscopy (TIFF files)
│   └── SDSS/                    # Astronomy (FITS files)
└── preprocessed/                # Output of pipeline
    ├── manifests/               # Global statistics
    ├── prior_clean/            # Clean tiles for diffusion training
    │   ├── photography/train/   # tile_000000.pt to tile_NNNNNN.pt
    │   ├── microscopy/train/
    │   └── astronomy/train/
    └── posterior/              # Full scenes for evaluation
        ├── photography/train/   # scene_00000.pt to scene_NNNNN.pt
        ├── microscopy/train/
        └── astronomy/train/
```

## File Formats

### Prior Training Tiles (`prior_clean/{domain}/{split}/tile_{idx}.pt`)

Each file contains one 128×128 clean tile:

```python
{
    'clean_norm': torch.Tensor,      # [C, 128, 128] normalized image
    'domain_id': int,                # 0=photo, 1=micro, 2=astro
    'metadata': {
        'scene_id': str,             # Original image identifier
        'tile_idx': int,             # Tile number within dataset
        'augmented': bool,           # Was augmentation applied
    }
}
```

### Posterior Scenes (`posterior/{domain}/{split}/scene_{id}.pt`)

Each file contains one complete scene:

```python
{
    'noisy_norm': torch.Tensor,      # [C, H, W] normalized noisy image
    'clean_norm': torch.Tensor,      # [C, H, W] normalized GT (or None)
    'calibration': {
        'scale': float,              # Normalization scale (electrons)
        'gain': float,               # e-/ADU conversion
        'read_noise': float,         # Read noise (electrons)
        'background': float,         # Background level (electrons)
        'black_level': float,        # Sensor black level (ADU)
        'white_level': float,        # Sensor white level (ADU)
    },
    'masks': {
        'valid': torch.Tensor,       # [1, H, W] valid pixel mask
        'saturated': torch.Tensor,   # [1, H, W] saturation mask
    },
    'metadata': {...}                # Domain-specific metadata
}
```

## Usage

### 1. Environment Setup

```bash
pip install torch numpy scipy scikit-learn scikit-image
pip install rawpy tifffile astropy  # Domain-specific dependencies
pip install matplotlib  # For visualization
```

### 2. Run Preprocessing

```bash
# Process all domains
python cross_domain/scripts/preprocess_all_domains.py \
    --output_root data/preprocessed \
    --photography_root data/raw/SID \
    --microscopy_root data/raw/FMD \
    --astronomy_root data/raw/SDSS

# Process single domain
python cross_domain/core/photography_processor.py \
    --raw_root data/raw/SID \
    --output_root data/preprocessed
```

### 3. Validate Results

```bash
python cross_domain/scripts/validate_preprocessing.py data/preprocessed
```

### 4. Inspect Data

```bash
# Inspect a training tile
python cross_domain/scripts/inspect_data.py \
    data/preprocessed/prior_clean/photography/train/tile_000000.pt

# Inspect a full scene
python cross_domain/scripts/inspect_data.py \
    data/preprocessed/posterior/microscopy/train/seq001_frame001.pt
```

### 5. Use in Training

```python
from cross_domain.data import PreprocessedDataLoader

# Create data loader for prior training
train_loader = PreprocessedDataLoader.create_prior_loader(
    root="data/preprocessed",
    domains=["photography", "microscopy"],
    split="train",
    batch_size=32,
    balance_domains=True
)

# Create data loader for evaluation
eval_loader = PreprocessedDataLoader.create_posterior_loader(
    root="data/preprocessed",
    domains=["photography"],
    split="test",
    batch_size=1
)
```

## Domain-Specific Processing

### Photography (SID Dataset)

- **Input**: Sony ARW raw files (short/long exposure pairs)
- **Processing**: Bayer demosaicing → 4-channel RGGB format
- **Noise Model**: Photon transfer curve estimation
- **Normalization**: 99.9th percentile of clean images in electrons
- **Output**: ~50 tiles per scene, 4-channel images

### Microscopy (Fluorescence)

- **Input**: TIFF image pairs (low/high SNR or noisy/clean)
- **Processing**: Flat-field correction → single channel
- **Noise Model**: Photon transfer or read noise estimation
- **Normalization**: 99.9th percentile of clean images
- **Output**: ~20 tiles per frame, 1-channel images

### Astronomy (SDSS)

- **Input**: FITS frames (fpC format)
- **Processing**: Cosmic ray detection → single channel
- **Noise Model**: Header-based calibration (gain ~4.6, read noise ~4.7)
- **Normalization**: 99.9th percentile of frame intensities
- **Output**: ~30 tiles per frame, 1-channel images

## Key Features

### Noise Parameter Estimation

- **Photon Transfer Curve**: Robust estimation of gain and read noise
- **Outlier Rejection**: Huber regression for robustness
- **Validation**: Chi-squared tests on noise model

### Data Quality Assurance

- **Format Validation**: Tensor shapes, value ranges, required keys
- **Statistical Validation**: Normalization quality, noise model fit
- **Split Validation**: No scene overlap between train/val/test

### Efficient Storage

- **Preprocessing Once**: No on-the-fly processing during training
- **Optimized Formats**: PyTorch tensors with proper dtypes
- **Metadata Preservation**: Full calibration and provenance tracking

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "No scenes found" | Check raw data directory structure |
| "Negative values after black subtraction" | Verify black level calibration |
| "Chi-squared not near 1.0" | Check gain/read noise estimation |
| "Poor normalization quality" | Verify scale computation from clean data |
| "Import errors" | Install required dependencies (rawpy, tifffile, astropy) |

### Debug Mode

```bash
python cross_domain/scripts/preprocess_all_domains.py \
    --debug \
    --photography_root data/raw/SID \
    --output_root data/preprocessed
```

## Extension

To add a new domain:

1. Create `{domain}_processor.py` in `cross_domain/core/`
2. Implement noise estimation and normalization for your data format
3. Add to `preprocess_all_domains.py`
4. Update validation and inspection scripts
5. Add domain-specific tests

The modular design makes it easy to extend to new imaging modalities while maintaining consistent interfaces and data formats.
