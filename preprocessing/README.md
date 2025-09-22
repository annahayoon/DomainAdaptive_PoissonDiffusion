# Cross-Domain Data Preprocessing Pipeline

This module contains the complete data preprocessing pipeline for the domain-adaptive Poisson-Gaussian diffusion model. All preprocessing components are now properly organized under `cross_domain/data/` for better structure and maintainability.

## Directory Structure

```
cross_domain/data/
├── preprocessing/                    # Main preprocessing pipeline
│   ├── __init__.py                  # Module exports
│   ├── preprocessing_utils.py       # Core utilities (noise estimation, etc.)
│   ├── photography_processor.py     # Sony ARW/SID processing
│   ├── microscopy_processor.py      # MRC/Super-resolution processing
│   └── astronomy_processor.py       # FITS/HLF processing
├── preprocessed_datasets.py         # Dataset classes for .pt files
├── base_dataset.py                  # Backward-compatible base classes
└── __init__.py                      # Data module exports
```

## Testing Structure

```
cross_domain/tests/
├── preprocessing/                   # Preprocessing tests
│   ├── __init__.py
│   ├── test_core_utils.py          # Core utilities tests
│   ├── test_datasets.py            # Dataset integration tests
│   ├── test_real_data.py           # Real data processing tests
│   ├── test_validation.py          # Validation script tests
│   └── run_all_tests.py            # Main test runner
├── test_preprocessing_integration.py # Pytest-compatible tests
└── __init__.py
```

## Key Components

### Core Preprocessing Utilities (`preprocessing_utils.py`)
- **Noise Estimation**: Photon transfer curve analysis
- **Scale Computation**: Global normalization from clean images
- **Tile Extraction**: With augmentation support
- **Bayer Processing**: RAW sensor pattern handling
- **Background Estimation**: Morphological and percentile methods

### Domain Processors
- **Photography**: Sony ARW files → 4-channel RGGB format
- **Microscopy**: MRC files → single-channel with multi-level noise simulation
- **Astronomy**: FITS files → single-channel with cosmic ray detection

### Dataset Classes
- **PreprocessedPriorDataset**: Loads clean tiles for diffusion training
- **PreprocessedPosteriorDataset**: Loads full scenes for evaluation
- **MultiDomainDataset**: Balanced sampling across domains
- **PreprocessedDataLoader**: Convenient loader creation

## Usage

### Import Components
```python
# From preprocessing module
from cross_domain.data.preprocessing import (
    PhotographyProcessor,
    MicroscopyProcessor,
    AstronomyProcessor,
    estimate_noise_params_photon_transfer,
    compute_global_scale,
)

# From data module (convenience imports)
from cross_domain.data import (
    PhotographyProcessor,
    PreprocessedPriorDataset,
    PreprocessedDataLoader,
)
```

### Run Preprocessing
```bash
# Process all domains
python cross_domain/scripts/preprocess_all_domains.py \
    --output_root data/preprocessed \
    --photography_root data/raw/SID \
    --microscopy_root data/raw/microscopy \
    --astronomy_root data/raw/HLF

# Process single domain
python -m cross_domain.data.preprocessing.photography_processor \
    --raw_root data/raw/SID \
    --output_root data/preprocessed
```

### Run Tests
```bash
# Comprehensive test suite
python cross_domain/tests/preprocessing/run_all_tests.py

# Pytest-compatible
pytest cross_domain/tests/test_preprocessing_integration.py -v

# Individual test modules
python cross_domain/tests/preprocessing/test_core_utils.py
python cross_domain/tests/preprocessing/test_datasets.py
```

### Load Preprocessed Data
```python
from cross_domain.data import PreprocessedDataLoader

# Prior training
train_loader = PreprocessedDataLoader.create_prior_loader(
    root="data/preprocessed",
    domains=["photography", "microscopy"],
    batch_size=32,
    balance_domains=True
)

# Posterior evaluation
eval_loader = PreprocessedDataLoader.create_posterior_loader(
    root="data/preprocessed",
    domains=["photography"],
    split="test",
    batch_size=1
)
```

## File Formats

### Prior Training Tiles (`.pt` files)
```python
{
    'clean_norm': torch.Tensor,      # [C, 128, 128] normalized clean image
    'domain_id': int,                # 0=photo, 1=micro, 2=astro
    'metadata': {
        'scene_id': str,             # Original scene identifier
        'tile_idx': int,             # Tile index
        'augmented': bool,           # Augmentation applied
    }
}
```

### Posterior Scenes (`.pt` files)
```python
{
    'noisy_norm': torch.Tensor,      # [C, H, W] normalized noisy image
    'clean_norm': torch.Tensor,      # [C, H, W] normalized clean (or None)
    'calibration': {                 # Sensor calibration parameters
        'scale': float,              # Normalization scale (electrons)
        'gain': float,               # e-/ADU conversion
        'read_noise': float,         # Read noise (electrons)
        'background': float,         # Background level
    },
    'masks': {                       # Pixel masks
        'valid': torch.Tensor,       # [1, H, W] valid pixels
        'saturated': torch.Tensor,   # [1, H, W] saturated pixels
    },
    'metadata': {...}                # Domain-specific metadata
}
```

## Benefits of New Organization

1. **Better Structure**: All data-related code under `cross_domain/data/`
2. **Modular Testing**: Focused test modules for different aspects
3. **Clean Imports**: Logical import paths and convenience exports
4. **Maintainability**: Easier to find and modify specific components
5. **Extensibility**: Simple to add new domains or processors

## Migration from Old Structure

The reorganization maintains backward compatibility:
- Old imports still work through convenience re-exports
- Existing scripts automatically detect new structure
- Base dataset classes seamlessly use preprocessed data when available

This ensures existing code continues to work while providing the improved organization.
