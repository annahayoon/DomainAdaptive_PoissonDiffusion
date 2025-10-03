# Data Processing Pipeline Documentation

**Updated: October 2025, Anna Yoon**

## Overview

The system processes multi-domain imaging data (Photography, Microscopy, Astronomy) into unified **8-bit PNG tiles (256×256)** with domain-specific physics-based calibration, scene-based splitting, and complete metadata tracking using a **file-based pipeline**.

### Key Features

- ✅ **Local file-based processing** - No Spark/Parquet/Delta Lake dependencies
- ✅ **Domain-specific physics calibration** applied to full image BEFORE tiling
- ✅ **8-bit PNG storage** with lossless compression (~2-4x space savings)
- ✅ **Consistent bit depth** (8-bit) across all domains for cross-domain training
- ✅ **Camera-specific photography downsampling** with anti-aliasing
- ✅ **Correct astronomy labels**: Direct image = clean, G800L grism = noisy
- ✅ **Professional astronomy preprocessing** with astronomy_asinh scaling module
- ✅ **Scene-based train/test/validation splitting** with no data leakage

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Data Organization](#data-organization)
4. [Processing Flow](#processing-flow)
5. [Domain-Specific Processing](#domain-specific-processing)
6. [Physics-Based Calibration](#physics-based-calibration)
7. [Scene-Based Splitting](#scene-based-splitting)
8. [Data Schema](#data-schema)
9. [Training Integration](#training-integration)
10. [Verification](#verification)
11. [File Reference](#file-reference)

---

## Quick Start

### Demo Mode (Recommended First)
```bash
cd /home/jilab/anna_OS_ML
python process_tiles_pipeline.py --demo_mode
```

### Test Run (Limited Files)
```bash
python process_tiles_pipeline.py --max_files 3
```

### Full Production Run
```bash
python process_tiles_pipeline.py
```

### Expected Processing Time
- **Demo mode**: ~1-2 minutes (calibration examples only)
- **Test run** (3 files/domain): ~5-10 minutes
- **Full run** (all files): ~2-4 hours
- **Total output**: 78,806 tiles (39,403 clean + 39,403 noisy, perfect 1:1 mapping)

### Expected Tile Counts (1:1 Clean ↔ Noisy Mapping)
- **Photography**: 17,106 clean + 17,106 noisy = 34,212 tiles
  - Sony: 231 pairs × 54 tiles = 12,474 clean + 12,474 noisy
  - Fuji: 193 pairs × 24 tiles = 4,632 clean + 4,632 noisy
- **Microscopy**: 9,904 clean + 9,904 noisy = 19,808 tiles
  - 619 pairs × 16 tiles = 9,904 per type
  - Standard cells (211): RawSIMData_gt (noisy) ↔ SIM_gt/SIM_gt_a (clean)
  - ER cells (408 = 68×6): RawGTSIMData_level_XX (noisy) ↔ GTSIM_level_XX (clean)
- **Astronomy**: 12,393 clean + 12,393 noisy = 24,786 tiles
  - 153 pairs × 81 tiles = 12,393 per type
  - g800l_sci (noisy) ↔ detection_sci (clean)

---

## Pipeline Architecture

### Simple PNG Tiles Processing Flow

The pipeline processes images in the correct order for physics-based calibration:

```
1. Load full image (RAW sensor data in ADU units)
   ↓
2. Apply camera-specific photography downsampling (with anti-aliasing)
   • Sony: 2848×4256 → 1424×2128 (2x factor)
   • Fuji: 4032×6032 → 1008×1508 (4x factor)
   • Microscopy/Astronomy: No downsampling
   ↓
3. Apply domain-specific physics-based calibration (ADU → electrons)
   • Photography: Camera-specific calibration (Sony/Fuji)
   • Microscopy: sCMOS detector calibration
   • Astronomy: HST instrument calibration
   • Formula: electrons = ADU × gain - read_noise
   ↓
4. Apply domain-specific scaling for dynamic range compression
   • Photography: Linear scaling
   • Microscopy: Percentile-based for dark images
   • Astronomy: Asinh scaling with percentile fallback
   ↓
5. Normalize & Convert to 8-bit PNG [0, 255]
   ↓
6. Tile the calibrated 8-bit image into 256×256 PNG patches
   • Custom tiling for Sony (2×4 = 8 tiles)
   • Systematic tiling for Fuji (naturally achieves 4×6 = 24 tiles)
   • Systematic tiling for microscopy/astronomy (0% overlap)
   ↓
7. Separate prior (clean) vs posterior (noisy)
   ↓
8. Assign train/test/validation splits (scene-aware)
   ↓
9. Save PNG tiles with metadata
```

---

## Data Organization

### Raw Data Structure
```
PKL-DiffusionDenoising/data/raw/
├── SID/                    # Photography domain
│   ├── Sony/short/         # Noisy (low exposure)
│   ├── Sony/long/          # Clean (high exposure)
│   ├── Fuji/short/         # Noisy
│   └── Fuji/long/          # Clean
├── microscopy/structures/  # Microscopy domain
│   ├── CCPs/
│   ├── ER/
│   ├── F-actin/
│   └── Microtubules/
│       └── Cell_*/
│           ├── RawSIMData_gt.mrc          # Clean
│           └── RawSIMData_level_*.mrc     # Noisy
└── astronomy/hla_associations/  # Astronomy domain
    ├── direct_images/      # Noisy observations
    └── g800l_images/       # Clean spectroscopic
```

### Processed Data Structure
```
PKL-DiffusionDenoising/data/processed/
├── png_tiles/              # 8-bit PNG tiles organized by domain and type
│   ├── photography/
│   │   ├── noisy/         # Noisy photography tiles
│   │   └── clean/         # Clean photography tiles
│   ├── microscopy/
│   │   ├── noisy/         # Noisy microscopy tiles
│   │   └── clean/         # Clean microscopy tiles
│   └── astronomy/
│       ├── noisy/         # Noisy astronomy tiles
│       └── clean/         # Clean astronomy tiles
└── tiles_metadata.json    # Complete metadata for all tiles
```

---

## Processing Flow

### Core Processing Steps

```python
# STEP 1: Load full image
image, metadata = load_image(file_path)  # RAW ADU values

# STEP 2: Apply camera-specific photography downsampling (with anti-aliasing)
if domain == "photography" and self.downsample_photography:
    if file_path.endswith('.ARW'):
        image = _downsample_with_antialiasing(image, 2.0)  # Sony: 2848×4256 → 1424×2128
    elif file_path.endswith('.RAF'):
        image = _downsample_with_antialiasing(image, 4.0)  # Fuji: 4032×6032 → 1008×1508

# STEP 3: Apply domain-specific physics-based calibration
gain, read_noise, method = _get_physics_based_calibration(domain, metadata)
calibrated_image = _apply_physics_calibration(image, gain, read_noise, domain)
# Formula: electrons = (ADU × gain) - read_noise

# STEP 4: Apply domain-specific scaling
if domain == "photography":
    # Linear scaling for photography
    img_min, img_max = np.min(calibrated_image), np.max(calibrated_image)
    scaled_image = (calibrated_image - img_min) / (img_max - img_min)
elif domain == "astronomy":
    # Professional asinh scaling for astronomy (default)
    try:
        from astronomy_asinh_preprocessing import AstronomyAsinhPreprocessor
        scaled_image = astro_preprocessor.preprocess_astronomy_image(calibrated_image)
    except ImportError:
        # Fallback to percentile-based scaling if module unavailable
        p1, p99 = np.percentile(calibrated_image, [1, 99])
        scaled_image = np.clip((calibrated_image - p1) / (p99 - p1), 0, 1)
else:
    # Percentile-based scaling for microscopy
    p1, p99 = np.percentile(calibrated_image, [1, 99])
    scaled_image = np.clip((calibrated_image - p1) / (p99 - p1), 0, 1)

# STEP 5: Convert to 8-bit PNG
image_8bit = np.clip(scaled_image * 255.0, 0, 255).astype(np.uint8)

# STEP 6: Tile the 8-bit image
if domain == "photography" and file_path.endswith('.ARW'):
    # Custom Sony tiling (2×4 = 8 tiles)
    tiles = _extract_sony_tiles(image_8bit)
else:
    # Systematic tiling for Fuji/microscopy/astronomy (0% overlap)
    # Fuji: 1008×1508 → 4×6 = 24 tiles (systematic tiling naturally achieves this)
    tiler = SystematicTiler(tile_size=256, overlap_ratio=0.0)
    tiles = tiler.extract_tiles(image_8bit)

# STEP 7-9: Classify, split, and save as PNG
for i, tile_info in enumerate(tiles):
    data_type = _determine_data_type(file_path, domain)  # clean/noisy
    scene_id = _get_scene_id(file_path, domain)
    split = _assign_split(scene_id, data_type)  # train/val/test

    # Save as PNG file
    tile_id = f"{domain}_{Path(file_path).stem}_tile_{i:04d}"
    save_tile_as_png(tile_info.tile_data, tile_id, domain, data_type, gain, read_noise, method)
```

---

## Domain-Specific Processing

### Photography: Camera-Specific Downsampling + Linear Scaling
- **Camera-specific downsampling with anti-aliasing**: Sony (2x), Fuji (1.96x) prevents aliasing artifacts
- **Linear scaling**: `(x - min) / (max - min)` preserves original characteristics
- **RGB demosaicing**: Basic RGGB → RGB conversion (no white balance)
- **Appropriate for moderate dynamic ranges**

### Microscopy: Percentile-Based Scaling
- **No downsampling**: Preserves fine cellular details
- **Percentile normalization**: Uses 1st-99th percentiles for dark images
- **Handles very low signals**: Common in fluorescence microscopy
- **Preserves biological structures**

### Astronomy: Professional Asinh Scaling
- **No downsampling**: Preserves astronomical features
- **Professional asinh scaling**: Uses `astronomy_asinh_preprocessing` module (default)
- **Fallback percentile normalization**: For very dark astronomical images when module unavailable
- **Handles extreme dynamic ranges**: From background noise to bright stars
- **Preserves faint astronomical features**
- **Cosmic ray detection and removal**: Integrated preprocessing step

---

## Physics-Based Calibration

### Photography Domain

**Sony A7S II**:
- Gain: 2.1 e⁻/ADU (ISO 2000), 0.79 e⁻/ADU (ISO 4000, unity gain)
- Read noise: 2.5 e⁻ (above ISO 4000), 6.0 e⁻ (below ISO 4000)
- Resolution: 2848×4256 → 1424×2128 (2x downsampling) → 2×4 = 8 tiles

**Fuji X-T30**:
- Gain: 0.75-1.8 e⁻/ADU (base ISO to high ISO)
- Read noise: 2.5-3.75 e⁻
- Resolution: 4032×6032 → 1008×1508 (4x downsampling + 2.1% overlap) → 4×6 = 24 tiles

### Microscopy Domain

**BioSR sCMOS**:
- Gain: 1.0 e⁻/ADU (typical range 0.5-1.5)
- Read noise: 1.5 e⁻ RMS (typical range 1-2)
- Resolution: 502×502 → 256×256 tiles
- Photon counts: 15-600 photons depending on noise level

### Astronomy Domain (Hubble Legacy Archive)

**HST Instruments**:
- **ACS/WFC**: Gain 1.0 e⁻/DN, Read noise 3.5 e⁻
- **WFC3/UVIS**: Gain 1.5 e⁻/DN, Read noise 3.09 e⁻
- **WFC3/IR**: Gain 2.5 e⁻/DN, Read noise 15.0 e⁻
- **WFPC2**: Gain 7.0 e⁻/DN, Read noise 6.5 e⁻
- Resolution: ~4232×4220 → 256×256 tiles

**Image Classification**:
- **Direct Image** (detection_sci.fits): Clean reference image
  - Standard photometric filter
  - High signal-to-noise ratio
  - True spatial appearance
- **G800L Grism** (g800l_sci.fits): Noisy spectroscopic image
  - Slitless spectroscopy mode
  - Lower SNR with spectral dispersion
  - Contamination from overlapping spectra

---

## Scene-Based Splitting

### Split Distribution

| Split | Data Source | Percentage | Purpose |
|-------|-------------|------------|---------|
| **Train** | Noisy only | ~70% | Learn denoising |
| **Validation** | Noisy only | ~15% | Tune hyperparameters |
| **Test** | Clean only | ~15% | Final evaluation |

### Scene Grouping Logic

**Photography**: Exposure pairs share same scene ID
- `00001_00_0.04s.ARW` (noisy) → `photo_00001` → train/validation
- `00001_00_10s.ARW` (clean) → `photo_00001` → test

**Microscopy**: Cell and noise level grouping
- `Cell_01/RawSIMData_level_02.mrc` (noisy) → `micro_RawSIMData_level_02` → train/validation
- `Cell_01/RawSIMData_gt.mrc` (clean) → `micro_gt` → test

**Astronomy**: Observation ID grouping
- `j8hqbifjq_detection_sci.fits` (noisy) → `astro_j8hqbifjq` → train/validation
- `j8hqbifjq_g800l_sci.fits` (clean) → `astro_j8hqbifjq` → test

### Hash-Based Deterministic Splitting

```python
# Ensures same scene always gets same split
hash_val = int(hashlib.md5(scene_id.encode()).hexdigest(), 16)
split_val = hash_val % 100  # Maps to 0-99

# Noisy: 70/15/15 train/val/train
# Clean: 100% test
```

---

## Data Schema

### PNG Tiles Metadata Schema

```python
{
    # Identification
    "tile_id": str,                    # Unique tile identifier
    "domain": str,                     # photography/microscopy/astronomy
    "scene_id": str,                   # Scene grouping identifier

    # Data classification
    "data_type": str,                  # noisy/clean
    "split": str,                      # train/validation/test

    # PNG file information
    "png_path": str,                   # Path to PNG file
    "tile_size": int,                  # 256
    "channels": int,                   # 1 (grayscale) or 3 (RGB)

    # Position information
    "grid_x": int,                     # Grid position X
    "grid_y": int,                     # Grid position Y
    "image_x": int,                    # Image position X
    "image_y": int,                    # Image position Y

    # Physics-based calibration
    "gain": float,                     # Calibration gain (e-/ADU)
    "read_noise": float,               # Read noise (e-)
    "calibration_method": str,         # Calibration method name

    # Quality metrics
    "quality_score": float,            # Mean tile intensity
    "valid_ratio": float,              # Valid pixel ratio
    "is_edge_tile": bool,              # Edge tile flag
    "overlap_ratio": float,             # Tile overlap ratio

    # Processing metadata
    "source_file": str,                # Original source file path
    "processing_timestamp": str,       # ISO timestamp
}
```

### Example Tile Record

```python
{
    "tile_id": "photography_00001_00_0_tile_0000",
    "domain": "photography",
    "scene_id": "photo_00001",
    "data_type": "noisy",
    "split": "train",
    "png_path": "/path/to/processed/png_tiles/photography/noisy/photography_00001_00_0_tile_0000.png",
    "tile_size": 256,
    "channels": 3,
    "gain": 2.1,
    "read_noise": 6.0,
    "calibration_method": "photon_transfer_curve",
    "quality_score": 0.45,
    "valid_ratio": 1.0,
    "is_edge_tile": False,
    "overlap_ratio": 0.0,
    "source_file": "/path/to/raw/SID/Sony/short/00001_00_0.04s.ARW",
    "processing_timestamp": "2024-12-19T10:30:45.123456"
}
```

---

## Training Integration

### Step 1: Load Data

```python
import json
import pandas as pd
from PIL import Image

# Load processed tiles metadata
with open("PKL-DiffusionDenoising/data/processed/tiles_metadata.json", 'r') as f:
    tiles_metadata = json.load(f)

# Convert to DataFrame
tiles_df = pd.DataFrame(tiles_metadata)

# Split by split field
train = tiles_df[tiles_df['split'] == 'train']
val = tiles_df[tiles_df['split'] == 'validation']
test = tiles_df[tiles_df['split'] == 'test']

print(f"Train: {len(train)} tiles (all noisy)")
print(f"Validation: {len(val)} tiles (all noisy)")
print(f"Test: {len(test)} tiles (all clean)")
```

### Step 2: Verify Data Integrity

```python
# Verify clean/noisy separation
assert (train['data_type'] == 'noisy').all(), "Train should be noisy only"
assert (val['data_type'] == 'noisy').all(), "Val should be noisy only"
assert (test['data_type'] == 'clean').all(), "Test should be clean only"

# Verify split distribution
total = len(tiles_df)
print(f"Train: {len(train)/total:.1%}")
print(f"Val: {len(val)/total:.1%}")
print(f"Test: {len(test)/total:.1%}")
```

### Step 3: Load PNG Tiles

```python
def load_png_tile(png_path):
    """Load PNG tile and convert to tensor"""
    image = Image.open(png_path)
    if image.mode == 'L':  # Grayscale
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.unsqueeze(0)  # Add channel dimension
    elif image.mode == 'RGB':  # RGB
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
    return tensor

# Example usage
tile = train.iloc[0]
tile_data = load_png_tile(tile['png_path'])  # Load from PNG file
print(f"Tile shape: {tile_data.shape}")  # [C, H, W]
```

### Step 4: PyTorch Dataset

```python
from torch.utils.data import Dataset, DataLoader
import torch

class DiffusionTileDataset(Dataset):
    def __init__(self, tiles_df, split="train"):
        self.tiles = tiles_df[tiles_df['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        row = self.tiles.iloc[idx]

        # Load PNG tile
        tile_data = load_png_tile(row['png_path'])

        return {
            'image': tile_data,
            'domain': row['domain'],
            'scene_id': row['scene_id'],
            'gain': row['gain'],
            'read_noise': row['read_noise'],
            'tile_id': row['tile_id']
        }

# Usage
train_dataset = DiffusionTileDataset(tiles_df, split="train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

## Verification

### Verify Pipeline Configuration

```bash
python process_tiles_pipeline.py --demo_mode
# Will show:
# 🎯 DOMAIN-SPECIFIC CALIBRATION DEMONSTRATION
# 📊 Photography: Camera sensor calibration (Gain: 5.0 e-/ADU, Read noise: 3.6 e-)
# 📊 Microscopy: sCMOS detector calibration (Gain: 1.0 e-/ADU, Read noise: 1.5 e-)
# 📊 Astronomy: HST instrument calibration (Gain: 1.0 e-/ADU, Read noise: 3.5 e-)
# ✅ DEMONSTRATION COMPLETE
```

### Verify PNG Tiles Generation

```bash
python process_tiles_pipeline.py --max_files 3
# Will show:
# 🚀 Starting Simple PNG Tiles Pipeline
# 📷 Found X photography files
# 🔬 Found Y microscopy files
# 🌟 Found Z astronomy files
# ✅ Generated N PNG tiles from each domain
# 💾 Metadata saved to: tiles_metadata.json
```

### Verify Split Distribution

```python
import json
import pandas as pd

# Load metadata
with open("PKL-DiffusionDenoising/data/processed/tiles_metadata.json", 'r') as f:
    tiles_metadata = json.load(f)

tiles_df = pd.DataFrame(tiles_metadata)

# Check split distribution
print("📊 Split Distribution:")
print(tiles_df['split'].value_counts(normalize=True))

# Check data type by split
print("\n📊 Data Type by Split:")
print(pd.crosstab(tiles_df['split'], tiles_df['data_type']))

# Check domain distribution
print("\n📊 Domain Distribution:")
print(tiles_df['domain'].value_counts())
```

---

## File Reference

### Main Pipeline Files
- `process_tiles_pipeline.py` - Main simple PNG tiles pipeline with domain-specific calibration
- `domain_processors.py` - Domain-specific image processors
- `complete_systematic_tiling.py` - Systematic tiling implementation
- `astronomy_asinh_preprocessing.py` - Astronomy preprocessing module
- `physics_based_calibration.py` - Physics-based calibration utilities
- `visualize_calibration_standalone.py` - Calibration visualization tools

### Output Directories
- `PKL-DiffusionDenoising/data/processed/png_tiles/` - PNG tiles organized by domain/type
- `PKL-DiffusionDenoising/data/processed/tiles_metadata.json` - Complete metadata

### Key Methods in Pipeline
- `run_png_tiles_pipeline()` - Main pipeline execution method
- `process_file_to_png_tiles()` - Individual file processing method
- `process_single_file_to_tiles()` - Alternative single file processing method
- `_get_physics_based_calibration()` - Domain-specific calibration
- `_apply_physics_calibration()` - Physics-based calibration implementation
- `_downsample_with_antialiasing()` - Camera-specific photography downsampling
- `_extract_sony_tiles()` - Custom Sony tiling (2×4 grid)
- `_assign_split()` - Scene-aware train/test/validation splitting
- `demosaic_rggb_to_rgb()` - Basic demosaicing for photography
- `save_tile_as_png()` - PNG tile saving with metadata
- `run_domain_calibration_demo()` - Demo mode for calibration examples

---

## Contact & Support

For issues or questions about the pipeline:
1. Run demo mode to verify calibration parameters
2. Check PNG tiles in output directory
3. Review JSON metadata for processing details
4. Verify input data structure matches expected format
