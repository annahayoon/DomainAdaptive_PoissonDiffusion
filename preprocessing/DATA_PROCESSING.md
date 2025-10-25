# Data Processing Pipeline Documentation

**Updated: October 2025** (Documentation aligned with `SimpleTilesPipeline` implementation)

## Overview

The system processes multi-domain imaging data (Photography, Microscopy, Astronomy) into unified **float32 .pt tiles (256×256)** with domain-specific normalization ranges, scene-based splitting, and complete metadata tracking using a **file-based pipeline**.

### Key Features

- ✅ **Local file-based processing** - No Spark/Parquet/Delta Lake dependencies
- ✅ **Domain-specific normalization ranges** based on comprehensive pixel distribution analysis
- ✅ **Float32 .pt storage** with full precision preservation (~4x space savings vs raw)
- ✅ **Consistent tile size** (256×256) across all domains for cross-domain training
- ✅ **Camera-specific photography downsampling** with anti-aliasing
- ✅ **Correct astronomy labels**: Direct image = clean, G800L grism = noisy
- ✅ **Domain-specific range normalization** [domain_min, domain_max] → [0,1] → [-1,1] for EDM training
- ✅ **Scene-based train/test/validation splitting** with no data leakage

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Data Organization](#data-organization)
4. [Processing Flow](#processing-flow)
5. [Domain-Specific Processing](#domain-specific-processing)
6. [Data Characteristics and Pixel Distributions](#data-characteristics-and-pixel-distributions)
7. [Scene-Based Splitting](#scene-based-splitting)
8. [Data Schema](#data-schema)
9. [Training Integration](#training-integration)
10. [Verification](#verification)
11. [File Reference](#file-reference)

---

## Quick Start

### Test Run (Limited Files)
```bash
cd /home/jilab/Jae/preprocessing
python process_tiles_pipeline.py --max_files 3
```

### Full Production Run
```bash
python process_tiles_pipeline.py
```

### Expected Processing Time
- **Test run** (3 files/domain): ~5-10 minutes
- **Full run** (all files): ~2-4 hours
- **Total output**: ~78,000+ tiles (clean + noisy pairs, perfect 1:1 mapping)

### Generate Visualizations
```bash
python process_tiles_pipeline.py --max_files 3 --visualize
```
This creates 4-step processing visualizations showing: Raw Loading → After Tiling → Domain Normalization → Tensor Conversion

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

### Simple .pt Tiles Processing Flow

The pipeline processes images with domain-specific normalization:

```
1. Load full image (RAW sensor data in ADU units)
   ↓
2. Apply camera-specific photography downsampling (with anti-aliasing)
   • Sony: 2848×4256 → 1424×2128 (2x factor)
   • Fuji: 4032×6032 → 1008×1508 (4x factor)
   • Microscopy/Astronomy: No downsampling
   ↓
3. Tile the image into 256×256 patches
   • Sony: Custom tiling (6×9 = 54 tiles)
   • Fuji: Custom tiling (4×6 = 24 tiles)
   • Microscopy: Custom tiling (4×4 = 16 tiles)
   • Astronomy: Custom tiling (9×9 = 81 tiles)
   ↓
4. Apply domain-specific range normalization to each tile
   • Photography: [0, 15871] → [0,1] → [-1,1]
   • Microscopy: [0, 65535] → [0,1] → [-1,1]
   • Astronomy: [-65, 385] → [0,1] → [-1,1]
   ↓
5. Separate prior (clean) vs posterior (noisy)
   ↓
6. Assign train/test/validation splits (scene-aware)
   ↓
7. Save float32 .pt tiles with metadata
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
│           ├── RawSIMData_gt.mrc          # Noisy
│           └── SIM_gt/SIM_gt_a.mrc        # Clean
└── astronomy/hla_associations/  # Astronomy domain
    ├── detection_sci.fits      # Clean (direct images)
    └── g800l_sci.fits          # Noisy (grism spectroscopy)
```

### Processed Data Structure
```
PKL-DiffusionDenoising/data/processed/
├── pt_tiles/              # Float32 .pt tiles organized by domain and type
│   ├── photography/
│   │   ├── noisy/         # Noisy photography tiles (short exposure)
│   │   └── clean/         # Clean photography tiles (long exposure)
│   ├── microscopy/
│   │   ├── noisy/         # Noisy microscopy tiles (RawSIMData)
│   │   └── clean/         # Clean microscopy tiles (SIM_gt)
│   └── astronomy/
│       ├── noisy/         # Noisy astronomy tiles (g800l grism)
│       └── clean/         # Clean astronomy tiles (detection)
├── comprehensive_tiles_metadata.json    # Complete metadata for all tiles
├── metadata_*_incremental.json         # Incremental saves per domain
└── visualizations/                      # Optional 4-step processing visualizations
```

---

## Processing Flow

### Core Processing Steps

```python
# STEP 1: Load full image
image, metadata = load_image(file_path)  # RAW ADU values

# STEP 2: Apply camera-specific photography downsampling (with anti-aliasing)
if domain == "photography":
    if file_path.endswith('.ARW'):
        image = _downsample_with_antialiasing(image, 2.0)  # Sony: 2848×4256 → 1424×2128
    elif file_path.endswith('.RAF'):
        image = _downsample_with_antialiasing(image, 4.0)  # Fuji: 4032×6032 → 1008×1508

# STEP 3: Tile the image into 256×256 patches
if domain == "photography":
    if file_path.endswith('.ARW'):
        tiles = _extract_sony_tiles(image)  # 6×9 = 54 tiles
    elif file_path.endswith('.RAF'):
        tiles = _extract_fuji_tiles(image)  # 4×6 = 24 tiles
elif domain == "microscopy":
    tiles = _extract_microscopy_tiles(image)  # 4×4 = 16 tiles
elif domain == "astronomy":
    tiles = _extract_astronomy_tiles(image)  # 9×9 = 81 tiles

# STEP 4: Apply domain-specific range normalization to each tile
domain_range = domain_ranges.get(domain)  # e.g., {"min": 0.0, "max": 15871.0}
# Normalize: [domain_min, domain_max] → [0,1] → [-1,1]
normalized_tile = (tile_data - domain_range["min"]) / (domain_range["max"] - domain_range["min"])
normalized_tile = np.clip(normalized_tile, 0, 1)
normalized_tile = 2 * normalized_tile - 1  # [0,1] → [-1,1]

# STEP 5-7: Classify, split, and save as .pt
for i, tile_info in enumerate(tiles):
    data_type = _determine_data_type(file_path, domain)  # clean/noisy
    scene_id = _get_scene_id(file_path, domain)
    split = _assign_split(scene_id, data_type)  # train/val/test

    # Save as .pt file
    tile_id = f"{domain}_{Path(file_path).stem}_tile_{i:04d}"
    save_tile_as_pt(normalized_tile, tile_id, domain, data_type, domain_range)
```

---

## Domain-Specific Processing

### Photography: Camera-Specific Downsampling + Range Normalization
- **Camera-specific downsampling with anti-aliasing**: Sony (2x), Fuji (4x) prevents aliasing artifacts
- **Range normalization**: [0, 15871] → [0,1] → [-1,1] for EDM training
- **RGB demosaicing**: Basic RGGB → RGB conversion (no white balance)
- **Custom tiling**: Sony (6×9 = 54 tiles), Fuji (4×6 = 24 tiles)

### Microscopy: Range Normalization
- **No downsampling**: Preserves fine cellular details
- **Range normalization**: [0, 65535] → [0,1] → [-1,1] for EDM training
- **Handles full 16-bit range**: Common in scientific cameras
- **Custom tiling**: 4×4 = 16 tiles per image

### Astronomy: Range Normalization with Negative Values
- **No downsampling**: Preserves astronomical features
- **Range normalization with negatives**: [-65, 385] → [0,1] → [-1,1] for EDM training
- **Handles background-subtracted data**: Preserves negative values from sky subtraction
- **Custom tiling**: 9×9 = 81 tiles per image
- **Correct labeling**: Direct image = clean, G800L grism = noisy

---

## Data Characteristics and Pixel Distributions

This section provides comprehensive analysis of pixel value distributions across all three imaging domains, examining both noisy and clean samples to understand noise characteristics and signal quality.

### Methodology

- **Photography**: Sony ARW raw files loaded and processed in RGGB Bayer format (4 channels)
- **Microscopy**: MRC files loaded as single-channel grayscale data
- **Astronomy**: FITS files loaded as single-channel data with background subtraction
- Statistics calculated after removing NaN and infinite values
- All values reported as floating-point numbers

### Photography Domain (Sony ARW - RGGB Format)

#### Noisy Sample (0.04s exposure)
**Combined Channels Analysis:**
- **Pixel Count:** 12,121,088 pixels
- **Range:** [0.00, 562.00]
- **Mean:** 9.11
- **Median:** 4.00
- **Standard Deviation:** 12.40

**Individual Channel Breakdown:**
| Channel | Count | Range | Mean | Median | Std Dev |
|---------|-------|-------|------|--------|---------|
| R | 3,030,272 | [0.00, 312.00] | 8.08 | 3.00 | 11.43 |
| G1 | 3,030,272 | [0.00, 344.00] | 10.25 | 5.00 | 13.30 |
| G2 | 3,030,272 | [0.00, 562.00] | 10.43 | 5.00 | 13.42 |
| B | 3,030,272 | [0.00, 345.00] | 7.68 | 2.00 | 11.01 |

#### Clean Sample (10s exposure)
**Combined Channels Analysis:**
- **Pixel Count:** 12,121,088 pixels
- **Range:** [0.00, 15868.00]
- **Mean:** 1182.23
- **Median:** 916.00
- **Standard Deviation:** 990.31

**Individual Channel Breakdown:**
| Channel | Count | Range | Mean | Median | Std Dev |
|---------|-------|-------|------|--------|---------|
| R | 3,030,272 | [0.00, 9368.00] | 873.50 | 708.00 | 661.37 |
| G1 | 3,030,272 | [0.00, 15868.00] | 1585.16 | 1316.00 | 1119.94 |
| G2 | 3,030,272 | [0.00, 15868.00] | 1584.92 | 1316.00 | 1119.83 |
| B | 3,030,272 | [0.00, 9332.00] | 685.33 | 500.00 | 557.12 |

### Microscopy Domain (MRC Format)

#### Noisy Sample
- **Pixel Count:** 252,004 pixels
- **Range:** [47.00, 206.00]
- **Mean:** 117.40
- **Median:** 116.00
- **Standard Deviation:** 11.14

#### Clean Sample
- **Pixel Count:** 252,004 pixels
- **Range:** [140.00, 3204.00]
- **Mean:** 737.31
- **Median:** 694.00
- **Standard Deviation:** 348.14

### Astronomy Domain (FITS Format)

#### Noisy Sample
- **Pixel Count:** 17,859,040 pixels
- **Range:** [-34.35, 92.49]
- **Mean:** 0.00
- **Median:** 0.00
- **Standard Deviation:** 0.26

#### Clean Sample
- **Pixel Count:** 17,859,040 pixels
- **Range:** [-0.04, 103.54]
- **Mean:** 0.00
- **Median:** 0.00
- **Standard Deviation:** 0.11

### Cross-Domain Comparison Summary

#### Photography
- **Noisy:** Range [0.00, 562.00], Mean: 9.11, Median: 4.00
- **Clean:** Range [0.00, 15868.00], Mean: 1182.23, Median: 916.00
- **Signal-to-Noise Ratio:** ~100x higher intensity in clean sample

#### Microscopy
- **Noisy:** Range [47.00, 206.00], Mean: 117.40, Median: 116.00
- **Clean:** Range [140.00, 3204.00], Mean: 737.31, Median: 694.00
- **Signal-to-Noise Ratio:** ~6x higher intensity in clean sample

#### Astronomy
- **Noisy:** Range [-34.35, 92.49], Mean: 0.00, Median: 0.00
- **Clean:** Range [-0.04, 103.54], Mean: 0.00, Median: 0.00
- **Signal-to-Noise Ratio:** Both background-subtracted, but noisy shows wider spread

### Key Insights

1. **Dynamic Range Variation**: Photography shows the largest dynamic range difference between noisy and clean samples (~100x), while astronomy shows the most consistent baseline (background-subtracted values).

2. **Noise Characteristics**:
   - **Photography**: Very low values in noisy samples (typical of underexposed raw photography)
   - **Microscopy**: Moderate noise with clean samples showing significantly higher intensity
   - **Astronomy**: Noise manifests as wider value spread and more negative values

3. **Data Structure Differences**:
   - Photography: Multi-channel RGGB format requiring channel-wise analysis
   - Microscopy & Astronomy: Single-channel data allowing direct pixel-wise analysis

4. **Exposure/Quality Indicators**:
   - Photography clean samples show much higher green channel values (typical of Bayer sensors)
   - Microscopy shows clear intensity scaling between noisy and clean
   - Astronomy shows effective background subtraction in both samples

### Comprehensive Statistics (20 samples per domain)

#### Photography Domain (Sony ARW - RGGB Format)

**Noisy Samples (0.04s-0.1s exposure)**
- **R Channel:** Min=0.00, Max=15871.00, Mean=9.75, Median=3.00 (20 files)
- **G1 Channel:** Min=0.00, Max=15871.00, Mean=20.04, Median=6.50 (20 files)
- **G2 Channel:** Min=0.00, Max=15871.00, Mean=20.03, Median=6.50 (20 files)
- **B Channel:** Min=0.00, Max=15871.00, Mean=9.80, Median=2.00 (20 files)

**Clean Samples (10s-30s exposure)**
- **R Channel:** Min=0.00, Max=15871.00, Mean=1113.46, Median=693.50 (20 files)
- **G1 Channel:** Min=0.00, Max=15871.00, Mean=1780.01, Median=1086.50 (20 files)
- **G2 Channel:** Min=0.00, Max=15871.00, Mean=1779.78, Median=1088.50 (20 files)
- **B Channel:** Min=0.00, Max=15871.00, Mean=525.01, Median=345.50 (20 files)

#### Microscopy Domain (MRC Format)

**Noisy Samples (SIM level_01)**
- **Min:** 31.00, **Max:** 457.00, **Mean:** 120.34, **Median:** 116.00 (20 files)

**Clean Samples (SIM ground truth)**
- **Min:** 0.00, **Max:** 65535.00, **Mean:** 3941.85, **Median:** 2148.50 (20 files)

#### Astronomy Domain (FITS Format)

**Noisy Samples (G800L grating)**
- **Min:** -64.22, **Max:** 189.71, **Mean:** 0.05, **Median:** 0.00 (20 files)

**Clean Samples (Direct detection)**
- **Min:** -0.15, **Max:** 216.19, **Mean:** 0.00, **Median:** 0.00 (20 files)

### Dataset Scale Summary

#### Total Available Files
- **Photography:** 2,697 noisy + 231 clean = 2,928 ARW files
- **Microscopy:** 415 noisy + 473 clean = 888 MRC files
- **Astronomy:** 153 noisy + 153 clean = 306 FITS files

#### Analysis Coverage
- **Photography:** 20 noisy + 20 clean samples analyzed
- **Microscopy:** 20 noisy + 20 clean samples analyzed
- **Astronomy:** 20 noisy + 20 clean samples analyzed

#### Key Findings
1. **Photography** shows the largest dynamic range with clean samples having ~100x higher intensity than noisy samples
2. **Microscopy** demonstrates clear signal enhancement with clean samples showing ~33x higher mean intensity
3. **Astronomy** maintains background-subtracted values near zero for both noisy and clean samples
4. **Photography** exhibits strong green channel dominance in clean samples (typical of Bayer sensors)
5. **Microscopy** shows the widest range in clean samples (0-65535) indicating full dynamic range utilization

### Domain-Specific Normalization Ranges

For consistent normalization across all domains in model training, the following ranges are used:

#### Photography Domain
- **Range:** [0.00, 15871.00]
- **Format:** Sony ARW RGGB (4 channels)
- **Notes:** Based on comprehensive analysis of 20 noisy and 20 clean samples

#### Microscopy Domain
- **Range:** [0.00, 65535.00]
- **Format:** MRC single-channel (16-bit)
- **Notes:** Full 16-bit dynamic range utilization in clean samples

#### Astronomy Domain
- **Range:** [-65.00, 385.00]
- **Format:** FITS single-channel (background-subtracted)
- **Notes:** Updated range with safety margin from comprehensive analysis of all 306 FITS files (153 noisy + 153 clean)
- **Breakdown:**
  - **Noisy samples (G800L):** [-64.22, 201.48] (153 files)
  - **Clean samples (Detection):** [-0.15, 382.81] (153 files)

---

## Scene-Based Splitting

### Split Distribution

| Split | Percentage | Purpose |
|-------|------------|---------|
| **Train** | ~70% | Learn denoising (all data types) |
| **Validation** | ~15% | Tune hyperparameters (all data types) |
| **Test** | ~15% | Final evaluation (all data types) |

**IMPORTANT**: ALL data types (clean AND noisy) from the same scene are assigned to the SAME split to prevent data leakage.

### Scene Grouping Logic

**Photography**: Exposure pairs share same scene ID
- `00001_00_0.04s.ARW` (noisy) → `photo_00001` → same split (e.g., train)
- `00001_00_10s.ARW` (clean) → `photo_00001` → same split (e.g., train)

**Microscopy**: Cell and structure grouping with unique identifiers
- `structures/F-actin/Cell_005/RawSIMData_gt.mrc` (noisy) → `micro_F-actin_Cell_005_RawSIMData` → same split
- `structures/F-actin/Cell_005/SIM_gt.mrc` (clean) → `micro_F-actin_Cell_005_RawSIMData` → same split

**Astronomy**: Observation ID grouping
- `j8hqbifjq_detection_sci.fits` (clean) → `astro_j8hqbifjq` → same split (e.g., validation)
- `j8hqbifjq_g800l_sci.fits` (noisy) → `astro_j8hqbifjq` → same split (e.g., validation)

### Hash-Based Deterministic Splitting

```python
# Ensures same scene always gets same split (prevents data leakage)
import hashlib
import random

seed = int(hashlib.md5(scene_id.encode(), usedforsecurity=False).hexdigest(), 16) % (2**32)
random.seed(seed)
split_val = random.random() * 100  # 0-100

# ALL data types from same scene get same split
if split_val < 70:
    return "train"      # 70%
elif split_val < 85:
    return "validation"  # 15%
else:
    return "test"        # 15%
```

---

## Data Schema

### .pt Tiles Metadata Schema

```python
{
    # Identification
    "tile_id": str,                    # Unique tile identifier
    "domain": str,                     # photography/microscopy/astronomy
    "scene_id": str,                   # Scene grouping identifier

    # Data classification
    "data_type": str,                  # noisy/clean
    "split": str,                      # train/validation/test

    # .pt file information
    "pt_path": str,                    # Path to .pt file (PyTorch tensor)
    "tile_size": int,                  # 256
    "channels": int,                   # 1 (grayscale) or 3 (RGB)

    # Position information
    "grid_x": int,                     # Grid position X
    "grid_y": int,                     # Grid position Y
    "image_x": int,                    # Image position X
    "image_y": int,                    # Image position Y

    # Quality metrics
    "quality_score": float,            # Mean tile intensity
    "valid_ratio": float,              # Valid pixel ratio
    "is_edge_tile": bool,              # Edge tile flag
    "overlap_ratio": float,             # Tile overlap ratio
    "systematic_coverage": bool,       # True for proper tiling coverage

    # Processing metadata
    "processing_timestamp": str,       # ISO timestamp
    "domain_range": dict,              # Domain-specific normalization range
    "original_min": float,             # Original data range min (before normalization)
    "original_max": float,             # Original data range max (before normalization)
    "original_mean": float,            # Original data mean (before normalization)
    "original_std": float,             # Original data std (before normalization)
    "normalized_min": float,           # Normalized range min (after [-1,1] transform)
    "normalized_max": float,           # Normalized range max (after [-1,1] transform)
    "normalized_mean": float,          # Normalized range mean (after [-1,1] transform)
    "normalized_std": float,           # Normalized range std (after [-1,1] transform)
}
```

### Example Tile Record

```python
{
    "tile_id": "photography_sony_00001_00_0_tile_0000",
    "domain": "photography",
    "scene_id": "photo_00001",
    "data_type": "noisy",
    "split": "train",
    "pt_path": "/path/to/processed/pt_tiles/photography/noisy/photography_sony_00001_00_0_tile_0000.pt",
    "tile_size": 256,
    "channels": 3,
    "grid_x": 0,
    "grid_y": 0,
    "image_x": 0,
    "image_y": 0,
    "quality_score": 0.45,
    "valid_ratio": 1.0,
    "is_edge_tile": True,
    "overlap_ratio": 0.09,
    "systematic_coverage": True,
    "processing_timestamp": "2025-01-07T10:30:45.123456",
    "domain_range": {"min": 0.0, "max": 15871.0},
    "original_min": 0.0,
    "original_max": 562.0,
    "original_mean": 9.11,
    "original_std": 12.40,
    "normalized_min": -1.0,
    "normalized_max": 0.95,
    "normalized_mean": -0.87,
    "normalized_std": 0.12
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
with open("PKL-DiffusionDenoising/data/processed/comprehensive_tiles_metadata.json", 'r') as f:
    tiles_metadata = json.load(f)

# Convert to DataFrame
tiles_df = pd.DataFrame(tiles_metadata['tiles'])

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

### Step 3: Load .pt Tiles

```python
import torch

def load_pt_tile(pt_path):
    """Load .pt tile (PyTorch tensor)"""
    # Load tensor directly - already in correct format (C, H, W)
    tensor = torch.load(pt_path)  # Shape: (C, H, W)

    # Tensor is already:
    # - float32 dtype
    # - Range: [-1, 1] (domain-normalized)
    # - Shape: (1, 256, 256) for grayscale or (3, 256, 256) for RGB

    return tensor

# Example usage
tile = train.iloc[0]
tile_data = load_pt_tile(tile['pt_path'])  # Load from .pt file
print(f"Tile shape: {tile_data.shape}")  # [C, H, W]
print(f"Tile dtype: {tile_data.dtype}")  # torch.float32
print(f"Tile range: [{tile_data.min():.3f}, {tile_data.max():.3f}]")  # [-1, 1]
print(f"Domain: {tile['domain']}")  # photography/microscopy/astronomy
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

        # Load .pt tile (already normalized to [-1, 1])
        tile_data = load_pt_tile(row['pt_path'])

        return {
            'image': tile_data,  # Shape: (C, H, W), Range: [-1, 1]
            'domain': row['domain'],  # photography/microscopy/astronomy
            'scene_id': row['scene_id'],
            'data_type': row['data_type'],  # clean/noisy
            'tile_id': row['tile_id'],
            'domain_range': row['domain_range']  # Original domain range
        }

# Usage
train_dataset = DiffusionTileDataset(tiles_df, split="train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Example: iterate through batches
for batch in train_loader:
    images = batch['image']  # Shape: (B, C, H, W)
    domains = batch['domain']  # List of domain names
    print(f"Batch shape: {images.shape}, Range: [{images.min():.3f}, {images.max():.3f}]")
    break
```

---

## Verification

### Verify .pt Tiles Generation

```bash
python process_tiles_pipeline.py --max_files 3
# Will show:
# 🚀 Starting .pt tiles pipeline with domain-specific range normalization: [domain_min, domain_max] → [0,1] → [-1,1]...
# 📸 Photography file selection (long=clean, first short=noisy):
#    • Sony scenes with pairs: X
#    • Fuji scenes with pairs: Y
# 🔬 Microscopy file selection (RawSIMData_gt=noisy, SIM_gt/SIM_gt_a=clean):
#    • Total pairs: Z
# ✅ Comprehensive metadata saved to: comprehensive_tiles_metadata.json
# 🎊 SUCCESS: .pt Tiles Pipeline Completed!
# 📊 Total .pt tiles generated: N
```

### Verify Split Distribution

```python
import json
import pandas as pd

# Load metadata
with open("PKL-DiffusionDenoising/data/processed/comprehensive_tiles_metadata.json", 'r') as f:
    tiles_metadata = json.load(f)

tiles_df = pd.DataFrame(tiles_metadata['tiles'])

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
- `process_tiles_pipeline.py` - Main .pt tiles pipeline with domain-specific normalization
  - Class: `SimpleTilesPipeline`
- `complete_systematic_tiling.py` - Systematic tiling implementation (SystematicTiler)

### Output Directories
- `PKL-DiffusionDenoising/data/processed/pt_tiles/` - .pt tiles organized by domain/type
- `PKL-DiffusionDenoising/data/processed/comprehensive_tiles_metadata.json` - Complete metadata
- `PKL-DiffusionDenoising/data/processed/metadata_*_incremental.json` - Incremental saves per domain
- `PKL-DiffusionDenoising/data/processed/visualizations/` - Optional processing step visualizations

### Key Methods in Pipeline (`SimpleTilesPipeline`)
- `run_pt_tiles_pipeline()` - Main pipeline execution method
- `process_file_to_pt_tiles()` - Individual file processing with optional visualization
- `load_photography_raw()` - Load Sony/Fuji raw files using rawpy
- `load_microscopy_mrc()` - Load MRC files using BioSR reader
- `load_astronomy_raw()` - Load FITS files using astropy
- `_downsample_with_antialiasing()` - Anti-aliasing downsampling (scipy)
- `_extract_sony_tiles()` - Custom Sony tiling (6×9 = 54 tiles, ~9% overlap)
- `_extract_fuji_tiles()` - Custom Fuji tiling (4×6 = 24 tiles, ~2.3% overlap)
- `_extract_microscopy_tiles()` - Custom microscopy tiling (4×4 = 16 tiles, ~2.7% overlap)
- `_extract_astronomy_tiles()` - Custom astronomy tiling (9×9 = 81 tiles, ~9.5% overlap)
- `_assign_split()` - Scene-aware deterministic splitting (70/15/15)
- `_determine_data_type()` - Classify files as clean/noisy
- `_get_scene_id()` - Extract scene identifiers for grouping
- `_select_photography_file_pairs()` - Select clean/noisy photography pairs
- `_select_microscopy_file_pairs()` - Select clean/noisy microscopy pairs
- `demosaic_rggb_to_rgb()` - Basic RGGB → RGB demosaicing
- `save_tile_as_pt()` - Save tiles as .pt with normalization: [domain_min, domain_max] → [0,1] → [-1,1]
- `create_scene_visualization()` - Generate 4-step processing visualizations
- `reconstruct_metadata_from_incremental()` - Recover metadata from incremental saves

---

## Contact & Support

For issues or questions about the pipeline:
1. Run test mode to verify processing: `python process_tiles_pipeline.py --max_files 3`
2. Check .pt tiles in output directory: `data/processed/pt_tiles/`
3. Review comprehensive metadata: `data/processed/comprehensive_tiles_metadata.json`
4. Check incremental saves if interrupted: `data/processed/metadata_*_incremental.json`
5. Generate visualizations for debugging: `python process_tiles_pipeline.py --max_files 3 --visualize`
6. Verify input data structure matches expected format in raw data directories
