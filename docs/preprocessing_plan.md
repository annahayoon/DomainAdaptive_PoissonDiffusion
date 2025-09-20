# Revised Data Preprocessing Plan for Domain-Adaptive Poisson-Gaussian Diffusion
**Version 2.0 - Phase 1 Implementation (No PSF/Blur)**

---

## Executive Summary

This document provides **exact steps** to preprocess raw imaging data from three domains (photography, microscopy, astronomy) into standardized `.pt` files for training and evaluation.

**Key Principle**: Preprocessing runs **once** before training, not on-the-fly. Raw files → Preprocessed `.pt` files → Model training/inference.

---

## Part A: Data Flow Architecture

### Overall Pipeline
```
Step 1: RAW DATA (One-time download)
├── photography: .ARW/.DNG files
├── microscopy: .TIFF files
└── astronomy: .FITS files
    ↓
Step 2: PREPROCESSING (This document - run once)
├── Calibration extraction
├── Noise model fitting
├── Normalization by scale s
└── Tiling to 128×128
    ↓
Step 3: PREPROCESSED FILES (.pt format)
├── prior_clean/: Clean images for prior training
└── posterior/: Noisy+GT pairs for evaluation
    ↓
Step 4: MODEL TRAINING (Uses .pt files only)
├── Prior training: Clean tiles only
└── Posterior sampling: Full pipeline with noise
```

### Directory Structure (Final)
```bash
data/
├── raw/                          # Original downloads (keep untouched)
│   ├── SID/
│   │   └── Sony/
│   │       ├── short/           # Noisy: 00001_00_0.04s.ARW
│   │       └── long/            # Clean: 00001_00_10s.ARW
│   ├── FMD/
│   │   ├── dataset1/
│   │   │   ├── noisy/          # Low SNR: cell001_low.tif
│   │   │   └── clean/          # High SNR: cell001_high.tif
│   └── SDSS/
│       └── stripe82/            # frame-r-001234-1-0123.fits
│
├── preprocessed/                # Output of this pipeline
│   ├── manifests/
│   │   ├── photography.json    # Global statistics
│   │   ├── microscopy.json
│   │   └── astronomy.json
│   ├── prior_clean/            # For diffusion prior training
│   │   ├── photography/
│   │   │   ├── train/         # tile_00000.pt to tile_99999.pt
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── microscopy/
│   │   └── astronomy/
│   └── posterior/              # For evaluation/posterior sampling
│       ├── photography/
│       │   ├── train/         # scene_00001.pt (full images)
│       │   ├── val/
│       │   └── test/
│       ├── microscopy/
│       └── astronomy/
```

---

## Part B: File Format Specification

### B.1 Prior Training Files (`prior_clean/{domain}/{split}/tile_{idx}.pt`)

Each file contains **one** 128×128 clean tile:
```python
{
    'clean_norm': torch.Tensor,      # [C, 128, 128] float32, range [0, ~1]
    'domain_id': int,                # 0=photo, 1=micro, 2=astro
    'metadata': {
        'scene_id': str,              # Original image identifier
        'tile_idx': int,              # Tile number within scene
        'augmented': bool,            # Was augmentation applied
    }
}
```

### B.2 Posterior/Evaluation Files (`posterior/{domain}/{split}/scene_{id}.pt`)

Each file contains **one** complete scene (or large tile):
```python
{
    'noisy_norm': torch.Tensor,      # [C, H, W] normalized noisy, float32
    'clean_norm': torch.Tensor,      # [C, H, W] normalized GT (or None)
    'calibration': {
        'scale': float,               # s (electrons per 1.0)
        'gain': float,                # g (electrons per ADU)
        'read_noise': float,          # σ_r (electrons)
        'background': float,          # b (electrons)
        'black_level': float,         # ADU units
        'white_level': float,         # ADU units
    },
    'masks': {
        'valid': torch.Tensor,        # [1, H, W] bool, True=valid pixel
        'saturated': torch.Tensor,    # [1, H, W] bool, True=saturated
    },
    'metadata': {
        'domain_id': int,
        'scene_id': str,
        'original_shape': [H_orig, W_orig],
        'bit_depth': int,
        'iso': int,                  # Photography only
        'exposure_ms': float,         # If available
    }
}
```

### B.3 Manifest Files (`manifests/{domain}.json`)

Global statistics computed during preprocessing:
```json
{
    "domain": "photography",
    "date_processed": "2024-01-15",
    "num_scenes": {"train": 1000, "val": 200, "test": 200},
    "num_tiles": {"train": 50000, "val": 10000, "test": 10000},
    "scale_p999": 15234.5,           // 99.9th percentile in electrons
    "gain_mean": 2.47,                // Mean gain (e-/ADU)
    "read_noise_mean": 1.82,         // Mean read noise (e-)
    "tile_size": 128,
    "channels": 4,                    // 4 for photo, 1 for others
    "preprocessing_version": "2.0"
}
```

---

## Part C: Domain-Specific Processing

### C.1 Photography (SID Dataset)

#### Input Files
- Noisy: `data/raw/SID/Sony/short/XXXXX_00_0.04s.ARW`
- Clean: `data/raw/SID/Sony/long/XXXXX_00_10s.ARW`

#### Step-by-Step Processing

**1. Setup Environment**
```bash
pip install rawpy numpy torch tqdm scikit-learn
```

**2. Load and Extract Raw Bayer Data**
```python
import rawpy
import numpy as np
import torch

def load_sony_raw(filepath):
    """Load Sony ARW and extract Bayer pattern."""
    with rawpy.imread(filepath) as raw:
        # Get raw Bayer data (no demosaicing!)
        bayer = raw.raw_image_visible.astype(np.float32)

        # Get metadata
        black_level = np.array(raw.black_level_per_channel)
        white_level = raw.white_level

        # Get color pattern (usually RGGB for Sony)
        pattern = raw.raw_pattern

    return bayer, black_level, white_level, pattern
```

**3. Black Level Subtraction and Pack to 4-Channel**
```python
def process_bayer(bayer, black_level, white_level):
    """Convert Bayer to 4-channel packed format."""
    # Create black level map for full Bayer
    H, W = bayer.shape
    black_map = np.zeros((H, W), dtype=np.float32)

    # Sony is usually RGGB:
    # R  G
    # G  B
    black_map[0::2, 0::2] = black_level[0]  # R
    black_map[0::2, 1::2] = black_level[1]  # G1
    black_map[1::2, 0::2] = black_level[2]  # G2
    black_map[1::2, 1::2] = black_level[3]  # B

    # Subtract black level
    bayer_corrected = np.maximum(bayer - black_map, 0)

    # Pack to 4-channel half-resolution
    packed = np.zeros((4, H//2, W//2), dtype=np.float32)
    packed[0] = bayer_corrected[0::2, 0::2]  # R
    packed[1] = bayer_corrected[0::2, 1::2]  # G1
    packed[2] = bayer_corrected[1::2, 0::2]  # G2
    packed[3] = bayer_corrected[1::2, 1::2]  # B

    # Create saturation mask (before normalization)
    saturated = bayer >= (white_level - black_map) * 0.95
    sat_mask = np.zeros((1, H//2, W//2), dtype=bool)
    sat_mask[0] = saturated[0::2, 0::2] | saturated[0::2, 1::2] | \
                  saturated[1::2, 0::2] | saturated[1::2, 1::2]

    return packed, sat_mask
```

**4. Estimate Gain and Read Noise**
```python
def estimate_noise_params(bayer_imgs, black_levels):
    """Estimate gain and read noise from multiple frames."""
    # Collect patches from uniform regions
    patches = []
    for img, black in zip(bayer_imgs, black_levels):
        # Find low-gradient regions
        gradient = np.abs(np.gradient(img)[0]) + np.abs(np.gradient(img)[1])
        uniform_mask = gradient < np.percentile(gradient, 10)

        # Extract 32x32 patches
        for _ in range(100):
            h = np.random.randint(0, img.shape[0] - 32)
            w = np.random.randint(0, img.shape[1] - 32)
            if uniform_mask[h:h+32, w:w+32].mean() > 0.8:
                patch = img[h:h+32, w:w+32]
                patches.append(patch)

    # Compute mean and variance for each patch
    means = [p.mean() for p in patches]
    vars = [p.var(ddof=1) for p in patches]

    # Robust linear fit: var = gain * mean + read_noise^2/gain
    from sklearn.linear_model import HuberRegressor
    X = np.array(means).reshape(-1, 1)
    y = np.array(vars)

    reg = HuberRegressor()
    reg.fit(X, y)

    gain = reg.coef_[0]  # electrons per ADU
    read_var_adu = reg.intercept_
    read_noise = np.sqrt(max(read_var_adu * gain, 0))  # electrons

    return gain, read_noise

# For Sony A7S (typical values as fallback)
DEFAULT_GAIN = 2.47  # e-/ADU
DEFAULT_READ_NOISE = 1.82  # e-
```

**5. Compute Global Scale**
```python
def compute_global_scale(all_clean_imgs_electrons):
    """Compute 99.9th percentile across all training clean images."""
    all_pixels = []
    for img in all_clean_imgs_electrons:
        all_pixels.extend(img.flatten())

    # Sample if too many pixels
    if len(all_pixels) > 1e7:
        all_pixels = np.random.choice(all_pixels, int(1e7), replace=False)

    scale = np.percentile(all_pixels, 99.9)
    return scale
```

**6. Extract Tiles for Training**
```python
def extract_128_tiles(image, num_tiles=50, augment=True):
    """Extract 128x128 tiles from image."""
    C, H, W = image.shape
    tiles = []

    for _ in range(num_tiles):
        # Random position
        h = np.random.randint(0, H - 128 + 1)
        w = np.random.randint(0, W - 128 + 1)
        tile = image[:, h:h+128, w:w+128]

        if augment:
            # Random flips
            if np.random.rand() > 0.5:
                tile = np.flip(tile, axis=1)
            if np.random.rand() > 0.5:
                tile = np.flip(tile, axis=2)
            # Random 90-degree rotations
            k = np.random.randint(0, 4)
            tile = np.rot90(tile, k, axes=(1, 2))

        tiles.append(tile.copy())

    return tiles
```

**7. Main Processing Pipeline**
```python
def preprocess_photography():
    """Complete preprocessing for photography domain."""

    # Step 1: Discover all scene pairs
    scenes = discover_sid_scenes()  # Returns [(noisy_path, clean_path), ...]

    # Step 2: Split scenes 70/15/15
    train_scenes, val_scenes, test_scenes = split_scenes(scenes)

    # Step 3: Estimate noise parameters from training set
    print("Estimating noise parameters...")
    sample_raws = [load_sony_raw(s[0])[0] for s in train_scenes[:20]]
    sample_blacks = [load_sony_raw(s[0])[1] for s in train_scenes[:20]]
    gain, read_noise = estimate_noise_params(sample_raws, sample_blacks)
    print(f"Gain: {gain:.3f} e-/ADU, Read noise: {read_noise:.3f} e-")

    # Step 4: Process all clean images to find scale
    print("Computing global scale...")
    all_clean_electrons = []
    for _, clean_path in train_scenes[:100]:  # Sample
        bayer, black, white, _ = load_sony_raw(clean_path)
        packed, _ = process_bayer(bayer, black, white)
        electrons = packed * gain
        all_clean_electrons.append(electrons)
    scale = compute_global_scale(all_clean_electrons)
    print(f"Global scale: {scale:.1f} electrons")

    # Step 5: Process and save all scenes
    tile_idx = 0

    for split_name, scene_list in [('train', train_scenes),
                                   ('val', val_scenes),
                                   ('test', test_scenes)]:

        print(f"Processing {split_name} split...")

        for scene_id, (noisy_path, clean_path) in enumerate(tqdm(scene_list)):
            # Load and process noisy
            noisy_bayer, black_n, white_n, _ = load_sony_raw(noisy_path)
            noisy_packed, sat_mask = process_bayer(noisy_bayer, black_n, white_n)
            noisy_electrons = noisy_packed * gain
            noisy_norm = noisy_electrons / scale

            # Load and process clean
            clean_bayer, black_c, white_c, _ = load_sony_raw(clean_path)
            clean_packed, _ = process_bayer(clean_bayer, black_c, white_c)
            clean_electrons = clean_packed * gain
            clean_norm = clean_electrons / scale

            # Background estimation (5th percentile of darkest regions)
            background = np.percentile(noisy_electrons[noisy_electrons < scale*0.1], 5)

            # Save full scene for posterior/evaluation
            scene_data = {
                'noisy_norm': torch.from_numpy(noisy_norm).float(),
                'clean_norm': torch.from_numpy(clean_norm).float(),
                'calibration': {
                    'scale': scale,
                    'gain': gain,
                    'read_noise': read_noise,
                    'background': background,
                    'black_level': float(black_n.mean()),
                    'white_level': float(white_n),
                },
                'masks': {
                    'valid': torch.ones(1, *noisy_norm.shape[1:], dtype=torch.bool),
                    'saturated': torch.from_numpy(sat_mask),
                },
                'metadata': {
                    'domain_id': 0,
                    'scene_id': f"scene_{scene_id:05d}",
                    'original_shape': list(noisy_bayer.shape),
                    'bit_depth': 14,
                    'iso': extract_iso_from_filename(noisy_path),
                }
            }

            save_path = f"data/preprocessed/posterior/photography/{split_name}/scene_{scene_id:05d}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(scene_data, save_path)

            # Extract and save clean tiles for prior training
            if split_name == 'train':
                tiles = extract_128_tiles(clean_norm, num_tiles=50, augment=True)
                for tile in tiles:
                    tile_data = {
                        'clean_norm': torch.from_numpy(tile).float(),
                        'domain_id': 0,
                        'metadata': {
                            'scene_id': f"scene_{scene_id:05d}",
                            'tile_idx': tile_idx,
                            'augmented': True,
                        }
                    }
                    tile_path = f"data/preprocessed/prior_clean/photography/train/tile_{tile_idx:06d}.pt"
                    os.makedirs(os.path.dirname(tile_path), exist_ok=True)
                    torch.save(tile_data, tile_path)
                    tile_idx += 1

    # Step 6: Save manifest
    manifest = {
        'domain': 'photography',
        'date_processed': str(datetime.now()),
        'num_scenes': {
            'train': len(train_scenes),
            'val': len(val_scenes),
            'test': len(test_scenes)
        },
        'num_tiles': {'train': tile_idx, 'val': 0, 'test': 0},
        'scale_p999': scale,
        'gain_mean': gain,
        'read_noise_mean': read_noise,
        'tile_size': 128,
        'channels': 4,
        'preprocessing_version': '2.0'
    }

    with open('data/preprocessed/manifests/photography.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Preprocessing complete! Generated {tile_idx} training tiles.")
```

### C.2 Microscopy (Fluorescence)

#### Input Files
- Noisy: `data/raw/FMD/sequence_X/noisy/*.tif`
- Clean: `data/raw/FMD/sequence_X/clean/*.tif`

#### Key Differences from Photography
- Single channel (grayscale)
- 16-bit TIFFs
- Need flat-field correction
- Different gain units (may already be in electrons)

#### Processing Pipeline
```python
def preprocess_microscopy():
    """Preprocess microscopy data."""
    import tifffile
    from scipy.ndimage import gaussian_filter

    # Load all image pairs
    sequences = discover_fmd_sequences()
    train_seq, val_seq, test_seq = split_sequences(sequences)

    # Step 1: Estimate flat field from multiple frames
    print("Estimating flat field...")
    flat_field = estimate_flat_field(train_seq[:20])

    # Step 2: Estimate noise parameters
    if data_already_in_electrons():
        gain = 1.0
        # Estimate read noise from dark regions
        read_noise = estimate_read_noise_electrons(train_seq[:20])
    else:
        gain, read_noise = estimate_noise_params_microscopy(train_seq[:20])

    # Step 3: Find global scale from clean images
    scale = compute_scale_microscopy(train_seq, gain)

    # Step 4: Process all sequences
    tile_idx = 0

    for split_name, seq_list in [('train', train_seq), ('val', val_seq), ('test', test_seq)]:
        for seq_id, (noisy_paths, clean_paths) in enumerate(seq_list):
            for frame_id, (noisy_path, clean_path) in enumerate(zip(noisy_paths, clean_paths)):

                # Load images
                noisy = tifffile.imread(noisy_path).astype(np.float32)
                clean = tifffile.imread(clean_path).astype(np.float32)

                # Apply flat field correction
                noisy = noisy / flat_field
                clean = clean / flat_field

                # Convert to electrons and normalize
                noisy_e = noisy * gain
                clean_e = clean * gain

                # Background estimation (morphological)
                from skimage.morphology import disk, opening
                background = opening(noisy_e, disk(30))
                background = gaussian_filter(background, sigma=10)
                background_scalar = np.percentile(background, 10)

                # Normalize
                noisy_norm = noisy_e / scale
                clean_norm = clean_e / scale

                # Add channel dimension
                noisy_norm = noisy_norm[np.newaxis, :, :]
                clean_norm = clean_norm[np.newaxis, :, :]

                # Save full frame for posterior
                scene_data = {
                    'noisy_norm': torch.from_numpy(noisy_norm).float(),
                    'clean_norm': torch.from_numpy(clean_norm).float(),
                    'calibration': {
                        'scale': scale,
                        'gain': gain,
                        'read_noise': read_noise,
                        'background': background_scalar,
                        'black_level': 100.0,  # Typical for sCMOS
                        'white_level': 65535.0,
                    },
                    'masks': {
                        'valid': torch.ones(1, *noisy_norm.shape[1:], dtype=torch.bool),
                        'saturated': torch.from_numpy(noisy > 65000)[None, :, :],
                    },
                    'metadata': {
                        'domain_id': 1,
                        'scene_id': f"seq{seq_id:03d}_frame{frame_id:03d}",
                        'original_shape': list(noisy.shape),
                        'bit_depth': 16,
                    }
                }

                save_path = f"data/preprocessed/posterior/microscopy/{split_name}/"
                save_path += f"seq{seq_id:03d}_frame{frame_id:03d}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(scene_data, save_path)

                # Extract clean tiles for prior training
                if split_name == 'train':
                    tiles = extract_128_tiles(clean_norm, num_tiles=20, augment=True)
                    for tile in tiles:
                        tile_data = {
                            'clean_norm': torch.from_numpy(tile).float(),
                            'domain_id': 1,
                            'metadata': {
                                'scene_id': f"seq{seq_id:03d}_frame{frame_id:03d}",
                                'tile_idx': tile_idx,
                                'augmented': True,
                            }
                        }
                        tile_path = f"data/preprocessed/prior_clean/microscopy/train/tile_{tile_idx:06d}.pt"
                        os.makedirs(os.path.dirname(tile_path), exist_ok=True)
                        torch.save(tile_data, tile_path)
                        tile_idx += 1
```

### C.3 Astronomy (SDSS)

#### Input Files
- SDSS frames: `data/raw/SDSS/stripe82/frame-r-*.fits`

#### Key Differences
- FITS format with headers
- Already bias/flat corrected (fpC files)
- Need cosmic ray detection
- No paired clean data (use coadds or external clean images)

#### Processing Pipeline
```python
def preprocess_astronomy():
    """Preprocess SDSS astronomy data."""
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats

    # Load all FITS files
    frames = discover_sdss_frames()
    train_frames, val_frames, test_frames = split_frames(frames)

    # Process each frame
    for split_name, frame_list in [('train', train_frames),
                                   ('val', val_frames),
                                   ('test', test_frames)]:

        for frame_id, frame_path in enumerate(frame_list):
            # Load FITS
            with fits.open(frame_path) as hdul:
                data = hdul[0].data.astype(np.float32)
                header = hdul[0].header

                # Extract calibration from header
                gain = header.get('GAIN', 4.62)  # Default for SDSS
                read_noise = header.get('RDNOISE', 4.72)

            # Data is already bias-subtracted and flat-fielded
            electrons = data * gain

            # Background estimation (sigma-clipped median)
            _, bg_median, _ = sigma_clipped_stats(electrons, sigma=3.0)
            background = bg_median

            # Find scale (if not computed yet)
            if not hasattr(preprocess_astronomy, 'scale'):
                # Use 99.9th percentile of this and other frames
                preprocess_astronomy.scale = np.percentile(electrons, 99.9)
            scale = preprocess_astronomy.scale

            # Normalize
            noisy_norm = electrons / scale
            noisy_norm = noisy_norm[np.newaxis, :, :]  # Add channel

            # Cosmic ray detection (simple)
            from scipy.ndimage import median_filter
            med = median_filter(data, size=5)
            cosmic_rays = (data - med) > 5 * read_noise

            # Save for posterior (no clean GT for real observations)
            scene_data = {
                'noisy_norm': torch.from_numpy(noisy_norm).float(),
                'clean_norm': None,  # No ground truth
                'calibration': {
                    'scale': scale,
                    'gain': gain,
                    'read_noise': read_noise,
                    'background': background,
                    'black_level': 0.0,  # Already subtracted
                    'white_level': 65535.0,
                },
                'masks': {
                    'valid': torch.from_numpy(~cosmic_rays)[None, :, :],
                    'saturated': torch.from_numpy(data > 60000)[None, :, :],
                },
                'metadata': {
                    'domain_id': 2,
                    'scene_id': f"frame_{frame_id:05d}",
                    'original_shape': list(data.shape),
                    'bit_depth': 16,
                }
            }

            save_path = f"data/preprocessed/posterior/astronomy/{split_name}/"
            save_path += f"frame_{frame_id:05d}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(scene_data, save_path)

            # For prior training, use external clean images or coadds
            # (Not shown here - depends on your clean data source)
```

---

## Part D: Validation and Quality Control

### D.1 Validation Script (`scripts/validate_preprocessing.py`)

```python
#!/usr/bin/env python
"""Validate preprocessed data meets specifications."""

import torch
import numpy as np
import json
from pathlib import Path

def validate_domain(domain: str):
    """Complete validation for one domain."""

    print(f"\n=== Validating {domain} ===")

    # Load manifest
    manifest_path = f"data/preprocessed/manifests/{domain}.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check prior training tiles
    prior_dir = Path(f"data/preprocessed/prior_clean/{domain}/train")
    tiles = list(prior_dir.glob("*.pt"))

    print(f"Checking {len(tiles)} prior training tiles...")

    for i, tile_path in enumerate(tiles[:100]):  # Sample
        data = torch.load(tile_path)

        # Check structure
        assert 'clean_norm' in data
        assert 'domain_id' in data
        assert 'metadata' in data

        # Check dimensions
        clean = data['clean_norm']
        expected_channels = manifest['channels']
        assert clean.shape == (expected_channels, 128, 128), \
            f"Wrong shape: {clean.shape}"

        # Check range
        assert clean.min() >= 0, f"Negative values: {clean.min()}"
        assert clean.max() <= 2.0, f"Too large: {clean.max()}"

        # Check normalization (most pixels should be < 1.0)
        assert (clean < 1.0).float().mean() > 0.9, \
            "Poor normalization"

    # Check posterior files
    posterior_dir = Path(f"data/preprocessed/posterior/{domain}/train")
    scenes = list(posterior_dir.glob("*.pt"))

    print(f"Checking {len(scenes)} posterior scenes...")

    chi2_values = []

    for scene_path in scenes[:20]:  # Sample
        data = torch.load(scene_path)

        # Check structure
        assert 'noisy_norm' in data
        assert 'calibration' in data
        assert 'masks' in data

        # If ground truth exists, compute chi-squared
        if data['clean_norm'] is not None:
            noisy = data['noisy_norm']
            clean = data['clean_norm']
            calib = data['calibration']

            # Compute expected signal
            lambda_e = clean * calib['scale']

            # Compute variance
            var = lambda_e + calib['read_noise']**2

            # Chi-squared
            chi2 = ((noisy * calib['scale'] - lambda_e)**2 / var).mean()
            chi2_values.append(chi2.item())

    if chi2_values:
        mean_chi2 = np.mean(chi2_values)
        print(f"Mean chi-squared: {mean_chi2:.3f} (should be ~1.0)")
        assert 0.5 < mean_chi2 < 2.0, f"Bad chi-squared: {mean_chi2}"

    # Check no scene overlap between splits
    train_scenes = set(s.stem for s in Path(f"data/preprocessed/posterior/{domain}/train").glob("*.pt"))
    val_scenes = set(s.stem for s in Path(f"data/preprocessed/posterior/{domain}/val").glob("*.pt"))
    test_scenes = set(s.stem for s in Path(f"data/preprocessed/posterior/{domain}/test").glob("*.pt"))

    assert len(train_scenes & val_scenes) == 0, "Train/val overlap!"
    assert len(train_scenes & test_scenes) == 0, "Train/test overlap!"
    assert len(val_scenes & test_scenes) == 0, "Val/test overlap!"

    print(f"✓ {domain} validation passed!")

    return True

def main():
    """Run all validations."""

    domains = ['photography', 'microscopy', 'astronomy']

    for domain in domains:
        try:
            validate_domain(domain)
        except AssertionError as e:
            print(f"✗ {domain} failed: {e}")
            return False

    print("\n=== All validations passed! ===")
    return True

if __name__ == "__main__":
    main()
```

### D.2 Quick Inspection Script (`scripts/inspect_data.py`)

```python
#!/usr/bin/env python
"""Quick visual inspection of preprocessed data."""

import torch
import matplotlib.pyplot as plt
import numpy as np

def inspect_sample(filepath):
    """Visualize one preprocessed file."""

    data = torch.load(filepath)

    if 'clean_norm' in data and data['clean_norm'] is not None:
        # Prior training tile
        img = data['clean_norm'].numpy()

        if img.shape[0] == 4:  # Photography RGGB
            # Simple demosaic for visualization
            rgb = np.zeros((3, img.shape[1], img.shape[2]))
            rgb[0] = img[0]  # R
            rgb[1] = (img[1] + img[2]) / 2  # G
            rgb[2] = img[3]  # B
            img = rgb
        elif img.shape[0] == 1:  # Grayscale
            img = img[0]

        plt.figure(figsize=(8, 8))
        if len(img.shape) == 3:
            plt.imshow(np.transpose(img, (1, 2, 0)))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(f"Clean normalized\nShape: {img.shape}, Range: [{img.min():.3f}, {img.max():.3f}]")
        plt.colorbar()

    if 'noisy_norm' in data:
        # Posterior file - show noisy and clean
        noisy = data['noisy_norm'].numpy()
        clean = data.get('clean_norm')

        fig, axes = plt.subplots(1, 2 if clean is not None else 1, figsize=(12, 6))

        if noisy.shape[0] == 1:
            noisy_show = noisy[0]
        else:
            noisy_show = noisy

        ax = axes[0] if clean is not None else axes
        ax.imshow(noisy_show if len(noisy_show.shape) == 2 else np.transpose(noisy_show, (1,2,0)))
        ax.set_title(f"Noisy normalized\nRange: [{noisy.min():.3f}, {noisy.max():.3f}]")

        if clean is not None:
            clean = clean.numpy()
            if clean.shape[0] == 1:
                clean_show = clean[0]
            else:
                clean_show = clean
            axes[1].imshow(clean_show if len(clean_show.shape) == 2 else np.transpose(clean_show, (1,2,0)))
            axes[1].set_title(f"Clean normalized\nRange: [{clean.min():.3f}, {clean.max():.3f}]")

        # Print calibration
        print("Calibration:")
        for k, v in data['calibration'].items():
            print(f"  {k}: {v}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inspect_data.py <path_to_pt_file>")
    else:
        inspect_sample(sys.argv[1])
```

---

## Part E: Running the Complete Pipeline

### Step-by-Step Execution Guide for Junior Engineer

**1. Environment Setup**
```bash
# Create conda environment
conda create -n poisson-diff python=3.10
conda activate poisson-diff

# Install dependencies
pip install torch torchvision numpy scipy
pip install rawpy tifffile astropy scikit-learn scikit-image
pip install tqdm matplotlib
```

**2. Download Raw Data**
```bash
# Photography (SID)
wget https://storage.googleapis.com/isl-datasets/SID/Sony.zip
unzip Sony.zip -d data/raw/SID/

# Microscopy (example - adjust for your dataset)
# Download FMD or your microscopy data to data/raw/FMD/

# Astronomy (SDSS)
python scripts/download_sdss.py --output data/raw/SDSS/
```

**3. Run Preprocessing**
```bash
# Process each domain separately
python scripts/preprocess_photography.py
python scripts/preprocess_microscopy.py
python scripts/preprocess_astronomy.py
```

**4. Validate Output**
```bash
# Run validation checks
python scripts/validate_preprocessing.py

# Should see:
# ✓ photography validation passed!
# ✓ microscopy validation passed!
# ✓ astronomy validation passed!
# === All validations passed! ===
```

**5. Inspect Samples**
```bash
# Look at a few samples
python scripts/inspect_data.py data/preprocessed/prior_clean/photography/train/tile_000000.pt
python scripts/inspect_data.py data/preprocessed/posterior/photography/train/scene_00000.pt
```

**6. Check Statistics**
```bash
# View manifest files
cat data/preprocessed/manifests/photography.json
cat data/preprocessed/manifests/microscopy.json
cat data/preprocessed/manifests/astronomy.json
```

---

## Part F: Integration with Model Training

### How the Model Uses Preprocessed Data

```python
# In your training script:

class PreprocessedDataset(Dataset):
    """Dataset that loads preprocessed .pt files."""

    def __init__(self, root: str, split: str, mode: str):
        """
        Args:
            root: Path to preprocessed data
            split: 'train', 'val', or 'test'
            mode: 'prior' or 'posterior'
        """
        self.root = Path(root)
        self.split = split
        self.mode = mode

        if mode == 'prior':
            # Load clean tiles for prior training
            self.files = list((self.root / 'prior_clean' / '*' / split).glob('*.pt'))
        else:
            # Load full scenes for posterior evaluation
            self.files = list((self.root / 'posterior' / '*' / split).glob('*.pt'))

        print(f"Found {len(self.files)} files for {mode}/{split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])

        if self.mode == 'prior':
            # Return clean image and domain conditioning
            return {
                'image': data['clean_norm'],
                'domain_id': data['domain_id']
            }
        else:
            # Return everything for posterior sampling
            return data

# Usage in training:
train_dataset = PreprocessedDataset(
    root='data/preprocessed',
    split='train',
    mode='prior'
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
for batch in train_loader:
    clean = batch['image']  # [B, C, 128, 128]
    domain = batch['domain_id']  # [B]

    # Train diffusion prior
    loss = model.training_step(clean, condition=domain)
    ...
```

---

## Part G: Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| "Negative values after black subtraction" | Use `np.maximum(data - black, 0)` to clip |
| "Scale too large/small" | Check if using 99.9th percentile of CLEAN images in electrons |
| "Chi-squared not near 1.0" | Verify gain and read noise estimates |
| "Memory issues with large images" | Process in tiles, not full frames |
| "Inconsistent channels" | Photography=4 (RGGB), Others=1 |
| "Can't find clean astronomy data" | Use coadds or synthetic clean images |

---

## Summary

This preprocessing pipeline:
1. **Converts** raw sensor data to standardized `.pt` files
2. **Normalizes** all intensities by domain-specific scale `s`
3. **Preserves** calibration for physical noise modeling
4. **Outputs** exactly what the model expects: 128×128 tiles
5. **Validates** output quality with chi-squared and range checks

The key insight: **Preprocessing happens once**, creating clean `.pt` files that are fast to load during training. This ensures reproducibility and eliminates on-the-fly processing overhead.
