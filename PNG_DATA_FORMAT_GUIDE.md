# PNG Data Format Guide

## Overview

The diffusion model training pipeline uses **8-bit PNG images** as the exclusive data format. This provides the simplest and most accessible way to train the model on your photography data.

> **Note**: Legacy parquet and .pt format support has been moved to the `old/` folder. PNG is now the only supported format.

## ✅ **CONFIRMED: PNG Compatibility**

Your diffusion model training from `run_photography_train.sh` **CAN** train directly on 8-bit PNG images with the new `PNGDataset` implementation.

## PNG Data Format Specifications

### Supported Image Formats
- **Primary**: 8-bit PNG files (`.png`, `.PNG`)
- **Color Modes**: RGB, RGBA, Grayscale, Grayscale+Alpha
- **Bit Depth**: 8-bit (0-255 pixel values)
- **Size**: Any size (automatically resized to target resolution)

### Automatic Processing Pipeline

1. **Image Loading**: PIL-based loading with format detection
2. **Channel Conversion**: 
   - RGB → RGBA (adds alpha channel)
   - Grayscale → RGBA (converts and adds alpha)
   - Any format → 4-channel RGBA for photography domain
3. **Resizing**: Automatic resize to target size (default 128×128)
4. **Normalization**: Converts 8-bit values (0-255) to float (-1.0 to 1.0) for diffusion training
5. **Noise Simulation**: Realistic Poisson + Gaussian noise generation

### Directory Structure

The training script will automatically detect PNG files in any subdirectory structure:

```
/your/data/directory/
├── image1.png
├── image2.png
├── subfolder/
│   ├── image3.png
│   └── image4.png
└── another_folder/
    └── more_images.png
```

## Model Architecture (Unchanged)

The model architecture remains the same 810M parameter EDM model:

- **Input Channels**: 4 (RGBA)
- **Output Channels**: 4 (denoised RGBA)
- **Image Resolution**: 128×128 (configurable via `--target_size`)
- **Architecture**: EDM with 256 base channels, [1,2,3,4] channel multipliers
- **Attention**: Multi-scale attention at 8×8, 16×16, 32×32 resolutions

## Usage Instructions

### 1. Prepare Your PNG Data

Simply place your 8-bit PNG images in a directory:

```bash
# Example data structure
mkdir -p /path/to/your/png/data
cp your_images/*.png /path/to/your/png/data/
```

### 2. Update Data Path

Edit `run_photography_train.sh` or use the command line:

```bash
# In run_photography_train.sh, the script will automatically detect PNG files at:
PNG_DATA_PATH="/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data"

# Or specify directly when running:
python train_photography_model.py --data_root /path/to/your/png/data
```

### 3. Run Training

```bash
cd /home/jilab/Jae
./run_photography_train.sh
```

The script will automatically:
- Detect PNG files in the data directory
- Create train/val/test splits (80%/10%/10%)
- Load and preprocess images on-the-fly
- Generate realistic noise for training

## Data Processing Details

### Noise Simulation

The `PNGDataset` automatically generates realistic camera noise:

```python
# Convert from [-1, 1] to [0, 1] for electron count calculation
clean_01 = (clean_image + 1.0) / 2.0
electrons = clean_01 * 1000.0

# Add Poisson noise (shot noise)
poisson_noise = torch.poisson(electrons) - electrons

# Add Gaussian read noise
read_noise_std = 2.0
gaussian_noise = torch.randn_like(electrons) * read_noise_std

# Combine and normalize back to [-1, 1]
noisy_electrons = electrons + poisson_noise + gaussian_noise
noisy_01 = torch.clamp(noisy_electrons / 1000.0, 0.0, 1.0)
noisy = noisy_01 * 2.0 - 1.0  # Convert to [-1, 1]
```

### Train/Val/Test Splits

- **Automatic Splitting**: 80% train, 10% validation, 10% test
- **Reproducible**: Uses seed for consistent splits across runs
- **Balanced**: Ensures good distribution across splits

### Memory Efficiency

- **On-the-fly Loading**: Images loaded and processed during training
- **Automatic Resizing**: Reduces memory usage for large images
- **Efficient Transforms**: Uses optimized torchvision transforms

## Performance Considerations

### Advantages of PNG Format:
- **Simplicity**: No preprocessing required
- **Accessibility**: Standard image format
- **Flexibility**: Any image size, color mode supported
- **Quality**: Lossless compression preserves image quality

### Potential Considerations:
- **I/O Overhead**: PNG decompression during training
- **Storage**: Less compressed than parquet format
- **Processing**: Real-time noise simulation

### Optimization Tips:
1. **Use SSD Storage**: Faster PNG loading
2. **Optimize Workers**: Tune `--num_workers` for PNG loading
3. **Prefetching**: Use `--prefetch_factor 4` for better GPU utilization
4. **Image Size**: Consider resizing large images beforehand if memory is limited

## Supported Format

The training script now supports only:

- **PNG** - 8-bit PNG images with [-1, 1] normalization

Legacy formats (parquet, .pt) have been moved to `old/` folder for reference.

## Configuration Options

### Command Line Arguments:

```bash
python train_photography_model.py \
    --data_root /path/to/png/images \
    --target_size 128 \              # Image resolution
    --batch_size 4 \                 # Batch size
    --max_files 1000 \              # Limit number of images (optional)
    --seed 42                       # Random seed for splits
```

### Model Architecture Options:

```bash
# Standard configuration (810M parameters)
--model_channels 256 \
--channel_mult 1 2 3 4 \
--num_blocks 6 \
--attn_resolutions 8 16 32

# Memory-optimized configuration
--model_channels 128 \
--channel_mult 1 2 2 \
--num_blocks 3
```

## Validation and Testing

### Quick Test:
```python
from data.png_dataset import PNGDataset

# Test PNG loading
dataset = PNGDataset("/path/to/png/data", split="train")
sample = dataset[0]

print(f"Clean shape: {sample['clean'].shape}")      # Should be (4, 128, 128)
print(f"Noisy shape: {sample['noisy'].shape}")      # Should be (4, 128, 128)
print(f"Value range: {sample['clean'].min():.3f} - {sample['clean'].max():.3f}")  # Should be -1.0 to 1.0
```

### Training Validation:
The training script will automatically validate:
- PNG file detection and loading
- Proper tensor shapes and data types
- Train/val split creation
- Noise simulation quality

## Migration from Other Formats

### From Legacy Formats (Parquet/.pt):
- Legacy loaders moved to `old/` folder
- Convert existing data to PNG format
- Use PNG images directly with current pipeline

### From Other Image Formats:
```python
# Convert JPEG/TIFF/etc to PNG
from PIL import Image
import os

def convert_to_png(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.tiff', '.bmp')):
            img = Image.open(os.path.join(input_dir, file))
            png_name = os.path.splitext(file)[0] + '.png'
            img.save(os.path.join(output_dir, png_name), 'PNG')
```

## Troubleshooting

### Common Issues:

1. **No PNG files found**:
   - Check file extensions (`.png`, `.PNG`)
   - Verify directory path
   - Ensure files are not corrupted

2. **Memory issues**:
   - Reduce `--batch_size`
   - Use smaller `--target_size`
   - Reduce `--num_workers`

3. **Slow loading**:
   - Use SSD storage
   - Increase `--prefetch_factor`
   - Consider preprocessing large images

4. **Color channel issues**:
   - PNG dataset automatically handles RGB/RGBA conversion
   - Check image mode with PIL: `Image.open(file).mode`

## Conclusion

The PNG data format provides the most straightforward way to train your diffusion model on photography data. The automatic detection, preprocessing, and noise simulation make it ideal for rapid experimentation and deployment.

**Ready to use**: Simply place your PNG images in a directory and run the training script - everything else is handled automatically!
