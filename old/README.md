# Old/Deprecated Data Format Support

This folder contains deprecated data loading scripts that are no longer used in the main training pipeline.

## Moved Files

### Parquet Format Support (Deprecated)
- `parquet_dataset.py` - Dataset loader for compressed parquet tiles
- `PARQUET_COMPATIBILITY_ANALYSIS.md` - Analysis of parquet format compatibility

### Preprocessed Tensor Format Support (Deprecated)  
- `preprocessed_datasets.py` - Dataset loader for .pt tensor files

## Current Data Format

The training pipeline now uses **PNG format only** with the following benefits:

- **Simplicity**: Standard 8-bit PNG images
- **Accessibility**: No preprocessing required
- **Flexibility**: Any image size, color mode supported
- **Normalization**: Automatic [-1, 1] normalization for diffusion training

## Migration

If you need to use these old formats:

1. **From Parquet**: Convert parquet tiles to PNG images
2. **From .pt**: Convert tensor files to PNG images
3. **Current**: Use PNG images directly with `data/png_dataset.py`

## Usage (Deprecated)

These scripts are kept for reference only. To use them, you would need to:

1. Move the desired script back to the `data/` directory
2. Update the training script imports
3. Ensure dependencies are available

**Note**: The current PNG-based pipeline is recommended for all new projects.
