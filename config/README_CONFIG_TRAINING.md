# Training Configuration Files

This directory contains YAML configuration files for training EDM models on different camera brands.

**Note**: Currently, all configurations use the same metadata file. To train on specific camera brands, you would need to create separate metadata files or modify the dataset to support sensor filtering.

## Available Configurations

### 1. `photo.yaml` (Default)
General photography configuration - trains on all available sensors.
- **Data**: All camera brands from `dataset/processed/pt_tiles/`
- **Usage**: `./train/run_edm_pt_photography_train.sh`

### 2. `sony.yaml`
Sony camera only configuration (currently same as default - requires metadata filtering).
- **Data**: All sensors (config doesn't currently filter by sensor type)
- **Usage**: `CONFIG_FILE="config/sony.yaml" ./train/run_edm_pt_photography_train.sh`

### 3. `fuji.yaml`
Fuji camera only configuration (currently same as default - requires metadata filtering).
- **Data**: All sensors (config doesn't currently filter by sensor type)
- **Usage**: `CONFIG_FILE="config/fuji.yaml" ./train/run_edm_pt_photography_train.sh`

### 4. `sony_fuji.yaml`
Combined Sony + Fuji cameras configuration (currently same as default - requires metadata filtering).
- **Data**: All sensors (config doesn't currently filter by sensor type)
- **Usage**: `CONFIG_FILE="config/sony_fuji.yaml" ./train/run_edm_pt_photography_train.sh`

## Configuration Format

Each config file contains:

```yaml
# Training hyperparameters
batch_size: 4
learning_rate: 0.0001
total_kimg: 300

# EMA configuration
ema_halflife_kimg: 50

# Learning rate schedule
lr_rampup_kimg: 10

# Training progress
kimg_per_tick: 12
snapshot_ticks: 2

# Early stopping
early_stopping_patience: 5

# Model architecture
img_resolution: 256
channels: 3
model_channels: 192
channel_mult: [1, 2, 3, 4]

# Device and seed
device: cuda
seed: 42

# Dataset configuration
data_root: dataset/processed
metadata_json: dataset/processed/metadata_photography_incremental.json
# Note: sensor_types field exists in config but is not currently used by the dataset loader
```

## Usage Examples

### Train on Sony only:
```bash
CONFIG_FILE="config/sony.yaml" ./train/run_edm_pt_photography_train.sh
```

### Train on Fuji only:
```bash
CONFIG_FILE="config/fuji.yaml" ./train/run_edm_pt_photography_train.sh
```

### Train on Sony + Fuji combined:
```bash
CONFIG_FILE="config/sony_fuji.yaml" ./train/run_edm_pt_photography_train.sh
```

### Use default config (all sensors):
```bash
./train/run_edm_pt_photography_train.sh
```

## How It Works

1. **Config Loading**: The training script loads hyperparameters from the YAML config file
2. **Clean/Noisy Filtering**: The dataset loader filters tiles to only include clean tiles (filters out noisy tiles)
3. **Path Resolution**: Data loader searches for .pt files in the specified data_root directory
4. **Training**: Only clean tiles are used for training the prior P(x_clean)

## Current Limitations

**Important**: The current implementation does not support filtering by sensor type. The `sensor_types` field in the config files is present but not used. All configurations currently train on the same dataset from the metadata file.

To train on specific camera brands, you would need to:
1. Create separate metadata files for each brand, OR
2. Add sensor type filtering back to the dataset loader

## Custom Configurations

You can create custom config files by copying one of the existing configs and modifying:
- `data_root`: Path to the processed data directory
- `metadata_json`: Path to your metadata file
- Any other hyperparameters as needed

## Notes

- The dataset loader filters out noisy tiles, keeping only clean tiles for training
- All config files currently use the same metadata file and data directory
- The dataset works with clean/noisy pairs, loading clean images for unconditional training
