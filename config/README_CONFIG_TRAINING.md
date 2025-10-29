# Training Configuration Files

This directory contains YAML configuration files for training EDM models on different camera brands.

## Available Configurations

### 1. `photo.yaml` (Default)
General photography configuration - trains on all available sensors.
- **Data**: All camera brands from `dataset/processed/pt_tiles/`
- **Sensor Filter**: None (includes all sensors)

### 2. `sony.yaml`
Sony camera only configuration.
- **Data**: Sony camera data from `dataset/processed/pt_tiles/sony/long`
- **Sensor Filter**: `sony` only
- **Usage**: `CONFIG_FILE="config/sony.yaml" ./train/run_edm_pt_photography_train.sh`

### 3. `fuji.yaml`
Fuji camera only configuration.
- **Data**: Fuji camera data from `dataset/processed/pt_tiles/fuji/long`
- **Sensor Filter**: `fuji` only
- **Usage**: `CONFIG_FILE="config/fuji.yaml" ./train/run_edm_pt_photography_train.sh`

### 4. `sony_fuji.yaml`
Combined Sony + Fuji cameras configuration.
- **Data**: Both Sony and Fuji camera data
- **Sensor Filter**: `sony` and `fuji`
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
sensor_types: [sony]  # or [fuji] or [sony, fuji]
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
2. **Sensor Filtering**: The dataset loader filters tiles by `sensor_type` field from metadata
3. **Path Resolution**: Data loader automatically resolves paths based on `sensor_type` to find the correct subdirectory
4. **Training**: Only tiles matching the specified sensor types are included in training

## Custom Configurations

You can create custom config files by copying one of the existing configs and modifying:
- `sensor_types`: List of sensor types to include (e.g., `[sony]`, `[fuji]`, `[sony, fuji]`)
- `data_root`: Path to the processed data directory
- Any other hyperparameters as needed

## Notes

- The dataset loader automatically handles path resolution based on `sensor_type` from metadata
- Metadata JSON contains all tiles, but training only uses 'long' exposure tiles
- Sensor type filtering happens at dataset load time for efficiency
