# Quick Start: EDM Native Training

## TL;DR

```bash
# 1. Verify setup
python test_edm_native_setup.py

# 2. Start training
bash run_edm_native_train.sh
```

## What Changed?

✅ Created new EDM native training script using EDM's original loss function
✅ Updated data pipeline to keep [0, 255] range (more explicit normalization)
✅ Simplified from ~1,600 lines to ~400 lines
✅ Uses one-hot domain encoding: photography = [1, 0, 0]

## Data Flow

```
PNG Files (uint8)  →  Dataset (float32)  →  Training (float32)  →  EDM Model
   [0, 255]              [0, 255]              [-1, 1]              [-1, 1]
```

Conversion in training script: `images = (clean / 127.5) - 1.0`

## Training Command

```bash
python train_photography_edm_native.py \
    --data_root /home/jilab/Jae/dataset/processed/png_tiles/photography \
    --metadata_json /home/jilab/Jae/dataset/processed/metadata_photography_incremental.json \
    --batch_size 4 \
    --total_kimg 10000 \
    --img_resolution 256 \
    --model_channels 128
```

Or use the provided script:
```bash
bash run_edm_native_train.sh
```

## Verification

All tests should pass:
```bash
python test_edm_native_setup.py          # 6/6 tests ✓
python verify_png_normalization.py       # Normalization ✓
python test_png_dataset_integration.py   # 7/7 dataset tests ✓
```

## Key Files

| File | What It Does |
|------|--------------|
| `train_photography_edm_native.py` | Main training script |
| `run_edm_native_train.sh` | Shell script to start training |
| `data/png_dataset.py` | Dataset loader (keeps [0, 255] range) |
| `EDM_NATIVE_TRAINING.md` | Full documentation |

## Training Parameters

Default values in `run_edm_native_train.sh`:
- Batch size: 4
- Learning rate: 1e-4
- Total training: 10,000 kimg (10 million images)
- Image resolution: 256×256
- Model channels: 128
- EMA half-life: 500 kimg
- Save interval: 500 kimg

## Output

Checkpoints saved to: `results/edm_native_photography/checkpoints/`
- `checkpoint-000500.pt` (every 500 kimg)
- `best_model.pt` (best validation loss)

## Documentation

Read the full docs:
- **Quick Start**: This file
- **Full Guide**: `EDM_NATIVE_TRAINING.md`
- **Comparison**: `TRAINING_COMPARISON.md`
- **Changes**: `CHANGES_SUMMARY.md`

## Ready to Go!

Everything is tested and working. Just run:
```bash
bash run_edm_native_train.sh
```
