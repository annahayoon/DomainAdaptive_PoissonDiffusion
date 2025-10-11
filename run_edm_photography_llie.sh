#!/bin/bash

# EDM Photography LLIE (Low-Light Image Enhancement) Script
# Runs the comprehensive EDM restoration visualization script for low-light image enhancement

MODEL_PATH="results/edm_pt_training_20251008_032055/best_model.pkl"
OUTPUT_DIR="results/edm_photography_llie"
DOMAIN="photography"
DATA_ROOT="/home/jilab/Jae/dataset/processed/pt_tiles"
METADATA_JSON="/home/jilab/Jae/dataset/processed/metadata_photography_incremental.json"
NUM_STEPS=18
NUM_SAMPLES=2

echo "==========================================="
echo "EDM PHOTOGRAPHY LLIE (LOW-LIGHT IMAGE ENHANCEMENT)"
echo "==========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_ROOT"
echo "Metadata: $METADATA_JSON"
echo "Output: $OUTPUT_DIR"
echo "Domain: $DOMAIN"
echo "Steps: $NUM_STEPS"
echo "Samples: $NUM_SAMPLES"
echo "================================================="

# Run the comprehensive restoration script
python sample_pt_edm_photography_llie.py \
    --model_path "$MODEL_PATH" \
    --data_root "$DATA_ROOT" \
    --metadata_json "$METADATA_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --domain "$DOMAIN" \
    --num_steps "$NUM_STEPS" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size 1 \
    --device cuda \
    --seed 42

echo "==========================================="
echo "LLIE COMPLETED"
echo "==========================================="
echo "Check the output directory: $OUTPUT_DIR"
echo "for comprehensive visualizations showing:"
echo "  - Step-by-step low-light enhancement process"
echo "  - Model space vs physical units"
echo "  - Display normalization examples"
echo "  - Before/after comparisons with real noisy inputs"
echo "==========================================="
