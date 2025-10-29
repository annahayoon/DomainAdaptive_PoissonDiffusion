#!/bin/bash

# Ultra-Low Sigma Sweep Script
# Tests extremely low sigma values (0.0000001 to 0.000001) for very clean images
# Based on noise analysis showing estimated noise std: 0.000059 - 0.000156

MODEL_PATH="results/edm_pt_training_photography_20251008_032055/best_model.pkl"
METADATA_JSON="dataset/processed/comprehensive_tiles_metadata.json"
NOISY_DIR="dataset/processed/pt_tiles/photography/noisy"
OUTPUT_DIR="results/ultra_low_sigma_sweep"
DOMAIN="photography"
NUM_STEPS=18
NUM_EXAMPLES=3

echo "========================================"
echo "ULTRA-LOW SIGMA SWEEP"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Metadata: $METADATA_JSON"
echo "Noisy images: $NOISY_DIR"
echo "Output: $OUTPUT_DIR"
echo "Domain: $DOMAIN"
echo "Steps: $NUM_STEPS"
echo "Examples: $NUM_EXAMPLES"
echo "Sigma range: 0.0000001 to 0.000001"
echo "Target: Very clean images (noise std ~0.0001)"
echo "========================================"

# Run the ultra-low sigma sweep
python sample/sigma_sweep_comprehensive.py \
    --model_path "$MODEL_PATH" \
    --metadata_json "$METADATA_JSON" \
    --noisy_dir "$NOISY_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --domain "$DOMAIN" \
    --num_steps "$NUM_STEPS" \
    --num_examples "$NUM_EXAMPLES" \
    --device cuda \
    --seed 42

echo "========================================"
echo "ULTRA-LOW SIGMA SWEEP COMPLETED"
echo "========================================"
echo "Check the output directory: $OUTPUT_DIR"
echo "for comprehensive visualizations showing:"
echo "  - Same noisy input across ultra-low sigma values"
echo "  - Denoised outputs for each sigma"
echo "  - Model space vs physical units comparison"
echo "  - Ultra-low sigma range: 0.0000001 to 0.000001"
echo "========================================"
