#!/bin/bash

# Comprehensive EDM Denoising Script (Starting from Noisy Test Images)
# Runs the comprehensive denoising visualization script with noisy test images from the dataset

MODEL_PATH="results/edm_pt_training_photography_20251008_032055/best_model.pkl"
METADATA_JSON="dataset/processed/comprehensive_tiles_metadata.json"
NOISY_DIR="dataset/processed/pt_tiles/photography/noisy"
OUTPUT_DIR="results/edm_denoising_photography_test"
DOMAIN="photography"
NUM_STEPS=18
NUM_SAMPLES=8
# IMPORTANT: noise_sigma is in [-1,1] model space, NOT physical units
# Start small and increase if denoising is insufficient
# Too high values cause over-smoothing!
NOISE_SIGMA=0.001  # Very light noise (try 0.001, 0.002, 0.005, 0.01 for comparison)

echo "========================================"
echo "EDM DENOISING FROM NOISY TEST IMAGES"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Metadata: $METADATA_JSON"
echo "Noisy images: $NOISY_DIR"
echo "Output: $OUTPUT_DIR"
echo "Domain: $DOMAIN"
echo "Steps: $NUM_STEPS"
echo "Noise sigma: $NOISE_SIGMA"
echo "Samples: $NUM_SAMPLES"
echo "========================================"

# Run the denoising script
python sample/sample_noisy_pt.py \
    --model_path "$MODEL_PATH" \
    --metadata_json "$METADATA_JSON" \
    --noisy_dir "$NOISY_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --domain "$DOMAIN" \
    --num_steps "$NUM_STEPS" \
    --noise_sigma "$NOISE_SIGMA" \
    --num_samples "$NUM_SAMPLES" \
    --device cuda \
    --seed 42

echo "========================================"
echo "DENOISING COMPLETED"
echo "========================================"
echo "Check the output directory: $OUTPUT_DIR"
echo "for comprehensive visualizations showing:"
echo "  - Step-by-step denoising process"
echo "  - Noisy input vs denoised output"
echo "  - Model space vs physical units"
echo "  - Display normalization examples"
echo "========================================"

