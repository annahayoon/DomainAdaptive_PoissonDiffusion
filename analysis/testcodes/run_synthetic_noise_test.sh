#!/bin/bash

# Test EDM Denoising with Synthetic Gaussian Noise
# Adds known noise to clean images, then denoises

MODEL_PATH="results/edm_pt_training_sony_20251008_032055/best_model.pkl"
CLEAN_DIR="dataset/processed/pt_tiles/sony/clean"
OUTPUT_DIR="results/synthetic_noise_denoising_test"
NOISE_SIGMA=0.05  # Very small noise level (subtle degradation)
NUM_STEPS=18
DOMAIN="sony"  # Use 'sony' or 'fuji'
NUM_SAMPLES=8

echo "========================================"
echo "EDM DENOISING TEST"
echo "Clean + Synthetic Gaussian Noise"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Clean images: $CLEAN_DIR"
echo "Output: $OUTPUT_DIR"
echo "Noise sigma: $NOISE_SIGMA"
echo "Denoising steps: $NUM_STEPS"
echo "Samples: $NUM_SAMPLES"
echo "========================================"

python sample/sample_clean_plus_synthetic_noise.py \
    --model_path "$MODEL_PATH" \
    --clean_dir "$CLEAN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --noise_sigma "$NOISE_SIGMA" \
    --num_steps "$NUM_STEPS" \
    --domain "$DOMAIN" \
    --num_samples "$NUM_SAMPLES" \
    --device cuda \
    --seed 42

echo "========================================"
echo "TEST COMPLETED"
echo "========================================"
echo "Results in: $OUTPUT_DIR"
echo "Compare Clean vs Noisy vs Denoised"
echo "Check PSNR improvements"
echo "========================================"
