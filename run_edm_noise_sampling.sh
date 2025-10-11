#!/bin/bash

# Comprehensive EDM Sampling Script
# Runs the comprehensive sampling visualization script with the trained model

MODEL_PATH="results/edm_pt_training_20251008_032055/best_model.pkl"
OUTPUT_DIR="results/comprehensive_sampling_viz"
DOMAIN="photography"
NUM_STEPS=18
NUM_SAMPLES=4

echo "========================================"
echo "COMPREHENSIVE EDM SAMPLING VISUALIZATION"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Domain: $DOMAIN"
echo "Steps: $NUM_STEPS"
echo "Samples: $NUM_SAMPLES"
echo "========================================"

# Run the comprehensive sampling script
python sample_pt_edm_native_comprehensive_visualization.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --domain "$DOMAIN" \
    --num_steps "$NUM_STEPS" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size 1 \
    --device cuda \
    --seed 42

echo "========================================"
echo "SAMPLING COMPLETED"
echo "========================================"
echo "Check the output directory: $OUTPUT_DIR"
echo "for comprehensive visualizations showing:"
echo "  - Step-by-step denoising process"
echo "  - Model space vs physical units"
echo "  - Display normalization examples"
echo "========================================"
