#!/bin/bash

# Sigma Sweep Script - Test Multiple Noise Levels
# This script runs denoising with different sigma values to help you find the optimal setting

MODEL_PATH="results/edm_pt_training_photography_20251008_032055/best_model.pkl"
METADATA_JSON="dataset/processed/comprehensive_tiles_metadata.json"
NOISY_DIR="dataset/processed/pt_tiles/photography/noisy"
BASE_OUTPUT_DIR="results/edm_denoising_photography_sigma_sweep"
DOMAIN="photography"
NUM_STEPS=18
NUM_SAMPLES=2  # Use fewer samples for quick comparison

# Test different sigma values - methodical sweep from very low to moderate noise
SIGMAS=(0.0001 0.0002 0.0005 0.001 0.002 0.005)

echo "========================================"
echo "EDM DENOISING - SIGMA SWEEP TEST"
echo "========================================"
echo "Testing noise sigma values: ${SIGMAS[@]}"
echo "This helps find the optimal denoising strength"
echo "========================================"

for SIGMA in "${SIGMAS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/sigma_${SIGMA}"
    
    echo ""
    echo "Testing sigma = $SIGMA"
    echo "Output: $OUTPUT_DIR"
    echo "----------------------------------------"
    
    python sample/sample_noisy_pt.py \
        --model_path "$MODEL_PATH" \
        --metadata_json "$METADATA_JSON" \
        --noisy_dir "$NOISY_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --domain "$DOMAIN" \
        --num_steps "$NUM_STEPS" \
        --noise_sigma "$SIGMA" \
        --num_samples "$NUM_SAMPLES" \
        --device cuda \
        --seed 42
    
    echo "✓ Sigma $SIGMA completed"
done

echo ""
echo "========================================"
echo "SIGMA SWEEP COMPLETED"
echo "========================================"
echo "Results saved to: $BASE_OUTPUT_DIR"
echo ""

# Create comprehensive comparison visualization
echo "Creating sigma sweep comparison visualization..."
python sample/create_sigma_sweep_comparison.py \
    --sweep_dir "$BASE_OUTPUT_DIR" \
    --output_path "${BASE_OUTPUT_DIR}/sigma_sweep_comparison.png" \
    --domain "$DOMAIN"

echo ""
echo "Compare the results to find the optimal noise level:"
echo "  - Too small: insufficient denoising"
echo "  - Too large: over-smoothing, loss of detail"
echo "  - Just right: removes noise while preserving details"
echo ""
echo "Check: ${BASE_OUTPUT_DIR}/sigma_sweep_comparison.png"
echo "========================================"

