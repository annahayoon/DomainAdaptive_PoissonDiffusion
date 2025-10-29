#!/bin/bash

# Comprehensive Enhancement Analysis on All Test Tiles
# Tests all available test tiles and generates:
# 1. DataFrame with all metrics (CSV)
# 2. Histogram distributions per sigma value
# 3. Best sigma recommendation

MODEL_PATH="results/edm_pt_training_photography_20251008_032055/best_model.pkl"
METADATA_JSON="dataset/processed/comprehensive_tiles_metadata.json"
NOISY_DIR="dataset/processed/pt_tiles/photography/noisy"
OUTPUT_DIR="results/comprehensive_enhancement_analysis"
DOMAIN="photography"
NUM_STEPS=18
NUM_SAMPLES=50  # Number of tiles to sample

echo "========================================"
echo "COMPREHENSIVE ENHANCEMENT ANALYSIS"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Metadata: $METADATA_JSON"
echo "Noisy images: $NOISY_DIR"
echo "Output: $OUTPUT_DIR"
echo "Domain: $DOMAIN"
echo "Steps: $NUM_STEPS"
echo "Sigma range: 0.0002 to 0.002 (14 values)"
echo "Number of samples: $NUM_SAMPLES"
echo "========================================"
echo "This will test $NUM_SAMPLES randomly sampled tiles"
echo "Expected time: ~50 minutes for 50 tiles"
echo "========================================"

# Run the comprehensive analysis
python sample/comprehensive_enhancement_analysis.py \
    --model_path "$MODEL_PATH" \
    --metadata_json "$METADATA_JSON" \
    --noisy_dir "$NOISY_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --domain "$DOMAIN" \
    --num_steps "$NUM_STEPS" \
    --num_samples "$NUM_SAMPLES" \
    --device cuda

echo "========================================"
echo "COMPREHENSIVE ANALYSIS COMPLETED"
echo "========================================"
echo "Check the output directory: $OUTPUT_DIR"
echo "Files generated:"
echo "  - enhancement_metrics_all_tiles.csv (DataFrame)"
echo "  - enhancement_summary_statistics.csv (Summary)"
echo "  - histogram_*_by_sigma.png (Distributions)"
echo "  - best_sigma_recommendation.json (Recommendation)"
echo "========================================"
