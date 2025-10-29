#!/bin/bash

# Low-Light Image Enhancement Script
# Uses EDM model to enhance low-light images to normal brightness levels

MODEL_PATH="results/edm_pt_training_photography_20251008_032055/best_model.pkl"
METADATA_JSON="dataset/processed/comprehensive_tiles_metadata.json"
NOISY_DIR="dataset/processed/pt_tiles/photography/noisy"
OUTPUT_DIR="results/low_light_enhancement"
DOMAIN="photography"
NUM_STEPS=18
NUM_EXAMPLES=3

echo "========================================"
echo "LOW-LIGHT IMAGE ENHANCEMENT"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Metadata: $METADATA_JSON"
echo "Noisy images: $NOISY_DIR"
echo "Output: $OUTPUT_DIR"
echo "Domain: $DOMAIN"
echo "Steps: $NUM_STEPS"
echo "Examples: $NUM_EXAMPLES"
echo "Enhancement sigma range: 0.0002 to 0.002 (fine-grained)"
echo "========================================"

# Run the low-light enhancement script
python sample/low_light_enhancement.py \
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
echo "LOW-LIGHT ENHANCEMENT COMPLETED"
echo "========================================"
echo "Check the output directory: $OUTPUT_DIR"
echo "for comprehensive visualizations showing:"
echo "  - Original reference images"
echo "  - Simulated low-light inputs"
echo "  - Enhanced outputs across sigma range"
echo "  - Enhancement sigma range: 0.0002 to 0.002 (fine-grained)"
echo "========================================"
