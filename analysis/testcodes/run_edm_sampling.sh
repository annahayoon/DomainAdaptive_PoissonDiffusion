#!/bin/bash
# Unified EDM Sampling Script
# Supports multiple sampling modes: noisy, pure_noise, and synthetic
#
# Usage:
#   ./test/run_edm_sampling.sh [MODE] [OPTIONS...]
#
# Arguments:
#   MODE: 'noisy', 'pure_noise', or 'synthetic' (default: 'pure_noise')
#
# Options (mode-specific):
#   For 'noisy' mode:
#     --model_path MODEL_PATH
#     --metadata_json METADATA_JSON
#     --noisy_dir NOISY_DIR
#     --domain DOMAIN (sony/fuji)
#     --noise_sigma NOISE_SIGMA (default: 0.001)
#
#   For 'pure_noise' mode:
#     --model_path MODEL_PATH
#     --domain DOMAIN (sony/fuji)
#     --batch_size BATCH_SIZE (default: 1)
#
#   For 'synthetic' mode:
#     --model_path MODEL_PATH
#     --clean_dir CLEAN_DIR
#     --domain DOMAIN (sony/fuji)
#     --noise_sigma NOISE_SIGMA (default: 0.05)
#
# Common options:
#   --output_dir OUTPUT_DIR
#   --num_steps NUM_STEPS (default: 18)
#   --num_samples NUM_SAMPLES (default: 4 for pure_noise, 8 for others)
#   --device DEVICE (default: cuda)
#   --seed SEED (default: 42)
#
# Examples:
#   # Pure noise generation (default)
#   ./test/run_edm_sampling.sh pure_noise --model_path results/edm_pt_training_20251008_032055/best_model.pkl --domain sony
#
#   # Denoising from noisy images
#   ./test/run_edm_sampling.sh noisy --model_path results/edm_pt_training_sony_20251008_032055/best_model.pkl --metadata_json dataset/processed/comprehensive_tiles_metadata.json --noisy_dir dataset/processed/pt_tiles/sony/noisy --domain sony
#
#   # Synthetic noise test
#   ./test/run_edm_sampling.sh synthetic --model_path results/edm_pt_training_sony_20251008_032055/best_model.pkl --clean_dir dataset/processed/pt_tiles/sony/clean --domain sony

set -e

# Parse mode
MODE="${1:-pure_noise}"

# Validate mode
if [ "$MODE" != "noisy" ] && [ "$MODE" != "pure_noise" ] && [ "$MODE" != "synthetic" ]; then
    echo "ERROR: Invalid MODE. Use 'noisy', 'pure_noise', or 'synthetic'"
    echo "Usage: $0 [MODE] [OPTIONS...]"
    exit 1
fi

# Shift to parse remaining arguments
shift

# Default values
NUM_STEPS=18
NUM_SAMPLES=4
DOMAIN="sony"
DEVICE="cuda"
SEED=42
NOISE_SIGMA=0.001
BATCH_SIZE=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --metadata_json)
            METADATA_JSON="$2"
            shift 2
            ;;
        --noisy_dir)
            NOISY_DIR="$2"
            shift 2
            ;;
        --clean_dir)
            CLEAN_DIR="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --noise_sigma)
            NOISE_SIGMA="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default NUM_SAMPLES based on mode if not specified
if [ "$MODE" != "pure_noise" ] && [ "$NUM_SAMPLES" = "4" ]; then
    NUM_SAMPLES=8
fi

# Process based on mode
case "$MODE" in
    pure_noise)
        echo "========================================"
        echo "COMPREHENSIVE EDM SAMPLING VISUALIZATION"
        echo "Mode: Pure Noise Generation"
        echo "========================================"

        if [ -z "$MODEL_PATH" ]; then
            MODEL_PATH="results/edm_pt_training_20251008_032055/best_model.pkl"
        fi

        if [ -z "$OUTPUT_DIR" ]; then
            OUTPUT_DIR="results/comprehensive_sampling_viz"
        fi

        echo "Model: $MODEL_PATH"
        echo "Output: $OUTPUT_DIR"
        echo "Domain: $DOMAIN"
        echo "Steps: $NUM_STEPS"
        echo "Samples: $NUM_SAMPLES"
        echo "Batch size: $BATCH_SIZE"
        echo "========================================"

        # Run the comprehensive sampling script
        python test/sample_noise.py \
            --model_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --domain "$DOMAIN" \
            --num_steps "$NUM_STEPS" \
            --num_samples "$NUM_SAMPLES" \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --seed "$SEED"

        echo ""
        echo "========================================"
        echo "SAMPLING COMPLETED"
        echo "========================================"
        echo "Check the output directory: $OUTPUT_DIR"
        echo "for comprehensive visualizations showing:"
        echo "  - Step-by-step denoising process"
        echo "  - Model space vs physical units"
        echo "  - Display normalization examples"
        echo "========================================"
        ;;

    noisy)
        echo "========================================"
        echo "EDM DENOISING FROM NOISY TEST IMAGES"
        echo "Mode: Noisy Image Denoising"
        echo "========================================"

        if [ -z "$MODEL_PATH" ]; then
            MODEL_PATH="results/edm_pt_training_sony_20251008_032055/best_model.pkl"
        fi

        if [ -z "$METADATA_JSON" ]; then
            METADATA_JSON="dataset/processed/comprehensive_tiles_metadata.json"
        fi

        if [ -z "$NOISY_DIR" ]; then
            NOISY_DIR="dataset/processed/pt_tiles/${DOMAIN}/noisy"
        fi

        if [ -z "$OUTPUT_DIR" ]; then
            OUTPUT_DIR="results/edm_denoising_${DOMAIN}_test"
        fi

        echo "Model: $MODEL_PATH"
        echo "Metadata: $METADATA_JSON"
        echo "Noisy images: $NOISY_DIR"
        echo "Output: $OUTPUT_DIR"
        echo "Domain: $DOMAIN"
        echo "Steps: $NUM_STEPS"
        echo "Noise sigma: $NOISE_SIGMA"
        echo "Samples: $NUM_SAMPLES"
        echo "========================================"
        echo "NOTE: noise_sigma is in [-1,1] model space, NOT physical units"
        echo "Start small and increase if denoising is insufficient"
        echo "Too high values cause over-smoothing!"
        echo "========================================"

        # Check if sample_noisy_pt.py exists
        if [ ! -f "test/sample_noisy_pt.py" ]; then
            echo "ERROR: test/sample_noisy_pt.py not found"
            echo "This script may need to be created or the path may be incorrect"
            exit 1
        fi

        # Run the denoising script
        python test/sample_noisy_pt.py \
            --model_path "$MODEL_PATH" \
            --metadata_json "$METADATA_JSON" \
            --noisy_dir "$NOISY_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --domain "$DOMAIN" \
            --num_steps "$NUM_STEPS" \
            --noise_sigma "$NOISE_SIGMA" \
            --num_samples "$NUM_SAMPLES" \
            --device "$DEVICE" \
            --seed "$SEED"

        echo ""
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
        ;;

    synthetic)
        echo "========================================"
        echo "EDM DENOISING TEST"
        echo "Mode: Clean + Synthetic Gaussian Noise"
        echo "========================================"

        if [ -z "$MODEL_PATH" ]; then
            MODEL_PATH="results/edm_pt_training_sony_20251008_032055/best_model.pkl"
        fi

        if [ -z "$CLEAN_DIR" ]; then
            CLEAN_DIR="dataset/processed/pt_tiles/${DOMAIN}/clean"
        fi

        if [ -z "$OUTPUT_DIR" ]; then
            OUTPUT_DIR="results/synthetic_noise_denoising_test"
        fi

        # Default noise sigma for synthetic mode
        if [ "$NOISE_SIGMA" = "0.001" ]; then
            NOISE_SIGMA=0.05
        fi

        echo "Model: $MODEL_PATH"
        echo "Clean images: $CLEAN_DIR"
        echo "Output: $OUTPUT_DIR"
        echo "Noise sigma: $NOISE_SIGMA"
        echo "Denoising steps: $NUM_STEPS"
        echo "Samples: $NUM_SAMPLES"
        echo "Domain: $DOMAIN"
        echo "========================================"

        # Run the synthetic noise test script
        python test/sample_clean_plus_synthetic_noise.py \
            --model_path "$MODEL_PATH" \
            --clean_dir "$CLEAN_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --noise_sigma "$NOISE_SIGMA" \
            --num_steps "$NUM_STEPS" \
            --domain "$DOMAIN" \
            --num_samples "$NUM_SAMPLES" \
            --device "$DEVICE" \
            --seed "$SEED"

        echo ""
        echo "========================================"
        echo "TEST COMPLETED"
        echo "========================================"
        echo "Results in: $OUTPUT_DIR"
        echo "Compare Clean vs Noisy vs Denoised"
        echo "Check PSNR improvements"
        echo "========================================"
        ;;
esac
