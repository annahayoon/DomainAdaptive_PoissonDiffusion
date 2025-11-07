#!/bin/bash
# Script to run calibrated sampling for all test tiles per sensor
# Uses calibration JSON files from sensor_noise_calibrations.py

set -e  # Exit on error

# Configuration
DATA_ROOT="dataset/processed"
METADATA_JSON="${DATA_ROOT}/metadata_sony_incremental.json"  # Will be updated per sensor
SHORT_DIR="${DATA_ROOT}/pt_tiles"
LONG_DIR="${DATA_ROOT}/pt_tiles"
CALIBRATION_DIR="${DATA_ROOT}"
OUTPUT_BASE="results/calibrated_sampling"

# Sensor configurations
declare -A SENSOR_MODELS=(
    ["sony"]="results/edm_sony_training_*/best_model.pkl"
    ["fuji"]="results/edm_fuji_training_*/best_model.pkl"
)

declare -A SENSOR_METADATA=(
    ["sony"]="${DATA_ROOT}/metadata_sony_incremental.json"
    ["fuji"]="${DATA_ROOT}/metadata_fuji_incremental.json"
)

# Function to find the latest model for a sensor
find_latest_model() {
    local sensor=$1
    local pattern="${SENSOR_MODELS[$sensor]}"

    # Find all matching directories and get the latest one
    local latest_dir=$(ls -td ${pattern%/best_model.pkl} 2>/dev/null | head -1)

    if [ -z "$latest_dir" ]; then
        echo "ERROR: No model found for sensor $sensor matching pattern $pattern" >&2
        return 1
    fi

    local model_path="${latest_dir}/best_model.pkl"

    if [ ! -f "$model_path" ]; then
        echo "ERROR: Model file not found: $model_path" >&2
        return 1
    fi

    echo "$model_path"
}

# Function to run sampling for a sensor
run_sensor_sampling() {
    local sensor=$1
    local model_path=$2
    local metadata_json="${SENSOR_METADATA[$sensor]}"
    local output_dir="${OUTPUT_BASE}/${sensor}_all_test_tiles"
    local short_dir="${SHORT_DIR}/${sensor}/short"
    local long_dir="${SHORT_DIR}/${sensor}/long"

    echo "=========================================="
    echo "Running sampling for sensor: $sensor"
    echo "=========================================="
    echo "Model: $model_path"
    echo "Metadata: $metadata_json"
    echo "Output: $output_dir"
    echo ""

    # Check if calibration file exists (calibration is enabled by default)
    local calibration_file="${CALIBRATION_DIR}/${sensor}_noise_calibration.json"
    if [ ! -f "$calibration_file" ]; then
        echo "ERROR: Calibration file not found: $calibration_file"
        echo "Calibration is enabled by default. Please run sensor_noise_calibrations.py first to generate calibration files"
        echo "Or use --no-noise-calibration to disable (not recommended)"
        return 1
    else
        echo "Using calibration: $calibration_file"
        USE_CALIBRATION_FLAG="--calibration_dir ${CALIBRATION_DIR}"
    fi

    # Run sampling for all test tiles
    # Note: num_examples should be set to a large number to process all tiles
    # The script will automatically limit to available test tiles
    # Noise calibration is ENABLED BY DEFAULT - sigma_r is automatically computed from calibration JSON
    python sample/sample_noisy_pt_lle_PGguidance.py \
        --model_path "$model_path" \
        --metadata_json "$metadata_json" \
        --short_dir "$short_dir" \
        --long_dir "$long_dir" \
        --output_dir "$output_dir" \
        --sensor_filter "$sensor" \
        --num_examples 10000 \
        --use_sensor_calibration \
        $USE_CALIBRATION_FLAG \
        --run_methods short long exposure_scaled gaussian_x0 pg_x0 \
        --guidance_level x0 \
        --pg_mode wls \
        --device cuda \
        --seed 42

    # sigma_r is automatically computed from calibration JSON:
    #   sigma_r = sqrt(b_calibration * (sensor_range/2)^2)
    # where b_calibration comes from {sensor}_noise_calibration.json
    # Calibration is enabled by default - no need to specify --use_noise_calibration

    echo ""
    echo "Completed sampling for $sensor"
    echo "Results saved to: $output_dir"
    echo ""
}

# Main execution
main() {
    echo "Calibrated Sampling for All Test Tiles"
    echo "======================================"
    echo ""

    # Process each sensor
    for sensor in sony fuji; do
        echo "Finding model for sensor: $sensor"
        model_path=$(find_latest_model "$sensor")

        if [ $? -eq 0 ]; then
            run_sensor_sampling "$sensor" "$model_path"
        else
            echo "Skipping sensor $sensor due to missing model"
            echo ""
        fi
    done

    echo "=========================================="
    echo "All sampling completed!"
    echo "=========================================="
}

# Run main function
main
