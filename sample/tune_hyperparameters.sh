#!/bin/bash
# Script to tune kappa and num_steps hyperparameters
# Tests multiple combinations and finds optimal settings
# Uses VALIDATION tiles (not test tiles) for proper hyperparameter tuning
#
# Usage:
#   ./sample/tune_hyperparameters.sh [sensor] [num_tiles] [save_viz]
#
# Performs separate parameter sweeps:
#   1. Sweep kappa values (with fixed num_steps=18)
#   2. Sweep num_steps values (with fixed kappa=0.15)
#
# Examples:
#   ./sample/tune_hyperparameters.sh sony 10 false    # Fast: no visualizations
#   ./sample/tune_hyperparameters.sh sony 10 true     # Slower: saves .png comparison images
#   ./sample/tune_hyperparameters.sh fuji 20 true     # Tune Fuji with 20 tiles + visualizations

set -e  # Exit on error
set -o pipefail  # Exit on error in pipes

# Configuration
SENSOR="${1:-sony}"  # Default to sony, can pass fuji as argument
NUM_EXAMPLES="${2:-10}"  # Number of validation tiles to use for tuning
SAVE_VISUALIZATIONS="${3:-false}"  # Set to "true" to save .png comparison images (slower but allows visual comparison)
DATA_ROOT="dataset/processed"
CALIBRATION_DIR="${DATA_ROOT}"
OUTPUT_BASE="results/hyperparameter_tuning"

# Find latest model for sensor
if [ "$SENSOR" = "sony" ]; then
    MODEL_PATTERN="results/edm_sony_training_*/best_model.pkl"
    METADATA_JSON="${DATA_ROOT}/metadata_sony_incremental.json"
elif [ "$SENSOR" = "fuji" ]; then
    MODEL_PATTERN="results/edm_fuji_training_*/best_model.pkl"
    METADATA_JSON="${DATA_ROOT}/metadata_fuji_incremental.json"
else
    echo "ERROR: Invalid sensor. Use 'sony' or 'fuji'"
    exit 1
fi

MODEL_PATH=$(ls -td ${MODEL_PATTERN%/best_model.pkl} 2>/dev/null | head -1)/best_model.pkl

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    exit 1
fi

SHORT_DIR="${DATA_ROOT}/pt_tiles/${SENSOR}/short"
LONG_DIR="${DATA_ROOT}/pt_tiles/${SENSOR}/long"

# Check calibration file
CALIBRATION_FILE="${CALIBRATION_DIR}/${SENSOR}_noise_calibration.json"
if [ ! -f "$CALIBRATION_FILE" ]; then
    echo "ERROR: Calibration file not found: $CALIBRATION_FILE"
    echo "Please run sensor_noise_calibrations.py first"
    exit 1
fi

# Hyperparameter ranges to test
KAPPA_VALUES=(0.05 0.1 0.15 0.2 0.3)
NUM_STEPS_VALUES=(10 18 25 30)

# Fixed values for separate sweeps
FIXED_KAPPA=0.15      # Fixed kappa when sweeping num_steps
FIXED_NUM_STEPS=18    # Fixed num_steps when sweeping kappa

# Create output directory
OUTPUT_DIR="${OUTPUT_BASE}/${SENSOR}"
mkdir -p "${OUTPUT_DIR}"

# Load validation tile IDs from split file (like sensor_noise_calibrations.py does)
echo "Loading validation tiles from split file..."
SPLITS_DIR="${DATA_ROOT}/../splits"
if [ ! -d "$SPLITS_DIR" ]; then
    SPLITS_DIR="dataset/splits"
fi

SENSOR_CAPITALIZED=$(echo "${SENSOR}" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
SPLIT_FILE="${SPLITS_DIR}/${SENSOR_CAPITALIZED}_val_list.txt"

if [ ! -f "$SPLIT_FILE" ]; then
    echo "ERROR: Split file not found: $SPLIT_FILE"
    echo "Looking for: ${SENSOR_CAPITALIZED}_val_list.txt in ${SPLITS_DIR}"
    exit 1
fi

VALIDATION_TILE_IDS=$(python3 << EOF
import json
import sys
import random
from pathlib import Path

try:
    # Set random seed for reproducibility
    random.seed(42)

    # Load metadata
    metadata_path = Path("${METADATA_JSON}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load split file to get filenames
    split_file = Path("${SPLIT_FILE}")
    split_filenames = set()
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # Parse format: ./Sensor/short/filename.ext ./Sensor/long/filename.ext ISO F
            short_path = parts[0]
            short_filename = Path(short_path).name
            split_filenames.add(short_filename)

    # Match files by filename in metadata
    files = metadata.get("files", [])
    matching_file_paths = set()
    for file_data in files:
        file_meta = file_data.get("file_metadata", {})
        file_path = file_meta.get("file_path", "")
        if file_path:
            file_name = Path(file_path).name
            if file_name in split_filenames:
                matching_file_paths.add(file_path)

    # Collect all tile IDs from matching files (short exposure only)
    matching_tile_ids = []
    for file_data in files:
        file_meta = file_data.get("file_metadata", {})
        file_path = file_meta.get("file_path", "")

        if file_path not in matching_file_paths:
            continue

        tiles = file_data.get("tiles", [])
        for tile in tiles:
            tile_id = tile.get("tile_id")
            tile_data_type = tile.get("data_type", "")

            if tile_id and tile_data_type == "short":
                matching_tile_ids.append(tile_id)

    # Randomly sample the requested number of tiles
    num_tiles = ${NUM_EXAMPLES}
    if len(matching_tile_ids) < num_tiles:
        print(f"WARNING: Only found {len(matching_tile_ids)} validation tiles, using all of them", file=sys.stderr)
        selected_tiles = matching_tile_ids
    else:
        selected_tiles = random.sample(matching_tile_ids, num_tiles)

    print(" ".join(selected_tiles))

except Exception as e:
    print(f"Error loading validation tiles: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
EOF
)

if [ -z "$VALIDATION_TILE_IDS" ]; then
    echo "ERROR: No validation tiles found for sensor ${SENSOR}"
    exit 1
fi

NUM_VALIDATION_TILES=$(echo $VALIDATION_TILE_IDS | wc -w)
echo "Found ${NUM_VALIDATION_TILES} validation tiles"

echo ""
echo "=========================================="
echo "Hyperparameter Tuning (Using VALIDATION tiles)"
echo "=========================================="
echo "Sensor: $SENSOR"
echo "Model: $MODEL_PATH"
echo "Num validation tiles: $NUM_VALIDATION_TILES"
echo "Method: Separate parameter sweeps"
echo "Kappa values to test: ${KAPPA_VALUES[@]} (with num_steps=${FIXED_NUM_STEPS})"
echo "Num steps values to test: ${NUM_STEPS_VALUES[@]} (with kappa=${FIXED_KAPPA})"
echo "Total runs: $((${#KAPPA_VALUES[@]} + ${#NUM_STEPS_VALUES[@]}))"
echo "Save visualizations: $SAVE_VISUALIZATIONS"
echo "Output: $OUTPUT_DIR"
echo ""

# Results file
RESULTS_FILE="${OUTPUT_DIR}/tuning_results.json"
RESULTS_CSV="${OUTPUT_DIR}/tuning_results.csv"

# Initialize results
echo "{" > "${RESULTS_FILE}"
echo "  \"sensor\": \"${SENSOR}\"," >> "${RESULTS_FILE}"
echo "  \"model_path\": \"${MODEL_PATH}\"," >> "${RESULTS_FILE}"
echo "  \"num_validation_tiles\": ${NUM_VALIDATION_TILES}," >> "${RESULTS_FILE}"
echo "  \"split\": \"val\"," >> "${RESULTS_FILE}"
echo "  \"kappa_values\": [$(IFS=,; echo "${KAPPA_VALUES[*]}")]," >> "${RESULTS_FILE}"
echo "  \"num_steps_values\": [$(IFS=,; echo "${NUM_STEPS_VALUES[*]}")]," >> "${RESULTS_FILE}"
echo "  \"results\": [" >> "${RESULTS_FILE}"

# CSV header
echo "kappa,num_steps,psnr_mean,psnr_std,ssim_mean,ssim_std,mse_mean,mse_std,avg_time_per_tile" > "${RESULTS_CSV}"

TOTAL_RUNS=$((${#KAPPA_VALUES[@]} + ${#NUM_STEPS_VALUES[@]}))
RUN_COUNT=0
FIRST_RESULT=true

# Function to test a single combination
test_combination() {
    local kappa=$1
    local num_steps=$2
    local sweep_type=$3  # "kappa" or "num_steps"

    RUN_COUNT=$((RUN_COUNT + 1))

    echo ""
    echo "=========================================="
    echo "Testing: kappa=${kappa}, num_steps=${num_steps} (${sweep_type} sweep)"
    echo "Progress: ${RUN_COUNT}/${TOTAL_RUNS}"
    echo "=========================================="

    # Create unique output directory for this combination
    COMBO_OUTPUT_DIR="${OUTPUT_DIR}/kappa_${kappa}_steps_${num_steps}"

    # Run sampling with this combination
    START_TIME=$(date +%s)

    # Build visualization flag
    if [ "$SAVE_VISUALIZATIONS" = "true" ]; then
        VIZ_FLAG=""
    else
        VIZ_FLAG="--skip_visualization"
    fi

    # Run sampling - continue even if this combination fails
    if ! python sample/sample_noisy_pt_lle_PGguidance.py \
        --model_path "$MODEL_PATH" \
        --metadata_json "$METADATA_JSON" \
        --short_dir "$SHORT_DIR" \
        --long_dir "$LONG_DIR" \
        --output_dir "$COMBO_OUTPUT_DIR" \
        --tile_ids $VALIDATION_TILE_IDS \
        --use_sensor_calibration \
        --calibration_dir "${CALIBRATION_DIR}" \
        --run_methods pg_x0 \
        --kappa $kappa \
        --num_steps $num_steps \
        --guidance_level x0 \
        --pg_mode wls \
        --device cuda \
        --seed 42 \
        $VIZ_FLAG 2>&1 | tee "${COMBO_OUTPUT_DIR}/run.log"; then
        echo "ERROR: Failed to run sampling for kappa=${kappa}, num_steps=${num_steps}"
        echo "Skipping this combination and continuing..."
        return 0  # Return 0 to continue with next combination
    fi

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    AVG_TIME_PER_TILE=$(echo "scale=2; $ELAPSED / $NUM_VALIDATION_TILES" | bc)

    # Extract metrics from results.json
    RESULTS_JSON="${COMBO_OUTPUT_DIR}/results.json"

    if [ ! -f "$RESULTS_JSON" ]; then
        echo "WARNING: Results file not found: $RESULTS_JSON"
        echo "Skipping metrics extraction for this combination..."
        return 0  # Return 0 to continue with next combination
    fi

    # Extract aggregate metrics using Python
    METRICS=$(python3 << EOF
import json
import sys
import numpy as np

try:
    with open('${RESULTS_JSON}', 'r') as f:
        data = json.load(f)

    # Collect metrics from all results
    psnr_values = []
    ssim_values = []
    mse_values = []

    for result in data.get('results', []):
        if 'comprehensive_metrics' in result:
            metrics = result['comprehensive_metrics'].get('pg_x0', {})
            if 'psnr' in metrics and not np.isnan(metrics['psnr']):
                psnr_values.append(metrics['psnr'])
            if 'ssim' in metrics and not np.isnan(metrics['ssim']):
                ssim_values.append(metrics['ssim'])
            if 'mse' in metrics and not np.isnan(metrics['mse']):
                mse_values.append(metrics['mse'])

    if psnr_values:
        psnr_mean = np.mean(psnr_values)
        psnr_std = np.std(psnr_values)
    else:
        psnr_mean = 0.0
        psnr_std = 0.0

    if ssim_values:
        ssim_mean = np.mean(ssim_values)
        ssim_std = np.std(ssim_values)
    else:
        ssim_mean = 0.0
        ssim_std = 0.0

    if mse_values:
        mse_mean = np.mean(mse_values)
        mse_std = np.std(mse_values)
    else:
        mse_mean = 0.0
        mse_std = 0.0

    print(f"{psnr_mean:.4f},{psnr_std:.4f},{ssim_mean:.6f},{ssim_std:.6f},{mse_mean:.8f},{mse_std:.8f}")

except Exception as e:
    print(f"Error extracting metrics: {e}", file=sys.stderr)
    print("0.0,0.0,0.0,0.0,0.0,0.0")  # Print default values to stdout, continue execution
    # Don't exit - let script continue with next combination
EOF
)

    # Validate METRICS was extracted successfully
    if [ -z "$METRICS" ]; then
        echo "WARNING: Failed to extract metrics, using default values"
        METRICS="0.0,0.0,0.0,0.0,0.0,0.0"
    fi

    # Write to CSV
    echo "${kappa},${num_steps},${METRICS},${AVG_TIME_PER_TILE}" >> "${RESULTS_CSV}"

    # Parse metrics for JSON
    IFS=',' read -r psnr_mean psnr_std ssim_mean ssim_std mse_mean mse_std <<< "$METRICS"

    # Write to JSON
    if [ "$FIRST_RESULT" = false ]; then
        echo "," >> "${RESULTS_FILE}"
    fi
    FIRST_RESULT=false

    cat >> "${RESULTS_FILE}" << EOF
    {
      "kappa": ${kappa},
      "num_steps": ${num_steps},
      "metrics": {
        "psnr": {"mean": ${psnr_mean}, "std": ${psnr_std}},
        "ssim": {"mean": ${ssim_mean}, "std": ${ssim_std}},
        "mse": {"mean": ${mse_mean}, "std": ${mse_std}}
      },
      "avg_time_per_tile": ${AVG_TIME_PER_TILE},
      "output_dir": "${COMBO_OUTPUT_DIR}"
    }
EOF

    printf "Results: PSNR=%.2f±%.2f, SSIM=%.4f±%.4f, Time=%.2fs/tile\n" "$psnr_mean" "$psnr_std" "$ssim_mean" "$ssim_std" "$AVG_TIME_PER_TILE"
}

# Sweep 1: Vary kappa, keep num_steps fixed
echo ""
echo "=========================================="
echo "Sweep 1: Varying kappa (num_steps=${FIXED_NUM_STEPS})"
echo "=========================================="
for kappa in "${KAPPA_VALUES[@]}"; do
    test_combination "$kappa" "$FIXED_NUM_STEPS" "kappa"
done

# Sweep 2: Vary num_steps, keep kappa fixed
echo ""
echo "=========================================="
echo "Sweep 2: Varying num_steps (kappa=${FIXED_KAPPA})"
echo "=========================================="
for num_steps in "${NUM_STEPS_VALUES[@]}"; do
    test_combination "$FIXED_KAPPA" "$num_steps" "num_steps"
done

# Close JSON
echo "  ]" >> "${RESULTS_FILE}"
echo "}" >> "${RESULTS_FILE}"

# Find best combination
echo ""
echo "=========================================="
echo "Finding Best Hyperparameters"
echo "=========================================="

BEST=$(python3 << EOF
import json
import sys

try:
    with open('${RESULTS_FILE}', 'r') as f:
        data = json.load(f)

    results = data['results']

    # Separate results by sweep type
    kappa_sweep_results = [r for r in results if r['num_steps'] == ${FIXED_NUM_STEPS}]
    num_steps_sweep_results = [r for r in results if r['kappa'] == ${FIXED_KAPPA}]

    # Find best kappa (from kappa sweep)
    if kappa_sweep_results:
        best_kappa_psnr = max(kappa_sweep_results, key=lambda x: x['metrics']['psnr']['mean'])
        best_kappa_ssim = max(kappa_sweep_results, key=lambda x: x['metrics']['ssim']['mean'])

        print("Best kappa (from kappa sweep, num_steps=${FIXED_NUM_STEPS}):")
        print("  By PSNR:")
        print(f"    kappa={best_kappa_psnr['kappa']}, num_steps={best_kappa_psnr['num_steps']}")
        print(f"    PSNR: {best_kappa_psnr['metrics']['psnr']['mean']:.2f}±{best_kappa_psnr['metrics']['psnr']['std']:.2f}")
        print(f"    SSIM: {best_kappa_psnr['metrics']['ssim']['mean']:.4f}±{best_kappa_psnr['metrics']['ssim']['std']:.4f}")
        print(f"    Time: {best_kappa_psnr['avg_time_per_tile']:.2f}s/tile")
        print("  By SSIM:")
        print(f"    kappa={best_kappa_ssim['kappa']}, num_steps={best_kappa_ssim['num_steps']}")
        print(f"    PSNR: {best_kappa_ssim['metrics']['psnr']['mean']:.2f}±{best_kappa_ssim['metrics']['psnr']['std']:.2f}")
        print(f"    SSIM: {best_kappa_ssim['metrics']['ssim']['mean']:.4f}±{best_kappa_ssim['metrics']['ssim']['std']:.4f}")
        print(f"    Time: {best_kappa_ssim['avg_time_per_tile']:.2f}s/tile")
        print()

    # Find best num_steps (from num_steps sweep)
    if num_steps_sweep_results:
        best_num_steps_psnr = max(num_steps_sweep_results, key=lambda x: x['metrics']['psnr']['mean'])
        best_num_steps_ssim = max(num_steps_sweep_results, key=lambda x: x['metrics']['ssim']['mean'])

        print("Best num_steps (from num_steps sweep, kappa=${FIXED_KAPPA}):")
        print("  By PSNR:")
        print(f"    kappa={best_num_steps_psnr['kappa']}, num_steps={best_num_steps_psnr['num_steps']}")
        print(f"    PSNR: {best_num_steps_psnr['metrics']['psnr']['mean']:.2f}±{best_num_steps_psnr['metrics']['psnr']['std']:.2f}")
        print(f"    SSIM: {best_num_steps_psnr['metrics']['ssim']['mean']:.4f}±{best_num_steps_psnr['metrics']['ssim']['std']:.4f}")
        print(f"    Time: {best_num_steps_psnr['avg_time_per_tile']:.2f}s/tile")
        print("  By SSIM:")
        print(f"    kappa={best_num_steps_ssim['kappa']}, num_steps={best_num_steps_ssim['num_steps']}")
        print(f"    PSNR: {best_num_steps_ssim['metrics']['psnr']['mean']:.2f}±{best_num_steps_ssim['metrics']['psnr']['std']:.2f}")
        print(f"    SSIM: {best_num_steps_ssim['metrics']['ssim']['mean']:.4f}±{best_num_steps_ssim['metrics']['ssim']['std']:.4f}")
        print(f"    Time: {best_num_steps_ssim['avg_time_per_tile']:.2f}s/tile")
        print()

    # Overall best across all results
    best_overall_psnr = max(results, key=lambda x: x['metrics']['psnr']['mean'])
    best_overall_ssim = max(results, key=lambda x: x['metrics']['ssim']['mean'])

    print("Overall best (across all runs):")
    print("  By PSNR:")
    print(f"    kappa={best_overall_psnr['kappa']}, num_steps={best_overall_psnr['num_steps']}")
    print(f"    PSNR: {best_overall_psnr['metrics']['psnr']['mean']:.2f}±{best_overall_psnr['metrics']['psnr']['std']:.2f}")
    print(f"    SSIM: {best_overall_psnr['metrics']['ssim']['mean']:.4f}±{best_overall_psnr['metrics']['ssim']['std']:.4f}")
    print(f"    Time: {best_overall_psnr['avg_time_per_tile']:.2f}s/tile")
    print("  By SSIM:")
    print(f"    kappa={best_overall_ssim['kappa']}, num_steps={best_overall_ssim['num_steps']}")
    print(f"    PSNR: {best_overall_ssim['metrics']['psnr']['mean']:.2f}±{best_overall_ssim['metrics']['psnr']['std']:.2f}")
    print(f"    SSIM: {best_overall_ssim['metrics']['ssim']['mean']:.4f}±{best_overall_ssim['metrics']['ssim']['std']:.4f}")
    print(f"    Time: {best_overall_ssim['avg_time_per_tile']:.2f}s/tile")

except Exception as e:
    print(f"Error finding best: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

echo "$BEST"

echo ""
echo "=========================================="
echo "Tuning Complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - JSON: ${RESULTS_FILE}"
echo "  - CSV: ${RESULTS_CSV}"
echo ""
echo "Individual results in: ${OUTPUT_DIR}/kappa_*_steps_*"
echo ""
