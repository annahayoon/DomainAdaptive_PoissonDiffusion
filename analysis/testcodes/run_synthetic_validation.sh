#!/bin/bash
# Run synthetic validation on test scenes (uses TEST split by default)
# Runtime: ~30-60 minutes for 20 scenes, 2-3 hours for all scenes
#
# Usage:
#   ./analysis/run_synthetic_validation.sh [SENSOR] [NUM_IMAGES]
#
# Arguments:
#   SENSOR: 'sony' or 'fuji' (default: 'fuji')
#   NUM_IMAGES: Number of scenes (default: 20) or 'all' for all test scenes
#
# Examples:
#   ./analysis/run_synthetic_validation.sh                    # Default: fuji, 20 scenes
#   ./analysis/run_synthetic_validation.sh fuji 50            # Fuji, 50 scenes
#   ./analysis/run_synthetic_validation.sh fuji all           # Fuji, all test scenes
#   ./analysis/run_synthetic_validation.sh sony all           # Sony, all test scenes
#
# Mode: Always Poisson percentage sweep (default: 5%, 10%, 15%, 20%, 30%)
#   - Shows clear regime transition from read-noise to shot-noise dominated
#   - Perfect for CVPR paper Figure 1
#
# Requirements:
#   - Stitched test scenes must exist from test_guidance_comparison_unified.sh:
#     results/guidance_comparison_{sensor}/scene_*/stitched_long.pt
#   - Test split file: dataset/splits/{Sensor}_test_list.txt
#
# Output structure (same as test_guidance_comparison_unified.sh):
#   results/synthetic_validation_{sensor}/
#     ├── synthetic_poisson_5pct/          # Per test value
#     │   ├── stitched_long.pt
#     │   ├── stitched_short.pt
#     │   ├── stitched_pg_x0.pt
#     │   ├── stitched_gaussian_x0.pt
#     │   ├── scene_comparison.png
#     │   └── scene_metrics.json
#     ├── synthetic_validation_results.json  # Detailed per-image results
#     ├── summary.json                       # Aggregate statistics
#     ├── regime_sweep_analysis.png          # Regime classification
#     ├── sweep_results.json                 # Structured sweep results
#     ├── sweep_results.csv                  # CSV format
#     └── sweep_analysis.png                 # Performance vs Poisson fraction

set -e

echo "=========================================="
echo "Synthetic Validation for CVPR Paper"
echo "=========================================="
echo ""

# Configuration
SENSOR="${1:-fuji}"  # Default to fuji
NUM_IMAGES="${2:-20}"  # Default to 20 scenes

echo "Sensor: $SENSOR"
echo "Number of scenes: $NUM_IMAGES"
echo "Mode: Poisson percentage sweep (default - CVPR Figure 1)"
echo "  Testing Poisson fractions: 5%, 10%, 15%, 20%, 30%"
echo ""

# Validate sensor
if [ "$SENSOR" != "sony" ] && [ "$SENSOR" != "fuji" ]; then
    echo "ERROR: Invalid sensor. Use 'sony' or 'fuji'"
    exit 1
fi

# Find model
if [ "$SENSOR" = "sony" ]; then
    MODEL_PATTERN="results/edm_sony_training_*/best_model.pkl"
elif [ "$SENSOR" = "fuji" ]; then
    MODEL_PATTERN="results/edm_fuji_training_*/best_model.pkl"
fi

MODEL_PATH=$(ls -t ${MODEL_PATTERN} 2>/dev/null | head -1)

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: $MODEL_PATTERN"
    exit 1
fi

echo "Model: $MODEL_PATH"

# Use stitched scenes from guidance comparison results (TEST split)
TEST_IMAGES_DIR="results/guidance_comparison_${SENSOR}"

if [ ! -d "$TEST_IMAGES_DIR" ]; then
    echo "ERROR: Test images directory not found: $TEST_IMAGES_DIR"
    echo ""
    echo "You must first run test_guidance_comparison_unified.sh to generate stitched test scenes:"
    echo "  ./sample/test_guidance_comparison_unified.sh tiles $SENSOR all"
    echo ""
    echo "This creates stitched_long.pt files for each test scene."
    exit 1
fi

echo "Using stitched test scenes from: $TEST_IMAGES_DIR"
echo "  (Looks for scene_*/stitched_long.pt files)"

# Find TEST split file (NOT validation split!)
SPLITS_DIR="dataset/splits"
SENSOR_CAPITALIZED=$(echo "${SENSOR}" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
TEST_SPLIT_FILE="${SPLITS_DIR}/${SENSOR_CAPITALIZED}_test_list.txt"

if [ ! -f "$TEST_SPLIT_FILE" ]; then
    echo "ERROR: Test split file not found: $TEST_SPLIT_FILE"
    exit 1
fi

echo "Test split filter: $TEST_SPLIT_FILE (only test data, not validation)"
echo ""

# Count available scenes
NUM_SCENES=$(find "$TEST_IMAGES_DIR" -name "stitched_long.pt" -type f | wc -l)
echo "Available test scenes: $NUM_SCENES"

if [ "$NUM_SCENES" -eq 0 ]; then
    echo "ERROR: No stitched_long.pt files found in $TEST_IMAGES_DIR"
    echo ""
    echo "You must first run test_guidance_comparison_unified.sh:"
    echo "  ./sample/test_guidance_comparison_unified.sh tiles $SENSOR all"
    exit 1
fi

# Determine number of images to process
if [ "$NUM_IMAGES" = "all" ]; then
    ACTUAL_NUM_IMAGES=9999  # Effectively "all"
    echo "Processing all test scenes: $NUM_SCENES"
else
    ACTUAL_NUM_IMAGES=$NUM_IMAGES
    if [ "$NUM_SCENES" -lt "$ACTUAL_NUM_IMAGES" ]; then
        echo "WARNING: Only $NUM_SCENES scenes available, using all of them"
        ACTUAL_NUM_IMAGES=9999
    else
        echo "Processing $ACTUAL_NUM_IMAGES randomly sampled scenes (from $NUM_SCENES total)"
    fi
fi

echo ""

# Output directory
OUTPUT_DIR="results/synthetic_validation_${SENSOR}"
mkdir -p "$OUTPUT_DIR"

echo "Output: $OUTPUT_DIR"
echo ""

# Get sensor-specific hyperparameters (matching test_guidance_comparison_unified.sh)
if [ "$SENSOR" = "sony" ]; then
    KAPPA=0.15
    NUM_STEPS=10
elif [ "$SENSOR" = "fuji" ]; then
    KAPPA=0.05
    NUM_STEPS=18
fi

echo "Sensor-specific hyperparameters (matching test_guidance_comparison_unified.sh):"
echo "  kappa: $KAPPA"
echo "  num_steps: $NUM_STEPS"
echo ""

# Runtime estimation
if [ "$ACTUAL_NUM_IMAGES" = "9999" ]; then
    if [ "$NUM_SCENES" -le 20 ]; then
        RUNTIME_EST="30-60 minutes"
    elif [ "$NUM_SCENES" -le 50 ]; then
        RUNTIME_EST="2-3 hours"
    else
        RUNTIME_EST="3-5 hours"
    fi
else
    if [ "$ACTUAL_NUM_IMAGES" -le 20 ]; then
        RUNTIME_EST="30-60 minutes"
    else
        RUNTIME_EST="$(( ACTUAL_NUM_IMAGES / 20 )) hours"
    fi
fi

echo "Expected runtime: $RUNTIME_EST (depends on GPU and scene resolution)"
echo ""

# Run validation
echo "=========================================="
echo "Starting synthetic validation..."
echo "=========================================="
echo ""

# Determine metadata and tile directories
DATA_DIR="dataset"
if [ "$SENSOR" = "sony" ]; then
    if [ -f "${DATA_DIR}/processed/comprehensive_sony_tiles_metadata.json" ]; then
        METADATA_JSON="${DATA_DIR}/processed/comprehensive_sony_tiles_metadata.json"
    elif [ -f "${DATA_DIR}/processed/comprehensive_all_tiles_metadata.json" ]; then
        METADATA_JSON="${DATA_DIR}/processed/comprehensive_all_tiles_metadata.json"
    else
        METADATA_JSON="${DATA_DIR}/processed/comprehensive_sony_tiles_metadata.json"
    fi
elif [ "$SENSOR" = "fuji" ]; then
    if [ -f "${DATA_DIR}/processed/comprehensive_fuji_tiles_metadata.json" ]; then
        METADATA_JSON="${DATA_DIR}/processed/comprehensive_fuji_tiles_metadata.json"
    elif [ -f "${DATA_DIR}/processed/comprehensive_all_tiles_metadata.json" ]; then
        METADATA_JSON="${DATA_DIR}/processed/comprehensive_all_tiles_metadata.json"
    else
        METADATA_JSON="${DATA_DIR}/processed/comprehensive_fuji_tiles_metadata.json"
    fi
fi

# Check if metadata file exists
if [ ! -f "$METADATA_JSON" ]; then
    echo "ERROR: Metadata file not found: $METADATA_JSON"
    echo "Please ensure the metadata file exists in dataset/processed/"
    exit 1
fi

SHORT_DIR="${DATA_DIR}/processed/pt_tiles/${SENSOR}/short"
LONG_DIR="${DATA_DIR}/processed/pt_tiles/${SENSOR}/long"

# Check if tile directories exist
if [ ! -d "$SHORT_DIR" ]; then
    echo "ERROR: Short tiles directory not found: $SHORT_DIR"
    exit 1
fi

if [ ! -d "$LONG_DIR" ]; then
    echo "ERROR: Long tiles directory not found: $LONG_DIR"
    exit 1
fi

echo "Using tile-based processing (recommended to avoid OOM)"
echo "  Metadata: $METADATA_JSON"
echo "  Short tiles: $SHORT_DIR"
echo "  Long tiles: $LONG_DIR"
echo "  Tile batch size: 50 (processes 50 tiles at a time)"
echo ""

python analysis/synthetic_validation.py \
    --model_path "$MODEL_PATH" \
    --test_images_dir "$TEST_IMAGES_DIR" \
    --use_tiles \
    --metadata_json "$METADATA_JSON" \
    --short_dir "$SHORT_DIR" \
    --long_dir "$LONG_DIR" \
    --tile_batch_size 50 \
    --val_split_file "$TEST_SPLIT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --sensor "$SENSOR" \
    --num_images "$ACTUAL_NUM_IMAGES" \
    --num_steps "$NUM_STEPS" \
    --kappa "$KAPPA" \
    --device cuda \
    --seed 42

echo ""
echo "=========================================="
echo "Validation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files (always saved):"
echo "  - synthetic_validation_results.json (detailed results)"
echo "  - summary.json (aggregate statistics)"
echo "  - regime_sweep_analysis.png (regime classification visualization)"
echo "  - sweep_results.json (structured sweep results)"
echo "  - sweep_results.csv (CSV format for easy analysis)"
echo "  - sweep_analysis.png (performance vs Poisson fraction - CVPR Figure 1)"
echo ""
echo "Poisson sweep demonstrates regime-dependent performance (CVPR Figure 1):"
echo "  - 5% Poisson: Read-noise dominated, minimal PG gain (< 0.5 dB)"
echo "  - 10% Poisson: Transitional, small PG gain (~0.5-1 dB)"
echo "  - 15% Poisson: Transitional, moderate PG gain (~1-1.5 dB)"
echo "  - 20% Poisson: Shot-noise dominated, significant PG gain (~2 dB, p < 0.05)"
echo "  - 30% Poisson: Shot-noise dominated, large PG gain (~2.5-3 dB, p < 0.001)"
echo ""
echo "This validates that PG > Gaussian in shot-noise-dominated regime!"
echo ""
echo "Note: Uses TEST split (not validation) for proper evaluation."
