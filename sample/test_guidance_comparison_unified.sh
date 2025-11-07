#!/bin/bash
# Unified script to test and visualize guidance comparison
# Supports both SIDD dataset and processed tiles (Sony/Fuji)
# Compares homoscedastic (Gaussian) vs heteroscedastic (Poisson-Gaussian) guidance
#
# Usage:
#   ./sample/test_guidance_comparison_unified.sh [DATASET] [SENSOR] [NUM_EXAMPLES] [--unique-only]
#
# Arguments:
#   DATASET: 'sidd' or 'tiles' (default: 'tiles')
#   SENSOR: 'sony' or 'fuji' (required for 'tiles', ignored for 'sidd')
#   NUM_EXAMPLES: Number of scenes/tiles to process (default: 5) or 'all' for entire test set
#   --unique-only: (optional, tiles only) Deduplicate by scene_id + exposure_time
#
# Examples:
#   # SIDD dataset
#   ./sample/test_guidance_comparison_unified.sh sidd 5
#   ./sample/test_guidance_comparison_unified.sh sidd 10
#
#   # Tiles dataset
#   ./sample/test_guidance_comparison_unified.sh tiles sony 5
#   ./sample/test_guidance_comparison_unified.sh tiles fuji all
#   ./sample/test_guidance_comparison_unified.sh tiles sony 10 --unique-only
#
# Runtime estimates (full test set):
#   SIDD: ~30-60 minutes for 20 scenes
#   Sony tiles: ~7-11 minutes (598 unique SHORT files, ~50 unique scene_ids, 216 tiles/scene)
#   Fuji tiles: ~5-8 minutes (524 unique SHORT files, ~41 unique scene_ids, 96 tiles/scene)
#
# Output:
#   SIDD: Per-scene: scene_XXX_*/restored_*.pt, scene_comparison.png, scene_metrics.json
#   Tiles: Per-scene: scene_XXX_*/stitched_*.pt, scene_comparison.png, scene_metrics.json

set -e

# Parse arguments
DATASET="${1:-tiles}"  # Default to tiles
SENSOR="${2:-sony}"    # Default sensor (only used for tiles)
NUM_EXAMPLES="${3:-5}" # Default number of examples

# Parse --unique-only flag (only for tiles)
UNIQUE_ONLY=false
for arg in "$@"; do
    if [ "$arg" = "--unique-only" ]; then
        UNIQUE_ONLY=true
        break
    fi
done

# Validate dataset
if [ "$DATASET" != "sidd" ] && [ "$DATASET" != "tiles" ]; then
    echo "ERROR: Invalid DATASET. Use 'sidd' or 'tiles'"
    exit 1
fi

# Process SIDD dataset
if [ "$DATASET" = "sidd" ]; then
    echo "=========================================="
    echo "SIDD Guidance Comparison Test"
    echo "=========================================="

    NUM_EXAMPLES="${NUM_EXAMPLES:-5}"
    SIDD_DATA_ROOT="/home/jilab/Jae/external/dataset/sidd/SIDD_Small_Raw_Only/Data"
    CALIBRATION_FILE="${SIDD_DATA_ROOT}/Data_noise_calibration.json"
    OUTPUT_DIR="results/guidance_comparison_sidd"

    # Find latest Fuji model for testing
    MODEL_PATTERN="results/edm_fuji_training_*/best_model.pkl"
    MODEL_PATH=$(ls -td ${MODEL_PATTERN%/best_model.pkl} 2>/dev/null | head -1)/best_model.pkl

    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: Fuji model not found: $MODEL_PATH"
        echo "Please train a Fuji model first"
        exit 1
    fi

    if [ ! -f "$CALIBRATION_FILE" ]; then
        echo "ERROR: Calibration file not found: $CALIBRATION_FILE"
        echo "Please run sensor_noise_calibrations.py first to generate calibration"
        exit 1
    fi

    # Optimal hyperparameters from tuning for Fuji: kappa=0.05, num_steps=18
    KAPPA=0.05
    NUM_STEPS=18

    echo "Data root: $SIDD_DATA_ROOT"
    echo "Model: $MODEL_PATH"
    echo "Calibration: $CALIBRATION_FILE"
    echo "Num examples: $NUM_EXAMPLES"
    echo "Optimal hyperparameters:"
    echo "  kappa: $KAPPA"
    echo "  num_steps: $NUM_STEPS"
    echo "Output: $OUTPUT_DIR"
    echo ""

    # Get list of scene directories
    SCENES=$(find "$SIDD_DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | head -n 160)

    if [ -z "$SCENES" ]; then
        echo "ERROR: No scene directories found in $SIDD_DATA_ROOT"
        exit 1
    fi

    # Count available scenes
    NUM_AVAILABLE=$(echo "$SCENES" | wc -l)
    echo "Found $NUM_AVAILABLE scene directories"

    # Select random scenes (or all if NUM_EXAMPLES >= NUM_AVAILABLE)
    if [ "$NUM_EXAMPLES" = "all" ] || [ "$NUM_EXAMPLES" -ge "$NUM_AVAILABLE" ]; then
        SELECTED_SCENES="$SCENES"
        NUM_EXAMPLES=$NUM_AVAILABLE
    else
        SELECTED_SCENES=$(echo "$SCENES" | shuf -n "$NUM_EXAMPLES")
    fi

    echo "Processing $NUM_EXAMPLES scenes"
    echo ""

    # Export variables for Python script
    export MODEL_PATH_SCRIPT="$MODEL_PATH"
    export SIDD_DATA_ROOT_SCRIPT="$SIDD_DATA_ROOT"
    export CALIBRATION_FILE_SCRIPT="$CALIBRATION_FILE"
    export OUTPUT_DIR_SCRIPT="$OUTPUT_DIR"
    export KAPPA_SCRIPT="$KAPPA"
    export NUM_STEPS_SCRIPT="$NUM_STEPS"
    export NUM_EXAMPLES="$NUM_EXAMPLES"

    # Process scenes using Python script
    python3 << PROCESS_SCENES_EOF
import sys
import json
import subprocess
import os
from pathlib import Path

try:
    model_path = os.environ["MODEL_PATH_SCRIPT"]
    sidd_data_root = Path(os.environ["SIDD_DATA_ROOT_SCRIPT"])
    calibration_file = os.environ["CALIBRATION_FILE_SCRIPT"]
    output_dir = Path(os.environ["OUTPUT_DIR_SCRIPT"])
    kappa = os.environ["KAPPA_SCRIPT"]
    num_steps = os.environ["NUM_STEPS_SCRIPT"]

    # Get list of scene directories
    scene_dirs = sorted([d for d in sidd_data_root.iterdir() if d.is_dir()])

    # Select random scenes
    import random
    random.seed(42)
    num_examples_str = os.environ.get("NUM_EXAMPLES", "5")
    try:
        num_examples = int(num_examples_str)
    except ValueError:
        num_examples = 5

    if num_examples >= len(scene_dirs):
        selected_scenes = scene_dirs
    else:
        selected_scenes = random.sample(scene_dirs, num_examples)

    print(f"Processing {len(selected_scenes)} scenes")
    print("")

    for idx, scene_dir in enumerate(selected_scenes):
        scene_name = scene_dir.name
        print("=" * 50)
        print(f"Processing scene {idx + 1}/{len(selected_scenes)}: {scene_name}")
        print("=" * 50)

        # Build command
        # Use max_size=1024 to avoid OOM (SIDD images are ~5328×3000)
        cmd = [
            "python", "sample/sample_sidd_PGguidance.py",
            "--model_path", model_path,
            "--scene_dir", str(scene_dir),
            "--output_dir", str(output_dir),
            "--calibration_file", calibration_file,
            "--run_methods", "noisy", "gt", "exposure_scaled", "gaussian_x0", "pg_x0",
            "--guidance_level", "x0",
            "--pg_mode", "wls",
            "--kappa", kappa,
            "--num_steps", num_steps,
            "--device", "cuda",
            "--seed", "42",
            "--no_heun",
            "--max_size", "1024"  # Resize large SIDD images to avoid OOM
        ]

        # Run command with output capture
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=False,  # Show output in real-time
                text=True
            )

            if result.returncode != 0:
                print(f"ERROR: Scene {scene_name} failed with return code {result.returncode}. Continuing...")
                continue
        except Exception as e:
            print(f"ERROR: Exception running scene {scene_name}: {e}", file=sys.stderr)
            continue

    print("")
    print("=" * 50)
    print("Processing complete!")
    print("=" * 50)
    print(f"Check results in: {output_dir}")

except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

PROCESS_SCENES_EOF

    if [ $? -ne 0 ]; then
        echo "ERROR: Processing failed."
        exit 1
    fi

    echo ""
    echo "=========================================="
    echo "Processing complete!"
    echo "=========================================="
    echo "Check results in: $OUTPUT_DIR"
    echo ""
    echo "Scene-level outputs (in scene_XXX_* directories):"
    echo "  - restored_*.pt files: Restored images for each method"
    echo "  - scene_comparison.png: Visual comparison of all methods"
    echo "  - scene_metrics.json: Metrics per scene"
    echo ""
    echo "Methods compared:"
    echo "  - noisy: Original noisy image"
    echo "  - gt: Ground truth (reference)"
    echo "  - exposure_scaled: Simple exposure scaling baseline"
    echo "  - gaussian_x0: Homoscedastic guidance (constant noise assumption)"
    echo "  - pg_x0: Heteroscedastic guidance (Poisson-Gaussian, signal-dependent noise)"
    echo ""

    exit 0
fi

# Process Tiles dataset
if [ "$DATASET" = "tiles" ]; then
    echo "=========================================="
    echo "Guidance Comparison Test (Tiles)"
    echo "=========================================="

    # Validate sensor
    if [ "$SENSOR" != "sony" ] && [ "$SENSOR" != "fuji" ]; then
        echo "ERROR: Invalid sensor. Use 'sony' or 'fuji'"
        exit 1
    fi

    DATA_ROOT="dataset/processed"
    CALIBRATION_DIR="${DATA_ROOT}"

    # Find latest model for sensor and set optimal hyperparameters
    # REQUIRE comprehensive metadata for accurate stitching (has image_x/image_y positions)
    if [ "$SENSOR" = "sony" ]; then
        MODEL_PATTERN="results/edm_sony_training_*/best_model.pkl"
        # REQUIRED: Comprehensive metadata for accurate tile stitching
        if [ -f "${DATA_ROOT}/comprehensive_sony_tiles_metadata.json" ]; then
            METADATA_JSON="${DATA_ROOT}/comprehensive_sony_tiles_metadata.json"
        elif [ -f "${DATA_ROOT}/comprehensive_all_tiles_metadata.json" ]; then
            METADATA_JSON="${DATA_ROOT}/comprehensive_all_tiles_metadata.json"
        else
            echo "ERROR: Comprehensive metadata required for accurate stitching!"
            echo "  Required: ${DATA_ROOT}/comprehensive_sony_tiles_metadata.json"
            echo "  Or: ${DATA_ROOT}/comprehensive_all_tiles_metadata.json"
            echo ""
            echo "  Comprehensive metadata includes image_x/image_y positions needed for"
            echo "  accurate tile stitching. Please run preprocessing pipeline to generate it."
            exit 1
        fi
        # Optimal hyperparameters from tuning: kappa=0.15, num_steps=10
        KAPPA=0.15
        NUM_STEPS=10
        # Batch size matches tiles per scene: 12 rows × 18 columns = 216 tiles
        BATCH_SIZE=216
    elif [ "$SENSOR" = "fuji" ]; then
        MODEL_PATTERN="results/edm_fuji_training_*/best_model.pkl"
        # REQUIRED: Comprehensive metadata for accurate stitching
        if [ -f "${DATA_ROOT}/comprehensive_fuji_tiles_metadata.json" ]; then
            METADATA_JSON="${DATA_ROOT}/comprehensive_fuji_tiles_metadata.json"
        elif [ -f "${DATA_ROOT}/comprehensive_all_tiles_metadata.json" ]; then
            METADATA_JSON="${DATA_ROOT}/comprehensive_all_tiles_metadata.json"
        else
            echo "ERROR: Comprehensive metadata required for accurate stitching!"
            echo "  Required: ${DATA_ROOT}/comprehensive_fuji_tiles_metadata.json"
            echo "  Or: ${DATA_ROOT}/comprehensive_all_tiles_metadata.json"
            echo ""
            echo "  Comprehensive metadata includes image_x/image_y positions needed for"
            echo "  accurate tile stitching. Please run preprocessing pipeline to generate it."
            exit 1
        fi
        # Optimal hyperparameters from tuning: kappa=0.05, num_steps=18
        KAPPA=0.05
        NUM_STEPS=18
        # Batch size matches tiles per scene: 8 rows × 12 columns = 96 tiles
        BATCH_SIZE=96
    fi

    MODEL_PATH=$(ls -td ${MODEL_PATTERN%/best_model.pkl} 2>/dev/null | head -1)/best_model.pkl

    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: Model not found: $MODEL_PATH"
        exit 1
    fi

    SHORT_DIR="${DATA_ROOT}/pt_tiles/${SENSOR}/short"
    LONG_DIR="${DATA_ROOT}/pt_tiles/${SENSOR}/long"
    OUTPUT_DIR="results/guidance_comparison_${SENSOR}"

    echo "Sensor: $SENSOR"
    echo "Model: $MODEL_PATH"
    echo "Metadata: $METADATA_JSON"
    echo "Num examples: $NUM_EXAMPLES"
    if [ "$UNIQUE_ONLY" = "true" ]; then
        echo "Mode: Unique scenes only (deduplicated by scene_id + exposure_time)"
    else
        echo "Mode: All unique SHORT files (may include multiple frames per scene_id + exposure_time)"
    fi
    echo "Optimal hyperparameters (from tuning):"
    echo "  kappa: $KAPPA"
    echo "  num_steps: $NUM_STEPS"
    echo "Output: $OUTPUT_DIR"
    echo ""
    echo "Processing mode:"
    echo "  - Groups tiles by scene_id + exposure_time"
    echo "  - Processes all tiles from each scene in batch"
    if [ "$SENSOR" = "sony" ]; then
        echo "  - Batch size: $BATCH_SIZE (matches tiles per scene: Sony = 12×18 = 216 tiles)"
    elif [ "$SENSOR" = "fuji" ]; then
        echo "  - Batch size: $BATCH_SIZE (matches tiles per scene: Fuji = 8×12 = 96 tiles)"
    fi
    echo "  - Stitches tiles back into full scene images"
    echo "  - Saves: stitched .pt files, scene_comparison.png, scene_metrics.json"
    echo ""

    # Get test scenes grouped by scene_id + exposure_time from split file
    echo "Loading test scenes from split file..."
    SPLITS_DIR="${DATA_ROOT}/../splits"
    if [ ! -d "$SPLITS_DIR" ]; then
        SPLITS_DIR="dataset/splits"
    fi

    SENSOR_CAPITALIZED=$(echo "${SENSOR}" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
    SPLIT_FILE="${SPLITS_DIR}/${SENSOR_CAPITALIZED}_test_list.txt"

    if [ ! -f "$SPLIT_FILE" ]; then
        echo "ERROR: Test split file not found: $SPLIT_FILE"
        exit 1
    fi

    # Group test files by scene_id + exposure_time and get tile IDs for each scene
    # Note: stderr is preserved separately so errors are visible, stdout contains only JSON
    UNIQUE_ONLY_FLAG="${UNIQUE_ONLY}"
    SCENE_GROUPS=$(python3 << SCENE_GROUPS_EOF
import json
import sys
import random
import os
from pathlib import Path
from collections import defaultdict

try:
    # Set random seed for reproducibility
    random.seed(42)

    # Check if unique-only mode is enabled (passed from bash variable substitution)
    unique_only_str = "${UNIQUE_ONLY_FLAG}"
    unique_only = unique_only_str.lower() == "true"

    metadata_path = Path("${METADATA_JSON}")
    if not metadata_path.exists():
        print(f"ERROR: Metadata file does not exist: {metadata_path}", file=sys.stderr)
        print("[]", file=sys.stdout)
        sys.exit(1)

    if metadata_path.stat().st_size == 0:
        print(f"ERROR: Metadata file is empty: {metadata_path}", file=sys.stderr)
        print("[]", file=sys.stdout)
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if not isinstance(metadata, dict) or "files" not in metadata:
        print(f"ERROR: Metadata file has invalid structure. Expected dict with 'files' key.", file=sys.stderr)
        print("[]", file=sys.stdout)
        sys.exit(1)

    # Load test split file and group by unique SHORT files
    # IMPORTANT: Each SHORT file is a unique scene (scene_id + frame_id + exposure_time)
    # Format: ./Sensor/short/10003_00_0.04s.ARW ./Sensor/long/10003_00_10s.ARW ISO F
    # Each short filename {scene_id}_{frame_id}_{exposure_time}s.* is a unique scene
    # We group by the full short filename to preserve frame_id differences
    split_file = Path("${SPLIT_FILE}")
    if not split_file.exists():
        print(f"ERROR: Split file does not exist: {split_file}", file=sys.stderr)
        print("[]", file=sys.stdout)
        sys.exit(1)

    short_file_groups = {}  # short_filename -> extract scene_id and exposure_time for display

    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # Parse format: ./Sensor/short/10003_00_0.04s.ARW ./Sensor/long/10003_00_10s.ARW ISO F
            # Or: ./Fuji/short/10019_00_0.033s.RAF ./Fuji/long/10019_00_10s.RAF ISO F
            # FIRST column is SHORT file, SECOND column is LONG file
            short_path = parts[0]  # SHORT file (first column)
            short_filename = Path(short_path).name

            # Extract scene_id, frame_id, and exposure_time from SHORT filename
            # Format: {scene_id}_{frame_id}_{exposure_time}s.{ARW|RAF}
            filename_no_ext = short_filename.replace('.ARW', '').replace('.RAF', '').replace('.arw', '').replace('.raf', '')
            name_parts = filename_no_ext.split('_')
            if len(name_parts) >= 3:
                scene_id = name_parts[0]  # Scene ID
                frame_id = name_parts[1]   # Frame ID (e.g., "00", "01", "02")
                exposure_str = name_parts[2]  # e.g., "0.04s" or "0.033s"
                exposure_time = float(exposure_str.replace('s', ''))
                # Each short filename is a unique scene (scene_id + frame_id + exposure_time)
                short_file_groups[short_filename] = {
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "exposure_time": exposure_time,
                    "short_filename": short_filename
                }

    # Match each SHORT file in metadata and collect tile IDs
    # IMPORTANT: Each short filename is processed as a separate scene
    files = metadata.get("files", [])
    scene_tile_groups = {}  # short_filename -> list of tile_ids

    for short_filename, file_info in short_file_groups.items():
        scene_id = file_info["scene_id"]
        exposure_time = file_info["exposure_time"]

        # Find metadata entry that matches this SHORT filename
        matching_file_path = None
        for file_data in files:
            file_meta = file_data.get("file_metadata", {})
            file_path = file_meta.get("file_path", "")
            if file_path:
                file_name = Path(file_path).name
                # Match by exact SHORT filename
                if file_name == short_filename:
                    matching_file_path = file_path
                    break

        if not matching_file_path:
            continue

        # Collect tile IDs from this specific SHORT file (data_type == "short" only)
        tile_ids = []
        for file_data in files:
            file_meta = file_data.get("file_metadata", {})
            file_path = file_meta.get("file_path", "")

            if file_path != matching_file_path:
                continue

            tiles = file_data.get("tiles", [])
            for tile in tiles:
                tile_id = tile.get("tile_id")
                tile_data_type = tile.get("data_type", "")

                # Only collect tiles from SHORT exposure files
                if tile_id and tile_data_type == "short":
                    tile_ids.append(tile_id)

        if tile_ids:
            # Use short_filename as the key (unique per scene_id + frame_id + exposure_time)
            scene_tile_groups[short_filename] = {
                "scene_id": scene_id,
                "exposure_time": exposure_time,
                "tile_ids": tile_ids
            }

    # Process: "all" or sample scenes
    num_examples_str = "${NUM_EXAMPLES}"
    scene_list = list(scene_tile_groups.items())  # List of (short_filename, scene_info) tuples

    # If unique_only mode, deduplicate by scene_id + exposure_time
    if unique_only:
        # Group by (scene_id, exposure_time) and keep only one scene per combination
        seen_combinations = {}
        deduplicated_scenes = []
        for short_filename, scene_info in scene_list:
            key = (scene_info["scene_id"], scene_info["exposure_time"])
            if key not in seen_combinations:
                seen_combinations[key] = True
                deduplicated_scenes.append((short_filename, scene_info))
        scene_list = deduplicated_scenes
        original_count = len(scene_tile_groups)
        unique_count = len(scene_list)
        print(f"Deduplication: {original_count} scenes -> {unique_count} unique scenes (by scene_id + exposure_time)", file=sys.stderr)

    if num_examples_str.lower() == "all":
        selected_scenes = scene_list
        unique_scene_ids = len(set(info["scene_id"] for _, info in scene_list))
        print(f"Processing all {len(selected_scenes)} unique SHORT files ({unique_scene_ids} unique scene_ids)", file=sys.stderr)
    else:
        try:
            num_scenes = int(num_examples_str)
            if len(scene_list) < num_scenes:
                print(f"WARNING: Only found {len(scene_list)} unique SHORT files, using all of them", file=sys.stderr)
                selected_scenes = scene_list
            else:
                selected_scenes = random.sample(scene_list, num_scenes)
        except ValueError:
            print(f"ERROR: Invalid NUM_EXAMPLES value: {num_examples_str}. Use a number or 'all'", file=sys.stderr)
            sys.exit(1)

    # Output: JSON format for bash to parse (ONLY JSON to stdout, all errors to stderr)
    # Format: [{"scene_id": "10003", "exposure_time": 0.04, "tile_ids": [...]}, ...]
    import json
    output = []
    for scene_data in selected_scenes:
        # scene_data is either a tuple (short_filename, dict) or just dict
        if isinstance(scene_data, tuple):
            short_filename, scene_info = scene_data
        else:
            scene_info = scene_data
            short_filename = scene_info.get("short_filename", "")

        output.append({
            "scene_id": scene_info["scene_id"],
            "exposure_time": scene_info["exposure_time"],
            "tile_ids": scene_info["tile_ids"]
        })
    # Only output JSON to stdout - everything else goes to stderr
    json.dump(output, sys.stdout)
    sys.stdout.flush()

except Exception as e:
    # All errors must go to stderr, not stdout
    print(f"Error loading test scenes: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    # Output empty JSON array to stdout so bash doesn't fail on JSON parse
    import json
    print("[]", file=sys.stdout)
    sys.exit(1)
SCENE_GROUPS_EOF
)

    # Validate that we got valid JSON
    if [ -z "$SCENE_GROUPS" ]; then
        echo "ERROR: Python script returned empty output. Check for errors above."
        exit 1
    fi

    # Test if it's valid JSON
    if ! echo "$SCENE_GROUPS" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
        echo "ERROR: Python script did not output valid JSON."
        echo "This usually means the Python script encountered an error."
        echo "Output received (first 20 lines):"
        echo "$SCENE_GROUPS" | head -20
        echo ""
        echo "Please check:"
        echo "  1. That the metadata file exists and is valid JSON: ${METADATA_JSON}"
        echo "  2. That the split file exists: ${SPLIT_FILE}"
        echo "  3. That the metadata file contains entries matching the split file"
        exit 1
    fi

    # Count scenes and tiles (with error handling)
    NUM_UNIQUE_SHORT_FILES=$(echo "$SCENE_GROUPS" | python3 -c "import sys, json; scenes=json.load(sys.stdin); print(len(scenes))" 2>/dev/null)
    if [ -z "$NUM_UNIQUE_SHORT_FILES" ]; then
        echo "ERROR: Failed to parse scene groups as JSON"
        echo "SCENE_GROUPS content:"
        echo "$SCENE_GROUPS" | head -20
        exit 1
    fi

    NUM_UNIQUE_SCENE_IDS=$(echo "$SCENE_GROUPS" | python3 -c "import sys, json; scenes=json.load(sys.stdin); print(len(set(s['scene_id'] for s in scenes)))" 2>/dev/null)
    TOTAL_TILES=$(echo "$SCENE_GROUPS" | python3 -c "import sys, json; scenes=json.load(sys.stdin); print(sum(len(s['tile_ids']) for s in scenes))" 2>/dev/null)

    if [ "$NUM_UNIQUE_SHORT_FILES" = "0" ]; then
        echo "ERROR: No test scenes found for sensor ${SENSOR}"
        echo "This could mean:"
        echo "  - The metadata file doesn't contain matching entries"
        echo "  - The split file doesn't match the metadata"
        echo "  - No tiles were found for the test scenes"
        exit 1
    fi

    if [ "$NUM_EXAMPLES" = "all" ]; then
        echo "Processing ${NUM_UNIQUE_SHORT_FILES} unique SHORT files (${NUM_UNIQUE_SCENE_IDS} unique scene_ids, ${TOTAL_TILES} total tiles)"
        echo "Each SHORT file will be processed in batch (batch_size=$BATCH_SIZE = tiles per scene)"
        echo ""
    else
        echo "Processing ${NUM_UNIQUE_SHORT_FILES} randomly sampled SHORT files (${TOTAL_TILES} total tiles)"
        echo ""
    fi

    # Check calibration (calibration is enabled by default)
    CALIBRATION_FILE="${CALIBRATION_DIR}/${SENSOR}_noise_calibration.json"
    if [ ! -f "$CALIBRATION_FILE" ]; then
        echo "ERROR: Calibration file not found: $CALIBRATION_FILE"
        echo "Calibration is enabled by default. Please run sensor_noise_calibrations.py first"
        echo "Or use --no-noise-calibration to disable (not recommended)"
        exit 1
    else
        echo "Using calibration: $CALIBRATION_FILE"
        USE_CALIBRATION_FLAG="--calibration_dir ${CALIBRATION_DIR}"
    fi

    echo ""
    echo "Running with all methods for comparison..."
    echo ""

    # Process each scene group separately
    echo "=========================================="
    echo "Processing ${NUM_UNIQUE_SHORT_FILES} scenes"
    echo "Each scene processed in batch (batch_size=$BATCH_SIZE = tiles per scene)"
    echo "=========================================="
    echo ""

    # Process each scene in a Python loop to avoid bash parsing complexity
    # Pass bash variables via environment or as arguments
    export MODEL_PATH_SCRIPT="$MODEL_PATH"
    export METADATA_JSON_SCRIPT="$METADATA_JSON"
    export SHORT_DIR_SCRIPT="$SHORT_DIR"
    export LONG_DIR_SCRIPT="$LONG_DIR"
    export OUTPUT_DIR_SCRIPT="$OUTPUT_DIR"
    export SENSOR_SCRIPT="$SENSOR"
    export CALIBRATION_DIR_SCRIPT="$CALIBRATION_DIR"
    export KAPPA_SCRIPT="$KAPPA"
    export NUM_STEPS_SCRIPT="$NUM_STEPS"
    export BATCH_SIZE_SCRIPT="$BATCH_SIZE"

    # Verify SCENE_GROUPS is not empty before processing
    if [ -z "$SCENE_GROUPS" ]; then
        echo "ERROR: SCENE_GROUPS is empty! This should not happen after validation."
        echo "This might indicate a shell variable issue."
        exit 1
    fi

    # Use a temporary file to pass large JSON data (avoids pipe/buffer issues with large data)
    SCENE_GROUPS_TMPFILE=$(mktemp)
    echo "$SCENE_GROUPS" > "$SCENE_GROUPS_TMPFILE"

    # Verify the file was written correctly
    if [ ! -s "$SCENE_GROUPS_TMPFILE" ]; then
        echo "ERROR: Failed to write SCENE_GROUPS to temporary file"
        rm -f "$SCENE_GROUPS_TMPFILE"
        exit 1
    fi

    # Clean up temp file on exit
    trap "rm -f $SCENE_GROUPS_TMPFILE" EXIT

    export SCENE_GROUPS_TMPFILE_SCRIPT="$SCENE_GROUPS_TMPFILE"

    python3 << PROCESS_SCENES_EOF
import sys
import json
import subprocess
import os
from pathlib import Path

try:
    # Load from temporary file instead of stdin (more reliable for large data)
    tmpfile = Path(os.environ["SCENE_GROUPS_TMPFILE_SCRIPT"])
    if not tmpfile.exists():
        print(f"ERROR: Temporary file does not exist: {tmpfile}", file=sys.stderr)
        sys.exit(1)

    if tmpfile.stat().st_size == 0:
        print(f"ERROR: Temporary file is empty: {tmpfile}", file=sys.stderr)
        sys.exit(1)

    with open(tmpfile, 'r') as f:
        scenes = json.load(f)

    num_scenes = len(scenes)

    if num_scenes == 0:
        print("ERROR: No scenes found in SCENE_GROUPS", file=sys.stderr)
        sys.exit(1)

except json.JSONDecodeError as e:
    print(f"ERROR: Failed to parse JSON from temporary file: {e}", file=sys.stderr)
    tmpfile = Path(os.environ.get("SCENE_GROUPS_TMPFILE_SCRIPT", "unknown"))
    if tmpfile.exists():
        with open(tmpfile, 'r') as f:
            content = f.read()
        print(f"File size: {len(content)} characters", file=sys.stderr)
        print(f"First 500 chars: {content[:500]}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Unexpected error loading scenes: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

for scene_idx, scene in enumerate(scenes):
    scene_id = scene['scene_id']
    exposure_time = scene['exposure_time']
    tile_ids = scene['tile_ids']

    print("")
    print("=" * 42)
    print(f"Processing scene {scene_idx + 1}/{num_scenes}: {scene_id} (exposure {exposure_time}s)")
    print(f"  Tiles: {len(tile_ids)}")
    print("=" * 42)

    # Build command using environment variables
    cmd = [
        "python", "sample/sample_noisy_pt_lle_PGguidance.py",
        "--model_path", os.environ["MODEL_PATH_SCRIPT"],
        "--metadata_json", os.environ["METADATA_JSON_SCRIPT"],
        "--short_dir", os.environ["SHORT_DIR_SCRIPT"],
        "--long_dir", os.environ["LONG_DIR_SCRIPT"],
        "--output_dir", os.environ["OUTPUT_DIR_SCRIPT"],
        "--sensor_filter", os.environ["SENSOR_SCRIPT"],
        "--use_sensor_calibration",
        "--calibration_dir", os.environ["CALIBRATION_DIR_SCRIPT"],
        "--tile_ids"] + tile_ids + [
        "--run_methods", "short", "long", "exposure_scaled", "gaussian_x0", "pg_x0",
        "--guidance_level", "x0",
        "--pg_mode", "wls",
        "--kappa", os.environ["KAPPA_SCRIPT"],
        "--num_steps", os.environ["NUM_STEPS_SCRIPT"],
        "--device", "cuda",
        "--seed", "42",
        "--batch_size", os.environ["BATCH_SIZE_SCRIPT"],
        "--num_workers", "4",
        "--prefetch_factor", "2",
        "--no_heun"
    ]

    # Run command
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"ERROR: Scene {scene_id} (exposure {exposure_time}s) failed. Stopping.")
        sys.exit(1)

PROCESS_SCENES_EOF

    if [ $? -ne 0 ]; then
        echo "ERROR: Processing failed. Stopping."
        exit 1
    fi

    echo ""
    echo "=========================================="
    echo "Processing complete!"
    echo "=========================================="
    echo "Check results in: $OUTPUT_DIR"
    echo ""
    echo "Scene-level outputs (in scene_XXX_* directories):"
    echo "  - stitched_*.pt files: Full stitched scene images for each method"
    echo "  - scene_comparison.png: Visual comparison of all methods"
    echo "  - scene_metrics.json: Aggregate metrics per scene"
    echo ""
    echo "Note: Per-tile outputs are not saved - only scene-level outputs are generated."
    echo ""
    echo "Methods compared:"
    echo "  - short: Original noisy short exposure"
    echo "  - exposure_scaled: Simple exposure scaling baseline"
    echo "  - gaussian_x0: Homoscedastic guidance (constant noise assumption)"
    echo "  - pg_x0: Heteroscedastic guidance (Poisson-Gaussian, signal-dependent noise)"
    echo ""
    echo "Processing summary:"
    echo "  - Tiles grouped by scene_id + exposure_time"
    echo "  - All tiles from each scene processed in batch"
    echo "  - Tiles stitched back into full scene images"
    echo "  - Stitched images saved as .pt files and visualizations"
    echo ""
fi
