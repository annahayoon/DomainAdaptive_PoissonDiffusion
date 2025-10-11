#!/bin/bash
# Compare different guidance settings side-by-side

set -e

echo "=========================================="
echo "EDM Guidance Settings Comparison"
echo "=========================================="

# Configuration
MODEL_PATH="/home/jilab/Jae/results/edm_pt_training_20251008_032055/best_model.pkl"
DATA_ROOT="/home/jilab/Jae/dataset/processed/pt_tiles"
METADATA_JSON="/home/jilab/Jae/dataset/processed/metadata_photography_incremental.json"
BASE_OUTPUT="results/guidance_comparison_$(date +%Y%m%d_%H%M%S)"

# Number of test samples (use small number for quick comparison)
NUM_SAMPLES=10
BATCH_SIZE=4
NUM_STEPS=18

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Samples: $NUM_SAMPLES"
echo "  Steps: $NUM_STEPS"
echo "  Output: $BASE_OUTPUT"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Create base output directory
mkdir -p "$BASE_OUTPUT"

# Configuration 1: Vanilla EDM (no guidance)
echo "=========================================="
echo "1/6: Vanilla EDM (baseline)"
echo "=========================================="
python sample_pt_edm_native_photography.py \
    --data_root "$DATA_ROOT" \
    --metadata_json "$METADATA_JSON" \
    --model_path "$MODEL_PATH" \
    --output_dir "${BASE_OUTPUT}/1_vanilla_edm" \
    --max_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --no_guidance \
    --save_comparisons \
    --device cuda

echo "✅ Done"
echo ""

# Configuration 2: Weak guidance
echo "=========================================="
echo "2/6: Weak Guidance (kappa=0.3)"
echo "=========================================="
python sample_pt_edm_native_photography.py \
    --data_root "$DATA_ROOT" \
    --metadata_json "$METADATA_JSON" \
    --model_path "$MODEL_PATH" \
    --output_dir "${BASE_OUTPUT}/2_weak_guidance_k03" \
    --max_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --guidance_strength 0.3 \
    --guidance_mode wls \
    --save_comparisons \
    --device cuda

echo "✅ Done"
echo ""

# Configuration 3: Moderate guidance (default)
echo "=========================================="
echo "3/6: Moderate Guidance (kappa=0.6)"
echo "=========================================="
python sample_pt_edm_native_photography.py \
    --data_root "$DATA_ROOT" \
    --metadata_json "$METADATA_JSON" \
    --model_path "$MODEL_PATH" \
    --output_dir "${BASE_OUTPUT}/3_moderate_guidance_k06" \
    --max_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --guidance_strength 0.6 \
    --guidance_mode wls \
    --save_comparisons \
    --device cuda

echo "✅ Done"
echo ""

# Configuration 4: Strong guidance
echo "=========================================="
echo "4/6: Strong Guidance (kappa=0.8)"
echo "=========================================="
python sample_pt_edm_native_photography.py \
    --data_root "$DATA_ROOT" \
    --metadata_json "$METADATA_JSON" \
    --model_path "$MODEL_PATH" \
    --output_dir "${BASE_OUTPUT}/4_strong_guidance_k08" \
    --max_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --guidance_strength 0.8 \
    --guidance_mode wls \
    --save_comparisons \
    --device cuda

echo "✅ Done"
echo ""

# Configuration 5: Moderate guidance (full mode)
echo "=========================================="
echo "5/6: Moderate Guidance Full Mode (kappa=0.6)"
echo "=========================================="
python sample_pt_edm_native_photography.py \
    --data_root "$DATA_ROOT" \
    --metadata_json "$METADATA_JSON" \
    --model_path "$MODEL_PATH" \
    --output_dir "${BASE_OUTPUT}/5_moderate_full_k06" \
    --max_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --num_steps $NUM_STEPS \
    --guidance_strength 0.6 \
    --guidance_mode full \
    --save_comparisons \
    --device cuda

echo "✅ Done"
echo ""

# Configuration 6: High quality (more steps)
echo "=========================================="
echo "6/6: High Quality (50 steps, kappa=0.6)"
echo "=========================================="
python sample_pt_edm_native_photography.py \
    --data_root "$DATA_ROOT" \
    --metadata_json "$METADATA_JSON" \
    --model_path "$MODEL_PATH" \
    --output_dir "${BASE_OUTPUT}/6_high_quality_50steps" \
    --max_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --num_steps 50 \
    --guidance_strength 0.6 \
    --guidance_mode wls \
    --save_comparisons \
    --device cuda

echo "✅ Done"
echo ""

# Create summary
echo "=========================================="
echo "Creating comparison summary..."
echo "=========================================="

SUMMARY_FILE="${BASE_OUTPUT}/COMPARISON_SUMMARY.txt"

cat > "$SUMMARY_FILE" << EOF
EDM Guidance Settings Comparison
Generated: $(date)

Configuration Details:
- Model: $MODEL_PATH
- Test samples: $NUM_SAMPLES
- Default steps: $NUM_STEPS
- Batch size: $BATCH_SIZE

Results Directories:
1. Vanilla EDM (no guidance)
   ${BASE_OUTPUT}/1_vanilla_edm/

2. Weak Guidance (kappa=0.3, WLS)
   ${BASE_OUTPUT}/2_weak_guidance_k03/

3. Moderate Guidance (kappa=0.6, WLS) [RECOMMENDED DEFAULT]
   ${BASE_OUTPUT}/3_moderate_guidance_k06/

4. Strong Guidance (kappa=0.8, WLS)
   ${BASE_OUTPUT}/4_strong_guidance_k08/

5. Moderate Guidance (kappa=0.6, Full mode)
   ${BASE_OUTPUT}/5_moderate_full_k06/

6. High Quality (50 steps, kappa=0.6, WLS)
   ${BASE_OUTPUT}/6_high_quality_50steps/

How to Review:
1. Open comparison images in each directory
2. Look for:
   - Noise reduction (less grain)
   - Detail preservation (sharp edges)
   - Color accuracy (natural tones)
   - Artifacts (over-smoothing, halos)

What to Expect:
- Vanilla EDM: Good baseline, may be slightly noisy
- Weak guidance: Similar to vanilla, subtle improvements
- Moderate guidance: Best balance (recommended)
- Strong guidance: Aggressive denoising, may over-smooth
- Full mode: More accurate for very dark images
- High quality: Best results but slower

Recommendations:
- For general use: Configuration 3 (kappa=0.6, WLS)
- For dark images: Configuration 4 or 5 (stronger guidance)
- For best quality: Configuration 6 (more steps)
- For comparison: Configuration 1 (vanilla EDM baseline)

Next Steps:
1. Visually inspect comparison images
2. Choose best configuration for your use case
3. Run full test set with chosen configuration
4. Compute metrics if ground truth available
EOF

cat "$SUMMARY_FILE"

echo ""
echo "=========================================="
echo "✅ All comparisons completed!"
echo "=========================================="
echo ""
echo "Results saved to: $BASE_OUTPUT"
echo "Summary: $SUMMARY_FILE"
echo ""
echo "To view results:"
echo "  cd $BASE_OUTPUT"
echo "  ls -lh */"
echo ""
echo "To create a collage (requires ImageMagick):"
echo "  # For a specific tile ID (e.g., tile_0000)"
echo "  montage \\
echo "    1_vanilla_edm/*_0000_comparison.png \\
echo "    2_weak_guidance_k03/*_0000_comparison.png \\
echo "    3_moderate_guidance_k06/*_0000_comparison.png \\
echo "    4_strong_guidance_k08/*_0000_comparison.png \\
echo "    5_moderate_full_k06/*_0000_comparison.png \\
echo "    6_high_quality_50steps/*_0000_comparison.png \\
echo "    -tile 2x3 -geometry +5+5 collage_tile_0000.png"
echo ""
