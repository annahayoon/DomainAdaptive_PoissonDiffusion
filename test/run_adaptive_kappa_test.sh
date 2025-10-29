#!/bin/bash

# Adaptive Kappa Test Script
# Tests adaptive kappa scheduling on photography tiles

echo "=================================================================================="
echo "ADAPTIVE KAPPA SCHEDULING TEST"
echo "=================================================================================="

# Set up directories
OUTPUT_DIR="results/adaptive_kappa_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"

# Test on a few low-signal tiles
echo "Testing adaptive kappa on low-signal photography tiles..."

# Run inference with current parameters (baseline)
echo "Running baseline inference (constant kappa=0.8)..."
python sample/sample_noisy_pt_lle_PGguidance.py \
    --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
    --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
    --noisy_dir dataset/processed/pt_tiles/photography/noisy \
    --clean_dir dataset/processed/pt_tiles/photography/clean \
    --output_dir "$OUTPUT_DIR/baseline" \
    --domain photography \
    --kappa 0.8 \
    --sigma_r 4.0 \
    --num_steps 15 \
    --run_methods gaussian_x0 pg_x0 \
    --use_sensor_calibration \
    --preserve_details \
    --edge_aware \
    --max_tiles 20 \
    --tile_filter "signal_level < 50" \
    --log_level INFO

echo "Baseline inference completed."

# Note: For adaptive kappa, you would need to modify the PG guidance class
# This is a placeholder for the adaptive version
echo ""
echo "To test adaptive kappa:"
echo "1. Apply the modification from adaptive_kappa_modification.py"
echo "2. Run the same command with modified PG guidance"
echo "3. Compare results between baseline and adaptive versions"

echo ""
echo "Expected improvements with adaptive kappa:"
echo "- Ultra-low signal (<5 e⁻): κ = 0.05 → Better stability"
echo "- Low signal (5-20 e⁻): κ = 0.1-0.4 → Balanced performance"
echo "- Medium signal (20-50 e⁻): κ = 0.4-0.8 → Gradual improvement"
echo "- Normal signal (≥50 e⁻): κ = 0.8 → Maintained performance"

echo ""
echo "Test completed. Check $OUTPUT_DIR for results."
