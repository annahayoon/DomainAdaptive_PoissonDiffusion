#!/bin/bash
# Launch optimized inference for ALL test tiles across all domains in tmux sessions
# Using parameters from single-domain optimization results (October 21, 2025)

# Base paths
OUTPUT_BASE="results/optimized_inference_all_tiles"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

echo "=========================================="
echo "Launching optimized inference on ALL test tiles"
echo "=========================================="

# 1. ASTRONOMY - κ=0.05, σ_r=9.0, steps=25
SESSION_NAME="inference_astronomy_all"
echo "Creating tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" "cd /home/jilab/Jae && \
python sample/sample_noisy_pt_lle_PGguidance.py \
    --model_path results/edm_pt_training_astronomy_20251009_172141/best_model.pkl \
    --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
    --noisy_dir dataset/processed/pt_tiles/astronomy/noisy \
    --clean_dir dataset/processed/pt_tiles/astronomy/clean \
    --output_dir ${OUTPUT_BASE}/astronomy_optimized \
    --domain astronomy \
    --num_examples 10000 \
    --kappa 0.05 \
    --sigma_r 9.0 \
    --num_steps 25 \
    --use_sensor_calibration \
    --compare_gaussian \
    --skip_visualization \
    --run_methods noisy clean exposure_scaled gaussian_x0 pg_x0 \
    2>&1 | tee ${OUTPUT_BASE}/astronomy_optimized.log; \
echo 'Press Enter to close'; read"

# 2. MICROSCOPY - κ=0.4, σ_r=0.5, steps=20
SESSION_NAME="inference_microscopy_all"
echo "Creating tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" "cd /home/jilab/Jae && \
python sample/sample_noisy_pt_lle_PGguidance.py \
    --model_path results/edm_pt_training_microscopy_20251008_044631/best_model.pkl \
    --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
    --noisy_dir dataset/processed/pt_tiles/microscopy/noisy \
    --clean_dir dataset/processed/pt_tiles/microscopy/clean \
    --output_dir ${OUTPUT_BASE}/microscopy_optimized \
    --domain microscopy \
    --num_examples 10000 \
    --kappa 0.4 \
    --sigma_r 0.5 \
    --num_steps 20 \
    --use_sensor_calibration \
    --compare_gaussian \
    --skip_visualization \
    --run_methods noisy clean exposure_scaled gaussian_x0 pg_x0 \
    2>&1 | tee ${OUTPUT_BASE}/microscopy_optimized.log; \
echo 'Press Enter to close'; read"

# 3. PHOTOGRAPHY (SONY) - Combined photography parameters: κ=0.8, σ_r=4.0, steps=15
SESSION_NAME="inference_sony_all"
echo "Creating tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" "cd /home/jilab/Jae && \
python sample/sample_noisy_pt_lle_PGguidance.py \
    --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
    --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
    --noisy_dir dataset/processed/pt_tiles/photography/noisy \
    --clean_dir dataset/processed/pt_tiles/photography/clean \
    --output_dir ${OUTPUT_BASE}/photography_sony_optimized \
    --domain photography \
    --sensor_filter sony \
    --num_examples 10000 \
    --kappa 0.8 \
    --sigma_r 4.0 \
    --num_steps 15 \
    --use_sensor_calibration \
    --compare_gaussian \
    --skip_visualization \
    --run_methods noisy clean exposure_scaled gaussian_x0 pg_x0 \
    2>&1 | tee ${OUTPUT_BASE}/photography_sony_optimized.log; \
echo 'Press Enter to close'; read"

# 4. PHOTOGRAPHY (FUJI) - Combined photography parameters: κ=0.8, σ_r=4.0, steps=15
SESSION_NAME="inference_fuji_all"
echo "Creating tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" "cd /home/jilab/Jae && \
python sample/sample_noisy_pt_lle_PGguidance.py \
    --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
    --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
    --noisy_dir dataset/processed/pt_tiles/photography/noisy \
    --clean_dir dataset/processed/pt_tiles/photography/clean \
    --output_dir ${OUTPUT_BASE}/photography_fuji_optimized \
    --domain photography \
    --sensor_filter fuji \
    --num_examples 10000 \
    --kappa 0.8 \
    --sigma_r 4.0 \
    --num_steps 15 \
    --use_sensor_calibration \
    --compare_gaussian \
    --skip_visualization \
    --run_methods noisy clean exposure_scaled gaussian_x0 pg_x0 \
    2>&1 | tee ${OUTPUT_BASE}/photography_fuji_optimized.log; \
echo 'Press Enter to close'; read"

echo ""
echo "=========================================="
echo "✅ All tmux sessions created!"
echo "=========================================="
echo ""
echo "Active sessions:"
tmux list-sessions | grep inference_
echo ""
echo "Note: Processing ALL available test tiles (no limit)"
echo "Note: Sony and Fuji both use COMBINED photography parameters (κ=0.8, σ_r=4.0, steps=15)"
echo ""
echo "To attach to a session:"
echo "  tmux attach -t inference_astronomy_all"
echo "  tmux attach -t inference_microscopy_all"
echo "  tmux attach -t inference_sony_all"
echo "  tmux attach -t inference_fuji_all"
echo ""
echo "Logs saved to:"
echo "  ${OUTPUT_BASE}/astronomy_optimized.log"
echo "  ${OUTPUT_BASE}/microscopy_optimized.log"
echo "  ${OUTPUT_BASE}/photography_sony_optimized.log"
echo "  ${OUTPUT_BASE}/photography_fuji_optimized.log"
echo ""
echo "Metrics JSON will be saved to:"
echo "  ${OUTPUT_BASE}/astronomy_optimized/results.json"
echo "  ${OUTPUT_BASE}/microscopy_optimized/results.json"
echo "  ${OUTPUT_BASE}/photography_sony_optimized/results.json"
echo "  ${OUTPUT_BASE}/photography_fuji_optimized/results.json"
