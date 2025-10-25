#!/bin/bash
# Launch cross-domain optimized inference for ALL test tiles across all domains in tmux sessions
# Using unified cross-domain optimized parameters: κ=0.2, σ_r=2.0, steps=15

# Base paths
OUTPUT_BASE="results/cross_domain_inference_all_tiles"
CROSS_DOMAIN_MODEL="results/edm_pt_training_multi_domain_20251010_182807/best_model.pkl"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

echo "=========================================="
echo "Launching cross-domain inference on ALL test tiles"
echo "Unified Parameters: κ=0.2, σ_r=2.0, steps=15"
echo "=========================================="

# 1. PHOTOGRAPHY (ALL - Sony + Fuji combined)
SESSION_NAME="inference_cross_photography"
echo "Creating tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" "cd /home/jilab/Jae && \
python sample/sample_noisy_pt_lle_PGguidance.py \
    --model_path ${CROSS_DOMAIN_MODEL} \
    --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
    --noisy_dir dataset/processed/pt_tiles/photography/noisy \
    --clean_dir dataset/processed/pt_tiles/photography/clean \
    --output_dir ${OUTPUT_BASE}/photography_cross_domain \
    --domain photography \
    --num_examples 10000 \
    --cross_domain_kappa 0.2 \
    --cross_domain_sigma_r 2.0 \
    --cross_domain_num_steps 15 \
    --use_sensor_calibration \
    --skip_visualization \
    --run_methods exposure_scaled gaussian_x0_cross pg_x0_cross \
    2>&1 | tee ${OUTPUT_BASE}/photography_cross_domain.log; \
echo 'Press Enter to close'; read"

# 2. MICROSCOPY
SESSION_NAME="inference_cross_microscopy"
echo "Creating tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" "cd /home/jilab/Jae && \
python sample/sample_noisy_pt_lle_PGguidance.py \
    --model_path ${CROSS_DOMAIN_MODEL} \
    --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
    --noisy_dir dataset/processed/pt_tiles/microscopy/noisy \
    --clean_dir dataset/processed/pt_tiles/microscopy/clean \
    --output_dir ${OUTPUT_BASE}/microscopy_cross_domain \
    --domain microscopy \
    --num_examples 10000 \
    --cross_domain_kappa 0.2 \
    --cross_domain_sigma_r 2.0 \
    --cross_domain_num_steps 15 \
    --use_sensor_calibration \
    --skip_visualization \
    --run_methods exposure_scaled gaussian_x0_cross pg_x0_cross \
    2>&1 | tee ${OUTPUT_BASE}/microscopy_cross_domain.log; \
echo 'Press Enter to close'; read"

# 3. ASTRONOMY
SESSION_NAME="inference_cross_astronomy"
echo "Creating tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" "cd /home/jilab/Jae && \
python sample/sample_noisy_pt_lle_PGguidance.py \
    --model_path ${CROSS_DOMAIN_MODEL} \
    --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
    --noisy_dir dataset/processed/pt_tiles/astronomy/noisy \
    --clean_dir dataset/processed/pt_tiles/astronomy/clean \
    --output_dir ${OUTPUT_BASE}/astronomy_cross_domain \
    --domain astronomy \
    --num_examples 10000 \
    --cross_domain_kappa 0.2 \
    --cross_domain_sigma_r 2.0 \
    --cross_domain_num_steps 15 \
    --use_sensor_calibration \
    --skip_visualization \
    --run_methods exposure_scaled gaussian_x0_cross pg_x0_cross \
    2>&1 | tee ${OUTPUT_BASE}/astronomy_cross_domain.log; \
echo 'Press Enter to close'; read"

echo ""
echo "=========================================="
echo "✅ All cross-domain tmux sessions created!"
echo "=========================================="
echo ""
echo "Active cross-domain sessions:"
tmux list-sessions | grep inference_cross_
echo ""
echo "Unified Parameters: κ=0.2, σ_r=2.0, steps=15"
echo "Processing ALL available test tiles (no limit)"
echo ""
echo "To attach to a session:"
echo "  tmux attach -t inference_cross_photography"
echo "  tmux attach -t inference_cross_microscopy"
echo "  tmux attach -t inference_cross_astronomy"
echo ""
echo "Logs saved to:"
echo "  ${OUTPUT_BASE}/photography_cross_domain.log"
echo "  ${OUTPUT_BASE}/microscopy_cross_domain.log"
echo "  ${OUTPUT_BASE}/astronomy_cross_domain.log"
echo ""
echo "Metrics JSON will be saved to:"
echo "  ${OUTPUT_BASE}/photography_cross_domain/results.json"
echo "  ${OUTPUT_BASE}/microscopy_cross_domain/results.json"
echo "  ${OUTPUT_BASE}/astronomy_cross_domain/results.json"

