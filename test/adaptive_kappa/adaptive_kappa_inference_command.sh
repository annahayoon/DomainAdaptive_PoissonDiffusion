
# Command to test adaptive kappa sampling:

python sample/sample_noisy_pt_lle_PGguidance.py \
    --model_path results/edm_pt_training_photography_20251008_032055/best_model.pkl \
    --metadata_json dataset/processed/comprehensive_tiles_metadata.json \
    --noisy_dir dataset/processed/pt_tiles/photography/noisy \
    --clean_dir dataset/processed/pt_tiles/photography/clean \
    --output_dir results/adaptive_kappa_test \
    --domain photography \
    --kappa 0.8 \
    --sigma_r 4.0 \
    --num_steps 15 \
    --run_methods gaussian_x0 pg_x0 \
    --use_sensor_calibration \
    --preserve_details \
    --adaptive_strength \
    --edge_aware \
    --max_tiles 50

# Note: You'll need to modify the PG guidance class to use adaptive kappa
# See adaptive_kappa_modification.py for the implementation
