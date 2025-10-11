#!/bin/bash
# Test DDNM+ with different configurations

echo "Testing DDNM+ with different sigma_y values..."
echo "=============================================="

# Test 1: No DDNM (vanilla EDM)
echo -e "\n[1/4] Vanilla EDM (no DDNM)..."
python sample_pt_edmddnm.py --no_ddnm --save_comparisons --max_samples 2 --batch_size 1 --num_steps 18 --output_dir results/ddnm_test/no_ddnm

# Test 2: DDNM with sigma_y = 0.1
echo -e "\n[2/4] DDNM+ with σ_y = 0.1..."
python sample_pt_edmddnm.py --sigma_y 0.1 --save_comparisons --max_samples 2 --batch_size 1 --num_steps 18 --output_dir results/ddnm_test/sigma_0.1

# Test 3: DDNM with sigma_y = 0.5
echo -e "\n[3/4] DDNM+ with σ_y = 0.5..."
python sample_pt_edmddnm.py --sigma_y 0.5 --save_comparisons --max_samples 2 --batch_size 1 --num_steps 18 --output_dir results/ddnm_test/sigma_0.5

# Test 4: DDNM with sigma_y = 2.0 (very large)
echo -e "\n[4/4] DDNM+ with σ_y = 2.0 (large noise assumption)..."
python sample_pt_edmddnm.py --sigma_y 2.0 --save_comparisons --max_samples 2 --batch_size 1 --num_steps 18 --output_dir results/ddnm_test/sigma_2.0

echo -e "\n✅ Done! Check results/ddnm_test/ for comparisons"
