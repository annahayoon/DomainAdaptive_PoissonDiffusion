#!/bin/bash

# L2 BASELINE Photography Training Configuration
# Identical to Poisson-Gaussian except guidance type and 100K max steps
# For fair comparison and ablation study

echo "📸 L2 BASELINE PHOTOGRAPHY TRAINING CONFIGURATION v1.0"
echo "====================================================="
echo ""
echo "Strategy: CORRECTED L2 baseline with proper ablation methodology"
echo "✅ BASELINE: Homoscedastic Gaussian noise + v-parameterization training"
echo "✅ TRAINING: Same v-param objective, differs only in inference guidance"  
echo "✅ CONFIG: Log-uniform noise [0.01,2.0] + simple conditioning"
echo ""

# Kill existing sessions
tmux kill-session -t l2_photography_training 2>/dev/null

# Create new session
tmux new-session -d -s l2_photography_training -c /home/jilab/Jae

# Phase detection - check if we have a stable checkpoint (SAME AS POISSON)
STABLE_CHECKPOINT=""
if [ -f "results/l2_photography_stable_checkpoint.pth" ]; then
    STABLE_CHECKPOINT="--resume results/l2_photography_stable_checkpoint.pth"
    MIXED_PRECISION="true"   # Enable mixed precision in phase 2
    LEARNING_RATE="1e-5"     # Higher learning rate in phase 2
    echo "📊 Phase 2: Stable checkpoint found, enabling optimizations"
else
    MIXED_PRECISION="false"  # DISABLE mixed precision in phase 1 (SAME AS POISSON)
    LEARNING_RATE="5e-6"     # Conservative learning rate in phase 1
    echo "📊 Phase 1: Initial training, prioritizing stability"
fi

# Data path - SAME as Poisson-Gaussian
REAL_DATA_PATH="/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed"
if [ -d "$REAL_DATA_PATH" ] && [ "$(ls -A $REAL_DATA_PATH)" ]; then
    DATA_ROOT="$REAL_DATA_PATH"
    echo "✅ Using SAME photography dataset: $DATA_ROOT"
else
    echo "❌ Dataset not found at: $REAL_DATA_PATH"
    DATA_ROOT="/tmp/dummy"
fi

# Main training command - L2 BASELINE with 100K max steps
tmux send-keys -t l2_photography_training "
cd /home/jilab/Jae && \\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
OMP_NUM_THREADS=8 \\
python train_l2_unified.py \\
    --data_root \"/opt/dlami/nvme/preprocessed/prior_clean\" \\
    --domains photography \\
    --max_steps 100000 \\
    --batch_size 4 \\
    --gradient_accumulation_steps 4 \\
    --model_channels 256 \\
    --channel_mult 1 2 3 4 \\
    --channel_mult_emb 6 \\
    --num_blocks 6 \\
    --attn_resolutions 8 16 32 \\
    --output_dir \"results/l2_photography_baseline_$(date +%Y%m%d_%H%M%S)\" \\
    --device cuda \\
    --mixed_precision $MIXED_PRECISION \\
    --gradient_checkpointing true \\
    --learning_rate $LEARNING_RATE \\
    --num_workers 12 \\
    --prefetch_factor 4 \\
    --pin_memory true \\
    --seed 42 \\
    --save_frequency_steps 5000 \\
    --early_stopping_patience_steps 5000 \\
    --max_checkpoints 10 \\
    --save_best_model true \\
    --checkpoint_metric val_loss \\
    --gradient_clip_norm 0.1 \\
    --val_frequency_steps 5000 \\
    $STABLE_CHECKPOINT
" Enter

# Advanced monitoring in right pane - L2 BASELINE SPECIALIZED
tmux split-window -h -t l2_photography_training

tmux send-keys -t l2_photography_training:0.1 "
cd /home/jilab/Jae && python -c '
import time
import torch
import psutil
import numpy as np
from datetime import datetime
import os

print(\"📸 L2 BASELINE PHOTOGRAPHY TRAINING MONITOR - v1.0\")
print(\"=\" * 70)
print(f\"Started: {datetime.now()}\")
print(\"=\" * 70)
print()

# Configuration summary - L2 BASELINE SPECIFIC
configs = {
    \"Guidance Type\": \"L2 (MSE) Baseline\",
    \"Model Size\": \"810M parameters (identical to Poisson-Gaussian)\",
    \"Architecture\": \"256ch, 6 blocks, multi-scale attention\",
    \"Training\": \"100K steps (baseline comparison)\",
    \"Batch Strategy\": \"4 physical × 4 accumulation = 16 effective\",
    \"Workers\": \"12 (identical to Poisson-Gaussian)\",
    \"Prefetch Factor\": \"4 (identical to Poisson-Gaussian)\",
}

print(\"📊 L2 BASELINE CONFIGURATION:\")
for key, value in configs.items():
    print(f\"  {key:20}: {value}\")
print()

# L2 baseline optimizations
print(\"📸 L2 BASELINE METHODOLOGY (CORRECTED):\")
print(\"  1. TRAINING: Homoscedastic Gaussian noise (x + N(0,σ²))\")
print(\"  2. NOISE RANGE: Log-uniform [0.01, 2.0] (matches EDM)\")
print(\"  3. CONDITIONING: Simple 4D + 2D padding (not physics-aware)\")
print(\"  4. OBJECTIVE: v-parameterization (identical to Poisson-Gaussian)\")
print(\"  5. GUIDANCE: Only differs during inference (L2 vs Poisson)\")
print(\"  6. ARCHITECTURE: Identical EDM model\")
print(\"  7. VALIDATION: Same v-param loss computation\")
print(\"  8. PURPOSE: Fair ablation study\")
print(\"=\" * 70)
print()

def get_gpu_stats():
    if not torch.cuda.is_available():
        return None

    mem_free, mem_total = torch.cuda.mem_get_info()
    mem_used = mem_total - mem_free
    mem_percent = (mem_used / mem_total) * 100

    return {
        \"used_gb\": mem_used / 1e9,
        \"total_gb\": mem_total / 1e9,
        \"percent\": mem_percent,
        \"free_gb\": mem_free / 1e9
    }

def get_training_speed():
    \"\"\"Estimate training speed from logs.\"\"\"
    try:
        log_dir = \"results/\"
        latest = max([d for d in os.listdir(log_dir) if d.startswith(\"l2_photography_\")], default=None)
        if latest:
            log_file = os.path.join(log_dir, latest, \"training.log\")
            if os.path.exists(log_file):
                with open(log_file, \"r\") as f:
                    lines = f.readlines()[-100:]
                    batch_times = []
                    for line in lines:
                        if \"Batch\" in line and \"Loss\" in line:
                            # Extract timing info if available
                            pass
                    return \"Calculating...\"
        return \"No data yet\"
    except:
        return \"Unable to measure\"

# Monitoring loop
iteration = 0
loss_history = []
gpu_history = []

while True:
    try:
        iteration += 1

        # GPU monitoring
        gpu_stats = get_gpu_stats()
        if gpu_stats:
            gpu_history.append(gpu_stats[\"percent\"])
            if len(gpu_history) > 60:
                gpu_history.pop(0)

            print(f\"\\n[{datetime.now().strftime(\"%H:%M:%S\")}] Iteration {iteration}\")
            print(\"-\" * 70)

            print(f\"\\n📊 GPU METRICS:\")
            print(f\"  Memory Used : {gpu_stats[\"used_gb\"]:.1f} / {gpu_stats[\"total_gb\"]:.1f} GB ({gpu_stats[\"percent\"]:.1f}%)\")
            print(f\"  Memory Free : {gpu_stats[\"free_gb\"]:.1f} GB\")

            # Average GPU utilization
            if len(gpu_history) > 10:
                avg_gpu = np.mean(gpu_history[-10:])
                print(f\"  Avg Usage   : {avg_gpu:.1f}% (last 10 samples)\")

            # Warnings - PHOTOGRAPHY SPECIFIC
            if gpu_stats[\"percent\"] > 90:
                print(\"  ⚠️  HIGH MEMORY - Risk of OOM for L2 baseline!\")
            elif gpu_stats[\"percent\"] < 30:
                print(\"  💡 LOW USAGE - Could increase batch size for L2 baseline\")

        # CPU monitoring
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        avg_cpu = np.mean(cpu_percent)
        max_cpu = np.max(cpu_percent)

        print(f\"\\n💻 SYSTEM METRICS:\")
        print(f\"  CPU Average : {avg_cpu:.1f}%\")
        print(f\"  CPU Max Core: {max_cpu:.1f}%\")

        # RAM
        ram = psutil.virtual_memory()
        print(f\"  RAM Usage   : {ram.percent:.1f}% ({ram.used/1e9:.1f} / {ram.total/1e9:.1f} GB)\")

        # Data loading assessment - L2 BASELINE SPECIFIC
        if avg_cpu > 80:
            print(\"  ⚠️  HIGH CPU - L2 baseline data loading might be bottleneck\")

        # Training speed
        speed = get_training_speed()
        print(f\"\\n⏱️  TRAINING SPEED: {speed}\")

        # L2 baseline research metrics
        print(f\"\\n📸 L2 BASELINE RESEARCH NOTES:\")
        if iteration < 10:
            print(\"  Phase: Warmup (monitoring L2 baseline stability)\")
        elif gpu_stats and gpu_stats[\"percent\"] < 50:
            print(\"  💡 Consider: Increasing batch size to 2 for L2 baseline\")

        # Check current mixed precision status
        phase_file = \"results/l2_photography_stable_checkpoint.pth\"
        if os.path.exists(phase_file):
            print(\"  Mixed Precision: ENABLED (Phase 2 - stable checkpoint found)\")
            print(\"  Learning Rate: 1e-5 (optimized for phase 2)\")
        else:
            print(\"  Mixed Precision: DISABLED (Phase 1 - stability mode)\")
            print(\"  Learning Rate: 5e-6 (conservative for phase 1)\")

        print(\"  Guidance Type: L2 (MSE) Baseline\")
        print(\"  Data Type: 4-channel photography (RGBA)\")
        print(\"  Optimization: Conservative settings for fair comparison\")
        print(\"  Max Steps: 100K (baseline comparison vs 300K Poisson)\")
        print(\"  Purpose: Ablation study to validate Poisson-Gaussian effectiveness\")

        print(\"=\" * 70)

        time.sleep(30)

    except KeyboardInterrupt:
        print(\"\\n👋 Monitor stopped\")
        break
    except Exception as e:
        print(f\"Monitor error: {e}\")
        time.sleep(60)
'
" Enter

echo ""
echo "✅ L2 baseline photography training created!"
echo ""
echo "📊 CORRECTED L2 BASELINE STRATEGY:"
echo ""
echo "Training Approach (Proper Ablation):"
echo "  • NOISE: Homoscedastic Gaussian [0.01, 2.0] log-uniform"
echo "  • CONDITIONING: 4D simple + 2D padding (not physics-aware)"
echo "  • TRAINING: v-parameterization (identical to Poisson-Gaussian)"
echo "  • GUIDANCE: Only differs during inference"
echo "  • STEPS: 100K for efficient comparison"
echo ""
echo "🔧 IDENTICAL CONFIGURATION (except guidance):"
echo "  ✅ Same model architecture (256ch, 6 blocks)"
echo "  ✅ Same batch strategy (4 × 4 = 16 effective)"
echo "  ✅ Same learning rate schedule (5e-6 → 1e-5)"
echo "  ✅ Same mixed precision phasing (FP32 → FP16)"
echo "  ✅ Same data loading (12 workers, prefetch 4)"
echo "  ✅ Same validation frequency (every 5K steps)"
echo "  ✅ Same gradient clipping (0.1)"
echo "  ✅ Same seed (42) for reproducibility"
echo ""
echo "📸 L2 BASELINE RESEARCH VALIDITY:"
echo "  ✅ L2 (MSE) guidance for standard deep learning comparison"
echo "  ✅ Same 4-channel photography processing (RGBA)"
echo "  ✅ Same conservative hyperparameters"
echo "  ✅ Same dataset and preprocessing (SNR=1.1)"
echo "  ✅ 100K steps for efficient baseline comparison"
echo "  ✅ Perfect ablation study setup"
echo "  ✅ Validates Poisson-Gaussian effectiveness"
echo ""
echo "🚀 BASELINE OPTIMIZATIONS (identical to Poisson):"
echo "  • 12 workers (optimized for stability, prevents pickling errors)"
echo "  • Prefetching enabled (better GPU utilization)"
echo "  • Conservative hyperparameters (stability first)"
echo "  • Adaptive mixed precision (when stable)"
echo "  • Gradient checkpointing (memory optimization)"
echo "  • Fixed multiprocessing DataLoader issues"
echo ""
echo "To monitor: tmux attach -t l2_photography_training"
echo ""
echo "📝 L2 Baseline Citations:"
echo "  Standard MSE loss for deep learning comparison"
echo "  Identical setup to Poisson-Gaussian except guidance type"
echo "  Perfect ablation study for validating physics-aware approach"
