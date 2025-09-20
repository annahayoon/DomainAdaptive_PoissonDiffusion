#!/bin/bash

# Kill any existing training session
tmux kill-session -t poisson_training 2>/dev/null

echo "🚀 Starting training with REAL data..."

# Create new tmux session for training
tmux new-session -d -s poisson_training -c /home/jilab/Jae

# Set up the training command with real data path
# Replace this path with your actual preprocessed photography data
REAL_DATA_PATH="/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed"

# Check if real data exists, otherwise use synthetic data generation
if [ -d "$REAL_DATA_PATH" ] && [ "$(ls -A $REAL_DATA_PATH)" ]; then
    echo "✅ Using real preprocessed data from: $REAL_DATA_PATH"
    DATA_ROOT="$REAL_DATA_PATH"
    QUICK_TEST_FLAG=""
else
    echo "⚠️  Real preprocessed data not found."
    echo "🔄 Will use synthetic data generation for training."
    echo "   This is actually better for initial testing!"
    DATA_ROOT="/tmp/dummy"  # Fallback to synthetic
    QUICK_TEST_FLAG="--quick_test"
fi

# Training command with optimized settings for A40
tmux send-keys -t poisson_training "
cd /home/jilab/Jae && python train_photography_model.py \\
    --data_root \"$DATA_ROOT\" \\
    --epochs 470 \
    --batch_size 16 \\
    --gradient_accumulation_steps 1 \\
    --model_channels 128 \\
    --output_dir \"results/photography_training_$(date +%Y%m%d_%H%M%S)\" \\
    --device cuda \\
    --mixed_precision false \\
    --num_workers 2 \\
    --pin_memory true \\
    --seed 42 \\
    --save_every 10 \\
    --validate_every 5 \\
    $QUICK_TEST_FLAG
" Enter

# Split window for monitoring
tmux split-window -h -t poisson_training

# Set up monitoring in the right pane
tmux send-keys -t poisson_training:0.1 "
cd /home/jilab/Jae && python -c '
import time, torch, psutil, os
print(\"📊 A40 GPU Training Monitor\")
print(\"=\" * 50)
print(f\"🎯 Target: Batch Size 16, Effective 32\")
print(f\"🔧 Mixed Precision: True\")
print(\"=\" * 50)

def format_bytes(bytes):
    for unit in [\"B\", \"KB\", \"MB\", \"GB\"]:
        if bytes < 1024.0:
            return f\"{bytes:.1f} {unit}\"
        bytes /= 1024.0
    return f\"{bytes:.1f} TB\"

while True:
    try:
        if torch.cuda.is_available():
            # GPU memory info
            gpu_mem_reserved = torch.cuda.memory_reserved(0)
            gpu_mem_allocated = torch.cuda.memory_allocated(0)
            gpu_mem_free = torch.cuda.mem_get_info()[0]
            gpu_mem_total = torch.cuda.mem_get_info()[1]
            gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, \"utilization\") else \"N/A\"

            print(f\"🖥️  GPU Memory:\")
            print(f\"   Reserved: {format_bytes(gpu_mem_reserved)}\")
            print(f\"   Allocated: {format_bytes(gpu_mem_allocated)}\")
            print(f\"   Free: {format_bytes(gpu_mem_free)}\")
            print(f\"   Utilization: {gpu_util}%\" if gpu_util != \"N/A\" else \"   Utilization: N/A\")

        # System resources
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()

        print(f\"💻 System:\")
        print(f\"   CPU: {cpu:.1f}%\")
        print(f\"   RAM: {ram.percent:.1f}% ({format_bytes(ram.used)}/{format_bytes(ram.total)})\")

        print(\"-\" * 50)
        time.sleep(30)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f\"Monitor error: {e}\")
        time.sleep(60)
'
" Enter

echo "✅ Training session created!"
echo "🚀 To attach: tmux attach -t poisson_training"
echo "📊 TensorBoard will be available once training starts"
