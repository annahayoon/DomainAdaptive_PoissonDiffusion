#!/bin/bash

# Quick restart script to fix the training issues
# Usage: ./restart_training.sh

SESSION_NAME="poisson_training"
OUTPUT_DIR="results/photography_training_20250920_013606"

echo "🔧 Fixing training issues and restarting..."

# Kill existing session if it exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⏹️  Stopping existing session..."
    tmux kill-session -t "$SESSION_NAME"
    sleep 2
fi

# Fixed training command
TRAIN_CMD="cd /home/jilab/Jae && python train_photography_model.py \
    --data_root \"/home/jilab/Jae/data/domain_dataset_test/test_data/photography\" \
    --epochs 100 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --model_channels 128 \
    --output_dir \"$OUTPUT_DIR\" \
    --device cuda \
    --mixed_precision true \
    --num_workers 8 \
    --pin_memory true \
    --seed 42 \
    --save_every 10 \
    --validate_every 5"

echo "🚀 Starting fixed training session..."

# Create new tmux session
tmux new-session -d -s "$SESSION_NAME" -n "training"

# Split window for monitoring
tmux split-window -v -t "$SESSION_NAME:0"
tmux select-pane -t "$SESSION_NAME:0.0"
tmux send-keys -t "$SESSION_NAME:0.0" "$TRAIN_CMD" C-m

# Setup fixed monitoring pane
tmux select-pane -t "$SESSION_NAME:0.1"
tmux send-keys -t "$SESSION_NAME:0.1" "cd /home/jilab/Jae && python -c '
import time, torch, psutil, os
print(\"📊 A40 GPU Training Monitor - FIXED\")
print(\"=\" * 50)
print(f\"🎯 Target: Batch Size 16, Effective 32\")
print(f\"🔧 Mixed Precision: True\")
print(\"=\" * 50)

def format_bytes(bytes_val):
    for unit in [\"B\", \"KB\", \"MB\", \"GB\"]:
        if bytes_val < 1024.0:
            return f\"{bytes_val:.1f} {unit}\"
        bytes_val /= 1024.0
    return f\"{bytes_val:.1f} TB\"

while True:
    try:
        if torch.cuda.is_available():
            # GPU memory info
            gpu_mem_reserved = torch.cuda.memory_reserved(0)
            gpu_mem_allocated = torch.cuda.memory_allocated(0)
            gpu_mem_free = torch.cuda.mem_get_info()[0]
            gpu_mem_total = torch.cuda.mem_get_info()[1]

            print(f\"🖥️  GPU Memory:\")
            print(f\"   Reserved: {format_bytes(gpu_mem_reserved)}\")
            print(f\"   Allocated: {format_bytes(gpu_mem_allocated)}\")
            print(f\"   Free: {format_bytes(gpu_mem_free)}\")

        # System resources
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()

        print(f\"💻 System:\")
        print(f\"   CPU: {cpu:.1f}%\")
        print(f\"   RAM: {ram.percent:.1f}% ({format_bytes(ram.used)}/{format_bytes(ram.total)})\")

        # Check for training logs
        log_file = \"$OUTPUT_DIR/logs/training.log\"
        if os.path.exists(log_file):
            with open(log_file, \"r\") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if \"epoch\" in last_line.lower() or \"loss\" in last_line.lower():
                        print(f\"📈 Latest: {last_line[-100:]}\")

        print(\"-\" * 50)
        time.sleep(30)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f\"Monitor error: {e}\")
        time.sleep(60)
'" C-m

# Setup TensorBoard pane
tmux split-window -h -t "$SESSION_NAME:0.0"
tmux select-pane -t "$SESSION_NAME:0.2"
tmux send-keys -t "$SESSION_NAME:0.2" "cd /home/jilab/Jae && tensorboard --logdir $OUTPUT_DIR/tensorboard --host 0.0.0.0 --port 6006 --reload_interval 30" C-m

# Final layout
tmux select-layout -t "$SESSION_NAME" tiled

echo "✅ Fixed training session created!"
echo "🚀 To attach: tmux attach -t $SESSION_NAME"
echo "📊 TensorBoard: http://localhost:6006"
