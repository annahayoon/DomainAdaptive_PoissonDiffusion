#!/bin/bash

# Optimized setup script for training Poisson-Gaussian model with A40 GPU
# Usage: ./setup_training_session_optimized.sh [data_path] [session_name]

set -e

# Optimized Configuration for A40 GPU (45GB VRAM)
DATA_PATH=${1:-"/home/jilab/Jae/data/domain_dataset_test/test_data/photography"}
SESSION_NAME=${2:-"poisson_training"}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-16}  # Optimized for A40: 4x larger than original
MODEL_CHANNELS=${MODEL_CHANNELS:-128}
OUTPUT_DIR="results/photography_training_$(date +%Y%m%d_%H%M%S)"

# A40-specific optimizations
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}  # Effective batch size: 32
MIXED_PRECISION=${MIXED_PRECISION:-true}  # Enable for A40
NUM_WORKERS=${NUM_WORKERS:-8}  # Optimize data loading
PIN_MEMORY=${PIN_MEMORY:-true}  # Faster GPU transfers

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up OPTIMIZED Poisson-Gaussian training session${NC}"
echo "============================================"
echo -e "Data path: ${YELLOW}$DATA_PATH${NC}"
echo -e "Session name: ${YELLOW}$SESSION_NAME${NC}"
echo -e "Epochs: ${YELLOW}$EPOCHS${NC}"
echo -e "Batch size: ${YELLOW}$BATCH_SIZE${NC} (effective: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)))"
echo -e "Model channels: ${YELLOW}$MODEL_CHANNELS${NC}"
echo -e "Mixed precision: ${YELLOW}$MIXED_PRECISION${NC}"
echo -e "Gradient accumulation: ${YELLOW}$GRADIENT_ACCUMULATION_STEPS${NC}"
echo -e "Output directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo "============================================"

# Check if data exists
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}❌ Data directory not found: $DATA_PATH${NC}"
    echo -e "${YELLOW}💡 Available data directories:${NC}"
    find /home/jilab/Jae -name "*photography*" -type d 2>/dev/null | head -5
    exit 1
fi

# Count available data files
DATA_COUNT=$(find "$DATA_PATH" -name "*.arw" -o -name "*.dng" -o -name "*.nef" -o -name "*.cr2" | wc -l)
echo -e "${GREEN}✅ Found $DATA_COUNT photography files${NC}"

if [ "$DATA_COUNT" -lt 10 ]; then
    echo -e "${YELLOW}⚠️  Limited data ($DATA_COUNT files). Consider reducing batch size or epochs.${NC}"
fi

# Check for existing session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Session '$SESSION_NAME' already exists${NC}"
    echo -e "${YELLOW}💡 Attach with: tmux attach -t $SESSION_NAME${NC}"
    echo -e "${YELLOW}💡 Kill with: tmux kill-session -t $SESSION_NAME${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Enhanced GPU check with memory optimization
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ GPU detected${NC}"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits)
    echo "$GPU_INFO"

    # Extract memory info
    GPU_TOTAL=$(echo "$GPU_INFO" | cut -d',' -f2 | tr -d ' ')
    GPU_FREE=$(echo "$GPU_INFO" | cut -d',' -f3 | tr -d ' ')

    # Memory recommendations
    if [ "$GPU_TOTAL" -gt 40000 ]; then
        echo -e "${GREEN}✅ A40/A100 detected - using optimized settings${NC}"
    elif [ "$GPU_TOTAL" -gt 20000 ]; then
        echo -e "${YELLOW}⚠️  RTX 3090/4090 detected - consider batch_size=8${NC}"
        BATCH_SIZE=8
    else
        echo -e "${YELLOW}⚠️  Smaller GPU detected - using conservative batch_size=4${NC}"
        BATCH_SIZE=4
    fi
else
    echo -e "${YELLOW}⚠️  No GPU detected, will use CPU${NC}"
    BATCH_SIZE=2  # Much smaller for CPU
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/monitoring"
mkdir -p "$OUTPUT_DIR/tensorboard"

echo -e "${GREEN}✅ Created output directory: $OUTPUT_DIR${NC}"

# Create optimized training command
TRAIN_CMD="cd /home/jilab/Jae && python train_photography_model.py \
    --data_root \"$DATA_PATH\" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --model_channels $MODEL_CHANNELS \
    --output_dir \"$OUTPUT_DIR\" \
    --device cuda \
    --mixed_precision $MIXED_PRECISION \
    --num_workers $NUM_WORKERS \
    --pin_memory $PIN_MEMORY \
    --seed 42 \
    --save_every 10 \
    --validate_every 5"

echo -e "${BLUE}📝 Optimized training command:${NC}"
echo -e "${YELLOW}$TRAIN_CMD${NC}"

# Create tmux session
echo -e "${BLUE}🪟 Creating tmux session...${NC}"

tmux new-session -d -s "$SESSION_NAME" -n "training"

# Split window for monitoring
tmux split-window -v -t "$SESSION_NAME:0"
tmux select-pane -t "$SESSION_NAME:0.0"
tmux send-keys -t "$SESSION_NAME:0.0" "$TRAIN_CMD" C-m

# Setup enhanced monitoring pane
tmux select-pane -t "$SESSION_NAME:0.1"
tmux send-keys -t "$SESSION_NAME:0.1" "cd /home/jilab/Jae && python -c '
import time, torch, psutil, os
print(\"📊 A40 GPU Training Monitor\")
print(\"=\" * 50)
print(f\"🎯 Target: Batch Size {$BATCH_SIZE}, Effective {$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))}\")
print(f\"🔧 Mixed Precision: $MIXED_PRECISION\")
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

echo -e "${GREEN}✅ Optimized tmux session '$SESSION_NAME' created!${NC}"
echo "============================================"
echo -e "${BLUE}📋 Session Layout:${NC}"
echo -e "  ${YELLOW}Pane 1 (top-left):${NC} Training script"
echo -e "  ${YELLOW}Pane 2 (bottom-left):${NC} A40 GPU monitoring"
echo -e "  ${YELLOW}Pane 3 (top-right):${NC} TensorBoard"
echo
echo -e "${BLUE}🚀 To start training:${NC}"
echo -e "  ${YELLOW}tmux attach -t $SESSION_NAME${NC}"
echo
echo -e "${BLUE}📊 To monitor from another terminal:${NC}"
echo -e "  ${YELLOW}tmux attach -t $SESSION_NAME${NC}"
echo -e "  ${YELLOW}tensorboard --logdir $OUTPUT_DIR/tensorboard${NC} (access at http://localhost:6006)"
echo
echo -e "${BLUE}⏹️  To stop training:${NC}"
echo -e "  ${YELLOW}tmux kill-session -t $SESSION_NAME${NC}"
echo
echo -e "${BLUE}💾 Outputs will be saved to:${NC}"
echo -e "  ${YELLOW}$OUTPUT_DIR${NC}"
echo
echo -e "${GREEN}🎉 Ready for A40 GPU training!${NC}"
echo -e "${BLUE}💡 Optimizations applied:${NC}"
echo -e "  • Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)))"
echo -e "  • Mixed precision training enabled"
echo -e "  • Enhanced GPU memory monitoring"
echo -e "  • Optimized data loading ($NUM_WORKERS workers)"
