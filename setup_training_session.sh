#!/bin/bash

# Setup script for training Poisson-Gaussian model in tmux session
# Usage: ./setup_training_session.sh [data_path] [session_name]

set -e

# Configuration
DATA_PATH=${1:-"/home/jilab/Jae/data/domain_dataset_test/test_data/photography"}
SESSION_NAME=${2:-"poisson_training"}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4}
MODEL_CHANNELS=${MODEL_CHANNELS:-128}
OUTPUT_DIR="results/photography_training_$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up Poisson-Gaussian training session${NC}"
echo "============================================"
echo -e "Data path: ${YELLOW}$DATA_PATH${NC}"
echo -e "Session name: ${YELLOW}$SESSION_NAME${NC}"
echo -e "Epochs: ${YELLOW}$EPOCHS${NC}"
echo -e "Batch size: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "Model channels: ${YELLOW}$MODEL_CHANNELS${NC}"
echo -e "Output directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo "============================================"

# Check if data exists
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}❌ Data directory not found: $DATA_PATH${NC}"
    echo -e "${YELLOW}💡 Available data directories:${NC}"
    find /home/jilab/Jae -name "*photography*" -type d 2>/dev/null | head -5
    exit 1
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

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo -e "${YELLOW}⚠️  No GPU detected, will use CPU${NC}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/monitoring"

echo -e "${GREEN}✅ Created output directory: $OUTPUT_DIR${NC}"

# Create training command
TRAIN_CMD="cd /home/jilab/Jae && python train_photography_model.py \
    --data_root \"$DATA_PATH\" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --model_channels $MODEL_CHANNELS \
    --output_dir \"$OUTPUT_DIR\" \
    --device cuda \
    --seed 42"

echo -e "${BLUE}📝 Training command:${NC}"
echo -e "${YELLOW}$TRAIN_CMD${NC}"

# Create tmux session
echo -e "${BLUE}🪟 Creating tmux session...${NC}"

tmux new-session -d -s "$SESSION_NAME" -n "training"

# Split window for monitoring
tmux split-window -v -t "$SESSION_NAME:0"
tmux select-pane -t "$SESSION_NAME:0.0"
tmux send-keys -t "$SESSION_NAME:0.0" "$TRAIN_CMD" C-m

# Setup monitoring pane
tmux select-pane -t "$SESSION_NAME:0.1"
tmux send-keys -t "$SESSION_NAME:0.1" "cd /home/jilab/Jae && python -c '
import time, torch, psutil
print(\"📊 Real-time Training Monitor\")
print(\"=\" * 40)
while True:
    try:
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info()[0]
            gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, \"utilization\") else \"N/A\"
            print(f\"🖥️  GPU: {gpu_mem/1e9:.1f} GB used, Util: {gpu_util}\")
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        print(f\"💻 CPU: {cpu:.1f}%, RAM: {ram:.1f}%\")
        print(\"-\" * 30)
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
tmux send-keys -t "$SESSION_NAME:0.2" "cd /home/jilab/Jae && tensorboard --logdir $OUTPUT_DIR/tensorboard --host 0.0.0.0 --port 6006" C-m

# Final layout
tmux select-layout -t "$SESSION_NAME" tiled

echo -e "${GREEN}✅ Tmux session '$SESSION_NAME' created!${NC}"
echo "============================================"
echo -e "${BLUE}📋 Session Layout:${NC}"
echo -e "  ${YELLOW}Pane 1 (top-left):${NC} Training script"
echo -e "  ${YELLOW}Pane 2 (bottom-left):${NC} System monitoring"
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
echo -e "${GREEN}🎉 Ready to train!${NC}"
