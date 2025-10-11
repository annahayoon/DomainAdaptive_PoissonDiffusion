#!/bin/bash

# Training script for EDM with float32 .pt files (NO QUANTIZATION)
# This script trains on full-precision 32-bit float PyTorch tensors
# Conservative settings optimized for smaller dataset size
# Uses more realistic parameters than default EDM settings
# Includes split-screen GPU monitoring for real-time resource tracking
#
# USAGE:
#   1. Start fresh training (auto-generates timestamped directory):
#      ./run_edm_npy_train.sh
#
#   2. Resume training from existing directory:
#      OUTPUT_DIR="results/edm_pt_training_20251007_081108" ./run_edm_npy_train.sh
#      OR just run the script again - it auto-detects and resumes if checkpoints exist
#
#   3. Override output directory:
#      OUTPUT_DIR="results/my_custom_training" ./run_edm_npy_train.sh
#
# The script automatically:
# - Creates timestamped directories for fresh training
# - Detects and resumes from existing checkpoints
# - Restores optimizer state for smooth continuation
# - Monitors GPU usage in real-time

set -e

# GPU monitoring function
get_gpu_stats() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "nvidia-smi not available"
        return 1
    fi

    # Get GPU memory info using nvidia-smi for more accurate readings
    local mem_info=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1)
    if [ $? -ne 0 ]; then
        echo "Failed to get GPU memory info"
        return 1
    fi

    local used_mem=$(echo $mem_info | cut -d',' -f1 | tr -d ' ')
    local total_mem=$(echo $mem_info | cut -d',' -f2 | tr -d ' ')

    # Calculate GB values and percentages (with fallback if bc not available)
    if command -v bc &> /dev/null; then
        local used_gb=$(echo "scale=2; $used_mem / 1024" | bc)
        local total_gb=$(echo "scale=2; $total_mem / 1024" | bc)
        local percent=$(echo "scale=1; ($used_mem * 100) / $total_mem" | bc)
        local free_gb=$(echo "scale=2; $total_gb - $used_gb" | bc)
    else
        # Fallback calculation using awk for basic arithmetic
        local used_gb=$(echo "$used_mem 1024" | awk '{printf "%.2f", $1/$2}')
        local total_gb=$(echo "$total_mem 1024" | awk '{printf "%.2f", $1/$2}')
        local percent=$(echo "$used_mem $total_mem" | awk '{printf "%.1f", ($1*100)/$2}')
        local free_gb=$(echo "$total_gb $used_gb" | awk '{printf "%.2f", $1-$2}')
    fi

    echo "used_gb:$used_gb,total_gb:$total_gb,percent:$percent,free_gb:$free_gb"
}

# Split-screen monitoring function
monitor_training() {
    local log_file="$1"
    local iteration=0
    local gpu_history=()

    echo "🔍 SPLIT-SCREEN GPU MONITORING STARTED"
    echo "========================================"

    while true; do
        iteration=$((iteration + 1))

        # Get GPU stats
        local gpu_data=$(get_gpu_stats)
        if [ $? -eq 0 ]; then
            # Parse GPU data
            local used_gb=$(echo $gpu_data | cut -d',' -f1 | cut -d':' -f2)
            local total_gb=$(echo $gpu_data | cut -d',' -f2 | cut -d':' -f2)
            local percent=$(echo $gpu_data | cut -d',' -f3 | cut -d':' -f2)
            local free_gb=$(echo $gpu_data | cut -d',' -f4 | cut -d':' -f2)

            # Track GPU history for averaging
            gpu_history+=($percent)
            if [ ${#gpu_history[@]} -gt 60 ]; then
                gpu_history=("${gpu_history[@]:1}")
            fi

            # Calculate average GPU usage
            local avg_gpu="0"
            if [ ${#gpu_history[@]} -gt 10 ]; then
                local sum=0
                for val in "${gpu_history[@]: -10}"; do
                    if command -v bc &> /dev/null; then
                        sum=$(echo "$sum + $val" | bc)
                    else
                        sum=$(echo "$sum $val" | awk '{print $1 + $2}')
                    fi
                done
                if command -v bc &> /dev/null; then
                    avg_gpu=$(echo "scale=1; $sum / 10" | bc)
                else
                    avg_gpu=$(echo "$sum 10" | awk '{printf "%.1f", $1/$2}')
                fi
            fi

            # Clear screen and display monitoring info
            echo -e "\033[2J\033[H"  # Clear screen and move cursor to top

            echo "╔════════════════════════════════════════════════════════════════╗"
            echo "║            📊 GPU MONITORING DASHBOARD (PT Float32)           ║"
            echo "║════════════════════════════════════════════════════════════════║"
            printf "║  Iteration: %-8s  Time: %-15s                     ║\n" "$iteration" "$(date +'%H:%M:%S')"
            echo "║════════════════════════════════════════════════════════════════║"
            echo "║                                                                ║"
            printf "║  📊 GPU METRICS:                                               ║\n"
            printf "║    Memory Used : %6.1f / %5.1f GB (%5.1f%%)               ║\n" "$used_gb" "$total_gb" "$percent"
            printf "║    Memory Free : %6.1f GB                                     ║\n" "$free_gb"
            printf "║    Avg Usage   : %5.1f%% (last 10 samples)                    ║\n" "$avg_gpu"
            echo "║                                                                ║"
            printf "║  💻 SYSTEM METRICS:                                            ║\n"
            printf "║    CPU Usage   : %5.1f%%                                        ║\n" "$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')"
            printf "║    RAM Usage   : %5.1f%%                                        ║\n" "$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')"
            echo "║                                                                ║"
            echo "║  🎯 TRAINING STATUS:                                          ║"
            printf "║    Log File    : %-45s ║\n" "$(basename "$log_file")"
            echo "║    Last Update : $(date +'%H:%M:%S')                             ║"
            echo "║                                                                ║"

            # Warnings
            local high_mem=false
            local low_usage=false

            if command -v bc &> /dev/null; then
                high_mem=$(echo "$percent > 85" | bc -l)
                low_usage=$(echo "$percent < 30" | bc -l)
            else
                high_mem=$(echo "$percent 85" | awk '{print ($1 > $2) ? 1 : 0}')
                low_usage=$(echo "$percent 30" | awk '{print ($1 < $2) ? 1 : 0}')
            fi

            if [ "$high_mem" = "1" ]; then
                echo "║  ⚠️  HIGH MEMORY USAGE - Monitor GPU consumption              ║"
            elif [ "$low_usage" = "1" ]; then
                echo "║  💡 LOW USAGE - Could increase batch size                      ║"
            else
                echo "║  ✅ GPU usage within optimal range                             ║"
            fi

            echo "║                                                                ║"
            echo "╚════════════════════════════════════════════════════════════════╝"

            # Check if training is still running
            if ! pgrep -f "train_pt_edm_native_astronomy.py" > /dev/null; then
                echo "🎉 TRAINING COMPLETED - MONITORING STOPPED"
                break
            fi
        else
            echo "❌ Failed to get GPU stats - monitoring paused"
        fi

        sleep 60  # Update every 1 minute
    done
}

# Configuration
DATA_ROOT="${DATA_ROOT:-dataset/processed/pt_tiles/astronomy/clean}"
METADATA_JSON="${METADATA_JSON:-dataset/processed/metadata_astronomy_incremental.json}"

# Output directory configuration
# To resume training: Set OUTPUT_DIR to existing directory path
# To start fresh: Leave OUTPUT_DIR empty or set to non-existent directory
OUTPUT_DIR="${OUTPUT_DIR:-}"  # Can be set via environment variable or left empty for auto-generation

# Auto-generate output directory if not specified or doesn't exist
if [ -z "$OUTPUT_DIR" ]; then
    # Generate timestamp-based directory name
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="results/edm_pt_training_astronomy_${TIMESTAMP}"
    echo "📁 Creating new training directory: $OUTPUT_DIR"
elif [ ! -d "$OUTPUT_DIR" ]; then
    echo "📁 Directory does not exist, will create: $OUTPUT_DIR"
else
    echo "📁 Directory exists, will attempt to resume: $OUTPUT_DIR"
fi

# Training hyperparameters (optimized settings for efficiency and stability)
# Batch size optimized for memory efficiency - maximum safe value
BATCH_SIZE=4            # Total batch size (optimal balance of efficiency and memory)
BATCH_GPU=4             # Batch size per GPU (same as total for single GPU)
TOTAL_KIMG=300          # 300 kimg (~25 epochs) - conservative to avoid overfitting
EMA_HALFLIFE_KIMG=50    # EMA half-life
LR_RAMPUP_KIMG=10       # Learning rate warmup (increased for stability)
KIMG_PER_TICK=12        # Progress print interval (~1 epoch)
SNAPSHOT_TICKS=2        # Save every 2 ticks (~2 epochs) - more frequent validation
EARLY_STOPPING_PATIENCE=5   # Stop if no improvement for 5 validation checks (~10 epochs)

# Model architecture (ADM defaults from EDM)
IMG_RESOLUTION=256
CHANNELS=1              # 1 for grayscale, 3 for RGB
MODEL_CHANNELS=192      # ADM default
CHANNEL_MULT="1 2 3 4"  # ADM default
LR=0.0001               # Learning rate

# Device
DEVICE="cuda"
SEED=42

echo "=================================================="
echo "EDM Training with Float32 .pt Files - ASTRONOMY"
echo "NO QUANTIZATION - FULL PRECISION"
echo "Conservative training to prevent overfitting"
echo "=================================================="
echo "Data root: $DATA_ROOT"
echo "Metadata: $METADATA_JSON"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE (optimal for efficiency and memory)"
echo "Total training: $TOTAL_KIMG kimg (~25 epochs)"
echo "Architecture: ADM (DhariwalUNet)"
echo "Model channels: $MODEL_CHANNELS"
echo "Early stopping: Enabled (patience=$EARLY_STOPPING_PATIENCE)"
echo "Overfitting prevention: Conservative 25 epochs + early stopping"
echo "=================================================="
echo "📊 SPLIT-SCREEN GPU MONITORING: ENABLED"
echo "   Real-time GPU memory, CPU usage, and training status"
echo "   Monitoring updates every 1 minute"
echo "   Note: Float32 tensors preserve full precision"
echo "=================================================="

# Check if required tools are available
echo "🔧 Checking system requirements..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found - GPU monitoring will be limited"
fi

if ! command -v bc &> /dev/null; then
    echo "⚠️  WARNING: bc (calculator) not found - will use awk fallback"
fi

# Find latest checkpoint if resuming
LATEST_CHECKPOINT=""
RESUME_MODE="false"
if [ -d "$OUTPUT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -v "$OUTPUT_DIR"/network-snapshot-*.pkl 2>/dev/null | tail -n 1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        RESUME_MODE="true"
        # Extract kimg from checkpoint filename
        CHECKPOINT_KIMG=$(basename "$LATEST_CHECKPOINT" | sed 's/network-snapshot-0*\([0-9]*\)\.pkl/\1/')
        REMAINING_KIMG=$((TOTAL_KIMG - CHECKPOINT_KIMG))
        echo ""
        echo "🔄 RESUME MODE DETECTED"
        echo "   Found checkpoint: $LATEST_CHECKPOINT"
        echo "   Current progress: ${CHECKPOINT_KIMG} kimg"
        echo "   Remaining: ${REMAINING_KIMG} kimg (to reach ${TOTAL_KIMG} kimg)"
        echo ""
    fi
fi

if [ "$RESUME_MODE" = "false" ]; then
    echo ""
    echo "🆕 STARTING FRESH TRAINING"
    echo "   No checkpoints found in output directory"
    echo "   Training from scratch for ${TOTAL_KIMG} kimg"
    echo ""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start GPU monitoring in background
echo ""
echo "🚀 Starting training with split-screen GPU monitoring..."
echo "   Training will run in foreground"
echo "   GPU monitoring will run in background"
echo "   Press Ctrl+C to stop both processes"
echo ""

LOG_FILE="$OUTPUT_DIR/training.log"
monitor_training "$LOG_FILE" &
MONITOR_PID=$!

# Give monitoring a moment to start
sleep 2

# Trap to kill monitoring when script is terminated
cleanup() {
    echo ""
    echo "🛑 Stopping GPU monitoring..."
    kill $MONITOR_PID 2>/dev/null
    echo "✅ Monitoring stopped"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Run training (Python script auto-detects and resumes from latest checkpoint)
python3 train_pt_edm_native_astronomy.py \
    --data_root "$DATA_ROOT" \
    --metadata_json "$METADATA_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --batch_gpu $BATCH_GPU \
    --total_kimg $TOTAL_KIMG \
    --ema_halflife_kimg $EMA_HALFLIFE_KIMG \
    --lr_rampup_kimg $LR_RAMPUP_KIMG \
    --kimg_per_tick $KIMG_PER_TICK \
    --snapshot_ticks $SNAPSHOT_TICKS \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --img_resolution $IMG_RESOLUTION \
    --channels $CHANNELS \
    --model_channels $MODEL_CHANNELS \
    --channel_mult $CHANNEL_MULT \
    --lr $LR \
    --device $DEVICE \
    --seed $SEED

# Training completed - stop monitoring
echo ""
echo "🛑 Training completed - stopping GPU monitoring..."
kill $MONITOR_PID 2>/dev/null
echo "✅ Monitoring stopped"

echo "=================================================="
echo "Training completed!"
echo "Checkpoints saved in: $OUTPUT_DIR"
echo "Best model: $OUTPUT_DIR/best_model.pkl"
echo "=================================================="
