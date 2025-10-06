#!/bin/bash

# Training script for EDM native training on photography data
# This script uses EDM's native utilities and training structure
# Conservative settings optimized for smaller photography dataset size
# Uses more realistic parameters than default EDM settings
# Includes split-screen GPU monitoring for real-time resource tracking

# Activate conda environment if needed
# conda activate your_env_name

# GPU monitoring function (inspired by astronomy training script)
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
            echo "║                    📊 GPU MONITORING DASHBOARD                 ║"
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
            printf "║    RAM Usage   : %5.1f%%                                        ║\n" "$(free | grep Mem | awk '{printf \"%.1f\", \$3/\$2 * 100.0}')"
            echo "║                                                                ║"
            echo "║  🎯 TRAINING STATUS:                                           ║"
            printf "║    Log File    : %-45s ║\n" "$(basename "$log_file")"
            echo "║    Last Update : $(date +'%H:%M:%S')                             ║"
            echo "║                                                                ║"

            # Warnings for photography training
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
                echo "║  ⚠️  HIGH MEMORY - Risk of OOM for photography model!         ║"
            elif [ "$low_usage" = "1" ]; then
                echo "║  💡 LOW USAGE - Could increase batch size                      ║"
            else
                echo "║  ✅ GPU usage within optimal range                             ║"
            fi

            echo "║                                                                ║"
            echo "╚════════════════════════════════════════════════════════════════╝"

            # Check if training is still running
            if ! pgrep -f "train_photography_edm_native.py" > /dev/null; then
                echo "🎉 TRAINING COMPLETED - MONITORING STOPPED"
                break
            fi
        else
            echo "❌ Failed to get GPU stats - monitoring paused"
        fi

        sleep 60  # Update every 1 minute (changed from 5 seconds)
    done
}

# Set paths
DATA_ROOT="/home/jilab/Jae/dataset/processed/png_tiles/photography"
METADATA_JSON="/home/jilab/Jae/dataset/processed/metadata_photography_incremental.json"
OUTPUT_DIR="results/edm_native_photography_$(date +%Y%m%d)_$(date +%H%M%S)"

# Training hyperparameters (optimized for 12K samples, conservative to avoid overfitting)
# Batch size 5 tested safe (44.07 GB peak memory) - ~24% faster than batch_size=2
# Conservative total training to prevent overfitting on 12K samples
# Calculation: 12,000 samples / batch_size=5 = 2,400 steps/epoch
# 25 epochs × 2,400 steps × 5 images = 300,000 images = 300 kimg
BATCH_SIZE=5            # Total batch size (tested safe, 44.07 GB peak memory)
BATCH_GPU=5             # Batch size per GPU
TOTAL_KIMG=300          # 300 kimg (~25 epochs over 12K samples) - conservative to avoid overfitting
EMA_HALFLIFE_KIMG=50    # EMA half-life
LR_RAMPUP_KIMG=10       # Learning rate warmup (increased for stability)
KIMG_PER_TICK=12        # Progress print interval (~1 epoch = 12 kimg, ~1.5 hours with batch_size=5)
SNAPSHOT_TICKS=2        # Save every 2 ticks (~2 epochs = 24 kimg, ~3 hours) - more frequent validation
EARLY_STOPPING_PATIENCE=5   # Stop if no improvement for 5 validation checks (~10 epochs) - more aggressive

# Model architecture (ADM defaults from EDM)
IMG_RESOLUTION=256
MODEL_CHANNELS=192      # ADM default
CHANNEL_MULT="1 2 3 4"  # ADM default
LR=0.0001               # Learning rate

# Device
DEVICE="cuda"
SEED=42

echo "=================================================="
echo "EDM Native Training - Photography Domain"
echo "Conservative training to prevent overfitting on 12K samples"
echo "=================================================="
echo "Data root: $DATA_ROOT"
echo "Metadata: $METADATA_JSON"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE (tested safe, 44.07 GB peak memory)"
echo "Total training: $TOTAL_KIMG kimg (~25 epochs)"
echo "Architecture: ADM (DhariwalUNet)"
echo "Model channels: $MODEL_CHANNELS"
echo "Early stopping: Enabled (patience=$EARLY_STOPPING_PATIENCE)"
echo "Overfitting prevention: Conservative 25 epochs + early stopping"
echo "=================================================="
echo "📊 SPLIT-SCREEN GPU MONITORING: ENABLED"
echo "   Real-time GPU memory, CPU usage, and training status"
echo "   Monitoring updates every 1 minute"
echo "   Expected: ~1.5 hours per epoch, ~3 hours between checkpoints"
echo "=================================================="

# Check if required tools are available
echo "🔧 Checking system requirements..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found - GPU monitoring will be limited"
fi

if ! command -v bc &> /dev/null; then
    echo "⚠️  WARNING: bc (calculator) not found - installing for GPU monitoring..."
    # Try to install bc if available
    apt-get update && apt-get install -y bc 2>/dev/null || echo "Could not install bc - monitoring may have issues"
fi

# Run training using EDM's native pattern with monitoring
echo "🚀 Starting training with split-screen GPU monitoring..."
echo "   Training will run in foreground"
echo "   GPU monitoring will run in background"
echo "   Press Ctrl+C to stop both processes"
echo ""

# Start GPU monitoring in background
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

# Run training (this will be the main process)
echo "🎯 Starting EDM Native Training..."
python train_photography_edm_native.py \
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
