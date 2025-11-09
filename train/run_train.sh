#!/bin/bash
# Sensor training script for EDM with float32 .pt files (Sony/Fuji)
# Usage: CONFIG_FILE="config/sony.yaml" ./train/run_train.sh

set -e

get_gpu_stats() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "nvidia-smi not available"
        return 1
    fi

    local mem_info=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1)
    if [ $? -ne 0 ]; then
        echo "Failed to get GPU memory info"
        return 1
    fi

    local used_mem=$(echo $mem_info | cut -d',' -f1 | tr -d ' ')
    local total_mem=$(echo $mem_info | cut -d',' -f2 | tr -d ' ')

    if command -v bc &> /dev/null; then
        local used_gb=$(echo "scale=2; $used_mem / 1024" | bc)
        local total_gb=$(echo "scale=2; $total_mem / 1024" | bc)
        local percent=$(echo "scale=1; ($used_mem * 100) / $total_mem" | bc)
        local free_gb=$(echo "scale=2; $total_gb - $used_gb" | bc)
    else
        local used_gb=$(echo "$used_mem 1024" | awk '{printf "%.2f", $1/$2}')
        local total_gb=$(echo "$total_mem 1024" | awk '{printf "%.2f", $1/$2}')
        local percent=$(echo "$used_mem $total_mem" | awk '{printf "%.1f", ($1*100)/$2}')
        local free_gb=$(echo "$total_gb $used_gb" | awk '{printf "%.2f", $1-$2}')
    fi

    echo "used_gb:$used_gb,total_gb:$total_gb,percent:$percent,free_gb:$free_gb"
}

monitor_training() {
    local log_file="$1"
    local iteration=0
    local gpu_history=()

    while true; do
        iteration=$((iteration + 1))
        local gpu_data=$(get_gpu_stats)
        if [ $? -eq 0 ]; then
            local used_gb=$(echo $gpu_data | cut -d',' -f1 | cut -d':' -f2)
            local total_gb=$(echo $gpu_data | cut -d',' -f2 | cut -d':' -f2)
            local percent=$(echo $gpu_data | cut -d',' -f3 | cut -d':' -f2)
            local free_gb=$(echo $gpu_data | cut -d',' -f4 | cut -d':' -f2)

            gpu_history+=($percent)
            if [ ${#gpu_history[@]} -gt 60 ]; then
                gpu_history=("${gpu_history[@]:1}")
            fi

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

            echo -e "\033[2J\033[H"

            echo "╔════════════════════════════════════════════════════════════════╗"
            echo "║        📊 GPU MONITORING DASHBOARD (Sensor PT Float32)   ║"
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

            if ! pgrep -f "train/train.py" > /dev/null; then
                echo "🎉 TRAINING COMPLETED - MONITORING STOPPED"
                break
            fi
        else
            echo "❌ Failed to get GPU stats - monitoring paused"
        fi

        sleep 60  # Update every 1 minute
    done
}

if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: CONFIG_FILE environment variable is required"
    echo "Usage: CONFIG_FILE=config/sony.yaml ./train/run_train.sh"
    exit 1
fi

HAS_MULTIPLE_DATASETS=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print('datasets' in c and isinstance(c.get('datasets'), list))" 2>/dev/null || echo "False")

if [ "$HAS_MULTIPLE_DATASETS" = "True" ]; then
    DATA_ROOT=""
    METADATA_JSON=""
else
    DATA_ROOT=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('data_root', 'dataset/processed/pt_tiles/sony'))" 2>/dev/null || echo "dataset/processed/pt_tiles/sony")
    METADATA_JSON=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('metadata_json', 'dataset/processed/metadata_sony_incremental.json'))" 2>/dev/null || echo "dataset/processed/metadata_sony_incremental.json")
fi

OUTPUT_DIR="${OUTPUT_DIR:-}"
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    # Extract sensor from config file name (sony, fuji, sidd, etc.)
    CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
    SENSOR=$(echo "$CONFIG_BASENAME" | grep -oE "(sony|fuji|sidd)" || echo "$CONFIG_BASENAME")
    OUTPUT_DIR="results/edm_${SENSOR}_training_${TIMESTAMP}"
fi
BATCH_SIZE=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('batch_size', 4))" 2>/dev/null || echo "4")
BATCH_GPU=$BATCH_SIZE
TOTAL_KIMG=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('total_kimg', 300))" 2>/dev/null || echo "300")
EMA_HALFLIFE_KIMG=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('ema_halflife_kimg', 50))" 2>/dev/null || echo "50")
LR_RAMPUP_KIMG=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('lr_rampup_kimg', 10))" 2>/dev/null || echo "10")
KIMG_PER_TICK=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('kimg_per_tick', 12))" 2>/dev/null || echo "12")
SNAPSHOT_TICKS=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('snapshot_ticks', 2))" 2>/dev/null || echo "2")
EARLY_STOPPING_PATIENCE=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('early_stopping_patience', 5))" 2>/dev/null || echo "5")
IMG_RESOLUTION=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('img_resolution', 256))" 2>/dev/null || echo "256")
CHANNELS=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('channels', 3))" 2>/dev/null || echo "3")
MODEL_CHANNELS=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('model_channels', 192))" 2>/dev/null || echo "192")
CHANNEL_MULT=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(' '.join(map(str, c.get('channel_mult', [1, 2, 3, 4]))))" 2>/dev/null || echo "1 2 3 4")
LR=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('learning_rate', 0.0001))" 2>/dev/null || echo "0.0001")
DEVICE=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('device', 'cuda'))" 2>/dev/null || echo "cuda")
SEED=$(python3 -c "import yaml; f=open('$CONFIG_FILE'); c=yaml.safe_load(f); print(c.get('seed', 42))" 2>/dev/null || echo "42")

echo "Configuration: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/training.log"
monitor_training "$LOG_FILE" &
MONITOR_PID=$!
sleep 2

cleanup() {
    kill $MONITOR_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM
TRAIN_ARGS=(
    --config "$CONFIG_FILE"
    --output_dir "$OUTPUT_DIR"
    --batch_size $BATCH_SIZE
    --batch_gpu $BATCH_GPU
    --total_kimg $TOTAL_KIMG
    --ema_halflife_kimg $EMA_HALFLIFE_KIMG
    --lr_rampup_kimg $LR_RAMPUP_KIMG
    --kimg_per_tick $KIMG_PER_TICK
    --snapshot_ticks $SNAPSHOT_TICKS
    --early_stopping_patience $EARLY_STOPPING_PATIENCE
    --img_resolution $IMG_RESOLUTION
    --channels $CHANNELS
    --model_channels $MODEL_CHANNELS
    --channel_mult $CHANNEL_MULT
    --lr $LR
    --device $DEVICE
    --seed $SEED
)

if [ "$HAS_MULTIPLE_DATASETS" != "True" ]; then
    TRAIN_ARGS+=(--data_root "$DATA_ROOT")
    TRAIN_ARGS+=(--metadata_json "$METADATA_JSON")
fi

python3 train/train.py "${TRAIN_ARGS[@]}"
kill $MONITOR_PID 2>/dev/null
echo "Training completed. Checkpoints: $OUTPUT_DIR"
