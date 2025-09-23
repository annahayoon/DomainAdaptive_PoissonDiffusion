#!/bin/bash

# UNIFIED MULTI-DOMAIN RESEARCH TRAINING CONFIGURATION
# Runs unified model on ALL domains simultaneously with domain conditioning
# This is our key research contribution: domain-agnostic restoration

echo "🌟 UNIFIED MULTI-DOMAIN RESEARCH TRAINING CONFIGURATION v2.0"
echo "=================================================================="
echo ""
echo "Strategy: Train ONE model for ALL domains (photography, microscopy, astronomy)"
echo "✅ UNIFIED: Single model with domain conditioning vectors"
echo "✅ BALANCED: Weighted sampling across all three domains"
echo "✅ RESEARCH: Proves domain transfer learning works"
echo ""

# Kill existing sessions
tmux kill-session -t unified_research_training 2>/dev/null

# Create new session
tmux new-session -d -s unified_research_training -c /home/jilab/Jae

# Phase detection - check if we have a stable checkpoint
STABLE_CHECKPOINT=""
if [ -f "results/unified_training/checkpoints/best_model.pth" ]; then
    STABLE_CHECKPOINT="--resume_checkpoint results/unified_training/checkpoints/best_model.pth"
    MIXED_PRECISION_FLAG="true"
    LEARNING_RATE="1e-5"  # Lower rate for fine-tuning
    echo "📊 Phase 2: Stable unified checkpoint found, enabling optimizations"
elif [ -f "results/unified_training/checkpoints/latest.pth" ]; then
    STABLE_CHECKPOINT="--resume_checkpoint results/unified_training/checkpoints/latest.pth"
    MIXED_PRECISION_FLAG="true"
    LEARNING_RATE="2e-5"
    echo "📊 Phase 2: Latest checkpoint found, resuming training"
else
    MIXED_PRECISION_FLAG="false"  # Start with FP32 for multi-domain stability
    LEARNING_RATE="3e-5"  # Slightly higher for initial training
    echo "📊 Phase 1: Initial unified training, prioritizing stability (FP32)"
fi

# Configuration summary - UNIFIED MULTI-DOMAIN
echo "🌟 UNIFIED MULTI-DOMAIN CONFIGURATION:"
echo "  Model Size: ~400M parameters (unified research architecture)"
echo "  Architecture: 256ch, 6 blocks, all domains (unified)"
echo "  Training: 300K steps (step-based, unified)"
echo "  Batch Strategy: 2 physical × 4 accumulation = 8 effective"
echo "  Domain Conditioning: ENABLED (research contribution)"
echo "  Balanced Sampling: ENABLED (fair domain representation)"
echo "  Learning Rate: $LEARNING_RATE (conservative for multi-domain)"
echo "  Mixed Precision: $MIXED_PRECISION_FLAG"

# Set environment variables for multi-domain training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Launch unified multi-domain training using our new script
tmux send-keys -t unified_research_training "
cd /home/jilab/Jae && python train_unified_model.py \\
    --data_root data \\
    --domains photography microscopy astronomy \\
    --max_steps 300000 \\
    --batch_size 2 \\
    --gradient_accumulation_steps 4 \\
    --learning_rate $LEARNING_RATE \\
    --model_channels 256 \\
    --channel_mult 1 2 3 4 \\
    --channel_mult_emb 6 \\
    --num_blocks 6 \\
    --attn_resolutions 16 32 64 \\
    --output_dir results/unified_training \\
    --device cuda \\
    --mixed_precision $MIXED_PRECISION_FLAG \\
    --gradient_checkpointing false \\
    --seed 42 \\
    --val_frequency 3 \\
    --warmup_steps 5000 \\
    --lr_scheduler cosine \\
    --gradient_clip_norm 0.1 \\
    $STABLE_CHECKPOINT
" C-m

echo ""
echo "🚀 UNIFIED MULTI-DOMAIN TRAINING STARTED!"
echo ""
echo "📊 Configuration:"
echo "  • Model: 256-channel unified architecture"
echo "  • Training: 300K steps (comprehensive multi-domain)"
echo "  • Batch Size: 2 (memory-efficient for multi-domain)"
echo "  • Effective Batch: 8 (2×4 accumulation)"
echo "  • Domain Conditioning: ENABLED (research key)"
echo "  • Balanced Sampling: ENABLED (fair representation)"
echo "  • Learning Rate: $LEARNING_RATE (conservative for stability)"
echo ""
echo "📈 Research Features:"
echo "  • Single model learns all 3 domains simultaneously"
echo "  • Domain vectors condition the model per domain"
echo "  • Balanced sampling ensures fair domain representation"
echo "  • Physics-aware loss optimized for all domains"
echo ""
echo "📱 Monitor with:"
echo "  tmux attach -t unified_research_training"
echo ""
echo "🎯 This unified model demonstrates the research hypothesis:"
echo "   A single physics-aware diffusion model can learn domain transfer"
echo "   across photography, microscopy, and astronomy with proper conditioning."
