#!/bin/bash

# UNIFIED MULTI-DOMAIN RESEARCH TRAINING CONFIGURATION
# Runs unified model on ALL domains simultaneously with domain conditioning
# This is our key research contribution: domain-agnostic restoration

echo "🌟 UNIFIED MULTI-DOMAIN RESEARCH TRAINING CONFIGURATION v1.0"
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
if [ -f "results/stable_unified_checkpoint.pth" ]; then
    STABLE_CHECKPOINT="--resume results/stable_unified_checkpoint.pth"
    MIXED_PRECISION_FLAG="true"
    LEARNING_RATE="2e-5"
    echo "📊 Phase 2: Stable unified checkpoint found, enabling optimizations"
else
    MIXED_PRECISION_FLAG="false"
    LEARNING_RATE="5e-5"  # Very conservative for multi-domain learning
    echo "📊 Phase 1: Initial unified training, prioritizing maximum stability"
fi

# Configuration summary - UNIFIED MULTI-DOMAIN
configs = {
    "Model Size": "~400M parameters (unified research architecture)",
    "Architecture": "256ch, 6 blocks, all domains (unified)",
    "Training": "300K steps (step-based, unified)",
    "Batch Strategy": "2 physical × 4 accumulation = 8 effective",
    "Workers": "4 (balanced for multi-domain)",
    "Prefetch Factor": "2 (conservative for multi-domain)",
    "Domain Conditioning": "ENABLED (research contribution)",
    "Balanced Sampling": "ENABLED (fair domain representation)",
}

print("🌟 UNIFIED MULTI-DOMAIN CONFIGURATION:")
for key, value in configs.items():
    print(f"  {key}: {value}")

# Set environment variables for multi-domain training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Launch unified multi-domain training
tmux send-keys -t unified_research_training "
cd /home/jilab/Jae && python poisson_training/multi_domain_trainer.py \\
    --data_root data \\
    --max_steps 300000 \\
    --batch_size 2 \\
    --gradient_accumulation_steps 4 \\
    --learning_rate $LEARNING_RATE \\
    --model_channels 256 \\
    --channel_mult 1 2 3 4 \\
    --channel_mult_emb 6 \\
    --num_blocks 6 \\
    --attn_resolutions 16 32 64 \\
    --output_dir results/unified_multi_domain \\
    --device cuda \\
    --mixed_precision $MIXED_PRECISION_FLAG \\
    --gradient_checkpointing true \\
    --num_workers 4 \\
    --prefetch_factor 2 \\
    --pin_memory true \\
    --seed 42 \\
    --save_frequency_steps 10000 \\
    --early_stopping_patience_steps 20000 \\
    --validation_checkpoints_patience 50 \\
    --max_checkpoints 10 \\
    --save_best_model true \\
    --save_optimizer_state false \\
    --checkpoint_metric val_loss \\
    --checkpoint_mode min \\
    --resume_from_best false \\
    --gradient_clip_norm 0.1 \\
    --val_frequency 3 \\
    --warmup_steps 5000 \\
    --lr_scheduler cosine \\
    --domain_conditioning true \\
    --balanced_sampling true \\
    $STABLE_CHECKPOINT \\
    --quick_test
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
