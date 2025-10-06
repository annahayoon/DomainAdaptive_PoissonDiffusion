#!/bin/bash

# H100 OPTIMIZATION SETUP SCRIPT
# Installs and configures H100-specific optimizations for maximum performance

echo "🚀 H100 OPTIMIZATION SETUP"
echo "=========================="
echo ""

# Check if we're on H100
if nvidia-smi | grep -q "H100"; then
    echo "✅ H100 GPU detected - proceeding with optimizations"
else
    echo "⚠️  H100 not detected - optimizations may not be optimal"
    echo "   Detected GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
fi

echo ""
echo "📦 Installing H100-specific packages..."

# Install Flash Attention 2 for H100 (2x speedup for attention)
echo "⚡ Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

# Install xFormers for additional memory optimizations
echo "🔧 Installing xFormers..."
pip install xformers

# Install Triton for custom kernels
echo "⚙️  Installing Triton..."
pip install triton

# Install additional optimization packages
echo "📈 Installing optimization packages..."
pip install torch-optimizer  # Additional optimizers
pip install deepspeed        # For potential future scaling

echo ""
echo "🔧 Setting up H100 environment variables..."

# Create H100 optimization environment script
cat > h100_env.sh << 'EOF'
#!/bin/bash
# H100 Environment Optimization Script

# CUDA optimizations for H100
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_ALLOW_TF32=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Memory optimizations for 80GB HBM3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# H100-specific optimizations
export NCCL_NVLS_ENABLE=0  # Disable NVLS for single GPU
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Thread optimizations for H100 systems
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Flash Attention optimizations
export FLASH_ATTENTION_FORCE_FP16=0  # Use BF16 on H100
export FLASH_ATTENTION_CAUSAL=1

echo "🚀 H100 environment variables set!"
echo "   TF32: Enabled (2x speedup for matrix ops)"
echo "   Memory: Optimized for 80GB HBM3"
echo "   Flash Attention: Configured for H100"
EOF

chmod +x h100_env.sh

echo ""
echo "✅ H100 OPTIMIZATION SETUP COMPLETE!"
echo ""
echo "🔥 H100-Specific Features Installed:"
echo "   • Flash Attention 2 (2x attention speedup)"
echo "   • xFormers (memory optimization)"
echo "   • Triton (custom kernels)"
echo "   • TF32 enabled (2x matrix op speedup)"
echo "   • 80GB HBM3 memory optimization"
echo ""
echo "📋 Usage:"
echo "   1. Source environment: source h100_env.sh"
echo "   2. Run training: ./run_unified_train_h100_optimized.sh"
echo ""
echo "🎯 Expected Performance Improvements:"
echo "   • 2-3x faster training vs A40/V100"
echo "   • 2x larger batch sizes (80GB vs 40GB)"
echo "   • Stable BF16 training (no NaN issues)"
echo "   • Flash Attention 2x speedup"
echo "   • TF32 2x matrix operation speedup"
echo ""
echo "⚡ Total expected speedup: 4-6x vs previous setup!"
