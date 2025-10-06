#!/bin/bash
#SBATCH --job-name=test-single-gpu
#SBATCH --account=fc_oscnn
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:A40:1
#SBATCH --time=00:30:00
#SBATCH --output=test_single_gpu_%j.out
#SBATCH --error=test_single_gpu_%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Test basic functionality
echo "Testing basic Python and GPU access..."

# Load modules
module load gcc/11.3.0 || module load gcc
module load cuda/11.8 || module load cuda
module load python/3.9 || module load python

# Test GPU
nvidia-smi

# Test Python and PyTorch
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('✅ GPU test successful')
else:
    print('❌ No GPU available')
"

echo "✅ Basic test completed at: $(date)"
