# 🧪 Poisson-Gaussian Training Setup Guide

## 🎯 **Comprehensive Training Setup for Photography Data**

### **📊 Training Configuration Summary**

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | EDM with 128 channels | Optimized for 128x128 images |
| **Training Epochs** | 100 | With early stopping (patience=20) |
| **Batch Size** | 4 | Optimized for A40 GPU memory |
| **Learning Rate** | 1e-4 | With cosine annealing scheduler |
| **Early Stopping** | ✅ Enabled | Stops if no improvement for 20 epochs |
| **Mixed Precision** | ✅ Enabled | Faster training, lower memory usage |
| **Multi-GPU Support** | ⚠️ Not yet | Single GPU optimized |

### **⏱️ Training Time Estimates**

#### **Single A40 GPU (Your Current Setup)**
- **Estimated Time**: 12-24 hours for 100 epochs
- **Memory Usage**: ~8-12 GB GPU memory
- **Early Stopping**: May complete in 6-12 hours if converges quickly

#### **4 A40 GPUs (HPC Cluster)**
- **Estimated Time**: 4-8 hours for 100 epochs
- **Speedup**: ~3-4x faster than single GPU
- **Multi-GPU Code**: ⚠️ Would need modifications for distributed training

### **📈 Metrics to Monitor During Training**

#### **Primary Metrics (Critical for Success)**
1. **Validation Loss** 📉 - Should decrease steadily
   - Target: < 0.01 (excellent), < 0.1 (good), < 0.5 (acceptable)
   - Monitor for: Sudden increases, NaN values, or plateauing

2. **Training Loss** 📉 - Should decrease consistently
   - Target: Similar trend to validation loss
   - Monitor for: Overfitting (training loss much lower than validation)

3. **Early Stopping Counter** ⏱️
   - Should stay low (< 10) for healthy training
   - Monitor for: Rapid increases indicating convergence issues

#### **Secondary Metrics (Important for Performance)**
1. **GPU Memory Usage** 🖥️
   - Should stay < 90% (threshold alert at 90%)
   - Monitor for: Memory leaks, increasing usage over time

2. **Domain Distribution** 📊
   - Photography should be 100% (single domain training)
   - Monitor for: Imbalance issues

3. **Learning Rate** 📈
   - Should follow cosine annealing schedule
   - Monitor for: Sudden drops or stalls

#### **Physics-Specific Metrics** 🔬
1. **χ² Consistency** - Should be ~1.0
   - Values > 2.0 or < 0.5 indicate problems
2. **PSNR** - Should increase over time
   - Target: > 25 dB (good), > 30 dB (excellent)
3. **SSIM** - Should approach 0.9+
   - Values < 0.7 indicate poor restoration quality

### **🚨 Alert Conditions to Watch For**

#### **Critical Alerts (Stop Training)**
- **NaN/Inf Losses**: Training diverged
- **GPU Memory > 95%**: Risk of OOM errors
- **No Progress > 2 hours**: Training stuck

#### **Warning Alerts (Monitor Closely)**
- **GPU Memory > 90%**: Consider reducing batch size
- **Early Stopping Counter > 15**: Model may be struggling
- **Domain Imbalance > 50%**: Data loading issues

### **🪟 Tmux Session Setup**

#### **Quick Start (Recommended)**
```bash
# Setup training session with photography data
./setup_training_session.sh /home/jilab/Jae/data/domain_dataset_test/test_data/photography

# OR use the restart script (fixes common issues)
./restart_training.sh

# Attach to session
tmux attach -t poisson_training
```

#### **Session Layout**
```
┌─────────────────────────────────────────────────────────┐
│ Training Script (Top Left)  │ TensorBoard (Top Right)   │
│                             │ http://localhost:6006     │
├─────────────────────────────┴───────────────────────────┤
│ System Monitor (Bottom)                                 │
│ GPU: 45.2% | CPU: 23.1% | RAM: 67.8%                  │
└─────────────────────────────────────────────────────────┘
```

#### **Session Management**
```bash
# Attach to session
tmux attach -t poisson_training

# Detach (keep running)
Ctrl+B, D

# Kill session
tmux kill-session -t poisson_training

# List sessions
tmux ls
```

### **📊 Real-Time Monitoring**

#### **Option 1: Tmux Session (Recommended)**
```bash
# Attach to see live updates
tmux attach -t poisson_training
```

#### **Option 2: Dedicated Monitor**
```bash
# In a separate terminal
python monitor_training.py --log_file logs/training_photography.log --interval 30
```

#### **Option 3: TensorBoard**
```bash
# Access at http://localhost:6006
tensorboard --logdir results/photography_training_*/tensorboard
```

### **💾 Output Structure**

```
results/photography_training_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best_checkpoint.pt          # Best model
│   ├── checkpoint_epoch_005.pt     # Every 5 epochs
│   └── checkpoint_epoch_010.pt
├── logs/
│   ├── training_photography.log    # Main training log
│   └── tensorboard/                # TensorBoard logs
├── monitoring/
│   ├── monitor_training.py         # Monitoring script
│   └── training_results.json       # Final results
└── training_config.json            # Training configuration
```

### **⚡ Advanced Configuration Options**

#### **Performance Optimization**
```bash
# High-performance configuration for A40 GPU
python train_photography_model.py \
    --data_root /path/to/data \
    --epochs 100 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --mixed_precision \
    --num_workers 8 \
    --pin_memory \
    --device cuda \
    --model_channels 128
```

#### **Memory-Efficient Configuration**
```bash
# Lower memory usage for testing
python train_photography_model.py \
    --data_root /path/to/data \
    --epochs 50 \
    --batch_size 4 \
    --mixed_precision \
    --num_workers 4 \
    --device cuda
```

### **🔧 Troubleshooting Common Issues**

#### **1. GPU Memory Issues**
```bash
# Reduce batch size
python train_photography_model.py --batch_size 2 --data_root /path/to/data

# Use CPU (for testing)
python train_photography_model.py --device cpu --data_root /path/to/data
```

#### **2. Training Stuck**
- Check GPU utilization in monitoring pane
- Look for error messages in training log
- Verify data loading isn't causing bottlenecks

#### **3. Poor Convergence**
- Check learning rate scheduling
- Verify data preprocessing quality
- Ensure physics-aware loss is functioning

#### **4. Domain Issues**
- Verify photography data format (.arw, .dng, .nef, .cr2)
- Check calibration file compatibility
- Ensure sufficient data samples

### **🎯 Success Indicators**

#### **Training Completion**
- ✅ Reaches 100 epochs OR early stopping
- ✅ Best validation loss < 0.1
- ✅ No critical errors in logs
- ✅ Checkpoints saved successfully

#### **Model Quality**
- ✅ PSNR > 25 dB on validation set
- ✅ SSIM > 0.8 on validation set
- ✅ χ² consistency ~ 1.0
- ✅ Physics metrics within acceptable ranges

### **🚀 Next Steps After Training**

#### **1. Model Evaluation**
```bash
# Evaluate trained model
python scripts/evaluate_baselines.py --model_path results/photography_training_*/checkpoints/best_checkpoint.pt

# Test on new data
python scripts/test_domain_dataset_integration.py
```

#### **2. Multi-Domain Extension**
```bash
# Add microscopy and astronomy domains
python train_photography_model.py --multi_domain --domains photography,microscopy,astronomy
```

#### **3. HPC Cluster Setup (4 GPUs)**
```bash
# Would require modifications for multi-GPU training
# Expected speedup: 3-4x faster training
# Contact for distributed training setup
```

### **📞 Support and Monitoring**

#### **Real-Time Monitoring**
- **Primary**: Tmux session with live updates
- **Secondary**: TensorBoard web interface
- **Backup**: Dedicated monitoring script

#### **Log Analysis**
```bash
# Check training progress
tail -f logs/training_photography.log

# Search for errors
grep -i "error\|exception\|failed" logs/training_photography.log

# Monitor loss trends
grep -E "val_loss|training_loss" logs/training_photography.log
```

#### **Contact Information**
- **Training Logs**: `logs/training_photography.log`
- **Output Directory**: `results/photography_training_*/`
- **TensorBoard**: http://localhost:6006
- **System Monitor**: Tmux session monitoring pane

### **🎉 Ready to Train!**

Your training setup is **production-ready** with comprehensive monitoring, early stopping, and physics-aware optimization. The single A40 GPU setup should complete training in 12-24 hours with optimal convergence.

**Start training with:**
```bash
./setup_training_session.sh /home/jilab/Jae/data/domain_dataset_test/test_data/photography
```

**Monitor progress with:**
```bash
tmux attach -t poisson_training
```
