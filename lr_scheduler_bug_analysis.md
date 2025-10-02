# Learning Rate Scheduler Bug Analysis

## 🚨 **CRITICAL BUG FOUND**

### **Configuration Claims vs. Reality**

**Configuration says:**
```python
# Line 510 & 547 in train_unified_prior_clean.py
"lr_scheduler": config_kwargs.get("lr_scheduler", "cosine"),
"warmup_steps": config_kwargs.get("warmup_steps", 20000),
```

**But the training loop NEVER implements it:**
```python
# Lines 700-704: Only optimizer creation
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=1e-3
)

# Lines 735-767: Training step - NO scheduler.step() call!
optimizer.zero_grad()
# ... forward pass ...
optimizer.step()  # ❌ No scheduler!
```

## 🎯 **This Explains Everything!**

### Training Trajectory Analysis:
```
Steps 0-25K:   Loss 6.0 → 0.06  ✅ Learning rate 1e-4 works initially
Steps 25K-90K: Loss 0.06 → 6.74 ❌ LR stays 1e-4, too high, causes divergence
```

**What should have happened with cosine scheduler:**
```
Steps 0-20K:   Warmup: 0 → 1e-4      (gradual increase)
Steps 20K-450K: Cosine: 1e-4 → ~1e-6  (gradual decrease)
```

**What actually happened:**
```
Steps 0-450K: Fixed LR: 1e-4  (way too high for later training)
```

## 🔧 **Fix Required**

### Option A: Resume from 25K with Proper Scheduler
```python
# Add after optimizer creation (line 704):
if config["lr_scheduler"] == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config["max_steps"] - config["warmup_steps"],
        eta_min=config["learning_rate"] * 0.01  # 1% of initial LR
    )
    
    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,  # Start at 1% of LR
        total_iters=config["warmup_steps"]
    )

# Add in training loop (after optimizer.step()):
if step < config["warmup_steps"]:
    warmup_scheduler.step()
else:
    scheduler.step()
```

### Option B: Simple Fix - Lower Learning Rate
```bash
# Resume from 25K checkpoint with:
LEARNING_RATE="1e-5"  # 10x smaller than current
```

## 📊 **Why 25K Model is Better**

The 25K checkpoint represents the **optimal point** where:
1. Model learned successfully with 1e-4 LR (good for early training)
2. Before LR became too high for fine-tuning (bad for later training)

## 🎯 **Recommendation**

**YES, absolutely resume from 25K checkpoint** with either:

### Quick Fix (Recommended):
```bash
# Resume training with much lower LR
python train_unified_prior_clean.py \
    --resume_checkpoint checkpoint_step_025000.pth \
    --learning_rate 1e-5 \  # 10x smaller
    --max_steps 200000      # Continue to 200K total
```

### Proper Fix:
1. Fix the scheduler implementation in the training script
2. Resume from 25K with proper cosine annealing
3. This will give optimal convergence

## 💡 **Key Insights**

1. **The training script has a scheduler config but doesn't use it**
2. **Fixed high LR (1e-4) caused divergence after 25K steps**
3. **25K checkpoint is at peak performance before divergence**
4. **With proper scheduling, could train to 200K+ steps successfully**

The 25K model isn't just "better trained" - it's at the **sweet spot** before the missing scheduler caused catastrophic forgetting!
