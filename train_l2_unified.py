#!/usr/bin/env python3
"""
L2 Baseline unified training script for proper ablation study.

This script creates a fair L2 baseline by training under conditions optimal for L2 guidance:
- IDENTICAL architecture to Poisson-Gaussian model
- HOMOSCEDASTIC Gaussian noise training (x + N(0, σ²)) instead of heteroscedastic Poisson
- SIMPLE noise-level conditioning instead of physics-aware conditioning  
- Uses same v-parameterization training, differs only in inference guidance

This ensures a rigorous ablation study that compares models optimized for their respective worlds.

Key differences from Poisson-Gaussian training:
- Training noise: Homoscedastic Gaussian instead of Poisson+Gaussian
- Conditioning: Simple noise level instead of physics parameters
- Same v-parameterization training, differs only in inference guidance

Usage:
    python train_l2_unified.py --data_root /opt/dlami/nvme/preprocessed/prior_clean
"""

import argparse
import json
import logging
import os
import sys
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.error_handlers import ErrorHandler
from core.logging_config import LoggingManager
from models.edm_wrapper import create_edm_wrapper, create_progressive_edm
from poisson_training import (
    MultiDomainTrainingConfig,
    set_deterministic_mode,
)

# Setup logging
logging_manager = LoggingManager()
logger = logging_manager.setup_logging(
    level="INFO",
    log_dir="logs",
    console_output=True,
    file_output=True,
    json_format=False,
)


class L2BaselineDataset(Dataset):
    """L2 Baseline dataset with homoscedastic Gaussian noise and simple conditioning."""
    
    def __init__(
        self, 
        data_root: str, 
        domains: List[str] = ["photography", "microscopy", "astronomy"],
        split: str = "train", 
        max_files_per_domain: Optional[int] = None,
        balance_domains: bool = True,
        seed: int = 42,
        noise_level_range: Tuple[float, float] = (0.01, 2.0)  # Homoscedastic noise range matching EDM levels
    ):
        self.data_root = Path(data_root)
        self.domains = domains
        self.split = split
        self.max_files_per_domain = max_files_per_domain
        self.balance_domains = balance_domains
        self.seed = seed
        self.noise_level_range = noise_level_range
        
        # Set random seed for reproducible noise sampling
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Domain info (same architecture requirements)
        self.domain_info = {
            "photography": {"id": 0, "channels": 4},
            "microscopy": {"id": 1, "channels": 1}, 
            "astronomy": {"id": 2, "channels": 1}
        }
        
        # Load files for each domain
        self.files = []
        self.domain_weights = []
        
        for domain in self.domains:
            domain_path = self.data_root / domain / split
            if not domain_path.exists():
                logger.warning(f"Domain {domain} split {split} not found: {domain_path}")
                continue
                
            # Get all files for this domain
            domain_files = list(domain_path.glob("*.pt"))
            domain_files.sort()  # For reproducibility
            
            # Apply max files limit
            if self.max_files_per_domain is not None:
                domain_files = domain_files[:self.max_files_per_domain]
            
            # Add domain info to each file
            for file_path in domain_files:
                self.files.append({
                    'path': file_path,
                    'domain': domain,
                    'domain_id': self.domain_info[domain]['id'],
                    'channels': self.domain_info[domain]['channels']
                })
            
            logger.info(f"  {domain}: {len(domain_files):,} files")
        
        if len(self.files) == 0:
            raise ValueError(f"No files found in {self.data_root}")
        
        # Create sampling weights for balanced training
        if self.balance_domains:
            self._create_balanced_weights()
        
        logger.info(f"Total {split} files: {len(self.files):,}")
        logger.info(f"L2 Baseline training with homoscedastic noise range: {noise_level_range}")
    
    def _create_balanced_weights(self):
        """Create sampling weights for balanced domain training."""
        domain_counts = {}
        for file_info in self.files:
            domain = file_info['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Calculate weights (inverse frequency)
        total_files = len(self.files)
        domain_weights = {}
        for domain, count in domain_counts.items():
            domain_weights[domain] = total_files / (len(self.domains) * count)
        
        # Assign weights to each file
        self.sample_weights = []
        for file_info in self.files:
            domain = file_info['domain']
            self.sample_weights.append(domain_weights[domain])
        
        logger.info("Domain balancing weights:")
        for domain, weight in domain_weights.items():
            count = domain_counts[domain]
            logger.info(f"  {domain}: {count:,} files, weight: {weight:.4f}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load sample with homoscedastic noise and simple conditioning."""
        file_info = self.files[idx]
        file_path = file_info['path']
        domain = file_info['domain']
        domain_id = file_info['domain_id']
        channels = file_info['channels']
        
        # Load the .pt file
        data = torch.load(file_path, map_location='cpu')
        
        # Extract clean image
        clean_norm = data['clean_norm']  # [C, 128, 128]
        metadata = data.get('metadata', {})
        
        # Ensure 4 channels for unified model
        if clean_norm.shape[0] < 4:
            # Pad with zeros to reach 4 channels
            pad_channels = 4 - clean_norm.shape[0]
            padding = torch.zeros(pad_channels, 128, 128)
            clean_norm = torch.cat([clean_norm, padding], dim=0)
        
        # CRITICAL: Simple conditioning for L2 baseline
        # Only domain + noise level (not physics parameters)
        
        # Domain one-hot (3D)
        domain_onehot = torch.zeros(3, dtype=torch.float32)
        domain_onehot[domain_id] = 1.0
        
        # Sample homoscedastic noise level for this sample
        # Use log-uniform distribution to match EDM's log-normal noise schedule
        min_noise, max_noise = self.noise_level_range
        # Sample in log space for better coverage of low noise levels
        log_min, log_max = np.log(min_noise), np.log(max_noise)
        log_noise = np.random.uniform(log_min, log_max)
        noise_level = np.exp(log_noise)
        
        # Noise level conditioning (1D) - log-normalized to [0,1] to match sampling
        log_noise_norm = torch.tensor((log_noise - log_min) / (log_max - log_min), dtype=torch.float32)
        
        # L2 Baseline conditioning: [domain_one_hot(3), noise_level(1), padding(2)] = 6D total
        # We pad to 6D to match EDM wrapper requirements, but only use first 4D meaningfully
        # This is still much simpler than the physics-aware 6D conditioning of Poisson-Gaussian
        condition = torch.cat([
            domain_onehot,                    # 3D: domain one-hot
            log_noise_norm.unsqueeze(0),      # 1D: log-normalized noise level
            torch.zeros(2, dtype=torch.float32)  # 2D: padding (unused, for EDM compatibility)
        ])
        
        return {
            'clean_norm': clean_norm,
            'condition': condition,  # 6D padded (4D meaningful + 2D padding)
            'domain': domain,
            'domain_id': torch.tensor(domain_id, dtype=torch.long),
            'metadata': metadata,
            'noise_level': torch.tensor(noise_level, dtype=torch.float32),  # For training noise generation
        }
    
    def get_balanced_sampler(self) -> Optional[WeightedRandomSampler]:
        """Get balanced sampler for training."""
        if self.balance_domains and hasattr(self, 'sample_weights'):
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.files),
                replacement=True
            )
        return None


class L2BaselineTrainingManager:
    """L2 Baseline training manager optimized for homoscedastic Gaussian noise."""
    
    def __init__(
        self,
        data_root: str,
        domains: List[str] = ["photography", "microscopy", "astronomy"],
        output_dir: str = "results/l2_unified_training",
        device: str = "auto",
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.domains = domains
        self.output_dir = Path(output_dir)
        self.device = self._setup_device(device)
        self.seed = seed
        
        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        
        # Initialize error handler
        self.error_handler = ErrorHandler(
            logger=logger, enable_recovery=True, strict_mode=False
        )
        
        logger.info("🔬 L2 Baseline Training Manager initialized (PROPER ABLATION STUDY)")
        logger.info(f"  Data root: {self.data_root}")
        logger.info(f"  Domains: {', '.join(self.domains)}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Seed: {self.seed}")
        logger.info("  🎯 TRAINING: Homoscedastic Gaussian noise (x + N(0,σ²))")
        logger.info("  🎯 CONDITIONING: Simple noise-level based (4D meaningful + 2D padding)")
        logger.info("  🎯 GUIDANCE: L2 (MSE) optimized for this training regime")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device with H100-specific optimizations."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                
                # H100-specific optimizations
                device_name = torch.cuda.get_device_name()
                is_h100 = "H100" in device_name
                
                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                torch.backends.cuda.enable_flash_sdp(True)
                
                # H100-specific settings
                if is_h100:
                    logger.info("🚀 H100 detected - enabling advanced optimizations")
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                
                logger.info(f"🚀 Using CUDA device: {device_name}")
                logger.info(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                
                if is_h100:
                    logger.info("  🔥 H100 optimizations: TF32, Flash Attention, 80GB HBM3")
                    
            else:
                device = "cpu"
                logger.info("💻 Using CPU device")
        
        return device
    
    def create_datasets(
        self, 
        max_files_per_domain: Optional[int] = None,
        balance_domains: bool = True,
        noise_level_range: Tuple[float, float] = (0.01, 2.0)  # Match EDM noise schedule range
    ) -> Dict[str, L2BaselineDataset]:
        """Create train/val datasets with homoscedastic noise."""
        logger.info("📁 Creating L2 baseline datasets...")
        logger.info(f"  Noise range: {noise_level_range} (homoscedastic Gaussian)")
        
        datasets = {}
        for split in ["train", "val"]:
            try:
                dataset = L2BaselineDataset(
                    data_root=str(self.data_root),
                    domains=self.domains,
                    split=split,
                    max_files_per_domain=max_files_per_domain,
                    balance_domains=balance_domains,
                    seed=self.seed,
                    noise_level_range=noise_level_range
                )
                datasets[split] = dataset
                logger.info(f"  {split}: {len(dataset):,} samples")
            except Exception as e:
                logger.warning(f"  {split}: Failed to load - {e}")
        
        if not datasets:
            raise ValueError("No datasets could be created")
        
        return datasets
    
    def create_model(
        self,
        use_multi_resolution: bool = False,
        mixed_precision: bool = True,
        h100_optimizations: bool = True,
        **model_kwargs
    ) -> nn.Module:
        """Create optimized unified model (identical to Poisson-Gaussian training)."""
        logger.info("🤖 Creating L2 baseline model (identical architecture, simple conditioning)...")
        
        # H100-optimized model configuration (IDENTICAL to Poisson-Gaussian)
        if h100_optimizations and "H100" in torch.cuda.get_device_name():
            logger.info("🔥 Applying H100 optimizations to L2 model")
            # Larger model for H100
            default_channels = model_kwargs.get("model_channels", 320)
            default_blocks = model_kwargs.get("num_blocks", 8)
        else:
            # Standard configuration
            default_channels = model_kwargs.get("model_channels", 256)
            default_blocks = model_kwargs.get("num_blocks", 6)
        
        if use_multi_resolution:
            logger.info("📈 Using Progressive Multi-Resolution EDM for L2")
            
            model = create_progressive_edm(
                min_resolution=32,
                max_resolution=128,
                num_stages=4,
                model_channels=default_channels,
                img_channels=4,  # Unified 4-channel
                label_dim=6,  # EDM requires 6D, but L2 baseline only uses first 4D meaningfully
                use_fp16=False,
                dropout=0.1,
                **{k: v for k, v in model_kwargs.items() 
                   if k not in ["model_channels", "img_channels", "label_dim", "use_fp16", "dropout"]}
            )
        else:
            logger.info("📊 Using Standard EDM for L2")
            
            model_config = {
                "img_resolution": 128,
                "img_channels": 4,  # Unified 4-channel
                "model_channels": default_channels,
                "channel_mult": model_kwargs.get("channel_mult", [1, 2, 3, 4]),
                "channel_mult_emb": model_kwargs.get("channel_mult_emb", 8),
                "num_blocks": default_blocks,
                "attn_resolutions": model_kwargs.get("attn_resolutions", [16, 32, 64]),
                "label_dim": 6,  # EDM requires 6D, but L2 baseline only uses first 4D meaningfully
                "use_fp16": False,
                "dropout": 0.1,
            }
            
            model = create_edm_wrapper(**model_config)
        
        # Move to device and ensure float32
        model = model.to(self.device).float()
        param_count = sum(p.numel() for p in model.parameters())
        
        logger.info("✅ L2 baseline model created successfully")
        logger.info(f"  Parameters: {param_count:,}")
        logger.info(f"  Domains: {', '.join(self.domains)}")
        logger.info(f"  Architecture: {default_channels}ch, {default_blocks} blocks")
        logger.info("  🔬 CONDITIONING: 6D padded (4D meaningful: domain+noise, 2D padding)")
        logger.info("  🔬 TRAINING: Optimized for homoscedastic Gaussian noise")
        logger.info("  🔬 GUIDANCE: L2 (MSE) - Fair comparison with Poisson-Gaussian")

        return model

    def validate_model(self, model: nn.Module, val_loader: DataLoader, device: torch.device, step: int) -> float:
        """Run validation and return average loss."""
        model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                clean = batch['clean_norm'].to(device).float()
                condition = batch['condition'].to(device).float()
                noise_levels = batch['noise_level'].to(device).float()

                # Use same homoscedastic noise as training
                noise = torch.randn_like(clean)
                sigma = noise_levels.view(-1, 1, 1, 1)
                noisy = clean + sigma * noise

                # Forward pass - same v-parameterization as training
                predicted = model(noisy, noise_levels, condition=condition)

                # v-parameterization loss (same as training and Poisson-Gaussian)
                c_skip = 1 / (sigma**2 + 1)
                c_out = sigma / (sigma**2 + 1).sqrt()
                target = (clean - c_skip * noisy) / c_out
                loss = F.mse_loss(predicted, target)

                total_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        model.train()
        return avg_val_loss

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, config: Dict[str, Any], loss: float = None, val_loss: float = None, is_best: bool = False, ema_model: nn.Module = None):
        """Save a checkpoint with comprehensive metadata."""
        checkpoint_info = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'config': config,
            'guidance_type': 'L2',  # Mark as L2 guidance
            'ablation_study': True,
        }
        
        # Add EMA model if available
        if ema_model is not None:
            checkpoint_info['ema_model_state_dict'] = ema_model.state_dict()

        if loss is not None:
            checkpoint_info['train_loss'] = loss
        if val_loss is not None:
            checkpoint_info['val_loss'] = val_loss

        # Regular checkpoint
        checkpoint_path = self.output_dir / f"l2_checkpoint_step_{step:07d}.pth"
        torch.save(checkpoint_info, checkpoint_path)
        logger.info(f"💾 L2 Checkpoint saved: {checkpoint_path}")

        # Best model checkpoint
        if is_best and val_loss is not None:
            best_path = self.output_dir / "l2_best_model.pth"
            torch.save(checkpoint_info, best_path)
            logger.info(f"🏆 L2 Best model updated: {best_path} (val_loss: {val_loss:.6f})")

    def save_phase_checkpoint(self, model: nn.Module, step: int, config: Dict[str, Any]):
        """Save checkpoint at different training phases for L2 unified model."""
        # Phase milestones (in training steps) - ADJUSTED for 225K steps
        phases = {
            50000: "photography_only",
            100000: "photography_microscopy", 
            150000: "all_domains_phase1",
            200000: "all_domains_phase2",
            225000: "final_l2_unified_model"
        }

        # Find the closest phase milestone
        phase_step = None
        phase_name = None
        for milestone in sorted(phases.keys()):
            if step >= milestone:
                phase_step = milestone
                phase_name = phases[milestone]
            else:
                break

        if phase_step is not None:
            phase_checkpoint_path = self.output_dir / f"l2_phase_{phase_name}_step_{phase_step:06d}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': phase_step,
                'phase': phase_name,
                'config': config,
                'guidance_type': 'L2',
                'ablation_study': True,
            }, phase_checkpoint_path)
            logger.info(f"🎯 L2 Phase checkpoint saved: {phase_checkpoint_path}")
            logger.info(f"   Phase: {phase_name} (step {phase_step:,})")

    def create_optimized_config(
        self,
        h100_optimizations: bool = True,
        mixed_precision: bool = True,
        **config_kwargs
    ) -> Dict[str, Any]:
        """Create H100-optimized L2 training configuration (IDENTICAL to Poisson-Gaussian)."""
        
        is_h100 = "H100" in torch.cuda.get_device_name()
        
        if h100_optimizations and is_h100:
            logger.info("🔥 Creating H100-optimized L2 configuration")

            # H100-optimized settings (IDENTICAL to Poisson-Gaussian)
            config = {
                # Training scale - identical to Poisson-Gaussian for fair comparison
                "max_steps": config_kwargs.get("max_steps", 225000),
                "batch_size": config_kwargs.get("batch_size", 32),
                "gradient_accumulation_steps": config_kwargs.get("gradient_accumulation_steps", 2),
                "learning_rate": config_kwargs.get("learning_rate", 1e-4),

                # Model architecture - IDENTICAL
                "model_channels": config_kwargs.get("model_channels", 320),
                "num_blocks": config_kwargs.get("num_blocks", 8),
                "channel_mult_emb": config_kwargs.get("channel_mult_emb", 8),

                # Optimization - IDENTICAL
                "mixed_precision": mixed_precision,
                "precision": "bf16" if mixed_precision else "fp32",
                "gradient_clip_norm": config_kwargs.get("gradient_clip_norm", 1.0),

                # Scheduling - IDENTICAL
                "warmup_steps": config_kwargs.get("warmup_steps", 20000),
                "lr_scheduler": config_kwargs.get("lr_scheduler", "cosine"),

                # Validation and saving - IDENTICAL
                "val_frequency": config_kwargs.get("val_frequency", 5000),
                "save_frequency_steps": config_kwargs.get("save_frequency_steps", 5000),
                "phase_save_frequency": config_kwargs.get("phase_save_frequency", 25000),

                # Data loading - IDENTICAL
                "num_workers": config_kwargs.get("num_workers", 8),
                "pin_memory": True,
                "persistent_workers": True,
                
                # EMA for better inference
                "ema_decay": config_kwargs.get("ema_decay", 0.999),
                
                # L2-specific
                "guidance_type": "L2",
            }
            
            logger.info("🔥 H100 L2 Configuration:")
            logger.info(f"  Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['gradient_accumulation_steps']})")
            logger.info(f"  Learning rate: {config['learning_rate']}")
            logger.info(f"  Model: {config['model_channels']}ch, {config['num_blocks']} blocks")
            logger.info(f"  Precision: {config['precision']}")
            logger.info(f"  🔬 Guidance: L2 (MSE) - ABLATION STUDY")
            
        else:
            logger.info("📊 Creating standard L2 configuration")
            
            # Conservative settings - IDENTICAL to Poisson-Gaussian
            config = {
                "max_steps": config_kwargs.get("max_steps", 225000),
                "batch_size": config_kwargs.get("batch_size", 16),
                "gradient_accumulation_steps": config_kwargs.get("gradient_accumulation_steps", 4),
                "learning_rate": config_kwargs.get("learning_rate", 5e-5),
                "model_channels": config_kwargs.get("model_channels", 256),
                "num_blocks": config_kwargs.get("num_blocks", 6),
                "mixed_precision": False,
                "precision": "fp32",
                "gradient_clip_norm": config_kwargs.get("gradient_clip_norm", 0.5),
                "warmup_steps": config_kwargs.get("warmup_steps", 10000),
                "lr_scheduler": config_kwargs.get("lr_scheduler", "cosine"),
                "val_frequency": config_kwargs.get("val_frequency", 5000),
                "save_frequency_steps": config_kwargs.get("save_frequency_steps", 5000),
                "phase_save_frequency": config_kwargs.get("phase_save_frequency", 25000),
                "num_workers": config_kwargs.get("num_workers", 4),
                "pin_memory": True,
                "persistent_workers": False,
                "ema_decay": config_kwargs.get("ema_decay", 0.999),
                "guidance_type": "L2",
            }
        
        # Override with any provided kwargs
        config.update(config_kwargs)
        
        return config


def main():
    """Main L2 training function for prior_clean dataset (ablation study)."""
    parser = argparse.ArgumentParser(
        description="Train L2-guided unified model on prior_clean dataset (ablation study)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to prior_clean dataset root",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["photography", "microscopy", "astronomy"],
        help="Domains to train on",
    )
    
    # Training arguments
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation")
    
    # Model arguments
    parser.add_argument("--model_channels", type=int, default=None, help="Model channels")
    parser.add_argument("--num_blocks", type=int, default=None, help="Number of blocks")
    parser.add_argument("--channel_mult_emb", type=int, default=None, help="Channel multiplier for embeddings")
    parser.add_argument("--multi_resolution", action="store_true", help="Use multi-resolution")
    
    # Optimization arguments
    parser.add_argument("--h100_optimizations", action="store_true", default=True, help="Enable H100 optimizations")
    parser.add_argument("--mixed_precision", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--conservative", action="store_true", help="Use conservative settings")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/l2_unified_training")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    
    # Performance arguments
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)
    
    # Testing arguments
    parser.add_argument("--max_files_per_domain", type=int, default=None)
    parser.add_argument("--quick_test", action="store_true", help="Quick test mode")
    parser.add_argument("--val_frequency", type=int, default=5000, help="Run validation every N steps")
    parser.add_argument("--save_frequency_steps", type=int, default=5000, help="Save regular checkpoints every N steps")
    parser.add_argument("--phase_save_frequency", type=int, default=25000, help="Save phase checkpoints every N steps")
    
    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.max_files_per_domain = 100
        args.max_steps = 1000
        args.batch_size = 8
        args.conservative = True
        logger.info("🧪 L2 Quick test mode enabled")
    
    # Conservative mode
    if args.conservative:
        args.h100_optimizations = False
        args.mixed_precision = False
        logger.info("🛡️ L2 Conservative mode enabled")
    
    # Initialize training manager
    logger.info("🔬 INITIALIZING L2 UNIFIED TRAINING (ABLATION STUDY)")
    logger.info("=" * 70)
    logger.info("🎯 PURPOSE: Compare L2 (MSE) vs Poisson-Gaussian guidance")
    logger.info("🎯 METHOD: Identical architecture, identical training, different guidance")
    logger.info("=" * 70)
    
    training_manager = L2BaselineTrainingManager(
        data_root=args.data_root,
        domains=args.domains,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    
    # Create datasets
    datasets = training_manager.create_datasets(
        max_files_per_domain=args.max_files_per_domain,
        balance_domains=True,
    )
    
    # Create optimized configuration
    config_kwargs = {k: v for k, v in vars(args).items() if v is not None and k not in ['h100_optimizations', 'mixed_precision']}
    config = training_manager.create_optimized_config(
        h100_optimizations=args.h100_optimizations,
        mixed_precision=args.mixed_precision,
        **config_kwargs
    )
    
    # Create model
    model = training_manager.create_model(
        use_multi_resolution=args.multi_resolution,
        mixed_precision=config["mixed_precision"],
        h100_optimizations=args.h100_optimizations,
        model_channels=config["model_channels"],
        num_blocks=config["num_blocks"],
    )
    
    # Create data loaders
    train_sampler = datasets["train"].get_balanced_sampler()
    
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config.get("persistent_workers", False),
        drop_last=True,
    )
    
    val_loader = None
    if "val" in datasets:
        val_loader = DataLoader(
            datasets["val"],
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            drop_last=False,
        )
    
    logger.info("🚀 Starting L2 baseline training (PROPER ABLATION STUDY)...")
    logger.info(f"  Total training samples: {len(datasets['train']):,}")
    logger.info(f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    logger.info(f"  Steps per epoch: {len(train_loader) // config['gradient_accumulation_steps']:,}")
    logger.info(f"  Max steps: {config['max_steps']:,}")
    logger.info("  🔬 TRAINING: Homoscedastic Gaussian noise (x + N(0,σ²))")
    logger.info("  🔬 CONDITIONING: 6D padded (4D meaningful: domain+noise, 2D padding)")
    logger.info("  🔬 OBJECTIVE: v-parameterization (identical to Poisson-Gaussian)")
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-3
    )

    # EMA model for better inference performance
    ema_model = None
    ema_decay = config.get("ema_decay", 0.999)
    if ema_decay > 0:
        import copy
        ema_model = copy.deepcopy(model).eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        logger.info(f"✅ EMA model created with decay: {ema_decay}")

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if config["mixed_precision"] else None

    # Resume from checkpoint if provided
    start_step = 0
    if args.resume_checkpoint:
        logger.info(f"🔄 Loading checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=training_manager.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step']
        
        # Load EMA model if available
        if ema_model is not None and 'ema_model_state_dict' in checkpoint:
            ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            logger.info("✅ EMA model state loaded")
        
        logger.info(f"✅ Resumed from step {start_step:,}")
        
        # Load best validation loss if available
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
            logger.info(f"📊 Resumed with best val loss: {best_val_loss:.6f}")
        else:
            best_val_loss = float('inf')
    else:
        best_val_loss = float('inf')

    # Best model tracking
    step = start_step
    model.train()
    
    try:
        while step < config["max_steps"]:
            for batch in train_loader:
                if step >= config["max_steps"]:
                    break

                # Move to device and ensure float32
                clean = batch['clean_norm'].to(training_manager.device).float()
                condition = batch['condition'].to(training_manager.device).float()

                # Training step
                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=config["mixed_precision"], dtype=torch.bfloat16 if config.get("precision") == "bf16" else torch.float16):
                    # L2 BASELINE: Homoscedastic Gaussian noise training
                    # This is DIFFERENT from Poisson-Gaussian to create fair comparison
                    
                    # Use the noise levels from dataset for homoscedastic training
                    noise_levels = batch['noise_level'].to(training_manager.device).float()  # [B]
                    
                    # Generate Gaussian noise
                    noise = torch.randn_like(clean)
                    
                    # Apply homoscedastic noise: y = x + N(0, σ²)
                    # Expand noise levels to match image dimensions
                    sigma = noise_levels.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
                    noisy = clean + sigma * noise
                    
                    # For L2 baseline, we use the same v-parameterization as Poisson-Gaussian
                    # but with simpler homoscedastic noise - this ensures fair comparison
                    predicted = model(noisy, noise_levels, condition=condition)
                    
                    # Use same v-parameterization as Poisson-Gaussian model for fair comparison
                    # This ensures identical model behavior, differing only in guidance during inference
                    c_skip = 1 / (sigma**2 + 1)
                    c_out = sigma / (sigma**2 + 1).sqrt()
                    target = (clean - c_skip * noisy) / c_out
                    
                    # MSE loss in v-space (same as Poisson-Gaussian)
                    loss = F.mse_loss(predicted, target)

                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update EMA model
                if ema_model is not None:
                    with torch.no_grad():
                        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                            ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

                # Logging with debugging info
                if step % 100 == 0:
                    # Debug information
                    noise_mean = noise_levels.mean().item()
                    noise_std = noise_levels.std().item()
                    clean_mean = clean.mean().item()
                    clean_std = clean.std().item()
                    noisy_mean = noisy.mean().item()
                    pred_mean = predicted.mean().item()
                    
                    logger.info(f"L2 Baseline Step {step:,}: Loss = {loss.item():.6f}")
                    logger.info(f"  Debug - Noise σ: {noise_mean:.4f}±{noise_std:.4f}, Clean: {clean_mean:.4f}±{clean_std:.4f}")
                    logger.info(f"  Debug - Noisy: {noisy_mean:.4f}, Predicted: {pred_mean:.4f}")

                # Validation every N steps
                if step % config["val_frequency"] == 0 and step > 0 and val_loader is not None:
                    logger.info(f"🔍 Running L2 baseline validation at step {step:,}...")
                    val_loss = training_manager.validate_model(model, val_loader, training_manager.device, step)
                    logger.info(f"   L2 Baseline Validation Loss: {val_loss:.6f}")

                    # Track best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        training_manager.save_checkpoint(model, optimizer, step, config, val_loss=val_loss, is_best=True, ema_model=ema_model)
                        logger.info(f"   🏆 New L2 best model! Previous: {best_val_loss:.6f} → Current: {val_loss:.6f}")
                    else:
                        logger.info(f"   No improvement (best: {best_val_loss:.6f})")

                # Regular checkpointing every N steps
                if step % config["save_frequency_steps"] == 0 and step > 0:
                    training_manager.save_checkpoint(model, optimizer, step, config, loss.item(), ema_model=ema_model)

                # Phase-based checkpointing for unified training
                if step % config.get("phase_save_frequency", 25000) == 0 and step > 0:
                    training_manager.save_phase_checkpoint(model, step, config)

                step += 1
        
        logger.info("✅ L2 Baseline Training completed!")
        
        # Save final model
        final_path = training_manager.output_dir / "l2_baseline_final_model.pth"
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'step': step,
            'config': config,
            'guidance_type': 'L2_baseline',
            'training_type': 'homoscedastic_gaussian',
            'conditioning_dim': 4,
            'ablation_study': True,
        }
        
        # Add EMA model to final checkpoint
        if ema_model is not None:
            final_checkpoint['ema_model_state_dict'] = ema_model.state_dict()
            
        torch.save(final_checkpoint, final_path)
        
        logger.info(f"L2 Baseline final model saved: {final_path}")
        logger.info("🎯 L2 BASELINE TRAINING COMPLETE - Fair comparison ready!")
        
    except KeyboardInterrupt:
        logger.info("L2 Training interrupted by user")
        
        # Save interrupted checkpoint
        interrupt_path = training_manager.output_dir / "l2_interrupted_checkpoint.pth"
        interrupt_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'config': config,
            'guidance_type': 'L2',
            'ablation_study': True,
        }
        
        # Add EMA model to interrupted checkpoint
        if ema_model is not None:
            interrupt_checkpoint['ema_model_state_dict'] = ema_model.state_dict()
            
        torch.save(interrupt_checkpoint, interrupt_path)
        
        logger.info(f"L2 Interrupted checkpoint saved: {interrupt_path}")


if __name__ == "__main__":
    main()
