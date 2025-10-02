"""
Training Configuration for Imbalanced Cross-Domain Dataset

This configuration provides specific settings for handling your severely imbalanced dataset:
- Photography: 78.9% (298K tiles)
- Microscopy: 15.9% (60K tiles)  
- Astronomy: 5.2% (20K tiles)
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import torch.nn as nn

@dataclass
class ImbalancedTrainingConfig:
    """Configuration for training with severe class imbalance."""
    
    # Sampling Strategy
    sampling_strategy: str = "progressive_reweight"  # or "hierarchical", "focal_weighted"
    reweight_power: float = 0.7  # Between 0.5 (sqrt) and 1.0 (inverse frequency)
    min_domain_ratio: float = 0.15  # Minimum 15% for smallest domain per batch
    
    # Multi-stage Training
    enable_multi_stage: bool = True
    stage_1_epochs: int = 20  # Domain-specific pre-training
    stage_2_epochs: int = 15  # Gradual mixing
    stage_3_epochs: int = 50  # Full multi-domain
    
    # Domain-specific Augmentation
    augmentation_factors: Dict[str, float] = None  # Will be set in __post_init__
    
    # Loss Function Weighting
    domain_loss_weights: Dict[str, float] = None  # Will be set in __post_init__
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Adaptive Reweighting
    enable_adaptive_reweighting: bool = True
    adaptation_frequency: int = 500  # Every N batches
    adaptation_strength: float = 0.1
    
    # Evaluation Strategy
    stratified_validation: bool = True
    per_domain_metrics: bool = True
    macro_averaging: bool = True  # vs micro-averaging
    
    # Early Stopping per Domain
    domain_patience: Dict[str, int] = None  # Will be set in __post_init__
    min_improvement_threshold: float = 0.001
    
    def __post_init__(self):
        """Set default values that depend on domain knowledge."""
        if self.augmentation_factors is None:
            self.augmentation_factors = {
                'photography': 1.0,    # No extra augmentation (already large)
                'microscopy': 3.0,     # 3x augmentation
                'astronomy': 8.0,      # 8x augmentation (most aggressive)
            }
        
        if self.domain_loss_weights is None:
            # Inverse frequency weights with some smoothing
            total_samples = 298000 + 60000 + 20000  # 378K total
            self.domain_loss_weights = {
                'photography': total_samples / (3 * 298000),  # ~0.42
                'microscopy': total_samples / (3 * 60000),    # ~2.1
                'astronomy': total_samples / (3 * 20000),     # ~6.3
            }
            
            # Apply square root smoothing to reduce extreme weights
            import math
            for domain in self.domain_loss_weights:
                self.domain_loss_weights[domain] = math.sqrt(self.domain_loss_weights[domain])
        
        if self.domain_patience is None:
            # More patience for smaller domains (they're harder to optimize)
            self.domain_patience = {
                'photography': 10,
                'microscopy': 15,
                'astronomy': 20,
            }


class ImbalancedLossFunction(nn.Module):
    """Loss function designed for imbalanced cross-domain training."""
    
    def __init__(self, config: ImbalancedTrainingConfig, base_loss: nn.Module):
        super().__init__()
        self.config = config
        self.base_loss = base_loss
        self.domain_weights = config.domain_loss_weights
        
        if config.use_focal_loss:
            self.focal_loss = self._create_focal_loss()
    
    def _create_focal_loss(self):
        """Create focal loss for hard example mining."""
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
            
            def forward(self, pred, target):
                # Compute base loss (e.g., MSE)
                base_loss = nn.functional.mse_loss(pred, target, reduction='none')
                
                # Compute focal weight: (1 - p)^gamma where p is "confidence"
                # For regression, use inverse of normalized loss as confidence proxy
                normalized_loss = base_loss / (base_loss.mean() + 1e-8)
                confidence = 1.0 / (1.0 + normalized_loss)
                focal_weight = (1 - confidence) ** self.gamma
                
                return (self.alpha * focal_weight * base_loss).mean()
        
        return FocalLoss(self.config.focal_alpha, self.config.focal_gamma)
    
    def forward(self, outputs, batch):
        """Compute weighted loss with domain balancing."""
        # Get domain information
        domain_ids = batch.get('domain_id', batch.get('domain'))
        
        if domain_ids is None:
            # Fallback to standard loss
            return self.base_loss(outputs, batch)
        
        # Compute base loss per sample
        if self.config.use_focal_loss:
            sample_losses = self.focal_loss(outputs['prediction'], batch['target'])
        else:
            sample_losses = self.base_loss(outputs, batch)
        
        # Apply domain weights
        if isinstance(domain_ids, torch.Tensor):
            # Convert domain IDs to weights
            domain_weights_tensor = torch.zeros_like(domain_ids, dtype=torch.float)
            for i, domain_id in enumerate(domain_ids):
                if isinstance(domain_id, torch.Tensor):
                    domain_id = domain_id.item()
                
                # Map domain ID to domain name (you'll need to adapt this)
                domain_name = self._id_to_domain_name(domain_id)
                domain_weights_tensor[i] = self.domain_weights.get(domain_name, 1.0)
            
            # Weight the losses
            weighted_losses = sample_losses * domain_weights_tensor
            return weighted_losses.mean()
        
        return sample_losses.mean()
    
    def _id_to_domain_name(self, domain_id):
        """Convert domain ID to domain name. Adapt this to your encoding."""
        id_to_name = {0: 'photography', 1: 'microscopy', 2: 'astronomy'}
        return id_to_name.get(domain_id, 'photography')


class MultiStageTrainingScheduler:
    """Scheduler for multi-stage training with imbalanced domains."""
    
    def __init__(self, config: ImbalancedTrainingConfig):
        self.config = config
        self.current_stage = 1
        self.stage_epochs = {
            1: config.stage_1_epochs,
            2: config.stage_2_epochs, 
            3: config.stage_3_epochs,
        }
        self.total_epochs = sum(self.stage_epochs.values())
    
    def get_current_stage(self, epoch: int) -> int:
        """Get current training stage based on epoch."""
        if epoch < self.stage_epochs[1]:
            return 1
        elif epoch < self.stage_epochs[1] + self.stage_epochs[2]:
            return 2
        else:
            return 3
    
    def get_sampling_config(self, epoch: int) -> Dict:
        """Get sampling configuration for current stage."""
        stage = self.get_current_stage(epoch)
        
        if stage == 1:
            # Stage 1: Domain-specific training (train one domain at a time)
            domain_cycle = epoch % 3
            domain_names = ['photography', 'microscopy', 'astronomy']
            return {
                'strategy': 'single_domain',
                'active_domain': domain_names[domain_cycle],
                'augmentation_factor': self.config.augmentation_factors[domain_names[domain_cycle]]
            }
        
        elif stage == 2:
            # Stage 2: Gradual mixing (start balanced, gradually add photography)
            progress = (epoch - self.stage_epochs[1]) / self.stage_epochs[2]
            photography_weight = 0.33 + 0.4 * progress  # 33% → 73%
            
            return {
                'strategy': 'gradual_mixing',
                'domain_weights': {
                    'photography': photography_weight,
                    'microscopy': (1 - photography_weight) * 0.75,
                    'astronomy': (1 - photography_weight) * 0.25,
                }
            }
        
        else:
            # Stage 3: Full multi-domain with enhanced balancing
            return {
                'strategy': self.config.sampling_strategy,
                'reweight_power': self.config.reweight_power,
                'min_domain_ratio': self.config.min_domain_ratio,
            }
    
    def should_adapt_weights(self, batch_idx: int) -> bool:
        """Check if weights should be adapted at this batch."""
        return (self.config.enable_adaptive_reweighting and 
                batch_idx % self.config.adaptation_frequency == 0)


# Example usage
def create_imbalanced_training_setup():
    """Create complete training setup for imbalanced cross-domain dataset."""
    
    # Configuration
    config = ImbalancedTrainingConfig(
        sampling_strategy="progressive_reweight",
        reweight_power=0.7,
        min_domain_ratio=0.15,
        enable_multi_stage=True,
        use_focal_loss=True,
    )
    
    # Multi-stage scheduler
    scheduler = MultiStageTrainingScheduler(config)
    
    # Enhanced sampler (from previous file)
    from enhanced_domain_sampler import EnhancedDomainSampler
    
    domain_sizes = {
        'photography': 298000,
        'microscopy': 60000,
        'astronomy': 20000,
    }
    
    sampler = EnhancedDomainSampler(
        domain_sizes=domain_sizes,
        batch_size=32,
        strategy=config.sampling_strategy,
        reweight_power=config.reweight_power,
        min_domain_ratio=config.min_domain_ratio,
    )
    
    return config, scheduler, sampler


if __name__ == "__main__":
    config, scheduler, sampler = create_imbalanced_training_setup()
    
    print("Imbalanced Training Configuration:")
    print(f"  Sampling strategy: {config.sampling_strategy}")
    print(f"  Domain loss weights: {config.domain_loss_weights}")
    print(f"  Augmentation factors: {config.augmentation_factors}")
    print(f"  Multi-stage training: {config.enable_multi_stage}")
    
    # Show stage progression
    print("\nTraining Stage Progression:")
    for epoch in [0, 10, 25, 40, 60]:
        stage = scheduler.get_current_stage(epoch)
        sampling_config = scheduler.get_sampling_config(epoch)
        print(f"  Epoch {epoch}: Stage {stage}, Config: {sampling_config}")
