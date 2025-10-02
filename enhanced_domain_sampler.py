"""
Enhanced Domain Sampling Strategy for Severe Class Imbalance

This module provides improved sampling strategies for your cross-domain dataset:
- Photography: 78.9% (298K tiles)
- Microscopy: 15.9% (60K tiles)  
- Astronomy: 5.2% (20K tiles)
"""

import math
import torch
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict

class EnhancedDomainSampler:
    """Enhanced sampling strategy for severely imbalanced cross-domain datasets."""
    
    def __init__(
        self,
        domain_sizes: Dict[str, int],
        batch_size: int,
        strategy: str = "progressive_reweight",
        min_domain_ratio: float = 0.15,  # Minimum 15% for smallest domain
        reweight_power: float = 0.7,     # Between 0.5 (sqrt) and 1.0 (inverse)
    ):
        """
        Initialize enhanced domain sampler.
        
        Args:
            domain_sizes: Dict mapping domain names to sample counts
            batch_size: Training batch size
            strategy: Sampling strategy ('progressive_reweight', 'hierarchical', 'focal_weighted')
            min_domain_ratio: Minimum ratio for smallest domain in each batch
            reweight_power: Power for reweighting (0.5=sqrt, 1.0=inverse frequency)
        """
        self.domain_sizes = domain_sizes
        self.batch_size = batch_size
        self.strategy = strategy
        self.min_domain_ratio = min_domain_ratio
        self.reweight_power = reweight_power
        
        # Calculate domain statistics
        self.total_samples = sum(domain_sizes.values())
        self.domains = list(domain_sizes.keys())
        self.num_domains = len(self.domains)
        
        # Initialize sampling weights
        self.domain_weights = self._compute_enhanced_weights()
        self.sample_weights = self._create_sample_weights()
        
        # Performance tracking for adaptive adjustment
        self.domain_losses = defaultdict(list)
        self.adaptation_history = []
        
        print(f"Enhanced Domain Sampler initialized:")
        print(f"  Strategy: {strategy}")
        print(f"  Domain sizes: {domain_sizes}")
        print(f"  Domain weights: {self.domain_weights}")
        print(f"  Expected samples per batch: {self._expected_samples_per_batch()}")
    
    def _compute_enhanced_weights(self) -> Dict[str, float]:
        """Compute enhanced sampling weights with multiple strategies."""
        
        if self.strategy == "progressive_reweight":
            # Progressive reweighting with configurable power
            weights = {}
            for domain, size in self.domain_sizes.items():
                # Use power scaling instead of pure inverse frequency
                weight = (self.total_samples / size) ** self.reweight_power
                weights[domain] = weight
                
        elif self.strategy == "hierarchical":
            # Ensure minimum representation for each domain
            weights = {}
            min_samples_per_batch = max(1, int(self.batch_size * self.min_domain_ratio))
            
            for domain, size in self.domain_sizes.items():
                # Calculate weight to achieve minimum representation
                natural_prob = size / self.total_samples
                target_prob = max(natural_prob, min_samples_per_batch / self.batch_size)
                weights[domain] = target_prob / natural_prob
                
        elif self.strategy == "focal_weighted":
            # Focal loss inspired weighting - focus on hard (rare) examples
            weights = {}
            for domain, size in self.domain_sizes.items():
                # Focal-style weighting: (1 - p)^gamma where p is natural probability
                natural_prob = size / self.total_samples
                focal_weight = (1 - natural_prob) ** 2  # gamma = 2
                inverse_freq = self.total_samples / size
                weights[domain] = focal_weight * inverse_freq
                
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Normalize weights
        total_weight = sum(weights.values())
        return {domain: w / total_weight for domain, w in weights.items()}
    
    def _create_sample_weights(self) -> List[float]:
        """Create per-sample weights for WeightedRandomSampler."""
        sample_weights = []
        
        for domain in self.domains:
            domain_weight = self.domain_weights[domain]
            domain_size = self.domain_sizes[domain]
            
            # Each sample in this domain gets the domain weight
            sample_weights.extend([domain_weight] * domain_size)
            
        return sample_weights
    
    def _expected_samples_per_batch(self) -> Dict[str, float]:
        """Calculate expected samples per domain per batch."""
        expected = {}
        for domain, weight in self.domain_weights.items():
            expected[domain] = weight * self.batch_size
        return expected
    
    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Get PyTorch WeightedRandomSampler."""
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True
        )
    
    def create_hierarchical_batch(self, dataset_indices: Dict[str, List[int]]) -> List[int]:
        """
        Create a batch with guaranteed minimum samples per domain.
        
        Args:
            dataset_indices: Dict mapping domain names to lists of sample indices
            
        Returns:
            List of sample indices for the batch
        """
        batch_indices = []
        
        # Calculate minimum samples per domain
        min_per_domain = max(1, int(self.batch_size * self.min_domain_ratio))
        remaining_slots = self.batch_size
        
        # First, ensure minimum representation
        for domain in self.domains:
            domain_indices = dataset_indices[domain]
            if len(domain_indices) == 0:
                continue
                
            # Sample minimum required
            n_samples = min(min_per_domain, remaining_slots, len(domain_indices))
            sampled = np.random.choice(domain_indices, size=n_samples, replace=False)
            batch_indices.extend(sampled)
            remaining_slots -= n_samples
        
        # Fill remaining slots with weighted sampling
        if remaining_slots > 0:
            all_indices = []
            all_weights = []
            
            for domain, indices in dataset_indices.items():
                if len(indices) == 0:
                    continue
                domain_weight = self.domain_weights[domain]
                all_indices.extend(indices)
                all_weights.extend([domain_weight] * len(indices))
            
            if all_indices:
                # Normalize weights
                all_weights = np.array(all_weights)
                all_weights = all_weights / all_weights.sum()
                
                # Sample remaining
                additional = np.random.choice(
                    all_indices, 
                    size=remaining_slots, 
                    replace=True, 
                    p=all_weights
                )
                batch_indices.extend(additional)
        
        return batch_indices
    
    def update_domain_performance(self, domain_losses: Dict[str, float]):
        """Update domain performance for adaptive reweighting."""
        for domain, loss in domain_losses.items():
            self.domain_losses[domain].append(loss)
            
            # Keep only recent history
            if len(self.domain_losses[domain]) > 100:
                self.domain_losses[domain] = self.domain_losses[domain][-100:]
    
    def adapt_weights(self, adaptation_strength: float = 0.1):
        """Adapt sampling weights based on recent performance."""
        if not self.domain_losses:
            return
            
        # Calculate recent average losses
        recent_losses = {}
        for domain, losses in self.domain_losses.items():
            if losses:
                recent_losses[domain] = np.mean(losses[-20:])  # Last 20 batches
        
        if len(recent_losses) < 2:
            return
            
        # Adapt weights: increase weight for domains with higher loss
        max_loss = max(recent_losses.values())
        adaptation_factors = {}
        
        for domain, loss in recent_losses.items():
            # Higher loss → higher weight
            factor = 1.0 + adaptation_strength * (loss / max_loss - 0.5)
            adaptation_factors[domain] = max(0.5, min(2.0, factor))  # Clamp
        
        # Update weights
        for domain in self.domain_weights:
            if domain in adaptation_factors:
                self.domain_weights[domain] *= adaptation_factors[domain]
        
        # Renormalize
        total_weight = sum(self.domain_weights.values())
        self.domain_weights = {
            domain: w / total_weight 
            for domain, w in self.domain_weights.items()
        }
        
        # Update sample weights
        self.sample_weights = self._create_sample_weights()
        
        print(f"Adapted domain weights: {self.domain_weights}")


# Example usage for your specific dataset
def create_sampler_for_cross_domain_dataset():
    """Example of how to use the enhanced sampler for your dataset."""
    
    # Your dataset distribution
    domain_sizes = {
        'photography': 298000,  # 78.9%
        'microscopy': 60000,    # 15.9%
        'astronomy': 20000,     # 5.2%
    }
    
    batch_size = 32
    
    # Strategy 1: Progressive reweighting (recommended starting point)
    sampler_progressive = EnhancedDomainSampler(
        domain_sizes=domain_sizes,
        batch_size=batch_size,
        strategy="progressive_reweight",
        reweight_power=0.7,  # Between sqrt (0.5) and inverse (1.0)
    )
    
    # Strategy 2: Hierarchical with guaranteed minimum
    sampler_hierarchical = EnhancedDomainSampler(
        domain_sizes=domain_sizes,
        batch_size=batch_size,
        strategy="hierarchical",
        min_domain_ratio=0.2,  # At least 20% of batch for smallest domain
    )
    
    # Strategy 3: Focal-weighted for hard examples
    sampler_focal = EnhancedDomainSampler(
        domain_sizes=domain_sizes,
        batch_size=batch_size,
        strategy="focal_weighted",
    )
    
    return sampler_progressive, sampler_hierarchical, sampler_focal


if __name__ == "__main__":
    # Test the samplers
    samplers = create_sampler_for_cross_domain_dataset()
    
    for i, sampler in enumerate(samplers):
        print(f"\n=== Sampler {i+1} ===")
        expected = sampler._expected_samples_per_batch()
        for domain, count in expected.items():
            print(f"{domain}: {count:.2f} samples per batch ({count/32*100:.1f}%)")
