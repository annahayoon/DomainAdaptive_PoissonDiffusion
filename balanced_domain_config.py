"""
Simplified Configuration for Balanced Cross-Domain Dataset

Updated distribution:
- Photography: ~85.6K tiles (51.8%) [Sony: 23.4K + Fuji: 62.2K]
  - Sony: ~2,928 files × 8 tiles = 23,424 tiles
  - Fuji: ~2,590 files × 24 tiles = 62,160 tiles
- Microscopy: ~60.1K tiles (36.4%) [3,757 files × 16 tiles = 60,112 tiles]
- Astronomy: ~19.6K tiles (11.9%) [306 files × 64 tiles = 19,584 tiles]

Total: ~165,280 tiles across all domains
This is reasonably balanced and requires moderate reweighting strategies.
"""

import math
from typing import Dict, Optional
from torch.utils.data import WeightedRandomSampler

class BalancedDomainConfig:
    """Simple configuration for the now-balanced cross-domain dataset."""
    
    def __init__(self, reweight_strategy: str = "sqrt"):
        """
        Initialize balanced domain configuration.
        
        Args:
            reweight_strategy: 'none', 'sqrt', 'inverse', or 'custom'
        """
        self.domain_sizes = {
            'photography': 85584,  # 51.8% (Sony: 23,424 + Fuji: 62,160)
            'microscopy': 60112,   # 36.4%
            'astronomy': 19584,    # 11.9%
        }
        
        self.total_samples = sum(self.domain_sizes.values())  # 165,280
        self.reweight_strategy = reweight_strategy
        
        # Calculate domain weights
        self.domain_weights = self._calculate_weights()
        
        # Light augmentation factors (much more conservative)
        self.augmentation_factors = {
            'photography': 1.0,  # No extra augmentation
            'microscopy': 1.0,   # No extra augmentation  
            'astronomy': 2.0,    # Light augmentation
        }
        
        print(f"Balanced Domain Configuration:")
        print(f"  Strategy: {reweight_strategy}")
        print(f"  Domain sizes: {self.domain_sizes}")
        print(f"  Domain weights: {self.domain_weights}")
        print(f"  Expected batch distribution: {self._expected_batch_distribution()}")
    
    def _calculate_weights(self) -> Dict[str, float]:
        """Calculate domain weights based on strategy."""
        
        if self.reweight_strategy == "none":
            # No reweighting - natural distribution
            return {domain: 1.0 for domain in self.domain_sizes.keys()}
        
        elif self.reweight_strategy == "sqrt":
            # Square root of inverse frequency (recommended)
            weights = {}
            for domain, size in self.domain_sizes.items():
                weight = math.sqrt(self.total_samples / size)
                weights[domain] = weight
        
        elif self.reweight_strategy == "inverse":
            # Pure inverse frequency (more aggressive)
            weights = {}
            for domain, size in self.domain_sizes.items():
                weight = self.total_samples / (len(self.domain_sizes) * size)
                weights[domain] = weight
        
        elif self.reweight_strategy == "custom":
            # Custom weights for your specific needs
            weights = {
                'photography': 0.8,   # Slight down-weight (most common)
                'microscopy': 0.9,    # Slight down-weight
                'astronomy': 1.5,     # Moderate up-weight (least common)
            }
        
        else:
            raise ValueError(f"Unknown reweight_strategy: {self.reweight_strategy}")
        
        # Normalize weights
        total_weight = sum(weights.values())
        return {domain: w / total_weight for domain, w in weights.items()}
    
    def _expected_batch_distribution(self, batch_size: int = 32) -> Dict[str, float]:
        """Calculate expected samples per domain in a batch."""
        return {
            domain: weight * batch_size 
            for domain, weight in self.domain_weights.items()
        }
    
    def create_sample_weights(self, domain_labels: list) -> list:
        """
        Create per-sample weights for WeightedRandomSampler.
        
        Args:
            domain_labels: List of domain names for each sample
            
        Returns:
            List of weights for each sample
        """
        return [self.domain_weights[domain] for domain in domain_labels]
    
    def get_weighted_sampler(self, domain_labels: list) -> WeightedRandomSampler:
        """Get PyTorch WeightedRandomSampler."""
        sample_weights = self.create_sample_weights(domain_labels)
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def should_use_complex_strategies(self) -> bool:
        """Check if complex balancing strategies are needed."""
        # With this balance, complex strategies are overkill
        return False
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights for domain-weighted loss function."""
        # Much gentler loss weighting needed
        if self.reweight_strategy == "none":
            return {domain: 1.0 for domain in self.domain_sizes.keys()}
        
        # Use same weights as sampling, but with less extreme values
        loss_weights = {}
        for domain, weight in self.domain_weights.items():
            # Dampen the weights for loss (less aggressive than sampling)
            dampened_weight = 1.0 + 0.5 * (weight - 1.0)
            loss_weights[domain] = max(0.5, min(2.0, dampened_weight))
        
        return loss_weights


def compare_strategies():
    """Compare different reweighting strategies for the balanced dataset."""
    
    strategies = ["none", "sqrt", "inverse", "custom"]
    batch_size = 32
    
    print("Strategy Comparison for Balanced Dataset (165K total samples):")
    print("=" * 70)
    
    for strategy in strategies:
        config = BalancedDomainConfig(reweight_strategy=strategy)
        expected = config._expected_batch_distribution(batch_size)
        
        print(f"\n{strategy.upper()} Strategy:")
        print(f"  Weights: {config.domain_weights}")
        print(f"  Expected per batch (size={batch_size}):")
        for domain, count in expected.items():
            percentage = (count / batch_size) * 100
            print(f"    {domain}: {count:.1f} samples ({percentage:.1f}%)")


def integration_example():
    """Example of how to integrate with your existing training code."""
    
    # Create configuration
    config = BalancedDomainConfig(reweight_strategy="sqrt")  # Recommended
    
    # Example domain labels for your dataset
    # You'll need to create this list based on your actual data
    domain_labels = (
        ['photography'] * 85584 + 
        ['microscopy'] * 60112 + 
        ['astronomy'] * 19584
    )
    
    # Get weighted sampler
    weighted_sampler = config.get_weighted_sampler(domain_labels)
    
    # Get loss weights
    loss_weights = config.get_loss_weights()
    
    print("Integration Example:")
    print(f"  Sample weights created for {len(domain_labels):,} samples")
    print(f"  Loss weights: {loss_weights}")
    print(f"  Use complex strategies: {config.should_use_complex_strategies()}")
    
    return config, weighted_sampler, loss_weights


if __name__ == "__main__":
    # Compare different strategies
    compare_strategies()
    
    print("\n" + "=" * 70)
    
    # Show integration example
    integration_example()
