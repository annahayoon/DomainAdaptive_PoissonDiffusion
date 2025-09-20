"""
Automatic training configuration based on dataset size and diffusion best practices.
"""

import math
from pathlib import Path
from typing import Any, Dict, Tuple


def calculate_optimal_training_config(
    dataset_size: int,
    batch_size: int = 16,
    target_steps_per_sample: float = 10.0,
    min_steps: int = 50000,
    max_steps: int = 1000000,
    min_epochs: int = 50,
    max_epochs: int = 1000,
) -> Dict[str, Any]:
    """
    Calculate optimal training configuration based on dataset size.

    Follows diffusion model best practices:
    - Small datasets (1K-10K): 100K-500K steps (10-50 steps per sample)
    - Medium datasets (10K-100K): 200K-800K steps (2-8 steps per sample)
    - Large datasets (100K+): 500K+ steps (5+ steps per sample)

    Args:
        dataset_size: Number of training samples
        batch_size: Training batch size
        target_steps_per_sample: Target training steps per sample
        min_steps: Minimum total training steps
        max_steps: Maximum total training steps
        min_epochs: Minimum epochs
        max_epochs: Maximum epochs

    Returns:
        Dictionary with optimal training configuration
    """

    # Calculate steps per epoch
    steps_per_epoch = math.ceil(dataset_size / batch_size)

    # Calculate target total steps based on dataset size
    if dataset_size <= 1000:
        # Very small datasets need more steps per sample
        target_total_steps = max(dataset_size * 50, 100000)
    elif dataset_size <= 10000:
        # Small datasets
        target_total_steps = max(dataset_size * 20, 200000)
    elif dataset_size <= 50000:
        # Medium datasets
        target_total_steps = max(dataset_size * 10, 300000)
    elif dataset_size <= 100000:
        # Large datasets
        target_total_steps = max(dataset_size * 5, 500000)
    else:
        # Very large datasets
        target_total_steps = max(dataset_size * 3, 1000000)

    # Apply constraints
    target_total_steps = max(min_steps, min(max_steps, target_total_steps))

    # Calculate epochs needed
    target_epochs = math.ceil(target_total_steps / steps_per_epoch)
    target_epochs = max(min_epochs, min(max_epochs, target_epochs))

    # Recalculate actual total steps
    actual_total_steps = target_epochs * steps_per_epoch
    actual_steps_per_sample = actual_total_steps / dataset_size

    # Estimate training time (assuming 1.5s per step on A40)
    estimated_time_hours = (actual_total_steps * 1.5) / 3600

    return {
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "recommended_epochs": target_epochs,
        "total_training_steps": actual_total_steps,
        "steps_per_sample": round(actual_steps_per_sample, 1),
        "estimated_time_hours": round(estimated_time_hours, 1),
        "category": _get_dataset_category(dataset_size),
    }


def _get_dataset_category(dataset_size: int) -> str:
    """Get dataset size category."""
    if dataset_size <= 1000:
        return "Very Small (≤1K)"
    elif dataset_size <= 10000:
        return "Small (1K-10K)"
    elif dataset_size <= 50000:
        return "Medium (10K-50K)"
    elif dataset_size <= 100000:
        return "Large (50K-100K)"
    else:
        return "Very Large (100K+)"


def print_training_analysis(config: Dict[str, Any]) -> None:
    """Print detailed training analysis."""
    print("🎯 OPTIMAL TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"📊 Dataset: {config['dataset_size']:,} samples ({config['category']})")
    print(f"🔢 Batch Size: {config['batch_size']}")
    print(f"📈 Steps per Epoch: {config['steps_per_epoch']:,}")
    print(f"🎯 Recommended Epochs: {config['recommended_epochs']:,}")
    print(f"🚀 Total Training Steps: {config['total_training_steps']:,}")
    print(f"📝 Steps per Sample: {config['steps_per_sample']}")
    print(f"⏱️  Estimated Time: {config['estimated_time_hours']:.1f} hours")
    print()

    # Best practices check
    if config["steps_per_sample"] < 5:
        print("⚠️  WARNING: Low steps per sample - may underfit")
    elif config["steps_per_sample"] > 50:
        print("⚠️  WARNING: High steps per sample - may overfit")
    else:
        print("✅ Steps per sample in optimal range")

    if config["estimated_time_hours"] > 48:
        print("⚠️  WARNING: Very long training time (>48h)")
    elif config["estimated_time_hours"] < 2:
        print("⚠️  WARNING: Very short training time (<2h)")
    else:
        print("✅ Training time reasonable")


if __name__ == "__main__":
    # Test with different dataset sizes
    test_sizes = [1000, 5000, 10000, 25000, 50000, 100000]

    for size in test_sizes:
        config = calculate_optimal_training_config(size)
        print_training_analysis(config)
        print()
