#!/usr/bin/env python3
"""
Debug script to diagnose training issues.
Run this to check if your training is actually working.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def check_data_statistics(data_dir="/home/jilab/Jae/data/preprocessed_photography"):
    """Check if the preprocessed data has correct statistics."""

    print("=" * 70)
    print("DATA STATISTICS CHECK")
    print("=" * 70)

    # Load a few samples
    data_files = list(Path(data_dir).glob("*.pt"))[:10]

    if not data_files:
        print("❌ No preprocessed data found!")
        return False

    clean_vals = []
    noisy_vals = []
    electron_vals = []

    for f in data_files:
        data = torch.load(f, map_location="cpu")
        clean_vals.append(data["clean"].numpy())
        noisy_vals.append(data["noisy"].numpy())
        electron_vals.append(data["electrons"].numpy())

    clean_vals = np.concatenate([v.flatten() for v in clean_vals])
    noisy_vals = np.concatenate([v.flatten() for v in noisy_vals])
    electron_vals = np.concatenate([v.flatten() for v in electron_vals])

    print(f"\n📊 Clean Image Statistics:")
    print(f"   Min: {clean_vals.min():.4f}")
    print(f"   Max: {clean_vals.max():.4f}")
    print(f"   Mean: {clean_vals.mean():.4f}")
    print(f"   Std: {clean_vals.std():.4f}")

    print(f"\n📊 Noisy Image Statistics:")
    print(f"   Min: {noisy_vals.min():.4f}")
    print(f"   Max: {noisy_vals.max():.4f}")
    print(f"   Mean: {noisy_vals.mean():.4f}")
    print(f"   Std: {noisy_vals.std():.4f}")

    print(f"\n📊 Electron Count Statistics:")
    print(f"   Min: {electron_vals.min():.4f}")
    print(f"   Max: {electron_vals.max():.4f}")
    print(f"   Mean: {electron_vals.mean():.4f}")
    print(f"   Std: {electron_vals.std():.4f}")

    # Check difference between clean and noisy
    diff = np.abs(clean_vals[: len(noisy_vals)] - noisy_vals).mean()
    print(f"\n📊 Mean Absolute Difference (Clean - Noisy): {diff:.4f}")

    # Warnings
    issues = []

    if diff < 0.01:
        issues.append("⚠️ Clean and noisy images are nearly identical!")

    if clean_vals.max() > 100:
        issues.append("⚠️ Data values are very large (not normalized?)")

    if clean_vals.min() < -1:
        issues.append("⚠️ Data has large negative values (incorrect preprocessing?)")

    if electron_vals.max() < 10:
        issues.append("⚠️ Electron counts are very low (wrong scaling?)")

    if issues:
        print("\n🚨 ISSUES DETECTED:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("\n✅ Data statistics look reasonable")
        return True


def check_loss_calculation():
    """Test if the loss function is working correctly."""

    print("\n" + "=" * 70)
    print("LOSS FUNCTION CHECK")
    print("=" * 70)

    from poisson_training.losses import PoissonGaussianLoss

    # Create synthetic test data
    batch_size = 2
    channels = 4
    size = 128

    # Test case 1: Identical clean and noisy (should give near-zero loss)
    clean = torch.randn(batch_size, channels, size, size) * 0.1 + 0.5
    noisy = clean.clone()
    electrons = torch.ones_like(clean) * 100  # Moderate photon count

    loss_fn = PoissonGaussianLoss()
    loss1 = loss_fn(
        predictions=clean,
        electrons=electrons,
        scale=torch.ones(batch_size, 1, 1, 1) * 100,
        background=torch.zeros(batch_size, 1, 1, 1),
        read_noise=torch.ones(batch_size, 1, 1, 1) * 0.1,
    )

    print(f"\nTest 1 - Identical images:")
    print(f"   Loss: {loss1.item():.6f}")
    print(f"   Expected: ~0 (should be very small)")

    # Test case 2: Very different images (should give large loss)
    noisy2 = torch.randn_like(clean) * 0.5 + 0.5
    loss2 = loss_fn(
        predictions=clean,
        electrons=electrons,
        scale=torch.ones(batch_size, 1, 1, 1) * 100,
        background=torch.zeros(batch_size, 1, 1, 1),
        read_noise=torch.ones(batch_size, 1, 1, 1) * 0.1,
    )

    print(f"\nTest 2 - Random noise:")
    print(f"   Loss: {loss2.item():.6f}")
    print(f"   Expected: >10 (should be large)")

    # Test case 3: Check for NaN/Inf
    electrons_zero = torch.zeros_like(clean)
    try:
        loss3 = loss_fn(
            predictions=clean,
            electrons=electrons_zero,
            scale=torch.ones(batch_size, 1, 1, 1) * 100,
            background=torch.zeros(batch_size, 1, 1, 1),
            read_noise=torch.ones(batch_size, 1, 1, 1) * 0.1,
        )
        print(f"\nTest 3 - Zero electrons:")
        print(f"   Loss: {loss3.item():.6f}")
        if torch.isnan(loss3) or torch.isinf(loss3):
            print(f"   ❌ Loss is NaN/Inf with zero electrons!")
            return False
    except Exception as e:
        print(f"   ❌ Loss calculation failed with zero electrons: {e}")
        return False

    # Validate results
    if loss1.item() > 1.0:
        print("\n❌ Loss for identical images is too high!")
        return False

    if loss2.item() < 1.0:
        print("\n❌ Loss for random noise is too low!")
        return False

    print("\n✅ Loss function appears to be working correctly")
    return True


def check_model_output():
    """Check if the model is actually producing meaningful outputs."""

    print("\n" + "=" * 70)
    print("MODEL OUTPUT CHECK")
    print("=" * 70)

    # Try to load the latest checkpoint
    checkpoint_dir = Path("results").glob("research_*")
    latest_dir = max(checkpoint_dir, key=lambda p: p.stat().st_mtime, default=None)

    if not latest_dir:
        print("❌ No training results found")
        return False

    checkpoint_path = latest_dir / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        # Try any checkpoint
        checkpoints = list((latest_dir / "checkpoints").glob("*.pt"))
        if not checkpoints:
            print("❌ No checkpoints found")
            return False
        checkpoint_path = checkpoints[0]

    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Check training history
        if "training_history" in checkpoint:
            history = checkpoint["training_history"]
            if "train_total_loss" in history:
                losses = history["train_total_loss"][-100:]  # Last 100 losses

                print(f"\n📊 Recent Training Losses:")
                print(f"   Min: {min(losses):.4f}")
                print(f"   Max: {max(losses):.4f}")
                print(f"   Mean: {np.mean(losses):.4f}")
                print(f"   Std: {np.std(losses):.4f}")

                # Check for learning
                if len(losses) > 10:
                    early_mean = np.mean(losses[:10])
                    late_mean = np.mean(losses[-10:])
                    improvement = (early_mean - late_mean) / early_mean * 100

                    print(f"\n📈 Learning Progress:")
                    print(f"   Early mean: {early_mean:.4f}")
                    print(f"   Late mean: {late_mean:.4f}")
                    print(f"   Improvement: {improvement:.1f}%")

                    if improvement < 1:
                        print("   ⚠️ Model is not improving!")
                        return False

        print("\n✅ Model checkpoint loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False


def main():
    """Run all diagnostic checks."""

    print("\n" + "🔬 TRAINING DIAGNOSTICS " + "🔬")
    print("=" * 70)

    results = {
        "Data Statistics": check_data_statistics(),
        "Loss Function": check_loss_calculation(),
        "Model Output": check_model_output(),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:20}: {status}")

    if all(results.values()):
        print("\n✅ All checks passed - training should be working")
    else:
        print("\n🚨 Some checks failed - training has issues!")
        print("\nRecommended actions:")

        if not results["Data Statistics"]:
            print("  1. Re-run data preprocessing")
            print("  2. Check if clean/noisy pairs are correctly generated")

        if not results["Loss Function"]:
            print("  1. Review loss function implementation")
            print("  2. Check for numerical stability issues")

        if not results["Model Output"]:
            print("  1. Check if model is actually updating")
            print("  2. Verify optimizer and learning rate")


if __name__ == "__main__":
    main()
