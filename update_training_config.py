#!/usr/bin/env python3
"""
Update training configuration based on dataset size analysis.
"""

import sys
from pathlib import Path

from utils.training_config import (
    calculate_optimal_training_config,
    print_training_analysis,
)


def update_training_script(dataset_size: int, batch_size: int = 16):
    """Update run_real_training.sh with optimal configuration."""

    # Calculate optimal config
    config = calculate_optimal_training_config(dataset_size, batch_size)

    print("🎯 DATASET ANALYSIS & CONFIGURATION UPDATE")
    print("=" * 60)
    print_training_analysis(config)

    # Read current script
    script_path = Path("run_real_training.sh")
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return

    content = script_path.read_text()

    # Update epochs
    old_epochs_line = None
    new_epochs_line = f"    --epochs {config['recommended_epochs']} \\"

    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "--epochs" in line and "\\" in line:
            old_epochs_line = line.strip()
            lines[i] = new_epochs_line
            break

    if old_epochs_line:
        # Write updated script
        updated_content = "\n".join(lines)
        script_path.write_text(updated_content)

        print()
        print("🔄 SCRIPT UPDATE:")
        print(f"   Old: {old_epochs_line}")
        print(f"   New: {new_epochs_line}")
        print(f"✅ Updated {script_path}")

        print()
        print("📊 TRAINING IMPACT:")
        print(f"   Training Steps: {config['total_training_steps']:,}")
        print(f"   Steps per Sample: {config['steps_per_sample']}")
        print(f"   Estimated Time: {config['estimated_time_hours']:.1f} hours")

    else:
        print("❌ Could not find --epochs line in script")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_size = int(sys.argv[1])
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    else:
        # Default to our current dataset size
        dataset_size = 10220
        batch_size = 16

    update_training_script(dataset_size, batch_size)
