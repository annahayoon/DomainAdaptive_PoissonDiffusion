#!/usr/bin/env python3
"""
Train L2-guided diffusion models for all domains.

This script provides a convenient way to train L2 baseline models
for comparison with Poisson-Gaussian models.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import sys

from scripts.train_with_guidance_type import main as train_main


def main():
    parser = argparse.ArgumentParser(description="Train L2 baseline models")
    parser.add_argument(
        "--domain",
        choices=["photography", "microscopy", "astronomy", "all"],
        default="all",
        help="Domain to train (default: all)",
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--experiment-suffix", default="", help="Experiment suffix")
    args = parser.parse_args()

    domains = (
        ["photography", "microscopy", "astronomy"]
        if args.domain == "all"
        else [args.domain]
    )

    for domain in domains:
        config_path = f"configs/l2_baseline_{domain}.yaml"

        print(f"\n{'='*60}")
        print(f"Training L2 baseline for {domain.upper()} domain")
        print(f"Config: {config_path}")
        print(f"{'='*60}")

        # Prepare arguments for training script
        train_args = [
            "train_with_guidance_type.py",
            "--config",
            config_path,
            "--guidance-type",
            "l2",
            "--device",
            args.device,
        ]

        if args.experiment_suffix:
            train_args.extend(["--experiment-suffix", args.experiment_suffix])

        # Override sys.argv for the training script
        original_argv = sys.argv
        sys.argv = train_args

        try:
            train_main()
            print(f"✓ Successfully trained L2 baseline for {domain}")
        except Exception as e:
            print(f"✗ Failed to train L2 baseline for {domain}: {e}")
        finally:
            sys.argv = original_argv

    print(f"\nL2 baseline training completed for domains: {', '.join(domains)}")


if __name__ == "__main__":
    main()
