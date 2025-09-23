#!/usr/bin/env python3
"""
Unified training script supporting both Poisson and L2 guidance.

This script allows training identical models with different guidance types
for perfect ablation studies.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml

from core.guidance_factory import create_guidance_from_config
from poisson_training.multi_domain_trainer import MultiDomainTrainer
from poisson_training.utils import set_deterministic_mode


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train model with specified guidance type"
    )
    parser.add_argument("--config", required=True, help="Training configuration file")
    parser.add_argument(
        "--guidance-type",
        choices=["poisson", "l2"],
        help="Override guidance type from config",
    )
    parser.add_argument(
        "--experiment-suffix", default="", help="Suffix to add to experiment name"
    )
    parser.add_argument("--device", default="cuda", help="Device to use for training")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override guidance type if specified
    if args.guidance_type:
        if "guidance" not in config:
            config["guidance"] = {}
        config["guidance"]["type"] = args.guidance_type

    # Add suffix to experiment name for identification
    if args.experiment_suffix:
        if "training" not in config:
            config["training"] = {}
        experiment_name = config["training"].get("experiment_name", "experiment")
        config["training"][
            "experiment_name"
        ] = f"{experiment_name}_{args.experiment_suffix}"

    # Set deterministic mode for reproducibility
    if config.get("training", {}).get("deterministic", False):
        seed = config.get("training", {}).get("seed", 42)
        set_deterministic_mode(seed)
        print(f"Set deterministic mode with seed: {seed}")

    # Validate required configuration
    if "guidance" not in config or "type" not in config["guidance"]:
        raise ValueError("Configuration must specify guidance.type")

    # Create guidance computer
    try:
        guidance = create_guidance_from_config(config)
    except Exception as e:
        print(f"Error creating guidance: {e}")
        print("Config structure:")
        print(f"  guidance: {config.get('guidance', 'MISSING')}")
        print(f"  data: {config.get('data', 'MISSING')}")
        raise

    # Create trainer with guidance
    trainer = MultiDomainTrainer(config, guidance=guidance)

    guidance_type = config["guidance"]["type"]
    experiment_name = config.get("training", {}).get("experiment_name", "unnamed")
    seed = config.get("training", {}).get("seed", 42)

    print(f"Starting training with {guidance_type.upper()} guidance")
    print(f"Experiment: {experiment_name}")
    print(f"Seed: {seed}")
    print(f"Device: {args.device}")

    # Train model
    try:
        trainer.train()
        print(f"Training completed successfully: {experiment_name}")
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
