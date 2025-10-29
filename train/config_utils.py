#!/usr/bin/env python3
"""
Configuration utilities for EDM native training.

This module provides common configuration setup functions that are shared
across all training scripts.
"""

import os
import pickle  # nosec B403
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Add project root and EDM to path
project_root = Path(__file__).parent.parent  # Go up from train/ to project root
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

import external.edm.dnnlib
from external.edm.torch_utils import distributed as dist


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file (required).

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If config_path is not provided or file does not exist.

    Example:
        config = load_yaml_config("config/sony.yaml")
        config = load_yaml_config("config/fuji.yaml")
        config = load_yaml_config("config/sony_fuji.yaml")
    """
    if config_path is None:
        raise ValueError(
            "Config path is required. Please specify one of:\n"
            "  - config/sony.yaml\n"
            "  - config/fuji.yaml\n"
            "  - config/sony_fuji.yaml"
        )

    config_path = Path(config_path)

    if not config_path.exists():
        raise ValueError(
            f"Config file not found: {config_path}\n"
            "Available config files:\n"
            "  - config/sony.yaml\n"
            "  - config/fuji.yaml\n"
            "  - config/sony_fuji.yaml"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dist.print0(f"Loaded configuration from: {config_path}")
    return config


def apply_config_to_args(args, config: Dict[str, Any]):
    """
    Apply configuration values to argparse arguments.

    Priority: Command-line args > Config file > Defaults

    Args:
        args: Parsed arguments object
        config: Configuration dictionary
    """
    # Default values for arguments (fallback if not in config or command line)
    defaults = {
        "batch_size": 4,
        "lr": 0.0001,
        "total_kimg": 300,
        "ema_halflife_kimg": 50,
        "lr_rampup_kimg": 10,
        "kimg_per_tick": 12,
        "snapshot_ticks": 2,
        "early_stopping_patience": 5,
        "img_resolution": 256,
        "channels": 3,
        "model_channels": 192,
        "channel_mult": [1, 2, 3, 4],
        "device": "cuda",
        "seed": 42,
    }

    # Mapping from config keys to argument names
    config_mapping = {
        "batch_size": "batch_size",
        "learning_rate": "lr",
        "total_kimg": "total_kimg",
        "ema_halflife_kimg": "ema_halflife_kimg",
        "lr_rampup_kimg": "lr_rampup_kimg",
        "kimg_per_tick": "kimg_per_tick",
        "snapshot_ticks": "snapshot_ticks",
        "early_stopping_patience": "early_stopping_patience",
        "img_resolution": "img_resolution",
        "channels": "channels",
        "model_channels": "model_channels",
        "channel_mult": "channel_mult",
        "device": "device",
        "seed": "seed",
    }

    # Apply config values if not set via command line
    for config_key, arg_name in config_mapping.items():
        current_value = getattr(args, arg_name, None)

        # If command line arg is None, use config value or default
        if current_value is None or current_value == []:
            config_value = config.get(config_key)
            if config_value is not None:
                setattr(args, arg_name, config_value)
                dist.print0(f"Applied config: {arg_name} = {config_value}")
            elif arg_name in defaults:
                setattr(args, arg_name, defaults[arg_name])
                dist.print0(f"Applied default: {arg_name} = {defaults[arg_name]}")


def setup_output_directory(output_dir):
    """Setup output directory and print header."""
    if dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        dist.print0("=" * 60)
        dist.print0("EDM DIFFUSION TRAINING WITH FLOAT32 .PT FILES")
        dist.print0("NO QUANTIZATION - FULL PRECISION")
        dist.print0("=" * 60)


def setup_logging(run_dir):
    """Setup logging to file."""
    if dist.get_rank() == 0:
        log_file = os.path.join(run_dir, "log.txt")
        logger = external.edm.dnnlib.util.Logger(
            file_name=log_file, file_mode="a", should_flush=True
        )


def create_network_config(train_dataset, args):
    """Create network configuration."""
    network_kwargs = external.edm.dnnlib.EasyDict(
        class_name="external.edm.training.networks.EDMPrecond",
        img_resolution=train_dataset.resolution,
        img_channels=train_dataset.num_channels,
        label_dim=train_dataset.label_dim,
        use_fp16=False,
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        model_type="DhariwalUNet",
        model_channels=args.model_channels,
        channel_mult=args.channel_mult,
    )
    return network_kwargs


def create_loss_config():
    """Create loss configuration."""
    loss_kwargs = external.edm.dnnlib.EasyDict(
        class_name="external.edm.training.loss.EDMLoss",
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
    )
    return loss_kwargs


def create_optimizer_config(args):
    """Create optimizer configuration."""
    optimizer_kwargs = external.edm.dnnlib.EasyDict(
        class_name="torch.optim.Adam",
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer_kwargs


def load_checkpoint(checkpoint_path):
    """Load checkpoint from file."""
    if checkpoint_path is None:
        return None, 0

    dist.print0(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        resume_state = pickle.load(f)  # nosec B301
    checkpoint_name = os.path.basename(checkpoint_path)
    start_kimg = int(checkpoint_name.split("-")[2].split(".")[0])
    dist.print0(f"Resuming from {start_kimg} kimg")
    return resume_state, start_kimg


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in a directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("network-snapshot-") and f.endswith(".pkl")
    ]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split("-")[2].split(".")[0]))
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
        dist.print0(f"Auto-resuming from: {checkpoint_path}")
        return checkpoint_path
    return None
