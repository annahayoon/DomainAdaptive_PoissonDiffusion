#!/usr/bin/env python3
"""Train unconditional diffusion model on preprocessed image tiles."""

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

from core.dataset import SimplePTDataset
from core.utils.file_utils import load_yaml_config
from core.utils.tensor_utils import get_device
from core.utils.training_utils import (
    apply_config_to_args,
    create_loss_config,
    create_network_config,
    create_optimizer_config,
    load_checkpoint,
    setup_output_directory,
)
from core.utils.training_utils import setup_training_logging as setup_logging
from core.utils.training_utils import training_loop
from external.edm.torch_utils import distributed as dist


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all training arguments."""
    parser = argparse.ArgumentParser(
        description="Train unconditional diffusion model on preprocessed clean images"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to data directory containing preprocessed .pt tiles (for single dataset mode)",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        default=None,
        help="Path to metadata JSON file with train/validation splits (for single dataset mode)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Total batch size across GPUs"
    )
    parser.add_argument(
        "--batch_gpu", type=int, default=None, help="Batch size per GPU"
    )
    parser.add_argument(
        "--total_kimg", type=int, default=None, help="Training duration in kimg"
    )
    parser.add_argument(
        "--ema_halflife_kimg", type=int, default=None, help="EMA half-life in kimg"
    )
    parser.add_argument(
        "--lr_rampup_kimg", type=int, default=None, help="LR ramp-up in kimg"
    )
    parser.add_argument(
        "--kimg_per_tick",
        type=int,
        default=None,
        help="Progress print interval in kimg",
    )
    parser.add_argument(
        "--snapshot_ticks",
        type=int,
        default=None,
        help="Snapshot save interval in ticks",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Early stopping patience in ticks",
    )
    parser.add_argument(
        "--img_resolution", type=int, default=None, help="Image resolution"
    )
    parser.add_argument(
        "--channels", type=int, default=None, help="Number of image channels"
    )
    parser.add_argument(
        "--model_channels", type=int, default=None, help="Model base channels"
    )
    parser.add_argument(
        "--channel_mult",
        type=int,
        nargs="+",
        default=None,
        help="Channel multipliers per resolution",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/edm_diffusion_training",
        help="Output directory",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None, help="Resume from checkpoint file"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Pin memory for faster GPU transfer",
    )

    return parser


def setup_training(args, config):
    """Setup all training components from args and config."""
    setup_output_directory(args.output_dir)
    setup_logging(args.output_dir)

    dist.print0("Loading datasets...")
    train_dataset, val_dataset = _create_datasets(args, config)

    device = get_device(prefer_cuda=(args.device != "cpu" if args.device else True))
    dist.print0(f"Using device: {device}")

    network_kwargs = create_network_config(train_dataset, args)
    loss_kwargs = create_loss_config()
    optimizer_kwargs = create_optimizer_config(args)

    resume_state, start_kimg = load_checkpoint(args.resume_from)

    return (
        train_dataset,
        val_dataset,
        device,
        network_kwargs,
        loss_kwargs,
        optimizer_kwargs,
        resume_state,
        start_kimg,
    )


def run_training(
    args,
    train_dataset,
    val_dataset,
    network_kwargs,
    loss_kwargs,
    optimizer_kwargs,
    device,
    resume_state,
    start_kimg,
):
    """Run the training loop."""
    training_loop(
        run_dir=args.output_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        network_kwargs=network_kwargs,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        seed=args.seed,
        batch_size=args.batch_size,
        batch_gpu=args.batch_gpu,
        total_kimg=args.total_kimg,
        ema_halflife_kimg=args.ema_halflife_kimg,
        lr_rampup_kimg=args.lr_rampup_kimg,
        kimg_per_tick=args.kimg_per_tick,
        snapshot_ticks=args.snapshot_ticks,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
        resume_state=resume_state,
        start_kimg=start_kimg,
    )


def _create_single_dataset(data_root, metadata_json, split, image_size, channels):
    """Create a single SimplePTDataset instance."""
    return SimplePTDataset(
        data_root=data_root,
        metadata_json=metadata_json,
        split=split,
        image_size=image_size,
        channels=channels,
    )


def _create_datasets(args, config):
    """Create train and validation datasets from config."""
    if "datasets" in config and isinstance(config["datasets"], list):
        from torch.utils.data import ConcatDataset

        train_datasets = []
        val_datasets = []

        for dataset_config in config["datasets"]:
            train_datasets.append(
                _create_single_dataset(
                    data_root=dataset_config["data_root"],
                    metadata_json=dataset_config["metadata_json"],
                    split="train",
                    image_size=args.img_resolution,
                    channels=args.channels,
                )
            )
            val_datasets.append(
                _create_single_dataset(
                    data_root=dataset_config["data_root"],
                    metadata_json=dataset_config["metadata_json"],
                    split="validation",
                    image_size=args.img_resolution,
                    channels=args.channels,
                )
            )

        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    else:
        data_root = args.data_root or config.get("data_root")
        metadata_json = args.metadata_json or config.get("metadata_json")

        train_dataset = _create_single_dataset(
            data_root=data_root,
            metadata_json=metadata_json,
            split="train",
            image_size=args.img_resolution,
            channels=args.channels,
        )
        val_dataset = _create_single_dataset(
            data_root=data_root,
            metadata_json=metadata_json,
            split="validation",
            image_size=args.img_resolution,
            channels=args.channels,
        )

    return train_dataset, val_dataset


def main():
    """Train unconditional EDM diffusion model on clean images."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        config = load_yaml_config(args.config)
        apply_config_to_args(args, config)
        dist.print0(f"Configuration loaded from: {args.config}")
    except ValueError as e:
        dist.print0(f"ERROR: {e}")
        sys.exit(1)

    (
        train_dataset,
        val_dataset,
        device,
        network_kwargs,
        loss_kwargs,
        optimizer_kwargs,
        resume_state,
        start_kimg,
    ) = setup_training(args, config)

    run_training(
        args=args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        network_kwargs=network_kwargs,
        loss_kwargs=loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
        resume_state=resume_state,
        start_kimg=start_kimg,
    )


if __name__ == "__main__":
    main()
