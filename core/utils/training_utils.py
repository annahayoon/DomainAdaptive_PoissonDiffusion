"""
Training utility functions for the Poisson-Gaussian Diffusion project.

This module provides training loop, validation, checkpoint management,
configuration utilities for EDM model training, and model-related utilities
like noise schedule creation and conditioning preparation.
"""

import copy
import json
import logging
import os
import pickle  # nosec B403
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Import shared utilities from sampling_utils to avoid duplication
from core.utils.sampling_utils import create_edm_noise_schedule, prepare_conditioning

logger = logging.getLogger(__name__)

# Global logger instance for training logging
_logger_instance = None


def training_loop(
    run_dir=".",
    train_dataset=None,
    val_dataset=None,
    network_kwargs={},
    loss_kwargs={},
    optimizer_kwargs={},
    seed=0,
    batch_size=4,
    batch_gpu=None,
    total_kimg=10000,
    ema_halflife_kimg=500,
    lr_rampup_kimg=1000,
    kimg_per_tick=50,
    snapshot_ticks=10,
    early_stopping_patience=5,
    device=torch.device("cuda"),
    resume_state=None,
    start_kimg=0,
):
    """Main training loop for float32 .pt data following EDM's structure."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    edm_path = project_root / "external" / "edm"
    if str(edm_path) not in sys.path:
        sys.path.insert(0, str(edm_path))

    import external.edm.dnnlib
    from external.edm.torch_utils import distributed as dist
    from external.edm.torch_utils import misc, training_stats

    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = True

    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = 0

    # Select batch size per GPU
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Setup data loader using EDM's InfiniteSampler
    dist.print0("Loading dataset...")
    dataset_sampler = misc.InfiniteSampler(
        dataset=train_dataset,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=train_dataset,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            pin_memory=True,
            num_workers=2,
            prefetch_factor=2,
        )
    )

    dist.print0(
        f"Dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples"
    )

    dist.print0("Constructing network...")
    net = external.edm.dnnlib.util.construct_class_by_name(**network_kwargs)
    net.train().requires_grad_(True).to(device)

    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros(
                [batch_gpu, net.img_channels, net.img_resolution, net.img_resolution],
                device=device,
            )
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    dist.print0("Setting up optimizer...")
    if resume_state is not None and "loss_fn" in resume_state:
        loss_fn = resume_state["loss_fn"]
        dist.print0("Restored loss function from checkpoint")
    else:
        loss_fn = external.edm.dnnlib.util.construct_class_by_name(**loss_kwargs)

    optimizer = external.edm.dnnlib.util.construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )

    if resume_state is not None and "optimizer_state" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        dist.print0("Restored optimizer state from checkpoint")

    if dist.get_world_size() > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    else:
        ddp = net

    if resume_state is not None and "ema" in resume_state:
        ema = resume_state["ema"].to(device)
        dist.print0("Restored EMA model from checkpoint")
        training_net = ddp.module if dist.get_world_size() > 1 else ddp
        training_net.load_state_dict(ema.state_dict())
        dist.print0("Copied EMA weights to training network")
        cur_nimg = start_kimg * 1000
        cur_tick = cur_nimg // (kimg_per_tick * 1000)
        tick_start_nimg = cur_tick * kimg_per_tick * 1000
        dist.print0(
            f"Checkpoint loaded - continuing from {start_kimg} kimg (tick {cur_tick})"
        )
        dist.print0(
            f"Remaining: {total_kimg - start_kimg} kimg to reach {total_kimg} kimg"
        )
    else:
        ema = copy.deepcopy(net).eval().requires_grad_(False)

    remaining_kimg = total_kimg - start_kimg
    if resume_state is not None:
        dist.print0(
            f"Training for remaining {remaining_kimg} kimg (to reach {total_kimg} kimg)..."
        )
    else:
        dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()

    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    best_val_loss = float("inf")
    patience_counter = 0

    while True:
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device)
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=None)
                training_stats.report("Loss/loss", loss)
                loss.sum().mul(1.0 / batch_gpu_total).backward()

        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
        for param in net.parameters():
            if param.grad is not None:
                misc.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        optimizer.step()

        ema_halflife_nimg = ema_halflife_kimg * 1000
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000

        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)
        ):
            continue
        tick_end_time = time.time()
        fields = [
            f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}",
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}",
            f"time {external.edm.dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}",
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}",
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}",
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}",
        ]

        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / 2**30
            current_mem_gb = torch.cuda.memory_allocated(device) / 2**30
            fields.append(
                f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', peak_mem_gb):<6.2f}"
            )
            fields.append(f"gpu_cur {current_mem_gb:<6.2f}")
            torch.cuda.reset_peak_memory_stats()

        dist.print0(" ".join(fields))

        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            val_loss = validate_training(ema, val_dataset, loss_fn, device)
            dist.print0(f"Validation loss: {val_loss:.4f}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                if dist.get_rank() == 0:
                    checkpoint_path = os.path.join(run_dir, "best_model.pkl")
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump({"ema": ema}, f)  # nosec B301
                    dist.print0(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    dist.print0(
                        f"Early stopping triggered after {patience_counter} ticks without improvement"
                    )
                    break

            if dist.get_rank() == 0:
                checkpoint_path = os.path.join(
                    run_dir, f"network-snapshot-{cur_nimg // 1000:06d}.pkl"
                )
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(
                        {  # nosec B301
                            "ema": ema,
                            "loss_fn": loss_fn,
                            "optimizer_state": optimizer.state_dict(),
                        },
                        f,
                    )

        if done:
            break

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()

    dist.print0()
    dist.print0("Exiting...")


@torch.no_grad()
def validate_training(ema_model, val_dataset, loss_fn, device):
    """Run validation on validation dataset."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    edm_path = project_root / "external" / "edm"
    if str(edm_path) not in sys.path:
        sys.path.insert(0, str(edm_path))

    from external.edm.torch_utils import distributed as dist
    from external.edm.torch_utils import misc

    ema_model.eval()

    val_sampler = misc.InfiniteSampler(
        dataset=val_dataset,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=42,
    )
    val_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=val_dataset,
            sampler=val_sampler,
            batch_size=4,
            pin_memory=True,
            num_workers=2,
        )
    )

    total_loss = 0.0
    num_batches = 0

    for _ in range(min(50, len(val_dataset) // 4)):
        try:
            images, labels = next(val_iterator)
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)
            loss = loss_fn(
                net=ema_model, images=images, labels=labels, augment_pipe=None
            )
            total_loss += loss.mean().item()
            num_batches += 1
        except StopIteration:
            break

    return total_loss / max(num_batches, 1)


def apply_config_to_args(args, config: Dict[str, Any]):
    """Apply configuration values to argparse arguments. Priority: CLI > Config > Defaults."""
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

    for config_key, arg_name in config_mapping.items():
        current_value = getattr(args, arg_name, None)
        if current_value is None or current_value == []:
            config_value = config.get(config_key)
            if config_value is not None:
                setattr(args, arg_name, config_value)
            elif arg_name in defaults:
                setattr(args, arg_name, defaults[arg_name])


def setup_output_directory(output_dir):
    """Setup output directory."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    edm_path = project_root / "external" / "edm"
    if str(edm_path) not in sys.path:
        sys.path.insert(0, str(edm_path))

    from external.edm.torch_utils import distributed as dist

    if dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)


def setup_training_logging(run_dir):
    """Setup logging to file."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    edm_path = project_root / "external" / "edm"
    if str(edm_path) not in sys.path:
        sys.path.insert(0, str(edm_path))

    import external.edm.dnnlib
    from external.edm.torch_utils import distributed as dist

    global _logger_instance
    if dist.get_rank() == 0:
        log_file = os.path.join(run_dir, "log.txt")
        _logger_instance = external.edm.dnnlib.util.Logger(
            file_name=log_file, file_mode="a", should_flush=True
        )


def create_network_config(train_dataset, args):
    """Create network configuration."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    edm_path = project_root / "external" / "edm"
    if str(edm_path) not in sys.path:
        sys.path.insert(0, str(edm_path))

    from torch.utils.data import ConcatDataset

    import external.edm.dnnlib

    if isinstance(train_dataset, ConcatDataset):
        base_dataset = train_dataset.datasets[0]
        resolution = base_dataset.resolution
        num_channels = base_dataset.num_channels
        label_dim = base_dataset.label_dim
    else:
        resolution = train_dataset.resolution
        num_channels = train_dataset.num_channels
        label_dim = train_dataset.label_dim

    network_kwargs = external.edm.dnnlib.EasyDict(
        class_name="external.edm.training.networks.EDMPrecond",
        img_resolution=resolution,
        img_channels=num_channels,
        label_dim=label_dim,
        use_fp16=False,
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        model_channels=args.model_channels,
        channel_mult=args.channel_mult,
    )
    return network_kwargs


def create_loss_config():
    """Create loss configuration."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    edm_path = project_root / "external" / "edm"
    if str(edm_path) not in sys.path:
        sys.path.insert(0, str(edm_path))

    import external.edm.dnnlib

    loss_kwargs = external.edm.dnnlib.EasyDict(
        class_name="external.edm.training.loss.EDMLoss",
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
    )
    return loss_kwargs


def create_optimizer_config(args):
    """Create optimizer configuration."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    edm_path = project_root / "external" / "edm"
    if str(edm_path) not in sys.path:
        sys.path.insert(0, str(edm_path))

    import external.edm.dnnlib

    optimizer_kwargs = external.edm.dnnlib.EasyDict(
        class_name="torch.optim.Adam",
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer_kwargs


def load_checkpoint(checkpoint_path):
    """Load checkpoint from file."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    edm_path = project_root / "external" / "edm"
    if str(edm_path) not in sys.path:
        sys.path.insert(0, str(edm_path))

    from external.edm.torch_utils import distributed as dist

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
        return checkpoint_path
    return None


def load_edm_model(
    model_path: Union[str, Path], device: str = "cuda"
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load EDM model from pickle checkpoint file.

    This is a centralized utility to load EDM models, replacing duplicate
    model loading code across test files.

    Args:
        model_path: Path to .pkl checkpoint file
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Tuple of (model, checkpoint_dict)
        - model: The EMA model from checkpoint
        - checkpoint_dict: Full checkpoint dictionary
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading EDM model from {model_path}...")

    with open(model_path, "rb") as f:
        checkpoint = pickle.load(f)  # nosec B301

    if "ema" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain 'ema' key. Available keys: {checkpoint.keys()}"
        )

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = checkpoint["ema"].to(device_obj)
    model.eval()

    logger.info("✓ Model loaded successfully")
    logger.info(f"  Resolution: {model.img_resolution}")
    logger.info(f"  Channels: {model.img_channels}")
    logger.info(f"  Label dim: {model.label_dim}")
    logger.info(f"  Sigma range: [{model.sigma_min}, {model.sigma_max}]")

    return model, checkpoint


def create_domain_labels(
    domain: str,
    batch_size: int,
    label_dim: int,
    device: Union[str, torch.device] = "cuda",
) -> Optional[torch.Tensor]:
    """
    Create domain labels for sensor conditioning.

    This centralizes the domain label creation logic that was duplicated
    across multiple test files.

    Args:
        domain: Sensor name ('sony' or 'fuji')
        batch_size: Batch size for labels
        label_dim: Label dimension from model (0 for unconditional)
        device: Device for labels

    Returns:
        Tensor of shape (batch_size, label_dim) or None if label_dim == 0
    """
    if label_dim == 0:
        return None

    device_obj = torch.device(device) if isinstance(device, str) else device
    class_labels = torch.zeros(batch_size, label_dim, device=device_obj)

    # Set label based on sensor: sony=0, fuji=1 (or adjust based on model training)
    # Note: This assumes the model was trained with this labeling scheme
    if domain == "sony":
        class_labels[:, 0] = 1.0
    elif domain == "fuji":
        class_labels[:, 0] = 1.0  # Adjust if model uses different labels

    return class_labels
