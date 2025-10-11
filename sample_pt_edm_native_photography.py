#!/usr/bin/env python3
"""
Sample/inference script for EDM photography restoration with Poisson-Gaussian guidance.

This script:
1. Loads test images from metadata (noisy photography tiles)
2. Loads trained EDM model (best_model.pkl)
3. Applies Poisson-Gaussian guided sampling to denoise images
4. Saves reconstructed clean images

Key Features:
- Uses EDM's native sampler with physics-informed guidance
- Handles float32 .pt files in [-1, 1] range
- Applies domain-specific Poisson-Gaussian guidance
- Supports batch processing and visualization
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Add project root and EDM to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

# Import EDM components
import external.edm.dnnlib
from external.edm.torch_utils import distributed as dist
from inference.modules.initialization import create_initializer
from inference.validation.statistical_validator import (
    StatisticalValidator,
    validate_batch,
)
from poisson_gaussian_guidance import (
    PoissonGaussianGuidance,
    create_photography_guidance,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestDataLoader:
    """Load test images from metadata for sampling."""

    def __init__(
        self,
        data_root: str,
        metadata_json: str,
        domain: str = "photography",
        data_type: str = "noisy",  # We want to denoise noisy images
    ):
        self.data_root = Path(data_root)
        self.metadata_json = Path(metadata_json)
        self.domain = domain
        self.data_type = data_type

        # Load metadata and extract test tiles
        self.test_tiles = self._load_test_tiles()

        logger.info(
            f"Loaded {len(self.test_tiles)} test {data_type} tiles from metadata"
        )

    def _load_test_tiles(self) -> List[Dict[str, Any]]:
        """Load test split tiles from metadata."""
        with open(self.metadata_json, "r") as f:
            metadata = json.load(f)

        # Extract tiles
        all_tiles = metadata.get("tiles", [])

        # Filter by domain, data_type, and split
        test_tiles = [
            tile
            for tile in all_tiles
            if tile.get("domain") == self.domain
            and tile.get("data_type") == self.data_type
            and tile.get("split") == "test"
        ]

        if not test_tiles:
            logger.warning(f"No test {self.data_type} tiles found for {self.domain}")

        return test_tiles

    def __len__(self):
        return len(self.test_tiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a test tile."""
        tile_info = self.test_tiles[idx]

        # Load .pt file
        pt_path = Path(tile_info.get("pt_path"))
        if not pt_path.exists():
            # Try alternative paths
            tile_id = tile_info.get("tile_id")
            pt_path = self.data_root / self.domain / self.data_type / f"{tile_id}.pt"

        if not pt_path.exists():
            raise FileNotFoundError(f"Test tile not found: {pt_path}")

        # Load tensor (already in [-1, 1] range)
        tensor = torch.load(str(pt_path), map_location="cpu")

        # Ensure float32
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        return {
            "image": tensor,
            "tile_info": tile_info,
            "tile_id": tile_info.get("tile_id"),
            "scene_id": tile_info.get("scene_id"),
            "pt_path": str(pt_path),
        }


def edm_sampler_with_guidance(
    net,
    latents,
    guidance: PoissonGaussianGuidance,
    y_obs: torch.Tensor,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    """
    EDM sampler with Poisson-Gaussian guidance (Algorithm 2 + guidance).

    This integrates physics-informed guidance into EDM's deterministic sampler.
    At each timestep, we apply the guidance gradient to steer the denoising
    toward the observed noisy measurement.

    Args:
        net: EDM denoising network
        latents: Initial noise (B, C, H, W)
        guidance: PoissonGaussianGuidance module
        y_obs: Observed noisy image (B, C, H, W) in [-1, 1] range
        class_labels: Conditioning labels
        randn_like: Random noise generator
        num_steps: Number of sampling steps
        sigma_min, sigma_max: Noise level range
        rho: Time step discretization parameter
        S_churn, S_min, S_max, S_noise: Stochasticity parameters

    Returns:
        Denoised image (B, C, H, W)
    """
    # Adjust noise levels based on what's supported by the network
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop
    # For restoration: start from noisy observation (already in latents)
    # For generation: would multiply by t_steps[0] to add noise
    x_next = latents.to(torch.float64)  # Start from noisy image, not pure noise!

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily (stochasticity)
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step (denoising)
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)

        # Apply Poisson-Gaussian guidance IN PHYSICAL UNITS (if guidance is provided)
        if guidance is not None:
            # CRITICAL FIX: Convert from [-1,1] model space to physical units [0, max_ADU]
            # for proper variance calculation in guidance

            # Get domain range (s = max_ADU for full-well capacity)
            s = guidance.s.item()  # e.g., 15871 for photography

            # Convert denoised from [-1, 1] to [0, s] physical units
            # [-1, 1] -> [0, 1] -> [0, s]
            denoised_f32 = denoised.to(torch.float32)
            denoised_physical = (denoised_f32 + 1.0) / 2.0 * s

            # Convert y_obs from [-1, 1] to [0, s] physical units
            y_obs_f32 = y_obs.to(torch.float32)
            y_obs_physical = (y_obs_f32 + 1.0) / 2.0 * s

            # Apply guidance in physical space (electrons/ADU)
            with torch.enable_grad():
                denoised_guided_physical = guidance(
                    denoised_physical, y_obs_physical, sigma_t=t_hat.item()
                )

            # Convert back from [0, s] to [-1, 1] for model
            # [0, s] -> [0, 1] -> [-1, 1]
            denoised_guided = (denoised_guided_physical / s) * 2.0 - 1.0
            denoised = denoised_guided.to(torch.float64)

        # Compute update direction
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction (Heun's method)
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)

            # Apply guidance to 2nd order prediction as well (if guidance is provided)
            if guidance is not None:
                denoised_f32 = denoised.to(torch.float32)
                denoised_physical = (denoised_f32 + 1.0) / 2.0 * s

                with torch.enable_grad():
                    denoised_guided_physical = guidance(
                        denoised_physical, y_obs_physical, sigma_t=t_next.item()
                    )

                denoised_guided = (denoised_guided_physical / s) * 2.0 - 1.0
                denoised = denoised_guided.to(torch.float64)

            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def denormalize_to_physical(
    tensor: torch.Tensor, domain_range: Dict[str, float]
) -> np.ndarray:
    """
    Correct denormalization: [-1,1] → [0,1] → [domain_min, domain_max]
    Matches process_tiles_pipeline.py normalization (lines 624-635)

    Args:
        tensor: Image tensor in [-1, 1] range, shape (C, H, W)
        domain_range: Domain range dict with 'min' and 'max' values

    Returns:
        Image array in physical units (e.g., ADU for photography)
    """
    # Step 1: [-1,1] → [0,1]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)

    # Step 2: [0,1] → [domain_min, domain_max]
    domain_min = domain_range["min"]
    domain_max = domain_range["max"]
    tensor = tensor * (domain_max - domain_min) + domain_min

    # Convert to numpy
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    return tensor


def denormalize_for_display(
    tensor: torch.Tensor,
    domain_range: Dict[str, float],
) -> np.ndarray:
    """
    Convert tensor to display-ready uint8 [0,255] using per-image dynamic range.
    Uses percentile normalization (1-99%) like process_tiles_pipeline.py line 485-487.
    This makes all images visible while preserving detail.

    Args:
        tensor: Image tensor in [-1, 1] range, shape (C, H, W)
        domain_range: Domain range dict with 'min' and 'max' values

    Returns:
        Image array in [0, 255] range, shape (H, W, C) for RGB or (H, W) for grayscale
    """
    # Get physical values in native units (e.g., ADU)
    physical = denormalize_to_physical(tensor, domain_range)

    # Handle multi-channel data for display
    if physical.ndim == 3:
        if physical.shape[0] == 3:  # RGB (CHW)
            # Convert CHW to HWC for display
            display_data = np.transpose(physical, (1, 2, 0))

            # Apply percentile normalization per-channel (like preprocessing viz)
            normalized = np.zeros_like(display_data)
            for c in range(3):
                channel = display_data[:, :, c]
                valid_mask = np.isfinite(channel)
                if np.any(valid_mask):
                    p_low, p_high = np.percentile(channel[valid_mask], [1, 99])
                    clipped = np.clip(channel, p_low, p_high)
                    normalized[:, :, c] = (clipped - p_low) / (p_high - p_low + 1e-8)
            display = (normalized * 255).clip(0, 255).astype(np.uint8)
            return display

        elif physical.shape[0] == 1:  # Grayscale (1, H, W)
            display_data = physical[0]
        else:
            # Average channels for display
            display_data = np.mean(physical, axis=0)
    else:
        display_data = physical

    # Grayscale: Apply percentile normalization (like preprocessing viz)
    valid_mask = np.isfinite(display_data)
    if not np.any(valid_mask):
        return np.zeros(display_data.shape, dtype=np.uint8)

    p_low, p_high = np.percentile(display_data[valid_mask], [1, 99])
    clipped = np.clip(display_data, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low + 1e-8)

    # Convert to uint8
    display = (normalized * 255).clip(0, 255).astype(np.uint8)

    return display


def find_paired_clean_tile(
    noisy_tile_id: str,
    metadata: Dict[str, Any],
    data_root: Path,
) -> Optional[torch.Tensor]:
    """
    Find the paired clean (long exposure) tile for a given noisy (short exposure) tile.

    Photography pairing logic (from process_tiles_pipeline.py lines 1128-1233):
    - Noisy: short exposure (e.g., 0.1s)
    - Clean: long exposure (e.g., 10s)
    - Both from same scene_id

    Args:
        noisy_tile_id: Tile ID of noisy image (e.g., "photography_sony_00159_00_0.1s_tile_0000")
        metadata: Full metadata dict with all tiles
        data_root: Root directory for .pt files

    Returns:
        Clean tile tensor or None if not found
    """
    # Parse noisy tile to get scene_id and tile number
    # Example: "photography_sony_00159_00_0.1s_tile_0000"
    # Extract: scene_id="photo_00159", tile_num="0000"

    parts = noisy_tile_id.split("_")
    if len(parts) < 7:
        return None

    # scene_id is the 4th part (e.g., "00159")
    scene_num = parts[3]  # "00159"
    tile_num = parts[-1]  # "0000"
    camera_type = parts[2]  # "sony" or "fuji"

    # Search for matching clean tile in metadata
    all_tiles = metadata.get("tiles", [])

    for tile in all_tiles:
        tile_id = tile.get("tile_id", "")

        # Check if this is a clean tile from same scene and same tile position
        if (
            tile.get("data_type") == "clean"
            and tile.get("domain") == "photography"
            and scene_num in tile_id
            and tile_id.endswith(f"_tile_{tile_num}")
            and camera_type in tile_id
        ):
            # Load the clean tile
            clean_pt_path = Path(tile.get("pt_path"))
            if not clean_pt_path.exists():
                # Try alternative path
                clean_pt_path = data_root / "photography" / "clean" / f"{tile_id}.pt"

            if clean_pt_path.exists():
                try:
                    clean_tensor = torch.load(str(clean_pt_path), map_location="cpu")
                    if clean_tensor.dtype != torch.float32:
                        clean_tensor = clean_tensor.to(torch.float32)
                    return clean_tensor
                except Exception as e:
                    logger.debug(f"Failed to load clean tile {clean_pt_path}: {e}")
                    continue

    return None


def save_three_panel_comparison(
    noisy: torch.Tensor,
    clean: Optional[torch.Tensor],
    denoised: torch.Tensor,
    save_path: Path,
    domain_range: Dict[str, float],
    tile_id: str,
):
    """
    Create 3-panel comparison: Noisy | Clean (paired) | Denoised

    Statistics show NATIVE PHYSICAL VALUES (ADU units after domain scaling).
    Display uses per-image percentile normalization (1-99%) like preprocessing viz.
    This makes all images visible while showing true physical intensity values.

    Args:
        noisy: Noisy input tensor (C, H, W) in [-1, 1]
        clean: Clean ground truth tensor (C, H, W) in [-1, 1], or None if not available
        denoised: Denoised output tensor (C, H, W) in [-1, 1]
        save_path: Path to save the comparison image
        domain_range: Domain range for denormalization
        tile_id: Tile identifier for title
    """
    # Denormalize all consistently using native domain range scaling
    noisy_display = denormalize_for_display(noisy, domain_range)
    denoised_display = denormalize_for_display(denoised, domain_range)

    # Get physical statistics (in ADU)
    noisy_phys = denormalize_to_physical(noisy, domain_range)
    denoised_phys = denormalize_to_physical(denoised, domain_range)

    # Create figure
    if clean is not None:
        # 3-panel: Noisy | Clean | Denoised
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        clean_display = denormalize_for_display(clean, domain_range)
        clean_phys = denormalize_to_physical(clean, domain_range)

        # Panel 1: Noisy
        axes[0].imshow(noisy_display)
        axes[0].set_title(
            f"Noisy Input\n"
            f"ADU: [{noisy_phys.min():.0f}, {noisy_phys.max():.0f}]\n"
            f"Mean: {noisy_phys.mean():.1f}, Median: {np.median(noisy_phys):.1f}\n"
            f"(Display normalized per-image)",
            fontsize=9,
            fontweight="bold",
        )
        axes[0].axis("off")

        # Panel 2: Clean
        axes[1].imshow(clean_display)
        axes[1].set_title(
            f"Clean Ground Truth\n"
            f"ADU: [{clean_phys.min():.0f}, {clean_phys.max():.0f}]\n"
            f"Mean: {clean_phys.mean():.1f}, Median: {np.median(clean_phys):.1f}\n"
            f"(Display normalized per-image)",
            fontsize=9,
            fontweight="bold",
        )
        axes[1].axis("off")

        # Panel 3: Denoised
        axes[2].imshow(denoised_display)
        axes[2].set_title(
            f"Denoised Output\n"
            f"ADU: [{denoised_phys.min():.0f}, {denoised_phys.max():.0f}]\n"
            f"Mean: {denoised_phys.mean():.1f}, Median: {np.median(denoised_phys):.1f}\n"
            f"(Display normalized per-image)",
            fontsize=9,
            fontweight="bold",
        )
        axes[2].axis("off")

    else:
        # 2-panel fallback: Noisy | Denoised
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Panel 1: Noisy
        axes[0].imshow(noisy_display)
        axes[0].set_title(
            f"Noisy Input\n"
            f"ADU: [{noisy_phys.min():.0f}, {noisy_phys.max():.0f}]\n"
            f"Mean: {noisy_phys.mean():.1f}, Median: {np.median(noisy_phys):.1f}\n"
            f"(Display normalized per-image)",
            fontsize=9,
            fontweight="bold",
        )
        axes[0].axis("off")

        # Panel 2: Denoised
        axes[1].imshow(denoised_display)
        axes[1].set_title(
            f"Denoised Output\n"
            f"ADU: [{denoised_phys.min():.0f}, {denoised_phys.max():.0f}]\n"
            f"Mean: {denoised_phys.mean():.1f}, Median: {np.median(denoised_phys):.1f}\n"
            f"(Display normalized per-image)",
            fontsize=9,
            fontweight="bold",
        )
        axes[1].axis("off")

    # Add overall title with domain range info
    plt.suptitle(
        f"{tile_id}\n"
        f'Physical values in ADU (domain: [{domain_range["min"]:.0f}, {domain_range["max"]:.0f}])\n'
        f"Display: per-image 1-99 percentile normalization (like preprocessing viz)",
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(
        description="Sample/denoise test images using trained EDM model with Poisson-Gaussian guidance"
    )

    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/jilab/Jae/dataset/processed/pt_tiles",
        help="Root directory containing .pt tiles",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        default="/home/jilab/Jae/dataset/processed/metadata_photography_incremental.json",
        help="Metadata JSON file with test split info",
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/jilab/Jae/results/edm_pt_training_20251008_032055/best_model.pkl",
        help="Path to trained model checkpoint (.pkl)",
    )

    # Sampling arguments
    parser.add_argument(
        "--num_steps",
        type=int,
        default=18,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for sampling",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test samples to process (None = all)",
    )

    # Guidance arguments
    parser.add_argument(
        "--guidance_strength",
        type=float,
        default=0.5,
        help="Poisson-Gaussian guidance strength (kappa)",
    )
    parser.add_argument(
        "--guidance_tau",
        type=float,
        default=0.01,
        help="Guidance threshold (only apply when sigma_t > tau)",
    )
    parser.add_argument(
        "--guidance_mode",
        type=str,
        default="wls",
        choices=["wls", "full"],
        help="Guidance mode: 'wls' (weighted least squares) or 'full' (complete gradient)",
    )
    parser.add_argument(
        "--no_guidance",
        action="store_true",
        help="Disable Poisson-Gaussian guidance (vanilla EDM sampling)",
    )

    # Initialization arguments
    parser.add_argument(
        "--init_strategy",
        type=str,
        default="observation",
        choices=["pure_prior", "observation", "noise_matching"],
        help="Initialization strategy for diffusion sampling",
    )

    # Validation arguments
    parser.add_argument(
        "--validate_statistics",
        action="store_true",
        help="Compute χ² statistical validation for each denoised image",
    )
    parser.add_argument(
        "--save_validation_plots",
        action="store_true",
        help="Save diagnostic plots for statistical validation",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sampling_output",
        help="Output directory for denoised images",
    )
    parser.add_argument(
        "--save_comparisons",
        action="store_true",
        help="Save 3-panel comparison images (noisy | clean | denoised) - this is the only output",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("EDM SAMPLING WITH POISSON-GAUSSIAN GUIDANCE")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_root}")
    logger.info(f"Metadata: {args.metadata_json}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Sampling steps: {args.num_steps}")
    logger.info(f"Batch size: {args.batch_size}")
    if not args.no_guidance:
        logger.info(f"Guidance strength (kappa): {args.guidance_strength}")
        logger.info(f"Guidance threshold (tau): {args.guidance_tau}")
        logger.info(f"Guidance mode: {args.guidance_mode}")
    else:
        logger.info("Guidance: DISABLED (vanilla EDM)")
    logger.info(f"Initialization strategy: {args.init_strategy}")
    if args.validate_statistics:
        logger.info("Statistical validation: ENABLED")
        if args.save_validation_plots:
            logger.info("Validation plots: ENABLED")
    else:
        logger.info("Statistical validation: DISABLED")
    logger.info("=" * 80)

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    with open(args.model_path, "rb") as f:
        checkpoint = pickle.load(f)  # nosec B301

    net = checkpoint["ema"].to(device)
    net.eval()

    logger.info(f"✓ Model loaded: {net.__class__.__name__}")
    logger.info(f"  Resolution: {net.img_resolution}")
    logger.info(f"  Channels: {net.img_channels}")
    logger.info(f"  Label dim: {net.label_dim}")

    # Load test data
    logger.info("Loading test data...")
    test_loader = TestDataLoader(
        data_root=args.data_root,
        metadata_json=args.metadata_json,
        domain="photography",
        data_type="noisy",
    )

    if len(test_loader) == 0:
        logger.error("No test data found! Exiting.")
        return

    logger.info(f"✓ Loaded {len(test_loader)} test samples")

    # Load full metadata for guidance and clean tile pairing
    with open(args.metadata_json, "r") as f:
        full_metadata = json.load(f)

    # Get domain range from first tile
    domain_range = None
    for tile in full_metadata.get("tiles", []):
        if tile.get("domain") == "photography":
            domain_range = tile.get("domain_range", {"min": 0.0, "max": 15871.0})
            break

    if domain_range is None:
        domain_range = {"min": 0.0, "max": 15871.0}
        logger.warning(
            f"Domain range not found in metadata, using default: {domain_range}"
        )

    logger.info(f"Domain range: {domain_range}")
    logger.info(f"Total tiles in metadata: {len(full_metadata.get('tiles', []))}")

    # Create Poisson-Gaussian guidance
    guidance = None
    if not args.no_guidance:
        logger.info("Creating Poisson-Gaussian guidance...")
        guidance = create_photography_guidance(
            domain_range=domain_range,
            kappa=args.guidance_strength,
            tau=args.guidance_tau,
            mode=args.guidance_mode,
        ).to(device)
        logger.info("✓ Guidance module ready")

    # Create initialization strategy
    logger.info(f"Creating initialization strategy: {args.init_strategy}...")
    initializer = create_initializer(
        args.init_strategy, sigma_max=80.0
    )  # Use default sigma_max
    logger.info("✓ Initialization strategy ready")

    # Create statistical validator (if requested)
    validator = None
    if args.validate_statistics:
        logger.info("Creating statistical validator...")
        validation_dir = None
        if args.save_validation_plots:
            validation_dir = output_dir / "validation"
            validation_dir.mkdir(exist_ok=True)
        validator = StatisticalValidator(guidance, save_dir=validation_dir)
        logger.info("✓ Statistical validator ready")

    # Sample images with diversity across scenes/FOVs
    logger.info("\nStarting sampling...")
    logger.info("=" * 80)

    # Group tiles by scene_id to ensure diversity
    tiles_by_scene = {}
    for i in range(len(test_loader)):
        item = test_loader[i]
        scene_id = item.get("scene_id", "unknown")
        if scene_id not in tiles_by_scene:
            tiles_by_scene[scene_id] = []
        tiles_by_scene[scene_id].append(i)

    logger.info(f"Found {len(tiles_by_scene)} unique scenes/FOVs in test set")

    # Sample diverse tiles: take 1-2 tiles per scene for better diversity
    if args.max_samples:
        # Strategy: sample from different scenes
        sampled_indices = []
        scenes = list(tiles_by_scene.keys())
        np.random.shuffle(scenes)

        tiles_per_scene = (
            max(1, args.max_samples // len(scenes)) if len(scenes) > 0 else 1
        )
        for scene_id in scenes:
            scene_tiles = tiles_by_scene[scene_id]
            # Randomly sample tiles from this scene
            n_to_sample = min(tiles_per_scene, len(scene_tiles))
            sampled = np.random.choice(scene_tiles, size=n_to_sample, replace=False)
            sampled_indices.extend(sampled)
            if len(sampled_indices) >= args.max_samples:
                break

        sampled_indices = sampled_indices[: args.max_samples]
        logger.info(
            f"Sampling {len(sampled_indices)} tiles from {len(set([test_loader[i]['scene_id'] for i in sampled_indices]))} different scenes"
        )
    else:
        sampled_indices = list(range(len(test_loader)))
        logger.info(f"Processing all {len(sampled_indices)} test tiles")

    num_samples = len(sampled_indices)
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size

    total_processed = 0
    all_chi_squared = []
    consistent_count = 0

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Sampling", unit="batch"):
            # Get batch
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, num_samples)
            batch_size = batch_end - batch_start

            # Load batch data from sampled indices
            batch_items = [
                test_loader[sampled_indices[i]] for i in range(batch_start, batch_end)
            ]

            # Stack into batch
            noisy_images = torch.stack([item["image"] for item in batch_items]).to(
                device
            )
            tile_ids = [item["tile_id"] for item in batch_items]

            # Create class labels (domain encoding for photography)
            class_labels = None
            if net.label_dim > 0:
                # Photography: [1, 0, 0]
                class_labels = torch.zeros(batch_size, net.label_dim, device=device)
                class_labels[:, 0] = 1.0  # Photography domain

            # Initialize latents using selected strategy
            logger.debug(f"Initializing latents with strategy: {args.init_strategy}")
            latents = initializer.initialize(noisy_images, guidance)

            # Sample with or without guidance
            # Note: Always use edm_sampler_with_guidance because the vanilla edm_sampler
            # multiplies latents by sigma_max which is for GENERATION from pure noise,
            # not RESTORATION from noisy observations
            denoised_images = edm_sampler_with_guidance(
                net=net,
                latents=latents,
                guidance=guidance,  # None for vanilla, PoissonGaussianGuidance for guided
                y_obs=noisy_images,
                class_labels=class_labels,
                num_steps=args.num_steps,
            )

            # Check for NaN or Inf in output
            if torch.isnan(denoised_images).any() or torch.isinf(denoised_images).any():
                logger.warning(
                    f"⚠️ Batch {batch_idx}: Denoised images contain NaN or Inf values!"
                )
                logger.warning(
                    f"   NaN count: {torch.isnan(denoised_images).sum().item()}"
                )
                logger.warning(
                    f"   Inf count: {torch.isinf(denoised_images).sum().item()}"
                )
                logger.warning(
                    f"   Value range: [{denoised_images[torch.isfinite(denoised_images)].min():.4f}, {denoised_images[torch.isfinite(denoised_images)].max():.4f}]"
                )
                # Replace NaN/Inf with clamped values
                denoised_images = torch.nan_to_num(
                    denoised_images, nan=0.0, posinf=1.0, neginf=-1.0
                )
                denoised_images = torch.clamp(denoised_images, -1.0, 1.0)

            # Validate and save results
            for i in range(batch_size):
                tile_id = tile_ids[i]
                noisy = noisy_images[i]
                denoised = denoised_images[i]

                # Log value ranges for debugging
                logger.debug(f"Tile {tile_id}:")
                logger.debug(f"  Noisy range: [{noisy.min():.4f}, {noisy.max():.4f}]")
                logger.debug(
                    f"  Denoised range: [{denoised.min():.4f}, {denoised.max():.4f}]"
                )

                # Statistical validation
                if validator is not None:
                    val_results, is_consistent = validator.validate(
                        noisy.cpu(), denoised.cpu(), save_prefix=tile_id
                    )
                    chi2 = val_results["chi_squared"]
                    all_chi_squared.append(chi2)
                    if is_consistent:
                        consistent_count += 1
                    status = "✓" if is_consistent else "✗"
                    logger.info(f"  χ² = {chi2:.3f} {status}")

                # Save 3-panel comparison (skip individual denoised.png)
                if args.save_comparisons:
                    comparison_path = output_dir / f"{tile_id}_comparison.png"

                    # Try to find paired clean tile
                    clean_tile = find_paired_clean_tile(
                        tile_id, full_metadata, Path(args.data_root)
                    )

                    # Create 3-panel comparison (or 2-panel if clean not found)
                    save_three_panel_comparison(
                        noisy=noisy.cpu(),
                        clean=clean_tile,
                        denoised=denoised.cpu(),
                        save_path=comparison_path,
                        domain_range=domain_range,
                        tile_id=tile_id,
                    )

                total_processed += 1

    logger.info("=" * 80)
    logger.info(f"✅ Sampling completed!")
    logger.info(f"Total samples processed: {total_processed}")
    if args.save_comparisons:
        logger.info(f"3-panel comparison images saved to: {output_dir}")
    else:
        logger.info(f"No comparisons saved (use --save_comparisons to enable)")

    # Statistical validation summary
    if args.validate_statistics and len(all_chi_squared) > 0:
        import numpy as np

        mean_chi2 = np.mean(all_chi_squared)
        std_chi2 = np.std(all_chi_squared)
        consistency_rate = 100.0 * consistent_count / total_processed

        logger.info(f"\nStatistical Validation Summary:")
        logger.info(f"  Mean χ²: {mean_chi2:.3f} ± {std_chi2:.3f}")
        logger.info(f"  Target: 1.0 (range: 0.8-1.2)")

        if 0.9 < mean_chi2 < 1.1:
            status = "✓ EXCELLENT"
        elif 0.8 < mean_chi2 < 1.2:
            status = "✓ GOOD"
        else:
            status = "⚠ NEEDS TUNING"

        logger.info(f"  Status: {status}")
        logger.info(f"  Consistency rate: {consistency_rate:.1f}%")
        logger.info(f"  Target: >90%")

        if args.save_validation_plots:
            logger.info(f"  Validation plots: {output_dir}/validation/")

        # Save summary to file
        summary = {
            "config": {
                "init_strategy": args.init_strategy,
                "guidance_strength": args.guidance_strength,
                "num_steps": args.num_steps,
            },
            "num_processed": total_processed,
            "validation": {
                "mean_chi_squared": float(mean_chi2),
                "std_chi_squared": float(std_chi2),
                "consistency_rate": float(consistency_rate),
                "all_chi_squared": [float(x) for x in all_chi_squared],
            },
        }

        summary_path = output_dir / "summary.json"
        import json

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Summary saved: {summary_path}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
