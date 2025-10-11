#!/usr/bin/env python3
"""
EDM Photography LLIE with Poisson-Gaussian Guidance and Step-by-Step Visualizations

This script demonstrates EDM-based low-light image enhancement by denoising noisy observations.
Starting from actual low-light noisy images, it applies the learned denoising prior to
progressively enhance them while preserving scene content.

Key Features:
- Starts from actual noisy low-light observations (not random noise)
- Applies learned EDM denoising prior to enhance images
- Optional: Physics-informed Poisson-Gaussian guidance for noise-aware processing
- Shows complete enhancement process with step-by-step visualization
- Handles domain-specific normalization and denormalization
- Creates comprehensive comparison panels showing enhancement progress

Approach:
- Model trained on: Both clean (long exposure) and noisy (short exposure) images
- At inference: Start from noisy observation, progressively denoise/enhance
- Optional guidance: PG-guidance for physics-aware noise handling

Usage:
    python sample_pt_edm_photography_llie_pg_guided.py \
        --model_path results/edm_pt_training_20251008_032055/best_model.pkl \
        --data_root dataset/processed/pt_tiles \
        --metadata_json dataset/processed/metadata_photography_incremental.json \
        --output_dir results/edm_photography_llie_pg_guided \
        --domain photography \
        --num_samples 2 \
        --use_guidance \
        --guidance_scale 15871.0 \
        --guidance_sigma_r 10.0 \
        --guidance_kappa 0.5
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root and EDM to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
edm_path = project_root / "external" / "edm"
sys.path.insert(0, str(edm_path))

# Setup logging
import logging

# Import EDM components
import external.edm.dnnlib
from external.edm.torch_utils import distributed as dist

# Import PG-guidance
from poisson_gaussian_guidance import PoissonGaussianGuidance

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestTileLoader:
    """Load test tiles from the processed dataset for low-light image enhancement."""

    def __init__(
        self,
        data_root: str,
        metadata_json: str,
        domain: str = "photography",
        data_type: str = "clean",  # Can be "noisy" or "clean"
    ):
        self.data_root = Path(data_root)
        self.metadata_json = Path(metadata_json)
        self.domain = domain
        self.data_type = data_type

        # Load metadata and extract test tiles
        self.test_tiles = self._load_test_tiles()
        self.domain_ranges = self._get_domain_ranges()

        logger.info(
            f"Loaded {len(self.test_tiles)} test {data_type} tiles for {domain} processing"
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

    def _get_domain_ranges(self) -> Dict[str, Dict[str, float]]:
        """Get domain-specific normalization ranges."""
        with open(self.metadata_json, "r") as f:
            metadata = json.load(f)

        # Find domain range from any tile of this domain
        for tile in metadata.get("tiles", []):
            if tile.get("domain") == self.domain:
                return tile.get("domain_range", {"min": 0.0, "max": 15871.0})

        # Default fallback
        return {"min": 0.0, "max": 15871.0}

    def __len__(self):
        return len(self.test_tiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a test tile for restoration."""
        tile_info = self.test_tiles[idx]

        # Load .pt file (already in [-1, 1] range from preprocessing)
        pt_path = Path(tile_info.get("pt_path"))
        if not pt_path.exists():
            # Try alternative paths
            tile_id = tile_info.get("tile_id")
            pt_path = self.data_root / self.domain / self.data_type / f"{tile_id}.pt"

        if not pt_path.exists():
            raise FileNotFoundError(f"Test tile not found: {pt_path}")

        # Load tensor (already in [-1, 1] range from preprocessing pipeline)
        tensor = torch.load(str(pt_path), map_location="cpu")

        # Ensure float32 and proper shape
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        return {
            "image": tensor,
            "tile_info": tile_info,
            "tile_id": tile_info.get("tile_id"),
            "scene_id": tile_info.get("scene_id"),
            "pt_path": str(pt_path),
            "domain_range": tile_info.get("domain_range", self.domain_ranges),
        }

    def get_sample_tiles(
        self, num_samples: int, seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Get a diverse set of sample tiles across different scenes."""
        np.random.seed(seed)

        # Group tiles by scene for diversity
        tiles_by_scene = {}
        for i in range(len(self)):
            item = self[i]
            scene_id = item.get("scene_id", "unknown")
            if scene_id not in tiles_by_scene:
                tiles_by_scene[scene_id] = []
            tiles_by_scene[scene_id].append(i)

        # Sample diverse tiles: take 1-2 tiles per scene
        sampled_indices = []
        scenes = list(tiles_by_scene.keys())
        np.random.shuffle(scenes)

        tiles_per_scene = max(1, num_samples // len(scenes)) if len(scenes) > 0 else 1
        for scene_id in scenes:
            scene_tiles = tiles_by_scene[scene_id]
            # Randomly sample tiles from this scene
            n_to_sample = min(tiles_per_scene, len(scene_tiles))
            sampled = np.random.choice(scene_tiles, size=n_to_sample, replace=False)
            sampled_indices.extend(sampled)
            if len(sampled_indices) >= num_samples:
                break

        sampled_indices = sampled_indices[:num_samples]

        # Load the selected tiles
        samples = []
        for idx in sampled_indices:
            try:
                samples.append(self[idx])
            except Exception as e:
                logger.warning(f"Failed to load sample {idx}: {e}")
                continue

        logger.info(
            f"Selected {len(samples)} diverse tiles from {len(set([s['scene_id'] for s in samples]))} different scenes"
        )
        return samples


class ComprehensiveEDMLLIE:
    """
    Comprehensive EDM low-light image enhancement with Poisson-Gaussian guidance
    and step-by-step visualization capabilities.

    This enhancement process starts from real test images and shows the
    complete enhancement process with physics-informed guidance and detailed visualizations.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_guidance: bool = False,
        guidance_scale: float = 15871.0,
        guidance_sigma_r: float = 10.0,
        guidance_kappa: float = 0.5,
        guidance_tau: float = 0.01,
        guidance_mode: str = "wls",
    ):
        """
        Initialize the restoration system with trained model and optional PG-guidance.

        Args:
            model_path: Path to trained model checkpoint
            device: Device for computation
            use_guidance: Whether to use Poisson-Gaussian guidance
            guidance_scale: Scale factor for PG-guidance (typically domain max value)
            guidance_sigma_r: Read noise standard deviation for PG-guidance
            guidance_kappa: Guidance strength multiplier
            guidance_tau: Guidance threshold - only apply when sigma_t > tau
            guidance_mode: Guidance mode ('wls' or 'full')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_guidance = use_guidance

        # Load model
        logger.info(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)  # nosec B301

        self.net = checkpoint["ema"].to(self.device)
        self.net.eval()

        logger.info("✓ Model loaded successfully")
        logger.info(f"  Resolution: {self.net.img_resolution}")
        logger.info(f"  Channels: {self.net.img_channels}")
        logger.info(f"  Label dim: {self.net.label_dim}")
        logger.info(f"  Sigma range: [{self.net.sigma_min}, {self.net.sigma_max}]")

        # Initialize PG-guidance if requested
        self.guidance = None
        if use_guidance:
            self.guidance = PoissonGaussianGuidance(
                s=guidance_scale,
                sigma_r=guidance_sigma_r,
                kappa=guidance_kappa,
                tau=guidance_tau,
                mode=guidance_mode,
            ).to(self.device)
            logger.info("✓ Poisson-Gaussian guidance enabled")
            logger.info(f"  Scale (s): {guidance_scale}")
            logger.info(f"  Read noise (σ_r): {guidance_sigma_r}")
            logger.info(f"  Kappa (κ): {guidance_kappa}")
            logger.info(f"  Tau (τ): {guidance_tau}")
            logger.info(f"  Mode: {guidance_mode}")
        else:
            logger.info("✓ No guidance (unconditional sampling)")

    def enhance_with_visualization(
        self,
        noisy_image: torch.Tensor,
        domain_range: Dict[str, float],
        domain: str = "photography",
        class_labels: Optional[torch.Tensor] = None,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
        save_steps: bool = True,
        output_dir: Optional[Path] = None,
        tile_id: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Perform EDM low-light image enhancement by denoising the input observation.

        This starts from the actual noisy low-light image and applies the learned
        denoising prior to progressively enhance it. The noisy image contains the actual
        scene content and structure, just underexposed and noisy.

        Args:
            noisy_image: Noisy observation tensor in [-1, 1] range (starting point)
            domain_range: Domain normalization range
            domain: Domain name for conditioning
            class_labels: Class conditioning labels
            num_steps: Number of enhancement steps
            sigma_min, sigma_max: Noise level range
            rho: Time step discretization parameter
            S_churn, S_min, S_max, S_noise: Stochasticity parameters
            save_steps: Whether to save intermediate steps
            output_dir: Directory to save step visualizations
            tile_id: Tile identifier for naming

        Returns:
            Dictionary containing all enhancement results and intermediate steps
        """
        logger.info(f"Starting low-light enhancement of {tile_id}: {num_steps} steps")

        # Ensure noisy image is on correct device and shape
        noisy_image = (
            noisy_image.to(self.device).unsqueeze(0)
            if noisy_image.ndim == 3
            else noisy_image.to(self.device)
        )

        # Adjust noise levels based on network capabilities
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        logger.info(
            f"✓ Starting from noisy observation: shape {noisy_image.shape}, range [{noisy_image.min():.3f}, {noisy_image.max():.3f}]"
        )
        logger.info(
            f"✓ Will progressively enhance the noisy image using learned denoising prior"
        )

        # Time step discretization
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=self.device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Storage for intermediate results
        step_results = []
        guidance_norms = []

        # Main enhancement loop - START FROM NOISY IMAGE for LLIE!
        # For LLIE: We enhance the actual noisy observation, not sample from prior
        # The noisy image contains the actual scene content, just underexposed
        x_next = (
            noisy_image.to(torch.float64) * t_steps[0]
        )  # Start from noisy observation scaled by sigma

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            # Increase noise temporarily (stochasticity)
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step (denoising)
            with torch.no_grad():
                denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)

            # Validate denoised output
            if torch.isnan(denoised).any() or torch.isinf(denoised).any():
                logger.warning(
                    f"NaN/Inf detected in denoised output at step {i}, clamping values"
                )
                denoised = torch.nan_to_num(denoised, nan=0.0, posinf=1.0, neginf=-1.0)
                denoised = torch.clamp(denoised, -1.0, 1.0)

            # Apply Poisson-Gaussian guidance to denoised prediction
            guided_denoised = denoised
            guidance_norm = 0.0
            if self.use_guidance and self.guidance is not None:
                # Apply guidance to the denoised prediction
                with torch.no_grad():
                    guided_denoised = self.guidance(
                        denoised.to(torch.float32),
                        noisy_image.to(
                            torch.float32
                        ),  # Use original noisy image as observation
                        t_hat.item(),
                    ).to(torch.float64)

                    # Compute guidance magnitude for diagnostics
                    guidance_correction = guided_denoised - denoised
                    guidance_norm = torch.norm(guidance_correction).item()
                    guidance_norms.append(guidance_norm)

                    logger.debug(
                        f"Step {i}: Applied PG-guidance, norm={guidance_norm:.6f}"
                    )

            # Compute update direction using guided denoised prediction
            d_cur = (x_hat - guided_denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction (Heun's method)
            if i < num_steps - 1:
                with torch.no_grad():
                    denoised_next = self.net(x_next, t_next, class_labels).to(
                        torch.float64
                    )

                # Validate second denoised output
                if torch.isnan(denoised_next).any() or torch.isinf(denoised_next).any():
                    logger.warning(
                        f"NaN/Inf detected in second denoised output at step {i}, clamping values"
                    )
                    denoised_next = torch.nan_to_num(
                        denoised_next, nan=0.0, posinf=1.0, neginf=-1.0
                    )
                    denoised_next = torch.clamp(denoised_next, -1.0, 1.0)

                # Apply guidance to second-order prediction as well
                guided_denoised_next = denoised_next
                if self.use_guidance and self.guidance is not None:
                    with torch.no_grad():
                        guided_denoised_next = self.guidance(
                            denoised_next.to(torch.float32),
                            noisy_image.to(torch.float32),
                            t_next.item(),
                        ).to(torch.float64)

                d_prime = (x_next - guided_denoised_next) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            # Store intermediate results for visualization
            if save_steps:
                # Validate tensor shapes and ranges for debugging
                logger.debug(
                    f"Step {i}: x_hat shape={x_hat.shape}, range=[{x_hat.min():.3f}, {x_hat.max():.3f}]"
                )
                logger.debug(
                    f"Step {i}: denoised shape={denoised.shape}, range=[{denoised.min():.3f}, {denoised.max():.3f}]"
                )
                logger.debug(
                    f"Step {i}: x_next shape={x_next.shape}, range=[{x_next.min():.3f}, {x_next.max():.3f}]"
                )

                step_data = {
                    "step": i,
                    "t_cur": t_cur.item(),
                    "t_next": t_next.item(),
                    "x_hat": x_hat.clone(),
                    "denoised": denoised.clone(),
                    "guided_denoised": guided_denoised.clone()
                    if self.use_guidance
                    else denoised.clone(),
                    "x_next": x_next.clone(),
                    "noise_level": t_cur.item(),
                    "guidance_norm": guidance_norm,
                }
                step_results.append(step_data)

                # Save visualization for this step
                if output_dir is not None:
                    self._save_step_visualization(
                        step_data, i, output_dir, domain_range, domain
                    )

        logger.info(
            f"✓ Enhancement completed. Final range: [{x_next.min():.3f}, {x_next.max():.3f}]"
        )

        if self.use_guidance and len(guidance_norms) > 0:
            logger.info(
                f"  Guidance statistics: mean={np.mean(guidance_norms):.6f}, max={np.max(guidance_norms):.6f}"
            )

        return {
            "final_enhanced": x_next,
            "initial_noisy": noisy_image,
            "step_results": step_results,
            "guidance_norms": guidance_norms,
            "class_labels": class_labels,
            "tile_id": tile_id,
            "domain_range": domain_range,
            "enhancement_params": {
                "num_steps": num_steps,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
                "use_guidance": self.use_guidance,
            },
        }

    def _save_step_visualization(
        self,
        step_data: Dict[str, Any],
        step_idx: int,
        output_dir: Path,
        domain_range: Dict[str, float],
        domain: str,
    ):
        """Save visualization for a single enhancement step."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Convert tensors to numpy for visualization
            x_hat = step_data["x_hat"].cpu().numpy()
            denoised = step_data["denoised"].cpu().numpy()
            x_next = step_data["x_next"].cpu().numpy()

            # Handle NaN/inf values that might corrupt visualization
            x_hat = np.nan_to_num(x_hat, nan=0.0, posinf=1.0, neginf=-1.0)
            denoised = np.nan_to_num(denoised, nan=0.0, posinf=1.0, neginf=-1.0)
            x_next = np.nan_to_num(x_next, nan=0.0, posinf=1.0, neginf=-1.0)

            logger.debug(
                f"Step {step_idx} visualization: x_hat shape={x_hat.shape}, denoised shape={denoised.shape}, x_next shape={x_next.shape}"
            )

            # Show first channel/first sample for each - handle RGB properly
            try:
                if x_hat.ndim == 4:  # (batch, channels, height, width)
                    if x_hat.shape[1] == 3:  # RGB - show as RGB
                        x_hat_vis = np.transpose(x_hat[0], (1, 2, 0))  # Convert to HWC
                        denoised_vis = np.transpose(denoised[0], (1, 2, 0))
                        x_next_vis = np.transpose(x_next[0], (1, 2, 0))
                    elif x_hat.shape[1] == 1:  # Grayscale
                        x_hat_vis = x_hat[0, 0]
                        denoised_vis = denoised[0, 0]
                        x_next_vis = x_next[0, 0]
                    else:  # Multi-channel - average
                        x_hat_vis = np.mean(x_hat[0], axis=0)
                        denoised_vis = np.mean(denoised[0], axis=0)
                        x_next_vis = np.mean(x_next[0], axis=0)
                elif x_hat.ndim == 3:  # (channels, height, width)
                    if x_hat.shape[0] == 3:  # RGB
                        x_hat_vis = np.transpose(x_hat, (1, 2, 0))
                        denoised_vis = np.transpose(denoised, (1, 2, 0))
                        x_next_vis = np.transpose(x_next, (1, 2, 0))
                    else:  # Other
                        x_hat_vis = x_hat[0] if x_hat.shape[0] > 0 else x_hat
                        denoised_vis = (
                            denoised[0] if denoised.shape[0] > 0 else denoised
                        )
                        x_next_vis = x_next[0] if x_next.shape[0] > 0 else x_next
                else:  # 2D
                    x_hat_vis = x_hat
                    denoised_vis = denoised
                    x_next_vis = x_next

                # Ensure we have valid 2D arrays for visualization
                for name, vis_data in [
                    ("x_hat", x_hat_vis),
                    ("denoised", denoised_vis),
                    ("x_next", x_next_vis),
                ]:
                    if vis_data.ndim > 2:
                        # For multi-dimensional arrays, take mean or first channel
                        if vis_data.ndim == 3 and vis_data.shape[2] > 1:
                            # RGB case - already handled above
                            pass
                        else:
                            vis_data = (
                                np.mean(vis_data, axis=0)
                                if vis_data.ndim > 2
                                else vis_data
                            )
                            if name == "x_hat":
                                x_hat_vis = vis_data
                            elif name == "denoised":
                                denoised_vis = vis_data
                            elif name == "x_next":
                                x_next_vis = vis_data

                # Final check - ensure 2D
                x_hat_vis = x_hat_vis.squeeze() if x_hat_vis.ndim > 2 else x_hat_vis
                denoised_vis = (
                    denoised_vis.squeeze() if denoised_vis.ndim > 2 else denoised_vis
                )
                x_next_vis = x_next_vis.squeeze() if x_next_vis.ndim > 2 else x_next_vis

            except Exception as e:
                logger.warning(
                    f"Error processing visualization data for step {step_idx}: {e}"
                )
                # Fallback to simple 2D representation
                x_hat_vis = np.ones((256, 256)) * 0.5
                denoised_vis = np.ones((256, 256)) * 0.5
                x_next_vis = np.ones((256, 256)) * 0.5

            # Better normalization for display - use percentile-based for extreme values
            def normalize_for_display(img):
                # Use percentile clipping for better visualization of extreme values
                if img.ndim == 2:
                    valid_mask = np.isfinite(img)
                    if np.any(valid_mask):
                        p_low, p_high = np.percentile(img[valid_mask], [1, 99])
                        img = np.clip(img, p_low, p_high)
                else:  # RGB - normalize each channel
                    for c in range(img.shape[2]):
                        channel = img[:, :, c]
                        valid_mask = np.isfinite(channel)
                        if np.any(valid_mask):
                            p_low, p_high = np.percentile(channel[valid_mask], [1, 99])
                            img[:, :, c] = np.clip(channel, p_low, p_high)

                # Normalize to [0, 1]
                if img.ndim == 2:
                    return (img - img.min()) / (img.max() - img.min() + 1e-8)
                else:  # RGB
                    result = np.zeros_like(img, dtype=np.float32)
                    for c in range(img.shape[2]):
                        channel = img[:, :, c]
                        result[:, :, c] = (channel - channel.min()) / (
                            channel.max() - channel.min() + 1e-8
                        )
                    return result

            # Plot each stage
            if x_hat_vis.ndim == 3:  # RGB
                axes[0].imshow(np.clip(normalize_for_display(x_hat_vis), 0, 1))
            else:  # Grayscale
                axes[0].imshow(
                    normalize_for_display(x_hat_vis), cmap="gray", vmin=0, vmax=1
                )
            axes[0].set_title(
                f'Step {step_idx}: Noisy Input\nσ = {step_data["noise_level"]:.3f}'
            )
            axes[0].axis("off")

            if denoised_vis.ndim == 3:  # RGB
                axes[1].imshow(np.clip(normalize_for_display(denoised_vis), 0, 1))
            else:  # Grayscale
                axes[1].imshow(
                    normalize_for_display(denoised_vis), cmap="gray", vmin=0, vmax=1
                )
            axes[1].set_title(f"Step {step_idx}: Denoised Prediction")
            axes[1].axis("off")

            if x_next_vis.ndim == 3:  # RGB
                axes[2].imshow(np.clip(normalize_for_display(x_next_vis), 0, 1))
            else:  # Grayscale
                axes[2].imshow(
                    normalize_for_display(x_next_vis), cmap="gray", vmin=0, vmax=1
                )
            axes[2].set_title(f"Step {step_idx}: Updated Sample")
            axes[2].axis("off")

            plt.tight_layout()

            # Save step visualization
            step_dir = output_dir / "steps"
            step_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                step_dir / f"step_{step_idx:03d}.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            logger.warning(f"Failed to save step visualization {step_idx}: {e}")

    def denormalize_to_physical(
        self, tensor: torch.Tensor, domain_range: Dict[str, float]
    ) -> np.ndarray:
        """
        Convert tensor from [-1,1] model space to physical units.

        Args:
            tensor: Image tensor in [-1, 1] range, shape (B, C, H, W)
            domain_range: Domain range dict with 'min' and 'max' values

        Returns:
            Image array in physical units
        """
        # Step 1: [-1,1] → [0,1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)

        # Step 2: [0,1] → [domain_min, domain_max]
        domain_min = domain_range["min"]
        domain_max = domain_range["max"]
        tensor = tensor * (domain_max - domain_min) + domain_min

        return tensor.cpu().numpy()

    def normalize_tensor_for_display(
        self, tensor_data, domain, percentile_range=(1, 99)
    ):
        """
        Normalize tensor data for display using percentile clipping (like original pipeline).
        This handles tensor data in [-1, 1] range and applies display normalization.

        Args:
            tensor_data: Tensor data in [-1, 1] range, shape (C, H, W) or (B, C, H, W)
            domain: Domain name for proper handling
            percentile_range: Percentile range for clipping

        Returns:
            Normalized display data in [0, 1] range
        """
        if tensor_data is None:
            return None

        # Handle batch dimension
        if tensor_data.ndim == 4:  # (B, C, H, W)
            # Take first sample
            tensor_data = tensor_data[0]

        # Handle different channel configurations
        if tensor_data.ndim == 3:
            if domain == "photography" and tensor_data.shape[0] == 3:  # RGB
                # For RGB, preserve color by converting CHW to HWC and normalizing together
                display_data = np.transpose(tensor_data, (1, 2, 0))  # CHW -> HWC
            elif tensor_data.shape[0] == 1:  # Grayscale
                display_data = tensor_data[0]
            else:  # Multi-channel - average
                display_data = np.mean(tensor_data, axis=0)
        else:
            display_data = tensor_data

        # Ensure we have valid data for display (2D for grayscale, 3D for RGB)
        if display_data.ndim not in [2, 3]:
            logger.warning(
                f"Expected 2D or 3D data for display, got shape {display_data.shape}"
            )
            return None

        # Remove NaN and infinite values
        valid_mask = np.isfinite(display_data)
        if not np.any(valid_mask):
            return None

        # Clip to percentiles and normalize to [0, 1] - normalize all channels together for RGB
        p_low, p_high = np.percentile(display_data[valid_mask], percentile_range)
        clipped = np.clip(display_data, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low + 1e-8)

        return normalized

    def create_display_visualization(
        self,
        tensor: torch.Tensor,
        domain_range: Dict[str, float],
        domain: str,
        title: str = "Image",
        save_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Create display-ready visualization using percentile normalization.

        Args:
            tensor: Image tensor in [-1, 1] range
            domain_range: Domain range for physical units conversion
            domain: Domain name
            title: Plot title
            save_path: Optional path to save the visualization

        Returns:
            Display-ready uint8 image array
        """
        # Convert to physical units first
        physical = self.denormalize_to_physical(tensor, domain_range)

        # Handle multi-channel data
        if physical.ndim == 4:  # (B, C, H, W)
            if physical.shape[1] == 3:  # RGB
                display_data = np.transpose(physical[0], (1, 2, 0))  # HWC
            elif physical.shape[1] == 1:  # Grayscale
                display_data = physical[0, 0]  # HW
            else:
                display_data = np.mean(physical[0], axis=0)  # Average channels
        elif physical.ndim == 3:  # (C, H, W)
            if physical.shape[0] == 3:  # RGB
                display_data = np.transpose(physical, (1, 2, 0))  # HWC
            elif physical.shape[0] == 1:  # Grayscale
                display_data = physical[0]  # HW
            else:
                display_data = np.mean(physical, axis=0)  # Average channels
        else:  # (H, W)
            display_data = physical

        # Apply percentile normalization (1-99%)
        valid_mask = np.isfinite(display_data)
        if np.any(valid_mask):
            p_low, p_high = np.percentile(display_data[valid_mask], [1, 99])
            clipped = np.clip(display_data, p_low, p_high)
            normalized = (clipped - p_low) / (p_high - p_low + 1e-8)
            display = (normalized * 255).clip(0, 255).astype(np.uint8)
        else:
            display = np.zeros_like(display_data, dtype=np.uint8)

        # Save if requested
        if save_path is not None:
            plt.figure(figsize=(8, 8))
            plt.imshow(display, cmap="gray" if display.ndim == 2 else None)
            plt.title(
                f"{title}\nDomain: {domain}\nPhysical range: [{display_data.min():.1f}, {display_data.max():.1f}]"
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

        return display

    def create_comprehensive_comparison(
        self,
        initial_noisy: torch.Tensor,
        final_restored: torch.Tensor,
        step_results: List[Dict[str, Any]],
        domain_range: Dict[str, float],
        domain: str,
        tile_id: str,
        save_path: Path,
    ):
        """
        Create comprehensive comparison visualization showing the restoration process.

        Args:
            initial_noisy: Initial noisy image tensor
            final_restored: Final restored image tensor
            step_results: List of intermediate step results
            domain_range: Domain normalization range
            domain: Domain name
            tile_id: Tile identifier
            save_path: Path to save the comparison
        """
        logger.info("Creating comprehensive enhancement comparison visualization...")

        # Create figure with multiple panels
        n_steps_to_show = min(5, len(step_results))  # Show first few + last few steps
        step_indices = (
            [0, 1, 2] + [len(step_results) - 2, len(step_results) - 1]
            if len(step_results) > 3
            else list(range(len(step_results)))
        )

        fig, axes = plt.subplots(
            3, len(step_indices) + 2, figsize=(4 * (len(step_indices) + 2), 12)
        )

        # Row 1: Physical pixel values (domain scaling) - Noisy Input
        noisy_physical = self.denormalize_to_physical(initial_noisy, domain_range)
        noisy_min, noisy_max = float(noisy_physical.min()), float(noisy_physical.max())
        self._add_panel_physical_rgb(
            axes[0, 0],
            noisy_physical,
            "Noisy Input",
            domain,
            domain_range,
            show_range=True,
            pixel_min=noisy_min,
            pixel_max=noisy_max,
        )

        # Row 1: Intermediate steps in physical units
        for i, step_idx in enumerate(step_indices):
            if step_idx < len(step_results):
                step_data = step_results[step_idx]
                step_physical = self.denormalize_to_physical(
                    step_data["x_next"], domain_range
                )
                self._add_panel_physical_rgb(
                    axes[0, i + 1],
                    step_physical,
                    f"Step {step_idx}",
                    domain,
                    domain_range,
                    show_range=False,
                )

        # Row 1: Final restored in physical units
        final_physical = self.denormalize_to_physical(final_restored, domain_range)
        final_min, final_max = float(final_physical.min()), float(final_physical.max())
        self._add_panel_physical_rgb(
            axes[0, -1],
            final_physical,
            "Final Enhanced",
            domain,
            domain_range,
            show_range=True,
            pixel_min=final_min,
            pixel_max=final_max,
        )

        # Row 2: Physical units - Noisy Input
        noisy_physical = self.denormalize_to_physical(initial_noisy, domain_range)
        self._add_panel_physical(
            axes[1, 0], noisy_physical, "Noisy Input", domain, domain_range
        )

        # Row 2: Intermediate steps in physical units
        for i, step_idx in enumerate(step_indices):
            if step_idx < len(step_results):
                step_data = step_results[step_idx]
                step_physical = self.denormalize_to_physical(
                    step_data["x_next"], domain_range
                )
                self._add_panel_physical(
                    axes[1, i + 1],
                    step_physical,
                    f"Step {step_idx}",
                    domain,
                    domain_range,
                )

        # Row 2: Final restored in physical units
        final_physical = self.denormalize_to_physical(final_restored, domain_range)
        self._add_panel_physical(
            axes[1, -1], final_physical, "Final Restored", domain, domain_range
        )

        # Row 3: Display normalization (percentile normalized tensors) - Noisy Input
        noisy_display = self.normalize_tensor_for_display(
            initial_noisy.cpu().numpy(), domain
        )
        axes[2, 0].imshow(noisy_display, cmap="gray", vmin=0, vmax=1)
        axes[2, 0].set_title("Noisy Input\n(Tensor Display Normalized)")
        axes[2, 0].axis("off")

        # Row 3: Intermediate steps in display format (percentile normalized tensors)
        for i, step_idx in enumerate(step_indices):
            if step_idx < len(step_results):
                step_data = step_results[step_idx]
                step_display = self.normalize_tensor_for_display(
                    step_data["x_next"].cpu().numpy(), domain
                )
                axes[2, i + 1].imshow(step_display, cmap="gray", vmin=0, vmax=1)
                axes[2, i + 1].set_title(
                    f"Step {step_idx}\n(Tensor Display Normalized)"
                )
                axes[2, i + 1].axis("off")

        # Row 3: Final restored in display format (percentile normalized tensors)
        final_display = self.normalize_tensor_for_display(
            final_restored.cpu().numpy(), domain
        )
        axes[2, -1].imshow(final_display, cmap="gray", vmin=0, vmax=1)
        axes[2, -1].set_title("Final Enhanced\n(Tensor Display Normalized)")
        axes[2, -1].axis("off")

        # Add row labels
        axes[0, 0].set_ylabel(
            "Physical Units (RGB)\n[domain_min, domain_max]",
            fontsize=12,
            fontweight="bold",
        )
        axes[1, 0].set_ylabel(
            "Physical Units (Grayscale)\n[domain_min, domain_max]",
            fontsize=12,
            fontweight="bold",
        )
        axes[2, 0].set_ylabel(
            "Tensor Display Normalized\n[0, 1]", fontsize=12, fontweight="bold"
        )

        plt.suptitle(
            f"EDM Enhancement Process - Domain: {domain} - Tile: {tile_id}\n"
            + f"Domain Range: [{domain_range['min']:.1f}, {domain_range['max']:.1f}]",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"✓ Comprehensive enhancement comparison saved: {save_path}")

    def _add_panel(self, ax, tensor, title, subtitle):
        """Add a panel showing tensor in model space [-1, 1]."""
        if tensor.ndim == 4:
            img = (
                tensor[0, 0].cpu().numpy()
                if tensor.shape[1] > 0
                else np.mean(tensor[0].cpu().numpy(), axis=0)
            )
        elif tensor.ndim == 3:
            img = (
                tensor[0].cpu().numpy()
                if tensor.shape[0] > 0
                else np.mean(tensor.cpu().numpy(), axis=0)
            )
        else:
            img = tensor.cpu().numpy()

        # Normalize for display
        img = np.clip(img, -3, 3)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        ax.imshow(img_norm, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.axis("off")

    def _add_panel_physical_rgb(
        self,
        ax,
        physical_array,
        title,
        domain,
        domain_range,
        show_range=False,
        pixel_min=None,
        pixel_max=None,
    ):
        """Add a panel showing RGB data in physical units with proper color balance."""
        # Extract RGB data
        if physical_array.ndim == 4:
            if physical_array.shape[1] == 3:  # RGB
                img = np.transpose(physical_array[0], (1, 2, 0))  # CHW -> HWC
            else:
                img = np.mean(physical_array[0], axis=0)
        elif physical_array.ndim == 3:
            if physical_array.shape[0] == 3:  # RGB
                img = np.transpose(physical_array, (1, 2, 0))  # CHW -> HWC
            else:
                img = np.mean(physical_array, axis=0)
        else:
            img = physical_array

        # Apply display normalization (percentile 1-99%) while preserving color balance
        valid_mask = np.isfinite(img)
        if np.any(valid_mask):
            p_low, p_high = np.percentile(img[valid_mask], [1, 99])
            img_norm = np.clip(img, p_low, p_high)
            img_norm = (img_norm - p_low) / (p_high - p_low + 1e-8)
        else:
            img_norm = np.zeros_like(img)

        # Display RGB or grayscale
        if img_norm.ndim == 3:  # RGB
            ax.imshow(np.clip(img_norm, 0, 1))
        else:  # Grayscale
            ax.imshow(img_norm, cmap="gray", vmin=0, vmax=1)

        # Add title with optional pixel range annotation
        if show_range and pixel_min is not None and pixel_max is not None:
            ax.set_title(
                f"{title}\nPixels: [{pixel_min:.1f}, {pixel_max:.1f}]",
                fontsize=10,
                fontweight="bold",
            )
        else:
            ax.set_title(f"{title}", fontsize=10)
        ax.axis("off")

    def _add_panel_physical(self, ax, physical_array, title, domain, domain_range):
        """Add a panel showing data in physical units with display normalization."""
        if physical_array.ndim == 4:
            img = (
                physical_array[0, 0]
                if physical_array.shape[1] > 0
                else np.mean(physical_array[0], axis=0)
            )
        elif physical_array.ndim == 3:
            img = (
                physical_array[0]
                if physical_array.shape[0] > 0
                else np.mean(physical_array, axis=0)
            )
        else:
            img = physical_array

        # Apply display normalization (percentile 1-99%)
        valid_mask = np.isfinite(img)
        if np.any(valid_mask):
            p_low, p_high = np.percentile(img[valid_mask], [1, 99])
            img_norm = np.clip(img, p_low, p_high)
            img_norm = (img_norm - p_low) / (p_high - p_low + 1e-8)
        else:
            img_norm = np.zeros_like(img)

        ax.imshow(img_norm, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\nRange: [{img.min():.1f}, {img.max():.1f}]", fontsize=10)
        ax.axis("off")


def main():
    """Main function for comprehensive EDM restoration with visualizations."""
    parser = argparse.ArgumentParser(
        description="Comprehensive EDM restoration with step-by-step visualizations (starting from real noisy images)"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pkl)",
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
    parser.add_argument(
        "--domain",
        type=str,
        default="photography",
        choices=["photography", "microscopy", "astronomy"],
        help="Domain for processing",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="clean",
        choices=["noisy", "clean"],
        help="Type of test data to process (noisy or clean)",
    )

    # Sampling arguments
    parser.add_argument(
        "--num_steps",
        type=int,
        default=18,
        help="Number of restoration steps",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of different noisy tiles to restore",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comprehensive_restoration_viz",
        help="Output directory for visualizations",
    )

    # Sampling parameters
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=80.0,
        help="Maximum noise level for restoration",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=7.0,
        help="Time step discretization parameter",
    )

    # PG-Guidance parameters
    parser.add_argument(
        "--use_guidance",
        action="store_true",
        help="Enable Poisson-Gaussian guidance",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=15871.0,
        help="Scale factor for PG-guidance (typically domain max value)",
    )
    parser.add_argument(
        "--guidance_sigma_r",
        type=float,
        default=10.0,
        help="Read noise standard deviation for PG-guidance",
    )
    parser.add_argument(
        "--guidance_kappa",
        type=float,
        default=0.5,
        help="Guidance strength multiplier (typical range: 0.3-1.0)",
    )
    parser.add_argument(
        "--guidance_tau",
        type=float,
        default=0.01,
        help="Guidance threshold - only apply when sigma_t > tau",
    )
    parser.add_argument(
        "--guidance_mode",
        type=str,
        default="wls",
        choices=["wls", "full"],
        help="Guidance mode: 'wls' (weighted least squares) or 'full' (exact gradient)",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for restoration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(
        "COMPREHENSIVE EDM LLIE WITH PG-GUIDANCE AND STEP-BY-STEP VISUALIZATIONS"
    )
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_root}")
    logger.info(f"Metadata: {args.metadata_json}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Data Type: {args.data_type}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Enhancement steps: {args.num_steps}")
    logger.info(f"Number of samples: {args.num_samples}")
    if args.use_guidance:
        logger.info(f"PG-Guidance: ENABLED")
        logger.info(f"  Scale: {args.guidance_scale}")
        logger.info(f"  Read noise (σ_r): {args.guidance_sigma_r}")
        logger.info(f"  Kappa (κ): {args.guidance_kappa}")
        logger.info(f"  Tau (τ): {args.guidance_tau}")
        logger.info(f"  Mode: {args.guidance_mode}")
    else:
        logger.info(f"PG-Guidance: DISABLED")
    logger.info("=" * 80)

    # Initialize LLIE system
    llie_system = ComprehensiveEDMLLIE(
        model_path=args.model_path,
        device=args.device,
        use_guidance=args.use_guidance,
        guidance_scale=args.guidance_scale,
        guidance_sigma_r=args.guidance_sigma_r,
        guidance_kappa=args.guidance_kappa,
        guidance_tau=args.guidance_tau,
        guidance_mode=args.guidance_mode,
    )

    # Load test tiles
    logger.info("Loading test data...")
    test_loader = TestTileLoader(
        data_root=args.data_root,
        metadata_json=args.metadata_json,
        domain=args.domain,
        data_type=args.data_type,
    )

    if len(test_loader) == 0:
        logger.error("No test data found! Exiting.")
        return

    logger.info(f"✓ Loaded {len(test_loader)} test samples")

    # Get diverse sample tiles
    sample_tiles = test_loader.get_sample_tiles(args.num_samples, args.seed)

    # Create domain labels
    if llie_system.net.label_dim > 0:
        class_labels = torch.zeros(
            1, llie_system.net.label_dim, device=llie_system.device
        )
        if args.domain == "photography":
            class_labels[:, 0] = 1.0  # Photography domain
        elif args.domain == "microscopy":
            class_labels[:, 1] = 1.0  # Microscopy domain
        elif args.domain == "astronomy":
            class_labels[:, 2] = 1.0  # Astronomy domain
        logger.info(f"✓ Using domain conditioning: {args.domain}")
    else:
        class_labels = None
        logger.info("✓ No domain conditioning (unconditional model)")

    # Process each sample tile
    all_results = []

    for sample_idx, tile_data in enumerate(sample_tiles):
        tile_id = tile_data["tile_id"]
        logger.info(
            f"\n🔧 Restoring sample {sample_idx + 1}/{len(sample_tiles)}: {tile_id}"
        )

        # Create output directory for this sample
        sample_dir = output_dir / f"sample_{sample_idx:02d}"
        sample_dir.mkdir(exist_ok=True)

        # Enhance with step-by-step visualization
        results = llie_system.enhance_with_visualization(
            noisy_image=tile_data["image"],
            domain_range=tile_data["domain_range"],
            domain=args.domain,
            class_labels=class_labels,
            num_steps=args.num_steps,
            sigma_max=args.sigma_max,
            rho=args.rho,
            save_steps=True,
            output_dir=sample_dir,
            tile_id=tile_id,
        )

        all_results.append(results)

        # Create comprehensive comparison
        comparison_path = sample_dir / "comprehensive_enhancement_comparison.png"
        llie_system.create_comprehensive_comparison(
            initial_noisy=results["initial_noisy"],
            final_restored=results["final_enhanced"],
            step_results=results["step_results"],
            domain_range=results["domain_range"],
            domain=args.domain,
            tile_id=results["tile_id"],
            save_path=comparison_path,
        )

        # Save individual final sample visualization
        final_vis_path = sample_dir / "final_enhanced_display.png"
        display_img = llie_system.create_display_visualization(
            results["final_enhanced"],
            results["domain_range"],
            args.domain,
            title=f"Final Enhanced - {tile_id}",
            save_path=final_vis_path,
        )

        # Save physical units data
        physical_data = llie_system.denormalize_to_physical(
            results["final_enhanced"], results["domain_range"]
        )
        np.save(sample_dir / "final_enhanced_physical.npy", physical_data)

        # Save initial noisy for comparison
        noisy_physical = llie_system.denormalize_to_physical(
            results["initial_noisy"], results["domain_range"]
        )
        np.save(sample_dir / "initial_noisy_physical.npy", noisy_physical)

        logger.info(
            f"✓ Sample {sample_idx + 1} enhancement completed and saved to {sample_dir}"
        )

    # Create summary visualization
    logger.info("\n📊 Creating summary visualization...")

    # Create a grid showing all enhancement results
    fig, axes = plt.subplots(3, args.num_samples, figsize=(4 * args.num_samples, 12))

    # Handle single sample case where axes is 1D
    if args.num_samples == 1:
        axes = axes.reshape(3, 1)

    for sample_idx in range(args.num_samples):
        results = all_results[sample_idx]

        # Row 1: Initial noisy images (model space)
        try:
            initial_tensor = results["initial_noisy"]
            initial_np = initial_tensor.cpu().numpy()
            initial_np = np.nan_to_num(initial_np, nan=0.0, posinf=1.0, neginf=-1.0)

            if initial_np.ndim == 4:
                if initial_np.shape[1] == 3:  # RGB
                    img_model = np.transpose(initial_np[0], (1, 2, 0))
                else:  # Grayscale or other
                    img_model = (
                        initial_np[0, 0]
                        if initial_np.shape[1] > 0
                        else np.mean(initial_np[0], axis=0)
                    )
            elif initial_np.ndim == 3:
                if initial_np.shape[0] == 3:  # RGB
                    img_model = np.transpose(initial_np, (1, 2, 0))
                else:
                    img_model = initial_np[0]
            else:
                img_model = initial_np

            # Normalize for display (preserve RGB if available)
            if img_model.ndim == 3 and img_model.shape[2] == 3:
                # RGB image - normalize all channels together to preserve color balance
                valid_mask = np.isfinite(img_model)
                if np.any(valid_mask):
                    p_low, p_high = np.percentile(img_model[valid_mask], [1, 99])
                    img_model_norm = np.clip(img_model, p_low, p_high)
                    img_model_norm = (img_model_norm - p_low) / (p_high - p_low + 1e-8)
                else:
                    img_model_norm = np.ones_like(img_model) * 0.5
                axes[0, sample_idx].imshow(np.clip(img_model_norm, 0, 1))
            elif img_model.ndim == 2:
                # Grayscale image
                valid_mask = np.isfinite(img_model)
                if np.any(valid_mask):
                    p_low, p_high = np.percentile(img_model[valid_mask], [1, 99])
                    img_model_norm = np.clip(img_model, p_low, p_high)
                    img_model_norm = (img_model_norm - img_model_norm.min()) / (
                        img_model_norm.max() - img_model_norm.min() + 1e-8
                    )
                else:
                    img_model_norm = np.ones_like(img_model) * 0.5
                axes[0, sample_idx].imshow(img_model_norm, cmap="gray", vmin=0, vmax=1)
            else:
                axes[0, sample_idx].imshow(np.ones((256, 256)) * 0.5, cmap="gray")

        except Exception as e:
            logger.warning(
                f"Error displaying initial noisy image for sample {sample_idx}: {e}"
            )
            axes[0, sample_idx].imshow(np.ones((256, 256)) * 0.5, cmap="gray")

        axes[0, sample_idx].set_title(f"Sample {sample_idx + 1}\nInitial Noisy [-1, 1]")
        axes[0, sample_idx].axis("off")

        # Row 2: Final enhanced images (model space)
        try:
            final_tensor = results["final_enhanced"]
            final_np = final_tensor.cpu().numpy()
            final_np = np.nan_to_num(final_np, nan=0.0, posinf=1.0, neginf=-1.0)

            if final_np.ndim == 4:
                if final_np.shape[1] == 3:  # RGB
                    img_model = np.transpose(final_np[0], (1, 2, 0))
                else:  # Grayscale or other
                    img_model = (
                        final_np[0, 0]
                        if final_np.shape[1] > 0
                        else np.mean(final_np[0], axis=0)
                    )
            elif final_np.ndim == 3:
                if final_np.shape[0] == 3:  # RGB
                    img_model = np.transpose(final_np, (1, 2, 0))
                else:
                    img_model = final_np[0]
            else:
                img_model = final_np

            # Normalize for display (preserve RGB if available)
            if img_model.ndim == 3 and img_model.shape[2] == 3:
                # RGB image - normalize all channels together to preserve color balance
                valid_mask = np.isfinite(img_model)
                if np.any(valid_mask):
                    p_low, p_high = np.percentile(img_model[valid_mask], [1, 99])
                    img_model_norm = np.clip(img_model, p_low, p_high)
                    img_model_norm = (img_model_norm - p_low) / (p_high - p_low + 1e-8)
                else:
                    img_model_norm = np.ones_like(img_model) * 0.5
                axes[1, sample_idx].imshow(np.clip(img_model_norm, 0, 1))
            elif img_model.ndim == 2:
                # Grayscale image
                valid_mask = np.isfinite(img_model)
                if np.any(valid_mask):
                    p_low, p_high = np.percentile(img_model[valid_mask], [1, 99])
                    img_model_norm = np.clip(img_model, p_low, p_high)
                    img_model_norm = (img_model_norm - img_model_norm.min()) / (
                        img_model_norm.max() - img_model_norm.min() + 1e-8
                    )
                else:
                    img_model_norm = np.ones_like(img_model) * 0.5
                axes[1, sample_idx].imshow(img_model_norm, cmap="gray", vmin=0, vmax=1)
            else:
                axes[1, sample_idx].imshow(np.ones((256, 256)) * 0.5, cmap="gray")

        except Exception as e:
            logger.warning(
                f"Error displaying final enhanced image for sample {sample_idx}: {e}"
            )
            axes[1, sample_idx].imshow(np.ones((256, 256)) * 0.5, cmap="gray")

        axes[1, sample_idx].set_title(
            f"Sample {sample_idx + 1}\nFinal Enhanced [-1, 1]"
        )
        axes[1, sample_idx].axis("off")

        # Row 3: Final enhanced in display format (tensor percentile normalized)
        final_display = llie_system.normalize_tensor_for_display(
            results["final_enhanced"].cpu().numpy(), args.domain
        )
        if final_display is not None:
            # Handle RGB or grayscale
            if final_display.ndim == 3:  # RGB
                axes[2, sample_idx].imshow(np.clip(final_display, 0, 1))
            else:  # Grayscale
                axes[2, sample_idx].imshow(final_display, cmap="gray", vmin=0, vmax=1)
        else:
            axes[2, sample_idx].imshow(np.ones((256, 256)) * 0.5, cmap="gray")
        axes[2, sample_idx].set_title(f"Sample {sample_idx + 1}\nDisplay Normalized")
        axes[2, sample_idx].axis("off")

    plt.suptitle(
        f"EDM Low-Light Enhancement Results - {args.domain.capitalize()}\n"
        + f"Starting from real noisy images → {args.num_steps}-step enhancement process",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    summary_path = output_dir / "all_enhancement_summary.png"
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Summary visualization saved: {summary_path}")

    # Save enhancement metadata
    metadata_path = output_dir / "enhancement_metadata.json"
    metadata = {
        "enhancement_info": {
            "model_path": args.model_path,
            "domain": args.domain,
            "data_type": args.data_type,
            "num_steps": args.num_steps,
            "num_samples": args.num_samples,
            "sigma_max": args.sigma_max,
            "rho": args.rho,
            "device": args.device,
            "seed": args.seed,
            "use_guidance": args.use_guidance,
            "guidance_scale": args.guidance_scale if args.use_guidance else None,
            "guidance_sigma_r": args.guidance_sigma_r if args.use_guidance else None,
            "guidance_kappa": args.guidance_kappa if args.use_guidance else None,
            "guidance_tau": args.guidance_tau if args.use_guidance else None,
            "guidance_mode": args.guidance_mode if args.use_guidance else None,
        },
        "model_info": {
            "resolution": llie_system.net.img_resolution,
            "channels": llie_system.net.img_channels,
            "label_dim": llie_system.net.label_dim,
            "sigma_min": llie_system.net.sigma_min,
            "sigma_max": llie_system.net.sigma_max,
        },
        "output_files": {
            "summary_visualization": str(summary_path),
            "enhancement_metadata": str(metadata_path),
            "individual_samples": [
                str(output_dir / f"sample_{i:02d}") for i in range(args.num_samples)
            ],
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Metadata saved: {metadata_path}")
    logger.info("=" * 80)
    logger.info("🎉 COMPREHENSIVE LLIE WITH PG-GUIDANCE COMPLETED!")
    logger.info(f"📁 All results saved to: {output_dir}")
    logger.info(
        f"📊 Summary: {args.num_samples} enhancements, {args.num_steps} steps each"
    )
    if args.use_guidance:
        logger.info(
            f"⚡ PG-Guidance: ENABLED (κ={args.guidance_kappa}, σ_r={args.guidance_sigma_r})"
        )
    else:
        logger.info(f"⚡ PG-Guidance: DISABLED")
    logger.info(f"🎯 Starting from {args.data_type} images → enhanced images")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
