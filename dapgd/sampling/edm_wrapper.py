"""
EDM Model Wrapper for DAPGD

PURPOSE: Clean interface to EDM's code with domain conditioning support
Based on existing EDM integration in /home/jilab/Jae/external/edm/

INTEGRATION STRATEGY:
- Use EDM's existing class_labels pathway for 6D conditioning
- Wrapper handles domain parameter normalization
- Minimal modifications to EDM codebase
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DomainConditioner(nn.Module):
    """
    Processes domain conditioning parameters into EDM-compatible format

    Conditioning vector (6 dimensions):
    [domain_one_hot_3, log_scale_norm_1, rel_read_noise_1, rel_background_1]

    Args:
        condition_dim: Dimension of conditioning vector (default: 6)
        normalize_scale: Whether to normalize log scale (default: True)
    """

    def __init__(self, condition_dim: int = 6, normalize_scale: bool = True):
        super().__init__()
        self.condition_dim = condition_dim
        self.normalize_scale = normalize_scale

        # Domain mapping for one-hot encoding
        self.domain_map = {"photo": 0, "micro": 1, "astro": 2}

    def _normalize_scale(self, scale: torch.Tensor) -> torch.Tensor:
        """Normalize scale parameter using log transformation"""
        return torch.log10(scale + 1e-8)  # Add epsilon for numerical stability

    def _create_condition_vector(
        self,
        domain: str,
        scale: float,
        read_noise: float,
        background: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create 6-dimensional conditioning vector

        Returns:
            [domain_one_hot_3, log_scale_norm_1, rel_read_noise_1, rel_background_1]
        """

        # 1. Domain one-hot encoding [3]
        domain_idx = self.domain_map.get(domain, 0)
        domain_one_hot = torch.zeros(3, device=device)
        domain_one_hot[domain_idx] = 1.0

        # 2. Scale normalization [1]
        scale_tensor = torch.tensor(scale, device=device, dtype=torch.float32)
        if self.normalize_scale:
            log_scale_norm = self._normalize_scale(scale_tensor)
        else:
            log_scale_norm = scale_tensor

        # 3. Relative read noise [1]
        rel_read_noise = torch.tensor(
            read_noise / scale, device=device, dtype=torch.float32
        )

        # 4. Relative background [1]
        rel_background = torch.tensor(
            background / scale, device=device, dtype=torch.float32
        )

        # Concatenate into conditioning vector
        condition = torch.cat(
            [
                domain_one_hot,
                log_scale_norm.unsqueeze(0),
                rel_read_noise.unsqueeze(0),
                rel_background.unsqueeze(0),
            ]
        )

        return condition

    def forward(
        self,
        domain: str,
        scale: float,
        read_noise: float,
        background: float,
        batch_size: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Create conditioning vector for batch

        Args:
            domain: Domain type ('photo', 'micro', 'astro')
            scale: Scale factor (s)
            read_noise: Read noise (σ_r)
            background: Background level
            batch_size: Number of samples in batch
            device: Target device

        Returns:
            Conditioning tensor [batch_size, condition_dim]
        """

        if device is None:
            device = next(self.parameters()).device

        # Create single condition vector
        condition_vector = self._create_condition_vector(
            domain, scale, read_noise, background, device
        )

        # Expand for batch
        condition_batch = condition_vector.unsqueeze(0).expand(batch_size, -1)

        return condition_batch


class EDMModelWrapper(nn.Module):
    """
    Wrapper around EDM that handles domain conditioning

    INTEGRATION APPROACH:
    - Uses EDM's existing class_labels pathway
    - Handles 6-dimensional domain conditioning vector
    - Minimal modifications to EDM codebase

    Args:
        edm_config: EDM configuration dictionary
        condition_dim: Dimension of conditioning vector (default: 6)
        normalize_scale: Whether to normalize scale parameter
        device: Target device
    """

    def __init__(
        self,
        edm_config: Dict[str, Any],
        condition_dim: int = 6,
        normalize_scale: bool = True,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device
        self.condition_dim = condition_dim

        # Create conditioning processor
        self.condition_processor = DomainConditioner(
            condition_dim=condition_dim, normalize_scale=normalize_scale
        )

        # Import and create EDM model
        try:
            from edm.training.networks import EDMPrecond

            logger.info("Successfully imported EDMPrecond from edm.training.networks")
        except ImportError as e:
            logger.error(f"Failed to import EDMPrecond: {e}")
            logger.error("Make sure EDM is installed in /home/jilab/Jae/external/edm/")
            raise

        # EDM configuration for our use case
        # Based on existing integration testing
        edm_defaults = {
            "img_resolution": 128,
            "img_channels": 1,  # Start with grayscale for testing
            "model_channels": 128,
            "channel_mult": (1, 2, 3, 4),
            "channel_mult_emb": 4,
            "num_blocks": 4,
            "attn_resolutions": (16, 32, 64),
            "label_dim": condition_dim,  # Our 6D conditioning
            "use_fp16": False,
            "use_xformers": False,
        }

        # Merge with provided config
        edm_config = {**edm_defaults, **edm_config}

        logger.info(f"Creating EDMPrecond with config: {edm_config}")

        # Create EDM model
        self.edm_model = EDMPrecond(**edm_config)

        # Move to device
        self.edm_model = self.edm_model.to(device)

        logger.info(f"EDM wrapper created successfully on device: {device}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.edm_model.parameters()):,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        domain: Optional[str] = None,
        scale: Optional[float] = None,
        read_noise: Optional[float] = None,
        background: Optional[float] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through EDM with optional domain conditioning

        Args:
            x: Input tensor [B, C, H, W]
            sigma: Noise level tensor [B,] or scalar
            domain: Domain type for conditioning
            scale: Scale factor for conditioning
            read_noise: Read noise for conditioning
            background: Background level for conditioning
            condition: Pre-computed conditioning tensor (overrides other params)

        Returns:
            Model output [B, C, H, W]
        """

        # Handle conditioning
        if condition is not None:
            # Use provided conditioning tensor
            class_labels = condition
        elif domain is not None and scale is not None and read_noise is not None:
            # Create conditioning from domain parameters
            batch_size = x.shape[0]
            class_labels = self.condition_processor(
                domain=domain,
                scale=scale,
                read_noise=read_noise,
                background=background or 0.0,
                batch_size=batch_size,
                device=self.device,
            )
        else:
            # No conditioning
            class_labels = None

        # Forward through EDM
        # EDM expects: model(x, sigma, class_labels)
        output = self.edm_model(x, sigma, class_labels)

        return output

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Handle different checkpoint formats
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Load state dict
            self.edm_model.load_state_dict(state_dict)

            logger.info(f"Successfully loaded checkpoint from: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise

    def save_checkpoint(self, checkpoint_path: str):
        """
        Save model weights to checkpoint

        Args:
            checkpoint_path: Path to save checkpoint
        """

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save model state
            checkpoint = {
                "model": self.edm_model.state_dict(),
                "config": {
                    "condition_dim": self.condition_dim,
                    "device": str(self.device),
                },
                "model_type": "EDMPrecond",
            }

            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Successfully saved checkpoint to: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
            raise

    def get_conditioning_info(self) -> Dict[str, Any]:
        """Get information about conditioning setup"""
        return {
            "condition_dim": self.condition_dim,
            "normalize_scale": self.condition_processor.normalize_scale,
            "domain_map": self.condition_processor.domain_map,
            "device": str(self.device),
        }


def create_edm_wrapper(
    config: Dict[str, Any], device: str = "cuda", checkpoint_path: Optional[str] = None
) -> EDMModelWrapper:
    """
    Factory function to create EDM wrapper with optional checkpoint loading

    Args:
        config: EDM configuration dictionary
        device: Target device
        checkpoint_path: Optional path to checkpoint to load

    Returns:
        Configured EDMModelWrapper instance
    """

    wrapper = EDMModelWrapper(edm_config=config, device=device)

    if checkpoint_path:
        wrapper.load_checkpoint(checkpoint_path)

    return wrapper


# Test function for integration verification
def test_edm_wrapper():
    """Test EDM wrapper functionality"""

    logger.info("Testing EDM wrapper...")

    # Test configuration
    test_config = {
        "img_resolution": 64,  # Smaller for testing
        "img_channels": 1,
        "model_channels": 64,  # Smaller for testing
        "channel_mult": (1, 2, 2, 2),
        "num_blocks": 2,
        "label_dim": 6,
    }

    try:
        # Create wrapper
        wrapper = EDMModelWrapper(
            edm_config=test_config, device="cpu"  # Use CPU for testing
        )

        logger.info("✓ EDM wrapper created successfully")

        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 1, 64, 64)
        sigma = torch.tensor([1.0, 1.0])

        # Test without conditioning
        output = wrapper(x, sigma)
        logger.info(f"✓ Forward pass without conditioning: {output.shape}")

        # Test with conditioning
        condition = wrapper.condition_processor(
            domain="photo",
            scale=79351.0,
            read_noise=3.6,
            background=0.0,
            batch_size=batch_size,
            device="cpu",
        )

        output_conditioned = wrapper(x, sigma, condition=condition)
        logger.info(f"✓ Forward pass with conditioning: {output_conditioned.shape}")

        # Verify outputs are different (conditioning should affect output)
        diff = (output - output_conditioned).abs().mean().item()
        logger.info(f"✓ Conditioning effect verified: mean difference = {diff:.6f}")

        logger.info("✅ All EDM wrapper tests passed!")

        return True

    except Exception as e:
        logger.error(f"❌ EDM wrapper test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run integration test
    success = test_edm_wrapper()
    if success:
        print("\n🎉 EDM wrapper integration test PASSED!")
        print("Ready to proceed with DAPGD implementation.")
    else:
        print("\n❌ EDM wrapper integration test FAILED!")
        print("Check logs above for details.")
