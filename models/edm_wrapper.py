"""
EDM Model Wrapper with Domain Conditioning.

This module provides a wrapper around the external EDM (Elucidating the Design Space
of Diffusion-Based Generative Models) codebase, adding domain-specific conditioning
for multi-domain Poisson-Gaussian image restoration.

The wrapper integrates:
- 6-dimensional domain conditioning vectors
- FiLM (Feature-wise Linear Modulation) conditioning architecture
- Model initialization utilities and factory functions
- Seamless integration with the existing EDM training pipeline

Requirements addressed: 2.1, 4.1 from requirements.md
Task: 3.1 from tasks.md
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Add EDM to path
project_root = Path(__file__).parent.parent
edm_path = project_root / "external" / "edm"
if str(edm_path) not in sys.path:
    sys.path.insert(0, str(edm_path))

try:
    from training.networks import EDMPrecond

    EDM_AVAILABLE = True
except ImportError as e:
    EDM_AVAILABLE = False
    EDMPrecond = None
    print(f"Warning: EDM not available: {e}")
    print("Run 'bash external/setup_edm.sh' to set up EDM integration")

from core.exceptions import ConfigurationError, ModelError
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EDMConfig:
    """Configuration for EDM model parameters."""

    # Image parameters
    img_resolution: int = 128
    img_channels: int = 1

    # Model architecture
    model_channels: int = 128
    channel_mult: List[int] = None
    channel_mult_emb: int = 4
    num_blocks: int = 4
    attn_resolutions: List[int] = None
    dropout: float = 0.0

    # Conditioning
    label_dim: int = 6  # Our 6D domain conditioning
    use_fp16: bool = False

    # EDM-specific parameters
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5

    def __post_init__(self):
        """Set default values for list parameters."""
        if self.channel_mult is None:
            self.channel_mult = [1, 2, 2, 2]
        if self.attn_resolutions is None:
            self.attn_resolutions = [16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for EDM model creation."""
        return {
            "img_resolution": self.img_resolution,
            "img_channels": self.img_channels,
            "model_channels": self.model_channels,
            "channel_mult": self.channel_mult,
            "channel_mult_emb": self.channel_mult_emb,
            "num_blocks": self.num_blocks,
            "attn_resolutions": self.attn_resolutions,
            "dropout": self.dropout,
            "label_dim": self.label_dim,
            "use_fp16": self.use_fp16,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "sigma_data": self.sigma_data,
        }


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer for conditioning.

    Applies affine transformation to features based on conditioning vector:
    output = gamma * input + beta
    """

    def __init__(self, feature_dim: int, condition_dim: int):
        """
        Initialize FiLM layer.

        Args:
            feature_dim: Number of feature channels
            condition_dim: Dimension of conditioning vector
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.condition_dim = condition_dim

        # Linear layers to generate gamma and beta
        self.gamma_layer = nn.Linear(condition_dim, feature_dim)
        self.beta_layer = nn.Linear(condition_dim, feature_dim)

        # Initialize gamma to 1 and beta to 0 (identity transformation)
        nn.init.zeros_(self.gamma_layer.weight)
        nn.init.ones_(self.gamma_layer.bias)  # bias = 1 for gamma
        nn.init.zeros_(self.beta_layer.weight)
        nn.init.zeros_(self.beta_layer.bias)  # bias = 0 for beta

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            features: Input features [B, C, H, W]
            condition: Conditioning vector [B, condition_dim]

        Returns:
            Modulated features [B, C, H, W]
        """
        # Generate gamma and beta
        gamma = self.gamma_layer(condition)  # [B, C]
        beta = self.beta_layer(condition)  # [B, C]

        # Reshape for broadcasting
        gamma = gamma.view(-1, self.feature_dim, 1, 1)  # [B, C, 1, 1]
        beta = beta.view(-1, self.feature_dim, 1, 1)  # [B, C, 1, 1]

        # Apply modulation
        return gamma * features + beta


class DomainEncoder(nn.Module):
    """
    Encode domain information into conditioning vectors.

    Creates 6-dimensional conditioning vectors:
    - Domain one-hot [3]: photography/microscopy/astronomy
    - Log scale [1]: normalized log10(scale)
    - Relative read noise [1]: read_noise / scale
    - Relative background [1]: background / scale
    """

    def __init__(self):
        """Initialize domain encoder."""
        super().__init__()

        # Domain mappings
        self.domain_to_idx = {"photography": 0, "microscopy": 1, "astronomy": 2}

        # Normalization parameters (learned from typical ranges)
        self.log_scale_mean = 2.5  # log10(~300)
        self.log_scale_std = 1.0

        logger.info("DomainEncoder initialized with 6D conditioning")

    def encode_domain(
        self,
        domain: Union[str, List[str], torch.Tensor],
        scale: Union[float, torch.Tensor],
        read_noise: Union[float, torch.Tensor],
        background: Union[float, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode domain parameters into conditioning vector.

        Args:
            domain: Domain name(s) or indices
            scale: Dataset normalization scale (electrons)
            read_noise: Read noise standard deviation (electrons)
            background: Background offset (electrons)
            device: Target device for tensors

        Returns:
            Conditioning vector [B, 6]
        """
        # Handle different input types
        if isinstance(domain, str):
            domain = [domain]
        if isinstance(domain, list):
            batch_size = len(domain)
        else:
            batch_size = domain.shape[0] if hasattr(domain, "shape") else 1

        # Convert scalars to tensors
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)
        if not isinstance(read_noise, torch.Tensor):
            read_noise = torch.tensor(read_noise, dtype=torch.float32)
        if not isinstance(background, torch.Tensor):
            background = torch.tensor(background, dtype=torch.float32)

        # Ensure tensors are on correct device
        if device is not None:
            scale = scale.to(device)
            read_noise = read_noise.to(device)
            background = background.to(device)

        # Expand to batch size if needed
        if scale.dim() == 0:
            scale = scale.expand(batch_size)
        if read_noise.dim() == 0:
            read_noise = read_noise.expand(batch_size)
        if background.dim() == 0:
            background = background.expand(batch_size)

        # Create domain one-hot encoding
        if isinstance(domain, list):
            domain_indices = [self.domain_to_idx[d] for d in domain]
            domain_tensor = torch.tensor(domain_indices, dtype=torch.long)
        else:
            domain_tensor = domain

        if device is not None:
            domain_tensor = domain_tensor.to(device)

        domain_onehot = torch.zeros(batch_size, 3, device=device or scale.device)
        domain_onehot.scatter_(1, domain_tensor.unsqueeze(1), 1.0)

        # Normalize log scale
        log_scale = torch.log10(torch.clamp(scale, min=1e-6))
        log_scale_norm = (log_scale - self.log_scale_mean) / self.log_scale_std

        # Compute relative parameters
        rel_read_noise = read_noise / torch.clamp(scale, min=1e-6)
        rel_background = background / torch.clamp(scale, min=1e-6)

        # Concatenate conditioning vector
        condition = torch.cat(
            [
                domain_onehot,  # [B, 3]
                log_scale_norm.unsqueeze(1),  # [B, 1]
                rel_read_noise.unsqueeze(1),  # [B, 1]
                rel_background.unsqueeze(1),  # [B, 1]
            ],
            dim=1,
        )  # [B, 6]

        return condition

    def encode_batch(
        self, batch_params: List[Dict[str, Any]], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode a batch of domain parameters.

        Args:
            batch_params: List of parameter dictionaries
            device: Target device

        Returns:
            Batch conditioning tensor [B, 6]
        """
        batch_conditions = []

        for params in batch_params:
            condition = self.encode_domain(
                domain=params["domain"],
                scale=params["scale"],
                read_noise=params["read_noise"],
                background=params["background"],
                device=device,
            )
            batch_conditions.append(condition)

        return torch.cat(batch_conditions, dim=0)

    def decode_domain(self, condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode conditioning vector back to domain parameters.

        Args:
            condition: Conditioning vector [B, 6]

        Returns:
            Dictionary with decoded parameters
        """
        domain_onehot = condition[:, :3]
        log_scale_norm = condition[:, 3]
        rel_read_noise = condition[:, 4]
        rel_background = condition[:, 5]

        # Decode domain
        domain_idx = torch.argmax(domain_onehot, dim=1)

        # Denormalize log scale
        log_scale = log_scale_norm * self.log_scale_std + self.log_scale_mean
        scale = torch.pow(10, log_scale)

        # Compute absolute parameters
        read_noise = rel_read_noise * scale
        background = rel_background * scale

        return {
            "domain_idx": domain_idx,
            "scale": scale,
            "read_noise": read_noise,
            "background": background,
        }


class EDMModelWrapper(nn.Module):
    """
    Wrapper around EDM model with domain conditioning.

    This wrapper integrates the external EDM codebase with our domain-specific
    conditioning system, providing a unified interface for multi-domain
    Poisson-Gaussian image restoration.
    """

    def __init__(
        self,
        config: Optional[EDMConfig] = None,
        conditioning_mode: str = "class_labels",
        use_film: bool = False,
    ):
        """
        Initialize EDM wrapper.

        Args:
            config: EDM configuration (uses default if None)
            conditioning_mode: How to pass conditioning ('class_labels' or 'film')
            use_film: Whether to use FiLM layers for additional conditioning
        """
        super().__init__()

        if not EDM_AVAILABLE:
            raise ModelError(
                "EDM not available. Run 'bash external/setup_edm.sh' to set up."
            )

        self.config = config or EDMConfig()
        self.conditioning_mode = conditioning_mode
        self.use_film = use_film

        # Validate configuration
        self._validate_config()

        # Create EDM model
        try:
            edm_config = self.config.to_dict()
            self.edm_model = EDMPrecond(**edm_config)
            logger.info(
                f"EDM model created with {sum(p.numel() for p in self.edm_model.parameters())} parameters"
            )
        except Exception as e:
            raise ModelError(f"Failed to create EDM model: {e}")

        # Create domain encoder
        self.domain_encoder = DomainEncoder()

        # Optional FiLM conditioning layers
        if self.use_film:
            # Add FiLM layers at multiple resolutions
            self.film_layers = nn.ModuleDict()
            # Note: These would need to be integrated into EDM's forward pass
            # For now, we'll use the class_labels pathway
            logger.info("FiLM conditioning enabled (requires EDM modification)")

        # Model metadata
        self.model_type = "edm_wrapper"
        self.condition_dim = 6

        logger.info(
            f"EDMModelWrapper initialized: {conditioning_mode} conditioning, "
            f"FiLM={use_film}, resolution={self.config.img_resolution}"
        )

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.label_dim != 6:
            raise ConfigurationError(
                f"label_dim must be 6 for domain conditioning, got {self.config.label_dim}"
            )

        if self.conditioning_mode not in ["class_labels", "film"]:
            raise ConfigurationError(
                f"Unknown conditioning_mode: {self.conditioning_mode}"
            )

        if self.config.img_channels != 1:
            logger.warning(
                f"img_channels={self.config.img_channels}, expected 1 for grayscale"
            )

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        domain: Optional[Union[str, List[str], torch.Tensor]] = None,
        scale: Optional[Union[float, torch.Tensor]] = None,
        read_noise: Optional[Union[float, torch.Tensor]] = None,
        background: Optional[Union[float, torch.Tensor]] = None,
        condition: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through EDM with domain conditioning.

        Args:
            x: Input images [B, C, H, W]
            sigma: Noise levels [B] or scalar
            domain: Domain specification (if condition not provided)
            scale: Scale parameter (if condition not provided)
            read_noise: Read noise parameter (if condition not provided)
            background: Background parameter (if condition not provided)
            condition: Pre-computed conditioning vector [B, 6]
            **kwargs: Additional arguments for EDM

        Returns:
            Model output [B, C, H, W]
        """
        # Prepare conditioning
        if condition is None:
            if any(p is None for p in [domain, scale, read_noise, background]):
                raise ValueError(
                    "Must provide either 'condition' or all domain parameters"
                )

            condition = self.domain_encoder.encode_domain(
                domain=domain,
                scale=scale,
                read_noise=read_noise,
                background=background,
                device=x.device,
            )

        # Ensure condition has correct batch size
        if condition.shape[0] != x.shape[0]:
            if condition.shape[0] == 1:
                condition = condition.expand(x.shape[0], -1)
            else:
                raise ValueError(
                    f"Condition batch size {condition.shape[0]} != input batch size {x.shape[0]}"
                )

        # Forward through EDM with conditioning
        if self.conditioning_mode == "class_labels":
            # Use EDM's built-in class conditioning
            output = self.edm_model(x, sigma, class_labels=condition, **kwargs)
        else:
            # FiLM conditioning (would require EDM modification)
            raise NotImplementedError(
                "FiLM conditioning requires EDM model modification"
            )

        return output

    def encode_conditioning(
        self,
        domain: Union[str, List[str]],
        scale: Union[float, torch.Tensor],
        read_noise: Union[float, torch.Tensor],
        background: Union[float, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encode domain parameters into conditioning vector.

        Args:
            domain: Domain name(s)
            scale: Scale parameter(s)
            read_noise: Read noise parameter(s)
            background: Background parameter(s)
            device: Target device

        Returns:
            Conditioning vector [B, 6]
        """
        return self.domain_encoder.encode_domain(
            domain=domain,
            scale=scale,
            read_noise=read_noise,
            background=background,
            device=device,
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": self.model_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "condition_dim": self.condition_dim,
            "conditioning_mode": self.conditioning_mode,
            "use_film": self.use_film,
            "config": self.config.to_dict(),
            "img_resolution": self.config.img_resolution,
            "img_channels": self.config.img_channels,
        }

    def estimate_memory_usage(self, batch_size: int = 1) -> Dict[str, float]:
        """
        Estimate memory usage for given batch size.

        Args:
            batch_size: Batch size for estimation

        Returns:
            Memory usage estimates in MB
        """
        # Model parameters
        param_memory = (
            sum(p.numel() * 4 for p in self.parameters()) / 1e6
        )  # 4 bytes per float32

        # Activations (rough estimate)
        img_size = self.config.img_resolution
        channels = self.config.img_channels

        # Input/output tensors
        io_memory = (
            batch_size * channels * img_size * img_size * 4 * 2 / 1e6
        )  # input + output

        # Intermediate activations (rough estimate based on U-Net structure)
        intermediate_memory = (
            batch_size * channels * img_size * img_size * 4 * 8 / 1e6
        )  # ~8x for intermediates

        total_memory = param_memory + io_memory + intermediate_memory

        return {
            "parameters_mb": param_memory,
            "io_tensors_mb": io_memory,
            "intermediate_mb": intermediate_memory,
            "total_estimated_mb": total_memory,
            "batch_size": batch_size,
        }


# Factory functions for easy model creation


def create_edm_wrapper(
    img_resolution: int = 128,
    img_channels: int = 1,
    model_channels: int = 128,
    conditioning_mode: str = "class_labels",
    **config_overrides,
) -> EDMModelWrapper:
    """
    Create EDM wrapper with standard configuration.

    Args:
        img_resolution: Image resolution
        img_channels: Number of image channels
        model_channels: Base number of model channels
        conditioning_mode: Conditioning mode
        **config_overrides: Additional configuration overrides

    Returns:
        Configured EDMModelWrapper
    """
    config = EDMConfig(
        img_resolution=img_resolution,
        img_channels=img_channels,
        model_channels=model_channels,
        **config_overrides,
    )

    return EDMModelWrapper(config=config, conditioning_mode=conditioning_mode)


def create_domain_edm_wrapper(
    domain: str, img_resolution: int = 128, **config_overrides
) -> EDMModelWrapper:
    """
    Create domain-optimized EDM wrapper.

    Args:
        domain: Target domain ('photography', 'microscopy', 'astronomy')
        img_resolution: Image resolution
        **config_overrides: Configuration overrides

    Returns:
        Domain-optimized EDMModelWrapper
    """
    # Domain-specific configurations
    domain_configs = {
        "photography": {"model_channels": 128, "num_blocks": 4, "dropout": 0.1},
        "microscopy": {
            "model_channels": 192,  # More capacity for precision
            "num_blocks": 6,
            "dropout": 0.0,
        },
        "astronomy": {
            "model_channels": 96,  # Lighter for low-light
            "num_blocks": 4,
            "dropout": 0.2,
        },
    }

    if domain not in domain_configs:
        raise ValueError(
            f"Unknown domain: {domain}. Supported: {list(domain_configs.keys())}"
        )

    # Merge domain config with overrides
    config_dict = domain_configs[domain].copy()
    config_dict.update(config_overrides)

    return create_edm_wrapper(img_resolution=img_resolution, **config_dict)


# Utility functions


def load_pretrained_edm(
    checkpoint_path: str,
    config: Optional[EDMConfig] = None,
    device: Optional[torch.device] = None,
) -> EDMModelWrapper:
    """
    Load pretrained EDM wrapper from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration (inferred if None)
        device: Target device

    Returns:
        Loaded EDMModelWrapper
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract configuration if not provided
    if config is None:
        if "config" in checkpoint:
            config = EDMConfig(**checkpoint["config"])
        else:
            raise ValueError("No configuration found in checkpoint and none provided")

    # Create model
    model = EDMModelWrapper(config=config)

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Move to device
    if device is not None:
        model = model.to(device)

    logger.info(f"Loaded pretrained EDM wrapper from {checkpoint_path}")
    return model


def test_edm_wrapper_basic() -> bool:
    """
    Basic test of EDM wrapper functionality.

    Returns:
        True if test passes
    """
    try:
        # Create model
        model = create_edm_wrapper(img_resolution=64, model_channels=64)
        model.eval()

        # Test data
        batch_size = 2
        x = torch.randn(batch_size, 1, 64, 64)
        sigma = torch.tensor([1.0, 0.5])

        # Test conditioning
        condition = model.encode_conditioning(
            domain=["photography", "microscopy"],
            scale=[1000.0, 500.0],
            read_noise=[3.0, 2.0],
            background=[10.0, 5.0],
        )

        # Forward pass
        with torch.no_grad():
            output = model(x, sigma, condition=condition)

        # Check output
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

        logger.info("EDM wrapper basic test passed")
        return True

    except Exception as e:
        logger.error(f"EDM wrapper basic test failed: {e}")
        return False


if __name__ == "__main__":
    # Run basic test
    if EDM_AVAILABLE:
        success = test_edm_wrapper_basic()
        print(f"EDM wrapper test: {'PASS' if success else 'FAIL'}")
    else:
        print("EDM not available - run 'bash external/setup_edm.sh' first")
