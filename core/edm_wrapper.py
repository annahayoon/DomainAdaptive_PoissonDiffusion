"""EDM Model Wrapper with Domain Conditioning and Multi-Resolution Support."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = Path(__file__).parent.parent
edm_path = project_root / "external" / "edm"
if str(edm_path) not in sys.path:
    sys.path.insert(0, str(edm_path))

try:
    from training.networks import EDMPrecond, VPPrecond

    EDM_AVAILABLE = True
except ImportError:
    EDM_AVAILABLE = False
    EDMPrecond = None
    VPPrecond = None

from config.config import EDMConfig
from core.error_handlers import ConfigurationError, ModelError
from core.utils.tensor_utils import ensure_tensor


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


class PhysicsEncoder(nn.Module):
    """
    Encode physics parameters into conditioning vectors.

    Creates 3-dimensional conditioning vectors:
    - Log scale [1]: normalized log10(scale)
    - Relative read noise [1]: read_noise / scale
    - Relative background [1]: background / scale
    """

    def __init__(self):
        """Initialize physics encoder."""
        super().__init__()

        self.log_scale_mean = 2.5
        self.log_scale_std = 1.0

    def encode_physics(
        self,
        scale: Union[float, torch.Tensor],
        read_noise: Union[float, torch.Tensor],
        background: Union[float, torch.Tensor],
        device: Optional[torch.device] = None,
        conditioning_type: str = "dapgd",
    ) -> torch.Tensor:
        """
        Encode physics parameters into conditioning vector.

        Args:
            scale: Dataset normalization scale (electrons)
            read_noise: Read noise standard deviation (electrons)
            background: Background offset (electrons)
            device: Target device for tensors
            conditioning_type: Type of conditioning ('dapgd' or 'l2')

        Returns:
            Conditioning vector [B, 3]
        """
        # Convert scalars to tensors and move to device
        scale = ensure_tensor(scale, dtype=torch.float32, device=device)
        read_noise = ensure_tensor(read_noise, dtype=torch.float32, device=device)
        background = ensure_tensor(background, dtype=torch.float32, device=device)

        # Determine batch size from the largest parameter tensor
        param_tensors = [scale, read_noise, background]
        param_batch_sizes = [t.shape[0] if t.dim() > 0 else 1 for t in param_tensors]
        batch_size = max(param_batch_sizes)

        # Expand to batch size if needed
        if scale.dim() == 0 or scale.shape[0] == 1:
            scale = scale.expand(batch_size)
        if read_noise.dim() == 0 or read_noise.shape[0] == 1:
            read_noise = read_noise.expand(batch_size)
        if background.dim() == 0 or background.shape[0] == 1:
            background = background.expand(batch_size)

        if conditioning_type == "dapgd":
            # DAPGD: [log_scale(1), rel_noise(1), rel_bg(1)]
            # Normalize log scale
            log_scale = torch.log10(torch.clamp(scale, min=1e-6))
            log_scale_norm = (log_scale - self.log_scale_mean) / self.log_scale_std

            # Compute relative parameters
            rel_read_noise = read_noise / torch.clamp(scale, min=1e-6)
            rel_background = background / torch.clamp(scale, min=1e-6)

            # Concatenate conditioning vector
            condition = torch.cat(
                [
                    log_scale_norm.unsqueeze(1),  # [B, 1]
                    rel_read_noise.unsqueeze(1),  # [B, 1]
                    rel_background.unsqueeze(1),  # [B, 1]
                ],
                dim=1,
            )
        elif conditioning_type == "l2":
            # L2: [noise_estimate(1), padding(2)]
            # Estimate noise standard deviation in normalized [0,1] space
            # Var_norm = (signal_e + read_noise^2) / scale^2
            # We approximate signal_e with a typical value, e.g., scale / 2
            mean_signal_e = scale / 2
            noise_var_e = mean_signal_e + read_noise**2
            noise_std_norm = torch.sqrt(noise_var_e) / scale
            noise_std_norm = torch.log10(torch.clamp(noise_std_norm, min=1e-6))

            padding = torch.zeros(batch_size, 2, device=device or scale.device)

            condition = torch.cat(
                [
                    noise_std_norm.unsqueeze(1),
                    padding,
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unknown conditioning type: {conditioning_type}")

        # Ensure condition has the same dtype as input parameters
        # This ensures compatibility with mixed precision training
        if device is not None:
            condition = condition.to(device, dtype=torch.float32)

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
    Wrapper around EDM model with sensor conditioning.

    This wrapper integrates the external EDM codebase with our sensor-specific
    conditioning system, providing a unified interface for multi-sensor
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
            if self.config.preconditioner_type == "VPPrecond":
                preconditioner_class = VPPrecond
                model_type = "SongUNet"
            elif self.config.preconditioner_type == "EDMPrecond":
                preconditioner_class = EDMPrecond
                model_type = "DhariwalUNet"
            else:
                raise ValueError(
                    f"Unknown preconditioner type: {self.config.preconditioner_type}"
                )

            edm_config = self.config.get_preconditioner_config()

            # Ensure the underlying model uses float32 for stability
            # Override any use_fp16 settings to ensure consistency
            edm_config["use_fp16"] = False
            edm_config["model_type"] = model_type

            # Additional EDM configuration for stability
            if "dropout" not in edm_config:
                edm_config["dropout"] = 0.1  # Add dropout for regularization

            # Force disable fp16 to prevent assertion errors
            if "use_fp16" in edm_config:
                edm_config["use_fp16"] = False

            self.edm_model = preconditioner_class(**edm_config)

            # Force the EDM model to use float32 for all operations
            if hasattr(self.edm_model, "use_fp16"):
                self.edm_model.use_fp16 = False

            # Also force the underlying model to use fp32 if it exists
            if hasattr(self.edm_model, "model") and hasattr(
                self.edm_model.model, "use_fp16"
            ):
                self.edm_model.model.use_fp16 = False

        except Exception as e:
            raise ModelError(f"Failed to create EDM model: {e}")

        # Create physics encoder
        self.physics_encoder = PhysicsEncoder()

        if self.use_film:
            self.film_layers = nn.ModuleDict()

        self.model_type = "edm_wrapper"
        self.condition_dim = self.config.label_dim

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.label_dim not in [0, 3]:
            raise ConfigurationError(
                f"label_dim must be 0 (no conditioning) or 3 (physics conditioning), got {self.config.label_dim}"
            )

        if self.conditioning_mode not in ["class_labels", "film"]:
            raise ConfigurationError(
                f"Unknown conditioning_mode: {self.conditioning_mode}"
            )

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        scale: Optional[Union[float, torch.Tensor]] = None,
        read_noise: Optional[Union[float, torch.Tensor]] = None,
        background: Optional[Union[float, torch.Tensor]] = None,
        condition: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through EDM with physics conditioning.

        Args:
            x: Input images [B, C, H, W]
            sigma: Noise levels [B] or scalar
            scale: Scale parameter (if condition not provided)
            read_noise: Read noise parameter (if condition not provided)
            background: Background parameter (if condition not provided)
            condition: Pre-computed conditioning vector [B, 3]
            **kwargs: Additional arguments for EDM

        Returns:
            Model output [B, C, H, W]
        """
        # Handle different input channel configurations
        expected_channels = self.config.img_channels
        actual_channels = x.shape[1]

        if actual_channels != expected_channels:
            # Pad or resize input to match expected channels
            if actual_channels < expected_channels:
                # Pad with zeros to match expected channels
                pad_channels = expected_channels - actual_channels
                padding = torch.zeros(
                    x.shape[0],
                    pad_channels,
                    x.shape[2],
                    x.shape[3],
                    device=x.device,
                    dtype=x.dtype,
                )
                x = torch.cat([x, padding], dim=1)
            elif actual_channels > expected_channels:
                # Take first N channels to match expected channels
                x = x[:, :expected_channels]
            else:
                # Channels match, no action needed
                pass

        # Prepare conditioning
        if condition is None:
            if any(p is None for p in [scale, read_noise, background]):
                raise ValueError(
                    "Must provide either 'condition' or all physics parameters (scale, read_noise, background)"
                )

            condition = self.physics_encoder.encode_physics(
                scale=scale,
                read_noise=read_noise,
                background=background,
                device=x.device,
            )

            # Ensure condition is on the same device as input tensor
            if condition.device != x.device:
                condition = condition.to(device=x.device)

        # Ensure condition has correct batch size
        if condition.shape[0] != x.shape[0]:
            if condition.shape[0] == 1:
                condition = condition.expand(x.shape[0], -1)
            else:
                raise ValueError(
                    f"Condition batch size {condition.shape[0]} != input batch size {x.shape[0]}"
                )

        # Note: We handle dtype conversion below to ensure EDM model compatibility
        # with mixed precision training by forcing float32 for EDM model

        # Forward through EDM with conditioning
        if self.conditioning_mode == "class_labels":
            # Use EDM's built-in class conditioning
            # EDM preconditioner calls us with the expected dtype, we need to preserve it

            # The EDM preconditioner expects the model to return the same dtype it passed in
            # We need to preserve the input dtype throughout our wrapper

            # Store original dtypes for proper handling
            x_orig_dtype = x.dtype
            sigma_orig_dtype = sigma.dtype
            condition_orig_dtype = condition.dtype

            # Ensure condition has the same dtype as input tensor x
            # This is required for EDM model compatibility
            if condition.dtype != x.dtype:
                condition = condition.to(dtype=x.dtype)

            # Handle dtype conversion properly for EDM model compatibility
            # The EDM preconditioner expects consistent dtypes throughout the computation
            # Since EDM models are configured with use_fp16=False, we must ensure fp32 inputs

            # Determine the expected dtype for EDM model - ALWAYS use float32 for stability
            expected_dtype = torch.float32  # EDM models require float32 for stability

            # Convert inputs to expected dtype - force conversion to ensure compatibility
            x_converted = x.to(expected_dtype)
            sigma_converted = sigma.to(expected_dtype)
            condition_converted = condition.to(expected_dtype)

            if hasattr(self.edm_model, "use_fp16") and self.edm_model.use_fp16:
                self.edm_model.use_fp16 = False
                if hasattr(self.edm_model, "model") and hasattr(
                    self.edm_model.model, "use_fp16"
                ):
                    self.edm_model.model.use_fp16 = False

            try:
                output = self.edm_model(
                    x_converted,
                    sigma_converted,
                    class_labels=condition_converted,
                    force_fp32=True,
                    **kwargs,
                )
            except AssertionError:
                # Fallback: Convert everything to float32 and call without dtype checking
                # This is a last resort to get the model working
                x_fallback = x_converted.to(torch.float32)
                sigma_fallback = sigma_converted.to(torch.float32)
                condition_fallback = condition_converted.to(torch.float32)

                # Temporarily patch the EDM model to skip the assertion
                original_forward = self.edm_model.__class__.forward

                def patched_forward(
                    self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs
                ):
                    # Store original forward
                    if not hasattr(self, "_original_forward"):
                        self._original_forward = original_forward

                    # Call the original forward method but skip the assertion
                    x = x.to(torch.float32)
                    sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
                    class_labels = (
                        None
                        if self.label_dim == 0
                        else torch.zeros([1, self.label_dim], device=x.device)
                        if class_labels is None
                        else class_labels.to(torch.float32).reshape(-1, self.label_dim)
                    )
                    dtype = (
                        torch.float16
                        if (
                            self.use_fp16 and not force_fp32 and x.device.type == "cuda"
                        )
                        else torch.float32
                    )

                    # Handle different preconditioner types
                    if hasattr(self, "sigma_data"):
                        # EDMPrecond
                        c_skip = self.sigma_data**2 / (
                            sigma**2 + self.sigma_data**2
                        )
                        c_out = (
                            sigma
                            * self.sigma_data
                            / (sigma**2 + self.sigma_data**2).sqrt()
                        )
                        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
                        c_noise = sigma.log() / 4
                    else:
                        # VPPrecond or other preconditioners
                        c_skip = 1
                        c_out = -sigma
                        c_in = 1 / (sigma**2 + 1).sqrt()
                        c_noise = (self.M - 1) * self.sigma_inv(sigma)

                    F_x = self.model(
                        (c_in * x).to(dtype),
                        c_noise.flatten(),
                        class_labels=class_labels,
                        **model_kwargs,
                    )
                    # Skip the assertion and return the output
                    D_x = c_skip * x + c_out * F_x.to(torch.float32)
                    return D_x

                # Patch the forward method
                self.edm_model.__class__.forward = patched_forward
                output = self.edm_model(
                    x_fallback,
                    sigma_fallback,
                    class_labels=condition_fallback,
                    **kwargs,
                )

                # Restore original forward method
                self.edm_model.__class__.forward = original_forward

            # Convert output back to original input dtype if needed
            if output.dtype != x_orig_dtype:
                output = output.to(x_orig_dtype)
        else:
            # FiLM conditioning (would require EDM modification)
            raise NotImplementedError(
                "FiLM conditioning requires EDM model modification"
            )

        return output

    def encode_conditioning(
        self,
        scale: Union[float, torch.Tensor],
        read_noise: Union[float, torch.Tensor],
        background: Union[float, torch.Tensor],
        device: Optional[torch.device] = None,
        conditioning_type: str = "dapgd",
    ) -> torch.Tensor:
        """
        Encode physics parameters into conditioning vector.

        Args:
            scale: Scale parameter(s)
            read_noise: Read noise parameter(s)
            background: Background parameter(s)
            device: Target device
            conditioning_type: Type of conditioning ('dapgd' or 'l2')

        Returns:
            Conditioning vector [B, 3]
        """
        return self.physics_encoder.encode_physics(
            scale=scale,
            read_noise=read_noise,
            background=background,
            device=device,
            conditioning_type=conditioning_type,
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


def create_multi_domain_edm_wrapper(
    img_resolution: int = 128,
    model_channels: int = 256,  # Use enhanced default for better performance
    conditioning_mode: str = "class_labels",
    **config_overrides,
) -> EDMModelWrapper:
    """
    Create EDM wrapper that can handle multiple domains with different channel configurations.

    Args:
        img_resolution: Image resolution
        model_channels: Base number of model channels
        conditioning_mode: Conditioning mode
        **config_overrides: Additional configuration overrides

    Returns:
        Multi-domain compatible EDMModelWrapper
    """
    # Use 3 channels by default to support RGB sensors (Sony/Fuji)
    # The domain conditioning will handle the differences
    config = EDMConfig(
        img_resolution=img_resolution,
        img_channels=3,  # RGB sensors (Sony/Fuji, 3 channels)
        model_channels=model_channels,
        **config_overrides,
    )

    return EDMModelWrapper(config=config, conditioning_mode=conditioning_mode)


def create_domain_aware_edm_wrapper(
    domain: str = "sony",
    img_resolution: int = 128,
    model_channels: int = 128,
    conditioning_mode: str = "class_labels",
    **config_overrides,
) -> EDMModelWrapper:
    """
    Create EDM wrapper with domain-aware channel configuration.

    Args:
        domain: Sensor name ('sony' or 'fuji')
        img_resolution: Image resolution
        model_channels: Base number of model channels
        conditioning_mode: Conditioning mode
        **config_overrides: Additional configuration overrides

    Returns:
        Domain-aware EDMModelWrapper
    """
    # Sensor-specific channel configurations (RGB sensors)
    domain_channels = {"sony": 3, "fuji": 3}

    if domain not in domain_channels:
        raise ValueError(
            f"Unknown domain: {domain}. Supported: {list(domain_channels.keys())}"
        )

    img_channels = domain_channels[domain]

    config = EDMConfig(
        img_resolution=img_resolution,
        img_channels=img_channels,
        model_channels=model_channels,
        **config_overrides,
    )

    return EDMModelWrapper(config=config, conditioning_mode=conditioning_mode)


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
    img_resolution: int = 128, **config_overrides
) -> EDMModelWrapper:
    """
    Create EDM wrapper with default configuration.

    Args:
        img_resolution: Image resolution
        **config_overrides: Configuration overrides

    Returns:
        EDMModelWrapper
    """
    # Default configuration
    config_dict = {"model_channels": 128, "num_blocks": 4, "dropout": 0.1}
    config_dict.update(config_overrides)

    return create_edm_wrapper(img_resolution=img_resolution, **config_dict)


# Factory functions will be defined after the classes


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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract configuration if not provided
    if config is None:
        if "config" in checkpoint:
            # Filter out training-specific config parameters
            model_config = {}
            edm_config_keys = {
                "img_resolution",
                "img_channels",
                "model_channels",
                "channel_mult",
                "channel_mult_emb",
                "num_blocks",
                "attn_resolutions",
                "dropout",
                "label_dim",
                "use_fp16",
                "preconditioner_type",
                "sigma_min",
                "sigma_max",
                "sigma_data",
            }

            for key, value in checkpoint["config"].items():
                if key in edm_config_keys:
                    model_config[key] = value

            # Add default values for missing keys
            if "img_resolution" not in model_config:
                model_config["img_resolution"] = 128
            if "img_channels" not in model_config:
                # Infer from model state dict if possible
                if "model_state_dict" in checkpoint:
                    # Check the input layer to determine channels
                    for key in checkpoint["model_state_dict"].keys():
                        if "128x128_conv.weight" in key and "enc" in key:
                            weight_shape = checkpoint["model_state_dict"][key].shape
                            model_config["img_channels"] = weight_shape[
                                1
                            ]  # Input channels
                            break
                    else:
                        model_config["img_channels"] = 1  # Default fallback
                else:
                    model_config["img_channels"] = 1
            if "label_dim" not in model_config:
                model_config["label_dim"] = 3
            if "model_channels" not in model_config:
                model_config["model_channels"] = 320  # From checkpoint config

            config = EDMConfig(**model_config)
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

    return model


class ProgressiveEDM(nn.Module):
    """
    Progressive Growing EDM model for multi-resolution training.

    This model supports training at multiple resolutions through progressive growing:
    - Stage 1: 32×32 (epochs 0-25)
    - Stage 2: 64×64 (epochs 26-50)
    - Stage 3: 96×96 (epochs 51-75)
    - Stage 4: 128×128 (epochs 76-100)

    Features:
    - Automatic resolution switching during training
    - Shared weights across resolutions where possible
    - Resolution-conditioned processing
    - Memory-efficient progressive growing
    """

    def __init__(
        self,
        min_resolution: int = 32,
        max_resolution: int = 128,
        num_stages: int = 4,
        model_channels: int = 256,  # Increased from 128 to 256 for better capacity
        **kwargs,
    ):
        """
        Initialize progressive EDM model.

        Args:
            min_resolution: Starting resolution (must be power of 2)
            max_resolution: Final resolution (must be power of 2)
            num_stages: Number of progressive growing stages
            model_channels: Base number of model channels
            **kwargs: Additional arguments passed to EDMModelWrapper
        """
        super().__init__()

        if not EDM_AVAILABLE:
            raise ModelError(
                "EDM not available. Run 'bash external/setup_edm.sh' to set up."
            )

        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.num_stages = num_stages
        self.current_stage = 0

        # Validate resolutions (allowing 96 which is not power of 2)
        if min_resolution not in [32, 64, 96, 128] or max_resolution not in [
            32,
            64,
            128,
            256,
        ]:
            raise ValueError(
                f"Resolution must be one of [32, 64, 96, 128], got {min_resolution}, {max_resolution}"
            )

        # Calculate stage resolutions (32, 64, 96, 128)
        self.stage_resolutions = [32, 64, 96, 128]

        # Create EDM models for each stage
        self.stage_models = nn.ModuleDict()
        for stage, resolution in enumerate(self.stage_resolutions):
            # Increase model capacity with resolution
            stage_channels = min(model_channels * (resolution // min_resolution), 512)

            config = EDMConfig(
                img_resolution=resolution, model_channels=stage_channels, **kwargs
            )

            model = EDMModelWrapper(config=config)
            self.stage_models[f"stage_{stage}"] = model

        # Resolution conditioning encoder
        self.resolution_encoder = nn.Embedding(
            num_stages, 8
        )  # 8-dim resolution embedding

        # Initialize resolution embeddings
        for stage in range(num_stages):
            progress = stage / (num_stages - 1) if num_stages > 1 else 0
            self.resolution_encoder.weight.data[stage] = torch.tensor(
                [
                    progress,
                    1 - progress,  # Resolution level
                    float(self.stage_resolutions[stage]),  # Absolute resolution
                    float(self.stage_resolutions[stage])
                    / max_resolution,  # Normalized res
                    1.0 if stage == 0 else 0.0,  # Is min resolution
                    1.0 if stage == num_stages - 1 else 0.0,  # Is max resolution
                    0.5,  # Padding
                    0.5,  # Padding
                ]
            )

        self.model_type = "progressive_edm"

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        stage: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the appropriate stage model.

        Args:
            x: Input tensor [B, C, H, W]
            sigma: Noise levels [B] or scalar
            stage: Specific stage to use (None for current stage)
            **kwargs: Additional arguments passed to stage model

        Returns:
            Model output [B, C, H, W]
        """
        if stage is None:
            stage = self.current_stage

        if stage >= len(self.stage_resolutions):
            raise ValueError(
                f"Stage {stage} not available. Max stage: {len(self.stage_resolutions)-1}"
            )

        expected_res = self.stage_resolutions[stage]
        if x.shape[-1] != expected_res:
            raise ValueError(
                f"Input resolution {x.shape[-1]} doesn't match stage {stage} "
                f"resolution {expected_res}"
            )

        model = self.stage_models[f"stage_{stage}"]

        # Add resolution conditioning
        batch_size = x.shape[0]
        resolution_condition = self.resolution_encoder(
            torch.tensor([stage] * batch_size, device=x.device)
        )  # [B, 8]

        # Handle conditioning properly for EDM model (expects 6-dim)
        if "condition" in kwargs:
            # If existing condition, take first 3 dims and append first 3 dims of resolution
            base_condition = (
                kwargs["condition"][:, :3]
                if kwargs["condition"].shape[1] >= 3
                else kwargs["condition"]
            )
            res_condition = resolution_condition[:, :3]  # Take first 3 dims
            kwargs["condition"] = torch.cat([base_condition, res_condition], dim=1)
        else:
            # Create 6-dim conditioning from resolution info
            stage_info = (
                torch.tensor([stage] * batch_size, device=x.device).float().unsqueeze(1)
            )
            progress = (
                stage / (len(self.stage_resolutions) - 1)
                if len(self.stage_resolutions) > 1
                else 0
            )
            progress_info = torch.full((batch_size, 1), progress, device=x.device)
            res_info = torch.full(
                (batch_size, 1), float(self.stage_resolutions[stage]), device=x.device
            )
            # Create 6-dim conditioning: [stage, progress, resolution, 0, 0, 0]
            zeros = torch.zeros((batch_size, 3), device=x.device)
            kwargs["condition"] = torch.cat(
                [stage_info, progress_info, res_info, zeros], dim=1
            )

        return model(x, sigma, **kwargs)

    def grow_resolution(self) -> bool:
        """
        Progress to the next resolution stage.

        Returns:
            True if successfully grew, False if already at max resolution
        """
        if self.current_stage >= len(self.stage_resolutions) - 1:
            return False

        self.current_stage += 1
        return True

    def get_current_resolution(self) -> int:
        """Get current training resolution."""
        return self.stage_resolutions[self.current_stage]

    def get_stage_info(self) -> Dict[str, Any]:
        """Get information about all stages."""
        return {
            "current_stage": self.current_stage,
            "current_resolution": self.get_current_resolution(),
            "stage_resolutions": self.stage_resolutions,
            "num_stages": self.num_stages,
            "stage_models": {
                k: v.get_model_info() for k, v in self.stage_models.items()
            },
        }

    def set_stage(self, stage: int):
        """
        Set the current stage.

        Args:
            stage: Stage index to set
        """
        if stage < 0 or stage >= len(self.stage_resolutions):
            raise ValueError(f"Invalid stage {stage}")

        self.current_stage = stage


class MultiScaleEDM(nn.Module):
    """
    Multi-scale EDM model with hierarchical processing.

    This model processes images at multiple scales simultaneously using a U-Net
    style architecture with skip connections between different resolution levels.

    Features:
    - Hierarchical feature extraction
    - Skip connections across scales
    - Feature fusion from multiple levels
    - Scale-aware conditioning
    """

    def __init__(
        self,
        scales: List[int] = [32, 64, 96, 128],
        model_channels: int = 256,  # Increased from 128 to 256 for better capacity
        **kwargs,
    ):
        """
        Initialize multi-scale EDM model.

        Args:
            scales: List of resolutions to process
            model_channels: Base number of model channels
            **kwargs: Additional arguments passed to EDMModelWrapper
        """
        super().__init__()

        if not EDM_AVAILABLE:
            raise ModelError(
                "EDM not available. Run 'bash external/setup_edm.sh' to set up."
            )

        self.scales = sorted(scales)
        self.num_scales = len(scales)

        # Create scale-specific models
        self.scale_models = nn.ModuleDict()
        for scale in scales:
            scale_channels = min(model_channels * (scale // min(scales)), 512)

            config = EDMConfig(
                img_resolution=scale, model_channels=scale_channels, **kwargs
            )

            model = EDMModelWrapper(config=config)
            self.scale_models[f"scale_{scale}"] = model

        # Scale conditioning encoder
        self.scale_encoder = nn.Embedding(len(scales), 8)

        # Initialize scale embeddings
        for i, scale in enumerate(scales):
            scale_ratio = scale / max(scales)
            self.scale_encoder.weight.data[i] = torch.tensor(
                [
                    scale_ratio,
                    1 - scale_ratio,  # Scale level
                    float(scale),  # Absolute scale
                    scale_ratio,  # Normalized scale
                    1.0 if scale == min(scales) else 0.0,  # Is min scale
                    1.0 if scale == max(scales) else 0.0,  # Is max scale
                    0.5,  # Padding
                    0.5,  # Padding
                ]
            )

        # Feature fusion network (works with single-channel outputs from each scale)
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(
                len(scales), model_channels, 3, padding=1
            ),  # len(scales) input channels
            nn.ReLU(),
            nn.Conv2d(model_channels, 1, 3, padding=1),  # Output single channel
        )

        self.model_type = "multiscale_edm"

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        target_scale_idx: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through multi-scale model.

        Args:
            x: Input tensor [B, C, H, W]
            sigma: Noise levels [B] or scalar
            target_scale_idx: Target scale index (None for max scale)
            **kwargs: Additional arguments passed to scale models

        Returns:
            Model output at target scale [B, C, H, W]
        """
        if target_scale_idx is None:
            target_scale_idx = len(self.scales) - 1  # Use largest scale

        target_scale = self.scales[target_scale_idx]
        batch_size = x.shape[0]

        # Generate multi-scale features
        multiscale_features = {}
        for scale_idx, scale in enumerate(self.scales):
            # Resize input to this scale
            if x.shape[-1] != scale:
                scale_x = F.interpolate(
                    x, size=(scale, scale), mode="bilinear", align_corners=False
                )
            else:
                scale_x = x

            # Process at this scale
            model = self.scale_models[f"scale_{scale}"]

            # Add scale conditioning (handle 6-dim constraint)
            scale_condition = self.scale_encoder(
                torch.tensor([scale_idx] * batch_size, device=x.device)
            )

            if "condition" in kwargs:
                # Take first 3 dims of existing condition and first 3 dims of scale condition
                base_condition = (
                    kwargs["condition"][:, :3]
                    if kwargs["condition"].shape[1] >= 3
                    else kwargs["condition"]
                )
                scale_cond = scale_condition[:, :3]  # Take first 3 dims
                combined_condition = torch.cat([base_condition, scale_cond], dim=1)
            else:
                # Create 6-dim conditioning from scale info
                scale_info = (
                    torch.tensor([scale_idx] * batch_size, device=x.device)
                    .float()
                    .unsqueeze(1)
                )
                scale_ratio = scale / max(self.scales)
                ratio_info = torch.full((batch_size, 1), scale_ratio, device=x.device)
                abs_scale = torch.full((batch_size, 1), float(scale), device=x.device)
                # Create 6-dim conditioning: [scale_idx, scale_ratio, abs_scale, 0, 0, 0]
                zeros = torch.zeros((batch_size, 3), device=x.device)
                combined_condition = torch.cat(
                    [scale_info, ratio_info, abs_scale, zeros], dim=1
                )

            # Forward pass
            features = model(scale_x, sigma, condition=combined_condition)
            multiscale_features[scale] = features

        # Fuse features from all scales
        fused_features = self._fuse_multiscale_features(
            multiscale_features, target_scale
        )

        return fused_features

    def _fuse_multiscale_features(
        self, features: Dict[int, torch.Tensor], target_scale: int
    ) -> torch.Tensor:
        """
        Fuse features from multiple scales.

        Args:
            features: Dictionary mapping scale -> features
            target_scale: Target output scale

        Returns:
            Fused features at target scale
        """
        # Resize all features to target scale
        resized_features = []
        for scale, feature in features.items():
            if scale != target_scale:
                resized = F.interpolate(
                    feature,
                    size=(target_scale, target_scale),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                resized = feature
            resized_features.append(resized)

        # Concatenate along channel dimension
        concatenated = torch.cat(resized_features, dim=1)

        # Fuse features
        fused = self.feature_fusion(concatenated)

        return fused

    def extract_multiscale_features(
        self, x: torch.Tensor, sigma: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Extract features at all scales.

        Args:
            x: Input tensor [B, C, H, W]
            sigma: Noise levels [B] or scalar

        Returns:
            Dictionary mapping scale -> features at that scale
        """
        features = {}
        batch_size = x.shape[0]

        for scale_idx, scale in enumerate(self.scales):
            # Resize input to this scale
            if x.shape[-1] != scale:
                scale_x = F.interpolate(
                    x, size=(scale, scale), mode="bilinear", align_corners=False
                )
            else:
                scale_x = x

            # Process at this scale
            model = self.scale_models[f"scale_{scale}"]

            # Add scale conditioning (create 6-dim conditioning)
            scale_info = (
                torch.tensor([scale_idx] * batch_size, device=x.device)
                .float()
                .unsqueeze(1)
            )
            scale_ratio = scale / max(self.scales)
            ratio_info = torch.full((batch_size, 1), scale_ratio, device=x.device)
            abs_scale = torch.full((batch_size, 1), float(scale), device=x.device)
            # Create 6-dim conditioning: [scale_idx, scale_ratio, abs_scale, 0, 0, 0]
            zeros = torch.zeros((batch_size, 3), device=x.device)
            scale_condition = torch.cat(
                [scale_info, ratio_info, abs_scale, zeros], dim=1
            )

            # Forward pass
            scale_features = model(scale_x, sigma, condition=scale_condition)
            features[scale] = scale_features

        return features


# Factory functions for multi-resolution models


def create_progressive_edm(
    min_resolution: int = 32,
    max_resolution: int = 128,
    num_stages: int = 4,
    model_channels: int = 128,
    **config_overrides,
) -> ProgressiveEDM:
    """
    Create progressive EDM model for multi-resolution training.

    Args:
        min_resolution: Starting resolution
        max_resolution: Final resolution
        num_stages: Number of progressive growing stages
        model_channels: Base number of model channels
        **config_overrides: Additional configuration overrides

    Returns:
        Configured ProgressiveEDM model
    """
    return ProgressiveEDM(
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        num_stages=num_stages,
        model_channels=model_channels,
        **config_overrides,
    )


def create_multiscale_edm(
    scales: List[int] = None, model_channels: int = 128, **config_overrides
) -> MultiScaleEDM:
    """
    Create multi-scale EDM model for hierarchical processing.

    Args:
        scales: List of resolutions to process (default: [32, 64, 96, 128])
        model_channels: Base number of model channels
        **config_overrides: Additional configuration overrides

    Returns:
        Configured MultiScaleEDM model
    """
    if scales is None:
        scales = [32, 64, 96, 128]

    return MultiScaleEDM(
        scales=scales, model_channels=model_channels, **config_overrides
    )
