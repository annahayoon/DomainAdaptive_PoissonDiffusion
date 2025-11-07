"""Configuration classes and constants for the Poisson-Gaussian Diffusion project.

This module centralizes all configuration-related code including:
- Base configuration classes
- Model configurations (EDM, sampling)
- Preprocessing configurations
- Visualization configurations
"""

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

from core.error_handlers import ConfigurationError

logger = logging.getLogger(__name__)

# ============================================================================
# Base Configuration Classes
# ============================================================================


@dataclass
class BaseConfig:
    """Base configuration class with common functionality."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self) -> str:
        """Convert config to YAML string."""
        if yaml is None:
            raise ImportError("yaml module is required for to_yaml()")
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """Create config from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "BaseConfig":
        """Create config from YAML string."""
        if yaml is None:
            raise ImportError("yaml module is required for from_yaml()")
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)


# ============================================================================
# Model Configurations
# ============================================================================


@dataclass
class EDMConfig:
    """Configuration for EDM model parameters."""

    # Image parameters
    img_resolution: int = 256
    img_channels: int = 1

    # Model architecture - Scaled up for research-level performance
    model_channels: int = 256  # Increased from 128 to 256 for better capacity
    channel_mult: List[int] = None
    channel_mult_emb: int = 6  # Increased from 4 to 6 for larger embedding
    num_blocks: int = 6  # Increased from 4 to 6 for deeper network
    attn_resolutions: List[int] = None
    dropout: float = 0.1  # Small dropout for regularization

    # Conditioning
    label_dim: int = 3  # Our 3D physics conditioning
    use_fp16: bool = False

    # Preconditioner type for stability
    preconditioner_type: str = "VPPrecond"  # "EDMPrecond" or "VPPrecond"

    # EDM-specific parameters
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5

    def __post_init__(self):
        """Set default values for list parameters - Enhanced for research-level capacity."""
        if self.channel_mult is None:
            # Simplified channel scaling: 256→128→64→32 (3 downsampling steps)
            self.channel_mult = [1, 2, 4]
        if self.attn_resolutions is None:
            # No self-attention layers - pure convolutional U-Net
            self.attn_resolutions = []

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

    def get_preconditioner_config(self) -> Dict[str, Any]:
        """Get configuration specific to the selected preconditioner."""
        base_config = self.to_dict()

        if self.preconditioner_type == "VPPrecond":
            # VPPrecond uses different parameters than EDMPrecond
            # Remove EDM-specific parameters that SongUNet doesn't accept
            base_config.pop("sigma_data", None)
            base_config.pop("sigma_min", None)
            base_config.pop("sigma_max", None)

            # Add VP-specific parameters
            base_config["beta_d"] = 19.9
            base_config["beta_min"] = 0.1
            base_config["M"] = 1000
            base_config["epsilon_t"] = 1e-5
        elif self.preconditioner_type == "EDMPrecond":
            # EDMPrecond needs these parameters
            pass  # Keep the default parameters
        else:
            raise ValueError(f"Unknown preconditioner type: {self.preconditioner_type}")

        return base_config


@dataclass
class BaseSamplingConfig:
    """Base configuration for sampling with common parameters and validation."""

    num_steps: int = 18
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    S_churn: float = 0.0
    S_min: float = 0.0
    S_max: float = float("inf")
    S_noise: float = 1.0
    guidance_scale: float = 1.0
    guidance_start_step: int = 0
    guidance_end_step: Optional[int] = None
    clip_denoised: bool = True
    clip_range: Tuple[float, float] = (0.0, 1.0)

    def _validate_common(self):
        """Validate common sampling parameters."""
        if self.num_steps <= 0:
            raise ConfigurationError(
                f"num_steps must be positive, got {self.num_steps}"
            )

        if self.sigma_min >= self.sigma_max:
            raise ConfigurationError(
                f"sigma_min ({self.sigma_min}) must be < sigma_max ({self.sigma_max})"
            )

        if (
            self.guidance_end_step is not None
            and self.guidance_end_step <= self.guidance_start_step
        ):
            raise ConfigurationError("guidance_end_step must be > guidance_start_step")


@dataclass
class SamplingConfig(BaseSamplingConfig):
    """Unified configuration for EDM sampling with both native and solver modes."""

    # Solver mode: "native" uses exact EDM algorithm, others use flexible solvers
    solver: str = "heun"  # Options: "native", "euler", "heun", "dpm"
    # Guidance type: "pg" for Poisson-Gaussian, "l2" for L2 guidance
    guidance_type: str = "pg"  # Options: "pg", "l2"
    # Diagnostic and intermediate saving
    save_intermediates: bool = False
    collect_diagnostics: bool = True
    # Exposure ratio support (for backward compatibility with sample/sampler.py)
    exposure_ratio: float = 1.0
    # Disable Heun correction step (for backward compatibility)
    no_heun: bool = False
    # Guidance level: "x0" applies guidance to denoised estimate, "score" applies to score
    guidance_level: str = "x0"  # Options: "x0", "score"

    def __post_init__(self):
        """Validate configuration."""
        self._validate_common()
        if self.solver not in ["native", "euler", "heun", "dpm"]:
            raise ConfigurationError(
                f"Unknown solver: {self.solver}. Must be one of: native, euler, heun, dpm"
            )
        if self.guidance_type not in ["pg", "l2"]:
            raise ConfigurationError(
                f"Unknown guidance type: {self.guidance_type}. Must be one of: pg, l2"
            )
        if self.guidance_level not in ["x0", "score"]:
            raise ConfigurationError(
                f"Unknown guidance level: {self.guidance_level}. Must be one of: x0, score"
            )


# ============================================================================
# Visualization Configurations
# ============================================================================


@dataclass
class VizStepConfig:
    """Configuration for a visualization step."""

    key: str  # Key to extract from data dict
    label: str  # Row label
    precision: int = 3  # Decimal precision for stats
    extract_fn: Optional[Callable] = None  # Custom extraction function


# ============================================================================
# Preprocessing Configurations
# ============================================================================

# Base data paths - configurable via environment variable
BASE_DATA_PATH = Path(os.environ.get("DATA_PATH", "./data"))

# Validate paths exist on import
if not BASE_DATA_PATH.exists():
    logger.warning(
        f"DATA_PATH not found: {BASE_DATA_PATH}\n"
        f"To fix, set environment variable: export DATA_PATH=/path/to/data\n"
        f"Or pass --data_path argument to the pipeline script"
    )

RAW_DATA_PATH = BASE_DATA_PATH / "raw"
SONY_PATH = RAW_DATA_PATH / "SID" / "Sony"
FUJI_PATH = RAW_DATA_PATH / "SID" / "Fuji"

# Tile size for all domains
TILE_SIZE = 256

TILE_CONFIGS = {
    "sony": {
        "tile_size": 256,
        "target_tiles": 216,
        "target_grid": (12, 18),
    },
    "fuji": {
        "tile_size": 256,
        "target_tiles": 96,
        "target_grid": (8, 12),
    },
    "s6": {
        "tile_size": 256,
        "target_grid": (11, 21),
        "target_tiles": 231,
    },
    "g4": {
        "tile_size": 256,
        "target_grid": (11, 21),
        "target_tiles": 231,
    },
    "n6": {
        "tile_size": 256,
        "target_grid": (13, 17),
        "target_tiles": 221,
    },
    "gp": {
        "tile_size": 256,
        "target_grid": (12, 16),
        "target_tiles": 192,
    },
    "ip": {
        "tile_size": 256,
        "target_grid": (12, 16),
        "target_tiles": 192,
    },
}

# Note: SENSOR_RANGES and BLACK_LEVELS have been moved to core.sensor_config
# to avoid duplication. Use get_sensor_config(), get_sensor_range(), etc. from sensor_config.

CAMERA_CONFIGS = {
    "sony": {
        "extension": ".ARW",
        "base_dir": "Sony",
        "channels": 4,
    },
    "fuji": {
        "extension": ".RAF",
        "base_dir": "Fuji",
        "channels": 9,
    },
    "s6": {
        "extension": ".MAT",
        "base_dir": "SIDD",
        "channels": 3,
        "dimensions": (3000, 5328),
    },
    "g4": {
        "extension": ".MAT",
        "base_dir": "SIDD",
        "channels": 3,
        "dimensions": (2988, 5312),
    },
    "n6": {
        "extension": ".MAT",
        "base_dir": "SIDD",
        "channels": 3,
        "dimensions": (3120, 4208),
    },
    "gp": {
        "extension": ".MAT",
        "base_dir": "SIDD",
        "channels": 3,
        "dimensions": (3044, 4048),
    },
    "ip": {
        "extension": ".MAT",
        "base_dir": "SIDD",
        "channels": 3,
        "dimensions": (3024, 4032),
    },
}

# ============================================================================
# Sampling Default Constants
# ============================================================================

# Default sampling parameters (matching BaseSamplingConfig defaults)
DEFAULT_NUM_STEPS = 18
DEFAULT_SIGMA_MIN = 0.002
DEFAULT_SIGMA_MAX = 80.0
DEFAULT_RHO = 7.0

# Default guidance parameters
DEFAULT_SIGMA_R = 3.0
DEFAULT_KAPPA = 0.1
DEFAULT_TAU = 0.01
DEFAULT_CONSERVATIVE_FACTOR = 1.0

# Sensor name mapping for sampling
SENSOR_NAME_MAPPING = {
    "sony": "sony_a7s_ii",
    "fuji": "fuji_xt2",
}

SUPPORTED_SENSORS = {"sony", "fuji"}
