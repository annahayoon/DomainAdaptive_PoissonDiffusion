"""
Configuration classes for Poisson-Gaussian likelihood guidance.

This module defines configuration dataclasses that control the behavior
of the guidance system, including different modes, scheduling options,
and numerical stability parameters.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


@dataclass
class GuidanceConfig:
    """
    Configuration for Poisson-Gaussian likelihood guidance.

    This class controls all aspects of the guidance computation including
    the mathematical mode, scheduling, and numerical stability measures.
    """

    # Core guidance mode
    mode: Literal["wls", "exact"] = "wls"
    """
    Guidance computation mode:
    - 'wls': Weighted Least Squares approximation (recommended)
    - 'exact': Exact heteroscedastic likelihood with variance derivatives
    """

    # Gamma scheduling for guidance weighting
    gamma_schedule: Literal["sigma2", "linear", "const"] = "sigma2"
    """
    Guidance weight scheduling γ(σ):
    - 'sigma2': γ(σ) = κ·σ² (default, balances prior and likelihood)
    - 'linear': γ(σ) = κ·σ (linear decay)
    - 'const': γ(σ) = κ (constant weight)
    """

    # Guidance strength parameter
    kappa: float = 0.5
    """
    Base guidance strength parameter κ.
    Higher values give more weight to data fidelity.
    Typical range: [0.1, 2.0]
    """

    # Numerical stability
    gradient_clip: float = 100.0
    """
    Maximum absolute value for guidance gradients.
    Prevents numerical instability in extreme cases.
    Higher values preserve physics accuracy better.
    """

    variance_eps: float = 0.01
    """
    Small epsilon added to variance to prevent division by zero.
    Should be much smaller than typical noise variance.
    Smaller values improve accuracy.
    """

    # Advanced options
    enable_masking: bool = True
    """Whether to use pixel masks to exclude invalid regions."""

    normalize_gradients: bool = False
    """Whether to normalize gradients by their norm."""

    adaptive_kappa: bool = False
    """Whether to adapt kappa based on signal level (experimental)."""

    # Diagnostic options
    collect_diagnostics: bool = True
    """Whether to collect diagnostic statistics during computation."""

    max_diagnostics: int = 1000
    """Maximum number of diagnostic samples to store."""

    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate that all parameters are reasonable."""
        errors = []

        # Validate kappa
        if self.kappa <= 0:
            errors.append(f"kappa must be positive, got {self.kappa}")
        if self.kappa > 10:
            errors.append(f"kappa is very large ({self.kappa}), may cause instability")

        # Validate numerical parameters
        if self.gradient_clip <= 0:
            errors.append(f"gradient_clip must be positive, got {self.gradient_clip}")
        if self.variance_eps <= 0:
            errors.append(f"variance_eps must be positive, got {self.variance_eps}")
        if self.variance_eps > 1.0:
            errors.append(
                f"variance_eps is large ({self.variance_eps}), may affect accuracy"
            )

        # Validate diagnostic parameters
        if self.max_diagnostics <= 0:
            errors.append(
                f"max_diagnostics must be positive, got {self.max_diagnostics}"
            )

        if errors:
            raise ValueError(f"Invalid GuidanceConfig parameters: {'; '.join(errors)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "mode": self.mode,
            "gamma_schedule": self.gamma_schedule,
            "kappa": self.kappa,
            "gradient_clip": self.gradient_clip,
            "variance_eps": self.variance_eps,
            "enable_masking": self.enable_masking,
            "normalize_gradients": self.normalize_gradients,
            "adaptive_kappa": self.adaptive_kappa,
            "collect_diagnostics": self.collect_diagnostics,
            "max_diagnostics": self.max_diagnostics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuidanceConfig":
        """Create configuration from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize configuration to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "GuidanceConfig":
        """Deserialize configuration from JSON."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def copy(self, **overrides) -> "GuidanceConfig":
        """Create a copy with optional parameter overrides."""
        data = self.to_dict()
        data.update(overrides)
        return self.from_dict(data)


# Predefined configurations for common use cases
class GuidancePresets:
    """Predefined guidance configurations for common scenarios."""

    @staticmethod
    def default() -> GuidanceConfig:
        """Default configuration - balanced performance and accuracy."""
        return GuidanceConfig()

    @staticmethod
    def high_fidelity() -> GuidanceConfig:
        """High data fidelity - strong guidance for clean reconstruction."""
        return GuidanceConfig(
            mode="exact", kappa=1.0, gradient_clip=20.0, normalize_gradients=True
        )

    @staticmethod
    def robust() -> GuidanceConfig:
        """Robust configuration - stable in noisy conditions."""
        return GuidanceConfig(
            mode="wls",
            kappa=0.3,
            gradient_clip=5.0,
            variance_eps=0.5,
            adaptive_kappa=True,
        )

    @staticmethod
    def fast() -> GuidanceConfig:
        """Fast configuration - minimal overhead."""
        return GuidanceConfig(
            mode="wls", kappa=0.5, collect_diagnostics=False, enable_masking=False
        )

    @staticmethod
    def debug() -> GuidanceConfig:
        """Debug configuration - maximum diagnostics."""
        return GuidanceConfig(
            collect_diagnostics=True,
            max_diagnostics=10000,
            gradient_clip=1000.0,  # Don't clip for debugging
        )

    @staticmethod
    def for_domain(domain: str) -> GuidanceConfig:
        """
        Domain-specific configuration.

        Args:
            domain: 'photography', 'microscopy', or 'astronomy'

        Returns:
            Optimized configuration for the domain
        """
        if domain == "photography":
            # Photography: balance quality and speed
            return GuidanceConfig(mode="wls", kappa=0.6, gamma_schedule="sigma2")

        elif domain == "microscopy":
            # Microscopy: high precision for low photon counts
            return GuidanceConfig(
                mode="wls",
                kappa=0.8,
                gamma_schedule="sigma2",
                gradient_clip=5.0,
                variance_eps=0.05,
                adaptive_kappa=True,
                normalize_gradients=False,
            )

        elif domain == "astronomy":
            # Astronomy: tuned for extreme low-light with reduced white FOV artifacts
            return GuidanceConfig(
                mode="wls",  # Changed from "exact" to "wls" for stability
                kappa=0.2,   # Reduced from 1.0 to 0.2 to prevent over-amplification
                gamma_schedule="sigma2",  # Changed from "linear" to "sigma2" for better balance
                gradient_clip=5.0,  # Reduced from 20.0 to 5.0 for stability
                variance_eps=0.1,   # Increased from 0.01 to 0.1 for dark regions
                adaptive_kappa=True,
                normalize_gradients=False,  # Disabled to prevent over-normalization
            )

        else:
            raise ValueError(
                f"Unknown domain: {domain}. "
                f"Supported domains: photography, microscopy, astronomy"
            )


# Convenience function
def create_guidance_config(preset: str = "default", **overrides) -> GuidanceConfig:
    """
    Create guidance configuration from preset with optional overrides.

    Args:
        preset: Preset name ('default', 'high_fidelity', 'robust', 'fast', 'debug')
        **overrides: Parameter overrides

    Returns:
        Configured GuidanceConfig instance
    """
    preset_func = getattr(GuidancePresets, preset, None)
    if preset_func is None:
        available = [
            name
            for name in dir(GuidancePresets)
            if not name.startswith("_") and name != "for_domain"
        ]
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")

    config = preset_func()
    if overrides:
        config = config.copy(**overrides)

    return config
