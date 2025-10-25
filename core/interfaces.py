"""
Abstract base classes and interfaces for the core components.

This module defines the contracts that all implementations must follow,
ensuring consistency and enabling easy testing and extension.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class MetadataContainer(ABC):
    """Abstract base class for metadata containers."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        pass

    @abstractmethod
    def to_json(self) -> str:
        """Serialize metadata to JSON string."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetadataContainer":
        """Create metadata from dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, json_str: str) -> "MetadataContainer":
        """Deserialize metadata from JSON string."""
        pass


class Transform(ABC):
    """Abstract base class for image transformations."""

    @abstractmethod
    def forward(
        self, image: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, MetadataContainer]:
        """
        Apply forward transformation.

        Args:
            image: Input image tensor [B, C, H, W]
            **kwargs: Additional metadata for transformation

        Returns:
            Tuple of (transformed_image, metadata)
        """
        pass

    @abstractmethod
    def inverse(self, image: torch.Tensor, metadata: MetadataContainer) -> torch.Tensor:
        """
        Apply inverse transformation.

        Args:
            image: Transformed image tensor [B, C, H, W]
            metadata: Metadata from forward transformation

        Returns:
            Reconstructed image tensor
        """
        pass


class GuidanceComputer(ABC):
    """Abstract base class for likelihood guidance computation."""

    @abstractmethod
    def compute_score(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute likelihood score ∇_x log p(y|x).

        Args:
            x_hat: Current estimate [B, C, H, W]
            y_observed: Observed data [B, C, H, W]
            mask: Valid pixel mask [B, C, H, W]
            **kwargs: Additional parameters

        Returns:
            Score gradient tensor
        """
        pass

    @abstractmethod
    def compute(
        self,
        x_hat: torch.Tensor,
        y_observed: torch.Tensor,
        sigma_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute scaled guidance gradient.

        Args:
            x_hat: Current estimate
            y_observed: Observed data
            sigma_t: Current noise level
            mask: Valid pixel mask
            **kwargs: Additional parameters

        Returns:
            Guidance gradient
        """
        pass

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic statistics from recent computations."""
        pass


class CalibrationManager(ABC):
    """Abstract base class for sensor calibration management."""

    @abstractmethod
    def adu_to_electrons(self, adu: np.ndarray) -> np.ndarray:
        """Convert ADU values to electron counts."""
        pass

    @abstractmethod
    def electrons_to_adu(self, electrons: np.ndarray) -> np.ndarray:
        """Convert electron counts to ADU values."""
        pass

    @abstractmethod
    def process_raw(
        self, raw_adu: np.ndarray, return_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Process raw sensor data to calibrated electrons.

        Args:
            raw_adu: Raw ADU data
            return_mask: Whether to return validity mask

        Returns:
            Calibrated electron data, optionally with mask
        """
        pass

    @abstractmethod
    def validate_parameters(self) -> bool:
        """Validate that calibration parameters are physically reasonable."""
        pass


class DomainEncoder(ABC):
    """Abstract base class for domain conditioning."""

    @abstractmethod
    def encode(
        self,
        domain: str,
        scale: float,
        read_noise: float,
        background: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Create conditioning vector for domain adaptation.

        Args:
            domain: Domain name ('photography', 'microscopy', 'astronomy')
            scale: Dataset normalization scale (electrons)
            read_noise: Read noise (electrons)
            background: Background level (electrons)
            **kwargs: Additional domain-specific parameters

        Returns:
            Conditioning vector tensor
        """
        pass

    @abstractmethod
    def get_condition_dim(self) -> int:
        """Get the dimensionality of conditioning vectors."""
        pass


class ModelWrapper(ABC):
    """Abstract base class for diffusion model wrappers."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor [B, C, H, W]
            sigma: Noise level tensor [B] or scalar
            condition: Conditioning vector [B, condition_dim]
            **kwargs: Additional model parameters

        Returns:
            Model output (v-parameterization)
        """
        pass

    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration dictionary."""
        pass


class Sampler(ABC):
    """Abstract base class for diffusion samplers."""

    @abstractmethod
    def sample(
        self,
        y_observed: torch.Tensor,
        metadata: MetadataContainer,
        guidance_computer: GuidanceComputer,
        model: ModelWrapper,
        condition: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Sample from posterior distribution.

        Args:
            y_observed: Observed noisy data
            metadata: Transformation metadata
            guidance_computer: Likelihood guidance
            model: Diffusion model
            condition: Conditioning vector
            **kwargs: Sampling parameters

        Returns:
            Tuple of (denoised_sample, info_dict)
        """
        pass


class Dataset(ABC):
    """Abstract base class for domain datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Get dataset length."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item.

        Returns:
            Dictionary with keys:
            - 'clean': Clean image (if available)
            - 'noisy_electrons': Noisy observations (electrons)
            - 'normalized': Normalized for model [0,1]
            - 'mask': Valid pixel mask
            - 'metadata': Complete reconstruction metadata
            - 'condition': Model conditioning vector
        """
        pass

    @abstractmethod
    def get_domain_info(self) -> Dict[str, Any]:
        """Get domain-specific information."""
        pass


class Evaluator(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def compute_metrics(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predicted: Predicted images
            target: Ground truth images
            mask: Valid pixel mask

        Returns:
            Dictionary of metric values
        """
        pass

    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Get list of available metric names."""
        pass


# Configuration dataclasses
@dataclass
class BaseConfig:
    """Base configuration class with common functionality."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    def to_yaml(self) -> str:
        """Convert config to YAML string."""
        import yaml

        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """Create config from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "BaseConfig":
        """Create config from YAML string."""
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
