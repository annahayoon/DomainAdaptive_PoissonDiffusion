"""Abstract base classes and interfaces for the core components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from config.config import BaseConfig

# Import GuidanceComputer from guidance module
from core.guidance import GuidanceComputer


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
        """Apply forward transformation."""
        pass

    @abstractmethod
    def inverse(self, image: torch.Tensor, metadata: MetadataContainer) -> torch.Tensor:
        """Apply inverse transformation."""
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
        """Process raw sensor data to calibrated electrons."""
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
        """Create conditioning vector for domain adaptation."""
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
        """Forward pass through the model."""
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
        """Sample from posterior distribution."""
        pass


class Dataset(ABC):
    """Abstract base class for domain datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Get dataset length."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        pass

    @abstractmethod
    def get_sensor_info(self) -> Dict[str, Any]:
        """Get sensor-specific information."""
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
        """Compute evaluation metrics."""
        pass

    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Get list of available metric names."""
        pass
