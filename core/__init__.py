"""
Core module for Domain-Adaptive Poisson-Gaussian Diffusion.

This module contains the fundamental components:
- Reversible transforms with metadata preservation
- Physics-aware Poisson-Gaussian guidance
- Base interfaces and abstract classes
- Error handling and logging utilities
- Numerical stability management
"""

# Calibration imports (optional - only needed for inference)
try:
    from .calibration import (
        CalibrationParams,
        SensorCalibration,
        create_calibration_from_params,
        load_calibration,
    )

    _CALIBRATION_AVAILABLE = True
except ImportError:
    # Calibration module not available - not needed for training
    CalibrationParams = None
    SensorCalibration = None
    create_calibration_from_params = None
    load_calibration = None
    _CALIBRATION_AVAILABLE = False
from .error_handlers import (
    DiagnosticCollector,
    ErrorHandler,
    NumericalStabilityManager,
    error_context,
    safe_operation,
)
from .interfaces import (
    BaseConfig,
    CalibrationManager,
    Dataset,
    DomainEncoder,
    Evaluator,
    GuidanceComputer,
    MetadataContainer,
    ModelWrapper,
    Sampler,
    Transform,
)
from .logging_config import (
    LoggingManager,
    get_logger,
    setup_project_logging,
    temporary_log_level,
)

# Import implementations when they exist
from .transforms import ImageMetadata, ReversibleTransform

# Guidance imports (optional - only needed for inference)
try:
    from .guidance_config import GuidanceConfig, GuidancePresets
    from .guidance_factory import create_guidance, create_guidance_from_config
    from .l2_guidance import L2Guidance
    from .poisson_guidance import (
        PoissonGuidance,
        create_domain_guidance,
        create_poisson_guidance,
    )

    _GUIDANCE_AVAILABLE = True
except ImportError:
    # Guidance modules not available - not needed for training
    GuidanceConfig = None
    GuidancePresets = None
    create_guidance = None
    create_guidance_from_config = None
    L2Guidance = None
    PoissonGuidance = None
    create_poisson_guidance = None
    create_domain_guidance = None
    _GUIDANCE_AVAILABLE = False


__version__ = "0.1.0"

__all__ = [
    # Interfaces
    "Transform",
    "GuidanceComputer",
    "CalibrationManager",
    "MetadataContainer",
    "DomainEncoder",
    "ModelWrapper",
    "Sampler",
    "Dataset",
    "Evaluator",
    "BaseConfig",
    # Error handling
    "ErrorHandler",
    "NumericalStabilityManager",
    "safe_operation",
    "error_context",
    "DiagnosticCollector",
    # Logging
    "LoggingManager",
    "setup_project_logging",
    "get_logger",
    "temporary_log_level",
    # Implementations
    "ReversibleTransform",
    "ImageMetadata",
]

# Add calibration exports only if available
if _CALIBRATION_AVAILABLE:
    __all__.extend(
        [
            "SensorCalibration",
            "CalibrationParams",
            "load_calibration",
            "create_calibration_from_params",
        ]
    )

# Add guidance exports only if available
if _GUIDANCE_AVAILABLE:
    __all__.extend(
        [
            "PoissonGuidance",
            "L2Guidance",
            "GuidanceConfig",
            "GuidancePresets",
            "create_guidance",
            "create_guidance_from_config",
            "create_poisson_guidance",
            "create_domain_guidance",
        ]
    )
