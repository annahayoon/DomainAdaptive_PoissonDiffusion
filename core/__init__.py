"""Core module for Sensor-Adaptive Poisson-Gaussian Diffusion."""

from config.config import BaseConfig, SamplingConfig
from config.logging_config import (
    LoggingManager,
    get_logger,
    setup_project_logging,
    temporary_log_level,
)

from .error_handlers import (
    DiagnosticCollector,
    ErrorHandler,
    NumericalStabilityManager,
    error_context,
    safe_operation,
)
from .interfaces import (
    CalibrationManager,
    Dataset,
    Evaluator,
    GuidanceComputer,
    MetadataContainer,
    ModelWrapper,
    Sampler,
    Transform,
)
from .transforms import ImageMetadata, ReversibleTransform

# Import models (previously in models module)
try:
    from .edm_wrapper import EDMModelWrapper, FiLMLayer
    from .sampler import (
        EDMSampler,
        create_edm_sampler,
        create_fast_native_sampler,
        create_fast_sampler,
        create_high_quality_native_sampler,
        create_high_quality_sampler,
        create_native_sampler,
        sample_batch,
    )

    _MODELS_AVAILABLE = True
except ImportError:
    EDMModelWrapper = None
    FiLMLayer = None
    EDMSampler = None
    create_edm_sampler = None
    create_fast_native_sampler = None
    create_fast_sampler = None
    create_high_quality_native_sampler = None
    create_high_quality_sampler = None
    create_native_sampler = None
    sample_batch = None
    _MODELS_AVAILABLE = False

try:
    from .guidance import (
        GaussianGuidance,
        GuidanceComputer,
        PoissonGaussianGuidance,
        select_guidance,
        validate_guidance_inputs,
    )

    _GUIDANCE_AVAILABLE = True
except ImportError:
    GaussianGuidance = None
    GuidanceComputer = None
    PoissonGaussianGuidance = None
    select_guidance = None
    validate_guidance_inputs = None
    _GUIDANCE_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    "Transform",
    "GuidanceComputer",
    "CalibrationManager",
    "MetadataContainer",
    "ModelWrapper",
    "Sampler",
    "Dataset",
    "Evaluator",
    "BaseConfig",
    "SamplingConfig",
    "ErrorHandler",
    "NumericalStabilityManager",
    "safe_operation",
    "error_context",
    "DiagnosticCollector",
    "LoggingManager",
    "setup_project_logging",
    "get_logger",
    "temporary_log_level",
    "ReversibleTransform",
    "ImageMetadata",
]

if _GUIDANCE_AVAILABLE:
    __all__.extend(
        [
            "GaussianGuidance",
            "GuidanceComputer",
            "PoissonGaussianGuidance",
            "select_guidance",
            "validate_guidance_inputs",
        ]
    )

if _MODELS_AVAILABLE:
    __all__.extend(
        [
            "EDMModelWrapper",
            "FiLMLayer",
            "EDMSampler",
            "create_edm_sampler",
            "create_fast_sampler",
            "create_high_quality_sampler",
            "create_native_sampler",
            "create_fast_native_sampler",
            "create_high_quality_native_sampler",
            "sample_batch",
        ]
    )
