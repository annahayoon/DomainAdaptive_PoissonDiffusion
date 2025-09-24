"""
Custom exceptions for the Poisson-Gaussian Diffusion project.

This module defines domain-specific exceptions that provide clear
error messages and enable proper error handling throughout the system.
"""


class PoissonDiffusionError(Exception):
    """Base exception for all project-specific errors."""

    pass


class CalibrationError(PoissonDiffusionError):
    """Raised when sensor calibration is invalid or missing."""

    pass


class TransformError(PoissonDiffusionError):
    """Raised when image transformation fails."""

    pass


class MetadataError(PoissonDiffusionError):
    """Raised when metadata is invalid or corrupted."""

    pass


class GuidanceError(PoissonDiffusionError):
    """Raised when likelihood guidance computation fails."""

    pass


class ModelError(PoissonDiffusionError):
    """Raised when model operations fail."""

    pass


class DataError(PoissonDiffusionError):
    """Raised when data loading or processing fails."""

    pass


class NumericalStabilityError(PoissonDiffusionError):
    """Raised when numerical instability is detected."""

    pass


class DomainError(PoissonDiffusionError):
    """Raised when domain-specific operations fail."""

    pass


class ConfigurationError(PoissonDiffusionError):
    """Raised when configuration is invalid."""

    pass


# Specific error types for common scenarios
class InvalidPhotonCountError(GuidanceError):
    """Raised when photon counts are invalid (negative, NaN, etc.)."""

    pass


class UnsupportedDomainError(DomainError):
    """Raised when an unsupported domain is specified."""

    pass


class IncompatibleMetadataError(MetadataError):
    """Raised when metadata is incompatible with current operation."""

    pass


class ReconstructionError(TransformError):
    """Raised when image reconstruction fails."""

    pass


class CalibrationParameterError(CalibrationError):
    """Raised when calibration parameters are physically unreasonable."""

    pass


class ValidationError(PoissonDiffusionError):
    """Raised when validation checks fail."""

    pass


class SamplingError(PoissonDiffusionError):
    """Raised when sampling operations fail."""

    pass


class TrainingError(PoissonDiffusionError):
    """Raised when training operations fail."""

    pass


class PerformanceError(PoissonDiffusionError):
    """Raised when performance operations fail."""

    pass


class AnalysisError(PoissonDiffusionError):
    """Raised when analysis operations fail."""

    pass
