#!/usr/bin/env python3
"""
Demonstration of the error handling framework.

This script shows how to use the various error handling components
in realistic scenarios.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from core import (
    DiagnosticCollector,
    ErrorHandler,
    NumericalStabilityManager,
    error_context,
    get_logger,
    safe_operation,
    setup_project_logging,
    temporary_log_level,
)
from core.exceptions import CalibrationError, NumericalStabilityError


def demo_basic_error_handling():
    """Demonstrate basic error handling with recovery."""
    print("\n" + "=" * 60)
    print("DEMO: Basic Error Handling with Recovery")
    print("=" * 60)

    # Setup error handler
    logger = get_logger("demo")
    handler = ErrorHandler(logger=logger, enable_recovery=True)

    # Define a recovery function
    def recovery_function():
        logger.info("Recovery: Using default calibration values")
        return {"gain": 1.0, "read_noise": 5.0, "black_level": 100}

    # Simulate a calibration error
    def load_calibration():
        raise CalibrationError("Calibration file not found")

    try:
        # This will fail but recovery will provide defaults
        calibration = handler.handle_error(
            CalibrationError("Calibration file not found"),
            context="load_calibration",
            recovery_func=recovery_function,
        )
        print(f"✓ Calibration loaded (via recovery): {calibration}")

    except Exception as e:
        print(f"✗ Failed to load calibration: {e}")

    # Show statistics
    stats = handler.get_statistics()
    print(f"Error statistics: {stats}")


def demo_numerical_stability():
    """Demonstrate numerical stability management."""
    print("\n" + "=" * 60)
    print("DEMO: Numerical Stability Management")
    print("=" * 60)

    # Create stability manager
    stability = NumericalStabilityManager(
        eps_variance=0.1, grad_clip=5.0, adaptive=True
    )

    # Create problematic tensors
    print("Creating problematic tensors...")

    # Tensor with NaN and Inf values
    problematic = torch.tensor(
        [1.0, float("nan"), float("inf"), -float("inf"), -1.0, 10.0]
    )
    print(f"Original tensor: {problematic}")

    # Fix the tensor
    fixed = stability.check_and_fix_tensor(
        problematic, name="demo_tensor", fix_issues=True
    )
    print(f"Fixed tensor: {fixed}")

    # Demonstrate variance stabilization
    print("\nVariance stabilization:")
    variance = torch.tensor([0.0, -0.1, 0.05, 1.0, 2.0])
    print(f"Original variance: {variance}")

    stabilized_var = stability.stabilize_variance(variance, "demo_variance")
    print(f"Stabilized variance: {stabilized_var}")

    # Demonstrate gradient clipping
    print("\nGradient clipping:")
    gradients = torch.tensor([-10.0, -3.0, 0.0, 3.0, 10.0])
    print(f"Original gradients: {gradients}")

    clipped_grads = stability.clip_gradients(gradients, "demo_gradients")
    print(f"Clipped gradients: {clipped_grads}")


def demo_safe_operations():
    """Demonstrate safe operation decorators."""
    print("\n" + "=" * 60)
    print("DEMO: Safe Operation Decorators")
    print("=" * 60)

    # Define recovery function
    def default_processing(*args, **kwargs):
        return torch.ones(2, 2) * 0.5

    @safe_operation(
        operation_name="image_processing",
        recovery_func=default_processing,
        error_types=(RuntimeError, ValueError),
    )
    def process_image(image_data):
        if image_data is None:
            raise ValueError("No image data provided")

        # Simulate processing
        return image_data * 2

    # Test successful operation
    print("Testing successful operation:")
    result = process_image(torch.ones(2, 2))
    print(f"✓ Processing result: {result}")

    # Test operation with error and recovery
    print("\nTesting operation with error (will recover):")
    result = process_image(None)  # This will trigger recovery
    print(f"✓ Recovery result: {result}")


def demo_error_context():
    """Demonstrate error context managers."""
    print("\n" + "=" * 60)
    print("DEMO: Error Context Managers")
    print("=" * 60)

    logger = get_logger("context_demo")

    # Successful context
    print("Testing successful context:")
    with error_context("data_loading", logger=logger):
        data = torch.randn(10, 10)
        print(f"✓ Data loaded successfully: shape {data.shape}")

    # Context with suppressed error
    print("\nTesting context with suppressed error:")
    with error_context("optional_operation", logger=logger, suppress_errors=True):
        raise RuntimeError("This error will be suppressed")

    print("✓ Continued execution after suppressed error")


def demo_diagnostics():
    """Demonstrate diagnostic collection."""
    print("\n" + "=" * 60)
    print("DEMO: Diagnostic Collection")
    print("=" * 60)

    collector = DiagnosticCollector()

    # Simulate collecting various diagnostics
    print("Collecting diagnostics...")

    # Collect error information
    collector.record("errors", "NaN detected in tensor")
    collector.record("errors", "Gradient explosion")
    collector.record("errors", "Memory allocation failed")

    # Collect performance metrics
    for i in range(10):
        collector.record("processing_time", np.random.uniform(0.1, 2.0))
        collector.record("memory_usage", np.random.uniform(100, 1000))

    # Collect model metrics
    collector.record("loss", 0.5)
    collector.record("loss", 0.3)
    collector.record("loss", 0.2)

    # Get summary
    summary = collector.get_summary()
    print("Diagnostic Summary:")
    for category, stats in summary.items():
        print(f"  {category}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.3f}")
            else:
                print(f"    {key}: {value}")


def demo_adaptive_thresholds():
    """Demonstrate adaptive threshold adjustment."""
    print("\n" + "=" * 60)
    print("DEMO: Adaptive Threshold Adjustment")
    print("=" * 60)

    stability = NumericalStabilityManager(
        eps_variance=1.0, adaptive=True  # Start with high threshold
    )

    print(f"Initial variance threshold: {stability.eps_variance}")

    # Simulate many operations with lower variance values
    print("Simulating 150 operations with low variance values...")
    for i in range(150):
        # Generate variance values mostly below current threshold
        variance = torch.tensor([0.1, 0.2, 0.3, 0.4])
        stability.stabilize_variance(variance)

        if i % 50 == 49:  # Print progress
            print(f"  After {i+1} operations: threshold = {stability.eps_variance:.6f}")

    print(f"Final variance threshold: {stability.eps_variance:.6f}")
    print(f"Number of adaptations: {stability.adaptation_count}")


def demo_logging_levels():
    """Demonstrate logging level management."""
    print("\n" + "=" * 60)
    print("DEMO: Logging Level Management")
    print("=" * 60)

    logger = get_logger("level_demo")

    print("Testing different log levels:")

    # Test with INFO level
    print("\nWith INFO level:")
    with temporary_log_level("INFO"):
        logger.debug("This debug message won't appear")
        logger.info("This info message will appear")
        logger.warning("This warning will appear")

    # Test with DEBUG level
    print("\nWith DEBUG level:")
    with temporary_log_level("DEBUG"):
        logger.debug("This debug message will appear")
        logger.info("This info message will appear")
        logger.warning("This warning will appear")


def main():
    """Run all demonstrations."""
    print("Poisson-Gaussian Diffusion Error Handling Framework Demo")
    print("=" * 60)

    # Setup logging
    logger = setup_project_logging(
        level="INFO", console=True, files=False  # Don't create files for demo
    )

    # Run demonstrations
    try:
        demo_basic_error_handling()
        demo_numerical_stability()
        demo_safe_operations()
        demo_error_context()
        demo_diagnostics()
        demo_adaptive_thresholds()
        demo_logging_levels()

        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("The error handling framework is ready for use.")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
