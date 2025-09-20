"""
Test the error handling framework.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from core.error_handlers import (
    DiagnosticCollector,
    ErrorHandler,
    NumericalStabilityManager,
    error_context,
    safe_operation,
)
from core.exceptions import (
    CalibrationError,
    NumericalStabilityError,
    PoissonDiffusionError,
)
from core.logging_config import (
    LoggingManager,
    get_logger,
    setup_project_logging,
    temporary_log_level,
)


class TestErrorHandler:
    """Test the ErrorHandler class."""

    def test_error_handler_initialization(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler()
        assert handler.enable_recovery is True
        assert handler.strict_mode is False
        assert len(handler.error_counts) == 0

        # Test strict mode
        strict_handler = ErrorHandler(strict_mode=True)
        assert strict_handler.enable_recovery is False
        assert strict_handler.strict_mode is True

    def test_error_handling_without_recovery(self):
        """Test error handling without recovery function."""
        handler = ErrorHandler()

        test_error = ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            handler.handle_error(test_error, context="test_context")

        # Check that error was counted
        assert handler.error_counts["ValueError"] == 1

    def test_error_handling_with_recovery(self):
        """Test error handling with successful recovery."""
        handler = ErrorHandler()

        def recovery_func():
            return "recovered_value"

        test_error = ValueError("Test error")

        result = handler.handle_error(
            test_error, context="test_context", recovery_func=recovery_func
        )

        assert result == "recovered_value"
        assert handler.error_counts["ValueError"] == 1
        assert handler.recovery_counts["ValueError"] == 1

    def test_error_handling_with_failed_recovery(self):
        """Test error handling with failed recovery."""
        handler = ErrorHandler()

        def failing_recovery_func():
            raise RuntimeError("Recovery failed")

        test_error = ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            handler.handle_error(
                test_error, context="test_context", recovery_func=failing_recovery_func
            )

        assert handler.error_counts["ValueError"] == 1
        assert handler.recovery_counts.get("ValueError", 0) == 0

    def test_statistics(self):
        """Test error statistics collection."""
        handler = ErrorHandler()

        # Simulate some errors
        for i in range(3):
            try:
                handler.handle_error(ValueError(f"Error {i}"), "test")
            except ValueError:
                pass

        # Simulate recovery
        def recovery():
            return "ok"

        handler.handle_error(ValueError("Recoverable"), "test", recovery)

        stats = handler.get_statistics()
        assert stats["error_counts"]["ValueError"] == 4
        assert stats["recovery_counts"]["ValueError"] == 1
        assert stats["recovery_rate"]["ValueError"] == 0.25


class TestNumericalStabilityManager:
    """Test the NumericalStabilityManager class."""

    def test_initialization(self):
        """Test NumericalStabilityManager initialization."""
        manager = NumericalStabilityManager()
        assert manager.eps_variance == 0.1
        assert manager.grad_clip == 10.0
        assert manager.adaptive is True

    def test_tensor_validation_and_fixing(self):
        """Test tensor validation and automatic fixing."""
        manager = NumericalStabilityManager()

        # Create problematic tensor
        problematic = torch.tensor(
            [1.0, float("nan"), float("inf"), -float("inf"), -1.0, 100.0]
        )

        # Test with fixing enabled
        fixed = manager.check_and_fix_tensor(
            problematic, name="test_tensor", fix_issues=True
        )

        # Should have no NaN or Inf values
        assert not torch.isnan(fixed).any()
        assert not torch.isinf(fixed).any()

        # Values should be in valid range
        assert (fixed >= manager.range_min).all()
        if manager.range_max is not None:
            assert (fixed <= manager.range_max).all()

    def test_tensor_validation_without_fixing(self):
        """Test tensor validation without fixing (should raise errors)."""
        manager = NumericalStabilityManager()

        # NaN tensor should raise error
        nan_tensor = torch.tensor([1.0, float("nan")])
        with pytest.raises(NumericalStabilityError, match="NaN"):
            manager.check_and_fix_tensor(nan_tensor, fix_issues=False)

        # Inf tensor should raise error
        inf_tensor = torch.tensor([1.0, float("inf")])
        with pytest.raises(NumericalStabilityError, match="Inf"):
            manager.check_and_fix_tensor(inf_tensor, fix_issues=False)

        # Range violation should raise error
        range_tensor = torch.tensor([-1.0, 0.5])
        with pytest.raises(NumericalStabilityError, match="below minimum"):
            manager.check_and_fix_tensor(range_tensor, fix_issues=False)

    def test_variance_stabilization(self):
        """Test variance stabilization."""
        manager = NumericalStabilityManager(eps_variance=0.1)

        variance = torch.tensor([0.0, -0.05, 0.05, 1.0, 2.0])
        stabilized = manager.stabilize_variance(variance)

        # All values should be >= eps_variance
        assert (stabilized >= manager.eps_variance).all()

        # Values above threshold should be unchanged
        assert stabilized[3] == 1.0
        assert stabilized[4] == 2.0

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        manager = NumericalStabilityManager(grad_clip=5.0)

        gradients = torch.tensor([-10.0, -3.0, 0.0, 3.0, 10.0])
        clipped = manager.clip_gradients(gradients)

        # Values should be clipped to [-5, 5]
        expected = torch.tensor([-5.0, -3.0, 0.0, 3.0, 5.0])
        assert torch.allclose(clipped, expected)

    def test_adaptive_thresholds(self):
        """Test adaptive threshold adjustment."""
        manager = NumericalStabilityManager(adaptive=True, eps_variance=1.0)

        # Simulate many variance values below current threshold
        for _ in range(150):
            variance = torch.tensor([0.1, 0.2, 0.3])  # All below 1.0
            manager.stabilize_variance(variance)

        # Threshold should have adapted downward
        assert manager.eps_variance < 1.0
        assert manager.adaptation_count > 0


class TestSafeOperation:
    """Test the safe_operation decorator."""

    def test_successful_operation(self):
        """Test decorator with successful operation."""

        @safe_operation("test_op")
        def successful_func(x):
            return x * 2

        result = successful_func(5)
        assert result == 10

    def test_operation_with_error_no_recovery(self):
        """Test decorator with error and no recovery."""

        @safe_operation("test_op")
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()

    def test_operation_with_error_and_recovery(self):
        """Test decorator with error and recovery."""

        def recovery_func():
            return "recovered"

        @safe_operation("test_op", recovery_func=recovery_func)
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert result == "recovered"


class TestErrorContext:
    """Test the error_context context manager."""

    def test_successful_context(self):
        """Test context manager with successful operation."""
        with error_context("test_context"):
            result = 1 + 1

        assert result == 2

    def test_context_with_error(self):
        """Test context manager with error."""
        with pytest.raises(ValueError, match="Test error"):
            with error_context("test_context"):
                raise ValueError("Test error")

    def test_context_with_suppressed_error(self):
        """Test context manager with suppressed error."""
        with error_context("test_context", suppress_errors=True):
            raise ValueError("Test error")

        # Should not raise


class TestDiagnosticCollector:
    """Test the DiagnosticCollector class."""

    def test_diagnostic_collection(self):
        """Test diagnostic data collection."""
        collector = DiagnosticCollector()

        # Record some data
        collector.record("errors", "error1")
        collector.record("errors", "error2")
        collector.record("metrics", 1.5)
        collector.record("metrics", 2.0)
        collector.record("metrics", 1.8)

        summary = collector.get_summary()

        assert summary["errors"]["count"] == 2
        assert summary["errors"]["latest"] == "error2"

        assert summary["metrics"]["count"] == 3
        assert summary["metrics"]["mean"] == pytest.approx(1.77, abs=0.01)
        assert summary["metrics"]["min"] == 1.5
        assert summary["metrics"]["max"] == 2.0

    def test_diagnostic_clearing(self):
        """Test clearing diagnostic data."""
        collector = DiagnosticCollector()

        collector.record("test", "data")
        assert len(collector.diagnostics["test"]) == 1

        # Clear specific category
        collector.clear("test")
        assert "test" not in collector.diagnostics

        # Clear all
        collector.record("test1", "data1")
        collector.record("test2", "data2")
        collector.clear()
        assert len(collector.diagnostics) == 0


class TestLoggingManager:
    """Test the LoggingManager class."""

    def test_logging_setup(self):
        """Test logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoggingManager()

            logger = manager.setup_logging(
                level="DEBUG", log_dir=temp_dir, console_output=True, file_output=True
            )

            assert logger.name == "poisson_diffusion"
            # Logger inherits from root, check effective level
            assert logger.getEffectiveLevel() == logging.DEBUG
            assert manager.configured is True

            # Check that log files are created
            log_dir = Path(temp_dir)
            assert (log_dir / "poisson_diffusion.log").exists()
            assert (log_dir / "errors.log").exists()
            assert (log_dir / "debug.log").exists()

    def test_logger_retrieval(self):
        """Test logger retrieval."""
        manager = LoggingManager()

        logger1 = manager.get_logger("test_logger")
        logger2 = manager.get_logger("test_logger")

        # Should return the same logger instance
        assert logger1 is logger2
        assert logger1.name == "test_logger"

    def test_level_setting(self):
        """Test setting log levels."""
        manager = LoggingManager()
        manager.setup_logging(level="INFO")

        test_logger = manager.get_logger("test")
        original_level = test_logger.level

        # Set specific logger level
        manager.set_level("DEBUG", "test")
        assert test_logger.level == logging.DEBUG

        # Set all loggers level
        manager.set_level("ERROR")
        assert test_logger.level == logging.ERROR


class TestTemporaryLogLevel:
    """Test the temporary_log_level context manager."""

    def test_temporary_level_change(self):
        """Test temporary log level change."""
        # Setup logging
        logger = setup_project_logging(level="INFO")
        original_level = logger.level

        # Test temporary change
        with temporary_log_level("DEBUG"):
            assert logger.level == logging.DEBUG

        # Should be restored
        assert logger.level == original_level


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
