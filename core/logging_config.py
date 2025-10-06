"""
Logging configuration for Poisson-Gaussian Diffusion project.

This module provides comprehensive logging setup with different levels,
formatters, and handlers for development and production environments.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry)


class LoggingManager:
    """
    Centralized logging management for the project.
    """

    def __init__(self):
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.configured = False

    def setup_logging(
        self,
        level: Union[str, int] = "INFO",
        log_dir: Optional[Union[str, Path]] = None,
        console_output: bool = True,
        file_output: bool = True,
        json_format: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        logger_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> logging.Logger:
        """
        Set up comprehensive logging configuration.

        Args:
            level: Logging level
            log_dir: Directory for log files
            console_output: Whether to output to console
            file_output: Whether to output to files
            json_format: Whether to use JSON formatting
            max_file_size: Maximum size for log files before rotation
            backup_count: Number of backup files to keep
            logger_configs: Specific configurations for different loggers

        Returns:
            Main project logger
        """
        # Convert string level to int
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        # Create log directory if needed
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set root level
        root_logger.setLevel(level)

        # Create formatters
        if json_format:
            formatter = JSONFormatter()
            console_formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
            self.handlers["console"] = console_handler

        # File handlers
        if file_output and log_dir is not None:
            # Main log file
            main_handler = logging.handlers.RotatingFileHandler(
                log_dir / "poisson_diffusion.log",
                maxBytes=max_file_size,
                backupCount=backup_count,
            )
            main_handler.setLevel(level)
            main_handler.setFormatter(formatter)
            root_logger.addHandler(main_handler)
            self.handlers["main_file"] = main_handler

            # Error log file
            error_handler = logging.handlers.RotatingFileHandler(
                log_dir / "errors.log", maxBytes=max_file_size, backupCount=backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)
            self.handlers["error_file"] = error_handler

            # Debug log file (if debug level)
            if level <= logging.DEBUG:
                debug_handler = logging.handlers.RotatingFileHandler(
                    log_dir / "debug.log",
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                )
                debug_handler.setLevel(logging.DEBUG)
                debug_handler.setFormatter(formatter)
                root_logger.addHandler(debug_handler)
                self.handlers["debug_file"] = debug_handler

        # Configure specific loggers
        if logger_configs:
            for logger_name, config in logger_configs.items():
                logger = logging.getLogger(logger_name)
                logger.setLevel(config.get("level", level))
                self.loggers[logger_name] = logger

        # Create main project logger
        main_logger = logging.getLogger("poisson_diffusion")
        self.loggers["main"] = main_logger

        self.configured = True
        main_logger.info("Logging system initialized")

        return main_logger

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name."""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]

    def set_level(self, level: Union[str, int], logger_name: Optional[str] = None):
        """Set logging level for specific logger or all loggers."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        if logger_name is not None:
            if logger_name in self.loggers:
                self.loggers[logger_name].setLevel(level)
        else:
            # Set for all loggers
            for logger in self.loggers.values():
                logger.setLevel(level)

            # Set for all handlers
            for handler in self.handlers.values():
                handler.setLevel(level)

    def add_file_handler(
        self,
        filename: Union[str, Path],
        level: Union[str, int] = "INFO",
        formatter: Optional[logging.Formatter] = None,
        max_size: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> logging.Handler:
        """Add a new file handler."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        handler = logging.handlers.RotatingFileHandler(
            filename, maxBytes=max_size, backupCount=backup_count
        )
        handler.setLevel(level)

        if formatter is None:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        handler.setFormatter(formatter)

        # Add to root logger
        logging.getLogger().addHandler(handler)

        # Store reference
        handler_name = Path(filename).stem
        self.handlers[handler_name] = handler

        return handler

    def shutdown(self):
        """Shutdown logging system."""
        for handler in self.handlers.values():
            handler.close()

        logging.shutdown()
        self.configured = False


logging_manager = LoggingManager()


def setup_project_logging(
    level: str = "INFO",
    log_dir: Optional[str] = "logs",
    console: bool = True,
    files: bool = True,
    json_format: bool = False,
) -> logging.Logger:
    """
    Convenience function to set up project logging.

    Args:
        level: Logging level
        log_dir: Directory for log files
        console: Enable console output
        files: Enable file output
        json_format: Use JSON formatting

    Returns:
        Main project logger
    """
    return logging_manager.setup_logging(
        level=level,
        log_dir=log_dir,
        console_output=console,
        file_output=files,
        json_format=json_format,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging_manager.get_logger(name)


class temporary_log_level:
    """Context manager for temporarily changing log level."""

    def __init__(self, level: Union[str, int], logger_name: Optional[str] = None):
        self.level = level
        self.logger_name = logger_name
        self.original_levels: Dict[str, int] = {}

    def __enter__(self):
        if isinstance(self.level, str):
            self.level = getattr(logging, self.level.upper())

        if self.logger_name is not None:
            logger = logging_manager.get_logger(self.logger_name)
            self.original_levels[self.logger_name] = logger.level
            logger.setLevel(self.level)
        else:
            # Store original levels for all loggers
            for name, logger in logging_manager.loggers.items():
                self.original_levels[name] = logger.level
                logger.setLevel(self.level)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original levels
        for name, original_level in self.original_levels.items():
            if name in logging_manager.loggers:
                logging_manager.loggers[name].setLevel(original_level)
