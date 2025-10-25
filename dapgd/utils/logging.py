"""
Logging utilities for DAPGD

PURPOSE: Structured logging for experiments
- Console output for development
- File logs for reproducibility
- TensorBoard/WandB for visualization

Adapted from core/logging_config.py with experiment tracking features
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


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


class ExperimentLogger:
    """
    Unified logger for experiments

    PURPOSE: One interface for console, file, and experiment tracking

    Example:
        logger = ExperimentLogger("my_experiment")
        logger.info("Starting sampling...")
        logger.log_metric("psnr", 28.5, step=10)
        logger.log_image("result", image_tensor, step=10)
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "experiments/runs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        config: Optional[dict] = None,
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup console and file logging
        self.logger = self._setup_logger()

        # Setup experiment tracking
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(self.log_dir / "tensorboard")
            except ImportError:
                self.logger.warning(
                    "TensorBoard not available. Install with: pip install tensorboard"
                )

        self.wandb = None
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb
                wandb.init(
                    project="dapgd",
                    name=experiment_name,
                    config=config,
                    dir=self.log_dir,
                )
            except ImportError:
                self.logger.warning(
                    "WandB not available. Install with: pip install wandb"
                )

    def _setup_logger(self) -> logging.Logger:
        """Setup Python logger with console and file handlers"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()  # Clear existing handlers

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(self.log_dir / "experiment.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Error file handler
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)

        return logger

    # Delegate standard logging methods
    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def log_metric(self, name: str, value: float, step: int = 0):
        """Log a scalar metric"""
        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, step)
        if self.wandb:
            self.wandb.log({name: value, "step": step})

    def log_image(self, name: str, image: torch.Tensor, step: int = 0):
        """Log an image (expects [C,H,W] or [B,C,H,W])"""
        if image.dim() == 4:
            image = image[0]  # Take first in batch

        if self.tb_writer:
            self.tb_writer.add_image(name, image, step)

        if self.wandb:
            import wandb

            # Convert to numpy for wandb
            img_np = image.cpu().numpy()
            if img_np.shape[0] in [1, 3]:  # CHW format
                img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # Grayscale
                img_np = img_np.squeeze(2)
            self.wandb.log({name: wandb.Image(img_np), "step": step})

    def log_config(self, config: dict):
        """Log experiment configuration"""
        self.info(f"Configuration: {config}")

        # Save config to file
        config_path = self.log_dir / "config.yaml"
        try:
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            # Fall back to JSON if YAML not available
            config_path = self.log_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        if self.tb_writer:
            # TensorBoard doesn't have native config logging, save as text
            config_str = "\n".join(f"{k}: {v}" for k, v in config.items())
            self.tb_writer.add_text("config", config_str, 0)

    def close(self):
        """Close all handlers"""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb:
            self.wandb.finish()


# Convenience function
def get_logger(
    experiment_name: str,
    log_dir: str = "experiments/runs",
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    config: Optional[dict] = None,
) -> ExperimentLogger:
    """
    Factory function for creating loggers

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
        use_tensorboard: Enable TensorBoard logging
        use_wandb: Enable Weights & Biases logging
        config: Configuration dictionary to log

    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        config=config,
    )


class temporary_log_level:
    """Context manager for temporarily changing log level."""

    def __init__(self, level: str, logger: Optional[logging.Logger] = None):
        self.level = getattr(logging, level.upper())
        self.logger = logger or logging.getLogger()
        self.original_level = self.logger.level

    def __enter__(self):
        self.logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
