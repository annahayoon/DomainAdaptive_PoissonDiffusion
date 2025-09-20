"""
Performance optimization utilities for Poisson-Gaussian diffusion restoration.

This module provides comprehensive performance optimizations including:
- Mixed precision training and inference
- Memory management and optimization
- Batch processing optimization
- GPU utilization optimization
- Tiled processing for large images
- Performance profiling and benchmarking

Requirements addressed: 7.1-7.6 from requirements.md
Task: 6.1 from tasks.md
"""

import gc
import logging
import math
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from core.error_handlers import ErrorHandler, safe_operation
from core.exceptions import PoissonDiffusionError
from core.logging_config import get_logger

logger = get_logger(__name__)


class PerformanceError(PoissonDiffusionError):
    """Raised when performance operations fail."""

    pass


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""

    # Mixed precision settings
    mixed_precision: bool = True
    mixed_precision_dtype: torch.dtype = torch.float16
    loss_scale: float = 2**16

    # Memory management
    memory_efficient: bool = True
    max_memory_gb: Optional[float] = None  # Auto-detect if None
    memory_fraction: float = 0.8  # Use 80% of available memory
    enable_memory_mapping: bool = True

    # Batch processing
    auto_batch_size: bool = True
    max_batch_size: int = 32
    min_batch_size: int = 1
    batch_size_growth_factor: float = 1.5

    # Tiled processing
    enable_tiling: bool = True
    tile_size: Tuple[int, int] = (512, 512)
    tile_overlap: int = 64
    tile_blend_mode: str = "linear"  # "linear", "cosine", "gaussian"

    # GPU optimization
    enable_cudnn_benchmark: bool = True
    enable_tf32: bool = True
    enable_flash_attention: bool = True

    # Compilation
    enable_torch_compile: bool = False  # Experimental
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"

    # Profiling
    enable_profiling: bool = False
    profile_memory: bool = True
    profile_compute: bool = True


class MemoryManager:
    """Advanced memory management for large-scale processing."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Memory tracking
        self.peak_memory = 0
        self.memory_history = []

        # Initialize memory settings
        self._setup_memory_management()

        logger.info(f"Initialized MemoryManager on {self.device}")
        if self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU memory: {total_memory:.1f} GB total")

    def _setup_memory_management(self):
        """Setup memory management settings."""
        if self.device.type == "cuda":
            # Set memory fraction
            if self.config.max_memory_gb is not None:
                torch.cuda.set_per_process_memory_fraction(
                    self.config.max_memory_gb
                    / (torch.cuda.get_device_properties(0).total_memory / 1024**3)
                )
            else:
                torch.cuda.set_per_process_memory_fraction(self.config.memory_fraction)

            # Enable memory mapping if requested
            if self.config.enable_memory_mapping:
                torch.backends.cuda.matmul.allow_tf32 = self.config.enable_tf32
                torch.backends.cudnn.allow_tf32 = self.config.enable_tf32

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        info = {}

        # System memory
        system_memory = psutil.virtual_memory()
        info.update(
            {
                "system_total_gb": system_memory.total / 1024**3,
                "system_used_gb": system_memory.used / 1024**3,
                "system_available_gb": system_memory.available / 1024**3,
                "system_percent": system_memory.percent,
            }
        )

        # GPU memory
        if self.device.type == "cuda":
            info.update(
                {
                    "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_cached_gb": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_max_allocated_gb": torch.cuda.max_memory_allocated()
                    / 1024**3,
                    "gpu_max_cached_gb": torch.cuda.max_memory_reserved() / 1024**3,
                }
            )

        return info

    def optimize_memory(self):
        """Optimize memory usage."""
        # Clear cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        # Update peak memory
        if self.device.type == "cuda":
            current_memory = torch.cuda.memory_allocated() / 1024**3
            self.peak_memory = max(self.peak_memory, current_memory)
            self.memory_history.append(current_memory)

            # Keep only recent history
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-500:]

    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """Context manager for memory tracking."""
        start_memory = self.get_memory_info()
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.get_memory_info()

            # Log memory usage
            if self.device.type == "cuda":
                memory_used = (
                    end_memory["gpu_allocated_gb"] - start_memory["gpu_allocated_gb"]
                )
                logger.debug(
                    f"{operation_name}: {memory_used:.2f} GB memory, "
                    f"{end_time - start_time:.2f}s"
                )

            # Optimize memory after operation
            self.optimize_memory()


class MixedPrecisionManager:
    """Mixed precision training and inference manager."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize mixed precision components
        self.scaler = None
        self.enabled = config.mixed_precision and self.device.type == "cuda"

        if self.enabled:
            self.scaler = GradScaler(init_scale=config.loss_scale)
            logger.info(f"Mixed precision enabled with {config.mixed_precision_dtype}")
        else:
            logger.info("Mixed precision disabled")

    @contextmanager
    def autocast_context(self):
        """Context manager for mixed precision."""
        if self.enabled:
            with autocast(dtype=self.config.mixed_precision_dtype):
                yield
        else:
            yield

    @property
    def autocast(self):
        """Provide autocast property for backward compatibility."""
        return self.autocast_context

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> bool:
        """Step optimizer with mixed precision."""
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
            return True
        else:
            optimizer.step()
            return True

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients for gradient clipping."""
        if self.enabled and self.scaler is not None:
            self.scaler.unscale_(optimizer)


class BatchSizeOptimizer:
    """Automatic batch size optimization."""

    def __init__(self, config: PerformanceConfig, memory_manager: MemoryManager):
        self.config = config
        self.memory_manager = memory_manager

        # Batch size tracking
        self.current_batch_size = config.min_batch_size
        self.optimal_batch_size = None
        self.batch_size_history = []

        # Performance tracking
        self.throughput_history = []
        self.memory_usage_history = []

        logger.info(
            f"Initialized BatchSizeOptimizer: {config.min_batch_size}-{config.max_batch_size}"
        )

    def find_optimal_batch_size(
        self,
        model: nn.Module,
        sample_batch: Dict[str, torch.Tensor],
        max_trials: int = 10,
    ) -> int:
        """Find optimal batch size through binary search."""
        if not self.config.auto_batch_size:
            return self.config.max_batch_size

        logger.info("Finding optimal batch size...")

        # Start with minimum batch size
        low, high = self.config.min_batch_size, self.config.max_batch_size
        optimal_size = low

        model.eval()

        for trial in range(max_trials):
            test_size = (low + high) // 2

            try:
                # Create test batch
                test_batch = self._create_test_batch(sample_batch, test_size)

                # Test forward pass
                with torch.no_grad():
                    with self.memory_manager.memory_context(
                        f"batch_size_test_{test_size}"
                    ):
                        start_time = time.time()
                        _ = model(test_batch)
                        end_time = time.time()

                # Calculate throughput
                throughput = test_size / (end_time - start_time)

                # Check memory usage
                memory_info = self.memory_manager.get_memory_info()
                memory_usage = memory_info.get("gpu_allocated_gb", 0)

                # If successful, try larger batch size
                optimal_size = test_size
                low = test_size + 1

                logger.debug(
                    f"Batch size {test_size}: {throughput:.1f} samples/s, {memory_usage:.2f} GB"
                )

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower():
                    # Reduce batch size
                    high = test_size - 1
                    logger.debug(f"Batch size {test_size}: OOM")
                else:
                    raise e

            if low > high:
                break

        self.optimal_batch_size = optimal_size
        self.current_batch_size = optimal_size

        logger.info(f"Optimal batch size found: {optimal_size}")
        return optimal_size

    def _create_test_batch(
        self, sample_batch: Dict[str, torch.Tensor], batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Create test batch with specified size."""
        test_batch = {}

        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                # Repeat tensor to match batch size
                if value.dim() == 0:  # Scalar
                    test_batch[key] = value.expand(batch_size)
                else:
                    # Repeat along batch dimension
                    repeats = [1] * value.dim()
                    repeats[0] = batch_size
                    test_batch[key] = value.repeat(*repeats)
            else:
                # Non-tensor data
                test_batch[key] = [value] * batch_size

        return test_batch

    def update_performance(
        self, batch_size: int, throughput: float, memory_usage: float
    ):
        """Update performance tracking."""
        self.batch_size_history.append(batch_size)
        self.throughput_history.append(throughput)
        self.memory_usage_history.append(memory_usage)

        # Keep only recent history
        max_history = 100
        if len(self.batch_size_history) > max_history:
            self.batch_size_history = self.batch_size_history[-max_history:]
            self.throughput_history = self.throughput_history[-max_history:]
            self.memory_usage_history = self.memory_usage_history[-max_history:]


class TiledProcessor:
    """Tiled processing for large images."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.tile_size = config.tile_size
        self.overlap = config.tile_overlap
        self.blend_mode = config.tile_blend_mode

        logger.info(
            f"Initialized TiledProcessor: {self.tile_size} tiles, {self.overlap} overlap"
        )

    def process_tiled(
        self,
        image: torch.Tensor,
        process_fn: Callable[[torch.Tensor], torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """Process large image using tiled approach."""
        if not self.config.enable_tiling:
            return process_fn(image, **kwargs)

        # Check if tiling is needed
        _, _, height, width = image.shape
        tile_h, tile_w = self.tile_size

        if height <= tile_h and width <= tile_w:
            # Image is small enough, process directly
            return process_fn(image, **kwargs)

        logger.info(f"Processing {height}x{width} image with {tile_h}x{tile_w} tiles")

        # Calculate tile positions
        tile_positions = self._calculate_tile_positions(
            height, width, tile_h, tile_w, self.overlap
        )

        # Process tiles
        processed_tiles = []
        tile_weights = []

        for i, (y_start, y_end, x_start, x_end) in enumerate(tile_positions):
            # Extract tile
            tile = image[:, :, y_start:y_end, x_start:x_end]

            # Process tile
            with torch.no_grad():
                processed_tile = process_fn(tile, **kwargs)

            # Create weight mask for blending
            weight_mask = self._create_weight_mask(
                processed_tile.shape[-2:], self.overlap, self.blend_mode
            )

            processed_tiles.append(
                (processed_tile, weight_mask, y_start, y_end, x_start, x_end)
            )

        # Reconstruct full image
        result = self._reconstruct_from_tiles(
            processed_tiles, (height, width), image.shape[:2]
        )

        return result

    def _calculate_tile_positions(
        self, height: int, width: int, tile_h: int, tile_w: int, overlap: int
    ) -> List[Tuple[int, int, int, int]]:
        """Calculate tile positions with overlap."""
        positions = []

        # Calculate step sizes
        step_h = tile_h - overlap
        step_w = tile_w - overlap

        # Generate tile positions
        for y in range(0, height, step_h):
            for x in range(0, width, step_w):
                y_start = y
                y_end = min(y + tile_h, height)
                x_start = x
                x_end = min(x + tile_w, width)

                # Adjust for minimum tile size
                if y_end - y_start < tile_h // 2:
                    continue
                if x_end - x_start < tile_w // 2:
                    continue

                positions.append((y_start, y_end, x_start, x_end))

        return positions

    def _create_weight_mask(
        self, tile_shape: Tuple[int, int], overlap: int, blend_mode: str
    ) -> torch.Tensor:
        """Create weight mask for tile blending."""
        height, width = tile_shape
        mask = torch.ones(height, width)

        if overlap == 0:
            return mask

        if blend_mode == "linear":
            # Linear blending
            for i in range(overlap):
                weight = i / overlap
                # Top edge
                if i < height:
                    mask[i, :] *= weight
                # Bottom edge
                if height - 1 - i >= 0:
                    mask[height - 1 - i, :] *= weight
                # Left edge
                if i < width:
                    mask[:, i] *= weight
                # Right edge
                if width - 1 - i >= 0:
                    mask[:, width - 1 - i] *= weight

        elif blend_mode == "cosine":
            # Cosine blending
            for i in range(overlap):
                weight = 0.5 * (1 - math.cos(math.pi * i / overlap))
                if i < height:
                    mask[i, :] *= weight
                if height - 1 - i >= 0:
                    mask[height - 1 - i, :] *= weight
                if i < width:
                    mask[:, i] *= weight
                if width - 1 - i >= 0:
                    mask[:, width - 1 - i] *= weight

        elif blend_mode == "gaussian":
            # Gaussian blending
            sigma = overlap / 3.0
            for i in range(overlap):
                weight = math.exp(-0.5 * ((i - overlap) / sigma) ** 2)
                if i < height:
                    mask[i, :] *= weight
                if height - 1 - i >= 0:
                    mask[height - 1 - i, :] *= weight
                if i < width:
                    mask[:, i] *= weight
                if width - 1 - i >= 0:
                    mask[:, width - 1 - i] *= weight

        return mask

    def _reconstruct_from_tiles(
        self,
        processed_tiles: List[Tuple[torch.Tensor, torch.Tensor, int, int, int, int]],
        output_shape: Tuple[int, int],
        batch_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Reconstruct full image from processed tiles."""
        height, width = output_shape
        batch_size, channels = batch_shape

        # Initialize output tensors
        result = torch.zeros(
            batch_size, channels, height, width, dtype=processed_tiles[0][0].dtype
        )
        weight_sum = torch.zeros(height, width)

        # Accumulate tiles
        for (
            processed_tile,
            weight_mask,
            y_start,
            y_end,
            x_start,
            x_end,
        ) in processed_tiles:
            # Add weighted tile to result
            result[:, :, y_start:y_end, x_start:x_end] += processed_tile * weight_mask
            weight_sum[y_start:y_end, x_start:x_end] += weight_mask

        # Normalize by weight sum
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        result = result / weight_sum.unsqueeze(0).unsqueeze(0)

        return result


class PerformanceProfiler:
    """Performance profiling and benchmarking."""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.enabled = config.enable_profiling

        # Profiling data
        self.timing_data = defaultdict(list)
        self.memory_data = defaultdict(list)
        self.compute_data = defaultdict(list)

        if self.enabled:
            logger.info("Performance profiling enabled")

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile an operation."""
        if not self.enabled:
            yield
            return

        # Start profiling
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            yield
        finally:
            # End profiling
            end_time = time.time()
            end_memory = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )

            # Record data
            duration = end_time - start_time
            memory_used = (end_memory - start_memory) / 1024**2  # MB

            self.timing_data[operation_name].append(duration)
            self.memory_data[operation_name].append(memory_used)

            logger.debug(f"{operation_name}: {duration:.3f}s, {memory_used:.1f}MB")

    def get_profile_summary(self) -> Dict[str, Dict[str, float]]:
        """Get profiling summary."""
        summary = {}

        for operation in self.timing_data:
            timings = self.timing_data[operation]
            memories = self.memory_data[operation]

            summary[operation] = {
                "count": len(timings),
                "total_time": sum(timings),
                "avg_time": np.mean(timings),
                "min_time": min(timings),
                "max_time": max(timings),
                "std_time": np.std(timings),
                "total_memory": sum(memories),
                "avg_memory": np.mean(memories),
                "peak_memory": max(memories) if memories else 0,
            }

        return summary

    def log_profile_summary(self):
        """Log profiling summary."""
        if not self.enabled:
            return

        summary = self.get_profile_summary()

        logger.info("Performance Profile Summary:")
        logger.info("-" * 60)

        for operation, stats in summary.items():
            logger.info(f"{operation}:")
            logger.info(f"  Count: {stats['count']}")
            logger.info(
                f"  Time: {stats['avg_time']:.3f}s avg, {stats['total_time']:.3f}s total"
            )
            logger.info(
                f"  Memory: {stats['avg_memory']:.1f}MB avg, {stats['peak_memory']:.1f}MB peak"
            )


class PerformanceOptimizer:
    """Main performance optimization coordinator."""

    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()

        # Initialize components
        self.memory_manager = MemoryManager(self.config)
        self.mixed_precision = MixedPrecisionManager(self.config)
        self.batch_optimizer = BatchSizeOptimizer(self.config, self.memory_manager)
        self.tiled_processor = TiledProcessor(self.config)
        self.profiler = PerformanceProfiler(self.config)

        # Setup optimizations
        self._setup_optimizations()

        logger.info("PerformanceOptimizer initialized")

    def _setup_optimizations(self):
        """Setup global optimizations."""
        if torch.cuda.is_available():
            # Enable cuDNN benchmark
            if self.config.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN benchmark enabled")

            # Enable TF32
            if self.config.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 enabled")

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply model-level optimizations."""
        # Compile model if requested
        if self.config.enable_torch_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode=self.config.compile_mode)
                logger.info(f"Model compiled with mode: {self.config.compile_mode}")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")

        return model

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "memory_info": self.memory_manager.get_memory_info(),
            "batch_optimizer": {
                "current_batch_size": self.batch_optimizer.current_batch_size,
                "optimal_batch_size": self.batch_optimizer.optimal_batch_size,
                "batch_size_history": self.batch_optimizer.batch_size_history[
                    -10:
                ],  # Last 10
            },
            "mixed_precision": {
                "enabled": self.mixed_precision.enabled,
                "dtype": str(self.config.mixed_precision_dtype),
            },
            "tiled_processing": {
                "enabled": self.config.enable_tiling,
                "tile_size": self.config.tile_size,
                "overlap": self.config.tile_overlap,
            },
            "profiler": self.profiler.get_profile_summary()
            if self.profiler.enabled
            else {},
        }

        return stats
