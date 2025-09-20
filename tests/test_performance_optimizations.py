"""
Tests for performance optimizations.

This module tests the performance optimization components including:
- Mixed precision training and inference
- Memory management and optimization
- Batch processing optimization
- Tiled processing for large images
- Performance profiling and benchmarking

Requirements tested: 7.1-7.6 from requirements.md
Task tested: 6.1 from tasks.md
"""

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from core.benchmarking import (
    BenchmarkConfig,
    ComprehensiveBenchmark,
    InferenceBenchmark,
    MemoryBenchmark,
    ThroughputBenchmark,
)
from core.exceptions import PerformanceError
from core.optimized_guidance import (
    GuidanceConfig,
    OptimizedPoissonGuidance,
    OptimizedSampler,
)
from core.performance import (
    BatchSizeOptimizer,
    MemoryManager,
    MixedPrecisionManager,
    PerformanceConfig,
    PerformanceOptimizer,
    TiledProcessor,
)


class SimpleTestModel(nn.Module):
    """Simple model for testing performance optimizations."""

    def __init__(self, input_channels: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, input_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if isinstance(x, dict):
            x = x.get("x", x.get("electrons", x.get("input")))

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))

        return {"prediction": x, "output": x}


class TestPerformanceConfig:
    """Test performance configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PerformanceConfig()

        assert config.mixed_precision is True
        assert config.memory_efficient is True
        assert config.auto_batch_size is True
        assert config.enable_tiling is True
        assert config.enable_cudnn_benchmark is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PerformanceConfig(
            mixed_precision=False,
            max_memory_gb=4.0,
            tile_size=(256, 256),
            enable_profiling=True,
        )

        assert config.mixed_precision is False
        assert config.max_memory_gb == 4.0
        assert config.tile_size == (256, 256)
        assert config.enable_profiling is True


class TestMemoryManager:
    """Test memory management functionality."""

    def test_initialization(self):
        """Test memory manager initialization."""
        config = PerformanceConfig()
        manager = MemoryManager(config)

        assert manager.config == config
        assert manager.peak_memory == 0
        assert isinstance(manager.memory_history, list)

    def test_memory_info(self):
        """Test memory information retrieval."""
        config = PerformanceConfig()
        manager = MemoryManager(config)

        info = manager.get_memory_info()

        # Check required fields
        assert "system_total_gb" in info
        assert "system_used_gb" in info
        assert "system_available_gb" in info
        assert "system_percent" in info

        # Check values are reasonable
        assert info["system_total_gb"] > 0
        assert 0 <= info["system_percent"] <= 100

    def test_memory_context(self):
        """Test memory context manager."""
        config = PerformanceConfig()
        manager = MemoryManager(config)

        # Test context manager doesn't crash
        with manager.memory_context("test_operation"):
            # Allocate some memory
            test_tensor = torch.randn(100, 100)
            assert test_tensor.numel() == 10000

    def test_memory_optimization(self):
        """Test memory optimization."""
        config = PerformanceConfig()
        manager = MemoryManager(config)

        # Should not crash
        manager.optimize_memory()

        # Memory history should be maintained
        assert isinstance(manager.memory_history, list)


class TestMixedPrecisionManager:
    """Test mixed precision functionality."""

    def test_initialization_cuda_available(self):
        """Test initialization when CUDA is available."""
        config = PerformanceConfig(mixed_precision=True)

        with patch("torch.cuda.is_available", return_value=True):
            manager = MixedPrecisionManager(config)
            assert manager.enabled is True
            assert manager.scaler is not None

    def test_initialization_cuda_unavailable(self):
        """Test initialization when CUDA is unavailable."""
        config = PerformanceConfig(mixed_precision=True)

        with patch("torch.cuda.is_available", return_value=False):
            manager = MixedPrecisionManager(config)
            assert manager.enabled is False
            assert manager.scaler is None

    def test_autocast_context(self):
        """Test autocast context manager."""
        config = PerformanceConfig(mixed_precision=True)

        with patch("torch.cuda.is_available", return_value=True):
            manager = MixedPrecisionManager(config)

            # Test context manager
            with manager.autocast_context():
                x = torch.randn(10, 10)
                y = x * 2
                assert y.shape == (10, 10)

    def test_loss_scaling(self):
        """Test loss scaling functionality."""
        config = PerformanceConfig(mixed_precision=True)

        with patch("torch.cuda.is_available", return_value=True):
            manager = MixedPrecisionManager(config)

            loss = torch.tensor(1.0, requires_grad=True)
            scaled_loss = manager.scale_loss(loss)

            # Should return a tensor
            assert isinstance(scaled_loss, torch.Tensor)

    def test_optimizer_step(self):
        """Test optimizer step with mixed precision."""
        config = PerformanceConfig(mixed_precision=True)

        with patch("torch.cuda.is_available", return_value=True):
            manager = MixedPrecisionManager(config)

            # Create mock optimizer
            optimizer = Mock()

            # Should not crash
            result = manager.step_optimizer(optimizer)
            assert isinstance(result, bool)


class TestBatchSizeOptimizer:
    """Test batch size optimization."""

    def test_initialization(self):
        """Test batch size optimizer initialization."""
        config = PerformanceConfig(min_batch_size=1, max_batch_size=16)
        memory_manager = MemoryManager(config)
        optimizer = BatchSizeOptimizer(config, memory_manager)

        assert optimizer.current_batch_size == config.min_batch_size
        assert optimizer.optimal_batch_size is None

    def test_create_test_batch(self):
        """Test test batch creation."""
        config = PerformanceConfig()
        memory_manager = MemoryManager(config)
        optimizer = BatchSizeOptimizer(config, memory_manager)

        sample_batch = {
            "x": torch.randn(1, 3, 32, 32),
            "y": torch.randn(1, 3, 32, 32),
            "scale": torch.tensor(1000.0),
            "metadata": {"test": "value"},
        }

        test_batch = optimizer._create_test_batch(sample_batch, 4)

        assert test_batch["x"].shape == (4, 3, 32, 32)
        assert test_batch["y"].shape == (4, 3, 32, 32)
        assert len(test_batch["metadata"]) == 4

    def test_find_optimal_batch_size(self):
        """Test optimal batch size finding."""
        config = PerformanceConfig(
            auto_batch_size=True, min_batch_size=1, max_batch_size=4
        )
        memory_manager = MemoryManager(config)
        optimizer = BatchSizeOptimizer(config, memory_manager)

        model = SimpleTestModel()
        sample_batch = {"x": torch.randn(1, 3, 32, 32), "scale": torch.tensor(1000.0)}

        optimal_size = optimizer.find_optimal_batch_size(
            model, sample_batch, max_trials=3
        )

        assert 1 <= optimal_size <= 4
        assert optimizer.optimal_batch_size == optimal_size


class TestTiledProcessor:
    """Test tiled processing functionality."""

    def test_initialization(self):
        """Test tiled processor initialization."""
        config = PerformanceConfig(tile_size=(128, 128), tile_overlap=32)
        processor = TiledProcessor(config)

        assert processor.tile_size == (128, 128)
        assert processor.overlap == 32

    def test_calculate_tile_positions(self):
        """Test tile position calculation."""
        config = PerformanceConfig(tile_size=(128, 128), tile_overlap=32)
        processor = TiledProcessor(config)

        positions = processor._calculate_tile_positions(256, 256, 128, 128, 32)

        # Should have multiple tiles for 256x256 image with 128x128 tiles
        assert len(positions) > 1

        # Check position format
        for pos in positions:
            y_start, y_end, x_start, x_end = pos
            assert 0 <= y_start < y_end <= 256
            assert 0 <= x_start < x_end <= 256

    def test_create_weight_mask(self):
        """Test weight mask creation."""
        config = PerformanceConfig()
        processor = TiledProcessor(config)

        # Test different blend modes
        for blend_mode in ["linear", "cosine", "gaussian"]:
            mask = processor._create_weight_mask((64, 64), 16, blend_mode)

            assert mask.shape == (64, 64)
            assert torch.all(mask >= 0)
            assert torch.all(mask <= 1)

    def test_process_tiled_small_image(self):
        """Test tiled processing with small image (no tiling needed)."""
        config = PerformanceConfig(tile_size=(128, 128), enable_tiling=True)
        processor = TiledProcessor(config)

        # Small image that doesn't need tiling
        image = torch.randn(1, 3, 64, 64)

        def dummy_process_fn(x):
            return x * 2

        result = processor.process_tiled(image, dummy_process_fn)

        # Should be processed directly without tiling
        expected = dummy_process_fn(image)
        assert torch.allclose(result, expected)

    def test_process_tiled_large_image(self):
        """Test tiled processing with large image."""
        config = PerformanceConfig(
            tile_size=(64, 64), tile_overlap=16, enable_tiling=True
        )
        processor = TiledProcessor(config)

        # Large image that needs tiling
        image = torch.randn(1, 3, 128, 128)

        def dummy_process_fn(x):
            return x * 2

        result = processor.process_tiled(image, dummy_process_fn)

        # Result should have same shape as input
        assert result.shape == image.shape

        # Result should be approximately correct (some blending artifacts expected)
        expected = dummy_process_fn(image)
        # Allow for some difference due to blending
        assert torch.allclose(result, expected, atol=0.1)


class TestOptimizedPoissonGuidance:
    """Test optimized Poisson guidance."""

    def test_initialization(self):
        """Test guidance initialization."""
        config = GuidanceConfig()
        guidance = OptimizedPoissonGuidance(config)

        assert guidance.config == config
        assert hasattr(guidance, "eps_variance")
        assert hasattr(guidance, "eps_likelihood")

    def test_prepare_parameters(self):
        """Test parameter preparation for broadcasting."""
        guidance = OptimizedPoissonGuidance()

        scale = torch.tensor(1000.0)
        background = torch.tensor(100.0)
        read_noise = torch.tensor(5.0)
        target_shape = torch.Size([2, 3, 32, 32])

        prep_scale, prep_bg, prep_rn = guidance._prepare_parameters(
            scale, background, read_noise, target_shape
        )

        # Should be reshaped for broadcasting
        assert prep_scale.shape == (2, 1, 1, 1)
        assert prep_bg.shape == (2, 1, 1, 1)
        assert prep_rn.shape == (2, 1, 1, 1)

    def test_wls_guidance_computation(self):
        """Test WLS guidance computation."""
        guidance = OptimizedPoissonGuidance()

        # Create test data
        x_pred = torch.randn(2, 3, 32, 32)
        y_obs = torch.randn(2, 3, 32, 32) * 1000 + 100
        scale = torch.tensor(1000.0)
        background = torch.tensor(100.0)
        read_noise = torch.tensor(5.0)

        # Test WLS guidance
        grad = guidance(x_pred, y_obs, scale, background, read_noise, mode="wls")

        assert grad.shape == x_pred.shape
        assert torch.isfinite(grad).all()

    def test_exact_guidance_computation(self):
        """Test exact guidance computation."""
        guidance = OptimizedPoissonGuidance()

        # Create test data
        x_pred = torch.randn(2, 3, 32, 32)
        y_obs = torch.randn(2, 3, 32, 32) * 1000 + 100
        scale = torch.tensor(1000.0)
        background = torch.tensor(100.0)
        read_noise = torch.tensor(5.0)

        # Test exact guidance
        grad = guidance(x_pred, y_obs, scale, background, read_noise, mode="exact")

        assert grad.shape == x_pred.shape
        assert torch.isfinite(grad).all()

    def test_likelihood_computation(self):
        """Test likelihood computation."""
        guidance = OptimizedPoissonGuidance()

        # Create test data
        x_pred = torch.randn(2, 3, 32, 32)
        y_obs = torch.randn(2, 3, 32, 32) * 1000 + 100
        scale = torch.tensor(1000.0)
        background = torch.tensor(100.0)
        read_noise = torch.tensor(5.0)

        likelihood = guidance.compute_likelihood(
            x_pred, y_obs, scale, background, read_noise
        )

        assert likelihood.shape == (2,)  # Batch dimension only
        assert torch.isfinite(likelihood).all()


class TestOptimizedSampler:
    """Test optimized sampler."""

    def test_initialization(self):
        """Test sampler initialization."""
        guidance = OptimizedPoissonGuidance()
        sampler = OptimizedSampler(guidance)

        assert sampler.guidance == guidance

    def test_estimate_max_chunk_size(self):
        """Test chunk size estimation."""
        guidance = OptimizedPoissonGuidance()
        sampler = OptimizedSampler(guidance)

        sample_tensor = torch.randn(8, 3, 64, 64)
        chunk_size = sampler._estimate_max_chunk_size(sample_tensor)

        assert 1 <= chunk_size <= 8

    def test_sample_step(self):
        """Test single sampling step."""
        guidance = OptimizedPoissonGuidance()
        sampler = OptimizedSampler(guidance)

        # Create mock model
        model = SimpleTestModel()

        # Create test data
        x_t = torch.randn(1, 3, 32, 32)
        t = torch.tensor([0.5])
        y_obs = torch.randn(1, 3, 32, 32) * 1000 + 100
        scale = torch.tensor(1000.0)
        background = torch.tensor(100.0)
        read_noise = torch.tensor(5.0)

        result = sampler.sample_step(
            model, x_t, t, y_obs, scale, background, read_noise
        )

        assert result.shape == x_t.shape
        assert torch.isfinite(result).all()


class TestPerformanceOptimizer:
    """Test main performance optimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        assert optimizer.config == config
        assert isinstance(optimizer.memory_manager, MemoryManager)
        assert isinstance(optimizer.mixed_precision, MixedPrecisionManager)
        assert isinstance(optimizer.batch_optimizer, BatchSizeOptimizer)
        assert isinstance(optimizer.tiled_processor, TiledProcessor)

    def test_optimize_model(self):
        """Test model optimization."""
        config = PerformanceConfig(enable_torch_compile=False)  # Disable for testing
        optimizer = PerformanceOptimizer(config)

        model = SimpleTestModel()
        optimized_model = optimizer.optimize_model(model)

        # Should return a model (possibly the same one)
        assert isinstance(optimized_model, nn.Module)

    def test_get_performance_stats(self):
        """Test performance statistics retrieval."""
        config = PerformanceConfig()
        optimizer = PerformanceOptimizer(config)

        stats = optimizer.get_performance_stats()

        # Check required fields
        assert "memory_info" in stats
        assert "batch_optimizer" in stats
        assert "mixed_precision" in stats
        assert "tiled_processing" in stats


class TestBenchmarking:
    """Test benchmarking functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_benchmark_config(self):
        """Test benchmark configuration."""
        config = BenchmarkConfig(
            image_sizes=[(128, 128), (256, 256)],
            batch_sizes=[1, 2],
            num_warmup_runs=2,
            num_benchmark_runs=3,
        )

        assert config.image_sizes == [(128, 128), (256, 256)]
        assert config.batch_sizes == [1, 2]
        assert config.num_warmup_runs == 2
        assert config.num_benchmark_runs == 3

    def test_inference_benchmark(self):
        """Test inference benchmarking."""
        config = BenchmarkConfig(
            image_sizes=[(64, 64)],
            batch_sizes=[1, 2],
            num_warmup_runs=1,
            num_benchmark_runs=2,
        )
        benchmark = InferenceBenchmark(config)

        model = SimpleTestModel()
        results = benchmark.benchmark_model(model, "test_model")

        assert len(results) > 0

        for result in results:
            assert result.test_name == "test_model"
            assert result.inference_time > 0
            assert result.throughput > 0
            assert result.peak_memory_mb >= 0

    def test_memory_benchmark(self):
        """Test memory benchmarking."""
        config = BenchmarkConfig(image_sizes=[(64, 64)], batch_sizes=[1, 2])
        benchmark = MemoryBenchmark(config)

        model = SimpleTestModel()
        results = benchmark.benchmark_memory_scaling(model, "test_memory")

        assert "image_sizes" in results
        assert "batch_sizes" in results
        assert "memory_usage" in results
        assert len(results["image_sizes"]) > 0

    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        config = BenchmarkConfig()
        benchmark = MemoryBenchmark(config)

        model = SimpleTestModel()
        results = benchmark.test_memory_leaks(
            model, num_iterations=10, test_name="leak_test"
        )

        assert "has_memory_leak" in results
        assert "memory_growth_ratio" in results
        assert "memory_history" in results
        assert isinstance(results["has_memory_leak"], bool)
        assert results["memory_growth_ratio"] > 0

    def test_throughput_benchmark(self):
        """Test throughput benchmarking."""
        config = BenchmarkConfig(throughput_duration=2.0)  # Short duration for testing
        benchmark = ThroughputBenchmark(config)

        model = SimpleTestModel()
        results = benchmark.benchmark_sustained_throughput(
            model, image_size=(64, 64), batch_size=1, test_name="throughput_test"
        )

        assert "avg_throughput" in results
        assert "peak_throughput" in results
        assert "throughput_history" in results
        assert results["avg_throughput"] > 0
        assert len(results["throughput_history"]) > 0

    def test_comprehensive_benchmark(self, temp_dir):
        """Test comprehensive benchmarking suite."""
        config = BenchmarkConfig(
            image_sizes=[(64, 64)],
            batch_sizes=[1],
            num_warmup_runs=1,
            num_benchmark_runs=2,
            throughput_duration=1.0,
            save_results=True,
            results_dir=temp_dir,
            plot_results=False,  # Disable plotting for testing
        )
        benchmark = ComprehensiveBenchmark(config)

        model = SimpleTestModel()
        results = benchmark.run_full_benchmark(model, "comprehensive_test")

        assert "inference" in results
        assert "memory" in results
        assert "throughput" in results

        # Check that results were saved
        results_file = Path(temp_dir) / "comprehensive_test_results.json"
        assert results_file.exists()

    def test_performance_summary(self, temp_dir):
        """Test performance summary generation."""
        config = BenchmarkConfig(
            image_sizes=[(64, 64)],
            batch_sizes=[1],
            num_warmup_runs=1,
            num_benchmark_runs=2,
            throughput_duration=1.0,
            results_dir=temp_dir,
            plot_results=False,
        )
        benchmark = ComprehensiveBenchmark(config)

        model = SimpleTestModel()
        benchmark.run_full_benchmark(model, "summary_test")

        summary = benchmark.get_performance_summary("summary_test")

        assert "inference" in summary
        assert "memory" in summary
        assert "throughput" in summary


class TestIntegration:
    """Integration tests for performance optimizations."""

    def test_end_to_end_optimization(self):
        """Test end-to-end performance optimization pipeline."""
        # Create performance optimizer
        perf_config = PerformanceConfig(
            mixed_precision=True,
            memory_efficient=True,
            enable_tiling=True,
            enable_profiling=True,
        )
        optimizer = PerformanceOptimizer(perf_config)

        # Create model and optimize it
        model = SimpleTestModel()
        optimized_model = optimizer.optimize_model(model)

        # Create guidance
        guidance_config = GuidanceConfig(use_fused_ops=True)
        guidance = OptimizedPoissonGuidance(guidance_config, perf_config)

        # Test inference with optimizations
        with optimizer.memory_manager.memory_context("test_inference"):
            with optimizer.mixed_precision.autocast_context():
                # Create test data
                x = torch.randn(2, 3, 64, 64)
                y = torch.randn(2, 3, 64, 64) * 1000 + 100

                # Model inference
                outputs = optimized_model({"x": x})

                # Guidance computation
                grad = guidance(
                    outputs["prediction"],
                    y,
                    torch.tensor(1000.0),
                    torch.tensor(100.0),
                    torch.tensor(5.0),
                )

                assert grad.shape == x.shape
                assert torch.isfinite(grad).all()

        # Get performance statistics
        stats = optimizer.get_performance_stats()
        assert "memory_info" in stats
        assert "mixed_precision" in stats

    def test_tiled_processing_integration(self):
        """Test tiled processing integration."""
        perf_config = PerformanceConfig(
            enable_tiling=True, tile_size=(32, 32), tile_overlap=8
        )
        optimizer = PerformanceOptimizer(perf_config)

        # Create large image
        large_image = torch.randn(1, 3, 128, 128)

        # Define processing function
        def process_fn(x):
            # Simple processing function
            return x * 2 + 1

        # Process with tiling
        result = optimizer.tiled_processor.process_tiled(large_image, process_fn)

        assert result.shape == large_image.shape
        assert torch.isfinite(result).all()

        # Result should be approximately correct
        expected = process_fn(large_image)
        assert torch.allclose(result, expected, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
