"""
Demonstration of performance optimizations for Poisson-Gaussian diffusion restoration.

This script showcases the various performance optimization features including:
- Mixed precision training and inference
- Memory management and optimization
- Batch size optimization
- Tiled processing for large images
- Performance profiling and benchmarking

Usage:
    python scripts/demo_performance_optimizations.py [--demo TYPE] [--device DEVICE]

    --demo: Type of demo to run (basic, memory, tiling, benchmarking, all)
    --device: Device to use (cpu, cuda)
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from core.benchmarking import BenchmarkConfig, ComprehensiveBenchmark
from core.logging_config import LoggingManager
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

# Setup logging
logging_manager = LoggingManager()
logger = logging_manager.setup_logging(level="INFO")


class DemoModel(nn.Module):
    """Demo model for performance testing."""

    def __init__(self, input_channels: int = 3, hidden_dim: int = 128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_channels, 3, padding=1),
            nn.Sigmoid(),
        )

        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        if isinstance(x, dict):
            x = x.get("x", x.get("electrons", x.get("input")))

        features = self.encoder(x)
        output = self.decoder(features)

        return {"prediction": output, "output": output, "features": features}


def demo_basic_optimizations(device: str = "cpu"):
    """Demonstrate basic performance optimizations."""
    print("\n" + "=" * 60)
    print("🚀 BASIC PERFORMANCE OPTIMIZATIONS DEMO")
    print("=" * 60)

    # Create performance configuration
    config = PerformanceConfig(
        mixed_precision=True,
        memory_efficient=True,
        auto_batch_size=True,
        enable_tiling=True,
        enable_profiling=True,
    )

    # Initialize performance optimizer
    optimizer = PerformanceOptimizer(config)

    print(f"✓ Performance optimizer initialized")
    print(f"  Device: {device}")
    print(f"  Mixed precision: {config.mixed_precision}")
    print(f"  Memory efficient: {config.memory_efficient}")
    print(f"  Auto batch size: {config.auto_batch_size}")
    print(f"  Tiling enabled: {config.enable_tiling}")

    # Create and optimize model
    model = DemoModel(hidden_dim=64).to(device)
    optimized_model = optimizer.optimize_model(model)

    print(f"✓ Model created and optimized")
    print(f"  Parameters: {model.num_params:,}")

    # Test mixed precision inference
    print(f"\n🔧 Testing Mixed Precision Inference:")

    test_input = torch.randn(4, 3, 256, 256).to(device)

    # Standard precision
    start_time = time.time()
    with torch.no_grad():
        standard_output = model(test_input)
    standard_time = time.time() - start_time

    # Mixed precision
    start_time = time.time()
    with torch.no_grad():
        with optimizer.mixed_precision.autocast_context():
            mixed_output = optimized_model(test_input)
    mixed_time = time.time() - start_time

    speedup = standard_time / mixed_time if mixed_time > 0 else 1.0

    print(f"  Standard precision: {standard_time:.3f}s")
    print(f"  Mixed precision: {mixed_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")

    # Test memory management
    print(f"\n💾 Testing Memory Management:")

    memory_info = optimizer.memory_manager.get_memory_info()
    print(
        f"  System memory: {memory_info['system_used_gb']:.1f}/{memory_info['system_total_gb']:.1f} GB"
    )

    if device == "cuda" and torch.cuda.is_available():
        print(f"  GPU memory: {memory_info['gpu_allocated_gb']:.2f} GB allocated")

    # Test with memory context
    with optimizer.memory_manager.memory_context("demo_inference"):
        large_batch = torch.randn(8, 3, 512, 512).to(device)
        with torch.no_grad():
            _ = optimized_model(large_batch)

    print(f"  ✓ Memory context test completed")

    # Get performance statistics
    stats = optimizer.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"  Mixed precision enabled: {stats['mixed_precision']['enabled']}")
    print(f"  Tiling enabled: {stats['tiled_processing']['enabled']}")
    print(f"  Tile size: {stats['tiled_processing']['tile_size']}")


def demo_memory_optimization(device: str = "cpu"):
    """Demonstrate memory optimization features."""
    print("\n" + "=" * 60)
    print("💾 MEMORY OPTIMIZATION DEMO")
    print("=" * 60)

    config = PerformanceConfig(
        memory_efficient=True,
        max_memory_gb=4.0 if device == "cuda" else None,
        memory_fraction=0.8,
    )

    memory_manager = MemoryManager(config)

    print(f"✓ Memory manager initialized")

    # Test memory tracking
    print(f"\n📈 Memory Tracking Test:")

    initial_info = memory_manager.get_memory_info()
    print(f"  Initial system memory: {initial_info['system_used_gb']:.1f} GB")

    # Allocate memory progressively
    tensors = []
    memory_history = []

    for i in range(5):
        # Allocate 100MB tensor
        tensor = torch.randn(100, 1024, 1024).to(device)  # ~400MB
        tensors.append(tensor)

        memory_info = memory_manager.get_memory_info()
        memory_history.append(memory_info["system_used_gb"])

        print(f"  Step {i+1}: {memory_info['system_used_gb']:.1f} GB system memory")

        if device == "cuda" and torch.cuda.is_available():
            print(f"           {memory_info['gpu_allocated_gb']:.2f} GB GPU memory")

    # Test memory optimization
    print(f"\n🔧 Memory Optimization:")

    pre_optimization = memory_manager.get_memory_info()
    memory_manager.optimize_memory()
    post_optimization = memory_manager.get_memory_info()

    print(f"  Before optimization: {pre_optimization['system_used_gb']:.1f} GB")
    print(f"  After optimization: {post_optimization['system_used_gb']:.1f} GB")

    # Clear tensors
    del tensors
    memory_manager.optimize_memory()

    final_info = memory_manager.get_memory_info()
    print(f"  After cleanup: {final_info['system_used_gb']:.1f} GB")


def demo_batch_size_optimization(device: str = "cpu"):
    """Demonstrate batch size optimization."""
    print("\n" + "=" * 60)
    print("📦 BATCH SIZE OPTIMIZATION DEMO")
    print("=" * 60)

    config = PerformanceConfig(
        auto_batch_size=True,
        min_batch_size=1,
        max_batch_size=16,
        batch_size_growth_factor=1.5,
    )

    memory_manager = MemoryManager(config)
    batch_optimizer = BatchSizeOptimizer(config, memory_manager)

    print(f"✓ Batch size optimizer initialized")
    print(f"  Range: {config.min_batch_size} - {config.max_batch_size}")

    # Create model and sample batch
    model = DemoModel(hidden_dim=32).to(device)
    sample_batch = {
        "x": torch.randn(1, 3, 128, 128).to(device),
        "scale": torch.tensor(1000.0).to(device),
    }

    print(f"\n🔍 Finding Optimal Batch Size:")

    # Find optimal batch size
    optimal_size = batch_optimizer.find_optimal_batch_size(
        model, sample_batch, max_trials=5
    )

    print(f"  ✓ Optimal batch size found: {optimal_size}")

    # Test different batch sizes
    print(f"\n⚡ Performance Comparison:")

    test_sizes = [1, 2, 4, 8] if optimal_size >= 8 else [1, 2, 4]

    for batch_size in test_sizes:
        if batch_size > optimal_size:
            continue

        try:
            # Create test batch
            test_batch = batch_optimizer._create_test_batch(sample_batch, batch_size)

            # Time inference
            start_time = time.time()
            with torch.no_grad():
                _ = model(test_batch)
            end_time = time.time()

            inference_time = end_time - start_time
            throughput = batch_size / inference_time

            print(
                f"  Batch {batch_size:2d}: {inference_time:.3f}s, {throughput:.1f} samples/s"
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch {batch_size:2d}: OOM")
            else:
                raise e


def demo_tiled_processing(device: str = "cpu"):
    """Demonstrate tiled processing for large images."""
    print("\n" + "=" * 60)
    print("🧩 TILED PROCESSING DEMO")
    print("=" * 60)

    config = PerformanceConfig(
        enable_tiling=True,
        tile_size=(128, 128),
        tile_overlap=32,
        tile_blend_mode="linear",
    )

    tiled_processor = TiledProcessor(config)

    print(f"✓ Tiled processor initialized")
    print(f"  Tile size: {config.tile_size}")
    print(f"  Overlap: {config.tile_overlap}")
    print(f"  Blend mode: {config.tile_blend_mode}")

    # Create model
    model = DemoModel(hidden_dim=32).to(device)

    # Test with different image sizes
    test_sizes = [(256, 256), (512, 512), (1024, 1024)]

    print(f"\n🖼️ Processing Different Image Sizes:")

    for height, width in test_sizes:
        print(f"\n  Testing {height}x{width} image:")

        # Create large image
        large_image = torch.randn(1, 3, height, width).to(device)

        # Define processing function
        def process_fn(x):
            with torch.no_grad():
                return model(x)["prediction"]

        # Process with tiling
        start_time = time.time()
        tiled_result = tiled_processor.process_tiled(large_image, process_fn)
        tiled_time = time.time() - start_time

        # Process without tiling (if memory allows)
        try:
            start_time = time.time()
            direct_result = process_fn(large_image)
            direct_time = time.time() - start_time

            # Compare results
            mse = torch.mean((tiled_result - direct_result) ** 2).item()

            print(f"    Tiled processing: {tiled_time:.3f}s")
            print(f"    Direct processing: {direct_time:.3f}s")
            print(f"    MSE difference: {mse:.6f}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    Tiled processing: {tiled_time:.3f}s")
                print(f"    Direct processing: OOM (tiling required)")
            else:
                raise e

        print(f"    ✓ Result shape: {tiled_result.shape}")


def demo_guidance_optimization(device: str = "cpu"):
    """Demonstrate optimized guidance computation."""
    print("\n" + "=" * 60)
    print("🎯 OPTIMIZED GUIDANCE DEMO")
    print("=" * 60)

    # Create guidance configuration
    guidance_config = GuidanceConfig(
        use_fused_ops=True, use_vectorized_ops=True, cache_intermediate=True
    )

    performance_config = PerformanceConfig(mixed_precision=True, memory_efficient=True)

    # Initialize optimized guidance
    guidance = OptimizedPoissonGuidance(guidance_config, performance_config)

    print(f"✓ Optimized guidance initialized")
    print(f"  Fused operations: {guidance_config.use_fused_ops}")
    print(f"  Vectorized operations: {guidance_config.use_vectorized_ops}")
    print(f"  Mixed precision: {performance_config.mixed_precision}")

    # Create test data
    batch_size = 4
    image_size = (128, 128)

    x_pred = torch.randn(batch_size, 3, *image_size).to(device)
    y_obs = torch.randn(batch_size, 3, *image_size).to(device) * 1000 + 100
    scale = torch.tensor(1000.0).to(device)
    background = torch.tensor(100.0).to(device)
    read_noise = torch.tensor(5.0).to(device)

    print(f"\n⚡ Performance Comparison:")

    # Test WLS guidance
    modes = ["wls", "exact"]

    for mode in modes:
        print(f"\n  {mode.upper()} Guidance:")

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = guidance(x_pred, y_obs, scale, background, read_noise, mode=mode)

        # Benchmark
        times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                grad = guidance(x_pred, y_obs, scale, background, read_noise, mode=mode)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"    Time: {avg_time:.4f} ± {std_time:.4f}s")
        print(f"    Gradient shape: {grad.shape}")
        print(f"    Gradient range: [{grad.min():.3f}, {grad.max():.3f}]")
        print(f"    All finite: {torch.isfinite(grad).all()}")

    # Test likelihood computation
    print(f"\n  Likelihood Computation:")

    start_time = time.time()
    likelihood = guidance.compute_likelihood(
        x_pred, y_obs, scale, background, read_noise
    )
    end_time = time.time()

    print(f"    Time: {end_time - start_time:.4f}s")
    print(f"    Likelihood shape: {likelihood.shape}")
    print(f"    Likelihood range: [{likelihood.min():.3f}, {likelihood.max():.3f}]")


def demo_comprehensive_benchmarking(device: str = "cpu"):
    """Demonstrate comprehensive benchmarking."""
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE BENCHMARKING DEMO")
    print("=" * 60)

    # Create benchmark configuration
    config = BenchmarkConfig(
        image_sizes=[(128, 128), (256, 256)],
        batch_sizes=[1, 2, 4],
        num_warmup_runs=3,
        num_benchmark_runs=5,
        throughput_duration=5.0,
        save_results=True,
        results_dir="demo_benchmark_results",
        plot_results=False,  # Disable for demo
    )

    # Initialize benchmark suite
    benchmark = ComprehensiveBenchmark(config)

    print(f"✓ Benchmark suite initialized")
    print(f"  Image sizes: {config.image_sizes}")
    print(f"  Batch sizes: {config.batch_sizes}")
    print(f"  Benchmark runs: {config.num_benchmark_runs}")

    # Create models for comparison
    models = [
        (DemoModel(hidden_dim=32).to(device), "small_model"),
        (DemoModel(hidden_dim=64).to(device), "medium_model"),
    ]

    print(f"\n🏃 Running Benchmarks:")

    results = {}

    for model, model_name in models:
        print(f"\n  Benchmarking {model_name}...")

        # Run comprehensive benchmark
        model_results = benchmark.run_full_benchmark(model, model_name)
        results[model_name] = model_results

        # Get performance summary
        summary = benchmark.get_performance_summary(model_name)

        print(f"    ✓ Inference completed")
        if "inference" in summary:
            print(
                f"      Max throughput: {summary['inference']['max_throughput']:.1f} samples/s"
            )
            print(
                f"      Memory usage: {summary['inference']['min_memory_usage']:.1f}-{summary['inference']['max_memory_usage']:.1f} MB"
            )

        if "memory" in summary:
            print(
                f"      Memory leak: {'Yes' if summary['memory']['has_memory_leak'] else 'No'}"
            )

        if "throughput" in summary:
            print(
                f"      Sustained throughput: {summary['throughput']['sustained_throughput']:.1f} samples/s"
            )

    print(f"\n📈 Benchmark Comparison:")

    # Compare models
    if len(results) > 1:
        model_names = list(results.keys())

        for size in config.image_sizes:
            print(f"\n  {size[0]}x{size[1]} Images:")

            for model_name in model_names:
                model_results = results[model_name]

                # Find results for this image size
                size_results = [
                    r for r in model_results["inference"] if r["image_size"] == size
                ]

                if size_results:
                    max_throughput = max(r["throughput"] for r in size_results)
                    min_memory = min(r["peak_memory_mb"] for r in size_results)

                    print(
                        f"    {model_name}: {max_throughput:.1f} samples/s, {min_memory:.1f} MB"
                    )

    print(f"\n✓ Benchmark results saved to: {config.results_dir}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Performance Optimizations Demo")
    parser.add_argument(
        "--demo",
        choices=[
            "basic",
            "memory",
            "batch",
            "tiling",
            "guidance",
            "benchmarking",
            "all",
        ],
        default="all",
        help="Type of demo to run",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    print("🚀 PERFORMANCE OPTIMIZATIONS DEMONSTRATION")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")

    if args.device == "cuda":
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(
                f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
        else:
            print("CUDA not available, falling back to CPU")
            args.device = "cpu"

    # Run selected demos
    demos = {
        "basic": demo_basic_optimizations,
        "memory": demo_memory_optimization,
        "batch": demo_batch_size_optimization,
        "tiling": demo_tiled_processing,
        "guidance": demo_guidance_optimization,
        "benchmarking": demo_comprehensive_benchmarking,
    }

    if args.demo == "all":
        for demo_name, demo_func in demos.items():
            try:
                demo_func(args.device)
            except Exception as e:
                print(f"\n❌ Error in {demo_name} demo: {e}")
                continue
    else:
        demos[args.demo](args.device)

    print("\n" + "=" * 60)
    print("✅ PERFORMANCE OPTIMIZATIONS DEMO COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
