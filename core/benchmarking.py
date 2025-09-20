"""
Performance benchmarking and profiling for Poisson-Gaussian diffusion restoration.

This module provides comprehensive benchmarking tools to measure and optimize
performance across different components and configurations.

Key features:
- Inference speed benchmarking
- Memory usage profiling
- Throughput measurement
- Scalability testing
- Performance regression detection

Requirements addressed: 7.1-7.6 from requirements.md
Task: 6.1 from tasks.md
"""

import gc
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn

from core.error_handlers import ErrorHandler, safe_operation
from core.exceptions import PoissonDiffusionError
from core.logging_config import get_logger
from core.optimized_guidance import OptimizedPoissonGuidance, OptimizedSampler
from core.performance import PerformanceConfig, PerformanceOptimizer

logger = get_logger(__name__)


class BenchmarkError(PoissonDiffusionError):
    """Raised when benchmarking operations fail."""

    pass


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking."""

    # Test parameters
    image_sizes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    )
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    num_warmup_runs: int = 5
    num_benchmark_runs: int = 10

    # Memory testing
    test_memory_usage: bool = True
    memory_growth_threshold: float = 1.1  # 10% growth threshold

    # Throughput testing
    test_throughput: bool = True
    throughput_duration: float = 30.0  # seconds

    # Scalability testing
    test_scalability: bool = True
    max_test_memory_gb: float = 8.0

    # Output settings
    save_results: bool = True
    results_dir: str = "benchmark_results"
    plot_results: bool = True


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    # Test configuration
    test_name: str
    image_size: Tuple[int, int]
    batch_size: int
    device: str

    # Timing results
    inference_time: float
    throughput: float  # samples per second

    # Memory results
    peak_memory_mb: float
    memory_efficiency: float  # samples per MB

    # Additional metrics
    gpu_utilization: Optional[float] = None
    cpu_utilization: Optional[float] = None

    # Metadata
    timestamp: str = ""
    notes: str = ""


class InferenceBenchmark:
    """Benchmark inference performance."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Results storage
        self.results = []

        logger.info(f"Initialized InferenceBenchmark on {self.device}")

    def benchmark_model(
        self,
        model: nn.Module,
        test_name: str = "model_inference",
        guidance: Optional[OptimizedPoissonGuidance] = None,
    ) -> List[BenchmarkResult]:
        """Benchmark model inference across different configurations."""
        logger.info(f"Benchmarking {test_name}...")

        model.eval()
        results = []

        for image_size in self.config.image_sizes:
            for batch_size in self.config.batch_sizes:
                try:
                    result = self._benchmark_single_config(
                        model, test_name, image_size, batch_size, guidance
                    )
                    results.append(result)

                    logger.info(
                        f"{test_name} - {image_size}x{batch_size}: "
                        f"{result.inference_time:.3f}s, {result.throughput:.1f} samples/s, "
                        f"{result.peak_memory_mb:.1f}MB"
                    )

                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(
                            f"OOM at {image_size}x{batch_size}, skipping larger configs"
                        )
                        break
                    else:
                        raise e

        self.results.extend(results)
        return results

    def _benchmark_single_config(
        self,
        model: nn.Module,
        test_name: str,
        image_size: Tuple[int, int],
        batch_size: int,
        guidance: Optional[OptimizedPoissonGuidance] = None,
    ) -> BenchmarkResult:
        """Benchmark single configuration."""
        height, width = image_size

        # Create test data
        test_batch = self._create_test_batch(batch_size, height, width)

        # Warmup runs
        for _ in range(self.config.num_warmup_runs):
            with torch.no_grad():
                _ = model(test_batch["x"])

        # Clear cache and sync
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Benchmark runs
        times = []
        memory_usage = []

        for _ in range(self.config.num_benchmark_runs):
            # Record start state
            start_memory = (
                torch.cuda.memory_allocated() if self.device.type == "cuda" else 0
            )

            # Time inference
            start_time = time.time()

            with torch.no_grad():
                outputs = model(test_batch["x"])

                # Add guidance if provided
                if guidance is not None:
                    guidance_grad = guidance(
                        outputs
                        if isinstance(outputs, torch.Tensor)
                        else outputs.get("prediction", outputs),
                        test_batch["y"],
                        test_batch["scale"],
                        test_batch["background"],
                        test_batch["read_noise"],
                    )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()

            # Record end state
            end_memory = (
                torch.cuda.memory_allocated() if self.device.type == "cuda" else 0
            )

            # Store results
            times.append(end_time - start_time)
            memory_usage.append((end_memory - start_memory) / 1024**2)  # MB

        # Compute statistics
        avg_time = np.mean(times)
        peak_memory = max(memory_usage) if memory_usage else 0
        throughput = batch_size / avg_time
        memory_efficiency = batch_size / peak_memory if peak_memory > 0 else 0

        return BenchmarkResult(
            test_name=test_name,
            image_size=image_size,
            batch_size=batch_size,
            device=str(self.device),
            inference_time=avg_time,
            throughput=throughput,
            peak_memory_mb=peak_memory,
            memory_efficiency=memory_efficiency,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _create_test_batch(
        self, batch_size: int, height: int, width: int
    ) -> Dict[str, torch.Tensor]:
        """Create test batch for benchmarking."""
        return {
            "x": torch.randn(batch_size, 3, height, width, device=self.device),
            "y": torch.randn(batch_size, 3, height, width, device=self.device) * 1000
            + 100,
            "scale": torch.tensor(1000.0, device=self.device),
            "background": torch.tensor(100.0, device=self.device),
            "read_noise": torch.tensor(5.0, device=self.device),
        }


class MemoryBenchmark:
    """Benchmark memory usage and efficiency."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initialized MemoryBenchmark on {self.device}")

    def benchmark_memory_scaling(
        self, model: nn.Module, test_name: str = "memory_scaling"
    ) -> Dict[str, List[float]]:
        """Benchmark memory usage scaling with image size and batch size."""
        logger.info(f"Benchmarking memory scaling for {test_name}...")

        results = {
            "image_sizes": [],
            "batch_sizes": [],
            "memory_usage": [],
            "memory_efficiency": [],
        }

        model.eval()

        for image_size in self.config.image_sizes:
            height, width = image_size
            pixel_count = height * width

            for batch_size in self.config.batch_sizes:
                try:
                    # Create test data
                    test_input = torch.randn(
                        batch_size, 3, height, width, device=self.device
                    )

                    # Clear cache
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()

                    # Run inference
                    with torch.no_grad():
                        _ = model(test_input)

                    # Measure memory
                    if self.device.type == "cuda":
                        peak_memory = (
                            torch.cuda.max_memory_allocated() / 1024**2
                        )  # MB
                    else:
                        peak_memory = 0  # CPU memory tracking is more complex

                    # Calculate efficiency
                    total_pixels = batch_size * pixel_count
                    memory_efficiency = (
                        total_pixels / peak_memory if peak_memory > 0 else 0
                    )

                    # Store results
                    results["image_sizes"].append(pixel_count)
                    results["batch_sizes"].append(batch_size)
                    results["memory_usage"].append(peak_memory)
                    results["memory_efficiency"].append(memory_efficiency)

                    logger.debug(
                        f"Size {image_size}, Batch {batch_size}: "
                        f"{peak_memory:.1f}MB, {memory_efficiency:.1f} pixels/MB"
                    )

                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    logger.warning(f"OOM at {image_size}x{batch_size}")
                    break

        return results

    def test_memory_leaks(
        self,
        model: nn.Module,
        num_iterations: int = 100,
        test_name: str = "memory_leak_test",
    ) -> Dict[str, Any]:
        """Test for memory leaks during repeated inference."""
        logger.info(f"Testing memory leaks for {test_name}...")

        model.eval()

        # Create test data
        test_input = torch.randn(1, 3, 512, 512, device=self.device)

        memory_history = []

        for i in range(num_iterations):
            # Run inference
            with torch.no_grad():
                _ = model(test_input)

            # Record memory usage
            if self.device.type == "cuda":
                memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                memory_usage = psutil.Process().memory_info().rss / 1024**2  # MB

            memory_history.append(memory_usage)

            # Periodic cleanup
            if i % 10 == 0:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Analyze memory growth
        initial_memory = np.mean(memory_history[:10])
        final_memory = np.mean(memory_history[-10:])
        memory_growth = final_memory / initial_memory if initial_memory > 0 else 1.0

        has_leak = memory_growth > self.config.memory_growth_threshold

        results = {
            "has_memory_leak": has_leak,
            "memory_growth_ratio": memory_growth,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_history": memory_history,
        }

        if has_leak:
            logger.warning(
                f"Potential memory leak detected: {memory_growth:.2f}x growth"
            )
        else:
            logger.info(f"No memory leak detected: {memory_growth:.2f}x growth")

        return results


class ThroughputBenchmark:
    """Benchmark throughput and sustained performance."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initialized ThroughputBenchmark on {self.device}")

    def benchmark_sustained_throughput(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (512, 512),
        batch_size: int = 4,
        test_name: str = "sustained_throughput",
    ) -> Dict[str, Any]:
        """Benchmark sustained throughput over time."""
        logger.info(f"Benchmarking sustained throughput for {test_name}...")

        model.eval()
        height, width = image_size

        # Create test data
        test_input = torch.randn(batch_size, 3, height, width, device=self.device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(test_input)

        # Benchmark
        start_time = time.time()
        samples_processed = 0
        throughput_history = []

        while time.time() - start_time < self.config.throughput_duration:
            batch_start = time.time()

            with torch.no_grad():
                _ = model(test_input)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            batch_end = time.time()

            # Calculate instantaneous throughput
            batch_time = batch_end - batch_start
            batch_throughput = batch_size / batch_time
            throughput_history.append(batch_throughput)

            samples_processed += batch_size

        total_time = time.time() - start_time
        avg_throughput = samples_processed / total_time

        results = {
            "avg_throughput": avg_throughput,
            "peak_throughput": max(throughput_history),
            "min_throughput": min(throughput_history),
            "throughput_std": np.std(throughput_history),
            "total_samples": samples_processed,
            "total_time": total_time,
            "throughput_history": throughput_history,
        }

        logger.info(
            f"Sustained throughput: {avg_throughput:.1f} samples/s "
            f"(peak: {results['peak_throughput']:.1f}, std: {results['throughput_std']:.1f})"
        )

        return results


class ComprehensiveBenchmark:
    """Comprehensive benchmarking suite."""

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()

        # Initialize benchmark components
        self.inference_benchmark = InferenceBenchmark(self.config)
        self.memory_benchmark = MemoryBenchmark(self.config)
        self.throughput_benchmark = ThroughputBenchmark(self.config)

        # Results storage
        self.all_results = {}

        # Create results directory
        if self.config.save_results:
            Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Initialized ComprehensiveBenchmark")

    def run_full_benchmark(
        self,
        model: nn.Module,
        test_name: str = "full_benchmark",
        guidance: Optional[OptimizedPoissonGuidance] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        logger.info(f"Running full benchmark suite for {test_name}...")

        results = {}

        # 1. Inference benchmarking
        logger.info("Running inference benchmarks...")
        inference_results = self.inference_benchmark.benchmark_model(
            model, f"{test_name}_inference", guidance
        )
        results["inference"] = [
            {
                "image_size": r.image_size,
                "batch_size": r.batch_size,
                "inference_time": r.inference_time,
                "throughput": r.throughput,
                "peak_memory_mb": r.peak_memory_mb,
                "memory_efficiency": r.memory_efficiency,
            }
            for r in inference_results
        ]

        # 2. Memory benchmarking
        if self.config.test_memory_usage:
            logger.info("Running memory benchmarks...")
            memory_scaling = self.memory_benchmark.benchmark_memory_scaling(
                model, f"{test_name}_memory"
            )
            memory_leak_test = self.memory_benchmark.test_memory_leaks(
                model, test_name=f"{test_name}_leak"
            )

            results["memory"] = {
                "scaling": memory_scaling,
                "leak_test": memory_leak_test,
            }

        # 3. Throughput benchmarking
        if self.config.test_throughput:
            logger.info("Running throughput benchmarks...")
            throughput_results = (
                self.throughput_benchmark.benchmark_sustained_throughput(
                    model, test_name=f"{test_name}_throughput"
                )
            )
            results["throughput"] = throughput_results

        # Store results
        self.all_results[test_name] = results

        # Save results
        if self.config.save_results:
            self._save_results(test_name, results)

        # Generate plots
        if self.config.plot_results:
            self._plot_results(test_name, results)

        logger.info(f"Benchmark suite completed for {test_name}")
        return results

    def _save_results(self, test_name: str, results: Dict[str, Any]):
        """Save benchmark results to file."""
        results_file = Path(self.config.results_dir) / f"{test_name}_results.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

    def _make_json_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    def _plot_results(self, test_name: str, results: Dict[str, Any]):
        """Generate performance plots."""
        try:
            import matplotlib.pyplot as plt

            # Create plots directory
            plots_dir = Path(self.config.results_dir) / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Plot inference performance
            if "inference" in results:
                self._plot_inference_results(test_name, results["inference"], plots_dir)

            # Plot memory usage
            if "memory" in results:
                self._plot_memory_results(test_name, results["memory"], plots_dir)

            # Plot throughput
            if "throughput" in results:
                self._plot_throughput_results(
                    test_name, results["throughput"], plots_dir
                )

            logger.info(f"Plots saved to {plots_dir}")

        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")

    def _plot_inference_results(
        self, test_name: str, inference_results: List[Dict], plots_dir: Path
    ):
        """Plot inference performance results."""
        # Group by image size
        size_groups = defaultdict(list)
        for result in inference_results:
            size_key = f"{result['image_size'][0]}x{result['image_size'][1]}"
            size_groups[size_key].append(result)

        # Plot throughput vs batch size
        plt.figure(figsize=(12, 8))

        for size_key, results in size_groups.items():
            batch_sizes = [r["batch_size"] for r in results]
            throughputs = [r["throughput"] for r in results]
            plt.plot(batch_sizes, throughputs, marker="o", label=size_key)

        plt.xlabel("Batch Size")
        plt.ylabel("Throughput (samples/s)")
        plt.title(f"Inference Throughput - {test_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            plots_dir / f"{test_name}_throughput.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Plot memory efficiency
        plt.figure(figsize=(12, 8))

        for size_key, results in size_groups.items():
            batch_sizes = [r["batch_size"] for r in results]
            memory_effs = [r["memory_efficiency"] for r in results]
            plt.plot(batch_sizes, memory_effs, marker="s", label=size_key)

        plt.xlabel("Batch Size")
        plt.ylabel("Memory Efficiency (samples/MB)")
        plt.title(f"Memory Efficiency - {test_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            plots_dir / f"{test_name}_memory_efficiency.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_memory_results(
        self, test_name: str, memory_results: Dict, plots_dir: Path
    ):
        """Plot memory usage results."""
        if "scaling" in memory_results:
            scaling = memory_results["scaling"]

            plt.figure(figsize=(12, 8))

            # Plot memory usage vs image size
            unique_sizes = sorted(set(scaling["image_sizes"]))
            for batch_size in sorted(set(scaling["batch_sizes"])):
                sizes = []
                memories = []

                for i, bs in enumerate(scaling["batch_sizes"]):
                    if bs == batch_size:
                        sizes.append(scaling["image_sizes"][i])
                        memories.append(scaling["memory_usage"][i])

                if sizes:
                    plt.plot(sizes, memories, marker="o", label=f"Batch {batch_size}")

            plt.xlabel("Image Size (pixels)")
            plt.ylabel("Memory Usage (MB)")
            plt.title(f"Memory Scaling - {test_name}")
            plt.legend()
            plt.grid(True)
            plt.xscale("log")
            plt.yscale("log")
            plt.savefig(
                plots_dir / f"{test_name}_memory_scaling.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _plot_throughput_results(
        self, test_name: str, throughput_results: Dict, plots_dir: Path
    ):
        """Plot throughput results."""
        if "throughput_history" in throughput_results:
            history = throughput_results["throughput_history"]

            plt.figure(figsize=(12, 6))

            time_points = np.arange(len(history)) * (
                self.config.throughput_duration / len(history)
            )
            plt.plot(time_points, history, alpha=0.7)
            plt.axhline(
                y=throughput_results["avg_throughput"],
                color="r",
                linestyle="--",
                label=f"Average: {throughput_results['avg_throughput']:.1f}",
            )

            plt.xlabel("Time (seconds)")
            plt.ylabel("Throughput (samples/s)")
            plt.title(f"Sustained Throughput - {test_name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                plots_dir / f"{test_name}_sustained_throughput.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def compare_configurations(
        self,
        models_and_configs: List[
            Tuple[nn.Module, str, Optional[OptimizedPoissonGuidance]]
        ],
        comparison_name: str = "configuration_comparison",
    ) -> Dict[str, Any]:
        """Compare multiple model configurations."""
        logger.info(f"Running configuration comparison: {comparison_name}")

        comparison_results = {}

        for model, config_name, guidance in models_and_configs:
            logger.info(f"Benchmarking configuration: {config_name}")
            results = self.run_full_benchmark(model, config_name, guidance)
            comparison_results[config_name] = results

        # Generate comparison plots
        if self.config.plot_results:
            self._plot_comparison(comparison_name, comparison_results)

        return comparison_results

    def _plot_comparison(
        self, comparison_name: str, comparison_results: Dict[str, Any]
    ):
        """Plot comparison between configurations."""
        try:
            plots_dir = Path(self.config.results_dir) / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Compare throughput
            plt.figure(figsize=(12, 8))

            for config_name, results in comparison_results.items():
                if "inference" in results:
                    inference_results = results["inference"]
                    # Get results for 512x512 images
                    filtered_results = [
                        r for r in inference_results if r["image_size"] == (512, 512)
                    ]

                    if filtered_results:
                        batch_sizes = [r["batch_size"] for r in filtered_results]
                        throughputs = [r["throughput"] for r in filtered_results]
                        plt.plot(
                            batch_sizes, throughputs, marker="o", label=config_name
                        )

            plt.xlabel("Batch Size")
            plt.ylabel("Throughput (samples/s)")
            plt.title(f"Configuration Comparison - {comparison_name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                plots_dir / f"{comparison_name}_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        except Exception as e:
            logger.error(f"Error generating comparison plots: {e}")

    def get_performance_summary(self, test_name: str) -> Dict[str, Any]:
        """Get performance summary for a test."""
        if test_name not in self.all_results:
            raise ValueError(f"No results found for test: {test_name}")

        results = self.all_results[test_name]
        summary = {}

        # Inference summary
        if "inference" in results:
            inference_results = results["inference"]
            if inference_results:
                throughputs = [r["throughput"] for r in inference_results]
                memory_usage = [r["peak_memory_mb"] for r in inference_results]

                summary["inference"] = {
                    "max_throughput": max(throughputs),
                    "avg_throughput": np.mean(throughputs),
                    "min_memory_usage": min(memory_usage),
                    "max_memory_usage": max(memory_usage),
                }

        # Memory summary
        if "memory" in results:
            memory_results = results["memory"]
            summary["memory"] = {
                "has_memory_leak": memory_results.get("leak_test", {}).get(
                    "has_memory_leak", False
                ),
                "memory_growth_ratio": memory_results.get("leak_test", {}).get(
                    "memory_growth_ratio", 1.0
                ),
            }

        # Throughput summary
        if "throughput" in results:
            throughput_results = results["throughput"]
            summary["throughput"] = {
                "sustained_throughput": throughput_results["avg_throughput"],
                "throughput_stability": 1.0
                - (
                    throughput_results["throughput_std"]
                    / throughput_results["avg_throughput"]
                ),
            }

        return summary
