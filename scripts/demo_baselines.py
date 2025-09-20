#!/usr/bin/env python
"""
Demonstration of baseline comparison methods.

This script provides a quick demonstration of the baseline comparison
framework with synthetic data and visualization of results.

Usage:
    python scripts/demo_baselines.py
    python scripts/demo_baselines.py --domain microscopy --num-samples 20
    python scripts/demo_baselines.py --methods BM3D Gaussian DnCNN
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.baselines import BaselineComparator, create_baseline_suite
from core.calibration import CalibrationParams, SensorCalibration
from core.logging_config import setup_project_logging

logger = setup_project_logging(level="INFO")


class BaselineDemo:
    """
    Interactive demonstration of baseline comparison methods.

    This class provides a simple interface to demonstrate the capabilities
    of the baseline comparison framework with synthetic data.
    """

    def __init__(self, device: str = "auto"):
        """Initialize baseline demo."""
        self.device = self._setup_device(device)
        self.baseline_suite = create_baseline_suite(device=self.device)

        logger.info(f"Initialized baseline demo on {self.device}")
        logger.info(
            f"Available methods: {list(self.baseline_suite.available_baselines.keys())}"
        )

    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = "cpu"

        return device

    def create_synthetic_sample(
        self,
        domain: str = "microscopy",
        size: tuple = (128, 128),
        scale: float = 1000.0,
        background: float = 100.0,
        read_noise: float = 5.0,
    ) -> tuple:
        """
        Create a synthetic noisy/clean image pair.

        Args:
            domain: Domain type ('photography', 'microscopy', 'astronomy')
            size: Image size (height, width)
            scale: Normalization scale (electrons)
            background: Background level (electrons)
            read_noise: Read noise standard deviation (electrons)

        Returns:
            (noisy_electrons, clean_normalized, metadata) tuple
        """
        height, width = size

        # Generate domain-appropriate clean image
        if domain == "photography":
            # Smooth natural image structure
            clean = torch.rand(1, 1, height, width)
            clean = torch.nn.functional.avg_pool2d(clean, 5, stride=1, padding=2)
            clean = clean * 0.7 + 0.2  # Scale to [0.2, 0.9]

        elif domain == "microscopy":
            # Point sources (fluorescent spots)
            clean = torch.zeros(1, 1, height, width)
            num_spots = np.random.randint(10, 30)

            for _ in range(num_spots):
                y = np.random.randint(10, height - 10)
                x = np.random.randint(10, width - 10)
                intensity = np.random.uniform(0.3, 1.0)
                sigma = np.random.uniform(1.5, 3.0)

                # Create Gaussian spot
                yy, xx = torch.meshgrid(
                    torch.arange(height, dtype=torch.float32),
                    torch.arange(width, dtype=torch.float32),
                    indexing="ij",
                )
                spot = intensity * torch.exp(
                    -((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma**2)
                )
                clean[0, 0] += spot

            clean = torch.clamp(clean, 0, 1)

        elif domain == "astronomy":
            # Point sources (stars) with background
            clean = torch.ones(1, 1, height, width) * 0.1  # Background
            num_stars = np.random.randint(20, 50)

            for _ in range(num_stars):
                y = np.random.randint(5, height - 5)
                x = np.random.randint(5, width - 5)
                intensity = np.random.uniform(0.2, 1.0)

                # Create point source
                yy, xx = torch.meshgrid(
                    torch.arange(height, dtype=torch.float32),
                    torch.arange(width, dtype=torch.float32),
                    indexing="ij",
                )
                star = intensity * torch.exp(
                    -((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.0**2)
                )
                clean[0, 0] += star

            clean = torch.clamp(clean, 0, 1)

        else:
            # Generic random image
            clean = torch.rand(1, 1, height, width) * 0.8 + 0.1

        # Generate noisy observation
        clean_electrons = clean * scale + background

        # Add Poisson noise
        poisson_component = torch.poisson(clean_electrons)

        # Add Gaussian read noise
        gaussian_component = torch.randn_like(clean_electrons) * read_noise

        # Combined noisy observation
        noisy_electrons = poisson_component + gaussian_component

        metadata = {
            "domain": domain,
            "scale": scale,
            "background": background,
            "read_noise": read_noise,
            "size": size,
            "snr_db": 10 * np.log10(clean_electrons.mean().item() / read_noise),
        }

        return noisy_electrons, clean, metadata

    def run_single_comparison(
        self,
        domain: str = "microscopy",
        methods: list = None,
        size: tuple = (128, 128),
        show_plots: bool = True,
    ) -> dict:
        """
        Run baseline comparison on a single synthetic sample.

        Args:
            domain: Domain type
            methods: List of methods to compare (None for all available)
            size: Image size
            show_plots: Whether to display result plots

        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Running single comparison for {domain} domain")

        # Create synthetic sample
        noisy, clean, metadata = self.create_synthetic_sample(domain, size)
        noisy = noisy.to(self.device)
        clean = clean.to(self.device)

        logger.info(
            f"Created synthetic sample: {size}, SNR: {metadata['snr_db']:.1f} dB"
        )

        # Filter methods if specified
        if methods:
            available_methods = {
                name: method
                for name, method in self.baseline_suite.available_baselines.items()
                if name in methods
            }
            if not available_methods:
                logger.error(f"None of the specified methods {methods} are available")
                return {}
        else:
            available_methods = self.baseline_suite.available_baselines

        # Run baseline comparison
        results = {}
        processing_times = {}

        for method_name, method in available_methods.items():
            logger.info(f"Running {method_name}...")

            try:
                import time

                start_time = time.time()

                # Run denoising
                denoised = method.denoise(
                    noisy=noisy,
                    scale=metadata["scale"],
                    background=metadata["background"],
                    read_noise=metadata["read_noise"],
                )

                processing_time = time.time() - start_time
                processing_times[method_name] = processing_time

                # Compute metrics
                psnr = self._compute_psnr(denoised, clean)
                ssim = self._compute_ssim(denoised, clean)

                results[method_name] = {
                    "denoised": denoised.cpu(),
                    "psnr": psnr,
                    "ssim": ssim,
                    "processing_time": processing_time,
                }

                logger.info(
                    f"  {method_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.3f}, Time={processing_time:.3f}s"
                )

            except Exception as e:
                logger.error(f"Failed to run {method_name}: {e}")
                continue

        # Add input data to results
        results["_input_data"] = {
            "noisy": noisy.cpu(),
            "clean": clean.cpu(),
            "metadata": metadata,
        }

        # Create visualization if requested
        if show_plots and results:
            self._create_comparison_plot(results)

        return results

    def run_noise_level_analysis(
        self,
        domain: str = "microscopy",
        methods: list = None,
        noise_levels: list = None,
        num_samples: int = 5,
    ) -> dict:
        """
        Analyze method performance across different noise levels.

        Args:
            domain: Domain type
            methods: List of methods to compare
            noise_levels: List of read noise levels to test
            num_samples: Number of samples per noise level

        Returns:
            Analysis results
        """
        if noise_levels is None:
            noise_levels = [1.0, 2.0, 5.0, 10.0, 20.0]

        if methods is None:
            methods = ["Gaussian", "Richardson-Lucy", "DnCNN"]  # Fast methods for demo

        logger.info(f"Running noise level analysis for {domain}")
        logger.info(f"Noise levels: {noise_levels}")
        logger.info(f"Methods: {methods}")

        results = {
            method: {
                "noise_levels": [],
                "psnr_means": [],
                "psnr_stds": [],
                "ssim_means": [],
                "ssim_stds": [],
            }
            for method in methods
        }

        for noise_level in noise_levels:
            logger.info(f"Testing noise level: {noise_level}")

            # Collect results for this noise level
            method_results = {method: {"psnr": [], "ssim": []} for method in methods}

            for sample_idx in range(num_samples):
                # Create sample with specific noise level
                noisy, clean, metadata = self.create_synthetic_sample(
                    domain=domain, read_noise=noise_level
                )
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                # Test each method
                for method_name in methods:
                    if method_name not in self.baseline_suite.available_baselines:
                        continue

                    try:
                        method = self.baseline_suite.available_baselines[method_name]
                        denoised = method.denoise(
                            noisy=noisy,
                            scale=metadata["scale"],
                            background=metadata["background"],
                            read_noise=metadata["read_noise"],
                        )

                        psnr = self._compute_psnr(denoised, clean)
                        ssim = self._compute_ssim(denoised, clean)

                        method_results[method_name]["psnr"].append(psnr)
                        method_results[method_name]["ssim"].append(ssim)

                    except Exception as e:
                        logger.warning(
                            f"Failed {method_name} at noise {noise_level}: {e}"
                        )

            # Compute statistics for this noise level
            for method_name in methods:
                if (
                    method_name in method_results
                    and method_results[method_name]["psnr"]
                ):
                    psnr_values = method_results[method_name]["psnr"]
                    ssim_values = method_results[method_name]["ssim"]

                    results[method_name]["noise_levels"].append(noise_level)
                    results[method_name]["psnr_means"].append(np.mean(psnr_values))
                    results[method_name]["psnr_stds"].append(np.std(psnr_values))
                    results[method_name]["ssim_means"].append(np.mean(ssim_values))
                    results[method_name]["ssim_stds"].append(np.std(ssim_values))

        # Create analysis plot
        self._create_noise_analysis_plot(results, domain)

        return results

    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR between prediction and target."""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute SSIM between prediction and target (simplified)."""
        # Simplified SSIM computation
        mu1 = torch.mean(pred)
        mu2 = torch.mean(target)
        sigma1_sq = torch.var(pred)
        sigma2_sq = torch.var(target)
        sigma12 = torch.mean((pred - mu1) * (target - mu2))

        c1 = 0.01**2
        c2 = 0.03**2

        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
        )

        return ssim.item()

    def _create_comparison_plot(self, results: dict):
        """Create comparison visualization."""
        # Filter out metadata
        method_results = {k: v for k, v in results.items() if not k.startswith("_")}
        input_data = results["_input_data"]

        num_methods = len(method_results)
        fig, axes = plt.subplots(2, num_methods + 2, figsize=(4 * (num_methods + 2), 8))

        if axes.ndim == 1:
            axes = axes.reshape(2, -1)

        # Show input images
        noisy_display = input_data["noisy"][0, 0].numpy()
        clean_display = input_data["clean"][0, 0].numpy()

        axes[0, 0].imshow(noisy_display, cmap="gray")
        axes[0, 0].set_title("Noisy Input")
        axes[0, 0].axis("off")

        axes[1, 0].imshow(clean_display, cmap="gray")
        axes[1, 0].set_title("Clean Target")
        axes[1, 0].axis("off")

        # Show method results
        for i, (method_name, result) in enumerate(method_results.items()):
            denoised_display = result["denoised"][0, 0].numpy()

            axes[0, i + 1].imshow(denoised_display, cmap="gray")
            axes[0, i + 1].set_title(f'{method_name}\nPSNR: {result["psnr"]:.2f} dB')
            axes[0, i + 1].axis("off")

            # Show difference image
            diff = np.abs(denoised_display - clean_display)
            axes[1, i + 1].imshow(diff, cmap="hot")
            axes[1, i + 1].set_title(f'Error\nSSIM: {result["ssim"]:.3f}')
            axes[1, i + 1].axis("off")

        # Performance comparison
        if num_methods > 1:
            methods = list(method_results.keys())
            psnr_values = [method_results[m]["psnr"] for m in methods]
            ssim_values = [method_results[m]["ssim"] for m in methods]
            times = [method_results[m]["processing_time"] for m in methods]

            # PSNR comparison
            axes[0, -1].bar(range(len(methods)), psnr_values)
            axes[0, -1].set_title("PSNR Comparison")
            axes[0, -1].set_ylabel("PSNR (dB)")
            axes[0, -1].set_xticks(range(len(methods)))
            axes[0, -1].set_xticklabels(methods, rotation=45)

            # Processing time comparison
            axes[1, -1].bar(range(len(methods)), times)
            axes[1, -1].set_title("Processing Time")
            axes[1, -1].set_ylabel("Time (s)")
            axes[1, -1].set_xticks(range(len(methods)))
            axes[1, -1].set_xticklabels(methods, rotation=45)

        plt.tight_layout()
        plt.show()

    def _create_noise_analysis_plot(self, results: dict, domain: str):
        """Create noise level analysis plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # PSNR vs noise level
        for method_name, method_results in results.items():
            if method_results["noise_levels"]:
                ax1.errorbar(
                    method_results["noise_levels"],
                    method_results["psnr_means"],
                    yerr=method_results["psnr_stds"],
                    label=method_name,
                    marker="o",
                    capsize=5,
                )

        ax1.set_xlabel("Read Noise Level (electrons)")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_title(f"PSNR vs Noise Level ({domain})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        # SSIM vs noise level
        for method_name, method_results in results.items():
            if method_results["noise_levels"]:
                ax2.errorbar(
                    method_results["noise_levels"],
                    method_results["ssim_means"],
                    yerr=method_results["ssim_stds"],
                    label=method_name,
                    marker="s",
                    capsize=5,
                )

        ax2.set_xlabel("Read Noise Level (electrons)")
        ax2.set_ylabel("SSIM")
        ax2.set_title(f"SSIM vs Noise Level ({domain})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")

        plt.tight_layout()
        plt.show()


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Baseline comparison demonstration")
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        default="microscopy",
        choices=["photography", "microscopy", "astronomy"],
        help="Domain for demonstration",
    )
    parser.add_argument(
        "--methods",
        "-m",
        nargs="+",
        help="Specific methods to compare (default: all available)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples for noise analysis",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[128, 128],
        help="Image size (height width)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for computation",
    )
    parser.add_argument(
        "--demo-type",
        type=str,
        default="single",
        choices=["single", "noise-analysis"],
        help="Type of demonstration",
    )

    args = parser.parse_args()

    # Initialize demo
    demo = BaselineDemo(device=args.device)

    print(f"\n{'='*60}")
    print("BASELINE COMPARISON DEMONSTRATION")
    print(f"{'='*60}")
    print(f"Domain: {args.domain}")
    print(f"Device: {demo.device}")
    print(f"Available methods: {list(demo.baseline_suite.available_baselines.keys())}")
    print(f"{'='*60}\n")

    if args.demo_type == "single":
        print("Running single sample comparison...")
        results = demo.run_single_comparison(
            domain=args.domain,
            methods=args.methods,
            size=tuple(args.size),
            show_plots=True,
        )

        if results:
            print("\nResults Summary:")
            method_results = {k: v for k, v in results.items() if not k.startswith("_")}
            for method_name, result in method_results.items():
                print(
                    f"  {method_name:15}: PSNR={result['psnr']:6.2f} dB, "
                    f"SSIM={result['ssim']:5.3f}, Time={result['processing_time']:6.3f}s"
                )

    elif args.demo_type == "noise-analysis":
        print("Running noise level analysis...")

        # Use fast methods for demo
        fast_methods = []
        available = list(demo.baseline_suite.available_baselines.keys())

        # Prioritize fast methods
        preferred_methods = ["Gaussian", "Richardson-Lucy", "DnCNN", "Wiener"]
        for method in preferred_methods:
            if method in available:
                fast_methods.append(method)

        # Add others if we don't have enough
        for method in available:
            if method not in fast_methods and len(fast_methods) < 4:
                fast_methods.append(method)

        if args.methods:
            fast_methods = [m for m in args.methods if m in available]

        print(f"Testing methods: {fast_methods}")

        results = demo.run_noise_level_analysis(
            domain=args.domain, methods=fast_methods, num_samples=args.num_samples
        )

        print("\nNoise Analysis Complete!")
        print("Check the generated plots for detailed results.")

    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    print("\nKey Features Demonstrated:")
    print("✓ Multiple baseline method implementations")
    print("✓ Automatic method availability detection")
    print("✓ Comprehensive performance evaluation")
    print("✓ Domain-specific synthetic data generation")
    print("✓ Real-time visualization and comparison")
    print("✓ Statistical analysis across noise levels")


if __name__ == "__main__":
    main()
