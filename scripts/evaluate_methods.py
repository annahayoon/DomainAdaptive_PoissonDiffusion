#!/usr/bin/env python
"""
Comprehensive evaluation script for Poisson-Gaussian diffusion restoration.

This script demonstrates how to use the evaluation framework to:
1. Evaluate our method against baselines
2. Generate comprehensive reports
3. Perform statistical analysis
4. Create visualizations

Usage:
    python scripts/evaluate_methods.py --config configs/evaluation.yaml
    python scripts/evaluate_methods.py --method our_method --dataset test_data/
    python scripts/evaluate_methods.py --compare-baselines --output results/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.baselines import BaselineComparator, create_baseline_suite
from core.metrics import EvaluationReport, EvaluationSuite, save_evaluation_results

# from core.transforms import ReversibleTransform  # TODO: Implement in Phase 1.3
from scripts.generate_synthetic_data import SyntheticDataGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Main class for running comprehensive evaluations."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluation runner.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize evaluation components
        self.evaluation_suite = EvaluationSuite(device=self.device)
        self.baseline_comparator = create_baseline_suite(device=self.device)

        # Results storage
        self.results = {}

        logger.info(f"EvaluationRunner initialized on device: {self.device}")

    def load_test_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Load test data from directory or file.

        Args:
            data_path: Path to test data

        Returns:
            List of test data dictionaries
        """
        data_path = Path(data_path)
        test_data = []

        if data_path.is_file() and data_path.suffix == ".json":
            # Load from JSON file
            with open(data_path, "r") as f:
                data_config = json.load(f)

            # Generate synthetic data based on config
            from scripts.generate_synthetic_data import SyntheticConfig

            synthetic_config = SyntheticConfig(
                output_dir="temp_synthetic",
                num_images=1,
                image_size=max(item["height"], item["width"]),
                photon_levels=[item["scale"]],
                read_noise_levels=[item.get("read_noise", 0.0)],
                background_level=item.get("background", 0.0),
            )
            generator = SyntheticDataGenerator(synthetic_config)

            for item in data_config["test_cases"]:
                clean_img = torch.rand(1, 1, item["height"], item["width"])
                noisy_img, metadata = generator.add_poisson_gaussian_noise(
                    clean_img.numpy().squeeze(),
                    photon_level=item["scale"],
                    background=item.get("background", 0.0),
                    read_noise=item.get("read_noise", 0.0),
                )

                test_data.append(
                    {
                        "clean": clean_img,
                        "noisy": torch.from_numpy(noisy_img)
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0),
                        "scale": item["scale"],
                        "background": item.get("background", 0.0),
                        "read_noise": item.get("read_noise", 0.0),
                        "domain": item.get("domain", "photography"),
                        "dataset_name": item.get("name", f"test_{len(test_data)}"),
                        "metadata": metadata,
                    }
                )

        elif data_path.is_dir():
            # Load from directory (placeholder - would implement actual loading)
            logger.warning("Directory loading not implemented, using synthetic data")
            test_data = self._generate_synthetic_test_data()

        else:
            # Generate default synthetic test data
            logger.info("Generating synthetic test data")
            test_data = self._generate_synthetic_test_data()

        logger.info(f"Loaded {len(test_data)} test cases")
        return test_data

    def _generate_synthetic_test_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic test data for evaluation."""
        from scripts.generate_synthetic_data import SyntheticConfig

        synthetic_config = SyntheticConfig(
            output_dir="temp_synthetic",
            num_images=1,
            image_size=128,
            photon_levels=[100, 1000, 5000],
            read_noise_levels=[2, 5, 10],
            background_level=5.0,
        )
        generator = SyntheticDataGenerator(synthetic_config)
        test_data = []

        # Test cases with different noise levels and domains
        test_configs = [
            # Photography - high photon count
            {
                "scale": 5000,
                "read_noise": 10,
                "domain": "photography",
                "size": (128, 128),
            },
            {
                "scale": 2000,
                "read_noise": 15,
                "domain": "photography",
                "size": (128, 128),
            },
            # Microscopy - medium photon count
            {
                "scale": 1000,
                "read_noise": 5,
                "domain": "microscopy",
                "size": (128, 128),
            },
            {"scale": 500, "read_noise": 8, "domain": "microscopy", "size": (128, 128)},
            # Astronomy - low photon count
            {"scale": 100, "read_noise": 3, "domain": "astronomy", "size": (128, 128)},
            {"scale": 50, "read_noise": 2, "domain": "astronomy", "size": (128, 128)},
        ]

        for i, config in enumerate(test_configs):
            # Generate clean image with domain-appropriate features
            clean_img = self._generate_domain_image(config["domain"], config["size"])

            # Add noise
            noisy_img, metadata = generator.add_poisson_gaussian_noise(
                clean_img.numpy().squeeze(),
                photon_level=config["scale"],
                background=5.0,
                read_noise=config["read_noise"],
            )

            test_data.append(
                {
                    "clean": clean_img,
                    "noisy": torch.from_numpy(noisy_img)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0),
                    "scale": config["scale"],
                    "background": 5.0,
                    "read_noise": config["read_noise"],
                    "domain": config["domain"],
                    "dataset_name": f"synthetic_{config['domain']}_{i}",
                    "metadata": metadata,
                }
            )

        return test_data

    def _generate_domain_image(
        self, domain: str, size: Tuple[int, int]
    ) -> torch.Tensor:
        """Generate synthetic image appropriate for domain."""
        h, w = size
        image = torch.zeros(1, 1, h, w)

        if domain == "photography":
            # Natural scene with various brightness levels
            # Add some structure
            x = torch.linspace(0, 2 * np.pi, w)
            y = torch.linspace(0, 2 * np.pi, h)
            xx, yy = torch.meshgrid(x, y, indexing="ij")

            # Sinusoidal pattern with noise
            pattern = 0.3 + 0.2 * torch.sin(xx) * torch.cos(yy)
            pattern += 0.1 * torch.sin(3 * xx + yy)

            # Add some bright spots
            for _ in range(5):
                cx, cy = torch.randint(10, w - 10, (1,)), torch.randint(
                    10, h - 10, (1,)
                )
                r = torch.randint(3, 8, (1,))
                yy_spot, xx_spot = torch.meshgrid(
                    torch.arange(h), torch.arange(w), indexing="ij"
                )
                mask = ((xx_spot - cx) ** 2 + (yy_spot - cy) ** 2) < r**2
                pattern[mask] += 0.3

            image[0, 0] = torch.clamp(pattern, 0, 1)

        elif domain == "microscopy":
            # Fluorescent spots (simulating cells/particles)
            for _ in range(10):
                cx, cy = torch.randint(5, w - 5, (1,)), torch.randint(5, h - 5, (1,))
                sigma = torch.rand(1) * 3 + 2

                yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
                spot = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
                intensity = torch.rand(1) * 0.8 + 0.2

                image[0, 0] += intensity * spot

            # Add background
            image[0, 0] += 0.05
            image[0, 0] = torch.clamp(image[0, 0], 0, 1)

        elif domain == "astronomy":
            # Point sources (stars) on dark background
            for _ in range(8):
                cx, cy = torch.randint(0, w, (1,)), torch.randint(0, h, (1,))
                intensity = torch.rand(1) * 0.9 + 0.1

                # PSF-like profile (Gaussian)
                yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
                psf = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 8)

                image[0, 0] += intensity * psf

            # Very low background
            image[0, 0] += 0.01
            image[0, 0] = torch.clamp(image[0, 0], 0, 1)

        return image

    def evaluate_our_method(
        self, test_data: List[Dict[str, Any]], method_func: Optional[callable] = None
    ) -> List[EvaluationReport]:
        """
        Evaluate our Poisson-Gaussian diffusion method.

        Args:
            test_data: List of test data dictionaries
            method_func: Optional method function (uses dummy if None)

        Returns:
            List of evaluation reports
        """
        logger.info("Evaluating our Poisson-Gaussian diffusion method")

        if method_func is None:
            # Use dummy method for demonstration
            method_func = self._dummy_our_method

        reports = []

        for i, data in enumerate(test_data):
            logger.info(
                f"Processing test case {i+1}/{len(test_data)}: {data['dataset_name']}"
            )

            try:
                # Run our method
                start_time = time.time()
                pred = method_func(
                    noisy=data["noisy"],
                    scale=data["scale"],
                    background=data["background"],
                    read_noise=data["read_noise"],
                )
                processing_time = time.time() - start_time

                # Evaluate
                report = self.evaluation_suite.evaluate_restoration(
                    pred=pred,
                    target=data["clean"],
                    noisy=data["noisy"],
                    scale=data["scale"],
                    domain=data["domain"],
                    background=data["background"],
                    read_noise=data["read_noise"],
                    method_name="Poisson-Gaussian-Diffusion",
                    dataset_name=data["dataset_name"],
                )

                # Update processing time
                report.processing_time = processing_time

                reports.append(report)

                logger.info(
                    f"  PSNR: {report.psnr.value:.2f} dB, "
                    f"SSIM: {report.ssim.value:.3f}, "
                    f"χ²: {report.chi2_consistency.value:.3f}"
                )

            except Exception as e:
                logger.error(f"Failed to evaluate test case {i}: {e}")

        return reports

    def _dummy_our_method(
        self,
        noisy: torch.Tensor,
        scale: float,
        background: float = 0.0,
        read_noise: float = 0.0,
    ) -> torch.Tensor:
        """
        Dummy implementation of our method for demonstration.

        In practice, this would be replaced with the actual
        Poisson-Gaussian diffusion implementation.
        """
        # Simple denoising: normalize and apply light Gaussian filtering
        noisy_norm = (noisy - background) / scale
        noisy_norm = torch.clamp(noisy_norm, 0, 1)

        # Apply light Gaussian filtering
        kernel_size = 3
        sigma = 0.5

        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=noisy.device)
        x = x - kernel_size // 2
        xx, yy = torch.meshgrid(x, x, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        # Apply convolution
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(noisy_norm.shape[1], 1, -1, -1)

        pad = kernel_size // 2
        noisy_padded = torch.nn.functional.pad(
            noisy_norm, (pad, pad, pad, pad), mode="reflect"
        )

        denoised = torch.nn.functional.conv2d(
            noisy_padded, kernel, groups=noisy_norm.shape[1]
        )

        return torch.clamp(denoised, 0, 1)

    def evaluate_baselines(
        self, test_data: List[Dict[str, Any]], results_dir: str = "baseline_results"
    ) -> Dict[str, List[EvaluationReport]]:
        """
        Evaluate all baseline methods.

        Args:
            test_data: List of test data dictionaries
            results_dir: Directory to save results

        Returns:
            Dictionary mapping method names to lists of reports
        """
        logger.info("Evaluating baseline methods")

        baseline_results = {}

        for i, data in enumerate(test_data):
            logger.info(
                f"Processing test case {i+1}/{len(test_data)}: {data['dataset_name']}"
            )

            # Evaluate all baselines for this test case
            case_results = self.baseline_comparator.evaluate_all_baselines(
                noisy=data["noisy"],
                target=data["clean"],
                scale=data["scale"],
                domain=data["domain"],
                background=data["background"],
                read_noise=data["read_noise"],
                dataset_name=data["dataset_name"],
                results_dir=results_dir,
            )

            # Organize results by method
            for method_name, report in case_results.items():
                if method_name not in baseline_results:
                    baseline_results[method_name] = []
                baseline_results[method_name].append(report)

        return baseline_results

    def compare_methods(
        self,
        our_reports: List[EvaluationReport],
        baseline_results: Dict[str, List[EvaluationReport]],
        output_dir: str = "comparison_results",
    ) -> Dict[str, Any]:
        """
        Compare our method with baselines.

        Args:
            our_reports: Reports for our method
            baseline_results: Baseline method reports
            output_dir: Output directory for results

        Returns:
            Comparison results
        """
        logger.info("Comparing methods")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Combine all reports
        all_reports = our_reports.copy()
        for method_reports in baseline_results.values():
            all_reports.extend(method_reports)

        # Generate comparison
        comparison_file = output_path / "method_comparison.json"
        comparison = self.evaluation_suite.compare_methods(
            all_reports, str(comparison_file)
        )

        # Generate summary statistics
        summary = self.evaluation_suite.generate_summary_statistics(all_reports)

        summary_file = output_path / "summary_statistics.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Generate detailed report
        if our_reports:
            detailed_comparison = self.baseline_comparator.compare_with_our_method(
                our_reports[0],  # Use first report as representative
                {
                    name: reports[0]
                    for name, reports in baseline_results.items()
                    if reports
                },
            )

            report_file = output_path / "comparison_report.md"
            self.baseline_comparator.generate_comparison_report(
                detailed_comparison, str(report_file)
            )

        logger.info(f"Comparison results saved to {output_dir}")

        return {
            "comparison": comparison,
            "summary": summary,
            "output_dir": str(output_path),
        }

    def create_visualizations(
        self, comparison_results: Dict[str, Any], output_dir: str = "visualizations"
    ) -> None:
        """
        Create visualization plots for the evaluation results.

        Args:
            comparison_results: Results from method comparison
            output_dir: Output directory for plots
        """
        logger.info("Creating visualizations")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        comparison = comparison_results["comparison"]

        for key, result in comparison.items():
            dataset, domain = key.split("_", 1)

            methods = result["metrics"]["methods"]
            psnr_values = result["metrics"]["psnr"]
            ssim_values = result["metrics"]["ssim"]
            chi2_values = result["metrics"]["chi2_consistency"]

            # Filter out NaN values
            valid_data = []
            for i, method in enumerate(methods):
                if not (
                    np.isnan(psnr_values[i])
                    or np.isnan(ssim_values[i])
                    or np.isnan(chi2_values[i])
                ):
                    valid_data.append(
                        (method, psnr_values[i], ssim_values[i], chi2_values[i])
                    )

            if not valid_data:
                continue

            methods_clean, psnr_clean, ssim_clean, chi2_clean = zip(*valid_data)

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Method Comparison: {dataset} - {domain}", fontsize=16)

            # PSNR comparison
            axes[0, 0].bar(methods_clean, psnr_clean)
            axes[0, 0].set_title("PSNR (dB) - Higher is Better")
            axes[0, 0].set_ylabel("PSNR (dB)")
            axes[0, 0].tick_params(axis="x", rotation=45)

            # SSIM comparison
            axes[0, 1].bar(methods_clean, ssim_clean)
            axes[0, 1].set_title("SSIM - Higher is Better")
            axes[0, 1].set_ylabel("SSIM")
            axes[0, 1].tick_params(axis="x", rotation=45)

            # Chi-squared comparison
            axes[1, 0].bar(methods_clean, chi2_clean)
            axes[1, 0].set_title("χ² Consistency - Closer to 1 is Better")
            axes[1, 0].set_ylabel("χ²")
            axes[1, 0].axhline(
                y=1.0, color="red", linestyle="--", alpha=0.7, label="Ideal (χ²=1)"
            )
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis="x", rotation=45)

            # PSNR vs SSIM scatter plot
            axes[1, 1].scatter(psnr_clean, ssim_clean, s=100, alpha=0.7)
            for i, method in enumerate(methods_clean):
                axes[1, 1].annotate(
                    method,
                    (psnr_clean[i], ssim_clean[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
            axes[1, 1].set_xlabel("PSNR (dB)")
            axes[1, 1].set_ylabel("SSIM")
            axes[1, 1].set_title("PSNR vs SSIM")
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_file = output_path / f"comparison_{dataset}_{domain}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()

        # Create summary plot across all datasets
        self._create_summary_plot(comparison_results["summary"], output_path)

        logger.info(f"Visualizations saved to {output_dir}")

    def _create_summary_plot(self, summary: Dict[str, Any], output_path: Path) -> None:
        """Create summary plot across all methods and domains."""
        if not summary:
            return

        # Extract data for plotting
        methods = []
        domains = []
        psnr_means = []
        ssim_means = []
        chi2_means = []

        for key, stats in summary.items():
            method, domain = key.split("_", 1)
            methods.append(method)
            domains.append(domain)
            psnr_means.append(stats["psnr_stats"]["mean"])
            ssim_means.append(stats["ssim_stats"]["mean"])
            chi2_means.append(stats["chi2_stats"]["mean"])

        # Create grouped bar plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Method Performance Summary Across All Domains", fontsize=16)

        x_pos = np.arange(len(methods))

        # PSNR
        bars1 = axes[0].bar(x_pos, psnr_means, alpha=0.8)
        axes[0].set_title("Average PSNR (dB)")
        axes[0].set_ylabel("PSNR (dB)")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(methods, rotation=45, ha="right")

        # Add value labels on bars
        for bar, value in zip(bars1, psnr_means):
            if not np.isnan(value):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # SSIM
        bars2 = axes[1].bar(x_pos, ssim_means, alpha=0.8)
        axes[1].set_title("Average SSIM")
        axes[1].set_ylabel("SSIM")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(methods, rotation=45, ha="right")

        for bar, value in zip(bars2, ssim_means):
            if not np.isnan(value):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Chi-squared
        bars3 = axes[2].bar(x_pos, chi2_means, alpha=0.8)
        axes[2].set_title("Average χ² Consistency")
        axes[2].set_ylabel("χ²")
        axes[2].axhline(
            y=1.0, color="red", linestyle="--", alpha=0.7, label="Ideal (χ²=1)"
        )
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(methods, rotation=45, ha="right")
        axes[2].legend()

        for bar, value in zip(bars3, chi2_means):
            if not np.isnan(value):
                axes[2].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()

        # Save plot
        plot_file = output_path / "summary_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

    def run_full_evaluation(
        self,
        test_data_path: str,
        output_dir: str = "evaluation_results",
        method_func: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            test_data_path: Path to test data
            output_dir: Output directory for all results
            method_func: Optional method function

        Returns:
            Complete evaluation results
        """
        logger.info("Starting full evaluation pipeline")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load test data
        test_data = self.load_test_data(test_data_path)

        # Evaluate our method
        our_reports = self.evaluate_our_method(test_data, method_func)

        # Save our method results
        our_results_file = output_path / "our_method_results.json"
        save_evaluation_results(our_reports, str(our_results_file))

        # Evaluate baselines
        baseline_results = self.evaluate_baselines(
            test_data, str(output_path / "baselines")
        )

        # Save baseline results
        for method_name, reports in baseline_results.items():
            baseline_file = output_path / f"baseline_{method_name}_results.json"
            save_evaluation_results(reports, str(baseline_file))

        # Compare methods
        comparison_results = self.compare_methods(
            our_reports, baseline_results, str(output_path / "comparison")
        )

        # Create visualizations
        self.create_visualizations(
            comparison_results, str(output_path / "visualizations")
        )

        # Create final summary
        final_results = {
            "our_method_reports": len(our_reports),
            "baseline_methods": list(baseline_results.keys()),
            "total_test_cases": len(test_data),
            "output_directory": str(output_path),
            "comparison_results": comparison_results,
        }

        summary_file = output_path / "evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        logger.info(f"Full evaluation completed. Results saved to {output_dir}")

        return final_results


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate Poisson-Gaussian diffusion restoration"
    )

    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument(
        "--test-data",
        type=str,
        default="synthetic",
        help='Test data path or "synthetic"',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dummy",
        help="Method to evaluate (dummy, our_method)",
    )
    parser.add_argument(
        "--compare-baselines", action="store_true", help="Include baseline comparison"
    )
    parser.add_argument(
        "--create-plots",
        action="store_true",
        default=True,
        help="Create visualization plots",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda, cpu, auto)"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Override config with command line arguments
    if args.device == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        config["device"] = args.device

    # Initialize evaluation runner
    runner = EvaluationRunner(config)

    # Run evaluation
    try:
        results = runner.run_full_evaluation(
            test_data_path=args.test_data,
            output_dir=args.output_dir,
            method_func=None,  # Use dummy method
        )

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Results saved to: {results['output_directory']}")
        print(f"Test cases processed: {results['total_test_cases']}")
        print(f"Our method reports: {results['our_method_reports']}")
        print(f"Baseline methods: {', '.join(results['baseline_methods'])}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
