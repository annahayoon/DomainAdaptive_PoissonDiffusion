#!/usr/bin/env python
"""
Comprehensive baseline evaluation script.

This script provides a complete framework for evaluating and comparing
baseline methods against our Poisson-Gaussian diffusion approach.

Usage:
    python scripts/evaluate_baselines.py --config configs/baseline_evaluation.yaml
    python scripts/evaluate_baselines.py --synthetic --num-samples 100
    python scripts/evaluate_baselines.py --data-dir /path/to/test/data
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.baselines import BaselineComparator, create_baseline_suite
from core.calibration import CalibrationParams, SensorCalibration
from core.logging_config import setup_project_logging
from core.metrics import EvaluationSuite
from data.loaders import FormatDetector

logger = setup_project_logging(level="INFO")


class BaselineEvaluationFramework:
    """
    Comprehensive framework for baseline method evaluation.

    This class orchestrates the complete evaluation pipeline:
    1. Data loading and preprocessing
    2. Baseline method execution
    3. Comprehensive metric evaluation
    4. Statistical analysis and comparison
    5. Report generation and visualization
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        device: str = "auto",
        output_dir: str = "baseline_evaluation_results",
    ):
        """
        Initialize evaluation framework.

        Args:
            config_path: Path to configuration file
            device: Device for computation ('cuda', 'cpu', 'auto')
            output_dir: Directory for saving results
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.baseline_comparator = create_baseline_suite(device=self.device)
        self.evaluation_suite = EvaluationSuite(device=self.device)
        self.format_detector = FormatDetector()

        # Results storage
        self.results = {}
        self.summary_stats = {}

        logger.info(f"Initialized baseline evaluation framework")
        logger.info(f"Device: {self.device}")
        logger.info(
            f"Available baselines: {list(self.baseline_comparator.available_baselines.keys())}"
        )
        logger.info(f"Output directory: {self.output_dir}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "evaluation": {
                "domains": ["photography", "microscopy", "astronomy"],
                "metrics": ["psnr", "ssim", "lpips", "chi2_consistency"],
                "save_individual_results": True,
                "save_visualizations": True,
            },
            "baselines": {
                "classical": [
                    "BM3D",
                    "Anscombe+BM3D",
                    "Richardson-Lucy",
                    "Gaussian",
                    "Wiener",
                ],
                "deep_learning": ["DnCNN", "NAFNet"],
                "self_supervised": ["Noise2Void"],
                "diffusion": [],  # Will be added if model provided
            },
            "synthetic_data": {
                "num_samples": 50,
                "image_sizes": [(128, 128), (256, 256)],
                "noise_levels": [0.1, 0.5, 1.0, 2.0],
                "signal_levels": [100, 1000, 10000],
            },
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
        else:
            config = default_config
            if config_path:
                logger.warning(f"Config file {config_path} not found, using defaults")

        return config

    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = "cpu"

        return device

    def generate_synthetic_data(
        self, num_samples: int = 50, domains: List[str] = None
    ) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]]:
        """
        Generate synthetic test data for evaluation.

        Args:
            num_samples: Number of samples per domain
            domains: List of domains to generate data for

        Returns:
            Dictionary mapping domain names to lists of (noisy, clean, metadata) tuples
        """
        if domains is None:
            domains = self.config["evaluation"]["domains"]

        logger.info(f"Generating synthetic data for domains: {domains}")

        synthetic_data = {}

        for domain in domains:
            logger.info(f"Generating {num_samples} samples for {domain}")

            domain_data = []

            # Domain-specific parameters
            if domain == "photography":
                scale_range = (1000, 10000)
                background_range = (100, 500)
                read_noise_range = (3, 10)
                size_range = [(256, 256), (512, 512)]
            elif domain == "microscopy":
                scale_range = (500, 5000)
                background_range = (50, 200)
                read_noise_range = (1, 5)
                size_range = [(128, 128), (256, 256)]
            elif domain == "astronomy":
                scale_range = (10000, 50000)
                background_range = (0, 100)
                read_noise_range = (1, 3)
                size_range = [(256, 256), (512, 512)]
            else:
                # Default parameters
                scale_range = (1000, 10000)
                background_range = (100, 500)
                read_noise_range = (3, 10)
                size_range = [(128, 128), (256, 256)]

            for i in range(num_samples):
                # Random parameters
                scale = np.random.uniform(*scale_range)
                background = np.random.uniform(*background_range)
                read_noise = np.random.uniform(*read_noise_range)
                height, width = size_range[np.random.randint(len(size_range))]

                # Generate clean image
                clean = self._generate_clean_image(height, width, domain)

                # Generate noisy observation
                clean_electrons = clean * scale + background
                poisson_noise = torch.poisson(clean_electrons) - clean_electrons
                gaussian_noise = torch.randn_like(clean_electrons) * read_noise
                noisy_electrons = clean_electrons + poisson_noise + gaussian_noise

                # Metadata
                metadata = {
                    "domain": domain,
                    "scale": scale,
                    "background": background,
                    "read_noise": read_noise,
                    "height": height,
                    "width": width,
                    "sample_id": i,
                }

                domain_data.append((noisy_electrons, clean, metadata))

            synthetic_data[domain] = domain_data

        logger.info(
            f"Generated synthetic data: {sum(len(data) for data in synthetic_data.values())} total samples"
        )
        return synthetic_data

    def _generate_clean_image(
        self, height: int, width: int, domain: str
    ) -> torch.Tensor:
        """Generate domain-appropriate clean image."""
        # Base random image
        clean = torch.rand(1, 1, height, width)

        if domain == "photography":
            # Add some structure typical of photography
            # Smooth regions with some edges
            clean = torch.nn.functional.avg_pool2d(clean, 3, stride=1, padding=1)
            clean = clean * 0.8 + 0.1  # Scale to [0.1, 0.9]

        elif domain == "microscopy":
            # Add point-like structures typical of fluorescence microscopy
            num_points = np.random.randint(5, 20)
            for _ in range(num_points):
                y = np.random.randint(10, height - 10)
                x = np.random.randint(10, width - 10)
                intensity = np.random.uniform(0.5, 1.0)
                sigma = np.random.uniform(1.0, 3.0)

                # Add Gaussian spot
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
            # Add point sources typical of astronomy
            num_stars = np.random.randint(10, 50)
            for _ in range(num_stars):
                y = np.random.randint(5, height - 5)
                x = np.random.randint(5, width - 5)
                intensity = np.random.uniform(0.1, 1.0)

                # Add point source with PSF
                yy, xx = torch.meshgrid(
                    torch.arange(height, dtype=torch.float32),
                    torch.arange(width, dtype=torch.float32),
                    indexing="ij",
                )
                psf = intensity * torch.exp(
                    -((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5**2)
                )
                clean[0, 0] += psf

            # Add background gradient
            gradient = 0.1 * (torch.arange(width, dtype=torch.float32) / width)
            clean[0, 0] += gradient.unsqueeze(0)

            clean = torch.clamp(clean, 0, 1)

        return clean

    def load_real_data(
        self, data_dir: str
    ) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]]:
        """
        Load real test data from directory.

        Args:
            data_dir: Directory containing test images

        Returns:
            Dictionary mapping domain names to lists of (noisy, clean, metadata) tuples
        """
        data_dir = Path(data_dir)
        logger.info(f"Loading real data from {data_dir}")

        real_data = {}

        # Look for organized directory structure: data_dir/domain/noisy/ and data_dir/domain/clean/
        for domain_dir in data_dir.iterdir():
            if not domain_dir.is_dir():
                continue

            domain = domain_dir.name
            noisy_dir = domain_dir / "noisy"
            clean_dir = domain_dir / "clean"

            if not (noisy_dir.exists() and clean_dir.exists()):
                logger.warning(
                    f"Skipping {domain}: missing noisy/ or clean/ subdirectories"
                )
                continue

            domain_data = []

            # Match noisy and clean images
            noisy_files = sorted(noisy_dir.glob("*"))
            clean_files = sorted(clean_dir.glob("*"))

            for noisy_file, clean_file in zip(noisy_files, clean_files):
                try:
                    # Load images using format detector
                    noisy_data, noisy_metadata = self.format_detector.load_auto(
                        noisy_file
                    )
                    clean_data, clean_metadata = self.format_detector.load_auto(
                        clean_file
                    )

                    # Convert to tensors
                    noisy_tensor = torch.from_numpy(noisy_data).float()
                    clean_tensor = torch.from_numpy(clean_data).float()

                    # Add batch dimension if needed
                    if noisy_tensor.ndim == 2:
                        noisy_tensor = noisy_tensor.unsqueeze(0).unsqueeze(0)
                    elif noisy_tensor.ndim == 3:
                        noisy_tensor = noisy_tensor.unsqueeze(0)

                    if clean_tensor.ndim == 2:
                        clean_tensor = clean_tensor.unsqueeze(0).unsqueeze(0)
                    elif clean_tensor.ndim == 3:
                        clean_tensor = clean_tensor.unsqueeze(0)

                    # Metadata
                    metadata = {
                        "domain": domain,
                        "noisy_file": str(noisy_file),
                        "clean_file": str(clean_file),
                        "scale": 1000.0,  # Default scale
                        "background": 0.0,
                        "read_noise": 5.0,
                        "noisy_metadata": noisy_metadata,
                        "clean_metadata": clean_metadata,
                    }

                    domain_data.append((noisy_tensor, clean_tensor, metadata))

                except Exception as e:
                    logger.warning(f"Failed to load {noisy_file}, {clean_file}: {e}")

            if domain_data:
                real_data[domain] = domain_data
                logger.info(f"Loaded {len(domain_data)} image pairs for {domain}")

        return real_data

    def evaluate_baselines(
        self,
        test_data: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]],
        save_individual: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all baseline methods on test data.

        Args:
            test_data: Dictionary mapping domains to test data
            save_individual: Whether to save individual results

        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting baseline evaluation")

        all_results = {}

        for domain, domain_data in test_data.items():
            logger.info(
                f"Evaluating baselines on {domain} domain ({len(domain_data)} samples)"
            )

            domain_results = []

            for i, (noisy, clean, metadata) in enumerate(domain_data):
                logger.info(f"Processing {domain} sample {i+1}/{len(domain_data)}")

                # Move to device
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                # Extract parameters
                scale = metadata.get("scale", 1000.0)
                background = metadata.get("background", 0.0)
                read_noise = metadata.get("read_noise", 5.0)

                # Evaluate all baselines
                baseline_results = self.baseline_comparator.evaluate_all_baselines(
                    noisy=noisy,
                    target=clean,
                    scale=scale,
                    domain=domain,
                    background=background,
                    read_noise=read_noise,
                    dataset_name=f"sample_{i}",
                    save_results=save_individual,
                    results_dir=str(self.output_dir / "individual_results" / domain),
                )

                # Add metadata to results
                for method_name, report in baseline_results.items():
                    if hasattr(report.psnr, "metadata") and report.psnr.metadata:
                        report.psnr.metadata.update({"sample_metadata": metadata})

                domain_results.append(baseline_results)

            all_results[domain] = domain_results

        # Aggregate results
        self.results = all_results
        self._compute_summary_statistics()

        return all_results

    def _compute_summary_statistics(self):
        """Compute summary statistics across all results."""
        logger.info("Computing summary statistics")

        self.summary_stats = {}

        for domain, domain_results in self.results.items():
            domain_stats = {}

            # Collect all method names
            all_methods = set()
            for sample_results in domain_results:
                all_methods.update(sample_results.keys())

            # Compute statistics for each method
            for method in all_methods:
                method_stats = {
                    "psnr": [],
                    "ssim": [],
                    "lpips": [],
                    "chi2_consistency": [],
                    "processing_times": [],
                }

                for sample_results in domain_results:
                    if method in sample_results:
                        report = sample_results[method]

                        if not np.isnan(report.psnr.value):
                            method_stats["psnr"].append(report.psnr.value)
                        if not np.isnan(report.ssim.value):
                            method_stats["ssim"].append(report.ssim.value)
                        if hasattr(report, "lpips") and not np.isnan(
                            report.lpips.value
                        ):
                            method_stats["lpips"].append(report.lpips.value)
                        if not np.isnan(report.chi2_consistency.value):
                            method_stats["chi2_consistency"].append(
                                report.chi2_consistency.value
                            )
                        if hasattr(report, "processing_time"):
                            method_stats["processing_times"].append(
                                report.processing_time
                            )

                # Compute summary statistics
                method_summary = {}
                for metric, values in method_stats.items():
                    if values:
                        method_summary[metric] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "median": np.median(values),
                            "min": np.min(values),
                            "max": np.max(values),
                            "count": len(values),
                        }
                    else:
                        method_summary[metric] = {
                            "mean": np.nan,
                            "std": np.nan,
                            "median": np.nan,
                            "min": np.nan,
                            "max": np.nan,
                            "count": 0,
                        }

                domain_stats[method] = method_summary

            self.summary_stats[domain] = domain_stats

    def generate_comparison_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive comparison report.

        Args:
            output_file: Optional output file path

        Returns:
            Report content as string
        """
        if output_file is None:
            output_file = self.output_dir / "baseline_comparison_report.md"

        logger.info(f"Generating comparison report: {output_file}")

        report_lines = []
        report_lines.append("# Baseline Method Comparison Report")
        report_lines.append("")
        report_lines.append(
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"**Device**: {self.device}")
        report_lines.append("")

        # Executive summary
        report_lines.append("## Executive Summary")
        report_lines.append("")

        total_samples = sum(
            len(domain_results) for domain_results in self.results.values()
        )
        total_methods = len(self.baseline_comparator.available_baselines)

        report_lines.append(f"- **Total samples evaluated**: {total_samples}")
        report_lines.append(f"- **Baseline methods tested**: {total_methods}")
        report_lines.append(f"- **Domains evaluated**: {list(self.results.keys())}")
        report_lines.append("")

        # Domain-specific results
        for domain, domain_stats in self.summary_stats.items():
            report_lines.append(f"## {domain.title()} Domain Results")
            report_lines.append("")

            # Create results table
            methods = list(domain_stats.keys())
            if methods:
                report_lines.append(
                    "| Method | PSNR (dB) | SSIM | χ² Consistency | Processing Time (s) |"
                )
                report_lines.append(
                    "|--------|-----------|------|----------------|---------------------|"
                )

                for method in sorted(methods):
                    stats = domain_stats[method]
                    psnr_mean = stats["psnr"]["mean"]
                    psnr_std = stats["psnr"]["std"]
                    ssim_mean = stats["ssim"]["mean"]
                    ssim_std = stats["ssim"]["std"]
                    chi2_mean = stats["chi2_consistency"]["mean"]
                    chi2_std = stats["chi2_consistency"]["std"]
                    time_mean = stats["processing_times"]["mean"]

                    report_lines.append(
                        f"| {method} | "
                        f"{psnr_mean:.2f} ± {psnr_std:.2f} | "
                        f"{ssim_mean:.3f} ± {ssim_std:.3f} | "
                        f"{chi2_mean:.3f} ± {chi2_std:.3f} | "
                        f"{time_mean:.3f} |"
                    )

                report_lines.append("")

                # Best performing methods
                report_lines.append("### Best Performing Methods")
                report_lines.append("")

                # Find best methods for each metric
                best_psnr = max(
                    methods,
                    key=lambda m: domain_stats[m]["psnr"]["mean"]
                    if not np.isnan(domain_stats[m]["psnr"]["mean"])
                    else -np.inf,
                )
                best_ssim = max(
                    methods,
                    key=lambda m: domain_stats[m]["ssim"]["mean"]
                    if not np.isnan(domain_stats[m]["ssim"]["mean"])
                    else -np.inf,
                )
                best_chi2 = min(
                    methods,
                    key=lambda m: abs(domain_stats[m]["chi2_consistency"]["mean"] - 1.0)
                    if not np.isnan(domain_stats[m]["chi2_consistency"]["mean"])
                    else np.inf,
                )

                report_lines.append(
                    f"- **Best PSNR**: {best_psnr} ({domain_stats[best_psnr]['psnr']['mean']:.2f} dB)"
                )
                report_lines.append(
                    f"- **Best SSIM**: {best_ssim} ({domain_stats[best_ssim]['ssim']['mean']:.3f})"
                )
                report_lines.append(
                    f"- **Best χ² Consistency**: {best_chi2} ({domain_stats[best_chi2]['chi2_consistency']['mean']:.3f})"
                )
                report_lines.append("")

        # Method analysis
        report_lines.append("## Method Analysis")
        report_lines.append("")

        # Classical methods
        classical_methods = [
            "BM3D",
            "Anscombe+BM3D",
            "Richardson-Lucy",
            "Gaussian",
            "Wiener",
        ]
        available_classical = [
            m
            for m in classical_methods
            if m in self.baseline_comparator.available_baselines
        ]

        if available_classical:
            report_lines.append("### Classical Methods")
            report_lines.append("")
            for method in available_classical:
                report_lines.append(
                    f"- **{method}**: {self._get_method_description(method)}"
                )
            report_lines.append("")

        # Deep learning methods
        dl_methods = ["DnCNN", "NAFNet"]
        available_dl = [
            m for m in dl_methods if m in self.baseline_comparator.available_baselines
        ]

        if available_dl:
            report_lines.append("### Deep Learning Methods")
            report_lines.append("")
            for method in available_dl:
                report_lines.append(
                    f"- **{method}**: {self._get_method_description(method)}"
                )
            report_lines.append("")

        # Conclusions
        report_lines.append("## Conclusions")
        report_lines.append("")
        report_lines.append("### Key Findings")
        report_lines.append("")
        report_lines.append(
            "1. **Classical vs. Deep Learning**: [Analysis based on results]"
        )
        report_lines.append(
            "2. **Domain-Specific Performance**: [Domain-specific insights]"
        )
        report_lines.append("3. **Physics-Aware vs. Generic**: [Comparison insights]")
        report_lines.append("")

        # Write report
        report_content = "\n".join(report_lines)

        with open(output_file, "w") as f:
            f.write(report_content)

        logger.info(f"Comparison report saved to {output_file}")
        return report_content

    def _get_method_description(self, method: str) -> str:
        """Get description for a method."""
        descriptions = {
            "BM3D": "Block-matching 3D denoising with Gaussian noise assumption",
            "Anscombe+BM3D": "Anscombe transform + BM3D for Poisson noise",
            "Richardson-Lucy": "Iterative deconvolution algorithm",
            "Gaussian": "Simple Gaussian filtering",
            "Wiener": "Wiener filtering in frequency domain",
            "DnCNN": "Deep convolutional neural network for denoising",
            "NAFNet": "Nonlinear activation free network",
            "Noise2Void": "Self-supervised denoising method",
        }
        return descriptions.get(method, "No description available")

    def create_visualizations(self):
        """Create visualization plots for results."""
        logger.info("Creating result visualizations")

        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        for domain, domain_stats in self.summary_stats.items():
            # PSNR comparison
            self._create_metric_comparison_plot(
                domain_stats,
                "psnr",
                "PSNR (dB)",
                viz_dir / f"{domain}_psnr_comparison.png",
            )

            # SSIM comparison
            self._create_metric_comparison_plot(
                domain_stats, "ssim", "SSIM", viz_dir / f"{domain}_ssim_comparison.png"
            )

            # Chi-squared consistency
            self._create_metric_comparison_plot(
                domain_stats,
                "chi2_consistency",
                "χ² Consistency",
                viz_dir / f"{domain}_chi2_comparison.png",
            )

            # Processing time comparison
            self._create_metric_comparison_plot(
                domain_stats,
                "processing_times",
                "Processing Time (s)",
                viz_dir / f"{domain}_timing_comparison.png",
            )

        # Overall comparison across domains
        self._create_cross_domain_comparison(viz_dir / "cross_domain_comparison.png")

        logger.info(f"Visualizations saved to {viz_dir}")

    def _create_metric_comparison_plot(
        self, domain_stats: Dict[str, Any], metric: str, ylabel: str, output_path: Path
    ):
        """Create comparison plot for a specific metric."""
        methods = list(domain_stats.keys())
        means = [domain_stats[m][metric]["mean"] for m in methods]
        stds = [domain_stats[m][metric]["std"] for m in methods]

        # Filter out NaN values
        valid_data = [
            (m, mean, std)
            for m, mean, std in zip(methods, means, stds)
            if not np.isnan(mean)
        ]

        if not valid_data:
            return

        methods, means, stds = zip(*valid_data)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(methods)), means, yerr=stds, capsize=5, alpha=0.7)

        plt.xlabel("Method")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Comparison")
        plt.xticks(range(len(methods)), methods, rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + bar.get_height() * 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _create_cross_domain_comparison(self, output_path: Path):
        """Create cross-domain comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        metrics = ["psnr", "ssim", "chi2_consistency", "processing_times"]
        metric_labels = ["PSNR (dB)", "SSIM", "χ² Consistency", "Processing Time (s)"]

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]

            # Collect data for all domains
            domain_data = {}
            for domain, domain_stats in self.summary_stats.items():
                domain_data[domain] = {}
                for method, stats in domain_stats.items():
                    if not np.isnan(stats[metric]["mean"]):
                        domain_data[domain][method] = stats[metric]["mean"]

            # Create grouped bar plot
            methods = set()
            for domain_methods in domain_data.values():
                methods.update(domain_methods.keys())
            methods = sorted(list(methods))

            x = np.arange(len(methods))
            width = 0.25

            for j, (domain, domain_methods) in enumerate(domain_data.items()):
                values = [domain_methods.get(method, np.nan) for method in methods]
                ax.bar(x + j * width, values, width, label=domain, alpha=0.7)

            ax.set_xlabel("Method")
            ax.set_ylabel(label)
            ax.set_title(f"{label} Across Domains")
            ax.set_xticks(x + width)
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def save_results(self, output_file: Optional[str] = None):
        """Save complete results to JSON file."""
        if output_file is None:
            output_file = self.output_dir / "complete_results.json"

        logger.info(f"Saving complete results to {output_file}")

        # Convert results to serializable format
        serializable_results = {}

        for domain, domain_results in self.results.items():
            serializable_results[domain] = []

            for sample_results in domain_results:
                sample_data = {}

                for method, report in sample_results.items():
                    sample_data[method] = {
                        "psnr": report.psnr.value,
                        "ssim": report.ssim.value,
                        "lpips": getattr(report.lpips, "value", np.nan),
                        "chi2_consistency": report.chi2_consistency.value,
                        "processing_time": getattr(report, "processing_time", np.nan),
                        "method_name": report.method_name,
                        "dataset_name": report.dataset_name,
                        "domain": report.domain,
                    }

                serializable_results[domain].append(sample_data)

        # Add summary statistics
        complete_data = {
            "results": serializable_results,
            "summary_statistics": self.summary_stats,
            "config": self.config,
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "device": self.device,
                "available_baselines": list(
                    self.baseline_comparator.available_baselines.keys()
                ),
                "total_samples": sum(
                    len(domain_results) for domain_results in self.results.values()
                ),
            },
        }

        with open(output_file, "w") as f:
            json.dump(complete_data, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Comprehensive baseline method evaluation"
    )
    parser.add_argument("--config", "-c", type=str, help="Configuration file path")
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data for evaluation"
    )
    parser.add_argument(
        "--data-dir", type=str, help="Directory containing real test data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of synthetic samples per domain",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for computation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["photography", "microscopy", "astronomy"],
        help="Domains to evaluate",
    )

    args = parser.parse_args()

    # Initialize framework
    framework = BaselineEvaluationFramework(
        config_path=args.config, device=args.device, output_dir=args.output_dir
    )

    # Load or generate test data
    if args.data_dir:
        logger.info(f"Loading real data from {args.data_dir}")
        test_data = framework.load_real_data(args.data_dir)
    else:
        logger.info("Generating synthetic test data")
        test_data = framework.generate_synthetic_data(
            num_samples=args.num_samples, domains=args.domains
        )

    if not test_data:
        logger.error("No test data available")
        return

    # Run evaluation
    logger.info("Starting baseline evaluation")
    results = framework.evaluate_baselines(test_data)

    # Generate outputs
    logger.info("Generating outputs")
    framework.generate_comparison_report()
    framework.create_visualizations()
    framework.save_results()

    logger.info("Baseline evaluation complete!")
    logger.info(f"Results saved to: {framework.output_dir}")


if __name__ == "__main__":
    main()
