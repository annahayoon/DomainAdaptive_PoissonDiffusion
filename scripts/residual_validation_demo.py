#!/usr/bin/env python3
"""
Demonstration script for residual validation functionality.

This script shows how to use the comprehensive residual validation system
to assess whether restoration methods produce physically correct residuals.

Usage:
    python residual_validation_demo.py --config <config_file> --output <output_dir>

Requirements addressed: 2.3.1-2.3.5 from evaluation_enhancement_todos.md
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.exceptions import AnalysisError
from core.metrics import ResidualAnalyzer
from core.residual_analysis import ResidualValidationSuite
from visualization.residual_plots import ResidualPlotter, create_publication_plots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_test_data(
    n_images: int = 5,
    image_size: Tuple[int, int] = (128, 128),
    domain: str = "photography",
    scale_range: Tuple[float, float] = (100, 1000),
    read_noise_range: Tuple[float, float] = (5, 20),
    background_range: Tuple[float, float] = (0, 50),
) -> List[Dict]:
    """
    Generate synthetic test data for residual validation demonstration.

    Args:
        n_images: Number of test images to generate
        image_size: Size of each image (H, W)
        domain: Domain name for metadata
        scale_range: Range for scale parameters
        read_noise_range: Range for read noise
        background_range: Range for background

    Returns:
        List of dictionaries containing test data
    """
    test_data = []
    np.random.seed(42)  # For reproducible results

    for i in range(n_images):
        # Generate clean image (simple pattern)
        clean = np.zeros((1, 1, image_size[0], image_size[1]), dtype=np.float32)

        # Create a test pattern with varying intensity levels
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        intensity = np.sin(x / 20.0) * np.cos(y / 20.0) * 0.5 + 0.5
        clean[0, 0] = intensity.astype(np.float32)

        # Generate noisy observation with known Poisson-Gaussian noise
        scale = np.random.uniform(*scale_range)
        background = np.random.uniform(*background_range)
        read_noise = np.random.uniform(*read_noise_range)

        # Convert to electrons
        clean_electrons = clean * scale + background

        # Generate Poisson noise (photon noise)
        mean_photons = np.maximum(clean_electrons, 0.1)  # Avoid zero mean
        noisy_electrons = np.random.poisson(mean_photons)

        # Add Gaussian read noise
        noisy_electrons = noisy_electrons + np.random.normal(0, read_noise, noisy_electrons.shape)

        # Convert back to normalized space
        noisy = (noisy_electrons - background) / scale

        test_data.append({
            "clean": torch.tensor(clean),
            "noisy": torch.tensor(noisy_electrons.astype(np.float32)),
            "scale": scale,
            "background": background,
            "read_noise": read_noise,
            "domain": domain,
            "image_id": f"synthetic_{i:03d}",
        })

    logger.info(f"Generated {n_images} synthetic test images")
    return test_data


def simulate_restoration_methods(
    noisy_batch: torch.Tensor,
    clean_batch: torch.Tensor,
    method_configs: Dict,
) -> Dict[str, torch.Tensor]:
    """
    Simulate different restoration methods for comparison.

    Args:
        noisy_batch: Batch of noisy images [B, C, H, W] (electrons)
        clean_batch: Batch of clean images [B, C, H, W] (normalized)
        method_configs: Configuration for different methods

    Returns:
        Dictionary mapping method names to predicted images
    """
    results = {}
    B, C, H, W = noisy_batch.shape

    # Method 1: Oracle (perfect restoration using clean image)
    oracle_pred = clean_batch.clone()
    results["Oracle"] = oracle_pred

    # Method 2: Simple denoising (Gaussian blur)
    gaussian_kernel = torch.tensor([
        [[1, 4, 6, 4, 1],
         [4, 16, 24, 16, 4],
         [6, 24, 36, 24, 6],
         [4, 16, 24, 16, 4],
         [1, 4, 6, 4, 1]]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 256.0

    gaussian_pred = torch.zeros_like(clean_batch)
    for b in range(B):
        for c in range(C):
            # Convert to [0,1] for processing
            noisy_norm = noisy_batch[b, c] / 1000.0  # Approximate normalization
            gaussian_pred[b, c] = torch.nn.functional.conv2d(
                noisy_norm.unsqueeze(0).unsqueeze(0),
                gaussian_kernel,
                padding=2
            ).squeeze()

    results["Gaussian_Denoise"] = gaussian_pred

    # Method 3: Mean filter (more aggressive smoothing)
    mean_kernel = torch.ones((1, 1, 5, 5), dtype=torch.float32) / 25.0
    mean_pred = torch.zeros_like(clean_batch)
    for b in range(B):
        for c in range(C):
            noisy_norm = noisy_batch[b, c] / 1000.0
            mean_pred[b, c] = torch.nn.functional.conv2d(
                noisy_norm.unsqueeze(0).unsqueeze(0),
                mean_kernel,
                padding=2
            ).squeeze()

    results["Mean_Filter"] = mean_pred

    # Method 4: Wiener-like filter (adaptive based on local variance)
    wiener_pred = torch.zeros_like(clean_batch)
    for b in range(B):
        for c in range(C):
            noisy_norm = noisy_batch[b, c] / 1000.0

            # Compute local variance
            kernel_size = 5
            padding = kernel_size // 2

            # Simple variance estimation
            mean_local = torch.nn.functional.avg_pool2d(
                noisy_norm.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=padding
            )
            mean_sq_local = torch.nn.functional.avg_pool2d(
                (noisy_norm**2).unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=padding
            )
            var_local = mean_sq_local - mean_local**2

            # Wiener filter: signal = (variance / (variance + noise^2)) * (noisy - mean) + mean
            noise_var = 0.01  # Assumed noise variance
            wiener_factor = var_local / (var_local + noise_var)
            wiener_pred[b, c] = mean_local.squeeze() + wiener_factor.squeeze() * (noisy_norm - mean_local.squeeze())

    results["Wiener_Filter"] = wiener_pred

    logger.info(f"Simulated {len(results)} restoration methods")
    return results


def main():
    """Main function demonstrating residual validation."""
    parser = argparse.ArgumentParser(description="Residual validation demonstration")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--output", type=str, default="residual_validation_demo",
                       help="Output directory")
    parser.add_argument("--n-images", type=int, default=5, help="Number of test images")
    parser.add_argument("--domain", type=str, default="photography",
                       choices=["photography", "microscopy", "astronomy"],
                       help="Domain for test data")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    # Load configuration if provided
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")

    # Override with command line arguments
    n_images = args.n_images
    domain = args.domain
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting residual validation demonstration")
    logger.info(f"Domain: {domain}")
    logger.info(f"Number of images: {n_images}")
    logger.info(f"Output directory: {output_dir}")

    try:
        # Initialize validation suite
        validation_suite = ResidualValidationSuite(device="cpu")  # Use CPU for demo
        plotter = ResidualPlotter()

        # Generate synthetic test data
        logger.info("Generating synthetic test data...")
        test_data = generate_synthetic_test_data(
            n_images=n_images,
            domain=domain,
            scale_range=config.get("scale_range", (100, 1000)),
            read_noise_range=config.get("read_noise_range", (5, 20)),
            background_range=config.get("background_range", (0, 50)),
        )

        # Simulate restoration methods
        logger.info("Simulating restoration methods...")
        noisy_batch = torch.stack([data["noisy"] for data in test_data])
        clean_batch = torch.stack([data["clean"] for data in test_data])

        method_results = simulate_restoration_methods(
            noisy_batch, clean_batch, config.get("methods", {})
        )

        # Validate residuals for each method
        validation_reports = []

        for method_name, pred_batch in method_results.items():
            logger.info(f"Validating residuals for {method_name}...")

            for i, test_data_item in enumerate(test_data):
                try:
                    report = validation_suite.validate_residuals(
                        pred=pred_batch[i:i+1],
                        noisy=test_data_item["noisy"].unsqueeze(0),
                        scale=test_data_item["scale"],
                        background=test_data_item["background"],
                        read_noise=test_data_item["read_noise"],
                        method_name=method_name,
                        dataset_name="synthetic_demo",
                        domain=domain,
                        image_id=test_data_item["image_id"],
                    )

                    validation_reports.append(report)

                    # Save individual report
                    report.save_json(output_dir / f"validation_{method_name}_{test_data_item['image_id']}.json")

                    logger.info(f"  Image {i+1}: Physics correct = {report.physics_correct}")

                except AnalysisError as e:
                    logger.warning(f"  Failed to validate {method_name} image {i}: {e}")
                    continue

        if not validation_reports:
            logger.error("No validation reports generated")
            return 1

        # Generate statistical summary
        logger.info("Generating statistical summary...")
        summary = validation_suite.generate_statistical_summary(
            validation_reports,
            output_dir / "validation_summary.json"
        )

        # Print summary to console
        logger.info("Validation Summary:")
        for method_domain, stats in summary.items():
            logger.info(f"  {method_domain}:")
            logger.info(f"    Physics correct rate: {stats['physics_correct_rate']:.3f}")
            logger.info(f"    Gaussian fit rate: {stats['gaussian_fit_rate']:.3f}")
            logger.info(f"    Spatial uncorrelated rate: {stats['spatial_uncorrelated_rate']:.3f}")
            logger.info(f"    White spectrum rate: {stats['white_spectrum_rate']:.3f}")

        # Generate publication plots if requested
        if not args.no_plots:
            logger.info("Generating publication plots...")

            # Create individual validation plots for each report
            for report in validation_reports[:3]:  # Limit to first 3 for demo
                try:
                    fig = plotter.create_residual_validation_plots(report)
                    plt.savefig(output_dir / f"validation_plots_{report.method_name}_{report.image_id}.png",
                              dpi=150, bbox_inches='tight')
                    plt.close()
                    logger.info(f"  Created validation plots for {report.method_name}")
                except Exception as e:
                    logger.warning(f"  Failed to create plots for {report.method_name}: {e}")

            # Create comparison plots
            try:
                create_publication_plots(validation_reports, output_dir)
                logger.info("Created publication comparison plots")
            except Exception as e:
                logger.warning(f"Failed to create publication plots: {e}")

        # Generate comprehensive report
        logger.info("Generating comprehensive validation report...")
        report_content = generate_comprehensive_report(validation_reports, summary, config)
        report_file = output_dir / "comprehensive_validation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_content)

        logger.info(f"Comprehensive report saved to {report_file}")

        # Print success message
        logger.info("Residual validation demonstration completed successfully!")
        logger.info(f"Results saved to: {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        return 1


def generate_comprehensive_report(
    reports: List,
    summary: Dict,
    config: Dict
) -> str:
    """Generate comprehensive validation report."""
    lines = [
        "=" * 80,
        "COMPREHENSIVE RESIDUAL VALIDATION REPORT",
        "=" * 80,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        "This report presents a comprehensive analysis of residual validation",
        "for Poisson-Gaussian diffusion restoration methods. The analysis",
        "assesses whether restoration methods produce physically correct",
        "residuals that follow expected statistical distributions.",
        "",
        "Key findings:",
        "• Physics correctness is determined by multiple statistical tests",
        "• Residuals should be normally distributed with zero mean",
        "• Spatial correlation indicates systematic errors",
        "• Spectral analysis reveals noise characteristics",
        "",
        "METHODS TESTED",
        "-" * 40,
    ]

    # List tested methods
    methods = set(r.method_name for r in reports)
    for method in sorted(methods):
        lines.append(f"• {method}")

    lines.extend(["", "OVERALL RESULTS", "-" * 40])

    # Overall statistics
    total_reports = len(reports)
    physics_correct = sum(1 for r in reports if r.physics_correct)
    gaussian_fit = sum(1 for r in reports if r.gaussian_fit)
    spatial_uncorr = sum(1 for r in reports if r.spatial_uncorrelated)
    white_spectrum = sum(1 for r in reports if r.white_spectrum)

    lines.extend([
        f"Total validation runs: {total_reports}",
        f"Physics correct: {physics_correct}/{total_reports} ({100*physics_correct/total_reports:.1f}%)",
        f"Gaussian fit: {gaussian_fit}/{total_reports} ({100*gaussian_fit/total_reports:.1f}%)",
        f"Spatially uncorrelated: {spatial_uncorr}/{total_reports} ({100*spatial_uncorr/total_reports:.1f}%)",
        f"White spectrum: {white_spectrum}/{total_reports} ({100*white_spectrum/total_reports:.1f}%)",
        "",
        "DETAILED METHOD COMPARISON",
        "-" * 40,
    ])

    # Method-by-method breakdown
    for method_domain, stats in summary.items():
        lines.extend([
            f"Method: {stats['method']} ({stats['domain']})",
            f"  Evaluations: {stats['num_evaluations']}",
            f"  Physics correct: {stats['physics_correct_rate']:.3f}",
            f"  Gaussian fit: {stats['gaussian_fit_rate']:.3f}",
            f"  Spatial uncorrelated: {stats['spatial_uncorrelated_rate']:.3f}",
            f"  White spectrum: {stats['white_spectrum_rate']:.3f}",
            f"  Avg processing time: {stats['avg_processing_time']:.2f}s",
            "",
        ])

    # Individual report details
    lines.extend(["INDIVIDUAL REPORT DETAILS", "-" * 40])

    for i, report in enumerate(reports):
        lines.extend([
            f"Report {i+1}: {report.method_name} - {report.image_id}",
            f"  Physics correct: {'✓' if report.physics_correct else '✗'}",
            f"  Validation summary: {report.validation_summary}",
            f"  KS statistic: {report.ks_statistic:.4f}",
            f"  Spatial uncorrelated: {report.spatial_uncorrelated}",
            f"  White spectrum: {report.white_spectrum}",
        ])

        if report.recommendations:
            lines.append("  Recommendations:")
            for rec in report.recommendations:
                lines.append(f"    • {rec}")

        lines.append("")

    # Configuration info
    if config:
        lines.extend([
            "CONFIGURATION",
            "-" * 40,
            f"Domain: {config.get('domain', 'N/A')}",
            f"Scale range: {config.get('scale_range', 'N/A')}",
            f"Read noise range: {config.get('read_noise_range', 'N/A')}",
            f"Background range: {config.get('background_range', 'N/A')}",
            "",
        ])

    lines.extend([
        "CONCLUSION",
        "-" * 40,
        "This validation demonstrates the effectiveness of the residual",
        "analysis framework in assessing the physical correctness of",
        "image restoration methods. The framework provides:",
        "",
        "• Comprehensive statistical validation",
        "• Spatial and spectral analysis",
        "• Publication-quality visualizations",
        "• Automated report generation",
        "",
        "For scientifically rigorous image restoration, methods should",
        "achieve physics correctness rates > 0.95 across all tests.",
        "",
        "=" * 80,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    exit(main())
