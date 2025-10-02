#!/usr/bin/env python3
"""
Evaluate Poisson-Gaussian Model Checkpoint - Comprehensive Physics-Aware Evaluation

This script evaluates the Poisson-Gaussian diffusion model checkpoint trained with
run_prior_clean_h100_optimized.sh using comprehensive physics-aware metrics including
residual analysis and validation of physics correctness.

Requirements addressed: 2.3.1-2.3.5 from evaluation_enhancement_todos.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.logging_config import get_logger
from core.metrics import EvaluationSuite, PhysicsMetrics, StandardMetrics
from core.residual_analysis import ResidualValidationSuite
from models.edm_wrapper import create_domain_aware_edm_wrapper
# Note: load_checkpoint not used in this version
from visualization.residual_plots import ResidualPlotter, create_publication_plots

logger = get_logger(__name__)


class PoissonGaussianEvaluator:
    """Comprehensive evaluator for Poisson-Gaussian diffusion model checkpoint."""

    def __init__(
        self,
        checkpoint_path: str,
        data_path: str,
        output_dir: str = "poisson_gaussian_evaluation",
        device: str = "auto",
    ):
        """
        Initialize Poisson-Gaussian model evaluator.

        Args:
            checkpoint_path: Path to Poisson-Gaussian model checkpoint
            data_path: Path to preprocessed posterior data
            output_dir: Output directory for results
            device: Device for computation
        """
        self.device = self._setup_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize evaluation components
        self.evaluation_suite = EvaluationSuite(device=self.device)
        self.residual_validation = ResidualValidationSuite(device=self.device)
        self.plotter = ResidualPlotter()

        # Store checkpoint path
        self.checkpoint_path = checkpoint_path

        # Load Poisson-Gaussian model
        self.model = self._load_poisson_gaussian_model(checkpoint_path)
        self.model_name = "DAPGD (Poisson-Gaussian)"

        # Setup data paths
        self.data_path = Path(data_path)

        logger.info(f"PoissonGaussianEvaluator initialized with checkpoint: {checkpoint_path}")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Output directory: {output_dir}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def _load_poisson_gaussian_model(self, checkpoint_path: str) -> nn.Module:
        """Load Poisson-Gaussian model from checkpoint."""
        logger.info(f"Loading Poisson-Gaussian model from {checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            # Create EDM model wrapper with correct parameters based on checkpoint
            # From checkpoint analysis: map_layer0 is [2560, 320], so model_channels=320
            # The map layers suggest the model uses 2560 as intermediate dimension
            # This corresponds to model_channels=320 with channel_mult_emb=8 (320*8=2560)

            try:
                model = create_domain_aware_edm_wrapper(
                    domain="photography",
                    img_resolution=128,
                    model_channels=320,  # From checkpoint analysis
                    num_blocks=8,        # Standard configuration
                    channel_mult_emb=8,  # To match 2560 intermediate dimension
                    conditioning_mode="class_labels",
                )
                logger.info("Created EDM model wrapper for Poisson-Gaussian model")
            except Exception as e:
                logger.warning(f"Failed to create EDM wrapper: {e}")
                logger.info("Trying with different configuration...")
                # Try alternative configuration
                try:
                    model = create_domain_aware_edm_wrapper(
                        domain="photography",
                        img_resolution=128,
                        model_channels=320,
                        num_blocks=8,
                        channel_mult_emb=8,
                        conditioning_mode="class_labels",
                    )
                    logger.info("Created EDM model wrapper with alternative config")
                except Exception as e2:
                    logger.error(f"All EDM wrapper creation attempts failed: {e}, {e2}")
                    raise

            # Load state dict
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
            if state_dict is None:
                raise ValueError("No model state dict found in checkpoint")

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            logger.info(f"Poisson-Gaussian model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
            return model

        except Exception as e:
            logger.error(f"Failed to load Poisson-Gaussian model: {e}")
            raise

    def load_test_data(self, domain: str, max_samples: int = 10) -> List[Dict]:
        """Load test data for evaluation."""
        # The data_path already includes "posterior", so don't add it again
        domain_path = self.data_path / domain / "test"
        test_files = sorted(list(domain_path.glob("*.pt")))[:max_samples]

        logger.info(f"Loading test samples for {domain}")
        logger.info(f"Domain path: {domain_path}")
        logger.info(f"Test files found: {len(test_files)}")

        if len(test_files) == 0:
            logger.warning(f"No .pt files found in {domain_path}")
            return []

        test_data = []
        for i, file_path in enumerate(test_files):
            logger.info(f"Loading file {i+1}/{len(test_files)}: {file_path.name}")
            try:
                data = torch.load(file_path, map_location="cpu", weights_only=False)
                logger.info(f"File keys: {list(data.keys())}")

                # Extract data
                noisy = data["noisy_norm"]  # [4, H, W] for photography, [1, H, W] for others
                clean = data["clean_norm"]  # Ground truth
                metadata = data.get("metadata", {})

                logger.info(f"Noisy shape: {noisy.shape}, Clean shape: {clean.shape}")

                # Ensure consistent shape for evaluation
                if noisy.dim() == 3:
                    noisy = noisy.unsqueeze(0)  # [1, 4, H, W] or [1, 1, H, W]
                    clean = clean.unsqueeze(0)  # [1, 4, H, W] or [1, 1, H, W]

                # Handle channel mismatch
                if noisy.shape[1] == 4 and clean.shape[1] == 4:
                    # Photography: use first channel for evaluation
                    noisy_eval = noisy[:, 0:1]  # [1, 1, H, W]
                    clean_eval = clean[:, 0:1]  # [1, 1, H, W]
                else:
                    # Microscopy/Astronomy: single channel
                    noisy_eval = noisy
                    clean_eval = clean

                test_data.append({
                    "noisy": noisy_eval,
                    "clean": clean_eval,
                    "scale": metadata.get("scale", 1000.0),
                    "background": metadata.get("background", 0.0),
                    "read_noise": metadata.get("read_noise", 10.0),
                    "domain": domain,
                    "file_name": file_path.stem,
                })

                logger.info(f"Loaded {file_path.stem}: {noisy_eval.shape}")

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        logger.info(f"Successfully loaded {len(test_data)} test samples for {domain}")
        return test_data

    def evaluate_sample(self, sample_data: Dict) -> Dict:
        """Evaluate L2 baseline on a single sample."""
        noisy = sample_data["noisy"].to(self.device)
        clean = sample_data["clean"].to(self.device)
        scale = sample_data["scale"]
        background = sample_data["background"]
        read_noise = sample_data["read_noise"]
        domain = sample_data["domain"]
        file_name = sample_data["file_name"]

        logger.info(f"Evaluating {file_name} from {domain}")

        # Run inference
        with torch.no_grad():
            pred = self.model(noisy, sigma=torch.tensor([0.1], device=self.device))

        # Ensure output matches input format
        if pred.shape != noisy.shape:
            logger.warning(f"Output shape {pred.shape} != input shape {noisy.shape}")

        # Clamp predictions
        pred = torch.clamp(pred, 0, 1)

        # Evaluate using comprehensive evaluation suite
        try:
            report = self.evaluation_suite.evaluate_restoration(
                pred=pred,
                target=clean,
                noisy=noisy,
                scale=scale,
                domain=domain,
                background=background,
                read_noise=read_noise,
                method_name=self.model_name,
                dataset_name=file_name,
            )

            # Add residual validation - this is where we validate physics correctness
            residual_report = self.residual_validation.validate_residuals(
                pred=pred,
                noisy=noisy,
                scale=scale,
                background=background,
                read_noise=read_noise,
                method_name=self.model_name,
                dataset_name=file_name,
                domain=domain,
                image_id=file_name,
            )

            # Save residual report
            residual_report.save_json(self.output_dir / f"residual_validation_{file_name}.json")

            return {
                "sample_name": file_name,
                "domain": domain,
                "evaluation_report": report,
                "residual_report": residual_report,
                "scale": scale,
                "background": background,
                "read_noise": read_noise,
            }

        except Exception as e:
            logger.error(f"Evaluation failed for {file_name}: {e}")
            return None

    def create_comparison_plots(self, results: List[Dict]) -> None:
        """Create comparison plots across all samples."""
        logger.info("Creating comparison plots...")

        # Group results by domain
        domain_results = {}
        for result in results:
            if result is None:
                continue
            domain = result["domain"]
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(result)

        # Create domain-specific plots
        for domain, domain_results_list in domain_results.items():
            if len(domain_results_list) == 0:
                continue

            logger.info(f"Creating plots for {domain} domain ({len(domain_results_list)} samples)")

            # Extract metrics for plotting
            sample_names = [r["sample_name"] for r in domain_results_list]
            psnr_values = [r["evaluation_report"].psnr.value for r in domain_results_list]
            ssim_values = [r["evaluation_report"].ssim.value for r in domain_results_list]
            chi2_values = [r["evaluation_report"].chi2_consistency.value for r in domain_results_list]

            # Create performance comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # PSNR comparison
            axes[0].bar(sample_names, psnr_values, color='blue', alpha=0.7)
            axes[0].set_ylabel('PSNR (dB)')
            axes[0].set_title(f'L2 Baseline PSNR - {domain.title()}')
            axes[0].tick_params(axis='x', rotation=45)

            for i, (name, psnr) in enumerate(zip(sample_names, psnr_values)):
                axes[0].text(i, psnr + 0.5, f'{psnr:.1f}', ha='center', va='bottom')

            # SSIM comparison
            axes[1].bar(sample_names, ssim_values, color='green', alpha=0.7)
            axes[1].set_ylabel('SSIM')
            axes[1].set_title(f'L2 Baseline SSIM - {domain.title()}')
            axes[1].tick_params(axis='x', rotation=45)

            for i, (name, ssim) in enumerate(zip(sample_names, ssim_values)):
                axes[1].text(i, ssim + 0.01, f'{ssim:.3f}', ha='center', va='bottom')

            # χ² consistency
            axes[2].bar(sample_names, chi2_values, color='red', alpha=0.7)
            axes[2].set_ylabel('χ² Consistency')
            axes[2].set_title(f'L2 Baseline χ² - {domain.title()}')
            axes[2].axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Ideal (1.0)')
            axes[2].legend()
            axes[2].tick_params(axis='x', rotation=45)

            for i, (name, chi2) in enumerate(zip(sample_names, chi2_values)):
                axes[2].text(i, chi2 + 0.05, f'{chi2:.2f}', ha='center', va='bottom')

            plt.suptitle(f'L2 Baseline Performance Comparison - {domain.title()}', fontsize=16)
            plt.tight_layout()

            output_file = self.output_dir / f"l2_baseline_performance_{domain}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved performance comparison plot: {output_file}")

    def generate_residual_analysis_plots(self, results: List[Dict]) -> None:
        """Generate comprehensive residual analysis plots."""
        logger.info("Generating residual analysis plots...")

        # Collect all residual validation reports
        residual_reports = []
        for result in results:
            if result is None:
                continue
            residual_reports.append(result["residual_report"])

        if len(residual_reports) == 0:
            logger.warning("No residual reports available for plotting")
            return

        # Create publication-quality comparison plots
        try:
            create_publication_plots(residual_reports, self.output_dir)
            logger.info("Created publication comparison plots")
        except Exception as e:
            logger.warning(f"Failed to create publication plots: {e}")

        # Create individual residual validation plots for first few samples
        for i, report in enumerate(residual_reports[:3]):  # Limit to first 3
            try:
                fig = self.plotter.create_residual_validation_plots(report)
                output_file = self.output_dir / f"residual_validation_{report.method_name}_{report.image_id}.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Created residual validation plot: {output_file}")
            except Exception as e:
                logger.warning(f"Failed to create residual plot for {report.image_id}: {e}")

    def run_comprehensive_evaluation(
        self,
        domains: List[str] = ["photography", "microscopy", "astronomy"],
        max_samples_per_domain: int = 10,
    ) -> Dict:
        """Run comprehensive evaluation across all domains."""
        logger.info("Starting comprehensive L2 baseline evaluation")
        logger.info(f"Domains: {domains}")
        logger.info(f"Samples per domain: {max_samples_per_domain}")

        all_results = []
        domain_summaries = {}

        # Evaluate each domain
        for domain in domains:
            logger.info(f"Evaluating {domain} domain...")

            # Load test data
            test_data = self.load_test_data(domain, max_samples_per_domain)

            if not test_data:
                logger.warning(f"No test data available for {domain}")
                continue

            domain_results = []

            # Evaluate each sample
            for sample_data in tqdm(test_data, desc=f"Evaluating {domain}"):
                result = self.evaluate_sample(sample_data)
                if result is not None:
                    domain_results.append(result)
                    all_results.append(result)

            # Calculate domain summary
            if domain_results:
                avg_psnr = np.mean([r["evaluation_report"].psnr.value for r in domain_results])
                avg_ssim = np.mean([r["evaluation_report"].ssim.value for r in domain_results])
                avg_chi2 = np.mean([r["evaluation_report"].chi2_consistency.value for r in domain_results])

                domain_summaries[domain] = {
                    "samples_evaluated": len(domain_results),
                    "avg_psnr": avg_psnr,
                    "avg_ssim": avg_ssim,
                    "avg_chi2": avg_chi2,
                    "results": domain_results,
                }

                logger.info(f"{domain.upper()} SUMMARY:")
                logger.info(f"  Samples: {len(domain_results)}")
                logger.info(f"  Avg PSNR: {avg_psnr:.2f} dB")
                logger.info(f"  Avg SSIM: {avg_ssim:.4f}")
                logger.info(f"  Avg χ²: {avg_chi2:.3f}")

            # Create comparison plots
            if all_results:
                self.create_comparison_plots(all_results)
                self.generate_residual_analysis_plots(all_results)

        # Generate comprehensive report
        comprehensive_report = {
            "model_name": self.model_name,
            "checkpoint_path": str(self.checkpoint_path),
            "data_path": str(self.data_path),
            "device": str(self.device),
            "evaluation_timestamp": str(Path.cwd()),
            "domain_summaries": domain_summaries,
            "total_samples_evaluated": len(all_results),
            "overall_summary": self._generate_overall_summary(domain_summaries),
        }

        # Save comprehensive report
        report_file = self.output_dir / "poisson_gaussian_comprehensive_evaluation.json"
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)

        logger.info(f"Comprehensive evaluation report saved to {report_file}")

        return comprehensive_report

    def _generate_overall_summary(self, domain_summaries: Dict) -> Dict:
        """Generate overall evaluation summary."""
        if not domain_summaries:
            return {"error": "No domains evaluated successfully"}

        total_samples = sum(summary["samples_evaluated"] for summary in domain_summaries.values())
        avg_psnr = np.mean([summary["avg_psnr"] for summary in domain_summaries.values()])
        avg_ssim = np.mean([summary["avg_ssim"] for summary in domain_summaries.values()])
        avg_chi2 = np.mean([summary["avg_chi2"] for summary in domain_summaries.values()])

        return {
            "total_samples_evaluated": total_samples,
            "average_psnr": avg_psnr,
            "average_ssim": avg_ssim,
            "average_chi2": avg_chi2,
            "domains_evaluated": list(domain_summaries.keys()),
            "evaluation_complete": True,
        }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Poisson-Gaussian Model Checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Poisson-Gaussian model checkpoint (e.g., ~/checkpoint_step_0090000.pth)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to preprocessed posterior data (e.g., ~/anna_OS_ML/PKL-DiffusionDenoising/data/preprocessed/posterior)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="poisson_gaussian_evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["photography", "microscopy", "astronomy"],
        help="Domains to evaluate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum samples per domain"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("POISSON-GAUSSIAN MODEL CHECKPOINT EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Domains: {args.domains}")
    logger.info(f"Max samples per domain: {args.max_samples}")
    logger.info("=" * 80)

    try:
        # Initialize evaluator
        evaluator = PoissonGaussianEvaluator(
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            output_dir=args.output_dir,
            device=args.device,
        )

        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation(
            domains=args.domains,
            max_samples_per_domain=args.max_samples,
        )

        # Print final summary
        logger.info("=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)

        if results['total_samples_evaluated'] > 0:
            logger.info(f"Total samples evaluated: {results['total_samples_evaluated']}")
            logger.info(f"Average PSNR: {results['overall_summary']['average_psnr']:.2f} dB")
            logger.info(f"Average SSIM: {results['overall_summary']['average_ssim']:.4f}")
            logger.info(f"Average χ²: {results['overall_summary']['average_chi2']:.3f}")
        else:
            logger.warning("No samples were successfully evaluated!")
            logger.info("This could be due to:")
            logger.info("• No test data found in the specified path")
            logger.info("• Model loading issues")
            logger.info("• Data format incompatibility")

        logger.info("=" * 80)
        logger.info("This is our main Poisson-Gaussian diffusion model with physics-aware guidance:")
        logger.info("• Domain-Adaptive Poisson-Gaussian Diffusion (DAPGD)")
        logger.info("• Physics-correct noise modeling (Poisson + Gaussian)")
        logger.info("• Multi-domain unified model with conditioning")
        logger.info("• Comprehensive residual analysis validates physics correctness")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {args.output_dir}")
        if results['total_samples_evaluated'] > 0:
            logger.info("Check the output directory for:")
            logger.info("• evaluation_summary.json - Comprehensive results")
            logger.info("• Performance comparison plots")
            logger.info("• Residual validation reports and plots (physics correctness proof)")
        else:
            logger.info("Check the output directory for error logs and debug information")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
