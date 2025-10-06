#!/usr/bin/env python
"""
DAPGD Inference Script

Main entry point for running inference on test data.

Usage:
    # Baseline (no guidance)
    python scripts/inference.py --mode baseline --input data/test/image.png --checkpoint model.pt

    # With PG guidance
    python scripts/inference.py --mode guided --input data/test/image.png \
        --checkpoint model.pt --s 1000 --sigma_r 5 --kappa 0.5

    # Batch processing
    python scripts/inference.py --mode guided --input data/test/ \
        --checkpoint model.pt --batch_size 4 --s 1000 --sigma_r 5
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dapgd.data.transforms import CalibrationPreservingTransform
from dapgd.guidance.pg_guidance import PoissonGaussianGuidance
from dapgd.metrics.image_quality import compute_psnr, compute_ssim
from dapgd.metrics.physical import analyze_residual_statistics, compute_chi_squared
from dapgd.sampling.dapgd_sampler import DAPGDSampler
from dapgd.sampling.edm_wrapper import EDMModelWrapper
from dapgd.utils.visualization import save_comparison_image


def setup_logging(args):
    """Setup logging configuration"""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "inference.log"),
        ],
    )
    return logging.getLogger("dapgd.inference")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_image(image_path: Path, device: str = "cuda") -> torch.Tensor:
    """
    Load image from file

    Returns:
        image: Tensor [1,C,H,W] in appropriate range
    """
    image_path = Path(image_path)

    if image_path.suffix == ".npy":
        # NumPy array (assumed to be in electron counts)
        img = np.load(image_path)
        img_tensor = torch.from_numpy(img).float()

        # Add channel dimension if grayscale
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0)

        # Add batch dimension
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

    elif image_path.suffix in [".png", ".jpg", ".jpeg"]:
        # Standard image file
        img = Image.open(image_path)
        img_array = np.array(img).astype(np.float32) / 255.0

        # Handle different formats
        if img_array.ndim == 2:  # Grayscale
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        else:  # RGB
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    elif image_path.suffix in [".tif", ".tiff"]:
        # TIFF (common in scientific imaging)
        import tifffile

        img = tifffile.imread(image_path)
        img_tensor = torch.from_numpy(img).float()

        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        elif img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

    else:
        raise ValueError(f"Unsupported file format: {image_path.suffix}")

    return img_tensor.to(device)


def save_image(image: torch.Tensor, path: Path, format: str = "png"):
    """Save image to file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove batch dimension
    if image.dim() == 4:
        image = image.squeeze(0)

    # Convert to numpy
    img_np = image.cpu().numpy()

    if format == "npy":
        np.save(path, img_np)
    elif format in ["png", "jpg"]:
        # Convert to [0, 255] uint8
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        # Handle channel dimension
        if img_np.shape[0] in [1, 3]:  # CHW format
            img_np = np.transpose(img_np, (1, 2, 0))

        if img_np.shape[2] == 1:  # Grayscale
            img_np = img_np.squeeze(2)

        Image.fromarray(img_np).save(path)
    elif format == "tiff":
        import tifffile

        tifffile.imwrite(path, img_np)
    else:
        raise ValueError(f"Unsupported save format: {format}")


def process_single_image(
    image_path: Path,
    sampler: DAPGDSampler,
    args,
    logger: logging.Logger,
    ground_truth_path: Optional[Path] = None,
) -> dict:
    """
    Process a single image

    Returns:
        Dictionary with results and metrics
    """
    start_time = time.time()

    logger.info(f"Processing: {image_path.name}")

    # 1. Load image
    y_noisy = load_image(image_path, device=args.device)
    logger.info(f"  Input shape: {y_noisy.shape}")
    logger.info(f"  Input range: [{y_noisy.min():.2f}, {y_noisy.max():.2f}]")

    # 2. Apply transforms if needed
    if args.use_transforms:
        transform = CalibrationPreservingTransform(target_size=args.image_size)
        y_noisy, metadata = transform.forward(y_noisy)
        logger.info(f"  Transformed to: {y_noisy.shape}")
    else:
        metadata = None

    # 3. Run sampling
    logger.info("  Running sampling...")
    x_restored = sampler.sample(
        y_e=y_noisy if args.mode == "guided" else None,
        image_size=(y_noisy.shape[2], y_noisy.shape[3]),
        channels=y_noisy.shape[1],
        show_progress=not args.no_progress,
        seed=args.seed,
    )

    # 4. Apply inverse transform if needed
    if args.use_transforms and metadata is not None:
        x_restored = transform.inverse(x_restored, metadata)
        logger.info(f"  Restored to original size: {x_restored.shape}")

    # 5. Compute metrics if ground truth available
    metrics = {}
    if ground_truth_path is not None:
        x_gt = load_image(ground_truth_path, device=args.device)

        metrics["psnr"] = compute_psnr(x_restored, x_gt)
        metrics["ssim"] = compute_ssim(x_restored, x_gt)

        logger.info(f"  PSNR: {metrics['psnr']:.2f} dB")
        logger.info(f"  SSIM: {metrics['ssim']:.4f}")

    # 6. Compute chi-squared if guided
    if args.mode == "guided" and sampler.guidance is not None:
        metrics["chi2"] = sampler.compute_chi_squared(x_restored, y_noisy)
        logger.info(f"  χ²: {metrics['chi2']:.4f}")

        # Detailed residual analysis
        if args.analyze_residuals:
            residual_stats = analyze_residual_statistics(
                x_restored, y_noisy, args.s, args.sigma_r
            )
            metrics["residual_stats"] = residual_stats

    # 7. Save results
    output_path = args.output_dir / image_path.stem

    # Save restored image
    save_image(
        x_restored,
        output_path.with_suffix(f"_restored.{args.save_format}"),
        format=args.save_format,
    )

    # Save comparison if ground truth available
    if ground_truth_path is not None and args.save_comparison:
        save_comparison_image(
            y_noisy, x_restored, x_gt, output_path.with_suffix("_comparison.png")
        )

    elapsed_time = time.time() - start_time
    metrics["time"] = elapsed_time

    logger.info(f"  Completed in {elapsed_time:.2f}s")

    return {"path": str(image_path), "metrics": metrics, "output": str(output_path)}


def main(args):
    """Main inference pipeline"""

    # Setup
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(args)
    logger.info("Starting DAPGD Inference")
    logger.info(f"Arguments: {vars(args)}")

    # Load configuration if provided
    if args.config is not None:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = {}

    # Create sampler
    logger.info("Loading model...")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Extract network from checkpoint
    if "ema" in checkpoint:
        network = checkpoint["ema"]
    elif "model" in checkpoint:
        network = checkpoint["model"]
    else:
        network = checkpoint

    # Wrap network
    edm_wrapper = EDMModelWrapper(network, img_channels=3)
    edm_wrapper.eval()

    # Create guidance if in guided mode
    if args.mode == "guided":
        guidance = PoissonGaussianGuidance(
            s=args.s,
            sigma_r=args.sigma_r,
            kappa=args.kappa,
            tau=args.tau,
            mode=args.guidance_mode,
        )
    else:
        guidance = None

    # Create sampler
    sampler = DAPGDSampler(
        edm_wrapper=edm_wrapper,
        guidance=guidance,
        num_steps=args.num_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        device=args.device,
    )

    logger.info("Model loaded successfully")

    # Get list of input files
    input_path = Path(args.input)

    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        # Find all images in directory
        extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"]
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
        image_files = sorted(image_files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    logger.info(f"Found {len(image_files)} image(s) to process")

    # Find ground truth files if directory provided
    ground_truth_files = {}
    if args.ground_truth_dir is not None:
        gt_dir = Path(args.ground_truth_dir)
        for img_file in image_files:
            gt_file = gt_dir / img_file.name
            if gt_file.exists():
                ground_truth_files[img_file] = gt_file
        logger.info(f"Found {len(ground_truth_files)} ground truth images")

    # Process all images
    all_results = []

    for img_file in tqdm(image_files, desc="Processing images"):
        gt_file = ground_truth_files.get(img_file)

        try:
            result = process_single_image(img_file, sampler, args, logger, gt_file)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing {img_file}: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()

    # Aggregate results
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)

    if all_results:
        # Compute average metrics
        metrics_with_gt = [r for r in all_results if "psnr" in r["metrics"]]

        if metrics_with_gt:
            avg_psnr = np.mean([r["metrics"]["psnr"] for r in metrics_with_gt])
            avg_ssim = np.mean([r["metrics"]["ssim"] for r in metrics_with_gt])
            logger.info(f"Average PSNR: {avg_psnr:.2f} dB")
            logger.info(f"Average SSIM: {avg_ssim:.4f}")

        if args.mode == "guided":
            metrics_with_chi2 = [r for r in all_results if "chi2" in r["metrics"]]
            if metrics_with_chi2:
                avg_chi2 = np.mean([r["metrics"]["chi2"] for r in metrics_with_chi2])
                logger.info(f"Average χ²: {avg_chi2:.4f}")

        avg_time = np.mean([r["metrics"]["time"] for r in all_results])
        logger.info(f"Average time per image: {avg_time:.2f}s")

        # Save results summary
        results_file = args.output_dir / "results.yaml"
        with open(results_file, "w") as f:
            yaml.dump(all_results, f)
        logger.info(f"Results saved to: {results_file}")

    logger.info("Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DAPGD Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="guided",
        choices=["baseline", "guided"],
        help="Inference mode",
    )

    # Input/output
    parser.add_argument(
        "--input", type=str, required=True, help="Input image or directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments/results", help="Output directory"
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default=None,
        help="Directory with ground truth images (for evaluation)",
    )

    # Model
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML file"
    )

    # Sampling
    parser.add_argument(
        "--num_steps", type=int, default=50, help="Number of sampling steps"
    )
    parser.add_argument(
        "--sigma_min", type=float, default=0.002, help="Minimum noise level"
    )
    parser.add_argument(
        "--sigma_max", type=float, default=80.0, help="Maximum noise level"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    # Guidance (for guided mode)
    parser.add_argument(
        "--s",
        type=float,
        default=1000.0,
        help="Scale factor (photon count at saturation)",
    )
    parser.add_argument(
        "--sigma_r", type=float, default=5.0, help="Read noise standard deviation"
    )
    parser.add_argument("--kappa", type=float, default=0.5, help="Guidance strength")
    parser.add_argument("--tau", type=float, default=0.01, help="Guidance threshold")
    parser.add_argument(
        "--guidance_mode",
        type=str,
        default="wls",
        choices=["wls", "full"],
        help="Gradient computation mode",
    )

    # Preprocessing
    parser.add_argument(
        "--use_transforms",
        action="store_true",
        help="Apply calibration-preserving transforms",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Target image size for transforms"
    )

    # Output
    parser.add_argument(
        "--save_format",
        type=str,
        default="png",
        choices=["png", "tiff", "npy"],
        help="Output format",
    )
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="Save comparison images (if ground truth available)",
    )
    parser.add_argument(
        "--analyze_residuals",
        action="store_true",
        help="Perform detailed residual analysis",
    )

    # System
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for computation"
    )
    parser.add_argument(
        "--no_progress", action="store_true", help="Disable progress bars"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Run inference
    main(args)
