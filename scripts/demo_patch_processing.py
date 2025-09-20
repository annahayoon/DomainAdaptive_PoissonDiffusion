#!/usr/bin/env python
"""
Demonstration of patch processing capabilities.

This script demonstrates the patch extraction, processing, and reconstruction
system for handling large scientific images.

Usage:
    python scripts/demo_patch_processing.py --demo basic
    python scripts/demo_patch_processing.py --demo memory-efficient --size 2048
    python scripts/demo_patch_processing.py --demo dataset --data-dir /path/to/images
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.calibration import CalibrationParams, SensorCalibration
from core.logging_config import setup_project_logging
from core.patch_processing import (
    MemoryEfficientPatchProcessor,
    PatchExtractor,
    PatchReconstructor,
    calculate_optimal_patch_size,
    create_patch_processor,
)
from data.patch_dataset import (
    PatchDataset,
    PatchDatasetConfig,
    create_patch_dataloader,
    create_patch_dataset_from_directory,
)

logger = setup_project_logging(level="INFO")


class PatchProcessingDemo:
    """
    Interactive demonstration of patch processing capabilities.

    This class provides various demonstrations of the patch processing
    system for different use cases and scenarios.
    """

    def __init__(self, device: str = "auto"):
        """Initialize patch processing demo."""
        self.device = self._setup_device(device)
        logger.info(f"Initialized patch processing demo on {self.device}")

    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = "cpu"

        return device

    def create_synthetic_image(
        self, size: tuple = (1024, 1024), pattern: str = "microscopy"
    ) -> torch.Tensor:
        """
        Create synthetic test image.

        Args:
            size: Image size (height, width)
            pattern: Pattern type ('microscopy', 'astronomy', 'photography')

        Returns:
            Synthetic image tensor
        """
        height, width = size

        if pattern == "microscopy":
            # Create fluorescent spots
            image = torch.zeros(1, 1, height, width)
            num_spots = np.random.randint(50, 200)

            for _ in range(num_spots):
                y = np.random.randint(10, height - 10)
                x = np.random.randint(10, width - 10)
                intensity = np.random.uniform(0.3, 1.0)
                sigma = np.random.uniform(2.0, 5.0)

                # Create Gaussian spot
                yy, xx = torch.meshgrid(
                    torch.arange(height, dtype=torch.float32),
                    torch.arange(width, dtype=torch.float32),
                    indexing="ij",
                )
                spot = intensity * torch.exp(
                    -((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma**2)
                )
                image[0, 0] += spot

            # Add background noise
            image += torch.randn_like(image) * 0.05
            image = torch.clamp(image, 0, 1)

        elif pattern == "astronomy":
            # Create star field
            image = torch.ones(1, 1, height, width) * 0.1  # Background
            num_stars = np.random.randint(100, 500)

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
                    -((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5**2)
                )
                image[0, 0] += star

            # Add gradient background
            gradient_x = torch.linspace(0, 0.1, width).unsqueeze(0).expand(height, -1)
            gradient_y = torch.linspace(0, 0.05, height).unsqueeze(1).expand(-1, width)
            image[0, 0] += gradient_x + gradient_y

            image = torch.clamp(image, 0, 1)

        elif pattern == "photography":
            # Create natural image-like structure
            # Start with random noise
            image = torch.rand(1, 1, height, width)

            # Apply multiple smoothing operations at different scales
            for scale in [5, 15, 31]:
                kernel_size = scale
                sigma = scale / 3.0

                # Create Gaussian kernel
                x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
                xx, yy = torch.meshgrid(x, x, indexing="ij")
                kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                kernel = kernel / kernel.sum()
                kernel = kernel.unsqueeze(0).unsqueeze(0)

                # Apply convolution
                pad = kernel_size // 2
                image_padded = torch.nn.functional.pad(
                    image, (pad, pad, pad, pad), mode="reflect"
                )
                smoothed = torch.nn.functional.conv2d(image_padded, kernel)

                # Blend with original
                alpha = 0.3
                image = alpha * smoothed + (1 - alpha) * image

            # Scale to [0.2, 0.9] range
            image = image * 0.7 + 0.2

        else:
            # Generic random pattern
            image = torch.rand(1, 1, height, width)

        return image

    def demo_basic_patch_processing(self, image_size: tuple = (512, 512)):
        """Demonstrate basic patch extraction and reconstruction."""
        logger.info("=== Basic Patch Processing Demo ===")

        # Create synthetic image
        logger.info(f"Creating synthetic image of size {image_size}")
        image = self.create_synthetic_image(image_size, "microscopy")

        # Initialize patch extractor
        patch_size = 128
        overlap = 32
        extractor = PatchExtractor(
            patch_size=patch_size, overlap=overlap, device=self.device
        )

        logger.info(f"Extracting patches: patch_size={patch_size}, overlap={overlap}")

        # Extract patches
        start_time = time.time()
        patches, patch_infos = extractor.extract_patches(image, return_info=True)
        extraction_time = time.time() - start_time

        logger.info(f"Extracted {len(patches)} patches in {extraction_time:.3f}s")

        # Simple processing function (add some noise)
        def add_noise(patch):
            noise = torch.randn_like(patch) * 0.02
            return torch.clamp(patch + noise, 0, 1)

        # Process patches
        logger.info("Processing patches (adding noise)")
        start_time = time.time()
        processed_patches = [add_noise(patch) for patch in patches]
        processing_time = time.time() - start_time

        logger.info(f"Processed {len(patches)} patches in {processing_time:.3f}s")

        # Reconstruct image
        reconstructor = PatchReconstructor(device=self.device)

        logger.info("Reconstructing image from processed patches")
        start_time = time.time()
        reconstructed = reconstructor.reconstruct_image(
            processed_patches, patch_infos, image_size[0], image_size[1], 1
        )
        reconstruction_time = time.time() - start_time

        logger.info(f"Reconstructed image in {reconstruction_time:.3f}s")

        # Calculate reconstruction quality
        mse = torch.mean((reconstructed - image) ** 2).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")

        logger.info(f"Reconstruction quality: MSE={mse:.6f}, PSNR={psnr:.2f} dB")

        # Create visualization
        self._visualize_patch_processing(
            image, reconstructed, patches[:4], processed_patches[:4]
        )

        return {
            "original": image,
            "reconstructed": reconstructed,
            "patches": patches,
            "processed_patches": processed_patches,
            "patch_infos": patch_infos,
            "metrics": {"mse": mse, "psnr": psnr},
            "timing": {
                "extraction": extraction_time,
                "processing": processing_time,
                "reconstruction": reconstruction_time,
            },
        }

    def demo_memory_efficient_processing(self, image_size: tuple = (2048, 2048)):
        """Demonstrate memory-efficient processing of large images."""
        logger.info("=== Memory-Efficient Processing Demo ===")

        # Create large synthetic image
        logger.info(f"Creating large synthetic image of size {image_size}")
        image = self.create_synthetic_image(image_size, "astronomy")

        # Calculate optimal patch size
        optimal_patch_h, optimal_patch_w = calculate_optimal_patch_size(
            image_size[0], image_size[1], available_memory_gb=2.0
        )

        logger.info(
            f"Calculated optimal patch size: {optimal_patch_h}x{optimal_patch_w}"
        )

        # Create memory-efficient processor
        processor = MemoryEfficientPatchProcessor(
            patch_size=(optimal_patch_h, optimal_patch_w),
            overlap=(optimal_patch_h // 8, optimal_patch_w // 8),
            max_patches_in_memory=4,
            device=self.device,
        )

        # Estimate memory usage
        memory_est = processor.estimate_memory_usage(image_size[0], image_size[1], 1)

        logger.info(f"Memory estimation:")
        logger.info(f"  Full image: {memory_est['full_image_mb']:.1f} MB")
        logger.info(f"  Single patch: {memory_est['single_patch_mb']:.1f} MB")
        logger.info(f"  Patch cache: {memory_est['patch_cache_mb']:.1f} MB")
        logger.info(f"  Total patches: {memory_est['total_patches']}")
        logger.info(f"  Estimated peak: {memory_est['estimated_peak_mb']:.1f} MB")

        # Define processing function (simple denoising)
        def denoise_func(patch):
            # Simple Gaussian smoothing
            kernel_size = 5
            sigma = 1.0

            # Create Gaussian kernel
            x = (
                torch.arange(kernel_size, dtype=torch.float32, device=patch.device)
                - kernel_size // 2
            )
            xx, yy = torch.meshgrid(x, x, indexing="ij")
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            kernel = kernel.unsqueeze(0).unsqueeze(0)

            # Apply convolution
            pad = kernel_size // 2
            patch_padded = torch.nn.functional.pad(
                patch, (pad, pad, pad, pad), mode="reflect"
            )
            denoised = torch.nn.functional.conv2d(patch_padded, kernel)

            return denoised

        # Process image
        logger.info("Processing large image with memory-efficient pipeline")
        start_time = time.time()

        result = processor.process_image(
            image, denoise_func, batch_size=2, show_progress=True
        )

        processing_time = time.time() - start_time

        logger.info(f"Processed large image in {processing_time:.3f}s")

        # Calculate quality metrics
        mse = torch.mean((result - image) ** 2).item()
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")

        logger.info(f"Processing quality: MSE={mse:.6f}, PSNR={psnr:.2f} dB")

        # Test streaming processing
        logger.info("Testing streaming processing")
        start_time = time.time()

        result_streaming = processor.process_image_streaming(image, denoise_func)

        streaming_time = time.time() - start_time

        logger.info(f"Streaming processing completed in {streaming_time:.3f}s")

        # Compare results
        streaming_mse = torch.mean((result_streaming - result) ** 2).item()
        logger.info(f"Streaming vs batch MSE: {streaming_mse:.8f}")

        return {
            "original": image,
            "processed_batch": result,
            "processed_streaming": result_streaming,
            "memory_estimate": memory_est,
            "timing": {
                "batch_processing": processing_time,
                "streaming_processing": streaming_time,
            },
            "metrics": {"batch_psnr": psnr, "streaming_difference": streaming_mse},
        }

    def demo_patch_dataset(self, data_dir: str = None):
        """Demonstrate patch dataset functionality."""
        logger.info("=== Patch Dataset Demo ===")

        if data_dir and Path(data_dir).exists():
            logger.info(f"Using real data from {data_dir}")
            # Use real data
            config = PatchDatasetConfig(
                patch_size=256,
                overlap=64,
                random_patches=True,
                patches_per_image=4,
                device=self.device,
            )

            try:
                dataset = create_patch_dataset_from_directory(
                    data_dir=data_dir,
                    config=config,
                    domain="microscopy",  # Adjust based on your data
                    mode="train",
                )

                logger.info(f"Created dataset with {len(dataset)} patches")

                # Create dataloader
                dataloader = create_patch_dataloader(
                    dataset, batch_size=4, shuffle=True, num_workers=0
                )

                # Test loading a few batches
                logger.info("Testing data loading")
                for i, batch in enumerate(dataloader):
                    if i >= 3:  # Test first 3 batches
                        break

                    electrons = batch["electrons"]
                    masks = batch["mask"]
                    metadata = batch["metadata"]

                    logger.info(
                        f"Batch {i}: electrons shape={electrons.shape}, "
                        f"masks shape={masks.shape}"
                    )
                    logger.info(f"  Image paths: {metadata['image_paths'][:2]}...")

                return {"dataset": dataset, "dataloader": dataloader}

            except Exception as e:
                logger.warning(f"Failed to create dataset from {data_dir}: {e}")
                logger.info("Falling back to synthetic data demo")

        # Synthetic data demo
        logger.info("Creating synthetic patch dataset")

        # Create synthetic images and save them
        import tempfile

        temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Creating synthetic images in {temp_dir}")

        # Create a few synthetic images
        for i in range(5):
            image = self.create_synthetic_image((512, 512), "microscopy")

            # Save as tensor (simulating processed data)
            torch.save(image, temp_dir / f"synthetic_image_{i:03d}.pt")

        # Create custom dataset for synthetic data
        class SyntheticPatchDataset(torch.utils.data.Dataset):
            def __init__(self, image_dir, patch_size=128, patches_per_image=4):
                self.image_files = list(Path(image_dir).glob("*.pt"))
                self.patch_size = patch_size
                self.patches_per_image = patches_per_image

                self.extractor = PatchExtractor(patch_size=patch_size, overlap=32)

            def __len__(self):
                return len(self.image_files) * self.patches_per_image

            def __getitem__(self, idx):
                image_idx = idx // self.patches_per_image
                patch_idx = idx % self.patches_per_image

                # Load image
                image = torch.load(self.image_files[image_idx])

                # Extract random patch
                patches = self.extractor.extract_patches(image, return_info=False)
                patch = patches[np.random.randint(len(patches))]

                return {
                    "electrons": patch.squeeze(0),  # Remove batch dim
                    "mask": torch.ones_like(patch.squeeze(0)),
                    "image_path": str(self.image_files[image_idx]),
                    "image_idx": image_idx,
                    "patch_idx": patch_idx,
                    "domain": "synthetic",
                    "mode": "demo",
                }

        # Create synthetic dataset
        synthetic_dataset = SyntheticPatchDataset(temp_dir)

        logger.info(f"Created synthetic dataset with {len(synthetic_dataset)} patches")

        # Test loading
        for i in range(3):
            sample = synthetic_dataset[i]
            logger.info(f"Sample {i}: electrons shape={sample['electrons'].shape}")

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

        return {"synthetic_dataset": synthetic_dataset}

    def demo_blending_modes(self):
        """Demonstrate different blending modes for patch reconstruction."""
        logger.info("=== Blending Modes Demo ===")

        # Create test image
        image = self.create_synthetic_image((256, 256), "photography")

        # Extract patches with significant overlap
        extractor = PatchExtractor(patch_size=128, overlap=64, device=self.device)
        patches, patch_infos = extractor.extract_patches(image, return_info=True)

        # Test different blending modes
        blending_modes = ["linear", "cosine", "gaussian"]
        results = {}

        for mode in blending_modes:
            logger.info(f"Testing {mode} blending")

            reconstructor = PatchReconstructor(blending_mode=mode, device=self.device)

            start_time = time.time()
            reconstructed = reconstructor.reconstruct_image(
                patches, patch_infos, 256, 256, 1
            )
            reconstruction_time = time.time() - start_time

            # Calculate quality
            mse = torch.mean((reconstructed - image) ** 2).item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")

            results[mode] = {
                "reconstructed": reconstructed,
                "mse": mse,
                "psnr": psnr,
                "time": reconstruction_time,
            }

            logger.info(
                f"  {mode}: PSNR={psnr:.2f} dB, Time={reconstruction_time:.3f}s"
            )

        # Visualize results
        self._visualize_blending_comparison(image, results)

        return results

    def _visualize_patch_processing(
        self, original, reconstructed, sample_patches, processed_patches
    ):
        """Visualize patch processing results."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Original image
        axes[0, 0].imshow(original[0, 0].cpu().numpy(), cmap="gray")
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Reconstructed image
        axes[0, 1].imshow(reconstructed[0, 0].cpu().numpy(), cmap="gray")
        axes[0, 1].set_title("Reconstructed Image")
        axes[0, 1].axis("off")

        # Difference
        diff = torch.abs(reconstructed - original)
        axes[0, 2].imshow(diff[0, 0].cpu().numpy(), cmap="hot")
        axes[0, 2].set_title("Absolute Difference")
        axes[0, 2].axis("off")

        # Histogram comparison
        orig_flat = original.flatten().cpu().numpy()
        recon_flat = reconstructed.flatten().cpu().numpy()

        axes[0, 3].hist(orig_flat, bins=50, alpha=0.7, label="Original", density=True)
        axes[0, 3].hist(
            recon_flat, bins=50, alpha=0.7, label="Reconstructed", density=True
        )
        axes[0, 3].set_title("Intensity Histograms")
        axes[0, 3].legend()

        # Sample patches
        for i in range(min(4, len(sample_patches))):
            if i < len(sample_patches):
                # Original patch
                axes[1, i].imshow(sample_patches[i][0, 0].cpu().numpy(), cmap="gray")
                axes[1, i].set_title(f"Patch {i+1}")
                axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()

    def _visualize_blending_comparison(self, original, results):
        """Visualize blending mode comparison."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Original
        axes[0, 0].imshow(original[0, 0].cpu().numpy(), cmap="gray")
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        # Blending results
        for i, (mode, result) in enumerate(results.items()):
            if i < 3:
                axes[0, i + 1].imshow(
                    result["reconstructed"][0, 0].cpu().numpy(), cmap="gray"
                )
                axes[0, i + 1].set_title(
                    f'{mode.title()} Blending\nPSNR: {result["psnr"]:.2f} dB'
                )
                axes[0, i + 1].axis("off")

                # Difference images
                diff = torch.abs(result["reconstructed"] - original)
                axes[1, i + 1].imshow(diff[0, 0].cpu().numpy(), cmap="hot")
                axes[1, i + 1].set_title(f"{mode.title()} Difference")
                axes[1, i + 1].axis("off")

        # Performance comparison
        modes = list(results.keys())
        psnrs = [results[mode]["psnr"] for mode in modes]
        times = [results[mode]["time"] for mode in modes]

        axes[1, 0].bar(modes, psnrs)
        axes[1, 0].set_title("PSNR Comparison")
        axes[1, 0].set_ylabel("PSNR (dB)")

        plt.tight_layout()
        plt.show()


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Patch processing demonstration")
    parser.add_argument(
        "--demo",
        "-d",
        type=str,
        default="basic",
        choices=["basic", "memory-efficient", "dataset", "blending", "all"],
        help="Type of demonstration to run",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Image size for demos (height width)",
    )
    parser.add_argument(
        "--data-dir", type=str, help="Directory containing real images for dataset demo"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for computation",
    )

    args = parser.parse_args()

    # Initialize demo
    demo = PatchProcessingDemo(device=args.device)

    print(f"\n{'='*60}")
    print("PATCH PROCESSING DEMONSTRATION")
    print(f"{'='*60}")
    print(f"Device: {demo.device}")
    print(f"Demo type: {args.demo}")
    print(f"Image size: {args.size}")
    print(f"{'='*60}\n")

    results = {}

    if args.demo == "basic" or args.demo == "all":
        print("Running basic patch processing demo...")
        results["basic"] = demo.demo_basic_patch_processing(tuple(args.size))
        print()

    if args.demo == "memory-efficient" or args.demo == "all":
        print("Running memory-efficient processing demo...")
        # Use larger size for memory demo
        large_size = (max(1024, args.size[0]), max(1024, args.size[1]))
        results["memory_efficient"] = demo.demo_memory_efficient_processing(large_size)
        print()

    if args.demo == "dataset" or args.demo == "all":
        print("Running patch dataset demo...")
        results["dataset"] = demo.demo_patch_dataset(args.data_dir)
        print()

    if args.demo == "blending" or args.demo == "all":
        print("Running blending modes demo...")
        results["blending"] = demo.demo_blending_modes()
        print()

    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    print("\nKey Features Demonstrated:")
    print("✓ Patch extraction with configurable overlap")
    print("✓ Seamless reconstruction with multiple blending modes")
    print("✓ Memory-efficient processing for large images")
    print("✓ Adaptive patch sizing based on available memory")
    print("✓ Integration with data loading and calibration systems")
    print("✓ Dataset integration for training and inference")
    print("✓ Performance optimization and memory management")

    if results:
        print("\nPerformance Summary:")
        for demo_name, result in results.items():
            if "timing" in result:
                timing = result["timing"]
                print(f"  {demo_name.title()}:")
                for operation, time_taken in timing.items():
                    print(f"    {operation}: {time_taken:.3f}s")


if __name__ == "__main__":
    main()
