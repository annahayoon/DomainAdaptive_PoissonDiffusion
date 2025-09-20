#!/usr/bin/env python
"""
Synthetic Poisson-Gaussian data generation for physics validation.

This script generates synthetic images with exact Poisson-Gaussian noise
statistics for validating the physics implementation. Critical for Phase 2.2.1
validation checkpoint.

Usage:
    python scripts/generate_synthetic_data.py --output_dir data/synthetic
    python scripts/generate_synthetic_data.py --config configs/synthetic_validation.yaml
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    # Image parameters
    image_size: int = 128
    num_images: int = 100

    # Noise parameters (in electrons)
    photon_levels: List[float] = None  # Will be set in __post_init__
    read_noise_levels: List[float] = None  # Will be set in __post_init__
    background_level: float = 10.0

    # Pattern types
    pattern_types: List[str] = None  # Will be set in __post_init__

    # Output
    output_dir: str = "data/synthetic"
    save_plots: bool = True
    save_metadata: bool = True

    def __post_init__(self):
        if self.photon_levels is None:
            # Cover critical regimes: very low (<10), low (10-100), medium (100-1000), high (>1000)
            self.photon_levels = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]

        if self.read_noise_levels is None:
            # Typical read noise levels in electrons
            self.read_noise_levels = [0.5, 1.0, 2.0, 5.0, 10.0]

        if self.pattern_types is None:
            self.pattern_types = [
                "constant",  # Uniform intensity
                "gradient",  # Linear gradient
                "checkerboard",  # High frequency pattern
                "gaussian_spots",  # Gaussian blobs
                "natural_image",  # Realistic image patterns
            ]


class SyntheticDataGenerator:
    """
    Generate synthetic images with exact Poisson-Gaussian noise statistics.

    This generator creates ground truth images with known noise characteristics
    for validating physics-based restoration algorithms.
    """

    def __init__(self, config: SyntheticConfig):
        """
        Initialize generator with configuration.

        Args:
            config: Synthetic data configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

    def generate_pattern(self, pattern_type: str, size: int) -> np.ndarray:
        """
        Generate base pattern before adding noise.

        Args:
            pattern_type: Type of pattern to generate
            size: Image size (square)

        Returns:
            Clean pattern [0, 1] normalized
        """
        if pattern_type == "constant":
            return np.ones((size, size), dtype=np.float32) * 0.5

        elif pattern_type == "gradient":
            x = np.linspace(0, 1, size)
            y = np.linspace(0, 1, size)
            X, Y = np.meshgrid(x, y)
            return ((X + Y) / 2).astype(np.float32)

        elif pattern_type == "checkerboard":
            # High frequency checkerboard
            x = np.arange(size)
            y = np.arange(size)
            X, Y = np.meshgrid(x, y)
            pattern = ((X // 8) + (Y // 8)) % 2
            return pattern.astype(np.float32) * 0.8 + 0.1

        elif pattern_type == "gaussian_spots":
            # Multiple Gaussian spots
            pattern = np.zeros((size, size), dtype=np.float32)
            num_spots = 5

            for _ in range(num_spots):
                # Random center
                cx = np.random.randint(size // 4, 3 * size // 4)
                cy = np.random.randint(size // 4, 3 * size // 4)

                # Random size and amplitude
                sigma = np.random.uniform(size // 20, size // 8)
                amplitude = np.random.uniform(0.3, 0.9)

                # Generate Gaussian
                x = np.arange(size)
                y = np.arange(size)
                X, Y = np.meshgrid(x, y)

                gaussian = amplitude * np.exp(
                    -((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2)
                )
                pattern += gaussian

            # Normalize to [0, 1]
            pattern = np.clip(pattern, 0, 1)
            return pattern

        elif pattern_type == "natural_image":
            # Simulate natural image with multiple frequency components
            pattern = np.zeros((size, size), dtype=np.float32)

            # Add multiple sine waves at different frequencies
            frequencies = [1, 2, 4, 8, 16]
            for freq in frequencies:
                # Random phase and amplitude
                phase_x = np.random.uniform(0, 2 * np.pi)
                phase_y = np.random.uniform(0, 2 * np.pi)
                amplitude = (
                    np.random.uniform(0.1, 0.3) / freq
                )  # Higher freq = lower amplitude

                x = np.linspace(0, 2 * np.pi * freq, size)
                y = np.linspace(0, 2 * np.pi * freq, size)
                X, Y = np.meshgrid(x, y)

                wave = amplitude * np.sin(X + phase_x) * np.sin(Y + phase_y)
                pattern += wave

            # Add some noise texture
            texture = np.random.normal(0, 0.05, (size, size))
            pattern += texture

            # Normalize to [0, 1] with some baseline
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
            pattern = pattern * 0.8 + 0.1  # Scale to [0.1, 0.9]

            return pattern.astype(np.float32)

        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

    def add_poisson_gaussian_noise(
        self,
        clean: np.ndarray,
        photon_level: float,
        read_noise: float,
        background: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Add exact Poisson-Gaussian noise to clean image.

        Args:
            clean: Clean image [0, 1]
            photon_level: Peak photon count (electrons)
            read_noise: Read noise std deviation (electrons)
            background: Background offset (electrons)

        Returns:
            (noisy_image, noise_params)
        """
        # Convert to electron counts
        # Scale clean image to desired photon level
        lambda_e = photon_level * clean + background

        # Generate Poisson noise (shot noise)
        # Note: np.random.poisson expects rate parameter λ
        poisson_noise = np.random.poisson(lambda_e).astype(np.float32)

        # Generate Gaussian read noise
        if read_noise > 0:
            gaussian_noise = np.random.normal(0, read_noise, clean.shape).astype(
                np.float32
            )
            noisy_electrons = poisson_noise + gaussian_noise
        else:
            noisy_electrons = poisson_noise

        # Ensure non-negative (physical constraint)
        noisy_electrons = np.maximum(noisy_electrons, 0)

        # Compute actual statistics for validation
        actual_mean = np.mean(lambda_e)
        actual_var_theory = np.mean(lambda_e + read_noise**2)  # Theoretical variance
        actual_var_empirical = np.var(noisy_electrons - lambda_e)  # Empirical variance

        noise_params = {
            "photon_level": photon_level,
            "read_noise": read_noise,
            "background": background,
            "mean_photons": float(actual_mean),
            "theoretical_variance": float(actual_var_theory),
            "empirical_variance": float(actual_var_empirical),
            "snr_db": float(10 * np.log10(actual_mean / np.sqrt(actual_var_theory)))
            if actual_var_theory > 0
            else float("inf"),
        }

        return noisy_electrons, noise_params

    def generate_validation_set(self) -> Dict[str, List]:
        """
        Generate complete validation dataset.

        Returns:
            Dictionary with generated data and metadata
        """
        results = {"images": [], "metadata": [], "statistics": []}

        total_images = (
            len(self.config.pattern_types)
            * len(self.config.photon_levels)
            * len(self.config.read_noise_levels)
        )

        print(f"Generating {total_images} synthetic images...")

        with tqdm(total=total_images, desc="Generating") as pbar:
            for pattern_type in self.config.pattern_types:
                for photon_level in self.config.photon_levels:
                    for read_noise in self.config.read_noise_levels:
                        # Generate clean pattern
                        clean = self.generate_pattern(
                            pattern_type, self.config.image_size
                        )

                        # Add noise
                        noisy, noise_params = self.add_poisson_gaussian_noise(
                            clean,
                            photon_level,
                            read_noise,
                            self.config.background_level,
                        )

                        # Store data
                        image_data = {
                            "clean": clean,
                            "noisy": noisy,
                            "pattern_type": pattern_type,
                            "noise_params": noise_params,
                        }

                        results["images"].append(image_data)
                        results["metadata"].append(
                            {
                                "pattern_type": pattern_type,
                                "photon_level": photon_level,
                                "read_noise": read_noise,
                                "background": self.config.background_level,
                                "image_size": self.config.image_size,
                            }
                        )
                        results["statistics"].append(noise_params)

                        pbar.update(1)

        return results

    def save_dataset(self, results: Dict[str, List]) -> None:
        """
        Save generated dataset to disk.

        Args:
            results: Generated data from generate_validation_set()
        """
        # Create subdirectories
        images_dir = self.output_dir / "images"
        metadata_dir = self.output_dir / "metadata"
        plots_dir = self.output_dir / "plots"

        for dir_path in [images_dir, metadata_dir, plots_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"Saving {len(results['images'])} images to {self.output_dir}")

        # Save individual images and metadata
        for i, (image_data, metadata) in enumerate(
            zip(results["images"], results["metadata"])
        ):
            # Create filename
            pattern = metadata["pattern_type"]
            photons = metadata["photon_level"]
            read_noise = metadata["read_noise"]
            filename = f"{pattern}_p{photons:g}_r{read_noise:g}_{i:04d}"

            # Save images as NPZ (preserves exact values)
            np.savez_compressed(
                images_dir / f"{filename}.npz",
                clean=image_data["clean"],
                noisy=image_data["noisy"],
            )

            # Save metadata as JSON
            with open(metadata_dir / f"{filename}.json", "w") as f:
                combined_metadata = {**metadata, **image_data["noise_params"]}
                json.dump(combined_metadata, f, indent=2)

        # Save summary statistics
        summary = {
            "config": asdict(self.config),
            "total_images": len(results["images"]),
            "statistics_summary": self._compute_summary_stats(results["statistics"]),
        }

        with open(self.output_dir / "dataset_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Dataset saved to {self.output_dir}")
        print(f"Total images: {len(results['images'])}")

    def _compute_summary_stats(self, statistics: List[Dict]) -> Dict:
        """Compute summary statistics across all generated images."""
        if not statistics:
            return {}

        # Extract arrays
        photon_levels = [s["photon_level"] for s in statistics]
        read_noises = [s["read_noise"] for s in statistics]
        snr_values = [s["snr_db"] for s in statistics if np.isfinite(s["snr_db"])]

        return {
            "photon_level_range": [
                float(np.min(photon_levels)),
                float(np.max(photon_levels)),
            ],
            "read_noise_range": [
                float(np.min(read_noises)),
                float(np.max(read_noises)),
            ],
            "snr_range_db": [float(np.min(snr_values)), float(np.max(snr_values))]
            if snr_values
            else [0, 0],
            "num_patterns": len(
                set(s.get("pattern_type", "unknown") for s in statistics)
            ),
        }

    def create_validation_plots(self, results: Dict[str, List]) -> None:
        """
        Create diagnostic plots for validation.

        Args:
            results: Generated data from generate_validation_set()
        """
        if not self.config.save_plots:
            return

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        print("Creating validation plots...")

        # Plot 1: Example images for each pattern type
        self._plot_pattern_examples(results, plots_dir)

        # Plot 2: Noise characteristics vs photon level
        self._plot_noise_characteristics(results, plots_dir)

        # Plot 3: SNR analysis
        self._plot_snr_analysis(results, plots_dir)

        # Plot 4: Variance validation (Poisson property)
        self._plot_variance_validation(results, plots_dir)

        print(f"Validation plots saved to {plots_dir}")

    def _plot_pattern_examples(self, results: Dict[str, List], plots_dir: Path) -> None:
        """Plot examples of each pattern type."""
        fig, axes = plt.subplots(2, len(self.config.pattern_types), figsize=(15, 6))

        pattern_examples = {}
        for image_data in results["images"]:
            pattern_type = image_data["pattern_type"]
            if pattern_type not in pattern_examples:
                pattern_examples[pattern_type] = image_data

        for i, pattern_type in enumerate(self.config.pattern_types):
            if pattern_type in pattern_examples:
                data = pattern_examples[pattern_type]

                # Clean image
                axes[0, i].imshow(data["clean"], cmap="gray", vmin=0, vmax=1)
                axes[0, i].set_title(f"{pattern_type}\n(clean)")
                axes[0, i].axis("off")

                # Noisy image
                noisy_normalized = data["noisy"] / np.max(data["noisy"])
                axes[1, i].imshow(noisy_normalized, cmap="gray")
                axes[1, i].set_title(
                    f'Noisy\n(SNR: {data["noise_params"]["snr_db"]:.1f} dB)'
                )
                axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(plots_dir / "pattern_examples.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_noise_characteristics(
        self, results: Dict[str, List], plots_dir: Path
    ) -> None:
        """Plot noise characteristics vs photon level."""
        statistics = results["statistics"]

        # Group by read noise level
        read_noise_groups = {}
        for stat in statistics:
            rn = stat["read_noise"]
            if rn not in read_noise_groups:
                read_noise_groups[rn] = {"photons": [], "snr": [], "var_ratio": []}

            read_noise_groups[rn]["photons"].append(stat["photon_level"])
            read_noise_groups[rn]["snr"].append(stat["snr_db"])

            # Variance ratio (empirical / theoretical)
            var_ratio = stat["empirical_variance"] / max(
                stat["theoretical_variance"], 1e-6
            )
            read_noise_groups[rn]["var_ratio"].append(var_ratio)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # SNR vs photon level
        for rn, data in read_noise_groups.items():
            ax1.semilogx(
                data["photons"],
                data["snr"],
                "o-",
                label=f"Read noise: {rn} e⁻",
                alpha=0.7,
            )

        ax1.set_xlabel("Photon Level (electrons)")
        ax1.set_ylabel("SNR (dB)")
        ax1.set_title("SNR vs Photon Level")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Variance validation
        for rn, data in read_noise_groups.items():
            ax2.semilogx(
                data["photons"],
                data["var_ratio"],
                "o-",
                label=f"Read noise: {rn} e⁻",
                alpha=0.7,
            )

        ax2.axhline(y=1.0, color="red", linestyle="--", label="Perfect match")
        ax2.set_xlabel("Photon Level (electrons)")
        ax2.set_ylabel("Variance Ratio (Empirical/Theoretical)")
        ax2.set_title("Poisson-Gaussian Variance Validation")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "noise_characteristics.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    def _plot_snr_analysis(self, results: Dict[str, List], plots_dir: Path) -> None:
        """Plot SNR analysis across different conditions."""
        statistics = results["statistics"]

        # Extract data
        photon_levels = [s["photon_level"] for s in statistics]
        read_noises = [s["read_noise"] for s in statistics]
        snr_values = [s["snr_db"] for s in statistics]

        # Create 2D histogram
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use log scale for photon levels
        log_photons = np.log10(photon_levels)

        # Create scatter plot with color coding
        scatter = ax.scatter(
            log_photons, read_noises, c=snr_values, cmap="viridis", s=50, alpha=0.7
        )

        ax.set_xlabel("Log₁₀(Photon Level)")
        ax.set_ylabel("Read Noise (electrons)")
        ax.set_title("SNR Landscape")

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("SNR (dB)")

        # Add contour lines for constant SNR
        unique_photons = sorted(set(photon_levels))
        unique_read_noises = sorted(set(read_noises))

        if len(unique_photons) > 1 and len(unique_read_noises) > 1:
            # Create grid for contour
            P, R = np.meshgrid(unique_photons, unique_read_noises)
            SNR_grid = 10 * np.log10(P / np.sqrt(P + R**2))

            contours = ax.contour(
                np.log10(P),
                R,
                SNR_grid,
                levels=[0, 10, 20, 30],
                colors="white",
                alpha=0.5,
            )
            ax.clabel(contours, inline=True, fontsize=8, fmt="%d dB")

        plt.tight_layout()
        plt.savefig(plots_dir / "snr_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_variance_validation(
        self, results: Dict[str, List], plots_dir: Path
    ) -> None:
        """Plot variance validation to check Poisson-Gaussian properties."""
        statistics = results["statistics"]

        theoretical_vars = [s["theoretical_variance"] for s in statistics]
        empirical_vars = [s["empirical_variance"] for s in statistics]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot: empirical vs theoretical variance
        ax1.loglog(theoretical_vars, empirical_vars, "o", alpha=0.6)

        # Perfect correlation line
        min_var = min(min(theoretical_vars), min(empirical_vars))
        max_var = max(max(theoretical_vars), max(empirical_vars))
        ax1.loglog([min_var, max_var], [min_var, max_var], "r--", label="Perfect match")

        ax1.set_xlabel("Theoretical Variance")
        ax1.set_ylabel("Empirical Variance")
        ax1.set_title("Variance Validation")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Histogram of variance ratios
        var_ratios = np.array(empirical_vars) / np.array(theoretical_vars)
        ax2.hist(var_ratios, bins=30, alpha=0.7, edgecolor="black")
        ax2.axvline(x=1.0, color="red", linestyle="--", label="Perfect match")
        ax2.axvline(
            x=np.mean(var_ratios),
            color="blue",
            linestyle="-",
            label=f"Mean: {np.mean(var_ratios):.3f}",
        )

        ax2.set_xlabel("Variance Ratio (Empirical/Theoretical)")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of Variance Ratios")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "variance_validation.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Poisson-Gaussian data"
    )
    parser.add_argument("--config", type=str, help="Configuration YAML file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--num_images", type=int, default=100, help="Number of images per condition"
    )
    parser.add_argument(
        "--image_size", type=int, default=128, help="Image size (square)"
    )
    parser.add_argument(
        "--no_plots", action="store_true", help="Skip generating validation plots"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick generation with reduced parameter space",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        config = SyntheticConfig(**config_dict)
    else:
        config = SyntheticConfig(
            output_dir=args.output_dir,
            num_images=args.num_images,
            image_size=args.image_size,
            save_plots=not args.no_plots,
        )

    # Quick mode - reduce parameter space
    if args.quick:
        config.photon_levels = [1.0, 10.0, 100.0, 1000.0]
        config.read_noise_levels = [1.0, 5.0]
        config.pattern_types = ["constant", "gradient"]
        print("Running in quick mode with reduced parameter space")

    # Generate data
    generator = SyntheticDataGenerator(config)
    results = generator.generate_validation_set()

    # Save dataset
    generator.save_dataset(results)

    # Create validation plots
    if config.save_plots:
        generator.create_validation_plots(results)

    print("\nSynthetic data generation completed successfully!")
    print(f"Generated {len(results['images'])} images")
    print(f"Output directory: {config.output_dir}")

    # Print summary statistics
    stats = results["statistics"]
    if stats:
        photon_range = [
            min(s["photon_level"] for s in stats),
            max(s["photon_level"] for s in stats),
        ]
        snr_range = [
            min(s["snr_db"] for s in stats if np.isfinite(s["snr_db"])),
            max(s["snr_db"] for s in stats if np.isfinite(s["snr_db"])),
        ]

        print(f"\nDataset characteristics:")
        print(
            f"  Photon level range: {photon_range[0]:.1f} - {photon_range[1]:.1f} electrons"
        )
        print(f"  SNR range: {snr_range[0]:.1f} - {snr_range[1]:.1f} dB")
        print(
            f"  Pattern types: {len(set(s.get('pattern_type', 'unknown') for s in stats))}"
        )


if __name__ == "__main__":
    main()
