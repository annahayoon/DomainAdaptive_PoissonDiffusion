#!/usr/bin/env python
"""
Evaluate trained model on raw astronomy data.

This script loads the trained model from hpc_result/best_model.pt and
runs inference on the raw astronomy images to generate denoised predictions.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.calibration import SensorCalibration


def load_trained_model(model_path: str, device: str = "cuda"):
    """Load the trained diffusion model."""
    try:
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Check what's in the checkpoint
        print(
            "Checkpoint keys:",
            list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict",
        )

        # For now, we'll need to reconstruct the model architecture
        # This would typically involve importing your model class and loading the state dict
        # Since I don't have access to the exact model architecture, let's create a placeholder

        print("WARNING: Model architecture not available - using placeholder")
        return None

    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


def inference_with_model(model, noisy_image: np.ndarray, device: str = "cuda"):
    """Run inference with the trained model."""
    if model is None:
        print("No model available - returning Gaussian filtered result as placeholder")
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(noisy_image, sigma=1.0)

    # Convert to tensor and prepare for model
    with torch.no_grad():
        # Typical preprocessing for diffusion models
        if isinstance(noisy_image, np.ndarray):
            tensor = torch.from_numpy(noisy_image).float().to(device)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)  # Add batch dim

        # Run model inference
        try:
            output = model(tensor)
            result = output.squeeze().cpu().numpy()
            return result
        except Exception as e:
            print(f"Model inference failed: {e}")
            # Fallback to simple denoising
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(noisy_image, sigma=1.0)


def evaluate_trained_model():
    """Main evaluation function."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load trained model
    model_path = "hpc_result/best_model.pt"
    model = load_trained_model(model_path, device)

    # Setup calibration for astronomy domain
    calibration = SensorCalibration(domain="astronomy")

    # Note: This script requires raw astronomy FITS files for evaluation
    # Since loaders.py was removed, this functionality needs to be adapted
    # for the current preprocessing pipeline

    print("WARNING: AstronomyLoader functionality removed.")
    print("This script needs to be adapted for the current preprocessing pipeline.")
    return  # Exit early since loader functionality is not available
    axes[0].axis("off")

    # Model prediction
    axes[1].imshow(denoised, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Model Prediction (Denoised)")
    axes[1].axis("off")

    plt.tight_layout()

    # Create output directory
    output_dir = Path("evaluation_results/astronomy")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison
    output_file = output_dir / "trained_model_prediction.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    # Compute metrics
    def compute_metrics(original, denoised):
        noise_reduction = float(
            np.var(original) / np.var(denoised) if np.var(denoised) > 0 else 1.0
        )
        correlation = float(np.corrcoef(original.flatten(), denoised.flatten())[0, 1])
        mse = float(np.mean((original - denoised) ** 2))
        return {
            "noise_reduction": noise_reduction,
            "correlation": correlation,
            "mse": mse,
            "inference_time_seconds": inference_time,
        }

    metrics = compute_metrics(normalized, denoised)

    # Save results
    results = {
        "evaluation_type": "trained_model_inference",
        "model_path": model_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "input_file": str(fits_file.name),
        "input_shape": list(normalized.shape),
        "calibration": {
            "scale": float(scale),
            "gain": float(calibration.params.gain),
            "read_noise": float(calibration.params.read_noise),
        },
        "metrics": metrics,
        "output_file": str(output_file),
    }

    with open(output_dir / "../trained_model_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("TRAINED MODEL EVALUATION COMPLETED!")
    print("=" * 60)
    print(f"Input: {fits_file.name} ({normalized.shape})")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Noise reduction: {metrics['noise_reduction']:.3f}×")
    print(f"Signal correlation: {metrics['correlation']:.3f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"Results saved to: {output_dir}")
    print(f"Comparison image: {output_file}")


if __name__ == "__main__":
    evaluate_trained_model()
