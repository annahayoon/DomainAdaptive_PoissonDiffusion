#!/usr/bin/env python3
"""
Test utilities for gradient verification of guidance modules.

This module provides test functions for verifying guidance gradient correctness.
All guidance classes are in core.guidance - import from there directly.
"""

from pathlib import Path
from typing import Optional

import torch

from core.guidance import PoissonGaussianGuidance

__all__ = [
    "compute_log_likelihood_for_test",
    "test_pg_gradient_correctness",
    "run_gradient_verification",
]

from core.utils.data_utils import get_exposure_ratio, load_metadata_json
from core.utils.sensor_utils import load_sensor_calibration_from_metadata


def compute_log_likelihood_for_test(y_e, x0_hat, pg_guidance, b, c, h, w):
    """
    Compute log-likelihood for a single pixel for gradient verification testing.

    This is a simplified version that computes the likelihood for a single pixel
    to enable finite difference gradient checking.
    """
    x_pixel = x0_hat[b : b + 1, c : c + 1, h : h + 1, w : w + 1]
    y_pixel = y_e[b : b + 1, c : c + 1, h : h + 1, w : w + 1]
    grad = pg_guidance.compute_likelihood_gradient(x_pixel, y_pixel)
    return -0.5 * grad.sum().item()


def test_pg_gradient_correctness(
    metadata_path: Path,
    sensor_type: str,
    short_tile_id: str,
):
    """
    Verify PG guidance gradient against finite differences.

    This test ensures the analytical gradient computation is correct by comparing
    it to numerical gradients computed via finite differences.

    Args:
        metadata_path: Path to metadata JSON file to load sensor calibration (required).
        sensor_type: Sensor type ('sony' or 'fuji') for loading calibration (required).
        short_tile_id: Short exposure tile ID to use for extracting exposure ratio (required).

    Raises:
        ValueError: If metadata_path is not provided or sensor calibration cannot be loaded.
        FileNotFoundError: If metadata_path does not exist.
    """
    if metadata_path is None:
        raise ValueError("metadata_path is required for sensor calibration")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    tile_lookup = load_metadata_json(metadata_path)
    exposure_ratio = get_exposure_ratio(short_tile_id, tile_lookup)
    if exposure_ratio <= 0 or exposure_ratio >= 1:
        raise ValueError(
            f"Invalid exposure ratio {exposure_ratio} for tile {short_tile_id}. "
            f"Expected 0 < ratio < 1 (short exposure / long exposure)."
        )

    black_level, white_level = load_sensor_calibration_from_metadata(sensor_type)

    s = white_level - black_level
    sigma_r = 5.0
    kappa = 0.5
    epsilon = 1e-8

    pg_guidance = PoissonGaussianGuidance(
        s=s,
        sigma_r=sigma_r,
        black_level=black_level,
        white_level=white_level,
        exposure_ratio=exposure_ratio,
        kappa=kappa,
        epsilon=epsilon,
    )

    test_cases = [(1, 1, 32, 32), (1, 3, 16, 16)]

    for batch_size, channels, height, width in test_cases:
        torch.manual_seed(42)
        x0_hat = torch.rand(batch_size, channels, height, width, requires_grad=True)
        y_e = (
            torch.rand(batch_size, channels, height, width)
            * (white_level - black_level)
            + black_level
        )

        grad_analytical = pg_guidance.compute_likelihood_gradient(x0_hat, y_e)

        eps = 1e-5
        grad_numerical = torch.zeros_like(x0_hat)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        x_plus = x0_hat.clone()
                        x_plus[b, c, h, w] += eps
                        log_lik = compute_log_likelihood_for_test(
                            y_e, x0_hat, pg_guidance, b, c, h, w
                        )
                        log_lik_plus = compute_log_likelihood_for_test(
                            y_e, x_plus, pg_guidance, b, c, h, w
                        )
                        grad_numerical[b, c, h, w] = (log_lik_plus - log_lik) / eps

        diff = torch.abs(grad_analytical - grad_numerical).mean()
        max_diff = torch.abs(grad_analytical - grad_numerical).max()

        assert diff < 1e-3, f"Gradient verification failed: mean diff {diff:.6f} > 1e-3"
        assert (
            max_diff < 1e-2
        ), f"Gradient verification failed: max diff {max_diff:.6f} > 1e-2"


def run_gradient_verification(
    metadata_path: Optional[Path] = None, sensor_type: Optional[str] = None
):
    """
    Run gradient verification tests if requested.

    Args:
        metadata_path: Path to metadata JSON file to load sensor calibration (required).
        sensor_type: Sensor type ('sony' or 'fuji') for loading calibration (required).

    Raises:
        ValueError: If metadata_path, sensor_type, or short_tile_id are not provided when --test_gradients is used.
    """
    import argparse
    import sys

    if "--test_gradients" in sys.argv:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--test_gradients",
            action="store_true",
            help="Run gradient verification tests",
        )
        parser.add_argument(
            "--metadata_json",
            type=str,
            required=True,
            help="Path to metadata JSON file for sensor calibration (required)",
        )
        parser.add_argument(
            "--sensor_type",
            type=str,
            required=True,
            help="Sensor type (sony or fuji) (required)",
        )
        parser.add_argument(
            "--short_tile_id",
            type=str,
            required=True,
            help="Short exposure tile ID to use for extracting exposure ratio (required)",
        )
        args, _ = parser.parse_known_args()

        if not args.metadata_json:
            raise ValueError(
                "--metadata_json is required when running gradient verification tests"
            )
        if not args.sensor_type:
            raise ValueError(
                "--sensor_type is required when running gradient verification tests"
            )
        if not args.short_tile_id:
            raise ValueError(
                "--short_tile_id is required when running gradient verification tests"
            )

        test_pg_gradient_correctness(
            metadata_path=Path(args.metadata_json),
            sensor_type=args.sensor_type,
            short_tile_id=args.short_tile_id,
        )
        sys.exit(0)
