"""
Debugging utilities for DAPGD

Tools for diagnosing and fixing issues during development.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SamplingDebugger:
    """
    Debug helper for sampling process

    PURPOSE: Catch and diagnose common issues during sampling

    Usage:
        debugger = SamplingDebugger()

        # In sampling loop:
        debugger.check_step(x_t, sigma_t, step_idx)
    """

    def __init__(self, check_frequency: int = 1):
        self.check_frequency = check_frequency
        self.history = []

    def check_step(
        self,
        x: torch.Tensor,
        sigma: float,
        step: int,
        denoised: Optional[torch.Tensor] = None,
        guidance_grad: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Check one sampling step for issues

        Returns True if everything looks good, False if issues detected
        """
        if step % self.check_frequency != 0:
            return True

        issues = []

        # Check for NaN or Inf
        if torch.isnan(x).any():
            issues.append(f"Step {step}: x contains NaN")
        if torch.isinf(x).any():
            issues.append(f"Step {step}: x contains Inf")

        # Check range
        if x.min() < -1.0 or x.max() > 2.0:
            issues.append(
                f"Step {step}: x range [{x.min():.3f}, {x.max():.3f}] "
                f"is unusual (expected ~[0,1])"
            )

        # Check denoised if provided
        if denoised is not None:
            if torch.isnan(denoised).any():
                issues.append(f"Step {step}: denoised contains NaN")
            if denoised.min() < -0.5 or denoised.max() > 1.5:
                issues.append(
                    f"Step {step}: denoised range [{denoised.min():.3f}, "
                    f"{denoised.max():.3f}] is unusual"
                )

        # Check guidance if provided
        if guidance_grad is not None:
            if torch.isnan(guidance_grad).any():
                issues.append(f"Step {step}: guidance contains NaN")
            grad_magnitude = guidance_grad.abs().mean().item()
            if grad_magnitude > 100:
                issues.append(
                    f"Step {step}: guidance magnitude {grad_magnitude:.2e} "
                    f"is very large (possible instability)"
                )

        # Log issues
        if issues:
            for issue in issues:
                logger.warning(issue)
            return False

        # Record statistics
        self.history.append(
            {
                "step": step,
                "sigma": sigma,
                "x_min": x.min().item(),
                "x_max": x.max().item(),
                "x_mean": x.mean().item(),
                "x_std": x.std().item(),
            }
        )

        return True

    def print_summary(self):
        """Print summary of sampling statistics"""
        if not self.history:
            logger.info("No history recorded")
            return

        logger.info("=== Sampling Summary ===")
        for entry in self.history:
            logger.info(
                f"Step {entry['step']:3d} (σ={entry['sigma']:.3f}): "
                f"range=[{entry['x_min']:.3f}, {entry['x_max']:.3f}], "
                f"mean={entry['x_mean']:.3f}, std={entry['x_std']:.3f}"
            )


def check_gradient_numerically(
    guidance: Any, x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-4
) -> Dict[str, float]:
    """
    Verify gradient computation using finite differences

    PURPOSE: Validate that analytical gradient matches numerical gradient

    Returns dictionary with error metrics
    """
    # Analytical gradient
    grad_analytical = guidance._compute_gradient(x, y)

    # Numerical gradient (slow - only check a few pixels)
    grad_numerical = torch.zeros_like(x)

    # Only check center region (faster)
    h, w = x.shape[2:4]
    h_start, w_start = h // 4, w // 4
    h_end, w_end = 3 * h // 4, 3 * w // 4

    def log_likelihood(x_test):
        """Compute log p(y|x)"""
        expected = guidance.s * x_test
        variance = guidance.s * x_test + guidance.sigma_r**2 + guidance.epsilon
        residual = y - expected
        return -0.5 * (residual**2 / variance).sum()

    count = 0
    for i in range(h_start, h_end, 4):  # Subsample
        for j in range(w_start, w_end, 4):
            x_plus = x.clone()
            x_plus[0, 0, i, j] += epsilon

            x_minus = x.clone()
            x_minus[0, 0, i, j] -= epsilon

            grad_numerical[0, 0, i, j] = (
                log_likelihood(x_plus) - log_likelihood(x_minus)
            ) / (2 * epsilon)
            count += 1

    # Compare
    mask = grad_numerical != 0
    if mask.sum() == 0:
        return {"error": float("nan"), "count": 0}

    abs_error = (grad_analytical[mask] - grad_numerical[mask]).abs()
    rel_error = abs_error / (grad_numerical[mask].abs() + 1e-10)

    return {
        "mean_abs_error": abs_error.mean().item(),
        "max_abs_error": abs_error.max().item(),
        "mean_rel_error": rel_error.mean().item(),
        "max_rel_error": rel_error.max().item(),
        "num_checked": count,
    }


def diagnose_sampling_failure(
    sampler: Any, y_e: torch.Tensor, num_steps_to_test: int = 10
) -> Dict[str, Any]:
    """
    Run diagnostic tests to identify why sampling might be failing

    PURPOSE: Automated troubleshooting

    Returns dictionary with diagnostic information
    """
    diagnostics = {}

    logger.info("Running sampling diagnostics...")

    # Test 1: Can we even initialize?
    try:
        shape = y_e.shape
        x_init = torch.randn(shape, device=y_e.device) * sampler.sigmas[0]
        diagnostics["initialization"] = "PASS"
    except Exception as e:
        diagnostics["initialization"] = f"FAIL: {str(e)}"
        return diagnostics

    # Test 2: Can we denoise one step?
    try:
        denoised = sampler.edm_wrapper.denoise(x_init, sampler.sigmas[0].item())
        diagnostics["denoising"] = "PASS"
        diagnostics["denoised_range"] = f"[{denoised.min():.3f}, {denoised.max():.3f}]"
    except Exception as e:
        diagnostics["denoising"] = f"FAIL: {str(e)}"
        return diagnostics

    # Test 3: Can we compute guidance?
    if sampler.guidance is not None:
        try:
            grad = sampler.guidance._compute_gradient(denoised.clamp(0, 1), y_e)
            diagnostics["guidance"] = "PASS"
            diagnostics["guidance_magnitude"] = f"{grad.abs().mean():.3e}"
        except Exception as e:
            diagnostics["guidance"] = f"FAIL: {str(e)}"

    # Test 4: Can we run a few steps?
    try:
        x_test = x_init.clone()
        for i in range(min(num_steps_to_test, len(sampler.sigmas) - 1)):
            t_cur = sampler.sigmas[i].item()
            t_next = sampler.sigmas[i + 1].item()

            # Simple Euler step
            denoised = sampler.edm_wrapper.denoise(x_test, t_cur)
            d_cur = (
                (x_test - denoised) / t_cur if t_cur > 0 else torch.zeros_like(x_test)
            )
            x_test = x_test + (t_next - t_cur) * d_cur

            if torch.isnan(x_test).any():
                diagnostics["sampling_steps"] = f"FAIL: NaN at step {i}"
                break
        else:
            diagnostics["sampling_steps"] = f"PASS ({num_steps_to_test} steps)"
    except Exception as e:
        diagnostics["sampling_steps"] = f"FAIL: {str(e)}"

    return diagnostics


def quick_sanity_check() -> bool:
    """
    Run quick sanity check to verify basic functionality

    Returns True if all checks pass
    """
    logger.info("Running quick sanity check...")

    try:
        # Test 1: Can we import core modules?
        from dapgd.guidance.pg_guidance import PoissonGaussianGuidance
        from dapgd.sampling.dapgd_sampler import DAPGDSampler

        logger.info("✓ Core modules import successfully")

        # Test 2: Can we create guidance?
        guidance = PoissonGaussianGuidance(s=1000, sigma_r=5.0)
        logger.info("✓ Guidance initialization works")

        # Test 3: Can we compute a gradient?
        x = torch.rand(1, 1, 8, 8)
        y = torch.rand(1, 1, 8, 8) * 1000
        grad = guidance._compute_gradient(x, y)
        assert not torch.isnan(grad).any()
        logger.info("✓ Gradient computation works")

        # Test 4: Can we apply guidance?
        x_guided = guidance(x, y, sigma_t=0.1)
        assert not torch.isnan(x_guided).any()
        assert x_guided.shape == x.shape
        logger.info("✓ Guidance application works")

        logger.info("All sanity checks passed!")
        return True

    except Exception as e:
        logger.error(f"Sanity check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run quick sanity check
    quick_sanity_check()
