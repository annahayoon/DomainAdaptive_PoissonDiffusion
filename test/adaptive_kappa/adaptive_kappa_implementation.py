#!/usr/bin/env python3
"""
Adaptive Kappa Implementation for PG Guidance

This module provides a drop-in replacement for the PoissonGaussianGuidance class
with adaptive kappa scheduling based on signal level.
"""

import numpy as np
import torch
import torch.nn as nn


class AdaptiveKappaPGGuidance(nn.Module):
    """
    PG Guidance with adaptive kappa scheduling.

    This class extends the standard PG guidance with signal-dependent kappa
    to improve stability in ultra-low light regions.
    """

    def __init__(
        self,
        s: float,
        sigma_r: float,
        domain_min: float,
        domain_max: float,
        offset: float = 0.0,
        exposure_ratio: float = 1.0,
        kappa: float = 0.8,  # Base kappa
        tau: float = 0.01,
        mode: str = "wls",
        epsilon: float = 1e-8,
        guidance_level: str = "x0",
        # Adaptive kappa parameters
        use_adaptive_kappa: bool = True,
        min_kappa: float = 0.05,
        max_kappa: float = 2.0,
        signal_threshold: float = 50.0,
    ):
        super().__init__()

        # Store original parameters
        self.s = s
        self.sigma_r = sigma_r
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.offset = offset
        self.alpha = exposure_ratio
        self.base_kappa = kappa  # Store base kappa
        self.kappa = kappa  # Current kappa (will be modified)
        self.tau = tau
        self.mode = mode
        self.epsilon = epsilon
        self.guidance_level = guidance_level

        # Adaptive kappa parameters
        self.use_adaptive_kappa = use_adaptive_kappa
        self.min_kappa = min_kappa
        self.max_kappa = max_kappa
        self.signal_threshold = signal_threshold

        print(f"AdaptiveKappaPGGuidance initialized:")
        print(f"  Base kappa: {self.base_kappa}")
        print(f"  Adaptive kappa: {'enabled' if use_adaptive_kappa else 'disabled'}")
        print(f"  Kappa range: [{self.min_kappa}, {self.max_kappa}]")
        print(f"  Signal threshold: {self.signal_threshold} electrons")

    def compute_adaptive_kappa(self, signal_level):
        """
        Compute adaptive kappa based on signal level.

        Strategy based on theoretical analysis:
        - Ultra-low signal (<5 e⁻): κ = 0.05 (minimum for stability)
        - Low signal (5-20 e⁻): κ = 0.1-0.4 (gradual increase)
        - Medium signal (20-50 e⁻): κ = 0.4-0.8 (approaching full strength)
        - Normal signal (≥50 e⁻): κ = 0.8 (full strength)
        """
        if not self.use_adaptive_kappa:
            return self.base_kappa

        if signal_level >= self.signal_threshold:
            return self.base_kappa  # Full strength - PG should work well
        elif signal_level <= 5:
            return self.min_kappa  # Very low strength - prevent instability
        elif signal_level <= 20:
            # Gradual increase: 0.05 to 0.4
            ratio = (signal_level - 5) / 15
            return self.min_kappa + (0.4 - self.min_kappa) * ratio
        else:
            # Gradual increase: 0.4 to base_kappa
            ratio = (signal_level - 20) / (self.signal_threshold - 20)
            return 0.4 + (self.base_kappa - 0.4) * ratio

    def estimate_signal_level(self, x_pred):
        """
        Estimate signal level from current prediction.

        Args:
            x_pred: Current prediction [B, C, H, W] in [0,1]

        Returns:
            Estimated signal level in electrons
        """
        # Convert to physical units to estimate signal
        signal_estimate = self.alpha * self.s * x_pred.mean()
        return signal_estimate.item()

    def forward(self, x_pred, y_obs, sigma_t, **kwargs):
        """
        Forward pass with adaptive kappa scheduling.

        Args:
            x_pred: Current prediction [B, C, H, W] in [0,1]
            y_obs: Observed data [B, C, H, W] in physical units
            sigma_t: Current noise level
            **kwargs: Additional arguments

        Returns:
            Guidance gradient [B, C, H, W]
        """
        if self.use_adaptive_kappa:
            # Estimate signal level
            signal_level = self.estimate_signal_level(x_pred)

            # Compute adaptive kappa
            adaptive_kappa = self.compute_adaptive_kappa(signal_level)

            # Temporarily update kappa
            original_kappa = self.kappa
            self.kappa = adaptive_kappa

            # Store signal level for debugging
            if hasattr(self, "debug_info"):
                self.debug_info["signal_level"] = signal_level
                self.debug_info["adaptive_kappa"] = adaptive_kappa

        # Call the standard PG guidance computation
        # (This would be the actual PG guidance forward pass)
        result = self._compute_pg_guidance(x_pred, y_obs, sigma_t, **kwargs)

        if self.use_adaptive_kappa:
            # Restore original kappa
            self.kappa = original_kappa

        return result

    def _compute_pg_guidance(self, x_pred, y_obs, sigma_t, **kwargs):
        """
        Compute PG guidance gradient.

        This is a placeholder - you would integrate this with your actual
        PG guidance implementation.
        """
        # Placeholder implementation
        # In practice, this would call your existing PG guidance computation
        # with the adaptive kappa value

        # For now, return a zero gradient as placeholder
        return torch.zeros_like(x_pred)

    def get_debug_info(self):
        """
        Get debugging information about adaptive kappa usage.
        """
        if hasattr(self, "debug_info"):
            return self.debug_info
        return {}


def test_adaptive_kappa():
    """
    Test the adaptive kappa implementation.
    """
    print("=" * 80)
    print("ADAPTIVE KAPPA IMPLEMENTATION TEST")
    print("=" * 80)

    # Create adaptive PG guidance
    guidance = AdaptiveKappaPGGuidance(
        s=15871.0,  # Fuji domain range
        sigma_r=4.0,
        domain_min=0.0,
        domain_max=15871.0,
        kappa=0.8,  # Base kappa
        exposure_ratio=0.04,
        use_adaptive_kappa=True,
    )

    # Test different signal levels
    test_signals = [1, 5, 10, 15, 20, 30, 40, 50, 100]

    print("Signal Level -> Adaptive Kappa Mapping:")
    print("Signal (e⁻) | Adaptive κ | Strategy")
    print("-" * 40)

    for signal in test_signals:
        adaptive_kappa = guidance.compute_adaptive_kappa(signal)

        if signal < 5:
            strategy = "Ultra-low (min κ)"
        elif signal < 20:
            strategy = "Low (gradual κ)"
        elif signal < 50:
            strategy = "Medium (approaching full κ)"
        else:
            strategy = "Normal (full κ)"

        print(f"{signal:8.0f} | {adaptive_kappa:8.3f} | {strategy}")

    print("\nImplementation ready for integration!")
    print("Next steps:")
    print("1. Integrate with your existing PG guidance class")
    print("2. Test on low-signal photography tiles")
    print("3. Compare with constant kappa baseline")


if __name__ == "__main__":
    test_adaptive_kappa()
