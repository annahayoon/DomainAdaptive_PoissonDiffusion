# Add this to your PoissonGaussianGuidance class


def compute_adaptive_kappa(self, signal_level):
    """
    Compute adaptive kappa based on signal level.
    Based on theoretical analysis of PG guidance stability.
    """
    if signal_level >= 50:
        return self.kappa  # Full strength - PG should work well
    elif signal_level <= 5:
        return 0.05  # Very low strength - prevent instability
    elif signal_level <= 20:
        return 0.1 + (signal_level - 5) * 0.02  # Gradual increase: 0.1 to 0.4
    else:
        return 0.4 + (signal_level - 20) * 0.013  # Gradual increase: 0.4 to 0.8


def forward(self, x_pred, y_obs, sigma_t, **kwargs):
    """
    Forward pass with adaptive kappa scheduling.
    """
    # Estimate signal level from current prediction
    signal_estimate = self.alpha * self.s * x_pred.mean()
    signal_level = signal_estimate.item()

    # Compute adaptive kappa
    adaptive_kappa = self.compute_adaptive_kappa(signal_level)

    # Temporarily update kappa
    original_kappa = self.kappa
    self.kappa = adaptive_kappa

    # Call parent forward method
    result = super().forward(x_pred, y_obs, sigma_t, **kwargs)

    # Restore original kappa
    self.kappa = original_kappa

    return result
