#!/usr/bin/env python3
"""
Variance Model Comparison for Noise Characterization

This script rigorously tests and compares different variance models
to prove which mathematical model best describes the noise structure.

Models tested:
1. Constant (Homoscedastic): σ²(L) = σ²
2. Linear (Poisson-Gaussian): σ²(L) = σ²_read + α·L
3. Quadratic: σ²(L) = a + b·L + c·L²
4. Square-root (Poisson): σ²(L) = σ²_read + α·√L
5. Exponential: σ²(L) = a·exp(b·L)

Statistical tests:
- R² (coefficient of determination)
- AIC/BIC (information criteria)
- F-test for nested models
- Residual analysis

Usage:
    python sample/variance_model_comparison.py --num_tiles 30
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_tile_pair(noisy_path: Path, clean_dir: Path):
    """Load matching low-light and high-light tile pair."""
    noisy = torch.load(noisy_path, map_location="cpu").numpy()

    parts = noisy_path.stem.split("_")
    clean_name = "_".join(parts[:4]) + "_10s_" + "_".join(parts[5:]) + ".pt"
    clean_path = clean_dir / clean_name

    if clean_path.exists():
        clean = torch.load(clean_path, map_location="cpu")
        if isinstance(clean, dict):
            clean = clean.get("clean", clean.get("image", clean))
        clean = clean.numpy()
        return noisy, clean
    return None, None


def compute_binned_variance(L, H, n_bins=150, min_count=200):
    """Compute variance in bins for model fitting.

    Uses adaptive binning to ensure sufficient data points per bin.
    """
    # Try quantile-based binning for better distribution
    try:
        # Create bins with roughly equal number of samples
        percentiles = np.linspace(0, 100, n_bins + 1)
        L_bins = np.percentile(L, percentiles)
        # Ensure unique bin edges
        L_bins = np.unique(L_bins)
        n_bins = len(L_bins) - 1
        logger.info(f"Using quantile-based binning: {n_bins} bins")
    except:
        # Fallback to linear binning
        L_bins = np.linspace(L.min(), L.max(), n_bins + 1)
        logger.info(f"Using linear binning: {n_bins} bins")

    bin_centers = []
    bin_variances = []
    bin_counts = []
    bin_means = []

    for i in range(len(L_bins) - 1):
        mask = (L >= L_bins[i]) & (L < L_bins[i + 1])
        count = mask.sum()

        if count >= min_count:
            H_bin = H[mask]
            bin_centers.append((L_bins[i] + L_bins[i + 1]) / 2)
            bin_variances.append(np.var(H_bin))
            bin_counts.append(count)
            bin_means.append(np.mean(H_bin))

    return (
        np.array(bin_centers),
        np.array(bin_variances),
        np.array(bin_counts),
        np.array(bin_means),
    )


# Define variance models
def constant_var(L, sigma2):
    """Homoscedastic: constant variance."""
    return np.full_like(L, sigma2)


def linear_var(L, sigma2_read, alpha):
    """Poisson-Gaussian: linear variance."""
    return sigma2_read + alpha * L


def quadratic_var(L, a, b, c):
    """Quadratic variance."""
    return a + b * L + c * L**2


def sqrt_var(L, sigma2_read, alpha):
    """Square-root (pure Poisson-like)."""
    return sigma2_read + alpha * np.sqrt(np.abs(L - L.min()) + 1e-10)


def exponential_var(L, a, b):
    """Exponential variance."""
    L_scaled = (L - L.min()) / (L.max() - L.min() + 1e-10)
    return a * np.exp(b * L_scaled)


def fit_variance_model(
    model_func, bin_L, bin_var, bin_counts, p0, bounds=(-np.inf, np.inf)
):
    """Fit a variance model with weighted least squares."""
    try:
        weights = np.sqrt(bin_counts)
        popt, pcov = curve_fit(
            model_func,
            bin_L,
            bin_var,
            p0=p0,
            bounds=bounds,
            sigma=1 / weights,
            absolute_sigma=False,
            maxfev=10000,
        )

        # Compute predictions and metrics
        var_pred = model_func(bin_L, *popt)

        # R²
        ss_res = np.sum((bin_var - var_pred) ** 2)
        ss_tot = np.sum((bin_var - np.mean(bin_var)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Weighted RMSE
        rmse = np.sqrt(np.mean((bin_var - var_pred) ** 2))

        # AIC and BIC
        n = len(bin_L)
        k = len(popt)

        # Log-likelihood (assuming Gaussian errors)
        residuals = bin_var - var_pred
        sigma_resid = np.std(residuals)
        log_likelihood = (
            -n / 2 * np.log(2 * np.pi)
            - n / 2 * np.log(sigma_resid**2)
            - np.sum(residuals**2) / (2 * sigma_resid**2)
        )

        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        # Parameter uncertainties
        perr = np.sqrt(np.diag(pcov))

        return {
            "success": True,
            "params": popt,
            "param_errors": perr,
            "predictions": var_pred,
            "r2": r2,
            "rmse": rmse,
            "aic": aic,
            "bic": bic,
            "log_likelihood": log_likelihood,
            "n_params": k,
        }
    except Exception as e:
        logger.warning(f"Model fit failed: {e}")
        return {"success": False, "error": str(e)}


def compare_variance_models(num_tiles=30):
    """Compare different variance models."""

    noisy_dir = Path("dataset/processed/pt_tiles/photography/noisy")
    clean_dir = Path("dataset/processed/pt_tiles/photography/clean")
    output_dir = Path("results/noise_model_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    noisy_files = sorted(noisy_dir.glob("*.pt"))[:num_tiles]

    all_L = []
    all_H = []

    logger.info(f"Loading {len(noisy_files)} tile pairs...")
    for noisy_path in noisy_files:
        L_tile, H_tile = load_tile_pair(noisy_path, clean_dir)
        if L_tile is not None and H_tile is not None:
            all_L.append(L_tile.flatten())
            all_H.append(H_tile.flatten())

    L = np.concatenate(all_L)
    H = np.concatenate(all_H)

    logger.info(f"\n{'='*70}")
    logger.info(f"VARIANCE MODEL COMPARISON")
    logger.info(f"{'='*70}")
    logger.info(f"Total pixels before filtering: {len(L):,}")
    logger.info(f"L range before filtering: [{L.min():.6f}, {L.max():.6f}]")
    logger.info(f"H range before filtering: [{H.min():.6f}, {H.max():.6f}]")

    # Remove outliers: filter out extreme values and data near L=0
    # Keep data in reasonable range (MORE AGGRESSIVE FILTERING)
    L_threshold_low = 0.1  # Remove data too close to 0 (increased from 0.05)
    percentile_high = 99.0  # Remove top 1% outliers (increased from 99.5%)
    percentile_low = 1.0  # Remove bottom 1% outliers

    L_max_threshold = np.percentile(np.abs(L), percentile_high)
    L_min_threshold = np.percentile(np.abs(L), percentile_low)

    # Also remove based on H outliers
    H_max_threshold = np.percentile(np.abs(H), percentile_high)
    H_min_threshold = np.percentile(np.abs(H), percentile_low)

    mask = (
        (np.abs(L) > max(L_threshold_low, L_min_threshold))
        & (np.abs(L) < L_max_threshold)
        & (np.abs(H) > H_min_threshold)
        & (np.abs(H) < H_max_threshold)
    )

    L = L[mask]
    H = H[mask]

    logger.info(f"\nAfter removing outliers:")
    logger.info(
        f"Removed {(~mask).sum():,} pixels ({100*(~mask).sum()/len(mask):.2f}%)"
    )
    logger.info(f"Remaining pixels: {len(L):,}")
    logger.info(f"L range: [{L.min():.6f}, {L.max():.6f}]")
    logger.info(f"H range: [{H.min():.6f}, {H.max():.6f}]")

    # First, fit linear mean to get residuals
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    reg.fit(L.reshape(-1, 1), H)
    H_pred = reg.predict(L.reshape(-1, 1))
    residuals = H - H_pred

    a, b = reg.coef_[0], reg.intercept_
    logger.info(f"\nLinear mean: H = {a:.6f}·L + {b:.6f}")
    logger.info(f"R² = {reg.score(L.reshape(-1, 1), H):.6f}")

    # Compute binned variance with many bins for smoother appearance
    logger.info(f"\nComputing binned variances with fine granularity...")
    bin_L, bin_var, bin_counts, bin_means = compute_binned_variance(
        L, H, n_bins=150, min_count=200
    )
    logger.info(f"Number of bins: {len(bin_L)}")
    logger.info(f"Pixels per bin: {bin_counts.min():,} to {bin_counts.max():,}")

    # Fit all models
    models = {}

    logger.info(f"\n{'='*70}")
    logger.info("FITTING VARIANCE MODELS")
    logger.info(f"{'='*70}")

    # 1. Constant (Homoscedastic)
    logger.info("\n1. Constant (Homoscedastic): σ²(L) = σ²")
    const_var = np.var(residuals)
    models["constant"] = fit_variance_model(
        constant_var, bin_L, bin_var, bin_counts, p0=[const_var], bounds=(0, np.inf)
    )
    if models["constant"]["success"]:
        logger.info(f"   σ² = {models['constant']['params'][0]:.6f}")
        logger.info(f"   R² = {models['constant']['r2']:.6f}")
        logger.info(f"   AIC = {models['constant']['aic']:.2f}")
        logger.info(f"   BIC = {models['constant']['bic']:.2f}")

    # 2. Linear (Poisson-Gaussian)
    logger.info("\n2. Linear (Poisson-Gaussian): σ²(L) = σ²_read + α·L")
    models["linear"] = fit_variance_model(
        linear_var, bin_L, bin_var, bin_counts, p0=[const_var, 0.0], bounds=(0, np.inf)
    )
    if models["linear"]["success"]:
        sigma2_read, alpha = models["linear"]["params"]
        err_read, err_alpha = models["linear"]["param_errors"]
        logger.info(f"   σ²_read = {sigma2_read:.6f} ± {err_read:.6f}")
        logger.info(f"   α = {alpha:.6f} ± {err_alpha:.6f}")
        logger.info(f"   R² = {models['linear']['r2']:.6f}")
        logger.info(f"   AIC = {models['linear']['aic']:.2f}")
        logger.info(f"   BIC = {models['linear']['bic']:.2f}")

        # Statistical significance of α
        t_stat = alpha / err_alpha if err_alpha > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(bin_L) - 2))
        logger.info(f"   α significance: t = {t_stat:.2f}, p = {p_value:.6f}")

    # 3. Quadratic
    logger.info("\n3. Quadratic: σ²(L) = a + b·L + c·L²")
    models["quadratic"] = fit_variance_model(
        quadratic_var, bin_L, bin_var, bin_counts, p0=[const_var, 0.0, 0.0]
    )
    if models["quadratic"]["success"]:
        a, b, c = models["quadratic"]["params"]
        logger.info(f"   a = {a:.6f}, b = {b:.6f}, c = {c:.6f}")
        logger.info(f"   R² = {models['quadratic']['r2']:.6f}")
        logger.info(f"   AIC = {models['quadratic']['aic']:.2f}")
        logger.info(f"   BIC = {models['quadratic']['bic']:.2f}")

    # 4. Square-root (Poisson)
    logger.info("\n4. Square-root (Pure Poisson): σ²(L) = σ²_read + α·√L")
    models["sqrt"] = fit_variance_model(
        sqrt_var, bin_L, bin_var, bin_counts, p0=[const_var, 0.0], bounds=(0, np.inf)
    )
    if models["sqrt"]["success"]:
        logger.info(f"   σ²_read = {models['sqrt']['params'][0]:.6f}")
        logger.info(f"   α = {models['sqrt']['params'][1]:.6f}")
        logger.info(f"   R² = {models['sqrt']['r2']:.6f}")
        logger.info(f"   AIC = {models['sqrt']['aic']:.2f}")
        logger.info(f"   BIC = {models['sqrt']['bic']:.2f}")

    # 5. Exponential
    logger.info("\n5. Exponential: σ²(L) = a·exp(b·L)")
    models["exponential"] = fit_variance_model(
        exponential_var, bin_L, bin_var, bin_counts, p0=[const_var, 0.0]
    )
    if models["exponential"]["success"]:
        logger.info(f"   a = {models['exponential']['params'][0]:.6f}")
        logger.info(f"   b = {models['exponential']['params'][1]:.6f}")
        logger.info(f"   R² = {models['exponential']['r2']:.6f}")
        logger.info(f"   AIC = {models['exponential']['aic']:.2f}")
        logger.info(f"   BIC = {models['exponential']['bic']:.2f}")

    # Model comparison
    logger.info(f"\n{'='*70}")
    logger.info("MODEL COMPARISON")
    logger.info(f"{'='*70}")

    # Sort by AIC
    successful_models = {name: m for name, m in models.items() if m["success"]}
    sorted_by_aic = sorted(successful_models.items(), key=lambda x: x[1]["aic"])
    sorted_by_bic = sorted(successful_models.items(), key=lambda x: x[1]["bic"])
    sorted_by_r2 = sorted(
        successful_models.items(), key=lambda x: x[1]["r2"], reverse=True
    )

    logger.info("\nBy AIC (lower is better):")
    for i, (name, model) in enumerate(sorted_by_aic):
        delta_aic = model["aic"] - sorted_by_aic[0][1]["aic"]
        logger.info(
            f"  {i+1}. {name:15s}  AIC = {model['aic']:8.2f}  (Δ = {delta_aic:6.2f})"
        )

    logger.info("\nBy BIC (lower is better):")
    for i, (name, model) in enumerate(sorted_by_bic):
        delta_bic = model["bic"] - sorted_by_bic[0][1]["bic"]
        logger.info(
            f"  {i+1}. {name:15s}  BIC = {model['bic']:8.2f}  (Δ = {delta_bic:6.2f})"
        )

    logger.info("\nBy R² (higher is better):")
    for i, (name, model) in enumerate(sorted_by_r2):
        logger.info(f"  {i+1}. {name:15s}  R² = {model['r2']:.6f}")

    # F-test: Linear vs Constant (nested models)
    if models["constant"]["success"] and models["linear"]["success"]:
        logger.info(f"\n{'='*70}")
        logger.info("F-TEST: Linear vs Constant (nested models)")
        logger.info(f"{'='*70}")

        n = len(bin_L)

        # Calculate F-statistic
        ss_res_const = np.sum((bin_var - models["constant"]["predictions"]) ** 2)
        ss_res_lin = np.sum((bin_var - models["linear"]["predictions"]) ** 2)

        df_const = n - 1
        df_lin = n - 2

        F = ((ss_res_const - ss_res_lin) / 1) / (ss_res_lin / df_lin)
        p_value_F = 1 - stats.f.cdf(F, 1, df_lin)

        logger.info(f"H₀: Constant variance is sufficient")
        logger.info(f"H₁: Linear variance is better")
        logger.info(f"F-statistic = {F:.4f}")
        logger.info(f"p-value = {p_value_F:.6f}")

        if p_value_F < 0.05:
            logger.info(f"✓ REJECT H₀: Linear model significantly better (p < 0.05)")
        else:
            logger.info(f"✗ FAIL TO REJECT H₀: Constant model sufficient (p ≥ 0.05)")

    # Best model determination
    best_model_name = sorted_by_aic[0][0]
    logger.info(f"\n{'='*70}")
    logger.info(f"BEST MODEL: {best_model_name.upper()}")
    logger.info(f"{'='*70}")

    # Create comprehensive visualization
    create_variance_comparison_viz(
        bin_L,
        bin_var,
        bin_counts,
        models,
        best_model_name,
        L,
        output_dir / "variance_model_comparison.png",
    )

    # Save results
    results = {
        "n_tiles": len(all_L),
        "n_pixels": int(len(L)),
        "n_bins": int(len(bin_L)),
        "models": {},
    }

    for name, model in models.items():
        if model["success"]:
            results["models"][name] = {
                "params": [float(p) for p in model["params"]],
                "param_errors": [float(e) for e in model["param_errors"]],
                "r2": float(model["r2"]),
                "rmse": float(model["rmse"]),
                "aic": float(model["aic"]),
                "bic": float(model["bic"]),
                "n_params": int(model["n_params"]),
            }

    results["best_model"] = {
        "by_aic": sorted_by_aic[0][0],
        "by_bic": sorted_by_bic[0][0],
        "by_r2": sorted_by_r2[0][0],
    }

    with open(output_dir / "variance_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {output_dir}/variance_comparison_results.json")

    return models, bin_L, bin_var, bin_counts


def create_variance_comparison_viz(
    bin_L, bin_var, bin_counts, models, best_model, L, output_path
):
    """Create comprehensive variance model comparison visualization."""

    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(3, 6, hspace=0.4, wspace=0.35)

    # Color scheme
    colors = {
        "constant": "#1f77b4",
        "linear": "#d62728",
        "quadratic": "#ff7f0e",
        "sqrt": "#2ca02c",
        "exponential": "#9467bd",
    }

    labels = {
        "constant": "Constant",
        "linear": "Linear (P-G)",
        "quadratic": "Quadratic",
        "sqrt": "Sqrt (Poisson)",
        "exponential": "Exponential",
    }

    # 1. Main plot: All models overlaid (top left, 2 rows x 3 cols)
    ax1 = fig.add_subplot(gs[0:2, 0:3])

    # Plot empirical variance as connected line with small markers
    ax1.plot(
        bin_L,
        bin_var,
        "o-",
        color="black",
        markersize=4,
        linewidth=1.5,
        alpha=0.7,
        label="Empirical Var($H|L$)",
        zorder=5,
    )

    # Add subtle shading to show data density (lighter approach)
    # Normalize counts for color intensity
    norm_counts = (bin_counts - bin_counts.min()) / (
        bin_counts.max() - bin_counts.min()
    )
    for i in range(len(bin_L) - 1):
        ax1.fill_between(
            [bin_L[i], bin_L[i + 1]],
            [bin_var[i], bin_var[i + 1]],
            alpha=0.1 + 0.2 * norm_counts[i],
            color="gray",
            zorder=1,
        )

    # Plot all fitted models
    L_plot = np.linspace(bin_L.min(), bin_L.max(), 300)

    for name, model in models.items():
        if model["success"]:
            if name == "constant":
                var_plot = constant_var(L_plot, *model["params"])
            elif name == "linear":
                var_plot = linear_var(L_plot, *model["params"])
            elif name == "quadratic":
                var_plot = quadratic_var(L_plot, *model["params"])
            elif name == "sqrt":
                var_plot = sqrt_var(L_plot, *model["params"])
            elif name == "exponential":
                var_plot = exponential_var(L_plot, *model["params"])

            linestyle = "-" if name == best_model else "--"
            linewidth = 5 if name == best_model else 3
            alpha = 1.0 if name == best_model else 0.7

            label_str = f"{labels[name]}"
            if name == best_model:
                label_str = f"⭐ {label_str} (BEST, R²={model['r2']:.3f})"
            else:
                label_str = f"{label_str} (R²={model['r2']:.3f})"

            ax1.plot(
                L_plot,
                var_plot,
                linestyle=linestyle,
                linewidth=linewidth,
                color=colors[name],
                label=label_str,
                alpha=alpha,
            )

    ax1.set_xlabel("Low-Light Intensity ($L$)", fontsize=15, fontweight="bold")
    ax1.set_ylabel("Var($H|L$)", fontsize=15, fontweight="bold")
    ax1.set_title(
        "(a) Variance Models: Empirical vs Fitted",
        fontsize=17,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(
        fontsize=11, loc="best", framealpha=0.95, edgecolor="black", fancybox=False
    )
    ax1.grid(True, alpha=0.25, linewidth=1)
    ax1.tick_params(labelsize=12)

    # Add data point count annotation
    ax1.text(
        0.02,
        0.98,
        f"$n$ = {len(bin_L)} bins\nTotal pixels: {len(L):,}",
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="wheat",
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        ),
    )

    # 2. Residual plots - ALL IN ONE ROW with SAME Y-SCALE
    successful_models = [(n, m) for n, m in models.items() if m["success"]]

    # Calculate global y-limits for residuals
    all_residuals = []
    for name, model in successful_models:
        residuals = bin_var - model["predictions"]
        all_residuals.extend(residuals)

    if len(all_residuals) > 0:
        y_margin = 0.1
        y_min = np.min(all_residuals)
        y_max = np.max(all_residuals)
        y_range = y_max - y_min
        ylim = [y_min - y_margin * y_range, y_max + y_margin * y_range]
    else:
        ylim = [-0.1, 0.1]

    # Plot residuals for each model in the same row
    for idx, (name, model) in enumerate(successful_models[:5]):  # Up to 5 models
        ax = (
            fig.add_subplot(gs[0, 3 + idx])
            if idx < 3
            else fig.add_subplot(gs[1, idx - 3 + 3])
        )

        residuals = bin_var - model["predictions"]
        # Use connected line with small markers instead of scattered points
        ax.plot(
            bin_L,
            residuals,
            "o-",
            markersize=3,
            linewidth=1,
            alpha=0.8,
            color=colors[name],
            zorder=5,
        )
        ax.axhline(0, color="red", linestyle="--", linewidth=2.5, alpha=0.8)
        ax.fill_between(
            [bin_L.min(), bin_L.max()],
            -2 * np.std(residuals),
            2 * np.std(residuals),
            color="gray",
            alpha=0.15,
            zorder=1,
        )

        ax.set_xlabel("$L$", fontsize=12, fontweight="bold")
        if idx == 0 or idx == 3:  # Only leftmost plots get y-label
            ax.set_ylabel("Residuals", fontsize=12, fontweight="bold")
        ax.set_title(f"{labels[name]}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.tick_params(labelsize=10)
        ax.set_ylim(ylim)  # Same y-scale for all

        # Add RMSE annotation
        rmse = np.sqrt(np.mean(residuals**2))
        ax.text(
            0.95,
            0.95,
            f"RMSE:\n{rmse:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            ),
        )

        # Highlight best model
        if name == best_model:
            for spine in ax.spines.values():
                spine.set_edgecolor("gold")
                spine.set_linewidth(4)

    # 3. Model comparison bar charts (bottom row)
    ax_aic = fig.add_subplot(gs[2, 0:2])
    ax_bic = fig.add_subplot(gs[2, 2:4])
    ax_r2 = fig.add_subplot(gs[2, 4:6])

    model_names = [labels[n] for n, m in models.items() if m["success"]]
    model_colors = [colors[n] for n in models.keys() if models[n]["success"]]

    # AIC comparison
    aics = [m["aic"] for m in models.values() if m["success"]]
    bars = ax_aic.barh(
        model_names,
        aics,
        color=model_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax_aic.set_xlabel("AIC (lower is better)", fontsize=13, fontweight="bold")
    ax_aic.set_title("(b) AIC Comparison", fontsize=14, fontweight="bold")
    ax_aic.grid(True, alpha=0.25, axis="x", linewidth=0.8)
    ax_aic.tick_params(labelsize=11)

    min_aic_idx = np.argmin(aics)
    bars[min_aic_idx].set_edgecolor("gold")
    bars[min_aic_idx].set_linewidth(4)

    # BIC comparison
    bics = [m["bic"] for m in models.values() if m["success"]]
    bars = ax_bic.barh(
        model_names,
        bics,
        color=model_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax_bic.set_xlabel("BIC (lower is better)", fontsize=13, fontweight="bold")
    ax_bic.set_title("(c) BIC Comparison", fontsize=14, fontweight="bold")
    ax_bic.grid(True, alpha=0.25, axis="x", linewidth=0.8)
    ax_bic.tick_params(labelsize=11)

    min_bic_idx = np.argmin(bics)
    bars[min_bic_idx].set_edgecolor("gold")
    bars[min_bic_idx].set_linewidth(4)

    # R² comparison
    r2s = [m["r2"] for m in models.values() if m["success"]]
    bars = ax_r2.barh(
        model_names,
        r2s,
        color=model_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax_r2.set_xlabel("R² (higher is better)", fontsize=13, fontweight="bold")
    ax_r2.set_title("(d) R² Comparison", fontsize=14, fontweight="bold")
    ax_r2.grid(True, alpha=0.25, axis="x", linewidth=0.8)
    ax_r2.tick_params(labelsize=11)

    max_r2_idx = np.argmax(r2s)
    bars[max_r2_idx].set_edgecolor("gold")
    bars[max_r2_idx].set_linewidth(4)

    plt.suptitle(
        "Rigorous Variance Model Comparison\nStatistical Evidence for Noise Structure",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare variance models")
    parser.add_argument("--num_tiles", type=int, default=30)
    args = parser.parse_args()

    compare_variance_models(args.num_tiles)

    logger.info(f"\n{'='*70}")
    logger.info("✓ VARIANCE MODEL COMPARISON COMPLETE!")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
