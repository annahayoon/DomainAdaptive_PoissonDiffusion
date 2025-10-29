#!/usr/bin/env python3
"""
Pixel-wise Exposure Mapping Analysis

This script performs 1:1 pixel mapping between low-light and high-light images
and tests various mathematical models for the relationship.

Models tested:
1. Deterministic: H = f(L) + noise
   - Linear, Polynomial, etc.

2. Probabilistic: P(H|L)
   - Gaussian: H|L ~ N(μ(L), σ²)
   - Heteroscedastic Gaussian: H|L ~ N(μ(L), σ²(L))
   - Poisson-Gaussian: σ²(L) = σ²_read + α·L

Usage:
    python sample/pixel_wise_exposure_mapping.py --num_tiles 10
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

    # Get matching high-light (10s exposure)
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


def analyze_pixel_wise_mapping(num_tiles=10):
    """Analyze pixel-wise mapping with mathematical models."""

    noisy_dir = Path("dataset/processed/pt_tiles/photography/noisy")
    clean_dir = Path("dataset/processed/pt_tiles/photography/clean")
    output_dir = Path("results/noise_model_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load multiple tile pairs
    noisy_files = sorted(noisy_dir.glob("*.pt"))[:num_tiles]

    all_L = []  # Low-light pixels
    all_H = []  # High-light pixels

    logger.info(f"Loading {len(noisy_files)} tile pairs for pixel-wise analysis...")

    for noisy_path in noisy_files:
        L_tile, H_tile = load_tile_pair(noisy_path, clean_dir)
        if L_tile is not None and H_tile is not None:
            all_L.append(L_tile.flatten())
            all_H.append(H_tile.flatten())
            logger.info(f"  Loaded: {noisy_path.name}")

    L = np.concatenate(all_L)
    H = np.concatenate(all_H)

    logger.info(f"\n{'='*70}")
    logger.info(f"PIXEL-WISE MAPPING ANALYSIS")
    logger.info(f"{'='*70}")
    logger.info(f"Tiles: {len(all_L)}")
    logger.info(f"Total pixels: {len(L):,}")
    logger.info(f"Low-light range: [{L.min():.6f}, {L.max():.6f}]")
    logger.info(f"High-light range: [{H.min():.6f}, {H.max():.6f}]")

    # Basic statistics
    correlation = np.corrcoef(L, H)[0, 1]
    logger.info(f"\nPearson correlation: {correlation:.6f}")
    logger.info(f"Low-light mean: {L.mean():.6f}, std: {L.std():.6f}")
    logger.info(f"High-light mean: {H.mean():.6f}, std: {H.std():.6f}")

    # Binned analysis for visualization and variance analysis
    n_bins = 50
    L_bins = np.linspace(L.min(), L.max(), n_bins + 1)
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_vars = []
    bin_counts = []

    logger.info(f"\nBinning data for variance analysis (visualization only)...")

    for i in range(n_bins):
        mask = (L >= L_bins[i]) & (L < L_bins[i + 1])
        if mask.sum() > 100:  # Need sufficient samples
            H_bin = H[mask]
            bin_centers.append((L_bins[i] + L_bins[i + 1]) / 2)
            bin_means.append(np.mean(H_bin))
            bin_stds.append(np.std(H_bin))
            bin_vars.append(np.var(H_bin))
            bin_counts.append(mask.sum())

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_vars = np.array(bin_vars)
    bin_counts = np.array(bin_counts)

    logger.info(f"Computed {len(bin_centers)} bins for visualization")

    # ===== TEST MODELS ON ALL PIXELS =====

    logger.info(f"\n{'='*70}")
    logger.info("MODEL 1: DETERMINISTIC MEAN FUNCTION μ(L)")
    logger.info(f"  Fitting on ALL {len(L):,} individual pixels")
    logger.info(f"{'='*70}")

    # Subsample for faster fitting if needed
    max_fit_samples = 500000
    if len(L) > max_fit_samples:
        logger.info(f"Subsampling to {max_fit_samples:,} pixels for fitting...")
        idx_fit = np.random.choice(len(L), max_fit_samples, replace=False)
        L_fit = L[idx_fit]
        H_fit = H[idx_fit]
    else:
        L_fit = L
        H_fit = H

    # Linear mean - fit on ALL individual pixels
    def linear_mean(L, a, b):
        return a * L + b

    logger.info(f"\nFitting Linear model on {len(L_fit):,} pixels...")
    popt_lin, _ = curve_fit(linear_mean, L_fit, H_fit)
    a_lin, b_lin = popt_lin

    # Calculate R² on all data
    H_pred_lin_all = linear_mean(L, a_lin, b_lin)
    ss_res = np.sum((H - H_pred_lin_all) ** 2)
    ss_tot = np.sum((H - np.mean(H)) ** 2)
    r2_lin = 1 - (ss_res / ss_tot)
    rmse_lin = np.sqrt(np.mean((H - H_pred_lin_all) ** 2))

    logger.info(f"Linear: H = {a_lin:.6f}·L + {b_lin:.6f}")
    logger.info(f"  R² = {r2_lin:.6f} (on all {len(L):,} pixels)")
    logger.info(f"  RMSE = {rmse_lin:.6f}")

    # Polynomial mean - fit on individual pixels
    def poly3_mean(L, a, b, c, d):
        return a * L**3 + b * L**2 + c * L + d

    try:
        logger.info(f"\nFitting Polynomial model on {len(L_fit):,} pixels...")
        popt_poly, _ = curve_fit(poly3_mean, L_fit, H_fit, maxfev=10000)
        a3, b3, c3, d3 = popt_poly

        # Calculate R² on all data
        H_pred_poly_all = poly3_mean(L, a3, b3, c3, d3)
        ss_res_poly = np.sum((H - H_pred_poly_all) ** 2)
        r2_poly = 1 - (ss_res_poly / ss_tot)
        rmse_poly = np.sqrt(np.mean((H - H_pred_poly_all) ** 2))

        logger.info(
            f"Polynomial: H = {a3:.4f}·L³ + {b3:.4f}·L² + {c3:.4f}·L + {d3:.4f}"
        )
        logger.info(f"  R² = {r2_poly:.6f} (on all {len(L):,} pixels)")
        logger.info(f"  RMSE = {rmse_poly:.6f}")
        poly_fitted = True
    except Exception as e:
        logger.warning(f"Polynomial fit failed: {e}")
        poly_fitted = False
        r2_poly = -np.inf

    # Choose best mean function
    best_mean = "polynomial" if (poly_fitted and r2_poly > r2_lin) else "linear"

    logger.info(f"\n{'='*70}")
    logger.info("MODEL 2: VARIANCE FUNCTION σ²(L)")
    logger.info(f"  Analyzing variance using {len(bin_centers)} bins")
    logger.info(f"{'='*70}")

    # Test 1: Constant variance (Homoscedastic Gaussian)
    # Use residuals from best mean function to compute actual variance
    if best_mean == "linear":
        residuals = H - H_pred_lin_all
    else:
        residuals = H - H_pred_poly_all

    const_var = np.var(residuals)

    logger.info(f"\nHomoscedastic Gaussian:")
    logger.info(f"  σ²(L) = {const_var:.6f} (constant)")
    logger.info(f"  σ(L) = {np.sqrt(const_var):.6f}")
    logger.info(f"  Computed from residuals of {len(L):,} pixels")

    # For binned variance model comparison
    ss_tot = np.sum((bin_vars - np.mean(bin_vars)) ** 2)
    ss_res_const = np.sum((bin_vars - const_var) ** 2)
    r2_const = 1 - (ss_res_const / ss_tot) if ss_tot > 0 else 0
    logger.info(f"  R² (binned comparison) = {r2_const:.6f}")

    # Test 2: Linear variance (Poisson-Gaussian)
    def linear_var(L, sigma2_read, alpha):
        return sigma2_read + alpha * L

    try:
        # Use weighted fit
        weights = np.sqrt(bin_counts)
        popt_var, pcov_var = curve_fit(
            linear_var,
            bin_centers,
            bin_vars,
            p0=[const_var, 0.0],
            sigma=1 / weights,
            absolute_sigma=False,
        )
        sigma2_read, alpha = popt_var
        perr = np.sqrt(np.diag(pcov_var))

        var_pred = linear_var(bin_centers, sigma2_read, alpha)
        ss_res_lin = np.sum((bin_vars - var_pred) ** 2)
        r2_lin_var = 1 - (ss_res_lin / ss_tot) if ss_tot > 0 else 0

        # Statistical test for alpha
        t_stat = alpha / perr[1] if perr[1] > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(bin_centers) - 2))

        logger.info(f"\nPoisson-Gaussian (Heteroscedastic):")
        logger.info(f"  σ²(L) = {sigma2_read:.6f} + {alpha:.6f}·L")
        logger.info(f"  σ_read = {np.sqrt(abs(sigma2_read)):.6f}")
        logger.info(f"  α = {alpha:.6f} ± {perr[1]:.6f}")
        logger.info(f"  R² = {r2_lin_var:.6f}")

        logger.info(f"\n  Statistical Test (H0: α = 0):")
        logger.info(f"    t-statistic: {t_stat:.4f}")
        logger.info(f"    p-value: {p_value:.6f}")

        if p_value < 0.05:
            if alpha > 0:
                logger.info(
                    f"    ✓ HETEROSCEDASTIC: variance increases with L (Poisson-like)"
                )
            else:
                logger.info(f"    ✓ HETEROSCEDASTIC: variance decreases with L")
        else:
            logger.info(f"    ✗ HOMOSCEDASTIC: constant variance (α not significant)")

        var_fitted = True
    except Exception as e:
        logger.warning(f"Linear variance fit failed: {e}")
        var_fitted = False
        r2_lin_var = -np.inf
        sigma2_read = const_var
        alpha = 0
        p_value = 1.0

    # Model comparison
    logger.info(f"\n{'='*70}")
    logger.info("BEST MODEL SELECTION")
    logger.info(f"{'='*70}")

    logger.info(f"\nVariance models:")
    logger.info(f"  Constant (Homoscedastic):    R² = {r2_const:.6f}")
    if var_fitted:
        logger.info(f"  Linear (Poisson-Gaussian):   R² = {r2_lin_var:.6f}")
        logger.info(f"  Improvement: ΔR² = {r2_lin_var - r2_const:.6f}")

    if var_fitted and r2_lin_var > r2_const + 0.05:
        best_var_model = "Poisson-Gaussian"
    else:
        best_var_model = "Homoscedastic Gaussian"

    logger.info(f"\n✓ Best variance model: {best_var_model}")

    # Final model
    logger.info(f"\n{'='*70}")
    logger.info("FINAL MATHEMATICAL MODEL")
    logger.info(f"  Based on {len(L):,} individual pixel data points")
    logger.info(f"{'='*70}")
    logger.info(f"\nConditional Distribution: H | L ~ N(μ(L), σ²)")
    logger.info(f"\nMean function μ(L):")
    if best_mean == "linear":
        logger.info(f"  μ(L) = {a_lin:.6f}·L + {b_lin:.6f}")
        logger.info(f"  R² = {r2_lin:.6f}, RMSE = {rmse_lin:.6f}")
    else:
        logger.info(f"  μ(L) = {a3:.4f}·L³ + {b3:.4f}·L² + {c3:.4f}·L + {d3:.4f}")
        logger.info(f"  R² = {r2_poly:.6f}, RMSE = {rmse_poly:.6f}")

    logger.info(f"\nVariance function σ²:")
    logger.info(f"  σ² = {const_var:.6f} (constant across all L)")
    logger.info(f"  σ = {np.sqrt(const_var):.6f}")
    logger.info(f"  Model: {best_var_model}")

    # Create visualization
    create_pixelwise_visualization(
        L,
        H,
        bin_centers,
        bin_means,
        bin_stds,
        bin_vars,
        popt_lin,
        const_var,
        sigma2_read if var_fitted else 0,
        alpha if var_fitted else 0,
        r2_const,
        r2_lin_var if var_fitted else 0,
        best_var_model,
        correlation,
        output_dir / "pixelwise_exposure_mapping.png",
    )

    # Save results
    results = {
        "n_tiles": len(all_L),
        "n_pixels": int(len(L)),
        "n_pixels_used_for_fitting": int(len(L_fit)),
        "correlation": float(correlation),
        "lowlight_range": [float(L.min()), float(L.max())],
        "highlight_range": [float(H.min()), float(H.max())],
        "lowlight_stats": {"mean": float(L.mean()), "std": float(L.std())},
        "highlight_stats": {"mean": float(H.mean()), "std": float(H.std())},
        "mean_function": {
            "type": best_mean,
            "linear": {
                "params": [float(a_lin), float(b_lin)],
                "formula": f"H = {a_lin:.6f}·L + {b_lin:.6f}",
                "r2": float(r2_lin),
                "rmse": float(rmse_lin),
            },
            "polynomial": {
                "params": [float(a3), float(b3), float(c3), float(d3)]
                if poly_fitted
                else None,
                "r2": float(r2_poly) if poly_fitted else None,
                "rmse": float(rmse_poly) if poly_fitted else None,
            }
            if poly_fitted
            else None,
        },
        "variance_function": {
            "type": best_var_model,
            "constant_var": float(const_var),
            "constant_sigma": float(np.sqrt(const_var)),
            "poisson_gaussian_params": {
                "sigma2_read": float(sigma2_read) if var_fitted else None,
                "alpha": float(alpha) if var_fitted else None,
                "p_value": float(p_value) if var_fitted else None,
            }
            if var_fitted
            else None,
            "r2_constant": float(r2_const),
            "r2_linear": float(r2_lin_var) if var_fitted else None,
        },
    }

    with open(output_dir / "pixelwise_mapping_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {output_dir}/pixelwise_mapping_results.json")


def create_pixelwise_visualization(
    L,
    H,
    bin_centers,
    bin_means,
    bin_stds,
    bin_vars,
    popt_lin,
    const_var,
    sigma2_read,
    alpha,
    r2_const,
    r2_lin_var,
    best_var_model,
    correlation,
    output_path,
):
    """Create publication-quality visualization of pixel-wise mapping."""

    # Publication settings
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.linewidth": 1.5,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
        }
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        2, 3, hspace=0.35, wspace=0.35, left=0.08, right=0.95, top=0.92, bottom=0.08
    )

    # Sample for plotting - create 2D density map
    idx_sample = np.random.choice(len(L), min(100000, len(L)), replace=False)

    # 1. Main scatter plot with density and mean function (larger, publication quality)
    ax1 = fig.add_subplot(gs[:, 0:2])

    # Create 2D histogram for better visualization
    hist2d, xedges, yedges = np.histogram2d(
        L[idx_sample],
        H[idx_sample],
        bins=150,
        range=[[L.min(), L.max()], [H.min(), H.max()]],
    )

    # Plot as heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax1.imshow(
        hist2d.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="Blues",
        interpolation="gaussian",
        alpha=0.9,
        zorder=5,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, label="Pixel Density", pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    # Plot mean function (very thin line) - on top with higher zorder
    L_plot = np.linspace(L.min(), L.max(), 300)
    a, b = popt_lin
    H_mean = a * L_plot + b
    ax1.plot(
        L_plot,
        H_mean,
        "r-",
        linewidth=1.5,
        label=rf"$\mu(L) = {a:.2f} \cdot L + {b:.2f}$",
        zorder=10,
    )

    # Plot ±σ bands removed per user request
    sigma = np.sqrt(const_var)

    ax1.set_xlabel("Low-Light Pixel Intensity ($L$)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("High-Light Pixel Intensity ($H$)", fontsize=14, fontweight="bold")
    ax1.set_title(
        "(a) Pixel-Wise Exposure Mapping: Low-Light to High-Light",
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax1.legend(
        fontsize=11,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fancybox=False,
    )
    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)

    # Apply symlog scale to y-axis (handles negative values)
    ax1.set_yscale("symlog")

    # Rescale to actual data range with small margin
    H_min_sample, H_max_sample = H[idx_sample].min(), H[idx_sample].max()
    L_min_sample, L_max_sample = L[idx_sample].min(), L[idx_sample].max()
    L_range = L_max_sample - L_min_sample

    ax1.set_xlim([L_min_sample - 0.02 * L_range, L_max_sample + 0.02 * L_range])

    # Set equal aspect ratio for square scaling
    ax1.set_aspect("equal", adjustable="box")

    # Add text box with correlation
    textstr = f"Pearson $r$ = {correlation:.4f}\n$n$ = {len(L):,} pixels"
    props = dict(
        boxstyle="round", facecolor="wheat", alpha=0.8, edgecolor="black", linewidth=1.5
    )
    ax1.text(
        0.98,
        0.02,
        textstr,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    # 2. Conditional mean with error bars
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(
        bin_centers,
        bin_means,
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidth=1.5,
        color="#2E86AB",
        label="Binned means",
        zorder=5,
    )
    ax2.plot(L_plot, H_mean, "r-", linewidth=2.5, label="Linear fit")
    ax2.set_xlabel(r"$L$", fontsize=13, fontweight="bold")
    ax2.set_ylabel(r"$\mathbb{E}[H|L]$", fontsize=13, fontweight="bold")
    ax2.set_title("(b) Conditional Mean", fontsize=13, fontweight="bold", loc="left")
    ax2.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax2.tick_params(labelsize=10)

    # Apply log scale to y-axis
    ax2.set_yscale("log")

    # Rescale to data range with small margin
    if len(bin_means) > 0:
        x_range = bin_centers.max() - bin_centers.min()
        ax2.set_xlim(
            [bin_centers.min() - 0.02 * x_range, bin_centers.max() + 0.02 * x_range]
        )

        # Set equal aspect ratio for square scaling
        ax2.set_aspect("equal", adjustable="box")

    # Add R² text - moved to bottom right corner
    from sklearn.metrics import r2_score

    r2_mean = r2_score(bin_means, a * bin_centers + b)
    ax2.text(
        0.95,
        0.05,
        rf"$R^2$ = {r2_mean:.4f}",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"),
    )

    # 3. Conditional variance with rigorous model comparison
    ax3 = fig.add_subplot(gs[1, 2])

    # Plot empirical variances
    ax3.scatter(
        bin_centers,
        bin_vars,
        s=120,
        alpha=0.9,
        edgecolors="black",
        linewidth=2,
        color="#A23B72",
        label="Empirical",
        zorder=5,
    )

    # Compute predictions and R² for both models
    const_pred = np.full_like(bin_vars, const_var)
    ss_tot = np.sum((bin_vars - np.mean(bin_vars)) ** 2)

    # Constant model
    ss_res_const = np.sum((bin_vars - const_pred) ** 2)
    r2_const = 1 - (ss_res_const / ss_tot) if ss_tot > 0 else 0
    ax3.axhline(
        const_var,
        color="#1f77b4",
        linestyle="--",
        linewidth=2.5,
        label=f"Constant: $R^2$ = {r2_const:.3f}",
        alpha=0.9,
        zorder=3,
    )

    # Poisson-Gaussian model
    if r2_lin_var > -np.inf and alpha != 0:
        var_linear_plot = sigma2_read + alpha * L_plot
        var_linear_bins = sigma2_read + alpha * bin_centers
        ss_res_pg = np.sum((bin_vars - var_linear_bins) ** 2)
        r2_pg = 1 - (ss_res_pg / ss_tot) if ss_tot > 0 else 0

        ax3.plot(
            L_plot,
            var_linear_plot,
            color="#d62728",
            linestyle="-",
            linewidth=2.5,
            label=f"P-G: $R^2$ = {r2_pg:.3f}",
            alpha=0.9,
            zorder=4,
        )

        # F-test for model comparison
        from scipy import stats as sp_stats

        n = len(bin_vars)
        F_stat = ((ss_res_const - ss_res_pg) / 1) / (ss_res_pg / (n - 2))
        p_value_F = 1 - sp_stats.f.cdf(F_stat, 1, n - 2)

        # Determine best model
        if p_value_F < 0.05 and r2_pg > r2_const:
            best_label = "Poisson-Gaussian"
            best_color = "lightgreen"
            edge_color = "darkgreen"
            significance = f"✓ Sig. (p={p_value_F:.4f})"
        else:
            best_label = "Constant"
            best_color = "lightblue"
            edge_color = "darkblue"
            significance = f"Not sig. (p={p_value_F:.4f})"

        # Add statistical comparison box - moved to upper right corner
        comparison_text = f"F-test: {significance}\nΔ$R^2$ = {r2_pg - r2_const:+.3f}\nBest: {best_label}"
        ax3.text(
            0.97,
            0.97,
            comparison_text,
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round",
                facecolor=best_color,
                alpha=0.85,
                edgecolor=edge_color,
                linewidth=2.5,
            ),
        )
    else:
        # Only constant model available - upper right corner
        ax3.text(
            0.97,
            0.97,
            f"Best: Constant\n$R^2$ = {r2_const:.3f}",
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round",
                facecolor="lightblue",
                alpha=0.85,
                edgecolor="darkblue",
                linewidth=2.5,
            ),
        )

    ax3.set_xlabel(r"$L$", fontsize=13, fontweight="bold")
    ax3.set_ylabel(r"$\mathrm{Var}[H|L]$", fontsize=13, fontweight="bold")
    ax3.set_title(
        "(c) Variance Model Comparison", fontsize=13, fontweight="bold", loc="left"
    )
    ax3.legend(
        fontsize=9,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fancybox=False,
    )
    ax3.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax3.tick_params(labelsize=10)

    # Apply log scale to y-axis
    ax3.set_yscale("log")

    # Rescale to data range with margin
    if len(bin_vars) > 0:
        x_range = bin_centers.max() - bin_centers.min()
        ax3.set_xlim(
            [bin_centers.min() - 0.02 * x_range, bin_centers.max() + 0.02 * x_range]
        )

        # Set equal aspect ratio for square scaling
        ax3.set_aspect("equal", adjustable="box")

    plt.suptitle(
        "Pixel-Wise Exposure Mapping: Statistical Analysis",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )

    # Save with high quality
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()

    # Reset matplotlib settings
    plt.rcParams.update(plt.rcParamsDefault)

    logger.info(f"✓ Publication-quality visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pixel-wise exposure mapping analysis")
    parser.add_argument(
        "--num_tiles", type=int, default=10, help="Number of tile pairs to analyze"
    )
    args = parser.parse_args()

    analyze_pixel_wise_mapping(args.num_tiles)


if __name__ == "__main__":
    main()
