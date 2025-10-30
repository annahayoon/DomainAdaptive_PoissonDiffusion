#!/usr/bin/env python3
"""
Analyze results CSV files to find best tiles per domain based on PG-x0 metrics.
Criteria:
- High PSNR and SSIM (better quality)
- Low LPIPS and NIQE (better perceptual quality)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_metrics_for_composite_score(df):
    """
    Normalize metrics to [0,1] scale for composite score calculation.
    Higher is better for PSNR/SSIM, lower is better for LPIPS/NIQE.
    """
    # Normalize higher-is-better metrics
    for metric in ["psnr", "ssim"]:
        if metric in df.columns:
            col_min = df[metric].min()
            col_max = df[metric].max()
            if col_max > col_min:
                df[f"{metric}_norm"] = (df[metric] - col_min) / (col_max - col_min)
            else:
                df[f"{metric}_norm"] = 0.5

    # Normalize lower-is-better metrics (invert so higher is better)
    for metric in ["lpips", "niqe"]:
        if metric in df.columns:
            col_min = df[metric].min()
            col_max = df[metric].max()
            if col_max > col_min:
                df[f"{metric}_norm"] = 1 - (df[metric] - col_min) / (col_max - col_min)
            else:
                df[f"{metric}_norm"] = 0.5

    return df


def analyze_single_domain_csv(csv_path, domain_name):
    """Analyze single domain CSV file."""
    print(f"Analyzing {domain_name} from {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["method"] == "pg_x0"]  # Only look at pg_x0 results

    if len(df) == 0:
        print(f"No pg_x0 results found for {domain_name}")
        return None

    # Normalize metrics and calculate composite score
    df = normalize_metrics_for_composite_score(df)

    # Composite score (higher is better)
    norm_cols = [col for col in df.columns if col.endswith("_norm")]
    if norm_cols:
        df["composite_score"] = df[norm_cols].mean(axis=1)
    else:
        df["composite_score"] = 0.0

    # Sort by composite score and get top 3
    top_tiles = df.nlargest(3, "composite_score")

    return top_tiles[
        ["tile_id", "ssim", "psnr", "lpips", "niqe", "mse", "composite_score"]
    ]


def analyze_photography_csv(csv_path):
    """Analyze photography CSV with both Sony and Fuji."""
    print(f"Analyzing photography from {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["method"] == "pg_x0"]

    results = {}

    # Analyze Sony
    sony_df = df[df["sensor"] == "sony"].copy()
    if len(sony_df) > 0:
        sony_df = normalize_metrics_for_composite_score(sony_df)
        norm_cols = [col for col in sony_df.columns if col.endswith("_norm")]
        if norm_cols:
            sony_df["composite_score"] = sony_df[norm_cols].mean(axis=1)
        results["sony"] = sony_df.nlargest(3, "composite_score")[
            ["tile_id", "ssim", "psnr", "lpips", "niqe", "mse", "composite_score"]
        ]

    # Analyze Fuji
    fuji_df = df[df["sensor"] == "fuji"].copy()
    if len(fuji_df) > 0:
        fuji_df = normalize_metrics_for_composite_score(fuji_df)
        norm_cols = [col for col in fuji_df.columns if col.endswith("_norm")]
        if norm_cols:
            fuji_df["composite_score"] = fuji_df[norm_cols].mean(axis=1)
        results["fuji"] = fuji_df.nlargest(3, "composite_score")[
            ["tile_id", "ssim", "psnr", "lpips", "niqe", "mse", "composite_score"]
        ]

    return results


def analyze_cross_domain_csv(csv_path, domain_name):
    """Analyze cross domain CSV file."""
    print(f"Analyzing {domain_name} cross domain from {csv_path}")

    df = pd.read_csv(csv_path)

    # Extract PG-x0 cross metrics
    pg_cols = [col for col in df.columns if "pg_x0_cross" in col]
    if not pg_cols:
        print(f"No pg_x0_cross columns found for {domain_name}")
        return None

    # Create a new dataframe with just the tile_id and pg_x0_cross metrics
    pg_metrics = {}
    for col in pg_cols:
        metric = col.replace("pg_x0_cross_", "")
        pg_metrics[metric] = df[col]

    pg_df = pd.DataFrame(pg_metrics)
    pg_df["tile_id"] = df["tile_id"]

    # Normalize metrics
    metrics_to_normalize = {}
    for col in pg_df.columns:
        if col != "tile_id":
            values = pg_df[col].dropna()
            if len(values) > 0:
                min_val, max_val = values.min(), values.max()
                if max_val > min_val:
                    metrics_to_normalize[col] = (min_val, max_val)

    # Normalize metrics using shared function
    # Map columns to standard names for normalization
    column_mapping = {}
    for col in pg_df.columns:
        if col != "tile_id":
            if "psnr" in col.lower():
                column_mapping[col] = "psnr"
            elif "ssim" in col.lower():
                column_mapping[col] = "ssim"
            elif "lpips" in col.lower():
                column_mapping[col] = "lpips"
            elif "niqe" in col.lower():
                column_mapping[col] = "niqe"

    # Create temporary dataframe with standardized names for normalization
    temp_df = pg_df.copy()
    for old_col, new_col in column_mapping.items():
        if old_col in temp_df.columns:
            temp_df[new_col] = temp_df[old_col]

    # Normalize
    temp_df = normalize_metrics_for_composite_score(temp_df)

    # Map normalized columns back
    for old_col, new_col in column_mapping.items():
        if f"{new_col}_norm" in temp_df.columns:
            pg_df[f"{old_col}_norm"] = temp_df[f"{new_col}_norm"]

    # Calculate composite score
    norm_cols = [col for col in pg_df.columns if "_norm" in col and col != "tile_id"]
    if norm_cols:
        pg_df["composite_score"] = pg_df[norm_cols].mean(axis=1)

    return pg_df.nlargest(3, "composite_score")[
        ["tile_id"]
        + [
            col.replace("_norm", "")
            for col in norm_cols
            if col.replace("_norm", "") in pg_df.columns
        ]
        + ["composite_score"]
    ]


def main():
    base_path = Path("/home/jilab/Jae/results")

    results = {}

    # Analyze astronomy
    astro_path = (
        base_path
        / "optimized_inference_all_tiles"
        / "astronomy_optimized"
        / "results.csv"
    )
    if astro_path.exists():
        results["astronomy"] = analyze_single_domain_csv(astro_path, "astronomy")

    # Analyze microscopy
    micro_path = (
        base_path
        / "optimized_inference_all_tiles"
        / "microscopy_optimized"
        / "microscopy_individual_results.csv"
    )
    if micro_path.exists():
        results["microscopy"] = analyze_single_domain_csv(micro_path, "microscopy")

    # Analyze photography (Sony and Fuji)
    photo_path = (
        base_path
        / "optimized_inference_all_tiles"
        / "photography_single_domain_detailed_results.csv"
    )
    if photo_path.exists():
        results["photography"] = analyze_photography_csv(photo_path)

    # Analyze cross domain results
    cross_results = {}
    for domain in ["astronomy", "microscopy", "photography"]:
        cross_path = (
            base_path
            / "cross_domain_inference_all_tiles"
            / f"{domain}_cross_domain"
            / "results.csv"
        )
        if cross_path.exists():
            cross_results[domain] = analyze_cross_domain_csv(cross_path, domain)

    # Save results
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    # Save single domain results
    for domain, data in results.items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_dir / f"{domain}_best_tiles.csv", index=False)
            print(f"\n{domain.upper()} TOP TILES:")
            print(data.to_string(index=False))
        elif isinstance(data, dict):
            for sub_domain, sub_data in data.items():
                sub_data.to_csv(
                    output_dir / f"{domain}_{sub_domain}_best_tiles.csv", index=False
                )
                print(f"\n{domain.upper()} {sub_domain.upper()} TOP TILES:")
                print(sub_data.to_string(index=False))

    # Save cross domain results
    for domain, data in cross_results.items():
        if data is not None:
            data.to_csv(
                output_dir / f"{domain}_cross_domain_best_tiles.csv", index=False
            )
            print(f"\n{domain.upper()} CROSS DOMAIN TOP TILES:")
            print(data.to_string(index=False))

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
