#!/usr/bin/env python3
"""
Generate results tables and analyze best tiles from evaluation outputs.

This script combines functionality from:
- generate_results_table.py: Generate CSV tables and LaTeX tables
- analyze_results.py: Find best tiles per sensor based on metrics

Modes:
1. tables (default): Generate CSV tables, summary statistics, and stratified tables
2. best_tiles: Analyze and find best tiles per sensor based on composite scores

Usage:
    # Generate tables
    python generate_results_table.py --mode tables --results_dir results/

    # Find best tiles
    python generate_results_table.py --mode best_tiles --csv_path results/metrics.csv
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.utils.visualization_utils import (
    add_common_visualization_arguments,
    ensure_output_dir,
    get_results_base_path,
    load_results_from_directory,
    normalize_metrics_for_composite_score,
    resolve_output_path,
    setup_visualization_logging,
)

logger = setup_visualization_logging()


def extract_metrics_dataframe(all_results: List[Dict]) -> pd.DataFrame:
    """
    Extract all metrics into a pandas DataFrame.

    Creates one row per (tile, method) combination.

    Args:
        all_results: List of result dictionaries

    Returns:
        DataFrame with columns:
            - tile_id
            - method
            - metric_name (psnr, ssim, mse, lpips, niqe)
            - metric_value
            - sigma_max_used
            - exposure_ratio
    """
    rows = []

    for result in all_results:
        tile_id = result.get("tile_id", "unknown")
        sigma_max = result.get("sigma_max_used", float("nan"))
        exposure_ratio = result.get("exposure_ratio", float("nan"))

        # Extract comprehensive_metrics if available
        if "comprehensive_metrics" in result:
            for method, metrics in result["comprehensive_metrics"].items():
                for metric_name, metric_value in metrics.items():
                    rows.append(
                        {
                            "tile_id": tile_id,
                            "method": method,
                            "metric_name": metric_name,
                            "metric_value": metric_value,
                            "sigma_max_used": sigma_max,
                            "exposure_ratio": exposure_ratio,
                        }
                    )

        # Also check for top-level metrics
        elif "metrics" in result:
            method = result.get("method", "unknown")
            for metric_name, metric_value in result["metrics"].items():
                rows.append(
                    {
                        "tile_id": tile_id,
                        "method": method,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "sigma_max_used": sigma_max,
                        "exposure_ratio": exposure_ratio,
                    }
                )

    if not rows:
        logger.warning("No metrics found in results")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics DataFrame.

    Computes mean, std, median for each (method, metric) combination.

    Args:
        df: Metrics DataFrame from extract_metrics_dataframe

    Returns:
        DataFrame with columns: method, metric_name, mean, std, median, count
    """
    if df.empty:
        return pd.DataFrame()

    summary_rows = []

    for (method, metric_name), group in df.groupby(["method", "metric_name"]):
        values = group["metric_value"].dropna()

        if len(values) > 0:
            summary_rows.append(
                {
                    "method": method,
                    "metric_name": metric_name,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "count": len(values),
                }
            )

    return pd.DataFrame(summary_rows)


def generate_stratified_table(
    stratified_results: Optional[Dict],
    all_results: List[Dict],
    baseline_method: str = "gaussian_x0",
    proposed_method: str = "pg_x0",
) -> Dict[str, Any]:
    """
    Generate Table 1: Stratified results by ADC bin.

    Args:
        stratified_results: Stratified evaluation results dict
        all_results: List of per-tile results
        baseline_method: Name of baseline method
        proposed_method: Name of proposed method

    Returns:
        Dictionary with:
            - 'table_data': DataFrame with stratified metrics
            - 'improvements': Improvement matrix
            - 'statistical_significance': P-values and significance
            - 'formatted_table': ASCII formatted table string
    """
    table_data = {
        "bin_name": [],
        "bin_label": [],
        "baseline_psnr": [],
        "proposed_psnr": [],
        "improvement": [],
        "p_value": [],
        "p_value_corrected": [],
        "significant": [],
        "n_samples": [],
    }

    bin_labels = {
        "very_low": "Very Low (ADC < 100)",
        "low": "Low (100-500)",
        "medium": "Medium (500-2000)",
        "high": "High (> 2000)",
    }

    bin_order = ["very_low", "low", "medium", "high"]

    # Extract stratified metrics from all_results or stratified_results
    if stratified_results and "statistical_significance" in stratified_results:
        sig_results = stratified_results["statistical_significance"]

        for bin_name in bin_order:
            if bin_name in sig_results:
                sig_data = sig_results[bin_name]
                table_data["bin_name"].append(bin_name)
                table_data["bin_label"].append(bin_labels[bin_name])
                table_data["improvement"].append(
                    sig_data.get("mean_improvement", float("nan"))
                )
                table_data["p_value"].append(sig_data.get("p_value", float("nan")))
                table_data["p_value_corrected"].append(
                    sig_data.get("p_value_corrected", float("nan"))
                )
                table_data["significant"].append(sig_data.get("significant", False))
                table_data["n_samples"].append(sig_data.get("n_samples", 0))

                # Get baseline and proposed PSNR from comparison data
                baseline_psnr = float("nan")
                proposed_psnr = float("nan")

                if "comparison_by_tile" in stratified_results:
                    # Aggregate PSNR across tiles
                    baseline_values = []
                    proposed_values = []

                    for tile_id, comparison in stratified_results[
                        "comparison_by_tile"
                    ].items():
                        if (
                            baseline_method in comparison
                            and bin_name in comparison[baseline_method]
                        ):
                            psnr = comparison[baseline_method][bin_name].get(
                                "psnr", float("nan")
                            )
                            if not np.isnan(psnr):
                                baseline_values.append(psnr)

                        if (
                            proposed_method in comparison
                            and bin_name in comparison[proposed_method]
                        ):
                            psnr = comparison[proposed_method][bin_name].get(
                                "psnr", float("nan")
                            )
                            if not np.isnan(psnr):
                                proposed_values.append(psnr)

                    if baseline_values:
                        baseline_psnr = np.mean(baseline_values)
                    if proposed_values:
                        proposed_psnr = np.mean(proposed_values)

                table_data["baseline_psnr"].append(baseline_psnr)
                table_data["proposed_psnr"].append(proposed_psnr)

    elif all_results:
        # Extract from individual results if stratified_results not available
        # Collect per-bin metrics
        bin_data = {
            bin_name: {"baseline": [], "proposed": []} for bin_name in bin_order
        }

        for result in all_results:
            if "stratified_metrics" in result:
                stratified = result["stratified_metrics"]

                if baseline_method in stratified and proposed_method in stratified:
                    for bin_name in bin_order:
                        baseline_entry = stratified[baseline_method].get(bin_name, {})
                        proposed_entry = stratified[proposed_method].get(bin_name, {})

                        baseline_psnr = baseline_entry.get("psnr", float("nan"))
                        proposed_psnr = proposed_entry.get("psnr", float("nan"))

                        if not (np.isnan(baseline_psnr) or np.isnan(proposed_psnr)):
                            bin_data[bin_name]["baseline"].append(baseline_psnr)
                            bin_data[bin_name]["proposed"].append(proposed_psnr)

        # Compute means
        for bin_name in bin_order:
            baseline_vals = bin_data[bin_name]["baseline"]
            proposed_vals = bin_data[bin_name]["proposed"]

            baseline_psnr = np.mean(baseline_vals) if baseline_vals else float("nan")
            proposed_psnr = np.mean(proposed_vals) if proposed_vals else float("nan")
            improvement = (
                proposed_psnr - baseline_psnr
                if not (np.isnan(baseline_psnr) or np.isnan(proposed_psnr))
                else float("nan")
            )

            table_data["bin_name"].append(bin_name)
            table_data["bin_label"].append(bin_labels[bin_name])
            table_data["baseline_psnr"].append(baseline_psnr)
            table_data["proposed_psnr"].append(proposed_psnr)
            table_data["improvement"].append(improvement)
            table_data["p_value"].append(float("nan"))
            table_data["p_value_corrected"].append(float("nan"))
            table_data["significant"].append(False)
            table_data["n_samples"].append(len(baseline_vals))

    df = pd.DataFrame(table_data)

    # Generate formatted table string
    formatted_lines = []
    formatted_lines.append("=" * 100)
    formatted_lines.append("Table 1: Stratified PSNR by Signal Level (ADC)")
    formatted_lines.append("=" * 100)
    formatted_lines.append(
        f"{'Bin':<25} | {'Baseline PSNR':<15} | {'Proposed PSNR':<15} | {'Δ PSNR':<12} | {'p-value':<12} | {'Significant':<12} | {'n':<6}"
    )
    formatted_lines.append("-" * 100)

    for _, row in df.iterrows():
        baseline_str = (
            f"{row['baseline_psnr']:.2f} dB"
            if not np.isnan(row["baseline_psnr"])
            else "N/A"
        )
        proposed_str = (
            f"{row['proposed_psnr']:.2f} dB"
            if not np.isnan(row["proposed_psnr"])
            else "N/A"
        )

        improvement_str = (
            f"{row['improvement']:+.2f} dB"
            if not np.isnan(row["improvement"])
            else "N/A"
        )

        # Format p-value
        p_val = (
            row["p_value_corrected"]
            if not np.isnan(row["p_value_corrected"])
            else row["p_value"]
        )
        if not np.isnan(p_val):
            if p_val < 0.001:
                p_str = "p < 0.001 ***"
            elif p_val < 0.01:
                p_str = f"p = {p_val:.3f} **"
            elif p_val < 0.05:
                p_str = f"p = {p_val:.3f} *"
            else:
                p_str = f"p = {p_val:.3f}"
        else:
            p_str = "N/A"

        sig_str = "Yes" if row["significant"] else "No"

        formatted_lines.append(
            f"{row['bin_label']:<25} | {baseline_str:<15} | {proposed_str:<15} | {improvement_str:<12} | {p_str:<12} | {sig_str:<12} | {row['n_samples']:<6}"
        )

    formatted_lines.append("=" * 100)
    formatted_table = "\n".join(formatted_lines)

    return {
        "table_data": df,
        "formatted_table": formatted_table,
    }


def generate_latex_table(stratified_table: Dict[str, Any]) -> str:
    """
    Generate LaTeX-formatted table for paper.

    Args:
        stratified_table: Output from generate_stratified_table

    Returns:
        LaTeX table code
    """
    df = stratified_table["table_data"]

    if df.empty:
        return "% No stratified data available"

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Stratified PSNR by Signal Level}")
    lines.append("\\label{tab:stratified_results}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append(
        "Bin & Baseline & Proposed & $\\Delta$ PSNR & p-value & Sig. & $n$ \\\\"
    )
    lines.append("\\midrule")

    for _, row in df.iterrows():
        baseline_str = (
            f"{row['baseline_psnr']:.2f}"
            if not np.isnan(row["baseline_psnr"])
            else "N/A"
        )
        proposed_str = (
            f"{row['proposed_psnr']:.2f}"
            if not np.isnan(row["proposed_psnr"])
            else "N/A"
        )
        improvement_str = (
            f"{row['improvement']:+.2f}" if not np.isnan(row["improvement"]) else "N/A"
        )

        p_val = (
            row["p_value_corrected"]
            if not np.isnan(row["p_value_corrected"])
            else row["p_value"]
        )
        if not np.isnan(p_val):
            if p_val < 0.001:
                p_str = "$<$ 0.001"
            else:
                p_str = f"{p_val:.3f}"
        else:
            p_str = "N/A"

        sig_str = "$\\checkmark$" if row["significant"] else "---"

        # Escape LaTeX special characters in bin label
        bin_label = row["bin_label"].replace("_", "\\_")

        lines.append(
            f"{bin_label} & {baseline_str} & {proposed_str} & {improvement_str} & {p_str} & {sig_str} & {row['n_samples']} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ============================================================================
# Best Tiles Analysis Functions (from analyze_results.py)
# ============================================================================


def analyze_single_sensor_csv(
    csv_path: Path, sensor_name: str
) -> Optional[pd.DataFrame]:
    """Analyze single sensor CSV file to find best tiles."""
    logger.info(f"Analyzing {sensor_name} from {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["method"] == "pg_x0"]  # Only look at pg_x0 results

    if len(df) == 0:
        logger.warning(f"No pg_x0 results found for {sensor_name}")
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


def analyze_photography_csv(csv_path: Path) -> Dict[str, pd.DataFrame]:
    """Analyze sensor CSV with both Sony and Fuji."""
    logger.info(f"Analyzing sensors (Sony/Fuji) from {csv_path}")

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


def analyze_cross_sensor_csv(
    csv_path: Path, sensor_name: str
) -> Optional[pd.DataFrame]:
    """Analyze cross sensor CSV file to find best tiles."""
    logger.info(f"Analyzing {sensor_name} cross sensor from {csv_path}")

    df = pd.read_csv(csv_path)

    # Extract PG-x0 cross metrics
    pg_cols = [col for col in df.columns if "pg_x0_cross" in col]
    if not pg_cols:
        logger.warning(f"No pg_x0_cross columns found for {sensor_name}")
        return None

    # Create a new dataframe with just the tile_id and pg_x0_cross metrics
    pg_metrics = {}
    for col in pg_cols:
        metric = col.replace("pg_x0_cross_", "")
        pg_metrics[metric] = df[col]

    pg_df = pd.DataFrame(pg_metrics)
    pg_df["tile_id"] = df["tile_id"]

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


def run_best_tiles_analysis(args) -> None:
    """Run best tiles analysis mode."""
    base_path = get_results_base_path()
    results = {}
    cross_results = {}

    # Check if CSV path is provided or use default
    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return

        # Try to determine sensor from CSV
        if (
            "sensors" in csv_path.name
            or "sony" in csv_path.name.lower()
            or "fuji" in csv_path.name.lower()
        ):
            results["sensors"] = analyze_photography_csv(csv_path)
        else:
            # Try single sensor analysis
            sensor_name = args.sensor_name if args.sensor_name else "unknown"
            result = analyze_single_sensor_csv(csv_path, sensor_name)
            if result is not None:
                results[sensor_name] = result
    else:
        # Use default paths from analyze_results.py
        photo_path = (
            base_path
            / "optimized_inference_all_tiles"
            / "sensors_single_sensor_detailed_results.csv"
        )
        if photo_path.exists():
            results["sensors"] = analyze_photography_csv(photo_path)

        # Analyze cross sensor results
        for sensor in ["sensors"]:
            cross_path = (
                base_path
                / "cross_sensor_inference_all_tiles"
                / f"{sensor}_cross_sensor"
                / "results.csv"
            )
            if cross_path.exists():
                cross_results[sensor] = analyze_cross_sensor_csv(cross_path, sensor)

    # Create output directory
    default_output = Path("analysis_results")
    output_dir = resolve_output_path(args.output_dir, default_output, create_dir=True)

    # Save single sensor results
    for sensor, data in results.items():
        if isinstance(data, pd.DataFrame):
            output_file = output_dir / f"{sensor}_best_tiles.csv"
            data.to_csv(output_file, index=False)
            logger.info(f"Saved {sensor} best tiles to {output_file}")
            print(f"\n{sensor.upper()} TOP TILES:")
            print(data.to_string(index=False))
        elif isinstance(data, dict):
            for sub_sensor, sub_data in data.items():
                output_file = output_dir / f"{sensor}_{sub_sensor}_best_tiles.csv"
                sub_data.to_csv(output_file, index=False)
                logger.info(f"Saved {sensor} {sub_sensor} best tiles to {output_file}")
                print(f"\n{sensor.upper()} {sub_sensor.upper()} TOP TILES:")
                print(sub_data.to_string(index=False))

    # Save cross sensor results
    for sensor, data in cross_results.items():
        if data is not None:
            output_file = output_dir / f"{sensor}_cross_sensor_best_tiles.csv"
            data.to_csv(output_file, index=False)
            logger.info(f"Saved {sensor} cross sensor best tiles to {output_file}")
            print(f"\n{sensor.upper()} CROSS SENSOR TOP TILES:")
            print(data.to_string(index=False))

    logger.info(f"Best tiles analysis complete! Results saved to {output_dir}")


def main():
    """Main function for generating results tables or analyzing best tiles."""
    parser = argparse.ArgumentParser(
        description="Generate results tables and analyze best tiles from evaluation outputs"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["tables", "best_tiles"],
        default="tables",
        help="Mode: 'tables' to generate CSV/LaTeX tables, 'best_tiles' to find best tiles",
    )

    # Tables mode arguments
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing results.json files (required for tables mode)",
    )

    parser.add_argument(
        "--baseline_method",
        type=str,
        default="gaussian_x0",
        help="Baseline method name for stratified comparison (tables mode)",
    )

    parser.add_argument(
        "--proposed_method",
        type=str,
        default="pg_x0",
        help="Proposed method name for stratified comparison (tables mode)",
    )

    parser.add_argument(
        "--generate_latex",
        action="store_true",
        help="Generate LaTeX table code (tables mode)",
    )

    # Best tiles mode arguments
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to CSV file with metrics (best_tiles mode, optional - uses default paths if not provided)",
    )

    parser.add_argument(
        "--sensor_name",
        type=str,
        default=None,
        help="Sensor name for single sensor analysis (best_tiles mode)",
    )

    # Common arguments
    add_common_visualization_arguments(parser)

    args = parser.parse_args()

    if args.mode == "best_tiles":
        # Best tiles analysis mode
        run_best_tiles_analysis(args)
        return

    # Tables mode (default)
    if args.results_dir is None:
        parser.error("--results_dir is required for tables mode")

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    output_dir = resolve_output_path(args.output_dir, results_dir, create_dir=True)

    # Load results
    logger.info(f"Loading results from {results_dir}")
    results_data = load_results_from_directory(results_dir)
    all_results = results_data["all_results"]

    if not all_results:
        logger.error("No results found!")
        return

    # Generate metrics DataFrame
    logger.info("Extracting metrics into DataFrame")
    df_metrics = extract_metrics_dataframe(all_results)

    if not df_metrics.empty:
        # Save all metrics CSV
        metrics_csv = output_dir / "enhancement_metrics_all_tiles.csv"
        logger.info(f"Saving metrics to {metrics_csv}")

        # Pivot to wide format: one row per tile, columns for each method-metric combination
        if len(df_metrics) > 0:
            df_wide = df_metrics.pivot_table(
                index=["tile_id", "sigma_max_used", "exposure_ratio"],
                columns=["method", "metric_name"],
                values="metric_value",
                aggfunc="first",
            ).reset_index()

            # Flatten column names
            df_wide.columns = [
                "_".join(str(c) for c in col).strip("_")
                if col[0] != "tile_id"
                and col[0] != "sigma_max_used"
                and col[0] != "exposure_ratio"
                else str(col[0])
                for col in df_wide.columns.values
            ]

            df_wide.to_csv(metrics_csv, index=False)
            logger.info(f"  Saved {len(df_wide)} rows to {metrics_csv}")

        # Generate summary statistics
        logger.info("Computing summary statistics")
        df_summary = create_summary_statistics(df_metrics)

        if not df_summary.empty:
            summary_csv = output_dir / "enhancement_summary_statistics.csv"
            logger.info(f"Saving summary to {summary_csv}")
            df_summary.to_csv(summary_csv, index=False)
            logger.info(f"  Saved {len(df_summary)} rows to {summary_csv}")

    # Generate stratified table if available
    stratified_results = results_data["stratified_results"]
    if stratified_results or any("stratified_metrics" in r for r in all_results):
        logger.info("Generating stratified results table (Table 1)")
        stratified_table = generate_stratified_table(
            stratified_results,
            all_results,
            baseline_method=args.baseline_method,
            proposed_method=args.proposed_method,
        )

        if not stratified_table["table_data"].empty:
            # Save CSV
            stratified_csv = output_dir / "stratified_results_table.csv"
            logger.info(f"Saving stratified table to {stratified_csv}")
            stratified_table["table_data"].to_csv(stratified_csv, index=False)

            # Save formatted ASCII table
            formatted_txt = output_dir / "stratified_results_table.txt"
            logger.info(f"Saving formatted table to {formatted_txt}")
            with open(formatted_txt, "w") as f:
                f.write(stratified_table["formatted_table"])

            print("\n" + stratified_table["formatted_table"] + "\n")

            # Generate LaTeX if requested
            if args.generate_latex:
                latex_table = generate_latex_table(stratified_table)
                latex_file = output_dir / "stratified_results_table.tex"
                logger.info(f"Saving LaTeX table to {latex_file}")
                with open(latex_file, "w") as f:
                    f.write(latex_table)

    logger.info(f"Results tables generated in {output_dir}")


if __name__ == "__main__":
    main()
