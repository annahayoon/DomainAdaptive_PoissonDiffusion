#!/usr/bin/env python3
"""
Generate results tables from evaluation outputs.

This script processes results JSON files and generates:
1. CSV table with all metrics per tile/method (enhancement_metrics_all_tiles.csv)
2. Summary statistics CSV (enhancement_summary_statistics.csv)
3. Table 1: Stratified results by ADC bin (for paper)
4. LaTeX-formatted table output (optional)

Based on IMPLEMENTATION_STATUS.md requirements for stratified evaluation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> Dict[str, Any]:
    """
    Load results from a results directory.

    Looks for:
    - results.json (main summary file)
    - stratified_results.json (if stratified evaluation was run)
    - Individual example_*/results.json files

    Args:
        results_dir: Directory containing results

    Returns:
        Dictionary with:
            - 'all_results': List of per-tile results
            - 'summary': Summary dict
            - 'stratified_results': Stratified evaluation results (if available)
    """
    results_data = {"all_results": [], "summary": {}, "stratified_results": None}

    # Try to load main results.json
    main_results_file = results_dir / "results.json"
    if main_results_file.exists():
        logger.info(f"Loading main results from {main_results_file}")
        with open(main_results_file, "r") as f:
            data = json.load(f)

        if "results" in data:
            results_data["all_results"] = data["results"]
            results_data["summary"] = {k: v for k, v in data.items() if k != "results"}
        else:
            # Single result file
            results_data["all_results"] = [data]
            results_data["summary"] = data

    # Try to load stratified results
    stratified_file = results_dir / "stratified_results.json"
    if stratified_file.exists():
        logger.info(f"Loading stratified results from {stratified_file}")
        with open(stratified_file, "r") as f:
            results_data["stratified_results"] = json.load(f)

    # Also look for individual example directories
    example_dirs = list(results_dir.glob("example_*"))
    if example_dirs and len(results_data["all_results"]) == 0:
        logger.info(
            f"Found {len(example_dirs)} example directories, loading individual results"
        )
        for example_dir in example_dirs:
            example_results = example_dir / "results.json"
            if example_results.exists():
                with open(example_results, "r") as f:
                    result = json.load(f)
                    results_data["all_results"].append(result)

    logger.info(f"Loaded {len(results_data['all_results'])} result entries")
    return results_data


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


def main():
    """Main function for generating results tables."""
    parser = argparse.ArgumentParser(
        description="Generate results tables from evaluation outputs"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing results.json and/or example_*/results.json files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated tables (default: same as results_dir)",
    )

    parser.add_argument(
        "--baseline_method",
        type=str,
        default="gaussian_x0",
        help="Baseline method name for stratified comparison",
    )

    parser.add_argument(
        "--proposed_method",
        type=str,
        default="pg_x0",
        help="Proposed method name for stratified comparison",
    )

    parser.add_argument(
        "--generate_latex",
        action="store_true",
        help="Generate LaTeX table code",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    logger.info(f"Loading results from {results_dir}")
    results_data = load_results(results_dir)
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
