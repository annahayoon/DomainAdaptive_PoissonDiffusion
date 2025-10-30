"""
Utilities for extracting and formatting metrics.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional


def extract_metrics_from_json(
    results_file: Path, method: str, method_key_map: Optional[Dict[str, str]] = None
) -> Optional[Dict]:
    """
    Extract metrics for a specific method from results.json.

    Args:
        results_file: Path to results.json
        method: Method name (e.g., 'pg_x0', 'gaussian_x0')
        method_key_map: Optional mapping from method names to JSON keys

    Returns:
        Dictionary of metrics or None if not found
    """
    if not results_file.exists():
        return None

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        comprehensive_metrics = data.get("comprehensive_metrics", {})

        # Default method key mapping
        if method_key_map is None:
            method_key_map = {
                "noisy": "noisy",
                "clean": "clean",
                "exposure_scaled": "exposure_scaled",
                "gaussian_x0": "gaussian_x0",
                "pg_x0_single": "pg_x0",
                "pg_x0_cross": "pg_x0_cross",
                "gaussian_x0_cross": "gaussian_x0_cross",
            }

        if method in method_key_map:
            key = method_key_map[method]
            return comprehensive_metrics.get(key)

    except Exception as e:
        print(f"Error extracting metrics for {method}: {e}")

    return None


def extract_pixel_range(tile_path: Path) -> Optional[Dict]:
    """Extract pixel range from results.json."""
    results_file = tile_path / "results.json"
    if not results_file.exists():
        return None

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        brightness = data.get("brightness_analysis", {})
        return {
            "min": brightness.get("min", 0),
            "max": brightness.get("max", 1),
            "mean": brightness.get("mean", 0),
            "std": brightness.get("std", 0),
        }
    except:
        return None


def format_pixel_range(pixel_range: Dict) -> str:
    """Format pixel range for display."""
    return f"[{pixel_range['min']:.0f}, {pixel_range['max']:.0f}]"


def format_metrics(metrics_dict: Dict, methods_to_show: List[str]) -> str:
    """Format metrics string for display."""
    lines = []
    for method in methods_to_show:
        if method in metrics_dict:
            m = metrics_dict[method]
            line = f"{method.replace('_', '-').replace('x0', 'x0')}: "
            line += f"PSNR={m.get('psnr', 0):.1f}, SSIM={m.get('ssim', 0):.3f}, "
            line += f"LPIPS={m.get('lpips', 0):.3f}, NIQE={m.get('niqe', 'N/A')}"
            lines.append(line)
    return "\n".join(lines)


def format_metric_value(value, metric_name: str = "") -> str:
    """
    Format a single metric value for display.

    Args:
        value: Metric value
        metric_name: Name of metric (for special formatting)

    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"

    if isinstance(value, str):
        return value

    if metric_name == "psnr":
        return f"{value:.1f}"
    elif metric_name in ["ssim", "lpips"]:
        return f"{value:.3f}"
    elif metric_name == "niqe":
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        return str(value)
    else:
        return f"{value:.3f}"
