#!/usr/bin/env python3
"""Convert RAW files to PNG images for baseline comparison using rawpy.postprocess()."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import get_logger
from core.utils.sensor_utils import process_sensor_baseline_conversion

logger = get_logger(__name__)
FORMAT_CONFIGS = {
    "uint8_png": {
        "name": "uint8 PNG [0,255]",
        "baselines": "SNR-Aware-Low-Light-Enhance, Zero-DCE, ECAFormer",
    },
    "float32_png": {
        "name": "float32 PNG [0,1]",
        "baselines": "RetinexNet, KinD",
    },
}


def print_summary(all_stats: Dict[str, Dict], output_base: Path) -> None:
    """Print conversion summary."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Conversion Summary")
    logger.info("=" * 70)

    for sensor, stats in all_stats.items():
        logger.info(f"\n{sensor.upper()}:")
        for key, stat in stats.items():
            logger.info(
                f"  {key}: {stat['success']} success, "
                f"{stat['failed']} failed, {stat['skipped']} skipped"
            )

    logger.info("")
    logger.info("=" * 70)
    logger.info("Conversion complete!")
    logger.info(f"Output directories:")
    for format_type, config in FORMAT_CONFIGS.items():
        logger.info(
            f"  - {config['name']}: {output_base}/{format_type}/{{long,short}}/"
        )
        logger.info(f"    For: {config['baselines']}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert RAW files to PNG tiles for baseline comparison using rawpy.postprocess()"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory containing processed data (default: dataset/processed)",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        choices=["sony", "fuji"],
        default=["sony", "fuji"],
        help="Sensor types to process (default: both)",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=None,
        help="Directory containing split files (default: dataset/splits)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=list(FORMAT_CONFIGS.keys()),
        default=None,
        help="Formats to convert (default: all). Use 'float32_png' to convert only from uint8_png files.",
    )

    args = parser.parse_args()

    # Determine paths
    data_root = (
        Path(args.data_root)
        if args.data_root
        else project_root / "dataset" / "processed"
    )
    splits_dir = (
        Path(args.splits_dir)
        if args.splits_dir
        else project_root / "dataset" / "splits"
    )
    output_base = data_root

    logger.info("=" * 70)
    logger.info("Baseline Dataset Conversion")
    logger.info("=" * 70)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Splits dir: {splits_dir}")
    logger.info(f"Output base: {output_base}")
    logger.info(f"Sensors: {args.sensors}")
    logger.info(f"Formats: {args.formats if args.formats else 'all'}")

    # Process each sensor
    all_stats = {}
    for sensor in args.sensors:
        sensor_stats = process_sensor_baseline_conversion(
            sensor,
            data_root,
            splits_dir,
            output_base,
            FORMAT_CONFIGS,
            formats=args.formats,
        )
        if sensor_stats:
            all_stats[sensor] = sensor_stats

    # Print summary
    if all_stats:
        print_summary(all_stats, output_base)


if __name__ == "__main__":
    main()
