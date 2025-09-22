#!/usr/bin/env python3
"""
GPU Performance Monitor for Multi-GPU Training
Helps identify bottlenecks in distributed training
"""

import subprocess
import time
from datetime import datetime

import numpy as np
import psutil
import torch


def get_gpu_stats():
    """Get current GPU utilization stats."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )

        stats = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split(", ")
            stats.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "gpu_util": float(parts[2]),
                    "mem_used": float(parts[3]),
                    "mem_total": float(parts[4]),
                    "temp": float(parts[5]),
                }
            )
        return stats
    except:
        return []


def monitor_training(interval=5, duration=300):
    """
    Monitor GPU performance during training.

    Args:
        interval: Seconds between measurements
        duration: Total monitoring duration in seconds
    """
    print("=" * 80)
    print("GPU PERFORMANCE MONITOR - Multi-GPU Training Optimization")
    print("=" * 80)
    print(f"Monitoring every {interval} seconds for {duration/60:.1f} minutes")
    print("")

    # Track metrics over time
    gpu_utils = []
    mem_utils = []
    cpu_util = []

    start_time = time.time()
    iteration = 0

    while time.time() - start_time < duration:
        iteration += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Get GPU stats
        gpu_stats = get_gpu_stats()
        if gpu_stats:
            # Calculate averages
            avg_gpu_util = np.mean([g["gpu_util"] for g in gpu_stats])
            avg_mem_util = np.mean(
                [g["mem_used"] / g["mem_total"] * 100 for g in gpu_stats]
            )

            gpu_utils.append(avg_gpu_util)
            mem_utils.append(avg_mem_util)

            # Print current stats
            print(f"\n[{timestamp}] Iteration {iteration}")
            print("-" * 50)

            for gpu in gpu_stats:
                mem_pct = (gpu["mem_used"] / gpu["mem_total"]) * 100
                print(
                    f"GPU {gpu['index']}: {gpu['gpu_util']:5.1f}% util | "
                    f"{gpu['mem_used']/1024:5.1f}GB/{gpu['mem_total']/1024:.0f}GB ({mem_pct:4.1f}%) | "
                    f"{gpu['temp']:.0f}°C"
                )

            print(
                f"\nAverages: GPU Util: {avg_gpu_util:.1f}% | Mem Util: {avg_mem_util:.1f}%"
            )

            # Identify bottlenecks
            if avg_gpu_util < 50:
                print("⚠️  LOW GPU UTILIZATION - Likely data loading bottleneck!")
                print("   Fix: Increase num_workers or optimize data pipeline")
            elif avg_gpu_util < 80:
                print("⚠️  MODERATE GPU UTILIZATION - Room for improvement")
                print(
                    "   Check: Data loading, CPU bottlenecks, or communication overhead"
                )
            else:
                print("✅ GOOD GPU UTILIZATION - GPUs are well-utilized")

        # Get CPU stats
        cpu = psutil.cpu_percent(interval=0.1)
        cpu_util.append(cpu)
        ram = psutil.virtual_memory()

        print(
            f"\nSystem: CPU: {cpu:.1f}% | RAM: {ram.percent:.1f}% ({ram.used/1e9:.1f}GB/{ram.total/1e9:.0f}GB)"
        )

        # Check for CPU bottleneck
        if cpu > 90:
            print("⚠️  HIGH CPU USAGE - May be bottlenecking GPU training")

        time.sleep(interval)

    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    if gpu_utils:
        print(
            f"Average GPU Utilization: {np.mean(gpu_utils):.1f}% (std: {np.std(gpu_utils):.1f}%)"
        )
        print(f"Average Memory Utilization: {np.mean(mem_utils):.1f}%")
        print(f"Average CPU Utilization: {np.mean(cpu_util):.1f}%")

        # Performance diagnosis
        avg_gpu = np.mean(gpu_utils)
        if avg_gpu < 50:
            print("\n❌ POOR PERFORMANCE - Major bottleneck detected!")
            print("   LIKELY CAUSE: Data loading (num_workers too low)")
            print("   SOLUTION: Set num_workers=8-16 for 4 GPUs")
        elif avg_gpu < 80:
            print("\n⚠️  SUBOPTIMAL PERFORMANCE - Can be improved")
            print("   POSSIBLE CAUSES:")
            print("   - Insufficient data loading workers")
            print("   - pin_memory=false slowing transfers")
            print("   - NCCL communication overhead")
        else:
            print("\n✅ EXCELLENT PERFORMANCE - Well optimized!")
            print("   GPUs are being efficiently utilized")

    print("\nMonitoring complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor GPU performance during training"
    )
    parser.add_argument(
        "--interval", type=int, default=5, help="Seconds between measurements"
    )
    parser.add_argument(
        "--duration", type=int, default=300, help="Total monitoring duration in seconds"
    )

    args = parser.parse_args()
    monitor_training(args.interval, args.duration)
