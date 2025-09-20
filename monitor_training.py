#!/usr/bin/env python3
"""
Advanced training monitor for Poisson-Gaussian model training.

This script provides:
- Real-time GPU/CPU/memory monitoring
- Training progress tracking from logs
- Performance metrics visualization
- Early stopping detection
- Alert system for issues

Usage:
    python monitor_training.py --log_file logs/training_photography.log
"""

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import psutil
import torch


class TrainingMonitor:
    """Advanced training monitor with real-time metrics."""

    def __init__(
        self,
        log_file: str = "logs/training_photography.log",
        update_interval: int = 30,
        enable_alerts: bool = True,
        alert_thresholds: Dict[str, float] = None,
    ):
        """
        Initialize training monitor.

        Args:
            log_file: Path to training log file
            update_interval: Update interval in seconds
            enable_alerts: Whether to enable alert system
            alert_thresholds: Alert thresholds for different metrics
        """
        self.log_file = Path(log_file)
        self.update_interval = update_interval
        self.enable_alerts = enable_alerts
        self.alert_thresholds = alert_thresholds or {
            "gpu_memory_percent": 90.0,
            "cpu_percent": 95.0,
            "ram_percent": 90.0,
            "training_stuck_hours": 2.0,
        }

        # Monitoring state
        self.running = True
        self.last_update_time = time.time()
        self.last_epoch_time = time.time()
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0
        self.training_stuck_since = None

        # Metrics history
        self.metrics_history = defaultdict(list)
        self.alert_history = []

        # Performance tracking
        self.start_time = time.time()
        self.epoch_times = []

        print("📊 Advanced Training Monitor Initialized")
        print("=" * 50)
        print(f"Log file: {self.log_file}")
        print(f"Update interval: {self.update_interval}s")
        print(f"Alerts enabled: {self.enable_alerts}")
        print("=" * 50)

    def parse_training_progress(self) -> Dict[str, Any]:
        """Parse training progress from log file."""
        if not self.log_file.exists():
            return {}

        progress = {
            "current_epoch": 0,
            "total_epochs": 100,
            "best_val_loss": float("inf"),
            "current_val_loss": None,
            "training_loss": None,
            "learning_rate": None,
            "early_stopping_counter": 0,
            "domain_stats": {},
            "recent_messages": [],
        }

        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()

            # Parse last 100 lines for recent info
            recent_lines = lines[-100:] if len(lines) > 100 else lines

            for line in reversed(recent_lines):
                line = line.strip()

                # Extract epoch information
                if "Epoch" in line and "/" in line:
                    try:
                        epoch_str = line.split("Epoch")[1].split("/")[0].strip()
                        progress["current_epoch"] = int(epoch_str)
                    except (ValueError, IndexError):
                        pass

                # Extract loss information
                if "val_loss" in line.lower() or "validation loss" in line.lower():
                    try:
                        loss = float(line.split(":")[-1].strip().split()[0])
                        progress["current_val_loss"] = loss
                        if loss < progress["best_val_loss"]:
                            progress["best_val_loss"] = loss
                    except (ValueError, IndexError):
                        pass

                # Extract training loss
                if "training_loss" in line.lower() or "train_loss" in line.lower():
                    try:
                        loss = float(line.split(":")[-1].strip().split()[0])
                        progress["training_loss"] = loss
                    except (ValueError, IndexError):
                        pass

                # Extract domain statistics
                if "domain distribution" in line.lower():
                    progress["domain_stats"] = self._parse_domain_stats(line)

                # Collect recent messages
                if any(
                    keyword in line.lower()
                    for keyword in [
                        "error",
                        "warning",
                        "failed",
                        "exception",
                        "nan",
                        "inf",
                    ]
                ):
                    progress["recent_messages"].append(line)

            # Limit recent messages
            progress["recent_messages"] = progress["recent_messages"][-5:]

        except Exception as e:
            progress["parse_error"] = str(e)

        return progress

    def _parse_domain_stats(self, line: str) -> Dict[str, float]:
        """Parse domain statistics from log line."""
        stats = {}
        try:
            # Example: "photography: 45.2%, microscopy: 32.1%, astronomy: 22.7%"
            parts = line.split(":")
            if len(parts) > 1:
                domain_part = parts[1]
                for item in domain_part.split(","):
                    if ":" in item:
                        domain, percentage = item.strip().split(":")
                        domain = domain.strip()
                        percentage = percentage.strip().rstrip("%")
                        stats[domain] = float(percentage)
        except Exception:
            pass
        return stats

    def check_system_health(self) -> Dict[str, Any]:
        """Check system health metrics."""
        health = {}

        # GPU metrics
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_used = gpu_memory - torch.cuda.mem_get_info()[0]
            gpu_percent = (gpu_used / gpu_memory) * 100

            health["gpu_memory_mb"] = gpu_used / 1024 / 1024
            health["gpu_memory_percent"] = gpu_percent
            health["gpu_utilization"] = (
                torch.cuda.utilization() if hasattr(torch.cuda, "utilization") else 0
            )

        # CPU and RAM metrics
        health["cpu_percent"] = psutil.cpu_percent(interval=1)
        health["ram_percent"] = psutil.virtual_memory().percent
        health["ram_mb"] = psutil.virtual_memory().used / 1024 / 1024

        # Process information
        try:
            current_process = psutil.Process(os.getpid())
            health["process_memory_mb"] = (
                current_process.memory_info().rss / 1024 / 1024
            )
            health["process_cpu_percent"] = current_process.cpu_percent()
        except:
            health["process_memory_mb"] = 0
            health["process_cpu_percent"] = 0

        return health

    def check_alerts(
        self, progress: Dict[str, Any], health: Dict[str, Any]
    ) -> List[str]:
        """Check for alert conditions."""
        alerts = []

        # GPU memory alert
        if (
            health.get("gpu_memory_percent", 0)
            > self.alert_thresholds["gpu_memory_percent"]
        ):
            alerts.append(
                f"⚠️  High GPU memory usage: {health['gpu_memory_percent']:.1f}%"
            )

        # CPU usage alert
        if health.get("cpu_percent", 0) > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"⚠️  High CPU usage: {health['cpu_percent']:.1f}%")

        # RAM usage alert
        if health.get("ram_percent", 0) > self.alert_thresholds["ram_percent"]:
            alerts.append(f"⚠️  High RAM usage: {health['ram_percent']:.1f}%")

        # Training stuck detection
        time_since_update = time.time() - self.last_update_time
        if time_since_update > self.alert_thresholds["training_stuck_hours"] * 3600:
            alerts.append(
                f"⚠️  Training may be stuck - no progress for {time_since_update / 3600:.1f} hours"
            )

        # Validation loss issues
        if progress.get("current_val_loss") is not None:
            val_loss = progress["current_val_loss"]
            if not torch.isfinite(torch.tensor(val_loss)):
                alerts.append(f"⚠️  Invalid validation loss: {val_loss}")

        # Domain imbalance
        domain_stats = progress.get("domain_stats", {})
        if domain_stats:
            percentages = list(domain_stats.values())
            min_pct, max_pct = min(percentages), max(percentages)
            if max_pct - min_pct > 50:  # More than 50% difference
                alerts.append(
                    f"⚠️  Domain imbalance detected: {min_pct:.1f}% - {max_pct:.1f}%"
                )

        return alerts

    def print_status(self):
        """Print current training status."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        print(f"\n{'='*60}")
        print("📊 TRAINING STATUS REPORT")
        print(f"{'='*60}")
        print(f"Elapsed time: {elapsed_time / 3600:.2f} hours")
        print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Training progress
        progress = self.parse_training_progress()
        if progress.get("current_epoch"):
            epoch = progress["current_epoch"]
            total = progress.get("total_epochs", 100)
            progress_pct = (epoch / total) * 100
            print(f"Epoch: {epoch}/{total} ({progress_pct:.1f}%)")

        # Losses
        if progress.get("training_loss") is not None:
            print(f"Training loss: {progress['training_loss']:.6f}")
        if progress.get("current_val_loss") is not None:
            val_loss = progress["current_val_loss"]
            best_loss = progress.get("best_val_loss", float("inf"))
            print(f"Validation loss: {val_loss:.6f}")
            print(f"Best validation loss: {best_loss:.6f}")

        if progress.get("early_stopping_counter"):
            print(f"Early stopping counter: {progress['early_stopping_counter']}")

        # System health
        health = self.check_system_health()
        print("\n🖥️  System Health:")
        if "gpu_memory_percent" in health:
            print(f"  GPU Memory: {health['gpu_memory_percent']:.1f}%")
        print(f"  CPU Usage: {health['cpu_percent']:.1f}%")
        print(f"  RAM Usage: {health['ram_percent']:.1f}%")

        # Domain statistics
        if progress.get("domain_stats"):
            print("\n📈 Domain Distribution:")
            for domain, pct in progress["domain_stats"].items():
                print(f"  {domain}: {pct:.1f}%")

        # Alerts
        alerts = self.check_alerts(progress, health)
        if alerts:
            print("\n🚨 ALERTS:")
            for alert in alerts:
                print(f"  {alert}")

        # Recent messages
        if progress.get("recent_messages"):
            print("\n📝 Recent Log Messages:")
            for msg in progress["recent_messages"]:
                print(f"  {msg}")

        print(f"{'='*60}")

    def run_monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.running:
                self.print_status()
                time.sleep(self.update_interval)

        except KeyboardInterrupt:
            print("\n⏹️  Monitor stopped by user")
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")
        finally:
            self.print_final_summary()

    def print_final_summary(self):
        """Print final training summary."""
        print("\n" + "=" * 60)
        print("🎉 TRAINING MONITOR FINAL SUMMARY")
        print("=" * 60)

        elapsed_time = time.time() - self.start_time
        progress = self.parse_training_progress()

        print(f"Total monitoring time: {elapsed_time / 3600:.2f} hours")
        print(f"Final epoch: {progress.get('current_epoch', 'Unknown')}")
        print(f"Best validation loss: {progress.get('best_val_loss', 'Unknown')}")

        if self.alert_history:
            print(f"Total alerts: {len(self.alert_history)}")
            for i, alert in enumerate(self.alert_history[-5:], 1):
                print(f"  {i}. {alert['time']}: {alert['message']}")

        print("=" * 60)
        print("📁 Check logs and TensorBoard for detailed metrics")
        print("=" * 60)


def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description="Advanced training monitor")
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/training_photography.log",
        help="Path to training log file",
    )
    parser.add_argument(
        "--interval", type=int, default=30, help="Update interval in seconds"
    )
    parser.add_argument("--no-alerts", action="store_true", help="Disable alert system")
    parser.add_argument(
        "--alert-thresholds", type=str, help="JSON string of alert thresholds"
    )

    args = parser.parse_args()

    # Parse alert thresholds if provided
    alert_thresholds = None
    if args.alert_thresholds:
        try:
            alert_thresholds = json.loads(args.alert_thresholds)
        except json.JSONDecodeError:
            print("❌ Invalid alert thresholds JSON")
            return

    # Create and run monitor
    monitor = TrainingMonitor(
        log_file=args.log_file,
        update_interval=args.interval,
        enable_alerts=not args.no_alerts,
        alert_thresholds=alert_thresholds,
    )

    try:
        monitor.run_monitoring_loop()
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Monitoring failed: {e}")
        raise


if __name__ == "__main__":
    main()
