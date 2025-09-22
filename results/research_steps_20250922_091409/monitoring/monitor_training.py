import json
import time
from pathlib import Path

import psutil
import torch


def monitor_training(log_file="logs/training_photography.log"):
    """Monitor training progress in real-time."""
    print("📊 Real-time Training Monitor")
    print("=" * 50)

    last_epoch = -1
    last_loss = None

    while True:
        try:
            # Check if training is still running
            if not any(
                "python" in p.info["name"] and "train" in " ".join(p.cmdline())
                for p in psutil.process_iter(["pid", "name", "cmdline"])
            ):
                print("⏹️  Training process ended")
                break

            # Monitor GPU if available
            if torch.cuda.is_available():
                gpu_memory = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.mem_get_info()[0]
                )
                gpu_util = torch.cuda.utilization()
                print(
                    f"🖥️  GPU Memory: {gpu_memory / 1e9:.1f} GB used, Utilization: {gpu_util}%"
                )

            # Monitor CPU and RAM
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            print(f"💻 CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}%")

            # Check for new log entries (simplified)
            if Path(log_file).exists():
                print("📝 Training log updated...")

            print("-" * 30)
            time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            print("⏹️  Monitor stopped by user")
            break
        except Exception as e:
            print(f"⚠️  Monitor error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    monitor_training()
