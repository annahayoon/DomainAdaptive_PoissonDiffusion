#!/usr/bin/env python3
"""
Get real-time GPU node allocation information from SLURM.
"""

import json
import subprocess
import sys
from datetime import datetime


def run_slurm_command(command):
    """Run a SLURM command and return the output."""
    try:
        result = subprocess.run(
            command, shell=False, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Error running command: {command}")
            print(f"Error: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {command}")
        return None
    except Exception as e:
        print(f"Exception running command: {e}")
        return None


def get_node_info():
    """Get detailed node information using sinfo."""
    print("🔍 Getting current node information...")

    # Get node information with GPU details
    sinfo_cmd = ["sinfo", "-N", "-o", "%N %T %G %m %c %O %E", "--noheader"]
    output = run_slurm_command(sinfo_cmd)

    if not output:
        print("❌ Failed to get node information")
        return None

    nodes = []
    for line in output.split("\n"):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) >= 7:
            node_info = {
                "hostname": parts[0],
                "state": parts[1],
                "gres": parts[2] if parts[2] != "N/A" else "",
                "memory": parts[3],
                "cpus": parts[4],
                "load": parts[5],
                "reason": parts[6] if len(parts) > 6 else "",
            }
            nodes.append(node_info)

    return nodes


def get_gpu_nodes(nodes):
    """Filter nodes that have GPUs."""
    gpu_nodes = []

    for node in nodes:
        if "gpu:" in node["gres"].lower():
            # Parse GPU information
            gres = node["gres"]
            if "gpu:" in gres:
                # Extract GPU type and count
                gpu_parts = gres.split("gpu:")[1].split(",")[0]
                if ":" in gpu_parts:
                    gpu_type, gpu_count = gpu_parts.split(":")
                    node["gpu_type"] = gpu_type
                    node["gpu_count"] = int(gpu_count)
                else:
                    node["gpu_type"] = gpu_parts
                    node["gpu_count"] = 1

                gpu_nodes.append(node)

    return gpu_nodes


def get_job_info():
    """Get current job information."""
    print("🔍 Getting current job information...")

    # Get running jobs
    squeue_cmd = ["squeue", "-o", "%i %j %u %T %N %G %M %L", "--noheader"]
    output = run_slurm_command(squeue_cmd)

    if not output:
        print("❌ Failed to get job information")
        return []

    jobs = []
    for line in output.split("\n"):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) >= 8:
            job_info = {
                "job_id": parts[0],
                "name": parts[1],
                "user": parts[2],
                "state": parts[3],
                "nodes": parts[4],
                "gres": parts[5],
                "time": parts[6],
                "time_limit": parts[7],
            }
            jobs.append(job_info)

    return jobs


def get_gpu_utilization():
    """Get GPU utilization information."""
    print("🔍 Getting GPU utilization...")

    # Try to get GPU utilization using scontrol
    scontrol_cmd = ["scontrol", "show", "nodes"]
    output = run_slurm_command(scontrol_cmd)

    return output


def analyze_current_status(gpu_nodes, jobs):
    """Analyze current GPU node status."""

    # GPU performance rankings
    gpu_performance = {
        "A100": 312,
        "A40": 149,
        "V100": 125,
        "A5000": 111,
        "GTX2080TI": 34,
        "GTX1080TI": 22,
        "TITAN": 20,
        "L40": 10,
        "r6k": 10,
    }

    # Calculate power for each node
    for node in gpu_nodes:
        gpu_type = node.get("gpu_type", "Unknown")
        gpu_count = node.get("gpu_count", 1)

        if gpu_type in gpu_performance:
            node["total_power"] = gpu_performance[gpu_type] * gpu_count
            node["single_power"] = gpu_performance[gpu_type]
        else:
            node["total_power"] = 10 * gpu_count
            node["single_power"] = 10

    # Categorize nodes
    available = []
    allocated = []
    mixed = []
    down = []
    other = []

    for node in gpu_nodes:
        state = node["state"]

        if "down" in state.lower() or "drain" in state.lower():
            down.append(node)
        elif "alloc" in state.lower() and "idle" not in state.lower():
            allocated.append(node)
        elif "mix" in state.lower():
            mixed.append(node)
        elif "idle" in state.lower():
            available.append(node)
        else:
            other.append(node)

    return available, allocated, mixed, down, other


def print_results(gpu_nodes, jobs, available, allocated, mixed, down, other):
    """Print the analysis results."""

    print(
        f"\n📊 CURRENT GPU NODE STATUS ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
    )
    print("=" * 80)

    # Status summary
    total_nodes = len(gpu_nodes)
    total_gpus = sum(node.get("gpu_count", 0) for node in gpu_nodes)
    total_power = sum(node.get("total_power", 0) for node in gpu_nodes)

    available_gpus = sum(node.get("gpu_count", 0) for node in available)
    available_power = sum(node.get("total_power", 0) for node in available)

    print(f"Total GPU nodes: {total_nodes}")
    print(f"Total GPUs: {total_gpus}")
    print(f"Total power: {total_power:.0f} TFLOPS")
    print(f"")
    print(
        f"🟢 Available: {len(available)} nodes, {available_gpus} GPUs, {available_power:.0f} TFLOPS"
    )
    print(f"🔴 Allocated: {len(allocated)} nodes")
    print(f"🟡 Mixed: {len(mixed)} nodes")
    print(f"⚫ Down: {len(down)} nodes")
    print(f"❓ Other: {len(other)} nodes")
    print(f"")
    print(f"Utilization: {((total_gpus - available_gpus) / total_gpus * 100):.1f}%")

    # Show available nodes
    if available:
        print(f"\n🎯 AVAILABLE NODES FOR TRAINING:")
        print("-" * 80)
        available_sorted = sorted(
            available, key=lambda x: x.get("total_power", 0), reverse=True
        )

        for i, node in enumerate(available_sorted, 1):
            print(
                f"{i:2d}. {node['hostname']:<20} | {node.get('gpu_count', 0)}x {node.get('gpu_type', 'Unknown'):<10} | "
                f"{node.get('total_power', 0):>6.0f} TFLOPS | {node['memory']} | {node['cpus']} CPUs"
            )

    # Show current jobs
    if jobs:
        print(f"\n📋 CURRENT RUNNING JOBS:")
        print("-" * 80)
        for job in jobs[:10]:  # Show first 10 jobs
            print(
                f"Job {job['job_id']:<8} | {job['user']:<15} | {job['name']:<20} | "
                f"{job['state']:<8} | {job['nodes']:<15} | {job['time']}"
            )
        if len(jobs) > 10:
            print(f"... and {len(jobs) - 10} more jobs")


def main():
    print("🚀 Getting real-time HPC cluster allocation information...")
    print("=" * 60)

    # Get node information
    nodes = get_node_info()
    if not nodes:
        print("❌ Failed to get node information. Make sure you're on the HPC cluster.")
        return

    # Filter GPU nodes
    gpu_nodes = get_gpu_nodes(nodes)
    if not gpu_nodes:
        print("❌ No GPU nodes found.")
        return

    # Get job information
    jobs = get_job_info()

    # Analyze status
    available, allocated, mixed, down, other = analyze_current_status(gpu_nodes, jobs)

    # Print results
    print_results(gpu_nodes, jobs, available, allocated, mixed, down, other)

    # Show commands to get more info
    print(f"\n🔧 USEFUL SLURM COMMANDS:")
    print("-" * 40)
    print("sinfo -N -o '%N %T %G %m %c' --noheader  # Node status")
    print("squeue -u $USER                          # Your jobs")
    print("squeue -o '%i %j %u %T %N %G %M'         # All jobs")
    print("scontrol show node <nodename>            # Detailed node info")
    print("scontrol show job <jobid>                # Detailed job info")


if __name__ == "__main__":
    main()
