#!/usr/bin/env python
"""
Profile DAPGD inference to identify bottlenecks

Usage:
    python scripts/profile_inference.py --checkpoint model.pt --size 256
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dapgd.guidance.pg_guidance import (
    PoissonGaussianGuidance,
    simulate_poisson_gaussian_noise,
)
from dapgd.sampling.dapgd_sampler import DAPGDSampler
from dapgd.sampling.edm_wrapper import EDMModelWrapper


def profile_sampling(args):
    """Profile the sampling process"""

    print("=" * 60)
    print("DAPGD Performance Profiling")
    print("=" * 60)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Extract network
    if "ema" in checkpoint:
        network = checkpoint["ema"]
    elif "model" in checkpoint:
        network = checkpoint["model"]
    else:
        network = checkpoint

    # Create wrapper and sampler
    edm_wrapper = EDMModelWrapper(network, img_channels=3)
    edm_wrapper.eval()

    guidance = PoissonGaussianGuidance(s=1000.0, sigma_r=5.0, kappa=0.5)

    sampler = DAPGDSampler(
        edm_wrapper=edm_wrapper,
        guidance=guidance,
        num_steps=args.num_steps,
        device=args.device,
    )

    # Create test input
    clean = torch.rand(1, 3, args.size, args.size).to(args.device)
    y_e = simulate_poisson_gaussian_noise(clean, s=1000.0, sigma_r=5.0)

    # Warm-up
    print("\nWarming up...")
    for _ in range(3):
        _ = sampler.sample(y_e=y_e, show_progress=False)

    # Profile with PyTorch profiler
    print("\nProfiling with PyTorch Profiler...")

    activities = [ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities, record_shapes=True, profile_memory=True, with_stack=True
    ) as prof:
        with record_function("full_sampling"):
            restored = sampler.sample(y_e=y_e, show_progress=False)

    # Print results
    print("\n" + "=" * 60)
    print("Top 10 operations by CPU time:")
    print("=" * 60)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if args.device == "cuda":
        print("\n" + "=" * 60)
        print("Top 10 operations by CUDA time:")
        print("=" * 60)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Memory usage
    if args.device == "cuda":
        print("\n" + "=" * 60)
        print("Memory Usage:")
        print("=" * 60)
        print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Current memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Timing breakdown
    print("\n" + "=" * 60)
    print("Timing Breakdown:")
    print("=" * 60)

    times = {}

    # Time individual components
    with torch.no_grad():
        # Denoising
        x_test = torch.randn_like(y_e)
        sigma_test = 1.0

        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = sampler.edm_wrapper.denoise(x_test, sigma_test)
        if args.device == "cuda":
            torch.cuda.synchronize()
        times["denoising_per_step"] = (time.time() - start) / 10

        # Guidance
        if sampler.guidance:
            x_clean = torch.rand_like(y_e)

            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                _ = sampler.guidance._compute_gradient(x_clean, y_e)
            if args.device == "cuda":
                torch.cuda.synchronize()
            times["guidance_per_step"] = (time.time() - start) / 10

        # Full sampling
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(5):
            _ = sampler.sample(y_e=y_e, show_progress=False)
        if args.device == "cuda":
            torch.cuda.synchronize()
        times["full_sampling"] = (time.time() - start) / 5

    for name, duration in times.items():
        print(f"{name}: {duration*1000:.2f} ms")

    # Estimate total time breakdown
    if "denoising_per_step" in times:
        total_denoising = times["denoising_per_step"] * args.num_steps
        print(
            f"\nEstimated time in denoising: {total_denoising*1000:.2f} ms "
            f"({total_denoising/times['full_sampling']*100:.1f}%)"
        )

    if "guidance_per_step" in times:
        total_guidance = times["guidance_per_step"] * args.num_steps
        print(
            f"Estimated time in guidance: {total_guidance*1000:.2f} ms "
            f"({total_guidance/times['full_sampling']*100:.1f}%)"
        )

    # Export Chrome trace for detailed analysis
    trace_path = Path(args.output_dir) / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nChrome trace saved to: {trace_path}")
    print("View at: chrome://tracing")

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="experiments/profiling")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    profile_sampling(args)
