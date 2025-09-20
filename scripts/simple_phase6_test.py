#!/usr/bin/env python3
"""
Simple test for Phase 6 implementation.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.baselines import GaussianBaseline, create_baseline_suite
from core.performance import MemoryManager, PerformanceConfig


def main():
    print("Testing Phase 6 implementation...")

    # Test 1: Performance config
    try:
        config = PerformanceConfig()
        print("✅ Performance config created")
    except Exception as e:
        print(f"❌ Performance config failed: {e}")
        return

    # Test 2: Memory manager
    try:
        memory_manager = MemoryManager(config)
        memory_info = memory_manager.get_memory_info()
        print(
            f"✅ Memory manager working, available: {memory_info.get('available_memory', 'unknown')}"
        )
    except Exception as e:
        print(f"❌ Memory manager failed: {e}")

    # Test 3: Simple baseline
    try:
        baseline = GaussianBaseline(device="cpu")
        test_image = torch.rand(1, 1, 32, 32) * 1000 + 100
        result = baseline.denoise(test_image, 1000.0, 100.0, 5.0)
        print(f"✅ Gaussian baseline working, output shape: {result.shape}")
    except Exception as e:
        print(f"❌ Gaussian baseline failed: {e}")

    # Test 4: Baseline suite
    try:
        suite = create_baseline_suite(device="cpu")
        available_methods = [
            name for name, method in suite.items() if method.is_available
        ]
        print(f"✅ Baseline suite created, available methods: {len(available_methods)}")
        print(f"   Methods: {available_methods}")
    except Exception as e:
        print(f"❌ Baseline suite failed: {e}")

    print("\nPhase 6 basic functionality test completed!")


if __name__ == "__main__":
    main()
