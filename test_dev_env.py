#!/usr/bin/env python3
"""
Quick test script to verify development environment setup.
"""

import importlib
import sys
from typing import List, Tuple


def test_import(module_name: str, description: str = "") -> Tuple[bool, str]:
    """Test if a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        desc = description or module_name
        return True, f"✓ {desc} ({version})"
    except ImportError as e:
        desc = description or module_name
        return False, f"✗ {desc} - {str(e)}"


def main():
    """Run environment verification tests."""
    print("Development Environment Verification")
    print("=" * 50)

    # Core ML libraries
    tests = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
    ]

    # Domain-specific libraries
    domain_tests = [
        ("rawpy", "RAW Photography"),
        ("astropy", "Astronomy FITS"),
        ("PIL", "Pillow (Image Processing)"),
        ("tifffile", "TIFF Microscopy"),
        ("scipy", "SciPy"),
        ("skimage", "Scikit-Image"),
    ]

    # Development tools
    dev_tests = [
        ("pytest", "Testing Framework"),
        ("black", "Code Formatter"),
        ("isort", "Import Sorter"),
        ("yaml", "YAML Parser"),
        ("tqdm", "Progress Bars"),
    ]

    # Optional libraries
    optional_tests = [
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("jupyter", "Jupyter"),
        ("h5py", "HDF5 Support"),
    ]

    all_tests = [
        ("Core ML Libraries", tests),
        ("Domain-Specific Libraries", domain_tests),
        ("Development Tools", dev_tests),
        ("Optional Libraries", optional_tests),
    ]

    total_passed = 0
    total_tests = 0

    for category, test_list in all_tests:
        print(f"\n{category}:")
        category_passed = 0

        for module_name, description in test_list:
            success, message = test_import(module_name, description)
            print(f"  {message}")
            if success:
                category_passed += 1
            total_tests += 1

        total_passed += category_passed
        print(f"  → {category_passed}/{len(test_list)} passed")

    # Special tests
    print(f"\nSpecial Tests:")

    # CUDA availability
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = (
                torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            )
            print(f"  ✓ CUDA available ({device_count} devices: {device_name})")
        else:
            print(f"  ⚠ CUDA not available (CPU only)")
    except:
        print(f"  ✗ Could not check CUDA availability")

    # Python version
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if sys.version_info >= (3, 8):
        print(f"  ✓ Python version {python_version} (>= 3.8)")
    else:
        print(f"  ✗ Python version {python_version} (< 3.8 required)")

    # EDM integration
    try:
        sys.path.insert(0, "external/edm")
        from training.networks import EDMPrecond

        print(f"  ✓ EDM integration available")
    except ImportError:
        print(f"  ⚠ EDM integration not available (run external/setup_edm.sh)")

    # Summary
    print(f"\n" + "=" * 50)
    print(f"Summary: {total_passed}/{total_tests} imports successful")

    if total_passed == total_tests:
        print("🎉 Environment setup is complete and working!")
        return True
    else:
        missing = total_tests - total_passed
        print(f"⚠ {missing} libraries missing. Run setup_dev_env.sh to install.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
