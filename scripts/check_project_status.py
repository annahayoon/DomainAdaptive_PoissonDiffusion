#!/usr/bin/env python3
"""
Check project status and verify all components are properly set up.
"""

import importlib
import sys
from pathlib import Path
from typing import List, Tuple


def check_directory_structure() -> Tuple[bool, List[str]]:
    """Check that all required directories exist."""
    required_dirs = [
        "core",
        "models",
        "data",
        "configs",
        "scripts",
        "tests",
        "docs",
        "external",
    ]

    missing = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing.append(directory)

    return len(missing) == 0, missing


def check_core_modules() -> Tuple[bool, List[str]]:
    """Check that core modules can be imported."""
    modules = [
        "core.interfaces",
        "core.exceptions",
        "core.utils",
    ]

    failed = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as e:
            failed.append(f"{module}: {e}")

    return len(failed) == 0, failed


def check_external_dependencies() -> Tuple[bool, List[str]]:
    """Check external dependencies."""
    issues = []

    # Check EDM integration
    try:
        sys.path.insert(0, "external/edm")
        from training.networks import EDMPrecond

        issues.append("✓ EDM integration available")
    except ImportError:
        issues.append("⚠ EDM integration not available")

    # Check key libraries
    libraries = [
        ("torch", "PyTorch"),
        ("rawpy", "RAW Photography"),
        ("astropy", "Astronomy"),
        ("tifffile", "Microscopy"),
    ]

    for module, name in libraries:
        try:
            importlib.import_module(module)
            issues.append(f"✓ {name}")
        except ImportError:
            issues.append(f"✗ {name} missing")

    return True, issues  # Always return True, just report status


def check_configuration() -> Tuple[bool, List[str]]:
    """Check configuration files."""
    config_files = [
        "configs/default.yaml",
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
    ]

    missing = []
    for config_file in config_files:
        if not Path(config_file).exists():
            missing.append(config_file)

    return len(missing) == 0, missing


def run_tests() -> Tuple[bool, str]:
    """Run basic tests."""
    import subprocess

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/test_interfaces.py", "-v"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return True, "All tests passed"
        else:
            return False, f"Tests failed:\n{result.stdout}\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Tests timed out"
    except Exception as e:
        return False, f"Error running tests: {e}"


def main():
    """Run all status checks."""
    print("Project Status Check")
    print("=" * 50)

    checks = [
        ("Directory Structure", check_directory_structure),
        ("Core Modules", check_core_modules),
        ("External Dependencies", check_external_dependencies),
        ("Configuration Files", check_configuration),
        ("Basic Tests", run_tests),
    ]

    all_passed = True

    for check_name, check_func in checks:
        print(f"\n{check_name}:")

        try:
            success, details = check_func()

            if success:
                print(f"  ✓ PASS")
            else:
                print(f"  ✗ FAIL")
                all_passed = False

            if isinstance(details, list):
                for detail in details:
                    print(f"    {detail}")
            else:
                print(f"    {details}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Project setup is complete and ready for development!")
        print("\nNext steps:")
        print("  - Proceed to Task 1.2: Implement error handling framework")
        print("  - Or continue with Task 1.3: Implement reversible transforms")
    else:
        print("⚠ Some issues found. Please address them before continuing.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
