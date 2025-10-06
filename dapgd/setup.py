"""
DAPGD Package Setup

PURPOSE: Make dapgd installable as a package
This allows imports like: from dapgd.guidance import PoissonGaussianGuidance

Installation:
    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate

    # Install in development mode
    pip install -e .

    # Install with development dependencies
    pip install -e ".[dev]"
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read the requirements file
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

# Separate core and optional dependencies
core_requirements = []
dev_requirements = []

in_dev_section = False
for req in requirements:
    if "# Testing" in req or "# Development" in req:
        in_dev_section = True
        continue
    if in_dev_section:
        dev_requirements.append(req)
    else:
        core_requirements.append(req)

setup(
    name="dapgd",
    version="0.1.0",
    description="Domain-Adaptive Poisson-Gaussian Diffusion for Photon-Limited Imaging",
    author="Your Name",
    author_email="your.email@institution.edu",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "dev": dev_requirements,
        "all": core_requirements + dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "dapgd-inference=scripts.inference:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
