#!/usr/bin/env python3
"""
Setup script for Domain-Adaptive Poisson-Gaussian Diffusion project.
"""

import os

from setuptools import find_packages, setup


# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="poisson-diffusion",
    version="0.1.0",
    description="Domain-Adaptive Poisson-Gaussian Diffusion for Low-Light Image Restoration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Research Team",
    author_email="research@example.com",
    url="https://github.com/example/poisson-diffusion",
    # Package configuration
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    # Entry points for CLI tools
    entry_points={
        "console_scripts": [
            "poisson-train=scripts.train_prior:main",
            "poisson-denoise=scripts.denoise:main",
            "poisson-evaluate=scripts.evaluate:main",
        ],
    },
    # Package data
    package_data={
        "configs": ["*.yaml", "*.json"],
        "": ["*.md", "*.txt"],
    },
    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "wandb": [
            "wandb>=0.13.0",
        ],
    },
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    # Keywords
    keywords="diffusion models, image restoration, low-light, poisson noise, computer vision",
)
