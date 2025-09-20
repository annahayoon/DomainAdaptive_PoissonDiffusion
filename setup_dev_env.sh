#!/bin/bash
# Development environment setup script for Poisson-Gaussian Diffusion project

set -e  # Exit on any error

echo "Setting up development environment for Poisson-Gaussian Diffusion..."
echo "=================================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
elif [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    echo "✓ Conda environment detected: $CONDA_DEFAULT_ENV"
else
    echo "Warning: No virtual environment detected. Consider creating one:"
    echo "  python3 -m venv venv && source venv/bin/activate"
    echo "  OR"
    echo "  conda create -n poisson-diffusion python=3.11 && conda activate poisson-diffusion"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install core dependencies
echo "Installing core dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (adjust for your system)
echo "Installing PyTorch with CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected, installing CPU version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install domain-specific libraries
echo "Installing domain-specific libraries..."

# Photography (RAW files)
pip install rawpy

# Astronomy (FITS files)
pip install astropy

# Microscopy and general image processing
pip install pillow tifffile scikit-image

# Scientific computing
pip install scipy numpy

# Utilities
pip install tqdm click pyyaml h5py

# Development tools
echo "Installing development tools..."
pip install pytest pytest-cov black isort flake8 mypy

# Jupyter for analysis
pip install jupyter matplotlib seaborn

# Install project in development mode
echo "Installing project in development mode..."
pip install -e .

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
pip install pre-commit
pre-commit install

# Verify installations
echo "Verifying installations..."

# Test PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test domain-specific libraries
python3 -c "import rawpy; print('✓ rawpy (photography)')"
python3 -c "import astropy; print('✓ astropy (astronomy)')"
python3 -c "import tifffile; print('✓ tifffile (microscopy)')"

# Test scientific libraries
python3 -c "import numpy, scipy, sklearn; print('✓ Scientific libraries')"

# Test development tools
python3 -c "import pytest, black, isort; print('✓ Development tools')"

# Create basic directory structure
echo "Creating project directory structure..."
mkdir -p core models data configs scripts tests docs

# Create __init__.py files
touch core/__init__.py
touch models/__init__.py
touch data/__init__.py

# Create basic configuration
echo "Creating basic configuration files..."

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints

# Testing
.coverage
htmlcov/
.tox/
.pytest_cache/
.mypy_cache/

# Data and models
data/raw/
data/processed/
models/checkpoints/
*.pth
*.pt
*.h5
*.hdf5

# Logs and outputs
logs/
outputs/
wandb/
*.log

# OS
.DS_Store
Thumbs.db

# Project specific
external/edm/
!external/README.md
!external/*.py
!external/*.sh
!external/*.md
EOF

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Summary:"
echo "  ✓ Python dependencies installed"
echo "  ✓ PyTorch with CUDA support"
echo "  ✓ Domain-specific libraries (rawpy, astropy, tifffile)"
echo "  ✓ Development tools (pytest, black, isort, flake8, mypy)"
echo "  ✓ Pre-commit hooks configured"
echo "  ✓ Project structure created"
echo "  ✓ Git configuration added"
echo ""
echo "Next steps:"
echo "  1. Run tests: pytest"
echo "  2. Format code: black ."
echo "  3. Check types: mypy core/ models/ data/"
echo "  4. Start development: proceed to Phase 1 tasks"
echo ""
echo "Useful commands:"
echo "  - Run all checks: pre-commit run --all-files"
echo "  - Install in dev mode: pip install -e ."
echo "  - Run tests with coverage: pytest --cov"
