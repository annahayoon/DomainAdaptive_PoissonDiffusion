#!/bin/bash
# Setup script for EDM integration

set -e  # Exit on any error

echo "Setting up EDM integration for Poisson-Gaussian Diffusion project..."
echo "=================================================================="

# Check if we're in the right directory
if [ ! -f "test_edm_integration.py" ]; then
    echo "Error: Please run this script from the external/ directory"
    exit 1
fi

# Clone EDM repository if it doesn't exist
if [ ! -d "edm" ]; then
    echo "Cloning EDM repository..."
    git clone https://github.com/NVlabs/edm.git
    echo "✓ EDM repository cloned"
else
    echo "✓ EDM repository already exists"
fi

# Enter EDM directory
cd edm

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Warning: requirements.txt not found in EDM repository"
    echo "Creating minimal requirements for EDM..."
    cat > requirements.txt << EOF
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pillow>=8.3.0
scipy>=1.7.0
tqdm>=4.62.0
click>=8.0.0
EOF
fi

# Install EDM dependencies
echo "Installing EDM dependencies..."
pip install -r requirements.txt
echo "✓ EDM dependencies installed"

# Go back to external directory
cd ..

# Run integration tests
echo "Running EDM integration tests..."
python test_edm_integration.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 EDM integration setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Review the integration test results above"
    echo "2. If conditioning modifications are needed, see external/README.md"
    echo "3. Proceed to Phase 1: Core Infrastructure"
else
    echo ""
    echo "⚠ EDM integration tests failed. Please check the output above."
    echo "See external/README.md for troubleshooting guidance."
    exit 1
fi
