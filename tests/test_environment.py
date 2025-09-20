"""
Test that the development environment is properly configured.
"""

import numpy as np
import pytest
import torch


def test_pytorch_installation():
    """Test that PyTorch is properly installed."""
    assert torch.__version__ is not None
    # Test basic tensor operations
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = x + y
    assert z.shape == (2, 3)


def test_cuda_availability():
    """Test CUDA availability (if GPU present)."""
    if torch.cuda.is_available():
        # Test basic CUDA operations
        x = torch.randn(2, 3).cuda()
        y = torch.randn(2, 3).cuda()
        z = x + y
        assert z.is_cuda
        assert z.shape == (2, 3)
    else:
        pytest.skip("CUDA not available")


def test_numpy_integration():
    """Test NumPy integration with PyTorch."""
    # NumPy array to tensor
    np_array = np.random.randn(2, 3)
    tensor = torch.from_numpy(np_array)
    assert tensor.shape == (2, 3)

    # Tensor to NumPy array
    back_to_numpy = tensor.numpy()
    np.testing.assert_array_equal(np_array, back_to_numpy)


def test_domain_libraries():
    """Test that domain-specific libraries can be imported."""
    # Photography
    import rawpy

    assert rawpy.__version__ is not None

    # Astronomy
    import astropy

    assert astropy.__version__ is not None

    # Microscopy
    import tifffile

    assert tifffile.__version__ is not None

    # Image processing
    import skimage
    from PIL import Image

    assert skimage.__version__ is not None


def test_edm_integration():
    """Test that EDM can be imported and used."""
    import os
    import sys

    # Add EDM to path
    edm_path = os.path.join(os.path.dirname(__file__), "..", "external", "edm")
    if os.path.exists(edm_path):
        sys.path.insert(0, edm_path)

        try:
            from training.networks import EDMPrecond

            # Test model creation
            model = EDMPrecond(
                img_resolution=64,  # Small for testing
                img_channels=1,
                model_channels=32,  # Small for testing
            )

            # Test forward pass
            x = torch.randn(1, 1, 64, 64)
            sigma = torch.tensor([1.0])

            with torch.no_grad():
                output = model(x, sigma)

            assert output.shape == x.shape

        except ImportError:
            pytest.skip("EDM not available - run external/setup_edm.sh")
    else:
        pytest.skip("EDM directory not found")


def test_project_structure():
    """Test that project structure is correct."""
    import core
    import data
    import models

    # Check version attributes
    assert hasattr(core, "__version__")
    assert hasattr(models, "__version__")
    assert hasattr(data, "__version__")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
