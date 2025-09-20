#!/usr/bin/env python3
"""
Test script for EDM integration.
Run this after cloning and setting up the EDM repository.
"""

import torch
import sys
import os
from pathlib import Path

def test_edm_import():
    """Test that EDM can be imported successfully."""
    print("Testing EDM import...")
    
    # Add EDM to path
    edm_path = Path(__file__).parent / "edm"
    if not edm_path.exists():
        print("✗ EDM directory not found. Please clone EDM repository first:")
        print("  cd external/")
        print("  git clone https://github.com/NVlabs/edm.git")
        return False
    
    sys.path.insert(0, str(edm_path))
    
    try:
        from training.networks import EDMPrecond
        print("✓ EDM import successful")
        return True
    except ImportError as e:
        print(f"✗ EDM import failed: {e}")
        print("Make sure EDM dependencies are installed:")
        print("  cd external/edm/")
        print("  pip install -r requirements.txt")
        return False

def test_edm_basic_functionality():
    """Test basic EDM model creation and forward pass."""
    print("Testing EDM basic functionality...")
    
    try:
        from training.networks import EDMPrecond
        
        # Create model with standard parameters
        model = EDMPrecond(
            img_resolution=128,
            img_channels=1,
            model_channels=128,
            channel_mult=[1, 2, 2, 2],
            augment_dim=0,
            label_dim=0  # We'll modify this for conditioning later
        )
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 1, 128, 128)
        sigma = torch.tensor([1.0, 0.5])
        
        with torch.no_grad():
            output = model(x, sigma)
        
        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        print("✓ EDM basic functionality test passed")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        return True
        
    except Exception as e:
        print(f"✗ EDM basic functionality test failed: {e}")
        return False

def test_conditioning_requirements():
    """Test what modifications are needed for conditioning."""
    print("Testing conditioning requirements...")
    
    try:
        from training.networks import EDMPrecond
        import inspect
        
        # Check EDMPrecond constructor signature
        sig = inspect.signature(EDMPrecond.__init__)
        params = list(sig.parameters.keys())
        
        print("EDMPrecond constructor parameters:")
        for param in params[1:]:  # Skip 'self'
            print(f"  - {param}")
        
        # Check if label_dim can be used for conditioning
        if 'label_dim' in params:
            print("✓ label_dim parameter available for conditioning")
            
            # Test with label conditioning
            model = EDMPrecond(
                img_resolution=128,
                img_channels=1,
                model_channels=64,  # Smaller for testing
                label_dim=6  # Our conditioning dimension
            )
            
            x = torch.randn(1, 1, 128, 128)
            sigma = torch.tensor([1.0])
            labels = torch.randn(1, 6)  # 6D conditioning vector
            
            with torch.no_grad():
                output = model(x, sigma, class_labels=labels)
            
            print("✓ Conditioning via label_dim works")
            return True
        else:
            print("⚠ label_dim not available, will need custom modifications")
            return True
            
    except Exception as e:
        print(f"✗ Conditioning requirements test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("EDM Integration Test Suite")
    print("=" * 60)
    
    tests = [
        test_edm_import,
        test_edm_basic_functionality,
        test_conditioning_requirements
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! EDM integration is ready.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)