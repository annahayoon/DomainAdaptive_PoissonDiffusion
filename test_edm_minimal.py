#!/usr/bin/env python3
"""
Minimal test to isolate EDM import issue in training context.
"""

import sys
from pathlib import Path

# Add project root to path (exactly like train_photography_model.py)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🔍 Minimal EDM Test")
print("=" * 40)
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

# Test EDM import
try:
    from models.edm_wrapper import EDM_AVAILABLE, create_edm_wrapper

    print(f"✅ EDM import successful")
    print(f"EDM_AVAILABLE: {EDM_AVAILABLE}")

    if EDM_AVAILABLE:
        print("🤖 Testing model creation...")
        model = create_edm_wrapper(
            img_channels=1, img_resolution=64, model_channels=32, label_dim=6
        )
        print(f"✅ Model created: {type(model)}")

        # Test forward pass
        import torch

        x = torch.randn(1, 1, 64, 64)
        sigma = torch.tensor([1.0])
        condition = torch.randn(1, 6)

        with torch.no_grad():
            output = model(x, sigma, condition=condition)

        print(f"✅ Forward pass successful: {output.shape}")
        print("🎉 All tests passed!")

    else:
        print("❌ EDM not available")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
