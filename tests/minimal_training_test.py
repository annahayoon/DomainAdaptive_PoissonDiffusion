#!/usr/bin/env python3
"""
Minimal training test to isolate initialization issues.
"""
import sys
import time
from pathlib import Path

import torch

print("🚀 Starting minimal training test...")
print(f"Time: {time.strftime('%H:%M:%S')}")

# Step 1: Basic setup
print("1️⃣ Basic setup...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")

# Step 2: Data check
print("2️⃣ Data check...")
data_path = Path("dataset/processed/png_tiles/photography")
clean_files = list((data_path / "clean").glob("*.png"))[:5]  # Just first 5 files
print(f"   Found {len(clean_files)} sample files")

# Step 3: Simple dataset creation
print("3️⃣ Simple dataset creation...")
try:
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset

    class SimplePhotographyDataset(Dataset):
        def __init__(self, file_list, transform=None):
            self.files = file_list
            self.transform = transform or transforms.ToTensor()

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            from PIL import Image

            img = Image.open(self.files[idx]).convert("RGB")
            return self.transform(img)

    # Create tiny dataset with just 2 files for testing
    tiny_dataset = SimplePhotographyDataset(clean_files[:2])
    print(f"   Created dataset with {len(tiny_dataset)} samples")

    # Test data loading
    sample = tiny_dataset[0]
    print(f"   Sample shape: {sample.shape}")

except Exception as e:
    print(f"   ❌ Dataset creation failed: {e}")
    sys.exit(1)

# Step 4: Simple model creation
print("4️⃣ Simple model creation...")
try:
    # Create a very simple model for testing
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 3, 3, padding=1),
    ).to(device)
    print(
        f"   Created simple model: {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Test forward pass
    with torch.no_grad():
        output = model(sample.unsqueeze(0).to(device))
    print(f"   Forward pass successful: {output.shape}")

except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    sys.exit(1)

# Step 5: Training loop test
print("5️⃣ Training loop test...")
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # Just 2 training steps
    for step in range(2):
        optimizer.zero_grad()
        output = model(sample.unsqueeze(0).to(device))
        loss = criterion(output, sample.unsqueeze(0).to(device))
        loss.backward()
        optimizer.step()
        print(f"   Step {step+1}: loss = {loss.item():.6f}")

    print("   ✅ Training test successful!")

except Exception as e:
    print(f"   ❌ Training test failed: {e}")
    sys.exit(1)

print(f"\n✅ All tests passed! Time: {time.strftime('%H:%M:%S')}")
print("🚀 Basic training functionality works correctly.")
