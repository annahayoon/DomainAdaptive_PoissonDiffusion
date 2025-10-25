#!/usr/bin/env python3
"""
Compute FID (Fréchet Inception Distance) between two sets of images

This script calculates FID scores to compare batches of generated/restored images
against clean reference images. FID measures the distribution similarity between
two sets of images using InceptionV3 features.

Usage:
    python sample/compute_fid.py \
        --restored_dir results/posterior_sampling_pg/example_00_photography_sony_00145_00_0.1s_tile_0000 \
        --clean_dir dataset/processed/pt_tiles/photography/clean \
        --output_file results/fid_comparison.json \
        --domain photography

Requirements:
    pip install torch torchvision piq
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.linalg import sqrtm

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FIDCalculator:
    """Calculate Fréchet Inception Distance (FID) between two sets of images."""
    
    def __init__(self, device: str = "cuda"):
        """Initialize FID calculator with pre-trained InceptionV3 model."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained InceptionV3 model
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        self.inception_model.eval()
        self.inception_model.to(self.device)
        
        # Remove the final classification layer to get features
        self.inception_model.fc = nn.Identity()
        
        # Image preprocessing for InceptionV3
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),  # InceptionV3 expects 299x299
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"FID calculator initialized on device: {self.device}")
    
    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for InceptionV3 model.
        
        Args:
            images: Input images [B, C, H, W] in range [0, 1] (already converted from [-1,1])
            
        Returns:
            Preprocessed images [B, 3, 299, 299] ready for InceptionV3
        """
        batch_size = images.shape[0]
        
        # Convert to RGB if grayscale
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {images.shape[1]}")
        
        # Resize and normalize for InceptionV3
        processed_images = []
        for i in range(batch_size):
            img = images[i:i+1]  # Keep batch dimension
            img_resized = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
            img_normalized = self.transform(img_resized)
            processed_images.append(img_normalized)
        
        return torch.cat(processed_images, dim=0)
    
    def _extract_features(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract features from images using InceptionV3.
        
        Args:
            images: Input images [B, C, H, W] in range [0, 1] (already converted from [-1,1])
            
        Returns:
            Feature vectors [B, 2048]
        """
        with torch.no_grad():
            # Ensure images are on the same device as the model
            images = images.to(self.device)
            
            # Preprocess images
            processed_images = self._preprocess_images(images)
            
            # Extract features
            features = self.inception_model(processed_images)
            
            return features.cpu().numpy()
    
    def _calculate_fid(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Calculate FID between two sets of features.
        
        Args:
            features1: Feature vectors from first set [N1, 2048]
            features2: Feature vectors from second set [N2, 2048]
            
        Returns:
            FID score (lower is better)
        """
        # Ensure we have enough samples for covariance calculation
        if features1.shape[0] < 2 or features2.shape[0] < 2:
            logger.warning(f"Insufficient samples for FID: {features1.shape[0]}, {features2.shape[0]}")
            return float('nan')
        
        # Calculate mean and covariance
        mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
        
        # Ensure covariance matrices are 2D
        if sigma1.ndim == 0:
            sigma1 = np.array([[sigma1]])
        if sigma2.ndim == 0:
            sigma2 = np.array([[sigma2]])
        
        # Calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        
        # Calculate sqrt of product between cov
        try:
            covmean = sqrtm(sigma1.dot(sigma2))
            
            # Check and correct imaginary numbers from sqrt
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            # Calculate FID
            fid = ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
            
            return float(fid)
        except Exception as e:
            logger.warning(f"FID calculation failed: {e}")
            return float('nan')
    
    def compute_fid(self, images1: torch.Tensor, images2: torch.Tensor) -> float:
        """
        Compute FID between two sets of images.
        
        Args:
            images1: First set of images [B, C, H, W] in range [-1, 1]
            images2: Second set of images [B, C, H, W] in range [-1, 1]
            
        Returns:
            FID score (lower is better)
        """
        # Convert from [-1, 1] to [0, 1] range for InceptionV3 preprocessing
        # InceptionV3 expects images in [0, 1] range before normalization
        images1 = (images1 + 1.0) / 2.0  # [-1,1] -> [0,1]
        images2 = (images2 + 1.0) / 2.0  # [-1,1] -> [0,1]
        
        # Clamp to valid range
        images1 = torch.clamp(images1, 0.0, 1.0)
        images2 = torch.clamp(images2, 0.0, 1.0)
        
        # Extract features
        features1 = self._extract_features(images1)
        features2 = self._extract_features(images2)
        
        # Calculate FID
        fid_score = self._calculate_fid(features1, features2)
        
        return fid_score


def load_images_from_directory(
    directory: Path,
    prefix: str = "",
    suffix: str = ".pt",
    device: torch.device = None
) -> List[torch.Tensor]:
    """
    Load all .pt files from a directory.
    
    Args:
        directory: Directory containing .pt files
        prefix: Optional prefix to filter files
        suffix: File suffix to match (default: .pt)
        device: Device to load tensors to
        
    Returns:
        List of image tensors
    """
    images = []
    
    # Find all .pt files
    pt_files = sorted(directory.glob(f"{prefix}*{suffix}"))
    
    logger.info(f"Loading {len(pt_files)} images from {directory}")
    
    for pt_file in pt_files:
        try:
            tensor = torch.load(pt_file, map_location=device)
            
            # Handle different tensor formats
            if isinstance(tensor, dict):
                if 'restored' in tensor:
                    tensor = tensor['restored']
                elif 'noisy' in tensor:
                    tensor = tensor['noisy']
                elif 'clean' in tensor:
                    tensor = tensor['clean']
                elif 'image' in tensor:
                    tensor = tensor['image']
                else:
                    raise ValueError(f"Unrecognized dict structure in {pt_file}")
            
            # Ensure float32
            tensor = tensor.float()
            
            # Ensure proper shape [C, H, W] or [H, W]
            if tensor.ndim == 2:  # (H, W)
                tensor = tensor.unsqueeze(0)  # (1, H, W)
            elif tensor.ndim == 3 and tensor.shape[-1] in [1, 3]:  # (H, W, C)
                tensor = tensor.permute(2, 0, 1)  # (C, H, W)
            
            images.append(tensor)
            
        except Exception as e:
            logger.warning(f"Failed to load {pt_file}: {e}")
            continue
    
    return images


def main():
    """Main function for FID computation."""
    parser = argparse.ArgumentParser(
        description="Compute FID between two sets of images"
    )
    
    # Input arguments
    parser.add_argument("--restored_dir", type=str, required=True,
                       help="Directory containing restored/generated images (.pt files)")
    parser.add_argument("--clean_dir", type=str, required=True,
                       help="Directory containing clean reference images (.pt files)")
    parser.add_argument("--restored_prefix", type=str, default="restored_",
                       help="Prefix for restored image files (default: restored_)")
    parser.add_argument("--clean_prefix", type=str, default="",
                       help="Prefix for clean image files (default: empty)")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default="fid_results.json",
                       help="Output JSON file for FID results")
    
    # Domain arguments
    parser.add_argument("--domain", type=str, default="photography",
                       choices=["photography", "microscopy", "astronomy"],
                       help="Domain name for data handling")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for computation (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for feature extraction")
    
    args = parser.parse_args()
    
    # Setup
    restored_dir = Path(args.restored_dir)
    clean_dir = Path(args.clean_dir)
    output_file = Path(args.output_file)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 80)
    logger.info("FID COMPUTATION")
    logger.info("=" * 80)
    logger.info(f"Restored dir: {restored_dir}")
    logger.info(f"Clean dir: {clean_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)
    
    # Load images
    logger.info("Loading restored images...")
    restored_images = load_images_from_directory(
        restored_dir,
        prefix=args.restored_prefix,
        device=device
    )
    
    logger.info("Loading clean images...")
    clean_images = load_images_from_directory(
        clean_dir,
        prefix=args.clean_prefix,
        device=device
    )
    
    if len(restored_images) == 0:
        logger.error("No restored images found!")
        return
    
    if len(clean_images) == 0:
        logger.error("No clean images found!")
        return
    
    logger.info(f"Loaded {len(restored_images)} restored images and {len(clean_images)} clean images")
    
    # Initialize FID calculator
    fid_calculator = FIDCalculator(device=device)
    
    # Batch images for efficient processing
    def batch_images(images: List[torch.Tensor], batch_size: int) -> List[torch.Tensor]:
        """Batch images into groups."""
        batches = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i+batch_size], dim=0)
            batches.append(batch)
        return batches
    
    restored_batches = batch_images(restored_images, args.batch_size)
    clean_batches = batch_images(clean_images, args.batch_size)
    
    # Compute FID
    logger.info("Computing FID...")
    
    # Collect all features
    all_restored_features = []
    all_clean_features = []
    
    for batch in restored_batches:
        batch_01 = (batch + 1.0) / 2.0  # Convert [-1,1] to [0,1]
        batch_01 = torch.clamp(batch_01, 0.0, 1.0)
        features = fid_calculator._extract_features(batch_01)
        all_restored_features.append(features)
    
    for batch in clean_batches:
        batch_01 = (batch + 1.0) / 2.0  # Convert [-1,1] to [0,1]
        batch_01 = torch.clamp(batch_01, 0.0, 1.0)
        features = fid_calculator._extract_features(batch_01)
        all_clean_features.append(features)
    
    # Concatenate all features
    restored_features = np.concatenate(all_restored_features, axis=0)
    clean_features = np.concatenate(all_clean_features, axis=0)
    
    # Calculate FID
    fid_score = fid_calculator._calculate_fid(restored_features, clean_features)
    
    # Save results
    results = {
        'fid_score': float(fid_score),
        'num_restored_images': len(restored_images),
        'num_clean_images': len(clean_images),
        'restored_dir': str(restored_dir),
        'clean_dir': str(clean_dir),
        'domain': args.domain,
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("FID COMPUTATION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"FID Score: {fid_score:.4f}")
    logger.info(f"Restored images: {len(restored_images)}")
    logger.info(f"Clean images: {len(clean_images)}")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

