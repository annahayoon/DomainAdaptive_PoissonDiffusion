"""
Complete Systematic Tiling Implementation
Addresses all issues with current tiling approach:
1. Consistent overlap/stride calculations
2. Complete image coverage (no missing pixels)
3. Systematic sampling (not random)
4. Proper edge handling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json

@dataclass
class SystematicTilingConfig:
    """Configuration for systematic tiling with complete coverage"""
    tile_size: int
    overlap_ratio: float  # Fraction of tile_size (0.0 to 0.5)
    coverage_mode: str    # 'complete', 'valid_only', 'sliding_window'
    edge_handling: str    # 'pad_reflect', 'pad_constant', 'crop', 'partial'
    min_valid_ratio: float = 0.5  # Minimum fraction of valid pixels for edge tiles
    
    def __post_init__(self):
        # Calculate derived values
        self.overlap_pixels = int(self.tile_size * self.overlap_ratio)
        self.stride = self.tile_size - self.overlap_pixels
        
        # Validation
        if self.stride <= 0:
            raise ValueError(f"Stride must be positive. Got stride={self.stride} "
                           f"(tile_size={self.tile_size}, overlap_ratio={self.overlap_ratio})")
        
        if self.overlap_ratio >= 0.5:
            raise ValueError(f"Overlap ratio must be < 0.5. Got {self.overlap_ratio}")

@dataclass
class TileInfo:
    """Information about an extracted tile"""
    tile_data: np.ndarray
    tile_id: int
    grid_position: Tuple[int, int]  # (row, col) in tile grid
    image_position: Tuple[int, int]  # (y, x) in original image coordinates
    tile_size: int
    valid_region: Tuple[int, int, int, int]  # (y_start, y_end, x_start, x_end) of valid data
    valid_ratio: float  # Fraction of tile that contains valid image data
    is_edge_tile: bool
    padding_applied: Optional[Dict[str, int]] = None  # If padding was used
    reconstruction_weight: float = 1.0  # Weight for reconstruction

class SystematicTiler:
    """Systematic tiling with complete coverage and consistent overlap"""
    
    def __init__(self, config: SystematicTilingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate tiling configuration"""
        if self.config.tile_size <= 0:
            raise ValueError(f"Tile size must be positive, got {self.config.tile_size}")
        
        if not 0 <= self.config.overlap_ratio < 0.5:
            raise ValueError(f"Overlap ratio must be in [0, 0.5), got {self.config.overlap_ratio}")
        
        if self.config.coverage_mode not in ['complete', 'valid_only', 'sliding_window']:
            raise ValueError(f"Invalid coverage mode: {self.config.coverage_mode}")
    
    def extract_tiles(self, image: np.ndarray) -> List[TileInfo]:
        """
        Extract tiles with systematic coverage
        
        Args:
            image: Input image (H, W) or (C, H, W)
            
        Returns:
            List of TileInfo objects with complete coverage
        """
        
        # Normalize image shape
        if len(image.shape) == 2:
            H, W = image.shape
            image = image[np.newaxis, :, :]  # Add channel dimension
        elif len(image.shape) == 3:
            C, H, W = image.shape
        else:
            raise ValueError(f"Image must be 2D or 3D, got shape {image.shape}")
        
        self.logger.info(f"Extracting tiles from {H}×{W} image with {self.config.tile_size}×{self.config.tile_size} tiles")
        self.logger.info(f"Configuration: overlap={self.config.overlap_ratio:.2f}, stride={self.config.stride}, mode={self.config.coverage_mode}")
        
        if self.config.coverage_mode == 'complete':
            tiles = self._extract_complete_coverage(image)
        elif self.config.coverage_mode == 'valid_only':
            tiles = self._extract_valid_only(image)
        elif self.config.coverage_mode == 'sliding_window':
            tiles = self._extract_sliding_window(image)
        
        self.logger.info(f"Extracted {len(tiles)} tiles with {self.config.coverage_mode} coverage")
        
        # Compute coverage statistics
        coverage_stats = self._compute_coverage_stats(tiles, (H, W))
        self.logger.info(f"Coverage: {coverage_stats['coverage_ratio']:.1%}, "
                        f"Avg overlap: {coverage_stats['avg_coverage']:.1f}×")
        
        return tiles
    
    def _extract_complete_coverage(self, image: np.ndarray) -> List[TileInfo]:
        """Extract tiles with complete image coverage using padding"""
        
        C, H, W = image.shape
        tiles = []
        
        # Calculate grid dimensions for complete coverage
        tiles_h = int(np.ceil((H - self.config.overlap_pixels) / self.config.stride))
        tiles_w = int(np.ceil((W - self.config.overlap_pixels) / self.config.stride))
        
        # Calculate required padded image size
        required_h = (tiles_h - 1) * self.config.stride + self.config.tile_size
        required_w = (tiles_w - 1) * self.config.stride + self.config.tile_size
        
        # Calculate padding needed
        pad_h = max(0, required_h - H)
        pad_w = max(0, required_w - W)
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            image = self._apply_padding(image, pad_h, pad_w)
            self.logger.info(f"Applied padding: {pad_h}×{pad_w} pixels")
        
        # Extract tiles from (possibly padded) image
        tile_id = 0
        
        for i in range(tiles_h):
            for j in range(tiles_w):
                # Calculate tile position
                y = i * self.config.stride
                x = j * self.config.stride
                
                # Extract tile
                tile_data = image[:, y:y+self.config.tile_size, x:x+self.config.tile_size]
                
                # Calculate original image coordinates (before padding)
                orig_y = y - (pad_h // 2) if pad_h > 0 else y
                orig_x = x - (pad_w // 2) if pad_w > 0 else x
                
                # Calculate valid region within tile
                valid_y_start = max(0, -orig_y) if orig_y < 0 else 0
                valid_y_end = min(self.config.tile_size, H - orig_y) if orig_y >= 0 else min(self.config.tile_size, H + abs(orig_y))
                valid_x_start = max(0, -orig_x) if orig_x < 0 else 0
                valid_x_end = min(self.config.tile_size, W - orig_x) if orig_x >= 0 else min(self.config.tile_size, W + abs(orig_x))
                
                # Ensure valid bounds
                valid_y_start = max(0, min(valid_y_start, self.config.tile_size))
                valid_y_end = max(valid_y_start, min(valid_y_end, self.config.tile_size))
                valid_x_start = max(0, min(valid_x_start, self.config.tile_size))
                valid_x_end = max(valid_x_start, min(valid_x_end, self.config.tile_size))
                
                # Calculate valid ratio
                valid_pixels = (valid_y_end - valid_y_start) * (valid_x_end - valid_x_start)
                total_pixels = self.config.tile_size * self.config.tile_size
                valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0
                
                # Create tile info
                tile_info = TileInfo(
                    tile_data=tile_data,
                    tile_id=tile_id,
                    grid_position=(i, j),
                    image_position=(orig_y, orig_x),
                    tile_size=self.config.tile_size,
                    valid_region=(valid_y_start, valid_y_end, valid_x_start, valid_x_end),
                    valid_ratio=valid_ratio,
                    is_edge_tile=valid_ratio < 1.0,
                    padding_applied={'pad_h': pad_h, 'pad_w': pad_w} if (pad_h > 0 or pad_w > 0) else None,
                    reconstruction_weight=valid_ratio  # Weight by valid content
                )
                
                tiles.append(tile_info)
                tile_id += 1
        
        return tiles
    
    def _extract_valid_only(self, image: np.ndarray) -> List[TileInfo]:
        """Extract only tiles that fit completely within the image"""
        
        C, H, W = image.shape
        tiles = []
        tile_id = 0
        
        # Calculate number of tiles that fit completely
        tiles_h = (H - self.config.tile_size) // self.config.stride + 1
        tiles_w = (W - self.config.tile_size) // self.config.stride + 1
        
        for i in range(tiles_h):
            for j in range(tiles_w):
                y = i * self.config.stride
                x = j * self.config.stride
                
                # Extract tile
                tile_data = image[:, y:y+self.config.tile_size, x:x+self.config.tile_size]
                
                tile_info = TileInfo(
                    tile_data=tile_data,
                    tile_id=tile_id,
                    grid_position=(i, j),
                    image_position=(y, x),
                    tile_size=self.config.tile_size,
                    valid_region=(0, self.config.tile_size, 0, self.config.tile_size),
                    valid_ratio=1.0,
                    is_edge_tile=False,
                    reconstruction_weight=1.0
                )
                
                tiles.append(tile_info)
                tile_id += 1
        
        return tiles
    
    def _extract_sliding_window(self, image: np.ndarray) -> List[TileInfo]:
        """Extract tiles with stride=1 (dense sampling)"""
        
        C, H, W = image.shape
        tiles = []
        tile_id = 0
        
        # Dense extraction with stride=1
        for y in range(H - self.config.tile_size + 1):
            for x in range(W - self.config.tile_size + 1):
                tile_data = image[:, y:y+self.config.tile_size, x:x+self.config.tile_size]
                
                tile_info = TileInfo(
                    tile_data=tile_data,
                    tile_id=tile_id,
                    grid_position=(y, x),  # Dense grid
                    image_position=(y, x),
                    tile_size=self.config.tile_size,
                    valid_region=(0, self.config.tile_size, 0, self.config.tile_size),
                    valid_ratio=1.0,
                    is_edge_tile=False,
                    reconstruction_weight=1.0
                )
                
                tiles.append(tile_info)
                tile_id += 1
        
        return tiles
    
    def _apply_padding(self, image: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
        """Apply padding to image based on edge handling mode"""
        
        # Symmetric padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        if self.config.edge_handling == 'pad_reflect':
            # Reflect padding (mirror)
            padded = np.pad(image, 
                          ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                          mode='reflect')
        elif self.config.edge_handling == 'pad_constant':
            # Constant padding with edge values
            padded = np.pad(image,
                          ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                          mode='edge')
        else:
            # Default to reflect
            padded = np.pad(image,
                          ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                          mode='reflect')
        
        return padded
    
    def reconstruct_image(self, tiles: List[TileInfo], 
                         original_shape: Tuple[int, int],
                         method: str = 'weighted_average') -> np.ndarray:
        """
        Reconstruct image from tiles with proper overlap handling
        
        Args:
            tiles: List of TileInfo objects
            original_shape: (H, W) of original image
            method: 'weighted_average', 'simple_average', 'first_wins'
            
        Returns:
            Reconstructed image
        """
        
        H, W = original_shape
        C = tiles[0].tile_data.shape[0] if tiles else 1
        
        # Initialize reconstruction arrays
        reconstructed = np.zeros((C, H, W), dtype=np.float32)
        weight_map = np.zeros((H, W), dtype=np.float32)
        
        for tile_info in tiles:
            # Get tile position and data
            y, x = tile_info.image_position
            tile_data = tile_info.tile_data
            
            # Calculate bounds in original image
            y_start = max(0, y)
            y_end = min(H, y + self.config.tile_size)
            x_start = max(0, x)
            x_end = min(W, x + self.config.tile_size)
            
            # Calculate corresponding bounds in tile
            tile_y_start = max(0, -y)
            tile_y_end = tile_y_start + (y_end - y_start)
            tile_x_start = max(0, -x)
            tile_x_end = tile_x_start + (x_end - x_start)
            
            # Skip if no valid overlap
            if y_end <= y_start or x_end <= x_start:
                continue
            
            # Extract valid tile region
            valid_tile = tile_data[:, tile_y_start:tile_y_end, tile_x_start:tile_x_end]
            
            # Determine weight
            if method == 'weighted_average':
                weight = tile_info.reconstruction_weight
            elif method == 'simple_average':
                weight = 1.0
            elif method == 'first_wins':
                # Only add if pixel not already covered
                weight = (weight_map[y_start:y_end, x_start:x_end] == 0).astype(float)
            else:
                weight = 1.0
            
            # Add to reconstruction
            reconstructed[:, y_start:y_end, x_start:x_end] += valid_tile * weight
            weight_map[y_start:y_end, x_start:x_end] += weight
        
        # Normalize by weights
        weight_map[weight_map == 0] = 1.0  # Avoid division by zero
        reconstructed = reconstructed / weight_map[np.newaxis, :, :]
        
        return reconstructed
    
    def _compute_coverage_stats(self, tiles: List[TileInfo], 
                              image_shape: Tuple[int, int]) -> Dict[str, float]:
        """Compute coverage statistics"""
        
        H, W = image_shape
        coverage_map = np.zeros((H, W), dtype=int)
        
        for tile_info in tiles:
            y, x = tile_info.image_position
            
            # Calculate bounds
            y_start = max(0, y)
            y_end = min(H, y + self.config.tile_size)
            x_start = max(0, x)
            x_end = min(W, x + self.config.tile_size)
            
            if y_end > y_start and x_end > x_start:
                coverage_map[y_start:y_end, x_start:x_end] += 1
        
        total_pixels = H * W
        covered_pixels = np.sum(coverage_map > 0)
        
        return {
            'coverage_ratio': covered_pixels / total_pixels,
            'avg_coverage': np.mean(coverage_map[coverage_map > 0]) if covered_pixels > 0 else 0,
            'max_coverage': np.max(coverage_map),
            'uncovered_pixels': total_pixels - covered_pixels
        }
    
    def save_tiling_info(self, tiles: List[TileInfo], save_path: str):
        """Save tiling information to JSON"""
        
        tiling_info = {
            'config': {
                'tile_size': self.config.tile_size,
                'overlap_ratio': self.config.overlap_ratio,
                'stride': self.config.stride,
                'coverage_mode': self.config.coverage_mode,
                'edge_handling': self.config.edge_handling
            },
            'tiles': [
                {
                    'tile_id': tile.tile_id,
                    'grid_position': tile.grid_position,
                    'image_position': tile.image_position,
                    'valid_region': tile.valid_region,
                    'valid_ratio': tile.valid_ratio,
                    'is_edge_tile': tile.is_edge_tile,
                    'padding_applied': tile.padding_applied
                }
                for tile in tiles
            ],
            'summary': {
                'total_tiles': len(tiles),
                'edge_tiles': sum(1 for t in tiles if t.is_edge_tile),
                'avg_valid_ratio': np.mean([t.valid_ratio for t in tiles])
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(tiling_info, f, indent=2)

# Domain-specific configurations - UPDATED FOR UNIFIED 256x256 RESOLUTION
def get_domain_tiling_config(domain_name: str, tile_size: int = None) -> SystematicTilingConfig:
    """Get unified tiling configuration for all domains (256x256)"""
    
    # All domains now use unified 256x256 tiles with domain-appropriate overlap
    unified_tile_size = tile_size or 256  # Default to unified size
    
    configs = {
        "photography": SystematicTilingConfig(
            tile_size=unified_tile_size,
            overlap_ratio=0.1,  # 10% overlap for context preservation
            coverage_mode='complete',
            edge_handling='pad_reflect',
            min_valid_ratio=0.7
        ),
        "microscopy": SystematicTilingConfig(
            tile_size=unified_tile_size,
            overlap_ratio=0.0,  # No overlap needed (upsampled from 128)
            coverage_mode='complete',
            edge_handling='pad_reflect',
            min_valid_ratio=0.8
        ),
        "astronomy": SystematicTilingConfig(
            tile_size=unified_tile_size,
            overlap_ratio=0.05,  # Minimal overlap (downsampled from 1024)
            coverage_mode='complete',
            edge_handling='pad_reflect',
            min_valid_ratio=0.6
        )
    }
    
    return configs.get(domain_name, configs["microscopy"])

# Example usage and testing
def main():
    """Test systematic tiling implementation"""
    
    print("🔧 SYSTEMATIC TILING IMPLEMENTATION")
    print("="*50)
    
    # Test cases - ALL NOW USE UNIFIED 256x256 TILES
    test_cases = [
        ("microscopy", (512, 512)),
        ("photography", (2048, 3072)),
        ("astronomy", (4096, 4096))
    ]
    
    unified_tile_size = 256
    
    for domain, image_shape in test_cases:
        print(f"\n📊 Testing {domain}: {image_shape} → {unified_tile_size}×{unified_tile_size} (UNIFIED)")
        
        # Get unified domain configuration
        config = get_domain_tiling_config(domain, unified_tile_size)
        tiler = SystematicTiler(config)
        
        # Create dummy image
        dummy_image = np.random.rand(*image_shape).astype(np.float32)
        
        # Extract tiles
        tiles = tiler.extract_tiles(dummy_image)
        
        # Compute statistics
        coverage_stats = tiler._compute_coverage_stats(tiles, image_shape)
        
        print(f"   Tiles extracted: {len(tiles)}")
        print(f"   Coverage: {coverage_stats['coverage_ratio']:.1%}")
        print(f"   Avg overlap: {coverage_stats['avg_coverage']:.1f}×")
        print(f"   Edge tiles: {sum(1 for t in tiles if t.is_edge_tile)}")
        
        # Test reconstruction
        reconstructed = tiler.reconstruct_image(tiles, image_shape)
        reconstruction_error = np.mean(np.abs(reconstructed[0] - dummy_image))
        print(f"   Reconstruction error: {reconstruction_error:.6f}")

if __name__ == "__main__":
    main()
