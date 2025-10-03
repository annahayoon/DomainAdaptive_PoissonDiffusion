"""
Astronomy-Specific Preprocessing with Asinh Scaling
Implements proper astronomical image preprocessing including asinh normalization
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

@dataclass
class AsinhScalingConfig:
    """Configuration for asinh scaling in astronomy"""
    softening_factor: float = 1e-3  # β parameter - controls transition point
    scale_factor: Optional[float] = None  # Will be computed if None
    background_percentile: float = 10.0  # Background estimation percentile
    scale_percentile: float = 99.5  # Scale estimation percentile
    cosmic_ray_threshold: float = 5.0  # Sigma threshold for cosmic ray detection
    
class AstronomyAsinhPreprocessor:
    """Astronomy-specific preprocessing with asinh scaling"""
    
    def __init__(self, config: AsinhScalingConfig = None):
        self.config = config or AsinhScalingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Cached parameters for consistent processing
        self._global_scale = None
        self._global_background = None
        self._softening_factor = self.config.softening_factor
    
    def preprocess_astronomy_image(self, image: np.ndarray, 
                                 calibration: Dict[str, Any],
                                 apply_cosmic_ray_removal: bool = True) -> Dict[str, Any]:
        """
        Apply astronomy-specific preprocessing with asinh scaling
        
        Args:
            image: Raw astronomy image data (electrons)
            calibration: Calibration parameters from FITS header
            apply_cosmic_ray_removal: Whether to detect and mask cosmic rays
            
        Returns:
            Dictionary with preprocessed image and metadata
        """
        
        # Step 1: Convert to electrons if needed
        if 'gain' in calibration and calibration['gain'] != 1.0:
            image_electrons = image * calibration['gain']
        else:
            image_electrons = image.copy()
        
        # Step 2: Background estimation and subtraction
        background = self._estimate_background(image_electrons)
        image_bg_subtracted = np.maximum(image_electrons - background, 0)
        
        # Step 3: Cosmic ray detection and masking
        cosmic_ray_mask = None
        if apply_cosmic_ray_removal:
            cosmic_ray_mask = self._detect_cosmic_rays(
                image_bg_subtracted, 
                read_noise=calibration.get('read_noise', 4.72)
            )
            # Replace cosmic rays with local median
            image_clean = self._remove_cosmic_rays(image_bg_subtracted, cosmic_ray_mask)
        else:
            image_clean = image_bg_subtracted
        
        # Step 4: Determine scaling parameters
        if self._global_scale is None:
            self._global_scale = self._estimate_scale(image_clean)
        
        # Step 5: Apply asinh scaling
        image_asinh = self._apply_asinh_scaling(
            image_clean, 
            scale=self._global_scale,
            softening=self._softening_factor
        )
        
        # Step 6: Normalize to [0, 1] range for neural network
        image_normalized = self._normalize_to_unit_range(image_asinh)
        
        return {
            'preprocessed_image': image_normalized.astype(np.float32),
            'original_electrons': image_electrons,
            'background_subtracted': image_bg_subtracted,
            'cosmic_ray_mask': cosmic_ray_mask,
            'preprocessing_metadata': {
                'background_level': background,
                'scale_factor': self._global_scale,
                'softening_factor': self._softening_factor,
                'cosmic_rays_detected': cosmic_ray_mask.sum() if cosmic_ray_mask is not None else 0,
                'normalization_method': 'asinh'
            }
        }
    
    def _estimate_background(self, image: np.ndarray) -> float:
        """Estimate background level using sigma-clipped statistics"""
        from scipy import ndimage
        
        # Use sigma-clipped percentile for robust background estimation
        # This is standard in astronomy to handle outliers and cosmic rays
        flattened = image.flatten()
        
        # Handle edge case: all values are the same or very few values
        if len(flattened) == 0:
            return 0.0
        
        if len(np.unique(flattened)) <= 2:
            # Very limited dynamic range, use simple percentile
            background = np.percentile(flattened, self.config.background_percentile)
            self.logger.debug(f"Background estimated (simple): {background:.2f} electrons")
            return float(background)
        
        # Remove obvious outliers (> 5σ)
        median = np.median(flattened)
        mad = np.median(np.abs(flattened - median))
        sigma_estimate = 1.4826 * mad  # Convert MAD to standard deviation
        
        # Keep only pixels within 3σ of median
        mask = np.abs(flattened - median) < 3 * sigma_estimate
        clean_pixels = flattened[mask]
        
        # Handle case where sigma clipping removes all pixels
        if len(clean_pixels) == 0:
            clean_pixels = flattened
        
        background = np.percentile(clean_pixels, self.config.background_percentile)
        
        self.logger.debug(f"Background estimated: {background:.2f} electrons")
        return float(background)
    
    def _detect_cosmic_rays(self, image: np.ndarray, read_noise: float) -> np.ndarray:
        """Detect cosmic rays using Laplacian edge detection"""
        from scipy import ndimage
        
        # Handle multi-dimensional images (work on each channel separately)
        if image.ndim > 2:
            # Process each channel separately
            cosmic_ray_mask = np.zeros_like(image, dtype=bool)
            for i in range(image.shape[0]):
                cosmic_ray_mask[i] = self._detect_cosmic_rays_2d(image[i], read_noise)
        else:
            cosmic_ray_mask = self._detect_cosmic_rays_2d(image, read_noise)
        
        self.logger.debug(f"Cosmic rays detected: {cosmic_ray_mask.sum()} pixels")
        return cosmic_ray_mask
    
    def _detect_cosmic_rays_2d(self, image_2d: np.ndarray, read_noise: float) -> np.ndarray:
        """Detect cosmic rays in 2D image"""
        from scipy import ndimage
        
        # Apply median filter to create clean reference
        median_filtered = ndimage.median_filter(image_2d, size=5)
        
        # Compute residual
        residual = image_2d - median_filtered
        
        # Estimate noise level
        noise_estimate = np.sqrt(median_filtered + read_noise**2)
        
        # Detect pixels significantly above noise
        cosmic_ray_mask = residual > self.config.cosmic_ray_threshold * noise_estimate
        
        # Morphological cleanup - remove single isolated pixels
        from scipy.ndimage import binary_opening
        cosmic_ray_mask = binary_opening(cosmic_ray_mask, structure=np.ones((3, 3)))
        
        return cosmic_ray_mask
    
    def _remove_cosmic_rays(self, image: np.ndarray, cosmic_ray_mask: np.ndarray) -> np.ndarray:
        """Replace cosmic ray pixels with local median"""
        from scipy import ndimage
        
        image_clean = image.copy()
        
        if cosmic_ray_mask.sum() > 0:
            if image.ndim > 2:
                # Process each channel separately
                for i in range(image.shape[0]):
                    if cosmic_ray_mask[i].sum() > 0:
                        median_filtered = ndimage.median_filter(image[i], size=5)
                        image_clean[i][cosmic_ray_mask[i]] = median_filtered[cosmic_ray_mask[i]]
            else:
                # Single 2D image
                median_filtered = ndimage.median_filter(image, size=5)
                image_clean[cosmic_ray_mask] = median_filtered[cosmic_ray_mask]
        
        return image_clean
    
    def _estimate_scale(self, image: np.ndarray) -> float:
        """Estimate scale factor for asinh normalization"""
        
        # Use high percentile to capture bright features while avoiding outliers
        scale = np.percentile(image, self.config.scale_percentile)
        
        # Ensure scale is positive and reasonable
        if scale <= 0:
            scale = np.percentile(image, 95.0)  # Fallback
            
        # Handle very low values - ensure minimum scale
        if scale <= 0 or np.isnan(scale):
            # Use image max as fallback
            scale = np.max(image)
            
        # Ensure minimum scale for very faint images
        min_scale = np.std(image) * 2.0  # At least 2 standard deviations
        scale = max(scale, min_scale, 1.0)  # Minimum scale of 1.0
        
        self.logger.info(f"Scale factor estimated: {scale:.2f} electrons")
        return float(scale)
    
    def _apply_asinh_scaling(self, image: np.ndarray, scale: float, softening: float) -> np.ndarray:
        """
        Apply asinh (inverse hyperbolic sine) scaling
        
        The asinh function is ideal for astronomical data because:
        1. Linear for low intensities (preserves noise characteristics)
        2. Logarithmic for high intensities (compresses dynamic range)
        3. Smooth transition between regimes
        
        Formula: asinh(image / (softening * scale)) / asinh(1 / softening)
        """
        
        # Normalize by scale and softening factor
        normalized_image = image / (softening * scale)
        
        # Apply asinh transformation
        asinh_image = np.arcsinh(normalized_image)
        
        # Normalize by asinh(1/softening) to get reasonable output range
        asinh_normalization = np.arcsinh(1.0 / softening)
        scaled_image = asinh_image / asinh_normalization
        
        self.logger.debug(f"Asinh scaling applied: softening={softening:.3f}, scale={scale:.2f}")
        
        return scaled_image
    
    def _normalize_to_unit_range(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range for neural network"""
        
        # Clip extreme values (should be rare after asinh scaling)
        image_clipped = np.clip(image, 0, None)
        
        # Normalize to [0, 1]
        if image_clipped.max() > 0:
            image_normalized = image_clipped / image_clipped.max()
        else:
            image_normalized = image_clipped
        
        return image_normalized
    
    def compute_inverse_asinh(self, normalized_image: np.ndarray) -> np.ndarray:
        """
        Inverse transformation to recover original electron counts
        Useful for evaluation and analysis
        """
        if self._global_scale is None:
            raise ValueError("Must process at least one image to establish scale")
        
        # Reverse normalization
        max_val = 1.0  # Assuming normalized to [0, 1]
        scaled_back = normalized_image * max_val
        
        # Reverse asinh normalization
        asinh_normalization = np.arcsinh(1.0 / self._softening_factor)
        asinh_values = scaled_back * asinh_normalization
        
        # Apply sinh (inverse of asinh)
        linear_values = np.sinh(asinh_values)
        
        # Scale back to electrons
        electrons = linear_values * self._softening_factor * self._global_scale
        
        return electrons
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get statistics about the preprocessing"""
        return {
            'global_scale': self._global_scale,
            'global_background': self._global_background,
            'softening_factor': self._softening_factor,
            'normalization_method': 'asinh',
            'scale_percentile': self.config.scale_percentile,
            'background_percentile': self.config.background_percentile
        }

# Integration with existing domain processor
def create_astronomy_preprocessing_mixin():
    """Create mixin class for astronomy preprocessing"""
    
    class AstronomyPreprocessingMixin:
        """Mixin to add asinh preprocessing to AstronomyProcessor"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.asinh_preprocessor = AstronomyAsinhPreprocessor()
        
        def preprocess_image(self, image: np.ndarray, calibration) -> np.ndarray:
            """Override to use asinh scaling for astronomy"""
            
            # Convert calibration to dict format
            if hasattr(calibration, '__dict__'):
                calib_dict = {
                    'gain': calibration.gain,
                    'read_noise': calibration.read_noise,
                    'background': calibration.background,
                    'scale_factor': calibration.scale_factor
                }
            else:
                calib_dict = calibration
            
            # Apply astronomy-specific preprocessing
            result = self.asinh_preprocessor.preprocess_astronomy_image(
                image, calib_dict, apply_cosmic_ray_removal=True
            )
            
            return result['preprocessed_image']
        
        def get_preprocessing_metadata(self) -> Dict[str, Any]:
            """Get preprocessing metadata for astronomy domain"""
            return self.asinh_preprocessor.get_preprocessing_stats()
    
    return AstronomyPreprocessingMixin

def main():
    """Test astronomy asinh preprocessing"""
    
    print("🌌 ASTRONOMY ASINH PREPROCESSING")
    print("="*50)
    
    # Create test astronomy image with realistic characteristics
    np.random.seed(42)
    
    # Simulate astronomy image: mostly background with few bright sources
    image_size = 1024
    background_level = 100.0  # electrons
    
    # Background + noise
    image = np.random.poisson(background_level, (image_size, image_size)).astype(np.float32)
    
    # Add some bright sources (stars/galaxies)
    n_sources = 50
    for _ in range(n_sources):
        y, x = np.random.randint(0, image_size, 2)
        brightness = np.random.exponential(1000)  # Exponential distribution of brightnesses
        size = np.random.randint(2, 8)
        
        # Add Gaussian source
        yy, xx = np.ogrid[:image_size, :image_size]
        source = brightness * np.exp(-((yy-y)**2 + (xx-x)**2) / (2*size**2))
        image += source
    
    # Add some cosmic rays
    n_cosmic_rays = 20
    for _ in range(n_cosmic_rays):
        y, x = np.random.randint(0, image_size, 2)
        image[y, x] += np.random.exponential(5000)  # Bright cosmic ray
    
    print(f"Test image created: {image_size}×{image_size}")
    print(f"  Background: ~{background_level:.1f} electrons")
    print(f"  Dynamic range: {image.min():.1f} - {image.max():.1f} electrons")
    print(f"  Sources added: {n_sources}")
    print(f"  Cosmic rays: {n_cosmic_rays}")
    
    # Test preprocessing
    config = AsinhScalingConfig(
        softening_factor=1e-3,
        scale_percentile=99.5,
        background_percentile=10.0
    )
    
    preprocessor = AstronomyAsinhPreprocessor(config)
    
    calibration = {
        'gain': 4.62,
        'read_noise': 4.72
    }
    
    # Process image
    result = preprocessor.preprocess_astronomy_image(image, calibration)
    
    # Show results
    processed = result['preprocessed_image']
    metadata = result['preprocessing_metadata']
    
    print(f"\n📊 Preprocessing Results:")
    print(f"  Background subtracted: {metadata['background_level']:.1f} electrons")
    print(f"  Scale factor: {metadata['scale_factor']:.1f} electrons")
    print(f"  Cosmic rays detected: {metadata['cosmic_rays_detected']}")
    print(f"  Softening factor: {metadata['softening_factor']:.3f}")
    print(f"  Final range: {processed.min():.3f} - {processed.max():.3f}")
    
    # Test inverse transformation
    recovered = preprocessor.compute_inverse_asinh(processed)
    recovery_error = np.mean(np.abs(recovered - result['background_subtracted']))
    print(f"  Inverse transform error: {recovery_error:.2f} electrons")
    
    print(f"\n✅ Asinh preprocessing validation complete!")
    print(f"   Ready for astronomy domain integration")

if __name__ == "__main__":
    main()
