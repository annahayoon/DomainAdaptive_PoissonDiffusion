"""
Domain-Specific Processors for Diffusion Denoising Data Pipeline
Modular processors for Photography, Microscopy, and Astronomy domains
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import json

@dataclass
class ImageMetadata:
    """Standard metadata structure for all domains"""
    image_id: str
    domain_id: int
    domain_name: str
    source_path: str
    is_clean: bool
    pair_id: Optional[str] = None
    noise_level: Optional[int] = None
    processing_notes: List[str] = None
    
    def __post_init__(self):
        if self.processing_notes is None:
            self.processing_notes = []

@dataclass
class CalibrationData:
    """Camera/sensor calibration parameters"""
    gain: float  # electrons per ADU
    read_noise: float  # electrons RMS
    black_level: float  # ADU offset
    white_level: float  # ADU saturation
    background: float  # electrons
    scale_factor: float  # normalization scale
    saturation_threshold: Optional[float] = None
    # New normalization parameters for [0,1] range
    norm_min: Optional[float] = None  # Minimum value for [0,1] normalization
    norm_max: Optional[float] = None  # Maximum value for [0,1] normalization
    norm_percentiles: Tuple[float, float] = (0.1, 99.9)  # Percentiles used for robust normalization
    
    def __post_init__(self):
        if self.saturation_threshold is None:
            self.saturation_threshold = self.white_level * 0.95

class BaseDomainProcessor(ABC):
    """Abstract base class for domain-specific processors"""
    
    def __init__(self, domain_id: int, domain_name: str):
        self.domain_id = domain_id
        self.domain_name = domain_name
        self.logger = logging.getLogger(f"{__name__}.{domain_name}")
        
        # Processing configuration
        self.config = {
            "tile_size": 128,
            "overlap": 0.1,
            "min_valid_pixels": 0.8,
            "augmentation_enabled": True,
            "quality_threshold": 0.7
        }
    
    @abstractmethod
    def discover_files(self, raw_data_path: str) -> List[Dict[str, Any]]:
        """Discover and catalog all files in the domain"""
        pass
    
    @abstractmethod
    def load_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load image and extract metadata from file"""
        pass
    
    @abstractmethod
    def estimate_calibration(self, images: List[np.ndarray]) -> CalibrationData:
        """Estimate calibration parameters from sample images"""
        pass
    
    def compute_normalization_parameters(self, images: List[np.ndarray], 
                                       calibration: CalibrationData) -> CalibrationData:
        """Compute [0,1] normalization parameters from sample images"""
        all_pixels = []
        
        for image in images:
            # Apply same preprocessing steps as in preprocess_image (without final normalization)
            if hasattr(calibration, 'gain') and calibration.gain != 1.0:
                image_electrons = image * calibration.gain
            else:
                image_electrons = image.copy()
            
            if hasattr(calibration, 'background'):
                image_electrons = np.maximum(image_electrons - calibration.background, 0)
            
            all_pixels.extend(image_electrons.flatten())
        
        all_pixels = np.array(all_pixels)
        
        # Compute robust normalization range using percentiles
        norm_min = float(np.percentile(all_pixels, calibration.norm_percentiles[0]))
        norm_max = float(np.percentile(all_pixels, calibration.norm_percentiles[1]))
        
        # Ensure we have a valid range
        if norm_max <= norm_min:
            norm_max = norm_min + 1.0
        
        # Update calibration with normalization parameters
        calibration.norm_min = norm_min
        calibration.norm_max = norm_max
        
        self.logger.info(f"Computed [0,1] normalization parameters:")
        self.logger.info(f"  Range: [{norm_min:.2f}, {norm_max:.2f}] → [0, 1]")
        self.logger.info(f"  Percentiles: {calibration.norm_percentiles}")
        
        return calibration
    
    @abstractmethod
    def create_domain_metadata(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create domain-specific metadata"""
        pass
    
    def preprocess_image(self, image: np.ndarray, calibration: CalibrationData) -> np.ndarray:
        """Apply standard preprocessing steps with [0,1] normalization"""
        # Convert to electrons if needed
        if hasattr(calibration, 'gain') and calibration.gain != 1.0:
            image_electrons = image * calibration.gain
        else:
            image_electrons = image.copy()
        
        # Background subtraction
        if hasattr(calibration, 'background'):
            image_electrons = np.maximum(image_electrons - calibration.background, 0)
        
        # Apply [0,1] normalization using calibration parameters
        if calibration.norm_min is not None and calibration.norm_max is not None:
            # Clip to normalization range
            image_clipped = np.clip(image_electrons, calibration.norm_min, calibration.norm_max)
            # Normalize to [0,1]
            if calibration.norm_max > calibration.norm_min:
                normalized = (image_clipped - calibration.norm_min) / (calibration.norm_max - calibration.norm_min)
            else:
                normalized = np.zeros_like(image_clipped)
        else:
            # Fallback to old normalization if norm parameters not set
            normalized = image_electrons / calibration.scale_factor
        
        return normalized.astype(np.float32)
    
    def create_quality_masks(self, image: np.ndarray, 
                           calibration: CalibrationData) -> Dict[str, np.ndarray]:
        """Create standard quality masks"""
        masks = {}
        
        # Valid pixels mask
        masks['valid'] = np.ones(image.shape[-2:], dtype=bool)
        
        # Saturated pixels
        if hasattr(calibration, 'saturation_threshold'):
            masks['saturated'] = image > calibration.saturation_threshold
            masks['valid'] &= ~masks['saturated']
        
        # Dead/hot pixels (simple detection)
        masks['dead'] = image == 0
        masks['hot'] = image > calibration.white_level
        masks['valid'] &= ~(masks['dead'] | masks['hot'])
        
        return masks
    
    def get_optimal_tile_size(self) -> int:
        """Returns the unified tile size for all domains"""
        # All domains now use unified 256x256 resolution for cross-domain training
        return 256
    
    def get_native_tile_size(self) -> int:
        """Returns the original native tile size for the domain"""
        native_sizes = {
            0: 512,  # Photography: 512x512 for 2048x3072 images
            1: 128,  # Microscopy: 128x128 for 512x512 images  
            2: 1024  # Astronomy: 1024x1024 for 4096x4096 images
        }
        return native_sizes.get(self.domain_id, 128)
    
    def extract_tiles_unified(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract tiles using unified 256x256 resolution with complete coverage"""
        from complete_systematic_tiling import SystematicTiler
        # from unified_resolution_system import UnifiedResolutionManager  # REMOVED - not needed
        import torch
        
        # Create tiling configuration for 256x256 tiles (no unified manager needed)
        from complete_systematic_tiling import SystematicTilingConfig
        tiling_config = SystematicTilingConfig(
            tile_size=256,  # Unified size
            overlap_ratio=0.1,  # Default overlap
            coverage_mode='complete',
            edge_handling='pad_reflect',
            min_valid_ratio=0.5
        )
        tiler = SystematicTiler(tiling_config)
        
        # Extract tiles at unified resolution
        tiles = tiler.extract_tiles(image)
        
        # Process tiles (should already be 256x256 from SystematicTiler)
        unified_tiles = []
        for tile_info in tiles:
            # Ensure tile is 256x256 (should be by default with our config)
            if tile_info.tile_data.shape[-2:] != (256, 256):
                self.logger.warning(f"Tile {tile_info.tile_id} is not 256x256: {tile_info.tile_data.shape}")
                continue
            
            unified_tiles.append({
                'tile_data': tile_info.tile_data,
                'position': tile_info.image_position,
                'tile_id': tile_info.tile_id,
                'grid_position': tile_info.grid_position,
                'valid_ratio': tile_info.valid_ratio,
                'is_edge_tile': tile_info.is_edge_tile,
                'unified_quality': 1.0,  # Default quality score
                'tile_size': 256,  # Always 256x256 now
                'systematic_info': tile_info
            })
        
        self.logger.info(f"Extracted {len(unified_tiles)} unified 256x256 tiles")
        
        return unified_tiles
    
    def extract_tiles_systematic(self, image: np.ndarray, 
                                tile_size: int = None) -> List[Dict[str, Any]]:
        """Legacy method - now uses unified 256x256 resolution"""
        return self.extract_tiles_unified(image)
    
    def extract_tiles(self, image: np.ndarray, tile_size: int = None, 
                     overlap: float = None) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """Legacy tile extraction - now uses unified 256x256 resolution"""
        # Use unified approach but return in legacy format for compatibility
        unified_tiles = self.extract_tiles_unified(image)
        
        # Convert to legacy format
        legacy_tiles = []
        for tile_dict in unified_tiles:
            legacy_tiles.append((tile_dict['tile_data'], tile_dict['position']))
        
        return legacy_tiles
    
    def compute_image_statistics(self, image: np.ndarray, 
                               valid_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute comprehensive image statistics"""
        if valid_mask is not None:
            pixels = image[valid_mask]
        else:
            pixels = image.flatten()
        
        if len(pixels) == 0:
            return {
                "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "percentile_99": 0.0, "snr_estimate": 0.0
            }
        
        stats = {
            "mean": float(np.mean(pixels)),
            "std": float(np.std(pixels)),
            "min": float(np.min(pixels)),
            "max": float(np.max(pixels)),
            "percentile_99": float(np.percentile(pixels, 99)),
        }
        
        # SNR estimate
        if stats["std"] > 0:
            stats["snr_estimate"] = stats["mean"] / stats["std"]
        else:
            stats["snr_estimate"] = 0.0
        
        return stats

class PhotographyProcessor(BaseDomainProcessor):
    """Processor for SID photography dataset (Sony ARW and Fuji RAF files)"""
    
    def __init__(self):
        super().__init__(domain_id=0, domain_name="photography")
        
        # Camera-specific parameters - UPDATED WITH PRECISE MEASUREMENTS
        self.camera_defaults = {
            "sony": {
                "gain": 2.1,  # e-/ADU for Sony A7S II at ISO 2000 (measured)
                "gain_iso_4000": 0.79,  # e-/ADU at ISO 4000 (unity gain, measured)
                "read_noise": 2.5,  # electrons above ISO 4000 (2-3e range, measured)
                "read_noise_low_iso": 6.0,  # electrons below ISO 4000 (measured)
                "bayer_pattern": "RGGB",
                "file_extension": ".ARW",
                "resolution": (2848, 4256)  # H × W (verified from actual data)
            },
            "fuji": {
                "gain": 0.75,   # e-/ADU for Fuji X-T30 near base ISO (measured)
                "read_noise": 2.5,  # electrons at base ISO (measured)
                "bayer_pattern": "RGGB",  # Fuji uses X-Trans but we'll treat as Bayer
                "file_extension": ".RAF", 
                "resolution": (4032, 6032)  # H × W (verified from actual data)
            }
        }
    
    def discover_files(self, raw_data_path: str) -> List[Dict[str, Any]]:
        """Discover Sony ARW and Fuji RAF file pairs"""
        raw_path = Path(raw_data_path)
        pairs = []
        
        # Process Sony data
        sony_short_path = raw_path / "Sony/short"
        sony_long_path = raw_path / "Sony/long"
        
        if sony_short_path.exists() and sony_long_path.exists():
            sony_pairs = self._discover_camera_pairs(
                sony_short_path, sony_long_path, "sony", "*.ARW"
            )
            pairs.extend(sony_pairs)
            self.logger.info(f"Discovered {len(sony_pairs)} Sony pairs")
        
        # Process Fuji data
        fuji_short_path = raw_path / "Fuji/short"
        fuji_long_path = raw_path / "Fuji/long"
        
        if fuji_short_path.exists() and fuji_long_path.exists():
            fuji_pairs = self._discover_camera_pairs(
                fuji_short_path, fuji_long_path, "fuji", "*.RAF"
            )
            pairs.extend(fuji_pairs)
            self.logger.info(f"Discovered {len(fuji_pairs)} Fuji pairs")
        
        self.logger.info(f"Total discovered {len(pairs)} photography pairs (Sony + Fuji)")
        return pairs
    
    def _discover_camera_pairs(self, short_path: Path, long_path: Path, 
                              camera_type: str, pattern: str) -> List[Dict[str, Any]]:
        """Discover file pairs for a specific camera type"""
        short_files = list(short_path.glob(pattern))
        long_files = list(long_path.glob(pattern))
        
        pairs = []
        for short_file in short_files:
            # Extract scene ID (e.g., 00001_00_0.04s.ARW -> 00001_00)
            scene_parts = short_file.stem.split("_")
            if len(scene_parts) >= 2:
                scene_id = "_".join(scene_parts[:2])
                
                # Find corresponding long exposure
                long_candidates = [f for f in long_files if scene_id in f.stem]
                if long_candidates:
                    pairs.append({
                        "scene_id": scene_id,
                        "camera_type": camera_type,
                        "noisy_path": str(short_file),
                        "clean_path": str(long_candidates[0]),
                        "exposure_short": self._extract_exposure_time(short_file.stem),
                        "exposure_long": self._extract_exposure_time(long_candidates[0].stem),
                        "file_format": self.camera_defaults[camera_type]["file_extension"]
                    })
        
        return pairs
    
    def load_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load Sony ARW or Fuji RAF file and extract Bayer data"""
        import rawpy
        
        # Determine camera type from file extension
        file_ext = Path(file_path).suffix.upper()
        camera_type = "sony" if file_ext == ".ARW" else "fuji" if file_ext == ".RAF" else "unknown"
        
        with rawpy.imread(file_path) as raw:
            # Get raw Bayer data
            bayer = raw.raw_image_visible.astype(np.float32)
            
            # Extract metadata with camera-specific defaults
            camera_defaults = self.camera_defaults.get(camera_type, self.camera_defaults["sony"])
            
            metadata = {
                "black_level": np.array(raw.black_level_per_channel),
                "white_level": raw.white_level,
                "camera_model": getattr(raw, 'camera', camera_type.title()),
                "camera_type": camera_type,
                "iso": getattr(raw, 'iso', None),
                "exposure_time": getattr(raw, 'exposure_time', None),
                "color_desc": getattr(raw, 'color_desc', 'RGBG'),
                "raw_pattern": raw.raw_pattern.tolist(),
                "daylight_whitebalance": getattr(raw, 'daylight_whitebalance', None),
                "file_format": file_ext,
                "expected_resolution": camera_defaults["resolution"]
            }
        
        # Pack Bayer to 4-channel RGGB (handles both Sony and Fuji)
        packed = self._pack_bayer_to_4channel(bayer, metadata["black_level"], camera_type)
        
        return packed, metadata
    
    def _pack_bayer_to_4channel(self, bayer: np.ndarray, 
                               black_level: np.ndarray, camera_type: str = "sony") -> np.ndarray:
        """Convert Bayer pattern to 4-channel packed format"""
        H, W = bayer.shape
        
        # Create black level map
        black_map = np.zeros((H, W), dtype=np.float32)
        black_map[0::2, 0::2] = black_level[0]  # R
        black_map[0::2, 1::2] = black_level[1]  # G1
        black_map[1::2, 0::2] = black_level[2]  # G2
        black_map[1::2, 1::2] = black_level[3]  # B
        
        # Subtract black level
        bayer_corrected = np.maximum(bayer - black_map, 0)
        
        # Pack to 4-channel
        packed = np.zeros((4, H//2, W//2), dtype=np.float32)
        packed[0] = bayer_corrected[0::2, 0::2]  # R
        packed[1] = bayer_corrected[0::2, 1::2]  # G1
        packed[2] = bayer_corrected[1::2, 0::2]  # G2
        packed[3] = bayer_corrected[1::2, 1::2]  # B
        
        return packed
    
    def estimate_calibration(self, image_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                            metadata: List[Dict[str, Any]] = None) -> CalibrationData:
        """Estimate gain and read noise using physics-based photon transfer curve"""
        
        # Use physics-based calibration
        from physics_based_calibration import PhotographyPhotonTransferCalibration
        
        calibrator = PhotographyPhotonTransferCalibration()
        
        # Convert image pairs format if needed
        if image_pairs and isinstance(image_pairs[0], np.ndarray):
            # Handle case where single images are passed instead of pairs
            single_images = image_pairs
            # Create dummy pairs for fallback
            image_pairs = [(img, img) for img in single_images[:len(single_images)//2]]
        
        # Use physics-based calibration
        physics_calib = calibrator.estimate_calibration(image_pairs, metadata or [])
        
        self.logger.info(f"Physics-based calibration: gain={physics_calib.gain:.3f}±{physics_calib.gain_uncertainty:.3f}, "
                        f"read_noise={physics_calib.read_noise:.2f}±{physics_calib.read_noise_uncertainty:.2f}")
        
        # Convert to legacy CalibrationData format
        calibration = CalibrationData(
            gain=physics_calib.gain,
            read_noise=physics_calib.read_noise,
            black_level=physics_calib.black_level,
            white_level=physics_calib.white_level,
            background=physics_calib.background,
            scale_factor=physics_calib.scale_factor
        )
        
        # Compute [0,1] normalization parameters from image pairs
        if image_pairs:
            # Extract all images from pairs for normalization computation
            all_images = []
            for pair in image_pairs:
                if isinstance(pair, tuple) and len(pair) == 2:
                    all_images.extend([pair[0], pair[1]])
                else:
                    all_images.append(pair)
            
            # Compute normalization parameters
            calibration = self.compute_normalization_parameters(all_images, calibration)
        
        return calibration
    
    def create_domain_metadata(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create photography-specific metadata"""
        camera_type = file_info.get("camera_type", "sony")
        camera_defaults = self.camera_defaults.get(camera_type, self.camera_defaults["sony"])
        
        return {
            "photography_meta": {
                "camera_model": file_info.get("camera_model", camera_type.title()),
                "camera_type": camera_type,
                "iso": file_info.get("iso"),
                "exposure_time": file_info.get("exposure_time"),
                "bayer_pattern": camera_defaults["bayer_pattern"],
                "file_format": file_info.get("file_format", camera_defaults["file_extension"]),
                "resolution": camera_defaults["resolution"],
                "scene_category": self._classify_scene(file_info),
                "lens_model": None,  # Not available in raw metadata
                "aperture": None,
                "focal_length": None,
                "white_balance": file_info.get("daylight_whitebalance"),
                "color_space": "Linear"
            }
        }
    
    def _extract_exposure_time(self, filename: str) -> Optional[float]:
        """Extract exposure time from filename"""
        parts = filename.split("_")
        for part in parts:
            if "s." in part or "s" == part[-1]:
                try:
                    return float(part.replace("s", ""))
                except ValueError:
                    continue
        return None
    
    def _classify_scene(self, file_info: Dict[str, Any]) -> str:
        """Simple scene classification based on filename/metadata"""
        filename = file_info.get("source_path", "").lower()
        
        if "indoor" in filename or "interior" in filename:
            return "indoor"
        elif "outdoor" in filename or "exterior" in filename:
            return "outdoor"
        elif "night" in filename or "dark" in filename:
            return "night"
        else:
            return "unknown"

class MicroscopyProcessor(BaseDomainProcessor):
    """Processor for BioSR microscopy dataset (MRC files)"""
    
    def __init__(self):
        super().__init__(domain_id=1, domain_name="microscopy")
        
        # BioSR-specific parameters
        self.biosr_config = {
            "structures": ["CCPs", "ER", "F-actin", "Microtubules"],
            "noise_levels": list(range(1, 10)),
            "photon_count_range": (15, 600)  # Average photon counts
        }
        
        # sCMOS camera specifications for BioSR dataset - UPDATED WITH PRECISE MEASUREMENTS
        self.scmos_defaults = {
            "gain": 1.0,  # e-/ADU (BioSR typical range: 0.5-1.5, measured)
            "read_noise": 1.5,  # electrons RMS per pixel (BioSR typical: 1-2, measured)
            "detector_type": "sCMOS",
            "bit_depth": 16,
            "quantum_efficiency": 0.85,  # Typical for modern sCMOS
            "full_well_capacity": 30000,  # electrons (typical sCMOS)
            "pixel_size_um": 6.5,  # Typical sCMOS pixel size
            "dark_current": 0.1  # e-/pixel/s at room temperature
        }
    
    def discover_files(self, raw_data_path: str) -> List[Dict[str, Any]]:
        """Discover BioSR MRC files"""
        raw_path = Path(raw_data_path)
        files = []
        
        for structure in self.biosr_config["structures"]:
            structure_path = raw_path / "structures" / structure
            if not structure_path.exists():
                continue
            
            # Process each cell directory
            cell_dirs = [d for d in structure_path.iterdir() 
                        if d.is_dir() and d.name.startswith("Cell_")]
            
            for cell_dir in cell_dirs:
                cell_id = cell_dir.name
                
                # Ground truth file
                gt_file = cell_dir / "RawSIMData_gt.mrc"
                if gt_file.exists():
                    files.append({
                        "structure": structure,
                        "cell_id": cell_id,
                        "file_path": str(gt_file),
                        "data_type": "clean",
                        "noise_level": None,
                        "is_ground_truth": True
                    })
                
                # Noise level files
                for level in self.biosr_config["noise_levels"]:
                    level_file = cell_dir / f"RawSIMData_level_{level:02d}.mrc"
                    if level_file.exists():
                        files.append({
                            "structure": structure,
                            "cell_id": cell_id,
                            "file_path": str(level_file),
                            "data_type": "noisy",
                            "noise_level": level,
                            "is_ground_truth": False
                        })
        
        self.logger.info(f"Discovered {len(files)} microscopy files")
        return files
    
    def load_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load MRC file using BioSR custom reader"""
        import sys
        
        # Add BioSR MRC reader to path
        biosr_path = '/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/data/raw/microscopy/Supplementary Files for BioSR/IO_MRC_Python'
        if biosr_path not in sys.path:
            sys.path.append(biosr_path)
        
        try:
            from read_mrc import read_mrc
            
            # Use BioSR custom MRC reader
            header, data = read_mrc(file_path)
            data = data.astype(np.float32)
            
            # Extract header information
            metadata = {
                "data_type": str(data.dtype),
                "shape": data.shape,
                "header": header if header is not None else {}
            }
            
        except ImportError:
            # Fallback to standard mrcfile if BioSR reader not available
            import mrcfile
            
            with mrcfile.open(file_path, mode='r') as mrc:
                data = mrc.data.astype(np.float32)
                
                # Extract header information
                metadata = {
                    "pixel_spacing": mrc.voxel_size,
                    "data_type": str(mrc.data.dtype),
                    "shape": mrc.data.shape,
                    "header": dict(mrc.header) if hasattr(mrc, 'header') else {}
                }
        
        # Handle 3D data (take first slice for now)
        if len(data.shape) == 3:
            data = data[:, :, 0]  # Take first slice (Z=0)
        
        # Add channel dimension for consistency
        if len(data.shape) == 2:
            data = data[np.newaxis, :, :]
        
        return data, metadata
    
    def estimate_calibration(self, images: List[np.ndarray]) -> CalibrationData:
        """Estimate calibration for microscopy data"""
        # Microscopy data is often already in photons/electrons
        all_pixels = []
        for img in images[:10]:
            all_pixels.extend(img.flatten())
        
        scale_factor = float(np.percentile(all_pixels, 99.9))
        
        calibration = CalibrationData(
            gain=self.scmos_defaults["gain"],  # sCMOS gain: 0.5-1.5 e-/ADU
            read_noise=self.scmos_defaults["read_noise"],  # sCMOS read noise: 1-2 electrons
            black_level=100.0,  # Typical offset
            white_level=65535.0,  # 16-bit
            background=float(np.percentile(all_pixels, 10)),
            scale_factor=scale_factor
        )
        
        # Compute [0,1] normalization parameters
        if images:
            calibration = self.compute_normalization_parameters(images, calibration)
        
        return calibration
    
    def create_domain_metadata(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create microscopy-specific metadata"""
        # Estimate photon count from noise level
        noise_level = file_info.get("noise_level")
        if noise_level is not None:
            # Linear interpolation between 15-600 photons
            photon_count = 15 + (noise_level - 1) * (600 - 15) / 8
        else:
            photon_count = 600  # Ground truth assumed high SNR
        
        return {
            "microscopy_meta": {
                "structure_type": file_info.get("structure"),
                "cell_id": file_info.get("cell_id"),
                "imaging_mode": "SIM",  # Structured Illumination Microscopy
                "upscaling_factor": "linear",  # Default assumption
                "signal_level_index": noise_level,
                "photon_count_avg": photon_count,
                "pixel_size_nm": 32.5,  # Typical for super-resolution
                "objective_na": 1.4,  # High NA objective
                "wavelength_nm": 488.0,  # Typical fluorescence
                "acquisition_parameters": {
                    "exposure_time_ms": "100",
                    "gain": self.scmos_defaults["gain"],
                    "binning": "1x1"
                },
                "camera_specifications": {
                    "detector_type": self.scmos_defaults["detector_type"],
                    "read_noise_electrons": self.scmos_defaults["read_noise"],
                    "quantum_efficiency": self.scmos_defaults["quantum_efficiency"],
                    "full_well_electrons": self.scmos_defaults["full_well_capacity"],
                    "pixel_size_um": self.scmos_defaults["pixel_size_um"],
                    "bit_depth": self.scmos_defaults["bit_depth"],
                    "dark_current_e_per_pixel_per_s": self.scmos_defaults["dark_current"]
                }
            }
        }

class AstronomyProcessor(BaseDomainProcessor):
    """Processor for HLA astronomy dataset (FITS files) with asinh scaling"""
    
    def __init__(self):
        super().__init__(domain_id=2, domain_name="astronomy")
        
        # HST-specific parameters (accurate values from HST documentation)
        self.hst_defaults = {
            "ACS_WFC": {
                "gain": 1.0,  # e-/ADU (default for most amplifiers)
                "read_noise": 3.5,  # electrons (typical 2.0-5.0 range, using mid-value)
                "detector": "WFC",
                "pixel_scale": 0.05  # arcsec/pixel
            },
            "WFC3_UVIS": {
                "gain": 1.5,  # e-/ADU (default)
                "read_noise": 3.1,  # electrons (mean 2.95-3.22 across amplifiers)
                "detector": "UVIS", 
                "pixel_scale": 0.04  # arcsec/pixel
            },
            "WFC3_IR": {
                "gain": 2.5,  # e-/ADU (default)
                "read_noise": 15.0,  # electrons (typical <20, using conservative estimate)
                "detector": "IR",
                "pixel_scale": 0.13  # arcsec/pixel
            },
            "WFPC2": {
                "gain": 7.0,  # e-/DN (ATD-GAIN=7, measured - also supported: 14)
                "read_noise": 6.5,  # electrons RMS (gain 7, measured range 5-8)
                "detector": "WF/PC",
                "pixel_scale": 0.1  # arcsec/pixel
            },
            "instruments": ["ACS", "WFC3", "WFPC2"]
        }
        
        # Initialize astronomy-specific asinh preprocessing
        from astronomy_asinh_preprocessing import AstronomyAsinhPreprocessor, AsinhScalingConfig
        
        self.asinh_config = AsinhScalingConfig(
            softening_factor=1e-3,  # Standard for astronomy
            scale_percentile=99.5,  # Capture bright sources
            background_percentile=10.0,  # Robust background estimation
            cosmic_ray_threshold=5.0  # 5-sigma cosmic ray detection
        )
        
        self.asinh_preprocessor = AstronomyAsinhPreprocessor(self.asinh_config)
    
    def discover_files(self, raw_data_path: str) -> List[Dict[str, Any]]:
        """Discover HLA FITS files"""
        raw_path = Path(raw_data_path)
        files = []
        
        # Direct images
        direct_path = raw_path / "hla_associations/direct_images"
        if direct_path.exists():
            fits_files = list(direct_path.glob("*.fits*"))
            
            for fits_file in fits_files:
                files.append({
                    "file_path": str(fits_file),
                    "image_type": "direct",
                    "association_id": fits_file.stem.split("_")[0],
                    "data_type": "noisy"  # Real observations
                })
        
        # G800L spectroscopic images
        g800l_path = raw_path / "hla_associations/g800l_images"
        if g800l_path.exists():
            fits_files = list(g800l_path.glob("*.fits*"))
            
            for fits_file in fits_files:
                files.append({
                    "file_path": str(fits_file),
                    "image_type": "g800l",
                    "association_id": fits_file.stem.split("_")[0],
                    "data_type": "noisy"
                })
        
        self.logger.info(f"Discovered {len(files)} astronomy files")
        return files
    
    def load_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load FITS file"""
        from astropy.io import fits
        import gzip
        
        # Handle gzipped files
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rb') as f:
                hdul = fits.open(f)
                data = hdul[0].data.astype(np.float32)
                header = dict(hdul[0].header)
                hdul.close()
        else:
            with fits.open(file_path) as hdul:
                data = hdul[0].data.astype(np.float32)
                header = dict(hdul[0].header)
        
        # Add channel dimension
        if len(data.shape) == 2:
            data = data[np.newaxis, :, :]
        
        # Extract relevant metadata from header
        metadata = {
            "telescope": header.get("TELESCOP", "HST"),
            "instrument": header.get("INSTRUME", "ACS"),
            "detector": header.get("DETECTOR", "WFC"),
            "filter": header.get("FILTER", "CLEAR"),
            "exposure_time": header.get("EXPTIME", 0.0),
            "gain": self._get_instrument_gain(header),
            "read_noise": self._get_instrument_read_noise(header),
            "ra": header.get("RA_TARG", 0.0),
            "dec": header.get("DEC_TARG", 0.0),
            "proposal_id": header.get("PROPOSID"),
            "observation_id": header.get("ROOTNAME"),
            "date_obs": header.get("DATE-OBS"),
            "full_header": header
        }
        
        return data, metadata
    
    def _get_instrument_gain(self, header: Dict[str, Any]) -> float:
        """Get appropriate gain value based on instrument and detector"""
        instrument = header.get("INSTRUME", "ACS")
        detector = header.get("DETECTOR", "WFC")
        
        # Check header first for explicit gain value
        if "GAIN" in header and header["GAIN"] is not None:
            return float(header["GAIN"])
        
        # Use instrument-specific defaults
        instrument_key = f"{instrument}_{detector}"
        if instrument_key in self.hst_defaults:
            return self.hst_defaults[instrument_key]["gain"]
        elif instrument == "ACS":
            return self.hst_defaults["ACS_WFC"]["gain"]
        elif instrument == "WFC3":
            # Default to UVIS if detector not specified
            if detector in ["IR", "INFRARED"]:
                return self.hst_defaults["WFC3_IR"]["gain"]
            else:
                return self.hst_defaults["WFC3_UVIS"]["gain"]
        elif instrument == "WFPC2":
            return self.hst_defaults["WFPC2"]["gain"]
        else:
            # Fallback to ACS default
            return self.hst_defaults["ACS_WFC"]["gain"]
    
    def _get_instrument_read_noise(self, header: Dict[str, Any]) -> float:
        """Get appropriate read noise value based on instrument and detector"""
        instrument = header.get("INSTRUME", "ACS")
        detector = header.get("DETECTOR", "WFC")
        
        # Check header first for explicit read noise value
        if "RDNOISE" in header and header["RDNOISE"] is not None:
            return float(header["RDNOISE"])
        
        # Use instrument-specific defaults
        instrument_key = f"{instrument}_{detector}"
        if instrument_key in self.hst_defaults:
            return self.hst_defaults[instrument_key]["read_noise"]
        elif instrument == "ACS":
            return self.hst_defaults["ACS_WFC"]["read_noise"]
        elif instrument == "WFC3":
            # Default to UVIS if detector not specified
            if detector in ["IR", "INFRARED"]:
                return self.hst_defaults["WFC3_IR"]["read_noise"]
            else:
                return self.hst_defaults["WFC3_UVIS"]["read_noise"]
        elif instrument == "WFPC2":
            return self.hst_defaults["WFPC2"]["read_noise"]
        else:
            # Fallback to ACS default
            return self.hst_defaults["ACS_WFC"]["read_noise"]
    
    def estimate_calibration(self, images: List[np.ndarray]) -> CalibrationData:
        """Estimate calibration for astronomy data"""
        all_pixels = []
        for img in images[:10]:
            all_pixels.extend(img.flatten())
        
        scale_factor = float(np.percentile(all_pixels, 99.9))
        
        calibration = CalibrationData(
            gain=self.hst_defaults["ACS_WFC"]["gain"],  # Use ACS as default
            read_noise=self.hst_defaults["ACS_WFC"]["read_noise"],
            black_level=0.0,  # Already bias-subtracted
            white_level=65535.0,
            background=float(np.percentile(all_pixels, 10)),
            scale_factor=scale_factor
        )
        
        # Compute [0,1] normalization parameters
        if images:
            calibration = self.compute_normalization_parameters(images, calibration)
        
        return calibration
    
    def preprocess_image(self, image: np.ndarray, calibration: CalibrationData) -> np.ndarray:
        """Apply astronomy-specific preprocessing with asinh scaling"""
        
        # Convert calibration to dict format for asinh preprocessor
        calib_dict = {
            'gain': calibration.gain,
            'read_noise': calibration.read_noise,
            'background': calibration.background,
            'scale_factor': calibration.scale_factor
        }
        
        # Apply astronomy-specific asinh preprocessing
        result = self.asinh_preprocessor.preprocess_astronomy_image(
            image, calib_dict, apply_cosmic_ray_removal=True
        )
        
        # Log preprocessing metadata
        metadata = result['preprocessing_metadata']
        self.logger.info(f"Astronomy preprocessing applied:")
        self.logger.info(f"  Background: {metadata['background_level']:.1f} electrons")
        self.logger.info(f"  Scale: {metadata['scale_factor']:.1f} electrons") 
        self.logger.info(f"  Cosmic rays detected: {metadata['cosmic_rays_detected']}")
        self.logger.info(f"  Asinh softening: {metadata['softening_factor']:.3f}")
        
        return result['preprocessed_image']
    
    def get_preprocessing_metadata(self) -> Dict[str, Any]:
        """Get astronomy-specific preprocessing metadata"""
        base_metadata = self.asinh_preprocessor.get_preprocessing_stats()
        base_metadata.update({
            'domain': 'astronomy',
            'cosmic_ray_detection': True,
            'preprocessing_method': 'asinh_scaling',
            'recommended_for': 'high_dynamic_range_astronomical_data'
        })
        return base_metadata
    
    def create_domain_metadata(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create astronomy-specific metadata"""
        return {
            "astronomy_meta": {
                "telescope": file_info.get("telescope", "HST"),
                "instrument": file_info.get("instrument", "ACS"),
                "filter_name": file_info.get("filter", "F814W"),
                "detector": file_info.get("detector", "WFC"),
                "proposal_id": str(file_info.get("proposal_id", "")),
                "observation_id": file_info.get("observation_id", ""),
                "target_name": file_info.get("targname", ""),
                "ra_deg": float(file_info.get("ra", 0.0)),
                "dec_deg": float(file_info.get("dec", 0.0)),
                "exposure_time": float(file_info.get("exposure_time", 0.0)),
                "observation_date": file_info.get("date_obs", ""),
                "airmass": file_info.get("airmass", 1.0),
                "seeing_arcsec": file_info.get("seeing", 0.1),
                "sky_background": float(file_info.get("background", 0.0)),
                "photometric_zeropoint": file_info.get("photzp", 25.0)
            }
        }
    
    def create_quality_masks(self, image: np.ndarray, 
                           calibration: CalibrationData) -> Dict[str, np.ndarray]:
        """Create astronomy-specific quality masks including cosmic rays"""
        masks = super().create_quality_masks(image, calibration)
        
        # Cosmic ray detection using median filtering
        from scipy.ndimage import median_filter
        
        if len(image.shape) == 3:
            img_2d = image[0]  # Use first channel
        else:
            img_2d = image
        
        med_filtered = median_filter(img_2d, size=5)
        cosmic_ray_threshold = 5 * calibration.read_noise
        masks['cosmic_rays'] = (img_2d - med_filtered) > cosmic_ray_threshold
        masks['valid'] &= ~masks['cosmic_rays']
        
        return masks

# Factory function for creating processors
def create_processor(domain_name: str) -> BaseDomainProcessor:
    """Factory function to create appropriate processor"""
    processors = {
        "photography": PhotographyProcessor,
        "microscopy": MicroscopyProcessor,
        "astronomy": AstronomyProcessor
    }
    
    if domain_name not in processors:
        raise ValueError(f"Unknown domain: {domain_name}")
    
    return processors[domain_name]()

# Example usage
if __name__ == "__main__":
    # Test each processor
    domains = ["photography", "microscopy", "astronomy"]
    
    for domain in domains:
        processor = create_processor(domain)
        print(f"✅ Created {domain} processor: {processor.__class__.__name__}")
        
        # Test file discovery (would need actual data paths)
        # files = processor.discover_files(f"/path/to/{domain}/data")
        # print(f"  Would discover files: {len(files) if files else 0}")
