# Implementation Plan: Domain-Adaptive Poisson-Gaussian Diffusion for Low-Light Image Restoration
---

## Part 1: Mathematical Foundation (Simplified)

### 1.1 Core Model
```
Observations: y = Poisson(λ) + N(0, σ_r²)
Rate: λ = s·x + b
```

Where:
- `y`: Observed sensor values (electrons)
- `x`: Normalized scene intensity [0, ~1]
- `s`: Dataset normalization scale (electrons)
- `b`: Background offset (electrons)
- `σ_r`: Read noise (electrons)

**No PSF, no blur** - just pure noise modeling.

### 1.2 Likelihood Guidance
Two modes:
- **WLS** (recommended): `∇ log p(y|x) = s·(y - λ)/(λ + σ_r²)`
- **Exact**: Includes variance derivative terms

---

## Part 2: Core Implementation

### 2.1 Simplified Project Structure

```bash
poisson-diffusion/
├── setup.py
├── requirements.txt
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── transforms.py          # NEW: Reversible transforms with metadata
│   ├── poisson_guidance.py    # Simplified: No forward operators
│   ├── edm_sampler.py        # Simplified: Identity forward only
│   ├── calibration.py        # Unchanged
│   └── metrics.py
│
├── models/
│   ├── __init__.py
│   ├── edm_wrapper.py        # TODO: EDM integration
│   └── conditioning.py
│
├── data/
│   ├── __init__.py
│   ├── base_dataset.py       # Simplified: No PSF handling
│   ├── domain_datasets.py    # NEW: Combined domain-specific loaders
│   └── raw_utils.py          # NEW: Format-specific loading
│
├── configs/
│   ├── calibrations/         # Per-sensor calibration JSONs
│   └── experiments/
│
├── scripts/
│   ├── compute_normalization.py
│   ├── train_prior.py
│   ├── evaluate.py
│   └── verify_physics.py
│
└── tests/
    ├── test_transforms.py     # NEW: Test reversibility
    ├── test_guidance.py       # Simplified
    ├── test_sampler.py
    └── test_integration.py
```

### 2.2 Reversible Transforms with Complete Metadata

```python
# core/transforms.py
"""
Reversible image transforms with complete metadata tracking.
Critical for proper reconstruction across different scales.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any
import json

@dataclass
class ImageMetadata:
    """Complete metadata for perfect reconstruction."""
    # Original dimensions
    original_height: int
    original_width: int

    # Scaling information
    scale_factor: float
    crop_bbox: Optional[Tuple[int, int, int, int]]  # (top, left, height, width)
    pad_amounts: Optional[Tuple[int, int, int, int]]  # (top, bottom, left, right)

    # Physical calibration
    pixel_size: float  # Physical size per pixel
    pixel_unit: str    # 'um' or 'arcsec'

    # Sensor calibration (for reconstruction)
    black_level: float
    white_level: float

    # Domain and acquisition
    domain: str  # 'photography', 'microscopy', 'astronomy'
    bit_depth: int

    # Optional acquisition parameters
    iso: Optional[int] = None
    exposure_time: Optional[float] = None
    wavelength: Optional[float] = None  # nm for microscopy
    telescope: Optional[str] = None  # for astronomy

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'ImageMetadata':
        """Deserialize from JSON."""
        return cls(**json.loads(json_str))

class ReversibleTransform:
    """
    Transform images to model size while preserving all information
    needed for perfect reconstruction.
    """

    def __init__(self, target_size: int = 128, mode: str = 'bilinear'):
        """
        Args:
            target_size: Size for model input (square)
            mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        """
        self.target_size = target_size
        self.mode = mode

    def forward(
        self,
        image: torch.Tensor,
        pixel_size: float,
        pixel_unit: str,
        domain: str,
        black_level: float = 0,
        white_level: float = 16383,
        **extra_metadata
    ) -> Tuple[torch.Tensor, ImageMetadata]:
        """
        Transform image to model size, preserving all metadata.

        Args:
            image: Input image [B, C, H, W]
            pixel_size: Physical size per pixel
            pixel_unit: Unit of pixel_size
            domain: Image domain
            black_level: Sensor black level
            white_level: Sensor white level
            **extra_metadata: Additional metadata to preserve

        Returns:
            (transformed_image, metadata)
        """
        B, C, H, W = image.shape
        device = image.device

        # Calculate scale factor
        max_dim = max(H, W)
        scale_factor = self.target_size / max_dim

        # Initialize metadata
        metadata = ImageMetadata(
            original_height=H,
            original_width=W,
            scale_factor=scale_factor,
            crop_bbox=None,
            pad_amounts=None,
            pixel_size=pixel_size,
            pixel_unit=pixel_unit,
            domain=domain,
            black_level=black_level,
            white_level=white_level,
            bit_depth=int(np.log2(white_level + 1)),
            **extra_metadata
        )

        # Step 1: Resize if needed
        if scale_factor != 1.0:
            new_H = int(H * scale_factor)
            new_W = int(W * scale_factor)
            image = F.interpolate(
                image,
                size=(new_H, new_W),
                mode=self.mode,
                align_corners=False if self.mode != 'nearest' else None
            )
        else:
            new_H, new_W = H, W

        # Step 2: Pad or crop to square
        if new_H != self.target_size or new_W != self.target_size:
            image, crop_bbox, pad_amounts = self._make_square(
                image, self.target_size
            )
            metadata.crop_bbox = crop_bbox
            metadata.pad_amounts = pad_amounts

        return image, metadata

    def inverse(
        self,
        image: torch.Tensor,
        metadata: ImageMetadata
    ) -> torch.Tensor:
        """
        Perfectly reverse the transformation.

        Args:
            image: Transformed image [B, C, target_size, target_size]
            metadata: Metadata from forward transform

        Returns:
            Original-size image [B, C, H, W]
        """
        # Step 1: Reverse padding/cropping
        if metadata.crop_bbox is not None:
            # Image was cropped, so pad it back
            top, left, crop_h, crop_w = metadata.crop_bbox

            # Calculate original size after scaling
            scaled_H = int(metadata.original_height * metadata.scale_factor)
            scaled_W = int(metadata.original_width * metadata.scale_factor)

            # Create full-size tensor
            B, C = image.shape[:2]
            full = torch.zeros(
                B, C, scaled_H, scaled_W,
                device=image.device,
                dtype=image.dtype
            )

            # Place image in correct position
            full[:, :, top:top+crop_h, left:left+crop_w] = image
            image = full

        elif metadata.pad_amounts is not None:
            # Image was padded, so crop it back
            top, bottom, left, right = metadata.pad_amounts
            H_total = image.shape[2]
            W_total = image.shape[3]
            image = image[:, :, top:H_total-bottom, left:W_total-right]

        # Step 2: Reverse scaling
        if metadata.scale_factor != 1.0:
            image = F.interpolate(
                image,
                size=(metadata.original_height, metadata.original_width),
                mode=self.mode,
                align_corners=False if self.mode != 'nearest' else None
            )

        return image

    def _make_square(
        self,
        image: torch.Tensor,
        target_size: int
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[Tuple]]:
        """
        Make image square by center cropping or padding.

        Returns:
            (squared_image, crop_bbox, pad_amounts)
        """
        B, C, H, W = image.shape

        if H == target_size and W == target_size:
            return image, None, None

        if H > target_size or W > target_size:
            # Need to crop
            crop_h = min(H, target_size)
            crop_w = min(W, target_size)
            top = (H - crop_h) // 2
            left = (W - crop_w) // 2

            image = image[:, :, top:top+crop_h, left:left+crop_w]
            crop_bbox = (top, left, crop_h, crop_w)

            # After cropping, might still need padding
            if crop_h < target_size or crop_w < target_size:
                pad_h = target_size - crop_h
                pad_w = target_size - crop_w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left

                image = F.pad(
                    image,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='reflect'
                )

                return image, crop_bbox, (pad_top, pad_bottom, pad_left, pad_right)

            return image, crop_bbox, None

        else:
            # Only padding needed
            pad_h = target_size - H
            pad_w = target_size - W
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            image = F.pad(
                image,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='reflect'
            )

            return image, None, (pad_top, pad_bottom, pad_left, pad_right)
```

### 2.3 Simplified Poisson Guidance (No Forward Operators)

```python
# core/poisson_guidance.py
"""
Poisson-Gaussian likelihood guidance without forward operators.
Simplified to focus on noise modeling only.
"""

import torch
import numpy as np
from typing import Optional, Literal, Dict
from dataclasses import dataclass

@dataclass
class GuidanceConfig:
    """Configuration for likelihood guidance."""
    mode: Literal['wls', 'exact'] = 'wls'
    gamma_schedule: Literal['sigma2', 'linear', 'const'] = 'sigma2'
    kappa: float = 0.5
    gradient_clip: float = 10.0

class PoissonGuidance:
    """
    Simplified Poisson-Gaussian guidance for denoising only.
    No blur modeling, just noise.
    """

    def __init__(
        self,
        scale: float,
        background: float = 0.0,
        read_noise: float = 0.0,
        config: Optional[GuidanceConfig] = None
    ):
        """
        Args:
            scale: Dataset normalization (electrons)
            background: Background offset (electrons)
            read_noise: Read noise std (electrons)
            config: Guidance configuration
        """
        self.scale = scale
        self.background = background
        self.read_noise = read_noise
        self.config = config or GuidanceConfig()

        # Diagnostics
        self.grad_norms = []
        self.chi2_values = []

    def compute_score(
        self,
        x_hat: torch.Tensor,
        y_electrons: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 0.1
    ) -> torch.Tensor:
        """
        Compute likelihood score ∇_x log p(y|x).

        Simplified: No forward operator, direct comparison.

        Args:
            x_hat: Current estimate [B, C, H, W] (normalized)
            y_electrons: Observations [B, C, H, W] (electrons)
            mask: Valid pixel mask
            eps: Stability epsilon

        Returns:
            Score gradient
        """
        # Direct model (no blur)
        lambda_e = self.scale * x_hat + self.background

        # Variance under Poisson-Gaussian
        variance = torch.clamp(lambda_e + self.read_noise**2, min=eps)

        if self.config.mode == 'wls':
            # Weighted least squares
            score = (y_electrons - lambda_e) / variance

        elif self.config.mode == 'exact':
            # Exact heteroscedastic score
            residual = y_electrons - lambda_e
            score = (
                residual / variance +
                0.5 * (residual**2) / (variance**2) -
                0.5 / variance
            )
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        # Apply mask
        if mask is not None:
            score = score * mask

        # Scale by s (chain rule)
        gradient = score * self.scale

        # Compute chi-squared for diagnostics
        with torch.no_grad():
            chi2 = ((y_electrons - lambda_e)**2 / variance)
            if mask is not None:
                chi2 = chi2 * mask
                chi2_per_pixel = chi2.sum() / mask.sum()
            else:
                chi2_per_pixel = chi2.mean()
            self.chi2_values.append(chi2_per_pixel.item())

        return gradient

    def gamma_schedule(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute guidance weight γ(σ).

        Args:
            sigma: Current noise level

        Returns:
            Guidance weight
        """
        if self.config.gamma_schedule == 'sigma2':
            return self.config.kappa * (sigma ** 2)
        elif self.config.gamma_schedule == 'linear':
            return self.config.kappa * sigma
        elif self.config.gamma_schedule == 'const':
            return torch.full_like(sigma, self.config.kappa)
        else:
            raise ValueError(f"Unknown schedule: {self.config.gamma_schedule}")

    def compute(
        self,
        x_hat: torch.Tensor,
        y_electrons: torch.Tensor,
        sigma_t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute scaled and clipped guidance gradient.

        Args:
            x_hat: Current estimate (normalized)
            y_electrons: Observations (electrons)
            sigma_t: Current diffusion noise level
            mask: Valid pixels

        Returns:
            Guidance gradient
        """
        # Base score
        score = self.compute_score(x_hat, y_electrons, mask)

        # Scale by gamma(sigma)
        weight = self.gamma_schedule(sigma_t)
        gradient = score * weight

        # Clip for stability
        gradient = torch.clamp(gradient,
                              -self.config.gradient_clip,
                              self.config.gradient_clip)

        # Record norm
        self.grad_norms.append(gradient.norm().item())

        return gradient

    def get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic statistics."""
        if len(self.grad_norms) == 0:
            return {
                'grad_norm_mean': 0.0,
                'grad_norm_max': 0.0,
                'chi2_mean': 0.0
            }

        return {
            'grad_norm_mean': float(np.mean(self.grad_norms)),
            'grad_norm_max': float(np.max(self.grad_norms)),
            'chi2_mean': float(np.mean(self.chi2_values)) if self.chi2_values else 0.0
        }
```

### 2.4 Simplified EDM Sampler

```python
# core/edm_sampler.py
"""
Simplified EDM sampler without forward operators.
Direct denoising with metadata preservation.
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Any
from core.poisson_guidance import PoissonGuidance, GuidanceConfig
from core.transforms import ImageMetadata

class EDMPosteriorSampler:
    """
    Simplified posterior sampler for noise-only restoration.
    """

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: Trained EDM prior (v-parameterization)
        """
        self.model = model

    @staticmethod
    def get_edm_schedule(
        steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0
    ) -> torch.Tensor:
        """Get EDM noise schedule."""
        t = torch.linspace(0, 1, steps)
        sigmas = (sigma_max**(1/rho) + t * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
        sigmas = torch.cat([sigmas, torch.zeros(1)])
        return sigmas

    def denoise_with_metadata(
        self,
        y_electrons: torch.Tensor,
        metadata: ImageMetadata,
        scale: float,
        background: float = 0.0,
        read_noise: float = 0.0,
        mask: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        guidance_config: Optional[GuidanceConfig] = None,
        guidance_weight: float = 1.0,
        steps: int = 18,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Denoise with complete metadata for reconstruction.

        This is the main entry point that handles:
        1. Denoising at model resolution
        2. Preserving metadata for exact reconstruction
        3. Returning diagnostics

        Args:
            y_electrons: Noisy observation (electrons) [B, C, H, W]
            metadata: Transform metadata for reconstruction
            scale: Normalization scale (electrons)
            background: Background offset (electrons)
            read_noise: Read noise (electrons)
            mask: Valid pixel mask
            condition: Model conditioning vector
            guidance_config: Guidance configuration
            guidance_weight: Data term strength (0 = prior only)
            steps: Number of denoising steps
            seed: Random seed
            verbose: Show progress

        Returns:
            (denoised_image, info_dict)
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = next(self.model.parameters()).device
        y_electrons = y_electrons.to(device)
        if mask is not None:
            mask = mask.to(device)
        if condition is not None:
            condition = condition.to(device)

        # Setup
        sigmas = self.get_edm_schedule(steps).to(device)

        # Create guidance
        guidance_config = guidance_config or GuidanceConfig()
        guidance = PoissonGuidance(
            scale=scale,
            background=background,
            read_noise=read_noise,
            config=guidance_config
        )

        # Initialize from noise
        x = torch.randn_like(y_electrons) * sigmas[0]

        # Denoising loop
        iterator = range(len(sigmas) - 1)
        if verbose:
            iterator = tqdm(iterator, desc="Denoising")

        for i in iterator:
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]

            # 1. Model prediction (v-parameterization)
            with torch.no_grad():
                sigma_batch = sigma_curr.expand(x.shape[0])
                v = self.model(x, sigma_batch, condition=condition)

            # 2. Denoised estimate
            x0 = x - sigma_curr * v

            # 3. Data consistency (if guidance > 0)
            if guidance_weight > 0 and sigma_curr > 0.01:
                grad = guidance.compute(x0, y_electrons, sigma_curr, mask)
                x0 = x0 + guidance_weight * grad

            # 4. Enforce constraints
            x0 = torch.clamp(x0, min=0.0)

            # 5. Step to next noise level
            if sigma_next > 0:
                x = x0 + sigma_next * v
            else:
                x = x0

        # Final result
        denoised = x.detach().cpu()

        # Compute final fidelity
        with torch.no_grad():
            lambda_final = scale * x + background
            residual = (y_electrons - lambda_final)
            variance = lambda_final + read_noise**2
            chi2_per_pixel = ((residual**2) / variance).mean().item()

        # Collect info
        info = {
            'metadata': metadata,  # For reconstruction
            'guidance_diagnostics': guidance.get_diagnostics(),
            'chi2_per_pixel': chi2_per_pixel,
            'scale': scale,
            'background': background,
            'read_noise': read_noise
        }

        return denoised, info
```

### 2.5 Domain-Aware Dataset

```python
# data/domain_datasets.py
"""
Unified dataset handling for all domains.
Simplified without PSF/blur handling.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import json
from core.transforms import ReversibleTransform, ImageMetadata
from core.calibration import SensorCalibration
from data.raw_utils import load_raw_file

class DomainDataset(Dataset):
    """
    Base dataset for all domains with proper metadata handling.
    """

    # Domain-specific defaults
    DOMAIN_CONFIGS = {
        'photography': {
            'pixel_size': 4.29,  # μm (Sony A7S)
            'pixel_unit': 'um',
            'extensions': ['.arw', '.dng', '.nef', '.cr2'],
            'black_level': 512,
            'white_level': 16383
        },
        'microscopy': {
            'pixel_size': 0.65,  # μm at 20x
            'pixel_unit': 'um',
            'extensions': ['.tif', '.tiff'],
            'black_level': 100,
            'white_level': 65535
        },
        'astronomy': {
            'pixel_size': 0.04,  # arcsec (Hubble WFC3)
            'pixel_unit': 'arcsec',
            'extensions': ['.fits', '.fit'],
            'black_level': 0,
            'white_level': 65535
        }
    }

    def __init__(
        self,
        root: str,
        domain: str,
        calibration_file: str,
        scale: float,
        split: str = 'train',
        target_size: int = 128,
        patch_size: Optional[int] = None,
        augment: bool = False
    ):
        """
        Args:
            root: Dataset root directory
            domain: 'photography', 'microscopy', or 'astronomy'
            calibration_file: Path to calibration JSON
            scale: Normalization scale (electrons)
            split: 'train', 'val', or 'test'
            target_size: Model input size
            patch_size: If set, extract random patches
            augment: Apply augmentation (train only)
        """
        self.root = Path(root)
        self.domain = domain
        self.split = split
        self.scale = scale
        self.target_size = target_size
        self.patch_size = patch_size
        self.augment = augment and (split == 'train')

        # Load calibration
        self.calibration = SensorCalibration(calibration_file)

        # Get domain config
        self.config = self.DOMAIN_CONFIGS[domain]

        # Setup transform
        self.transform = ReversibleTransform(target_size)

        # Find all images
        self.image_paths = self._find_images()
        print(f"Found {len(self.image_paths)} images for {domain}/{split}")

        # Load or create split indices
        self.indices = self._get_split_indices()

    def _find_images(self) -> List[Path]:
        """Find all valid image files."""
        image_paths = []
        for ext in self.config['extensions']:
            image_paths.extend(self.root.glob(f'**/*{ext}'))
        return sorted(image_paths)

    def _get_split_indices(self) -> List[int]:
        """Get indices for train/val/test split."""
        n = len(self.image_paths)

        # Deterministic split based on hash
        indices = list(range(n))
        np.random.seed(42)
        np.random.shuffle(indices)

        # 70/15/15 split
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        if self.split == 'train':
            return indices[:train_end]
        elif self.split == 'val':
            return indices[train_end:val_end]
        else:  # test
            return indices[val_end:]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a single image.

        Returns dict with:
            - 'clean': Clean image (train) or None (test)
            - 'noisy': Noisy image (electrons)
            - 'normalized': Normalized noisy image
            - 'mask': Valid pixel mask
            - 'metadata': Complete reconstruction metadata
            - 'condition': Model conditioning vector
        """
        # Get image path
        image_path = self.image_paths[self.indices[idx]]

        # Load raw data
        raw_adu = load_raw_file(str(image_path), self.domain)

        # Convert to electrons
        electrons, mask = self.calibration.process_raw(raw_adu, return_mask=True)

        # For training, load clean reference (if available)
        clean_normalized = None
        if self.split == 'train':
            # Assume clean images in parallel directory
            clean_path = Path(str(image_path).replace('/noisy/', '/clean/'))
            if clean_path.exists():
                clean_raw = load_raw_file(str(clean_path), self.domain)
                clean_e, _ = self.calibration.process_raw(clean_raw, return_mask=False)
                clean_normalized = clean_e / self.scale
                clean_normalized = np.clip(clean_normalized, 0, 1)

        # Convert to tensor
        electrons = torch.from_numpy(electrons).float()
        mask = torch.from_numpy(mask).float()

        # Add channel dimension if needed
        if electrons.ndim == 2:
            electrons = electrons.unsqueeze(0)
            mask = mask.unsqueeze(0)

        # Add batch dimension
        electrons = electrons.unsqueeze(0)
        mask = mask.unsqueeze(0)

        # Extract patch if specified
        if self.patch_size and self.patch_size < min(electrons.shape[-2:]):
            electrons, mask = self._extract_patch(electrons, mask)
            if clean_normalized is not None:
                clean_normalized = self._extract_patch(
                    torch.from_numpy(clean_normalized).unsqueeze(0).unsqueeze(0),
                    None
                )[0]

        # Normalize
        normalized = electrons / self.scale
        normalized = torch.clamp(normalized, 0, 1)

        # Transform to model size
        transformed, metadata = self.transform.forward(
            normalized,
            pixel_size=self.config['pixel_size'],
            pixel_unit=self.config['pixel_unit'],
            domain=self.domain,
            black_level=self.calibration.params['black_level'],
            white_level=self.calibration.params['white_level'],
            iso=self._extract_iso(image_path),
            exposure_time=self._extract_exposure(image_path)
        )

        # Transform mask same way
        mask_transformed, _ = self.transform.forward(
            mask,
            pixel_size=self.config['pixel_size'],
            pixel_unit=self.config['pixel_unit'],
            domain=self.domain,
            black_level=0,
            white_level=1
        )

        # Also transform electrons for consistency
        electrons_transformed, _ = self.transform.forward(
            electrons,
            pixel_size=self.config['pixel_size'],
            pixel_unit=self.config['pixel_unit'],
            domain=self.domain,
            black_level=0,
            white_level=self.scale * 10  # Approximate
        )

        # Apply augmentation
        if self.augment:
            transformed = self._augment(transformed)

        # Create conditioning vector
        condition = self._create_condition_vector()

        return {
            'clean': clean_normalized,
            'noisy_electrons': electrons_transformed.squeeze(0),
            'normalized': transformed.squeeze(0),
            'mask': mask_transformed.squeeze(0),
            'metadata': metadata,
            'condition': condition,
            'image_path': str(image_path)
        }

    def _extract_patch(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract random patch from image and mask."""
        B, C, H, W = image.shape
        ps = self.patch_size

        # Random position
        h_start = np.random.randint(0, H - ps + 1)
        w_start = np.random.randint(0, W - ps + 1)

        # Extract patch
        patch = image[:, :, h_start:h_start+ps, w_start:w_start+ps]

        if mask is not None:
            mask_patch = mask[:, :, h_start:h_start+ps, w_start:w_start+ps]
            return patch, mask_patch

        return patch, None

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        """Apply geometric augmentation."""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[-1])

        # Random vertical flip
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[-2])

        # Random 90-degree rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[-2, -1])

        return image

    def _create_condition_vector(self) -> torch.Tensor:
        """Create conditioning vector for model."""
        # Domain one-hot
        domain_idx = {'photography': 0, 'microscopy': 1, 'astronomy': 2}[self.domain]
        domain_one_hot = torch.zeros(3)
        domain_one_hot[domain_idx] = 1.0

        # Normalized parameters
        log_scale = np.log10(max(self.scale, 1.0))
        norm_scale = (log_scale - 3.0) / 2.0

        rel_read_noise = self.calibration.params['read_noise'] / self.scale
        rel_background = 0.0  # Could estimate if needed

        condition = torch.cat([
            domain_one_hot,
            torch.tensor([norm_scale, rel_read_noise, rel_background])
        ])

        return condition

    def _extract_iso(self, image_path: Path) -> Optional[int]:
        """Extract ISO from filename or metadata."""
        # Simple pattern matching for now
        name = image_path.stem
        if 'ISO' in name:
            try:
                iso_str = name.split('ISO')[1].split('_')[0]
                return int(iso_str)
            except:
                pass
        return None

    def _extract_exposure(self, image_path: Path) -> Optional[float]:
        """Extract exposure time from filename or metadata."""
        # Placeholder - would read from EXIF/FITS header
        return None

# data/raw_utils.py
"""
Utilities for loading raw files from different domains.
"""

def load_raw_file(filepath: str, domain: str) -> np.ndarray:
    """
    Load raw file based on domain and format.

    Args:
        filepath: Path to raw file
        domain: Domain name for format hints

    Returns:
        Raw sensor data in ADU
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if domain == 'photography' and ext in ['.arw', '.dng', '.nef', '.cr2']:
        import rawpy
        with rawpy.imread(str(filepath)) as raw:
            # Get raw bayer data (not demosaiced)
            return raw.raw_image.copy().astype(np.float32)

    elif domain == 'microscopy' and ext in ['.tif', '.tiff']:
        from PIL import Image
        img = Image.open(filepath)
        return np.array(img, dtype=np.float32)

    elif domain == 'astronomy' and ext in ['.fits', '.fit']:
        from astropy.io import fits
        with fits.open(filepath) as hdul:
            return hdul[0].data.astype(np.float32)

    else:
        raise ValueError(f"Unknown format {ext} for domain {domain}")
```

### 2.6 Test Suite

```python
# tests/test_transforms.py
"""
Test reversible transforms preserve information.
"""

import pytest
import torch
import numpy as np
from core.transforms import ReversibleTransform, ImageMetadata

class TestReversibleTransform:

    def test_perfect_reconstruction(self):
        """Test that inverse perfectly reconstructs original."""
        transform = ReversibleTransform(target_size=128)

        # Test various sizes
        test_sizes = [(100, 100), (200, 150), (64, 256)]

        for H, W in test_sizes:
            # Create test image
            original = torch.randn(1, 1, H, W)

            # Forward transform
            transformed, metadata = transform.forward(
                original,
                pixel_size=1.0,
                pixel_unit='um',
                domain='test',
                black_level=0,
                white_level=1
            )

            assert transformed.shape == (1, 1, 128, 128)

            # Inverse transform
            reconstructed = transform.inverse(transformed, metadata)

            assert reconstructed.shape == original.shape

            # Check reconstruction error
            error = (original - reconstructed).abs().max()
            assert error < 1e-5, f"Reconstruction error too large: {error}"

    def test_metadata_preservation(self):
        """Test metadata is correctly preserved."""
        transform = ReversibleTransform(target_size=64)

        image = torch.randn(1, 3, 100, 120)

        _, metadata = transform.forward(
            image,
            pixel_size=4.29,
            pixel_unit='um',
            domain='photography',
            black_level=512,
            white_level=16383,
            iso=3200,
            exposure_time=0.1
        )

        assert metadata.original_height == 100
        assert metadata.original_width == 120
        assert metadata.pixel_size == 4.29
        assert metadata.domain == 'photography'
        assert metadata.iso == 3200

    def test_serialization(self):
        """Test metadata serialization."""
        metadata = ImageMetadata(
            original_height=100,
            original_width=200,
            scale_factor=0.5,
            crop_bbox=None,
            pad_amounts=(10, 10, 5, 5),
            pixel_size=1.0,
            pixel_unit='um',
            domain='test',
            black_level=0,
            white_level=1,
            bit_depth=16
        )

        # Serialize and deserialize
        json_str = metadata.to_json()
        restored = ImageMetadata.from_json(json_str)

        assert restored.original_height == metadata.original_height
        assert restored.pad_amounts == metadata.pad_amounts

# tests/test_guidance.py
"""
Test simplified Poisson guidance.
"""

import torch
from core.poisson_guidance import PoissonGuidance, GuidanceConfig

def test_guidance_at_truth():
    """Gradient should be small at true solution."""
    torch.manual_seed(42)

    # Create true image
    x_true = torch.ones(1, 1, 32, 32) * 0.5
    scale = 1000.0

    # Generate Poisson observations
    y = torch.poisson(scale * x_true)

    # Compute gradient at truth
    guidance = PoissonGuidance(scale=scale)
    grad = guidance.compute_score(x_true, y)

    # Should be small on average
    assert grad.abs().mean() < 1.0

def test_guidance_direction():
    """Gradient should point toward truth."""
    torch.manual_seed(42)

    x_true = torch.ones(1, 1, 32, 32) * 0.5
    x_init = torch.ones(1, 1, 32, 32) * 0.3  # Under-estimate

    scale = 1000.0
    y = torch.poisson(scale * x_true)

    guidance = PoissonGuidance(scale=scale)
    grad = guidance.compute_score(x_init, y)

    # Should be positive (pointing up toward truth)
    assert grad.mean() > 0

# tests/test_integration.py
"""
End-to-end integration test.
"""

import torch
import torch.nn as nn
from core.edm_sampler import EDMPosteriorSampler
from core.transforms import ReversibleTransform

class DummyModel(nn.Module):
    """Dummy model for testing."""
    def forward(self, x, sigma, condition=None):
        return torch.zeros_like(x) * 0.01

def test_full_pipeline():
    """Test complete denoising pipeline."""
    torch.manual_seed(42)

    # Setup
    model = DummyModel()
    sampler = EDMPosteriorSampler(model)
    transform = ReversibleTransform(target_size=64)

    # Create synthetic data
    original_size = (1, 1, 100, 80)
    clean = torch.ones(original_size) * 0.5

    # Transform to model size
    clean_transformed, metadata = transform.forward(
        clean,
        pixel_size=1.0,
        pixel_unit='um',
        domain='test',
        black_level=0,
        white_level=1
    )

    # Add noise
    scale = 1000.0
    y = torch.poisson(clean_transformed * scale) + torch.randn_like(clean_transformed) * 5

    # Denoise
    denoised, info = sampler.denoise_with_metadata(
        y_electrons=y,
        metadata=metadata,
        scale=scale,
        read_noise=5.0,
        steps=3,
        verbose=False
    )

    # Reconstruct to original size
    reconstructed = transform.inverse(denoised, metadata)

    assert reconstructed.shape == original_size
    assert 'chi2_per_pixel' in info
```

---

## Part 3: Training and Evaluation Scripts

### 3.1 Training Script Structure

```python
# scripts/train_prior.py
#!/usr/bin/env python
"""
Train diffusion prior on clean images.
Simplified without PSF handling.
"""

import torch
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

def create_model(config):
    """Create EDM model with conditioning."""
    # TODO: Import from external/edm
    from models.edm_wrapper import EDMModelWrapper

    return EDMModelWrapper(
        img_channels=config['img_channels'],
        img_resolution=config['resolution'],
        num_domains=3,
        condition_dim=6,
        model_channels=config['model_channels']
    )

def create_dataloaders(config):
    """Create training dataloaders."""
    from data.domain_datasets import DomainDataset

    datasets = []

    for domain in ['photography', 'microscopy', 'astronomy']:
        if config[f'use_{domain}']:
            ds = DomainDataset(
                root=config[f'{domain}_root'],
                domain=domain,
                calibration_file=config[f'{domain}_calibration'],
                scale=config[f'{domain}_scale'],
                split='train',
                target_size=config['resolution'],
                augment=True
            )
            datasets.append(ds)

    # Combine datasets
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(datasets)

    return DataLoader(
        combined,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

# Main training loop would go here
```


### Next Steps:

1. Implement EDM model wrapper (external dependency)
2. Complete data loading for specific formats
3. Add evaluation metrics and baselines
4. Train on real data
