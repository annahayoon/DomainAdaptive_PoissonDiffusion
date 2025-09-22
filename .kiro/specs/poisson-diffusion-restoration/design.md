# Improved Design Document

## Overview

The Domain-Adaptive Poisson-Gaussian Diffusion system implements a unified framework for low-light image restoration across photography, microscopy, and astronomy. The architecture combines three key innovations:

1. **Physically-Correct Noise Modeling**: Exact Poisson-Gaussian likelihood without approximations
2. **Unified Cross-Domain Model**: Single model with domain conditioning instead of separate models
3. **Perfect Reconstruction Pipeline**: Reversible transforms preserving all metadata and calibration

The system explicitly focuses on **noise-only restoration** (identity forward operator) without modeling PSF or motion blur, simplifying implementation while maintaining the critical physics of photon noise.

## System Architecture

### Conceptual Data Flow

```
Input Pipeline:
RAW File → Calibration → Electrons → Normalize[0,1] → Transform(128×128) → Add Metadata

Processing Pipeline:
Noisy(128×128) → Domain Conditioning → EDM Diffusion → Physics Guidance → Denoised(128×128)

Output Pipeline:
Denoised(128×128) → Inverse Transform → Original Size → Denormalize → Final Output
                                    ↑
                              [Metadata]
```

### Key Design Decisions

1. **Separation of Spaces**:
   - Model operates in normalized [0,1] space for stability
   - Physics computed in electron space for correctness
   - Clear conversion via scale parameter s

2. **Multi-Resolution Architecture**:
   - Progressive growing: start with 32×32, grow to 64×64, 96×96, 128×128
   - Hierarchical multi-scale processing with skip connections
   - Adaptive resolution selection based on image characteristics
   - Enables optimal quality-efficiency tradeoffs

3. **No Forward Operator**:
   - Simplified to noise-only (H = Identity)
   - No PSF/blur modeling in this version
   - Focus on getting noise physics correct first

## Components and Interfaces

### 1. Reversible Transform System

**Purpose**: Handle arbitrary input sizes while preserving all information needed for perfect reconstruction.

**Core Classes**:
- `ReversibleTransform`: Main transform class with forward/inverse operations
- `ImageMetadata`: Complete metadata container for reconstruction
- `TransformPipeline`: Orchestrates multi-step transformations

**Key Interfaces**:
```python
class ReversibleTransform:
    def forward(self, image: Tensor, **metadata) -> Tuple[Tensor, ImageMetadata]
    def inverse(self, image: Tensor, metadata: ImageMetadata) -> Tensor

@dataclass
class ImageMetadata:
    original_height: int
    original_width: int
    scale_factor: float
    crop_bbox: Optional[Tuple[int, int, int, int]]
    pad_amounts: Optional[Tuple[int, int, int, int]]
    pixel_size: float
    pixel_unit: str
    domain: str
    # ... additional calibration and acquisition parameters
```

**Design Decisions**:
- Use bilinear interpolation for scaling to preserve smooth gradients
- Apply reflection padding to avoid boundary artifacts
- Store complete transformation chain for perfect invertibility
- Support serialization to JSON for reproducibility

### 2. Physics-Aware Guidance System

**Purpose**: Incorporate exact Poisson-Gaussian likelihood into diffusion sampling.

**Core Classes**:
- `PoissonGuidance`: Computes likelihood gradients and guidance weights
- `GuidanceConfig`: Configuration for different guidance modes
- `CalibrationManager`: Handles sensor-specific parameters

**Key Interfaces**:
```python
class PoissonGuidance:
    def compute_score(self, x_hat: Tensor, y_electrons: Tensor, mask: Tensor) -> Tensor
    def gamma_schedule(self, sigma: Tensor) -> Tensor
    def compute(self, x_hat: Tensor, y_electrons: Tensor, sigma_t: Tensor) -> Tensor

@dataclass
class GuidanceConfig:
    mode: Literal['wls', 'exact'] = 'wls'
    gamma_schedule: Literal['sigma2', 'linear', 'const'] = 'sigma2'
    kappa: float = 0.5
    gradient_clip: float = 10.0
```

**Mathematical Foundation**:
- **WLS Mode**: `∇ log p(y|x) = s·(y - λ)/(λ + σ_r²)` for computational efficiency
- **Exact Mode**: Includes variance derivative terms for theoretical completeness
- **Guidance Weighting**: `γ(σ) = κ·σ²` to balance prior and likelihood terms
- **Gradient Clipping**: Prevents numerical instability in extreme cases

### 3. Domain-Conditioned Diffusion Model

**Purpose**: Single model that adapts to different imaging domains through conditioning.

**Core Classes**:
- `EDMModelWrapper`: Wraps external EDM implementation with conditioning
- `DomainEncoder`: Creates conditioning vectors from metadata
- `ConditionalSampler`: Handles conditioned sampling process

**Key Interfaces**:
```python
class EDMModelWrapper(nn.Module):
    def forward(self, x: Tensor, sigma: Tensor, condition: Tensor) -> Tensor

class DomainEncoder:
    def encode(self, domain: str, scale: float, read_noise: float, **kwargs) -> Tensor

class ConditionalSampler:
    def sample(self, y_electrons: Tensor, metadata: ImageMetadata,
               guidance_config: GuidanceConfig) -> Tuple[Tensor, Dict]
```

**Conditioning Strategy**:
- Domain one-hot encoding (3 dimensions for photography/microscopy/astronomy)
- Normalized scale parameter (log10 scale normalized to [-1, 1])
- Relative noise parameters (read_noise/scale, background/scale)
- Total conditioning dimension: 6 (3 + 1 + 2)

### 4. Multi-Format Data Loading System

**Purpose**: Unified interface for loading and calibrating data from different domains.

**Core Classes**:
- `DomainDataset`: Base dataset class with domain-specific handling
- `SensorCalibration`: Manages calibration parameters and conversions
- `FormatLoaders`: Format-specific loading utilities

**Key Interfaces**:
```python
class DomainDataset(Dataset):
    def __init__(self, root: str, domain: str, calibration_file: str, scale: float)
    def __getitem__(self, idx: int) -> Dict[str, Any]

class SensorCalibration:
    def process_raw(self, raw_adu: ndarray) -> Tuple[ndarray, ndarray]
    def adu_to_electrons(self, adu: ndarray) -> ndarray
```

**Format Support**:
- **Photography**: RAW formats (.arw, .dng, .nef, .cr2) via rawpy
- **Microscopy**: TIFF formats (.tif, .tiff) with proper bit depth
- **Astronomy**: FITS format (.fits, .fit) with header metadata

### 5. Training and Evaluation Framework

**Purpose**: Comprehensive tools for model training, evaluation, and comparison.

**Core Classes**:
- `MultiDomainTrainer`: Handles combined dataset training
- `EvaluationSuite`: Comprehensive metric computation
- `BaselineComparator`: Standardized baseline comparisons

**Key Interfaces**:
```python
class MultiDomainTrainer:
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]
    def validate(self, val_loaders: Dict[str, DataLoader]) -> Dict[str, float]

class EvaluationSuite:
    def compute_metrics(self, pred: Tensor, target: Tensor) -> Dict[str, float]
    def physics_metrics(self, pred: Tensor, noisy: Tensor, scale: float) -> Dict[str, float]
```

## Data Models

### Image Data Flow

```mermaid
graph LR
    A[Raw ADU] --> B[Electrons]
    B --> C[Normalized [0,1]]
    C --> D[Model Size 128x128]
    D --> E[Denoised 128x128]
    E --> F[Original Size]
    F --> G[Final Electrons]

    subgraph "Metadata Tracking"
        H[Transform Metadata]
        I[Calibration Data]
        J[Domain Info]
    end

    D -.-> H
    B -.-> I
    C -.-> J
```

### Calibration Data Structure

```python
@dataclass
class CalibrationParams:
    black_level: float          # ADU offset
    white_level: float          # ADU saturation
    gain: float                 # electrons/ADU
    read_noise: float           # electrons RMS
    dark_current: float         # electrons/sec
    quantum_efficiency: float   # [0,1]
    pixel_size: float          # physical size
    pixel_unit: str            # 'um' or 'arcsec'
```

### Training Data Structure

```python
@dataclass
class TrainingBatch:
    clean: Optional[Tensor]     # Ground truth (if available)
    noisy_electrons: Tensor     # Noisy observations (electrons)
    normalized: Tensor          # Normalized for model [0,1]
    mask: Tensor               # Valid pixel mask
    condition: Tensor          # Domain conditioning vector
    metadata: ImageMetadata    # Complete reconstruction info
```

## Error Handling

### Numerical Stability

1. **Gradient Clipping**: Limit guidance gradients to prevent divergence
2. **Variance Regularization**: Add small epsilon to prevent division by zero
3. **Range Clamping**: Ensure predictions stay in valid physical range [0, ∞)
4. **Overflow Protection**: Monitor for NaN/Inf values during sampling

### Data Validation

1. **Format Validation**: Verify file formats match expected domain
2. **Calibration Validation**: Check calibration parameters are physically reasonable
3. **Metadata Consistency**: Ensure transform metadata enables perfect reconstruction
4. **Mask Validation**: Verify masks correctly identify valid pixels

### Graceful Degradation

1. **Missing Calibration**: Fall back to default parameters with warnings
2. **Corrupted Files**: Skip problematic files with detailed logging
3. **Memory Constraints**: Support patch-based processing for large images
4. **GPU Unavailability**: Automatic fallback to CPU processing

## Testing Strategy

### Unit Testing

1. **Transform Reversibility**: Verify perfect reconstruction across all sizes
2. **Physics Accuracy**: Test guidance gradients against analytical solutions
3. **Calibration Correctness**: Validate ADU to electron conversions
4. **Metadata Serialization**: Test JSON export/import functionality

### Integration Testing

1. **End-to-End Pipeline**: Complete workflow from raw files to restored images
2. **Multi-Domain Training**: Verify balanced training across domains
3. **Cross-Platform Compatibility**: Test on different operating systems
4. **Memory Efficiency**: Profile memory usage with large datasets

### Physics Validation

1. **Synthetic Data**: Generate known Poisson-Gaussian data for validation
2. **Chi-Squared Analysis**: Verify statistical consistency of results
3. **Noise Characterization**: Compare estimated vs. true noise parameters
4. **Convergence Analysis**: Monitor sampling convergence properties

### Performance Testing

1. **Inference Speed**: Benchmark processing time vs. image size
2. **Memory Usage**: Profile peak memory consumption
3. **Scalability**: Test with varying batch sizes and resolutions
4. **GPU Utilization**: Monitor compute efficiency during training/inference

## Multi-Resolution System Components

### 5. Progressive Growing Architecture

**Purpose**: Enable training and inference at multiple resolutions with progressive quality improvement.

**Core Classes**:
- `ProgressiveEDM`: Main progressive growing model
- `ResolutionScheduler`: Manages resolution transitions during training
- `MultiResDataset`: Handles data loading at multiple resolutions
- `AdaptiveResolution`: Automatic resolution selection for inference

**Key Interfaces**:
```python
class ProgressiveEDM(nn.Module):
    def __init__(self, min_res: int = 64, max_res: int = 512, num_stages: int = 4)
    def forward(self, x: Tensor, sigma: Tensor, resolution: int) -> Tensor
    def grow_resolution(self) -> bool  # Add new resolution stage

class ResolutionScheduler:
    def __init__(self, schedule: List[Tuple[int, int]])  # (epoch, resolution)
    def get_current_resolution(self, epoch: int) -> int
    def should_grow_resolution(self, epoch: int) -> bool

class AdaptiveResolution:
    def select_resolution(self, image: Tensor, target_quality: str) -> int
    def get_batch_size_for_resolution(self, resolution: int) -> int
```

**Progressive Growing Strategy**:
- **Stage 1**: 32×32 (epochs 0-25) - Fast training, basic features
- **Stage 2**: 64×64 (epochs 26-50) - Add mid-level features
- **Stage 3**: 96×96 (epochs 51-75) - Add fine details
- **Stage 4**: 128×128 (epochs 76-100) - Full resolution processing

### 6. Hierarchical Multi-Scale Processing

**Purpose**: Process images at multiple scales simultaneously for better feature extraction.

**Core Classes**:
- `MultiScaleEDM`: U-Net style hierarchical processing
- `ScalePyramid`: Creates and manages multi-scale image pyramids
- `FeatureFusion`: Combines features across different scales
- `MultiScaleConditioning`: Scale-aware conditioning vectors

**Key Interfaces**:
```python
class MultiScaleEDM(nn.Module):
    def __init__(self, scales: List[int] = [32, 64, 96, 128])
    def forward(self, x: Tensor, sigma: Tensor, scale_mask: Tensor) -> Tensor
    def extract_multiscale_features(self, x: Tensor) -> Dict[int, Tensor]

class ScalePyramid:
    def build_pyramid(self, x: Tensor) -> Dict[int, Tensor]
    def reconstruct_from_pyramid(self, pyramid: Dict[int, Tensor]) -> Tensor

class FeatureFusion:
    def fuse_features(self, features: Dict[int, Tensor]) -> Tensor
    def upsample_with_skip(self, x: Tensor, skip: Tensor) -> Tensor
```

**Multi-Scale Feature Processing**:
- **Encoder**: Downsample through scales (128→96→64→32)
- **Decoder**: Upsample with skip connections (32→64→96→128)
- **Feature Fusion**: Combine features from all scales at each level
- **Scale-Aware Conditioning**: Different conditioning for different scales

### 7. Adaptive Resolution Pipeline

**Purpose**: Automatically select optimal resolution based on image characteristics and computational constraints.

**Core Classes**:
- `ResolutionManager`: Orchestrates resolution selection and processing
- `QualityEstimator`: Estimates expected quality improvement per resolution
- `MemoryPlanner`: Plans memory usage across different resolutions
- `BatchScheduler`: Dynamic batch sizing for different resolutions

**Key Interfaces**:
```python
class ResolutionManager:
    def process_adaptive(self, images: List[Tensor], constraints: Dict) -> List[Tensor]
    def select_optimal_resolution(self, image: Tensor, constraints: Dict) -> int
    def get_processing_plan(self, images: List[Tensor]) -> ProcessingPlan

@dataclass
class ProcessingConstraints:
    max_memory_gb: float = 8.0
    target_time_sec: float = 30.0
    min_resolution: int = 64
    max_resolution: int = 512
    quality_preference: str = "balanced"  # speed, balanced, quality
```

**Adaptive Processing Algorithm**:
1. **Analyze Input**: Estimate noise level, detail content, size
2. **Evaluate Options**: Test different resolutions for quality/speed tradeoff
3. **Select Optimal**: Choose resolution maximizing quality within constraints
4. **Process Efficiently**: Use hierarchical processing when beneficial

## Implementation Priorities

### Phase 1: Core Infrastructure (Weeks 1-2)
- Reversible transforms with metadata preservation
- Basic Poisson guidance implementation
- Simple data loading for one domain
- Unit tests for core components

### Phase 2: Model Integration (Weeks 3-4)
- EDM model wrapper with conditioning
- Complete sampling pipeline
- Multi-domain dataset handling
- Integration testing

### Phase 3: Training Framework (Weeks 5-6)
- Multi-domain training pipeline
- Comprehensive evaluation metrics
- Baseline implementations
- Performance optimization

### Phase 4: Multi-Resolution Implementation (Weeks 7-8)
- Progressive growing architecture
- Hierarchical multi-scale processing
- Adaptive resolution pipeline
- Multi-resolution training framework

### Phase 5: Validation and Polish (Weeks 9-10)
- Scientific validation on real data
- Documentation and examples
- Performance benchmarking
- Final testing and bug fixes

## Dependencies and External Components

### Core Dependencies
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Pillow/OpenCV**: Image processing
- **rawpy**: RAW image loading
- **astropy**: FITS file handling
- **tqdm**: Progress bars

### External Models
- **EDM Implementation**: Will integrate existing EDM codebase
- **Baseline Models**: DnCNN, NAFNet implementations for comparison
- **Classical Methods**: BM3D, Richardson-Lucy implementations

### Hardware Requirements
- **Training**: 4x A100 GPUs (40GB VRAM each)
- **Inference**: Single GPU with 8GB+ VRAM
- **Storage**: 500GB for datasets and models
- **Memory**: 64GB+ RAM for large image processing
