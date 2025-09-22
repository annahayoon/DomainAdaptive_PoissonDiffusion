# Requirements Document

## Introduction

This project implements a Domain-Adaptive Poisson-Gaussian Diffusion model for low-light image restoration across multiple scientific and consumer domains (photography, microscopy, astronomy). The system addresses the fundamental challenge that current deep learning methods fail in extreme low-light conditions because they assume Gaussian noise, while photon arrival is fundamentally Poisson-distributed. At photon counts below 100 per pixel, this distinction becomes critical for accurate restoration.

The solution combines physically-correct likelihood guidance with diffusion models to achieve both mathematical correctness and practical performance. This implementation focuses on **noise-only restoration** (identity forward operator) without PSF or motion blur modeling, simplifying the problem while maintaining physical accuracy.

## Core Design Principles

1. **Physical Correctness**: Exact Poisson-Gaussian noise modeling with proper electron-space calculations
2. **Domain Generalization**: Single model adapts to multiple imaging modalities through conditioning
3. **Perfect Reconstruction**: Lossless reversible transforms preserve all metadata and dimensions
4. **Practical Deployment**: Efficient inference suitable for real-world applications

## Requirements

### Requirement 1: Physics-Aware Noise Modeling

**User Story:** As a researcher working with low-light images, I want the system to correctly model Poisson-Gaussian noise statistics, so that the restoration preserves the physical accuracy required for quantitative analysis.

#### Acceptance Criteria

1. **Photon Statistics**: WHEN the system processes images with <100 photons per pixel THEN it SHALL use exact Poisson noise modeling instead of Gaussian approximation
2. **Likelihood Computation**: WHEN computing likelihood gradients THEN the system SHALL use the formula: `∇ log p(y|x) = s·(y - λ)/(λ + σ_r²)` where `λ = s·x + b` (s=scale, b=background, σ_r=read noise)
3. **Normalization Separation**: The system SHALL maintain clear separation between normalized intensity values [0,1] for the model and physical electron counts for likelihood computation
4. **Statistical Consistency**: WHEN evaluating restoration quality THEN the system SHALL achieve χ² values within [0.9, 1.1] on calibrated data
5. **Automatic Adaptation**: WHEN processing bright regions (>1000 photons) THEN the system SHALL automatically transition to near-Gaussian behavior (σ ≈ √signal)
6. **Calibration Handling**: WHEN given sensor calibration THEN the system SHALL correctly convert between ADU and electrons using: `electrons = (ADU - black_level) × gain`
7. **Model Simplification**: For this implementation, the system SHALL model noise-only restoration without PSF or motion blur (identity forward operator)

#### Required Calibration Parameters
- Gain (e⁻/ADU): Conversion factor from ADU to electrons
- Read noise (e⁻): Gaussian noise standard deviation
- Black level (ADU): Sensor offset
- White level (ADU): Saturation point
- Optional: Dark current (e⁻/sec), Quantum efficiency

### Requirement 2: Cross-Domain Generalization

**User Story:** As a user working across different imaging domains, I want a single model that adapts to photography, microscopy, and astronomy data without requiring separate models for each application.

#### Acceptance Criteria

1. **Domain Conditioning**: WHEN training on combined datasets THEN the system SHALL use 6-dimensional conditioning vectors: [domain_one_hot(3), log_scale_norm(1), rel_read_noise(1), rel_background(1)]
2. **Performance Parity**: WHEN processing images from any supported domain THEN the unified model SHALL perform within 0.5 dB PSNR of domain-specific models
3. **Metadata Preservation**: WHEN switching between domains THEN the system SHALL preserve domain-specific physical units (μm for microscopy, arcseconds for astronomy, pixels for photography)
4. **Domain Extension**: WHEN adding a new domain THEN the system SHALL support configuration-based extension without full model retraining
5. **Balanced Training**: WHEN training on multiple domains THEN the system SHALL implement proper dataset balancing to prevent domain bias

#### Supported Domains
| Domain | Typical Photons | Pixel Size | Bit Depth | Key Challenge |
|--------|----------------|------------|-----------|---------------|
| Photography | 10-10,000 | 1-6 μm | 12-14 bit | High ISO noise |
| Microscopy | 1-1,000 | 0.1-1 μm | 12-16 bit | Photobleaching |
| Astronomy | 0.1-100 | 0.01-1" | 16-32 bit | Cosmic rays |

### Requirement 3: Multi-Resolution Processing and Metadata Preservation

**User Story:** As a scientist requiring quantitative analysis, I want the system to process images at optimal resolutions while preserving all physical calibration data and enabling perfect reconstruction.

#### Acceptance Criteria

1. **Progressive Resolution Training**: WHEN training the model THEN the system SHALL support progressive growing from 64×64 to 512×512 resolution in stages
2. **Adaptive Resolution Selection**: WHEN processing images THEN the system SHALL automatically select optimal resolution based on image characteristics and computational constraints
3. **Hierarchical Multi-Scale Processing**: WHEN processing high-resolution images THEN the system SHALL use U-Net style hierarchical processing with skip connections across scales
4. **Perfect Reconstruction**: WHEN reconstructing processed images THEN the system SHALL restore exact original dimensions with <1e-5 relative error
5. **Metadata Tracking**: The system SHALL track complete transformation metadata including: scale_factor, crop_bbox, pad_amounts, original dimensions, processing resolution
6. **Physical Units**: WHEN handling different pixel scales THEN the system SHALL preserve and correctly transform physical units across all resolution levels
7. **Serialization Support**: WHEN saving/loading metadata THEN the system SHALL support JSON serialization for reproducibility
8. **Boundary Handling**: WHEN padding is required THEN the system SHALL use reflection padding to minimize boundary artifacts at all resolution levels

#### Multi-Resolution Specifications
- **Supported Resolutions**: 32×32, 64×64, 96×96, 128×128
- **Progressive Growing**: 4-stage training (25 epochs per stage)
- **Adaptive Selection**: Automatic resolution choice based on noise level and detail content
- **Memory Management**: Dynamic batch sizing per resolution level
- **Quality Scaling**: PSNR improvement of 1-2 dB per resolution stage

#### Metadata Structure
```python
@dataclass
class ImageMetadata:
    # Dimensions
    original_height: int
    original_width: int
    # Transformations
    scale_factor: float
    crop_bbox: Optional[Tuple[int, int, int, int]]
    pad_amounts: Optional[Tuple[int, int, int, int]]
    # Physical calibration
    pixel_size: float
    pixel_unit: str
    black_level: float
    white_level: float
    # Domain info
    domain: str
    bit_depth: int
    # Optional acquisition
    iso: Optional[int]
    exposure_time: Optional[float]
```

### Requirement 4: Multi-Resolution Diffusion Model Integration

**User Story:** As a developer implementing the restoration pipeline, I want a multi-resolution diffusion model that can process images at optimal resolutions while incorporating physics-based guidance for optimal restoration quality.

#### Acceptance Criteria

1. **Progressive Growing Architecture**: WHEN training the model THEN the system SHALL support progressive growing from low to high resolution in discrete stages
2. **Multi-Scale Feature Processing**: WHEN processing images THEN the system SHALL use hierarchical feature extraction with skip connections across resolution levels
3. **Adaptive Resolution Selection**: WHEN inferring optimal processing THEN the system SHALL automatically select resolution based on image characteristics and computational constraints
4. **Model Architecture**: WHEN implementing the diffusion model THEN the system SHALL use EDM v-parameterization with resolution-conditioned architecture
5. **Guidance Integration**: WHEN performing guided sampling THEN the system SHALL incorporate Poisson-Gaussian likelihood gradients at each denoising step where σ > 0.01, with resolution-aware weighting
6. **Guidance Scheduling**: WHEN computing guidance weights THEN the system SHALL support configurable schedules: σ² (default), linear, or constant, with resolution-specific parameters
7. **Sampling Algorithm**: The system SHALL implement the EDM sampling schedule with 18 steps by default (configurable 5-50), supporting variable steps per resolution
8. **Stability Control**: WHEN guidance gradients exceed threshold THEN the system SHALL clip values to [-10, 10] for numerical stability across all resolution levels
9. **Prior-Only Mode**: WHEN guidance_weight=0 THEN the system SHALL fall back to pure prior sampling at the selected resolution

#### Multi-Resolution Specifications
- **Resolution Stages**: 32×32 → 64×64 → 96×96 → 128×128
- **Progressive Training**: 25 epochs per resolution stage
- **Feature Hierarchy**: Skip connections and feature fusion across scales
- **Memory Efficiency**: Dynamic batch sizing based on resolution
- **Quality Scaling**: 1-2 dB PSNR improvement per resolution stage

#### EDM Sampling Parameters
- Default steps: 18
- σ_min: 0.002
- σ_max: 80.0
- ρ: 7.0 (schedule curvature)

### Requirement 5: Multi-Format Data Loading

**User Story:** As a researcher with diverse data sources, I want the system to automatically load and calibrate raw files from different instruments.

#### Acceptance Criteria

1. **RAW Photography**: WHEN loading photography data THEN the system SHALL support formats (.arw, .dng, .nef, .cr2) using rawpy library
2. **Microscopy TIFF**: WHEN loading microscopy data THEN the system SHALL support 8/12/16-bit TIFF with proper scaling
3. **Astronomy FITS**: WHEN loading astronomy data THEN the system SHALL support FITS format with header metadata extraction
4. **Bayer Handling**: WHEN processing RAW camera data THEN the system SHALL either process Bayer patterns directly or apply appropriate demosaicing
5. **Calibration Application**: WHEN processing any raw format THEN the system SHALL apply sensor-specific calibration (dark frame, flat field, bias)
6. **Invalid Pixel Masking**: WHEN encountering saturated/dead pixels THEN the system SHALL create appropriate masks for guidance exclusion
7. **Memory Efficiency**: WHEN loading large files THEN the system SHALL support memory-mapped loading and patch extraction

### Requirement 6: Training and Evaluation Framework

**User Story:** As a machine learning researcher, I want comprehensive training and evaluation tools for reproducible research.

#### Acceptance Criteria

1. **Unsupervised Training**: WHEN training the diffusion prior THEN the system SHALL require only unpaired clean images (no noise pairs needed)
2. **Multi-Domain Support**: WHEN training on combined datasets THEN the system SHALL support weighted sampling across domains
3. **Standard Metrics**: WHEN evaluating results THEN the system SHALL compute: PSNR, SSIM, LPIPS, MS-SSIM
4. **Physics Metrics**: WHEN validating physics THEN the system SHALL compute: χ² consistency, residual whiteness, photon transfer curve
5. **Baseline Comparison**: WHEN comparing methods THEN the system SHALL include: BM3D, Anscombe+BM3D, DnCNN, NAFNet, Noise2Void
6. **Ablation Studies**: The system SHALL support configurable ablations: noise model (Poisson vs Gaussian), guidance mode (WLS vs exact), conditioning (with/without)

#### Validation Metrics
- Bias: < 1% of signal level
- Variance: Within 5% of theoretical prediction
- Residuals: White noise (no structure)
- χ²: Within [0.9, 1.1] for calibrated data

### Requirement 7: Performance and Scalability

**User Story:** As a user processing large datasets, I want efficient inference that scales to high-resolution images.

#### Acceptance Criteria

1. **Inference Speed**: WHEN processing megapixel images THEN the system SHALL complete 18-step inference within 10-20 seconds on single V100/A100 GPU
2. **Batch Processing**: WHEN processing multiple images THEN the system SHALL support efficient batching with automatic batch size optimization
3. **Memory Management**: WHEN handling 4K+ images THEN the system SHALL implement tiled/patch processing to fit in 8GB GPU memory
4. **Multi-GPU Training**: WHEN training on large datasets THEN the system SHALL support distributed data parallel training
5. **CPU Fallback**: WHEN GPU is unavailable THEN the system SHALL automatically fall back to CPU processing with appropriate warnings
6. **Progressive Processing**: WHEN dealing with very large images THEN the system SHALL support progressive refinement from coarse to fine

#### Multi-Resolution Performance Targets
| Input Size | Processing Resolution | GPU Memory | Inference Time | Quality Level |
|------------|---------------------|------------|----------------|---------------|
| 512×512 | 128×128 | 2 GB | 1-2 sec | High |
| 1024×1024 | 128×128 | 2 GB | 2-4 sec | High |
| 1024×1024 | 256×256 | 4 GB | 4-8 sec | Very High |
| 2048×2048 | 128×128 | 2 GB | 4-8 sec | High |
| 2048×2048 | 256×256 | 4 GB | 8-15 sec | Very High |
| 4096×4096 | 128×128 | 2 GB | 8-15 sec | High |
| 4096×4096 | 256×256 | 4 GB | 15-30 sec | Very High |
| 4096×4096 | 512×512 | 8 GB | 30-60 sec | Maximum |

#### Adaptive Processing Features
- **Automatic Resolution Selection**: Choose optimal resolution based on available memory and quality requirements
- **Progressive Refinement**: Option to start at low resolution and progressively refine
- **Batch Size Adaptation**: Dynamic batch sizing based on resolution and available memory
- **Quality-Efficiency Tradeoff**: Configurable balance between quality and speed

### Requirement 8: Scientific Validation and Reproducibility

**User Story:** As a scientist publishing research, I want complete reproducibility and validation tools.

#### Acceptance Criteria

1. **Code Release**: WHEN publishing results THEN the system SHALL provide complete source code with MIT/Apache license
2. **Model Checkpoints**: WHEN distributing models THEN the system SHALL include pre-trained weights for all three domains
3. **Synthetic Validation**: WHEN validating physics THEN the system SHALL generate synthetic data with exact Poisson-Gaussian statistics
4. **Benchmark Datasets**: WHEN evaluating methods THEN the system SHALL provide standardized train/val/test splits with checksums
5. **Deterministic Mode**: WHEN reproducibility is required THEN the system SHALL support fully deterministic execution with fixed seeds
6. **Documentation**: The system SHALL include comprehensive documentation: API reference, tutorials, domain-specific examples

#### Reproducibility Checklist
- [ ] Fixed random seeds for data loading
- [ ] Deterministic GPU operations
- [ ] Version-locked dependencies
- [ ] Dataset checksums
- [ ] Training logs and configs
- [ ] Evaluation protocols

### Requirement 9: Error Handling and Robustness

**User Story:** As a user working with imperfect data, I want the system to handle edge cases gracefully.

#### Acceptance Criteria

1. **Missing Calibration**: WHEN calibration data is unavailable THEN the system SHALL use domain-specific defaults with warnings
2. **Extreme Underexposure**: WHEN average photon count <1 THEN the system SHALL gracefully degrade to prior-only reconstruction
3. **Numerical Stability**: WHEN encountering NaN/Inf values THEN the system SHALL detect and handle appropriately with informative errors
4. **Corrupted Files**: WHEN loading fails THEN the system SHALL log detailed errors and continue processing remaining files
5. **Out-of-Distribution**: WHEN input statistics differ significantly from training THEN the system SHALL issue warnings about potential quality degradation
