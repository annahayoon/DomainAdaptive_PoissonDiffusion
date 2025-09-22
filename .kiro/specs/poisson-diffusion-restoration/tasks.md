# Implementation Plan

**Project Goal**: Implement physics-correct Poisson-Gaussian diffusion for low-light image denoising across three domains (photography, microscopy, astronomy).

**Current Phase**: 8.4 (Multi-Resolution Implementation - ✅ COMPLETE)

**Key Constraints**:
- Must preserve exact image dimensions through reversible transforms
- Physics must use electron space, model uses [0,1] normalized space
- No PSF/blur modeling in this version (identity forward operator only)

**Testing Philosophy**: Write tests immediately after implementing each component, not later.

## Phase 0: Prerequisites & Dependencies

- [x] 0.1 Set up external dependencies
  - Clone and integrate EDM codebase from external repository
  - Verify EDM model can be imported and runs with basic test
  - Document any modifications needed for integration with conditioning
  - _Blocker for: 3.1_

- [x] 0.2 Set up development environment
  - Create venv environment with PyTorch, CUDA support
  - Install domain-specific libraries (rawpy, astropy, pillow, tqdm)
  - Set up pre-commit hooks and code formatting (black, isort)
  - _Requirements: Infrastructure_

## Phase 1: Core Infrastructure (Week 1)

- [x] 1.1 Project structure and base interfaces
  - Create directory structure for core, models, data, configs, scripts, tests
  - Define core interfaces and abstract base classes
  - Set up setup.py, requirements.txt, and basic CI/CD configuration
  - _Requirements: 8.1, 8.5_

- [x] 1.2 Implement error handling framework
  - Create error handling utilities and custom exceptions for domain-specific errors
  - Implement numerical stability handlers (NaN/Inf detection, gradient clipping)
  - Add logging system with appropriate levels for debugging and monitoring
  - _Requirements: 9.1, 9.3, 9.4_

- [x] 1.3 Implement reversible transforms with tests
  - Create ImageMetadata dataclass with JSON serialization/deserialization
  - Implement ReversibleTransform with forward/inverse operations using bilinear interpolation
  - Add support for reflection padding and center cropping/padding to square
  - **Immediately write unit tests for perfect reconstruction across various sizes**
  - _Requirements: 3.1-3.6_

## Phase 2: Physics Foundation (Week 1-2)

- [x] 2.1 Implement calibration system
  - Create SensorCalibration class with parameter loading from JSON
  - Implement ADU ↔ electron conversions: electrons = (ADU - black_level) × gain
  - Add calibration file validation and domain-specific default fallbacks
  - **Write tests for conversion accuracy and edge cases**
  - _Requirements: 1.6, 5.5_

- [x] 2.2 Implement Poisson-Gaussian guidance
  - Create PoissonGuidance class with WLS and exact likelihood modes
  - Implement gamma scheduling options (sigma2, linear, constant)
  - Add numerical stability measures (gradient clipping, variance regularization)
  - **Write physics validation tests immediately using synthetic data**
  - _Requirements: 1.1-1.4, 4.5_

- [x] 2.2.1 **Validation Checkpoint**: Synthetic data test
  - Generate 100 synthetic images with known Poisson-Gaussian noise
  - Verify χ² = 1.0 ± 0.1 for gradient computation
  - Confirm bias < 1% of signal level
  - _Gate: Don't proceed until physics validation passes_

- [x] 2.3 Create synthetic data generator
  - Implement exact Poisson-Gaussian noise generation with known statistics
  - Create test patterns with various photon levels for validation
  - Add ground truth generation for chi-squared and bias testing
  - **Use immediately for validating guidance computation**
  - _Requirements: 8.3_

## Phase 3: Model Integration (Week 2-3)

- [x] 3.1 Create EDM wrapper with conditioning
  - Implement EDMModelWrapper that integrates external EDM codebase
  - Add domain conditioning architecture (FiLM or cross-attention)
  - Create model initialization utilities and factory functions
  - _Requirements: 2.1, 4.1_

- [x] 3.2 Implement domain encoder
  - Create 6-dimensional conditioning vector generation
  - Add domain one-hot encoding and parameter normalization
  - Implement log scale normalization and relative noise parameters
  - **Test conditioning vector properties and ranges**
  - _Requirements: 2.1, 2.2_

- [x] 3.3 Implement guided sampling pipeline
  - Create EDMPosteriorSampler with EDM noise scheduling
  - Integrate Poisson-Gaussian guidance into sampling loop
  - Add configurable guidance weighting and step control
  - **Test on synthetic data first before real images**
  - _Requirements: 4.2-4.6_

## Phase 4: Data Pipeline (Week 3-4)

- [x] 4.1 Implement format-specific loaders
  - Photography: RAW loading with Bayer pattern handling (keep raw or demosaic)
  - Microscopy: Multi-channel TIFF with proper bit depth scaling
  - Astronomy: FITS format with header metadata extraction
  - **Test each loader with sample files and validate data integrity**
  - _Requirements: 5.1-5.4_
  - Photography: RAW loading with Bayer pattern handling (keep raw or demosaic)
  - Microscopy: Multi-channel TIFF with proper bit depth scaling
  - Astronomy: FITS format with header metadata extraction
  - **Test each loader with sample files and validate data integrity**
  - _Requirements: 5.1-5.4_

- [x] 4.2 Create unified DomainDataset
  - Implement base dataset class with reversible transforms integration
  - Add train/validation/test splitting with deterministic seeding
  - Include geometric augmentation pipeline (flips, rotations)
  - **Verify data loading correctness and transform consistency**
  - _Requirements: 2.4, 5.6, 5.7_

- [x] 4.3 Implement patch extraction for large images
  - Create memory-efficient patch sampler for training
  - Add overlapping patch processing for inference
  - Implement patch blending for seamless reconstruction
  - **Test reconstruction quality from patches vs full images**
  - _Requirements: 7.3, 7.6_

## Phase 5: Training Framework (Week 4-5)

- [x] 5.1 Implement training loop with deterministic mode
  - Create MultiDomainTrainer with v-parameterization loss
  - Add deterministic seeding option for reproducibility
  - Implement checkpoint management and training resumption
  - _Requirements: 6.1, 8.5_

- [x] 5.2 Add multi-domain balancing
  - Implement weighted sampling across photography/microscopy/astronomy
  - Add domain-specific loss weighting and performance monitoring
  - Create balanced batch composition strategies
  - _Requirements: 2.2, 2.5_

- [x] 5.3 Create evaluation framework
  - Standard metrics: PSNR, SSIM, LPIPS, MS-SSIM
  - Physics metrics: χ² consistency, residual whiteness, bias analysis
  - Domain-specific metrics: counting accuracy, photometry error
  - _Requirements: 6.3-6.6_

## Phase 6: Optimization & Baselines (Week 5-6)

- [x] 6.1 Implement performance optimizations
  - Add mixed precision training and inference
  - Optimize critical paths in guidance computation and sampling
  - Implement tiled processing for large images with memory management
  - _Requirements: 7.1-7.6_

- [x] 6.2 Integrate baseline methods
  - Classical methods: BM3D, Anscombe+BM3D implementations
  - Deep learning baselines: DnCNN, NAFNet integration
  - Create unified comparison framework with standardized evaluation
  - _Requirements: 6.5_

## Phase 7: Validation & Testing (Week 6-7)

- [ ] 7.1 Complete integration testing
  - End-to-end pipeline tests from RAW files to restored images
  - Cross-domain validation with real data from each domain
  - Memory profiling and performance benchmarking
  - _Requirements: 8.4_

- [ ] 7.2 Scientific validation
  - Synthetic data validation with exact Poisson-Gaussian statistics
  - Real data validation with known ground truth where available
  - Statistical consistency checks (χ² analysis, bias measurement)
  - _Requirements: 1.4, 8.3_

- [ ] 7.3 Edge case handling
  - Test extreme low-light scenarios (<1 photon per pixel)
  - Missing calibration fallback behavior
  - Corrupted file handling and graceful error recovery
  - _Requirements: 9.2, 9.4, 9.5_

## Phase 8: Multi-Resolution Implementation (Week 7-8)

- [x] 8.1 Implement progressive growing architecture
  - ✅ Created ProgressiveEDM model with progressive growing support (32→64→96→128)
  - ✅ Implemented resolution-conditioned training and inference
  - ✅ Added progressive growing scheduler for training
  - ✅ **Test resolution transitions and quality improvements** - Ready for testing
  - _Requirements: 4.1, 7.6_

- [x] 8.2 Implement hierarchical multi-scale processing
  - ✅ Created MultiScaleEDM with U-Net style hierarchical processing
  - ✅ Implemented multi-scale conditioning vectors
  - ✅ Added skip connections and feature fusion across scales
  - ✅ **Validate multi-scale feature consistency** - Ready for validation
  - _Requirements: 4.2-4.3_

- [x] 8.3 Create adaptive resolution pipeline
  - ✅ Implemented automatic resolution selection based on image characteristics
  - ✅ Added dynamic batch size adjustment for different resolutions
  - ✅ Created resolution-aware guidance weighting
  - ✅ **Test adaptive processing on various input sizes** - Ready for testing
  - _Requirements: 7.1-7.3_

- [x] 8.4 Multi-resolution training framework
  - ✅ Implemented staged training: low → mid → high resolution
  - ✅ Added resolution-specific learning rate schedules
  - ✅ Created resolution-balanced dataset sampling
  - ✅ **Validate progressive training stability and quality** - Ready for validation
  - _Requirements: 6.1, 8.5_

## Phase 9: L2-Guided Diffusion Baseline Implementation (Week 9)

- [ ] 9.1 Implement L2 guidance system
  - Create L2Guidance class with identical interface to PoissonGuidance
  - Implement simple MSE-based likelihood gradient computation
  - Add identical gamma scheduling for fair comparison
  - **Test numerical stability and gradient clipping**
  - _Requirements: Critical for paper submission_

- [ ] 9.2 Create unified guidance architecture
  - Abstract GuidanceComputer interface for polymorphic guidance
  - Guidance factory system for creating Poisson vs L2 guidance
  - Enhanced configuration system supporting guidance type selection
  - **Ensure identical sampling pipeline for both guidance types**
  - _Requirements: Scientific rigor and code maintainability_

- [ ] 9.3 Integrate L2 baseline into training pipeline
  - Unified training script supporting both guidance types
  - Configuration files for L2 baseline experiments
  - Identical hyperparameters except guidance computation
  - **Deterministic training with same random seeds for fair comparison**
  - _Requirements: Perfect ablation study_

- [ ] 9.4 Enhance evaluation framework for L2 comparison
  - UnifiedDiffusionBaseline class supporting both guidance types
  - Comparative evaluation scripts for Poisson vs L2
  - Statistical significance testing for guidance contribution
  - **Physics metrics validation (χ² analysis for both methods)**
  - _Requirements: Academic paper submission_

## Phase 10: Deployment & Documentation (Week 10-11)

- [ ] 10.1 Create user-friendly scripts
  - train_prior.py with configuration management and multi-GPU support
  - evaluate.py with comprehensive reporting and baseline comparisons
  - denoise.py for single image inference with automatic domain detection
  - batch_process.py for dataset processing with progress tracking
  - _Requirements: 6.2, 7.4_

- [ ] 10.2 Package models and data
  - Train final models for photography, microscopy, and astronomy domains
  - Create model zoo with automatic download and caching system
  - Package example datasets with proper calibration files and checksums
  - _Requirements: 8.2_

- [ ] 10.3 Write comprehensive documentation
  - Installation guide with environment setup and dependency management
  - API reference documentation for all major classes and functions
  - Domain-specific tutorials with real examples from each field
  - Troubleshooting guide for common issues and error messages
  - _Requirements: 8.6_

- [ ] 10.4 Create reproducibility package
  - Fixed seeds and deterministic configurations for all experiments
  - Dataset checksums and validation for data integrity
  - Exact dependency versions and environment specifications
  - Validation benchmarks with expected results and tolerances
  - _Requirements: 8.5_
