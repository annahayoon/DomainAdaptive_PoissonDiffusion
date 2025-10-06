# Domain-Adaptive Poisson-Gaussian Diffusion for Low-Light Image Restoration

A unified framework for low-light image restoration across photography, microscopy, and astronomy domains using physically-correct Poisson-Gaussian noise modeling with diffusion models.

## Project Status

🚧 **Currently in development** - Phase 3: Model Training & Validation

## Recent Progress

- ✅ **Phase 0 Complete**: Prerequisites & Dependencies
  - ✅ **Task 0.1**: EDM external dependencies integrated successfully
  - ✅ **Task 0.2**: Development environment fully configured
- ✅ **Phase 1 Complete**: Core Infrastructure
  - ✅ **Task 1.1**: Project structure and base interfaces
  - ✅ **Task 1.2**: Error handling framework
  - ✅ **Task 1.3**: Reversible transforms with tests
- ✅ **Phase 2 Complete**: Physics Foundation
  - ✅ **Task 2.1**: Implement calibration system (EXCELLENT)
  - ✅ **Task 2.2**: Implement Poisson-Gaussian guidance
- 🚧 **Phase 3 In Progress**: Model Training & Validation
  - ✅ **Task 3.1**: EDM native training implementation
  - ✅ **Task 3.2**: Photography domain training pipeline
  - ⏳ **Task 3.3**: Cross-domain validation and evaluation

## Key Features

- **Physics-Aware**: Exact Poisson-Gaussian noise modeling without approximations
- **Cross-Domain**: Single model adapts to photography, microscopy, and astronomy
- **Perfect Reconstruction**: Reversible transforms preserve all metadata and calibration
- **Noise-Only Focus**: Identity forward operator (no PSF/blur modeling in this version)

## Architecture Overview

```
Input Pipeline:
RAW File → Calibration → Electrons → Normalize[0,1] → Transform(256×256)

Processing Pipeline:
Noisy(256×256) → Domain Conditioning → EDM Diffusion → Physics Guidance → Denoised(256×256)

Output Pipeline:
Denoised(256×256) → Inverse Transform → Original Size → Final Output
```

## Recent Implementation Highlights

### EDM Native Training
- **Native EDM Integration**: Full integration with EDM's training utilities and architecture
- **Photography Domain**: Optimized training pipeline for photography datasets
- **Real-time Monitoring**: Split-screen GPU monitoring with detailed memory and performance tracking
- **Conservative Training**: Optimized hyperparameters to prevent overfitting on limited datasets
- **Early Stopping**: Automated validation and early stopping to prevent overtraining

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- PyTorch 2.0+

### External Dependencies

This project integrates with the EDM (Elucidating the Design Space of Diffusion-Based Generative Models) codebase. See `external/README.md` for integration details.

## Project Structure

```
poisson-diffusion/
├── core/                   # Core algorithms and transforms
├── models/                 # Model architectures and wrappers
├── data/                   # Data loading and preprocessing
├── configs/                # Configuration files
├── scripts/                # Training and evaluation scripts
├── tests/                  # Unit and integration tests
├── external/               # External dependencies (EDM)
└── docs/                   # Documentation
```

## Development Status

- [x] Project specification and design
- [x] External dependencies setup (EDM integration)
- [x] Core infrastructure implementation
- [x] Physics-based guidance system
- [x] EDM native training implementation
- [x] Photography domain training pipeline
- [ ] Cross-domain validation and evaluation
- [ ] Multi-domain model training
- [ ] Comprehensive evaluation framework

## Citation

```bibtex
@article{poisson-diffusion-2025,
  title={Domain-Adaptive Poisson-Gaussian Diffusion for Low-Light Image Restoration},
  author={[Yoon, Ha Yun Anna Yoon; Hong, Jaewan
  journal={[Journal]},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.
