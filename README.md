# Domain-Adaptive Poisson-Gaussian Diffusion for Low-Light Image Restoration

A unified framework for low-light image restoration across photography, microscopy, and astronomy domains using physically-correct Poisson-Gaussian noise modeling with diffusion models.

## Project Status

🚧 **Currently in development** - Phase 1: Core Infrastructure

## Recent Progress

- ✅ **Phase 0 Complete**: Prerequisites & Dependencies
  - ✅ **Task 0.1**: EDM external dependencies integrated successfully
  - ✅ **Task 0.2**: Development environment fully configured
- ✅ **Phase 1 Complete**: Core Infrastructure
  - ✅ **Task 1.1**: Project structure and base interfaces
  - ✅ **Task 1.2**: Error handling framework
  - ✅ **Task 1.3**: Reversible transforms with tests
- ⏳ **Phase 2 In Progress**: Physics Foundation
  - ✅ **Task 2.1**: Implement calibration system (EXCELLENT)
  - ⏳ **Task 2.2**: Implement Poisson-Gaussian guidance

## Key Features

- **Physics-Aware**: Exact Poisson-Gaussian noise modeling without approximations
- **Cross-Domain**: Single model adapts to photography, microscopy, and astronomy
- **Perfect Reconstruction**: Reversible transforms preserve all metadata and calibration
- **Noise-Only Focus**: Identity forward operator (no PSF/blur modeling in this version)

## Architecture Overview

```
Input Pipeline:
RAW File → Calibration → Electrons → Normalize[0,1] → Transform(128×128)

Processing Pipeline:
Noisy(128×128) → Domain Conditioning → EDM Diffusion → Physics Guidance → Denoised(128×128)

Output Pipeline:
Denoised(128×128) → Inverse Transform → Original Size → Final Output
```

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
- [ ] External dependencies setup (EDM integration)
- [ ] Core infrastructure implementation
- [ ] Physics-based guidance system
- [ ] Model integration and training
- [ ] Evaluation and validation

## Citation

```bibtex
@article{poisson-diffusion-2024,
  title={Domain-Adaptive Poisson-Gaussian Diffusion for Low-Light Image Restoration},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.
