# DAPGD: Domain-Adaptive Poisson-Gaussian Diffusion

Physics-informed guidance for photon-limited image restoration using diffusion models.

## Quick Start

### Installation

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install DAPGD in development mode
pip install -e .

# 4. Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```python
from dapgd.sampling.dapgd_sampler import create_sampler

# Create sampler with guidance
sampler = create_sampler(
    checkpoint_path="checkpoints/model.pt",
    guidance_config={'s': 1000, 'sigma_r': 5, 'kappa': 0.5},
    num_steps=50
)

# Run inference
restored = sampler.sample(y_e=noisy_observation)
```

### Command Line

```bash
# Inference with guidance
python scripts/inference.py \
    --mode guided \
    --input data/test/image.png \
    --checkpoint checkpoints/model.pt \
    --s 1000 --sigma_r 5

# Baseline (no guidance)
python scripts/inference.py \
    --mode baseline \
    --input data/test/image.png \
    --checkpoint checkpoints/model.pt
```

## Project Structure

```
dapgd/
├── guidance/          # Poisson-Gaussian guidance implementation
├── sampling/          # Guided sampler
├── models/            # Model utilities
├── data/              # Data pipeline
├── metrics/           # Evaluation metrics
└── utils/             # Utilities (logging, visualization)

config/                # Configuration files
scripts/               # Executable scripts
tests/                 # Test suite
```

## Configuration

Domain-specific configurations are in `config/`:
- `photo.yaml`: Photography (10³ photons/pixel)
- `micro.yaml`: Microscopy (10¹ photons/pixel)
- `astro.yaml`: Astronomy (10⁰ photons/pixel)

Load a configuration:
```python
import yaml
with open('config/photo.yaml') as f:
    config = yaml.safe_load(f)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_pg_guidance.py -v

# Run with coverage
pytest tests/ --cov=dapgd --cov-report=html
```

## Documentation

See `DAPGD_Implementation.md` for detailed implementation guide.

## License

[Add your license here]

## Citation

[Add citation information]
