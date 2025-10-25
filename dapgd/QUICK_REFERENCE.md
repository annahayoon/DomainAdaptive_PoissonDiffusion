# DAPGD Quick Reference

## Installation

```bash
python -m venv venv
source venv/bin/activate
cd dapgd
pip install -e .
```

## Configuration

```python
from dapgd.utils.config import load_config

# Load domain-specific config
config = load_config(domain='photo')  # or 'micro', 'astro'

# Access nested values
from dapgd.utils.config import get_nested_value
s = get_nested_value(config, 'physics.s')  # 1000.0
```

### Domain Parameters

| Domain | s | σ_r | κ | Steps |
|--------|---|-----|---|-------|
| photo  | 1000 | 5.0 | 0.5 | 50 |
| micro  | 100 | 2.0 | 0.7 | 50 |
| astro  | 10 | 1.0 | 1.0 | 75 |

## Logging

```python
from dapgd.utils.logging import get_logger

logger = get_logger('my_exp', use_tensorboard=True)
logger.info("Message")
logger.log_metric("psnr", 28.5, step=100)
logger.log_image("result", img_tensor, step=100)
logger.close()
```

## Visualization

```python
from dapgd.utils.visualization import (
    save_comparison_image,
    plot_residual_analysis,
    save_image_grid
)

# Comparison
save_comparison_image(noisy, restored, clean, "out.png")

# Residual analysis
plot_residual_analysis(restored, noisy, s=1000, sigma_r=5, "residuals.png")

# Grid
save_image_grid([img1, img2, img3], titles=['A', 'B', 'C'], output_path="grid.png")
```

## File Locations

- **Configs**: `config/*.yaml`
- **Logs**: `experiments/runs/<exp_name>/`
- **Results**: `experiments/results/`
- **Tests**: `tests/`
- **Scripts**: `scripts/`

## Common Commands

```bash
# Verify setup
python verify_dapgd_setup.py

# Run tests (when implemented)
pytest tests/ -v

# With coverage
pytest tests/ --cov=dapgd --cov-report=html
```

## Next Steps

See `DAPGD_Implementation.md` Section 4 for implementation guide:
1. Implement `dapgd/guidance/pg_guidance.py`
2. Implement `dapgd/sampling/dapgd_sampler.py`
3. Implement `dapgd/metrics/*.py`
4. Write tests in `tests/`
5. Create inference script in `scripts/`
