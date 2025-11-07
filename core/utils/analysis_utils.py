"""Shared utilities for analysis scripts to avoid code duplication."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def json_default(obj: Any) -> Any:
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json_safe(data: Dict, filepath: Path, indent: int = 2) -> None:
    """Save dictionary to JSON file with safe serialization."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent, default=json_default)


def load_json_safe(filepath: Path) -> Dict:
    """Load JSON file safely."""
    with open(filepath, "r") as f:
        return json.load(f)


def ensure_tensor_format(tensor: torch.Tensor, expected_dims: int = 3) -> torch.Tensor:
    """Ensure tensor is in correct format (C, H, W) or (B, C, H, W)."""
    if tensor.dim() == 4 and expected_dims == 3:
        tensor = tensor.squeeze(0)  # Remove batch dimension if present
    elif tensor.dim() == 2 and expected_dims == 3:
        tensor = tensor.unsqueeze(0)  # Add channel dimension
    return tensor


def compute_aggregate_stats(values: List[float]) -> Dict[str, float]:
    """Compute aggregate statistics for a list of values."""
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }

    values_array = np.array(values)
    valid_values = values_array[np.isfinite(values_array)]

    if len(valid_values) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": len(values),
        }

    return {
        "mean": float(np.mean(valid_values)),
        "std": float(np.std(valid_values)),
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "count": len(values),
    }
