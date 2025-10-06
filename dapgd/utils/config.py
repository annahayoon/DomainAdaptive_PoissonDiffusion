"""
Configuration utilities for DAPGD

PURPOSE: Load and manage hierarchical configuration files

Supports:
- YAML configuration files
- Hierarchical config merging (default + domain-specific)
- Command-line argument overrides
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file

    Args:
        path: Path to YAML file

    Returns:
        Configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from: {path}")
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries recursively

    Args:
        base_config: Base configuration
        override_config: Override configuration (takes precedence)

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def load_config(
    domain: Optional[str] = None,
    config_dir: Union[str, Path] = "config",
    custom_config: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load configuration with hierarchical merging

    Priority (highest to lowest):
    1. Custom config file (if provided)
    2. Domain-specific config (photo.yaml, micro.yaml, astro.yaml)
    3. Default config (default.yaml)

    Args:
        domain: Domain type ("photo", "micro", "astro")
        config_dir: Directory containing config files
        custom_config: Path to custom config file (overrides all)

    Returns:
        Merged configuration dictionary

    Example:
        # Load photography config (default + photo)
        config = load_config(domain="photo")

        # Load with custom overrides
        config = load_config(domain="photo", custom_config="my_config.yaml")
    """
    config_dir = Path(config_dir)

    # 1. Load default config
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        logger.warning(f"Default config not found: {default_path}")
        config = {}
    else:
        config = load_yaml(default_path)

    # 2. Merge domain-specific config
    if domain is not None:
        domain_path = config_dir / f"{domain}.yaml"
        if domain_path.exists():
            domain_config = load_yaml(domain_path)
            config = merge_configs(config, domain_config)
            logger.info(f"Merged domain config: {domain}")
        else:
            logger.warning(f"Domain config not found: {domain_path}")

    # 3. Merge custom config (highest priority)
    if custom_config is not None:
        custom_path = Path(custom_config)
        if custom_path.exists():
            custom_cfg = load_yaml(custom_path)
            config = merge_configs(config, custom_cfg)
            logger.info(f"Merged custom config: {custom_path}")
        else:
            logger.warning(f"Custom config not found: {custom_path}")

    return config


def save_config(config: Dict[str, Any], path: Union[str, Path]):
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to: {path}")


def get_nested_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """
    Get value from nested dictionary using dot notation

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "model.checkpoint")
        default: Default value if key not found

    Returns:
        Value at key_path or default

    Example:
        config = {"model": {"checkpoint": "model.pt"}}
        value = get_nested_value(config, "model.checkpoint")
        # Returns: "model.pt"
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def set_nested_value(config: Dict, key_path: str, value: Any):
    """
    Set value in nested dictionary using dot notation

    Args:
        config: Configuration dictionary (modified in-place)
        key_path: Dot-separated key path
        value: Value to set

    Example:
        config = {}
        set_nested_value(config, "model.checkpoint", "model.pt")
        # Result: {"model": {"checkpoint": "model.pt"}}
    """
    keys = key_path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


class ConfigManager:
    """
    Configuration manager for experiments

    Provides easy access to configuration values with validation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value"""
        return get_nested_value(self.config, key_path, default)

    def set(self, key_path: str, value: Any):
        """Set configuration value"""
        set_nested_value(self.config, key_path, value)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        """Dictionary-style setting"""
        self.config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self.config.copy()

    def save(self, path: Union[str, Path]):
        """Save configuration to file"""
        save_config(self.config, path)

    @classmethod
    def from_file(
        cls,
        domain: Optional[str] = None,
        config_dir: Union[str, Path] = "config",
        custom_config: Optional[Union[str, Path]] = None,
    ) -> "ConfigManager":
        """Create ConfigManager from files"""
        config = load_config(domain, config_dir, custom_config)
        return cls(config)
