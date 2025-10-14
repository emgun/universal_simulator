"""Configuration loading utilities with include support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config_with_includes(path: str | Path, _visited: set[Path] | None = None) -> Dict[str, Any]:
    """Load a YAML config file with support for include directives.

    Recursively resolves `include: path` directives by loading and merging
    the included config first, then overlaying the current config on top.

    Args:
        path: Path to the YAML config file
        _visited: Internal set to track visited files and prevent circular includes

    Returns:
        Merged configuration dictionary

    Example config.yaml:
        include: base.yaml

        training:
          batch_size: 32  # Overrides base.yaml's batch_size
    """
    config_path = Path(path).resolve()

    # Prevent circular includes
    if _visited is None:
        _visited = set()
    if config_path in _visited:
        raise ValueError(f"Circular include detected: {config_path}")
    _visited.add(config_path)

    # Load current config
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    # Check for include directive
    include_path = config.pop("include", None)
    if include_path is None:
        return config

    # Resolve include path relative to current config's directory
    include_full_path = (config_path.parent / include_path).resolve()

    # Handle missing file extensions
    if not include_full_path.exists() and not str(include_path).endswith(".yaml"):
        include_full_path = include_full_path.with_suffix(".yaml")

    if not include_full_path.exists():
        raise FileNotFoundError(f"Included config not found: {include_full_path} (from {config_path})")

    # Recursively load the included config
    base_config = load_config_with_includes(include_full_path, _visited)

    # Merge: current config overlays on top of base config
    merged = _deep_merge(base_config, config)

    return merged


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with overlay taking precedence.

    Args:
        base: Base dictionary
        overlay: Dictionary to overlay on top (higher priority)

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = _deep_merge(result[key], value)
        else:
            # Overlay wins
            result[key] = value

    return result
