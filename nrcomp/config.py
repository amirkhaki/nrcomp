"""Configuration loading, validation, and serialization."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Lightweight YAML parser fallback for dev environments without PyYAML."""
    result: Dict[str, Any] = {}
    current_key = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip()
            val = parts[1].strip()

            # Handle boolean and numeric types
            if val.lower() == "true":
                result[key] = True
            elif val.lower() == "false":
                result[key] = False
            elif val.isdigit():
                result[key] = int(val)
            else:
                try:
                    result[key] = float(val)
                except ValueError:
                    # Strip quotes if string
                    result[key] = val.strip("'\"")
    return result


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        import yaml
        cfg = yaml.safe_load(content)
        return cfg if cfg is not None else {}
    except ImportError:
        return parse_simple_yaml(content)


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """Validate that required configuration keys exist."""
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")
    return True
