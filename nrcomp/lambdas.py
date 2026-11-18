"""Single source of truth for ELIC codec lambda configurations."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LAMBDAS_YAML = PROJECT_ROOT / "configs" / "lambdas.yaml"


def load_lambdas_config(yaml_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Load the verified ELIC lambda list from configs/lambdas.yaml."""
    p = Path(yaml_path) if yaml_path is not None else DEFAULT_LAMBDAS_YAML
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    if not p.exists():
        raise FileNotFoundError(f"Lambda configuration file not found at: {p}")

    from nrcomp.config import load_config
    cfg = load_config(p)
    lambdas = cfg.get("lambdas", [])
    if not lambdas:
        raise ValueError(f"No lambdas defined in {p}")
    return sorted(lambdas, key=lambda x: int(x["idx"]))


def get_lambda_names(yaml_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Get list of lambda names in order of lambda_idx."""
    return [item["name"] for item in load_lambdas_config(yaml_path)]


def get_lambda_values(yaml_path: Optional[Union[str, Path]] = None) -> List[float]:
    """Get list of numeric lambda values in order of lambda_idx."""
    return [float(item["value"]) for item in load_lambdas_config(yaml_path)]


def get_lambda_checkpoints(yaml_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Get list of updated checkpoint filenames in order of lambda_idx."""
    return [item["checkpoint"] for item in load_lambdas_config(yaml_path)]
