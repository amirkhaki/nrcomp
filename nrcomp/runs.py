"""Single source of truth for run naming, artifact locations and model registries.

Every stage (train, evaluate, aggregate, ablation report) MUST derive paths from this
module. The previous pipeline used three incompatible naming schemes (`blind_a`,
`blind_a_seed43`, `student_blind_a`), which made multi-seed aggregation impossible and
let seeds overwrite each other's checkpoints.

Layout (relative to the project root):

    checkpoints/{config}_seed{N}/best_model.pth     best checkpoint of one training run
    checkpoints/{config}_seed{N}/latest_model.pth   last-epoch checkpoint (with optimizer)
    reports/runs/{config}_seed{N}/run_meta.json     training provenance
    reports/runs/{config}_seed{N}/metrics.json      training summary + per-epoch history
    reports/runs/{config}_seed{N}/history.csv       per-epoch history
    reports/runs/{config}_seed{N}/curves.png        training curves
    reports/runs/{config}_seed{N}/eval_metrics.json evaluation of this run (written by evaluate)
    reports/runs/{config}_seed{N}/predictions.csv   standardized predictions (+ .parquet)
    reports/runs/{config}_ensemble/...              seed-ensemble evaluation (evaluate)
    reports/runs/baseline_*/...                     shortcut baselines (evaluate)

The run name ALWAYS carries the seed suffix, including seed 42.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNS_SUBDIR = Path("reports") / "runs"
CHECKPOINT_SUBDIR = Path("checkpoints")

DEFAULT_SEEDS: List[int] = [42, 43, 44]

# Training configs (value of `config_name` in configs/train_*.yaml)
PRIMARY_MODELS: List[str] = ["blind_a", "conditioned_a", "conditioned_b"]
ABLATION_MODELS: List[str] = [
    "no_rank_a",
    "pure_rank_a",
    "unmasked_a",
    "teacher_dists_b",
    "loss_l1_a",
]
ALL_MODELS: List[str] = PRIMARY_MODELS + ABLATION_MODELS

BASELINE_MODELS: List[str] = [
    "baseline_bpp",
    "baseline_lambda",
    "baseline_stats",
    "baseline_bpp_per_lambda",
    "baseline_bpp_lambda",
]

DISPLAY_NAMES: Dict[str, str] = {
    "baseline_bpp": "Baseline (a) [bpp only]",
    "baseline_lambda": "Baseline (b) [lambda only]",
    "baseline_stats": "Baseline (c) [latent stats + bpp]",
    "baseline_bpp_per_lambda": "Baseline (d) [bpp, per-lambda linear]",
    "baseline_bpp_lambda": "Baseline (e) [bpp + lambda, GBDT]",
    "blind_a": "Student Blind A (y_hat)",
    "conditioned_a": "Student Conditioned A (y_hat + lambda)",
    "conditioned_b": "Student Conditioned B (y_hat + z_hat + lambda)",
    "student_blind_a": "Student Blind A (y_hat)",
    "student_conditioned_a": "Student Conditioned A (y_hat + lambda)",
    "student_conditioned_b": "Student Conditioned B (y_hat + z_hat + lambda)",
    "no_rank_a": "A2 No-Rank (alpha=0, regression only)",
    "pure_rank_a": "A2 Pure-Rank (alpha=1, ranking only)",
    "unmasked_a": "A7 Unmasked",
    "teacher_dists_b": "A5 Teacher=DISTS (Config B)",
    "loss_l1_a": "A9 L1 loss",
}

# Evaluation split names used in predictions files and eval_metrics.json
SPLIT_VAL = "val"
SPLIT_KONIQ_TEST = "test"
SPLIT_LIVEC_TEST = "livec_test"
SPLIT_EXCLUDED = "excluded"  # manifest rows that must never be used (no verified MOS)
KONIQ_SPLITS = ("train", "val", "test")

# File names
RUN_META_FILE = "run_meta.json"
TRAIN_METRICS_FILE = "metrics.json"
HISTORY_FILE = "history.csv"
CURVES_FILE = "curves.png"
EVAL_METRICS_FILE = "eval_metrics.json"
PREDICTIONS_STEM = "predictions"  # -> predictions.csv / predictions.parquet
BEST_CKPT_FILE = "best_model.pth"
LATEST_CKPT_FILE = "latest_model.pth"


def run_name(config_name: str, seed: int) -> str:
    """Canonical run name. Always includes the seed suffix."""
    return f"{config_name}_seed{int(seed)}"


def ensemble_name(config_name: str) -> str:
    """Run name of the seed-ensemble evaluation of a config."""
    return f"{config_name}_ensemble"


def parse_run_name(name: str) -> Optional[Dict[str, object]]:
    """Inverse of run_name(); returns None if `name` is not `{config}_seed{N}`."""
    marker = "_seed"
    idx = name.rfind(marker)
    if idx <= 0:
        return None
    tail = name[idx + len(marker):]
    if not tail.isdigit():
        return None
    return {"config_name": name[:idx], "seed": int(tail)}


def _root(project_root: Optional[PathLike]) -> Path:
    return Path(project_root) if project_root is not None else PROJECT_ROOT


def runs_dir(project_root: Optional[PathLike] = None) -> Path:
    return _root(project_root) / RUNS_SUBDIR


def run_dir(config_name: str, seed: int, project_root: Optional[PathLike] = None) -> Path:
    return runs_dir(project_root) / run_name(config_name, seed)


def ensemble_dir(config_name: str, project_root: Optional[PathLike] = None) -> Path:
    return runs_dir(project_root) / ensemble_name(config_name)


def checkpoint_dir(config_name: str, seed: int, project_root: Optional[PathLike] = None) -> Path:
    return _root(project_root) / CHECKPOINT_SUBDIR / run_name(config_name, seed)


def best_checkpoint_path(config_name: str, seed: int, project_root: Optional[PathLike] = None) -> Path:
    primary = checkpoint_dir(config_name, seed, project_root) / BEST_CKPT_FILE
    if not primary.exists() and seed == 42:
        root = _root(project_root)
        for alt in [config_name, f"student_{config_name}"]:
            cand = root / CHECKPOINT_SUBDIR / alt / BEST_CKPT_FILE
            if cand.exists():
                return cand
    return primary


def latest_checkpoint_path(config_name: str, seed: int, project_root: Optional[PathLike] = None) -> Path:
    primary = checkpoint_dir(config_name, seed, project_root) / LATEST_CKPT_FILE
    if not primary.exists() and seed == 42:
        root = _root(project_root)
        for alt in [config_name, f"student_{config_name}"]:
            cand = root / CHECKPOINT_SUBDIR / alt / LATEST_CKPT_FILE
            if cand.exists():
                return cand
    return primary
