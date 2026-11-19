"""Provenance helpers shared by every stage (design.md Section 0, rule 2).

Every generated report and run summary must carry: git SHA, config path + SHA-256,
manifest path + SHA-256, lambdas config SHA-256, seed(s), timestamp, and the list of
input artifacts with their SHA-256. Hashes are written in FULL (never truncated).

Pure standard library so it works on the dev machine and on the server.
"""

import datetime
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAMBDAS_YAML = PROJECT_ROOT / "configs" / "lambdas.yaml"

PathLike = Union[str, Path]
MISSING = "MISSING"


def sha256_file(path: Optional[PathLike]) -> str:
    """Hex SHA-256 of a file, or 'MISSING' if it does not exist."""
    if path is None:
        return MISSING
    p = Path(path)
    if not p.is_file():
        return MISSING
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: Sequence[str], cwd: Optional[PathLike]) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd or PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def git_sha(cwd: Optional[PathLike] = None) -> str:
    return _git(["rev-parse", "HEAD"], cwd) or "UNKNOWN"


def git_dirty(cwd: Optional[PathLike] = None) -> Optional[bool]:
    """True if the working tree has uncommitted changes, None if git is unavailable."""
    status = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    if status is None:
        return None
    return len(status) > 0


def now_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def lambdas_sha256() -> str:
    return sha256_file(LAMBDAS_YAML)


def collect_provenance(
    config_path: Optional[PathLike] = None,
    manifest_path: Optional[PathLike] = None,
    seeds: Optional[Union[int, Iterable[int]]] = None,
    input_artifacts: Optional[Iterable[PathLike]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect a machine-readable provenance record (all hashes are full SHA-256)."""
    if seeds is None:
        seed_list: List[int] = []
    elif isinstance(seeds, int):
        seed_list = [seeds]
    else:
        seed_list = [int(s) for s in seeds]

    artifacts = []
    for art in input_artifacts or []:
        artifacts.append({"path": str(art), "sha256": sha256_file(art)})

    prov: Dict[str, Any] = {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "config_path": str(config_path) if config_path is not None else None,
        "config_sha256": sha256_file(config_path) if config_path is not None else None,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path is not None else None,
        "lambdas_path": str(LAMBDAS_YAML),
        "lambdas_sha256": lambdas_sha256(),
        "seeds": seed_list,
        "generated_at": now_utc_iso(),
        "input_artifacts": artifacts,
    }
    if extra:
        prov["extra"] = dict(extra)
    return prov


def format_provenance_header(prov: Dict[str, Any], title: str = "Provenance") -> str:
    """Render a provenance record as the mandatory markdown header block."""
    dirty = prov.get("git_dirty")
    dirty_txt = "unknown" if dirty is None else ("DIRTY (uncommitted changes)" if dirty else "clean")
    seeds = prov.get("seeds") or []
    lines = [
        "<!-- PROVENANCE HEADER START -->",
        "> [!NOTE]",
        f"> **{title}**",
        f"> - **Git Commit:** `{prov.get('git_sha', 'UNKNOWN')}` (working tree: {dirty_txt})",
        f"> - **Config:** `{prov.get('config_path')}` (SHA-256: `{prov.get('config_sha256')}`)",
        f"> - **Manifest:** `{prov.get('manifest_path')}` (SHA-256: `{prov.get('manifest_sha256')}`)",
        f"> - **Lambdas config:** `{prov.get('lambdas_path')}` (SHA-256: `{prov.get('lambdas_sha256')}`)",
        f"> - **Seeds:** `{seeds}`",
        f"> - **Generated at (UTC):** `{prov.get('generated_at')}`",
    ]
    artifacts = prov.get("input_artifacts") or []
    if artifacts:
        lines.append("> - **Input artifacts:**")
        for art in artifacts:
            lines.append(f">   - `{art['path']}` (SHA-256: `{art['sha256']}`)")
    else:
        lines.append("> - **Input artifacts:** none")
    lines.append("<!-- PROVENANCE HEADER END -->")
    lines.append("")
    return "\n".join(lines)


def required_provenance_keys() -> List[str]:
    """Keys that must be present in every provenance record (used by tests)."""
    return [
        "git_sha",
        "config_sha256",
        "manifest_sha256",
        "lambdas_sha256",
        "seeds",
        "generated_at",
        "input_artifacts",
    ]
