"""Evaluation of Pixel-Domain References (E6) and Systematic Ablations.

Enforces Ground Rule 1: NO hardcoded numbers. All numbers must be loaded
from real run artifacts in reports/runs/{run_name}/metrics.json or predictions.parquet.
If an artifact is missing, raises an error or writes 'NOT RUN'.

Outputs:
- reports/eval_refs.md (E6 comparison)
- reports/ablations.md (Ablation table from real run artifacts)
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.config import load_config
from nrcomp.lambdas import load_lambdas_config


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 hash of a file."""
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_revision_hash() -> str:
    """Return git commit hash if available."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
        return out.decode("ascii").strip()
    except Exception:
        return "UNKNOWN_GIT_SHA"


def load_run_artifact(run_name: str, runs_dir: Path, strict: bool = False) -> Optional[Dict[str, Any]]:
    """Load metrics.json artifact for a given run name."""
    run_path = runs_dir / run_name / "metrics.json"
    if not run_path.exists():
        if strict:
            raise FileNotFoundError(f"Required run artifact missing: {run_path}")
        return None

    try:
        with open(run_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_artifact_path"] = str(run_path)
        data["_artifact_sha256"] = compute_file_sha256(run_path)
        return data
    except Exception as e:
        if strict:
            raise RuntimeError(f"Corrupt run artifact {run_path}: {e}")
        return None


def format_pixel_refs_report(
    provenance_info: Dict[str, Any],
    latent_student_run: Optional[Dict[str, Any]],
    pixel_student_run: Optional[Dict[str, Any]],
    pretrained_nr_runs: Dict[str, Optional[Dict[str, Any]]],
    latency_stats: Optional[Dict[str, float]],
) -> str:
    """Format reports/eval_refs.md strictly from loaded run artifacts."""
    lines = []
    lines.append("# Stage 6: Pixel References & Comparative Baselines (`eval_refs.md`)")
    lines.append("")
    lines.append("## Provenance Header")
    lines.append(f"- **Git SHA:** `{provenance_info.get('git_sha', 'UNKNOWN')}`")
    lines.append(f"- **Date:** {provenance_info.get('date', datetime.datetime.now().isoformat())}")
    lines.append(f"- **Config Path:** `{provenance_info.get('config_path', '')}`")
    lines.append(f"- **Config SHA256:** `{provenance_info.get('config_sha256', '')}`")
    lines.append(f"- **Manifest Path:** `{provenance_info.get('manifest_path', '')}`")
    lines.append(f"- **Manifest SHA256:** `{provenance_info.get('manifest_sha256', '')}`")
    lines.append("- **Input Run Artifacts:**")
    for name, art in [("latent_student", latent_student_run), ("pixel_student", pixel_student_run)] + list(pretrained_nr_runs.items()):
        if art is not None:
            lines.append(f"  - `{name}`: `{art.get('_artifact_path')}` (SHA256: `{art.get('_artifact_sha256')}`)")
        else:
            lines.append(f"  - `{name}`: NOT RUN")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Latency & Computational Efficiency Trade-Off (E8)")
    lines.append("")
    lines.append("| Pipeline Approach | Synthesis Decode ($\\hat{y} \\to \\hat{x}$) | Model Forward Pass | Total Latency (ms) | Speedup vs Pixel |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    if latency_stats:
        lat_student = f"{latency_stats.get('latent_student_ms', 0.0):.2f} ms"
        decode_ms = f"{latency_stats.get('synthesis_decode_ms', 0.0):.1f} ms"
        pix_fwd = f"{latency_stats.get('pixel_student_ms', 0.0):.2f} ms"
        pix_tot = f"{latency_stats.get('total_pixel_pipeline_ms', 0.0):.2f} ms"
        speedup = f"{latency_stats.get('speedup_ratio', 1.0):.1f}x"
        lines.append(f"| **Latent Student (Ours)** | **0.0 ms (Bypassed)** | **{lat_student}** | **{lat_student}** | **{speedup} FASTER** |")
        lines.append(f"| Pixel Student (Custom CNN) | {decode_ms} | {pix_fwd} | {pix_tot} | 1.0x |")
    else:
        lines.append("| **Latent Student (Ours)** | **0.0 ms (Bypassed)** | NOT RUN | NOT RUN | NOT RUN |")
        lines.append("| Pixel Student (Custom CNN) | NOT RUN | NOT RUN | NOT RUN | NOT RUN |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Accuracy Comparison on KonIQ Test Split (E6)")
    lines.append("")
    lines.append("| Model / Approach | Domain | Overall SROCC | Within-$\\lambda$ SROCC | Spearman with $\\text{orig\\_mos}$ | Partial SROCC ($s, \\text{mos} \\mid y$) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    # Latent Student row
    if latent_student_run:
        srocc = f"{latent_student_run.get('srocc', 0.0):.4f}"
        wl_srocc = f"{latent_student_run.get('within_lambda_srocc', 0.0):.4f}"
        r_mos = f"{latent_student_run.get('rho_mos', 0.0):.4f}" if 'rho_mos' in latent_student_run else "N/A"
        p_mos = f"{latent_student_run.get('partial_mos', 0.0):.4f}" if 'partial_mos' in latent_student_run else "N/A"
        lines.append(f"| **Latent Student** | **Quantized Latents** | **{srocc}** | **{wl_srocc}** | {r_mos} | **{p_mos}** |")
    else:
        lines.append("| **Latent Student** | **Quantized Latents** | NOT RUN | NOT RUN | NOT RUN | NOT RUN |")

    # Pixel Student row
    if pixel_student_run:
        srocc = f"{pixel_student_run.get('srocc', 0.0):.4f}"
        wl_srocc = f"{pixel_student_run.get('within_lambda_srocc', 0.0):.4f}"
        r_mos = f"{pixel_student_run.get('rho_mos', 0.0):.4f}" if 'rho_mos' in pixel_student_run else "N/A"
        p_mos = f"{pixel_student_run.get('partial_mos', 0.0):.4f}" if 'partial_mos' in pixel_student_run else "N/A"
        lines.append(f"| Pixel Student (Custom CNN) | Decoded Pixels | {srocc} | {wl_srocc} | {r_mos} | {p_mos} |")
    else:
        lines.append("| Pixel Student (Custom CNN) | Decoded Pixels | NOT RUN | NOT RUN | NOT RUN | NOT RUN |")

    # Pretrained NR models rows
    for name, run_data in pretrained_nr_runs.items():
        if run_data:
            srocc = f"{run_data.get('srocc', 0.0):.4f}"
            wl_srocc = f"{run_data.get('within_lambda_srocc', 0.0):.4f}"
            r_mos = f"{run_data.get('rho_mos', 0.0):.4f}" if 'rho_mos' in run_data else "N/A"
            p_mos = f"{run_data.get('partial_mos', 0.0):.4f}" if 'partial_mos' in run_data else "N/A"
            lines.append(f"| Pretrained {name.upper()} | Decoded Pixels | {srocc} | {wl_srocc} | {r_mos} | {p_mos} |")
        else:
            lines.append(f"| Pretrained {name.upper()} | Decoded Pixels | NOT RUN | NOT RUN | NOT RUN | NOT RUN |")

    lines.append("")
    return "\n".join(lines)


def format_ablations_report(
    provenance_info: Dict[str, Any],
    ablation_runs: Dict[str, Optional[Dict[str, Any]]],
) -> str:
    """Format reports/ablations.md strictly from loaded ablation run artifacts."""
    lines = []
    lines.append("# Core Architectural & Methodological Ablations (`ablations.md`)")
    lines.append("")
    lines.append("## Provenance Header")
    lines.append(f"- **Git SHA:** `{provenance_info.get('git_sha', 'UNKNOWN')}`")
    lines.append(f"- **Date:** {provenance_info.get('date', datetime.datetime.now().isoformat())}")
    lines.append(f"- **Manifest SHA256:** `{provenance_info.get('manifest_sha256', '')}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Ablation Experiments Summary")
    lines.append("")
    lines.append("| Ablation ID | Description | Investigated Variant | Baseline / Full Model | Within-$\\lambda$ SROCC (KonIQ) | LiveC Within-$\\lambda$ | Run Status |")
    lines.append("| :--- | :--- | :--- | :--- | :---: | :---: | :--- |")

    ablation_defs = [
        ("A3", "Hyperprior Inputs", "Config A (y_hat only)", "Config B (y_hat + z_hat)", "student_conditioned_a", "student_conditioned_b"),
        ("A4", "Lambda Conditioning", "Blind Mode (No lambda)", "Conditioned Mode", "student_blind_a", "student_conditioned_a"),
        ("A7", "Spatial Masking", "Unmasked Global Pooling", "Masked Mean + Attn Pooling", "unmasked_a", "student_conditioned_a"),
        ("A2", "Pairwise Ranking Loss", "SmoothL1 Only (alpha=0.0)", "Composite Loss (alpha=0.5)", "no_rank_a", "student_conditioned_a"),
        ("A5", "Target Metric", "Teacher = DISTS (1 - DISTS)", "Teacher = HaarPSI", "teacher_dists_b", "student_conditioned_b"),
    ]

    for aid, desc, var_name, base_name, var_run_key, base_run_key in ablation_defs:
        var_data = ablation_runs.get(var_run_key)
        if var_data:
            score_k = f"{var_data.get('koniq_within_lambda_srocc', var_data.get('within_lambda_srocc', 0.0)):.4f}"
            score_l = f"{var_data.get('livec_within_lambda_srocc', 0.0):.4f}" if 'livec_within_lambda_srocc' in var_data else "N/A"
            status = f"Completed (`{var_data.get('_artifact_path')}`)"
        else:
            score_k = "NOT RUN"
            score_l = "NOT RUN"
            status = f"Run artifact `reports/runs/{var_run_key}/metrics.json` missing"
        lines.append(f"| **{aid}** | {desc} | {var_name} | {base_name} | **{score_k}** | {score_l} | {status} |")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Pixel References & Ablations Reports from Run Artifacts.")
    parser.add_argument("--config", type=str, default="configs/eval_refs.yaml", help="Config file path.")
    parser.add_argument("--runs-dir", type=str, default="reports/runs", help="Directory where run artifacts are stored.")
    parser.add_argument("--strict", action="store_true", help="Raise error if any run artifact is missing.")
    args = parser.parse_args()

    cfg_path = (PROJECT_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else {}

    runs_dir = (PROJECT_ROOT / args.runs_dir) if not Path(args.runs_dir).is_absolute() else Path(args.runs_dir)
    manifest_path = PROJECT_ROOT / cfg.get("manifest_path", "data/manifest.parquet")

    output_refs = Path(cfg.get("eval_refs_report", "reports/eval_refs.md"))
    if not output_refs.is_absolute():
        output_refs = PROJECT_ROOT / output_refs
    output_refs.parent.mkdir(parents=True, exist_ok=True)

    output_ablations = Path(cfg.get("ablations_report", "reports/ablations.md"))
    if not output_ablations.is_absolute():
        output_ablations = PROJECT_ROOT / output_ablations
    output_ablations.parent.mkdir(parents=True, exist_ok=True)

    provenance_info = {
        "git_sha": get_git_revision_hash(),
        "date": datetime.datetime.now().isoformat(),
        "config_path": str(cfg_path),
        "config_sha256": compute_file_sha256(cfg_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": compute_file_sha256(manifest_path),
    }

    # Load artifacts without fabrication
    latent_student_run = load_run_artifact("student_conditioned_b", runs_dir, strict=args.strict)
    pixel_student_run = load_run_artifact("pixel_student_conditioned", runs_dir, strict=args.strict)
    pretrained_nr_runs = {
        "clipiqa": load_run_artifact("pretrained_clipiqa", runs_dir, strict=args.strict),
        "niqe": load_run_artifact("pretrained_niqe", runs_dir, strict=args.strict),
        "musiq": load_run_artifact("pretrained_musiq", runs_dir, strict=args.strict),
    }

    # Latency artifact if measured
    latency_run = load_run_artifact("latency_benchmark", runs_dir, strict=False)
    latency_stats = latency_run.get("latency_stats") if latency_run else None

    # Write eval_refs.md
    report_refs = format_pixel_refs_report(
        provenance_info=provenance_info,
        latent_student_run=latent_student_run,
        pixel_student_run=pixel_student_run,
        pretrained_nr_runs=pretrained_nr_runs,
        latency_stats=latency_stats,
    )
    with open(output_refs, "w", encoding="utf-8") as f:
        f.write(report_refs)
    print(f"[SUCCESS] Pixel references report written to {output_refs}")

    # Load ablation runs
    ablation_run_keys = [
        "student_conditioned_a",
        "student_conditioned_b",
        "student_blind_a",
        "unmasked_a",
        "no_rank_a",
        "teacher_dists_b",
    ]
    ablation_runs = {k: load_run_artifact(k, runs_dir, strict=args.strict) for k in ablation_run_keys}

    # Write ablations.md
    report_ablations = format_ablations_report(provenance_info, ablation_runs)
    with open(output_ablations, "w", encoding="utf-8") as f:
        f.write(report_ablations)
    print(f"[SUCCESS] Ablations report written to {output_ablations}")


if __name__ == "__main__":
    main()
