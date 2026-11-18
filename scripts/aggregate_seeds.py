#!/usr/bin/env python3
"""Aggregate multi-seed training and evaluation runs for nrcomp.

Supports Milestone M8 (B3):
- Aggregates metrics (mean +/- std) across >= 3 seeds (e.g., 42, 43, 44).
- Evaluates seed ensembles (averaging predictions across seeds).
- Emits reports/seed_aggregation.md with mandatory provenance header.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.baselines import (
    compute_within_lambda_srocc,
    partial_spearman,
    pearson_r,
    spearman_rho,
)
from nrcomp.evaluate import (
    compute_pwrc_within_lambda,
    compute_rmse,
    compute_within_lambda_partial_spearman,
    get_file_sha256,
    get_git_commit_sha,
)


def compute_sample_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute sample mean and sample standard deviation."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean_val = sum(values) / n
    if n < 2:
        return mean_val, 0.0
    var = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    return mean_val, math.sqrt(max(0.0, var))


def load_run_predictions(run_dir: Path) -> List[Dict[str, Any]]:
    """Load predictions from predictions.parquet or predictions.csv."""
    parquet_path = run_dir / "predictions.parquet"
    csv_path = run_dir / "predictions.csv"

    # Try pandas / parquet first
    try:
        import pandas as pd
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            return df.to_dict(orient="records")
    except Exception:
        pass

    # Fallback to CSV
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = []
            for r in reader:
                rec = dict(r)
                for int_k in ["lambda_idx"]:
                    if rec.get(int_k) not in (None, ""):
                        rec[int_k] = int(rec[int_k])
                for float_k in ["lambda_value", "y", "s", "mapped_pred", "bpp", "orig_mos", "dists"]:
                    if rec.get(float_k) not in (None, ""):
                        try:
                            rec[float_k] = float(rec[float_k])
                        except ValueError:
                            pass
                    else:
                        rec[float_k] = None
                records.append(rec)
            return records

    return []


def aggregate_model_seeds(
    model_name: str,
    seeds: List[int],
    runs_dir: Path,
) -> Dict[str, Any]:
    """Aggregate evaluation metrics across multiple seeds for a single model."""
    seed_runs: Dict[int, Dict[str, Any]] = {}
    seed_predictions: Dict[int, List[Dict[str, Any]]] = {}
    artifact_paths: List[Path] = []

    for seed in seeds:
        # Check both {model} (for seed 42) and {model}_seed{seed}
        candidates = [runs_dir / f"{model_name}_seed{seed}"]
        if seed == 42:
            candidates.insert(0, runs_dir / model_name)

        found_run = False
        for c in candidates:
            eval_metrics_path = c / "eval_metrics.json"
            if eval_metrics_path.exists():
                try:
                    with open(eval_metrics_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    seed_runs[seed] = data
                    artifact_paths.append(eval_metrics_path)
                    found_run = True
                except Exception as e:
                    print(f"[WARN] Error reading {eval_metrics_path}: {e}")

            preds = load_run_predictions(c)
            if preds:
                seed_predictions[seed] = preds
                csv_p = c / "predictions.csv"
                if csv_p.exists():
                    artifact_paths.append(csv_p)
                found_run = True

            if found_run:
                break

    if not seed_runs and not seed_predictions:
        return {"status": "NOT RUN", "seeds_found": []}

    # Extract metrics per seed
    metrics_to_agg = [
        "srocc",
        "plcc",
        "rmse",
        "within_lambda_srocc",
        "within_lambda_partial_spearman_bpp",
        "ds_auc",
    ]

    per_seed_data: Dict[int, Dict[str, float]] = {}
    aggregated_stats: Dict[str, Tuple[float, float]] = {}

    for seed, run_data in seed_runs.items():
        e1 = run_data.get("e1", {})
        per_seed_data[seed] = {
            "srocc": float(e1.get("srocc", 0.0)),
            "plcc": float(e1.get("plcc", 0.0)),
            "rmse": float(e1.get("rmse", 0.0)),
            "within_lambda_srocc": float(e1.get("within_lambda_srocc", 0.0)),
            "within_lambda_partial_spearman_bpp": float(e1.get("within_lambda_partial_spearman_bpp", e1.get("partial_spearman_bpp", 0.0))),
            "ds_auc": float(e1.get("ds_auc", 0.0)),
        }

    for m in metrics_to_agg:
        vals = [per_seed_data[s][m] for s in sorted(per_seed_data.keys()) if m in per_seed_data[s]]
        aggregated_stats[m] = compute_sample_mean_std(vals)

    # Compute Ensemble if predictions available for >= 2 seeds
    ensemble_results: Optional[Dict[str, float]] = None
    if len(seed_predictions) >= 2:
        ensemble_results = evaluate_ensemble(seed_predictions)

    return {
        "status": "RUN",
        "seeds_found": sorted(per_seed_data.keys()),
        "per_seed": per_seed_data,
        "aggregated": aggregated_stats,
        "ensemble": ensemble_results,
        "artifacts": artifact_paths,
    }


def evaluate_ensemble(
    seed_predictions: Dict[int, List[Dict[str, Any]]],
    split: str = "test",
) -> Dict[str, float]:
    """Evaluate performance of ensemble predictions averaged across seeds."""
    # Filter by split and group by (image_id, lambda_idx)
    grouped_preds: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    targets: Dict[Tuple[str, int], float] = {}
    bpps: Dict[Tuple[str, int], float] = {}

    for seed, recs in seed_predictions.items():
        for r in recs:
            if r.get("split") == split:
                key = (str(r["image_id"]), int(r["lambda_idx"]))
                grouped_preds[key].append(float(r["s"]))
                targets[key] = float(r["y"])
                bpps[key] = float(r.get("bpp", 0.0))

    if not grouped_preds:
        return {}

    keys = sorted(grouped_preds.keys())
    ens_s = [sum(grouped_preds[k]) / len(grouped_preds[k]) for k in keys]
    ens_y = [targets[k] for k in keys]
    ens_bpp = [bpps[k] for k in keys]
    ens_lambdas = [k[1] for k in keys]

    srocc = spearman_rho(ens_s, ens_y)
    plcc = pearson_r(ens_s, ens_y)
    rmse = compute_rmse(ens_s, ens_y)
    wl_srocc, _ = compute_within_lambda_srocc(ens_s, ens_y, ens_lambdas)
    wl_part, _ = compute_within_lambda_partial_spearman(ens_s, ens_y, ens_bpp, ens_lambdas)
    pwrc = compute_pwrc_within_lambda(ens_s, ens_y, ens_lambdas)

    return {
        "srocc": srocc,
        "plcc": plcc,
        "rmse": rmse,
        "within_lambda_srocc": wl_srocc,
        "within_lambda_partial_spearman_bpp": wl_part,
        "ds_auc": pwrc["ds_auc"],
    }


def format_seed_aggregation_report(
    model_aggregations: Dict[str, Dict[str, Any]],
    seeds: List[int],
    manifest_path: Path,
    lambdas_path: Path,
) -> str:
    """Format publication-quality markdown report with multi-seed aggregation."""
    now_str = datetime.now(timezone.utc).isoformat()
    all_artifacts = []
    for agg in model_aggregations.values():
        all_artifacts.extend(agg.get("artifacts", []))

    lines = []
    lines.append("<!-- PROVENANCE HEADER START -->")
    lines.append("> [!NOTE]")
    lines.append("> **Multi-Seed Aggregation Provenance & Integrity Metadata**")
    lines.append(f"> - **Git Commit:** `{get_git_commit_sha()}`")
    lines.append(f"> - **Manifest:** `{manifest_path}` (SHA256: `{get_file_sha256(manifest_path)[:16]}...`)")
    lines.append(f"> - **Lambdas Config:** `{lambdas_path}` (SHA256: `{get_file_sha256(lambdas_path)[:16]}...`)")
    lines.append(f"> - **Target Seeds:** `{seeds}`")
    lines.append(f"> - **Generated At:** `{now_str}`")
    lines.append(f"> - **Total Run Artifacts Checked:** `{len(all_artifacts)}`")
    lines.append("<!-- PROVENANCE HEADER END -->")
    lines.append("")
    lines.append("# Multi-Seed Evaluation & Stability Report (`seed_aggregation.md`)")
    lines.append("")
    lines.append("Evaluates prediction stability, variance, and ensemble benefits across multiple random seeds on the **KonIQ Test Split**.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Multi-Seed Aggregated Performance (Mean $\\pm$ Sample Std)")
    lines.append("")
    lines.append("| Model | Seeds Found | Within-Lambda SROCC | Overall SROCC | PLCC | RMSE | Within-$\\lambda$ Partial ($s, y \\mid \\text{bpp}$) | DS-AUC |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    display_names = {
        "student_blind_a": "Student Blind A (y_hat)",
        "student_conditioned_a": "Student Conditioned A (y_hat + lambda)",
        "student_conditioned_b": "Student Conditioned B (y_hat + z_hat + lambda)",
    }

    for model_key, disp_name in display_names.items():
        if model_key not in model_aggregations:
            continue
        agg = model_aggregations[model_key]
        if agg.get("status") == "NOT RUN":
            lines.append(f"| **{disp_name}** | 0 / {len(seeds)} | *NOT RUN* | - | - | - | - | - |")
            continue

        n_found = len(agg["seeds_found"])
        st = agg["aggregated"]
        wl_str = f"{st['within_lambda_srocc'][0]:.4f} +/- {st['within_lambda_srocc'][1]:.4f}"
        s_str = f"{st['srocc'][0]:.4f} +/- {st['srocc'][1]:.4f}"
        p_str = f"{st['plcc'][0]:.4f} +/- {st['plcc'][1]:.4f}"
        r_str = f"{st['rmse'][0]:.4f} +/- {st['rmse'][1]:.4f}"
        part_str = f"{st['within_lambda_partial_spearman_bpp'][0]:.4f} +/- {st['within_lambda_partial_spearman_bpp'][1]:.4f}"
        ds_str = f"{st['ds_auc'][0]:.4f} +/- {st['ds_auc'][1]:.4f}"
        lines.append(f"| **{disp_name}** | {n_found} / {len(seeds)} | **{wl_str}** | {s_str} | {p_str} | {r_str} | **{part_str}** | {ds_str} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Seed Ensemble Performance vs Single-Seed Mean")
    lines.append("")
    lines.append("Ensemble predictions formed by uniformly averaging latent student predictions ($s_{ens} = \\frac{1}{K}\\sum s_k$) across all available seeds:")
    lines.append("")
    lines.append("| Model | Single-Seed Mean Within-$\\lambda$ | Ensemble Within-$\\lambda$ | $\\Delta$ Gain | Ensemble SROCC | Ensemble DS-AUC |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for model_key, disp_name in display_names.items():
        if model_key not in model_aggregations:
            continue
        agg = model_aggregations[model_key]
        ens = agg.get("ensemble")
        if not ens:
            continue
        mean_wl = agg["aggregated"]["within_lambda_srocc"][0]
        ens_wl = ens["within_lambda_srocc"]
        gain = ens_wl - mean_wl
        gain_str = f"+{gain:.4f}" if gain >= 0 else f"{gain:.4f}"
        lines.append(f"| **{disp_name}** | {mean_wl:.4f} | **{ens_wl:.4f}** | **{gain_str}** | {ens['srocc']:.4f} | {ens['ds_auc']:.4f} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Detailed Per-Seed Breakdown")
    lines.append("")

    for model_key, disp_name in display_names.items():
        if model_key not in model_aggregations or agg.get("status") == "NOT RUN":
            continue
        agg = model_aggregations[model_key]
        per_seed = agg.get("per_seed", {})
        if not per_seed:
            continue
        lines.append(f"### {disp_name}")
        lines.append("")
        lines.append("| Seed | Within-Lambda SROCC | Overall SROCC | PLCC | RMSE | DS-AUC |")
        lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
        for s in sorted(per_seed.keys()):
            row = per_seed[s]
            lines.append(f"| {s} | {row['within_lambda_srocc']:.4f} | {row['srocc']:.4f} | {row['plcc']:.4f} | {row['rmse']:.4f} | {row['ds_auc']:.4f} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Evaluation Aggregator")
    parser.add_argument("--runs-dir", type=str, default="reports/runs", help="Directory where run artifacts reside.")
    parser.add_argument("--models", type=str, default="student_blind_a,student_conditioned_a,student_conditioned_b", help="Comma-separated list of models.")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated list of seeds.")
    parser.add_argument("--manifest", type=str, default="data/manifest.parquet", help="Path to dataset manifest.")
    parser.add_argument("--output", type=str, default="reports/seed_aggregation.md", help="Markdown report output path.")
    parser.add_argument("--json-output", type=str, default="reports/seed_aggregation.json", help="JSON artifact output path.")
    args = parser.parse_args()

    runs_dir = (PROJECT_ROOT / args.runs_dir) if not Path(args.runs_dir).is_absolute() else Path(args.runs_dir)
    manifest_path = (PROJECT_ROOT / args.manifest) if not Path(args.manifest).is_absolute() else Path(args.manifest)
    lambdas_path = PROJECT_ROOT / "configs" / "lambdas.yaml"
    output_path = (PROJECT_ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    json_path = (PROJECT_ROOT / args.json_output) if not Path(args.json_output).is_absolute() else Path(args.json_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print(f"[AGGREGATE] Aggregating multi-seed runs for models: {model_names}")
    print(f"[AGGREGATE] Seeds: {seeds}")
    print(f"[AGGREGATE] Scanning directory: {runs_dir}")

    aggregations: Dict[str, Dict[str, Any]] = {}
    for model in model_names:
        print(f"Aggregating {model}...")
        res = aggregate_model_seeds(model, seeds, runs_dir)
        aggregations[model] = res
        print(f"  Seeds found: {res.get('seeds_found', [])}")

    report_md = format_seed_aggregation_report(aggregations, seeds, manifest_path, lambdas_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[SUCCESS] Multi-seed report saved to {output_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregations, f, indent=2, default=str)
    print(f"[SUCCESS] Multi-seed JSON artifact saved to {json_path}")


if __name__ == "__main__":
    main()
