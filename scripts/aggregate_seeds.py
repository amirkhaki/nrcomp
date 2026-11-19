#!/usr/bin/env python3
"""Aggregate multi-seed evaluation runs (Milestone M8, B3).

Reads ONLY artifacts written by `python -m nrcomp.evaluate`:
    reports/runs/{model}_seed{N}/eval_metrics.json     (per-seed point metrics)
    reports/runs/{model}_ensemble/eval_metrics.json    (seed-ensemble metrics with CIs)
and writes reports/seed_aggregation.md (+ .json) with the mandatory provenance header.

Run it AFTER evaluate (the old pipeline ran it before any evaluation existed and found
nothing). Seeds or splits without artifacts are reported as NOT RUN, never as zeros.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp import runs
from nrcomp.eval_metrics import compute_sample_mean_std
from nrcomp.provenance import LAMBDAS_YAML, collect_provenance, format_provenance_header

METRICS = [
    ("within_lambda_srocc", "Within-λ SROCC"),
    ("within_lambda_partial_spearman_bpp", "Within-λ partial (| bpp)"),
    ("ds_auc", "DS-AUC"),
    ("srocc", "Overall SROCC (rate-confounded)"),
    ("rmse", "RMSE"),
]
SPLITS = [("koniq_test", "KonIQ test"), ("livec_test", "LiveC test")]


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_model(model: str, seeds: List[int], runs_dir: Path) -> Dict[str, Any]:
    """Collect per-seed metrics and the ensemble for one model from eval artifacts."""
    per_seed: Dict[int, Dict[str, Any]] = {}
    artifacts: List[Path] = []
    
    # Model name aliases (e.g. blind_a <-> student_blind_a)
    model_aliases = [model]
    if model.startswith("student_"):
        model_aliases.append(model[len("student_"):])
    else:
        model_aliases.append(f"student_{model}")

    for seed in seeds:
        doc = None
        for m_alias in model_aliases:
            candidates = [
                runs_dir / runs.run_name(m_alias, seed) / runs.EVAL_METRICS_FILE,
                runs_dir / f"{m_alias}_seed{seed}" / runs.EVAL_METRICS_FILE,
            ]
            if seed == 42:
                candidates.append(runs_dir / m_alias / runs.EVAL_METRICS_FILE)
            for p in candidates:
                if p.exists():
                    doc = _load(p)
                    if doc is not None:
                        per_seed[seed] = doc
                        artifacts.append(p)
                        break
            if doc is not None:
                break

    ens_path = None
    ensemble = None
    for m_alias in model_aliases:
        cand = runs_dir / runs.ensemble_name(m_alias) / runs.EVAL_METRICS_FILE
        if cand.exists():
            ensemble = _load(cand)
            if ensemble is not None:
                ens_path = cand
                artifacts.append(ens_path)
                break

    if not per_seed:
        return {"status": "NOT RUN", "seeds_found": [], "seeds_missing": list(seeds), "artifacts": artifacts}

    stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for split, _ in SPLITS:
        stats[split] = {}
        for metric, _name in METRICS:
            vals = {}
            for seed, doc in per_seed.items():
                sp = (doc.get("splits") or {}).get(split)
                # Fallback for old schema where e1 was koniq_test
                if sp is None and split == "koniq_test":
                    sp = doc.get("e1")
                elif sp is None and split == "livec_test":
                    sp = doc.get("e2")
                if sp is not None and sp.get(metric) is not None:
                    vals[seed] = sp[metric]
            mean, std = compute_sample_mean_std(list(vals.values()))
            stats[split][metric] = {"mean": mean, "std": std, "n": len(vals), "values": vals}

    return {
        "status": "RUN",
        "seeds_found": sorted(per_seed),
        "seeds_missing": [s for s in seeds if s not in per_seed],
        "stats": stats,
        "aggregated": stats.get("koniq_test", {}),
        "ensemble": ensemble,
        "artifacts": artifacts,
    }


def evaluate_ensemble(
    seed_predictions: Dict[int, List[Dict[str, Any]]],
    split: str = "test",
) -> Dict[str, Any]:
    """Average predictions across seeds and compute basic metrics."""
    from nrcomp.baselines import pearson_r, spearman_rho
    from nrcomp.eval_metrics import compute_rmse, ensemble_records

    normalized: Dict[int, List[Dict[str, Any]]] = {}
    for s, recs in seed_predictions.items():
        norm_list = []
        for r in recs:
            if r.get("split") != split and split != "all":
                continue
            nr = dict(r)
            if "raw_pred" not in nr and "s" in nr:
                nr["raw_pred"] = nr["s"]
            if "target" not in nr and "y" in nr:
                nr["target"] = nr["y"]
            norm_list.append(nr)
        normalized[s] = norm_list

    ens_recs, _ = ensemble_records(normalized)
    if not ens_recs:
        return {"srocc": 0.0, "plcc": 0.0, "rmse": 0.0}
    preds = [r["raw_pred"] for r in ens_recs]
    targets = [r["target"] for r in ens_recs]
    return {
        "srocc": spearman_rho(preds, targets),
        "plcc": pearson_r(preds, targets),
        "rmse": compute_rmse(preds, targets),
        "records": ens_recs,
    }


# Backwards compatibility alias
aggregate_model_seeds = aggregate_model


def _f(v: Optional[float], nd: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def format_seed_aggregation_report(
    aggregations: Dict[str, Dict[str, Any]],
    seeds: List[int],
    provenance_header: Optional[Union[str, Path, Dict[str, Any]]] = None,
    manifest_path: Optional[Path] = None,
    lambdas_path: Optional[Path] = None,
) -> str:
    if provenance_header is None or isinstance(provenance_header, Path):
        man = manifest_path or (provenance_header if isinstance(provenance_header, Path) else None)
        prov = collect_provenance(manifest_path=man, seeds=seeds)
        provenance_header = format_provenance_header(prov, "Multi-Seed Aggregation Provenance")
    elif isinstance(provenance_header, dict):
        provenance_header = format_provenance_header(provenance_header, "Multi-Seed Aggregation Provenance")
    lines = [str(provenance_header), "# Multi-Seed Evaluation & Stability Report (`seed_aggregation.md`)", ""]
    lines.append(f"Seeds requested: `{seeds}`. Mean ± sample standard deviation (ddof=1) over the seeds found; "
                 "the ensemble averages raw predictions over those seeds. Splits are never pooled.")
    lines.append("")

    for split, title in SPLITS:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Model | Seeds | " + " | ".join(name for _, name in METRICS) + " |")
        lines.append("| :--- | :---: | " + " | ".join([":---:"] * len(METRICS)) + " |")
        for model, agg in aggregations.items():
            disp = runs.DISPLAY_NAMES.get(model, model)
            if agg["status"] == "NOT RUN":
                lines.append(f"| {disp} | 0 / {len(seeds)} | **NOT RUN** |" + " - |" * (len(METRICS) - 1))
                continue
            cells = []
            for metric, _ in METRICS:
                st = agg["stats"][split][metric]
                if st["mean"] is None:
                    cells.append("n/a")
                elif st["std"] is None:
                    cells.append(f"{st['mean']:.4f}")
                else:
                    cells.append(f"{st['mean']:.4f} ± {st['std']:.4f}")
            lines.append(f"| {disp} | {len(agg['seeds_found'])} / {len(seeds)} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append(f"### {title}: seed ensemble vs single-seed mean (within-λ SROCC)")
        lines.append("")
        lines.append("| Model | Single-seed mean | Ensemble [95% CI] | Δ (ensemble − mean) |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for model, agg in aggregations.items():
            if agg["status"] == "NOT RUN" or not agg.get("ensemble"):
                continue
            sp = (agg["ensemble"].get("splits") or {}).get(split)
            if not sp:
                continue
            mean = agg["stats"][split]["within_lambda_srocc"]["mean"]
            ens = sp.get("within_lambda_srocc")
            ci = sp.get("within_lambda_srocc_ci")
            ci_txt = f" [{ci[0]:.4f}, {ci[1]:.4f}]" if ci and ci[0] is not None else ""
            gain = (ens - mean) if (ens is not None and mean is not None) else None
            lines.append(f"| {runs.DISPLAY_NAMES.get(model, model)} | {_f(mean)} | {_f(ens)}{ci_txt} | "
                         f"{'n/a' if gain is None else f'{gain:+.4f}'} |")
        lines.append("")

    lines.append("## Per-seed values (within-λ SROCC)")
    lines.append("")
    lines.append("| Model | Split | " + " | ".join(f"seed {s}" for s in seeds) + " |")
    lines.append("| :--- | :--- | " + " | ".join([":---:"] * len(seeds)) + " |")
    for model, agg in aggregations.items():
        if agg["status"] == "NOT RUN":
            continue
        for split, title in SPLITS:
            vals = agg["stats"][split]["within_lambda_srocc"]["values"]
            lines.append(f"| {runs.DISPLAY_NAMES.get(model, model)} | {title} | "
                         + " | ".join(_f(vals.get(s, vals.get(str(s)))) for s in seeds) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Evaluation Aggregator")
    parser.add_argument("--runs-dir", type=str, default="reports/runs")
    parser.add_argument("--models", type=str, default=",".join(runs.ALL_MODELS))
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in runs.DEFAULT_SEEDS))
    parser.add_argument("--manifest", type=str, default="data/manifest.parquet")
    parser.add_argument("--output", type=str, default="reports/seed_aggregation.md")
    parser.add_argument("--json-output", type=str, default="reports/seed_aggregation.json")
    args = parser.parse_args()

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p

    runs_dir = resolve(args.runs_dir)
    manifest_path = resolve(args.manifest)
    output_path, json_path = resolve(args.output), resolve(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    aggregations = {m: aggregate_model(m, seeds, runs_dir) for m in models}
    for m, a in aggregations.items():
        print(f"[AGGREGATE] {m}: seeds found {a['seeds_found']}")

    artifacts: List[Path] = []
    for a in aggregations.values():
        artifacts.extend(a.get("artifacts", []))
    prov = collect_provenance(manifest_path=manifest_path, seeds=seeds, input_artifacts=artifacts)
    md = format_seed_aggregation_report(aggregations, seeds, format_provenance_header(prov, "Multi-Seed Aggregation Provenance"))
    output_path.write_text(md, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"provenance": prov, "aggregations": aggregations}, f, indent=2, default=str)
    print(f"[SUCCESS] Wrote {output_path} and {json_path}")


if __name__ == "__main__":
    main()
