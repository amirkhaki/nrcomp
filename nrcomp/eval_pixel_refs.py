"""Stage 6 reports: pixel/pretrained references (E6/E8) and the ablation table.

Enforces design.md Section 0 rule 1: NO hardcoded numbers. Every number is read from
artifacts written by `python -m nrcomp.evaluate` (reports/runs/{run}/eval_metrics.json and
predictions.csv); an absent artifact renders as NOT RUN -- never as 0.0.

Outputs:
  reports/eval_refs.md   -- E6 comparison (NOT RUN until the pixel student / pretrained NR runs exist)
  reports/ablations.md   -- ablations vs their reference model, per split, with paired bootstrap tests
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp import runs
from nrcomp.config import load_config
from nrcomp.eval_metrics import (
    compute_paired_difference_test,
    compute_sample_mean_std,
    records_with_teacher_target,
)
from nrcomp.provenance import collect_provenance, format_provenance_header, sha256_file

SPLITS = [("koniq_test", "KonIQ test"), ("livec_test", "LiveC test")]
NOT_RUN = "NOT RUN"

# (id, description, variant model, reference model, teacher(s) used for the comparison)
ABLATION_DEFS: List[Tuple[str, str, str, str, List[Optional[str]]]] = [
    ("A2a", "Ranking loss: regression only (alpha=0)", "no_rank_a", "conditioned_a", [None]),
    ("A2b", "Ranking loss: ranking only (alpha=1)", "pure_rank_a", "conditioned_a", [None]),
    ("A3", "Hyperprior input: Config A (y_hat only) vs Config B (y_hat + z_hat)", "conditioned_a", "conditioned_b", [None]),
    ("A4", "Lambda conditioning: blind vs conditioned", "blind_a", "conditioned_a", [None]),
    ("A5", "Teacher: DISTS-trained vs HaarPSI-trained (both Config B)", "teacher_dists_b", "conditioned_b", ["haarpsi", "one_minus_dists"]),
    ("A7", "Spatial masking: unmasked vs masked", "unmasked_a", "conditioned_a", [None]),
    ("A9", "Regression loss: plain L1 vs Smooth L1", "loss_l1_a", "conditioned_a", [None]),
]


def load_run_artifact(
    run_name: str,
    runs_dir: Union[Path, Dict[str, Any]],
    strict: bool = False,
    filename: str = runs.EVAL_METRICS_FILE,
) -> Optional[Dict[str, Any]]:
    """Load an evaluation artifact (default eval_metrics.json) for a run, or None / raise."""
    if isinstance(runs_dir, dict):
        if run_name in runs_dir:
            return runs_dir[run_name]
        return None
    path = runs_dir / run_name / filename
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"Required run artifact missing: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        if strict:
            raise RuntimeError(f"Corrupt run artifact {path}: {e}") from e
        return None
    data["_artifact_path"] = str(path)
    data["_artifact_sha256"] = sha256_file(path)
    return data


def _fmt(v: Any, nd: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _seed_txt(mean: Optional[float], std: Optional[float], n: int) -> str:
    if n == 0 or mean is None:
        return NOT_RUN
    if std is None:
        return f"{_fmt(mean)} (n={n})"
    return f"{_fmt(mean)} ± {_fmt(std)} (n={n})"


def _fmt_stat(mean: Optional[float], std: Optional[float], n: Optional[int] = None) -> str:
    if mean is None:
        return NOT_RUN
    if std is None or std == 0.0:
        return _fmt(mean)
    return f"{_fmt(mean)} ± {_fmt(std)}"


def _split_metric(doc: Optional[Dict[str, Any]], split: str, metric: str) -> Optional[float]:
    if not doc or not isinstance(doc, dict):
        return None
    sp = ((doc or {}).get("splits") or {}).get(split)
    if sp and metric in sp:
        return sp.get(metric)
    if metric in doc:
        return doc.get(metric)
    return None


def _seed_stat(
    model: str,
    split: str,
    runs_dir: Union[Path, Dict[str, Any]],
    metric: str = "within_lambda_srocc",
) -> Tuple[Optional[float], Optional[float], int]:
    vals = []
    if isinstance(runs_dir, dict):
        for candidate in [model, f"student_{model}", runs.ensemble_name(model)]:
            if candidate in runs_dir:
                v = _split_metric(runs_dir[candidate], split, metric)
                if v is not None:
                    vals.append(v)
                    break
        for seed in runs.DEFAULT_SEEDS:
            cand = runs.run_name(model, seed)
            if cand in runs_dir:
                v = _split_metric(runs_dir[cand], split, metric)
                if v is not None:
                    vals.append(v)
    else:
        for seed in runs.DEFAULT_SEEDS:
            doc = load_run_artifact(runs.run_name(model, seed), runs_dir)
            v = _split_metric(doc, split, metric)
            if v is not None:
                vals.append(v)
    mean, std = compute_sample_mean_std(vals)
    return mean, std, len(vals)


def _load_ensemble_records(
    model: str, split: str, runs_dir: Union[Path, Dict[str, Any]]
) -> Optional[List[Dict[str, Any]]]:
    if isinstance(runs_dir, dict):
        return None
    from nrcomp.evaluate import load_predictions_csv

    path = runs_dir / runs.ensemble_name(model) / f"{runs.PREDICTIONS_STEM}.csv"
    if not path.exists():
        return None
    return [r for r in load_predictions_csv(path) if r["split"] == split] or None


def format_pixel_refs_report(
    provenance_header: Union[str, Dict[str, Any]],
    latent_student_run: Optional[Dict[str, Any]],
    pixel_student_run: Optional[Dict[str, Any]],
    pretrained_nr_runs: Dict[str, Optional[Dict[str, Any]]],
    latency_stats: Optional[Dict[str, float]],
    runs_dir: Optional[Union[Path, Dict[str, Any]]] = None,
) -> str:
    """eval_refs.md, strictly from loaded artifacts."""
    if isinstance(provenance_header, dict):
        header_str = format_provenance_header(provenance_header, "Stage 6 Provenance")
    else:
        header_str = str(provenance_header) if provenance_header else ""
    lines = [header_str, "# Stage 6: Pixel References & Comparative Baselines (`eval_refs.md`)", ""]
    lines.append("## Input run artifacts")
    lines.append("")
    named = [("latent_student", latent_student_run), ("pixel_student", pixel_student_run)] + list(pretrained_nr_runs.items())
    for name, art in named:
        if art is not None:
            lines.append(f"- `{name}`: `{art.get('_artifact_path')}` (SHA-256: `{art.get('_artifact_sha256')}`)")
        else:
            lines.append(f"- `{name}`: {NOT_RUN}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Latency & Computational Efficiency (E8)")
    lines.append("")
    lines.append("| Pipeline | Synthesis decode (ms) | Model forward (ms) | Total (ms) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    if latency_stats:
        speedup = latency_stats.get('speedup_ratio')
        speedup_txt = f" ({_fmt(speedup, 1)}x FASTER)" if speedup else ""
        lines.append(f"| Latent student | bypassed | {_fmt(latency_stats.get('latent_student_ms'), 2)} | {_fmt(latency_stats.get('latent_student_ms'), 2)}{speedup_txt} |")
        lines.append(f"| Pixel student (decode + CNN) | {_fmt(latency_stats.get('synthesis_decode_ms'), 2)} | "
                     f"{_fmt(latency_stats.get('pixel_student_ms'), 2)} | {_fmt(latency_stats.get('total_pixel_pipeline_ms'), 2)} |")
    else:
        lines.append(f"| Latent student | bypassed | {NOT_RUN} | {NOT_RUN} |")
        lines.append(f"| Pixel student (decode + CNN) | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Accuracy comparison (E6), within-λ SROCC per split")
    lines.append("")
    lines.append("| Model | Domain | KonIQ test within-λ | LiveC test within-λ |")
    lines.append("| :--- | :---: | :---: | :---: |")

    # Headline: seed mean ± std for Latent Student Conditioned B
    if runs_dir:
        mean_k, std_k, n_k = _seed_stat("conditioned_b", "koniq_test", runs_dir)
        mean_l, std_l, n_l = _seed_stat("conditioned_b", "livec_test", runs_dir)
        if mean_k is not None:
            lines.append(f"| **Latent student (Conditioned B, seed mean ± std)** | Quantized latents | **{_fmt_stat(mean_k, std_k, n_k)}** | **{_fmt_stat(mean_l, std_l, n_l)}** |")
            lines.append(f"| ↳ Latent student (Conditioned B, ensemble) | Quantized latents | {_fmt(_split_metric(latent_student_run, 'koniq_test', 'within_lambda_srocc'))} | {_fmt(_split_metric(latent_student_run, 'livec_test', 'within_lambda_srocc'))} |")
        else:
            lines.append(f"| Latent student (Conditioned B, ensemble) | Quantized latents | {_fmt(_split_metric(latent_student_run, 'koniq_test', 'within_lambda_srocc'))} | {_fmt(_split_metric(latent_student_run, 'livec_test', 'within_lambda_srocc'))} |")
    else:
        lines.append(f"| Latent student (Conditioned B, ensemble) | Quantized latents | {_fmt(_split_metric(latent_student_run, 'koniq_test', 'within_lambda_srocc'))} | {_fmt(_split_metric(latent_student_run, 'livec_test', 'within_lambda_srocc'))} |")

    # Pixel student
    if pixel_student_run:
        lines.append(f"| Pixel student (custom CNN) | Decoded pixels | {_fmt(_split_metric(pixel_student_run, 'koniq_test', 'within_lambda_srocc'))} | {_fmt(_split_metric(pixel_student_run, 'livec_test', 'within_lambda_srocc'))} |")
    else:
        lines.append(f"| Pixel student (custom CNN) | Decoded pixels | {NOT_RUN} | {NOT_RUN} |")

    # Pretrained NR models
    for n, d in pretrained_nr_runs.items():
        if d is None:
            lines.append(f"| Pretrained {n.upper()} | Decoded pixels | {NOT_RUN} | {NOT_RUN} |")
        else:
            lines.append(f"| Pretrained {n.upper()} | Decoded pixels | {_fmt(_split_metric(d, 'koniq_test', 'within_lambda_srocc'))} | {_fmt(_split_metric(d, 'livec_test', 'within_lambda_srocc'))} |")
    lines.append("")
    return "\n".join(lines)


def format_ablations_report(
    provenance_header: Union[str, Dict[str, Any]],
    runs_dir: Union[Path, Dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> str:
    """ablations.md: each variant vs its reference, per split, strictly from evaluation artifacts."""
    if isinstance(provenance_header, dict):
        header_str = format_provenance_header(provenance_header, "Ablations Provenance")
    else:
        header_str = str(provenance_header) if provenance_header else ""
    lines = [header_str, "# Core Architectural & Methodological Ablations (`ablations.md`)", ""]
    lines.append("Each row compares an ablated variant with its reference model. 'Seed mean ± std' is over the individual "
                 "seeds; Δ, CI and p come from the SEED ENSEMBLES with a paired image-clustered bootstrap "
                 f"(N={n_resamples}) on within-λ SROCC. p-values are NOT corrected for multiple comparisons.")
    lines.append("")

    masked = load_run_artifact(runs.ensemble_name("unmasked_a"), runs_dir)
    if masked is not None:
        vf = masked.get("valid_cell_fraction") or {}
        lines.append(f"Measured valid-cell fraction (A7 relevance): KonIQ test = {_fmt(vf.get('koniq_test'))}, "
                     f"LiveC test = {_fmt(vf.get('livec_test'))}. A fraction of 1.0 means that split has no padding, so "
                     "masking cannot change the result there.")
        lines.append("")

    for split, title in SPLITS:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| ID | Variant | Reference | Teacher | Variant seed mean ± std | Reference seed mean ± std | "
                     "Δ ensemble (variant − reference) [95% CI] | p | Status |")
        lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
        for aid, desc, var_m, ref_m, teachers in ABLATION_DEFS:
            var_mean, var_std, var_n = _seed_stat(var_m, split, runs_dir)
            ref_mean, ref_std, ref_n = _seed_stat(ref_m, split, runs_dir)
            ra = _load_ensemble_records(var_m, split, runs_dir)
            rb = _load_ensemble_records(ref_m, split, runs_dir)
            for teacher in teachers:
                v_txt = _seed_txt(var_mean, var_std, var_n)
                r_txt = _seed_txt(ref_mean, ref_std, ref_n)
                if ra is None or rb is None or var_n == 0 or ref_n == 0:
                    missing = [m for m, r in ((var_m, ra), (ref_m, rb)) if r is None]
                    lines.append(f"| **{aid}** | {runs.DISPLAY_NAMES.get(var_m, var_m)} | {runs.DISPLAY_NAMES.get(ref_m, ref_m)} | "
                                 f"{teacher or 'own'} | {v_txt} | {r_txt} | {NOT_RUN} | - | missing ensemble artifacts: {missing or 'seed runs'} |")
                    continue
                if teacher is not None:
                    recs_a = records_with_teacher_target(ra, teacher)
                    recs_b = records_with_teacher_target(rb, teacher)
                    # per-seed means are on each model's own target; for a common teacher use ensemble values only
                    v_txt = r_txt = "see Δ"
                else:
                    recs_a, recs_b = ra, rb
                    v_txt = _seed_txt(var_mean, var_std, var_n)
                    r_txt = _seed_txt(ref_mean, ref_std, ref_n)
                pt = compute_paired_difference_test(recs_a, recs_b, "within_lambda_srocc", n_resamples=n_resamples, seed=seed)
                ci = pt["ci"]
                delta_txt = (f"{_fmt(pt['delta_obs'])} [{_fmt(ci[0])}, {_fmt(ci[1])}]"
                             if pt["delta_obs"] is not None else "n/a")
                lines.append(f"| **{aid}** | {runs.DISPLAY_NAMES.get(var_m, var_m)} | {runs.DISPLAY_NAMES.get(ref_m, ref_m)} | "
                             f"{teacher or 'own'} | {v_txt} | {r_txt} | {delta_txt} | {_fmt(pt['p_value'], 3)} | "
                             f"{'significant' if pt['significant'] else 'not significant'} (n_img={pt['n_images']}) |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate eval_refs.md and ablations.md from evaluation artifacts.")
    parser.add_argument("--config", type=str, default="configs/eval_refs.yaml")
    parser.add_argument("--runs-dir", type=str, default="reports/runs")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--strict", action="store_true", help="Raise if a required artifact is missing.")
    args = parser.parse_args()

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p

    cfg_path = resolve(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else {}
    runs_dir = resolve(args.runs_dir)
    manifest_path = resolve(cfg.get("manifest_path", "data/manifest.parquet"))
    out_refs = resolve(cfg.get("eval_refs_report", "reports/eval_refs.md"))
    out_abl = resolve(cfg.get("ablations_report", "reports/ablations.md"))
    out_refs.parent.mkdir(parents=True, exist_ok=True)

    latent = load_run_artifact(runs.ensemble_name("conditioned_b"), runs_dir, strict=args.strict)
    pixel = load_run_artifact("pixel_student_conditioned", runs_dir, strict=False)
    pretrained = {
        n: load_run_artifact(f"pretrained_{n}", runs_dir, strict=False)
        for n in ("clipiqa", "musiq", "niqe", "brisque", "maniqa")
    }
    latency_doc = load_run_artifact("latency_benchmark", runs_dir, strict=False)
    latency_stats = latency_doc.get("latency_stats") if latency_doc else None

    artifacts = [d["_artifact_path"] for d in [latent, pixel, latency_doc, *pretrained.values()] if d]
    prov = collect_provenance(config_path=cfg_path, manifest_path=manifest_path, seeds=runs.DEFAULT_SEEDS, input_artifacts=artifacts)
    header = format_provenance_header(prov, "Stage 6 Provenance")

    out_refs.write_text(format_pixel_refs_report(header, latent, pixel, pretrained, latency_stats, runs_dir=runs_dir), encoding="utf-8")
    print(f"[SUCCESS] Pixel references report written to {out_refs}")

    abl_artifacts = [str(p) for p in sorted(runs_dir.glob("*/eval_metrics.json"))]
    abl_prov = collect_provenance(config_path=cfg_path, manifest_path=manifest_path, seeds=runs.DEFAULT_SEEDS, input_artifacts=abl_artifacts)
    out_abl.write_text(
        format_ablations_report(format_provenance_header(abl_prov, "Ablations Provenance"), runs_dir, n_resamples=args.n_resamples),
        encoding="utf-8",
    )
    print(f"[SUCCESS] Ablations report written to {out_abl}")


if __name__ == "__main__":
    main()
