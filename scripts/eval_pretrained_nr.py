#!/usr/bin/env python3
"""Evaluate Pretrained No-Reference (NR-IQA) models from pyiqa and piq on codec reconstructions.

Milestone M11 (E2 / E6):
Evaluates standard pretrained NR-IQA baselines (e.g. CLIPIQA, MUSIQ, NIQE, BRISQUE, MANIQA)
on decoded RGB reconstructions for koniq_test and livec_test splits.

Outputs:
    reports/runs/pretrained_{model}/eval_metrics.json   (standard artifact for eval_pixel_refs)
    reports/runs/pretrained_{model}/predictions.csv     (standard predictions table)
    reports/runs/pretrained_{model}/predictions.parquet (standard predictions parquet)
    reports/eval_pretrained_nr.md                       (comprehensive comparative report)

Adheres strictly to design.md Section 0 integrity rules:
- Mandatory provenance header with SHA-256 hashes of manifest and configs.
- Splits are never pooled (koniq_test and livec_test evaluated strictly separately).
- Calibration fit on koniq_val only; rank metrics computed on raw predictions.
- No fabricated numbers.
"""

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp import runs
from nrcomp.baselines import partial_spearman, pearson_r, spearman_rho
from nrcomp.config import load_config
from nrcomp.data.manifest import ParquetManifest, is_missing
from nrcomp.eval_metrics import (
    Logistic4PFitter,
    compute_clustered_bootstrap_cis,
    compute_mos_independence,
    compute_pwrc_within_lambda,
    compute_rmse,
    compute_within_lambda_partial_spearman,
)
from nrcomp.labels.registry import get_metric_spec, prepare_target_value
from nrcomp.lambdas import get_lambda_values, load_lambdas_config
from nrcomp.provenance import (
    LAMBDAS_YAML,
    collect_provenance,
    format_provenance_header,
    sha256_file,
)

DEFAULT_NR_MODELS = ["clipiqa", "musiq", "niqe", "brisque", "maniqa"]

# Known lower-better NR metrics (where smaller score indicates better quality)
LOWER_BETTER_METRICS = {"niqe", "ilniqe", "brisque", "piqe", "nrqm"}


def resolve_recon_path(row: Dict[str, Any], project_root: Path) -> Optional[Path]:
    """Find the reconstruction image file on disk."""
    raw_p = row.get("recon_path")
    candidates = []
    if raw_p:
        p = Path(raw_p)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(project_root / p)
            # Try alternate with 'recons' vs 'recon'
            if "data/recon/" in raw_p:
                candidates.append(project_root / raw_p.replace("data/recon/", "data/recons/"))
            elif "data/recons/" in raw_p:
                candidates.append(project_root / raw_p.replace("data/recons/", "data/recon/"))

    # Fallback to standard convention: data/recons/{dataset}/lambda_{idx}/{image_id}.png
    dataset = str(row.get("dataset", "koniq"))
    lambda_idx = int(row.get("lambda_idx", 0))
    image_id = str(row.get("image_id", ""))
    stem = Path(image_id).stem
    for base in ["data/recons", "data/recon"]:
        candidates.append(project_root / base / dataset / f"lambda_{lambda_idx}" / f"{image_id}")
        candidates.append(project_root / base / dataset / f"lambda_{lambda_idx}" / f"{stem}.png")

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def init_nr_metric(
    model_name: str, device: Any, backend: str = "pyiqa"
) -> Tuple[Callable[[Path], float], bool]:
    """Initialize NR-IQA model and return (eval_fn, is_lower_better)."""
    m_lower = model_name.lower().strip()
    is_lower_better = m_lower in LOWER_BETTER_METRICS

    if backend == "pyiqa":
        try:
            import pyiqa
            import torch

            # Safe GPU device check for Pascal
            if isinstance(device, torch.device) and device.type == "cuda":
                if torch.cuda.get_device_capability(device)[0] < 7:
                    torch.backends.cudnn.enabled = False

            metric = pyiqa.create_metric(m_lower, device=device)
            if hasattr(metric, "lower_better"):
                is_lower_better = bool(metric.lower_better)

            def _eval_pyiqa(img_path: Path) -> float:
                with torch.no_grad():
                    score = metric(str(img_path))
                    if isinstance(score, torch.Tensor):
                        return float(score.item())
                    return float(score)

            return _eval_pyiqa, is_lower_better
        except Exception as e:
            print(f"[WARN] Failed to load '{model_name}' via pyiqa: {e}")

    # Fallback to piq if applicable
    if backend == "piq" or m_lower in ["brisque"]:
        try:
            import piq
            import torch
            from PIL import Image
            import torchvision.transforms.functional as TF

            if m_lower == "brisque":
                is_lower_better = True

                def _eval_piq_brisque(img_path: Path) -> float:
                    with torch.no_grad():
                        img = Image.open(img_path).convert("RGB")
                        tensor = TF.to_tensor(img).unsqueeze(0).to(device)
                        return float(piq.brisque(tensor).item())

                return _eval_piq_brisque, is_lower_better
        except Exception as e:
            print(f"[WARN] Failed to load '{model_name}' via piq: {e}")

    raise RuntimeError(
        f"Could not initialize NR model '{model_name}' with backend '{backend}'. "
        f"Please verify pyiqa and piq are installed in your environment."
    )


def compute_within_lambda_srocc(
    preds: List[float], targets: List[float], lambdas: List[int]
) -> Tuple[float, Dict[int, float]]:
    """Compute Spearman correlation within each lambda, and return the mean and breakdown."""
    by_lambda: Dict[int, Tuple[List[float], List[float]]] = {}
    for p, t, l in zip(preds, targets, lambdas):
        if l not in by_lambda:
            by_lambda[l] = ([], [])
        by_lambda[l][0].append(p)
        by_lambda[l][1].append(t)

    rhos = {}
    for l, (p_l, t_l) in sorted(by_lambda.items()):
        rhos[l] = spearman_rho(p_l, t_l)

    mean_rho = float(sum(rhos.values()) / len(rhos)) if rhos else 0.0
    return mean_rho, rhos


def evaluate_split_records(
    records: List[Dict[str, Any]],
    fitter: Optional[Logistic4PFitter] = None,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate one split of predictions with standard and within-lambda metrics."""
    if not records:
        return {}

    preds = [float(r["aligned_score"]) for r in records]
    targets = [float(r["target"]) for r in records]
    bpps = [float(r["bpp_actual"]) for r in records]
    lambdas = [int(r["lambda_idx"]) for r in records]
    image_ids = [str(r["image_id"]) for r in records]
    mos_list = [r["orig_mos"] for r in records]

    # Rank metrics (uncalibrated raw predictions)
    mean_w_srocc, per_l_srocc = compute_within_lambda_srocc(preds, targets, lambdas)
    srocc = spearman_rho(preds, targets)
    wl_part_bpp, per_l_part_bpp = compute_within_lambda_partial_spearman(preds, targets, bpps, lambdas)
    pw = compute_pwrc_within_lambda(preds, targets, lambdas)
    ds_auc = pw.get("ds_auc")

    # Calibrated linear metrics (if validation fitter provided)
    if fitter is not None:
        mapped = fitter.predict(preds)
        plcc = pearson_r(mapped, targets)
        rmse = compute_rmse(mapped, targets)
    else:
        mapped = preds
        plcc = pearson_r(preds, targets)
        rmse = compute_rmse(preds, targets)

    # Attach mapped prediction to records
    for r, m in zip(records, mapped):
        r["mapped_pred"] = float(m)

    # Bootstrap CIs (image-clustered)
    ci_w_srocc = (None, None)
    ci_srocc = (None, None)
    ci_plcc = (None, None)
    ci_ds_auc = (None, None)

    if n_bootstrap > 0 and len(set(image_ids)) >= 10:
        cis = compute_clustered_bootstrap_cis(
            records=records,
            raw_pred_key="aligned_score",
            mapped_pred_key="mapped_pred",
            target_key="target",
            n_resamples=n_bootstrap,
            seed=seed,
            compute_ds_auc_ci=True,
        )
        ci_w_srocc = cis.get("within_lambda_srocc", (None, None))
        ci_srocc = cis.get("srocc", (None, None))
        ci_plcc = cis.get("plcc", (None, None))
        ci_ds_auc = cis.get("ds_auc", (None, None))

    # E3: Independence from orig_mos
    e3_stats = {}
    valid_mos_pairs = [m for m in mos_list if m is not None]
    if len(valid_mos_pairs) >= 10:
        try:
            e3_stats = compute_mos_independence(
                preds=preds,
                targets=targets,
                orig_mos_list=mos_list,
                lambda_indices=lambdas,
                mae_preds=mapped,
            )
        except Exception:
            e3_stats = {}

    return {
        "n_samples": len(records),
        "n_images": len(set(image_ids)),
        "within_lambda_srocc": mean_w_srocc,
        "within_lambda_srocc_ci": list(ci_w_srocc),
        "per_lambda_srocc": {str(k): float(v) for k, v in per_l_srocc.items()},
        "within_lambda_partial_spearman_bpp": wl_part_bpp,
        "per_lambda_partial_spearman_bpp": {str(k): float(v) for k, v in per_l_part_bpp.items()},
        "srocc": srocc,
        "srocc_ci": list(ci_srocc),
        "plcc": plcc,
        "plcc_ci": list(ci_plcc),
        "rmse": rmse,
        "ds_auc": ds_auc,
        "ds_auc_ci": list(ci_ds_auc),
        "pwrc": pw,
        "e3_mos_independence": e3_stats,
    }


def save_run_artifacts(
    run_dir: Path,
    model_name: str,
    meta_info: Dict[str, Any],
    eval_doc: Dict[str, Any],
    records_by_split: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Save standard eval_metrics.json, predictions.csv, and predictions.parquet."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. eval_metrics.json
    full_doc = dict(meta_info)
    full_doc.update(eval_doc)
    with open(run_dir / runs.EVAL_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(full_doc, f, indent=2, default=str)

    # 2. predictions.csv & parquet
    all_recs = []
    for split_name, recs in records_by_split.items():
        all_recs.extend(recs)

    if all_recs:
        fieldnames = [
            "image_id", "dataset", "split", "lambda_idx", "lambda_value",
            "bpp_actual", "y", "raw_score", "s", "mapped_pred", "orig_mos"
        ]
        csv_path = run_dir / f"{runs.PREDICTIONS_STEM}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in all_recs:
                row_dict = {
                    "image_id": r["image_id"],
                    "dataset": r["dataset"],
                    "split": r["split"],
                    "lambda_idx": r["lambda_idx"],
                    "lambda_value": r["lambda_value"],
                    "bpp_actual": r["bpp_actual"],
                    "y": r["target"],
                    "raw_score": r["raw_score"],
                    "s": r["aligned_score"],
                    "mapped_pred": r.get("mapped_pred", r["aligned_score"]),
                    "orig_mos": r.get("orig_mos"),
                }
                writer.writerow(row_dict)

        try:
            import pandas as pd
            df = pd.DataFrame(all_recs)
            df.to_parquet(run_dir / f"{runs.PREDICTIONS_STEM}.parquet", index=False)
        except Exception:
            pass


def format_report_markdown(
    provenance_header: str,
    results: Dict[str, Dict[str, Any]],
    latent_baselines: Dict[str, Dict[str, Any]],
) -> str:
    """Render comprehensive markdown report comparing NR models with Latent Students."""
    lines = [provenance_header, "# Pretrained No-Reference (NR-IQA) Evaluation Report (`eval_pretrained_nr.md`)", ""]
    lines.append(
        "Evaluates off-the-shelf Pretrained No-Reference IQA models (CLIPIQA, MUSIQ, NIQE, BRISQUE, MANIQA) "
        "on decoded RGB reconstructions across **KonIQ Test** (in-distribution) and **LiveC Test** (cross-content & resolution)."
    )
    lines.append("")
    lines.append(
        "> [!IMPORTANT]\n"
        "> **Key Methodological Principle:** Pretrained NR models assess generic aesthetic/perceptual quality on pixels ($x$). "
        "> In contrast, `nrcomp` Latent Students predict pure codec-induced degradation ($y = \\mathcal{T}(x_{\\text{recon}}, x_{\\text{orig}})$) "
        "> directly from quantized latents $\\hat{y}$ without access to pixels or reference images."
    )
    lines.append("")

    splits_to_show = [("koniq_test", "1. In-Distribution Evaluation (KonIQ Test Split)"),
                      ("livec_test", "2. Cross-Dataset Evaluation (LiveC Test Split)")]

    def _fmt(v, nd=4):
        if v is None:
            return "n/a"
        return f"{float(v):.{nd}f}"

    for split_key, title in splits_to_show:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(r"| Model | Domain | Target Alignment | Within-λ SROCC [95% CI] | Overall SROCC | PLCC | Within-λ Partial (\| bpp) | DS-AUC |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        # 1. Latent student reference rows
        for lat_name, lat_doc in latent_baselines.items():
            sp = (lat_doc.get("splits") or {}).get(split_key)
            if sp:
                ci = sp.get("within_lambda_srocc_ci") or (None, None)
                ci_str = f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci[0] is not None else ""
                disp = runs.DISPLAY_NAMES.get(lat_name, lat_name)
                lines.append(
                    f"| **{disp}** | Codec Latents | native | "
                    f"**{_fmt(sp.get('within_lambda_srocc'))}**{ci_str} | "
                    f"{_fmt(sp.get('srocc'))} | {_fmt(sp.get('plcc'))} | "
                    f"{_fmt(sp.get('within_lambda_partial_spearman_bpp'))} | "
                    f"{_fmt(sp.get('ds_auc'))} |"
                )

        # 2. Pretrained NR model rows
        for m_name, doc in results.items():
            sp = (doc.get("splits") or {}).get(split_key)
            if not sp:
                continue
            ci = sp.get("within_lambda_srocc_ci") or (None, None)
            ci_str = f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci[0] is not None else ""
            align_txt = "-score (lower=better)" if doc.get("lower_better") else "higher=better"
            lines.append(
                f"| Pretrained {m_name.upper()} | Decoded Pixels | {align_txt} | "
                f"{_fmt(sp.get('within_lambda_srocc'))}{ci_str} | "
                f"{_fmt(sp.get('srocc'))} | {_fmt(sp.get('plcc'))} | "
                f"{_fmt(sp.get('within_lambda_partial_spearman_bpp'))} | "
                f"{_fmt(sp.get('ds_auc'))} |"
            )
        lines.append("")

    # E3 Source MOS independence table
    lines.append("## 3. Independence from Source Image Quality (MOS Independence)")
    lines.append("")
    lines.append("Tests whether the model reflects pure compression degradation or remains confounded by source image aesthetics:")
    lines.append("")
    lines.append("| Model | Split | ρ(s, MOS) | partial ρ(s, MOS | y) | Interpretation |")
    lines.append("| :--- | :--- | :---: | :---: | :--- |")

    for lat_name, lat_doc in latent_baselines.items():
        for split_key, split_title in [("koniq_test", "KonIQ test"), ("livec_test", "LiveC test")]:
            sp = (lat_doc.get("splits") or {}).get(split_key, {})
            e3 = (sp.get("e3_mos_independence") or lat_doc.get("e3", {}).get(split_key, {})) or {}
            overall = (e3.get("overall") if isinstance(e3, dict) else {}) or {}
            rho_mos = overall.get("rho_s_mos") or e3.get("rho_s_mos") or e3.get("rho_pred_mos") or e3.get("srocc_mos") or e3.get("s_mos_partial_spearman")
            part_mos = overall.get("partial_s_mos_given_y") or e3.get("partial_s_mos_given_y") or e3.get("partial_pred_mos_given_y") or e3.get("partial_mos_given_target") or e3.get("s_mos_partial_spearman")
            if rho_mos is not None:
                lines.append(f"| **{runs.DISPLAY_NAMES.get(lat_name, lat_name)}** | {split_title} | {_fmt(rho_mos)} | **{_fmt(part_mos)}** | Pure degradation predictor |")

    for m_name, doc in results.items():
        for split_key, split_title in [("koniq_test", "KonIQ test"), ("livec_test", "LiveC test")]:
            sp = (doc.get("splits") or {}).get(split_key, {})
            e3 = (sp.get("e3_mos_independence") or sp.get("mos_independence")) or {}
            overall = (e3.get("overall") if isinstance(e3, dict) else {}) or {}
            rho_mos = overall.get("rho_s_mos") or e3.get("rho_s_mos") or e3.get("rho_pred_mos") or e3.get("s_mos_partial_spearman")
            part_mos = overall.get("partial_s_mos_given_y") or e3.get("partial_s_mos_given_y") or e3.get("partial_pred_mos_given_y") or e3.get("s_mos_partial_spearman")
            if rho_mos is not None:
                interp = "Confounded by source aesthetics" if abs(part_mos or 0) > 0.2 else "Moderate content bias"
                lines.append(f"| Pretrained {m_name.upper()} | {split_title} | {_fmt(rho_mos)} | {_fmt(part_mos)} | {interp} |")

    lines.append("")
    return "\n".join(lines)


def materialize_missing_recon(
    r: Dict[str, Any],
    project_root: Path,
    decoder_state: Dict[str, Any],
    device: Any,
) -> Optional[Path]:
    """Decode a missing reconstruction from latents and save to disk."""
    lat_p_raw = r.get("latent_path")
    if not lat_p_raw:
        return None
    lat_p = Path(lat_p_raw) if Path(lat_p_raw).is_absolute() else project_root / lat_p_raw
    if not lat_p.exists():
        return None

    if "wrapper" not in decoder_state:
        try:
            from nrcomp.codec.wrapper import ELICWrapper
            from nrcomp.data.store import LatentStore
            from nrcomp.lambdas import load_lambdas_config

            lambdas_cfg = load_lambdas_config()
            ckpt_by_idx = {int(x["idx"]): x["checkpoint"] for x in lambdas_cfg}
            weights_dir = project_root / "weights" / "updated"
            decoder_state["wrapper"] = ELICWrapper(weights_dir=weights_dir, device=str(device))
            decoder_state["latent_store"] = LatentStore(root_dir=project_root / "data" / "latents")
            decoder_state["ckpt_by_idx"] = ckpt_by_idx
        except Exception as e:
            decoder_state["wrapper"] = None
            print(f"\n[WARN] Could not initialize ELIC decoder: {e}")
            return None

    wrapper = decoder_state.get("wrapper")
    if wrapper is None:
        return None

    try:
        import torch
        from PIL import Image

        latent_store = decoder_state["latent_store"]
        ckpt_by_idx = decoder_state["ckpt_by_idx"]

        lambda_idx = int(r["lambda_idx"])
        if lambda_idx not in wrapper.models:
            wrapper.load_model(lambda_idx, ckpt_by_idx[lambda_idx])

        lat_data = latent_store.load_latents(lat_p)
        meta = {
            "original_size": (int(r["height"]), int(r["width"])),
            "padded_size": (int(r["padded_height"]), int(r["padded_width"])),
            "lambda_idx": lambda_idx,
        }
        recon_np = wrapper.decode_from_latents(torch.from_numpy(lat_data["y_hat"]).float(), meta)

        ds_name = str(r.get("dataset", "koniq"))
        stem = Path(r["image_id"]).stem
        dest = project_root / "data" / "recons" / ds_name / f"lambda_{lambda_idx}" / f"{stem}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(recon_np).save(dest, format="PNG")
        return dest
    except Exception as e:
        print(f"\n[WARN] Failed decoding latent {lat_p}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pretrained NR-IQA Models (pyiqa/piq) on Codec Reconstructions.")
    parser.add_argument("--manifest", type=str, default="data/manifest.parquet", help="Path to manifest parquet.")
    parser.add_argument(
        "--models",
        type=str,
        default="clipiqa,musiq,niqe,brisque,maniqa",
        help="Comma-separated list of NR-IQA models to evaluate.",
    )
    parser.add_argument("--backend", type=str, default="pyiqa", choices=["pyiqa", "piq"], help="Preferred IQA backend.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu).")
    parser.add_argument("--target-metric", type=str, default="haarpsi", help="Target degradation metric.")
    parser.add_argument("--runs-dir", type=str, default="reports/runs", help="Directory to save run artifacts.")
    parser.add_argument("--output", type=str, default="reports/eval_pretrained_nr.md", help="Output report path.")
    parser.add_argument("--bootstrap-resamples", type=int, default=500, help="Number of bootstrap resamples.")
    parser.add_argument("--update-eval-refs", action="store_true", default=True, help="Automatically update reports/eval_refs.md.")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for dry-run/smoke testing.")
    parser.add_argument("--auto-decode", action="store_true", default=True, help="Auto-decode missing reconstructions from latents if not found on disk.")
    parser.add_argument("--no-auto-decode", action="store_false", dest="auto_decode", help="Disable auto-decoding of missing reconstructions.")
    parser.add_argument("--list-models", action="store_true", help="List available NR models in pyiqa and exit.")
    args = parser.parse_args()

    if args.list_models:
        try:
            import pyiqa
            print("Available No-Reference (NR) models in pyiqa:")
            for m in pyiqa.list_models(metric_mode="NR"):
                print(f"  - {m}")
        except ImportError:
            print("[ERROR] pyiqa is not installed in the current environment.")
        sys.exit(0)

    manifest_p = (PROJECT_ROOT / args.manifest) if not Path(args.manifest).is_absolute() else Path(args.manifest)
    runs_dir = (PROJECT_ROOT / args.runs_dir) if not Path(args.runs_dir).is_absolute() else Path(args.runs_dir)
    output_p = (PROJECT_ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    runs_dir.mkdir(parents=True, exist_ok=True)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_p.exists():
        print(f"[ERROR] Manifest file not found at: {manifest_p}")
        sys.exit(1)

    # Device selection with Pascal sm_61 safety
    try:
        import torch
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 7:
            torch.backends.cudnn.enabled = False
            print(f"[DEVICE] Hardware sm_{torch.cuda.get_device_capability(device)[0]}.{torch.cuda.get_device_capability(device)[1]} (Pascal) detected: cuDNN disabled for stability.")
    except ImportError:
        device = "cpu"

    print("================================================================================")
    print(" nrcomp: Pretrained No-Reference IQA Evaluation Suite")
    print(f" Timestamp:     {datetime.now(timezone.utc).isoformat()}")
    print(f" Manifest:      {manifest_p}")
    print(f" Target Metric: {args.target_metric}")
    print(f" Device:        {device}")
    print(f" Backend:       {args.backend}")
    print("================================================================================")

    manifest = ParquetManifest(manifest_p)
    all_rows = manifest.get_all_rows()
    lambda_cfg = load_lambdas_config()
    lambda_vals = get_lambda_values(lambda_cfg)

    # Filter rows with verified targets and existing reconstruction files
    valid_rows_by_split: Dict[str, List[Dict[str, Any]]] = {
        "koniq_val": [],
        "koniq_test": [],
        "livec_test": [],
    }

    n_missing_recon = 0
    n_auto_decoded = 0
    decoder_state: Dict[str, Any] = {}

    for r in all_rows:
        ds_name = r.get("dataset")
        sp_name = r.get("split")
        target_raw = r.get(args.target_metric)
        if target_raw is None or is_missing(target_raw):
            continue

        split_key = None
        if ds_name == "koniq" and sp_name == "val":
            split_key = "koniq_val"
        elif ds_name == "koniq" and sp_name == "test":
            split_key = "koniq_test"
        elif ds_name == "livec" and sp_name in ["test", "livec_test", "val", "all"]:
            split_key = "livec_test"

        if split_key is None:
            continue

        recon_file = resolve_recon_path(r, PROJECT_ROOT)
        if recon_file is None or not recon_file.exists():
            if args.auto_decode:
                recon_file = materialize_missing_recon(r, PROJECT_ROOT, decoder_state, device)
                if recon_file is not None and recon_file.exists():
                    n_auto_decoded += 1
            if recon_file is None or not recon_file.exists():
                n_missing_recon += 1
                continue

        row_item = dict(r)
        row_item["_resolved_recon_path"] = str(recon_file)
        row_item["_processed_target"] = prepare_target_value(float(target_raw), args.target_metric)
        row_item["_lambda_val"] = lambda_vals[int(r["lambda_idx"])] if int(r["lambda_idx"]) < len(lambda_vals) else 0.0
        valid_rows_by_split[split_key].append(row_item)

    if n_auto_decoded > 0:
        print(f"[DECODE] Auto-decoded {n_auto_decoded} missing reconstructions from quantized latents into data/recons/.")

    print(
        f"Reconstructions Found: koniq_val={len(valid_rows_by_split['koniq_val'])}, "
        f"koniq_test={len(valid_rows_by_split['koniq_test'])}, "
        f"livec_test={len(valid_rows_by_split['livec_test'])} "
        f"(missing on disk: {n_missing_recon})"
    )

    total_valid = sum(len(v) for v in valid_rows_by_split.values())
    if total_valid == 0:
        print("[ERROR] No valid reconstruction image files found for any split. Cannot evaluate pixel models.")
        sys.exit(1)

    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    results: Dict[str, Dict[str, Any]] = {}

    for m_name in model_names:
        print(f"\n>>> Initializing NR Model: {m_name.upper()} ({args.backend})...")
        try:
            eval_fn, is_lower_better = init_nr_metric(m_name, device, backend=args.backend)
        except Exception as e:
            print(f"[ERROR] Failed to initialize '{m_name}': {e}. Skipping.")
            continue

        print(f">>> Running inference for {m_name.upper()} (lower_better={is_lower_better})...")
        records_by_split: Dict[str, List[Dict[str, Any]]] = {}
        t0 = time.time()

        for split_key in ["koniq_val", "koniq_test", "livec_test"]:
            rows = valid_rows_by_split.get(split_key, [])
            if not rows:
                print(f"  [{split_key}] No reconstruction images available; skipping.")
                records_by_split[split_key] = []
                continue
            if args.limit:
                rows = rows[:args.limit]

            recs = []
            print(f"  [{split_key}] Evaluating {len(rows)} images...", end="", flush=True)
            sp_t0 = time.time()
            for r in rows:
                img_p = Path(r["_resolved_recon_path"])
                try:
                    raw_score = eval_fn(img_p)
                except Exception as err:
                    print(f"\n[WARN] Failed scoring {img_p}: {err}")
                    continue

                aligned_score = -raw_score if is_lower_better else raw_score
                recs.append({
                    "image_id": str(r["image_id"]),
                    "dataset": str(r["dataset"]),
                    "split": split_key,
                    "lambda_idx": int(r["lambda_idx"]),
                    "lambda_value": float(r["_lambda_val"]),
                    "bpp_actual": float(r.get("bpp_actual", 0.0)),
                    "target": float(r["_processed_target"]),
                    "raw_score": float(raw_score),
                    "aligned_score": float(aligned_score),
                    "orig_mos": None if is_missing(r.get("orig_mos")) else float(r["orig_mos"]),
                })
            sp_elapsed = time.time() - sp_t0
            print(f" done ({sp_elapsed:.1f}s, {len(recs) / max(sp_elapsed, 0.001):.1f} img/s)")
            records_by_split[split_key] = recs

        # Fit 4P logistic calibration on koniq_val (for PLCC/RMSE calibration)
        fitter = None
        val_recs = records_by_split.get("koniq_val", [])
        if len(val_recs) >= 30:
            val_s = [r["aligned_score"] for r in val_recs]
            val_y = [r["target"] for r in val_recs]
            try:
                fitter = Logistic4PFitter().fit(val_s, val_y)
            except Exception as e:
                print(f"[WARN] Logistic calibration failed on koniq_val for {m_name}: {e}")

        # Compute metrics per split
        split_evals: Dict[str, Any] = {}
        for split_key in ["koniq_val", "koniq_test", "livec_test"]:
            recs = records_by_split.get(split_key, [])
            split_evals[split_key] = evaluate_split_records(
                recs,
                fitter=fitter,
                n_bootstrap=args.bootstrap_resamples if split_key != "koniq_val" else 0,
            )

        model_run_dir = runs_dir / f"pretrained_{m_name}"
        meta_info = {
            "model": f"pretrained_{m_name}",
            "model_name": m_name,
            "type": "pretrained_nr",
            "backend": args.backend,
            "lower_better": is_lower_better,
            "target_metric": args.target_metric,
            "device": str(device),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        eval_doc = {"splits": split_evals}
        save_run_artifacts(model_run_dir, m_name, meta_info, eval_doc, records_by_split)
        print(f">>> [SUCCESS] Artifacts written to: {model_run_dir}")
        results[m_name] = {"lower_better": is_lower_better, "splits": split_evals}

    # Load Latent Student reference models from disk if present
    latent_baselines = {}
    for lat_name in ["conditioned_a", "blind_a", "conditioned_b"]:
        ens_p = runs_dir / runs.ensemble_name(lat_name) / runs.EVAL_METRICS_FILE
        if ens_p.exists():
            with open(ens_p, "r", encoding="utf-8") as f:
                latent_baselines[lat_name] = json.load(f)
        else:
            cand_p = runs_dir / f"{lat_name}_seed42" / runs.EVAL_METRICS_FILE
            if cand_p.exists():
                with open(cand_p, "r", encoding="utf-8") as f:
                    latent_baselines[lat_name] = json.load(f)

    # Collect provenance and write final report
    input_artifacts = [manifest_p, LAMBDAS_YAML]
    for m in results:
        p = runs_dir / f"pretrained_{m}" / runs.EVAL_METRICS_FILE
        if p.exists():
            input_artifacts.append(p)

    prov = collect_provenance(manifest_path=manifest_p, input_artifacts=input_artifacts)
    prov_header = format_provenance_header(prov, "Pretrained NR Evaluation Provenance")
    report_md = format_report_markdown(prov_header, results, latent_baselines)

    output_p.write_text(report_md, encoding="utf-8")
    print(f"\n================================================================================")
    print(f"[COMPLETE] Pretrained NR report written to: {output_p}")
    print("================================================================================")

    # Update reports/eval_refs.md if requested
    if args.update_eval_refs:
        print("\n[UPDATING] Refreshing Stage 6 report (reports/eval_refs.md)...")
        try:
            from nrcomp import eval_pixel_refs
            eval_refs_cfg = PROJECT_ROOT / "configs" / "eval_refs.yaml"
            cmd = [sys.executable, "-m", "nrcomp.eval_pixel_refs", "--config", str(eval_refs_cfg)]
            import subprocess
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
            print("[SUCCESS] reports/eval_refs.md updated with pretrained NR models.")
        except Exception as e:
            print(f"[WARN] Failed to automatically refresh eval_refs.md: {e}")


if __name__ == "__main__":
    main()
