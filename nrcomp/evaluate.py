"""Stage 5: Evaluation & Diagnostic Suite for nrcomp.

CLI:
    python -m nrcomp.evaluate --config configs/evaluate.yaml

For every trained config and seed found under checkpoints/{config}_seed{N}/best_model.pth:
  * runs inference on koniq_val (calibration only), koniq_test and livec_test,
  * writes per-seed artifacts to reports/runs/{config}_seed{N}/ (eval_metrics.json, predictions.*),
  * builds the SEED ENSEMBLE (mean of raw predictions) and evaluates it with 95% image-clustered
    bootstrap CIs, E3 (orig_mos independence), cross-teacher agreement and valid-cell fractions,
    written to reports/runs/{config}_ensemble/.
Shortcut baselines are fit on koniq_train only and written to reports/runs/baseline_*/.
Finally reports/eval_main.md is generated strictly from these artifacts.

Integrity rules enforced here (design.md Section 0):
  * The manifest must pass scripts/audit_manifest.py before anything is evaluated.
  * A checkpoint trained on a different manifest (SHA-256 mismatch) is refused.
  * splits are never pooled; koniq_val is used only to fit the logistic calibration.
  * Missing data is reported as `n/a` / `NOT RUN`, never as 0.0; no conclusions are hardcoded.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp import runs
from nrcomp.baselines import (
    BppBaseline,
    BppLambdaBaseline,
    LambdaBaseline,
    LatentStatsBaseline,
    PerLambdaBppBaseline,
    compute_within_lambda_srocc,
    partial_spearman,
    pearson_r,
    spearman_rho,
)
from nrcomp.config import load_config
from nrcomp.data.manifest import ParquetManifest, is_missing
from nrcomp.data.metadata import KONIQ_MOS_SCALE, LIVEC_MOS_SCALE
from nrcomp.data.store import LatentStore
from nrcomp.eval_metrics import (  # noqa: F401  (re-exported for tests / other scripts)
    MIN_PAIRS_PER_BIN,
    PWRC_THRESHOLDS,
    Logistic4PFitter,
    compute_clustered_bootstrap_cis,
    compute_cross_teacher,
    compute_mean_within_image_srocc,
    compute_mos_independence,
    compute_paired_difference_test,
    compute_pwrc_within_lambda,
    compute_rmse,
    compute_sample_mean_std,
    compute_second_teacher_agreement,
    compute_teacher_agreement_matrix,
    compute_within_lambda_partial_spearman,
    ensemble_records,
    logistic_4p,
)
from nrcomp.labels.registry import get_metric_spec, prepare_target_value
from nrcomp.lambdas import get_lambda_names, get_lambda_values, load_lambdas_config
from nrcomp.provenance import (
    LAMBDAS_YAML,
    collect_provenance,
    format_provenance_header,
    git_sha,
    sha256_file,
)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

get_file_sha256 = sha256_file
get_git_commit_sha = git_sha

SEED_STAT_METRICS = [
    "within_lambda_srocc",
    "within_lambda_partial_spearman_bpp",
    "ds_auc",
    "srocc",
    "plcc",
    "rmse",
]
SPLIT_KEYS = ("koniq_test", "livec_test")
SPLIT_TITLES = {"koniq_test": "KonIQ test", "livec_test": "LiveC test"}
PRIMARY_BASELINES = list(runs.BASELINE_MODELS)


# ==============================================================================
# 1. Provenance + prediction artifacts
# ==============================================================================

def build_provenance_header(
    config_path: Path,
    manifest_path: Path,
    lambdas_path: Path,
    seed: Union[int, List[int]],
    input_artifacts: Optional[List[Path]] = None,
) -> str:
    """Mandatory provenance header (design.md Section 0, rule 2); hashes are full SHA-256."""
    prov = collect_provenance(
        config_path=config_path, manifest_path=manifest_path, seeds=seed, input_artifacts=input_artifacts
    )
    prov["lambdas_path"] = str(lambdas_path)
    prov["lambdas_sha256"] = sha256_file(lambdas_path)
    return format_provenance_header(prov, title="Evaluation Provenance & Integrity Metadata")


PREDICTION_FIELDS = [
    "run", "image_id", "dataset", "split", "lambda_idx", "lambda_value",
    "y", "s", "mapped_pred", "bpp", "orig_mos", "haarpsi", "dists", "psnr",
]


def save_standardized_predictions(
    records: List[Dict[str, Any]],
    output_path: Union[str, Path],
    run: Optional[str] = None,
) -> None:
    """Write predictions.csv (+ .parquet when pandas/pyarrow are available).

    Missing values stay EMPTY/None -- they are never replaced by 0.0.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lambda_values = None

    def _opt(v: Any) -> Optional[float]:
        return None if is_missing(v) else float(v)

    clean_rows = []
    for r in records:
        lam_val = r.get("lambda_value")
        if is_missing(lam_val):
            if lambda_values is None:
                lambda_values = get_lambda_values()
            lam_val = lambda_values[int(r["lambda_idx"])]
        clean_rows.append({
            "run": run if run is not None else r.get("run", ""),
            "image_id": str(r["image_id"]),
            "dataset": str(r.get("dataset", "")),
            "split": str(r.get("split", "")),
            "lambda_idx": int(r["lambda_idx"]),
            "lambda_value": float(lam_val),
            "y": float(r.get("target", r.get("y"))),
            "s": float(r.get("raw_pred", r.get("s"))),
            "mapped_pred": _opt(r.get("mapped_pred")),
            "bpp": _opt(r.get("bpp_actual", r.get("bpp"))),
            "orig_mos": _opt(r.get("orig_mos")),
            "haarpsi": _opt(r.get("haarpsi")),
            "dists": _opt(r.get("dists")),
            "psnr": _opt(r.get("psnr")),
        })

    if HAS_PANDAS:
        try:
            pd.DataFrame(clean_rows, columns=PREDICTION_FIELDS).to_parquet(
                output_path.with_suffix(".parquet"), index=False
            )
        except ImportError:
            pass  # pyarrow missing: the CSV below is the authoritative artifact

    with open(output_path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(clean_rows)


def load_predictions_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a predictions.csv back into evaluation records (None for empty cells)."""
    out = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def num(k: str) -> Optional[float]:
                v = row.get(k)
                return None if v in (None, "") else float(v)

            out.append({
                "run": row.get("run", ""),
                "image_id": row["image_id"],
                "dataset": row.get("dataset", ""),
                "split": row.get("split", ""),
                "lambda_idx": int(row["lambda_idx"]),
                "lambda_value": num("lambda_value"),
                "target": num("y"),
                "raw_pred": num("s"),
                "mapped_pred": num("mapped_pred"),
                "bpp_actual": num("bpp"),
                "orig_mos": num("orig_mos"),
                "haarpsi": num("haarpsi"),
                "dists": num("dists"),
                "psnr": num("psnr"),
            })
    return out


# ==============================================================================
# 2. Model loading, inference, latency
# ==============================================================================

def load_student_model(
    checkpoint_path: Path,
    device: Any,
    expected_manifest_sha256: Optional[str] = None,
    allow_manifest_mismatch: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Load a trained LatentStudent. Refuses a checkpoint trained on a different manifest."""
    import torch
    from nrcomp.model.student import LatentStudent

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # older torch without weights_only
        checkpoint = torch.load(checkpoint_path, map_location=device)

    trained_sha = checkpoint.get("manifest_sha256")
    if expected_manifest_sha256 and trained_sha and trained_sha != expected_manifest_sha256:
        msg = (
            f"{checkpoint_path}: trained on manifest {trained_sha[:12]}..., but the current manifest is "
            f"{expected_manifest_sha256[:12]}...; the splits may differ."
        )
        if not allow_manifest_mismatch:
            raise RuntimeError(msg + " Retrain, or pass --allow-manifest-mismatch to override (not for reported results).")
        print(f"[WARN] {msg}")

    eff = checkpoint.get("effective_config")
    if eff is None:
        from nrcomp.train import resolve_effective_config

        eff = resolve_effective_config(checkpoint.get("config", {}))
    m = eff["model"]
    model = LatentStudent(
        config=m["config"],
        in_channels_y=int(m["in_channels_y"]),
        in_channels_z=int(m["in_channels_z"]),
        width=int(m["width"]),
        num_res_blocks=int(m["num_res_blocks"]),
        lambda_mode=m["lambda_mode"],
        num_lambdas=int(m["num_lambdas"]),
        lambda_embed_dim=int(m["lambda_embed_dim"]),
        dropout=float(m["dropout"]),
        use_mask=bool(m["use_mask"]),
        kernel_size=int(m.get("kernel_size", 7)),
        expansion=int(m.get("expansion", 4)),
        layer_scale_init=float(m.get("layer_scale_init", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])  # normalization buffers are part of the state
    model.to(device)
    model.eval()
    return model, checkpoint


def run_student_inference(model: Any, dataloader: Any, device: Any) -> List[Dict[str, Any]]:
    """fp32 inference; returns minimal records (manifest fields are attached afterwards)."""
    import torch

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            y_hat = batch["y_hat"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            lambda_idx = batch["lambda_indices"].to(device, non_blocking=True)
            z_hat = batch.get("z_hat")
            if z_hat is not None:
                z_hat = z_hat.to(device, non_blocking=True)

            preds = model(y_hat=y_hat, z_hat=z_hat, valid_mask=valid_mask, lambda_idx=lambda_idx)
            preds_list = preds.float().cpu().tolist()
            targets_list = batch["targets"].float().cpu().tolist()
            n_valid = valid_mask.reshape(valid_mask.shape[0], -1).sum(dim=1).cpu().tolist()
            lambdas = batch["lambda_indices"].cpu().tolist()
            for i in range(len(preds_list)):
                records.append({
                    "image_id": str(batch["image_ids"][i]),
                    "dataset": str(batch["dataset"][i]),
                    "lambda_idx": int(lambdas[i]),
                    "raw_pred": float(preds_list[i]),
                    "target": float(targets_list[i]),
                    "n_valid_cells": int(n_valid[i]),
                })
    return records


def attach_manifest_fields(
    records: List[Dict[str, Any]],
    row_index: Dict[Tuple[str, str, int], Dict[str, Any]],
    split_name: str,
    lambda_values: List[float],
) -> None:
    """Fill labels/metadata from the manifest (single source of truth). A missing row raises."""
    for r in records:
        key = (str(r["dataset"]), str(r["image_id"]), int(r["lambda_idx"]))
        row = row_index[key]
        r["split"] = split_name
        r["lambda_value"] = float(lambda_values[int(r["lambda_idx"])])
        r["bpp_actual"] = float(row["bpp_actual"])
        r["orig_mos"] = None if is_missing(row.get("orig_mos")) else float(row["orig_mos"])
        r["haarpsi"] = None if is_missing(row.get("haarpsi")) else float(row["haarpsi"])
        r["dists"] = None if is_missing(row.get("dists")) else float(row["dists"])
        r["psnr"] = None if is_missing(row.get("psnr")) else float(row["psnr"])
        r["n_total_cells"] = (int(row["padded_height"]) // 16) * (int(row["padded_width"]) // 16)


def benchmark_gpu_latency(
    model: Any,
    config_type: str = "A",
    device: Any = None,
    warmup_runs: int = 20,
    test_runs: int = 100,
) -> Dict[str, Any]:
    """Latency in ms/image at batch sizes 1 and 16 for a 1024x768 image (latent grid 64x48)."""
    import torch

    num_params = sum(p.numel() for p in model.parameters())
    info: Dict[str, Any] = {"params_millions": num_params / 1e6, "torch_version": torch.__version__}
    if device is None or device.type != "cuda":
        info.update({"latency_bs1_ms": None, "latency_bs16_ms": None, "throughput_img_per_sec": None,
                     "device_name": None, "note": "no CUDA device: latency not measured"})
        return info

    info["device_name"] = torch.cuda.get_device_name(device)
    h, w = 64, 48

    def make(bs: int):
        y = torch.zeros((bs, 320, h, w), device=device)
        m = torch.ones((bs, h, w), dtype=torch.bool, device=device)
        l = torch.zeros((bs,), dtype=torch.long, device=device)
        z = torch.zeros((bs, 192, h // 4, w // 4), device=device) if config_type == "B" else None
        return y, m, l, z

    def time_it(bs: int) -> float:
        y, m, l, z = make(bs)
        with torch.no_grad():
            for _ in range(warmup_runs):
                model(y_hat=y, z_hat=z, valid_mask=m, lambda_idx=l)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(test_runs):
                model(y_hat=y, z_hat=z, valid_mask=m, lambda_idx=l)
            end.record()
            torch.cuda.synchronize()
        return start.elapsed_time(end) / test_runs

    bs1_ms = time_it(1)
    bs16_ms = time_it(16) / 16
    info.update({
        "latency_bs1_ms": bs1_ms,
        "latency_bs16_ms": bs16_ms,
        "throughput_img_per_sec": (1000.0 / bs16_ms) if bs16_ms > 0 else None,
    })
    return info


# ==============================================================================
# 3. Per-split evaluation
# ==============================================================================

def evaluate_model_on_split(
    records: List[Dict[str, Any]],
    fitter: Logistic4PFitter,
    n_resamples: int = 1000,
    seed: int = 42,
    min_pairs: int = MIN_PAIRS_PER_BIN,
    compute_ds_auc_ci: bool = True,
    confidence_level: float = 0.95,
) -> Optional[Dict[str, Any]]:
    """Full metric battery on ONE split. Returns None when the split has no samples.

    `n_resamples = 0` skips the bootstrap (used for per-seed point estimates).
    """
    if not records:
        return None
    raw = [r["raw_pred"] for r in records]
    tgt = [r["target"] for r in records]
    mapped = fitter.predict(raw)
    for r, mp in zip(records, mapped):
        r["mapped_pred"] = mp

    lam = [int(r["lambda_idx"]) for r in records]
    bpps = [float(r["bpp_actual"]) for r in records]
    img_ids = [str(r["image_id"]) for r in records]

    wl_srocc, per_lambda = compute_within_lambda_srocc(raw, tgt, lam)
    wl_part, per_l_part = compute_within_lambda_partial_spearman(raw, tgt, bpps, lam)
    pw = compute_pwrc_within_lambda(raw, tgt, lam, min_pairs=min_pairs)
    wi, _ = compute_mean_within_image_srocc(raw, tgt, img_ids)

    cis: Dict[str, Any] = {}
    if n_resamples > 0:
        cis = compute_clustered_bootstrap_cis(
            records, raw_pred_key="raw_pred", mapped_pred_key="mapped_pred", target_key="target",
            n_resamples=n_resamples, confidence_level=confidence_level, seed=seed,
            min_pairs=min_pairs, compute_ds_auc_ci=compute_ds_auc_ci,
        )

    imgs_per_lambda: Dict[int, set] = defaultdict(set)
    for r in records:
        imgs_per_lambda[int(r["lambda_idx"])].add(str(r["image_id"]))

    return {
        "n_samples": len(records),
        "n_images": len(set(img_ids)),
        "n_lambdas": len(set(lam)),
        "n_images_per_lambda": {l: len(s) for l, s in sorted(imgs_per_lambda.items())},
        "within_lambda_srocc": wl_srocc,
        "within_lambda_srocc_ci": cis.get("within_lambda_srocc"),
        "per_lambda_srocc": per_lambda,
        "within_lambda_partial_spearman_bpp": wl_part,
        "per_lambda_partial_spearman_bpp": per_l_part,
        "pwrc": pw,
        "ds_auc": pw["ds_auc"],
        "ds_auc_ci": cis.get("ds_auc"),
        "srocc": spearman_rho(raw, tgt),
        "srocc_ci": cis.get("srocc"),
        "plcc": pearson_r(mapped, tgt),
        "plcc_ci": cis.get("plcc"),
        "rmse": compute_rmse(mapped, tgt),
        "rmse_ci": cis.get("rmse"),
        "partial_spearman_bpp": partial_spearman(raw, tgt, bpps),
        "mean_within_image_srocc": wi,
    }


def apply_calibration(records: List[Dict[str, Any]], fitter: Logistic4PFitter) -> None:
    if records:
        for r, mp in zip(records, fitter.predict([r["raw_pred"] for r in records])):
            r["mapped_pred"] = mp


def valid_cell_fraction(records: List[Dict[str, Any]]) -> Optional[float]:
    """Mean fraction of latent cells that are valid (not codec padding), measured from the data."""
    fr = [r["n_valid_cells"] / r["n_total_cells"] for r in records
          if r.get("n_valid_cells") is not None and r.get("n_total_cells")]
    return (sum(fr) / len(fr)) if fr else None


def evaluate_run_splits(
    split_records: Dict[str, List[Dict[str, Any]]],
    n_resamples: int,
    seed: int,
    min_pairs: int,
    compute_ds_auc_ci: bool,
) -> Dict[str, Any]:
    """Calibrate on koniq_val, then evaluate koniq_test / livec_test (+ val for transparency)."""
    val = split_records.get("koniq_val", [])
    fitter = Logistic4PFitter().fit([r["raw_pred"] for r in val], [r["target"] for r in val])
    out: Dict[str, Any] = {
        "calibration": {"status": fitter.status, "n_fit_samples": fitter.n_fit},
        "splits": {},
        "e3": {},
        "cross_teacher": {},
        "valid_cell_fraction": {},
    }
    apply_calibration(val, fitter)
    if val:
        out["splits"]["koniq_val"] = evaluate_model_on_split(val, fitter, 0, seed, min_pairs, False)
    for key in SPLIT_KEYS:
        recs = split_records.get(key, [])
        out["splits"][key] = evaluate_model_on_split(
            recs, fitter, n_resamples, seed, min_pairs, compute_ds_auc_ci
        )
        if recs:
            out["e3"][key] = compute_mos_independence(
                [r["raw_pred"] for r in recs], [r["target"] for r in recs],
                [r.get("orig_mos") for r in recs], [int(r["lambda_idx"]) for r in recs],
                mae_preds=[r["mapped_pred"] for r in recs],
            )
            out["cross_teacher"][key] = compute_cross_teacher(recs)
            out["valid_cell_fraction"][key] = valid_cell_fraction(recs)
        else:
            out["e3"][key] = None
            out["cross_teacher"][key] = None
            out["valid_cell_fraction"][key] = None
    return out


# ==============================================================================
# 4. Evaluation context, baselines and students
# ==============================================================================

@dataclass
class EvalContext:
    project_root: Path
    manifest_path: Path
    manifest_sha256: str
    cfg_path: Path
    rows: List[Dict[str, Any]]
    row_index: Dict[Tuple[str, str, int], Dict[str, Any]]
    train_rows: List[Dict[str, Any]]
    val_rows: List[Dict[str, Any]]
    test_rows: List[Dict[str, Any]]
    livec_rows: List[Dict[str, Any]]
    latent_store: Any
    lambda_values: List[float]
    n_resamples: int = 1000
    seed: int = 42
    min_pairs: int = MIN_PAIRS_PER_BIN
    ds_auc_ci_models: Optional[List[str]] = None
    allow_manifest_mismatch: bool = False
    device: Any = None


def build_context(cfg: Dict[str, Any], cfg_path: Path, manifest_path: Path, args: argparse.Namespace) -> EvalContext:
    rows = ParquetManifest(manifest_path).get_all_rows()
    keep = [r for r in rows if r.get("split") != runs.SPLIT_EXCLUDED]
    latent_dir = Path(cfg.get("latent_store_dir", "data/latents"))
    if not latent_dir.is_absolute():
        latent_dir = PROJECT_ROOT / latent_dir
    ev = cfg.get("evaluation", {}) or {}
    return EvalContext(
        project_root=PROJECT_ROOT,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        cfg_path=cfg_path,
        rows=keep,
        row_index={(str(r["dataset"]), str(r["image_id"]), int(r["lambda_idx"])): r for r in keep},
        train_rows=[r for r in keep if r["dataset"] == "koniq" and r["split"] == "train"],
        val_rows=[r for r in keep if r["dataset"] == "koniq" and r["split"] == "val"],
        test_rows=[r for r in keep if r["dataset"] == "koniq" and r["split"] == "test"],
        livec_rows=[r for r in keep if r["dataset"] == "livec" and r["split"] == "test"],
        latent_store=LatentStore(latent_dir),
        lambda_values=get_lambda_values(),
        n_resamples=int(args.n_resamples if args.n_resamples is not None else ev.get("bootstrap_resamples", 1000)),
        seed=int(ev.get("seed", 42)),
        min_pairs=int(ev.get("min_pairs_per_bin", MIN_PAIRS_PER_BIN)),
        ds_auc_ci_models=list(ev.get("ds_auc_ci_models", cfg.get("primary_models", runs.PRIMARY_MODELS))),
        allow_manifest_mismatch=bool(args.allow_manifest_mismatch),
    )


def _save_run(
    run_dir: Path,
    name: str,
    kind: str,
    payload: Dict[str, Any],
    split_records: Dict[str, List[Dict[str, Any]]],
    ctx: EvalContext,
    seeds: List[int],
    artifacts: List[Path],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = {"run": name, "kind": kind}
    doc.update(payload)
    if extra:
        doc.update(extra)
    doc["provenance"] = collect_provenance(
        config_path=ctx.cfg_path, manifest_path=ctx.manifest_path, seeds=seeds, input_artifacts=artifacts
    )
    with open(run_dir / runs.EVAL_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, default=str)
    all_recs = [r for key in ("koniq_val", "koniq_test", "livec_test") for r in split_records.get(key, [])]
    save_standardized_predictions(all_recs, run_dir / runs.PREDICTIONS_STEM, run=name)


def evaluate_baselines(ctx: EvalContext, target_metric: str = "haarpsi") -> Tuple[Dict[str, Any], Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """Fit baselines on koniq_train only; evaluate on koniq_val / koniq_test / livec_test."""
    from scripts.shortcut_baselines import extract_latent_summary_features

    def tgt(r: Dict[str, Any]) -> Optional[float]:
        if is_missing(r.get(target_metric)):
            return None
        return prepare_target_value(float(r[target_metric]), target_metric)

    train = [r for r in ctx.train_rows if tgt(r) is not None]
    y = [tgt(r) for r in train]
    bpp = [float(r["bpp_actual"]) for r in train]
    lam = [int(r["lambda_idx"]) for r in train]

    def feats(r: Dict[str, Any]) -> List[float]:
        lp = Path(r["latent_path"])
        if not lp.is_absolute():
            lp = PROJECT_ROOT / lp
        return extract_latent_summary_features(ctx.latent_store.load_latents(lp), float(r["bpp_actual"]))

    print(f"[EVAL] Fitting shortcut baselines on {len(train)} koniq_train samples...")
    m_bpp = BppBaseline(); m_bpp.fit(bpp, y)
    m_lam = LambdaBaseline(); m_lam.fit(lam, y)
    m_stats = LatentStatsBaseline(); m_stats.fit([feats(r) for r in train], y)
    m_pl = PerLambdaBppBaseline(); m_pl.fit(bpp, lam, y)
    m_bl = BppLambdaBaseline(); m_bl.fit(bpp, lam, y)

    def predict(name: str, rows: List[Dict[str, Any]]) -> List[float]:
        b = [float(r["bpp_actual"]) for r in rows]
        l = [int(r["lambda_idx"]) for r in rows]
        if name == "baseline_bpp":
            return m_bpp.predict(b)
        if name == "baseline_lambda":
            return m_lam.predict(l)
        if name == "baseline_stats":
            return m_stats.predict([feats(r) for r in rows])
        if name == "baseline_bpp_per_lambda":
            return m_pl.predict(b, l)
        if name == "baseline_bpp_lambda":
            return m_bl.predict(b, l)
        raise KeyError(name)

    split_rows = {"koniq_val": ctx.val_rows, "koniq_test": ctx.test_rows, "livec_test": ctx.livec_rows}
    results: Dict[str, Any] = {}
    all_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for name in PRIMARY_BASELINES:
        print(f"[EVAL] Evaluating {name}...")
        split_records: Dict[str, List[Dict[str, Any]]] = {}
        for split_key, rows in split_rows.items():
            usable = [r for r in rows if tgt(r) is not None]
            preds = predict(name, usable)
            recs = [{
                "image_id": str(r["image_id"]), "dataset": str(r["dataset"]), "lambda_idx": int(r["lambda_idx"]),
                "raw_pred": float(p), "target": float(tgt(r)),
            } for r, p in zip(usable, preds)]
            attach_manifest_fields(recs, ctx.row_index, split_key, ctx.lambda_values)
            split_records[split_key] = recs
        body = evaluate_run_splits(
            split_records, ctx.n_resamples, ctx.seed, ctx.min_pairs, compute_ds_auc_ci=True
        )
        res = {
            "key": name, "kind": "baseline", "target_metric": target_metric, "seeds_found": [], "seeds_missing": [],
            "seed_stats": {}, "efficiency": None,
        }
        res.update(body)
        results[name] = res
        all_records[name] = split_records
        _save_run(runs.runs_dir(PROJECT_ROOT) / name, name, "baseline", body, split_records, ctx, [], [],
                  extra={"target_metric": target_metric})
    return results, all_records


def seed_stat_block(per_seed: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """mean / sample-std / n of the headline metrics across seeds, per split."""
    stats: Dict[str, Any] = {}
    for split in SPLIT_KEYS:
        stats[split] = {}
        for metric in SEED_STAT_METRICS:
            vals = {}
            for seed, body in per_seed.items():
                sp = body["splits"].get(split)
                if sp is not None and sp.get(metric) is not None:
                    vals[seed] = sp[metric]
            mean, std = compute_sample_mean_std(list(vals.values()))
            stats[split][metric] = {"mean": mean, "std": std, "n": len(vals), "values": vals}
    return stats


def evaluate_student_config(
    config_name: str,
    seeds: List[int],
    ctx: EvalContext,
    measure_latency: bool,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Evaluate every available seed of one config, then the seed ensemble."""
    import torch
    from torch.utils.data import DataLoader
    from nrcomp.data.dataset import LatentDataset, collate_latent_samples

    per_seed_records: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    per_seed_body: Dict[int, Dict[str, Any]] = {}
    ckpts: Dict[int, Path] = {}
    seeds_missing: List[int] = []
    target_metric = "haarpsi"
    model_config_type = "A"
    efficiency: Optional[Dict[str, Any]] = None

    for seed in seeds:
        ckpt_path = runs.best_checkpoint_path(config_name, seed, PROJECT_ROOT)
        if not ckpt_path.exists():
            seeds_missing.append(seed)
            print(f"[EVAL] {runs.run_name(config_name, seed)}: checkpoint not found ({ckpt_path}) -> NOT RUN")
            continue
        print(f"[EVAL] Evaluating {runs.run_name(config_name, seed)} from {ckpt_path}...")
        model, ckpt = load_student_model(
            ckpt_path, ctx.device, expected_manifest_sha256=ctx.manifest_sha256,
            allow_manifest_mismatch=ctx.allow_manifest_mismatch,
        )
        target_metric = str(ckpt.get("target_metric", "haarpsi"))
        model_config_type = model.config
        ckpts[seed] = ckpt_path

        if measure_latency and efficiency is None:
            efficiency = benchmark_gpu_latency(model, config_type=model_config_type, device=ctx.device)

        split_defs = [
            ("koniq_val", ctx.val_rows, "koniq", "val"),
            ("koniq_test", ctx.test_rows, "koniq", "test"),
            ("livec_test", ctx.livec_rows, "livec", "test"),
        ]
        split_records: Dict[str, List[Dict[str, Any]]] = {}
        for split_key, rows, dataset_name, split in split_defs:
            ds = LatentDataset(
                rows, latent_store=ctx.latent_store, split=split, config=model_config_type,
                target_metric=target_metric, dataset_name=dataset_name,
            )
            loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_latent_samples, num_workers=0)
            recs = run_student_inference(model, loader, ctx.device)
            attach_manifest_fields(recs, ctx.row_index, split_key, ctx.lambda_values)
            split_records[split_key] = recs
        per_seed_records[seed] = split_records

        body = evaluate_run_splits(split_records, 0, ctx.seed, ctx.min_pairs, compute_ds_auc_ci=False)
        per_seed_body[seed] = body
        _save_run(
            runs.run_dir(config_name, seed, PROJECT_ROOT), runs.run_name(config_name, seed), "seed", body,
            split_records, ctx, [seed], [ckpt_path],
            extra={"config_name": config_name, "seed": seed, "target_metric": target_metric},
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not per_seed_records:
        return None, {}

    # ---- seed ensemble -------------------------------------------------------------------
    found = sorted(per_seed_records)
    ens_records: Dict[str, List[Dict[str, Any]]] = {}
    ens_info: Dict[str, Any] = {}
    for split_key in ("koniq_val", "koniq_test", "livec_test"):
        recs, info = ensemble_records({s: per_seed_records[s][split_key] for s in found})
        ens_records[split_key] = recs
        ens_info[split_key] = info
    with_ds_ci = config_name in (ctx.ds_auc_ci_models or [])
    body = evaluate_run_splits(ens_records, ctx.n_resamples, ctx.seed, ctx.min_pairs, compute_ds_auc_ci=with_ds_ci)
    stats = seed_stat_block(per_seed_body)
    result = {
        "key": config_name, "kind": "student", "target_metric": target_metric,
        "model_config": model_config_type, "seeds_found": found, "seeds_missing": seeds_missing,
        "ensemble_info": ens_info, "seed_stats": stats, "efficiency": efficiency,
    }
    result.update(body)
    _save_run(
        runs.ensemble_dir(config_name, PROJECT_ROOT), runs.ensemble_name(config_name), "ensemble",
        body, ens_records, ctx, found, [ckpts[s] for s in found],
        extra={"config_name": config_name, "seeds": found, "seeds_missing": seeds_missing,
               "target_metric": target_metric, "ensemble_info": ens_info, "seed_stats": stats,
               "efficiency": efficiency},
    )
    return result, ens_records


# ==============================================================================
# 5. Report formatting (strictly data driven)
# ==============================================================================

def fmt(v: Any, nd: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_ci(v: Any, ci: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if not ci or ci[0] is None or ci[1] is None:
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def fmt_seed_stat(stat: Optional[Dict[str, Any]], nd: int = 4) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    return f"{stat['mean']:.{nd}f} ± {stat['std']:.{nd}f} (n={stat['n']})"


def _split_metrics(res: Dict[str, Any], split: str) -> Optional[Dict[str, Any]]:
    if not isinstance(res, dict):
        return None
    splits = res.get("splits")
    if isinstance(splits, dict) and split in splits:
        return splits.get(split)
    if split == "koniq_test" and "e1" in res:
        return res.get("e1")
    if split == "livec_test" and "e2" in res:
        return res.get("e2")
    return None


def _display(key: str) -> str:
    return runs.DISPLAY_NAMES.get(key, key)


def _model_order(results: Dict[str, Dict[str, Any]], model_order: Optional[List[str]]) -> List[str]:
    if model_order is not None:
        return list(model_order)
    default_keys = runs.BASELINE_MODELS + runs.ALL_MODELS + ["student_blind_a", "student_conditioned_a", "student_conditioned_b"]
    known = [k for k in default_keys if k in results]
    extras = [k for k in results if k not in known]
    return known + extras


def format_markdown_report(
    results: Dict[str, Dict[str, Any]],
    provenance_header: Optional[Union[str, Dict[str, Any]]] = None,
    paired_tests: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    teacher_matrices: Optional[Dict[str, Dict[str, Any]]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    model_order: Optional[List[str]] = None,
    run_date_utc: Optional[str] = None,
) -> str:
    """Render eval_main.md from evaluation results. Nothing here is hardcoded: every number
    comes from `results`; absent data renders as `n/a` / `NOT RUN`."""
    lines: List[str] = []
    order = _model_order(results, model_order)
    lam_items = load_lambdas_config()
    lam_idx = [int(it["idx"]) for it in lam_items]
    lam_cols = [f"λ{it['idx']} ({it['name']})" for it in lam_items]

    if provenance_header:
        if isinstance(provenance_header, dict):
            if "git_sha" in provenance_header or "manifest_sha256" in provenance_header:
                lines.append(format_provenance_header(provenance_header, "Evaluation Provenance & Integrity Metadata"))
            elif any(isinstance(v, dict) and ("params_millions" in v or "latency_bs1_ms" in v) for v in provenance_header.values()):
                # It's an efficiency_dict passed from old caller
                for m_name, eff_data in provenance_header.items():
                    if m_name in results and isinstance(results[m_name], dict):
                        results[m_name]["efficiency"] = eff_data
            else:
                lines.append(format_provenance_header(provenance_header, "Evaluation Provenance & Integrity Metadata"))
        else:
            lines.append(str(provenance_header))
    lines.append("# Stage 5: Main Evaluation Report (`eval_main.md`)")
    lines.append("")
    if run_date_utc:
        lines.append(f"**Generated (UTC):** {run_date_utc}  ")
    lines.append("**Confidence intervals:** 95% percentile bootstrap, resampling SOURCE IMAGES (all λ of an image together).  ")
    lines.append("**Headline metric:** Within-Lambda SROCC (within-λ SROCC, mean over λ of the Spearman correlation inside each λ).  ")
    lines.append("**Student rows:** point estimate and CI are for the SEED ENSEMBLE (mean of raw predictions over seeds); the "
                 "'seed mean ± std' column is over individual seeds.  ")
    lines.append("**Calibration:** 4-parameter logistic fit on koniq_val only (PLCC/RMSE only; rank metrics use raw predictions).  ")
    lines.append(f"**MOS scales:** KonIQ {KONIQ_MOS_SCALE}; LiveC {LIVEC_MOS_SCALE}. Datasets are never pooled.  ")
    lines.append("")

    # ---- 0. data + run completeness ----------------------------------------------------
    lines.append("## 0. Data & Run Completeness")
    lines.append("")
    if dataset_info:
        lines.append("| Split | Images | Samples (images × λ) |")
        lines.append("| :--- | :---: | :---: |")
        for key, info in dataset_info.items():
            lines.append(f"| {key} | {info.get('images', 'n/a')} | {info.get('samples', 'n/a')} |")
        lines.append("")
    lines.append("| Model | Seeds found | Seeds missing | Calibration (koniq_val fit) |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for key in order:
        res = results.get(key)
        if res is None:
            lines.append(f"| {_display(key)} | **NOT RUN** | - | - |")
            continue
        cal = res.get("calibration") or {}
        found = res.get("seeds_found") or []
        miss = res.get("seeds_missing") or []
        found_txt = str(found) if res.get("kind") == "student" else "n/a (deterministic)"
        lines.append(f"| {_display(key)} | {found_txt} | {miss if miss else '-'} | "
                     f"{cal.get('status', 'n/a')} (n={cal.get('n_fit_samples', 'n/a')}) |")
    lines.append("")

    # ---- 1/2. E1 + E2 -------------------------------------------------------------------
    for sec_no, (split, label) in enumerate([("koniq_test", "E1: In-Distribution (KonIQ test)"),
                                              ("livec_test", "E2: Cross-Content & Cross-Resolution (LiveC test)")], start=1):
        lines.append("---")
        lines.append("")
        lines.append(f"## {sec_no}. {label}")
        lines.append("")
        any_metrics = next((_split_metrics(results[k], split) for k in order if k in results and _split_metrics(results[k], split)), None)
        if any_metrics and any_metrics.get("n_samples") is not None:
            n_samp = any_metrics.get("n_samples", 0)
            n_img = any_metrics.get("n_images", 0)
            n_lam = any_metrics.get("n_lambdas", 6)
            lines.append(f"{n_samp:,} samples = {n_img:,} images × {n_lam} λ.")
            lines.append("")
        lines.append("| Model | Within-Lambda SROCC [95% CI] | Seed mean ± std | Within-λ partial (s,y \\| bpp) | DS-AUC [95% CI] | DS-AUC thresholds supported |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for key in order:
            res = results.get(key)
            m = _split_metrics(res, split) if res else None
            if m is None:
                lines.append(f"| {_display(key)} | **NOT RUN** | - | - | - | - |")
                continue
            stat = (res.get("seed_stats") or {}).get(split, {}).get("within_lambda_srocc")
            sup = (m.get("pwrc") or {}).get("supported_max_threshold")
            lines.append(
                f"| {_display(key)} | {fmt_ci(m.get('within_lambda_srocc'), m.get('within_lambda_srocc_ci'))} | "
                f"{fmt_seed_stat(stat) if res.get('kind') == 'student' else 'n/a'} | "
                f"{fmt(m.get('within_lambda_partial_spearman_bpp'))} | {fmt_ci(m.get('ds_auc'), m.get('ds_auc_ci'))} | "
                f"{'T ≤ ' + fmt(sup, 2) if sup is not None else 'n/a'} |"
            )
        lines.append("")
        lines.append(f"### {sec_no}.1 Within-λ SROCC per λ")
        lines.append("")
        lines.append("| Model | Mean | " + " | ".join(lam_cols) + " |")
        lines.append("| :--- | :---: | " + " | ".join([":---:"] * len(lam_cols)) + " |")
        for key in order:
            res = results.get(key)
            m = _split_metrics(res, split) if res else None
            if m is None:
                continue
            per = m.get("per_lambda_srocc") or {}
            vals = [fmt(per.get(i, per.get(str(i)))) for i in lam_idx]
            lines.append(f"| {_display(key)} | {fmt(m['within_lambda_srocc'])} | " + " | ".join(vals) + " |")
        lines.append("")
        lines.append(f"### {sec_no}.2 Pairwise concordance (PWRC) within λ")
        lines.append("")
        lines.append("Accuracy over pairs with |y_i − y_j| ≥ T (ties in s count 0.5). `n/a` = fewer pairs than the minimum per threshold.")
        lines.append("")
        thr_cols = [0.0, 0.05, 0.10, 0.15, 0.20]
        lines.append("| Model | " + " | ".join(f"Acc (T={t:.2f}) [pairs]" for t in thr_cols) + " |")
        lines.append("| :--- | " + " | ".join([":---:"] * len(thr_cols)) + " |")
        for key in order:
            res = results.get(key)
            m = _split_metrics(res, split) if res else None
            if m is None:
                continue
            curve = (m.get("pwrc") or {}).get("curve") or {}
            counts = (m.get("pwrc") or {}).get("pair_counts") or {}

            def cell(t: float) -> str:
                v = curve.get(t, curve.get(str(t)))
                c = counts.get(t, counts.get(str(t)))
                return f"{fmt(v)} [{int(c) if c is not None else 'n/a'}]"

            lines.append(f"| {_display(key)} | " + " | ".join(cell(t) for t in thr_cols) + " |")
        lines.append("")
        pts = (paired_tests or {}).get(split) or []
        if pts:
            lines.append(f"### {sec_no}.3 Paired difference tests (within-λ SROCC, image-clustered bootstrap)")
            lines.append("")
            lines.append("| Comparison (A − B) | A | B | Δ [95% CI] | two-sided p | p < 0.05 and CI excludes 0 |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
            for pt in pts:
                lines.append(
                    f"| {pt.get('comparison', '')} | {fmt(pt.get('val_a'))} | {fmt(pt.get('val_b'))} | "
                    f"{fmt_ci(pt.get('delta_obs'), pt.get('ci'))} | {fmt(pt.get('p_value'), 3)} | "
                    f"{'**YES**' if pt.get('significant') else 'no'} |"
                )
            lines.append("")

    # ---- 3. E3 ---------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## 3. E3: Independence from Source-Image MOS (`orig_mos`)")
    lines.append("")
    lines.append("Rank correlations use RAW predictions. Teacher ρ(y, MOS) is the reference level.")
    lines.append("")
    for split in SPLIT_KEYS:
        lines.append(f"### {SPLIT_TITLES[split]}")
        lines.append("")
        lines.append("| Model | ρ(s, MOS) | ρ(y, MOS) | partial ρ(s, MOS \\| y) | MAE low / mid / high MOS tertile |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for key in order:
            res = results.get(key)
            e3 = ((res or {}).get("e3") or {}).get(split)
            if res is None or e3 is None:
                continue
            if "error" in e3:
                lines.append(f"| {_display(key)} | n/a ({e3['error']}) | - | - | - |")
                continue
            ov = e3["overall"]
            tm = e3.get("tertile_mae") or {}
            tert = " / ".join(fmt(tm.get(k)) for k in ("Low (<= 33%)", "Mid (33% - 66%)", "High (> 66%)"))
            lines.append(f"| {_display(key)} | {fmt(ov['rho_s_mos'])} | {fmt(ov['rho_y_mos'])} | "
                         f"{fmt(ov['partial_s_mos_given_y'])} | {tert} |")
        lines.append("")

    # ---- 4. E4 ---------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## 4. E4: Teacher Agreement")
    lines.append("")
    for split in SPLIT_KEYS:
        tm = (teacher_matrices or {}).get(split)
        if not tm or tm.get("sample_count", 0) == 0:
            lines.append(f"### {SPLIT_TITLES[split]}: NOT RUN (no samples)")
            lines.append("")
            continue
        lines.append(f"### {SPLIT_TITLES[split]} label-teacher matrix ({tm['sample_count']} samples)")
        lines.append("")
        for title, mat_key in (("Overall SROCC", "overall"), ("Mean within-λ SROCC", "within_lambda")):
            lines.append(f"**{title}**")
            lines.append("")
            lines.append("| Metric | " + " | ".join(tm["labels"]) + " |")
            lines.append("| :--- | " + " | ".join([":---:"] * len(tm["labels"])) + " |")
            for k1, lbl in zip(tm["keys"], tm["labels"]):
                lines.append(f"| **{lbl}** | " + " | ".join(fmt(tm[mat_key][k1].get(k2)) for k2 in tm["keys"]) + " |")
            lines.append("")
        wl = tm["within_lambda"]
        lines.append(f"Measured within-λ SROCC with bpp: HaarPSI = {fmt(wl['haarpsi']['bpp'])}, "
                     f"1 − DISTS = {fmt(wl['one_minus_dists']['bpp'])}, PSNR = {fmt(wl['psnr']['bpp'])}.")
        lines.append("")

    lines.append("### Model predictions vs each teacher (within-λ SROCC)")
    lines.append("")
    lines.append("| Model (trained on) | KonIQ vs HaarPSI | KonIQ vs 1−DISTS | LiveC vs HaarPSI | LiveC vs 1−DISTS |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for key in order:
        res = results.get(key)
        if res is None:
            continue
        cells = []
        for split in SPLIT_KEYS:
            ct = ((res.get("cross_teacher") or {}).get(split)) or {}
            for teacher in ("haarpsi", "one_minus_dists"):
                t = ct.get(teacher)
                cells.append(fmt(t["within_lambda_srocc"]) if t else "n/a")
        lines.append(f"| {_display(key)} ({res.get('target_metric', 'haarpsi')}) | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- 5. E8 ---------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## 5. E8: Latent-Student Efficiency")
    lines.append("")
    lines.append("| Model | Parameters (M) | Latency BS=1 (ms) | Latency BS=16 (ms/img) | Throughput (img/s) | Device |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    any_eff = False
    for key in order:
        res = results.get(key)
        eff = (res or {}).get("efficiency")
        if not eff:
            continue
        any_eff = True
        lines.append(f"| {_display(key)} | {fmt(eff.get('params_millions'), 2)}M | {fmt(eff.get('latency_bs1_ms'), 2)} ms | "
                     f"{fmt(eff.get('latency_bs16_ms'), 2)} ms | {fmt(eff.get('throughput_img_per_sec'), 1)} | "
                     f"{eff.get('device_name') or 'n/a'} (torch {eff.get('torch_version', 'n/a')}) |")
    if not any_eff:
        lines.append("| NOT RUN | - | - | - | - | - |")
    lines.append("")
    lines.append("Decoder-synthesis and pixel-student latencies are reported in `eval_refs.md` (E6/E8), not here.")
    lines.append("")

    # ---- appendix ------------------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append("## Appendix A: Rate-confounded metrics (do NOT use as headline)")
    lines.append("")
    lines.append("Overall SROCC/PLCC are dominated by the rate point (λ alone reaches a high value). RMSE and PLCC use the "
                 "val-fitted calibration; `status` = linear/identity means the logistic fit was rejected.")
    lines.append("")
    for split in SPLIT_KEYS:
        lines.append(f"### {SPLIT_TITLES[split]}")
        lines.append("")
        lines.append("| Model | Overall SROCC [CI] | PLCC [CI] | RMSE [CI] | Calibration | Mean within-image SROCC |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for key in order:
            res = results.get(key)
            m = _split_metrics(res, split) if res else None
            if m is None:
                continue
            lines.append(
                f"| {_display(key)} | {fmt_ci(m.get('srocc'), m.get('srocc_ci'))} | {fmt_ci(m.get('plcc'), m.get('plcc_ci'))} | "
                f"{fmt_ci(m.get('rmse'), m.get('rmse_ci'))} | {(res.get('calibration') or {}).get('status', 'n/a')} | "
                f"{fmt(m.get('mean_within_image_srocc'))} |"
            )
        lines.append("")

    lines.append("## Appendix B: Valid (non-padding) latent-cell fraction")
    lines.append("")
    lines.append("A fraction of 1.0 means no codec padding exists in that split, so masking cannot matter there (ablation A7).")
    lines.append("")
    lines.append("| Model | KonIQ test | LiveC test |")
    lines.append("| :--- | :---: | :---: |")
    for key in order:
        res = results.get(key)
        vf = (res or {}).get("valid_cell_fraction") or {}
        if res is None or res.get("kind") != "student":
            continue
        lines.append(f"| {_display(key)} | {fmt(vf.get('koniq_test'))} | {fmt(vf.get('livec_test'))} |")
    lines.append("")
    return "\n".join(lines)


# ==============================================================================
# 6. Orchestration
# ==============================================================================

def build_paired_tests(
    records: Dict[str, Dict[str, List[Dict[str, Any]]]],
    n_resamples: int,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    pairs = [
        ("conditioned_a", "baseline_bpp_per_lambda"),
        ("conditioned_a", "baseline_stats"),
        ("blind_a", "baseline_stats"),
        ("conditioned_b", "baseline_stats"),
        ("conditioned_a", "blind_a"),
        ("conditioned_b", "conditioned_a"),
    ]
    out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLIT_KEYS}
    for split in SPLIT_KEYS:
        for a, b in pairs:
            ra = (records.get(a) or {}).get(split)
            rb = (records.get(b) or {}).get(split)
            if not ra or not rb:
                continue
            pt = compute_paired_difference_test(ra, rb, "within_lambda_srocc", n_resamples=n_resamples, seed=seed)
            pt["comparison"] = f"{_display(a)} − {_display(b)}"
            out[split].append(pt)
    return out


def main():
    parser = argparse.ArgumentParser(description="Stage 5 Evaluation & Diagnostic Suite")
    parser.add_argument("--config", type=str, default="configs/evaluate.yaml")
    parser.add_argument("--output", type=str, default=None, help="Override eval_main.md path.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated config names (default: from config).")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (default: from config).")
    parser.add_argument("--n-resamples", type=int, default=None, help="Override bootstrap resamples (smoke tests).")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-audit", action="store_true", help="Skip the manifest audit (never for reported results).")
    parser.add_argument("--allow-manifest-mismatch", action="store_true")
    args = parser.parse_args()

    cfg_path = (PROJECT_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config file not found: {cfg_path}")
        sys.exit(1)
    cfg = load_config(cfg_path)

    manifest_path = Path(cfg.get("manifest_path", "data/manifest.parquet"))
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    if not manifest_path.exists() and manifest_path.with_suffix(".csv").exists():
        manifest_path = manifest_path.with_suffix(".csv")
    if not manifest_path.exists():
        print(f"[ERROR] Manifest file not found: {manifest_path}")
        sys.exit(1)

    output_path = Path(args.output or cfg.get("eval_main_report", "reports/eval_main.md"))
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_audit:
        from scripts.audit_manifest import audit_manifest

        passed, errors = audit_manifest(manifest_path, check_latents_exist=False, min_train_images=100, min_eval_images=30)
        if not passed:
            print("[ERROR] Manifest audit FAILED; refusing to evaluate. Fix with scripts/rebuild_manifest_meta.py:")
            for e in errors[:20]:
                print(f"  - {e}")
            sys.exit(1)

    models = [m.strip() for m in args.models.split(",")] if args.models else list(cfg.get("models", runs.ALL_MODELS))
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [int(s) for s in cfg.get("seeds", runs.DEFAULT_SEEDS)]
    primary = list(cfg.get("primary_models", runs.PRIMARY_MODELS))

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 7:
        torch.backends.cudnn.enabled = False
    print(f"[EVAL] Stage 5 evaluation on {device}; models={models}; seeds={seeds}")

    ctx = build_context(cfg, cfg_path, manifest_path, args)
    ctx.device = device
    print(f"[EVAL] Rows: train={len(ctx.train_rows)}, koniq_val={len(ctx.val_rows)}, "
          f"koniq_test={len(ctx.test_rows)}, livec_test={len(ctx.livec_rows)} (excluded rows dropped)")

    results: Dict[str, Dict[str, Any]] = {}
    records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    if not args.skip_baselines:
        b_res, b_rec = evaluate_baselines(ctx)
        results.update(b_res)
        records.update(b_rec)

    for name in models:
        res, rec = evaluate_student_config(name, seeds, ctx, measure_latency=(not args.skip_latency))
        if res is not None:
            results[name] = res
            records[name] = rec

    paired = build_paired_tests(records, ctx.n_resamples, ctx.seed)
    teacher_matrices = {
        "koniq_test": compute_teacher_agreement_matrix(ctx.test_rows),
        "livec_test": compute_teacher_agreement_matrix(ctx.livec_rows),
    }
    dataset_info = {}
    for key, rows in (("koniq_train", ctx.train_rows), ("koniq_val", ctx.val_rows),
                      ("koniq_test", ctx.test_rows), ("livec_test", ctx.livec_rows)):
        dataset_info[key] = {"images": len({str(r['image_id']) for r in rows}), "samples": len(rows)}

    artifacts = [manifest_path, cfg_path, LAMBDAS_YAML]
    for name in models:
        for s in seeds:
            p = runs.best_checkpoint_path(name, s, PROJECT_ROOT)
            if p.exists():
                artifacts.append(p)
    prov = collect_provenance(config_path=cfg_path, manifest_path=manifest_path, seeds=seeds, input_artifacts=artifacts)
    report_order = [k for k in PRIMARY_BASELINES + primary if k in results]
    report_md = format_markdown_report(
        results,
        provenance_header=format_provenance_header(prov, "Evaluation Provenance & Integrity Metadata"),
        paired_tests=paired,
        teacher_matrices=teacher_matrices,
        dataset_info=dataset_info,
        model_order=report_order,
        run_date_utc=prov["generated_at"],
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[SUCCESS] Stage 5 evaluation complete. Report: {output_path}")


if __name__ == "__main__":
    main()
