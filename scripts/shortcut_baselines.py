"""E5 Shortcut Baselines evaluation script.

Fits baselines (a) bpp-only, (b) lambda-only, and (c) latent-stats + bpp on the koniq_train split,
and evaluates on koniq_val, koniq_test, and livec strictly separately.
Writes results with full provenance to reports/e5_baselines.md.
"""

import argparse
import datetime
import hashlib
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.baselines import (
    BppBaseline,
    LambdaBaseline,
    LatentStatsBaseline,
    compute_within_lambda_srocc,
    partial_spearman,
    pearson_r,
    spearman_rho,
)
from nrcomp.data.manifest import ParquetManifest
from nrcomp.data.store import LatentStore
from nrcomp.labels.registry import get_metric_spec, prepare_target_value
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


def extract_latent_summary_features(latent_dict: Dict[str, Any], bpp_actual: float) -> List[float]:
    """Extract summary statistics vector from latent arrays."""
    try:
        import numpy as np
        y = np.asarray(latent_dict["y_hat"], dtype=np.float32)
        flat = y.ravel()
        abs_flat = np.abs(flat)
        mean_abs = float(np.mean(abs_flat))
        std = float(np.std(flat))
        l2_energy = float(np.sqrt(np.mean(flat ** 2)))
        p10 = float(np.percentile(abs_flat, 10))
        p50 = float(np.percentile(abs_flat, 50))
        p90 = float(np.percentile(abs_flat, 90))
        near_zero_frac = float(np.mean(abs_flat < 0.5))
        return [float(bpp_actual), mean_abs, std, l2_energy, p10, p50, p90, near_zero_frac]
    except ImportError:
        # Fallback for pure Python / mock
        y = latent_dict.get("y_hat", [])
        if hasattr(y, "__iter__") and not isinstance(y, (str, bytes)):
            # Flatten
            flat = []
            stack = [y]
            while stack:
                curr = stack.pop()
                if hasattr(curr, "__iter__") and not isinstance(curr, (str, bytes)):
                    stack.extend(curr)
                else:
                    flat.append(float(curr))
            if flat:
                mean_abs = sum(abs(x) for x in flat) / len(flat)
                std = math.sqrt(sum((x - mean_abs) ** 2 for x in flat) / len(flat))
                l2 = math.sqrt(sum(x ** 2 for x in flat) / len(flat))
                s_sorted = sorted(abs(x) for x in flat)
                p10 = s_sorted[int(0.1 * len(s_sorted))]
                p50 = s_sorted[int(0.5 * len(s_sorted))]
                p90 = s_sorted[int(0.9 * len(s_sorted))]
                near_zero = sum(1 for x in flat if abs(x) < 0.5) / len(flat)
                return [float(bpp_actual), mean_abs, std, l2, p10, p50, p90, near_zero]
        return [float(bpp_actual), 0.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.1]


def evaluate_predictions(
    preds: List[float], targets: List[float], lambdas: List[int], bpps: List[float]
) -> Dict[str, Any]:
    """Compute overall SROCC, PLCC, within-lambda SROCC, and partial Spearman."""
    srocc = spearman_rho(preds, targets)
    plcc = pearson_r(preds, targets)
    mean_within_srocc, per_lambda = compute_within_lambda_srocc(preds, targets, lambdas)
    partial_rho = partial_spearman(preds, targets, bpps)

    return {
        "srocc": srocc,
        "plcc": plcc,
        "within_lambda_srocc": mean_within_srocc,
        "per_lambda_srocc": per_lambda,
        "partial_spearman_bpp": partial_rho,
    }


def main():
    parser = argparse.ArgumentParser(description="Run E5 Shortcut Baselines Evaluation.")
    parser.add_argument("--manifest", type=str, default="data/manifest.parquet", help="Path to manifest.")
    parser.add_argument("--output", type=str, default="reports/e5_baselines.md", help="Output report path.")
    parser.add_argument("--target-metric", type=str, default="haarpsi", help="Target metric column.")
    parser.add_argument("--allow-missing-latents", action="store_true", help="Allow fallback for missing latent files in mock test mode.")
    args = parser.parse_args()

    manifest_path = (PROJECT_ROOT / args.manifest) if not Path(args.manifest).is_absolute() else Path(args.manifest)
    output_path = (PROJECT_ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output)

    if not manifest_path.exists():
        if manifest_path.suffix == ".parquet" and manifest_path.with_suffix(".csv").exists():
            manifest_path = manifest_path.with_suffix(".csv")
        elif manifest_path.suffix == ".csv" and manifest_path.with_suffix(".parquet").exists():
            manifest_path = manifest_path.with_suffix(".parquet")
        else:
            print(f"[ERROR] Manifest file not found: {manifest_path}")
            sys.exit(1)

    print(f"Loading manifest from {manifest_path}...")
    manifest = ParquetManifest(manifest_path)
    rows = manifest.get_all_rows()

    metric_spec = get_metric_spec(args.target_metric)

    # Filter rows with valid targets
    valid_rows = []
    for r in rows:
        target_val = r.get(args.target_metric)
        if target_val is not None:
            try:
                tf = float(target_val)
                if not (math.isnan(tf) or math.isinf(tf)):
                    r_copy = dict(r)
                    r_copy["_processed_target"] = prepare_target_value(tf, args.target_metric)
                    valid_rows.append(r_copy)
            except (ValueError, TypeError):
                pass

    print(f"Loaded {len(valid_rows)} valid rows with target '{args.target_metric}' (transform: {metric_spec.transform}).")

    # Strictly separate splits (Never pool KonIQ test with LiveC!)
    koniq_train = [r for r in valid_rows if r.get("dataset") == "koniq" and r.get("split") == "train"]
    koniq_val = [r for r in valid_rows if r.get("dataset") == "koniq" and r.get("split") == "val"]
    koniq_test = [r for r in valid_rows if r.get("dataset") == "koniq" and r.get("split") == "test"]
    livec_test = [r for r in valid_rows if r.get("dataset") == "livec"]

    print(f"Strict Splits: koniq_train={len(koniq_train)}, koniq_val={len(koniq_val)}, koniq_test={len(koniq_test)}, livec_test={len(livec_test)}")

    latent_store = LatentStore(PROJECT_ROOT / "data" / "latents")

    # Prepare training data on koniq_train
    train_bpp = [float(r["bpp_actual"]) for r in koniq_train]
    train_lambdas = [int(r["lambda_idx"]) for r in koniq_train]
    train_y = [float(r["_processed_target"]) for r in koniq_train]

    print("Extracting summary latent statistics for Baseline (c) on koniq_train...")
    train_feats = []
    for r in koniq_train:
        lp = Path(r["latent_path"])
        if not lp.is_absolute():
            lp = PROJECT_ROOT / lp
        if not lp.exists():
            if args.allow_missing_latents:
                lat_data = {"y_hat": [0.0]}
            else:
                raise FileNotFoundError(f"Latent file missing on disk: {lp}. Cannot compute Baseline (c).")
        else:
            lat_data = latent_store.load_latents(lp)
        feats = extract_latent_summary_features(lat_data, float(r["bpp_actual"]))
        train_feats.append(feats)

    # 1. Fit Baselines
    print("Fitting Baseline (a): bpp_actual alone...")
    model_a = BppBaseline()
    model_a.fit(train_bpp, train_y)

    print("Fitting Baseline (b): lambda_idx alone...")
    model_b = LambdaBaseline()
    model_b.fit(train_lambdas, train_y)

    print("Fitting Baseline (c): latent summary statistics + bpp_actual...")
    model_c = LatentStatsBaseline()
    model_c.fit(train_feats, train_y)

    # 2. Evaluate on koniq_val, koniq_test, and livec_test separately
    eval_splits = [
        ("koniq_val", koniq_val),
        ("koniq_test", koniq_test),
        ("livec_test", livec_test),
    ]

    results_report = {}
    for split_name, sub_rows in eval_splits:
        if not sub_rows:
            continue

        bpps = [float(r["bpp_actual"]) for r in sub_rows]
        lambdas = [int(r["lambda_idx"]) for r in sub_rows]
        targets = [float(r["_processed_target"]) for r in sub_rows]

        sub_feats = []
        for r in sub_rows:
            lp = Path(r["latent_path"])
            if not lp.is_absolute():
                lp = PROJECT_ROOT / lp
            if not lp.exists():
                if args.allow_missing_latents:
                    lat_data = {"y_hat": [0.0]}
                else:
                    raise FileNotFoundError(f"Latent file missing on disk: {lp}. Cannot compute Baseline (c).")
            else:
                lat_data = latent_store.load_latents(lp)
            feats = extract_latent_summary_features(lat_data, float(r["bpp_actual"]))
            sub_feats.append(feats)

        preds_a = model_a.predict(bpps)
        preds_b = model_b.predict(lambdas)
        preds_c = model_c.predict(sub_feats)

        res_a = evaluate_predictions(preds_a, targets, lambdas, bpps)
        res_b = evaluate_predictions(preds_b, targets, lambdas, bpps)
        res_c = evaluate_predictions(preds_c, targets, lambdas, bpps)

        results_report[split_name] = {
            "a": res_a,
            "b": res_b,
            "c": res_c,
            "count": len(sub_rows),
        }

    # 3. Generate Markdown Report with Provenance Header
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# E5 Shortcut Baselines Diagnostic Report\n\n")
        f.write("## Provenance Header\n")
        f.write(f"- **Git SHA:** `{get_git_revision_hash()}`\n")
        f.write(f"- **Date:** {datetime.datetime.now().isoformat()}\n")
        f.write(f"- **Target Metric:** `{args.target_metric}` (Transform: `{metric_spec.transform}`)\n")
        f.write(f"- **Manifest Path:** `{manifest_path}`\n")
        f.write(f"- **Manifest SHA256:** `{compute_file_sha256(manifest_path)}`\n")
        f.write(f"- **Lambdas Source:** `configs/lambdas.yaml` (`{compute_file_sha256(PROJECT_ROOT / 'configs' / 'lambdas.yaml')}`)\n")
        f.write(f"- **Total Samples:** {len(valid_rows)}\n\n")

        f.write("> [!NOTE]\n")
        f.write("> **Baseline Definitions:**\n")
        f.write("> - **Baseline (a) [bpp-only]:** Fit from $bpp\\_actual \\to y$ on `koniq_train`.\n")
        f.write("> - **Baseline (b) [lambda-only]:** Per-lambda mean degradation on `koniq_train`. Constant within each lambda (within-lambda SROCC is 0.0000).\n")
        f.write("> - **Baseline (c) [latent-stats + bpp]:** Gradient boosting on summary statistics of $\\hat{y}$ plus $bpp\\_actual$ fit on `koniq_train`.\n\n")

        for s_name, res in results_report.items():
            f.write(f"## Evaluation on Split: `{s_name}` ({res['count']} samples)\n\n")
            f.write("| Baseline Model | Overall SROCC | Overall PLCC | Within-Lambda SROCC | Partial SROCC (ctl bpp) |\n")
            f.write("| --- | :---: | :---: | :---: | :---: |\n")
            f.write(f"| **(a) bpp_actual alone** | {res['a']['srocc']:.4f} | {res['a']['plcc']:.4f} | {res['a']['within_lambda_srocc']:.4f} | {res['a']['partial_spearman_bpp']:.4f} |\n")
            f.write(f"| **(b) lambda_idx alone** | {res['b']['srocc']:.4f} | {res['b']['plcc']:.4f} | {res['b']['within_lambda_srocc']:.4f} | {res['b']['partial_spearman_bpp']:.4f} |\n")
            f.write(f"| **(c) latent stats + bpp** | {res['c']['srocc']:.4f} | {res['c']['plcc']:.4f} | {res['c']['within_lambda_srocc']:.4f} | {res['c']['partial_spearman_bpp']:.4f} |\n\n")

            f.write("### Per-Lambda SROCC Breakdown\n\n")
            f.write("| Lambda Index | Baseline (a) bpp | Baseline (b) lambda | Baseline (c) stats |\n")
            f.write("| :---: | :---: | :---: | :---: |\n")
            lambdas_list = sorted(res["a"]["per_lambda_srocc"].keys())
            for l_idx in lambdas_list:
                f.write(f"| $\\lambda_{l_idx}$ | {res['a']['per_lambda_srocc'].get(l_idx, 0.0):.4f} | {res['b']['per_lambda_srocc'].get(l_idx, 0.0):.4f} | {res['c']['per_lambda_srocc'].get(l_idx, 0.0):.4f} |\n")
            f.write("\n---\n\n")

    print(f"[SUCCESS] E5 Baselines report saved to {output_path}")


if __name__ == "__main__":
    main()
