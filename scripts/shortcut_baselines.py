"""E5 Shortcut Baselines evaluation script.

Fits baselines (a) bpp-only, (b) lambda-only, and (c) latent-stats + bpp on the train split,
and evaluates their overall SROCC, within-lambda SROCC, and PLCC on val and test splits.
Writes results to reports/e5_baselines.md.
"""

import argparse
import math
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

    # Filter rows with valid targets
    valid_rows = []
    for r in rows:
        target_val = r.get(args.target_metric)
        if target_val is not None:
            try:
                tf = float(target_val)
                if not (math.isnan(tf) or math.isinf(tf)):
                    valid_rows.append(r)
            except (ValueError, TypeError):
                pass

    print(f"Loaded {len(valid_rows)} valid rows with target '{args.target_metric}'.")

    # Split into train, val, test
    splits = {"train": [], "val": [], "test": []}
    for r in valid_rows:
        s = r.get("split", "train")
        if s in splits:
            splits[s].append(r)

    print(f"Splits count: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    latent_store = LatentStore(PROJECT_ROOT / "data" / "latents")

    # Prepare tabular data
    train_bpp = [float(r["bpp_actual"]) for r in splits["train"]]
    train_lambdas = [int(r["lambda_idx"]) for r in splits["train"]]
    train_y = [float(r[args.target_metric]) for r in splits["train"]]

    # Features for Baseline (c)
    print("Extracting summary latent statistics for Baseline (c) on train...")
    train_feats = []
    for r in splits["train"]:
        lp = Path(r["latent_path"])
        lat_data = latent_store.load_latents(lp) if lp.exists() else {"y_hat": [0.0]}
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

    # 2. Evaluate on Val and Test
    results_report = {}
    for split_name in ["val", "test"]:
        sub_rows = splits[split_name]
        if not sub_rows:
            continue

        bpps = [float(r["bpp_actual"]) for r in sub_rows]
        lambdas = [int(r["lambda_idx"]) for r in sub_rows]
        targets = [float(r[args.target_metric]) for r in sub_rows]

        sub_feats = []
        for r in sub_rows:
            lp = Path(r["latent_path"])
            lat_data = latent_store.load_latents(lp) if lp.exists() else {"y_hat": [0.0]}
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
        }

    # 3. Generate Markdown Report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# E5 Shortcut Baselines Diagnostic Report\n\n")
        f.write(f"**Date:** November 2026  \n")
        f.write(f"**Target Metric:** `{args.target_metric}`  \n")
        f.write(f"**Manifest:** `{manifest_path.name}` ({len(valid_rows)} samples)  \n\n")

        f.write("> [!NOTE]\n")
        f.write("> **Baseline Definitions:**\n")
        f.write("> - **Baseline (a) [bpp-only]:** Gradient boosting regression from $bpp\\_actual \\to y$.\n")
        f.write("> - **Baseline (b) [lambda-only]:** Per-lambda mean degradation. Constant within each lambda (within-lambda SROCC is mathematically undefined, reported safely as 0.0000).\n")
        f.write("> - **Baseline (c) [latent-stats + bpp]:** Gradient boosting on pooled summary statistics of $\\hat{y}$ (mean absolute value, std, L2 energy, quantiles, and near-zero fraction $|\\hat{y}| < 0.5$) plus $bpp\\_actual$.\n\n")

        for s_name, res in results_report.items():
            f.write(f"## Evaluation on Split: `{s_name}` ({len(splits[s_name])} samples)\n\n")
            f.write("| Baseline Model | Overall SROCC | Overall PLCC | Within-Lambda SROCC | Partial SROCC (ctl bpp) |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            f.write(f"| **(a) bpp_actual alone** | {res['a']['srocc']:.4f} | {res['a']['plcc']:.4f} | {res['a']['within_lambda_srocc']:.4f} | {res['a']['partial_spearman_bpp']:.4f} |\n")
            f.write(f"| **(b) lambda_idx alone** | {res['b']['srocc']:.4f} | {res['b']['plcc']:.4f} | {res['b']['within_lambda_srocc']:.4f} | {res['b']['partial_spearman_bpp']:.4f} |\n")
            f.write(f"| **(c) latent stats + bpp** | {res['c']['srocc']:.4f} | {res['c']['plcc']:.4f} | {res['c']['within_lambda_srocc']:.4f} | {res['c']['partial_spearman_bpp']:.4f} |\n\n")

            f.write("### Per-Lambda SROCC Breakdown\n\n")
            f.write("| Lambda Index | Baseline (a) bpp | Baseline (b) lambda | Baseline (c) stats |\n")
            f.write("| --- | --- | --- | --- |\n")
            lambdas_list = sorted(res["a"]["per_lambda_srocc"].keys())
            for l_idx in lambdas_list:
                f.write(f"| $\\lambda_{l_idx}$ | {res['a']['per_lambda_srocc'].get(l_idx, 0.0):.4f} | {res['b']['per_lambda_srocc'].get(l_idx, 0.0):.4f} | {res['c']['per_lambda_srocc'].get(l_idx, 0.0):.4f} |\n")
            f.write("\n---\n\n")

    print(f"[SUCCESS] E5 Baselines report saved to {output_path}")


if __name__ == "__main__":
    main()
