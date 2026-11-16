"""Stage 3 Gate: Quality Label Diagnostics.

Evaluates quality metric labels in the manifest against Section 8.3 criteria:
1. Histogram & percentiles of HaarPSI per lambda (flags ceiling clustering > 0.98).
2. Share of images where HaarPSI is non-monotone in lambda (flags if > 10%).
3. Spearman rank correlation between HaarPSI and (1 - DISTS), overall and per-image.
4. Spearman rank correlation between bpp_actual and HaarPSI.
Writes output to reports/label_diagnostics.md.
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.data.manifest import ParquetManifest


def compute_ranks(values: List[float]) -> List[float]:
    """Assign fractional ranks to values (handles ties)."""
    sorted_pairs = sorted((val, idx) for idx, val in enumerate(values))
    ranks = [0.0] * len(values)

    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based rank
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(x) < 2 or len(x) != len(y):
        return 0.0

    rx = compute_ranks(x)
    ry = compute_ranks(y)

    mean_rx = sum(rx) / len(rx)
    mean_ry = sum(ry) / len(ry)

    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = sum((a - mean_rx) ** 2 for a in rx)
    den_y = sum((b - mean_ry) ** 2 for b in ry)

    den = math.sqrt(den_x * den_y)
    if den == 0:
        return 0.0
    return num / den


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Compute min, 25%, 50%, 75%, 95%, max, mean, std."""
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in s) / n) if n > 1 else 0.0

    def get_p(pct):
        idx = int(round(pct * (n - 1)))
        return s[max(0, min(n - 1, idx))]

    return {
        "min": s[0],
        "p25": get_p(0.25),
        "median": get_p(0.50),
        "p75": get_p(0.75),
        "p95": get_p(0.95),
        "max": s[-1],
        "mean": mean,
        "std": std,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Quality Label Diagnostics Report.")
    parser.add_argument("--manifest", type=str, default="data/manifest_pilot.parquet", help="Path to parquet manifest.")
    parser.add_argument("--output", type=str, default="reports/label_diagnostics.md", help="Output markdown path.")
    args = parser.parse_args()

    manifest_path = (PROJECT_ROOT / args.manifest) if not Path(args.manifest).is_absolute() else Path(args.manifest)
    output_path = (PROJECT_ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output)

    if not manifest_path.exists():
        if manifest_path.suffix == ".parquet" and manifest_path.with_suffix(".csv").exists():
            manifest_path = manifest_path.with_suffix(".csv")
        elif manifest_path.suffix == ".csv" and manifest_path.with_suffix(".parquet").exists():
            manifest_path = manifest_path.with_suffix(".parquet")
        else:
            print(f"[ERROR] Manifest not found: {manifest_path}")
            sys.exit(1)

    manifest = ParquetManifest(manifest_path)
    rows = manifest.get_all_rows()
    print(f"Loaded {len(rows)} samples from {manifest_path}.")

    def is_valid_metric(v):
        if v is None:
            return False
        try:
            f = float(v)
            return not (math.isnan(f) or math.isinf(f))
        except (ValueError, TypeError):
            return False

    # Filter rows with complete metrics
    valid_rows = [
        r for r in rows
        if is_valid_metric(r.get("haarpsi")) and is_valid_metric(r.get("dists")) and is_valid_metric(r.get("bpp_actual"))
    ]
    if len(valid_rows) == 0:
        sample_vals = {k: rows[0].get(k) for k in ["haarpsi", "dists", "psnr", "bpp_actual"]} if rows else {}
        print(f"[ERROR] No rows with complete labels (haarpsi, dists, bpp_actual) found.")
        print(f"Sample row metric values: {sample_vals}")
        sys.exit(1)

    print(f"Analyzing {len(valid_rows)} fully-labelled samples.")

    # 1. Group by lambda
    by_lambda = defaultdict(list)
    for r in valid_rows:
        by_lambda[r["lambda_idx"]].append(r)

    # 2. Group by source image
    by_image = defaultdict(dict)
    for r in valid_rows:
        img_key = (r["dataset"], r["image_id"])
        by_image[img_key][r["lambda_idx"]] = r

    # Monotonicity check across lambdas per image
    non_monotone_images = 0
    total_multi_lambda_images = 0

    per_image_spearman_dists = []
    for img_key, lambda_dict in by_image.items():
        if len(lambda_dict) >= 3:
            total_multi_lambda_images += 1
            sorted_indices = sorted(lambda_dict.keys())
            haarpsi_vals = [lambda_dict[idx]["haarpsi"] for idx in sorted_indices]
            one_minus_dists = [1.0 - lambda_dict[idx]["dists"] for idx in sorted_indices]

            # Check if strictly monotone non-decreasing in lambda_idx
            is_monotone = all(haarpsi_vals[k] <= haarpsi_vals[k + 1] + 1e-4 for k in range(len(haarpsi_vals) - 1))
            if not is_monotone:
                non_monotone_images += 1

            # Per-image Spearman between HaarPSI and (1 - DISTS)
            rho_img = spearman_rho(haarpsi_vals, one_minus_dists)
            per_image_spearman_dists.append(rho_img)

    non_monotone_rate = (non_monotone_images / total_multi_lambda_images * 100.0) if total_multi_lambda_images > 0 else 0.0

    # Overall Spearman correlations
    all_haarpsi = [r["haarpsi"] for r in valid_rows]
    all_one_minus_dists = [1.0 - r["dists"] for r in valid_rows]
    all_bpp = [r["bpp_actual"] for r in valid_rows]

    overall_spearman_dists = spearman_rho(all_haarpsi, all_one_minus_dists)
    mean_per_image_spearman_dists = (sum(per_image_spearman_dists) / len(per_image_spearman_dists)) if per_image_spearman_dists else 0.0
    overall_spearman_bpp = spearman_rho(all_bpp, all_haarpsi)

    # Check ceiling clustering (> 0.98)
    near_ceiling_count = sum(1 for val in all_haarpsi if val > 0.98)
    near_ceiling_rate = (near_ceiling_count / len(all_haarpsi) * 100.0)

    # Generate Markdown Report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Stage 3 Quality Label Diagnostics Report\n\n")
        f.write(f"- **Manifest:** `{manifest_path.name}`\n")
        f.write(f"- **Total Labelled Samples:** {len(valid_rows)}\n")
        f.write(f"- **Unique Source Images:** {len(by_image)}\n\n")

        f.write("## 1. Metric Correlations & Agreement\n\n")
        f.write("| Diagnostic Test | Target / Benchmark | Measured Value | Gate Status |\n")
        f.write("| --- | --- | --- | --- |\n")

        # Non-monotonicity gate
        mono_status = "PASS" if non_monotone_rate <= 10.0 else "WARN (>10%)"
        f.write(f"| Non-monotone HaarPSI share | <= 10.0% | {non_monotone_rate:.1f}% ({non_monotone_images}/{total_multi_lambda_images}) | **{mono_status}** |\n")

        # Ceiling clustering
        ceil_status = "PASS" if near_ceiling_rate < 30.0 else "WARN (>30%)"
        f.write(f"| Ceiling clustering (HaarPSI > 0.98) | Informational | {near_ceiling_rate:.1f}% ({near_ceiling_count}/{len(all_haarpsi)}) | **{ceil_status}** |\n")

        # Spearman with 1 - DISTS
        f.write(f"| Spearman(HaarPSI, 1 - DISTS) [Overall] | Higher = better (> 0.70) | {overall_spearman_dists:.4f} | **{'PASS' if overall_spearman_dists > 0.7 else 'CHECK'}** |\n")
        f.write(f"| Spearman(HaarPSI, 1 - DISTS) [Mean Per-Image] | Higher = better (> 0.80) | {mean_per_image_spearman_dists:.4f} | **{'PASS' if mean_per_image_spearman_dists > 0.8 else 'CHECK'}** |\n")

        # Spearman with bpp
        f.write(f"| Spearman(bpp_actual, HaarPSI) | Reference baseline | {overall_spearman_bpp:.4f} | **INFO** |\n\n")

        f.write("## 2. Per-Lambda HaarPSI Distribution\n\n")
        f.write("| Lambda Idx | Name | Count | Mean | Std | Min | 25% | Median | 75% | 95% | Max |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for l_idx in sorted(by_lambda.keys()):
            items = by_lambda[l_idx]
            l_name = items[0].get("lambda_value", str(l_idx))
            vals = [x["haarpsi"] for x in items]
            st = compute_percentiles(vals)
            f.write(
                f"| {l_idx} | {l_name} | {len(items)} | {st['mean']:.4f} | {st['std']:.4f} | "
                f"{st['min']:.4f} | {st['p25']:.4f} | {st['median']:.4f} | {st['p75']:.4f} | {st['p95']:.4f} | {st['max']:.4f} |\n"
            )

        f.write("\n## 3. Per-Lambda DISTS Distribution (Lower = Better)\n\n")
        f.write("| Lambda Idx | Name | Count | Mean | Std | Min | Median | Max |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for l_idx in sorted(by_lambda.keys()):
            items = by_lambda[l_idx]
            l_name = items[0].get("lambda_value", str(l_idx))
            vals = [x["dists"] for x in items]
            st = compute_percentiles(vals)
            f.write(f"| {l_idx} | {l_name} | {len(items)} | {st['mean']:.4f} | {st['std']:.4f} | {st['min']:.4f} | {st['median']:.4f} | {st['max']:.4f} |\n")

    print(f"Report written to: {output_path}")
    print(f"Non-monotone HaarPSI share: {non_monotone_rate:.1f}% (Gate <= 10.0%) -> {mono_status}")
    print(f"Spearman(HaarPSI, 1 - DISTS): Overall={overall_spearman_dists:.4f}, Mean Per-Image={mean_per_image_spearman_dists:.4f}")


if __name__ == "__main__":
    main()
