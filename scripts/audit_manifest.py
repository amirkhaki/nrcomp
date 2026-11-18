#!/usr/bin/env python3
"""Audit script for nrcomp manifest integrity.

Enforces Milestone M7 acceptance criteria:
1. orig_mos must not be null and must have nonzero variance within each dataset (catches constant-zero lookup bugs).
2. Every image must have all K lambdas (no missing operating points).
3. No duplicate (image_id, lambda_idx) records.
4. All latent files specified in latent_path must exist on disk (unless --skip-latent-files is passed for offline test environments).
5. Split sizes per dataset must be valid (koniq: train, val, test present; livec: test).
6. bpp_est must not be a duplicate copy of bpp_actual across all rows.

Exits with code 0 on PASS, code 1 on FAIL.
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.data.manifest import ParquetManifest
from nrcomp.lambdas import load_lambdas_config


def audit_manifest(
    manifest_path: Path,
    check_latents_exist: bool = True,
    expected_k: int = 6,
) -> Tuple[bool, List[str]]:
    """Run comprehensive integrity audit over manifest.

    Returns:
        (passed: bool, errors: List[str])
    """
    errors: List[str] = []
    if not manifest_path.exists():
        return False, [f"Manifest file not found: {manifest_path}"]

    try:
        manifest = ParquetManifest(manifest_path)
        rows = manifest.get_all_rows()
    except Exception as e:
        return False, [f"Failed to read manifest: {e}"]

    if not rows:
        return False, ["Manifest is completely empty."]

    # Load verified lambdas
    try:
        verified_lambdas = load_lambdas_config()
        expected_lambda_indices = {int(x["idx"]) for x in verified_lambdas}
    except Exception as e:
        expected_lambda_indices = set(range(expected_k))

    # 1. Duplicates check and K lambdas grouping
    image_lambdas: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    seen_pairs: Set[Tuple[str, str, int]] = set()
    rows_by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    bpp_actual_list: List[float] = []
    bpp_est_list: List[float] = []

    for idx, r in enumerate(rows):
        ds = str(r.get("dataset", "unknown"))
        img_id = str(r.get("image_id", ""))
        l_idx = int(r.get("lambda_idx", -1))
        rows_by_dataset[ds].append(r)

        # Check duplicate
        pair_key = (ds, img_id, l_idx)
        if pair_key in seen_pairs:
            errors.append(f"Duplicate record found: dataset={ds}, image_id={img_id}, lambda_idx={l_idx}")
        seen_pairs.add(pair_key)

        image_lambdas[(ds, img_id)].add(l_idx)

        # Bitrate check
        bpp_act = r.get("bpp_actual")
        bpp_est = r.get("bpp_est")
        if bpp_act is not None:
            bpp_actual_list.append(float(bpp_act))
        if bpp_est is not None:
            bpp_est_list.append(float(bpp_est))

        # Check latent file existence
        if check_latents_exist:
            l_path_str = r.get("latent_path")
            if not l_path_str:
                errors.append(f"Row {idx} ({img_id}, lambda {l_idx}) has missing latent_path.")
            else:
                lp = Path(l_path_str)
                if not lp.is_absolute():
                    lp = PROJECT_ROOT / lp
                if not lp.exists():
                    # Truncate error output to max 10 missing files to avoid flooding
                    if len([e for e in errors if "Latent file missing" in e]) < 10:
                        errors.append(f"Latent file missing on disk: {lp}")

    # 2. Check each image has all K lambdas
    for (ds, img_id), l_set in image_lambdas.items():
        missing = expected_lambda_indices - l_set
        if missing:
            errors.append(f"Image {ds}/{img_id} lacks lambdas: {sorted(missing)} (has {sorted(l_set)})")

    # 3. Check orig_mos per dataset
    for ds, ds_rows in rows_by_dataset.items():
        mos_values = []
        for r in ds_rows:
            mos = r.get("orig_mos")
            if mos is None:
                errors.append(f"Dataset {ds} has null orig_mos for image {r.get('image_id')}")
            else:
                try:
                    f_mos = float(mos)
                    if math.isnan(f_mos):
                        errors.append(f"Dataset {ds} has NaN orig_mos for image {r.get('image_id')}")
                    else:
                        mos_values.append(f_mos)
                except (ValueError, TypeError):
                    errors.append(f"Dataset {ds} has non-numeric orig_mos: {mos}")

        if mos_values:
            mean_mos = sum(mos_values) / len(mos_values)
            var_mos = sum((x - mean_mos) ** 2 for x in mos_values) / len(mos_values)
            if var_mos < 1e-6:
                errors.append(
                    f"Dataset {ds}: orig_mos has zero variance (mean={mean_mos:.4f}, var={var_mos:.2e}). "
                    f"Original MOS lookup failed or is constant!"
                )

    # 4. Check split sizes per dataset
    for ds, ds_rows in rows_by_dataset.items():
        splits_found = set(r.get("split") for r in ds_rows)
        if ds == "koniq":
            required = {"train", "val", "test"}
            missing_splits = required - splits_found
            if missing_splits:
                errors.append(f"Dataset koniq is missing required splits: {missing_splits}")
        elif ds == "livec":
            if "test" not in splits_found:
                errors.append(f"Dataset livec must have 'test' split (found: {splits_found})")

    # 5. Check bpp_est is not identical duplicate of bpp_actual across all rows
    if bpp_est_list and bpp_actual_list and len(bpp_est_list) == len(bpp_actual_list):
        exact_matches = sum(1 for a, e in zip(bpp_actual_list, bpp_est_list) if math.isclose(a, e, abs_tol=1e-7))
        if exact_matches == len(bpp_actual_list) and len(bpp_actual_list) > 10:
            errors.append(
                f"bpp_est is a 100% duplicate copy of bpp_actual ({exact_matches}/{len(bpp_actual_list)} rows match exactly). "
                f"Must drop placeholder or compute real estimate from entropy model."
            )

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Audit nrcomp manifest integrity.")
    parser.add_argument("--manifest", type=str, default="data/manifest.parquet", help="Path to manifest file.")
    parser.add_argument("--skip-latent-files", action="store_true", help="Skip checking physical existence of latent files.")
    args = parser.parse_args()

    m_path = Path(args.manifest)
    if not m_path.is_absolute():
        m_path = PROJECT_ROOT / m_path

    if not m_path.exists():
        if m_path.suffix == ".parquet" and m_path.with_suffix(".csv").exists():
            m_path = m_path.with_suffix(".csv")

    print(f"[AUDIT] Auditing manifest: {m_path}")
    passed, errors = audit_manifest(m_path, check_latents_exist=not args.skip_latent_files)

    if passed:
        print("[AUDIT PASS] All manifest integrity checks passed successfully!")
        sys.exit(0)
    else:
        print(f"[AUDIT FAIL] Found {len(errors)} integrity error(s):")
        for err in errors[:50]:  # Limit output
            print(f"  - {err}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
