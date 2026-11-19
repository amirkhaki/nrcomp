#!/usr/bin/env python3
"""Audit script for nrcomp manifest integrity (Milestone M7 gate).

Checks (any failure => exit code 1):
  1. `split` is exactly one of train|val|test|excluded (catches stray numeric codes).
  2. Excluded images have no orig_mos and are never counted as data.
  3. Every non-excluded image has all K lambdas; no duplicate (dataset, image, lambda).
  4. Per-image consistency: all lambda rows of an image share the same split and orig_mos.
  5. Labels (haarpsi, dists, psnr) and bpp_actual are present and finite on every row.
  6. orig_mos is present, finite, has non-zero variance, and no single value is shared by
     more than 2% of a dataset's images (the signature of mean/constant imputation).
  7. KonIQ has train/val/test with ~80/10/10 image counts and (optionally) minimum sizes;
     LiveC is test-only.
  8. All latent files exist on disk (skip with --skip-latent-files).
  9. bpp_est is not a duplicate copy of bpp_actual.
"""

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.data.manifest import ALLOWED_SPLITS, ParquetManifest, is_missing
from nrcomp.lambdas import load_lambdas_config

MAX_SHARED_MOS_FRACTION = 0.02


def _finite(v: Any) -> bool:
    if is_missing(v):
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def summarize_manifest(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Unique image counts per (dataset, split)."""
    per: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        per[str(r.get("dataset", "unknown"))][str(r.get("split"))].add(str(r.get("image_id", "")))
    return {ds: {sp: len(imgs) for sp, imgs in sorted(d.items())} for ds, d in sorted(per.items())}


def audit_manifest(
    manifest_path: Path,
    check_latents_exist: bool = True,
    expected_k: int = 6,
    min_train_images: int = 0,
    min_eval_images: int = 0,
) -> Tuple[bool, List[str]]:
    """Run the integrity audit. Returns (passed, errors)."""
    errors: List[str] = []
    if not manifest_path.exists():
        return False, [f"Manifest file not found: {manifest_path}"]

    try:
        rows = ParquetManifest(manifest_path).get_all_rows()
    except Exception as e:  # noqa: BLE001
        return False, [f"Failed to read manifest: {e}"]
    if not rows:
        return False, ["Manifest is completely empty."]

    try:
        expected_lambda_indices = {int(x["idx"]) for x in load_lambdas_config()}
    except Exception:  # noqa: BLE001
        expected_lambda_indices = set(range(expected_k))

    # ---- 1. split labels -------------------------------------------------------------
    bad_splits = Counter(str(r.get("split")) for r in rows if str(r.get("split")) not in ALLOWED_SPLITS)
    if bad_splits:
        errors.append(
            f"Invalid split label(s) {dict(bad_splits)} (rows). Allowed: {list(ALLOWED_SPLITS)}. "
            f"Run `python scripts/rebuild_manifest_meta.py` to repair."
        )

    seen_pairs: Set[Tuple[str, str, int]] = set()
    image_lambdas: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    image_split: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    image_mos: Dict[Tuple[str, str], Set[Optional[float]]] = defaultdict(set)
    bpp_actual_list: List[float] = []
    bpp_est_list: List[float] = []
    n_label_errors = 0
    n_missing_latent = 0

    for idx, r in enumerate(rows):
        ds = str(r.get("dataset", "unknown"))
        img_id = str(r.get("image_id", ""))
        sp = str(r.get("split"))
        try:
            l_idx = int(r.get("lambda_idx", -1))
        except (TypeError, ValueError):
            l_idx = -1

        pair_key = (ds, img_id, l_idx)
        if pair_key in seen_pairs:
            errors.append(f"Duplicate record: dataset={ds}, image_id={img_id}, lambda_idx={l_idx}")
        seen_pairs.add(pair_key)

        key = (ds, img_id)
        image_lambdas[key].add(l_idx)
        image_split[key].add(sp)
        image_mos[key].add(None if is_missing(r.get("orig_mos")) else round(float(r["orig_mos"]), 9))

        if sp == "excluded":
            continue

        for lab in ("haarpsi", "dists", "psnr", "bpp_actual"):
            if not _finite(r.get(lab)):
                n_label_errors += 1
                if n_label_errors <= 10:
                    errors.append(f"Row {idx} ({ds}/{img_id}, lambda {l_idx}): label '{lab}' is missing/non-finite")

        if _finite(r.get("bpp_actual")):
            bpp_actual_list.append(float(r["bpp_actual"]))
        if _finite(r.get("bpp_est")):
            bpp_est_list.append(float(r["bpp_est"]))

        if check_latents_exist:
            l_path_str = r.get("latent_path")
            if not l_path_str:
                errors.append(f"Row {idx} ({img_id}, lambda {l_idx}) has missing latent_path.")
            else:
                lp = Path(str(l_path_str))
                if not lp.is_absolute():
                    lp = PROJECT_ROOT / lp
                if not lp.exists():
                    n_missing_latent += 1
                    if n_missing_latent <= 10:
                        errors.append(f"Latent file missing on disk: {lp}")
    if n_label_errors > 10:
        errors.append(f"... {n_label_errors - 10} more label errors")
    if n_missing_latent > 10:
        errors.append(f"... {n_missing_latent - 10} more missing latent files")

    # ---- 3/4. completeness + per-image consistency -----------------------------------
    for (ds, img_id), l_set in image_lambdas.items():
        if image_split[(ds, img_id)] == {"excluded"}:
            continue
        missing = expected_lambda_indices - l_set
        if missing:
            errors.append(f"Image {ds}/{img_id} lacks lambdas: {sorted(missing)} (has {sorted(l_set)})")
        if len(image_split[(ds, img_id)]) > 1:
            errors.append(f"Image {ds}/{img_id} has inconsistent splits across lambdas: {sorted(image_split[(ds, img_id)])}")
        if len(image_mos[(ds, img_id)]) > 1:
            errors.append(f"Image {ds}/{img_id} has inconsistent orig_mos across lambdas")

    # ---- 2/6. MOS checks per dataset -------------------------------------------------
    datasets = sorted({ds for ds, _ in image_lambdas})
    for ds in datasets:
        mos_per_image: Dict[str, Optional[float]] = {}
        for (d, img_id), moss in image_mos.items():
            if d != ds:
                continue
            sp = next(iter(image_split[(d, img_id)]))
            if sp == "excluded":
                if any(m is not None for m in moss):
                    errors.append(f"Excluded image {ds}/{img_id} must have orig_mos=None")
                continue
            mos_per_image[img_id] = next(iter(moss)) if moss else None

        missing_mos = [i for i, m in mos_per_image.items() if m is None]
        if missing_mos:
            errors.append(f"Dataset {ds}: {len(missing_mos)} non-excluded image(s) have null orig_mos (e.g. {missing_mos[:3]})")
        vals = [m for m in mos_per_image.values() if m is not None]
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            if var < 1e-9:
                errors.append(f"Dataset {ds}: orig_mos has zero variance (all = {mean:.4f}); MOS lookup failed.")
            top_val, top_count = Counter(vals).most_common(1)[0]
            frac = top_count / len(vals)
            if top_count >= 3 and frac > MAX_SHARED_MOS_FRACTION:
                errors.append(
                    f"Dataset {ds}: orig_mos value {top_val:.6f} is shared by {top_count}/{len(vals)} images "
                    f"({frac:.1%} > {MAX_SHARED_MOS_FRACTION:.0%}); this is the signature of MOS imputation."
                )

    # ---- 7. split structure ----------------------------------------------------------
    for ds in datasets:
        imgs_by_split: Dict[str, Set[str]] = defaultdict(set)
        for (d, img_id), sps in image_split.items():
            if d == ds:
                imgs_by_split[next(iter(sps))].add(img_id)
        counts = {sp: len(v) for sp, v in imgs_by_split.items()}
        if ds == "koniq":
            required = ("train", "val", "test")
            missing_splits = [s for s in required if counts.get(s, 0) == 0]
            if missing_splits:
                errors.append(f"Dataset koniq is missing required splits: {missing_splits}")
            else:
                total = sum(counts[s] for s in required)
                train_pct, val_pct, test_pct = (counts[s] / total for s in required)
                if train_pct < 0.60 or val_pct < 0.05 or test_pct < 0.05:
                    errors.append(
                        f"Dataset koniq split distribution is invalid/unbalanced: {counts} "
                        f"(train={train_pct:.1%}, val={val_pct:.1%}, test={test_pct:.1%}). "
                        f"Expected ~80/10/10 across {total} unique images. "
                        f"Run `python scripts/rebuild_manifest_meta.py` to regenerate."
                    )
                if counts["train"] < min_train_images:
                    errors.append(f"Dataset koniq train has {counts['train']} images (< {min_train_images}).")
                for s in ("val", "test"):
                    if counts[s] < min_eval_images:
                        errors.append(f"Dataset koniq {s} has {counts[s]} images (< {min_eval_images}).")
        elif ds == "livec":
            non_test = {s: c for s, c in counts.items() if s not in ("test", "excluded")}
            if non_test:
                errors.append(f"Dataset livec must be test-only (found {non_test}).")
            if counts.get("test", 0) == 0:
                errors.append("Dataset livec has no 'test' images.")
            elif counts.get("test", 0) < min_eval_images:
                errors.append(f"Dataset livec test has {counts['test']} images (< {min_eval_images}).")

    # ---- 9. bpp_est duplicate --------------------------------------------------------
    if bpp_est_list and len(bpp_est_list) == len(bpp_actual_list) and len(bpp_actual_list) > 10:
        exact = sum(1 for a, e in zip(bpp_actual_list, bpp_est_list) if math.isclose(a, e, abs_tol=1e-7))
        if exact == len(bpp_actual_list):
            errors.append("bpp_est is a 100% duplicate copy of bpp_actual; drop it or compute a real estimate.")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Audit nrcomp manifest integrity.")
    parser.add_argument("--manifest", type=str, default="data/manifest.parquet")
    parser.add_argument("--skip-latent-files", action="store_true", help="Skip checking latent files on disk.")
    parser.add_argument("--min-train-images", type=int, default=100)
    parser.add_argument("--min-eval-images", type=int, default=30)
    args = parser.parse_args()

    m_path = Path(args.manifest)
    if not m_path.is_absolute():
        m_path = PROJECT_ROOT / m_path
    if not m_path.exists() and m_path.suffix == ".parquet" and m_path.with_suffix(".csv").exists():
        m_path = m_path.with_suffix(".csv")

    print(f"[AUDIT] Auditing manifest: {m_path}")
    passed, errors = audit_manifest(
        m_path,
        check_latents_exist=not args.skip_latent_files,
        min_train_images=args.min_train_images,
        min_eval_images=args.min_eval_images,
    )
    try:
        print("[AUDIT] Unique images per (dataset, split):", summarize_manifest(ParquetManifest(m_path).get_all_rows()))
    except Exception:  # noqa: BLE001
        pass

    if passed:
        print("[AUDIT PASS] All manifest integrity checks passed.")
        sys.exit(0)
    print(f"[AUDIT FAIL] Found {len(errors)} integrity error(s):")
    for err in errors[:50]:
        print(f"  - {err}")
    if len(errors) > 50:
        print(f"  ... and {len(errors) - 50} more errors.")
    sys.exit(1)


if __name__ == "__main__":
    main()
