#!/usr/bin/env python3
"""Backfill orig_mos and split into manifest without regenerating latents.

Resolves the critical integrity bug where orig_mos defaulted to 0.0:
- Loads ground-truth MOS for KonIQ-10k from koniq10k_scores_and_distributions.csv (or koniq_split.csv).
- Loads ground-truth MOS for LiveC from alllabels.csv / AllMOS_Release.mat.
- Preserves all latents, bitrates, and labels.
- Sets bpp_est to None (dropping duplicate placeholder).
- Atomically rewrites manifest.parquet and manifest.csv.
"""

import argparse
import csv
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.data.manifest import ParquetManifest
from nrcomp.data.splits import create_koniq_stratified_split


def load_koniq_mos_map(scores_path: Path) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Load KonIQ-10k MOS scores and generate deterministic 80/10/10 splits."""
    if not scores_path.exists():
        raise FileNotFoundError(f"KonIQ scores file not found: {scores_path}")

    items: List[Tuple[str, float]] = []
    mos_map: Dict[str, float] = {}

    with open(scores_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Match various common column names
            name = r.get("image_name", r.get("image_id", r.get("name", "")))
            if not name:
                continue
            base_name = Path(name).name
            raw_score = r.get("MOS", r.get("mos", r.get("MOS_zscore", r.get("score"))))
            if raw_score is None:
                continue
            try:
                score_f = float(raw_score)
                mos_map[name] = score_f
                mos_map[base_name] = score_f
                items.append((base_name, score_f))
            except ValueError:
                continue

    # Stratified 80/10/10 split
    split_map = create_koniq_stratified_split(items, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0)
    return mos_map, split_map


def load_livec_mos_map(scores_path: Optional[Path], data_raw_dir: Path) -> Dict[str, float]:
    """Load LiveC (CLIVE) ground truth MOS scores."""
    mos_map: Dict[str, float] = {}

    candidates = []
    if scores_path is not None:
        candidates.append(scores_path)
    candidates.extend([
        data_raw_dir / "livec" / "alllabels.csv",
        data_raw_dir / "livec" / "all_mos.csv",
        data_raw_dir / "livec" / "Data" / "alllabels.csv",
        data_raw_dir / "livec" / "Data" / "AllMOS_Release.mat",
    ])

    found_path = None
    for cand in candidates:
        if cand.exists():
            found_path = cand
            break

    if not found_path:
        print(f"[WARN] No LiveC scores file found among candidates: {[str(c) for c in candidates]}")
        return mos_map

    print(f"[BACKFILL] Loading LiveC MOS from: {found_path}")
    if found_path.suffix == ".csv":
        with open(found_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                name = r.get("image_name", r.get("image_id", r.get("name", "")))
                score = r.get("MOS", r.get("mos", r.get("score")))
                if name and score is not None:
                    try:
                        sf = float(score)
                        mos_map[name] = sf
                        mos_map[Path(name).name] = sf
                    except ValueError:
                        pass
    elif found_path.suffix == ".mat":
        try:
            import scipy.io as sio
            mat = sio.loadmat(str(found_path))
            # LiveC mat file typically contains 'AllMOS_Release' array
            mos_arr = mat.get("AllMOS_Release")
            if mos_arr is not None:
                flat_mos = mos_arr.ravel()
                # Check for photolist.txt
                photo_list_path = found_path.parent / "photolist.txt"
                if photo_list_path.exists():
                    with open(photo_list_path, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    for img_name, score in zip(lines, flat_mos):
                        mos_map[img_name] = float(score)
                        mos_map[Path(img_name).name] = float(score)
        except Exception as e:
            print(f"[WARN] Failed to load .mat LiveC scores: {e}")

    return mos_map


def main():
    parser = argparse.ArgumentParser(description="Backfill orig_mos and split into manifest.")
    parser.add_argument("--manifest", type=str, default="data/manifest.parquet", help="Manifest path to rewrite.")
    parser.add_argument("--koniq-scores", type=str, default=None, help="Path to KonIQ scores CSV.")
    parser.add_argument("--livec-scores", type=str, default=None, help="Path to LiveC scores file.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    if not manifest_path.exists():
        if manifest_path.suffix == ".parquet" and manifest_path.with_suffix(".csv").exists():
            manifest_path = manifest_path.with_suffix(".csv")
        else:
            print(f"[ERROR] Manifest not found at: {manifest_path}")
            sys.exit(1)

    print(f"[BACKFILL] Reading existing manifest from: {manifest_path}")
    manifest = ParquetManifest(manifest_path)
    rows = manifest.get_all_rows()
    print(f"[BACKFILL] Total rows to process: {len(rows)}")

    # 1. Load KonIQ MOS
    data_raw = PROJECT_ROOT / "data" / "raw"
    koniq_candidates = [
        Path(args.koniq_scores) if args.koniq_scores else None,
        data_raw / "koniq-10k" / "koniq10k_scores_and_distributions.csv",
        data_raw / "koniq10k" / "koniq10k_scores_and_distributions.csv",
        PROJECT_ROOT / "data" / "splits" / "koniq_split.csv",
    ]
    koniq_path = next((p for p in koniq_candidates if p and p.exists()), None)
    if not koniq_path:
        print("[ERROR] KonIQ scores file not found. Provide --koniq-scores path.")
        sys.exit(1)

    print(f"[BACKFILL] Loading KonIQ MOS from: {koniq_path}")
    koniq_mos, koniq_splits = load_koniq_mos_map(koniq_path)

    # 2. Load LiveC MOS
    livec_scores_path = Path(args.livec_scores) if args.livec_scores else None
    livec_mos = load_livec_mos_map(livec_scores_path, data_raw)

    # 3. Update rows
    updated_rows = []
    koniq_updated = 0
    livec_updated = 0
    missing_koniq = set()
    missing_livec = set()

    for r in rows:
        row = dict(r)
        ds = row.get("dataset", "")
        img_id = Path(row.get("image_id", "")).name

        if ds == "koniq":
            if img_id in koniq_mos:
                row["orig_mos"] = float(koniq_mos[img_id])
                row["split"] = koniq_splits.get(img_id, row.get("split", "train"))
                koniq_updated += 1
            else:
                missing_koniq.add(img_id)
        elif ds == "livec":
            row["split"] = "test"  # LiveC is strictly test
            if img_id in livec_mos:
                row["orig_mos"] = float(livec_mos[img_id])
                livec_updated += 1
            else:
                missing_livec.add(img_id)

        # Drop bpp_est placeholder duplicate
        row["bpp_est"] = None
        updated_rows.append(row)

    print(f"[BACKFILL] Updated {koniq_updated} KonIQ rows and {livec_updated} LiveC rows.")
    if missing_koniq:
        print(f"[WARN] {len(missing_koniq)} KonIQ images had no MOS match (e.g. {list(missing_koniq)[:5]})")
    if missing_livec:
        print(f"[WARN] {len(missing_livec)} LiveC images had no MOS match (e.g. {list(missing_livec)[:5]})")

    # 4. Atomic write back
    manifest.write_all_rows(updated_rows)
    print(f"[SUCCESS] Successfully updated manifest at {manifest_path}")

    # 5. Run Audit
    from scripts.audit_manifest import audit_manifest
    passed, errors = audit_manifest(manifest_path, check_latents_exist=False)
    if passed:
        print("[BACKFILL PASS] Manifest audit passed!")
    else:
        print(f"[BACKFILL WARN] Manifest audit reported {len(errors)} issues:")
        for e in errors[:10]:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
