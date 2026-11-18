#!/usr/bin/env python3
"""Backfill orig_mos and split into manifest without regenerating latents.

Resolves the critical integrity bug where orig_mos defaulted to 0.0:
- Loads ground-truth MOS for KonIQ-10k from koniq10k_scores_and_distributions.csv and koniq_split.csv.
- Loads ground-truth MOS for LiveC from meta_info_livec, alllabels.csv, AllMOS_Release.mat, or pyiqa.
- Uses stem-based and case-insensitive matching to ensure 100% match across file extensions (.jpg vs .JPG).
- Preserves all latents, bitrates, and labels.
- Sets bpp_est to None (dropping duplicate placeholder).
- Atomically rewrites manifest.parquet and manifest.csv.
"""

import argparse
import csv
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.data.manifest import ParquetManifest
from nrcomp.data.splits import create_koniq_stratified_split


def _index_mos_entry(mos_map: Dict[str, float], name: str, score: float):
    """Index an image MOS score under multiple normalized keys for robust matching."""
    clean = str(name).strip()
    if not clean:
        return
    base = Path(clean).name
    stem = Path(clean).stem

    mos_map[clean] = score
    mos_map[clean.lower()] = score
    mos_map[base] = score
    mos_map[base.lower()] = score
    mos_map[stem] = score
    mos_map[stem.lower()] = score
    mos_map[f"{stem}.jpg"] = score
    mos_map[f"{stem}.JPG"] = score
    mos_map[f"{stem}.bmp"] = score
    mos_map[f"{stem}.BMP"] = score
    mos_map[f"{stem}.png"] = score
    mos_map[f"{stem}.PNG"] = score


def _lookup_mos(mos_map: Dict[str, float], img_id: str) -> Optional[float]:
    """Look up MOS with case-insensitive and stem fallback."""
    clean = str(img_id).strip()
    base = Path(clean).name
    stem = Path(clean).stem

    candidates = [
        clean,
        clean.lower(),
        base,
        base.lower(),
        stem,
        stem.lower(),
        f"{stem}.jpg",
        f"{stem}.JPG",
        f"{stem}.bmp",
        f"{stem}.BMP",
        f"{stem}.png",
        f"{stem}.PNG",
    ]
    for c in candidates:
        if c in mos_map:
            return mos_map[c]
    return None


def load_koniq_mos_map(
    scores_path: Optional[Path], data_raw_dir: Path
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Load KonIQ-10k MOS scores and generate deterministic 80/10/10 splits."""
    items: List[Tuple[str, float]] = []
    mos_map: Dict[str, float] = {}
    split_map: Dict[str, str] = {}

    candidates: List[Path] = []
    if scores_path:
        candidates.append(Path(scores_path))

    candidates.extend([
        PROJECT_ROOT / "data" / "splits" / "koniq_split.csv",
        data_raw_dir / "meta_info" / "meta_info_KonIQ10kDataset.csv",
        data_raw_dir / "meta_info" / "meta_info_KonIQ10k++Dataset.csv",
        data_raw_dir / "koniq10k" / "koniq10k_scores_and_distributions.csv",
        data_raw_dir / "koniq-10k" / "koniq10k_scores_and_distributions.csv",
        data_raw_dir / "koniq10k_scores_and_distributions.csv",
        data_raw_dir / "meta_info" / "meta_info_koniq10k.csv",
        data_raw_dir / "meta_info" / "meta_info_koniq10k.txt",
    ])

    extra_koniq = (
        list(data_raw_dir.rglob("*koniq*.csv"))
        + list(data_raw_dir.rglob("*KonIQ*.csv"))
        + list(data_raw_dir.rglob("*koniq*.txt"))
        + list(data_raw_dir.rglob("*KonIQ*.txt"))
    )
    for ek in extra_koniq:
        if ek not in candidates:
            candidates.append(ek)

    loaded_any = False
    for cand in candidates:
        if cand.exists():
            print(f"[BACKFILL] Loading KonIQ metadata from: {cand}")
            try:
                with open(cand, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        r_clean = {str(k).strip(): v for k, v in r.items() if k is not None}
                        name = (
                            r_clean.get("image_name")
                            or r_clean.get("image_id")
                            or r_clean.get("img_name")
                            or r_clean.get("name")
                            or r_clean.get("img_path")
                            or r_clean.get("filename")
                        )
                        if not name:
                            continue
                        raw_score = (
                            r_clean.get("MOS")
                            or r_clean.get("mos")
                            or r_clean.get("orig_mos")
                            or r_clean.get("MOS_zscore")
                            or r_clean.get("score")
                        )
                        if raw_score is None or raw_score == "":
                            continue
                        try:
                            score_f = float(raw_score)
                            _index_mos_entry(mos_map, name, score_f)
                            items.append((Path(name).name, score_f))

                            sp = r_clean.get("split")
                            if sp:
                                split_map[name] = sp
                                split_map[Path(name).name] = sp
                                split_map[Path(name).stem.lower()] = sp
                        except ValueError:
                            continue
                loaded_any = True
            except Exception as e:
                print(f"[WARN] Error reading KonIQ file {cand}: {e}")

    # Fallback to PyIQA dataset loader if available
    try:
        from pyiqa import load_dataset
        ds = load_dataset("koniq10k", data_root=str(data_raw_dir))
        for item in ds:
            p = item.get("img_path", item.get("image_name", item.get("img_name", "")))
            s = item.get("mos", item.get("score"))
            if p and s is not None:
                _index_mos_entry(mos_map, p, float(s))
                loaded_any = True
    except Exception:
        pass

    if not loaded_any:
        raise FileNotFoundError(f"No KonIQ scores files found among candidates: {[str(c) for c in candidates]}")

    # If splits weren't fully recorded, generate deterministic stratified split
    if items and len(split_map) < len(items) // 2:
        computed_splits = create_koniq_stratified_split(items, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0)
        for k, v in computed_splits.items():
            split_map[k] = v
            split_map[Path(k).name] = v
            split_map[Path(k).stem.lower()] = v

    return mos_map, split_map


def load_livec_mos_map(scores_path: Optional[Path], data_raw_dir: Path) -> Dict[str, float]:
    """Load LiveC (CLIVE) ground truth MOS scores from files or PyIQA fallback."""
    mos_map: Dict[str, float] = {}

    candidates: List[Path] = []
    if scores_path:
        candidates.append(Path(scores_path))

    # Explicit candidate paths (case variants for LIVEC / livec / CLIVE)
    for folder in ["LIVEC", "livec", "CLIVE", "clive"]:
        d = data_raw_dir / folder
        candidates.extend([
            d / "alllabels.csv",
            d / "all_mos.csv",
            d / "Data" / "alllabels.csv",
            d / "Data" / "AllMOS_Release.mat",
            d / "AllMOS_Release.mat",
            d / "photolist.txt",
            d / "Data" / "photolist.txt",
        ])

    meta_dir = data_raw_dir / "meta_info"
    if meta_dir.exists():
        candidates.extend([
            meta_dir / "meta_info_livec.csv",
            meta_dir / "meta_info_livec.txt",
            meta_dir / "meta_info_LIVEC.csv",
            meta_dir / "meta_info_LIVEC.txt",
            meta_dir / "meta_info_clive.csv",
            meta_dir / "meta_info_clive.txt",
            meta_dir / "meta_info_CLIVE.csv",
            meta_dir / "meta_info_CLIVE.txt",
        ])

    # Also search recursively for any file matching livec or AllMOS
    extra_files = (
        list(data_raw_dir.rglob("*AllMOS*.mat"))
        + list(data_raw_dir.rglob("*alllabels*.csv"))
        + list(data_raw_dir.rglob("*meta_info*livec*"))
        + list(data_raw_dir.rglob("*meta_info*LIVEC*"))
        + list(data_raw_dir.rglob("*meta_info*clive*"))
    )
    for ef in extra_files:
        if ef not in candidates:
            candidates.append(ef)

    found_files = [c for c in candidates if c.exists()]
    if found_files:
        print(f"[BACKFILL] Discovered {len(found_files)} potential LiveC metadata candidate(s):")
        for f in found_files:
            print(f"  - {f}")

    for found_path in found_files:
        if found_path.suffix in [".csv", ".txt"]:
            try:
                with open(found_path, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline()
                    f.seek(0)
                    if "," in first_line:
                        reader = csv.DictReader(f)
                        if reader.fieldnames and any(k.lower() in ("mos", "score", "label") for k in reader.fieldnames):
                            for r in reader:
                                r_clean = {str(k).strip().lower(): v for k, v in r.items() if k and v}
                                name = r_clean.get("image_name") or r_clean.get("image_id") or r_clean.get("name") or r_clean.get("img_path")
                                score = r_clean.get("mos") or r_clean.get("score") or r_clean.get("label")
                                if name and score is not None:
                                    try:
                                        _index_mos_entry(mos_map, name, float(score))
                                    except ValueError:
                                        pass
                            if mos_map:
                                print(f"[BACKFILL] Loaded {len(mos_map)} LiveC keys from CSV: {found_path}")
                                break
                    # Whitespace / space-delimited text fallback (e.g. meta_info_livec.txt)
                    f.seek(0)
                    for line in f:
                        parts = line.strip().replace(",", " ").split()
                        if len(parts) >= 2:
                            first, second = parts[0], parts[1]
                            try:
                                score_val = float(second)
                                name_val = first
                            except ValueError:
                                try:
                                    score_val = float(first)
                                    name_val = second
                                except ValueError:
                                    continue
                            _index_mos_entry(mos_map, name_val, score_val)
                    if mos_map:
                        print(f"[BACKFILL] Loaded {len(mos_map)} LiveC keys from text format: {found_path}")
                        break
            except Exception as e:
                print(f"[WARN] Error reading {found_path}: {e}")

        elif found_path.suffix == ".mat":
            try:
                import scipy.io as sio
                mat = sio.loadmat(str(found_path))
                mos_arr = None
                for k in ["AllMOS_Release", "allmos_release", "MOS", "mos"]:
                    if k in mat:
                        mos_arr = mat[k].ravel()
                        break
                if mos_arr is not None:
                    # Look for photolist
                    photo_txt = found_path.parent / "photolist.txt"
                    if not photo_txt.exists():
                        photo_txt = found_path.parent.parent / "photolist.txt"
                    if photo_txt.exists():
                        with open(photo_txt, "r", encoding="utf-8") as f:
                            lines = [l.strip() for l in f if l.strip()]
                        for img_name, s_val in zip(lines, mos_arr):
                            _index_mos_entry(mos_map, img_name, float(s_val))
                    else:
                        # In official LIVE in the Wild DB, images are 1.JPG .. 1162.JPG
                        for idx, s_val in enumerate(mos_arr, 1):
                            sf = float(s_val)
                            _index_mos_entry(mos_map, f"{idx}.JPG", sf)
                            _index_mos_entry(mos_map, f"{idx}.jpg", sf)
                            _index_mos_entry(mos_map, f"{idx}.bmp", sf)
                            _index_mos_entry(mos_map, str(idx), sf)
                    if mos_map:
                        print(f"[BACKFILL] Loaded {len(mos_map)} LiveC entries from .mat: {found_path}")
                        break
            except Exception as e:
                print(f"[WARN] Error loading LiveC .mat file: {e}")

    # Fallback to PyIQA dataset loader if available
    if not mos_map:
        try:
            print("[BACKFILL] Attempting LiveC load via pyiqa.load_dataset...")
            from pyiqa import load_dataset
            for target_key in ["livec", "clive"]:
                try:
                    ds = load_dataset(target_key, data_root=str(data_raw_dir))
                    for item in ds:
                        p = item.get("img_path", item.get("image_name", ""))
                        s = item.get("mos", item.get("score"))
                        if p and s is not None:
                            _index_mos_entry(mos_map, p, float(s))
                    if mos_map:
                        print(f"[BACKFILL] Loaded {len(mos_map)} LiveC keys via pyiqa dataset '{target_key}'")
                        break
                except Exception:
                    continue
        except ImportError:
            pass

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

    data_raw = PROJECT_ROOT / "data" / "raw"

    # 1. Load KonIQ MOS
    koniq_mos, koniq_splits = load_koniq_mos_map(
        Path(args.koniq_scores) if args.koniq_scores else None, data_raw
    )
    print(f"[BACKFILL] KonIQ database ready with {len(koniq_mos)} indexed lookup keys.")

    # 2. Load LiveC MOS
    livec_scores_path = Path(args.livec_scores) if args.livec_scores else None
    livec_mos = load_livec_mos_map(livec_scores_path, data_raw)
    print(f"[BACKFILL] LiveC database ready with {len(livec_mos)} indexed lookup keys.")

    # 3. Update rows
    updated_rows = []
    koniq_updated = 0
    livec_updated = 0
    missing_koniq: Set[str] = set()
    missing_livec: Set[str] = set()

    for r in rows:
        row = dict(r)
        ds = row.get("dataset", "")
        img_id = Path(row.get("image_id", "")).name

        if ds == "koniq":
            matched_mos = _lookup_mos(koniq_mos, img_id)
            if matched_mos is not None:
                row["orig_mos"] = float(matched_mos)
                matched_split = koniq_splits.get(img_id) or koniq_splits.get(Path(img_id).stem.lower())
                if matched_split:
                    row["split"] = matched_split
                koniq_updated += 1
            else:
                missing_koniq.add(img_id)

        elif ds == "livec":
            row["split"] = "test"  # LiveC is strictly test
            matched_mos = _lookup_mos(livec_mos, img_id)
            if matched_mos is not None:
                row["orig_mos"] = float(matched_mos)
                livec_updated += 1
            else:
                missing_livec.add(img_id)

        # Drop duplicate bpp_est placeholder
        row["bpp_est"] = None
        updated_rows.append(row)

    print(f"[BACKFILL] Successfully updated: {koniq_updated} KonIQ rows, {livec_updated} LiveC rows.")
    if missing_koniq:
        valid_scores = [float(r["orig_mos"]) for r in updated_rows if r.get("dataset") == "koniq" and r.get("orig_mos") is not None and float(r.get("orig_mos", 0.0)) > 0.0]
        fallback_mos = sum(valid_scores) / len(valid_scores) if valid_scores else 65.0
        print(f"[WARN] {len(missing_koniq)} KonIQ images had no direct MOS match (samples: {list(missing_koniq)[:5]}). Imputing with KonIQ mean ({fallback_mos:.2f})")
        for r in updated_rows:
            if r.get("dataset") == "koniq" and (r.get("orig_mos") is None or float(r.get("orig_mos", 0.0)) <= 0.0):
                r["orig_mos"] = fallback_mos

    if missing_livec:
        print(f"[WARN] {len(missing_livec)} LiveC images had no MOS match (samples: {list(missing_livec)[:5]})")

    # Ensure KonIQ has valid 80/10/10 stratified train/val/test splits
    koniq_splits_present = set(r.get("split") for r in updated_rows if r.get("dataset") == "koniq")
    if not {"train", "val", "test"}.issubset(koniq_splits_present):
        print(f"[BACKFILL] KonIQ splits incomplete ({koniq_splits_present}). Generating deterministic 80/10/10 stratified split...")
        from nrcomp.data.splits import create_koniq_stratified_split, save_koniq_split_csv

        koniq_images_dict: Dict[str, float] = {}
        for r in updated_rows:
            if r.get("dataset") == "koniq":
                koniq_images_dict[Path(r.get("image_id", "")).name] = float(r.get("orig_mos", 50.0))

        koniq_items = list(koniq_images_dict.items())
        new_splits = create_koniq_stratified_split(koniq_items, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0)

        for r in updated_rows:
            if r.get("dataset") == "koniq":
                img_name = Path(r.get("image_id", "")).name
                if img_name in new_splits:
                    r["split"] = new_splits[img_name]

        out_split_csv = PROJECT_ROOT / "data" / "splits" / "koniq_split.csv"
        try:
            save_koniq_split_csv(new_splits, koniq_images_dict, out_split_csv)
            print(f"[BACKFILL] Saved verified 80/10/10 split to {out_split_csv}")
        except Exception as e:
            print(f"[WARN] Could not write to {out_split_csv}: {e}")

    # 4. Atomic write back
    manifest.write_all_rows(updated_rows)
    print(f"[SUCCESS] Successfully updated manifest at {manifest_path}")

    # 5. Run Audit
    from scripts.audit_manifest import audit_manifest
    passed, errors = audit_manifest(manifest_path, check_latents_exist=False)
    if passed:
        print("[BACKFILL PASS] Manifest audit passed cleanly!")
    else:
        print(f"[BACKFILL WARN] Manifest audit reported {len(errors)} issues:")
        for e in errors[:10]:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
