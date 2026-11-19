#!/usr/bin/env python3
"""Repair `split` and `orig_mos` in an existing manifest WITHOUT regenerating anything.

Replaces the old `backfill_mos.py`, which (a) imputed the dataset mean for images it could
not match, (b) mixed MOS scales from several candidate files, and (c) wrote pyiqa's numeric
split codes ('0','1','2') into the manifest, leaving only 27 KonIQ images labelled 'train'.

What it does:
  1. Backs up manifest.parquet / manifest.csv (timestamped copies next to them).
  2. KonIQ: MOS from koniq10k_scores_and_distributions.csv (`MOS`, scale 1-5). Images with
     no verified MOS are EXCLUDED (split='excluded', orig_mos=None) -- never imputed.
     Remaining images get a deterministic 80/10/10 split stratified by MOS quintile.
  3. LiveC: MOS from meta_info_LIVEChallengeDataset.csv (`mos`, scale 0-100); test-only.
  4. Writes the manifest, data/splits/koniq_split.csv and data/splits/koniq_excluded.csv.
  5. Runs the manifest audit; exits 1 if it fails.

Latents, recons, bitrates and proxy labels are never touched.

Usage (server):
    python scripts/rebuild_manifest_meta.py --manifest data/manifest.parquet --dry-run
    python scripts/rebuild_manifest_meta.py --manifest data/manifest.parquet
"""

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.data.manifest import ManifestManager
from nrcomp.data.metadata import (
    default_koniq_scores_path,
    default_livec_meta_path,
    load_koniq_scores,
    load_livec_scores,
    rebuild_manifest_metadata,
    write_split_csvs,
)
from scripts.audit_manifest import audit_manifest, summarize_manifest


def _backup(path: Path, stamp: str) -> None:
    for p in (path, path.with_suffix(".csv"), path.with_suffix(".parquet")):
        if p.exists():
            dest = p.with_name(f"{p.name}.bak-{stamp}")
            if not dest.exists():
                shutil.copy2(p, dest)
                print(f"[REBUILD] Backup: {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild split/orig_mos in the manifest (no regeneration).")
    parser.add_argument("--manifest", default="data/manifest.parquet")
    parser.add_argument("--koniq-scores", default=None, help="Override KonIQ scores CSV.")
    parser.add_argument("--livec-meta", default=None, help="Override LiveC meta_info CSV.")
    parser.add_argument("--seed", type=int, default=0, help="Split seed (default 0, per design).")
    parser.add_argument(
        "--keep-existing-livec-mos",
        action="store_true",
        help="If a LiveC image is missing from the meta CSV, keep its existing (non-imputed) MOS.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would change; write nothing.")
    parser.add_argument("--skip-latent-files", action="store_true", help="Skip latent existence check in the audit.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest = ManifestManager(manifest_path)
    if len(manifest) == 0:
        print(f"[REBUILD][ERROR] Manifest empty or missing: {manifest_path}")
        return 1
    rows = manifest.get_all_rows()
    print(f"[REBUILD] Loaded {len(rows)} rows from {manifest.manifest_path}")
    print(f"[REBUILD] BEFORE (unique images per dataset/split): {summarize_manifest(rows)}")

    koniq_path = Path(args.koniq_scores) if args.koniq_scores else default_koniq_scores_path(PROJECT_ROOT)
    livec_path = Path(args.livec_meta) if args.livec_meta else default_livec_meta_path(PROJECT_ROOT)
    koniq_table = load_koniq_scores(koniq_path) if any(r.get("dataset") == "koniq" for r in rows) else None
    livec_table = None
    if any(r.get("dataset") == "livec" for r in rows):
        try:
            livec_table = load_livec_scores(livec_path)
        except (FileNotFoundError, ValueError) as e:
            if not args.keep_existing_livec_mos:
                print(f"[REBUILD][ERROR] Cannot read LiveC meta ({e}). Re-run with --keep-existing-livec-mos "
                      f"to keep the existing LiveC MOS values, or pass --livec-meta.")
                return 1
            print(f"[REBUILD][WARN] LiveC meta unreadable ({e}); keeping existing LiveC MOS.")
    print(f"[REBUILD] KonIQ MOS entries: {len(koniq_table) if koniq_table else 0} ({koniq_path})")
    print(f"[REBUILD] LiveC MOS entries: {len(livec_table) if livec_table else 0} ({livec_path})")

    new_rows, report = rebuild_manifest_metadata(
        rows, koniq_table, livec_table, seed=args.seed, keep_existing_livec_mos=args.keep_existing_livec_mos
    )
    print(f"[REBUILD] AFTER  (unique images per dataset/split): {summarize_manifest(new_rows)}")
    print(f"[REBUILD] KonIQ images excluded (no verified MOS): {len(report['koniq_excluded_images'])}"
          f" e.g. {report['koniq_excluded_images'][:5]}")
    print(f"[REBUILD] LiveC images excluded: {len(report['livec_excluded_images'])}")
    if report["livec_max_abs_diff_vs_existing"] is not None:
        print(f"[REBUILD] LiveC: max |new MOS - existing MOS| = {report['livec_max_abs_diff_vs_existing']:.6g}")

    if args.dry_run:
        print("[REBUILD] --dry-run: nothing written.")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    _backup(manifest.manifest_path, stamp)
    manifest.write_all_rows(new_rows)
    split_csv, excl_csv = write_split_csvs(new_rows, PROJECT_ROOT / "data" / "splits", report["koniq_excluded_images"])
    report_path = PROJECT_ROOT / "data" / "splits" / "rebuild_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[REBUILD] Wrote manifest, {split_csv}, {excl_csv}, {report_path}")

    passed, errors = audit_manifest(
        manifest.manifest_path,
        check_latents_exist=not args.skip_latent_files,
        min_train_images=100,
        min_eval_images=30,
    )
    if not passed:
        print(f"[REBUILD][AUDIT FAIL] {len(errors)} error(s):")
        for e in errors[:30]:
            print(f"  - {e}")
        return 1
    print("[REBUILD][AUDIT PASS] Manifest metadata rebuilt and audited successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
