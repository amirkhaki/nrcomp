"""Dataset metadata (MOS scores, splits) and the manifest metadata rebuild.

Single source of truth for how `orig_mos` and `split` get into the manifest. Used by
`nrcomp.generate` (for new data) and `scripts/rebuild_manifest_meta.py` (to repair an
existing manifest WITHOUT regenerating latents, recons or labels).

Hard rules enforced here (design.md Section 0):
  * NEVER impute a MOS. An image without a verified MOS is marked split="excluded" and
    is never used for training, validation or testing.
  * ONE canonical MOS source per dataset, with a documented scale:
        KonIQ-10k : column `MOS` of koniq10k_scores_and_distributions.csv (scale 1-5)
        LiveC     : column `mos` of meta_info_LIVEChallengeDataset.csv     (scale 0-100)
    MOS values from different datasets are never pooled, so the scales may differ.
  * Splits are image-disjoint: KonIQ 80/10/10 stratified by MOS quintile (seed 0);
    LiveC is test-only.
  * Split labels are exactly: train | val | test | excluded.

Pure standard library.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from nrcomp.data.splits import create_koniq_stratified_split

PathLike = Union[str, Path]

ALLOWED_SPLITS = ("train", "val", "test", "excluded")
KONIQ_MOS_SCALE = "1-5 (MOS column of koniq10k_scores_and_distributions.csv)"
LIVEC_MOS_SCALE = "0-100 (mos column of meta_info_LIVEChallengeDataset.csv)"

KONIQ_NAME_COLUMNS = ("image_name", "img_name", "image_id", "name")
KONIQ_SCORE_COLUMN = "MOS"
LIVEC_NAME_COLUMNS = ("img_name", "image_name", "image_id", "name", "img_path")
LIVEC_SCORE_COLUMNS = ("mos", "MOS", "score")


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == "" or v.strip().lower() in ("nan", "none")
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


class MosTable:
    """Strict image-name -> MOS lookup.

    Matching order: exact basename, case-insensitive basename, then stem (case-insensitive)
    ONLY if that stem is unambiguous. Never guesses; returns None when there is no
    unambiguous match.
    """

    def __init__(self, entries: Dict[str, float]):
        self.exact: Dict[str, float] = {}
        self.lower: Dict[str, float] = {}
        stems: Dict[str, List[float]] = defaultdict(list)
        self._ambiguous_lower = set()
        for name, mos in entries.items():
            base = Path(str(name).strip()).name
            self.exact[base] = float(mos)
            low = base.lower()
            if low in self.lower and self.lower[low] != float(mos):
                self._ambiguous_lower.add(low)
            self.lower[low] = float(mos)
            stems[Path(base).stem.lower()].append(float(mos))
        self.stem: Dict[str, float] = {}
        self._ambiguous_stems = set()
        for stem, vals in stems.items():
            if len(set(vals)) == 1 and len(vals) == 1:
                self.stem[stem] = vals[0]
            else:
                self._ambiguous_stems.add(stem)

    def __len__(self) -> int:
        return len(self.exact)

    def get(self, image_id: str) -> Optional[float]:
        base = Path(str(image_id).strip()).name
        if base in self.exact:
            return self.exact[base]
        low = base.lower()
        if low in self.lower and low not in self._ambiguous_lower:
            return self.lower[low]
        stem = Path(base).stem.lower()
        if stem in self.stem:
            return self.stem[stem]
        return None


def _read_scores_csv(
    csv_path: PathLike,
    name_columns: Iterable[str],
    score_columns: Iterable[str],
) -> Dict[str, float]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scores file not found: {path}")
    out: Dict[str, float] = {}
    with open(path, "r", newline="", encoding="utf-8-sig", errors="strict") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty file / no header")
        fields = {str(c).strip(): c for c in reader.fieldnames}
        name_col = next((fields[c] for c in name_columns if c in fields), None)
        score_col = next((fields[c] for c in score_columns if c in fields), None)
        if name_col is None or score_col is None:
            raise ValueError(
                f"{path}: needs a name column in {list(name_columns)} and a score column in "
                f"{list(score_columns)}; found header {list(fields)}"
            )
        for row in reader:
            name = str(row[name_col]).strip()
            raw = row[score_col]
            if not name or _is_missing(raw):
                continue
            out[Path(name).name] = float(raw)
    if not out:
        raise ValueError(f"{path}: no (name, score) rows parsed")
    return out


def load_koniq_scores(csv_path: PathLike) -> MosTable:
    """KonIQ-10k MOS (1-5 scale) from koniq10k_scores_and_distributions.csv."""
    return MosTable(_read_scores_csv(csv_path, KONIQ_NAME_COLUMNS, (KONIQ_SCORE_COLUMN,)))


def load_livec_scores(csv_path: PathLike) -> MosTable:
    """LiveC MOS (0-100 scale) from the pyiqa meta_info_LIVEChallengeDataset.csv."""
    return MosTable(_read_scores_csv(csv_path, LIVEC_NAME_COLUMNS, LIVEC_SCORE_COLUMNS))


def default_koniq_scores_path(project_root: PathLike) -> Path:
    return Path(project_root) / "data" / "raw" / "koniq10k" / "koniq10k_scores_and_distributions.csv"


def default_livec_meta_path(project_root: PathLike) -> Path:
    return Path(project_root) / "data" / "raw" / "meta_info" / "meta_info_LIVEChallengeDataset.csv"


def assign_koniq_splits(image_mos: Dict[str, float], seed: int = 0) -> Dict[str, str]:
    """Deterministic image-disjoint 80/10/10 split stratified by MOS quintile."""
    items = sorted(image_mos.items(), key=lambda kv: (kv[1], kv[0]))
    split = create_koniq_stratified_split(items, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=seed)
    return split


def _row_image(row: Dict[str, Any]) -> str:
    return Path(str(row.get("image_id", ""))).name


def rebuild_manifest_metadata(
    rows: List[Dict[str, Any]],
    koniq_table: Optional[MosTable],
    livec_table: Optional[MosTable],
    seed: int = 0,
    keep_existing_livec_mos: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rewrite `split` and `orig_mos` of every row from the canonical sources.

    Returns (new_rows, report). Does not touch latents, recons, labels or bitrates.
    """
    new_rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "seed": seed,
        "koniq_mos_scale": KONIQ_MOS_SCALE,
        "livec_mos_scale": LIVEC_MOS_SCALE,
        "koniq_excluded_images": [],
        "livec_excluded_images": [],
        "koniq_split_images": {},
        "livec_split_images": {},
        "livec_max_abs_diff_vs_existing": None,
        "livec_kept_existing": 0,
    }

    # ---- KonIQ: decide splits per image ------------------------------------------------
    koniq_images = sorted({_row_image(r) for r in rows if r.get("dataset") == "koniq"})
    koniq_mos: Dict[str, float] = {}
    if koniq_images:
        if koniq_table is None:
            raise ValueError("Manifest contains KonIQ rows but no KonIQ scores table was provided.")
        for img in koniq_images:
            mos = koniq_table.get(img)
            if mos is not None:
                koniq_mos[img] = mos
            else:
                report["koniq_excluded_images"].append(img)
    koniq_split = assign_koniq_splits(koniq_mos, seed=seed) if koniq_mos else {}
    for img in koniq_mos:
        if img not in koniq_split:
            raise RuntimeError(f"Split assignment dropped image {img}")

    # ---- LiveC: per-image MOS ----------------------------------------------------------
    livec_images = sorted({_row_image(r) for r in rows if r.get("dataset") == "livec"})
    livec_mos: Dict[str, float] = {}
    max_diff = 0.0
    n_compared = 0
    existing_by_img: Dict[str, Optional[float]] = {}
    for r in rows:
        if r.get("dataset") == "livec":
            existing_by_img[_row_image(r)] = None if _is_missing(r.get("orig_mos")) else float(r["orig_mos"])
    for img in livec_images:
        mos = livec_table.get(img) if livec_table is not None else None
        existing = existing_by_img.get(img)
        if mos is not None:
            livec_mos[img] = mos
            if existing is not None:
                max_diff = max(max_diff, abs(existing - mos))
                n_compared += 1
        elif keep_existing_livec_mos and existing is not None:
            livec_mos[img] = existing
            report["livec_kept_existing"] += 1
        else:
            report["livec_excluded_images"].append(img)
    if n_compared:
        report["livec_max_abs_diff_vs_existing"] = max_diff

    # ---- Rewrite rows ------------------------------------------------------------------
    for r in rows:
        row = dict(r)
        ds = row.get("dataset")
        img = _row_image(row)
        if ds == "koniq":
            if img in koniq_mos:
                row["split"] = koniq_split[img]
                row["orig_mos"] = koniq_mos[img]
            else:
                row["split"] = "excluded"
                row["orig_mos"] = None
        elif ds == "livec":
            if img in livec_mos:
                row["split"] = "test"
                row["orig_mos"] = livec_mos[img]
            else:
                row["split"] = "excluded"
                row["orig_mos"] = None
        else:
            raise ValueError(f"Unknown dataset '{ds}' in manifest row for {img}")
        row["bpp_est"] = None
        new_rows.append(row)

    for ds_key, name in (("koniq", "koniq_split_images"), ("livec", "livec_split_images")):
        per_split: Dict[str, set] = defaultdict(set)
        for row in new_rows:
            if row.get("dataset") == ds_key:
                per_split[row["split"]].add(_row_image(row))
        report[name] = {k: len(v) for k, v in sorted(per_split.items())}
    return new_rows, report


def write_split_csvs(
    new_rows: List[Dict[str, Any]],
    splits_dir: PathLike,
    koniq_excluded: List[str],
) -> Tuple[Path, Path]:
    """Persist the KonIQ split (image_id, split, orig_mos) and the excluded-image list."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    per_image: Dict[str, Tuple[str, Optional[float]]] = {}
    for r in new_rows:
        if r.get("dataset") == "koniq" and r.get("split") != "excluded":
            per_image[_row_image(r)] = (r["split"], r.get("orig_mos"))
    split_path = splits_dir / "koniq_split.csv"
    with open(split_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "split", "orig_mos"])
        for img in sorted(per_image):
            sp, mos = per_image[img]
            w.writerow([img, sp, mos])
    excl_path = splits_dir / "koniq_excluded.csv"
    with open(excl_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "reason"])
        for img in sorted(koniq_excluded):
            w.writerow([img, "no verified MOS in koniq10k_scores_and_distributions.csv"])
    return split_path, excl_path
