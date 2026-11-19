"""Manifest management for nrcomp datasets, latents, and labels.

Provides atomic file operations and schema validation. Supports PyArrow/Pandas for
Parquet and a pure-Python CSV fallback for offline testing environments.

Integrity rules implemented here:
  * Writing never silently falls back: if pandas/pyarrow is installed and the parquet
    write fails, the error propagates (the old code swallowed it, leaving a stale
    parquet that `load()` would later prefer over the fresh CSV).
  * Float NaN read back from parquet is normalized to None, so `value is None` checks
    elsewhere in the code base are reliable.
  * Invalid rows raise instead of being silently accepted.
"""

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Expected columns and types
MANIFEST_COLUMNS = [
    "dataset",
    "image_id",
    "split",
    "lambda_idx",
    "lambda_value",
    "orig_path",
    "latent_path",
    "recon_path",
    "height",
    "width",
    "padded_height",
    "padded_width",
    "bpp_actual",
    "bpp_est",
    "psnr",
    "haarpsi",
    "dists",
    "orig_mos",
]

ALLOWED_SPLITS = ("train", "val", "test", "excluded")
FLOAT_COLUMNS = ["lambda_value", "bpp_actual", "bpp_est", "psnr", "haarpsi", "dists", "orig_mos"]
INT_COLUMNS = ["height", "width", "padded_height", "padded_width", "lambda_idx"]


def is_missing(value: Any) -> bool:
    """True for None, NaN, and empty/'nan'/'none' strings."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in ("nan", "none")
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _opt_float(value: Any) -> Optional[float]:
    return None if is_missing(value) else float(value)


@dataclass
class ManifestRow:
    dataset: str
    image_id: str
    split: str
    lambda_idx: int
    lambda_value: float
    orig_path: str
    latent_path: str
    height: int
    width: int
    padded_height: int
    padded_width: int
    bpp_actual: float
    bpp_est: Optional[float] = None
    orig_mos: Optional[float] = None
    recon_path: Optional[str] = None
    psnr: Optional[float] = None
    haarpsi: Optional[float] = None
    dists: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestRow":
        if "orig_mos" not in data:
            raise KeyError("orig_mos")  # the column must exist (value may be None for excluded rows)
        split = str(data["split"])
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {ALLOWED_SPLITS}, got {split!r}")
        return cls(
            dataset=str(data["dataset"]),
            image_id=str(data["image_id"]),
            split=split,
            lambda_idx=int(data["lambda_idx"]),
            lambda_value=float(data["lambda_value"]),
            orig_path=str(data["orig_path"]),
            latent_path=str(data["latent_path"]),
            recon_path=None if is_missing(data.get("recon_path")) else str(data["recon_path"]),
            height=int(data["height"]),
            width=int(data["width"]),
            padded_height=int(data["padded_height"]),
            padded_width=int(data["padded_width"]),
            bpp_actual=float(data["bpp_actual"]),
            bpp_est=_opt_float(data.get("bpp_est")),
            psnr=_opt_float(data.get("psnr")),
            haarpsi=_opt_float(data.get("haarpsi")),
            dists=_opt_float(data.get("dists")),
            orig_mos=_opt_float(data.get("orig_mos")),
        )


def validate_manifest_dict(d: Dict[str, Any]) -> bool:
    """Validate that dictionary contains all required columns with correct types."""
    try:
        ManifestRow.from_dict(d)
        return True
    except (KeyError, ValueError, TypeError):
        return False


def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NaN/empty values to None and numpy scalars to Python scalars."""
    out: Dict[str, Any] = {}
    for k, v in rec.items():
        if hasattr(v, "item") and not isinstance(v, (str, bytes)):
            try:
                v = v.item()
            except Exception:
                pass
        out[k] = None if is_missing(v) else v
    return out


class ManifestManager:
    """Handles reading, updating, and atomically saving the manifest."""

    def __init__(self, manifest_path: Union[str, Path]):
        self.manifest_path = Path(manifest_path)
        # If requested path does not exist, check fallback extensions
        if not self.manifest_path.exists():
            if self.manifest_path.suffix == ".parquet" and self.manifest_path.with_suffix(".csv").exists():
                self.manifest_path = self.manifest_path.with_suffix(".csv")
            elif self.manifest_path.suffix == ".csv" and self.manifest_path.with_suffix(".parquet").exists():
                self.manifest_path = self.manifest_path.with_suffix(".parquet")
        self.rows: List[Dict[str, Any]] = []
        if self.manifest_path.exists():
            self.load()

    def load(self):
        """Load manifest from Parquet (preferred) or CSV."""
        path = self.manifest_path
        if path.suffix == ".parquet" and path.exists():
            try:
                import pandas as pd
            except ImportError:
                pd = None
            if pd is not None:
                df = pd.read_parquet(path)  # errors propagate: a corrupt parquet must be loud
                self.rows = [_normalize_record(r) for r in df.to_dict(orient="records")]
                return

        csv_path = path if path.suffix == ".csv" else path.with_suffix(".csv")
        if csv_path.exists():
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                loaded_rows = []
                for r in reader:
                    row_dict = _normalize_record(dict(r))
                    for int_k in INT_COLUMNS:
                        if row_dict.get(int_k) is not None:
                            row_dict[int_k] = int(float(row_dict[int_k]))
                    for float_k in FLOAT_COLUMNS:
                        if row_dict.get(float_k) is not None:
                            row_dict[float_k] = float(row_dict[float_k])
                    loaded_rows.append(row_dict)
                self.rows = loaded_rows
            return

        if path.suffix == ".jsonl" and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.rows = [_normalize_record(json.loads(line)) for line in f]
            return

    def save(self):
        """Save manifest atomically (temp file + rename) as parquet (if available) and CSV."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        path = self.manifest_path

        try:
            import pandas as pd
            have_pandas = True
        except ImportError:
            have_pandas = False

        if have_pandas:
            parquet_path = path if path.suffix == ".parquet" else path.with_suffix(".parquet")
            tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
            try:
                df = pd.DataFrame(self.rows, columns=MANIFEST_COLUMNS)
                df.to_parquet(tmp_parquet, index=False)
            except ImportError:
                # pyarrow/fastparquet not installed: CSV-only is the documented fallback.
                if tmp_parquet.exists():
                    tmp_parquet.unlink()
            else:
                os.replace(tmp_parquet, parquet_path)

        csv_path = path if path.suffix == ".csv" else path.with_suffix(".csv")
        tmp_csv = csv_path.with_suffix(".csv.tmp")
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in MANIFEST_COLUMNS})
        os.replace(tmp_csv, csv_path)

    def add_or_update_row(self, row_dict: Dict[str, Any]):
        """Add or update a row indexed by (dataset, image_id, lambda_idx). Invalid rows raise."""
        # Complete optional columns so schema validation sees every key.
        candidate = {k: row_dict.get(k) for k in MANIFEST_COLUMNS}
        candidate.update(row_dict)
        ManifestRow.from_dict(candidate)  # raises KeyError/ValueError/TypeError when invalid
        key = (row_dict["dataset"], str(row_dict["image_id"]), int(row_dict["lambda_idx"]))
        for i, existing in enumerate(self.rows):
            existing_key = (
                existing["dataset"],
                str(existing["image_id"]),
                int(existing["lambda_idx"]),
            )
            if existing_key == key:
                self.rows[i].update(row_dict)
                return
        self.rows.append(row_dict)

    def add_row(self, row_dict: Dict[str, Any]):
        """Alias for add_or_update_row."""
        self.add_or_update_row(row_dict)

    def update_row(self, dataset: str, image_id: str, lambda_idx: int, row_dict: Dict[str, Any]):
        """Update the row identified by (dataset, image_id, lambda_idx)."""
        merged = dict(row_dict)
        merged["dataset"] = dataset
        merged["image_id"] = image_id
        merged["lambda_idx"] = lambda_idx
        self.add_or_update_row(merged)

    def exists(self, dataset: str, image_id: str, lambda_idx: int) -> bool:
        """Check if (dataset, image_id, lambda_idx) is already recorded in manifest."""
        key = (dataset, str(image_id), int(lambda_idx))
        return key in self.get_existing_keys()

    def get_all_rows(self) -> List[Dict[str, Any]]:
        """Return all rows in the manifest."""
        return self.rows

    def write_all_rows(self, rows: List[Dict[str, Any]]):
        """Replace all rows and atomically save."""
        self.rows = [_normalize_record(r) for r in rows]
        self.save()

    def flush(self):
        """Alias for save."""
        self.save()

    def get_existing_keys(self) -> set:
        """Return set of (dataset, image_id, lambda_idx) already in manifest."""
        keys = set()
        for r in self.rows:
            keys.add((r["dataset"], str(r["image_id"]), int(r["lambda_idx"])))
        return keys

    def __len__(self) -> int:
        return len(self.rows)


ParquetManifest = ManifestManager
