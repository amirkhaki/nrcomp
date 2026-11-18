"""Manifest management for nrcomp datasets, latents, and labels.

Adheres to Section 7.5 of the design document as the single source of truth.
Provides atomic file operations and schema validation.
Supports both PyArrow/Pandas for Parquet and pure-Python CSV/JSON fallback for
offline testing environments.
"""

import csv
import json
import os
import shutil
import tempfile
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
    bpp_est: float
    orig_mos: float
    recon_path: Optional[str] = None
    psnr: Optional[float] = None
    haarpsi: Optional[float] = None
    dists: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestRow":
        return cls(
            dataset=str(data["dataset"]),
            image_id=str(data["image_id"]),
            split=str(data["split"]),
            lambda_idx=int(data["lambda_idx"]),
            lambda_value=float(data["lambda_value"]),
            orig_path=str(data["orig_path"]),
            latent_path=str(data["latent_path"]),
            recon_path=str(data["recon_path"]) if data.get("recon_path") else None,
            height=int(data["height"]),
            width=int(data["width"]),
            padded_height=int(data["padded_height"]),
            padded_width=int(data["padded_width"]),
            bpp_actual=float(data["bpp_actual"]),
            bpp_est=float(data["bpp_est"]),
            psnr=float(data["psnr"]) if data.get("psnr") is not None else None,
            haarpsi=float(data["haarpsi"]) if data.get("haarpsi") is not None else None,
            dists=float(data["dists"]) if data.get("dists") is not None else None,
            orig_mos=float(data["orig_mos"]),
        )


def validate_manifest_dict(d: Dict[str, Any]) -> bool:
    """Validate that dictionary contains all required columns with correct types."""
    try:
        ManifestRow.from_dict(d)
        return True
    except (KeyError, ValueError, TypeError):
        return False


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
        """Load manifest from Parquet or fallback formats."""
        path = self.manifest_path
        if path.suffix == ".parquet" and path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(path)
                self.rows = df.to_dict(orient="records")
                return
            except (ImportError, Exception):
                pass

        csv_path = path if path.suffix == ".csv" else path.with_suffix(".csv")
        if csv_path.exists():
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                loaded_rows = []
                for r in reader:
                    row_dict = dict(r)
                    # Convert typed numeric fields
                    for int_k in ["height", "width", "padded_height", "padded_width", "lambda_idx"]:
                        if row_dict.get(int_k) not in (None, ""):
                            try:
                                row_dict[int_k] = int(row_dict[int_k])
                            except ValueError:
                                pass
                    for float_k in ["lambda_value", "bpp_actual", "bpp_est", "psnr", "haarpsi", "dists", "orig_mos"]:
                        if row_dict.get(float_k) not in (None, ""):
                            try:
                                row_dict[float_k] = float(row_dict[float_k])
                            except ValueError:
                                pass
                        elif float_k in ["psnr", "haarpsi", "dists"]:
                            row_dict[float_k] = None
                    loaded_rows.append(row_dict)
                self.rows = loaded_rows
            return

        # Attempt to read as json lines if fallback
        if path.suffix == ".jsonl" and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.rows = [json.loads(line) for line in f]
            return

    def save(self):
        """Save manifest atomically using a temp file and rename."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        path = self.manifest_path

        # If pandas is available, always write parquet
        try:
            import pandas as pd
            parquet_path = path if path.suffix == ".parquet" else path.with_suffix(".parquet")
            tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
            df = pd.DataFrame(self.rows)
            df.to_parquet(tmp_parquet, index=False)
            os.replace(tmp_parquet, parquet_path)
        except (ImportError, Exception):
            pass

        # Always also save/mirror CSV for resilience
        csv_path = path if path.suffix == ".csv" else path.with_suffix(".csv")
        tmp_csv = csv_path.with_suffix(".csv.tmp")
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})
        os.replace(tmp_csv, csv_path)

    def add_or_update_row(self, row_dict: Dict[str, Any]):
        """Add or update a row indexed by (dataset, image_id, lambda_idx)."""
        validate_manifest_dict(row_dict)
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
        """Alias for add_or_update_row."""
        self.add_or_update_row(row_dict)

    def exists(self, dataset: str, image_id: str, lambda_idx: int) -> bool:
        """Check if (dataset, image_id, lambda_idx) is already recorded in manifest."""
        key = (dataset, str(image_id), int(lambda_idx))
        return key in self.get_existing_keys()

    def get_all_rows(self) -> List[Dict[str, Any]]:
        """Return all rows in the manifest."""
        return self.rows

    def write_all_rows(self, rows: List[Dict[str, Any]]):
        """Replace all rows and atomically save."""
        self.rows = list(rows)
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
