#!/usr/bin/env python3
"""Compatibility wrapper delegating to scripts/rebuild_manifest_meta.py.

Historically named backfill_mos.py; delegates all metadata and split rebuild logic
to the unified nrcomp.data.metadata and scripts.rebuild_manifest_meta pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.data.metadata import MosTable
from scripts.rebuild_manifest_meta import main


def _index_mos_entry(mos_map: Dict[str, float], name: str, score: float) -> None:
    """Index an image MOS score under multiple normalized keys for robust matching."""
    clean = str(name).strip()
    if not clean:
        return
    base = Path(clean).name
    stem = Path(clean).stem

    mos_map[clean] = float(score)
    mos_map[clean.lower()] = float(score)
    mos_map[base] = float(score)
    mos_map[base.lower()] = float(score)
    mos_map[stem] = float(score)
    mos_map[stem.lower()] = float(score)


def _lookup_mos(mos_map: Dict[str, float], image_id: str) -> Optional[float]:
    """Look up image MOS score by clean name, lowercase, basename, or stem."""
    clean = str(image_id).strip()
    if not clean:
        return None
    base = Path(clean).name
    stem = Path(clean).stem

    for candidate in [clean, clean.lower(), base, base.lower(), stem, stem.lower()]:
        if candidate in mos_map:
            return mos_map[candidate]
    return None


if __name__ == "__main__":
    main()
