"""Stage 3: Quality Metric Labelling Pipeline.

CLI Entry point:
    python -m nrcomp.label --config configs/label.yaml [--device cuda]
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.codec.wrapper import ELICWrapper
from nrcomp.config import load_config
from nrcomp.data.manifest import ParquetManifest
from nrcomp.data.store import LatentStore
from nrcomp.labels.metrics import QualityMetrics


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 Quality Metric Labelling.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--manifest", type=str, default=None, help="Override manifest path.")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda or cpu).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path_raw = args.manifest or cfg.get("manifest_path", "data/manifest.parquet")
    manifest_path = (PROJECT_ROOT / manifest_path_raw) if not Path(manifest_path_raw).is_absolute() else Path(manifest_path_raw)

    if not manifest_path.exists():
        if manifest_path.suffix == ".parquet" and manifest_path.with_suffix(".csv").exists():
            manifest_path = manifest_path.with_suffix(".csv")
        elif manifest_path.suffix == ".csv" and manifest_path.with_suffix(".parquet").exists():
            manifest_path = manifest_path.with_suffix(".parquet")
        else:
            print(f"[ERROR] Manifest not found: {manifest_path}")
            sys.exit(1)

    device = args.device or cfg.get("device", "cuda")
    metrics_list = [m["name"] if isinstance(m, dict) else m for m in cfg.get("metrics", ["haarpsi", "dists", "psnr"])]

    print(f"Loading manifest: {manifest_path}")
    manifest = ParquetManifest(manifest_path)
    rows = manifest.get_all_rows()
    print(f"Total manifest rows: {len(rows)}")

    def is_missing_value(val):
        if val is None:
            return True
        try:
            f = float(val)
            return math.isnan(f) or math.isinf(f)
        except (ValueError, TypeError):
            return True

    # Filter rows that need labelling
    rows_to_label = []
    for r in rows:
        needs_label = False
        for m in metrics_list:
            if is_missing_value(r.get(m)):
                needs_label = True
                break
        if needs_label:
            rows_to_label.append(r)

    print(f"Rows requiring metric computation: {len(rows_to_label)}")
    if len(rows_to_label) == 0:
        print("All rows already labelled. Nothing to do.")
        return

    import torch
    from PIL import Image
    import numpy as np

    actual_device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
    print(f"Initializing QualityMetrics on device: {actual_device}")
    quality_metrics = QualityMetrics(device=actual_device, enable_metrics=metrics_list)

    # Validate that models for required metrics are initialized
    missing_models = [m for m in metrics_list if m != "psnr" and m not in quality_metrics.models]
    if missing_models:
        print(f"[ERROR] Required metric models could not be loaded: {missing_models}")
        print(f"Loaded models: {list(quality_metrics.models.keys())}")
        print("Please ensure piq and pyiqa are installed in this environment:")
        print("  conda run -n nrcomp-main pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn piq pyiqa")
        sys.exit(1)

    # In case we need on-the-fly reconstruction from latents
    latent_store = LatentStore(root_dir=PROJECT_ROOT / "data" / "latents")
    wrapper: Optional[ELICWrapper] = None

    updated_count = 0
    for idx, row in enumerate(rows_to_label):
        orig_path = Path(row["orig_path"])
        recon_path = Path(row["recon_path"]) if row.get("recon_path") else None

        if not orig_path.exists():
            print(f"[WARN] Original image not found: {orig_path}, skipping row.")
            continue

        try:
            orig_pil = Image.open(orig_path).convert("RGB")
            orig_np = np.asarray(orig_pil, dtype=np.uint8)

            recon_np: Optional[np.ndarray] = None
            if recon_path and recon_path.exists():
                recon_pil = Image.open(recon_path).convert("RGB")
                recon_np = np.asarray(recon_pil, dtype=np.uint8)
            else:
                # Decode on-the-fly from latent store
                latent_path = Path(row["latent_path"])
                if latent_path.exists():
                    if wrapper is None:
                        weights_dir = PROJECT_ROOT / cfg.get("weights_dir", "weights/updated")
                        wrapper = ELICWrapper(weights_dir=weights_dir, device=actual_device)
                    lambda_idx = row["lambda_idx"]
                    if lambda_idx not in wrapper.models:
                        ckpt_name = f"elic_lambda_{row['lambda_value']}_updated.pth.tar"
                        wrapper.load_model(lambda_idx, ckpt_name)

                    lat_data = latent_store.load_latents(latent_path)
                    meta = {
                        "original_size": (row["height"], row["width"]),
                        "padded_size": (row["padded_height"], row["padded_width"]),
                        "lambda_idx": lambda_idx,
                    }
                    recon_np = wrapper.decode_from_latents(torch.from_numpy(lat_data["y_hat"]).float(), meta)
                else:
                    print(f"[WARN] Neither recon nor latent found for {row['image_id']}, skipping.")
                    continue

            # Compute metrics
            metric_results = quality_metrics.compute_all(orig_np, recon_np)
            for m_name, val in metric_results.items():
                row[m_name] = val

            manifest.update_row(row["dataset"], row["image_id"], row["lambda_idx"], row)
            updated_count += 1

            if (idx + 1) % 20 == 0 or (idx + 1) == len(rows_to_label):
                manifest.flush()
                print(f"Labelling progress: {idx + 1}/{len(rows_to_label)} rows processed.", flush=True)

        except Exception as e:
            print(f"[!] Error labelling {row['image_id']} lambda={row['lambda_idx']}: {e}", flush=True)

    # Final flush
    manifest.flush()
    print(f"\nLabelling complete! Successfully updated {updated_count} rows in {manifest_path}.", flush=True)


if __name__ == "__main__":
    main()
