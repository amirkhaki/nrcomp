"""Stage 2: Codec Latent and Reconstruction Generation Pipeline.

CLI Entry point:
    python -m nrcomp.generate --config configs/generate_pilot.yaml [--shard-id 0 --num-shards 1]

Metadata rules (see nrcomp/data/metadata.py):
  * KonIQ MOS comes from koniq10k_scores_and_distributions.csv (`MOS`, scale 1-5) and the
    80/10/10 split is assigned deterministically from it.
  * LiveC MOS comes from meta_info_LIVEChallengeDataset.csv (`mos`, 0-100); LiveC is test-only.
  * Images without a verified MOS are skipped (never given a default MOS of 0.0).

Known limitation: `bpp_actual` is 8 * len(pickle payload) / (H * W). The pickle wrapper adds
a small constant overhead on top of the entropy-coded strings; it is identical for every
sample, so rankings are unaffected, but absolute bpp is slightly inflated. Use
scripts/measure_bpp_overhead.py to quantify it.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.codec.wrapper import ELICWrapper
from nrcomp.config import load_config
from nrcomp.data.manifest import ParquetManifest
from nrcomp.data.metadata import (
    MosTable,
    assign_koniq_splits,
    default_koniq_scores_path,
    default_livec_meta_path,
    load_koniq_scores,
    load_livec_scores,
)
from nrcomp.data.store import LatentStore, check_disk_space
from nrcomp.lambdas import load_lambdas_config

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tif", ".tiff"}


def log_failure(failure_log_path: Path, dataset: str, image_id: str, lambda_idx: int, error_msg: str):
    """Append unexpected generation error to failure log CSV."""
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failure_log_path.exists()
    with open(failure_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "image_id", "lambda_idx", "error"])
        writer.writerow([dataset, image_id, lambda_idx, str(error_msg)])


def _find_image_dir(dataset_cfg: Dict[str, Any], project_root: Path, dataset_name: str) -> Optional[Path]:
    img_dir_raw = dataset_cfg.get("image_dir", f"data/raw/{dataset_name}")
    img_dir = (project_root / img_dir_raw) if not Path(img_dir_raw).is_absolute() else Path(img_dir_raw)
    if img_dir.exists():
        return img_dir
    candidates = [
        project_root / "data" / "raw" / "koniq10k",
        project_root / "data" / "raw" / "koniq-10k",
        project_root / "data" / "raw" / "KonIQ-10k",
        project_root / "data" / "raw" / "LIVEC",
        project_root / "data" / "raw" / "livec",
        project_root / "data" / "raw" / "CLIVE",
    ]
    for c in candidates:
        if c.exists() and (dataset_name.lower() in c.name.lower() or c.name.lower() in dataset_name.lower()):
            return c
    print(f"[WARN] Image directory not found: {img_dir}")
    return None


def _list_images(img_dir: Path) -> List[Path]:
    paths = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in VALID_EXTENSIONS)
    if not paths:
        # Subdirectories (e.g. koniq10k/1024x768/ or LIVEC/Images/)
        paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)
    return [p for p in paths if not p.name.startswith(".")]


def _read_split_csv(path: Path) -> Dict[str, Tuple[str, float]]:
    """Read (name -> (split, mos)); accepts `orig_mos` (splits.py output) or `mos`. Never defaults MOS."""
    info: Dict[str, Tuple[str, float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        mos_col = "orig_mos" if "orig_mos" in fields else ("mos" if "mos" in fields else None)
        name_col = "image_id" if "image_id" in fields else ("image_name" if "image_name" in fields else None)
        if mos_col is None or name_col is None or "split" not in fields:
            raise ValueError(
                f"{path}: split CSV needs columns [image_id|image_name], split, [orig_mos|mos]; found {sorted(fields)}"
            )
        for row in reader:
            name = str(row[name_col]).strip()
            raw_mos = row[mos_col]
            if raw_mos is None or str(raw_mos).strip() == "":
                raise ValueError(f"{path}: image '{name}' has an empty {mos_col}; refusing to default it.")
            info[name] = (str(row["split"]).strip(), float(raw_mos))
            info[Path(name).name] = info[name]
    return info


def collect_dataset_images(
    dataset_cfg: Dict[str, Any], project_root: Path
) -> List[Tuple[str, Path, str, float]]:
    """Collect image records: (dataset_name, file_path, split, orig_mos)."""
    dataset_name = dataset_cfg.get("name", dataset_cfg.get("dataset", "koniq"))
    img_dir = _find_image_dir(dataset_cfg, project_root, dataset_name)
    if img_dir is None:
        return []
    print(f"Using image directory for {dataset_name}: {img_dir}")
    all_img_paths = _list_images(img_dir)
    print(f"Found {len(all_img_paths)} images for {dataset_name}.")

    # ---- per-image (split, mos) --------------------------------------------------------
    split_info: Dict[str, Tuple[str, float]] = {}
    split_csv_path = dataset_cfg.get("split_csv")
    if split_csv_path:
        p = (project_root / split_csv_path) if not Path(split_csv_path).is_absolute() else Path(split_csv_path)
        if p.exists():
            split_info = _read_split_csv(p)

    if not split_info and dataset_name == "koniq":
        scores_path = Path(dataset_cfg["scores_csv"]) if dataset_cfg.get("scores_csv") else default_koniq_scores_path(project_root)
        table: MosTable = load_koniq_scores(scores_path)
        image_mos = {}
        for ip in all_img_paths:
            mos = table.get(ip.name)
            if mos is not None:
                image_mos[ip.name] = mos
        splits = assign_koniq_splits(image_mos, seed=int(dataset_cfg.get("split_seed", 0)))
        split_info = {name: (splits[name], image_mos[name]) for name in image_mos}
    elif not split_info and dataset_name == "livec":
        meta_path = Path(dataset_cfg["meta_csv"]) if dataset_cfg.get("meta_csv") else default_livec_meta_path(project_root)
        table = load_livec_scores(meta_path)
        for ip in all_img_paths:
            mos = table.get(ip.name)
            if mos is not None:
                split_info[ip.name] = ("test", mos)

    # ---- drop images without verified metadata (never default to MOS 0.0) -------------
    if dataset_name in ("koniq", "livec") or split_info:
        usable = [p for p in all_img_paths if p.name in split_info]
        dropped = len(all_img_paths) - len(usable)
        if dropped:
            print(f"[WARN] {dropped} {dataset_name} images have no verified MOS/split and are skipped.")
        all_img_paths = usable

    # ---- optional proportional subsetting ---------------------------------------------
    max_images = dataset_cfg.get("max_images")
    if max_images and len(all_img_paths) > max_images:
        if split_info:
            from collections import defaultdict

            by_split = defaultdict(list)
            for p in all_img_paths:
                by_split[split_info[p.name][0]].append(p)
            selected_paths: List[Path] = []
            for s in ["train", "val", "test"]:
                plist = by_split.get(s, [])
                if plist:
                    quota = max(1, int(round(len(plist) / len(all_img_paths) * max_images)))
                    selected_paths.extend(plist[:quota])
            all_img_paths = selected_paths[:max_images]
        else:
            all_img_paths = all_img_paths[:max_images]
        print(f"Subsetting {dataset_name} to {len(all_img_paths)} images.")

    records = []
    for img_path in all_img_paths:
        if split_info:
            split, mos = split_info[img_path.name]
        else:
            split = dataset_cfg.get("default_split", "train")
            mos = float(dataset_cfg.get("default_mos", 0.0))
        records.append((dataset_name, img_path, split, mos))
    return records


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 Codec Latent Generation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index (0-based).")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights_dir = PROJECT_ROOT / cfg.get("weights_dir", "weights/updated")
    manifest_path = PROJECT_ROOT / cfg.get("manifest_path", "data/manifest.parquet")
    latent_store_dir = PROJECT_ROOT / cfg.get("latent_store_dir", "data/latents")
    recon_dir = PROJECT_ROOT / cfg.get("recon_dir", "data/recons")
    save_z_hat = cfg.get("save_z_hat", True)
    global_store_recons = cfg.get("store_recons", False)
    failure_log_path = PROJECT_ROOT / cfg.get("failure_log", "reports/failures.csv")

    datasets_cfgs = cfg.get("datasets")
    if not datasets_cfgs:
        datasets_cfgs = [cfg]

    all_image_records: List[Tuple[str, Path, str, float]] = []
    for d_cfg in datasets_cfgs:
        all_image_records.extend(collect_dataset_images(d_cfg, PROJECT_ROOT))

    max_images = cfg.get("max_images")
    if max_images and len(all_image_records) > max_images:
        all_image_records = all_image_records[:max_images]

    sharded_records = all_image_records[args.shard_id :: args.num_shards]
    lambdas_cfg = cfg.get("lambdas") or load_lambdas_config()

    print(f"Total images: {len(all_image_records)}, Shard {args.shard_id}/{args.num_shards}: {len(sharded_records)} images.")
    print(f"Target lambdas: {len(lambdas_cfg)} operating points.")

    manifest = ParquetManifest(manifest_path)
    if len(manifest) == 0:
        for pilot_cand in [
            manifest_path.parent / "manifest_pilot.parquet",
            manifest_path.parent / "manifest_pilot.csv",
        ]:
            if pilot_cand.exists():
                p_man = ParquetManifest(pilot_cand)
                for r in p_man.get_all_rows():
                    manifest.add_row(r)
                manifest.flush()
                print(f"Imported {len(manifest)} existing pilot samples from {pilot_cand.name}.")
                break

    existing_keys = manifest.get_existing_keys()
    needed_samples = 0
    for dataset_name, img_path, _, _ in sharded_records:
        for l_item in lambdas_cfg:
            if (dataset_name, str(img_path.name), int(l_item["idx"])) not in existing_keys:
                needed_samples += 1

    has_space, free_bytes, required_bytes = check_disk_space(
        latent_store_dir, needed_samples, bytes_per_sample=int(1.8 * 1024 * 1024), safety_margin=1.1
    )
    free_gb = free_bytes / (1024 ** 3)
    req_gb = required_bytes / (1024 ** 3)
    print(f"Pre-flight disk check: {free_gb:.2f} GB free, estimated required for {needed_samples} new samples: {req_gb:.2f} GB.")
    if not has_space:
        print(f"[FATAL] Insufficient disk space on {latent_store_dir}!")
        print(f"Required: {req_gb:.2f} GB (with 1.1x safety margin), Available: {free_gb:.2f} GB.")
        sys.exit(1)

    import torch
    from PIL import Image
    import numpy as np

    device = args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu"
    print(f"Running on device: {device}")

    wrapper = ELICWrapper(weights_dir=weights_dir, device=device)
    for l_item in lambdas_cfg:
        print(f"Loading codec lambda {l_item['idx']} ({l_item['name']}): {l_item['checkpoint']}...")
        wrapper.load_model(l_item["idx"], l_item["checkpoint"])

    latent_store = LatentStore(root_dir=latent_store_dir, store_format=cfg.get("format", "npz"))

    completed_count = 0
    skipped_count = 0
    failed_count = 0

    print("\n--- Starting Latent Extraction & Data Generation ---")
    for img_idx, (dataset_name, img_path, split, orig_mos) in enumerate(sharded_records):
        image_id = img_path.name

        try:
            pil_img = Image.open(img_path).convert("RGB")
            orig_np = np.asarray(pil_img, dtype=np.uint8)
            orig_h, orig_w = orig_np.shape[:2]
        except Exception as e:
            print(f"[ERROR] Failed to load image {img_path}: {e}")
            log_failure(failure_log_path, dataset_name, image_id, -1, f"Image open failed: {e}")
            failed_count += 1
            continue

        for l_item in lambdas_cfg:
            lambda_idx = l_item["idx"]
            lambda_val = float(l_item["value"])

            if manifest.exists(dataset_name, image_id, lambda_idx):
                skipped_count += 1
                continue

            try:
                bitstream_bytes, meta = wrapper.compress(orig_np, lambda_idx)
                bpp_actual = float(8.0 * len(bitstream_bytes) / (orig_h * orig_w))

                latent_dict = wrapper.extract_latents_from_bitstream(bitstream_bytes, meta)
                y_hat = latent_dict["y_hat"]
                z_hat = latent_dict["z_hat"]
                valid_mask = latent_dict["valid_mask"]

                dest_latent_path = latent_store.get_path(dataset_name, image_id, lambda_idx)
                latent_store.save_latents(
                    dest_latent_path,
                    y_hat=y_hat,
                    z_hat=z_hat if save_z_hat else None,
                    valid_mask=valid_mask,
                )

                store_recon_for_sample = (
                    global_store_recons
                    or split in ["test", "livec_test", "val"]
                    or cfg.get("stage") == "generate_pilot"
                )
                recon_path_str: Optional[str] = None
                if store_recon_for_sample:
                    recon_np = wrapper.decode_from_latents(y_hat, meta)
                    dest_recon_path = (
                        recon_dir / dataset_name / f"lambda_{lambda_idx}" / f"{Path(image_id).stem}.png"
                    )
                    dest_recon_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(recon_np).save(dest_recon_path, format="PNG")
                    recon_path_str = str(dest_recon_path)

                padded_h, padded_w = meta["padded_size"]

                row = {
                    "dataset": dataset_name,
                    "image_id": image_id,
                    "split": split,
                    "lambda_idx": lambda_idx,
                    "lambda_value": lambda_val,
                    "orig_path": str(img_path),
                    "latent_path": str(dest_latent_path),
                    "recon_path": recon_path_str,
                    "height": orig_h,
                    "width": orig_w,
                    "padded_height": padded_h,
                    "padded_width": padded_w,
                    "bpp_actual": bpp_actual,
                    "bpp_est": None,
                    "psnr": None,
                    "haarpsi": None,
                    "dists": None,
                    "orig_mos": float(orig_mos),
                }
                manifest.add_row(row)
                completed_count += 1

            except Exception as e:
                print(f"[!] Error processing {image_id} lambda={lambda_idx}: {e}")
                log_failure(failure_log_path, dataset_name, image_id, lambda_idx, str(e))
                failed_count += 1

        if (img_idx + 1) % 5 == 0 or (img_idx + 1) == len(sharded_records):
            manifest.flush()
            print(
                f"Progress: {img_idx + 1}/{len(sharded_records)} images processed "
                f"({completed_count} generated, {skipped_count} skipped, {failed_count} failures).",
                flush=True,
            )

    manifest.flush()
    print("\n--- Data Generation Finished ---")
    print(f"Total Generated: {completed_count}, Skipped (existing): {skipped_count}, Failures: {failed_count}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
