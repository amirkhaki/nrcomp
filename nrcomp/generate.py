"""Stage 2: Codec Latent and Reconstruction Generation Pipeline.

CLI Entry point:
    python -m nrcomp.generate --config configs/generate_pilot.yaml [--shard-id 0 --num-shards 1]
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
from nrcomp.data.splits import create_or_load_koniq_splits
from nrcomp.data.store import LatentStore, check_disk_space
from nrcomp.lambdas import load_lambdas_config

DEFAULT_LAMBDAS = load_lambdas_config()


def log_failure(failure_log_path: Path, dataset: str, image_id: str, lambda_idx: int, error_msg: str):
    """Append unexpected generation error to failure log CSV."""
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failure_log_path.exists()
    with open(failure_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "image_id", "lambda_idx", "error"])
        writer.writerow([dataset, image_id, lambda_idx, str(error_msg)])


def collect_dataset_images(
    dataset_cfg: Dict[str, Any], project_root: Path
) -> List[Tuple[str, Path, str, float]]:
    """Collect image records: (dataset_name, file_path, split, orig_mos)."""
    dataset_name = dataset_cfg.get("name", dataset_cfg.get("dataset", "koniq"))
    img_dir_raw = dataset_cfg.get("image_dir", f"data/raw/{dataset_name}")
    img_dir = (project_root / img_dir_raw) if not Path(img_dir_raw).is_absolute() else Path(img_dir_raw)

    if not img_dir.exists():
        # Fallback directory matching (e.g. koniq10k vs koniq-10k, LIVEC vs livec)
        candidates = [
            project_root / "data" / "raw" / "koniq10k",
            project_root / "data" / "raw" / "koniq-10k",
            project_root / "data" / "raw" / "KonIQ-10k",
            project_root / "data" / "raw" / "LIVEC",
            project_root / "data" / "raw" / "livec",
            project_root / "data" / "raw" / "CLIVE",
        ]
        found = False
        for c in candidates:
            if c.exists() and (dataset_name.lower() in c.name.lower() or c.name.lower() in dataset_name.lower()):
                img_dir = c
                found = True
                break
        if not found:
            print(f"[WARN] Image directory not found: {img_dir}")
            return []

    print(f"Using image directory for {dataset_name}: {img_dir}")
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tif", ".tiff"}
    all_img_paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in valid_extensions])
    if not all_img_paths:
        # Check subdirectories (e.g., 1024x768/ or images/)
        all_img_paths = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in valid_extensions])

    # Filter out hidden files / non-images
    all_img_paths = [p for p in all_img_paths if not p.name.startswith(".")]
    print(f"Found {len(all_img_paths)} images for {dataset_name}.")

    # Attempt to load split information
    split_info: Dict[str, Tuple[str, float]] = {}
    split_csv_path = dataset_cfg.get("split_csv")
    if split_csv_path:
        p = (project_root / split_csv_path) if not Path(split_csv_path).is_absolute() else Path(split_csv_path)
        if p.exists():
            with open(p, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_name = row.get("image_name", row.get("image_id", ""))
                    split = row.get("split", "train")
                    mos = float(row.get("mos", 0.0))
                    split_info[img_name] = (split, mos)
                    split_info[Path(img_name).name] = (split, mos)

    # For KonIQ: if split CSV wasn't supplied or found, check for raw scores CSV
    if not split_info and dataset_name == "koniq":
        scores_csv_candidates = [
            img_dir / "koniq10k_scores_and_distributions.csv",
            img_dir.parent / "koniq-10k" / "koniq10k_scores_and_distributions.csv",
            img_dir / "koniq10k_mos.csv",
            project_root / "data" / "raw" / "koniq10k_scores_and_distributions.csv",
            project_root / "data" / "raw" / "koniq-10k" / "koniq10k_scores_and_distributions.csv",
        ]
        # Also search recursively in img_dir, data/raw/, and data/raw/meta_info/
        meta_info_dir = project_root / "data" / "raw" / "meta_info"
        extra_csvs = (
            list(img_dir.rglob("*.csv"))
            + list((project_root / "data" / "raw").glob("*.csv"))
            + (list(meta_info_dir.rglob("*.csv")) if meta_info_dir.exists() else [])
        )
        for p in extra_csvs:
            if p not in scores_csv_candidates:
                scores_csv_candidates.append(p)

        for candidate in scores_csv_candidates:
            if candidate.exists() and candidate.is_file():
                try:
                    out_split_csv = project_root / "data" / "splits" / "koniq_split.csv"
                    create_or_load_koniq_splits(candidate, out_split_csv, seed=0)
                    with open(out_split_csv, "r", newline="") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            img_name = row.get("image_id", row.get("image_name", ""))
                            if not img_name:
                                continue
                            mos_val = float(row.get("orig_mos", row.get("mos", 0.0)))
                            split_val = row.get("split", "train")
                            split_info[img_name] = (split_val, mos_val)
                            split_info[Path(img_name).name] = (split_val, mos_val)
                    if split_info:
                        break
                except Exception:
                    pass

    # Subsetting via max_images if specified in dataset config
    max_images = dataset_cfg.get("max_images")
    if max_images and len(all_img_paths) > max_images:
        if split_info:
            from collections import defaultdict
            by_split = defaultdict(list)
            for p in all_img_paths:
                info = split_info.get(p.name)
                s = info[0] if info else ("test" if dataset_name == "livec" else "train")
                by_split[s].append(p)

            selected_paths = []
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
            if img_path.name in split_info:
                split, mos = split_info[img_path.name]
            else:
                raise ValueError(
                    f"Image '{img_path.name}' in dataset '{dataset_name}' not found in split/scores map! "
                    f"Cannot default to 0.0 MOS (failed filename-to-MOS/split lookup)."
                )
        elif dataset_name in ("koniq", "livec"):
            raise ValueError(
                f"Missing split/scores metadata for '{dataset_name}' dataset. "
                f"Ensure KonIQ/LiveC split CSV or scores file is provided."
            )
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

    # Collect datasets
    datasets_cfgs = cfg.get("datasets")
    if not datasets_cfgs:
        datasets_cfgs = [cfg]

    all_image_records: List[Tuple[str, Path, str, float]] = []
    for d_cfg in datasets_cfgs:
        imgs = collect_dataset_images(d_cfg, PROJECT_ROOT)
        all_image_records.extend(imgs)

    max_images = cfg.get("max_images")
    if max_images and len(all_image_records) > max_images:
        all_image_records = all_image_records[:max_images]

    # Apply sharding
    sharded_records = all_image_records[args.shard_id :: args.num_shards]
    lambdas_cfg = cfg.get("lambdas", DEFAULT_LAMBDAS)

    print(f"Total images: {len(all_image_records)}, Shard {args.shard_id}/{args.num_shards}: {len(sharded_records)} images.")
    print(f"Target lambdas: {len(lambdas_cfg)} operating points.")

    # Initialize manifest and import pilot samples if manifest is currently empty
    manifest = ParquetManifest(manifest_path)
    if len(manifest) == 0:
        for pilot_cand in [
            manifest_path.parent / "manifest_pilot.parquet",
            manifest_path.parent / "manifest_pilot.csv",
        ]:
            if pilot_cand.exists():
                try:
                    p_man = ParquetManifest(pilot_cand)
                    for r in p_man.get_all_rows():
                        manifest.add_row(r)
                    manifest.flush()
                    print(f"Imported {len(manifest)} existing pilot samples from {pilot_cand.name}.")
                    break
                except Exception:
                    pass

    existing_keys = manifest.get_existing_keys()
    needed_samples = 0
    for dataset_name, img_path, _, _ in sharded_records:
        image_id = img_path.name
        for l_item in lambdas_cfg:
            lambda_idx = l_item["idx"]
            if (dataset_name, str(image_id), int(lambda_idx)) not in existing_keys:
                needed_samples += 1

    # Pre-flight disk space check on ungenerated samples
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

    # Initialize codec wrapper and load checkpoints
    wrapper = ELICWrapper(weights_dir=weights_dir, device=device)
    for l_item in lambdas_cfg:
        idx = l_item["idx"]
        ckpt = l_item["checkpoint"]
        print(f"Loading codec lambda {idx} ({l_item['name']}): {ckpt}...")
        wrapper.load_model(idx, ckpt)

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
            lambda_val = float(l_item["name"])

            # Idempotence: check if row already exists in manifest
            if manifest.exists(dataset_name, image_id, lambda_idx):
                skipped_count += 1
                continue

            try:
                # 1. Compress
                bitstream_bytes, meta = wrapper.compress(orig_np, lambda_idx)
                bpp_actual = float(8.0 * len(bitstream_bytes) / (orig_h * orig_w))

                # 2. Extract latents directly from bitstream payload
                latent_dict = wrapper.extract_latents_from_bitstream(bitstream_bytes, meta)
                y_hat = latent_dict["y_hat"]
                z_hat = latent_dict["z_hat"]
                valid_mask = latent_dict["valid_mask"]

                # 3. Save latents
                dest_latent_path = latent_store.get_path(dataset_name, image_id, lambda_idx)
                latent_store.save_latents(
                    dest_latent_path,
                    y_hat=y_hat,
                    z_hat=z_hat if save_z_hat else None,
                    valid_mask=valid_mask,
                )

                # 4. Handle reconstruction
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

                # 5. Add manifest row
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
                    "bpp_est": None,  # Dropped placeholder duplicate; set to None unless computed by entropy model
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

        # Periodic flush to disk every 5 images
        if (img_idx + 1) % 5 == 0 or (img_idx + 1) == len(sharded_records):
            manifest.flush()
            print(
                f"Progress: {img_idx + 1}/{len(sharded_records)} images processed "
                f"({completed_count} generated, {skipped_count} skipped, {failed_count} failures).",
                flush=True,
            )

    # Final flush
    manifest.flush()
    print("\n--- Data Generation Finished ---")
    print(f"Total Generated: {completed_count}, Skipped (existing): {skipped_count}, Failures: {failed_count}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
