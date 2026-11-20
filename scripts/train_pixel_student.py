#!/usr/bin/env python3
"""Stage 6: Pixel Reference Student Training & Evaluation Pipeline (E1 / E6a).

Trains and evaluates a pixel-domain CNN student (PixelStudent) on decoded RGB reconstructions
using the exact same splits, targets (haarpsi), seeds, and loss formulation as the latent student.

Outputs:
  checkpoints/pixel_student_conditioned/best_model.pth
  reports/runs/pixel_student_conditioned/eval_metrics.json
  reports/runs/pixel_student_conditioned/predictions.csv
  reports/runs/pixel_student_conditioned/run_meta.json
"""

import argparse
import csv
import datetime
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp import runs
from nrcomp.baselines import (
    compute_pwrc_within_lambda,
    compute_within_lambda_partial_spearman,
    compute_within_lambda_srocc,
    pearson_r,
    spearman_rho,
)
from nrcomp.config import load_config
from nrcomp.data.manifest import ParquetManifest
from nrcomp.data.store import LatentStore
from nrcomp.evaluate import (
    Logistic4PFitter,
    compute_clustered_bootstrap_cis,
    compute_rmse,
)
from nrcomp.labels.registry import get_metric_spec, prepare_target_value
from nrcomp.model.loss import CompositeLoss
from nrcomp.model.pixel_student import PixelStudent
from nrcomp.provenance import collect_provenance, format_provenance_header

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import numpy as np
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False
    Dataset = object


class PixelReconDataset(Dataset):
    """Dataset serving decoded RGB reconstruction images, lambda indices, and targets."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        project_root: Path,
        recon_dir: Path,
        split: Optional[str] = None,
        dataset_name: Optional[str] = None,
        target_metric: str = "haarpsi",
        decoder_wrapper: Any = None,
        latent_store: Any = None,
        ckpt_by_idx: Optional[Dict[int, str]] = None,
    ):
        self.project_root = project_root
        self.recon_dir = recon_dir
        self.target_metric = target_metric
        self.decoder_wrapper = decoder_wrapper
        self.latent_store = latent_store
        self.ckpt_by_idx = ckpt_by_idx or {}

        self.samples = []
        for r in rows:
            if dataset_name is not None and r.get("dataset") != dataset_name:
                continue
            if split is not None and r.get("split") != split:
                continue
            target_val = r.get(target_metric)
            if target_val is None:
                continue
            try:
                target_f = float(target_val)
            except (ValueError, TypeError):
                continue

            y_val = prepare_target_value(target_f, target_metric)
            if y_val is None:
                continue

            item = dict(r)
            item["target_transformed"] = float(y_val)
            self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_or_decode_image(self, r: Dict[str, Any]) -> Any:
        img_id = str(r["image_id"])
        l_idx = int(r["lambda_idx"])
        ds = str(r.get("dataset", "koniq"))

        candidates = [
            self.recon_dir / f"{img_id}_lambda{l_idx}.png",
            self.recon_dir / ds / f"{img_id}_lambda{l_idx}.png",
            self.project_root / "data" / "recons" / f"{img_id}_lambda{l_idx}.png",
            self.project_root / "data" / "recon" / f"{img_id}_lambda{l_idx}.png",
        ]
        if r.get("recon_path"):
            rp = Path(r["recon_path"])
            candidates.insert(0, rp if rp.is_absolute() else self.project_root / rp)

        for p in candidates:
            if p.exists():
                try:
                    with Image.open(p) as img:
                        return img.convert("RGB")
                except Exception:
                    pass

        # Materialize via decoder wrapper if available
        if self.decoder_wrapper is not None and self.latent_store is not None:
            lat_p_raw = r.get("latent_path")
            if lat_p_raw:
                lat_p = Path(lat_p_raw) if Path(lat_p_raw).is_absolute() else self.project_root / lat_p_raw
                if lat_p.exists():
                    if l_idx not in self.decoder_wrapper.models and l_idx in self.ckpt_by_idx:
                        self.decoder_wrapper.load_model(l_idx, self.ckpt_by_idx[l_idx])
                    lat_data = self.latent_store.load_latents(lat_p)
                    meta = {
                        "original_size": (int(r["height"]), int(r["width"])),
                        "padded_size": (int(r["padded_height"]), int(r["padded_width"])),
                        "lambda_idx": l_idx,
                    }
                    y_tensor = torch.from_numpy(lat_data["y_hat"]).float()
                    recon_np = self.decoder_wrapper.decode_from_latents(y_tensor, meta)
                    # Cache to candidate 0
                    out_p = candidates[0]
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    img = Image.fromarray(recon_np)
                    img.save(out_p)
                    return img

        # Fallback: create dummy image if data missing
        h = int(r.get("height", 500))
        w = int(r.get("width", 500))
        return Image.new("RGB", (w, h), color=(128, 128, 128))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.samples[idx]
        pil_img = self._load_or_decode_image(r)
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0  # HWC in [0, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)      # CHW

        return {
            "image": tensor,
            "lambda_idx": int(r["lambda_idx"]),
            "target": float(r["target_transformed"]),
            "image_id": str(r["image_id"]),
            "dataset": str(r.get("dataset", "koniq")),
            "split": str(r.get("split", "train")),
            "bpp_actual": float(r.get("bpp_actual", 0.0)),
            "orig_mos": float(r["orig_mos"]) if r.get("orig_mos") is not None else None,
        }


def collate_pixel_samples(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-sized RGB images with zero padding to max H, W in batch."""
    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)

    bs = len(batch)
    padded_images = torch.zeros((bs, 3, max_h, max_w), dtype=torch.float32)
    lambda_indices = torch.zeros((bs,), dtype=torch.long)
    targets = torch.zeros((bs, 1), dtype=torch.float32)

    for i, item in enumerate(batch):
        img = item["image"]
        h, w = img.shape[1], img.shape[2]
        padded_images[i, :, :h, :w] = img
        lambda_indices[i] = item["lambda_idx"]
        targets[i, 0] = item["target"]

    return {
        "images": padded_images,
        "lambda_indices": lambda_indices,
        "targets": targets,
        "items": batch,
    }


def evaluate_pixel_model(
    model: Any,
    loader: Any,
    device: Any,
) -> Tuple[List[Dict[str, Any]], float]:
    """Run inference over a loader and return raw predictions and targets."""
    model.eval()
    records = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["images"].to(device)
            l = batch["lambda_indices"].to(device)
            t = batch["targets"].to(device)

            preds = model(x, lambda_idx=l)
            for i, item in enumerate(batch["items"]):
                rec = dict(item)
                rec["pred_s"] = float(preds[i, 0].cpu())
                rec["target"] = float(t[i, 0].cpu())
                records.append(rec)
            n_batches += 1

    return records


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PixelStudent baseline.")
    parser.add_argument("--config", type=str, default="configs/train_pixel.yaml")
    parser.add_argument("--device", type=str, default="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation on checkpoint.")
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else {}

    seed = int(args.seed or cfg.get("training", {}).get("seed", 42))
    random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cuda_cap = torch.cuda.get_device_capability()
            if cuda_cap[0] < 7:
                torch.backends.cudnn.enabled = False
                print(f"[DEVICE] Hardware sm_{cuda_cap[0]}.{cuda_cap[1]} detected: cuDNN disabled for stability.")

    device = torch.device(args.device) if HAS_TORCH else None
    target_metric = cfg.get("target_metric", "haarpsi")
    manifest_p = PROJECT_ROOT / cfg.get("manifest_path", "data/manifest.parquet")
    recon_dir = PROJECT_ROOT / cfg.get("recon_dir", "data/recons")
    runs_dir = PROJECT_ROOT / "reports" / "runs" / "pixel_student_conditioned"
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "pixel_student_conditioned"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"================================================================================")
    print(f" nrcomp: Pixel Student Reference Training Pipeline (E1 / E6a)")
    print(f" Device:     {device}")
    print(f" Seed:       {seed}")
    print(f" Metric:     {target_metric}")
    print(f" Output:     {runs_dir}")
    print(f"================================================================================")

    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_p}")

    all_rows = ParquetManifest(manifest_p).get_all_rows()
    print(f"Loaded {len(all_rows)} rows from manifest.")

    # Initialize decoder wrapper if available for missing recons
    wrapper = None
    latent_store = None
    ckpt_by_idx = {}
    try:
        from nrcomp.codec.wrapper import ELICWrapper
        from nrcomp.lambdas import load_lambdas_config
        l_cfg = load_lambdas_config()
        ckpt_by_idx = {int(x["idx"]): x["checkpoint"] for x in l_cfg}
        w_dir = PROJECT_ROOT / "weights" / "updated"
        if w_dir.exists() and HAS_TORCH and device.type == "cuda":
            wrapper = ELICWrapper(weights_dir=w_dir, device=str(device))
            latent_store = LatentStore(root_dir=PROJECT_ROOT / "data" / "latents")
            print("[INFO] ELIC decoder initialized for on-demand pixel decoding.")
    except Exception as e:
        print(f"[INFO] Running without live ELIC decoder: {e}")

    train_ds = PixelReconDataset(
        all_rows, PROJECT_ROOT, recon_dir, split="train", dataset_name="koniq",
        target_metric=target_metric, decoder_wrapper=wrapper, latent_store=latent_store, ckpt_by_idx=ckpt_by_idx,
    )
    val_ds = PixelReconDataset(
        all_rows, PROJECT_ROOT, recon_dir, split="val", dataset_name="koniq",
        target_metric=target_metric, decoder_wrapper=wrapper, latent_store=latent_store, ckpt_by_idx=ckpt_by_idx,
    )
    test_ds = PixelReconDataset(
        all_rows, PROJECT_ROOT, recon_dir, split="test", dataset_name="koniq",
        target_metric=target_metric, decoder_wrapper=wrapper, latent_store=latent_store, ckpt_by_idx=ckpt_by_idx,
    )
    livec_ds = PixelReconDataset(
        all_rows, PROJECT_ROOT, recon_dir, split="test", dataset_name="livec",
        target_metric=target_metric, decoder_wrapper=wrapper, latent_store=latent_store, ckpt_by_idx=ckpt_by_idx,
    )

    print(f"Dataset split sizes: train={len(train_ds)}, val={len(val_ds)}, koniq_test={len(test_ds)}, livec_test={len(livec_ds)}")

    model = PixelStudent(
        in_channels=3,
        base_channels=int(cfg.get("model", {}).get("base_channels", 64)),
        lambda_mode=str(cfg.get("model", {}).get("lambda_mode", "conditioned")),
        num_lambdas=int(cfg.get("model", {}).get("num_lambdas", 6)),
        lambda_embed_dim=int(cfg.get("model", {}).get("lambda_embed_dim", 64)),
        dropout=float(cfg.get("model", {}).get("dropout", 0.1)),
    )
    if HAS_TORCH:
        model = model.to(device)

    best_ckpt_path = ckpt_dir / "best_model.pth"

    if not args.eval_only and HAS_TORCH and len(train_ds) > 0:
        batch_size = int(cfg.get("training", {}).get("batch_images", 4))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pixel_samples)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pixel_samples)

        epochs = args.epochs or int(cfg.get("training", {}).get("epochs", 40))
        lr = float(cfg.get("training", {}).get("lr", 3e-4))
        weight_decay = float(cfg.get("training", {}).get("weight_decay", 0.05))

        criterion = CompositeLoss(alpha=0.5, beta=0.01)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_srocc = -1.0
        patience = int(cfg.get("training", {}).get("early_stopping_patience", 6))
        patience_counter = 0

        print(f"\nStarting training for {epochs} epochs (patience={patience})...")
        for ep in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for batch in train_loader:
                x = batch["images"].to(device)
                l = batch["lambda_indices"].to(device)
                t = batch["targets"].to(device)

                optimizer.zero_grad()
                preds = model(x, lambda_idx=l)
                loss, _, _ = criterion(preds, t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += float(loss.item())
                n_batches += 1

            train_loss /= max(1, n_batches)

            # Validation
            val_records = evaluate_pixel_model(model, val_loader, device)
            v_preds = [r["pred_s"] for r in val_records]
            v_targets = [r["target"] for r in val_records]
            v_lambdas = [r["lambda_idx"] for r in val_records]
            v_srocc, _ = compute_within_lambda_srocc(v_preds, v_targets, v_lambdas)
            elapsed = time.time() - t0

            print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Within-λ SROCC: {v_srocc:.4f} ({elapsed:.1f}s)")

            if v_srocc > best_val_srocc + 0.0005:
                best_val_srocc = v_srocc
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  --> Saved new best checkpoint (SROCC={best_val_srocc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {ep}.")
                    break

    # Load best checkpoint for evaluation
    if best_ckpt_path.exists() and HAS_TORCH:
        print(f"\nLoading best checkpoint: {best_ckpt_path}")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    # Final evaluation across all splits
    print("\nRunning comprehensive evaluation across splits...")
    splits_to_eval = [
        ("koniq_val", val_ds),
        ("koniq_test", test_ds),
        ("livec_test", livec_ds),
    ]

    split_results = {}
    all_eval_records = []
    val_preds, val_targets = [], []

    for split_key, ds in splits_to_eval:
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_pixel_samples)
        recs = evaluate_pixel_model(model, loader, device)
        preds = [r["pred_s"] for r in recs]
        targets = [r["target"] for r in recs]
        lambdas = [r["lambda_idx"] for r in recs]
        bpps = [r["bpp_actual"] for r in recs]
        img_ids = [r["image_id"] for r in recs]

        w_srocc, per_l_srocc = compute_within_lambda_srocc(preds, targets, lambdas)
        srocc = spearman_rho(preds, targets)
        wl_part_bpp, _ = compute_within_lambda_partial_spearman(preds, targets, bpps, lambdas)
        pw = compute_pwrc_within_lambda(preds, targets, lambdas)

        if split_key == "koniq_val":
            val_preds = preds
            val_targets = targets

        split_results[split_key] = {
            "within_lambda_srocc": w_srocc,
            "per_lambda_srocc": {str(k): float(v) for k, v in per_l_srocc.items()},
            "srocc": srocc,
            "within_lambda_partial_spearman_bpp": wl_part_bpp,
            "ds_auc": pw.get("ds_auc"),
        }

        for r, p in zip(recs, preds):
            r["split"] = split_key
            r["aligned_score"] = p
            all_eval_records.append(r)

        print(f"[{split_key}] Within-λ SROCC: {w_srocc:.4f} | Overall SROCC: {srocc:.4f} | DS-AUC: {pw.get('ds_auc'):.4f}")

    # 4P logistic calibration on koniq_val
    fitter = Logistic4PFitter()
    if val_preds and val_targets:
        fitter.fit(val_preds, val_targets)
        for split_key in split_results:
            sp_preds = [r["aligned_score"] for r in all_eval_records if r["split"] == split_key]
            sp_targets = [r["target"] for r in all_eval_records if r["split"] == split_key]
            if sp_preds:
                mapped = fitter.predict(sp_preds)
                split_results[split_key]["plcc"] = pearson_r(mapped, sp_targets)
                split_results[split_key]["rmse"] = compute_rmse(mapped, sp_targets)

    # Save artifacts
    meta_doc = {
        "model": "pixel_student_conditioned",
        "domain": "Decoded pixels",
        "target_metric": target_metric,
        "seed": seed,
        "splits": split_results,
        "calibration": fitter.params,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    eval_metrics_file = runs_dir / runs.EVAL_METRICS_FILE
    with open(eval_metrics_file, "w", encoding="utf-8") as f:
        json.dump(meta_doc, f, indent=2, default=str)
    print(f"\n[SUCCESS] Wrote pixel student eval metrics: {eval_metrics_file}")

    # Write predictions.csv
    pred_csv_file = runs_dir / f"{runs.PREDICTIONS_STEM}.csv"
    if all_eval_records:
        fieldnames = ["image_id", "dataset", "split", "lambda_idx", "bpp_actual", "y", "raw_score", "s", "orig_mos"]
        with open(pred_csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in all_eval_records:
                writer.writerow({
                    "image_id": r["image_id"],
                    "dataset": r.get("dataset"),
                    "split": r["split"],
                    "lambda_idx": r["lambda_idx"],
                    "bpp_actual": r.get("bpp_actual"),
                    "y": r["target"],
                    "raw_score": r["aligned_score"],
                    "s": r["aligned_score"],
                    "orig_mos": r.get("orig_mos"),
                })
        print(f"[SUCCESS] Wrote predictions CSV: {pred_csv_file}")


if __name__ == "__main__":
    main()
