"""Stage 4: Latent Student Training Pipeline.

CLI Entry point:
    python -m nrcomp.train --config configs/train_blind.yaml [--device cuda]
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.baselines import (
    compute_within_lambda_srocc,
    pearson_r,
    spearman_rho,
)
from nrcomp.config import load_config
from nrcomp.data.dataset import (
    LatentDataset,
    SourceImageBatchSampler,
    collate_latent_samples,
    compute_train_norm_stats,
    seed_worker,
)
from nrcomp.data.manifest import ParquetManifest
from nrcomp.data.store import LatentStore
from nrcomp.model.loss import CompositeLoss
from nrcomp.model.student import LatentStudent


def get_git_revision_hash() -> str:
    """Return git commit hash if available."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def build_cosine_scheduler_with_warmup(
    optimizer: Any, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6
):
    """Build lambda LR scheduler with linear warmup and cosine annealing."""
    try:
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_epoch: int):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(max(1, warmup_epochs))
            progress = float(current_epoch - warmup_epochs) / float(
                max(1, total_epochs - warmup_epochs)
            )
            return max(
                min_lr,
                0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        return LambdaLR(optimizer, lr_lambda)
    except ImportError:
        return None


def plot_training_curves(
    history: List[Dict[str, Any]], output_path: Path, config_name: str
):
    """Generate multi-panel training plots for loss, SROCC, and learning rate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        val_within_srocc = [h["val_within_lambda_srocc"] for h in history]
        val_srocc = [h["val_srocc"] for h in history]
        lrs = [h["lr"] for h in history]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Training Progress: {config_name}", fontsize=14, fontweight="bold")

        # 1. Loss Curves
        ax1.plot(epochs, train_loss, label="Train Loss", color="tab:blue", lw=2)
        ax1.plot(epochs, val_loss, label="Val Loss", color="tab:orange", lw=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # 2. SROCC Curves
        ax2.plot(epochs, val_within_srocc, label="Within-Lambda SROCC", color="tab:green", lw=2)
        ax2.plot(epochs, val_srocc, label="Overall SROCC", color="tab:purple", lw=2, linestyle="--")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("SROCC")
        ax2.set_title("Validation Correlation")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        # 3. Learning Rate Schedule
        ax3.plot(epochs, lrs, label="Learning Rate", color="tab:red", lw=2)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("LR")
        ax3.set_title("Learning Rate Schedule")
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend()

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[Plot] Training curves saved to {output_path}")
    except Exception as e:
        print(f"[WARN] Could not plot training curves: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train Latent Student Model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu).")
    parser.add_argument("--manifest", type=str, default=None, help="Override manifest path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_name = cfg.get("config_name", "student_model")

    # Paths
    manifest_path_raw = args.manifest or cfg.get("manifest_path", "data/manifest.parquet")
    manifest_path = (PROJECT_ROOT / manifest_path_raw) if not Path(manifest_path_raw).is_absolute() else Path(manifest_path_raw)
    latent_store_dir = PROJECT_ROOT / cfg.get("latent_store_dir", "data/latents")

    output_dir_raw = cfg.get("output_dir", f"checkpoints/{config_name}")
    output_dir = (PROJECT_ROOT / output_dir_raw) if not Path(output_dir_raw).is_absolute() else Path(output_dir_raw)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path_raw = cfg.get("history_path", f"reports/training_{config_name}_history.csv")
    history_path = (PROJECT_ROOT / history_path_raw) if not Path(history_path_raw).is_absolute() else Path(history_path_raw)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    plot_path_raw = cfg.get("plot_path", f"reports/training_{config_name}_curves.png")
    plot_path = (PROJECT_ROOT / plot_path_raw) if not Path(plot_path_raw).is_absolute() else Path(plot_path_raw)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    # Training parameters
    train_cfg = cfg.get("training", {})
    epochs = int(train_cfg.get("epochs", 60))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 2))
    batch_images = int(train_cfg.get("batch_images", 8))
    lambdas_per_image = int(train_cfg.get("lambdas_per_image", 3))
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    precision = str(train_cfg.get("precision", "bf16"))
    seed = int(train_cfg.get("seed", 42))
    num_workers = int(train_cfg.get("num_workers", 4))

    # Model parameters
    model_cfg = cfg.get("model", {})
    model_config_type = str(model_cfg.get("config", "A")).upper()
    in_channels_y = int(model_cfg.get("in_channels", model_cfg.get("in_channels_y", 320)))
    in_channels_z = int(model_cfg.get("in_channels_z", 192))
    width = int(model_cfg.get("width", 256))
    num_res_blocks = int(model_cfg.get("num_res_blocks", 6))
    lambda_mode = str(model_cfg.get("lambda_mode", "blind")).lower()
    num_lambdas = int(model_cfg.get("num_lambdas", 6))
    lambda_embed_dim = int(model_cfg.get("lambda_embed_dim", 256))
    dropout = float(model_cfg.get("dropout", 0.1))

    # Loss parameters
    loss_cfg = cfg.get("loss", {})
    smooth_l1_beta = float(loss_cfg.get("smooth_l1_beta", 0.05))
    alpha_rank = float(loss_cfg.get("alpha_rank", 0.5))
    rank_margin = float(loss_cfg.get("rank_margin", 0.0))

    # Seeding
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"===========================================================")
    print(f" Training Latent Student [{config_name}] on device: {device}")
    print(f" Git Revision: {get_git_revision_hash()}")
    print(f"===========================================================")

    # Load Manifest
    if not manifest_path.exists():
        if manifest_path.suffix == ".parquet" and manifest_path.with_suffix(".csv").exists():
            manifest_path = manifest_path.with_suffix(".csv")
        else:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Loading manifest: {manifest_path}")
    manifest = ParquetManifest(manifest_path)
    all_rows = manifest.get_all_rows()
    print(f"Loaded {len(all_rows)} total rows from manifest.")

    latent_store = LatentStore(latent_store_dir)

    # 1. Initialize Datasets
    train_dataset = LatentDataset(
        rows=all_rows,
        latent_store=latent_store,
        split="train",
        config=model_config_type,
        target_metric="haarpsi",
    )
    val_dataset = LatentDataset(
        rows=all_rows,
        latent_store=latent_store,
        split="val",
        config=model_config_type,
        target_metric="haarpsi",
    )
    print(f"Dataset split sizes: train={len(train_dataset)}, val={len(val_dataset)}")

    # 2. Compute Normalization Statistics on Training Split
    print("Computing per-channel normalization statistics from training latents...")
    norm_stats = compute_train_norm_stats(train_dataset, max_samples=500, seed=seed)
    mean_y, std_y = norm_stats["y"]
    mean_z, std_z = norm_stats.get("z", (None, None))
    print(f"Norm stats computed: y_channels={len(mean_y)}, z_channels={len(mean_z) if mean_z else 0}")

    # 3. Setup DataLoaders
    batch_sampler = SourceImageBatchSampler(
        train_dataset.get_image_to_indices(),
        batch_images=batch_images,
        lambdas_per_image=lambdas_per_image,
        shuffle=True,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_latent_samples,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_images * lambdas_per_image,
        shuffle=False,
        collate_fn=collate_latent_samples,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # 4. Initialize Model
    model = LatentStudent(
        config=model_config_type,
        in_channels_y=in_channels_y,
        in_channels_z=in_channels_z,
        width=width,
        num_res_blocks=num_res_blocks,
        lambda_mode=lambda_mode,
        num_lambdas=num_lambdas,
        lambda_embed_dim=lambda_embed_dim,
        dropout=dropout,
    )
    model.set_normalization(mean_y=mean_y, std_y=std_y, mean_z=mean_z, std_z=std_z)
    model = model.to(device)

    # 5. Loss & Optimizer
    criterion = CompositeLoss(alpha=alpha_rank, beta=smooth_l1_beta, margin=rank_margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_cosine_scheduler_with_warmup(optimizer, warmup_epochs, epochs)

    # Mixed precision setup
    use_autocast = (device.type == "cuda" and precision in ["bf16", "fp16"])
    autocast_dtype = (
        torch.bfloat16
        if (precision == "bf16" and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_autocast and autocast_dtype == torch.float16))

    # Save run metadata
    meta = {
        "config_name": config_name,
        "git_hash": get_git_revision_hash(),
        "config": cfg,
        "device": str(device),
        "precision": precision,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    with open(output_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    history: List[Dict[str, Any]] = []
    best_within_lambda_srocc = -1.0
    best_epoch = -1

    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        train_loss_total = 0.0
        train_l1_total = 0.0
        train_rank_total = 0.0
        train_batches = 0

        for batch in train_loader:
            y_hat = batch["y_hat"].to(device, non_blocking=True).float()
            valid_mask = batch["valid_mask"].to(device, non_blocking=True).float()
            z_hat = batch["z_hat"].to(device, non_blocking=True).float() if "z_hat" in batch else None
            lambda_indices = batch["lambda_indices"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True).float()
            image_ids = batch["image_ids"]

            optimizer.zero_grad()

            try:
                amp_ctx = torch.amp.autocast(device_type=device.type, enabled=use_autocast, dtype=autocast_dtype)
            except (AttributeError, TypeError):
                amp_ctx = torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype)

            with amp_ctx:
                preds = model(
                    y_hat=y_hat,
                    valid_mask=valid_mask,
                    z_hat=z_hat,
                    lambda_idx=lambda_indices if lambda_mode == "conditioned" else None,
                )

            # Compute loss explicitly in fp32
            loss, loss_dict = criterion(
                predictions=preds,
                targets=targets,
                lambda_indices=lambda_indices,
                image_ids=image_ids,
            )

            # Check for NaN immediately
            if math.isnan(loss_dict["loss"]) or math.isinf(loss_dict["loss"]):
                raise FloatingPointError(f"NaN or Inf encountered in training loss at epoch {epoch}!")

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            train_loss_total += loss_dict["loss"]
            train_l1_total += loss_dict["smooth_l1"]
            train_rank_total += loss_dict["rank_loss"]
            train_batches += 1

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        avg_train_loss = train_loss_total / max(1, train_batches)
        avg_train_l1 = train_l1_total / max(1, train_batches)
        avg_train_rank = train_rank_total / max(1, train_batches)

        # -------------------------------------------------------------
        # Validation Loop
        # -------------------------------------------------------------
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        all_val_preds: List[float] = []
        all_val_targets: List[float] = []
        all_val_lambdas: List[int] = []

        with torch.no_grad():
            for batch in val_loader:
                y_hat = batch["y_hat"].to(device, non_blocking=True).float()
                valid_mask = batch["valid_mask"].to(device, non_blocking=True).float()
                z_hat = batch["z_hat"].to(device, non_blocking=True).float() if "z_hat" in batch else None
                lambda_indices = batch["lambda_indices"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True).float()
                image_ids = batch["image_ids"]

                try:
                    amp_ctx = torch.amp.autocast(device_type=device.type, enabled=use_autocast, dtype=autocast_dtype)
                except (AttributeError, TypeError):
                    amp_ctx = torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype)

                with amp_ctx:
                    preds = model(
                        y_hat=y_hat,
                        valid_mask=valid_mask,
                        z_hat=z_hat,
                        lambda_idx=lambda_indices if lambda_mode == "conditioned" else None,
                    )

                loss, loss_dict = criterion(
                    predictions=preds,
                    targets=targets,
                    lambda_indices=lambda_indices,
                    image_ids=image_ids,
                )

                val_loss_total += loss_dict["loss"]
                val_batches += 1

                all_val_preds.extend(preds.detach().cpu().float().tolist())
                all_val_targets.extend(targets.detach().cpu().float().tolist())
                all_val_lambdas.extend(lambda_indices.detach().cpu().tolist())

        avg_val_loss = val_loss_total / max(1, val_batches)

        # Compute validation correlation metrics
        val_srocc = spearman_rho(all_val_preds, all_val_targets)
        val_plcc = pearson_r(all_val_preds, all_val_targets)
        mean_within_srocc, per_lambda_srocc = compute_within_lambda_srocc(
            all_val_preds, all_val_targets, all_val_lambdas
        )

        epoch_time = time.time() - epoch_start

        # Record history
        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_l1": avg_train_l1,
            "train_rank": avg_train_rank,
            "val_loss": avg_val_loss,
            "val_srocc": val_srocc,
            "val_plcc": val_plcc,
            "val_within_lambda_srocc": mean_within_srocc,
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
        }
        for l_idx, rho in per_lambda_srocc.items():
            record[f"val_srocc_lambda_{l_idx}"] = rho

        history.append(record)

        # Print progress
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val SROCC: {val_srocc:.4f} | "
            f"Within-Lambda SROCC: {mean_within_srocc:.4f} | "
            f"LR: {current_lr:.2e} ({epoch_time:.1f}s)"
        )

        # Checkpoint: Latest
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "norm_stats": norm_stats,
            "config": cfg,
            "metrics": record,
        }
        torch.save(checkpoint_dict, output_dir / "latest_model.pth")

        # Checkpoint: Best (based on within-lambda SROCC)
        if mean_within_srocc > best_within_lambda_srocc:
            best_within_lambda_srocc = mean_within_srocc
            best_epoch = epoch
            torch.save(checkpoint_dict, output_dir / "best_model.pth")
            # Also mirror to checkpoints/ or weights/
            weights_best_path = PROJECT_ROOT / "checkpoints" / f"student_{config_name}_best.pt"
            weights_best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_dict, weights_best_path)
            print(f"  --> [NEW BEST] Within-Lambda SROCC = {mean_within_srocc:.4f} saved to {weights_best_path}")

    # Write history CSV
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(history[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"[SUCCESS] Training history logged to {history_path}")

    # Plot Curves
    plot_training_curves(history, plot_path, config_name)

    total_time = time.time() - start_time
    print("===========================================================")
    print(f" Training Complete in {total_time / 60:.1f} minutes!")
    print(f" Best Epoch: {best_epoch} with Within-Lambda SROCC = {best_within_lambda_srocc:.4f}")
    print(f" Best Checkpoint: {output_dir / 'best_model.pth'}")
    print("===========================================================")


if __name__ == "__main__":
    main()
