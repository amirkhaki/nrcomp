"""Stage 4: Latent Student Training Pipeline.

CLI Entry point:
    python -m nrcomp.train --config configs/train_blind.yaml [--seed 42] [--device cuda]

Artifacts of one run (see nrcomp/runs.py; the run name ALWAYS carries the seed suffix):
    checkpoints/{config}_seed{N}/best_model.pth, latest_model.pth
    reports/runs/{config}_seed{N}/run_meta.json, metrics.json, history.csv, curves.png

Integrity guards:
  * Training refuses to start when the train/val splits are tiny (default < 100 train or
    < 30 val images), because a 27-image "train" split silently happened once.
  * run_meta.json records the EFFECTIVE hyperparameters actually used plus full provenance.
  * The best checkpoint is saved whenever the validation score strictly improves
    (model selection uses koniq_val only; test splits are never touched here).
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
from nrcomp.labels.registry import get_metric_spec
from nrcomp.model.loss import CompositeLoss
from nrcomp.model.student import LatentStudent
from nrcomp.provenance import collect_provenance, git_sha, sha256_file

# Single place for defaults, shared by the training loop and run_meta.json so the recorded
# values can never disagree with what was actually used.
TRAIN_DEFAULTS: Dict[str, Any] = {
    "epochs": 60,
    "warmup_epochs": 2,
    "batch_images": 8,
    "lambdas_per_image": 3,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "grad_clip": 1.0,
    "precision": "bf16",
    "num_workers": 0,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0005,
    "min_train_images": 100,
    "min_val_images": 30,
}
LOSS_DEFAULTS: Dict[str, Any] = {"smooth_l1_beta": 0.01, "alpha_rank": 0.5, "rank_margin": 0.0}
MODEL_DEFAULTS: Dict[str, Any] = {
    "config": "A",
    "in_channels_y": 320,
    "in_channels_z": 192,
    "width": 256,
    "num_res_blocks": 6,
    "lambda_mode": "blind",
    "num_lambdas": 6,
    "lambda_embed_dim": 256,
    "dropout": 0.1,
    "use_mask": True,
    "kernel_size": 7,
    "expansion": 4,
    "layer_scale_init": 0.1,
}


def resolve_effective_config(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Merge a YAML config with the defaults actually used by the training loop."""
    model = dict(MODEL_DEFAULTS)
    model.update(cfg.get("model", {}) or {})
    if "in_channels" in model and "in_channels_y" not in (cfg.get("model", {}) or {}):
        model["in_channels_y"] = model["in_channels"]
    model.pop("in_channels", None)
    model["config"] = str(model["config"]).upper()
    model["lambda_mode"] = str(model["lambda_mode"]).lower()

    training = dict(TRAIN_DEFAULTS)
    training.update(cfg.get("training", {}) or {})
    loss = dict(LOSS_DEFAULTS)
    loss.update(cfg.get("loss", {}) or {})
    return {"model": model, "training": training, "loss": loss}


def create_run_meta(
    cfg: Dict[str, Any],
    out_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
    manifest_path: Optional[Union[str, Path]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create and write run_meta.json (full provenance + EFFECTIVE hyperparameters)."""
    target_metric = str(cfg.get("target_metric", "haarpsi")).lower()
    metric_spec = get_metric_spec(target_metric)
    eff = resolve_effective_config(cfg)

    man_p = Path(manifest_path) if manifest_path else Path(cfg.get("manifest_path", "data/manifest.parquet"))
    if not man_p.is_absolute():
        man_p = PROJECT_ROOT / man_p
    cfg_p = Path(config_path) if config_path else None

    config_name = cfg.get("config_name", "unnamed_config")
    run_meta: Dict[str, Any] = {
        "run_name": runs.run_name(config_name, seed),
        "config_name": config_name,
        "seed": seed,
        "target_metric": target_metric,
        "target_transform": metric_spec.transform,
        "effective": eff,
        "provenance": collect_provenance(
            config_path=cfg_p, manifest_path=man_p, seeds=[seed]
        ),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    # Flat copies kept for easy grepping / backwards compatibility.
    run_meta["git_sha"] = run_meta["provenance"]["git_sha"]
    run_meta["config_sha256"] = run_meta["provenance"]["config_sha256"]
    run_meta["manifest_sha256"] = run_meta["provenance"]["manifest_sha256"]
    run_meta["lambdas_sha256"] = run_meta["provenance"]["lambdas_sha256"]
    run_meta["precision"] = eff["training"]["precision"]
    run_meta["lr"] = eff["training"]["lr"]
    run_meta["epochs"] = eff["training"]["epochs"]
    if extra:
        run_meta.update(extra)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, default=str)
    return run_meta


class EarlyStopping:
    """Early stopping handler tracking validation metric improvements.

    `step()` returns True when the score improved by MORE than `min_delta` over the best
    score so far (this resets the patience counter). `is_new_best` is True whenever the
    score is strictly the highest seen so far -- even by less than `min_delta` -- and is
    what the trainer uses to decide whether to save the best checkpoint, so the saved
    checkpoint always corresponds to the best validation score.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0005, warmup_epochs: int = 2):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.is_new_best = False
        self.early_stop = False

    def _tick(self, epoch: int):
        if self.patience > 0 and epoch > self.warmup_epochs:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def step(self, score: float, epoch: int) -> bool:
        self.is_new_best = False
        if score is None or (isinstance(score, float) and math.isnan(score)):
            self._tick(epoch)  # a NaN score is never an improvement
            return False

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.is_new_best = True
            return True

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.is_new_best = True
            return True

        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.is_new_best = True

        self._tick(epoch)
        return False


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
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)
    except ImportError:
        return None


def plot_training_curves(history: List[Dict[str, Any]], output_path: Path, config_name: str):
    """Generate multi-panel plots for loss, SROCC, and learning rate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in history]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Training Progress: {config_name}", fontsize=14, fontweight="bold")

        ax1.plot(epochs, [h["train_loss"] for h in history], label="Train Loss", color="tab:blue", lw=2)
        ax1.plot(epochs, [h["val_loss"] for h in history], label="Val Loss", color="tab:orange", lw=2)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss Curves")
        ax1.grid(True, linestyle="--", alpha=0.6); ax1.legend()

        ax2.plot(epochs, [h["val_within_lambda_srocc"] for h in history], label="Within-Lambda SROCC", color="tab:green", lw=2)
        ax2.plot(epochs, [h["val_srocc"] for h in history], label="Overall SROCC", color="tab:purple", lw=2, linestyle="--")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("SROCC"); ax2.set_title("Validation Correlation")
        ax2.grid(True, linestyle="--", alpha=0.6); ax2.legend()

        ax3.plot(epochs, [h["lr"] for h in history], label="Learning Rate", color="tab:red", lw=2)
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("LR"); ax3.set_title("Learning Rate Schedule")
        ax3.grid(True, linestyle="--", alpha=0.6); ax3.legend()

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"[Plot] Training curves saved to {output_path}")
    except Exception as e:  # plotting must never fail a training run
        print(f"[WARN] Could not plot training curves: {e}")


def count_images(dataset: LatentDataset) -> int:
    return len({str(r["image_id"]) for r in dataset.samples})


def main():
    parser = argparse.ArgumentParser(description="Train Latent Student Model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu).")
    parser.add_argument("--manifest", type=str, default=None, help="Override manifest path.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--allow-small-data",
        action="store_true",
        help="Disable the minimum-split-size guard (smoke tests only; never for reported results).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_name = cfg.get("config_name")
    if not config_name:
        raise ValueError(f"{args.config}: 'config_name' is required")

    target_metric = str(cfg.get("target_metric", "haarpsi")).lower()
    metric_spec = get_metric_spec(target_metric)

    eff = resolve_effective_config(cfg)
    train_cfg, model_cfg, loss_cfg = eff["training"], eff["model"], eff["loss"]

    epochs = int(train_cfg["epochs"])
    warmup_epochs = int(train_cfg["warmup_epochs"])
    batch_images = int(train_cfg["batch_images"])
    lambdas_per_image = int(train_cfg["lambdas_per_image"])
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    grad_clip = float(train_cfg["grad_clip"])
    precision = str(train_cfg["precision"])
    num_workers = int(train_cfg["num_workers"])
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))

    run_name = runs.run_name(config_name, seed)
    run_dir = runs.run_dir(config_name, seed, PROJECT_ROOT)
    ckpt_dir = runs.checkpoint_dir(config_name, seed, PROJECT_ROOT)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest_path_raw = args.manifest or cfg.get("manifest_path", "data/manifest.parquet")
    manifest_path = (PROJECT_ROOT / manifest_path_raw) if not Path(manifest_path_raw).is_absolute() else Path(manifest_path_raw)
    latent_store_dir = PROJECT_ROOT / cfg.get("latent_store_dir", "data/latents")

    early_stopper = EarlyStopping(
        patience=int(train_cfg["early_stopping_patience"]),
        min_delta=float(train_cfg["early_stopping_min_delta"]),
        warmup_epochs=warmup_epochs,
    )

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print("===========================================================")
    print(f" Training Latent Student [{run_name}] on device: {device}")
    print(f" Git Revision: {git_sha()}")
    print("===========================================================")

    if not manifest_path.exists():
        if manifest_path.suffix == ".parquet" and manifest_path.with_suffix(".csv").exists():
            manifest_path = manifest_path.with_suffix(".csv")
        else:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Loading manifest: {manifest_path}")
    all_rows = ParquetManifest(manifest_path).get_all_rows()
    print(f"Loaded {len(all_rows)} total rows from manifest.")

    latent_store = LatentStore(latent_store_dir)

    train_dataset = LatentDataset(
        rows=all_rows, latent_store=latent_store, split="train",
        config=model_cfg["config"], target_metric=target_metric, dataset_name="koniq",
    )
    val_dataset = LatentDataset(
        rows=all_rows, latent_store=latent_store, split="val",
        config=model_cfg["config"], target_metric=target_metric, dataset_name="koniq",
    )
    n_train_images, n_val_images = count_images(train_dataset), count_images(val_dataset)
    print(f"Dataset sizes: train={len(train_dataset)} samples / {n_train_images} images, "
          f"val={len(val_dataset)} samples / {n_val_images} images")

    min_train, min_val = int(train_cfg["min_train_images"]), int(train_cfg["min_val_images"])
    if not args.allow_small_data and (n_train_images < min_train or n_val_images < min_val):
        raise RuntimeError(
            f"Refusing to train: {n_train_images} train images (< {min_train}) / {n_val_images} val images "
            f"(< {min_val}). Run `python scripts/audit_manifest.py` and, if needed, "
            f"`python scripts/rebuild_manifest_meta.py`. (--allow-small-data bypasses this for smoke tests.)"
        )

    print("Computing per-channel normalization statistics from training latents...")
    norm_stats = compute_train_norm_stats(train_dataset, max_samples=500, seed=seed)
    mean_y, std_y = norm_stats["y"]
    mean_z, std_z = norm_stats.get("z", (None, None))
    if model_cfg["config"] == "B" and mean_z is None:
        raise RuntimeError("Config B requires z_hat latents, but none were found in the latent store.")

    batch_sampler = SourceImageBatchSampler(
        train_dataset.get_image_to_indices(),
        batch_images=batch_images,
        lambdas_per_image=lambdas_per_image,
        shuffle=True,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=batch_sampler, collate_fn=collate_latent_samples,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0), worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_images * lambdas_per_image, shuffle=False,
        collate_fn=collate_latent_samples, num_workers=num_workers,
        pin_memory=(device.type == "cuda"), persistent_workers=(num_workers > 0),
    )

    model = LatentStudent(
        config=model_cfg["config"],
        in_channels_y=int(model_cfg["in_channels_y"]),
        in_channels_z=int(model_cfg["in_channels_z"]),
        width=int(model_cfg["width"]),
        num_res_blocks=int(model_cfg["num_res_blocks"]),
        lambda_mode=model_cfg["lambda_mode"],
        num_lambdas=int(model_cfg["num_lambdas"]),
        lambda_embed_dim=int(model_cfg["lambda_embed_dim"]),
        dropout=float(model_cfg["dropout"]),
        use_mask=bool(model_cfg["use_mask"]),
        kernel_size=int(model_cfg["kernel_size"]),
        expansion=int(model_cfg["expansion"]),
        layer_scale_init=float(model_cfg["layer_scale_init"]),
    )
    model.set_normalization(mean_y=mean_y, std_y=std_y, mean_z=mean_z, std_z=std_z)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = CompositeLoss(
        alpha=float(loss_cfg["alpha_rank"]),
        beta=float(loss_cfg["smooth_l1_beta"]),
        margin=float(loss_cfg["rank_margin"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_cosine_scheduler_with_warmup(optimizer, warmup_epochs, epochs)

    use_autocast = (device.type == "cuda" and precision in ["bf16", "fp16"])
    autocast_dtype = (
        torch.bfloat16 if (precision == "bf16" and torch.cuda.is_bf16_supported()) else torch.float16
    )
    try:
        scaler = torch.amp.GradScaler(device_type=device.type, enabled=(use_autocast and autocast_dtype == torch.float16))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(use_autocast and autocast_dtype == torch.float16))

    # run_meta.json: provenance + effective hyperparameters + the data actually used.
    create_run_meta(
        cfg=cfg,
        out_path=run_dir / runs.RUN_META_FILE,
        config_path=Path(args.config) if Path(args.config).is_absolute() else PROJECT_ROOT / args.config,
        seed=seed,
        manifest_path=manifest_path,
        extra={
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "n_parameters": n_params,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_images": n_train_images,
            "val_images": n_val_images,
            "autocast_dtype": str(autocast_dtype) if use_autocast else None,
        },
    )
    manifest_sha256 = sha256_file(manifest_path)

    history: List[Dict[str, Any]] = []
    print(f"Starting training for up to {epochs} epochs ({n_params / 1e6:.2f}M parameters)...")
    start_time = time.time()

    def _autocast():
        try:
            return torch.amp.autocast(device_type=device.type, enabled=use_autocast, dtype=autocast_dtype)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype)

    def _forward(batch):
        y_hat = batch["y_hat"].to(device, non_blocking=True).float()
        valid_mask = batch["valid_mask"].to(device, non_blocking=True).float()
        z_hat = batch["z_hat"].to(device, non_blocking=True).float() if "z_hat" in batch else None
        lambda_indices = batch["lambda_indices"].to(device, non_blocking=True)
        with _autocast():
            preds = model(
                y_hat=y_hat, valid_mask=valid_mask, z_hat=z_hat,
                lambda_idx=lambda_indices if model_cfg["lambda_mode"] == "conditioned" else None,
            )
        return preds, lambda_indices

    stopped_early = False
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_total = train_l1_total = train_rank_total = 0.0
        train_batches = 0

        for batch in train_loader:
            targets = batch["targets"].to(device, non_blocking=True).float()
            optimizer.zero_grad()
            preds, lambda_indices = _forward(batch)
            loss, loss_dict = criterion(
                predictions=preds, targets=targets, lambda_indices=lambda_indices, image_ids=batch["image_ids"]
            )
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

        if train_batches == 0:
            raise RuntimeError("The training sampler yielded no batches (too few images per batch?).")

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ------------------------------ validation (koniq_val only) ------------------------------
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        all_val_preds: List[float] = []
        all_val_targets: List[float] = []
        all_val_lambdas: List[int] = []
        with torch.no_grad():
            for batch in val_loader:
                targets = batch["targets"].to(device, non_blocking=True).float()
                preds, lambda_indices = _forward(batch)
                _, loss_dict = criterion(
                    predictions=preds, targets=targets, lambda_indices=lambda_indices, image_ids=batch["image_ids"]
                )
                val_loss_total += loss_dict["loss"]
                val_batches += 1
                all_val_preds.extend(preds.detach().cpu().float().tolist())
                all_val_targets.extend(targets.detach().cpu().float().tolist())
                all_val_lambdas.extend(lambda_indices.detach().cpu().tolist())

        avg_val_loss = val_loss_total / max(1, val_batches)
        val_srocc = spearman_rho(all_val_preds, all_val_targets)
        val_plcc = pearson_r(all_val_preds, all_val_targets)
        mean_within_srocc, per_lambda_srocc = compute_within_lambda_srocc(
            all_val_preds, all_val_targets, all_val_lambdas
        )
        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_loss_total / train_batches,
            "train_l1": train_l1_total / train_batches,
            "train_rank": train_rank_total / train_batches,
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

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {record['train_loss']:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val SROCC: {val_srocc:.4f} | "
            f"Within-Lambda SROCC: {mean_within_srocc:.4f} | LR: {current_lr:.2e} ({epoch_time:.1f}s)"
        )

        early_stopper.step(mean_within_srocc, epoch=epoch)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "norm_stats": norm_stats,
            "config": cfg,
            "effective_config": eff,
            "metrics": record,
            "run_name": run_name,
            "config_name": config_name,
            "seed": seed,
            "target_metric": target_metric,
            "manifest_sha256": manifest_sha256,
        }
        if early_stopper.is_new_best:
            torch.save(ckpt, ckpt_dir / runs.BEST_CKPT_FILE)
            print(f"  --> [NEW BEST] Within-Lambda SROCC = {mean_within_srocc:.4f} saved to {ckpt_dir / runs.BEST_CKPT_FILE}")
        elif early_stopper.patience > 0 and epoch > warmup_epochs:
            print(f"  --> Early stopping counter: {early_stopper.counter}/{early_stopper.patience}")

        latest = dict(ckpt)
        latest["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(latest, ckpt_dir / runs.LATEST_CKPT_FILE)

        if early_stopper.early_stop:
            stopped_early = True
            print(
                f"\n[EARLY STOPPING] Triggered at epoch {epoch}: no improvement > {early_stopper.min_delta} "
                f"for {early_stopper.patience} epochs. Best {early_stopper.best_score:.4f} at epoch {early_stopper.best_epoch}."
            )
            break

    # ------------------------------------ artifacts ------------------------------------
    history_path = run_dir / runs.HISTORY_FILE
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    plot_training_curves(history, run_dir / runs.CURVES_FILE, run_name)

    metrics_summary = {
        "run_name": run_name,
        "config_name": config_name,
        "seed": seed,
        "target_metric": target_metric,
        "target_transform": metric_spec.transform,
        "best_epoch": early_stopper.best_epoch,
        "best_within_lambda_srocc": early_stopper.best_score,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "train_images": n_train_images,
        "val_images": n_val_images,
        "history": history,
    }
    with open(run_dir / runs.TRAIN_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("===========================================================")
    print(f" Training Complete in {(time.time() - start_time) / 60:.1f} minutes!")
    print(f" Best Epoch: {early_stopper.best_epoch} with Within-Lambda SROCC = {early_stopper.best_score:.4f}")
    print(f" Best Checkpoint: {ckpt_dir / runs.BEST_CKPT_FILE}")
    print(f" Run artifacts:   {run_dir}")
    print("===========================================================")


if __name__ == "__main__":
    main()
