#!/usr/bin/env python3
"""Unified training runner for nrcomp.

Executes training across all primary configurations and architectural ablations,
supporting single-seed and multi-seed pipelines:
- Config A Blind (configs/train_blind.yaml)
- Config A Conditioned (configs/train_conditioned.yaml)
- Config B Conditioned (configs/train_config_b.yaml)
- Ablation A2 No-Rank Loss (configs/train_no_rank.yaml)
- Ablation A2 Pure-Rank Loss (configs/train_pure_rank.yaml)
- Ablation A7 Unmasked Pooling (configs/train_unmasked.yaml)
- Ablation A5 DISTS Teacher (configs/train_teacher_dists.yaml)
- Ablation A9 L1 / Huber Loss (configs/train_loss_l1.yaml)
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PRIMARY_CONFIGS = [
    "configs/train_blind.yaml",
    "configs/train_conditioned.yaml",
    "configs/train_config_b.yaml",
]

DEFAULT_ABLATION_CONFIGS = [
    "configs/train_no_rank.yaml",
    "configs/train_pure_rank.yaml",
    "configs/train_unmasked.yaml",
    "configs/train_teacher_dists.yaml",
    "configs/train_loss_l1.yaml",
]


def resolve_config_path(cfg_str: str) -> Path:
    p = Path(cfg_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Run training across all configurations and ablations."
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated list of YAML configs to train. Overrides defaults.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Train only the 3 primary models (blind_a, conditioned_a, conditioned_b).",
    )
    parser.add_argument(
        "--ablations-only",
        action="store_true",
        help="Train only the architectural ablation models.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated list of random seeds (e.g., '42' or '42,43,44').",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifest.parquet",
        help="Path to manifest file.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run Stage 5 evaluation suite (eval_main.md) after training completes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be executed without running them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue training remaining configs if one fails.",
    )
    args = parser.parse_args()

    # Determine configs to run
    if args.configs:
        config_paths = [resolve_config_path(c.strip()) for c in args.configs.split(",") if c.strip()]
    elif args.primary_only:
        config_paths = [resolve_config_path(c) for c in DEFAULT_PRIMARY_CONFIGS]
    elif args.ablations_only:
        config_paths = [resolve_config_path(c) for c in DEFAULT_ABLATION_CONFIGS]
    else:
        config_paths = [resolve_config_path(c) for c in DEFAULT_PRIMARY_CONFIGS + DEFAULT_ABLATION_CONFIGS]

    # Verify configs exist
    missing_configs = [c for c in config_paths if not c.exists()]
    if missing_configs:
        print(f"[ERROR] The following config files were not found: {missing_configs}")
        sys.exit(1)

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Validate manifest
    manifest_p = (PROJECT_ROOT / args.manifest) if not Path(args.manifest).is_absolute() else Path(args.manifest)
    if not manifest_p.exists() and not manifest_p.with_suffix(".csv").exists():
        if args.dry_run:
            print(f"[WARN] Manifest file not found at {manifest_p} (ignoring for dry-run).")
        else:
            print(f"[ERROR] Manifest file not found at {manifest_p}. Please generate latents and manifest first.")
            sys.exit(1)

    print("================================================================================")
    print(" nrcomp: Comprehensive Model Training & Ablation Runner")
    print(f" Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f" Manifest:  {manifest_p}")
    print(f" Seeds:     {seeds}")
    print(f" Configs ({len(config_paths)}):")
    for cp in config_paths:
        print(f"   - {cp.relative_to(PROJECT_ROOT)}")
    print(f" Total Runs to Execute: {len(seeds) * len(config_paths)}")
    print("================================================================================")

    if args.dry_run:
        print("\n[DRY RUN MODE] The following commands would be executed:")
        for seed in seeds:
            for cp in config_paths:
                cmd = [sys.executable, "-m", "nrcomp.train", "--config", str(cp.relative_to(PROJECT_ROOT)), "--seed", str(seed)]
                print(f"  {' '.join(cmd)}")
        if args.eval:
            print(f"  {sys.executable} -m nrcomp.evaluate --config configs/evaluate.yaml")
            if len(seeds) > 1:
                print(f"  {sys.executable} scripts/aggregate_seeds.py --seeds {','.join(str(s) for s in seeds)}")
            print(f"  {sys.executable} -m nrcomp.eval_pixel_refs --config configs/eval_refs.yaml")
        print("\n[DRY RUN COMPLETE] Exiting.")
        return

    # Execute training runs
    results: List[Tuple[str, int, float, str]] = []
    overall_start_time = time.time()

    for seed in seeds:
        for cp in config_paths:
            cfg_rel = str(cp.relative_to(PROJECT_ROOT))
            print(f"\n>>> Starting training: config={cfg_rel} | seed={seed}")
            cmd = [
                sys.executable,
                "-m",
                "nrcomp.train",
                "--config",
                cfg_rel,
                "--seed",
                str(seed),
            ]

            t0 = time.time()
            try:
                # Stream output unbuffered
                ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
                elapsed = time.time() - t0
                status = "SUCCESS"
                print(f">>> [SUCCESS] {cfg_rel} (seed {seed}) finished in {elapsed / 60:.1f}m")
            except subprocess.CalledProcessError as e:
                elapsed = time.time() - t0
                status = f"FAILED (exit {e.returncode})"
                print(f">>> [ERROR] {cfg_rel} (seed {seed}) failed after {elapsed / 60:.1f}m: {e}")
                if not args.continue_on_error:
                    print("[ABORT] Stopping due to error. Pass --continue-on-error to keep going.")
                    results.append((cfg_rel, seed, elapsed, status))
                    break

            results.append((cfg_rel, seed, elapsed, status))

    total_elapsed = time.time() - overall_start_time

    # Print summary table
    print("\n================================================================================")
    print(" Training Summary")
    print("================================================================================")
    print(f"{'Config':<35} {'Seed':<8} {'Duration':<12} {'Status'}")
    print("-" * 75)
    for cfg_name, s, dt, st in results:
        print(f"{cfg_name:<35} {s:<8} {dt / 60:.1f} min     {st}")
    print("-" * 75)
    print(f"Total Time: {total_elapsed / 60:.1f} minutes")
    print("================================================================================")

    # Stage 5 Evaluation & Multi-Seed Aggregation
    if args.eval:
        print("\n[EVALUATION] Running Stage 5 Evaluation (python -m nrcomp.evaluate)...")
        eval_cmd = [
            sys.executable,
            "-m",
            "nrcomp.evaluate",
            "--config",
            "configs/evaluate.yaml",
            "--output",
            "reports/eval_main.md",
        ]
        try:
            subprocess.run(eval_cmd, cwd=str(PROJECT_ROOT), check=True)
            print("[SUCCESS] Stage 5 Evaluation completed -> reports/eval_main.md")
        except Exception as e:
            print(f"[WARN] Stage 5 Evaluation failed: {e}")

        if len(seeds) > 1:
            print("\n[AGGREGATION] Running scripts/aggregate_seeds.py...")
            agg_cmd = [
                sys.executable,
                "scripts/aggregate_seeds.py",
                "--seeds",
                ",".join(str(s) for s in seeds),
                "--manifest",
                str(manifest_p.relative_to(PROJECT_ROOT) if manifest_p.is_relative_to(PROJECT_ROOT) else manifest_p),
            ]
            try:
                subprocess.run(agg_cmd, cwd=str(PROJECT_ROOT), check=True)
                print("[SUCCESS] Multi-seed aggregation completed -> reports/seed_aggregation.md")
            except Exception as e:
                print(f"[WARN] Multi-seed aggregation failed: {e}")

        print("\n[EVALUATION] Running Stage 6 Pixel References & Ablations (python -m nrcomp.eval_pixel_refs)...")
        refs_cmd = [
            sys.executable,
            "-m",
            "nrcomp.eval_pixel_refs",
            "--config",
            "configs/eval_refs.yaml",
        ]
        try:
            subprocess.run(refs_cmd, cwd=str(PROJECT_ROOT), check=True)
            print("[SUCCESS] Stage 6 Evaluation completed -> reports/eval_refs.md & reports/ablations.md")
        except Exception as e:
            print(f"[WARN] Stage 6 Evaluation failed: {e}")
    elif len(seeds) > 1:
        # Check if evaluation artifacts already exist
        runs_dir = PROJECT_ROOT / "reports" / "runs"
        eval_artifacts = list(runs_dir.glob("*/eval_metrics.json")) if runs_dir.exists() else []
        if eval_artifacts:
            print("\n[AGGREGATION] Found existing eval artifacts. Running scripts/aggregate_seeds.py...")
            agg_cmd = [
                sys.executable,
                "scripts/aggregate_seeds.py",
                "--seeds",
                ",".join(str(s) for s in seeds),
                "--manifest",
                str(manifest_p.relative_to(PROJECT_ROOT) if manifest_p.is_relative_to(PROJECT_ROOT) else manifest_p),
            ]
            try:
                subprocess.run(agg_cmd, cwd=str(PROJECT_ROOT), check=True)
                print("[SUCCESS] Multi-seed aggregation completed -> reports/seed_aggregation.md")
            except Exception as e:
                print(f"[WARN] Multi-seed aggregation failed: {e}")
        else:
            print("\n[INFO] Multi-seed training finished. Run 'make eval-main' to evaluate models and generate reports.")


if __name__ == "__main__":
    main()
