# Project Handover & Execution Status

**Project:** `nrcomp` — No-Reference Compression Degradation Predictor from Codec Latents via Proxy Labels  
**Date:** November 2026  
**Status:** Milestones M0–M6 completed; **Milestone M7 (Integrity Fixes & Audit Gate)** ready for server execution.

---

## 1. System Architecture & Environment Constraints

> [!IMPORTANT]
> **Strict Separation of Dev Machine vs. Execution Server:**
> - **Local Development Machine:** Linux, standard Python 3.13 only. **Never run `pip install torch` or heavy ML packages locally.** All unit tests in `tests/unit/` are pure Python (using mocks, lists, or standard library) and must run cleanly with `python3 -m unittest discover tests/unit` (76/76 passing).
> - **Remote Execution Server:** Linux, CUDA 12.4, working directory `/storage/nrcomp`.
>   - Conda environment `nrcomp-codec`: Python 3.9, CompressAI 1.1.5, PyTorch with CUDA (runs Stage 1 sanity & Stage 2 data generation).
>   - Conda environment `nrcomp-main`: Python 3.10, PyTorch with CUDA, PyIQA, PIQ, timm, pandas, pyarrow, CompressAI 1.1.5 with GNU C++ toolchain (`gxx_linux-64`). Runs Stage 3 labelling, Stage 4 training, and Stage 5 evaluation.
> - **Output Streaming:** Always pass `--no-capture-output` to `conda run` to ensure unbuffered live terminal output.
> - **Submodule Constraint:** `third_party/ELiC-ReImplemetation` is pinned to commit `92a9ece1e6e1a188a12dfd7a58f9b51c554f9f2d`. Never modify files inside `third_party/` directly; all compatibility patches reside in `nrcomp/codec/wrapper.py`.

---

## 2. Integrity Ground Rules & Results Policy

1. **No Fabricated Numbers:** Never hardcode metric values, tables, "findings", or conclusions in code or reports. Every number in any report must be read dynamically from a verified run artifact (`metrics.json`, `predictions.parquet`) produced by an actual execution run. If an input artifact is missing, report generators must write `"NOT RUN"` or raise an error.
2. **Mandatory Provenance Header:** Every generated report (`reports/*.md`) and run summary must begin with a complete provenance header containing Git SHA, config path and SHA-256, manifest path and SHA-256, lambdas config SHA-256, seed, timestamp, and input artifact paths.
3. **Never Pool Splits:** Never pool validation, test, or cross-dataset samples. Report `koniq_val`, `koniq_test`, and `livec` separately everywhere.
4. **No Tuning on Test Sets:** Model selection and hyperparameter tuning use `koniq_val` only. Test splits (`koniq_test`, `livec`) are evaluated strictly on final frozen checkpoints.
5. **Single Source of Truth for Lambdas:** All scripts, configs, and reports read codec operating points exclusively from `configs/lambdas.yaml` via `nrcomp.lambdas`.
6. **Codec Immutability:** Never edit `third_party/`. All wrappers and patches reside in `nrcomp/codec/wrapper.py`.

---

## 3. Milestone Progress Overview

| Milestone | Phase | Description | Status |
| :--- | :--- | :--- | :--- |
| **M0** | Scaffolding | Repo skeleton, environments, offline test suite | Complete |
| **M1** | Codec Sanity | ELIC wrapper, bit-exact check (20/20 bit-identical) | Complete |
| **M2** | Pilot | 100 KonIQ images $\times$ 6 lambdas pilot generation & diagnostics | Complete |
| **M3** | Full Dataset | 1,000 images $\times$ 6 lambdas (6,000 samples) in `data/manifest.parquet` | Complete |
| **M4** | Baselines & Config A | Baselines (a), (b), (c) + Student Config A (blind & conditioned) | Complete |
| **M5** | Multi-Modal Config B | Student Config B ($\hat{y} + \hat{z}$); initial evaluation suite | UNVERIFIED (Report archived pending M7) |
| **M6** | Pixel Refs & Protocol | Pixel student framework & human study protocol draft | UNVERIFIED (Report archived pending M7) |
| **M7** | **Integrity Fixes** | **A1–A6 fixes: manifest audit, MOS backfill, lambda source of truth** | **READY FOR SERVER GATE** |
| **M8** | **Eval Stats & Teachers** | **B1–B7: Headline metrics, PWRC, paired tests, teacher matrix, multi-seed agg** | **IMPLEMENTED (Ready for Server Run)** |
| **M9** | Stronger Baselines | C1–C5: Per-lambda bpp, global bpp+lambda, latent GBDT/Ridge | Planned |
| **M10**| Ablations | D1–D5: A2–A10 architectural ablations, data-scaling, flip aug | Planned |
| **M11**| Pixel Ref & Latency | E1–E3: Train real pixel CNN student, real hardware latency | Planned |
| **M12**| Synthesis | F1–F4: Paper-ready figures, final synthesis report, signoff | Planned |

---

## 4. Phase A (Milestone M7) & Phase B (Milestone M8) Completed Work

- **A1. Removed Hardcoded Results:**
  - Rewrote `nrcomp/eval_pixel_refs.py` to parse only verified run artifacts in `reports/runs/{run_name}/metrics.json` or write `"NOT RUN"`.
  - Added guard unit test `tests/unit/test_no_hardcoded_reports.py` that fails if any report generator contains numeric-literal metric tables.
  - Archived previous unverified report to `reports/archive/eval_main_UNVERIFIED.md` with explicit warning banner.
- **A2. Target Metric Registry & Directionality:**
  - Implemented `nrcomp/labels/registry.py` with `MetricSpec` supporting higher-better standardization (transforms like `one_minus` for DISTS and bounds clipping).
  - Updated `nrcomp/data/dataset.py` and `nrcomp/train.py` to log target metric and transform in `run_meta.json`.
  - Added unit test `tests/unit/test_metric_registry.py`.
- **A3. Manifest Audit & MOS Backfill:**
  - Created `scripts/backfill_mos.py`: Non-destructively updates `orig_mos` and `split` in `data/manifest.parquet` from KonIQ scores CSV and LiveC labels without regenerating latents, and drops duplicate `bpp_est`.
  - Created `scripts/audit_manifest.py`: Validates zero nulls, non-zero variance of `orig_mos`, presence of all 6 lambdas per image, no duplicates, valid latent files, and non-duplicate `bpp_est`. Exits with code 1 if any check fails.
  - Fixed `nrcomp/generate.py` to prevent silent fallback to `(train, 0.0)`.
- **A4. Strict Split Separation:**
  - Rewrote `scripts/shortcut_baselines.py` to output separate sections and tables for `koniq_val`, `koniq_test`, and `livec_test`.
  - Replaced silent fallback `{"y_hat": [0.0]}` with explicit `FileNotFoundError`.
- **A5. Single Source of Truth for Lambdas:**
  - Created `configs/lambdas.yaml` with verified ELIC operating points (`0.004, 0.008, 0.016, 0.032, 0.15, 0.45`).
  - Implemented `nrcomp/lambdas.py` helper functions.
  - Purged stale Cheng2020 anchor numbers (`0.0016..0.06`) across documentation and code.
  - Added unit test `tests/unit/test_lambdas_single_source.py`.
- **A6. Stale Documentation & Text Cleaned:**
  - Removed unsupported "~0.75" baseline assertions and "saves dozens of milliseconds" claim from `nrcomp/evaluate.py`.
  - Updated `design.md` to v3 with complete revision log and Section 8 additions.
- **B1. Headline Metrics (Within-$\lambda$ SROCC, RMSE, Within-$\lambda$ Partial Spearman, PWRC & DS-AUC):**
  - Updated `nrcomp/evaluate.py` to evaluate rank metrics on raw predictions and calibrate PLCC/RMSE via validation-fitted 4-parameter logistic mapping.
  - Implemented `compute_within_lambda_partial_spearman(preds, targets, bpps, lambdas)` assessing degradation prediction controlling for bit expenditure at fixed $\lambda$.
  - Implemented `compute_pwrc_within_lambda(preds, targets, lambdas)` evaluating concordance accuracy across threshold differences $T \in [0.00, 0.20]$ and computing Difference-Scaled Area Under the Curve (DS-AUC).
- **B2. Statistical Uncertainty & Paired Hypothesis Tests:**
  - Enhanced `compute_clustered_bootstrap_cis` to compute 95% bootstrap CIs clustered by source image for SROCC, PLCC, RMSE, within-$\lambda$ SROCC, and DS-AUC.
  - Implemented `compute_paired_difference_test` performing image-clustered bootstrap hypothesis tests on model-vs-model differences with two-tailed p-values.
- **B3. Multi-Seed Training Pipeline & Aggregator:**
  - `nrcomp/train.py` supports arbitrary `--seed N` and logs run artifacts into `reports/runs/{config_name}_seed{N}/`.
  - Created `scripts/aggregate_seeds.py` computing mean $\pm$ sample std across seeds, evaluating seed ensembles ($s_{ens} = \frac{1}{K}\sum s_k$), and generating `reports/seed_aggregation.md`.
- **B4. Verified E3 Rerun:**
  - Fully integrated Spearman and partial correlations with `orig_mos` and MAE across MOS tertiles in `nrcomp/evaluate.py`.
- **B5. Teacher Agreement Matrix (E4):**
  - Implemented `compute_teacher_agreement_matrix` generating overall and mean within-$\lambda$ cross-correlation matrices between HaarPSI, 1 - DISTS, PSNR, and bpp, highlighting the empirical content bias divergence.
- **B6. Standardized Predictions Schema:**
  - Implemented `save_standardized_predictions` writing `predictions.parquet` and mirrored `predictions.csv` with standard schema (`image_id`, `dataset`, `split`, `lambda_idx`, `lambda_value`, `y`, `s`, `mapped_pred`, `bpp`, `orig_mos`, `dists`) into each model's run directory.
- **B7. Pre-Registered Teacher Decision Rule:**
  - Formalized Section 10.4 in `design.md` explicitly defining primary teacher criteria, content-rate divergence expectations, symmetric DISTS student benchmarking, and human study arbitration.

---

## 5. GATE M7 & GATE M8: Server Execution Instructions

On the remote execution server (`/storage/nrcomp`), execute the following commands in order:

### Step 1: Pull Latest Code
```bash
cd /storage/nrcomp
git pull origin main
```

### Step 2: Rebuild & Audit Manifest Metadata (Gate M7)
Repair corrupted train/val/test splits and restore true KonIQ (1–5) and LiveC (0–100) MOS:
```bash
conda activate nrcomp-main

# Rebuild manifest metadata and stratified 80/10/10 splits
python scripts/rebuild_manifest_meta.py --manifest data/manifest.parquet

# Run integrity audit (must exit with returncode 0: [AUDIT PASS])
python scripts/audit_manifest.py --manifest data/manifest.parquet
```

### Step 3: Run Shortcut Baselines
```bash
make run-baselines
```

### Step 4: Retrain All Configurations & Ablations Across Seeds
Trains all 8 models across seeds 42, 43, 44 on the repaired splits:
```bash
make train-all-seeds
```
*(Alternatively, to run single-seed training first: `make train-all`)*

### Step 5: Run Main Evaluation & Multi-Seed Aggregation
Evaluates all models and seeds, computes ensemble metrics, and generates multi-seed aggregation:
```bash
make eval-main
```
This produces:
- `reports/runs/{model}_seed{seed}/eval_metrics.json` & predictions
- `reports/runs/{model}_ensemble/eval_metrics.json`
- `reports/eval_main.md`
- `reports/seed_aggregation.md` & `reports/seed_aggregation.json`

### Step 6: Run Pixel References & Ablations Report
```bash
make eval-refs
```
This produces:
- `reports/eval_refs.md`
- `reports/ablations.md`

### Step 7: Verification & Gate Clearance
Confirm that:
1. `audit_manifest.py` printed `[AUDIT PASS] Manifest passed all integrity checks.` and exited with code 0.
2. In `reports/eval_main.md`:
   - All provenance headers show matched SHA-256 hashes.
   - Primary models (Blind A, Conditioned A, Conditioned B) show valid Within-$\lambda$ SROCC and DS-AUC metrics.
   - E3 shows valid correlations with `orig_mos`.
3. In `reports/seed_aggregation.md`:
   - Shows `3 / 3` seeds found for evaluated models with mean $\pm$ std and ensemble delta gains.
4. In `reports/ablations.md`:
   - Shows ablation rows (A2a, A2b, A3, A4, A5, A7, A9) comparing variant vs reference models.
