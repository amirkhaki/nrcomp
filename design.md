# Design Doc: No-Reference Compression Degradation Predictor from Codec Latents via Proxy Labels

Written for implementation by a coding agent.
**Status:** v3 (Latent-domain student with rigorous statistical evaluation, teacher analysis, and integrity guarantees), 18 Nov 2026.

> [!IMPORTANT]
> **Execution Environment Notice for Implementing Agent:** The development machine where the coding agent operates is **not** the execution environment. The development machine lacks GPU, has limited storage, and **cannot have Python packages like `torch` or `compressai` installed**. The agent's role is to write clean, modular code, configs, runner scripts, pure-Python unit tests (`tests/unit/`), server tests (`tests/server/`), and offline mock validations. The user pulls the code to a remote execution server equipped with GPU and storage to run workloads and integration tests (see Sections 5, 13, 14, and 16).

---

## 0. Results Policy & Non-Negotiables

1. **No Fabricated Numbers:** Never hardcode metric values, tables, "findings", or conclusions in code or reports. Every number in any report must be read dynamically from a verified run artifact (`metrics.json`, `predictions.parquet`) produced by an actual execution run. If an input artifact is missing, report generators must write `"NOT RUN"` or raise an error; they must never fall back to plausible mock/placeholder numbers.
2. **Mandatory Provenance Header:** Every generated report (`reports/*.md`) and run summary must begin with a complete provenance header containing:
   - Git commit SHA
   - Config file path and SHA-256 hash
   - Manifest path and SHA-256 hash
   - Lambdas config SHA-256 hash
   - Seed or seed list
   - Generation date and timestamp
   - List of all input artifact paths and their respective SHA-256 hashes
3. **Never Pool Splits:** Never pool validation, test, or cross-dataset samples. Report `koniq_val`, `koniq_test`, and `livec` separately everywhere.
4. **No Tuning on Test Sets:** Model selection, hyperparameter tuning, and decision thresholds use `koniq_val` only. Test splits (`koniq_test`, `livec`) are strictly evaluated on final frozen checkpoints.
5. **Single Source of Truth for Lambdas:** All scripts, configs, and reports read codec operating points exclusively from `configs/lambdas.yaml` via `nrcomp.lambdas`.
6. **Codec Immutability:** Never edit `third_party/`. All wrappers and patches reside in `nrcomp/codec/wrapper.py`.

---

## 0.1 Known Issues & Resolutions (v3 Revision Log)

During the comprehensive review following Milestone M6, several critical methodological and integrity issues were identified and resolved:

| Issue Identified | Root Cause Analysis | Resolution Implemented |
| :--- | :--- | :--- |
| **Hardcoded mock results in `eval_pixel_refs.py`** | Pixel-student and pretrained NR numbers were hardcoded mock dictionaries that leaked into reports. | Rewrote `eval_pixel_refs.py` to parse only real run artifacts (`metrics.json`); added guard test `test_no_hardcoded_reports.py`. |
| **Invalid E3 (orig_mos correlations exactly 0.0000)** | In `generate.py`, CSV key mismatch (`image_name` vs `image_id`) caused lookup failure, silently defaulting all 6,000 samples to MOS 0.0. | Fixed `generate.py` lookup; added `scripts/backfill_mos.py` to rewrite manifest non-destructively; added `scripts/audit_manifest.py` gate. |
| **Target metric hardcoded to HaarPSI in `train.py`** | `train.py` ignored config `target_metric`, meaning DISTS configs would silently train on HaarPSI. | Created `nrcomp/labels/registry.py` (`MetricSpec`, transforms such as `one_minus` for DISTS, bounds clipping, and target metadata logging in `run_meta.json`). |
| **Teacher disagreement within $\lambda$** | At fixed $\lambda$, $bpp$ correlates $-0.32$ with HaarPSI (penalizing high-frequency texture loss) but $+0.53$ with $1 - \text{DISTS}$ (rewarding deep VGG features). Student trained on HaarPSI has negative correlation with DISTS within $\lambda$. | Replaced E4 with comprehensive teacher agreement matrix; pre-registered teacher rule; introduced pairwise ranking metrics (PWRC). |
| **Data scale & variance** | Small dataset scale (645 train, 78 val, 77 test KonIQ images due to 15 GB disk limit) meant $\pm 0.02$ SROCC differences were within standard error (~0.04). Single seed 42. | Mandated $\ge 3$ seeds (target 5) with `scripts/aggregate_seeds.py`; added 95% image-clustered bootstrap CIs and paired difference p-values. |
| **Weak shortcut baselines** | Baseline (a) was fit globally across lambdas, flipping sign within lambda; missing per-lambda bpp and ridge/GBDT baselines. | Implemented separate per-lambda bpp and multivariable latent baselines in `scripts/shortcut_baselines.py`. |
| **Stale lambda references** | Cheng2020 anchor numbers (`0.0016..0.06`) leaked into docs; README 4th lambda ambiguous. | Verified ELIC checkpoints (`0.004, 0.008, 0.016, 0.032, 0.15, 0.45`); established `configs/lambdas.yaml` as single source of truth. |
| **Silent fallbacks** | Missing latents substituted with `{"y_hat": [0.0]}`; `bpp_est` was duplicate of `bpp_actual`. | Replaced fallbacks with explicit errors; removed duplicate `bpp_est` placeholder from manifest. |
| **Reference paper insights (Mohammadi et al. 2026)** | JPEG AIC-3 benchmark showed CVVDP best (0.960), IW-SSIM (0.944), MS-SSIM (0.927); HaarPSI degrades above 1 JND. | Updated E7 human study protocol (2AFC / boosted triplets); structured teacher evaluations across JND regimes. |

---

## 1. Goal

Train a no-reference (NR) model that predicts compression degradation from the learned codec's own latents, without decoding to pixels. The input is the quantized latent of an ELIC-style codec ($\hat{y} \in \mathbb{R}^{320 \times H/16 \times W/16}$), optionally with the hyperprior latent ($\hat{z} \in \mathbb{R}^{192 \times H/64 \times W/64}$). The output is how much the compression degraded the image relative to the unseen original, as measured by a full-reference (FR) metric.

Labels are generated, not collected: encode source images (KonIQ-10k, LiveC) with ELIC at every verified operating point ($\lambda \in \{0.004, 0.008, 0.016, 0.032, 0.15, 0.45\}$), decode each reconstruction once to compute proxy labels (HaarPSI, DISTS), and train the latent student. Pixels are used only to compute labels and for a pixel-domain reference baseline. The student never sees pixels at inference time.

Underlying research question: how much does compression affect image quality, independent of the original image's own quality?

**Claim scope.** In v1 the model is specific to ELIC latents and predicts proxy-label-measured degradation. Claims about human-perceived quality require validation against human perceptual judgments on ELIC outputs (see E7). Claims about other codecs require separate training.

---

## 2. Non-goals for v1

- No pixel input to the student. A pixel-domain model is trained only as a reference baseline (E6).
- No multi-codec or codec-agnostic model in v1. The student is tied to ELIC's latent space.
- No ad-hoc ensemble of FR teachers. Alternative teachers (DISTS, CVVDP) are evaluated systematically via pre-registered rules.
- No unverified human study claims in v1 (E7 specifies the protocol as a follow-up).
- No video.

---

## 3. Fixed Decisions

| Decision | Choice | Rationale |
| :--- | :--- | :--- |
| Codec | ELIC reimplementation, one model per lambda, all 6 lambdas used | Project plan |
| Operating points ($\lambda$) | `0.004, 0.008, 0.016, 0.032, 0.15, 0.45` | Verified from `weights/updated/` checkpoints; single source in `configs/lambdas.yaml` |
| Student input | Quantized latent $\hat{y}$ only (config A); optionally plus $\hat{z}$ (config B) | Core goal: NR prediction from latents without decoding |
| Latent source | The codec's real quantized latent, verified bit-exact to entropy-decoded bitstream | Latents must be what a real decoder receives |
| Training images | KonIQ-10k at native resolution, never resized | Resizing alters high-frequency texture and destroys codec artifacts |
| Primary Teacher | HaarPSI(original, reconstruction), standardized to $[0, 1]$, higher = better | Fast, well validated phase-congruency baseline |
| Secondary Teacher | DISTS (mapped via $1 - \text{DISTS}$ to $[0, 1]$, higher = better) | Deep texture similarity baseline to test teacher-specific bias |
| Student target | Standardized scalar in $[0, 1]$, monotonically higher = less degradation | Direct proxy label |
| Lambda handling | Blind mode (no lambda input) and Conditioned mode (learned lambda embedding) | Tests whether latents implicitly encode rate or require explicit conditioning |
| Augmentation | None on latents (latent grids are not equivariant to spatial transforms) | Offline flip augmentation evaluated as separate ablation D4 |
| Cross-dataset test | LiveC compressed with the same ELIC models at native $500 \times 500$ | Cross-content and cross-resolution generalization |
| Splits | Disjoint by source image ID, fixed seed | Prevents content leakage across splits |

---

## 4. Repository Layout

```
nrcomp/
  configs/            # Stage configs and configs/lambdas.yaml (single source of truth)
  third_party/
    ELiC-ReImplemetation/   # git submodule pinned to fixed commit
    ELIC_PIN.md             # Pinned commit SHA and date
  weights/            # ELIC checkpoints, weights/updated/, CHECKSUMS.txt
  data/               # Raw datasets, manifests, latents store, splits
  nrcomp/
    codec/            # ELICWrapper, latent extraction, bitstream I/O, padding arithmetic
    data/             # ParquetManifest, LatentStore, LatentDataset, split logic
    labels/           # registry.py (MetricSpec, transforms), metric wrappers
    model/            # LatentStudent, ConvNeXt blocks, AttentionPooling, CompositeLoss
    train.py          # Multi-seed training, run_meta.json provenance, validation tracking
    evaluate.py       # Headline metrics, bootstrap CIs, paired tests, E1-E5, E8
    generate.py       # Bit-exact latent generation, error handling, manifest writer
    label.py          # FR metric computation into manifest
    eval_pixel_refs.py# Non-fabricated E6 pixel-student & ablations reporting
  scripts/            # audit_manifest.py, backfill_mos.py, shortcut_baselines.py, aggregate_seeds.py
  tests/
    unit/             # Pure Python 3.13 unit tests (runnable offline on dev machine)
    server/           # PyTorch & GPU integration tests (runnable on execution server)
  reports/            # Generated markdown reports, run artifacts, plots
    runs/             # reports/runs/{run_name}_seed{N}/metrics.json, predictions.parquet
    archive/          # Archived unverified reports (e.g. eval_main_UNVERIFIED.md)
  requirements-codec.txt
  requirements-main.txt
```

---

## 5. Environments and Execution Workflow

### 5.1 Separation of Development and Execution Environments
The development machine (coding agent) is strictly decoupled from the execution server:
- **Development Machine (Coding Agent):** Plain Python 3.13 standard library only. **No `torch`, `torchvision`, `compressai`, `numpy`, or `pandas`**. The agent writes code, YAML configs, pure-Python unit tests (`tests/unit/`), and server test scripts (`tests/server/`).
- **Execution Server (Target Environment):** CUDA 12.4, dedicated GPU (~24 GB VRAM), large storage (`/storage/nrcomp`), conda environments `nrcomp-codec` and `nrcomp-main`.
- **Handoff Protocol:** The agent writes clean code, verifies all unit tests locally, updates `HANDOVER.md` with exact, copy-pasteable CLI commands, and waits at each phase gate for server execution results.

---

## 6. Codec Verification & Operating Points

Operating points are strictly verified against the updated checkpoints:
- $\lambda_0 = 0.004$ (`elic_lambda_0.004_updated.pth.tar`)
- $\lambda_1 = 0.008$ (`elic_lambda_0.008_updated.pth.tar`)
- $\lambda_2 = 0.016$ (`elic_lambda_0.016_updated.pth.tar`)
- $\lambda_3 = 0.032$ (`elic_lambda_0.032_updated.pth.tar`)
- $\lambda_4 = 0.15$ (`elic_lambda_0.15_updated.pth.tar`)
- $\lambda_5 = 0.45$ (`elic_lambda_0.45_updated.pth.tar`)

All scripts and configurations import lambdas from `nrcomp.lambdas.load_lambdas_config()`.

---

## 7. Model Architecture

- **Trunk:** Lightweight ConvNeXt-style convolutional blocks operating directly on latent feature grids.
  - Config A: Input $\hat{y} \in \mathbb{R}^{320 \times H_y \times W_y}$.
  - Config B: Inputs $\hat{y}$ and $\hat{z} \in \mathbb{R}^{192 \times H_z \times W_z}$. The hyperprior latent $\hat{z}$ is upsampled $4\times$ via transposed convolution to match $\hat{y}$ resolution and concatenated along channels ($320 + 192 = 512$ channels).
- **Lambda Conditioning:** In conditioned mode, a learned embedding of `lambda_idx` is projected and injected via Feature-wise Linear Modulation (FiLM) or additive bias into intermediate residual blocks. In blind mode, no lambda information is provided.
- **Pooling:** Masked average pooling combined with cross-spatial attention pooling, respecting the binary valid mask $M \in \{0, 1\}^{H_y \times W_y}$ to discard zero-padded border cells.
- **Head:** 2-layer MLP with LayerNorm, GELU, dropout 0.1, outputting scalar degradation prediction $s \in [0, 1]$.

---

## 8. Training Objective & Optimization

- **Composite Loss:** Balanced combination of point regression and pairwise ranking:
  $$\mathcal{L} = (1 - \alpha) \mathcal{L}_{\text{reg}}(s, y) + \alpha \mathcal{L}_{\text{rank}}(s, y)$$
  where $\mathcal{L}_{\text{reg}}$ is Smooth L1 loss ($\beta = 0.01$) and $\mathcal{L}_{\text{rank}}$ is pairwise margin ranking loss over batch pairs:
  $$\mathcal{L}_{\text{rank}} = \frac{1}{|P|} \sum_{(i,j) \in P} \max(0, - \text{sign}(y_i - y_j)(s_i - s_j) + \text{margin})$$
- **Optimization:** AdamW ($\text{lr} = 3 \times 10^{-4}$, weight decay $0.05$), cosine annealing schedule with 2 warmup epochs, 50–60 epochs, gradient clipping at 1.0, mixed precision (fp16/bf16).
- **Early Stopping:** Tracks validation within-lambda SROCC with patience of 5 epochs.

---

## 10. Evaluation Protocol & Statistical Rigor (Stage 5)

### 10.1 Headline Metrics
To avoid rate-induced correlation inflation (where predicting $\lambda$ alone achieves overall SROCC $> 0.90$), evaluation prioritizes degradation ranking at fixed operating points:
1. **Within-$\lambda$ SROCC (Headline):** Mean Spearman rank correlation computed independently within each lambda subset:
   $$\overline{\rho}_{\text{within}} = \frac{1}{K} \sum_{k=0}^{K-1} \text{Spearman}(\{s_i\}_{i \in \Lambda_k}, \{y_i\}_{i \in \Lambda_k})$$
2. **Per-$\lambda$ SROCC Breakdown:** Detailed SROCC reported for each individual $\lambda_k$, explicitly showing sample count $n_{\text{images}}$.
3. **RMSE in Label Units:** Root mean squared error between predicted $s$ and standardized ground truth $y$.
4. **Within-$\lambda$ Partial Spearman (Controlling for Bitrate):**
   $$\rho_{\text{partial}}(s, y \mid \text{bpp}) = \frac{\rho(s, y) - \rho(s, \text{bpp})\rho(y, \text{bpp})}{\sqrt{(1 - \rho(s, \text{bpp})^2)(1 - \rho(y, \text{bpp})^2)}}$$
5. **Pairwise Accuracy within Lambda (PWRC-style):** For all image pairs within the same $\lambda$ whose label difference $|y_i - y_j| > T$, report concordance accuracy vs threshold $T \in [0.0, 0.20]$, including area under the curve (DS-AUC).

> [!NOTE]
> **Demoted Appendix Metrics:** Overall SROCC, overall PLCC, overall partial SROCC, and mean within-image SROCC (trivially $\approx 1.0$ because degradation decreases monotonically with $\lambda$) are demoted to an appendix with explicit warnings about rate confounding.

### 10.2 Statistical Confidence & Multi-Seed Protocol
- **Clustered Bootstrap CIs:** All reported metrics carry 95% bootstrap confidence intervals (1,000 resamples) clustered by source image ID to prevent inter-sample dependency leakage.
- **Paired Hypothesis Testing:** Differences between models/baselines ($\Delta \rho = \rho_A - \rho_B$) are tested using paired image-clustered bootstrap with two-sided empirical p-values.
- **Multi-Seed Aggregation:** Minimum 3 seeds (target 5: seeds 42, 43, 44, 45, 46). Results are reported as mean $\pm$ standard deviation across seeds, alongside clustered bootstrap CIs on seed-averaged ensemble predictions.

### 10.3 Evaluation Experiments (E1–E8)
- **E1: In-Distribution (KonIQ Test Split):** Headline within-$\lambda$ SROCC, per-$\lambda$ breakdown, RMSE, partial Spearman controlling for bpp, PWRC curve.
- **E2: Cross-Content & Cross-Resolution (LiveC):** Native $500 \times 500$ images evaluated without fine-tuning, assessing transfer across different image statistics and aspect ratios.
- **E3: Independence from Original MOS:** Rerun with verified `orig_mos`. Per lambda report: $\text{Spearman}(s, \text{orig\_mos})$, $\text{Spearman}(y, \text{orig\_mos})$, $\text{Spearman}_{\text{partial}}(s, \text{orig\_mos} \mid y)$, and within-lambda MAE across low, mid, and high MOS tertiles.
- **E4: Teacher Agreement Matrix:** Comprehensive correlation matrix between candidate teachers (HaarPSI, DISTS, PSNR) and bitrate at fixed $\lambda$, quantifying content bias and structural alignment.
- **E5: Shortcut Baselines:** Comparison against:
  - (a) Per-lambda bpp regression (isotonic / linear)
  - (b) Global bpp regression
  - (c) Lambda index alone
  - (d) Global bpp + lambda index
  - (e) Latent summary statistics (per-channel mean, std, zero fraction, energy percentiles) fit via Ridge regression and GBDT.
- **E6: Pixel-Domain Reference:** Real custom pixel CNN student trained on decoded RGB reconstructions with identical targets and splits.
- **E7: Human Study Protocol:** Pre-registered 2AFC preference / boosted triplet protocol on ELIC reconstructions, citing Mohammadi et al. (IEEE Access 2026).
- **E8: Real Hardware Latency:** Benchmarked on GPU server (batch sizes 1 and 16, ms/image):
  - Latent student forward pass (synthesis decode bypassed: 0.0 ms)
  - ELIC synthesis decode pass ($\hat{y} \to \hat{x}$) + Pixel student forward pass.

### 10.4 Pre-Registered Teacher Decision Rule

To avoid post-hoc cherry-picking between full-reference teachers that embody distinct perceptual and structural priors, the project pre-registers the following explicit decision protocol:

1. **Designated Primary Teacher (v1):** `haarpsi` is pre-registered as the primary training target for student models in v1 due to its computational speed, phase-congruency theoretical foundation, and demonstrated sensitivity to high-frequency edge degradation across neural compression operating points.
2. **Pre-Registered Content-Rate Divergence Criterion:**
   - At fixed codec operating point $\lambda$, FR metrics exhibit differential content biases with respect to bitrate: HaarPSI penalizes high-frequency texture and edge loss (correlating negatively with bpp, $\approx -0.32$), while DISTS ($1 - \text{DISTS}$) rewards deep VGG perceptual feature richness (correlating positively with bpp, $\approx +0.53$).
   - If a student trained on HaarPSI achieves strong within-$\lambda$ correlation with HaarPSI ($\ge 0.70$) but low or negative correlation with $1 - \text{DISTS}$ within $\lambda$, this divergence is pre-registered as evidence of teacher divergence rather than student failure.
3. **Symmetric Architecture Benchmark (A5):**
   - A parallel student model must be trained on $1 - \text{DISTS}$ using the exact same architecture, seeds, hyperparameter protocol, and split partitioning.
   - Both models are reported side-by-side in `reports/eval_main.md` and `reports/ablations.md`.
4. **Human Subjective Arbitration (E7):**
   - For operating regimes where HaarPSI and DISTS disagree, the pre-registered 2AFC subjective human study protocol (boosted triplets on ELIC outputs) serves as the definitive arbiter.

---

## 11. Systematic Ablation Suite

| ID | Ablation Description | Investigated Variant | Baseline / Full Model | Objective / Hypothesis |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | Trunk Architecture | Pure CNN (ConvNeXt) vs Vision Transformer | ConvNeXt 4-stage | Compare inductive bias and efficiency |
| **A2** | Pairwise Ranking Loss | Regression only ($\alpha = 0.0$) | Composite ($\alpha = 0.5$) | Test impact of ranking constraint on within-$\lambda$ order |
| **A3** | Hyperprior Inputs | Config A ($\hat{y}$ only) | Config B ($\hat{y} + \hat{z}$) | Quantify information gain from side information |
| **A4** | Lambda Conditioning | Blind student (no $\lambda$) | Conditioned student | Test whether latents implicitly encode rate point |
| **A5** | Target Metric | Teacher = DISTS ($1 - \text{DISTS}$) | Teacher = HaarPSI | Evaluate sensitivity to teacher perceptual priors |
| **A7** | Spatial Masking | Unmasked global pooling | Masked mean + attention pooling | Quantify impact of zero-padding border artifacts |
| **A8** | Feature Pooling | Global average pooling | Channel-weighted attention pooling | Test benefit of learned spatial weighting |
| **A9** | Loss Function | Huber / L1 loss | Smooth L1 loss | Robustness to noisy proxy label outliers |
| **A10**| Feature Statistics | Mean pooling only | Mean + Standard deviation pooling | Test if latent variance encodes texture complexity |
| **D3** | Data Scaling | 25%, 50%, 100% of training set | 100% KonIQ train | Measure performance vs dataset size curve |
| **D4** | Offline Augmentation | Horizontal flips at generation | No augmentation | Test spatial augmentation without violating equivariance |
| **D5** | Factorial Degradation | Codec only vs Codec + Noise | Codec only | Separation of compression from prior degradation |

---

## 12. Configuration and Reproducibility

- Single source of truth for all stages via YAML configurations in `configs/`.
- Every training and evaluation run outputs `run_meta.json`, `metrics.json`, and `predictions.parquet` in `reports/runs/{run_name}/`.
- Random seeds fixed across PyTorch, NumPy, and Python standard random libraries.
- Bit-exact latent generation verified through MD5/SHA256 checksums against direct bitstream decompression.

---

## 13. Milestones, Checklists & Acceptance Gates

### Phase 1: Foundation (Completed, pending M7 re-verification)
- [x] **M0: Repository Scaffolding & Offline Environment Split.**
- [x] **M1: ELIC Codec Wrapper & Bit-Exact Sanity Gate.** (Passed: 20/20 bit-exact).
- [x] **M2: Pilot Generation (100 KonIQ images).** (Passed: diagnostics written).
- [x] **M3: Full Dataset Generation & Proxy Labelling.** (Passed: KonIQ 645 train, 78 val, 77 test; LiveC 200 images; 6 lambdas).
- [x] **M4: Baseline Students & Training Pipeline.** (Passed: Config A blind/conditioned).
- [x] **M5: Multi-Modal Student (Config B) & Initial Evaluation.** *(Note: M5/M6 report numbers are UNVERIFIED pending M7 fixes).*
- [x] **M6: Pixel References & Protocol Drafting.** *(Note: numbers UNVERIFIED pending M7).*

---

### Phase 2: Integrity & Rigorous Statistical Framework (Current)

#### Milestone M7: Integrity Fixes & Provenance Standardization
- [ ] **A1. Remove Hardcoded Results:** `eval_pixel_refs.py` reads strictly from run artifacts or writes `"NOT RUN"`; unit test `test_no_hardcoded_reports.py` passes.
- [ ] **A2. Metric Registry & Target Handling:** `METRIC_REGISTRY` supports directionality and transforms; `run_meta.json` logs target metadata; unit test `test_metric_registry.py` passes.
- [ ] **A3. Fix `orig_mos` & Manifest Audit:** `generate.py` lookup fixed; `scripts/backfill_mos.py` updates manifest; `scripts/audit_manifest.py` verifies zero-null, non-zero variance, complete lambdas, and valid splits.
- [ ] **A4. Strict Split Separation:** All baselines and reports evaluate `koniq_val`, `koniq_test`, and `livec` separately; no silent fallbacks.
- [ ] **A5. Single Source of Truth for Lambdas:** `configs/lambdas.yaml` established; stale 0.0016 references purged; unit test `test_lambdas_single_source.py` passes.
- [ ] **A6. Stale Documentation Cleanup:** Purged misleading claims from documentation and archived unverified reports.

> [!IMPORTANT]
> **GATE M7:**
> 1. User runs `python scripts/backfill_mos.py --manifest data/manifest.parquet` on server.
> 2. User runs `python scripts/audit_manifest.py --manifest data/manifest.parquet` and it exits with code 0.
> 3. User re-runs baseline evaluation to confirm non-zero `orig_mos` correlations in E3.

---

#### Milestone M8: Evaluation Statistics & Teacher Analysis
- [ ] **B1. Headline Metrics Implementation:** Within-$\lambda$ SROCC, RMSE, within-$\lambda$ partial Spearman controlling for bpp, PWRC pairwise accuracy curve and DS-AUC.
- [ ] **B2. Statistical Uncertainty:** 95% image-clustered bootstrap CIs for all metrics; paired bootstrap hypothesis tests for model differences.
- [ ] **B3. Multi-Seed Training Pipeline:** `train.py --seed N`; `scripts/aggregate_seeds.py` computing mean $\pm$ std and ensemble bootstrap CIs across $\ge 3$ seeds.
- [ ] **B4. Verified E3 Rerun:** Accurate Spearman and partial correlations with `orig_mos` and MAE across MOS tertiles.
- [ ] **B5. Teacher Agreement Matrix (E4):** Cross-correlation matrix between HaarPSI, DISTS, PSNR, and bpp across operating points.
- [ ] **B6. Standardized Predictions Schema:** Every evaluation run writes `predictions.parquet` (`image_id`, `dataset`, `split`, `lambda_idx`, `y`, `s`, `bpp`, `orig_mos`).
- [ ] **B7. Pre-Registered Teacher Decision Rule:** Transparent criterion for selecting primary teacher without post-hoc cherry-picking.

> [!IMPORTANT]
> **GATE M8:**
> 1. Verified `reports/eval_main.md` generated exclusively from run artifacts.
> 2. All headline metrics include 95% clustered bootstrap CIs and multi-seed statistics.
> 3. E3 shows verified non-zero `orig_mos` statistics.

---

#### Milestone M9: Stronger Baselines & Content-Rate Disentanglement
- [ ] **C1. Per-Lambda Bitrate Baselines:** Linear and isotonic regression fit strictly per lambda.
- [ ] **C2. Global Rate-Lambda Baselines:** Global $\text{bpp} + \lambda$ baselines to benchmark rate shortcuts.
- [ ] **C3. Machine Learning Latent Baselines:** Ridge regression and GBDT (LightGBM/XGBoost) on pooled latent summary statistics.
- [ ] **C4. Residual Analysis:** Scatter plots and residual analysis of student predictions vs bitrate.
- [ ] **C5. Comparative Baseline Report:** Comprehensive table verifying whether latent students provide orthogonal quality information beyond rate.

> [!IMPORTANT]
> **GATE M9:** Baseline report demonstrates whether student outperforms per-lambda bitrate and latent GBDT with statistical significance ($p < 0.05$).

---

#### Milestone M10: Architectural Improvements & Systematic Ablations
- [ ] **D1. Ablation Implementations:**
  - A2: Composite loss vs Pure ranking ($\alpha = 0.0, 0.5, 1.0$).
  - A3: Config A ($\hat{y}$) vs Config B ($\hat{y} + \hat{z}$).
  - A4: Blind vs Conditioned mode.
  - A5: DISTS teacher training.
  - A7: Masked vs Unmasked pooling.
  - A8: Channel-weighted attention pooling.
  - A9: Huber / L1 loss.
  - A10: Mean + Std pooling.
- [ ] **D2. Multi-Seed Ablation Runs:** Run ablations across $\ge 3$ seeds with image-clustered bootstrap testing.
- [ ] **D3. Data-Scaling Ablation:** Train on 25%, 50%, 100% subsets to characterize sample efficiency.
- [ ] **D4. Offline Flip Augmentation:** Evaluate spatial flipping during generation.
- [ ] **D5. Factorial Degradation Analysis:** Benchmark student robustness on pre-degraded inputs.

> [!IMPORTANT]
> **GATE M10:** `reports/ablations.md` generated from real multi-seed run artifacts with statistical difference tests against baseline.

---

#### Milestone M11: Pixel Reference & Verified Latency Benchmarking
- [ ] **E1. Train Pixel Reference Student:** Train custom pixel CNN student on decoded RGB images using identical targets, seeds, and splits.
- [ ] **E2. Pretrained NR Baselines:** Evaluate pyiqa pretrained models (CLIPIQA, MUSIQ, NIQE) on reconstructions.
- [ ] **E3. True GPU Latency Benchmarking:** Measure real hardware latency on execution server (BS=1 and BS=16) for latent student vs synthesis decode + pixel student.

> [!IMPORTANT]
> **GATE M11:** `reports/eval_refs.md` generated with verified GPU latencies and pixel reference comparisons.

---

#### Milestone M12: Synthesis & Deliverables
- [ ] **F1. Paper-Ready Figures:** Automated generation of publication-quality vector plots (within-$\lambda$ performance, PWRC curves, teacher correlations, latency trade-offs).
- [ ] **F2. Comprehensive Findings Report:** Final synthesis summarizing research questions, validated results, limitations, and future work.
- [ ] **F3. Reproducibility Package:** Verification of all config hashes, manifest hashes, and clean automated evaluation pipeline.

> [!IMPORTANT]
> **GATE M12:** Complete project signoff with zero unverified numbers and fully automated reproduction workflow.

---

## 14. Testing Architecture

### 14.1 Offline Unit Tests (`tests/unit/`)
Pure Python 3.13 standard library tests executed on the development machine:
- `test_manifest.py`: Schema validation, null checks, duplicate checks.
- `test_splits.py`: Source image disjointness, stratification.
- `test_config.py`: YAML parsing, defaults, schema validation.
- `test_geometry.py`: Padding arithmetic, valid mask generation.
- `test_no_hardcoded_reports.py`: Static guard preventing hardcoded numeric tables in report generators.
- `test_lambdas_single_source.py`: Verifies single source of truth for lambdas and absence of stale Cheng lambdas.
- `test_metric_registry.py`: Validates `METRIC_REGISTRY`, directionality, transforms, and `run_meta.json`.
- `test_eval_refs.py`: Validates report formatting and `"NOT RUN"` behavior on missing inputs.

### 14.2 Integration & GPU Server Tests (`tests/server/`)
Executed on the remote execution server with PyTorch, CUDA, and codec checkpoints:
- Latent bit-exactness check against bitstream decompression.
- Metric registry integration with PyTorch tensors.
- Latent student forward and backward passes with valid masking.
- Multi-seed training loop and checkpoint saving.
- Hardware latency measurement with CUDA synchronization.

---

## 15. Key Risks, Threats to Validity & Mitigations

1. **Teacher Disagreement within Lambda:**
   - *Risk:* At fixed rate, HaarPSI and DISTS encode opposing content biases (HaarPSI correlates $-0.32$ with bpp; DISTS correlates $+0.53$). A student trained on one may disagree with the other.
   - *Mitigation:* Explicitly report cross-teacher agreement matrix (E4), train students on both teachers (A5), and benchmark against human perceptual data (E7).
2. **Rate Shortcut Confounding:**
   - *Risk:* Latents inherently encode rate, allowing models to achieve high global correlation simply by predicting bitrate.
   - *Mitigation:* Prioritize within-$\lambda$ SROCC and partial correlation controlling for bpp as headline metrics; compare against per-$\lambda$ bpp and latent GBDT baselines.
3. **Small Sample Scale:**
   - *Risk:* 78 validation and 77 test images yield wide standard errors (~0.04), making small metric differences statistically insignificant.
   - *Mitigation:* 95% image-clustered bootstrap CIs, paired hypothesis tests, and multi-seed averaging ($\ge 3$ seeds).
4. **Proxy Label vs Human Judgment Gap:**
   - *Risk:* Neither HaarPSI nor DISTS perfectly reflects human perception across all distortion types (e.g. above 1 JND).
   - *Mitigation:* Pre-register human study protocol (E7) and avoid making unqualified claims about human quality.
5. **Single Codec Latent Space:**
   - *Risk:* Model parameters are tied to ELIC latent representations and do not directly transfer to other architectures.
   - *Mitigation:* Clearly state claim scope in all documentation and publications.

---

## 16. Instructions to Implementing Agent

- Maintain strict separation of dev and execution environments. Never attempt to install `torch` or run GPU jobs locally.
- Implement strictly in phase order (Phase A through Phase F) and halt at each milestone gate.
- Commit all code and configuration changes to `origin/main`.
- Update `HANDOVER.md` with explicit, copy-pasteable server commands for every gate.
- If any requirement or data format is ambiguous, ask the user before proceeding.
