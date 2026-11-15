# Design Doc: No-Reference Compression Degradation Predictor from Codec Latents via Proxy Labels

Written for implementation by a coding agent. Status: v2 draft (latent-domain student), 15 Nov 2026.

> [!IMPORTANT]
> **Execution Environment Notice for Implementing Agent:** The development machine where the coding agent operates is **not** the execution environment. The development machine lacks GPU, has limited storage, and **cannot have Python packages like `torch` or `compressai` installed**. The agent's role is to write clean, modular code, configs, runner scripts, and offline/mock tests. The user pulls the code to a remote execution server equipped with GPU and storage to run workloads and integration tests (see Sections 5, 13, 14, and 16).

## 1. Goal

Train a no-reference (NR) model that predicts compression degradation from the learned codec's own latents, without decoding to pixels. The input is the quantized latent of an ELIC-style codec (about 320 channels at 1/16 of the image resolution, so a 256x256 image gives a 16x16x320 tensor), optionally with the hyperprior latent. The output is how much the compression degraded the image relative to the unseen original, as measured by a full-reference (FR) metric.

Labels are generated, not collected: encode KonIQ-10k with ELIC at every lambda, decode each result to pixels once, and score the reconstruction against its original with HaarPSI. Pixels are used only to compute labels and for a pixel-domain reference baseline. The student never sees them.

Underlying research question: how much does compression affect image quality, independent of the original image's own quality? Closest precedents (verify before citing): Torfason et al. (ICLR 2018) run classification and segmentation directly on learned compressed representations without decoding, and Huang and Wu (2024) use a codec's latent as a perceptual-distance proxy, but as a full-reference measure.

**Claim scope.** In v1 the model is specific to ELIC latents and predicts HaarPSI-measured degradation. Claims about human-perceived quality need new human data on ELIC outputs (see E7). Claims about other codecs need separate training.

## 2. Non-goals for v1

- No pixel input to the student. A pixel-domain model is trained only as a reference baseline.
- No multi-codec or codec-agnostic model in v1. The student is tied to ELIC's latent space.
- No ensemble of FR teachers. DISTS is computed for evaluation only.
- No new human study in v1 (E7 specifies one as the follow-up).
- No video.

## 3. Fixed decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Codec | ELIC reimplementation, one model per lambda, all lambdas used | Project plan |
| Student input | Quantized latent y\_hat only (config A); optionally plus hyperprior latent z\_hat (config B) | Core goal: NR prediction from latents without decoding |
| Latent source | The codec's real quantized latent, verified to decode to the bitstream's output | Latents must be what a decoder would receive |
| Training images | KonIQ-10k at native resolution, never resized | Resizing destroys codec artifacts |
| Teacher (labels) | HaarPSI(original, decoded reconstruction), RGB in \[0,1\] | Cheap, well validated, single teacher keeps things simple |
| Eval-only metric | DISTS (use 1 - DISTS) | Checks for teacher-specific bias |
| Student target | HaarPSI, range \[0,1\], higher = less degradation | Direct proxy label |
| Lambda handling | Two modes: blind (no lambda information) and conditioned (lambda embedding) | Each lambda model has its own latent space; see 9.2 |
| Augmentation | None on latents | Flipping or cropping a latent grid is not the same as encoding a flipped or cropped image |
| Cross-dataset test | LiveC compressed with the same ELIC models | Cross-content and cross-resolution test |
| Splits | By source image, fixed seed | Prevents content leakage |

## 4. Repository layout

```
nrcomp/
  configs/            # one YAML per stage
  third_party/
    ELiC-ReImplemetation/   # git submodule pinned to a fixed commit (see 6.1)
    ELIC_PIN.md             # pinned commit SHA and date
  weights/            # ELIC checkpoints, one per lambda, plus CHECKSUMS.txt
  data/               # raw datasets (user-provided), manifests, latents, optional recon images
  nrcomp/
    codec/            # wrapper around the submodule: padding, bitstream I/O, latent extraction
    data/             # dataset loaders, split logic, manifest I/O, latent store
    labels/           # metric wrappers (haarpsi, dists, psnr)
    model/            # latent student, pixel reference student, losses
    train.py
    evaluate.py
    generate.py       # encode, extract latents, decode, manifest
    label.py          # compute FR metrics into manifest
  scripts/            # label_diagnostics.py, latent_diagnostics.py, teacher_check.py, codec_sanity.py
  tests/
  reports/            # generated markdown/CSV outputs
  requirements-codec.txt
  requirements-main.txt
```

## 5. Environments and Execution Workflow

### 5.1 Separation of Development and Execution Environments
The development machine (where the coding agent writes code) is strictly decoupled from the execution server (where code is run):

- **Development Machine (Coding Agent):**
  - **Constraints:** Limited storage, no GPU, and **cannot install PyTorch (`torch`), torchvision, CompressAI, or other heavy ML dependencies**.
  - **Role:** Code authoring, repository scaffolding, configuration, pipeline orchestration scripts, documentation, and writing tests.
  - **Local Verification:** Use static analysis, syntax verification, schema validation, and lightweight unit tests that use mocks/stubs or standard library where possible. **The agent must never attempt to `pip install torch` or run model training/inference locally.**
- **Execution Server (Target Environment):**
  - **Hardware:** Dedicated GPU (~24 GB VRAM), large storage capacity (hundreds of GB for datasets, checkpoints, and latents).
  - **Role:** Pulls the code via git, runs dataset downloads, ELIC latent extraction, metric labelling, training, evaluation, and GPU-dependent integration tests.
- **Handoff Protocol:**
  - The agent writes the code, config files, and tests on the dev machine and commits them to git.
  - The agent provides explicit, copy-pasteable CLI commands or runner scripts for the user to execute on the execution server.
  - The user pulls the code on the execution server, runs the commands, and reports back outputs, logs, or metrics when needed to clear milestone gates.

### 5.2 Server Python Environments
The execution server uses two isolated Python virtualenvs that communicate only through disk files (latents, reconstructions, manifest):

- **Codec environment (`requirements-codec.txt`):** The ELIC reimplementation lists old dependencies in its README (`compressai==1.1.5`, plus `thop`, `ptflops`, `timm`), so install them in their own virtualenv with whichever PyTorch version that CompressAI release supports. It runs generation (Stage 2) only.
- **Main environment (`requirements-main.txt`):** Python 3.10+, PyTorch 2.1+, timm, pyiqa (HaarPSI, DISTS, baseline NR models; piq is an acceptable alternative), pandas + pyarrow, scipy, scikit-learn, PyYAML, tqdm. It runs labelling (Stage 3), training (Stage 4), and evaluation (Stage 5).

Pin all versions in `requirements-codec.txt` and `requirements-main.txt`. One GPU with about 24 GB memory is assumed on the execution server. Record package versions and git hash in every output directory.

## 6. Stage 1: ELIC codec wrapper

**6.1 Codec source and checkpoints.** Use the community ELIC reimplementation (not official): https://github.com/VincentChandelier/ELiC-ReImplemetation. It is built on CompressAI, and its README says the models were trained on the 8,000 largest ImageNet images picked as in the ELIC paper (its training command uses N=192, M=320, 256x256 patches).

- **Pin it as a git submodule at a fixed commit.** A submodule records an exact commit of another repository inside your repository. Run `git submodule add https://github.com/VincentChandelier/ELiC-ReImplemetation third_party/ELiC-ReImplemetation`, then `git -C third_party/ELiC-ReImplemetation checkout <SHA>`, then commit the parent repo so it stores that SHA. Write the SHA and date into `third_party/ELIC_PIN.md`. The researcher chooses the SHA (the latest `main` commit at clone time is fine). The agent must not update the submodule afterwards and must not edit files inside it: wrappers and any patches live in `nrcomp/codec/`.
- **Weights.** The README lists one Google Drive link per lambda. Provide a download script (`scripts/download_weights.py` or `.sh`) for the execution server to fetch each checkpoint into `weights/` (for example with `gdown`; if Drive blocks automated downloads, the user downloads them manually). Record a SHA256 for each file in `weights/CHECKSUMS.txt` and verify it on every run. Mirror the files to your own storage, since Drive links can rate-limit or disappear. The download and storage verification run on the execution server, not the local dev machine.
- **Inference-ready checkpoints.** The README documents `updata.py` (entropy model update) for preparing a checkpoint before real entropy coding. Read the pinned code to confirm the exact command, run it per checkpoint, and keep both the raw and updated files.
- **Reuse, don't reimplement.** Import the model class from `Network.py`. `Inference.py` shows the compress and decompress flow and uses `--patch 64`, which matches the padding rule in 6.3.
- **Stop conditions.** If any checkpoint is missing or fails its checksum, stop and report. Do not train codecs unless the researcher asks (optional milestone M1b, using the repo's `train.py`).

**6.2 Lambdas.** The README lists six checkpoints: lambda = 0.45, 0.15, 0.032, 0.016, 0.008, 0.004. The link text for the fourth entry reads 0.0016, so confirm from the checkpoint or its config whether it is 0.016 or 0.0016, and write the verified values into the config. Treat the numbers as names only. Define `lambda_idx = 0..K-1` (K = 6) in order of increasing mean `bpp_actual` as measured in the Kodak run (6.4), not by the numeric value.

**6.3 Interface.**

```python
codec.compress(img_uint8_hwc, lambda_idx)        -> (bytes, meta)
codec.decompress(bytes, meta)                    -> img_uint8_hwc   # same H, W as input
codec.extract_latents(img_uint8_hwc, lambda_idx) -> dict(
    y_hat,       # float32 (320, Hp/16, Wp/16): quantized latent as the decoder sees it
    z_hat,       # (192, Hp/64, Wp/64): quantized hyper-latent
    valid_mask,  # bool (Hp/16, Wp/16): True where the cell center lies inside the original image
    meta)        # original size, padded size, lambda_idx
codec.decode_from_latents(y_hat, meta)           -> img_uint8_hwc   # must equal decompress(bytes)
```

Channel counts (320 for `y_hat` from M=320, 192 for `z_hat` from N=192) match the reimplementation's training command. Confirm them against every lambda checkpoint.

- Pad input to a multiple of 64 (reflect padding), then crop back to the original size after decoding. Output shape must equal input shape.
- Run in eval mode, fp32, and compress/decompress on the same device.
- Measure bpp from the real bitstream: `bpp_actual = 8 * len(bytes) / (H * W)`. Also record `bpp_est` from the likelihoods.
- Verify that the decoded image matches the encoder-side reconstruction on 20 images (report max abs difference).

**6.4 Sanity gate (blocking).** On the 24-image Kodak set (user-provided), for all lambdas, write `reports/codec_sanity.md` with bpp and PSNR per lambda. Fail the gate if bpp or PSNR is non-monotone in lambda, if decoding fails, or if any of these latent checks fails on 20 images: (a) `y_hat` from `extract_latents` is bit-identical to the latent obtained by entropy-decoding the bitstream; (b) `decode_from_latents` equals `decompress(bytes)` exactly; (c) the latent has the expected channel count for every lambda. Compare to published ELIC numbers and describe any gap, since a reimplementation on an ImageNet subset will be somewhat worse.

**6.5 Latent extraction notes.**

- `y_hat` is the quantized latent after mean restoration (`y_hat = round(y - mu) + mu`), because that is what the synthesis transform receives. If the reimplementation exposes only the integer symbols, also expose `mu` and rebuild `y_hat`.
- The latent grid is computed on the padded image (multiples of 64). `valid_mask` marks cells whose center lies inside the original image. All pooling and losses in the student use it.
- Each lambda has its own checkpoint, so latent spaces are not guaranteed to be aligned across lambdas. `scripts/latent_diagnostics.py` reports per-lambda channel statistics, the fraction of zero symbols, and how accurately a simple classifier predicts lambda from pooled latent statistics (see 9.2).
- Hyperprior option in v1: `z_hat` only. Entropy parameters (means, scales) and a per-cell rate map are a later ablation (A6). They depend on the codec's channel-wise context, so compute them with the frozen codec on the fly and do not store them.

## 7. Stage 2: Data generation

**7.1 Inputs.** KonIQ-10k (about 10k images at 1024x768, plus MOS CSV) and LiveC (about 1.1k images at 500x500, plus MOS). Both are user-provided; verify sizes and licenses on arrival. Originals may already be JPEG. Decode each once to RGB uint8 and treat those decoded pixels as the reference, because that is what the codec saw.

**7.2 Splits.** KonIQ: 80/10/10 train/val/test by source image, stratified by MOS quintile, seed 0, saved to `data/splits/koniq_split.csv`. LiveC: test only.

**7.3 Generation.** For every image and every lambda: run `extract_latents`, produce the bitstream (for `bpp_actual`), decode the reconstruction with `decode_from_latents`, and compute labels (Stage 3) against the original. Save `y_hat` (and `z_hat` if enabled) to the latent store. Reconstructions are intermediate: keep them as lossless PNG only for the pilot, the test splits, and the pixel reference baseline (E6). Otherwise delete them after labelling.

**7.4 Storage.** `y_hat` is about 320 x 64 x 48 = 0.98 M values per 1024x768 image, roughly 2 MB in float16 (`z_hat` adds about 0.07 MB). For about 10k images x K lambdas that is on the order of 100 to 150 GB. Config option `latent_store`: `fp16` (default, optionally zstd-compressed) or `bitstream` (smallest, but entropy decoding at load time is slow). This storage requirement applies to the execution server (do not store latents on the local dev machine). The generation script must check free disk space on the server before running and abort if free disk is below 1.1x the estimate.

**7.5 Manifest** (one parquet file, the single source of truth):

| Column | Meaning |
| --- | --- |
| dataset, image\_id, split | `koniq` or `livec`; split name |
| lambda\_idx, lambda\_value | Codec operating point |
| orig\_path | Original image |
| latent\_path | Stored `y_hat` (and `z_hat` if enabled) |
| recon\_path | Decoded reconstruction, nullable |
| height, width, padded\_height, padded\_width | Original and padded sizes |
| bpp\_actual, bpp\_est | Bitrate measures |
| psnr, haarpsi, dists | Metrics (filled by Stage 3) |
| orig\_mos | Original image MOS from the dataset |

**7.6 Operations.** Idempotent (skip existing outputs), shardable by index for parallel runs, and failures logged to `reports/failures.csv` without stopping the run.

## 8. Stage 3: Labels

**8.1 Metrics.** For each manifest row, compute on the full-resolution images (float32, RGB, range \[0,1\], NCHW): `haarpsi(orig, recon)`, `dists(orig, recon)`, and `psnr`. Confirm pyiqa's `lower_better` flags: HaarPSI is higher-better, DISTS is lower-better. Store raw DISTS and convert with 1 - DISTS only at analysis time.

**8.2 Target.** `y = haarpsi`, raw, no transform in v1.

**8.3 Diagnostics** (`scripts/label_diagnostics.py`, output `reports/label_diagnostics.md`):

- Histogram of HaarPSI per lambda, with a warning if a large share of labels sit near the ceiling.
- Share of images where HaarPSI is not monotone in lambda. Flag if above 10%.
- Spearman between HaarPSI and (1 - DISTS), overall and averaged per image.
- Spearman between bpp\_actual and HaarPSI, overall (context for the shortcut baseline in E5).

## 9. Stage 4: Latent-domain student

**9.1 Architecture.** Train from scratch, because ImageNet backbones do not apply to latents. Stem: 1x1 conv from the input channels to 256, then GroupNorm. Trunk: 6 residual blocks (3x3 conv, GELU, 3x3 conv) at width 256 on the latent grid, no downsampling. Head: masked mean pooling and masked attention pooling over valid cells, concatenated, then `Linear(512,256)`, GELU, `Linear(256,1)`, sigmoid. Output `s` in (0,1). Expect a few million parameters. A transformer trunk over latent cells is ablation A1. Config A inputs `y_hat` only. Config B adds `z_hat` through its own small conv stem, upsampled 4x (nearest) to the `y_hat` grid and concatenated after the stems. Normalize inputs per channel with mean and standard deviation computed on the training set.

**9.2 Inputs and lambda handling.** The `forward` signature takes the latent tensor(s) and `valid_mask` only: no pixels, bpp, file size, or image dimensions. Because each lambda model has its own latent space, run two modes (config `lambda_mode`). `blind`: one shared stem, normalization statistics pooled over all lambdas, no lambda information. This tests whether latents alone carry the quality signal. `conditioned`: adds a learned lambda embedding and optionally per-lambda normalization statistics. This is the practical setting, since whoever holds the bitstream knows which codec model produced it. Report both. If `latent_diagnostics.py` shows lambda is trivially predictable from pooled latent statistics, state that in the report, because blind mode is then not truly blind.

**9.3 Loss.**

```
L = SmoothL1(s, y) + alpha * L_rank        # alpha = 0.5
L_rank = mean over pairs (i, j) of the same source image with
         lambda_i < lambda_j AND y_j > y_i of relu(m - (s_j - s_i)),  m = 0
```

Pairs are used only where the teacher agrees with the lambda ordering, to avoid forcing wrong orderings.

**9.4 Batching.** Each step samples B source images and k random lambdas per image (default B=8, k=3; latents are small, so larger batches fit) so that same-image pairs exist in each batch. Grids of different sizes are zero-padded within a batch and covered by `valid_mask`. KonIQ images share one size, so this matters for LiveC and other sets.

**9.5 Augmentation.** None on latents. Flipping, rotating or cropping a latent grid is not equivalent to encoding the transformed image, because the codec is not equivariant to those transforms. If augmentation is wanted, encode flipped originals at generation time as extra samples (off in v1; doubles storage).

**9.6 Optimization.** AdamW, lr 3e-4, weight decay 0.05, cosine schedule with 2 warmup epochs, 60 epochs (small model trained from scratch), bf16 autocast, gradient clipping at 1.0, dropout 0.1 in the head. Select the checkpoint by validation within-lambda SROCC, averaged over lambdas. Seed everything.

## 10. Stage 5: Evaluation

All metrics report SROCC, PLCC, and RMSE, with 95% bootstrap confidence intervals (1000 resamples over source images). For PLCC, fit a 4-parameter logistic mapping on the validation set only and apply it to test.

- **E1, in-distribution.** KonIQ test split vs HaarPSI labels: overall, within-lambda (the headline number), and mean within-image SROCC across lambdas. Within-lambda matters most: at a fixed lambda the model must tell which images degrade more.
- **E2, cross-content.** Encode LiveC with ELIC at all lambdas, label with HaarPSI, and evaluate the same way. Also report any effect of the different image size (500x500 vs 1024x768).
- **E3, independence from original quality.** Using `orig_mos`, per lambda report: Spearman(s, orig\_mos); partial Spearman(s, orig\_mos | y); and Spearman(y, orig\_mos) for the teacher itself. Also report mean absolute error of s vs y within MOS tertiles. Run on KonIQ test and LiveC.
- **E4, second teacher.** Spearman between s and (1 - DISTS), overall and per lambda.
- **E5, shortcut baselines the student must beat.** Fit on train, evaluate on test: (a) `bpp_actual` alone to y (isotonic regression or gradient boosting); (b) `lambda_idx` alone; (c) a latent-statistics baseline: gradient boosting on pooled summary statistics of `y_hat` (per-channel mean absolute value, fraction of zeros, channel energy percentiles) plus `bpp_actual`. Headline results: partial Spearman between s and y controlling for `bpp_actual`, and within-lambda SROCC against baseline (c). Latents inherently carry rate information, so the student must show it explains label variance beyond bitrate.
- **E6, pixel-domain references.** (a) A pixel-domain NR student trained on the decoded reconstructions with the same labels and splits (for example `convnext_tiny` on full images), as the reference for what latents give up or gain. (b) Pretrained pyiqa NR models (for example `musiq`, `maniqa`, `clipiqa`, `niqe`) applied to the reconstructions: report Spearman with y and with `orig_mos`.
- **E7, human validation (needs new data).** Existing human-labelled sets (MCL-JCI, KonJND-1k, JPEG AI SDR25, CIQA) use other codecs, so they cannot validate an ELIC-latent student. Plan: a small human study on ELIC reconstructions (pairwise or JND, JPEG AIC-3 style) over a few hundred stimuli across lambdas and source images, compared against the student and HaarPSI. Until it exists, report results only as agreement with HaarPSI and DISTS. `scripts/teacher_check.py` can still use the JPEG sets to compare HaarPSI and DISTS as teachers (Spearman with quality setting, JND-consistency spread). If DISTS is clearly better, switch with `teacher: dists`.
- **E8, efficiency.** Time per image for the student on stored latents vs decode plus the pixel NR student, on the same GPU. Report ms per image and parameter counts.

Never tune on E2, E7, or any test split.

## 11. Ablations (after the baseline works)

- A1: trunk (conv vs transformer; width and depth).
- A2: rank loss off (alpha = 0).
- A3: hyperprior, config A (`y_hat`) vs config B (`y_hat` + `z_hat`).
- A4: lambda mode, blind vs conditioned (also a main result).
- A5: teacher = DISTS.
- A6: entropy parameters and a per-cell rate map as extra inputs, computed on the fly with the frozen codec.
- A7: border and padding effects: evaluate on valid cells only vs all cells, and on interior cells only.

## 12. Configuration and reproducibility

One YAML config per stage, run as `python -m nrcomp.<stage> --config configs/<name>.yaml`. All randomness seeded. Each run saves a copy of its config, git hash, and package versions. No stage may read data that is not listed in the manifest. Experiment tracking is optional (CSV logging is enough).

## 13. Milestones and acceptance criteria

| Milestone | Deliverable | Acceptance | Run Location |
| --- | --- | --- | --- |
| M0 | Dependency definitions (`requirements-*.txt`), repo skeleton, loaders, offline unit tests | Offline unit tests pass on dev machine; server environments install successfully | Dev & Server |
| M1 | ELIC submodule pinned (SHA in `ELIC_PIN.md`), weights download script & checksums, wrapper, latent extraction, sanity gate | `codec_sanity.md` passes on server, including latent bit-exactness checks (6.4) | Execution Server |
| M2 | Pilot: 100 KonIQ images, all lambdas: latents, labels, diagnostics | Valid manifest, `label_diagnostics.md` and `latent_diagnostics.md` written | Execution Server |
| M3 | Full KonIQ and LiveC generation and labelling | Manifest complete, failure rate reported | Execution Server |
| M4 | E5 baselines; latent student config A trained in blind and conditioned modes | End-to-end run, validation within-lambda SROCC reported | Execution Server |
| M5 | Config B; E1 to E4 and E8 report tables | `reports/eval_main.md` | Execution Server |
| M6 | Pixel references (E6), ablations, E7 study protocol written | `reports/eval_refs.md`, ablation table, protocol draft | Execution Server |

**Workflow at milestone gates:** The agent develops the code, tests, and configuration on the dev machine, commits changes, and provides the user with exact CLI commands to pull and execute on the execution server. If an acceptance check fails on the execution server, the user reports the error/logs back to the agent to debug and fix. If a milestone's acceptance fails, stop and report instead of continuing.

## 14. Tests

Organize tests into two distinct tiers to preserve local testability without local GPU/torch dependencies:

### 14.1 Offline / Local Unit Tests (Runnable on Development Machine)
Must run with standard Python / lightweight dependencies without `torch`, `torchvision`, `compressai`, or GPU:
- Manifest schema and I/O validation (pandas/pyarrow or pure python).
- Train, val, and test split disjointness by source image and stratification logic.
- Coordinate padding and crop arithmetic (calculating padded dimensions and valid masks algebraically or via numpy).
- Configuration parser and schema validation.
- Dataloader batching / pairing logic using synthetic mock tensors/arrays.
- Forward signature inspection and interface contract checks (using mock modules or python typing).

### 14.2 Integration & GPU Tests (Runnable on Execution Server)
Run inside the respective server virtual environment with GPU, PyTorch, and weights:
- Pad then crop returns the original shape and content.
- `y_hat` from `extract_latents` is bit-identical to the entropy-decoded latent, and `decode_from_latents` equals `decompress(bytes)`.
- Latent channel counts match the expected values for every lambda checkpoint.
- `valid_mask` is correct for a 500x500 input (padded to 512x512): shape and which border cells are marked invalid.
- `haarpsi(x, x)` is close to 1 and `dists(x, x)` is close to 0.
- HaarPSI decreases as Gaussian noise increases on a test image.
- Student forward pass on actual latent tensors with `valid_mask`.
- Dataloader integration with real latent store files (verifying no latent augmentation).

Tag server tests with pytest marks (e.g. `@pytest.mark.gpu` or `@pytest.mark.server`) or place them in a dedicated directory (e.g. `tests/server/` vs `tests/unit/`) so local test suites on the dev machine can run cleanly with `pytest -m "not gpu"`.

## 15. Risks and open decisions for the researcher

- **Lambda values:** the README's fourth value is ambiguous (0.016 or 0.0016). Verify from the checkpoint.
- **Reimplementation provenance:** the codec is an unofficial reimplementation. Its RD performance may differ from the ELIC paper, which the 6.4 gate checks and reports.
- **Weights hosting and license:** weights come from Google Drive links in the README, which can rate-limit or disappear. No license file was visible in the repository listing, so confirm the terms before publishing results. Mirror the weights to your own storage.
- **Dependency age:** the repo pins `compressai==1.1.5`, which is why the codec runs in its own environment.
- **ELIC latent details:** channel counts (M=320, N=192) match the README's training command. Verify the hyperprior layout against the code and every checkpoint.
- **Misaligned latent spaces:** each lambda model has its own latent space, so a shared student may struggle or may implicitly learn which model produced the latent. Blind and conditioned modes and the lambda-predictability diagnostic address this.
- **Rate shortcut:** latents carry bitrate information, so predicting degradation partly reduces to predicting bpp. E5 and the partial correlation controlling for bpp measure the added value.
- **Teacher validity:** HaarPSI has not been validated on ELIC's smoothing artifacts specifically.
- **Single codec:** the student is tied to ELIC's latent space by design. Other codecs need separate training.
- **No human labels on ELIC outputs:** E7 needs a new study. Until then, claims are limited to agreement with FR metrics.
- **Double compression:** KonIQ originals are likely already compressed. The reference is the decoded original, so labels measure added degradation, and the student must separate it from existing artifacts.
- **Ceiling effect:** at high lambdas HaarPSI may saturate; check the histograms.
- **Storage:** roughly 100 to 150 GB for fp16 latents; confirm disk or use the `bitstream` option.
- **Dataset access and licenses:** confirm for KonIQ-10k, LiveC, and any dataset used for the teacher check.
- **Regime coverage:** ELIC's operating points mostly produce visible compression. Near-lossless behavior is not covered by the labels.

## 16. Instructions to the implementing agent

- **Respect the development vs. execution split:**
  - Never attempt to install `torch`, `torchvision`, `compressai`, or CUDA packages on the local development machine.
  - Do not attempt to download full datasets (KonIQ-10k, LiveC) or run heavy extraction/training jobs locally. The dev machine has strict storage and compute limits.
  - Treat the local environment strictly as an authoring, linting, and offline unit-testing workspace.
- **Author for remote execution:**
  - Structure all code cleanly in `nrcomp/` with modular CLI entry points (`python -m nrcomp.<stage> --config configs/<name>.yaml`).
  - Provide self-contained shell scripts or Makefiles for common server tasks (e.g. setting up environments, downloading weights, running sanity gates).
  - Write clear logging and robust exception handling with informative console output.
- **Provide turn-key handoff instructions:**
  - For each milestone, provide the user with the exact, copy-pasteable terminal commands to run on the execution server (including which virtualenv to activate).
  - When waiting for a milestone gate, ask the user to pull the code on the execution server, execute the step, and share the generated reports or terminal output.
- **Implement stages incrementally:**
  - Implement the stages in order and stop at each milestone gate.
  - Ask rather than guess when a dataset format, checkpoint, or requirement is unclear.
  - Do not add features outside this document (extra codecs, ensembles, new augmentations) without approval.
  - Keep every generated number traceable to the manifest and a config.
