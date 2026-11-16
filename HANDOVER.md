# Project Handover & Milestone M4 Roadmap
**Project:** `nrcomp` — No-Reference Compression Degradation Predictor from Codec Latents via Proxy Labels  
**Date:** November 2026  
**Status:** Milestones M0, M1, M2, and M3 are **100% Complete & Verified**. Ready to start **Milestone M4**.

---

## 1. System Architecture & Environment Constraints

> [!IMPORTANT]
> **Strict Separation of Dev Machine vs. Execution Server:**
> - **Local Development Machine:** Linux, standard Python 3.13 only. **Never run `pip install torch` or heavy ML packages locally.** All unit tests in `tests/unit/` are pure Python (using mocks, lists, or standard library) and must run cleanly with `python -m unittest discover tests/unit` (currently 30/30 passing).
> - **Remote Execution Server:** Linux, CUDA 12.4, working directory `/storage/nrcomp`.
>   - Conda environment `nrcomp-codec`: Python 3.9, CompressAI 1.1.5, PyTorch with CUDA (runs Stage 1 sanity & Stage 2 data generation).
>   - Conda environment `nrcomp-main`: Python 3.10, PyTorch with CUDA, PyIQA, PIQ, timm, pandas, pyarrow, CompressAI 1.1.5 with GNU C++ toolchain (`gxx_linux-64`). Runs Stage 3 labelling, Stage 4 training, and Stage 5 evaluation.
> - **Output Streaming:** Always pass `--no-capture-output` to `conda run` to ensure unbuffered live terminal output.
> - **Submodule Constraint:** `third_party/ELiC-ReImplemetation` is pinned to commit `92a9ece1e6e1a188a12dfd7a58f9b51c554f9f2d`. Never modify files inside `third_party/` directly; all compatibility patches reside in `nrcomp/codec/wrapper.py`.

---

## 2. Completed Milestones (M0 – M3)

### Milestone M0: Scaffolding & Setup
- Repository scaffolding, configuration files (`configs/`), conda setup script (`scripts/setup_conda_envs.sh`), and dataset download scripts (`scripts/download_datasets.py`).

### Milestone M1: ELIC Integration & Kodak Sanity Gate (PASSED)
- Weights downloaded and verified with SHA-256 (`weights/CHECKSUMS.txt`).
- State dict updated via `scripts/download_weights.py` to `weights/updated/`.
- 24 Kodak test images evaluated across all 6 lambdas:
  - Bitrate strictly monotone: $0.0470 \to 0.0771 \to 0.1254 \to 0.1988 \to 0.5009 \to 0.8687$ bpp.
  - PSNR strictly monotone: $26.10 \to 27.49 \to 28.95 \to 30.52 \to 34.47 \to 37.50$ dB.
  - Latent bit-exactness: 20/20 bit-identical.
  - Decode invertibility: `max_abs_diff = 0.000` (20/20).
  - Channels verified: $M=320$ for $\hat{y}$, $N=192$ for $\hat{z}$.

### Milestone M2: Pilot Generation & Labelling Gate (PASSED)
- 100 KonIQ source images $\times$ 6 lambdas = 600 samples generated in `data/latents/koniq/` and labelled with HaarPSI, DISTS, and PSNR in `data/manifest_pilot.parquet`.
- Diagnostic gates (`reports/label_diagnostics.md` & `reports/latent_diagnostics.md`):
  - Non-monotone HaarPSI share: **0.0% (0/100 images)** vs $\le 10\%$ target (**PASS**).
  - Ceiling clustering ($> 0.98$): **2.0% (12/600)** (**PASS**).
  - Mean per-image Spearman(HaarPSI, $1 - \text{DISTS}$): **0.9983** vs $> 0.80$ target (**PASS**).
  - Lambda predictability from pooled latent statistics: **77.00%** vs 16.67% random chance (**PASS**).

### Milestone M3: Full Dataset Generation & Labelling Gate (PASSED)
- Tailored for the 15 GB disk budget (`configs/generate_subset.yaml`):
  - KonIQ-10k: 800 source images $\times 6 = 4,800$ samples (stratified 80/10/10 train/val/test across MOS quintiles).
  - LiveC: 200 source images $\times 6 = 1,200$ samples (cross-content test set).
  - Total: 1,000 images $\times 6$ lambdas = **6,000 samples**.
- Data generation (`nrcomp.generate`) completed with **0 failures**.
- Quality metric labelling (`nrcomp.label`) completed with on-the-fly GPU latent decoding.
- Complete manifest stored at `data/manifest.parquet` (and mirrored to `data/manifest.csv`).
- **Full Diagnostic Gate Results (`reports/label_diagnostics_full.md` & `reports/latent_diagnostics_full.md`):**
  - **Non-monotone HaarPSI share:** **0.0% (0/1000 images)** vs $\le 10.0\%$ target (**PASS**).
  - **Spearman(HaarPSI, 1 - DISTS) [Overall]:** **0.7235** vs $> 0.70$ target (**PASS**).
  - **Spearman(HaarPSI, 1 - DISTS) [Mean Per-Image]:** **0.9980** vs $> 0.80$ target (**PASS**).
  - **Lambda Predictor Accuracy:** **80.00%** vs 16.67% random chance baseline (**PASS**).

---

## 3. Milestone M4: Student Model & Training Pipeline

### Goal
Train a lightweight convolutional student directly on quantized ELIC latents $\hat{y}$ (and optionally $\hat{z}$) to predict HaarPSI compression degradation $s \in (0, 1)$.

### Architecture Specification (`nrcomp/model/`)
```
Latent Input y_hat (320 channels, shape B x 320 x H/16 x W/16)
         |
  [ ConvStem: Conv2d 1x1 -> 256, GroupNorm(32, 256) ]
         |
  [ Trunk: 6 ResBlocks at width 256 ]
  Each block: Conv2d(3x3) -> GELU -> Conv2d(3x3) + Residual
         |
  +------+------+
  |             |
  v             v
[ MaskedMeanPool ] [ MaskedAttentionPool ]
  |             |
  +------+------+
         |
    Concat (512)
         |
  [ MLP Head: Linear(512, 256) -> GELU -> Dropout(0.1) -> Linear(256, 1) -> Sigmoid ]
         |
    Scalar prediction s in (0, 1)
```

1. **`nrcomp/model/trunk.py`:**
   - `ConvStem(in_channels=320, out_channels=256)`: 1x1 conv followed by `GroupNorm(32, 256)`.
   - `ResBlock(channels=256)`: Two 3x3 convs with GELU and residual skip connection.
   - `Trunk(depth=6, width=256)`: Sequential stack of 6 `ResBlock`s.
2. **`nrcomp/model/pooling.py`:**
   - `MaskedMeanPool`: Uses `valid_mask` to average only over unpadded spatial cells:
     $$\text{pool}(F) = \frac{\sum_{h, w} F_{c, h, w} \cdot M_{h, w}}{\max(\sum_{h, w} M_{h, w}, 1)}$$
   - `MaskedAttentionPool`: Learned query spatial attention over valid cells. Masked positions get $-\infty$ before softmax.
   - Concatenates both poolings into a 512-dimensional vector.
3. **`nrcomp/model/student.py`:**
   - `LatentStudent`:
     - **Config A (`y_hat` only):** 320 channels input.
     - **Config B (`y_hat` + `z_hat`):** 192-channel stem for $\hat{z}$, $4\times$ nearest-neighbor upsample, concatenated with $\hat{y}$ stem features.
     - **Blind Mode:** No lambda conditioning.
     - **Conditioned Mode:** Learned lambda embedding $\mathbf{e}_\lambda \in \mathbb{R}^{256}$ added to trunk feature maps.
4. **`nrcomp/model/loss.py`:**
   - `CompositeLoss`:
     $$\mathcal{L} = \text{SmoothL1}(s, y) + \alpha \mathcal{L}_{\text{rank}}$$
     with $\alpha = 0.5$, where $\mathcal{L}_{\text{rank}}$ operates over all same-image pairs $(i, j)$ in the batch where $\lambda_i < \lambda_j$ and $y_j > y_i$:
     $$\mathcal{L}_{\text{rank}} = \frac{1}{|P|} \sum_{(i, j) \in P} \text{ReLU}\big(0 - (s_j - s_i)\big)$$
5. **`nrcomp/data/dataset.py`:**
   - `LatentDataset`: Loads latents from `.npz` files using `LatentStore` and `manifest.parquet`.
   - Collate function: Zero-pads batches to the maximum spatial dimensions in the batch and updates `valid_mask`.
   - `SourceImageBatchSampler`: Samples $B=8$ source images and $k=3$ random lambdas per image (batch size $N = 24$), guaranteeing valid intra-image comparison pairs in every optimization step.
6. **`nrcomp/train.py`:**
   - CLI entry point: `python -m nrcomp.train --config configs/train_blind.yaml`
   - Optimizer: AdamW ($3 \times 10^{-4}$, weight decay 0.05).
   - Scheduler: Cosine annealing with 2 warmup epochs, 60 total epochs.
   - Mixed precision: `torch.autocast('cuda', dtype=torch.bfloat16)`.
   - Checkpoint selection: Best validation **within-lambda SROCC** averaged across all 6 lambdas.
   - Output artifacts:
     - `reports/training_{config}_history.csv`
     - `reports/training_{config}_curves.png` (Loss curves, SROCC/PLCC curves, LR schedule)
     - `weights/student_{config}_best.pt`

---

## 4. Key CLI Commands Reference

```bash
# Environment activation on execution server:
conda activate nrcomp-main

# Run Stage 3 Full Diagnostics:
conda run --no-capture-output -n nrcomp-main python scripts/label_diagnostics.py --manifest data/manifest.parquet --output reports/label_diagnostics_full.md
conda run --no-capture-output -n nrcomp-main python scripts/latent_diagnostics.py --manifest data/manifest.parquet --output reports/latent_diagnostics_full.md

# Train Blind Student (M4):
conda run --no-capture-output -n nrcomp-main python -m nrcomp.train --config configs/train_blind.yaml

# Train Conditioned Student (M4):
conda run --no-capture-output -n nrcomp-main python -m nrcomp.train --config configs/train_conditioned.yaml

# Train Config B Student (y_hat + z_hat) (M4):
conda run --no-capture-output -n nrcomp-main python -m nrcomp.train --config configs/train_config_b.yaml

# Run Unit Tests locally (Pure Python):
python -m unittest discover tests/unit
```
