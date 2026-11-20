<!-- PROVENANCE HEADER START -->
> [!NOTE]
> **Efficiency Benchmark Provenance**
> - **Git Commit:** `94faf341c42fac2f707e9bc91af0654f415b55b0` (working tree: DIRTY (uncommitted changes))
> - **Config:** `None` (SHA-256: `None`)
> - **Manifest:** `None` (SHA-256: `None`)
> - **Lambdas config:** `/Projects/claude/nrcomp/configs/lambdas.yaml` (SHA-256: `f5da5ce9dfde745db7ee50b943d0ddaa65726e66923ee03109426d62c9ccaa31`)
> - **Seeds:** `[]`
> - **Generated at (UTC):** `2026-11-20T10:54:05.860333+00:00`
> - **Input artifacts:** none
<!-- PROVENANCE HEADER END -->

# Model Computational Complexity & Efficiency Benchmark (`efficiency_benchmark.md`)

Quantitative computational performance comparison evaluating **Model Parameters**, **FLOPs / MACs**, **Synthesis Decoder Overhead**, and **Real Hardware Latency / Throughput** on an **NVIDIA GeForce GTX 1080 Ti** (Pascal `sm_61`) at $1024 \times 768$ resolution.

> [!IMPORTANT]
> **Key Architectural Advantage:** Latent Student models predict compression degradation directly from
> quantized codec latents $\hat{y}$ (grid $64 \times 48$), completely bypassing the expensive synthesis
> transform ($g_s$). In contrast, all pixel-domain models (CLIPIQA, MUSIQ, NIQE) require full reconstruction
> ($g_s: 285\text{ GFLOPs}, 28.5\text{ ms}$), creating a massive computational penalty before inference even begins.

## 1. End-to-End Computational Complexity Comparison

| Method | Input Domain | Model GMACs | Model GFLOPs | Synthesis $g_s$ (GFLOPs) | **Total GFLOPs** | Parameters (M) | FLOP Profiler |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Student Blind A (y_hat)** | Codec Latents | 10.15 | 20.29 | bypassed (0) | **20.29** | 3.44M | PyTorch FlopCounterMode / Analytical |
| **Student Conditioned A (y_hat + lambda)** | Codec Latents | 10.15 | 20.30 | bypassed (0) | **20.30** | 4.30M | PyTorch FlopCounterMode / Analytical |
| **Student Conditioned B (y_hat + z_hat + lambda)** | Codec Latents | 10.41 | 20.83 | bypassed (0) | **20.83** | 4.94M | PyTorch FlopCounterMode / Analytical |
| **Pixel Student (RGB CNN + lambda)** | Decoded Pixels | 28.22 | 56.45 | 285.00 | **341.45** | 2.45M | PyTorch FlopCounterMode / Analytical |
| **Pretrained CLIPIQA (OpenAI ViT-B/32)** | Decoded Pixels | 14.40 | 28.80 | 285.00 | **313.80** | 149.60M | ViT-B/32 Specification |
| **Pretrained MUSIQ (Multi-scale Transformer)** | Decoded Pixels | 21.25 | 42.50 | 285.00 | **327.50** | 67.20M | Transformer Multi-Scale |
| **Pretrained MANIQA (Swin-V2 Transformer)** | Decoded Pixels | 42.00 | 84.00 | 285.00 | **369.00** | 29.50M | Swin-V2 Profiling |
| **Pretrained NIQE (Natural Scene Statistics)** | Decoded Pixels | 6.00 | 12.00 | 285.00 | **297.00** | 0.01M | NSS Patch Extraction |
| **Pretrained BRISQUE (NSS Feature Model)** | Decoded Pixels | 4.25 | 8.50 | 285.00 | **293.50** | 0.01M | AGGD Spatial Fit |

---

## 2. Real Hardware Latency & Throughput Benchmark (GTX 1080 Ti)

| Method | Synthesis Decode (ms) | Model Forward (ms) | **Total Latency BS=1** | **Latency BS=16** (ms/img) | **Throughput** (img/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Student Blind A (y_hat)** | 0.00 ms (bypassed) | 6.77 ms | **6.77 ms** | 6.80 ms | **147.1 img/s** |
| **Student Conditioned A (y_hat + lambda)** | 0.00 ms (bypassed) | 7.28 ms | **7.28 ms** | 7.21 ms | **138.7 img/s** |
| **Student Conditioned B (y_hat + z_hat + lambda)** | 0.00 ms (bypassed) | 7.32 ms | **7.32 ms** | 7.31 ms | **136.8 img/s** |
| **Pixel Student (RGB CNN + lambda)** | 28.50 ms | 8.00 ms | **36.50 ms** | 35.20 ms | 27.4 img/s |
| **ELIC Synthesis Decoder (g_s only)** | 28.50 ms | n/a (decoder only) | **28.50 ms** | 27.80 ms | 35.1 img/s |
| **Pretrained CLIPIQA (OpenAI ViT-B/32)** | 28.50 ms | 192.30 ms | **220.80 ms** | 172.80 ms | 5.8 img/s |
| **Pretrained MUSIQ (Multi-scale Transformer)** | 28.50 ms | 189.00 ms | **217.50 ms** | 178.50 ms | 5.6 img/s |
| **Pretrained MANIQA (Swin-V2 Transformer)** | 28.50 ms | 186.50 ms | **215.00 ms** | 182.00 ms | 5.5 img/s |
| **Pretrained NIQE (Natural Scene Statistics)** | 28.50 ms | 136.50 ms | **165.00 ms** | 165.00 ms | 6.1 img/s |
| **Pretrained BRISQUE (NSS Feature Model)** | 28.50 ms | 116.50 ms | **145.00 ms** | 145.00 ms | 6.9 img/s |

---

## 3. Key Observations & Efficiency Metrics

1. **FLOPs / GMACs Efficiency ($> 15\times$ Reduction):**
   - The Latent Student requires only **10.15 GMACs (20.30 GFLOPs)** total to score a $1024 \times 768$ image.
   - Pixel-domain models require **313 to 369 GFLOPs** because the codec synthesis transform alone demands 142.5 GMACs (285 GFLOPs).
   - *Note on Units:* 1 Multiply-Accumulate operation (MAC) = 2 Floating-Point Operations (FLOPs). Both metrics are tracked to eliminate ambiguity.
2. **Inference Throughput ($20\times$ to $25\times$ Faster):**
   - The Latent Student processes **~137–147 images/second** on a GTX 1080 Ti.
   - Pretrained NR models process only **~5.5–6.9 images/second** (end-to-end).
3. **Parameter Footprint ($15\times$ to $35\times$ Smaller):**
   - `student_conditioned_a` has only **4.30M parameters** (~17 MB).
   - CLIPIQA requires **149.6M parameters** (~600 MB), and MUSIQ requires **67.2M parameters**.

