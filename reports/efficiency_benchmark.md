<!-- PROVENANCE HEADER START -->
> [!NOTE]
> **Efficiency Benchmark Provenance**
> - **Git Commit:** `901f03e70077b9134f8b505badfb6484a5d514a1` (working tree: DIRTY (uncommitted changes))
> - **Config:** `None` (SHA-256: `None`)
> - **Manifest:** `None` (SHA-256: `None`)
> - **Lambdas config:** `/Projects/claude/nrcomp/configs/lambdas.yaml` (SHA-256: `f5da5ce9dfde745db7ee50b943d0ddaa65726e66923ee03109426d62c9ccaa31`)
> - **Seeds:** `[]`
> - **Generated at (UTC):** `2026-11-20T11:58:52.792829+00:00`
> - **Input artifacts:** none
<!-- PROVENANCE HEADER END -->

# Model Computational Complexity & Efficiency Benchmark (`efficiency_benchmark.md`)

Empirical computational performance comparison evaluating **Model Parameters**, **FLOPs / MACs**, **Synthesis Decoder Overhead**, and **Measured Hardware Latency / Throughput** on an **NVIDIA GeForce GTX 1080 Ti** (Pascal `sm_61`) at $1024 \times 768$ resolution.

> [!IMPORTANT]
> **Key Architectural Advantage:** Latent Student models predict compression degradation directly from
> quantized codec latents $\hat{y}$ (grid $64 \times 48$), completely bypassing the expensive synthesis
> transform ($g_s$). In contrast, pixel-domain models require full reconstruction prior to inference.

## 1. Computational Complexity & Parameters

| Method | Input Domain | GMACs | GFLOPs | Parameters (M) | FLOP Profiler |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Student Blind A (y_hat)** | Codec Latents | 10.15 | **20.29** | 3.44M | Analytical formula |
| **Student Conditioned A (y_hat + lambda)** | Codec Latents | 10.15 | **20.30** | 4.30M | Analytical formula |
| **Student Conditioned B (y_hat + z_hat + lambda)** | Codec Latents | 10.41 | **20.83** | 4.94M | Analytical formula |
| **Pixel Student (RGB CNN)** | Decoded RGB Pixels | 28.22 | **56.45** | 2.45M | Analytical formula |

---

## 2. Real Hardware Latency & Throughput Benchmark (GTX 1080 Ti)

| Method | **Latency BS=1** | **Latency BS=16** (ms/img) | **Throughput** (img/s) | Speedup vs Cond-A |
| :--- | :---: | :---: | :---: | :---: |
| **Student Blind A (y_hat)** | **n/a** | n/a | n/a | n/a |
| **Student Conditioned A (y_hat + lambda)** | **n/a** | n/a | n/a | **Baseline (1.0x)** |
| **Student Conditioned B (y_hat + z_hat + lambda)** | **n/a** | n/a | n/a | n/a |
| **Pixel Student (RGB CNN)** | **n/a** | n/a | n/a | n/a |

---

## 3. Dynamic Observations Derived from Measured Values

3. **Parameter Footprint:** Latent Student Conditioned A has **4.30M parameters**.
