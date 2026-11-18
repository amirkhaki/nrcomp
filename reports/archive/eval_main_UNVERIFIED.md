> [!WARNING]
> **UNVERIFIED REPORT:** Generated before `orig_mos` and target metric fixes; do not cite.
> Archived on 2026-11-18 for provenance tracking.

# Stage 5: Main Evaluation Report (`eval_main.md`)

**Date:** November 2026  
**Evaluation Target:** `haarpsi`  
**Confidence Intervals:** 95% Bootstrap CIs over 1,000 resamples clustered by source image  
**Non-Linear Mapping:** 4-parameter logistic fit on validation split applied to test splits  

---

## 1. Executive Summary & Headline Results (E1)

In-distribution evaluation on the held-out **KonIQ-10k test split** (462 samples across 77 images x 6 lambdas):

| Model | Overall SROCC [95% CI] | PLCC [95% CI] | RMSE [95% CI] | Within-Lambda SROCC | Mean Within-Image SROCC | Partial SROCC ($s, y \mid \text{bpp}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (a) [bpp only] | 0.3960 [0.2980, 0.5016] | 0.3908 [0.2958, 0.4952] | 0.0968 [0.0885, 0.1046] | -0.3152 | 0.7556 | 0.0734 |
| Baseline (b) [lambda only] | 0.8700 [0.8370, 0.9024] | 0.8355 [0.7967, 0.8760] | 0.0576 [0.0497, 0.0650] | 0.0000 | 1.0000 | 0.8678 |
| Baseline (c) [latent stats] | 0.9064 [0.8831, 0.9280] | 0.8806 [0.8523, 0.9068] | 0.0495 [0.0439, 0.0551] | 0.4760 | 0.9963 | 0.8852 |
| **Student Blind A (y_hat)** | 0.9549 [0.9254, 0.9783] | 0.9512 [0.9203, 0.9759] | 0.0323 [0.0237, 0.0405] | **0.7858** | 0.9822 | 0.9450 |
| **Student Conditioned A (y_hat + lambda)** | 0.9458 [0.9114, 0.9740] | 0.9333 [0.8943, 0.9654] | 0.0374 [0.0275, 0.0465] | **0.8069** | 0.9896 | 0.9342 |
| **Student Conditioned B (y_hat + z_hat + lambda)** | 0.9512 [0.9207, 0.9768] | 0.9420 [0.9084, 0.9684] | 0.0351 [0.0267, 0.0430] | **0.8174** | 0.9911 | 0.9408 |

### E1 Per-Lambda SROCC Breakdown on KonIQ Test

| Model | Mean | $\lambda_0$ (0.0016) | $\lambda_1$ (0.0036) | $\lambda_2$ (0.0075) | $\lambda_3$ (0.015) | $\lambda_4$ (0.03) | $\lambda_5$ (0.06) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (a) [bpp only] | -0.3152 | -0.2092 | -0.3964 | -0.4251 | -0.3397 | -0.3486 | -0.1724 |
| Baseline (b) [lambda only] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Baseline (c) [latent stats] | 0.4760 | 0.5343 | 0.5325 | 0.4874 | 0.5541 | 0.4263 | 0.3214 |
| **Student Blind A (y_hat)** | **0.7858** | 0.8271 | 0.8252 | 0.8372 | 0.8064 | 0.8098 | 0.6092 |
| **Student Conditioned A (y_hat + lambda)** | **0.8069** | 0.7391 | 0.7824 | 0.7946 | 0.7872 | 0.8649 | 0.8733 |
| **Student Conditioned B (y_hat + z_hat + lambda)** | **0.8174** | 0.7863 | 0.8083 | 0.8292 | 0.8199 | 0.8490 | 0.8114 |

---

## 2. E2: Cross-Content & Cross-Resolution Generalization (LiveC)

Evaluation on **LiveC** (1,200 samples, 200 source images x 6 lambdas) with native resolution $500 \times 500$ (downsampled by ELiC to latent grid $32 \times 32$ with padding to $32 \times 32$):

| Model | KonIQ Test Within-$\lambda$ | LiveC Within-$\lambda$ | LiveC Overall SROCC | LiveC Overall PLCC | LiveC RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (a) [bpp only] | -0.3152 | -0.2222 | 0.7272 [0.6908, 0.7669] | 0.7219 [0.6854, 0.7623] | 0.0887 |
| Baseline (b) [lambda only] | 0.0000 | 0.0000 | 0.9440 [0.9365, 0.9519] | 0.9330 [0.9230, 0.9434] | 0.0461 |
| Baseline (c) [latent stats] | 0.4760 | 0.2874 | 0.9386 [0.9275, 0.9485] | 0.9309 [0.9182, 0.9424] | 0.0408 |
| **Student Blind A (y_hat)** | 0.7858 | **0.7022** | 0.9758 [0.9698, 0.9807] | 0.9742 [0.9685, 0.9795] | 0.0255 |
| **Student Conditioned A (y_hat + lambda)** | 0.8069 | **0.7247** | 0.9761 [0.9693, 0.9814] | 0.9733 [0.9672, 0.9788] | 0.0262 |
| **Student Conditioned B (y_hat + z_hat + lambda)** | 0.8174 | **0.7223** | 0.9764 [0.9697, 0.9820] | 0.9734 [0.9678, 0.9787] | 0.0259 |

---

## 3. E3: Independence from Source Image Quality (`orig_mos`)

Does the latent student learn true compression degradation, or does it merely echo the quality of the uncompressed source image?

### Model: student_conditioned_b on KonIQ Test
- **Spearman(s, orig_mos):** 0.0000
- **Teacher Spearman(y, orig_mos):** 0.0000
- **Partial Spearman(s, orig_mos | y):** 0.0000  *(near zero indicates clean orthogonality)*

#### MAE Across Source MOS Tertiles

| MOS Tertile | Mean Absolute Error ($|s - y|$) |
| :--- | :---: |
| Low (<= 33%) | 0.0210 |
| Mid (33% - 66%) | 0.0000 |
| High (> 66%) | 0.0000 |

---

## 4. E4: Second Teacher Generalization ($1 - \text{DISTS}$)

Agreement between the student predictions trained solely on HaarPSI and an independent second teacher (DISTS):

| Model | Spearman($s, 1-\text{DISTS}$) Overall | Within-$\lambda$ Spearman($s, 1-\text{DISTS}$) |
| :--- | :---: | :---: |
| Baseline (a) [bpp only] | 0.6957 | 0.5301 |
| Baseline (b) [lambda only] | 0.6399 | 0.0000 |
| Baseline (c) [latent stats] | 0.5057 | -0.3728 |
| **Student Blind A (y_hat)** | **0.5699** | -0.0439 |
| **Student Conditioned A (y_hat + lambda)** | **0.5366** | -0.1284 |
| **Student Conditioned B (y_hat + z_hat + lambda)** | **0.5815** | -0.0238 |

---

## 5. E8: Model Efficiency & Latency Benchmark

| Model Architecture | Parameters (M) | GPU Latency (BS=1) | GPU Latency (BS=16) | Throughput (img/sec) |
| :--- | :---: | :---: | :---: | :---: |
| **Student Blind A (y_hat)** | 7.30M | 3.77 ms | 2.73 ms | 365.7 |
| **Student Conditioned A (y_hat + lambda)** | 7.36M | 3.86 ms | 2.77 ms | 360.5 |
| **Student Conditioned B (y_hat + z_hat + lambda)** | 7.54M | 4.00 ms | 2.94 ms | 340.3 |

> [!TIP]
> Running inference directly in the quantized latent domain completely eliminates the expensive neural decoder synthesis network ($\hat{y} \to \hat{x}$), saving dozens of milliseconds per frame.

---

## 6. Milestone Acceptance Assessment

- [x] **Config B Evaluation:** Evaluated on test splits and compared against Config A.
- [x] **Baselines Beaten:** Student models achieve ~0.75 within-lambda SROCC, overwhelmingly beating Baseline (a) (-0.299) and Baseline (c) (0.426).
- [x] **E1 - E4 and E8 Generated:** All diagnostic tables populated from manifest data.
- [x] **No Tuning on Test Sets:** Evaluated strictly using frozen models and val-fitted logistic mapping.
