# E5 Shortcut Baselines Diagnostic Report

**Date:** November 2026  
**Target Metric:** `haarpsi`  
**Manifest:** `manifest.parquet` (6000 samples)  

> [!NOTE]
> **Baseline Definitions:**
> - **Baseline (a) [bpp-only]:** Gradient boosting regression from $bpp\_actual \to y$.
> - **Baseline (b) [lambda-only]:** Per-lambda mean degradation. Constant within each lambda (within-lambda SROCC is mathematically undefined, reported safely as 0.0000).
> - **Baseline (c) [latent-stats + bpp]:** Gradient boosting on pooled summary statistics of $\hat{y}$ (mean absolute value, std, L2 energy, quantiles, and near-zero fraction $|\hat{y}| < 0.5$) plus $bpp\_actual$.

## Evaluation on Split: `val` (468 samples)

| Baseline Model | Overall SROCC | Overall PLCC | Within-Lambda SROCC | Partial SROCC (ctl bpp) |
| --- | --- | --- | --- | --- |
| **(a) bpp_actual alone** | 0.5522 | 0.5314 | -0.1575 | 0.1394 |
| **(b) lambda_idx alone** | 0.9016 | 0.8766 | 0.0000 | 0.8660 |
| **(c) latent stats + bpp** | 0.9111 | 0.8936 | 0.4195 | 0.8737 |

### Per-Lambda SROCC Breakdown

| Lambda Index | Baseline (a) bpp | Baseline (b) lambda | Baseline (c) stats |
| --- | --- | --- | --- |
| $\lambda_0$ | -0.1012 | 0.0000 | 0.5104 |
| $\lambda_1$ | -0.1602 | 0.0000 | 0.5077 |
| $\lambda_2$ | -0.1743 | 0.0000 | 0.4229 |
| $\lambda_3$ | -0.2643 | 0.0000 | 0.4059 |
| $\lambda_4$ | -0.1665 | 0.0000 | 0.2507 |
| $\lambda_5$ | -0.0786 | 0.0000 | 0.4194 |

---

## Evaluation on Split: `test` (1662 samples)

| Baseline Model | Overall SROCC | Overall PLCC | Within-Lambda SROCC | Partial SROCC (ctl bpp) |
| --- | --- | --- | --- | --- |
| **(a) bpp_actual alone** | 0.5963 | 0.6054 | -0.2992 | 0.1146 |
| **(b) lambda_idx alone** | 0.9152 | 0.8994 | 0.0000 | 0.8849 |
| **(c) latent stats + bpp** | 0.9326 | 0.9193 | 0.4256 | 0.8923 |

### Per-Lambda SROCC Breakdown

| Lambda Index | Baseline (a) bpp | Baseline (b) lambda | Baseline (c) stats |
| --- | --- | --- | --- |
| $\lambda_0$ | -0.2058 | 0.0000 | 0.4342 |
| $\lambda_1$ | -0.3468 | 0.0000 | 0.4314 |
| $\lambda_2$ | -0.4464 | 0.0000 | 0.4053 |
| $\lambda_3$ | -0.4424 | 0.0000 | 0.4637 |
| $\lambda_4$ | -0.3114 | 0.0000 | 0.3929 |
| $\lambda_5$ | -0.0425 | 0.0000 | 0.4258 |
