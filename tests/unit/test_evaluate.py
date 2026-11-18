"""Unit tests for Stage 5 Evaluation utilities and reference functions (pure Python)."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.evaluate import (
    Logistic4PFitter,
    compute_clustered_bootstrap_cis,
    compute_mean_within_image_srocc,
    compute_mos_independence,
    compute_rmse,
    compute_second_teacher_agreement,
    format_markdown_report,
    logistic_4p,
)


class TestEvaluateOracles(unittest.TestCase):
    def test_logistic_4p_scalar_and_list(self):
        # f(s) = beta2 + (beta1 - beta2) / (1 + exp(-(s - beta3)/|beta4|))
        b1, b2, b3, b4 = 1.0, 0.0, 0.5, 0.1
        # At s = beta3 = 0.5, exp(0) = 1, f(0.5) = 0 + 1 / 2 = 0.5
        val_mid = logistic_4p(0.5, b1, b2, b3, b4)
        self.assertAlmostEqual(val_mid, 0.5, places=4)

        # At high s, f(s) -> beta1
        val_high = logistic_4p(5.0, b1, b2, b3, b4)
        self.assertAlmostEqual(val_high, 1.0, places=3)

        # At low s, f(s) -> beta2
        val_low = logistic_4p(-5.0, b1, b2, b3, b4)
        self.assertAlmostEqual(val_low, 0.0, places=3)

        # List input
        res_list = logistic_4p([0.5, 5.0], b1, b2, b3, b4)
        self.assertEqual(len(res_list), 2)
        self.assertAlmostEqual(res_list[0], 0.5, places=4)

    def test_logistic_fitter_fallback(self):
        # Even without scipy/numpy, fitter provides linear fallback
        s = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [0.2, 0.4, 0.6, 0.8, 1.0]  # y = 2 * s
        fitter = Logistic4PFitter().fit(s, y)
        preds = fitter.predict([0.15, 0.25])
        self.assertEqual(len(preds), 2)
        self.assertAlmostEqual(preds[0], 0.3, places=2)
        self.assertAlmostEqual(preds[1], 0.5, places=2)

    def test_compute_rmse(self):
        p = [1.0, 2.0, 3.0]
        t = [1.0, 2.0, 5.0]  # errors: 0, 0, -2 -> MSE = 4 / 3 = 1.3333 -> RMSE = 1.1547
        rmse_val = compute_rmse(p, t)
        self.assertAlmostEqual(rmse_val, (4.0 / 3.0) ** 0.5, places=4)

        # Empty / mismatched
        self.assertEqual(compute_rmse([], []), 0.0)
        self.assertEqual(compute_rmse([1.0], [1.0, 2.0]), 0.0)

    def test_mean_within_image_srocc(self):
        # 2 images, each with 3 lambdas
        # img1: monotonically increasing -> SROCC = 1.0
        # img2: monotonically decreasing -> SROCC = -1.0
        preds = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
        targets = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
        image_ids = ["img1", "img1", "img1", "img2", "img2", "img2"]

        mean_srocc, per_img = compute_mean_within_image_srocc(preds, targets, image_ids)
        self.assertAlmostEqual(per_img["img1"], 1.0, places=4)
        self.assertAlmostEqual(per_img["img2"], -1.0, places=4)
        self.assertAlmostEqual(mean_srocc, 0.0, places=4)

    def test_clustered_bootstrap_cis(self):
        # 10 synthetic images, 2 lambdas each
        records = []
        for i in range(10):
            img_id = f"img_{i}"
            records.append({
                "image_id": img_id,
                "mapped_pred": float(i) * 0.1,
                "target": float(i) * 0.1,
            })
            records.append({
                "image_id": img_id,
                "mapped_pred": float(i) * 0.1 + 0.05,
                "target": float(i) * 0.1 + 0.05,
            })

        cis = compute_clustered_bootstrap_cis(
            records, pred_key="mapped_pred", target_key="target", n_resamples=50, seed=42
        )
        self.assertIn("srocc", cis)
        self.assertIn("plcc", cis)
        self.assertIn("rmse", cis)
        # Perfect correlation -> CI should be centered at 1.0
        self.assertAlmostEqual(cis["srocc"][0], 1.0, places=2)
        self.assertAlmostEqual(cis["plcc"][0], 1.0, places=2)
        self.assertAlmostEqual(cis["rmse"][0], 0.0, places=4)

    def test_compute_mos_independence(self):
        preds = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
        targets = [0.25, 0.45, 0.65, 0.75, 0.85, 0.90]
        orig_mos = [20.0, 30.0, 50.0, 60.0, 80.0, 90.0]
        lambdas = [0, 1, 2, 3, 4, 5]

        res = compute_mos_independence(preds, targets, orig_mos, lambdas)
        self.assertIn("overall", res)
        self.assertIn("tertile_mae", res)
        self.assertIn("Low (<= 33%)", res["tertile_mae"])
        self.assertIn("High (> 66%)", res["tertile_mae"])

    def test_compute_second_teacher_agreement(self):
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        # Low DISTS = high quality, so 1 - DISTS increases
        dists = [0.8, 0.6, 0.4, 0.2, 0.1]
        lambdas = [0, 1, 2, 3, 4]

        res = compute_second_teacher_agreement(preds, dists, lambdas)
        self.assertAlmostEqual(res["overall_srocc"], 1.0, places=4)

    def test_format_markdown_report(self):
        eval_results = {
            "baseline_bpp": {
                "e1": {
                    "srocc": 0.55, "srocc_ci": (0.50, 0.60),
                    "plcc": 0.52, "plcc_ci": (0.48, 0.56),
                    "rmse": 0.08, "rmse_ci": (0.07, 0.09),
                    "within_lambda_srocc": -0.15,
                    "per_lambda_srocc": {0: -0.1, 1: -0.2},
                    "mean_within_image_srocc": 0.85,
                    "partial_spearman_bpp": 0.10,
                },
                "e2": {
                    "srocc": 0.50, "srocc_ci": (0.45, 0.55),
                    "plcc": 0.48, "plcc_ci": (0.42, 0.54),
                    "rmse": 0.09,
                    "within_lambda_srocc": -0.18,
                },
                "e4": {"overall_srocc": 0.51, "mean_within_lambda_srocc": -0.12},
            },
            "student_conditioned_b": {
                "e1": {
                    "srocc": 0.96, "srocc_ci": (0.95, 0.97),
                    "plcc": 0.95, "plcc_ci": (0.94, 0.96),
                    "rmse": 0.02, "rmse_ci": (0.018, 0.022),
                    "within_lambda_srocc": 0.76,
                    "per_lambda_srocc": {0: 0.70, 1: 0.72, 2: 0.75, 3: 0.78, 4: 0.79, 5: 0.80},
                    "mean_within_image_srocc": 0.98,
                    "partial_spearman_bpp": 0.88,
                },
                "e2": {
                    "srocc": 0.94, "srocc_ci": (0.92, 0.96),
                    "plcc": 0.93, "plcc_ci": (0.91, 0.95),
                    "rmse": 0.025,
                    "within_lambda_srocc": 0.73,
                },
                "e3_koniq": {
                    "overall": {"rho_s_mos": 0.45, "rho_y_mos": 0.44, "partial_s_mos_given_y": 0.02},
                    "tertile_mae": {"Low (<= 33%)": 0.02, "Mid (33% - 66%)": 0.019, "High (> 66%)": 0.018},
                },
                "e4": {"overall_srocc": 0.89, "mean_within_lambda_srocc": 0.71},
            },
        }
        eff = {
            "student_conditioned_b": {
                "params_millions": 4.25,
                "latency_bs1_ms": 3.45,
                "latency_bs16_ms": 0.85,
                "throughput_img_per_sec": 1176.4,
            }
        }
        md = format_markdown_report(eval_results, eff)
        self.assertIn("# Stage 5: Main Evaluation Report", md)
        self.assertIn("Within-Lambda SROCC", md)
        self.assertIn("Student Conditioned B", md)
        self.assertIn("4.25M", md)


if __name__ == "__main__":
    unittest.main()
