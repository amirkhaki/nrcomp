"""Unit tests for scripts/eval_pretrained_nr.py (offline / pure Python)."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_pretrained_nr import (
    DEFAULT_NR_MODELS,
    LOWER_BETTER_METRICS,
    evaluate_split_records,
    format_report_markdown,
    resolve_orig_path,
    resolve_recon_path,
)


class TestEvalPretrainedNR(unittest.TestCase):
    def test_lower_better_metrics_definition(self):
        self.assertIn("niqe", LOWER_BETTER_METRICS)
        self.assertIn("brisque", LOWER_BETTER_METRICS)
        self.assertNotIn("clipiqa", LOWER_BETTER_METRICS)
        self.assertNotIn("musiq", LOWER_BETTER_METRICS)

    def test_resolve_recon_path(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            target = tmp / "data" / "recons" / "koniq" / "lambda_2" / "test1234.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("dummy")

            # 1. Path in recon_path
            row1 = {"recon_path": str(target)}
            res1 = resolve_recon_path(row1, tmp)
            self.assertEqual(res1, target)

            # 2. Relative path in recon_path
            rel = "data/recons/koniq/lambda_2/test1234.png"
            row2 = {"recon_path": rel}
            res2 = resolve_recon_path(row2, tmp)
            self.assertEqual(res2, target)

            # 3. Inferred from dataset, lambda_idx, image_id
            row3 = {"dataset": "koniq", "lambda_idx": 2, "image_id": "test1234.png"}
            res3 = resolve_recon_path(row3, tmp)
            self.assertEqual(res3, target)

            # 4. Inferred stem
            row4 = {"dataset": "koniq", "lambda_idx": 2, "image_id": "test1234.jpg"}
            res4 = resolve_recon_path(row4, tmp)
            self.assertEqual(res4, target)

            # 5. Missing
            row5 = {"dataset": "koniq", "lambda_idx": 5, "image_id": "nonexistent.jpg"}
            res5 = resolve_recon_path(row5, tmp)
            self.assertIsNone(res5)

    def test_resolve_orig_path(self):
        with TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            orig_target = tmp / "data" / "raw" / "koniq10k" / "1024x768" / "orig123.jpg"
            orig_target.parent.mkdir(parents=True, exist_ok=True)
            orig_target.write_text("dummy")

            # Inferred from raw directory
            row = {"dataset": "koniq", "image_id": "orig123.jpg"}
            res = resolve_orig_path(row, tmp)
            self.assertEqual(res, orig_target)

            # Direct path in orig_path
            row2 = {"orig_path": str(orig_target)}
            res2 = resolve_orig_path(row2, tmp)
            self.assertEqual(res2, orig_target)

    def test_evaluate_split_records(self):
        records = [
            {
                "image_id": f"img_{i}",
                "lambda_idx": i % 3,
                "aligned_score": float(i) * 0.1,
                "s_recon": float(i) * 0.1,
                "s_orig": 0.8,
                "delta_s": float(i) * 0.1 - 0.8,
                "target": float(i) * 0.15,
                "bpp_actual": 0.5,
                "orig_mos": 50.0 + i,
            }
            for i in range(15)
        ]
        res = evaluate_split_records(records, fitter=None, n_bootstrap=50)
        self.assertEqual(res["n_samples"], 15)
        self.assertEqual(res["n_images"], 15)
        self.assertIn("within_lambda_srocc", res)
        self.assertIn("ds_auc", res)
        self.assertIn("pwrc", res)
        self.assertIn("within_lambda_partial_spearman_bpp", res)
        self.assertIn("per_lambda_partial_spearman_bpp", res)
        self.assertIsNotNone(res["within_lambda_srocc_ci"][0])
        self.assertIn("delta", res)
        self.assertIn("orig_stats", res)
        self.assertEqual(res["orig_stats"]["n_unique_images"], 15)
        self.assertAlmostEqual(res["orig_stats"]["mean_s_orig"], 0.8)

    def test_format_report_markdown(self):
        prov = "> [!NOTE]\n> Provenance test header\n"
        results = {
            "clipiqa": {
                "model": "clipiqa",
                "backend": "pyiqa",
                "target_metric": "haarpsi",
                "is_lower_better": False,
                "splits": {
                    "koniq_test": {
                        "n_samples": 480,
                        "within_lambda_srocc": 0.3123,
                        "within_lambda_srocc_ci": [0.2500, 0.3700],
                        "srocc": 0.4500,
                        "plcc": 0.4200,
                        "within_lambda_partial_spearman_bpp": 0.1200,
                        "ds_auc": 0.6500,
                        "mos_independence": {"s_mos_partial_spearman": 0.6200},
                    },
                    "livec_test": {
                        "n_samples": 1200,
                        "within_lambda_srocc": 0.2840,
                        "within_lambda_srocc_ci": [0.2200, 0.3400],
                        "srocc": 0.4100,
                        "plcc": 0.3900,
                        "within_lambda_partial_spearman_bpp": 0.1100,
                        "ds_auc": 0.6100,
                        "mos_independence": {"s_mos_partial_spearman": 0.5800},
                    },
                },
            }
        }
        latent = {
            "student_conditioned_a": {
                "splits": {
                    "koniq_test": {
                        "within_lambda_srocc": 0.7797,
                        "within_lambda_srocc_ci": [0.6948, 0.8490],
                        "srocc": 0.9450,
                        "plcc": 0.9320,
                        "within_lambda_partial_spearman_bpp": 0.7600,
                        "ds_auc": 0.9229,
                        "mos_independence": {"s_mos_partial_spearman": 0.0330},
                    },
                    "livec_test": {
                        "within_lambda_srocc": 0.7419,
                        "within_lambda_srocc_ci": [0.6869, 0.7874],
                        "srocc": 0.9380,
                        "plcc": 0.9250,
                        "within_lambda_partial_spearman_bpp": 0.7200,
                        "ds_auc": 0.9513,
                        "mos_independence": {"s_mos_partial_spearman": -0.0290},
                    },
                }
            }
        }
        md = format_report_markdown(prov, results, latent)
        self.assertIn("# Pretrained No-Reference (NR-IQA) Evaluation Report", md)
        self.assertIn("Provenance test header", md)
        self.assertIn("CLIPIQA", md)
        self.assertIn("0.3123", md)
        self.assertIn("0.7797", md)
        self.assertIn("## 3. Independence from Source Image Quality (MOS Independence)", md)


if __name__ == "__main__":
    unittest.main()
