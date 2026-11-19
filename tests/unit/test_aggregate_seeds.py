"""Unit tests for scripts/aggregate_seeds.py (pure Python)."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_seeds import (
    aggregate_model_seeds,
    compute_sample_mean_std,
    evaluate_ensemble,
    format_seed_aggregation_report,
)


class TestAggregateSeeds(unittest.TestCase):
    def test_compute_sample_mean_std(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean_val, std_val = compute_sample_mean_std(vals)
        self.assertAlmostEqual(mean_val, 3.0, places=4)
        # sample std: sqrt(10 / 4) = sqrt(2.5) = 1.5811
        self.assertAlmostEqual(std_val, 1.5811, places=3)

        # Single element
        m1, s1 = compute_sample_mean_std([42.0])
        self.assertEqual(m1, 42.0)
        self.assertEqual(s1, 0.0)

        # Empty
        m0, s0 = compute_sample_mean_std([])
        self.assertIsNone(m0)
        self.assertIsNone(s0)

    def test_evaluate_ensemble(self):
        seed_predictions = {
            42: [
                {"split": "test", "image_id": "img1", "lambda_idx": 0, "s": 0.8, "y": 0.85, "bpp": 0.2},
                {"split": "test", "image_id": "img1", "lambda_idx": 1, "s": 0.9, "y": 0.92, "bpp": 0.4},
            ],
            43: [
                {"split": "test", "image_id": "img1", "lambda_idx": 0, "s": 0.82, "y": 0.85, "bpp": 0.2},
                {"split": "test", "image_id": "img1", "lambda_idx": 1, "s": 0.94, "y": 0.92, "bpp": 0.4},
            ],
        }
        res = evaluate_ensemble(seed_predictions, split="test")
        self.assertIn("srocc", res)
        self.assertIn("plcc", res)
        self.assertIn("rmse", res)
        self.assertAlmostEqual(res["srocc"], 1.0, places=4)

    def test_aggregate_model_seeds_and_format_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runs_dir = Path(tmp_dir) / "reports" / "runs"
            runs_dir.mkdir(parents=True)

            # Create mock runs for seeds 42 and 43
            for seed in [42, 43]:
                rdir = runs_dir / f"student_blind_a_seed{seed}"
                rdir.mkdir(parents=True)
                mock_metrics = {
                    "e1": {
                        "srocc": 0.90 + seed * 0.001,
                        "plcc": 0.91 + seed * 0.001,
                        "rmse": 0.02,
                        "within_lambda_srocc": 0.75 + seed * 0.001,
                        "within_lambda_partial_spearman_bpp": 0.70,
                        "ds_auc": 0.85,
                    }
                }
                with open(rdir / "eval_metrics.json", "w") as f:
                    json.dump(mock_metrics, f)

                # Create predictions.csv
                csv_p = rdir / "predictions.csv"
                with open(csv_p, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["split", "image_id", "lambda_idx", "s", "y", "bpp"])
                    writer.writeheader()
                    writer.writerow({"split": "test", "image_id": "img1", "lambda_idx": 0, "s": 0.8, "y": 0.85, "bpp": 0.2})
                    writer.writerow({"split": "test", "image_id": "img1", "lambda_idx": 1, "s": 0.9, "y": 0.95, "bpp": 0.4})

            res = aggregate_model_seeds("student_blind_a", [42, 43], runs_dir)
            self.assertEqual(res["status"], "RUN")
            self.assertEqual(res["seeds_found"], [42, 43])
            self.assertIn("within_lambda_srocc", res["aggregated"])

            # Format report
            report_md = format_seed_aggregation_report(
                {"student_blind_a": res},
                seeds=[42, 43],
                manifest_path=Path("data/manifest.parquet"),
                lambdas_path=Path("configs/lambdas.yaml"),
            )
            self.assertIn("Multi-Seed Evaluation & Stability Report", report_md)
            self.assertIn("Student Blind A", report_md)
            self.assertIn("PROVENANCE HEADER START", report_md)


if __name__ == "__main__":
    unittest.main()
