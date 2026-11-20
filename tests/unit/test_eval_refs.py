"""Unit tests for Stage 6 Pixel References and Ablations reporting (pure Python)."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.eval_pixel_refs import (
    format_ablations_report,
    format_pixel_refs_report,
    load_run_artifact,
)


class TestEvalRefs(unittest.TestCase):
    def test_format_pixel_refs_report_not_run(self):
        prov = {"git_sha": "abc1234", "date": "2026-11-18"}
        md = format_pixel_refs_report(prov, None, None, {"clipiqa": None, "niqe": None}, None)
        self.assertIn("# Stage 6: Pixel References & Comparative Baselines", md)
        self.assertIn("NOT RUN", md)
        self.assertIn("abc1234", md)

    def test_format_pixel_refs_report_with_data(self):
        prov = {"git_sha": "abc1234", "date": "2026-11-18"}
        lat = {"srocc": 0.9412, "within_lambda_srocc": 0.8123, "rho_mos": 0.421, "partial_mos": 0.123}
        pix = {"srocc": 0.9350, "within_lambda_srocc": 0.7950, "rho_mos": 0.410, "partial_mos": 0.089}
        pre = {
            "clipiqa": {"srocc": 0.48, "within_lambda_srocc": 0.21, "rho_mos": 0.68, "partial_mos": 0.54},
            "niqe": {"srocc": 0.39, "within_lambda_srocc": 0.18, "rho_mos": 0.52, "partial_mos": 0.41},
        }
        bench = {
            "latent_student_ms": 3.8,
            "synthesis_decode_ms": 28.5,
            "pixel_student_ms": 11.2,
            "total_pixel_pipeline_ms": 39.7,
            "speedup_ratio": 10.4,
        }
        md = format_pixel_refs_report(prov, lat, pix, pre, bench)
        self.assertIn("# Stage 6: Pixel References & Comparative Baselines", md)
        self.assertIn("10.4x FASTER", md)
        self.assertIn("0.8123", md)
        self.assertIn("0.7950", md)

    def test_format_pixel_refs_report_with_runs_dir(self):
        prov = {"git_sha": "abc1234", "date": "2026-11-18"}
        lat = {"srocc": 0.9412, "within_lambda_srocc": 0.8123}
        runs_dir_dict = {
            "conditioned_b_seed42": {"splits": {"koniq_test": {"within_lambda_srocc": 0.770}, "livec_test": {"within_lambda_srocc": 0.750}}},
            "conditioned_b_seed43": {"splits": {"koniq_test": {"within_lambda_srocc": 0.772}, "livec_test": {"within_lambda_srocc": 0.752}}},
            "conditioned_b_seed44": {"splits": {"koniq_test": {"within_lambda_srocc": 0.768}, "livec_test": {"within_lambda_srocc": 0.748}}},
        }
        md = format_pixel_refs_report(prov, lat, None, {"clipiqa": None}, None, runs_dir=runs_dir_dict)
        self.assertIn("Latent student (Conditioned B, seed mean ± std)", md)
        self.assertIn("0.7700 ± 0.0020", md)
        self.assertIn("0.7500 ± 0.0020", md)
        self.assertIn("↳ Latent student (Conditioned B, ensemble)", md)

    def test_format_ablations_report(self):
        prov = {"git_sha": "abc1234", "date": "2026-11-18"}
        runs = {
            "student_conditioned_a": {"within_lambda_srocc": 0.8069, "_artifact_path": "reports/runs/a/metrics.json"},
            "student_conditioned_b": {"within_lambda_srocc": 0.8174, "_artifact_path": "reports/runs/b/metrics.json"},
        }
        md = format_ablations_report(prov, runs)
        self.assertIn("# Core Architectural & Methodological Ablations", md)
        self.assertIn("A3", md)
        self.assertIn("A4", md)
        self.assertIn("A7", md)
        self.assertIn("A2", md)
        self.assertIn("A5", md)
        self.assertIn("0.8069", md)
        self.assertIn("NOT RUN", md)


if __name__ == "__main__":
    unittest.main()
