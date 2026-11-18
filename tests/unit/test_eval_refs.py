"""Unit tests for Stage 6 Pixel References and Ablations reporting (pure Python)."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.eval_pixel_refs import (
    benchmark_pixel_vs_latent_pipeline,
    format_ablations_report,
    format_pixel_refs_report,
)


class TestEvalRefs(unittest.TestCase):
    def test_benchmark_pixel_vs_latent_pipeline_fallback(self):
        stats = benchmark_pixel_vs_latent_pipeline(None, None, device=None)
        self.assertIn("latent_student_ms", stats)
        self.assertIn("total_pixel_pipeline_ms", stats)
        self.assertIn("speedup_ratio", stats)
        self.assertGreater(stats["speedup_ratio"], 1.0)

    def test_format_pixel_refs_report(self):
        pre = {
            "clipiqa": {"overall_srocc": 0.48, "within_lambda_srocc": 0.21, "rho_mos": 0.68, "partial_mos": 0.54},
            "niqe": {"overall_srocc": 0.39, "within_lambda_srocc": 0.18, "rho_mos": 0.52, "partial_mos": 0.41},
        }
        pix = {
            "overall_srocc": 0.94, "within_lambda_srocc": 0.79, "rho_mos": 0.44, "partial_mos": 0.03
        }
        lat = {
            "latent_student_ms": 3.8,
            "synthesis_decode_ms": 28.5,
            "pixel_student_ms": 11.2,
            "total_pixel_pipeline_ms": 39.7,
            "speedup_ratio": 10.4,
        }
        md = format_pixel_refs_report(pre, pix, lat)
        self.assertIn("# Stage 6: Pixel References & Comparative Baselines", md)
        self.assertIn("10.4", md)
        self.assertIn("CLIPIQA", md)
        self.assertIn("NIQE", md)

    def test_format_ablations_report(self):
        md = format_ablations_report({})
        self.assertIn("# Core Architectural & Methodological Ablations", md)
        self.assertIn("A3", md)
        self.assertIn("A4", md)
        self.assertIn("A7", md)
        self.assertIn("A2", md)


if __name__ == "__main__":
    unittest.main()
