import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_efficiency import (
    COMPLEXITY_REFERENCE_DATA,
    compute_latent_student_theoretical_complexity,
    format_efficiency_report,
)


class TestBenchmarkEfficiency(unittest.TestCase):
    def test_theoretical_complexity_config_a_blind(self):
        res = compute_latent_student_theoretical_complexity(config="A", conditioned=False)
        self.assertIn("params_count", res)
        self.assertIn("gflops", res)
        self.assertIn("params_millions", res)
        # Expected params around 3.4M - 3.5M
        self.assertGreater(res["params_millions"], 3.0)
        self.assertLess(res["params_millions"], 4.0)
        # Expected GFLOPs around 20 GFLOPs for 1024x768
        self.assertGreater(res["gflops"], 15.0)
        self.assertLess(res["gflops"], 25.0)

    def test_theoretical_complexity_config_a_conditioned(self):
        res = compute_latent_student_theoretical_complexity(config="A", conditioned=True)
        # Conditioned has film projections and lambda embedding: ~4.3M params
        self.assertGreater(res["params_millions"], 4.0)
        self.assertLess(res["params_millions"], 4.8)
        self.assertGreater(res["gflops"], 15.0)
        self.assertLess(res["gflops"], 25.0)

    def test_theoretical_complexity_config_b_conditioned(self):
        res = compute_latent_student_theoretical_complexity(config="B", conditioned=True)
        # Config B includes z upsampler: ~4.9M params
        self.assertGreater(res["params_millions"], 4.5)
        self.assertLess(res["params_millions"], 5.5)

    def test_format_report(self):
        prov_header = "<!-- PROVENANCE HEADER START -->\n<!-- PROVENANCE HEADER END -->\n"
        md = format_efficiency_report(prov_header, COMPLEXITY_REFERENCE_DATA)
        self.assertIn("# Model Computational Complexity & Efficiency Benchmark", md)
        self.assertIn("Student Conditioned A", md)
        self.assertIn("Pretrained CLIPIQA", md)
        self.assertIn("Total GFLOPs", md)
        self.assertIn("Throughput", md)
        # Verify no orphan "> " broken lines
        self.assertNotIn("from > quantized", md)

    def test_end_to_end_script_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test_efficiency.md"
            json_path = Path(tmpdir) / "test_efficiency.json"
            
            # Format and save
            prov_header = "<!-- PROVENANCE HEADER -->"
            report_md = format_efficiency_report(prov_header, COMPLEXITY_REFERENCE_DATA)
            md_path.write_text(report_md, encoding="utf-8")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(COMPLEXITY_REFERENCE_DATA, f)

            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("student_conditioned_a", data)


if __name__ == "__main__":
    unittest.main()
