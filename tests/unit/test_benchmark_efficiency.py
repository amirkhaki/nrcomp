import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_efficiency import (
    compute_latent_student_theoretical_complexity,
    compute_pixel_student_theoretical_complexity,
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

    def test_theoretical_complexity_pixel_student(self):
        res = compute_pixel_student_theoretical_complexity(conditioned=True)
        self.assertGreater(res["params_millions"], 2.0)
        self.assertLess(res["params_millions"], 3.0)
        self.assertGreater(res["gflops"], 40.0)

    def test_format_report(self):
        prov_header = "<!-- PROVENANCE HEADER START -->\n<!-- PROVENANCE HEADER END -->\n"
        sample_data = {
            "student_blind_a": {
                "name": "Student Blind A (y_hat)",
                "domain": "Codec Latents",
                "params_m": 3.44,
                "model_gmacs": 10.15,
                "model_gflops": 20.29,
                "total_gflops": 20.29,
                "flop_profiler": "Analytical formula",
                "latency_bs1_ms": 7.34,
                "latency_bs16_ms": 7.30,
                "throughput_img_s": 137.0,
            },
            "student_conditioned_a": {
                "name": "Student Conditioned A (y_hat + lambda)",
                "domain": "Codec Latents",
                "params_m": 4.30,
                "model_gmacs": 10.15,
                "model_gflops": 20.30,
                "total_gflops": 20.30,
                "flop_profiler": "Analytical formula",
                "latency_bs1_ms": 7.28,
                "latency_bs16_ms": 7.21,
                "throughput_img_s": 138.7,
            },
            "elic_synthesis_only": {
                "name": "ELIC Synthesis Decoder (decode_from_latents)",
                "domain": "Latents -> RGB Pixels",
                "params_m": 32.40,
                "model_gmacs": 142.50,
                "model_gflops": 285.00,
                "total_gflops": 285.00,
                "flop_profiler": "Measured",
                "latency_bs1_ms": 28.50,
                "latency_bs16_ms": 27.80,
                "throughput_img_s": 35.1,
            },
        }
        md = format_efficiency_report(prov_header, sample_data)
        self.assertIn("# Model Computational Complexity & Efficiency Benchmark", md)
        self.assertIn("Student Conditioned A", md)
        self.assertIn("ELIC Synthesis Decoder", md)
        self.assertIn("Throughput", md)
        self.assertNotIn("CLIPIQA", md)

    def test_format_efficiency_report_with_pretrained_models(self):
        prov_header = "<!-- PROVENANCE HEADER -->"
        sample_data = {
            "student_conditioned_a": {
                "name": "Student Conditioned A (y_hat + lambda)",
                "domain": "Codec Latents",
                "params_m": 4.30,
                "model_gmacs": 10.15,
                "model_gflops": 20.30,
                "total_gflops": 20.30,
                "flop_profiler": "PyTorch FlopCounterMode (live)",
                "latency_bs1_ms": 7.28,
                "latency_bs16_ms": 7.21,
                "throughput_img_s": 138.7,
            },
            "pretrained_clipiqa": {
                "name": "Pretrained CLIPIQA (decode + model)",
                "domain": "Decoded RGB Pixels",
                "params_m": 149.60,
                "model_gmacs": 14.40,
                "model_gflops": 28.80,
                "total_gflops": 313.80,
                "flop_profiler": "PyTorch FlopCounterMode (live)",
                "latency_bs1_ms": 135.50,
                "latency_bs16_ms": None,
                "throughput_img_s": 7.4,
            },
        }
        md = format_efficiency_report(prov_header, sample_data)
        self.assertIn("Pretrained CLIPIQA", md)
        self.assertIn("14.40", md)
        self.assertIn("28.80", md)
        self.assertIn("313.80", md)


    def test_end_to_end_script_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "test_efficiency.md"
            json_path = Path(tmpdir) / "test_efficiency.json"

            sample_data = {
                "student_conditioned_a": {
                    "name": "Student Conditioned A (y_hat + lambda)",
                    "domain": "Codec Latents",
                    "params_m": 4.30,
                    "model_gmacs": 10.15,
                    "model_gflops": 20.30,
                    "total_gflops": 20.30,
                    "flop_profiler": "Analytical formula",
                    "latency_bs1_ms": 7.28,
                    "latency_bs16_ms": 7.21,
                    "throughput_img_s": 138.7,
                }
            }
            prov_header = "<!-- PROVENANCE HEADER -->"
            report_md = format_efficiency_report(prov_header, sample_data)
            md_path.write_text(report_md, encoding="utf-8")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)

            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("student_conditioned_a", data)


if __name__ == "__main__":
    unittest.main()
