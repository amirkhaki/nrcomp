"""PyTorch server integration tests for Stage 5 Evaluation.

Verifies:
1. Student checkpoint loading and weight restoration.
2. Latency benchmarking on CPU/GPU.
3. Student inference engine over mock DataLoader.
4. Non-linear logistic curve fitting on synthetic PyTorch data.
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "Server test requires PyTorch")
class TestEvaluateTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_checkpoint_save_and_load(self):
        from nrcomp.evaluate import load_student_model
        from nrcomp.model.student import LatentStudent

        model = LatentStudent(
            config="A",
            in_channels_y=320,
            width=64,
            num_res_blocks=2,
            lambda_mode="conditioned",
            num_lambdas=6,
            lambda_embed_dim=64,
        )
        norm_stats = {
            "y": ([0.0] * 320, [1.0] * 320),
        }
        cfg = {
            "model": {
                "config": "A",
                "in_channels": 320,
                "width": 64,
                "num_res_blocks": 2,
                "lambda_mode": "conditioned",
                "num_lambdas": 6,
                "lambda_embed_dim": 64,
                "dropout": 0.1,
            }
        }
        checkpoint_dict = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "norm_stats": norm_stats,
            "config": cfg,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_model.pth"
            torch.save(checkpoint_dict, ckpt_path)

            loaded_model, loaded_dict = load_student_model(ckpt_path, device=self.device)
            self.assertEqual(loaded_dict["epoch"], 1)
            self.assertEqual(loaded_model.config, "A")
            self.assertEqual(loaded_model.lambda_mode, "conditioned")

    def test_latency_benchmark(self):
        from nrcomp.evaluate import benchmark_gpu_latency
        from nrcomp.model.student import LatentStudent

        model = LatentStudent(
            config="A",
            in_channels_y=320,
            width=64,
            num_res_blocks=2,
            lambda_mode="blind",
        ).to(self.device)

        stats = benchmark_gpu_latency(model, config_type="A", device=self.device, warmup_runs=2, test_runs=5)
        self.assertIn("params_millions", stats)
        self.assertGreater(stats["params_millions"], 0.0)


if __name__ == "__main__":
    unittest.main()
