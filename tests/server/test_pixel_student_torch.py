"""PyTorch server integration tests for PixelStudent and E6 components."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "Server test requires PyTorch")
class TestPixelStudentTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_pixel_student_forward(self):
        from nrcomp.model.pixel_student import PixelStudent

        model = PixelStudent(
            in_channels=3,
            base_channels=16,
            lambda_mode="conditioned",
            num_lambdas=6,
            lambda_embed_dim=16,
        ).to(self.device)

        # Batch of 2 images (3x64x64)
        x = torch.randn(2, 3, 64, 64, device=self.device)
        lambda_idx = torch.tensor([0, 3], device=self.device)

        out = model(x, lambda_idx)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_pixel_student_blind_forward(self):
        from nrcomp.model.pixel_student import PixelStudent

        model = PixelStudent(
            in_channels=3,
            base_channels=16,
            lambda_mode="blind",
        ).to(self.device)

        x = torch.randn(2, 3, 64, 64, device=self.device)
        out = model(x)
        self.assertEqual(out.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
