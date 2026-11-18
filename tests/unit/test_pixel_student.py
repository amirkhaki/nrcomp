"""Unit tests for PixelStudent model structure (pure Python)."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nrcomp.model.pixel_student import PixelStudent


class TestPixelStudent(unittest.TestCase):
    def test_pixel_student_interface_conditioned(self):
        model = PixelStudent(
            in_channels=3,
            base_channels=64,
            lambda_mode="conditioned",
            num_lambdas=6,
            lambda_embed_dim=64,
            dropout=0.1,
        )
        self.assertEqual(model.in_channels, 3)
        self.assertEqual(model.base_channels, 64)
        self.assertEqual(model.lambda_mode, "conditioned")
        self.assertEqual(model.num_lambdas, 6)

    def test_pixel_student_interface_blind(self):
        model = PixelStudent(
            in_channels=3,
            base_channels=32,
            lambda_mode="blind",
        )
        self.assertEqual(model.lambda_mode, "blind")
        self.assertEqual(model.base_channels, 32)


if __name__ == "__main__":
    unittest.main()
