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

    def test_baselines_reexport(self):
        from nrcomp.baselines import (
            compute_pwrc_within_lambda,
            compute_within_lambda_partial_spearman,
        )
        self.assertTrue(callable(compute_pwrc_within_lambda))
        self.assertTrue(callable(compute_within_lambda_partial_spearman))

    def test_train_pixel_student_importable(self):
        import scripts.train_pixel_student as tps
        self.assertTrue(hasattr(tps, "PixelReconDataset"))
        self.assertTrue(hasattr(tps, "main"))



if __name__ == "__main__":
    unittest.main()
