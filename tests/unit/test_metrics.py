"""Unit tests for QualityMetrics and PSNR calculation (pure Python)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.labels.metrics import QualityMetrics, compute_psnr


class TestQualityMetrics(unittest.TestCase):
    def test_psnr_identical_images(self):
        img1 = [128] * (64 * 64 * 3)
        img2 = [128] * (64 * 64 * 3)
        psnr = compute_psnr(img1, img2)
        self.assertEqual(psnr, float("inf"))

    def test_psnr_different_images(self):
        img1 = [0] * (64 * 64 * 3)
        img2 = [255] * (64 * 64 * 3)
        psnr = compute_psnr(img1, img2)
        self.assertAlmostEqual(psnr, 0.0, places=2)

    def test_psnr_known_diff(self):
        # Difference of 1 LSB: MSE = 1.0, PSNR = 10 * log10(255^2 / 1) = 48.13 dB
        img1 = [0] * 100
        img2 = [1] * 100
        psnr = compute_psnr(img1, img2)
        self.assertAlmostEqual(psnr, 48.13, places=1)

    def test_quality_metrics_structure(self):
        metrics = QualityMetrics(device="cpu", enable_metrics=["haarpsi", "dists", "psnr"])
        self.assertEqual(metrics.device, "cpu")
        self.assertIn("haarpsi", metrics.enable_metrics)

        img1 = [100] * 100
        img2 = [105] * 100
        res = metrics.compute_all(img1, img2)
        self.assertIn("psnr", res)
        self.assertGreater(res["psnr"], 0.0)


if __name__ == "__main__":
    unittest.main()
