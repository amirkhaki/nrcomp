"""Unit tests for SmoothL1 and CompositeLoss reference oracles (pure Python)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.model.loss import composite_loss_oracle, smooth_l1_oracle


class TestLossOracle(unittest.TestCase):
    def test_smooth_l1_oracle_quadratic_region(self):
        # diff < beta: 0.5 * diff^2 / beta
        beta = 0.05
        diff = 0.02
        pred = 0.82
        target = 0.80
        loss = smooth_l1_oracle(pred, target, beta=beta)
        expected = 0.5 * (diff ** 2) / beta
        self.assertAlmostEqual(loss, expected, places=6)

    def test_smooth_l1_oracle_linear_region(self):
        # diff >= beta: diff - 0.5 * beta
        beta = 0.05
        diff = 0.10
        pred = 0.90
        target = 0.80
        loss = smooth_l1_oracle(pred, target, beta=beta)
        expected = diff - 0.5 * beta
        self.assertAlmostEqual(loss, expected, places=6)

    def test_composite_loss_oracle_perfect_ordering(self):
        # Two samples from the same image with lambda 0 and 1
        # lambda 0: target 0.70, pred 0.70
        # lambda 1: target 0.85, pred 0.85
        predictions = [0.70, 0.85]
        targets = [0.70, 0.85]
        lambda_indices = [0, 1]
        image_ids = ["img_001", "img_001"]

        total_loss, details = composite_loss_oracle(
            predictions, targets, lambda_indices, image_ids, alpha=0.5, beta=0.05, margin=0.0
        )
        self.assertAlmostEqual(details["smooth_l1"], 0.0)
        self.assertAlmostEqual(details["rank_loss"], 0.0)
        self.assertEqual(details["num_pairs"], 1.0)
        self.assertAlmostEqual(total_loss, 0.0)

    def test_composite_loss_oracle_inverted_ordering(self):
        # Inverted predictions: pred for lambda 1 is LESS than lambda 0
        # lambda 0: pred 0.80
        # lambda 1: pred 0.60 -> diff = pred_1 - pred_0 = -0.20
        # ReLU(0 - (-0.20)) = 0.20
        predictions = [0.80, 0.60]
        targets = [0.70, 0.85]
        lambda_indices = [0, 1]
        image_ids = ["img_001", "img_001"]

        total_loss, details = composite_loss_oracle(
            predictions, targets, lambda_indices, image_ids, alpha=0.5, beta=0.05, margin=0.0
        )
        self.assertAlmostEqual(details["rank_loss"], 0.20, places=5)
        self.assertEqual(details["num_pairs"], 1.0)

    def test_composite_loss_different_images_ignored(self):
        # Samples from different images should never form a ranking pair
        predictions = [0.90, 0.50]
        targets = [0.60, 0.80]
        lambda_indices = [0, 1]
        image_ids = ["img_001", "img_002"]

        total_loss, details = composite_loss_oracle(
            predictions, targets, lambda_indices, image_ids, alpha=0.5
        )
        self.assertEqual(details["num_pairs"], 0.0)
        self.assertAlmostEqual(details["rank_loss"], 0.0)

    def test_composite_loss_teacher_disagreement_ignored(self):
        # Even if same image, if teacher has targets[1] <= targets[0], pair is ignored
        predictions = [0.80, 0.60]
        targets = [0.85, 0.70]  # teacher decreases with lambda
        lambda_indices = [0, 1]
        image_ids = ["img_001", "img_001"]

        total_loss, details = composite_loss_oracle(
            predictions, targets, lambda_indices, image_ids, alpha=0.5
        )
        self.assertEqual(details["num_pairs"], 0.0)
        self.assertAlmostEqual(details["rank_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
