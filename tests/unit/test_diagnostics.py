"""Unit tests for label_diagnostics and latent_diagnostics helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.label_diagnostics import compute_percentiles, compute_ranks, spearman_rho
from scripts.latent_diagnostics import nearest_centroid_classifier


class TestDiagnostics(unittest.TestCase):
    def test_compute_ranks_simple(self):
        vals = [10.0, 20.0, 30.0]
        ranks = compute_ranks(vals)
        self.assertEqual(ranks, [1.0, 2.0, 3.0])

    def test_compute_ranks_with_ties(self):
        vals = [10.0, 20.0, 20.0, 30.0]
        ranks = compute_ranks(vals)
        self.assertEqual(ranks, [1.0, 2.5, 2.5, 4.0])

    def test_spearman_rho_perfect(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        rho = spearman_rho(x, y)
        self.assertAlmostEqual(rho, 1.0, places=4)

    def test_spearman_rho_negative(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [50.0, 40.0, 30.0, 20.0, 10.0]
        rho = spearman_rho(x, y)
        self.assertAlmostEqual(rho, -1.0, places=4)

    def test_compute_percentiles(self):
        vals = list(range(101))
        st = compute_percentiles(vals)
        self.assertEqual(st["min"], 0)
        self.assertEqual(st["max"], 100)
        self.assertEqual(st["median"], 50)
        self.assertEqual(st["p25"], 25)
        self.assertEqual(st["p75"], 75)

    def test_nearest_centroid_classifier(self):
        X_train = [[1.0, 1.0], [1.1, 1.2], [10.0, 10.0], [10.2, 9.8]]
        y_train = [0, 0, 1, 1]
        X_test = [[1.05, 0.95], [10.1, 10.2]]
        y_test = [0, 1]
        acc = nearest_centroid_classifier(X_train, y_train, X_test, y_test)
        self.assertEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
