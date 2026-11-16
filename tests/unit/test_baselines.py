"""Unit tests for E5 shortcut baselines, rankings, and NaN-safe correlations."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nrcomp.baselines import (
    BppBaseline,
    LambdaBaseline,
    LatentStatsBaseline,
    compute_within_lambda_srocc,
    partial_spearman,
    pearson_r,
    spearman_rho,
)


class TestBaselines(unittest.TestCase):
    def test_spearman_constant_series_safe_zero(self):
        # When all predictions are constant, Spearman must return 0.0, never NaN
        x = [0.85, 0.85, 0.85, 0.85]
        y = [0.70, 0.75, 0.80, 0.85]
        rho = spearman_rho(x, y)
        self.assertEqual(rho, 0.0)

    def test_pearson_constant_series_safe_zero(self):
        x = [0.5, 0.5, 0.5]
        y = [1.0, 2.0, 3.0]
        r = pearson_r(x, y)
        self.assertEqual(r, 0.0)

    def test_within_lambda_srocc_constant_safe(self):
        # Baseline (b) lambda-only produces constant predictions within each lambda
        preds = [0.70, 0.70, 0.85, 0.85]
        targets = [0.65, 0.72, 0.82, 0.89]
        lambdas = [0, 0, 1, 1]

        mean_rho, per_lambda = compute_within_lambda_srocc(preds, targets, lambdas)
        self.assertEqual(per_lambda[0], 0.0)
        self.assertEqual(per_lambda[1], 0.0)
        self.assertEqual(mean_rho, 0.0)

    def test_within_lambda_srocc_varying(self):
        # Perfect ranking within each lambda
        preds = [0.60, 0.70, 0.80, 0.90]
        targets = [10.0, 20.0, 30.0, 40.0]
        lambdas = [0, 0, 1, 1]

        mean_rho, per_lambda = compute_within_lambda_srocc(preds, targets, lambdas)
        self.assertAlmostEqual(per_lambda[0], 1.0)
        self.assertAlmostEqual(per_lambda[1], 1.0)
        self.assertAlmostEqual(mean_rho, 1.0)

    def test_partial_spearman_identity(self):
        # If x and y are strongly correlated and z is independent
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.1, 1.9, 3.2, 3.8, 5.1]
        z = [10.0, 10.0, 10.0, 10.0, 10.0]  # constant control
        part_rho = partial_spearman(x, y, z)
        self.assertAlmostEqual(part_rho, 1.0, places=4)

    def test_bpp_baseline_fit_predict(self):
        model = BppBaseline()
        bpps = [0.1, 0.2, 0.5, 0.8]
        y = [0.6, 0.7, 0.85, 0.95]
        model.fit(bpps, y)
        preds = model.predict([0.15, 0.6])
        self.assertEqual(len(preds), 2)
        self.assertGreater(preds[1], preds[0])

    def test_lambda_baseline_fit_predict(self):
        model = LambdaBaseline()
        lambdas = [0, 0, 1, 1]
        y = [0.5, 0.6, 0.8, 0.9]
        model.fit(lambdas, y)
        preds = model.predict([0, 1])
        self.assertAlmostEqual(preds[0], 0.55)
        self.assertAlmostEqual(preds[1], 0.85)

    def test_latent_stats_baseline(self):
        model = LatentStatsBaseline()
        feats = [[0.1, 1.0], [0.2, 2.0]]
        y = [0.6, 0.8]
        model.fit(feats, y)
        preds = model.predict([[0.15, 1.5]])
        self.assertEqual(len(preds), 1)


if __name__ == "__main__":
    unittest.main()
