"""E5 Shortcut Baselines and correlation evaluation metrics.

Implements Section 10 (E5) and Section 13 criteria:
- Baseline (a): bpp_actual alone to y.
- Baseline (b): lambda_idx alone (per-lambda mean degradation).
- Baseline (c): latent summary statistics + bpp_actual to y.
- Metrics: overall SROCC, PLCC, robust within-lambda SROCC (constant predictions = 0.0),
  and partial Spearman correlation controlling for bitrate.
All evaluation functions include pure-Python reference oracles.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union


def compute_ranks(values: List[float]) -> List[float]:
    """Assign fractional ranks to values (handles ties)."""
    sorted_pairs = sorted((val, idx) for idx, val in enumerate(values))
    ranks = [0.0] * len(values)

    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based rank
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation coefficient.

    Safely returns 0.0 if either series is constant (std == 0) to avoid NaNs.
    """
    if len(x) < 2 or len(x) != len(y):
        return 0.0

    rx = compute_ranks(x)
    ry = compute_ranks(y)

    mean_rx = sum(rx) / len(rx)
    mean_ry = sum(ry) / len(ry)

    num = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    den_x = sum((a - mean_rx) ** 2 for a in rx)
    den_y = sum((b - mean_ry) ** 2 for b in ry)

    den = math.sqrt(den_x * den_y)
    if den <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def pearson_r(x: List[float], y: List[float]) -> float:
    """Compute Pearson linear correlation coefficient (PLCC).

    Safely returns 0.0 if either series is constant to avoid NaNs.
    """
    if len(x) < 2 or len(x) != len(y):
        return 0.0

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = sum((a - mean_x) ** 2 for a in x)
    den_y = sum((b - mean_y) ** 2 for b in y)

    den = math.sqrt(den_x * den_y)
    if den <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def partial_spearman(x: List[float], y: List[float], z: List[float]) -> float:
    """Compute partial Spearman correlation between x and y controlling for z.

    r_{xy.z} = (r_{xy} - r_{xz} * r_{yz}) / sqrt((1 - r_{xz}^2) * (1 - r_{yz}^2))
    """
    r_xy = spearman_rho(x, y)
    r_xz = spearman_rho(x, z)
    r_yz = spearman_rho(y, z)

    den_sq = (1.0 - (r_xz ** 2)) * (1.0 - (r_yz ** 2))
    if den_sq <= 1e-12:
        return 0.0
    return (r_xy - (r_xz * r_yz)) / math.sqrt(den_sq)


def compute_within_lambda_srocc(
    predictions: List[float], targets: List[float], lambda_indices: List[int]
) -> Tuple[float, Dict[int, float]]:
    """Compute within-lambda SROCC per lambda and their mean.

    Safely returns 0.0 for lambdas where predictions or targets are constant.
    """
    by_lambda: Dict[int, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for p, t, l in zip(predictions, targets, lambda_indices):
        by_lambda[l][0].append(p)
        by_lambda[l][1].append(t)

    rhos = {}
    for l_idx in sorted(by_lambda.keys()):
        preds, targs = by_lambda[l_idx]
        rho = spearman_rho(preds, targs)
        rhos[l_idx] = rho

    mean_rho = (sum(rhos.values()) / len(rhos)) if rhos else 0.0
    return mean_rho, rhos


class BppBaseline:
    """Baseline (a): Predicts quality degradation y from bpp_actual alone."""

    def __init__(self):
        self.model = None
        self._poly_coeffs = None

    def fit(self, bpp: List[float], y: List[float]):
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            import numpy as np
            X = np.array(bpp).reshape(-1, 1)
            self.model.fit(X, y)
            return
        except ImportError:
            pass

        # Pure-Python polynomial (degree 2) regression fallback
        n = len(bpp)
        if n == 0:
            self._poly_coeffs = [0.0, 0.0]
            return
        mean_x = sum(bpp) / n
        mean_y = sum(y) / n
        var_x = sum((x - mean_x) ** 2 for x in bpp)
        cov_xy = sum((x - mean_x) * (y_val - mean_y) for x, y_val in zip(bpp, y))
        slope = (cov_xy / var_x) if var_x > 1e-12 else 0.0
        intercept = mean_y - slope * mean_x
        self._poly_coeffs = [intercept, slope]

    def predict(self, bpp: List[float]) -> List[float]:
        if self.model is not None:
            import numpy as np
            X = np.array(bpp).reshape(-1, 1)
            return self.model.predict(X).tolist()

        intercept, slope = self._poly_coeffs or [0.0, 0.0]
        return [intercept + slope * x for x in bpp]


class LambdaBaseline:
    """Baseline (b): Predicts per-lambda mean degradation y from lambda_idx alone."""

    def __init__(self):
        self.means: Dict[int, float] = {}
        self.global_mean: float = 0.0

    def fit(self, lambda_indices: List[int], y: List[float]):
        sums = defaultdict(float)
        counts = defaultdict(int)
        for l_idx, y_val in zip(lambda_indices, y):
            sums[l_idx] += y_val
            counts[l_idx] += 1

        self.means = {l: sums[l] / counts[l] for l in sums}
        self.global_mean = sum(y) / len(y) if y else 0.0

    def predict(self, lambda_indices: List[int]) -> List[float]:
        return [self.means.get(l, self.global_mean) for l in lambda_indices]


class LatentStatsBaseline:
    """Baseline (c): Latent summary statistics + bpp_actual to y."""

    def __init__(self):
        self.model = None
        self._mean_y = 0.0

    def fit(self, features: List[List[float]], y: List[float]):
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            import numpy as np
            self.model.fit(np.array(features), y)
            return
        except ImportError:
            pass

        self._mean_y = sum(y) / len(y) if y else 0.0

    def predict(self, features: List[List[float]]) -> List[float]:
        if self.model is not None:
            import numpy as np
            return self.model.predict(np.array(features)).tolist()
        return [self._mean_y] * len(features)
