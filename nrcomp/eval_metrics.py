"""Pure-Python evaluation metrics, bootstrap statistics and prediction ensembling.

No torch / numpy required, so everything here is unit-tested on the dev machine.

Design decisions (design.md Section 10):
  * Headline metric = within-lambda SROCC; overall metrics are appendix-only (rate-confounded).
  * PWRC / DS-AUC: pairwise concordance within a lambda for pairs with |y_i - y_j| >= T.
    A threshold bin with fewer than `min_pairs` pairs is UNDEFINED (None) and excluded from
    the AUC. The previous implementation reported such empty bins as accuracy 1.0, which
    inflated DS-AUC (e.g. a constant-per-lambda predictor "scored" 1.0 at T = 0.20).
  * Bootstrap CIs resample SOURCE IMAGES with replacement (all lambdas of an image move
    together). DS-AUC uses exactly the same threshold grid as the point estimate.
  * 4-parameter logistic calibration is fit on koniq_val only and validated; a collapsed
    mapping falls back to linear/identity and the fit status is reported.
"""

import math
import random
from bisect import bisect_right
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from nrcomp.baselines import (
    compute_within_lambda_srocc,
    partial_spearman,
    pearson_r,
    spearman_rho,
)

PWRC_THRESHOLDS: Tuple[float, ...] = tuple(round(i * 0.01, 4) for i in range(21))  # 0.00 .. 0.20
MIN_PAIRS_PER_BIN = 30  # used by the evaluation pipeline (unit tests use smaller values)
TEACHERS = ("haarpsi", "one_minus_dists")


def _is_none(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


# ==============================================================================
# 1. 4-Parameter Logistic Calibration (ITU-R BT.500 / VQEG)
# ==============================================================================

def logistic_4p(s: Any, beta1: float, beta2: float, beta3: float, beta4: float) -> Any:
    """f(s) = beta2 + (beta1 - beta2) / (1 + exp(-(s - beta3) / |beta4|))."""
    try:
        import numpy as np
        if isinstance(s, np.ndarray):
            b4 = max(abs(float(beta4)), 1e-6)
            z = np.clip(-(s - beta3) / b4, -60.0, 60.0)
            return beta2 + (beta1 - beta2) / (1.0 + np.exp(z))
    except ImportError:
        pass

    b4 = max(abs(float(beta4)), 1e-6)
    if isinstance(s, (list, tuple)):
        return [logistic_4p(x, beta1, beta2, beta3, beta4) for x in s]
    z = max(-60.0, min(60.0, -(s - beta3) / b4))
    return beta2 + (beta1 - beta2) / (1.0 + math.exp(z))


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / n)


class Logistic4PFitter:
    """Fits s -> y on the validation split only, with a validated fallback chain.

    status: 'logistic' | 'linear' | 'identity'. Reported next to every PLCC/RMSE so a
    degenerate calibration can never be mistaken for a real one.
    """

    MIN_STD_RATIO = 0.05  # mapped predictions must keep >= 5% of the target std

    def __init__(self):
        self.params: Optional[Tuple[float, float, float, float]] = None
        self.linear_fallback: Optional[Tuple[float, float]] = None  # (slope, intercept)
        self.status: str = "unfitted"
        self.n_fit: int = 0

    def fit(self, s: List[float], y: List[float]) -> "Logistic4PFitter":
        self.params = None
        self.n_fit = len(s)
        if len(s) < 4 or len(s) != len(y):
            self.linear_fallback = (1.0, 0.0)
            self.status = "identity"
            return self

        n = len(s)
        mean_s = sum(s) / n
        mean_y = sum(y) / n
        var_s = sum((x - mean_s) ** 2 for x in s)
        cov_sy = sum((x - mean_s) * (v - mean_y) for x, v in zip(s, y))
        slope = (cov_sy / var_s) if var_s > 1e-12 else 0.0
        if var_s <= 1e-12 or abs(slope) <= 1e-12:
            self.linear_fallback = (1.0, 0.0)
            self.status = "identity"
        else:
            self.linear_fallback = (slope, mean_y - slope * mean_s)
            self.status = "linear"

        try:
            import numpy as np
            from scipy.optimize import curve_fit
        except ImportError:
            return self

        try:
            s_arr = np.asarray(s, dtype=np.float64)
            y_arr = np.asarray(y, dtype=np.float64)
            p0 = [
                float(np.percentile(y_arr, 95)),
                float(np.percentile(y_arr, 5)),
                float(np.median(s_arr)),
                float(max(np.std(s_arr) * 0.5, 0.05)),
            ]
            popt, _ = curve_fit(logistic_4p, s_arr, y_arr, p0=p0, maxfev=5000)
            mapped = [float(logistic_4p(x, *popt)) for x in s]
            ok = (
                all(math.isfinite(v) for v in mapped)
                and _std(mapped) >= self.MIN_STD_RATIO * max(_std(list(y)), 1e-12)
            )
            if ok:
                self.params = tuple(float(p) for p in popt)  # type: ignore[assignment]
                self.status = "logistic"
        except Exception:  # noqa: BLE001 - any optimizer failure -> keep the linear fallback
            self.params = None
        return self

    def predict(self, s: Union[List[float], float]) -> Union[List[float], float]:
        if isinstance(s, (list, tuple)):
            if self.params is not None:
                return [float(logistic_4p(x, *self.params)) for x in s]
            slope, intercept = self.linear_fallback or (1.0, 0.0)
            return [slope * x + intercept for x in s]
        if self.params is not None:
            return float(logistic_4p(s, *self.params))
        slope, intercept = self.linear_fallback or (1.0, 0.0)
        return slope * s + intercept


# ==============================================================================
# 2. Basic metric helpers
# ==============================================================================

def compute_rmse(preds: List[float], targets: List[float]) -> float:
    """Root mean squared error (0.0 for empty / mismatched input, as before)."""
    if not preds or len(preds) != len(targets):
        return 0.0
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(preds))


def compute_mean_within_image_srocc(
    preds: List[float], targets: List[float], image_ids: List[str]
) -> Tuple[float, Dict[str, float]]:
    """Mean SROCC across lambdas for each source image (appendix metric: ~1 by construction)."""
    by_image: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for p, t, img_id in zip(preds, targets, image_ids):
        by_image[img_id][0].append(p)
        by_image[img_id][1].append(t)
    rhos = {i: spearman_rho(p, t) for i, (p, t) in by_image.items() if len(p) >= 2}
    return ((sum(rhos.values()) / len(rhos)) if rhos else 0.0), rhos


def compute_within_lambda_partial_spearman(
    preds: List[float], targets: List[float], bpps: List[float], lambdas: List[int]
) -> Tuple[float, Dict[int, float]]:
    """Partial Spearman of (s, y) controlling for bpp, computed within each lambda."""
    by_l: Dict[int, Tuple[List[float], List[float], List[float]]] = defaultdict(lambda: ([], [], []))
    for p, y, b, l in zip(preds, targets, bpps, lambdas):
        by_l[int(l)][0].append(float(p))
        by_l[int(l)][1].append(float(y))
        by_l[int(l)][2].append(float(b))
    per_l = {l: (partial_spearman(lp, ly, lb) if len(lp) >= 3 else 0.0) for l, (lp, ly, lb) in by_l.items()}
    return ((sum(per_l.values()) / len(per_l)) if per_l else 0.0), per_l


# ==============================================================================
# 3. PWRC / DS-AUC via a reusable weighted pair table
# ==============================================================================

class PairTable:
    """All distinguishable within-lambda pairs, bucketed by |dy| threshold index.

    Each pair stores the two GROUP ids (source image index), its threshold bucket and its
    concordance (1 concordant, 0.5 prediction tie, 0 discordant). A bootstrap resample is then
    just a weighted sum with weight = count[group_a] * count[group_b], which is exactly
    equivalent to explicit resampling (pairs between copies of one image have dy = 0 and are
    never in the table) but avoids rebuilding records.
    """

    def __init__(self, thresholds: Sequence[float]):
        self.thresholds = tuple(thresholds)
        self.ga: List[int] = []
        self.gb: List[int] = []
        self.bucket: List[int] = []
        self.conc: List[float] = []

    def __len__(self) -> int:
        return len(self.bucket)


def build_pair_table(
    preds: Sequence[float],
    targets: Sequence[float],
    lambdas: Sequence[int],
    group_ids: Optional[Sequence[int]] = None,
    thresholds: Optional[Sequence[float]] = None,
) -> PairTable:
    thr = tuple(thresholds) if thresholds is not None else PWRC_THRESHOLDS
    table = PairTable(thr)
    by_l: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for idx, (p, y, l) in enumerate(zip(preds, targets, lambdas)):
        g = idx if group_ids is None else int(group_ids[idx])
        by_l[int(l)].append((g, float(p), float(y)))

    for samples in by_l.values():
        n = len(samples)
        for i in range(n):
            gi, pi, yi = samples[i]
            for j in range(i + 1, n):
                gj, pj, yj = samples[j]
                dy = yi - yj
                ady = abs(dy)
                if ady <= 1e-12:
                    continue
                ds = pi - pj
                if (ds > 1e-12 and dy > 0) or (ds < -1e-12 and dy < 0):
                    c = 1.0
                elif abs(ds) <= 1e-12:
                    c = 0.5
                else:
                    c = 0.0
                b = bisect_right(thr, ady + 1e-9) - 1
                table.ga.append(gi)
                table.gb.append(gj)
                table.bucket.append(max(b, 0))
                table.conc.append(c)
    return table


def pwrc_from_pair_table(
    table: PairTable,
    weights: Optional[Sequence[float]] = None,
    min_pairs: int = 1,
) -> Dict[str, Any]:
    thr = table.thresholds
    k = len(thr)
    tot_b = [0.0] * k
    conc_b = [0.0] * k
    if weights is None:
        for b, c in zip(table.bucket, table.conc):
            tot_b[b] += 1.0
            conc_b[b] += c
    else:
        for ga, gb, b, c in zip(table.ga, table.gb, table.bucket, table.conc):
            w = weights[ga] * weights[gb]
            if w:
                tot_b[b] += w
                conc_b[b] += w * c

    # |dy| >= T  <=>  bucket index >= index(T): cumulative sums from the top
    tot = [0.0] * k
    conc = [0.0] * k
    run_t = run_c = 0.0
    for idx in range(k - 1, -1, -1):
        run_t += tot_b[idx]
        run_c += conc_b[idx]
        tot[idx] = run_t
        conc[idx] = run_c

    curve: Dict[float, Optional[float]] = {}
    counts: Dict[float, float] = {}
    for idx, t in enumerate(thr):
        counts[t] = tot[idx]
        curve[t] = (conc[idx] / tot[idx]) if tot[idx] >= max(min_pairs, 1e-12) and tot[idx] > 0 else None

    defined = [(t, curve[t]) for t in thr if curve[t] is not None]
    if not defined:
        ds_auc = None
    elif len(defined) == 1:
        ds_auc = defined[0][1]
    else:
        auc = 0.0
        for (t1, a1), (t2, a2) in zip(defined[:-1], defined[1:]):
            auc += 0.5 * (a1 + a2) * (t2 - t1)
        ds_auc = auc / (defined[-1][0] - defined[0][0])

    return {
        "ds_auc": ds_auc,
        "curve": curve,
        "pair_counts": counts,
        "total_pairs": int(round(tot[0])) if k else 0,
        "supported_max_threshold": defined[-1][0] if defined else None,
    }


def compute_pwrc_within_lambda(
    preds: List[float],
    targets: List[float],
    lambdas: List[int],
    thresholds: Optional[Union[List[float], Tuple[float, ...]]] = None,
    min_pairs: int = 1,
) -> Dict[str, Any]:
    """Within-lambda pairwise concordance curve and DS-AUC (ties in s count 0.5)."""
    table = build_pair_table(preds, targets, lambdas, thresholds=thresholds)
    return pwrc_from_pair_table(table, None, min_pairs=min_pairs)


# ==============================================================================
# 4. Clustered bootstrap
# ==============================================================================

def _percentile_ci(values: List[float], confidence_level: float) -> Tuple[Optional[float], Optional[float]]:
    vals = sorted(v for v in values if not _is_none(v))
    n = len(vals)
    if n == 0:
        return (None, None)
    alpha = 1.0 - confidence_level
    lo = max(0, min(n - 1, int(alpha / 2.0 * n)))
    hi = max(0, min(n - 1, int((1.0 - alpha / 2.0) * n) - 1))
    return (vals[lo], vals[hi])


def compute_clustered_bootstrap_cis(
    records: List[Dict[str, Any]],
    pred_key: Optional[str] = None,
    raw_pred_key: str = "raw_pred",
    mapped_pred_key: str = "mapped_pred",
    target_key: str = "target",
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    thresholds: Optional[Sequence[float]] = None,
    min_pairs: int = 1,
    compute_ds_auc_ci: bool = True,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """95% bootstrap CIs resampling SOURCE IMAGES (all lambdas of an image stay together).

    `compute_ds_auc_ci=False` skips the (most expensive) DS-AUC interval.

    SROCC / within-lambda SROCC / DS-AUC use raw predictions (calibration-invariant);
    PLCC and RMSE use the calibrated predictions.
    """
    if not records:
        return {k: (None, None) for k in ("srocc", "plcc", "rmse", "within_lambda_srocc", "ds_auc")}

    raw_k = pred_key if pred_key is not None else raw_pred_key
    mapped_k = pred_key if pred_key is not None else mapped_pred_key
    thr = tuple(thresholds) if thresholds is not None else PWRC_THRESHOLDS

    unique_images = sorted({str(r["image_id"]) for r in records})
    img_index = {img: i for i, img in enumerate(unique_images)}
    m = len(unique_images)
    has_lambdas = any("lambda_idx" in r for r in records)

    raw = [r.get(raw_k, r.get(mapped_k, 0.0)) for r in records]
    mapped = [r.get(mapped_k, r.get(raw_k, 0.0)) for r in records]
    tgt = [r[target_key] for r in records]
    lam = [int(r.get("lambda_idx", 0)) for r in records]
    grp = [img_index[str(r["image_id"])] for r in records]
    by_group: Dict[int, List[int]] = defaultdict(list)
    for idx, g in enumerate(grp):
        by_group[g].append(idx)

    want_ds = has_lambdas and compute_ds_auc_ci
    pair_table = build_pair_table(raw, tgt, lam, group_ids=grp, thresholds=thr) if want_ds else None

    if m < 2:
        wl, _ = compute_within_lambda_srocc(raw, tgt, lam) if has_lambdas else (None, {})
        ds = pwrc_from_pair_table(pair_table, None, min_pairs)["ds_auc"] if pair_table is not None else None
        s_val, p_val, r_val = spearman_rho(raw, tgt), pearson_r(mapped, tgt), compute_rmse(mapped, tgt)
        return {
            "srocc": (s_val, s_val), "plcc": (p_val, p_val), "rmse": (r_val, r_val),
            "within_lambda_srocc": (wl, wl), "ds_auc": (ds, ds),
        }

    rng = random.Random(seed)
    srocc_d: List[float] = []
    plcc_d: List[float] = []
    rmse_d: List[float] = []
    wl_d: List[float] = []
    ds_d: List[Optional[float]] = []

    for _ in range(n_resamples):
        sampled = rng.choices(range(m), k=m)
        idxs = [i for g in sampled for i in by_group[g]]
        b_raw = [raw[i] for i in idxs]
        b_map = [mapped[i] for i in idxs]
        b_tgt = [tgt[i] for i in idxs]
        srocc_d.append(spearman_rho(b_raw, b_tgt))
        plcc_d.append(pearson_r(b_map, b_tgt))
        rmse_d.append(compute_rmse(b_map, b_tgt))
        if has_lambdas:
            b_lam = [lam[i] for i in idxs]
            wl_d.append(compute_within_lambda_srocc(b_raw, b_tgt, b_lam)[0])
            if want_ds:
                weights = Counter(sampled)
                w_list = [weights.get(g, 0) for g in range(m)]
                ds_d.append(pwrc_from_pair_table(pair_table, w_list, min_pairs)["ds_auc"])

    res: Dict[str, Tuple[Optional[float], Optional[float]]] = {
        "srocc": _percentile_ci(srocc_d, confidence_level),
        "plcc": _percentile_ci(plcc_d, confidence_level),
        "rmse": _percentile_ci(rmse_d, confidence_level),
    }
    if has_lambdas:
        res["within_lambda_srocc"] = _percentile_ci(wl_d, confidence_level)
        res["ds_auc"] = (
            _percentile_ci([v for v in ds_d if v is not None], confidence_level) if want_ds else (None, None)
        )
    return res


def compute_paired_difference_test(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    metric_name: str = "within_lambda_srocc",
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    min_pairs: int = 1,
) -> Dict[str, Any]:
    """Paired image-clustered bootstrap test of metric(A) - metric(B) (two-sided p-value)."""
    dict_a = {(str(r["image_id"]), int(r["lambda_idx"])): r for r in records_a}
    dict_b = {(str(r["image_id"]), int(r["lambda_idx"])): r for r in records_b}
    common_keys = sorted(set(dict_a) & set(dict_b))
    empty = {
        "metric_name": metric_name, "val_a": None, "val_b": None, "delta_obs": None,
        "ci": (None, None), "p_value": None, "significant": False, "n_images": 0,
    }
    if not common_keys:
        return empty

    grouped: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for k in common_keys:
        grouped[k[0]].append((dict_a[k], dict_b[k]))
    unique_images = sorted(grouped)
    m = len(unique_images)

    def calc_metric(recs: List[Dict[str, Any]]) -> Optional[float]:
        p = [r["raw_pred"] for r in recs]
        y = [r["target"] for r in recs]
        l = [int(r.get("lambda_idx", 0)) for r in recs]
        if metric_name == "within_lambda_srocc":
            return compute_within_lambda_srocc(p, y, l)[0]
        if metric_name == "srocc":
            return spearman_rho(p, y)
        if metric_name == "ds_auc":
            return compute_pwrc_within_lambda(p, y, l, min_pairs=min_pairs)["ds_auc"]
        if metric_name == "rmse":
            return compute_rmse([r.get("mapped_pred", r["raw_pred"]) for r in recs], y)
        raise ValueError(f"Unsupported metric_name: {metric_name}")

    val_a = calc_metric([dict_a[k] for k in common_keys])
    val_b = calc_metric([dict_b[k] for k in common_keys])
    if val_a is None or val_b is None:
        out = dict(empty)
        out.update({"val_a": val_a, "val_b": val_b, "n_images": m})
        return out
    delta_obs = val_a - val_b

    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(n_resamples):
        boot_a: List[Dict[str, Any]] = []
        boot_b: List[Dict[str, Any]] = []
        for img in rng.choices(unique_images, k=m):
            for pair_a, pair_b in grouped[img]:
                boot_a.append(pair_a)
                boot_b.append(pair_b)
        va, vb = calc_metric(boot_a), calc_metric(boot_b)
        if va is not None and vb is not None:
            deltas.append(va - vb)

    if not deltas:
        out = dict(empty)
        out.update({"val_a": val_a, "val_b": val_b, "delta_obs": delta_obs, "n_images": m})
        return out

    ci_low, ci_high = _percentile_ci(deltas, confidence_level)
    n = len(deltas)
    le = sum(1 for d in deltas if d <= 0.0)
    ge = sum(1 for d in deltas if d >= 0.0)
    p_val = min(1.0, 2.0 * min(le, ge) / n)
    return {
        "metric_name": metric_name, "val_a": val_a, "val_b": val_b, "delta_obs": delta_obs,
        "ci": (ci_low, ci_high), "p_value": p_val,
        "significant": bool(p_val < 0.05 and (ci_low > 0.0 or ci_high < 0.0)),
        "n_images": m,
    }


# ==============================================================================
# 5. MOS independence (E3), teachers (E4), cross-teacher evaluation
# ==============================================================================

def compute_mos_independence(
    preds: List[float],
    targets: List[float],
    orig_mos_list: List[Optional[float]],
    lambda_indices: List[int],
    mae_preds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """E3: Spearman(s, MOS), Spearman(y, MOS), partial Spearman(s, MOS | y), MAE by MOS tertile.

    `preds` should be RAW predictions (rank-based, calibration-free). `mae_preds` (optional)
    are calibrated predictions used only for the label-unit MAE.
    """
    mae_src = mae_preds if mae_preds is not None else preds
    valid = []
    for p, y, mos, l, mp in zip(preds, targets, orig_mos_list, lambda_indices, mae_src):
        if not _is_none(mos):
            valid.append((p, y, float(mos), l, mp))
    if not valid:
        return {"error": "No valid orig_mos data found."}

    v_preds = [x[0] for x in valid]
    v_targets = [x[1] for x in valid]
    v_mos = [x[2] for x in valid]
    v_lambdas = [x[3] for x in valid]
    v_mae = [x[4] for x in valid]

    if len(set(v_mos)) < 2:
        return {"error": "orig_mos is constant; E3 is undefined (check the manifest audit)."}

    per_lambda: Dict[int, Dict[str, float]] = {}
    for l_idx in sorted(set(v_lambdas)):
        lp = [p for p, l in zip(v_preds, v_lambdas) if l == l_idx]
        ly = [y for y, l in zip(v_targets, v_lambdas) if l == l_idx]
        lm = [m for m, l in zip(v_mos, v_lambdas) if l == l_idx]
        per_lambda[l_idx] = {
            "rho_s_mos": spearman_rho(lp, lm),
            "rho_y_mos": spearman_rho(ly, lm),
            "partial_s_mos_given_y": partial_spearman(lp, lm, ly),
        }

    sorted_mos = sorted(v_mos)
    t1 = sorted_mos[len(sorted_mos) // 3]
    t2 = sorted_mos[(2 * len(sorted_mos)) // 3]
    tertiles = {
        "Low (<= 33%)": [(p, y) for p, y, m in zip(v_mae, v_targets, v_mos) if m <= t1],
        "Mid (33% - 66%)": [(p, y) for p, y, m in zip(v_mae, v_targets, v_mos) if t1 < m <= t2],
        "High (> 66%)": [(p, y) for p, y, m in zip(v_mae, v_targets, v_mos) if m > t2],
    }
    tertile_mae: Dict[str, Optional[float]] = {
        name: (sum(abs(p - y) for p, y in pairs) / len(pairs) if pairs else None)
        for name, pairs in tertiles.items()
    }

    return {
        "overall": {
            "rho_s_mos": spearman_rho(v_preds, v_mos),
            "rho_y_mos": spearman_rho(v_targets, v_mos),
            "partial_s_mos_given_y": partial_spearman(v_preds, v_mos, v_targets),
        },
        "per_lambda": per_lambda,
        "tertile_mae": tertile_mae,
        "n_samples": len(valid),
    }


def compute_second_teacher_agreement(
    preds: List[float],
    dists_list: List[Optional[float]],
    lambda_indices: List[int],
) -> Dict[str, Any]:
    """E4 (per model): Spearman between predicted s and (1 - DISTS)."""
    valid = [(p, 1.0 - float(d), l) for p, d, l in zip(preds, dists_list, lambda_indices) if not _is_none(d)]
    if not valid:
        return {"error": "No valid DISTS data found."}
    v_preds = [x[0] for x in valid]
    v_q = [x[1] for x in valid]
    v_l = [x[2] for x in valid]
    mean_within, per_lambda = compute_within_lambda_srocc(v_preds, v_q, v_l)
    return {
        "overall_srocc": spearman_rho(v_preds, v_q),
        "mean_within_lambda_srocc": mean_within,
        "per_lambda_srocc": per_lambda,
    }


def compute_teacher_agreement_matrix(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-correlation of the label teachers (HaarPSI, 1 - DISTS, PSNR) and bpp (E4).

    Uses only the raw manifest labels (never model targets), overall and within-lambda.
    """
    valid = []
    for r in records:
        h, d, p, b = r.get("haarpsi"), r.get("dists"), r.get("psnr"), r.get("bpp_actual")
        if any(_is_none(v) for v in (h, d, p, b)):
            continue
        valid.append({
            "haarpsi": float(h), "one_minus_dists": 1.0 - float(d), "psnr": float(p),
            "bpp": float(b), "lambda_idx": int(r.get("lambda_idx", 0)),
        })

    keys = ["haarpsi", "one_minus_dists", "psnr", "bpp"]
    labels = ["HaarPSI", "1 - DISTS", "PSNR", "bpp"]
    if not valid:
        return {"keys": keys, "labels": labels, "overall": {}, "within_lambda": {}, "sample_count": 0, "image_count": 0}

    lambdas = [v["lambda_idx"] for v in valid]
    overall: Dict[str, Dict[str, float]] = {}
    within: Dict[str, Dict[str, float]] = {}
    for k1 in keys:
        overall[k1], within[k1] = {}, {}
        v1 = [v[k1] for v in valid]
        for k2 in keys:
            v2 = [v[k2] for v in valid]
            overall[k1][k2] = spearman_rho(v1, v2)
            within[k1][k2] = compute_within_lambda_srocc(v1, v2, lambdas)[0]
    return {
        "keys": keys, "labels": labels, "overall": overall, "within_lambda": within,
        "sample_count": len(valid),
    }


def teacher_target(record: Dict[str, Any], teacher: str) -> Optional[float]:
    """Raw label of `record` under a teacher ('haarpsi' or 'one_minus_dists')."""
    if teacher == "haarpsi":
        v = record.get("haarpsi")
        return None if _is_none(v) else float(v)
    if teacher == "one_minus_dists":
        v = record.get("dists")
        return None if _is_none(v) else 1.0 - float(v)
    raise ValueError(f"Unknown teacher {teacher!r}; expected one of {TEACHERS}")


def records_with_teacher_target(records: List[Dict[str, Any]], teacher: str) -> List[Dict[str, Any]]:
    """Copies of `records` whose `target` is replaced by the given teacher's raw label."""
    out = []
    for r in records:
        t = teacher_target(r, teacher)
        if t is None:
            continue
        nr = dict(r)
        nr["target"] = t
        out.append(nr)
    return out


def compute_cross_teacher(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Within-lambda and overall SROCC of a model's raw predictions against EACH teacher."""
    res: Dict[str, Any] = {}
    for teacher in TEACHERS:
        recs = records_with_teacher_target(records, teacher)
        if not recs:
            res[teacher] = None
            continue
        s = [r["raw_pred"] for r in recs]
        y = [r["target"] for r in recs]
        l = [int(r["lambda_idx"]) for r in recs]
        mean_within, per_l = compute_within_lambda_srocc(s, y, l)
        res[teacher] = {
            "within_lambda_srocc": mean_within,
            "per_lambda_srocc": per_l,
            "overall_srocc": spearman_rho(s, y),
            "n_samples": len(recs),
        }
    return res


# ==============================================================================
# 6. Seed ensembling
# ==============================================================================

def ensemble_records(
    per_seed_records: Dict[int, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Average raw predictions across seeds, key = (split, image_id, lambda_idx).

    Only keys present in EVERY seed are kept (so every ensemble member votes on every
    sample); the number dropped is reported.
    """
    seeds = sorted(per_seed_records)
    if not seeds:
        return [], {"n_seeds": 0, "n_samples": 0, "n_dropped": 0}

    def key(r: Dict[str, Any]) -> Tuple[str, str, int]:
        return (str(r.get("split", "")), str(r["image_id"]), int(r["lambda_idx"]))

    index = {s: {key(r): r for r in per_seed_records[s]} for s in seeds}
    common = set(index[seeds[0]])
    union = set(index[seeds[0]])
    for s in seeds[1:]:
        common &= set(index[s])
        union |= set(index[s])

    out: List[Dict[str, Any]] = []
    for r in per_seed_records[seeds[0]]:
        k = key(r)
        if k not in common:
            continue
        new = dict(r)
        preds = [index[s][k]["raw_pred"] for s in seeds]
        new["raw_pred"] = sum(preds) / len(preds)
        new["seed_std"] = _std(preds)
        new.pop("mapped_pred", None)
        out.append(new)
    return out, {"n_seeds": len(seeds), "seeds": seeds, "n_samples": len(out), "n_dropped": len(union) - len(common)}


def compute_sample_mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Sample mean and sample standard deviation (ddof=1); None for empty input."""
    vals = [v for v in values if not _is_none(v)]
    if not vals:
        return None, None
    n = len(vals)
    mean_val = sum(vals) / n
    if n < 2:
        return mean_val, 0.0
    var = sum((x - mean_val) ** 2 for x in vals) / (n - 1)
    return mean_val, math.sqrt(max(0.0, var))
