"""Evaluation metrics and article-level bootstrap intervals.

Macro scores average only over labels with a positive in the evaluated passages.
"""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def per_label(y: np.ndarray, pred: np.ndarray, probs: np.ndarray | None = None) -> dict:
    y = (np.asarray(y) > 0).astype(int)
    pred = np.asarray(pred).astype(int)
    out = {"precision": [], "recall": [], "f1": [], "support": [], "average_precision": []}
    for j in range(y.shape[1]):
        out["precision"].append(precision_score(y[:, j], pred[:, j], zero_division=0))
        out["recall"].append(recall_score(y[:, j], pred[:, j], zero_division=0))
        out["f1"].append(f1_score(y[:, j], pred[:, j], zero_division=0))
        out["support"].append(int(y[:, j].sum()))
        has_both = 0 < y[:, j].sum() < len(y)
        out["average_precision"].append(
            average_precision_score(y[:, j], probs[:, j]) if probs is not None and has_both
            else np.nan
        )
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def macro_f1(y: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    y = (np.asarray(y) > 0).astype(int)
    present = y.sum(0) > 0
    if mask is not None:
        present &= mask
    scores = [f1_score(y[:, j], pred[:, j], zero_division=0) for j in np.flatnonzero(present)]
    return float(np.mean(scores)) if scores else float("nan")


def micro_f1(y: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    y = (np.asarray(y) > 0).astype(int)
    cols = np.arange(y.shape[1]) if mask is None else np.flatnonzero(mask)
    return float(f1_score(y[:, cols].ravel(), np.asarray(pred)[:, cols].ravel(), zero_division=0))


def macro_average_precision(y: np.ndarray, probs: np.ndarray, mask=None) -> float:
    ap = per_label(y, np.zeros_like(probs), probs)["average_precision"]
    if mask is not None:
        ap = np.where(mask, ap, np.nan)
    return float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan")


def average_precision_by_label(y: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """sklearn's per-label AP (NaN without both classes), fast enough to bootstrap."""
    y = np.asarray(y) > 0
    probs = np.asarray(probs, dtype=float)
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        n_pos = y[:, j].sum()
        if not 0 < n_pos < len(y):
            continue
        order = np.argsort(-probs[:, j], kind="mergesort")
        p, hit = probs[order, j], y[order, j]
        last_of_tie = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
        tp = np.cumsum(hit)[last_of_tie]
        precision = tp / (last_of_tie + 1)
        recall = tp / n_pos
        out[j] = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    return out


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Pooled over labels; equal-width bins weighted by count."""
    y = (np.asarray(y) > 0).astype(float).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean())
    return float(ece)


def reliability_table(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> list[dict]:
    y = (np.asarray(y) > 0).astype(float).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            rows.append({"bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}", "n": int(in_bin.sum()),
                         "mean_predicted": float(p[in_bin].mean()),
                         "observed": float(y[in_bin].mean())})
    return rows


def f1_by_label(y: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(per-label F1, 0 where undefined; mask of labels with positives)."""
    y = np.asarray(y) > 0
    pred = np.asarray(pred).astype(bool)
    tp = (y & pred).sum(0)
    denominator = 2 * tp + (~y & pred).sum(0) + (y & ~pred).sum(0)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(y.shape[1]), where=denominator > 0)
    return f1, y.sum(0) > 0


def bootstrap_samples(groups: np.ndarray, n_boot: int, seed: int = 0):
    """Yield row indices, resampling whole groups with replacement."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    ends = np.append(starts[1:], len(order))
    for _ in range(n_boot):
        chosen = rng.integers(0, len(unique), len(unique))
        yield np.concatenate([order[starts[g]:ends[g]] for g in chosen])


def group_bootstrap(
    statistic: Callable[[np.ndarray], float], groups: np.ndarray,
    n_boot: int = 1000, seed: int = 0, level: float = 0.95,
) -> tuple[float, float]:
    """(lo, hi) for statistic(row_indices), resampling whole groups (articles)."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    values = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(unique), len(unique))
        values.append(statistic(np.concatenate([members[g] for g in chosen])))
    values = np.asarray(values, dtype=float)
    alpha = (1 - level) / 2
    return float(np.nanquantile(values, alpha)), float(np.nanquantile(values, 1 - alpha))
