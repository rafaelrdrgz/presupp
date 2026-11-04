"""Probabilities to yes/no decisions: thresholds and the IPC exclusivity rule.

min_positives: per-label thresholds for labels with enough val positives, one shared threshold
for the rest; p >= t. v1 (the dissertation's): grid 0.15..0.85, p > t, 0.5 if no F1 above 0.
"""

import numpy as np
from sklearn.metrics import f1_score

GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
V1_GRID = np.round(np.arange(0.15, 0.851, 0.05), 2)


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, zero_division=0))


def tune_thresholds(
    probs: np.ndarray, targets: np.ndarray, policy: str, min_positives: int = 10
) -> np.ndarray:
    y = (np.asarray(targets) > 0).astype(int)
    n_labels = y.shape[1]
    if policy == "v1":
        thresholds = np.full(n_labels, 0.5)
        for j in range(n_labels):
            scores = [_f1(y[:, j], probs[:, j] > t) for t in V1_GRID]
            if max(scores) > 0:
                thresholds[j] = V1_GRID[int(np.argmax(scores))]
        return thresholds
    if policy != "min_positives":
        raise ValueError(f"unknown threshold policy {policy!r}")

    present = y.sum(0) > 0
    def shared_score(t):
        pred = probs >= t
        return np.mean([_f1(y[:, j], pred[:, j]) for j in np.flatnonzero(present)] or [0.0])
    shared = GRID[int(np.argmax([shared_score(t) for t in GRID]))]
    thresholds = np.full(n_labels, shared)
    for j in np.flatnonzero(y.sum(0) >= min_positives):
        scores = [_f1(y[:, j], probs[:, j] >= t) for t in GRID]
        thresholds[j] = GRID[int(np.argmax(scores))]
    return thresholds


def decide(
    probs: np.ndarray, thresholds: np.ndarray, policy: str, decode: str, ipc_index: int
) -> np.ndarray:
    """ipc_exclusive: IPC or the categories, whichever clears its threshold by the larger ratio."""
    pred = probs > thresholds if policy == "v1" else probs >= thresholds
    if decode == "independent":
        return pred
    if decode != "ipc_exclusive":
        raise ValueError(f"unknown decode rule {decode!r}")
    ratio = probs / np.maximum(thresholds, 1e-9)
    categories = np.delete(np.arange(probs.shape[1]), ipc_index)
    conflict = pred[:, ipc_index] & pred[:, categories].any(1)
    ipc_wins = ratio[:, ipc_index] >= ratio[:, categories].max(1)
    pred = pred.copy()
    rows_ipc = np.flatnonzero(conflict & ipc_wins)
    rows_cat = np.flatnonzero(conflict & ~ipc_wins)
    pred[np.ix_(rows_ipc, categories)] = False
    pred[rows_cat, ipc_index] = False
    return pred
