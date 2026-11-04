"""Group-aware, multi-label stratified fold assignment.

Iterative stratification (Sechidis et al., 2011) over whole split groups.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldAssignment:
    fold: np.ndarray  # per passage
    n_folds: int
    seed: int


def _pick(scores: list[np.ndarray], rng: np.random.Generator) -> int:
    """argmax of scores[0], ties broken by scores[1], ..., then at random."""
    candidates = np.arange(len(scores[0]))
    for s in scores:
        best = s[candidates].max()
        candidates = candidates[np.isclose(s[candidates], best)]
    return int(rng.choice(candidates))


def assign_folds(
    group_ids: np.ndarray,
    labels: np.ndarray,
    labelled: np.ndarray,
    n_folds: int = 10,
    seed: int = 42,
) -> FoldAssignment:
    """Fold per passage. labels is (N, L) 0/1; rows of unlabelled passages are ignored."""
    labels = np.asarray(labels, dtype=float)
    labelled = np.asarray(labelled, dtype=bool)
    if labels.ndim != 2 or len(labels) != len(group_ids) or len(labelled) != len(group_ids):
        raise ValueError("group_ids, labels and labelled must describe the same passages")
    rng = np.random.default_rng(seed)

    groups, inverse = np.unique(np.asarray(group_ids), return_inverse=True)
    n_groups = len(groups)
    counts = np.zeros((n_groups, labels.shape[1]))
    np.add.at(counts, inverse, labels * labelled[:, None])
    n_labelled = np.bincount(inverse, weights=labelled.astype(float), minlength=n_groups)
    n_total = np.bincount(inverse, minlength=n_groups).astype(float)

    fold_of = np.full(n_groups, -1)
    want_label = np.tile(counts.sum(0) / n_folds, (n_folds, 1))
    want_size = np.full(n_folds, n_labelled.sum() / n_folds)
    remaining = counts.sum(0)
    unassigned = n_labelled > 0
    priority = rng.permutation(n_groups)  # tie-break order

    while unassigned.any():
        open_labels = np.flatnonzero(remaining > 0.5)
        if len(open_labels) == 0:
            break
        label = open_labels[np.argmin(remaining[open_labels])]
        candidates = np.flatnonzero(unassigned & (counts[:, label] > 0))
        group = candidates[np.lexsort((priority[candidates], -counts[candidates, label]))[0]]
        fold = _pick([want_label[:, label], want_size], rng)
        fold_of[group] = fold
        unassigned[group] = False
        want_label[fold] -= counts[group]
        want_size[fold] -= n_labelled[group]
        remaining -= counts[group]

    # labelled groups with no positives (all-zero label rows)
    for group in np.flatnonzero(unassigned):
        fold = _pick([want_size], rng)
        fold_of[group] = fold
        want_size[fold] -= n_labelled[group]

    # unlabelled groups: balance passage counts, biggest first
    totals = np.bincount(fold_of[fold_of >= 0], weights=n_total[fold_of >= 0], minlength=n_folds)
    for group in sorted(np.flatnonzero(fold_of < 0), key=lambda g: (-n_total[g], priority[g])):
        fold = _pick([-totals], rng)
        fold_of[group] = fold
        totals[fold] += n_total[group]

    return FoldAssignment(fold=fold_of[inverse], n_folds=n_folds, seed=seed)


def choose_lockbox_fold(
    fold: np.ndarray, labels: np.ndarray, labelled: np.ndarray, n_folds: int
) -> int:
    """Fold whose label mix is closest to the whole set's, preferring folds with every category."""
    labels = np.asarray(labels, dtype=float)[labelled]
    fold = np.asarray(fold)[labelled]
    present = labels.sum(0) > 0
    overall = labels[:, present].mean(0)
    best = None
    for k in range(n_folds):
        in_fold = labels[fold == k][:, present]
        if len(in_fold) == 0:
            continue
        deviation = float((np.abs(in_fold.mean(0) - overall) / overall).mean())
        complete = bool((in_fold.sum(0) > 0).all())
        key = (not complete, deviation, k)
        if best is None or key < best:
            best = key
    if best is None:
        raise ValueError("No fold has labelled passages")
    return best[2]


def random_group_folds(group_ids: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Baseline: whole groups dealt to folds at random."""
    rng = np.random.default_rng(seed)
    groups, inverse = np.unique(np.asarray(group_ids), return_inverse=True)
    order = rng.permutation(len(groups))
    fold_of = np.empty(len(groups), dtype=int)
    fold_of[order] = np.arange(len(groups)) % n_folds
    return fold_of[inverse]


def fold_label_counts(
    fold: np.ndarray, labels: np.ndarray, labelled: np.ndarray, label_names: list[str], n_folds: int
) -> pd.DataFrame:
    """One row per fold: n_total, n_labelled, then positives per label."""
    rows = []
    for k in range(n_folds):
        in_fold = fold == k
        lab = in_fold & labelled
        row = {"fold": k, "n_total": int(in_fold.sum()), "n_labelled": int(lab.sum())}
        row.update(dict(zip(label_names, labels[lab].sum(0).astype(int), strict=True)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("fold")


def stratification_error(
    fold: np.ndarray, labels: np.ndarray, labelled: np.ndarray, n_folds: int
) -> float:
    """Mean relative deviation of per-fold positives from the ideal count; 0 is perfect."""
    labels = np.asarray(labels, dtype=float)[labelled]
    fold = np.asarray(fold)[labelled]
    totals = labels.sum(0)
    keep = totals > 0
    per_fold = np.stack([labels[fold == k].sum(0) for k in range(n_folds)])
    ideal = totals / n_folds
    deviation = np.abs(per_fold - ideal).mean(0)
    return float((deviation[keep] / ideal[keep]).mean())


def check_no_leakage(frame: pd.DataFrame, fold_column: str, key_columns: list[str]) -> dict:
    """{key column: values found in more than one fold}; empty means no leakage."""
    problems = {}
    for key in key_columns:
        n_folds_per_key = frame.groupby(key)[fold_column].nunique()
        leaked = n_folds_per_key[n_folds_per_key > 1]
        if len(leaked):
            problems[key] = leaked.index.tolist()
    return problems
