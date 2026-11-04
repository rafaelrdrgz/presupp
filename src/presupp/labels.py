"""Model labels, and training targets from the gold annotations."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from presupp.taxonomy import Taxonomy

IPC = "insufficient_philosophical_content"

# min articles with positives for a label to count as measurable
MEASURABLE_MIN_ARTICLES = 20

# v1 soft-label weights by annotator confidence; exclusion tags have none and count 1.0
CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.75, "low": 0.5}


def model_labels(taxonomy: Taxonomy) -> list[str]:
    """12 philosophical categories + IPC. insufficient_context was never used."""
    return taxonomy.philosophical_ids + [IPC]


def target_matrix(
    passage_ids: Sequence[str], gold: pd.DataFrame, labels: Sequence[str], mode: str = "hard"
) -> np.ndarray:
    """(N, L) float32 targets. hard: 1 for any tag; soft: the tag's confidence weight."""
    if mode not in ("hard", "soft"):
        raise ValueError(f"unknown label mode {mode!r}")
    row = {pid: i for i, pid in enumerate(passage_ids)}
    col = {lab: j for j, lab in enumerate(labels)}
    targets = np.zeros((len(passage_ids), len(labels)), dtype=np.float32)
    sub = gold[gold["passage_id"].isin(row) & gold["category"].isin(col)]
    for r in sub.itertuples():
        value = 1.0 if mode == "hard" else CONFIDENCE_WEIGHTS.get(r.confidence, 1.0)
        targets[row[r.passage_id], col[r.category]] = value
    return targets


def confidence_weight_matrix(
    passage_ids: Sequence[str], gold: pd.DataFrame, labels: Sequence[str]
) -> np.ndarray:
    """(N, L) loss weights: a tag's confidence weight, 1.0 everywhere else."""
    row = {pid: i for i, pid in enumerate(passage_ids)}
    col = {lab: j for j, lab in enumerate(labels)}
    weights = np.ones((len(passage_ids), len(labels)), dtype=np.float32)
    sub = gold[gold["passage_id"].isin(row) & gold["category"].isin(col)]
    for r in sub.itertuples():
        weights[row[r.passage_id], col[r.category]] = CONFIDENCE_WEIGHTS.get(r.confidence, 1.0)
    return weights
