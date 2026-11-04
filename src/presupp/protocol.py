"""Which passages train, validate and test in each round.

Round i tests on dev fold i, validates on fold i+1 mod n and trains on the rest; the lockbox is
never in a round. passage_cv re-deals folds ignoring articles, as v1 did.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from presupp.data.splits import assign_folds
from presupp.labels import target_matrix

PASSAGE_CV_SEED = 42


@dataclass(frozen=True)
class Round:
    index: int
    train_folds: tuple[int, ...]
    val_fold: int
    test_fold: int


def cv_rounds(folds: list[int]) -> list[Round]:
    folds = sorted(folds)
    n = len(folds)
    if n < 3:
        raise ValueError("Need at least 3 folds for train/val/test rounds")
    rounds = []
    for i, test in enumerate(folds):
        val = folds[(i + 1) % n]
        train = tuple(f for f in folds if f not in (test, val))
        rounds.append(Round(index=i, train_folds=train, val_fold=val, test_fold=test))
    return rounds


def labelled_dev(passages: pd.DataFrame, gold: pd.DataFrame, labels: list[str], split: str):
    """Labelled dev passages with the fold column this split uses."""
    data = passages[passages["is_labelled"]].sort_values("passage_id", ignore_index=True)
    if split == "passage_cv":
        hard = target_matrix(data["passage_id"], gold, labels, "hard")
        folds = sorted(data["fold"].unique())
        assignment = assign_folds(
            data["passage_id"].to_numpy(), hard, np.ones(len(data), bool),
            n_folds=len(folds), seed=PASSAGE_CV_SEED,
        )
        data = data.assign(fold=np.asarray(folds)[assignment.fold])
    elif split != "grouped_cv":
        raise ValueError(f"unknown split {split!r}")
    return data


def round_parts(data: pd.DataFrame, rnd: Round) -> dict[str, pd.DataFrame]:
    return {
        "train": data[data["fold"].isin(rnd.train_folds)],
        "val": data[data["fold"] == rnd.val_fold],
        "test": data[data["fold"] == rnd.test_fold],
    }


def final_parts(dev: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train on every dev fold but the highest, which picks the epoch and thresholds.
    Used for both the lockbox run and the released model."""
    dev = dev[dev["is_labelled"]]
    val_fold = int(dev["fold"].max())
    return {"train": dev[dev["fold"] != val_fold], "val": dev[dev["fold"] == val_fold]}
