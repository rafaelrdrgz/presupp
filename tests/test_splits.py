import numpy as np
import pandas as pd
import pytest

from presupp.data.splits import (
    assign_folds,
    check_no_leakage,
    choose_lockbox_fold,
    random_group_folds,
    stratification_error,
)


def synthetic(n_groups=300, n_labels=8, seed=0):
    """Groups of 1-6 passages; labels with very different base rates; 20% unlabelled."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 7, n_groups)
    groups = np.repeat([f"g{i}" for i in range(n_groups)], sizes)
    rates = np.geomspace(0.5, 0.01, n_labels)
    labels = (rng.random((len(groups), n_labels)) < rates).astype(int)
    labelled = rng.random(len(groups)) > 0.2
    return groups, labels, labelled


def test_every_group_lands_in_exactly_one_fold():
    groups, labels, labelled = synthetic()
    fold = assign_folds(groups, labels, labelled, n_folds=5).fold
    frame = pd.DataFrame({"group": groups, "fold": fold})
    assert check_no_leakage(frame, "fold", ["group"]) == {}
    assert set(fold) == set(range(5))


def test_deterministic_for_a_seed():
    groups, labels, labelled = synthetic()
    a = assign_folds(groups, labels, labelled, n_folds=5, seed=1).fold
    b = assign_folds(groups, labels, labelled, n_folds=5, seed=1).fold
    c = assign_folds(groups, labels, labelled, n_folds=5, seed=2).fold
    assert (a == b).all() and (a != c).any()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_beats_random_group_folds(seed):
    groups, labels, labelled = synthetic(seed=seed)
    ours = stratification_error(
        assign_folds(groups, labels, labelled, n_folds=5, seed=seed).fold, labels, labelled, 5
    )
    rand = np.mean([
        stratification_error(random_group_folds(groups, 5, s), labels, labelled, 5)
        for s in range(10)
    ])
    assert ours < 0.6 * rand


def test_spreads_a_rare_label_evenly_when_possible():
    # ten singleton groups carry the only positives of label 1: one should go to each fold
    groups = np.array([f"g{i}" for i in range(50)])
    labels = np.zeros((50, 2), dtype=int)
    labels[:, 0] = 1
    labels[:10, 1] = 1
    fold = assign_folds(groups, labels, np.ones(50, bool), n_folds=10).fold
    assert sorted(fold[:10]) == list(range(10))
    assert np.bincount(fold, minlength=10).tolist() == [5] * 10


def test_unlabelled_only_groups_are_assigned_and_balance_sizes():
    groups, labels, _ = synthetic()
    labelled = np.zeros(len(groups), bool)
    labelled[:100] = True
    fold = assign_folds(groups, labels, labelled, n_folds=4).fold
    sizes = np.bincount(fold, minlength=4)
    assert (fold >= 0).all() and sizes.max() - sizes.min() <= 6


def test_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        assign_folds(np.array(["a", "b"]), np.zeros((3, 2)), np.ones(2, bool))


def _rows(n, n_label0, n_label1):
    m = np.zeros((n, 2), dtype=int)
    m[:n_label0, 0] = 1
    m[:n_label1, 1] = 1
    return m


def test_lockbox_rule_picks_the_most_representative_fold():
    labels = np.vstack([_rows(4, 4, 1), _rows(4, 3, 4), _rows(4, 4, 2)])
    fold = np.repeat([0, 1, 2], 4)
    assert choose_lockbox_fold(fold, labels, np.ones(12, bool), 3) == 2


def test_lockbox_rule_never_picks_a_fold_missing_a_category():
    # fold 0 is closest to the overall mix but has no positives of label 1
    labels = np.vstack([_rows(10, 6, 0), _rows(10, 2, 1), _rows(10, 9, 5)])
    fold = np.repeat([0, 1, 2], 10)
    assert choose_lockbox_fold(fold, labels, np.ones(30, bool), 3) == 1
