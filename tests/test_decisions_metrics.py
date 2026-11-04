import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score

from presupp.compare import compare
from presupp.decisions import decide, tune_thresholds
from presupp.labels import target_matrix
from presupp.metrics import (
    average_precision_by_label,
    bootstrap_samples,
    expected_calibration_error,
    f1_by_label,
    macro_f1,
)
from presupp.protocol import cv_rounds, labelled_dev


def test_min_positives_policy_tunes_common_labels_and_shares_one_threshold_for_rare():
    rng = np.random.default_rng(0)
    y = np.zeros((200, 3), int)
    y[:60, 0] = 1  # common: 60 positives
    y[:3, 1] = 1   # rare: 3 positives
    probs = rng.random((200, 3)) * 0.3
    probs[:60, 0] += 0.25  # label 0 separable around 0.3
    probs[:3, 1] += 0.5
    thresholds = tune_thresholds(probs, y, "min_positives", min_positives=10)
    assert 0.25 <= thresholds[0] <= 0.35
    assert thresholds[1] == thresholds[2]  # both fall back to the shared threshold


def test_v1_policy_falls_back_to_half_when_nothing_works():
    y = np.zeros((10, 1), int)
    assert tune_thresholds(np.full((10, 1), 0.9), y, "v1")[0] == 0.5


def test_ipc_exclusive_keeps_whichever_clears_its_threshold_by_more():
    thresholds = np.array([0.5, 0.5, 0.4])  # last label is IPC
    probs = np.array([
        [0.9, 0.1, 0.5],   # category far above, IPC just above -> category wins
        [0.55, 0.1, 0.9],  # IPC far above -> IPC wins
        [0.1, 0.2, 0.9],   # no conflict
    ])
    pred = decide(probs, thresholds, "min_positives", "ipc_exclusive", ipc_index=2)
    assert pred.tolist() == [[True, False, False], [False, False, True], [False, False, True]]
    independent = decide(probs, thresholds, "min_positives", "independent", ipc_index=2)
    assert independent[:2, 2].all() and independent[0, 0]


def test_v1_policy_uses_strict_inequality():
    t = np.array([0.5])
    assert not decide(np.array([[0.5]]), t, "v1", "independent", 0)[0, 0]
    assert decide(np.array([[0.5]]), t, "min_positives", "independent", 0)[0, 0]


def test_vectorised_f1_matches_sklearn_and_macro_skips_labels_without_positives():
    rng = np.random.default_rng(1)
    y = rng.random((100, 4)) < 0.3
    y[:, 3] = False
    pred = rng.random((100, 4)) < 0.3
    f1, present = f1_by_label(y, pred)
    for j in range(3):
        assert f1[j] == pytest.approx(f1_score(y[:, j], pred[:, j]))
    assert present.tolist() == [True, True, True, False]
    assert macro_f1(y, pred) == pytest.approx(f1[:3].mean())


def test_fast_average_precision_matches_sklearn_including_ties():
    rng = np.random.default_rng(3)
    y = rng.random((300, 4)) < 0.2
    y[:, 3] = False
    probs = rng.random((300, 4))
    probs[:, 1] = np.round(probs[:, 1], 1)  # many tied scores
    ap = average_precision_by_label(y, probs)
    for j in range(3):
        assert ap[j] == pytest.approx(average_precision_score(y[:, j], probs[:, j]))
    assert np.isnan(ap[3])


def test_paired_comparison_detects_a_better_model_and_not_a_copy():
    rng = np.random.default_rng(4)
    y = rng.random((400, 3)) < 0.3
    groups = np.repeat(np.arange(100), 4)
    ids = np.arange(400)
    good = np.where(y, 0.8, 0.2) + rng.normal(0, 0.2, y.shape)
    bad = np.where(y, 0.6, 0.4) + rng.normal(0, 0.3, y.shape)
    experiments = {name: {0: (ids, p, p > 0.5)} for name, p in [("good", good), ("bad", bad)]}
    experiments["copy"] = experiments["good"]
    rows = compare(experiments, [("good", "bad"), ("good", "copy")], y, groups,
                   np.ones(3, bool), n_boot=200)
    better, same = rows
    assert (better["ci95"][:, 0] > 0).all() and (better["a_ahead"] == 1).all()
    assert np.allclose(same["difference"], 0) and np.allclose(same["ci95"], 0)


def test_calibration_error_is_small_when_calibrated_and_large_when_overconfident():
    rng = np.random.default_rng(2)
    p = rng.random(20000)
    y = rng.random(20000) < p
    assert expected_calibration_error(y[:, None], p[:, None]) < 0.02
    assert expected_calibration_error(y[:, None], np.clip(p * 3, 0, 1)[:, None]) > 0.2


def test_bootstrap_resamples_whole_groups():
    groups = np.array(["a", "a", "a", "b", "c", "c"])
    for idx in bootstrap_samples(groups, n_boot=20, seed=0):
        drawn = pd.Series(groups[idx]).value_counts()
        expected = {"a": 3, "b": 1, "c": 2}
        assert all(n % expected[g] == 0 for g, n in drawn.items())


def test_cv_rounds_test_every_fold_once_and_never_overlap():
    rounds = cv_rounds([0, 1, 2, 3, 5, 7])
    assert sorted(r.test_fold for r in rounds) == [0, 1, 2, 3, 5, 7]
    for r in rounds:
        assert r.val_fold != r.test_fold
        assert r.val_fold not in r.train_folds and r.test_fold not in r.train_folds
        assert len(r.train_folds) == 4
    assert rounds[-1].val_fold == 0  # wraps around


def test_targets_hard_and_soft():
    gold = pd.DataFrame({"passage_id": ["a", "a", "b"], "category": ["x", "y", "x"],
                         "confidence": ["low", None, "medium"]})
    assert target_matrix(["a", "b"], gold, ["x", "y"], "hard").tolist() == [[1, 1], [1, 0]]
    assert target_matrix(["a", "b"], gold, ["x", "y"], "soft").tolist() == [[0.5, 1.0],
                                                                          [0.75, 0.0]]


def test_passage_cv_splits_articles_that_grouped_cv_keeps_together(tiny_data):
    passages = pd.read_parquet(tiny_data / "processed" / "passages.parquet")
    gold = pd.read_parquet(tiny_data / "processed" / "gold_labels.parquet")
    dev = passages[passages.split == "dev"]
    labels = sorted(gold.category.unique())
    grouped = labelled_dev(dev, gold, labels, "grouped_cv")
    shuffled = labelled_dev(dev, gold, labels, "passage_cv")
    assert grouped.groupby("group_id").fold.nunique().max() == 1
    assert shuffled.groupby("group_id").fold.nunique().max() > 1
    assert set(shuffled.fold) == set(grouped.fold)
