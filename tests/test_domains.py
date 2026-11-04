"""Domain-level scores: categories rolled up into their domains."""

import numpy as np
import pytest
from conftest import LABELS

from presupp.domains import domain_columns, score_domains, to_domains
from presupp.labels import IPC
from presupp.taxonomy import load_taxonomy

COLUMNS = domain_columns(LABELS, load_taxonomy("v1"))


def _matrix(rows: list[list[str]]) -> np.ndarray:
    m = np.zeros((len(rows), len(LABELS)), dtype=bool)
    for i, row in enumerate(rows):
        m[i, [LABELS.index(label) for label in row]] = True
    return m


def test_domain_columns_follow_the_taxonomy():
    assert list(COLUMNS) == ["Exclusion", "Metaphysical", "Epistemological", "Ethical",
                             "Political-Philosophical"]
    assert [LABELS[j] for j in COLUMNS["Exclusion"]] == [IPC]
    assert sum(len(idx) for idx in COLUMNS.values()) == len(LABELS)
    rolled = to_domains(_matrix([["deontology", "empiricism"], []]), COLUMNS)
    assert rolled.tolist() == [[False, False, True, True, False], [False] * 5]


def test_domain_scores_on_a_hand_worked_example():
    gold = _matrix([["consequentialism"], ["empiricism"], [IPC],
                    ["individualism", "deontology"]])
    predicted = _matrix([["deontology"], [], [IPC], ["individualism", "empiricism"]])
    seeds = {0: (np.arange(4), predicted.astype(float), predicted)}
    result = score_domains(seeds, gold, COLUMNS, groups=np.arange(4), n_boot=20)
    rows = {r["domain"]: r for r in result["domains"]}

    # Ethical: present in passages 0 and 3, predicted in 0 (with the wrong category, which
    # still finds the domain)
    assert (rows["Ethical"]["precision"], rows["Ethical"]["recall"]) == (1.0, 0.5)
    assert rows["Ethical"]["f1"] == pytest.approx(2 / 3)
    assert rows["Epistemological"]["f1"] == 0.0  # missed in 1, wrongly flagged in 3
    assert rows["Political-Philosophical"]["f1"] == 1.0
    assert rows["Exclusion"]["f1"] == 1.0
    assert np.isnan(rows["Metaphysical"]["f1"]) and rows["Metaphysical"]["passages"] == 0
    # the macro average leaves out Metaphysical (no positives) and Exclusion
    assert result["macro_f1"]["mean"] == pytest.approx((0 + 2 / 3 + 1) / 3)
    assert result["exact_match"]["mean"] == 0.5  # passages 0 and 2
    lo, hi = result["exact_match"]["ci95"]
    assert 0 <= lo <= hi <= 1
