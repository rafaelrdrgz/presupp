"""Pointing to the sentences a position rests on, and checking that against annotated evidence."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from conftest import LABELS

from presupp.evidence import MAX_MARKED, check, evidence_of, point
from presupp.labels import IPC


def test_point_marks_every_sentence_that_would_be_flagged_alone_strongest_first():
    assert point(np.array([0.1, 0.8, 0.6, 0.3]), 0.5) == [1, 2]
    assert point(np.array([0.9, 0.8, 0.7, 0.6]), 0.5) == [0, 1, 2][:MAX_MARKED]


def test_point_falls_back_to_the_strongest_and_skips_very_short_sentences():
    assert point(np.array([0.1, 0.4, 0.3]), 0.5) == [1]
    assert point(np.array([0.9, 0.2]), 0.5, words=[2, 10]) == [1]  # "It didn't." isn't scored
    assert point(np.array([0.9]), 0.5, words=[2]) == []


def _label(recorded, used, reading):
    return SimpleNamespace(evidence=recorded, evidence_0based=used, evidence_reading=reading)


def test_evidence_comes_from_the_data_build():
    assert evidence_of(_label([1, 3], [0, 2], "1-based"), 4) == {0, 2}
    assert evidence_of(_label([1, 3], [0, 2], "1-based"), 4, as_recorded=True) == {1, 3}
    assert evidence_of(_label([2, 6], [1, 4], "realigned: NLTK"), 5) == {1, 4}
    assert evidence_of(_label([2, 6], [1, 4], "realigned: NLTK"), 5, as_recorded=True) == {1, 4}
    assert evidence_of(_label([5], None, "doesn't fit"), 4) is None
    assert evidence_of(_label([], None, "none"), 4) is None
    assert evidence_of(_label([1, 3], [0, 2], "1-based"), 2) is None  # a shorter passage


class _Keyword:
    """Scores deontology high for any text that mentions a right."""
    labels = LABELS
    thresholds = np.full(len(LABELS), 0.5)
    release = {"ipc": IPC}

    def probabilities(self, texts, max_length=None):
        j = LABELS.index("deontology")
        out = np.full((len(texts), len(LABELS)), 0.05)
        out[:, j] = [0.9 if "right" in t else 0.1 for t in texts]
        return out


def test_check_compares_the_first_marked_sentence_with_the_evidence():
    passages = pd.DataFrame({
        "passage_id": ["P1", "P2"], "group_id": ["a", "b"],
        "sentences": [["The weather was fine today.", "Everyone has a right to speak freely.",
                       "The meeting ended at noon."],
                      ["People have a right to know.", "Prices rose again this year.",
                       "The rain kept falling all day."]]})
    gold = pd.DataFrame({
        "passage_id": ["P1", "P2", "P2", "P1"],
        "category": ["deontology", "deontology", IPC, "empiricism"],
        "evidence": [[2], [2], [], [7]],
        "evidence_0based": [[1], [1], None, None],  # P1's empiricism list doesn't fit: left out
        "evidence_reading": ["1-based", "1-based", "none", "doesn't fit"]})
    result = check(_Keyword(), passages, gold, n_boot=10)
    assert result["n_cases"] == 2 and result["n_passages"] == 2
    assert result["first_hit"]["mean"] == pytest.approx(0.5)  # P1 right, P2 wrong
    assert result["chance"]["mean"] == pytest.approx(1 / 3)
    assert result["precision"] == pytest.approx(0.5) and result["recall"] == pytest.approx(0.5)
    assert [r["category"] for r in result["by_category"]] == ["deontology"]
