import pandas as pd
import pytest

from presupp.data.adjudication import AdjudicationError, load_duplicate_decisions
from presupp.taxonomy import load_taxonomy

TAX = load_taxonomy("v1")
CORPUS = pd.DataFrame({
    "passage_id": ["A1", "A2", "B1"],
    "text_cluster": ["A1", "A1", "B1"],
    "n_sentences": [4, 4, 6],
})
HEADER = "text_cluster,keep_passage_id,category,confidence,evidence_sentences,decided_on\n"


def _load(tmp_path, rows):
    path = tmp_path / "duplicates_v1.csv"
    path.write_text(HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    return load_duplicate_decisions(path, CORPUS, TAX)


def test_missing_file_means_no_decisions(tmp_path):
    records, labels, keep = load_duplicate_decisions(tmp_path / "none.csv", CORPUS, TAX)
    assert records.empty and labels.empty and keep == {}


def test_valid_decision_is_converted_to_0_based(tmp_path):
    records, labels, keep = _load(tmp_path, [
        "A1,A2,deontology,high,1;4,2026-10-23",
        "A1,A2,individualism,low,,2026-10-23",
    ])
    assert keep == {"A1": "A2"}
    assert records.annotator.tolist() == ["adjudicated"]
    assert labels.evidence.tolist() == [[0, 3], []]


def test_exclusion_decision(tmp_path):
    _, labels, _ = _load(tmp_path, ["A1,A1,insufficient_philosophical_content,,,2026-10-23"])
    assert labels.confidence.tolist() == [None]


@pytest.mark.parametrize("rows, message", [
    (["A1,A2,deontology,high,5,2026-10-23"], "outside S1..S4"),
    (["A1,A2,deontology,high,0,2026-10-23"], "outside S1..S4"),
    (["A1,B1,deontology,high,1,2026-10-23"], "not a passage in this text cluster"),
    (["A1,A1,deontology,high,1,2026-10-23", "A1,A2,empiricism,high,1,2026-10-23"],
     "more than one keep_passage_id"),
    (["A1,A1,deontology,certain,1,2026-10-23"], "confidence"),
    (["A1,A1,insufficient_philosophical_content,high,,2026-10-23"], "no confidence"),
    (["A1,A1,insufficient_philosophical_content,,,2026-10-23",
      "A1,A1,deontology,high,1,2026-10-23"], "exclusion category combined"),
    (["A1,A1,utilitarianism,high,1,2026-10-23"], "unknown category"),
    (["A1,A1,deontology,high,S3,2026-10-23"], "S-numbers"),
])
def test_rejects_invalid_decisions(tmp_path, rows, message):
    with pytest.raises(AdjudicationError, match=message):
        _load(tmp_path, rows)


REVIEW_HEADER = "passage_id,decision,category,confidence,evidence_sentences,decided_on,source\n"


def _review(tmp_path, rows):
    from presupp.data.adjudication import load_label_review
    path = tmp_path / "label_review_v1.csv"
    path.write_text(REVIEW_HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    return load_label_review(path, CORPUS, TAX)


def test_label_review_keep_and_replace(tmp_path):
    records, labels, decisions = _review(tmp_path, [
        "A1,keep,,,,2026-10-23,run1",
        "B1,replace,deontology,medium,2;5,2026-10-23,run1",
        "B1,replace,individualism,low,,2026-10-23,run1",
    ])
    assert decisions == {"A1": "kept", "B1": "replaced"}
    assert records.passage_id.tolist() == ["B1"]
    assert labels.evidence.tolist() == [[1, 4], []]


@pytest.mark.parametrize("rows, message", [
    (["A1,keep,deontology,,,2026-10-23,run1"], "keep decision is one row"),
    (["A1,keep,,,,2026-10-23,run1", "A1,replace,deontology,high,1,2026-10-23,run1"],
     "all keep or all replace"),
    (["ZZ9,keep,,,,2026-10-23,run1"], "not a passage"),
    (["B1,replace,deontology,high,7,2026-10-23,run1"], "outside S1..S6"),
    (["B1,replace,insufficient_philosophical_content,,,2026-10-23,run1",
      "B1,replace,deontology,high,1,2026-10-23,run1"], "exclusion category combined"),
])
def test_label_review_rejects_invalid_rows(tmp_path, rows, message):
    with pytest.raises(AdjudicationError, match=message):
        _review(tmp_path, rows)
