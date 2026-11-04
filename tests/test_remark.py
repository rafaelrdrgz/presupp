"""The evidence re-marking page: which passages it holds, and that their text is embedded
safely."""

import json
import re

import pandas as pd

from presupp.data.remark import page, to_remark
from presupp.taxonomy import load_taxonomy

TAXONOMY = load_taxonomy("v1")


def _data():
    passages = pd.DataFrame({
        "passage_id": ["P1", "P2", "P3"], "source": ["Alpha", "Beta", "Gamma"],
        "sentences": [["One.", "Two </script><b>bold</b>.", "Three."], ["A.", "B."], ["X."]]})
    gold = pd.DataFrame({
        "passage_id": ["P1", "P1", "P1", "P2", "Q9"],
        "category": ["deontology", "empiricism", "insufficient_philosophical_content",
                     "deontology", "deontology"],
        "confidence": ["low", "high", None, "medium", "medium"],
        "evidence": [[4, 5], [2], [], [1], [9]],
        "evidence_0based": [None, [1], None, [0], None],
        "evidence_reading": ["doesn't fit", "1-based", "none", "1-based", "doesn't fit"]})
    return passages, gold


def test_only_passages_with_evidence_that_doesnt_fit_are_included_with_all_their_positions():
    passages, gold = _data()
    (item,) = to_remark(passages, gold, TAXONOMY)  # P2 fits; Q9 isn't a development passage
    assert item["id"] == "P1" and item["source"] == "Alpha" and len(item["sentences"]) == 3
    deontology, empiricism = item["positions"]  # exclusion labels have no evidence to mark
    assert (deontology["id"], deontology["fits"], deontology["start"]) == ("deontology", False, [])
    assert deontology["recorded"] == [4, 5] and deontology["domain"] == "Ethical"
    assert deontology["confidence"] == "low"
    assert (empiricism["fits"], empiricism["start"], empiricism["recorded"]) == (True, [1], [2])


def test_the_page_embeds_the_passages_without_breaking_out_of_its_script():
    passages, gold = _data()
    html = page(to_remark(passages, gold, TAXONOMY), TAXONOMY, made_on="2026-10-29")
    assert "</script><b>" not in html
    payload = re.search(r'<script type="application/json" id="data">(.*?)</script>', html,
                        flags=re.DOTALL).group(1)
    data = json.loads(payload)
    assert data["passages"][0]["sentences"][1] == "Two </script><b>bold</b>."
    assert data["madeOn"] == "2026-10-29"
    # every philosophical category can be added; the exclusion label can't
    assert len(data["categories"]) == 12
    assert "insufficient_philosophical_content" not in {c["id"] for c in data["categories"]}
