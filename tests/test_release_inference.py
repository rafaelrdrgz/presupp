"""The released model: trained without the lockbox, saved, reloaded and used to classify text."""

import dataclasses
import json
import re
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import LABELS
from test_sentence_model import _tiny as _tiny_v3
from test_training import _tiny_transformer

from presupp import paths, reading
from presupp.decisions import decide
from presupp.evidence import point
from presupp.experiment import ExperimentConfig
from presupp.inference import (
    RELEASE_FILE,
    SENTENCE_HEAD_FILE,
    VAL_FILE,
    VAL_SENTENCES_FILE,
    Classifier,
    describe,
)
from presupp.labels import IPC
from presupp.release import reliability, train_release
from presupp.runs import PROB_PREFIX, probabilities
from presupp.taxonomy import load_taxonomy

TAXONOMY = load_taxonomy("v1")


@pytest.fixture
def release(tiny_data, tiny_model):
    cfg = _tiny_transformer(tiny_model, epochs=1)
    out = tiny_data / "release"
    meta = train_release(cfg, 0, out, tiny_data / "processed", tiny_data / "results",
                         tiny_data / "runs", device="cpu")
    return cfg, out, meta


def _val_texts(tiny_data, out) -> tuple[pd.DataFrame, list[str]]:
    val = pd.read_parquet(out / VAL_FILE)
    passages = pd.read_parquet(tiny_data / "processed" / "passages.parquet")
    return val, passages.set_index("passage_id").loc[val.passage_id, "text"].tolist()


def test_release_is_trained_on_dev_folds_and_validated_on_the_last(release):
    _, out, meta = release
    val = pd.read_parquet(out / VAL_FILE)
    assert set(val.passage_id.str[1]) == {"3"}  # dev folds are 0-3; fold 4 is the lockbox
    assert meta["training"]["sizes"] == {"train": 24, "val": 8}
    assert meta["training"]["val_fold"] == 3
    assert meta["training"]["compared_with_final_run"] is None  # no final run in this test


def test_reloaded_release_gives_the_same_probabilities(tiny_data, release):
    _, out, meta = release
    val, texts = _val_texts(tiny_data, out)
    classifier = Classifier.load(out)
    np.testing.assert_allclose(classifier.probabilities(texts), probabilities(val, LABELS),
                               atol=1e-5)
    probs, sentence_scores = classifier.predict(texts)
    assert sentence_scores is None  # a passage model scores sentences separately
    np.testing.assert_allclose(probs, probabilities(val, LABELS), atol=1e-5)
    saved = json.loads((out / RELEASE_FILE).read_text(encoding="utf-8"))
    assert saved["thresholds"] == meta["thresholds"] and len(saved["thresholds"]) == len(LABELS)
    assert [label["id"] for label in saved["labels"]] == LABELS


def test_classify_makes_the_evaluation_decisions(tiny_data, release):
    cfg, out, meta = release
    val, texts = _val_texts(tiny_data, out)
    expected = decide(probabilities(val, LABELS), np.asarray(meta["thresholds"]),
                      cfg.threshold_policy, cfg.decode, LABELS.index(IPC))
    done = []
    results = Classifier.load(out).classify_texts(texts, lambda n, total, stage: done.append(n))
    assert done[-1] == len(texts)
    for result, row in zip(results, expected, strict=True):
        found = {c.id for d in result.domains for c in d.categories}
        if result.ipc.detected:
            found.add(IPC)
        assert found == {LABELS[j] for j in np.flatnonzero(row)}
        assert len(result.categories) == len(LABELS) - 1


def test_reliability_is_read_from_the_results_reports(tmp_path):
    row = {"label": "deontology", "articles": 9, "measurable": True, "dev_positives": 12,
           "precision": 0.5, "recall": 0.7, "f1": 0.58, "f1_ci_low": 0.4, "f1_ci_high": 0.7,
           "average_precision": 0.6}
    payload = {"name": "x", "n_passages": 40, "seeds": [0], "headline": {"macro_f1": {}},
               "per_label": [row], "calibration": []}
    (tmp_path / "x.json").write_text(json.dumps(payload), encoding="utf-8")
    found = reliability("x", tmp_path)
    assert set(found) == {"development"}  # no lockbox or domain report for "x"
    assert found["development"]["per_label"] == [{k: v for k, v in row.items()
                                                  if k != "average_precision"}]
    stat = {"mean": 0.6, "sd": 0.01, "ci95": [0.55, 0.65]}
    fold = {"n_passages": 9, "n_cases": 12, "first_hit": stat, "chance": stat, "lift": stat,
            "precision": 0.4, "recall": 0.5}
    (tmp_path / "evidence_x.json").write_text(json.dumps(fold), encoding="utf-8")
    assert reliability("x", tmp_path)["evidence"]["over"] == "validation fold"
    cv = {**fold, "n_cases": 900, "precision": stat, "recall": stat}
    (tmp_path / "evidence_cv_x.json").write_text(json.dumps(cv), encoding="utf-8")
    evidence = reliability("x", tmp_path)["evidence"]  # cross-validation, where it was run
    assert (evidence["over"], evidence["n_cases"], evidence["precision"]) == \
        ("cross-validation", 900, 0.6)


# ---- a v3 release: the sentence model ----

@pytest.fixture
def v3_release(tiny_data, tiny_model):
    cfg = _tiny_v3(tiny_model, name="tiny_mil", passage_from="sentences", epochs=1)
    out = tiny_data / "release_v3"
    meta = train_release(cfg, 0, out, tiny_data / "processed", tiny_data / "results",
                         tiny_data / "runs", device="cpu")
    return cfg, out, meta


def _val_passages(tiny_data, out) -> tuple[pd.DataFrame, list[str], list[list[str]]]:
    val = pd.read_parquet(out / VAL_FILE)
    passages = pd.read_parquet(tiny_data / "processed" / "passages.parquet")
    passages = passages.set_index("passage_id").loc[val.passage_id]
    return val, passages.text.tolist(), [list(s) for s in passages.sentences]


def test_a_v3_release_reloads_with_its_sentence_head(tiny_data, v3_release):
    _, out, meta = v3_release
    assert meta["format"] == 2 and (out / SENTENCE_HEAD_FILE).exists()
    gold = pd.read_parquet(tiny_data / "processed" / "gold_labels.parquet")
    in_val = gold.passage_id.str.startswith("P3") & gold.evidence_0based.notna()
    assert meta["marking"]["cases"] == in_val.sum()
    assert len(meta["marking_thresholds"]) == len(LABELS) - 1
    val, texts, split = _val_passages(tiny_data, out)
    classifier = Classifier.load(out)
    assert classifier.reads_sentences
    probs, scores = classifier.predict(texts, split)
    np.testing.assert_allclose(probs, probabilities(val, LABELS), atol=1e-5)
    saved = pd.read_parquet(out / VAL_SENTENCES_FILE)
    np.testing.assert_allclose(np.concatenate(scores),
                               saved[[PROB_PREFIX + p for p in LABELS[:-1]]], atol=1e-5)
    # each position's passage score is its best sentence's
    np.testing.assert_allclose(probs[:, :-1], np.stack([s.max(0) for s in scores]), atol=1e-6)


def _everything_flagged(classifier: Classifier) -> Classifier:
    """Every position at a mark the tiny model always reaches, and IPC never."""
    classifier.thresholds = np.array([0.01] * (len(LABELS) - 1) + [1.0])
    return classifier


def test_a_v3_release_marks_the_sentences_it_scores_highest(tiny_data, v3_release):
    _, out, meta = v3_release
    _, texts, split = _val_passages(tiny_data, out)
    classifier = _everything_flagged(Classifier.load(out))
    _, scores = classifier.predict(texts, split)
    results = classifier.classify_texts(texts, split=split)
    for result, sentences, s in zip(results, split, scores, strict=True):
        assert result.outcome == "domains" and result.sentences == tuple(sentences)
        flagged = [c for d in result.domains for c in d.categories]
        assert len(flagged) == len(LABELS) - 1
        for c in flagged:
            j = LABELS.index(c.id)
            assert c.evidence[0] == int(np.argmax(s[:, j]))  # the sentence that flagged it
            assert list(c.evidence) == point(s[:, j], meta["marking_thresholds"][j], [4, 4, 4])


def test_a_v3_release_never_marks_a_sentence_it_did_not_reach(v3_release, display):
    _, out, _ = v3_release
    classifier = _everything_flagged(Classifier.load(out))  # reads 16 tokens of a passage
    read, (result,) = classifier.classify(_prose("rights"))
    assert len(result.sentences) == 8
    assert {c.evidence for d in result.domains for c in d.categories} == {(0,)}
    page = display.results_html(read, [result])
    assert page.count('class="pf-s pf-u-') == 1


def test_the_about_text_uses_the_release_numbers(v3_release, display):
    _, _, meta = v3_release
    assert "fine-tuned on 24 passages" in display.how_it_reads_markdown(meta)
    assert "about a thousand" in display.how_it_reads_markdown({})
    row = {"label": "deontology", "articles": 9, "measurable": True, "dev_positives": 12,
           "precision": 0.5, "recall": 0.7, "f1": 0.58, "f1_ci_low": 0.4, "f1_ci_high": 0.7}
    rel = {"development": {"n_passages": 1114, "seeds": [0, 1, 2], "per_label": [row],
                           "headline": {"macro_f1": {"mean": 0.473}}},
           "lockbox": {"n_passages": 125, "headline": {"macro_f1": {"mean": 0.437}}}}
    text = display.reliability_markdown({**meta, "reliability": rel})
    assert "the 1,114 labelled passages" in text and "averages over 3 repeats" in text
    assert "| Deontology | Ethical | 0.58 |" in text
    assert "Average F1 over all labels: 0.47." in text
    assert "On 125 passages kept aside until the end: 0.44." in text


# ---- the domain roll-up, with a stub model ----

class _WhitespaceTokenizer:
    def __call__(self, texts, verbose=True):
        return {"input_ids": [t.split() for t in texts]}


def _stub(monkeypatch, probs: dict[str, float], threshold: float = 0.5) -> Classifier:
    torch = pytest.importorskip("torch")
    release = {
        "config": ExperimentConfig(name="stub", description="", max_length=6).to_dict(),
        "labels": [{"id": label, "name": TAXONOMY.get(label).name,
                    "domain": TAXONOMY.domain_of(label), "description": ""} for label in LABELS],
        "thresholds": [threshold] * len(LABELS), "ipc": IPC,
        "reliability": {"development": {"per_label": [{"label": "deontology", "f1": 0.61}]}},
    }
    classifier = Classifier(torch.nn.Identity(), _WhitespaceTokenizer(), release)
    row = np.array([probs.get(label, 0.05) for label in LABELS])
    monkeypatch.setattr(classifier, "probabilities",
                        lambda texts, max_length=None: np.tile(row, (len(texts), 1)))
    return classifier


def test_categories_are_grouped_into_domains_strongest_first(monkeypatch):
    classifier = _stub(monkeypatch, {"consequentialism": 0.9, "deontology": 0.7,
                                     "empiricism": 0.6, IPC: 0.2})
    (p,) = classifier.classify_texts(["one two three"])
    assert p.outcome == "domains" and not p.truncated
    assert [d.name for d in p.domains] == ["Ethical", "Epistemological"]
    assert [c.id for c in p.domains[0].categories] == ["consequentialism", "deontology"]
    assert p.domains[0].strength == pytest.approx(0.9)
    assert p.domains[0].categories[1].dev_f1 == 0.61
    assert p.domains[0].categories[0].dev_f1 is None
    assert describe(p)[0] == "Ethical (0.90)"


def test_ipc_wins_when_it_clears_its_threshold_by_more(monkeypatch):
    classifier = _stub(monkeypatch, {"consequentialism": 0.55, IPC: 0.9})
    (p,) = classifier.classify_texts(["one two"])
    assert p.outcome == "no_philosophy" and p.domains == () and p.ipc.detected


def test_nothing_cleared_shows_the_nearest_categories(monkeypatch):
    classifier = _stub(monkeypatch, {"deontology": 0.45, "virtue_ethics": 0.3,
                                     "individualism": 0.4, IPC: 0.1})
    (p,) = classifier.classify_texts(["one two seven eight nine ten eleven"])
    assert p.outcome == "unclear" and p.truncated  # 7 tokens, max_length 6
    assert [c.id for c in p.closest] == ["deontology", "individualism", "virtue_ethics"]
    assert describe(p)[0].startswith("Nothing cleared its threshold")


# ---- the demo's HTML ----

@pytest.fixture
def display():
    sys.path.insert(0, str(paths.ROOT / "apps" / "demo"))
    try:
        import display
        yield display
    finally:
        sys.path.pop(0)


def _by_keyword(monkeypatch) -> Classifier:
    """A stub that flags deontology in passages about rights, consequentialism in passages about
    outcomes, and nothing philosophical otherwise."""
    classifier = _stub(monkeypatch, {})

    def row(text: str) -> np.ndarray:
        found = {"deontology": 0.9} if "rights" in text else \
            {"consequentialism": 0.8} if "outcomes" in text else {IPC: 0.9}
        return np.array([found.get(label, 0.05) for label in LABELS])

    monkeypatch.setattr(classifier, "probabilities",
                        lambda texts, max_length=None: np.stack([row(t) for t in texts]))
    return classifier


def _prose(topic: str, k: int = 0) -> str:
    """About 190 words: enough to be a passage of its own. k makes it differ from the others
    (a repeated paragraph is read once, and digits don't count)."""
    tag = chr(ord("a") + k % 26) + chr(ord("a") + k // 26)
    return " ".join([f"The report {tag} says the {topic} of the people affected must come "
                     "first, and the council has now accepted that view in full."] * 8)


def test_demo_html_escapes_the_users_text(monkeypatch, display):
    classifier = _by_keyword(monkeypatch)
    text = (_prose("rights") + " <script>alert('x')</script>\n\n"
            "<img src=x onerror=alert(1)>")
    page = display.results_html(*classifier.classify(text))
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "<img" not in page and "&lt;img src=x" in page  # the skipped line, shown escaped


def test_demo_html_reads_the_text_back_passage_by_passage(monkeypatch, display):
    classifier = _by_keyword(monkeypatch)
    text = "\n\n".join(["Opinion", _prose("rights", 1), "Share", _prose("outcomes", 2),
                        _prose("weather", 3)])
    page = display.results_html(*classifier.classify(text))
    assert "Read 3 passages, " in page
    assert page.index("Metaphysical") < page.index("Epistemological") < page.index(">Ethical")
    counts = re.findall(r'class="pf-count">(\d) of 3<', page)
    assert counts == ["0", "0", "2", "0"]  # both stub positions are Ethical
    assert "1 passage showed no philosophical position." in page
    assert page.count("Skipped 1 line: too short to be prose") == 2
    assert page.count('<article class="pf-passage') == 3
    assert page.count('<article class="pf-passage pf-quiet"') == 1
    assert '<span class="ethical"></span>' in page  # the passage's rule
    # every sentence of these passages would be flagged alone, so the first three are underlined
    assert '>Deontology<span class="pf-refs">1, 2, 3</span></li>' in page
    assert '>Consequentialism<span class="pf-refs">1, 2, 3</span></li>' in page
    assert "No philosophical position</span>" in page and "F1 in testing" in page
    assert "0.61" in page  # deontology's development F1, from release.json


def test_the_sentences_a_position_rests_on_are_found_and_underlined(monkeypatch, display):
    classifier = _by_keyword(monkeypatch)
    plain = " ".join(["The council met on Tuesday morning to discuss the plans."] * 9)
    text = (plain + " Every resident has rights that the plan must respect. "
            + plain + "\n\n" + _prose("weather", 3))
    read, results = classifier.classify(text)
    (flagged,) = [r for r in results if r.outcome == "domains"]
    (deontology,) = [c for d in flagged.domains for c in d.categories]
    assert deontology.id == "deontology" and len(deontology.evidence) == 1
    assert "rights" in flagged.sentences[deontology.evidence[0]]
    assert [c.evidence for c in flagged.categories if c.id == "deontology"] == \
        [deontology.evidence]
    page = display.results_html(read, results)
    assert page.count('class="pf-s pf-u-ethical"') == 1
    assert ('data-ref="1" title="Deontology">Every resident has rights that the plan must '
            'respect.</span><sup class="pf-ref">1</sup>') in page
    assert '<li data-refs="1">Deontology<span class="pf-refs">1</span></li>' in page
    assert "Underlined: the sentences each position rests on most." in page


def test_no_sentence_is_underlined_when_the_sentences_dont_match(monkeypatch, display):
    classifier = _by_keyword(monkeypatch)
    read, results = classifier.classify(_prose("rights", 1))
    wrong = [dataclasses.replace(r, sentences=r.sentences[:-1]) for r in results]
    page = display.results_html(read, wrong)
    assert "pf-s pf-u" not in page and "<li data-refs=\"\">Deontology</li>" in page


def test_demo_html_explains_why_nothing_was_read(display):
    for text, message in [("", "Paste some text first."),
                          ("https://example.com/story", "That’s a link."),
                          ("Too short.", "There isn’t enough prose here"),
                          ("Die Regierung hat am Montag angekündigt, dass die neuen Regeln ab "
                           "dem nächsten Jahr für alle Schulen gelten werden. " * 4,
                           "This doesn’t look like English.")]:
        assert message in display.results_html(reading.read(text), [])


def test_demo_html_notes_unread_text(monkeypatch, display):
    classifier = _by_keyword(monkeypatch)
    many = "\n\n".join(_prose("outcomes", k) for k in range(reading.MAX_PASSAGES + 2))
    read, results = classifier.classify(many)
    page = display.results_html(read, results)
    assert f"Read {reading.MAX_PASSAGES} passages" in page
    assert "weren’t read" in page
