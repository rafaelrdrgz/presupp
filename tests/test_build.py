"""End-to-end build on a small synthetic copy of the v1 raw exports, plus checks on the real
build outputs when they exist locally (they are gitignored, so CI skips those)."""

import csv
import json

import numpy as np
import pandas as pd
import pytest

from presupp import paths
from presupp.data.adjudication import AdjudicationError
from presupp.data.build import LEAKAGE_KEYS, BuildConfig, BuildError, build
from presupp.data.dataset import load_dev, open_lockbox
from presupp.data.splits import check_no_leakage

VOCAB = [f"{a}{b}" for a in ["lex", "mor", "tav", "quin", "sol", "bru", "kep", "dar"]
         for b in ["ar", "ent", "ious", "ism", "ology", "ance", "ify", "ward"]]
LABELS = [
    {"consequentialism": {"confidence": "high", "evidence": [0]}},
    {"deontology": {"confidence": "medium", "evidence": [1]},
     "individualism": {"confidence": "low", "evidence": [2]}},
    {"empiricism": {"confidence": "high", "evidence": [0, 1]}},
    {"insufficient_philosophical_content": {"confidence": None, "evidence": []}},
]
LEGACY = {"primary": [], "secondary": [], "confidence": "medium",
          "evidence_sentences": [], "category_evidence": {}}


def _text(rng):
    sentences = [" ".join(rng.choice(VOCAB, 12)).capitalize() + "." for _ in range(3)]
    return " ".join(sentences), sentences


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def raw_v1(tmp_path):
    """6 articles x 3 paragraphs, one paragraph re-scraped under another URL, one 2-paragraph
    supplementary article renamed in the classifier export, and one empty legacy record."""
    rng = np.random.default_rng(0)
    raw, ref = tmp_path / "raw", tmp_path / "reference"
    raw.mkdir()
    ref.mkdir()
    _write_csv(ref / "publishers.csv", ["domain", "publisher", "country", "genre"],
               [["a.org", "Alpha", "GB", "news"], ["b.org", "Beta", "US", "think_tank"]])

    main, csv_rows = [], []
    for art in range(6):
        url = f"https://www.{'ab'[art % 2]}.org/story-{art}/"
        for par in range(3):
            pid = f"M{art * 3 + par}"
            text, sents = _text(rng)
            main.append({"id": pid, "text": text, "word_count": 36, "source_key": "alpha",
                         "source_name": "Alpha", "article_id": f"alpha_{art}",
                         "article_title": f"Story {art}", "article_url": url,
                         "paragraph_indices": [par], "score": 4, "priority": "LOW",
                         "score_breakdown": {}, "markers_found": {}})
            csv_rows.append([pid, text, json.dumps(sents), "Alpha", f"Story {art}", "N/A", 36,
                             url, 4, "LOW"])
    # M17 is the same paragraph as M4, scraped again from another article's page
    main[17]["text"] = main[4]["text"]
    csv_rows[17][1:3] = csv_rows[4][1:3]
    for par in range(2):
        text, sents = _text(rng)
        csv_rows.append([f"p_abc123_00{par}", text, json.dumps(sents), "Gamma", "Supp",
                         "2024-01-01", 36, "https://a.org/supp", "", "MEDIUM"])
    with open(raw / "passages_main.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p) + "\n" for p in main)
    header = ["id", "text", "sentences", "source", "article_title", "date", "word_count",
              "article_url", "score", "priority"]
    _write_csv(raw / "passages_all.csv", header, csv_rows)

    labelled = [f"M{i}" for i in range(16)] + ["M17", "p_abc123_000", "p_abc123_001"]
    renamed = {"p_abc123_000": "X1", "p_abc123_001": "X2"}
    by_id = {r[0]: r for r in csv_rows}
    labels = {pid: LABELS[i % len(LABELS)] for i, pid in enumerate(labelled)}
    _write_csv(raw / "classifier_passages.csv", ["id", "text", "sentences", "source", "word_count"],
               [[renamed.get(p, p), by_id[p][1], by_id[p][2], "Alpha", 36] for p in labelled])
    _write_csv(raw / "primary_labels.csv",
               ["passage_id", "explicit_philosophy_flag", "categories"],
               [[p, "FALSE", json.dumps(labels[p])] for p in labelled]
               + [["M16", "FALSE", json.dumps(LEGACY)]])
    _write_csv(raw / "classifier_primary_labels.csv",
               ["id", "explicit_philosophy_flag", "categories"],
               [[renamed.get(p, p), "FALSE", json.dumps(labels[p])] for p in labelled])
    _write_csv(raw / "expert_02.csv",
               ["timestamp", "passage_id", "duration_seconds", "explicit_philosophy_flag",
                "categories", "notes"],
               [["2026-03-09T10:00:00Z", p, "60", "FALSE", json.dumps(labels[p]), ""]
                for p in labelled[:3]])
    return tmp_path


def _config(root, **kw):
    return BuildConfig(raw_dir=root / "raw", reference_dir=root / "reference",
                       adjudication_dir=root / "adjudication",
                       processed_dir=root / "processed", splits_dir=root / "splits",
                       reports_dir=root / "reports", n_folds=3, **kw)


def _relabel(root, passage_id, categories):
    """Change one passage's primary label in both label exports."""
    for name, id_col in [("primary_labels.csv", 0), ("classifier_primary_labels.csv", 0)]:
        path = root / "raw" / name
        rows = list(csv.reader(open(path, encoding="utf-8")))
        for row in rows[1:]:
            if row[id_col] == passage_id:
                row[2] = json.dumps(categories)
        _write_csv(path, rows[0], rows[1:])


def test_synthetic_build(raw_v1):
    result = build(_config(raw_v1))
    c = result.corpus.set_index("passage_id")

    assert len(c) == 20
    # duplicated paragraph joins its two articles into one split group
    assert c.loc["M17", "group_id"] == c.loc["M4", "group_id"] == c.loc["M3", "group_id"]
    assert c.loc["M17", "text_cluster"] == "M17" and c.loc["M4", "text_cluster"] == "M17"
    assert c.group_id.nunique() == 6  # 7 articles, two merged
    # renamed supplementary ids resolve to the real ones; publisher comes from the URL
    assert result.facts["aliases"] == {"X1": "p_abc123_000", "X2": "p_abc123_001"}
    assert c.loc["p_abc123_000", "publisher"] == "Alpha"
    # the legacy record is an incomplete annotation, not a labelled passage; of the two
    # identically labelled copies M4/M17 only one stays in the gold set
    assert not c.loc["M16", "is_labelled"]
    assert c.loc["M17", "is_canonical"] and not c.loc["M4", "is_canonical"]
    assert c.is_labelled.sum() == 18
    assert set(result.gold.passage_id) == set(c.index[c.is_labelled])
    assert set(result.records.annotator) == {"primary", "expert_02"}

    folds = pd.read_csv(raw_v1 / "splits" / "folds_v1.csv")
    assert check_no_leakage(result.corpus, "fold", LEAKAGE_KEYS) == {}
    assert folds.passage_id.tolist() == sorted(folds.passage_id)

    report = (raw_v1 / "reports" / "data_build_v1.md").read_text(encoding="utf-8")
    assert "Leakage check" in report
    assert all(text not in report for text in c.text)


def test_build_fails_if_label_exports_disagree(raw_v1):
    path = raw_v1 / "raw" / "classifier_primary_labels.csv"
    rows = list(csv.reader(open(path, encoding="utf-8")))
    rows[1][2] = json.dumps({"virtue_ethics": {"confidence": "high", "evidence": [0]}})
    _write_csv(path, rows[0], rows[1:])
    with pytest.raises(BuildError, match="disagree"):
        build(_config(raw_v1))


def test_conflicting_duplicates_need_a_decision(raw_v1):
    _relabel(raw_v1, "M17", {"deontology": {"confidence": "high", "evidence": [0]}})
    with pytest.raises(BuildError, match="need a decision"):
        build(_config(raw_v1))


def test_decision_becomes_the_gold_label(raw_v1):
    _relabel(raw_v1, "M17", {"deontology": {"confidence": "high", "evidence": [0]}})
    (raw_v1 / "adjudication").mkdir()
    _write_csv(raw_v1 / "adjudication" / "duplicates_v1.csv",
               ["text_cluster", "keep_passage_id", "category", "confidence",
                "evidence_sentences", "decided_on"],
               [["M17", "M4", "virtue_ethics", "high", "1;2", "2026-10-23"]])
    result = build(_config(raw_v1))
    gold = result.gold.set_index("passage_id")
    assert "M17" not in gold.index
    assert gold.loc["M4", "category"] == "virtue_ethics"
    assert gold.loc["M4", "label_source"] == "adjudicated"
    assert list(gold.loc["M4", "evidence"]) == [0, 1]  # S1;S2 stored 0-based
    # the original annotations are kept untouched alongside the decision
    raw = result.labels[result.labels.passage_id.isin(["M4", "M17"])]
    kept = set(zip(raw.passage_id, raw.annotator, strict=True))
    assert kept >= {("M4", "primary"), ("M17", "primary")}


def test_evidence_is_read_by_how_it_was_saved_and_realigned_from_the_file(raw_v1):
    (raw_v1 / "adjudication").mkdir()
    header = ["passage_id", "category", "recorded", "evidence_sentences", "splitter"]
    path = raw_v1 / "adjudication" / "evidence_realigned_v1.csv"
    _write_csv(path, header, [["M1", "individualism", "2", "3", "a finer split"]])
    result = build(_config(raw_v1))
    labels = result.labels[result.labels.annotator == "primary"].set_index(
        ["passage_id", "category"])
    # M0's [0] can only have been saved 0-based; M1's deontology [1] is S1
    assert labels.loc[("M0", "consequentialism"), "evidence_reading"] == "0-based (contains 0)"
    assert list(labels.loc[("M0", "consequentialism"), "evidence_0based"]) == [0]
    assert labels.loc[("M1", "deontology"), "evidence_reading"] == "1-based"
    assert list(labels.loc[("M1", "deontology"), "evidence_0based"]) == [0]
    assert labels.loc[("M1", "individualism"), "evidence_reading"] == "realigned: a finer split"
    assert list(labels.loc[("M1", "individualism"), "evidence_0based"]) == [2]
    assert list(labels.loc[("M1", "individualism"), "evidence"]) == [2]  # kept as recorded
    assert labels.loc[("M3", "insufficient_philosophical_content"), "evidence_reading"] == "none"
    report = (raw_v1 / "reports" / "data_build_v1.md").read_text(encoding="utf-8")
    assert "realigned: a finer split" in report

    _write_csv(path, header, [["M1", "individualism", "3", "3", "a finer split"]])
    with pytest.raises(AdjudicationError, match="isn't what the label holds"):
        build(_config(raw_v1))


REMARK_HEADER = ["passage_id", "category", "confidence", "evidence_sentences", "status",
                  "decided_on", "note"]


def _remark(raw_v1, rows, header=REMARK_HEADER):
    (raw_v1 / "adjudication").mkdir(exist_ok=True)
    _write_csv(raw_v1 / "adjudication" / "evidence_remarked_v1.csv", header, rows)


def test_a_re_marked_passage_gets_its_rows_as_gold_labels(raw_v1):
    # M1 was labelled deontology (medium, S2) and individualism (low, S3)
    _remark(raw_v1, [
        ["M1", "individualism", "high", "1;3", "original", "2026-10-29", "a note"],
        ["M1", "deontology", "medium", "", "original", "2026-10-29", "a note"],
        ["M1", "virtue_ethics", "medium", "2", "added", "2026-10-29", "a note"]])
    result = build(_config(raw_v1))
    gold = result.gold[result.gold.passage_id == "M1"].set_index("category")
    assert set(gold.index) == {"individualism", "deontology", "virtue_ethics"}
    assert set(gold.label_source) == {"remarked"}
    assert gold.loc["individualism", "confidence"] == "high"
    assert list(gold.loc["individualism", "evidence_0based"]) == [0, 2]
    assert gold.loc["individualism", "evidence_reading"] == "re-marked"
    assert gold.loc["deontology", "evidence_reading"] == "re-marked: no sentence"
    assert gold.loc["deontology", "evidence_0based"] is None
    assert list(gold.loc["virtue_ethics", "evidence_0based"]) == [1]
    # the original annotation is kept alongside
    primary = result.labels[(result.labels.passage_id == "M1")
                            & (result.labels.annotator == "primary")]
    assert set(primary.category) == {"deontology", "individualism"}
    assert result.facts["remarked"] == {"passages": 1, "added": 1, "removed": 0,
                                        "confidence_changed": 1, "evidence_only": 0,
                                        "no_philosophy": 0, "by_claude": 0}
    report = (raw_v1 / "reports" / "data_build_v1.md").read_text(encoding="utf-8")
    assert "Re-marked passages: 1" in report


def test_rows_taken_from_claude_are_recorded_as_claudes(raw_v1):
    _remark(raw_v1, [
        ["M1", "individualism", "medium", "3", "original", "2026-10-29", "", "claude_review/x"],
        ["M1", "deontology", "", "", "removed", "2026-10-29", "", "claude_review/x"]],
        header=REMARK_HEADER + ["source"])
    result = build(_config(raw_v1))
    gold = result.gold[result.gold.passage_id == "M1"]
    assert gold.label_source.tolist() == ["remarked_claude"]
    assert gold.evidence_reading.tolist() == ["re-marked by Claude"]
    records = result.records.set_index("annotator")
    assert records.loc["remarked_claude", "notes"] == "claude_review/x"
    assert result.facts["remarked"]["by_claude"] == 1


def test_a_re_marked_passage_can_turn_out_to_have_no_philosophical_content(raw_v1):
    _remark(raw_v1, [
        ["M1", "deontology", "", "", "removed", "2026-10-29", ""],
        ["M1", "individualism", "", "", "removed", "2026-10-29", ""],
        ["M1", "insufficient_philosophical_content", "", "", "added", "2026-10-29", ""]])
    result = build(_config(raw_v1))
    gold = result.gold[result.gold.passage_id == "M1"]
    assert gold.category.tolist() == ["insufficient_philosophical_content"]
    assert gold.confidence.isna().all() and gold.evidence_reading.tolist() == ["none"]
    assert result.facts["remarked"]["no_philosophy"] == 1


def test_removing_a_position_and_files_from_the_first_page_version(raw_v1):
    _remark(raw_v1, [["M1", "individualism", "", "", "removed", "2026-10-29", ""],
                     ["M1", "deontology", "low", "2", "original", "2026-10-29", ""]])
    gold = build(_config(raw_v1)).gold
    assert gold.loc[gold.passage_id == "M1", "category"].tolist() == ["deontology"]
    # the first version wrote no confidence or status: original labels, original confidence
    _remark(raw_v1, [["M1", "individualism", "1", "2026-10-29", ""],
                     ["M1", "deontology", "2", "2026-10-29", ""]],
            header=["passage_id", "category", "evidence_sentences", "decided_on", "note"])
    gold = build(_config(raw_v1)).gold.set_index(["passage_id", "category"])
    assert gold.loc[("M1", "individualism"), "confidence"] == "low"
    assert list(gold.loc[("M1", "deontology"), "evidence_0based"]) == [1]


D = "2026-10-29"


@pytest.mark.parametrize("rows, message", [
    ([["M1", "deontology", "medium", "2", "original", D, ""]], "not listed"),
    ([["M1", "deontology", "medium", "2", "added", D, ""],
      ["M1", "individualism", "low", "3", "original", D, ""]], "doesn't match"),
    ([["M1", "deontology", "", "", "removed", D, ""],
      ["M1", "individualism", "", "", "removed", D, ""]], "every position was removed"),
    ([["M1", "deontology", "medium", "2", "original", D, ""],
      ["M1", "individualism", "low", "3", "original", D, ""],
      ["M1", "virtue_ethics", "low", "", "added", D, ""]], "needs its sentences"),
    ([["M1", "deontology", "sure", "2", "original", D, ""],
      ["M1", "individualism", "low", "3", "original", D, ""]], "confidence must be"),
    ([["M1", "deontology", "medium", "4", "original", D, ""],
      ["M1", "individualism", "low", "3", "original", D, ""]], "outside S1..S3"),
    ([["M1", "deontology", "medium", "2", "original", D, ""],
      ["M1", "individualism", "", "", "removed", D, ""],
      ["M1", "insufficient_philosophical_content", "", "", "added", D, ""]],
     "with positions still kept"),
    ([["M1", "deontology", "medium", "2", "original", "29/10/2026", ""],
      ["M1", "individualism", "low", "3", "original", "29/10/2026", ""]], "isn't an ISO date"),
])
def test_re_marking_files_are_checked(raw_v1, rows, message):
    _remark(raw_v1, rows)
    with pytest.raises(AdjudicationError, match=message):
        build(_config(raw_v1))


def test_lockbox_is_one_whole_fold_reachable_only_through_the_log(raw_v1):
    config = _config(raw_v1)
    result = build(config)
    corpus = result.corpus
    assert sorted(corpus.split.unique()) == ["dev", "lockbox"]
    assert corpus.groupby("fold").split.nunique().eq(1).all()  # folds are never split
    assert corpus.loc[corpus.split == "lockbox", "fold"].nunique() == 1

    dev, dev_gold = load_dev(config.processed_dir)
    assert (dev.split == "dev").all()
    assert set(dev_gold.passage_id) <= set(dev.passage_id)

    log = raw_v1 / "reports" / "lockbox_log.md"
    with pytest.raises(ValueError, match="reason"):
        open_lockbox("   ", config.processed_dir, log)
    assert not log.exists()
    box, box_gold = open_lockbox("final | evaluation", config.processed_dir, log)
    assert len(box) and (box.split == "lockbox").all()
    assert set(box.passage_id).isdisjoint(dev.passage_id)
    assert set(box_gold.passage_id) <= set(box.passage_id)
    assert "| final / evaluation |" in log.read_text(encoding="utf-8")


def test_label_review_replaces_labels_and_folds_stay_frozen(raw_v1):
    first = build(_config(raw_v1))
    folds_before = (raw_v1 / "splits" / "folds_v1.csv").read_text(encoding="utf-8")
    (raw_v1 / "adjudication").mkdir()
    _write_csv(raw_v1 / "adjudication" / "label_review_v1.csv",
               ["passage_id", "decision", "category", "confidence", "evidence_sentences",
                "decided_on", "source"],
               [["M0", "replace", "virtue_ethics", "high", "2", "2026-10-23", "test"],
                ["M1", "keep", "", "", "", "2026-10-23", "test"]])
    second = build(_config(raw_v1))
    gold = second.gold.set_index("passage_id")
    assert gold.loc["M0", "category"] == "virtue_ethics"
    assert gold.loc["M0", "label_source"] == "adjudicated"
    assert set(second.gold[second.gold.passage_id == "M1"].label_source) == {"primary"}
    assert second.corpus.set_index("passage_id").label_review.to_dict()["M1"] == "kept"
    # the changed labels do not move any passage between folds
    assert (raw_v1 / "splits" / "folds_v1.csv").read_text(encoding="utf-8") == folds_before
    assert (first.corpus.fold.to_numpy() == second.corpus.fold.to_numpy()).all()
    assert second.facts["folds_frozen"] and not first.facts["folds_frozen"]


real = pytest.mark.skipif(
    not (paths.PROCESSED_DIR / "passages.parquet").exists(), reason="real data not built locally"
)


@real
def test_real_build_has_no_leakage_and_matches_committed_folds():
    corpus = pd.read_parquet(paths.PROCESSED_DIR / "passages.parquet")
    assert check_no_leakage(corpus, "fold", LEAKAGE_KEYS) == {}
    committed = pd.read_csv(paths.SPLITS_DIR / "folds_v1.csv")
    merged = committed.merge(corpus[["passage_id", "fold", "group_id"]], on="passage_id",
                             suffixes=("", "_built"), validate="one_to_one")
    assert len(merged) == len(corpus)
    assert (merged.fold == merged.fold_built).all()
    assert (merged.group_id == merged.group_id_built).all()


@real
def test_real_build_gold_set_is_one_copy_per_text_with_decisions_applied():
    corpus = pd.read_parquet(paths.PROCESSED_DIR / "passages.parquet").set_index("passage_id")
    gold = pd.read_parquet(paths.PROCESSED_DIR / "gold_labels.parquet")
    assert set(gold.passage_id) == set(corpus.index[corpus.is_labelled])
    assert corpus.loc[gold.passage_id.unique(), "is_canonical"].all()
    assert not corpus.loc[gold.passage_id.unique(), "text_cluster"].duplicated().any()
    duplicates = pd.read_csv(paths.ADJUDICATION_DIR / "duplicates_v1.csv", dtype=str)
    expected = set(zip(duplicates.keep_passage_id, duplicates.category, strict=True))
    review_file = paths.ADJUDICATION_DIR / "label_review_v1.csv"
    if review_file.exists():
        review = pd.read_csv(review_file, dtype=str, keep_default_na=False)
        replaced = review[review.decision == "replace"]
        expected |= set(zip(replaced.passage_id, replaced.category, strict=True))
    adjudicated = gold[gold.label_source == "adjudicated"]
    assert set(zip(adjudicated.passage_id, adjudicated.category, strict=True)) == expected
