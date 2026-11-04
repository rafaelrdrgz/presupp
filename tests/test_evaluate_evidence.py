"""Scoring sentence marks against the evidence over cross-validation, and choosing v3."""

import numpy as np
import pandas as pd
import pytest
from conftest import LABELS

from presupp.choose import MARGIN, choose, replaces_v1
from presupp.data.dataset import load_dev
from presupp.evaluate_evidence import (
    _average_precision,
    evaluate,
    evidence_cases,
    score_runs,
    tune_marking,
    tune_on,
)
from presupp.experiment import ExperimentConfig
from presupp.runs import data_fingerprint, evidence_fingerprint
from presupp.train import run_experiment

POSITIONS = LABELS[:-1]
N_DEV_ROUNDS = 4


def test_marking_thresholds_are_tuned_per_position_or_shared():
    words = [5, 5, 5, 5]
    two_sentences = [(np.array([0.9, 0.8, 0.1, 0.1]), {0, 1}, words)] * 10
    one_sentence = [(np.array([0.9, 0.3, 0.3, 0.2]), {0}, words)] * 3
    marking = tune_marking({"a": two_sentences, "b": one_sentence}, ["a", "b", "c"])
    assert marking["a"] == 0.15  # the lowest threshold that marks S1 and S2 and nothing else
    # b has too few cases: it shares the threshold tuned on all 13 (which must leave out b's
    # 0.3s), as does c, which has none
    assert marking["b"] == marking["c"] == 0.35


def test_tuning_with_no_case_among_the_passages_fails():
    cases = pd.DataFrame([{"passage_id": "lockbox", "category": "a", "evidence": {0},
                           "words": [5, 5]}])
    with pytest.raises(ValueError, match="no case"):
        tune_on({"validation": np.array([[0.9], [0.1]])}, cases, ["a"])


def test_sentence_average_precision():
    scores = np.array([0.9, 0.1, 0.5])
    assert _average_precision(scores, {0, 2}) == 1.0
    assert _average_precision(scores, {1}) == pytest.approx(1 / 3)
    assert _average_precision(np.array([np.nan, 0.2, 0.1]), {0}) == pytest.approx(1 / 3)


def _cfg(model_dir, name, **overrides):
    settings = dict(name=name, description=name, model_name=str(model_dir), max_length=16,
                    batch_size=4, epochs=1, patience=1, freeze_layers=1, seeds=(0,))
    settings.update(overrides)
    return ExperimentConfig(**settings)


def test_experiments_are_scored_on_the_same_positions_and_compared(tiny_data, tiny_model):
    runs, processed = tiny_data / "runs", tiny_data / "processed"
    configs = [_cfg(tiny_model, "v3_mil", model_type="sentence_transformer",
                    passage_from="sentences", confidence_weighting="weights"),
               _cfg(tiny_model, "fixed_unweighted_pointing", score_sentences_alone=True)]
    for cfg in configs:
        run_experiment(cfg, runs, processed, device="cpu")
    passages, gold = load_dev(processed)
    dev = passages[passages.is_labelled]
    cases = evidence_cases(dev, gold, POSITIONS)
    fingerprints = (data_fingerprint(passages, gold), evidence_fingerprint(gold))
    experiments = {}
    for cfg in configs:
        seeds, incomplete, thresholds = score_runs(cfg, cases, POSITIONS, runs, N_DEV_ROUNDS,
                                                   fingerprints)
        assert list(seeds) == [0] and incomplete == {} and len(thresholds) == N_DEV_ROUNDS
        experiments[cfg.name] = seeds
    pairs = [("v3_mil", "fixed_unweighted_pointing")]
    results, (comparison,) = evaluate(experiments, passages.set_index("passage_id")["group_id"],
                                      pairs, n_boot=20)
    # every dev position with evidence that fits, each in one sentence of three
    assert results["v3_mil"]["n_cases"] == len(cases) == (gold.evidence_0based.notna()
                                                          & gold.passage_id.isin(dev.passage_id)
                                                          ).sum()
    assert results["v3_mil"]["chance"]["mean"] == pytest.approx(1 / 3)
    for r in results.values():
        assert 0 <= r["first_hit"]["mean"] <= 1
        assert r["first_hit"]["ci95"][0] <= r["first_hit"]["mean"] <= r["first_hit"]["ci95"][1]
    expected = (results["v3_mil"]["first_hit"]["mean"]
                - results["fixed_unweighted_pointing"]["first_hit"]["mean"])
    assert comparison["first_hit"]["difference"] == pytest.approx(expected)

    gold_file = processed / "gold_labels.parquet"
    changed = pd.read_parquet(gold_file)
    changed["evidence_0based"] = [None if ev is None else [2] for ev in changed.evidence_0based]
    changed.to_parquet(gold_file, index=False)
    passages, gold = load_dev(processed)
    with pytest.raises(RuntimeError, match="evidence"):
        score_runs(configs[0], evidence_cases(dev, gold, POSITIONS), POSITIONS, runs,
                   N_DEV_ROUNDS, (data_fingerprint(passages, gold), evidence_fingerprint(gold)))


def test_the_lockbox_job_compares_v3_with_its_baseline(tiny_data, tiny_model):
    import json

    from presupp.final import run_final
    from presupp.runs import load_run, load_sentences

    runs, processed, results = tiny_data / "runs", tiny_data / "processed", tiny_data / "results"
    configs = [_cfg(tiny_model, "v3_mil", model_type="sentence_transformer",
                    passage_from="sentences", confidence_weighting="weights"),
               _cfg(tiny_model, "fixed_unweighted_pointing", score_sentences_alone=True)]
    log = tiny_data / "lockbox_log.md"
    scored = run_final(configs, "unit test", processed, runs, results, log,
                       tiny_data / "label_support.csv", device="cpu", compare_first=True)
    assert [r.n_passages for r in scored] == [8, 8]  # fold 4
    assert sum(line.startswith("| 20") for line in log.read_text().splitlines()) == 1
    for cfg in configs:
        meta, _ = load_run(runs / cfg.name / "final_seed0")
        assert meta["sizes"] == {"train": 24, "val": 8, "lockbox": 8}  # never trained on it
        assert set(load_sentences(runs / cfg.name / "final_seed0").part) == {"val", "lockbox"}
        assert (results / f"lockbox_{cfg.name}.md").exists()
    assert "v3_mil − fixed_unweighted_pointing" in (results / "lockbox_comparisons.md").read_text(
        encoding="utf-8")
    evidence = json.loads((results / "lockbox_evidence_v3_mil.json").read_text())
    gold = pd.read_parquet(processed / "gold_labels.parquet")
    assert evidence["n_cases"] == (gold.passage_id.str.startswith("P4")
                                   & gold.evidence_0based.notna()).sum()
    assert "lockbox" in (results / "lockbox_evidence_v3_mil.md").read_text(encoding="utf-8")
    (comparison,) = json.loads((results / "lockbox_evidence_comparisons.json").read_text())
    assert (comparison["a"], comparison["b"]) == ("v3_mil", "fixed_unweighted_pointing")


def test_the_choice_between_the_variants():
    f = {"v3_joint": 0.50, "v3_mil": 0.52}
    h = {"v3_joint": 0.80, "v3_mil": 0.60}
    assert choose(f, h)[0] == "v3_mil"  # passage scores apart: they decide
    f["v3_mil"] = 0.50 + MARGIN / 2
    assert choose(f, h)[0] == "v3_joint"  # close on passages: the evidence decides
    h["v3_mil"] = 0.80 - MARGIN / 2
    assert choose(f, h)[0] == "v3_mil"  # close on both


def test_whether_v3_replaces_v1():
    assert replaces_v1(0.495, 0.50, [0.01, 0.05])[0]
    assert not replaces_v1(0.48, 0.50, [0.01, 0.05])[0]  # passages more than MARGIN worse
    assert not replaces_v1(0.52, 0.50, [-0.01, 0.05])[0]  # evidence not reliably better
