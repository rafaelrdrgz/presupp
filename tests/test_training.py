"""End-to-end training, scoring and lockbox evaluation on the tiny synthetic dataset."""

import numpy as np
import pandas as pd
import pytest
from conftest import LABELS

from presupp.data.dataset import load_dev
from presupp.evaluate import load_label_support, score, seed_decisions
from presupp.experiment import ExperimentConfig
from presupp.final import run_final
from presupp.runs import load_run, probabilities
from presupp.train import run_experiment

N_DEV_ROUNDS = 4  # dev folds 0-3


def _tiny_transformer(model_dir, **overrides):
    settings = dict(name="tiny", description="tiny test model", model_name=str(model_dir),
                    max_length=16, batch_size=4, epochs=2, patience=1, freeze_layers=1,
                    seeds=(0,))
    settings.update(overrides)
    return ExperimentConfig(**settings)


def test_transformer_round_writes_predictions_and_resumes(tiny_data, tiny_model):
    cfg = _tiny_transformer(tiny_model)
    runs, processed = tiny_data / "runs", tiny_data / "processed"
    written = run_experiment(cfg, runs, processed, rounds=[0], device="cpu")
    assert len(written) == 1

    meta, preds = load_run(written[0])
    assert (meta["test_fold"], meta["val_fold"], meta["train_folds"]) == (0, 1, [2, 3])
    assert set(preds.part) == {"val", "test"}
    assert not preds.passage_id.str.startswith("P4").any()  # lockbox never seen
    assert "U000" not in set(preds.passage_id)  # unlabelled passages are not scored
    probs = probabilities(preds, LABELS)
    assert probs.shape == (16, len(LABELS)) and ((probs >= 0) & (probs <= 1)).all()
    assert 1 <= meta["best_epoch"] <= meta["epochs_run"] <= 2
    assert meta["frozen_parameters"] > 0

    assert run_experiment(cfg, runs, processed, rounds=[0], device="cpu") == []  # resumes


def test_v1_recipe_options_run(tiny_data, tiny_model):
    cfg = _tiny_transformer(
        tiny_model, name="tiny_v1", split="passage_cv", label_mode="soft",
        class_weighting="inverse_capped", scheduler="onecycle_cosine",
        decay_norm_and_bias=True, selection_metric="macro_f1_at_0.5", threshold_policy="v1",
        decode="independent", epochs=1,
    )
    (run_dir,) = run_experiment(cfg, tiny_data / "runs", tiny_data / "processed", rounds=[1],
                                device="cpu")
    meta, _ = load_run(run_dir)
    assert len(meta["class_weights"]) == len(LABELS)
    assert max(meta["class_weights"]) <= 15.0


def test_same_seed_gives_same_predictions(tiny_data, tiny_model):
    processed = tiny_data / "processed"
    a = run_experiment(_tiny_transformer(tiny_model, name="a"), tiny_data / "runs", processed,
                       rounds=[2], device="cpu")
    b = run_experiment(_tiny_transformer(tiny_model, name="b"), tiny_data / "runs", processed,
                       rounds=[2], device="cpu")
    pa, pb = load_run(a[0])[1], load_run(b[0])[1]
    np.testing.assert_allclose(probabilities(pa, LABELS), probabilities(pb, LABELS), atol=1e-6)


@pytest.mark.parametrize("model_type", ["prior", "tfidf_lr"])
def test_baseline_full_cv_is_scored_on_every_dev_passage(tiny_data, model_type):
    cfg = ExperimentConfig(name=model_type, description="baseline", model_type=model_type,
                           seeds=(0,))
    runs, processed = tiny_data / "runs", tiny_data / "processed"
    assert len(run_experiment(cfg, runs, processed)) == N_DEV_ROUNDS

    passages, gold = load_dev(processed)
    seeds, incomplete = seed_decisions(cfg, runs, gold, LABELS, N_DEV_ROUNDS)
    assert list(seeds) == [0] and incomplete == {}
    result = score(cfg.name, cfg.description, seeds, gold, LABELS,
                   passages.set_index("passage_id")["group_id"],
                   load_label_support(tiny_data / "label_support.csv"), n_boot=50)
    assert result.n_passages == 32
    h = result.headline
    assert 0 <= h["macro_f1"]["mean"] <= 1 and h["macro_f1"]["ci95"][0] <= h["macro_f1"]["ci95"][1]
    assert len(result.per_label) == len(LABELS)
    if model_type == "prior":
        # constant probabilities within a round: every passage in a test fold gets the same
        # decisions (ids are P<fold><n>)
        ids, _, decisions = seeds[0]
        for fold in set(i[1] for i in ids):
            in_fold = decisions[[i[1] == fold for i in ids]]
            assert (in_fold == in_fold[0]).all()


def test_runs_made_on_different_labels_are_refused(tiny_data):
    cfg = ExperimentConfig(name="prior", description="b", model_type="prior", seeds=(0,))
    runs, processed = tiny_data / "runs", tiny_data / "processed"
    run_experiment(cfg, runs, processed, rounds=[0])
    gold = pd.read_parquet(processed / "gold_labels.parquet")
    gold.loc[0, "category"] = "virtue_ethics"  # a label correction after the run
    gold.to_parquet(processed / "gold_labels.parquet", index=False)
    with pytest.raises(RuntimeError, match="different|current data"):
        run_experiment(cfg, runs, processed, rounds=[0])


def test_incomplete_seeds_are_reported_not_scored(tiny_data):
    cfg = ExperimentConfig(name="prior", description="b", model_type="prior", seeds=(0,))
    run_experiment(cfg, tiny_data / "runs", tiny_data / "processed", rounds=[0, 1])
    _, gold = load_dev(tiny_data / "processed")
    seeds, incomplete = seed_decisions(cfg, tiny_data / "runs", gold, LABELS, N_DEV_ROUNDS)
    assert seeds == {} and incomplete == {0: 2}


def test_final_opens_the_lockbox_once_and_scores_it(tiny_data):
    cfg = ExperimentConfig(name="tfidf_lr", description="b", model_type="tfidf_lr", seeds=(0,))
    log = tiny_data / "lockbox_log.md"
    (result,) = run_final([cfg], "unit test", tiny_data / "processed", tiny_data / "runs",
                          tiny_data / "results", log, tiny_data / "label_support.csv")
    assert result.n_passages == 8  # fold 4
    entries = [line for line in log.read_text(encoding="utf-8").splitlines()
               if line.startswith("| 20")]
    assert len(entries) == 1 and "unit test" in entries[0]
    assert (tiny_data / "results" / "lockbox_tfidf_lr.md").exists()
    _, preds = load_run(tiny_data / "runs" / "tfidf_lr" / "final_seed0")
    assert set(preds.part) == {"val", "lockbox"}
    assert set(preds.loc[preds.part == "val", "passage_id"].str[1]) == {"3"}  # last dev fold
    assert pd.Series(preds.passage_id).is_unique


def test_data_fingerprint_is_independent_of_the_operating_system():
    import hashlib

    from presupp.runs import data_fingerprint

    passages = pd.DataFrame({"passage_id": ["b", "a"], "fold": [1, 0], "split": ["dev", "dev"]})
    gold = pd.DataFrame({"passage_id": ["a", "b"], "category": ["x", "y"],
                         "confidence": ["high", None]})
    # computed by hand with "\n" line breaks: the same value on Windows and on the Linux cluster
    expected = hashlib.sha1(
        b"passage_id,fold,split\na,0,dev\nb,1,dev\n"
        b"passage_id,category,confidence\na,x,high\nb,y,\n"
    ).hexdigest()[:12]
    assert data_fingerprint(passages, gold) == expected
