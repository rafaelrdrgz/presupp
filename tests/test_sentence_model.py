"""The v3 sentence model: finding sentences in the text, its targets and loss, and full runs."""

import numpy as np
import pandas as pd
import pytest
from conftest import LABELS

from presupp.experiment import ExperimentConfig
from presupp.runs import PROB_PREFIX, load_run, load_sentences, probabilities
from presupp.sentence_model import (
    passage_input,
    sentence_spans,
    sentence_targets,
    token_sentence_ids,
    v3_loss,
)
from presupp.train import run_experiment

POSITIONS = LABELS[:-1]


def _tiny(model_dir, **overrides):
    settings = dict(name="tiny_v3", description="tiny v3", model_name=str(model_dir),
                    model_type="sentence_transformer", confidence_weighting="weights",
                    max_length=16, batch_size=4, epochs=2, patience=1, freeze_layers=1,
                    seeds=(0,))
    settings.update(overrides)
    return ExperimentConfig(**settings)


def test_sentences_are_found_in_the_text_ignoring_whitespace():
    text = "It rose.[6] Then  it fell. "
    stored = ["It rose. [6]", "Then it fell."]  # the stored split spaces the citation
    spans = sentence_spans(text, stored)
    assert [text[a:b] for a, b in spans] == ["It rose.[6]", "Then  it fell."]
    assert sentence_spans(text, ["It rose.", "Then it fell."]) is None  # "[6]" left over
    assert sentence_spans("Posted 3 May. It rose.", ["It rose."]) is None
    source, spans, joined = passage_input("Something else.", ["One two.", "Three."])
    assert joined and source == "One two. Three."
    assert [source[a:b] for a, b in spans] == ["One two.", "Three."]


def test_tokens_map_to_their_sentence_and_special_tokens_to_none(tiny_model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    text = "w1 w2 w3 w4 w5 w6"
    _, spans, _ = passage_input(text, ["w1 w2 w3", "w4 w5 w6"])
    encoded = tokenizer(text, return_offsets_mapping=True)
    assert token_sentence_ids(encoded["offset_mapping"], spans) == [-1, 0, 0, 0, 1, 1, 1, -1]
    cut = tokenizer(text, truncation=True, max_length=5, return_offsets_mapping=True)
    assert token_sentence_ids(cut["offset_mapping"], spans) == [-1, 0, 0, 0, -1]  # S2 cut off


def test_targets_follow_the_evidence_and_the_confidence():
    gold = pd.DataFrame([
        {"passage_id": "A", "category": POSITIONS[0], "confidence": "low",
         "evidence_0based": [1]},
        {"passage_id": "A", "category": POSITIONS[1], "confidence": "high",
         "evidence_0based": None},  # labelled, evidence doesn't fit
        {"passage_id": "B", "category": LABELS[-1], "confidence": None, "evidence_0based": None},
    ])
    (ta, wa, ia), (tb, wb, ib) = sentence_targets(["A", "B"], [3, 2], gold, POSITIONS, "weights")
    assert ta[:, 0].tolist() == [0, 1, 0] and wa[:, 0].tolist() == [1, 0.5, 1]
    assert ia[0] and not ia[1] and ia[2:].all()  # absent positions: every sentence negative
    assert ta[:, 2:].sum() == 0 and (wa[:, 2:] == 1).all()
    assert tb.sum() == 0 and ib.all() and (wb == 1).all()  # an IPC passage: all negatives
    (ta, wa, _), _ = sentence_targets(["A", "B"], [3, 2], gold, POSITIONS, "none")
    assert wa[:, 0].tolist() == [1, 1, 1]


def test_the_loss_weights_and_its_two_parts():
    import torch
    from torch.nn import functional as F

    passage = torch.tensor([[0.3, -1.0]])
    sentence = torch.tensor([[[2.0], [-0.5], [0.0]]])  # 3 sentences, 1 position
    visible = torch.tensor([[True, True, False]])  # S3 cut off
    batch = {"labels": torch.tensor([[1.0, 0.0]]), "weights": torch.tensor([[0.5, 1.0]]),
             "sentence_targets": torch.tensor([[[1.0], [0.0], [1.0]]]),
             "sentence_weights": torch.tensor([[[0.5], [1.0], [1.0]]]),
             "sentence_include": torch.tensor([[True]])}
    total, p_loss, s_loss = v3_loss(passage, sentence, visible, batch, sentence_loss_weight=2.0)
    bce = F.binary_cross_entropy_with_logits
    expected_p = (0.5 * bce(passage[0, 0], torch.tensor(1.0))
                  + bce(passage[0, 1], torch.tensor(0.0))) / 2
    expected_s = (0.5 * bce(sentence[0, 0, 0], torch.tensor(1.0))
                  + bce(sentence[0, 1, 0], torch.tensor(0.0))) / 2  # over visible sentences
    assert torch.isclose(p_loss, expected_p) and torch.isclose(s_loss, expected_s)
    assert torch.isclose(total, expected_p + 2.0 * expected_s)
    batch["sentence_include"] = torch.tensor([[False]])
    total, p_loss, s_loss = v3_loss(passage, sentence, visible, batch, sentence_loss_weight=2.0)
    assert s_loss.item() == 0 and torch.isclose(total, p_loss)


def test_sentence_scores_are_mean_pooled_and_the_mil_passage_takes_the_best(tiny_model):
    import torch
    from transformers import AutoTokenizer

    from presupp.sentence_model import SentencePassageClassifier
    torch.manual_seed(0)
    model = SentencePassageClassifier(str(tiny_model), len(LABELS), 0.2, "sentences").eval()
    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    text = "w1 w2 w3 w4 w5 w6 w7"
    _, spans, _ = passage_input(text, ["w1 w2 w3", "w4 w5", "w6 w7"])
    encoded = tokenizer(text, truncation=True, max_length=7, return_offsets_mapping=True)
    ids = token_sentence_ids(encoded["offset_mapping"], spans)  # w7 and </s> cut: S3 has w6
    ids = ids[:-1] + [-1]  # and drop S3's only token, so it isn't visible
    input_ids = torch.tensor([encoded["input_ids"]])
    with torch.no_grad():
        passage, sentence, visible = model(input_ids, torch.ones_like(input_ids),
                                           torch.tensor([ids]), 3)
        hidden = model.encoder(input_ids=input_ids).last_hidden_state[0]
        token_logits = model.sentence_head(hidden)
    assert visible.tolist() == [[True, True, False]]
    for k in (0, 1):  # a linear head on the mean is the mean of the token logits
        mine = [t for t, s in enumerate(ids) if s == k]
        assert torch.allclose(sentence[0, k], token_logits[mine].mean(0), atol=1e-5)
    assert torch.allclose(passage[0, :-1], sentence[0, :2].amax(0))  # S3 ignored
    assert torch.allclose(passage[0, -1], model.head(hidden[0])[-1])  # IPC from the first token


@pytest.mark.parametrize("passage_from", ["cls", "sentences"])
def test_a_v3_round_writes_sentence_scores_and_resumes(tiny_data, tiny_model, passage_from):
    cfg = _tiny(tiny_model, name=f"tiny_{passage_from}", passage_from=passage_from)
    runs, processed = tiny_data / "runs", tiny_data / "processed"
    (run_dir,) = run_experiment(cfg, runs, processed, rounds=[0], device="cpu")
    meta, preds = load_run(run_dir)
    sentences = load_sentences(run_dir)
    assert set(sentences.part) == {"val", "test"}
    assert not sentences.passage_id.str.startswith("P4").any()  # lockbox never seen
    assert len(sentences) == 16 * 3 and sentences.groupby("passage_id").sentence.max().eq(2).all()
    scores = sentences[[PROB_PREFIX + p for p in POSITIONS]].to_numpy()
    assert ((scores > 0) & (scores < 1)).all()
    assert meta["joined_passages"] == {"train": 0, "val": 0, "test": 0}
    assert meta["train_evidence_pairs"] > 0 and meta["evidence_fingerprint"]
    assert {"train_passage_loss", "train_sentence_loss"} <= set(meta["history"][0])
    if passage_from == "sentences":  # the passage score is the best sentence's
        best = sentences.groupby("passage_id")[[PROB_PREFIX + p for p in POSITIONS]].max()
        mine = preds.set_index("passage_id").loc[best.index, best.columns]
        np.testing.assert_allclose(mine.to_numpy(), best.to_numpy(), atol=1e-6)
    assert run_experiment(cfg, runs, processed, rounds=[0], device="cpu") == []  # resumes

    gold = pd.read_parquet(processed / "gold_labels.parquet")
    gold["evidence_0based"] = [None if ev is None else [2] for ev in gold.evidence_0based]
    gold.to_parquet(processed / "gold_labels.parquet", index=False)  # re-marked evidence
    with pytest.raises(RuntimeError, match="evidence"):
        run_experiment(cfg, runs, processed, rounds=[0], device="cpu")


def test_v3_is_reproducible_and_reads_cut_and_mismatched_passages(tiny_data, tiny_model):
    processed = tiny_data / "processed"
    passages = pd.read_parquet(processed / "passages.parquet")
    passages.loc[passages.passage_id == "P000", "text"] += " w1"  # stored sentences differ
    passages.to_parquet(processed / "passages.parquet", index=False)
    a, b = (run_experiment(_tiny(tiny_model, name=n, max_length=11), tiny_data / "runs",
                           processed, rounds=[0], device="cpu")[0] for n in ("a", "b"))
    np.testing.assert_allclose(probabilities(load_run(a)[1], LABELS),
                               probabilities(load_run(b)[1], LABELS), atol=1e-6)
    meta = load_run(a)[0]
    assert meta["joined_passages"]["test"] == 1  # P000 is in fold 0, the test fold
    # 11 tokens: <s>, two 4-word sentences, a word of the third, </s>
    assert meta["unscored_sentences"] == {"val": 0, "test": 0}
    a_sentences = load_sentences(a)
    assert a_sentences[PROB_PREFIX + POSITIONS[0]].notna().all()
    # 10 tokens: nothing of the third sentence
    short = run_experiment(_tiny(tiny_model, name="c", max_length=10), tiny_data / "runs",
                           processed, rounds=[0], device="cpu")[0]
    assert load_run(short)[0]["unscored_sentences"] == {"val": 8, "test": 8}
    cut = load_sentences(short)
    assert cut.loc[cut.sentence == 2, PROB_PREFIX + POSITIONS[0]].isna().all()


def test_the_pointing_baseline_scores_each_sentence_alone(tiny_data, tiny_model):
    cfg = ExperimentConfig(name="tiny_pointing", description="baseline",
                           model_name=str(tiny_model), max_length=16, batch_size=4, epochs=1,
                           patience=1, freeze_layers=1, seeds=(0,), score_sentences_alone=True)
    (run_dir,) = run_experiment(cfg, tiny_data / "runs", tiny_data / "processed", rounds=[0],
                                device="cpu")
    sentences = load_sentences(run_dir)
    assert len(sentences) == 16 * 3 and set(sentences.part) == {"val", "test"}
    assert list(sentences.columns[3:]) == [PROB_PREFIX + p for p in POSITIONS]  # no IPC
    assert sentences[sentences.columns[3:]].notna().all().all()
    assert load_run(run_dir)[0]["evidence_fingerprint"]


def test_sentence_settings_are_checked():
    with pytest.raises(ValueError, match="hard labels"):
        ExperimentConfig(name="x", description="", model_type="sentence_transformer",
                         label_mode="soft")
    with pytest.raises(ValueError, match="sentence_transformer"):
        ExperimentConfig(name="x", description="", passage_from="sentences")
    with pytest.raises(ValueError, match="score_sentences_alone"):
        ExperimentConfig(name="x", description="", model_type="prior",
                         score_sentences_alone=True)
