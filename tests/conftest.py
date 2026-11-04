"""Shared fixtures: a tiny synthetic processed dataset and an offline tiny transformer."""

import numpy as np
import pandas as pd
import pytest

from presupp.labels import IPC, MEASURABLE_MIN_ARTICLES, model_labels
from presupp.taxonomy import load_taxonomy

LABELS = model_labels(load_taxonomy("v1"))
WORDS = [f"w{i}" for i in range(40)]
N_FOLDS, PER_FOLD, LOCKBOX_FOLD = 5, 8, 4


@pytest.fixture
def tiny_data(tmp_path):
    """5 folds x 8 labelled passages (fold 4 = lockbox), two passages per article, plus one
    unlabelled dev passage. Each passage is 3 stored sentences of 4 words; a passage's first
    category has its evidence in one sentence, except in every fourth passage (k % 4 == 1),
    whose evidence doesn't fit. Returns the root folder with processed/ and label_support.csv."""
    rng = np.random.default_rng(0)
    passages, gold = [], []
    for fold in range(N_FOLDS):
        for k in range(PER_FOLD):
            pid = f"P{fold}{k:02d}"
            words = list(rng.choice(WORDS, 12))
            sentences = [" ".join(words[i:i + 4]) for i in (0, 4, 8)]
            passages.append({"passage_id": pid, "text": " ".join(words), "sentences": sentences,
                             "fold": fold, "group_id": f"g{fold}_{k // 2}", "is_labelled": True,
                             "split": "lockbox" if fold == LOCKBOX_FOLD else "dev"})
            if k % 4 == 0:
                cats = [IPC]
            else:
                cats = [LABELS[i] for i in rng.choice(len(LABELS) - 1, 2, replace=False)]
            for n, c in enumerate(cats):
                fits = c != IPC and not (n == 0 and k % 4 == 1)
                gold.append({"passage_id": pid, "category": c,
                             "confidence": None if c == IPC else ("high", "low")[n],
                             "evidence": [] if c == IPC else [k % 3],
                             "evidence_0based": [k % 3] if fits else None,
                             "evidence_reading": ("none" if c == IPC
                                                  else "0-based" if fits else "doesn't fit"),
                             "label_source": "primary"})
    passages.append({"passage_id": "U000", "text": "unlabelled words here",
                     "sentences": ["unlabelled words here"], "fold": 0,
                     "group_id": "gU", "is_labelled": False, "split": "dev"})
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(passages).to_parquet(processed / "passages.parquet", index=False)
    gold = pd.DataFrame(gold)
    gold.to_parquet(processed / "gold_labels.parquet", index=False)

    groups = pd.DataFrame(passages).set_index("passage_id")["group_id"]
    support = (gold.assign(group_id=gold.passage_id.map(groups)).groupby("category")
               .agg(positives=("passage_id", "nunique"), articles=("group_id", "nunique"))
               .reindex(LABELS, fill_value=0))
    support["measurable"] = support["articles"] >= min(MEASURABLE_MIN_ARTICLES, 3)
    support.rename_axis("label").reset_index().to_csv(tmp_path / "label_support.csv", index=False)
    return tmp_path


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """A 2-layer, 16-dimensional RoBERTa with a word-level tokenizer over WORDS, saved locally.
    Like roberta-base's, the tokenizer wraps each text in <s> ... </s>."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel

    directory = tmp_path_factory.mktemp("tiny_roberta")
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    vocab.update({w: i + 4 for i, w in enumerate(WORDS + ["unlabelled", "words", "here"])})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
                            bos_token="<s>", eos_token="</s>").save_pretrained(directory)
    config = RobertaConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2,
                           num_attention_heads=2, intermediate_size=32,
                           max_position_embeddings=64, pad_token_id=1, bos_token_id=0,
                           eos_token_id=2, type_vocab_size=1)
    RobertaModel(config).save_pretrained(directory)
    return directory
