"""v3: encode the passage once, score every sentence for the 12 positions.

A sentence is the mean of its tokens' final states. passage_from sets the passage score:
cls = first token, sentence head trained alongside (v3_joint); sentences = each position's
best sentence (v3_mil). IPC always comes from the first token. Labels: 12 positions, then IPC.

Sentence targets: a labelled position's evidence sentences are positives, its other sentences
negatives; a position the passage lacks makes every sentence negative. Evidence that doesn't fit
the stored sentences trains the passage score only. confidence_weighting=weights weights
positives by confidence. loss = passage + sentence_loss_weight * sentence.
Sentences cut off by max_length score NaN.
"""

import time
from dataclasses import replace

import numpy as np
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from presupp.evidence import SENTENCE_TOKENS
from presupp.experiment import ExperimentConfig
from presupp.labels import CONFIDENCE_WEIGHTS, IPC, confidence_weight_matrix, target_matrix
from presupp.transformer import (
    PassageClassifier,
    _param_groups,
    _scheduler,
    _selection_score,
    freeze_bottom,
    predict_texts,
    set_seed,
)


def sentence_spans(text: str, sentences: list[str]) -> list[tuple[int, int]] | None:
    """Char spans of the sentences in text, ignoring whitespace. None if they don't match."""
    spans, i = [], 0
    for sentence in sentences:
        start = None
        for c in sentence:
            if c.isspace():
                continue
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text) or text[i] != c:
                return None
            start = i if start is None else start
            i += 1
        spans.append((i, i) if start is None else (start, i))
    return spans if not text[i:].strip() else None


def passage_input(text: str, sentences: list[str]) -> tuple[str, list[tuple[int, int]], bool]:
    """(text to read, sentence spans in it, whether the sentences had to be joined)."""
    spans = sentence_spans(text, sentences)
    if spans is not None:
        return text, spans, False
    joined, spans = "", []
    for k, sentence in enumerate(sentences):
        joined += " " if k else ""
        spans.append((len(joined), len(joined) + len(sentence)))
        joined += sentence
    return joined, spans, True


def token_sentence_ids(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    """Sentence index per token, by start offset; -1 for special tokens and tokens outside."""
    starts = np.array([s for s, _ in spans])
    out = []
    for a, b in offsets:
        k = int(np.searchsorted(starts, a, side="right")) - 1
        out.append(k if b > a and k >= 0 and a < spans[k][1] else -1)
    return out


def sentence_targets(passage_ids, n_sentences: list[int], gold, positions: list[str],
                     confidence_weighting: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per passage (targets, weights, include); include = positions that train the sentence head.
    gold needs evidence_0based."""
    col = {p: j for j, p in enumerate(positions)}
    labelled = gold[gold["passage_id"].isin(set(passage_ids)) & gold["category"].isin(col)]
    by_passage = {pid: g for pid, g in labelled.groupby("passage_id")}
    out = []
    for pid, n in zip(passage_ids, n_sentences, strict=True):
        targets = np.zeros((n, len(positions)), dtype=np.float32)
        weights = np.ones_like(targets)
        include = np.ones(len(positions), dtype=bool)
        rows = by_passage.get(pid)
        for r in [] if rows is None else rows.itertuples():
            j = col[r.category]
            evidence = [] if r.evidence_0based is None else [int(e) for e in r.evidence_0based]
            evidence = [e for e in evidence if 0 <= e < n]
            if not evidence:  # labelled, but no sentence to point at
                include[j] = False
                continue
            targets[evidence, j] = 1.0
            if confidence_weighting == "weights":
                weights[evidence, j] = CONFIDENCE_WEIGHTS.get(r.confidence, 1.0)
        out.append((targets, weights, include))
    return out


class SentencePassageClassifier(PassageClassifier):
    """PassageClassifier plus a sentence head. With passage_from="sentences", only self.head's
    IPC output is used."""

    def __init__(self, model_name: str, n_labels: int, dropout: float, passage_from: str):
        super().__init__(model_name, n_labels, dropout)
        self.passage_from = passage_from
        self.sentence_head = nn.Linear(self.encoder.config.hidden_size, n_labels - 1)

    def forward(self, input_ids, attention_mask, sentence_ids, n_sentences: int):
        """(passage (B, L), sentence (B, S, P), visible (B, S))."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        passage = self.head(self.dropout(hidden[:, 0]))
        member = (sentence_ids.unsqueeze(-1)
                  == torch.arange(n_sentences, device=hidden.device)).to(hidden.dtype)  # B, T, S
        counts = member.sum(1)
        pooled = member.transpose(1, 2) @ hidden / counts.clamp(min=1).unsqueeze(-1)
        sentence = self.sentence_head(self.dropout(pooled))
        visible = counts > 0
        if self.passage_from == "sentences":
            best = sentence.masked_fill(~visible.unsqueeze(-1), float("-inf")).amax(1)
            best = torch.where(visible.any(1, keepdim=True), best, passage[:, :-1])
            passage = torch.cat([best, passage[:, -1:]], dim=1)
        return passage, sentence, visible


def v3_loss(passage, sentence, visible, batch, sentence_loss_weight: float):
    """(total, passage loss, sentence loss). Sentence BCE is averaged over visible sentences,
    then over included pairs."""
    passage_loss = (batch["weights"] * F.binary_cross_entropy_with_logits(
        passage, batch["labels"], reduction="none")).mean()
    bce = F.binary_cross_entropy_with_logits(sentence, batch["sentence_targets"], reduction="none")
    weighted = (batch["sentence_weights"] * visible.unsqueeze(-1) * bce).sum(1)
    per_pair = weighted / visible.sum(1, keepdim=True).clamp(min=1)
    include = batch["sentence_include"] & visible.any(1, keepdim=True)
    sentence_loss = per_pair[include].mean() if include.any() else per_pair.sum() * 0.0
    return passage_loss + sentence_loss_weight * sentence_loss, passage_loss, sentence_loss


def _items(tokenizer, texts, sentences, max_length: int) -> tuple[list[dict], int]:
    """(tokenised items with sentence ids, how many passages were read joined)."""
    items, joined = [], 0
    for text, ss in zip(texts, sentences, strict=True):
        source, spans, was_joined = passage_input(str(text), [str(s) for s in ss])
        encoded = tokenizer(source, truncation=True, max_length=max_length,
                            return_offsets_mapping=True)
        items.append({"input_ids": encoded["input_ids"], "n_sentences": len(spans),
                      "sentence_ids": token_sentence_ids(encoded["offset_mapping"], spans)})
        joined += was_joined
    return items, joined


def _loader(tokenizer, items, targets, weights, sentence_sets, cfg: ExperimentConfig,
            shuffle: bool, seed: int) -> DataLoader:
    """sentence_sets is None for prediction-only parts."""
    targets = torch.as_tensor(targets, dtype=torch.float32)
    weights = torch.as_tensor(weights, dtype=torch.float32)
    positions = targets.shape[1] - 1

    def collate(indices):
        batch = tokenizer.pad([{"input_ids": items[i]["input_ids"]} for i in indices],
                              return_tensors="pt")
        n_tokens = batch["input_ids"].shape[1]
        n_sentences = [items[i]["n_sentences"] for i in indices]
        width = max(n_sentences)
        sentence_ids = torch.full((len(indices), n_tokens), -1, dtype=torch.long)
        s_targets = torch.zeros(len(indices), width, positions)
        s_weights = torch.zeros(len(indices), width, positions)
        include = torch.zeros(len(indices), positions, dtype=torch.bool)
        for b, i in enumerate(indices):
            ids = items[i]["sentence_ids"]
            sentence_ids[b, :len(ids)] = torch.as_tensor(ids)
            if sentence_sets is not None:
                t, w, inc = sentence_sets[i]
                s_targets[b, :len(t)] = torch.from_numpy(t)
                s_weights[b, :len(w)] = torch.from_numpy(w)
                include[b] = torch.from_numpy(inc)
        batch.update(labels=targets[indices], weights=weights[indices], sentence_ids=sentence_ids,
                     sentence_targets=s_targets, sentence_weights=s_weights,
                     sentence_include=include)
        batch["n_sentences"] = n_sentences
        return batch

    generator = torch.Generator().manual_seed(seed)
    size = cfg.batch_size if shuffle else cfg.batch_size * 4
    return DataLoader(list(range(len(items))), batch_size=size, shuffle=shuffle,
                      generator=generator, collate_fn=collate)


def _forward(model, batch, device):
    tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    out = model(tensors["input_ids"], tensors["attention_mask"], tensors["sentence_ids"],
                tensors["sentence_targets"].shape[1])
    return out, tensors


@torch.no_grad()
def _predict(model, loader, device, sentence_loss_weight: float | None = None
             ) -> tuple[np.ndarray, list[np.ndarray], float]:
    """(passage probs, one (sentences, positions) array per passage, mean loss)."""
    model.eval()
    probs, sentences, total, count = [], [], 0.0, 0
    for batch in loader:
        (passage, sentence, visible), tensors = _forward(model, batch, device)
        if sentence_loss_weight is not None:
            loss, _, _ = v3_loss(passage, sentence, visible, tensors, sentence_loss_weight)
            total += loss.item() * len(passage)
            count += len(passage)
        probs.append(torch.sigmoid(passage).float().cpu().numpy())
        scores = torch.sigmoid(sentence).float().masked_fill(~visible.unsqueeze(-1), float("nan"))
        scores = scores.cpu().numpy()
        sentences += [scores[b, :n] for b, n in enumerate(batch["n_sentences"])]
    return np.concatenate(probs), sentences, (total / count if count else float("nan"))


def train_and_predict(cfg: ExperimentConfig, parts: dict, gold, labels: list[str], seed: int,
                      device: torch.device, max_epochs: int | None = None, log=print):
    """transformer.train_and_predict for the sentence model; parts need text and sentences.
    Returns (model, tokenizer, {part: probs}, {part: sentence score arrays}, meta)."""
    if labels[-1] != IPC:
        raise ValueError("the sentence model expects the 12 positions, then IPC")
    positions = labels[:-1]
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.padding_side != "right":
        raise ValueError("sentence ids are laid out for right padding")

    loaders, joined, evidence_pairs = {}, {}, 0
    for k, df in parts.items():
        items, joined[k] = _items(tokenizer, df["text"], df["sentences"], cfg.max_length)
        targets = target_matrix(df["passage_id"], gold, labels, "hard")
        weights = (confidence_weight_matrix(df["passage_id"], gold, labels)
                   if cfg.confidence_weighting == "weights" else np.ones_like(targets))
        sets = None
        if k in ("train", "val"):  # test, lockbox: predictions only
            sets = sentence_targets(df["passage_id"], [it["n_sentences"] for it in items], gold,
                                    positions, cfg.confidence_weighting)
            if k == "train":
                evidence_pairs = int(sum((t.sum(0) > 0).sum() for t, _, _ in sets))
        loaders[k] = _loader(tokenizer, items, targets, weights, sets, cfg, k == "train", seed)

    model = SentencePassageClassifier(cfg.model_name, len(labels), cfg.dropout,
                                      cfg.passage_from).to(device)
    frozen = freeze_bottom(model, cfg.freeze_layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    epochs = max_epochs or cfg.epochs
    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg.weight_decay, cfg.decay_norm_and_bias), lr=cfg.learning_rate
    )
    scheduler = _scheduler(cfg, optimizer, epochs * len(loaders["train"]))
    val_targets = target_matrix(parts["val"]["passage_id"], gold, labels, "hard")

    best_state, best_score, best_epoch, stale, history = None, -np.inf, 0, 0, []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        sums, count = np.zeros(3), 0
        for batch in loaders["train"]:
            (passage, sentence, visible), tensors = _forward(model, batch, device)
            losses = v3_loss(passage, sentence, visible, tensors, cfg.sentence_loss_weight)
            optimizer.zero_grad()
            losses[0].backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            sums += [v.item() * len(passage) for v in losses]
            count += len(passage)
        val_probs, _, val_loss = _predict(model, loaders["val"], device, cfg.sentence_loss_weight)
        score = _selection_score(cfg.selection_metric, val_probs, val_targets)
        improved = best_state is None or score > best_score + 1e-6
        train_loss, passage_loss, sentence_loss = sums / count
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_passage_loss": passage_loss, "train_sentence_loss": sentence_loss,
                        "val_loss": val_loss, cfg.selection_metric: score,
                        "seconds": round(time.time() - start, 1)})
        log(f"  epoch {epoch:2d}  train_loss {train_loss:.4f} (passage {passage_loss:.4f}, "
            f"sentences {sentence_loss:.4f})  val_loss {val_loss:.4f}  "
            f"{cfg.selection_metric} {score:.4f}  [{time.time() - start:.0f}s]"
            + ("  *" if improved else ""))
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    model.load_state_dict(best_state)
    probs, sentences = {}, {}
    for k in parts:
        if k != "train":
            probs[k], sentences[k], _ = _predict(model, loaders[k], device)
    meta = {
        "best_epoch": best_epoch, "best_score": best_score, "epochs_run": len(history),
        "history": history, "class_weights": None,
        "parameters": sum(p.numel() for p in model.parameters()), "frozen_parameters": frozen,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "joined_passages": joined, "train_evidence_pairs": evidence_pairs,
        "unscored_sentences": {k: int(sum(np.isnan(s[:, 0]).sum() for s in v))
                               for k, v in sentences.items()},
    }
    return model, tokenizer, probs, sentences, meta


def predict_passages(model, tokenizer, texts, sentences, cfg: ExperimentConfig, device
                     ) -> tuple[np.ndarray, list[np.ndarray]]:
    """(passage probs, sentence scores per passage, NaN where truncated) for new passages."""
    items, _ = _items(tokenizer, texts, sentences, cfg.max_length)
    dummy = np.zeros((len(items), model.head.out_features), dtype=np.float32)
    loader = _loader(tokenizer, items, dummy, dummy, None, cfg, False, 0)
    probs, scores, _ = _predict(model, loader, device)
    return probs, scores


def score_sentences_alone(model, tokenizer, sentences: list[list[str]], cfg: ExperimentConfig,
                          device) -> list[np.ndarray]:
    """Score each sentence alone with a passage model, as the v1 app does. One array per passage."""
    flat = [str(s) for ss in sentences for s in ss]
    order = np.argsort([len(s) for s in flat], kind="stable")  # similar lengths batch together
    probs = predict_texts(model, tokenizer, [flat[i] for i in order],
                          replace(cfg, max_length=SENTENCE_TOKENS), device)
    probs = probs[np.argsort(order)][:, :-1]  # back in order; IPC dropped
    out, at = [], 0
    for ss in sentences:
        out.append(probs[at:at + len(ss)])
        at += len(ss)
    return out
