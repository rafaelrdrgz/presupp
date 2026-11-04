"""Classify new text with a released model (see presupp.release).

    python -m presupp.inference models/presupp-roberta-v3 --text "A paragraph. Another one."
    python -m presupp.inference models/presupp-roberta-v3 --file article.txt

Text is split into passages (presupp.reading), classified, grouped into domains, and each
flagged position gets its evidence sentences (presupp.evidence.point).
"""

import argparse
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from presupp.decisions import decide
from presupp.evidence import SENTENCE_TOKENS, point
from presupp.experiment import ExperimentConfig
from presupp.reading import Reading, read, sentences

# progress(done, total, stage): stage is "passages", then "sentences"
Progress = Callable[[int, int, str], None]

# format 2 (v3) adds the sentence head, val sentence scores and marking_thresholds
RELEASE_FORMATS = {"transformer": 1, "sentence_transformer": 2}
RELEASE_FILE = "release.json"
ENCODER_DIR = "encoder"
HEAD_FILE = "head.safetensors"
SENTENCE_HEAD_FILE = "sentence_head.safetensors"
VAL_FILE = "val_predictions.parquet"
VAL_SENTENCES_FILE = "val_sentences.parquet"

BATCH = 8  # passages per forward pass, and per progress report
SENTENCE_BATCH = 48  # sentences, which are about a seventh as long


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    domain: str
    description: str
    probability: float
    threshold: float
    detected: bool
    dev_f1: float | None  # development F1
    evidence: tuple[int, ...] = ()  # when detected: indices into PassageResult.sentences


@dataclass(frozen=True)
class Domain:
    name: str
    strength: float  # its strongest category's probability
    categories: tuple[Category, ...]


@dataclass(frozen=True)
class PassageResult:
    text: str
    truncated: bool  # only the start was classified
    outcome: str     # "domains", "no_philosophy" (IPC) or "unclear" (nothing flagged)
    domains: tuple[Domain, ...]
    ipc: Category
    closest: tuple[Category, ...]  # when unclear: the categories nearest their thresholds
    categories: tuple[Category, ...] = ()  # all, flagged or not, in taxonomy order
    sentences: tuple[str, ...] = ()  # what Category.evidence indexes


class Classifier:
    def __init__(self, model, tokenizer, release: dict, device: str = "cpu"):
        import torch

        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.release = release
        self.device = torch.device(device)
        self.cfg = ExperimentConfig(**{**release["config"],
                                       "seeds": tuple(release["config"]["seeds"])})
        self.labels = [label["id"] for label in release["labels"]]
        self.info = {label["id"]: label for label in release["labels"]}
        self.thresholds = np.asarray(release["thresholds"], dtype=float)
        self.ipc_index = self.labels.index(release["ipc"])
        development = release.get("reliability", {}).get("development", {})
        self.dev_f1 = {r["label"]: r["f1"] for r in development.get("per_label", [])}
        # otherwise sentences are scored separately, in find_evidence
        self.reads_sentences = self.cfg.model_type == "sentence_transformer"
        # per-label threshold for marking a sentence
        self.marking = (np.asarray(release["marking_thresholds"], dtype=float)
                        if self.reads_sentences else self.thresholds)

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> "Classifier":
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        from presupp.transformer import PassageClassifier

        directory = Path(directory)
        with open(directory / RELEASE_FILE, encoding="utf-8") as f:
            release = json.load(f)
        model_type = release["config"]["model_type"]
        if release.get("format") != RELEASE_FORMATS.get(model_type):
            raise ValueError(f"{directory}: release format {release.get('format')} for a "
                             f"{model_type} model, expected {RELEASE_FORMATS.get(model_type)}")
        encoder = directory / ENCODER_DIR
        n_labels, dropout = len(release["labels"]), release["config"]["dropout"]
        if model_type == "sentence_transformer":
            from presupp.sentence_model import SentencePassageClassifier

            model = SentencePassageClassifier(str(encoder), n_labels, dropout,
                                              release["config"]["passage_from"])
            model.sentence_head.load_state_dict(load_file(directory / SENTENCE_HEAD_FILE))
        else:
            model = PassageClassifier(str(encoder), n_labels, dropout)
        model.head.load_state_dict(load_file(directory / HEAD_FILE))
        return cls(model, AutoTokenizer.from_pretrained(encoder), release, device)

    def probabilities(self, texts: list[str], max_length: int | None = None) -> np.ndarray:
        """Passage probabilities; a passage model takes a shorter max_length for sentences."""
        if self.reads_sentences:
            if max_length is not None:
                raise ValueError("a sentence model reads whole passages")
            return self.predict(texts)[0]
        from presupp.transformer import predict_texts

        cfg = self.cfg if max_length is None else replace(self.cfg, max_length=max_length)
        return predict_texts(self.model, self.tokenizer, texts, cfg, self.device)

    def predict(self, texts: list[str], split: list[list[str]] | None = None
                ) -> tuple[np.ndarray, list[np.ndarray] | None]:
        """(probs, sentence scores or None). Sentence scores are one (sentences, positions)
        array per text; split defaults to reading.sentences."""
        if not self.reads_sentences:
            return self.probabilities(texts), None
        from presupp.sentence_model import predict_passages

        split = [sentences(t) for t in texts] if split is None else split
        return predict_passages(self.model, self.tokenizer, texts, split, self.cfg, self.device)

    def classify(self, text: str, progress: Progress | None = None
                 ) -> tuple[Reading, list[PassageResult]]:
        reading = read(text)
        split = [[s for i in p.blocks for s in sentences(reading.blocks[i].text)]
                 for p in reading.passages]
        results = self.classify_texts([p.text for p in reading.passages], progress, split)
        if not self.reads_sentences:
            results = self.find_evidence(results, split, progress)
        return reading, results

    def classify_texts(self, texts: list[str], progress: Progress | None = None,
                       split: list[list[str]] | None = None) -> list[PassageResult]:
        """Each text as one passage. A sentence model also marks evidence in split's sentences."""
        if not texts:
            return []
        if self.reads_sentences and split is None:
            split = [sentences(t) for t in texts]
        probs, scores = [], []
        for start in range(0, len(texts), BATCH):
            p, s = self.predict(texts[start:start + BATCH],
                                None if split is None else split[start:start + BATCH])
            probs.append(p)
            scores += s or []
            if progress:
                progress(min(start + BATCH, len(texts)), len(texts), "passages")
        probs = np.concatenate(probs)
        decisions = decide(probs, self.thresholds, self.cfg.threshold_policy, self.cfg.decode,
                           self.ipc_index)
        lengths = [len(ids) for ids in self.tokenizer(texts, verbose=False)["input_ids"]]
        results = [self._result(text, p, d, n > self.cfg.max_length)
                   for text, p, d, n in zip(texts, probs, decisions, lengths, strict=True)]
        if self.reads_sentences:
            results = [self._marked(r, ss, s)
                       for r, ss, s in zip(results, split, scores, strict=True)]
        return results

    def find_evidence(self, results: list[PassageResult], split: list[list[str]],
                      progress: Progress | None = None) -> list[PassageResult]:
        """Passage model only: evidence from scoring each of split[k]'s sentences alone."""
        flagged = [k for k, r in enumerate(results) if r.outcome == "domains"]
        flat = [s for k in flagged for s in split[k]]
        # shortest first, to cut padding
        order = sorted(range(len(flat)), key=lambda i: len(flat[i]))
        probs = np.zeros((len(flat), len(self.labels)))
        for start in range(0, len(flat), SENTENCE_BATCH):
            batch = order[start:start + SENTENCE_BATCH]
            probs[batch] = self.probabilities([flat[i] for i in batch], SENTENCE_TOKENS)
            if progress:
                progress(min(start + SENTENCE_BATCH, len(flat)), len(flat), "sentences")
        out, start = list(results), 0
        for k in flagged:
            out[k] = self._marked(results[k], split[k], probs[start:start + len(split[k])])
            start += len(split[k])
        return out

    def _marked(self, result: PassageResult, split: list[str], scores: np.ndarray
                ) -> PassageResult:
        """Add evidence. scores is (sentences, labels), or without IPC, which comes last."""
        scores = np.nan_to_num(scores, nan=-np.inf)  # sentences the model didn't reach
        words = [len(s.split()) for s in split]
        evidence = {}
        for c in (c for d in result.domains for c in d.categories):
            j = self.labels.index(c.id)
            evidence[c.id] = tuple(point(scores[:, j], self.marking[j], words))

        def mark(c: Category) -> Category:
            return replace(c, evidence=evidence.get(c.id, ()))

        domains = tuple(replace(d, categories=tuple(mark(c) for c in d.categories))
                        for d in result.domains)
        return replace(result, domains=domains, sentences=tuple(split),
                       categories=tuple(mark(c) for c in result.categories))

    def _category(self, j: int, probs: np.ndarray, decisions: np.ndarray) -> Category:
        info = self.info[self.labels[j]]
        return Category(id=info["id"], name=info["name"], domain=info["domain"],
                        description=info["description"], probability=float(probs[j]),
                        threshold=float(self.thresholds[j]), detected=bool(decisions[j]),
                        dev_f1=self.dev_f1.get(info["id"]))

    def _result(self, text: str, probs: np.ndarray, decisions: np.ndarray,
                truncated: bool) -> PassageResult:
        categories = [self._category(j, probs, decisions) for j in range(len(self.labels))
                      if j != self.ipc_index]
        by_domain: dict[str, list[Category]] = {}
        for c in sorted((c for c in categories if c.detected), key=lambda c: -c.probability):
            by_domain.setdefault(c.domain, []).append(c)
        domains = tuple(sorted((Domain(name, found[0].probability, tuple(found))
                                for name, found in by_domain.items()),
                               key=lambda d: -d.strength))
        ipc = self._category(self.ipc_index, probs, decisions)
        outcome = "domains" if domains else "no_philosophy" if ipc.detected else "unclear"
        closest = ()
        if outcome == "unclear":
            closest = tuple(sorted(categories,
                                   key=lambda c: -c.probability / max(c.threshold, 1e-9))[:3])
        return PassageResult(text=text, truncated=truncated, outcome=outcome, domains=domains,
                             ipc=ipc, closest=closest, categories=tuple(categories))


def describe(result: PassageResult) -> list[str]:
    if result.outcome == "no_philosophy":
        return [f"No load-bearing philosophical content ({result.ipc.probability:.2f})"]
    if result.outcome == "unclear":
        return ["Nothing cleared its threshold. Closest: " + ", ".join(
            f"{c.name} {c.probability:.2f} (needs {c.threshold:.2f})" for c in result.closest)]
    lines = []
    for domain in result.domains:
        lines.append(f"{domain.name} ({domain.strength:.2f})")
        for c in domain.categories:
            lines.append(f"    {c.name} {c.probability:.2f} (threshold {c.threshold:.2f})")
            if c.evidence and result.sentences:
                lines.append(f'        most in: "{result.sentences[c.evidence[0]][:100]}"')
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify text with a released model.")
    parser.add_argument("model", type=Path, help="a release folder written by presupp.release")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    args = parser.parse_args()
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    reading, results = Classifier.load(args.model).classify(text)
    if reading.problem:
        print(f"Nothing was read: {reading.problem}")
    for i, (passage, result) in enumerate(zip(reading.passages, results, strict=True), 1):
        note = " (truncated: only the start was read)" if result.truncated else ""
        print(f"Passage {i}, {passage.words} words{note}: {result.text[:70]}...")
        print("\n".join("  " + line for line in describe(result)))
    skipped = Counter(b.skip for b in reading.blocks if b.skip)
    if skipped:
        print("Not read: " + ", ".join(f"{n} {reason}" for reason, n in skipped.most_common())
              + (f" ({reading.unread_words} words past the limit)" if reading.unread_words else ""))


if __name__ == "__main__":
    main()
