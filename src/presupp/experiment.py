"""Experiment configuration: one YAML file in configs/ per experiment."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

CHOICES = {
    "model_type": ("transformer", "sentence_transformer", "tfidf_lr", "prior"),
    "passage_from": ("cls", "sentences"),
    "confidence_weighting": ("none", "weights"),
    "split": ("grouped_cv", "passage_cv"),
    "label_mode": ("hard", "soft"),
    "class_weighting": ("none", "inverse_capped"),
    "scheduler": ("linear", "onecycle_cosine"),
    "selection_metric": ("macro_pr_auc", "macro_f1_at_0.5"),
    "threshold_policy": ("min_positives", "v1"),
    "decode": ("ipc_exclusive", "independent"),
}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    model_type: str = "transformer"
    # data
    split: str = "grouped_cv"            # passage_cv ignores articles, as v1 did
    label_mode: str = "hard"             # soft = v1's confidence-weighted targets
    seeds: tuple[int, ...] = (0, 1, 2)
    # model
    model_name: str = "roberta-base"
    max_length: int = 512
    dropout: float = 0.2
    freeze_layers: int = 6               # bottom layers + embeddings frozen
    # optimisation
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.02
    decay_norm_and_bias: bool = False    # v1 decayed every parameter
    scheduler: str = "linear"
    warmup_ratio: float = 0.1
    epochs: int = 20
    patience: int = 5
    class_weighting: str = "none"
    max_class_weight: float = 15.0
    selection_metric: str = "macro_pr_auc"
    # decisions
    threshold_policy: str = "min_positives"
    threshold_min_positives: int = 10
    decode: str = "ipc_exclusive"
    # sentences (presupp.sentence_model)
    passage_from: str = "cls"            # sentences = a position's score is its best sentence's
    confidence_weighting: str = "none"   # weights = labels count by confidence
    sentence_loss_weight: float = 1.0
    score_sentences_alone: bool = False  # passage model also scores each sentence alone

    def __post_init__(self):
        for key, allowed in CHOICES.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"{self.name}: {key} must be one of {allowed}")
        if not self.seeds:
            raise ValueError(f"{self.name}: at least one seed is required")
        sentences = self.model_type == "sentence_transformer"
        if sentences and (self.label_mode, self.class_weighting) != ("hard", "none"):
            raise ValueError(f"{self.name}: the sentence model takes hard labels and no class "
                             "weights (confidence_weighting weights the labels instead)")
        if not sentences and (self.passage_from, self.confidence_weighting) != ("cls", "none"):
            raise ValueError(f"{self.name}: passage_from and confidence_weighting are for "
                             "model_type sentence_transformer")
        if self.score_sentences_alone and self.model_type != "transformer":
            raise ValueError(f"{self.name}: score_sentences_alone is for model_type transformer")

    @property
    def scores_sentences(self) -> bool:
        return self.model_type == "sentence_transformer" or self.score_sentences_alone

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        return d


def load_config(path: Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(ExperimentConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path.name}: unknown settings {unknown}")
    if "seeds" in raw:
        raw["seeds"] = tuple(raw["seeds"])
    return ExperimentConfig(**raw)
