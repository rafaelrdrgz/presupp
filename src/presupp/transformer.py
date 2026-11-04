"""Fine-tune a transformer encoder as a multi-label passage classifier.

Config switches can rerun the v1 recipe.
"""

import random
import time

import numpy as np
import torch
import transformers
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from presupp.experiment import ExperimentConfig
from presupp.labels import target_matrix
from presupp.metrics import macro_average_precision, macro_f1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(name)


class PassageClassifier(nn.Module):
    def __init__(self, model_name: str, n_labels: int, dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.dropout(hidden[:, 0]))


def freeze_bottom(model: PassageClassifier, n_layers: int) -> int:
    """Freeze embeddings and the bottom n_layers. Returns the frozen parameter count."""
    if n_layers == 0:
        return 0
    embeddings = getattr(model.encoder, "embeddings", None)
    layers = getattr(getattr(model.encoder, "encoder", None), "layer", None)
    if embeddings is None or layers is None:
        raise ValueError(f"Can't freeze layers of {type(model.encoder).__name__}")
    if n_layers > len(layers):
        raise ValueError(f"freeze_layers={n_layers} but the model has {len(layers)} layers")
    frozen = 0
    for module in [embeddings, *layers[:n_layers]]:
        for p in module.parameters():
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def class_weights(train_targets: np.ndarray, cap: float) -> np.ndarray:
    """v1's pos_weight: negatives / positives per label, capped."""
    positives = (train_targets > 0).sum(0)
    negatives = len(train_targets) - positives
    weights = np.where(positives > 0, negatives / np.maximum(positives, 1), cap)
    return np.minimum(weights, cap).astype(np.float32)


def _param_groups(model: nn.Module, weight_decay: float, decay_everything: bool) -> list[dict]:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if decay_everything:
        return [{"params": [p for _, p in named], "weight_decay": weight_decay}]

    def no_decay(name):
        return name.endswith("bias") or "LayerNorm" in name or "layer_norm" in name
    return [
        {"params": [p for n, p in named if not no_decay(n)], "weight_decay": weight_decay},
        {"params": [p for n, p in named if no_decay(n)], "weight_decay": 0.0},
    ]


def _scheduler(cfg: ExperimentConfig, optimizer, total_steps: int):
    if cfg.scheduler == "onecycle_cosine":  # v1
        return OneCycleLR(optimizer, max_lr=cfg.learning_rate, total_steps=total_steps,
                          pct_start=cfg.warmup_ratio, anneal_strategy="cos")
    return get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps),
                                           total_steps)


def _loader(tokenizer, texts, targets, cfg, shuffle: bool, seed: int) -> DataLoader:
    encoded = tokenizer(list(texts), truncation=True, max_length=cfg.max_length)["input_ids"]
    targets = torch.as_tensor(targets, dtype=torch.float32)

    def collate(indices):
        batch = tokenizer.pad([{"input_ids": encoded[i]} for i in indices], return_tensors="pt")
        batch["labels"] = targets[indices]
        return batch

    generator = torch.Generator().manual_seed(seed)
    size = cfg.batch_size if shuffle else cfg.batch_size * 4
    return DataLoader(list(range(len(encoded))), batch_size=size, shuffle=shuffle,
                      generator=generator, collate_fn=collate)


@torch.no_grad()
def _predict(model, loader, device, loss_fn=None) -> tuple[np.ndarray, float]:
    model.eval()
    probs, total, count = [], 0.0, 0
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        if loss_fn is not None:
            labels = batch["labels"].to(device)
            total += loss_fn(logits, labels).item() * len(labels)
            count += len(labels)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs), (total / count if count else float("nan"))


def _selection_score(metric: str, probs: np.ndarray, targets: np.ndarray) -> float:
    hard = targets > 0
    if metric == "macro_pr_auc":
        return macro_average_precision(hard, probs)
    return macro_f1(hard, probs > 0.5)  # v1


def train_and_predict(cfg: ExperimentConfig, parts: dict, gold, labels: list[str], seed: int,
                      device: torch.device, max_epochs: int | None = None,
                      log=print) -> tuple[PassageClassifier, object, dict, dict]:
    """Train on parts['train'], pick the epoch on parts['val'], predict the rest.
    Returns (model, tokenizer, {part: probs}, meta)."""
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    targets = {k: target_matrix(df["passage_id"], gold, labels, cfg.label_mode)
               for k, df in parts.items()}
    loaders = {k: _loader(tokenizer, df["text"], targets[k], cfg, k == "train", seed)
               for k, df in parts.items()}

    model = PassageClassifier(cfg.model_name, len(labels), cfg.dropout).to(device)
    frozen = freeze_bottom(model, cfg.freeze_layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    weights = (class_weights(targets["train"], cfg.max_class_weight)
               if cfg.class_weighting == "inverse_capped" else None)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=None if weights is None else torch.as_tensor(weights, device=device)
    )
    epochs = max_epochs or cfg.epochs
    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg.weight_decay, cfg.decay_norm_and_bias), lr=cfg.learning_rate
    )
    scheduler = _scheduler(cfg, optimizer, epochs * len(loaders["train"]))

    best_state, best_score, best_epoch, stale, history = None, -np.inf, 0, 0, []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total, count = 0.0, 0
        for batch in loaders["train"]:
            labels_t = batch["labels"].to(device)
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = loss_fn(logits, labels_t)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(labels_t)
            count += len(labels_t)
        val_probs, val_loss = _predict(model, loaders["val"], device, loss_fn)
        score = _selection_score(cfg.selection_metric, val_probs, targets["val"])
        improved = best_state is None or score > best_score + 1e-6
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss,
                        cfg.selection_metric: score, "seconds": round(time.time() - start, 1)})
        log(f"  epoch {epoch:2d}  train_loss {total / count:.4f}  val_loss {val_loss:.4f}  "
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
    probs = {k: _predict(model, loaders[k], device)[0] for k in parts if k != "train"}
    meta = {
        "best_epoch": best_epoch, "best_score": best_score, "epochs_run": len(history),
        "history": history, "class_weights": None if weights is None else weights.tolist(),
        "parameters": sum(p.numel() for p in model.parameters()), "frozen_parameters": frozen,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    return model, tokenizer, probs, meta


def predict_texts(model, tokenizer, texts, cfg: ExperimentConfig, device) -> np.ndarray:
    dummy = np.zeros((len(texts), model.head.out_features), dtype=np.float32)
    return _predict(model, _loader(tokenizer, texts, dummy, cfg, False, 0), device)[0]
