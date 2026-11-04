"""Run an experiment's cross-validation: every (round, seed) on the development folds.

    python -m presupp.train configs/fixed_unweighted.yaml
    python -m presupp.train configs/fixed_unweighted.yaml --rounds 0 --seeds 0 --max-epochs 1

Writes runs/<experiment>/roundRR_seedS/; finished runs are skipped. Never loads the lockbox.
"""

import argparse
import time
from pathlib import Path

from presupp import paths
from presupp.data.dataset import load_dev
from presupp.experiment import ExperimentConfig, load_config
from presupp.labels import model_labels
from presupp.protocol import cv_rounds, labelled_dev, round_parts
from presupp.runs import (
    RUNS_DIR,
    check_same_data,
    check_same_evidence,
    cv_run_name,
    data_fingerprint,
    evidence_fingerprint,
    git_state,
    is_done,
    save_run,
)
from presupp.taxonomy import load_taxonomy


def fit_round(cfg: ExperimentConfig, parts: dict, gold, labels: list[str], seed: int,
              device: str, max_epochs: int | None) -> tuple[dict, dict, dict | None]:
    """({part: probs}, meta, {part: sentence scores} or None if no sentences are scored)."""
    if cfg.model_type == "sentence_transformer":
        from presupp.sentence_model import train_and_predict
        from presupp.transformer import resolve_device
        _, _, probs, sentences, meta = train_and_predict(cfg, parts, gold, labels, seed,
                                                         resolve_device(device), max_epochs)
        return probs, meta, sentences
    if cfg.model_type == "transformer":
        from presupp.transformer import resolve_device, train_and_predict
        model, tokenizer, probs, meta = train_and_predict(cfg, parts, gold, labels, seed,
                                                          resolve_device(device), max_epochs)
        sentences = None
        if cfg.score_sentences_alone:
            from presupp.sentence_model import score_sentences_alone
            sentences = {k: score_sentences_alone(model, tokenizer, list(parts[k]["sentences"]),
                                                  cfg, next(model.parameters()).device)
                         for k in probs}
        return probs, meta, sentences
    from presupp.baselines import fit_predict
    return *fit_predict(cfg.model_type, parts, gold, labels, seed), None


def save_fitted(run_dir: Path, parts: dict, labels: list[str], probs: dict, meta: dict,
                sentences: dict | None, gold) -> None:
    """save_run for fit_round's output."""
    if sentences is not None:
        meta["evidence_fingerprint"] = evidence_fingerprint(gold)
    ids = {k: parts[k]["passage_id"].tolist() for k in probs}
    save_run(run_dir, {k: (ids[k], p) for k, p in probs.items()}, labels, meta,
             None if sentences is None else {k: (ids[k], s) for k, s in sentences.items()})


def run_experiment(cfg: ExperimentConfig, runs_dir: Path = RUNS_DIR,
                   processed_dir: Path = paths.PROCESSED_DIR, rounds: list[int] | None = None,
                   seeds: list[int] | None = None, device: str = "auto",
                   max_epochs: int | None = None) -> list[Path]:
    labels = model_labels(load_taxonomy("v1"))
    passages, gold = load_dev(processed_dir)
    data = labelled_dev(passages, gold, labels, cfg.split)
    all_rounds = cv_rounds(sorted(data["fold"].unique()))
    chosen = [r for r in all_rounds if rounds is None or r.index in rounds]
    fingerprint = data_fingerprint(passages, gold)
    written = []
    for seed in seeds if seeds is not None else cfg.seeds:
        for rnd in chosen:
            run_dir = runs_dir / cfg.name / cv_run_name(rnd.index, seed)
            if is_done(run_dir):
                check_same_data(run_dir, fingerprint)
                if cfg.model_type == "sentence_transformer":  # it trains on the evidence
                    check_same_evidence(run_dir, evidence_fingerprint(gold))
                print(f"[{cfg.name}] round {rnd.index} seed {seed}: already done, skipping")
                continue
            parts = round_parts(data, rnd)
            print(f"[{cfg.name}] round {rnd.index} seed {seed}: train {len(parts['train'])}, "
                  f"val {len(parts['val'])} (fold {rnd.val_fold}), "
                  f"test {len(parts['test'])} (fold {rnd.test_fold})", flush=True)
            start = time.time()
            probs, meta, sentences = fit_round(cfg, parts, gold, labels, seed, device, max_epochs)
            meta.update({
                "experiment": cfg.name, "config": cfg.to_dict(), "seed": seed,
                "round": rnd.index, "train_folds": list(rnd.train_folds),
                "val_fold": rnd.val_fold, "test_fold": rnd.test_fold, "labels": labels,
                "sizes": {k: len(df) for k, df in parts.items()},
                "max_epochs_override": max_epochs, "data_fingerprint": fingerprint,
                "git": git_state(), "seconds": round(time.time() - start, 1),
            })
            save_fitted(run_dir, parts, labels, probs, meta, sentences, gold)
            written.append(run_dir)
            print(f"[{cfg.name}] round {rnd.index} seed {seed}: done in {meta['seconds']:.0f}s",
                  flush=True)
    return written


def _ints(text: str | None) -> list[int] | None:
    return None if text is None else [int(v) for v in text.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experiment's cross-validation.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--rounds", help="comma-separated round indices (default: all)")
    parser.add_argument("--seeds", help="comma-separated seeds (default: the config's)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-epochs", type=int, help="override, for smoke tests only")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg, args.runs_dir, rounds=_ints(args.rounds), seeds=_ints(args.seeds),
                   device=args.device, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
