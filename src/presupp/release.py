"""Train and save the model behind the demo app.

    python -m presupp.release configs/v3_mil.yaml --seed 0
    python -m presupp.release configs/v3_mil.yaml --reliability-only

Trained like presupp.final's lockbox models: on every dev fold but the highest-numbered, which
picks the epoch and thresholds. Never loads the lockbox. --out and --results default by model
type (DEFAULTS).
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from presupp import paths
from presupp.data.dataset import load_dev
from presupp.decisions import tune_thresholds
from presupp.evaluate import RESULTS_DIR
from presupp.experiment import ExperimentConfig, load_config
from presupp.inference import (
    ENCODER_DIR,
    HEAD_FILE,
    RELEASE_FILE,
    RELEASE_FORMATS,
    SENTENCE_HEAD_FILE,
    VAL_FILE,
    VAL_SENTENCES_FILE,
)
from presupp.labels import IPC, model_labels, target_matrix
from presupp.metrics import macro_average_precision
from presupp.protocol import final_parts
from presupp.runs import (
    PROB_PREFIX,
    RUNS_DIR,
    data_fingerprint,
    git_state,
    is_done,
    load_run,
    probabilities,
)
from presupp.taxonomy import load_taxonomy

# model_type -> (release dir, results dir for reliability)
DEFAULTS = {"transformer": (paths.ROOT / "models" / "presupp-roberta-v1", RESULTS_DIR),
            "sentence_transformer": (paths.ROOT / "models" / "presupp-roberta-v3",
                                     RESULTS_DIR / "v3")}
PER_LABEL_FIELDS = ("label", "articles", "measurable", "dev_positives", "precision", "recall",
                    "f1", "f1_ci_low", "f1_ci_high")


def train_release(cfg: ExperimentConfig, seed: int, out: Path,
                  processed_dir: Path = paths.PROCESSED_DIR, results_dir: Path = RESULTS_DIR,
                  runs_dir: Path = RUNS_DIR, device: str = "auto",
                  max_epochs: int | None = None) -> dict:
    """Train, tune thresholds on val, save to out. Returns release.json's contents."""
    if cfg.model_type not in RELEASE_FORMATS:
        raise ValueError(f"{cfg.name}: only transformer experiments can be released")
    from safetensors.torch import save_file

    from presupp.transformer import resolve_device

    taxonomy = load_taxonomy("v1")
    labels = model_labels(taxonomy)
    dev, gold = load_dev(processed_dir)
    dev = dev[dev["is_labelled"]]
    parts = final_parts(dev)
    val_ids = parts["val"]["passage_id"].tolist()
    print(f"Training on {len(parts['train'])} passages, validating on fold "
          f"{int(parts['val']['fold'].iloc[0])} ({len(val_ids)}).", flush=True)

    start = time.time()
    reads_sentences = cfg.model_type == "sentence_transformer"
    if reads_sentences:
        from presupp.sentence_model import train_and_predict
        model, tokenizer, probs, sentence_scores, meta = train_and_predict(
            cfg, parts, gold, labels, seed, resolve_device(device), max_epochs)
    else:
        from presupp.transformer import train_and_predict
        model, tokenizer, probs, meta = train_and_predict(cfg, parts, gold, labels, seed,
                                                          resolve_device(device), max_epochs)
    y_val = target_matrix(val_ids, gold, labels, "hard")
    thresholds = tune_thresholds(probs["val"], y_val, cfg.threshold_policy,
                                 cfg.threshold_min_positives)

    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out / ENCODER_DIR)
    tokenizer.save_pretrained(out / ENCODER_DIR)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()},
              out / HEAD_FILE)
    frame = pd.DataFrame({"passage_id": val_ids})
    for j, label in enumerate(labels):
        frame[PROB_PREFIX + label] = probs["val"][:, j].astype(np.float32)
    frame.to_parquet(out / VAL_FILE, index=False)
    sentence_sections = {}
    if reads_sentences:
        save_file({k: v.detach().cpu().contiguous()
                   for k, v in model.sentence_head.state_dict().items()}, out / SENTENCE_HEAD_FILE)
        sentence_sections = marking(sentence_scores["val"], parts["val"], gold, labels, out)

    release = {
        "format": RELEASE_FORMATS[cfg.model_type],
        "experiment": cfg.name,
        "seed": seed,
        "config": cfg.to_dict(),
        "taxonomy_version": taxonomy.version,
        "labels": [{"id": label, "name": taxonomy.get(label).name,
                    "domain": taxonomy.domain_of(label),
                    "description": taxonomy.get(label).description} for label in labels],
        "ipc": IPC,
        "thresholds": [float(t) for t in thresholds],
        **sentence_sections,
        "training": {
            "best_epoch": meta["best_epoch"], "epochs_run": meta["epochs_run"],
            "selection_metric": cfg.selection_metric, "best_score": meta["best_score"],
            "history": meta["history"], "device": meta["device"], "versions": meta["versions"],
            "val_fold": int(parts["val"]["fold"].iloc[0]),
            "sizes": {k: len(df) for k, df in parts.items()},
            "data_fingerprint": data_fingerprint(dev, gold),
            "seconds": round(time.time() - start, 1),
            "max_epochs_override": max_epochs,
            "compared_with_final_run": compare_with_final(cfg, seed, val_ids, probs["val"],
                                                          thresholds, y_val, labels, runs_dir),
        },
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "reliability": reliability(cfg.name, results_dir),
    }
    write_release_json(out, release)
    return release


def marking(val_scores: list[np.ndarray], val: pd.DataFrame, gold: pd.DataFrame,
            labels: list[str], out: Path) -> dict:
    """Save val sentence scores and tune marking thresholds on them.
    Returns release.json's marking_thresholds and marking sections."""
    from presupp.evaluate_evidence import evidence_cases, tune_on

    if labels[-1] != IPC:
        raise ValueError("a sentence model's labels are the positions, then IPC")
    positions = labels[:-1]
    frames = []
    for pid, scores in zip(val["passage_id"], val_scores, strict=True):
        frame = pd.DataFrame(np.asarray(scores, dtype=np.float32),
                             columns=[PROB_PREFIX + p for p in positions])
        frame.insert(0, "sentence", np.arange(len(frame)))
        frame.insert(0, "passage_id", pid)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(out / VAL_SENTENCES_FILE, index=False)
    cases = evidence_cases(val, gold, positions)
    tuned = tune_on(dict(zip(val["passage_id"], val_scores, strict=True)), cases, positions)
    return {"marking_thresholds": [tuned[p] for p in positions],
            "marking": {"tuned_on": "validation fold", "cases": len(cases)}}


def compare_with_final(cfg: ExperimentConfig, seed: int, val_ids: list[str], val_probs: np.ndarray,
                       thresholds: np.ndarray, y_val: np.ndarray, labels: list[str],
                       runs_dir: Path) -> dict | None:
    """Check against the same-seed final run on the shared val fold. None if that run is missing."""
    run_dir = runs_dir / cfg.name / f"final_seed{seed}"
    if not is_done(run_dir):
        return None
    _, preds = load_run(run_dir)
    val = preds[preds.part == "val"].set_index("passage_id")
    if set(val.index) != set(val_ids):
        return {"run": run_dir.name, "comparable": False}
    final_probs = probabilities(val.loc[val_ids].reset_index(), labels)
    final_thresholds = tune_thresholds(final_probs, y_val, cfg.threshold_policy,
                                       cfg.threshold_min_positives)
    return {
        "run": run_dir.name, "comparable": True,
        "max_abs_probability_difference": float(np.abs(final_probs - val_probs).max()),
        "val_macro_average_precision": {
            "release": macro_average_precision(y_val > 0, val_probs),
            "final_run": macro_average_precision(y_val > 0, final_probs),
        },
        "same_thresholds": bool(np.array_equal(final_thresholds, thresholds)),
    }


def reliability(name: str, results_dir: Path = RESULTS_DIR) -> dict:
    """Evaluation results for this recipe; missing files are skipped. Evidence: CV file first,
    else the val-fold one."""
    out = {}
    for key, filename in [("development", f"{name}.json"), ("lockbox", f"lockbox_{name}.json"),
                          ("domains", f"domains_{name}.json"),
                          ("evidence", f"evidence_cv_{name}.json"),
                          ("evidence", f"evidence_{name}.json")]:
        path = results_dir / filename
        if key in out or not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if key == "domains":
            out[key] = {k: data[k] for k in ("n_passages", "seeds", "domains", "macro_f1",
                                             "exact_match")}
        elif key == "evidence":
            out[key] = {k: data[k] for k in ("n_passages", "n_cases", "first_hit", "chance",
                                             "lift")}
            # CV results have intervals, the fold check bare numbers
            out[key].update({k: data[k]["mean"] if isinstance(data[k], dict) else data[k]
                             for k in ("precision", "recall")})
            out[key]["over"] = ("cross-validation" if filename.startswith("evidence_cv_")
                                else "validation fold")
        else:
            out[key] = {"n_passages": data["n_passages"], "seeds": data["seeds"],
                        "headline": data["headline"],
                        "per_label": [{k: r[k] for k in PER_LABEL_FIELDS}
                                      for r in data["per_label"]]}
    return out


def write_release_json(out: Path, release: dict) -> None:
    with open(out / RELEASE_FILE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(release, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the demo app's model.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="default: by kind of model (see above)")
    parser.add_argument("--results", type=Path, help="default: by kind of model (see above)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-epochs", type=int, help="override, for smoke tests only")
    parser.add_argument("--force", action="store_true", help="replace an existing release")
    parser.add_argument("--reliability-only", action="store_true",
                        help="refresh release.json's reliability section without retraining")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.model_type not in DEFAULTS:
        raise SystemExit(f"{cfg.name}: only transformer experiments can be released")
    args.out = args.out or DEFAULTS[cfg.model_type][0]
    args.results = args.results or DEFAULTS[cfg.model_type][1]

    if args.reliability_only:
        with open(args.out / RELEASE_FILE, encoding="utf-8") as f:
            release = json.load(f)
        if release["experiment"] != cfg.name:
            raise SystemExit(f"{args.out} holds {release['experiment']}, not {cfg.name}")
        release["reliability"] = reliability(cfg.name, args.results)
        write_release_json(args.out, release)
        print(f"Updated the reliability section of {args.out / RELEASE_FILE}")
        return
    if (args.out / RELEASE_FILE).exists() and not args.force:
        raise SystemExit(f"{args.out} already holds a release; pass --force to replace it")

    release = train_release(cfg, args.seed, args.out, results_dir=args.results,
                            device=args.device, max_epochs=args.max_epochs)
    t = release["training"]
    print(f"Saved to {args.out}: best epoch {t['best_epoch']} of {t['epochs_run']}, "
          f"validation {t['selection_metric']} {t['best_score']:.4f}, "
          f"data {t['data_fingerprint']}.")
    check = t["compared_with_final_run"]
    if check and check["comparable"]:
        ap = check["val_macro_average_precision"]
        print(f"Against {check['run']}: largest probability difference "
              f"{check['max_abs_probability_difference']:.2e}, validation macro AP "
              f"{ap['release']:.4f} vs {ap['final_run']:.4f}, "
              f"same thresholds: {check['same_thresholds']}.")


if __name__ == "__main__":
    main()
