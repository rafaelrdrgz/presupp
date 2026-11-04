"""Final evaluation on the lockbox fold.

    python -m presupp.final configs/a.yaml configs/b.yaml --reason "final evaluation"
    python -m presupp.final configs/v3_mil.yaml configs/fixed_unweighted_pointing.yaml \\
        --reason "v3 against its baseline" --compare --out reports/results/v3

Trains on every dev fold but the highest (early stopping, thresholds) and scores the lockbox,
writing <out>/lockbox_*.md. Nothing ever trains on a lockbox passage.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from presupp import compare, evaluate_evidence, paths
from presupp.data.dataset import LOCKBOX_LOG, load_dev, open_lockbox
from presupp.decisions import decide, tune_thresholds
from presupp.evaluate import RESULTS_DIR, Scored, load_label_support, score, write
from presupp.experiment import ExperimentConfig, load_config
from presupp.labels import IPC, model_labels, target_matrix
from presupp.protocol import final_parts
from presupp.runs import (
    RUNS_DIR,
    check_same_data,
    check_same_evidence,
    data_fingerprint,
    evidence_fingerprint,
    git_state,
    is_done,
    load_run,
    probabilities,
)
from presupp.taxonomy import load_taxonomy
from presupp.train import fit_round, save_fitted

LOCKBOX_SCOPE = "The lockbox, after training on the development folds."


def run_final(configs: list[ExperimentConfig], reason: str,
              processed_dir: Path = paths.PROCESSED_DIR, runs_dir: Path = RUNS_DIR,
              results_dir: Path = RESULTS_DIR, log_path: Path = LOCKBOX_LOG,
              support_path: Path = paths.SPLITS_DIR / "label_support_v1.csv",
              device: str = "auto", max_epochs: int | None = None,
              compare_first: bool = False) -> list[Scored]:
    taxonomy = load_taxonomy("v1")
    labels = model_labels(taxonomy)
    ipc = labels.index(IPC)
    dev, dev_gold = load_dev(processed_dir)
    dev = dev[dev["is_labelled"]]
    parts = final_parts(dev)
    val_fold = int(parts["val"]["fold"].iloc[0])
    fingerprint, evidence = data_fingerprint(dev, dev_gold), evidence_fingerprint(dev_gold)
    box, box_gold = open_lockbox(reason, processed_dir, log_path)
    box = box[box["is_labelled"]].sort_values("passage_id", ignore_index=True)
    gold = pd.concat([dev_gold, box_gold], ignore_index=True)
    groups = box.set_index("passage_id")["group_id"]
    parts["lockbox"] = box
    print(f"Lockbox opened: {len(box)} labelled passages. Training on {len(parts['train'])}, "
          f"validating on fold {val_fold} ({len(parts['val'])}).", flush=True)

    support = load_label_support(support_path)
    results, decided = [], {}
    for cfg in configs:
        seeds = {}
        for seed in cfg.seeds:
            run_dir = runs_dir / cfg.name / f"final_seed{seed}"
            if is_done(run_dir):
                check_same_data(run_dir, fingerprint)
                if cfg.model_type == "sentence_transformer":  # it trains on the evidence
                    check_same_evidence(run_dir, evidence)
            else:
                start = time.time()
                probs, meta, sentences = fit_round(cfg, parts, gold, labels, seed, device,
                                                   max_epochs)
                meta.update({"experiment": cfg.name, "config": cfg.to_dict(), "seed": seed,
                             "final": True, "val_fold": val_fold, "labels": labels,
                             "sizes": {k: len(df) for k, df in parts.items()},
                             "data_fingerprint": fingerprint,
                             "git": git_state(), "seconds": round(time.time() - start, 1)})
                save_fitted(run_dir, parts, labels, probs, meta, sentences, dev_gold)
            _, preds = load_run(run_dir)
            val, test = preds[preds.part == "val"], preds[preds.part == "lockbox"]
            y_val = target_matrix(val.passage_id.tolist(), gold, labels, "hard")
            thresholds = tune_thresholds(probabilities(val, labels), y_val,
                                         cfg.threshold_policy, cfg.threshold_min_positives)
            p_box = probabilities(test, labels)
            order = np.argsort(test.passage_id.to_numpy())
            decisions = decide(p_box, thresholds, cfg.threshold_policy, cfg.decode, ipc)
            seeds[seed] = (test.passage_id.to_numpy()[order], p_box[order], decisions[order])
        decided[cfg.name] = seeds
        result = score(cfg.name, cfg.description, seeds, gold, labels, groups, support)
        write(result, taxonomy, results_dir, prefix="lockbox_")
        results.append(result)
        print(f"{cfg.name}: lockbox macro-F1 {result.headline['macro_f1']['mean']:.3f}")

    pairs = [(configs[0].name, cfg.name) for cfg in configs[1:]] if compare_first else []
    if pairs:
        compare_passages(decided, pairs, box_gold, labels, groups, support, results_dir)
    pointing = [cfg for cfg in configs if cfg.scores_sentences]
    if pointing:
        score_evidence(pointing, parts["val"], box, gold, labels, runs_dir, results_dir, pairs)
    return results


def compare_passages(decided: dict[str, dict], pairs: list[tuple[str, str]], gold: pd.DataFrame,
                     labels: list[str], groups: pd.Series, support: pd.DataFrame,
                     results_dir: Path) -> None:
    """presupp.compare's paired comparisons on the lockbox passages."""
    names = sorted({n for pair in pairs for n in pair})
    ids = next(iter(decided[names[0]].values()))[0]
    for name in names:
        if any(list(s[0]) != list(ids) for s in decided[name].values()):
            raise ValueError(f"{name} was scored on different lockbox passages")
    y = target_matrix(list(ids), gold, labels, "hard") > 0
    measurable = support.reindex(labels)["measurable"].eq(True).to_numpy()
    rows = compare.compare({n: decided[n] for n in names}, pairs, y,
                           groups.reindex(ids).to_numpy(), measurable)
    questions = {(a, b): q for a, b, q in compare.PLANNED + compare.V3}
    text = compare.render(rows, questions, LOCKBOX_SCOPE)
    with open(results_dir / "lockbox_comparisons.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


def score_evidence(configs: list[ExperimentConfig], val: pd.DataFrame, box: pd.DataFrame,
                   gold: pd.DataFrame, labels: list[str], runs_dir: Path, results_dir: Path,
                   pairs: list[tuple[str, str]]) -> None:
    """evaluate_evidence on the lockbox. Thresholds come from val, so gold needs both label sets."""
    positions = [label for label in labels if label != IPC]
    cases = evaluate_evidence.evidence_cases(pd.concat([val, box], ignore_index=True), gold,
                                             positions)
    seeds, thresholds = {}, {}
    for cfg in configs:
        seeds[cfg.name], found = evaluate_evidence.score_final_runs(cfg, cases, positions,
                                                                    runs_dir)
        thresholds[cfg.name] = pd.DataFrame(found)
    pairs = [(a, b) for a, b in pairs if a in seeds and b in seeds]
    results, comparisons = evaluate_evidence.evaluate(seeds, box.set_index("passage_id")
                                                      ["group_id"], pairs)
    evaluate_evidence.write_results(results, comparisons, thresholds, results_dir, "lockbox")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on all dev folds, score on the lockbox.")
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--reason", required=True, help="recorded in reports/lockbox_log.md")
    parser.add_argument("--compare", action="store_true",
                        help="compare the first config with each of the others, paired")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    configs = [load_config(p) for p in args.configs]  # fail on a bad config before opening
    if args.compare and len(configs) < 2:
        parser.error("--compare needs at least two configs")
    run_final(configs, args.reason, runs_dir=args.runs_dir, results_dir=args.out,
              device=args.device, compare_first=args.compare)


if __name__ == "__main__":
    main()
