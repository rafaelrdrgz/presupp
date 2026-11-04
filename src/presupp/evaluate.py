"""Score experiments from their saved cross-validation runs; write reports/results/.

    python -m presupp.evaluate configs/*.yaml

Thresholds come from each round's val fold; test folds are pooled, so every dev passage is
scored once per seed.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from presupp import paths
from presupp.data.dataset import load_dev
from presupp.decisions import decide, tune_thresholds
from presupp.experiment import ExperimentConfig, load_config
from presupp.labels import IPC, model_labels, target_matrix
from presupp.metrics import (
    bootstrap_samples,
    expected_calibration_error,
    f1_by_label,
    macro_average_precision,
    per_label,
    reliability_table,
)
from presupp.protocol import cv_rounds, labelled_dev
from presupp.runs import (
    RUNS_DIR,
    check_same_data,
    data_fingerprint,
    finished_runs,
    load_run,
    probabilities,
)
from presupp.taxonomy import load_taxonomy

RESULTS_DIR = paths.REPORTS_DIR / "results"
N_BOOT = 1000


@dataclass
class Scored:
    name: str
    description: str
    n_passages: int
    seeds: list[int]
    incomplete: dict = field(default_factory=dict)
    headline: dict = field(default_factory=dict)
    per_label: pd.DataFrame | None = None
    calibration: list = field(default_factory=list)


def load_label_support(path: Path = paths.SPLITS_DIR / "label_support_v1.csv") -> pd.DataFrame:
    return pd.read_csv(path).set_index("label")


def seed_decisions(cfg: ExperimentConfig, runs_dir: Path, gold, labels: list[str],
                   n_rounds: int, fingerprint: str | None = None) -> tuple[dict, dict]:
    """({seed: (ids, probs, decisions)} over the test folds, {unfinished seed: rounds done})."""
    by_seed: dict[int, list] = {}
    for run_dir in finished_runs(runs_dir, cfg.name):
        if fingerprint is not None:
            check_same_data(run_dir, fingerprint)
        meta, preds = load_run(run_dir)
        by_seed.setdefault(meta["seed"], []).append(preds)
    complete, incomplete = {}, {}
    ipc = labels.index(IPC)
    for seed, runs in sorted(by_seed.items()):
        if len(runs) < n_rounds:
            incomplete[seed] = len(runs)
            continue
        ids, probs, decisions = [], [], []
        for preds in runs:
            val, test = preds[preds.part == "val"], preds[preds.part == "test"]
            y_val = target_matrix(val.passage_id.tolist(), gold, labels, "hard")
            thresholds = tune_thresholds(probabilities(val, labels), y_val,
                                         cfg.threshold_policy, cfg.threshold_min_positives)
            p_test = probabilities(test, labels)
            ids += test.passage_id.tolist()
            probs.append(p_test)
            decisions.append(decide(p_test, thresholds, cfg.threshold_policy, cfg.decode, ipc))
        order = np.argsort(ids)
        complete[seed] = (np.asarray(ids)[order], np.concatenate(probs)[order],
                          np.concatenate(decisions)[order])
    return complete, incomplete


def load_seeds(cfg: ExperimentConfig, passages, gold, labels: list[str],
               runs_dir: Path = RUNS_DIR) -> tuple[dict, dict]:
    """seed_decisions for a config on the current data."""
    n_rounds = len(cv_rounds(sorted(labelled_dev(passages, gold, labels, cfg.split)
                                    ["fold"].unique())))
    return seed_decisions(cfg, runs_dir, gold, labels, n_rounds, data_fingerprint(passages, gold))


def score(name: str, description: str, seeds: dict, gold, labels: list[str],
          groups: pd.Series, support: pd.DataFrame, incomplete: dict | None = None,
          n_boot: int = N_BOOT) -> Scored:
    """Mean and sd over seeds, plus article-bootstrap intervals of the seed means."""
    first = next(iter(seeds.values()))
    ids = first[0]
    if any(list(s[0]) != list(ids) for s in seeds.values()):
        raise ValueError(f"{name}: seeds were scored on different passages")
    y = target_matrix(list(ids), gold, labels, "hard") > 0
    measurable = support.reindex(labels)["measurable"].eq(True).to_numpy()
    decisions = [s[2] for s in seeds.values()]
    probs = [s[1] for s in seeds.values()]

    def headline(y_, decisions_):
        f1s = [f1_by_label(y_, d) for d in decisions_]
        macro = np.mean([_masked_mean(f1, present) for f1, present in f1s])
        macro_m = np.mean([_masked_mean(f1, present & measurable) for f1, present in f1s])
        micro = np.mean([_micro(y_, d) for d in decisions_])
        return macro, macro_m, micro

    per_seed = {
        "macro_f1": [], "macro_f1_measurable": [], "micro_f1": [], "macro_average_precision": [],
        "calibration_error": [], "share_with_no_label": [],
    }
    for p, d in zip(probs, decisions, strict=True):
        macro, macro_m, micro = headline(y, [d])
        per_seed["macro_f1"].append(macro)
        per_seed["macro_f1_measurable"].append(macro_m)
        per_seed["micro_f1"].append(micro)
        per_seed["macro_average_precision"].append(macro_average_precision(y, p))
        per_seed["calibration_error"].append(expected_calibration_error(y, p))
        per_seed["share_with_no_label"].append(float((~d.any(1)).mean()))

    group_ids = groups.reindex(ids).to_numpy()
    boot_headline, boot_labels = [], []
    for idx in bootstrap_samples(group_ids, n_boot, seed=0):
        boot_headline.append(headline(y[idx], [d[idx] for d in decisions]))
        boot_labels.append(np.mean([f1_by_label(y[idx], d[idx])[0] for d in decisions], axis=0))
    boot_headline, boot_labels = np.asarray(boot_headline), np.asarray(boot_labels)

    summary = {}
    for i, key in enumerate(["macro_f1", "macro_f1_measurable", "micro_f1"]):
        lo, hi = np.nanquantile(boot_headline[:, i], [0.025, 0.975])
        summary[key] = {"mean": float(np.mean(per_seed[key])), "sd": float(np.std(per_seed[key])),
                        "ci95": [float(lo), float(hi)]}
    for key in ["macro_average_precision", "calibration_error", "share_with_no_label"]:
        summary[key] = {"mean": float(np.mean(per_seed[key])), "sd": float(np.std(per_seed[key]))}
    summary["labels_in_macro"] = int(y.any(0).sum())
    summary["labels_in_measurable_macro"] = int((y.any(0) & measurable).sum())

    rows = []
    stats = [per_label(y, d, p) for d, p in zip(decisions, probs, strict=True)]
    for j, label in enumerate(labels):
        lo, hi = np.nanquantile(boot_labels[:, j], [0.025, 0.975])
        rows.append({
            "label": label,
            "articles": int(support.loc[label, "articles"]) if label in support.index else 0,
            "measurable": bool(measurable[j]),
            "dev_positives": int(y[:, j].sum()),
            "precision": float(np.mean([s["precision"][j] for s in stats])),
            "recall": float(np.mean([s["recall"][j] for s in stats])),
            "f1": float(np.mean([s["f1"][j] for s in stats])),
            "f1_ci_low": float(lo), "f1_ci_high": float(hi),
            "average_precision": _nanmean([s["average_precision"][j] for s in stats]),
        })
    return Scored(name=name, description=description, n_passages=len(ids),
                  seeds=sorted(seeds), incomplete=incomplete or {}, headline=summary,
                  per_label=pd.DataFrame(rows), calibration=reliability_table(y, np.mean(probs, 0)))


def _nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else float("nan")


def _micro(y: np.ndarray, d: np.ndarray) -> float:
    tp = (y & d).sum()
    denominator = 2 * tp + (~y & d).sum() + (y & ~d).sum()
    return float(2 * tp / denominator) if denominator else 0.0


def _fmt(stat: dict) -> str:
    text = f"{stat['mean']:.3f}"
    if "sd" in stat:
        text += f" ± {stat['sd']:.3f}"
    if "ci95" in stat:
        text += f" [{stat['ci95'][0]:.3f}, {stat['ci95'][1]:.3f}]"
    return text


def render(result: Scored, taxonomy, title_prefix: str = "") -> str:
    h = result.headline
    out = [f"# {title_prefix}{result.name}", "", result.description.strip(), ""]
    out.append(f"Passages scored: {result.n_passages}; seeds: {result.seeds}. Numbers are "
               "mean ± sd across seeds [95% interval from resampling articles].")
    if result.incomplete:
        out.append(f"\n**Incomplete seeds left out** (rounds finished): {result.incomplete}")
    out += ["", "| metric | value |", "|---|---|",
            f"| macro-F1, all {h['labels_in_macro']} labels | {_fmt(h['macro_f1'])} |",
            f"| macro-F1, {h['labels_in_measurable_macro']} labels in >= 20 articles | "
            f"{_fmt(h['macro_f1_measurable'])} |",
            f"| micro-F1 | {_fmt(h['micro_f1'])} |",
            f"| macro average precision (threshold-free) | "
            f"{_fmt(h['macro_average_precision'])} |",
            f"| calibration error (lower is better) | {_fmt(h['calibration_error'])} |",
            f"| passages with no label predicted | {_fmt(h['share_with_no_label'])} |", ""]
    out += ["| label | articles | positives | precision | recall | F1 [95% interval] | "
            "average precision |", "|---|---|---|---|---|---|---|"]
    for r in result.per_label.itertuples():
        name = taxonomy.get(r.label).name + ("" if r.measurable else " (†)")
        out.append(f"| {name} | {r.articles} | {r.dev_positives} | {r.precision:.3f} | "
                   f"{r.recall:.3f} | {r.f1:.3f} [{r.f1_ci_low:.3f}, {r.f1_ci_high:.3f}] | "
                   f"{r.average_precision:.3f} |")
    out += ["", "(†) in fewer than 20 articles: too little evidence to measure reliably.", ""]
    return "\n".join(out)


def render_summary(results: list[Scored]) -> str:
    out = ["# Results summary", "",
           "Development cross-validation (lockbox not used). mean ± sd [95% interval].", "",
           "| experiment | macro-F1 (all) | macro-F1 (>= 20 articles) | micro-F1 | "
           "avg precision | calibration error |", "|---|---|---|---|---|---|"]
    for r in results:
        h = r.headline
        out.append(f"| {r.name} | {_fmt(h['macro_f1'])} | {_fmt(h['macro_f1_measurable'])} | "
                   f"{_fmt(h['micro_f1'])} | {_fmt(h['macro_average_precision'])} | "
                   f"{_fmt(h['calibration_error'])} |")
    return "\n".join(out) + "\n"


def write(result: Scored, taxonomy, directory: Path, prefix: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"{prefix}{result.name}.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(render(result, taxonomy, "Lockbox: " if prefix else ""))
    payload = {"name": result.name, "n_passages": result.n_passages, "seeds": result.seeds,
               "incomplete": result.incomplete, "headline": result.headline,
               "per_label": result.per_label.to_dict("records"),
               "calibration": result.calibration}
    with open(directory / f"{prefix}{result.name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score experiments' cross-validation runs.")
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    taxonomy = load_taxonomy("v1")
    labels = model_labels(taxonomy)
    passages, gold = load_dev()
    groups = passages.set_index("passage_id")["group_id"]
    support = load_label_support()
    results = []
    for path in args.configs:
        cfg = load_config(path)
        seeds, incomplete = load_seeds(cfg, passages, gold, labels, args.runs_dir)
        if not seeds:
            print(f"{cfg.name}: no complete seeds yet ({incomplete or 'no runs'}), skipping")
            continue
        result = score(cfg.name, cfg.description, seeds, gold, labels, groups, support, incomplete)
        write(result, taxonomy, args.out)
        results.append(result)
        print(f"{cfg.name}: macro-F1 {_fmt(result.headline['macro_f1'])}")
    if results:
        with open(args.out / "summary.md", "w", encoding="utf-8", newline="\n") as f:
            f.write(render_summary(results))


if __name__ == "__main__":
    main()
