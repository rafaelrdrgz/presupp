"""Paired comparisons between experiments, scored on the same development passages.

    python -m presupp.compare                       # the four main comparisons
    python -m presupp.compare --v3 --out reports/results/v3    # v3 against its baseline
    python -m presupp.compare --pair configs/a.yaml configs/b.yaml

Each article resample scores both experiments; the interval is for A minus B.
"""

import argparse
from pathlib import Path

import numpy as np

from presupp import paths
from presupp.data.dataset import load_dev
from presupp.evaluate import N_BOOT, RESULTS_DIR, load_label_support, load_seeds
from presupp.experiment import load_config
from presupp.labels import model_labels, target_matrix
from presupp.metrics import average_precision_by_label, bootstrap_samples, f1_by_label
from presupp.runs import RUNS_DIR
from presupp.taxonomy import load_taxonomy

CONFIGS_DIR = paths.ROOT / "configs"
METRICS = ["macro-F1 (all)", "macro-F1 (>= 20 articles)", "avg precision"]

# (A, B, question): a positive difference means "yes"
PLANNED = [
    ("v1_recipe_passage_cv", "v1_recipe_grouped_cv",
     "Did article leakage inflate the v1 recipe's score?"),
    ("fixed_weighted", "fixed_unweighted", "Do class weights help?"),
    ("fixed_unweighted", "v1_recipe_grouped_cv",
     "Do the v2 fixes beat the v1 recipe, both without leakage?"),
    ("fixed_unweighted", "tfidf_lr", "Does the transformer beat TF-IDF?"),
]
# re-marked data: baseline is fixed_unweighted retrained on it
V3 = [
    ("v3_joint", "fixed_unweighted_pointing",
     "Does training on the sentences and confidence change the passage scores?"),
    ("v3_mil", "fixed_unweighted_pointing",
     "Does deciding by the best sentence change the passage scores?"),
    ("v3_mil", "v3_joint", "Is deciding by the best sentence better than the first token?"),
]


def statistics(y: np.ndarray, seeds: dict, measurable: np.ndarray, idx=slice(None)) -> np.ndarray:
    """Seed-averaged [macro-F1, macro-F1 over measurable labels, macro AP]."""
    y = y[idx]
    out = []
    for _, probs, decisions in seeds.values():
        f1, present = f1_by_label(y, decisions[idx])
        ap = average_precision_by_label(y, probs[idx])
        out.append([f1[present].mean(), f1[present & measurable].mean(), np.nanmean(ap)])
    return np.mean(out, axis=0)


def compare(experiments: dict[str, dict], pairs: list[tuple[str, str]], y: np.ndarray,
            groups: np.ndarray, measurable: np.ndarray, n_boot: int = N_BOOT) -> list[dict]:
    """Per pair: observed difference, 95% interval, share of resamples with A ahead."""
    observed = {name: statistics(y, seeds, measurable) for name, seeds in experiments.items()}
    boot = {name: [] for name in experiments}
    for idx in bootstrap_samples(groups, n_boot, seed=0):
        for name, seeds in experiments.items():
            boot[name].append(statistics(y, seeds, measurable, idx))
    rows = []
    for a, b in pairs:
        diffs = np.asarray(boot[a]) - np.asarray(boot[b])
        lo, hi = np.nanquantile(diffs, [0.025, 0.975], axis=0)
        rows.append({"a": a, "b": b, "difference": observed[a] - observed[b],
                     "ci95": np.stack([lo, hi], 1), "a_ahead": np.mean(diffs > 0, axis=0)})
    return rows


def render(rows: list[dict], questions: dict,
           where: str = "Development cross-validation (lockbox not used).") -> str:
    out = ["# Paired comparisons", "",
           f"{where} Each row is A minus B, scored on the same passages: the observed difference "
           "[95% interval from resampling articles], and the share of resamples in which A is "
           "ahead. An interval that excludes 0 is a difference the data can detect.", "",
           "| question | A − B | " + " | ".join(METRICS) + " |",
           "|---|---|" + "---|" * len(METRICS)]
    for r in rows:
        cells = [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}], {share:.0%}"
                 for d, (lo, hi), share in zip(r["difference"], r["ci95"], r["a_ahead"],
                                               strict=True)]
        question = questions.get((r["a"], r["b"]), "")
        out.append(f"| {question} | {r['a']} − {r['b']} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparisons between experiments.")
    parser.add_argument("--pair", nargs=2, type=Path, action="append", metavar=("A", "B"),
                        help="two config files; repeat for more pairs (default: the main four)")
    parser.add_argument("--v3", action="store_true", help="the v3 comparisons instead")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    questions = {(a, b): q for a, b, q in PLANNED + V3}
    pair_paths = args.pair or [(CONFIGS_DIR / f"{a}.yaml", CONFIGS_DIR / f"{b}.yaml")
                               for a, b, _ in (V3 if args.v3 else PLANNED)]
    configs = {}
    for path in (p for pair in pair_paths for p in pair):
        cfg = load_config(path)
        configs[cfg.name] = cfg
    pairs = [(load_config(a).name, load_config(b).name) for a, b in pair_paths]

    labels = model_labels(load_taxonomy("v1"))
    passages, gold = load_dev()
    support = load_label_support()
    experiments, ids = {}, None
    for name, cfg in configs.items():
        seeds, incomplete = load_seeds(cfg, passages, gold, labels, args.runs_dir)
        if incomplete or not seeds:
            raise SystemExit(f"{name}: not every seed has finished ({incomplete or 'no runs'})")
        these = next(iter(seeds.values()))[0]
        if ids is not None and list(these) != list(ids):
            raise SystemExit(f"{name} was scored on different passages from the others")
        ids, experiments[name] = these, seeds

    y = target_matrix(list(ids), gold, labels, "hard") > 0
    measurable = support.reindex(labels)["measurable"].eq(True).to_numpy()
    groups = passages.set_index("passage_id")["group_id"].reindex(ids).to_numpy()
    rows = compare(experiments, pairs, y, groups, measurable)
    text = render(rows, questions)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "comparisons.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
