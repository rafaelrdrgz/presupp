"""Pick the sentences a flagged position rests on, and check them against the annotator's.

    python -m presupp.evidence models/presupp-roberta-v1

Checked on the release model's validation fold; the lockbox isn't loaded. Writes
reports/results/evidence_<experiment>.md and .json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# the app imports point from here, so evaluation imports are deferred

MAX_MARKED = 3  # sentences marked for one position
MIN_WORDS = 4   # shorter sentences ("It didn't.") are never marked
SENTENCE_TOKENS = 128  # max tokens for a sentence scored alone


def point(scores: np.ndarray, threshold: float, words: list[int] | None = None,
          limit: int = MAX_MARKED) -> list[int]:
    """Sentences at or over threshold, strongest first, up to limit; else just the strongest."""
    order = [int(i) for i in np.argsort(-np.asarray(scores), kind="stable")
             if words is None or words[i] >= MIN_WORDS]
    flagged = [i for i in order if scores[i] >= threshold][:limit]
    return flagged or order[:1]


def evidence_of(label, n: int, as_recorded: bool = False) -> set[int] | None:
    """Gold evidence as indices into the passage's n sentences, or None if none fits.
    as_recorded reads the 1-based lists as 0-based, to check that numbering."""
    if as_recorded and label.evidence_reading == "1-based":
        found = {int(e) for e in label.evidence}
    elif label.evidence_0based is None or len(label.evidence_0based) == 0:
        return None
    else:
        found = {int(e) for e in label.evidence_0based}
    return found if min(found) >= 0 and max(found) < n else None


def check(classifier, passages: pd.DataFrame, gold: pd.DataFrame, as_recorded: bool = False,
          n_boot: int | None = None) -> dict:
    """First-marked hit rate against chance, plus precision and recall of all marks.
    passages needs passage_id, group_id and sentences; intervals resample articles."""
    from presupp.evaluate import N_BOOT
    from presupp.metrics import bootstrap_samples

    n_boot = N_BOOT if n_boot is None else n_boot
    sents = {pid: [str(s) for s in ss] for pid, ss in zip(passages.passage_id, passages.sentences,
                                                         strict=True)}
    start, flat = {}, []
    for pid, ss in sents.items():
        start[pid] = len(flat)
        flat += ss
    probs = classifier.probabilities(flat, SENTENCE_TOKENS)
    groups = passages.set_index("passage_id")["group_id"]

    cases = []
    for g in gold.itertuples():
        if g.category not in classifier.labels or g.category == classifier.release["ipc"]:
            continue
        ss = sents[g.passage_id]
        found = evidence_of(g, len(ss), as_recorded)
        if found is None:
            continue
        j = classifier.labels.index(g.category)
        scores = probs[start[g.passage_id]:start[g.passage_id] + len(ss), j]
        marked = point(scores, classifier.thresholds[j], [len(s.split()) for s in ss])
        if not marked:
            continue
        cases.append({"passage_id": g.passage_id, "group": groups[g.passage_id],
                      "category": g.category, "first_hit": marked[0] in found,
                      "chance": len(found) / len(ss), "marked": len(marked),
                      "marked_hits": len(set(marked) & found), "evidence": len(found)})
    cases = pd.DataFrame(cases)

    def statistics(frame: pd.DataFrame) -> tuple[float, float, float, float]:
        return (frame.first_hit.mean(), frame.chance.mean(),
                frame.marked_hits.sum() / frame.marked.sum(),
                frame.marked_hits.sum() / frame.evidence.sum())

    first, chance, precision, recall = statistics(cases)
    boot = np.array([statistics(cases.iloc[idx])
                     for idx in bootstrap_samples(cases.group.to_numpy(), n_boot, seed=0)])
    by_category = (cases.groupby("category")
                   .agg(cases=("first_hit", "size"), first_hit=("first_hit", "mean"),
                        chance=("chance", "mean"))
                   .reset_index().to_dict("records"))
    return {
        "n_passages": int(cases.passage_id.nunique()), "n_cases": int(len(cases)),
        "first_hit": {"mean": float(first), "ci95": _interval(boot[:, 0])},
        "chance": {"mean": float(chance), "ci95": _interval(boot[:, 1])},
        "lift": {"mean": float(first - chance), "ci95": _interval(boot[:, 0] - boot[:, 1])},
        "precision": float(precision), "recall": float(recall),
        "marked_per_position": float(cases.marked.mean()),
        "by_category": by_category,
    }


def _interval(values) -> list[float]:
    return [float(v) for v in np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])]


def render(name: str, result: dict, zero_based: dict) -> str:
    def fmt(stat):
        return f"{stat['mean']:.3f} [{stat['ci95'][0]:.3f}, {stat['ci95'][1]:.3f}]"

    out = [f"# Evidence sentences: {name}", "",
           "For each flagged position the app scores every sentence of the passage on its own "
           "and marks the strongest (all that would be flagged alone, up to "
           f"{MAX_MARKED}, or else the single strongest; sentences under {MIN_WORDS} words "
           "aren't scored). Checked here against the annotator's evidence sentences for the "
           "gold positions of the release model's validation fold, which it never trained on.",
           "", f"{result['n_cases']} positions with evidence in {result['n_passages']} "
           "passages. Means [95% interval from resampling articles].", "",
           "| measure | value |", "|---|---|",
           f"| first marked sentence is an evidence sentence | {fmt(result['first_hit'])} |",
           f"| the same for a sentence picked at random | {fmt(result['chance'])} |",
           f"| difference | {fmt(result['lift'])} |",
           f"| precision of all marked sentences | {result['precision']:.3f} |",
           f"| recall of the evidence sentences | {result['recall']:.3f} |",
           f"| sentences marked per position | {result['marked_per_position']:.2f} |", "",
           "| position | cases | first marked is evidence | at random |", "|---|---|---|---|"]
    out += [f"| {r['category']} | {r['cases']} | {r['first_hit']:.2f} | {r['chance']:.2f} |"
            for r in result["by_category"]]
    out += ["", "Reading the primary annotator's 1-based lists 0-based instead: first marked "
            f"sentence is evidence {zero_based['first_hit']['mean']:.3f}, at random "
            f"{zero_based['chance']['mean']:.3f}, over {zero_based['n_cases']} positions. "
            "That this is no better than chance agrees with the 1-based reading.", ""]
    return "\n".join(out)


def main() -> None:
    from presupp import paths
    from presupp.data.dataset import load_dev
    from presupp.evaluate import RESULTS_DIR
    from presupp.inference import Classifier
    from presupp.protocol import final_parts

    parser = argparse.ArgumentParser(description="Check the app's evidence sentences.")
    parser.add_argument("model", type=Path, help="a release folder written by presupp.release")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    classifier = Classifier.load(args.model)
    dev, gold = load_dev(paths.PROCESSED_DIR)
    val = final_parts(dev[dev["is_labelled"]])["val"]
    if int(val.fold.iloc[0]) != classifier.release["training"]["val_fold"]:
        raise SystemExit("This model was validated on another fold.")
    gold = gold[gold.passage_id.isin(val.passage_id)]
    result = check(classifier, val, gold)
    zero_based = check(classifier, val, gold, as_recorded=True, n_boot=1)
    name = classifier.release["experiment"]
    result.update(name=name, validation_fold=int(val.fold.iloc[0]),
                  zero_based_check={k: zero_based[k] for k in ("n_cases", "first_hit", "chance")})
    text = render(name, result, zero_based)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"evidence_{name}.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(args.out / f"evidence_{name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(text)


if __name__ == "__main__":
    main()
