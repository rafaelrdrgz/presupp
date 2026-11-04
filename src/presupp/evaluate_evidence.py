"""How well experiments' sentence scores find the annotated evidence, over cross-validation.

    python -m presupp.evaluate_evidence configs/v3_joint.yaml configs/v3_mil.yaml \\
        configs/fixed_unweighted_pointing.yaml

Marking thresholds are tuned on each run's validation passages. Writes
<out>/evidence_cv_<name>.md and .json, and evidence_cv_comparisons.md (A minus B, paired).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from presupp.data.dataset import load_dev
from presupp.decisions import GRID
from presupp.evaluate import N_BOOT, RESULTS_DIR
from presupp.evidence import MAX_MARKED, MIN_WORDS, evidence_of, point
from presupp.experiment import ExperimentConfig, load_config
from presupp.labels import IPC, model_labels
from presupp.metrics import bootstrap_samples
from presupp.protocol import cv_rounds, labelled_dev
from presupp.runs import (
    PROB_PREFIX,
    RUNS_DIR,
    check_same_data,
    check_same_evidence,
    data_fingerprint,
    evidence_fingerprint,
    finished_runs,
    load_meta,
    load_sentences,
)
from presupp.taxonomy import load_taxonomy

V3_RESULTS = RESULTS_DIR / "v3"
MIN_VAL_CASES = 10
STATISTICS = ["first_hit", "chance", "lift", "precision", "recall", "f1", "sentence_ap",
              "marked_per_position"]
# (A, B, question): a positive difference means "yes"
PAIRS = [
    ("v3_joint", "fixed_unweighted_pointing",
     "Does reading the passage once, trained on the sentences, find the evidence better?"),
    ("v3_mil", "fixed_unweighted_pointing",
     "Does deciding by the best sentence, trained on the sentences, find the evidence better?"),
    ("v3_mil", "v3_joint", "Is deciding by the best sentence better than the first token?"),
]


def evidence_cases(passages: pd.DataFrame, gold: pd.DataFrame, positions: list[str]
                   ) -> pd.DataFrame:
    """Gold positions whose evidence fits the stored sentences, one row each."""
    sentences = passages.set_index("passage_id")["sentences"]
    groups = passages.set_index("passage_id")["group_id"]
    rows = []
    for g in gold[gold["category"].isin(positions)
                  & gold["passage_id"].isin(sentences.index)].itertuples():
        ss = sentences[g.passage_id]
        found = evidence_of(g, len(ss))
        if found is not None:
            rows.append({"passage_id": g.passage_id, "category": g.category,
                         "group": groups[g.passage_id], "evidence": found,
                         "words": [len(str(s).split()) for s in ss]})
    return pd.DataFrame(rows).sort_values(["passage_id", "category"], ignore_index=True)


def _marks(scores: np.ndarray, threshold: float, words: list[int]) -> list[int]:
    return point(np.nan_to_num(scores, nan=-np.inf), threshold, words)  # unscored: never marked


def _f1(cases: list[tuple[np.ndarray, set, list[int]]], threshold: float) -> float:
    hits = marked = evidence = 0
    for scores, found, words in cases:
        chosen = _marks(scores, threshold, words)
        hits += len(set(chosen) & found)
        marked += len(chosen)
        evidence += len(found)
    return 2 * hits / (marked + evidence) if marked + evidence else 0.0


def tune_marking(val_cases: dict[str, list], positions: list[str]) -> dict[str, float]:
    """{position: [(scores, evidence, words)]} -> best-F1 GRID threshold per position.
    Positions with fewer than MIN_VAL_CASES cases share one."""
    def best(cases):
        return float(GRID[int(np.argmax([_f1(cases, t) for t in GRID]))])
    everything = [c for cases in val_cases.values() for c in cases]
    shared = best(everything) if everything else 0.5
    return {p: best(val_cases[p]) if len(val_cases.get(p, [])) >= MIN_VAL_CASES else shared
            for p in positions}


def _by_passage(cases: pd.DataFrame) -> dict[str, list]:
    out: dict[str, list] = {}
    for r in cases.itertuples():
        out.setdefault(r.passage_id, []).append(r)
    return out


def tune_on(scores: dict[str, np.ndarray], cases: pd.DataFrame, positions: list[str]
            ) -> dict[str, float]:
    """tune_marking on the cases of the passages in scores."""
    val_cases: dict[str, list] = {}
    by_passage = _by_passage(cases)
    for pid, s in scores.items():
        for r in by_passage.get(pid, []):
            val_cases.setdefault(r.category, []).append(
                (s[:, positions.index(r.category)], r.evidence, r.words))
    if not val_cases:
        raise ValueError("no case among the passages to tune the marking thresholds on")
    return tune_marking(val_cases, positions)


def _average_precision(scores: np.ndarray, found: set) -> float:
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    hit = np.isin(order, list(found))
    precision = np.cumsum(hit) / np.arange(1, len(order) + 1)
    return float((precision * hit).sum() / len(found))


def _run_scores(sentences: pd.DataFrame, part: str, positions: list[str]) -> dict:
    """{passage_id: (sentences, positions) scores} for one part of a run."""
    rows = sentences[sentences["part"] == part].sort_values(["passage_id", "sentence"])
    columns = [PROB_PREFIX + p for p in positions]
    return {pid: g[columns].to_numpy(dtype=float) for pid, g in rows.groupby("passage_id")}


def _score_run(run_dir: Path, cases: pd.DataFrame, positions: list[str], test_part: str
               ) -> tuple[list[dict], dict[str, float]]:
    """(rows for the test_part cases, thresholds tuned on val)."""
    sentences = load_sentences(run_dir)
    val, test = (_run_scores(sentences, part, positions) for part in ("val", test_part))
    marking = tune_on(val, cases, positions)
    by_passage = _by_passage(cases)
    rows = []
    for pid, scores in test.items():
        for r in by_passage.get(pid, []):
            if len(scores) != len(r.words):
                raise ValueError(f"{run_dir.name}: {pid} has {len(scores)} sentence "
                                 f"scores for {len(r.words)} sentences")
            s = scores[:, positions.index(r.category)]
            chosen = _marks(s, marking[r.category], r.words)
            if not chosen:  # no sentence long enough to mark
                continue
            rows.append({"passage_id": pid, "category": r.category,
                         "first_hit": chosen[0] in r.evidence,
                         "chance": len(r.evidence) / len(r.words),
                         "marked": len(chosen), "hits": len(set(chosen) & r.evidence),
                         "evidence": len(r.evidence),
                         "sentence_ap": _average_precision(s, r.evidence)})
    return rows, marking


def _frame(name: str, seed: int, rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).sort_values(["passage_id", "category"], ignore_index=True)
    if frame.duplicated(["passage_id", "category"]).any():
        raise ValueError(f"{name} seed {seed}: a passage was tested more than once")
    return frame


def score_runs(cfg: ExperimentConfig, cases: pd.DataFrame, positions: list[str],
               runs_dir: Path, n_rounds: int, fingerprints: tuple[str, str]
               ) -> tuple[dict[int, pd.DataFrame], dict[int, int], list[dict]]:
    """({seed: case rows}, {unfinished seed: rounds done}, thresholds per run)."""
    runs: dict[int, list] = {}
    for run_dir in finished_runs(runs_dir, cfg.name):
        check_same_data(run_dir, fingerprints[0])
        check_same_evidence(run_dir, fingerprints[1])
        runs.setdefault(load_meta(run_dir)["seed"], []).append(run_dir)

    complete, incomplete, thresholds = {}, {}, []
    for seed, dirs in sorted(runs.items()):
        if len(dirs) < n_rounds:
            incomplete[seed] = len(dirs)
            continue
        rows = []
        for run_dir in dirs:
            found, marking = _score_run(run_dir, cases, positions, "test")
            rows += found
            thresholds.append({"seed": seed, "run": run_dir.name, **marking})
        complete[seed] = _frame(cfg.name, seed, rows)
    return complete, incomplete, thresholds


def score_final_runs(cfg: ExperimentConfig, cases: pd.DataFrame, positions: list[str],
                     runs_dir: Path) -> tuple[dict[int, pd.DataFrame], list[dict]]:
    """score_runs for presupp.final's runs. cases must cover the val fold and the lockbox."""
    seeds, thresholds = {}, []
    for seed in cfg.seeds:
        run_dir = runs_dir / cfg.name / f"final_seed{seed}"
        rows, marking = _score_run(run_dir, cases, positions, "lockbox")
        seeds[seed] = _frame(cfg.name, seed, rows)
        thresholds.append({"seed": seed, "run": run_dir.name, **marking})
    return seeds, thresholds


def _arrays(seeds: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """(case keys, {quantity: (cases, seeds) array})."""
    frames = list(seeds.values())
    keys = frames[0][["passage_id", "category"]]
    for f in frames[1:]:
        if not f[["passage_id", "category"]].equals(keys):
            raise ValueError("seeds were scored on different positions")
    names = ["first_hit", "chance", "marked", "hits", "evidence", "sentence_ap"]
    return keys, {n: np.stack([f[n].to_numpy(dtype=float) for f in frames], 1) for n in names}


def statistics(q: dict[str, np.ndarray], idx=slice(None)) -> np.ndarray:
    """STATISTICS, each a mean over seeds."""
    hits, marked, evidence = (q[n][idx].sum(0) for n in ("hits", "marked", "evidence"))
    first, chance = q["first_hit"][idx].mean(), q["chance"][idx].mean()
    return np.array([first, chance, first - chance, (hits / marked).mean(),
                     (hits / evidence).mean(), (2 * hits / (marked + evidence)).mean(),
                     q["sentence_ap"][idx].mean(), q["marked"][idx].mean()])


def evaluate(experiments: dict[str, dict[int, pd.DataFrame]], groups: pd.Series,
             pairs: list[tuple[str, str]], n_boot: int = N_BOOT) -> tuple[dict, list[dict]]:
    """(statistics with 95% intervals, paired differences), all on the same resamples."""
    arrays, keys = {}, None
    for name, seeds in experiments.items():
        these, arrays[name] = _arrays(seeds)
        if keys is not None and not these.equals(keys):
            raise ValueError(f"{name} was scored on different positions from the others")
        keys = these
    observed = {name: statistics(q) for name, q in arrays.items()}
    boot = {name: [] for name in arrays}
    for idx in bootstrap_samples(groups.reindex(keys["passage_id"]).to_numpy(), n_boot, seed=0):
        for name, q in arrays.items():
            boot[name].append(statistics(q, idx))
    boot = {name: np.asarray(b) for name, b in boot.items()}

    results = {}
    for name, q in arrays.items():
        per_seed = np.array([statistics({k: v[:, [s]] for k, v in q.items()})
                             for s in range(q["first_hit"].shape[1])])
        lo, hi = np.nanquantile(boot[name], [0.025, 0.975], axis=0)
        results[name] = {
            "n_cases": len(keys), "n_passages": int(keys["passage_id"].nunique()),
            "seeds": sorted(experiments[name]),
            **{s: {"mean": float(observed[name][i]), "sd": float(per_seed[:, i].std()),
                   "ci95": [float(lo[i]), float(hi[i])]} for i, s in enumerate(STATISTICS)},
            "by_category": _by_category(keys, q),
        }
    comparisons = []
    for a, b in pairs:
        diffs = boot[a] - boot[b]
        lo, hi = np.nanquantile(diffs, [0.025, 0.975], axis=0)
        comparisons.append({"a": a, "b": b, **{
            s: {"difference": float(observed[a][i] - observed[b][i]),
                "ci95": [float(lo[i]), float(hi[i])], "a_ahead": float(np.mean(diffs[:, i] > 0))}
            for i, s in enumerate(STATISTICS)}})
    return results, comparisons


def _by_category(keys: pd.DataFrame, q: dict[str, np.ndarray]) -> list[dict]:
    frame = keys.assign(first_hit=q["first_hit"].mean(1), chance=q["chance"].mean(1),
                        sentence_ap=q["sentence_ap"].mean(1))
    return (frame.groupby("category")
            .agg(cases=("first_hit", "size"), first_hit=("first_hit", "mean"),
                 chance=("chance", "mean"), sentence_ap=("sentence_ap", "mean"))
            .reset_index().to_dict("records"))


def _fmt(stat: dict) -> str:
    return f"{stat['mean']:.3f} [{stat['ci95'][0]:.3f}, {stat['ci95'][1]:.3f}]"


SCOPES = {
    "cv": ("cross-validation", "a test passage", "Thresholds are tuned on each run's validation "
           "passages. Lockbox not used."),
    "lockbox": ("lockbox", "a lockbox passage", "Thresholds are tuned on each run's validation "
                "fold. The lockbox's evidence wasn't realigned or re-marked as the development "
                "passages' was, so compare experiments with each other here, not with the "
                "cross-validation figures."),
}


def render(name: str, result: dict, thresholds: pd.DataFrame, scope: str = "cv") -> str:
    edges = thresholds.drop(columns=["seed", "run"]).isin([GRID[0], GRID[-1]]).to_numpy().mean()
    where, passage, how = SCOPES[scope]
    out = [f"# Evidence sentences, {where}: {name}", "",
           f"Each gold position of {passage} with evidence that fits the stored sentences "
           f"has sentences marked from its sentence scores: all at or above its marking "
           f"threshold (at most {MAX_MARKED}), or else the strongest; sentences under "
           f"{MIN_WORDS} words aren't marked. {how}", "",
           f"{result['n_cases']} positions in {result['n_passages']} passages, seeds "
           f"{result['seeds']}. Means over seeds [95% interval from resampling articles].", "",
           "| measure | value |", "|---|---|",
           f"| first marked sentence is an evidence sentence | {_fmt(result['first_hit'])} |",
           f"| the same for a sentence picked at random | {_fmt(result['chance'])} |",
           f"| difference | {_fmt(result['lift'])} |",
           f"| precision of the marked sentences | {_fmt(result['precision'])} |",
           f"| recall of the evidence sentences | {_fmt(result['recall'])} |",
           f"| F1 of the marked sentences | {_fmt(result['f1'])} |",
           f"| sentence AP (threshold-free) | {_fmt(result['sentence_ap'])} |",
           f"| sentences marked per position | {result['marked_per_position']['mean']:.2f} |", "",
           f"Marking thresholds at the edge of the grid ({GRID[0]} or {GRID[-1]}): "
           f"{edges:.0%}.", "",
           "| position | cases | first marked is evidence | at random | sentence AP |",
           "|---|---|---|---|---|"]
    out += [f"| {r['category']} | {r['cases']} | {r['first_hit']:.2f} | {r['chance']:.2f} | "
            f"{r['sentence_ap']:.2f} |" for r in result["by_category"]]
    return "\n".join(out) + "\n"


def render_comparisons(comparisons: list[dict], scope: str = "cv") -> str:
    questions = {(a, b): q for a, b, q in PAIRS}
    shown = ["first_hit", "precision", "recall", "f1", "sentence_ap"]
    where = ("Development cross-validation (lockbox not used)." if scope == "cv" else
             "The lockbox, after training on the development folds.")
    out = ["# Evidence sentences: paired comparisons", "",
           f"{where} Each row is A minus B on the same positions: the observed difference [95% "
           "interval from resampling articles], and the share of resamples in which A is "
           "ahead.", "",
           "| question | A − B | " + " | ".join(shown) + " |", "|---|---|" + "---|" * len(shown)]
    for c in comparisons:
        cells = [f"{c[s]['difference']:+.3f} [{c[s]['ci95'][0]:+.3f}, {c[s]['ci95'][1]:+.3f}], "
                 f"{c[s]['a_ahead']:.0%}" for s in shown]
        out.append(f"| {questions.get((c['a'], c['b']), '')} | {c['a']} − {c['b']} | "
                   + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def write_results(results: dict, comparisons: list[dict], thresholds: dict[str, pd.DataFrame],
                  out: Path, scope: str = "cv") -> None:
    prefix = {"cv": "evidence_cv_", "lockbox": "lockbox_evidence_"}[scope]
    out.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        with open(out / f"{prefix}{name}.md", "w", encoding="utf-8", newline="\n") as f:
            f.write(render(name, result, thresholds[name], scope))
        with open(out / f"{prefix}{name}.json", "w", encoding="utf-8") as f:
            json.dump({"name": name, **result, "thresholds": thresholds[name].to_dict("records")},
                      f, indent=2)
        print(f"{name}: first marked is evidence {_fmt(result['first_hit'])}, "
              f"at random {_fmt(result['chance'])}")
    if comparisons:
        text = render_comparisons(comparisons, scope)
        with open(out / f"{prefix}comparisons.md", "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(out / f"{prefix}comparisons.json", "w", encoding="utf-8") as f:
            json.dump(comparisons, f, indent=2)
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score experiments' evidence sentences.")
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=V3_RESULTS)
    args = parser.parse_args()

    labels = model_labels(load_taxonomy("v1"))
    positions = [label for label in labels if label != IPC]
    passages, gold = load_dev()
    fingerprints = (data_fingerprint(passages, gold), evidence_fingerprint(gold))
    experiments = {}
    for path in args.configs:
        cfg = load_config(path)
        dev = labelled_dev(passages, gold, labels, cfg.split)
        n_rounds = len(cv_rounds(sorted(dev["fold"].unique())))
        cases = evidence_cases(dev, gold, positions)
        seeds, incomplete, thresholds = score_runs(cfg, cases, positions, args.runs_dir,
                                                   n_rounds, fingerprints)
        if not seeds:
            raise SystemExit(f"{cfg.name}: no complete seeds ({incomplete or 'no runs'})")
        if incomplete:
            print(f"{cfg.name}: leaving out unfinished seeds {incomplete}")
        experiments[cfg.name] = (seeds, pd.DataFrame(thresholds))
    pairs = [(a, b) for a, b, _ in PAIRS if a in experiments and b in experiments]
    results, comparisons = evaluate({n: s for n, (s, _) in experiments.items()},
                                    passages.set_index("passage_id")["group_id"], pairs)
    write_results(results, comparisons, {n: t for n, (_, t) in experiments.items()}, args.out)


if __name__ == "__main__":
    main()
