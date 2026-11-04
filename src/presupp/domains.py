"""How reliably a model finds each domain, from its dev cross-validation runs.

    python -m presupp.domains configs/fixed_unweighted.yaml

A domain is predicted (present) when any of its categories is predicted (gold). Writes
reports/results/domains_<experiment>.md and .json; no retraining, lockbox not used.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from presupp.data.dataset import load_dev
from presupp.evaluate import N_BOOT, RESULTS_DIR, load_seeds
from presupp.experiment import load_config
from presupp.labels import model_labels, target_matrix
from presupp.metrics import bootstrap_samples, f1_by_label
from presupp.runs import RUNS_DIR
from presupp.taxonomy import EXCLUSION_DOMAIN, Taxonomy, load_taxonomy


def domain_columns(labels: list[str], taxonomy: Taxonomy) -> dict[str, list[int]]:
    """{domain: label indices}, in taxonomy order."""
    columns = {}
    for domain in taxonomy.domains:
        idx = [j for j, label in enumerate(labels) if taxonomy.domain_of(label) == domain]
        if idx:
            columns[domain] = idx
    return columns


def to_domains(matrix: np.ndarray, columns: dict[str, list[int]]) -> np.ndarray:
    """(N, labels) -> (N, domains) booleans."""
    matrix = np.asarray(matrix) > 0
    return np.stack([matrix[:, idx].any(1) for idx in columns.values()], axis=1)


def _precision_recall(y: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp = (y & d).sum(0)
    predicted, present = d.sum(0), y.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(y.shape[1]), where=predicted > 0)
    recall = np.divide(tp, present, out=np.zeros(y.shape[1]), where=present > 0)
    return precision, recall


def score_domains(seeds: dict, y: np.ndarray, columns: dict[str, list[int]],
                  groups: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Per-domain P/R/F1, macro-F1 over the philosophical domains, exact-match share."""
    names = list(columns)
    philosophical = np.array([n != EXCLUSION_DOMAIN for n in names])
    y_dom = to_domains(y, columns)
    decided = [to_domains(s[2], columns) for s in seeds.values()]

    def statistics(idx):
        per_seed = [f1_by_label(y_dom[idx], d[idx]) for d in decided]
        present = per_seed[0][1]
        f1 = np.where(present, np.mean([f for f, _ in per_seed], axis=0), np.nan)
        keep = philosophical & present
        exact = np.mean([(y_dom[idx] == d[idx]).all(1).mean() for d in decided])
        return f1, (f1[keep].mean() if keep.any() else np.nan), exact

    f1, macro, exact = statistics(slice(None))
    pr = [_precision_recall(y_dom, d) for d in decided]
    boot = [statistics(idx) for idx in bootstrap_samples(groups, n_boot, seed=0)]
    boot_f1 = np.array([b[0] for b in boot])

    rows = []
    for k, name in enumerate(names):
        rows.append({
            "domain": name, "passages": int(y_dom[:, k].sum()),
            "precision": float(np.mean([p[k] for p, _ in pr])),
            "recall": float(np.mean([r[k] for _, r in pr])),
            "f1": float(f1[k]), "f1_ci95": _interval(boot_f1[:, k]),
        })
    return {
        "n_passages": int(len(y)), "seeds": sorted(seeds), "domains": rows,
        "macro_f1": {"mean": float(macro), "ci95": _interval([b[1] for b in boot]),
                     "domains": [n for n, keep in zip(names, philosophical, strict=True) if keep]},
        "exact_match": {"mean": float(exact), "ci95": _interval([b[2] for b in boot])},
    }


def _interval(values) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():  # a domain with no positives in any resample
        return [float("nan"), float("nan")]
    return [float(v) for v in np.nanquantile(values, [0.025, 0.975])]


def render(name: str, result: dict) -> str:
    def fmt(stat):
        return f"{stat['mean']:.3f} [{stat['ci95'][0]:.3f}, {stat['ci95'][1]:.3f}]"

    out = [f"# Domains: {name}", "",
           "How reliably the model finds each domain. A domain counts as predicted for a passage "
           "when any of its categories is predicted, and as present when any of its categories "
           "is in the gold labels.", "",
           f"Development cross-validation (lockbox not used): {result['n_passages']} passages, "
           f"seeds {result['seeds']}. Means over seeds [95% interval from resampling articles].",
           "", "| domain | passages with it | precision | recall | F1 [95% interval] |",
           "|---|---|---|---|---|"]
    for r in result["domains"]:
        out.append(f"| {r['domain']} | {r['passages']} | {r['precision']:.3f} | "
                   f"{r['recall']:.3f} | {r['f1']:.3f} [{r['f1_ci95'][0]:.3f}, "
                   f"{r['f1_ci95'][1]:.3f}] |")
    n = len(result["macro_f1"]["domains"])
    out += ["", "| summary | value |", "|---|---|",
            f"| macro-F1 over the {n} philosophical domains | {fmt(result['macro_f1'])} |",
            f"| passages whose whole set of domains is right | {fmt(result['exact_match'])} |",
            ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-level reliability from dev CV runs.")
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    taxonomy = load_taxonomy("v1")
    labels = model_labels(taxonomy)
    columns = domain_columns(labels, taxonomy)
    passages, gold = load_dev()
    groups = passages.set_index("passage_id")["group_id"]
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.configs:
        cfg = load_config(path)
        seeds, incomplete = load_seeds(cfg, passages, gold, labels, args.runs_dir)
        if incomplete or not seeds:
            raise SystemExit(f"{cfg.name}: not every seed has finished ({incomplete or 'no runs'})")
        ids = next(iter(seeds.values()))[0]
        y = target_matrix(list(ids), gold, labels, "hard") > 0
        result = score_domains(seeds, y, columns, groups.reindex(ids).to_numpy())
        result["name"] = cfg.name
        text = render(cfg.name, result)
        with open(args.out / f"domains_{cfg.name}.md", "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(args.out / f"domains_{cfg.name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(text)


if __name__ == "__main__":
    main()
