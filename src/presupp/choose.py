"""Pick the v3 variant on dev cross-validation, and whether it replaces v1 in the app.

    python -m presupp.choose

Reads reports/results/v3/, writes selection_v3.md there. F is dev macro-F1, H the share of
positions whose first marked sentence is evidence. Higher F wins if the gap is over MARGIN, else
higher H, else v3_mil. It replaces v1 if its F is at most MARGIN below the baseline's and the 95%
interval of its H minus the baseline's is above 0.
"""

import argparse
import json
from pathlib import Path

from presupp.evaluate_evidence import V3_RESULTS

MARGIN = 0.010  # about one seed's standard deviation of dev macro-F1
VARIANTS = ("v3_joint", "v3_mil")
BASELINE = "fixed_unweighted_pointing"
TIE = "v3_mil"


def choose(f: dict[str, float], h: dict[str, float]) -> tuple[str, str]:
    """(the chosen variant, why)."""
    a, b = VARIANTS
    if abs(f[a] - f[b]) > MARGIN:
        return max(VARIANTS, key=f.get), f"its macro-F1 is higher by {abs(f[a] - f[b]):.3f}"
    if abs(h[a] - h[b]) > MARGIN:
        return (max(VARIANTS, key=h.get), f"macro-F1 is within {MARGIN} and its first marked "
                f"sentence is evidence {abs(h[a] - h[b]):.3f} more often")
    return TIE, f"both measures are within {MARGIN}, and its marks explain its decisions"


def replaces_v1(f_chosen: float, f_baseline: float, h_interval: list[float]) -> tuple[bool, str]:
    passages_ok = f_chosen - f_baseline >= -MARGIN
    evidence_ok = h_interval[0] > 0
    return passages_ok and evidence_ok, (
        f"macro-F1 {f_chosen - f_baseline:+.4f} against the baseline "
        f"({'no more than' if passages_ok else 'more than'} {MARGIN} below it); first marked "
        f"sentence is evidence [{h_interval[0]:+.3f}, {h_interval[1]:+.3f}] more often "
        f"({'above' if evidence_ok else 'not above'} 0)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose the v3 variant.")
    parser.add_argument("--results", type=Path, default=V3_RESULTS)
    args = parser.parse_args()

    def read(name):
        with open(args.results / name, encoding="utf-8") as f:
            return json.load(f)

    names = (*VARIANTS, BASELINE)
    f = {n: read(f"{n}.json")["headline"]["macro_f1"]["mean"] for n in names}
    h = {n: read(f"evidence_cv_{n}.json")["first_hit"]["mean"] for n in names}
    against = {(c["a"], c["b"]): c for c in read("evidence_cv_comparisons.json")}
    chosen, why = choose(f, h)
    ok, detail = replaces_v1(f[chosen], f[BASELINE], against[(chosen, BASELINE)]["first_hit"]
                             ["ci95"])
    lines = ["# Choosing v3", "", "Development cross-validation (lockbox not used).", "",
             "| experiment | macro-F1 (all labels) | first marked sentence is evidence |",
             "|---|---|---|"]
    lines += [f"| {n} | {f[n]:.3f} | {h[n]:.3f} |" for n in names]
    lines += ["", f"**Chosen: {chosen}**, because {why}.", "",
              f"**{'Replaces' if ok else 'Does not replace'} v1 in the app:** {detail}.", ""]
    text = "\n".join(lines)
    with open(args.results / "selection_v3.md", "w", encoding="utf-8", newline="\n") as out:
        out.write(text)
    print(text)


if __name__ == "__main__":
    main()
