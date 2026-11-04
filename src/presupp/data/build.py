"""Build the canonical v1 dataset and its leakage-free folds from the raw v1 exports.

    python -m presupp.data.build            # defaults: 10 folds, seed 42

Applies the decisions in data/adjudication/ on top of the raw exports.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from presupp import paths
from presupp.data.adjudication import (
    ADJUDICATED,
    REMARKED,
    REMARKED_ANNOTATORS,
    REMARKED_BY_CLAUDE,
    load_duplicate_decisions,
    load_evidence_realignment,
    load_label_review,
    load_remarks,
)
from presupp.data.annotations import label_sets, load_annotation_file
from presupp.data.corpus import load_corpus, text_differences
from presupp.data.dedup import build_groups, cluster_ids, near_duplicate_pairs, text_hash
from presupp.data.splits import (
    assign_folds,
    check_no_leakage,
    choose_lockbox_fold,
    fold_label_counts,
    random_group_folds,
    stratification_error,
)
from presupp.labels import MEASURABLE_MIN_ARTICLES, model_labels
from presupp.taxonomy import Taxonomy, load_taxonomy

PRIMARY = "primary"
REMARK_READING = {REMARKED: "re-marked", REMARKED_BY_CLAUDE: "re-marked by Claude"}
LEAKAGE_KEYS = ["group_id", "url_norm", "article_id", "text_hash", "text_cluster"]
BORDERLINE_SIMILARITY = 0.8


class BuildError(RuntimeError):
    pass


@dataclass
class BuildConfig:
    raw_dir: Path = paths.RAW_V1_DIR
    reference_dir: Path = paths.REFERENCE_DIR
    adjudication_dir: Path = paths.ADJUDICATION_DIR
    processed_dir: Path = paths.PROCESSED_DIR
    splits_dir: Path = paths.SPLITS_DIR
    reports_dir: Path = paths.REPORTS_DIR
    taxonomy_dir: Path = paths.TAXONOMY_DIR
    taxonomy_version: str = "v1"
    n_folds: int = 10
    seed: int = 42
    near_duplicate_threshold: float = 0.9
    # reuse an existing folds_<version>.csv: label fixes mustn't reshuffle the split
    freeze_folds: bool = True


@dataclass
class BuildResult:
    corpus: pd.DataFrame
    records: pd.DataFrame
    labels: pd.DataFrame
    gold: pd.DataFrame
    folds: pd.DataFrame
    facts: dict = field(default_factory=dict)


def _id_aliases(classifier_passages: Path, corpus: pd.DataFrame) -> dict[str, str]:
    """The classifier export renamed the supplementary passages; map them back by text."""
    exported = pd.read_csv(classifier_passages, dtype=str, keep_default_na=False)
    known = set(corpus["passage_id"])
    by_text = corpus.groupby(corpus["text"].str.strip())["passage_id"].agg(list)
    aliases = {}
    for row in exported.itertuples():
        if row.id in known:
            continue
        matches = by_text.get(row.text.strip(), [])
        if len(matches) != 1:
            raise BuildError(f"Cannot map exported id {row.id}: {len(matches)} text matches")
        aliases[row.id] = matches[0]
    return aliases


def _text_clusters(corpus: pd.DataFrame, threshold: float) -> tuple[pd.Series, list, list]:
    ids = corpus["passage_id"].tolist()
    hashes = corpus["text_hash"].tolist()
    exact_links = []
    first_by_hash: dict[str, str] = {}
    for pid, h in zip(ids, hashes, strict=True):
        if h in first_by_hash:
            exact_links.append((first_by_hash[h], pid))
        else:
            first_by_hash[h] = pid

    hash_of = dict(zip(ids, hashes, strict=True))
    similar = near_duplicate_pairs(ids, corpus["text"].tolist(), BORDERLINE_SIMILARITY)
    near = [(a, b, s) for a, b, s in similar if s >= threshold and hash_of[a] != hash_of[b]]
    borderline = [(a, b, s) for a, b, s in similar if s < threshold]
    clusters = cluster_ids(ids, exact_links + [(a, b) for a, b, _ in near])
    return pd.Series(clusters).reindex(ids).to_numpy(), near, borderline


def _evidence_checks(labels: pd.DataFrame, corpus: pd.DataFrame,
                     realigned: dict[tuple[str, str], tuple[list[int], str]],
                     exclusions: set[str]) -> pd.DataFrame:
    """Range checks, plus evidence_0based (the evidence to use) and evidence_reading.

    Primary annotator's old tool stored 1-based S-numbers, except lists containing a 0."""
    n_sentences = corpus.set_index("passage_id")["n_sentences"]
    n = labels["passage_id"].map(n_sentences)
    ev_min = labels["evidence"].map(lambda ev: min(ev) if ev else np.nan)
    ev_max = labels["evidence"].map(lambda ev: max(ev) if ev else np.nan)
    has_ev = labels["evidence"].map(len) > 0
    labels = labels.copy()
    # NA where there's no evidence (exclusion labels)
    in_0based = (ev_min >= 0) & (ev_max < n)
    in_1based = (ev_min >= 1) & (ev_max <= n)
    labels["evidence_in_range_0based"] = in_0based.astype("boolean").where(has_ev)
    labels["evidence_in_range_1based"] = in_1based.astype("boolean").where(has_ev)
    labels["evidence_convention"] = np.where(
        labels["annotator"] == PRIMARY, "unverified", "0-based"
    )
    used, reading = [], []
    for row, ok0, ok1 in zip(labels.itertuples(), in_0based, in_1based, strict=True):
        ev = sorted(int(e) for e in row.evidence)
        key = (row.passage_id, row.category)
        remarked = REMARK_READING.get(row.annotator)
        if not ev:
            used.append(None)
            no_sentence = remarked and row.category not in exclusions
            reading.append(f"{remarked}: no sentence" if no_sentence else "none")
        elif remarked:
            used.append(ev)  # checked against the passage when loaded
            reading.append(remarked)
        elif row.annotator != PRIMARY:
            used.append(ev if ok0 else None)
            reading.append("0-based" if ok0 else "doesn't fit")
        elif key in realigned:
            used.append(realigned[key][0])
            reading.append(f"realigned: {realigned[key][1]}")
        elif ev[0] == 0:
            used.append(ev if ok0 else None)
            reading.append("0-based (contains 0)" if ok0 else "doesn't fit")
        else:
            used.append([e - 1 for e in ev] if ok1 else None)
            reading.append("1-based" if ok1 else "doesn't fit")
    labels["evidence_0based"] = used
    labels["evidence_reading"] = reading
    return labels


def _duplicate_label_review(corpus: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    primary_sets = label_sets(labels, PRIMARY)
    rows = []
    for cluster, members in corpus.groupby("text_cluster")["passage_id"]:
        annotated = [pid for pid in sorted(members) if pid in primary_sets]
        if len(annotated) < 2:
            continue
        sets = [primary_sets[pid] for pid in annotated]
        rows.append({
            "text_cluster": cluster,
            "passage_ids": ";".join(annotated),
            "labels": " | ".join(",".join(sorted(s)) for s in sets),
            "is_conflict": len(set(sets)) > 1,
        })
    return pd.DataFrame(rows, columns=["text_cluster", "passage_ids", "labels", "is_conflict"])


def _canonical_copies(
    corpus: pd.DataFrame, annotated: set[str], keep: dict[str, str]
) -> np.ndarray:
    """Mask of one copy per text cluster: adjudicated, else first annotated, else first."""
    chosen = {}
    for cluster, members in corpus.groupby("text_cluster")["passage_id"]:
        members = sorted(members)
        chosen[cluster] = keep.get(cluster) or next(
            (m for m in members if m in annotated), members[0]
        )
    return (corpus["passage_id"] == corpus["text_cluster"].map(chosen)).to_numpy()


def gold_labels(
    labels: pd.DataFrame, records: pd.DataFrame, corpus: pd.DataFrame
) -> pd.DataFrame:
    """Re-marked labels, else adjudicated, else primary; canonical copies only."""
    canonical = set(corpus.loc[corpus["is_canonical"], "passage_id"])
    remarked = set(records.loc[records["annotator"].isin(REMARKED_ANNOTATORS), "passage_id"])
    adjudicated = set(records.loc[records["annotator"] == ADJUDICATED, "passage_id"])
    primary = records.query("annotator == @PRIMARY and is_complete")["passage_id"]
    use_primary = labels["annotator"].eq(PRIMARY) & labels["passage_id"].isin(
        set(primary) - adjudicated - remarked
    )
    use = labels["annotator"].isin(REMARKED_ANNOTATORS) | use_primary | (
        labels["annotator"].eq(ADJUDICATED) & ~labels["passage_id"].isin(remarked))
    gold = labels[use & labels["passage_id"].isin(canonical)]
    gold = gold.rename(columns={"annotator": "label_source"})
    return gold.sort_values(["passage_id", "category"], ignore_index=True)


def label_support(gold: pd.DataFrame, corpus: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Positives and articles (split groups) per model label."""
    groups = corpus.set_index("passage_id")["group_id"]
    stats = (gold.assign(group_id=gold["passage_id"].map(groups))
             .groupby("category")
             .agg(positives=("passage_id", "nunique"), articles=("group_id", "nunique"))
             .reindex(labels, fill_value=0))
    stats["measurable"] = stats["articles"] >= MEASURABLE_MIN_ARTICLES
    return stats.rename_axis("label").reset_index()


def label_matrix(
    corpus: pd.DataFrame, gold: pd.DataFrame, taxonomy: Taxonomy
) -> tuple[np.ndarray, np.ndarray]:
    """(N, n_categories) 0/1 matrix in taxonomy order, and the labelled-passage mask."""
    index = {pid: i for i, pid in enumerate(corpus["passage_id"])}
    column = {cid: j for j, cid in enumerate(taxonomy.ids)}
    matrix = np.zeros((len(corpus), len(column)), dtype=np.int8)
    matrix[gold["passage_id"].map(index).to_numpy(), gold["category"].map(column).to_numpy()] = 1
    labelled = corpus["passage_id"].isin(set(gold["passage_id"])).to_numpy()
    return matrix, labelled


def build(config: BuildConfig) -> BuildResult:
    raw = config.raw_dir
    taxonomy = load_taxonomy(config.taxonomy_version, config.taxonomy_dir)
    facts: dict = {"config": config}

    # 1. corpus
    corpus = load_corpus(
        raw / "passages_main.jsonl", raw / "passages_all.csv",
        config.reference_dir / "publishers.csv",
    )
    facts["text_differences"] = text_differences(
        raw / "passages_main.jsonl", raw / "passages_all.csv"
    )

    # 2. duplicates and split groups
    corpus["text_hash"] = corpus["text"].map(text_hash)
    corpus["text_cluster"], facts["near_duplicates"], facts["borderline_pairs"] = _text_clusters(
        corpus, config.near_duplicate_threshold
    )
    corpus["text_cluster_size"] = corpus.groupby("text_cluster")["passage_id"].transform("size")
    corpus["group_id"] = build_groups(corpus["url_norm"].tolist(), corpus["text_cluster"].tolist())

    # 3. annotations
    aliases = _id_aliases(raw / "classifier_passages.csv", corpus)
    records, labels = load_annotation_file(raw / "primary_labels.csv", PRIMARY, "primary", taxonomy)
    _, exported = load_annotation_file(
        raw / "classifier_primary_labels.csv", PRIMARY, "primary", taxonomy,
        id_column="id", id_aliases=aliases,
    )
    canonical_sets, exported_sets = label_sets(labels, PRIMARY), label_sets(exported, PRIMARY)
    mismatched = sorted(
        pid for pid in exported_sets if canonical_sets.get(pid) != exported_sets[pid]
    )
    if mismatched:
        raise BuildError(f"Primary label exports disagree on {len(mismatched)}: {mismatched[:10]}")
    facts["exported_label_check"] = len(exported_sets)

    all_records, all_labels = [records], [labels]
    for path in sorted(raw.glob("expert_*.csv")):
        r, lab = load_annotation_file(path, path.stem, "expert", taxonomy)
        all_records.append(r)
        all_labels.append(lab)
    records = pd.concat(all_records, ignore_index=True)
    labels = pd.concat(all_labels, ignore_index=True)

    unknown = sorted(set(records["passage_id"]) - set(corpus["passage_id"]))
    if unknown:
        raise BuildError(f"Annotations for passages not in corpus: {unknown[:10]}")
    review = _duplicate_label_review(corpus, labels)

    # 4. adjudication decisions and gold labels
    decisions = config.adjudication_dir / f"duplicates_{config.taxonomy_version}.csv"
    adj_records, adj_labels, keep = load_duplicate_decisions(decisions, corpus, taxonomy)
    unresolved = sorted(set(review.loc[review["is_conflict"], "text_cluster"]) - set(keep))
    if unresolved:
        raise BuildError(
            f"Duplicated passages were labelled differently and need a decision in "
            f"{decisions.name}: {unresolved}"
        )
    if keep:
        records = pd.concat([records, adj_records], ignore_index=True)
        labels = pd.concat([labels, adj_labels], ignore_index=True)
    review_file = config.adjudication_dir / f"label_review_{config.taxonomy_version}.csv"
    rev_records, rev_labels, reviewed = load_label_review(review_file, corpus, taxonomy)
    both = sorted(set(reviewed) & set(keep.values()))
    if both:
        raise BuildError(f"Passages decided in both adjudication files: {both}")
    if not rev_records.empty:
        records = pd.concat([records, rev_records], ignore_index=True)
        labels = pd.concat([labels, rev_labels], ignore_index=True)
    corpus["label_review"] = corpus["passage_id"].map(reviewed)
    facts["label_review"] = reviewed
    realigned = load_evidence_realignment(
        config.adjudication_dir / f"evidence_realigned_{config.taxonomy_version}.csv",
        labels, corpus)
    remark_file = config.adjudication_dir / f"evidence_remarked_{config.taxonomy_version}.csv"
    rem_records, rem_labels, facts["remarked"] = load_remarks(remark_file, labels, corpus,
                                                              taxonomy)
    both = sorted(set(rem_records["passage_id"]) & (set(reviewed) | set(keep.values())))
    if both:
        raise BuildError(f"Passages re-marked and also decided in an adjudication file: {both}")
    if not rem_records.empty:
        records = pd.concat([records, rem_records], ignore_index=True)
        labels = pd.concat([labels, rem_labels], ignore_index=True)
    labels = _evidence_checks(labels, corpus, realigned, set(taxonomy.exclusion_ids))

    complete = records.loc[records["is_complete"], ["passage_id", "annotator"]]
    primary_done = set(complete.loc[complete["annotator"] == PRIMARY, "passage_id"])
    corpus["has_primary_annotation"] = corpus["passage_id"].isin(primary_done)
    corpus["is_canonical"] = _canonical_copies(corpus, primary_done | set(keep.values()), keep)
    gold = gold_labels(labels, records, corpus)
    not_gold = sorted(set(reviewed) - set(gold["passage_id"]))
    if not_gold:
        raise BuildError(f"Label-review decisions for passages outside the gold set: {not_gold}")
    facts["adjudicated"] = keep
    facts["dropped_duplicates"] = int(
        (corpus["has_primary_annotation"] & ~corpus["is_canonical"]).sum()
    )

    facts["label_support"] = label_support(gold, corpus, model_labels(taxonomy))

    # 5. folds
    matrix, labelled = label_matrix(corpus, gold, taxonomy)
    frozen = config.splits_dir / f"folds_{config.taxonomy_version}.csv"
    facts["folds_frozen"] = config.freeze_folds and frozen.exists()
    if facts["folds_frozen"]:
        previous = pd.read_csv(frozen, dtype={"passage_id": str, "group_id": str})
        previous = previous.set_index("passage_id").reindex(corpus["passage_id"])
        if previous["fold"].isna().any():
            raise BuildError(f"{frozen.name} lacks passages now in the corpus; rebuild with "
                             "freeze_folds=False to reassign")
        if (previous["group_id"].to_numpy() != corpus["group_id"].to_numpy()).any():
            raise BuildError(f"Split groups changed since {frozen.name} was written")
        corpus["fold"] = previous["fold"].astype(int).to_numpy()
        frozen_split = previous["split"].to_numpy()
    else:
        assignment = assign_folds(
            corpus["group_id"].to_numpy(), matrix, labelled, config.n_folds, config.seed
        )
        corpus["fold"] = assignment.fold
    corpus["is_labelled"] = labelled
    leaks = check_no_leakage(corpus, "fold", LEAKAGE_KEYS)
    if leaks:
        raise BuildError(f"Leakage between folds: { {k: v[:5] for k, v in leaks.items()} }")

    baseline = [
        stratification_error(
            random_group_folds(corpus["group_id"].to_numpy(), config.n_folds, s),
            matrix, labelled, config.n_folds,
        )
        for s in range(20)
    ]
    facts["stratification_error"] = stratification_error(
        corpus["fold"].to_numpy(), matrix, labelled, config.n_folds
    )
    facts["stratification_error_random"] = (float(np.mean(baseline)), float(np.std(baseline)))
    facts["fold_counts"] = fold_label_counts(
        corpus["fold"].to_numpy(), matrix, labelled, taxonomy.ids, config.n_folds
    )

    # 6. lockbox
    if facts["folds_frozen"]:
        corpus["split"] = frozen_split
        lockbox = int(corpus.loc[corpus["split"] == "lockbox", "fold"].iloc[0])
    else:
        lockbox = choose_lockbox_fold(corpus["fold"].to_numpy(), matrix, labelled,
                                      config.n_folds)
        corpus["split"] = np.where(corpus["fold"] == lockbox, "lockbox", "dev")
    facts["lockbox_fold"] = lockbox

    folds = corpus[["passage_id", "group_id", "fold", "split", "is_labelled"]].copy()
    result = BuildResult(
        corpus=corpus, records=records, labels=labels, gold=gold, folds=folds, facts=facts
    )
    facts["review"] = review
    facts["aliases"] = aliases
    facts["taxonomy"] = taxonomy
    _write_outputs(result, config)
    return result


def _write_outputs(result: BuildResult, config: BuildConfig) -> None:
    for d in (config.processed_dir, config.splits_dir, config.reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    result.corpus.to_parquet(config.processed_dir / "passages.parquet", index=False)
    result.records.to_parquet(config.processed_dir / "annotation_records.parquet", index=False)
    result.labels.to_parquet(config.processed_dir / "annotation_labels.parquet", index=False)
    result.gold.to_parquet(config.processed_dir / "gold_labels.parquet", index=False)
    result.facts["review"].to_csv(config.processed_dir / "duplicate_label_review.csv", index=False)
    pd.Series(result.facts["aliases"], name="canonical_id").rename_axis("exported_id").to_csv(
        config.processed_dir / "id_aliases.csv"
    )
    result.facts["label_support"].to_csv(
        config.splits_dir / f"label_support_{config.taxonomy_version}.csv", index=False,
        lineterminator="\n",
    )
    result.folds.to_csv(
        config.splits_dir / f"folds_{config.taxonomy_version}.csv", index=False, lineterminator="\n"
    )
    report = render_report(result, config)
    with open(config.reports_dir / f"data_build_{config.taxonomy_version}.md", "w",
              encoding="utf-8", newline="\n") as f:
        f.write(report)


def _manifest(raw_dir: Path) -> dict:
    path = raw_dir / "MANIFEST.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _md_table(frame: pd.DataFrame) -> str:
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join([frame.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in frame.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def render_report(result: BuildResult, config: BuildConfig) -> str:
    c, r, lab, f = result.corpus, result.records, result.labels, result.facts
    taxonomy: Taxonomy = f["taxonomy"]
    out = [f"# Data build report ({config.taxonomy_version})", ""]
    out += ["Generated by `python -m presupp.data.build`. "
            "Contains ids and counts only, no text.", ""]

    manifest = _manifest(config.raw_dir)
    if manifest:
        out += ["## Inputs", "", "| file | from v1 path | sha256 |", "|---|---|---|"]
        out += [f"| {k} | {v['source']} | `{v['sha256'][:12]}` |" for k, v in manifest.items()]
        out.append("")

    out += ["## Corpus", ""]
    out.append(f"- Passages: **{len(c)}** ({(c.origin == 'main').sum()} main, "
               f"{(c.origin == 'supplementary').sum()} supplementary)")
    out.append(f"- Articles (normalised URL): **{c.url_norm.nunique()}**; split groups: "
               f"**{c.group_id.nunique()}**; largest group: "
               f"{c.group_id.value_counts().max()} passages")
    out.append(f"- Publishers: {c.publisher.nunique()}; countries: "
               + ", ".join(f"{k} {v}" for k, v in c.country.value_counts().items()))
    out.append("- Genres: " + ", ".join(f"{k} {v}" for k, v in c.genre.value_counts().items()))
    out.append(f"- Passages whose scraper feed differs from the URL's publisher: "
               f"{(c.source != c.publisher).sum()} (country/genre use the publisher)")
    out.append(f"- Passages containing a redaction marker: {int(c.has_redaction.sum())}")
    if f["text_differences"]:
        out.append(f"- Text differs between the JSONL and CSV exports for: "
                   f"{', '.join(f['text_differences'])} (JSONL text used, as shown to annotators)")
    out.append("")

    multi = c[c.text_cluster_size > 1]
    out += ["## Duplicates", ""]
    out.append(f"- Text clusters with more than one passage: **{multi.text_cluster.nunique()}** "
               f"({len(multi)} passages); exact duplicates are matched on normalised text, near "
               f"duplicates on character 5-gram TF-IDF cosine >= {config.near_duplicate_threshold}")
    url_merged = c.groupby("group_id").url_norm.nunique()
    out.append(f"- Split groups spanning more than one URL (merged via duplicate text): "
               f"{int((url_merged > 1).sum())}")
    if f["near_duplicates"]:
        out += ["", "Near-duplicate (non-identical) pairs merged:", ""]
        out += [f"- {a} ~ {b} (cos {s:.3f})" for a, b, s in f["near_duplicates"]]
    if f["borderline_pairs"]:
        out += ["", f"Borderline pairs NOT merged ({BORDERLINE_SIMILARITY} <= cos < "
                f"{config.near_duplicate_threshold}), for manual review:", ""]
        out += [f"- {a} ~ {b} (cos {s:.3f})" for a, b, s in f["borderline_pairs"]]
    out.append("")

    out += ["## Annotations", ""]
    summary = (
        r.groupby("annotator")
        .agg(records=("passage_id", "size"), complete=("is_complete", "sum"),
             legacy_empty=("format", lambda s: int((s == "legacy_empty").sum())))
        .join(lab.groupby("annotator").size().rename("labels"))
    )
    out += [_md_table(summary), ""]
    out.append(f"- The classifier's label export matches the canonical primary labels on all "
               f"{f['exported_label_check']} passages, after mapping {len(f['aliases'])} renamed "
               f"supplementary ids back to their original ids.")
    issues = r.issues.dropna()
    out.append(f"- Records with an exclusion label plus other labels: {len(issues)}")
    out.append("")

    out += ["### Evidence sentence indices", ""]
    ev = lab[lab.evidence.map(len) > 0]
    evidence = ev.groupby("annotator").agg(
        lists=("passage_id", "size"),
        valid_0based=("evidence_in_range_0based", "mean"),
        valid_1based=("evidence_in_range_1based", "mean"),
        contains_index_0=("evidence", lambda s: float(s.map(lambda e: 0 in e).mean())),
    ).round(3)
    out += [_md_table(evidence), ""]
    out.append("Expert evidence is 0-based. The primary annotator's older tool stored 1-based "
               "S-numbers, except lists containing a 0. Lists clicked on a different sentence "
               "split are realigned (`evidence_realigned_v1.csv`) or re-marked "
               "(`evidence_remarked_v1.csv`) where possible. `evidence_reading` says how each "
               "list was read:")
    readings = (ev.groupby(["annotator", "evidence_reading"]).size().rename("lists")
                .reset_index().set_index("annotator"))
    out += ["", _md_table(readings), ""]
    remarked = f.get("remarked") or {}
    if remarked.get("passages"):
        out += [f"Re-marked passages: {remarked['passages']} ({remarked['evidence_only']} with "
                f"only their evidence re-marked; positions added {remarked['added']}, removed "
                f"{remarked['removed']}, confidence changed {remarked['confidence_changed']}; "
                f"{remarked['no_philosophy']} found to have no philosophical content). "
                f"{remarked['by_claude']} of them were taken from Claude's annotations and are "
                f"recorded as `{REMARKED_BY_CLAUDE}`.", ""]

    review = f["review"]
    out += ["### Duplicated texts with more than one primary annotation", ""]
    out.append(f"- Clusters: {len(review)}; labelled differently: "
               f"**{int(review.is_conflict.sum())}**, each decided in "
               f"`data/adjudication/duplicates_{config.taxonomy_version}.csv`")
    out.append("")

    gold = result.gold
    reviewed = f["label_review"]
    if reviewed:
        kept = sorted(p for p, d in reviewed.items() if d == "kept")
        replaced = sorted(p for p, d in reviewed.items() if d == "replaced")
        out += ["### Label review", ""]
        out.append(f"- Passages re-examined by the annotator after Claude's blind review: "
                   f"{len(reviewed)}; original labels kept: {len(kept)}; replaced: "
                   f"{len(replaced)} ({', '.join(replaced) or 'none'})")
        out.append("")
    out += ["### Gold labels", ""]
    out.append(f"- Passages with gold labels: **{gold.passage_id.nunique()}** (one copy per "
               f"duplicated text; {f['dropped_duplicates']} annotated duplicate copies left out)")
    adjudicated = ", ".join(sorted(f["adjudicated"].values())) or "none"
    n_replaced = sum(d == "replaced" for d in f["label_review"].values())
    out.append(f"- Labels from decisions: {len(f['adjudicated'])} duplicate decisions "
               f"({adjudicated}) and {n_replaced} label-review replacements; all other "
               "passages use the primary annotation")
    out.append("")

    counts: pd.DataFrame = f["fold_counts"]
    table = counts.T
    table["total"] = table.sum(axis=1)
    table.index.name = "label \\ fold"
    err, (rand_mean, rand_std) = f["stratification_error"], f["stratification_error_random"]
    out += [f"## Folds ({config.n_folds} folds, seed {config.seed})", ""]
    if f["folds_frozen"]:
        out.append(f"- Fold and lockbox assignment frozen from `data/splits/folds_"
                   f"{config.taxonomy_version}.csv`; counts below use the current gold labels.")
    out.append(f"- Leakage check: no `{'`, `'.join(LEAKAGE_KEYS)}` value occurs in more than "
               f"one fold.")
    out.append(f"- Stratification error (mean relative deviation of per-fold positives): "
               f"**{err:.3f}**, against {rand_mean:.3f} ± {rand_std:.3f} for random group folds "
               f"(20 seeds).")
    unused = [cid for cid in taxonomy.ids if counts[cid].sum() == 0]
    if unused:
        out.append(f"- Categories with no primary labels: {', '.join(unused)}")
    out += ["", _md_table(table.drop(index=unused)), ""]

    box = c[c.split == "lockbox"]
    out += ["## Lockbox", ""]
    out.append(f"- Fold **{f['lockbox_fold']}** is held out: {int(box.is_labelled.sum())} "
               f"labelled passages ({len(box)} in total, from {box.group_id.nunique()} split "
               f"groups). Development uses the other {config.n_folds - 1} folds.")
    out.append("- Chosen by a fixed rule that looks at labels only: among folds containing "
               "every category, the one whose label mix is closest to the whole dataset's.")
    out.append("- Code reaches it only through `presupp.data.dataset.open_lockbox`, which "
               "records every opening in `reports/lockbox_log.md`.")
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the v1 dataset and folds.")
    parser.add_argument("--raw-dir", type=Path, default=paths.RAW_V1_DIR)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.9)
    args = parser.parse_args()
    config = BuildConfig(
        raw_dir=args.raw_dir, n_folds=args.n_folds, seed=args.seed,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    result = build(config)
    print(f"Built {len(result.corpus)} passages, {result.corpus.group_id.nunique()} groups, "
          f"{config.n_folds} folds. Report: reports/data_build_{config.taxonomy_version}.md")


if __name__ == "__main__":
    main()
