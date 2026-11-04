"""Load the adjudication files: labels decided later, which override the primary annotation.

evidence_sentences in these CSVs are 1-based S-numbers; the label tables store 0-based.
"""

from pathlib import Path

import pandas as pd

from presupp.data.annotations import CONFIDENCE_LEVELS, LABEL_COLUMNS, RECORD_COLUMNS
from presupp.taxonomy import Taxonomy

ADJUDICATED = "adjudicated"
COLUMNS = ["text_cluster", "keep_passage_id", "category", "confidence", "evidence_sentences",
           "decided_on"]


class AdjudicationError(ValueError):
    pass


def _evidence(value: str, n_sentences: int, where: str) -> list[int]:
    if not value:
        return []
    try:
        s_numbers = sorted({int(v) for v in value.split(";")})
    except ValueError as e:
        raise AdjudicationError(f"{where}: evidence must be S-numbers like 1;3") from e
    if s_numbers[0] < 1 or s_numbers[-1] > n_sentences:
        raise AdjudicationError(f"{where}: evidence {s_numbers} outside S1..S{n_sentences}")
    return [s - 1 for s in s_numbers]


def load_duplicate_decisions(
    path: Path, corpus: pd.DataFrame, taxonomy: Taxonomy
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """(records, labels, keep); keep maps text_cluster -> kept passage id. Empty if no file."""
    empty = (pd.DataFrame(columns=RECORD_COLUMNS), pd.DataFrame(columns=LABEL_COLUMNS), {})
    if not path.exists():
        return empty
    decisions = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(COLUMNS) - set(decisions.columns)
    if missing:
        raise AdjudicationError(f"{path.name} is missing columns {sorted(missing)}")
    if decisions.empty:
        return empty

    passages = corpus.set_index("passage_id")
    keep: dict[str, str] = {}
    records, labels = [], []
    for cluster, rows in decisions.groupby("text_cluster", sort=True):
        kept = rows["keep_passage_id"].unique()
        if len(kept) != 1:
            raise AdjudicationError(f"{cluster}: more than one keep_passage_id {list(kept)}")
        pid = kept[0]
        if pid not in passages.index or passages.loc[pid, "text_cluster"] != cluster:
            raise AdjudicationError(f"{cluster}: {pid} is not a passage in this text cluster")
        if rows["category"].duplicated().any():
            raise AdjudicationError(f"{cluster}: a category is listed twice")

        chosen = set(rows["category"])
        exclusions = chosen & set(taxonomy.exclusion_ids)
        if exclusions and len(chosen) > len(exclusions):
            raise AdjudicationError(f"{cluster}: exclusion category combined with others")
        for row in rows.itertuples():
            where = f"{path.name} {cluster}/{row.category}"
            if row.category not in taxonomy:
                raise AdjudicationError(f"{where}: unknown category")
            is_exclusion = taxonomy.get(row.category).is_exclusion
            confidence = row.confidence or None
            if is_exclusion and (confidence or row.evidence_sentences):
                raise AdjudicationError(f"{where}: exclusions take no confidence or evidence")
            if not is_exclusion and confidence not in CONFIDENCE_LEVELS:
                raise AdjudicationError(f"{where}: confidence must be one of {CONFIDENCE_LEVELS}")
            evidence = _evidence(row.evidence_sentences, passages.loc[pid, "n_sentences"], where)
            labels.append({"passage_id": pid, "annotator": ADJUDICATED, "category": row.category,
                           "confidence": confidence, "evidence": evidence})

        keep[cluster] = pid
        records.append({
            "passage_id": pid, "annotator": ADJUDICATED, "role": "adjudication",
            "source_file": path.name, "format": "adjudication", "is_complete": True,
            "explicit_philosophy_flag": None, "timestamp": rows["decided_on"].max() or None,
            "duration_seconds": None, "notes": None, "issues": None,
        })
    return (pd.DataFrame(records, columns=RECORD_COLUMNS),
            pd.DataFrame(labels, columns=LABEL_COLUMNS), keep)


REALIGNED_COLUMNS = ["passage_id", "category", "recorded", "evidence_sentences", "splitter"]


def load_evidence_realignment(path: Path, labels: pd.DataFrame, corpus: pd.DataFrame
                              ) -> dict[tuple[str, str], tuple[list[int], str]]:
    """evidence_realigned_<version>.csv: primary clicks mapped onto the stored sentences.

    Returns {(passage_id, category): (0-based stored indices, splitter)}; {} if no file."""
    if not path.exists():
        return {}
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REALIGNED_COLUMNS) - set(rows.columns)
    if missing:
        raise AdjudicationError(f"{path.name} is missing columns {sorted(missing)}")
    primary = labels[labels["annotator"] == "primary"].set_index(["passage_id", "category"])
    n_sentences = corpus.set_index("passage_id")["n_sentences"]
    out = {}
    for row in rows.itertuples():
        key, where = (row.passage_id, row.category), f"{path.name} {row.passage_id}/{row.category}"
        if key not in primary.index:
            raise AdjudicationError(f"{where}: no such primary label")
        recorded = [int(v) for v in row.recorded.split(";")] if row.recorded else []
        if recorded != sorted(int(e) for e in primary.loc[key, "evidence"]):
            raise AdjudicationError(f"{where}: recorded {recorded} isn't what the label holds")
        out[key] = (_evidence(row.evidence_sentences, n_sentences[row.passage_id], where),
                    row.splitter)
    return out


REMARKED = "remarked"
REMARKED_BY_CLAUDE = "remarked_claude"
REMARKED_ANNOTATORS = (REMARKED, REMARKED_BY_CLAUDE)
REMARKED_COLUMNS = ["passage_id", "category", "evidence_sentences", "decided_on", "note"]
REMARK_STATUSES = ("original", "added", "removed")


def load_remarks(path: Path, labels: pd.DataFrame, corpus: pd.DataFrame, taxonomy: Taxonomy
                 ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """evidence_remarked_<version>.csv from the re-marking page; rows replace a passage's labels.

    Returns (records, labels, summary). Older files lack the confidence and status columns."""
    empty = pd.DataFrame(columns=RECORD_COLUMNS), pd.DataFrame(columns=LABEL_COLUMNS), {}
    if not path.exists():
        return empty
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REMARKED_COLUMNS) - set(rows.columns)
    if missing:
        raise AdjudicationError(f"{path.name} is missing columns {sorted(missing)}")
    for column, default in [("status", "original"), ("confidence", ""), ("source", "")]:
        if column not in rows:
            rows[column] = default
    if rows[["passage_id", "category"]].duplicated().any():
        raise AdjudicationError(f"{path.name}: a (passage, category) is listed twice")
    dates = rows["decided_on"]
    not_iso = dates[~dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}")]
    if len(not_iso):
        raise AdjudicationError(f"{path.name}: decided_on {not_iso.iloc[0]!r} isn't an ISO date "
                                "like 2026-10-29 (a spreadsheet program may have changed it)")

    primary = labels[labels["annotator"] == "primary"]
    original = {pid: dict(zip(g["category"], g["confidence"], strict=True))
                for pid, g in primary.groupby("passage_id")}
    n_sentences = corpus.set_index("passage_id")["n_sentences"]
    records, out = [], []
    summary = {"passages": 0, "added": 0, "removed": 0, "confidence_changed": 0,
               "evidence_only": 0, "no_philosophy": 0, "by_claude": 0}
    for pid, group in rows.groupby("passage_id", sort=True):
        if pid not in original:
            raise AdjudicationError(f"{path.name} {pid}: not a primary-annotated passage")
        unlisted = set(original[pid]) - set(group["category"])
        if unlisted:
            raise AdjudicationError(f"{path.name} {pid}: labels {sorted(unlisted)} not listed")
        sources = set(group["source"])
        if len(sources) > 1:
            raise AdjudicationError(f"{path.name} {pid}: rows from more than one source")
        source = sources.pop()
        if source and not source.startswith("claude_review/"):
            raise AdjudicationError(f"{path.name} {pid}: unknown source {source!r}")
        annotator = REMARKED_BY_CLAUDE if source else REMARKED
        kept = 0
        exclusion = None
        changed_labels = False
        for row in group.itertuples():
            where = f"{path.name} {pid}/{row.category}"
            if row.status not in REMARK_STATUSES:
                raise AdjudicationError(f"{where}: status must be one of {REMARK_STATUSES}")
            if row.category not in taxonomy:
                raise AdjudicationError(f"{where}: unknown category")
            was = original[pid].get(row.category)
            if (row.status == "added") == (row.category in original[pid]):
                raise AdjudicationError(f"{where}: {row.status} doesn't match the labels")
            if row.status == "removed":
                summary["removed"] += 1
                changed_labels = True
                continue
            if taxonomy.get(row.category).is_exclusion:
                if row.confidence or row.evidence_sentences:
                    raise AdjudicationError(f"{where}: exclusions take no confidence or evidence")
                exclusion = row.category
                changed_labels |= row.status == "added"
                out.append({"passage_id": pid, "annotator": annotator, "category": row.category,
                            "confidence": None, "evidence": []})
                kept += 1
                continue
            confidence = row.confidence or was
            if confidence not in CONFIDENCE_LEVELS:
                raise AdjudicationError(f"{where}: confidence must be one of {CONFIDENCE_LEVELS}")
            evidence = _evidence(row.evidence_sentences, n_sentences[pid], where)
            if row.status == "added":
                if not evidence:
                    raise AdjudicationError(f"{where}: an added position needs its sentences")
                summary["added"] += 1
                changed_labels = True
            elif confidence != was:
                summary["confidence_changed"] += 1
                changed_labels = True
            out.append({"passage_id": pid, "annotator": annotator, "category": row.category,
                        "confidence": confidence, "evidence": evidence})
            kept += 1
        if not kept:
            raise AdjudicationError(f"{path.name} {pid}: every position was removed")
        if exclusion and kept > 1:
            raise AdjudicationError(f"{path.name} {pid}: {exclusion} with positions still kept")
        summary["passages"] += 1
        summary["evidence_only"] += not changed_labels
        summary["no_philosophy"] += exclusion is not None and changed_labels
        summary["by_claude"] += bool(source)
        note = next((n for n in group["note"] if n), "")  # the page repeats it on every row
        records.append({
            "passage_id": pid, "annotator": annotator, "role": "adjudication",
            "source_file": path.name, "format": "remark", "is_complete": True,
            "explicit_philosophy_flag": None, "timestamp": group["decided_on"].max() or None,
            "duration_seconds": None, "notes": "; ".join(n for n in (source, note) if n) or None,
            "issues": None,
        })
    return (pd.DataFrame(records, columns=RECORD_COLUMNS),
            pd.DataFrame(out, columns=LABEL_COLUMNS), summary)


REVIEW_COLUMNS = ["passage_id", "decision", "category", "confidence", "evidence_sentences",
                  "decided_on", "source"]


def load_label_review(
    path: Path, corpus: pd.DataFrame, taxonomy: Taxonomy
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """label_review_<version>.csv: one `keep` row, or `replace` rows that replace the labels.

    Returns (records, labels, decision); decision maps passage_id -> "kept"/"replaced"."""
    empty = (pd.DataFrame(columns=RECORD_COLUMNS), pd.DataFrame(columns=LABEL_COLUMNS), {})
    if not path.exists():
        return empty
    review = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REVIEW_COLUMNS) - set(review.columns)
    if missing:
        raise AdjudicationError(f"{path.name} is missing columns {sorted(missing)}")
    passages = corpus.set_index("passage_id")
    decisions, records, labels = {}, [], []
    for pid, rows in review.groupby("passage_id", sort=True):
        where = f"{path.name} {pid}"
        if pid not in passages.index:
            raise AdjudicationError(f"{where}: not a passage in the corpus")
        kinds = set(rows["decision"])
        if kinds == {"keep"}:
            label_fields = rows[["category", "confidence", "evidence_sentences"]]
            if len(rows) != 1 or (label_fields != "").any(axis=None):
                raise AdjudicationError(f"{where}: a keep decision is one row with no labels")
            decisions[pid] = "kept"
            continue
        if kinds != {"replace"}:
            raise AdjudicationError(f"{where}: decision must be all keep or all replace, "
                                    f"not {kinds}")
        if rows["category"].duplicated().any():
            raise AdjudicationError(f"{where}: a category is listed twice")
        chosen = set(rows["category"])
        if chosen & set(taxonomy.exclusion_ids) and len(chosen) > 1:
            raise AdjudicationError(f"{where}: exclusion category combined with others")
        for row in rows.itertuples():
            if row.category not in taxonomy:
                raise AdjudicationError(f"{where}: unknown category {row.category!r}")
            is_exclusion = taxonomy.get(row.category).is_exclusion
            confidence = row.confidence or None
            if is_exclusion and (confidence or row.evidence_sentences):
                raise AdjudicationError(f"{where}: exclusions take no confidence or evidence")
            if not is_exclusion and confidence not in CONFIDENCE_LEVELS:
                raise AdjudicationError(f"{where}: confidence must be one of {CONFIDENCE_LEVELS}")
            evidence = _evidence(row.evidence_sentences, passages.loc[pid, "n_sentences"], where)
            labels.append({"passage_id": pid, "annotator": ADJUDICATED, "category": row.category,
                           "confidence": confidence, "evidence": evidence})
        decisions[pid] = "replaced"
        records.append({
            "passage_id": pid, "annotator": ADJUDICATED, "role": "adjudication",
            "source_file": path.name, "format": "label_review", "is_complete": True,
            "explicit_philosophy_flag": None, "timestamp": rows["decided_on"].max() or None,
            "duration_seconds": None, "notes": rows["source"].iloc[0] or None, "issues": None,
        })
    return (pd.DataFrame(records, columns=RECORD_COLUMNS),
            pd.DataFrame(labels, columns=LABEL_COLUMNS), decisions)
