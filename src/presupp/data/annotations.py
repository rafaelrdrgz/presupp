"""Parse v1 annotation-tool exports into a records table and a labels table.

Legacy {"primary": [], "secondary": [], ...} records are all empty; kept as incomplete.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from presupp.taxonomy import Taxonomy

CONFIDENCE_LEVELS = ("high", "medium", "low")
LEGACY_KEYS = {"primary", "secondary"}

RECORD_COLUMNS = [
    "passage_id", "annotator", "role", "source_file", "format", "is_complete",
    "explicit_philosophy_flag", "timestamp", "duration_seconds", "notes", "issues",
]
LABEL_COLUMNS = ["passage_id", "annotator", "category", "confidence", "evidence"]


class AnnotationFormatError(ValueError):
    pass


@dataclass
class ParsedCategories:
    format: str  # "v1" or "legacy_empty"
    labels: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def parse_categories(raw: str, taxonomy: Taxonomy) -> ParsedCategories:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise AnnotationFormatError(f"categories is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise AnnotationFormatError(f"categories must be a JSON object, got {type(data).__name__}")

    if LEGACY_KEYS <= data.keys():
        has_content = data.get("primary") or data.get("secondary") or data.get("category_evidence")
        if has_content:
            raise AnnotationFormatError("legacy-format record with content: needs a converter")
        return ParsedCategories(format="legacy_empty")

    labels = []
    for category, detail in data.items():
        if category not in taxonomy:
            raise AnnotationFormatError(f"unknown category {category!r}")
        detail = detail or {}
        confidence = detail.get("confidence")
        if confidence is not None and confidence not in CONFIDENCE_LEVELS:
            raise AnnotationFormatError(f"{category}: unknown confidence {confidence!r}")
        evidence = detail.get("evidence") or []
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in evidence):
            raise AnnotationFormatError(f"{category}: evidence must be a list of ints")
        labels.append(
            {"category": category, "confidence": confidence, "evidence": sorted(set(evidence))}
        )

    issues = []
    chosen = {lab["category"] for lab in labels}
    exclusions = chosen & set(taxonomy.exclusion_ids)
    if exclusions and len(chosen) > len(exclusions):
        issues.append("exclusion_with_other_labels")
    return ParsedCategories(format="v1", labels=labels, issues=issues)


def _parse_flag(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    v = value.strip().upper()
    if v in {"TRUE", "1", "YES"}:
        return True
    if v in {"FALSE", "0", "NO"}:
        return False
    raise AnnotationFormatError(f"cannot read explicit_philosophy_flag {value!r}")


def load_annotation_file(
    path: Path,
    annotator: str,
    role: str,
    taxonomy: Taxonomy,
    id_column: str = "passage_id",
    id_aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (records, labels)."""
    id_aliases = id_aliases or {}
    records, labels = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            passage_id = id_aliases.get(row[id_column], row[id_column])
            try:
                parsed = parse_categories(row["categories"], taxonomy)
                flag = _parse_flag(row.get("explicit_philosophy_flag"))
            except AnnotationFormatError as e:
                raise AnnotationFormatError(f"{path.name}:{line_no} ({passage_id}): {e}") from e
            duration = row.get("duration_seconds")
            records.append({
                "passage_id": passage_id,
                "annotator": annotator,
                "role": role,
                "source_file": path.name,
                "format": parsed.format,
                "is_complete": bool(parsed.labels),
                "explicit_philosophy_flag": flag,
                "timestamp": row.get("timestamp") or None,
                "duration_seconds": float(duration) if duration else None,
                "notes": row.get("notes") or None,
                "issues": ";".join(parsed.issues) or None,
            })
            for lab in parsed.labels:
                labels.append({"passage_id": passage_id, "annotator": annotator, **lab})

    records_df = pd.DataFrame(records, columns=RECORD_COLUMNS)
    labels_df = pd.DataFrame(labels, columns=LABEL_COLUMNS)
    dupes = records_df["passage_id"].duplicated()
    if dupes.any():
        twice = records_df.loc[dupes, "passage_id"].tolist()
        raise AnnotationFormatError(f"{path.name}: passage ids annotated twice: {twice}")
    return records_df, labels_df


def label_sets(labels: pd.DataFrame, annotator: str) -> dict[str, frozenset[str]]:
    """passage_id -> categories, for one annotator."""
    sub = labels[labels["annotator"] == annotator]
    return {pid: frozenset(g["category"]) for pid, g in sub.groupby("passage_id")}
