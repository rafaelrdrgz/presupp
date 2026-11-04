"""Load the v1 passage corpus into one DataFrame (one row per passage)."""

import csv
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

REDACTION_MARKER = "[political content removed]"

CORPUS_COLUMNS = [
    "passage_id", "origin", "source", "source_key", "publisher", "country", "genre",
    "domain", "article_id", "article_url", "url_norm", "article_title", "date",
    "paragraph_index", "text", "sentences", "n_sentences", "word_count", "priority",
    "heuristic_score", "markers_found", "has_redaction",
]


def url_domain(url: str) -> str:
    return urlsplit(url.strip()).netloc.lower().removeprefix("www.")


def normalise_url(url: str) -> str:
    """Article key: no scheme, www, trailing slash or #fragment (BBC URLs had #comments)."""
    parts = urlsplit(url.strip())
    key = url_domain(url) + parts.path.rstrip("/")
    if parts.query:
        key += "?" + parts.query
    return key.lower()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_publishers(path: Path) -> pd.DataFrame:
    publishers = pd.read_csv(path, dtype=str)
    if publishers["domain"].duplicated().any():
        raise ValueError(f"Duplicate domains in {path}")
    return publishers


def _read_csv_rows(path: Path) -> list[dict]:
    csv.field_size_limit(1 << 30)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_corpus(main_jsonl: Path, all_csv: Path, publishers_csv: Path) -> pd.DataFrame:
    """Main passages from the JSONL plus supplementary ones only in the CSV.

    Sentences always come from the CSV, the split the annotation tool showed."""
    with open(main_jsonl, encoding="utf-8") as f:
        main = [json.loads(line) for line in f if line.strip()]
    csv_rows = {r["id"]: r for r in _read_csv_rows(all_csv)}

    rows = []
    for p in main:
        extra = csv_rows.get(p["id"])
        if extra is None:
            raise ValueError(f"{p['id']} is in {main_jsonl.name} but not in {all_csv.name}")
        (paragraph_index,) = p["paragraph_indices"]
        rows.append({
            "passage_id": p["id"],
            "origin": "main",
            "source": p["source_name"],
            "source_key": p["source_key"],
            "article_id": p["article_id"],
            "article_url": p["article_url"],
            "article_title": p["article_title"],
            "date": None,
            "paragraph_index": paragraph_index,
            "text": p["text"],
            "sentences": json.loads(extra["sentences"]),
            "word_count": int(p["word_count"]),
            "priority": p["priority"],
            "heuristic_score": float(p["score"]),
            "markers_found": json.dumps(p.get("markers_found"), sort_keys=True),
        })

    main_ids = {p["id"] for p in main}
    for pid, r in csv_rows.items():
        if pid in main_ids:
            continue
        # supplementary ids look like p_<article hash>_<paragraph number>
        article_hash, _, paragraph = pid.rpartition("_")
        rows.append({
            "passage_id": pid,
            "origin": "supplementary",
            "source": r["source"],
            "source_key": _slug(r["source"]),
            "article_id": article_hash,
            "article_url": r["article_url"],
            "article_title": r["article_title"],
            "date": None if r["date"] in ("", "N/A") else r["date"],
            "paragraph_index": int(paragraph),
            "text": r["text"],
            "sentences": json.loads(r["sentences"]),
            "word_count": int(r["word_count"]),
            "priority": r["priority"] or None,
            "heuristic_score": float(r["score"]) if r["score"] else None,
            "markers_found": None,
        })

    corpus = pd.DataFrame(rows)
    corpus["domain"] = corpus["article_url"].map(url_domain)
    corpus["url_norm"] = corpus["article_url"].map(normalise_url)
    corpus["n_sentences"] = corpus["sentences"].map(len)
    corpus["has_redaction"] = corpus["text"].str.contains(REDACTION_MARKER, regex=False)

    publishers = load_publishers(publishers_csv)
    unknown = sorted(set(corpus["domain"]) - set(publishers["domain"]))
    if unknown:
        raise ValueError(f"Domains missing from {publishers_csv.name}: {unknown}")
    corpus = corpus.merge(publishers, on="domain", how="left", validate="many_to_one")

    if corpus["passage_id"].duplicated().any():
        raise ValueError("Duplicate passage ids in corpus")
    return corpus[CORPUS_COLUMNS].sort_values("passage_id", ignore_index=True)


def text_differences(main_jsonl: Path, all_csv: Path) -> list[str]:
    """Ids whose text differs between the JSONL and CSV exports."""
    with open(main_jsonl, encoding="utf-8") as f:
        main = {p["id"]: p["text"] for p in (json.loads(line) for line in f if line.strip())}
    csv_rows = {r["id"]: r["text"] for r in _read_csv_rows(all_csv)}
    return sorted(pid for pid, text in main.items() if csv_rows.get(pid, text) != text)
