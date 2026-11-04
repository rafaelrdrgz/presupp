"""Duplicate detection and leakage groups.

Text cluster: identical or near-identical text. Split group: same article or same text cluster.
"""

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # smallest member as root keeps ids stable
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra

    def components(self) -> dict[str, str]:
        return {x: self.find(x) for x in self.parent}


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^\w]+", " ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalise_text(text).encode("utf-8")).hexdigest()[:16]


def near_duplicate_pairs(
    ids: Sequence[str], texts: Sequence[str], threshold: float
) -> list[tuple[str, str, float]]:
    """(a, b, cosine) for pairs with char 5-gram TF-IDF cosine >= threshold."""
    vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(5, 5), min_df=1).fit_transform(
        [normalise_text(t) for t in texts]
    )
    sims = (vectors @ vectors.T).tocoo()
    pairs = []
    for i, j, s in zip(sims.row, sims.col, sims.data, strict=True):
        if i < j and s >= threshold:
            pairs.append((ids[i], ids[j], float(s)))
    return sorted(pairs)


def cluster_ids(ids: Sequence[str], links: Iterable[tuple[str, str]]) -> dict[str, str]:
    """id -> smallest member of its connected component."""
    uf = UnionFind(ids)
    for a, b in links:
        uf.union(a, b)
    return uf.components()


def group_key(members: Iterable[str]) -> str:
    """Text-free group id, hashed from the smallest member."""
    return "g" + hashlib.sha1(min(members).encode("utf-8")).hexdigest()[:10]


def build_groups(url_keys: Sequence[str], text_cluster: Sequence[str]) -> np.ndarray:
    """Split group per passage; ids come from the smallest URL key, stable as the corpus grows."""
    nodes = [f"url:{u}" for u in url_keys] + [f"txt:{c}" for c in text_cluster]
    uf = UnionFind(set(nodes))
    for u, c in zip(url_keys, text_cluster, strict=True):
        uf.union(f"url:{u}", f"txt:{c}")
    members: dict[str, set[str]] = {}
    for u in url_keys:
        members.setdefault(uf.find(f"url:{u}"), set()).add(u)
    root_to_group = {root: group_key(urls) for root, urls in members.items()}
    return np.array([root_to_group[uf.find(f"url:{u}")] for u in url_keys])
