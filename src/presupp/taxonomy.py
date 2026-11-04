"""Category taxonomy, loaded from taxonomy/categories_<version>.json.

Don't hard-code the category list anywhere else.
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from presupp.paths import TAXONOMY_DIR

EXCLUSION_DOMAIN = "Exclusion"


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    domain: str
    description: str
    markers: tuple[str, ...]

    @property
    def is_exclusion(self) -> bool:
        return self.domain == EXCLUSION_DOMAIN


@dataclass(frozen=True)
class Taxonomy:
    version: str
    categories: tuple[Category, ...]

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.categories]

    @property
    def philosophical_ids(self) -> list[str]:
        return [c.id for c in self.categories if not c.is_exclusion]

    @property
    def exclusion_ids(self) -> list[str]:
        return [c.id for c in self.categories if c.is_exclusion]

    @property
    def domains(self) -> list[str]:
        return list(dict.fromkeys(c.domain for c in self.categories))

    def get(self, category_id: str) -> Category:
        for c in self.categories:
            if c.id == category_id:
                return c
        raise KeyError(f"{category_id!r} is not in taxonomy {self.version}")

    def domain_of(self, category_id: str) -> str:
        return self.get(category_id).domain

    def __contains__(self, category_id: object) -> bool:
        return any(c.id == category_id for c in self.categories)


def _short_domain_name(name: str) -> str:
    # "Ethical Presuppositions" -> "Ethical", "Exclusion Categories" -> "Exclusion"
    for suffix in (" Presuppositions", " Categories"):
        name = name.removesuffix(suffix)
    return name


def parse_taxonomy(raw: dict, version: str) -> Taxonomy:
    categories = []
    for domain in raw["domains"]:
        domain_name = _short_domain_name(domain["name"])
        for c in domain["categories"]:
            categories.append(
                Category(
                    id=c["id"],
                    name=c["name"],
                    domain=domain_name,
                    description=c.get("description", ""),
                    markers=tuple(c.get("markers", [])),
                )
            )
    ids = [c.id for c in categories]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate category ids in taxonomy {version}")
    return Taxonomy(version=version, categories=tuple(categories))


@cache
def load_taxonomy(version: str = "v1", directory: Path = TAXONOMY_DIR) -> Taxonomy:
    with open(directory / f"categories_{version}.json", encoding="utf-8") as f:
        return parse_taxonomy(json.load(f), version)
