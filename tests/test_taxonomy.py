import pytest

from presupp.taxonomy import load_taxonomy, parse_taxonomy


def test_v1_taxonomy_shape():
    tax = load_taxonomy("v1")
    assert len(tax.philosophical_ids) == 12
    assert tax.exclusion_ids == ["insufficient_philosophical_content", "insufficient_context"]
    assert tax.domains == ["Exclusion", "Metaphysical", "Epistemological", "Ethical",
                           "Political-Philosophical"]
    assert tax.domain_of("deontology") == "Ethical"
    assert "individualism" in tax and "utilitarianism" not in tax


def test_duplicate_ids_rejected():
    category = {"id": "x", "name": "X"}
    raw = {"domains": [{"name": "A Presuppositions", "categories": [category, category]}]}
    with pytest.raises(ValueError, match="Duplicate"):
        parse_taxonomy(raw, "test")
