import json

import pytest

from presupp.data.annotations import AnnotationFormatError, load_annotation_file, parse_categories
from presupp.taxonomy import load_taxonomy

TAX = load_taxonomy("v1")


def test_parses_v1_format_and_sorts_evidence():
    raw = json.dumps({"deontology": {"confidence": "medium", "evidence": [3, 1, 3]},
                      "individualism": {"confidence": None, "evidence": []}})
    parsed = parse_categories(raw, TAX)
    assert parsed.format == "v1"
    assert parsed.labels[0] == {
        "category": "deontology", "confidence": "medium", "evidence": [1, 3]
    }
    assert parsed.labels[1]["confidence"] is None


def test_empty_legacy_record_is_kept_as_incomplete():
    raw = json.dumps({"primary": [], "secondary": [], "confidence": "medium",
                      "evidence_sentences": [], "category_evidence": {}})
    parsed = parse_categories(raw, TAX)
    assert parsed.format == "legacy_empty" and parsed.labels == []


@pytest.mark.parametrize("raw, message", [
    ('{"primary": ["deontology"], "secondary": []}', "legacy"),
    ('{"utilitarianism": {"confidence": "high", "evidence": []}}', "unknown category"),
    ('{"deontology": {"confidence": "certain", "evidence": []}}', "confidence"),
    ('{"deontology": {"confidence": "high", "evidence": ["1"]}}', "evidence"),
    ('["deontology"]', "JSON object"),
    ("not json", "valid JSON"),
])
def test_rejects_what_it_cannot_interpret(raw, message):
    with pytest.raises(AnnotationFormatError, match=message):
        parse_categories(raw, TAX)


def test_flags_exclusion_combined_with_categories():
    raw = json.dumps({"insufficient_philosophical_content": {"confidence": None, "evidence": []},
                      "deontology": {"confidence": "high", "evidence": [0]}})
    assert parse_categories(raw, TAX).issues == ["exclusion_with_other_labels"]


def test_load_file_applies_aliases_and_rejects_double_annotation(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "id,explicit_philosophy_flag,categories\n"
        'OLD1,FALSE,"{""deontology"": {""confidence"": ""high"", ""evidence"": [0]}}"\n',
        encoding="utf-8",
    )
    records, labels = load_annotation_file(path, "primary", "primary", TAX, id_column="id",
                                           id_aliases={"OLD1": "p_abc_000"})
    assert records.passage_id.tolist() == ["p_abc_000"]
    assert records.explicit_philosophy_flag.tolist() == [False]
    assert labels.category.tolist() == ["deontology"]

    with open(path, "a", encoding="utf-8") as f:
        f.write('OLD1,FALSE,"{}"\n')
    with pytest.raises(AnnotationFormatError, match="annotated twice"):
        load_annotation_file(path, "primary", "primary", TAX, id_column="id")
