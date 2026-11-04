from presupp.data.corpus import normalise_url
from presupp.data.dedup import (
    build_groups,
    cluster_ids,
    near_duplicate_pairs,
    normalise_text,
    text_hash,
)


def test_normalise_url_merges_scrape_variants():
    a = normalise_url("https://www.bbc.co.uk/news/articles/c14regjvk6no#comments")
    b = normalise_url("http://bbc.co.uk/news/articles/C14REGJVK6NO/")
    assert a == b == "bbc.co.uk/news/articles/c14regjvk6no"
    assert normalise_url("https://x.org/a?page=2") != normalise_url("https://x.org/a?page=3")


def test_text_hash_ignores_case_whitespace_and_punctuation():
    assert normalise_text("Rights,  are\nINVIOLABLE!") == "rights are inviolable"
    assert text_hash("Rights are inviolable.") == text_hash("rights  are inviolable")
    assert text_hash("Rights are inviolable.") != text_hash("Rights are negotiable.")


def test_near_duplicates_found_and_distinct_texts_not():
    base = ("The committee concluded that the policy would reduce harm "
            "across every region it covered.")
    texts = [base, base.replace("every", "each"), "An entirely different sentence about soil."]
    pairs = near_duplicate_pairs(["a", "b", "c"], texts, threshold=0.8)
    assert [(a, b) for a, b, _ in pairs] == [("a", "b")]


def test_clusters_are_transitive_and_rooted_at_smallest_id():
    clusters = cluster_ids(["c", "b", "a", "z"], [("c", "b"), ("b", "a")])
    assert clusters == {"a": "a", "b": "a", "c": "a", "z": "z"}


def test_groups_join_articles_through_shared_text():
    # p1/p2 share an article; p3 is a copy of p2 under another URL; p4 is unrelated
    urls = ["site/a", "site/a", "site/b", "site/c"]
    text_cluster = ["p1", "p2", "p2", "p4"]
    groups = build_groups(urls, text_cluster)
    assert groups[0] == groups[1] == groups[2] != groups[3]
    # ids are stable: adding an unrelated article does not change existing ids
    more = build_groups(urls + ["site/d"], text_cluster + ["p5"])
    assert list(more[:4]) == list(groups)
