"""Mapping evidence clicks made on one sentence split onto the stored sentences."""

import re

from presupp.data.realign import realign_passage, to_stored

TEXT = ("The plan was announced on Monday. It will force the rest of the U.S. political system to "
        "act. Critics say: It goes too far. Supporters disagree.")
STORED = ["The plan was announced on Monday.",
          "It will force the rest of the U.S. political system to act.",
          "Critics say: It goes too far.", "Supporters disagree."]


def plain(text, stored):  # cuts at "U.S." too
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def colons(text, stored):
    return [p for s in stored for p in re.split(r"(?<=:)\s+(?=[A-Z])", s)]


def test_clicks_map_to_the_stored_sentence_they_fall_in():
    seen = plain(TEXT, STORED)  # 5 sentences: the second is cut in two at "U.S."
    assert len(seen) == 5
    assert to_stored(TEXT, STORED, seen, [2, 3]) == [1]  # both halves are stored sentence 2
    assert to_stored(TEXT, STORED, seen, [1, 5]) == [0, 3]
    assert to_stored(TEXT, STORED, seen, [6]) is None


def test_a_passage_is_realigned_only_when_that_is_unambiguous():
    # the clicks run past the stored four sentences, and only the plain split has five
    found = realign_passage(TEXT, STORED, {"deontology": [2, 3, 5]}, {"plain": plain})
    assert found == ("plain", {"deontology": [1, 3]})
    # lists that fit the stored sentences are left alone
    assert realign_passage(TEXT, STORED, {"deontology": [2]}, {"plain": plain}) is None
    both = {"plain": plain, "colons": colons}  # each makes five sentences, cut differently
    # where they agree on what was clicked there is an answer ("Supporters disagree.")...
    assert realign_passage(TEXT, STORED, {"deontology": [2, 5]}, both) == (
        "plain", {"deontology": [1, 3]})
    # ...and where they don't, none (S3 is "political system to act." or "Critics say:")
    assert realign_passage(TEXT, STORED, {"deontology": [3, 5]}, both) is None
    # nothing fits
    assert realign_passage(TEXT, STORED, {"deontology": [7]}, both) is None
