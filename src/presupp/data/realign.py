"""Map the primary annotator's evidence clicks onto the stored sentences.

Some passages were clicked on a different sentence split from the stored NLTK one.
"""

from collections.abc import Callable

Splitter = Callable[[str, list[str]], list[str]]  # (text, stored sentences) -> sentences


def spans(text: str, sentences: list[str]) -> list[tuple[int, int]] | None:
    """(start, end) of each sentence in text, or None if they aren't all found in order."""
    out, at = [], 0
    for sentence in sentences:
        sentence = sentence.strip()
        start = text.find(sentence[:40], at)
        if start < 0:
            return None
        out.append((start, start + len(sentence)))
        at = start + 1
    return out


def to_stored(text: str, stored: list[str], seen: list[str], clicks: list[int]
              ) -> list[int] | None:
    """0-based stored sentences holding the midpoints of the 1-based clicks on `seen`."""
    a, b = spans(text, stored), spans(text, seen)
    if a is None or b is None or max(clicks) > len(seen) or min(clicks) < 1:
        return None
    found = set()
    for click in clicks:
        start, end = b[click - 1]
        middle = (start + end) // 2
        found |= {i for i, (x, y) in enumerate(a) if x <= middle < y}
    return sorted(found)


def realign_passage(text: str, stored: list[str], lists: dict[str, list[int]],
                    splitters: dict[str, Splitter]) -> tuple[str, dict[str, list[int]]] | None:
    """(splitter, {category: 0-based stored indices}) for {category: 1-based clicks}.

    None if the clicks already fit, no splitter fits, or the fitting splitters disagree."""
    if all(1 <= min(c) and max(c) <= len(stored) for c in lists.values()):
        return None
    answers = []
    for name, split in splitters.items():
        seen = split(text, stored)
        if not all(1 <= min(c) and max(c) <= len(seen) for c in lists.values()):
            continue
        mapped = {cat: to_stored(text, stored, seen, clicks) for cat, clicks in lists.items()}
        if any(m is None for m in mapped.values()):
            continue
        answers.append((name, mapped))
    if not answers or any(m != answers[0][1] for _, m in answers[1:]):
        return None
    return answers[0]
