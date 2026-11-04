"""Turn pasted text into passages like the training ones (about 150-250 words).
Paragraphs that aren't English prose are skipped, with the reason kept for display.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace

MAX_INPUT_CHARS = 60_000
MAX_PASSAGES = 20  # about 4,000 words
TARGET_WORDS = 150  # close a passage at this; 95% of training passages reach it
MAX_WORDS = 350     # hard cap; the longest training passage has 357
CHUNK_WORDS = 200   # longer paragraphs are cut at sentence ends into about this
SHORT_WORDS = 100   # 99% of training passages are longer than this
MIN_WORDS = 40      # less prose than this in total isn't classified

# Why a paragraph wasn't read
FRAGMENT = "fragment"        # headline, byline, date, menu item
BOILERPLATE = "boilerplate"  # cookie notices, newsletter sign-ups, photo credits
REPEAT = "repeat"            # page headers, repeated captions
NOT_PROSE = "not_prose"      # contents lines, tables, figures
NOT_ENGLISH = "not_english"
UNREAD = "unread"            # past the limits

# Why nothing was read
EMPTY, LINK, ONLY_NOT_ENGLISH, TOO_SHORT = "empty", "link", "not_english", "too_short"


@dataclass(frozen=True)
class Block:
    """A paragraph of the input, or a piece of a long one."""
    text: str
    words: int
    skip: str | None = None  # why it wasn't read
    continues: bool = False  # a later piece of a cut paragraph


@dataclass(frozen=True)
class Passage:
    """What the model reads: a run of blocks, minus skipped ones."""
    blocks: tuple[int, ...]
    text: str
    words: int


@dataclass(frozen=True)
class Reading:
    blocks: tuple[Block, ...]
    passages: tuple[Passage, ...]
    problem: str | None = None  # why nothing was read
    unread_words: int = 0       # cut off by MAX_INPUT_CHARS or MAX_PASSAGES


def read(text: str | None) -> Reading:
    text = normalise(text or "")
    if not text.strip():
        return Reading((), (), EMPTY)
    if re.fullmatch(r"\s*(?:https?://|www\.)\S+\s*", text):
        return Reading((), (), LINK)
    unread = 0
    if len(text) > MAX_INPUT_CHARS:
        cut = text.rfind("\n", 0, MAX_INPUT_CHARS)
        cut = cut if cut > MAX_INPUT_CHARS // 2 else MAX_INPUT_CHARS
        unread = len(text[cut:].split())
        text = text[:cut]
    blocks = _blocks(paragraphs(text))
    passages = _passages(blocks)
    if len(passages) > MAX_PASSAGES:
        dropped = {i for p in passages[MAX_PASSAGES:] for i in p.blocks}
        unread += sum(p.words for p in passages[MAX_PASSAGES:])
        blocks = [replace(b, skip=UNREAD) if i in dropped else b for i, b in enumerate(blocks)]
        passages = passages[:MAX_PASSAGES]
    problem = None
    if sum(p.words for p in passages) < MIN_WORDS:
        foreign = sum(len(b.text) for b in blocks if b.skip == NOT_ENGLISH)
        problem = ONLY_NOT_ENGLISH if foreign >= 5 * MIN_WORDS else TOO_SHORT  # in characters
        passages = []
    return Reading(tuple(blocks), tuple(passages), problem, unread)


# ---- paragraphs ----

_SPACES = dict.fromkeys(map(ord, "\t\xa0  　"), " ")
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"))
_LINE_BREAKS = dict.fromkeys(map(ord, "\r  \x0b\x0c"), "\n")


_NOTE_MARK = re.compile(r"\[(?:\d{1,3}|[a-z]|citation needed|clarification needed)\]")


def normalise(text: str) -> str:
    """Plain spaces and line breaks, no invisible characters or footnote marks like "[12]"."""
    text = text.replace("\r\n", "\n").translate(_LINE_BREAKS).translate(_SPACES)
    text = _NOTE_MARK.sub("", text.translate(_INVISIBLE))
    return unicodedata.normalize("NFC", text)


def paragraphs(text: str) -> list[str]:
    """Split on blank lines, and on single line breaks unless the lines look hard-wrapped."""
    out = []
    for block in re.split(r"\n[ ]*\n", text):
        lines = [" ".join(line.split()) for line in block.split("\n")]
        lines = [line for line in lines if line]
        if lines:
            out += _unwrap(lines) if _wrapped(lines) else lines
    return out


_END = re.compile(r"""[.!?…:;]["'”’)\]]*$""")
_BULLET = re.compile(r"^(?:[•·▪●◦○■□‣⁃–—*-]|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)])\s+\S")


def _wrap_width(lines: list[str]) -> int:
    lengths = sorted(len(line) for line in lines)
    return lengths[int(0.8 * (len(lengths) - 1))]


def _wrapped(lines: list[str]) -> bool:
    """Most lines run to about the same width and stop mid-sentence."""
    if len(lines) < 2:
        return False
    width, body = _wrap_width(lines), lines[:-1]
    if not 40 <= width <= 130:  # longer lines are whole paragraphs; shorter ones, a list
        return False
    full = sum(len(line) >= 0.7 * width for line in body)
    open_ended = sum(not _END.search(line) for line in body)
    return full >= 0.6 * len(body) and open_ended >= 0.5 * len(body)


def _unwrap(lines: list[str]) -> list[str]:
    """Join wrapped lines. A short line ends a paragraph; a list item starts one."""
    width = _wrap_width(lines)
    out, current = [], ""
    for line in lines:
        if current and _BULLET.match(line):
            out.append(current)
            current = ""
        if not current:
            current = line
        elif re.search(r"[A-Za-z]-$", current) and line[:1].isalpha():
            current += line  # keep the hyphen, it may be a compound
        else:
            current += " " + line
        if len(line) < 0.7 * width:
            out.append(current)
            current = ""
    return out + [current] if current else out


# ---- what is read ----

_BOILERPLATE = re.compile("|".join([
    r"\bcookies?\b.*\b(?:accept|consent|settings|preferences|policy)\b",
    r"\b(?:we|this (?:site|website)) uses? cookies\b",
    r"\bnewsletters?\b.*\b(?:sign up|subscribe|inbox)\b",
    r"\b(?:sign up|subscribe)\b.*\bnewsletters?\b",
    r"^(?:advertisement|sponsored( content)?|skip to (?:main )?content|share (?:this|on)\b.*"
    r"|read more\b.*|related (?:articles|stories|content|topics)\b.*)$",
    r"\ball rights reserved\b", "©",
    r"\b(?:photograph|photo|image|illustration|picture)(?: credit)?s?: ",
    r"\b(?:getty images|shutterstock|alamy)\b",
]), re.IGNORECASE)
_LEADER = re.compile(r"(?:\.\s?){4,}\s*\d+\s*$|(?:\s\.){3,}")
# reference-list lines; the weak signs only count in short lines
_REFERENCE = re.compile(r"↑|↩|\bISBN\b|\bdoi:|\bRetrieved\b.*\d{4}|\bArchived from\b"
                        r"|\bIn: [A-Z].*\d{4}|\b(?:Band|Bd\.|Nr\.|S\.) \d")
_REFERENCE_WEAK = re.compile(r"https?://|\bpp?\.\s?\d|\(eds?\.\)|\(Hrsg\.\)|\baccessed\b.*\d{4}",
                             re.IGNORECASE)

# common English words: 40-50% of English prose, under 10% of French, German or Spanish
_ENGLISH = frozenset("""the of and to a in is that for it with as was on are be by this an
from at or which have has not but they their its been were will would can more than there these
those our you your he she his her them all such what when who other if may how could any some
only into about also so no should because between while where after before most many we over
just""".split())
# French, German, Spanish, Italian, Portuguese and Dutch function words
_FOREIGN = frozenset("""le la les des et est une du que qui dans pour pas sur au aux avec ce
cette ces sont nous vous leur mais ou très je ne il elle en der die das und ist nicht ein eine
einen zu den von mit sich dem auch werden wird durch oder bei aus nach wie dass auf für im sie
er sind haben wenn nur kann es el los las del se por con para una su al lo como más pero sus y
ser fue han este esta entre sobre di che della sono per non nel gli dei uma um os em não com ao
pela pelo também muito het een van niet zijn voor met ook dat wordt""".split())


def _words(text: str) -> int:
    return len(text.split())


def _title_case(text: str) -> bool:
    """A heading like "7. Arguments for Consequentialism" rather than a list item."""
    words = [w for w in text.split()[1:] if w[:1].isalpha()]
    return not _END.search(text) and sum(w[0].isupper() for w in words) >= 0.6 * len(words)


def _other_script(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 20 and sum(c.isascii() for c in letters) < 0.7 * len(letters)


def _english(text: str, words: int) -> bool:
    tokens = re.findall(r"[^\W\d_]+", text.lower())
    english = sum(t in _ENGLISH for t in tokens)
    if sum(t in _FOREIGN for t in tokens) > english:
        return False
    # a long line with few function words is a list, not prose
    return words < 12 or english >= 0.12 * len(tokens)


def _skip_reason(text: str, words: int) -> str | None:
    if words < 40 and _BOILERPLATE.search(text):
        return BOILERPLATE
    if _other_script(text):  # before the word counts: Chinese or Japanese has few spaces
        return NOT_ENGLISH
    visible = text.replace(" ", "")
    if (_LEADER.search(text) or (words < 150 and _REFERENCE.search(text))
            or (words < 40 and _REFERENCE_WEAK.search(text))
            or sum(c.isalpha() for c in visible) < 0.7 * len(visible)):
        return NOT_PROSE
    sentence = words >= 5 and _END.search(text)
    item = words >= 4 and _BULLET.match(text) and not _title_case(text)
    if words < 15 and not (sentence or item) or words < 30 and " | " in text:
        return FRAGMENT
    if not _english(text, words):
        return NOT_ENGLISH
    return None


def _repeats(texts: list[str]) -> list[bool]:
    """Which lines to skip as repeats: short ones every time, longer ones after the first."""
    keys = [re.sub(r"[\d\s]+", " ", t.lower()).strip() for t in texts]
    counts, seen = Counter(keys), set()
    out = []
    for key, text in zip(keys, texts, strict=True):
        out.append(counts[key] > 1 and (_words(text) < 25 or key in seen))
        seen.add(key)
    return out


def _blocks(texts: list[str]) -> list[Block]:
    out = []
    for text, repeated in zip(texts, _repeats(texts), strict=True):
        words = _words(text)
        skip = _skip_reason(text, words) or (REPEAT if repeated else None)
        if skip or words <= MAX_WORDS:
            out.append(Block(text, words, skip))
        else:
            out += [Block(piece, _words(piece), None, continues=i > 0)
                    for i, piece in enumerate(_cut(text))]
    return out


# space after a sentence end; closing quotes and brackets stay with the sentence
_SENTENCE_END = re.compile(r"""(?:(?<=[.!?…])|(?<=[.!?…]["'”’)\]]))\s+(?=["'“‘(\[]?[A-Z0-9])""")


def sentences(text: str) -> list[str]:
    """A paragraph's sentences; " ".join gives the paragraph back."""
    return [s for s in _SENTENCE_END.split(text) if s]


def _cut(text: str) -> list[str]:
    """Pieces of about CHUNK_WORDS words, cut at sentence ends."""
    pieces, current = [], []
    for sentence in sentences(text):
        words = sentence.split()
        while len(words) > MAX_WORDS:  # one "sentence" this long has no punctuation to cut at
            pieces.append(" ".join(current + words[:CHUNK_WORDS - len(current)]))
            words, current = words[CHUNK_WORDS - len(current):], []
        if current and len(current) + len(words) > MAX_WORDS:
            pieces.append(" ".join(current))
            current = []
        current += words
        if len(current) >= CHUNK_WORDS:
            pieces.append(" ".join(current))
            current = []
    if current:
        if pieces and _words(pieces[-1]) + len(current) <= MAX_WORDS:
            pieces[-1] += " " + " ".join(current)
        else:
            pieces.append(" ".join(current))
    return pieces


def _passages(blocks: list[Block]) -> list[Passage]:
    """Group read blocks into passages of TARGET_WORDS to MAX_WORDS words."""
    groups: list[list[int]] = []
    current: list[int] = []

    def size(group: list[int]) -> int:
        return sum(blocks[i].words for i in group)

    def close() -> None:
        # a short group joins the previous one if it fits
        room = groups and size(groups[-1]) + size(current) <= MAX_WORDS
        if size(current) < TARGET_WORDS and room:
            groups[-1] += current
        else:
            groups.append(list(current))
        current.clear()

    for i, block in enumerate(blocks):
        if block.skip:
            continue
        if current and size(current) + block.words > MAX_WORDS:
            close()
        current.append(i)
        if size(current) >= TARGET_WORDS:
            close()
    if current:
        close()
    return [Passage(tuple(g), " ".join(blocks[i].text for i in g), size(g)) for g in groups]
