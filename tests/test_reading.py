"""Reading pasted text into passages: the shapes real pastes come in (a careful selection, a
whole web page, a PDF's wrapped lines) and the lines that must not reach the model."""

from presupp import reading
from presupp.reading import paragraphs, read

SENTENCE = "The council said that the {} scheme would be reviewed by the end of the year."  # 16
_made = iter(range(10**6))


def para(n: int) -> str:
    """A paragraph of n sentences (16 words each), different from every other one made, since
    a repeated paragraph is only read once."""
    k, tag = next(_made), ""
    while True:
        k, digit = divmod(k, 26)
        tag += chr(ord("a") + digit)
        if not k:
            break
    return " ".join([SENTENCE.format("housing-" + tag)] * n)


def skips(text: str) -> list[str | None]:
    return [b.skip for b in read(text).blocks]


# ---- paragraphs ----

def test_normalise_removes_what_web_pages_and_pdfs_add():
    text = "Rights matter.​ It is a funda­mental point,[12] and\r\nit stands.[a]"
    assert reading.normalise(text) == "Rights matter. It is a fundamental point, and\nit stands."


def test_blank_lines_and_single_line_breaks_both_end_paragraphs():
    long = para(2)
    assert paragraphs(f"{long}\n\n{long}") == [long, long]
    assert paragraphs(f"{long}\n{long}\nShort heading") == [long, long, "Short heading"]


def test_hard_wrapped_lines_are_joined_back_into_paragraphs():
    wrapped = ("Employers wanted their workforce to be more diverse, but there were various\n"
               "barriers to achieving this, including the lack of a clear career path and the\n"
               "cost of training staff who had come into the sector from other fields of\n"
               "work. This was a barrier for smaller firms.\n"
               "Executive summary\n"
               "• A survey of 118 firms and public sector organisations, including those whose\n"
               "core business was developing AI-led products or services and many others in\n"
               "wider sectors, carried out over the summer and autumn of the year;\n"
               "• A total of 50 in-depth interviews with firms and their employees, each of an\n"
               "hour or more.")
    assert paragraphs(wrapped) == [
        "Employers wanted their workforce to be more diverse, but there were various barriers "
        "to achieving this, including the lack of a clear career path and the cost of training "
        "staff who had come into the sector from other fields of work. This was a barrier for "
        "smaller firms.",
        "Executive summary",
        "• A survey of 118 firms and public sector organisations, including those whose core "
        "business was developing AI-led products or services and many others in wider "
        "sectors, carried out over the summer and autumn of the year;",
        "• A total of 50 in-depth interviews with firms and their employees, each of an hour "
        "or more."]


def test_a_word_split_across_wrapped_lines_keeps_its_hyphen():
    lines = ("The trial, which ran for a full year in three cities, found that the self-\n"
             "driving buses were safe enough to carry passengers on the main routes, and\n"
             "that they could run in all but the worst weather, according to the report.")
    assert len(paragraphs(lines)) == 1
    assert "self-driving buses" in paragraphs(lines)[0]


def test_sentences_split_a_paragraph_and_join_back_into_it():
    text = 'He said it was "fair." Then he left. Prices rose 3.5% (a record). 2024 was worse.'
    assert reading.sentences(text) == ['He said it was "fair."', "Then he left.",
                                       "Prices rose 3.5% (a record).", "2024 was worse."]
    assert " ".join(reading.sentences(text)) == text


# ---- what isn't read ----

def test_headings_bylines_and_menus_are_fragments():
    text = "\n\n".join(["Opinion", "Simon Writer", "Wed 28 Oct 2026 06.00 GMT", "Share",
                      "Why the council must think again about its housing plans", para(10)])
    assert skips(text)[:5] == [reading.FRAGMENT, reading.FRAGMENT, reading.NOT_PROSE,
                               reading.FRAGMENT, reading.FRAGMENT]
    assert skips(text)[5] is None


def test_short_sentences_and_list_items_are_read():
    text = "\n\n".join([para(10), "The minister declined to comment.",
                      "• Lower fares for young people", "7. Arguments for Consequentialism"])
    assert skips(text) == [None, None, None, reading.FRAGMENT]  # the last is a numbered heading


def test_site_text_is_boilerplate():
    for line in ["We use cookies to improve your experience. Accept cookies or manage settings.",
                 "Sign up to our free newsletter for the best stories in your inbox each week.",
                 "The minister at a press conference on Tuesday. Photograph: A Person/Getty Images",
                 "Read more: what the budget means for you"]:
        assert skips(f"{para(10)}\n\n{line}")[1] == reading.BOILERPLATE, line


def test_contents_tables_and_references_are_not_prose():
    for line in ["Training needs ................................ 40",
                 "41% 48% 9% 2,300 1,450 12.5 17.9 22.0",
                 "Smith, J. (2004). The Theory of Things. London: Press. ISBN 978-0-00-000000-0.",
                 "↑ \"Bread and Flour Regulations\". UK Flour Millers. October 2023."]:
        assert skips(f"{para(10)}\n\n{line}")[1] == reading.NOT_PROSE, line


def test_running_headers_are_skipped_every_time_and_repeated_paragraphs_after_the_first():
    header, body = "Annual Report on the State of the Market 2024 | Page", para(10)
    text = "\n\n".join([f"{header} 4", body, f"{header} 5", body])
    assert skips(text) == [reading.FRAGMENT, None, reading.FRAGMENT, reading.REPEAT]
    caption = "Ministers arrive for the cabinet meeting in Downing Street on a cold Tuesday."
    assert skips("\n\n".join([caption, caption, para(10)]))[:2] == [reading.REPEAT] * 2


def test_other_languages_are_not_read():
    german = ("Die Regierung hat am Montag angekündigt, dass die neuen Regeln ab dem nächsten "
              "Jahr für alle Schulen gelten werden, und die Opposition kritisiert den Plan.")
    spanish = ("El gobierno anunció el lunes que las nuevas normas se aplicarán a todas las "
               "escuelas a partir del próximo año, y la oposición criticó el plan.")
    short_german = "Diese Maxime ging auch in das Gesetz von 2005 ein."
    chinese = "政府周一宣布，新规定将从明年起适用于所有学校，反对派批评了这一计划。"
    for line in [german, spanish, short_german, chinese]:
        assert skips(f"{para(10)}\n\n{line}")[1] == reading.NOT_ENGLISH, line
    assert read("\n\n".join([german, spanish] * 3)).problem == reading.ONLY_NOT_ENGLISH
    assert read(chinese * 8).problem == reading.ONLY_NOT_ENGLISH


def test_a_list_of_names_is_not_english_prose():
    names = ("The commissioners are: Baroness Jane Example (Chair), Matthew Sample, Andrew "
             "Placeholder, Louise Specimen, Dr Patricia Instance, Simon Model, Ann Exemplar.")
    assert skips(f"{para(10)}\n\n{names}")[1] == reading.NOT_ENGLISH


# ---- passages ----

def test_paragraphs_are_grouped_into_passages_of_the_training_size():
    paras = [para(2) for _ in range(12)]  # 12 paragraphs of 32 words
    r = read("\n\n".join(paras))
    # five paragraphs reach 150 words; the last two are too few alone, so join the second
    assert [p.words for p in r.passages] == [160, 224]
    assert r.passages[0].text == " ".join(paras[:5])
    assert r.passages[1].blocks == tuple(range(5, 12))


def test_a_closing_quote_stays_with_its_sentence_when_a_paragraph_is_cut():
    quoted = 'The minister said the plan was "fair and right." ' + para(30)
    r = read(quoted)
    assert r.passages[0].text.startswith('The minister said the plan was "fair and right." The')


def test_a_short_tail_joins_the_passage_before_it():
    r = read("\n\n".join([para(10), para(2)]))  # 160 words, then 32
    assert [p.words for p in r.passages] == [192]
    assert r.passages[0].blocks == (0, 1)


def test_a_long_paragraph_is_cut_at_sentence_ends():
    r = read(para(40))  # 640 words in one paragraph
    assert [b.continues for b in r.blocks] == [False, True, True]
    assert all(p.words <= reading.MAX_WORDS for p in r.passages)
    assert all(p.text.endswith("year.") for p in r.passages)
    assert sum(p.words for p in r.passages) == 640


def test_skipped_lines_inside_a_passage_stay_where_they_were():
    r = read("\n\n".join([para(3), "Share", para(7)]))
    assert len(r.passages) == 1
    assert r.passages[0].blocks == (0, 2)
    assert "Share" not in r.passages[0].text


# ---- whole inputs ----

def test_inputs_that_cannot_be_read():
    assert read("").problem == reading.EMPTY
    assert read("  \n ").problem == reading.EMPTY
    assert read("https://www.example.com/news/article-123").problem == reading.LINK
    assert read("Short. Too short to read.").problem == reading.TOO_SHORT
    assert read("\n".join(["Home", "News", "Sport", "Weather"])).problem == reading.TOO_SHORT


def test_text_past_the_limits_is_counted_but_not_read():
    many = "\n\n".join(para(10) for _ in range(reading.MAX_PASSAGES + 3))
    r = read(many)
    assert len(r.passages) == reading.MAX_PASSAGES
    assert r.unread_words == 3 * 160
    assert sum(b.skip == reading.UNREAD for b in r.blocks) == 3
    huge = "\n\n".join(para(10) for _ in range(80))  # longer than MAX_INPUT_CHARS too
    assert len(huge) > reading.MAX_INPUT_CHARS
    r = read(huge)
    assert sum(p.words for p in r.passages) + r.unread_words == 80 * 160
