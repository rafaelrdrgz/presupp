# Data

| folder | in git | contents |
|---|---|---|
| `raw/v1/` | no | the original annotation exports (file list in `reports/data_build_v1.md`) |
| `processed/` | no | build outputs, which contain article text |
| `reference/` | yes | hand-maintained lookup tables |
| `adjudication/` | yes | label decisions made after annotation (ids and labels only) |
| `splits/` | yes | fold assignments (ids only) |

Article text stays out of git because most sources are copyrighted. GOV.UK text is under the
Open Government Licence and could be released; the rest could be shared as ids, URLs and labels.

`python -m presupp.data.build` reads `raw/v1/` and `adjudication/` and writes `processed/`,
`splits/` and `reports/data_build_v1.md`. It is deterministic, and once `splits/folds_v1.csv`
exists it keeps every passage's fold, so label corrections never move a passage between folds.

## Processed tables

**`passages.parquet`**: one row per passage (2,373).
- Identity: `passage_id`, `origin` (`main` = scraped corpus, `supplementary` = 79 passages
  added later).
- Source: `source` is the scraper feed. `publisher`, `country` and `genre` come from the URL's
  domain via `reference/publishers.csv`, because feeds sometimes carry other publishers'
  articles.
- Article: `article_id`, `article_url`, `url_norm` (scheme, `www.` and `#fragment` removed).
- Content: `text`, `sentences` (the segmentation the annotation tool displayed), `n_sentences`,
  `word_count`.
- Scraper metadata: `priority`, `heuristic_score`, `markers_found`.
- Duplicates: `text_hash`, `text_cluster` (identical or near-identical text),
  `text_cluster_size`.
- Splits: `group_id` (same article or same text cluster), `fold`, `split` (`dev` or `lockbox`),
  `is_labelled`, `is_canonical`, `has_primary_annotation`, `label_review`.

**`annotation_records.parquet`**: one row per (passage, annotator).
- `format` (`v1` or `legacy_empty`), `is_complete`, `explicit_philosophy_flag`.
- Expert rows also have `timestamp` and `duration_seconds`.

**`annotation_labels.parquet`**: one row per (passage, annotator, category).
- `confidence`, `evidence` (sentence indices as recorded), `evidence_in_range_0based` /
  `_1based`, `evidence_convention`.
- `evidence_0based` is the evidence to use, as indices into the stored sentences (empty when
  none fits), and `evidence_reading` says how it was read. Expert evidence is 0-based. The
  primary annotator's clicks are 1-based S-numbers, apart from a few lists stored 0-based and
  some realigned from another sentence split (see below). Re-marked passages get a second set of
  rows with annotator `remarked` or `remarked_claude`.

**`gold_labels.parquet`**: the labels models train and are scored on (1,239 passages).
- One copy per duplicated text (`is_canonical`); the other copies' annotations stay in
  `annotation_labels.parquet`.
- Labels come from `adjudication/evidence_remarked_v1.csv` for a re-marked passage, then from
  `duplicates_v1.csv` or `label_review_v1.csv` where a decision exists, otherwise from the
  primary annotation. `label_source` says which.

**`duplicate_label_review.csv`**: duplicated texts the primary annotator labelled more than
once. The build fails if any of them disagree without a decision in `duplicates_v1.csv`.

**`id_aliases.csv`**: ids the v1 classifier export used for the supplementary passages, mapped
to their original ids.

## Lockbox

Fold 6 of 10 is held out for final evaluation. It was picked by a rule that looks at labels
only: among folds containing every category, the one whose label mix is closest to the whole
dataset's.

- `presupp.data.dataset.load_dev` never returns lockbox passages, labelled or not.
- `open_lockbox(reason)` is the only way to load it, and every call is appended to
  `reports/lockbox_log.md`.

## Adjudication

All evidence in these files is written as S-numbers, 1-based, as the annotation tool showed
them. The build validates every row and stores the evidence 0-based.

**`duplicates_v1.csv`**: one row per (text cluster, category) for the 6 duplicated texts that
were labelled differently: `keep_passage_id`, `confidence`, `evidence_sentences`.

**`label_review_v1.csv`**: verdicts on passages re-examined after Claude's blind second
annotation. A `keep` decision is one row with no label fields,
and the original labels stand. A `replace` decision has one row per category, as above, and
those become the gold labels.

Reviewed passages are marked in `passages.parquet` (`label_review` is `kept` or `replaced`) and
are left out of later reviews.

**`evidence_realigned_v1.csv`**: primary evidence clicked on another sentence split, mapped onto
the stored sentences where that is unambiguous: `recorded` (checked against the label),
`evidence_sentences` and `splitter`. Development passages only.

**`evidence_remarked_v1.csv`**: saved from the re-marking page, which
`python -m presupp.data.remark` writes to `processed/remark_evidence_v1.html`. One row for each
position of each re-marked passage.

| column | contents |
|---|---|
| `confidence` | `high`, `medium` or `low` |
| `evidence_sentences` | empty when no sentence carries the position |
| `status` | `original`, `added` (a position the passage wasn't labelled with) or `removed` |
| `decided_on`, `note` | ISO date, free text |
| `source` | empty for my own re-marking, else the Claude review the rows came from |

A re-marked passage's rows replace its labels, so every original position must be listed. Its
gold labels have `label_source` `remarked`, or `remarked_claude` when they came from Claude. An
exclusion category can be added once every position is removed. Rows saved by the first
version of the page have no `confidence` or `status`, and keep the original confidence. A
passage can't be both re-marked and adjudicated.

The file covers 85 development passages: 36 re-marked by me and 49 taken from Claude's blind
annotations (48 from `claude_review/remark`, and G116 from `run1`).

## Reference tables

**`publishers.csv`** maps a URL domain to its publisher, country (ISO 3166 alpha-2) and genre
(`government`, `think_tank`, `news`, `magazine`, `science_media`). The genres are judgement
calls. The build fails on any domain the table doesn't cover.
