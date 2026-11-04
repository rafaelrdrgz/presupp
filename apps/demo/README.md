# Philosophical Presupposition Finder

A Gradio app around the sentence model (`v3_mil`). Paste an article, a speech or a policy paper,
either a selection or a whole page copied with Ctrl+A, and it reads the text back passage by
passage. A passage with a position gets a coloured rule for each domain and a note in the
margin naming the positions. The sentences each position rests on are underlined and numbered;
hovering over a number or a sentence highlights the other. **Scores** under a passage lists every
position's score against its threshold, its F1 in testing, and definitions of the ones flagged.
Skipped lines (headings, menus, captions, reference lists, tables, other languages) are folded
away with the reason.

The examples in `examples.py` are GOV.UK excerpts under the Open Government Licence, none of them
in the training corpus.

## How the text is read

`presupp.reading` turns a paste into passages like the training ones. The model learned from
passages of about 150 to 250 words, so short paragraphs are joined and long ones cut at sentence
ends. Lines that a PDF or an email hard-wrapped are joined back up, and anything that isn't
English prose is set aside. At most 20 passages (about 4,000 words) are read per request. The
rules were tuned on real pastes: news and opinion pieces, GOV.UK pages and PDFs, Wikipedia, a
recipe, and German and Spanish pages.

Decisions go through the same code the evaluation used (`presupp.decisions`). A domain is shown
when any of its categories is flagged; how reliable that is, per domain, is in
[reports/results/v3/domains_v3_mil.md](../../reports/results/v3/domains_v3_mil.md).

A passage's score for a position is its strongest sentence's, so the first sentence underlined
is the one that flagged it (sentences under four words are never marked). Up to two more are
underlined if they pass a separate marking threshold tuned on the validation fold. In
cross-validation the first underlined sentence was an annotated evidence sentence 63% of the
time, against 61% for scoring each sentence on its own with the passage model and 41% for a
random sentence.

## The model

The app loads a release folder written by `python -m presupp.release`, trained on the cluster
(`scripts/cluster/release.sbatch`) and copied to `models/presupp-roberta-v3/`.
The folder holds weights, thresholds, label definitions, and validation scores by passage id; no
passage text. It isn't in git.

After the results change, refresh the reliability figures the app shows without retraining:

```bash
python -m presupp.release configs/v3_mil.yaml --reliability-only
```

The older passage model (`models/presupp-roberta-v1`, from `configs/fixed_unweighted.yaml`) still
loads if `PRESUPP_MODEL_DIR` points at it.

## Running it locally

The demo needs its own environment: gradio 5 requires pandas < 3, and gradio 6 needs a
`huggingface_hub` that transformers 4 doesn't accept. From the repository root:

```bash
python3.12 -m venv .venv-demo
source .venv-demo/bin/activate        # .venv-demo\Scripts\activate on Windows
pip install -r apps/demo/requirements.txt "gradio==5.50.0"
pip install --no-deps -e .
python apps/demo/app.py
```

Then open http://127.0.0.1:7860. `PRESUPP_MODEL_DIR` selects another release folder. Without the
web page:

```bash
python -m presupp.inference models/presupp-roberta-v3 --file article.txt
```

## Hosting on Modal

`modal_app.py` serves the app on [Modal](https://modal.com), with the model folder uploaded
alongside the code to your own workspace. It keeps one container running, capped at half a CPU
core and 1.5 GiB of memory (the app peaks at about 830 MB).

```bash
pip install modal==1.6.3
modal setup                           # logs in through the browser
modal deploy apps/demo/modal_app.py   # prints the public URL
```

Run the deploy again to update it, and set a spend limit on Modal's billing page.
