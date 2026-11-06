---
language:
- en
license: cc-by-nc-4.0
base_model: FacebookAI/roberta-base
pipeline_tag: text-classification
tags:
- philosophy
- multi-label-classification
- argument-mining
---

# Philosophical presupposition classifier

This model detects the philosophical assumptions that arguments in news and policy writing rely
on implicitly. An editorial using the sentence "evidence-based policy should rely on what works"
relies on an empiricist view of knowledge and a consequentialist view of what makes a policy
right, though it says neither. Given a passage, the model assigns any of 12 philosophical
positions from four domains and identifies the sentences each one rests on.

It is the sentence model (`v3_mil`) from [presupp](https://github.com/rafaelrdrgz/presupp), an
improved rebuild of my final-year dissertation (BSc Philosophy and Computer Science,
Northeastern University London, 2026). The repository has the code, the evaluation and the label
decisions. The [demo](https://rafaelrdrgz--presupposition-finder.us-east.modal.run) runs this
model, and I wrote about the project in a guest post for
[The Computational Philosopher](https://computationalphilosopher.substack.com/p/how-much-philosophy-can-a-language).

## Labels

| Domain | Positions |
|---|---|
| Metaphysical | physicalism and naturalism, anti-reductionism, determinism and free will |
| Epistemological | empiricism, rationalism, relativism and objectivism |
| Ethical | consequentialism, deontology, virtue ethics |
| Political-philosophical | individualism, communitarianism, state neutrality and perfectionism |

A passage can hold several positions or none. A passage with nothing philosophically
load-bearing gets a 13th label, insufficient philosophical content (IPC). A label applies when
rejecting the assumption would force the author to substantially reconstruct their argument.
The definitions are in `release.json` and
[taxonomy/categories_v1.json](https://github.com/rafaelrdrgz/presupp/blob/main/taxonomy/categories_v1.json).

## Using it

This isn't a standard `transformers` model, so it won't load with
`AutoModelForSequenceClassification` or run in the widget on this page. `encoder/` is a
fine-tuned RoBERTa-base, but the heads and the decision rules come from the presupp package
(Python 3.10 or later):

```bash
pip install "presupp[train] @ git+https://github.com/rafaelrdrgz/presupp"
```

```python
from huggingface_hub import snapshot_download
from presupp.inference import Classifier, describe

model = Classifier.load(snapshot_download("{{repo}}"))
with open("article.txt", encoding="utf-8") as f:
    reading, results = model.classify(f.read())
for result in results:
    print("\n".join(describe(result)))
```

`classify` takes a whole article. It splits it into passages like the training ones (150 to 350
words), sets aside menus, captions, reference lists and non-English text, and reads at most 20
passages. Each result has the positions found, grouped by domain, with their scores, thresholds
and evidence sentences. `classify_texts` takes passages that are already split. From the command
line:

```bash
python -m presupp.inference path/to/snapshot --file article.txt
```

## Files

| File | Contents |
|---|---|
| `encoder/` | the fine-tuned RoBERTa-base encoder and its tokenizer |
| `sentence_head.safetensors` | scores each sentence for the 12 positions |
| `head.safetensors` | the passage head, used for IPC |
| `release.json` | labels and definitions, thresholds, the training config and history, and the reliability figures the demo shows |
| `val_predictions.parquet`, `val_sentences.parquet` | scores on the validation fold by passage id and sentence number, with no text |

## Training

The encoder reads a passage once, up to 512 tokens, averages the token states within each
sentence and scores every sentence for every position. A passage's score for a position is its
best sentence's score, so every flag comes with the sentence that triggered it. IPC is decided
from the first token. The loss covers both the passage labels and the annotated evidence
sentences, with each positive weighted by the confidence I gave it.

Targets are hard, with no class weights. The embeddings and the bottom 6 of 12 layers are frozen.
It used AdamW at 2e-5 with weight decay 0.02, a linear schedule with 10% warm-up, batch size 8,
and up to 20 epochs with early stopping on validation macro average precision. The best epoch
was 12, and training took under four minutes on one V100.

Labels with at least 10 validation positives get their own threshold and the rest share one.
IPC is never predicted alongside a position. The strongest sentence for a flagged position is
always marked, and up to two more if they pass a separate marking threshold tuned on validation.
Sentences under four words are never marked.

## Data

The corpus is 2,373 paragraphs from 1,194 articles by 34 publishers: UK and US newspapers,
magazines, think tanks and GOV.UK. After removing duplicate texts, 1,239 passages have gold
labels, all from one annotator (me). They are split into ten folds grouped by article, and one
fold (125 labelled passages) is held back as a lockbox that no model trains on. These weights
were trained on 992 development passages, with fold 9 (122 passages) used for early stopping and
thresholds.

The article text isn't distributed, here or in the repository, because most of it is
copyrighted.

Some labels come from Claude, used as a second annotator. It annotated a random sample of
passages blind, working from the guidelines alone with no labels or publication names; I went
back over 26 passages where it disagreed with me and changed the labels on 10. Later, 85
passages had to be re-marked because their evidence had been clicked against a different
sentence split from the stored one. I re-marked 36 myself and adopted Claude's blind annotations
for the other 49.

## Results

Development figures are 9-fold cross-validation grouped by article, the mean of three seeds, with
95% intervals from resampling whole articles. Lockbox figures are three final runs scored on the
held-out fold; these weights are the seed 0 run.

| | Development | Lockbox |
|---|---|---|
| Macro-F1, all 13 labels | 0.473 [0.418, 0.503] | 0.437 [0.358, 0.554] |
| Macro-F1, 11 labels in 20 or more articles | 0.487 [0.451, 0.513] | 0.486 [0.408, 0.551] |
| Micro-F1 | 0.624 [0.604, 0.643] | 0.604 [0.548, 0.666] |
| First marked sentence is annotated evidence | 0.632 [0.608, 0.657] | 0.630 [0.560, 0.698] |
| The same for a random sentence | 0.408 | 0.428 |

For comparison, a TF-IDF and logistic regression baseline scores 0.318 macro-F1 in development.
The passage model this one replaced in the demo scores 0.483, and its first marked sentence is
evidence 0.606 of the time.

Before scoring this model, I set a rule that it would only replace the passage model if its
macro-F1 was no more than 0.010 below. It came in 0.0106 below. I used it anyway, because the
paired interval on that difference (−0.032 to +0.010) can't separate the two, while the evidence
gain (+0.026, interval +0.007 to +0.044) is clear. The lockbox has been opened three times over
the project, each one logged in
[reports/lockbox_log.md](https://github.com/rafaelrdrgz/presupp/blob/main/reports/lockbox_log.md).

| Label | Articles | F1, development | Lockbox positives | F1, lockbox |
|---|---|---|---|---|
| Physicalism and Naturalism (†) | 13 | 0.397 | 2 | 0.333 |
| Anti-Reductionism | 23 | 0.430 | 4 | 0.241 |
| Determinism and Free Will | 51 | 0.394 | 6 | 0.542 |
| Empiricism | 262 | 0.742 | 38 | 0.698 |
| Rationalism (†) | 5 | 0.385 | 1 | 0.000 |
| Relativism and Objectivism | 72 | 0.474 | 8 | 0.523 |
| Consequentialism | 429 | 0.797 | 64 | 0.770 |
| Deontology | 172 | 0.618 | 21 | 0.745 |
| Virtue Ethics | 27 | 0.344 | 3 | 0.172 |
| Individualism | 88 | 0.401 | 12 | 0.323 |
| Communitarianism | 76 | 0.491 | 10 | 0.495 |
| State Neutrality and Perfectionism | 40 | 0.132 | 5 | 0.207 |
| Insufficient Philosophical Content | 147 | 0.540 | 18 | 0.628 |

(†) in fewer than 20 articles, too few to measure reliably. Most lockbox labels have only a
handful of positives, so the development column is the better guide. Full tables are in
[reports/results/v3](https://github.com/rafaelrdrgz/presupp/tree/main/reports/results/v3).

## Limitations

- The gold labels are mine. Four philosophy specialists who annotated 15 passages each marked
  about half as many positions as I did, so the model has learned my reading of the guidelines.
- It assigns positions more often than it should, so each one is worth checking against the
  sentences it marks.
- Rationalism and physicalism appear in fewer than 20 articles, and state neutrality and
  perfectionism scores 0.13 in development. Their outputs mean very little.
- It has only seen English news, opinion and policy writing from the UK and US.
- Only the first 512 tokens of a longer passage are read.

## Licence

The weights are released for research and non-commercial use under CC BY-NC 4.0. The base
model, RoBERTa-base, is MIT-licensed.

If you would like to help improve this classifier, please get in touch through GitHub/LinkedIn.
