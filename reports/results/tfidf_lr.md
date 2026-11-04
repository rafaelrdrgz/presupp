# tfidf_lr

Baseline. Word unigram+bigram TF-IDF with one class-balanced logistic regression per label, fixed settings, same folds, thresholds and IPC rule as the fixed transformer recipe.

Passages scored: 1114; seeds: [0]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.318 ± 0.000 [0.286, 0.351] |
| macro-F1, 11 labels in >= 20 articles | 0.369 ± 0.000 [0.333, 0.400] |
| micro-F1 | 0.514 ± 0.000 [0.491, 0.539] |
| macro average precision (threshold-free) | 0.398 ± 0.000 |
| calibration error (lower is better) | 0.175 ± 0.000 |
| passages with no label predicted | 0.021 ± 0.000 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 11 | 23 | 0.333 | 0.043 | 0.077 [0.000, 0.300] | 0.263 |
| Anti-Reductionism | 23 | 39 | 0.500 | 0.231 | 0.316 [0.103, 0.532] | 0.451 |
| Determinism and Free Will | 47 | 65 | 0.274 | 0.446 | 0.339 [0.183, 0.482] | 0.267 |
| Empiricism | 255 | 340 | 0.579 | 0.709 | 0.638 [0.584, 0.687] | 0.726 |
| Rationalism (†) | 5 | 9 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.650 |
| Relativism and Objectivism | 69 | 86 | 0.168 | 0.640 | 0.266 [0.199, 0.336] | 0.303 |
| Consequentialism | 426 | 581 | 0.656 | 0.790 | 0.717 [0.682, 0.747] | 0.798 |
| Deontology | 166 | 197 | 0.430 | 0.563 | 0.488 [0.417, 0.553] | 0.538 |
| Virtue Ethics | 25 | 32 | 0.375 | 0.094 | 0.150 [0.000, 0.312] | 0.090 |
| Individualism | 83 | 119 | 0.233 | 0.454 | 0.308 [0.167, 0.455] | 0.249 |
| Communitarianism | 75 | 91 | 0.296 | 0.495 | 0.370 [0.285, 0.459] | 0.337 |
| State Neutrality and Perfectionism | 35 | 39 | 0.143 | 0.051 | 0.075 [0.000, 0.186] | 0.129 |
| Insufficient Philosophical Content | 145 | 159 | 0.435 | 0.358 | 0.393 [0.320, 0.468] | 0.371 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
