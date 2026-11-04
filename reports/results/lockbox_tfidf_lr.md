# Lockbox: tfidf_lr

Baseline. Word unigram+bigram TF-IDF with one class-balanced logistic regression per label, fixed settings, same folds, thresholds and IPC rule as the fixed transformer recipe.

Passages scored: 125; seeds: [0]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.249 ± 0.000 [0.195, 0.320] |
| macro-F1, 11 labels in >= 20 articles | 0.294 ± 0.000 [0.220, 0.356] |
| micro-F1 | 0.422 ± 0.000 [0.364, 0.485] |
| macro average precision (threshold-free) | 0.438 ± 0.000 |
| calibration error (lower is better) | 0.182 ± 0.000 |
| passages with no label predicted | 0.000 ± 0.000 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 11 | 2 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.833 |
| Anti-Reductionism | 23 | 4 | 0.200 | 0.250 | 0.222 [0.000, 0.500] | 0.357 |
| Determinism and Free Will | 47 | 6 | 0.089 | 0.667 | 0.157 [0.042, 0.286] | 0.137 |
| Empiricism | 255 | 38 | 0.731 | 0.500 | 0.594 [0.433, 0.747] | 0.677 |
| Rationalism (†) | 5 | 1 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.167 |
| Relativism and Objectivism | 69 | 8 | 0.107 | 1.000 | 0.193 [0.049, 0.360] | 0.710 |
| Consequentialism | 426 | 64 | 0.734 | 0.734 | 0.734 [0.624, 0.814] | 0.797 |
| Deontology | 166 | 21 | 0.381 | 0.762 | 0.508 [0.291, 0.682] | 0.625 |
| Virtue Ethics | 25 | 3 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.100 |
| Individualism | 83 | 12 | 0.077 | 0.083 | 0.080 [0.000, 0.286] | 0.138 |
| Communitarianism | 75 | 10 | 0.167 | 0.100 | 0.125 [0.000, 0.353] | 0.367 |
| State Neutrality and Perfectionism | 35 | 5 | 0.133 | 0.400 | 0.200 [0.000, 0.522] | 0.198 |
| Insufficient Philosophical Content | 145 | 18 | 0.467 | 0.389 | 0.424 [0.083, 0.649] | 0.594 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
