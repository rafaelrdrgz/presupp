# Lockbox: v3_mil

v3, passages decided by their sentences: as v3_joint, but a position's passage score is its best sentence's score, so every flagged position has a sentence that reaches the mark. IPC is still decided from the first token.

Passages scored: 125; seeds: [0, 1, 2]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.437 ± 0.015 [0.358, 0.554] |
| macro-F1, 11 labels in >= 20 articles | 0.486 ± 0.018 [0.408, 0.551] |
| micro-F1 | 0.604 ± 0.013 [0.548, 0.666] |
| macro average precision (threshold-free) | 0.503 ± 0.020 |
| calibration error (lower is better) | 0.038 ± 0.007 |
| passages with no label predicted | 0.005 ± 0.008 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 13 | 2 | 0.250 | 0.500 | 0.333 [0.000, 1.000] | 0.313 |
| Anti-Reductionism | 23 | 4 | 0.233 | 0.250 | 0.241 [0.000, 0.545] | 0.324 |
| Determinism and Free Will | 51 | 6 | 0.416 | 0.833 | 0.542 [0.227, 0.753] | 0.607 |
| Empiricism | 262 | 38 | 0.680 | 0.719 | 0.698 [0.581, 0.794] | 0.820 |
| Rationalism (†) | 5 | 1 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.150 |
| Relativism and Objectivism | 72 | 8 | 0.363 | 0.958 | 0.523 [0.174, 0.698] | 0.675 |
| Consequentialism | 429 | 64 | 0.810 | 0.734 | 0.770 [0.677, 0.837] | 0.891 |
| Deontology | 172 | 21 | 0.695 | 0.810 | 0.745 [0.584, 0.864] | 0.829 |
| Virtue Ethics | 27 | 3 | 0.116 | 0.333 | 0.172 [0.000, 0.530] | 0.148 |
| Individualism | 88 | 12 | 0.260 | 0.444 | 0.323 [0.074, 0.474] | 0.302 |
| Communitarianism | 76 | 10 | 0.343 | 0.900 | 0.495 [0.205, 0.739] | 0.631 |
| State Neutrality and Perfectionism | 40 | 5 | 0.156 | 0.333 | 0.207 [0.000, 0.492] | 0.209 |
| Insufficient Philosophical Content | 147 | 18 | 0.576 | 0.704 | 0.628 [0.404, 0.775] | 0.637 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
