# v3_mil

v3, passages decided by their sentences: as v3_joint, but a position's passage score is its best sentence's score, so every flagged position has a sentence that reaches the mark. IPC is still decided from the first token.

Passages scored: 1114; seeds: [0, 1, 2]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.473 ± 0.015 [0.418, 0.503] |
| macro-F1, 11 labels in >= 20 articles | 0.487 ± 0.011 [0.451, 0.513] |
| micro-F1 | 0.624 ± 0.007 [0.604, 0.643] |
| macro average precision (threshold-free) | 0.490 ± 0.006 |
| calibration error (lower is better) | 0.032 ± 0.002 |
| passages with no label predicted | 0.031 ± 0.003 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 13 | 25 | 0.339 | 0.480 | 0.397 [0.147, 0.633] | 0.267 |
| Anti-Reductionism | 23 | 39 | 0.365 | 0.530 | 0.430 [0.149, 0.645] | 0.478 |
| Determinism and Free Will | 51 | 69 | 0.342 | 0.464 | 0.394 [0.225, 0.544] | 0.336 |
| Empiricism | 262 | 348 | 0.729 | 0.759 | 0.742 [0.698, 0.779] | 0.783 |
| Rationalism (†) | 5 | 9 | 0.310 | 0.519 | 0.385 [0.000, 0.671] | 0.568 |
| Relativism and Objectivism | 72 | 89 | 0.413 | 0.558 | 0.474 [0.378, 0.549] | 0.502 |
| Consequentialism | 429 | 586 | 0.743 | 0.859 | 0.797 [0.770, 0.820] | 0.852 |
| Deontology | 172 | 205 | 0.573 | 0.672 | 0.618 [0.566, 0.670] | 0.649 |
| Virtue Ethics | 27 | 34 | 0.328 | 0.363 | 0.344 [0.178, 0.474] | 0.303 |
| Individualism | 88 | 123 | 0.329 | 0.515 | 0.401 [0.278, 0.518] | 0.372 |
| Communitarianism | 76 | 91 | 0.426 | 0.579 | 0.491 [0.396, 0.575] | 0.557 |
| State Neutrality and Perfectionism | 40 | 44 | 0.120 | 0.159 | 0.132 [0.064, 0.198] | 0.125 |
| Insufficient Philosophical Content | 147 | 161 | 0.658 | 0.458 | 0.540 [0.458, 0.617] | 0.583 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
