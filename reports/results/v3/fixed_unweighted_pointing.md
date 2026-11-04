# fixed_unweighted_pointing

The baseline for v3: fixed_unweighted retrained on the final data, which also scores each stored sentence of its validation and test passages on its own, as the v1 app finds evidence.

Passages scored: 1114; seeds: [0, 1, 2]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.483 ± 0.016 [0.432, 0.509] |
| macro-F1, 11 labels in >= 20 articles | 0.491 ± 0.004 [0.457, 0.516] |
| micro-F1 | 0.619 ± 0.012 [0.597, 0.641] |
| macro average precision (threshold-free) | 0.505 ± 0.008 |
| calibration error (lower is better) | 0.020 ± 0.001 |
| passages with no label predicted | 0.017 ± 0.002 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 13 | 25 | 0.301 | 0.467 | 0.363 [0.150, 0.543] | 0.296 |
| Anti-Reductionism | 23 | 39 | 0.366 | 0.615 | 0.458 [0.226, 0.640] | 0.548 |
| Determinism and Free Will | 51 | 69 | 0.319 | 0.517 | 0.392 [0.249, 0.524] | 0.361 |
| Empiricism | 262 | 348 | 0.697 | 0.768 | 0.731 [0.686, 0.773] | 0.797 |
| Rationalism (†) | 5 | 9 | 0.509 | 0.593 | 0.517 [0.000, 0.700] | 0.632 |
| Relativism and Objectivism | 72 | 89 | 0.348 | 0.577 | 0.434 [0.339, 0.514] | 0.461 |
| Consequentialism | 429 | 586 | 0.744 | 0.848 | 0.792 [0.765, 0.816] | 0.863 |
| Deontology | 172 | 205 | 0.564 | 0.673 | 0.614 [0.559, 0.670] | 0.659 |
| Virtue Ethics | 27 | 34 | 0.311 | 0.324 | 0.308 [0.136, 0.436] | 0.254 |
| Individualism | 88 | 123 | 0.340 | 0.550 | 0.420 [0.302, 0.525] | 0.401 |
| Communitarianism | 76 | 91 | 0.402 | 0.623 | 0.487 [0.403, 0.570] | 0.539 |
| State Neutrality and Perfectionism | 40 | 44 | 0.223 | 0.242 | 0.226 [0.111, 0.329] | 0.139 |
| Insufficient Philosophical Content | 147 | 161 | 0.640 | 0.468 | 0.540 [0.462, 0.612] | 0.618 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
