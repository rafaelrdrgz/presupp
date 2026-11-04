# Lockbox: fixed_unweighted_pointing

The baseline for v3: fixed_unweighted retrained on the final data, which also scores each stored sentence of its validation and test passages on its own, as the v1 app finds evidence.

Passages scored: 125; seeds: [0, 1, 2]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.439 ± 0.021 [0.358, 0.552] |
| macro-F1, 11 labels in >= 20 articles | 0.485 ± 0.022 [0.404, 0.551] |
| micro-F1 | 0.587 ± 0.018 [0.524, 0.653] |
| macro average precision (threshold-free) | 0.512 ± 0.015 |
| calibration error (lower is better) | 0.022 ± 0.009 |
| passages with no label predicted | 0.003 ± 0.004 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 13 | 2 | 0.306 | 0.500 | 0.378 [0.000, 1.000] | 0.403 |
| Anti-Reductionism | 23 | 4 | 0.233 | 0.250 | 0.236 [0.000, 0.530] | 0.224 |
| Determinism and Free Will | 51 | 6 | 0.460 | 0.722 | 0.554 [0.227, 0.784] | 0.604 |
| Empiricism | 262 | 38 | 0.682 | 0.675 | 0.678 [0.568, 0.784] | 0.761 |
| Rationalism (†) | 5 | 1 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.187 |
| Relativism and Objectivism | 72 | 8 | 0.326 | 0.917 | 0.480 [0.144, 0.667] | 0.735 |
| Consequentialism | 429 | 64 | 0.809 | 0.724 | 0.764 [0.662, 0.842] | 0.919 |
| Deontology | 172 | 21 | 0.741 | 0.762 | 0.751 [0.586, 0.875] | 0.836 |
| Virtue Ethics | 27 | 3 | 0.183 | 0.333 | 0.230 [0.000, 0.611] | 0.183 |
| Individualism | 88 | 12 | 0.224 | 0.611 | 0.315 [0.071, 0.510] | 0.280 |
| Communitarianism | 76 | 10 | 0.319 | 1.000 | 0.484 [0.196, 0.743] | 0.711 |
| State Neutrality and Perfectionism | 40 | 5 | 0.161 | 0.400 | 0.225 [0.000, 0.543] | 0.223 |
| Insufficient Philosophical Content | 147 | 18 | 0.507 | 0.796 | 0.619 [0.389, 0.764] | 0.593 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
