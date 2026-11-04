# prior

Baseline. Every passage gets each label's training-set frequency as its probability, so after thresholds each label is predicted for every passage or for none.

Passages scored: 1114; seeds: [0]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.158 ± 0.000 [0.151, 0.165] |
| macro-F1, 11 labels in >= 20 articles | 0.186 ± 0.000 [0.179, 0.194] |
| micro-F1 | 0.309 ± 0.000 [0.296, 0.322] |
| macro average precision (threshold-free) | 0.119 ± 0.000 |
| calibration error (lower is better) | 0.002 ± 0.000 |
| passages with no label predicted | 0.000 ± 0.000 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 11 | 23 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.017 |
| Anti-Reductionism | 23 | 39 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.031 |
| Determinism and Free Will | 47 | 65 | 0.058 | 1.000 | 0.110 [0.069, 0.151] | 0.055 |
| Empiricism | 255 | 340 | 0.305 | 1.000 | 0.468 [0.427, 0.511] | 0.301 |
| Rationalism (†) | 5 | 9 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.007 |
| Relativism and Objectivism | 69 | 86 | 0.077 | 1.000 | 0.143 [0.109, 0.180] | 0.075 |
| Consequentialism | 426 | 581 | 0.522 | 1.000 | 0.686 [0.654, 0.716] | 0.513 |
| Deontology | 166 | 197 | 0.177 | 1.000 | 0.301 [0.262, 0.338] | 0.171 |
| Virtue Ethics | 25 | 32 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.026 |
| Individualism | 83 | 119 | 0.107 | 1.000 | 0.193 [0.139, 0.263] | 0.097 |
| Communitarianism | 75 | 91 | 0.082 | 1.000 | 0.151 [0.119, 0.185] | 0.080 |
| State Neutrality and Perfectionism | 35 | 39 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.033 |
| Insufficient Philosophical Content | 145 | 159 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.140 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
