# v3_joint

v3, sentences trained alongside the passage: fixed_unweighted's settings, with the encoder's token states averaged per stored sentence and a sentence head scoring each sentence for every position, trained on the annotated evidence sentences. Labels count by their confidence (high 1, medium 0.75, low 0.5). Passage decisions come from the first token, as in fixed_unweighted.

Passages scored: 1114; seeds: [0, 1, 2]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.478 ± 0.006 [0.431, 0.505] |
| macro-F1, 11 labels in >= 20 articles | 0.504 ± 0.008 [0.467, 0.531] |
| micro-F1 | 0.634 ± 0.008 [0.612, 0.655] |
| macro average precision (threshold-free) | 0.492 ± 0.013 |
| calibration error (lower is better) | 0.027 ± 0.002 |
| passages with no label predicted | 0.010 ± 0.003 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 13 | 25 | 0.309 | 0.440 | 0.363 [0.155, 0.547] | 0.263 |
| Anti-Reductionism | 23 | 39 | 0.419 | 0.581 | 0.487 [0.242, 0.671] | 0.503 |
| Determinism and Free Will | 51 | 69 | 0.321 | 0.488 | 0.384 [0.232, 0.515] | 0.347 |
| Empiricism | 262 | 348 | 0.728 | 0.784 | 0.755 [0.711, 0.794] | 0.811 |
| Rationalism (†) | 5 | 9 | 0.317 | 0.333 | 0.318 [0.000, 0.492] | 0.574 |
| Relativism and Objectivism | 72 | 89 | 0.372 | 0.603 | 0.459 [0.370, 0.537] | 0.485 |
| Consequentialism | 429 | 586 | 0.747 | 0.858 | 0.799 [0.772, 0.823] | 0.868 |
| Deontology | 172 | 205 | 0.556 | 0.720 | 0.627 [0.573, 0.679] | 0.669 |
| Virtue Ethics | 27 | 34 | 0.308 | 0.382 | 0.340 [0.166, 0.471] | 0.227 |
| Individualism | 88 | 123 | 0.394 | 0.504 | 0.442 [0.302, 0.578] | 0.410 |
| Communitarianism | 76 | 91 | 0.422 | 0.560 | 0.479 [0.383, 0.563] | 0.500 |
| State Neutrality and Perfectionism | 40 | 44 | 0.178 | 0.220 | 0.195 [0.097, 0.277] | 0.129 |
| Insufficient Philosophical Content | 147 | 161 | 0.626 | 0.528 | 0.571 [0.495, 0.638] | 0.608 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
