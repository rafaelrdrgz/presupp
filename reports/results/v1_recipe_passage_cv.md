# v1_recipe_passage_cv

The dissertation's best recipe (lr2e5) under v1's protocol: passages are dealt into folds one at a time, so passages from the same article can be on both sides of a split. Compare with v1_recipe_grouped_cv to see how much of v1's score came from article leakage.

Passages scored: 1114; seeds: [0, 1, 2]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.538 ± 0.008 [0.473, 0.562] |
| macro-F1, 11 labels in >= 20 articles | 0.527 ± 0.009 [0.487, 0.553] |
| micro-F1 | 0.652 ± 0.001 [0.635, 0.669] |
| macro average precision (threshold-free) | 0.600 ± 0.006 |
| calibration error (lower is better) | 0.032 ± 0.005 |
| passages with no label predicted | 0.021 ± 0.001 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 11 | 23 | 0.450 | 0.768 | 0.567 [0.228, 0.758] | 0.614 |
| Anti-Reductionism | 23 | 39 | 0.466 | 0.607 | 0.526 [0.287, 0.685] | 0.660 |
| Determinism and Free Will | 47 | 65 | 0.387 | 0.482 | 0.429 [0.273, 0.550] | 0.430 |
| Empiricism | 255 | 340 | 0.737 | 0.747 | 0.742 [0.699, 0.780] | 0.835 |
| Rationalism (†) | 5 | 9 | 0.583 | 0.704 | 0.635 [0.000, 0.869] | 0.748 |
| Relativism and Objectivism | 69 | 86 | 0.422 | 0.504 | 0.459 [0.354, 0.551] | 0.481 |
| Consequentialism | 426 | 581 | 0.749 | 0.854 | 0.798 [0.770, 0.821] | 0.875 |
| Deontology | 166 | 197 | 0.565 | 0.667 | 0.610 [0.553, 0.662] | 0.666 |
| Virtue Ethics | 25 | 32 | 0.338 | 0.385 | 0.360 [0.185, 0.494] | 0.454 |
| Individualism | 83 | 119 | 0.453 | 0.515 | 0.478 [0.318, 0.610] | 0.513 |
| Communitarianism | 75 | 91 | 0.537 | 0.560 | 0.548 [0.455, 0.627] | 0.584 |
| State Neutrality and Perfectionism | 35 | 39 | 0.229 | 0.291 | 0.251 [0.090, 0.381] | 0.272 |
| Insufficient Philosophical Content | 145 | 159 | 0.575 | 0.623 | 0.597 [0.522, 0.656] | 0.664 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
