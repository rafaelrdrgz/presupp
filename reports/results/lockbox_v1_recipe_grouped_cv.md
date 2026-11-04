# Lockbox: v1_recipe_grouped_cv

The dissertation's best recipe (lr2e5), unchanged, but evaluated with article-grouped folds: every passage of an article is in the same fold. The only difference from v1_recipe_passage_cv is the split.

Passages scored: 125; seeds: [0, 1, 2]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.452 ± 0.012 [0.361, 0.563] |
| macro-F1, 11 labels in >= 20 articles | 0.504 ± 0.014 [0.411, 0.572] |
| micro-F1 | 0.618 ± 0.015 [0.554, 0.681] |
| macro average precision (threshold-free) | 0.519 ± 0.013 |
| calibration error (lower is better) | 0.046 ± 0.016 |
| passages with no label predicted | 0.035 ± 0.015 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 11 | 2 | 0.250 | 0.500 | 0.333 [0.000, 0.756] | 0.497 |
| Anti-Reductionism | 23 | 4 | 0.294 | 0.333 | 0.274 [0.000, 0.598] | 0.354 |
| Determinism and Free Will | 47 | 6 | 0.421 | 0.389 | 0.398 [0.000, 0.680] | 0.475 |
| Empiricism | 255 | 38 | 0.690 | 0.684 | 0.682 [0.553, 0.791] | 0.775 |
| Rationalism (†) | 5 | 1 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.194 |
| Relativism and Objectivism | 69 | 8 | 0.515 | 0.833 | 0.614 [0.192, 0.751] | 0.778 |
| Consequentialism | 426 | 64 | 0.823 | 0.771 | 0.795 [0.681, 0.872] | 0.899 |
| Deontology | 166 | 21 | 0.725 | 0.698 | 0.709 [0.511, 0.840] | 0.809 |
| Virtue Ethics | 25 | 3 | 0.175 | 0.333 | 0.227 [0.000, 0.611] | 0.291 |
| Individualism | 83 | 12 | 0.262 | 0.583 | 0.359 [0.117, 0.548] | 0.235 |
| Communitarianism | 75 | 10 | 0.392 | 0.900 | 0.542 [0.210, 0.801] | 0.616 |
| State Neutrality and Perfectionism | 35 | 5 | 0.304 | 0.467 | 0.328 [0.000, 0.667] | 0.196 |
| Insufficient Philosophical Content | 145 | 18 | 0.551 | 0.722 | 0.613 [0.405, 0.745] | 0.626 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
