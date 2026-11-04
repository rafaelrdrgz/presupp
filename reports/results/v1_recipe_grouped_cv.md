# v1_recipe_grouped_cv

The dissertation's best recipe (lr2e5), unchanged, but evaluated with article-grouped folds: every passage of an article is in the same fold. The only difference from v1_recipe_passage_cv is the split.

Passages scored: 1114; seeds: [0, 1, 2]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.489 ± 0.010 [0.441, 0.518] |
| macro-F1, 11 labels in >= 20 articles | 0.508 ± 0.009 [0.471, 0.534] |
| micro-F1 | 0.625 ± 0.007 [0.605, 0.644] |
| macro average precision (threshold-free) | 0.517 ± 0.012 |
| calibration error (lower is better) | 0.060 ± 0.015 |
| passages with no label predicted | 0.018 ± 0.002 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 11 | 23 | 0.288 | 0.507 | 0.365 [0.185, 0.523] | 0.298 |
| Anti-Reductionism | 23 | 39 | 0.407 | 0.684 | 0.508 [0.290, 0.674] | 0.530 |
| Determinism and Free Will | 47 | 65 | 0.339 | 0.523 | 0.409 [0.243, 0.558] | 0.392 |
| Empiricism | 255 | 340 | 0.749 | 0.758 | 0.753 [0.710, 0.790] | 0.819 |
| Rationalism (†) | 5 | 9 | 0.335 | 0.556 | 0.415 [0.000, 0.699] | 0.653 |
| Relativism and Objectivism | 69 | 86 | 0.343 | 0.570 | 0.427 [0.326, 0.510] | 0.463 |
| Consequentialism | 426 | 581 | 0.719 | 0.853 | 0.780 [0.751, 0.805] | 0.869 |
| Deontology | 166 | 197 | 0.531 | 0.702 | 0.605 [0.549, 0.659] | 0.641 |
| Virtue Ethics | 25 | 32 | 0.263 | 0.417 | 0.319 [0.157, 0.446] | 0.286 |
| Individualism | 83 | 119 | 0.399 | 0.485 | 0.437 [0.307, 0.552] | 0.430 |
| Communitarianism | 75 | 91 | 0.423 | 0.641 | 0.509 [0.425, 0.591] | 0.519 |
| State Neutrality and Perfectionism | 35 | 39 | 0.206 | 0.368 | 0.263 [0.115, 0.382] | 0.201 |
| Insufficient Philosophical Content | 145 | 159 | 0.558 | 0.591 | 0.573 [0.506, 0.635] | 0.623 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
