# fixed_unweighted

The v1 model with the v2 fixes: yes/no targets, no class weights (rare labels are helped by per-label thresholds instead), weight decay off for biases and LayerNorm, linear warmup and decay, the best epoch chosen by threshold-free macro average precision, per-label thresholds only for labels with at least 10 validation positives, and IPC never predicted together with a category. Article-grouped folds.

Passages scored: 1114; seeds: [0, 1, 2]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.486 ± 0.010 [0.437, 0.512] |
| macro-F1, 11 labels in >= 20 articles | 0.499 ± 0.004 [0.464, 0.526] |
| micro-F1 | 0.621 ± 0.005 [0.598, 0.643] |
| macro average precision (threshold-free) | 0.508 ± 0.002 |
| calibration error (lower is better) | 0.022 ± 0.003 |
| passages with no label predicted | 0.019 ± 0.000 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 11 | 23 | 0.349 | 0.551 | 0.425 [0.197, 0.593] | 0.302 |
| Anti-Reductionism | 23 | 39 | 0.374 | 0.607 | 0.462 [0.253, 0.640] | 0.549 |
| Determinism and Free Will | 47 | 65 | 0.312 | 0.523 | 0.389 [0.247, 0.517] | 0.377 |
| Empiricism | 255 | 340 | 0.747 | 0.754 | 0.750 [0.708, 0.788] | 0.822 |
| Rationalism (†) | 5 | 9 | 0.308 | 0.593 | 0.403 [0.000, 0.620] | 0.615 |
| Relativism and Objectivism | 69 | 86 | 0.316 | 0.581 | 0.409 [0.310, 0.495] | 0.429 |
| Consequentialism | 426 | 581 | 0.740 | 0.835 | 0.785 [0.755, 0.810] | 0.866 |
| Deontology | 166 | 197 | 0.552 | 0.689 | 0.613 [0.553, 0.671] | 0.648 |
| Virtue Ethics | 25 | 32 | 0.307 | 0.438 | 0.359 [0.204, 0.482] | 0.280 |
| Individualism | 83 | 119 | 0.321 | 0.501 | 0.391 [0.271, 0.490] | 0.384 |
| Communitarianism | 75 | 91 | 0.466 | 0.542 | 0.500 [0.418, 0.580] | 0.525 |
| State Neutrality and Perfectionism | 35 | 39 | 0.239 | 0.308 | 0.267 [0.123, 0.383] | 0.173 |
| Insufficient Philosophical Content | 145 | 159 | 0.630 | 0.518 | 0.567 [0.489, 0.634] | 0.629 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
