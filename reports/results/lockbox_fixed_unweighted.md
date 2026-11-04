# Lockbox: fixed_unweighted

The v1 model with the v2 fixes: yes/no targets, no class weights (rare labels are helped by per-label thresholds instead), weight decay off for biases and LayerNorm, linear warmup and decay, the best epoch chosen by threshold-free macro average precision, per-label thresholds only for labels with at least 10 validation positives, and IPC never predicted together with a category. Article-grouped folds.

Passages scored: 125; seeds: [0, 1, 2]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.466 ± 0.015 [0.365, 0.580] |
| macro-F1, 11 labels in >= 20 articles | 0.510 ± 0.011 [0.413, 0.575] |
| micro-F1 | 0.643 ± 0.012 [0.581, 0.707] |
| macro average precision (threshold-free) | 0.506 ± 0.017 |
| calibration error (lower is better) | 0.028 ± 0.008 |
| passages with no label predicted | 0.003 ± 0.004 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 11 | 2 | 0.417 | 0.500 | 0.444 [0.000, 0.933] | 0.406 |
| Anti-Reductionism | 23 | 4 | 0.333 | 0.250 | 0.273 [0.000, 0.536] | 0.246 |
| Determinism and Free Will | 47 | 6 | 0.722 | 0.444 | 0.484 [0.127, 0.731] | 0.584 |
| Empiricism | 255 | 38 | 0.662 | 0.754 | 0.705 [0.599, 0.808] | 0.750 |
| Rationalism (†) | 5 | 1 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.145 |
| Relativism and Objectivism | 69 | 8 | 0.439 | 0.875 | 0.572 [0.208, 0.715] | 0.683 |
| Consequentialism | 426 | 64 | 0.805 | 0.812 | 0.806 [0.715, 0.871] | 0.924 |
| Deontology | 166 | 21 | 0.776 | 0.714 | 0.743 [0.544, 0.873] | 0.828 |
| Virtue Ethics | 25 | 3 | 0.250 | 0.222 | 0.229 [0.000, 0.508] | 0.236 |
| Individualism | 83 | 12 | 0.264 | 0.361 | 0.290 [0.040, 0.467] | 0.228 |
| Communitarianism | 75 | 10 | 0.414 | 0.833 | 0.549 [0.203, 0.805] | 0.764 |
| State Neutrality and Perfectionism | 35 | 5 | 0.329 | 0.400 | 0.343 [0.000, 0.748] | 0.221 |
| Insufficient Philosophical Content | 145 | 18 | 0.536 | 0.722 | 0.615 [0.394, 0.750] | 0.563 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
