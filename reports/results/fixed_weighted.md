# fixed_weighted

fixed_unweighted with v1's class weights added back (a positive counts negatives/positives times, capped at 15). The only difference from fixed_unweighted: does weighting help rare labels get learned, or only make the probabilities overconfident?

Passages scored: 1114; seeds: [0, 1, 2]. Numbers are mean ± sd across seeds [95% interval from resampling articles].

| metric | value |
|---|---|
| macro-F1, all 13 labels | 0.476 ± 0.010 [0.431, 0.503] |
| macro-F1, 11 labels in >= 20 articles | 0.493 ± 0.007 [0.455, 0.521] |
| micro-F1 | 0.610 ± 0.002 [0.587, 0.630] |
| macro average precision (threshold-free) | 0.490 ± 0.008 |
| calibration error (lower is better) | 0.083 ± 0.006 |
| passages with no label predicted | 0.014 ± 0.006 |

| label | articles | positives | precision | recall | F1 [95% interval] | average precision |
|---|---|---|---|---|---|---|
| Physicalism and Naturalism (†) | 11 | 23 | 0.286 | 0.507 | 0.366 [0.182, 0.519] | 0.268 |
| Anti-Reductionism | 23 | 39 | 0.398 | 0.615 | 0.482 [0.266, 0.652] | 0.503 |
| Determinism and Free Will | 47 | 65 | 0.307 | 0.528 | 0.385 [0.250, 0.510] | 0.348 |
| Empiricism | 255 | 340 | 0.716 | 0.755 | 0.735 [0.694, 0.774] | 0.815 |
| Rationalism (†) | 5 | 9 | 0.319 | 0.556 | 0.403 [0.000, 0.613] | 0.577 |
| Relativism and Objectivism | 69 | 86 | 0.298 | 0.640 | 0.405 [0.312, 0.488] | 0.395 |
| Consequentialism | 426 | 581 | 0.735 | 0.820 | 0.775 [0.746, 0.801] | 0.854 |
| Deontology | 166 | 197 | 0.545 | 0.675 | 0.603 [0.545, 0.660] | 0.607 |
| Virtue Ethics | 25 | 32 | 0.300 | 0.385 | 0.337 [0.176, 0.453] | 0.279 |
| Individualism | 83 | 119 | 0.351 | 0.496 | 0.410 [0.285, 0.528] | 0.413 |
| Communitarianism | 75 | 91 | 0.456 | 0.564 | 0.504 [0.415, 0.590] | 0.504 |
| State Neutrality and Perfectionism | 35 | 39 | 0.204 | 0.316 | 0.248 [0.084, 0.384] | 0.187 |
| Insufficient Philosophical Content | 145 | 159 | 0.576 | 0.501 | 0.536 [0.460, 0.603] | 0.615 |

(†) in fewer than 20 articles: too little evidence to measure reliably.
