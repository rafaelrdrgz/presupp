# Paired comparisons

The lockbox, after training on the development folds. Each row is A minus B, scored on the same passages: the observed difference [95% interval from resampling articles], and the share of resamples in which A is ahead. An interval that excludes 0 is a difference the data can detect.

| question | A − B | macro-F1 (all) | macro-F1 (>= 20 articles) | avg precision |
|---|---|---|---|---|
| Does deciding by the best sentence change the passage scores? | v3_mil − fixed_unweighted_pointing | -0.003 [-0.027, +0.028], 51% | +0.001 [-0.028, +0.033], 56% | -0.009 [-0.062, +0.041], 36% |
