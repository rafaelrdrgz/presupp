# Paired comparisons

Development cross-validation (lockbox not used). Each row is A minus B, scored on the same passages: the observed difference [95% interval from resampling articles], and the share of resamples in which A is ahead. An interval that excludes 0 is a difference the data can detect.

| question | A − B | macro-F1 (all) | macro-F1 (>= 20 articles) | avg precision |
|---|---|---|---|---|
| Does training on the sentences and confidence change the passage scores? | v3_joint − fixed_unweighted_pointing | -0.005 [-0.023, +0.017], 38% | +0.012 [-0.004, +0.026], 94% | -0.013 [-0.026, +0.004], 9% |
| Does deciding by the best sentence change the passage scores? | v3_mil − fixed_unweighted_pointing | -0.011 [-0.032, +0.010], 19% | -0.004 [-0.020, +0.012], 32% | -0.015 [-0.034, +0.010], 15% |
| Is deciding by the best sentence better than the first token? | v3_mil − v3_joint | -0.006 [-0.024, +0.010], 21% | -0.016 [-0.033, +0.000], 3% | -0.001 [-0.022, +0.018], 44% |
