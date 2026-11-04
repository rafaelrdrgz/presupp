# Evidence sentences: paired comparisons

Development cross-validation (lockbox not used). Each row is A minus B on the same positions: the observed difference [95% interval from resampling articles], and the share of resamples in which A is ahead.

| question | A − B | first_hit | precision | recall | f1 | sentence_ap |
|---|---|---|---|---|---|---|
| Does reading the passage once, trained on the sentences, find the evidence better? | v3_joint − fixed_unweighted_pointing | +0.007 [-0.013, +0.028], 72% | +0.037 [+0.026, +0.049], 100% | -0.014 [-0.025, -0.002], 1% | +0.007 [-0.004, +0.019], 91% | +0.024 [+0.015, +0.033], 100% |
| Does deciding by the best sentence, trained on the sentences, find the evidence better? | v3_mil − fixed_unweighted_pointing | +0.026 [+0.007, +0.044], 100% | +0.036 [+0.024, +0.048], 100% | +0.007 [-0.004, +0.017], 91% | +0.019 [+0.009, +0.030], 100% | +0.034 [+0.025, +0.043], 100% |
| Is deciding by the best sentence better than the first token? | v3_mil − v3_joint | +0.019 [+0.006, +0.033], 100% | -0.001 [-0.009, +0.007], 40% | +0.021 [+0.014, +0.027], 100% | +0.012 [+0.006, +0.018], 100% | +0.010 [+0.004, +0.016], 100% |
