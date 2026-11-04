# Paired comparisons

Development cross-validation (lockbox not used). Each row is A minus B, scored on the same passages: the observed difference [95% interval from resampling articles], and the share of resamples in which A is ahead. An interval that excludes 0 is a difference the data can detect.

| question | A − B | macro-F1 (all) | macro-F1 (>= 20 articles) | avg precision |
|---|---|---|---|---|
| Did article leakage inflate the v1 recipe's score? | v1_recipe_passage_cv − v1_recipe_grouped_cv | +0.049 [+0.014, +0.070], 100% | +0.020 [-0.001, +0.036], 97% | +0.082 [+0.040, +0.102], 100% |
| Do class weights help? | fixed_weighted − fixed_unweighted | -0.010 [-0.021, +0.003], 8% | -0.006 [-0.020, +0.005], 16% | -0.018 [-0.033, +0.002], 4% |
| Do the v2 fixes beat the v1 recipe, both without leakage? | fixed_unweighted − v1_recipe_grouped_cv | -0.003 [-0.026, +0.017], 38% | -0.008 [-0.023, +0.008], 15% | -0.010 [-0.027, +0.012], 20% |
| Does the transformer beat TF-IDF? | fixed_unweighted − tfidf_lr | +0.168 [+0.114, +0.208], 100% | +0.130 [+0.090, +0.166], 100% | +0.110 [+0.073, +0.137], 100% |
