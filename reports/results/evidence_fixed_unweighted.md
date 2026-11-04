# Evidence sentences: fixed_unweighted

For each flagged position the app scores every sentence of the passage on its own and marks the strongest (all that would be flagged alone, up to 3, or else the single strongest; sentences under 4 words aren't scored). Checked here against the annotator's evidence sentences for the gold positions of the release model's validation fold, which it never trained on.

158 positions with evidence in 98 passages. Means [95% interval from resampling articles].

| measure | value |
|---|---|
| first marked sentence is an evidence sentence | 0.658 [0.589, 0.729] |
| the same for a sentence picked at random | 0.426 [0.394, 0.458] |
| difference | 0.233 [0.171, 0.298] |
| precision of all marked sentences | 0.598 |
| recall of the evidence sentences | 0.429 |
| sentences marked per position | 2.20 |

| position | cases | first marked is evidence | at random |
|---|---|---|---|
| anti_reductionism | 4 | 0.50 | 0.32 |
| communitarianism | 9 | 0.44 | 0.38 |
| consequentialism | 53 | 0.72 | 0.51 |
| deontology | 20 | 0.70 | 0.42 |
| determinism_free_will | 6 | 0.33 | 0.26 |
| empiricism | 35 | 0.63 | 0.41 |
| individualism | 12 | 0.83 | 0.36 |
| physicalism_naturalism | 6 | 0.50 | 0.35 |
| relativism_objectivism | 7 | 0.71 | 0.35 |
| state_neutrality_perfectionism | 4 | 0.75 | 0.33 |
| virtue_ethics | 2 | 0.50 | 0.50 |

Reading the primary annotator's 1-based lists 0-based instead: first marked sentence is evidence 0.403, at random 0.385, over 119 positions. That this is no better than chance agrees with the 1-based reading.
