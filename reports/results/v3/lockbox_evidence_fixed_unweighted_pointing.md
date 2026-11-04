# Evidence sentences, lockbox: fixed_unweighted_pointing

Each gold position of a lockbox passage with evidence that fits the stored sentences has sentences marked from its sentence scores: all at or above its marking threshold (at most 3), or else the strongest; sentences under 4 words aren't marked. Thresholds are tuned on each run's validation fold. The lockbox's evidence wasn't realigned or re-marked as the development passages' was, so compare experiments with each other here, not with the cross-validation figures.

156 positions in 97 passages, seeds [0, 1, 2]. Means over seeds [95% interval from resampling articles].

| measure | value |
|---|---|
| first marked sentence is an evidence sentence | 0.615 [0.542, 0.701] |
| the same for a sentence picked at random | 0.428 [0.391, 0.475] |
| difference | 0.188 [0.127, 0.255] |
| precision of the marked sentences | 0.553 [0.498, 0.608] |
| recall of the evidence sentences | 0.433 [0.395, 0.478] |
| F1 of the marked sentences | 0.485 [0.444, 0.529] |
| sentence AP (threshold-free) | 0.674 [0.627, 0.727] |
| sentences marked per position | 2.54 |

Marking thresholds at the edge of the grid (0.05 or 0.95): 53%.

| position | cases | first marked is evidence | at random | sentence AP |
|---|---|---|---|---|
| anti_reductionism | 4 | 0.25 | 0.26 | 0.43 |
| communitarianism | 8 | 0.67 | 0.44 | 0.71 |
| consequentialism | 58 | 0.63 | 0.47 | 0.69 |
| deontology | 20 | 0.52 | 0.33 | 0.61 |
| determinism_free_will | 6 | 0.56 | 0.41 | 0.59 |
| empiricism | 32 | 0.70 | 0.48 | 0.73 |
| individualism | 11 | 0.52 | 0.29 | 0.60 |
| physicalism_naturalism | 2 | 0.50 | 0.26 | 0.50 |
| rationalism | 1 | 0.00 | 0.38 | 0.40 |
| relativism_objectivism | 7 | 1.00 | 0.60 | 0.82 |
| state_neutrality_perfectionism | 4 | 0.50 | 0.30 | 0.65 |
| virtue_ethics | 3 | 0.56 | 0.39 | 0.76 |
