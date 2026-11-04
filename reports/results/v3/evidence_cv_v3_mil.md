# Evidence sentences, cross-validation: v3_mil

Each gold position of a test passage with evidence that fits the stored sentences has sentences marked from its sentence scores: all at or above its marking threshold (at most 3), or else the strongest; sentences under 4 words aren't marked. Thresholds are tuned on each run's validation passages. Lockbox not used.

1660 positions in 953 passages, seeds [0, 1, 2]. Means over seeds [95% interval from resampling articles].

| measure | value |
|---|---|
| first marked sentence is an evidence sentence | 0.632 [0.608, 0.657] |
| the same for a sentence picked at random | 0.408 [0.396, 0.420] |
| difference | 0.224 [0.203, 0.246] |
| precision of the marked sentences | 0.575 [0.556, 0.595] |
| recall of the evidence sentences | 0.481 [0.462, 0.502] |
| F1 of the marked sentences | 0.524 [0.506, 0.544] |
| sentence AP (threshold-free) | 0.710 [0.692, 0.727] |
| sentences marked per position | 2.51 |

Marking thresholds at the edge of the grid (0.05 or 0.95): 90%.

| position | cases | first marked is evidence | at random | sentence AP |
|---|---|---|---|---|
| anti_reductionism | 39 | 0.44 | 0.33 | 0.59 |
| communitarianism | 91 | 0.59 | 0.38 | 0.69 |
| consequentialism | 586 | 0.66 | 0.46 | 0.74 |
| deontology | 205 | 0.66 | 0.36 | 0.72 |
| determinism_free_will | 69 | 0.57 | 0.37 | 0.65 |
| empiricism | 346 | 0.69 | 0.42 | 0.74 |
| individualism | 123 | 0.57 | 0.33 | 0.66 |
| physicalism_naturalism | 25 | 0.32 | 0.31 | 0.49 |
| rationalism | 9 | 0.37 | 0.27 | 0.43 |
| relativism_objectivism | 89 | 0.58 | 0.40 | 0.68 |
| state_neutrality_perfectionism | 44 | 0.48 | 0.33 | 0.61 |
| virtue_ethics | 34 | 0.65 | 0.38 | 0.73 |
