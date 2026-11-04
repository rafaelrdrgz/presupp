# Evidence sentences, cross-validation: v3_joint

Each gold position of a test passage with evidence that fits the stored sentences has sentences marked from its sentence scores: all at or above its marking threshold (at most 3), or else the strongest; sentences under 4 words aren't marked. Thresholds are tuned on each run's validation passages. Lockbox not used.

1660 positions in 953 passages, seeds [0, 1, 2]. Means over seeds [95% interval from resampling articles].

| measure | value |
|---|---|
| first marked sentence is an evidence sentence | 0.613 [0.591, 0.637] |
| the same for a sentence picked at random | 0.408 [0.396, 0.420] |
| difference | 0.205 [0.185, 0.225] |
| precision of the marked sentences | 0.576 [0.559, 0.593] |
| recall of the evidence sentences | 0.460 [0.440, 0.481] |
| F1 of the marked sentences | 0.512 [0.494, 0.531] |
| sentence AP (threshold-free) | 0.700 [0.685, 0.716] |
| sentences marked per position | 2.40 |

Marking thresholds at the edge of the grid (0.05 or 0.95): 89%.

| position | cases | first marked is evidence | at random | sentence AP |
|---|---|---|---|---|
| anti_reductionism | 39 | 0.42 | 0.33 | 0.56 |
| communitarianism | 91 | 0.53 | 0.38 | 0.67 |
| consequentialism | 586 | 0.66 | 0.46 | 0.73 |
| deontology | 205 | 0.64 | 0.36 | 0.71 |
| determinism_free_will | 69 | 0.47 | 0.37 | 0.61 |
| empiricism | 346 | 0.69 | 0.42 | 0.74 |
| individualism | 123 | 0.53 | 0.33 | 0.64 |
| physicalism_naturalism | 25 | 0.33 | 0.31 | 0.51 |
| rationalism | 9 | 0.33 | 0.27 | 0.43 |
| relativism_objectivism | 89 | 0.54 | 0.40 | 0.66 |
| state_neutrality_perfectionism | 44 | 0.48 | 0.33 | 0.62 |
| virtue_ethics | 34 | 0.60 | 0.38 | 0.68 |
