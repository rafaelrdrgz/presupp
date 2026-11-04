# Evidence sentences, cross-validation: fixed_unweighted_pointing

Each gold position of a test passage with evidence that fits the stored sentences has sentences marked from its sentence scores: all at or above its marking threshold (at most 3), or else the strongest; sentences under 4 words aren't marked. Thresholds are tuned on each run's validation passages. Lockbox not used.

1660 positions in 953 passages, seeds [0, 1, 2]. Means over seeds [95% interval from resampling articles].

| measure | value |
|---|---|
| first marked sentence is an evidence sentence | 0.606 [0.583, 0.632] |
| the same for a sentence picked at random | 0.408 [0.396, 0.420] |
| difference | 0.198 [0.176, 0.221] |
| precision of the marked sentences | 0.539 [0.522, 0.557] |
| recall of the evidence sentences | 0.474 [0.457, 0.492] |
| F1 of the marked sentences | 0.504 [0.488, 0.521] |
| sentence AP (threshold-free) | 0.676 [0.661, 0.691] |
| sentences marked per position | 2.64 |

Marking thresholds at the edge of the grid (0.05 or 0.95): 73%.

| position | cases | first marked is evidence | at random | sentence AP |
|---|---|---|---|---|
| anti_reductionism | 39 | 0.35 | 0.33 | 0.54 |
| communitarianism | 91 | 0.59 | 0.38 | 0.67 |
| consequentialism | 586 | 0.64 | 0.46 | 0.70 |
| deontology | 205 | 0.61 | 0.36 | 0.68 |
| determinism_free_will | 69 | 0.52 | 0.37 | 0.61 |
| empiricism | 346 | 0.64 | 0.42 | 0.70 |
| individualism | 123 | 0.59 | 0.33 | 0.65 |
| physicalism_naturalism | 25 | 0.40 | 0.31 | 0.53 |
| rationalism | 9 | 0.26 | 0.27 | 0.42 |
| relativism_objectivism | 89 | 0.56 | 0.40 | 0.64 |
| state_neutrality_perfectionism | 44 | 0.55 | 0.33 | 0.64 |
| virtue_ethics | 34 | 0.61 | 0.38 | 0.70 |
