# Evidence sentences, lockbox: v3_mil

Each gold position of a lockbox passage with evidence that fits the stored sentences has sentences marked from its sentence scores: all at or above its marking threshold (at most 3), or else the strongest; sentences under 4 words aren't marked. Thresholds are tuned on each run's validation fold. The lockbox's evidence wasn't realigned or re-marked as the development passages' was, so compare experiments with each other here, not with the cross-validation figures.

156 positions in 97 passages, seeds [0, 1, 2]. Means over seeds [95% interval from resampling articles].

| measure | value |
|---|---|
| first marked sentence is an evidence sentence | 0.630 [0.560, 0.698] |
| the same for a sentence picked at random | 0.428 [0.391, 0.475] |
| difference | 0.202 [0.147, 0.259] |
| precision of the marked sentences | 0.590 [0.541, 0.643] |
| recall of the evidence sentences | 0.433 [0.395, 0.479] |
| F1 of the marked sentences | 0.499 [0.462, 0.542] |
| sentence AP (threshold-free) | 0.711 [0.670, 0.756] |
| sentences marked per position | 2.38 |

Marking thresholds at the edge of the grid (0.05 or 0.95): 89%.

| position | cases | first marked is evidence | at random | sentence AP |
|---|---|---|---|---|
| anti_reductionism | 4 | 0.42 | 0.26 | 0.52 |
| communitarianism | 8 | 0.62 | 0.44 | 0.76 |
| consequentialism | 58 | 0.69 | 0.47 | 0.73 |
| deontology | 20 | 0.55 | 0.33 | 0.63 |
| determinism_free_will | 6 | 0.44 | 0.41 | 0.62 |
| empiricism | 32 | 0.64 | 0.48 | 0.76 |
| individualism | 11 | 0.76 | 0.29 | 0.72 |
| physicalism_naturalism | 2 | 0.50 | 0.26 | 0.56 |
| rationalism | 1 | 0.00 | 0.38 | 0.34 |
| relativism_objectivism | 7 | 0.71 | 0.60 | 0.74 |
| state_neutrality_perfectionism | 4 | 0.50 | 0.30 | 0.70 |
| virtue_ethics | 3 | 0.44 | 0.39 | 0.69 |
