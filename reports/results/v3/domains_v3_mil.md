# Domains: v3_mil

How reliably the model finds each domain. A domain counts as predicted for a passage when any of its categories is predicted, and as present when any of its categories is in the gold labels.

Development cross-validation (lockbox not used): 1114 passages, seeds [0, 1, 2]. Means over seeds [95% interval from resampling articles].

| domain | passages with it | precision | recall | F1 [95% interval] |
|---|---|---|---|---|
| Exclusion | 161 | 0.658 | 0.458 | 0.540 [0.458, 0.617] |
| Metaphysical | 113 | 0.480 | 0.516 | 0.497 [0.366, 0.612] |
| Epistemological | 428 | 0.688 | 0.750 | 0.717 [0.676, 0.749] |
| Ethical | 715 | 0.798 | 0.889 | 0.841 [0.822, 0.860] |
| Political-Philosophical | 221 | 0.489 | 0.597 | 0.537 [0.459, 0.608] |

| summary | value |
|---|---|
| macro-F1 over the 4 philosophical domains | 0.648 [0.608, 0.682] |
| passages whose whole set of domains is right | 0.423 [0.395, 0.451] |
