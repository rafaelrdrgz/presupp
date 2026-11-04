# Domains: fixed_unweighted

How reliably the model finds each domain. A domain counts as predicted for a passage when any of its categories is predicted, and as present when any of its categories is in the gold labels.

Development cross-validation (lockbox not used): 1114 passages, seeds [0, 1, 2]. Means over seeds [95% interval from resampling articles].

| domain | passages with it | precision | recall | F1 [95% interval] |
|---|---|---|---|---|
| Exclusion | 159 | 0.630 | 0.518 | 0.567 [0.489, 0.634] |
| Metaphysical | 106 | 0.471 | 0.610 | 0.530 [0.419, 0.625] |
| Epistemological | 419 | 0.653 | 0.760 | 0.702 [0.663, 0.737] |
| Ethical | 709 | 0.791 | 0.874 | 0.830 [0.808, 0.850] |
| Political-Philosophical | 216 | 0.490 | 0.568 | 0.526 [0.453, 0.594] |

| summary | value |
|---|---|
| macro-F1 over the 4 philosophical domains | 0.647 [0.609, 0.679] |
| passages whose whole set of domains is right | 0.417 [0.387, 0.448] |
