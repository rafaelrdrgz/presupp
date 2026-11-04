# Choosing v3

Development cross-validation (lockbox not used).

| experiment | macro-F1 (all labels) | first marked sentence is evidence |
|---|---|---|
| v3_joint | 0.478 | 0.613 |
| v3_mil | 0.473 | 0.632 |
| fixed_unweighted_pointing | 0.483 | 0.606 |

**Chosen: v3_mil**, because macro-F1 is within 0.01 and its first marked sentence is evidence 0.019 more often.

**Does not replace v1 in the app:** macro-F1 -0.0106 against the baseline (more than 0.01 below it); first marked sentence is evidence [+0.007, +0.044] more often (above 0).
