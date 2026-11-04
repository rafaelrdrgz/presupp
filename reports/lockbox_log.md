# Lockbox log

Every opening of the held-out fold. It is meant to be opened once, for the final evaluation.

| when (UTC) | reason |
|---|---|
| 2026-10-24 16:42 | Headline chosen on dev CV before opening: fixed_unweighted (ties v1_recipe_grouped_cv on accuracy, better calibrated). v1_recipe_grouped_cv and tfidf_lr reported alongside. |
| 2026-10-29 21:28 | v3: v3_mil, chosen on dev CV and put in the app, against fixed_unweighted_pointing. Second use of the lockbox; no model trains on it. |
| 2026-10-29 23:19 | v3 re-scored: the lockbox evidence's marking thresholds hadn't been tuned on fold 9 (a bug; every one fell back to 0.5). Same final runs, nothing retrained or re-chosen. |
