# E004-F3: After asymmetric training, the network predicts near-zero flow on all random dots

| id | confidence | reviewed |
|---|---|---|
| E004-F3 | moderate for the effect (all four random-pattern cases); weak for the cause (untested) | sankarshan.dh, 2026-11-14 |

## Claim
E004 (from scratch) tracks even normal, matchable random-dot motion poorly, close to predicting zero flow. A likely cause: random-dot texture appeared in training only in unmatchable asymmetric batches, so the network learned "random dots → nothing moves".

## Evidence
- Normal random dots (A → A): EPE 0.937 vs E003 0.255; zero flow is 1.036. (analysis/asymmetric_views/scratch-p03_final/asymmetric_views.json; E003 analysis/asymmetric_views/final/)
- Random-pattern cases occluded / hollow / roi / visible: 0.974 / 0.761 / 0.614 / 0.924 vs E003 0.276 / 0.265 / 0.180 / 0.256. Occluded object interior 4.398 vs 1.038. (tests/E004-asymviews/scratch-p03_final/report.md)
- Inside a static ROI it predicts exactly zero: 0.001 vs texture flow. (same report)
- Sintel is affected much less: 1.613 / 2.104 vs 1.507 / 1.998. (same report)

## Against
- none known; the cause has not been tested.

## Open
- Test the cause: mix normal (symmetric) random-dot batches into training alongside the asymmetric ones. If random-dot EPE returns to E003's level, the cause holds.
