# E003-R1: What E003 tested, what it missed, what to do next

| id | date | asked by | covers | reviewed |
|---|---|---|---|---|
| E003-R1 | 2026-11-12 | sankarshan.dh | E003, E003-F1, S001 | sankarshan.dh, 2026-11-12 |

## Ask
Question experiment E003 (fixed biphasic DoG units): is the conclusion sound, and what should change next?

## Looked at
E003 card and config; runs/customraft-typed (val_metrics.jsonl, filters.jsonl); analysis/temporal/typed_final_2026-11-12.json; analysis/silencing_check.json; S001 data.

## Observations
1. The temporal constraint held (odd fraction 1.0 throughout), but the spatial DoG was unconstrained, and that is where the silencing happened (σc → σs; spatial kernel total 0.0004 / 0.0002 for neurons 3 and 5).
2. The ε floor helps the silencing: a raw response below ε = 0.005 is not normalised back up (neuron 1: raw RMS 0.00056, drive 0.11).
3. "EPE equal to E002" is measured on the training split (val EPE = fit, not generalisation), so it can't tell whether the units would help on unseen scenes.
4. The silencing was only found after the 11 h run; drive std is not logged during training.
5. Single seed; "Expected" was not written before the run.

## Recommendations
1. E004: typed mode + a minimum surround/centre width ratio (e.g. σs ≥ 1.5 σc) for biphasic neurons, closing the spatial route. (~11 h; answers E003-Q1)
2. In the same E004: L1-normalise the spatial kernel too (Σ|k| = 1), so a biphasic neuron's output can't be shrunk at all. (no extra run)
3. Before E004: add a held-out validation split (whole Sintel scenes left out of training), so EPE differences measure generalisation. (code change; ~1 h)
4. Log drive std and the freeze test at every validation, so silencing is visible at 5k instead of after the run. (code change; ~30 min, slightly slower validation)
5. Write the card's Expected section before every run. (process)
6. Seed replicate of E003 to answer E003-Q3. (~11 h; lower priority)

## Decision
Pending: no decisions taken yet (2026-11-12).
