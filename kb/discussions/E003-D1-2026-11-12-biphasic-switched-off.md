# E003-D1: Were the biphasic neurons "effectively switched off"?

| id | about | started | status | reviewed |
|---|---|---|---|---|
| E003-D1 | E003-F1, E003 card | 2026-11-12 | resolved | sankarshan.dh, 2026-11-12 |

## Thread
- 2026-11-12 · Claude: after the run, "the biphasic units were effectively switched off" (Sintel drive std 0.08 / 0 / 0 / 0.016).
- 2026-11-12 · sankarshan.dh: "But you told biphasic was effectively switched off": neuron 1 responds strongly in the random-pattern probe (S001).
- 2026-11-12 · Claude: partly wrong. Neurons 3 and 5 are dead and 7 nearly so, but neuron 1 is only attenuated; strong motion on random dots drives it through (raw RMS 0.0019 vs 0.00056 on Sintel). Nothing downstream uses it (Gabor change ≤ 1.5%). (evidence: analysis/silencing_check.json, S001 data)
- 2026-11-12 · Claude: correction on re-measuring: the attenuation is 43× relative to the median monophasic neuron, not "about 30×". (evidence: analysis/silencing_check.json)

## Outcome
The claim now reads "three dead, neuron 1 attenuated and unused downstream". E003-F1's Against section records the neuron-1 caveat. The "30×" figure appears only in `kb/legacy/`, which stays unedited.
