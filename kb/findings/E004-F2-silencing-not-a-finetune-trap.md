# E004-F2: The biphasic units are switched off from scratch too, not only when fine-tuning

| id | confidence | reviewed |
|---|---|---|
| E004-F2 | moderate: one from-scratch run, one seed; the same path as E003 is visible in the parameters | sankarshan.dh, 2026-11-14 |

Answers [E004-Q1](../questions.md). Related: [E003-F1](E003-F1-biphasic-silenced-spatially.md).

## Claim
Training E004 from scratch (asym_prob 0.3) switches off 3 of the 4 biphasic DoG units by spatial centre/surround cancellation, as in E003, so their silence is not just a trap left over from E003's weights.

## Evidence
- At 10k, biphasic centre/surround gaps 9.9 / 1.3 / 5.1 / 2.8% (E003 at 10k: 13.1 / 1.4 / 0.6 / 5.4%; 50% at init). (runs/scratch-p03/filters/filters.jsonl; E003 runs/customraft-typed/filters/filters.jsonl)
- At 100k, attenuation vs the median monophasic neuron for neurons 1 / 3 / 5 / 7: 12× / 756,044× / 415× / 37,286× (E003: 43× / 13,292× / 26,525× / 301×). (tests/E004-asymviews/scratch-p03_final/frontend/silencing_check.json)
- Neurons 3 and 7 cancel almost exactly: centre / surround σ 4.787 / 4.787 and 4.987 / 4.988. (same file)

## Against
- Neuron 1 is more active than in E003: drive std on Sintel 0.410 vs 0.112, widths 1.601 / 1.756. But it does not respond more inside moving objects than on the background (ratio 0.84). (same report, DoG table)
- Neuron 5 cancels less exactly (4.748 / 4.769) but is still 415× attenuated.

## Open
- Another seed. Does the same happen with asym_prob 0 from scratch (that is E003) for every seed?
