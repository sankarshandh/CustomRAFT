# E003-F1: Biphasic DoG units that cannot collapse are silenced spatially

| id | confidence | reviewed |
|---|---|---|
| E003-F1 | moderate: one run and one seed, but the mechanism is directly visible in the parameters | sankarshan.dh, 2026-11-12 |

Discussions: [E003-D1](../discussions/E003-D1-2026-11-12-biphasic-switched-off.md), [E003-D2](../discussions/E003-D2-2026-11-12-why-biphasic-silenced.md)

## Claim
On Sintel, when DoG neurons are forced to be biphasic in time, training silences them by making their spatial centre and surround equal, at no EPE cost.

## Evidence
- E003: centre σ ≈ surround σ for neuron 3 (4.495 / 4.496) and neuron 5 (4.940 / 4.941) at 100k; drive std 0 for both, 0.08 for neuron 1 and 0.016 for neuron 7 (filters.jsonl; analysis/temporal/typed_final_2026-11-12.json).
- E003 vs E002: val EPE within about 0.1 from 50k on (val_metrics.jsonl of both).
- S001: neuron 1 still responds to strong motion (+0.29 inside a moving disk), but Gabor energy changes by ≤ 1.5% and the flow output ignores the ring and the ROI (S001 data/random_pattern_probe.json).

## Against
- Neuron 1 is attenuated, not fully silenced.

## Open
- Would a minimum centre/surround gap, or a fixed per-neuron output norm, keep them active, and would they then help? Does another seed give the same result?
