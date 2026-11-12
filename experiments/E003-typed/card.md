# E003-typed

| id | parent | status | dates | tag | reviewed |
|---|---|---|---|---|---|
| E003-typed | E002-contrast | done | 2026-11-11 19:28 → 2026-11-12 06:18 | E003-typed | sankarshan.dh, 2026-11-12 |

Run: `runs/customraft-typed/` (smoke: `runs/smoke-typed/`) · Config: [config.json](config.json) · Findings: [E003-F1](../../kb/findings/E003-F1-biphasic-silenced-spatially.md) · Questions: [E003-Q1…Q4](../../kb/questions.md) · Reviews: [E003-R1](../../kb/reviews/E003-R1-2026-11-12-typed-review.md) · Discussions: [E003-D1](../../kb/discussions/E003-D1-2026-11-12-biphasic-switched-off.md)

## Question
Are biphasic (transient) DoG units useful or costly on Sintel when training cannot turn them into monophasic ones?

## Change vs parent
- `temporal_mode`: free → typed (the only change; config.json).
- [Typed mode](../../kb/glossary.md): neurons 0/2/4/6 monophasic G(0, σ); neurons 1/3/5/7 biphasic G(−d, σ) − G(+d, σ), exactly odd, zero DC, L1-normalised. Learnable: σ ∈ [0.3, 2], d ∈ [0.25, 1]. Init: σ 0.5 (mono), σ 0.8 and d 0.5 (bi).

## Expected
(Not written before the run; reconstructed from the 2026-11-11 plan.) If biphasic units help, val EPE falls below E002; if they cost, it rises; if the network ignores them, their drive std falls to the ε floor.

## Result
- Val EPE (clean / final, 200 pairs per pass) is within about 0.1 of E002 at the same step from 50k on: at 85k, 1.54 / 2.02 vs 1.53 / 2.18. At 100k: 1.51 / 1.92; full set (1,041 pairs per pass): 1.51 / 2.00. (runs/*/val_metrics.jsonl)
- Biphasic kernels stayed exactly odd (odd fraction 1.0 throughout), so the constraint held. (filters/filters.jsonl)
- The biphasic neurons were silenced through the spatial DoG: centre σ grew to match surround σ (neuron 3: 4.495 vs 4.496; neuron 5: 4.940 vs 4.941), so the spatial kernel cancels. Their gains settled at 1.02–1.14, near the lower bound of 1. (filters.jsonl, step 100000)
- Drive std at 100k, biphasic 1/3/5/7: 0.08 / 0 / 0 / 0.016, vs 2.9–3.9 for monophasic. (analysis/temporal/typed_final_2026-11-12.json)
- Monophasic σ_t reached its 0.3 floor. The encoder is per-frame: in the freeze test, features change by 3.5% and EPE goes from 1.785 to 1.803 (E002 at 90k: 3.6%). (same json; E002 analysis/review_2026-11-11/)

## Verdict
Inconclusive. The biphasic units were not useful, but the run doesn't show whether they could be: the network found a second way to switch them off (spatial cancellation). Neuron 1 was only attenuated; it responds to strong motion ([S001](../../kb/studies/S001-random-pattern-probe/study.md)), but nothing downstream uses it.

## Files
- analysis/temporal/typed_final_2026-11-12.json: freeze test, drive, Gabor input mix, RandomDot EPE
- runs/customraft-typed/filters/: filter-parameter trajectories (jsonl + PNGs)
- kb/legacy/README.md: original log row, including the 2026-11-12 correction
