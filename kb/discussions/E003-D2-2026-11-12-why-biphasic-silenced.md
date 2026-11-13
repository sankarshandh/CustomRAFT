# E003-D2: Why are the biphasic units silenced: dataset, unit capability, or architecture?

| id | about | started | status | reviewed |
|---|---|---|---|---|
| E003-D2 | E003-F1, E003 card, E003-Q2, E003-Q4 | 2026-11-12 | resolved (dataset and unit capability untested) | sankarshan.dh, 2026-11-12 |

Reference: Tangemann, Kümmerer, Bethge, "Object segmentation from common fate", NeurIPS 2024 (arXiv 2411.01505).

## Thread
- 2026-11-12 · sankarshan.dh: Why are the biphasic units silenced: (1) the dataset, (2) biphasic units can't solve the problem, or (3) the architecture? Read the reference paper.
- 2026-11-12 · Claude: Leans architecture: correlation of fmap1 with fmap2 is the only path from the encoder to the flow, and the Gabor output goes through InstanceNorm (the paper's worst swap, Table 3). Second, redundancy with the Gabor temporal stage. Dataset weakest (the paper sees no dataset effect, Table 1). Paper Table 2: a learned front end loses random-dot generalisation. (evidence: tests/E003-typed/final/report.md: freeze test 3.5% feature change, pair-only EPE = full clip, obj/bg ≤ 0.93)
- 2026-11-12 · sankarshan.dh: Check the learned Gabor velocities.
- 2026-11-12 · Claude: Correction: the Gabor stage does no temporal work either. Speeds drifted from 1.0 to median |v| 3.14, but sigma_t sits at the 0.3 floor for 9/16 neurons (max 0.35), so the side frames get 0.4–1.7% of the centre weight; frequency is at its floor (1/15) for 12/16 and σ at its ceiling (3.5) for most. The redundancy argument is withdrawn; the whole front end is per-frame. E002 is the same (median |v| 3.11). (evidence: runs/customraft-typed/filters/filters.jsonl, step 100000)
- 2026-11-12 · sankarshan.dh: Who's doing the heavy lifting: is it the correlation block?
- 2026-11-12 · Claude: Yes. On 100 Sintel clean val pairs (32 iters), EPE is 1.227 normal, 12.864 with correlation zeroed, 12.456 with fmap2 = fmap1, 12.457 for zero flow: without correlation the model is no better than predicting zero flow. In the GRU, motion features supply 76% of the update-gate input against 3% for context; zeroing inp does not hurt (0.311 → 0.286), zeroing net does (0.311 → 0.400). (evidence: experiments/E003-typed/analysis/corr_ablation/final/corr_ablation.json; tests/E003-typed/final/static_viz/show_context/context_stats.json)

## Outcome
The biphasic units are silenced because the architecture gives temporal signals no path to the flow except through appearance matching; training then removes time from the whole front end (DoG and Gabor), not only from the biphasic units. Dataset and unit capability remain untested as secondary causes. Supports the hypothesis in [E003-Q2](../questions.md) and motivates [E003-Q4](../questions.md) (a direct motion path). No claim in [E003-F1](../findings/E003-F1-biphasic-silenced-spatially.md) changed.

Follow-up: [E003-D3](E003-D3-2026-11-12-next-experiments.md) (next experiments).
