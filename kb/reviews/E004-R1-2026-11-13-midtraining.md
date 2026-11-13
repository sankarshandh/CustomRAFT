# E004-R1: Mid-training review of E004 at 10k, against the expectations set before training

| id | date | asked by | covers | reviewed |
|---|---|---|---|---|
| E004-R1 | 2026-11-13 | sankarshan.dh | E004 (main run, 10k of 30k), E003, E003-D4 | sankarshan.dh, 2026-11-13 |

## Ask
The DoG kernels still cancel at 10k. Review the mid-training state against the card's Expected section.

## Looked at
E004 `runs/E004-asymviews/filters/filters.jsonl` (steps 0/5k/10k) against E003 at 100k; E004 `train.log` (validation at 5k and 10k); the [E004 card](../../experiments/E004-asymviews/card.md)'s Expected section.

## Observations
1. The biphasic DoG kernels still cancel. Centre and surround σ for neurons 3, 5 and 7 differ by 0.001 / 0.001 / 0.042 at 10k, against 0.001 / 0.001 / 0.036 in E003. Their gains sit at 1.02–1.18, near the lower bound. (filters.jsonl)
2. No temporal widening. Gabor temporal σ is 0.300 / 0.304 / 0.346 (min / median / max), against 0.300 / 0.305 / 0.349 in E003. Monophasic DoG temporal σ stays at its 0.3 floor. (filters.jsonl)
3. Sintel val (200-pair subset): 1.591 / 2.067 at 5k and 1.587 / 2.026 at 10k, against E003's 1.51 / 1.92 at 100k. No control run yet. (train.log)
4. The training loss mixes Sintel and asymmetric batches, so we can't tell whether the asymmetric loss is falling: it isn't logged separately.
5. A likely reason nothing moves: fine-tuning starts inside the cancelled state. A cancelled kernel outputs about 0 (spatial L1 0.0004 for neuron 3), so almost no gradient reaches its shape parameters, at an effective rate of 1.25e-4 × 20. The asymmetric task can then be handled without the biphasic units, for example by predicting zero flow when nothing matches (the card's "shortcut" outcome).

Against the expectations set before training: none of the "time is used" signals show in the parameters at 10k. The shortcut outcome fits so far, but needs the asymmetric-view number to confirm. The cost criterion can't be judged until the control run exists.

## Recommendations
1. Run `eval_asymmetric_views` on E004@10000 now (~2 min). It tells three outcomes apart: still diverging (about 249), the shortcut (about 1.04), or real matching (below 1.04, with object and boundary EPE falling).
2. Let the main run finish, with a stopping rule: if at 20k the asymmetric-view EPE is about 1.04 and the biphasic/Gabor parameters still haven't moved, stop and skip the control run.
3. New question E004-Q1: is the cancelled state a trap that fine-tuning can't escape? Test either by re-initialising the biphasic units at fine-tune start (a new argument, default off), or with a from-scratch run with `asym_prob` (a candidate E005).
4. Log the asymmetric and Sintel batch losses separately in future runs. This only adds log columns; training behaviour is unchanged.

## Decision
1. accepted, done 2026-11-13. Result: A → B 1.049 vs zero flow 1.036 (object 4.303 vs 4.304): the shortcut (experiments/E004-asymviews/analysis/asymmetric_views/10000/asymmetric_views.json)
2. accepted (modified): review at 20k with the pipeline and the asymmetric-view check. "Switched on" means any of: A → B < 0.9 (zero flow 1.04) with object EPE < 4.30; biphasic centre/surround gap > 5%, or biphasic drive std clearly up; Gabor temporal σ median > 0.4. If none holds: stop the main run, skip the control run, and train from scratch inside E004 (not a new experiment). Outcome at 20k: not switched on (A → B 1.047 vs zero flow 1.036; gaps and Gabor σ unchanged; see the E004 card, step 2). Main run stopped at about 23.5k, control cancelled.
3. accepted → [E004-Q1](../questions.md)
4.
