# E004-F1: Asymmetric views teach "nothing moves", not the use of time

| id | confidence | reviewed |
|---|---|---|
| E004-F1 | moderate: two runs (fine-tuned and from scratch), one seed each; the shortcut is measured directly | sankarshan.dh, 2026-11-14 |

Discussions: [E003-D4](../discussions/E003-D4-2026-11-13-input-manipulations.md), [E003-D5](../discussions/E003-D5-2026-11-13-retinal-hypothesis.md)

## Claim
When 30% of training batches are [asymmetric views](../glossary.md) (pairs that cannot be matched), the network learns to predict zero flow on them instead of using its temporal filters.

## Evidence
- E004 fine-tune: A → B 1.049 at 10k and 1.047 at 20k, against zero flow 1.036. (experiments/E004-asymviews/analysis/asymmetric_views/10000/, 20000/)
- E004 from scratch: A → B 1.036, equal to zero flow in every region. (analysis/asymmetric_views/scratch-p03_final/asymmetric_views.json)
- During training the asymmetric loss sat at the zero-flow level from about step 2k on: ratio 1.04–1.07. (runs/scratch-p03/train_metrics_split.csv)
- Time is still unused: pair-only input gives the same EPE as the full clip (1.617 vs 1.613), and freezing the window changes EPE only from 1.825 to 1.836. (tests/E004-asymviews/scratch-p03_final/report.md)

## Against
- Freeze-test feature change rose to 9.3% (E003: 3.5%), and biphasic neuron 1 is less attenuated (12× vs 43×; see [E004-F2](E004-F2-silencing-not-a-finetune-trap.md)). Neither changes the EPE.

## Open
- Would a direct motion path (Gabor responses into the update block, [E003-Q4](../questions.md)) let the network use time on these batches? Another seed?
