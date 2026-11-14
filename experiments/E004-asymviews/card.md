# E004-asymviews

| id | parent | status | dates | tag | reviewed |
|---|---|---|---|---|---|
| E004-asymviews | E003-typed | done | 2026-11-13 12:39 → 2026-11-14 07:00 | E004-asymviews | sankarshan.dh, 2026-11-13 (plan) |

Run: `runs/E004-asymviews/` (control: `runs/control-p0/`) · Config: [config.json](config.json) · Findings: [E004-F1](../../kb/findings/E004-F1-asymmetric-views-zero-flow-shortcut.md), [E004-F2](../../kb/findings/E004-F2-silencing-not-a-finetune-trap.md), [E004-F3](../../kb/findings/E004-F3-random-dots-learned-as-no-motion.md) · Questions: [E004-Q1](../../kb/questions.md) · Reviews: [E004-R1](../../kb/reviews/E004-R1-2026-11-13-midtraining.md) · Origin: [backlog #3](../../kb/backlog.md), [E003-D4](../../kb/discussions/E003-D4-2026-11-13-input-manipulations.md)

## Question
Does training with [asymmetric views](../../kb/glossary.md) (fmap1 and fmap2 share only layout and motion) make the front end use time?

## Change vs parent
- [`asym_prob`](../../kb/glossary.md): 0 (absent in E003) → 0.3: 30% of steps use an asymmetric random-dot batch instead of Sintel.
- Fine-tuned from E003: `restore_ckpt` = E003 final.pth, `num_steps` 100000 → 30000, `lr` 4e-4 → 1.25e-4 (RAFT's Sintel fine-tune rate).
- Control run `control-p0` in this experiment: identical, with `asym_prob 0`, to separate the views from 30k extra steps.

## Steps taken (all runs stay in this experiment)
1. 2026-11-13, main run (fine-tune, above). At 10k: biphasic kernels still cancel, A → B 1.049 = zero flow 1.036, the shortcut ([E004-R1](../../kb/reviews/E004-R1-2026-11-13-midtraining.md); analysis/asymmetric_views/10000/).
2. Checkpoint at 20k with the pipeline and the asymmetric-view check, against the "switched on" criteria in E004-R1 decision 2. Outcome: not switched on. A → B 1.047 vs zero flow 1.036 (object 4.307 vs 4.304). Biphasic gaps 5.27 / 0.04 / 0.02 / 0.82% (neuron 1 was already 4.1% at step 0). Biphasic drive std on Sintel 0.153 / 0 / 0 / 0.019 (E003: 0.112 / 0 / 0 / 0.016). Gabor temporal σ median 0.304. (analysis/asymmetric_views/20000/; tests/E004-asymviews/20000/report.md)
3. 2026-11-13: main run stopped at about 23.5k (last checkpoint 20000.pth), `control-p0` cancelled. Next: train from scratch here. Reason: [E004-Q1](../../kb/questions.md), the switched-off units may be a trap for fine-tuning.
4. 2026-11-13: from-scratch run `runs/scratch-p03/` started: `--restore_ckpt none --num_steps 100000 --lr 4e-4 --asym_prob 0.3` (E003's schedule, so E003 is the comparison). Code: commit ae8b1bc. Sintel vs asymmetric loss: `train_metrics_split.csv`.

## Expected
- Time is used: asymmetric EPE (`tools/eval_asymmetric_views.py`, A → B) falls below zero flow (1.04), against E003's 249. The freeze-test feature change rises above E003's 3.5%. Biphasic drive std rises above 0.08 / 0 / 0 / 0.016, or the temporal σ leaves its 0.3 floor. In the pipeline, pair-only EPE differs from full-clip EPE.
- A shortcut instead: A → B settles at about 1.04 (zero flow when there is no match) while the freeze test is unchanged.
- Cost: Sintel val EPE more than 0.1 above the control run.

## Result
- Asymmetric views: zero flow. A → B 1.047 at 20k (fine-tune) and 1.036 from scratch, equal to zero flow 1.036 ([E004-F1](../../kb/findings/E004-F1-asymmetric-views-zero-flow-shortcut.md); analysis/asymmetric_views/).
- From scratch, Sintel val 1.613 / 2.104 vs E003 1.507 / 1.998 (+0.106 each); pair-only = full clip (1.617 vs 1.613) (tests/E004-asymviews/scratch-p03_final/report.md).
- 3 of 4 biphasic units switched off from scratch too ([E004-F2](../../kb/findings/E004-F2-silencing-not-a-finetune-trap.md)); neuron 1 more active (12× vs 43× attenuation).
- Random dots in general tracked 3–4× worse than E003, close to zero flow ([E004-F3](../../kb/findings/E004-F3-random-dots-learned-as-no-motion.md)).

## Verdict
Refuted: asymmetric views don't make the network use time; it learns to say "nothing moves" instead, both fine-tuned and from scratch, at a Sintel cost of 0.106.

## Files
analysis/asymmetric_views/{10000,20000,scratch-p03_final}/ · tests/E004-asymviews/{20000, scratch-p03_*}/ · runs/scratch-p03/train_metrics_split.csv
