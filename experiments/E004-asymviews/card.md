# E004-asymviews

| id | parent | status | dates | tag | reviewed |
|---|---|---|---|---|---|
| E004-asymviews | E003-typed | planned | – | E004-asymviews | sankarshan.dh, 2026-11-13 (plan) |

Run: `runs/E004-asymviews/` (control: `runs/control-p0/`) · Config: [config.json](config.json) · Origin: [backlog #3](../../kb/backlog.md), [E003-D4](../../kb/discussions/E003-D4-2026-11-13-input-manipulations.md)

## Question
Does training with [asymmetric views](../../kb/glossary.md) (fmap1 and fmap2 share only layout and motion) make the front end use time?

## Change vs parent
- [`asym_prob`](../../kb/glossary.md): 0 (absent in E003) → 0.3: 30% of steps use an asymmetric random-dot batch instead of Sintel.
- Fine-tuned from E003: `restore_ckpt` = E003 final.pth, `num_steps` 100000 → 30000, `lr` 4e-4 → 1.25e-4 (RAFT's Sintel fine-tune rate).
- Control run `control-p0` in this experiment: identical, with `asym_prob 0`, to separate the views from 30k extra steps.

## Expected
- Time is used: asymmetric EPE (`tools/eval_asymmetric_views.py`, A → B) falls below zero flow (1.04), against E003's 249. The freeze-test feature change rises above E003's 3.5%. Biphasic drive std rises above 0.08 / 0 / 0 / 0.016, or the temporal σ leaves its 0.3 floor. In the pipeline, pair-only EPE differs from full-clip EPE.
- A shortcut instead: A → B settles at about 1.04 (zero flow when there is no match) while the freeze test is unchanged.
- Cost: Sintel val EPE more than 0.1 above the control run.

## Result
(after training)

## Verdict
(after training)

## Files
(after training)
