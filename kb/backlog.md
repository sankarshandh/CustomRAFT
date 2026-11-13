# Backlog

Planned experiments and studies, one line each. When one starts, it gets an E/S id and a folder; set its status here to that id.

| # | hypothesis / question | proposed change | priority | status |
|---|---|---|---|---|
| 1 | Biphasic units are useful if they can't be silenced | typed mode + minimum centre/surround σ gap (or fixed per-neuron output norm) | high | idea |
| 2 | A per-frame matcher cannot correlate two views that share only motion edges ([E003-D4](discussions/E003-D4-2026-11-13-input-manipulations.md)) | No training: evaluate E003 with fmap1 from random-dot frames with visible objects and fmap2 from the same scene layout rendered camouflaged with independent dots. Predicted: about the zero-flow EPE | high | done 2026-11-13 (`tools/eval_asymmetric_views.py`): E003 diverges, A → B EPE 249 vs zero flow 1.04; the control with only the dots changed fails too (253). See E003-D4 |
| 3 | Asymmetric views make the front end temporal ([E003-D4](discussions/E003-D4-2026-11-13-input-manipulations.md)) | Training augmentation: with probability p, fmap2's window is re-textured with random dots (random-dot scenes first, then Sintel). New train.py argument, default off | high | E004-asymviews (fine-tune stopped at 23.5k: shortcut; from-scratch run planned) |
