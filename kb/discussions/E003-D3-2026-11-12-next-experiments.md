# E003-D3: Given E003-D2, what should the next experiments be?

| id | about | started | status | reviewed |
|---|---|---|---|---|
| E003-D3 | [E003-D2](E003-D2-2026-11-12-why-biphasic-silenced.md), E003-R1 (recommendations 1–6), backlog #1, E003-Q1, E003-Q4 | 2026-11-12 | open | – |

Follows from [E003-D2](E003-D2-2026-11-12-why-biphasic-silenced.md): correlation is the only encoder-to-flow path, and training removed time from the whole front end.

## Thread
- 2026-11-12 · sankarshan.dh: Based on E003-D2, what are the suggestions / next experiments?
- 2026-11-12 · Claude: Backlog #1 / E003-R1 rec. 1–2 (close the spatial silencing route) now looks low-value on its own: D2 says nothing in the loss asks for a temporal signal, so a third silencing route (e.g. the Gabor channel mix) or dead weight is the likely result. Suggest lowering its priority. (evidence: E003-D2 Outcome; Gabor sigma_t at floor, filters.jsonl step 100000)
- 2026-11-12 · Claude: Sintel EPE alone cannot show a benefit: matching already reaches 1.51 without time, and val is the training split (E003-R1 obs. 3). Suggest fixing the success criteria first: held-out Sintel scenes (R1 rec. 3), occluded-pixel EPE (10.37 now), and the random-pattern scores already in the pipeline. Also log Gabor/DoG sigma_t and the freeze test at each validation (R1 rec. 4). (evidence: tests/E003-typed/final/report.md)
- 2026-11-12 · Claude: Proposed E004 (answers E003-Q4): a direct motion path. Feed the front-end response at the reference time into the update block next to the correlation features, behind a new train.py argument whose default is off. One change vs E003.
- 2026-11-12 · Claude: Proposed E005, if E004 still drives time to the floor: fixed front end, as in the paper's Table 2. Freeze the DoG/Gabor shape parameters at designed values (wide sigma_t, a spread of speeds) and train the rest. `--bio_lr_mult 0` already freezes them, but the init has all speeds at 1.0, so a designed init is needed. Caveat: the channel mix is an ordinary weight and could still neutralise them.
- 2026-11-12 · Claude: Cheap checks before any training: (a) Sintel per-frame displacement vs DoG/Gabor scales, to test the dataset hypothesis; (b) E005 smoke variant: Gabor output normalisation cross-channel instead of InstanceNorm (paper Table 3), lower priority.

## Outcome
Open. To settle: the user's decision on (1) lowering backlog #1, (2) success criteria before E004, (3) E004 = direct motion path, (4) E005 = fixed front end, (5) the displacement check. Accepted items become backlog rows or experiments.
