# CustomRAFT knowledge base

**Read this file first.** Then open only the card, finding or study a row points to. Go to `kb/legacy/` (the old KB, kept verbatim) or raw files in `analysis/` only when those are not enough.
Everything here is filled in by hand, together: a row is trusted once its `reviewed` column is set.

## Now (updated 2026-11-14)
- Nothing training. Last experiment: E004-asymviews (done, refuted): asymmetric views teach "nothing moves",
  not the use of time; biphasic units switched off from scratch too (E004-F1, E004-F2).
- Pending: your decisions in E003-D5 (restate the hypothesis as "filters help if read out as motion"?) and
  E003-D3; E003-R1 recommendations 1–6.
- Open lead: E004-F3 (random dots learned as "no motion"); test by mixing normal random-dot batches into training.
- Next step: still under discussion (E003-D5). Candidate: a direct motion path (E003-Q4). No experiment opened yet.

## Project
(3 lines: what the model is, what it is trained on, the main question)

## What we know now
(5–8 bullets, each ending with a finding link, e.g. → [E003-F1](findings/E003-F1-name.md))

## Experiments
| id | parent | change vs parent | status | val EPE clean / final | verdict | reviewed |
|---|---|---|---|---|---|---|
| [E000-infra](../experiments/E000-infra/card.md) | – | | | | | |
| [E001-baseline](../experiments/E001-baseline/card.md) | – | | | | | |
| [E002-contrast](../experiments/E002-contrast/card.md) | E001 | | | | | |
| [E003-typed](../experiments/E003-typed/card.md) | E002 | DoG temporal type fixed: 4 monophasic + 4 biphasic | done | 1.51 / 2.00 (full) | inconclusive: biphasic units silenced spatially; EPE = E002 | sankarshan.dh 2026-11-12 |
| [E004-asymviews](../experiments/E004-asymviews/card.md) | E003 | asymmetric views (asym_prob 0.3): fine-tune stopped at 23.5k, then 100k from scratch (scratch-p03) | done | 1.61 / 2.10 (full, from scratch) | refuted: zero-flow shortcut on asymmetric views; biphasic units still off | sankarshan.dh 2026-11-14 |

## Findings
| id | claim (one line) | confidence | evidence | reviewed |
|---|---|---|---|---|
| [E003-F1](findings/E003-F1-biphasic-silenced-spatially.md) | Forced-biphasic DoG units are silenced by spatial centre/surround cancellation, at no EPE cost | moderate | E003, S001 | sankarshan.dh 2026-11-12 |
| [E004-F1](findings/E004-F1-asymmetric-views-zero-flow-shortcut.md) | Asymmetric views teach "nothing moves" (zero flow), not the use of time | moderate | E004 (fine-tune + scratch) | sankarshan.dh 2026-11-14 |
| [E004-F2](findings/E004-F2-silencing-not-a-finetune-trap.md) | Biphasic units are switched off from scratch too, not only when fine-tuning | moderate | E004 scratch-p03, E003 | sankarshan.dh 2026-11-14 |
| [E004-F3](findings/E004-F3-random-dots-learned-as-no-motion.md) | After asymmetric training the network predicts near-zero flow on all random dots (cause untested) | moderate effect / weak cause | E004 scratch-p03 | sankarshan.dh 2026-11-14 |

## Studies
| id | question | result (one line) | reviewed |
|---|---|---|---|
| [S001-random-pattern-probe](studies/S001-random-pattern-probe/study.md) | | | |

## Open questions
(top 3–5 open rows from [questions.md](questions.md))
- [E003-Q1](questions.md): Do biphasic units improve flow if they can't be silenced?
- [E003-Q2](questions.md): Why does training silence transient units?
- [E003-Q4](questions.md): Would neuron 1's motion signal be used if it reached the flow update directly?

## Reviews
| id | topic | recommendations accepted / total | reviewed |
|---|---|---|---|
| [E003-R1](reviews/E003-R1-2026-11-12-typed-review.md) | What E003 tested, what it missed, what to do next | pending / 6 | sankarshan.dh 2026-11-12 |
| [E004-R1](reviews/E004-R1-2026-11-13-midtraining.md) | Mid-training review of E004 at 10k vs expectations | 4 accepted / 4 | sankarshan.dh 2026-11-13 |

## Discussions
| id | topic | status | outcome (one line) |
|---|---|---|---|
| [E003-D1](discussions/E003-D1-2026-11-12-biphasic-switched-off.md) | Were the biphasic neurons "effectively switched off"? | resolved | three dead, neuron 1 attenuated (43×) and unused downstream |
| [E003-D2](discussions/E003-D2-2026-11-12-why-biphasic-silenced.md) | Why are the biphasic units silenced: dataset, capability or architecture? | resolved | architecture: correlation is the only encoder-to-flow path (EPE 1.23 → 12.86 without it); whole front end went per-frame |
| [E003-D3](discussions/E003-D3-2026-11-12-next-experiments.md) | Given E003-D2, what should the next experiments be? | open | pending decisions: direct motion path (E004), fixed front end, success criteria first |
| [E003-D4](discussions/E003-D4-2026-11-13-input-manipulations.md) | Can changing the training input (temporal noise, asymmetric views) make the front end use time? | open | – |
| [E003-D5](discussions/E003-D5-2026-11-13-retinal-hypothesis.md) | Where does the "retinal bandpass filters → motion" hypothesis go wrong? | open | front end is read only by matching; the correlation already does delay-and-compare; proposed: direct motion path |
| [G-D1](discussions/G-D1-2026-11-12-id-numbering.md) | Do E, S and F numbers match? | resolved | IDs carry their origin: E003-F1, E003-Q1, …; G- for general |

## Next up
(top 3 from [backlog.md](backlog.md))

## Map
- `experiments/E00N-name/`: `card.md` (summary), `config.json` (exact args + diff vs parent), `runs/`, `checkpoints/`, `analysis/`
- `kb/findings/`: one claim per file · `kb/studies/`: cross-experiment analyses · `kb/glossary.md`: terms
- `kb/questions.md`: open questions · `kb/reviews/`: reviews and plans with decisions · `kb/discussions/`: debates, with who said what and the outcome · `kb/backlog.md`: ideas to run
- `kb/templates/`: blank card, finding, study, review and discussion · `kb/legacy/`: the old KB, unchanged
- **IDs:** `E003-F1` = finding 1 that came out of E003 (likewise `-Q` question, `-R` review, `-D` discussion); `G-` = not tied to one experiment; `S001` = a study (compares experiments)
