# CustomRAFT knowledge base

**Read this file first.** Then open only the card, finding or study a row points to. Go to `kb/legacy/` (the old KB, kept verbatim) or raw files in `analysis/` only when those are not enough.
Everything here is filled in by hand, together: a row is trusted once its `reviewed` column is set.

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

## Findings
| id | claim (one line) | confidence | evidence | reviewed |
|---|---|---|---|---|
| [E003-F1](findings/E003-F1-biphasic-silenced-spatially.md) | Forced-biphasic DoG units are silenced by spatial centre/surround cancellation, at no EPE cost | moderate | E003, S001 | sankarshan.dh 2026-11-12 |

## Studies
| id | question | result (one line) | reviewed |
|---|---|---|---|
| [S001-random-pattern-probe](studies/S001-random-pattern-probe/study.md) | | | |

## Open questions
(top 3–5 open rows from [questions.md](questions.md))

## Reviews
| id | topic | recommendations accepted / total | reviewed |
|---|---|---|---|

## Next up
(top 3 from [backlog.md](backlog.md))

## Map
- `experiments/E00N-name/`: `card.md` (summary), `config.json` (exact args + diff vs parent), `runs/`, `checkpoints/`, `analysis/`
- `kb/findings/`: one claim per file · `kb/studies/`: cross-experiment analyses · `kb/glossary.md`: terms
- `kb/questions.md`: open questions · `kb/reviews/`: reviews and plans with decisions · `kb/backlog.md`: ideas to run
- `kb/templates/`: blank card, finding, study and review · `kb/legacy/`: the old KB, unchanged
- **IDs:** `E003-F1` = finding 1 that came out of E003 (likewise `-Q` question, `-R` review); `G-` = not tied to one experiment; `S001` = a study (compares experiments)
