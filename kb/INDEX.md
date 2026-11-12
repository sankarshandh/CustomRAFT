# CustomRAFT knowledge base

**Read this file first.** Then open only the card, finding or study a row points to. Go to `kb/legacy/` (the old KB, kept verbatim) or raw files in `analysis/` only when those are not enough.
Everything here is filled in by hand, together: a row is trusted once its `reviewed` column is set.

## Project
(3 lines: what the model is, what it is trained on, the main question)

## What we know now
(5–8 bullets, each ending with a finding link, e.g. → [F01](findings/F01-name.md))

## Experiments
| id | parent | change vs parent | status | val EPE clean / final | verdict | reviewed |
|---|---|---|---|---|---|---|
| [E000-infra](../experiments/E000-infra/card.md) | – | | | | | |
| [E001-baseline](../experiments/E001-baseline/card.md) | – | | | | | |
| [E002-contrast](../experiments/E002-contrast/card.md) | E001 | | | | | |
| [E003-typed](../experiments/E003-typed/card.md) | E002 | | | | | |

## Findings
| id | claim (one line) | confidence | evidence | reviewed |
|---|---|---|---|---|

## Studies
| id | question | result (one line) | reviewed |
|---|---|---|---|
| [S001-random-pattern-probe](studies/S001-random-pattern-probe/study.md) | | | |

## Next up
(top 3 from [backlog.md](backlog.md))

## Map
- `experiments/E00N-name/`: `card.md` (summary), `config.json` (exact args + diff vs parent), `runs/`, `checkpoints/`, `analysis/`
- `kb/findings/`: one claim per file · `kb/studies/`: cross-experiment analyses · `kb/glossary.md`: terms
- `kb/templates/`: blank card, finding and study · `kb/legacy/`: the old KB, unchanged
