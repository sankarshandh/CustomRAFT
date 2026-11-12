# CustomRAFT: working rules (for humans and LLMs)

RAFT-small whose feature encoder is a learnable spatiotemporal DoG → Gabor front end, trained on MPI-Sintel.
Python env: conda `motion`. Data: `D:\IISc\Motion Datasets\` (Sintel, RandomDot, `generate_random_dot.py`).

## Start here
- New session: run `/catchup` (or read `kb/INDEX.md`, starting with its "Now" section).
- At the end of a session, propose an updated "Now" section (3–5 lines) and write it after the user approves.
- Read `kb/INDEX.md` first. Open a card, finding or study only when you need it, and `kb/legacy/` only as a last resort.
- `kb/legacy/` is the old KB, kept verbatim. Never edit it.

## Layout
- `core/`: model code · `train.py` · `evaluate.py`
- `tools/`: checks, probes, analyses
- `configs/base.json`: default train arguments
- `experiments/E00N-name/` holds everything for one experiment:
  - `card.md`, `config.json` (args + `changes_vs_parent`), `compat.pt`
  - `runs/<run>/`: logs, metrics, filters, `git_commit.txt`, `code_snapshot/`
  - `checkpoints/<step>.pth` and `final.pth`
  - `analysis/<tool>/<step>/`
- `kb/`: `INDEX.md`, `findings/`, `studies/S00N-name/`, `questions.md`, `reviews/`, `discussions/`, `glossary.md`, `backlog.md`, `templates/`
- `tests/<E00N-name>/<step>/`: pipeline output (`tools/pipeline.py`): `report.md`, `summary.json`, `index.html`, one folder per stage. Generated, not reviewed. `tests/README.md` lists them all.
- `scratch/`: throwaway output (git-ignored)

## Checkpoint references
Tools and `evaluate.load_model` accept `E003` (final checkpoint), `E003@30000`, or a `.pth` path.

## Changing code
1. **Never change the behaviour of a past experiment.** Put new behaviour behind a new `train.py` argument whose default is the old behaviour. `temporal_mode` and `dog_gain_mode` are examples.
2. Run `python tools/check_compat.py` before every commit that touches `core/`. It must print "all experiments reproduce".
3. Commit code before training. `train.py --exp` records the commit, any uncommitted diff, and a code snapshot.

## Running an experiment
1. `python tools/new_experiment.py E004-name --parent E003 --set key=value`. This writes `config.json` and an empty `card.md`.
2. Fill in the card's **Question**, **Change vs parent** and **Expected** together with the user, before training.
3. `git add -A && git commit -m "E004-name: <change>"`
4. `python -u train.py --exp experiments/E004-name`. Add `--name smoke --num_steps 300` for a smoke run inside the same experiment.
5. `git tag E004-name`
6. When training is done, run `python tools/check_compat.py --make E004`, then commit.
7. `python tools/pipeline.py E004`. Then read `tests/E004-name/final/report.md` and open its `index.html` (the motion of E004 and its parent, played in sync). The parent must have been through the pipeline too.

## Outputs
- An analysis of one experiment goes to `experiments/E00N/analysis/`.
- A comparison across experiments is a study and goes to `kb/studies/S00N-name/data` and `figures`.

## KB flow and IDs
```
question (Q) ──► review / plan (R) ──► idea (backlog #) ──► experiment (E) ──► finding (F)
       └──────────── answered directly by a study (S) ────────────────────────────┘
discussion (D): a debate about any of these; it records who argued what and what changed as a result
```
- **IDs carry their origin:** `E003-F1` is the first finding that came out of E003. `-Q` is a question and `-R` a review.
  - `G-Q1` and `G-R1` are general, not tied to one experiment.
  - Studies are `S001…`, because they compare experiments.
  - Later evidence from other experiments goes into the same item; the ID never changes.
- **New question:** add a row to `kb/questions.md`. Close it as `answered → <id>`.
- **New idea:** add a row to `kb/backlog.md`. When it becomes an experiment, set its status to the E id.
- **Review or plan the user asks for:**
  1. Draft it in chat first.
  2. After approval, write it to `kb/reviews/<id>-<date>-<topic>.md` from `kb/templates/review.md`, at most 60 lines.
  3. The user fills in Decision for each recommendation. Accepted items become backlog rows or experiments.
- **Discussions and debates** (a challenge to a claim, a disagreement, a correction) go in `kb/discussions/<id>-<date>-<topic>.md`, using `kb/templates/discussion.md`:
  - The thread has one dated entry per point (who, the point in 1–2 lines, the evidence) and ends with an Outcome saying what changed, with links.
  - Link the discussion from every item it changed (card / finding / review line "Discussions: …").
  - At the end of a discussion, Claude drafts the thread summary in chat. It is written only after the user approves it, never word for word from memory alone.
- **Attribution ("raised by", "asked by", thread entries):** the user's name only for what the user raised; "Claude" for what Claude raised.
- `kb/INDEX.md` lists open questions (top 3–5), all reviews and discussions (one line each), findings and experiments.

## KB rules
- **The KB is filled together with the user, never automatically.**
  0. Before drafting a card, finding, review or study, the assistant reads `tests/<E>/…/report.md` (key numbers, the unusual list, the trajectory) for the experiment and its parent, and cites those files. Auto-flags are leads to check, not conclusions. Nothing is copied from `tests/` into `kb/` without agreement.
  1. The assistant prepares a fact sheet in chat: numbers with their source files, what is uncertain, and open questions.
  2. The user decides the claim, verdict and wording.
  3. The assistant writes exactly the agreed text and sets `reviewed`.
- Use the templates in `kb/templates/`.
  - A card or study is at most 40 lines; a finding is at most 25.
  - Every number names its source file.
  - Define terms once, in `glossary.md`.
- An experiment that confirms or contradicts a finding gets a new evidence bullet in that finding; don't rewrite the claim silently.
