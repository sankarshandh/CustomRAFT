# CustomRAFT: working rules (for humans and LLMs)

RAFT-small whose feature encoder is a learnable spatiotemporal DoG → Gabor front end, trained on MPI-Sintel.
Python env: conda `motion`. Data: `D:\IISc\Motion Datasets\` (Sintel, RandomDot, `generate_random_dot.py`).

## Start here
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
- `kb/`: `INDEX.md`, `findings/`, `studies/S00N-name/`, `glossary.md`, `backlog.md`, `templates/`
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

## Outputs
- An analysis of one experiment goes to `experiments/E00N/analysis/`.
- A comparison across experiments is a study and goes to `kb/studies/S00N-name/data` and `figures`.

## KB rules
- **The KB is filled together with the user, never automatically.**
  1. The assistant prepares a fact sheet in chat: numbers with their source files, what is uncertain, and open questions.
  2. The user decides the claim, verdict and wording.
  3. The assistant writes exactly the agreed text and sets `reviewed`.
- Use the templates in `kb/templates/`.
  - A card or study is at most 40 lines; a finding is at most 25.
  - Every number names its source file.
  - Define terms once, in `glossary.md`.
- An experiment that confirms or contradicts a finding gets a new evidence bullet in that finding; don't rewrite the claim silently.
