# CustomRAFT

RAFT-small with a bio-inspired spatiotemporal feature encoder: a learnable DoG front end followed by oriented Gabor units, trained on MPI-Sintel. The project is organised by experiment.

```
python tools/new_experiment.py E004-name --parent E003 --set key=value   # define an experiment
python -u train.py --exp experiments/E004-name                           # train it
python evaluate.py --model E004 --dataset sintel                         # evaluate it
python tools/check_compat.py                                             # past experiments still reproduce
```

## Tracking experiments with git

Code, configs, cards and the KB are in git. Runs, checkpoints and figures stay on disk only (see `.gitignore`).
Each experiment has a **tag** with the same name as its folder (`E001-baseline`, `E002-contrast`, `E003-typed`). E000 has no tag because its code is gone.

> **Note on E001–E003:** these three were migrated together, so their tags point at the same import commit. Their code is identical, and they differ only in `config.json` (E002: `dog_gain_mode`, E003: `temporal_mode`). So `git diff` between them is empty, and `git describe` may name any of the three. Use the config comparison below for them. From E004 on, each tag gets its own commit and the code diffs become meaningful.

| I want to… | Command |
|---|---|
| list all experiments | `git tag` (or `ls experiments`) |
| see which experiment the current code belongs to | `git describe --tags`: `E003-typed` means the code is exactly what E003 ran; `E003-typed-2-g1a2b3c` means 2 commits after it |
| see what changed in the code between two experiments | `git diff E002-contrast E003-typed -- core train.py` |
| list just the files that changed | `git diff --stat E002-contrast E003-typed` |
| see what I changed since the last experiment | `git diff E003-typed` (add `--stat` for a summary) |
| compare two experiments' settings | `git diff --no-index experiments/E002-contrast/config.json experiments/E003-typed/config.json` (or read `changes_vs_parent` in config.json) |
| read a file as it was at an experiment | `git show E002-contrast:core/layers/spatiotemporal.py` |
| get the exact code of an experiment in a separate folder (to read or re-run) | `git worktree add ../CustomRAFT-E002 E002-contrast` (remove later with `git worktree remove ../CustomRAFT-E002`) |
| see the history: experiments, code and KB changes | `git log --oneline --decorate` (tags appear next to their commits) |
| see the history of one experiment's folder or one KB item | `git log --oneline -- experiments/E003-typed` · `git log -p -- kb/findings/E003-F1-biphasic-silenced-spatially.md` |
| find who wrote or changed a line in the KB, and when | `git blame kb/INDEX.md` |
| check which commit a run actually used | `cat experiments/E00N-name/runs/<run>/git_commit.txt` (also `git_diff.patch` if the code had uncommitted changes, and `code_snapshot/` for the code itself) |
| confirm that code changes didn't break past experiments | `python tools/check_compat.py` |

When a new experiment starts:
```
git add -A && git commit -m "E004-name: <one-line change>"   # commit the code + config + card first
python -u train.py --exp experiments/E004-name
git tag E004-name                                            # mark the code this experiment ran on
```
Never move or delete a tag: it is how an experiment stays tied to its code.

## More
- Rules and layout: [CLAUDE.md](CLAUDE.md)
- Knowledge base: [kb/INDEX.md](kb/INDEX.md)
- Migrated on 2026-11-12 from `D:\IISc\CustomRAFT`, which is left unchanged. The old KB is in [kb/legacy/](kb/legacy/README.md).
