# CustomRAFT

RAFT-small with a bio-inspired spatiotemporal feature encoder: a learnable DoG front end followed by oriented Gabor units, trained on MPI-Sintel. The project is organised by experiment.

```
python tools/new_experiment.py E004-name --parent E003 --set key=value   # define an experiment
python -u train.py --exp experiments/E004-name                           # train it
python evaluate.py --model E004 --dataset sintel                         # evaluate it
python tools/check_compat.py                                             # past experiments still reproduce
python tools/pipeline.py E004                                            # test + visualise it -> tests/E004-name/
```

## Testing and visualising a checkpoint

After training, run the pipeline. It writes only to `tests/`, never to `kb/`.

```
python tools/pipeline.py E004                       # final checkpoint (full) + 4 evenly spaced steps (light)
python tools/pipeline.py E004 --steps 50000,final   # chosen checkpoints
python tools/pipeline.py E004 --only motion         # one stage (metrics, frontend, probes, static_viz, motion, report)
python tools/pipeline.py E004 --force               # redo stages that are already done
python tools/pipeline.py --report_only              # rebuild all reports and tests/README.md
```

What you get in `tests/E004-name/<step>/`:

| file | what it is |
|---|---|
| `report.md` | key numbers vs the parent experiment, an auto-flagged "unusual" list, a per-DoG-neuron table, the training trajectory, links to all figures |
| `index.html` | open in a browser: this checkpoint and its parent on the same random-pattern scenes (occluded, hollow, roi, visible), played in sync (space = play/pause, ←/→ = step, flow or layer panel) |
| `metrics/` | Sintel val EPE (read from the run's logs when available), training curves vs the parent |
| `frontend/` | `silencing_check.json`, `temporal.json` (temporal kernels, drive, Gabor input mix, freeze test), filter parameters over training |
| `patterns/` | EPE on the random-pattern cases (see below), against texture flow and shape flow, per region (object, hole, boundary, background) |
| `probes/` | per-neuron responses to moving disks, rings and second-order motion |
| `static_viz/` | layer outputs, correlation inputs, context encoder + ablation |
| `motion/` | one scene per random-pattern case: frame sequences and `.mp4` of the flow and layer panels (hollow and roi get an extra row against the shape motion), DoG time courses, EPE with the full clip vs pair-only input |

### Random-pattern test cases

All textures are binary random dots with the same statistics, so single frames show nothing (except in `visible`). As in the original camouflage set, 50% of scenes also have a moving background.

| case | what moves | dataset |
|---|---|---|
| occluded | a textured shape moves over the random-dot background (same as the old `camouflage` set) | `RandomDot/occluded` |
| hollow | only a ring of texture (8 px) moves with the shape; the background shows inside the ring | `RandomDot/hollow` |
| roi | a window moves: pattern 2 inside, pattern 1 outside; pattern 2 itself is static | `RandomDot/roi` |
| visible | control: like occluded, but the objects are +60 grey, so they show in single frames | `RandomDot/visible` |

EPE is measured against two ground truths. **Texture flow** is how the dots move, which is what RAFT is trained to report: 0 inside a ROI, the background motion in a hollow ring's hole. **Shape flow** is how the outline moves. To regenerate the datasets (motion env, in `D:\IISc\Motion Datasets`):
```
python generate_random_dot.py --out RandomDot/occluded
python generate_random_dot.py --out RandomDot/hollow --object_type hollow
python generate_random_dot.py --out RandomDot/roi --object_type aperture
```
`RandomDot/camouflage` is kept on disk (old analysis files refer to it) but is no longer used by the pipeline.

`tests/README.md` lists every checkpoint that has been through the pipeline. The stimuli and flag thresholds live in `configs/pipeline.json`. Keep them fixed so that experiments stay comparable. Run the pipeline on the parent too, so the report has something to compare against.

Every stage can also be run on its own:

```
python tools/analyse_temporal.py E004 [--out_dir DIR]       # temporal use, drive, Gabor mix, freeze test (+ RandomDot camouflage/visible EPE unless --no_rdk)
python tools/eval_random_patterns.py E004                   # EPE on occluded / hollow / roi / visible, texture and shape flow
python tools/silencing_check.py E004 [--out FILE]           # how silent each DoG neuron is, and why
python tools/probe_random_patterns.py E004 --out_json F --fig_dir D
python tools/show_layer_outputs.py --model E004             # DoG / Gabor / head features on one random-dot scene
python tools/show_corr_inputs.py --model E004               # what enters the 4D correlation
python tools/show_context.py --model E004                   # context encoder and how much the GRU uses it
python tools/render_motion.py E004 [--light]                # motion frame sequences + videos
```

### Live views (interactive windows)

```
python tools/live_random_dot.py --model E004 --layers       # flow on random-dot scenes, plus the layer window
python tools/live_encoder.py --model E004 --stage dog       # DoG layer: every neuron, the selected one large, its kernels
python tools/live_encoder.py --model E004 --stage gabor     # Gabor energy + direction plot at a clicked pixel
python tools/live_encoder.py --model E004 --stage cnn       # fmap1/fmap2, correlation at a clicked pixel, context
python tools/live_encoder.py --model E004 --sintel alley_1 --stage gabor   # the same on a real Sintel sequence
```
Keys in `live_encoder`: `d`/`g`/`k` switch view, `0`–`9` or `[` `]` pick a neuron or unit, `a` switches between a shared scale (silenced units look dark) and a per-tile scale, `space` pauses, `←`/`→` step, `n` next scene, `c` next random-pattern case (occluded → hollow → roi → visible), `f` clean/final, `s` screenshot, `q` quit.

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
