# `customraft-contrast`: run review and improvement plan

*2026-11-11. Runs: [`runs/customraft-contrast`](../runs/customraft-contrast) (stopped at 90k) and [`runs/customraft`](../runs/customraft) (stopped at about 34k).*
*Status: **plan only. Nothing below has been implemented, and the experiment log in the README has not been updated.***
*Measured numbers come from [`data/contrast_review_2026-11-11.json`](data/contrast_review_2026-11-11.json), from each run's `filters/filters.jsonl`, `val_metrics.jsonl` and `train_metrics.csv`, and from reading the code. The JSON was produced by a read-only script that is not part of the repo; section "How the measurements were made" describes it.*

## TL;DR
- The contrast normalisation did what it was built to do: the ON/OFF GELUs now rectify.
- The temporal filters collapsed anyway, and more completely than in the baseline. At 90k the feature encoder is a per-frame encoder that costs 8 frames of input.
- So the near-linear front end (H4 in the [collapse note](2026-11-05_biphasic-filters-collapse.md)) was not what removed the temporal kernels.
- Three things in the training pipeline penalise temporal filters or hide the result: the eraser augmentation, edge-frame replication, and validation on the training set. They should be fixed before drawing further conclusions.

## What the run shows

### 1. The encoder is per-frame at 90k (measured)
| | `customraft` 30k | `contrast` 30k | `contrast` 90k |
|---|---|---|---|
| DoG neurons with ≥ 96% of kernel weight on the centre frame | 2 of 8 | 7 of 8 | 7 of 8 |
| Gabor envelope weight on the ±1 frames (max over 16 units) | 0.24 | 0.022 | 0.013 |
| Feature change when each window is frozen to its centre frame | 0.9% | 5.1% | 3.6% |
| Sintel EPE normal → frozen (60 pairs per pass, 32 iters) | 2.45 → 2.43 | 2.83 → 2.73 | 1.83 → 1.87 |

- The README's last entry for this run (5k: 24.6% feature change) was the high point; it decayed afterwards.
- DoG neuron 7 is the only one with temporal extent left (taps 0.13 / 0.77 / 0.09), and it is nearly silent: drive std 0.21, vs 1.1–3.8 for the others.

### 2. The contrast fix worked, which rules out H4 as the cause of the collapse
- Pre-GELU drive std is 1.1–3.8 at 90k (baseline 30k: 0.000–0.007). 18–53% of the drive exceeds ±1.
- Gabor ON/OFF mixing imbalance is 0.16–0.95, median about 0.8 (baseline 30k: 0.18–0.79, median about 0.5).
- The front end is now nonlinear and signed, and the temporal kernels still collapsed. H1–H3 remain, plus the pipeline issues in section 5.

### 3. No measurable EPE difference between the two runs
- Training loss is within 1–2% in every 5k block up to 30k.
- The validation ordering flips with the subset. On the 200-pair log at 30k, contrast is better (2.19 / 2.82 vs 2.27 / 3.27 clean / final). On the 60-pair-per-pass subset used here it is worse (2.83 vs 2.45, both passes pooled).
- Consecutive checkpoints of the same run swing by 0.1–0.4 EPE (contrast clean: 2.51 → 2.90 → 2.24 at 15k / 20k / 25k). The 200-pair validation cannot resolve the differences this project is looking for.

### 4. Many filter parameters sit on their bounds (step 85k)
| Parameter | At bound | Bound |
|---|---|---|
| DoG σ_t centre | 7 of 8 | lower, 0.30 |
| Gabor σ_t | 16 of 16 within 0.04 | lower, 0.30 |
| Gabor frequency | 10 of 16 | lower, 1/15 = 0.067 |
| Gabor σ | 13 of 16 | upper, 3.5 |
| Gabor speed | 6 of 16 | upper, 4.0 |
| DoG gain | 2 at 4.0, 2 at 1.07–1.10 | both ends of [1, 4] |

- The two Gabor scales merged: all 8 coarse units and 3 fine units are at f ≈ 0.067, the rest at 0.08–0.11.
- With no temporal extent, θ and θ + 180° are the same filter, so the 16 units cover about 5 distinct orientations (units 0, 4, 8 and 12 are all at 0°/180°).
- Speed has no effect once σ_t = 0.3. Its reaching the bound is most likely Adam drift on a near-zero gradient at 20× learning rate. This is an inference, not a measurement.

### 5. Pipeline issues that bias the result against temporal filters (from the code)
- **Eraser augmentation** (`core/utils/augmentor.py`, `eraser_transform`, probability 0.5) paints rectangles on clip frames ref+1 onward (frames 4–7). Window 1 spans frames 0–6, so any filter with temporal extent leaks the rectangle into fmap1, while a per-frame filter keeps fmap1 clean. The incentive is consistent but small. A simulation with the project's own `FlowAugmentor` (2,000 draws) puts a rectangle inside the crop in 38% of samples, covering 2.4% of the crop on average (4.7% within the 17 px reach of the DoG + Gabor stages). That is about 1.8% of all training pixels. *(An earlier version of this note said "half of all training samples"; that overstated it.)*
- **Edge replication** (`--edge_mode replicate`, the default) gives the first 3 and last 3 pairs of every scene repeated frames: 138 of 1,041 pairs, about 13%. In those, the temporal context is fake. `--edge_mode skip` already exists.
- **No held-out data.** "Val EPE" is measured on the training set. After about 290 epochs over 2,082 pairs, 1.53 clean says nothing about generalisation. `sintel_train_val(val_scenes=...)` in `core/datasets.py` already supports a scene hold-out; `train.py` does not expose it.

### 6. Run hygiene
- The run stopped during the 90k validation: the clean pass printed (1.559), the final pass never did. `train.err` holds no error. `90000_customraft-contrast.pth` exists, but there is no final checkpoint, no full 1,041-pair evaluation and no 90k filter snapshot.
- Checkpoints hold only model weights, so the run cannot be resumed on its schedule.
- The README experiment log still says "Running" with the 5k numbers.
- The random-dot benchmark is uninformative as logged. All-pixel EPE at 90k is 0.275 camouflage vs 0.259 visible, dominated by the background. `scripts/live_random_dot.py` already separates interior, boundary and background (30k baseline, scene 0: 0.73 / 3.05 / 0.23), but that is not part of evaluation.

## Recommended changes

### A. Make results trustworthy (do first)
1. **Held-out validation.** Add `--val_scenes` to `train.py`, passed to `fetch_dataloaders` for training and to `validate_sintel(scenes=...)`, which already takes it. Report train-split and held-out EPE separately.
2. **Stable comparison.** Run periodic validation on the full held-out scenes, not 200 evenly spaced pairs, and compare runs on identical pairs.
3. **Log the project's key metric.** At every validation, write to `val_metrics.jsonl`: feature change and EPE change under centre-frame freeze, per-neuron drive std, and the count of filter parameters within 1% of a bound.
4. **Resumable checkpoints.** Save optimizer, scheduler, GradScaler and step in `save_checkpoint`, and add `--resume`. Also log how many optimizer steps the GradScaler skips; the first one is skipped in both runs, per the warning in `train.err`.
5. **Random-dot evaluation.** Move the interior / boundary / background EPE split from `scripts/live_random_dot.py` (`boundary_band`) into `evaluate.py` as `validate_random_dot`, using the `segmentation/` masks the generator writes.
6. **Close out this run.** Run the full evaluation of `90000_customraft-contrast.pth` (`python evaluate.py --model checkpoints/90000_customraft-contrast.pth`) and update the README log: status, 90k numbers, the per-frame finding, H4 ruled out.

### B. Remove the confounds, then re-test (next run)
7. Add `--no_eraser` (sets `eraser_aug_prob=0`) and run with `--edge_mode skip --no_eraser --dog_gain_mode contrast`. 30k steps is enough, about 3 h, because the collapse is decided by 5–10k. If the temporal kernels survive, the earlier collapse was partly an artefact of the pipeline.

### C. Architecture experiments, in priority order
8. **`customraft-magno` (H1).** Give the temporal signal its own path to the GRU, bypassing correlation. This is the only change that gives transient units a reason to exist.
9. **`customraft-fixedbiphasic`.** Raise the σ_t lower bounds or pin a subpopulation, so the cost or benefit of temporal units can be measured when they cannot collapse.
10. **Train on motion-defined data.** Sintel does not reward camouflage. Mix `RandomDot/camouflage` and `aperture` into training (the loader reads them unchanged) and judge by boundary EPE.
11. **Gabor bank.** Initialise θ over 180° when the bank is static, and relieve the binding bounds (a larger kernel, or a second Gabor bank at half resolution) so the two scales do not merge.
12. **Contrast normalisation.** Pool locally instead of over the whole clip, since the global pool makes a region's response depend on the rest of the frame and on crop size. Widen the gain range: 4 of 8 neurons sit on it.
13. **Lower priority.** `customraft-perframe` is now mostly answered by the freeze test, because the trained models already are per-frame. `customraft-dog32` (H5) and `customraft-biolr5` (H6) can wait.

## Files the plan would touch
- `train.py`: `--val_scenes`, `--no_eraser`, `--resume`, full checkpoint state, skipped-step logging.
- `evaluate.py`: freeze test, `validate_random_dot`.
- `core/utils/filter_log.py`: drive statistics and bound-saturation count.
- `kb/README.md`: experiment log update and a row for this note.
- Items 8–12 touch `core/custom_raft.py`, `core/update.py` and `core/layers/*`. Each is its own run and should be designed when it is scheduled.

## Verification once implemented
- Smoke run: `python -u train.py --name smoke-heldout --num_steps 300 --val_freq 150 --val_scenes <2–3 scenes> --edge_mode skip --no_eraser --dog_gain_mode contrast --mixed_precision`. The new columns should appear in `val_metrics.jsonl`, and the train pair count should drop by the held-out scenes.
- Resume: stop the smoke run at 150, restart with `--resume`, and confirm the learning rate continues on the schedule.
- Freeze metric: on `90000_customraft-contrast.pth` it should reproduce about 3.6% feature change.
- `validate_random_dot` on `30000_customraft.pth` should give boundary EPE well above interior EPE, in line with the existing live script.

## How the measurements were made
All on `90000_customraft-contrast.pth`, `30000_customraft-contrast.pth` and `30000_customraft.pth`, in eval mode, fp32.
- **Kernel weight on the centre frame:** each DoG temporal kernel divided by the sum of its absolute taps. Gabor side-frame weight is `exp(−1 / (2 σ_t²))`.
- **Centre-frame freeze:** fmap1 is recomputed from a clip in which all 8 frames equal frame `ref`, fmap2 from a clip in which all equal frame `ref + 1`. Feature change is `‖frozen − normal‖ / ‖normal‖` over both maps, averaged over pairs. EPE is from the full model run on the frozen features, 32 iterations.
- **Sintel subset:** 60 evenly spaced pairs from each of `training/clean` and `training/final`, edge frames replicated.
- **Drive statistics:** pre-GELU DoG response (after normalisation and gain) on every tenth pair of that subset.
- **ON/OFF imbalance:** `Σ|w_ON − w_OFF| / (Σ|w_ON| + Σ|w_OFF|)` per Gabor unit, from `gabor.channel_mix`.
- **Random dot:** `validate_sintel` on `RandomDot/camouflage` and `RandomDot/visible`, 100 pairs per pass, all-pixel EPE.

## Caveats
- The freeze-test EPE uses 120 pairs; differences of a few percent between normal and frozen are within noise.
- No run was repeated with a second seed, so "no difference between the runs" means "none that this validation can resolve".
- The claims in section 5 come from reading the code. Their size is untested until the run in item 7.
