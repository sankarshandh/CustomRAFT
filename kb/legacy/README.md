# CustomRAFT knowledge base

Notes, analyses and the experiment log for CustomRAFT: a RAFT-small variant whose feature encoder is a
learnable spatiotemporal DoG → oriented Gabor front end. The model is trained on MPI-Sintel following
RAFT's split protocol.

## Conventions
- **Notes:** one file per question, named `YYYY-MM-DD_short-topic.md`. Each states the question, the evidence and the conclusions, and says how strong the evidence is.
- **Figures and data:** figures go in `kb/figures/` and the numbers behind them in `kb/data/`. Every number quoted in a note should come from one of these files or from a run directory.
- **Runs:** each run lives in `runs/<name>/`, containing `train.log`, `train_metrics.csv`, `val_metrics.jsonl`, `args.json` and `filters/`. Its checkpoints go in `checkpoints/<step>_<name>.pth`.
- **Log updates:** when a run starts or finishes, update the experiment log below.
- **Validation:** "val EPE" means `evaluate.validate_sintel`, which runs on the Sintel `training/` split as RAFT does and measures fit, not generalisation. During training it uses 200 evenly spaced pairs per pass; the final evaluation uses all 1,041.

## Notes
| Date | Note | Question |
|---|---|---|
| 2026-11-05 | [Why the biphasic temporal filters collapsed](2026-11-05_biphasic-filters-collapse.md) | Why did training flatten the parasol-like biphasic DoG temporal kernels? |
| 2026-11-11 | [`customraft-contrast` review and improvement plan](2026-11-11_contrast-run-review-and-plan.md) | What did the contrast-normalisation run show, and what should change next? (Plan only, not implemented) |
| 2026-11-12 | [Random-pattern probe](2026-11-12_random-pattern-probe.md) | Which neurons respond to camouflaged (occluded), hollow-ring and second-order (ROI) random-dot motion, and is that motion recovered? |

## Experiment log
| Run | Date | Command (from `CustomRAFT/`, `motion` env) | Change vs. baseline | Key result | Status |
|---|---|---|---|---|---|
| [smoke](../runs/smoke) | 2026-11-05 | `python -u train.py --name smoke --num_steps 300 --val_freq 150 --val_max_pairs 50` | First end-to-end test (older data pipeline: CPU colour jitter, float32 clips) | Pipeline works. About 1.0 s/step (data-bound). Val EPE after 300 steps: 11.98 clean / 12.21 final (50 pairs per pass). | Done |
| [cmp_fp32](../runs/cmp_fp32) | 2026-11-05 | `python -u train.py --name cmp_fp32 --num_steps 400 --val_freq 400 --val_max_pairs 50 --sum_freq 50` | GPU colour jitter, uint8 clips, 10 workers | 0.625 s/step (GPU-bound). Full val EPE: 11.31 clean / 11.54 final. | Done |
| [cmp_amp](../runs/cmp_amp) | 2026-11-05 | as above plus `--mixed_precision` | Mixed precision | **0.363 s/step (1.7× faster)**, same loss curve. Full val EPE: 10.31 clean / 10.51 final. Filter parameters move as much as in fp32. | Done; adopted |
| [customraft](../runs/customraft) | 2026-11-05 | `python -u train.py --name customraft --mixed_precision` | Main run: 100k steps, `temporal_mode=free` | Val EPE (clean / final): 5k 4.42 / 5.11; 10k **3.74 / 4.80**. Biphasic DoG kernels collapse by 5k, and most DoG neurons become low-pass (see note). **At 30k the feature encoder is effectively per-frame:** if every frame in each window is replaced by its own centre frame, fmap1 and fmap2 change by under 1% and EPE is unchanged (2.52 vs 2.51 on 40 pairs). At 5k the change was 5%. | Stopped by the user at about 34.1k; last checkpoint `30000_customraft.pth` |
| [customraft-contrast](../runs/customraft-contrast) | 2026-11-05 | `python -u train.py --name customraft-contrast --mixed_precision --dog_gain_mode contrast` | **Only change:** DoG output scaling before the ON/OFF GELU. Divisive contrast normalisation r / sqrt(mean r² + 0.005²), pooled per sample and neuron over the whole clip output, then a gain bounded to [1, 4] (init 2). | Pre-GELU drive std over 12 Sintel-final clips: **≈ 1.9** at init vs 0.000–0.007 at 30k in `customraft`. About 30% of the drive exceeds ±1 (0% before). corr(ON, OFF) **−0.15** vs −1.00 before, so the GELUs now rectify. Translation equivariance still 1.000. 300-step smoke run: train loss 55 → 50, same as baseline; 0.376 s/step. **5k:**
- Val EPE: 4.58 clean / 5.55 final, vs 4.42 / 5.11 for `customraft`. Training loss is within 1–3% of baseline.
- Gains rise towards the upper bound 4 on the sustained neurons and fall towards 1 on the transient neurons 1 and 7.
- Neuron 7 stays biphasic (odd/even 1.24 vs 0.33 in baseline) but is mostly quiet: drive std 0.16, inside the ε floor.
- Temporal σ narrows to about 0.4; Gabor frequency falls by about half.
- Features change by **24.6%** when each window is frozen to its centre frame, vs 6.0% for `customraft` at 5k. | Running |
| [customraft-typed](../runs/customraft-typed) | 2026-11-11 | `python -u train.py --name customraft-typed --mixed_precision --dog_gain_mode contrast --temporal_mode typed` | Same as `customraft-contrast` except the DoG temporal **type is fixed by construction** (`temporal_mode=typed`): even neurons 0/2/4/6 monophasic `G(0, σ)`; odd neurons 1/3/5/7 biphasic `G(−d, σ) − G(+d, σ)`, exactly odd and zero-DC. Only the band is learned: σ ∈ [0.3, 2] and d ∈ [0.25, 1]. Kernels are L1-normalised so a biphasic unit can't be silenced by shrinking. Init: σ 0.5 (mono), σ 0.8 and d 0.5 (bi). Gabor, pipeline, seed and schedule are unchanged. Tests `customraft-fixedbiphasic`: are biphasic units useful or costly when they can't collapse? | 300-step smoke run (`smoke-typed`): 0.37 s/step, loss 48–53 (vs about 50–55 for `smoke-contrast`). The biphasic units stayed exactly odd while narrowing (σ 0.80 → 0.65–0.69, d 0.50 → 0.43–0.45), and their contrast gain fell from 2.0 to 1.6–1.9. **100k:**
- Full val EPE (all 1,041 pairs per pass): **1.51 clean / 2.00 final**; with 200 pairs per pass, 1.51 / 1.92. From 50k on it is within noise (≤ 0.1) of `customraft-contrast` at the same step.
- **The biphasic neurons are silenced through the spatial DoG.** Their temporal kernels stayed exactly odd, but centre σ grew to match surround σ (neuron 3: 4.49 vs 4.50; neuron 5: 4.94 vs 4.94), so the spatial kernel cancels. Spatial-kernel L1 fell to 0.0004 and 0.0002 (0.08 and 0.05 at 5k).
- The raw response then falls below the contrast ε (0.005). Pre-GELU drive std is 0.08 / 0.000 / 0.000 / 0.016 for biphasic neurons vs 2.9–3.9 for monophasic. Biphasic gains sit at about 1.
- **Correction (2026-11-12):** only 3B and 5B are exactly dead (raw output 0 for any input), and 7B is practically dead. **1B is attenuated, not off.**
  - Its raw RMS is 0.0006 on Sintel vs 0.023–0.028 for monophasic neurons, about 30× lower.
  - It still responds to strong temporal change: on high-contrast moving random dots its drive std is 0.37 vs 0.11 on Sintel (see the [random-pattern probe](2026-11-12_random-pattern-probe.md)).
  - Downstream it is ignored: Gabor energy changes by ≤ 1.5%, and the model reports zero flow for motion only 1B detects.
- Monophasic σ_t is at the 0.3 floor and Gabor σ_t is at about 0.3.
- Freeze test: features change by 3.5% and EPE goes from 1.785 to 1.803 (60 pairs per pass), so the encoder is still effectively per-frame.
- Random-dot benchmark: 0.275 camouflage vs 0.256 visible.

**Conclusion:** biphasic units that can't collapse are silenced (three dead, 1B attenuated about 30× and unused downstream); they neither help nor hurt. Data: `kb/data/typed_final_2026-11-12.json`. | Done (2026-11-12 06:18) |

### Planned experiments
Each tests one of the causes discussed in the [biphasic-collapse note](2026-11-05_biphasic-filters-collapse.md#7-experiments-that-would-separate-the-causes).

| Proposed run | Change | Tests | Status |
|---|---|---|---|
| customraft-bandpass | `--temporal_mode bandpass` (zero-DC, even temporal kernels only) | Whether learnable temporal shape helps at all | Planned |
| customraft-perframe | DoG and Gabor temporal size 1 (each frame encoded separately, like RAFT) | Whether any spatiotemporal front end helps vs a per-frame encoder | Planned |
| customraft-dog32 | 32 DoG neurons instead of 8 | H5 (capacity) | Planned |
| customraft-magno | Transient (biphasic) channels also fed into the GRU/context | H1 (readout only via correlation) | Planned |
| customraft-fixedbiphasic | A subpopulation kept biphasic by construction | Whether biphasic units are useful when not allowed to collapse | Done as [`customraft-typed`](../runs/customraft-typed): silenced through the spatial DoG (centre σ = surround σ). A follow-up needs a minimum centre/surround σ gap or a fixed per-neuron output norm. |
| customraft-gelugain | Superseded by `customraft-contrast` (see the experiment log): re-initialising the gain alone would not stick, because the learned gain fell from 1.0 to 0.02–0.47 by 30k | H4/H7 | Superseded |
| customraft-biolr5 | `--bio_lr_mult 5` | H6 (optimisation speed) | Planned |
| RDK benchmark | Random-dot kinematogram with a camouflaged moving square, spatiotemporal vs per-frame | The user's camouflage/ambiguity hypothesis | Datasets generated with `Motion Datasets/generate_random_dot.py`: `RandomDot/camouflage` and `RandomDot/visible` (identical scenes, objects +60 grey levels), 760 pairs per pass. 30k checkpoint, 150 pairs per pass, all-pixel EPE: 0.29 camouflage vs 0.28 visible. Boundary EPE not measured yet. |
