# Why did the biphasic (parasol-like) temporal filters collapse?

*2026-11-05. Run: [`runs/customraft`](../runs/customraft), analysed at steps 0, 5,000 and 10,000.*
*Numbers come from [`data/temporal_filters.json`](data/temporal_filters.json) and the figures in [`figures/`](figures/), produced by a read-only analysis script that is not part of the repo. They are not re-measured by hand.*

## TL;DR
- Training turned the four biphasic ("differentiator") DoG temporal kernels into roughly symmetric, mostly **low-pass (sustained, integrator-like)** kernels within the first 5,000 steps.
- The strongest measured reason: on Sintel, a neuron whose response depends on **several frames** gives a response that **is not the same** in the two temporal windows the correlation compares once motion is fast. At step 0, for pixels moving more than 16 px/frame:
  - the biphasic neurons' window-to-window consistency was only **0.15–0.30**
  - the narrow, monophasic neurons kept **0.71–0.79**
  - this matters because a fifth of Sintel pixels move that fast
- Two architectural facts make the network's front end behave almost **linearly**, which removes the reason to keep a separate ON/OFF "transient" channel:
  - the ON and OFF channels **cancel exactly at initialisation** inside the Gabor stage
  - the responses are so small that GELU barely bends them
- Direction selectivity, the thing parasol cells feed, **did not disappear**. Part of it moved to the Gabor stage: 3 of the 8 fine-scale Gabor units are clearly direction-selective at step 10,000.
- None of this tests your camouflage hypothesis. Sintel simply doesn't reward transient units through the only path they have to the loss. Section 7 lists the experiments that would separate the causes.

---

## 1. What we observed

The DoG temporal kernel of neuron *n* is `G(μ_c, σ_c) − k_t · G(μ_s, σ_s)`, built from two 1-D Gaussians over 5 frames. "Odd/even ratio" is the energy of its antisymmetric part divided by that of its symmetric part. A large value means a first-derivative-like, biphasic kernel.

| DoG neuron | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| odd/even, step 0 | 0.00 | 0.68 | 0.00 | 1.19 | 0.00 | 2.54 | 0.00 | 16.97 |
| odd/even, step 5k | 0.06 | 0.11 | 0.03 | 0.22 | 0.05 | 0.28 | 0.01 | 0.33 |
| odd/even, step 10k | 0.18 | 0.32 | 0.03 | 0.20 | 0.12 | 0.17 | 0.01 | 1.40* |
| μ_c, step 0 → 10k | 0 → 0.10 | −0.5 → −0.22 | 0 → 0.04 | −0.5 → −0.09 | 0 → 0.10 | −0.5 → −0.09 | 0 → 0.01 | −0.5 → 0.01 |
| μ_s, step 0 → 10k | 0 → −0.07 | 0.5 → −0.06 | 0 → −0.05 | 0.5 → 0.02 | 0 → 0.06 | 0.5 → 0.02 | 0 → −0.03 | 0.5 → 0.05 |
| DC gain (1 − k_t), step 10k | 0.48 | 0.40 | 0.67 | 0.27 | 0.36 | 0.36 | 0.72 | 0.02 |

\* Neuron 7 is **not** a recovered differentiator. Its k_t = 0.98, so the kernel sums to 0.02 and is almost zero everywhere: its temporal-frequency response is flat at about 0.02 in figure CD, middle panel. A near-zero kernel can show any odd/even ratio, so this value is meaningless. Neuron 7 is effectively dead, and its output gain also fell from 1 to 0.22.

**What it looks like** ([`figures/E_dog_trajectories.png`](figures/E_dog_trajectories.png); middle panel of [`figures/CD_static_tuning_dsi.png`](figures/CD_static_tuning_dsi.png)):
- The temporal centres μ_c and μ_s that I offset by ±0.5 frame were pulled back to within about 0.1–0.2 frame of each other. That offset is what made the kernels biphasic.
- At step 0, several neurons had **band-pass** temporal-frequency responses peaking at about 0.2–0.27 cycles/frame. By step 10k almost every neuron is **low-pass** (largest response at 0 cycles/frame, falling with frequency), with DC gains of 0.27–0.72.
- Neuron 6 is close to "use only the current frame", nearly flat in frequency.
- Most of the change happened **before step 5,000**, during learning-rate warm-up, when the filter parameters' learning rate climbed to 8e-3.

Meanwhile the model kept improving. Validation EPE (200 pairs per pass, on the training split as in RAFT) went from 4.42 / 5.11 (clean / final) at 5k to **3.74 / 4.80** at 10k.

---

## 2. How a filter's shape is learned in this model (background)

The retina's filters were shaped by an evolutionary objective with many uses. These filters are shaped by **one** objective, through **one** path:

```
DoG (8 neurons, ON/OFF -> 16 ch) -> Gabor (16 neurons, quadrature) -> CNN head -> 128-d features at 1/8 res
   frame-window [0..6]  -> fmap1 ┐
   frame-window [1..7]  -> fmap2 ┴-> correlation volume -> GRU -> flow -> L1 loss vs ground truth
```

- The DoG's only influence on the loss is through **fmap1 and fmap2**, which the correlation volume compares by inner product.
- A filter shape survives only if it makes the feature of a scene point in window 1 **similar** to the feature of the same point (moved by the flow) in window 2, and **dissimilar** to everything else.
- Nothing rewards a neuron for signalling motion directly. The GRU reads motion only from where the correlation peaks, not from the features themselves.

This is the key difference from the retina. Parasol and magnocellular signals feed motion-energy units (V1 direction-selective cells, then MT) that **read velocity from the transient signal itself**. Here the transient signal would have to earn its place by **improving a matching score**.

---

## 3. Candidate causes and the evidence for each

### H1. The objective rewards stable, distinctive appearance, not change
**Mechanism.** For a pattern translating at velocity **v**, the temporal derivative is `∂I/∂t = −v·∇I`. A differentiator's output therefore:
- is **zero** wherever nothing moves (much of a typical frame)
- **confounds speed with contrast**: a fast faint edge looks like a slow strong edge
- **changes sign and magnitude** whenever the velocity changes

A sustained (integrator) neuron instead reports local appearance, which is what a matcher needs.

**Evidence.**
- At step 0, the transient neurons 6 and 7 had static-to-moving response ratios of **0.42 and 0.35**. On static pixels (< 0.5 px/frame) they gave about a third of their moving-pixel response, so they carried little information there. The integrators 0 and 1 had **1.7 and 2.0**. See [`figures/CD_static_tuning_dsi.png`](figures/CD_static_tuning_dsi.png), left panel.
- By step 10k those two ratios rose to 1.0, i.e. the transient neurons stopped being transient.
- The trained 1/8 feature vectors became strongly **position-specific**. For pixels moving 4–16 px, the similarity between fmap1 and the motion-compensated fmap2 is **0.70**, versus **0.30** without compensation. At step 0, with the untrained head, both values were the same (0.88 and 0.87). The network learned features that pin down *where* things are, which is appearance.

**Strength:** moderate. This is consistent with what happened, but it doesn't by itself show that transient units would *hurt*.

### H2. Frame-rate aliasing: Sintel moves too far per frame for temporal derivatives to mean anything
**Mechanism.** A retinal cell integrates over tens of milliseconds, during which an edge moves a fraction of its receptive field, so a temporal derivative behaves well. Here time is sampled at 24 fps. A filter with peak spatial frequency f cycles/px can only represent speeds below `v_max = 1/(2f)` px/frame. Faster than that, consecutive frames are "different textures", not "the same texture shifted a little", and the derivative turns into flicker.

**Evidence** ([`figures/A_flow_vs_aliasing.png`](figures/A_flow_vs_aliasing.png)):

| \|flow\| per frame (Sintel training, all 1,041 pairs) | > 1 px | > 2 px | > 4 px | > 8 px | > 16 px | > 32 px |
|---|---|---|---|---|---|---|
| fraction of pixels | 79% | 64% | 49% | 36% | 21% | 12% |

- The median speed is 3.9 px/frame, and the 90th percentile is 39 px/frame.
- At step 0 the DoG neurons' `v_max` was 2.3–7.1 px/frame, so **38–61% of pixels** moved faster than a given neuron can represent.
- During training most DoG neurons moved to **coarser** spatial scales (σ_center from 0.7–2.5 up to 0.75–4.3 px), which raised `v_max` to 3.4–8 px for 7 of the 8 neurons.
- Neuron 0 went the opposite way (`v_max` 1.7 px). It is the fine-detail channel.

**Strength:** strong as a description of the data regime. It applies to *any* filtering across frames, not only to biphasic kernels (see H3).

### H3. Fragility: responses that combine several frames stop matching between windows when motion is fast or not uniform
**Mechanism.**
- Window 1 covers clip frames 0–6 and window 2 covers frames 1–7. For pure constant-velocity translation the two responses are exact shifted copies; I verified this earlier with a synthetic texture (cosine similarity 1.000).
- Sintel violates this all the time: acceleration, rotation, zoom, deformation, occlusion, motion blur.
- The more frames a neuron mixes, the more its two window responses differ.
- A biphasic or band-pass kernel is a **difference of two nearly equal terms**, so these violations are amplified relative to the signal.

**Evidence: measured consistency.** This is the cosine between a neuron's response in window 1 and its motion-compensated response in window 2, on non-occluded pixels of 40 Sintel pairs ([`figures/B_temporal_consistency.png`](figures/B_temporal_consistency.png)).

| Step 0, DoG neuron | kernel | 0–1 px | 1–4 px | 4–16 px | > 16 px |
|---|---|---|---|---|---|
| 0, 2, 4, 6 | even, narrow (σ_t 0.5) | 0.84–0.98 | 0.92–0.97 | 0.91–0.93 | **0.71–0.79** |
| 1, 3, 5, 7 | biphasic (σ_t 0.8) | 0.78–0.98 | 0.91–0.95 | 0.79–0.83 | **0.15–0.30** |

- At slow speeds all neurons are consistent. At fast speeds the biphasic neurons lose almost all consistency, while the narrow even neurons keep most of it. That is exactly the property the correlation needs, failing on the 21% of pixels that move fastest.
- **Confound:** at step 0 the biphasic neurons also had **broader** temporal Gaussians (σ_t 0.8 vs 0.5). The step-10k data shows breadth matters on its own. Neuron 5 is no longer biphasic but has the broadest centre (σ_t 1.32) and still scores only **0.24** above 16 px. The narrowest neurons (2, 4, 6; σ_t 0.39–0.57) score **0.78–0.79**.
- So the measured effect is "mixing frames hurts at speed", with biphasic kernels among the worst. It is not uniquely "biphasic is bad".

The final 1/8 features show the same trend after training: compensated similarity **0.94 / 0.81 / 0.70 / 0.54** across the four speed bins at step 10k. Correlation has noticeably weaker evidence for fast motion. Without a per-frame encoder to compare against (planned run `customraft-perframe`), I can't say how much of this drop the temporal front end causes and how much any encoder would show.

**Strength:** strong for "multi-frame responses are inconsistent at high speed". It only partly separates biphasic shape from temporal breadth.

### H4. The front end is almost linear, so a separate transient ON/OFF channel adds nothing
This is a structural finding I didn't anticipate, so it's spelled out in full.

1. **ON and OFF cancel exactly at initialisation.** The Gabor stage mixes the 16 DoG channels (8 ON, 8 OFF) with `channel_mix`, which starts at the same weight (1/16) for every channel. Mathematically, `GELU(r) + GELU(−r) = r·erf(r/√2)`, an **even** function of r. So at step 0 the Gabor stage received **no signed (linear) DoG signal at all**, only something like `0.8·r²`. Leading-edge (+) and trailing-edge (−) responses were added together and lost their sign.
   - This also explains a puzzle in [`figures/CD_static_tuning_dsi.png`](figures/CD_static_tuning_dsi.png) (right panel). At step 0 the 8 fine-scale Gabor units had a direction-selectivity index of **exactly 0**. Squaring a drifting grating doubles its spatial frequency. For the fine units (f = 0.25 cycles/px) that lands on the Nyquist limit of 0.5, where direction can't be represented. The coarse units (f = 0.125) receive a doubled grating at 0.25, which still drifts in the same direction, so they kept a DSI of 0.6–0.76.
2. **After training the cancellation is broken only slightly.** The ON/OFF imbalance in the Gabor's mixing weights, `|w_ON − w_OFF| / (|w_ON| + |w_OFF|)`, is 0.06–0.33 per Gabor unit at step 10k. DoG responses are tiny, so the linear part of what gets through dominates the quadratic part. The DoG → Gabor stage is effectively one **linear** spatiotemporal filter followed by the Gabor energy.
3. **The network pushed further into the linear regime.** The DoG output gains fell from 1.0 to **0.13–0.96** (median about 0.2). The InstanceNorm after the Gabor stage removes overall scale, so the only effect of a smaller gain is to make GELU *more* linear. (This is an interpretation, not a measurement of intent.)

**Consequence.** In a linear cascade the DoG's temporal kernel and the Gabor's temporal kernel simply convolve; the combined kernel is 7 frames long. The network can place temporal tuning in either stage, so the DoG's temporal shape isn't separately "needed". Direction selectivity can come from the drifting Gabor alone. At step 10k, fine-scale Gabor units **1, 2 and 4** have DSI **0.62, 0.68 and 0.55**, at preferred speeds 0.5–2.8 px/frame. The coarse units became essentially static orientation detectors. Their temporal Gaussians shrank to σ_t ≈ 0.33–0.47 frames, so they effectively see one frame, and every coarse DSI is ≤ 0.07. Most of their preferred speeds are near 0. Two (units 10 and 11) still nominally prefer 1.6 and 2.4 px/frame, but with so little temporal extent that preference has no measurable effect.

**Strength:** strong for the facts (cancellation, near-linearity, where the direction selectivity sits). Moderate for the conclusion that this is *why* the DoG became sustained.

### H5. Capacity: 8 DoG neurons all have to serve appearance
In the retina, parasol cells sit alongside a much larger population of sustained midget cells. Here 8 DoG neurons are the **only** route for image information into everything downstream, so each one is under pressure to carry the appearance signal the matcher needs.
**Evidence:** none yet; this is untested. Run `customraft-dog32` would test it.

### H6. Optimisation: a very high learning rate for the filter parameters during warm-up
The filter-shape parameters train at 20× the base learning rate, peaking at 8e-3 at step 5,000. The collapse happened almost entirely within those first 5,000 steps (figure E), and later changes were small. A high learning rate can push parameters quickly into whichever regime gives the first easy gain, and the network may never come back.
**Evidence:** only the timing, which is suggestive but not conclusive. Run `customraft-biolr5` would test it.

### Summary of the evidence

| Cause | Mechanism | Evidence | Support |
|---|---|---|---|
| H1 Objective rewards appearance | Correlation needs stable, distinctive features | Transient units weak on static pixels; trained features position-specific | Moderate |
| H2 Frame-rate aliasing | 24 fps, large motion; v_max ≈ 1/(2f) | 21% of pixels > 16 px/frame; neurons moved to coarser scales | Strong (data regime) |
| H3 Multi-frame fragility | Window responses diverge when motion is fast or non-uniform | Consistency above 16 px: biphasic 0.15–0.30 vs narrow 0.71–0.79 | Strong (confounded with breadth) |
| H4 Near-linear front end | ON/OFF cancellation, tiny responses, DoG and Gabor redundant | Exact cancellation at init; DSI = 0 for fine Gabors at init; gains fell | Strong (facts), moderate (causal) |
| H5 Capacity | 8 neurons must carry appearance | none | Untested |
| H6 Optimisation | High filter learning rate during warm-up | Collapse within the first 5k steps | Weak / untested |

These causes aren't exclusive. My best current reading: **H3 and H2 explain why multi-frame transient responses are bad for matching on Sintel, H1 explains why matching is all that counts, and H4 explains why nothing else in the architecture protects them.**

---

## 4. The retina vs. this model

| | Retina (parasol / magno pathway) | CustomRAFT DoG stage |
|---|---|---|
| Time | Continuous; tens of ms of integration; small displacement within the integration window | 24 fps samples; median 3.9 px and 90th percentile 39 px per frame |
| Readout of the transient signal | Motion-energy units (V1 direction-selective, MT) read velocity directly | Only via a matching score (inner product of two windows) |
| Population | Parasol cells coexist with many sustained midget cells | 8 neurons carry everything |
| ON/OFF | Separate, strongly rectified pathways | GELU near-linear; ON and OFF mixed with nearly equal weights, so their signed parts largely cancel |
| Objective | Many tasks, energy efficiency, robustness | L1 flow error on one dataset |
| Typical stimulus | Smooth, continuous motion | Rendered film: fast camera motion, motion blur, occlusion |

---

## 5. What this does and doesn't mean for your hypothesis

- **It doesn't refute it.** Your hypothesis is about **ambiguity**: camouflaged random-dot regions and motion-defined boundaries, where per-frame appearance can't tell you what belongs together. Sintel is full of texture and nearly free of camouflage, and its large motions are the regime where temporal filters are weakest (H2, H3). The training signal for transient units here is small to begin with.
- **It does say that** the current architecture gives transient units no independent reason to exist (H1, H4). If they're useful, the useful thing they carry, velocity, has no path to the GRU except through matching.
- **It also warns that** at 24 fps with large motions, multi-frame filtering costs feature consistency (H3). Any design that keeps transient units should expect that cost on fast motion.

---

## 6. Caveats
- Only 3 snapshots (0, 5k, 10k) of a 100k-step run. The learning rate decays from 5k onwards, and filters may still reorganise.
- Consistency uses 40 pairs (20 clean, 20 final) and the pre-GELU DoG response. It is a proxy for what the correlation sees, not the correlation itself.
- The "odd/even ratio" is unreliable for near-zero kernels (neuron 7).
- The direction-selectivity index uses drifting gratings at each unit's own preferred frequency and speed, passed through the whole DoG → Gabor path. Units with near-zero preferred speed were tested at 0.25 px/frame.
- There's no per-frame baseline yet, so no absolute statement about whether the spatiotemporal front end helps or hurts EPE.

---

## 7. Experiments that would separate the causes
These are in the [experiment log](README.md#planned-experiments).

| Run | Change | Tests | If the cause is real, expect |
|---|---|---|---|
| customraft-perframe | DoG and Gabor temporal size 1 (RAFT-like per-frame encoding) | Baseline for H2/H3; is the temporal front end worth it? | Per-frame features more consistent above 16 px; EPE comparison decides overall value |
| customraft-gelugain | Initialise `log_gain` so pre-GELU std ≈ 1.5, and initialise ON/OFF mixing unequally | H4 | Real rectification and a signed signal into the Gabor stage; biphasic units more likely to survive |
| customraft-magno | Also feed transient DoG/Gabor channels into the GRU/context, bypassing correlation | H1 | Transient units keep their shape because they're rewarded for carrying velocity |
| customraft-fixedbiphasic | Keep a subpopulation biphasic by construction (minimum μ_c − μ_s separation) | Are biphasic units useful if they can't collapse? | Same or better EPE means useful but not discoverable; worse means costly on Sintel |
| customraft-dog32 | 32 DoG neurons | H5 | Some neurons stay transient once capacity isn't scarce |
| customraft-biolr5 | `--bio_lr_mult 5` | H6 | Slower collapse; whether the end state differs tells whether optimisation speed matters |
| customraft-bandpass | `--temporal_mode bandpass` | Value of learnable temporal shape | Worse EPE than `free` means the network benefits from turning filters into integrators |
| RDK benchmark | Camouflaged random-dot square; spatiotemporal vs per-frame | Your camouflage hypothesis | Spatiotemporal wins at motion boundaries and with fine dots, but only below about v_max |

## How to reproduce the numbers
The figures and `data/temporal_filters.json` were produced by an analysis script kept outside the repo, because no project code was changed. It loads `checkpoints/{5000,10000}_customraft.pth` and the step-0 model, which is deterministic, and evaluates them on 40 Sintel pairs. To refresh this note at later checkpoints, ask for the analysis to be re-run; it is read-only and safe to run during training.
