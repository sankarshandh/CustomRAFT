# Random-pattern probe: which neurons respond to camouflaged, hollow and second-order motion?

**Question.** On synthetic random-dot stimuli where no single frame shows the object, which DoG neurons and Gabor
units are most active, which ones respond to the motion itself, and does the model recover the motion?

**Script:** `scripts/probe_random_patterns.py`.
**Data:** `kb/data/random_pattern_probe.json`.
**Figures** (`kb/figures/`):
- `probe_dog_heatmap_<ckpt>.png`
- `probe_maps_<ckpt>.png`

**Checkpoints:**
- `100000_customraft-typed`
- `90000_customraft-contrast`
- `30000_customraft` (baseline)

## Stimuli
- **Textures:** binary random dots, 2 px dots, density 0.5. Each frame is 256 × 384 px and each clip has 8 frames.
- **Object:** a disk of radius 40 px, centred in the reference frame (frame 3).
- **Motion:** speed 4 px/frame in four directions (±x, ±y), with 6 random seeds per condition.
- **Static control:** every case is also rendered with v = 0. Its statistics match the moving case, so any difference is caused by motion alone.

| Case | What moves | Texture optical flow |
|---|---|---|
| **occluded** | A disk carrying its own random texture B moves over a static random background A of the same statistics, like a camouflaged occluder. | v inside the disk, 0 outside |
| **hollow** | Only a 3 px-wide **ring** of texture B moves. The hole and the outside show the static background A. | v on the ring, 0 elsewhere |
| **roi** | Two static patterns A and B. A disk-shaped ROI moves; inside it B is shown, outside it A. The dots themselves never move; only the place where A is swapped for B moves (second-order motion). | 0 everywhere (the ROI moves at v) |

**What is measured:**
- **DoG neuron activation:** ON + OFF output in the window centred on frame 3.
- **Gabor activation:** quadrature energy in the same window.
- **Regions** (defined in frame 3):
  - object (the disk or ring)
  - edge (±4 px around the outer boundary)
  - hole (hollow case only)
  - background (more than 24 px outside the disk)
- **Normalisation:** values are divided by the model's strongest unit on the static texture, so 1.0 means "as active as the most active neuron on a still image".
- **Rise:** activation on the moving stimulus minus the static control, on the same scale.

## Results

### Flow (12 iterations, mean over 24 clips per case)
| Case | Quantity | typed 100k | contrast 90k | baseline 30k |
|---|---|---|---|---|
| occluded | flow inside the disk along v, out of 4 px | 2.61 | 2.43 | 2.52 |
| occluded | EPE inside the disk | 1.46 | 1.66 | 1.62 |
| hollow | flow on the ring along v, out of 4 px | 0.01 | 0.01 | 0.02 |
| hollow | EPE on the ring | 3.99 | 3.99 | 3.98 |
| roi | flow inside the ROI along the ROI's motion | 0.00 | 0.00 | 0.01 |

- **Occluded:** all three models partly recover the camouflaged disk, and about equally well.
- **Hollow:** none of them sees the 3 px ring move. Its flow is 0, so EPE ≈ |v|.
- **ROI:** none of them follows the ROI. They report the texture flow (0), which is correct under the RAFT definition but means the second-order motion is invisible to them.

### Most active units (same in every case and region)
| Model | DoG, most active | Gabor, most active |
|---|---|---|
| typed | 0M, 2M, 6M, 4M (≈ 0.8–1.0); all biphasic units ≈ 0 | 0, 15, 11, 9 |
| contrast | 0, 2, 5, 4 | 11, 9, 15, 4 |
| baseline | **only neuron 6**; 2 is at 0.04 and the rest are below 0.01 | 9, 14, 15, 13 |

These are the static-texture responders. Their activation is the same on the moving stimulus and on the static control, and the same in the object, edge and background regions.

### Motion-driven response (rise over static control)
| Model | occluded: object / edge | hollow: ring / edge | roi: inside / edge |
|---|---|---|---|
| typed **1B** | **+0.29 / +0.24** | **+0.14 / +0.17** | +0.04 / **+0.21** |
| typed, all other neurons (3B, 5B, 7B and all M) | ≤ 0.001 | ≤ 0.001 | ≤ 0.001 |
| contrast, best neuron (5) | +0.003 / +0.002 | +0.002 / +0.001 | +0.000 / +0.001 |
| baseline, best neuron (6) | +0.019 / +0.013 | +0.010 / +0.008 | +0.002 / +0.006 |
| Gabor, best unit (any model) | ≤ +0.015 | ≤ +0.007 | ≤ +0.012 |

## Conclusions (strength: strong for the activations, which are deterministic and consistent across all 24 clips; the stimuli are a single synthetic family)
1. **Only one DoG neuron in any model detects motion: typed neuron 1B.**
   - It is a change detector. It fires wherever pixels change: over the whole occluder, on the ring, and along the ROI boundary (strongly at the edge, weakly inside, because inside the ROI B is static). See `probe_maps_100000_customraft-typed.png`.
   - The other biphasic neurons (3B, 5B, 7B) are silent, as the spatial-DoG cancellation found in the typed-run review predicts.
   - 1B was not switched off in training, only attenuated. Raw response RMS before contrast normalisation:

     | Stimulus | 1B | Monophasic |
     |---|---|---|
     | Sintel | 0.0006 | 0.023–0.028 |
     | Moving dots | 0.0019 | 0.08–0.19 |

     That is below ε = 0.005, so its output scales linearly with its input. Its whole-image drive std is 0.37 on the dots vs 0.11 on Sintel. The +0.29 above is a region average inside the disk, where all of the change is concentrated; over the whole image it is about 9% of a monophasic neuron.
2. **Every other neuron in every model is a per-frame texture encoder.**
   - The monophasic neurons and all neurons of the contrast and baseline models respond the same to a moving and a static random texture (rise ≤ 2% of max).
   - The baseline is also nearly a single-neuron encoder: neuron 6 carries essentially all of the DoG output.
3. **1B's motion signal barely reaches the Gabor stage.**
   - Gabor energy rises by at most 1.5% of max.
   - The flow outputs show the same: the ring and the ROI, which only 1B can see, get zero flow in the typed model too.
   - So the occluded disk is found by matching the per-frame textures in the correlation volume, not by temporal filtering. That is why all three models perform alike.
4. **Implication for the biphasic experiments:** a surviving transient channel exists (1B) and does carry exactly the information a per-frame encoder lacks: the camouflaged ring and the second-order ROI. But the network discards it downstream, because RAFT's texture-flow loss on Sintel never rewards it. Testing whether it can be used would need a readout that bypasses correlation (the planned `customraft-magno`) or training data with second-order or camouflaged motion (`RandomDot/aperture`).
