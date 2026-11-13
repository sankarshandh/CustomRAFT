# Glossary

Define each term once here; cards, findings and studies link to it instead of re-explaining.

| term | one-line definition | where it is measured |
|---|---|---|
| val EPE | End-point error on the Sintel training split (RAFT protocol): 200 pairs per pass during training, all 1,041 at the end. It measures fit, not generalisation. | `runs/<run>/val_metrics.jsonl` |
| typed mode | `temporal_mode=typed`: each DoG neuron's temporal shape (monophasic or biphasic) is fixed by construction; only its band is learned. | `core/layers/spatiotemporal.py` |
| odd fraction | Share of a temporal kernel's energy in its odd (antisymmetric) part: 1 = pure biphasic / derivative-like, 0 = symmetric. | `runs/<run>/filters/filters.jsonl` |
| drive std | Std of a DoG neuron's output before the ON/OFF GELU (after contrast normalisation and gain). About 0 means the neuron is silent. | `tools/analyse_temporal.py`, `tools/silencing_check.py` |
| ε floor | Contrast normalisation divides by sqrt(mean r² + ε²) with ε = 0.005, so responses well below ε are not amplified. | `core/layers/spatiotemporal.py` |
| freeze test | Replace every frame of each encoder window with its centre frame. A feature change near 0 means the encoder works frame by frame. | `tools/analyse_temporal.py` |
| pair-only input | Every clip frame except the pair (t, t+1) is replaced by the nearer pair frame. If EPE equals full-clip EPE, the extra frames are not used. | `tools/render_motion.py`, `tools/live_random_dot.py` (key p) |
| random-pattern cases | Random-dot test scenes with identical dot statistics: **occluded** (a textured shape moves over the background), **hollow** (only a ring of texture moves; the background shows inside), **roi** (a moving window: pattern 2 inside, pattern 1 outside, pattern 2 static), **visible** (control: objects +60 grey). | `D:\IISc\Motion Datasets\RandomDot\<case>`, `tools/eval_random_patterns.py` |
| texture flow / shape flow | Two ground truths for random patterns: texture flow is how the dots move (what RAFT is trained to report; 0 inside a static ROI), shape flow is how the outline moves. They are the same for occluded and visible. | `flow/`, `flow_shape/` in the RandomDot datasets |
| asymmetric views | fmap1 and fmap2 computed from two different renderings of one scene: view A with visible objects, view B camouflaged with independent dots. They share only layout and motion, so outlines in B exist only as motion boundaries. | `tools/eval_asymmetric_views.py`, `core/datasets.py` (RandomDotAsymmetric) |
| asym_prob | `train.py --asym_prob p`: with probability p a training step uses an asymmetric-view random-dot batch instead of a Sintel batch; which view feeds fmap1 is random. 0 = original training. | `train.py`, `config.json` |
| zero-flow EPE | EPE of predicting zero flow everywhere: the floor a model must beat to show it found any correspondence. | `tools/eval_asymmetric_views.py` |
