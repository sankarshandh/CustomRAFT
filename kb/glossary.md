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
