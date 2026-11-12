# CustomRAFT

RAFT-small with a bio-inspired spatiotemporal feature encoder: a learnable DoG front end followed by oriented Gabor units, trained on MPI-Sintel. The project is organised by experiment.

```
python tools/new_experiment.py E004-name --parent E003 --set key=value   # define an experiment
python -u train.py --exp experiments/E004-name                           # train it
python evaluate.py --model E004 --dataset sintel                         # evaluate it
python tools/check_compat.py                                             # past experiments still reproduce
```

- Rules and layout: [CLAUDE.md](CLAUDE.md)
- Knowledge base: [kb/INDEX.md](kb/INDEX.md)
- Migrated on 2026-11-12 from `D:\IISc\CustomRAFT`, which is left unchanged. The old KB is in [kb/legacy/](kb/legacy/README.md).
