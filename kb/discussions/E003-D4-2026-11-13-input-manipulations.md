# E003-D4: Can changing the training input make the front end use time?

| id | about | started | status | reviewed |
|---|---|---|---|---|
| E003-D4 | [E003-D2](E003-D2-2026-11-12-why-biphasic-silenced.md), [E003-D3](E003-D3-2026-11-12-next-experiments.md), E003-F1, E003-Q2 | 2026-11-13 | open | – |

Follows from [E003-D2](E003-D2-2026-11-12-why-biphasic-silenced.md): correlation is the only encoder-to-flow path, and training removed time from the whole front end. That leaves the dataset as an untested cause, and this thread looks at it.

## Thread
- 2026-11-13 · sankarshan.dh: What would happen if we add temporal noise (independent per frame) to the training data?
- 2026-11-13 · Claude: Predicted (not measured), little effect. Noise does not open a new path to the flow. Biphasic (temporal-difference) units amplify per-frame noise, so they get silenced harder. At most the front end learns sustained temporal averaging, which the 8-frame window caps at about 2.8× less noise and which blurs fast motion. Moderate noise is already tolerated: E003 scores RandomDot camouflage 0.275 clean vs 0.277 with per-frame noise σ=10 (evidence: tests/E003-typed/final/frontend/temporal.json). The augmentor already shares colour jitter across frames to avoid flicker (core/utils/augmentor.py). Suggested cheap check: an EPE-vs-noise sweep on E003, full clip vs pair-only input.
- 2026-11-13 · sankarshan.dh: Asymmetric views: fmap1 from raw frames, fmap2 from the time-shifted frames re-textured with random patterns. The only common feature is the motion edges, so the encoder must be temporal to find them before correlating.
- 2026-11-13 · Claude: Agrees that this makes time necessary, unlike noise. Pairwise appearance matching is impossible, and the biphasic units already respond at moving camouflaged objects (evidence: tests/E003-typed/final/motion/, layer panels, per-tile scale). Risks: interiors only by filling in from edges; test-time mismatch (use as an augmentation with probability p); shared weights; approximate Sintel re-texturing. First step: render random-dot scenes twice from the same layout (visible vs camouflaged, independent dots) and evaluate E003 on asymmetric input. Predicted: about the zero-flow EPE.
- 2026-11-13 · Claude: Result (backlog #2): the prediction was wrong. E003 diverges on asymmetric views: A → B EPE 249.4 px vs zero flow 1.04 (normal A → A 0.255, B → B 0.269). The control A → A′ (same visibility, only the dots differ) fails the same way (253.1), so E003 relies entirely on matching dot texture and uses no motion edges. A trained model must beat zero flow (1.04) and stay stable when the correlation has no peak. Only random dots were tested. (evidence: experiments/E003-typed/analysis/asymmetric_views/final/asymmetric_views.json, 20 scenes × 8 pairs, 32 iters)

## Outcome
Open. Led to [backlog](../backlog.md) #2 (asymmetric-view check on E003, done) and #3 (asymmetric-view training augmentation, now E004). Recorded under "Next" in the [E003 card](../../experiments/E003-typed/card.md). Corrected in [E003-D5](E003-D5-2026-11-13-retinal-hypothesis.md): the asymmetric views lost to a cheaper zero-flow shortcut.
