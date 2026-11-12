"""Sanity-check CustomRAFT.

    python tools/check_model.py [--root SINTEL_ROOT] [--overfit_steps 200]

1. Parameter breakdown vs RAFT-small (990,162), total within +-10%.
2. Forward / backward on a clip; gradients reach DoG sigmas, k_t, delays and Gabor parameters.
3. Translation test: a texture translating by a constant velocity gives fmap1 / fmap2 whose
   correlation peaks at the true displacement (validates the temporally offset windows).
4. Temporal specialisation at init ('free' mode): integrator-like neurons respond to a static
   clip, band-pass ones barely do; some neurons have odd (first-derivative-like) temporal kernels.
   'bandpass' mode gives ~zero response to a static clip.
5. Overfit one Sintel batch: the loss must fall.
"""
import argparse
import os.path as osp
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from core.custom_raft import CustomRAFT  # noqa: E402
from core.datasets import DEFAULT_SINTEL_ROOT, MpiSintel  # noqa: E402
from core.layers.correlation import correlation_volume  # noqa: E402
from core.layers.spatiotemporal import SpatioTemporalDoG  # noqa: E402
from train import fetch_optimizer, sequence_loss  # noqa: E402

DEV = 'cuda'
RAFT_SMALL = {'fnet': 55264, 'cnet': 58368, 'update_block': 876530, 'total': 990162}


def check_params(model):
    print('--- parameters (CustomRAFT vs RAFT-small)')
    c = lambda m: sum(p.numel() for p in m.parameters())
    ours = {'fnet': c(model.fnet), 'cnet': c(model.cnet), 'update_block': c(model.update_block), 'total': c(model)}
    for k in ours:
        print(f'  {k:13s} {ours[k]:9,d}  vs {RAFT_SMALL[k]:9,d}')
    print(f'    fnet = DoG {c(model.fnet.dog)} + Gabor {c(model.fnet.gabor)} + head {c(model.fnet.head)}')
    assert abs(ours['total'] / RAFT_SMALL['total'] - 1) < 0.10


def check_forward_backward(model):
    print('--- forward / backward')
    clip = torch.rand(2, 3, model.num_frames, 128, 256, device=DEV) * 255
    preds = model(clip, iters=4)
    assert len(preds) == 4 and preds[-1].shape == (2, 2, 128, 256)
    sum(p.abs().mean() for p in preds).backward()
    dog, gab = model.fnet.dog, model.fnet.gabor
    for name, p in [('dog.spatial_center_raw', dog.spatial_center_raw), ('dog.temporal_k_raw', dog.temporal_k_raw),
                    ('dog.temporal_mu_c_raw', dog.temporal_mu_c_raw), ('dog.temporal_mu_s_raw', dog.temporal_mu_s_raw),
                    ('gabor.theta', gab.theta), ('gabor.sigma_raw', gab.sigma_raw), ('gabor.speed_raw', gab.speed_raw)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    print(f'  {model.num_frames}-frame clip -> {len(preds)} x {tuple(preds[-1].shape)}; gradients reach all filter params')
    model.zero_grad()


@torch.no_grad()
def check_translation(model, shift=(16, -8)):
    print('--- translation test (temporally offset windows)')
    rng = np.random.default_rng(0)
    h, w = 256, 384
    tex = torch.from_numpy(rng.random((1, 3, h, w), dtype=np.float32)).to(DEV)
    tex = F.avg_pool2d(F.pad(tex, (2, 2, 2, 2), mode='circular'), 5, stride=1) * 255  # mildly smoothed noise
    # frame k = texture translated by k * shift (circular, so the motion is exact everywhere)
    clip = torch.stack([torch.roll(tex, (k * shift[1], k * shift[0]), dims=(2, 3))[0]
                        for k in range(model.num_frames)], 1)[None]
    fmap1, fmap2 = model.fnet(2 * clip / 255 - 1)
    H, W = fmap1.shape[-2:]
    sx, sy = shift[0] // 8, shift[1] // 8

    # 1) equivariance: f2(p + shift) == f1(p), i.e. rolling f1 by the true displacement reproduces f2
    inner_map = (slice(None), slice(None), slice(4, -4), slice(4, -4))
    cos = lambda a, b: F.cosine_similarity(a[inner_map], b[inner_map], dim=1).mean().item()
    aligned, unaligned = cos(torch.roll(fmap1, (sy, sx), dims=(2, 3)), fmap2), cos(fmap1, fmap2)
    print(f'  feature cosine similarity: shifted by true displacement {aligned:.3f}, unshifted {unaligned:.3f}')
    assert aligned > 0.95 and aligned > unaligned + 0.2

    # 2) matching: argmax of the cosine-normalised correlation (an untrained network has widely varying
    #    feature norms, so the raw inner product would favour large-norm features rather than matches)
    corr = correlation_volume(F.normalize(fmap1, dim=1), F.normalize(fmap2, dim=1))[0]  # (H, W, H, W)
    best = corr.reshape(H, W, -1).argmax(-1)
    dy, dx = best // W - torch.arange(H, device=DEV)[:, None], best % W - torch.arange(W, device=DEV)[None]
    dx, dy = (dx + W // 2) % W - W // 2, (dy + H // 2) % H - H // 2  # unwrap circular shifts
    inner = (slice(2, -2), slice(2, -2))
    hit = ((dx[inner] == shift[0] // 8) & (dy[inner] == shift[1] // 8)).float().mean().item()
    print(f'  true displacement at 1/8 res: ({sx}, {sy}); '
          f'normalised-correlation argmax correct for {hit:.0%} of pixels (untrained network)')
    assert hit > 0.5


@torch.no_grad()
def check_temporal_specialisation():
    print('--- temporal specialisation at init')
    static = torch.rand(1, 3, 1, 64, 64, device=DEV).repeat(1, 1, 5, 1, 1)
    free = SpatioTemporalDoG(3, 8, spatial_size=21, temporal_size=5, temporal_mode='free',
                             temporal_padding='valid', on_off=False).to(DEV)
    dc, odd = free.temporal_profile()
    resp = free(static).abs().mean((0, 2, 3, 4))
    print('  k_t          ', np.round(free.k_temporal.tolist(), 2))
    print('  DC gain      ', np.round(dc.tolist(), 2))
    print('  odd/even     ', np.round(odd.tolist(), 2))
    print('  |static resp|', np.round(resp.tolist(), 4))
    lo, hi = free.k_temporal.argmin(), free.k_temporal.argmax()
    assert resp[lo] > 5 * resp[hi], 'integrator-like neuron should respond to static input'
    assert (odd > 0.5).any(), 'expected some odd (differentiator-like) temporal kernels'

    band = SpatioTemporalDoG(3, 8, spatial_size=21, temporal_size=5, temporal_mode='bandpass',
                             temporal_padding='valid', on_off=False).to(DEV)
    r = band(static).abs().max().item()
    print(f'  bandpass mode: max |static response| {r:.1e}')
    assert r < 1e-4


def check_typed_temporal(draws=1000):
    """'typed' mode: even neurons stay monophasic and odd neurons stay biphasic for ANY parameter values."""
    print(f'--- typed temporal kernels (init + {draws} random parameter draws)')
    dog = SpatioTemporalDoG(3, 8, spatial_size=21, temporal_size=5, temporal_mode='typed',
                            temporal_padding='valid').to(DEV)
    bi = dog.is_biphasic

    def invariants(k):
        mono, biph = k[~bi], k[bi]
        assert (mono >= 0).all(), 'monophasic kernel has a negative tap'
        assert torch.allclose(mono.sum(-1), torch.ones_like(mono.sum(-1)), atol=1e-5), 'monophasic DC gain != 1'
        assert biph.sum(-1).abs().max() < 1e-6, 'biphasic kernel has non-zero DC'
        assert ((biph > 1e-6).any(-1) & (biph < -1e-6).any(-1)).all(), 'biphasic kernel lost a lobe'
        assert torch.allclose(k.abs().sum(-1), torch.ones(len(k), device=k.device), atol=1e-5), 'sum |k| != 1'

    with torch.no_grad():
        k = dog.temporal_kernels()
        frac = dog.temporal_odd_fraction()
        print('  kernels at init:')
        for i, row in enumerate(k.tolist()):
            print(f'    {i} {"B" if bi[i] else "M"}  ' + ' '.join(f'{x:+.3f}' for x in row))
        invariants(k)
        assert (frac[bi] > 0.999).all() and (frac[~bi] < 1e-6).all(), frac
        g = torch.Generator(device='cpu').manual_seed(0)
        for _ in range(draws):
            dog.temporal_center_raw.copy_(torch.randn(8, generator=g) * 6)
            dog.temporal_sep_raw.copy_(torch.randn(8, generator=g) * 6)
            invariants(dog.temporal_kernels())

    dog.temporal_center_raw.data.zero_()
    dog.temporal_sep_raw.data.zero_()
    # random tap weights: a target such as "sum of the first two taps" is constant (0.5) for every L1-normalised odd kernel
    w = torch.randn(dog.temporal_size, generator=torch.Generator().manual_seed(1)).to(DEV)
    (dog.temporal_kernels() * w).sum().backward()
    assert dog.temporal_center_raw.grad.abs().min() > 0, 'sigma_t gets no gradient'
    assert dog.temporal_sep_raw.grad[bi].abs().min() > 0, 'separation d gets no gradient'
    print('  invariants hold; gradients reach sigma_t and d')


def check_overfit(root, steps):
    print(f'--- overfit one Sintel batch ({steps} steps)')
    torch.manual_seed(0)
    model = CustomRAFT().to(DEV).train()
    ds = MpiSintel(root, 'training', 'final', num_frames=model.num_frames, ref_offset=model.ref_offset,
                   aug_params={'crop_size': (368, 768)})
    np.random.seed(0)
    samples = [ds[i] for i in (100, 600)]
    clip = torch.stack([s['clip'] for s in samples]).to(DEV)
    flow = torch.stack([s['flow'] for s in samples]).to(DEV)
    valid = torch.stack([s['valid'] for s in samples]).to(DEV)
    args = SimpleNamespace(lr=4e-4, bio_lr_mult=20.0, wdecay=1e-5, epsilon=1e-8, num_steps=steps)
    opt, sched = fetch_optimizer(args, model)
    sigma0 = model.fnet.dog.sigma_center.detach().clone()
    losses = []
    for i in range(steps):
        opt.zero_grad()
        loss, metrics = sequence_loss(model(clip, iters=12), flow, valid, gamma=0.85)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if i % max(steps // 5, 1) == 0 or i == steps - 1:
            print(f'  step {i:4d}: loss {loss.item():.3f}, epe {metrics["epe"]:.3f}')
    moved = (model.fnet.dog.sigma_center - sigma0).abs().max().item()
    print(f'  max change in DoG sigma_center: {moved:.4f}')
    assert np.mean(losses[-10:]) < 0.7 * np.mean(losses[:10]), 'loss did not decrease'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=DEFAULT_SINTEL_ROOT)
    p.add_argument('--overfit_steps', type=int, default=200)
    args = p.parse_args()
    torch.manual_seed(0)

    model = CustomRAFT().to(DEV)
    check_params(model)
    check_forward_backward(model)
    check_translation(model.eval())
    check_temporal_specialisation()
    check_typed_temporal()
    if args.overfit_steps > 0:
        check_overfit(args.root, args.overfit_steps)
    print('all model checks passed')


if __name__ == '__main__':
    main()
