"""Sanity-check the standalone layers in core/layers.

    python tools/check_layers.py [--root SINTEL_ROOT] [--out OUT_DIR]

1. SpatioTemporalDoG : band-pass (no response to constant input), gradients reach every sigma,
                       one optimiser step actually moves the sigmas.
2. OrientationTunedLayer : drifting gratings excite the neuron tuned to their direction more
                       than the one tuned to the opposite direction; gradients reach theta/f/sigma.
3. CNNEncoder        : output shapes for 4D and 5D inputs.
4. Correlation       : matches a brute-force loop; pyramid lookup at zero flow returns the diagonal.
5. Runs all four on a real Sintel pair and saves kernel / response images.
"""
import argparse
import math
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from core.datasets import DEFAULT_SINTEL_ROOT, MpiSintel  # noqa: E402
from core.layers import (CNNEncoder, CorrelationLayer, CorrPyramid,  # noqa: E402
                         OrientationTunedLayer, SpatioTemporalDoG)
from core.utils.utils import coords_grid  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def tile(kernels, scale=8, cols=8):
    """(N, K, K) -> uint8 grid image, each kernel normalised to [-max, max] -> [0, 255]."""
    k = kernels.detach().cpu().numpy()
    k = k / (np.abs(k).reshape(len(k), -1).max(1)[:, None, None] + 1e-12)
    k = ((k + 1) * 127.5).astype(np.uint8)
    rows = math.ceil(len(k) / cols)
    ks = k.shape[-1]
    grid = np.full((rows * (ks + 1), cols * (ks + 1)), 127, np.uint8)
    for i, ker in enumerate(k):
        r, c = divmod(i, cols)
        grid[r * (ks + 1):r * (ks + 1) + ks, c * (ks + 1):c * (ks + 1) + ks] = ker
    return Image.fromarray(grid).resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)


def to_img(a):
    a = a.detach().float().cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-12)
    return Image.fromarray((a * 255).astype(np.uint8))


def check_dog():
    print('--- SpatioTemporalDoG')
    dog = SpatioTemporalDoG(in_channels=3, num_filters=8, spatial_size=15, temporal_size=3).to(DEV)
    print(dog)
    print('  sigma_center  ', np.round(dog.sigma_center.tolist(), 2))
    print('  sigma_surround', np.round(dog.sigma_surround.tolist(), 2))
    print('  sigma_t c / s ', np.round(dog.sigma_t_center.tolist()[0], 2), np.round(dog.sigma_t_surround.tolist()[0], 2))

    const = torch.full((2, 3, 4, 48, 48), 0.7, device=DEV)
    out = dog(const)
    assert out.shape == (2, 16, 4, 48, 48), out.shape
    print(f'  constant input -> max |response| {out.abs().max():.2e} (band-pass => ~0)')
    assert out.abs().max() < 1e-4
    assert dog(torch.rand(2, 3, 48, 48, device=DEV)).shape == (2, 16, 48, 48)

    clip = torch.rand(2, 3, 4, 48, 48, device=DEV)
    target = torch.rand(2, 16, 4, 48, 48, device=DEV)
    before = {n: p.detach().clone() for n, p in [('sc', dog.sigma_center), ('ss', dog.sigma_surround),
                                                 ('tc', dog.sigma_t_center), ('ts', dog.sigma_t_surround)]}
    loss = (dog(clip) - target).pow(2).mean()
    loss.backward()
    for name in ('spatial_center_raw', 'spatial_surround_raw', 'temporal_center_raw', 'temporal_surround_raw'):
        g = getattr(dog, name).grad
        assert g is not None and g.abs().sum() > 0, name
        print(f'  grad {name:22s} |g|={g.abs().mean():.2e}')
    torch.optim.SGD(dog.parameters(), lr=10.0).step()
    after = {'sc': dog.sigma_center, 'ss': dog.sigma_surround, 'tc': dog.sigma_t_center, 'ts': dog.sigma_t_surround}
    for k in before:
        delta = (after[k] - before[k]).abs().max().item()
        assert delta > 0, k
        print(f'  after one step, max change in {k}: {delta:.3e}')
    assert (dog.sigma_surround > dog.sigma_center).all()
    return dog


def check_orientation():
    print('--- OrientationTunedLayer')
    layer = OrientationTunedLayer(in_channels=1, num_orientations=8, num_scales=1, kernel_size=15,
                                  temporal_size=5, base_frequency=0.125, init_speed=1.0).to(DEV)
    print(layer)
    t, y, x = torch.meshgrid(torch.arange(9.), torch.arange(64.), torch.arange(64.), indexing='ij')
    for k in range(8):
        phi = k * 2 * math.pi / 8
        grating = torch.cos(2 * math.pi * 0.125 * (x * math.cos(phi) + y * math.sin(phi) - 1.0 * t))
        resp = layer(grating[None, None].to(DEV))  # (1, 8, T, H, W)
        r = resp[0, :, 2:-2, 16:-16, 16:-16].mean((1, 2, 3))
        pref, null = r[k].item(), r[(k + 4) % 8].item()
        print(f'  grating moving at {math.degrees(phi):5.1f} deg: best neuron {r.argmax().item()} (expect {k}), '
              f'direction index {(pref - null) / (pref + null):.2f}')
        assert r.argmax().item() == k

    static = OrientationTunedLayer(in_channels=3, temporal_size=1).to(DEV)
    assert static(torch.rand(2, 3, 32, 32, device=DEV)).shape == (2, 16, 32, 32)
    strided = OrientationTunedLayer(in_channels=3, stride=2, mode='quadrature').to(DEV)
    assert strided(torch.rand(2, 3, 3, 32, 32, device=DEV)).shape == (2, 32, 3, 16, 16)

    layer(torch.rand(1, 1, 5, 32, 32, device=DEV)).mean().backward()
    for name in ('theta', 'freq_raw', 'sigma_raw', 'aspect_raw', 'speed_raw', 'sigma_t_raw', 'channel_mix'):
        g = getattr(layer, name).grad
        assert g is not None and g.abs().sum() > 0, name
    print('  gradients reach theta, frequency, sigma, aspect, speed, sigma_t, channel_mix')
    return layer


def check_encoder():
    print('--- CNNEncoder')
    enc = CNNEncoder(in_channels=16, output_dim=256).to(DEV)
    print(f'  params: {sum(p.numel() for p in enc.parameters()) / 1e6:.2f}M')
    assert enc(torch.rand(2, 16, 128, 256, device=DEV)).shape == (2, 256, 16, 32)
    assert enc(torch.rand(2, 16, 2, 128, 256, device=DEV)).shape == (2, 256, 2, 16, 32)
    print('  (2,16,128,256) -> (2,256,16,32); 5D input keeps T')


def check_correlation():
    print('--- Correlation')
    f1, f2 = torch.randn(2, 32, 6, 7, device=DEV), torch.randn(2, 32, 5, 9, device=DEV)
    corr = CorrelationLayer()(f1, f2)
    assert corr.shape == (2, 6, 7, 5, 9)
    brute = torch.zeros_like(corr)
    for i in range(6):
        for j in range(7):
            for k in range(5):
                for l in range(9):
                    brute[:, i, j, k, l] = (f1[:, :, i, j] * f2[:, :, k, l]).sum(1) / math.sqrt(32)
    err = (corr - brute).abs().max().item()
    print(f'  matches brute force: max err {err:.1e}')
    assert err < 1e-4

    f1, f2 = torch.randn(2, 64, 16, 24, device=DEV), torch.randn(2, 64, 16, 24, device=DEV)
    pyr = CorrPyramid(f1, f2, num_levels=4, radius=3)
    look = pyr(coords_grid(2, 16, 24, device=DEV))
    assert look.shape == (2, 4 * 49, 16, 24)
    diag = (f1 * f2).sum(1) / 8.0
    err = (look[:, 49 // 2] - diag).abs().max().item()
    print(f'  pyramid lookup {tuple(look.shape)}; centre of window at zero flow == diagonal (err {err:.1e})')
    assert err < 1e-4


@torch.no_grad()
def run_on_sintel(root, out_dir, dog, orient):
    print('--- Sintel pair through all layers')
    s = MpiSintel(root, 'training', 'final')[0]
    clip = torch.stack([s['img1'], s['img2']], 1)[None].to(DEV) / 255.0  # (1, 3, 2, 436, 1024)

    dog = SpatioTemporalDoG(3, 8, spatial_size=15, temporal_size=3).to(DEV)
    resp = dog(clip)
    print(f'  DoG: {tuple(clip.shape)} -> {tuple(resp.shape)}')
    to_img(resp[0, 0, 1]).save(osp.join(out_dir, 'dog_on_fine.png'))
    to_img(resp[0, 8, 1]).save(osp.join(out_dir, 'dog_off_fine.png'))
    to_img(resp[0, 7, 1]).save(osp.join(out_dir, 'dog_on_coarse.png'))

    enc = CNNEncoder(in_channels=resp.shape[1]).to(DEV).eval()
    feats = enc(resp)
    print(f'  encoder: -> {tuple(feats.shape)}')

    ori = OrientationTunedLayer(in_channels=feats.shape[1], num_orientations=8, num_scales=1,
                                kernel_size=7, temporal_size=1, base_frequency=0.25).to(DEV)
    ofeat = ori(feats)
    print(f'  orientation (on encoder features): -> {tuple(ofeat.shape)}')
    on_frames = OrientationTunedLayer(in_channels=3, temporal_size=1).to(DEV)(clip[:, :, 0])
    to_img(on_frames[0, 0]).save(osp.join(out_dir, 'gabor_0deg.png'))
    to_img(on_frames[0, 4]).save(osp.join(out_dir, 'gabor_90deg.png'))

    corr = CorrelationLayer()(feats[:, :, 0], feats[:, :, 1])
    print(f'  correlation volume: {tuple(corr.shape)} ({corr.numel() * 4 / 2**20:.0f} MB)')
    pyr = CorrPyramid(feats[:, :, 0], feats[:, :, 1])
    h, w = feats.shape[-2:]
    print(f'  pyramid lookup: -> {tuple(pyr(coords_grid(1, h, w, device=DEV)).shape)}')

    tile(dog.spatial_kernels()).save(osp.join(out_dir, 'kernels_dog_spatial.png'))
    tile(orient.kernels()[0][:, orient.temporal_size // 2]).save(osp.join(out_dir, 'kernels_gabor_even.png'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=DEFAULT_SINTEL_ROOT)
    p.add_argument('--out', default=osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'scratch', 'check_out'))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    print('device:', DEV)

    dog = check_dog()
    orient = check_orientation()
    check_encoder()
    check_correlation()
    run_on_sintel(args.root, args.out, dog, orient)
    print(f'all checks passed; images in {args.out}')


if __name__ == '__main__':
    main()
