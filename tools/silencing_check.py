"""How silent is each DoG neuron, and why? Spatial-kernel size vs raw response, on Sintel and on moving random dots.

    python tools/silencing_check.py E003 [--out experiments/E003-typed/analysis/silencing_check.json]

Per DoG neuron:
    sigma_center / sigma_surround   spatial DoG widths (equal widths -> the spatial kernel cancels)
    spatial_l1                      sum |spatial kernel| (0 = cancelled)
    channel_mix_norm                L2 norm of the neuron's RGB mixing weights
    raw_rms_*                       RMS of the DoG response BEFORE contrast normalisation (compare with eps)
    drive_std_*                     std after contrast normalisation and gain (the pre-GELU drive)
    attenuation_vs_mono             median monophasic raw RMS / this neuron's raw RMS (on Sintel)
Sintel: 10 evenly spaced clips per pass (clean + final). Dots: tools/probe_random_patterns.py 'occluded' stimulus
moving at 4 px/frame (seed 7).
"""
import argparse
import json
import os
import os.path as osp
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, 'tools'))

from core.datasets import DEFAULT_SINTEL_ROOT, MpiSintel  # noqa: E402
from core.utils import paths  # noqa: E402
from evaluate import load_model  # noqa: E402
from probe_random_patterns import render  # noqa: E402


@torch.no_grad()
def responses(dog, clip):
    """clip (1, 3, T, H, W) in [-1, 1] -> (raw RMS, drive std) per neuron."""
    w = dog.kernels()[:, None] * dog.channel_mix[:, :, None, None, None]
    r = dog.spatial_size // 2
    resp = F.conv3d(F.pad(clip, (r, r, r, r, 0, 0), mode='replicate'), w).float()
    raw = resp.pow(2).mean((0, 2, 3, 4)).sqrt()
    if dog.gain_mode == 'contrast':
        resp = resp * torch.rsqrt(resp.pow(2).mean((2, 3, 4), keepdim=True) + dog.contrast_eps ** 2)
    drive = (resp * dog.gain[None, :, None, None, None]).std((0, 2, 3, 4))
    return raw.cpu().numpy(), drive.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', help="checkpoint reference, e.g. E003 or E003@100000")
    p.add_argument('--out', default=None, help='default: <experiment>/analysis/silencing/<step>/silencing_check.json')
    a = p.parse_args()
    model = load_model(a.model).eval()
    dog = model.fnet.dog
    rnd = lambda x: np.round(np.asarray(x, dtype=float), 6).tolist()

    raws, drives = [], []
    for dstype in ('clean', 'final'):
        ds = MpiSintel(DEFAULT_SINTEL_ROOT, 'training', dstype, num_frames=model.num_frames, ref_offset=model.ref_offset)
        for i in np.linspace(0, len(ds) - 1, 10).astype(int):
            c = ds[int(i)]['clip'][None].cuda().float()
            r, d = responses(dog, 2 * c / 255 - 1)
            raws.append(r)
            drives.append(d)
    clip, _, _ = render('occluded', (4, 0), 7, SimpleNamespace(h=256, w=384, dot=2, radius=40, ring=3))
    c = torch.from_numpy(clip).cuda()[None, None].expand(1, 3, -1, -1, -1)
    raw_dots, drive_dots = responses(dog, 2 * c / 255 - 1)

    raw_sintel = np.mean(raws, 0)
    mono = (~dog.is_biphasic).cpu().numpy() if dog.temporal_mode == 'typed' else np.ones(len(raw_sintel), bool)
    ref = np.median(raw_sintel[mono])
    out = {
        'checkpoint': paths.describe(a.model)[1], 'contrast_eps': dog.contrast_eps,
        'is_biphasic': dog.is_biphasic.tolist() if dog.temporal_mode == 'typed' else None,
        'sigma_center': rnd(dog.sigma_center.tolist()), 'sigma_surround': rnd(dog.sigma_surround.tolist()),
        'spatial_l1': rnd(dog.spatial_kernels().abs().flatten(1).sum(1).tolist()),
        'channel_mix_norm': rnd(dog.channel_mix.norm(dim=1).tolist()),
        'gain': rnd(dog.gain.tolist()),
        'raw_rms_sintel': rnd(raw_sintel), 'drive_std_sintel': rnd(np.mean(drives, 0)),
        'raw_rms_dots': rnd(raw_dots), 'drive_std_dots': rnd(drive_dots),
        'monophasic_median_raw_rms_sintel': round(float(ref), 6),
        'attenuation_vs_mono': rnd(ref / np.maximum(raw_sintel, 1e-12)),
        'sintel_clips': 20,
    }
    path = a.out or osp.join(paths.analysis_dir(a.model, 'silencing'), 'silencing_check.json')
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    for k in ('sigma_center', 'sigma_surround', 'spatial_l1', 'raw_rms_sintel', 'drive_std_sintel', 'raw_rms_dots',
              'drive_std_dots', 'attenuation_vs_mono'):
        print(f'{k:20s}', out[k])
    print('wrote', path)


if __name__ == '__main__':
    main()
