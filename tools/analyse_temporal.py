"""Is the feature encoder still spatiotemporal? Temporal kernels, DoG drive, Gabor input mix and the freeze test.

    python tools/analyse_temporal.py E003 E002@90000

Freeze test: every frame of each encoder window is replaced by the window's centre frame; reports the relative
feature change and EPE (normal vs frozen) on 60 evenly spaced Sintel pairs per pass. Also RandomDot EPE.
Writes <experiment>/analysis/temporal/<step>/temporal.json for each checkpoint
(--out_dir DIR: DIR/temporal.json instead; only with a single checkpoint).
"""
import argparse
import json
import math
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.datasets import DEFAULT_SINTEL_ROOT, MpiSintel  # noqa: E402
from core.layers.correlation import CorrPyramid  # noqa: E402
from core.utils import paths  # noqa: E402
from core.utils.utils import InputPadder, coords_grid, upflow8  # noqa: E402
from evaluate import load_model, validate_sintel  # noqa: E402

DEV = 'cuda'


@torch.no_grad()
def run(model, clip, mode='normal', iters=32):
    """mode: normal | freeze (each window replaced by its centre frame) | shuffle (non-centre frames of each
    window replaced by frames of the *other* window's centre, i.e. wrong temporal context)."""
    ref = model.ref_offset
    x = (2 * (clip.float() / 255.0) - 1.0).contiguous()
    if mode == 'normal':
        f1, f2 = model.fnet(x)
    else:
        a = x[:, :, ref:ref + 1].expand(-1, -1, x.shape[2], -1, -1).contiguous()
        b = x[:, :, ref + 1:ref + 2].expand(-1, -1, x.shape[2], -1, -1).contiguous()
        f1, f2 = model.fnet(a)[0], model.fnet(b)[0]
    corr_fn = CorrPyramid(f1.float(), f2.float(), model.corr_levels, model.corr_radius)
    cnet = model.cnet(x[:, :, ref])
    net, inp = torch.split(cnet, [model.hidden_dim, model.context_dim], dim=1)
    net, inp = torch.tanh(net), torch.relu(inp)
    b_, _, h, w = f1.shape
    c0 = coords_grid(b_, h, w, device=x.device)
    c1 = c0.clone()
    for _ in range(iters):
        corr = corr_fn(c1)
        net, _, d = model.update_block(net, inp, corr, c1 - c0)
        c1 = c1 + d.float()
    return upflow8(c1 - c0), f1, f2


def temporal_summary(model):
    dog, gab = model.fnet.dog, model.fnet.gabor
    tk = dog.temporal_kernels().detach().cpu().numpy()
    tk_n = tk / np.abs(tk).sum(1, keepdims=True)
    st = gab.sigma_t.detach().cpu().numpy()
    side = np.exp(-1 / (2 * st ** 2))  # Gabor envelope weight at t = +-1 relative to the centre frame
    return {'dog_taps_abs_normalised': tk_n.round(3).tolist(),
            'dog_offcentre_mass': (1 - np.abs(tk_n[:, tk.shape[1] // 2])).round(4).tolist(),
            'gabor_side_frame_weight': side.round(4).tolist()}


@torch.no_grad()
def drive_stats(model, clips):
    dog = model.fnet.dog
    out = []
    for clip in clips:
        x = (2 * (clip.float() / 255.0) - 1.0).contiguous()
        weight = dog.kernels()[:, None] * dog.channel_mix[:, :, None, None, None]
        r = dog.spatial_size // 2
        xp = F.pad(x, (r, r, r, r, 0, 0), mode='replicate')
        resp = F.conv3d(xp, weight)
        if dog.gain_mode == 'contrast':
            ms = resp.pow(2).mean(dim=(2, 3, 4), keepdim=True)
            resp = resp * torch.rsqrt(ms + dog.contrast_eps ** 2)
        resp = resp * dog.gain[None, :, None, None, None]
        out.append(torch.stack([resp.std(dim=(0, 2, 3, 4)), (resp.abs() > 1).float().mean(dim=(0, 2, 3, 4))]))
    s = torch.stack(out).mean(0).cpu().numpy()
    return {'drive_std': s[0].round(3).tolist(), 'frac_abs_gt1': s[1].round(3).tolist()}


def mix_stats(model):
    m = model.fnet.gabor.channel_mix.detach().cpu()  # (16 gabor, 16 dog ch: 8 ON, 8 OFF)
    n = m.shape[1] // 2
    on, off = m[:, :n], m[:, n:]
    imb = ((on - off).abs().sum(1) / (on.abs().sum(1) + off.abs().sum(1))).numpy()
    usage = m.abs().sum(0).view(2, n).sum(0)
    usage = (usage / usage.sum()).numpy()
    return {'gabor_on_off_imbalance': imb.round(3).tolist(), 'dog_neuron_share_of_gabor_input': usage.round(3).tolist()}


def sintel_subset(model, n_per_pass):
    items = []
    for d in ('clean', 'final'):
        ds = MpiSintel(DEFAULT_SINTEL_ROOT, 'training', d, num_frames=model.num_frames, ref_offset=model.ref_offset)
        for i in np.linspace(0, len(ds) - 1, n_per_pass).astype(int):
            items.append(ds[int(i)])
    return items


@torch.no_grad()
def freeze_test(model, items):
    bins = [(0, 1), (1, 4), (4, 16), (16, 1e9)]
    acc = {m: {'all': [], 'noc': [], 'occ': [], **{f'{lo}-{hi}': [] for lo, hi in bins}} for m in ('normal', 'freeze')}
    rel = []
    clips = []
    for k, s in enumerate(items):
        clip = s['clip'][None].to(DEV).float()
        padder = InputPadder(clip.shape)
        (clip,) = padder.pad(clip)
        if k % 10 == 0:
            clips.append(clip)
        gt, occ = s['flow'], s['occ'] > 0
        mag = gt.norm(dim=0)
        feats = {}
        for m in ('normal', 'freeze'):
            flow, f1, f2 = run(model, clip, m)
            feats[m] = torch.cat([f1, f2])
            epe = (padder.unpad(flow[0]).cpu() - gt).norm(dim=0)
            acc[m]['all'].append(epe.flatten())
            acc[m]['noc'].append(epe[~occ])
            acc[m]['occ'].append(epe[occ])
            for lo, hi in bins:
                acc[m][f'{lo}-{hi}'].append(epe[(mag >= lo) & (mag < hi)])
        rel.append(((feats['freeze'] - feats['normal']).norm() / feats['normal'].norm()).item())
    res = {m: {k: round(torch.cat(v).mean().item(), 3) for k, v in d.items()} for m, d in acc.items()}
    res['feature_rel_change_mean'] = round(float(np.mean(rel)), 4)
    res['feature_rel_change_max'] = round(float(np.max(rel)), 4)
    return res, clips


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--out_dir', default=None, help='write DIR/temporal.json (single checkpoint only)')
    ap.add_argument('--no_rdk', action='store_true', help='skip the RandomDot camouflage/visible EPE '
                    '(tools/pipeline.py measures random patterns with eval_random_patterns.py instead)')
    a = ap.parse_args()
    if a.out_dir and len(a.checkpoints) != 1:
        ap.error('--out_dir needs exactly one checkpoint')
    torch.manual_seed(0)
    ckpts = a.checkpoints
    for name in ckpts:
        model = load_model(name).eval()
        r = {'temporal': temporal_summary(model), 'mix': mix_stats(model)}
        items = sintel_subset(model, 60)
        r['sintel_freeze'], clips = freeze_test(model, items)
        r['drive'] = drive_stats(model, clips)
        for rd in () if a.no_rdk else ('camouflage', 'visible'):
            v = validate_sintel(model, rf'D:\IISc\Motion Datasets\RandomDot\{rd}', max_pairs=100, num_workers=0)
            r[f'rdk_{rd}'] = {k: round(v[k], 3) for k in ('clean', 'clean_noc', 'clean_occ', 'final', 'final_noc', 'final_occ')}
        print('=====', name)
        print(json.dumps(r, indent=1))
        out_dir = a.out_dir or paths.analysis_dir(name, 'temporal')
        os.makedirs(out_dir, exist_ok=True)
        with open(osp.join(out_dir, 'temporal.json'), 'w') as f:
            json.dump({'checkpoint': paths.describe(name)[1], **r}, f, indent=1)
        print('wrote', out_dir)
        del model
        torch.cuda.empty_cache()
