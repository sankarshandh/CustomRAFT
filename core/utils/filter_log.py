"""Snapshot / visualise the learned DoG and Gabor filter parameters during training.

Every call appends one JSON line (all per-neuron values) to <out_dir>/filters.jsonl, saves a PNG
summary to <out_dir>/filters_<step>.png and, if a TensorBoard writer is given, logs histograms.
The scatter of temporal DC gain vs odd/even ratio shows whether DoG neurons split into
integrators (DC gain ~ 1) and differentiators (DC gain ~ 0 and/or odd ratio high).
Gabor theta is in image coordinates (x right, y down): 90 deg = downward motion.
"""
import json
import math
import os
import os.path as osp

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def filter_state(fnet):
    dog, gab = fnet.dog, fnet.gabor
    dc_gain, odd_ratio = dog.temporal_profile()
    v = lambda t: t.detach().float().cpu().numpy().round(5).tolist()
    d = {
        'sigma_center': v(dog.sigma_center), 'sigma_surround': v(dog.sigma_surround),
        'sigma_t_center': v(dog.sigma_t_center), 'sigma_t_surround': v(dog.sigma_t_surround),
        'k_temporal': v(dog.k_temporal), 'mu_t_center': v(dog.mu_t_center),
        'mu_t_surround': v(dog.mu_t_surround), 'temporal_dc_gain': v(dc_gain),
        'temporal_odd_ratio': v(odd_ratio), 'temporal_odd_fraction': v(dog.temporal_odd_fraction()),
        'gain': v(dog.gain),
    }
    if dog.temporal_separation is not None:  # 'typed' mode
        d['temporal_separation'] = v(dog.temporal_separation)
        d['is_biphasic'] = dog.is_biphasic.cpu().tolist()
    return {
        'dog': d,
        'gabor': {
            'theta_deg': v(torch.rad2deg(gab.theta) % 360), 'frequency': v(gab.frequency), 'sigma': v(gab.sigma),
            'aspect': v(gab.aspect), 'speed': v(gab.speed), 'sigma_t': v(gab.sigma_t),
        },
    }


def _tile(ax, kernels, title, cols=8):
    k = kernels.detach().float().cpu().numpy()
    k = k / (np.abs(k).reshape(len(k), -1).max(1)[:, None, None] + 1e-12)
    rows, ks = math.ceil(len(k) / cols), k.shape[-1]
    grid = np.zeros((rows * (ks + 1), cols * (ks + 1)))
    for i, ker in enumerate(k):
        r, c = divmod(i, cols)
        grid[r * (ks + 1):r * (ks + 1) + ks, c * (ks + 1):c * (ks + 1) + ks] = ker
    ax.imshow(grid, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis('off')


@torch.no_grad()
def log_filters(fnet, step, out_dir, writer=None):
    os.makedirs(out_dir, exist_ok=True)
    state = filter_state(fnet)
    with open(osp.join(out_dir, 'filters.jsonl'), 'a') as f:
        f.write(json.dumps({'step': step, **state}) + '\n')

    dog, gab = fnet.dog, fnet.gabor
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _tile(axes[0, 0], dog.spatial_kernels(), 'DoG spatial kernels')
    tk = dog.temporal_kernels().float().cpu().numpy()
    for i, k in enumerate(tk):
        axes[0, 1].plot(dog.ts.cpu().numpy(), k, marker='o', label=str(i))
    axes[0, 1].axhline(0, color='k', lw=0.5)
    axes[0, 1].set_title('DoG temporal kernels')
    axes[0, 1].set_xlabel('frame offset')
    axes[0, 1].legend(fontsize=6, ncol=2)
    _tile(axes[1, 0], gab.kernels()[0][:, gab.temporal_size // 2], 'Gabor even kernels (centre frame)')
    d = state['dog']
    axes[1, 1].scatter(d['temporal_dc_gain'], d['temporal_odd_fraction'])
    types = ['B' if b else 'M' for b in d['is_biphasic']] if 'is_biphasic' in d else [''] * len(d['temporal_dc_gain'])
    for i, (x, y) in enumerate(zip(d['temporal_dc_gain'], d['temporal_odd_fraction'])):
        axes[1, 1].annotate(f'{i}{types[i]}', (x, y), fontsize=7)
    axes[1, 1].set_xlim(-0.05, 1.05)
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_xlabel('temporal DC gain (1 = integrator, 0 = band-pass)')
    axes[1, 1].set_ylabel('odd energy fraction (1 = first-derivative-like)')
    axes[1, 1].set_title('DoG temporal specialisation')
    fig.suptitle(f'step {step}')
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f'filters_{step:07d}.png'), dpi=80)
    if writer is not None:
        writer.add_figure('filters', fig, step)
        for group, vals in state.items():
            for name, arr in vals.items():
                writer.add_histogram(f'{group}/{name}', np.asarray(arr), step)
    plt.close(fig)
