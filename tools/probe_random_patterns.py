"""Probe CustomRAFT with synthetic random-pattern stimuli: which DoG neurons / Gabor units respond, and where.

All textures are binary random dots with identical statistics, so no single frame shows the object.
    occluded  a disk carrying its own random texture B moves over a static random background A
              (a camouflaged occluder: it covers background at its leading edge and uncovers it behind).
    hollow    the same, but only a thin RING (--ring px wide) of texture B moves; the hole shows the static
              background A.
    roi       two static random patterns A and B; a disk-shaped region of interest moves, inside it B is shown,
              outside A. The dots never move -- only the place where A is replaced by B moves
              (second-order / "drift-balanced" motion). Texture optical flow is 0 everywhere.
Each case is also rendered with v = 0 (static control: same statistics, nothing changes).

The object is centred in the reference frame (clip frame 3, the frame whose flow the model predicts).
Activations are read at the window centred on that frame:
    DoG neuron i = ON_i + OFF_i (= x * erf(x / sqrt 2) >= 0, x = pre-GELU drive), full resolution
    Gabor unit j = quadrature energy sqrt(even_j^2 + odd_j^2), half resolution
and averaged over regions defined in the reference frame: object (disk / ring), edge (+-4 px around the outer
boundary), hole (hollow only), background (> radius + 24 px from the centre).

    python tools/probe_random_patterns.py E003@100000 E002@90000 E001@30000
Results are a cross-experiment study: written to kb/studies/<--study>/data/ and figures/ (data only; the
study.md write-up is filled in by hand).
"""
import argparse
import json
import os
import os.path as osp
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.utils import paths  # noqa: E402
from evaluate import load_model  # noqa: E402

DEV = 'cuda'
CASES = ('occluded', 'hollow', 'roi')
DIRECTIONS = [(4, 0), (0, 4), (-4, 0), (0, -4)]  # (vx, vy) px / frame
T, REF = 8, 3


def texture(rng, h, w, dot):
    g = (rng.random((h // dot + 1, w // dot + 1)) < 0.5).astype(np.float32) * 255
    return np.kron(g, np.ones((dot, dot), np.float32))[:h, :w]


def dist_map(h, w, cy, cx):
    yy, xx = np.mgrid[:h, :w]
    return np.hypot(yy - cy, xx - cx)


def render(case, v, seed, a):
    """-> clip (T, H, W) uint8-valued float, regions {name: bool mask}, gt texture flow (2, H, W)."""
    rng = np.random.default_rng(seed)
    H, W = a.h, a.w
    A, B = texture(rng, H, W, a.dot), texture(rng, H, W, a.dot)
    vx, vy = v
    frames, masks = [], []
    for t in range(T):
        s = t - REF
        d = dist_map(H, W, H / 2 + vy * s, W / 2 + vx * s)
        m = d <= a.radius
        if case == 'hollow':
            m &= d > a.radius - a.ring
        tex = B if case == 'roi' else np.roll(B, (vy * s, vx * s), (0, 1))
        frames.append(np.where(m, tex, A))
        masks.append(m)
    d = dist_map(H, W, H / 2, W / 2)
    reg = {'object': masks[REF], 'edge': np.abs(d - a.radius) <= 4, 'background': d > a.radius + 24}
    if case == 'hollow':
        reg['hole'] = d < a.radius - a.ring - 4
    gt = np.zeros((2, H, W), np.float32)
    if case != 'roi':
        gt[0][masks[REF]], gt[1][masks[REF]] = vx, vy
    return np.stack(frames), reg, gt


@torch.no_grad()
def probe(model, clip):
    x = torch.from_numpy(clip).to(DEV)[None, None].expand(1, 3, -1, -1, -1).contiguous()  # 0..255
    fn = model.fnet
    dog = fn.dog(2 * x / 255 - 1)                      # (1, 2N, 4, H, W)
    gab = fn.gabor(dog)                                # (1, 2M, 2, H/2, W/2)
    n = fn.dog.num_filters
    dog_act = dog[0, :n, REF - 2] + dog[0, n:, REF - 2]  # DoG output t is centred on frame t + 2
    g = gab[0, :, 0]                                   # Gabor output 0 is centred on frame REF
    m = g.shape[0] // 2
    gab_act = torch.sqrt(g[:m] ** 2 + g[m:] ** 2)
    flow = model(x, iters=12, test_mode=True)[1][0]    # (2, H, W)
    return dog_act.float().cpu().numpy(), gab_act.float().cpu().numpy(), flow.float().cpu().numpy()


def region_means(act, reg):
    """act (C, h, w) -> {region: (C,) mean}; masks are nearest-downsampled to act's resolution."""
    f = reg['object'].shape[0] // act.shape[1]
    return {k: act[:, m[::f, ::f][:act.shape[1], :act.shape[2]]].mean(1) for k, m in reg.items()}


def flow_stats(case, flow, gt, reg, v):
    vhat = np.array(v, np.float32) / np.hypot(*v) if any(v) else np.zeros(2, np.float32)
    o, bg = reg['object'], reg['background']
    out = {'obj_mean_flow': flow[:, o].mean(1).round(3).tolist(),
           'obj_along_v': float((flow[:, o] * vhat[:, None]).sum(0).mean()),  # projection on motion direction
           'bg_mag': float(np.hypot(*flow[:, bg]).mean())}
    epe = np.hypot(*(flow - gt))
    out['epe_obj'], out['epe_bg'] = float(epe[o].mean()), float(epe[bg].mean())
    if case == 'hollow':
        out['hole_along_v'] = float((flow[:, reg['hole']] * vhat[:, None]).sum(0).mean())
    return out


def run_model(name, a):
    model = load_model(name).eval()
    dog = model.fnet.dog
    info = {'sigma_center': dog.sigma_center.tolist(), 'sigma_t_center': dog.sigma_t_center.tolist(),
            'temporal_mode': dog.temporal_mode,
            'temporal_odd_fraction': dog.temporal_odd_fraction().tolist(),
            'is_biphasic': dog.is_biphasic.tolist() if dog.temporal_mode == 'typed' else None}
    res, example = {}, {}
    for case in CASES:
        for moving in (True, False):
            key = f'{case}_{"moving" if moving else "static"}'
            dogs, gabs, flows = [], [], []
            for seed in range(a.seeds):
                for v in (DIRECTIONS if moving else [(0, 0)]):
                    clip, reg, gt = render(case, v, 1000 * seed + 7, a)
                    da, ga, fl = probe(model, clip)
                    dogs.append(region_means(da, reg))
                    gabs.append(region_means(ga, reg))
                    flows.append(flow_stats(case, fl, gt, reg, v))
                    if seed == 0 and v == (DIRECTIONS[0] if moving else (0, 0)):
                        example[key] = (clip, da, ga, fl, reg)
            # full precision: the baseline's DoG activations are ~1e-6 (its pre-GELU drive is ~1e-3)
            avg = lambda lst: {k: np.mean([d[k] for d in lst], 0).tolist() for k in lst[0]}
            res[key] = {'dog': avg(dogs), 'gabor': avg(gabs),
                        'flow': {k: np.mean([f[k] for f in flows], 0).round(3).tolist() for k in flows[0]}}
    del model
    torch.cuda.empty_cache()
    return {'filters': info, 'results': res}, example


def type_labels(info):
    if info['is_biphasic'] is not None:
        return [f'{i}{"B" if b else "M"}' for i, b in enumerate(info['is_biphasic'])]
    return [f'{i}{"o" if f > 0.5 else ""}' for i, f in enumerate(info['temporal_odd_fraction'])]


def plot_heatmap(name, out, path):
    """Rows: DoG neurons; columns: case x region; left = moving, right = moving - static."""
    res, labels = out['results'], type_labels(out['filters'])
    cols, mov, diff = [], [], []
    for case in CASES:
        for r in res[f'{case}_moving']['dog']:
            cols.append(f'{case}\n{r}')
            mov.append(res[f'{case}_moving']['dog'][r])
            diff.append(np.subtract(res[f'{case}_moving']['dog'][r], res[f'{case}_static']['dog'][r]))
    mov, diff = np.array(mov).T, np.array(diff).T
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    for ax, m, title, cmap, sym in ((axes[0], mov, 'DoG activation (moving)', 'viridis', False),
                                    (axes[1], diff, 'moving - static', 'RdBu_r', True)):
        lim = np.abs(m).max() + 1e-9
        im = ax.imshow(m, cmap=cmap, aspect='auto', vmin=-lim if sym else 0, vmax=lim)
        ax.set_xticks(range(len(cols)), cols, fontsize=7)
        ax.set_yticks(range(len(labels)), labels)
        for (i, j), val in np.ndenumerate(m):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6,
                    color='w' if (not sym and val < 0.6 * lim) else 'k')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle(f'{name}: DoG neuron activation per region (B = biphasic, M = monophasic, o = odd kernel)')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_maps(name, out, example, path, top=3):
    """Per case: frame 3, |frame 4 - frame 3|, the `top` most motion-driven DoG neurons, top Gabor unit, flow."""
    res, labels = out['results'], type_labels(out['filters'])
    fig, axes = plt.subplots(len(CASES), 4 + top, figsize=(2.6 * (4 + top), 2.4 * len(CASES)))
    for r, case in enumerate(CASES):
        clip, da, ga, fl, reg = example[f'{case}_moving']
        d = np.subtract(res[f'{case}_moving']['dog']['edge'], res[f'{case}_static']['dog']['edge'])
        g = np.subtract(res[f'{case}_moving']['gabor']['edge'], res[f'{case}_static']['gabor']['edge'])
        panels = [(clip[REF], f'{case}: frame {REF}', 'gray'),
                  (np.abs(clip[REF + 1] - clip[REF]), f'|frame {REF + 1} - {REF}|', 'gray')]
        panels += [(da[i], f'DoG {labels[i]} (+{d[i]:.2f})', 'magma') for i in np.argsort(-d)[:top]]
        j = int(np.argmax(g))
        panels += [(ga[j], f'Gabor {j} (+{g[j]:.2f})', 'magma'), (fl[0], 'flow u (v = +4 px)', 'RdBu_r')]
        for c, (img, title, cmap) in enumerate(panels):
            ax = axes[r, c]
            kw = dict(vmin=-6, vmax=6) if cmap == 'RdBu_r' else {}
            ax.imshow(img, cmap=cmap, **kw)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
    fig.suptitle(f'{name}: moving-stimulus maps (DoG/Gabor ranked by edge response above static control)')
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoints', nargs='+', help="checkpoint references, e.g. E003@100000 E002 (or .pth paths)")
    p.add_argument('--h', type=int, default=256)
    p.add_argument('--w', type=int, default=384)
    p.add_argument('--dot', type=int, default=2, help='dot size in px')
    p.add_argument('--radius', type=int, default=40)
    p.add_argument('--ring', type=int, default=3, help='ring width for the hollow case (px)')
    p.add_argument('--seeds', type=int, default=6)
    p.add_argument('--study', default='S001-random-pattern-probe', help='study folder in kb/studies/')
    p.add_argument('--out_json', default=None, help='default: kb/studies/<study>/data/random_pattern_probe.json')
    p.add_argument('--fig_dir', default=None, help='default: kb/studies/<study>/figures')
    a = p.parse_args()
    sdir = osp.join(ROOT, 'kb', 'studies', a.study)
    a.out_json = a.out_json or osp.join(sdir, 'data', 'random_pattern_probe.json')
    a.fig_dir = a.fig_dir or osp.join(sdir, 'figures')
    os.makedirs(osp.dirname(a.out_json), exist_ok=True)
    os.makedirs(a.fig_dir, exist_ok=True)
    allres = {'stimulus': vars(a) | {'directions': DIRECTIONS}}
    for name in a.checkpoints:
        out, example = run_model(name, a)
        allres[name] = out
        plot_heatmap(name, out, osp.join(a.fig_dir, f'probe_dog_heatmap_{name}.png'))
        plot_maps(name, out, example, osp.join(a.fig_dir, f'probe_maps_{name}.png'))
        print(f'===== {name}')
        labels = type_labels(out['filters'])
        for case in CASES:
            mv, st = out['results'][f'{case}_moving'], out['results'][f'{case}_static']
            print(f'  {case}: flow {mv["flow"]}')
            for layer in ('dog', 'gabor'):
                scale = max(np.max(st[layer]['background']), 1e-30)  # strongest unit on the static texture
                lab = labels if layer == 'dog' else [str(j) for j in range(len(st[layer]['background']))]
                for reg in mv[layer]:
                    act = np.asarray(mv[layer][reg]) / scale
                    d = (np.asarray(mv[layer][reg]) - np.asarray(st[layer][reg])) / scale
                    print(f'    {layer:5s} {reg:10s} most active: ' +
                          ' '.join(f'{lab[i]}:{act[i]:.2f}' for i in np.argsort(-act)[:4]) +
                          ' | rise vs static: ' + ' '.join(f'{lab[i]}:{d[i]:+.3f}' for i in np.argsort(-d)[:4]))
    with open(a.out_json, 'w') as f:
        json.dump(allres, f, indent=1)
    print('saved', a.out_json)


if __name__ == '__main__':
    main()
