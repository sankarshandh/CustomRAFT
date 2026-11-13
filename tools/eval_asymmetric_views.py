"""Asymmetric-view check (backlog #2, kb/discussions/E003-D4): can the model correlate two views that share ONLY
the scene layout and motion?

    python tools/eval_asymmetric_views.py E003
    python tools/eval_asymmetric_views.py E003 --scenes 40 --out scratch/asym.json

Each random-dot scene layout (shapes, sizes, positions, velocities: generate_random_dot.make_scene) is rendered twice:
    view A  objects VISIBLE (+60 grey): luminance edges at the object outlines, like a raw frame
    view B  objects CAMOUFLAGED, and every dot (background and objects) drawn independently of view A
So A and B share nothing pixel-wise: only where the surfaces are and how they move. In view B the outlines exist
only as motion boundaries, so finding them needs time.

In the model fmap1 and fmap2 come from ONE 8-frame clip (windows centred on the reference frame and the next one).
Here the feature encoder runs on two clips and takes fmap1 from the first, fmap2 from the second:
    A -> A          normal input, visible objects (reference)
    B -> B          normal input, camouflaged objects (reference)
    A -> B          asymmetric: fmap1 from view A, fmap2 from view B (the test)
    A -> A'         control: view A' = view A with independent dots (same visibility), i.e. only the dots differ
    zero flow       EPE of predicting zero everywhere
The context encoder always sees view A's reference frame. EPE is reported over all pixels, object interior,
motion boundary (within --band px) and background. Prediction (E003-D4): A -> B is close to zero flow, because the
trained encoder matches appearance per frame.

Writes <experiment>/analysis/asymmetric_views/<step>/asymmetric_views.json and example.png.
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
sys.path.insert(0, osp.join(ROOT, 'tools'))

from core.layers.correlation import CorrPyramid  # noqa: E402
from core.utils import paths  # noqa: E402
from core.utils.flow_viz import flow_to_color  # noqa: E402
from core.utils.utils import coords_grid, upflow8  # noqa: E402
from evaluate import load_model  # noqa: E402
from viz_common import GENERATOR, boundary_band, load_generator, scene_args  # noqa: E402

CONDITIONS = ('A -> A', 'B -> B', 'A -> B', "A -> A'")


@torch.no_grad()
def run(model, clip1, clip2, iters):
    """Flow with fmap1 from clip1's first window and fmap2 from clip2's second window (clips: (1,3,T,H,W), 0..255).
    With clip1 is clip2 this is exactly the model's own forward pass (checked in main)."""
    ref = model.ref_offset
    x1 = (2 * clip1 / 255 - 1).contiguous()
    x2 = x1 if clip2 is clip1 else (2 * clip2 / 255 - 1).contiguous()
    f1 = model.fnet(x1)[0]
    f2 = model.fnet(x2)[1]
    corr_fn = CorrPyramid(f1.float(), f2.float(), model.corr_levels, model.corr_radius)
    cnet = model.cnet(x1[:, :, ref])
    net, inp = torch.split(cnet, [model.hidden_dim, model.context_dim], dim=1)
    net, inp = torch.tanh(net), torch.relu(inp)
    b, _, h, w = f1.shape
    c0 = coords_grid(b, h, w, device=x1.device)
    c1 = c0.clone()
    for _ in range(iters):
        corr = corr_fn(c1)
        net, _, d = model.update_block(net, inp, corr, c1 - c0)
        c1 = c1 + d.float()
    return upflow8(c1 - c0)[0].permute(1, 2, 0).cpu().numpy()


def render(gen, desc, args, tex_seed, visible):
    a = scene_args(**{**vars(args), 'object_offset': 60.0 if visible else 0.0})
    frames, labels, flows, _ = gen.render_scene(np.random.default_rng(tex_seed), desc, a)
    return frames.astype(np.float32), labels, flows


def clip_of(model, frames, t):
    idx = np.arange(model.num_frames) + t - model.ref_offset
    return torch.from_numpy(frames[idx]).cuda()[None, None].expand(1, 3, -1, -1, -1).contiguous()


def region_epe(pred, gt, lab, band):
    e = np.linalg.norm(pred - gt, axis=-1)
    bnd = boundary_band(lab, band)
    obj = lab > 0
    regs = {'all': np.ones_like(obj), 'object': obj & ~bnd, 'boundary': bnd, 'background': ~obj & ~bnd}
    return {k: (float(e[m].sum()), int(m.sum())) for k, m in regs.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', help="checkpoint reference, e.g. E003 or E003@40000")
    p.add_argument('--scenes', type=int, default=20)
    p.add_argument('--pairs', type=int, default=8, help='pairs per scene, evenly spaced')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=32)
    p.add_argument('--band', type=int, default=4)
    p.add_argument('--generator', default=GENERATOR)
    p.add_argument('--out', default=None)
    a = p.parse_args()

    torch.manual_seed(0)
    model = load_model(a.model).eval()
    gen = load_generator(a.generator)
    sargs = scene_args(seed=a.seed)
    T, ref = model.num_frames, model.ref_offset
    ts = np.linspace(ref, sargs.num_frames - (T - ref - 1) - 2, a.pairs).astype(int)
    acc = {c: {} for c in CONDITIONS + ('zero flow',)}
    example = None
    for i in range(a.scenes):
        desc = gen.make_scene(np.random.default_rng([a.seed, i]), sargs)
        A, lab, flows = render(gen, desc, sargs, [a.seed, i, 1], visible=True)
        B, labB, _ = render(gen, desc, sargs, [a.seed, i, 2], visible=False)
        A2, _, _ = render(gen, desc, sargs, [a.seed, i, 3], visible=True)
        assert (lab == labB).all(), 'views must share the layout'
        for t in ts:
            cA, cB, cA2 = clip_of(model, A, t), clip_of(model, B, t), clip_of(model, A2, t)
            preds = {'A -> A': run(model, cA, cA, a.iters), 'B -> B': run(model, cB, cB, a.iters),
                     'A -> B': run(model, cA, cB, a.iters), "A -> A'": run(model, cA, cA2, a.iters),
                     'zero flow': np.zeros_like(flows[t])}
            if i == 0 and t == ts[0]:  # the wrapper must reproduce the model's own forward pass
                with torch.no_grad():
                    _, up = model(cA, iters=a.iters, test_mode=True)  # forward() takes 0..255 and normalises
                ref_flow = up[0].permute(1, 2, 0).cpu().numpy()
                dev = float(np.abs(ref_flow - preds['A -> A']).max())
                print(f'check: wrapper vs model forward, max |diff| {dev:.2e} px', flush=True)
                assert dev < 1e-2, dev
            for c, pr in preds.items():
                for k, (s, n) in region_epe(pr, flows[t], lab[t], a.band).items():
                    v = acc[c].setdefault(k, [0.0, 0])
                    v[0] += s
                    v[1] += n
            if example is None and (lab[t] > 0).mean() > 0.03:
                example = (A[t], A[t + 1], B[t], B[t + 1], flows[t], preds, lab[t], i, t)
        print(f'scene {i}: ' + '  '.join(f'{c} {acc[c]["all"][0] / acc[c]["all"][1]:.3f}' for c in acc), flush=True)

    res = {c: {k: round(s / max(n, 1), 4) for k, (s, n) in d.items()} for c, d in acc.items()}
    out = {'checkpoint': paths.describe(a.model)[1], 'scenes': a.scenes, 'pairs_per_scene': a.pairs,
           'seed': a.seed, 'iters': a.iters, 'band': a.band, 'epe': res}
    path = a.out or osp.join(paths.analysis_dir(a.model, 'asymmetric_views'), 'asymmetric_views.json')
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)

    if example is not None:
        a0, a1, b0, b1, gt, preds, lab, i, t = example
        vmax = max(np.abs(gt).max(), 1.0)
        fig, axes = plt.subplots(2, 5, figsize=(20, 6.5))
        tiles = [(a0, 'view A, frame t (visible)'), (a1, 'view A, frame t+1'), (b0, 'view B, frame t (camouflaged)'),
                 (np.abs(b1 - b0), 'view B |t+1 - t|'), (flow_to_color(gt, vmax), 'ground truth')]
        tiles += [(flow_to_color(preds[c], vmax), f'{c}: EPE {np.linalg.norm(preds[c] - gt, axis=-1).mean():.2f}')
                  for c in CONDITIONS] + [(lab > 0, 'object mask')]
        for ax, (img, title) in zip(axes.flat, tiles):
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None, interpolation='nearest')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        fig.suptitle(f'{out["checkpoint"]}: asymmetric views, scene {i}, pair {t}->{t + 1}')
        fig.tight_layout()
        fig.savefig(osp.join(osp.dirname(path), 'example.png'), dpi=70)
        plt.close(fig)

    print('\nEPE (px)            ' + ''.join(f'{k:>12s}' for k in ('all', 'object', 'boundary', 'background')))
    for c, d in res.items():
        print(f'{c:20s}' + ''.join(f'{d[k]:12.3f}' for k in ('all', 'object', 'boundary', 'background')))
    print('wrote', path)


if __name__ == '__main__':
    main()
