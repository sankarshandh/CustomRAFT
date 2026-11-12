"""Render what a checkpoint does on moving random-dot scenes, frame by frame, as videos and frame sequences.

    python tools/render_motion.py E003                 # all scenes in configs/pipeline.json -> tests/E003-typed/final/motion/
    python tools/render_motion.py E003@40000 --light   # first scene only, fewer pairs, no .mp4
    python tools/render_motion.py E003 --out_dir scratch/motion

The scenes, seeds and pair counts come from configs/pipeline.json ("motion"), so every experiment sees the same
input. Each scene is one random-pattern case: occluded, hollow, roi, or visible (control); see
tools/eval_random_patterns.py. For hollow and roi the flow panel gets a third row: EPE against the motion of the
whole shape (the dots inside a ROI are static, so texture flow and shape motion differ). For each scene, <out_dir>/<scene name>/ gets
    flow/0000.jpg ...    the live_random_dot.py panel: frame t, t+1, |difference|; GT flow, predicted flow, EPE map
    layers/0000.jpg ...  the spatiotemporal-layer panel: DoG ON / OFF / change, Gabor energy / change, head features
    flow.mp4, layers.mp4 the same as videos (not with --light)
    dog_timecourse.png   each DoG neuron's response (ON + OFF, ref frame) inside the moving objects vs the background
    stats.json           EPE per region (all / object interior / boundary / background), the same with PAIR-ONLY
                         input (all clip frames replaced by the nearest pair frame: how much the net uses time), and
                         per-neuron object / background response for DoG neurons and Gabor units
and <out_dir>/summary.json averages the scenes. tests/<exp>/<step>/index.html plays the frame sequences in sync.
"""
import argparse
import json
import os
import os.path as osp
import sys

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, 'tools'))

from core.utils import paths  # noqa: E402
from evaluate import load_model  # noqa: E402
from viz_common import GENERATOR, LayerView, Scene, load_generator, make_clip, predict, render_panel, scene_args  # noqa: E402

CONFIG = osp.join(ROOT, 'configs', 'pipeline.json')
SILENT_FRAC = 0.01  # a unit whose mean response is below 1% of the strongest unit's is marked silent


# random-pattern test cases (see tools/eval_random_patterns.py) -> generator settings
CASES = {'occluded': dict(),                                                   # textured shapes over the background
         'hollow': dict(object_type='hollow', ring_width=8),                   # only a ring of texture moves
         'roi': dict(object_type='aperture', aperture_texture='static'),       # a moving window onto pattern 2
         'visible': dict()}                                                     # control: occluded, objects +60 grey


def scene_name(s):
    return f"scene{s['scene']}_{s['case']}{'_final' if s.get('final') else ''}"


@torch.no_grad()
def unit_responses(model, scene, t):
    """Mean response of every DoG neuron (ON + OFF at the reference frame) and Gabor unit (quadrature energy at the
    reference frame) inside the moving objects and on the background of frame t."""
    f = model.fnet
    x = (2 * make_clip(model, scene, t, False) / 255 - 1).contiguous()
    dog = f.dog(x)                                     # (1, 2N, 4, H, W); time 1 <-> reference frame
    gab = f.gabor(dog)                                 # (1, 2M, 2, H/2, W/2); time 0 <-> reference frame
    N, M = f.dog.num_filters, f.gabor.num_filters
    d = (dog[0, :N, 1] + dog[0, N:, 1]).float()       # (N, H, W)
    g = torch.sqrt(gab[0, :M, 0] ** 2 + gab[0, M:, 0] ** 2).float()
    obj = torch.from_numpy(scene.labels[t] > 0).cuda()
    obj2 = F.interpolate(obj[None, None].float(), size=g.shape[-2:], mode='nearest')[0, 0] > 0.5
    out = {}
    for name, a, m in (('dog', d, obj), ('gabor', g, obj2)):
        if m.any() and (~m).any():
            out[name] = (a[:, m].mean(1).cpu().numpy(), a[:, ~m].mean(1).cpu().numpy())
    return out


def plot_timecourse(path, ts, obj, bg, labels, title):
    obj, bg = np.asarray(obj), np.asarray(bg)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={'width_ratios': [3, 1]})
    cols = plt.cm.tab10(np.arange(obj.shape[1]) % 10)
    for i in range(obj.shape[1]):
        axes[0].plot(ts, obj[:, i], '-', color=cols[i], label=labels[i])
        axes[0].plot(ts, bg[:, i], ':', color=cols[i])
    axes[0].set_xlabel('pair t (t -> t+1)')
    axes[0].set_ylabel('mean ON+OFF response')
    axes[0].set_title(f'{title}: solid = inside moving objects, dotted = background')
    axes[0].legend(fontsize=7, ncol=4)
    ratio = obj.mean(0) / np.maximum(bg.mean(0), 1e-12)
    silent = np.maximum(obj.mean(0), bg.mean(0)) < SILENT_FRAC * max(obj.max(), bg.max())
    bars = axes[1].bar(range(len(ratio)), ratio, color=cols)
    for i in np.flatnonzero(silent):
        bars[i].set_alpha(0.3)
        bars[i].set_hatch('//')
        axes[1].text(i, ratio[i], 'silent', ha='center', va='bottom', fontsize=7, rotation=90)
    axes[1].axhline(1, color='k', lw=0.7)
    axes[1].set_xticks(range(len(ratio)), labels, rotation=90, fontsize=7)
    axes[1].set_title(f'object / background (hatched: < {SILENT_FRAC:.0%} of max)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def dog_labels(model):
    dog = model.fnet.dog
    if dog.temporal_mode == 'typed':
        return [f"{i} {'bi' if b else 'mono'}" for i, b in enumerate(dog.is_biphasic.tolist())]
    return [str(i) for i in range(dog.num_filters)]


def render_scene(model, gen, spec, cfg, out_dir, light, lview):
    mc = cfg['motion']
    sargs = scene_args(seed=mc['seed'], iters=cfg['iters'], case=spec['case'], **CASES[spec['case']])
    scene = Scene(gen, sargs, spec['scene'], spec['case'] == 'visible', spec.get('final', False))
    T, ref = model.num_frames, model.ref_offset
    t_first, t_last = ref, sargs.num_frames - (T - ref - 1) - 1
    n = mc['light_pairs'] if light else mc['pairs']
    ts = list(range(t_first, min(t_first + n, t_last + 1)))
    for sub in ('flow', 'layers'):
        os.makedirs(osp.join(out_dir, sub), exist_ok=True)
    writers = {}
    stats, stats_pair, dog_o, dog_b, gab_o, gab_b = [], [], [], [], [], []
    q = [cv2.IMWRITE_JPEG_QUALITY, mc['jpeg_quality']]
    for k, t in enumerate(ts):
        flow = predict(model, scene, t, False, cfg['iters'])
        panel, st = render_panel(scene, t, flow, sargs, {'pair_only': False})
        layers = lview.render(scene, t, False)
        _, st_p = render_panel(scene, t, predict(model, scene, t, True, cfg['iters']), sargs, {'pair_only': True})
        stats.append(st)
        stats_pair.append(st_p)
        r = unit_responses(model, scene, t)
        if 'dog' in r:
            dog_o.append(r['dog'][0])
            dog_b.append(r['dog'][1])
        if 'gabor' in r:
            gab_o.append(r['gabor'][0])
            gab_b.append(r['gabor'][1])
        for sub, img in (('flow', panel), ('layers', layers)):
            cv2.imwrite(osp.join(out_dir, sub, f'{k:04d}.jpg'), img, q)
            if not light:
                if sub not in writers:
                    writers[sub] = cv2.VideoWriter(osp.join(out_dir, f'{sub}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                                   mc['fps'], (img.shape[1], img.shape[0]))
                writers[sub].write(img)
    for w in writers.values():
        w.release()

    def mean(rows):
        return {key: round(float(np.nanmean([s[key] for s in rows])), 4) for key in rows[0]}
    res = {'scene': spec, 'name': scene_name(spec), 'pairs': [ts[0], ts[-1]], 'n_frames': len(ts),
           'objects': [{'shape': o['shape'], 'size': [o['width'], o['height']], 'velocity': list(map(float, o['velocity']))}
                       for o in scene.desc['objects']],
           'background_velocity': list(map(float, scene.desc['background_velocity'])),
           'epe': mean(stats), 'epe_pair_only': mean(stats_pair)}
    labels = dog_labels(model)
    if dog_o:
        o, b = np.mean(dog_o, 0), np.mean(dog_b, 0)
        res['dog'] = {'labels': labels, 'object': np.round(o, 6).tolist(), 'background': np.round(b, 6).tolist(),
                      'object_over_background': np.round(o / np.maximum(b, 1e-12), 3).tolist(),
                      'silent': (np.maximum(o, b) < SILENT_FRAC * max(o.max(), b.max())).tolist()}
        plot_timecourse(osp.join(out_dir, 'dog_timecourse.png'), ts[:len(dog_o)], dog_o, dog_b, labels, res['name'])
    if gab_o:
        o, b = np.mean(gab_o, 0), np.mean(gab_b, 0)
        res['gabor'] = {'object': np.round(o, 6).tolist(), 'background': np.round(b, 6).tolist(),
                        'object_over_background': np.round(o / np.maximum(b, 1e-12), 3).tolist()}
    with open(osp.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f"  {res['name']}: EPE {res['epe']['all']:.3f} (pair only {res['epe_pair_only']['all']:.3f}), "
          f"{len(ts)} pairs", flush=True)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', help="checkpoint reference, e.g. E003 or E003@40000")
    p.add_argument('--out_dir', default=None, help='default: tests/<experiment>/<step>/motion/')
    p.add_argument('--config', default=CONFIG)
    p.add_argument('--generator', default=GENERATOR)
    p.add_argument('--light', action='store_true', help='first scene only, light_pairs pairs, no .mp4')
    a = p.parse_args()
    cfg = json.load(open(a.config))
    out_dir = a.out_dir or paths.tests_dir(a.model, 'motion')
    torch.manual_seed(0)
    model = load_model(a.model).eval()
    gen = load_generator(a.generator)
    lview = LayerView(model, cfg['motion']['tile_h'])
    specs = cfg['motion']['scenes'][:1] if a.light else cfg['motion']['scenes']
    print(f'render_motion {paths.describe(a.model)[1]} -> {out_dir}', flush=True)
    scenes = [render_scene(model, gen, s, cfg, osp.join(out_dir, scene_name(s)), a.light, lview) for s in specs]

    summary = {'checkpoint': paths.describe(a.model)[1], 'light': a.light, 'fps': cfg['motion']['fps'],
               'scenes': [{'name': s['name'], 'n_frames': s['n_frames'], 'epe': s['epe']['all'],
                           'epe_pair_only': s['epe_pair_only']['all'], 'epe_boundary': s['epe']['boundary'],
               'epe_shape_inside': s['epe'].get('shape: inside outline')}
                          for s in scenes]}
    if all('dog' in s for s in scenes):
        summary['dog_labels'] = scenes[0]['dog']['labels']
        summary['dog_object_over_background'] = np.round(
            np.mean([s['dog']['object_over_background'] for s in scenes], 0), 3).tolist()
        summary['dog_silent'] = np.all([s['dog']['silent'] for s in scenes], 0).tolist()
    with open(osp.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('wrote', out_dir, flush=True)


if __name__ == '__main__':
    main()
