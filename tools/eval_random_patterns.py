"""EPE on the random-pattern test cases (RandomDot datasets), against texture flow AND shape flow, per region.

    python tools/eval_random_patterns.py E003                         # all cases, clean + final, 40 scenes each
    python tools/eval_random_patterns.py E003 --max_scenes 8 --passes clean --out scratch/rp.json

Cases (folders in D:\\IISc\\Motion Datasets\\RandomDot, made by generate_random_dot.py; 50% of scenes have a moving
background, as in the original camouflage set):
    occluded  textured shapes move over a random-dot background with the same statistics (= camouflage)
    hollow    only a ring of texture moves with the shape; the background shows inside the ring
    roi       a moving window: pattern 2 inside, pattern 1 outside; pattern 2 is static, only the window moves
    visible   control: like occluded, but the objects are brighter (+60 grey), so single frames show them

Two ground truths:
    texture   flow/       how the dots move (what RAFT is trained to report). ROI interior = 0 (the dots are static);
                          hollow hole = background velocity.
    shape     flow_shape/ how the shape moves (object velocity over the whole outline). Same as texture for occluded
                          and visible.
Regions (from the reference frame): object = pixels carrying the moving texture (the ring for hollow, the window for
roi); hole = inside the outline but not object (hollow only); boundary = within --band px of a shape outline;
background = the rest. Only pairs whose whole clip lies inside the sequence are used.
Writes one JSON: {case: {pass: {'texture': {region: EPE}, 'shape': {region: EPE}, 'pairs': n}}}.
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, 'tools'))

from core.utils import paths  # noqa: E402
from core.utils.frame_utils import read_flo  # noqa: E402
from evaluate import load_model  # noqa: E402
from viz_common import boundary_band  # noqa: E402

RANDOMDOT = os.environ.get('RANDOMDOT_ROOT', r'D:\IISc\Motion Datasets\RandomDot')
CASES = ('occluded', 'hollow', 'roi', 'visible')
REGIONS = ('all', 'object', 'hole', 'boundary', 'background')


def read_png(path):
    return np.array(Image.open(path))


@torch.no_grad()
def eval_case(model, root, dstype, max_scenes, iters, band):
    tr = osp.join(root, 'training')
    scenes = sorted(os.listdir(osp.join(tr, dstype)))[:max_scenes or None]
    T, ref = model.num_frames, model.ref_offset
    acc = {g: {r: [0.0, 0] for r in REGIONS} for g in ('texture', 'shape')}
    n_pairs = 0
    for sc in scenes:
        files = sorted(os.listdir(osp.join(tr, dstype, sc)))
        frames = np.stack([read_png(osp.join(tr, dstype, sc, f))[..., 0] for f in files])  # grey saved as RGB
        x_all = torch.from_numpy(frames).cuda().float()
        for t in range(ref, len(files) - (T - ref - 1) - 1 + 1):
            if t + 1 >= len(files):
                break
            fn = files[t][:-4]
            clip = x_all[t - ref:t - ref + T][None, None].expand(1, 3, -1, -1, -1).contiguous()
            _, up = model(clip, iters=iters, test_mode=True)
            pred = up[0].permute(1, 2, 0).cpu().numpy()
            tex = read_flo(osp.join(tr, 'flow', sc, fn + '.flo'))
            shp_path = osp.join(tr, 'flow_shape', sc, fn + '.flo')
            shp = read_flo(shp_path) if osp.exists(shp_path) else tex
            seg = read_png(osp.join(tr, 'segmentation', sc, fn + '.png'))
            ss_path = osp.join(tr, 'segmentation_shape', sc, fn + '.png')
            sseg = read_png(ss_path) if osp.exists(ss_path) else seg
            bnd = boundary_band(sseg, band)
            regions = {'all': np.ones(seg.shape, bool), 'object': (seg > 0) & ~bnd,
                       'hole': (sseg > 0) & (seg == 0) & ~bnd, 'boundary': bnd, 'background': (sseg == 0) & ~bnd}
            for g, gt in (('texture', tex), ('shape', shp)):
                e = np.linalg.norm(pred - gt, axis=-1)
                for r, m in regions.items():
                    acc[g][r][0] += float(e[m].sum())
                    acc[g][r][1] += int(m.sum())
            n_pairs += 1
    out = {g: {r: (round(s / n, 4) if n else None) for r, (s, n) in d.items()} for g, d in acc.items()}
    out['pairs'] = n_pairs
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', help="checkpoint reference, e.g. E003 or E003@40000")
    p.add_argument('--cases', default=','.join(CASES))
    p.add_argument('--passes', default='clean,final')
    p.add_argument('--max_scenes', type=int, default=0, help='0 = all scenes')
    p.add_argument('--iters', type=int, default=32)
    p.add_argument('--band', type=int, default=4, help='boundary band half-width in px')
    p.add_argument('--root', default=RANDOMDOT)
    p.add_argument('--out', default=None, help='default: <experiment>/analysis/random_patterns/<step>/random_patterns.json')
    a = p.parse_args()
    torch.manual_seed(0)
    model = load_model(a.model).eval()
    res = {'checkpoint': paths.describe(a.model)[1], 'max_scenes': a.max_scenes, 'band': a.band, 'iters': a.iters}
    for case in a.cases.split(','):
        res[case] = {}
        for dstype in a.passes.split(','):
            r = eval_case(model, osp.join(a.root, case), dstype, a.max_scenes, a.iters, a.band)
            res[case][dstype] = r
            print(f'{case:9s} {dstype:5s} texture ' + ' '.join(f'{k} {v}' for k, v in r['texture'].items()) +
                  ' | shape ' + ' '.join(f'{k} {v}' for k, v in r['shape'].items()) + f' ({r["pairs"]} pairs)',
                  flush=True)
    path = a.out or osp.join(paths.analysis_dir(a.model, 'random_patterns'), 'random_patterns.json')
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
