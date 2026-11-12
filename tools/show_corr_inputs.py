"""Show the feature maps that CustomRAFT feeds into its 4D correlation, at every pyramid level, on a random-dot scene.

    python tools/show_corr_inputs.py                          # latest checkpoint, camouflaged scene 0
    python tools/show_corr_inputs.py --visible                # same scene, objects visible in single frames
    python tools/show_corr_inputs.py --min_size 12 --max_size 20   # small squares (patch no longer fits inside)

Only one feature scale enters the correlation: fmap1, fmap2 = fnet(clip), 128-d at 1/8 resolution. CorrPyramid builds
levels 1..3 by average-pooling the correlation volume over frame-2 positions; correlation is linear, so

    level k  =  < fmap1 , avgpool_{2^k}(fmap2) > / sqrt(128)

i.e. at level k, fmap1 (1/8) is compared with fmap2 pooled to cells of 8 * 2^k input pixels. The script checks this
identity against the model's own CorrPyramid before plotting.

Writes to runs/corr_inputs/<ckpt>_scene<S>_<camouflaged|visible>/ (cyan = true object outline, never seen by the net):

  1_pyramid_features.png  one row per level: frame t+1 averaged over the level's cells, fmap1 and pooled fmap2 as PCA
                          colours (one projection shared by all levels), and the 6 highest-variance pooled fmap2 channels.
  2_fmap_channels.png     the 16 highest-variance channels of fmap1 and fmap2 at 1/8.
  3_corr_slices.png       for query pixels on the object interior, boundary and background: the full correlation map of
                          that fmap1 pixel against all of pooled fmap2, per level. Red + = true target, white x = best
                          match, red box = the (2r+1)^2 window the GRU reads around the target.
                          Title: hit = best match within 1 cell of the target; z = (best - 2nd best peak) / std(map),
                          the 2nd peak taken outside the 3x3 cells around the best.
  4_neighbour_pooling.png the same query cells, match score vs displacement (+-search cells) after averaging the score
                          over k x k neighbouring cells (k = 1, 3, 5, 9), the way a convolutional update pools evidence.
                          One cell is ambiguous; a few cells that move together give a single clear peak.
"""
import argparse
import os
import os.path as osp
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, 'scripts'))

from core.layers.correlation import CorrPyramid  # noqa: E402
from core.utils import paths  # noqa: E402
from evaluate import load_model  # noqa: E402
from live_random_dot import GENERATOR, Scene, boundary_band, latest_checkpoint, load_generator  # noqa: E402
from show_layer_outputs import SCENE_DEFAULTS, pct_range, small_mask, tile  # noqa: E402

REGIONS = ('interior', 'boundary', 'background')


def peak_stats(maps, targets, hw):
    """maps: (N, Hk*Wk) correlation rows; targets: (N, 2) target (x, y) in level cells.
    Returns hit (N,) bool, z (N,), argmax (N, 2) as (x, y)."""
    hk, wk = hw
    top, idx = maps.max(1)
    ay, ax = idx // wk, idx % wk
    yy, xx = torch.meshgrid(torch.arange(hk, device=maps.device), torch.arange(wk, device=maps.device), indexing='ij')
    yy, xx = yy.reshape(1, -1), xx.reshape(1, -1)
    near = torch.maximum((yy - ay[:, None]).abs(), (xx - ax[:, None]).abs()) <= 1
    second = maps.masked_fill(near, float('-inf')).max(1).values
    z = (top - second) / maps.std(1).clamp_min(1e-8)
    tx, ty = targets[:, 0].round(), targets[:, 1].round()
    hit = torch.maximum((ax - tx).abs(), (ay - ty).abs()) <= 1
    return hit, z, torch.stack([ax, ay], 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=None, help="checkpoint: 'E003', 'E003@30000' or a .pth path (default: newest experiment's final)")
    p.add_argument('--generator', default=GENERATOR)
    p.add_argument('--scene', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--t', type=int, default=20, help='reference frame of the pair (t -> t+1), 0-based')
    p.add_argument('--visible', action='store_true')
    p.add_argument('--final', action='store_true')
    p.add_argument('--dot_lifetime', type=int, default=None)
    p.add_argument('--min_size', type=float, default=None)
    p.add_argument('--max_size', type=float, default=None)
    p.add_argument('--queries', type=int, default=1, help='query pixels per region in 3_corr_slices.png')
    p.add_argument('--search', type=int, default=8, help='displacement search half-width in 1/8 cells for figure 4')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    gen = load_generator(args.generator)
    ckpt = args.model or latest_checkpoint()
    model = load_model(ckpt).eval()
    tag = (f'{paths.describe(ckpt)[1]}_scene{args.scene}_{"visible" if args.visible else "camouflaged"}'
           + (f'_life{args.dot_lifetime}' if args.dot_lifetime else '')
           + (f'_size{args.min_size:g}-{args.max_size:g}' if args.min_size or args.max_size else ''))
    out_dir = args.out or osp.join(osp.dirname(paths.analysis_dir(ckpt, 'corr_inputs')), tag.split('@')[-1])
    os.makedirs(out_dir, exist_ok=True)

    sd = dict(SCENE_DEFAULTS)
    for k in ('dot_lifetime', 'min_size', 'max_size'):
        if getattr(args, k) is not None:
            sd[k] = getattr(args, k)
    scene = Scene(gen, SimpleNamespace(seed=args.seed, **sd), args.scene, args.visible, args.final)
    T, ref, t = model.num_frames, model.ref_offset, args.t
    idx = np.arange(T) + t - ref
    assert idx[0] >= 0 and idx[-1] < len(scene.frames), 'choose --t so that the whole clip lies inside the scene'
    clip = torch.from_numpy(scene.frames[idx]).float().cuda()[None, None].expand(1, 3, -1, -1, -1)
    clip = (2 * clip / 255 - 1).contiguous()
    obj1, obj2 = scene.labels[t] > 0, scene.labels[t + 1] > 0
    print(f'{tag}: pair {t}->{t + 1}; objects: ' +
          ', '.join(f'{o["shape"]} {o["width"]}x{o["height"]} v={o["velocity"]}' for o in scene.desc['objects']) +
          f'; background v={scene.desc["background_velocity"]}')

    L, R = model.corr_levels, model.corr_radius
    with torch.no_grad():
        f1, f2 = model.fnet(clip)
        f1, f2 = f1.float(), f2.float()
        C, h, w = f1.shape[1:]
        pyr = CorrPyramid(f1, f2, L, R).pyramid               # level k: (h*w, 1, h/2^k, w/2^k)
        f2s = [F.avg_pool2d(f2, 2 ** k) if k else f2 for k in range(L)]
        for k in range(L):
            mine = torch.einsum('bchw,bcuv->bhwuv', f1, f2s[k]).reshape(pyr[k].shape) / C ** 0.5
            err = (mine - pyr[k]).abs().max().item()
            assert err < 1e-4 * max(1.0, pyr[k].abs().max().item()), f'level {k}: max |diff| {err}'
        print(f'check passed: CorrPyramid level k == <fmap1, avgpool_2^k(fmap2)> / sqrt({C}) for k = 0..{L - 1}')

    # ---------------------------------------------------------------- shared PCA projection (fit on level-0 f1 + f2)
    X = torch.cat([f1[0].flatten(1), f2[0].flatten(1)], 1).T
    mu = X.mean(0)
    _, _, V = torch.pca_lowrank(X - mu, q=3, center=False)
    def pca(a):
        hh, ww = a.shape[-2:]
        return ((a.flatten(1).T - mu) @ V[:, :3]).T.reshape(3, hh, ww).cpu().numpy()
    z1 = pca(f1[0])
    lo, hi = pct_range(np.concatenate([z1.ravel(), pca(f2[0]).ravel()]), 1, 99)
    to_img = lambda z: np.clip((z.transpose(1, 2, 0) - lo) / (hi - lo), 0, 1)
    var = X.var(0)
    top = var.argsort(descending=True).cpu().numpy()

    # ---------------------------------------------------------------- 1. pyramid features
    nch = 6
    frame2 = torch.from_numpy(scene.frames[t + 1]).float()[None, None]
    fig, axes = plt.subplots(L, 3 + nch, figsize=(2.6 * (3 + nch), 2.3 * L + 0.6), squeeze=False)
    for k in range(L):
        cell = 8 * 2 ** k
        g = f2s[k][0]
        shp = g.shape[-2:]
        m2 = small_mask(obj2, shp)
        tile(axes[k, 0], F.avg_pool2d(frame2, cell)[0, 0].numpy(), m2, 'gray', 0, 255,
             f'level {k}: frame t+1 averaged\nover {cell}x{cell} px cells')
        tile(axes[k, 1], to_img(z1), small_mask(obj1, (h, w)), None, title='fmap1 (1/8, same every level)')
        tile(axes[k, 2], to_img(pca(g)), m2, None, title=f'fmap2 pooled to 1/{cell}  ({shp[0]}x{shp[1]})')
        for j in range(nch):
            c = top[j]
            a = g[c].cpu().numpy()
            v = pct_range(np.abs(f2[0, c].cpu().numpy()), 0, 99.5)[1]
            tile(axes[k, 3 + j], a, m2, 'RdBu_r', -v, v, f'fmap2 ch {c}' if k == 0 else None)
    fig.suptitle(f'{tag}: what enters the correlation at each pyramid level '
                 f'(level k compares fmap1 with fmap2 pooled to 1/{8}·2^k)', fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, '1_pyramid_features.png'), dpi=80)
    plt.close(fig)

    # ---------------------------------------------------------------- 2. channels at 1/8
    n = 16
    fig, axes = plt.subplots(4, 8, figsize=(2.6 * 8, 2.1 * 4 + 0.6))
    m1, m2 = small_mask(obj1, (h, w)), small_mask(obj2, (h, w))
    for i in range(n):
        c = top[i]
        a1, a2 = f1[0, c].cpu().numpy(), f2[0, c].cpu().numpy()
        v = pct_range(np.abs(np.concatenate([a1.ravel(), a2.ravel()])), 0, 99.5)[1]
        r, col = 2 * (i // 8), i % 8
        tile(axes[r, col], a1, m1, 'RdBu_r', -v, v, f'ch {c}  fmap1 (t)')
        tile(axes[r + 1, col], a2, m2, 'RdBu_r', -v, v, f'ch {c}  fmap2 (t+1)')
    fig.suptitle(f'{tag}: 16 highest-variance channels of fmap1 / fmap2 (128-d, 1/8 resolution)', fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, '2_fmap_channels.png'), dpi=80)
    plt.close(fig)

    # ---------------------------------------------------------------- region masks and targets at 1/8
    lab8 = F.interpolate(torch.from_numpy(scene.labels[t].astype(np.float32))[None, None], size=(h, w),
                         mode='nearest')[0, 0].numpy().astype(np.int32)
    band8 = boundary_band(lab8, 1)
    regions = {'interior': (lab8 > 0) & ~band8, 'boundary': band8, 'background': (lab8 == 0) & ~band8}
    fl = torch.from_numpy(scene.flows[t]).permute(2, 0, 1).float()[None]
    fl8 = (F.avg_pool2d(fl, 8) / 8.0)[0].cuda()                             # (2, h, w) flow in 1/8 cells
    yy, xx = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
    tgt0 = torch.stack([xx + fl8[0], yy + fl8[1]], -1).reshape(-1, 2)      # (h*w, 2) target at level 0

    # ---------------------------------------------------------------- statistics over all pixels per region
    print(f'\n{"region":11s} {"level":>5s} {"cell px":>7s} {"hit rate":>8s} {"chance":>7s} {"mean z":>7s}  (pixels)')
    stats = {}
    for name, mask in regions.items():
        sel = torch.from_numpy(mask.reshape(-1)).cuda()
        if not sel.any():
            continue
        for k in range(L):
            hk, wk = pyr[k].shape[-2:]
            maps = pyr[k].reshape(h * w, -1)[sel]
            hit, z, _ = peak_stats(maps, tgt0[sel] / 2 ** k, (hk, wk))
            chance = min(1.0, 9 / (hk * wk))
            stats[name, k] = (hit.float().mean().item(), z.mean().item())
            print(f'{name:11s} {k:5d} {8 * 2 ** k:7d} {hit.float().mean().item():8.2f} {chance:7.3f} '
                  f'{z.mean().item():7.2f}  ({int(sel.sum())})')

    # ---------------------------------------------------------------- 3. correlation slices for query pixels
    rng = np.random.default_rng(args.seed)
    queries = []
    for name in REGIONS:
        ys, xs = np.nonzero(regions[name])
        if len(ys):
            for i in rng.choice(len(ys), min(args.queries, len(ys)), replace=False):
                queries.append((name, int(xs[i]), int(ys[i])))
    fig, axes = plt.subplots(len(queries), 1 + L, figsize=(3.3 * (1 + L), 2.7 * len(queries) + 0.6), squeeze=False)
    for qi, (name, qx, qy) in enumerate(queries):
        ax = axes[qi, 0]
        tile(ax, scene.frames[t], obj1, 'gray', 0, 255, f'{name}: query cell ({qx}, {qy}) in frame t')
        ax.add_patch(Rectangle((qx * 8 - 0.5, qy * 8 - 0.5), 8, 8, fill=False, ec='red', lw=1.5))
        q = qy * w + qx
        for k in range(L):
            hk, wk = pyr[k].shape[-2:]
            row = pyr[k][q, 0]
            tg = tgt0[q] / 2 ** k
            hit, z, am = peak_stats(row.reshape(1, -1), tg[None], (hk, wk))
            ax = axes[qi, 1 + k]
            v = pct_range(row.cpu().numpy(), 0.5, 99.9)
            tile(ax, row.cpu().numpy(), small_mask(obj2, (hk, wk)), 'viridis', v[0], v[1],
                 f'level {k} (1/{8 * 2 ** k})  hit {"yes" if hit.item() else "NO"}  z {z.item():.1f}')
            tx, ty = tg.tolist()
            ax.add_patch(Rectangle((tx - R - 0.5, ty - R - 0.5), 2 * R + 1, 2 * R + 1, fill=False, ec='red', lw=1))
            ax.plot(tx, ty, '+', color='red', ms=10, mew=2)
            ax.plot(am[0, 0].item(), am[0, 1].item(), 'x', color='white', ms=8, mew=2)
    fig.suptitle(f'{tag}: correlation of one fmap1 cell with ALL of pooled fmap2 '
                 f'(red + = true target, white x = best match, red box = GRU lookup window r={R})', fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, '3_corr_slices.png'), dpi=80)
    plt.close(fig)

    # ---------------------------------------------------------------- 4. pooling the evidence over neighbouring cells
    D, KS = args.search, (1, 3, 5, 9)
    with torch.no_grad():
        cost = displacement_cost(f1[0], f2[0], D)                          # (h, w, 2D+1, 2D+1), NaN outside image
        pooled = {k: pool_cost(cost, k) for k in KS}
    gdx, gdy = fl8[0].round().long(), fl8[1].round().long()
    print(f'\nexact best displacement among {(2 * D + 1) ** 2} candidates (search +-{D} cells), '
          f'after averaging the cost over k x k neighbouring cells:')
    print(f'{"region":11s}' + ''.join(f'  k={k} ({8 * k}px)' for k in KS))
    for name, mask in regions.items():
        sel = torch.from_numpy(mask).cuda()
        if not sel.any():
            continue
        rates = []
        for k in KS:
            am = pooled[k].reshape(h, w, -1).argmax(-1)
            hit = (am // (2 * D + 1) - D == gdy) & (am % (2 * D + 1) - D == gdx)
            rates.append(hit[sel].float().mean().item())
        print(f'{name:11s}' + ''.join(f'  {r:11.2f}' for r in rates))

    fig, axes = plt.subplots(len(queries), len(KS), figsize=(3.0 * len(KS), 2.9 * len(queries) + 0.6), squeeze=False)
    for qi, (name, qx, qy) in enumerate(queries):
        tx, ty = fl8[0, qy, qx].item() + D, fl8[1, qy, qx].item() + D
        for j, k in enumerate(KS):
            mp = pooled[k][qy, qx].cpu().numpy()
            fin = np.isfinite(mp)
            ax = axes[qi, j]
            v = pct_range(mp[fin], 0.5, 100) if fin.any() else (0, 1)
            tile(ax, np.where(fin, mp, np.nan), None, 'viridis', v[0], v[1],
                 f'{name}: {k}x{k} cells ({8 * k} px)' if j else f'{name} ({qx},{qy}): 1 cell (8 px)')
            ay, axx = np.unravel_index(np.nanargmax(np.where(fin, mp, -np.inf)), mp.shape)
            ax.plot(tx, ty, '+', color='red', ms=12, mew=2)
            ax.plot(axx, ay, 'x', color='white', ms=9, mew=2)
            ax.set_xlabel('dx (cells)', fontsize=7)
    fig.suptitle(f'{tag}: match score vs displacement (+-{D} cells = +-{8 * D} px) for one query cell,\n'
                 f'averaged over its k x k neighbours (red + = true displacement, white x = best)', fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, '4_neighbour_pooling.png'), dpi=80)
    plt.close(fig)
    print('\nwrote', out_dir)


def displacement_cost(f1, f2, D):
    """cost[y, x, D+dy, D+dx] = <f1[:, y, x], f2[:, y+dy, x+dx]> / sqrt(C); NaN where (y+dy, x+dx) is outside."""
    C, h, w = f1.shape
    f2p = F.pad(f2, (D, D, D, D), value=float('nan'))
    return torch.stack([torch.stack([(f1 * f2p[:, D + dy:D + dy + h, D + dx:D + dx + w]).sum(0) / C ** 0.5
                                     for dx in range(-D, D + 1)], -1) for dy in range(-D, D + 1)], -2)


def pool_cost(cost, k):
    """Average each displacement's score over the k x k neighbouring query cells (ignoring out-of-image entries);
    -inf where the displacement itself leaves the image."""
    h, w, n, _ = cost.shape
    c = cost.permute(2, 3, 0, 1).reshape(1, n * n, h, w)
    valid = (~c.isnan()).float()
    agg = F.avg_pool2d(torch.nan_to_num(c), k, 1, k // 2) / F.avg_pool2d(valid, k, 1, k // 2).clamp_min(1e-6)
    agg = agg.reshape(n, n, h, w).permute(2, 3, 0, 1)
    return torch.where(cost.isnan(), torch.full_like(agg, float('-inf')), agg)


if __name__ == '__main__':
    main()
