"""Show what each spatiotemporal layer of CustomRAFT outputs on a random-dot scene.

    python tools/show_layer_outputs.py                       # trained model (latest checkpoint), camouflaged scene
    python tools/show_layer_outputs.py --init                # the same layers at initialisation (untrained)
    python tools/show_layer_outputs.py --visible --scene 3   # objects visible in single frames (control)
    python tools/show_layer_outputs.py --show                # also open matplotlib windows

Writes three PNGs to runs/layer_outputs/<tag>/ (the cyan outline is the TRUE object boundary, drawn for you only;
the network never sees it):

  1_dog.png      DoG layer, one column per neuron. Row 1: its temporal kernel (5 frames, last frame on the right).
                 Row 2/3: ON = GELU(r) and OFF = GELU(-r) at the reference frame. Row 4: r(ref+1) - r(ref), i.e. how
                 the neuron's response changes from one frame to the next (blue/red = down/up).
  2_gabor.png    Oriented layer, 16 neurons (2 scales x 8 orientations), quadrature energy sqrt(even^2 + odd^2) at the
                 reference frame (row pairs: scale 0 then scale 1) and the same energy at ref+1 minus ref.
                 Titles: learned direction theta (image coordinates, y down), spatial frequency and drift speed.
  3_features.png Output of the strided CNN head for both frames (fmap1, fmap2; 128-d at 1/8 resolution) as a 3-component
                 PCA colour image (same projection for both), and the pixel-wise match between them: cosine
                 similarity of fmap1(x) with fmap2(x + GT flow), which is what correlation can exploit.
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

from core.custom_raft import CustomRAFT  # noqa: E402
from core.utils import paths  # noqa: E402
from evaluate import load_model  # noqa: E402
from live_random_dot import GENERATOR, Scene, latest_checkpoint, load_generator  # noqa: E402

SCENE_DEFAULTS = dict(num_frames=48, height=384, width=512, dot_size=2, dot_mode='binary', density=0.5,
                      dot_lifetime=0, shapes='all', max_objects=2, min_size=48, max_size=160, min_speed=1.0,
                      max_speed=8.0, static_bg_prob=0.5, bg_max_speed=3.0, final_noise=10.0,
                      subpixel=1, object_density=None, object_offset=0.0)


def pct_range(x, lo=0.5, hi=99.5):
    a, b = np.percentile(x, [lo, hi])
    return float(a), float(max(b, a + 1e-6))


def outline(ax, mask):
    ax.contour(mask.astype(float), levels=[0.5], colors='cyan', linewidths=0.7)


def tile(ax, img, mask, cmap, vmin=None, vmax=None, title=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    if mask is not None:
        outline(ax, mask)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def small_mask(mask, shape):
    return F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None], size=shape, mode='nearest')[0, 0].numpy() > 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=None, help="checkpoint: 'E003', 'E003@30000' or a .pth path (default: newest experiment's final)")
    p.add_argument('--init', action='store_true', help='use freshly initialised layers instead of a checkpoint')
    p.add_argument('--generator', default=GENERATOR)
    p.add_argument('--scene', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--t', type=int, default=20, help='reference frame of the scene (0-based)')
    p.add_argument('--visible', action='store_true')
    p.add_argument('--final', action='store_true')
    p.add_argument('--out', default=None)
    p.add_argument('--show', action='store_true')
    args = p.parse_args()
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gen = load_generator(args.generator)
    if args.init:
        torch.manual_seed(args.seed)
        model, tag_model = CustomRAFT().cuda().eval(), 'init'
    else:
        ckpt = args.model or latest_checkpoint()
        model, tag_model = load_model(ckpt).eval(), paths.describe(ckpt)[1]
    tag = f'{tag_model}_scene{args.scene}_{"visible" if args.visible else "camouflaged"}'
    if args.init:  # untrained network: no experiment to file it under
        default_dir = osp.join(ROOT, 'scratch', 'layer_outputs', tag)
    else:
        default_dir = osp.join(osp.dirname(paths.analysis_dir(ckpt, 'layer_outputs')), tag.split('@')[-1])
    out_dir = args.out or default_dir
    os.makedirs(out_dir, exist_ok=True)

    sargs = SimpleNamespace(seed=args.seed, **SCENE_DEFAULTS)
    scene = Scene(gen, sargs, args.scene, args.visible, args.final)
    fnet = model.fnet
    T, ref = model.num_frames, model.ref_offset
    t = args.t
    idx = np.arange(T) + t - ref
    assert idx[0] >= 0 and idx[-1] < len(scene.frames), 'choose --t so that the whole clip lies inside the scene'
    clip = torch.from_numpy(scene.frames[idx]).float().cuda()[None, None].expand(1, 3, -1, -1, -1)  # (1,3,T,H,W)
    clip = (2 * clip / 255 - 1).contiguous()
    lab = scene.labels[t]
    obj = lab > 0
    flow = scene.flows[t]
    print(f'{tag}: scene {args.scene}, pair {t + 1}->{t + 2}; objects: ' +
          ', '.join(f'{o["shape"]} {o["width"]}x{o["height"]} v={o["velocity"]}' for o in scene.desc['objects']) +
          f'; background v={scene.desc["background_velocity"]}')

    with torch.no_grad():
        dog = fnet.dog(clip)                      # (1, 2N, 4, H, W): channel [0:N] ON, [N:2N] OFF; time k <-> frame k+2
        gab = fnet.gabor(dog)                     # (1, 2M, 2, H/2, W/2): even [0:M], odd [M:2M]; time 0 <-> ref, 1 <-> ref+1
        f1, f2 = fnet(clip)                       # (1, 128, H/8, W/8)
        dc, ratio = fnet.dog.temporal_profile()
        tk = fnet.dog.temporal_kernels().cpu().numpy()
        resp_ref = (dog[0, :fnet.dog.num_filters, 1] - dog[0, fnet.dog.num_filters:, 1]).cpu().numpy()   # r = ON - OFF
        resp_next = (dog[0, :fnet.dog.num_filters, 2] - dog[0, fnet.dog.num_filters:, 2]).cpu().numpy()
    N = fnet.dog.num_filters
    on, off = dog[0, :N, 1].cpu().numpy(), dog[0, N:, 1].cpu().numpy()
    M = fnet.gabor.num_filters
    en = torch.sqrt(gab[0, :M] ** 2 + gab[0, M:] ** 2).cpu().numpy()  # (M, 2, h, w)
    th = (torch.rad2deg(fnet.gabor.theta.detach()) % 360).cpu().numpy()
    fr = fnet.gabor.frequency.detach().cpu().numpy()
    sp = fnet.gabor.speed.detach().cpu().numpy()

    # ---------------------------------------------------------------- 1. DoG
    fig, axes = plt.subplots(4, N, figsize=(3.0 * N, 9.4), squeeze=False)
    for n in range(N):
        ax = axes[0, n]
        ax.axhline(0, color='gray', lw=0.5)
        ax.plot(range(-2, 3), tk[n], 'o-', color='C3' if ratio[n] > 0.3 else 'C0', lw=1.5)
        ax.set_xticks(range(-2, 3)); ax.tick_params(labelsize=6)
        ax.set_title(f'DoG {n}   DC gain {dc[n]:.2f}   odd/even {ratio[n]:.2f}', fontsize=8)
        if n == 0:
            ax.set_ylabel('temporal kernel', fontsize=8)
        v = pct_range(np.concatenate([on[n].ravel(), off[n].ravel()]), 0, 99.5)[1]
        tile(axes[1, n], on[n], obj, 'gray', 0, v)
        tile(axes[2, n], off[n], obj, 'gray', 0, v)
        d = resp_next[n] - resp_ref[n]
        a = pct_range(np.abs(d), 0, 99.5)[1]
        tile(axes[3, n], d, obj, 'RdBu_r', -a, a)
    for r, name in enumerate(['', 'ON  GELU(r)', 'OFF  GELU(-r)', 'r(ref+1) - r(ref)']):
        if r:
            axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle(f'{tag}: spatiotemporal DoG layer (cyan = true object outline, not seen by the network)', fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, '1_dog.png'), dpi=80)

    # ---------------------------------------------------------------- 2. Gabor
    scales = M // 8
    fig, axes = plt.subplots(2 * scales, 8, figsize=(2.9 * 8, 2.3 * 2 * scales + 0.6), squeeze=False)
    ms = small_mask(obj, en.shape[-2:])
    for i in range(M):
        s, o = divmod(i, 8)
        v = pct_range(en[i, 0], 0, 99.5)[1]
        tile(axes[2 * s, o], en[i, 0], ms, 'magma', 0, v,
             f'n{i}  θ={th[i]:.0f}° f={fr[i]:.2f} v={sp[i]:+.1f}')
        d = en[i, 1] - en[i, 0]
        a = pct_range(np.abs(d), 0, 99.5)[1]
        tile(axes[2 * s + 1, o], d, ms, 'RdBu_r', -a, a)
        if o == 0:
            axes[2 * s, 0].set_ylabel(f'scale {s}: energy @ ref', fontsize=8)
            axes[2 * s + 1, 0].set_ylabel('energy(ref+1) - energy(ref)', fontsize=8)
    fig.suptitle(f'{tag}: oriented (Gabor) layer, quadrature energy', fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, '2_gabor.png'), dpi=80)

    # ---------------------------------------------------------------- 3. head features
    a1, a2 = f1[0].float(), f2[0].float()               # (128, h, w)
    h, w = a1.shape[-2:]
    X = torch.cat([a1.flatten(1), a2.flatten(1)], 1).T
    mu = X.mean(0)
    X = X - mu
    _, _, V = torch.pca_lowrank(X, q=3, center=False)
    def pca_img(a):
        return ((a.flatten(1).T - mu) @ V[:, :3]).T.reshape(3, h, w).cpu().numpy()
    z1, z2 = pca_img(a1), pca_img(a2)
    lo, hi = pct_range(np.concatenate([z1.ravel(), z2.ravel()]), 1, 99)
    to_img = lambda z: np.clip((z.transpose(1, 2, 0) - lo) / (hi - lo), 0, 1)
    # cosine similarity between fmap1(x) and fmap2(x + flow/8), nearest cell
    n1, n2 = F.normalize(a1, dim=0), F.normalize(a2, dim=0)
    fl = torch.from_numpy(flow).permute(2, 0, 1).float().cuda()[None]
    fl8 = F.avg_pool2d(fl, 8) / 8.0
    yy, xx = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
    # bilinear sampling of fmap2 at x + flow/8 (flow is usually a fraction of a 1/8-resolution cell)
    gx = 2 * (xx + fl8[0, 0]) / max(w - 1, 1) - 1
    gy = 2 * (yy + fl8[0, 1]) / max(h - 1, 1) - 1
    n2w = F.grid_sample(n2[None], torch.stack([gx, gy], -1)[None], mode='bilinear', padding_mode='border',
                        align_corners=True)[0]
    cos_true = (n1 * F.normalize(n2w, dim=0)).sum(0).cpu().numpy()
    cos_zero = (n1 * n2).sum(0).cpu().numpy()
    mk = small_mask(obj, (h, w))
    fig, axes = plt.subplots(1, 5, figsize=(21, 3.9))
    tile(axes[0], scene.frames[t], obj, 'gray', 0, 255, 'input frame t')
    tile(axes[1], to_img(z1), mk, None, title='fmap1 (PCA colours)')
    tile(axes[2], to_img(z2), mk, None, title='fmap2 (PCA colours)')
    tile(axes[3], cos_true, mk, 'viridis', -0.2, 1.0, 'cos( fmap1(x), fmap2(x + GT flow) )')
    tile(axes[4], cos_zero, mk, 'viridis', -0.2, 1.0, 'cos( fmap1(x), fmap2(x) )  [zero-flow match]')
    fig.suptitle(f'{tag}: head output (1/8 resolution)', fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, '3_features.png'), dpi=80)

    print(f'match with GT flow (cosine): object {cos_true[mk].mean():.2f}, background {cos_true[~mk].mean():.2f};'
          f' zero-flow match: object {cos_zero[mk].mean():.2f}, background {cos_zero[~mk].mean():.2f}')
    print('wrote', out_dir)
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
