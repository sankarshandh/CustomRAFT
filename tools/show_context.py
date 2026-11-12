"""What the context encoder outputs and how much the GRU uses it (random-dot scenes).

    python tools/show_context.py                      # latest checkpoint, camouflaged scene 0, ablation over 12 scenes
    python tools/show_context.py --visible --scene 3
    python tools/show_context.py --model E001@5000 --n_eval 20

CustomRAFT's context encoder sees ONLY the reference frame and returns 160 channels, split into
    net  (tanh, 96 ch)  -> the GRU's initial hidden state h0
    inp  (relu, 64 ch)  -> concatenated to the motion features and fed to the GRU at EVERY iteration
The GRU (core/update.py ConvGRU) has three 3x3 convs (update gate z, reset gate r, candidate q) whose input is
    [ h (96) | inp (64) | motion features (82) ]  = 242 channels.

Writes four PNGs to runs/context_outputs/<tag>/ (cyan = true object outline, drawn for you only):
  1_context.png    context features: PCA colours of net and inp, and the 6 highest-variance channels of each, with
                   their correlation to the object mask. Under camouflage the reference frame holds no object, so a
                   large correlation cannot come from appearance.
  2_gru_weights.png  the weights the GRU applies to each input channel (L2 norm of that channel's slice of
                   convz / convr / convq), coloured h / context (inp) / motion, and the same weights multiplied by the
                   channel's RMS activation (how much each channel really pushes the gate). Titles give each group's
                   share of the total.
  3_gru_usage.png  per pixel: mean update gate z, mean reset gate r, and the share of the gate pre-activation that comes
                   from the context channels (|conv(inp slice)| / sum of the three slices), at iterations 1, 4 and last.
  4_ablation.png   EPE with the context removed or swapped (inp = 0, net = 0, both = 0, context taken from another
                   scene), averaged over --n_eval scenes, split into object interior / boundary / background.
Numbers are also written to context_stats.json.
"""
import argparse
import json
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
from core.utils.flow_viz import flow_to_color  # noqa: E402
from core.utils.utils import coords_grid, upflow8  # noqa: E402
from core.utils import paths  # noqa: E402
from evaluate import load_model  # noqa: E402
from live_random_dot import GENERATOR, Scene, boundary_band, latest_checkpoint, load_generator  # noqa: E402
from show_layer_outputs import SCENE_DEFAULTS, outline, pct_range, small_mask, tile  # noqa: E402

GROUPS = ('hidden h (from net)', 'context inp', 'motion features')
COLORS = ('C0', 'C3', 'C2')


@torch.no_grad()
def run(model, clip, iters=32, mod=None, ctx_clip=None, trace=False):
    """CustomRAFT.forward, re-implemented to allow context ablations and to record the GRU inputs.
    mod: None | 'no_inp' | 'no_net' | 'no_ctx'.  ctx_clip: use this clip for the context encoder instead."""
    norm = lambda c: (2 * c / 255 - 1).contiguous()
    x = norm(clip)
    fmap1, fmap2 = model.fnet(x)
    corr_fn = CorrPyramid(fmap1.float(), fmap2.float(), model.corr_levels, model.corr_radius)
    src = x if ctx_clip is None else norm(ctx_clip)
    cnet = model.cnet(src[:, :, model.ref_offset])
    net, inp = torch.split(cnet, [model.hidden_dim, model.context_dim], dim=1)
    net, inp = torch.tanh(net), torch.relu(inp)
    ctx = {'net': net.clone(), 'inp': inp.clone()}
    if mod in ('no_inp', 'no_ctx'):
        inp = torch.zeros_like(inp)
    if mod in ('no_net', 'no_ctx'):
        net = torch.zeros_like(net)
    b, _, h, w = fmap1.shape
    coords0 = coords_grid(b, h, w, device=x.device)
    coords1 = coords0.clone()
    rec = []
    for i in range(iters):
        corr = corr_fn(coords1)
        flow = coords1 - coords0
        if trace:
            motion = model.update_block.encoder(flow, corr)
            rec.append({'h': net.clone(), 'inp': inp.clone(), 'motion': motion})
        net, _, delta = model.update_block(net, inp, corr, flow)
        coords1 = coords1 + delta
    return upflow8(coords1 - coords0), ctx, rec


@torch.no_grad()
def gru_parts(gru, rec):
    """Gates and per-slice pre-activation contributions for one recorded iteration."""
    h, inp, mot = rec['h'], rec['inp'], rec['motion']
    hx = torch.cat([h, inp, mot], 1)
    z = torch.sigmoid(gru.convz(hx))
    r = torch.sigmoid(gru.convr(hx))
    H, C = h.shape[1], inp.shape[1]
    sl = {'h': slice(0, H), 'ctx': slice(H, H + C), 'mot': slice(H + C, hx.shape[1])}
    hxq = torch.cat([r * h, inp, mot], 1)
    out = {'z': z, 'r': r}
    for name, conv, src in (('z', gru.convz, hx), ('q', gru.convq, hxq)):
        out['contrib_' + name] = {k: F.conv2d(src[:, s], conv.weight[:, s], padding=1).abs().mean(1)[0]
                                  for k, s in sl.items()}
    return out, sl


def mean_epe_regions(epe, lab, band):
    obj = lab > 0
    reg = {'all': np.ones_like(obj), 'object interior': obj & ~band, 'boundary': band, 'background': ~obj & ~band}
    return {k: (float(epe[m].sum()), int(m.sum())) for k, m in reg.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=None)
    p.add_argument('--generator', default=GENERATOR)
    p.add_argument('--scene', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--t', type=int, default=20)
    p.add_argument('--visible', action='store_true')
    p.add_argument('--final', action='store_true')
    p.add_argument('--iters', type=int, default=32)
    p.add_argument('--n_eval', type=int, default=12, help='scenes for the ablation (0 = skip)')
    p.add_argument('--band', type=int, default=6)
    p.add_argument('--out', default=None)
    p.add_argument('--show', action='store_true')
    args = p.parse_args()
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gen = load_generator(args.generator)
    ckpt = args.model or latest_checkpoint()
    model = load_model(ckpt).eval()
    tag = f'{paths.describe(ckpt)[1]}_scene{args.scene}_{"visible" if args.visible else "camouflaged"}'
    out_dir = args.out or osp.join(osp.dirname(paths.analysis_dir(ckpt, 'context_outputs')), tag.split('@')[-1])
    os.makedirs(out_dir, exist_ok=True)
    sargs = SimpleNamespace(seed=args.seed, **SCENE_DEFAULTS)
    T, ref, t = model.num_frames, model.ref_offset, args.t
    clip_of = lambda sc: torch.from_numpy(sc.frames[np.arange(T) + t - ref]).float().cuda()[None, None] \
        .expand(1, 3, -1, -1, -1).contiguous()

    scene = Scene(gen, sargs, args.scene, args.visible, args.final)
    clip = clip_of(scene)
    lab, flow_gt = scene.labels[t], scene.flows[t]
    obj = lab > 0
    gru = model.update_block.gru
    stats = {'tag': tag}

    flow_up, ctx, rec = run(model, clip, args.iters, trace=True)
    flow_pred = flow_up[0].permute(1, 2, 0).cpu().numpy()
    epe = np.linalg.norm(flow_pred - flow_gt, axis=-1)
    print(f'{tag}: EPE {epe.mean():.3f}; objects: ' +
          ', '.join(f'{o["shape"]} {o["width"]}x{o["height"]} v={o["velocity"]}' for o in scene.desc['objects']))

    # ------------------------------------------------------------------ 1. context features
    net, inp = ctx['net'][0].cpu(), ctx['inp'][0].cpu()          # (96, h, w), (64, h, w)
    h8, w8 = net.shape[-2:]
    m8 = small_mask(obj, (h8, w8)).astype(np.float32)
    edge8 = boundary_band(small_mask(obj, (h8, w8)).astype(np.uint8), 1).astype(np.float32)

    def corr_with(feat, mask):
        f = feat.flatten(1).numpy()
        f = f - f.mean(1, keepdims=True)
        mk = mask.ravel() - mask.mean()
        return (f @ mk) / (np.linalg.norm(f, axis=1) * np.linalg.norm(mk) + 1e-8)

    def pca_rgb(feat):
        X = feat.flatten(1).T
        X = X - X.mean(0)
        _, _, V = torch.pca_lowrank(X, q=3, center=False)
        z = (X @ V[:, :3]).T.reshape(3, h8, w8).numpy()
        lo, hi = pct_range(z, 1, 99)
        return np.clip((z.transpose(1, 2, 0) - lo) / (hi - lo), 0, 1)

    corr_obj = {'net': corr_with(net, m8), 'inp': corr_with(inp, m8)}
    corr_edge = {'net': corr_with(net, edge8), 'inp': corr_with(inp, edge8)}
    stats['context_vs_object_mask_max_abs_corr'] = {k: float(np.abs(v).max()) for k, v in corr_obj.items()}
    stats['context_vs_object_mask_mean_abs_corr'] = {k: float(np.abs(v).mean()) for k, v in corr_obj.items()}
    stats['context_vs_boundary_max_abs_corr'] = {k: float(np.abs(v).max()) for k, v in corr_edge.items()}
    stats['inp_dead_channels'] = int((inp.flatten(1).max(1).values < 1e-3).sum())
    stats['net_saturated_fraction'] = float((net.abs() > 0.97).float().mean())

    K = 6
    fig, axes = plt.subplots(3, K, figsize=(3.0 * K, 7.6), squeeze=False)
    tile(axes[0, 0], scene.frames[t], obj, 'gray', 0, 255, 'reference frame (the only input of the context encoder)')
    tile(axes[0, 1], flow_to_color(flow_gt, 8.0), None, None, title='ground-truth flow')
    outline(axes[0, 1], obj)
    tile(axes[0, 2], flow_to_color(flow_pred, 8.0), None, None, title='predicted flow')
    outline(axes[0, 2], obj)
    tile(axes[0, 3], pca_rgb(net), m8 > 0, None, title='net (tanh, 96 ch): PCA colours')
    tile(axes[0, 4], pca_rgb(inp), m8 > 0, None, title='inp (relu, 64 ch): PCA colours')
    axes[0, 5].axis('off')
    axes[0, 5].text(0, 0.95, f'max |corr| with the object mask\n  net  {np.abs(corr_obj["net"]).max():.2f}'
                             f'   inp  {np.abs(corr_obj["inp"]).max():.2f}\nmax |corr| with its boundary\n'
                             f'  net  {np.abs(corr_edge["net"]).max():.2f}   inp  {np.abs(corr_edge["inp"]).max():.2f}\n'
                             f'dead inp channels: {stats["inp_dead_channels"]} / 64\n'
                             f'net |value| > 0.97: {stats["net_saturated_fraction"]:.0%}',
                    va='top', fontsize=9, family='monospace')
    for row, (name, feat) in enumerate((('net', net), ('inp', inp)), start=1):
        order = feat.flatten(1).var(1).argsort(descending=True)[:K]
        for j, c in enumerate(order.tolist()):
            lo, hi = pct_range(feat[c].numpy(), 0.5, 99.5)
            tile(axes[row, j], feat[c].numpy(), m8 > 0, 'viridis', lo, hi,
                 f'{name} ch {c}   corr(object) {corr_obj[name][c]:+.2f}')
    fig.suptitle(f'{tag}: context encoder output (highest-variance channels; cyan = true object outline)', fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, '1_context.png'), dpi=80)

    # ------------------------------------------------------------------ 2. GRU weights on each input channel
    r1 = rec[0]
    sl_sizes = [model.hidden_dim, model.context_dim, rec[0]['motion'].shape[1]]
    bounds = np.cumsum([0] + sl_sizes)
    rms = torch.cat([r1[k].flatten(2).pow(2).mean(2)[0].sqrt() for k in ('h', 'inp', 'motion')]).cpu().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(19, 6.4), squeeze=False)
    for col, (nm, conv) in enumerate((('update gate z', gru.convz), ('reset gate r', gru.convr),
                                      ('candidate q', gru.convq))):
        wn = conv.weight.detach().pow(2).sum((0, 2, 3)).sqrt().cpu().numpy()  # (242,)
        for row, (vals, ylab) in enumerate(((wn, 'weight norm of the channel'),
                                            (wn * rms, 'weight norm x channel RMS activation (iteration 1)'))):
            ax = axes[row, col]
            share = [vals[bounds[g]:bounds[g + 1]].__pow__(2).sum() for g in range(3)]
            share = np.array(share) / sum(share)
            for g in range(3):
                ax.bar(np.arange(bounds[g], bounds[g + 1]), vals[bounds[g]:bounds[g + 1]], color=COLORS[g], width=1.0,
                       label=f'{GROUPS[g]}: {share[g]:.0%} of squared total, mean {vals[bounds[g]:bounds[g + 1]].mean():.3f}')
            ax.set_title(f'{nm}' if row == 0 else '', fontsize=10)
            ax.set_xlabel('GRU input channel', fontsize=8)
            ax.set_ylabel(ylab, fontsize=7)
            ax.legend(fontsize=7, loc='upper left')
            ax.tick_params(labelsize=7)
            stats.setdefault('gru_share_' + ('weight' if row == 0 else 'weight_x_activation'), {})[nm] = \
                {GROUPS[g]: float(share[g]) for g in range(3)}
    fig.suptitle(f'{tag}: what the GRU weights on each of its 242 input channels', fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, '2_gru_weights.png'), dpi=80)

    # ------------------------------------------------------------------ 3. spatial usage of the context
    its = sorted({0, 3, len(rec) - 1})
    fig, axes = plt.subplots(len(its), 4, figsize=(14.5, 2.9 * len(its)), squeeze=False)
    obj_s = small_mask(obj, (h8, w8))
    for row, it in enumerate(its):
        parts, _ = gru_parts(gru, rec[it])
        z, r = parts['z'][0].mean(0).cpu().numpy(), parts['r'][0].mean(0).cpu().numpy()
        def share(c):
            a = c['ctx'] / (c['h'] + c['ctx'] + c['mot'] + 1e-8)
            return a.cpu().numpy()
        sz, sq = share(parts['contrib_z']), share(parts['contrib_q'])
        tile(axes[row, 0], z, obj_s, 'viridis', 0, 1, f'iteration {it + 1}: update gate z (1 = take the new candidate)')
        tile(axes[row, 1], r, obj_s, 'viridis', 0, 1, f'iteration {it + 1}: reset gate r')
        tile(axes[row, 2], sz, obj_s, 'magma', 0, max(float(np.percentile(sz, 99)), 1e-3),
             f'context share of z pre-activation (mean {sz.mean():.2f})')
        tile(axes[row, 3], sq, obj_s, 'magma', 0, max(float(np.percentile(sq, 99)), 1e-3),
             f'context share of candidate q (mean {sq.mean():.2f})')
        stats.setdefault('context_share', {})[f'iter{it + 1}'] = {
            'z_all': float(sz.mean()), 'z_object': float(sz[obj_s].mean()) if obj_s.any() else None,
            'z_background': float(sz[~obj_s].mean()), 'q_all': float(sq.mean()),
            'mean_update_gate': float(z.mean()), 'mean_reset_gate': float(r.mean())}
    fig.suptitle(f'{tag}: how the GRU uses the context, per pixel (1/8 resolution; cyan = true object outline)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, '3_gru_usage.png'), dpi=80)

    # ------------------------------------------------------------------ 4. ablation over scenes
    if args.n_eval > 0:
        scenes = [Scene(gen, sargs, i, args.visible, args.final)
                  for i in range(args.n_eval)]
        clips = [clip_of(s) for s in scenes]
        conds = {'normal': dict(), 'inp = 0': dict(mod='no_inp'), 'net = 0': dict(mod='no_net'),
                 'inp = net = 0': dict(mod='no_ctx'), 'context from another scene': dict(shuffle=True)}
        acc = {c: {} for c in conds}
        for i, (s, cl) in enumerate(zip(scenes, clips)):
            band = boundary_band(s.labels[t], args.band)
            for c, kw in conds.items():
                kw = dict(kw)
                if kw.pop('shuffle', False):
                    kw['ctx_clip'] = clips[(i + 1) % len(clips)]
                fu, _, _ = run(model, cl, args.iters, **kw)
                e = np.linalg.norm(fu[0].permute(1, 2, 0).cpu().numpy() - s.flows[t], axis=-1)
                for reg, (sm, n) in mean_epe_regions(e, s.labels[t], band).items():
                    a = acc[c].setdefault(reg, [0.0, 0])
                    a[0] += sm; a[1] += n
        res = {c: {reg: v[0] / max(v[1], 1) for reg, v in d.items()} for c, d in acc.items()}
        stats['ablation_epe'] = res
        regions = ['all', 'object interior', 'boundary', 'background']
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
        for ax, reg in zip(axes, regions):
            vals = [res[c][reg] for c in conds]
            ax.bar(range(len(conds)), vals, color=['gray', 'C3', 'C0', 'C1', 'C4'])
            for i, v in enumerate(vals):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels(list(conds), rotation=25, ha='right', fontsize=7)
            ax.set_title(f'EPE (px): {reg}', fontsize=9)
        fig.suptitle(f'{tag}: removing / swapping the context, mean over {args.n_eval} scenes', fontsize=10)
        fig.tight_layout()
        fig.savefig(osp.join(out_dir, '4_ablation.png'), dpi=80)
        print('ablation EPE (px):')
        for c in conds:
            print(f'  {c:30s} ' + '  '.join(f'{reg} {res[c][reg]:.3f}' for reg in regions))

    with open(osp.join(out_dir, 'context_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print('context vs object mask, max |corr|:', {k: round(v, 2) for k, v in
                                                  stats['context_vs_object_mask_max_abs_corr'].items()})
    print('GRU share of squared weight norm (convz):', {k: round(v, 2) for k, v in
                                                       stats['gru_share_weight']['update gate z'].items()})
    print('wrote', out_dir)
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
