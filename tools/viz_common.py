"""Shared pieces for the random-dot viewers and renderers (moved verbatim from live_random_dot.py).

Used by tools/live_random_dot.py, tools/live_encoder.py, tools/render_motion.py and the show_* tools:
    GENERATOR, load_generator, latest_checkpoint   scene generator and default checkpoint
    Scene, make_clip, predict                      one rendered random-dot scene, the model's input clip, its flow
    render_panel, LayerView                        the flow panel and the spatiotemporal-layer panel (BGR uint8)
    caption, to_bgr, boundary_band, diverging      drawing helpers
    scene_args(**overrides)                        a namespace with live_random_dot.py's default scene parameters
"""
import importlib.util
import os
import os.path as osp
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.utils import paths  # noqa: E402
from core.utils.flow_viz import flow_to_color  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402

GENERATOR = os.environ.get('RDS_GENERATOR', r'D:\IISc\Motion Datasets\generate_random_dot.py')


def load_generator(path):
    spec = importlib.util.spec_from_file_location('generate_random_dot', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def latest_checkpoint():
    """Final checkpoint of the newest experiment, as a reference such as 'E003' (see core/utils/paths.py)."""
    return paths.latest_checkpoint()


def to_bgr(img):
    return np.ascontiguousarray(img[..., ::-1])


def caption(img, text, org=(6, 18), scale=0.5, color=(255, 255, 255)):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (org[0] - 4, org[1] - th - 4), (org[0] + tw + 4, org[1] + base + 2), (0, 0, 0), -1)  # backing box
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return img


def boundary_band(labels, width):
    """True within `width` px of a label change (the motion boundaries)."""
    lab = labels.astype(np.int32)
    edge = np.zeros(lab.shape, bool)
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return cv2.dilate(edge.astype(np.uint8), k) > 0


class Scene:
    """One rendered scene (T frames) with ground truth, for a given object visibility / noise setting."""

    def __init__(self, gen, args, seed, visible, final):
        a = SimpleNamespace(**vars(args))
        a.object_offset = 60.0 if visible else 0.0
        rng = np.random.default_rng([args.seed, seed])
        self.desc = gen.make_scene(rng, a)
        frames, self.labels, self.flows, self.occ = gen.render_scene(rng, self.desc, a)
        if final:
            noise = np.random.default_rng([args.seed, seed, 1]).normal(0, args.final_noise, frames.shape)
            frames = np.clip(np.round(frames + noise), 0, 255)
        self.frames = frames.astype(np.uint8)  # (T, H, W)
        self.visible, self.final, self.seed = visible, final, seed
        # hollow / aperture (ROI) scenes: the shape moves differently from the dots -> also keep the shape motion
        self.case = getattr(args, 'case', None)
        self.shape_flows = self.shape_labels = None
        if self.desc.get('object_type') in ('aperture', 'hollow'):
            self.shape_labels = (gen.shape_labels(self.desc, a) if self.desc['object_type'] == 'hollow'
                                 else self.labels)
            self.shape_flows = gen.shape_flows(self.desc, self.shape_labels)


def make_clip(model, scene, t, pair_only):
    """(1, 3, T, H, W) float clip in [0, 255] centred on the pair (t, t+1)."""
    T, ref = model.num_frames, model.ref_offset
    idx = np.arange(T) + t - ref
    clip = scene.frames[np.clip(idx, 0, len(scene.frames) - 1)]  # (T, H, W)
    clip = torch.from_numpy(clip).float().cuda()[None, None].expand(1, 3, -1, -1, -1).contiguous()  # (1,3,T,H,W)
    if pair_only:
        clip[:, :, :ref] = clip[:, :, ref:ref + 1]
        clip[:, :, ref + 2:] = clip[:, :, ref + 1:ref + 2]
    return clip


def diverging(d):
    """d in [-1, 1] -> uint8 BGR, red = positive, blue = negative, white = 0."""
    pos, neg = np.clip(d, 0, 1)[..., None], np.clip(-d, 0, 1)[..., None]
    rgb = 1 - (pos * np.array([0, 1, 1]) + neg * np.array([1, 1, 0]))
    return (rgb[..., ::-1] * 255).astype(np.uint8)


class LayerView:
    """Second window: outputs of the spatiotemporal layers for the current pair (see show_layer_outputs.py)."""

    def __init__(self, model, tile_h=96):
        self.model, self.th, self.tw = model, tile_h, tile_h * 4 // 3
        f = model.fnet
        with torch.no_grad():
            dc, ratio = f.dog.temporal_profile()
        self.dc, self.ratio = dc.cpu().numpy(), ratio.cpu().numpy()
        self.theta = (torch.rad2deg(f.gabor.theta.detach()) % 360).cpu().numpy()
        self.speed = f.gabor.speed.detach().cpu().numpy()
        g = torch.Generator().manual_seed(0)
        self.proj = (torch.randn(128, 3, generator=g) / 128 ** 0.5).cuda()  # fixed colours for the 128-d features

    def _norm(self, x, signed=False, q=0.995):
        """x (K, h, w) tensor -> (K, th, tw) numpy in [0, 1] (or [-1, 1]), per-tile scaling."""
        x = F.adaptive_avg_pool2d(x[:, None].float(), (self.th, self.tw))[:, 0]
        s = x.abs() if signed else x
        a = torch.quantile(s.flatten(1), q, dim=1).clamp_min(1e-6)[:, None, None]
        x = x / a
        return (x.clamp(-1, 1) if signed else x.clamp(0, 1)).cpu().numpy()

    @torch.no_grad()
    def render(self, scene, t, pair_only):
        m, f = self.model, self.model.fnet
        th, tw = self.th, self.tw
        x = (2 * make_clip(m, scene, t, pair_only) / 255 - 1).contiguous()
        dog = f.dog(x)                       # (1, 2N, 4, H, W): ON [0:N], OFF [N:2N]; time k <-> clip frame k + 2
        gab = f.gabor(dog)                   # (1, 2M, 2, H/2, W/2): even [0:M], odd [M:2M]; time 0 <-> ref, 1 <-> ref+1
        f1, f2 = f(x)
        N, M = f.dog.num_filters, f.gabor.num_filters
        on, off = dog[0, :N, 1], dog[0, N:, 1]
        dr = (dog[0, :N, 2] - dog[0, N:, 2]) - (dog[0, :N, 1] - dog[0, N:, 1])
        en = torch.sqrt(gab[0, :M] ** 2 + gab[0, M:] ** 2)           # (M, 2, h, w)
        den = en[:, 1] - en[:, 0]

        small = cv2.resize(scene.labels[t], (tw, th), interpolation=cv2.INTER_NEAREST)
        edge = boundary_band(small, 0)

        def gray(a): return np.repeat((a * 255).astype(np.uint8)[..., None], 3, 2)
        def mag(a): return cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        rows = []

        def add_row(tiles, label, titles=None):
            row = []
            for i in range(8):
                tile = tiles[i].copy() if i < len(tiles) else np.zeros((th, tw, 3), np.uint8)
                if i < len(tiles):
                    tile[edge] = (255, 255, 0)
                    if titles:
                        caption(tile, titles[i], (2, 11), 0.32)
                if i == 0:
                    caption(tile, label, (2, th - 5), 0.36, (120, 220, 255))
                row.append(tile)
                row.append(np.full((th, 2, 3), 40, np.uint8))
            rows.append(np.concatenate(row[:-1], axis=1))
            rows.append(np.full((2, rows[-1].shape[1], 3), 40, np.uint8))

        ton, toff, tdr = self._norm(on), self._norm(off), self._norm(dr, signed=True)
        v = np.maximum(ton.max(), toff.max())  # (kept per tile; ON and OFF each use their own scale)
        add_row([gray(a) for a in ton], 'DoG ON',
                [f'DoG{i} dc{self.dc[i]:.2f} odd{self.ratio[i]:.1f}' for i in range(N)])
        add_row([gray(a) for a in toff], 'DoG OFF')
        add_row([diverging(a) for a in tdr], 'DoG  r(t+1)-r(t)')
        ten, tden = self._norm(en[:, 0]), self._norm(den, signed=True)
        for s in range(M // 8):
            sl = slice(8 * s, 8 * s + 8)
            add_row([mag(a) for a in ten[sl]], f'Gabor energy, scale {s}',
                    [f'th{self.theta[i]:.0f} v{self.speed[i]:+.1f}' for i in range(8 * s, 8 * s + 8)])
            add_row([diverging(a) for a in tden[sl]], f'Gabor  dE (t+1 - t)')

        # head features: fixed random 128 -> 3 projection as colour, plus the match to the GT flow
        a1, a2 = f1[0].float(), f2[0].float()
        h, w = a1.shape[-2:]
        def colour(a):
            z = torch.einsum('chw,cd->dhw', a - a.mean((1, 2), keepdim=True), self.proj)
            lo, hi = torch.quantile(z.flatten(), 0.01), torch.quantile(z.flatten(), 0.99)
            img = ((z - lo) / (hi - lo + 1e-6)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            return cv2.resize((img[..., ::-1] * 255).astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
        n1, n2 = F.normalize(a1, dim=0), F.normalize(a2, dim=0)
        fl8 = F.avg_pool2d(torch.from_numpy(scene.flows[t]).permute(2, 0, 1).float().cuda()[None], 8)[0] / 8.0
        yy, xx = torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda'), indexing='ij')
        grid = torch.stack([2 * (xx + fl8[0]) / (w - 1) - 1, 2 * (yy + fl8[1]) / (h - 1) - 1], -1)[None]
        n2w = F.normalize(F.grid_sample(n2[None], grid, padding_mode='border', align_corners=True)[0], dim=0)
        def heat(c):
            c = ((c.clamp(-0.2, 1) + 0.2) / 1.2).cpu().numpy()
            return cv2.resize(cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), (tw, th),
                              interpolation=cv2.INTER_NEAREST)
        frame = cv2.resize(np.repeat(scene.frames[t][..., None], 3, 2), (tw, th), interpolation=cv2.INTER_AREA)
        add_row([frame, colour(a1), colour(a2), heat((n1 * n2w).sum(0)), heat((n1 * n2).sum(0))],
                'head features',
                ['frame t', 'fmap1', 'fmap2', 'match at true flow', 'match at zero flow'])

        body = np.concatenate(rows[:-1], axis=0)
        head = np.full((26, body.shape[1], 3), 24, np.uint8)
        caption(head, f'spatiotemporal layers, pair {t + 1}->{t + 2}   (cyan = true object outline, not seen by the net)'
                      f'   dc = DC gain, odd = odd/even ratio', (8, 18), 0.45)
        return np.concatenate([head, body], axis=0)


@torch.no_grad()
def predict(model, scene, t, pair_only, iters):
    """Flow from frame t to t+1 using the clip centred on the pair."""
    clip = make_clip(model, scene, t, pair_only)
    padder = InputPadder(clip.shape)
    (clip,) = padder.pad(clip)
    _, up = model(clip, iters=iters, test_mode=True)
    return padder.unpad(up)[0].permute(1, 2, 0).cpu().numpy()


def render_panel(scene, t, flow_pred, args, state):
    H, W = scene.frames[t].shape
    gt = scene.flows[t]
    lab = scene.labels[t]
    epe = np.linalg.norm(flow_pred - gt, axis=-1)
    band = boundary_band(lab, args.band)
    obj = lab > 0
    stats = {'all': epe.mean(),
             'object interior': epe[obj & ~band].mean() if (obj & ~band).any() else np.nan,
             'boundary': epe[band].mean() if band.any() else np.nan,
             'background': epe[~obj & ~band].mean() if (~obj & ~band).any() else np.nan}

    f0 = np.repeat(scene.frames[t][..., None], 3, 2)
    f1 = np.repeat(scene.frames[t + 1][..., None], 3, 2)
    diff = np.repeat(np.abs(scene.frames[t + 1].astype(int) - scene.frames[t].astype(int)).astype(np.uint8)[..., None], 3, 2)
    scale = args.max_speed
    gt_c = flow_to_color(gt, scale)
    pr_c = flow_to_color(flow_pred, scale)
    err = np.clip(epe / args.err_scale, 0, 1)
    err_c = cv2.applyColorMap((err * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    # outline the true object boundaries on the flow panels so shape errors are easy to see
    outline = boundary_band(lab, 0)
    for im in (gt_c, pr_c):
        im[outline] = 255

    tiles = [[f0, f1, diff], [gt_c, pr_c, err_c]]
    names = [['frame t', 'frame t+1', '|difference|'],
             ['ground-truth flow', 'predicted flow', f'EPE (white = {args.err_scale:g}px)']]
    sflows = getattr(scene, 'shape_flows', None)
    if sflows is not None:  # hollow / ROI: a third row against the motion of the whole shape
        gs = sflows[t]
        es = np.linalg.norm(flow_pred - gs, axis=-1)
        slab = scene.shape_labels[t]
        sband = boundary_band(slab, args.band)
        sobj = slab > 0
        stats.update({'shape: all': es.mean(),
                      'shape: inside outline': es[sobj & ~sband].mean() if (sobj & ~sband).any() else np.nan})
        gs_c = flow_to_color(gs, scale)
        gs_c[boundary_band(slab, 0)] = 255
        es_c = cv2.applyColorMap((np.clip(es / args.err_scale, 0, 1) * 255).astype(np.uint8),
                                 cv2.COLORMAP_INFERNO)[..., ::-1]
        names[1][0] = 'texture flow (how the dots move)'
        tiles.append([gs_c, pr_c.copy(), es_c])
        names.append(['shape flow (how the shape moves)', 'predicted flow', 'EPE vs shape flow'])
    rows = []
    for tr, nr in zip(tiles, names):
        row = []
        for im, nm in zip(tr, nr):
            tile = to_bgr(np.ascontiguousarray(im))
            caption(tile, nm)
            row.append(tile)
            row.append(np.full((H, 4, 3), 40, np.uint8))
        rows.append(np.concatenate(row[:-1], axis=1))
        rows.append(np.full((4, rows[-1].shape[1], 3), 40, np.uint8))
    body = np.concatenate(rows[:-1], axis=0)

    head = np.full((58, body.shape[1], 3), 24, np.uint8)
    mode = lambda b, a_, b_: a_ if b else b_
    caption(head, f'scene {scene.seed}  pair {t + 1}->{t + 2} of {len(scene.frames)}   '
                  f'objects: {getattr(scene, "case", None) and scene.case.upper() or mode(scene.visible, "VISIBLE (+60 grey)", "CAMOUFLAGED")}   '
                  f'pass: {mode(scene.final, "final (noisy)", "clean")}   '
                  f'input: {mode(state["pair_only"], "PAIR ONLY", "full 8-frame clip")}', (8, 18), 0.55)
    s = '   '.join(f'{k} {v:.2f}' for k, v in stats.items())
    caption(head, f'EPE (px)  {s}', (8, 44), 0.6, (120, 220, 255))
    return np.concatenate([head, body], axis=0), stats

# defaults of tools/live_random_dot.py (scene parameters = generate_random_dot.py, plus display settings)
SCENE_ARGS = dict(seed=0, num_frames=48, height=384, width=512, dot_size=2, dot_mode='binary', density=0.5,
                  dot_lifetime=0, shapes='all', max_objects=2, min_size=48, max_size=160, min_speed=1.0,
                  max_speed=8.0, static_bg_prob=0.5, bg_max_speed=3.0, final_noise=10.0,
                  subpixel=1, object_density=None, object_offset=0.0, band=6, err_scale=4.0, iters=32)


def scene_args(**overrides):
    """Namespace with live_random_dot.py's defaults, for Scene(...) and render_panel(...)."""
    return SimpleNamespace(**{**SCENE_ARGS, **overrides})
