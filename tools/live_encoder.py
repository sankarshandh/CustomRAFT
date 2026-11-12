"""Live view of the feature encoder: watch the DoG layer, the Gabor layer or the CNN features respond to motion.

    python tools/live_encoder.py --model E003 --stage dog          # random-pattern scenes (occluded), DoG layer
    python tools/live_encoder.py --model E003 --stage gabor --case roi   # occluded | hollow | roi | visible
    python tools/live_encoder.py --model E003 --stage cnn --sintel alley_1         # a real Sintel sequence
    python tools/live_encoder.py --model E003 --sintel market_2 --pass final --sintel_scale 1

Views (switch with d / g / k):
  dog    top: frame t, |frame t+1 - t|, and for the SELECTED neuron its ON, OFF and temporal change (r(t+1) - r(t)).
         middle: every DoG neuron (ON + OFF). 'a' toggles the scale: SHARED (one scale for all neurons, so a silenced
         neuron looks dark, which is the honest view) or per tile (each neuron stretched to its own range, so the shape of a
         weak response shows). bottom: the selected neuron's spatial kernel, its centre/surround profile and its
         temporal kernel, with sigma, gain and type.
  gabor  top: frame t (click to pick a pixel), the selected unit's even, odd and energy responses, and a direction plot
         at the picked pixel: one arrow per unit along its preferred motion (speed * (cos th, sin th)), length =
         energy there / the unit's mean energy (selectivity), plus the true flow (green) and predicted flow (magenta).
         middle: energy of all units (scale toggled with 'a'). bottom: energy change between the two time steps.
  cnn    frame t (click to pick a pixel), fmap1 and fmap2 as PCA colours (one projection for both), the correlation of
         the picked fmap1 cell with every fmap2 cell (green + = true target, magenta x = predicted target), predicted
         and true flow, the context encoder's net (GRU h0) and inp channels as PCA colours.

Keys:  space pause/play   <- / -> step when paused   n next scene   c next case (occluded, hollow, roi, visible)
       f clean <-> final
       d / g / k  DoG / Gabor / CNN view   0-9 select DoG neuron or Gabor unit   [ ] previous / next unit
       a  shared <-> per-tile scale   s  save a screenshot to <experiment>/analysis/live_shots/<step>/   q / Esc quit
"""
import argparse
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, 'tools'))

from core.datasets import DEFAULT_SINTEL_ROOT  # noqa: E402
from core.utils import paths  # noqa: E402
from core.utils.flow_viz import flow_to_color  # noqa: E402
from core.utils.frame_utils import read_flo, read_image  # noqa: E402
from evaluate import load_model  # noqa: E402
from viz_common import GENERATOR, Scene, caption, diverging, latest_checkpoint, load_generator, scene_args  # noqa: E402
from render_motion import CASES  # noqa: E402  occluded / hollow / roi / visible -> generator settings

SINTEL_SCENES = ['alley_1', 'alley_2', 'ambush_2', 'ambush_4', 'ambush_5', 'ambush_6', 'ambush_7', 'bamboo_1',
                 'bamboo_2', 'bandage_1', 'bandage_2', 'cave_2', 'cave_4', 'market_2', 'market_5', 'market_6',
                 'mountain_1', 'shaman_2', 'shaman_3', 'sleeping_1', 'sleeping_2', 'temple_2', 'temple_3']


# ----------------------------------------------------------------------------------------------- input sources
class Source:
    """frames (T, H, W, 3) uint8, flows (T-1, H, W, 2) or None, labels (T, H, W) or None (object masks)."""

    def __init__(self, name, frames, flows=None, labels=None):
        self.name, self.frames, self.flows, self.labels = name, frames, flows, labels


def random_dot_source(gen, sargs, scene_id, case, final):
    a = scene_args(**{**vars(sargs), 'case': case, **CASES[case]})
    s = Scene(gen, a, scene_id, case == 'visible', final)
    name = f'random pattern {case}, scene {scene_id}, {"final" if final else "clean"}'
    return Source(name, np.repeat(s.frames[..., None], 3, -1), s.flows, s.labels)


def sintel_source(root, scene, dstype, scale):
    d = osp.join(root, 'training', dstype, scene)
    files = sorted(f for f in os.listdir(d) if f.endswith('.png'))
    frames = np.stack([read_image(osp.join(d, f)) for f in files])
    fd = osp.join(root, 'training', 'flow', scene)
    flows = np.stack([read_flo(osp.join(fd, f[:-4] + '.flo')) for f in files[:-1]])
    if scale != 1:
        h, w = frames.shape[1:3]
        sz = (int(w * scale), int(h * scale))
        frames = np.stack([cv2.resize(f, sz, interpolation=cv2.INTER_AREA) for f in frames])
        flows = np.stack([cv2.resize(f, sz, interpolation=cv2.INTER_AREA) * scale for f in flows])
    h8, w8 = frames.shape[1] // 8 * 8, frames.shape[2] // 8 * 8  # crop to a multiple of 8 (no padding needed)
    return Source(f'Sintel {dstype} {scene} (x{scale:g})', frames[:, :h8, :w8], flows[:, :h8, :w8])


def make_clip(model, src, t):
    """(1, 3, T, H, W) in [-1, 1] centred on the pair (t, t+1)."""
    idx = np.clip(np.arange(model.num_frames) + t - model.ref_offset, 0, len(src.frames) - 1)
    x = torch.from_numpy(src.frames[idx]).cuda().float().permute(3, 0, 1, 2)[None]
    return (2 * x / 255 - 1).contiguous()


# ----------------------------------------------------------------------------------------------- drawing helpers
def fit(img, h):
    w = int(round(img.shape[1] * h / img.shape[0]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST if img.shape[0] < h else cv2.INTER_AREA)


def gray(a, lo=None, hi=None):
    lo = np.percentile(a, 0.5) if lo is None else lo
    hi = np.percentile(a, 99.5) if hi is None else hi
    v = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)
    return np.repeat((v * 255).astype(np.uint8)[..., None], 3, -1)


def magma(a, hi):
    v = np.clip(a / max(hi, 1e-12), 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def signed(a, hi):
    return diverging(np.clip(a / max(hi, 1e-12), -1, 1))


def titled(img, text, color=(255, 255, 255)):
    img = np.ascontiguousarray(img)
    return caption(img, text, (4, 14), 0.42, color)


def hstack(tiles, gap=3):
    h = max(t.shape[0] for t in tiles)
    out = []
    for t in tiles:
        if t.shape[0] < h:
            t = np.concatenate([t, np.zeros((h - t.shape[0], t.shape[1], 3), np.uint8)])
        out += [t, np.full((h, gap, 3), 40, np.uint8)]
    return np.concatenate(out[:-1], 1)


def vstack(rows, gap=3):
    w = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        if r.shape[1] < w:
            r = np.concatenate([r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)], 1)
        out += [r, np.full((gap, w, 3), 40, np.uint8)]
    return np.concatenate(out[:-1], 0)


def grid(tiles, cols):
    rows = [hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return vstack(rows)


def pca_rgb(feats):
    """list of (C, h, w) tensors -> list of (h, w, 3) uint8 BGR using ONE shared 3-component PCA projection."""
    X = torch.cat([f.flatten(1) for f in feats], 1).float()
    mu = X.mean(1, keepdim=True)
    X = X - mu
    _, vecs = torch.linalg.eigh(X @ X.T / X.shape[1])
    P = vecs[:, -3:].flip(1)
    out = []
    zs = [torch.einsum('chw,cd->dhw', f.float() - mu[:, :, None], P) for f in feats]
    allz = torch.cat([z.flatten(1) for z in zs], 1)
    lo, hi = torch.quantile(allz, 0.01, dim=1)[:, None, None], torch.quantile(allz, 0.99, dim=1)[:, None, None]
    for z in zs:
        img = ((z - lo) / (hi - lo + 1e-6)).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        out.append((img[..., ::-1] * 255).astype(np.uint8))
    return out


def plot_line(values, size=(200, 120), title='', color=(255, 200, 80), zero=True, bars=False):
    w, h = size
    img = np.full((h, w, 3), 24, np.uint8)
    v = np.asarray(values, float)
    hi = max(np.abs(v).max(), 1e-12)
    y0 = h // 2 if zero else h - 10
    sy = (h // 2 - 16) / hi if zero else (h - 26) / hi
    cv2.line(img, (4, y0), (w - 4, y0), (90, 90, 90), 1)
    xs = np.linspace(10, w - 10, len(v))
    if bars:
        bw = max(2, int((w - 20) / len(v) * 0.6))
        for x, a in zip(xs, v):
            cv2.rectangle(img, (int(x - bw / 2), y0), (int(x + bw / 2), int(y0 - a * sy)), color, -1)
    else:
        pts = np.stack([xs, y0 - v * sy], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
    return titled(img, title)


def arrow(img, c, v, color, thick=2):
    cv2.arrowedLine(img, tuple(int(a) for a in c), (int(c[0] + v[0]), int(c[1] + v[1])), color, thick, cv2.LINE_AA,
                    tipLength=0.15)


# ----------------------------------------------------------------------------------------------- views
class Viewer:
    def __init__(self, model, tile_h, iters):
        self.m, self.f, self.th, self.iters = model, model.fnet, tile_h, iters
        self.dog_types = (['bi' if b else 'mono' for b in self.f.dog.is_biphasic.tolist()]
                          if self.f.dog.temporal_mode == 'typed' else ['free'] * self.f.dog.num_filters)
        self.click_rect = None  # (x0, y0, w, h, H, W): where frame t is drawn, to map mouse clicks to pixels
        self.pixel = None

    def frame_tiles(self, src, t):
        a, b = src.frames[t], src.frames[min(t + 1, len(src.frames) - 1)]
        diff = np.abs(b.astype(int) - a.astype(int)).mean(-1)
        fr = np.ascontiguousarray(a[..., ::-1])
        if self.pixel is not None:
            cv2.drawMarker(fr, self.pixel, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        return fit(fr, self.th), titled(fit(gray(diff, 0, 64), self.th), '|frame t+1 - frame t|')

    def place_frame(self, x0, y0, src):
        H, W = src.frames.shape[1:3]
        self.click_rect = (x0, y0, int(round(W * self.th / H)), self.th, H, W)

    def on_mouse(self, ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN or self.click_rect is None:
            return
        x0, y0, w, h, H, W = self.click_rect
        if x0 <= x < x0 + w and y0 <= y < y0 + h:
            self.pixel = (int((x - x0) * W / w), int((y - y0) * H / h))

    @torch.no_grad()
    def flow(self, x):
        _, up = self.m(x, iters=self.iters, test_mode=True)
        return up[0].permute(1, 2, 0).cpu().numpy()

    # ---- DoG
    @torch.no_grad()
    def dog(self, src, t, sel, shared):
        dg = self.f.dog
        x = make_clip(self.m, src, t)
        r = self.f.dog(x)[0].float()                       # (2N, 4, H, W); time 1 <-> reference frame
        N = dg.num_filters
        on, off = r[:N, 1], r[N:, 1]
        both = (on + off).cpu().numpy()
        change = ((r[:N, 2] - r[N:, 2]) - (r[:N, 1] - r[N:, 1])).cpu().numpy()
        hi_all = np.percentile(both, 99.5)
        fr, df = self.frame_tiles(src, t)
        s = sel % N
        hs = np.percentile(both[s], 99.5) if not shared else hi_all
        top = hstack([titled(fr, 'frame t (input)'), df,
                      titled(fit(magma(on[s].cpu().numpy(), hs), self.th), f'DoG {s} ON'),
                      titled(fit(magma(off[s].cpu().numpy(), hs), self.th), f'DoG {s} OFF'),
                      titled(fit(signed(change[s], np.percentile(np.abs(change[s]), 99.5)), self.th),
                             f'DoG {s} change r(t+1)-r(t)')])
        small = self.th // 2
        tiles = []
        for i in range(N):
            hi = hi_all if shared else np.percentile(both[i], 99.5)
            col = (80, 255, 255) if i == s else (255, 255, 255)
            tiles.append(titled(fit(magma(both[i], hi), small), f'{i} {self.dog_types[i]} rms {both[i].std():.2g}', col))
        mid = grid(tiles, 8)
        sk = dg.spatial_kernels()[s].detach().cpu().numpy()
        tk = dg.temporal_kernels()[s].detach().cpu().numpy()
        kimg = titled(fit(signed(sk, np.abs(sk).max()), 120), 'spatial kernel')
        prof = plot_line(sk[sk.shape[0] // 2], (220, 120), 'centre/surround profile')
        temp = plot_line(tk, (160, 120), 'temporal kernel', (120, 200, 255), bars=True)
        info = np.full((120, 420, 3), 24, np.uint8)
        lines = [f'DoG {s}: {self.dog_types[s]}', f'sigma centre {dg.sigma_center[s]:.3f}  surround {dg.sigma_surround[s]:.3f}',
                 f'gain {dg.gain[s]:.3f}   spatial L1 {np.abs(sk).sum():.4f}',
                 f'scale: {"SHARED (all neurons, one scale)" if shared else "per tile"}  (a toggles)']
        for k, ln in enumerate(lines):
            caption(info, ln, (8, 22 + 26 * k), 0.45)
        bottom = hstack([kimg, prof, temp, info])
        self.place_frame(0, 0, src)
        return vstack([top, mid, bottom])

    # ---- Gabor
    @torch.no_grad()
    def gabor(self, src, t, sel, shared):
        gb = self.f.gabor
        x = make_clip(self.m, src, t)
        g = self.f.gabor(self.f.dog(x))[0].float()        # (2M, 2, h, w): even [0:M], odd [M:2M]; time 0 <-> ref
        M = gb.num_filters
        even, odd = g[:M, 0], g[M:, 0]
        en = torch.sqrt(even ** 2 + odd ** 2).cpu().numpy()
        en1 = torch.sqrt(g[:M, 1] ** 2 + g[M:, 1] ** 2).cpu().numpy()
        s = sel % M
        theta = (torch.rad2deg(gb.theta.detach()) % 360).cpu().numpy()
        speed = gb.speed.detach().cpu().numpy()
        fr, _ = self.frame_tiles(src, t)
        ev, od = even[s].cpu().numpy(), odd[s].cpu().numpy()
        he = np.percentile(np.abs(np.concatenate([ev.ravel(), od.ravel()])), 99.5)
        H, W = src.frames.shape[1:3]
        px = self.pixel or (W // 2, H // 2)
        polar = self.direction_plot(src, t, x, en, theta, speed, px)
        top = hstack([titled(fr, 'frame t (click a pixel)'),
                      titled(fit(signed(ev, he), self.th), f'unit {s} even'),
                      titled(fit(signed(od, he), self.th), f'unit {s} odd'),
                      titled(fit(magma(en[s], np.percentile(en[s], 99.5)), self.th),
                             f'unit {s} energy  th {theta[s]:.0f}  v {speed[s]:+.2f}'), polar])
        hi_all = np.percentile(en, 99.5)
        small = self.th // 2
        tiles, dtiles = [], []
        for i in range(M):
            hi = hi_all if shared else np.percentile(en[i], 99.5)
            col = (80, 255, 255) if i == s else (255, 255, 255)
            tiles.append(titled(fit(magma(en[i], hi), small), f'{i} th{theta[i]:.0f} v{speed[i]:+.1f}', col))
            d = en1[i] - en[i]
            dtiles.append(titled(fit(signed(d, np.percentile(np.abs(d), 99.5)), small), f'{i} dE', col))
        self.place_frame(0, 0, src)
        g1, g2 = grid(tiles, 8), grid(dtiles, 8)
        lab1 = caption(np.full((20, g1.shape[1], 3), 24, np.uint8),
                       f'energy at the reference frame, {"SHARED scale" if shared else "per-tile scale"} (a toggles)',
                       (6, 14), 0.42, (120, 220, 255))
        lab2 = caption(np.full((20, g2.shape[1], 3), 24, np.uint8), 'energy change E(t+1) - E(t), per tile',
                       (6, 14), 0.42, (120, 220, 255))
        return vstack([top, lab1, g1, lab2, g2])

    def direction_plot(self, src, t, x, en, theta, speed, px):
        R = self.th
        img = np.full((R, R, 3), 24, np.uint8)
        c = (R // 2, R // 2)
        cv2.circle(img, c, R // 2 - 6, (70, 70, 70), 1)
        h, w = en.shape[1:]
        H, W = src.frames.shape[1:3]
        yy, xx = min(int(px[1] * h / H), h - 1), min(int(px[0] * w / W), w - 1)
        sel = en[:, yy, xx] / np.maximum(en.mean((1, 2)), 1e-12)
        scale = (R // 2 - 10) / max(sel.max(), 1e-6)
        for i in range(len(sel)):
            d = np.array([np.cos(np.deg2rad(theta[i])), np.sin(np.deg2rad(theta[i]))]) * np.sign(speed[i] or 1)
            arrow(img, c, d * sel[i] * scale, (200, 160, 60), 1)
        fp = self.flow(x)
        vmax = max(4.0, np.abs(fp).max())
        if src.flows is not None:
            gt = src.flows[t][px[1], px[0]]
            arrow(img, c, gt / vmax * (R // 2 - 10), (80, 230, 80), 2)
        arrow(img, c, fp[px[1], px[0]] / vmax * (R // 2 - 10), (230, 80, 230), 2)
        return titled(img, f'direction at {px}: units / true / pred')

    # ---- CNN features
    @torch.no_grad()
    def cnn(self, src, t, sel, shared):
        m = self.m
        x = make_clip(m, src, t)
        f1, f2 = self.f(x)
        f1, f2 = f1[0].float(), f2[0].float()
        c = m.cnet(x[:, :, m.ref_offset])[0].float()
        net, inp = torch.tanh(c[:m.hidden_dim]), torch.relu(c[m.hidden_dim:])
        p1, p2 = pca_rgb([f1, f2])
        pn, = pca_rgb([net])
        pi, = pca_rgb([inp])
        H, W = src.frames.shape[1:3]
        h, w = f1.shape[1:]
        px = self.pixel or (W // 2, H // 2)
        cy, cx = min(px[1] * h // H, h - 1), min(px[0] * w // W, w - 1)
        corr = torch.einsum('c,chw->hw', f1[:, cy, cx], f2).cpu().numpy() / np.sqrt(f1.shape[0])
        cimg = cv2.applyColorMap((np.clip((corr - corr.min()) / (np.ptp(corr) + 1e-6), 0, 1) * 255).astype(np.uint8),
                                 cv2.COLORMAP_VIRIDIS)
        fp = self.flow(x)
        vmax = max(4.0, np.percentile(np.linalg.norm(fp, axis=-1), 99))
        k = self.th / h
        cimg = fit(cimg, self.th)
        tp = fp[px[1], px[0]]
        cv2.drawMarker(cimg, (int((px[0] + tp[0]) / W * w * k), int((px[1] + tp[1]) / H * h * k)), (230, 80, 230),
                       cv2.MARKER_TILTED_CROSS, 14, 2)
        tiles2 = [titled(fit(flow_to_color(fp, vmax)[..., ::-1], self.th), 'predicted flow')]
        if src.flows is not None:
            g = src.flows[t][px[1], px[0]]
            cv2.drawMarker(cimg, (int((px[0] + g[0]) / W * w * k), int((px[1] + g[1]) / H * h * k)), (80, 230, 80),
                           cv2.MARKER_CROSS, 14, 2)
            tiles2.append(titled(fit(np.ascontiguousarray(flow_to_color(src.flows[t], vmax)[..., ::-1]), self.th),
                                 'true flow'))
        fr, _ = self.frame_tiles(src, t)
        top = hstack([titled(fr, 'frame t (click a pixel)'), titled(fit(p1, self.th), 'fmap1 (PCA, shared)'),
                      titled(fit(p2, self.th), 'fmap2 (PCA, shared)'),
                      titled(cimg, f'corr of fmap1{(cx, cy)} with fmap2: + true, x pred')])
        bottom = hstack(tiles2 + [titled(fit(pn, self.th), 'context net -> GRU h0 (PCA)'),
                                  titled(fit(pi, self.th), 'context inp (PCA)')])
        self.place_frame(0, 0, src)
        return vstack([top, bottom])


# ----------------------------------------------------------------------------------------------- main loop
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=None, help="checkpoint: 'E003', 'E003@30000' or a .pth path (default: newest)")
    p.add_argument('--stage', default='dog', choices=['dog', 'gabor', 'cnn'])
    p.add_argument('--sintel', default=None, help='Sintel training scene, e.g. alley_1 (default: random dots)')
    p.add_argument('--pass', dest='dstype', default='clean', choices=['clean', 'final'])
    p.add_argument('--sintel_scale', type=float, default=0.5)
    p.add_argument('--sintel_root', default=DEFAULT_SINTEL_ROOT)
    p.add_argument('--generator', default=GENERATOR)
    p.add_argument('--scene', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--case', default='occluded', choices=list(CASES), help='random-pattern case')
    p.add_argument('--final', action='store_true')
    p.add_argument('--tile_h', type=int, default=240)
    p.add_argument('--iters', type=int, default=12, help='GRU iterations for the predicted flow (12 keeps it live)')
    p.add_argument('--fps', type=float, default=6.0)
    a = p.parse_args()

    ckpt = a.model or latest_checkpoint()
    model = load_model(ckpt).eval()
    view = Viewer(model, a.tile_h, a.iters)
    gen = None if a.sintel else load_generator(a.generator)
    sargs = scene_args(seed=a.seed)
    st = {'stage': a.stage, 'sel': 0, 'shared': True, 'paused': False, 'case': a.case, 'final': a.final,
          'scene': a.scene, 'sintel': a.sintel, 'dstype': a.dstype}

    def load():
        if st['sintel']:
            return sintel_source(a.sintel_root, st['sintel'], st['dstype'], a.sintel_scale)
        return random_dot_source(gen, sargs, st['scene'], st['case'], st['final'])

    src = load()
    t_first = model.ref_offset
    t = t_first
    win = 'feature encoder live'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, view.on_mouse)
    print(f'model {ckpt}; {src.name}', flush=True)
    while True:
        t0 = time.time()
        t_last = len(src.frames) - (model.num_frames - model.ref_offset - 1) - 1
        body = getattr(view, st['stage'])(src, t, st['sel'], st['shared'])
        head = np.full((26, body.shape[1], 3), 24, np.uint8)
        caption(head, f'{paths.describe(ckpt)[1]}   {src.name}   pair {t + 1}->{t + 2}   view: {st["stage"]}   '
                      'keys: d/g/k view, 0-9 [ ] unit, a scale, c/f/n scene, space, s save, q', (8, 18), 0.45)
        view.click_rect = view.click_rect and (view.click_rect[0], view.click_rect[1] + 29) + view.click_rect[2:]
        panel = vstack([head, body])
        cv2.imshow(win, panel)
        wait = max(1, int(1000 / a.fps - (time.time() - t0) * 1000))
        key = cv2.waitKeyEx(0 if st['paused'] else wait)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        left, right = key in (2424832, 65361), key in (2555904, 65363)
        key = key & 0xFF if key != -1 and not (left or right) else -1
        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            st['paused'] = not st['paused']
        elif key in (ord('d'), ord('g'), ord('k')):
            st['stage'] = {'d': 'dog', 'g': 'gabor', 'k': 'cnn'}[chr(key)]
            continue
        elif ord('0') <= key <= ord('9'):
            st['sel'] = key - ord('0')
            continue
        elif key in (ord('['), ord(']')):
            st['sel'] += 1 if key == ord(']') else -1
            continue
        elif key == ord('a'):
            st['shared'] = not st['shared']
            continue
        elif key == ord('s'):
            d = paths.analysis_dir(ckpt, 'live_shots')
            os.makedirs(d, exist_ok=True)
            path = osp.join(d, f'encoder_{st["stage"]}_{src.name.replace(" ", "_").replace(",", "")}_pair{t + 1}.png')
            cv2.imwrite(path, panel)
            print('saved', path, flush=True)
        elif key in (ord('c'), ord('f')) and not st['sintel']:
            if key == ord('c'):
                names = list(CASES)
                st['case'] = names[(names.index(st['case']) + 1) % len(names)]
            else:
                st['final'] ^= True
            src, t = load(), t_first
            print(src.name, flush=True)
            continue
        elif key == ord('f') and st['sintel']:
            st['dstype'] = 'final' if st['dstype'] == 'clean' else 'clean'
            src = load()
            continue
        elif key == ord('n'):
            if st['sintel']:
                st['sintel'] = SINTEL_SCENES[(SINTEL_SCENES.index(st['sintel']) + 1) % len(SINTEL_SCENES)] \
                    if st['sintel'] in SINTEL_SCENES else SINTEL_SCENES[0]
            else:
                st['scene'] += 1
            src, t, view.pixel = load(), t_first, None
            print(src.name, flush=True)
            continue
        step = 1
        if st['paused']:
            if left:
                step = -1
            elif not right:
                continue
        t += step
        if t > t_last:
            t = t_first
        if t < t_first:
            t = t_last
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
