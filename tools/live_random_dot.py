"""Live inference on random-dot motion scenes: generate a scene, run CustomRAFT on every frame pair, show the result.

    python tools/live_random_dot.py                                     # window, newest experiment's final checkpoint
    python tools/live_random_dot.py --model E001@30000
    python tools/live_random_dot.py --save rds_demo.mp4 --scenes 3   # no window needed: write a video

Scenes come from `Motion Datasets/generate_random_dot.py` (same generator as the RandomDot datasets), rendered
on the fly. The window shows, for the current frame pair:

    row 1:  frame t  |  frame t+1  |  |frame t+1 - frame t|     (what the network is given)
    row 2:  GT flow  |  predicted flow  |  EPE map (red = large)  (flow colours share one scale)

The title strip gives EPE over all pixels, inside the objects, in a band around their boundaries, and on the
background. The network sees the 8-frame clip centred on the pair (model.num_frames / model.ref_offset).

Keys:  space pause/play   n new scene   c camouflaged <-> visible objects   f clean <-> final (pixel noise)
       p full clip <-> pair only (all other frames replaced by the nearest pair frame: tests temporal use)
       l show / hide the spatiotemporal-layer window (DoG ON/OFF + change, Gabor energy, head features)
       <- / -> step one pair when paused   s save a screenshot to <experiment>/analysis/live_shots/<step>/   q or Esc quit
"""
import argparse
import importlib.util
import os
import os.path as osp
import sys
import time
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
from evaluate import load_model  # noqa: E402

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
                  f'objects: {mode(scene.visible, "VISIBLE (+60 grey)", "CAMOUFLAGED")}   '
                  f'pass: {mode(scene.final, "final (noisy)", "clean")}   '
                  f'input: {mode(state["pair_only"], "PAIR ONLY", "full 8-frame clip")}', (8, 18), 0.55)
    s = '   '.join(f'{k} {v:.2f}' for k, v in stats.items())
    caption(head, f'EPE (px)  {s}', (8, 44), 0.6, (120, 220, 255))
    return np.concatenate([head, body], axis=0), stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=None, help="checkpoint: 'E003', 'E003@30000' or a .pth path (default: newest experiment's final)")
    p.add_argument('--generator', default=GENERATOR)
    p.add_argument('--iters', type=int, default=32)
    p.add_argument('--fps', type=float, default=8.0)
    p.add_argument('--scale', type=float, default=1.0, help='display scale')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--first_scene', type=int, default=0)
    p.add_argument('--scenes', type=int, default=0, help='stop after this many scenes (0 = until you quit)')
    p.add_argument('--visible', action='store_true', help='start with visible objects (control)')
    p.add_argument('--final', action='store_true', help='start with the noisy pass')
    p.add_argument('--pair_only', action='store_true', help='start with only the two pair frames as input')
    p.add_argument('--layers', action='store_true', help='also open the spatiotemporal-layer window (key l toggles)')
    p.add_argument('--layer_h', type=int, default=96, help='tile height of the layer window in px')
    p.add_argument('--save', default=None, help='write an .mp4 instead of opening a window')
    p.add_argument('--band', type=int, default=6, help='boundary band half-width in px for the EPE breakdown')
    p.add_argument('--err_scale', type=float, default=4.0, help='EPE (px) that maps to white in the error map')
    # scene parameters (defaults = generate_random_dot.py)
    p.add_argument('--num_frames', type=int, default=48)
    p.add_argument('--height', type=int, default=384)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--dot_size', type=int, default=2)
    p.add_argument('--dot_mode', default='binary', choices=['binary', 'gray'])
    p.add_argument('--density', type=float, default=0.5)
    p.add_argument('--dot_lifetime', type=int, default=0)
    p.add_argument('--shapes', default='all')
    p.add_argument('--max_objects', type=int, default=2)
    p.add_argument('--min_size', type=float, default=48)
    p.add_argument('--max_size', type=float, default=160)
    p.add_argument('--min_speed', type=float, default=1.0)
    p.add_argument('--max_speed', type=float, default=8.0)
    p.add_argument('--static_bg_prob', type=float, default=0.5)
    p.add_argument('--bg_max_speed', type=float, default=3.0)
    p.add_argument('--final_noise', type=float, default=10.0)
    args = p.parse_args()
    args.subpixel, args.object_density, args.object_offset = 1, None, 0.0

    gen = load_generator(args.generator)
    ckpt = args.model or latest_checkpoint()
    model = load_model(ckpt).eval()
    T, ref = model.num_frames, model.ref_offset
    print(f'model {ckpt} (clip {T} frames, ref {ref}); scenes: {args.generator}', flush=True)
    t_first, t_last = ref, args.num_frames - (T - ref - 1) - 1  # pairs whose whole clip lies inside the scene

    state = {'visible': args.visible, 'final': args.final, 'pair_only': args.pair_only, 'paused': False,
             'layers': args.layers}
    layer_view, layer_window = LayerView(model, args.layer_h), 'spatiotemporal layers'
    writer = None
    scene_id, n_done = args.first_scene, 0
    scene = None
    t = t_first
    last_stats = []
    shot_dir = paths.analysis_dir(ckpt, 'live_shots')
    window = 'random-dot live inference'
    if args.save is None:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def new_scene(sid):
        nonlocal scene, t
        scene = Scene(gen, args, sid, state['visible'], state['final'])
        t = t_first
        print(f'scene {sid}: bg v={scene.desc["background_velocity"]}, objects ' +
              ', '.join(f'{o["shape"]} {o["width"]}x{o["height"]} v={o["velocity"]}' for o in scene.desc['objects']),
              flush=True)

    new_scene(scene_id)
    while True:
        t0 = time.time()
        flow_pred = predict(model, scene, t, state['pair_only'], args.iters)
        panel, stats = render_panel(scene, t, flow_pred, args, state)
        last_stats.append(stats)
        if args.scale != 1.0:
            panel = cv2.resize(panel, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

        if args.save is not None:
            if writer is None:
                os.makedirs(osp.dirname(osp.abspath(args.save)), exist_ok=True)
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                                         (panel.shape[1], panel.shape[0]))
            writer.write(panel)
            key = -1
        else:
            cv2.imshow(window, panel)
            if state['layers']:
                cv2.imshow(layer_window, layer_view.render(scene, t, state['pair_only']))
            wait = max(1, int(1000 / args.fps - (time.time() - t0) * 1000))
            key = cv2.waitKeyEx(0 if state['paused'] else wait)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        left, right = key in (2424832, 65361), key in (2555904, 65363)  # arrow keys (Windows / Linux codes)
        key = key & 0xFF if key not in (-1,) and not (left or right) else -1

        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            state['paused'] = not state['paused']
        elif key == ord('s'):
            os.makedirs(shot_dir, exist_ok=True)
            path = osp.join(shot_dir, f'scene{scene.seed}_pair{t + 1}.png')
            cv2.imwrite(path, panel)
            print('saved', path, flush=True)
        elif key in (ord('c'), ord('f')):
            state['visible' if key == ord('c') else 'final'] ^= True
            new_scene(scene.seed)  # same scene, other rendering
            continue
        elif key == ord('p'):
            state['pair_only'] ^= True
            continue
        elif key == ord('l'):
            state['layers'] ^= True
            if not state['layers']:
                cv2.destroyWindow(layer_window)
            continue
        elif key == ord('n'):
            scene_id += 1
            new_scene(scene_id)
            continue

        step = 1
        if state['paused']:
            if left:
                step = -1
            elif not right:
                continue  # paused: redraw the same pair until an arrow key is pressed
        t += step
        if t > t_last or t < t_first:
            if step < 0:
                t = t_first
            else:
                n_done += 1
                if args.scenes and n_done >= args.scenes:
                    break
                scene_id += 1
                new_scene(scene_id)

    if writer is not None:
        writer.release()
        print('wrote', args.save)
    if last_stats:
        keys = last_stats[0].keys()
        print('mean EPE over shown pairs: ' + ', '.join(
            f'{k} {np.nanmean([s[k] for s in last_stats]):.3f}' for k in keys))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
