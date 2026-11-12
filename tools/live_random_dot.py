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
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "tools"))

from core.utils import paths  # noqa: E402
from evaluate import load_model  # noqa: E402
# shared helpers live in viz_common.py; re-exported here because other tools import them from this module
from viz_common import (GENERATOR, LayerView, Scene, boundary_band, caption, diverging,  # noqa: E402,F401
                        latest_checkpoint, load_generator, make_clip, predict, render_panel, to_bgr)



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
