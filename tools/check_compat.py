"""Guard against breaking past experiments: every experiment must still build from its config.json, load its
checkpoint strictly, and give the same outputs as when its reference was recorded.

    python tools/check_compat.py                     # check every experiment that has compat.pt
    python tools/check_compat.py --make E004         # record compat.pt for a new experiment (after training)
    python tools/check_compat.py --make E001 E002 --code_root D:/IISc/CustomRAFT   # reference from another code tree

The reference (experiments/<id>/compat.pt) holds fmap1, fmap2 and the 4-iteration flow of the experiment's
reference checkpoint on a fixed random 8-frame clip (128 x 192; smaller clips make the 4-level correlation pyramid degenerate -> NaN flow), computed on the CPU in float32 so it is deterministic.
Run this before committing any change to core/. A mismatch means a past experiment would no longer reproduce:
put the new behaviour behind a config option whose default is the old behaviour.
"""
import argparse
import glob
import json
import os.path as osp
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
EXPERIMENTS = osp.join(ROOT, 'experiments')
TOL = 1e-5  # relative max-abs difference; CPU float32 is deterministic, so a pass is normally exactly 0


def reference_checkpoint(d):
    """final.pth if present, else the highest step."""
    ck = osp.join(d, 'checkpoints')
    if osp.exists(osp.join(ck, 'final.pth')):
        return osp.join(ck, 'final.pth')
    steps = sorted(int(osp.basename(f)[:-4]) for f in glob.glob(osp.join(ck, '*.pth')) if osp.basename(f)[:-4].isdigit())
    return osp.join(ck, f'{steps[-1]}.pth') if steps else None


def outputs(torch, CustomRAFT, cfg_args, ckpt_path):
    model = CustomRAFT(temporal_mode=cfg_args.get('temporal_mode', 'free'), dropout=0.0,
                       dog_gain_mode=cfg_args.get('dog_gain_mode', 'learned'))
    sd = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=True)
    model.eval()
    g = torch.Generator().manual_seed(0)
    clip = torch.rand(1, 3, model.num_frames, 128, 192, generator=g) * 255
    with torch.no_grad():
        f1, f2 = model.fnet(2 * clip / 255 - 1)
        _, flow = model(clip, iters=4, test_mode=True)
    out = {'fmap1': f1, 'fmap2': f2, 'flow': flow}
    bad = [k for k, v in out.items() if not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(f'non-finite outputs {bad} for {ckpt_path}')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--make', nargs='+', metavar='ID', help='record compat.pt for these experiments')
    p.add_argument('--code_root', default=ROOT, help='code tree whose core/ is used (default: this project)')
    a = p.parse_args()
    sys.path.insert(0, a.code_root)
    import torch
    from core.custom_raft import CustomRAFT
    print(f'code: {osp.abspath(sys.modules["core"].__file__)}')

    dirs = sorted(glob.glob(osp.join(EXPERIMENTS, 'E[0-9][0-9][0-9]-*')))
    if a.make:
        dirs = [d for d in dirs if osp.basename(d)[:4] in {m[:4] for m in a.make}]
    failed = 0
    for d in dirs:
        name, ref_file = osp.basename(d), osp.join(d, 'compat.pt')
        ckpt = reference_checkpoint(d)
        if ckpt is None or not osp.exists(osp.join(d, 'config.json')):
            print(f'  {name:24s} skip (no checkpoint or config.json)')
            continue
        cfg_args = json.load(open(osp.join(d, 'config.json')))['args']
        if a.make:
            out = outputs(torch, CustomRAFT, cfg_args, ckpt)
            torch.save({'checkpoint': osp.relpath(ckpt, d), 'code_root': osp.abspath(a.code_root), **out}, ref_file)
            print(f'  {name:24s} recorded compat.pt from {osp.relpath(ckpt, d)}')
            continue
        if not osp.exists(ref_file):
            print(f'  {name:24s} skip (no compat.pt)')
            continue
        ref = torch.load(ref_file, map_location='cpu')
        try:
            out = outputs(torch, CustomRAFT, cfg_args, osp.join(d, ref['checkpoint']))
        except Exception as e:  # noqa: BLE001  (build / load failure is a compatibility failure)
            print(f'  {name:24s} FAIL: {type(e).__name__}: {e}')
            failed += 1
            continue
        diffs = {k: ((out[k] - ref[k]).abs().max() / ref[k].abs().max().clamp_min(1e-12)).item() for k in out}
        ok = all(v <= TOL for v in diffs.values())
        failed += not ok
        print(f'  {name:24s} {"ok  " if ok else "FAIL"} ' + '  '.join(f'{k} {v:.1e}' for k, v in diffs.items()))
    if not a.make:
        print('all experiments reproduce' if failed == 0 else f'{failed} experiment(s) do NOT reproduce')
        sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
