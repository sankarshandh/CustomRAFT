"""Sanity-check the Sintel loaders.

    python tools/check_sintel.py [--root PATH] [--out OUT_DIR] [--batches N]

Checks dataset sizes and tensor shapes, verifies that flow is consistent with the frames
(backward-warping img2 by the GT flow should reproduce img1 on non-occluded pixels, both
before and after augmentation), times a few DataLoader batches and saves visualisations.
"""
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from core.datasets import (DEFAULT_SINTEL_ROOT, build_sintel, fetch_dataloaders,  # noqa: E402
                           sintel_train_val)
from core.utils.flow_viz import flow_to_color  # noqa: E402


def backward_warp(img, flow):
    """Sample img (B,3,H,W) at x + flow(x)."""
    _, _, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32),
                            indexing='ij')
    gx = (xs + flow[:, 0]) / (w - 1) * 2 - 1
    gy = (ys + flow[:, 1]) / (h - 1) * 2 - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1), align_corners=True)


def warp_error(sample):
    """Mean |img1 - warp(img2, flow)| on visible pixels, vs. the zero-flow baseline."""
    img1, img2 = sample['img1'][None], sample['img2'][None]
    flow = sample['flow'][None]
    mask = (sample['valid'] > 0) & (sample['occ'] == 0)
    # also drop pixels whose target lands outside img2
    h, w = img1.shape[-2:]
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    tx, ty = xs + flow[0, 0], ys + flow[0, 1]
    mask &= (tx >= 0) & (tx <= w - 1) & (ty >= 0) & (ty <= h - 1)
    warped = (img1 - backward_warp(img2, flow)).abs().mean(1)[0]
    zero = (img1 - img2).abs().mean(1)[0]
    return warped[mask].mean().item(), zero[mask].mean().item()


def save_panel(sample, path):
    img1 = sample['img1'].permute(1, 2, 0).numpy().astype(np.uint8)
    img2 = sample['img2'].permute(1, 2, 0).numpy().astype(np.uint8)
    flow = flow_to_color(sample['flow'].permute(1, 2, 0).numpy())
    masks = np.stack([sample['occ'].numpy() * 255, (1 - sample['valid'].numpy()) * 255,
                      np.zeros_like(sample['occ'].numpy())], -1).astype(np.uint8)
    panel = np.concatenate([np.concatenate([img1, img2], 1), np.concatenate([flow, masks], 1)], 0)
    Image.fromarray(panel).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=DEFAULT_SINTEL_ROOT)
    p.add_argument('--out', default=osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'scratch', 'check_out'))
    p.add_argument('--batches', type=int, default=5)
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    # sizes
    for split in ('training', 'test'):
        for d in ('clean', 'final'):
            ds = build_sintel(args.root, split, d)
            print(f'{split:8s} {d:5s}: {len(ds):5d} pairs, {len(ds.scenes)} scenes')
    train, val = sintel_train_val(args.root, 'both')
    print(f'train/val (both passes, RAFT protocol: val = all of training/): {len(train)} / {len(val)}')
    assert len(train) == len(val) == 2 * 1041

    # raw sample
    raw = build_sintel(args.root, 'training', 'final')[0]
    print('raw sample:', {k: (tuple(v.shape), v.dtype) if torch.is_tensor(v) else v for k, v in raw.items()})
    print(f'  flow range [{raw["flow"].min():.1f}, {raw["flow"].max():.1f}], '
          f'occluded {raw["occ"].mean():.1%}, invalid {1 - raw["valid"].mean():.1%}')
    we, ze = warp_error(build_sintel(args.root, 'training', 'clean')[0])
    print(f'  photometric error (clean): warped {we:.2f} vs zero-flow {ze:.2f}')
    assert we < ze, 'flow does not align frames'
    save_panel(raw, osp.join(args.out, 'raw.png'))

    # augmented samples: flow must still align the (cropped / scaled / flipped) frames
    clean_aug, _ = sintel_train_val(args.root, 'clean')
    ratios = []
    for i in np.random.choice(len(clean_aug), 8, replace=False):
        s = clean_aug[int(i)]
        assert s['img1'].shape == (3, 368, 768) and s['flow'].shape == (2, 368, 768), s['img1'].shape
        we, ze = warp_error(s)
        ratios.append(we / max(ze, 1e-6))
    save_panel(s, osp.join(args.out, 'augmented.png'))
    print(f'augmented: warped/zero-flow error ratio median {np.median(ratios):.2f} (expect << 1, colour jitter adds some)')

    # clips (temporal-window model): 8 frames, flow between clip frames 3 and 4
    for edge_mode, expect in (('replicate', 1041), ('skip', 1041 - 23 * 6)):
        ds = build_sintel(args.root, 'training', 'clean', num_frames=8, ref_offset=3, edge_mode=edge_mode)
        print(f'8-frame clips ({edge_mode}): {len(ds)} pairs')
        assert len(ds) == expect, len(ds)
    c = build_sintel(args.root, 'training', 'clean', num_frames=8, ref_offset=3)[10]
    pair = {'img1': c['clip'][:, 3].float(), 'img2': c['clip'][:, 4].float(), 'flow': c['flow'], 'valid': c['valid'], 'occ': c['occ']}
    we, ze = warp_error(pair)
    print(f'  clip {tuple(c["clip"].shape)}: warp error between ref frames {we:.2f} vs zero-flow {ze:.2f}')
    assert we < ze
    clip_aug, _ = sintel_train_val(args.root, 'clean', num_frames=8, ref_offset=3)
    c = clip_aug[10]
    assert c['clip'].shape == (3, 8, 368, 768), c['clip'].shape
    pair = {'img1': c['clip'][:, 3].float(), 'img2': c['clip'][:, 4].float(), 'flow': c['flow'], 'valid': c['valid'], 'occ': c['occ']}
    we, ze = warp_error(pair)
    print(f'  augmented clip {tuple(c["clip"].shape)}: warp error {we:.2f} vs zero-flow {ze:.2f}')

    # loader throughput
    train_loader, val_loader = fetch_dataloaders(args.root, 'both', num_workers=args.num_workers)
    t0 = time.time()
    for i, batch in enumerate(train_loader):
        if i == 0:
            print('train batch:', {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)})
        if i + 1 >= args.batches:
            break
    print(f'{args.batches} train batches in {time.time() - t0:.1f}s')
    vb = next(iter(val_loader))
    print('val batch:', {k: tuple(v.shape) for k, v in vb.items() if torch.is_tensor(v)}, vb['scene'])
    print(f'visualisations saved to {args.out}')


if __name__ == '__main__':  # required on Windows when num_workers > 0
    main()
