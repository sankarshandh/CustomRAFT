"""Evaluate CustomRAFT on Sintel, following RAFT's protocol.

    python evaluate.py --model E003 --dataset sintel              # final checkpoint of E003; training/ split
    python evaluate.py --model E003@30000 --dataset sintel        # a given step (or pass any .pth path)
    python evaluate.py --model E003 --dataset sintel_test         # write a test submission

Validation (RAFT's validate_sintel): every pair of training/, each pass separately, 32 iterations,
EPE averaged over all pixels. Non-occluded / occluded EPE is reported as well.
Clips at the start / end of a sequence are completed by replicating the edge frame.
"""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from core.custom_raft import CustomRAFT  # noqa: E402
from core.datasets import DEFAULT_SINTEL_ROOT, MpiSintel  # noqa: E402
from core.utils import frame_utils  # noqa: E402
from core.utils.paths import resolve_ckpt  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402


def _loader(model, root, split, dstype, scenes=None, num_workers=2):
    ds = MpiSintel(root, split, dstype, scenes=scenes, num_frames=model.num_frames,
                   ref_offset=model.ref_offset, edge_mode='replicate')
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def validate_sintel(model, root=DEFAULT_SINTEL_ROOT, iters=32, scenes=None, max_pairs=None, num_workers=2):
    """Returns {'clean': epe, 'clean_1px': ..., 'clean_noc': ..., 'clean_occ': ..., 'final': ...}."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    results = {}
    for dstype in ('clean', 'final'):
        loader = _loader(model, root, 'training', dstype, scenes, num_workers)
        if max_pairs is not None:  # evenly spaced subset for quick checks during training
            keep = set(np.linspace(0, len(loader.dataset) - 1, max_pairs).astype(int).tolist())
            loader = DataLoader(torch.utils.data.Subset(loader.dataset, sorted(keep)), batch_size=1,
                                num_workers=num_workers)
        epe_all, epe_noc, epe_occ = [], [], []
        for s in loader:
            clip = s['clip'].to(device).float()
            padder = InputPadder(clip.shape)
            (clip,) = padder.pad(clip)
            _, flow_up = model(clip, iters=iters, test_mode=True)
            flow = padder.unpad(flow_up[0]).cpu()
            epe = torch.linalg.vector_norm(flow - s['flow'][0], dim=0)
            occ = s['occ'][0] > 0
            epe_all.append(epe.view(-1).numpy())
            epe_noc.append(epe[~occ].numpy())
            epe_occ.append(epe[occ].numpy())

        e = np.concatenate(epe_all)
        results[dstype] = float(e.mean())
        for px in (1, 3, 5):
            results[f'{dstype}_{px}px'] = float((e < px).mean())
        results[f'{dstype}_noc'] = float(np.concatenate(epe_noc).mean())
        results[f'{dstype}_occ'] = float(np.concatenate(epe_occ).mean())
        print(f'Validation ({dstype}) EPE: {results[dstype]:.4f}, 1px: {results[dstype + "_1px"]:.4f}, '
              f'3px: {results[dstype + "_3px"]:.4f}, 5px: {results[dstype + "_5px"]:.4f}, '
              f'noc: {results[dstype + "_noc"]:.4f}, occ: {results[dstype + "_occ"]:.4f}')
    model.train(was_training)
    return results


@torch.no_grad()
def create_sintel_submission(model, root=DEFAULT_SINTEL_ROOT, iters=32, output_path='sintel_submission'):
    """Write test-split predictions as <output_path>/<pass>/<scene>/frameNNNN.flo (RAFT's layout)."""
    model.eval()
    device = next(model.parameters()).device
    for dstype in ('clean', 'final'):
        for s in _loader(model, root, 'test', dstype):
            clip = s['clip'].to(device).float()
            padder = InputPadder(clip.shape)
            (clip,) = padder.pad(clip)
            _, flow_up = model(clip, iters=iters, test_mode=True)
            flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
            out_dir = osp.join(output_path, dstype, s['scene'][0])
            os.makedirs(out_dir, exist_ok=True)
            frame_utils.write_flo(osp.join(out_dir, 'frame%04d.flo' % int(s['frame'][0])), flow)
    print(f'submission written to {output_path}; bundle it with the Sintel bundler before uploading')


def load_model(path, device='cuda'):
    """path: a .pth file or an experiment reference such as 'E003' or 'E003@100000' (see core/utils/paths.py)."""
    ckpt = torch.load(resolve_ckpt(path), map_location=device)
    margs = ckpt.get('args', {})
    model = CustomRAFT(temporal_mode=margs.get('temporal_mode', 'free'), dropout=0.0,
                       dog_gain_mode=margs.get('dog_gain_mode', 'learned')).to(device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    return model


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', default='sintel', choices=['sintel', 'sintel_test'])
    p.add_argument('--root', default=DEFAULT_SINTEL_ROOT)
    p.add_argument('--iters', type=int, default=32)
    p.add_argument('--max_pairs', type=int, default=None, help='evaluate an evenly spaced subset per pass')
    p.add_argument('--output_path', default='sintel_submission')
    args = p.parse_args()

    model = load_model(args.model)
    if args.dataset == 'sintel':
        validate_sintel(model, args.root, args.iters, max_pairs=args.max_pairs)
    else:
        create_sintel_submission(model, args.root, args.iters, args.output_path)
