"""Train CustomRAFT on MPI-Sintel (RAFT split: all of training/ for training, training/ for validation).

Experiment workflow (see CLAUDE.md):
    python tools/new_experiment.py E004-name --parent E003 --set key=value   # creates experiments/E004-name/
    git commit ...
    python train.py --exp experiments/E004-name                              # args from its config.json
    python train.py --exp experiments/E004-name --name smoke --num_steps 300 # CLI flags override the config
With --exp, everything is written inside the experiment folder:
    runs/<name>/        train.log + train.err (console is copied there), train_metrics.csv, val_metrics.jsonl,
                        args.json, filters/, git_commit.txt, git_diff.patch, code_snapshot/ (read-only code copy)
    checkpoints/        <step>.pth and final.pth (only for the main run, i.e. --name equal to the experiment name)
Without --exp the old behaviour is kept (runs/<name>/, checkpoints/<step>_<name>.pth).

Adapted from RAFT's train.py (sequence loss, AdamW + OneCycleLR, gradient clipping, AMP). Additions:
  * filter-shape parameters (DoG sigmas / delays / k_t, Gabor theta / frequency / sigma / speed) get their
    own optimiser group with lr x --bio_lr_mult and NO weight decay: their gradients are ~1e-6, and decay
    on the sigmoid-bounded raw parameters would drag every sigma towards the middle of its range;
  * learned filter parameters are logged (filters.jsonl + PNG, TensorBoard if installed) at every validation.
Note: RAFT trains Sintel after Chairs + Things pre-training; here the model is trained from scratch on
Sintel only (about 1k pairs per pass), so expect overfitting and watch the validation EPE.
"""
import argparse
import csv
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from core.custom_raft import CustomRAFT  # noqa: E402
from core.datasets import DEFAULT_SINTEL_ROOT, RandomDotAsymmetric, fetch_dataloaders  # noqa: E402
from core.utils.augmentor import ClipColorJitter  # noqa: E402
from core.utils.filter_log import log_filters  # noqa: E402
from evaluate import validate_sintel  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard not installed: fall back to CSV / JSON logs only
    SummaryWriter = None

MAX_FLOW = 400  # exclude extremely large displacements from the loss (as RAFT)


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """Exponentially weighted L1 over the sequence of flow predictions (RAFT eq. 7)."""
    n_predictions = len(flow_preds)
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    bio = model.bio_parameters()
    bio_ids = {id(p) for p in bio}
    other = [p for p in model.parameters() if id(p) not in bio_ids]
    groups = [{'params': other, 'lr': args.lr, 'weight_decay': args.wdecay},
              {'params': bio, 'lr': args.lr * args.bio_lr_mult, 'weight_decay': 0.0}]
    optimizer = optim.AdamW(groups, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr, args.lr * args.bio_lr_mult], args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


class Logger:
    def __init__(self, log_dir, scheduler, sum_freq):
        self.log_dir, self.scheduler, self.sum_freq = log_dir, scheduler, sum_freq
        self.running, self.total_steps = {}, 0
        self.t_last = time.time()
        self.writer = SummaryWriter(log_dir) if SummaryWriter is not None else None
        self.csv_path = osp.join(log_dir, 'train_metrics.csv')

    def push(self, metrics):
        self.total_steps += 1
        for k, v in metrics.items():
            self.running[k] = self.running.get(k, 0.0) + v
        if self.total_steps % self.sum_freq == 0:
            avg = {k: v / self.sum_freq for k, v in self.running.items()}
            lr = self.scheduler.get_last_lr()
            sec = (time.time() - self.t_last) / self.sum_freq  # includes any validation in this interval
            self.t_last = time.time()
            print(f'[{self.total_steps:6d}, lr {lr[0]:.2e} / bio {lr[1]:.2e}, {sec:.3f} s/step] '
                  + ', '.join(f'{k} {v:.4f}' for k, v in sorted(avg.items())), flush=True)
            self._write({'step': self.total_steps, 'lr': lr[0], 'bio_lr': lr[1], 's_per_step': sec, **avg},
                        prefix='train/')
            self.running = {}

    def write_dict(self, results, prefix='val/'):
        self._write({'step': self.total_steps, **results}, prefix)

    def _write(self, row, prefix):
        if self.writer is not None:
            for k, v in row.items():
                if k != 'step':
                    self.writer.add_scalar(prefix + k, v, row['step'])
        path = self.csv_path if prefix == 'train/' else osp.join(self.log_dir, 'val_metrics.jsonl')
        if prefix == 'train/':
            new = not osp.exists(path)
            with open(path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
                w.writerow(row)
        else:
            with open(path, 'a') as f:
                f.write(json.dumps(row) + '\n')

    def close(self):
        if self.writer is not None:
            self.writer.close()


def save_checkpoint(path, model, args, step):
    torch.save({'model': model.state_dict(), 'args': vars(args), 'step': step}, path)


ROOT = osp.dirname(osp.abspath(__file__))
SNAPSHOT = ('core', 'train.py', 'evaluate.py')


def record_provenance(log_dir):
    """Write the git commit, the uncommitted diff and a read-only code snapshot into the run folder."""
    def git(*a):
        r = subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None

    commit = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain', '--', *SNAPSHOT, 'tools')
    with open(osp.join(log_dir, 'git_commit.txt'), 'w') as f:
        f.write((commit or 'NOT A GIT REPO\n') + ('dirty: yes\n' if dirty else 'dirty: no\n'))
    if dirty:
        print('WARNING: uncommitted code changes; saved to git_diff.patch. Commit before real runs.', flush=True)
        with open(osp.join(log_dir, 'git_diff.patch'), 'w') as f:
            f.write(git('diff', 'HEAD') or '')
    snap = osp.join(log_dir, 'code_snapshot')
    shutil.rmtree(snap, ignore_errors=True)
    for item in SNAPSHOT:
        src = osp.join(ROOT, item)
        if osp.isdir(src):
            shutil.copytree(src, osp.join(snap, item), ignore=shutil.ignore_patterns('__pycache__'))
        else:
            os.makedirs(snap, exist_ok=True)
            shutil.copy2(src, snap)


class Tee:
    """Write a stream to the console and to a file (with --exp: runs/<name>/train.log and train.err)."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, 'a', buffering=1, encoding='utf-8')

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def output_paths(args):
    """(log_dir, checkpoint path for a step or 'final')."""
    if args.exp is None:
        return osp.join('runs', args.name), lambda s: osp.join(
            'checkpoints', f'{args.name}.pth' if s == 'final' else f'{s}_{args.name}.pth')
    log_dir = osp.join(args.exp, 'runs', args.name)
    main_run = args.name == osp.basename(osp.normpath(args.exp))
    ckdir = osp.join(args.exp, 'checkpoints' if main_run else osp.join('runs', args.name, 'checkpoints'))
    return log_dir, lambda s: osp.join(ckdir, f'{s}.pth')


def train(args):
    device = 'cuda'
    model = CustomRAFT(temporal_mode=args.temporal_mode, dropout=args.dropout,
                       mixed_precision=args.mixed_precision, dog_gain_mode=args.dog_gain_mode).to(device)
    print(f'Parameter count: {count_parameters(model)}  (RAFT-small: 990162)')
    print(f'clip: {model.num_frames} frames, flow between clip frames {model.ref_offset} -> {model.ref_offset + 1}')

    if args.restore_ckpt is not None:
        ckpt = torch.load(args.restore_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
    model.train()

    # RAFT's Sintel augmentation; colour jitter runs on the GPU (ClipColorJitter) instead of in the workers
    aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True,
                  'color_jitter': False}
    train_loader, _ = fetch_dataloaders(args.root, 'both', batch_size=args.batch_size, num_workers=args.num_workers,
                                        aug_params=aug_params, num_frames=model.num_frames,
                                        ref_offset=model.ref_offset, edge_mode=args.edge_mode)
    asym_iter = None
    if args.asym_prob > 0:  # asymmetric-view random-dot batches (kb/discussions/E003-D4); off by default
        asym = RandomDotAsymmetric(model.num_frames, model.ref_offset, crop_size=args.image_size, seed=args.seed)
        asym_loader = DataLoader(asym, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                 num_workers=2, prefetch_factor=2, pin_memory=True, persistent_workers=True)
        asym_iter = iter(asym_loader)
        print(f'asymmetric views: p = {args.asym_prob} of the steps use a random-dot asymmetric batch')
    color_jitter = ClipColorJitter()
    optimizer, scheduler = fetch_optimizer(args, model)
    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision)

    log_dir, ckpt_path = output_paths(args)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(osp.dirname(ckpt_path('final')), exist_ok=True)
    with open(osp.join(log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    if args.exp is not None:
        sys.stdout = Tee(sys.stdout, osp.join(log_dir, 'train.log'))
        sys.stderr = Tee(sys.stderr, osp.join(log_dir, 'train.err'))
        record_provenance(log_dir)
    logger = Logger(log_dir, scheduler, args.sum_freq)
    log_filters(model.fnet, 0, osp.join(log_dir, 'filters'), logger.writer)

    total_steps, t0 = 0, time.time()
    while total_steps < args.num_steps:
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            clip2 = None
            if asym_iter is not None and np.random.rand() < args.asym_prob:
                batch = next(asym_iter)  # replaces this Sintel batch
                clip2 = color_jitter(batch['clip2'].to(device, non_blocking=True).float())
            clip = color_jitter(batch['clip'].to(device, non_blocking=True).float())
            flow = batch['flow'].to(device, non_blocking=True)
            valid = batch['valid'].to(device, non_blocking=True)

            if args.add_noise:  # same noise level for the whole clip, independent per frame
                stdv = np.random.uniform(0.0, 5.0)
                clip = (clip + stdv * torch.randn_like(clip)).clamp(0.0, 255.0)

            flow_predictions = model(clip, iters=args.iters, clip2=clip2)
            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics['loss'] = loss.item()
            logger.push(metrics)
            total_steps += 1

            if total_steps % args.val_freq == 0 or total_steps == args.num_steps:
                save_checkpoint(ckpt_path(total_steps), model, args, total_steps)
                results = validate_sintel(model, args.root, iters=args.val_iters, max_pairs=args.val_max_pairs)
                logger.write_dict(results)
                log_filters(model.fnet, total_steps, osp.join(log_dir, 'filters'), logger.writer)
                print(f'step {total_steps}: {(time.time() - t0) / total_steps:.3f} s/step', flush=True)
                model.train()

            if total_steps >= args.num_steps:
                break

    path = ckpt_path('final')
    save_checkpoint(path, model, args, total_steps)
    if args.val_max_pairs is not None:  # final evaluation on the full RAFT validation set
        print('final validation on all of training/ (RAFT protocol)', flush=True)
        results = validate_sintel(model, args.root, iters=args.val_iters)
        logger.write_dict({f'full_{k}': v for k, v in results.items()})
    logger.close()
    return path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default=None,
                        help='experiment folder (experiments/E00N-name): its config.json supplies the defaults and '
                             'all outputs are written inside it')
    parser.add_argument('--name', default='customraft',
                        help='run name (with --exp: defaults to the experiment folder name = the main run)')
    parser.add_argument('--root', default=DEFAULT_SINTEL_ROOT)
    parser.add_argument('--restore_ckpt', help='restore checkpoint')
    parser.add_argument('--temporal_mode', default='free', choices=['free', 'bandpass', 'typed'],
                        help="DoG temporal kernels: 'free' (learnable k_t and delays), 'bandpass' (zero-DC ablation) or "
                             "'typed' (even neurons fixed monophasic, odd neurons fixed biphasic; only the band is learned)")
    parser.add_argument('--dog_gain_mode', default='learned', choices=['learned', 'contrast'],
                        help="DoG output scaling before GELU: 'learned' gain (original) or 'contrast' normalisation "
                             "with a bounded gain, which keeps the ON/OFF GELUs in their nonlinear range")

    # defaults follow RAFT's Sintel stage, except lr (training from scratch here)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--bio_lr_mult', type=float, default=20.0, help='lr multiplier for filter-shape params')
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[368, 768])
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting of the sequence loss')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--asym_prob', type=float, default=0.0,
                        help='probability that a step uses an asymmetric-view random-dot batch instead of Sintel: '
                             'fmap1 and fmap2 from two renderings of one scene that share only layout and motion '
                             '(core/datasets.py RandomDotAsymmetric). 0 = original training')
    parser.add_argument('--edge_mode', default='replicate', choices=['replicate', 'skip'],
                        help="clips at sequence ends: 'replicate' keeps RAFT's full pair set, 'skip' drops them")

    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--sum_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=5000)
    parser.add_argument('--val_iters', type=int, default=32)
    parser.add_argument('--val_max_pairs', type=int, default=200,
                        help='periodic validation on an evenly spaced subset per pass (saves time); the final '
                             'validation always uses all 1041 pairs per pass, as RAFT. Pass 0 for full every time')
    parser.add_argument('--seed', type=int, default=1234)
    return parser


def parse_args(argv=None):
    """CLI > experiment config.json > built-in defaults."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.exp is not None:
        with open(osp.join(args.exp, 'config.json')) as f:
            cfg = json.load(f)
        known = {a.dest for a in parser._actions}
        unknown = set(cfg['args']) - known
        if unknown:
            raise SystemExit(f'config.json has unknown args {sorted(unknown)}')
        parser.set_defaults(**cfg['args'], name=osp.basename(osp.normpath(args.exp)))
        args = parser.parse_args(argv)
    return args


if __name__ == '__main__':  # guard required on Windows for DataLoader workers
    args = parse_args()
    if args.val_max_pairs == 0:
        args.val_max_pairs = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
