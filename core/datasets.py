"""MPI-Sintel dataset and data loaders for optical-flow training.

Split protocol follows RAFT (princeton-vl/RAFT):
    train      : every consecutive pair of training/{clean,final} (23 scenes, 1041 pairs per pass)
    validation : the same training/ split, unaugmented, each pass evaluated separately
                 (RAFT's evaluate.validate_sintel) - see evaluate.py
    test       : test/{clean,final} (552 pairs per pass, no GT; submission only)
A scene-level hold-out is still available via sintel_train_val(val_scenes=...).

Each training sample is a dict:
    img1, img2 : float32 tensors (3, H, W) in [0, 255]               (num_frames == 2)
    clip       : uint8 tensor (3, T, H, W)                           (num_frames > 2; uint8 keeps the
                 worker -> main-process transfer 4x smaller, convert on the GPU)
    ref_offset : int, the flow is clip[:, ref_offset] -> clip[:, ref_offset + 1]
    flow       : float32 tensor (2, H, W), forward flow in pixels
    valid      : float32 tensor (H, W), 1 where the ground truth may be used
    occ        : float32 tensor (H, W), 1 where the pixel is occluded in the next frame
    scene, frame, dstype : metadata (frame = 1-based filename index of the reference frame)
Test samples contain only the images and the metadata.
"""
import os
import os.path as osp
from glob import glob

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor

DEFAULT_SINTEL_ROOT = os.environ.get('SINTEL_ROOT', r'D:\IISc\Motion Datasets\Sintel')

MAX_FLOW = 1000.0  # flows larger than this are treated as invalid


class MpiSintel(Dataset):
    def __init__(self, root=DEFAULT_SINTEL_ROOT, split='training', dstype='clean',
                 scenes=None, exclude_scenes=None, aug_params=None,
                 num_frames=2, ref_offset=0, edge_mode='replicate'):
        """
        split          : 'training' or 'test'
        dstype         : 'clean' or 'final' (use ConcatDataset / build_sintel for both)
        scenes         : if given, only use these scenes
        exclude_scenes : scenes to skip
        aug_params     : kwargs for FlowAugmentor (must include crop_size); None disables augmentation
        num_frames     : frames per sample; the clip is frames [i - ref_offset, i - ref_offset + num_frames)
                         for the pair (i, i + 1)
        edge_mode      : 'replicate' clamps frame indices at sequence ends so every pair is kept
                         (same pair set as RAFT); 'skip' drops pairs whose clip would leave the sequence
        """
        assert split in ('training', 'test'), split
        assert dstype in ('clean', 'final'), dstype
        assert edge_mode in ('replicate', 'skip'), edge_mode
        assert num_frames >= 2 and 0 <= ref_offset <= num_frames - 2, (num_frames, ref_offset)
        self.root, self.split, self.dstype = root, split, dstype
        self.num_frames, self.ref_offset = num_frames, ref_offset
        self.is_test = split == 'test'
        self.augmentor = FlowAugmentor(**aug_params) if aug_params is not None else None
        self.init_seed = False

        image_root = osp.join(root, split, dstype)
        if not osp.isdir(image_root):
            raise FileNotFoundError(f'Sintel images not found at {image_root}')

        all_scenes = sorted(d for d in os.listdir(image_root) if osp.isdir(osp.join(image_root, d)))
        if scenes is not None:
            missing = set(scenes) - set(all_scenes)
            if missing:
                raise ValueError(f'unknown scenes for {split}/{dstype}: {sorted(missing)}')
            all_scenes = [s for s in all_scenes if s in set(scenes)]
        if exclude_scenes:
            all_scenes = [s for s in all_scenes if s not in set(exclude_scenes)]
        self.scenes = all_scenes

        # samples: (frame paths, flow, occ, invalid, scene, frame); GT paths are None for test
        self.samples = []
        for scene in self.scenes:
            images = sorted(glob(osp.join(image_root, scene, 'frame_*.png')))
            n = len(images)
            for i in range(n - 1):
                idx = np.arange(num_frames) + i - ref_offset
                if edge_mode == 'skip' and (idx[0] < 0 or idx[-1] > n - 1):
                    continue
                frames = [images[j] for j in np.clip(idx, 0, n - 1)]
                frame = int(osp.basename(images[i])[6:10])
                name = osp.splitext(osp.basename(images[i]))[0]
                if self.is_test:
                    gt = (None, None, None)
                else:
                    gt = (osp.join(root, split, 'flow', scene, name + '.flo'),
                          osp.join(root, split, 'occlusions', scene, name + '.png'),
                          osp.join(root, split, 'invalid', scene, name + '.png'))
                self.samples.append((frames, *gt, scene, frame))

    def __len__(self):
        return len(self.samples)

    def _seed_worker(self):
        # give each DataLoader worker a distinct numpy RNG stream for augmentation
        if not self.init_seed:
            info = torch.utils.data.get_worker_info()
            if info is not None:
                np.random.seed(info.seed % 2 ** 32)
            self.init_seed = True

    def _pack_frames(self, frames):
        if self.num_frames == 2 and self.ref_offset == 0:
            return {'img1': _to_chw(frames[0]), 'img2': _to_chw(frames[1])}
        clip = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2)  # (3, T, H, W) uint8
        return {'clip': clip, 'ref_offset': self.ref_offset}

    def __getitem__(self, index):
        self._seed_worker()
        frame_paths, flow_p, occ_p, inv_p, scene, frame = self.samples[index]
        cache = {}
        frames = [cache.setdefault(p, frame_utils.read_image(p)) for p in frame_paths]  # replicated edges read once
        meta = {'scene': scene, 'frame': frame, 'dstype': self.dstype}

        if self.is_test:
            return {**self._pack_frames(frames), **meta}

        flow = frame_utils.read_flo(flow_p)
        occ = frame_utils.read_mask(occ_p)
        valid = ~frame_utils.read_mask(inv_p)

        if self.augmentor is not None:
            frames, flow, (valid, occ) = self.augmentor.augment_clip(frames, flow, [valid, occ], self.ref_offset)

        valid = valid & np.isfinite(flow).all(-1) & (np.abs(flow) < MAX_FLOW).all(-1)
        flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            **self._pack_frames(frames),
            'flow': torch.from_numpy(flow).permute(2, 0, 1).float(),
            'valid': torch.from_numpy(valid).float(),
            'occ': torch.from_numpy(occ).float(),
            **meta,
        }


def _to_chw(img):
    return torch.from_numpy(img).permute(2, 0, 1).float()


def _dstypes(dstype):
    return ('clean', 'final') if dstype == 'both' else (dstype,)


def build_sintel(root=DEFAULT_SINTEL_ROOT, split='training', dstype='both', **kwargs):
    """MpiSintel over one pass, or a ConcatDataset of clean + final when dstype='both'."""
    parts = [MpiSintel(root, split=split, dstype=d, **kwargs) for d in _dstypes(dstype)]
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def sintel_train_val(root=DEFAULT_SINTEL_ROOT, dstype='both', val_scenes=None,
                     crop_size=(368, 768), aug_params=None, **clip_kwargs):
    """Train / val datasets. Train is augmented and cropped, val is full resolution and unaugmented.

    val_scenes=None follows RAFT: train on all of training/, validate on all of training/.
    Passing scene names instead holds those scenes out of training and validates on them only.
    clip_kwargs (num_frames, ref_offset, edge_mode) are forwarded to MpiSintel.
    """
    if aug_params is None:
        # RAFT's Sintel-stage augmentation
        aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
    train = build_sintel(root, 'training', dstype, exclude_scenes=val_scenes, aug_params=aug_params, **clip_kwargs)
    val = build_sintel(root, 'training', dstype, scenes=val_scenes, **clip_kwargs)
    return train, val


def single_thread_worker(_):
    # one intra-op thread per DataLoader worker: many workers x many threads oversubscribes the CPU
    torch.set_num_threads(1)


RDS_GENERATOR = os.environ.get('RDS_GENERATOR', r'D:\IISc\Motion Datasets\generate_random_dot.py')


class RandomDotAsymmetric(Dataset):
    """Asymmetric-view training pairs rendered on the fly (train.py --asym_prob; kb/discussions/E003-D4).

    One random-dot scene layout (generate_random_dot.make_scene: shapes, positions, velocities) is rendered twice
    with independent dots: view A with VISIBLE objects (+60 grey), view B CAMOUFLAGED. The two views share only the
    layout and the motion, so in B the outlines exist only as motion boundaries. Each sample returns
        clip   (3, T, H, W) float 0..255, the view that feeds fmap1 (and the context encoder)
        clip2  (3, T, H, W) the other view, feeds fmap2
        flow   (2, H, W) ground truth for the pair (ref, ref + 1); valid (H, W) all ones
    Which view feeds fmap1 is random (50/50), so 'fmap2 is camouflaged' is not a usable shortcut. Item i is
    deterministic (seeded by i); the scene statistics are the generator's defaults (50% moving background)."""

    def __init__(self, num_frames, ref_offset, crop_size=(368, 768), length=10 ** 7, seed=0,
                 generator=RDS_GENERATOR):
        from types import SimpleNamespace
        self.generator, self._gen = generator, None  # module loaded lazily: loader workers can't pickle it
        self.T, self.ref, self.length, self.seed = num_frames, ref_offset, length, seed
        self.args = SimpleNamespace(num_frames=num_frames, height=crop_size[0], width=crop_size[1], dot_size=2,
                                    dot_mode='binary', density=0.5, object_density=None, object_offset=0.0,
                                    dot_lifetime=0, object_type='solid', shapes='all', max_objects=2, min_size=48,
                                    max_size=160, min_speed=1.0, max_speed=8.0, static_bg_prob=0.5,
                                    bg_max_speed=3.0, subpixel=1)

    def __len__(self):
        return self.length

    @property
    def gen(self):
        if self._gen is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location('generate_random_dot', self.generator)
            self._gen = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self._gen)
        return self._gen

    def __getstate__(self):
        return {**self.__dict__, '_gen': None}

    def _render(self, desc, tex_seed, offset):
        a = vars(self.args) | {'object_offset': offset}
        from types import SimpleNamespace
        frames, labels, flows, _ = self.gen.render_scene(np.random.default_rng(tex_seed), desc, SimpleNamespace(**a))
        return frames, flows

    def __getitem__(self, i):
        desc = self.gen.make_scene(np.random.default_rng([self.seed, i]), self.args)
        # make_scene centres positions on frame T // 2; the pair is (ref, ref + 1)
        a, flows = self._render(desc, [self.seed, i, 1], 60.0)
        b, _ = self._render(desc, [self.seed, i, 2], 0.0)
        if np.random.default_rng([self.seed, i, 3]).random() < 0.5:
            a, b = b, a
        to_clip = lambda f: torch.from_numpy(np.ascontiguousarray(f)).float()[None].expand(3, -1, -1, -1).clone()
        flow = torch.from_numpy(flows[self.ref]).permute(2, 0, 1).float()
        return {'clip': to_clip(a), 'clip2': to_clip(b), 'flow': flow, 'valid': torch.ones(flow.shape[1:])}


def fetch_dataloaders(root=DEFAULT_SINTEL_ROOT, dstype='both', val_scenes=None,
                      crop_size=(368, 768), batch_size=6, num_workers=4, aug_params=None, **clip_kwargs):
    """Return (train_loader, val_loader). val_loader uses batch size 1 (full 436x1024 frames)."""
    train, val = sintel_train_val(root, dstype, val_scenes, crop_size, aug_params, **clip_kwargs)
    extra = {}
    if num_workers > 0:
        extra = {'persistent_workers': True, 'prefetch_factor': 4, 'worker_init_fn': single_thread_worker}
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True, **extra)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, **extra)
    print(f'Sintel ({dstype}): {len(train)} train pairs, {len(val)} val pairs')
    return train_loader, val_loader
