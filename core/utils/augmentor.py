"""Training-time augmentation for dense optical flow (follows RAFT's FlowAugmentor).

Works on a frame pair or on a clip of T frames whose flow is defined between frames ref and ref+1.
Spatial transforms are applied identically to every frame, the flow and its masks, with the flow
vectors rescaled / sign-flipped to stay consistent. For clips (T > 2) colour jitter is always shared
across frames: per-frame jitter would add temporal flicker that the temporal filters would respond to.
"""
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import ColorJitter

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class ClipColorJitter:
    """Colour jitter on the GPU, one random draw per clip shared by all its frames.

    Same ranges and random op order as torchvision's ColorJitter(0.4, 0.4, 0.4, 0.5/3.14), which costs
    ~180 ms per 8-frame clip on the CPU and does not scale across DataLoader workers.
    The contrast mean is taken over the whole clip so all frames get the identical transform.
    """

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14):
        self.brightness, self.contrast, self.saturation, self.hue = brightness, contrast, saturation, hue

    @staticmethod
    def _gray(x):  # x (..., 3, H, W) in [0, 1]
        return (0.299 * x[..., 0, :, :] + 0.587 * x[..., 1, :, :] + 0.114 * x[..., 2, :, :]).unsqueeze(-3)

    @torch.no_grad()
    def __call__(self, clips):
        """clips: (B, 3, T, H, W) float in [0, 255] -> jittered copy, same shape and range."""
        out = []
        for clip in clips:
            x = clip.permute(1, 0, 2, 3) / 255.0  # (T, 3, H, W)
            for op in torch.randperm(4).tolist():
                if op == 0:
                    x = (x * np.random.uniform(1 - self.brightness, 1 + self.brightness)).clamp(0, 1)
                elif op == 1:
                    c = np.random.uniform(1 - self.contrast, 1 + self.contrast)
                    x = (c * x + (1 - c) * self._gray(x).mean()).clamp(0, 1)
                elif op == 2:
                    s = np.random.uniform(1 - self.saturation, 1 + self.saturation)
                    x = (s * x + (1 - s) * self._gray(x)).clamp(0, 1)
                else:
                    x = TF.adjust_hue(x, float(np.random.uniform(-self.hue, self.hue)))
            out.append(x.permute(1, 0, 2, 3) * 255.0)
        return torch.stack(out)


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.6, do_flip=True,
                 spatial_aug_prob=0.8, stretch_prob=0.8, max_stretch=0.2,
                 asymmetric_color_aug_prob=0.2, eraser_aug_prob=0.5, color_jitter=True):
        """color_jitter=False skips colour jitter here (apply ClipColorJitter on the GPU instead)."""
        self.crop_size = tuple(crop_size)  # (H, W)
        self.color_jitter = color_jitter
        # scale factor is 2 ** U(min_scale, max_scale)
        self.min_scale, self.max_scale = min_scale, max_scale
        self.spatial_aug_prob = spatial_aug_prob
        self.stretch_prob = stretch_prob
        self.max_stretch = max_stretch
        self.do_flip = do_flip
        self.h_flip_prob, self.v_flip_prob = 0.5, 0.1

        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = asymmetric_color_aug_prob
        self.eraser_aug_prob = eraser_aug_prob

    def color_transform(self, frames):
        if len(frames) == 2 and np.random.rand() < self.asymmetric_color_aug_prob:
            return [np.array(self.photo_aug(Image.fromarray(f)), dtype=np.uint8) for f in frames]
        stack = np.concatenate(frames, axis=0)
        stack = np.array(self.photo_aug(Image.fromarray(stack)), dtype=np.uint8)
        return list(np.split(stack, len(frames), axis=0))

    def eraser_transform(self, frames, ref, bounds=(50, 100)):
        """Paint random rectangles of mean colour over every frame after ref (simulated occlusion)."""
        ht, wd = frames[0].shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(frames[ref + 1].reshape(-1, 3), axis=0)
            rects = [(np.random.randint(0, wd), np.random.randint(0, ht),
                      np.random.randint(*bounds), np.random.randint(*bounds)) for _ in range(np.random.randint(1, 3))]
            frames = list(frames)
            for i in range(ref + 1, len(frames)):
                frames[i] = frames[i].copy()
                for x0, y0, dx, dy in rects:
                    frames[i][y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return frames

    def spatial_transform(self, frames, flow, masks):
        ht, wd = frames[0].shape[:2]
        # never shrink below the crop size
        min_scale = max((self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_x, scale_y = max(scale_x, min_scale), max(scale_y, min_scale)

        if np.random.rand() < self.spatial_aug_prob:
            resize = lambda a, interp: cv2.resize(a, None, fx=scale_x, fy=scale_y, interpolation=interp)
            frames = [resize(f, cv2.INTER_LINEAR) for f in frames]
            flow = resize(flow, cv2.INTER_LINEAR) * np.array([scale_x, scale_y], np.float32)
            masks = [resize(m.astype(np.uint8), cv2.INTER_NEAREST).astype(bool) for m in masks]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:
                frames = [f[:, ::-1] for f in frames]
                flow = flow[:, ::-1] * np.array([-1.0, 1.0], np.float32)
                masks = [m[:, ::-1] for m in masks]
            if np.random.rand() < self.v_flip_prob:
                frames = [f[::-1, :] for f in frames]
                flow = flow[::-1, :] * np.array([1.0, -1.0], np.float32)
                masks = [m[::-1, :] for m in masks]

        y0 = np.random.randint(0, frames[0].shape[0] - self.crop_size[0] + 1)
        x0 = np.random.randint(0, frames[0].shape[1] - self.crop_size[1] + 1)
        crop = (slice(y0, y0 + self.crop_size[0]), slice(x0, x0 + self.crop_size[1]))
        return [f[crop] for f in frames], flow[crop], [m[crop] for m in masks]

    def augment_clip(self, frames, flow, masks=(), ref=0):
        """frames: list of uint8 (H, W, 3); flow: float32 (H, W, 2) for frames[ref] -> frames[ref+1];
        masks: list of bool (H, W)."""
        frames = self.color_transform(list(frames)) if self.color_jitter else list(frames)
        frames = self.eraser_transform(frames, ref)
        frames, flow, masks = self.spatial_transform(frames, flow, list(masks))
        return ([np.ascontiguousarray(f) for f in frames], np.ascontiguousarray(flow),
                [np.ascontiguousarray(m) for m in masks])

    def __call__(self, img1, img2, flow, masks=()):
        """Pair version: img1/img2 uint8 (H, W, 3); flow float32 (H, W, 2); masks list of bool (H, W)."""
        (img1, img2), flow, masks = self.augment_clip([img1, img2], flow, masks, ref=0)
        return img1, img2, flow, masks
