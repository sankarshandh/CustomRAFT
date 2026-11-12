"""All-pairs cross-correlation between two feature maps (RAFT-style 4D cost volume)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.utils import bilinear_sampler


def correlation_volume(fmap1, fmap2, normalize=True):
    """Inner product of every feature vector in fmap1 with every one in fmap2.

    fmap1: (B, C, H1, W1), fmap2: (B, C, H2, W2) -> (B, H1, W1, H2, W2)
    corr[b, i, j, k, l] = <fmap1[b, :, i, j], fmap2[b, :, k, l]> / sqrt(C)
    Memory is B*H1*W1*H2*W2 floats, so use 1/8-resolution features
    (Sintel at 1/8: 55x128 -> ~50M entries per sample).
    """
    assert fmap1.shape[:2] == fmap2.shape[:2], (fmap1.shape, fmap2.shape)
    corr = torch.einsum('bchw,bcuv->bhwuv', fmap1, fmap2)
    return corr / math.sqrt(fmap1.shape[1]) if normalize else corr


class CorrelationLayer(nn.Module):
    """Parameter-free layer wrapping correlation_volume: (f1, f2) -> (B, H, W, H, W)."""

    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, fmap1, fmap2):
        return correlation_volume(fmap1, fmap2, self.normalize)


class CorrPyramid:
    """Multi-scale pyramid over the 4D volume plus local lookup around a flow estimate (as in RAFT).

    Built once per image pair; call with coords (B, 2, H, W) holding the (x, y) position in
    fmap2 that each fmap1 pixel currently maps to. Returns (B, num_levels * (2r+1)^2, H, W).
    """

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels, self.radius = num_levels, radius
        corr = correlation_volume(fmap1, fmap2)
        b, h1, w1, h2, w2 = corr.shape
        corr = corr.reshape(b * h1 * w1, 1, h2, w2)
        self.pyramid = [corr]
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        b, _, h, w = coords.shape
        coords = coords.permute(0, 2, 3, 1).reshape(b * h * w, 1, 1, 2)
        d = torch.linspace(-r, r, 2 * r + 1, device=coords.device, dtype=coords.dtype)
        dy, dx = torch.meshgrid(d, d, indexing='ij')
        delta = torch.stack([dx, dy], -1)[None]  # (1, 2r+1, 2r+1, 2) as (x, y) offsets

        out = []
        for i, corr in enumerate(self.pyramid):
            sampled = bilinear_sampler(corr, coords / 2 ** i + delta)  # (BHW, 1, 2r+1, 2r+1)
            out.append(sampled.reshape(b, h, w, -1))
        return torch.cat(out, -1).permute(0, 3, 1, 2).contiguous().float()
