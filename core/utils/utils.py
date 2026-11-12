import torch
import torch.nn.functional as F


class InputPadder:
    """Pad images so H and W are divisible by `divis_by` (Sintel is 436x1024; 436 % 8 != 0)."""

    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (-self.ht) % divis_by
        pad_wd = (-self.wd) % divis_by
        if mode == 'sintel':  # split padding between both sides
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        """Pads (B, C, H, W) images or (B, C, T, H, W) clips (T is left untouched)."""
        return [F.pad(x, self._pad + [0, 0] if x.dim() == 5 else self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def coords_grid(batch, ht, wd, device=None):
    """(B, 2, H, W) pixel coordinates, channel 0 = x, channel 1 = y."""
    ys, xs = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing='ij')
    return torch.stack([xs, ys]).float()[None].repeat(batch, 1, 1, 1)


def bilinear_sampler(img, coords):
    """Sample img (N, C, H, W) at pixel coords (N, h, w, 2) given as (x, y); zeros outside."""
    h, w = img.shape[-2:]
    x, y = coords.split(1, dim=-1)
    grid = torch.cat([2 * x / (w - 1) - 1, 2 * y / (h - 1) - 1], -1)
    return F.grid_sample(img, grid, align_corners=True)


def upflow8(flow, mode='bilinear'):
    """Upsample a 1/8-resolution flow field x8, scaling the vectors accordingly."""
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def endpoint_error(pred, gt, mask=None):
    """Mean EPE between (B, 2, H, W) flows, optionally restricted to a (B, H, W) mask."""
    epe = torch.linalg.vector_norm(pred - gt, dim=1)
    if mask is not None:
        mask = mask.bool()
        return epe[mask].mean() if mask.any() else epe.new_tensor(float('nan'))
    return epe.mean()
