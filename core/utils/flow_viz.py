"""Middlebury colour coding of optical flow (Baker et al., IJCV 2011).

Python port of Sintel's flow_code/MATLAB/computeColor.m and flowToColor.m.
"""
import numpy as np


def make_colorwheel():
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    wheel = np.zeros((RY + YG + GC + CB + BM + MR, 3))
    col = 0
    wheel[col:col + RY, 0] = 255
    wheel[col:col + RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY
    wheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    wheel[col:col + YG, 1] = 255
    col += YG
    wheel[col:col + GC, 1] = 255
    wheel[col:col + GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC
    wheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    wheel[col:col + CB, 2] = 255
    col += CB
    wheel[col:col + BM, 2] = 255
    wheel[col:col + BM, 0] = np.floor(255 * np.arange(BM) / BM)
    col += BM
    wheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    wheel[col:col + MR, 0] = 255
    return wheel


def flow_to_color(flow, max_flow=None):
    """(H, W, 2) flow -> (H, W, 3) uint8 RGB image. Normalised by max_flow (default: max magnitude)."""
    u, v = flow[..., 0].astype(np.float64), flow[..., 1].astype(np.float64)
    unknown = ~np.isfinite(u) | ~np.isfinite(v) | (np.abs(u) > 1e9) | (np.abs(v) > 1e9)
    u, v = np.where(unknown, 0, u), np.where(unknown, 0, v)

    rad = np.sqrt(u ** 2 + v ** 2)
    if max_flow is None:
        max_flow = rad.max()
    max_flow = max(max_flow, 1e-5)
    u, v, rad = u / max_flow, v / max_flow, rad / max_flow

    wheel = make_colorwheel()
    ncols = wheel.shape[0]
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    img = np.zeros(u.shape + (3,), np.uint8)
    for i in range(3):
        col = (1 - f) * wheel[k0, i] / 255 + f * wheel[k1, i] / 255
        inside = rad <= 1
        col[inside] = 1 - rad[inside] * (1 - col[inside])
        col[~inside] *= 0.75
        img[..., i] = np.floor(255 * col * (~unknown))
    return img
