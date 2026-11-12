"""Readers/writers for Sintel files (.flo flow, PNG frames and masks)."""
import numpy as np
from PIL import Image

FLO_TAG = 202021.25  # "PIEH" read as a little-endian float32


def read_flo(path):
    """Read a Middlebury .flo file -> float32 array of shape (H, W, 2) holding (u, v)."""
    with open(path, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)
        if tag.size == 0 or tag[0] != FLO_TAG:
            raise ValueError(f'{path}: bad .flo tag')
        w, h = np.fromfile(f, np.int32, count=2)
        data = np.fromfile(f, np.float32, count=2 * w * h)
    return data.reshape(h, w, 2)


def write_flo(path, flow):
    """Write an (H, W, 2) flow array as a Middlebury .flo file."""
    flow = np.asarray(flow, dtype=np.float32)
    assert flow.ndim == 3 and flow.shape[2] == 2, flow.shape
    h, w = flow.shape[:2]
    with open(path, 'wb') as f:
        np.array([FLO_TAG], np.float32).tofile(f)
        np.array([w, h], np.int32).tofile(f)
        flow.tofile(f)


def read_image(path):
    """Read an RGB frame -> uint8 array (H, W, 3)."""
    img = np.array(Image.open(path))
    if img.ndim == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    return img[..., :3]


def read_mask(path):
    """Read a Sintel occlusion/invalid mask (stored as 0/255) -> bool array (H, W)."""
    return np.array(Image.open(path)) > 0
