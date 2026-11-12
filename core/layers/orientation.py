"""Orientation- and direction-tuned layer built from learnable spatiotemporal Gabor filters.

Neuron n is a Gabor filter oriented at theta_n that drifts at speed v_n along theta_n:
    x' =  x cos(theta) + y sin(theta)          (axis the filter is tuned to)
    y' = -x sin(theta) + y cos(theta)
    envelope = exp(-x'^2 / 2 sigma^2 - y'^2 / 2 (aspect sigma)^2 - t^2 / 2 sigma_t^2)
    even / odd = envelope * cos / sin(2 pi f (x' - v t))
With temporal_size > 1 and v != 0, the phase-invariant energy sqrt(even^2 + odd^2) is
direction selective (Adelson-Bergen motion energy): it prefers motion along +theta over -theta.
With temporal_size == 1 (or a single frame) it reduces to an orientation-tuned (static) filter.

Learnable per neuron: theta (direction), f (spatial frequency), sigma (spatial resolution /
receptive-field size), aspect (elongation), v (preferred speed), sigma_t, and input-channel mixing.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layers.spatiotemporal import as_5d, bounded, inv_bounded


class OrientationTunedLayer(nn.Module):
    """Bank of num_orientations x num_scales learnable Gabor neurons.

    Input : (B, in_channels, T, H, W) or (B, in_channels, H, W).
    Output: mode='energy'     -> (B, N, T', H', W')  phase-invariant energy
            mode='quadrature' -> (B, 2N, T', H', W') even responses [0:N], odd responses [N:2N]
    where N = num_orientations * num_scales, H' = ceil(H / stride), and T' = T with
    temporal_padding='same' (replicate) or T - temporal_size + 1 with 'valid'.
    theta is measured in image coordinates (x right, y down): theta = 90 deg is downward motion.
    """

    def __init__(self, in_channels, num_orientations=8, num_scales=2, kernel_size=15, temporal_size=3,
                 base_frequency=0.25, init_speed=1.0, max_speed=4.0, stride=1, mode='energy',
                 temporal_padding='same'):
        super().__init__()
        assert kernel_size % 2 == 1 and temporal_size % 2 == 1, 'kernel sizes must be odd'
        assert mode in ('energy', 'quadrature'), mode
        assert temporal_padding in ('same', 'valid'), temporal_padding
        self.temporal_padding = temporal_padding
        self.in_channels, self.kernel_size, self.temporal_size = in_channels, kernel_size, temporal_size
        self.stride, self.mode = stride, mode
        self.num_filters = n = num_orientations * num_scales
        r, rt = kernel_size // 2, temporal_size // 2

        self.f_lo, self.f_hi = 1.0 / kernel_size, 0.5      # cycles / pixel (0.5 = Nyquist)
        self.s_lo, self.s_hi = 0.5, r / 2.0               # spatial sigma in pixels
        self.a_lo, self.a_hi = 0.25, 4.0                  # sigma_y / sigma_x
        self.t_lo, self.t_hi = 0.3, max(rt, 1) * 1.0
        self.max_speed = max_speed

        coords = torch.arange(-r, r + 1, dtype=torch.float32)
        tcoords = torch.arange(-rt, rt + 1, dtype=torch.float32)
        t, y, x = torch.meshgrid(tcoords, coords, coords, indexing='ij')
        self.register_buffer('grid', torch.stack([t, y, x]), persistent=False)  # (3, Kt, K, K)

        # directions span the full circle when there is a temporal axis (theta and theta + pi differ),
        # otherwise half of it (a static Gabor at theta + pi is the same orientation)
        span = 2 * math.pi if temporal_size > 1 else math.pi
        theta = torch.arange(num_orientations, dtype=torch.float32) * span / num_orientations
        freq = base_frequency / 2.0 ** torch.arange(num_scales, dtype=torch.float32)
        theta, freq = theta.repeat(num_scales), freq.repeat_interleave(num_orientations)
        freq = freq.clamp(self.f_lo * 1.05, self.f_hi * 0.95)
        sigma = (0.56 / freq).clamp(self.s_lo * 1.1, self.s_hi * 0.9)  # ~1 octave bandwidth

        self.theta = nn.Parameter(theta)  # periodic, left unconstrained
        self.freq_raw = nn.Parameter(inv_bounded(freq, self.f_lo, self.f_hi))
        self.sigma_raw = nn.Parameter(inv_bounded(sigma, self.s_lo, self.s_hi))
        self.aspect_raw = nn.Parameter(inv_bounded(torch.ones(n), self.a_lo, self.a_hi))
        self.speed_raw = nn.Parameter(inv_bounded(torch.full((n,), float(init_speed)), -max_speed, max_speed))
        self.sigma_t_raw = nn.Parameter(inv_bounded(torch.full((n,), 1.0).clamp(self.t_lo + 0.05, self.t_hi * 0.9),
                                                    self.t_lo, self.t_hi))
        self.channel_mix = nn.Parameter(torch.full((n, in_channels), 1.0 / in_channels))

    @property
    def frequency(self):
        return bounded(self.freq_raw, self.f_lo, self.f_hi)

    @property
    def sigma(self):
        return bounded(self.sigma_raw, self.s_lo, self.s_hi)

    @property
    def aspect(self):
        return bounded(self.aspect_raw, self.a_lo, self.a_hi)

    @property
    def speed(self):
        return bounded(self.speed_raw, -self.max_speed, self.max_speed)

    @property
    def sigma_t(self):
        return bounded(self.sigma_t_raw, self.t_lo, self.t_hi)

    def kernels(self):
        """Even and odd Gabor kernels, each (N, Kt, K, K), zero-mean and L2-normalised."""
        t, y, x = (g[None] for g in self.grid)
        p = lambda v: v[:, None, None, None]
        cos, sin = p(torch.cos(self.theta)), p(torch.sin(self.theta))
        xr, yr = x * cos + y * sin, -x * sin + y * cos
        sigma = p(self.sigma)
        env = torch.exp(-xr ** 2 / (2 * sigma ** 2) - yr ** 2 / (2 * (p(self.aspect) * sigma) ** 2))
        if self.temporal_size > 1:
            env = env * torch.exp(-t ** 2 / (2 * p(self.sigma_t) ** 2))
        phase = 2 * math.pi * p(self.frequency) * (xr - p(self.speed) * t)
        even, odd = env * torch.cos(phase), env * torch.sin(phase)
        # remove DC so the filters are band-pass, then equalise energy across neurons
        env_sum = env.sum((1, 2, 3), keepdim=True)
        even = even - env * even.sum((1, 2, 3), keepdim=True) / env_sum
        odd = odd - env * odd.sum((1, 2, 3), keepdim=True) / env_sum
        norm = even.pow(2).sum((1, 2, 3), keepdim=True).add(odd.pow(2).sum((1, 2, 3), keepdim=True)).sqrt() / math.sqrt(2)
        return even / norm, odd / norm

    def forward(self, x):
        x, squeeze_t = as_5d(x)
        assert x.shape[1] == self.in_channels, f'expected {self.in_channels} channels, got {x.shape[1]}'
        even, odd = self.kernels()
        mix = self.channel_mix[:, :, None, None, None]
        weight = torch.cat([even[:, None] * mix, odd[:, None] * mix], 0)  # (2N, C, Kt, K, K)
        r = self.kernel_size // 2
        rt = self.temporal_size // 2 if self.temporal_padding == 'same' else 0
        x = F.pad(x, (r, r, r, r, rt, rt), mode='replicate')
        resp = F.conv3d(x, weight, stride=(1, self.stride, self.stride))
        if self.mode == 'energy':
            e, o = resp.chunk(2, 1)
            resp = torch.sqrt(e ** 2 + o ** 2 + 1e-6)
        return resp.squeeze(2) if squeeze_t else resp

    def extra_repr(self):
        return (f'in_channels={self.in_channels}, num_filters={self.num_filters}, kernel_size={self.kernel_size}, '
                f'temporal_size={self.temporal_size}, stride={self.stride}, mode={self.mode}, '
                f'temporal_padding={self.temporal_padding}')
