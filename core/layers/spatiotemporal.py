"""Spatiotemporal band-pass layer built from learnable difference-of-Gaussian filters.

Each neuron n has a separable kernel  K_n(t, y, x) = DoG_t(t; n) * DoG_s(y, x; n)  with
    DoG_s = G2(sigma_center) - k_s * G2(sigma_surround)                    (spatial centre-surround)
    DoG_t = G1(mu_c, sigma_t_center) - k_t * G1(mu_s, sigma_t_surround)    (temporal)
Every Gaussian is normalised to unit sum, so the DC gain is exactly 1 - k.
Kernels are rebuilt from the parameters on every forward pass, so backprop tunes the sigmas directly.
The response r is split into ON / OFF channels: GELU(r) and GELU(-r).

temporal_mode
    'bandpass' : k_t = 1, mu_c = mu_s = 0, sigma_t_surround > sigma_t_center. Zero-DC, even
                 (second-derivative-like) temporal kernels.
    'free'     : k_t in (0, 1), mu_c, mu_s and both temporal sigmas are learnable per neuron, so a
                 neuron can become an integrator (k_t -> 0, sustained), a band-pass "differentiator"
                 (k_t -> 1), and odd / biphasic (first-derivative-like) when mu_c != mu_s.
                 Initialised with k_t spread over [0.1, 0.95] and every other neuron biphasic,
                 so both kinds exist from step 0.
    'typed'    : the temporal TYPE of every neuron is fixed by construction; training can only tune its band.
                 Even neurons are monophasic: G1(0, sigma_t), all taps >= 0, DC gain 1.
                 Odd neurons are biphasic: G1(-d, sigma_t) - G1(+d, sigma_t), exactly odd, zero DC, one positive
                 and one negative lobe for every parameter value. Learnable: sigma_t (both types) and the lobe
                 separation d in [0.25, mu_max] (biphasic). Kernels are L1-normalised (sum |k| = 1), so a
                 biphasic neuron cannot be silenced by shrinking its kernel.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def bounded(raw, lo, hi):
    """Map an unconstrained parameter into (lo, hi). Unlike clamp, gradients never vanish abruptly."""
    return lo + (hi - lo) * torch.sigmoid(raw)


def inv_bounded(value, lo, hi):
    p = ((torch.as_tensor(value, dtype=torch.float32) - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def gaussian_1d(sigma, coords, mu=None):
    """sigma (N,), coords (K,), optional mean mu (N,) -> (N, K) Gaussians normalised to unit sum."""
    d = coords[None] if mu is None else coords[None] - mu[:, None]
    g = torch.exp(-d ** 2 / (2 * sigma[:, None] ** 2))
    return g / g.sum(-1, keepdim=True)


def gaussian_2d(sigma, coords):
    """Isotropic 2D Gaussians (N, K, K) with unit sum."""
    g = gaussian_1d(sigma, coords)
    return g[:, :, None] * g[:, None, :]


def as_5d(x):
    """Accept (B, C, H, W) or (B, C, T, H, W); return a 5D tensor and whether T was added."""
    if x.dim() == 4:
        return x.unsqueeze(2), True
    if x.dim() == 5:
        return x, False
    raise ValueError(f'expected (B, C, H, W) or (B, C, T, H, W), got {tuple(x.shape)}')


class SpatioTemporalDoG(nn.Module):
    """Bank of learnable spatiotemporal DoG neurons with ON/OFF GELU outputs.

    Input : (B, in_channels, T, H, W) clip, or (B, in_channels, H, W) single frame (T = 1).
    Output: (B, 2 * num_filters, T', H, W) if on_off else (B, num_filters, T', H, W);
            channels [0:N] are ON (GELU(r)), [N:2N] are OFF (GELU(-r)).
            T' = T with temporal_padding='same' (replicate), T - temporal_size + 1 with 'valid'.

    Learnable per neuron: spatial sigma_center / sigma_surround, temporal sigmas (and in 'free'
    mode k_t and the temporal centres), input-channel mixing weights and an output gain.
    sigma_surround is always > sigma_center spatially, and all sigmas stay inside the kernel support.
    """

    def __init__(self, in_channels=3, num_filters=8, spatial_size=15, temporal_size=3,
                 spatial_sigma_range=(0.7, 2.5), surround_ratio=2.0,
                 temporal_sigma_center=0.5, temporal_sigma_surround=1.5,
                 temporal_mode='bandpass', learn_spatial_k=False,
                 temporal_padding='same', on_off=True,
                 gain_mode='learned', contrast_eps=0.005, gain_range=(1.0, 4.0), gain_init=2.0):
        """gain_mode: 'learned' -> r * exp(log_gain) (original; the gain is free to shrink r into GELU's linear range)
                      'contrast' -> divisive contrast normalisation, r / sqrt(mean(r^2) + eps^2) with the mean taken
                      per sample and neuron over the whole output (all T', H, W, so every temporal window is scaled
                      by the same number), then a gain bounded to gain_range. This fixes GELU's operating point:
                      the drive has RMS ~ gain, so ON/OFF really rectify. eps acts as a semi-saturation constant
                      (a transient neuron on a static scene stays quiet instead of amplifying noise)."""
        super().__init__()
        assert gain_mode in ('learned', 'contrast'), gain_mode
        self.gain_mode, self.contrast_eps = gain_mode, contrast_eps
        self.g_lo, self.g_hi = gain_range
        assert spatial_size % 2 == 1 and temporal_size % 2 == 1, 'kernel sizes must be odd'
        assert temporal_mode in ('bandpass', 'free', 'typed'), temporal_mode
        assert temporal_padding in ('same', 'valid'), temporal_padding
        self.in_channels, self.num_filters, self.on_off = in_channels, num_filters, on_off
        self.spatial_size, self.temporal_size = spatial_size, temporal_size
        self.temporal_mode, self.temporal_padding = temporal_mode, temporal_padding
        r, rt = spatial_size // 2, temporal_size // 2

        # sigma bounds: keep the surround within ~2 sigma of the kernel edge
        self.s_lo, self.s_hi = 0.4, r / 2.0
        self.t_lo, self.t_hi = 0.3, max(rt, 1) * 1.0
        self.mu_max = float(max(rt - 1, 1))
        # 'typed': lower bound of the biphasic lobe separation d (lobes at -d and +d frames). Any d > 0 gives an odd
        # kernel; the bound only keeps the unnormalised kernel well away from zero before L1 normalisation.
        self.sep_lo = 0.25

        self.register_buffer('xs', torch.arange(-r, r + 1, dtype=torch.float32), persistent=False)
        self.register_buffer('ts', torch.arange(-rt, rt + 1, dtype=torch.float32), persistent=False)

        # spatial: centre sigmas log-spaced over spatial_sigma_range (multi-scale bank)
        sc = torch.logspace(math.log10(spatial_sigma_range[0]), math.log10(spatial_sigma_range[1]), num_filters)
        sc = sc.clamp(self.s_lo + 0.05, self.s_hi * 0.8)
        ss = (sc * surround_ratio).clamp(max=self.s_hi * 0.98)
        self.spatial_center_raw = nn.Parameter(inv_bounded(sc, self.s_lo, self.s_hi))
        # surround = centre + (s_hi - centre) * sigmoid(raw)  => always larger than the centre
        self.spatial_surround_raw = nn.Parameter(inv_bounded((ss - sc) / (self.s_hi - sc), 0.0, 1.0))
        self.spatial_k_raw = nn.Parameter(torch.full((num_filters,), 3.0)) if learn_spatial_k else None

        tc = torch.full((num_filters,), float(temporal_sigma_center)).clamp(self.t_lo + 0.05, self.t_hi * 0.8)
        tsur = torch.full((num_filters,), float(temporal_sigma_surround)).clamp(max=self.t_hi * 0.98)
        tsur = torch.maximum(tsur, tc + 0.05)
        if temporal_mode == 'bandpass':
            self.temporal_center_raw = nn.Parameter(inv_bounded(tc, self.t_lo, self.t_hi))
            self.temporal_surround_raw = nn.Parameter(inv_bounded((tsur - tc) / (self.t_hi - tc), 0.0, 1.0))
            self.temporal_k_raw = self.temporal_mu_c_raw = self.temporal_mu_s_raw = None
        elif temporal_mode == 'typed':
            # even neurons monophasic, odd neurons biphasic (interleaved, so each type spans the spatial scales)
            self.register_buffer('is_biphasic', torch.arange(num_filters) % 2 == 1, persistent=False)
            sig = torch.where(self.is_biphasic, torch.full_like(tc, 0.8), tc).clamp(self.t_lo + 0.05, self.t_hi * 0.95)
            sep_hi = max(self.mu_max, self.sep_lo + 0.1)
            self.sep_hi = sep_hi
            self.temporal_center_raw = nn.Parameter(inv_bounded(sig, self.t_lo, self.t_hi))
            # d only shapes biphasic kernels; monophasic entries are unused (they get zero gradient)
            self.temporal_sep_raw = nn.Parameter(inv_bounded(torch.full((num_filters,), 0.5), self.sep_lo, sep_hi))
            self.temporal_surround_raw = self.temporal_k_raw = None
            self.temporal_mu_c_raw = self.temporal_mu_s_raw = None
        else:
            # every other neuron starts biphasic: equal-width Gaussians offset by +-0.5 frame (odd kernel)
            odd = torch.arange(num_filters) % 2 == 1
            mu_c = torch.where(odd, -0.5, 0.0)
            mu_s = torch.where(odd, 0.5, 0.0)
            sig_c = torch.where(odd, torch.full_like(tc, 0.8), tc).clamp(self.t_lo + 0.05, self.t_hi * 0.95)
            sig_s = torch.where(odd, sig_c, tsur)
            k_t = torch.linspace(0.1, 0.95, num_filters) if num_filters > 1 else torch.tensor([0.5])
            # independent sigmas (no centre < surround ordering when the centres can move)
            self.temporal_center_raw = nn.Parameter(inv_bounded(sig_c, self.t_lo, self.t_hi))
            self.temporal_surround_raw = nn.Parameter(inv_bounded(sig_s, self.t_lo, self.t_hi))
            self.temporal_k_raw = nn.Parameter(inv_bounded(k_t, 0.0, 1.0))
            self.temporal_mu_c_raw = nn.Parameter(inv_bounded(mu_c, -self.mu_max, self.mu_max))
            self.temporal_mu_s_raw = nn.Parameter(inv_bounded(mu_s, -self.mu_max, self.mu_max))

        self.channel_mix = nn.Parameter(torch.full((num_filters, in_channels), 1.0 / in_channels))
        if gain_mode == 'learned':
            self.log_gain = nn.Parameter(torch.zeros(num_filters))
        else:
            self.gain_raw = nn.Parameter(inv_bounded(torch.full((num_filters,), float(gain_init)), self.g_lo, self.g_hi))

    @property
    def gain(self):
        """Effective per-neuron output gain (multiplies r in 'learned' mode, r_hat in 'contrast' mode)."""
        if self.gain_mode == 'learned':
            return torch.exp(self.log_gain)
        return bounded(self.gain_raw, self.g_lo, self.g_hi)

    # ---- parameters in natural units (for logging / inspection) ---------------------------------
    @property
    def sigma_center(self):
        return bounded(self.spatial_center_raw, self.s_lo, self.s_hi)

    @property
    def sigma_surround(self):
        c = self.sigma_center
        return c + (self.s_hi - c) * torch.sigmoid(self.spatial_surround_raw)

    @property
    def sigma_t_center(self):
        return bounded(self.temporal_center_raw, self.t_lo, self.t_hi)

    @property
    def sigma_t_surround(self):
        if self.temporal_mode == 'free':
            return bounded(self.temporal_surround_raw, self.t_lo, self.t_hi)
        if self.temporal_mode == 'typed':  # both lobes share one width
            return self.sigma_t_center
        c = self.sigma_t_center
        return c + (self.t_hi - c) * torch.sigmoid(self.temporal_surround_raw)

    @property
    def k_spatial(self):
        if self.spatial_k_raw is None:
            return torch.ones(self.num_filters, device=self.xs.device)
        return torch.sigmoid(self.spatial_k_raw)

    @property
    def k_temporal(self):
        if self.temporal_mode == 'typed':  # 1 = biphasic (zero DC), 0 = monophasic (single Gaussian)
            return self.is_biphasic.float()
        if self.temporal_k_raw is None:
            return torch.ones(self.num_filters, device=self.xs.device)
        return torch.sigmoid(self.temporal_k_raw)

    @property
    def temporal_separation(self):
        """'typed' only: biphasic lobe separation d (0 for monophasic neurons); None in other modes."""
        if self.temporal_mode != 'typed':
            return None
        d = bounded(self.temporal_sep_raw, self.sep_lo, self.sep_hi)
        return torch.where(self.is_biphasic, d, torch.zeros_like(d))

    @property
    def mu_t_center(self):
        if self.temporal_mode == 'typed':
            return -self.temporal_separation
        if self.temporal_mu_c_raw is None:
            return torch.zeros(self.num_filters, device=self.xs.device)
        return bounded(self.temporal_mu_c_raw, -self.mu_max, self.mu_max)

    @property
    def mu_t_surround(self):
        if self.temporal_mode == 'typed':
            return self.temporal_separation
        if self.temporal_mu_s_raw is None:
            return torch.zeros(self.num_filters, device=self.xs.device)
        return bounded(self.temporal_mu_s_raw, -self.mu_max, self.mu_max)

    # ---- kernels -------------------------------------------------------------------------------
    def spatial_kernels(self):
        """(N, K, K) spatial DoG kernels."""
        ks = self.k_spatial[:, None, None]
        return gaussian_2d(self.sigma_center, self.xs) - ks * gaussian_2d(self.sigma_surround, self.xs)

    def temporal_kernels(self):
        """(N, Kt) temporal DoG kernels (identity when temporal_size == 1)."""
        if self.temporal_size == 1:
            return torch.ones(self.num_filters, 1, device=self.xs.device)
        if self.temporal_mode == 'typed':
            # monophasic: G(0, s). biphasic: G(-d, s) - G(+d, s), built as g - flip(g) so it is EXACTLY odd
            # (zero DC, zero centre tap) in floating point too. Then fixed amplitude: sum |k| = 1.
            g = gaussian_1d(self.sigma_t_center, self.ts, self.mu_t_center)
            k = torch.where(self.is_biphasic[:, None], g - g.flip(-1), g)
            return k / k.abs().sum(-1, keepdim=True).clamp_min(1e-8)
        kt = self.k_temporal[:, None]
        mu_c = mu_s = None
        if self.temporal_mode == 'free':
            mu_c, mu_s = self.mu_t_center, self.mu_t_surround
        return (gaussian_1d(self.sigma_t_center, self.ts, mu_c)
                - kt * gaussian_1d(self.sigma_t_surround, self.ts, mu_s))

    def temporal_profile(self):
        """Per-neuron temporal DC gain (1 - k_t; 1 = integrator, 0 = band-pass) and
        odd/even energy ratio ||K_odd|| / ||K_even|| (large = first-derivative-like)."""
        k = self.temporal_kernels()
        even, odd = (k + k.flip(-1)) / 2, (k - k.flip(-1)) / 2
        ratio = odd.norm(dim=-1) / even.norm(dim=-1).clamp_min(1e-8)
        return k.sum(-1), ratio

    def temporal_odd_fraction(self):
        """Share of each temporal kernel's energy in its odd part, ||K_odd||^2 / ||K||^2: 0 = even, 1 = odd.
        Bounded, unlike the odd/even ratio, which explodes for exactly odd kernels."""
        k = self.temporal_kernels()
        odd = (k - k.flip(-1)) / 2
        return odd.pow(2).sum(-1) / k.pow(2).sum(-1).clamp_min(1e-12)

    def kernels(self):
        """(N, Kt, K, K) full spatiotemporal kernels (before channel mixing)."""
        return self.temporal_kernels()[:, :, None, None] * self.spatial_kernels()[:, None]

    def forward(self, x):
        x, squeeze_t = as_5d(x)
        assert x.shape[1] == self.in_channels, f'expected {self.in_channels} channels, got {x.shape[1]}'
        weight = self.kernels()[:, None] * self.channel_mix[:, :, None, None, None]  # (N, C, Kt, K, K)
        r = self.spatial_size // 2
        rt = self.temporal_size // 2 if self.temporal_padding == 'same' else 0
        x = F.pad(x, (r, r, r, r, rt, rt), mode='replicate')
        resp = F.conv3d(x, weight)
        if self.gain_mode == 'contrast':  # statistics in fp32 (r^2 underflows in fp16 for quiet neurons)
            ms = resp.float().pow(2).mean(dim=(2, 3, 4), keepdim=True)
            resp = resp.float() * torch.rsqrt(ms + self.contrast_eps ** 2)
        resp = resp * self.gain[None, :, None, None, None]  # fp32 parameter: promotes to fp32 under AMP, as before
        out = torch.cat([F.gelu(resp), F.gelu(-resp)], 1) if self.on_off else resp
        return out.squeeze(2) if squeeze_t else out

    def extra_repr(self):
        return (f'in_channels={self.in_channels}, num_filters={self.num_filters}, '
                f'spatial_size={self.spatial_size}, temporal_size={self.temporal_size}, '
                f'temporal_mode={self.temporal_mode}, temporal_padding={self.temporal_padding}, on_off={self.on_off}, '
                f'gain_mode={self.gain_mode}')
